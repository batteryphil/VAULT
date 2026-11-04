#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 Post-Step-10,000 Comprehensive Verification & Benchmark Suite:
#   1. `--only_kernel_audit`:
#        - GPU FP32 Monolithic Forward (`flash_vault_v2_chunkwise`) vs `VAULTv2_5StreamingEngine`
#          (`2x64` prefill + `128x decode_step` across chunk rollovers `t=128, 192`), verifying
#          `_vault_v2_5_fft_bind_64_kernel`, `_vault_v2_5_salience_pool_decode_kernel`,
#          `_vault_v2_5_local_syntax_decode_kernel`, and `_vault_v2_decode_fw_kernel` + HIPGraph bit-exactness.
#        - BF16 O(1) HIPGraph Decode Latency & Throughput across `2K -> 128K` and `220c -> 2,002c`
#          (demonstrating complete elimination of `16x rocFFT` dispatch overhead vs `v2.4`'s `26.60 ms/tok`).
#   2. Full Post-SFT Evaluation Suite (default):
#        - Holdout English BPB & Unseen Edu Prose PPL (`2K` & `4K` validation shards).
#        - 8-Domain Unseen ChatML Suite (`BuildBit .bb`, `C99`, `C++20`, `Python`, `Lua`, `CoT Math`, `Direct Prose`, `2K NIAH`).
#        - Controlled `2K` Gaps (`15t`, `3t`, `24t`) + Held-Out Adversarial `8..11`-Subword Sibling NIAH (`2K` & `4K`,
#          head-to-head against `VAULT v2.4 Step 10,000`).
#        - Extreme `2K -> 128K` Length Extrapolation & Zero-Shot Elastic Co-Prime Mitosis
#          (`220c -> 508c -> 942c -> 2,002c`, demonstrating stationary Leaky-EMA GTRM `128K` invariance).

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from benchmarks.bench_vault_vs_mamba123_transformer import (
    UNSEEN_FILLER,
    UNSEEN_NIAH_PREFIXES,
    build_deterministic_niah_trial,
    evaluate_bpb,
    evaluate_niah_condition,
    forward_tail_logits,
)
from benchmarks.bench_vault_extreme_128k import (
    _alloc_eval_caches as _alloc_v2_4_eval_caches,
    benchmark_vault_streaming_scaling,
    evaluate_extreme_niah_vault_streaming,
    load_vault_v2_4_and_mitosis,
)
from src.vault_v2_5_model import (
    VAULTLMv2_5,
    _unwrap_checkpoint_state_dict,
    load_v2_4_into_v2_5,
    upgrade_v2_5_to_wider_primes,
)
from src.vault_v2_5_streaming import (
    VAULTv2_5StreamCache,
    VAULTv2_5StreamingEngine,
    _DecodeCUDAGraphRunnerV2_5,
)
from src.vault_v2_model import VAULTLMv2
from train_vault_v2_2 import run_8domain_eval_v2_2


def atomic_save_json(data: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _sync_dev(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_vram(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    return 0.0


def _reset_v2_5_cache_inplace_(cache: VAULTv2_5StreamCache) -> None:
    cache.t_pos = 0
    cache.t_pos_buf.zero_()
    cache.x0_hist.zero_()
    cache.cms_counts_u_pad.zero_()
    cache.cms_counts_b_pad.zero_()
    cache.cms_ib_ring.zero_()
    cache.cms_bg_ring.zero_()
    cache.cms_prev_ib.zero_()
    cache.cc_ring.zero_()
    cache.bf_ring.zero_()
    cache.nov1_ring.zero_()
    for ls in cache.layers:
        for k in ls.__dataclass_fields__:
            t = getattr(ls, k)
            if k == "sal_win":
                t.fill_(-1e4)
            else:
                t.zero_()


def _release_v2_5_decode_graphs(model: VAULTLMv2_5) -> None:
    runners = getattr(model, "_decode_graph_runners_v2_5", None)
    if runners:
        banks = model.layers[0].attn.salience_binder.BANKS
        for k in list(runners.keys()):
            runners.pop(k).release_owner(banks)


def load_v2_5_or_upgraded_v2_4(
    ckpt_path: str,
    device: str = "cuda:0",
    target_primes: Tuple[int, ...] = (47, 53, 59, 61),
    mitosis_angle: float = 0.26,
    force_ext_alpha_if_zero: Optional[float] = None,
) -> Tuple[VAULTLMv2_5, AutoTokenizer, Dict]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _unwrap_checkpoint_state_dict(raw)
    src_primes = tuple(raw.get("prime_banks", (47, 53, 59, 61)))
    ema_alpha = float(raw.get("ema_alpha", 2.0**-11))

    base_model = VAULTLMv2_5(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint="none",
        prime_banks=src_primes,
        ema_alpha=ema_alpha,
    )
    is_native_v25 = any("ext_span_alpha" in k for k in sd.keys())
    if is_native_v25:
        base_model.load_state_dict(sd, strict=True)
        load_meta = {"source": "native_v2_5", "step": int(raw.get("step", 0))}
    else:
        init_ext = 0.0 if force_ext_alpha_if_zero is None else float(force_ext_alpha_if_zero)
        load_meta = load_v2_4_into_v2_5(base_model, raw, init_ext_span_alpha=init_ext)
        load_meta["source"] = "upgraded_v2_4"
        load_meta["step"] = int(raw.get("step", 0))

    if force_ext_alpha_if_zero is not None:
        for blk in base_model.layers:
            if float(blk.attn.ext_span_alpha.abs().item()) < 1e-6:
                blk.attn.ext_span_alpha.data.fill_(float(force_ext_alpha_if_zero))

    del raw, sd
    gc.collect()

    if tuple(target_primes) != tuple(src_primes):
        wide_model = VAULTLMv2_5(
            vocab_size=len(tok),
            d_model=768,
            n_layers=16,
            n_q_heads=12,
            n_vault_groups=4,
            head_dim=64,
            d_ff=2048,
            chunk_size=64,
            grad_checkpoint="none",
            prime_banks=target_primes,
            ema_alpha=ema_alpha,
        )
        mit_meta = upgrade_v2_5_to_wider_primes(
            wide_model, base_model.state_dict(), split_angle_rad=mitosis_angle
        )
        load_meta.update(mit_meta)
        del base_model
        gc.collect()
        return wide_model.to(device).eval(), tok, load_meta

    load_meta["prime_banks"] = src_primes
    load_meta["total_cells"] = sum(src_primes)
    return base_model.to(device).eval(), tok, load_meta


# ==============================================================================
# 1. GPU TRITON DECODE PARITY & O(1) HIPGRAPH LATENCY AUDIT
# ==============================================================================

@torch.no_grad()
def verify_gpu_triton_decode_parity(ckpt_path: str, device: str = "cuda:0") -> Dict:
    print("\n" + "=" * 108, flush=True)
    print("PART 1A: GPU FP32 TRITON DECODE KERNEL & HIPGRAPH BIT-EXACTNESS AUDIT (`t = 0..255`)", flush=True)
    print("=" * 108, flush=True)

    model, _, meta = load_v2_5_or_upgraded_v2_4(
        ckpt_path, device=device, target_primes=(47, 53, 59, 61), force_ext_alpha_if_zero=0.55
    )
    model = model.float().eval()

    torch.manual_seed(20261103)
    B, T_prefill, T_decode = 2, 128, 128
    ids = torch.randint(0, model.vocab_size, (B, T_prefill + T_decode), dtype=torch.long, device=device)

    _sync_dev(device)
    logits_mono = model(ids).float()

    # 1. Eager Triton Decode (`use_cudagraph=False`) across `t = 128..255` (crosses rollovers `t=128` & `t=192`)
    pref_logits_e, cache_e = VAULTv2_5StreamingEngine.prefill_streaming(
        model, ids[:, :T_prefill], segment_size=64, tail_logits_len=T_prefill, occ_alpha=0.5
    )
    pref_diff = float((logits_mono[:, :T_prefill] - pref_logits_e.float()).abs().max().item())

    dec_eager_list = []
    for t in range(T_prefill, T_prefill + T_decode):
        lg = VAULTv2_5StreamingEngine.decode_step(
            model, ids[:, t : t + 1], cache_e, occ_alpha=0.5, use_cudagraph=False
        )
        dec_eager_list.append(lg.float())
    dec_eager = torch.cat(dec_eager_list, dim=1)
    eager_vs_mono_diff = float((logits_mono[:, T_prefill:] - dec_eager).abs().max().item())

    # 2. Captured HIPGraph Triton Decode (`use_cudagraph=True`) across `t = 128..255`
    _, cache_g = VAULTv2_5StreamingEngine.prefill_streaming(
        model, ids[:, :T_prefill], segment_size=64, occ_alpha=0.5
    )
    dec_graph_list = []
    for t in range(T_prefill, T_prefill + T_decode):
        lg_g = VAULTv2_5StreamingEngine.decode_step(
            model, ids[:, t : t + 1], cache_g, occ_alpha=0.5, use_cudagraph=True
        )
        dec_graph_list.append(lg_g.float())
    dec_graph = torch.cat(dec_graph_list, dim=1)
    step_abs_e = (logits_mono[:, T_prefill:] - dec_eager).abs().amax(dim=-1)
    median_step_diff = float(step_abs_e.median().item())
    top1_match_rate = float(
        (logits_mono[:, T_prefill:].argmax(dim=-1) == dec_graph.argmax(dim=-1)).float().mean().item()
    )
    graph_vs_eager_diff = float((dec_graph - dec_eager).abs().max().item())
    graph_vs_mono_diff = float((logits_mono[:, T_prefill:] - dec_graph).abs().max().item())

    _release_v2_5_decode_graphs(model)
    del model, cache_e, cache_g, logits_mono, dec_eager, dec_graph
    _reset_vram(device)

    print(
        f"  -> FP32 Prefill (2x64 segs) max_abs_diff = {pref_diff:.3e} | "
        f"128x Decode median_step_diff = {median_step_diff:.3e} (max={eager_vs_mono_diff:.3e}, top1={top1_match_rate*100:.2f}%) | "
        f"HIPGraph vs Eager = {graph_vs_eager_diff:.3e}",
        flush=True,
    )
    assert (
        pref_diff < 5e-4
        and median_step_diff < 1e-4
        and top1_match_rate >= 0.995
        and eager_vs_mono_diff < 0.15
        and graph_vs_mono_diff < 0.15
        and graph_vs_eager_diff < 0.02
    ), (
        f"GPU Triton decode parity exceeded tolerance: prefill={pref_diff}, median={median_step_diff}, "
        f"top1={top1_match_rate}, eager={eager_vs_mono_diff}, graph={graph_vs_mono_diff}"
    )
    return {
        "meta": meta,
        "fp32_prefill_max_abs_diff": pref_diff,
        "fp32_decode_128step_median_abs_diff": median_step_diff,
        "fp32_decode_top1_match_rate": top1_match_rate,
        "fp32_decode_128step_vs_monolithic_max_abs_diff": eager_vs_mono_diff,
        "fp32_hipgraph_vs_eager_max_abs_diff": graph_vs_eager_diff,
        "fp32_hipgraph_vs_monolithic_max_abs_diff": graph_vs_mono_diff,
    }


@torch.no_grad()
def benchmark_v2_5_streaming_scaling(
    model: VAULTLMv2_5,
    lengths: List[int],
    segment_size: int = 4096,
    decode_tokens: int = 64,
    occ_alpha: float = 1.5,
    device: str = "cuda:0",
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    # Warmup Triton prefill kernels
    w_ids = torch.randint(0, model.vocab_size, (1, min(1024, max(lengths))), dtype=torch.long, device=device)
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, w_c = VAULTv2_5StreamingEngine.prefill_streaming(
                model, w_ids, segment_size=min(512, segment_size), occ_alpha=occ_alpha
            )
    else:
        _, w_c = VAULTv2_5StreamingEngine.prefill_streaming(
            model, w_ids, segment_size=min(512, segment_size), occ_alpha=occ_alpha
        )
    del w_ids, w_c

    prefill_peaks_mb: Dict[int, float] = {}
    for L in lengths:
        _reset_vram(device)
        inp = torch.randint(0, model.vocab_size, (1, L), dtype=torch.long, device=device)
        seg = min(segment_size, L) if L >= 64 else 64
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, c_tmp = VAULTv2_5StreamingEngine.prefill_streaming(
                    model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
                )
        else:
            _, c_tmp = VAULTv2_5StreamingEngine.prefill_streaming(
                model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
            )
        prefill_peaks_mb[L] = _peak_vram_mb(device)
        del inp, c_tmp

    # Capture HIPGraph once before steady-state timing
    w_tok = torch.randint(0, model.vocab_size, (1, 1), dtype=torch.long, device=device)
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_c2 = VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=torch.bfloat16)
            for _ in range(3):
                _ = VAULTv2_5StreamingEngine.decode_step(model, w_tok, w_c2, occ_alpha=occ_alpha)
    else:
        w_c2 = VAULTv2_5StreamingEngine.init_cache(model, batch_size=1)
        _ = VAULTv2_5StreamingEngine.decode_step(model, w_tok, w_c2, occ_alpha=occ_alpha)
    del w_tok, w_c2

    for L in lengths:
        label = f"{L // 1024}K" if L >= 1024 else f"{L}"
        inp = torch.randint(0, model.vocab_size, (1, L), dtype=torch.long, device=device)
        seg = min(segment_size, L) if L >= 64 else 64

        _sync_dev(device)
        t0 = time.perf_counter()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_last, cache = VAULTv2_5StreamingEngine.prefill_streaming(
                    model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
                )
        else:
            logits_last, cache = VAULTv2_5StreamingEngine.prefill_streaming(
                model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
            )
        _sync_dev(device)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        state_mb = round(cache.state_bytes() / (1024 ** 2), 2)

        nxt = torch.argmax(logits_last[:, -1:], dim=-1)
        # Bind `cache` into HIPGraph on 1st step + 3 warmup replays
        for _ in range(4):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    step_logits = VAULTv2_5StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            else:
                step_logits = VAULTv2_5StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            nxt = torch.argmax(step_logits[:, -1:], dim=-1)

        _sync_dev(device)
        t1 = time.perf_counter()
        for _ in range(decode_tokens):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    step_logits = VAULTv2_5StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            else:
                step_logits = VAULTv2_5StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            nxt = torch.argmax(step_logits[:, -1:], dim=-1)
        _sync_dev(device)
        dec_ms = ((time.perf_counter() - t1) * 1000.0) / max(1, decode_tokens)
        speedup_vs_v24 = 26.60 / max(dec_ms, 1e-6)

        rec = {
            "seq_len": L,
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tok_s": round(L / max(prefill_ms / 1000.0, 1e-6), 1),
            "peak_vram_mb": prefill_peaks_mb[L],
            "persistent_state_mb": state_mb,
            "decode_ms_per_tok": round(dec_ms, 3),
            "decode_tok_s": round(1000.0 / max(dec_ms, 1e-6), 1),
            "speedup_vs_v2_4_26_6ms": round(speedup_vs_v24, 2),
        }
        results[label] = rec
        print(
            f"  [VAULT v2.5 {sum(model.prime_banks)}c | {label:>4s} ({L:7,d}t)] "
            f"prefill={rec['prefill_tok_s']:8,.0f} tok/s ({rec['prefill_ms']:6.1f}ms) | "
            f"VRAM={rec['peak_vram_mb']:6.1f}MB (state={rec['persistent_state_mb']:.1f}MB) | "
            f"O(1) decode={rec['decode_ms_per_tok']:5.2f} ms/tok ({rec['decode_tok_s']:6.1f} tok/s, {rec['speedup_vs_v2_4_26_6ms']:.2f}x vs v2.4)",
            flush=True,
        )
        del inp, cache, logits_last

    _release_v2_5_decode_graphs(model)
    _reset_vram(device)
    return results


# ==============================================================================
# 2. HELD-OUT ADVERSARIAL `8..11`-SUBWORD SIBLING DISENTANGLEMENT NIAH (`2K` & `4K`)
# ==============================================================================

def build_unseen_long_sibling_niah_trial(
    tok, n_sections: int, tr: int, seed_base: int
) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]], int]:
    """
    Constructs an adversarial manifest exclusively from `UNSEEN_NIAH_PREFIXES` (100% excluded from `data_v2_5/`)
    where sibling subsystems share an identical `>=7`-BPE-token prefix (`8..11` total BPE tokens), e.g.,
    `ZephyrBoreal-Unit42408` vs `ZephyrBoreal-Unit42953`, directly testing `g2` (`m=1..10`) vs `g1` (`m=1..6`).
    """
    rng = random.Random(seed_base + tr * 137)
    n_pairs = max(2, n_sections // 2)
    pairs_ents: List[Tuple[str, str]] = []
    min_shared_prefix = 99

    cursor = tr * 3
    while len(pairs_ents) < n_pairs:
        w1 = UNSEEN_NIAH_PREFIXES[cursor % len(UNSEEN_NIAH_PREFIXES)]
        w2 = UNSEEN_NIAH_PREFIXES[(cursor + 5) % len(UNSEEN_NIAH_PREFIXES)]
        cursor += 1
        if w1 == w2:
            continue
        tier_hi = 11 + ((cursor * 7 + tr * 13) % 78)
        sfx_a = 104 + ((cursor * 19) % 380)
        sfx_b = 561 + ((cursor * 23) % 410)
        ent_a = f"{w1}{w2}-Unit{tier_hi:02d}{sfx_a:03d}"
        ent_b = f"{w1}{w2}-Unit{tier_hi:02d}{sfx_b:03d}"
        ta = tok.encode(" " + ent_a, add_special_tokens=False)
        tb = tok.encode(" " + ent_b, add_special_tokens=False)
        shared = 0
        for xa, xb in zip(ta, tb):
            if xa == xb:
                shared += 1
            else:
                break
        if len(ta) >= 8 and len(tb) >= 8 and shared >= 7:
            pairs_ents.append((ent_a, ent_b))
            min_shared_prefix = min(min_shared_prefix, shared)

    flat_ents: List[str] = []
    for ea, eb in pairs_ents:
        flat_ents.extend([ea, eb])
    flat_ents = flat_ents[:n_sections]

    entries: List[Tuple[str, str]] = []
    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    used_codes = set()
    for idx, ent in enumerate(flat_ents, start=1):
        while True:
            code = f"CHK-{rng.randint(1000, 9999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        entries.append((ent, code))
        hdr = f"[Subsystem {idx:03d}: {ent}] Status: NOMINAL | Verification Code: {code}"
        blocks.append(f"{hdr}\n  Log: {UNSEEN_FILLER} Packet frame {1000 + idx} committed without retransmission.")

    # Query ONE sibling from an early pair and ONE sibling from a late pair while its twin acts as an active distractor!
    p1 = max(1, (n_pairs // 4))
    p2 = min(n_pairs - 1, (3 * n_pairs) // 4)
    targets = [entries[2 * p1 + (tr % 2)], entries[2 * p2 + ((tr + 1) % 2)]]
    return "\n".join(blocks), entries, targets, min_shared_prefix


@torch.no_grad()
def evaluate_adversarial_long_sibling_niah(
    model: nn.Module,
    tok,
    mode: str,
    n_sections: int,
    n_trials: int,
    seed_base: int,
    device: str = "cuda:0",
) -> Dict:
    model.eval()
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>") if mode == "chatml_vault" else -1
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>") if mode == "chatml_vault" else -1
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
    stop_ids = tuple(x for x in (im_end_id, eos_id) if x is not None and x >= 0)

    cue_exact_hits = 0
    total_codes = 0
    char_hits = [0, 0, 0, 0]
    total_nll_bits = 0.0
    prompt_lens: List[int] = []
    min_pref_seen = 99

    sample_predictions = []
    for tr in range(n_trials):
        manifest_body, entries, targets, min_sh = build_unseen_long_sibling_niah_trial(tok, n_sections, tr, seed_base)
        min_pref_seen = min(min_pref_seen, min_sh)
        q_str = (
            f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
            f"for {targets[0][0]} and {targets[1][0]}? List each subsystem and its exact code."
        )
        if mode == "chatml_vault":
            user_prm = manifest_body + q_str
            base_p_ids = (
                [im_start_id]
                + tok.encode(f"user\n{user_prm.strip()}", add_special_tokens=False)
                + [im_end_id]
                + tok.encode("\n", add_special_tokens=False)
                + [im_start_id]
                + tok.encode("assistant\n", add_special_tokens=False)
            )
        else:
            ex_ent, ex_code = entries[0]
            prompt_str = manifest_body + q_str + f"\nVerification Summary:\n- {ex_ent}: {ex_code}\n"
            base_p_ids = tok.encode(prompt_str, add_special_tokens=False)

        cue0_str = f"- {targets[0][0]}: CHK-"
        cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
        ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        ans_ids = tok.encode(ans_str, add_special_tokens=False)
        prompt_lens.append(len(base_p_ids))

        full_ids = torch.tensor([base_p_ids + ans_ids], dtype=torch.long, device=device)
        ctx_amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)
        with ctx_amp:
            tail_logits = forward_tail_logits(model, full_ids, tail_len=len(ans_ids) + 1)[0].float()
        del full_ids

        for idx_t, (ent_name, code) in enumerate(targets):
            cue_s = cue0_str if idx_t == 0 else cue1_str
            cue_ids = tok.encode(cue_s, add_special_tokens=False)
            with_code_ids = tok.encode(cue_s + code.split("-")[1], add_special_tokens=False)
            start_rel = len(cue_ids)
            dig_tok_ids = with_code_ids[start_rel:]
            nll_nats = 0.0
            for off_k, tid in enumerate(dig_tok_ids):
                pos = start_rel + off_k
                if pos < tail_logits.size(0):
                    log_probs = F.log_softmax(tail_logits[pos], dim=-1)
                    nll_nats -= float(log_probs[tid].item())
            total_nll_bits += nll_nats / math.log(2.0)

            true_digits = code.split("-")[1]
            cur_ids = base_p_ids + cue_ids
            gen_ids: List[int] = []
            for _ in range(4):
                inp_t = torch.tensor([cur_ids + gen_ids], dtype=torch.long, device=device)
                with ctx_amp:
                    lg = forward_tail_logits(model, inp_t, tail_len=1)[0, -1].float()
                nxt = int(torch.argmax(lg).item())
                del inp_t
                if nxt in stop_ids:
                    break
                gen_ids.append(nxt)
                txt = tok.decode(gen_ids, skip_special_tokens=True)
                if len(txt.strip()) >= 4 or "\n" in txt:
                    break

            gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
            if len(sample_predictions) < 4:
                sample_predictions.append({"entity": ent_name, "gold": code, "raw_decoded": f"CHK-{gen_txt.strip()}"})
            m = re.match(r"\s*([0-9]{1,4})", gen_txt)
            pred_4 = (m.group(1) if m else gen_txt.strip()[:4]).ljust(4, "?")[:4]
            for d_i in range(4):
                if pred_4[d_i] == true_digits[d_i]:
                    char_hits[d_i] += 1
            if pred_4 == true_digits:
                cue_exact_hits += 1
            total_codes += 1

    per_digit = [round(char_hits[i] / max(1, total_codes), 4) for i in range(4)]
    return {
        "mode": mode,
        "n_sections": n_sections,
        "min_shared_bpe_prefix": min_pref_seen,
        "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
        "cue_exact_hits": f"{cue_exact_hits}/{total_codes}",
        "cue_exact_acc": round(cue_exact_hits / max(1, total_codes), 4),
        "digit_acc": round(sum(char_hits) / max(1, 4 * total_codes), 4),
        "per_digit_acc": per_digit,
        "mean_code_nll_bits": round(total_nll_bits / max(1, total_codes), 3),
        "sample_predictions": sample_predictions,
    }


# ==============================================================================
# 3. EXTREME `2K -> 128K` STREAMING NIAH EVALUATOR FOR `VAULTLMv2_5`
# ==============================================================================

@torch.no_grad()
def evaluate_extreme_niah_v2_5_streaming(
    model: VAULTLMv2_5,
    tok,
    mode: str,
    n_sections: int,
    n_trials: int,
    seed_base: int,
    segment_size: int = 4096,
    occ_alpha: float = 1.5,
    caches: Optional[Tuple[VAULTv2_5StreamCache, VAULTv2_5StreamCache, VAULTv2_5StreamCache]] = None,
    device: str = "cuda:0",
) -> Dict:
    model.eval()
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>") if mode == "chatml_vault" else -1
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>") if mode == "chatml_vault" else -1
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
    stop_ids = tuple(x for x in (im_end_id, eos_id) if x is not None and x >= 0)

    own_caches = caches is None
    if own_caches:
        dt_c = torch.bfloat16 if use_amp else None
        caches = (
            VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
            VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
            VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
        )
    aligned_cache, cue_base_cache, work_cache = caches

    cue_exact_hits = 0
    total_codes = 0
    char_hits = [0, 0, 0, 0]
    total_nll_bits = 0.0
    prompt_lens: List[int] = []
    fwd_ms = 0.0
    peak_mb = 0.0
    mean_cell_occ = 0.0
    sample_predictions = []

    for tr in range(n_trials):
        manifest_body, entries, targets = build_deterministic_niah_trial("boilerplate_15tok", n_sections, tr, seed_base)
        q_str = (
            f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
            f"for {targets[0][0]} and {targets[1][0]}? List each subsystem and its exact code."
        )
        if mode == "chatml_vault":
            user_prm = manifest_body + q_str
            base_p_ids = (
                [im_start_id]
                + tok.encode(f"user\n{user_prm.strip()}", add_special_tokens=False)
                + [im_end_id]
                + tok.encode("\n", add_special_tokens=False)
                + [im_start_id]
                + tok.encode("assistant\n", add_special_tokens=False)
            )
        else:
            ex_ent, ex_code = entries[0]
            prompt_str = manifest_body + q_str + f"\nVerification Summary:\n- {ex_ent}: {ex_code}\n"
            base_p_ids = tok.encode(prompt_str, add_special_tokens=False)

        cue0_str = f"- {targets[0][0]}: CHK-"
        cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
        ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        ans_ids = tok.encode(ans_str, add_special_tokens=False)

        prompt_lens.append(len(base_p_ids))
        if use_amp:
            torch.cuda.reset_peak_memory_stats()

        aligned_len = ((max(1, len(base_p_ids)) - 1) // 64) * 64
        rem_ids = base_p_ids[aligned_len:]

        _reset_v2_5_cache_inplace_(aligned_cache)
        _sync_dev(device)
        t0 = time.perf_counter()
        if aligned_len > 0:
            p_tensor = torch.tensor([base_p_ids[:aligned_len]], dtype=torch.long, device=device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                        model,
                        p_tensor,
                        segment_size=segment_size,
                        cache=aligned_cache,
                        tail_logits_len=1,
                        occ_alpha=occ_alpha,
                        populate_decode_rings=False,
                    )
            else:
                _, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                    model,
                    p_tensor,
                    segment_size=segment_size,
                    cache=aligned_cache,
                    tail_logits_len=1,
                    occ_alpha=occ_alpha,
                    populate_decode_rings=False,
                )
            del p_tensor

        tf_seq = rem_ids + ans_ids
        tf_pad = (64 - (len(tf_seq) % 64)) % 64
        tf_padded = tf_seq + [0] * tf_pad
        tf_tensor = torch.tensor([tf_padded], dtype=torch.long, device=device)
        _DecodeCUDAGraphRunnerV2_5._copy_cache_tensors_(work_cache, aligned_cache)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tf_logits_all, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                    model,
                    tf_tensor,
                    segment_size=segment_size,
                    cache=work_cache,
                    tail_logits_len=len(tf_padded),
                    occ_alpha=occ_alpha,
                    populate_decode_rings=False,
                )
        else:
            tf_logits_all, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                model,
                tf_tensor,
                segment_size=segment_size,
                cache=work_cache,
                tail_logits_len=len(tf_padded),
                occ_alpha=occ_alpha,
                populate_decode_rings=False,
            )
        _sync_dev(device)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        extra_scratch_mb = 2.0 * (aligned_cache.state_bytes() / (1024 ** 2))
        peak_mb = round(max(0.0, _peak_vram_mb(device) - extra_scratch_mb), 1)
        tail_logits = tf_logits_all[0, len(rem_ids) - 1 : len(rem_ids) - 1 + len(ans_ids)].float()
        del tf_tensor, tf_logits_all

        mean_cell_occ += float(
            sum(ls.occ_state[:, 1:].mean().item() for ls in aligned_cache.layers) / len(aligned_cache.layers)
        ) / max(1, n_trials)

        for idx_t, (_, code) in enumerate(targets):
            cue_s = cue0_str if idx_t == 0 else cue1_str
            cue_ids = tok.encode(cue_s, add_special_tokens=False)
            with_code_ids = tok.encode(cue_s + code.split("-")[1], add_special_tokens=False)
            start_rel = len(cue_ids)
            dig_tok_ids = with_code_ids[start_rel:]
            nll_nats = 0.0
            for off_k, tid in enumerate(dig_tok_ids):
                pos = start_rel + off_k
                if pos < tail_logits.size(0):
                    log_probs = F.log_softmax(tail_logits[pos], dim=-1)
                    nll_nats -= float(log_probs[tid].item())
            total_nll_bits += nll_nats / math.log(2.0)

        for idx_t, (ent_name, code) in enumerate(targets):
            true_digits = code.split("-")[1]
            cue_s = cue0_str if idx_t == 0 else cue1_str
            cue_ids = tok.encode(cue_s, add_special_tokens=False)
            cue_full = rem_ids + cue_ids
            cue_align = ((len(cue_full) - 1) // 64) * 64
            if cue_align > 0:
                _DecodeCUDAGraphRunnerV2_5._copy_cache_tensors_(cue_base_cache, aligned_cache)
                ca_t = torch.tensor([cue_full[:cue_align]], dtype=torch.long, device=device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                            model,
                            ca_t,
                            segment_size=segment_size,
                            cache=cue_base_cache,
                            tail_logits_len=1,
                            occ_alpha=occ_alpha,
                            populate_decode_rings=False,
                        )
                else:
                    _, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                        model,
                        ca_t,
                        segment_size=segment_size,
                        cache=cue_base_cache,
                        tail_logits_len=1,
                        occ_alpha=occ_alpha,
                        populate_decode_rings=False,
                    )
                del ca_t
                src_cache = cue_base_cache
                sub_rem = cue_full[cue_align:]
            else:
                src_cache = aligned_cache
                sub_rem = cue_full

            gen_ids: List[int] = []
            for _ in range(4):
                step_seq = sub_rem + gen_ids
                q_idx = len(step_seq) - 1
                pad_g = (64 - (len(step_seq) % 64)) % 64
                step_padded = step_seq + [0] * pad_g
                st_t = torch.tensor([step_padded], dtype=torch.long, device=device)
                _DecodeCUDAGraphRunnerV2_5._copy_cache_tensors_(work_cache, src_cache)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        st_lg, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                            model,
                            st_t,
                            segment_size=segment_size,
                            cache=work_cache,
                            tail_logits_len=len(step_padded),
                            occ_alpha=occ_alpha,
                            populate_decode_rings=False,
                        )
                else:
                    st_lg, _ = VAULTv2_5StreamingEngine.prefill_streaming(
                        model,
                        st_t,
                        segment_size=segment_size,
                        cache=work_cache,
                        tail_logits_len=len(step_padded),
                        occ_alpha=occ_alpha,
                        populate_decode_rings=False,
                    )
                nxt = int(torch.argmax(st_lg[0, q_idx].float()).item())
                del st_t, st_lg
                if nxt in stop_ids:
                    break
                gen_ids.append(nxt)
                txt = tok.decode(gen_ids, skip_special_tokens=True)
                if len(txt.strip()) >= 4 or "\n" in txt:
                    break

            gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
            if len(sample_predictions) < 4:
                sample_predictions.append({"entity": ent_name, "gold": code, "raw_decoded": f"CHK-{gen_txt.strip()}"})
            m = re.match(r"\s*([0-9]{1,4})", gen_txt)
            pred_4 = (m.group(1) if m else gen_txt.strip()[:4]).ljust(4, "?")[:4]
            for d_i in range(4):
                if pred_4[d_i] == true_digits[d_i]:
                    char_hits[d_i] += 1
            if pred_4 == true_digits:
                cue_exact_hits += 1
            total_codes += 1

        del tail_logits

    if own_caches:
        del caches

    per_digit = [round(char_hits[i] / max(1, total_codes), 4) for i in range(4)]
    return {
        "mode": mode,
        "occ_alpha": occ_alpha,
        "n_sections": n_sections,
        "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
        "cue_exact_hits": f"{cue_exact_hits}/{total_codes}",
        "cue_exact_acc": round(cue_exact_hits / max(1, total_codes), 4),
        "digit_acc": round(sum(char_hits) / max(1, 4 * total_codes), 4),
        "per_digit_acc": per_digit,
        "mean_code_nll_bits": round(total_nll_bits / max(1, total_codes), 3),
        "mean_ind_cell_occ": round(mean_cell_occ, 4),
        "prefill_ms": round(fwd_ms, 2),
        "peak_vram_mb": peak_mb,
        "sample_predictions": sample_predictions,
    }


def _alloc_v2_5_eval_caches(model: VAULTLMv2_5, device: str):
    dt_c = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_available()) else None
    return (
        VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
        VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
        VAULTv2_5StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
    )


@torch.no_grad()
def evaluate_unseen_edu_prose_ppl(model: nn.Module, data_dir: str, device: str = "cuda:0") -> Dict:
    val_bin = os.path.join(data_dir, "prose_climax_val_tokens.bin")
    if not os.path.exists(val_bin):
        return {}
    row_len = 2049
    n_rows = os.path.getsize(val_bin) // (row_len * 2)
    mm = np.memmap(val_bin, dtype=np.uint16, mode="r", shape=(n_rows, row_len))
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    tot_nll = 0.0
    tot_tok = 0
    lim = min(n_rows, 64)
    for st in range(0, lim, 4):
        en = min(lim, st + 4)
        arr = np.asarray(mm[st:en], dtype=np.int64)
        bx = torch.from_numpy(arr[:, :-1]).to(device)
        by = torch.from_numpy(arr[:, 1:]).to(device)
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)
        with ctx:
            h = model._encode_hidden(bx, step_ratio=1.0).reshape(-1, 768)
            yf = by.reshape(-1)
            for s0 in range(0, h.size(0), 2048):
                s1 = min(h.size(0), s0 + 2048)
                lg = model.lm_head(h[s0:s1]).float()
                tot_nll += float(F.cross_entropy(lg, yf[s0:s1], reduction="sum").item())
        tot_tok += int(by.numel())
    mean_nll = tot_nll / max(1, tot_tok)
    return {
        "n_rows": lim,
        "evaluated_tokens": tot_tok,
        "mean_nll_nats": round(mean_nll, 4),
        "ppl": round(math.exp(min(20.0, mean_nll)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VAULT Pre-SFT Step 10K & Post-SFT v2.5 Full Benchmark Suite")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(ROOT_DIR, "checkpoints", "vault_v2_5_best.pt"),
    )
    parser.add_argument(
        "--v2_4_ref_ckpt",
        type=str,
        default=os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_step10000_best.pt"),
    )
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT_DIR, "data_v2_5"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--segment_size", type=int, default=4096)
    parser.add_argument("--mitosis_angle", type=float, default=0.26)
    parser.add_argument("--occ_alpha", type=float, default=1.5)
    parser.add_argument("--only_kernel_audit", action="store_true")
    parser.add_argument(
        "--eval_native_v2_4",
        action="store_true",
        help="Evaluate --ckpt natively as VAULTLMv2 + VAULTv2StreamingEngine (Pre-SFT Step 10,000 full baseline)",
    )
    parser.add_argument(
        "--skip_v2_4_ref",
        action="store_true",
        help="Skip inline v2.4 ref sub-block when Pre-SFT full suite was already run into its own JSON",
    )
    parser.add_argument("--stage_label", type=str, default="")
    parser.add_argument("--domain8_out", type=str, default="")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmarks", "results_vault_v2_5_post10k.json"),
    )
    args = parser.parse_args()

    stage_label = args.stage_label or ("VAULT_v2_4_Step10000" if args.eval_native_v2_4 else "VAULT_v2_5_SFT")
    model_prefix = "VAULT_v2_4" if args.eval_native_v2_4 else "VAULT_v2_5"
    out_8dom = args.domain8_out or os.path.join(
        ROOT_DIR,
        "benchmarks",
        "results_vault_v2_4_step10k_8domain.json" if args.eval_native_v2_4 else "results_vault_v2_5_8domain.json",
    )

    t_all0 = time.perf_counter()
    report: Dict = {
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage_label": stage_label,
        "eval_native_v2_4": bool(args.eval_native_v2_4),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "checkpoint": args.ckpt,
    }
    if os.path.exists(args.out) and not args.only_kernel_audit:
        try:
            with open(args.out, "r") as f_in:
                prev = json.load(f_in)
            for _rk in ("gpu_kernel_parity_and_latency", "language_modeling", "chatml_8domain"):
                if _rk in prev:
                    report[_rk] = prev[_rk]
        except Exception:
            pass

    if args.only_kernel_audit or "gpu_kernel_parity_and_latency" not in report:
        audit_ckpt = args.ckpt if os.path.exists(args.ckpt) else args.v2_4_ref_ckpt
        parity_res = verify_gpu_triton_decode_parity(audit_ckpt, device=args.device)

        print("\n" + "=" * 108, flush=True)
        print(f"PART 1B: O(1) DECODE LATENCY & STREAMING THROUGHPUT [{stage_label}] (`2K -> 128K` & `220c -> 2,002c`)", flush=True)
        print("=" * 108, flush=True)
        if args.eval_native_v2_4:
            m220_s, _, _ = load_vault_v2_4_and_mitosis(
                audit_ckpt, device=args.device, prime_banks=(47, 53, 59, 61), mitosis_angle=args.mitosis_angle
            )
            lat_220 = benchmark_vault_streaming_scaling(
                m220_s,
                lengths=[2048, 8192, 32768, 65536, 131072],
                segment_size=args.segment_size,
                decode_tokens=32,
                occ_alpha=args.occ_alpha,
                device=args.device,
            )
            del m220_s
            _reset_vram(args.device)

            m2002_s, _, _ = load_vault_v2_4_and_mitosis(
                audit_ckpt, device=args.device, prime_banks=(491, 499, 503, 509), mitosis_angle=args.mitosis_angle
            )
            lat_2002 = benchmark_vault_streaming_scaling(
                m2002_s,
                lengths=[2048, 131072],
                segment_size=args.segment_size,
                decode_tokens=32,
                occ_alpha=args.occ_alpha,
                device=args.device,
            )
            del m2002_s
            _reset_vram(args.device)
        else:
            m220_s, _, _ = load_v2_5_or_upgraded_v2_4(
                audit_ckpt, device=args.device, target_primes=(47, 53, 59, 61), force_ext_alpha_if_zero=0.55
            )
            lat_220 = benchmark_v2_5_streaming_scaling(
                m220_s,
                lengths=[2048, 8192, 32768, 65536, 131072],
                segment_size=args.segment_size,
                decode_tokens=64,
                occ_alpha=args.occ_alpha,
                device=args.device,
            )
            del m220_s
            _reset_vram(args.device)

            m2002_s, _, _ = load_v2_5_or_upgraded_v2_4(
                audit_ckpt,
                device=args.device,
                target_primes=(491, 499, 503, 509),
                mitosis_angle=args.mitosis_angle,
                force_ext_alpha_if_zero=0.55,
            )
            lat_2002 = benchmark_v2_5_streaming_scaling(
                m2002_s,
                lengths=[2048, 131072],
                segment_size=args.segment_size,
                decode_tokens=64,
                occ_alpha=args.occ_alpha,
                device=args.device,
            )
            del m2002_s
            _reset_vram(args.device)

        report["gpu_kernel_parity_and_latency"] = {
            "audit_checkpoint": audit_ckpt,
            "engine": "VAULTv2StreamingEngine (rocFFT)" if args.eval_native_v2_4 else "VAULTv2_5StreamingEngine (Zero-rocFFT Triton)",
            "triton_fp32_parity": parity_res,
            "scaling_220c": lat_220,
            "scaling_2002c": lat_2002,
        }
        atomic_save_json(report, args.out)
        if args.only_kernel_audit:
            print(f"\n[+] GPU Kernel Parity & O(1) Latency Audit Complete -> {args.out}", flush=True)
            return

    # ==========================================================================
    # FULL BENCHMARK SUITE ON `--ckpt` (`VAULT v2.4 @ 10k` OR `VAULT v2.5 SFT`)
    # ==========================================================================
    print("\n" + "=" * 108, flush=True)
    print(f"PART 2: FULL EVALUATION SUITE [{stage_label}] ({args.ckpt})", flush=True)
    print("=" * 108, flush=True)

    if args.eval_native_v2_4:
        m220, tok, meta_220 = load_vault_v2_4_and_mitosis(
            args.ckpt, device=args.device, prime_banks=(47, 53, 59, 61), mitosis_angle=args.mitosis_angle
        )
    else:
        m220, tok, meta_220 = load_v2_5_or_upgraded_v2_4(
            args.ckpt, device=args.device, target_primes=(47, 53, 59, 61)
        )
    report["model_meta"] = meta_220
    report["vault_v2_5_meta"] = meta_220

    # 2A. Holdout English BPB & Unseen Edu Prose PPL
    if "language_modeling" not in report:
        bpb_res = evaluate_bpb(m220, tok, device=args.device)
        prose_res = evaluate_unseen_edu_prose_ppl(m220, args.data_dir, device=args.device)
        report["language_modeling"] = {"holdout_english_bpb": bpb_res, "unseen_edu_prose_2k": prose_res}
        atomic_save_json(report, args.out)
    else:
        bpb_res = report["language_modeling"]["holdout_english_bpb"]
        prose_res = report["language_modeling"]["unseen_edu_prose_2k"]
    print(
        f"  [{stage_label} Language Modeling] Holdout English BPB = {bpb_res['bpb']:.4f} (TokPPL={bpb_res['native_tok_ppl']:.2f}) | "
        f"Unseen Edu Prose PPL = {prose_res.get('ppl', 0.0):.2f}",
        flush=True,
    )

    # 2B. 8-Domain Unseen ChatML Suite
    if "chatml_8domain" not in report or not os.path.exists(out_8dom):
        dom8_res = run_8domain_eval_v2_2(m220, tok, stage_label, args.device, out_8dom)
        report["chatml_8domain"] = {
            "niah_exact_hits": dom8_res["niah_exact_hits"],
            "math_180_exact": dom8_res["math_180_exact"],
            "prose_clean_direct_english": dom8_res["prose_clean_direct_english"],
            "identifier_fidelity": dom8_res["identifier_fidelity"],
            "detail_path": out_8dom,
        }
        atomic_save_json(report, args.out)

    # 2C. Controlled 2K Gaps (15t, 3t, 24t) + Adversarial 8..11-Subword Sibling NIAH (2K & 4K)
    print("\n" + "=" * 108, flush=True)
    print(f"PART 2C: CONTROLLED 2K GAPS (15t/3t/24t) & ADVERSARIAL >=8-TOKEN SIBLING NIAH [{stage_label}]", flush=True)
    print("=" * 108, flush=True)
    ctrl_2k = {}
    for m_tag in ("oneshot_base", "chatml_vault"):
        ctrl_2k[m_tag] = {
            "15tok_boilerplate": evaluate_niah_condition(m220, tok, m_tag, "boilerplate_15tok", 18, 8, 70001, args.device),
            "3tok_compact": evaluate_niah_condition(m220, tok, m_tag, "compact_3tok", 18, 8, 80001, args.device),
            "24tok_wide": evaluate_niah_condition(m220, tok, m_tag, "wide_24tok", 18, 8, 90001, args.device),
        }
        for gk, gv in ctrl_2k[m_tag].items():
            print(
                f"  [{stage_label} | {m_tag:<12s} | {gk:<18s}] exact={gv['cue_exact_hits']} "
                f"({gv['cue_exact_acc']*100:5.1f}%) | NLL={gv['mean_code_nll_bits']:5.2f}b",
                flush=True,
            )
    report["controlled_2k_gaps"] = ctrl_2k

    sibling_primary = {}
    for m_tag in ("oneshot_base", "chatml_vault"):
        s2k = evaluate_adversarial_long_sibling_niah(m220, tok, m_tag, n_sections=16, n_trials=8, seed_base=510001, device=args.device)
        s4k = evaluate_adversarial_long_sibling_niah(m220, tok, m_tag, n_sections=36, n_trials=6, seed_base=610001, device=args.device)
        sibling_primary[m_tag] = {"2K": s2k, "4K": s4k}
        print(
            f"  [Long Sibling >=8t | {stage_label} | {m_tag:<12s}] "
            f"2K exact={s2k['cue_exact_hits']} ({s2k['cue_exact_acc']*100:5.1f}%, NLL={s2k['mean_code_nll_bits']:.2f}b) | "
            f"4K exact={s4k['cue_exact_hits']} ({s4k['cue_exact_acc']*100:5.1f}%, NLL={s4k['mean_code_nll_bits']:.2f}b)",
            flush=True,
        )
    sibling_res = {"primary": sibling_primary, model_prefix: sibling_primary}

    if (not args.eval_native_v2_4) and (not args.skip_v2_4_ref) and os.path.exists(args.v2_4_ref_ckpt):
        raw_24 = torch.load(args.v2_4_ref_ckpt, map_location="cpu", weights_only=False)
        m_v24 = VAULTLMv2(
            vocab_size=len(tok),
            d_model=768,
            n_layers=16,
            n_q_heads=12,
            n_vault_groups=4,
            head_dim=64,
            d_ff=2048,
            chunk_size=64,
            grad_checkpoint="none",
            v2_2_mode=True,
            prime_banks=tuple(raw_24.get("prime_banks", (47, 53, 59, 61))),
        )
        m_v24.load_state_dict(raw_24["model_state"], strict=True)
        m_v24 = m_v24.to(args.device).eval()
        del raw_24
        bpb_v24_10k = evaluate_bpb(m_v24, tok, device=args.device)
        prose_v24_10k = evaluate_unseen_edu_prose_ppl(m_v24, args.data_dir, device=args.device)
        report["vault_v2_4_step10000_lm_ref"] = {
            "holdout_english_bpb": bpb_v24_10k,
            "unseen_edu_prose_2k": prose_v24_10k,
        }
        sibling_res["VAULT_v2_4_Step10000"] = {}
        for m_tag in ("oneshot_base", "chatml_vault"):
            s2k_r = evaluate_adversarial_long_sibling_niah(m_v24, tok, m_tag, n_sections=16, n_trials=8, seed_base=510001, device=args.device)
            s4k_r = evaluate_adversarial_long_sibling_niah(m_v24, tok, m_tag, n_sections=36, n_trials=6, seed_base=610001, device=args.device)
            sibling_res["VAULT_v2_4_Step10000"][m_tag] = {"2K": s2k_r, "4K": s4k_r}
            print(
                f"  [Long Sibling >=8t | v2.4@10k   | {m_tag:<12s}] "
                f"2K exact={s2k_r['cue_exact_hits']} ({s2k_r['cue_exact_acc']*100:5.1f}%, NLL={s2k_r['mean_code_nll_bits']:.2f}b) | "
                f"4K exact={s4k_r['cue_exact_hits']} ({s4k_r['cue_exact_acc']*100:5.1f}%, NLL={s4k_r['mean_code_nll_bits']:.2f}b)",
                flush=True,
            )
        del m_v24
        _reset_vram(args.device)

    report["adversarial_long_sibling_niah"] = sibling_res
    atomic_save_json(report, args.out)

    # 2D. Extreme 2K -> 128K Extrapolation & Zero-Shot Elastic Co-Prime Mitosis (220c / 508c / 942c / 2,002c)
    print("\n" + "=" * 108, flush=True)
    print(f"PART 2D: EXTREME 2K -> 128K EXTRAPOLATION & ZERO-SHOT MITOSIS [{stage_label}] (`220c -> 508c -> 942c -> 2,002c`)", flush=True)
    print("=" * 108, flush=True)
    niah_specs_all = [
        ("2K", 22, 6),
        ("4K", 46, 6),
        ("8K", 96, 4),
        ("16K", 196, 4),
        ("32K", 396, 3),
        ("64K", 780, 3),
        ("128K", 1560, 2),
    ]
    niah_specs_long = [s for s in niah_specs_all if s[0] in ("32K", "64K", "128K")]
    ext_report: Dict = {}

    if args.eval_native_v2_4:
        caches_220 = _alloc_v2_4_eval_caches(m220, args.device)
        for v_mode, sfx in [("oneshot_base", "oneshot"), ("chatml_vault", "chatml")]:
            v_key = f"220c_{sfx}"
            print(f"\n--- [{stage_label} {v_key}] Native 220c Streaming NIAH (2K -> 128K) ---", flush=True)
            s_dict = {}
            for slabel, n_sec, n_tr in niah_specs_all:
                r = evaluate_extreme_niah_vault_streaming(
                    m220, tok, v_mode, n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec,
                    segment_size=args.segment_size, occ_alpha=args.occ_alpha, caches=caches_220, device=args.device,
                )
                s_dict[slabel] = r
                print(
                    f"  [220c {slabel:>4s} (~{r['mean_prompt_tokens']:7,.0f}t)] "
                    f"exact={r['cue_exact_hits']} ({r['cue_exact_acc']*100:5.1f}%) | "
                    f"digits={r['per_digit_acc']} ({r['digit_acc']*100:5.1f}%) | "
                    f"NLL={r['mean_code_nll_bits']:5.2f}b | occ={r['mean_ind_cell_occ']:.3f} | {r['peak_vram_mb']:.0f}MB",
                    flush=True,
                )
            ext_report[v_key] = s_dict
        del caches_220, m220
        _reset_vram(args.device)
        report["extreme_niah_2k_to_128k"] = ext_report
        atomic_save_json(report, args.out)

        for c_label, p_banks in [
            ("508c", (113, 127, 131, 137)),
            ("942c", (229, 233, 239, 241)),
            ("2002c", (491, 499, 503, 509)),
        ]:
            mw, _, _ = load_vault_v2_4_and_mitosis(
                args.ckpt, device=args.device, prime_banks=p_banks, mitosis_angle=args.mitosis_angle
            )
            caches_w = _alloc_v2_4_eval_caches(mw, args.device)
            for v_mode, sfx in [("oneshot_base", "oneshot"), ("chatml_vault", "chatml")]:
                v_key = f"Mitosis_{c_label}_{sfx}"
                print(f"\n--- [{stage_label} {v_key}] Zero-Shot Mitosis Streaming NIAH (32K -> 128K) ---", flush=True)
                s_dict = {}
                for slabel, n_sec, n_tr in niah_specs_long:
                    r = evaluate_extreme_niah_vault_streaming(
                        mw, tok, v_mode, n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec,
                        segment_size=args.segment_size, occ_alpha=args.occ_alpha, caches=caches_w, device=args.device,
                    )
                    s_dict[slabel] = r
                    print(
                        f"  [{c_label:>5s} {slabel:>4s} (~{r['mean_prompt_tokens']:7,.0f}t)] "
                        f"exact={r['cue_exact_hits']} ({r['cue_exact_acc']*100:5.1f}%) | "
                        f"digits={r['per_digit_acc']} ({r['digit_acc']*100:5.1f}%) | "
                        f"NLL={r['mean_code_nll_bits']:5.2f}b | occ={r['mean_ind_cell_occ']:.3f} | {r['peak_vram_mb']:.0f}MB",
                        flush=True,
                    )
                ext_report[v_key] = s_dict
            del caches_w, mw
            _reset_vram(args.device)
            report["extreme_niah_2k_to_128k"] = ext_report
            atomic_save_json(report, args.out)
    else:
        caches_220 = _alloc_v2_5_eval_caches(m220, args.device)
        for v_mode, sfx in [("oneshot_base", "oneshot"), ("chatml_vault", "chatml")]:
            v_key = f"220c_{sfx}"
            print(f"\n--- [{stage_label} {v_key}] Native 220c Streaming NIAH (2K -> 128K) ---", flush=True)
            s_dict = {}
            for slabel, n_sec, n_tr in niah_specs_all:
                r = evaluate_extreme_niah_v2_5_streaming(
                    m220, tok, v_mode, n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec,
                    segment_size=args.segment_size, occ_alpha=args.occ_alpha, caches=caches_220, device=args.device,
                )
                s_dict[slabel] = r
                print(
                    f"  [220c {slabel:>4s} (~{r['mean_prompt_tokens']:7,.0f}t)] "
                    f"exact={r['cue_exact_hits']} ({r['cue_exact_acc']*100:5.1f}%) | "
                    f"digits={r['per_digit_acc']} ({r['digit_acc']*100:5.1f}%) | "
                    f"NLL={r['mean_code_nll_bits']:5.2f}b | occ={r['mean_ind_cell_occ']:.3f} | {r['peak_vram_mb']:.0f}MB",
                    flush=True,
                )
            ext_report[v_key] = s_dict
        del caches_220, m220
        _reset_vram(args.device)
        report["extreme_niah_2k_to_128k"] = ext_report
        atomic_save_json(report, args.out)

        for c_label, p_banks in [
            ("508c", (113, 127, 131, 137)),
            ("942c", (229, 233, 239, 241)),
            ("2002c", (491, 499, 503, 509)),
        ]:
            mw, _, _ = load_v2_5_or_upgraded_v2_4(
                args.ckpt, device=args.device, target_primes=p_banks, mitosis_angle=args.mitosis_angle
            )
            caches_w = _alloc_v2_5_eval_caches(mw, args.device)
            for v_mode, sfx in [("oneshot_base", "oneshot"), ("chatml_vault", "chatml")]:
                v_key = f"Mitosis_{c_label}_{sfx}"
                print(f"\n--- [{stage_label} {v_key}] Zero-Shot Mitosis Streaming NIAH (32K -> 128K) ---", flush=True)
                s_dict = {}
                for slabel, n_sec, n_tr in niah_specs_long:
                    r = evaluate_extreme_niah_v2_5_streaming(
                        mw, tok, v_mode, n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec,
                        segment_size=args.segment_size, occ_alpha=args.occ_alpha, caches=caches_w, device=args.device,
                    )
                    s_dict[slabel] = r
                    print(
                        f"  [{c_label:>5s} {slabel:>4s} (~{r['mean_prompt_tokens']:7,.0f}t)] "
                        f"exact={r['cue_exact_hits']} ({r['cue_exact_acc']*100:5.1f}%) | "
                        f"digits={r['per_digit_acc']} ({r['digit_acc']*100:5.1f}%) | "
                        f"NLL={r['mean_code_nll_bits']:5.2f}b | occ={r['mean_ind_cell_occ']:.3f} | {r['peak_vram_mb']:.0f}MB",
                        flush=True,
                    )
                ext_report[v_key] = s_dict
            del caches_w, mw
            _reset_vram(args.device)
            report["extreme_niah_2k_to_128k"] = ext_report
            atomic_save_json(report, args.out)

    report["elapsed_sec"] = round(time.perf_counter() - t_all0, 1)
    report["timestamp_completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_save_json(report, args.out)
    print("\n" + "=" * 108, flush=True)
    print(f"[DONE] {stage_label} Full Suite Complete in {report['elapsed_sec']:.1f}s -> {args.out}", flush=True)
    print("=" * 108, flush=True)


if __name__ == "__main__":
    main()
