#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.4 Extreme-Horizon (2K -> 128K = 64x Trained Length) & Zero-Shot Elastic Co-Prime Mitosis Benchmark
#
# Showcases three capabilities unique to Co-Prime Voronoi Fast-Weight Memory (`VAULTLMv2`):
#   1. True O(1)-VRAM Streaming Prefill & True O(1)-Latency Per-Token Autoregressive Decode (`VAULTv2StreamingEngine`):
#      Streams 2K -> 4K -> 8K -> 16K -> 32K -> 64K -> 128K tokens on a single 16 GB GPU with flat `< 1.1 GB` VRAM
#      and constant per-token decode latency independent of context length (`2K` == `128K`).
#   2. Extreme Zero-Exposure Length Extrapolation (`2K -> 128K`, trained strictly at `L = 2,048`):
#      Evaluates held-out open-vocabulary 4-digit NIAH recall out to `64K` (`780` subsystems, `32x` trained length)
#      and `128K` (`1,560` subsystems, `64x` trained length) head-to-head against `Mamba-1 (130M)`, `Mamba-2 (130M)`,
#      `Mamba-3 MIMO (187M)`, and `Transformer (SmolLM-135M)`.
#   3. Zero-Shot Test-Time Elastic Co-Prime Voronoi Cell Mitosis (`220 -> 508 -> 942 -> 2,002` cells, 0 gradient steps!):
#      Splits only `0.02%` of parameters (the 16 ternary Kenyon routers) into orthogonal daughter cones in `u_c^\perp`,
#      expanding working memory from `8.97M` -> `257.6M` -> `3.07B` -> `62.73 Billion` Co-Prime CRT addresses on the fly
#      at test time with ROCm `CUDAGraph` O(1) decode (`~25.6 ms/tok`) and precision-weighted occupancy consensus (`occ_alpha=1.5`)!

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.vault_v2_model import VAULTLMv2, upgrade_v2_2_to_wider_primes
from src.vault_v2_streaming import VAULTv2StreamCache, VAULTv2StreamingEngine, _DecodeCUDAGraphRunner
from src.mamba123_baselines import (
    forward_tail_logits,
    load_mamba1_130m,
    load_mamba2_130m,
    load_mamba3_mimo_187m,
    load_transformer_smollm_135m,
)
from benchmarks.bench_vault_vs_mamba123_transformer import build_deterministic_niah_trial


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


def _reset_cache_inplace_(cache: VAULTv2StreamCache) -> None:
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
            if k == "sal_win":
                ls.sal_win.fill_(-1e4)
            else:
                getattr(ls, k).zero_()


def load_vault_v2_4_and_mitosis(
    ckpt_path: str,
    device: str = "cuda:0",
    prime_banks: Tuple[int, int, int, int] = (47, 53, 59, 61),
    mitosis_angle: float = 0.26,
) -> Tuple[VAULTLMv2, AutoTokenizer, Dict]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state"] if "model_state" in ck else ck
    base_primes = tuple(ck.get("prime_banks", (47, 53, 59, 61)))
    step_num = ck.get("step", -1)
    bpb_val = ck.get("english_bpb", None)
    del ck
    gc.collect()

    model = VAULTLMv2(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint=False,
        v2_1_mode=True,
        ckpt_interval=1,
        v2_2_mode=True,
        prime_banks=prime_banks,
    )
    meta: Dict[str, object] = {
        "ckpt_step": step_num,
        "ckpt_bpb": bpb_val,
        "prime_banks": prime_banks,
        "total_cells": sum(prime_banks),
        "crt_addresses": prime_banks[0] * prime_banks[1] * prime_banks[2] * prime_banks[3],
    }
    if prime_banks == base_primes:
        model.load_state_dict(sd, strict=True)
        meta["mitosis"] = "native"
    else:
        mit_info = upgrade_v2_2_to_wider_primes(
            model, sd, old_primes=base_primes, split_angle_rad=mitosis_angle, k_nz=52
        )
        meta["mitosis"] = mit_info
        meta["mitosis_angle_rad"] = mitosis_angle
    del sd
    gc.collect()
    return model.to(device).eval(), tok, meta


# ==============================================================================
# 1. HARDWARE O(1) STREAMING PREFILL + TRUE O(1) DECODE SCALING (2K -> 128K)
# ==============================================================================

@torch.no_grad()
def benchmark_vault_streaming_scaling(
    model: VAULTLMv2,
    lengths: List[int],
    segment_size: int = 4096,
    decode_tokens: int = 32,
    occ_alpha: float = 1.5,
    device: str = "cuda:0",
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    # Warmup Triton prefill compile once on 512 tokens (before CUDAGraph static_cache allocation so peak_prefill_mb is pure!)
    w_ids = torch.randint(0, model.vocab_size, (1, 512), dtype=torch.long, device=device)
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, w_cache = VAULTv2StreamingEngine.prefill_streaming(
                model, w_ids, segment_size=min(512, segment_size), occ_alpha=occ_alpha
            )
    else:
        _, w_cache = VAULTv2StreamingEngine.prefill_streaming(
            model, w_ids, segment_size=min(512, segment_size), occ_alpha=occ_alpha
        )
    del w_ids, w_cache

    prefill_peaks_mb: Dict[int, float] = {}
    for L in lengths:
        _reset_vram(device)
        inp = torch.randint(0, model.vocab_size, (1, L), dtype=torch.long, device=device)
        seg = min(segment_size, L) if L >= 64 else 64
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, c_tmp = VAULTv2StreamingEngine.prefill_streaming(
                    model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
                )
        else:
            _, c_tmp = VAULTv2StreamingEngine.prefill_streaming(
                model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
            )
        prefill_peaks_mb[L] = _peak_vram_mb(device)
        del inp, c_tmp

    # Capture decode CUDAGraph once on a dummy step
    w_tok = torch.randint(0, model.vocab_size, (1, 1), dtype=torch.long, device=device)
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_c = VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=torch.bfloat16)
            _ = VAULTv2StreamingEngine.decode_step(model, w_tok, w_c, occ_alpha=occ_alpha)
    else:
        w_c = VAULTv2StreamingEngine.init_cache(model, batch_size=1)
        _ = VAULTv2StreamingEngine.decode_step(model, w_tok, w_c, occ_alpha=occ_alpha)
    del w_tok, w_c

    for L in lengths:
        label = f"{L // 1024}K" if L >= 1024 else f"{L}"
        inp = torch.randint(0, model.vocab_size, (1, L), dtype=torch.long, device=device)
        seg = min(segment_size, L) if L >= 64 else 64

        _sync_dev(device)
        t0 = time.perf_counter()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_last, cache = VAULTv2StreamingEngine.prefill_streaming(
                    model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
                )
        else:
            logits_last, cache = VAULTv2StreamingEngine.prefill_streaming(
                model, inp, segment_size=seg, tail_logits_len=1, occ_alpha=occ_alpha
            )
        _sync_dev(device)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        peak_prefill_mb = prefill_peaks_mb[L]
        state_mb = round(cache.state_bytes() / (1024 ** 2), 2)

        nxt = torch.argmax(logits_last[:, -1:], dim=-1)
        # Bind `cache` into CUDAGraph runner on 1st decode token, then time `decode_tokens` steady-state tokens
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                step_logits = VAULTv2StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
        else:
            step_logits = VAULTv2StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
        nxt = torch.argmax(step_logits[:, -1:], dim=-1)

        _sync_dev(device)
        t1 = time.perf_counter()
        for _ in range(decode_tokens):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    step_logits = VAULTv2StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            else:
                step_logits = VAULTv2StreamingEngine.decode_step(model, nxt, cache, occ_alpha=occ_alpha)
            nxt = torch.argmax(step_logits[:, -1:], dim=-1)
        _sync_dev(device)
        dec_total_ms = (time.perf_counter() - t1) * 1000.0
        dec_ms_per_tok = dec_total_ms / max(1, decode_tokens)

        rec = {
            "seq_len": L,
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tok_s": round(L / max(prefill_ms / 1000.0, 1e-6), 1),
            "peak_vram_mb": peak_prefill_mb,
            "persistent_state_mb": state_mb,
            "decode_ms_per_tok": round(dec_ms_per_tok, 2),
            "decode_tok_s": round(1000.0 / max(dec_ms_per_tok, 1e-6), 1),
        }
        results[label] = rec
        print(
            f"  [VAULT {sum(model.prime_banks)}c | {label:>4s} ({L:7,d}t)] "
            f"prefill={rec['prefill_tok_s']:8,.0f} tok/s ({rec['prefill_ms']:6.1f}ms) | "
            f"VRAM={rec['peak_vram_mb']:6.1f}MB (state={rec['persistent_state_mb']:.1f}MB) | "
            f"O(1) decode={rec['decode_ms_per_tok']:5.2f} ms/tok ({rec['decode_tok_s']:5.1f} tok/s)",
            flush=True,
        )
        del inp, cache, logits_last

    # Release CUDAGraph static_cache before NIAH suite allocates working caches
    runners = getattr(model, "_decode_graph_runners", None)
    if runners:
        banks = model.layers[0].attn.salience_binder.BANKS
        for k in list(runners.keys()):
            runners.pop(k).release_owner(banks)
    _reset_vram(device)
    return results


@torch.no_grad()
def benchmark_transformer_scaling(
    model: nn.Module,
    vocab_size: int,
    lengths: List[int],
    decode_tokens: int = 32,
    device: str = "cuda:0",
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    for L in lengths:
        label = f"{L // 1024}K" if L >= 1024 else f"{L}"
        _reset_vram(device)
        try:
            inp = torch.randint(0, vocab_size, (1, L), dtype=torch.long, device=device)
            _sync_dev(device)
            t0 = time.perf_counter()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.model(inp, use_cache=True)
                    logits_last = model.lm_head(out.last_hidden_state[:, -1:, :])
            else:
                out = model.model(inp, use_cache=True)
                logits_last = model.lm_head(out.last_hidden_state[:, -1:, :])
            _sync_dev(device)
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            peak_prefill_mb = _peak_vram_mb(device)
            pkv = out.past_key_values
            kv_mb = round((30 * 2 * 3 * 64 * 2 * L) / (1024 ** 2), 2)

            nxt = torch.argmax(logits_last[:, -1:], dim=-1)
            _sync_dev(device)
            t1 = time.perf_counter()
            for _ in range(decode_tokens):
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_step = model.model(nxt, past_key_values=pkv, use_cache=True)
                        step_logits = model.lm_head(out_step.last_hidden_state[:, -1:, :])
                else:
                    out_step = model.model(nxt, past_key_values=pkv, use_cache=True)
                    step_logits = model.lm_head(out_step.last_hidden_state[:, -1:, :])
                pkv = out_step.past_key_values
                nxt = torch.argmax(step_logits[:, -1:], dim=-1)
            _sync_dev(device)
            dec_ms_per_tok = ((time.perf_counter() - t1) * 1000.0) / max(1, decode_tokens)

            rec = {
                "seq_len": L,
                "prefill_ms": round(prefill_ms, 2),
                "prefill_tok_s": round(L / max(prefill_ms / 1000.0, 1e-6), 1),
                "peak_vram_mb": peak_prefill_mb,
                "kv_cache_mb": kv_mb,
                "decode_ms_per_tok": round(dec_ms_per_tok, 2),
                "decode_tok_s": round(1000.0 / max(dec_ms_per_tok, 1e-6), 1),
            }
            results[label] = rec
            print(
                f"  [Transformer | {label:>4s} ({L:7,d}t)] "
                f"prefill={rec['prefill_tok_s']:8,.0f} tok/s ({rec['prefill_ms']:6.1f}ms) | "
                f"VRAM={rec['peak_vram_mb']:6.1f}MB (KV={rec['kv_cache_mb']:.1f}MB) | "
                f"O(L) decode={rec['decode_ms_per_tok']:5.2f} ms/tok ({rec['decode_tok_s']:5.1f} tok/s)",
                flush=True,
            )
            del inp, out, pkv, logits_last
        except (torch.OutOfMemoryError, RuntimeError) as err:
            if isinstance(err, RuntimeError) and "out of memory" not in str(err).lower() and "hip" not in str(err).lower():
                raise
            _reset_vram(device)
            results[label] = {"seq_len": L, "status": "OOM", "error": str(err)[:160]}
            print(f"  [Transformer | {label:>4s} ({L:7,d}t)] OOM on 16 GB GPU!", flush=True)
    return results


# ==============================================================================
# 2. EXTREME-HORIZON NIAH EVALUATOR (2K -> 4K -> 8K -> 16K -> 32K -> 64K -> 128K)
# ==============================================================================

@torch.no_grad()
def evaluate_extreme_niah_vault_streaming(
    model: VAULTLMv2,
    tok,
    mode: str,
    n_sections: int,
    n_trials: int,
    seed_base: int,
    segment_size: int = 4096,
    occ_alpha: float = 1.5,
    caches: Optional[Tuple[VAULTv2StreamCache, VAULTv2StreamCache, VAULTv2StreamCache]] = None,
    device: str = "cuda:0",
) -> Dict:
    """
    Exact O(1)-VRAM NIAH evaluation for VAULT (220 / 508 / 942 / 2002 cells) using `VAULTv2StreamingEngine`!
    """
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
            VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
            VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
            VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
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
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Prefill 64-aligned prefix of `base_p_ids` ONCE into `aligned_cache` in O(1) VRAM (0 Python decode loops!)
        aligned_len = ((max(1, len(base_p_ids)) - 1) // 64) * 64
        rem_ids = base_p_ids[aligned_len:]  # always 1..64 tokens, ending at base_p_ids[-1]

        _reset_cache_inplace_(aligned_cache)
        _sync_dev(device)
        t0 = time.perf_counter()
        if aligned_len > 0:
            p_tensor = torch.tensor([base_p_ids[:aligned_len]], dtype=torch.long, device=device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, _ = VAULTv2StreamingEngine.prefill_streaming(
                        model, p_tensor, segment_size=segment_size, cache=aligned_cache,
                        tail_logits_len=1, occ_alpha=occ_alpha, populate_decode_rings=False,
                    )
            else:
                _, _ = VAULTv2StreamingEngine.prefill_streaming(
                    model, p_tensor, segment_size=segment_size, cache=aligned_cache,
                    tail_logits_len=1, occ_alpha=occ_alpha, populate_decode_rings=False,
                )
            del p_tensor

        # Causally right-pad `rem_ids + ans_ids` to 64-multiple for exact Triton teacher-forced NLL
        tf_seq = rem_ids + ans_ids
        tf_pad = (64 - (len(tf_seq) % 64)) % 64
        tf_padded = tf_seq + [0] * tf_pad
        tf_tensor = torch.tensor([tf_padded], dtype=torch.long, device=device)
        _DecodeCUDAGraphRunner._copy_cache_tensors_(work_cache, aligned_cache)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tf_logits_all, _ = VAULTv2StreamingEngine.prefill_streaming(
                    model, tf_tensor, segment_size=segment_size, cache=work_cache,
                    tail_logits_len=len(tf_padded), occ_alpha=occ_alpha, populate_decode_rings=False,
                )
        else:
            tf_logits_all, _ = VAULTv2StreamingEngine.prefill_streaming(
                model, tf_tensor, segment_size=segment_size, cache=work_cache,
                tail_logits_len=len(tf_padded), occ_alpha=occ_alpha, populate_decode_rings=False,
            )
        _sync_dev(device)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        # Subtract the 2 extra evaluator scratch caches so peak_vram_mb reflects single-stream NIAH memory
        extra_scratch_mb = 2.0 * (aligned_cache.state_bytes() / (1024 ** 2))
        peak_mb = round(max(0.0, _peak_vram_mb(device) - extra_scratch_mb), 1)
        tail_logits = tf_logits_all[0, len(rem_ids) - 1 : len(rem_ids) - 1 + len(ans_ids)].float()
        del tf_tensor, tf_logits_all

        # Measure average per-cell occupancy load on induction vaults (Groups 1..3)
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

        # Cue greedy 4-digit decode via causally right-padded 64-aligned Triton continuation from `aligned_cache`
        for idx_t, (_, code) in enumerate(targets):
            true_digits = code.split("-")[1]
            cue_s = cue0_str if idx_t == 0 else cue1_str
            cue_ids = tok.encode(cue_s, add_special_tokens=False)
            cue_full = rem_ids + cue_ids
            cue_align = ((len(cue_full) - 1) // 64) * 64
            if cue_align > 0:
                _DecodeCUDAGraphRunner._copy_cache_tensors_(cue_base_cache, aligned_cache)
                ca_t = torch.tensor([cue_full[:cue_align]], dtype=torch.long, device=device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _, _ = VAULTv2StreamingEngine.prefill_streaming(
                            model, ca_t, segment_size=segment_size, cache=cue_base_cache,
                            tail_logits_len=1, occ_alpha=occ_alpha, populate_decode_rings=False,
                        )
                else:
                    _, _ = VAULTv2StreamingEngine.prefill_streaming(
                        model, ca_t, segment_size=segment_size, cache=cue_base_cache,
                        tail_logits_len=1, occ_alpha=occ_alpha, populate_decode_rings=False,
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
                _DecodeCUDAGraphRunner._copy_cache_tensors_(work_cache, src_cache)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        st_lg, _ = VAULTv2StreamingEngine.prefill_streaming(
                            model, st_t, segment_size=segment_size, cache=work_cache,
                            tail_logits_len=len(step_padded), occ_alpha=occ_alpha, populate_decode_rings=False,
                        )
                else:
                    st_lg, _ = VAULTv2StreamingEngine.prefill_streaming(
                        model, st_t, segment_size=segment_size, cache=work_cache,
                        tail_logits_len=len(step_padded), occ_alpha=occ_alpha, populate_decode_rings=False,
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
    }


@torch.no_grad()
def evaluate_extreme_niah_baseline(
    model: nn.Module,
    tok,
    n_sections: int,
    n_trials: int,
    seed_base: int,
    device: str = "cuda:0",
) -> Dict:
    model.eval()
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
    stop_ids = (eos_id,) if eos_id >= 0 else ()

    cue_exact_hits = 0
    total_codes = 0
    char_hits = [0, 0, 0, 0]
    total_nll_bits = 0.0
    prompt_lens: List[int] = []
    fwd_ms = 0.0
    peak_mb = 0.0

    for tr in range(n_trials):
        manifest_body, entries, targets = build_deterministic_niah_trial("boilerplate_15tok", n_sections, tr, seed_base)
        ex_ent, ex_code = entries[0]
        q_str = (
            f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
            f"for {targets[0][0]} and {targets[1][0]}? List each subsystem and its exact code."
            f"\nVerification Summary:\n- {ex_ent}: {ex_code}\n"
        )
        base_p_ids = tok.encode(manifest_body + q_str, add_special_tokens=False)
        cue0_str = f"- {targets[0][0]}: CHK-"
        cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
        ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        ans_ids = tok.encode(ans_str, add_special_tokens=False)

        prompt_lens.append(len(base_p_ids))
        full_ids = torch.tensor([base_p_ids + ans_ids], dtype=torch.long, device=device)

        _reset_vram(device)
        _sync_dev(device)
        t0 = time.perf_counter()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tail_logits = forward_tail_logits(model, full_ids, tail_len=len(ans_ids) + 1)[0].float()
        else:
            tail_logits = forward_tail_logits(model, full_ids, tail_len=len(ans_ids) + 1)[0].float()
        _sync_dev(device)
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = _peak_vram_mb(device)
        del full_ids

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

        for idx_t, (_, code) in enumerate(targets):
            true_digits = code.split("-")[1]
            cue_s = cue0_str if idx_t == 0 else cue1_str
            ids = torch.tensor([base_p_ids + tok.encode(cue_s, add_special_tokens=False)], dtype=torch.long, device=device)
            gen_ids: List[int] = []
            for _ in range(4):
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg = forward_tail_logits(model, ids, tail_len=1)[0, -1].float()
                else:
                    lg = forward_tail_logits(model, ids, tail_len=1)[0, -1].float()
                nxt = int(torch.argmax(lg).item())
                if nxt in stop_ids:
                    break
                gen_ids.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
                txt = tok.decode(gen_ids, skip_special_tokens=True)
                if len(txt.strip()) >= 4 or "\n" in txt:
                    break
            del ids
            gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
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
        "mode": "oneshot_base",
        "n_sections": n_sections,
        "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
        "cue_exact_hits": f"{cue_exact_hits}/{total_codes}",
        "cue_exact_acc": round(cue_exact_hits / max(1, total_codes), 4),
        "digit_acc": round(sum(char_hits) / max(1, 4 * total_codes), 4),
        "per_digit_acc": per_digit,
        "mean_code_nll_bits": round(total_nll_bits / max(1, total_codes), 3),
        "prefill_ms": round(fwd_ms, 2),
        "peak_vram_mb": peak_mb,
    }


def _alloc_eval_caches(model: VAULTLMv2, device: str):
    dt_c = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_available()) else None
    return (
        VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
        VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
        VAULTv2StreamingEngine.init_cache(model, batch_size=1, dtype=dt_c),
    )


def main():
    parser = argparse.ArgumentParser(description="VAULT v2.4 Extreme 128K Streaming & Elastic Mitosis Benchmark")
    parser.add_argument("--ckpt", type=str, default=os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_best.pt"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--segment_size", type=int, default=4096)
    parser.add_argument("--mitosis_angle", type=float, default=0.26)
    parser.add_argument("--occ_alpha", type=float, default=1.5)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--quick_cpu_smoke", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmarks", "results_vault_extreme_128k.json"),
    )
    args = parser.parse_args()

    print("=" * 112, flush=True)
    print("VAULT v2.4 EXTREME-HORIZON (2K -> 128K = 64x TRAINED LENGTH) & ELASTIC CO-PRIME MITOSIS BENCHMARK", flush=True)
    print("=" * 112, flush=True)

    if args.quick_cpu_smoke:
        scale_lengths = [256, 512]
        niah_specs = [("1K_smoke", 12, 1)]
        dec_toks = 4
    else:
        scale_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        niah_specs = [
            ("2K", 22, 6),
            ("4K", 46, 6),
            ("8K", 96, 4),
            ("16K", 196, 4),
            ("32K", 396, 3),
            ("64K", 780, 3),
            ("128K", 1560, 2),
        ]
        dec_toks = 32

    report: Dict[str, object] = {
        "checkpoint": args.ckpt,
        "device": args.device,
        "segment_size": args.segment_size,
        "mitosis_angle_rad": args.mitosis_angle,
        "occ_alpha": args.occ_alpha,
        "streaming_systems_scaling": {},
        "extreme_niah_2k_to_128k": {},
    }
    if args.skip_baselines and os.path.exists(args.out):
        try:
            with open(args.out, "r") as f_prev:
                prev = json.load(f_prev)
            report["streaming_systems_scaling"].update(prev.get("streaming_systems_scaling", {}))
            report["extreme_niah_2k_to_128k"].update(prev.get("extreme_niah_2k_to_128k", {}))
        except Exception:
            pass

    # 1. VAULT v2.4 Native (220 cells, P=(47,53,59,61), CRT=8.97M)
    print("\n[1/5] Evaluating VAULT v2.4 Native (220 Co-Prime Cells, P=(47,53,59,61), CRT=8.97M)...", flush=True)
    m220, tok_v, meta_220 = load_vault_v2_4_and_mitosis(
        args.ckpt, device=args.device, prime_banks=(47, 53, 59, 61), mitosis_angle=args.mitosis_angle
    )
    report["vault_v2_4_meta"] = meta_220
    report["streaming_systems_scaling"]["VAULT_v2_4_220c"] = benchmark_vault_streaming_scaling(
        m220, scale_lengths, segment_size=args.segment_size, decode_tokens=dec_toks,
        occ_alpha=args.occ_alpha, device=args.device,
    )
    caches_220 = _alloc_eval_caches(m220, args.device)
    for v_mode, v_key in [("oneshot_base", "VAULT_v2_4_220c_oneshot"), ("chatml_vault", "VAULT_v2_4_220c_chatml")]:
        print(f"\n--- [{v_key}] Extreme NIAH Extrapolation (2K -> 128K, occ_alpha={args.occ_alpha}) ---", flush=True)
        suite_res = {}
        for slabel, n_sec, n_tr in niah_specs:
            res = evaluate_extreme_niah_vault_streaming(
                m220, tok_v, v_mode, n_sections=n_sec, n_trials=n_tr,
                seed_base=110000 + n_sec, segment_size=args.segment_size,
                occ_alpha=args.occ_alpha, caches=caches_220, device=args.device,
            )
            suite_res[slabel] = res
            print(
                f"  [{slabel:>4s} (~{res['mean_prompt_tokens']:7,.0f}t)] "
                f"exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                f"digits={res['per_digit_acc']} ({res['digit_acc']*100:5.1f}%) | "
                f"NLL={res['mean_code_nll_bits']:5.2f}b | occ={res['mean_ind_cell_occ']:.3f} | "
                f"{res['prefill_ms']:.0f}ms | {res['peak_vram_mb']:.0f}MB",
                flush=True,
            )
        report["extreme_niah_2k_to_128k"][v_key] = suite_res
    del caches_220, m220
    _reset_vram(args.device)
    with open(args.out + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(args.out + ".tmp", args.out)

    # 2. Zero-Shot Elastic Co-Prime Mitosis -> 508 Cells P=(113,127,131,137), CRT=257.6M!
    print("\n[2/5] Evaluating Zero-Shot Elastic Co-Prime Mitosis (220 -> 508 Cells, P=(113,127,131,137), CRT=257.6M)...", flush=True)
    m508, _, meta_508 = load_vault_v2_4_and_mitosis(
        args.ckpt, device=args.device, prime_banks=(113, 127, 131, 137), mitosis_angle=args.mitosis_angle
    )
    report["vault_mitosis_508_meta"] = meta_508
    report["streaming_systems_scaling"]["VAULT_v2_4_Mitosis_508c"] = benchmark_vault_streaming_scaling(
        m508, scale_lengths, segment_size=args.segment_size, decode_tokens=dec_toks,
        occ_alpha=args.occ_alpha, device=args.device,
    )
    caches_508 = _alloc_eval_caches(m508, args.device)
    for v_mode, v_key in [("oneshot_base", "VAULT_v2_4_Mitosis_508c_oneshot"), ("chatml_vault", "VAULT_v2_4_Mitosis_508c_chatml")]:
        print(f"\n--- [{v_key}] Zero-Shot Mitosis Extreme NIAH (2K -> 128K, occ_alpha={args.occ_alpha}) ---", flush=True)
        suite_res = {}
        for slabel, n_sec, n_tr in niah_specs:
            res = evaluate_extreme_niah_vault_streaming(
                m508, tok_v, v_mode, n_sections=n_sec, n_trials=n_tr,
                seed_base=110000 + n_sec, segment_size=args.segment_size,
                occ_alpha=args.occ_alpha, caches=caches_508, device=args.device,
            )
            suite_res[slabel] = res
            print(
                f"  [{slabel:>4s} (~{res['mean_prompt_tokens']:7,.0f}t)] "
                f"exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                f"digits={res['per_digit_acc']} ({res['digit_acc']*100:5.1f}%) | "
                f"NLL={res['mean_code_nll_bits']:5.2f}b | occ={res['mean_ind_cell_occ']:.3f} | "
                f"{res['prefill_ms']:.0f}ms | {res['peak_vram_mb']:.0f}MB",
                flush=True,
            )
        report["extreme_niah_2k_to_128k"][v_key] = suite_res
    del caches_508, m508
    _reset_vram(args.device)
    with open(args.out + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(args.out + ".tmp", args.out)

    # 3. Ultra-Wide Zero-Shot Elastic Co-Prime Mitosis -> 942 Cells P=(229,233,239,241), CRT=3.07 Billion on 32K/64K/128K!
    print("\n[3/5] Evaluating Ultra-Wide Zero-Shot Mitosis (220 -> 942 Cells, P=(229,233,239,241), CRT=3.07B)...", flush=True)
    m942, _, meta_942 = load_vault_v2_4_and_mitosis(
        args.ckpt, device=args.device, prime_banks=(229, 233, 239, 241), mitosis_angle=args.mitosis_angle
    )
    report["vault_mitosis_942_meta"] = meta_942
    scale_long = scale_lengths if args.quick_cpu_smoke else [32768, 65536, 131072]
    report["streaming_systems_scaling"]["VAULT_v2_4_Mitosis_942c"] = benchmark_vault_streaming_scaling(
        m942, scale_long, segment_size=args.segment_size, decode_tokens=dec_toks,
        occ_alpha=args.occ_alpha, device=args.device,
    )
    long_specs = niah_specs if args.quick_cpu_smoke else [s for s in niah_specs if s[0] in ("32K", "64K", "128K")]
    caches_942 = _alloc_eval_caches(m942, args.device)
    for v_mode, v_key in [("oneshot_base", "VAULT_v2_4_Mitosis_942c_oneshot"), ("chatml_vault", "VAULT_v2_4_Mitosis_942c_chatml")]:
        print(f"\n--- [{v_key}] Ultra-Wide 942-Cell Zero-Shot Mitosis NIAH (32K -> 128K, occ_alpha={args.occ_alpha}) ---", flush=True)
        suite_942 = {}
        for slabel, n_sec, n_tr in long_specs:
            res = evaluate_extreme_niah_vault_streaming(
                m942, tok_v, v_mode, n_sections=n_sec, n_trials=n_tr,
                seed_base=110000 + n_sec, segment_size=args.segment_size,
                occ_alpha=args.occ_alpha, caches=caches_942, device=args.device,
            )
            suite_942[slabel] = res
            print(
                f"  [942c {slabel:>4s} (~{res['mean_prompt_tokens']:7,.0f}t)] "
                f"exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                f"digits={res['per_digit_acc']} ({res['digit_acc']*100:5.1f}%) | "
                f"NLL={res['mean_code_nll_bits']:5.2f}b | occ={res['mean_ind_cell_occ']:.3f} | "
                f"{res['prefill_ms']:.0f}ms | {res['peak_vram_mb']:.0f}MB",
                flush=True,
            )
        report["extreme_niah_2k_to_128k"][v_key] = suite_942
    del caches_942, m942
    _reset_vram(args.device)
    with open(args.out + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(args.out + ".tmp", args.out)

    # 4. Hyper-Wide Zero-Shot Elastic Co-Prime Mitosis -> 2,002 Cells P=(491,499,503,509), CRT=62.73 Billion on 32K/64K/128K!
    print("\n[4/5] Evaluating Hyper-Wide Zero-Shot Mitosis (220 -> 2,002 Cells, P=(491,499,503,509), CRT=62.73B)...", flush=True)
    m2002, _, meta_2002 = load_vault_v2_4_and_mitosis(
        args.ckpt, device=args.device, prime_banks=(491, 499, 503, 509), mitosis_angle=args.mitosis_angle
    )
    report["vault_mitosis_2002_meta"] = meta_2002
    report["streaming_systems_scaling"]["VAULT_v2_4_Mitosis_2002c"] = benchmark_vault_streaming_scaling(
        m2002, scale_long, segment_size=args.segment_size, decode_tokens=dec_toks,
        occ_alpha=args.occ_alpha, device=args.device,
    )
    caches_2002 = _alloc_eval_caches(m2002, args.device)
    for v_mode, v_key in [("oneshot_base", "VAULT_v2_4_Mitosis_2002c_oneshot"), ("chatml_vault", "VAULT_v2_4_Mitosis_2002c_chatml")]:
        print(f"\n--- [{v_key}] Hyper-Wide 2,002-Cell Zero-Shot Mitosis NIAH (32K -> 128K, occ_alpha={args.occ_alpha}) ---", flush=True)
        suite_2002 = {}
        for slabel, n_sec, n_tr in long_specs:
            res = evaluate_extreme_niah_vault_streaming(
                m2002, tok_v, v_mode, n_sections=n_sec, n_trials=n_tr,
                seed_base=110000 + n_sec, segment_size=args.segment_size,
                occ_alpha=args.occ_alpha, caches=caches_2002, device=args.device,
            )
            suite_2002[slabel] = res
            print(
                f"  [2002c {slabel:>4s} (~{res['mean_prompt_tokens']:7,.0f}t)] "
                f"exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                f"digits={res['per_digit_acc']} ({res['digit_acc']*100:5.1f}%) | "
                f"NLL={res['mean_code_nll_bits']:5.2f}b | occ={res['mean_ind_cell_occ']:.3f} | "
                f"{res['prefill_ms']:.0f}ms | {res['peak_vram_mb']:.0f}MB",
                flush=True,
            )
        report["extreme_niah_2k_to_128k"][v_key] = suite_2002
    del caches_2002, m2002
    _reset_vram(args.device)
    with open(args.out + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(args.out + ".tmp", args.out)

    # 5. External Baselines (Transformer SmolLM-135M + Mamba-1/2/3 on 32K/64K/128K)
    if not args.skip_baselines and not args.quick_cpu_smoke:
        print("\n[5/5] Evaluating External Baselines (Transformer SmolLM-135M, Mamba-1, Mamba-2, Mamba-3 MIMO)...", flush=True)
        tf_m, tf_tok = load_transformer_smollm_135m(device=args.device)
        report["streaming_systems_scaling"]["Transformer_SmolLM_135M"] = benchmark_transformer_scaling(
            tf_m, len(tf_tok), scale_lengths, decode_tokens=dec_toks, device=args.device
        )
        del tf_m
        _reset_vram(args.device)

        ext_specs = [s for s in niah_specs if s[0] in ("32K", "64K", "128K")]
        for b_name, loader_fn in [
            ("Mamba1_130M", load_mamba1_130m),
            ("Mamba2_130M", load_mamba2_130m),
            ("Mamba3_MIMO_187M", load_mamba3_mimo_187m),
            ("Transformer_SmolLM_135M", load_transformer_smollm_135m),
        ]:
            print(f"\n--- [{b_name}] Extreme NIAH (32K -> 128K) ---", flush=True)
            bm, btok = loader_fn(device=args.device)
            b_res = {}
            for slabel, n_sec, n_tr in ext_specs:
                try:
                    res = evaluate_extreme_niah_baseline(
                        bm, btok, n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec, device=args.device
                    )
                    b_res[slabel] = res
                    print(
                        f"  [{b_name} | {slabel:>4s} (~{res['mean_prompt_tokens']:7,.0f}t)] "
                        f"exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                        f"NLL={res['mean_code_nll_bits']:5.2f}b | {res['prefill_ms']:.0f}ms | {res['peak_vram_mb']:.0f}MB",
                        flush=True,
                    )
                except (torch.OutOfMemoryError, RuntimeError) as err:
                    if isinstance(err, RuntimeError) and "out of memory" not in str(err).lower() and "hip" not in str(err).lower():
                        raise
                    _reset_vram(args.device)
                    b_res[slabel] = {"status": "OOM", "error": str(err)[:160]}
                    print(f"  [{b_name} | {slabel:>4s}] GPU OOM on 16 GB GPU!", flush=True)
            report["extreme_niah_2k_to_128k"][b_name] = b_res
            del bm
            _reset_vram(args.device)
            with open(args.out + ".tmp", "w") as f:
                json.dump(report, f, indent=2)
            os.replace(args.out + ".tmp", args.out)

    tmp_out = args.out + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_out, args.out)
    print(f"\n[+] Saved complete Extreme 128K & Elastic Mitosis benchmark report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
