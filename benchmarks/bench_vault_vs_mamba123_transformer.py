#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Rigorous 5-Architecture Head-to-Head Benchmark on AMD Radeon RX 9060 XT (16 GB, ROCm 7.2, BF16):
#   1. VAULT v2.2 (180.4M, 16L, d_model=768, 0.26B pretrain tokens, trained strictly at L=2048)
#   2. Mamba-1 (130.9M, 24L, d_model=768, d_state=16, 300B pretrain tokens, L=2048)
#   3. Mamba-2 (129.7M, 24L, d_model=768, d_state=128 SSD, 300B pretrain tokens, L=2048)
#   4. Mamba-3 MIMO (187.3M, 12L, d_model=768, d_state=128, mimo_rank=4, Exp-Trap + 4-Slice RoPE, 100B tokens, L=2048)
#   5. Transformer (134.5M, SmolLM-135M, 30L GQA RoPE + ROCm FlashAttention SDPA, 600B tokens, L=2048)
#
# Three Complementary Evaluation Tracks:
#   - Track A: Pretrained Language Modeling (`Bits-Per-Byte` / `BPB`) & 1-Shot In-Context Open-Vocab NIAH (2K..32K)
#   - Track B: Matched L=2048 Multi-Gap Open-Vocab NIAH Fine-Tuning Race (100 steps + prose replay, zero leakage)
#              evaluated on unseen 2K Gaps (+3t, +15t, +24t) AND Zero-Shot 2K -> 4K -> 8K -> 16K -> 32K Extrapolation
#   - Track C: Hardware Systems Scaling (Prefill Latency, Throughput tok/s, Peak VRAM, Persistent Decode Cache 2K..128K)

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.vault_v2_model import VAULTLMv2
from src.mamba123_baselines import (
    forward_tail_logits,
    load_mamba1_130m,
    load_mamba2_130m,
    load_mamba3_mimo_187m,
    load_transformer_smollm_135m,
    persistent_decode_state_mb,
)
from train_vault_v2 import UNSEEN_FILLER, UNSEEN_NIAH_PREFIXES
from prepare_vault_v2_1_data import FILLER_SENTENCES_V2_1, FORBIDDEN_NIAH_SUBWORDS
from prepare_vault_v2_2_data import HOLDOUT_NIAH_PREFIXES, HOLDOUT_SUBSTRINGS, build_clean_open_vocab_pool


# ==============================================================================
# 1. HELD-OUT EDUCATIONAL PROSE PASSAGES FOR TOKENIZER-AGNOSTIC BITS-PER-BYTE
# ==============================================================================

EVAL_BPB_PASSAGES = [
    (
        "Modern superscalar processors exploit instruction-level parallelism through out-of-order execution, "
        "register renaming, and speculative branch prediction. Because false write-after-read and write-after-write "
        "hazards arise solely from the reuse of architectural register names, mapping each destination operand to a "
        "fresh physical register eliminates anti-dependencies and allows independent instructions to issue as soon as "
        "their source values appear on the forwarding network."
    ),
    (
        "In thermodynamics, the second law establishes that the total entropy of an isolated macroscopic system can "
        "never decrease over time. Carnot demonstrated that no heat engine operating between two thermal reservoirs "
        "at absolute temperatures T_hot and T_cold can exceed the reversible efficiency 1 - T_cold / T_hot, regardless "
        "of the working fluid or mechanical construction employed."
    ),
    (
        "Distributed consensus protocols such as Raft and Paxos guarantee linearizable state machine replication across "
        "unreliable networks subject to asynchronous message delays and crash-recovery node failures. Safety relies on "
        "quorum intersection: any two majorities of cluster members share at least one server that retains the highest "
        "committed log term, preventing split-brain divergence during leader elections."
    ),
    (
        "Claude Shannon proved in his noisy-channel coding theorem that reliable communication is achievable at any "
        "information rate strictly below the channel capacity C = B log2(1 + S/N), whereas rates exceeding capacity "
        "incur a strictly positive error probability bounded away from zero. Source coding conversely compresses "
        "stationary ergodic sequences down to their fundamental entropy rate in bits per symbol."
    ),
    (
        "Maxwell unified electricity, magnetism, and optics by introducing the displacement current term into Ampere's "
        "circuital law, ensuring charge continuity for time-varying electric fields. In vacuum, the coupled curl "
        "equations yield transverse electromagnetic waves propagating at phase velocity c = 1 / sqrt(mu_0 * epsilon_0), "
        "with mutually orthogonal electric and magnetic field vectors."
    ),
    (
        "Virtual memory subsystems decouple contiguous process address spaces from fragmented physical DRAM frames "
        "using hierarchical page tables cached inside translation lookaside buffers. When a virtual page reference "
        "misses in the TLB, hardware page-table walkers traverse multi-level radix trees in physical memory, amortizing "
        "translation overhead through spatial locality and huge-page mappings."
    ),
    (
        "Static single assignment form transforms imperative control-flow graphs so every variable is assigned at "
        "exactly one lexical definition site, inserting phi-functions at dominance frontiers where distinct control "
        "paths converge. This representation simplifies constant propagation, dead-code elimination, global value "
        "numbering, and graph-coloring register allocation inside optimizing compilers."
    ),
    (
        "Balanced search trees and B-trees maintain logarithmic lookup, insertion, and deletion latencies by bounding "
        "tree height relative to fanout. On block-oriented NVMe solid-state drives, wide B+tree internal nodes pack "
        "hundreds of separator keys into cache-aligned pages, minimizing random read amplification while linking leaf "
        "pages sequentially for high-throughput range scans."
    ),
]

REPLAY_PROSE_PASSAGES = [
    (
        "Cache coherence protocols in multicore shared-memory architectures maintain single-writer, multiple-reader "
        "invariants across private L1 data caches. Directory-based MESI implementations track sharing vectors per cache "
        "line so invalidation messages target only cores holding stale copies rather than saturating the interconnect "
        "with broadcast snoops. Memory barriers enforce sequential ordering across relaxed store buffers."
    ),
    (
        "Gradient-based optimization of deep neural networks benefits substantially from preconditioning updates by "
        "exponential moving averages of squared per-parameter gradients. Decoupling weight decay from the adaptive "
        "second-moment denominator ensures uniform regularization strength across embedding matrices and normalization "
        "scales when combined with cosine learning-rate annealing."
    ),
    (
        "Relational database engines achieve atomicity and durability through write-ahead logging: physiological redo "
        "and undo records must be flushed to non-volatile storage before dirty buffer-pool pages are evicted or client "
        "transactions receive commit acknowledgments. Multi-version concurrency control lets read-only snapshots "
        "proceed without blocking concurrent writers."
    ),
    (
        "Fourier analysis decomposes square-integrable periodic signals into orthogonal complex exponentials whose "
        "amplitudes reveal spectral energy distribution across frequency bands. By exploiting symmetry and radix-2 "
        "divide-and-conquer recursion, the fast Fourier transform reduces discrete transform complexity from quadratic "
        "O(N^2) operations to O(N log N) butterfly stages."
    ),
    (
        "Operating system kernels isolate untrusted user processes via hardware privilege rings, system-call traps, "
        "and per-process file descriptor tables. Asynchronous completion queues such as io_uring eliminate repeated "
        "kernel-user context switches during high-IOPS network and storage workloads by sharing lock-free ring buffers "
        "in pinned memory."
    ),
    (
        "In linear algebra, the singular value decomposition factors any real or complex rectangular matrix into two "
        "orthogonal basis rotations surrounding a non-negative diagonal spectrum. Truncating the smallest singular "
        "values yields the optimal low-rank Frobenius and spectral norm approximation by the Eckart-Young theorem."
    ),
]


# ==============================================================================
# 2. TOKENIZER-INDEPENDENT BITS-PER-BYTE (BPB) EVALUATOR
# ==============================================================================

@torch.no_grad()
def evaluate_bpb(model: nn.Module, tok, device: str = "cuda:0") -> Dict:
    model.eval()
    total_nll_nats = 0.0
    total_bytes = 0
    total_pred_tokens = 0

    for text in EVAL_BPB_PASSAGES:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            continue
        # Predicted text corresponds to ids[1:]
        first_tok_bytes = len(tok.decode([ids[0]], skip_special_tokens=False).encode("utf-8"))
        full_bytes = len(text.encode("utf-8"))
        pred_bytes = max(1, full_bytes - first_tok_bytes)

        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_tail_logits(model, x, tail_len=None)[0].float()
            nll = F.cross_entropy(logits, y, reduction="sum").item()
        total_nll_nats += nll
        total_bytes += pred_bytes
        total_pred_tokens += len(ids) - 1

    bpb = (total_nll_nats / math.log(2.0)) / max(1, total_bytes)
    tok_ppl = math.exp(min(20.0, total_nll_nats / max(1, total_pred_tokens)))
    return {
        "bpb": round(bpb, 4),
        "native_tok_ppl": round(tok_ppl, 2),
        "evaluated_bytes": total_bytes,
        "evaluated_tokens": total_pred_tokens,
    }


# ==============================================================================
# 3. EXACT DETERMINISTIC HELD-OUT NIAH MANIFEST BUILDER (MATCHES VAULT v2.2)
# ==============================================================================

def build_deterministic_niah_trial(
    gap_mode: str, n_sections: int, tr: int, seed_base: int
) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Bit-for-bit identical manifest generator to `train_vault_v2_2.py:run_controlled_and_32k_extrapolation_eval`.
    Returns `(manifest_body, all_entries, two_targets)`.
    """
    rng = random.Random(seed_base + tr * 101)
    entries: List[Tuple[str, str]] = []
    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    used_codes = set()
    for idx in range(1, n_sections + 1):
        stem = UNSEEN_NIAH_PREFIXES[(idx + tr) % len(UNSEEN_NIAH_PREFIXES)]
        ent = (
            f"{stem}-{chr(65 + (idx % 24))}{((idx + tr) % 9) + 1}{idx:02d}"
            if n_sections > 24
            else f"{stem}-{chr(65 + (idx % 24))}{((idx + tr) % 9) + 1}"
        )
        while True:
            code = f"CHK-{rng.randint(1000, 9999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        entries.append((ent, code))
        if gap_mode == "compact_3tok":
            hdr = f"[Subsystem {idx:03d}] {ent}: {code} (NOMINAL)"
        elif gap_mode == "wide_24tok":
            hdr = (
                f"[Subsystem {idx:03d}: {ent}] Operational State: NOMINAL | Sensor Temp: 298K "
                f"| Link Lane: {idx % 8} | Verification Code: {code}"
            )
        else:
            hdr = f"[Subsystem {idx:03d}: {ent}] Status: NOMINAL | Verification Code: {code}"
        blocks.append(f"{hdr}\n  Log: {UNSEEN_FILLER} Packet frame {1000 + idx} committed without retransmission.")

    i1 = max(1, int(0.20 * n_sections))
    i2 = min(n_sections - 2, int(0.75 * n_sections))
    targets = [entries[i1], entries[i2]]
    return "\n".join(blocks), entries, targets


@torch.no_grad()
def greedy_decode_tokens(
    model: nn.Module,
    tok,
    prefix_ids: List[int],
    max_new_tokens: int,
    device: str,
    stop_ids: Tuple[int, ...] = (),
    min_chars_stop: int = 0,
    stop_after_lines: int = 0,
) -> List[int]:
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_tail_logits(model, ids, tail_len=1)[0, -1].float()
        nxt = int(torch.argmax(logits).item())
        if nxt in stop_ids:
            break
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        if min_chars_stop > 0 or stop_after_lines > 0:
            txt = tok.decode(generated, skip_special_tokens=True)
            if min_chars_stop > 0 and (len(txt.strip()) >= min_chars_stop or "\n" in txt):
                break
            if stop_after_lines > 0 and txt.strip().count("\n") >= stop_after_lines:
                break
    return generated


@torch.no_grad()
def evaluate_niah_condition(
    model: nn.Module,
    tok,
    mode: str,  # "oneshot_base", "adapted_qa", or "chatml_vault"
    gap_mode: str,
    n_sections: int,
    n_trials: int,
    seed_base: int,
    device: str = "cuda:0",
) -> Dict:
    """
    Tokenizer-agnostic NIAH evaluator across any vocabulary (`SmolLM`, `GPT-NeoX`, `Llama-3`):
      - `cue_exact_code_acc`: Greedy generation from `- {ent}: CHK-` matches all 4 target digits (`0.01%` random chance!)
      - `cue_per_digit_acc`:  Character-level `[d0, d1, d2, d3]` exact match (`10.0%` random chance per digit)
      - `full_resp_code_acc`: Full-response greedy generation from prompt start contains `CHK-XXXX` (for `adapted_qa` / `chatml_vault`)
      - `mean_code_nll_bits`: Exact teacher-forced `-log2 P(4-digit code | context, cue)` in bits (`13.29` bits = random uniform)
    """
    model.eval()
    cue_exact_hits = 0
    full_resp_hits = 0
    total_codes = 0
    char_hits = [0, 0, 0, 0]
    total_nll_bits = 0.0
    prompt_lens: List[int] = []
    fwd_ms = 0.0
    peak_mb = 0.0

    im_start_id = tok.convert_tokens_to_ids("<|im_start|>") if mode == "chatml_vault" else -1
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>") if mode == "chatml_vault" else -1
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
    stop_ids = tuple(x for x in (im_end_id, eos_id) if x is not None and x >= 0)

    for tr in range(n_trials):
        manifest_body, entries, targets = build_deterministic_niah_trial(gap_mode, n_sections, tr, seed_base)
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
            cue0_str = f"- {targets[0][0]}: CHK-"
            cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
            ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        elif mode == "adapted_qa":
            prompt_str = manifest_body + q_str + "\nAnswer:\n"
            base_p_ids = tok.encode(prompt_str, add_special_tokens=False)
            cue0_str = f"- {targets[0][0]}: CHK-"
            cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
            ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        elif mode == "oneshot_base":
            # Steelman 1-shot prefix completion using `entries[0]` (strictly non-overlapping with `targets`!)
            ex_ent, ex_code = entries[0]
            prompt_str = manifest_body + q_str + f"\nVerification Summary:\n- {ex_ent}: {ex_code}\n"
            base_p_ids = tok.encode(prompt_str, add_special_tokens=False)
            cue0_str = f"- {targets[0][0]}: CHK-"
            cue1_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: CHK-"
            ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
        else:
            raise ValueError(mode)

        prompt_lens.append(len(base_p_ids))
        ans_ids = tok.encode(ans_str, add_special_tokens=False)
        full_ids = torch.tensor([base_p_ids + ans_ids], dtype=torch.long, device=device)

        # 1. Single teacher-forced forward pass for timing, VRAM, and exact NLL bits
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tail_logits = forward_tail_logits(model, full_ids, tail_len=len(ans_ids) + 1)[0].float()
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        del full_ids

        # Compute exact NLL in bits over the 4 digits of each target
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

        # 2. Tokenizer-independent Cue Greedy 4-Digit Decode (`- {ent}: CHK-` -> 4 chars, stops as soon as 4 chars emitted!)
        for idx_t, (_, code) in enumerate(targets):
            true_digits = code.split("-")[1]
            cue_s = cue0_str if idx_t == 0 else cue1_str
            pref = base_p_ids + tok.encode(cue_s, add_special_tokens=False)
            gen_ids = greedy_decode_tokens(model, tok, pref, max_new_tokens=4, device=device, stop_ids=stop_ids, min_chars_stop=4)
            gen_txt = tok.decode(gen_ids, skip_special_tokens=True)
            m = re.match(r"\s*([0-9]{1,4})", gen_txt)
            pred_4 = m.group(1) if m else gen_txt.strip()[:4]
            pred_4 = pred_4.ljust(4, "?")[:4]
            for d_i in range(4):
                if pred_4[d_i] == true_digits[d_i]:
                    char_hits[d_i] += 1
            if pred_4 == true_digits:
                cue_exact_hits += 1
            total_codes += 1

        # 3. Full-response free greedy generation from prompt start up to 8K (at 16K/32K, cue_exact_hits already tests unforced 4-digit recall)
        if mode in ("chatml_vault", "adapted_qa") and n_sections <= 96:
            gen_full_ids = greedy_decode_tokens(
                model, tok, base_p_ids, max_new_tokens=48, device=device, stop_ids=stop_ids, stop_after_lines=0
            )
            gen_full_txt = tok.decode(gen_full_ids, skip_special_tokens=True)
            for _, code in targets:
                if code in gen_full_txt:
                    full_resp_hits += 1

    if mode == "oneshot_base" or n_sections > 96:
        full_resp_hits = cue_exact_hits

    per_digit = [round(char_hits[i] / max(1, total_codes), 4) for i in range(4)]
    mean_digit = round(sum(char_hits) / max(1, 4 * total_codes), 4)
    return {
        "mode": mode,
        "gap_mode": gap_mode,
        "n_sections": n_sections,
        "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
        "cue_exact_hits": f"{cue_exact_hits}/{total_codes}",
        "cue_exact_acc": round(cue_exact_hits / max(1, total_codes), 4),
        "full_resp_hits": f"{full_resp_hits}/{total_codes}",
        "full_resp_acc": round(full_resp_hits / max(1, total_codes), 4),
        "digit_acc": mean_digit,
        "per_digit_acc": per_digit,
        "d0_acc": per_digit[0],
        "mean_code_nll_bits": round(total_nll_bits / max(1, total_codes), 3),
        "last_fwd_ms": round(fwd_ms, 2),
        "peak_infer_vram_mb": round(peak_mb, 1),
    }


def run_niah_suite_for_model(
    model: nn.Module,
    tok,
    mode: str,
    label: str,
    device: str = "cuda:0",
    extrapolation_lengths: Tuple[Tuple[str, int, int], ...] = (
        ("2K", 22, 6),
        ("4K", 46, 6),
        ("8K", 96, 4),
        ("16K", 196, 4),
        ("32K", 396, 3),
    ),
) -> Dict:
    print(f"\n--- [{label}] Running Controlled 2K NIAH (A/B/C) & 2K->32K Suite ({mode}) ---", flush=True)
    controlled = {}
    for cname, gmode, sbase in [
        ("Test_B_unseen_3tok_compact", "compact_3tok", 80001),
        ("Test_A_unseen_15tok_boilerplate", "boilerplate_15tok", 70001),
        ("Test_C_unseen_24tok_wide", "wide_24tok", 90001),
    ]:
        res = evaluate_niah_condition(model, tok, mode, gmode, n_sections=18, n_trials=8, seed_base=sbase, device=device)
        controlled[cname] = res
        print(
            f"  [{cname:32s}] cue_exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
            f"full_resp={res['full_resp_hits']} ({res['full_resp_acc']*100:5.1f}%) | "
            f"digits={res['per_digit_acc']} (mean={res['digit_acc']*100:5.1f}%) | NLL={res['mean_code_nll_bits']:.2f}b",
            flush=True,
        )

    extrapolation = {}
    for slabel, n_sec, n_tr in extrapolation_lengths:
        try:
            res = evaluate_niah_condition(
                model, tok, mode, "boilerplate_15tok", n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec, device=device
            )
            extrapolation[slabel] = res
            print(
                f"  [Extrap {slabel:>3s} (~{res['mean_prompt_tokens']:5.0f}t)] "
                f"cue_exact={res['cue_exact_hits']} ({res['cue_exact_acc']*100:5.1f}%) | "
                f"full_resp={res['full_resp_hits']} ({res['full_resp_acc']*100:5.1f}%) | "
                f"digits={res['per_digit_acc']} (mean={res['digit_acc']*100:5.1f}%) | NLL={res['mean_code_nll_bits']:.2f}b | "
                f"{res['last_fwd_ms']:.0f}ms | {res['peak_infer_vram_mb']:.0f}MB",
                flush=True,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            extrapolation[slabel] = {"status": "OOM"}
            print(f"  [Extrap {slabel:>3s}] CUDA OOM on 16 GB GPU!", flush=True)

    return {"controlled_2k": controlled, "extrapolation_2k_to_32k": extrapolation}


# ==============================================================================
# 4. TRACK B: APPLES-TO-APPLES MATCHED L=2048 MULTI-GAP NIAH FINE-TUNING RACE
# ==============================================================================

def sample_clean_train_niah_example(rng: random.Random, stem_pool: List[str], tok, max_len: int = 1984) -> Tuple[List[int], int]:
    """
    Synthesizes a strictly holdout-disjoint L<=2048 open-vocabulary multi-gap NIAH training sequence.
    Returns `(full_token_ids, num_answer_tokens)` so loss is applied strictly on answer tokens (identical to VAULT v2.2!).
    """
    n_sections = rng.randint(14, 21)
    gap_choices = ["compact_3tok", "boilerplate_15tok", "gap_19", "wide_24tok", "gap_32"]
    mixed = rng.random() < 0.55
    uniform_gap = rng.choice(gap_choices)

    entries: List[Tuple[str, str]] = []
    used_ents, used_codes = set(), set()
    multi_fam = rng.random() < 0.40
    fams = [f"{rng.choice(stem_pool)}{rng.choice(stem_pool)}" for _ in range(4)] if multi_fam else []

    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    for idx in range(1, n_sections + 1):
        while True:
            if multi_fam:
                base = fams[idx % len(fams)]
            else:
                base = f"{rng.choice(stem_pool)}{rng.choice(stem_pool)}" if rng.random() < 0.65 else rng.choice(stem_pool)
            suf = f"{chr(65 + rng.randint(0, 23))}{rng.randint(1, 9)}"
            if rng.random() < 0.35:
                suf += f"{rng.randint(1, 99):02d}"
            ent = f"{base}-{suf}"
            el = ent.lower()
            if (
                ent not in used_ents
                and all(f not in el for f in FORBIDDEN_NIAH_SUBWORDS)
                and all(h.lower() not in el for h in HOLDOUT_NIAH_PREFIXES)
                and all(u.lower() not in el for u in UNSEEN_NIAH_PREFIXES)
                and all(hs.lower() not in el for hs in HOLDOUT_SUBSTRINGS)
            ):
                used_ents.add(ent)
                break
        while True:
            code = f"CHK-{rng.randint(1000, 9999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        entries.append((ent, code))

        g_mode = rng.choice(gap_choices) if mixed else uniform_gap
        status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE", "LOCKED"])
        if g_mode == "compact_3tok":
            hdr = f"[Subsystem {idx:03d}] {ent}: {code} ({status})"
        elif g_mode == "gap_19":
            hdr = f"[Subsystem {idx:03d}: {ent}] Health: {status} | Bus Channel: {rng.randint(0, 15):02d} | Verification Code: {code}"
        elif g_mode == "wide_24tok":
            hdr = (
                f"[Subsystem {idx:03d}: {ent}] Operational State: {status} | Sensor Temp: {rng.randint(285, 330)}K "
                f"| Link Lane: {idx % 8} | Verification Code: {code}"
            )
        elif g_mode == "gap_32":
            hdr = (
                f"[Subsystem {idx:03d}: {ent}] State: {status} | Envelope Temp: {rng.randint(285, 330)}K "
                f"| Roundtrip Latency: {rng.randint(12, 90)}us | Redundancy Tier: PRIMARY | Verification Code: {code}"
            )
        else:
            hdr = f"[Subsystem {idx:03d}: {ent}] Status: {status} | Verification Code: {code}"
        f1, f2 = rng.sample(FILLER_SENTENCES_V2_1, k=2)
        blocks.append(f"{hdr}\n  Log: {f1} {f2}")

    q_idx = sorted(rng.sample(range(n_sections), k=2))
    t0_ent, t0_code = entries[q_idx[0]]
    t1_ent, t1_code = entries[q_idx[1]]
    prompt_str = (
        "\n".join(blocks)
        + f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
        f"for {t0_ent} and {t1_ent}? List each subsystem and its exact code.\nAnswer:\n"
    )
    ans_str = f"- {t0_ent}: {t0_code}\n- {t1_ent}: {t1_code}"
    if tok.eos_token is not None:
        ans_str += tok.eos_token

    p_ids = tok.encode(prompt_str, add_special_tokens=False)
    a_ids = tok.encode(ans_str, add_special_tokens=False)
    if len(p_ids) + len(a_ids) > max_len:
        # Trim earliest filler lines if needed? Never trim target sections: keep prompt within max_len by picking shorter n_sections
        return sample_clean_train_niah_example(rng, stem_pool, tok, max_len=max_len)
    return p_ids + a_ids, len(a_ids)


def finetune_baseline_matched_2k(
    model: nn.Module,
    tok,
    stem_pool: List[str],
    arch_label: str,
    n_steps: int = 100,
    lr_peak: float = 1.0e-4,
    lr_min: float = 1.5e-5,
    device: str = "cuda:0",
    seed: int = 20261101,
) -> Dict:
    """
    Matched L <= 2,048 Multi-Gap Open-Vocabulary NIAH + Educational Prose Replay fine-tuning race:
      - Strictly L <= 2,048 (zero exposure to > 2,048 context during adaptation, identical to VAULT v2.2!)
      - Strictly holdout-disjoint entity stems (zero overlap with `UNSEEN_NIAH_PREFIXES`)
      - 3 retrieval microbatches (B=2) + 1 educational prose replay microbatch (B=2) per optimizer step
    """
    print(f"\n>>> [Track B] Fine-Tuning {arch_label} strictly at L<=2048 ({n_steps} steps, B=2x4=8 seqs/step)...", flush=True)
    rng = random.Random(seed)
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    decay_p, nodecay_p = [], []
    seen = set()
    for pname, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "embedding" not in pname:
            decay_p.append(p)
        else:
            nodecay_p.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": 0.01}, {"params": nodecay_p, "weight_decay": 0.0}],
        lr=lr_peak,
        betas=(0.9, 0.95),
    )

    model.train()
    loss_curve = []
    t_start = time.perf_counter()
    warmup = max(8, n_steps // 10)

    for step in range(1, n_steps + 1):
        if step <= warmup:
            lr = lr_peak * (step / warmup)
        else:
            prog = (step - warmup) / max(1, n_steps - warmup)
            lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        niah_ce_sum = 0.0

        # 3 NIAH microbatches (right-padded to exact multiples of 64 so RDNA4 hipBLASlt tiles are 100% aligned!)
        for _ in range(3):
            s1, a1 = sample_clean_train_niah_example(rng, stem_pool, tok, max_len=1984)
            s2, a2 = sample_clean_train_niah_example(rng, stem_pool, tok, max_len=1984)
            raw_L = max(len(s1) - 1, len(s2) - 1)
            L_seq = min(2048, ((raw_L + 63) // 64) * 64)
            p1_len = len(s1) - a1
            p2_len = len(s2) - a2
            s1_in = (s1[:-1] + [pad_id] * L_seq)[:L_seq]
            s2_in = (s2[:-1] + [pad_id] * L_seq)[:L_seq]
            x_np = [s1_in, s2_in]
            y1 = [-100] * L_seq
            y2 = [-100] * L_seq
            for j in range(p1_len - 1, min(len(s1) - 1, L_seq)):
                y1[j] = s1[j + 1]
            for j in range(p2_len - 1, min(len(s2) - 1, L_seq)):
                y2[j] = s2[j + 1]

            min_ans_start = min(p1_len - 1, p2_len - 1)
            raw_tail = L_seq - min_ans_start
            tail_len = min(L_seq, ((raw_tail + 63) // 64) * 64)
            x_t = torch.tensor(x_np, dtype=torch.long, device=device)
            y_tail = torch.tensor([y1[-tail_len:], y2[-tail_len:]], dtype=torch.long, device=device)

            use_amp = model.__class__.__name__ != "Mamba3MIMOForCausalLM"
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_tail_logits(model, x_t, tail_len=tail_len).float()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_tail.reshape(-1), ignore_index=-100)
            (loss * (0.80 / 3.0)).backward()
            niah_ce_sum += loss.item()

        # 1 Educational Prose Replay microbatch (64-aligned length preserves general English BPB)
        rp_texts = [
            " ".join(rng.sample(REPLAY_PROSE_PASSAGES, k=6)),
            " ".join(rng.sample(REPLAY_PROSE_PASSAGES, k=6)),
        ]
        rp_ids = [tok.encode(t, add_special_tokens=False)[:513] for t in rp_texts]
        rp_min = min(len(rp_ids[0]) - 1, len(rp_ids[1]) - 1)
        rp_seq = max(64, min(512, (rp_min // 64) * 64))
        rx_0 = (rp_ids[0][:rp_seq] + [pad_id] * rp_seq)[:rp_seq]
        rx_1 = (rp_ids[1][:rp_seq] + [pad_id] * rp_seq)[:rp_seq]
        ry_0 = (rp_ids[0][1 : rp_seq + 1] + [-100] * rp_seq)[:rp_seq]
        ry_1 = (rp_ids[1][1 : rp_seq + 1] + [-100] * rp_seq)[:rp_seq]
        rx = torch.tensor([rx_0, rx_1], dtype=torch.long, device=device)
        ry = torch.tensor([ry_0, ry_1], dtype=torch.long, device=device)
        use_amp = model.__class__.__name__ != "Mamba3MIMOForCausalLM"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            r_logits = forward_tail_logits(model, rx, tail_len=None).float()
            r_loss = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), ry.reshape(-1), ignore_index=-100)
        (r_loss * 0.20).backward()

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()

        mean_niah_ce = niah_ce_sum / 3.0
        if step in (1, 10, 25, 50, 75, n_steps):
            elapsed = time.perf_counter() - t_start
            rec = {
                "step": step,
                "niah_ce": round(mean_niah_ce, 4),
                "replay_ce": round(r_loss.item(), 4),
                "grad_norm": round(gnorm, 3),
                "elapsed_s": round(elapsed, 1),
            }
            loss_curve.append(rec)
            print(
                f"    [Step {step:3d}/{n_steps}] niah_CE={mean_niah_ce:.4f} | replay_CE={r_loss.item():.3f} | "
                f"gnorm={gnorm:.2f} | lr={lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )

    del opt
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    return {
        "steps": n_steps,
        "seqs_per_step": 8,
        "max_train_seq_len": 2048,
        "total_time_s": round(time.perf_counter() - t_start, 1),
        "loss_curve": loss_curve,
    }


# ==============================================================================
# 5. TRACK C: HARDWARE SYSTEMS SCALING (2K -> 4K -> 8K -> 16K -> 32K + 128K)
# ==============================================================================

@torch.no_grad()
def benchmark_hardware_scaling(
    model: nn.Module,
    arch_key: str,
    device: str = "cuda:0",
    seq_lens: Tuple[int, ...] = (2048, 4096, 8192, 16384, 32768),
) -> Dict:
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    weights_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    scaling = {}
    for L in seq_lens:
        label = f"{L // 1024}K"
        gc.collect()
        torch.cuda.empty_cache()
        try:
            x = torch.randint(16, 4096, (1, L), dtype=torch.long, device=device)
            # Warmup
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = forward_tail_logits(model, x, tail_len=1)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            n_reps = 3 if L <= 8192 else 2
            times_ms = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = forward_tail_logits(model, x, tail_len=1)
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            times_ms.sort()
            med_ms = times_ms[len(times_ms) // 2]
            peak_tot_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_act_mb = max(0.0, peak_tot_mb - weights_mb)
            tok_s = L / (med_ms / 1000.0)
            scaling[label] = {
                "seq_len": L,
                "prefill_ms": round(med_ms, 2),
                "tok_per_s": round(tok_s, 1),
                "peak_total_vram_mb": round(peak_tot_mb, 1),
                "peak_activation_vram_mb": round(peak_act_mb, 1),
                "persistent_decode_cache_mb": persistent_decode_state_mb(arch_key, L),
            }
            del x
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            scaling[label] = {
                "seq_len": L,
                "status": "OOM",
                "persistent_decode_cache_mb": persistent_decode_state_mb(arch_key, L),
            }

    decode_state_curve_mb = {
        f"{L // 1024}K": persistent_decode_state_mb(arch_key, L)
        for L in (2048, 8192, 32768, 131072)
    }
    return {
        "resident_weights_mb": round(weights_mb, 1),
        "prefill_scaling": scaling,
        "persistent_decode_cache_mb_curve": decode_state_curve_mb,
    }


# ==============================================================================
# 6. MAIN ORCHESTRATOR (RUNS ALL 5 ARCHITECTURES SEQUENTIALLY WITH ATOMIC SAVES)
# ==============================================================================

def load_vault_v2_2_180m(device: str = "cuda:0") -> Tuple[VAULTLMv2, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_2_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = VAULTLMv2(
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
        prime_banks=(17, 19, 23, 29),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), tok


def load_vault_v2_3_180m(device: str = "cuda:0") -> Tuple[VAULTLMv2, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_3_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    primes = tuple(ckpt.get("prime_banks", (47, 53, 59, 61)))
    model = VAULTLMv2(
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
        prime_banks=primes,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), tok


def load_vault_v2_4_180m(device: str = "cuda:0") -> Tuple[VAULTLMv2, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    primes = tuple(ckpt.get("prime_banks", (47, 53, 59, 61)))
    model = VAULTLMv2(
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
        prime_banks=primes,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), tok


def atomic_save_json(data: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Track B fine-tuning steps per baseline")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmarks", "vault_vs_mamba123_transformer.json"),
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated subset of model keys to run/update",
    )
    args = parser.parse_args()

    device = "cuda:0"
    smol_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    stem_pool = build_clean_open_vocab_pool(smol_tok)
    print(f"[+] Built holdout-disjoint open-vocabulary stem pool: {len(stem_pool):,} whole-word stems", flush=True)

    results: Dict = {
        "benchmark": "VAULT v2.2/v2.3/v2.4 vs Mamba-1 vs Mamba-2 vs Mamba-3 MIMO vs Transformer (SmolLM-135M)",
        "hardware": torch.cuda.get_device_name(0),
        "rocm_Experimental_FlashSDPA": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0"),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {},
    }
    if os.path.exists(args.out) and args.only:
        with open(args.out, "r") as f:
            results = json.load(f)

    only_set = {x.strip() for x in args.only.split(",") if x.strip()}

    v2_4_ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_best.pt")
    v2_4_tok_b = 0.28
    if os.path.exists(v2_4_ckpt_path):
        try:
            _meta = torch.load(v2_4_ckpt_path, map_location="cpu", weights_only=False)
            v2_4_tok_b = round(0.275 + float(_meta.get("total_tokens_seen", 0)) / 1e9, 3)
            del _meta
        except Exception:
            pass

    specs = [
        {
            "key": "VAULT_v2_2_180M",
            "title": "VAULT v2.2 (180.4M, 88 Cells)",
            "family": "Fast-Weight Coprime Voronoi Vaults (17,19,23,29) + W_delay=64 Two-Level Induction + 64-tok Local RoPE",
            "pretrain_tokens_B": 0.26,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": True,
            "loader": lambda: load_vault_v2_2_180m(device=device),
        },
        {
            "key": "VAULT_v2_3_180M",
            "title": "VAULT v2.3 (180.5M, 220 Cells)",
            "family": "Expanded Coprime Voronoi Vaults (47,53,59,61) + Shared-Group Scan + Continual Prose/ChatML/1-Shot Adaptation",
            "pretrain_tokens_B": 0.28,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": True,
            "loader": lambda: load_vault_v2_3_180m(device=device),
        },
        {
            "key": "VAULT_v2_4_180M",
            "title": "VAULT v2.4 (180.5M, 220 Cells, Extended Continual Pretraining)",
            "family": "Expanded Coprime Voronoi Vaults (47,53,59,61) + Large-Scale Unseen Textbook/STEM Continual Pretraining + ChatML/1-Shot Replay",
            "pretrain_tokens_B": v2_4_tok_b,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": True,
            "loader": lambda: load_vault_v2_4_180m(device=device),
        },
        {
            "key": "Mamba1_130M",
            "title": "Mamba-1 (130.9M)",
            "family": "Selective Diagonal ZOH SSM (d_state=16, 24L, Gu & Dao 2023)",
            "pretrain_tokens_B": 300.0,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": False,
            "loader": lambda: load_mamba1_130m(device=device, grad_checkpoint=True),
        },
        {
            "key": "Mamba2_130M",
            "title": "Mamba-2 (129.7M)",
            "family": "Structured State Space Duality SSD (d_state=128, 24L, Dao & Gu 2024)",
            "pretrain_tokens_B": 300.0,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": False,
            "loader": lambda: load_mamba2_130m(device=device, grad_checkpoint=True, fast_ssd=True),
        },
        {
            "key": "Mamba3_MIMO_187M",
            "title": "Mamba-3 MIMO (187.3M)",
            "family": "Exp-Trapezoidal + Complex Data-Dep 4-Slice RoPE + Rank-4 MIMO SSD (d_state=128, 12L, Dao & Gu 2026)",
            "pretrain_tokens_B": 100.0,
            "trained_seq_len": 2048,
            "complexity": "O(L) Time, O(1) State",
            "is_vault": False,
            "loader": lambda: load_mamba3_mimo_187m(device=device, grad_checkpoint=True),
        },
        {
            "key": "Transformer_SmolLM_135M",
            "title": "Transformer GQA RoPE (134.5M)",
            "family": "Dense Global Softmax Attention + RoPE theta=10000 (30L GQA, SmolLM-135M, FlashAttention SDPA)",
            "pretrain_tokens_B": 600.0,
            "trained_seq_len": 2048,
            "complexity": "O(L^2) Time, O(L) KV-Cache",
            "is_vault": False,
            "loader": lambda: load_transformer_smollm_135m(device=device, grad_checkpoint=True),
        },
    ]

    for sp in specs:
        key = sp["key"]
        if only_set and key not in only_set:
            continue

        print("\n" + "=" * 105, flush=True)
        print(f"BENCHMARKING MODEL: {sp['title']} [{key}]", flush=True)
        print("=" * 105, flush=True)

        gc.collect()
        torch.cuda.empty_cache()
        model, tok = sp["loader"]()
        n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        print(f"[+] Loaded {sp['title']}: {n_params:,} unique parameters ({n_params/1e6:.2f}M)", flush=True)

        # Track C first on clean eval model (so zero optimizer fragmentation)
        print(f"\n>>> [Track C] Measuring Hardware Scaling (2K -> 32K) for {sp['title']}...", flush=True)
        hw = benchmark_hardware_scaling(model, key, device=device)
        for sl, sv in hw["prefill_scaling"].items():
            if sv.get("status") == "OOM":
                print(f"    [{sl:>3s}] OOM | decode_cache={sv['persistent_decode_cache_mb']} MB", flush=True)
            else:
                print(
                    f"    [{sl:>3s}] prefill={sv['prefill_ms']:7.2f} ms ({sv['tok_per_s']:8.1f} tok/s) | "
                    f"act_VRAM={sv['peak_activation_vram_mb']:6.1f} MB | decode_cache={sv['persistent_decode_cache_mb']} MB",
                    flush=True,
                )

        # Track A: Pretrained BPB + 1-Shot Base NIAH (or ChatML + 1-Shot for VAULT v2.2)
        bpb_pre = evaluate_bpb(model, tok, device=device)
        print(f"\n>>> [Track A] Pretrained English BPB = {bpb_pre['bpb']:.4f} (native tok PPL = {bpb_pre['native_tok_ppl']:.2f})", flush=True)

        if sp["is_vault"]:
            # VAULT v2.2 is already adapted strictly at L=2048; evaluate both its native ChatML AND 1-shot prefix!
            track_a_niah = run_niah_suite_for_model(model, tok, mode="oneshot_base", label=f"{sp['title']} [1-Shot Prefix]", device=device)
            track_b_niah = run_niah_suite_for_model(model, tok, mode="chatml_vault", label=f"{sp['title']} [ChatML L=2048 Adapted]", device=device)
            entry = {
                "title": sp["title"],
                "family": sp["family"],
                "params": n_params,
                "params_M": round(n_params / 1e6, 2),
                "pretrain_tokens_B": sp["pretrain_tokens_B"],
                "trained_seq_len": sp["trained_seq_len"],
                "complexity": sp["complexity"],
                "track_a_pretrained": {
                    "english_bpb": bpb_pre,
                    "oneshot_niah": track_a_niah,
                },
                "track_b_matched_2k_adapted": {
                    "adaptation_note": "Trained strictly at L=2048 in VAULT v2.2 continual adaptation + clean SFT",
                    "english_bpb_post": bpb_pre,
                    "niah_eval": track_b_niah,
                },
                "track_c_hardware": hw,
            }
        else:
            track_a_niah = run_niah_suite_for_model(model, tok, mode="oneshot_base", label=f"{sp['title']} [Pretrained 1-Shot]", device=device)
            ft_meta = finetune_baseline_matched_2k(
                model, tok, stem_pool, arch_label=sp["title"], n_steps=args.steps, device=device
            )
            bpb_post = evaluate_bpb(model, tok, device=device)
            print(
                f"    [Post-Adapt English BPB] {bpb_pre['bpb']:.4f} -> {bpb_post['bpb']:.4f} "
                f"(delta={bpb_post['bpb'] - bpb_pre['bpb']:+.4f})",
                flush=True,
            )
            track_b_niah = run_niah_suite_for_model(
                model, tok, mode="adapted_qa", label=f"{sp['title']} [Matched L=2048 Adapted ({args.steps} steps)]", device=device
            )
            entry = {
                "title": sp["title"],
                "family": sp["family"],
                "params": n_params,
                "params_M": round(n_params / 1e6, 2),
                "pretrain_tokens_B": sp["pretrain_tokens_B"],
                "trained_seq_len": sp["trained_seq_len"],
                "complexity": sp["complexity"],
                "track_a_pretrained": {
                    "english_bpb": bpb_pre,
                    "oneshot_niah": track_a_niah,
                },
                "track_b_matched_2k_adapted": {
                    "finetune_telemetry": ft_meta,
                    "english_bpb_post": bpb_post,
                    "niah_eval": track_b_niah,
                },
                "track_c_hardware": hw,
            }

        results["models"][key] = entry
        results["timestamp_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_save_json(results, args.out)
        print(f"[+] Saved incremental benchmark telemetry for {sp['title']} -> {args.out}", flush=True)

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 105, flush=True)
    print(f"ALL 5 ARCHITECTURES BENCHMARKED SUCCESSFULLY -> {args.out}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
