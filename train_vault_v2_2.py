#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.2 (`VAULTLMv2(v2_2_mode=True)` + `FlashVAULT v2.2` Beta-Weighted Consensus & Two-Level Induction):
# Continual Adaptation + Clean ChatML SFT Forked Directly from `vault_v2_pretrain_best.pt`
# (Zero Restart from Step 0) + Automatic 8-Domain & 2K->32K Extrapolation Benchmark Suite!

import os
import sys
import json
import math
import time
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vault_v2_model import VAULTLMv2
from train_vault_v2 import (
    SEQ_LEN,
    NVMeMemmapDataset,
    UNSEEN_NIAH_PREFIXES,
    UNSEEN_FILLER,
    build_unseen_2k_niah_eval,
    evaluate_val_loss,
)


SAL_COPY_LEX_KEYWORDS = (
    "salience_binder",
    "v_ind_proj",
    "ind_out_proj",
    "lex_router_g2",
    "lex_router_g3",
    "q_subhead_delta",
    "ind_shared_proj",
)

FROZEN_V2_2_KEYWORDS = (
    "x0_rms_norm",
    "kenyon_proj",
    "apl_norm",
)


def normalize_row_weights(bw: torch.Tensor) -> torch.Tensor:
    """
    Normalizes token salience weights per sequence row so an `sft_mode=True` associative-retrieval
    row (`~30` target tokens) contributes the same `1/B` microbatch gradient weight as a packed
    `1,400`-token code/math/prose row while preserving exact within-row relative token salience.
    """
    row_sums = bw.sum(dim=-1, keepdim=True)
    active_mask = (row_sums > 1e-6).float()
    active_rows = active_mask.sum().clamp(min=1.0)
    return ((bw / row_sums.clamp(min=1e-6)) * active_mask) * (float(bw.size(0)) / active_rows)


class PartitionedMemmapSampler:
    """
    Partitions an `NVMeMemmapDataset` by active target token count `nz = (masks > 0).sum(axis=1)`:
      - `niah_idx`  (`8 <= nz < 160`, single-turn `<|im_start|>` == 2, zero `<file_sep>`): pure NIAH manifests
      - `skill_idx` (`nz >= 160`) : packed polyglot code, `<think>` CoT math, and direct English prose
    Maintains independent shuffled cursors per partition so every microbatch has deterministic stream ratios
    and allows `sal_copy_lex` + `ind_beta_proj` gradients to be isolated strictly to clean retrieval examples.
    """
    def __init__(self, ds: NVMeMemmapDataset, seed: int = 20261031):
        self.ds = ds
        toks = np.asarray(ds.tokens)
        nz = (np.asarray(ds.masks) > 0).sum(axis=1)
        clean_single_turn = ((toks == 1).sum(axis=1) == 2) & ((toks == 5).sum(axis=1) == 0)
        self.niah_idx = np.where((nz >= 8) & (nz < 160) & clean_single_turn)[0]
        self.skill_idx = np.where(nz >= 160)[0]
        if len(self.niah_idx) == 0 or len(self.skill_idx) == 0:
            raise ValueError(
                f"PartitionedMemmapSampler expected non-empty niah & skill splits, got "
                f"niah={len(self.niah_idx)}, skill={len(self.skill_idx)}"
            )
        self.rng = np.random.default_rng(seed)
        self.niah_order = self.rng.permutation(self.niah_idx)
        self.skill_order = self.rng.permutation(self.skill_idx)
        self.niah_cur = 0
        self.skill_cur = 0

    def _fetch(self, idx: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = np.asarray(self.ds.tokens[idx], dtype=np.int64)
        msks = np.asarray(self.ds.masks[idx], dtype=np.float32) * 0.1
        y_np = rows[:, 1:].copy()
        y_np[msks <= 0.0] = -100
        x = torch.from_numpy(rows[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)
        w = torch.from_numpy(msks).to(device, non_blocking=True)
        return x, y, w

    def next_niah_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.niah_cur + batch_size > len(self.niah_order):
            self.niah_order = self.rng.permutation(self.niah_idx)
            self.niah_cur = 0
        idx = self.niah_order[self.niah_cur : self.niah_cur + batch_size]
        self.niah_cur += batch_size
        return self._fetch(idx, device)

    def next_skill_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.skill_cur + batch_size > len(self.skill_order):
            self.skill_order = self.rng.permutation(self.skill_idx)
            self.skill_cur = 0
        idx = self.skill_order[self.skill_cur : self.skill_cur + batch_size]
        self.skill_cur += batch_size
        return self._fetch(idx, device)

    def fixed_partition_eval_batches(
        self, n_batches: int, batch_size: int, device: str
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        skill_batches = []
        niah_batches = []
        lim_s = min(len(self.skill_idx), n_batches * batch_size)
        for st in range(0, lim_s, batch_size):
            skill_batches.append(self._fetch(self.skill_idx[st : st + batch_size], device))
        lim_n = min(len(self.niah_idx), n_batches * batch_size)
        for st in range(0, lim_n, batch_size):
            niah_batches.append(self._fetch(self.niah_idx[st : st + batch_size], device))
        return skill_batches, niah_batches


@torch.no_grad()
def evaluate_val_loss_fast(
    model: VAULTLMv2, batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    tot_w_loss = 0.0
    tot_raw_ce = 0.0
    for vx, vy, vw in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model._encode_hidden(vx, step_ratio=1.0).reshape(-1, model.d_model)
            logits = model.lm_head(h).float()
            t_flat = vy.reshape(-1)
            w_flat = vw.reshape(-1)
            active_t = torch.where(w_flat > 0.0, t_flat, torch.full_like(t_flat, -100))
            ce_tok = F.cross_entropy(logits, active_t, ignore_index=-100, reduction="none")
            w_loss = (ce_tok * w_flat).sum() / w_flat.sum().clamp(min=1.0)
            raw_ce = ce_tok.sum() / (w_flat > 0.0).float().sum().clamp(min=1.0)
        tot_w_loss += w_loss.item()
        tot_raw_ce += raw_ce.item()
    if was_training:
        model.train()
    n = max(1, len(batches))
    return tot_w_loss / n, tot_raw_ce / n

@torch.no_grad()
def generate_chatml_v2_2(
    model: VAULTLMv2,
    tok,
    user_prompt: str,
    max_new_tokens: int = 220,
    device: str = "cuda:0",
    temperature: float = 0.08,
    max_ctx: int = 36864,
) -> str:
    was_training = model.training
    model.eval()

    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")

    p_ids = (
        [im_start_id]
        + tok.encode(f"user\n{user_prompt.strip()}", add_special_tokens=False)
        + [im_end_id]
        + tok.encode("\n", add_special_tokens=False)
        + [im_start_id]
        + tok.encode("assistant\n", add_special_tokens=False)
    )
    ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    prompt_len = ids.size(1)

    for _ in range(max_new_tokens):
        ctx = ids[:, -max_ctx:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            last_h = model._encode_hidden(ctx, step_ratio=1.0)[:, -1:, :]
            logits = model.lm_head(last_h)[:, -1, :].float()

        if temperature > 1e-4:
            for prev_id in set(ids[0, prompt_len:].tolist()):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= 1.04
                else:
                    logits[0, prev_id] *= 1.04

        if temperature <= 1e-4:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            top_v, top_i = torch.topk(logits / temperature, k=6, dim=-1)
            probs = F.softmax(top_v, dim=-1)
            next_tok = top_i.gather(-1, torch.multinomial(probs, 1))

        tok_val = next_tok.item()
        if tok_val in (im_end_id, eos_id):
            break
        ids = torch.cat([ids, next_tok], dim=1)

    if was_training:
        model.train()
    return tok.decode(ids[0, prompt_len:].tolist(), skip_special_tokens=False).strip()


def run_8domain_eval_v2_2(
    model: VAULTLMv2,
    tok,
    stage_label: str,
    device: str,
    out_path: str,
) -> Dict:
    niah_prompt, niah_targets = build_unseen_2k_niah_eval()
    suite = [
        (
            "1. BuildBit (.bb) GPU SoA",
            "Write a BuildBit (.bb) function that defines `soa struct Photon` with `f32` fields `x`, `vx`, `ax`, allocates 4096 elements in a 16 MB `gpu_arena`, and updates positions in parallel.",
        ),
        (
            "2. C99 Systems Code",
            "Write a C99 function to reverse an array of 32-bit integers in place.",
        ),
        (
            "3. C++20 Generic Template",
            "Write a C++20 generic class template `FixedStack<T, 128>` with `push`, `pop`, and `size` methods.",
        ),
        (
            "4. Python Algorithm",
            "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
        ),
        (
            "5. Lua Metatable Module",
            "Write a Lua module implementing a `Particle` class using metatables (`__index` and `:update(dt)`).",
        ),
        (
            "6. Chain-of-Thought Math (<think>)",
            "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?",
        ),
        (
            "7. Unseen CS Prose Explanation",
            "Explain why recursion needs a base case in programming.",
        ),
        (
            f"8. 2K +15-Tok-Gap NIAH ({niah_targets})",
            niah_prompt,
        ),
    ]

    print("\n" + "=" * 100, flush=True)
    print(f"VAULT v2.2 8-DOMAIN UNSEEN EVALUATION [{stage_label}]", flush=True)
    print("=" * 100, flush=True)

    records = []
    for dom, prm in suite:
        temp = 0.0 if ("NIAH" in dom or "Math" in dom) else 0.05
        out = generate_chatml_v2_2(model, tok, prm, max_new_tokens=240, device=device, temperature=temp)
        records.append({"domain": dom, "prompt_tail": prm[-300:], "output": out})
        print(f"  [{dom}]\n    -> {out.replace(chr(10), ' ⏎ ')[:280]}", flush=True)

    niah_out = records[-1]["output"]
    niah_hits = sum(1 for _, code in niah_targets if code in niah_out)
    prose_out = records[6]["output"]
    prose_clean = ("<think>" not in prose_out) and ("```" not in prose_out) and ("given code" not in prose_out.lower())
    math_hit = "180" in records[5]["output"]
    bb_photon_exact = "soa struct Photon" in records[0]["output"]
    cpp_fixedstack_exact = "FixedStack" in records[2]["output"]
    lua_particle_exact = "Particle" in records[4]["output"]

    summary = {
        "stage": stage_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "niah_expected": niah_targets,
        "niah_exact_hits": f"{niah_hits}/{len(niah_targets)}",
        "math_180_exact": math_hit,
        "prose_clean_direct_english": prose_clean,
        "identifier_fidelity": {
            "bb_Photon": bb_photon_exact,
            "cpp_FixedStack": cpp_fixedstack_exact,
            "lua_Particle": lua_particle_exact,
        },
        "evaluations": records,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[+] Summary: NIAH={niah_hits}/{len(niah_targets)} | Math(180)={math_hit} | DirectProse={prose_clean} "
        f"| IDs(Photon={bb_photon_exact}, FixedStack={cpp_fixedstack_exact}, Particle={lua_particle_exact}) -> {out_path}\n",
        flush=True,
    )
    return summary


@torch.no_grad()
def run_controlled_and_32k_extrapolation_eval(
    model: VAULTLMv2,
    tok,
    device: str,
    out_path: str,
) -> Dict:
    """
    Runs:
      1. Controlled Open-Vocabulary NIAH Tests A / B / C (`15-tok`, `3-tok`, `24-tok` gaps on unseen prefixes)
         with full per-digit `(d0, d1, d2, d3)` teacher-forced accuracy AND greedy exact 4-digit recall.
      2. Zero-Shot Length Extrapolation & Peak VRAM Curve across L in [2048, 4096, 8192, 16384, 32768]
         (trained strictly at L = 2048!).
    """
    was_training = model.training
    model.eval()

    print("\n" + "=" * 100, flush=True)
    print("VAULT v2.2 CONTROLLED NIAH (A/B/C) & 2K -> 32K ZERO-SHOT EXTRAPOLATION SUITE", flush=True)
    print("=" * 100, flush=True)

    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def _eval_manifest_condition(gap_mode: str, n_sections: int, n_trials: int, seed_base: int) -> Dict:
        exact_codes = 0
        total_codes = 0
        pos_hits = [0, 0, 0, 0]
        pos_tot = [0, 0, 0, 0]
        prompt_lens = []
        fwd_ms = 0.0
        peak_mb = 0.0

        for tr in range(n_trials):
            rng = random.Random(seed_base + tr * 101)
            entries = []
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
            prm = (
                "\n".join(blocks)
                + f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
                f"for {targets[0][0]} and {targets[1][0]}? List each subsystem and its exact code."
            )

            p_ids = (
                [im_start_id]
                + tok.encode(f"user\n{prm.strip()}", add_special_tokens=False)
                + [im_end_id]
                + tok.encode("\n", add_special_tokens=False)
                + [im_start_id]
                + tok.encode("assistant\n", add_special_tokens=False)
            )
            prompt_lens.append(len(p_ids))
            ans_str = f"- {targets[0][0]}: {targets[0][1]}\n- {targets[1][0]}: {targets[1][1]}"
            a_ids = tok.encode(ans_str, add_special_tokens=False)
            full_ids = torch.tensor([p_ids + a_ids], dtype=torch.long, device=device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_all = model._encode_hidden(full_ids, step_ratio=1.0)
                ans_logits = model.lm_head(h_all[0, len(p_ids) - 1 :]).float()
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            del h_all, full_ids

            for idx_t, (ent, code) in enumerate(targets):
                if idx_t == 0:
                    pref_str = f"- {ent}: CHK-"
                else:
                    pref_str = f"- {targets[0][0]}: {targets[0][1]}\n- {ent}: CHK-"
                pref_len = len(tok.encode(pref_str, add_special_tokens=False))
                dig_ids = tok.encode(code.split("-")[1], add_special_tokens=False)
                for off_d, did in enumerate(dig_ids):
                    rel_pos = pref_len + off_d
                    pred_id = int(torch.argmax(ans_logits[rel_pos]).item())
                    di = min(off_d, 3)
                    if pred_id == did:
                        pos_hits[di] += 1
                    pos_tot[di] += 1

            gen_out = generate_chatml_v2_2(
                model, tok, prm, max_new_tokens=56, device=device, temperature=0.0, max_ctx=36864
            )
            for _, code in targets:
                if code in gen_out:
                    exact_codes += 1
                total_codes += 1

        per_pos = [round(pos_hits[i] / max(1, pos_tot[i]), 4) for i in range(4)]
        return {
            "gap_mode": gap_mode,
            "n_sections": n_sections,
            "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
            "exact_code_hits": f"{exact_codes}/{total_codes}",
            "exact_code_acc": round(exact_codes / max(1, total_codes), 4),
            "tf_digit_top1_acc": round(sum(pos_hits) / max(1, sum(pos_tot)), 4),
            "per_digit_pos_acc": per_pos,
            "d0_acc": per_pos[0],
            "last_fwd_ms": round(fwd_ms, 2),
            "peak_infer_vram_mb": round(peak_mb, 1),
        }

    controlled = {
        "Test_A_unseen_15tok_boilerplate": _eval_manifest_condition("boilerplate_15tok", n_sections=18, n_trials=8, seed_base=70001),
        "Test_B_unseen_3tok_compact": _eval_manifest_condition("compact_3tok", n_sections=18, n_trials=8, seed_base=80001),
        "Test_C_unseen_24tok_wide": _eval_manifest_condition("wide_24tok", n_sections=18, n_trials=8, seed_base=90001),
    }
    for k, v in controlled.items():
        print(
            f"  [{k}] exact={v['exact_code_hits']} ({v['exact_code_acc']*100:.1f}%) | "
            f"digits={v['per_digit_pos_acc']} (mean={v['tf_digit_top1_acc']*100:.1f}%)",
            flush=True,
        )

    extrapolation = {}
    for label, n_sec, n_tr in [
        ("2K", 22, 6),
        ("4K", 46, 6),
        ("8K", 96, 4),
        ("16K", 196, 4),
        ("32K", 396, 3),
    ]:
        res = _eval_manifest_condition("boilerplate_15tok", n_sections=n_sec, n_trials=n_tr, seed_base=110000 + n_sec)
        extrapolation[label] = res
        print(
            f"  [Extrapolation {label:>3s} (~{res['mean_prompt_tokens']:.0f} tok)] "
            f"exact={res['exact_code_hits']} ({res['exact_code_acc']*100:.1f}%) | "
            f"digits={res['per_digit_pos_acc']} (mean={res['tf_digit_top1_acc']*100:.1f}%) | "
            f"fwd={res['last_fwd_ms']:.1f} ms | infer_VRAM={res['peak_infer_vram_mb']:.1f} MB",
            flush=True,
        )

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trained_seq_len": 2048,
        "controlled_unseen_niah_2k": controlled,
        "zero_shot_extrapolation_2k_to_32k": extrapolation,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if was_training:
        model.train()
    return out


def build_optimizer_v2_2(
    model: VAULTLMv2,
    base_lr: float,
    sal_mult: float = 0.025,
    beta_mult: float = 4.0,
    read_mult: float = 3.5,
    weight_decay: float = 0.05,
) -> Tuple[torch.optim.AdamW, Dict[str, List[torch.nn.Parameter]]]:
    sal_copy_lex = []
    beta_params = []
    read_params = []
    decay_params = []
    nodecay_params = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if any(k in name for k in SAL_COPY_LEX_KEYWORDS):
            sal_copy_lex.append(p)
        elif "ind_beta_proj" in name:
            beta_params.append(p)
        elif "ind_read_proj" in name:
            read_params.append(p)
        elif p.ndim >= 2 and p.numel() >= model.d_model and "conv" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": sal_copy_lex, "lr": base_lr * sal_mult, "lr_mult": sal_mult, "weight_decay": 0.0},
        {"params": beta_params, "lr": base_lr * beta_mult, "lr_mult": beta_mult, "weight_decay": 0.01},
        {"params": read_params, "lr": base_lr * read_mult, "lr_mult": read_mult, "weight_decay": 0.01},
        {"params": decay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    clip_buckets = {
        "sal": sal_copy_lex,
        "beta": beta_params,
        "read": read_params,
        "gates": beta_params + read_params,
        "base": decay_params + nodecay_params,
    }
    return opt, clip_buckets


def set_group_lrs(opt: torch.optim.AdamW, base_lr: float):
    for pg in opt.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return peak_lr * (step / float(max(1, warmup_steps)))
    prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def clip_v2_2_param_groups(clip_buckets: Dict[str, List[torch.nn.Parameter]], max_norm: float) -> Tuple[float, float, float]:
    gn_sal = float(torch.nn.utils.clip_grad_norm_(clip_buckets["sal"], 0.15).item())
    gn_vind = float(torch.nn.utils.clip_grad_norm_(clip_buckets["gates"], max_norm).item())
    gn_base = float(torch.nn.utils.clip_grad_norm_(clip_buckets["base"], max_norm).item())
    return gn_sal, gn_vind, gn_base


def main():
    parser = argparse.ArgumentParser(description="Train VAULT v2.2 forked from Stage-1 VAULT v2 checkpoint")
    parser.add_argument("--adapt_steps", type=int, default=600, help="Phase 1 continual adaptation steps (default 600 = 9.83M tok)")
    parser.add_argument("--sft_steps", type=int, default=300, help="Phase 2 clean ChatML SFT steps (default 300 = 4.92M tok)")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--ckpt_interval", type=int, default=8)
    parser.add_argument("--adapt_lr", type=float, default=1.2e-4)
    parser.add_argument("--sft_lr", type=float, default=4.0e-5)
    parser.add_argument("--router_aux_coef", type=float, default=0.0)
    parser.add_argument("--beta_hinge_coef", type=float, default=0.0)
    parser.add_argument("--beta_hinge_target", type=float, default=0.015)
    parser.add_argument("--ent_div_coef", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--probe_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=300)
    parser.add_argument("--save_interval", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20261031)
    np.random.seed(20261031)
    random.seed(20261031)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_v2_dir = os.path.join(base_dir, "data_v2")
    data_v2_2_dir = os.path.join(base_dir, "data_v2_2")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)

    model = VAULTLMv2(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint="half",
        v2_2_mode=True,
        ckpt_interval=args.ckpt_interval,
    ).to(device)

    base_fork_ckpt = os.path.join(ckpt_dir, "vault_v2_pretrain_best.pt")
    latest_v2_2_ckpt = os.path.join(ckpt_dir, "vault_v2_2_latest.pt")
    adapt_best_ckpt = os.path.join(ckpt_dir, "vault_v2_2_adapt_best.pt")
    best_v2_2_ckpt = os.path.join(ckpt_dir, "vault_v2_2_best.pt")
    telemetry_path = os.path.join(bench_dir, "vault_v2_2_telemetry.json")

    if args.eval_only:
        ck = torch.load(best_v2_2_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        run_8domain_eval_v2_2(model, tok, "VAULT_v2_2_EvalOnly", device, os.path.join(bench_dir, "eval_vault_v2_2_final.json"))
        run_controlled_and_32k_extrapolation_eval(model, tok, device, os.path.join(bench_dir, "eval_vault_v2_2_extrapolation_32k.json"))
        return

    ck_base = torch.load(base_fork_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck_base["model_state"], strict=False)
    assert len(unexpected) == 0, f"Unexpected keys when forking from Stage-1: {unexpected}"

    # Freeze Stage-1 Co-Prime Kenyon Voronoi cell router & `x0_rms_norm` coordinate basis
    for n, p in model.named_parameters():
        if any(k in n for k in FROZEN_V2_2_KEYWORDS):
            p.requires_grad_(False)

    stage1_replay_ds = NVMeMemmapDataset(data_v2_dir, "train", seed=20261031)
    stage1_val_ds = NVMeMemmapDataset(data_v2_dir, "val", seed=9001)
    adapt_train_ds = NVMeMemmapDataset(data_v2_2_dir, "adapt_train", seed=3101)
    adapt_val_ds = NVMeMemmapDataset(data_v2_2_dir, "adapt_val", seed=3102)
    sft_train_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_train", seed=4101)
    sft_val_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_val", seed=4102)

    adapt_sampler = PartitionedMemmapSampler(adapt_train_ds, seed=3101)
    adapt_val_sampler = PartitionedMemmapSampler(adapt_val_ds, seed=3102)
    sft_sampler = PartitionedMemmapSampler(sft_train_ds, seed=4101)
    sft_val_sampler = PartitionedMemmapSampler(sft_val_ds, seed=4102)

    val_general_batches = stage1_val_ds.fixed_eval_batches(n_batches=8, batch_size=2, device=device)
    val_adapt_skill_batches, val_adapt_niah_batches = adapt_val_sampler.fixed_partition_eval_batches(
        n_batches=8, batch_size=2, device=device
    )
    val_sft_skill_batches, val_sft_niah_batches = sft_val_sampler.fixed_partition_eval_batches(
        n_batches=8, batch_size=2, device=device
    )
    unseen_niah_prompt, unseen_niah_targets = build_unseen_2k_niah_eval()

    tok_per_step = args.micro_batch * args.grad_accum * SEQ_LEN
    print("=" * 100, flush=True)
    print("VAULT v2.2: VORONOI-DECOUPLED INDUCTION + WINDOWED CMS + CLEAN CHATML SFT + 32K ENGINE", flush=True)
    print("=" * 100, flush=True)
    print(f"  Forked Base Model   : {base_fork_ckpt} (Step {ck_base.get('step')}, 245.76M tokens, Val CE {ck_base.get('val_ce'):.4f})", flush=True)
    print(f"  v2.2 Parameters     : {model.count_parameters():,} (Missing/New v2.2 tensors: {len(missing)}, Isotropic SVD copy hook applied!)", flush=True)
    print(f"  v2.2 Architecture   : Voronoi-Decoupled RFFT Top-1 + Windowed Co-Prime CMS(503,509,521) + 33x Novelty-Beta + Unclamped SVD Readout", flush=True)
    print(f"  Adapt Partitions    : skill={len(adapt_sampler.skill_idx):,} rows | clean_niah={len(adapt_sampler.niah_idx):,} rows (25% Replay / 25% Skill / 50% NIAH)", flush=True)
    print(f"  SFT Partitions      : skill={len(sft_sampler.skill_idx):,} rows | clean_niah={len(sft_sampler.niah_idx):,} rows (25% Replay / 50% Skill / 25% NIAH)", flush=True)
    print(f"  Phase 1 Adaptation  : {args.adapt_steps:,} steps ({args.adapt_steps * tok_per_step / 1e6:.2f}M tok @ L={SEQ_LEN})", flush=True)
    print(f"  Phase 2 Clean SFT   : {args.sft_steps:,} steps ({args.sft_steps * tok_per_step / 1e6:.2f}M tok @ L={SEQ_LEN})", flush=True)
    print("=" * 100, flush=True)

    _, step0_gen_ce = evaluate_val_loss_fast(model, val_general_batches)
    _, step0_niah_ce = evaluate_val_loss_fast(model, val_adapt_niah_batches)
    print(
        f"[*] Verified Step-0 Holdout General Val CE = {step0_gen_ce:.4f} (PPL = {math.exp(step0_gen_ce):.2f}) "
        f"| Step-0 Clean NIAH Val CE = {step0_niah_ce:.4f}",
        flush=True,
    )

    telemetry = {
        "model": "VAULTLMv2.2",
        "forked_from": base_fork_ckpt,
        "params": model.count_parameters(),
        "ckpt_interval": args.ckpt_interval,
        "step0_general_val_ce": round(step0_gen_ce, 4),
        "step0_general_val_ppl": round(math.exp(step0_gen_ce), 2),
        "step0_niah_val_ce": round(step0_niah_ce, 4),
        "adapt_history": [],
        "sft_history": [],
        "milestone_evals": [],
    }

    # ==========================================================================
    # PHASE 1: CONTINUAL v2.2 ADAPTATION (600 STEPS = 9.83M TOKENS)
    # ==========================================================================
    if args.adapt_steps > 0:
        opt1, clip1 = build_optimizer_v2_2(
            model, base_lr=args.adapt_lr, sal_mult=0.025, beta_mult=4.0, read_mult=5.0, weight_decay=0.05
        )
        # Isolate `sal_copy_lex` + `ind_beta_proj` strictly to clean NIAH; `ind_read_proj` trains jointly on Replay+Skill+NIAH!
        ind_isolated_1 = clip1["sal"] + clip1["beta"]
        n_niah_micro_1 = max(1, args.grad_accum // 2)  # 2 of 4 microbatches = 50% NIAH in Phase 1
        first_niah_idx_1 = args.grad_accum - n_niah_micro_1
        start_step = 0
        best_adapt_score = float("inf")

        if args.resume and os.path.exists(latest_v2_2_ckpt):
            ck = torch.load(latest_v2_2_ckpt, map_location=device, weights_only=False)
            if ck.get("stage") == "adapt":
                model.load_state_dict(ck["model_state"])
                opt1.load_state_dict(ck["optimizer_state"])
                start_step = ck.get("step", 0)
                best_adapt_score = ck.get("best_adapt_score", float("inf"))
                print(f"[*] Resumed Phase 1 Adaptation at step {start_step}", flush=True)

        model.train()
        t0 = time.perf_counter()
        toks_run = 0
        warmup_1 = max(20, int(0.05 * args.adapt_steps))
        use_ediv = args.ent_div_coef > 0.0
        use_raux = args.router_aux_coef > 0.0

        for step in range(start_step + 1, args.adapt_steps + 1):
            base_lr = cosine_lr(step, args.adapt_steps, warmup_1, args.adapt_lr, 2.5e-5)
            set_group_lrs(opt1, base_lr)

            opt1.zero_grad(set_to_none=True)
            ce_accum = 0.0
            niah_ce_sum = 0.0
            beta_niah_sum = 0.0
            ediv_niah_sum = 0.0
            aux_val = 0.0
            for micro_i in range(args.grad_accum):
                is_niah_micro = (micro_i >= first_niah_idx_1)
                if micro_i == 0:
                    bx, by, bw = stage1_replay_ds.next_batch(args.micro_batch, device)
                elif is_niah_micro:
                    if micro_i == first_niah_idx_1:
                        for p in ind_isolated_1:
                            p.grad = None
                    bx, by, bw = adapt_sampler.next_niah_batch(args.micro_batch, device)
                else:
                    bx, by, bw = adapt_sampler.next_skill_batch(args.micro_batch, device)

                bw_row = normalize_row_weights(bw)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if is_niah_micro:
                        if use_ediv:
                            ce_loss, beta_mean, ent_div = model.forward_loss(
                                bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=True
                            )
                        else:
                            ce_loss, beta_mean = model.forward_loss(
                                bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=False
                            )
                            ent_div = torch.zeros_like(beta_mean)
                        beta_hinge = args.beta_hinge_coef * F.relu(beta_mean - args.beta_hinge_target)
                        loss = (ce_loss + beta_hinge + args.ent_div_coef * ent_div) / args.grad_accum
                    else:
                        ce_loss = model.forward_loss(bx, by, loss_weights=bw_row, step_ratio=1.0)
                        if micro_i == 0 and use_raux:
                            r_aux = model.compute_router_aux_loss()
                            aux_val = r_aux.item()
                            loss = (ce_loss / args.grad_accum) + args.router_aux_coef * r_aux
                        else:
                            loss = ce_loss / args.grad_accum
                loss.backward()

                if is_niah_micro:
                    niah_ce_sum += ce_loss.item() / float(n_niah_micro_1)
                    beta_niah_sum += beta_mean.item() / float(n_niah_micro_1)
                    ediv_niah_sum += ent_div.item() / float(n_niah_micro_1)
                    if micro_i == args.grad_accum - 1:
                        scale_ind = float(args.grad_accum) / float(n_niah_micro_1)
                        for p in ind_isolated_1:
                            if p.grad is not None:
                                p.grad.mul_(scale_ind)

                ce_accum += ce_loss.item() / args.grad_accum
                toks_run += bx.numel()

            niah_ce_last = niah_ce_sum
            beta_accum = beta_niah_sum
            ediv_accum = ediv_niah_sum
            if not math.isfinite(ce_accum):
                raise RuntimeError(f"Non-finite Phase 1 loss {ce_accum} at step {step}")

            gn_sal, gn_vind, gn_base = clip_v2_2_param_groups(clip1, 1.0)
            opt1.step()

            if step == 1 or step % args.log_interval == 0 or step == args.adapt_steps:
                _, gen_ce = evaluate_val_loss_fast(model, val_general_batches)
                _, adp_skill_ce = evaluate_val_loss_fast(model, val_adapt_skill_batches)
                adp_niah_wce, adp_niah_ce = evaluate_val_loss_fast(model, val_adapt_niah_batches)
                adp_score = 0.5 * adp_skill_ce + 0.5 * adp_niah_wce
                gen_ppl = math.exp(min(gen_ce, 15.0))
                adp_ppl = math.exp(min(adp_skill_ce, 15.0))
                dt = time.perf_counter() - t0
                tps = toks_run / max(dt, 1e-6)

                print(
                    f"[Adapt {step:4d}/{args.adapt_steps}] tr={ce_accum:.3f}(n={niah_ce_last:.3f}) | "
                    f"gen_PPL={gen_ppl:5.2f} | sk_PPL={adp_ppl:5.2f} | niah_CE={adp_niah_ce:.3f} | "
                    f"beta={beta_accum:.4f} | ediv={ediv_accum:.3f} | gn(s={gn_sal:.2f},v={gn_vind:.2f},b={gn_base:.2f}) | "
                    f"lr={base_lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )

                rec = {
                    "step": step,
                    "train_loss": round(ce_accum, 4),
                    "train_niah_wce": round(niah_ce_last, 4),
                    "beta_ind_mean": round(beta_accum, 5),
                    "ent_div_mean": round(ediv_accum, 4),
                    "router_aux": round(aux_val, 5),
                    "general_val_ce": round(gen_ce, 4),
                    "general_val_ppl": round(gen_ppl, 2),
                    "adapt_skill_val_ce": round(adp_skill_ce, 4),
                    "adapt_skill_val_ppl": round(adp_ppl, 2),
                    "adapt_niah_val_ce": round(adp_niah_ce, 4),
                    "adapt_niah_val_wce": round(adp_niah_wce, 4),
                    "adapt_composite_score": round(adp_score, 4),
                    "gn_sal": round(gn_sal, 3),
                    "gn_vind": round(gn_vind, 3),
                    "gn_base": round(gn_base, 3),
                    "lr": base_lr,
                    "tok_s": round(tps, 1),
                }

                if step % args.probe_interval == 0:
                    p_math = generate_chatml_v2_2(
                        model, tok, "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?",
                        max_new_tokens=110, device=device, temperature=0.0
                    )
                    p_niah = generate_chatml_v2_2(
                        model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0
                    )
                    niah_hits = sum(1 for _, code in unseen_niah_targets if code in p_niah)
                    rec["probe_math"] = p_math
                    rec["probe_niah"] = p_niah
                    rec["probe_niah_hits"] = f"{niah_hits}/{len(unseen_niah_targets)}"
                    print(f"   [Probe Math] {p_math.replace(chr(10), ' ⏎ ')[:180]}", flush=True)
                    print(f"   [Probe 2K NIAH ({niah_hits}/{len(unseen_niah_targets)}) {unseen_niah_targets}] {p_niah.replace(chr(10), ' ⏎ ')[:180]}", flush=True)

                telemetry["adapt_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if adp_score < best_adapt_score and step >= min(50, args.adapt_steps):
                    best_adapt_score = adp_score
                    torch.save(
                        {
                            "stage": "adapt_v2_2",
                            "step": step,
                            "adapt_composite_score": adp_score,
                            "adapt_skill_val_ce": adp_skill_ce,
                            "adapt_niah_val_ce": adp_niah_ce,
                            "gen_val_ce": gen_ce,
                            "model_state": model.state_dict(),
                        },
                        adapt_best_ckpt,
                    )

            if step % args.save_interval == 0 or step == args.adapt_steps:
                torch.save(
                    {
                        "stage": "adapt",
                        "step": step,
                        "best_adapt_score": best_adapt_score,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt1.state_dict(),
                    },
                    latest_v2_2_ckpt,
                )

            if step % args.eval_interval == 0 or step == args.adapt_steps:
                tag = f"v2_2_adapt_step_{step}"
                ev = run_8domain_eval_v2_2(model, tok, tag, device, os.path.join(bench_dir, f"eval_vault_{tag}.json"))
                telemetry["milestone_evals"].append({
                    "stage": tag,
                    "niah_exact_hits": ev["niah_exact_hits"],
                    "math_180_exact": ev["math_180_exact"],
                    "prose_clean": ev["prose_clean_direct_english"],
                    "identifier_fidelity": ev["identifier_fidelity"],
                })
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

    # ==========================================================================
    # PHASE 2: CLEAN CHATML SFT POLISH (300 STEPS = 4.92M TOKENS)
    # ==========================================================================
    if args.sft_steps > 0:
        if os.path.exists(adapt_best_ckpt):
            ck_adp = torch.load(adapt_best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck_adp["model_state"])
            print(
                f"\n[*] Reloaded best Phase-1 Adaptation checkpoint (step {ck_adp.get('step')}, "
                f"score={ck_adp.get('adapt_composite_score'):.4f}, niah_CE={ck_adp.get('adapt_niah_val_ce'):.4f}) for Phase 2 SFT!",
                flush=True,
            )

        print("\n>>> STARTING PHASE 2: VAULT v2.2 CLEAN CHATML SFT POLISH (25% REPLAY + 50% SKILL + 25% ISOLATED-(SAL+BETA) NIAH)...", flush=True)
        opt2, clip2 = build_optimizer_v2_2(
            model, base_lr=args.sft_lr, sal_mult=0.025, beta_mult=4.0, read_mult=5.0, weight_decay=0.02
        )
        ind_isolated_2 = clip2["sal"] + clip2["beta"]
        best_sft_score = float("inf")
        model.train()
        t_sft = time.perf_counter()
        toks_sft = 0
        warmup_2 = max(15, int(0.06 * args.sft_steps))
        use_ediv_2 = args.ent_div_coef > 0.0
        use_raux_2 = args.router_aux_coef > 0.0

        for step in range(1, args.sft_steps + 1):
            base_lr = cosine_lr(step, args.sft_steps, warmup_2, args.sft_lr, 5.0e-6)
            set_group_lrs(opt2, base_lr)

            opt2.zero_grad(set_to_none=True)
            ce_accum = 0.0
            niah_ce_last = 0.0
            beta_accum = 0.0
            ediv_accum = 0.0
            for micro_i in range(args.grad_accum):
                is_niah_micro = (micro_i == args.grad_accum - 1)
                if micro_i == 0:
                    bx, by, bw = stage1_replay_ds.next_batch(args.micro_batch, device)
                elif is_niah_micro:
                    for p in ind_isolated_2:
                        p.grad = None
                    bx, by, bw = sft_sampler.next_niah_batch(args.micro_batch, device)
                else:
                    bx, by, bw = sft_sampler.next_skill_batch(args.micro_batch, device)

                bw_row = normalize_row_weights(bw)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if is_niah_micro:
                        if use_ediv_2:
                            ce_loss, beta_mean, ent_div = model.forward_loss(
                                bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=True
                            )
                        else:
                            ce_loss, beta_mean = model.forward_loss(
                                bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=False
                            )
                            ent_div = torch.zeros_like(beta_mean)
                        beta_hinge = args.beta_hinge_coef * F.relu(beta_mean - args.beta_hinge_target)
                        loss = (ce_loss + beta_hinge + args.ent_div_coef * ent_div) / args.grad_accum
                    else:
                        ce_loss = model.forward_loss(bx, by, loss_weights=bw_row, step_ratio=1.0)
                        if micro_i == 0 and use_raux_2:
                            loss = (ce_loss / args.grad_accum) + (0.5 * args.router_aux_coef) * model.compute_router_aux_loss()
                        else:
                            loss = ce_loss / args.grad_accum
                loss.backward()

                if is_niah_micro:
                    niah_ce_last = ce_loss.item()
                    beta_accum = beta_mean.item()
                    ediv_accum = ent_div.item()
                    for p in ind_isolated_2:
                        if p.grad is not None:
                            p.grad.mul_(float(args.grad_accum))

                ce_accum += ce_loss.item() / args.grad_accum
                toks_sft += bx.numel()

            if not math.isfinite(ce_accum):
                raise RuntimeError(f"Non-finite Phase 2 SFT loss {ce_accum} at step {step}")

            gn_sal, gn_vind, gn_base = clip_v2_2_param_groups(clip2, 0.8)
            opt2.step()

            if step == 1 or step % args.log_interval == 0 or step == args.sft_steps:
                _, gen_ce = evaluate_val_loss_fast(model, val_general_batches)
                _, sft_skill_ce = evaluate_val_loss_fast(model, val_sft_skill_batches)
                sft_niah_wce, sft_niah_ce = evaluate_val_loss_fast(model, val_sft_niah_batches)
                sft_score = 0.5 * sft_skill_ce + 0.5 * sft_niah_wce
                gen_ppl = math.exp(min(gen_ce, 15.0))
                sft_ppl = math.exp(min(sft_skill_ce, 15.0))
                dt = time.perf_counter() - t_sft
                tps = toks_sft / max(dt, 1e-6)

                print(
                    f"[SFT {step:4d}/{args.sft_steps}] tr={ce_accum:.3f}(n={niah_ce_last:.3f}) | "
                    f"gen_PPL={gen_ppl:5.2f} | sft_PPL={sft_ppl:5.2f} | niah_CE={sft_niah_ce:.3f} | "
                    f"beta={beta_accum:.3f} | ediv={ediv_accum:.3f} | gn(s={gn_sal:.2f},v={gn_vind:.2f},b={gn_base:.2f}) | "
                    f"lr={base_lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )

                rec = {
                    "step": step,
                    "train_loss": round(ce_accum, 4),
                    "train_niah_wce": round(niah_ce_last, 4),
                    "beta_ind_mean": round(beta_accum, 4),
                    "ent_div_mean": round(ediv_accum, 4),
                    "general_val_ce": round(gen_ce, 4),
                    "general_val_ppl": round(gen_ppl, 2),
                    "sft_skill_val_ce": round(sft_skill_ce, 4),
                    "sft_skill_val_ppl": round(sft_ppl, 2),
                    "sft_niah_val_ce": round(sft_niah_ce, 4),
                    "sft_niah_val_wce": round(sft_niah_wce, 4),
                    "sft_composite_score": round(sft_score, 4),
                    "gn_sal": round(gn_sal, 3),
                    "gn_vind": round(gn_vind, 3),
                    "gn_base": round(gn_base, 3),
                    "lr": base_lr,
                    "tok_s": round(tps, 1),
                }

                if step % args.probe_interval == 0:
                    p_niah = generate_chatml_v2_2(
                        model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0
                    )
                    niah_hits = sum(1 for _, code in unseen_niah_targets if code in p_niah)
                    rec["probe_niah"] = p_niah
                    rec["probe_niah_hits"] = f"{niah_hits}/{len(unseen_niah_targets)}"
                    print(f"   [SFT Probe 2K NIAH ({niah_hits}/{len(unseen_niah_targets)}) {unseen_niah_targets}] {p_niah.replace(chr(10), ' ⏎ ')[:200]}", flush=True)

                telemetry["sft_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if sft_score < best_sft_score:
                    best_sft_score = sft_score
                    torch.save(
                        {
                            "stage": "sft_v2_2",
                            "step": step,
                            "sft_composite_score": sft_score,
                            "sft_val_ce": sft_skill_ce,
                            "sft_niah_val_ce": sft_niah_ce,
                            "gen_val_ce": gen_ce,
                            "model_state": model.state_dict(),
                        },
                        best_v2_2_ckpt,
                    )

        if os.path.exists(best_v2_2_ckpt):
            ck_best = torch.load(best_v2_2_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck_best["model_state"])
            print(
                f"\n[*] Reloaded best VAULT v2.2 checkpoint (SFT step {ck_best.get('step')}, "
                f"score={ck_best.get('sft_composite_score'):.4f}, niah_CE={ck_best.get('sft_niah_val_ce'):.4f})",
                flush=True,
            )

        ev_final = run_8domain_eval_v2_2(
            model, tok, "VAULT_v2_2_Final_Best", device, os.path.join(bench_dir, "eval_vault_v2_2_final.json")
        )
        extrap = run_controlled_and_32k_extrapolation_eval(
            model, tok, device, os.path.join(bench_dir, "eval_vault_v2_2_extrapolation_32k.json")
        )
        telemetry["final_8domain_summary"] = {
            "niah_exact_hits": ev_final["niah_exact_hits"],
            "math_180_exact": ev_final["math_180_exact"],
            "prose_clean": ev_final["prose_clean_direct_english"],
            "identifier_fidelity": ev_final["identifier_fidelity"],
        }
        telemetry["final_extrapolation_summary"] = extrap
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=2)

        print(f"ALL VAULT v2.2 TRAINING & 32K EVALUATION COMPLETE! Best Checkpoint -> {best_v2_2_ckpt}", flush=True)


if __name__ == "__main__":
    main()
