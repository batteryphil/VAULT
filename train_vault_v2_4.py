#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.4 (`180.5M`, 220 Co-Prime Voronoi Cells `P = (47, 53, 59, 61)`):
# Extended Large-Scale Continual Pretraining (`500M-616M+` Unseen Decontaminated Textbook & STEM Tokens)
# Forked Read-Only from `checkpoints/vault_v2_3_best.pt`.
#
# Architecture & Curriculum Design:
#   - 8-Microbatch Step (`B=2, grad_accum=8` = `32,768 tok/step` @ `L=2048`):
#       * 6/8 (`75.0%`) Unseen Textbook & STEM Prose (`data_v2_4/prose_train_tokens.bin`, ~614M unseen tokens)
#       * 1/8 (`12.5%`) 8-Domain Clean ChatML Skills (`data_v2_2/sft_train`)
#       * 1/8 (`12.5%`) Wide-Diversity Dual-Format Open-Vocab NIAH (`data_v2_4/niah_train`: 55% ChatML / 45% 1-Shot)
#   - Strict Induction-Write Gradient Isolation (`SAL_COPY_LEX_KEYWORDS + ind_beta_proj` receive gradients
#     exclusively from the NIAH microbatch; `x0_rms_norm` frozen to preserve the SVD token-copy cone).
#   - Multi-Cycle Cosine Annealing (`--cycle_steps 400` = `13.11M tok/cycle`): every ~80 min cycle cools down
#     smoothly to `min_lr`, guaranteeing a sharp, benchmark-ready `checkpoints/vault_v2_4_best.pt` at every cycle boundary.

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_vault_v2_data import ROW_LEN
from src.vault_v2_model import VAULTLMv2
from train_vault_v2 import SEQ_LEN, NVMeMemmapDataset, build_unseen_2k_niah_eval
from train_vault_v2_2 import (
    SAL_COPY_LEX_KEYWORDS,
    PartitionedMemmapSampler,
    evaluate_val_loss_fast,
    generate_chatml_v2_2,
    normalize_row_weights,
    run_8domain_eval_v2_2,
    run_controlled_and_32k_extrapolation_eval,
    set_group_lrs,
)
from benchmarks.bench_vault_vs_mamba123_transformer import evaluate_bpb, evaluate_niah_condition


class ProseOnlyMemmapDataset:
    """
    Zero-overhead NVMe uint16 token stream reader (`(N, 2049)`) without redundant `mask.bin`,
    supporting deterministic epoch-seeded shuffling and exact cursor/epoch state serialization for `--resume`.
    """

    def __init__(self, data_dir: str, split_prefix: str, seed: int = 20261102):
        tok_path = os.path.join(data_dir, f"{split_prefix}_tokens.bin")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Missing memmap file: {tok_path}")
        file_bytes = os.path.getsize(tok_path)
        row_bytes = ROW_LEN * 2
        if file_bytes % row_bytes != 0 or file_bytes == 0:
            raise ValueError(f"Invalid prose memmap size {file_bytes} bytes for {tok_path}")
        self.n_rows = file_bytes // row_bytes
        self.tokens = np.memmap(tok_path, dtype=np.uint16, mode="r", shape=(self.n_rows, ROW_LEN))
        self.base_seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = self._make_order(self.epoch)

    def _make_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.base_seed + int(epoch) * 10007)
        return rng.permutation(self.n_rows)

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": int(self.epoch), "cursor": int(self.cursor)}

    def load_state_dict(self, st: Optional[Dict[str, int]]) -> None:
        if not st:
            return
        self.epoch = int(st.get("epoch", 0))
        self.order = self._make_order(self.epoch)
        self.cursor = min(max(0, int(st.get("cursor", 0))), self.n_rows)

    def next_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + batch_size > self.n_rows:
            self.epoch += 1
            self.order = self._make_order(self.epoch)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        rows = np.asarray(self.tokens[idx], dtype=np.int64)
        bx = torch.from_numpy(rows[:, :-1]).to(device, non_blocking=True)
        by = torch.from_numpy(rows[:, 1:]).to(device, non_blocking=True)
        return bx, by

    def fixed_eval_batches(self, n_batches: int, batch_size: int, device: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        lim = min(self.n_rows, n_batches * batch_size)
        for st in range(0, lim, batch_size):
            rows = np.asarray(self.tokens[st : st + batch_size], dtype=np.int64)
            bx = torch.from_numpy(rows[:, :-1]).to(device)
            by = torch.from_numpy(rows[:, 1:]).to(device)
            batches.append((bx, by))
        return batches


@torch.no_grad()
def evaluate_prose_val_fast(model: VAULTLMv2, batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    was_training = model.training
    model.eval()
    tot_ce = 0.0
    for vx_cpu, vy_cpu in batches:
        vx = vx_cpu.to(device, non_blocking=True)
        vy = vy_cpu.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce = model.forward_loss(vx, vy, loss_weights=None, step_ratio=1.0)
        tot_ce += float(ce.item())
    if was_training:
        model.train()
    return tot_ce / max(1, len(batches))


@torch.no_grad()
def evaluate_masked_val_chunked(
    model: VAULTLMv2, batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device: str
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    tot_w_loss = 0.0
    tot_raw_ce = 0.0
    for vx_cpu, vy_cpu, vw_cpu in batches:
        vx = vx_cpu.to(device, non_blocking=True)
        vy = vy_cpu.to(device, non_blocking=True)
        vw = vw_cpu.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model._encode_hidden(vx, step_ratio=1.0).reshape(-1, model.d_model)
            t_flat = vy.reshape(-1)
            w_flat = vw.reshape(-1)
            active_t = torch.where(w_flat > 0.0, t_flat, torch.full_like(t_flat, -100))
            w_num = 0.0
            raw_num = 0.0
            for st in range(0, h.size(0), 2048):
                en = min(st + 2048, h.size(0))
                logits_sub = model.lm_head(h[st:en]).float()
                ce_sub = F.cross_entropy(logits_sub, active_t[st:en], ignore_index=-100, reduction="none")
                w_num += float((ce_sub * w_flat[st:en]).sum().item())
                raw_num += float(ce_sub.sum().item())
            w_loss = w_num / max(1.0, float(w_flat.sum().item()))
            raw_ce = raw_num / max(1.0, float((w_flat > 0.0).float().sum().item()))
        tot_w_loss += w_loss
        tot_raw_ce += raw_ce
    if was_training:
        model.train()
    n = max(1, len(batches))
    return tot_w_loss / n, tot_raw_ce / n


def fetch_masked_batch(
    ds: NVMeMemmapDataset, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx, by, bw = ds.next_batch(batch_size, device)
    by = torch.where(bw > 0.0, by, torch.full_like(by, -100))
    return bx, by, bw


def build_optimizer_v2_4(
    model: VAULTLMv2,
    base_lr: float,
    device: str,
    sal_mult: float = 0.06,
    beta_mult: float = 1.5,
    read_mult: float = 1.5,
    router_mult: float = 0.45,
    weight_decay: float = 0.04,
) -> Tuple[torch.optim.AdamW, Dict[str, List[torch.nn.Parameter]]]:
    sal_copy_lex = []
    beta_params = []
    read_params = []
    router_params = []
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
        elif "kenyon_proj" in name or "apl_norm" in name:
            router_params.append(p)
        elif p.ndim >= 2 and p.numel() >= model.d_model and "conv" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": sal_copy_lex, "lr": base_lr * sal_mult, "lr_mult": sal_mult, "weight_decay": 0.0},
        {"params": beta_params, "lr": base_lr * beta_mult, "lr_mult": beta_mult, "weight_decay": 0.01},
        {"params": read_params, "lr": base_lr * read_mult, "lr_mult": read_mult, "weight_decay": 0.01},
        {"params": router_params, "lr": base_lr * router_mult, "lr_mult": router_mult, "weight_decay": 0.0},
        {"params": decay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": 0.0},
    ]
    use_fused = device.startswith("cuda")
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    clip_buckets = {
        "sal": sal_copy_lex,
        "beta": beta_params,
        "gates": beta_params + read_params,
        "router": router_params,
        "base": decay_params + nodecay_params,
    }
    return opt, clip_buckets


def clip_v2_4_param_groups(
    clip_buckets: Dict[str, List[torch.nn.Parameter]], max_norm: float = 1.0
) -> Tuple[float, float, float, float]:
    gn_sal = float(torch.nn.utils.clip_grad_norm_(clip_buckets["sal"], 0.15).item())
    gn_gates = float(torch.nn.utils.clip_grad_norm_(clip_buckets["gates"], max_norm).item())
    gn_router = float(torch.nn.utils.clip_grad_norm_(clip_buckets["router"], 0.30).item())
    gn_base = float(torch.nn.utils.clip_grad_norm_(clip_buckets["base"], max_norm).item())
    return gn_sal, gn_gates, gn_router, gn_base


def multi_cycle_cosine_lr(
    step: int,
    cycle_steps: int,
    peak_lr: float,
    min_lr: float,
    cycle_decay: float = 0.92,
) -> Tuple[float, int, int]:
    """
    Multi-cycle cosine annealing with gentle peak decay:
      - Every `cycle_steps` steps (e.g., 400 steps = 13.11M tokens), LR warms up for 8% of the cycle
        and cosine-anneals cleanly to `min_lr`, crystallizing weights into a sharp, deployable BPB minimum.
    """
    c_idx = (step - 1) // cycle_steps
    s_in = ((step - 1) % cycle_steps) + 1
    c_peak = max(min_lr * 3.0, peak_lr * (cycle_decay ** c_idx))
    c_warm = max(12, int(0.08 * cycle_steps))
    start_lr = min_lr if c_idx > 0 else max(min_lr, 0.15 * c_peak)

    if s_in <= c_warm:
        frac = s_in / float(c_warm)
        lr = start_lr + frac * (c_peak - start_lr)
    else:
        prog = (s_in - c_warm) / float(max(1, cycle_steps - c_warm))
        lr = min_lr + 0.5 * (c_peak - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return lr, c_idx, s_in


def atomic_save_pt(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_save_json(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description="VAULT v2.4 Extended Continual Pretraining (~614M Unseen Tokens) forked from vault_v2_3_best.pt"
    )
    parser.add_argument("--steps", type=int, default=18000, help="Total continual steps (18000 = 589.8M tokens)")
    parser.add_argument("--cycle_steps", type=int, default=400, help="Steps per cosine cooldown cycle (400 = 13.11M tok)")
    parser.add_argument("--cycle_decay", type=float, default=0.92, help="Peak LR decay factor per completed cycle")
    parser.add_argument("--max_cycles_this_run", type=int, default=0, help="Optional clean exit after N cycles (0=all)")
    parser.add_argument("--max_wall_seconds", type=float, default=0.0, help="Optional clean exit after wall seconds (0=all)")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8, help="6 prose + 1 skill + 1 NIAH = 32,768 tok/step")
    parser.add_argument("--grad_checkpoint", type=str, default="half")
    parser.add_argument("--ckpt_interval", type=int, default=1)
    parser.add_argument("--peak_lr", type=float, default=8.0e-5)
    parser.add_argument("--min_lr", type=float, default=6.0e-6)
    parser.add_argument("--router_aux_coef", type=float, default=0.015)
    parser.add_argument("--beta_hinge_coef", type=float, default=0.50)
    parser.add_argument("--beta_hinge_target", type=float, default=0.016)
    parser.add_argument("--ent_div_coef", type=float, default=0.03)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--bpb_interval", type=int, default=50)
    parser.add_argument("--probe_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_final_eval", action="store_true")
    args = parser.parse_args()

    if args.grad_accum < 4:
        raise ValueError("grad_accum must be >= 4 so prose, skill, and isolated NIAH microbatches all execute")

    torch.manual_seed(20261102)
    np.random.seed(20261102)
    random.seed(20261102)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_v2_2_dir = os.path.join(base_dir, "data_v2_2")
    data_v2_4_dir = os.path.join(base_dir, "data_v2_4")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    src_v2_3_ckpt = os.path.join(ckpt_dir, "vault_v2_3_best.pt")
    latest_v2_4_ckpt = os.path.join(ckpt_dir, "vault_v2_4_latest.pt")
    best_v2_4_ckpt = os.path.join(ckpt_dir, "vault_v2_4_best.pt")
    telemetry_path = os.path.join(bench_dir, "vault_v2_4_telemetry.json")

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ck_v2_3 = torch.load(src_v2_3_ckpt, map_location="cpu", weights_only=False)
    prime_banks = tuple(ck_v2_3.get("prime_banks", (47, 53, 59, 61)))

    model = VAULTLMv2(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint=args.grad_checkpoint,
        ckpt_interval=args.ckpt_interval,
        v2_2_mode=True,
        prime_banks=prime_banks,
    ).to(device)

    if args.eval_only:
        target_ck = best_v2_4_ckpt if os.path.exists(best_v2_4_ckpt) else src_v2_3_ckpt
        ck = torch.load(target_ck, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        run_8domain_eval_v2_2(
            model, tok, "VAULT_v2_4_EvalOnly", device, os.path.join(bench_dir, "eval_vault_v2_4_final.json")
        )
        run_controlled_and_32k_extrapolation_eval(
            model, tok, device, os.path.join(bench_dir, "eval_vault_v2_4_extrapolation_32k.json")
        )
        return

    model.load_state_dict(ck_v2_3["model_state"], strict=True)
    del ck_v2_3

    # Freeze `x0_rms_norm` so the isotropic SVD token-copy cone stays anchored across multi-hour pretraining
    for name, p in model.named_parameters():
        if "x0_rms_norm" in name:
            p.requires_grad_(False)

    prose_train_ds = ProseOnlyMemmapDataset(data_v2_4_dir, "prose_train", seed=6101)
    prose_val_ds = ProseOnlyMemmapDataset(data_v2_4_dir, "prose_val", seed=6102)
    niah_train_ds = NVMeMemmapDataset(data_v2_4_dir, "niah_train", seed=6103)
    niah_val_ds = NVMeMemmapDataset(data_v2_4_dir, "niah_val", seed=6104)
    sft_train_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_train", seed=4101)
    sft_val_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_val", seed=4102)

    sft_sampler = PartitionedMemmapSampler(sft_train_ds, seed=4101)
    sft_val_sampler = PartitionedMemmapSampler(sft_val_ds, seed=4102)

    val_prose_batches = prose_val_ds.fixed_eval_batches(n_batches=16, batch_size=2, device="cpu")
    val_niah_batches = niah_val_ds.fixed_eval_batches(n_batches=8, batch_size=2, device="cpu")
    val_skill_batches, _ = sft_val_sampler.fixed_partition_eval_batches(n_batches=8, batch_size=2, device="cpu")
    unseen_niah_prompt, unseen_niah_targets = build_unseen_2k_niah_eval()

    tok_per_step = args.micro_batch * args.grad_accum * SEQ_LEN
    n_prose_micro = args.grad_accum - 2
    n_unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())

    print("=" * 108, flush=True)
    print("VAULT v2.4: LARGE-SCALE UNSEEN TEXTBOOK/STEM CONTINUAL PRETRAINING + MULTI-CYCLE COSINE COOLDOWN", flush=True)
    print("=" * 108, flush=True)
    print(f"  Forked From         : {src_v2_3_ckpt} (Protected, Read-Only | Primes={prime_banks}, {sum(prime_banks)} cells)", flush=True)
    print(
        f"  Unique Parameters   : {n_unique_params:,} ({n_unique_params / 1e6:.2f}M) | "
        f"grad_ckpt={args.grad_checkpoint} | ckpt_interval={args.ckpt_interval}",
        flush=True,
    )
    print(
        f"  Stream Mix ({args.grad_accum}-way) : "
        f"{100.0 * n_prose_micro / args.grad_accum:.1f}% Unseen Prose ({prose_train_ds.n_rows:,} rows = {prose_train_ds.n_rows * SEQ_LEN / 1e6:.1f}M tok) | "
        f"{100.0 / args.grad_accum:.1f}% 8-Domain ChatML ({len(sft_sampler.skill_idx):,} rows) | "
        f"{100.0 / args.grad_accum:.1f}% Open-Vocab NIAH ({len(niah_train_ds.tokens):,} rows)",
        flush=True,
    )
    print(
        f"  Multi-Cycle Cosine  : total={args.steps:,} steps ({args.steps * tok_per_step / 1e6:.1f}M tok), "
        f"cycle={args.cycle_steps} steps ({args.cycle_steps * tok_per_step / 1e6:.2f}M tok/cycle), "
        f"peak_lr={args.peak_lr:.2e} (x{args.cycle_decay:.2f}/cycle) -> min_lr={args.min_lr:.2e}",
        flush=True,
    )
    print("=" * 108, flush=True)

    opt, clip_buckets = build_optimizer_v2_4(model, base_lr=args.peak_lr, device=device)
    ind_isolated = clip_buckets["sal"] + clip_buckets["beta"]

    start_step = 0
    best_bpb = float("inf")
    best_score = float("inf")

    if args.resume and os.path.exists(latest_v2_4_ckpt):
        ck = torch.load(latest_v2_4_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        opt.load_state_dict(ck["optimizer_state"])
        start_step = int(ck.get("step", 0))
        best_bpb = float(ck.get("best_bpb", float("inf")))
        best_score = float(ck.get("best_score", float("inf")))
        prose_train_ds.load_state_dict(ck.get("prose_ds_state"))
        if "sft_skill_cur" in ck:
            sft_sampler.skill_cur = int(ck["sft_skill_cur"])
        if "niah_cur" in ck:
            niah_train_ds.cursor = int(ck["niah_cur"])
        last_bpb = float(ck.get("last_bpb", best_bpb if math.isfinite(best_bpb) else 1.5811))
        del ck
        gc.collect()
        if os.path.exists(telemetry_path):
            with open(telemetry_path, "r") as f:
                telemetry = json.load(f)
        else:
            telemetry = {"model": "VAULTLMv2.4", "history": []}
        step0_info = telemetry.get("step0", {})
        step0_skill_ce = float(step0_info.get("skill_val_ce", 0.98))
        step0_niah_wce = float(step0_info.get("niah_val_wce", 0.36))
        print(
            f"[*] Resumed VAULT v2.4 from step {start_step:,} "
            f"(prose_epoch={prose_train_ds.epoch}, prose_cursor={prose_train_ds.cursor:,}/{prose_train_ds.n_rows:,}, "
            f"best_BPB={best_bpb:.4f}, best_score={best_score:.4f})",
            flush=True,
        )
    else:
        step0_prose_ce = evaluate_prose_val_fast(model, val_prose_batches, device)
        _, step0_skill_ce = evaluate_masked_val_chunked(model, val_skill_batches, device)
        step0_niah_wce, step0_niah_ce = evaluate_masked_val_chunked(model, val_niah_batches, device)
        step0_bpb_res = evaluate_bpb(model, tok, device=device)
        last_bpb = float(step0_bpb_res["bpb"])
        best_bpb = last_bpb
        best_score = 0.75 * last_bpb + 0.15 * step0_prose_ce + 0.05 * step0_skill_ce + 0.05 * step0_niah_wce
        print(
            f"[*] Step-0 v2.3 Baseline: English_BPB={last_bpb:.4f} | "
            f"prose_v2_4_CE={step0_prose_ce:.4f} (PPL={math.exp(step0_prose_ce):.2f}) | "
            f"skill_CE={step0_skill_ce:.4f} | niah_CE={step0_niah_ce:.4f} (wCE={step0_niah_wce:.4f})",
            flush=True,
        )
        telemetry = {
            "model": "VAULTLMv2.4",
            "forked_from": src_v2_3_ckpt,
            "prime_banks": list(prime_banks),
            "unique_params": n_unique_params,
            "tok_per_step": tok_per_step,
            "cycle_steps": args.cycle_steps,
            "step0": {
                "english_bpb": last_bpb,
                "prose_val_ce": round(step0_prose_ce, 4),
                "prose_val_ppl": round(math.exp(step0_prose_ce), 2),
                "skill_val_ce": round(step0_skill_ce, 4),
                "niah_val_ce": round(step0_niah_ce, 4),
                "niah_val_wce": round(step0_niah_wce, 4),
                "composite_score": round(best_score, 4),
            },
            "history": [],
        }
        atomic_save_json(telemetry, telemetry_path)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t_run_start = time.perf_counter()
    toks_this_run = 0
    cycles_completed_this_run = 0
    stop_reason = "completed_all_steps"

    for step in range(start_step + 1, args.steps + 1):
        if not model.training:
            model.train()
        base_lr, c_idx, s_in = multi_cycle_cosine_lr(
            step, args.cycle_steps, args.peak_lr, args.min_lr, cycle_decay=args.cycle_decay
        )
        set_group_lrs(opt, base_lr)
        opt.zero_grad(set_to_none=True)

        ce_accum = 0.0
        prose_ce_sum = 0.0
        skill_ce_val = 0.0
        niah_ce_val = 0.0
        beta_val = 0.0
        ediv_val = 0.0
        raux_val = 0.0

        for micro_i in range(args.grad_accum):
            is_skill_micro = (micro_i == args.grad_accum - 2)
            is_niah_micro = (micro_i == args.grad_accum - 1)

            if is_niah_micro:
                for p in ind_isolated:
                    p.grad = None
                bx, by, bw = fetch_masked_batch(niah_train_ds, args.micro_batch, device)
                bw_row = normalize_row_weights(bw)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ce_loss, beta_mean, ent_div = model.forward_loss(
                        bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=True
                    )
                    beta_hinge = args.beta_hinge_coef * F.relu(beta_mean - args.beta_hinge_target)
                    loss = (ce_loss + beta_hinge + args.ent_div_coef * ent_div) / args.grad_accum
                loss.backward()
                niah_ce_val = float(ce_loss.item())
                beta_val = float(beta_mean.item())
                ediv_val = float(ent_div.item())
                for p in ind_isolated:
                    if p.grad is not None:
                        p.grad.mul_(float(args.grad_accum))

            elif is_skill_micro:
                bx, by, bw = sft_sampler.next_skill_batch(args.micro_batch, device)
                bw_row = normalize_row_weights(bw)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ce_loss = model.forward_loss(bx, by, loss_weights=bw_row, step_ratio=1.0)
                    loss = ce_loss / args.grad_accum
                loss.backward()
                skill_ce_val = float(ce_loss.item())

            else:
                bx, by = prose_train_ds.next_batch(args.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ce_loss = model.forward_loss(bx, by, loss_weights=None, step_ratio=1.0)
                    if micro_i == 0 and args.router_aux_coef > 0.0:
                        r_aux = model.compute_router_aux_loss()
                        raux_val = float(r_aux.item())
                        loss = (ce_loss / args.grad_accum) + args.router_aux_coef * r_aux
                    else:
                        loss = ce_loss / args.grad_accum
                loss.backward()
                prose_ce_sum += float(ce_loss.item()) / float(n_prose_micro)

            ce_accum += float(ce_loss.item()) / float(args.grad_accum)
            toks_this_run += int(bx.numel())

        if not math.isfinite(ce_accum):
            raise RuntimeError(f"Non-finite loss {ce_accum} at step {step}")

        gn_sal, gn_gates, gn_router, gn_base = clip_v2_4_param_groups(clip_buckets, max_norm=1.0)
        opt.step()

        is_cycle_end = (s_in == args.cycle_steps)
        if is_cycle_end:
            cycles_completed_this_run += 1

        wall_elapsed = time.perf_counter() - t_run_start
        hit_cycle_limit = (args.max_cycles_this_run > 0 and cycles_completed_this_run >= args.max_cycles_this_run)
        hit_wall_limit = (args.max_wall_seconds > 0.0 and wall_elapsed >= args.max_wall_seconds)
        stopping_now = (step == args.steps or hit_cycle_limit or hit_wall_limit)

        do_log = (step == 1 or step % args.log_interval == 0 or is_cycle_end or stopping_now)
        do_bpb = (step == 1 or step % args.bpb_interval == 0 or is_cycle_end or stopping_now)
        do_probe = (step % args.probe_interval == 0 or is_cycle_end or stopping_now)
        do_save_latest = (step % args.save_interval == 0 or is_cycle_end or stopping_now)

        if do_log:
            pr_ce = evaluate_prose_val_fast(model, val_prose_batches, device)
            _, sk_ce = evaluate_masked_val_chunked(model, val_skill_batches, device)
            ni_wce, ni_ce = evaluate_masked_val_chunked(model, val_niah_batches, device)

            if do_bpb:
                bpb_res = evaluate_bpb(model, tok, device=device)
                last_bpb = float(bpb_res["bpb"])
            model.train()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            comp_score = 0.75 * last_bpb + 0.15 * pr_ce + 0.05 * sk_ce + 0.05 * ni_wce
            tps = toks_this_run / max(wall_elapsed, 1e-6)
            total_tok_m = ( step * tok_per_step ) / 1e6
            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2) if device.startswith("cuda") else 0.0
            )

            print(
                f"[v2.4 {step:5d}/{args.steps} | c{c_idx}:{s_in:3d}/{args.cycle_steps} | +{total_tok_m:5.1f}M] "
                f"tr_p={prose_ce_sum:.3f} | BPB={last_bpb:.4f} | pr_PPL={math.exp(min(pr_ce, 15.0)):5.2f} | "
                f"sk={sk_ce:.3f} | ni_w={ni_wce:.3f} | beta={beta_val:.4f} | lr={base_lr:.2e} | "
                f"{tps:,.0f} tok/s ({peak_vram_mb:,.0f}MB)",
                flush=True,
            )

            rec = {
                "step": step,
                "cycle": c_idx,
                "step_in_cycle": s_in,
                "continual_tokens_M": round(total_tok_m, 2),
                "train_loss": round(ce_accum, 4),
                "train_prose_ce": round(prose_ce_sum, 4),
                "train_skill_ce": round(skill_ce_val, 4),
                "train_niah_wce": round(niah_ce_val, 4),
                "english_bpb": round(last_bpb, 4),
                "prose_val_ce": round(pr_ce, 4),
                "prose_val_ppl": round(math.exp(min(pr_ce, 15.0)), 2),
                "skill_val_ce": round(sk_ce, 4),
                "niah_val_ce": round(ni_ce, 4),
                "niah_val_wce": round(ni_wce, 4),
                "composite_score": round(comp_score, 4),
                "beta_ind_mean": round(beta_val, 5),
                "ent_div_mean": round(ediv_val, 4),
                "router_aux": round(raux_val, 5),
                "gn_sal": round(gn_sal, 3),
                "gn_gates": round(gn_gates, 3),
                "gn_router": round(gn_router, 3),
                "gn_base": round(gn_base, 3),
                "lr": base_lr,
                "tok_s": round(tps, 1),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "run_elapsed_s": round(wall_elapsed, 1),
            }

            probe_ok = True
            if do_probe and step >= 2:
                p_niah = generate_chatml_v2_2(
                    model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0
                )
                chatml_hits = sum(1 for _, code in unseen_niah_targets if code in p_niah)
                oneshot_probe = evaluate_niah_condition(
                    model,
                    tok,
                    mode="oneshot_base",
                    gap_mode="boilerplate_15tok",
                    n_sections=18,
                    n_trials=4,
                    seed_base=990000,
                    device=device,
                )
                model.train()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                rec["probe_chatml_2k"] = f"{chatml_hits}/{len(unseen_niah_targets)}"
                rec["probe_oneshot_2k_acc"] = oneshot_probe["cue_exact_acc"]
                rec["probe_oneshot_2k_nll_bits"] = oneshot_probe["mean_code_nll_bits"]
                probe_ok = (chatml_hits == len(unseen_niah_targets)) and (oneshot_probe["cue_exact_acc"] >= 0.75)
                print(
                    f"   [Probe @ {step}] ChatML_2K={chatml_hits}/{len(unseen_niah_targets)} | "
                    f"1Shot_Base_2K={oneshot_probe['cue_exact_acc']*100:.1f}% (NLL={oneshot_probe['mean_code_nll_bits']:.2f}b) | "
                    f"{p_niah.replace(chr(10), ' ⏎ ')[:150]}",
                    flush=True,
                )

            telemetry["history"].append(rec)
            atomic_save_json(telemetry, telemetry_path)

            guardrails_ok = (
                sk_ce <= step0_skill_ce + 0.10
                and ni_wce <= max(0.55, step0_niah_wce + 0.18)
                and probe_ok
            )
            if do_bpb and step >= min(25, args.steps) and guardrails_ok and (
                comp_score < best_score or last_bpb < best_bpb - 0.0015
            ):
                best_score = min(best_score, comp_score)
                best_bpb = min(best_bpb, last_bpb)
                atomic_save_pt(
                    {
                        "stage": "vault_v2_4",
                        "step": step,
                        "cycle": c_idx,
                        "total_tokens_seen": step * tok_per_step,
                        "prime_banks": prime_banks,
                        "composite_score": comp_score,
                        "english_bpb": last_bpb,
                        "prose_val_ce": pr_ce,
                        "skill_val_ce": sk_ce,
                        "niah_val_ce": ni_ce,
                        "niah_val_wce": ni_wce,
                        "model_state": model.state_dict(),
                    },
                    best_v2_4_ckpt,
                )
                print(
                    f"   [+] Saved NEW BEST VAULT v2.4 checkpoint at step {step} "
                    f"(BPB={last_bpb:.4f}, pr_PPL={math.exp(min(pr_ce, 15.0)):.2f}, score={comp_score:.4f}) -> {best_v2_4_ckpt}",
                    flush=True,
                )

        if do_save_latest:
            atomic_save_pt(
                {
                    "stage": "vault_v2_4_latest",
                    "step": step,
                    "prime_banks": prime_banks,
                    "best_bpb": best_bpb,
                    "best_score": best_score,
                    "last_bpb": last_bpb,
                    "prose_ds_state": prose_train_ds.state_dict(),
                    "sft_skill_cur": int(sft_sampler.skill_cur),
                    "niah_cur": int(getattr(niah_train_ds, "cursor", 0)),
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                },
                latest_v2_4_ckpt,
            )

        if hit_cycle_limit:
            stop_reason = f"completed_{cycles_completed_this_run}_cycles_at_step_{step}"
            break
        if hit_wall_limit:
            stop_reason = f"reached_max_wall_seconds_{args.max_wall_seconds:.0f}s_at_step_{step}"
            break

    print(
        f"\n[*] Segment finished cleanly ({stop_reason}). Latest={latest_v2_4_ckpt} | Best_BPB={best_bpb:.4f}",
        flush=True,
    )

    if not args.skip_final_eval and os.path.exists(best_v2_4_ckpt):
        del opt
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        ck_best = torch.load(best_v2_4_ckpt, map_location="cpu", weights_only=False)
        best_step_num = ck_best.get("step")
        best_bpb_val = ck_best.get("english_bpb")
        model.load_state_dict(ck_best["model_state"], strict=True)
        del ck_best
        gc.collect()
        print(
            f"[*] Running 8-Domain ChatML Evaluation on Best VAULT v2.4 (step {best_step_num}, "
            f"BPB={best_bpb_val:.4f})...",
            flush=True,
        )
        ev_final = run_8domain_eval_v2_2(
            model, tok, f"VAULT_v2_4_Step{best_step_num}", device, os.path.join(bench_dir, "eval_vault_v2_4_final.json")
        )
        telemetry["latest_8domain_summary"] = {
            "step": best_step_num,
            "english_bpb": best_bpb_val,
            "niah_exact_hits": ev_final["niah_exact_hits"],
            "math_180_exact": ev_final["math_180_exact"],
            "prose_clean": ev_final["prose_clean_direct_english"],
            "identifier_fidelity": ev_final["identifier_fidelity"],
        }
        atomic_save_json(telemetry, telemetry_path)


if __name__ == "__main__":
    main()
