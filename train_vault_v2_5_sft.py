#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 (`VAULTLMv2_5`, ~180.6M, 220 Co-Prime Voronoi Cells `P = (47, 53, 59, 61)`):
# Post-Step-10,000 Two-Stage Precision SFT, Contrastive Choice-Margin Calibration & 4K Horizon Cooldown
#
# Forked Read-Only from `checkpoints/vault_v2_4_step10000_best.pt` (NEVER modifies any v2.0..v2.4 checkpoint).
#
# Key Training & Scientific Guardrails:
#   1. Stationary Causal Leaky-EMA GTRM (`alpha = 2^-11 = 1/2048`) + Tri-Horizon `1..11`-Subword Sibling Binder (`g2`).
#   2. Strict Write-Gate Gradient Isolation: `ind_beta_proj`, `beta_proj`, and legacy `salience_binder` (`g0/g1`)
#      receive gradients EXCLUSIVELY from NIAH associative-retrieval microbatches (`0%` ChatML/prose write drift).
#   3. Sparse Shared-Prefix / Suffix Contrastive Choice-Margin Loss (`0.18 * L_MC / L_WG`) on decontaminated
#      official `train` splits (`0` leaks across `72,551` items verified by `audit_decontamination_v2_5.py`).
#   4. Two-Stage Curriculum (`1,400` steps @ `L=2,048` + `400` steps @ `L=4,096`), CPU-resident Polyak EMA (`0.992`),
#      Top-4 Checkpoint Soup (`vault_v2_5_soup.pt`), and Holdout English BPB Non-Regression Promotion Gate.

import argparse
import copy
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

from benchmarks.bench_vault_vs_mamba123_transformer import evaluate_bpb, evaluate_niah_condition
from src.vault_v2_5_model import VAULTLMv2_5, load_v2_4_into_v2_5
from src.vault_v2_5_streaming import VAULTv2_5StreamingEngine
from train_vault_v2_2 import normalize_row_weights


FROZEN_V2_5_KEYWORDS = (
    "x0_rms_norm",
    "kenyon_proj",
    "apl_norm",
)

LEGACY_SAL_LEX_KEYWORDS = (
    "salience_binder",
    "lex_router_g2",
    "lex_router_g3",
    "q_subhead_delta",
)


class FlexMemmapDataset:
    """
    Zero-overhead NVMe `(N, seq_len + 1)` uint16 tokens + `(N, seq_len)` uint8 salience masks loader
    supporting both `L = 2,048` (`Stage 1`) and `L = 4,096` (`Stage 2`) shards.
    """

    def __init__(self, data_dir: str, prefix: str, seq_len: int = 2048, seed: int = 20261103):
        tok_path = os.path.join(data_dir, f"{prefix}_tokens.bin")
        msk_path = os.path.join(data_dir, f"{prefix}_mask.bin")
        if not os.path.exists(tok_path) or not os.path.exists(msk_path):
            raise FileNotFoundError(f"Missing memmap shard: {tok_path} or {msk_path}")
        self.seq_len = int(seq_len)
        self.row_len = self.seq_len + 1
        row_bytes = self.row_len * 2
        file_bytes = os.path.getsize(tok_path)
        if file_bytes == 0 or file_bytes % row_bytes != 0:
            raise ValueError(f"Invalid memmap size {file_bytes} for {tok_path} (row_len={self.row_len})")
        self.n_rows = file_bytes // row_bytes
        self.tokens = np.memmap(tok_path, dtype=np.uint16, mode="r", shape=(self.n_rows, self.row_len))
        self.masks = np.memmap(msk_path, dtype=np.uint8, mode="r", shape=(self.n_rows, self.seq_len))
        self.base_seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = np.random.default_rng(self.base_seed).permutation(self.n_rows)

    def next_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cursor + batch_size > self.n_rows:
            self.epoch += 1
            self.order = np.random.default_rng(self.base_seed + self.epoch * 10007).permutation(self.n_rows)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size

        rows = np.asarray(self.tokens[idx], dtype=np.int64)
        msks = np.asarray(self.masks[idx], dtype=np.float32) * 0.1
        y_np = rows[:, 1:].copy()
        y_np[msks <= 0.0] = -100

        bx = torch.from_numpy(rows[:, :-1]).to(device, non_blocking=True)
        by = torch.from_numpy(y_np).to(device, non_blocking=True)
        bw = torch.from_numpy(msks).to(device, non_blocking=True)
        return bx, by, bw

    def fixed_eval_batches(
        self, n_batches: int, batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batches = []
        lim = min(self.n_rows, n_batches * batch_size)
        for st in range(0, lim, batch_size):
            rows = np.asarray(self.tokens[st : st + batch_size], dtype=np.int64)
            msks = np.asarray(self.masks[st : st + batch_size], dtype=np.float32) * 0.1
            y_np = rows[:, 1:].copy()
            y_np[msks <= 0.0] = -100
            batches.append(
                (
                    torch.from_numpy(rows[:, :-1]),
                    torch.from_numpy(y_np),
                    torch.from_numpy(msks),
                )
            )
        return batches


class ContrastiveChoiceSampler:
    """Loads pre-tokenized `mc_contrastive_{train,val}.npz` and `wg_contrastive_{train,val}.npz` arrays."""

    def __init__(self, npz_path: str, is_wg: bool = False, seed: int = 20261103):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing contrastive array: {npz_path}")
        d = np.load(npz_path)
        self.is_wg = is_wg
        if is_wg:
            self.seq_ids = d["seq_ids"]
            self.suf_start = d["suf_start"]
            self.suf_len = d["suf_len"]
            self.gold_idx = d["gold_idx"]
            self.n = len(self.seq_ids)
        else:
            self.prompt_ids = d["prompt_ids"]
            self.prompt_lens = d["prompt_lens"]
            self.choice_ids = d["choice_ids"]
            self.choice_lens = d["choice_lens"]
            self.choice_bytes = d["choice_bytes"]
            self.n_choices = d["n_choices"]
            self.gold_idx = d["gold_idx"]
            self.n = len(self.prompt_ids)
        self.base_seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = np.random.default_rng(self.base_seed).permutation(self.n)

    def next_mc_batch(self, batch_size: int, device: str):
        if self.cursor + batch_size > self.n:
            self.epoch += 1
            self.order = np.random.default_rng(self.base_seed + self.epoch * 7919).permutation(self.n)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        return (
            torch.from_numpy(np.asarray(self.prompt_ids[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.prompt_lens[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.choice_ids[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.choice_lens[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.choice_bytes[idx], dtype=np.float32)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.n_choices[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.gold_idx[idx], dtype=np.int64)).to(device, non_blocking=True),
        )

    def next_wg_batch(self, batch_size: int, device: str):
        if self.cursor + batch_size > self.n:
            self.epoch += 1
            self.order = np.random.default_rng(self.base_seed + self.epoch * 7919).permutation(self.n)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        return (
            torch.from_numpy(np.asarray(self.seq_ids[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.suf_start[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.suf_len[idx], dtype=np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.asarray(self.gold_idx[idx], dtype=np.int64)).to(device, non_blocking=True),
        )


def atomic_save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def configure_v2_5_trainable_and_optimizer(
    model: VAULTLMv2_5,
    base_lr: float,
    device: str,
    weight_decay: float = 0.05,
) -> Tuple[torch.optim.AdamW, Dict[str, List[torch.nn.Parameter]]]:
    """
    Freezes `x0_rms_norm`, `kenyon_proj`, and `apl_norm`, and constructs differential LR parameter groups:
      - `sal_legacy`    (`0.05x` LR, `wd=0.0`, NIAH-isolated): protects `1..7`-subword routing
      - `ext_span_g2`   (`2.50x` LR, `wd=0.0`): calibrates `8..11`-subword sibling channel (`ext_span_alpha`, `ind_shared_proj_g2`)
      - `write_gates`   (`3.00x` LR, `wd=0.0`, NIAH-isolated): sharpens `ind_beta_proj` & `beta_proj`
      - `vault_readout` (`2.00x` LR): `v_ind_proj`, `ind_out_proj`, `ind_read_proj`, `gtrm_proj`, branch gates
      - `backbone`      (`1.00x` LR): embeddings/lm_head, QKV projections, SwiGLU MLPs, RMSNorms
    """
    for name, p in model.named_parameters():
        if any(fk in name for fk in FROZEN_V2_5_KEYWORDS):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    sal_legacy: List[torch.nn.Parameter] = []
    ext_span_g2: List[torch.nn.Parameter] = []
    write_gates: List[torch.nn.Parameter] = []
    readout_decay: List[torch.nn.Parameter] = []
    readout_nodecay: List[torch.nn.Parameter] = []
    backbone_decay: List[torch.nn.Parameter] = []
    backbone_nodecay: List[torch.nn.Parameter] = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        if name.endswith(".ext_span_alpha") or name.endswith(".ind_shared_proj_g2"):
            ext_span_g2.append(p)
        elif name.endswith(".ind_shared_proj") or any(k in name for k in LEGACY_SAL_LEX_KEYWORDS):
            sal_legacy.append(p)
        elif "ind_beta_proj" in name or "beta_proj" in name:
            write_gates.append(p)
        elif any(
            k in name
            for k in ("v_ind_proj", "ind_out_proj", "ind_read_proj", "gtrm_proj", "gtrm_gate", "vault_gate", "syntax_gate")
        ):
            if p.ndim >= 2 and p.numel() >= model.d_model:
                readout_decay.append(p)
            else:
                readout_nodecay.append(p)
        elif p.ndim >= 2 and p.numel() >= model.d_model and "conv" not in name:
            backbone_decay.append(p)
        else:
            backbone_nodecay.append(p)

    param_groups = [
        {"name": "sal_legacy", "params": sal_legacy, "lr": base_lr * 0.05, "lr_mult": 0.05, "weight_decay": 0.0},
        {"name": "ext_span_g2", "params": ext_span_g2, "lr": base_lr * 2.50, "lr_mult": 2.50, "weight_decay": 0.0},
        {"name": "write_gates", "params": write_gates, "lr": base_lr * 3.00, "lr_mult": 3.00, "weight_decay": 0.0},
        {"name": "readout_decay", "params": readout_decay, "lr": base_lr * 2.00, "lr_mult": 2.00, "weight_decay": 0.01},
        {"name": "readout_nodecay", "params": readout_nodecay, "lr": base_lr * 2.00, "lr_mult": 2.00, "weight_decay": 0.0},
        {"name": "backbone_decay", "params": backbone_decay, "lr": base_lr, "lr_mult": 1.00, "weight_decay": weight_decay},
        {"name": "backbone_nodecay", "params": backbone_nodecay, "lr": base_lr, "lr_mult": 1.00, "weight_decay": 0.0},
    ]
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    use_fused = device.startswith("cuda") and torch.cuda.is_available()
    opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    isolated_groups = {
        "niah_only": sal_legacy + write_gates,
        "ext_span_g2": ext_span_g2,
    }
    return opt, isolated_groups


def update_optimizer_lrs(opt: torch.optim.AdamW, base_lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = base_lr * float(pg["lr_mult"])


class CPUPolyakEMA:
    """Maintains an exact FP32 Exponential Moving Average of trainable parameters in pinned/host CPU RAM (`0 MB` GPU VRAM!)."""

    def __init__(self, model: VAULTLMv2_5, decay: float = 0.992):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = v.detach().float().cpu().clone()

    @torch.no_grad()
    def update(self, model: VAULTLMv2_5, cur_decay: Optional[float] = None) -> None:
        d = self.decay if cur_decay is None else float(cur_decay)
        one_minus_d = 1.0 - d
        for k, v in model.state_dict().items():
            if not v.is_floating_point():
                self.shadow[k].copy_(v.detach().cpu())
            else:
                self.shadow[k].mul_(d).add_(v.detach().float().cpu(), alpha=one_minus_d)

    @torch.no_grad()
    def clone_state_bf16(self, ref_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, t_cpu in self.shadow.items():
            dt = ref_sd[k].dtype if k in ref_sd else torch.bfloat16
            out[k] = t_cpu.to(dtype=dt).clone()
        return out


@torch.no_grad()
def evaluate_masked_batches(
    model: VAULTLMv2_5,
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    tot_w_loss = 0.0
    tot_raw_ce = 0.0
    for vx_cpu, vy_cpu, vw_cpu in batches:
        vx = vx_cpu.to(device, non_blocking=True)
        vy = vy_cpu.to(device, non_blocking=True)
        vw = vw_cpu.to(device, non_blocking=True)
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)
        with ctx:
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
            tot_w_loss += w_num / max(1.0, float(w_flat.sum().item()))
            tot_raw_ce += raw_num / max(1.0, float((w_flat > 0.0).float().sum().item()))
    if was_training:
        model.train()
    n = max(1, len(batches))
    return tot_w_loss / n, tot_raw_ce / n


@torch.no_grad()
def evaluate_contrastive_val(
    model: VAULTLMv2_5,
    mc_val: ContrastiveChoiceSampler,
    wg_val: ContrastiveChoiceSampler,
    eos_id: int,
    device: str,
    max_mc_items: int = 400,
    max_wg_items: int = 240,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    n_mc = min(mc_val.n, max_mc_items)
    mc_hits = 0.0
    mc_cnt = 0
    for st in range(0, n_mc, 16):
        en = min(n_mc, st + 16)
        bs = en - st
        p_ids = torch.from_numpy(np.asarray(mc_val.prompt_ids[st:en], dtype=np.int64)).to(device)
        p_lens = torch.from_numpy(np.asarray(mc_val.prompt_lens[st:en], dtype=np.int64)).to(device)
        c_ids = torch.from_numpy(np.asarray(mc_val.choice_ids[st:en], dtype=np.int64)).to(device)
        c_lens = torch.from_numpy(np.asarray(mc_val.choice_lens[st:en], dtype=np.int64)).to(device)
        c_bytes = torch.from_numpy(np.asarray(mc_val.choice_bytes[st:en], dtype=np.float32)).to(device)
        n_ch = torch.from_numpy(np.asarray(mc_val.n_choices[st:en], dtype=np.int64)).to(device)
        g_i = torch.from_numpy(np.asarray(mc_val.gold_idx[st:en], dtype=np.int64)).to(device)
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)
        with ctx:
            _, acc = model.forward_mc_contrastive_loss(
                p_ids, p_lens, c_ids, c_lens, c_bytes, n_ch, g_i, eos_id=eos_id
            )
        mc_hits += acc * bs
        mc_cnt += bs

    n_wg = min(wg_val.n, max_wg_items)
    wg_hits = 0.0
    wg_cnt = 0
    for st in range(0, n_wg, 24):
        en = min(n_wg, st + 24)
        bs = en - st
        sq = torch.from_numpy(np.asarray(wg_val.seq_ids[st:en], dtype=np.int64)).to(device)
        sst = torch.from_numpy(np.asarray(wg_val.suf_start[st:en], dtype=np.int64)).to(device)
        sln = torch.from_numpy(np.asarray(wg_val.suf_len[st:en], dtype=np.int64)).to(device)
        g_w = torch.from_numpy(np.asarray(wg_val.gold_idx[st:en], dtype=np.int64)).to(device)
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)
        with ctx:
            _, acc_w = model.forward_wg_contrastive_loss(sq, sst, sln, g_w)
        wg_hits += acc_w * bs
        wg_cnt += bs

    if was_training:
        model.train()
    return 100.0 * mc_hits / max(1, mc_cnt), 100.0 * wg_hits / max(1, wg_cnt)


def compute_v2_5_lr(
    step: int,
    stage1_steps: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    stage1_end_lr: float,
    stage2_min_lr: float,
) -> float:
    if step <= warmup_steps:
        return peak_lr * (float(step) / float(max(1, warmup_steps)))
    if step <= stage1_steps:
        prog = float(step - warmup_steps) / float(max(1, stage1_steps - warmup_steps))
        return stage1_end_lr + 0.5 * (peak_lr - stage1_end_lr) * (1.0 + math.cos(math.pi * prog))
    prog2 = float(step - stage1_steps) / float(max(1, total_steps - stage1_steps))
    return stage2_min_lr + 0.5 * (stage1_end_lr - stage2_min_lr) * (1.0 + math.cos(math.pi * prog2))


def main() -> None:
    parser = argparse.ArgumentParser(description="VAULT v2.5 Post-10K Precision SFT & 4K Horizon Cooldown")
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "vault_v2_4_step10000_best.pt"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v2_5"),
    )
    parser.add_argument("--stage1_steps", type=int, default=1400)
    parser.add_argument("--stage2_steps", type=int, default=400)
    parser.add_argument("--warmup_steps", type=int, default=60)
    parser.add_argument("--peak_lr", type=float, default=4.0e-5)
    parser.add_argument("--stage1_end_lr", type=float, default=8.0e-6)
    parser.add_argument("--stage2_min_lr", type=float, default=1.5e-6)
    parser.add_argument("--mc_loss_weight", type=float, default=0.18)
    parser.add_argument("--ent_div_weight", type=float, default=0.03)
    parser.add_argument("--ema_decay", type=float, default=0.992)
    parser.add_argument("--init_ext_span_alpha", type=float, default=0.55)
    parser.add_argument("--bpb_guardrail_tol", type=float, default=0.008)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--cpu_smoke_test", action="store_true", help="Run a 2-step CPU smoke test with tiny dims/batches")
    args = parser.parse_args()

    t_run0 = time.perf_counter()
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    telemetry_path = os.path.join(base_dir, "benchmarks", "vault_v2_5_sft_telemetry.json")

    if args.cpu_smoke_test:
        device = "cpu"
        args.stage1_steps = 1
        args.stage2_steps = 1
        args.warmup_steps = 1
        args.eval_every = 1
        args.log_every = 1
        if not os.path.exists(args.init_ckpt):
            args.init_ckpt = os.path.join(ckpt_dir, "vault_v2_4_step7000_best.pt")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm GPU required for full VAULT v2.5 SFT training (pass --cpu_smoke_test for CPU verification).")
        device = "cuda:0"
        torch.backends.cuda.matmul.allow_tf32 = True

    total_steps = args.stage1_steps + args.stage2_steps
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    eos_id = int(tok.convert_tokens_to_ids("<|endoftext|>"))

    print("=" * 108, flush=True)
    print(f"VAULT v2.5 POST-10K PRECISION SFT & 4K HORIZON COOLDOWN | Device: {device}", flush=True)
    print(f"Init Checkpoint : {args.init_ckpt}", flush=True)
    print(f"Data Directory  : {args.data_dir}", flush=True)
    print("=" * 108, flush=True)

    # Load datasets
    ds_prose_2k = FlexMemmapDataset(args.data_dir, "prose_climax_train", seq_len=2048, seed=101)
    ds_skill_2k = FlexMemmapDataset(args.data_dir, "sft_skill_train", seq_len=2048, seed=102)
    ds_niah_2k = FlexMemmapDataset(args.data_dir, "niah_2k_train", seq_len=2048, seed=103)

    ds_prose_4k = FlexMemmapDataset(args.data_dir, "prose_climax_4k_train", seq_len=4096, seed=201)
    ds_skill_4k = FlexMemmapDataset(args.data_dir, "sft_skill_4k_train", seq_len=4096, seed=202)
    ds_niah_4k = FlexMemmapDataset(args.data_dir, "niah_4k_train", seq_len=4096, seed=203)

    mc_train = ContrastiveChoiceSampler(os.path.join(args.data_dir, "mc_contrastive_train.npz"), is_wg=False, seed=301)
    mc_val = ContrastiveChoiceSampler(os.path.join(args.data_dir, "mc_contrastive_val.npz"), is_wg=False, seed=302)
    wg_train = ContrastiveChoiceSampler(os.path.join(args.data_dir, "wg_contrastive_train.npz"), is_wg=True, seed=303)
    wg_val = ContrastiveChoiceSampler(os.path.join(args.data_dir, "wg_contrastive_val.npz"), is_wg=True, seed=304)

    n_val_b = 1 if args.cpu_smoke_test else 16
    val_prose_2k = FlexMemmapDataset(args.data_dir, "prose_climax_val", seq_len=2048).fixed_eval_batches(n_val_b, 2)
    val_skill_2k = FlexMemmapDataset(args.data_dir, "sft_skill_val", seq_len=2048).fixed_eval_batches(n_val_b, 2)
    val_niah_2k = FlexMemmapDataset(args.data_dir, "niah_2k_val", seq_len=2048).fixed_eval_batches(n_val_b, 2)
    val_niah_4k = FlexMemmapDataset(args.data_dir, "niah_4k_val", seq_len=4096).fixed_eval_batches(
        1 if args.cpu_smoke_test else 12, 1
    )

    # Instantiate & load VAULTLMv2_5
    if args.cpu_smoke_test:
        # Use compact 2-layer VAULTLMv2_5 on CPU so full 8-microbatch Stage 1 + Stage 2 + EMA + Soup runs in ~20s
        # (Checkpoint loading of full 16-layer `vault_v2_4_*` was already verified in `tests/test_vault_v2_5_cpu.py`!)
        model = VAULTLMv2_5(
            vocab_size=len(tok),
            d_model=256,
            n_layers=2,
            n_q_heads=12,
            n_vault_groups=4,
            head_dim=64,
            d_ff=512,
            chunk_size=64,
            grad_checkpoint="half",
            prime_banks=(47, 53, 59, 61),
        ).to(device)
        for blk in model.layers:
            blk.attn.ext_span_alpha.data.fill_(args.init_ext_span_alpha)
        base_tokens_seen = 0
        init_info = {"mode": "cpu_smoke_2L", "init_ext_span_alpha": args.init_ext_span_alpha}
    else:
        if not os.path.exists(args.init_ckpt):
            raise FileNotFoundError(f"Missing init checkpoint: {args.init_ckpt}")
        raw_ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        prime_banks = tuple(raw_ckpt.get("prime_banks", (47, 53, 59, 61)))
        base_tokens_seen = int(raw_ckpt.get("total_tokens_seen", 0))
        model = VAULTLMv2_5(
            vocab_size=len(tok),
            d_model=768,
            n_layers=16,
            n_q_heads=12,
            n_vault_groups=4,
            head_dim=64,
            d_ff=2048,
            chunk_size=64,
            grad_checkpoint="full",
            prime_banks=prime_banks,
        )
        init_info = load_v2_4_into_v2_5(model, raw_ckpt, init_ext_span_alpha=args.init_ext_span_alpha)
        del raw_ckpt
        gc.collect()
        model = model.to(device)

    opt, isolated_groups = configure_v2_5_trainable_and_optimizer(model, args.peak_lr, device=device)
    ema = CPUPolyakEMA(model, decay=args.ema_decay)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = model.count_parameters()
    print(
        f"[+] Initialized VAULTLMv2_5: {n_total:,} total params ({n_trainable:,} trainable, {n_total - n_trainable:,} frozen) "
        f"| load_info={init_info}",
        flush=True,
    )

    def run_validation(eval_model: VAULTLMv2_5, tag: str) -> Dict[str, float]:
        eval_model.eval()
        if args.cpu_smoke_test:
            bpb_res = {"bpb": 6.50, "ppl": 90.0}
        else:
            bpb_res = evaluate_bpb(eval_model, tok, device=device)
        _, prose_ce = evaluate_masked_batches(eval_model, val_prose_2k, device=device)
        _, skill_ce = evaluate_masked_batches(eval_model, val_skill_2k, device=device)
        niah2_wce, niah2_ce = evaluate_masked_batches(eval_model, val_niah_2k, device=device)
        niah4_wce, niah4_ce = evaluate_masked_batches(eval_model, val_niah_4k, device=device)
        mc_acc, wg_acc = evaluate_contrastive_val(
            eval_model,
            mc_val,
            wg_val,
            eos_id=eos_id,
            device=device,
            max_mc_items=(8 if args.cpu_smoke_test else 400),
            max_wg_items=(8 if args.cpu_smoke_test else 240),
        )
        # Higher `composite` is better: rewards MC/WG accuracy + low Skill CE + low 2K/4K NIAH CE + low English BPB
        composite = (
            0.40 * (mc_acc / 100.0)
            + 0.15 * (wg_acc / 100.0)
            - 0.12 * skill_ce
            - 0.12 * niah2_ce
            - 0.11 * niah4_ce
            - 0.25 * float(bpb_res["bpb"])
        )
        metrics = {
            "tag": tag,
            "english_bpb": round(float(bpb_res["bpb"]), 4),
            "prose_val_ce": round(float(prose_ce), 4),
            "skill_val_ce": round(float(skill_ce), 4),
            "niah_2k_val_ce": round(float(niah2_ce), 4),
            "niah_2k_val_wce": round(float(niah2_wce), 4),
            "niah_4k_val_ce": round(float(niah4_ce), 4),
            "niah_4k_val_wce": round(float(niah4_wce), 4),
            "mc_val_acc": round(float(mc_acc), 2),
            "wg_val_acc": round(float(wg_acc), 2),
            "composite": round(float(composite), 5),
        }
        print(
            f"  [Eval {tag:<12s}] BPB={metrics['english_bpb']:.4f} | MC_Acc={metrics['mc_val_acc']:5.2f}% | "
            f"WG_Acc={metrics['wg_val_acc']:5.2f}% | SkillCE={metrics['skill_val_ce']:.3f} | "
            f"NIAH2K_CE={metrics['niah_2k_val_ce']:.3f} | NIAH4K_CE={metrics['niah_4k_val_ce']:.3f} | "
            f"Comp={metrics['composite']:+.4f}",
            flush=True,
        )
        return metrics

    # Step-0 Baseline Audit (sets the English BPB non-regression guardrail!)
    step0_metrics = run_validation(model, "step_0_init")
    step0_bpb = float(step0_metrics["english_bpb"])
    bpb_ceiling = step0_bpb + float(args.bpb_guardrail_tol)
    print(f"[+] Holdout English BPB Guardrail Armed: candidate BPB must stay <= {bpb_ceiling:.4f} (step0={step0_bpb:.4f})", flush=True)

    best_composite = -1e9
    best_step = 0
    best_metrics = step0_metrics
    soup_candidates: List[Tuple[float, int, Dict[str, torch.Tensor]]] = []
    history: List[Dict] = [{"step": 0, **step0_metrics}]

    sft_tokens_seen = 0
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    eval_shadow_model: Optional[VAULTLMv2_5] = None

    t_loop0 = time.perf_counter()
    running_ce = 0.0
    running_mc = 0.0
    running_mc_acc = 0.0
    running_niah = 0.0
    running_cnt = 0

    for step in range(1, total_steps + 1):
        in_stage2 = step > args.stage1_steps
        if step == args.stage1_steps + 1:
            print("\n" + "=" * 108, flush=True)
            print(
                f">>> ENTERING STAGE 2 (`L = 4,096` Long-Horizon & Dense-Manifest Cooldown, Steps {step}..{total_steps})",
                flush=True,
            )
            print("=" * 108, flush=True)
            model.grad_checkpoint = "full"
            if use_amp:
                gc.collect()
                torch.cuda.empty_cache()

        cur_lr = compute_v2_5_lr(
            step,
            stage1_steps=args.stage1_steps,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            stage1_end_lr=args.stage1_end_lr,
            stage2_min_lr=args.stage2_min_lr,
        )
        update_optimizer_lrs(opt, cur_lr)
        model.train()
        opt.zero_grad(set_to_none=True)

        grad_accum = 2 if args.cpu_smoke_test else 8
        inv_accum = 1.0 / float(grad_accum)
        first_niah_micro = (grad_accum - 1) if (not in_stage2 or args.cpu_smoke_test) else (grad_accum - 2)
        n_niah_micros = grad_accum - first_niah_micro

        step_ce_acc = 0.0
        step_niah_acc = 0.0
        step_mc_val = 0.0
        step_mc_acc_val = 0.0

        for micro_i in range(grad_accum):
            # Zero `niah_only` (`sal_legacy` + `write_gates`) gradients right before the NIAH microbatches
            # so write gates & legacy binder NEVER absorb ChatML/prose gradient drift!
            if micro_i == first_niah_micro:
                for p in isolated_groups["niah_only"]:
                    if p.grad is not None:
                        p.grad.zero_()

            is_niah_mb = micro_i >= first_niah_micro
            is_prose_mb = (not is_niah_mb) and (micro_i < (0 if args.cpu_smoke_test else 2))
            is_mc_mb = (not is_niah_mb) and (micro_i == (0 if args.cpu_smoke_test else 2))

            if in_stage2:
                mb_size = 1
                seq_len_cur = 256 if args.cpu_smoke_test else 4096
                if is_niah_mb:
                    bx, by, bw = ds_niah_4k.next_batch(mb_size, device)
                elif is_prose_mb:
                    bx, by, bw = ds_prose_4k.next_batch(mb_size, device)
                else:
                    bx, by, bw = ds_skill_4k.next_batch(mb_size, device)
            else:
                mb_size = 1 if args.cpu_smoke_test else 2
                seq_len_cur = 256 if args.cpu_smoke_test else 2048
                if is_niah_mb:
                    bx, by, bw = ds_niah_2k.next_batch(mb_size, device)
                elif is_prose_mb:
                    bx, by, bw = ds_prose_2k.next_batch(mb_size, device)
                else:
                    bx, by, bw = ds_skill_2k.next_batch(mb_size, device)

            if args.cpu_smoke_test:
                bx = bx[:, :seq_len_cur]
                by = by[:, :seq_len_cur]
                bw = bw[:, :seq_len_cur]
                if (bw > 0).sum() == 0:
                    bw = torch.ones_like(bw)
                    by = torch.where(by == -100, bx, by)

            bw_norm = normalize_row_weights(bw)
            sft_tokens_seen += int(bx.numel())

            amp_ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else torch.amp.autocast("cpu", enabled=False)
            )
            with amp_ctx:
                if is_niah_mb:
                    ce_loss, _, ent_div = model.forward_loss(
                        bx,
                        by,
                        loss_weights=bw_norm,
                        step_ratio=1.0,
                        return_beta_mean=True,
                        return_ent_div=True,
                    )
                    mb_loss = (ce_loss + float(args.ent_div_weight) * ent_div) * inv_accum
                    step_niah_acc += float(ce_loss.item()) / float(n_niah_micros)
                else:
                    ce_loss = model.forward_loss(bx, by, loss_weights=bw_norm, step_ratio=1.0)
                    mb_loss = ce_loss * inv_accum
            step_ce_acc += float(ce_loss.item()) * inv_accum
            mb_loss.backward()

            # On `is_mc_mb`, run the sparse Shared-Prefix MC / Winogrande Choice-Margin backward pass
            # AFTER `mb_loss.backward()` has already freed the `L=2048/4096` sequence activations!
            if is_mc_mb and args.mc_loss_weight > 0.0:
                do_wg = (step % 4) == 0
                n_sub = 1 if args.cpu_smoke_test else 2
                w_sub = float(args.mc_loss_weight) / float(n_sub)
                step_mc_val = 0.0
                step_mc_acc_val = 0.0
                for _ in range(n_sub):
                    with amp_ctx:
                        if do_wg:
                            sq, sst, sln, gw = wg_train.next_wg_batch(2 if args.cpu_smoke_test else 6, device)
                            c_loss, c_acc = model.forward_wg_contrastive_loss(sq, sst, sln, gw, step_ratio=1.0)
                        else:
                            p_i, p_l, c_i, c_l, c_b, n_c, g_i = mc_train.next_mc_batch(
                                2 if args.cpu_smoke_test else 4, device
                            )
                            c_loss, c_acc = model.forward_mc_contrastive_loss(
                                p_i, p_l, c_i, c_l, c_b, n_c, g_i, eos_id=eos_id, step_ratio=1.0
                            )
                        scaled_c = w_sub * c_loss
                    scaled_c.backward()
                    step_mc_val += float(c_loss.item()) / float(n_sub)
                    step_mc_acc_val += 100.0 * float(c_acc) / float(n_sub)

        # Rescale NIAH-isolated parameter gradients by `grad_accum / n_niah_micros` so effective step size matches!
        niah_scale = float(grad_accum) / float(max(1, n_niah_micros))
        for p in isolated_groups["niah_only"]:
            if p.grad is not None:
                p.grad.mul_(niah_scale)

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        opt.step()

        ema_cur_d = min(args.ema_decay, float(step) / float(step + 10))
        ema.update(model, cur_decay=ema_cur_d)

        running_ce += step_ce_acc
        running_niah += step_niah_acc
        running_mc += step_mc_val
        running_mc_acc += step_mc_acc_val
        running_cnt += 1

        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t_loop0
            tok_s = sft_tokens_seen / max(1e-3, elapsed)
            ext_a0 = float(model.layers[0].attn.ext_span_alpha.item())
            ext_a15 = float(model.layers[-1].attn.ext_span_alpha.item())
            vram_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if use_amp
                else 0.0
            )
            print(
                f"[Step {step:4d}/{total_steps} | {'S2-4K' if in_stage2 else 'S1-2K'}] "
                f"lr={cur_lr:.2e} | CE={running_ce/running_cnt:.3f} | NIAH_CE={running_niah/running_cnt:.3f} | "
                f"MC_Loss={running_mc/running_cnt:.3f} ({running_mc_acc/running_cnt:4.1f}%) | "
                f"gnorm={grad_norm:.2f} | ext_a=[{ext_a0:.2f},{ext_a15:.2f}] | "
                f"{tok_s:,.0f} tok/s | VRAM={vram_gb:.2f}GB",
                flush=True,
            )
            running_ce = running_niah = running_mc = running_mc_acc = 0.0
            running_cnt = 0

        if step % args.eval_every == 0 or step == args.stage1_steps or step == total_steps:
            # Evaluate the Polyak EMA weights (`ema.shadow`) by swapping temporarily with live weights (0 extra GPU VRAM!)
            ref_sd = model.state_dict()
            live_backup = {k: v.detach().cpu().clone() for k, v in ref_sd.items()}
            ema_bf16 = ema.clone_state_bf16(ref_sd)
            model.load_state_dict(ema_bf16, strict=True)

            ev = run_validation(model, f"ema_s{step}")
            ev_record = {
                "step": step,
                "stage": 2 if in_stage2 else 1,
                "sft_tokens_seen": sft_tokens_seen,
                "total_tokens_seen": base_tokens_seen + sft_tokens_seen,
                **ev,
            }
            history.append(ev_record)

            bpb_ok = float(ev["english_bpb"]) <= bpb_ceiling
            if bpb_ok and float(ev["composite"]) > best_composite:
                best_composite = float(ev["composite"])
                best_step = step
                best_metrics = ev_record
                if not args.cpu_smoke_test:
                    best_payload = {
                        "version": "vault_v2_5",
                        "step": step,
                        "stage": 2 if in_stage2 else 1,
                        "prime_banks": model.prime_banks,
                        "ema_alpha": model.ema_alpha,
                        "sft_tokens_seen": sft_tokens_seen,
                        "total_tokens_seen": base_tokens_seen + sft_tokens_seen,
                        "english_bpb": ev["english_bpb"],
                        "metrics": ev_record,
                        "init_ckpt": args.init_ckpt,
                        "model_state": ema_bf16,
                    }
                    torch.save(best_payload, os.path.join(ckpt_dir, "vault_v2_5_best.pt"))
                    print(
                        f"  [★ NEW BEST VAULT v2.5] Step {step} -> saved checkpoints/vault_v2_5_best.pt "
                        f"(Comp={best_composite:+.4f}, BPB={ev['english_bpb']:.4f})",
                        flush=True,
                    )
            elif not bpb_ok:
                print(
                    f"  [!] Guardrail skipped promotion at Step {step}: BPB {ev['english_bpb']:.4f} > ceiling {bpb_ceiling:.4f}",
                    flush=True,
                )

            # Collect Top-4 diverse cooldown EMA snapshots (`step >= max(1, total_steps // 2)`) for Checkpoint Soup!
            if bpb_ok and (step >= max(1, total_steps // 2) or args.cpu_smoke_test):
                soup_candidates.append((float(ev["composite"]), step, copy.deepcopy(ema_bf16)))
                soup_candidates.sort(key=lambda x: x[0], reverse=True)
                if len(soup_candidates) > 4:
                    soup_candidates.pop()

            # Restore live training weights seamlessly
            model.load_state_dict({k: v.to(device=ref_sd[k].device, dtype=ref_sd[k].dtype) for k, v in live_backup.items()}, strict=True)
            del live_backup
            if not args.cpu_smoke_test:
                atomic_save_json(
                    {
                        "init_ckpt": args.init_ckpt,
                        "step0_bpb": step0_bpb,
                        "bpb_ceiling": bpb_ceiling,
                        "best_step": best_step,
                        "best_metrics": best_metrics,
                        "history": history,
                    },
                    telemetry_path,
                )

    # ==========================================================================
    # FINAL TOP-4 CHECKPOINT SOUP CONSTRUCTION & HEAD-TO-HEAD PROMOTION GATE
    # ==========================================================================
    soup_metrics = None
    promoted_source = f"ema_step_{best_step}"
    if len(soup_candidates) >= 2:
        soup_steps = [s for _, s, _ in soup_candidates]
        print(f"\n[+] Averaging Top-{len(soup_candidates)} Cooldown EMA Snapshots {soup_steps} -> Uniform Checkpoint Soup...", flush=True)
        ref_sd = model.state_dict()
        soup_sd: Dict[str, torch.Tensor] = {}
        inv_k = 1.0 / float(len(soup_candidates))
        for k in soup_candidates[0][2].keys():
            t0_s = soup_candidates[0][2][k]
            if not t0_s.is_floating_point():
                soup_sd[k] = t0_s.clone()
            else:
                acc = torch.zeros_like(t0_s, dtype=torch.float32)
                for _, _, cand_sd in soup_candidates:
                    acc.add_(cand_sd[k].float(), alpha=inv_k)
                soup_sd[k] = acc.to(dtype=t0_s.dtype)

        model.load_state_dict({k: v.to(device=ref_sd[k].device, dtype=ref_sd[k].dtype) for k, v in soup_sd.items()}, strict=True)
        soup_metrics = run_validation(model, "top4_soup")
        if not args.cpu_smoke_test:
            soup_payload = {
                "version": "vault_v2_5_soup",
                "soup_steps": soup_steps,
                "prime_banks": model.prime_banks,
                "ema_alpha": model.ema_alpha,
                "sft_tokens_seen": sft_tokens_seen,
                "total_tokens_seen": base_tokens_seen + sft_tokens_seen,
                "english_bpb": soup_metrics["english_bpb"],
                "metrics": soup_metrics,
                "init_ckpt": args.init_ckpt,
                "model_state": soup_sd,
            }
            torch.save(soup_payload, os.path.join(ckpt_dir, "vault_v2_5_soup.pt"))
            if float(soup_metrics["english_bpb"]) <= bpb_ceiling and float(soup_metrics["composite"]) >= best_composite:
                torch.save(soup_payload, os.path.join(ckpt_dir, "vault_v2_5_best.pt"))
                promoted_source = f"top4_soup_{soup_steps}"
                best_metrics = soup_metrics
                print(
                    f"[★ PROMOTED SOUP -> checkpoints/vault_v2_5_best.pt] Comp={soup_metrics['composite']:+.4f} | "
                    f"BPB={soup_metrics['english_bpb']:.4f}",
                    flush=True,
                )

    total_sec = round(time.perf_counter() - t_run0, 1)
    if not args.cpu_smoke_test:
        atomic_save_json(
            {
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_sec": total_sec,
                "init_ckpt": args.init_ckpt,
                "promoted_source": promoted_source,
                "step0_bpb": step0_bpb,
                "bpb_ceiling": bpb_ceiling,
                "best_step": best_step,
                "best_metrics": best_metrics,
                "soup_metrics": soup_metrics,
                "history": history,
            },
            telemetry_path,
        )
    print("\n" + "=" * 108, flush=True)
    print(
        f"[DONE] VAULT v2.5 SFT Complete in {total_sec:.1f}s | Promoted: {promoted_source} | "
        f"Best Comp={best_metrics['composite']:+.4f} | BPB={best_metrics['english_bpb']:.4f}",
        flush=True,
    )
    print("=" * 108, flush=True)


if __name__ == "__main__":
    main()
