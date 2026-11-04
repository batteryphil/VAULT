#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.3 (`180.5M`, Expanded Co-Prime Voronoi Banks `P = (47, 53, 59, 61)` = `220` Cells, `8.97M` CRT Capacity):
# Lossless Voronoi Cell Mitosis Fork from `checkpoints/vault_v2_2_best.pt` + Balanced Continual Adaptation
# (`50%` Unseen Educational Plain-Text Prose + `25%` 8-Domain ChatML Skills + `25%` Dense ChatML & 1-Shot NIAH).

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
from src.vault_v2_model import VAULTLMv2, upgrade_v2_2_to_wider_primes
from train_vault_v2 import SEQ_LEN, NVMeMemmapDataset, build_unseen_2k_niah_eval
from train_vault_v2_2 import (
    SAL_COPY_LEX_KEYWORDS,
    PartitionedMemmapSampler,
    normalize_row_weights,
    evaluate_val_loss_fast,
    generate_chatml_v2_2,
    run_8domain_eval_v2_2,
    run_controlled_and_32k_extrapolation_eval,
    cosine_lr,
    set_group_lrs,
)
from benchmarks.bench_vault_vs_mamba123_transformer import evaluate_bpb, evaluate_niah_condition


def fetch_masked_batch(
    ds: NVMeMemmapDataset, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx, by, bw = ds.next_batch(batch_size, device)
    by = torch.where(bw > 0.0, by, torch.full_like(by, -100))
    return bx, by, bw


def build_optimizer_v2_3(
    model: VAULTLMv2,
    base_lr: float,
    sal_mult: float = 0.08,
    beta_mult: float = 2.0,
    read_mult: float = 2.0,
    router_mult: float = 0.65,
    weight_decay: float = 0.03,
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
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    clip_buckets = {
        "sal": sal_copy_lex,
        "beta": beta_params,
        "gates": beta_params + read_params,
        "router": router_params,
        "base": decay_params + nodecay_params,
    }
    return opt, clip_buckets


def clip_v2_3_param_groups(
    clip_buckets: Dict[str, List[torch.nn.Parameter]], max_norm: float = 1.0
) -> Tuple[float, float, float, float]:
    gn_sal = float(torch.nn.utils.clip_grad_norm_(clip_buckets["sal"], 0.15).item())
    gn_gates = float(torch.nn.utils.clip_grad_norm_(clip_buckets["gates"], max_norm).item())
    gn_router = float(torch.nn.utils.clip_grad_norm_(clip_buckets["router"], 0.30).item())
    gn_base = float(torch.nn.utils.clip_grad_norm_(clip_buckets["base"], max_norm).item())
    return gn_sal, gn_gates, gn_router, gn_base


def atomic_save_pt(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Train VAULT v2.3 (220 Co-Prime Cells) forked from vault_v2_2_best.pt")
    parser.add_argument("--steps", type=int, default=400, help="Total continual training steps (400 = 6.55M tokens)")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--ckpt_interval", type=int, default=2)
    parser.add_argument("--peak_lr", type=float, default=4.5e-5)
    parser.add_argument("--min_lr", type=float, default=4.0e-6)
    parser.add_argument("--router_aux_coef", type=float, default=0.015)
    parser.add_argument("--beta_hinge_coef", type=float, default=0.50)
    parser.add_argument("--beta_hinge_target", type=float, default=0.016)
    parser.add_argument("--ent_div_coef", type=float, default=0.03)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--bpb_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20261101)
    np.random.seed(20261101)
    random.seed(20261101)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_v2_2_dir = os.path.join(base_dir, "data_v2_2")
    data_v2_3_dir = os.path.join(base_dir, "data_v2_3")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    prime_banks = (47, 53, 59, 61)
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
        ckpt_interval=args.ckpt_interval,
        v2_2_mode=True,
        prime_banks=prime_banks,
    ).to(device)

    src_v2_2_ckpt = os.path.join(ckpt_dir, "vault_v2_2_best.pt")
    latest_v2_3_ckpt = os.path.join(ckpt_dir, "vault_v2_3_latest.pt")
    best_v2_3_ckpt = os.path.join(ckpt_dir, "vault_v2_3_best.pt")
    telemetry_path = os.path.join(bench_dir, "vault_v2_3_telemetry.json")

    if args.eval_only:
        ck = torch.load(best_v2_3_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        run_8domain_eval_v2_2(model, tok, "VAULT_v2_3_EvalOnly", device, os.path.join(bench_dir, "eval_vault_v2_3_final.json"))
        run_controlled_and_32k_extrapolation_eval(
            model, tok, device, os.path.join(bench_dir, "eval_vault_v2_3_extrapolation_32k.json")
        )
        return

    ck_v2_2 = torch.load(src_v2_2_ckpt, map_location="cpu", weights_only=False)
    mitosis_info = upgrade_v2_2_to_wider_primes(model, ck_v2_2["model_state"])

    # Freeze only `x0_rms_norm` so the isotropic cone & SVD token-copy basis stays anchored;
    # allow all 220 daughter Co-Prime Voronoi cells (`kenyon_proj`, `apl_norm`) to sharpen!
    for name, p in model.named_parameters():
        if "x0_rms_norm" in name:
            p.requires_grad_(False)

    prose_train_ds = NVMeMemmapDataset(data_v2_3_dir, "prose_train", seed=5101)
    prose_val_ds = NVMeMemmapDataset(data_v2_3_dir, "prose_val", seed=5102)
    niah_train_ds = NVMeMemmapDataset(data_v2_3_dir, "niah_train", seed=5103)
    niah_val_ds = NVMeMemmapDataset(data_v2_3_dir, "niah_val", seed=5104)
    sft_train_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_train", seed=4101)
    sft_val_ds = NVMeMemmapDataset(data_v2_2_dir, "sft_val", seed=4102)

    sft_sampler = PartitionedMemmapSampler(sft_train_ds, seed=4101)
    sft_val_sampler = PartitionedMemmapSampler(sft_val_ds, seed=4102)

    val_prose_batches = prose_val_ds.fixed_eval_batches(n_batches=8, batch_size=2, device=device)
    val_niah_batches = niah_val_ds.fixed_eval_batches(n_batches=8, batch_size=2, device=device)
    val_skill_batches, _ = sft_val_sampler.fixed_partition_eval_batches(n_batches=8, batch_size=2, device=device)
    unseen_niah_prompt, unseen_niah_targets = build_unseen_2k_niah_eval()

    tok_per_step = args.micro_batch * args.grad_accum * SEQ_LEN
    n_unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print("=" * 105, flush=True)
    print("VAULT v2.3: 220-CELL CO-PRIME VORONOI MITOSIS + SHARED-GROUP SCAN + BALANCED CONTINUAL TRAINING", flush=True)
    print("=" * 105, flush=True)
    print(f"  Forked From         : {src_v2_2_ckpt} (Protected, Read-Only)", flush=True)
    print(f"  Co-Prime Expansion  : {mitosis_info['old_primes']} ({mitosis_info['old_cells']}c) -> {mitosis_info['new_primes']} ({mitosis_info['new_cells']}c, CRT={mitosis_info['crt_capacity']:,})", flush=True)
    print(f"  Unique Parameters   : {n_unique_params:,} ({n_unique_params / 1e6:.2f}M) | ckpt_interval={args.ckpt_interval}", flush=True)
    print(f"  Stream Mix (4-way)  : 50% Raw Educational Prose ({len(prose_train_ds.tokens):,} rows) | 25% 8-Domain ChatML ({len(sft_sampler.skill_idx):,} rows) | 25% Dense ChatML+1-Shot NIAH ({len(niah_train_ds.tokens):,} rows)", flush=True)
    print(f"  Schedule            : {args.steps} steps ({args.steps * tok_per_step / 1e6:.2f}M tok @ L={SEQ_LEN}), LR {args.peak_lr:.2e} -> {args.min_lr:.2e}", flush=True)
    print("=" * 105, flush=True)

    _, step0_prose_ce = evaluate_val_loss_fast(model, val_prose_batches)
    _, step0_skill_ce = evaluate_val_loss_fast(model, val_skill_batches)
    step0_niah_wce, step0_niah_ce = evaluate_val_loss_fast(model, val_niah_batches)
    step0_bpb = evaluate_bpb(model, tok, device=device)
    print(
        f"[*] Step-0 Mitosis Verification: English_BPB={step0_bpb['bpb']:.4f} | "
        f"prose_CE={step0_prose_ce:.4f} (PPL={math.exp(step0_prose_ce):.2f}) | "
        f"skill_CE={step0_skill_ce:.4f} | niah_CE={step0_niah_ce:.4f} (wCE={step0_niah_wce:.4f})",
        flush=True,
    )

    opt, clip_buckets = build_optimizer_v2_3(model, base_lr=args.peak_lr)
    ind_isolated = clip_buckets["sal"] + clip_buckets["beta"]

    start_step = 0
    best_score = float("inf")
    telemetry = {
        "model": "VAULTLMv2.3",
        "forked_from": src_v2_2_ckpt,
        "prime_banks": list(prime_banks),
        "mitosis_info": mitosis_info,
        "unique_params": n_unique_params,
        "step0": {
            "english_bpb": step0_bpb["bpb"],
            "prose_val_ce": round(step0_prose_ce, 4),
            "skill_val_ce": round(step0_skill_ce, 4),
            "niah_val_ce": round(step0_niah_ce, 4),
            "niah_val_wce": round(step0_niah_wce, 4),
        },
        "history": [],
    }

    if args.resume and os.path.exists(latest_v2_3_ckpt):
        ck = torch.load(latest_v2_3_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        opt.load_state_dict(ck["optimizer_state"])
        start_step = int(ck.get("step", 0))
        best_score = float(ck.get("best_score", float("inf")))
        if os.path.exists(telemetry_path):
            with open(telemetry_path, "r") as f:
                telemetry = json.load(f)
        print(f"[*] Resumed VAULT v2.3 training from step {start_step} (best_score={best_score:.4f})", flush=True)

    model.train()
    t0 = time.perf_counter()
    toks_run = 0
    warmup_steps = max(20, int(0.06 * args.steps))
    last_bpb = step0_bpb["bpb"]

    for step in range(start_step + 1, args.steps + 1):
        if not model.training:
            model.train()
        base_lr = cosine_lr(step, args.steps, warmup_steps, args.peak_lr, args.min_lr)
        set_group_lrs(opt, base_lr)
        opt.zero_grad(set_to_none=True)

        ce_accum = 0.0
        prose_ce_sum = 0.0
        niah_ce_val = 0.0
        beta_val = 0.0
        ediv_val = 0.0
        raux_val = 0.0

        for micro_i in range(args.grad_accum):
            is_niah_micro = (micro_i == args.grad_accum - 1)
            if micro_i in (0, 1):
                bx, by, bw = fetch_masked_batch(prose_train_ds, args.micro_batch, device)
            elif micro_i == 2:
                bx, by, bw = sft_sampler.next_skill_batch(args.micro_batch, device)
            else:
                for p in ind_isolated:
                    p.grad = None
                bx, by, bw = fetch_masked_batch(niah_train_ds, args.micro_batch, device)

            bw_row = normalize_row_weights(bw)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if is_niah_micro:
                    ce_loss, beta_mean, ent_div = model.forward_loss(
                        bx, by, loss_weights=bw_row, step_ratio=1.0, return_beta_mean=True, return_ent_div=True
                    )
                    beta_hinge = args.beta_hinge_coef * F.relu(beta_mean - args.beta_hinge_target)
                    loss = (ce_loss + beta_hinge + args.ent_div_coef * ent_div) / args.grad_accum
                else:
                    ce_loss = model.forward_loss(bx, by, loss_weights=bw_row, step_ratio=1.0)
                    if micro_i == 0 and args.router_aux_coef > 0.0:
                        r_aux = model.compute_router_aux_loss()
                        raux_val = float(r_aux.item())
                        loss = (ce_loss / args.grad_accum) + args.router_aux_coef * r_aux
                    else:
                        loss = ce_loss / args.grad_accum

            loss.backward()

            if micro_i in (0, 1):
                prose_ce_sum += ce_loss.item() * 0.5
            if is_niah_micro:
                niah_ce_val = ce_loss.item()
                beta_val = beta_mean.item()
                ediv_val = ent_div.item()
                for p in ind_isolated:
                    if p.grad is not None:
                        p.grad.mul_(float(args.grad_accum))

            ce_accum += ce_loss.item() / args.grad_accum
            toks_run += bx.numel()

        if not math.isfinite(ce_accum):
            raise RuntimeError(f"Non-finite loss {ce_accum} at step {step}")

        gn_sal, gn_gates, gn_router, gn_base = clip_v2_3_param_groups(clip_buckets, max_norm=1.0)
        opt.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            _, pr_ce = evaluate_val_loss_fast(model, val_prose_batches)
            _, sk_ce = evaluate_val_loss_fast(model, val_skill_batches)
            ni_wce, ni_ce = evaluate_val_loss_fast(model, val_niah_batches)

            do_bpb = (step == 1 or step % args.bpb_interval == 0 or step == args.steps)
            if do_bpb:
                bpb_res = evaluate_bpb(model, tok, device=device)
                last_bpb = bpb_res["bpb"]
            model.train()
            torch.cuda.empty_cache()

            comp_score = 0.45 * last_bpb + 0.25 * sk_ce + 0.30 * ni_wce
            dt = time.perf_counter() - t0
            tps = toks_run / max(dt, 1e-6)

            print(
                f"[v2.3 {step:4d}/{args.steps}] tr={ce_accum:.3f}(p={prose_ce_sum:.3f},n={niah_ce_val:.3f}) | "
                f"BPB={last_bpb:.4f} | pr_PPL={math.exp(min(pr_ce, 15.0)):5.2f} | sk_CE={sk_ce:.3f} | "
                f"ni_CE={ni_ce:.3f}(w={ni_wce:.3f}) | beta={beta_val:.4f} | raux={raux_val:.4f} | "
                f"gn(s={gn_sal:.2f},r={gn_router:.2f},b={gn_base:.2f}) | lr={base_lr:.2e} | {tps:,.0f} tok/s",
                flush=True,
            )

            rec = {
                "step": step,
                "train_loss": round(ce_accum, 4),
                "train_prose_ce": round(prose_ce_sum, 4),
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
                "elapsed_s": round(dt, 1),
            }

            if step % 100 == 0 or step == args.steps:
                p_niah = generate_chatml_v2_2(
                    model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0
                )
                chatml_hits = sum(1 for _, code in unseen_niah_targets if code in p_niah)
                oneshot_probe = evaluate_niah_condition(
                    model, tok, mode="oneshot_base", gap_mode="boilerplate_15tok", n_sections=18, n_trials=4, seed_base=990000, device=device
                )
                model.train()
                torch.cuda.empty_cache()
                rec["probe_chatml_2k"] = f"{chatml_hits}/{len(unseen_niah_targets)}"
                rec["probe_oneshot_2k_acc"] = oneshot_probe["cue_exact_acc"]
                print(
                    f"   [Probe @ {step}] ChatML_2K={chatml_hits}/{len(unseen_niah_targets)} | "
                    f"1Shot_Base_2K={oneshot_probe['cue_exact_acc']*100:.1f}% (NLL={oneshot_probe['mean_code_nll_bits']:.2f}b) | "
                    f"{p_niah.replace(chr(10), ' ⏎ ')[:160]}",
                    flush=True,
                )

            telemetry["history"].append(rec)
            with open(telemetry_path + ".tmp", "w") as f:
                json.dump(telemetry, f, indent=2)
            os.replace(telemetry_path + ".tmp", telemetry_path)

            if do_bpb and step >= min(25, args.steps) and comp_score < best_score:
                best_score = comp_score
                atomic_save_pt(
                    {
                        "stage": "vault_v2_3",
                        "step": step,
                        "prime_banks": prime_banks,
                        "composite_score": comp_score,
                        "english_bpb": last_bpb,
                        "prose_val_ce": pr_ce,
                        "skill_val_ce": sk_ce,
                        "niah_val_ce": ni_ce,
                        "niah_val_wce": ni_wce,
                        "model_state": model.state_dict(),
                    },
                    best_v2_3_ckpt,
                )
                print(
                    f"   [+] Saved new best VAULT v2.3 checkpoint at step {step} "
                    f"(score={comp_score:.4f}, BPB={last_bpb:.4f}, ni_wCE={ni_wce:.4f}) -> {best_v2_3_ckpt}",
                    flush=True,
                )

        if step % args.save_interval == 0 or step == args.steps:
            atomic_save_pt(
                {
                    "stage": "vault_v2_3_latest",
                    "step": step,
                    "prime_banks": prime_banks,
                    "best_score": best_score,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                },
                latest_v2_3_ckpt,
            )

    if os.path.exists(best_v2_3_ckpt):
        ck_best = torch.load(best_v2_3_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck_best["model_state"], strict=True)
        print(
            f"\n[*] Reloaded best VAULT v2.3 checkpoint (step {ck_best.get('step')}, "
            f"BPB={ck_best.get('english_bpb'):.4f}, score={ck_best.get('composite_score'):.4f})",
            flush=True,
        )

    ev_final = run_8domain_eval_v2_2(
        model, tok, "VAULT_v2_3_Final_Best", device, os.path.join(bench_dir, "eval_vault_v2_3_final.json")
    )
    telemetry["final_8domain_summary"] = {
        "niah_exact_hits": ev_final["niah_exact_hits"],
        "math_180_exact": ev_final["math_180_exact"],
        "prose_clean": ev_final["prose_clean_direct_english"],
        "identifier_fidelity": ev_final["identifier_fidelity"],
    }
    if getattr(args, "run_extrap_in_train", False):
        extrap = run_controlled_and_32k_extrapolation_eval(
            model, tok, device, os.path.join(bench_dir, "eval_vault_v2_3_extrapolation_32k.json")
        )
        telemetry["final_extrapolation_summary"] = extrap
    with open(telemetry_path + ".tmp", "w") as f:
        json.dump(telemetry, f, indent=2)
    os.replace(telemetry_path + ".tmp", telemetry_path)

    print(f"ALL VAULT v2.3 TRAINING & 8-DOMAIN EVALUATION COMPLETE! Best Checkpoint -> {best_v2_3_ckpt}", flush=True)


if __name__ == "__main__":
    main()
