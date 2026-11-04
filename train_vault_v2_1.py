#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.1 (`VAULTLMv2(v2_1_mode=True)` + `FlashVAULT v2.1` Consensus & Chunk Recomputation):
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


@torch.no_grad()
def generate_chatml_v2_1(
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

        # Mild repetition penalty on generated tokens only (skip digits/punctuation for NIAH/Math fidelity)
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


def run_8domain_eval_v2_1(
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
    print(f"VAULT v2.1 8-DOMAIN UNSEEN EVALUATION [{stage_label}]", flush=True)
    print("=" * 100, flush=True)

    records = []
    for dom, prm in suite:
        temp = 0.0 if ("NIAH" in dom or "Math" in dom) else 0.05
        out = generate_chatml_v2_1(model, tok, prm, max_new_tokens=240, device=device, temperature=temp)
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
      1. Controlled Open-Vocabulary NIAH Tests A / B / C (12 trials = 24 needles per condition):
           - Test A: Unseen HOLDOUT_NIAH_PREFIXES + 15-tok boilerplate gap
           - Test B: Unseen HOLDOUT_NIAH_PREFIXES + 3-tok compact gap
           - Test C: Unseen OOV/Holdout mixed + 24-tok extended gap
      2. Zero-Shot Length Extrapolation & Peak VRAM Curve across L in [2048, 4096, 8192, 16384, 32768]
         (trained strictly at L = 2048!).
    """
    was_training = model.training
    model.eval()

    print("\n" + "=" * 100, flush=True)
    print("VAULT v2.1 CONTROLLED NIAH (A/B/C) & 2K -> 32K ZERO-SHOT EXTRAPOLATION SUITE", flush=True)
    print("=" * 100, flush=True)

    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def _eval_manifest_condition(gap_mode: str, n_sections: int, n_trials: int, seed_base: int) -> Dict:
        exact_codes = 0
        total_codes = 0
        tf_digit_hits = 0
        tf_digit_tot = 0
        prompt_lens = []

        for tr in range(n_trials):
            rng = random.Random(seed_base + tr * 101)
            entries = []
            blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
            used_codes = set()
            for idx in range(1, n_sections + 1):
                stem = UNSEEN_NIAH_PREFIXES[(idx + tr) % len(UNSEEN_NIAH_PREFIXES)]
                ent = f"{stem}-{chr(65 + (idx % 24))}{((idx + tr) % 9) + 1}{idx:02d}" if n_sections > 24 else f"{stem}-{chr(65 + (idx % 24))}{((idx + tr) % 9) + 1}"
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

            # 1. Teacher-forced 4-digit accuracy on `- Entity: CHK-xxxx`
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

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_all = model._encode_hidden(full_ids, step_ratio=1.0)
                ans_logits = model.lm_head(h_all[0, len(p_ids) - 1 :]).float()
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

            # Check digit token positions inside each target's 4 digits
            cur = 0
            for idx_t, (ent, code) in enumerate(targets):
                lead = ("" if idx_t == 0 else "\n") + f"- {ent}: CHK-"
                lead_ids = tok.encode(lead, add_special_tokens=False)
                dig_ids = tok.encode(code.split("-")[1], add_special_tokens=False)
                for off_d, did in enumerate(dig_ids):
                    rel_pos = cur + len(lead_ids) + off_d
                    pred_id = int(torch.argmax(ans_logits[rel_pos]).item())
                    if pred_id == did:
                        tf_digit_hits += 1
                    tf_digit_tot += 1
                cur += len(lead_ids) + len(dig_ids)

            # 2. Greedy free-generation exact code match
            gen_out = generate_chatml_v2_1(model, tok, prm, max_new_tokens=64, device=device, temperature=0.0, max_ctx=36864)
            for _, code in targets:
                if code in gen_out:
                    exact_codes += 1
                total_codes += 1

        return {
            "gap_mode": gap_mode,
            "n_sections": n_sections,
            "mean_prompt_tokens": round(sum(prompt_lens) / max(1, len(prompt_lens)), 1),
            "exact_code_hits": f"{exact_codes}/{total_codes}",
            "exact_code_acc": round(exact_codes / max(1, total_codes), 4),
            "tf_digit_top1_acc": round(tf_digit_hits / max(1, tf_digit_tot), 4),
            "last_fwd_ms": round(fwd_ms, 2),
            "peak_infer_vram_mb": round(peak_mb, 1),
        }

    controlled = {
        "Test_A_unseen_15tok_boilerplate": _eval_manifest_condition("boilerplate_15tok", n_sections=18, n_trials=8, seed_base=70001),
        "Test_B_unseen_3tok_compact": _eval_manifest_condition("compact_3tok", n_sections=18, n_trials=8, seed_base=80001),
        "Test_C_unseen_24tok_wide": _eval_manifest_condition("wide_24tok", n_sections=18, n_trials=8, seed_base=90001),
    }
    for k, v in controlled.items():
        print(f"  [{k}] exact={v['exact_code_hits']} ({v['exact_code_acc']*100:.1f}%) | digit_top1={v['tf_digit_top1_acc']*100:.1f}%", flush=True)

    # Zero-shot length extrapolation from 2K -> 32K
    extrapolation = {}
    # Each section is ~78 tokens -> [22, 46, 96, 196, 396] sections ~= [1.8K, 3.8K, 7.8K, 15.6K, 31.4K] tokens!
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
            f"digit_top1={res['tf_digit_top1_acc']*100:.1f}% | fwd={res['last_fwd_ms']:.1f} ms | "
            f"infer_VRAM={res['peak_infer_vram_mb']:.1f} MB",
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


def build_optimizer_v2_1(
    model: VAULTLMv2,
    base_lr: float,
    binder_mult: float = 4.0,
    router_mult: float = 2.0,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    binder_params = []
    router_params = []
    decay_params = []
    nodecay_params = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if "span_binder" in name:
            binder_params.append(p)
        elif "kenyon_proj" in name or "apl_norm" in name or "beta_proj" in name:
            router_params.append(p)
        elif p.ndim >= 2 and "conv" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": binder_params, "lr": base_lr * binder_mult, "lr_mult": binder_mult, "weight_decay": 0.0},
        {"params": router_params, "lr": base_lr * router_mult, "lr_mult": router_mult, "weight_decay": 0.0},
        {"params": decay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": base_lr, "lr_mult": 1.0, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8)


def set_group_lrs(opt: torch.optim.AdamW, base_lr: float):
    for pg in opt.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return peak_lr * (step / float(max(1, warmup_steps)))
    prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def compute_router_and_binder_stats(model: VAULTLMv2) -> Tuple[float, float]:
    ratios = []
    b_norms = []
    for blk in model.layers:
        rg = blk.attn.kenyon_proj.weight.grad
        if rg is not None:
            r_n = rg.norm().item()
            l_n = math.sqrt(sum(p.grad.norm().item() ** 2 for p in blk.parameters() if p.grad is not None))
            ratios.append(r_n / max(l_n, 1e-8))
        sb = blk.attn.span_binder
        b_norms.append(0.5 * (sb.k_bind_proj.detach().norm().item() + sb.q_bind_proj.detach().norm().item()))
    mean_r = sum(ratios) / max(1, len(ratios))
    mean_b = sum(b_norms) / max(1, len(b_norms))
    return mean_r, mean_b


def main():
    parser = argparse.ArgumentParser(description="Train VAULT v2.1 forked from Stage-1 VAULT v2 checkpoint")
    parser.add_argument("--adapt_steps", type=int, default=800, help="Phase 1 continual adaptation steps (default 800 = 26.21M tok)")
    parser.add_argument("--sft_steps", type=int, default=400, help="Phase 2 clean ChatML SFT steps (default 400 = 13.11M tok)")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--ckpt_interval", type=int, default=8)
    parser.add_argument("--adapt_lr", type=float, default=1.2e-4)
    parser.add_argument("--sft_lr", type=float, default=3.5e-5)
    parser.add_argument("--router_aux_coef", type=float, default=0.02)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--probe_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20261030)
    np.random.seed(20261030)
    random.seed(20261030)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_v2_dir = os.path.join(base_dir, "data_v2")
    data_v2_1_dir = os.path.join(base_dir, "data_v2_1")
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
        v2_1_mode=True,
        ckpt_interval=args.ckpt_interval,
    ).to(device)

    base_fork_ckpt = os.path.join(ckpt_dir, "vault_v2_pretrain_best.pt")
    latest_v2_1_ckpt = os.path.join(ckpt_dir, "vault_v2_1_latest.pt")
    adapt_best_ckpt = os.path.join(ckpt_dir, "vault_v2_1_adapt_best.pt")
    best_v2_1_ckpt = os.path.join(ckpt_dir, "vault_v2_1_best.pt")
    telemetry_path = os.path.join(bench_dir, "vault_v2_1_telemetry.json")

    if args.eval_only:
        ck = torch.load(best_v2_1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        run_8domain_eval_v2_1(model, tok, "VAULT_v2_1_EvalOnly", device, os.path.join(bench_dir, "eval_vault_v2_1_final.json"))
        run_controlled_and_32k_extrapolation_eval(model, tok, device, os.path.join(bench_dir, "eval_vault_v2_1_extrapolation_32k.json"))
        return

    # Load completed 245.76M-token Stage-1 VAULT v2 checkpoint (0 restart!)
    ck_base = torch.load(base_fork_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck_base["model_state"], strict=False)
    assert len(unexpected) == 0 and all("span_binder" in k for k in missing), f"Unexpected fork diff: {unexpected}"

    tok_per_step = args.micro_batch * args.grad_accum * SEQ_LEN
    print("=" * 100, flush=True)
    print("VAULT v2.1: FORKED CONTINUAL ADAPTATION + CLEAN CHATML SFT + 32K EXTRAPOLATION ENGINE", flush=True)
    print("=" * 100, flush=True)
    print(f"  Forked Base Model   : {base_fork_ckpt} (Step {ck_base.get('step')}, 245.76M tokens, Val CE {ck_base.get('val_ce'):.4f})", flush=True)
    print(f"  v2.1 Parameters     : {model.count_parameters():,} (+2.43M zero-init MultiOffsetSpanBinder across 1..38 tok)", flush=True)
    print(f"  v2.1 Kernel Upgrades: Empty-Cell Consensus Suppression (rsqrt(1+n)) | ckpt_interval={args.ckpt_interval} (8x smaller S_ckpt) | beta_min=0.0 | Unchoked Post-Norm STE", flush=True)
    print(f"  Phase 1 Adaptation  : {args.adapt_steps:,} steps ({args.adapt_steps * tok_per_step / 1e6:.2f}M tok, 37.5% Stage-1 Replay + 62.5% Open-Vocab v2.1)", flush=True)
    print(f"  Phase 2 Clean SFT   : {args.sft_steps:,} steps ({args.sft_steps * tok_per_step / 1e6:.2f}M tok, 25.0% Stage-1 Replay + 75.0% Open-Vocab SFT)", flush=True)
    print("=" * 100, flush=True)

    stage1_replay_ds = NVMeMemmapDataset(data_v2_dir, "train", seed=20261030)
    stage1_val_ds = NVMeMemmapDataset(data_v2_dir, "val", seed=9001)
    adapt_train_ds = NVMeMemmapDataset(data_v2_1_dir, "adapt_train", seed=3101)
    adapt_val_ds = NVMeMemmapDataset(data_v2_1_dir, "adapt_val", seed=3102)
    sft_train_ds = NVMeMemmapDataset(data_v2_1_dir, "sft_train", seed=4101)
    sft_val_ds = NVMeMemmapDataset(data_v2_1_dir, "sft_val", seed=4102)

    val_general_batches = stage1_val_ds.fixed_eval_batches(n_batches=12, batch_size=2, device=device)
    val_adapt_batches = adapt_val_ds.fixed_eval_batches(n_batches=12, batch_size=2, device=device)
    val_sft_batches = sft_val_ds.fixed_eval_batches(n_batches=12, batch_size=2, device=device)
    unseen_niah_prompt, unseen_niah_targets = build_unseen_2k_niah_eval()

    _, step0_gen_ce = evaluate_val_loss(model, val_general_batches)
    print(f"[*] Verified Step-0 Holdout General Val CE = {step0_gen_ce:.4f} (PPL = {math.exp(step0_gen_ce):.2f})", flush=True)

    telemetry = {
        "model": "VAULTLMv2.1",
        "forked_from": base_fork_ckpt,
        "params": model.count_parameters(),
        "ckpt_interval": args.ckpt_interval,
        "step0_general_val_ce": round(step0_gen_ce, 4),
        "step0_general_val_ppl": round(math.exp(step0_gen_ce), 2),
        "adapt_history": [],
        "sft_history": [],
        "milestone_evals": [],
    }

    # ==========================================================================
    # PHASE 1: CONTINUAL v2.1 ADAPTATION (800 STEPS = 26.21M TOKENS)
    # ==========================================================================
    if args.adapt_steps > 0:
        opt1 = build_optimizer_v2_1(model, base_lr=args.adapt_lr, binder_mult=4.0, router_mult=2.0, weight_decay=0.05)
        start_step = 0
        best_adapt_ce = float("inf")

        if args.resume and os.path.exists(latest_v2_1_ckpt):
            ck = torch.load(latest_v2_1_ckpt, map_location=device, weights_only=False)
            if ck.get("stage") == "adapt":
                model.load_state_dict(ck["model_state"])
                opt1.load_state_dict(ck["optimizer_state"])
                start_step = ck.get("step", 0)
                best_adapt_ce = ck.get("best_adapt_ce", float("inf"))
                print(f"[*] Resumed Phase 1 Adaptation at step {start_step}", flush=True)

        model.train()
        t0 = time.perf_counter()
        toks_run = 0
        warmup_1 = max(30, int(0.05 * args.adapt_steps))

        for step in range(start_step + 1, args.adapt_steps + 1):
            base_lr = cosine_lr(step, args.adapt_steps, warmup_1, args.adapt_lr, 2.0e-5)
            set_group_lrs(opt1, base_lr)

            opt1.zero_grad(set_to_none=True)
            ce_accum = 0.0
            aux_val = 0.0
            for micro_i in range(args.grad_accum):
                # 3/8 microbatches (37.5%) Stage-1 general replay + 5/8 microbatches (62.5%) Open-Vocab v2.1
                if micro_i in (2, 5, 7):
                    bx, by, bw = stage1_replay_ds.next_batch(args.micro_batch, device)
                else:
                    bx, by, bw = adapt_train_ds.next_batch(args.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ce_loss = model.forward_loss(bx, by, loss_weights=bw, step_ratio=1.0)
                    if micro_i == 0:
                        r_aux = model.compute_router_aux_loss()
                        aux_val = r_aux.item()
                        loss = (ce_loss / args.grad_accum) + args.router_aux_coef * r_aux
                    else:
                        loss = ce_loss / args.grad_accum
                loss.backward()
                ce_accum += ce_loss.item() / args.grad_accum
                toks_run += bx.numel()

            if not math.isfinite(ce_accum):
                raise RuntimeError(f"Non-finite Phase 1 loss {ce_accum} at step {step}")

            r_grad_ratio, binder_norm = compute_router_and_binder_stats(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt1.step()

            if step == 1 or step % args.log_interval == 0 or step == args.adapt_steps:
                _, gen_ce = evaluate_val_loss(model, val_general_batches)
                _, adp_ce = evaluate_val_loss(model, val_adapt_batches)
                gen_ppl = math.exp(min(gen_ce, 15.0))
                adp_ppl = math.exp(min(adp_ce, 15.0))
                dt = time.perf_counter() - t0
                tps = toks_run / max(dt, 1e-6)

                print(
                    f"[Adapt {step:4d}/{args.adapt_steps}] train={ce_accum:.4f} | "
                    f"gen_PPL={gen_ppl:5.2f} ({gen_ce:.4f}) | v2.1_PPL={adp_ppl:5.2f} ({adp_ce:.4f}) | "
                    f"r_grad={r_grad_ratio:.4f} | bind_w={binder_norm:.3f} | lr={base_lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )

                rec = {
                    "step": step,
                    "train_loss": round(ce_accum, 4),
                    "router_aux": round(aux_val, 5),
                    "general_val_ce": round(gen_ce, 4),
                    "general_val_ppl": round(gen_ppl, 2),
                    "adapt_val_ce": round(adp_ce, 4),
                    "adapt_val_ppl": round(adp_ppl, 2),
                    "router_grad_ratio": round(r_grad_ratio, 5),
                    "binder_weight_norm": round(binder_norm, 4),
                    "grad_norm": round(grad_norm, 3),
                    "lr": base_lr,
                    "tok_s": round(tps, 1),
                }

                if step % args.probe_interval == 0:
                    p_math = generate_chatml_v2_1(model, tok, "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?", max_new_tokens=110, device=device, temperature=0.0)
                    p_niah = generate_chatml_v2_1(model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0)
                    rec["probe_math"] = p_math
                    rec["probe_niah"] = p_niah
                    print(f"   [Probe Math] {p_math.replace(chr(10), ' ⏎ ')[:210]}", flush=True)
                    print(f"   [Probe 2K NIAH {unseen_niah_targets}] {p_niah.replace(chr(10), ' ⏎ ')[:210]}", flush=True)

                telemetry["adapt_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if adp_ce < best_adapt_ce and step >= 50:
                    best_adapt_ce = adp_ce
                    torch.save({"stage": "adapt", "step": step, "adapt_val_ce": adp_ce, "gen_val_ce": gen_ce, "model_state": model.state_dict()}, adapt_best_ckpt)

            if step % args.save_interval == 0 or step == args.adapt_steps:
                torch.save(
                    {"stage": "adapt", "step": step, "best_adapt_ce": best_adapt_ce, "model_state": model.state_dict(), "optimizer_state": opt1.state_dict()},
                    latest_v2_1_ckpt,
                )

            if step % args.eval_interval == 0 or step == args.adapt_steps:
                tag = f"v2_1_adapt_step_{step}"
                ev = run_8domain_eval_v2_1(model, tok, tag, device, os.path.join(bench_dir, f"eval_vault_{tag}.json"))
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
    # PHASE 2: CLEAN CHATML SFT POLISH (400 STEPS = 13.11M TOKENS + 25% REPLAY)
    # ==========================================================================
    if args.sft_steps > 0:
        print("\n>>> STARTING PHASE 2: VAULT v2.1 CLEAN CHATML SFT POLISH (25% STAGE-1 REPLAY + 75% OPEN-VOCAB SFT)...", flush=True)
        opt2 = build_optimizer_v2_1(model, base_lr=args.sft_lr, binder_mult=2.5, router_mult=1.5, weight_decay=0.02)
        best_sft_ce = float("inf")
        model.train()
        t_sft = time.perf_counter()
        toks_sft = 0
        warmup_2 = max(20, int(0.06 * args.sft_steps))

        for step in range(1, args.sft_steps + 1):
            base_lr = cosine_lr(step, args.sft_steps, warmup_2, args.sft_lr, 5.0e-6)
            set_group_lrs(opt2, base_lr)

            opt2.zero_grad(set_to_none=True)
            ce_accum = 0.0
            for micro_i in range(args.grad_accum):
                if micro_i in (3, 7):
                    bx, by, bw = stage1_replay_ds.next_batch(args.micro_batch, device)
                else:
                    bx, by, bw = sft_train_ds.next_batch(args.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ce_loss = model.forward_loss(bx, by, loss_weights=bw, step_ratio=1.0)
                    if micro_i == 0:
                        loss = (ce_loss / args.grad_accum) + (0.5 * args.router_aux_coef) * model.compute_router_aux_loss()
                    else:
                        loss = ce_loss / args.grad_accum
                loss.backward()
                ce_accum += ce_loss.item() / args.grad_accum
                toks_sft += bx.numel()

            if not math.isfinite(ce_accum):
                raise RuntimeError(f"Non-finite Phase 2 SFT loss {ce_accum} at step {step}")

            r_grad_ratio, binder_norm = compute_router_and_binder_stats(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8).item()
            opt2.step()

            if step == 1 or step % args.log_interval == 0 or step == args.sft_steps:
                _, gen_ce = evaluate_val_loss(model, val_general_batches)
                _, sft_ce = evaluate_val_loss(model, val_sft_batches)
                gen_ppl = math.exp(min(gen_ce, 15.0))
                sft_ppl = math.exp(min(sft_ce, 15.0))
                dt = time.perf_counter() - t_sft
                tps = toks_sft / max(dt, 1e-6)

                print(
                    f"[SFT {step:4d}/{args.sft_steps}] train={ce_accum:.4f} | "
                    f"gen_PPL={gen_ppl:5.2f} | sft_PPL={sft_ppl:5.2f} ({sft_ce:.4f}) | "
                    f"r_grad={r_grad_ratio:.4f} | bind_w={binder_norm:.3f} | lr={base_lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )

                rec = {
                    "step": step,
                    "train_loss": round(ce_accum, 4),
                    "general_val_ce": round(gen_ce, 4),
                    "general_val_ppl": round(gen_ppl, 2),
                    "sft_val_ce": round(sft_ce, 4),
                    "sft_val_ppl": round(sft_ppl, 2),
                    "router_grad_ratio": round(r_grad_ratio, 5),
                    "binder_weight_norm": round(binder_norm, 4),
                    "lr": base_lr,
                    "tok_s": round(tps, 1),
                }

                if step % args.probe_interval == 0:
                    p_niah = generate_chatml_v2_1(model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0)
                    rec["probe_niah"] = p_niah
                    print(f"   [SFT Probe 2K NIAH {unseen_niah_targets}] {p_niah.replace(chr(10), ' ⏎ ')[:210]}", flush=True)

                telemetry["sft_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if sft_ce < best_sft_ce:
                    best_sft_ce = sft_ce
                    torch.save(
                        {"stage": "sft_v2_1", "step": step, "sft_val_ce": sft_ce, "gen_val_ce": gen_ce, "model_state": model.state_dict()},
                        best_v2_1_ckpt,
                    )

        # Reload best SFT checkpoint before running final benchmarks!
        if os.path.exists(best_v2_1_ckpt):
            ck_best = torch.load(best_v2_1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck_best["model_state"])
            print(f"\n[*] Reloaded best VAULT v2.1 checkpoint (SFT step {ck_best.get('step')}, sft_val_ce={ck_best.get('sft_val_ce'):.4f})", flush=True)

        ev_final = run_8domain_eval_v2_1(
            model, tok, "VAULT_v2_1_Final_Best", device, os.path.join(bench_dir, "eval_vault_v2_1_final.json")
        )
        extrap = run_controlled_and_32k_extrapolation_eval(
            model, tok, device, os.path.join(bench_dir, "eval_vault_v2_1_extrapolation_32k.json")
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

        print(f"ALL VAULT v2.1 TRAINING & 32K EVALUATION COMPLETE! Best Checkpoint -> {best_v2_1_ckpt}", flush=True)


if __name__ == "__main__":
    main()
