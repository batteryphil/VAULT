#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2 (`VAULTLMv2` + `FlashVAULT v2` True GVA):
# Full L=2048 From-Scratch WSD Pretraining + Clean ChatML SFT Engine
#
# Strict IP Scope: Uses ONLY Philip Bartsch's owned IP (`vault`, `PRIME-Moment-Attention`,
# `BioNPC`, `gist-memory`, `buildbit`, `buildbit-os`) + local NVMe memmap shards (`data_v2/`).

import os
import sys
import json
import math
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vault_v2_model import VAULTLMv2


SEQ_LEN = 2048
ROW_LEN = SEQ_LEN + 1


# ==============================================================================
# 1. ZERO-OVERHEAD NVME MEMMAP DATASET SAMPLER
# ==============================================================================

class NVMeMemmapDataset:
    def __init__(self, data_dir: str, prefix: str, seed: int = 1337):
        tok_path = os.path.join(data_dir, f"{prefix}_tokens.bin")
        msk_path = os.path.join(data_dir, f"{prefix}_mask.bin")
        if not os.path.exists(tok_path) or not os.path.exists(msk_path):
            raise FileNotFoundError(f"Missing memmap shard: {tok_path} or {msk_path}")

        n_rows = os.path.getsize(tok_path) // (ROW_LEN * 2)
        self.tokens = np.memmap(tok_path, dtype=np.uint16, mode="r", shape=(n_rows, ROW_LEN))
        self.masks = np.memmap(msk_path, dtype=np.uint8, mode="r", shape=(n_rows, SEQ_LEN))
        self.n_rows = n_rows
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(n_rows)
        self.cursor = 0

    def next_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cursor + batch_size > self.n_rows:
            self.order = self.rng.permutation(self.n_rows)
            self.cursor = 0
        idx = self.order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size

        rows = np.asarray(self.tokens[idx], dtype=np.int64)
        msks = np.asarray(self.masks[idx], dtype=np.float32) * 0.1

        x = torch.from_numpy(rows[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(rows[:, 1:]).to(device, non_blocking=True)
        w = torch.from_numpy(msks).to(device, non_blocking=True)
        return x, y, w

    def fixed_eval_batches(self, n_batches: int, batch_size: int, device: str) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batches = []
        limit = min(self.n_rows, n_batches * batch_size)
        for start in range(0, limit, batch_size):
            rows = np.asarray(self.tokens[start : start + batch_size], dtype=np.int64)
            msks = np.asarray(self.masks[start : start + batch_size], dtype=np.float32) * 0.1
            x = torch.from_numpy(rows[:, :-1]).to(device)
            y = torch.from_numpy(rows[:, 1:]).to(device)
            w = torch.from_numpy(msks).to(device)
            batches.append((x, y, w))
        return batches


# ==============================================================================
# 2. UNSEEN 8-DOMAIN EVALUATION SUITE (100% HELD-OUT TARGETS & NIAH ENTITIES)
# ==============================================================================

UNSEEN_NIAH_PREFIXES = [
    "ShieldPod", "CryoLock", "WarpCore", "IonThruster", "PlasmaGrid", "StarTracker",
    "BeaconMux", "GravSensor", "SolarWing", "HullRadar", "ThermalSink", "DockClamp",
]

UNSEEN_FILLER = (
    "All primary telemetry channels maintained nominal signal-to-noise margin across "
    "redundant optical transceivers during orbital attitude stabilization."
)


def build_unseen_2k_niah_eval() -> Tuple[str, List[Tuple[str, str]]]:
    rng = random.Random(2026102999)
    entries: List[Tuple[str, str]] = []
    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    used_codes = set()

    for idx in range(1, 19):
        stem = UNSEEN_NIAH_PREFIXES[(idx - 1) % len(UNSEEN_NIAH_PREFIXES)]
        ent = f"{stem}-{chr(64 + idx)}{(idx % 9) + 1}"
        while True:
            code = f"CHK-{rng.randint(1000, 9999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        entries.append((ent, code))
        blocks.append(
            f"[Subsystem {idx:02d}: {ent}] Status: NOMINAL | Verification Code: {code}\n"
            f"  Log: {UNSEEN_FILLER} Packet frame {100 + idx} committed without retransmission."
        )

    targets = [entries[4], entries[13]]  # Chunk ~6 and Chunk ~18 -> tests long cross-chunk retention!
    prompt = (
        "\n".join(blocks)
        + f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
        f"for {targets[0][0]} and {targets[1][0]}? List each subsystem and its exact code."
    )
    return prompt, targets


@torch.no_grad()
def generate_chatml(
    model: VAULTLMv2,
    tok,
    user_prompt: str,
    max_new_tokens: int = 200,
    device: str = "cuda:0",
    temperature: float = 0.08,
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
        ctx = ids[:, -4096:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx, step_ratio=1.0)[:, -1, :].float()

        # Mild repetition penalty on generated tokens only
        for prev_id in set(ids[0, prompt_len:].tolist()):
            if logits[0, prev_id] > 0:
                logits[0, prev_id] /= 1.05
            else:
                logits[0, prev_id] *= 1.05

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


def run_8domain_eval(
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
    print(f"VAULT v2 8-DOMAIN UNSEEN EVALUATION [{stage_label}]", flush=True)
    print("=" * 100, flush=True)

    records = []
    for dom, prm in suite:
        temp = 0.0 if "NIAH" in dom or "Math" in dom else 0.08
        out = generate_chatml(model, tok, prm, max_new_tokens=220, device=device, temperature=temp)
        records.append({"domain": dom, "prompt_tail": prm[-300:], "output": out})
        print(f"  [{dom}]\n    -> {out.replace(chr(10), ' ⏎ ')[:270]}", flush=True)

    niah_out = records[-1]["output"]
    niah_hits = sum(1 for _, code in niah_targets if code in niah_out)
    prose_out = records[6]["output"]
    prose_clean = ("<think>" not in prose_out) and ("```" not in prose_out) and ("given code" not in prose_out.lower())
    math_hit = "180" in records[5]["output"]

    summary = {
        "stage": stage_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "niah_expected": niah_targets,
        "niah_exact_hits": f"{niah_hits}/{len(niah_targets)}",
        "math_180_exact": math_hit,
        "prose_clean_direct_english": prose_clean,
        "evaluations": records,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[+] Summary: NIAH={niah_hits}/{len(niah_targets)} | Math(180)={math_hit} "
        f"| DirectProse={prose_clean} -> {out_path}\n",
        flush=True,
    )
    return summary


# ==============================================================================
# 3. PARAMETER GROUPING & WSD LEARNING RATE SCHEDULE
# ==============================================================================

def build_optimizer(model: VAULTLMv2, lr: float, weight_decay: float = 0.1) -> torch.optim.AdamW:
    decay_params = []
    nodecay_params = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and "conv" not in name and "beta_proj" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def get_wsd_lr(step: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    warmup_steps = max(40, int(0.04 * total_steps))
    stable_end = int(0.72 * total_steps)
    if step <= warmup_steps:
        return peak_lr * (step / float(warmup_steps))
    if step <= stable_end:
        return peak_lr
    decay_prog = (step - stable_end) / float(max(1, total_steps - stable_end))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, decay_prog)))


@torch.no_grad()
def evaluate_val_loss(model: VAULTLMv2, batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    tot_w_loss = 0.0
    tot_raw_ce = 0.0
    for vx, vy, vw in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_loss = model.forward_loss(vx, vy, loss_weights=vw, step_ratio=1.0)
            # Also compute unweighted token CE on active tokens (vw > 0) for standard PPL comparison
            active_y = torch.where(vw > 0.0, vy, torch.full_like(vy, -100))
            raw_ce = model.forward_loss(vx, active_y, loss_weights=None, step_ratio=1.0)
        tot_w_loss += w_loss.item()
        tot_raw_ce += raw_ce.item()
    if was_training:
        model.train()
    n = max(1, len(batches))
    return tot_w_loss / n, tot_raw_ce / n


# ==============================================================================
# 4. MAIN TWO-STAGE TRAINING ENGINE (WSD PRETRAIN @ L=2048 -> CLEAN CHATML SFT)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train VAULTLMv2 + FlashVAULT v2 at full L=2048")
    parser.add_argument("--pretrain_steps", type=int, default=6500, help="Stage 1 WSD pretraining steps (default 6500 = 213M tokens @ 32K/step)")
    parser.add_argument("--sft_steps", type=int, default=750, help="Stage 2 Clean ChatML SFT steps (default 750 = 24.6M tokens)")
    parser.add_argument("--micro_batch", type=int, default=2, help="Microbatch size at L=2048")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (2 * 8 * 2048 = 32,768 tok/step)")
    parser.add_argument("--grad_ckpt", type=str, default="half", choices=["none", "half", "three_quarter", "full"])
    parser.add_argument("--peak_lr", type=float, default=6.0e-4, help="Stage 1 WSD peak LR")
    parser.add_argument("--min_lr", type=float, default=3.0e-5, help="Stage 1 WSD end-of-decay LR")
    parser.add_argument("--sft_lr", type=float, default=4.5e-5, help="Stage 2 SFT peak LR")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--probe_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/vault_v2_latest.pt if present")
    parser.add_argument("--sft_only", action="store_true", help="Skip Stage 1 and run Stage 2 SFT from vault_v2_pretrain_best.pt")
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data_v2")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    gc_map = {"none": False, "half": "half", "three_quarter": "three_quarter", "full": True}

    model = VAULTLMv2(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint=gc_map[args.grad_ckpt],
    ).to(device)

    tok_per_step = args.micro_batch * args.grad_accum * SEQ_LEN
    print("=" * 100, flush=True)
    print("VAULT v2 (`VAULTLMv2` + `FlashVAULT v2` True 3:1 GVA): FULL L=2048 WSD PRETRAIN + CHATML SFT", flush=True)
    print("=" * 100, flush=True)
    print(f"  Device & Precision  : {device} ({torch.cuda.get_device_name(0)}) | BF16 RDNA4 WMMA", flush=True)
    print(f"  Model Parameters    : {model.count_parameters():,} (d=768, L=16, H_q=12, G_vault=4, ShortConv1D=4, RoPE-Local=128)", flush=True)
    print(f"  Batch Geometry      : B={args.micro_batch} x accum={args.grad_accum} x L={SEQ_LEN} = {tok_per_step:,} tok/step (gc={args.grad_ckpt})", flush=True)
    print(f"  Stage 1 WSD Budget  : {args.pretrain_steps:,} steps ({args.pretrain_steps * tok_per_step / 1e6:.2f}M tokens, LR {args.peak_lr:.1e} -> {args.min_lr:.1e})", flush=True)
    print(f"  Stage 2 SFT Budget  : {args.sft_steps:,} steps ({args.sft_steps * tok_per_step / 1e6:.2f}M tokens, LR {args.sft_lr:.1e} + 25% Pretrain Replay)", flush=True)
    print("=" * 100, flush=True)

    train_ds = NVMeMemmapDataset(data_dir, "train", seed=2026)
    val_ds = NVMeMemmapDataset(data_dir, "val", seed=9001)
    sft_train_ds = NVMeMemmapDataset(data_dir, "sft_train", seed=4040)
    sft_val_ds = NVMeMemmapDataset(data_dir, "sft_val", seed=9002)

    val_pre_batches = val_ds.fixed_eval_batches(n_batches=12, batch_size=2, device=device)
    val_sft_batches = sft_val_ds.fixed_eval_batches(n_batches=12, batch_size=2, device=device)
    unseen_niah_prompt, unseen_niah_targets = build_unseen_2k_niah_eval()

    latest_ckpt = os.path.join(ckpt_dir, "vault_v2_latest.pt")
    wsd_stable_ckpt = os.path.join(ckpt_dir, "vault_v2_wsd_stable.pt")
    pretrain_best_ckpt = os.path.join(ckpt_dir, "vault_v2_pretrain_best.pt")
    sft_best_ckpt = os.path.join(ckpt_dir, "vault_v2_sft_best.pt")
    telemetry_path = os.path.join(bench_dir, "vault_v2_telemetry.json")

    telemetry = {
        "model": "VAULTLMv2",
        "params": model.count_parameters(),
        "seq_len": SEQ_LEN,
        "tok_per_step": tok_per_step,
        "pretrain_history": [],
        "sft_history": [],
        "milestone_evals": [],
    }
    if os.path.exists(telemetry_path) and (args.resume or args.sft_only):
        try:
            with open(telemetry_path, "r") as f:
                telemetry = json.load(f)
        except Exception:
            pass

    # ==========================================================================
    # STAGE 1: FULL L=2048 WSD PRETRAINING FROM STEP 0
    # ==========================================================================
    if not args.sft_only and args.pretrain_steps > 0:
        opt1 = build_optimizer(model, lr=args.peak_lr, weight_decay=0.1)
        start_step = 0
        best_val_ce = float("inf")

        if args.resume and os.path.exists(latest_ckpt):
            ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if ck.get("stage") == "pretrain":
                model.load_state_dict(ck["model_state"])
                opt1.load_state_dict(ck["optimizer_state"])
                start_step = ck.get("step", 0)
                best_val_ce = ck.get("best_val_ce", float("inf"))
                train_ds.cursor = ck.get("dataset_cursor", 0)
                print(f"[*] Resumed Stage 1 Pretraining at step {start_step:,} (best_val_ce={best_val_ce:.4f})", flush=True)

        model.train()
        t0 = time.perf_counter()
        toks_run = 0
        stable_boundary = int(0.72 * args.pretrain_steps)
        router_anneal_steps = max(400, int(0.30 * args.pretrain_steps))

        for step in range(start_step + 1, args.pretrain_steps + 1):
            lr = get_wsd_lr(step, args.pretrain_steps, args.peak_lr, args.min_lr)
            for pg in opt1.param_groups:
                pg["lr"] = lr
            step_ratio = min(1.0, step / float(router_anneal_steps))

            opt1.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum):
                bx, by, bw = train_ds.next_batch(args.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.forward_loss(bx, by, loss_weights=bw, step_ratio=step_ratio) / args.grad_accum
                loss.backward()
                loss_accum += loss.item()
                toks_run += bx.numel()

            if not math.isfinite(loss_accum):
                raise RuntimeError(f"Non-finite pretrain loss {loss_accum} at step {step}")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt1.step()

            if step == 1 or step % args.log_interval == 0 or step == args.pretrain_steps:
                val_w_loss, val_ce = evaluate_val_loss(model, val_pre_batches)
                val_ppl = math.exp(min(val_ce, 15.0))
                dt = time.perf_counter() - t0
                tps = toks_run / max(dt, 1e-6)
                tot_tok_m = step * tok_per_step / 1e6

                print(
                    f"[Pretrain {step:5d}/{args.pretrain_steps} | {tot_tok_m:6.2f}M tok] "
                    f"train={loss_accum:.4f} | val_ce={val_ce:.4f} (PPL={val_ppl:6.2f}) | "
                    f"gnorm={grad_norm:.2f} | lr={lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )

                rec = {
                    "step": step,
                    "tokens_m": round(tot_tok_m, 3),
                    "train_loss": round(loss_accum, 4),
                    "val_w_loss": round(val_w_loss, 4),
                    "val_ce": round(val_ce, 4),
                    "val_ppl": round(val_ppl, 2),
                    "grad_norm": round(grad_norm, 3),
                    "lr": lr,
                    "tok_s": round(tps, 1),
                }

                if step % args.probe_interval == 0:
                    p_prose = generate_chatml(model, tok, "Explain why recursion needs a base case in programming.", max_new_tokens=96, device=device, temperature=0.08)
                    p_niah = generate_chatml(model, tok, unseen_niah_prompt, max_new_tokens=64, device=device, temperature=0.0)
                    rec["probe_prose"] = p_prose
                    rec["probe_niah"] = p_niah
                    print(f"   [Probe Prose] {p_prose.replace(chr(10), ' ⏎ ')[:190]}", flush=True)
                    print(f"   [Probe 2K NIAH {unseen_niah_targets}] {p_niah.replace(chr(10), ' ⏎ ')[:190]}", flush=True)

                telemetry["pretrain_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if val_ce < best_val_ce and step >= 50:
                    best_val_ce = val_ce
                    torch.save(
                        {"step": step, "tokens": step * tok_per_step, "val_ce": val_ce, "model_state": model.state_dict()},
                        pretrain_best_ckpt,
                    )

            if step == stable_boundary:
                torch.save(
                    {
                        "stage": "pretrain",
                        "step": step,
                        "tokens": step * tok_per_step,
                        "best_val_ce": best_val_ce,
                        "dataset_cursor": train_ds.cursor,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt1.state_dict(),
                    },
                    wsd_stable_ckpt,
                )
                print(f"  [+] Saved WSD Stable-Plateau Checkpoint -> {wsd_stable_ckpt}", flush=True)

            if step % args.save_interval == 0 or step == args.pretrain_steps:
                torch.save(
                    {
                        "stage": "pretrain",
                        "step": step,
                        "tokens": step * tok_per_step,
                        "best_val_ce": best_val_ce,
                        "dataset_cursor": train_ds.cursor,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt1.state_dict(),
                    },
                    latest_ckpt,
                )

            if step % args.eval_interval == 0 or step == args.pretrain_steps:
                m_tag = f"pretrain_step_{step}_{int(round(step * tok_per_step / 1e6))}M"
                ev_path = os.path.join(bench_dir, f"eval_vault_v2_{m_tag}.json")
                ev_sum = run_8domain_eval(model, tok, m_tag, device, ev_path)
                telemetry["milestone_evals"].append({
                    "stage": m_tag,
                    "niah_exact_hits": ev_sum["niah_exact_hits"],
                    "math_180_exact": ev_sum["math_180_exact"],
                    "prose_clean_direct_english": ev_sum["prose_clean_direct_english"],
                })
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

    elif args.sft_only:
        load_p = pretrain_best_ckpt if os.path.exists(pretrain_best_ckpt) else latest_ckpt
        ck = torch.load(load_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"[*] Loaded Stage 1 base model from {load_p} (step={ck.get('step')})", flush=True)

    # ==========================================================================
    # STAGE 2: GENTLE CLEAN CHATML SFT (+ 25% STAGE-1 PRETRAIN REPLAY)
    # ==========================================================================
    if args.sft_steps > 0:
        print("\n>>> STARTING STAGE 2: CLEAN CHATML SFT (STRICT PROMPT MASKING + 25% PRETRAIN REPLAY)...", flush=True)
        opt2 = build_optimizer(model, lr=args.sft_lr, weight_decay=0.02)
        best_sft_ce = float("inf")
        model.train()
        t_sft = time.perf_counter()
        toks_sft = 0
        sft_warmup = max(25, int(0.06 * args.sft_steps))

        for step in range(1, args.sft_steps + 1):
            if step <= sft_warmup:
                lr = args.sft_lr * (step / float(sft_warmup))
            else:
                prog = (step - sft_warmup) / float(max(1, args.sft_steps - sft_warmup))
                lr = 5.0e-6 + 0.5 * (args.sft_lr - 5.0e-6) * (1.0 + math.cos(math.pi * min(1.0, prog)))
            for pg in opt2.param_groups:
                pg["lr"] = lr

            opt2.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for micro_i in range(args.grad_accum):
                # Every 4th microbatch replays clean Stage-1 Textbooks/Code/Math/NIAH to prevent forgetting!
                if micro_i % 4 == 3:
                    bx, by, bw = train_ds.next_batch(args.micro_batch, device)
                else:
                    bx, by, bw = sft_train_ds.next_batch(args.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.forward_loss(bx, by, loss_weights=bw, step_ratio=1.0) / args.grad_accum
                loss.backward()
                loss_accum += loss.item()
                toks_sft += bx.numel()

            if not math.isfinite(loss_accum):
                raise RuntimeError(f"Non-finite SFT loss {loss_accum} at step {step}")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8).item()
            opt2.step()

            if step == 1 or step % args.log_interval == 0 or step == args.sft_steps:
                sft_w_loss, sft_ce = evaluate_val_loss(model, val_sft_batches)
                sft_ppl = math.exp(min(sft_ce, 15.0))
                dt = time.perf_counter() - t_sft
                tps = toks_sft / max(dt, 1e-6)

                print(
                    f"[SFT {step:4d}/{args.sft_steps}] train={loss_accum:.4f} | "
                    f"sft_val_ce={sft_ce:.4f} (PPL={sft_ppl:5.2f}) | gnorm={grad_norm:.2f} | "
                    f"lr={lr:.2e} | {tps:,.0f} tok/s",
                    flush=True,
                )
                rec = {
                    "step": step,
                    "train_loss": round(loss_accum, 4),
                    "sft_val_ce": round(sft_ce, 4),
                    "sft_val_ppl": round(sft_ppl, 2),
                    "lr": lr,
                    "tok_s": round(tps, 1),
                }
                telemetry["sft_history"].append(rec)
                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

                if sft_ce < best_sft_ce or step == args.sft_steps:
                    best_sft_ce = min(best_sft_ce, sft_ce)
                    torch.save(
                        {"stage": "sft", "step": step, "sft_val_ce": sft_ce, "model_state": model.state_dict()},
                        sft_best_ckpt,
                    )

        final_eval_path = os.path.join(bench_dir, "eval_vault_v2_final.json")
        run_8domain_eval(model, tok, f"VAULT_v2_Final_SFT_Step_{args.sft_steps}", device, final_eval_path)
        print(f"ALL VAULT v2 TRAINING & EVALUATION COMPLETE! Best SFT Checkpoint -> {sft_best_ckpt}", flush=True)


if __name__ == "__main__":
    main()
