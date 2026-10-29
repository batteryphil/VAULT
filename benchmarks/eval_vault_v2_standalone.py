#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Standalone 8-Domain Unseen Evaluation + O(L) Long-Context Scaling Audit for VAULT v2

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vault_v2_model import VAULTLMv2
from train_vault_v2 import run_8domain_eval


@torch.no_grad()
def bench_long_context_scaling(model: VAULTLMv2, device: str = "cuda:0") -> list:
    model.eval()
    results = []
    print("\n" + "=" * 100, flush=True)
    print("VAULT v2 STRICT O(L) LONG-CONTEXT PREFILL SCALING AUDIT (BF16)", flush=True)
    print("=" * 100, flush=True)

    for seq_len in [2048, 4096, 8192, 16384, 32768]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, model.vocab_size, (1, seq_len), device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model._encode_hidden(x, step_ratio=1.0)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model._encode_hidden(x, step_ratio=1.0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3.0
        tok_s = seq_len / dt
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  L = {seq_len:6,d} | {tok_s:9,.1f} tok/s ({dt * 1000:6.1f} ms) | Peak VRAM: {vram_gb:5.2f} GB", flush=True)
        results.append({"seq_len": seq_len, "tok_s": round(tok_s, 1), "ms": round(dt * 1000, 2), "peak_vram_gb": round(vram_gb, 3)})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/vault_v2_sft_best.pt")
    parser.add_argument("--out", type=str, default="benchmarks/eval_vault_v2_audit.json")
    parser.add_argument("--bench_scaling", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    model = VAULTLMv2(vocab_size=len(tok)).to(device)

    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = ck.get("model_state", ck.get("model", ck))
        model.load_state_dict(state)
        print(f"[+] Loaded VAULTLMv2 checkpoint: {args.ckpt}", flush=True)
    else:
        print(f"[!] Checkpoint {args.ckpt} not found yet — running with random init.", flush=True)

    summary = run_8domain_eval(model, tok, os.path.basename(args.ckpt), device, args.out)
    if args.bench_scaling:
        summary["scaling_audit"] = bench_long_context_scaling(model, device)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
