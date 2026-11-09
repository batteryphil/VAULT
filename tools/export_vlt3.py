#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Export VAULT v3 (Qwen2.5 backbone + trained SegmentScorer v2) to VLT3 files.
#
#   python tools/export_vlt3.py --seg_ckpt dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt --format q8_0 --out dist/vault_v3_1.5b_q8_0.vlt3
#
# Formats:
#   bf16 : matmul weights + embeddings BF16 (master release)
#   q8_0 : matmul weights + embeddings Q8_0 (GGML-compatible 32-weight blocks, fp16 scale)
#   q4_0 : matmul weights Q4_0 (with Q8_0 GQA KV & boundary Q/O projections), embeddings/tied output head Q8_0
#   Norms and biases are always F32 (small).

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from src.vault_v3_model import VAULTv3Config, VAULTv3ForCausalLM, load_qwen2_into_vault  # noqa: E402
from vlt3 import read_vlt3, write_vlt3  # noqa: E402

DEFAULT_QWEN = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def build_state(qwen_dir: str, vault_ckpt: str, window: int, seg_ckpt: str = "", seg_budget: int = 0):
    _ = vault_ckpt  # legacy Kenyon flag accepted as no-op for CLI compatibility
    with open(os.path.join(qwen_dir, "config.json")) as f:
        qc = json.load(f)
    cfg = VAULTv3Config.from_qwen_config(qc, window=window)
    sck = None
    if seg_ckpt:
        sck = torch.load(seg_ckpt, map_location="cpu", weights_only=False)
        layers = tuple(int(x) for x in sck.get("layers", (8, 16, 24)))
        cfg.seg_layers = layers
        cfg.seg_budget = int(seg_budget if seg_budget > 0 else 1024)
        cfg.seg_hidden = int(sck["state"]["net.1.weight"].shape[0] if "net.1.weight" in sck["state"] else sck["state"]["fc1.weight"].shape[0])
        cfg.seg_min = int(sck.get("min_seg", 16))
        cfg.seg_max = int(sck.get("max_seg", 48))
    m = VAULTv3ForCausalLM(cfg)
    load_qwen2_into_vault(m, qwen_dir)
    if sck is not None:
        raw_st = sck["state"]
        remap = {
            "net.0.weight": "ln.weight", "net.0.bias": "ln.bias",
            "net.1.weight": "fc1.weight", "net.1.bias": "fc1.bias",
            "net.3.weight": "fc2.weight", "net.3.bias": "fc2.bias",
        }
        mapped = {remap.get(k, k): v for k, v in raw_st.items()}
        m.seg_scorer.load_state_dict(mapped, strict=True)
    return m.state_dict(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_dir", default=DEFAULT_QWEN)
    ap.add_argument("--vault_ckpt", default="", help="Deprecated no-op (Kenyon modules archived in src/kenyon_cells.py)")
    ap.add_argument("--seg_ckpt", default="", help="Segment Scorer checkpoint (e.g. dist/hf_vault_v3/vault_params/vault_v3_seg_scorer.pt)")
    ap.add_argument("--seg_budget", type=int, default=1024, help="Default Segment Memory KV budget when --seg_ckpt is given")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--format", choices=["f32", "bf16", "q8_0", "q4_0"], default="q8_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="VAULT-v3-1.5B")
    args = ap.parse_args()

    t0 = time.time()
    sd, cfg = build_state(args.qwen_dir, args.vault_ckpt, args.window, args.seg_ckpt, args.seg_budget)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    info = write_vlt3(args.out, sd, cfg, args.qwen_dir, args.format, model_name=f"{args.name}-{args.format}")
    vf = read_vlt3(args.out)  # full validation incl. CRCs & NaN/Inf
    print(f"wrote {args.out}: {info['file_size']/1e6:.1f} MB, {info['n_tensors']} tensors, "
          f"seg_budget={cfg.seg_budget}, tokenizer {info['tokenizer_bytes']/1e6:.2f} MB, validated OK in {time.time()-t0:.1f}s")
    by_dt = {}
    for e in vf.entries:
        by_dt[e["dtype"]] = by_dt.get(e["dtype"], 0) + e["nbytes"]
    from vlt3 import DTYPE_NAMES
    print("  bytes by dtype: " + ", ".join(f"{DTYPE_NAMES[k]}={v/1e6:.1f}MB" for k, v in sorted(by_dt.items())))


if __name__ == "__main__":
    main()
