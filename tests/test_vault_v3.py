# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v3 numerical consistency tests.
#   1. Parallel (training/prefill) path == recurrent (streaming) path, tiny random model, fp32 CPU.
#   2. Parallel prefill + streaming continuation == full parallel forward.
#   3. (optional, --qwen) VAULT v3 backbone == HF Qwen2ForCausalLM logits.

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.vault_v3_model import VAULTv3Config, VAULTv3ForCausalLM, load_qwen2_into_vault  # noqa: E402


def tiny_model(seed: int = 0) -> VAULTv3ForCausalLM:
    torch.manual_seed(seed)
    cfg = VAULTv3Config(
        vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=96,
        window=8, vault_primes=(3, 5, 7), rope_theta=10000.0,
    )
    m = VAULTv3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "layernorm" in n or n == "norm.weight":
                continue
            if ".vault.a_" in n:
                p.add_(0.2 * torch.randn_like(p))
            elif ".vault.alpha" in n:
                p.copy_(torch.randn_like(p))
            else:
                p.normal_(0.0, 0.2)
    return m


def test_parallel_vs_recurrent():
    m = tiny_model()
    L = 53
    ids = torch.randint(0, m.cfg.vocab_size, (2, L))
    with torch.no_grad():
        ref = m(ids, mode="block", vault_on=True)                     # (B, L, V)
        st = m.new_state(2)
        outs = [m.step(ids[:, t], st) for t in range(L)]
    rec = torch.stack(outs, dim=1)
    err = (ref - rec).abs().max().item()
    mem_effect = (ref - m(ids, mode="block", vault_on=False)).abs().max().item()
    print(f"[1] parallel vs recurrent max|diff| = {err:.3e} (vault effect on logits = {mem_effect:.3e})")
    assert mem_effect > 1e-3, "vault had no effect; test would be vacuous"
    assert err < 1e-4, err


def test_prefill_then_stream():
    m = tiny_model(1)
    ids = torch.randint(0, m.cfg.vocab_size, (1, 61))
    with torch.no_grad():
        ref = m(ids, mode="block", vault_on=True)
        for split in (5, 8, 16, 29, 40):
            last, st = m.prefill(ids[:, :split])
            errs = [(last - ref[:, split - 1]).abs().max().item()]
            for t in range(split, ids.shape[1]):
                lg = m.step(ids[:, t], st)
                errs.append((lg - ref[:, t]).abs().max().item())
            print(f"[2] prefill({split}) + stream: max|diff| = {max(errs):.3e}")
            assert max(errs) < 1e-4


def test_against_hf_qwen(qwen_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import json
    with open(os.path.join(qwen_dir, "config.json")) as f:
        qc = json.load(f)
    cfg = VAULTv3Config.from_qwen_config(qc)
    m = VAULTv3ForCausalLM(cfg)
    load_qwen2_into_vault(m, qwen_dir)
    m = m.to(dev, torch.float32).eval()   # fp32: compare math, not bf16 rounding
    hf = AutoModelForCausalLM.from_pretrained(qwen_dir, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(qwen_dir)
    text = ("La photosynthèse convertit la lumière du soleil en énergie chimique. "
            "Photosynthesis converts sunlight into chemical energy stored in glucose. ") * 4
    ids = torch.tensor([tok.encode(text)], device=dev)
    with torch.no_grad():
        a = m(ids, mode="full", vault_on=False).float()
        b = hf(ids).logits.float()
        c = m(ids, mode="block", vault_on=True).float()   # L <= 2W -> identical path
    d_ab = (a - b).abs().max().item()
    top_agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    d_ac = (a - c).abs().max().item()
    print(f"[3] VAULT v3 vs HF Qwen2 (fp32, L={ids.shape[1]}): max|dlogit| = {d_ab:.2e}, top-1 agreement = {top_agree*100:.2f}%"
          f"  |  block==full at L<=2W: {d_ac:.2e}")
    assert d_ab < 5e-3 and top_agree == 1.0 and d_ac == 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", type=str, default="")
    args = ap.parse_args()
    test_parallel_vs_recurrent()
    test_prefill_then_stream()
    if args.qwen:
        test_against_hf_qwen(args.qwen)
    print("ALL VAULT v3 TESTS PASSED")
