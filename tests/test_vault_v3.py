# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v3 numerical consistency & archived Kenyon Cell module tests.
#   1. Parallel (prefill) path == recurrent (streaming) path across chunk boundaries (both Segment Memory + SWA/sinks).
#   2. Parallel prefill + streaming continuation == full parallel forward.
#   3. Archived Co-Prime Kenyon Cell module (src/kenyon_cells.py): routing, gradients, and homogeneous rescaling.
#   4. (optional, --qwen) VAULT v3 backbone == HF Qwen2ForCausalLM logits.

import argparse
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.kenyon_cells import (  # noqa: E402
    KenyonVault,
    kenyon_read,
    kenyon_route_cells,
    kenyon_write,
    make_kenyon_route,
    save_kenyon_params,
)
from src.vault_v3_model import VAULTv3Config, VAULTv3ForCausalLM, load_qwen2_into_vault  # noqa: E402


def tiny_model(seed: int = 0, seg_budget: int = 12) -> VAULTv3ForCausalLM:
    torch.manual_seed(seed)
    cfg = VAULTv3Config(
        vocab_size=256, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=96,
        window=8, n_sink=2, max_sink=8, rope_theta=10000.0,
        im_start_id=252, im_end_id=251, system_tok_id=250,
        seg_budget=seg_budget, seg_hidden=32, seg_min=2, seg_max=5,
        seg_layers=(0, 1, 2) if seg_budget > 0 else (),
    )
    m = VAULTv3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "layernorm" in n or n in ("norm.weight", "seg_scorer.ln.weight"):
                p.uniform_(0.85, 1.15)
            elif n == "seg_scorer.ln.bias":
                p.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.2)
    return m


def test_parallel_vs_recurrent():
    m = tiny_model(0, seg_budget=12)
    L = 53
    ids1 = torch.randint(0, 240, (1, L))
    ids2 = torch.randint(0, 240, (2, L))
    with torch.no_grad():
        ref = m(ids1, mode="block")                                 # (1, L, V) with Segment Memory
        ref_noseg = m(ids1, mode="block", seg_budget=0)             # (1, L, V) SWA + sinks only
        st = m.new_state(1)
        outs = [m.step(ids1[:, t], st) for t in range(L)]
        ref_b2 = m(ids2, mode="block", seg_budget=0)                # (2, L, V) batched SWA + sinks
        st0 = m.new_state(2, seg_budget=0)
        outs0 = [m.step(ids2[:, t], st0) for t in range(L)]
    rec = torch.stack(outs, dim=1)
    rec0 = torch.stack(outs0, dim=1)
    err = (ref - rec).abs().max().item()
    err0 = (ref_b2 - rec0).abs().max().item()
    mem_effect = (ref - ref_noseg).abs().max().item()
    print(
        f"[1] parallel vs recurrent max|diff| = {err:.3e} (B=2 SWA max|diff| = {err0:.3e}, "
        f"segment-memory effect on logits = {mem_effect:.3e})"
    )
    assert mem_effect > 1e-3, "segment memory had no effect; test would be vacuous"
    assert err < 1e-4 and err0 < 1e-4, (err, err0)


def test_prefill_then_stream():
    m = tiny_model(1, seg_budget=12)
    ids = torch.randint(0, 240, (1, 61))
    with torch.no_grad():
        ref = m(ids, mode="block")
        for split in (5, 8, 16, 29, 40):
            last, st = m.prefill(ids[:, :split])
            errs = [(last - ref[:, split - 1]).abs().max().item()]
            for t in range(split, ids.shape[1]):
                lg = m.step(ids[:, t], st)
                errs.append((lg - ref[:, t]).abs().max().item())
            print(f"[2] prefill({split}) + stream: max|diff| = {max(errs):.3e}")
            assert max(errs) < 1e-4


def test_archived_kenyon_cells():
    torch.manual_seed(7)
    primes = (3, 5, 7)
    C = sum(primes)
    B, G, H, T, D, d_model = 2, 2, 4, 8, 16, 64
    route = make_kenyon_route(D, primes)
    assert route.shape == (D, C) and set(route.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    kv = KenyonVault(d_model, H, G, D)
    with torch.no_grad():
        kv.alpha.uniform_(-0.5, 0.5)
        kv.beta.weight.normal_(0.0, 0.2)

    h = torch.randn(B, T, d_model, requires_grad=True)
    k = torch.randn(B, G, T, D, requires_grad=True)
    v = torch.randn(B, G, T, D, requires_grad=True)
    q = torch.randn(B, H, T, D, requires_grad=True)

    M0 = torch.zeros(B, G, C, D, D)
    z0 = torch.zeros(B, G, C, D)
    kt = kv.keys(k)
    beta = kv.write_strength(h)
    cells = kenyon_route_cells(kt.detach(), route, primes)
    assert cells.shape == (B, G, T, len(primes))

    M1, z1 = kenyon_write(M0, z0, kt, v, beta, route, primes)
    qt = kv.queries(q, H // G)
    mem = kenyon_read(M1, z1, qt, route, primes).reshape(B, H, T, D)
    gated = mem * kv.gate().view(1, H, 1, 1)
    loss = gated.pow(2).sum()
    loss.backward()
    assert kv.a_k.grad is not None and kv.a_k.grad.abs().max().item() > 0
    assert kv.a_q.grad is not None and kv.a_q.grad.abs().max().item() > 0
    assert kv.beta.weight.grad is not None and kv.beta.weight.grad.abs().max().item() > 0
    assert kv.alpha.grad is not None and kv.alpha.grad.abs().max().item() > 0

    # Verify homogeneous cell rescaling preserves exact readout when z_max > 65536
    with torch.no_grad():
        M_big = M1.detach() * 100000.0
        z_big = z1.detach() * 100000.0
        assert z_big.max().item() > 65536.0
        out_before = kenyon_read(M_big, z_big, qt.detach(), route, primes)
        M_res, z_res = kenyon_write(M_big, z_big, torch.zeros_like(kt), torch.zeros_like(v), torch.zeros_like(beta), route, primes)
        assert z_res.max().item() < z_big.max().item()
        out_after = kenyon_read(M_res, z_res, qt.detach(), route, primes)
        rescale_err = (out_before - out_after).abs().max().item()
        assert rescale_err < 1e-5, rescale_err

        class _DummyHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("vault_route", route)
                self.vault = kv

        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, "kc.pt")
            save_kenyon_params(_DummyHolder(), ck, extra={"primes": primes})
            loaded = torch.load(ck, map_location="cpu", weights_only=False)
            assert loaded["format"] == "vault_kenyon_archive" and "vault_route" in loaded["vault_state"]

    print(f"[3] archived Kenyon cells (src/kenyon_cells.py): grad OK, homogeneous rescale max|diff| = {rescale_err:.3e}")


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
        a = m(ids, mode="full").float()
        b = hf(ids).logits.float()
        c = m(ids, mode="block").float()   # L <= 2W -> identical path
    d_ab = (a - b).abs().max().item()
    top_agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    d_ac = (a - c).abs().max().item()
    print(f"[4] VAULT v3 vs HF Qwen2 (fp32, L={ids.shape[1]}): max|dlogit| = {d_ab:.2e}, top-1 agreement = {top_agree*100:.2f}%"
          f"  |  block==full at L<=2W: {d_ac:.2e}")
    assert d_ab < 5e-3 and top_agree == 1.0 and d_ac == 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", type=str, default="")
    args = ap.parse_args()
    test_parallel_vs_recurrent()
    test_prefill_then_stream()
    test_archived_kenyon_cells()
    if args.qwen:
        test_against_hf_qwen(args.qwen)
    print("ALL VAULT v3 TESTS PASSED")
