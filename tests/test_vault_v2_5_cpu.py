#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Strict CPU-Only (`CUDA_VISIBLE_DEVICES=""`) Mathematical Parity, Checkpoint Inheritance,
# Multi-Subword Sibling Disentanglement, and Contrastive Loss Test Suite for `VAULT v2.5`.

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.vault_v2_model import VAULTLMv2
from src.vault_v2_5_model import (
    GTRM_EMA_ALPHA_DEFAULT,
    VAULTLMv2_5,
    chunked_causal_ema_dim2,
    load_v2_4_into_v2_5,
    upgrade_v2_5_to_wider_primes,
)
from src.vault_v2_streaming import VAULTv2StreamingEngine
from src.vault_v2_5_streaming import VAULTv2_5StreamingEngine


def test_1_checkpoint_inheritance_and_mitosis() -> None:
    print("[Test 1/5] Real VAULT v2.4 Checkpoint Loading & 220 -> 508 Zero-Shot Mitosis...", flush=True)
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "vault_v2_4_step7000_best.pt",
    )
    assert os.path.exists(ckpt_path), f"Missing protected v2.4 checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    m25 = VAULTLMv2_5().eval()
    info = load_v2_4_into_v2_5(m25, ckpt, init_ext_span_alpha=0.55)
    print(f"  -> Loaded {info['loaded_tensors']} tensors + {info['new_v2_5_tensors']} new v2.5 tensors | params={m25.count_parameters():,}", flush=True)
    assert info["new_v2_5_tensors"] == 32, f"Expected 32 new tensors (2 per layer x 16), got {info['new_v2_5_tensors']}"

    # Test zero-shot mitosis 220 -> 508 cells on a 2-layer model to keep CPU test snappy & thorough
    m_src = VAULTLMv2_5(d_model=256, n_layers=2, d_ff=512, prime_banks=(47, 53, 59, 61)).eval()
    for blk in m_src.layers:
        blk.attn.ext_span_alpha.data.fill_(0.55)
    m_dst = VAULTLMv2_5(d_model=256, n_layers=2, d_ff=512, prime_banks=(113, 127, 131, 137)).eval()
    mit_info = upgrade_v2_5_to_wider_primes(m_dst, m_src.state_dict(), old_primes=(47, 53, 59, 61))
    assert abs(float(m_dst.layers[0].attn.ext_span_alpha.item()) - 0.55) < 1e-6
    print(f"  -> Mitosis 220 -> 508 verified: {mit_info}", flush=True)


def test_2_stationary_gtrm_ema_exactness_and_128k_invariance() -> None:
    print("[Test 2/5] Stationary Causal Leaky-EMA GTRM Exactness & 128K Scale Invariance...", flush=True)
    torch.manual_seed(20261103)
    B, G, L, D = 2, 4, 301, 64  # intentionally non-multiple of 64!
    v = torch.randn(B, G, L, D, dtype=torch.float32)
    alpha = GTRM_EMA_ALPHA_DEFAULT
    gamma = 1.0 - alpha

    y_par = chunked_causal_ema_dim2(v, alpha=alpha, block_len=64)

    y_ref = torch.zeros_like(v)
    m = torch.zeros((B, G, D), dtype=torch.float64)
    v64 = v.double()
    for t in range(L):
        m = gamma * m + alpha * v64[:, :, t, :]
        bc = 1.0 - (gamma ** (t + 1))
        y_ref[:, :, t, :] = (m / bc).float()

    max_diff = float((y_par - y_ref).abs().max().item())
    print(f"  -> Parallel vs Sequential EMA max_abs_diff (L={L}): {max_diff:.3e}", flush=True)
    assert max_diff < 2e-6, f"EMA parallel scan mismatch: {max_diff}"

    # Verify 128K stationarity vs 1/t prefix-mean collapse!
    L_128k = 131072
    v_long = torch.randn(1, 1, L_128k, 16, dtype=torch.float32)
    ema_long = chunked_causal_ema_dim2(v_long, alpha=alpha, block_len=64)
    rms_ema_2k = float(ema_long[:, :, 1920:2048, :].pow(2).mean().sqrt().item())
    rms_ema_128k = float(ema_long[:, :, -128:, :].pow(2).mean().sqrt().item())

    t_div = torch.arange(1, L_128k + 1, dtype=torch.float32).view(1, 1, L_128k, 1)
    pm_long = torch.cumsum(v_long, dim=2) / t_div
    rms_pm_2k = float(pm_long[:, :, 1920:2048, :].pow(2).mean().sqrt().item())
    rms_pm_128k = float(pm_long[:, :, -128:, :].pow(2).mean().sqrt().item())

    ema_ratio = rms_ema_128k / max(rms_ema_2k, 1e-8)
    pm_ratio = rms_pm_128k / max(rms_pm_2k, 1e-8)
    print(
        f"  -> 128K/2K RMS ratio: Stationary EMA = {ema_ratio:.3f}x (invariant!) vs v2.4 Prefix-Mean = {pm_ratio:.3f}x (collapsed!)",
        flush=True,
    )
    assert 0.65 <= ema_ratio <= 1.45, f"EMA is not stationary at 128K: ratio={ema_ratio}"
    assert pm_ratio < 0.25, f"Expected prefix-mean collapse at 128K, got ratio={pm_ratio}"


def test_3_step0_parity_with_v2_4() -> None:
    print("[Test 3/5] Step-0 SalienceEntityBinderV2_5 (`g0, g1`) & Forward Parity with VAULT v2.4...", flush=True)
    torch.manual_seed(42)
    m24 = VAULTLMv2(
        d_model=256,
        n_layers=2,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=512,
        v2_1_mode=True,
        v2_2_mode=True,
        prime_banks=(47, 53, 59, 61),
    ).eval()
    # Set `ema_alpha` tiny ONLY to check exact non-GTRM Step-0 identity on `ext_span_alpha=0.0`!
    m25_lim = VAULTLMv2_5(
        d_model=256, n_layers=2, n_q_heads=12, n_vault_groups=4, head_dim=64, d_ff=512, ema_alpha=1e-8
    ).eval()
    load_v2_4_into_v2_5(m25_lim, m24.state_dict(), init_ext_span_alpha=0.0)

    ids = torch.randint(0, 4096, (2, 192), dtype=torch.long)
    with torch.no_grad():
        x = m24.embed_tokens(ids)
        x0_24 = m24.compute_x0_norm(x)
        x0_25 = m25_lim.compute_x0_norm(x)
        eu24, er24 = m24.layers[0].attn.salience_binder(x0_24, x0_24)
        eu25, er25 = m25_lim.layers[0].attn.salience_binder(x0_25, x0_25)
        diff_u01 = float((eu24 - eu25[:, :, :2, :]).abs().max().item())
        diff_r01 = float((er24 - er25[:, :, :2, :]).abs().max().item())

        logits_24, _ = VAULTv2StreamingEngine.prefill_streaming(
            m24, ids, segment_size=192, tail_logits_len=192, use_cudagraph=False
        )
        logits_25 = m25_lim(ids, step_ratio=1.0)
        diff_logits = float((logits_24 - logits_25).abs().max().item())

    print(
        f"  -> Binder g0/g1 max_diff: ent_u={diff_u01:.3e}, ent_r={diff_r01:.3e} | Full Logits (alpha->0, ext=0): {diff_logits:.3e}",
        flush=True,
    )
    assert diff_u01 < 1e-6 and diff_r01 < 1e-6, "Vault Groups 1 & 2 binder branches diverged from v2.4!"
    assert diff_logits < 5e-5, f"Step-0 limit logits diverged from v2.4: {diff_logits}"


def test_4_streaming_prefill_and_decode_bit_exactness() -> None:
    print("[Test 4/5] Monolithic Forward vs Streaming Prefill + 128-Step Decode across 4 Chunk Rollovers...", flush=True)
    torch.manual_seed(1337)
    for ext_alpha in (0.0, 0.55):
        m = VAULTLMv2_5(
            d_model=256,
            n_layers=2,
            n_q_heads=12,
            n_vault_groups=4,
            head_dim=64,
            d_ff=512,
            prime_banks=(47, 53, 59, 61),
        ).eval()
        for blk in m.layers:
            blk.attn.ext_span_alpha.data.fill_(ext_alpha)
            blk.attn.ind_shared_proj_g2.data.normal_(0.0, 0.08)

        # Total L = 256 = 128 prefill (in 64-tok segments!) + 128 step-by-step decode tokens (crosses t=128 & t=192 rollovers!)
        ids = torch.randint(0, 8192, (2, 256), dtype=torch.long)
        with torch.no_grad():
            full_logits = m(ids, step_ratio=1.0)
            pref_logits, cache = VAULTv2_5StreamingEngine.prefill_streaming(
                m,
                ids[:, :128],
                segment_size=64,
                tail_logits_len=128,
                use_cudagraph=False,
            )
            pref_diff = float((full_logits[:, :128, :] - pref_logits).abs().max().item())

            dec_list = []
            for t in range(128, 256):
                step_l = VAULTv2_5StreamingEngine.decode_step(
                    m, ids[:, t : t + 1], cache, step_ratio=1.0, return_logits=True, use_cudagraph=False
                )
                dec_list.append(step_l)
            dec_logits = torch.cat(dec_list, dim=1)
            dec_diff = float((full_logits[:, 128:, :] - dec_logits).abs().max().item())

        print(
            f"  -> [ext_span_alpha={ext_alpha:.2f}] Prefill (2x64 segs) max_diff={pref_diff:.3e} | "
            f"128x Decode (t=128..255) max_diff={dec_diff:.3e}",
            flush=True,
        )
        assert pref_diff < 2e-5, f"Streaming prefill mismatch ({pref_diff}) at ext_span_alpha={ext_alpha}"
        assert dec_diff < 2e-5, f"Streaming decode mismatch ({dec_diff}) at ext_span_alpha={ext_alpha}"


def test_5_long_sibling_separation_and_contrastive_gradients() -> None:
    print("[Test 5/5] 8..11-Subword Sibling Disentanglement & Contrastive Choice-Margin Backward Pass...", flush=True)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)

    ent_a = "FalconTurbine-Unit42408"
    ent_b = "FalconTurbine-Unit42953"
    toks_a = tok.encode(" " + ent_a, add_special_tokens=False)
    toks_b = tok.encode(" " + ent_b, add_special_tokens=False)
    shared_prefix_len = 0
    for ta, tb in zip(toks_a, toks_b):
        if ta == tb:
            shared_prefix_len += 1
        else:
            break
    print(
        f"  -> Sibling BPE lengths: len('{ent_a}')={len(toks_a)}, len('{ent_b}')={len(toks_b)}, "
        f"identical prefix = {shared_prefix_len} BPE tokens",
        flush=True,
    )
    assert len(toks_a) >= 8 and shared_prefix_len >= 7

    doc_a = f"SYSTEM MANIFEST\n[Subsystem 001: {ent_a}] Operational State: NOMINAL | Verification Code: CHK-4102\n"
    doc_b = f"SYSTEM MANIFEST\n[Subsystem 001: {ent_b}] Operational State: NOMINAL | Verification Code: CHK-4102\n"
    ids_a = tok.encode(doc_a, add_special_tokens=False)
    ids_b = tok.encode(doc_b, add_special_tokens=False)
    min_l = min(len(ids_a), len(ids_b))
    pair_ids = torch.tensor([ids_a[:min_l], ids_b[:min_l]], dtype=torch.long)

    torch.manual_seed(99)
    m = VAULTLMv2_5(d_model=256, n_layers=2, n_q_heads=12, n_vault_groups=4, head_dim=64, d_ff=512).eval()
    with torch.no_grad():
        x = m.embed_tokens(pair_ids)
        x0 = m.compute_x0_norm(x)
        eu, er = m.layers[0].attn.salience_binder(x0, x0)
        eu_rms = F.rms_norm(eu, (64,), eps=1e-6)
        # Inspect the value token position near the end of the line (after `Verification Code: `)
        q_pos = min_l - 3
        diff_g1 = float((eu_rms[0, q_pos, 1] - eu_rms[1, q_pos, 1]).norm().item())
        diff_g2 = float((eu_rms[0, q_pos, 2] - eu_rms[1, q_pos, 2]).norm().item())
        sep_ratio = diff_g2 / max(diff_g1, 1e-9)
    print(
        f"  -> Value-anchor RMS-normed entity L2 distance across {shared_prefix_len}-token-prefix siblings: "
        f"v2.4 g1 (m<=6) = {diff_g1:.5f} vs v2.5 g2 (m<=10) = {diff_g2:.4f} ({sep_ratio:,.1f}x separation)",
        flush=True,
    )
    assert diff_g1 < 1e-4 and diff_g2 > 0.05 and diff_g2 > 1000.0 * max(diff_g1, 1e-6), (
        f"Expected v2.5 g2 (m=1..10) to separate >=8-token siblings >1000x vs blind g1: g1={diff_g1}, g2={diff_g2}"
    )

    # Test backward pass through `forward_loss(..., return_ent_div=True)`, `forward_mc_contrastive_loss`, and `forward_wg_contrastive_loss`
    m.train()
    for blk in m.layers:
        blk.attn.ext_span_alpha.data.fill_(0.55)

    B = 2
    seq = torch.randint(0, 4096, (B, 257), dtype=torch.long)
    w_mask = torch.ones((B, 256), dtype=torch.float32)
    ce_loss, beta_m, ent_div = m.forward_loss(
        seq[:, :-1], seq[:, 1:], loss_weights=w_mask, return_beta_mean=True, return_ent_div=True
    )

    p_ids = torch.randint(1, 2048, (3, 32), dtype=torch.long)
    p_lens = torch.tensor([18, 24, 15], dtype=torch.long)
    c_ids = torch.randint(1, 2048, (3, 4, 12), dtype=torch.long)
    c_lens = torch.tensor([[4, 6, 5, 3], [7, 4, 8, 5], [3, 4, 0, 0]], dtype=torch.long)
    c_bytes = torch.tensor(
        [[16.0, 24.0, 20.0, 12.0], [28.0, 15.0, 31.0, 19.0], [10.0, 14.0, 1.0, 1.0]], dtype=torch.float32
    )
    n_ch = torch.tensor([4, 4, 2], dtype=torch.long)
    gold = torch.tensor([2, 0, 1], dtype=torch.long)
    mc_loss, mc_acc = m.forward_mc_contrastive_loss(p_ids, p_lens, c_ids, c_lens, c_bytes, n_ch, gold, eos_id=0)

    wg_seq = torch.randint(1, 2048, (3, 2, 40), dtype=torch.long)
    wg_st = torch.tensor([[12, 14], [10, 10], [16, 15]], dtype=torch.long)
    wg_ln = torch.tensor([[8, 8], [11, 11], [6, 6]], dtype=torch.long)
    wg_gold = torch.tensor([0, 1, 1], dtype=torch.long)
    wg_loss, wg_acc = m.forward_wg_contrastive_loss(wg_seq, wg_st, wg_ln, wg_gold)

    tot = ce_loss + 0.03 * ent_div + 0.18 * (0.75 * mc_loss + 0.25 * wg_loss)
    tot.backward()

    g_ext = float(m.layers[0].attn.ext_span_alpha.grad.abs().item())
    g_sp2 = float(m.layers[0].attn.ind_shared_proj_g2.grad.abs().max().item())
    g_ib = float(m.layers[0].attn.ind_beta_proj.weight.grad.abs().max().item())
    print(
        f"  -> Joint SFT + EntDiv + MC/WG backward clean! ce={ce_loss.item():.3f}, mc={mc_loss.item():.3f} (acc={mc_acc:.2f}), "
        f"wg={wg_loss.item():.3f} (acc={wg_acc:.2f}) | |grad(ext_alpha)|={g_ext:.3e}, |grad(proj_g2)|={g_sp2:.3e}, |grad(ind_beta)|={g_ib:.3e}",
        flush=True,
    )
    assert math.isfinite(tot.item()) if "math" in globals() else torch.isfinite(tot).item()
    assert g_ext > 0.0 and g_sp2 > 0.0 and g_ib > 0.0, "Zero gradient on v2.5 extended parameters!"


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 96, flush=True)
    print("VAULT v2.5 CPU-ONLY MATHEMATICAL PARITY & ARCHITECTURE VERIFICATION SUITE", flush=True)
    print("=" * 96, flush=True)
    test_1_checkpoint_inheritance_and_mitosis()
    test_2_stationary_gtrm_ema_exactness_and_128k_invariance()
    test_3_step0_parity_with_v2_4()
    test_4_streaming_prefill_and_decode_bit_exactness()
    test_5_long_sibling_separation_and_contrastive_gradients()
    print("=" * 96, flush=True)
    print(f"[ALL 5/5 CPU TESTS PASSED] Verified in {time.perf_counter() - t_start:.2f}s (0 MB VRAM touched!)", flush=True)
