#!/usr/bin/env python3
"""
Phase B Live Checkpoint Diagnostic Suite for VAULTLMv2 (CPU-Safe & Read-Only).

Answers Reviewer Sections 2, 3, 4.2, 5.2, 8 (Risk 6), and Section 12 (Questions 1-3):
  1. Full 88-cell occupancy histograms per bank (17, 19, 23, 29), separately for Keys and Queries:
     - Normalized entropy H_r^norm (Key vs Query), Gini coefficient, dead cells, max/min occupancy,
       effective cell count exp(H_r), pairwise bank MI I(C_r; C_s), empirical 4-tuple collision rate,
       and Query/Write route agreement P(c_q(x) == c_k(x)).
  2. Router Migration & Flip Rate across WSD Decay (`Step 5,400` vault_v2_wsd_stable.pt vs
     latest `vault_v2_pretrain_best.pt`), including ternary weight flip rate in `kenyon_proj`.
  3. 3:1 GVA Intra-Group Head Overlap: P(c_{q,h1} == c_{q,h2}) and effective distinct cells
     simultaneously probed by the 3 query heads sharing each vault group.
  4. Per-Layer Router STE Gradient Ratio ||grad W_router||_2 / ||grad W_layer||_2 + beta_t statistics.
  5. Controlled 3-Way Open-Vocabulary NIAH Diagnostic (Tests A / B / C across delays d in {2, 8, 15, 35}
     and 6 completely unseen entity stems, plus Current vs Empty-Masked vs Equal-Weight consensus).
"""

import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src.vault_v2_model as v2m


# ==============================================================================
# 1. EXACT VECTORIZED COMPACT WY CPU FORWARD/BACKWARD KERNEL SUBSTITUTE
# ==============================================================================

CONSENSUS_MODE = "current"  # "current" | "option_a" | "equal" | "log_occ"


def flash_vault_v2_cpu_differentiable(
    phi_q: torch.Tensor,   # (B, H, L, D)
    phi_k: torch.Tensor,   # (B, G, L, D)
    v: torch.Tensor,       # (B, G, L, D)
    beta: torch.Tensor,    # (B, G, L)
    cell_q: torch.Tensor,  # (B, H, R, L)
    cell_k: torch.Tensor,  # (B, G, R, L)
    num_cells: int = 88,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, H, L, D = phi_q.shape
    _, G, _, _ = phi_k.shape
    R = cell_k.size(2)
    hpg = H // G
    orig_dtype = v.dtype

    pad = (chunk_size - (L % chunk_size)) % chunk_size
    if pad > 0:
        phi_q = F.pad(phi_q, (0, 0, 0, pad))
        phi_k = F.pad(phi_k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        cell_q = F.pad(cell_q, (0, pad), value=0)
        cell_k = F.pad(cell_k, (0, pad), value=0)

    Lp = phi_q.size(2)
    NC = Lp // chunk_size
    C = chunk_size

    pq = phi_q.float().view(B, G, hpg, NC, C, D)
    pk = phi_k.float().view(B, G, NC, C, D)
    vf = v.float().view(B, G, NC, C, D)
    bf = beta.float().view(B, G, NC, C)
    cq = cell_q.view(B, G, hpg, R, NC, C)
    ck = cell_k.view(B, G, R, NC, C)

    S = torch.zeros(B, G, num_cells, D, D, dtype=torch.float32)
    occ = torch.zeros(B, G, num_cells, dtype=torch.float32)
    out_chunks = []

    idx = torch.arange(C)
    strict_lower = idx[:, None] > idx[None, :]
    causal_inc = idx[:, None] >= idx[None, :]
    eye_C = torch.eye(C, dtype=torch.float32)

    b_g = torch.arange(B)[:, None, None, None]
    g_g = torch.arange(G)[None, :, None, None]
    b_q = torch.arange(B)[:, None, None, None, None]
    g_q = torch.arange(G)[None, :, None, None, None]

    for nc in range(NC):
        K_c = pk[:, :, nc]
        V_c = vf[:, :, nc]
        B_c = bf[:, :, nc]
        ck_c = ck[:, :, :, nc]
        cq_c = cq[:, :, :, :, nc]
        Q_c = pq[:, :, :, nc]

        KKT = torch.matmul(K_c, K_c.transpose(-1, -2))
        same_k = (ck_c.unsqueeze(-1) == ck_c.unsqueeze(-2)) & strict_lower
        L_mat = torch.where(same_k, B_c.unsqueeze(2).unsqueeze(-1) * KKT.unsqueeze(2), torch.zeros(1))

        BV = (B_c.unsqueeze(-1) * V_c).unsqueeze(2).expand(B, G, R, C, D)
        BK = (B_c.unsqueeze(-1) * K_c).unsqueeze(2).expand(B, G, R, C, D)
        U_c = torch.linalg.solve_triangular(eye_C + L_mat, BV, upper=False, unitriangular=True)
        W_c = torch.linalg.solve_triangular(eye_C + L_mat, BK, upper=False, unitriangular=True)

        S_k0 = S[b_g, g_g, ck_c]
        Delta = U_c - torch.einsum("bgrcvd,bgrcd->bgrcv", S_k0, W_c)

        S_q0 = S[b_q, g_q, cq_c]
        y_inter = torch.einsum("bghrcvd,bghcd->bghrcv", S_q0, Q_c)

        same_qk = (cq_c.unsqueeze(-1) == ck_c.unsqueeze(2).unsqueeze(-2)) & causal_inc
        QK_scores = torch.matmul(Q_c, K_c.unsqueeze(2).transpose(-1, -2)).unsqueeze(3)
        QK_masked = torch.where(same_qk, QK_scores, torch.zeros(1))
        y_intra = torch.matmul(QK_masked, Delta.unsqueeze(2))
        y_r = y_inter + y_intra

        occ_hist = occ[b_q, g_q, cq_c]
        occ_intra = same_qk.float().sum(dim=-1)
        n_q = occ_hist + occ_intra

        if CONSENSUS_MODE == "equal":
            w_r = torch.where(n_q > 0, torch.ones_like(n_q), torch.zeros_like(n_q))
        elif CONSENSUS_MODE == "log_occ":
            logits = torch.where(n_q > 0, -0.5 * torch.log1p(n_q), torch.full_like(n_q, -1e9))
            w_r = torch.softmax(logits, dim=3)
        else:
            sigma_sq = torch.clamp(n_q - 1.0, min=0.0) / float(D)
            w_r = 1.0 / (sigma_sq + 1e-2)
            if CONSENSUS_MODE == "option_a":
                w_r = torch.where(n_q > 0, w_r, torch.zeros_like(w_r))

        w_sum = w_r.sum(dim=3, keepdim=True)
        w_norm = torch.where(w_sum > 1e-6, w_r / w_sum.clamp(min=1e-6), torch.full_like(w_r, 1.0 / R))
        out_chunks.append((y_r * w_norm.unsqueeze(-1)).sum(dim=3))

        dS = torch.einsum("bgrcv,bgcd->bgrcvd", Delta, K_c)
        S_next = S.clone()
        occ_next = occ.clone()
        for b_i in range(B):
            for g_i in range(G):
                fc = ck_c[b_i, g_i].reshape(-1)
                fd = dS[b_i, g_i].reshape(-1, D, D)
                S_next[b_i, g_i] = S_next[b_i, g_i].index_add(0, fc, fd)
                occ_next[b_i, g_i] = occ_next[b_i, g_i].index_add(0, fc, torch.ones_like(fc, dtype=torch.float32))
        S = S_next
        occ = occ_next

    return torch.cat(out_chunks, dim=3).reshape(B, H, Lp, D)[:, :, :L, :].to(orig_dtype)


v2m.flash_vault_v2_chunkwise = flash_vault_v2_cpu_differentiable


# ==============================================================================
# 2. METRIC HELPERS (GINI, ENTROPY, MI, GVA OVERLAP)
# ==============================================================================

def gini_coefficient(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    s = x.sum()
    if s <= 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * s)) - (n + 1.0) / n)


def compute_router_stats_for_layer(cq: torch.Tensor, ck: torch.Tensor) -> Dict:
    """
    cq: (B, H=12, R=4, L)
    ck: (B, G=4, R=4, L)
    """
    primes = [17, 19, 23, 29]
    offsets = [0, 17, 36, 59]
    B, H, R, L = cq.shape
    G = ck.size(1)
    hpg = H // G

    key_h_norm, qry_h_norm = [], []
    key_gini, qry_gini = [], []
    key_eff_cells, qry_eff_cells = [], []
    key_dead, qry_dead = [], []
    key_max_share, qry_max_share = [], []
    bank_histograms_key = {}
    bank_histograms_qry = {}

    k_local_banks = []
    q_local_banks = []

    for r, p in enumerate(primes):
        kv = (ck[:, :, r, :] - offsets[r]).reshape(-1)
        qv = (cq[:, :, r, :] - offsets[r]).reshape(-1)
        k_local_banks.append(kv)
        q_local_banks.append(qv)

        kc = torch.bincount(kv, minlength=p).numpy()
        qc = torch.bincount(qv, minlength=p).numpy()
        bank_histograms_key[f"bank_{p}"] = kc.tolist()
        bank_histograms_qry[f"bank_{p}"] = qc.tolist()

        kp = kc / max(1, kc.sum())
        qp = qc / max(1, qc.sum())
        kh = float(-np.sum(kp[kp > 0] * np.log(kp[kp > 0])))
        qh = float(-np.sum(qp[qp > 0] * np.log(qp[qp > 0])))

        key_h_norm.append(round(kh / math.log(p), 4))
        qry_h_norm.append(round(qh / math.log(p), 4))
        key_eff_cells.append(round(math.exp(kh), 2))
        qry_eff_cells.append(round(math.exp(qh), 2))
        key_gini.append(round(gini_coefficient(kc), 4))
        qry_gini.append(round(gini_coefficient(qc), 4))
        key_dead.append(int((kc == 0).sum()))
        qry_dead.append(int((qc == 0).sum()))
        key_max_share.append(round(float(kp.max()), 4))
        qry_max_share.append(round(float(qp.max()), 4))

    # Pairwise bank mutual information I(C_r; C_s) on keys
    N_k = k_local_banks[0].numel()
    raw_kh = [h * math.log(p) for h, p in zip(key_h_norm, primes)]
    mi_pairs = {}
    for r1 in range(4):
        for r2 in range(r1 + 1, 4):
            pc = Counter(zip(k_local_banks[r1].tolist(), k_local_banks[r2].tolist()))
            hp = -sum((v / N_k) * math.log(v / N_k) for v in pc.values())
            mi = max(0.0, raw_kh[r1] + raw_kh[r2] - hp)
            mi_pairs[f"{primes[r1]}x{primes[r2]}"] = round(mi, 4)

    # Empirical within-group 4-tuple collision rate on keys
    grp_colls = []
    for b_i in range(B):
        for g_i in range(G):
            tups = [tuple(int((ck[b_i, g_i, r, t] - offsets[r]).item()) for r in range(4)) for t in range(L)]
            cnt = Counter(tups)
            grp_colls.append(sum(v * (v - 1) for v in cnt.values()) / max(1, L * (L - 1)))

    # Query/Write route agreement at same token position: P(c_q[h] == c_k[g])
    cq_grp = cq.view(B, G, hpg, R, L)
    qk_agree = float((cq_grp == ck.unsqueeze(2)).float().mean().item())

    # 3:1 GVA intra-group head cell overlap P(c_{q,h1} == c_{q,h2}) & distinct cells accessed per 3-head group
    p01 = (cq_grp[:, :, 0] == cq_grp[:, :, 1]).float().mean().item()
    p02 = (cq_grp[:, :, 0] == cq_grp[:, :, 2]).float().mean().item()
    p12 = (cq_grp[:, :, 1] == cq_grp[:, :, 2]).float().mean().item()
    pair_overlap = float((p01 + p02 + p12) / 3.0)

    # Distinct cells accessed by the 3 heads in (B, G, R, L):
    c0, c1, c2 = cq_grp[:, :, 0], cq_grp[:, :, 1], cq_grp[:, :, 2]
    distinct_3 = 1.0 + (c1 != c0).float() + ((c2 != c0) & (c2 != c1)).float()
    mean_distinct_heads = float(distinct_3.mean().item())

    return {
        "key_H_norm": key_h_norm,
        "qry_H_norm": qry_h_norm,
        "key_gini": key_gini,
        "qry_gini": qry_gini,
        "key_effective_cells": key_eff_cells,
        "qry_effective_cells": qry_eff_cells,
        "key_dead_cells": key_dead,
        "qry_dead_cells": qry_dead,
        "key_max_cell_share": key_max_share,
        "qry_max_cell_share": qry_max_share,
        "pairwise_MI_nats": mi_pairs,
        "empirical_group_tuple_collision": round(sum(grp_colls) / len(grp_colls), 6),
        "query_write_same_token_agreement": round(qk_agree, 4),
        "gva_intra_group_pair_overlap": round(pair_overlap, 4),
        "gva_mean_distinct_cells_per_3heads": round(mean_distinct_heads, 3),
        "histograms_key": bank_histograms_key,
        "histograms_qry": bank_histograms_qry,
    }


# ==============================================================================
# 3. MAIN DIAGNOSTIC EXECUTION
# ==============================================================================

def load_val_tokens(data_dir: str, batch_size: int = 4, seq_len: int = 512) -> torch.Tensor:
    row_len = 2049
    tokens_path = os.path.join(data_dir, "val_tokens.bin")
    mm = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    n_rows = len(mm) // row_len
    mm = mm.reshape(n_rows, row_len)
    rows = [torch.from_numpy(mm[i, :seq_len].astype(np.int64)) for i in range(batch_size)]
    return torch.stack(rows, dim=0)


def run_all_diagnostics():
    global CONSENSUS_MODE
    torch.set_num_threads(min(6, os.cpu_count() or 4))

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    data_dir = os.path.join(ROOT_DIR, "data_v2")

    stable_path = os.path.join(ckpt_dir, "vault_v2_wsd_stable.pt")
    best_path = os.path.join(ckpt_dir, "vault_v2_pretrain_best.pt")

    ck_stable = torch.load(stable_path, map_location="cpu", weights_only=False)
    ck_best = torch.load(best_path, map_location="cpu", weights_only=False)

    step_stable = ck_stable.get("step", 5400)
    step_best = ck_best.get("step", -1)
    ce_best = ck_best.get("val_ce", 0.0)
    print(f"=== PHASE B LIVE CHECKPOINT DIAGNOSTICS (Step {step_stable} vs Step {step_best}, PPL={math.exp(ce_best):.2f}) ===")

    model_stable = v2m.VAULTLMv2(vocab_size=len(tok), grad_checkpoint=False).eval()
    model_stable.load_state_dict(ck_stable["model_state"])

    model_best = v2m.VAULTLMv2(vocab_size=len(tok), grad_checkpoint=False).eval()
    model_best.load_state_dict(ck_best["model_state"])

    # Hook routes & beta on both models
    routes_stable, routes_best, betas_best = {}, {}, {}

    def attach_hooks(model: v2m.VAULTLMv2, store_routes: dict, store_betas: dict = None):
        for li, blk in enumerate(model.layers):
            orig_r = blk.attn._route_cells_v2
            def make_r(idx_l, fn):
                def wrapped(qu, ku, sr):
                    cq, ck_t, sq, sk = fn(qu, ku, sr)
                    store_routes[idx_l] = (cq.detach().cpu(), ck_t.detach().cpu())
                    return cq, ck_t, sq, sk
                return wrapped
            blk.attn._route_cells_v2 = make_r(li, orig_r)

            if store_betas is not None:
                orig_bp = blk.attn.beta_proj.forward
                def make_b(idx_l, bfn):
                    def bwrap(x_in):
                        out_b = bfn(x_in)
                        b_val = 0.02 + 0.96 * torch.sigmoid(out_b.detach().float().cpu())
                        store_betas[idx_l] = b_val
                        return out_b
                    return bwrap
                blk.attn.beta_proj.forward = make_b(li, orig_bp)

    attach_hooks(model_stable, routes_stable)
    attach_hooks(model_best, routes_best, betas_best)

    val_x = load_val_tokens(data_dir, batch_size=4, seq_len=512)  # 2,048 real multi-domain holdout tokens
    with torch.no_grad():
        _ = model_stable(val_x, step_ratio=1.0)
        _ = model_best(val_x, step_ratio=1.0)

    # 1. Per-layer router dashboard + migration flip rate (Step 5400 -> Step best) + GVA overlap
    print("\n--- 1. 88-CELL ROUTER TELEMETRY, GINI, KEY vs QUERY ENTROPY, GVA OVERLAP & MIGRATION FLIP RATE ---")
    router_report = {}
    for li in range(16):
        cq_b, ck_b = routes_best[li]
        cq_s, ck_s = routes_stable[li]
        st = compute_router_stats_for_layer(cq_b, ck_b)

        # Router flip rate across WSD decay (Step 5400 -> Step best) on identical holdout tokens
        key_flip = float((ck_b != ck_s).float().mean().item())
        qry_flip = float((cq_b != cq_s).float().mean().item())

        # Ternary weight sign/zero flip rate in kenyon_proj
        w_s = model_stable.layers[li].attn.kenyon_proj.weight.data
        w_b = model_best.layers[li].attn.kenyon_proj.weight.data
        s_scale = w_s.abs().mean().clamp(min=1e-5)
        b_scale = w_b.abs().mean().clamp(min=1e-5)
        q_ws = (w_s / s_scale).round().clamp(-1, 1)
        q_wb = (w_b / b_scale).round().clamp(-1, 1)
        tern_flip = float((q_ws != q_wb).float().mean().item())

        b_t = betas_best[li].reshape(-1)
        beta_stats = {
            "min": round(float(b_t.min().item()), 4),
            "p10": round(float(torch.quantile(b_t, 0.10).item()), 4),
            "median": round(float(torch.quantile(b_t, 0.50).item()), 4),
            "p90": round(float(torch.quantile(b_t, 0.90).item()), 4),
            "max": round(float(b_t.max().item()), 4),
        }

        st["key_route_flip_vs_step5400"] = round(key_flip, 4)
        st["qry_route_flip_vs_step5400"] = round(qry_flip, 4)
        st["ternary_weight_flip_vs_step5400"] = round(tern_flip, 4)
        st["beta_distribution"] = beta_stats
        router_report[f"layer_{li}"] = st

        if li in [0, 3, 7, 11, 15]:
            print(
                f"  L{li:02d} | KeyH={st['key_H_norm']} QryH={st['qry_H_norm']} | "
                f"KeyGini={st['key_gini']} DeadK={st['key_dead_cells']} | "
                f"GVA_3head_cells={st['gva_mean_distinct_cells_per_3heads']:.2f}/3 (ovlp={st['gva_intra_group_pair_overlap']:.3f}) | "
                f"Flip({step_stable}->{step_best}): K={100*key_flip:4.1f}% TernW={100*tern_flip:4.1f}% | "
                f"beta_med={beta_stats['median']:.3f}[{beta_stats['p10']:.2f}..{beta_stats['p90']:.2f}]"
            )

    # Print explicit 88-cell counts for Layer 8 as representative middle layer (Sec 12 Q2!)
    print(f"\n  [Layer 08 Explicit 88-Cell Key Histogram on 2,048 Holdout Tokens x 4 Groups = 8,192 Writes/Bank]:")
    for bname, hvals in router_report["layer_8"]["histograms_key"].items():
        print(f"    {bname:8s}: min={min(hvals):4d}, max={max(hvals):4d}, counts={hvals}")

    # 2. Per-layer Router STE Gradient Ratio Check (Sec 8 Risk 6)
    print("\n--- 2. PER-LAYER ROUTER STE GRADIENT RATIO ||grad W_router||_2 / ||grad W_layer||_2 (SEC 8 RISK 6) ---")
    model_grad = v2m.VAULTLMv2(vocab_size=len(tok), grad_checkpoint=False).train()
    model_grad.load_state_dict(ck_best["model_state"])
    model_grad.zero_grad()
    gx = val_x[:2, :256]
    gy = load_val_tokens(data_dir, batch_size=2, seq_len=257)[:, 1:257]
    loss_g = model_grad.forward_loss(gx, gy, step_ratio=1.0)
    loss_g.backward()

    grad_ratios = {}
    for li, blk in enumerate(model_grad.layers):
        r_sq = 0.0
        tot_sq = 0.0
        for pname, p in blk.named_parameters():
            if p.grad is not None:
                gn2 = float(p.grad.float().pow(2).sum().item())
                tot_sq += gn2
                if "kenyon_proj" in pname or "apl_norm" in pname:
                    r_sq += gn2
        ratio = math.sqrt(r_sq) / (math.sqrt(tot_sq) + 1e-12)
        grad_ratios[f"layer_{li}"] = round(ratio, 6)
    print("  Router/Layer Grad Norm Ratios:", [f"L{i}:{grad_ratios[f'layer_{i}']:.4f}" for i in [0, 4, 8, 12, 15]])
    del model_grad

    # 3. Controlled 3-Way Open-Vocabulary NIAH Diagnostic: Tests A, B, C + Delay Sweep (Sec 5.2)
    print("\n--- 3. CONTROLLED 3-WAY OPEN-VOCAB NIAH DIAGNOSTIC (TESTS A / B / C & DELAY SWEEP, SEC 5.2) ---")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    unseen_pairs = [
        ("ZephyrValve-K7", "CHK-7419"),
        ("CobaltLoom-R3", "CHK-2954"),
        ("QuartzBridge-T9", "CHK-6182"),
        ("ArgonSpindle-M2", "CHK-8305"),
        ("TitanLatch-V5", "CHK-4726"),
        ("HelixBaffle-J8", "CHK-5091"),
    ]
    distractors = [
        f"[Node {i:02d}: BayUnit-{i:02d}] State: ACTIVE | Telemetry: 98.{i%9}% | Checksum: CHK-{1100+37*i:04d}"
        for i in range(1, 25)
    ]

    def chat_ids(user_txt: str, assist_prefix: str = "") -> List[int]:
        return (
            [im_start]
            + tok.encode(f"user\n{user_txt.strip()}", add_special_tokens=False)
            + [im_end]
            + tok.encode("\n", add_special_tokens=False)
            + [im_start]
            + tok.encode("assistant\n" + assist_prefix, add_special_tokens=False)
        )

    @torch.no_grad()
    def greedy_and_tf_score(user_txt: str, assist_prefix: str, target_continuation: str) -> Tuple[str, bool, float, float]:
        pref = chat_ids(user_txt, assist_prefix)
        targ = tok.encode(target_continuation, add_special_tokens=False)
        full = torch.tensor([pref + targ], dtype=torch.long)
        logits = model_best(full, step_ratio=1.0)[0]
        # Teacher-forced digit/token accuracy & mean prob
        s_pos = len(pref)
        tf_hits = 0
        probs_list = []
        for off, tid in enumerate(targ):
            pr = F.softmax(logits[s_pos + off - 1], dim=-1)
            tf_hits += int(pr.argmax().item() == tid)
            probs_list.append(float(pr[tid].item()))
        tf_tok_acc = tf_hits / max(1, len(targ))
        mean_p = sum(probs_list) / max(1, len(probs_list))
        # By induction on causal masking, greedy decoding matches `targ` verbatim iff every token in `targ` is top-1!
        exact_pass = (tf_hits == len(targ))
        tf_pred_ids = [int(logits[s_pos + off - 1].argmax().item()) for off in range(len(targ))]
        gen_str = tok.decode(tf_pred_ids)
        return gen_str.strip(), exact_pass, tf_tok_acc, mean_p

    # Evaluate Delays d in {2, 8, 15, 35} on Test A (Oracle Entity -> Value) across Consensus Modes!
    gap_templates = {
        "d02_compact": lambda e, c: f"[Target] {e}: {c}",
        "d08_medium": lambda e, c: f"[Target: {e}] Status: NOMINAL | {c}",
        "d15_standard": lambda e, c: f"[Target: {e}] Status: NOMINAL | Verification Code: {c}",
        "d35_wide": lambda e, c: f"[Target: {e}] Status: NOMINAL | Thermal: STABLE | Bus: LOCKED | Frame: SYNCED | Verification Code: {c}",
    }

    niah_report = {"test_A_oracle_entity_by_delay": {}, "test_B_entity_copy": {}, "test_C_joint": {}}

    for mode in ["current", "option_a", "equal", "log_occ"]:
        CONSENSUS_MODE = mode
        niah_report["test_A_oracle_entity_by_delay"][mode] = {}
        for gname, gfn in gap_templates.items():
            exact_cnt = 0
            tf_acc_sum = 0.0
            prob_sum = 0.0
            for idx_p, (ent, code) in enumerate(unseen_pairs):
                digits = code.split("-")[1]  # e.g. '7419'
                lines = distractors[:10] + [gfn(ent, code)] + distractors[10:20]
                u_prompt = "System Telemetry Manifest:\n" + "\n".join(lines) + f"\n\nReport the verification code for {ent}."
                _, ex_ok, tf_acc, mp = greedy_and_tf_score(u_prompt, f"- {ent}: CHK-", digits)
                exact_cnt += int(ex_ok)
                tf_acc_sum += tf_acc
                prob_sum += mp
            n_p = len(unseen_pairs)
            niah_report["test_A_oracle_entity_by_delay"][mode][gname] = {
                "greedy_exact_rate": round(exact_cnt / n_p, 3),
                "tf_digit_top1_acc": round(tf_acc_sum / n_p, 3),
                "tf_mean_digit_prob": round(prob_sum / n_p, 3),
            }
        print(f"  [Test A Oracle Entity | mode={mode:8s}] -> {niah_report['test_A_oracle_entity_by_delay'][mode]}")

    CONSENSUS_MODE = "option_a"
    # Test B: Entity Reconstruction Only (no value lookup)
    b_exact, b_tf = 0, 0.0
    # Test C: Joint Entity + Value Retrieval
    c_exact, c_tf = 0, 0.0
    for ent, code in unseen_pairs:
        lines_b = distractors[:8] + [f"[Highlighted Subsystem: {ent}]"] + distractors[8:16]
        up_b = "Telemetry Log:\n" + "\n".join(lines_b) + "\n\nRepeat the Highlighted Subsystem identifier verbatim."
        _, ex_b, tf_b, _ = greedy_and_tf_score(up_b, "- ", ent)
        b_exact += int(ex_b)
        b_tf += tf_b

        lines_c = distractors[:8] + [f"[Highlighted Subsystem: {ent}] -> {code}"] + distractors[8:16]
        up_c = "Telemetry Log:\n" + "\n".join(lines_c) + "\n\nList the Highlighted Subsystem identifier and its CHK code."
        _, ex_c, tf_c, _ = greedy_and_tf_score(up_c, "- ", f"{ent}: {code}")
        c_exact += int(ex_c)
        c_tf += tf_c

    n_p = len(unseen_pairs)
    niah_report["test_B_entity_copy"] = {"greedy_exact_rate": round(b_exact / n_p, 3), "tf_subword_top1_acc": round(b_tf / n_p, 3)}
    niah_report["test_C_joint"] = {"greedy_exact_rate": round(c_exact / n_p, 3), "tf_subword_top1_acc": round(c_tf / n_p, 3)}
    print(f"  [Test B Entity Reconstruction Only] -> {niah_report['test_B_entity_copy']}")
    print(f"  [Test C Joint Entity + Value]       -> {niah_report['test_C_joint']}")

    out_json = os.path.join(ROOT_DIR, "benchmarks", "audit_phase_b_live_checkpoints.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "step_stable": step_stable,
                "step_best": step_best,
                "val_ce_best": ce_best,
                "val_ppl_best": math.exp(ce_best),
                "router_and_gva_by_layer": router_report,
                "router_ste_grad_ratios": grad_ratios,
                "controlled_niah_abc": niah_report,
            },
            f,
            indent=2,
        )
    print(f"\n[+] Saved complete Phase B live-checkpoint audit -> {out_json}")


if __name__ == "__main__":
    run_all_diagnostics()
