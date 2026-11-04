#!/usr/bin/env python3
"""
Exhaustive Mathematical, Conditioning, Full-State Equivalence, Capacity, and Consensus
Audit Suite for FlashVAULT v2 / VAULTLMv2.

Covers:
  1. Six-stage block-doubling unit lower-triangular inverse invariants:
     - Arbitrary random lower-triangular matrices vs. normalized-key Delta Gram matrices
     - Condition number kappa_2(I+L), ||(I+L)^-1||_2, max_s |X^(s)|, max_s |T^(s)|
     - Relative Frobenius error on U, W, and backward gradient magnitude ratios (FP64/FP32/BF16)
  2. All 8 cell-masked Compact WY equivalence patterns (Tests A-H), verifying BOTH:
     - Every intermediate token state S_t (t = 1..L) across multi-chunk & unaligned L=200
     - Full final state S_L and backward gradient L2 relative error + norm ratios
  3. Low-rank shared-subspace key attack (k_i = normalize(U a_i + eps_i)) across ranks
  4. Multi-bank capacity sweep & 4-way consensus comparison:
     - Equal-weight vs. Current Inverse-Occupancy vs. Empty-Masked Log-Occupancy vs.
       Residual-Calibrated Consensus (EMA prediction error E_c), stratified by k-of-4 bank hits.
"""

import json
import math
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ==============================================================================
# 1. EXACT 6-STAGE BLOCK-DOUBLING SOLVER & COMPACT WY REFERENCE
# ==============================================================================

def block_doubling_inv_64(L_strict: torch.Tensor, compute_dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, float, float]:
    """
    Mirrors `_invert_unit_lower_tri_64` in `src/kernels/flash_vault_v2_triton.py` step-for-step.
    Maintains invariant at stage s (s = 0..5, block size BS = 2^(s+1)):
      X^(s) equals the exact inverse of the block-diagonal restriction of (I + L_strict)
      to contiguous diagonal blocks of size 2^(s+1) x 2^(s+1).
    Returns (X_final_fp64, max_abs_X_across_stages, max_abs_T_across_stages).
    """
    C = L_strict.size(-1)
    assert C == 64
    A_neg = (-L_strict).to(compute_dtype)
    idx = torch.arange(C, device=L_strict.device)

    # Stage 0: exact 2x2 diagonal blocks: (I + L_2x2)^-1 = I - L_2x2
    diag_m = (idx[:, None] == idx[None, :]).to(compute_dtype)
    s0_mask = (idx[:, None] == idx[None, :] + 1) & ((idx[:, None] // 2) == (idx[None, :] // 2))
    X = diag_m + torch.where(s0_mask, A_neg, torch.zeros_like(A_neg))

    max_X = float(X.float().abs().max().item())
    max_T = 0.0

    # Stages s = 1..5 merge pairs of BS x BS blocks into BS2 x BS2 blocks (BS = 2, 4, 8, 16, 32)
    for s in range(1, 6):
        BS = 1 << s
        BS2 = 1 << (s + 1)
        same_super = (idx[:, None] // BS2) == (idx[None, :] // BS2)
        sub_mask = same_super & ((idx[:, None] % BS2) >= BS) & ((idx[None, :] % BS2) < BS)
        A_sub = torch.where(sub_mask, A_neg, torch.zeros_like(A_neg))
        if compute_dtype == torch.bfloat16:
            # Match RDNA4 WMMA bf16 input + fp32 accumulator
            T = torch.matmul(X.bfloat16().float(), A_sub.bfloat16().float())
            max_T = max(max_T, float(T.abs().max().item()))
            X = X.float() + torch.matmul(T.bfloat16().float(), X.bfloat16().float())
        else:
            T = torch.matmul(X, A_sub)
            max_T = max(max_T, float(T.abs().max().item()))
            X = X + torch.matmul(T, X)
        max_X = max(max_X, float(X.float().abs().max().item()))

    return X.to(torch.float64), max_X, max_T


def sequential_single_bank_with_history(
    K: torch.Tensor,     # (L, D) float64
    V: torch.Tensor,     # (L, D) float64
    Q: torch.Tensor,     # (L, D) float64
    beta: torch.Tensor,  # (L,) float64
    ck: torch.Tensor,    # (L,) int64 in [0, M_cells)
    cq: torch.Tensor,    # (L,) int64 in [0, M_cells)
    M_cells: int = 88,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (Y_seq (L, D), U_seq (L, D), S_all_steps (L, M_cells, D, D)).
    """
    L, D = K.shape
    S = torch.zeros(M_cells, D, D, dtype=K.dtype, device=K.device)
    S_hist = []
    U_list = []
    Y_list = []
    for t in range(L):
        c_w = int(ck[t].item())
        pred = S[c_w] @ K[t]
        u_t = beta[t] * (V[t] - pred)
        S = S.clone()
        S[c_w] = S[c_w] + torch.outer(u_t, K[t])
        S_hist.append(S)
        U_list.append(u_t)
        c_r = int(cq[t].item())
        Y_list.append(S[c_r] @ Q[t])
    return torch.stack(Y_list, dim=0), torch.stack(U_list, dim=0), torch.stack(S_hist, dim=0)


def chunkwise_wy_single_bank_with_history(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    beta: torch.Tensor,
    ck: torch.Tensor,
    cq: torch.Tensor,
    M_cells: int = 88,
    chunk_size: int = 64,
    inv_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Executes Cell-Masked Compact WY across arbitrary L (including non-multiple of 64)
    and reconstructs intermediate state S_t after EVERY token t=1..L to verify
    zero hidden state divergence.
    """
    L_orig, D = K.shape
    rem = L_orig % chunk_size
    pad = (chunk_size - rem) if rem != 0 else 0
    if pad > 0:
        K = F.pad(K, (0, 0, 0, pad))
        V = F.pad(V, (0, 0, 0, pad))
        Q = F.pad(Q, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad), value=0.0)
        ck = F.pad(ck, (0, pad), value=0)
        cq = F.pad(cq, (0, pad), value=0)

    L = K.size(0)
    NC = L // chunk_size
    C = chunk_size
    S = torch.zeros(M_cells, D, D, dtype=torch.float64)
    idx = torch.arange(C)
    strict_lower = idx[:, None] > idx[None, :]
    causal_inc = idx[:, None] >= idx[None, :]

    S_steps = []
    Delta_all = []
    Y_chunks = []

    stats = {"max_kappa": 1.0, "max_inv_2norm": 1.0, "max_abs_A": 1.0, "max_abs_T": 0.0}

    for nc in range(NC):
        sl = slice(nc * C, (nc + 1) * C)
        Kc, Vc, Qc, bc, ckc, cqc = K[sl].double(), V[sl].double(), Q[sl].double(), beta[sl].double(), ck[sl], cq[sl]

        M_mask = (ckc[:, None] == ckc[None, :]) & strict_lower
        L_strict = torch.where(M_mask, bc[:, None] * (Kc @ Kc.T), torch.zeros(C, C, dtype=torch.float64))

        # Exact conditioning metrics on (I + L_strict)
        I_plus_L = torch.eye(C, dtype=torch.float64) + L_strict
        svals = torch.linalg.svdvals(I_plus_L)
        kappa = float((svals.max() / svals.min().clamp(min=1e-15)).item())
        inv_2norm = float((1.0 / svals.min().clamp(min=1e-15)).item())

        A_inv, max_X, max_T = block_doubling_inv_64(L_strict, compute_dtype=inv_dtype)
        stats["max_kappa"] = max(stats["max_kappa"], kappa)
        stats["max_inv_2norm"] = max(stats["max_inv_2norm"], inv_2norm)
        stats["max_abs_A"] = max(stats["max_abs_A"], max_X)
        stats["max_abs_T"] = max(stats["max_abs_T"], max_T)

        if inv_dtype == torch.bfloat16:
            Uc = (A_inv.bfloat16().float() @ (bc[:, None] * Vc).bfloat16().float()).double()
            Wc = (A_inv.bfloat16().float() @ (bc[:, None] * Kc).bfloat16().float()).double()
            S0_k = S[ckc]  # (C, D, D)
            SW = torch.einsum("cvd,cd->cv", S0_k.bfloat16().float(), Wc.bfloat16().float()).double()
            Delta = Uc - SW
            S0_q = S[cqc]
            y_inter = torch.einsum("cvd,cd->cv", S0_q.bfloat16().float(), Qc.bfloat16().float()).double()
            same_qk = (cqc[:, None] == ckc[None, :]) & causal_inc
            QK = torch.where(same_qk, (Qc.bfloat16().float() @ Kc.bfloat16().float().T).double(), torch.zeros(C, C, dtype=torch.float64))
            y_intra = (QK.bfloat16().float() @ Delta.bfloat16().float()).double()
        elif inv_dtype == torch.float32:
            Uc = (A_inv.float() @ (bc[:, None] * Vc).float()).double()
            Wc = (A_inv.float() @ (bc[:, None] * Kc).float()).double()
            S0_k = S[ckc]
            SW = torch.einsum("cvd,cd->cv", S0_k.float(), Wc.float()).double()
            Delta = Uc - SW
            S0_q = S[cqc]
            y_inter = torch.einsum("cvd,cd->cv", S0_q.float(), Qc.float()).double()
            same_qk = (cqc[:, None] == ckc[None, :]) & causal_inc
            QK = torch.where(same_qk, (Qc.float() @ Kc.float().T).double(), torch.zeros(C, C, dtype=torch.float64))
            y_intra = (QK.float() @ Delta.float()).double()
        else:
            Uc = A_inv @ (bc[:, None] * Vc)
            Wc = A_inv @ (bc[:, None] * Kc)
            S0_k = S[ckc]
            SW = torch.einsum("cvd,cd->cv", S0_k, Wc)
            Delta = Uc - SW
            S0_q = S[cqc]
            y_inter = torch.einsum("cvd,cd->cv", S0_q, Qc)
            same_qk = (cqc[:, None] == ckc[None, :]) & causal_inc
            QK = torch.where(same_qk, Qc @ Kc.T, torch.zeros(C, C, dtype=torch.float64))
            y_intra = QK @ Delta

        Y_chunks.append(y_inter + y_intra)
        Delta_all.append(Delta)

        # Reconstruct intermediate states S_t for every token inside chunk (strict prefix check)
        S_running = S.clone()
        valid_c = min(C, max(0, L_orig - nc * C))
        for i in range(C):
            c_w = int(ckc[i].item())
            if inv_dtype == torch.bfloat16:
                dS_i = torch.outer(Delta[i].float(), Kc[i].bfloat16().float()).double()
            elif inv_dtype == torch.float32:
                dS_i = torch.outer(Delta[i].float(), Kc[i].float()).double()
            else:
                dS_i = torch.outer(Delta[i], Kc[i])
            S_running = S_running.clone()
            S_running[c_w] = S_running[c_w] + dS_i
            if i < valid_c:
                S_steps.append(S_running)
        S = S_running

    Y_out = torch.cat(Y_chunks, dim=0)[:L_orig]
    Delta_out = torch.cat(Delta_all, dim=0)[:L_orig]
    return Y_out, Delta_out, torch.stack(S_steps, dim=0), stats


# ==============================================================================
# 2. AUDIT SUITE RUNNERS
# ==============================================================================

def run_part1_solver_invariants() -> Dict:
    print("=== PART 1: 6-STAGE BLOCK-DOUBLING SOLVER INVARIANTS (SEC 1.2 & 1.3) ===")
    torch.manual_seed(2026)
    C, D = 64, 64
    idx = torch.arange(C)
    strict_lower = idx[:, None] > idx[None, :]
    results = {}

    cases = {}
    # 1. Arbitrary unstructured lower-triangular matrix (entries in [-0.5, 0.5] -> generic unitriangular blowup!)
    cases["generic_unstructured_L"] = torch.where(
        strict_lower, 0.45 * torch.randn(C, C, dtype=torch.float64), torch.zeros(C, C, dtype=torch.float64)
    )
    # 2. Normalized-key Delta Gram: random positive keys, beta in (0.05, 0.96)
    K_rand = F.normalize(torch.exp(1.8 * torch.randn(C, D, dtype=torch.float64).clamp(-3, 3)), dim=-1)
    b_rand = 0.05 + 0.91 * torch.rand(C, dtype=torch.float64)
    cases["delta_gram_random_b096"] = torch.where(strict_lower, b_rand[:, None] * (K_rand @ K_rand.T), torch.zeros(C, C, dtype=torch.float64))

    # 3. Extreme collinear: 64 identical unit keys, beta = 0.99
    K_id = K_rand[:1].expand(C, D).clone()
    b_99 = torch.full((C,), 0.99, dtype=torch.float64)
    cases["delta_gram_64_identical_b099"] = torch.where(strict_lower, b_99[:, None] * (K_id @ K_id.T), torch.zeros(C, C, dtype=torch.float64))

    # 4. Near-collinear rho=0.995, beta=0.98
    eps = F.normalize(torch.randn(C, D, dtype=torch.float64), dim=-1)
    K_nc = F.normalize(0.995 * K_rand[:1] + math.sqrt(1.0 - 0.995**2) * eps, dim=-1)
    b_98 = torch.full((C,), 0.98, dtype=torch.float64)
    cases["delta_gram_near_collinear_rho0995"] = torch.where(strict_lower, b_98[:, None] * (K_nc @ K_nc.T), torch.zeros(C, C, dtype=torch.float64))

    for name, L_mat in cases.items():
        I_plus_L = torch.eye(C, dtype=torch.float64) + L_mat
        A_ref = torch.linalg.solve_triangular(I_plus_L, torch.eye(C, dtype=torch.float64), upper=False, unitriangular=True)
        svals = torch.linalg.svdvals(I_plus_L)
        kappa = float((svals.max() / svals.min()).item())
        inv_2 = float((1.0 / svals.min()).item())

        A_64, maxX_64, maxT_64 = block_doubling_inv_64(L_mat, torch.float64)
        A_32, _, _ = block_doubling_inv_64(L_mat, torch.float32)
        A_16, maxX_16, maxT_16 = block_doubling_inv_64(L_mat, torch.bfloat16)

        rel_64 = float((torch.norm(A_64 - A_ref) / torch.norm(A_ref)).item())
        rel_32 = float((torch.norm(A_32 - A_ref) / torch.norm(A_ref)).item())
        rel_16 = float((torch.norm(A_16 - A_ref) / torch.norm(A_ref)).item())

        results[name] = {
            "kappa_2": kappa,
            "inv_spectral_norm": inv_2,
            "max_abs_X_fp64": maxX_64,
            "max_abs_T_fp64": maxT_64,
            "max_abs_X_bf16": maxX_16,
            "rel_frob_fp64": rel_64,
            "rel_frob_fp32": rel_32,
            "rel_frob_bf16": rel_16,
        }
        print(
            f"  {name:35s} | kappa={kappa:8.2f} | ||A||_2={inv_2:7.2f} | "
            f"max|X|={maxX_64:7.4f} | max|T|={maxT_64:7.4f} | "
            f"err64={rel_64:.2e} | err32={rel_32:.2e} | errBF16={rel_16:.2e}"
        )
        assert rel_64 < 1e-13, f"FP64 exact block-doubling failed on {name}: {rel_64}"
        if name.startswith("delta_gram_"):
            assert maxX_64 <= 1.0 + 1e-12 and maxT_64 <= 1.0 + 1e-12, "Contraction invariant violated!"

    return results


def run_part2_eight_pattern_full_state_tests() -> Dict:
    print("\n=== PART 2: ALL 8 CELL-MASK PATTERNS (TESTS A-H) — EVERY-TOKEN STATE & GRADIENT PARITY (SEC 1.4) ===")
    torch.manual_seed(1337)
    D = 64
    results = {}

    def make_inputs(L: int, identical_k: bool = False, identical_v: bool = False, alt_v: bool = False, high_beta: bool = False):
        if identical_k:
            k0 = F.normalize(torch.exp(1.5 * torch.randn(1, D, dtype=torch.float64)), dim=-1)
            K = k0.repeat(L, 1)
        else:
            K = F.normalize(torch.exp(1.5 * torch.randn(L, D, dtype=torch.float64).clamp(-3, 3)), dim=-1)
        if identical_v:
            V = torch.randn(1, D, dtype=torch.float64).repeat(L, 1)
        elif alt_v:
            v0 = torch.randn(D, dtype=torch.float64)
            signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(L)], dtype=torch.float64)[:, None]
            V = signs * v0[None, :]
        else:
            V = torch.randn(L, D, dtype=torch.float64)
        Q = F.normalize(torch.exp(1.5 * torch.randn(L, D, dtype=torch.float64).clamp(-3, 3)), dim=-1)
        if high_beta:
            beta = 0.94 + 0.04 * torch.rand(L, dtype=torch.float64)
        else:
            beta = 0.10 + 0.80 * torch.rand(L, dtype=torch.float64)
        return K, V, Q, beta

    specs = [
        ("A_all_one_cell_L192", 192, lambda L: torch.zeros(L, dtype=torch.long), dict()),
        ("B_all_distinct_cells_L64", 64, lambda L: torch.arange(L, dtype=torch.long), dict()),
        ("C_alternating_two_cells_L192", 192, lambda L: torch.arange(L, dtype=torch.long) % 2, dict()),
        ("D_four_equal_cells_L192", 192, lambda L: torch.arange(L, dtype=torch.long) % 4, dict()),
        ("E_skewed_63_plus_1_L64", 64, lambda L: torch.tensor([0] * 63 + [1], dtype=torch.long), dict()),
        ("F_identical_K_identical_V_L192", 192, lambda L: torch.zeros(L, dtype=torch.long), dict(identical_k=True, identical_v=True, high_beta=True)),
        ("G_identical_K_alternating_V_L192", 192, lambda L: torch.zeros(L, dtype=torch.long), dict(identical_k=True, alt_v=True, high_beta=True)),
        ("H_random_routing_beta096_L200", 200, lambda L: torch.randint(0, 17, (L,), dtype=torch.long), dict(high_beta=True)),
    ]

    for name, L, route_fn, kw in specs:
        K, V, Q, beta = make_inputs(L, **kw)
        ck = route_fn(L)
        # Probe both matching cells and random cells in Q
        cq = ck.clone()
        if L >= 8:
            cq[::4] =ck.roll(1)[::4]

        Y_seq, U_seq, S_all_seq = sequential_single_bank_with_history(K, V, Q, beta, ck, cq, M_cells=88)
        Y_64, U_64, S_all_64, st64 = chunkwise_wy_single_bank_with_history(K, V, Q, beta, ck, cq, M_cells=88, inv_dtype=torch.float64)
        Y_32, U_32, S_all_32, _ = chunkwise_wy_single_bank_with_history(K, V, Q, beta, ck, cq, M_cells=88, inv_dtype=torch.float32)
        Y_16, U_16, S_all_16, _ = chunkwise_wy_single_bank_with_history(K, V, Q, beta, ck, cq, M_cells=88, inv_dtype=torch.bfloat16)

        # Max intermediate state relative Frobenius error across EVERY token t=1..L!
        step_norms = torch.norm(S_all_seq.reshape(L, -1), dim=-1).clamp(min=1e-12)
        max_step_err_64 = float((torch.norm((S_all_64 - S_all_seq).reshape(L, -1), dim=-1) / step_norms).max().item())
        max_step_err_32 = float((torch.norm((S_all_32 - S_all_seq).reshape(L, -1), dim=-1) / step_norms).max().item())
        max_step_err_16 = float((torch.norm((S_all_16 - S_all_seq).reshape(L, -1), dim=-1) / step_norms).max().item())
        u_rel_16 = float((torch.norm(U_16 - U_seq) / (torch.norm(U_seq) + 1e-12)).item())

        # Also check backward gradient L2 relative error & magnitude ratio (FP32 Compact WY vs FP64 sequential)
        g_loss_w = torch.randn_like(Y_seq)
        K1 = K.clone().requires_grad_(True)
        V1 = V.clone().requires_grad_(True)
        b1 = beta.clone().requires_grad_(True)
        Ys1, _, _ = sequential_single_bank_with_history(K1, V1, Q, b1, ck, cq, M_cells=88)
        (Ys1 * g_loss_w).sum().backward()

        K2 = K.float().detach().requires_grad_(True)
        V2 = V.float().detach().requires_grad_(True)
        b2 = beta.float().detach().requires_grad_(True)
        # Differentiable chunkwise WY solve in BF16-emulated / FP32
        C = 64
        pad = (C - (L % C)) % C
        Kp = F.pad(K2, (0, 0, 0, pad))
        Vp = F.pad(V2, (0, 0, 0, pad))
        Qp = F.pad(Q.float(), (0, 0, 0, pad))
        bp = F.pad(b2, (0, pad))
        ckp = F.pad(ck, (0, pad))
        cqp = F.pad(cq, (0, pad))
        NC = Kp.size(0) // C
        S_run = torch.zeros(88, D, D, dtype=torch.float32)
        idx = torch.arange(C)
        sl_m = idx[:, None] > idx[None, :]
        inc_m = idx[:, None] >= idx[None, :]
        y_chunks = []
        for nc in range(NC):
            s = slice(nc * C, (nc + 1) * C)
            kc_i, vc_i, qc_i, bc_i, ckc_i, cqc_i = Kp[s], Vp[s], Qp[s], bp[s], ckp[s], cqp[s]
            Lm = torch.where((ckc_i[:, None] == ckc_i[None, :]) & sl_m, bc_i[:, None] * (kc_i @ kc_i.T), torch.zeros(C, C))
            Ai = torch.linalg.solve_triangular(torch.eye(C) + Lm, torch.eye(C), upper=False, unitriangular=True).bfloat16().float()
            Uc_i = Ai @ (bc_i[:, None] * vc_i)
            Wc_i = Ai @ (bc_i[:, None] * kc_i)
            Del_i = Uc_i - torch.einsum("cvd,cd->cv", S_run[ckc_i], Wc_i)
            yi = torch.einsum("cvd,cd->cv", S_run[cqc_i], qc_i) + torch.where((cqc_i[:, None] == ckc_i[None, :]) & inc_m, qc_i @ kc_i.T, torch.zeros(C, C)) @ Del_i
            y_chunks.append(yi)
            dS = torch.einsum("cv,cd->cvd", Del_i, kc_i)
            S_next = S_run.clone()
            S_next.index_add_(0, ckc_i, dS)
            S_run = S_next
        Yw2 = torch.cat(y_chunks, dim=0)[:L]
        (Yw2 * g_loss_w.float()).sum().backward()

        dk_rel = float((torch.norm(K2.grad.double() - K1.grad) / (torch.norm(K1.grad) + 1e-12)).item())
        dk_ratio = float((torch.norm(K2.grad.double()) / (torch.norm(K1.grad) + 1e-12)).item())
        db_rel = float((torch.norm(b2.grad.double() - b1.grad) / (torch.norm(b1.grad) + 1e-12)).item())
        db_ratio = float((torch.norm(b2.grad.double()) / (torch.norm(b1.grad) + 1e-12)).item())

        results[name] = {
            "L": L,
            "kappa_2": st64["max_kappa"],
            "max_step_S_rel_fp64": max_step_err_64,
            "max_step_S_rel_fp32": max_step_err_32,
            "max_step_S_rel_bf16": max_step_err_16,
            "U_rel_bf16": u_rel_16,
            "dK_rel_l2": dk_rel,
            "dK_norm_ratio": dk_ratio,
            "dBeta_rel_l2": db_rel,
            "dBeta_norm_ratio": db_ratio,
        }
        print(
            f"  {name:33s} | max_t S_err(FP64={max_step_err_64:.2e}, FP32={max_step_err_32:.2e}, BF16={max_step_err_16:.2e}) | "
            f"dK_rel={dk_rel:.2e} (ratio={dk_ratio:.4f}) | dBeta_ratio={db_ratio:.4f}"
        )
        assert max_step_err_64 < 1e-13, f"Every-token state equivalence failed on {name}!"

    return results


def run_part3_subspace_capacity_and_consensus() -> Dict:
    print("\n=== PART 3: LOW-RANK SUBSPACE ATTACK (SEC 7.2) & 4-WAY CONSENSUS + CAPACITY SWEEP (SEC 3 & 6) ===")
    torch.manual_seed(99)
    D = 64
    primes = [17, 19, 23, 29]
    offsets = [0, 17, 36, 59]
    out = {"low_rank_subspace": {}, "capacity_and_consensus": {}}

    # Fixed ternary Kenyon projection for deterministic routing evaluation
    W_ken = torch.randint(-1, 2, (88, D), dtype=torch.float32)

    def route_4banks(keys: torch.Tensor, correlated: bool = False) -> torch.Tensor:
        # keys: (N, D)
        z = keys @ W_ken.T  # (N, 88)
        z = (z - z.mean(dim=-1, keepdim=True)) / (z.std(dim=-1, keepdim=True) + 1e-5)
        banks = []
        for r, p in enumerate(primes):
            sl = z[:, offsets[r]:offsets[r] + p]
            if correlated and r > 0:
                # Inject deliberate cross-bank dependence to test correlated routing regime (Sec 3)
                sl = 0.65 * z[:, :min(17, p)] if p >= 17 else sl
                if p > 17:
                    sl = torch.cat([0.70 * z[:, :17] + 0.30 * sl[:, :17], sl[:, 17:]], dim=-1)
            banks.append(offsets[r] + sl.argmax(dim=-1))
        return torch.stack(banks, dim=-1)  # (N, 4)

    # 3A: Low-Rank Shared Subspace Attack (Sec 7.2): k_i = normalize(U_r a_i + 0.08 eps_i), N=256 unique pairs
    N_sub = 256
    V_dict = F.normalize(torch.randn(N_sub, D), dim=-1)
    for rank_r in [2, 4, 8, 16, 32, 64]:
        U_basis, _ = torch.linalg.qr(torch.randn(D, rank_r))
        a = torch.randn(N_sub, rank_r)
        eps = 0.08 * torch.randn(N_sub, D)
        K_sub = F.normalize(F.relu(a @ U_basis.T + eps) + 1e-4, dim=-1)
        cells = route_4banks(K_sub)

        S = torch.zeros(88, D, D)
        occ = torch.zeros(88)
        for i in range(N_sub):
            for r in range(4):
                c = int(cells[i, r].item())
                S[c] = S[c] + 0.90 * torch.outer(V_dict[i] - S[c] @ K_sub[i], K_sub[i])
                occ[c] += 1.0

        # Query all 256 associations
        hits = 0
        cos_sum = 0.0
        for i in range(N_sub):
            ys = torch.stack([S[int(cells[i, r].item())] @ K_sub[i] for r in range(4)])
            ns = torch.tensor([occ[int(cells[i, r].item())] for r in range(4)])
            w = torch.where(ns > 0, torch.softmax(-0.5 * torch.log1p(ns), dim=0), torch.zeros(4))
            w = w / w.sum().clamp(min=1e-6)
            y = (w[:, None] * ys).sum(dim=0)
            sims = F.cosine_similarity(y[None, :], V_dict, dim=-1)
            hits += int(sims.argmax().item() == i)
            cos_sum += float(sims[i].item())
        acc = hits / N_sub
        mean_cos = cos_sum / N_sub
        out["low_rank_subspace"][f"rank_{rank_r}"] = {"top1_acc": acc, "mean_cos": mean_cos}
        print(f"  [Low-Rank Subspace N=256] rank={rank_r:2d} -> top1_acc={100*acc:6.2f}% | mean_cos={mean_cos:.4f}")

    # 3B: Capacity & 4-Way Consensus Comparison (Sec 3 & Sec 6): N_unique in [256, 512, 1024, 2048, 4096]
    # Also inject 20% unqueried empty-bank probes / partial collisions so empty-cell & residual calibration are stressed!
    print("\n  --- Capacity & 4-Way Consensus Sweep (Equal vs InvOcc vs LogOcc vs Residual-Calibrated) ---")
    for N_u in [256, 512, 1024, 2048, 4096]:
        K_u = F.normalize(torch.exp(1.6 * torch.randn(N_u, D).clamp(-3, 3)), dim=-1)
        V_u = F.normalize(torch.randn(N_u, D), dim=-1)
        cells = route_4banks(K_u, correlated=False)

        S = torch.zeros(88, D, D)
        occ = torch.zeros(88)
        E_res = torch.zeros(88)  # Residual-Calibrated EMA write error ||v - S_pre k||^2 (Sec 3.1)

        for i in range(N_u):
            for r in range(4):
                c = int(cells[i, r].item())
                err_sq = float(((V_u[i] - S[c] @ K_u[i]).pow(2).sum()).item())
                E_res[c] = 0.85 * E_res[c] + 0.15 * err_sq
                S[c] = S[c] + 0.90 * torch.outer(V_u[i] - S[c] @ K_u[i], K_u[i])
                occ[c] += 1.0

        # Evaluate random subset of 256 items across all 4 consensus methods + k-of-4 bank breakdown
        probe_idx = torch.linspace(0, N_u - 1, min(256, N_u)).long()
        methods = {"equal": 0, "curr_inv_occ": 0, "log_occ": 0, "residual_cal": 0}
        by_k_correct = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}  # [equal_hits, res_hits] / count
        k_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        for idx_i in probe_idx.tolist():
            ys = torch.stack([S[int(cells[idx_i, r].item())] @ K_u[idx_i] for r in range(4)])  # (4, D)
            ns = torch.tensor([occ[int(cells[idx_i, r].item())] for r in range(4)])
            es = torch.tensor([E_res[int(cells[idx_i, r].item())] for r in range(4)])

            # Simulate occasional query bank slip on 1 bank (realistic noisy query representation!)
            if idx_i % 3 == 0:
                ys[0] = torch.zeros(D)
                ns[0] = 0.0
                es[0] = 1.0

            # Individual bank top-1 correctness
            bank_ok = 0
            for r in range(4):
                if ys[r].norm() > 1e-6:
                    if int((V_u @ F.normalize(ys[r], dim=-1)).argmax().item()) == idx_i:
                        bank_ok += 1
            k_counts[bank_ok] += 1

            # 1. Equal weight (with empty mask if n>0 vs raw equal: raw equal has same direction as masked equal!)
            w_eq = torch.full((4,), 0.25)
            # 2. Current inverse-occupancy
            sig = torch.clamp(ns - 1.0, min=0.0) / float(D)
            w_inv = (1.0 / (sig + 1e-2))
            w_inv = w_inv / w_inv.sum()
            # 3. Empty-masked Log-occupancy: softmax(-0.5 * log(1+n)) on n > 0
            logits_log = torch.where(ns > 0, -0.5 * torch.log1p(ns), torch.full((4,), -1e9))
            w_log = torch.softmax(logits_log, dim=0)
            # 4. Residual-calibrated + norm confidence on n > 0
            ynorm = ys.norm(dim=-1)
            logits_res = torch.where(ns > 0, 2.0 * ynorm - 1.2 * es - 0.25 * torch.log1p(ns), torch.full((4,), -1e9))
            w_res = torch.softmax(logits_res, dim=0)

            for m_name, w_vec in [("equal", w_eq), ("curr_inv_occ", w_inv), ("log_occ", w_log), ("residual_cal", w_res)]:
                y_comb = (w_vec[:, None] * ys).sum(dim=0)
                pred = int((V_u @ F.normalize(y_comb, dim=-1)).argmax().item())
                ok = int(pred == idx_i)
                methods[m_name] += ok
                if m_name == "curr_inv_occ":
                    by_k_correct[bank_ok][0] += ok
                elif m_name == "residual_cal":
                    by_k_correct[bank_ok][1] += ok

        M_tot = len(probe_idx)
        accs = {k: v / M_tot for k, v in methods.items()}
        k_break = {
            f"{k}_banks_correct(n={k_counts[k]})": {
                "curr_inv_occ": round(by_k_correct[k][0] / max(1, k_counts[k]), 3),
                "residual_cal": round(by_k_correct[k][1] / max(1, k_counts[k]), 3),
            }
            for k in range(5) if k_counts[k] > 0
        }
        out["capacity_and_consensus"][f"N_{N_u}"] = {"top1_acc": accs, "by_num_correct_banks": k_break}
        print(
            f"  N_unique={N_u:4d} (writes/cell~{4*N_u/88:5.1f}) | "
            f"Equal={100*accs['equal']:5.1f}% | CurrInvOcc={100*accs['curr_inv_occ']:5.1f}% | "
            f"LogOcc={100*accs['log_occ']:5.1f}% | ResidualCal={100*accs['residual_cal']:5.1f}% | k-hits={k_break}"
        )

    return out


def main():
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    r1 = run_part1_solver_invariants()
    r2 = run_part2_eight_pattern_full_state_tests()
    r3 = run_part3_subspace_capacity_and_consensus()

    out_path = os.path.join(ROOT_DIR, "benchmarks", "audit_phase_b_kernel_and_capacity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"part1_solver_invariants": r1, "part2_eight_patterns": r2, "part3_subspace_and_consensus": r3}, f, indent=2)
    print(f"\n[+] Saved complete Phase B kernel & capacity audit -> {out_path}")


if __name__ == "__main__":
    main()
