# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# FlashVAULT v2 / v2.3: True Grouped-Vault Attention (GVA) Chunkwise Compact WY Kernel
# Built for Triton 3.8 + ROCm 7.2 (AMD RDNA4 gfx1201 native bf16 WMMA).
#
# Key v2.3 Upgrades:
#   1. Generalized 4-Bank Co-Prime Voronoi Geometries (`prime_banks`, e.g. `(17,19,23,29)` = 88 cells
#      with `BANK_PAD=32`, or `(47,53,59,61)` = 220 cells with `BANK_PAD=64` -> 8,965,109 CRT addresses).
#   2. 100% Branch-Free WMMA Scan (`num_warps=8` for BF16 WMMA) — zero divergent CFG `tl.dot` blocks,
#      compiling in ~2 seconds on ROCm LLVM with zero register-allocator hangs.
#   3. Chunk-Parallel Occupancy Consensus (`Grid: (num_chunks, B * G, R)` + `cumsum` + `(num_chunks, B * H)`)
#      and Fused 4-Bank Intra-Chunk Readout (eliminates `OutBanks [B, H, 4, L, D]` HBM roundtrip).
#   4. Fused Contiguous `(B, L, C)` Causal Depthwise 1D Conv + SiLU (`causal_depthwise_conv1d`).

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

    class _TritonStub:
        @staticmethod
        def jit(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

        @staticmethod
        def cdiv(x: int, y: int) -> int:
            return (x + y - 1) // y

    class _TLStub:
        constexpr = int

    triton = _TritonStub()  # type: ignore[assignment]
    tl = _TLStub()  # type: ignore[assignment]


# ==============================================================================
# 1. EXACT PYTORCH REFERENCE FOR TRUE GVA PARITY VERIFICATION
# ==============================================================================

def flash_vault_v2_recurrent_reference(
    phi_q: torch.Tensor,   # (B, H, L, D)
    phi_k: torch.Tensor,   # (B, G, L, D)
    v: torch.Tensor,       # (B, G, L, D)
    beta: torch.Tensor,    # (B, G, L)
    cell_q: torch.Tensor,  # (B, H, R, L) int64 in [0, num_cells)
    cell_k: torch.Tensor,  # (B, G, R, L) int64 in [0, num_cells)
    num_cells: int = 88,
    v2_1_consensus: bool = False,
    v2_2_beta_occ: bool = False,
) -> torch.Tensor:
    """
    Exact step-by-step recurrent ground truth for True GVA FlashVAULT v2 / v2.1 / v2.2 / v2.3.
    """
    B, H, L, D = phi_q.shape
    _, G, _, _ = phi_k.shape
    R = cell_k.size(2)
    heads_per_group = H // G
    orig_dtype = v.dtype

    pq = phi_q.float()
    pk = phi_k.float()
    vf = v.float()
    bf = beta.float()
    bf_occ = beta.detach().float()

    S = torch.zeros(B, G, num_cells, D, D, device=v.device, dtype=torch.float32)
    occ = torch.zeros(B, G, num_cells, device=v.device, dtype=torch.float32)
    outputs = []

    b_idx_g = torch.arange(B, device=v.device)[:, None]
    g_idx_g = torch.arange(G, device=v.device)[None, :]

    for t in range(L):
        kt = pk[:, :, t, :]    # (B, G, D)
        vt = vf[:, :, t, :]    # (B, G, D)
        bt = bf[:, :, t]       # (B, G)
        bt_occ = bf_occ[:, :, t]

        for r in range(R):
            ck = cell_k[:, :, r, t]  # (B, G)
            S_k = S[b_idx_g, g_idx_g, ck]  # (B, G, D, D)
            pred_v = torch.einsum("bgvd,bgd->bgv", S_k, kt)
            delta = bt.unsqueeze(-1) * (vt - pred_v)
            S[b_idx_g, g_idx_g, ck] = S_k + torch.einsum("bgv,bgd->bgvd", delta, kt)
            if v2_2_beta_occ:
                occ[b_idx_g, g_idx_g, ck] = occ[b_idx_g, g_idx_g, ck] + bt_occ
            else:
                occ[b_idx_g, g_idx_g, ck] = occ[b_idx_g, g_idx_g, ck] + 1.0

        y_heads = []
        for h in range(H):
            g = h // heads_per_group
            qt_h = pq[:, h, t, :]  # (B, D)
            y_banks = []
            w_banks = []
            b_1d = torch.arange(B, device=v.device)
            for r in range(R):
                cq = cell_q[:, h, r, t]          # (B,)
                S_q = S[b_1d, g, cq]             # (B, D, D)
                n_q = occ[b_1d, g, cq]           # (B,)
                y_r = torch.einsum("bvd,bd->bv", S_q, qt_h)
                if v2_2_beta_occ:
                    w_r = torch.where(n_q > 1e-3, torch.rsqrt(1.0 + n_q), torch.zeros_like(n_q))
                elif v2_1_consensus:
                    w_r = torch.where(n_q > 0.0, torch.rsqrt(1.0 + n_q), torch.zeros_like(n_q))
                else:
                    sigma_sq = torch.clamp(n_q - 1.0, min=0.0) / float(D)
                    w_r = 1.0 / (sigma_sq + 1e-2)
                y_banks.append(y_r)
                w_banks.append(w_r)

            w_stack = torch.stack(w_banks, dim=-1)  # (B, R)
            w_sum = w_stack.sum(dim=-1, keepdim=True)
            if v2_1_consensus or v2_2_beta_occ:
                w_norm = torch.where(
                    w_sum > 1e-6,
                    w_stack / w_sum.clamp(min=1e-6),
                    torch.full_like(w_stack, 1.0 / float(R)),
                )
            else:
                w_norm = w_stack / w_sum.clamp(min=1e-6)
            y_stack = torch.stack(y_banks, dim=-2)  # (B, R, D)
            y_h = (y_stack * w_norm.unsqueeze(-1)).sum(dim=-2)  # (B, D)
            y_heads.append(y_h)

        outputs.append(torch.stack(y_heads, dim=1))  # (B, H, D)

    return torch.stack(outputs, dim=2).to(orig_dtype)


# ==============================================================================
# 2. TRITON 3.8 HELPERS & BRANCH-FREE FORWARD KERNELS
# ==============================================================================

@triton.jit
def _mdot(a, b, USE_BF16_WMMA: tl.constexpr):
    if USE_BF16_WMMA:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _invert_unit_lower_tri_64(A_neg, idx, USE_BF16_WMMA: tl.constexpr):
    diag_m = (idx[:, None] == idx[None, :]).to(tl.float32)
    X = diag_m + tl.where((idx[:, None] == idx[None, :] + 1) & ((idx[:, None] // 2) == (idx[None, :] // 2)), A_neg, 0.0)

    for s in tl.static_range(1, 6):
        BS = 1 << s
        BS2 = 1 << (s + 1)
        same_super = (idx[:, None] // BS2) == (idx[None, :] // BS2)
        sub_mask = same_super & ((idx[:, None] % BS2) >= BS) & ((idx[None, :] % BS2) < BS)
        A_sub = tl.where(sub_mask, A_neg, 0.0)
        T = _mdot(X, A_sub, USE_BF16_WMMA)
        X = X + _mdot(T, X, USE_BF16_WMMA)

    return X


@triton.jit
def _vault_v2_wy_fwd(
    K_ptr, V_ptr, Beta_ptr, CellK_ptr,
    U_ptr, W_ptr,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_bb, stride_bg, stride_bl,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    G: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bg // G
    g = bg % G

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)

    k_ptrs = K_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd
    b_ptrs = Beta_ptr + b * stride_bb + g * stride_bg + offs_c * stride_bl
    ck_ptrs = CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl

    k_tile = tl.load(k_ptrs).to(tl.float32)
    v_tile = tl.load(v_ptrs).to(tl.float32)
    beta_vec = tl.load(b_ptrs).to(tl.float32)
    ck_vec = tl.load(ck_ptrs)

    idx = tl.arange(0, CHUNK_SIZE)
    strict_lower = idx[:, None] > idx[None, :]
    same_cell = (ck_vec[:, None] == ck_vec[None, :]) & strict_lower

    kk = _mdot(k_tile, tl.trans(k_tile), USE_BF16_WMMA)
    A_neg = tl.where(same_cell, -beta_vec[:, None] * kk, 0.0)
    T_inv = _invert_unit_lower_tri_64(A_neg, idx, USE_BF16_WMMA)

    bv = beta_vec[:, None] * v_tile
    bk = beta_vec[:, None] * k_tile
    u_tile = _mdot(T_inv, bv, USE_BF16_WMMA)
    w_tile = _mdot(T_inv, bk, USE_BF16_WMMA)

    out_off = b * stride_ub + g * stride_ug + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    tl.store(U_ptr + out_off, u_tile)
    tl.store(W_ptr + out_off, w_tile)


@triton.jit
def _vault_v2_chunk_occ_hist_fwd(
    CellK_ptr, Beta_ptr, ChunkHist_ptr,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_bb, stride_bg, stride_bl,
    stride_hb, stride_hg, stride_hr, stride_hm, stride_hp,
    G: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BANK_PAD: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    V2_2_BETA_OCC: tl.constexpr,
):
    """Grid: (num_chunks, B * G, R) — 100% chunk-parallel per-bank cell occupancy histogram."""
    m = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, C1, tl.where(r32 == 2, C2, C3)))

    offs_l = m * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_p = tl.arange(0, BANK_PAD)

    ck = tl.load(CellK_ptr + b * stride_ckb + g * stride_ckg + r * stride_ckr + offs_l * stride_ckl) - base_cell
    if V2_2_BETA_OCC:
        beta_vec = tl.load(Beta_ptr + b * stride_bb + g * stride_bg + offs_l * stride_bl).to(tl.float32)
        hist = tl.sum(tl.where(ck[:, None] == offs_p[None, :], beta_vec[:, None], 0.0), axis=0)
    else:
        hist = tl.sum((ck[:, None] == offs_p[None, :]).to(tl.float32), axis=0)

    out_ptrs = (
        ChunkHist_ptr + b * stride_hb + g * stride_hg + r * stride_hr + m * stride_hm + offs_p.to(tl.int64) * stride_hp
    )
    tl.store(out_ptrs, hist)


@triton.jit
def _vault_v2_occ_weights_chunk_fwd(
    CellQ_ptr, CellK_ptr, Beta_ptr, ChunkPrior_ptr, Weight_ptr,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_bb, stride_bg, stride_bl,
    stride_pb, stride_pg, stride_pr, stride_pm, stride_pp,
    stride_wb, stride_wh, stride_wr, stride_wl,
    num_banks,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BANK_PAD: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    V2_1_CONSENSUS: tl.constexpr,
    V2_2_BETA_OCC: tl.constexpr,
    OCC_ALPHA: tl.constexpr = 0.5,
):
    """Grid: (num_chunks, B * H) — 100% chunk-parallel exact occupancy consensus weights."""
    m = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    idx = tl.arange(0, CHUNK_SIZE)
    offs_l = m * CHUNK_SIZE + idx.to(tl.int64)
    causal_mask = idx[:, None] >= idx[None, :]

    cq_base = CellQ_ptr + b * stride_cqb + h * stride_cqh
    ck_base = CellK_ptr + b * stride_ckb + g * stride_ckg
    p_base = ChunkPrior_ptr + b * stride_pb + g * stride_pg + m * stride_pm
    w_base = Weight_ptr + b * stride_wb + h * stride_wh

    if V2_2_BETA_OCC:
        beta_vec = tl.load(Beta_ptr + b * stride_bb + g * stride_bg + offs_l * stride_bl).to(tl.float32)

    w_sum = tl.zeros([CHUNK_SIZE], dtype=tl.float32)
    for r in range(num_banks):
        r64 = r.to(tl.int64)
        base_cell = tl.where(r == 0, 0, tl.where(r == 1, C1, tl.where(r == 2, C2, C3)))
        cq = tl.load(cq_base + r64 * stride_cqr + offs_l * stride_cql)
        ck = tl.load(ck_base + r64 * stride_ckr + offs_l * stride_ckl)
        cq_loc = tl.minimum(tl.maximum(cq - base_cell, 0), BANK_PAD - 1)
        occ_prior = tl.load(p_base + r64 * stride_pr + cq_loc * stride_pp).to(tl.float32)

        same_qk = (cq[:, None] == ck[None, :]) & causal_mask
        if V2_2_BETA_OCC:
            occ_intra = tl.sum(tl.where(same_qk, beta_vec[None, :], 0.0), axis=1)
            n_tot = occ_prior + occ_intra
            if OCC_ALPHA == 0.5:
                w_r = tl.where(n_tot > 1e-3, tl.rsqrt(1.0 + n_tot), 0.0)
            elif OCC_ALPHA == 1.0:
                w_r = tl.where(n_tot > 1e-3, 1.0 / (1.0 + n_tot), 0.0)
            elif OCC_ALPHA == 1.5:
                inv_op1 = 1.0 / (1.0 + n_tot)
                w_r = tl.where(n_tot > 1e-3, tl.rsqrt(1.0 + n_tot) * inv_op1, 0.0)
            else:
                w_r = tl.where(n_tot > 1e-3, tl.exp2(-OCC_ALPHA * tl.log2(1.0 + n_tot)), 0.0)
        else:
            occ_intra = tl.sum(same_qk.to(tl.float32), axis=1)
            n_tot = occ_prior + occ_intra
            if V2_1_CONSENSUS:
                w_r = tl.where(n_tot > 0.0, tl.rsqrt(1.0 + n_tot), 0.0)
            else:
                sigma_sq = tl.maximum(n_tot - 1.0, 0.0) / float(D)
                w_r = 1.0 / (sigma_sq + 1e-2)

        tl.store(w_base + r64 * stride_wr + offs_l * stride_wl, w_r)
        w_sum += w_r

    inv_w_sum = 1.0 / tl.maximum(w_sum, 1e-6)
    if V2_1_CONSENSUS or V2_2_BETA_OCC:
        uniform_w = 1.0 / num_banks.to(tl.float32)
        for r in range(num_banks):
            r64 = r.to(tl.int64)
            w_r = tl.load(w_base + r64 * stride_wr + offs_l * stride_wl)
            tl.store(w_base + r64 * stride_wr + offs_l * stride_wl, tl.where(w_sum > 1e-6, w_r * inv_w_sum, uniform_w))
    else:
        for r in range(num_banks):
            r64 = r.to(tl.int64)
            w_r = tl.load(w_base + r64 * stride_wr + offs_l * stride_wl)
            tl.store(w_base + r64 * stride_wr + offs_l * stride_wl, w_r * inv_w_sum)


@triton.jit
def _vault_v2_inter_scan_fwd(
    Q_ptr, K_ptr, U_ptr, W_ptr, CellQ_ptr, CellK_ptr,
    YCross_ptr, S_ckpt_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_yb, stride_yh, stride_yr, stride_yl, stride_yd,
    stride_sb, stride_sg, stride_sm, stride_sdvblk, stride_sc, stride_sflat,
    num_chunks,
    G: tl.constexpr,
    R: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FLAT_DVD: tl.constexpr,
    BANK_PAD: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    SAVE_CHECKPOINTS: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
    SHARED_GROUP_CELL_Q: tl.constexpr = False,
):
    """
    Grid: (B * G * R, D // BLOCK_DV) — 100% branch-free WMMA scan supporting arbitrary 4-bank Co-Prime widths!
    """
    bgr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bgr % R
    bg = bgr // R
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, C1, tl.where(r32 == 2, C2, C3)))
    bank_size = tl.where(r32 == 0, P0, tl.where(r32 == 1, P1, tl.where(r32 == 2, P2, P3)))

    offs_dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV).to(tl.int64)
    offs_dk = tl.arange(0, D).to(tl.int64)
    offs_p = tl.arange(0, BANK_PAD)
    offs_flat = tl.arange(0, FLAT_DVD).to(tl.int64)
    idx_c = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    c_mask = offs_p[:, None] < bank_size

    S_reg = tl.zeros([BANK_PAD, FLAT_DVD], dtype=tl.float32)

    k_grp_base = b * stride_kb + g * stride_kg
    u_bank_base = b * stride_ub + g * stride_ug + r * stride_ur
    ck_bank_base = b * stride_ckb + g * stride_ckg + r * stride_ckr
    s_grp_base = (
        b * stride_sb + g * stride_sg + dv_blk * stride_sdvblk
        + (base_cell + offs_p[:, None]).to(tl.int64) * stride_sc
        + offs_flat[None, :] * stride_sflat
    )

    for m in range(num_chunks):
        m64 = m.to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        if SAVE_CHECKPOINTS:
            if CKPT_INTERVAL == 1:
                tl.store(S_ckpt_ptr + s_grp_base + m64 * stride_sm, S_reg.to(S_ckpt_ptr.dtype.element_ty), mask=c_mask)
            else:
                if (m % CKPT_INTERVAL) == 0:
                    ckpt_idx = (m // CKPT_INTERVAL).to(tl.int64)
                    tl.store(S_ckpt_ptr + s_grp_base + ckpt_idx * stride_sm, S_reg.to(S_ckpt_ptr.dtype.element_ty), mask=c_mask)

        k_tile = tl.load(K_ptr + k_grp_base + offs_l[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)
        ck_vec = tl.load(CellK_ptr + ck_bank_base + offs_l * stride_ckl) - base_cell
        ok_f = (ck_vec[:, None] == offs_p[None, :]).to(tl.float32)

        off_u_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        u_sub = tl.load(U_ptr + off_u_sub).to(tl.float32)

        s_ck = tl.reshape(_mdot(ok_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        delta_sub = u_sub - tl.sum(s_ck * w_tile[:, None, :], axis=2)
        tl.store(Delta_ptr + off_u_sub, delta_sub)

        if SHARED_GROUP_CELL_Q:
            h0 = g * HEADS_PER_GROUP
            cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h0 * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
            oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)
            s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = h0 + j
                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
                off_y_sub = b * stride_yb + h * stride_yh + r * stride_yr + offs_l[:, None] * stride_yl + offs_dv[None, :] * stride_yd
                tl.store(YCross_ptr + off_y_sub, y_cross_sub)
        else:
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = g * HEADS_PER_GROUP + j
                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
                oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)
                s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
                y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
                off_y_sub = b * stride_yb + h * stride_yh + r * stride_yr + offs_l[:, None] * stride_yl + offs_dv[None, :] * stride_yd
                tl.store(YCross_ptr + off_y_sub, y_cross_sub)

        upd_flat = tl.reshape(delta_sub[:, :, None] * k_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        S_reg += _mdot(tl.trans(ok_f), upd_flat, USE_BF16_WMMA)


@triton.jit
def _vault_v2_intra_readout_fused_fwd(
    Q_ptr, K_ptr, Delta_ptr, YCross_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_yb, stride_yh, stride_yr, stride_yl, stride_yd,
    stride_wb, stride_wh, stride_wr, stride_wl,
    stride_ob, stride_oh, stride_ol, stride_od,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)

    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    idx = tl.arange(0, CHUNK_SIZE)
    causal = idx[:, None] >= idx[None, :]

    q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd).to(tl.float32)
    k_tile = tl.load(K_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    qk_raw = _mdot(q_tile, tl.trans(k_tile), USE_BF16_WMMA)

    out_acc = tl.zeros([CHUNK_SIZE, D], dtype=tl.float32)
    for r in tl.static_range(0, 4):
        delta_tile = tl.load(
            Delta_ptr + b * stride_ub + g * stride_ug + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
        ).to(tl.float32)
        y_cross_tile = tl.load(
            YCross_ptr + b * stride_yb + h * stride_yh + r * stride_yr + offs_c[:, None] * stride_yl + offs_d[None, :] * stride_yd
        ).to(tl.float32)
        cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_c * stride_cql)
        ck_vec = tl.load(CellK_ptr + b * stride_ckb + g * stride_ckg + r * stride_ckr + offs_c * stride_ckl)
        w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_c * stride_wl).to(tl.float32)

        same_qk = (cq_vec[:, None] == ck_vec[None, :]) & causal
        attn_qk = tl.where(same_qk, qk_raw, 0.0)
        y_intra_tile = _mdot(attn_qk, delta_tile, USE_BF16_WMMA)
        out_acc += w_norm[:, None] * (y_cross_tile + y_intra_tile)

    off_o = b * stride_ob + h * stride_oh + offs_c[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(Out_ptr + off_o, out_acc)


# ==============================================================================
# 3. TRITON 3.8 BACKWARD KERNELS (100% ATOMIC-FREE & BRANCH-FREE)
# ==============================================================================

@triton.jit
def _vault_v2_intra_readout_bwd(
    Q_ptr, K_ptr, Delta_ptr, dOut_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    dDeltaIntra_ptr, dQ_intra_ptr, dK_intra_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_yb, stride_yh, stride_yr, stride_yl, stride_yd,
    stride_wb, stride_wh, stride_wr, stride_wl,
    G: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bg // G
    g = bg % G

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    idx = tl.arange(0, CHUNK_SIZE)

    k_tile = tl.load(K_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    off_u = b * stride_ub + g * stride_ug + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    delta_tile = tl.load(Delta_ptr + off_u).to(tl.float32)
    ck_vec = tl.load(CellK_ptr + b * stride_ckb + g * stride_ckg + r * stride_ckr + offs_c * stride_ckl)

    ddelta_acc = tl.zeros([CHUNK_SIZE, D], dtype=tl.float32)
    dk_acc = tl.zeros([CHUNK_SIZE, D], dtype=tl.float32)

    for j in tl.static_range(0, HEADS_PER_GROUP):
        h = g * HEADS_PER_GROUP + j
        q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd).to(tl.float32)
        dout_tile = tl.load(dOut_ptr + b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd).to(tl.float32)
        cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_c * stride_cql)
        w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_c * stride_wl).to(tl.float32)

        dy_tile = w_norm[:, None] * dout_tile
        same_qk = (cq_vec[:, None] == ck_vec[None, :]) & (idx[:, None] >= idx[None, :])

        attn_qk = tl.where(same_qk, _mdot(q_tile, tl.trans(k_tile), USE_BF16_WMMA), 0.0)
        ddelta_acc += _mdot(tl.trans(attn_qk), dy_tile, USE_BF16_WMMA)

        d_attn = tl.where(same_qk, _mdot(dy_tile, tl.trans(delta_tile), USE_BF16_WMMA), 0.0)
        dq_intra = _mdot(d_attn, k_tile, USE_BF16_WMMA)
        dk_acc += _mdot(tl.trans(d_attn), q_tile, USE_BF16_WMMA)

        off_y = b * stride_yb + h * stride_yh + r * stride_yr + offs_c[:, None] * stride_yl + offs_d[None, :] * stride_yd
        tl.store(dQ_intra_ptr + off_y, dq_intra)

    tl.store(dDeltaIntra_ptr + off_u, ddelta_acc)
    tl.store(dK_intra_ptr + off_u, dk_acc)


@triton.jit
def _vault_v2_inter_scan_bwd(
    Q_ptr, K_ptr, W_ptr, Delta_ptr, dDeltaIntra_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    S_ckpt_ptr, dOut_ptr,
    dQ_dv_ptr, dK_dv_ptr, dU_ptr, dW_dv_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_wb, stride_wh, stride_wr, stride_wl,
    stride_sb, stride_sg, stride_sm, stride_sdvblk, stride_sc, stride_sflat,
    num_chunks,
    L_total,
    NUM_DV_BLOCKS: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FLAT_DVD: tl.constexpr,
    BANK_PAD: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    CKPT_INTERVAL: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
    SHARED_GROUP_CELL_Q: tl.constexpr = False,
):
    bgr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bgr % R
    bg = bgr // R
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, C1, tl.where(r32 == 2, C2, C3)))
    bank_size = tl.where(r32 == 0, P0, tl.where(r32 == 1, P1, tl.where(r32 == 2, P2, P3)))

    offs_dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV).to(tl.int64)
    offs_dk = tl.arange(0, D).to(tl.int64)
    offs_p = tl.arange(0, BANK_PAD)
    offs_flat = tl.arange(0, FLAT_DVD).to(tl.int64)
    idx_c = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    c_mask = offs_p[:, None] < bank_size

    dS_reg = tl.zeros([BANK_PAD, FLAT_DVD], dtype=tl.float32)

    k_grp_base = b * stride_kb + g * stride_kg
    u_bank_base = b * stride_ub + g * stride_ug + r * stride_ur
    ck_bank_base = b * stride_ckb + g * stride_ckg + r * stride_ckr
    s_grp_base = (
        b * stride_sb + g * stride_sg + dv_blk * stride_sdvblk
        + (base_cell + offs_p[:, None]).to(tl.int64) * stride_sc
        + offs_flat[None, :] * stride_sflat
    )
    L_tot64 = L_total.to(tl.int64)
    grp_dv_base = (bgr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

    for rev_m in range(num_chunks):
        m_i32 = num_chunks - 1 - rev_m
        m64 = m_i32.to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        if CKPT_INTERVAL == 1:
            S_snap = tl.load(S_ckpt_ptr + s_grp_base + m64 * stride_sm, mask=c_mask, other=0.0).to(tl.float32)
        else:
            seg_i32 = m_i32 // CKPT_INTERVAL
            seg64 = seg_i32.to(tl.int64)
            seg_start_i32 = seg_i32 * CKPT_INTERVAL
            S_snap = tl.load(S_ckpt_ptr + s_grp_base + seg64 * stride_sm, mask=c_mask, other=0.0).to(tl.float32)
            for rep_m in range(seg_start_i32, m_i32):
                offs_rep = rep_m.to(tl.int64) * CHUNK_SIZE + idx_c
                k_rep = tl.load(K_ptr + k_grp_base + offs_rep[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
                ck_rep = tl.load(CellK_ptr + ck_bank_base + offs_rep * stride_ckl) - base_cell
                ok_rep = (ck_rep[:, None] == offs_p[None, :]).to(tl.float32)
                off_d_rep = u_bank_base + offs_rep[:, None] * stride_ul + offs_dv[None, :] * stride_ud
                delta_rep = tl.load(Delta_ptr + off_d_rep).to(tl.float32)
                upd_rep = tl.reshape(delta_rep[:, :, None] * k_rep[:, None, :], [CHUNK_SIZE, FLAT_DVD])
                S_snap += _mdot(tl.trans(ok_rep), upd_rep, USE_BF16_WMMA)

        k_tile = tl.load(K_ptr + k_grp_base + offs_l[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)
        ck_vec = tl.load(CellK_ptr + ck_bank_base + offs_l * stride_ckl) - base_cell
        ok_f = (ck_vec[:, None] == offs_p[None, :]).to(tl.float32)

        off_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        delta_sub = tl.load(Delta_ptr + off_sub).to(tl.float32)
        ddelta_intra = tl.load(dDeltaIntra_ptr + off_sub).to(tl.float32)

        ds_ck = tl.reshape(_mdot(ok_f, dS_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        s_ck = tl.reshape(_mdot(ok_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])

        ddelta = ddelta_intra + tl.sum(ds_ck * k_tile[:, None, :], axis=2)
        tl.store(dU_ptr + off_sub, ddelta)

        dw_part = -tl.sum(s_ck * ddelta[:, :, None], axis=1)
        dk_state = tl.sum(ds_ck * delta_sub[:, :, None], axis=1)

        off_grp_out = grp_dv_base + offs_l[:, None] * D + offs_dk[None, :]
        tl.store(dW_dv_ptr + off_grp_out, dw_part)
        tl.store(dK_dv_ptr + off_grp_out, dk_state)

        if SHARED_GROUP_CELL_Q:
            h0 = g * HEADS_PER_GROUP
            cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h0 * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
            oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)
            s_cq = tl.reshape(_mdot(oq_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
            upd_q_acc = tl.zeros([CHUNK_SIZE, BLOCK_DV, D], dtype=tl.float32)
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = h0 + j
                bhr = (b * (G * HEADS_PER_GROUP) + h) * R + r
                q_dv_base = (bhr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                dout_sub = tl.load(dOut_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dv[None, :] * stride_qd).to(tl.float32)
                w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_l * stride_wl).to(tl.float32)

                dy_sub = w_norm[:, None] * dout_sub
                dq_cross = tl.sum(s_cq * dy_sub[:, :, None], axis=1)
                tl.store(dQ_dv_ptr + q_dv_base + offs_l[:, None] * D + offs_dk[None, :], dq_cross)
                upd_q_acc += dy_sub[:, :, None] * q_tile[:, None, :]

            upd_q_flat = tl.reshape(upd_q_acc, [CHUNK_SIZE, FLAT_DVD])
            dS_reg += _mdot(tl.trans(oq_f), upd_q_flat, USE_BF16_WMMA)
        else:
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = g * HEADS_PER_GROUP + j
                bhr = (b * (G * HEADS_PER_GROUP) + h) * R + r
                q_dv_base = (bhr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                dout_sub = tl.load(dOut_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dv[None, :] * stride_qd).to(tl.float32)
                cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
                w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_l * stride_wl).to(tl.float32)

                dy_sub = w_norm[:, None] * dout_sub
                oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)

                s_cq = tl.reshape(_mdot(oq_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
                dq_cross = tl.sum(s_cq * dy_sub[:, :, None], axis=1)
                tl.store(dQ_dv_ptr + q_dv_base + offs_l[:, None] * D + offs_dk[None, :], dq_cross)

                upd_q_flat = tl.reshape(dy_sub[:, :, None] * q_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
                dS_reg += _mdot(tl.trans(oq_f), upd_q_flat, USE_BF16_WMMA)

        upd_k_flat = tl.reshape(-ddelta[:, :, None] * w_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        dS_reg += _mdot(tl.trans(ok_f), upd_k_flat, USE_BF16_WMMA)


@triton.jit
def _vault_v2_wy_bwd(
    K_ptr, V_ptr, Beta_ptr, CellK_ptr,
    U_ptr, W_ptr, dU_ptr, dW_ptr,
    dK_wy_ptr, dV_wy_ptr, dBeta_wy_ptr,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_bb, stride_bg, stride_bl,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_db, stride_dg, stride_dr, stride_dl,
    G: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bg // G
    g = bg % G

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)

    k_tile = tl.load(K_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    v_tile = tl.load(V_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    beta_vec = tl.load(Beta_ptr + b * stride_bb + g * stride_bg + offs_c * stride_bl).to(tl.float32)
    ck_vec = tl.load(CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)

    off_u = b * stride_ub + g * stride_ug + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    u_tile = tl.load(U_ptr + off_u).to(tl.float32)
    w_tile = tl.load(W_ptr + off_u).to(tl.float32)
    du_tile = tl.load(dU_ptr + off_u).to(tl.float32)
    dw_tile = tl.load(dW_ptr + off_u).to(tl.float32)

    idx = tl.arange(0, CHUNK_SIZE)
    strict_lower = idx[:, None] > idx[None, :]
    same_cell = (ck_vec[:, None] == ck_vec[None, :]) & strict_lower

    kk = _mdot(k_tile, tl.trans(k_tile), USE_BF16_WMMA)
    A_neg = tl.where(same_cell, -beta_vec[:, None] * kk, 0.0)
    T_inv = _invert_unit_lower_tri_64(A_neg, idx, USE_BF16_WMMA)

    gu = _mdot(tl.trans(T_inv), du_tile, USE_BF16_WMMA)
    gw = _mdot(tl.trans(T_inv), dw_tile, USE_BF16_WMMA)

    dA_raw = _mdot(gu, tl.trans(u_tile), USE_BF16_WMMA) + _mdot(gw, tl.trans(w_tile), USE_BF16_WMMA)
    dA_neg = tl.where(same_cell, dA_raw, 0.0)
    P = -beta_vec[:, None] * dA_neg

    dv = beta_vec[:, None] * gu
    dk = beta_vec[:, None] * gw + _mdot(P + tl.trans(P), k_tile, USE_BF16_WMMA)
    dbeta = tl.sum(gu * v_tile + gw * k_tile, axis=1) - tl.sum(dA_neg * kk, axis=1)

    tl.store(dK_wy_ptr + off_u, dk)
    tl.store(dV_wy_ptr + off_u, dv)
    tl.store(dBeta_wy_ptr + b * stride_db + g * stride_dg + r * stride_dr + offs_c * stride_dl, dbeta)


# ==============================================================================
# 4. PYTORCH AUTOGRAD FUNCTION FOR TRUE GVA FLASHVAULT V2 / V2.1 / V2.2 / V2.3
# ==============================================================================

class FlashVAULTv2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        phi_q: torch.Tensor,   # (B, H, L, D)
        phi_k: torch.Tensor,   # (B, G, L, D)
        v: torch.Tensor,       # (B, G, L, D)
        beta: torch.Tensor,    # (B, G, L)
        cell_q: torch.Tensor,  # (B, H, R, L) int64
        cell_k: torch.Tensor,  # (B, G, R, L) int64
        chunk_size: int = 64,
        v2_1_consensus: bool = False,
        ckpt_interval: int = 1,
        v2_2_beta_occ: bool = False,
        prime_banks: Tuple[int, int, int, int] = (17, 19, 23, 29),
        shared_group_cell_q: bool = False,
    ) -> torch.Tensor:
        orig_dtype = v.dtype
        use_bf16_wmma = orig_dtype in (torch.bfloat16, torch.float16)

        p0, p1, p2, p3 = int(prime_banks[0]), int(prime_banks[1]), int(prime_banks[2]), int(prime_banks[3])
        c1 = p0
        c2 = p0 + p1
        c3 = p0 + p1 + p2
        total_cells = c3 + p3
        bank_pad = 32 if max(prime_banks) <= 32 else 64
        assert max(prime_banks) <= 64, f"Expected max(prime_banks) <= 64, got {prime_banks}"

        B, H, L_orig, D = phi_q.shape
        _, G, _, _ = phi_k.shape
        R = cell_k.size(2)
        assert H % G == 0, "n_q_heads must be divisible by n_vault_groups"
        assert D == 64 and R == 4
        assert ckpt_interval >= 1

        save_ckpts = any(ctx.needs_input_grad[:4])
        # Hardware-exact RDNA4 VGPR invariant (`BANK_PAD * BLOCK_DV == 128` in backward -> <=192 VGPRs, 7s compile!):
        #   - Eval (`not save_ckpts`, 1 state matrix `S_reg`): `block_dv=4, scan_warps=8` for both `BP=32` and `BP=64`
        #   - Train (`save_ckpts`, 3 state matrices in bwd): `block_dv=4` (`BP=32`) or `block_dv=2` (`BP=64`), `scan_warps=4`
        if not use_bf16_wmma:
            block_dv = 1
            scan_warps = 4
        elif not save_ckpts:
            block_dv = 4
            scan_warps = 8 if bank_pad <= 32 else 16
        else:
            block_dv = 4 if bank_pad <= 32 else 2
            scan_warps = 4

        rem = L_orig % chunk_size
        pad_l = (chunk_size - rem) if rem != 0 else 0
        if pad_l > 0:
            phi_q = F.pad(phi_q, (0, 0, 0, pad_l))
            phi_k = F.pad(phi_k, (0, 0, 0, pad_l))
            v = F.pad(v, (0, 0, 0, pad_l))
            beta = F.pad(beta, (0, pad_l), value=0.0)
            cell_q = torch.cat([cell_q, cell_q[..., -1:].expand(-1, -1, -1, pad_l)], dim=-1)
            cell_k = torch.cat([cell_k, cell_k[..., -1:].expand(-1, -1, -1, pad_l)], dim=-1)

        L = phi_q.size(2)
        num_chunks = L // chunk_size
        num_ckpt = (num_chunks + ckpt_interval - 1) // ckpt_interval
        heads_per_group = H // G
        num_dv_blocks = D // block_dv
        flat_dvd = block_dv * D

        pq = phi_q.contiguous()
        pk = phi_k.contiguous()
        vc = v.contiguous()
        bc = beta.contiguous()
        cq = cell_q.contiguous()
        ck = cell_k.contiguous()

        U = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
        W = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
        Delta = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
        YCross = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        Out = torch.empty((B, H, L, D), device=v.device, dtype=orig_dtype)
        Weights = torch.empty((B, H, R, L), device=v.device, dtype=torch.float32)
        ChunkHist = torch.empty((B, G, R, num_chunks, bank_pad), device=v.device, dtype=torch.float32)

        if save_ckpts:
            S_ckpt = torch.empty((B, G, num_ckpt, num_dv_blocks, total_cells, flat_dvd), device=v.device, dtype=orig_dtype)
        else:
            S_ckpt = torch.empty((1, 1, 1, 1, 1, 1), device=v.device, dtype=orig_dtype)

        _vault_v2_wy_fwd[(num_chunks, B * G, R)](
            pk, vc, bc, ck, U, W,
            *pk.stride(), *bc.stride(), *ck.stride(), *U.stride(),
            G=G, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )

        _vault_v2_chunk_occ_hist_fwd[(num_chunks, B * G, R)](
            ck, bc, ChunkHist,
            *ck.stride(), *bc.stride(), *ChunkHist.stride(),
            G=G, CHUNK_SIZE=chunk_size, BANK_PAD=bank_pad,
            C1=c1, C2=c2, C3=c3,
            V2_2_BETA_OCC=v2_2_beta_occ,
            num_warps=2, num_stages=1,
        )
        ChunkPrior = torch.cumsum(ChunkHist, dim=3) - ChunkHist
        del ChunkHist

        _vault_v2_occ_weights_chunk_fwd[(num_chunks, B * H)](
            cq, ck, bc, ChunkPrior, Weights,
            *cq.stride(), *ck.stride(), *bc.stride(), *ChunkPrior.stride(), *Weights.stride(),
            R,
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            BANK_PAD=bank_pad, C1=c1, C2=c2, C3=c3,
            V2_1_CONSENSUS=v2_1_consensus,
            V2_2_BETA_OCC=v2_2_beta_occ,
            num_warps=2, num_stages=1,
        )
        del ChunkPrior

        _vault_v2_inter_scan_fwd[(B * G * R, num_dv_blocks)](
            pq, pk, U, W, cq, ck,
            YCross, S_ckpt, Delta,
            *pq.stride(), *pk.stride(), *U.stride(), *cq.stride(), *ck.stride(), *YCross.stride(), *S_ckpt.stride(),
            num_chunks,
            G=G, R=R, HEADS_PER_GROUP=heads_per_group,
            CHUNK_SIZE=chunk_size, D=D,
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd, BANK_PAD=bank_pad,
            P0=p0, P1=p1, P2=p2, P3=p3, C1=c1, C2=c2, C3=c3,
            SAVE_CHECKPOINTS=save_ckpts, CKPT_INTERVAL=ckpt_interval,
            USE_BF16_WMMA=use_bf16_wmma,
            SHARED_GROUP_CELL_Q=shared_group_cell_q,
            num_warps=scan_warps, num_stages=1,
        )
        if not save_ckpts:
            del U, W

        _vault_v2_intra_readout_fused_fwd[(num_chunks, B * H)](
            pq, pk, Delta, YCross, cq, ck, Weights,
            Out,
            *pq.stride(), *pk.stride(), *Delta.stride(), *cq.stride(), *ck.stride(), *YCross.stride(), *Weights.stride(), *Out.stride(),
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        del YCross

        if save_ckpts:
            ctx.save_for_backward(pq, pk, vc, bc, cq, ck, U, W, Delta, Weights, S_ckpt)
            ctx.meta = (
                L_orig, pad_l, chunk_size, block_dv, scan_warps, use_bf16_wmma, ckpt_interval,
                bank_pad, p0, p1, p2, p3, c1, c2, c3, shared_group_cell_q,
            )

        return Out[:, :, :L_orig, :]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pq, pk, vc, bc, cq, ck, U, W, Delta, Weights, S_ckpt = ctx.saved_tensors
        (
            L_orig, pad_l, chunk_size, block_dv, scan_warps, use_bf16_wmma, ckpt_interval,
            bank_pad, p0, p1, p2, p3, c1, c2, c3, shared_group_cell_q,
        ) = ctx.meta
        orig_dtype = U.dtype

        if pad_l > 0:
            grad_out = F.pad(grad_out, (0, 0, 0, pad_l))
        dout = grad_out.contiguous()

        B, H, L, D = pq.shape
        _, G, R, _ = ck.shape
        num_chunks = L // chunk_size
        heads_per_group = H // G
        num_dv_blocks = D // block_dv
        flat_dvd = block_dv * D

        dDeltaIntra = torch.empty((B, G, R, L, D), device=pq.device, dtype=orig_dtype)
        dQ_intra = torch.empty((B, H, R, L, D), device=pq.device, dtype=orig_dtype)
        dK_intra = torch.empty((B, G, R, L, D), device=pq.device, dtype=orig_dtype)

        _vault_v2_intra_readout_bwd[(num_chunks, B * G, R)](
            pq, pk, Delta, dout, cq, ck, Weights,
            dDeltaIntra, dQ_intra, dK_intra,
            *pq.stride(), *pk.stride(), *U.stride(), *cq.stride(), *ck.stride(), *dQ_intra.stride(), *Weights.stride(),
            G=G, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )

        dU = torch.empty((B, G, R, L, D), device=pq.device, dtype=orig_dtype)
        dQ_dv = torch.empty((B * H * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)
        dK_dv = torch.empty((B * G * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)
        dW_dv = torch.empty((B * G * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)

        _vault_v2_inter_scan_bwd[(B * G * R, num_dv_blocks)](
            pq, pk, W, Delta, dDeltaIntra, cq, ck, Weights,
            S_ckpt, dout,
            dQ_dv, dK_dv, dU, dW_dv,
            *pq.stride(), *pk.stride(), *U.stride(), *cq.stride(), *ck.stride(), *Weights.stride(), *S_ckpt.stride(),
            num_chunks, L,
            NUM_DV_BLOCKS=num_dv_blocks,
            G=G, R=R, HEADS_PER_GROUP=heads_per_group,
            CHUNK_SIZE=chunk_size, D=D,
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd, BANK_PAD=bank_pad,
            P0=p0, P1=p1, P2=p2, P3=p3, C1=c1, C2=c2, C3=c3,
            CKPT_INTERVAL=ckpt_interval,
            USE_BF16_WMMA=use_bf16_wmma,
            SHARED_GROUP_CELL_Q=shared_group_cell_q,
            num_warps=scan_warps, num_stages=1,
        )
        del dDeltaIntra

        dQ = (dQ_intra.float() + dQ_dv.sum(dim=1, dtype=torch.float32).view(B, H, R, L, D)).sum(dim=2).to(pq.dtype)
        del dQ_intra, dQ_dv

        dK_banks = dK_intra.float() + dK_dv.sum(dim=1, dtype=torch.float32).view(B, G, R, L, D)
        del dK_intra, dK_dv

        dW = dW_dv.sum(dim=1, dtype=torch.float32).view(B, G, R, L, D).to(orig_dtype)
        del dW_dv

        dK_wy = torch.empty((B, G, R, L, D), device=pq.device, dtype=orig_dtype)
        dV_wy = torch.empty((B, G, R, L, D), device=pq.device, dtype=orig_dtype)
        dBeta_wy = torch.empty((B, G, R, L), device=pq.device, dtype=torch.float32)

        _vault_v2_wy_bwd[(num_chunks, B * G, R)](
            pk, vc, bc, ck, U, W, dU, dW,
            dK_wy, dV_wy, dBeta_wy,
            *pk.stride(), *bc.stride(), *ck.stride(), *U.stride(), *dBeta_wy.stride(),
            G=G, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        del dU, dW

        dK = (dK_banks + dK_wy.float()).sum(dim=2).to(pk.dtype)
        del dK_banks, dK_wy
        dV = dV_wy.float().sum(dim=2).to(vc.dtype)
        del dV_wy
        dBeta = dBeta_wy.sum(dim=2).to(bc.dtype)
        del dBeta_wy

        if pad_l > 0:
            dQ = dQ[:, :, :L_orig, :]
            dK = dK[:, :, :L_orig, :]
            dV = dV[:, :, :L_orig, :]
            dBeta = dBeta[:, :, :L_orig]

        return dQ, dK, dV, dBeta, None, None, None, None, None, None, None, None


def flash_vault_v2_chunkwise(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cell_q: torch.Tensor,
    cell_k: torch.Tensor,
    chunk_size: int = 64,
    v2_1_consensus: bool = False,
    ckpt_interval: int = 1,
    v2_2_beta_occ: bool = False,
    prime_banks: Tuple[int, int, int, int] = (17, 19, 23, 29),
    shared_group_cell_q: bool = False,
) -> torch.Tensor:
    if not (_HAS_TRITON and phi_q.is_cuda):
        return flash_vault_v2_recurrent_reference(
            phi_q,
            phi_k,
            v,
            beta,
            cell_q,
            cell_k,
            num_cells=sum(prime_banks),
            v2_1_consensus=v2_1_consensus,
            v2_2_beta_occ=v2_2_beta_occ,
        )
    return FlashVAULTv2Function.apply(
        phi_q,
        phi_k,
        v,
        beta,
        cell_q,
        cell_k,
        chunk_size,
        v2_1_consensus,
        ckpt_interval,
        v2_2_beta_occ,
        prime_banks,
        shared_group_cell_q,
    )


# ==============================================================================
# 5. FUSED CONTIGUOUS (B, L, C) CAUSAL DEPTHWISE 1D CONV (+ OPTIONAL SILU)
# ==============================================================================

@triton.jit
def _causal_dwconv1d_fwd_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    L, C,
    stride_xb, stride_xl, stride_xc,
    stride_wc, stride_wk,
    stride_yb, stride_yl, stride_yc,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int64)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    mask_c = offs_c < C
    mask_l = offs_l < L

    acc = tl.zeros([BLOCK_L, BLOCK_C], dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        acc += bias[None, :]

    x_base = X_ptr + b * stride_xb + offs_c[None, :] * stride_xc
    for k in tl.static_range(0, K):
        w_idx = (K - 1) - k
        w_k = tl.load(W_ptr + offs_c * stride_wc + w_idx * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        t_src = offs_l - k
        valid = (t_src >= 0) & mask_l
        x_k = tl.load(x_base + t_src[:, None] * stride_xl, mask=valid[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        acc += x_k * w_k[None, :]

    if APPLY_SILU:
        acc = acc * tl.sigmoid(acc)

    y_ptrs = Y_ptr + b * stride_yb + offs_l[:, None] * stride_yl + offs_c[None, :] * stride_yc
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_l[:, None] & mask_c[None, :])


@triton.jit
def _causal_dwconv1d_bwd_kernel(
    X_ptr, W_ptr, B_ptr, dY_ptr,
    dX_ptr, dW_part_ptr, dB_part_ptr,
    L, C,
    stride_xb, stride_xl, stride_xc,
    stride_wc, stride_wk,
    stride_dyb, stride_dyl, stride_dyc,
    stride_dxb, stride_dxl, stride_dxc,
    stride_dwp, stride_dwc, stride_dwk,
    stride_dbp, stride_dbc,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    num_l_blocks = tl.num_programs(0).to(tl.int64)
    part_idx = b * num_l_blocks + pid_l

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int64)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    mask_c = offs_c < C
    mask_l = offs_l < L

    x_base = X_ptr + b * stride_xb + offs_c[None, :] * stride_xc
    dy_base = dY_ptr + b * stride_dyb + offs_c[None, :] * stride_dyc
    bias = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_C], dtype=tl.float32)

    dy_cur = tl.load(dy_base + offs_l[:, None] * stride_dyl, mask=mask_l[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
    if APPLY_SILU:
        z_cur = bias[None, :] + tl.zeros([BLOCK_L, BLOCK_C], dtype=tl.float32)
        for k in tl.static_range(0, K):
            w_idx = (K - 1) - k
            w_k = tl.load(W_ptr + offs_c * stride_wc + w_idx * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
            t_src = offs_l - k
            v_src = (t_src >= 0) & mask_l
            x_k = tl.load(x_base + t_src[:, None] * stride_xl, mask=v_src[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
            z_cur += x_k * w_k[None, :]
        sig = tl.sigmoid(z_cur)
        dz_cur = tl.where(mask_l[:, None], dy_cur * sig * (1.0 + z_cur * (1.0 - sig)), 0.0)
    else:
        dz_cur = tl.where(mask_l[:, None], dy_cur, 0.0)

    if HAS_BIAS:
        db_sum = tl.sum(dz_cur, axis=0)
        tl.store(dB_part_ptr + part_idx * stride_dbp + offs_c * stride_dbc, db_sum, mask=mask_c)

    for k in tl.static_range(0, K):
        w_idx = (K - 1) - k
        t_src = offs_l - k
        v_src = (t_src >= 0) & mask_l
        x_k = tl.load(x_base + t_src[:, None] * stride_xl, mask=v_src[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        dw_k = tl.sum(dz_cur * x_k, axis=0)
        tl.store(dW_part_ptr + part_idx * stride_dwp + offs_c * stride_dwc + w_idx * stride_dwk, dw_k, mask=mask_c)

    dx_acc = tl.zeros([BLOCK_L, BLOCK_C], dtype=tl.float32)
    w_k0 = tl.load(W_ptr + offs_c * stride_wc + (K - 1) * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
    dx_acc += dz_cur * w_k0[None, :]

    for k in tl.static_range(1, K):
        w_idx = (K - 1) - k
        w_k = tl.load(W_ptr + offs_c * stride_wc + w_idx * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        t_dst = offs_l + k
        v_dst = t_dst < L
        dy_f = tl.load(dy_base + t_dst[:, None] * stride_dyl, mask=v_dst[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        if APPLY_SILU:
            z_f = bias[None, :] + tl.zeros([BLOCK_L, BLOCK_C], dtype=tl.float32)
            for m in tl.static_range(0, K):
                wm_idx = (K - 1) - m
                wm = tl.load(W_ptr + offs_c * stride_wc + wm_idx * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
                ts = t_dst - m
                vs = (ts >= 0) & v_dst
                xm = tl.load(x_base + ts[:, None] * stride_xl, mask=vs[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
                z_f += xm * wm[None, :]
            sig_f = tl.sigmoid(z_f)
            dz_f = tl.where(v_dst[:, None], dy_f * sig_f * (1.0 + z_f * (1.0 - sig_f)), 0.0)
        else:
            dz_f = tl.where(v_dst[:, None], dy_f, 0.0)
        dx_acc += dz_f * w_k[None, :]

    dx_ptrs = dX_ptr + b * stride_dxb + offs_l[:, None] * stride_dxl + offs_c[None, :] * stride_dxc
    tl.store(dx_ptrs, dx_acc.to(dX_ptr.dtype.element_ty), mask=mask_l[:, None] & mask_c[None, :])


class CausalDepthwiseConv1DFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        apply_silu: bool,
    ) -> torch.Tensor:
        B, L, C = x.shape
        K = weight.size(-1)
        xc = x.contiguous()
        wc = weight.squeeze(1).contiguous()
        has_bias = bias is not None
        bc = bias.contiguous() if has_bias else xc.new_empty((1,))

        y = torch.empty_like(xc)
        BLOCK_L = 128
        BLOCK_C = 64
        grid = (triton.cdiv(L, BLOCK_L), triton.cdiv(C, BLOCK_C), B)
        _causal_dwconv1d_fwd_kernel[grid](
            xc, wc, bc, y,
            L, C,
            *xc.stride(), *wc.stride(), *y.stride(),
            K=K, HAS_BIAS=has_bias, APPLY_SILU=apply_silu,
            BLOCK_L=BLOCK_L, BLOCK_C=BLOCK_C,
            num_warps=4, num_stages=1,
        )
        if any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(xc, weight, bias if has_bias else torch.empty(0, device=x.device))
            ctx.meta = (has_bias, apply_silu, K)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xc, weight, bias_saved = ctx.saved_tensors
        has_bias, apply_silu, K = ctx.meta
        B, L, C = xc.shape
        wc = weight.squeeze(1).contiguous()
        bc = bias_saved.contiguous() if has_bias else xc.new_empty((1,))
        dyc = grad_out.contiguous()

        BLOCK_L = 128
        BLOCK_C = 64
        num_l = triton.cdiv(L, BLOCK_L)
        num_parts = B * num_l

        dx = torch.empty_like(xc)
        dw_part = torch.empty((num_parts, C, K), device=xc.device, dtype=torch.float32)
        db_part = torch.empty((num_parts, C), device=xc.device, dtype=torch.float32) if has_bias else xc.new_empty((1, 1), dtype=torch.float32)

        grid = (num_l, triton.cdiv(C, BLOCK_C), B)
        _causal_dwconv1d_bwd_kernel[grid](
            xc, wc, bc, dyc,
            dx, dw_part, db_part,
            L, C,
            *xc.stride(), *wc.stride(), *dyc.stride(), *dx.stride(), *dw_part.stride(), *db_part.stride(),
            K=K, HAS_BIAS=has_bias, APPLY_SILU=apply_silu,
            BLOCK_L=BLOCK_L, BLOCK_C=BLOCK_C,
            num_warps=4, num_stages=1,
        )
        dw = dw_part.sum(dim=0).unsqueeze(1).to(weight.dtype)
        db = db_part.sum(dim=0).to(bias_saved.dtype) if has_bias else None
        return dx, dw, db, None


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    apply_silu: bool = False,
) -> torch.Tensor:
    if not (_HAS_TRITON and x.is_cuda):
        L = x.size(1)
        K = weight.size(-1)
        y = F.conv1d(x.transpose(1, 2), weight, bias, padding=K - 1, groups=x.size(-1))[..., :L].transpose(1, 2)
        return F.silu(y) if apply_silu else y
    return CausalDepthwiseConv1DFunction.apply(x, weight, bias, apply_silu)
