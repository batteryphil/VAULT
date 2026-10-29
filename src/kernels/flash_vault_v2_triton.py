# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# FlashVAULT v2: True Grouped-Vault Attention (GVA) Chunkwise Compact WY Kernel
# Built for Triton 3.8 + ROCm 7.2 (AMD RDNA4 gfx1201 native bf16 WMMA).
#
# Key v2 Upgrades over FlashVAULT v1:
#   1. True Grouped-Vault State (`G` shared Co-Prime Fast-Weight Vaults for `H` independent Query Heads,
#      `H = G * HEADS_PER_GROUP`):
#        - Each of the `H` query heads has its OWN `phi_q[h]` AND its OWN routed `cell_q[h]` (can probe
#          3 distinct Co-Prime cells simultaneously per vault group!).
#        - `phi_k`, `v`, token-selective `beta_t`, `cell_k`, WY factors `(U, W)`, and fast-weight state
#          `S_ckpt` live at group resolution `G` -> 3x less HBM bandwidth, 3x smaller state checkpoints,
#          and 3x higher WMMA reuse per loaded state snapshot!
#   2. Token-Selective Write/Erase Gate `beta_{b,g,t} in (0, 1)` with exact analytical `dBeta` adjoint.
#   3. Zero Global Atomics in both Forward and Backward (disjoint coalesced per-dv-block reductions).
#   4. Unconditionally Bounded 6-Step Block-Recursive Unit Lower-Triangular Inverse (`_invert_unit_lower_tri_64`).

from typing import Optional
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ==============================================================================
# 1. EXACT PYTHORCH REFERENCE FOR TRUE GVA PARITY VERIFICATION
# ==============================================================================

def flash_vault_v2_recurrent_reference(
    phi_q: torch.Tensor,   # (B, H, L, D)
    phi_k: torch.Tensor,   # (B, G, L, D)
    v: torch.Tensor,       # (B, G, L, D)
    beta: torch.Tensor,    # (B, G, L)
    cell_q: torch.Tensor,  # (B, H, R, L) int64 in [0, 88)
    cell_k: torch.Tensor,  # (B, G, R, L) int64 in [0, 88)
    num_cells: int = 88,
) -> torch.Tensor:
    """
    Exact step-by-step recurrent ground truth for True GVA FlashVAULT v2.
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

    S = torch.zeros(B, G, num_cells, D, D, device=v.device, dtype=torch.float32)
    occ = torch.zeros(B, G, num_cells, device=v.device, dtype=torch.float32)
    outputs = []

    b_idx_g = torch.arange(B, device=v.device)[:, None]
    g_idx_g = torch.arange(G, device=v.device)[None, :]

    for t in range(L):
        kt = pk[:, :, t, :]    # (B, G, D)
        vt = vf[:, :, t, :]    # (B, G, D)
        bt = bf[:, :, t]       # (B, G)

        # 1. Each vault group updates its R Co-Prime cells once
        for r in range(R):
            ck = cell_k[:, :, r, t]  # (B, G)
            S_k = S[b_idx_g, g_idx_g, ck]  # (B, G, D, D)
            pred_v = torch.einsum("bgvd,bgd->bgv", S_k, kt)
            delta = bt.unsqueeze(-1) * (vt - pred_v)
            S[b_idx_g, g_idx_g, ck] = S_k + torch.einsum("bgv,bgd->bgvd", delta, kt)
            occ[b_idx_g, g_idx_g, ck] = occ[b_idx_g, g_idx_g, ck] + 1.0

        # 2. All H query heads read concurrently from their group's updated vault
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
                sigma_sq = torch.clamp(n_q - 1.0, min=0.0) / float(D)
                w_r = 1.0 / (sigma_sq + 1e-2)
                y_banks.append(y_r)
                w_banks.append(w_r)

            w_stack = torch.stack(w_banks, dim=-1)  # (B, R)
            w_norm = w_stack / w_stack.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            y_stack = torch.stack(y_banks, dim=-2)  # (B, R, D)
            y_h = (y_stack * w_norm.unsqueeze(-1)).sum(dim=-2)  # (B, D)
            y_heads.append(y_h)

        outputs.append(torch.stack(y_heads, dim=1))  # (B, H, D)

    return torch.stack(outputs, dim=2).to(orig_dtype)


# ==============================================================================
# 2. TRITON 3.8 HELPERS & FORWARD KERNELS
# ==============================================================================

@triton.jit
def _mdot(a, b, USE_BF16_WMMA: tl.constexpr):
    if USE_BF16_WMMA:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _invert_unit_lower_tri_64(A_neg, idx, USE_BF16_WMMA: tl.constexpr):
    """
    Exact 6-step parallel block-recursive solver for `(I - A_neg)^{-1}` (`64 x 64` strictly lower-triangular).
    Cancels alternating paths inside each sub-block so all entries stay bounded in [-1, 1] in BF16.
    """
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
    """Grid: (num_chunks, B * G, R) — 3x smaller than per-head WY!"""
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
def _vault_v2_occ_weights_fwd(
    CellQ_ptr, CellK_ptr, Weight_ptr,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_wb, stride_wh, stride_wr, stride_wl,
    num_chunks, num_banks,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    """Grid: (B * H,) — exact per-query-head variance-weighted consensus over group occupancy."""
    bh = tl.program_id(0).to(tl.int64)
    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_88 = tl.arange(0, 128)
    occ = tl.zeros([128], dtype=tl.float32)

    cq_base = CellQ_ptr + b * stride_cqb + h * stride_cqh
    ck_base = CellK_ptr + b * stride_ckb + g * stride_ckg
    w_base = Weight_ptr + b * stride_wb + h * stride_wh

    idx = tl.arange(0, CHUNK_SIZE)
    causal_mask = idx[:, None] >= idx[None, :]

    for m in range(num_chunks):
        offs_l = m.to(tl.int64) * CHUNK_SIZE + idx.to(tl.int64)
        w_sum = tl.zeros([CHUNK_SIZE], dtype=tl.float32)

        for r in range(num_banks):
            r64 = r.to(tl.int64)
            cq = tl.load(cq_base + r64 * stride_cqr + offs_l * stride_cql)
            ck = tl.load(ck_base + r64 * stride_ckr + offs_l * stride_ckl)

            occ_prior = tl.sum(tl.where(cq[:, None] == offs_88[None, :], occ[None, :], 0.0), axis=1)
            same_qk = (cq[:, None] == ck[None, :]) & causal_mask
            occ_intra = tl.sum(same_qk.to(tl.float32), axis=1)
            n_tot = occ_prior + occ_intra
            sigma_sq = tl.maximum(n_tot - 1.0, 0.0) / float(D)
            w_r = 1.0 / (sigma_sq + 1e-2)
            tl.store(w_base + r64 * stride_wr + offs_l * stride_wl, w_r)
            w_sum += w_r

            chunk_counts = tl.sum((ck[:, None] == offs_88[None, :]).to(tl.float32), axis=0)
            occ += chunk_counts

        inv_w_sum = 1.0 / tl.maximum(w_sum, 1e-6)
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
    SAVE_CHECKPOINTS: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    """
    Grid: (B * G * R, D // BLOCK_DV) — 3x smaller grid & 3x smaller S_ckpt!
    All `HEADS_PER_GROUP` query heads read from the single register-resident vault `S_reg`!
    """
    bgr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bgr % R
    bg = bgr // R
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, 17, tl.where(r32 == 2, 36, 59)))
    bank_size = tl.where(r32 == 0, 17, tl.where(r32 == 1, 19, tl.where(r32 == 2, 23, 29)))

    offs_dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV).to(tl.int64)
    offs_dk = tl.arange(0, D).to(tl.int64)
    offs_32 = tl.arange(0, 32)
    offs_flat = tl.arange(0, FLAT_DVD).to(tl.int64)
    idx_c = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    c_mask = offs_32[:, None] < bank_size

    S_reg = tl.zeros([32, FLAT_DVD], dtype=tl.float32)

    k_grp_base = b * stride_kb + g * stride_kg
    u_bank_base = b * stride_ub + g * stride_ug + r * stride_ur
    ck_bank_base = b * stride_ckb + g * stride_ckg + r * stride_ckr
    s_grp_base = (
        b * stride_sb + g * stride_sg + dv_blk * stride_sdvblk
        + (base_cell + offs_32[:, None]).to(tl.int64) * stride_sc
        + offs_flat[None, :] * stride_sflat
    )

    for m in range(num_chunks):
        m64 = m.to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        if SAVE_CHECKPOINTS:
            tl.store(S_ckpt_ptr + s_grp_base + m64 * stride_sm, S_reg.to(S_ckpt_ptr.dtype.element_ty), mask=c_mask)

        # 1. Compute group Delta once
        k_tile = tl.load(K_ptr + k_grp_base + offs_l[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)
        ck_vec = tl.load(CellK_ptr + ck_bank_base + offs_l * stride_ckl) - base_cell
        ok_f = (ck_vec[:, None] == offs_32[None, :]).to(tl.float32)

        off_u_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        u_sub = tl.load(U_ptr + off_u_sub).to(tl.float32)

        s_ck = tl.reshape(_mdot(ok_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        delta_sub = u_sub - tl.sum(s_ck * w_tile[:, None, :], axis=2)
        tl.store(Delta_ptr + off_u_sub, delta_sub)

        # 2. All query heads in group read from register-resident S_reg!
        for j in tl.static_range(0, HEADS_PER_GROUP):
            h = g * HEADS_PER_GROUP + j
            q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
            cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
            oq_f = (cq_vec[:, None] == offs_32[None, :]).to(tl.float32)
            s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
            y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
            off_y_sub = b * stride_yb + h * stride_yh + r * stride_yr + offs_l[:, None] * stride_yl + offs_dv[None, :] * stride_yd
            tl.store(YCross_ptr + off_y_sub, y_cross_sub)

        # 3. Commit group state update once
        upd_flat = tl.reshape(delta_sub[:, :, None] * k_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        S_reg += _mdot(tl.trans(ok_f), upd_flat, USE_BF16_WMMA)


@triton.jit
def _vault_v2_intra_readout_fwd(
    Q_ptr, K_ptr, Delta_ptr, YCross_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    OutBanks_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_yb, stride_yh, stride_yr, stride_yl, stride_yd,
    stride_wb, stride_wh, stride_wr, stride_wl,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    """Grid: (num_chunks, B * H, R) — 100% chunk-parallel intra-chunk readout."""
    chunk_idx = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    idx = tl.arange(0, CHUNK_SIZE)

    q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd).to(tl.float32)
    k_tile = tl.load(K_ptr + b * stride_kb + g * stride_kg + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    delta_tile = tl.load(Delta_ptr + b * stride_ub + g * stride_ug + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud).to(tl.float32)

    off_y = b * stride_yb + h * stride_yh + r * stride_yr + offs_c[:, None] * stride_yl + offs_d[None, :] * stride_yd
    y_cross_tile = tl.load(YCross_ptr + off_y).to(tl.float32)

    cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_c * stride_cql)
    ck_vec = tl.load(CellK_ptr + b * stride_ckb + g * stride_ckg + r * stride_ckr + offs_c * stride_ckl)
    w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_c * stride_wl).to(tl.float32)

    same_qk = (cq_vec[:, None] == ck_vec[None, :]) & (idx[:, None] >= idx[None, :])
    attn_qk = tl.where(same_qk, _mdot(q_tile, tl.trans(k_tile), USE_BF16_WMMA), 0.0)
    y_intra_tile = _mdot(attn_qk, delta_tile, USE_BF16_WMMA)

    tl.store(OutBanks_ptr + off_y, w_norm[:, None] * (y_cross_tile + y_intra_tile))


# ==============================================================================
# 3. TRITON 3.8 BACKWARD KERNELS (100% ATOMIC-FREE TRUE GVA ADJOINTS)
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
    """
    Grid: (num_chunks, B * G, R) — accumulates dDeltaIntra & dK_intra across all
    `HEADS_PER_GROUP` query heads in registers with ZERO atomics!
    """
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
    USE_BF16_WMMA: tl.constexpr,
):
    """
    Grid: (B * G * R, D // BLOCK_DV) — 3x fewer state scans & ZERO global atomics!
    """
    bgr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bgr % R
    bg = bgr // R
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, 17, tl.where(r32 == 2, 36, 59)))
    bank_size = tl.where(r32 == 0, 17, tl.where(r32 == 1, 19, tl.where(r32 == 2, 23, 29)))

    offs_dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV).to(tl.int64)
    offs_dk = tl.arange(0, D).to(tl.int64)
    offs_32 = tl.arange(0, 32)
    offs_flat = tl.arange(0, FLAT_DVD).to(tl.int64)
    idx_c = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    c_mask = offs_32[:, None] < bank_size

    dS_reg = tl.zeros([32, FLAT_DVD], dtype=tl.float32)

    k_grp_base = b * stride_kb + g * stride_kg
    u_bank_base = b * stride_ub + g * stride_ug + r * stride_ur
    ck_bank_base = b * stride_ckb + g * stride_ckg + r * stride_ckr
    s_grp_base = (
        b * stride_sb + g * stride_sg + dv_blk * stride_sdvblk
        + (base_cell + offs_32[:, None]).to(tl.int64) * stride_sc
        + offs_flat[None, :] * stride_sflat
    )
    L_tot64 = L_total.to(tl.int64)
    grp_dv_base = (bgr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

    for rev_m in range(num_chunks):
        m64 = (num_chunks - 1 - rev_m).to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        S_snap = tl.load(S_ckpt_ptr + s_grp_base + m64 * stride_sm, mask=c_mask, other=0.0).to(tl.float32)

        k_tile = tl.load(K_ptr + k_grp_base + offs_l[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)
        ck_vec = tl.load(CellK_ptr + ck_bank_base + offs_l * stride_ckl) - base_cell
        ok_f = (ck_vec[:, None] == offs_32[None, :]).to(tl.float32)

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

        # Accumulate query-head state adjoints across HEADS_PER_GROUP
        ds_from_q = tl.zeros([32, FLAT_DVD], dtype=tl.float32)
        for j in tl.static_range(0, HEADS_PER_GROUP):
            h = g * HEADS_PER_GROUP + j
            bhr = (b * (G * HEADS_PER_GROUP) + h) * R + r
            q_dv_base = (bhr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

            q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
            dout_sub = tl.load(dOut_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dv[None, :] * stride_qd).to(tl.float32)
            cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_l * stride_cql) - base_cell
            w_norm = tl.load(Weight_ptr + b * stride_wb + h * stride_wh + r * stride_wr + offs_l * stride_wl).to(tl.float32)

            dy_sub = w_norm[:, None] * dout_sub
            oq_f = (cq_vec[:, None] == offs_32[None, :]).to(tl.float32)

            s_cq = tl.reshape(_mdot(oq_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
            dq_cross = tl.sum(s_cq * dy_sub[:, :, None], axis=1)
            tl.store(dQ_dv_ptr + q_dv_base + offs_l[:, None] * D + offs_dk[None, :], dq_cross)

            upd_q_flat = tl.reshape(dy_sub[:, :, None] * q_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
            ds_from_q += _mdot(tl.trans(oq_f), upd_q_flat, USE_BF16_WMMA)

        upd_k_flat = tl.reshape(-ddelta[:, :, None] * w_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        dS_reg += ds_from_q + _mdot(tl.trans(ok_f), upd_k_flat, USE_BF16_WMMA)


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
    """Grid: (num_chunks, B * G, R) — 3x smaller WY backward!"""
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
# 4. PYTORCH AUTOGRAD FUNCTION FOR TRUE GVA FLASHVAULT V2
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
    ) -> torch.Tensor:
        orig_dtype = v.dtype
        use_bf16_wmma = orig_dtype in (torch.bfloat16, torch.float16)
        block_dv = 4 if use_bf16_wmma else 1

        B, H, L_orig, D = phi_q.shape
        _, G, _, _ = phi_k.shape
        R = cell_k.size(2)
        assert H % G == 0, "n_q_heads must be divisible by n_vault_groups"
        assert D == 64 and R == 4

        rem = L_orig % chunk_size
        pad_l = (chunk_size - rem) if rem != 0 else 0
        if pad_l > 0:
            phi_q = F.pad(phi_q, (0, 0, 0, pad_l))
            phi_k = F.pad(phi_k, (0, 0, 0, pad_l))
            v = F.pad(v, (0, 0, 0, pad_l))
            beta = F.pad(beta, (0, pad_l), value=0.0)
            cell_q = F.pad(cell_q, (0, pad_l), value=0)
            cell_k = F.pad(cell_k, (0, pad_l), value=0)

        L = phi_q.size(2)
        num_chunks = L // chunk_size
        heads_per_group = H // G
        num_dv_blocks = D // block_dv
        flat_dvd = block_dv * D
        save_ckpts = any(ctx.needs_input_grad[:4])

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
        OutBanks = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        Weights = torch.empty((B, H, R, L), device=v.device, dtype=torch.float32)

        if save_ckpts:
            S_ckpt = torch.empty((B, G, num_chunks, num_dv_blocks, 88, flat_dvd), device=v.device, dtype=orig_dtype)
        else:
            S_ckpt = torch.empty((1, 1, 1, 1, 1, 1), device=v.device, dtype=orig_dtype)

        _vault_v2_wy_fwd[(num_chunks, B * G, R)](
            pk, vc, bc, ck, U, W,
            *pk.stride(), *bc.stride(), *ck.stride(), *U.stride(),
            G=G, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )

        _vault_v2_occ_weights_fwd[(B * H,)](
            cq, ck, Weights,
            *cq.stride(), *ck.stride(), *Weights.stride(),
            num_chunks, R,
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            num_warps=4, num_stages=1,
        )

        _vault_v2_inter_scan_fwd[(B * G * R, num_dv_blocks)](
            pq, pk, U, W, cq, ck,
            YCross, S_ckpt, Delta,
            *pq.stride(), *pk.stride(), *U.stride(), *cq.stride(), *ck.stride(), *YCross.stride(), *S_ckpt.stride(),
            num_chunks,
            G=G, R=R, HEADS_PER_GROUP=heads_per_group,
            CHUNK_SIZE=chunk_size, D=D,
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd,
            SAVE_CHECKPOINTS=save_ckpts, USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        if not save_ckpts:
            del U, W

        _vault_v2_intra_readout_fwd[(num_chunks, B * H, R)](
            pq, pk, Delta, YCross, cq, ck, Weights,
            OutBanks,
            *pq.stride(), *pk.stride(), *Delta.stride(), *cq.stride(), *ck.stride(), *OutBanks.stride(), *Weights.stride(),
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        del YCross

        Out = OutBanks.float().sum(dim=2).to(orig_dtype)
        del OutBanks

        if save_ckpts:
            ctx.save_for_backward(pq, pk, vc, bc, cq, ck, U, W, Delta, Weights, S_ckpt)
            ctx.meta = (L_orig, pad_l, chunk_size, block_dv, use_bf16_wmma)

        return Out[:, :, :L_orig, :]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pq, pk, vc, bc, cq, ck, U, W, Delta, Weights, S_ckpt = ctx.saved_tensors
        L_orig, pad_l, chunk_size, block_dv, use_bf16_wmma = ctx.meta
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
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        del dDeltaIntra

        dQ = (dQ_intra.float() + dQ_dv.float().sum(dim=1).view(B, H, R, L, D)).sum(dim=2).to(pq.dtype)
        del dQ_intra, dQ_dv

        dK_banks = dK_intra.float() + dK_dv.float().sum(dim=1).view(B, G, R, L, D)
        del dK_intra, dK_dv

        dW = dW_dv.float().sum(dim=1).view(B, G, R, L, D).to(orig_dtype)
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

        return dQ, dK, dV, dBeta, None, None, None


def flash_vault_v2_chunkwise(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cell_q: torch.Tensor,
    cell_k: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    return FlashVAULTv2Function.apply(phi_q, phi_k, v, beta, cell_q, cell_k, chunk_size)
