# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# FlashVAULT: Split Chunk-Parallel WMMA + Register-Resident State Scan (Triton 3.8 / ROCm 7.2)
#
# Architecture:
#   1. Chunk-Parallel WY Solver (`_vault_wy_intra_chunk_fwd/bwd`): 100% parallel across all `L/64` chunks
#      on RDNA4 `bf16` WMMA Matrix Cores (`6` nilpotent doubling steps).
#   2. Lean Register-Resident State Scan (`_vault_inter_chunk_scan_fwd/bwd`): Holds `[32, BLOCK_DV * D]`
#      fast-weight bank state in VGPR registers (`BLOCK_DV=4` in BF16, `1` in FP32) using `O_K @ S_reg`
#      WMMA gathers/scatters and 64-bit `tl.int64` addressing (safe past `128K+` tokens!).
#   3. Chunk-Parallel Intra-Readout (`_vault_intra_readout_fwd/bwd`): Executes all `64x64` intra-chunk
#      attention & gradient GEMMs across all `D=64` channels simultaneously in parallel over `num_chunks`!

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


PRIME_BANKS_DEFAULT = (17, 19, 23, 29)  # 88 Co-Prime Vault Cells


# ==============================================================================
# 1. EXACT TOKEN-BY-TOKEN RECURRENT REFERENCE (GROUND TRUTH FOR PARITY & GRADCHECK)
# ==============================================================================

def flash_vault_recurrent_ref(
    phi_q: torch.Tensor,      # (B, H, L, D)
    phi_k: torch.Tensor,      # (B, H, L, D)
    v: torch.Tensor,          # (B, H, L, D)
    beta: torch.Tensor,       # (B, H, L) in [0, 1]
    cell_q: torch.Tensor,     # (B, G, R, L) int64 global cell indices in [0, 88)
    cell_k: torch.Tensor,     # (B, G, R, L) int64 global cell indices in [0, 88)
    total_cells: int = 88,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, L, D = v.shape
    _, G, R, _ = cell_k.shape
    heads_per_group = H // G
    orig_dtype = v.dtype

    pq = phi_q.float()
    pk = phi_k.float()
    vf = v.float()
    bf = beta.float()

    S = torch.zeros(B, H, total_cells, D, D, device=v.device, dtype=torch.float32)
    occ = torch.zeros(B, H, total_cells, device=v.device, dtype=torch.float32)
    outputs = []

    for t in range(L):
        pk_t = pk[:, :, t, :].unsqueeze(2)
        pq_t = pq[:, :, t, :].unsqueeze(2)
        v_t = vf[:, :, t, :].unsqueeze(2)
        b_t = bf[:, :, t].view(B, H, 1, 1)

        ck_t = cell_k[:, :, :, t].repeat_interleave(heads_per_group, dim=1)
        cq_t = cell_q[:, :, :, t].repeat_interleave(heads_per_group, dim=1)

        idx_mat_k = ck_t.view(B, H, R, 1, 1).expand(B, H, R, D, D)
        S_ck = S.gather(2, idx_mat_k)

        pred_v = torch.matmul(S_ck, pk_t.unsqueeze(-1)).squeeze(-1)
        delta_t = b_t * (v_t - pred_v)
        upd = torch.matmul(delta_t.unsqueeze(-1), pk_t.unsqueeze(-2))
        S = S.scatter_add(2, idx_mat_k, upd)

        occ = occ.scatter_add(2, ck_t, torch.ones_like(ck_t, dtype=torch.float32))

        idx_mat_q = cq_t.view(B, H, R, 1, 1).expand(B, H, R, D, D)
        S_cq = S.gather(2, idx_mat_q)
        y_banks = torch.matmul(S_cq, pq_t.unsqueeze(-1)).squeeze(-1)

        n_cq = occ.gather(2, cq_t)
        sigma_sq = F.relu(n_cq - 1.0) / float(D)
        w_banks = (1.0 / (sigma_sq + 1e-2)).unsqueeze(-1)

        w_sum = w_banks.sum(dim=2, keepdim=True).clamp(min=1e-6)
        y_t = (w_banks * y_banks).sum(dim=2) / w_sum.squeeze(2)
        outputs.append(y_t)

    out = torch.stack(outputs, dim=2).to(orig_dtype)
    return out, S


# ==============================================================================
# 2. TRITON 3.8 FORWARD KERNELS (64-BIT SAFE + SPLIT CHUNK-PARALLEL WMMA)
# ==============================================================================

@triton.jit
def _mdot(a, b, USE_BF16_WMMA: tl.constexpr):
    if USE_BF16_WMMA:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, allow_tf32=False)


@triton.jit
def _invert_unit_lower_tri_64(A, idx, USE_BF16_WMMA: tl.constexpr):
    """
    Exact O(log C) Block-Recursive Unit Lower-Triangular Inverse `(I + A)^(-1)` for 64x64 `A` (`A` strictly lower-tri).
    Every intermediate sub-block inverse `T_00^(-1)` already cancels alternating paths and stays strictly in [-1, +1],
    eliminating BF16/FP32 C(63, 31) Neumann power overflow when 20-64 identical/padding tokens share a Co-Prime cell!
    """
    row = idx[:, None]
    col = idx[None, :]
    eye = (row == col).to(tl.float32)

    m1 = ((row >> 1) == (col >> 1)) & (row > col)
    inv = eye - tl.where(m1, A, 0.0)

    m2 = ((row >> 2) == (col >> 2)) & ((row >> 1) > (col >> 1))
    a2 = tl.where(m2, A, 0.0)
    f2 = -_mdot(inv, _mdot(a2, inv, USE_BF16_WMMA), USE_BF16_WMMA)
    inv = inv + tl.where(m2, f2, 0.0)

    m4 = ((row >> 3) == (col >> 3)) & ((row >> 2) > (col >> 2))
    a4 = tl.where(m4, A, 0.0)
    f4 = -_mdot(inv, _mdot(a4, inv, USE_BF16_WMMA), USE_BF16_WMMA)
    inv = inv + tl.where(m4, f4, 0.0)

    m8 = ((row >> 4) == (col >> 4)) & ((row >> 3) > (col >> 3))
    a8 = tl.where(m8, A, 0.0)
    f8 = -_mdot(inv, _mdot(a8, inv, USE_BF16_WMMA), USE_BF16_WMMA)
    inv = inv + tl.where(m8, f8, 0.0)

    m16 = ((row >> 5) == (col >> 5)) & ((row >> 4) > (col >> 4))
    a16 = tl.where(m16, A, 0.0)
    f16 = -_mdot(inv, _mdot(a16, inv, USE_BF16_WMMA), USE_BF16_WMMA)
    inv = inv + tl.where(m16, f16, 0.0)

    m32 = (row >> 5) > (col >> 5)
    a32 = tl.where(m32, A, 0.0)
    f32 = -_mdot(inv, _mdot(a32, inv, USE_BF16_WMMA), USE_BF16_WMMA)
    inv = inv + tl.where(m32, f32, 0.0)

    return inv


@triton.jit
def _vault_wy_intra_chunk_fwd(
    K_ptr, V_ptr, Beta_ptr, CellK_ptr,
    U_ptr, W_ptr,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_bb, stride_bh, stride_bl,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)

    k_tile = tl.load(K_ptr + b * stride_kb + h * stride_kh + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    v_tile = tl.load(V_ptr + b * stride_kb + h * stride_kh + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    beta_vec = tl.load(Beta_ptr + b * stride_bb + h * stride_bh + offs_c * stride_bl).to(tl.float32)
    ck_vec = tl.load(CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)

    gram = _mdot(k_tile, tl.trans(k_tile), USE_BF16_WMMA)

    idx = tl.arange(0, CHUNK_SIZE)
    mask = (idx[:, None] > idx[None, :]) & (ck_vec[:, None] == ck_vec[None, :])
    A = tl.where(mask, beta_vec[:, None] * gram, 0.0)
    inv = _invert_unit_lower_tri_64(A, idx, USE_BF16_WMMA)

    u_tile = _mdot(inv, beta_vec[:, None] * v_tile, USE_BF16_WMMA)
    w_tile = _mdot(inv, beta_vec[:, None] * k_tile, USE_BF16_WMMA)

    off_u = b * stride_ub + h * stride_uh + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    tl.store(U_ptr + off_u, u_tile)
    tl.store(W_ptr + off_u, w_tile)


@triton.jit
def _vault_occupancy_weights_fwd(
    CellQ_ptr, CellK_ptr, Weight_ptr,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_wb, stride_wg, stride_wr, stride_wl,
    num_chunks,
    num_banks,
    G: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    bg = tl.program_id(0).to(tl.int64)
    b = bg // G
    g = bg % G

    c_base_q = CellQ_ptr + b * stride_cb + g * stride_cg
    c_base_k = CellK_ptr + b * stride_cb + g * stride_cg
    w_base = Weight_ptr + b * stride_wb + g * stride_wg

    offs_88 = tl.arange(0, 128)
    idx_c = tl.arange(0, CHUNK_SIZE)
    causal_mask = idx_c[:, None] >= idx_c[None, :]

    occ = tl.zeros([128], dtype=tl.float32)

    for m in range(num_chunks):
        offs_l = m.to(tl.int64) * CHUNK_SIZE + idx_c.to(tl.int64)
        w_sum = tl.zeros([CHUNK_SIZE], dtype=tl.float32)

        for r in range(num_banks):
            r64 = r.to(tl.int64)
            cq = tl.load(c_base_q + r64 * stride_cr + offs_l * stride_cl)
            ck = tl.load(c_base_k + r64 * stride_cr + offs_l * stride_cl)

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
def _vault_inter_chunk_scan_fwd(
    Q_ptr, K_ptr, U_ptr, W_ptr, CellQ_ptr, CellK_ptr,
    YCross_ptr, S_ckpt_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_sb, stride_sh, stride_sm, stride_sdvblk, stride_sc, stride_sflat,
    num_chunks,
    H: tl.constexpr,
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
    Grid: (B * H * R, D // BLOCK_DV)
    Stripped-down Register-Resident State Scan: NO 64x64 Q@K^T or intra-chunk matmuls!
    Only executes 3 tiny WMMA operations on `S_reg` ([32, BLOCK_DV * D]) per chunk!
    """
    bhr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bhr % R
    bh = bhr // R
    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

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

    q_head_base = b * stride_qb + h * stride_qh
    u_bank_base = b * stride_ub + h * stride_uh + r * stride_ur
    c_bank_base = b * stride_cb + g * stride_cg + r * stride_cr
    s_head_base = b * stride_sb + h * stride_sh + dv_blk * stride_sdvblk + (base_cell + offs_32[:, None]).to(tl.int64) * stride_sc + offs_flat[None, :] * stride_sflat

    for m in range(num_chunks):
        m64 = m.to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        if SAVE_CHECKPOINTS:
            tl.store(S_ckpt_ptr + s_head_base + m64 * stride_sm, S_reg.to(S_ckpt_ptr.dtype.element_ty), mask=c_mask)

        q_tile = tl.load(Q_ptr + q_head_base + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
        k_tile = tl.load(K_ptr + q_head_base + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)

        cq_vec = tl.load(CellQ_ptr + c_bank_base + offs_l * stride_cl) - base_cell
        ck_vec = tl.load(CellK_ptr + c_bank_base + offs_l * stride_cl) - base_cell
        oq_f = (cq_vec[:, None] == offs_32[None, :]).to(tl.float32)
        ok_f = (ck_vec[:, None] == offs_32[None, :]).to(tl.float32)

        off_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        u_sub = tl.load(U_ptr + off_sub).to(tl.float32)

        s_ck = tl.reshape(_mdot(ok_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        delta_sub = u_sub - tl.sum(s_ck * w_tile[:, None, :], axis=2)
        tl.store(Delta_ptr + off_sub, delta_sub)

        s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
        tl.store(YCross_ptr + off_sub, y_cross_sub)

        upd_flat = tl.reshape(delta_sub[:, :, None] * k_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        S_reg += _mdot(tl.trans(ok_f), upd_flat, USE_BF16_WMMA)


@triton.jit
def _vault_intra_readout_fwd(
    Q_ptr, K_ptr, Delta_ptr, YCross_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    OutBanks_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_wb, stride_wg, stride_wr, stride_wl,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    """
    Grid: (num_chunks, B * H, R) — 100% PARALLEL ACROSS ALL CHUNKS AND ALL D=64 CHANNELS!
    """
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
    k_tile = tl.load(K_ptr + b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd).to(tl.float32)

    off_u = b * stride_ub + h * stride_uh + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    delta_tile = tl.load(Delta_ptr + off_u).to(tl.float32)
    y_cross_tile = tl.load(YCross_ptr + off_u).to(tl.float32)

    cq_vec = tl.load(CellQ_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)
    ck_vec = tl.load(CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)
    w_norm = tl.load(Weight_ptr + b * stride_wb + g * stride_wg + r * stride_wr + offs_c * stride_wl).to(tl.float32)

    same_qk = (cq_vec[:, None] == ck_vec[None, :]) & (idx[:, None] >= idx[None, :])
    attn_qk = tl.where(same_qk, _mdot(q_tile, tl.trans(k_tile), USE_BF16_WMMA), 0.0)
    y_intra_tile = _mdot(attn_qk, delta_tile, USE_BF16_WMMA)

    tl.store(OutBanks_ptr + off_u, w_norm[:, None] * (y_cross_tile + y_intra_tile))


# ==============================================================================
# 3. TRITON 3.8 BACKWARD KERNELS (SPLIT CHUNK-PARALLEL WMMA + LEAN STATE SCAN)
# ==============================================================================

@triton.jit
def _vault_intra_readout_bwd(
    Q_ptr, K_ptr, Delta_ptr, dOut_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    dDeltaIntra_ptr, dQ_banks_ptr, dK_banks_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_wb, stride_wg, stride_wr, stride_wl,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    """
    Grid: (num_chunks, B * H, R) — 100% PARALLEL ACROSS ALL CHUNKS, ZERO ATOMICS!
    Computes `ddelta_intra`, `dq_intra`, `dk_intra` across all D=64 channels simultaneously!
    """
    chunk_idx = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    idx = tl.arange(0, CHUNK_SIZE)

    off_q = b * stride_qb + h * stride_qh + offs_c[:, None] * stride_ql + offs_d[None, :] * stride_qd
    q_tile = tl.load(Q_ptr + off_q).to(tl.float32)
    k_tile = tl.load(K_ptr + off_q).to(tl.float32)
    dout_tile = tl.load(dOut_ptr + off_q).to(tl.float32)

    off_u = b * stride_ub + h * stride_uh + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    delta_tile = tl.load(Delta_ptr + off_u).to(tl.float32)

    cq_vec = tl.load(CellQ_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)
    ck_vec = tl.load(CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)
    w_norm = tl.load(Weight_ptr + b * stride_wb + g * stride_wg + r * stride_wr + offs_c * stride_wl).to(tl.float32)

    dy_tile = w_norm[:, None] * dout_tile
    same_qk = (cq_vec[:, None] == ck_vec[None, :]) & (idx[:, None] >= idx[None, :])

    attn_qk = tl.where(same_qk, _mdot(q_tile, tl.trans(k_tile), USE_BF16_WMMA), 0.0)
    ddelta_intra = _mdot(tl.trans(attn_qk), dy_tile, USE_BF16_WMMA)

    d_attn = tl.where(same_qk, _mdot(dy_tile, tl.trans(delta_tile), USE_BF16_WMMA), 0.0)
    dq_intra = _mdot(d_attn, k_tile, USE_BF16_WMMA)
    dk_intra = _mdot(tl.trans(d_attn), q_tile, USE_BF16_WMMA)

    tl.store(dDeltaIntra_ptr + off_u, ddelta_intra)
    tl.store(dQ_banks_ptr + off_u, dq_intra)
    tl.store(dK_banks_ptr + off_u, dk_intra)


@triton.jit
def _vault_inter_chunk_scan_bwd(
    Q_ptr, K_ptr, W_ptr, Delta_ptr, dDeltaIntra_ptr, CellQ_ptr, CellK_ptr, Weight_ptr,
    S_ckpt_ptr, dOut_ptr,
    dQ_banks_ptr, dK_banks_ptr, dU_ptr, dW_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_wb, stride_wg, stride_wr, stride_wl,
    stride_sb, stride_sh, stride_sm, stride_sdvblk, stride_sc, stride_sflat,
    num_chunks,
    L_total,
    NUM_DV_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FLAT_DVD: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
    ATOMIC_REDUCE: tl.constexpr,
):
    """
    Grid: (B * H * R, D // BLOCK_DV)
    Stripped-down Register-Resident Reverse Adjoint: NO 64x64 Q@K^T or d_attn GEMMs!
    Uses zero-atomic disjoint coalesced stores when ATOMIC_REDUCE=False.
    """
    bhr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)

    r = bhr % R
    bh = bhr // R
    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

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

    q_head_base = b * stride_qb + h * stride_qh
    u_bank_base = b * stride_ub + h * stride_uh + r * stride_ur
    c_bank_base = b * stride_cb + g * stride_cg + r * stride_cr
    w_bank_base = b * stride_wb + g * stride_wg + r * stride_wr
    s_head_base = b * stride_sb + h * stride_sh + dv_blk * stride_sdvblk + (base_cell + offs_32[:, None]).to(tl.int64) * stride_sc + offs_flat[None, :] * stride_sflat
    L_tot64 = L_total.to(tl.int64)
    dv_out_base = (bhr * NUM_DV_BLOCKS + dv_blk) * (L_tot64 * D)

    for rev_m in range(num_chunks):
        m64 = (num_chunks - 1 - rev_m).to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        S_snap = tl.load(S_ckpt_ptr + s_head_base + m64 * stride_sm, mask=c_mask, other=0.0).to(tl.float32)

        q_tile = tl.load(Q_ptr + q_head_base + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
        k_tile = tl.load(K_ptr + q_head_base + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)

        cq_vec = tl.load(CellQ_ptr + c_bank_base + offs_l * stride_cl) - base_cell
        ck_vec = tl.load(CellK_ptr + c_bank_base + offs_l * stride_cl) - base_cell
        w_norm = tl.load(Weight_ptr + w_bank_base + offs_l * stride_wl).to(tl.float32)

        oq_f = (cq_vec[:, None] == offs_32[None, :]).to(tl.float32)
        ok_f = (ck_vec[:, None] == offs_32[None, :]).to(tl.float32)

        dout_sub = tl.load(dOut_ptr + q_head_base + offs_l[:, None] * stride_ql + offs_dv[None, :] * stride_qd).to(tl.float32)
        dy_sub = w_norm[:, None] * dout_sub

        off_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        delta_sub = tl.load(Delta_ptr + off_sub).to(tl.float32)
        ddelta_intra = tl.load(dDeltaIntra_ptr + off_sub).to(tl.float32)

        ds_ck = tl.reshape(_mdot(ok_f, dS_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        s_ck = tl.reshape(_mdot(ok_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        s_cq = tl.reshape(_mdot(oq_f, S_snap, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])

        ddelta = ddelta_intra + tl.sum(ds_ck * k_tile[:, None, :], axis=2)
        tl.store(dU_ptr + off_sub, ddelta)

        dw_part = -tl.sum(s_ck * ddelta[:, :, None], axis=1)
        dq_cross = tl.sum(s_cq * dy_sub[:, :, None], axis=1)
        dk_state = tl.sum(ds_ck * delta_sub[:, :, None], axis=1)

        if ATOMIC_REDUCE:
            off_full = u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud
            tl.atomic_add(dW_ptr + off_full, dw_part)
            tl.atomic_add(dQ_banks_ptr + off_full, dq_cross)
            tl.atomic_add(dK_banks_ptr + off_full, dk_state)
        else:
            off_dv_out = dv_out_base + offs_l[:, None] * D + offs_dk[None, :]
            tl.store(dW_ptr + off_dv_out, dw_part)
            tl.store(dQ_banks_ptr + off_dv_out, dq_cross)
            tl.store(dK_banks_ptr + off_dv_out, dk_state)

        upd_q_flat = tl.reshape(dy_sub[:, :, None] * q_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        upd_k_flat = tl.reshape(-ddelta[:, :, None] * w_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        dS_reg += _mdot(tl.trans(oq_f), upd_q_flat, USE_BF16_WMMA)
        dS_reg += _mdot(tl.trans(ok_f), upd_k_flat, USE_BF16_WMMA)


@triton.jit
def _vault_wy_intra_chunk_bwd(
    K_ptr, V_ptr, Beta_ptr, CellK_ptr,
    U_ptr, W_ptr, dU_ptr, dW_ptr,
    dK_wy_ptr, dV_wy_ptr, dBeta_wy_ptr,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_bb, stride_bh, stride_bl,
    stride_cb, stride_cg, stride_cr, stride_cl,
    stride_ub, stride_uh, stride_ur, stride_ul, stride_ud,
    stride_db, stride_dh, stride_dr, stride_dl,
    H: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
):
    chunk_idx = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    r = tl.program_id(2).to(tl.int64)

    b = bh // H
    h = bh % H
    g = h // HEADS_PER_GROUP

    offs_c = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)

    k_tile = tl.load(K_ptr + b * stride_kb + h * stride_kh + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    v_tile = tl.load(V_ptr + b * stride_kb + h * stride_kh + offs_c[:, None] * stride_kl + offs_d[None, :] * stride_kd).to(tl.float32)
    beta_vec = tl.load(Beta_ptr + b * stride_bb + h * stride_bh + offs_c * stride_bl).to(tl.float32)
    ck_vec = tl.load(CellK_ptr + b * stride_cb + g * stride_cg + r * stride_cr + offs_c * stride_cl)

    off_u = b * stride_ub + h * stride_uh + r * stride_ur + offs_c[:, None] * stride_ul + offs_d[None, :] * stride_ud
    u_tile = tl.load(U_ptr + off_u).to(tl.float32)
    w_tile = tl.load(W_ptr + off_u).to(tl.float32)
    du_tile = tl.load(dU_ptr + off_u).to(tl.float32)
    dw_tile = tl.load(dW_ptr + off_u).to(tl.float32)

    gram = _mdot(k_tile, tl.trans(k_tile), USE_BF16_WMMA)
    idx = tl.arange(0, CHUNK_SIZE)
    mask = (idx[:, None] > idx[None, :]) & (ck_vec[:, None] == ck_vec[None, :])
    A = tl.where(mask, beta_vec[:, None] * gram, 0.0)
    inv = _invert_unit_lower_tri_64(A, idx, USE_BF16_WMMA)

    inv_T = tl.trans(inv)
    g_u = _mdot(inv_T, du_tile, USE_BF16_WMMA)
    g_w = _mdot(inv_T, dw_tile, USE_BF16_WMMA)

    raw_dA = -(_mdot(g_u, tl.trans(u_tile), USE_BF16_WMMA) + _mdot(g_w, tl.trans(w_tile), USE_BF16_WMMA))
    dA = tl.where(mask, raw_dA, 0.0)

    dv_r = beta_vec[:, None] * g_u
    dbeta_from_vw = tl.sum(g_u * v_tile + g_w * k_tile, axis=1)
    dbeta_from_A = tl.sum(dA * gram, axis=1)
    dbeta_r = dbeta_from_vw + dbeta_from_A

    dgram = beta_vec[:, None] * dA
    dk_from_A = _mdot(dgram + tl.trans(dgram), k_tile, USE_BF16_WMMA)
    dk_r = beta_vec[:, None] * g_w + dk_from_A

    tl.store(dK_wy_ptr + off_u, dk_r)
    tl.store(dV_wy_ptr + off_u, dv_r)
    tl.store(dBeta_wy_ptr + b * stride_db + h * stride_dh + r * stride_dr + offs_c * stride_dl, dbeta_r)


# ==============================================================================
# 4. PYTORCH AUTOGRAD WRAPPER (`FlashVAULTFunction`)
# ==============================================================================

class FlashVAULTFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        phi_q: torch.Tensor,   # (B, H, L, D)
        phi_k: torch.Tensor,   # (B, H, L, D)
        v: torch.Tensor,       # (B, H, L, D)
        beta: torch.Tensor,    # (B, H, L)
        cell_q: torch.Tensor,  # (B, G, R, L) int64
        cell_k: torch.Tensor,  # (B, G, R, L) int64
        chunk_size: int = 64,
    ) -> torch.Tensor:
        orig_dtype = v.dtype
        use_bf16_wmma = orig_dtype in (torch.bfloat16, torch.float16)
        block_dv = 4 if use_bf16_wmma else 1  # BF16 WMMA uses half LDS -> 4x larger DV tile (<40KB LDS)!

        B, H, L_orig, D = v.shape
        _, G, R, _ = cell_k.shape
        assert H % G == 0, "n_heads must be divisible by n_vault_groups"
        assert D == 64 and R == 4, "FlashVAULT targets head_dim=64, R=4 Co-Prime banks"

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
        save_ckpts = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]

        pq = phi_q.contiguous()
        pk = phi_k.contiguous()
        vc = v.contiguous()
        bc = beta.contiguous()
        cq = cell_q.contiguous()
        ck = cell_k.contiguous()

        U = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        W = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        Delta = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        YCross = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        OutBanks = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
        Weights = torch.empty((B, G, R, L), device=v.device, dtype=torch.float32)

        if save_ckpts:
            S_ckpt = torch.empty((B, H, num_chunks, num_dv_blocks, 88, flat_dvd), device=v.device, dtype=orig_dtype)
        else:
            S_ckpt = torch.empty((1, 1, 1, 1, 1, 1), device=v.device, dtype=orig_dtype)

        # 1. Chunk-Parallel WY Solver
        _vault_wy_intra_chunk_fwd[(num_chunks, B * H, R)](
            pk, vc, bc, ck, U, W,
            *pk.stride(), *bc.stride(), *ck.stride(), *U.stride(),
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )

        # 2. Exact Cell Occupancy Weights
        _vault_occupancy_weights_fwd[(B * G,)](
            cq, ck, Weights,
            ck.stride(0), ck.stride(1), ck.stride(2), ck.stride(3),
            Weights.stride(0), Weights.stride(1), Weights.stride(2), Weights.stride(3),
            num_chunks, R,
            G=G, CHUNK_SIZE=chunk_size, D=D,
            num_warps=4, num_stages=1,
        )

        # 3. Lean Register-Resident Inter-Chunk State Scan (zero 64x64 Q@K^T overhead!)
        _vault_inter_chunk_scan_fwd[(B * H * R, num_dv_blocks)](
            pq, pk, U, W, cq, ck,
            YCross, S_ckpt, Delta,
            *pq.stride(), *U.stride(), *ck.stride(), *S_ckpt.stride(),
            num_chunks,
            H=H, R=R, HEADS_PER_GROUP=heads_per_group,
            CHUNK_SIZE=chunk_size, D=D,
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd,
            SAVE_CHECKPOINTS=save_ckpts, USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )
        if not save_ckpts:
            del U, W

        # 4. Chunk-Parallel Intra-Chunk Readout & Consensus Fusion
        _vault_intra_readout_fwd[(num_chunks, B * H, R)](
            pq, pk, Delta, YCross, cq, ck, Weights,
            OutBanks,
            *pq.stride(), *Delta.stride(), *ck.stride(), *Weights.stride(),
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

        elem_bytes = 2 if orig_dtype in (torch.bfloat16, torch.float16) else 4
        dv_buf_bytes = 3 * B * H * R * num_dv_blocks * L * D * elem_bytes
        use_atomic_reduce = dv_buf_bytes > 1_800_000_000

        dDeltaIntra = torch.empty((B, H, R, L, D), device=pq.device, dtype=orig_dtype)
        intra_grad_dtype = torch.float32 if use_atomic_reduce else orig_dtype
        dQ_intra = torch.empty((B, H, R, L, D), device=pq.device, dtype=intra_grad_dtype)
        dK_intra = torch.empty((B, H, R, L, D), device=pq.device, dtype=intra_grad_dtype)

        # 1. Chunk-Parallel Intra-Readout Adjoint (initializes dQ_intra & dK_intra with ZERO atomics!)
        _vault_intra_readout_bwd[(num_chunks, B * H, R)](
            pq, pk, Delta, dout, cq, ck, Weights,
            dDeltaIntra, dQ_intra, dK_intra,
            *pq.stride(), *U.stride(), *ck.stride(), *Weights.stride(),
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
            USE_BF16_WMMA=use_bf16_wmma,
            num_warps=4, num_stages=1,
        )

        dU = torch.empty((B, H, R, L, D), device=pq.device, dtype=orig_dtype)
        if use_atomic_reduce:
            dW = torch.zeros((B, H, R, L, D), device=pq.device, dtype=torch.float32)
            dQ_scan_target = dQ_intra
            dK_scan_target = dK_intra
            dW_scan_target = dW
        else:
            dQ_scan_target = torch.empty((B * H * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)
            dK_scan_target = torch.empty((B * H * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)
            dW_scan_target = torch.empty((B * H * R, num_dv_blocks, L, D), device=pq.device, dtype=orig_dtype)

        # 2. Lean Register-Resident Reverse State Adjoint
        _vault_inter_chunk_scan_bwd[(B * H * R, num_dv_blocks)](
            pq, pk, W, Delta, dDeltaIntra, cq, ck, Weights,
            S_ckpt, dout,
            dQ_scan_target, dK_scan_target, dU, dW_scan_target,
            *pq.stride(), *U.stride(), *ck.stride(), *Weights.stride(), *S_ckpt.stride(),
            num_chunks, L,
            NUM_DV_BLOCKS=num_dv_blocks,
            H=H, R=R, HEADS_PER_GROUP=heads_per_group,
            CHUNK_SIZE=chunk_size, D=D,
            BLOCK_DV=block_dv, FLAT_DVD=flat_dvd,
            USE_BF16_WMMA=use_bf16_wmma,
            ATOMIC_REDUCE=use_atomic_reduce,
            num_warps=4, num_stages=1,
        )
        del dDeltaIntra

        if use_atomic_reduce:
            dQ = dQ_intra.sum(dim=2).to(pq.dtype)
            del dQ_intra
            dK_banks = dK_intra
        else:
            dQ_cross = dQ_scan_target.float().sum(dim=1).view(B, H, R, L, D)
            del dQ_scan_target
            dQ = (dQ_intra.float() + dQ_cross).sum(dim=2).to(pq.dtype)
            del dQ_intra, dQ_cross

            dK_cross = dK_scan_target.float().sum(dim=1).view(B, H, R, L, D)
            del dK_scan_target
            dK_banks = dK_intra.float() + dK_cross
            del dK_intra, dK_cross

            dW = dW_scan_target.float().sum(dim=1).view(B, H, R, L, D).to(orig_dtype)
            del dW_scan_target

        dK_wy = torch.empty((B, H, R, L, D), device=pq.device, dtype=orig_dtype)
        dV_wy = torch.empty((B, H, R, L, D), device=pq.device, dtype=orig_dtype)
        dBeta_wy = torch.empty((B, H, R, L), device=pq.device, dtype=torch.float32)

        # 3. Chunk-Parallel WY Adjoint
        _vault_wy_intra_chunk_bwd[(num_chunks, B * H, R)](
            pk, vc, bc, ck, U, W, dU, dW,
            dK_wy, dV_wy, dBeta_wy,
            *pk.stride(), *bc.stride(), *ck.stride(), *U.stride(), *dBeta_wy.stride(),
            H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
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

        return (
            dQ,
            dK,
            dV,
            dBeta,
            None,
            None,
            None,
        )


def flash_vault_chunkwise(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cell_q: torch.Tensor,
    cell_k: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Executes Split Chunk-Parallel RDNA4 WMMA + Register-Resident State Scan FlashVAULT.
    """
    return FlashVAULTFunction.apply(phi_q, phi_k, v, beta, cell_q, cell_k, chunk_size)
