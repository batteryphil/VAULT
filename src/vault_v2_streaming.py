# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2 / v2.3 / v2.4 True O(1)-State Streaming Prefill & Autoregressive Decode Engine
#
# Capabilities:
#   1. `stateful_flash_vault_v2_chunkwise`:
#      - Carries exact FP32 Co-Prime fast-weight matrices `S_state (B, G, C_tot, D, D)` and
#        occupancy counters `occ_state (B, G, C_tot)` across arbitrary streaming segments.
#      - Supports general 4-bank Co-Prime Voronoi widths up to `max(prime_banks) <= 256`
#        (`BANK_PAD in {32, 64, 128, 256}`), enabling both `220`-cell `VAULT v2.4`
#        `P=(47,53,59,61)` (`8.97M` CRT addresses) and zero-shot / continual Elastic Mitosis
#        `508`-cell `P=(113,127,131,137)` (`257.6M` CRT addresses) & `942`-cell `P=(229,233,239,241)`
#        (`3.07B` CRT addresses) on ROCm RDNA4 WMMA and CPU!
#   2. `VAULTv2StreamingEngine.prefill_streaming`:
#      - Streams arbitrarily long contexts (`32K`, `64K`, `128K`, `256K`, `1M+` tokens) in fixed
#        `segment_size` (e.g. `4096`) chunks with strictly O(1) constant activation VRAM (`< 1 GB`),
#        mathematically bit-exact to monolithic full-sequence `VAULTLMv2.forward`!
#   3. `VAULTv2StreamingEngine.decode_step` & `generate_greedy`:
#      - True O(1) per-token recurrent autoregressive decoding (`L = 1` GEMMs + O(1) incremental
#        Co-Prime Windowed CMS + O(1) 4-bank Voronoi fast-weight update/query), completely independent
#        of prompt length (`2K` vs `128K` has identical per-token latency & memory!).

import atexit
import math
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    from src.kernels.flash_vault_v2_triton import triton, tl  # type: ignore

from src.vault_v2_model import (
    VAULTLMv2,
    VAULTv2Attention,
    SalienceEntityBinder,
    _apply_local_rope,
    _chunked_cumsum_dim2,
    _shift_right_zero,
    compute_v2_2_shared_aux,
    fft_bind_with_id,
    fft_circular_bind,
)
from src.kernels.flash_vault_v2_triton import (
    _gpu_supports_bf16_triton,
    _is_rocm_backend,
    _mdot,
    _vault_v2_wy_fwd,
    _vault_v2_chunk_occ_hist_fwd,
    _vault_v2_occ_weights_chunk_fwd,
    _vault_v2_intra_readout_fused_fwd,
    causal_depthwise_conv1d,
)


# ==============================================================================
# 1. STATEFUL TRITON WMMA & VECTORIZED CPU COMPACT-WY CHUNK KERNELS
# ==============================================================================

@triton.jit
def _vault_v2_stateful_inter_scan_fwd(
    Q_ptr, K_ptr, U_ptr, W_ptr, CellQ_ptr, CellK_ptr,
    YCross_ptr, S_state_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kg, stride_kl, stride_kd,
    stride_ub, stride_ug, stride_ur, stride_ul, stride_ud,
    stride_cqb, stride_cqh, stride_cqr, stride_cql,
    stride_ckb, stride_ckg, stride_ckr, stride_ckl,
    stride_yb, stride_yh, stride_yr, stride_yl, stride_yd,
    stride_ssb, stride_ssg, stride_ssc,
    num_chunks,
    G: tl.constexpr,
    R: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FLAT_DVD: tl.constexpr,
    TILE_P: tl.constexpr,
    NUM_P_TILES: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    USE_BF16_WMMA: tl.constexpr,
    SHARED_GROUP_CELL_Q: tl.constexpr = False,
):
    """
    Stateful branch-free WMMA inter-chunk scan:
      - Loads initial `S_reg [TILE_P, FLAT_DVD]` from `S_state (B, G, total_cells, D, D)` FP32.
      - Scans `num_chunks` chunks of size `CHUNK_SIZE = 64`.
      - Writes updated `S_reg` back into `S_state` in-place.
      - Supports arbitrary `BANK_PAD` (`32 .. 512+`) via disjoint `TILE_P <= 256` cell sub-tiles
        with constant 32 VGPRs/lane and `<= 64 KB` LDS on RDNA4!
    """
    bgr = tl.program_id(0).to(tl.int64)
    dv_blk = tl.program_id(1).to(tl.int64)
    p_blk = tl.program_id(2).to(tl.int32)

    r = bgr % R
    bg = bgr // R
    b = bg // G
    g = bg % G

    r32 = r.to(tl.int32)
    base_cell = tl.where(r32 == 0, 0, tl.where(r32 == 1, C1, tl.where(r32 == 2, C2, C3)))
    bank_size = tl.where(r32 == 0, P0, tl.where(r32 == 1, P1, tl.where(r32 == 2, P3, P3)))
    bank_size = tl.where(r32 == 2, P2, bank_size)
    p_start = p_blk * TILE_P
    tile_base_cell = base_cell + p_start

    offs_dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV).to(tl.int64)
    offs_dk = tl.arange(0, D).to(tl.int64)
    offs_p = tl.arange(0, TILE_P)
    offs_flat = tl.arange(0, FLAT_DVD).to(tl.int64)
    idx_c = tl.arange(0, CHUNK_SIZE).to(tl.int64)
    c_mask = (p_start + offs_p[:, None]) < bank_size

    # Coalesced stride-1 slice into S_state[b, g, tile_base_cell + offs_p, dv_blk*BLOCK_DV : (dv_blk+1)*BLOCK_DV, :]
    s_state_ptrs = (
        S_state_ptr
        + b * stride_ssb
        + g * stride_ssg
        + (tile_base_cell + offs_p[:, None]).to(tl.int64) * stride_ssc
        + (dv_blk * FLAT_DVD + offs_flat[None, :])
    )
    S_reg = tl.load(s_state_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    k_grp_base = b * stride_kb + g * stride_kg
    u_bank_base = b * stride_ub + g * stride_ug + r * stride_ur
    ck_bank_base = b * stride_ckb + g * stride_ckg + r * stride_ckr

    for m in range(num_chunks):
        m64 = m.to(tl.int64)
        offs_l = m64 * CHUNK_SIZE + idx_c

        k_tile = tl.load(K_ptr + k_grp_base + offs_l[:, None] * stride_kl + offs_dk[None, :] * stride_kd).to(tl.float32)
        w_tile = tl.load(W_ptr + u_bank_base + offs_l[:, None] * stride_ul + offs_dk[None, :] * stride_ud).to(tl.float32)
        ck_vec = tl.load(CellK_ptr + ck_bank_base + offs_l * stride_ckl) - tile_base_cell
        ok_f = (ck_vec[:, None] == offs_p[None, :]).to(tl.float32)

        off_u_sub = u_bank_base + offs_l[:, None] * stride_ul + offs_dv[None, :] * stride_ud
        u_sub = tl.load(U_ptr + off_u_sub).to(tl.float32)

        s_ck = tl.reshape(_mdot(ok_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
        delta_sub = u_sub - tl.sum(s_ck * w_tile[:, None, :], axis=2)
        if NUM_P_TILES == 1:
            tl.store(Delta_ptr + off_u_sub, delta_sub)
        else:
            ck_own = (ck_vec >= 0) & (ck_vec < TILE_P)
            tl.store(Delta_ptr + off_u_sub, delta_sub, mask=ck_own[:, None])

        if SHARED_GROUP_CELL_Q:
            h0 = g * HEADS_PER_GROUP
            cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h0 * stride_cqh + r * stride_cqr + offs_l * stride_cql) - tile_base_cell
            oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)
            s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = h0 + j
                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
                off_y_sub = b * stride_yb + h * stride_yh + r * stride_yr + offs_l[:, None] * stride_yl + offs_dv[None, :] * stride_yd
                if NUM_P_TILES == 1:
                    tl.store(YCross_ptr + off_y_sub, y_cross_sub)
                else:
                    cq_own = (cq_vec >= 0) & (cq_vec < TILE_P)
                    tl.store(YCross_ptr + off_y_sub, y_cross_sub, mask=cq_own[:, None])
        else:
            for j in tl.static_range(0, HEADS_PER_GROUP):
                h = g * HEADS_PER_GROUP + j
                q_tile = tl.load(Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_dk[None, :] * stride_qd).to(tl.float32)
                cq_vec = tl.load(CellQ_ptr + b * stride_cqb + h * stride_cqh + r * stride_cqr + offs_l * stride_cql) - tile_base_cell
                oq_f = (cq_vec[:, None] == offs_p[None, :]).to(tl.float32)
                s_cq = tl.reshape(_mdot(oq_f, S_reg, USE_BF16_WMMA), [CHUNK_SIZE, BLOCK_DV, D])
                y_cross_sub = tl.sum(s_cq * q_tile[:, None, :], axis=2)
                off_y_sub = b * stride_yb + h * stride_yh + r * stride_yr + offs_l[:, None] * stride_yl + offs_dv[None, :] * stride_yd
                if NUM_P_TILES == 1:
                    tl.store(YCross_ptr + off_y_sub, y_cross_sub)
                else:
                    cq_own = (cq_vec >= 0) & (cq_vec < TILE_P)
                    tl.store(YCross_ptr + off_y_sub, y_cross_sub, mask=cq_own[:, None])

        upd_flat = tl.reshape(delta_sub[:, :, None] * k_tile[:, None, :], [CHUNK_SIZE, FLAT_DVD])
        S_reg += _mdot(tl.trans(ok_f), upd_flat, USE_BF16_WMMA)

    tl.store(s_state_ptrs, S_reg, mask=c_mask)


@torch.no_grad()
def _stateful_vault_v2_cpu_chunkwise(
    phi_q: torch.Tensor,      # (B, H, L, D)
    phi_k: torch.Tensor,      # (B, G, L, D)
    v: torch.Tensor,          # (B, G, L, D)
    beta: torch.Tensor,       # (B, G, L)
    cell_q: torch.Tensor,     # (B, H, R, L)
    cell_k: torch.Tensor,     # (B, G, R, L)
    S_state: torch.Tensor,    # (B, G, total_cells, D, D) float32, updated in-place
    occ_state: torch.Tensor,  # (B, G, total_cells) float32, updated in-place
    chunk_size: int = 64,
    v2_2_beta_occ: bool = True,
    occ_alpha: float = 0.5,
) -> torch.Tensor:
    """
    Vectorized Compact-WY chunked reference for CPU verification (zero per-token Python loops).
    Updates `S_state` and `occ_state` in-place with exact FP32 precision.
    """
    orig_dtype = v.dtype
    B, H, L_orig, D = phi_q.shape
    _, G, R, _ = cell_k.shape
    heads_per_group = H // G

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
    M = L // chunk_size
    C = chunk_size

    pq = phi_q.float().view(B, H, M, C, D)
    pk = phi_k.float().view(B, G, M, C, D)
    vf = v.float().view(B, G, M, C, D)
    bf = beta.float().view(B, G, M, C)
    cq = cell_q.view(B, H, R, M, C)
    ck = cell_k.view(B, G, R, M, C)

    idx = torch.arange(C, device=v.device)
    strict_lower = idx[:, None] > idx[None, :]
    causal = idx[:, None] >= idx[None, :]
    eye_c = torch.eye(C, device=v.device, dtype=torch.float32)

    b_g = torch.arange(B, device=v.device)[:, None, None, None]
    g_g = torch.arange(G, device=v.device)[None, :, None, None]
    b_h = torch.arange(B, device=v.device)[:, None, None, None]
    g_for_h = (torch.arange(H, device=v.device) // heads_per_group)[None, :, None, None]

    out_chunks = []
    for m in range(M):
        km = pk[:, :, m]        # (B, G, C, D)
        vm = vf[:, :, m]        # (B, G, C, D)
        bm = bf[:, :, m]        # (B, G, C)
        ckm = ck[:, :, :, m]    # (B, G, R, C)
        cqm = cq[:, :, :, m]    # (B, H, R, C)
        qm = pq[:, :, m]        # (B, H, C, D)

        kk = torch.matmul(km, km.transpose(-1, -2)).unsqueeze(2)  # (B, G, 1, C, C)
        same_cell = (ckm.unsqueeze(-1) == ckm.unsqueeze(-2)) & strict_lower
        T_mat = eye_c + torch.where(same_cell, bm.unsqueeze(2).unsqueeze(-1) * kk, 0.0)
        bv = (bm.unsqueeze(-1) * vm).unsqueeze(2).expand(B, G, R, C, D)
        bk = (bm.unsqueeze(-1) * km).unsqueeze(2).expand(B, G, R, C, D)
        uw = torch.linalg.solve_triangular(T_mat, torch.cat([bv, bk], dim=-1), upper=False, unitriangular=True)
        u_tile = uw[..., :D]    # (B, G, R, C, D)
        w_tile = uw[..., D:]    # (B, G, R, C, D)

        # Gather prior S at ckm: (B, G, R, C, D, D)
        S_ck = S_state[b_g, g_g, ckm]
        delta_tile = u_tile - torch.einsum("bgrcvd,bgrcd->bgrcv", S_ck, w_tile)

        # Gather prior S and occ at cqm: (B, H, R, C, ...)
        S_cq = S_state[b_h, g_for_h, cqm]
        occ_prior = occ_state[b_h, g_for_h, cqm]  # (B, H, R, C)
        y_cross = torch.einsum("bhrcvd,bhcd->bhrcv", S_cq, qm)

        # Intra-chunk readout + exact occupancy weights
        km_h = km.repeat_interleave(heads_per_group, dim=1)       # (B, H, C, D)
        ckm_h = ckm.repeat_interleave(heads_per_group, dim=1)     # (B, H, R, C)
        bm_h = bm.repeat_interleave(heads_per_group, dim=1)       # (B, H, C)
        delta_h = delta_tile.repeat_interleave(heads_per_group, dim=1)  # (B, H, R, C, D)

        qk_raw = torch.matmul(qm, km_h.transpose(-1, -2)).unsqueeze(2)  # (B, H, 1, C, C)
        same_qk = (cqm.unsqueeze(-1) == ckm_h.unsqueeze(-2)) & causal   # (B, H, R, C, C)
        y_intra = torch.matmul(torch.where(same_qk, qk_raw, 0.0), delta_h)

        if v2_2_beta_occ:
            occ_intra = torch.where(same_qk, bm_h.unsqueeze(2).unsqueeze(-2), 0.0).sum(dim=-1)
            n_tot = occ_prior + occ_intra
            if occ_alpha == 0.5:
                w_r = torch.where(n_tot > 1e-3, torch.rsqrt(1.0 + n_tot), torch.zeros_like(n_tot))
            else:
                w_r = torch.where(n_tot > 1e-3, (1.0 + n_tot).pow(-occ_alpha), torch.zeros_like(n_tot))
        else:
            occ_intra = same_qk.float().sum(dim=-1)
            n_tot = occ_prior + occ_intra
            w_r = torch.where(n_tot > 0.0, torch.rsqrt(1.0 + n_tot), torch.zeros_like(n_tot))

        w_sum = w_r.sum(dim=2, keepdim=True)
        w_norm = torch.where(w_sum > 1e-6, w_r / w_sum.clamp(min=1e-6), torch.full_like(w_r, 1.0 / float(R)))
        out_m = (w_norm.unsqueeze(-1) * (y_cross + y_intra)).sum(dim=2)  # (B, H, C, D)
        out_chunks.append(out_m)

        # Commit chunk m into S_state and occ_state (index_add_ handles duplicate cells within chunk m!)
        upd = torch.einsum("bgrcv,bgrcd->bgrcvd", delta_tile, km.unsqueeze(2))  # (B, G, R, C, D, D)
        flat_cells = S_state.size(2)
        S_flat = S_state.view(B * G, flat_cells, D * D)
        ck_idx = ckm.reshape(B * G, R * C)
        upd_flat = upd.reshape(B * G, R * C, D * D)
        occ_flat = occ_state.view(B * G, flat_cells)
        b_add = (bm.unsqueeze(2).expand(B, G, R, C) if v2_2_beta_occ else torch.ones_like(ckm, dtype=torch.float32)).reshape(B * G, R * C)
        for bg_i in range(B * G):
            S_flat[bg_i].index_add_(0, ck_idx[bg_i], upd_flat[bg_i])
            occ_flat[bg_i].index_add_(0, ck_idx[bg_i], b_add[bg_i])

    out = torch.cat(out_chunks, dim=2)  # (B, H, L, D)
    return out[:, :, :L_orig, :].to(orig_dtype)


@torch.no_grad()
def stateful_flash_vault_v2_chunkwise(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cell_q: torch.Tensor,
    cell_k: torch.Tensor,
    S_state: torch.Tensor,
    occ_state: torch.Tensor,
    chunk_size: int = 64,
    v2_1_consensus: bool = True,
    v2_2_beta_occ: bool = True,
    prime_banks: Tuple[int, int, int, int] = (47, 53, 59, 61),
    shared_group_cell_q: bool = True,
    occ_alpha: float = 0.5,
) -> torch.Tensor:
    """
    Stateful chunkwise Co-Prime fast-weight scan on GPU (Triton WMMA) or CPU (Compact-WY PyTorch).
    Supports `max(prime_banks) <= 512` (`BANK_PAD in {32, 64, 128, 256, 512}`)!
    """
    if not (_HAS_TRITON and phi_q.is_cuda):
        return _stateful_vault_v2_cpu_chunkwise(
            phi_q, phi_k, v, beta, cell_q, cell_k, S_state, occ_state,
            chunk_size=chunk_size, v2_2_beta_occ=v2_2_beta_occ, occ_alpha=occ_alpha,
        )

    orig_dtype = v.dtype
    bf16_ok = _gpu_supports_bf16_triton(v.device)
    if orig_dtype == torch.float16 or (orig_dtype == torch.bfloat16 and not bf16_ok):
        run_dt = torch.bfloat16 if bf16_ok else torch.float32
        out_run = stateful_flash_vault_v2_chunkwise(
            phi_q.to(run_dt),
            phi_k.to(run_dt),
            v.to(run_dt),
            beta.to(run_dt) if beta.dtype != torch.float32 else beta,
            cell_q,
            cell_k,
            S_state,
            occ_state,
            chunk_size=chunk_size,
            v2_1_consensus=v2_1_consensus,
            v2_2_beta_occ=v2_2_beta_occ,
            prime_banks=prime_banks,
            shared_group_cell_q=shared_group_cell_q,
            occ_alpha=occ_alpha,
        )
        return out_run.to(orig_dtype)

    if orig_dtype not in (torch.float32, torch.bfloat16):
        return _stateful_vault_v2_cpu_chunkwise(
            phi_q, phi_k, v, beta, cell_q, cell_k, S_state, occ_state,
            chunk_size=chunk_size, v2_2_beta_occ=v2_2_beta_occ, occ_alpha=occ_alpha,
        )

    use_bf16_wmma = orig_dtype == torch.bfloat16
    is_rocm = _is_rocm_backend()

    p0, p1, p2, p3 = int(prime_banks[0]), int(prime_banks[1]), int(prime_banks[2]), int(prime_banks[3])
    c1 = p0
    c2 = p0 + p1
    c3 = p0 + p1 + p2
    max_p = max(p0, p1, p2, p3)
    assert max_p <= 512, f"Expected max(prime_banks) <= 512, got {prime_banks}"
    if max_p <= 32:
        bank_pad = 32
    elif max_p <= 64:
        bank_pad = 64
    elif max_p <= 128:
        bank_pad = 128
    elif max_p <= 256:
        bank_pad = 256
    else:
        bank_pad = 512

    B, H, L_orig, D = phi_q.shape
    _, G, R, _ = cell_k.shape
    assert D == 64 and R == 4

    # Hardware-aware tile & warp scheduling:
    #   - ROCm RDNA4: `BANK_PAD * BLOCK_DV <= 512` (~32-64 VGPRs/lane with 8/16 wavefronts) and `tile_p <= 256`.
    #   - NVIDIA CUDA: `tile_p <= 64` and `scan_warps = 4` keeps shared memory `<= 32 KB` across all `88c..2002c`
    #     presets (fitting Turing T4 `64 KB`, Ampere/Ada RTX 30xx/40xx `99 KB`, and A100/H100/Blackwell) with valid
    #     `m16n8k16` TensorCore warp tiling!
    if not is_rocm:
        block_dv = 1 if not use_bf16_wmma else (4 if bank_pad <= 32 else 2)
        scan_warps = 4
    elif not use_bf16_wmma:
        block_dv = 1
        scan_warps = 4 if bank_pad <= 256 else 8
    elif bank_pad <= 32:
        block_dv = 4
        scan_warps = 8
    elif bank_pad <= 64:
        block_dv = 4
        scan_warps = 16
    elif bank_pad <= 128:
        block_dv = 2
        scan_warps = 16
    else:
        block_dv = 1
        scan_warps = 16

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
    heads_per_group = H // G
    num_dv_blocks = D // block_dv
    flat_dvd = block_dv * D

    pq = phi_q.contiguous()
    pk = phi_k.contiguous()
    vc = v.contiguous()
    bc = beta.contiguous()
    cq = cell_q.contiguous()
    ck = cell_k.contiguous()
    assert S_state.is_contiguous() and S_state.dtype == torch.float32

    U = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
    W = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
    Delta = torch.empty((B, G, R, L, D), device=v.device, dtype=orig_dtype)
    YCross = torch.empty((B, H, R, L, D), device=v.device, dtype=orig_dtype)
    Out = torch.empty((B, H, L, D), device=v.device, dtype=orig_dtype)
    Weights = torch.empty((B, H, R, L), device=v.device, dtype=torch.float32)
    ChunkHist = torch.empty((B, G, R, num_chunks, bank_pad), device=v.device, dtype=torch.float32)

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
        num_warps=2 if bank_pad <= 128 else 4, num_stages=1,
    )

    occ_init_pad = torch.zeros((B, G, R, 1, bank_pad), device=v.device, dtype=torch.float32)
    bank_offsets = (0, c1, c2, c3)
    bank_sizes = (p0, p1, p2, p3)
    for r_i, (b_off, b_sz) in enumerate(zip(bank_offsets, bank_sizes)):
        occ_init_pad[:, :, r_i, 0, :b_sz] = occ_state[:, :, b_off : b_off + b_sz]

    ChunkPrior = occ_init_pad + (torch.cumsum(ChunkHist, dim=3) - ChunkHist)
    hist_sum = ChunkHist.sum(dim=3)  # (B, G, R, bank_pad)
    for r_i, (b_off, b_sz) in enumerate(zip(bank_offsets, bank_sizes)):
        occ_state[:, :, b_off : b_off + b_sz] += hist_sum[:, :, r_i, :b_sz]
    del ChunkHist, hist_sum, occ_init_pad

    _vault_v2_occ_weights_chunk_fwd[(num_chunks, B * H)](
        cq, ck, bc, ChunkPrior, Weights,
        *cq.stride(), *ck.stride(), *bc.stride(), *ChunkPrior.stride(), *Weights.stride(),
        R,
        H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
        BANK_PAD=bank_pad, C1=c1, C2=c2, C3=c3,
        V2_1_CONSENSUS=v2_1_consensus,
        V2_2_BETA_OCC=v2_2_beta_occ,
        OCC_ALPHA=float(occ_alpha),
        num_warps=2, num_stages=1,
    )
    del ChunkPrior

    tile_p = min(bank_pad, 256 if is_rocm else 64)
    num_p_tiles = bank_pad // tile_p
    _vault_v2_stateful_inter_scan_fwd[(B * G * R, num_dv_blocks, num_p_tiles)](
        pq, pk, U, W, cq, ck,
        YCross, S_state, Delta,
        *pq.stride(), *pk.stride(), *U.stride(), *cq.stride(), *ck.stride(), *YCross.stride(),
        S_state.stride(0), S_state.stride(1), S_state.stride(2),
        num_chunks,
        G=G, R=R, HEADS_PER_GROUP=heads_per_group,
        CHUNK_SIZE=chunk_size, D=D,
        BLOCK_DV=block_dv, FLAT_DVD=flat_dvd,
        TILE_P=tile_p, NUM_P_TILES=num_p_tiles,
        P0=p0, P1=p1, P2=p2, P3=p3, C1=c1, C2=c2, C3=c3,
        USE_BF16_WMMA=use_bf16_wmma,
        SHARED_GROUP_CELL_Q=shared_group_cell_q,
        num_warps=scan_warps, num_stages=1,
    )
    del U, W

    _vault_v2_intra_readout_fused_fwd[(num_chunks, B * H)](
        pq, pk, Delta, YCross, cq, ck, Weights,
        Out,
        *pq.stride(), *pk.stride(), *Delta.stride(), *cq.stride(), *ck.stride(), *YCross.stride(), *Weights.stride(), *Out.stride(),
        H=H, HEADS_PER_GROUP=heads_per_group, CHUNK_SIZE=chunk_size, D=D,
        USE_BF16_WMMA=use_bf16_wmma,
        num_warps=4, num_stages=1,
    )
    del YCross, Delta, Weights

    return Out[:, :, :L_orig, :]


# ==============================================================================
# 2. CONSTANT-SIZE O(1) STREAMING & AUTOREGRESSIVE DECODE CACHE
# ==============================================================================

@dataclass
class LayerStreamState:
    S_state: torch.Tensor                  # (B, G, total_cells, D, D) float32
    occ_state: torch.Tensor                # (B, G, total_cells) float32
    v_cumsum: torch.Tensor                 # (B, G, 1, D) float32
    qkv_conv_state: torch.Tensor           # (B, 3, q_dim + 2*kv_dim)
    prev_ib_rms_p1: torch.Tensor           # (B, 1, 3)
    prev_sal_iso_p1: torch.Tensor          # (B, 1, 128)
    prev_ib_iso_p1: torch.Tensor           # (B, 1, 3)
    anchor_conv_state: torch.Tensor        # (B, 5, 256)
    span_conv_state: torch.Tensor          # (B, 5, d_model)
    a_raw_ring: torch.Tensor               # (B, 7, 2, 2, D) float32 — last 7 tokens [t-7..t-1]
    a_left_ring: torch.Tensor              # (B, 7, 2, 2, D) float32 — last 7 tokens [t-7..t-1]
    a_ctx_win: torch.Tensor                # (B, 2, 128, 128) comp_dt — [prev_chunk, cur_chunk]
    sal_win: torch.Tensor                  # (B, 2, 128) float32 — [prev_chunk, cur_chunk]
    prev_x0n_s1: torch.Tensor              # (B, 1, 393)
    prev_x0n_s2: torch.Tensor              # (B, 2, 128) — [t-2, t-1]
    r_all_prev: torch.Tensor               # (B, 1, 3, D) float32
    u_ind_prev: torch.Tensor               # (B, 1, 3, D) float32
    ind_delay_pk: torch.Tensor             # (B, 3, 64, D) float32
    ind_delay_v: torch.Tensor              # (B, 3, 64, D) comp_dt
    ind_delay_beta: torch.Tensor           # (B, 3, 64) float32
    ind_delay_ck: torch.Tensor             # (B, 3, 4, 64) int64
    ck_ind_first: torch.Tensor             # (B, 3, 4, 1) int64
    local_k_win: torch.Tensor              # (B, G, 128, D) comp_dt — [prev_chunk, cur_chunk]
    local_v_win: torch.Tensor              # (B, G, 128, D) comp_dt — [prev_chunk, cur_chunk]


@dataclass
class VAULTv2StreamCache:
    batch_size: int
    t_pos: int                             # Total tokens ingested so far (host mirror)
    t_pos_buf: torch.Tensor                # (1,) int64 device counter for zero-sync CUDAGraph decode
    comp_dt: torch.dtype
    device: torch.device
    # Fixed-size 1,185-token circular ring (`slot = t % 1185`) for `x0_norm` history
    x0_hist: torch.Tensor                  # (B, 1185, d_model)
    # Padded 3-bank Windowed CMS counters `(B, 3, 528)` + per-bank views `cms_counts_u` / `cms_counts_b`
    cms_counts_u_pad: torch.Tensor         # (B, 3, 528) int32
    cms_counts_b_pad: torch.Tensor         # (B, 3, 528) int32
    cms_counts_u: List[torch.Tensor]       # [3] views into cms_counts_u_pad[:, i, :p]
    cms_counts_b: List[torch.Tensor]       # [3] views into cms_counts_b_pad[:, i, :p]
    cms_ib_ring: torch.Tensor              # (B, 1057, 3) int64
    cms_bg_ring: torch.Tensor              # (B, 1057, 3) int64
    cms_prev_ib: torch.Tensor              # (B, 3) int64
    cc_ring: torch.Tensor                  # (B, 7, 1) float32 — [t-7..t-1]
    bf_ring: torch.Tensor                  # (B, 7, 1) float32 — [t-7..t-1]
    nov1_ring: torch.Tensor                # (B, 1, 3) float32 — [t-3..t-1]
    layers: List[LayerStreamState] = field(default_factory=list)
    _cg_runner_id: int = 0

    def state_bytes(self) -> int:
        tot = self.x0_hist.numel() * self.x0_hist.element_size()
        for t in [
            self.t_pos_buf,
            self.cms_counts_u_pad,
            self.cms_counts_b_pad,
            self.cms_ib_ring,
            self.cms_bg_ring,
            self.cms_prev_ib,
            self.cc_ring,
            self.bf_ring,
            self.nov1_ring,
        ]:
            tot += t.numel() * t.element_size()
        for ls in self.layers:
            for v in ls.__dict__.values():
                if isinstance(v, torch.Tensor):
                    tot += v.numel() * v.element_size()
        return tot


@triton.jit
def _vault_v2_decode_fw_kernel(
    S_ptr,
    Occ_ptr,
    K_ptr,
    V_ptr,
    Beta_ptr,
    CK_ptr,
    CQ_ptr,
    Q_ptr,
    Out_ptr,
    total_cells,
    OCC_ALPHA: tl.constexpr,
    D: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """
    Fused single-token 4-bank Co-Prime Delta-rule write + occupancy update + 3-subhead consensus readout.
    Replaces ~38 eager ATen/einsum/gather/scatter kernels per layer with 1 tiny 4-block Triton launch!
    """
    bg = tl.program_id(0).to(tl.int64)
    offs_v = tl.arange(0, D)
    offs_d = tl.arange(0, D)

    k = tl.load(K_ptr + bg * D + offs_d).to(tl.float32)
    v = tl.load(V_ptr + bg * D + offs_v).to(tl.float32)
    beta = tl.load(Beta_ptr + bg).to(tl.float32)
    q0 = tl.load(Q_ptr + (bg * 3 + 0) * D + offs_d).to(tl.float32)
    q1 = tl.load(Q_ptr + (bg * 3 + 1) * D + offs_d).to(tl.float32)
    q2 = tl.load(Q_ptr + (bg * 3 + 2) * D + offs_d).to(tl.float32)

    acc0 = tl.zeros([D], dtype=tl.float32)
    acc1 = tl.zeros([D], dtype=tl.float32)
    acc2 = tl.zeros([D], dtype=tl.float32)
    u0 = tl.zeros([D], dtype=tl.float32)
    u1 = tl.zeros([D], dtype=tl.float32)
    u2 = tl.zeros([D], dtype=tl.float32)
    w_sum = 0.0

    tc64 = total_cells.to(tl.int64)
    s_base = S_ptr + bg * tc64 * D * D
    o_base = Occ_ptr + bg * tc64

    for r in range(4):
        cw = tl.load(CK_ptr + bg * 4 + r).to(tl.int64)
        cq = tl.load(CQ_ptr + bg * 4 + r).to(tl.int64)
        cw_ptr = s_base + cw * D * D + offs_v[:, None] * D + offs_d[None, :]
        S_w = tl.load(cw_ptr)
        pred_v = tl.sum(S_w * k[None, :], axis=1)
        S_w_new = S_w + (beta * (v - pred_v))[:, None] * k[None, :]
        tl.store(cw_ptr, S_w_new)
        occ_w = tl.load(o_base + cw) + beta
        tl.store(o_base + cw, occ_w)

        if cq == cw:
            S_q = S_w_new
            nq = occ_w
        else:
            cq_ptr = s_base + cq * D * D + offs_v[:, None] * D + offs_d[None, :]
            S_q = tl.load(cq_ptr)
            nq = tl.load(o_base + cq)

        y0 = tl.sum(S_q * q0[None, :], axis=1)
        y1 = tl.sum(S_q * q1[None, :], axis=1)
        y2 = tl.sum(S_q * q2[None, :], axis=1)
        u0 += y0 * 0.25
        u1 += y1 * 0.25
        u2 += y2 * 0.25

        if OCC_ALPHA == 0.5:
            wr = tl.where(nq > 1e-3, tl.rsqrt(1.0 + nq), 0.0)
        elif OCC_ALPHA == 1.0:
            wr = tl.where(nq > 1e-3, 1.0 / (1.0 + nq), 0.0)
        elif OCC_ALPHA == 1.5:
            wr = tl.where(nq > 1e-3, tl.rsqrt(1.0 + nq) / (1.0 + nq), 0.0)
        else:
            wr = tl.where(nq > 1e-3, tl.exp2(-OCC_ALPHA * tl.log2(1.0 + nq)), 0.0)
        acc0 += y0 * wr
        acc1 += y1 * wr
        acc2 += y2 * wr
        w_sum += wr

    use_w = w_sum > 1e-6
    inv_w = 1.0 / tl.maximum(w_sum, 1e-6)
    o0 = tl.where(use_w, acc0 * inv_w, u0)
    o1 = tl.where(use_w, acc1 * inv_w, u1)
    o2 = tl.where(use_w, acc2 * inv_w, u2)
    if OUT_BF16:
        tl.store(Out_ptr + (bg * 3 + 0) * D + offs_v, o0.to(tl.bfloat16))
        tl.store(Out_ptr + (bg * 3 + 1) * D + offs_v, o1.to(tl.bfloat16))
        tl.store(Out_ptr + (bg * 3 + 2) * D + offs_v, o2.to(tl.bfloat16))
    else:
        tl.store(Out_ptr + (bg * 3 + 0) * D + offs_v, o0)
        tl.store(Out_ptr + (bg * 3 + 1) * D + offs_v, o1)
        tl.store(Out_ptr + (bg * 3 + 2) * D + offs_v, o2)


_ACTIVE_CUDAGRAPHS: List[weakref.ReferenceType] = []


def _cleanup_active_cudagraphs() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for ref in list(_ACTIVE_CUDAGRAPHS):
            r = ref()
            if r is not None and getattr(r, "graph", None) is not None:
                r.graph.reset()
                r.graph = None
    except Exception:
        pass


atexit.register(_cleanup_active_cudagraphs)


class _DecodeCUDAGraphRunner:
    """
    Captures a single-token recurrent `VAULTv2StreamingEngine._decode_step_static` pass inside a
    reusable `torch.cuda.CUDAGraph` (HIPGraph on ROCm 7.2), eliminating per-token Python/HIP
    kernel-launch dispatches while preserving 100% bit-exact O(1) recurrent state semantics.
    Captures hidden state `h_out (B, 1, d_model)` so both `prefill_streaming` tail tokens (`return_logits=False`)
    and `decode_step` (`return_logits=True`) share a single `static_cache` with zero copy on handoff.
    """

    def __init__(
        self,
        model: VAULTLMv2,
        batch_size: int,
        comp_dt: torch.dtype,
        step_ratio: float,
        occ_alpha: float,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
    ):
        self.runner_id = id(self)
        self.dev = model.embed_tokens.weight.device
        self.batch_size = batch_size
        self.comp_dt = comp_dt
        self.step_ratio = step_ratio
        self.occ_alpha = occ_alpha
        self.prime_banks = tuple(model.prime_banks)
        self.static_cache = VAULTv2StreamingEngine.init_cache(model, batch_size=batch_size, dtype=comp_dt)
        self.static_token_id = torch.zeros((batch_size, 1), device=self.dev, dtype=torch.long)
        self.owner_ref: Optional[weakref.ReferenceType] = None

        # 1. Warmup once on default stream so static weight packs, SwiGLUMLP._eval_cache, Triton JIT,
        #    hipBLASLt, and rocFFT initialize on the default pool rather than the graph capture pool.
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            _ = VAULTv2StreamingEngine._decode_step_static(
                model,
                self.static_token_id,
                self.static_cache,
                step_ratio=step_ratio,
                occ_alpha=occ_alpha,
                return_logits=False,
            )
        torch.cuda.current_stream(device=self.dev).synchronize()

        # 2. Retain `self.cap_stream` for the lifetime of `self.graph` (required by ROCm 7.2 rocFFT/hipGraph)
        self.cap_stream = torch.cuda.Stream(device=self.dev)
        self.cap_stream.wait_stream(torch.cuda.current_stream(device=self.dev))
        with torch.cuda.stream(self.cap_stream):
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    _ = VAULTv2StreamingEngine._decode_step_static(
                        model,
                        self.static_token_id,
                        self.static_cache,
                        step_ratio=step_ratio,
                        occ_alpha=occ_alpha,
                        return_logits=False,
                    )
        torch.cuda.current_stream(device=self.dev).wait_stream(self.cap_stream)
        torch.cuda.current_stream(device=self.dev).synchronize()

        self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.cap_stream):
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                self.static_h_out = VAULTv2StreamingEngine._decode_step_static(
                    model,
                    self.static_token_id,
                    self.static_cache,
                    step_ratio=step_ratio,
                    occ_alpha=occ_alpha,
                    return_logits=False,
                )
        torch.cuda.current_stream(device=self.dev).synchronize()
        _ACTIVE_CUDAGRAPHS.append(weakref.ref(self))

    @staticmethod
    def _copy_cache_tensors_(dst: VAULTv2StreamCache, src: VAULTv2StreamCache) -> None:
        dst.t_pos = src.t_pos
        dst.t_pos_buf.copy_(src.t_pos_buf)
        dst.x0_hist.copy_(src.x0_hist)
        dst.cms_counts_u_pad.copy_(src.cms_counts_u_pad)
        dst.cms_counts_b_pad.copy_(src.cms_counts_b_pad)
        dst.cms_ib_ring.copy_(src.cms_ib_ring)
        dst.cms_bg_ring.copy_(src.cms_bg_ring)
        dst.cms_prev_ib.copy_(src.cms_prev_ib)
        dst.cc_ring.copy_(src.cc_ring)
        dst.bf_ring.copy_(src.bf_ring)
        dst.nov1_ring.copy_(src.nov1_ring)
        for ls_d, ls_s in zip(dst.layers, src.layers):
            for k in ls_d.__dataclass_fields__:
                getattr(ls_d, k).copy_(getattr(ls_s, k))

    @staticmethod
    def _alias_cache_to_static_(cache: VAULTv2StreamCache, static_c: VAULTv2StreamCache, banks: Tuple[int, ...]) -> None:
        cache.t_pos_buf = static_c.t_pos_buf
        cache.x0_hist = static_c.x0_hist
        cache.cms_counts_u_pad = static_c.cms_counts_u_pad
        cache.cms_counts_b_pad = static_c.cms_counts_b_pad
        cache.cms_counts_u = [cache.cms_counts_u_pad[:, i, :p] for i, p in enumerate(banks)]
        cache.cms_counts_b = [cache.cms_counts_b_pad[:, i, :p] for i, p in enumerate(banks)]
        cache.cms_ib_ring = static_c.cms_ib_ring
        cache.cms_bg_ring = static_c.cms_bg_ring
        cache.cms_prev_ib = static_c.cms_prev_ib
        cache.cc_ring = static_c.cc_ring
        cache.bf_ring = static_c.bf_ring
        cache.nov1_ring = static_c.nov1_ring
        for ls_c, ls_st in zip(cache.layers, static_c.layers):
            for k in ls_c.__dataclass_fields__:
                setattr(ls_c, k, getattr(ls_st, k))

    @staticmethod
    def _detach_cache_clone_(cache: VAULTv2StreamCache, static_c: VAULTv2StreamCache, banks: Tuple[int, ...]) -> None:
        cache.t_pos_buf = static_c.t_pos_buf.clone()
        cache.x0_hist = static_c.x0_hist.clone()
        cache.cms_counts_u_pad = static_c.cms_counts_u_pad.clone()
        cache.cms_counts_b_pad = static_c.cms_counts_b_pad.clone()
        cache.cms_counts_u = [cache.cms_counts_u_pad[:, i, :p] for i, p in enumerate(banks)]
        cache.cms_counts_b = [cache.cms_counts_b_pad[:, i, :p] for i, p in enumerate(banks)]
        cache.cms_ib_ring = static_c.cms_ib_ring.clone()
        cache.cms_bg_ring = static_c.cms_bg_ring.clone()
        cache.cms_prev_ib = static_c.cms_prev_ib.clone()
        cache.cc_ring = static_c.cc_ring.clone()
        cache.bf_ring = static_c.bf_ring.clone()
        cache.nov1_ring = static_c.nov1_ring.clone()
        for ls_c, ls_st in zip(cache.layers, static_c.layers):
            for k in ls_c.__dataclass_fields__:
                setattr(ls_c, k, getattr(ls_st, k).clone())

    def release_owner(self, banks: Tuple[int, ...]) -> None:
        torch.cuda.current_stream(device=self.dev).synchronize()
        cur_owner = self.owner_ref() if self.owner_ref is not None else None
        if cur_owner is not None and cur_owner._cg_runner_id == self.runner_id:
            self._detach_cache_clone_(cur_owner, self.static_cache, banks)
            cur_owner._cg_runner_id = 0
        self.owner_ref = None
        if self.graph is not None:
            self.graph.reset()
            self.graph = None

    def __del__(self) -> None:
        try:
            if getattr(self, "graph", None) is not None and torch is not None and torch.cuda.is_available():
                torch.cuda.current_stream(device=self.dev).synchronize()
                self.graph.reset()
                self.graph = None
        except Exception:
            pass

    def step(self, token_id: torch.Tensor, cache: VAULTv2StreamCache, banks: Tuple[int, ...]) -> torch.Tensor:
        cur_owner = self.owner_ref() if self.owner_ref is not None else None
        if cur_owner is not cache or cache._cg_runner_id != self.runner_id:
            if cur_owner is not None and cur_owner._cg_runner_id == self.runner_id:
                self._detach_cache_clone_(cur_owner, self.static_cache, banks)
                cur_owner._cg_runner_id = 0
            # Ensure host t_pos and device t_pos_buf match before copying into static_cache
            cache.t_pos_buf.fill_(cache.t_pos)
            self._copy_cache_tensors_(self.static_cache, cache)
            self._alias_cache_to_static_(cache, self.static_cache, banks)
            cache._cg_runner_id = self.runner_id
            self.owner_ref = weakref.ref(cache)

        self.static_token_id.copy_(token_id)
        self.graph.replay()
        # ROCm 7.2 RDNA4 hipGraphLaunch requires strict single-in-flight serialization per hipGraphExec_t
        torch.cuda.current_stream(device=self.dev).synchronize()
        cache.t_pos += 1
        self.static_cache.t_pos = cache.t_pos
        return self.static_h_out


class VAULTv2StreamingEngine:
    """
    Bit-exact O(1)-state Streaming Prefill + True O(1) Autoregressive Decode Engine for `VAULTLMv2` (`v2_2_mode=True`).
    """

    @staticmethod
    def _infer_compute_dtype(model: VAULTLMv2, device: torch.device) -> torch.dtype:
        if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return model.embed_tokens.weight.dtype

    @classmethod
    @torch.no_grad()
    def init_cache(cls, model: VAULTLMv2, batch_size: int = 1, dtype: Optional[torch.dtype] = None) -> VAULTv2StreamCache:
        assert model.v2_2_mode, "VAULTv2StreamingEngine targets VAULTLMv2 (v2.2 / v2.3 / v2.4) with v2_2_mode=True"
        dev = model.embed_tokens.weight.device
        comp_dt = dtype if dtype is not None else cls._infer_compute_dtype(model, dev)
        B = batch_size
        d_m = model.d_model
        G = model.n_vault_groups
        D = model.head_dim
        total_cells = sum(model.prime_banks)
        qkv_c = model.n_q_heads * D + 2 * G * D
        b0 = model.layers[0].attn.salience_binder

        cms_u_pad = torch.zeros((B, 3, 528), device=dev, dtype=torch.int32)
        cms_b_pad = torch.zeros((B, 3, 528), device=dev, dtype=torch.int32)
        cms_u = [cms_u_pad[:, i, :p] for i, p in enumerate(b0.BANKS)]
        cms_b = [cms_b_pad[:, i, :p] for i, p in enumerate(b0.BANKS)]

        layers: List[LayerStreamState] = []
        for _ in range(model.n_layers):
            layers.append(
                LayerStreamState(
                    S_state=torch.zeros((B, G, total_cells, D, D), device=dev, dtype=torch.float32),
                    occ_state=torch.zeros((B, G, total_cells), device=dev, dtype=torch.float32),
                    v_cumsum=torch.zeros((B, G, 1, D), device=dev, dtype=torch.float32),
                    qkv_conv_state=torch.zeros((B, 3, qkv_c), device=dev, dtype=comp_dt),
                    prev_ib_rms_p1=torch.zeros((B, 1, 3), device=dev, dtype=comp_dt),
                    prev_sal_iso_p1=torch.zeros((B, 1, 128), device=dev, dtype=comp_dt),
                    prev_ib_iso_p1=torch.zeros((B, 1, 3), device=dev, dtype=comp_dt),
                    anchor_conv_state=torch.zeros((B, 5, 256), device=dev, dtype=comp_dt),
                    span_conv_state=torch.zeros((B, 5, d_m), device=dev, dtype=comp_dt),
                    a_raw_ring=torch.zeros((B, 7, 2, 2, D), device=dev, dtype=torch.float32),
                    a_left_ring=torch.zeros((B, 7, 2, 2, D), device=dev, dtype=torch.float32),
                    a_ctx_win=torch.zeros((B, 2, 128, 128), device=dev, dtype=comp_dt),
                    sal_win=torch.full((B, 2, 128), -1e4, device=dev, dtype=torch.float32),
                    prev_x0n_s1=torch.zeros((B, 1, 393), device=dev, dtype=comp_dt),
                    prev_x0n_s2=torch.zeros((B, 2, 128), device=dev, dtype=comp_dt),
                    r_all_prev=torch.zeros((B, 1, 3, D), device=dev, dtype=torch.float32),
                    u_ind_prev=torch.zeros((B, 1, 3, D), device=dev, dtype=torch.float32),
                    ind_delay_pk=torch.zeros((B, 3, 64, D), device=dev, dtype=torch.float32),
                    ind_delay_v=torch.zeros((B, 3, 64, D), device=dev, dtype=comp_dt),
                    ind_delay_beta=torch.zeros((B, 3, 64), device=dev, dtype=torch.float32),
                    ind_delay_ck=torch.zeros((B, 3, 4, 64), device=dev, dtype=torch.long),
                    ck_ind_first=torch.zeros((B, 3, 4, 1), device=dev, dtype=torch.long),
                    local_k_win=torch.zeros((B, G, 128, D), device=dev, dtype=comp_dt),
                    local_v_win=torch.zeros((B, G, 128, D), device=dev, dtype=comp_dt),
                )
            )

        return VAULTv2StreamCache(
            batch_size=B,
            t_pos=0,
            t_pos_buf=torch.zeros((1,), device=dev, dtype=torch.long),
            comp_dt=comp_dt,
            device=dev,
            x0_hist=torch.zeros((B, 1185, d_m), device=dev, dtype=comp_dt),
            cms_counts_u_pad=cms_u_pad,
            cms_counts_b_pad=cms_b_pad,
            cms_counts_u=cms_u,
            cms_counts_b=cms_b,
            cms_ib_ring=torch.zeros((B, 1057, 3), device=dev, dtype=torch.long),
            cms_bg_ring=torch.zeros((B, 1057, 3), device=dev, dtype=torch.long),
            cms_prev_ib=torch.zeros((B, 3), device=dev, dtype=torch.long),
            cc_ring=torch.zeros((B, 7, 1), device=dev, dtype=torch.float32),
            bf_ring=torch.zeros((B, 7, 1), device=dev, dtype=torch.float32),
            nov1_ring=torch.zeros((B, 1, 3), device=dev, dtype=torch.float32),
            layers=layers,
        )

    @staticmethod
    @torch.no_grad()
    def _sync_cms_decode_state_from_hist(
        cache: VAULTv2StreamCache, binder: SalienceEntityBinder, x0_tail: torch.Tensor, T_end: int
    ) -> None:
        """
        Synchronizes the exact O(1) incremental Windowed CMS ring & counters at chunk boundary `T_end`
        from `x0_tail` (which holds the last `min(T_end, 1058)` tokens ending at `T_end - 1`).
        """
        B, L_h, d_model = x0_tail.shape
        dev = x0_tail.device
        t_start = T_end - L_h  # global index of x0_tail[:, 0]

        x0_f = x0_tail.detach().float()
        cm = binder.x0_cone_mean.detach().float().view(1, 1, d_model)
        x0_c = F.normalize(x0_f - cm, p=2, dim=-1, eps=1e-6)
        proj = torch.matmul(x0_c, binder.cms_proj.detach().float().T)

        cache.cms_counts_u_pad.zero_()
        cache.cms_counts_b_pad.zero_()
        cache.cms_ib_ring.zero_()
        cache.cms_bg_ring.zero_()

        hi_prev = max(0, T_end - 33)
        lo_prev = max(0, T_end - 1057)

        off = 0
        for b_i, p in enumerate(binder.BANKS):
            ib = proj[..., off : off + p].argmax(dim=-1)  # (B, L_h)
            ib_p1 = F.pad(ib[:, :-1], (1, 0), value=0)
            bg = (ib_p1 * 131 + ib) % p                   # (B, L_h)
            cache.cms_prev_ib[:, b_i] = ib[:, -1]

            keep_n = min(L_h, 1057)
            k0 = L_h - keep_n
            g_idx = (torch.arange(t_start + k0, T_end, device=dev) % 1057)
            cache.cms_ib_ring[:, g_idx, b_i] = ib[:, k0:]
            cache.cms_bg_ring[:, g_idx, b_i] = bg[:, k0:]

            if hi_prev > lo_prev:
                loc_lo = lo_prev - t_start
                loc_hi = hi_prev - t_start
                ib_win = ib[:, loc_lo:loc_hi]
                ohu = F.one_hot(ib_win, num_classes=p).to(torch.int32).sum(dim=1)
                cache.cms_counts_u_pad[:, b_i, :p].copy_(ohu)

                bg_start = max(loc_lo, 1 - t_start)
                if loc_hi > bg_start:
                    bg_win = bg[:, bg_start:loc_hi]
                    ohb = F.one_hot(bg_win, num_classes=p).to(torch.int32).sum(dim=1)
                    cache.cms_counts_b_pad[:, b_i, :p].copy_(ohb)
            off += p

    @classmethod
    @torch.no_grad()
    def _prefill_aligned_segment(
        cls,
        model: VAULTLMv2,
        seg_ids: torch.Tensor,
        cache: VAULTv2StreamCache,
        step_ratio: float = 1.0,
        populate_decode_rings: bool = True,
        occ_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Streams an aligned segment `seg_ids (B, S)` where `cache.t_pos % 64 == 0` and `S % 64 == 0`.
        Returns hidden states `norm_f(x)` of shape `(B, S, d_model)`.
        """
        B, S = seg_ids.shape
        T_0 = cache.t_pos
        C = model.chunk_size
        assert T_0 % C == 0 and S % C == 0 and S >= C
        dev = seg_ids.device
        comp_dt = cache.comp_dt

        x = model.embed_tokens(seg_ids)
        if x.dtype != comp_dt:
            x = x.to(comp_dt)
        x0_seg = model.compute_x0_norm(x)  # (B, S, d_model)

        # Prefix needed: 1057 (CMS) + 128 (SalienceEntityBinder 2-chunk aligned prefix) = 1185 tokens
        if T_0 > 0:
            h_len = min(T_0, 1185)
            idx_h = torch.arange(T_0 - h_len, T_0, device=dev) % 1185
            x0_ext = torch.cat([cache.x0_hist.index_select(1, idx_h), x0_seg], dim=1)
        else:
            x0_ext = x0_seg
        b0 = model.layers[0].attn.salience_binder
        cms_ext = compute_v2_2_shared_aux(x0_ext, b0)

        # Slice exact 128-token-aligned window (`H_sb in {0, 64, 128}`) for SalienceEntityBinder
        H_sb = min(T_0, 2 * C)
        L_sb = H_sb + S
        sh_sb = {
            "shape": (B, L_sb),
            "x0_iso": cms_ext["x0_iso"][:, -L_sb:],
            "nov_stem": cms_ext["nov_stem"][:, -L_sb:],
            "k0_steps": tuple(t[:, -L_sb:] for t in cms_ext["k0_steps"]),
            "k1_steps": tuple(t[:, -L_sb:] for t in cms_ext["k1_steps"]),
            "aux_f": cms_ext["aux_f"][:, -L_sb:],
            "rep_pen": cms_ext["rep_pen"][:, :, -L_sb:],
        }
        x0_sb = x0_ext[:, -L_sb:]
        nov4_seg = cms_ext["nov4"][:, :, -S:]
        inv_counts_seg = (
            1.0 / torch.arange(T_0 + 1, T_0 + S + 1, device=dev, dtype=torch.float32)
        ).view(1, 1, S, 1)

        H_x0n = min(T_0, 2)
        x0_for_lex = x0_ext[:, -(H_x0n + S):]  # [T_0 - H_x0n .. T_0 + S - 1]

        for l_idx, blk in enumerate(model.layers):
            attn: VAULTv2Attention = blk.attn
            ls = cache.layers[l_idx]
            h_in = blk.norm1(x)

            n_keep_g = attn.n_keep_g
            n_keep_h = n_keep_g * attn.heads_per_group
            G = attn.n_vault_groups
            D = attn.head_dim
            q_dim = attn.n_q_heads * D
            kv_dim = attn.kv_dim

            w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = attn._get_v2_2_packed_weights(h_in)
            x_proj = F.linear(h_in, w_x_all)
            anchor = x_proj[..., :kv_dim]
            q_raw = x_proj[..., kv_dim : kv_dim + q_dim]
            k_raw = x_proj[..., kv_dim + q_dim : 2 * kv_dim + q_dim]
            v_raw = x_proj[..., 2 * kv_dim + q_dim : 3 * kv_dim + q_dim]
            beta_raw = x_proj[..., 3 * kv_dim + q_dim : 3 * kv_dim + q_dim + G] + attn.beta_proj.bias
            gate_raw = x_proj[..., 3 * kv_dim + q_dim + G :].contiguous()

            anchor_q = anchor.view(B, S, G, 1, D).expand(B, S, G, attn.heads_per_group, D).reshape(B, S, q_dim)
            qkv_in = torch.cat([anchor_q + q_raw, anchor + k_raw, v_raw], dim=-1)
            qkv_cat = torch.cat([ls.qkv_conv_state, qkv_in], dim=1) if T_0 > 0 else qkv_in
            qkv_out = causal_depthwise_conv1d(qkv_cat, qkv_w, qkv_b, apply_silu=True)
            if T_0 > 0:
                qkv_out = qkv_out[:, 3:, :]
            ls.qkv_conv_state.copy_(qkv_in[:, -3:, :])

            q_syn_raw = qkv_out[..., :q_dim].view(B, S, attn.n_q_heads, D).transpose(1, 2)
            k_syn_raw = qkv_out[..., q_dim : q_dim + kv_dim].view(B, S, G, D).transpose(1, 2)
            v = qkv_out[..., q_dim + kv_dim :].contiguous().view(B, S, G, D).transpose(1, 2)
            dtype = v.dtype

            qs_f, ks_f = q_syn_raw.float(), k_syn_raw.float()
            qs_rms = torch.rsqrt(torch.mean(qs_f * qs_f, dim=-1, keepdim=True) + 1e-6)
            ks_rms = torch.rsqrt(torch.mean(ks_f * ks_f, dim=-1, keepdim=True) + 1e-6)
            q_norm = (qs_f * qs_rms).to(dtype)
            k_norm = (ks_f * ks_rms).to(dtype)

            h_f = h_in.float()
            x_rms = (h_f * torch.rsqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)

            sb = attn.salience_binder
            sal_c = 2 * sb.ind_dim + 1 + 2 * sb.sal_hidden  # 513
            qsh_c = attn.n_ind_groups * (attn.heads_per_group - 1) * D  # 384
            vind_c = attn.n_ind_groups * D  # 192
            ib_c = attn.n_ind_groups  # 3

            iso_proj_sb = F.linear(sh_sb["x0_iso"], w_iso_all)  # (B, L_sb, 1095)
            pre_sal_sb = iso_proj_sb[..., :sal_c]
            iso_proj_seg = iso_proj_sb[:, -S:]
            qsh_proj = iso_proj_seg[..., sal_c : sal_c + qsh_c]
            vind_proj = iso_proj_seg[..., sal_c + qsh_c : sal_c + qsh_c + vind_c]
            ib_iso_src_seg = iso_proj_seg[..., sal_c + qsh_c + vind_c : sal_c + qsh_c + vind_c + ib_c]
            if T_0 > 0:
                ib_iso_p1 = torch.cat([ls.prev_ib_iso_p1, ib_iso_src_seg[:, :-1, :]], dim=1)
            else:
                ib_iso_p1 = _shift_right_zero(ib_iso_src_seg, 1)
            ls.prev_ib_iso_p1.copy_(ib_iso_src_seg[:, -1:, :])
            ib_iso_0 = iso_proj_seg[..., sal_c + qsh_c + vind_c + ib_c :]

            ent_u_sb, ent_r_sb = sb(x_rms, x0_sb, cms_aux=sh_sb, pre_x0_iso_sal=pre_sal_sb)
            ent_u = ent_u_sb[:, -S:]
            ent_r = ent_r_sb[:, -S:]

            m_shared = (0.20 / 8.0) * torch.tanh(attn.ind_shared_proj.float())
            ent_ur = torch.stack([ent_u, ent_r], dim=3)
            eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
            ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, m_shared)
            ur_ent_rms = ur_ent * torch.rsqrt(ur_ent.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            ur_comb = ur_ent_rms[:, :, 0:1, :, :] + ur_ent_rms[:, :, 1:2, :, :]
            ur_ent_3 = torch.cat(
                [ur_ent_rms, ur_comb * torch.rsqrt(ur_comb.pow(2).mean(dim=-1, keepdim=True) + 1e-6)], dim=2
            )

            w_ir = attn.ind_read_proj.weight
            lg2_c = attn.lex_router_g2.weight.size(0)  # 256
            lg3_c = attn.lex_router_g3.weight.size(0)  # 128
            ir_c = w_ir.size(0)                        # 9
            s0_c = lg2_c + lg3_c + ir_c                # 393

            x0n_proj_ext = F.linear(x0_for_lex, w_x0n_all)  # (B, H_x0n + S, 914)
            s01_ext = x0n_proj_ext[..., :s0_c] + _shift_right_zero(x0n_proj_ext[..., s0_c : 2 * s0_c], 1)
            s2_ext = _shift_right_zero(x0n_proj_ext[..., 2 * s0_c :], 2)
            s01 = s01_ext[:, -S:]
            s2_lg3 = s2_ext[:, -S:]

            lex_12 = s01[..., :lg2_c].float().view(B, S, 2, 2, D)
            lex_3 = (s01[..., lg2_c : lg2_c + lg3_c] + s2_lg3).float().view(B, S, 1, 2, D)
            ir_x0 = s01[..., lg2_c + lg3_c :] + attn.ind_read_proj.bias
            lex_both = torch.cat([lex_12, lex_3], dim=2)
            lex_zm = lex_both - lex_both.mean(dim=-1, keepdim=True)
            lex_rms = lex_zm * torch.rsqrt(lex_zm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

            ur_ind_raw = fft_circular_bind(ur_ent_3, lex_rms)
            u_ind_raw = ur_ind_raw[:, :, :, 0, :]
            r_ind_raw = ur_ind_raw[:, :, :, 1, :]
            u_ind = u_ind_raw * torch.rsqrt(u_ind_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            r_all = F.normalize(r_ind_raw, p=2, dim=-1, eps=1e-6)

            if T_0 > 0:
                r_all_prev = torch.cat([ls.r_all_prev, r_all[:, :-1, :, :]], dim=1)
                u_ind_prev = torch.cat([ls.u_ind_prev, u_ind[:, :-1, :, :]], dim=1)
            else:
                r_all_prev = F.pad(r_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
                u_ind_prev = F.pad(u_ind[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            ls.r_all_prev.copy_(r_all[:, -1:, :, :])
            ls.u_ind_prev.copy_(u_ind[:, -1:, :, :])

            shared_group_q = attn.prime_banks != (17, 19, 23, 29)
            lex_q_3 = r_all.transpose(1, 2)
            lex_k_3 = r_all_prev.transpose(1, 2)
            kr_f = torch.cat([ks_f[:, :n_keep_g], lex_k_3], dim=1)
            inv_sqrt_d = 1.0 / math.sqrt(D)
            kr_unit = (kr_f * (torch.rsqrt(kr_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)

            if shared_group_q:
                qr_4_f = torch.cat([qs_f[:, :n_keep_h].mean(dim=1, keepdim=True), lex_q_3], dim=1)
                qr_unit = (qr_4_f * (torch.rsqrt(qr_4_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)
                cq_4, cell_k_raw, _, _ = attn._route_cells_v2(qr_unit, kr_unit, step_ratio)
                cell_q = cq_4.repeat_interleave(attn.heads_per_group, dim=1)
            else:
                qr_6_f = torch.cat([qs_f[:, :n_keep_h], lex_q_3], dim=1)
                qr_unit = (qr_6_f * (torch.rsqrt(qr_6_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)
                cq_6, cell_k_raw, _, _ = attn._route_cells_v2(qr_unit, kr_unit, step_ratio)
                cell_q = torch.cat(
                    [cq_6[:, :n_keep_h], cq_6[:, n_keep_h:].repeat_interleave(attn.heads_per_group, dim=1)], dim=1
                )

            k_ind_3 = u_ind_prev.transpose(1, 2)
            q_sh_rest = 0.04 * torch.tanh(qsh_proj.float()).view(
                B, S, attn.n_ind_groups, attn.heads_per_group - 1, D
            )
            q_sh_zero = torch.zeros((B, S, attn.n_ind_groups, 1, D), device=dev, dtype=torch.float32)
            q_ind_9 = (
                (u_ind.unsqueeze(3) + torch.cat([q_sh_zero, q_sh_rest], dim=3))
                .reshape(B, S, attn.n_ind_groups * attn.heads_per_group, D)
                .transpose(1, 2)
            )

            bq = torch.clamp(attn.beta_sem_q.float(), 1.2, 3.5)
            bk = torch.clamp(attn.beta_sem_k.float(), 1.2, 3.5)
            pq_keep = F.normalize(
                torch.exp(bq[:, :n_keep_h] * torch.clamp(qs_f[:, :n_keep_h] * qs_rms[:, :n_keep_h], -3.0, 3.0)),
                p=2, dim=-1,
            )
            pk_keep = F.normalize(
                torch.exp(bk[:, :n_keep_g] * torch.clamp(ks_f[:, :n_keep_g] * ks_rms[:, :n_keep_g], -3.0, 3.0)),
                p=2, dim=-1,
            )
            zq_ind = q_ind_9 - q_ind_9.mean(dim=-1, keepdim=True)
            zk_ind = k_ind_3 - k_ind_3.mean(dim=-1, keepdim=True)
            phi_q = torch.cat([pq_keep, F.normalize(zq_ind, p=2, dim=-1)], dim=1).to(dtype)
            pk_ind = F.normalize(zk_ind, p=2, dim=-1)  # float32

            v_ind = vind_proj.contiguous().view(B, S, attn.n_ind_groups, D).transpose(1, 2)
            v_keep_w = v[:, :n_keep_g]
            v_ind_raw = 0.05 * v[:, n_keep_g:] + v_ind
            beta_keep_f = 0.96 * torch.sigmoid(beta_raw.float()).transpose(1, 2)[:, :n_keep_g, :]

            x_rms_zm = x_rms - x_rms.mean(dim=-1, keepdim=True)
            ib_rms_both = F.linear(x_rms_zm, w_ib_rms)
            if T_0 > 0:
                ib_rms_p1 = torch.cat([ls.prev_ib_rms_p1, ib_rms_both[:, :-1, :ib_c]], dim=1)
            else:
                ib_rms_p1 = _shift_right_zero(ib_rms_both[..., :ib_c], 1)
            ls.prev_ib_rms_p1.copy_(ib_rms_both[:, -1:, :ib_c])

            ib_lin = ib_rms_p1 + ib_rms_both[..., ib_c:] + ib_iso_p1 + ib_iso_0
            z_beta = 3.4 * torch.tanh(ib_lin.float() / 3.4)
            g_beta = torch.sigmoid(z_beta).transpose(1, 2)
            beta_ind_raw_f = 0.00005 + nov4_seg * (0.140 * g_beta)

            # W = 64 induction delay buffer across segment boundary
            W = C
            ck_ind_src = cell_k_raw[:, n_keep_g:]
            if T_0 == 0:
                ls.ck_ind_first.copy_(ck_ind_src[:, :, :, :1])
                if S > W:
                    pk_ind_w = F.pad(pk_ind[:, :, :-W, :], (0, 0, W, 0))
                    v_ind_w = F.pad(v_ind_raw[:, :, :-W, :], (0, 0, W, 0))
                    beta_ind_f = F.pad(beta_ind_raw_f[:, :, :-W], (W, 0), value=0.0)
                    ck_pad = ls.ck_ind_first.expand(-1, -1, -1, W)
                    ck_ind_w = torch.cat([ck_pad, ck_ind_src[:, :, :, :-W]], dim=3)
                else:
                    pk_ind_w = torch.zeros_like(pk_ind)
                    v_ind_w = torch.zeros_like(v_ind_raw)
                    beta_ind_f = torch.zeros_like(beta_ind_raw_f)
                    ck_ind_w = ck_ind_src
            else:
                pk_ind_w = torch.cat([ls.ind_delay_pk, pk_ind], dim=2)[:, :, :-W, :]
                v_ind_w = torch.cat([ls.ind_delay_v, v_ind_raw], dim=2)[:, :, :-W, :]
                beta_ind_f = torch.cat([ls.ind_delay_beta, beta_ind_raw_f], dim=2)[:, :, :-W]
                ck_ind_w = torch.cat([ls.ind_delay_ck, ck_ind_src], dim=3)[:, :, :, :-W]

            ls.ind_delay_pk.copy_(pk_ind[:, :, -W:, :])
            ls.ind_delay_v.copy_(v_ind_raw[:, :, -W:, :])
            ls.ind_delay_beta.copy_(beta_ind_raw_f[:, :, -W:])
            ls.ind_delay_ck.copy_(ck_ind_src[:, :, :, -W:])

            phi_k = torch.cat([pk_keep, pk_ind_w], dim=1).to(dtype)
            v_vault = torch.cat([v_keep_w, v_ind_w], dim=1)
            beta_write = torch.cat([beta_keep_f, beta_ind_f], dim=1).to(dtype)
            cell_k = torch.cat([cell_k_raw[:, :n_keep_g], ck_ind_w], dim=1)

            y_vault = stateful_flash_vault_v2_chunkwise(
                phi_q,
                phi_k,
                v_vault,
                beta_write,
                cell_q,
                cell_k,
                ls.S_state,
                ls.occ_state,
                chunk_size=C,
                v2_1_consensus=True,
                v2_2_beta_occ=True,
                prime_banks=attn.prime_banks,
                shared_group_cell_q=shared_group_q,
                occ_alpha=occ_alpha,
            )

            # Chunk-local RoPE syntax with 64-token previous-chunk carry
            if T_0 == 0:
                y_syntax = attn._chunk_local_rope_syntax(q_norm, k_norm, v)
            else:
                q_pad = torch.cat([torch.zeros_like(q_norm[:, :, :C, :]), q_norm], dim=2)
                k_pad = torch.cat([ls.local_k_win[:, :, C:, :], k_norm], dim=2)
                v_pad = torch.cat([ls.local_v_win[:, :, C:, :], v], dim=2)
                y_syntax = attn._chunk_local_rope_syntax(q_pad, k_pad, v_pad)[:, :, C:, :]
            ls.local_k_win[:, :, C:, :].copy_(k_norm[:, :, -C:, :])
            ls.local_v_win[:, :, C:, :].copy_(v[:, :, -C:, :])

            # GTRM running macro-summary
            v_cum = ls.v_cumsum + _chunked_cumsum_dim2(v.float())
            ls.v_cumsum.copy_(v_cum[:, :, -1:, :])
            v_mean = (v_cum * inv_counts_seg).to(dtype)
            v_macro = attn.gtrm_proj(v_mean).repeat_interleave(attn.heads_per_group, dim=1)

            y_keep = (
                attn.vault_gate[:, :n_keep_h].to(dtype) * y_vault[:, :n_keep_h]
                + attn.syntax_gate[:, :n_keep_h].to(dtype) * y_syntax[:, :n_keep_h]
                + attn.gtrm_gate[:, :n_keep_h].to(dtype) * v_macro[:, :n_keep_h]
            )
            y_base_ind = (
                attn.syntax_gate[:, n_keep_h:].to(dtype) * y_syntax[:, n_keep_h:]
                + attn.gtrm_gate[:, n_keep_h:].to(dtype) * v_macro[:, n_keep_h:]
            )
            y_norm_12 = attn.head_norm(torch.cat([y_keep, y_base_ind], dim=1))
            y_base_all = y_norm_12.transpose(1, 2).reshape(B, S, attn.n_q_heads * D)
            gate = torch.sigmoid(gate_raw).to(y_base_all.dtype)
            out_base = attn.o_proj(y_base_all * gate)

            yv_ind_f = y_vault[:, n_keep_h:].transpose(1, 2).float()
            yv_soft_f = yv_ind_f * torch.rsqrt(yv_ind_f.pow(2).mean(dim=-1, keepdim=True) + 2.5e-5)
            ir_lin = ir_x0 + F.linear(x_rms, w_ir[:, model.d_model : 2 * model.d_model])
            g_read_f = 2.60 * torch.sigmoid(ir_lin.float()).unsqueeze(-1)
            yv_grp = (
                (g_read_f * yv_soft_f)
                .view(B, S, attn.n_ind_groups, attn.heads_per_group, D)
                .mean(dim=3)
                .reshape(B, S, attn.n_ind_groups * D)
                .to(dtype)
            )
            attn_out = out_base + attn.ind_out_proj(yv_grp)

            x = x + attn_out
            x = x + blk.mlp(blk.norm2(x))

            if populate_decode_rings:
                k_dim = 2 * sb.ind_dim     # 256
                sh_dim = sb.sal_hidden     # 128
                L_tail = min(L_sb, C + 5)  # 69 when T_end > 64, 64 when T_end == 64
                ps_t = pre_sal_sb[:, -L_tail:]
                x0_iso_t = sh_sb["x0_iso"][:, -L_tail:]
                nov_s_t = sh_sb["nov_stem"][:, -L_tail:]
                aux_t = sh_sb["aux_f"][:, -L_tail:]
                rep_t = sh_sb["rep_pen"][:, :, -L_tail:]
                k0_t = tuple(kt[:, -C:] for kt in sh_sb["k0_steps"])
                k1_t = tuple(kt[:, -C:] for kt in sh_sb["k1_steps"])

                a_raw_t = ps_t[..., :k_dim].float()
                g_tok_t = torch.sigmoid((ps_t[..., k_dim : k_dim + 1] + sb.ent_tok_gate.bias).float()) * nov_s_t
                ga_in = (g_tok_t * a_raw_t).to(dtype)
                a_stem_t = causal_depthwise_conv1d(ga_in, sb.anchor_ent_conv.weight, None, apply_silu=False).float()
                a_left_t = a_raw_t * (0.65 + 0.85 * torch.tanh(1.5 * a_stem_t)) + 0.45 * a_stem_t

                ls.prev_sal_iso_p1.copy_(ps_t[:, -1:, k_dim + 1 : k_dim + 1 + sh_dim])
                ls.anchor_conv_state.copy_(ga_in[:, -5:, :])
                ls.span_conv_state.copy_(x0_iso_t[:, -5:, :])

                a_raw_c = a_raw_t[:, -C:].view(B, C, 2, 2, D)
                a_left_c = a_left_t[:, -C:].view(B, C, 2, 2, D)
                ls.a_raw_ring.copy_(a_raw_c[:, -7:])
                ls.a_left_ring.copy_(a_left_c[:, -7:])

                e0_pad = F.pad(a_raw_c[:, :, 0], (0, 0, 0, 0, 0, 7))
                e1_pad = F.pad(a_raw_c[:, :, 1], (0, 0, 0, 0, 0, 7))
                tail0 = (
                    1.40 * k0_t[0] * torch.roll(e0_pad[:, 1 : C + 1], shifts=3, dims=-1)
                    + 1.50 * k0_t[1] * torch.roll(e0_pad[:, 2 : C + 2], shifts=7, dims=-1)
                    + 1.60 * k0_t[2] * torch.roll(e0_pad[:, 3 : C + 3], shifts=13, dims=-1)
                )
                tail1 = (
                    1.10 * k1_t[0] * torch.roll(e1_pad[:, 1 : C + 1], shifts=3, dims=-1)
                    + 1.25 * k1_t[1] * torch.roll(e1_pad[:, 2 : C + 2], shifts=7, dims=-1)
                    + 1.45 * k1_t[2] * torch.roll(e1_pad[:, 3 : C + 3], shifts=13, dims=-1)
                    + 1.75 * k1_t[3] * torch.roll(e1_pad[:, 4 : C + 4], shifts=23, dims=-1)
                    + 2.10 * k1_t[4] * torch.roll(e1_pad[:, 5 : C + 5], shifts=37, dims=-1)
                    + 2.40 * k1_t[5] * torch.roll(e1_pad[:, 6 : C + 6], shifts=47, dims=-1)
                )
                a_ctx_c = (
                    fft_bind_with_id(a_left_c, torch.stack([tail0, tail1], dim=2), id_weight=sb._id_weights_2g)
                    .reshape(B, C, 2, 2 * D)
                    .to(dtype)
                    .permute(0, 2, 1, 3)
                )

                w_si = sb.sal_in.weight
                sal_iso_p1_t = _shift_right_zero(ps_t[..., k_dim + 1 : k_dim + 1 + sh_dim], 1)[:, -C:]
                sal_iso_0_t = ps_t[:, -C:, k_dim + 1 + sh_dim :]
                x0_span_c = causal_depthwise_conv1d(x0_iso_t, sb.sal_span_conv.weight, None, apply_silu=False)[:, -C:]
                sal_h_c = (
                    F.linear(x0_span_c, w_si[:, : model.d_model], sb.sal_in.bias)
                    + sal_iso_p1_t
                    + sal_iso_0_t
                    + F.linear(aux_t[:, -C:], w_si[:, 3 * model.d_model :])
                )
                raw_sal_c = sb.sal_out(F.silu(sal_h_c)).float().transpose(1, 2)
                sal_c_out = 4.0 * torch.tanh(raw_sal_c / 4.0) - rep_t[:, :, -C:]

                ls.a_ctx_win[:, :, C:, :].copy_(a_ctx_c)
                ls.sal_win[:, :, C:].copy_(sal_c_out)
                ls.prev_x0n_s1.copy_(x0n_proj_ext[:, -1:, s0_c : 2 * s0_c])
                ls.prev_x0n_s2.copy_(x0n_proj_ext[:, -2:, 2 * s0_c :])

        T_end = T_0 + S
        cache.t_pos = T_end
        cache.t_pos_buf.fill_(T_end)
        keep_h = min(x0_ext.size(1), 1185)
        idx_end = torch.arange(T_end - keep_h, T_end, device=dev) % 1185
        cache.x0_hist.index_copy_(1, idx_end, x0_ext[:, -keep_h:, :].to(cache.x0_hist.dtype))

        if populate_decode_rings:
            cone_cos_ext, _, bf_ext, _, _ = cms_ext["cms"]
            cache.cc_ring.copy_(cone_cos_ext[:, -7:, :])
            cache.bf_ring.copy_(bf_ext[:, -7:, :])
            nov1_ext = torch.sigmoid(12.0 * (0.62 - bf_ext[:, -3:, :])).pow(2).transpose(1, 2)
            cache.nov1_ring.copy_(nov1_ext)
            cls._sync_cms_decode_state_from_hist(cache, b0, x0_ext[:, -min(x0_ext.size(1), 1058):], T_end)

        return model.norm_f(x)

    @staticmethod
    @torch.no_grad()
    def _get_decode_static_pack(model: VAULTLMv2, comp_dt: torch.dtype, dev: torch.device) -> Dict:
        pack_key = (dev, comp_dt, tuple(model.prime_banks))
        cached = getattr(model, "_decode_static_pack", None)
        if cached is not None and cached[0] == pack_key:
            return cached[1]

        d_m = model.d_model
        D = model.head_dim
        b0 = model.layers[0].attn.salience_binder
        cm = b0.x0_cone_mean.detach().float().view(1, 1, d_m).contiguous()
        cone_norm = cm.norm(dim=-1, keepdim=True)
        cone_strength = (cone_norm / ((d_m ** 0.5) * 0.30)).clamp(max=1.0).contiguous()
        cone_u = (cm / cone_norm.clamp(min=1e-6)).contiguous()
        cms_proj_t = b0.cms_proj.detach().float().T.contiguous()

        max_cms = max(b0.BANKS)  # 521
        cms_gather = []
        cms_mask = torch.zeros((1, 3, max_cms), device=dev, dtype=torch.float32)
        off_c = 0
        for r_i, p_c in enumerate(b0.BANKS):
            cols = [off_c + min(j, p_c - 1) for j in range(max_cms)]
            cms_gather.extend(cols)
            if p_c < max_cms:
                cms_mask[0, r_i, p_c:] = -1e9
            off_c += p_c
        cms_gather_idx = torch.tensor(cms_gather, device=dev, dtype=torch.long)
        cms_banks_p = torch.tensor([[b0.BANKS[0], b0.BANKS[1], b0.BANKS[2]]], device=dev, dtype=torch.long)

        ar_d = torch.arange(D, device=dev, dtype=torch.long)
        roll_idx_0 = torch.stack([(ar_d - s) % D for s in (3, 7, 13)], dim=0).view(1, 3, 1, D).expand(1, 3, 2, D).contiguous()
        roll_idx_1 = torch.stack([(ar_d - s) % D for s in (3, 7, 13, 23, 37, 47)], dim=0).view(1, 6, 1, D).expand(1, 6, 2, D).contiguous()
        w0_3 = torch.tensor([1.40, 1.50, 1.60], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        w1_6 = torch.tensor([1.10, 1.25, 1.45, 1.75, 2.10, 2.40], device=dev, dtype=torch.float32).view(1, 6, 1, 1)

        p_banks = tuple(model.prime_banks)
        max_p = max(p_banks)
        b_gather = []
        b_mask = torch.zeros((1, 1, 4, max_p), device=dev, dtype=torch.float32)
        b_offs = []
        off_p = 0
        for r_i, p_v in enumerate(p_banks):
            b_offs.append(off_p)
            b_gather.extend([off_p + min(j, p_v - 1) for j in range(max_p)])
            if p_v < max_p:
                b_mask[0, 0, r_i, p_v:] = -1e9
            off_p += p_v
        bank_gather_idx = torch.tensor(b_gather, device=dev, dtype=torch.long)
        bank_offsets = torch.tensor(b_offs, device=dev, dtype=torch.long).view(1, 1, 4)

        dummy_h = torch.zeros((1, 1, d_m), device=dev, dtype=comp_dt)
        layer_packs = []
        for blk in model.layers:
            attn: VAULTv2Attention = blk.attn
            sb = attn.salience_binder
            n_keep_g = attn.n_keep_g
            n_keep_h = n_keep_g * attn.heads_per_group
            w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = attn._get_v2_2_packed_weights(dummy_h)
            wk_raw = attn.kenyon_proj.weight.detach().to(comp_dt)
            wk_th = 0.25 * wk_raw.abs().mean()
            wk_tern = torch.where(
                wk_raw > wk_th,
                torch.ones_like(wk_raw),
                torch.where(wk_raw < -wk_th, -torch.ones_like(wk_raw), torch.zeros_like(wk_raw)),
            ).contiguous()
            layer_packs.append(
                {
                    "w_x_all": w_x_all,
                    "w_qkv_t": qkv_w.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),  # (1, 4, 1280)
                    "qkv_b_f": qkv_b.float().view(1, 1, -1).contiguous(),
                    "w_iso_all": w_iso_all,
                    "w_x0n_all": w_x0n_all,
                    "w_ib_rms": w_ib_rms,
                    "w_ac": sb.anchor_ent_conv.weight.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),
                    "w_sc": sb.sal_span_conv.weight.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),
                    "w_si_span": sb.sal_in.weight[:, :d_m].to(comp_dt).contiguous(),
                    "w_si_aux": sb.sal_in.weight[:, 3 * d_m :].to(comp_dt).contiguous(),
                    "occ_sc": F.softplus(sb.occ_scale.float()).squeeze(2).squeeze(2).contiguous(),
                    "rec_sc": F.softplus(sb.recency_param.float()).squeeze(2).squeeze(2).contiguous(),
                    "m_shared": ((0.20 / 8.0) * torch.tanh(attn.ind_shared_proj.float())).contiguous(),
                    "w_ir_xrms": attn.ind_read_proj.weight[:, d_m : 2 * d_m].to(comp_dt).contiguous(),
                    "bq_keep": torch.clamp(attn.beta_sem_q.float(), 1.2, 3.5)[:, :n_keep_h].contiguous(),
                    "bk_keep": torch.clamp(attn.beta_sem_k.float(), 1.2, 3.5)[:, :n_keep_g].contiguous(),
                    "v_gate_keep": attn.vault_gate[:, :n_keep_h].to(comp_dt).contiguous(),
                    "s_gate_keep": attn.syntax_gate[:, :n_keep_h].to(comp_dt).contiguous(),
                    "g_gate_keep": attn.gtrm_gate[:, :n_keep_h].to(comp_dt).contiguous(),
                    "s_gate_ind": attn.syntax_gate[:, n_keep_h:].to(comp_dt).contiguous(),
                    "g_gate_ind": attn.gtrm_gate[:, n_keep_h:].to(comp_dt).contiguous(),
                    "w_kenyon_tern": wk_tern,
                    "apl_gain_f": F.relu(attn.apl_norm.gain.detach()).float().contiguous(),
                }
            )

        pack = {
            "cm": cm,
            "cone_strength": cone_strength,
            "cone_u": cone_u,
            "cms_proj_t": cms_proj_t,
            "cms_gather_idx": cms_gather_idx,
            "cms_mask": cms_mask,
            "cms_banks_p": cms_banks_p,
            "max_cms": max_cms,
            "roll_idx_0": roll_idx_0,
            "roll_idx_1": roll_idx_1,
            "w0_3": w0_3,
            "w1_6": w1_6,
            "bank_gather_idx": bank_gather_idx,
            "bank_valid_mask": b_mask,
            "bank_offsets": bank_offsets,
            "max_p": max_p,
            "layers": layer_packs,
        }
        model._decode_static_pack = (pack_key, pack)
        return pack

    @classmethod
    @torch.no_grad()
    def _decode_step_static(
        cls,
        model: VAULTLMv2,
        token_id: torch.Tensor,
        cache: VAULTv2StreamCache,
        step_ratio: float = 1.0,
        occ_alpha: float = 0.5,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        100% static-shape, zero-host-sync O(1) recurrent step driven by device counter `cache.t_pos_buf`.
        Capturable directly inside `torch.cuda.CUDAGraph` on ROCm / CUDA and runnable eagerly on CPU/GPU.
        """
        B = token_id.size(0)
        C = model.chunk_size
        dev = token_id.device
        comp_dt = cache.comp_dt
        d_m = model.d_model
        pack = cls._get_decode_static_pack(model, comp_dt, dev)

        t_buf = cache.t_pos_buf               # (1,) int64
        q_in_t = t_buf % C                    # (1,) int64
        is_t0 = t_buf == 0                    # (1,) bool
        do_roll = (q_in_t == 0) & (t_buf > 0) # (1,) bool

        x = model.embed_tokens(token_id)
        if x.dtype != comp_dt:
            x = x.to(comp_dt)
        x0_t = model.compute_x0_norm(x)  # (B, 1, d_model)
        cache.x0_hist.index_copy_(1, t_buf % 1185, x0_t.to(cache.x0_hist.dtype))

        # 1. Vectorized O(1) Incremental Windowed Co-Prime CMS (`[t-1056 .. t-32)`)
        x0_f = x0_t.float()
        cm = pack["cm"]
        cone_strength = pack["cone_strength"]
        cone_cos_t = torch.sum(F.normalize(x0_f, p=2, dim=-1, eps=1e-6) * pack["cone_u"], dim=-1, keepdim=True)
        x0_c = F.normalize(x0_f - cm, p=2, dim=-1, eps=1e-6)
        x0_iso_t = (x0_c * (d_m ** 0.5)).to(comp_dt)

        slot_add = (t_buf - 33).clamp(min=0) % 1057
        ib_add = cache.cms_ib_ring.index_select(1, slot_add).squeeze(1).unsqueeze(-1)
        bg_add = cache.cms_bg_ring.index_select(1, slot_add).squeeze(1).unsqueeze(-1)
        w_add_u = (t_buf >= 33).to(torch.int32).view(1, 1, 1).expand(B, 3, 1)
        w_add_b = (t_buf > 33).to(torch.int32).view(1, 1, 1).expand(B, 3, 1)
        cache.cms_counts_u_pad.scatter_add_(2, ib_add, w_add_u)
        cache.cms_counts_b_pad.scatter_add_(2, bg_add, w_add_b)

        slot_rem = (t_buf - 1057).clamp(min=0) % 1057
        ib_rem = cache.cms_ib_ring.index_select(1, slot_rem).squeeze(1).unsqueeze(-1)
        bg_rem = cache.cms_bg_ring.index_select(1, slot_rem).squeeze(1).unsqueeze(-1)
        w_rem_u = -(t_buf >= 1057).to(torch.int32).view(1, 1, 1).expand(B, 3, 1)
        w_rem_b = -(t_buf > 1057).to(torch.int32).view(1, 1, 1).expand(B, 3, 1)
        cache.cms_counts_u_pad.scatter_add_(2, ib_rem, w_rem_u)
        cache.cms_counts_b_pad.scatter_add_(2, bg_rem, w_rem_b)

        proj_t = torch.matmul(x0_c[:, 0, :], pack["cms_proj_t"]).float()
        proj_3 = proj_t.index_select(1, pack["cms_gather_idx"]).view(B, 3, pack["max_cms"]) + pack["cms_mask"]
        ib_t = proj_3.argmax(dim=-1)  # (B, 3)
        prev_ib = torch.where(is_t0.view(1, 1), torch.zeros_like(ib_t), cache.cms_prev_ib)
        bg_t = (prev_ib * 131 + ib_t) % pack["cms_banks_p"]
        cache.cms_prev_ib.copy_(ib_t)

        slot_cur = t_buf % 1057
        cache.cms_ib_ring.index_copy_(1, slot_cur, ib_t.unsqueeze(1))
        cache.cms_bg_ring.index_copy_(1, slot_cur, bg_t.unsqueeze(1))

        min_u = cache.cms_counts_u_pad.gather(2, ib_t.unsqueeze(-1)).squeeze(-1).amin(dim=-1).float()
        min_b = cache.cms_counts_b_pad.gather(2, bg_t.unsqueeze(-1)).squeeze(-1).amin(dim=-1).float()

        t_f = t_buf.float()
        diff_f = (t_f - 32.0).clamp(min=0.0) - (t_f - 1056.0).clamp(min=0.0)
        n_sec_est_f = ((diff_f.clamp(min=1.0) - 15.0) / 110.0).clamp(min=1.0)
        hh_f = (diff_f >= 45.0).float()
        scale_f = (hh_f / n_sec_est_f).view(1, 1, 1)
        uf_t = min_u.view(B, 1, 1) * scale_f
        bf_t = min_b.view(B, 1, 1) * scale_f
        hh_t = hh_f.view(1, 1, 1).expand(B, 1, 1)

        cone_gate_l = 1.0 - cone_strength + cone_strength * torch.sigmoid(8.0 * (cone_cos_t - 0.02))
        nov_stem_t = torch.sigmoid(6.0 * (0.75 - uf_t)) * cone_gate_l
        aux_f_t = torch.cat([cone_cos_t, uf_t.clamp(max=6.0), bf_t.clamp(max=6.0), hh_t], dim=-1).to(comp_dt)
        rep_pen_t = (
            9.0 * F.relu(uf_t - 0.42)
            + 7.0 * F.relu(bf_t - 0.35)
            + (8.0 * cone_strength) * F.relu(0.08 - cone_cos_t)
        ).clamp(max=15.0).transpose(1, 2)  # (B, 1, 1)

        nov1_t = torch.sigmoid(12.0 * (0.62 - bf_t)).pow(2).transpose(1, 2)
        nov4_win = torch.cat([cache.nov1_ring, nov1_t], dim=-1)
        nov4_t = nov4_win.amax(dim=-1, keepdim=True)
        cache.nov1_ring.copy_(nov4_win[:, :, 1:])

        cc_8 = torch.cat([cache.cc_ring, cone_cos_t], dim=1)
        bf_8 = torch.cat([cache.bf_ring, bf_t], dim=1)
        cache.cc_ring.copy_(cc_8[:, 1:, :])
        cache.bf_ring.copy_(bf_8[:, 1:, :])

        # Vectorized right-suffix step_keep weights for k0 (3 lags) and k1 (6 lags)
        sig_bf_1_7 = torch.sigmoid(8.0 * (0.68 - bf_8[:, 1:7, :]))
        sig_nx_2_8 = torch.sigmoid(8.0 * (0.72 - bf_8[:, 2:8, :]))
        prod_bf = sig_bf_1_7 * sig_nx_2_8
        np_0 = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_8[:, 4:7, :] + 0.32))
        np_1 = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_8[:, 1:7, :] + 0.36))
        k0_w = (prod_bf[:, 3:6, :] * np_0).unsqueeze(2) * pack["w0_3"]  # (B, 3, 1, 1)
        k1_w = (prod_bf * np_1).unsqueeze(2) * pack["w1_6"]              # (B, 6, 1, 1)

        inv_count_t = (1.0 / (t_f + 1.0)).view(1, 1, 1, 1)
        inv_tau = 1.0 / max(0.18, 0.45 * (1.0 - 0.60 * step_ratio))
        inv_sqrt_d = 1.0 / math.sqrt(model.head_dim)
        q_sh_zero = torch.zeros((B, 1, model.layers[0].attn.n_ind_groups, 1, model.head_dim), device=dev, dtype=torch.float32)
        y_vault_buf = torch.empty((B, model.n_q_heads, 1, model.head_dim), device=dev, dtype=comp_dt)
        occ_alpha_c = round(float(occ_alpha), 4)

        roll_idx_0 = pack["roll_idx_0"].expand(B, -1, -1, -1)
        roll_idx_1 = pack["roll_idx_1"].expand(B, -1, -1, -1)
        bank_gather_idx = pack["bank_gather_idx"]
        bank_valid_mask = pack["bank_valid_mask"]
        bank_offsets = pack["bank_offsets"]
        max_p = pack["max_p"]

        idx_sal = (C + q_in_t).view(1, 1, 1).expand(B, 2, 1)
        idx_k0 = (C - 4) + q_in_t
        idx_k1 = (C - 7) + q_in_t
        idx_k01_actx = torch.cat([idx_k0, idx_k1], dim=0).view(1, 2, 1, 1).expand(B, 2, 1, 2 * model.head_dim)
        idx_k01_cs = torch.cat([idx_k0, idx_k1], dim=0).view(1, 2, 1).expand(B, 2, 1)
        cond_k01 = torch.stack([t_buf >= 4, t_buf >= 7], dim=1).view(1, 2, 1, 1)
        cond_lt_c = (t_buf < C).view(1, 1, 1, 1)
        cond_lt_c_3d = (t_buf < C).view(1, 1, 1)

        # 2. Layer-by-layer static-shape O(1) recurrent step
        for l_idx, blk in enumerate(model.layers):
            attn: VAULTv2Attention = blk.attn
            ls = cache.layers[l_idx]
            lp = pack["layers"][l_idx]

            n_keep_g = attn.n_keep_g
            n_keep_h = n_keep_g * attn.heads_per_group
            G = attn.n_vault_groups
            D = attn.head_dim
            q_dim = attn.n_q_heads * D
            kv_dim = attn.kv_dim
            total_cells = sum(attn.prime_banks)

            x_rms_raw = F.rms_norm(x.float(), (d_m,), eps=1e-6)
            h_in = x_rms_raw.to(x.dtype) * blk.norm1.weight
            x_rms = F.rms_norm(h_in.float(), (d_m,), eps=1e-6).to(comp_dt)

            # Branch-free chunk roll at `q_in == 0 and t > 0`
            rk = torch.cat([ls.local_k_win[:, :, C:, :], torch.zeros_like(ls.local_k_win[:, :, :C, :])], dim=2)
            rv = torch.cat([ls.local_v_win[:, :, C:, :], torch.zeros_like(ls.local_v_win[:, :, :C, :])], dim=2)
            ra = torch.cat([ls.a_ctx_win[:, :, C:, :], torch.zeros_like(ls.a_ctx_win[:, :, :C, :])], dim=2)
            rs = torch.cat([ls.sal_win[:, :, C:], torch.full_like(ls.sal_win[:, :, :C], -1e4)], dim=2)
            ls.local_k_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), rk, ls.local_k_win))
            ls.local_v_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), rv, ls.local_v_win))
            ls.a_ctx_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), ra, ls.a_ctx_win))
            ls.sal_win.copy_(torch.where(do_roll.view(1, 1, 1), rs, ls.sal_win))

            x_proj = F.linear(h_in, lp["w_x_all"])
            anchor = x_proj[..., :kv_dim]
            q_raw = x_proj[..., kv_dim : kv_dim + q_dim]
            k_raw = x_proj[..., kv_dim + q_dim : 2 * kv_dim + q_dim]
            v_raw = x_proj[..., 2 * kv_dim + q_dim : 3 * kv_dim + q_dim]
            beta_raw = x_proj[..., 3 * kv_dim + q_dim : 3 * kv_dim + q_dim + G] + attn.beta_proj.bias
            gate_raw = x_proj[..., 3 * kv_dim + q_dim + G :]

            anchor_q = anchor.view(B, 1, G, 1, D).expand(B, 1, G, attn.heads_per_group, D).reshape(B, 1, q_dim)
            qkv_in_t = torch.cat([anchor_q + q_raw, anchor + k_raw, v_raw], dim=-1)

            qkv_4 = torch.cat([ls.qkv_conv_state, qkv_in_t], dim=1)
            ls.qkv_conv_state.copy_(qkv_4[:, 1:, :])
            qkv_lin = (qkv_4.float() * lp["w_qkv_t"]).sum(dim=1, keepdim=True) + lp["qkv_b_f"]
            qkv_out = F.silu(qkv_lin).to(comp_dt)

            q_syn_raw = qkv_out[..., :q_dim].view(B, 1, attn.n_q_heads, D).transpose(1, 2)
            k_syn_raw = qkv_out[..., q_dim : q_dim + kv_dim].view(B, 1, G, D).transpose(1, 2)
            v = qkv_out[..., q_dim + kv_dim :].contiguous().view(B, 1, G, D).transpose(1, 2)
            dtype = v.dtype

            qs_f, ks_f = q_syn_raw.float(), k_syn_raw.float()
            qs_norm_f = F.rms_norm(qs_f, (D,), eps=1e-6)
            ks_norm_f = F.rms_norm(ks_f, (D,), eps=1e-6)
            q_norm = qs_norm_f.to(dtype)
            k_norm = ks_norm_f.to(dtype)

            sb = attn.salience_binder
            sal_c = 2 * sb.ind_dim + 1 + 2 * sb.sal_hidden
            qsh_c = attn.n_ind_groups * (attn.heads_per_group - 1) * D
            vind_c = attn.n_ind_groups * D
            ib_c = attn.n_ind_groups

            iso_proj = F.linear(x0_iso_t, lp["w_iso_all"])
            pre_sal = iso_proj[..., :sal_c]
            qsh_proj = iso_proj[..., sal_c : sal_c + qsh_c]
            vind_proj = iso_proj[..., sal_c + qsh_c : sal_c + qsh_c + vind_c]
            ib_iso_src = iso_proj[..., sal_c + qsh_c + vind_c : sal_c + qsh_c + vind_c + ib_c]
            ib_iso_p1 = ls.prev_ib_iso_p1.clone()
            ls.prev_ib_iso_p1.copy_(ib_iso_src)
            ib_iso_0 = iso_proj[..., sal_c + qsh_c + vind_c + ib_c :]

            k_dim = 2 * sb.ind_dim
            sh_dim = sb.sal_hidden
            a_raw_t = pre_sal[..., :k_dim].float()
            g_tok_t = torch.sigmoid((pre_sal[..., k_dim : k_dim + 1] + sb.ent_tok_gate.bias).float()) * nov_stem_t
            sal_iso_p1_cur = pre_sal[..., k_dim + 1 : k_dim + 1 + sh_dim]
            sal_iso_p1 = ls.prev_sal_iso_p1.clone()
            ls.prev_sal_iso_p1.copy_(sal_iso_p1_cur)
            sal_iso_0 = pre_sal[..., k_dim + 1 + sh_dim :]

            ga_t = (g_tok_t * a_raw_t).to(dtype)
            ga_6 = torch.cat([ls.anchor_conv_state, ga_t], dim=1)
            ls.anchor_conv_state.copy_(ga_6[:, 1:, :])
            a_stem_t = (ga_6.float() * lp["w_ac"]).sum(dim=1, keepdim=True)
            a_left_t = a_raw_t * (0.65 + 0.85 * torch.tanh(1.5 * a_stem_t)) + 0.45 * a_stem_t

            span_6 = torch.cat([ls.span_conv_state, x0_iso_t], dim=1)
            ls.span_conv_state.copy_(span_6[:, 1:, :])
            x0_span_t = (span_6.float() * lp["w_sc"]).sum(dim=1, keepdim=True).to(dtype)

            sal_h_t = (
                F.linear(x0_span_t, lp["w_si_span"], sb.sal_in.bias)
                + sal_iso_p1
                + sal_iso_0
                + F.linear(aux_f_t, lp["w_si_aux"])
            )
            raw_sal_t = sb.sal_out(F.silu(sal_h_t)).float().transpose(1, 2)
            sal_t = 4.0 * torch.tanh(raw_sal_t / 4.0) - rep_pen_t
            ls.sal_win.scatter_(2, idx_sal, sal_t)

            ar_g = a_raw_t.view(B, 1, 2, 2, D)
            al_g = a_left_t.view(B, 1, 2, 2, D)
            ar_8 = torch.cat([ls.a_raw_ring, ar_g], dim=1)
            al_8 = torch.cat([ls.a_left_ring, al_g], dim=1)
            ls.a_raw_ring.copy_(ar_8[:, 1:])
            ls.a_left_ring.copy_(al_8[:, 1:])

            # Batched 2-group multi-offset circular bind (1 rfft + 1 irfft instead of 2 + 9 rolls)
            tail0 = (ar_8[:, 4:7, 0].gather(-1, roll_idx_0) * k0_w).sum(dim=1, keepdim=True)  # (B, 1, 2, D)
            tail1 = (ar_8[:, 1:7, 1].gather(-1, roll_idx_1) * k1_w).sum(dim=1, keepdim=True)  # (B, 1, 2, D)
            al_01 = torch.stack([al_8[:, 3, 0], al_8[:, 0, 1]], dim=1).unsqueeze(1)           # (B, 1, 2, 2, D)
            tail_01 = torch.stack([tail0[:, 0], tail1[:, 0]], dim=1).unsqueeze(1)             # (B, 1, 2, 2, D)
            actx_01 = fft_bind_with_id(al_01, tail_01, id_weight=sb._id_weights_2g).reshape(B, 2, 1, 2 * D).to(dtype)
            cur_actx = ls.a_ctx_win.gather(2, idx_k01_actx)
            ls.a_ctx_win.scatter_(2, idx_k01_actx, torch.where(cond_k01, actx_01, cur_actx))

            s_w = ls.sal_win
            p_w = torch.where(s_w > -5e3, torch.sigmoid(s_w), torch.zeros_like(s_w))
            cs = torch.cumsum(p_w, dim=-1)
            cs_new = cs.gather(2, idx_k01_cs)
            gg = sb.gap_guard
            cs_grd = torch.cat([cs[..., gg:], cs[..., -1:].expand(-1, -1, gg)], dim=-1) if gg > 0 else cs
            prior_occ = (cs_new - cs_grd).clamp(min=0.0)

            lag_m = sb.sal_lag_mask[:, :, 0].index_select(2, q_in_t).squeeze(2)
            lag_d = sb.sal_lag_dist[:, :, 0].index_select(2, q_in_t).squeeze(2)
            valid = lag_m & (s_w > -5e3)
            eff_logits = torch.where(valid, s_w - lp["occ_sc"] * prior_occ - lp["rec_sc"] * lag_d, torch.full_like(s_w, -1e4))
            any_val = valid.any(dim=-1, keepdim=True)
            attn_w = torch.where(any_val, F.softmax(eff_logits, dim=-1), torch.zeros_like(eff_logits))
            ent_both = torch.matmul(attn_w.to(dtype).unsqueeze(-2), ls.a_ctx_win).squeeze(-2).unsqueeze(1).float()
            ent_both = torch.where(is_t0.view(1, 1, 1, 1), torch.zeros_like(ent_both), ent_both)
            ent_u, ent_r = ent_both[..., :D], ent_both[..., D:]

            ent_ur = torch.stack([ent_u, ent_r], dim=3)
            eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
            ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, lp["m_shared"])
            ur_ent_rms = F.rms_norm(ur_ent, (D,), eps=1e-6)
            ur_comb = ur_ent_rms[:, :, 0:1, :, :] + ur_ent_rms[:, :, 1:2, :, :]
            ur_ent_3 = torch.cat([ur_ent_rms, F.rms_norm(ur_comb, (D,), eps=1e-6)], dim=2)

            lg2_c = attn.lex_router_g2.weight.size(0)
            lg3_c = attn.lex_router_g3.weight.size(0)
            ir_c = attn.ind_read_proj.weight.size(0)
            s0_c = lg2_c + lg3_c + ir_c

            x0n_proj = F.linear(x0_t, lp["w_x0n_all"])
            s01 = x0n_proj[..., :s0_c] + ls.prev_x0n_s1
            ls.prev_x0n_s1.copy_(x0n_proj[..., s0_c : 2 * s0_c])
            s2_lg3 = ls.prev_x0n_s2[:, :1, :].clone()
            prev_s2_1 = ls.prev_x0n_s2[:, 1:, :].clone()
            ls.prev_x0n_s2[:, :1, :].copy_(prev_s2_1)
            ls.prev_x0n_s2[:, 1:, :].copy_(x0n_proj[:, :, 2 * s0_c :])

            lex_12 = s01[..., :lg2_c].float().view(B, 1, 2, 2, D)
            lex_3 = (s01[..., lg2_c : lg2_c + lg3_c] + s2_lg3).float().view(B, 1, 1, 2, D)
            ir_x0 = s01[..., lg2_c + lg3_c :] + attn.ind_read_proj.bias
            lex_both = torch.cat([lex_12, lex_3], dim=2)
            lex_zm = lex_both - lex_both.mean(dim=-1, keepdim=True)
            lex_rms = F.rms_norm(lex_zm, (D,), eps=1e-6)

            ur_ind_raw = fft_circular_bind(ur_ent_3, lex_rms)
            u_ind = F.rms_norm(ur_ind_raw[:, :, :, 0, :], (D,), eps=1e-6)
            r_all = F.normalize(ur_ind_raw[:, :, :, 1, :], p=2, dim=-1, eps=1e-6)

            r_all_prev = ls.r_all_prev.clone()
            u_ind_prev = ls.u_ind_prev.clone()
            ls.r_all_prev.copy_(r_all)
            ls.u_ind_prev.copy_(u_ind)

            shared_group_q = attn.prime_banks != (17, 19, 23, 29)
            lex_q_3 = r_all.transpose(1, 2)
            lex_k_3 = r_all_prev.transpose(1, 2)
            kr_f = torch.cat([ks_f[:, :n_keep_g], lex_k_3], dim=1)
            kr_unit = (F.rms_norm(kr_f, (D,), eps=1e-6) * inv_sqrt_d).to(dtype)

            if shared_group_q:
                qr_4_f = torch.cat([qs_f[:, :n_keep_h].mean(dim=1, keepdim=True), lex_q_3], dim=1)
                qr_unit = (F.rms_norm(qr_4_f, (D,), eps=1e-6) * inv_sqrt_d).to(dtype)
            else:
                qr_6_f = torch.cat([qs_f[:, :n_keep_h], lex_q_3], dim=1)
                qr_unit = (F.rms_norm(qr_6_f, (D,), eps=1e-6) * inv_sqrt_d).to(dtype)

            # Precomputed ternary Kenyon projection + APL homeostatic inhibition across Q + K simultaneously
            qk_unit = torch.cat([qr_unit, kr_unit], dim=1)  # (B, H_qk, 1, D)
            n_q_u = qr_unit.size(1)
            k_lin = F.linear(qk_unit, lp["w_kenyon_tern"]).float()
            k_inh = torch.clamp(k_lin.abs().mean(dim=-1, keepdim=True) * lp["apl_gain_f"], 0.0, 0.90)
            k_apl = k_lin * (1.0 - k_inh)
            if attn.v2_1_mode:
                kc_qk = F.relu(k_apl) * inv_tau
            else:
                kc_qk = F.relu(k_apl) * attn.harmonic_anchors.float() * inv_tau
            kc_s4 = kc_qk.squeeze(2).index_select(2, bank_gather_idx).view(B, qk_unit.size(1), 4, max_p) + bank_valid_mask
            cell_qk = (kc_s4.argmax(dim=-1) + bank_offsets).unsqueeze(-1)  # (B, H_qk, 4, 1)
            cq_u = cell_qk[:, :n_q_u]
            cell_k_raw = cell_qk[:, n_q_u:]

            k_ind_3 = u_ind_prev.transpose(1, 2)
            q_sh_rest = 0.04 * torch.tanh(qsh_proj.float()).view(B, 1, attn.n_ind_groups, attn.heads_per_group - 1, D)
            q_ind_9 = (
                (u_ind.unsqueeze(3) + torch.cat([q_sh_zero, q_sh_rest], dim=3))
                .reshape(B, 1, attn.n_ind_groups * attn.heads_per_group, D)
                .transpose(1, 2)
            )

            pq_keep = F.normalize(
                torch.exp(lp["bq_keep"] * torch.clamp(qs_norm_f[:, :n_keep_h], -3.0, 3.0)),
                p=2, dim=-1,
            )
            pk_keep = F.normalize(
                torch.exp(lp["bk_keep"] * torch.clamp(ks_norm_f[:, :n_keep_g], -3.0, 3.0)),
                p=2, dim=-1,
            )
            zq_ind = q_ind_9 - q_ind_9.mean(dim=-1, keepdim=True)
            zk_ind = k_ind_3 - k_ind_3.mean(dim=-1, keepdim=True)
            phi_q = torch.cat([pq_keep, F.normalize(zq_ind, p=2, dim=-1)], dim=1).to(dtype)
            pk_ind = F.normalize(zk_ind, p=2, dim=-1)

            v_ind = vind_proj.contiguous().view(B, 1, attn.n_ind_groups, D).transpose(1, 2)
            v_keep_w = v[:, :n_keep_g]
            v_ind_raw = 0.05 * v[:, n_keep_g:] + v_ind
            beta_keep_f = 0.96 * torch.sigmoid(beta_raw.float()).transpose(1, 2)[:, :n_keep_g, :]

            x_rms_zm = x_rms - x_rms.mean(dim=-1, keepdim=True)
            ib_rms_both = F.linear(x_rms_zm, lp["w_ib_rms"])
            ib_rms_p1 = ls.prev_ib_rms_p1.clone()
            ls.prev_ib_rms_p1.copy_(ib_rms_both[..., :ib_c])
            ib_lin = ib_rms_p1 + ib_rms_both[..., ib_c:] + ib_iso_p1 + ib_iso_0
            g_beta = torch.sigmoid(3.4 * torch.tanh(ib_lin.float() / 3.4)).transpose(1, 2)
            beta_ind_raw_f = 0.00005 + nov4_t * (0.140 * g_beta)

            # Branch-free W = 64 induction delay ring
            ck_ind_src = cell_k_raw[:, n_keep_g:]
            ls.ck_ind_first.copy_(torch.where(is_t0.view(1, 1, 1, 1), ck_ind_src, ls.ck_ind_first))
            pk_old = ls.ind_delay_pk.index_select(2, q_in_t)
            v_old = ls.ind_delay_v.index_select(2, q_in_t)
            b_old = ls.ind_delay_beta.index_select(2, q_in_t)
            ck_old = ls.ind_delay_ck.index_select(3, q_in_t)

            pk_ind_w = torch.where(cond_lt_c, torch.zeros_like(pk_old), pk_old)
            v_ind_w = torch.where(cond_lt_c, torch.zeros_like(v_old), v_old)
            beta_ind_f = torch.where(cond_lt_c_3d, torch.zeros_like(b_old), b_old)
            ck_ind_w = torch.where(cond_lt_c, ls.ck_ind_first, ck_old)

            ls.ind_delay_pk.index_copy_(2, q_in_t, pk_ind)
            ls.ind_delay_v.index_copy_(2, q_in_t, v_ind_raw.to(ls.ind_delay_v.dtype))
            ls.ind_delay_beta.index_copy_(2, q_in_t, beta_ind_raw_f)
            ls.ind_delay_ck.index_copy_(3, q_in_t, ck_ind_src)

            phi_k_t = torch.cat([pk_keep, pk_ind_w], dim=1).to(dtype).float().squeeze(2).contiguous()   # (B, 4, D)
            v_vault_t = torch.cat([v_keep_w, v_ind_w], dim=1).float().squeeze(2).contiguous()           # (B, 4, D)
            beta_t = torch.cat([beta_keep_f, beta_ind_f], dim=1).to(dtype).float().squeeze(2).contiguous()  # (B, 4)
            ck_t = torch.cat([cell_k_raw[:, :n_keep_g], ck_ind_w], dim=1).squeeze(-1).contiguous()      # (B, 4, 4)
            pq_t = phi_q.float().squeeze(2).contiguous()                                                # (B, 12, D)

            use_triton_fw = _HAS_TRITON and dev.type == "cuda" and (
                comp_dt == torch.float32 or (comp_dt == torch.bfloat16 and _gpu_supports_bf16_triton(dev))
            )
            if use_triton_fw and shared_group_q and attn.heads_per_group == 3:
                cq_t = cq_u.squeeze(-1).contiguous()
                _vault_v2_decode_fw_kernel[(B * G,)](
                    ls.S_state,
                    ls.occ_state,
                    phi_k_t,
                    v_vault_t,
                    beta_t,
                    ck_t,
                    cq_t,
                    pq_t,
                    y_vault_buf,
                    total_cells,
                    OCC_ALPHA=occ_alpha_c,
                    D=D,
                    OUT_BF16=(dtype == torch.bfloat16),
                    num_warps=4,
                )
                y_vault = y_vault_buf if dtype == comp_dt else y_vault_buf.to(dtype)
            else:
                with torch.amp.autocast(dev.type, enabled=False):
                    S_flat = ls.S_state.view(B * G, total_cells, D * D)
                    occ_flat = ls.occ_state.view(B * G, total_cells)
                    ck_flat = ck_t.reshape(B * G, 4)
                    ck_exp = ck_flat.unsqueeze(-1).expand(-1, -1, D * D)
                    S_k = S_flat.gather(1, ck_exp).view(B, G, 4, D, D)
                    pred_v = torch.einsum("bgrvd,bgd->bgrv", S_k, phi_k_t)
                    delta = beta_t.unsqueeze(2).unsqueeze(-1) * (v_vault_t.unsqueeze(2) - pred_v)
                    upd_flat = torch.einsum("bgrv,bgd->bgrvd", delta, phi_k_t).reshape(B * G, 4, D * D)
                    S_flat.scatter_add_(1, ck_exp, upd_flat)
                    occ_flat.scatter_add_(1, ck_flat, beta_t.reshape(B * G, 1).expand(-1, 4))

                    if shared_group_q:
                        cq_flat = cq_u.squeeze(-1).reshape(B * G, 4)
                        S_q_grp = S_flat.gather(1, cq_flat.unsqueeze(-1).expand(-1, -1, D * D)).view(B, G, 4, D, D)
                        n_q_grp = occ_flat.gather(1, cq_flat).view(B, G, 4)
                        pq_grp = pq_t.view(B, G, attn.heads_per_group, D)
                        y_banks = torch.einsum("bgrvd,bghd->bghrv", S_q_grp, pq_grp)  # (B, G, 3, 4, D)
                        if occ_alpha == 0.5:
                            w_banks = torch.where(n_q_grp > 1e-3, torch.rsqrt(1.0 + n_q_grp), torch.zeros_like(n_q_grp))
                        else:
                            w_banks = torch.where(n_q_grp > 1e-3, (1.0 + n_q_grp).pow(-occ_alpha), torch.zeros_like(n_q_grp))
                        w_sum = w_banks.sum(dim=-1, keepdim=True)
                        w_norm = torch.where(w_sum > 1e-6, w_banks / w_sum.clamp(min=1e-6), torch.full_like(w_banks, 0.25))
                        y_vault = (y_banks * w_norm.unsqueeze(2).unsqueeze(-1)).sum(dim=3).reshape(B, attn.n_q_heads, 1, D).to(dtype)
                    else:
                        cell_q = torch.cat(
                            [cq_u[:, :n_keep_h], cq_u[:, n_keep_h:].repeat_interleave(attn.heads_per_group, dim=1)], dim=1
                        ).squeeze(-1)
                        b_h = torch.arange(B, device=dev)[:, None, None]
                        g_for_h = (torch.arange(attn.n_q_heads, device=dev) // attn.heads_per_group)[None, :, None]
                        S_q = ls.S_state[b_h, g_for_h, cell_q]
                        n_q = ls.occ_state[b_h, g_for_h, cell_q]
                        y_banks = torch.einsum("bhrvd,bhd->bhrv", S_q, pq_t)
                        if occ_alpha == 0.5:
                            w_banks = torch.where(n_q > 1e-3, torch.rsqrt(1.0 + n_q), torch.zeros_like(n_q))
                        else:
                            w_banks = torch.where(n_q > 1e-3, (1.0 + n_q).pow(-occ_alpha), torch.zeros_like(n_q))
                        w_sum = w_banks.sum(dim=-1, keepdim=True)
                        w_norm = torch.where(w_sum > 1e-6, w_banks / w_sum.clamp(min=1e-6), torch.full_like(w_banks, 0.25))
                        y_vault = (y_banks * w_norm.unsqueeze(-1)).sum(dim=2).unsqueeze(2).to(dtype)

            # Static 128-wide Chunk-Local RoPE Syntax (columns > C + q_in are causally masked to -1e9)
            idx_loc = (C + q_in_t).view(1, 1, 1, 1).expand(B, G, 1, D)
            ls.local_k_win.scatter_(2, idx_loc, k_norm)
            ls.local_v_win.scatter_(2, idx_loc, v)
            cos_q_t = attn.rope_cos_q.index_select(3, q_in_t)
            sin_q_t = attn.rope_sin_q.index_select(3, q_in_t)
            q_rot = _apply_local_rope(q_norm.unsqueeze(2), cos_q_t, sin_q_t).squeeze(2)
            k_rot = _apply_local_rope(ls.local_k_win.unsqueeze(2), attn.rope_cos_k, attn.rope_sin_k).squeeze(2)

            q_gm = q_rot.view(B, G, attn.heads_per_group, D)
            logits_loc = (torch.matmul(q_gm, k_rot.transpose(-1, -2)).float() * inv_sqrt_d).view(B, attn.n_q_heads, 1, 2 * C)
            mask_loc = attn.local_decay_mask[:, :, 0].index_select(2, q_in_t) > 0.0
            bias_loc = attn._local_attn_bias[:, :, 0].index_select(2, q_in_t)
            logits_loc = torch.where(mask_loc, logits_loc + bias_loc, torch.full_like(logits_loc, -1e9))
            probs_loc = F.softmax(logits_loc, dim=-1).to(dtype).view(B, G, attn.heads_per_group, 2 * C)
            y_syntax = torch.matmul(probs_loc, ls.local_v_win).view(B, attn.n_q_heads, 1, D)

            # O(1) GTRM macro-summary
            ls.v_cumsum.add_(v.float())
            v_mean = (ls.v_cumsum * inv_count_t).to(dtype)
            v_macro = attn.gtrm_proj(v_mean).repeat_interleave(attn.heads_per_group, dim=1)

            y_keep = lp["v_gate_keep"] * y_vault[:, :n_keep_h] + lp["s_gate_keep"] * y_syntax[:, :n_keep_h] + lp["g_gate_keep"] * v_macro[:, :n_keep_h]
            y_base_ind = lp["s_gate_ind"] * y_syntax[:, n_keep_h:] + lp["g_gate_ind"] * v_macro[:, n_keep_h:]
            yh_cat = torch.cat([y_keep, y_base_ind], dim=1)
            y_norm_12 = F.rms_norm(yh_cat.float(), (D,), eps=1e-6).to(yh_cat.dtype) * attn.head_norm.weight
            y_base_all = y_norm_12.transpose(1, 2).reshape(B, 1, q_dim)
            gate = torch.sigmoid(gate_raw).to(y_base_all.dtype)
            out_base = attn.o_proj(y_base_all * gate)

            yv_ind_f = y_vault[:, n_keep_h:].transpose(1, 2).float()
            yv_soft_f = F.rms_norm(yv_ind_f, (D,), eps=2.5e-5)
            ir_lin = ir_x0 + F.linear(x_rms, lp["w_ir_xrms"])
            g_read_f = 2.60 * torch.sigmoid(ir_lin.float()).unsqueeze(-1)
            yv_grp = (
                (g_read_f * yv_soft_f)
                .view(B, 1, attn.n_ind_groups, attn.heads_per_group, D)
                .mean(dim=3)
                .reshape(B, 1, attn.n_ind_groups * D)
                .to(dtype)
            )
            attn_out = out_base + attn.ind_out_proj(yv_grp)

            x = x + attn_out
            x_n2 = F.rms_norm(x.float(), (d_m,), eps=1e-6).to(x.dtype) * blk.norm2.weight
            x = x + blk.mlp(x_n2)

        t_buf.add_(1)
        h_out = F.rms_norm(x.float(), (d_m,), eps=1e-6).to(x.dtype) * model.norm_f.weight
        return model.lm_head(h_out) if return_logits else h_out

    @classmethod
    @torch.no_grad()
    def decode_step(
        cls,
        model: VAULTLMv2,
        token_id: torch.Tensor,
        cache: VAULTv2StreamCache,
        step_ratio: float = 1.0,
        return_logits: bool = True,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
    ) -> torch.Tensor:
        """
        True O(1) single-token recurrent forward step.
        When `use_cudagraph=True` on CUDA/ROCm, replays a captured static `CUDAGraph` with zero per-token
        Python/HIP kernel-launch overhead!
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        B, L_one = token_id.shape
        assert L_one == 1 and B == cache.batch_size

        dev = token_id.device
        if use_cudagraph and dev.type == "cuda" and not torch.is_grad_enabled():
            amp_en = torch.is_autocast_enabled("cuda")
            amp_dt = torch.get_autocast_dtype("cuda") if amp_en else cache.comp_dt
            cur_pb = tuple(model.prime_banks)
            banks = model.layers[0].attn.salience_binder.BANKS
            g_key = (
                B,
                cache.comp_dt,
                round(float(step_ratio), 4),
                round(float(occ_alpha), 4),
                amp_en,
                amp_dt,
                dev,
                cur_pb,
            )
            runners = getattr(model, "_decode_graph_runners", None)
            if runners is None:
                runners = {}
                model._decode_graph_runners = runners
            # Evict any non-matching runners to keep at most one static_cache in VRAM at a time
            stale_keys = [k for k in runners if k != g_key]
            for sk in stale_keys:
                runners.pop(sk).release_owner(banks)
            runner = runners.get(g_key)
            if runner is None:
                runner = _DecodeCUDAGraphRunner(
                    model,
                    batch_size=B,
                    comp_dt=cache.comp_dt,
                    step_ratio=step_ratio,
                    occ_alpha=occ_alpha,
                    amp_enabled=amp_en,
                    amp_dtype=amp_dt,
                )
                runners[g_key] = runner
            h_out = runner.step(token_id, cache, banks)
            return model.lm_head(h_out) if return_logits else h_out

        cache.t_pos_buf.fill_(cache.t_pos)
        out = cls._decode_step_static(
            model,
            token_id,
            cache,
            step_ratio=step_ratio,
            occ_alpha=occ_alpha,
            return_logits=return_logits,
        )
        cache.t_pos += 1
        return out

    @classmethod
    @torch.no_grad()
    def prefill_streaming(
        cls,
        model: VAULTLMv2,
        input_ids: torch.Tensor,
        segment_size: int = 4096,
        cache: Optional[VAULTv2StreamCache] = None,
        step_ratio: float = 1.0,
        tail_logits_len: int = 1,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
        populate_decode_rings: bool = True,
    ) -> Tuple[torch.Tensor, VAULTv2StreamCache]:
        """
        Streams an arbitrary-length `input_ids (B, L)` sequence in `segment_size` chunks (`segment_size % 64 == 0`)
        with O(1) constant activation VRAM, returning `(tail_logits, cache)` ready for `decode_step`!
        """
        model.eval()
        B, L = input_ids.shape
        assert L >= 1
        C = model.chunk_size
        assert segment_size >= C and segment_size % C == 0
        if cache is None:
            cache = cls.init_cache(model, batch_size=B)

        # If cache was at a non-chunk-aligned position, step up to the next 64-token boundary first
        pos = 0
        if (cache.t_pos % C) != 0:
            to_align = min(L, C - (cache.t_pos % C))
            h_list = []
            for i in range(to_align):
                h_one = cls.decode_step(
                    model,
                    input_ids[:, pos + i : pos + i + 1],
                    cache,
                    step_ratio=step_ratio,
                    return_logits=False,
                    occ_alpha=occ_alpha,
                    use_cudagraph=use_cudagraph,
                )
                h_list.append(h_one.clone() if use_cudagraph else h_one)
            pos += to_align
            if pos == L:
                h_cat = torch.cat(h_list[-tail_logits_len:], dim=1)
                return model.lm_head(h_cat), cache

        rem_total = L - pos
        aligned_len = (rem_total // C) * C
        last_seg_h: Optional[torch.Tensor] = None

        seg_cursor = pos
        seg_end_all = pos + aligned_len
        while seg_cursor < seg_end_all:
            nxt_end = min(seg_cursor + segment_size, seg_end_all)
            is_final_aligned = (nxt_end == seg_end_all)
            h_seg = cls._prefill_aligned_segment(
                model,
                input_ids[:, seg_cursor:nxt_end],
                cache,
                step_ratio=step_ratio,
                populate_decode_rings=(is_final_aligned and (populate_decode_rings or seg_end_all < L)),
                occ_alpha=occ_alpha,
            )
            if nxt_end + (L - seg_end_all) >= L - tail_logits_len:
                keep_from_seg = max(0, tail_logits_len - (L - nxt_end))
                if keep_from_seg > 0:
                    h_take = h_seg[:, -keep_from_seg:, :]
                    last_seg_h = h_take if last_seg_h is None else torch.cat([last_seg_h, h_take], dim=1)
            seg_cursor = nxt_end

        tail_h_list: List[torch.Tensor] = []
        for idx_t in range(seg_end_all, L):
            h_one = cls.decode_step(
                model,
                input_ids[:, idx_t : idx_t + 1],
                cache,
                step_ratio=step_ratio,
                return_logits=False,
                occ_alpha=occ_alpha,
                use_cudagraph=use_cudagraph,
            )
            if idx_t >= L - tail_logits_len:
                tail_h_list.append(h_one.clone() if use_cudagraph else h_one)

        pieces = []
        if last_seg_h is not None:
            pieces.append(last_seg_h)
        if tail_h_list:
            pieces.append(torch.cat(tail_h_list, dim=1))
        h_tail = torch.cat(pieces, dim=1)[:, -tail_logits_len:, :]
        return model.lm_head(h_tail), cache

    @classmethod
    @torch.no_grad()
    def generate_greedy(
        cls,
        model: VAULTLMv2,
        prefix_ids: List[int],
        max_new_tokens: int,
        segment_size: int = 4096,
        stop_ids: Tuple[int, ...] = (),
        tok=None,
        min_chars_stop: int = 0,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
    ) -> Tuple[List[int], VAULTv2StreamCache]:
        """
        End-to-end O(1)-VRAM streaming prefill + O(1)-latency token-by-token greedy generation!
        """
        dev = model.embed_tokens.weight.device
        inp = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
        last_logits, cache = cls.prefill_streaming(
            model, inp, segment_size=segment_size, tail_logits_len=1, occ_alpha=occ_alpha, use_cudagraph=use_cudagraph
        )
        generated: List[int] = []

        nxt = int(torch.argmax(last_logits[0, -1].float()).item())
        for step_i in range(max_new_tokens):
            if nxt in stop_ids:
                break
            generated.append(nxt)
            if tok is not None and min_chars_stop > 0:
                txt = tok.decode(generated, skip_special_tokens=True)
                if len(txt.strip()) >= min_chars_stop or "\n" in txt:
                    break
            if step_i + 1 < max_new_tokens:
                nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=dev)
                logits_step = cls.decode_step(
                    model, nxt_t, cache, return_logits=True, occ_alpha=occ_alpha, use_cudagraph=use_cudagraph
                )
                nxt = int(torch.argmax(logits_step[0, -1].float()).item())

        return generated, cache
