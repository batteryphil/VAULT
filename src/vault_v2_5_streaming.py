# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 True O(1)-State Streaming Prefill & Sub-5.5ms Fused-Triton Autoregressive Decode Engine
#
# Advances over `src/vault_v2_streaming.py` (100% Isolated from Active v2.4 Files):
#   1. Stationary Bias-Corrected Causal Leaky-EMA GTRM (`ls.v_ema`, `alpha = 2^-11 = 1/2048`):
#      Eliminates the `1/t` prefix-mean collapse at `t = 131,072` (`64x` shrinkage -> `1.00x` invariant),
#      matching `VAULTLMv2_5.forward` (`chunked_causal_ema_dim2`) to `< 1e-6` FP32 across arbitrary chunks.
#   2. Tri-Horizon `11`-BPE-Subword (`g0: m=1..3`, `g1: m=1..6`, `g2: m=1..10`) Streaming Ring Buffers:
#      Extends `cc_ring`, `bf_ring`, `a_raw_ring`, `a_left_ring` to length `11` and `a_ctx_win`/`sal_win`
#      to `3` branches, preserving exact write/read symmetry across `64`-token chunk rollovers.
#   3. Zero-`rocFFT` Fused Triton Decode Suite (`_DecodeCUDAGraphRunnerV2_5`):
#      - `_vault_v2_5_fft_bind_64_kernel`: Exact `D = 64` (`33`-bin) RFFT + per-bin / L2 phase normalization
#        + IRFFT in Triton registers using `16 KB` cached twiddle tables (`0` `rocFFT` launches per token!).
#      - `_vault_v2_5_salience_pool_decode_kernel`: Device-side conditional rollover + `128`-wide `cumsum` +
#        `prior_occ` + masked softmax + `a_ctx_win` pooling in 1 launch (`~25` ATen ops -> `1` launch!).
#      - `_vault_v2_5_local_syntax_decode_kernel`: Device-side conditional rollover + `64 x 128` local RoPE +
#        decay-masked softmax + `local_v_win` readout in 1 launch (`~16` ATen ops -> `1` launch!).
#      - `_vault_v2_decode_fw_kernel`: 4-bank Co-Prime Delta-rule write + occupancy + 3-subhead readout.

import atexit
import math
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    from src.kernels.flash_vault_v2_triton import triton, tl  # type: ignore

from src.kernels.flash_vault_v2_triton import _gpu_supports_bf16_triton, causal_depthwise_conv1d
from src.vault_v2_5_model import (
    VAULTLMv2_5,
    VAULTv2_5Attention,
    SalienceEntityBinderV2_5,
    _parallel_decay_scan_dim2,
    compute_ema_bias_correction,
    compute_v2_5_shared_aux,
)
from src.vault_v2_model import (
    _apply_local_rope,
    _shift_right_zero,
    fft_bind_with_id,
    fft_circular_bind,
)
from src.vault_v2_streaming import (
    _vault_v2_decode_fw_kernel,
    stateful_flash_vault_v2_chunkwise,
)


# ==============================================================================
# 1. ZERO-ROCFFT & FUSED WINDOW TRITON DECODE KERNELS (D = 64, 2C = 128)
# ==============================================================================

@triton.jit
def _vault_v2_5_fft_bind_64_kernel(
    U_ptr,
    B_ptr,
    IdW_ptr,
    F_cos_ptr,
    F_sin_ptr,
    I_cos_ptr,
    I_sin_ptr,
    Out_ptr,
    PER_BIN_UNIT_PHASE: tl.constexpr,
    POST_NORM_UR: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """
    Exact 64-point real-FFT phase binding in registers (zero `rocFFT` library dispatches!):
      - `PER_BIN_UNIT_PHASE=True`  -> matches `fft_bind_with_id(u, tail, id_weight)` (`tf / (|tf| + 1e-6)`).
      - `PER_BIN_UNIT_PHASE=False` -> matches `fft_circular_bind(u, lex)` (`F.normalize(rfft(lex), p=2, eps=1e-6)`).
      - `POST_NORM_UR=True`        -> fuses `F.rms_norm` on even rows (`u_ind`) and `F.normalize` on odd rows (`r_all`).
    """
    pid = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)
    valid_k = offs_k < 33

    u = tl.load(U_ptr + pid * 64 + offs_n).to(tl.float32)
    b = tl.load(B_ptr + pid * 64 + offs_n).to(tl.float32)

    # Twiddle tables are `(64, 64)` FP32 with zero-padded columns/rows `33..63`
    idx_2d = offs_n[:, None] * 64 + offs_k[None, :]
    f_cos = tl.load(F_cos_ptr + idx_2d)
    f_sin = tl.load(F_sin_ptr + idx_2d)

    u_re = tl.sum(u[:, None] * f_cos, axis=0)
    u_im = tl.sum(u[:, None] * f_sin, axis=0)
    b_re = tl.sum(b[:, None] * f_cos, axis=0)
    b_im = tl.sum(b[:, None] * f_sin, axis=0)

    if PER_BIN_UNIT_PHASE:
        id_w = tl.load(IdW_ptr + (pid % 6)).to(tl.float32)
        b_re = tl.where(valid_k, b_re + id_w, 0.0)
        mag = tl.sqrt(b_re * b_re + b_im * b_im) + 1e-6
        inv_m = tl.where(valid_k, 1.0 / mag, 0.0)
    else:
        l2_sq = tl.sum(b_re * b_re + b_im * b_im, axis=0)
        inv_m = 1.0 / tl.maximum(tl.sqrt(l2_sq), 1e-6)

    z_re = (u_re * b_re - u_im * b_im) * inv_m
    z_im = (u_re * b_im + u_im * b_re) * inv_m

    idx_inv = offs_k[:, None] * 64 + offs_n[None, :]
    i_cos = tl.load(I_cos_ptr + idx_inv)
    i_sin = tl.load(I_sin_ptr + idx_inv)
    out = tl.sum(z_re[:, None] * i_cos + z_im[:, None] * i_sin, axis=0)

    if POST_NORM_UR:
        sq_sum = tl.sum(out * out, axis=0)
        is_r = (pid % 2) == 1
        rms_scale = tl.rsqrt(sq_sum * (1.0 / 64.0) + 1e-6)
        l2_scale = 1.0 / tl.maximum(tl.sqrt(sq_sum), 1e-6)
        out = out * tl.where(is_r, l2_scale, rms_scale)

    if OUT_BF16:
        tl.store(Out_ptr + pid * 64 + offs_n, out.to(tl.bfloat16))
    else:
        tl.store(Out_ptr + pid * 64 + offs_n, out)


@triton.jit
def _vault_v2_5_salience_pool_decode_kernel(
    SalWin_ptr,      # (B, 3, 128) float32, updated in-place
    ACtxWin_ptr,     # (B, 3, 128, 128) comp_dt, updated in-place
    SalCur_ptr,      # (B, 3) float32
    ACtxCur_ptr,     # (B, 3, 128) comp_dt
    TPos_ptr,        # (1,) int64
    OccSc_ptr,       # (3,) float32
    RecSc_ptr,       # (3,) float32
    EntOut_ptr,      # (B, 3, 128) float32
    GAP_GUARD: tl.constexpr,
):
    """
    Fused 3-branch (`g0: lag>=4`, `g1: lag>=7`, `g2: lag>=11`) salience window rollover, scatter,
    128-wide `cumsum`, `prior_occ`, causal softmax, and `a_ctx_win` pooling in 1 kernel launch.
    Strictly two-phase (all global loads -> `tl.debug_barrier()` -> disjoint single-write global stores)
    to eliminate inter-warp RAW / WAR / WAW hazards on multi-wavefront execution.
    """
    bg = tl.program_id(0).to(tl.int64)
    g = (bg % 3).to(tl.int32)
    t_pos = tl.load(TPos_ptr).to(tl.int64)
    q_in = (t_pos % 64).to(tl.int32)
    do_roll = (q_in == 0) & (t_pos > 0)
    is_t0 = t_pos == 0
    write_actx = t_pos >= tl.where(g == 0, 4, tl.where(g == 1, 7, 11))

    min_lag = tl.where(g == 0, 4, tl.where(g == 1, 7, 11))
    offs_j = tl.arange(0, 128)
    offs_d = tl.arange(0, 128)
    offs_r64 = tl.arange(0, 64)

    sw_base = SalWin_ptr + bg * 128
    ac_base = ACtxWin_ptr + bg * (128 * 128)

    sal_cur = tl.load(SalCur_ptr + bg).to(tl.float32)
    actx_new = tl.load(ACtxCur_ptr + bg * 128 + offs_d)
    occ_sc = tl.load(OccSc_ptr + g).to(tl.float32)
    rec_sc = tl.load(RecSc_ptr + g).to(tl.float32)

    pos_cur = 64 + q_in
    idx_k = (64 - min_lag) + q_in
    j_win = q_in + offs_r64
    idx_j1 = tl.minimum(offs_j + 1, 127)
    idx_j2 = tl.minimum(offs_j + 2, 127)

    # Phase 1: Read-only loads from `sw_base` & `ac_base` (zero stores executed yet)
    if do_roll:
        s0_pre = tl.load(sw_base + tl.minimum(offs_j + 64, 127))
        s1_pre = tl.load(sw_base + tl.minimum(idx_j1 + 64, 127))
        s2_pre = tl.load(sw_base + tl.minimum(idx_j2 + 64, 127))
        s_w = tl.where(offs_j < 64, s0_pre, -1e4)
        s_p1 = tl.where(idx_j1 < 64, s1_pre, -1e4)
        s_p2 = tl.where(idx_j2 < 64, s2_pre, -1e4)
        blk_hi = tl.load(ac_base + (offs_r64[:, None] + 64) * 128 + offs_d[None, :])
        ac_win_upd = tl.where(write_actx & (offs_r64[:, None] == idx_k), actx_new[None, :], blk_hi)
    else:
        s_w = tl.load(sw_base + offs_j)
        s_p1 = tl.load(sw_base + idx_j1)
        s_p2 = tl.load(sw_base + idx_j2)
        ac_win_old = tl.load(ac_base + j_win[:, None] * 128 + offs_d[None, :])
        ac_win_upd = tl.where(write_actx & (j_win[:, None] == idx_k), actx_new[None, :], ac_win_old)

    s_w = tl.where(offs_j == pos_cur, sal_cur, s_w)
    s_p1 = tl.where(idx_j1 == pos_cur, sal_cur, s_p1)
    s_p2 = tl.where(idx_j2 == pos_cur, sal_cur, s_p2)

    p_w = tl.where(s_w > -5e3, tl.sigmoid(s_w), 0.0)
    cs = tl.cumsum(p_w, axis=0)
    cs_new = tl.sum(tl.where(offs_j == idx_k, cs, 0.0), axis=0)

    pw_1 = tl.where((offs_j + 1 <= 127) & (s_p1 > -5e3), tl.sigmoid(s_p1), 0.0)
    pw_2 = tl.where((offs_j + 2 <= 127) & (s_p2 > -5e3), tl.sigmoid(s_p2), 0.0)
    cs_grd = cs + pw_1 + pw_2
    prior_occ = tl.maximum(cs_new - cs_grd, 0.0)

    diff = pos_cur - offs_j
    valid = (diff >= min_lag) & (diff <= 64) & (s_w > -5e3)
    lag_d = tl.maximum(diff, 0).to(tl.float32)
    eff = tl.where(valid, s_w - occ_sc * prior_occ - rec_sc * lag_d, -1e4)

    max_eff = tl.max(eff, axis=0)
    any_val = max_eff > -5e3
    exp_e = tl.where(valid, tl.exp(eff - max_eff), 0.0)
    denom = tl.sum(exp_e, axis=0)
    attn_w = tl.where(any_val, exp_e / tl.maximum(denom, 1e-12), 0.0)

    w_win = tl.sum(
        tl.where(j_win[:, None] == offs_j[None, :], attn_w[None, :], 0.0),
        axis=1,
    )
    out_pooled = tl.sum(w_win[:, None] * ac_win_upd.to(tl.float32), axis=0)
    out_128 = tl.where(is_t0 | (not any_val), tl.zeros([128], dtype=tl.float32), out_pooled)

    # Ensure all wavefronts finish Phase-1 loads before any Phase-2 in-place store
    tl.debug_barrier()

    # Phase 2: Disjoint single-write stores (each global address written at most once)
    tl.store(sw_base + offs_j, s_w)
    if do_roll:
        tl.store(ac_base + offs_r64[:, None] * 128 + offs_d[None, :], ac_win_upd)
        tl.store(
            ac_base + (offs_r64[:, None] + 64) * 128 + offs_d[None, :],
            tl.zeros([64, 128], dtype=ac_win_upd.dtype),
        )
    else:
        if write_actx:
            tl.store(ac_base + idx_k * 128 + offs_d, actx_new)
    tl.store(EntOut_ptr + bg * 128 + offs_d, out_128)


@triton.jit
def _vault_v2_5_local_syntax_decode_kernel(
    KWin_ptr,        # (B, G, 128, 64) comp_dt, updated in-place
    VWin_ptr,        # (B, G, 128, 64) comp_dt, updated in-place
    QNorm_ptr,       # (B, 12, 64) comp_dt
    KNorm_ptr,       # (B, G, 64) comp_dt
    VCur_ptr,        # (B, G, 64) comp_dt
    CosQ_ptr,        # (64, 32) float32
    SinQ_ptr,        # (64, 32) float32
    CosK_ptr,        # (128, 32) float32
    SinK_ptr,        # (128, 32) float32
    Bias_ptr,        # (12, 64, 128) float32 (`_local_attn_bias`)
    TPos_ptr,        # (1,) int64
    YSyn_ptr,        # (B, 12, 64) comp_dt
    OUT_BF16: tl.constexpr,
):
    """
    Fused `64 x 128` Chunk-Local RoPE + Multi-Scale Decay Syntax Attention decode kernel (`G = 4`, `3` heads/group):
      - Strictly two-phase (all global loads -> `tl.debug_barrier()` -> disjoint single-write global stores)
        eliminating inter-warp RAW / WAR / WAW hazards at `64`-token chunk rollovers.
    """
    bg = tl.program_id(0).to(tl.int64)
    b = bg // 4
    g = bg % 4

    t_pos = tl.load(TPos_ptr).to(tl.int64)
    q_in = (t_pos % 64).to(tl.int64)
    do_roll = (q_in == 0) & (t_pos > 0)
    pos_cur = 64 + q_in

    offs_d = tl.arange(0, 64)
    offs_32 = tl.arange(0, 32)
    offs_r64 = tl.arange(0, 64)
    offs_128 = tl.arange(0, 128)

    kw_base = KWin_ptr + bg * (128 * 64)
    vw_base = VWin_ptr + bg * (128 * 64)

    # Phase 1: Read-only loads from `KNorm_ptr`, `VCur_ptr`, `kw_base`, `vw_base`, RoPE & bias tables
    k_cur = tl.load(KNorm_ptr + bg * 64 + offs_d)
    v_cur = tl.load(VCur_ptr + bg * 64 + offs_d)
    k_cur_ev = tl.load(KNorm_ptr + bg * 64 + 2 * offs_32).to(tl.float32)
    k_cur_od = tl.load(KNorm_ptr + bg * 64 + 2 * offs_32 + 1).to(tl.float32)

    if do_roll:
        kh = tl.load(kw_base + (offs_r64[:, None] + 64) * 64 + offs_d[None, :])
        vh = tl.load(vw_base + (offs_r64[:, None] + 64) * 64 + offs_d[None, :])
        src_row = tl.minimum(offs_128 + 64, 127)
        k_ev_pre = tl.load(kw_base + src_row[:, None] * 64 + (2 * offs_32)[None, :]).to(tl.float32)
        k_od_pre = tl.load(kw_base + src_row[:, None] * 64 + (2 * offs_32 + 1)[None, :]).to(tl.float32)
        v_pre = tl.load(vw_base + src_row[:, None] * 64 + offs_d[None, :]).to(tl.float32)
        is_r0_63 = offs_128[:, None] < 64
        is_r64 = offs_128[:, None] == 64
        k_ev = tl.where(is_r0_63, k_ev_pre, tl.where(is_r64, k_cur_ev[None, :], 0.0))
        k_od = tl.where(is_r0_63, k_od_pre, tl.where(is_r64, k_cur_od[None, :], 0.0))
        v_win = tl.where(is_r0_63, v_pre, tl.where(is_r64, v_cur.to(tl.float32)[None, :], 0.0))
    else:
        kh = tl.zeros([64, 64], dtype=k_cur.dtype)
        vh = tl.zeros([64, 64], dtype=v_cur.dtype)
        k_ev_old = tl.load(kw_base + offs_128[:, None] * 64 + (2 * offs_32)[None, :]).to(tl.float32)
        k_od_old = tl.load(kw_base + offs_128[:, None] * 64 + (2 * offs_32 + 1)[None, :]).to(tl.float32)
        v_old = tl.load(vw_base + offs_128[:, None] * 64 + offs_d[None, :]).to(tl.float32)
        is_cur = offs_128[:, None] == pos_cur
        k_ev = tl.where(is_cur, k_cur_ev[None, :], k_ev_old)
        k_od = tl.where(is_cur, k_cur_od[None, :], k_od_old)
        v_win = tl.where(is_cur, v_cur.to(tl.float32)[None, :], v_old)

    cq = tl.load(CosQ_ptr + q_in * 32 + offs_32)
    sq = tl.load(SinQ_ptr + q_in * 32 + offs_32)
    if OUT_BF16:
        cq = cq.to(tl.bfloat16).to(tl.float32)
        sq = sq.to(tl.bfloat16).to(tl.float32)

    h0 = g * 3
    q_base = QNorm_ptr + (b * 12 + h0) * 64

    q0_ev = tl.load(q_base + 0 * 64 + 2 * offs_32).to(tl.float32)
    q0_od = tl.load(q_base + 0 * 64 + 2 * offs_32 + 1).to(tl.float32)
    q1_ev = tl.load(q_base + 1 * 64 + 2 * offs_32).to(tl.float32)
    q1_od = tl.load(q_base + 1 * 64 + 2 * offs_32 + 1).to(tl.float32)
    q2_ev = tl.load(q_base + 2 * 64 + 2 * offs_32).to(tl.float32)
    q2_od = tl.load(q_base + 2 * 64 + 2 * offs_32 + 1).to(tl.float32)

    qr0_ev = q0_ev * cq - q0_od * sq
    qr0_od = q0_ev * sq + q0_od * cq
    qr1_ev = q1_ev * cq - q1_od * sq
    qr1_od = q1_ev * sq + q1_od * cq
    qr2_ev = q2_ev * cq - q2_od * sq
    qr2_od = q2_ev * sq + q2_od * cq
    if OUT_BF16:
        qr0_ev = qr0_ev.to(tl.bfloat16).to(tl.float32)
        qr0_od = qr0_od.to(tl.bfloat16).to(tl.float32)
        qr1_ev = qr1_ev.to(tl.bfloat16).to(tl.float32)
        qr1_od = qr1_od.to(tl.bfloat16).to(tl.float32)
        qr2_ev = qr2_ev.to(tl.bfloat16).to(tl.float32)
        qr2_od = qr2_od.to(tl.bfloat16).to(tl.float32)

    ck = tl.load(CosK_ptr + offs_128[:, None] * 32 + offs_32[None, :])
    sk = tl.load(SinK_ptr + offs_128[:, None] * 32 + offs_32[None, :])
    if OUT_BF16:
        ck = ck.to(tl.bfloat16).to(tl.float32)
        sk = sk.to(tl.bfloat16).to(tl.float32)

    kr_ev = k_ev * ck - k_od * sk
    kr_od = k_ev * sk + k_od * ck
    if OUT_BF16:
        kr_ev = kr_ev.to(tl.bfloat16).to(tl.float32)
        kr_od = kr_od.to(tl.bfloat16).to(tl.float32)

    scale = 0.125  # 1 / sqrt(64)
    valid = offs_128 <= pos_cur

    b_ptr0 = Bias_ptr + ((h0 + 0) * 64 + q_in) * 128 + offs_128
    b_ptr1 = Bias_ptr + ((h0 + 1) * 64 + q_in) * 128 + offs_128
    b_ptr2 = Bias_ptr + ((h0 + 2) * 64 + q_in) * 128 + offs_128

    l0 = (tl.sum(kr_ev * qr0_ev[None, :], axis=1) + tl.sum(kr_od * qr0_od[None, :], axis=1)) * scale
    l1 = (tl.sum(kr_ev * qr1_ev[None, :], axis=1) + tl.sum(kr_od * qr1_od[None, :], axis=1)) * scale
    l2 = (tl.sum(kr_ev * qr2_ev[None, :], axis=1) + tl.sum(kr_od * qr2_od[None, :], axis=1)) * scale

    l0 = tl.where(valid, l0 + tl.load(b_ptr0), -1e9)
    l1 = tl.where(valid, l1 + tl.load(b_ptr1), -1e9)
    l2 = tl.where(valid, l2 + tl.load(b_ptr2), -1e9)

    e0 = tl.exp(l0 - tl.max(l0, axis=0))
    e1 = tl.exp(l1 - tl.max(l1, axis=0))
    e2 = tl.exp(l2 - tl.max(l2, axis=0))
    p0 = e0 / tl.sum(e0, axis=0)
    p1 = e1 / tl.sum(e1, axis=0)
    p2 = e2 / tl.sum(e2, axis=0)
    if OUT_BF16:
        p0 = p0.to(tl.bfloat16).to(tl.float32)
        p1 = p1.to(tl.bfloat16).to(tl.float32)
        p2 = p2.to(tl.bfloat16).to(tl.float32)

    y0 = tl.sum(p0[:, None] * v_win, axis=0)
    y1 = tl.sum(p1[:, None] * v_win, axis=0)
    y2 = tl.sum(p2[:, None] * v_win, axis=0)

    # Ensure all wavefronts finish Phase-1 loads before any Phase-2 in-place store
    tl.debug_barrier()

    # Phase 2: Disjoint single-write stores (each global address written at most once)
    if do_roll:
        tl.store(kw_base + offs_r64[:, None] * 64 + offs_d[None, :], kh)
        tl.store(vw_base + offs_r64[:, None] * 64 + offs_d[None, :], vh)
        z_tile = tl.zeros([64, 64], dtype=k_cur.dtype)
        hi_k_new = tl.where(offs_r64[:, None] == 0, k_cur[None, :], z_tile)
        hi_v_new = tl.where(offs_r64[:, None] == 0, v_cur[None, :], z_tile)
        tl.store(kw_base + (offs_r64[:, None] + 64) * 64 + offs_d[None, :], hi_k_new)
        tl.store(vw_base + (offs_r64[:, None] + 64) * 64 + offs_d[None, :], hi_v_new)
    else:
        tl.store(kw_base + pos_cur * 64 + offs_d, k_cur)
        tl.store(vw_base + pos_cur * 64 + offs_d, v_cur)

    out_base = YSyn_ptr + (b * 12 + h0) * 64
    if OUT_BF16:
        tl.store(out_base + 0 * 64 + offs_d, y0.to(tl.bfloat16))
        tl.store(out_base + 1 * 64 + offs_d, y1.to(tl.bfloat16))
        tl.store(out_base + 2 * 64 + offs_d, y2.to(tl.bfloat16))
    else:
        tl.store(out_base + 0 * 64 + offs_d, y0)
        tl.store(out_base + 1 * 64 + offs_d, y1)
        tl.store(out_base + 2 * 64 + offs_d, y2)


# ==============================================================================
# 2. CONSTANT-SIZE O(1) STREAMING & AUTOREGRESSIVE DECODE CACHE (VAULT v2.5)
# ==============================================================================

@dataclass
class LayerStreamStateV2_5:
    S_state: torch.Tensor                  # (B, G, total_cells, D, D) float32
    occ_state: torch.Tensor                # (B, G, total_cells) float32
    v_ema: torch.Tensor                    # (B, G, 1, D) float32 — stationary Leaky-EMA GTRM state!
    qkv_conv_state: torch.Tensor           # (B, 3, q_dim + 2*kv_dim)
    prev_ib_rms_p1: torch.Tensor           # (B, 1, 3)
    prev_sal_iso_p1: torch.Tensor          # (B, 1, 128)
    prev_ib_iso_p1: torch.Tensor           # (B, 1, 3)
    anchor_conv_state: torch.Tensor        # (B, 5, 256)
    span_conv_state: torch.Tensor          # (B, 5, d_model)
    a_raw_ring: torch.Tensor               # (B, 11, 2, 2, D) float32 — last 11 tokens [t-11..t-1]
    a_left_ring: torch.Tensor              # (B, 11, 2, 2, D) float32 — last 11 tokens [t-11..t-1]
    a_ctx_win: torch.Tensor                # (B, 3, 128, 128) comp_dt — [prev_chunk, cur_chunk] across g0,g1,g2
    sal_win: torch.Tensor                  # (B, 3, 128) float32 — [prev_chunk, cur_chunk] across g0,g1,g2
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
class VAULTv2_5StreamCache:
    batch_size: int
    t_pos: int
    t_pos_buf: torch.Tensor                # (1,) int64 device counter
    comp_dt: torch.dtype
    device: torch.device
    x0_hist: torch.Tensor                  # (B, 1185, d_model)
    cms_counts_u_pad: torch.Tensor         # (B, 3, 528) int32
    cms_counts_b_pad: torch.Tensor         # (B, 3, 528) int32
    cms_counts_u: List[torch.Tensor]
    cms_counts_b: List[torch.Tensor]
    cms_ib_ring: torch.Tensor              # (B, 1057, 3) int64
    cms_bg_ring: torch.Tensor              # (B, 1057, 3) int64
    cms_prev_ib: torch.Tensor              # (B, 3) int64
    cc_ring: torch.Tensor                  # (B, 11, 1) float32 — [t-11..t-1]
    bf_ring: torch.Tensor                  # (B, 11, 1) float32 — [t-11..t-1]
    nov1_ring: torch.Tensor                # (B, 1, 3) float32 — [t-3..t-1]
    layers: List[LayerStreamStateV2_5] = field(default_factory=list)
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


_ACTIVE_V25_CUDAGRAPHS: List[weakref.ReferenceType] = []


def _cleanup_active_v25_cudagraphs() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for ref in list(_ACTIVE_V25_CUDAGRAPHS):
            r = ref()
            if r is not None and getattr(r, "graph", None) is not None:
                r.graph.reset()
                r.graph = None
    except Exception:
        pass


atexit.register(_cleanup_active_v25_cudagraphs)


class _DecodeCUDAGraphRunnerV2_5:
    """
    Captures `VAULTv2_5StreamingEngine._decode_step_static` inside a zero-`rocFFT` `torch.cuda.CUDAGraph`
    (HIPGraph on ROCm 7.2), achieving `< 5.5 ms/tok` O(1) latency across `2K -> 128K+` contexts and
    `220 -> 2,002` Co-Prime cells.
    """

    def __init__(
        self,
        model: VAULTLMv2_5,
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
        self.static_cache = VAULTv2_5StreamingEngine.init_cache(model, batch_size=batch_size, dtype=comp_dt)
        self.static_token_id = torch.zeros((batch_size, 1), device=self.dev, dtype=torch.long)
        self.owner_ref: Optional[weakref.ReferenceType] = None
        self.static_pack = VAULTv2_5StreamingEngine._get_decode_static_pack(model, comp_dt, self.dev)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False):
            _ = VAULTv2_5StreamingEngine._decode_step_static(
                model,
                self.static_token_id,
                self.static_cache,
                step_ratio=step_ratio,
                occ_alpha=occ_alpha,
                return_logits=False,
            )
        torch.cuda.current_stream(device=self.dev).synchronize()

        self.cap_stream = torch.cuda.Stream(device=self.dev)
        self.cap_stream.wait_stream(torch.cuda.current_stream(device=self.dev))
        with torch.cuda.stream(self.cap_stream):
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False):
                    _ = VAULTv2_5StreamingEngine._decode_step_static(
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
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False):
                self.static_h_out = VAULTv2_5StreamingEngine._decode_step_static(
                    model,
                    self.static_token_id,
                    self.static_cache,
                    step_ratio=step_ratio,
                    occ_alpha=occ_alpha,
                    return_logits=False,
                )
        torch.cuda.current_stream(device=self.dev).synchronize()
        _ACTIVE_V25_CUDAGRAPHS.append(weakref.ref(self))

    @staticmethod
    def _copy_cache_tensors_(dst: VAULTv2_5StreamCache, src: VAULTv2_5StreamCache) -> None:
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
    def _alias_cache_to_static_(
        cache: VAULTv2_5StreamCache, static_c: VAULTv2_5StreamCache, banks: Tuple[int, ...]
    ) -> None:
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
    def _detach_cache_clone_(
        cache: VAULTv2_5StreamCache, static_c: VAULTv2_5StreamCache, banks: Tuple[int, ...]
    ) -> None:
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

    def step(self, token_id: torch.Tensor, cache: VAULTv2_5StreamCache, banks: Tuple[int, ...]) -> torch.Tensor:
        cur_owner = self.owner_ref() if self.owner_ref is not None else None
        if cur_owner is not cache or cache._cg_runner_id != self.runner_id:
            if cur_owner is not None and cur_owner._cg_runner_id == self.runner_id:
                self._detach_cache_clone_(cur_owner, self.static_cache, banks)
                cur_owner._cg_runner_id = 0
            cache.t_pos_buf.fill_(cache.t_pos)
            self._copy_cache_tensors_(self.static_cache, cache)
            self._alias_cache_to_static_(cache, self.static_cache, banks)
            cache._cg_runner_id = self.runner_id
            self.owner_ref = weakref.ref(cache)

        self.static_token_id.copy_(token_id)
        self.graph.replay()
        torch.cuda.current_stream(device=self.dev).synchronize()
        cache.t_pos += 1
        self.static_cache.t_pos = cache.t_pos
        return self.static_h_out


class VAULTv2_5StreamingEngine:
    """
    Bit-exact O(1)-state Streaming Prefill + Zero-`rocFFT` Fused-Triton Autoregressive Decode Engine for `VAULTLMv2_5`.
    """

    @staticmethod
    def _infer_compute_dtype(model: VAULTLMv2_5, device: torch.device) -> torch.dtype:
        if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return model.embed_tokens.weight.dtype

    @classmethod
    @torch.no_grad()
    def init_cache(
        cls, model: VAULTLMv2_5, batch_size: int = 1, dtype: Optional[torch.dtype] = None
    ) -> VAULTv2_5StreamCache:
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

        layers: List[LayerStreamStateV2_5] = []
        for _ in range(model.n_layers):
            layers.append(
                LayerStreamStateV2_5(
                    S_state=torch.zeros((B, G, total_cells, D, D), device=dev, dtype=torch.float32),
                    occ_state=torch.zeros((B, G, total_cells), device=dev, dtype=torch.float32),
                    v_ema=torch.zeros((B, G, 1, D), device=dev, dtype=torch.float32),
                    qkv_conv_state=torch.zeros((B, 3, qkv_c), device=dev, dtype=comp_dt),
                    prev_ib_rms_p1=torch.zeros((B, 1, 3), device=dev, dtype=comp_dt),
                    prev_sal_iso_p1=torch.zeros((B, 1, 128), device=dev, dtype=comp_dt),
                    prev_ib_iso_p1=torch.zeros((B, 1, 3), device=dev, dtype=comp_dt),
                    anchor_conv_state=torch.zeros((B, 5, 256), device=dev, dtype=comp_dt),
                    span_conv_state=torch.zeros((B, 5, d_m), device=dev, dtype=comp_dt),
                    a_raw_ring=torch.zeros((B, 11, 2, 2, D), device=dev, dtype=torch.float32),
                    a_left_ring=torch.zeros((B, 11, 2, 2, D), device=dev, dtype=torch.float32),
                    a_ctx_win=torch.zeros((B, 3, 128, 128), device=dev, dtype=comp_dt),
                    sal_win=torch.full((B, 3, 128), -1e4, device=dev, dtype=torch.float32),
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

        return VAULTv2_5StreamCache(
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
            cc_ring=torch.zeros((B, 11, 1), device=dev, dtype=torch.float32),
            bf_ring=torch.zeros((B, 11, 1), device=dev, dtype=torch.float32),
            nov1_ring=torch.zeros((B, 1, 3), device=dev, dtype=torch.float32),
            layers=layers,
        )

    @staticmethod
    @torch.no_grad()
    def _sync_cms_decode_state_from_hist(
        cache: VAULTv2_5StreamCache, binder: SalienceEntityBinderV2_5, x0_tail: torch.Tensor, T_end: int
    ) -> None:
        B, L_h, d_model = x0_tail.shape
        dev = x0_tail.device
        t_start = T_end - L_h

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
            ib = proj[..., off : off + p].argmax(dim=-1)
            ib_p1 = F.pad(ib[:, :-1], (1, 0), value=0)
            bg = (ib_p1 * 131 + ib) % p
            cache.cms_prev_ib[:, b_i] = ib[:, -1]

            keep_n = min(L_h, 1057)
            k0 = L_h - keep_n
            g_idx = torch.arange(t_start + k0, T_end, device=dev) % 1057
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
        model: VAULTLMv2_5,
        seg_ids: torch.Tensor,
        cache: VAULTv2_5StreamCache,
        step_ratio: float = 1.0,
        populate_decode_rings: bool = True,
        occ_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Streams an aligned segment `seg_ids (B, S)` where `cache.t_pos % 64 == 0` and `S % 64 == 0`.
        Mathematically exact to monolithic `VAULTLMv2_5.forward` across arbitrary segment counts!
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
        x0_seg = model.compute_x0_norm(x)

        if T_0 > 0:
            h_len = min(T_0, 1185)
            idx_h = torch.arange(T_0 - h_len, T_0, device=dev) % 1185
            x0_ext = torch.cat([cache.x0_hist.index_select(1, idx_h), x0_seg], dim=1)
        else:
            x0_ext = x0_seg
        b0 = model.layers[0].attn.salience_binder
        cms_ext = compute_v2_5_shared_aux(x0_ext, b0, ema_alpha=model.ema_alpha)

        H_sb = min(T_0, 2 * C)
        L_sb = H_sb + S
        sh_sb = {
            "shape": (B, L_sb),
            "v2_5": True,
            "x0_iso": cms_ext["x0_iso"][:, -L_sb:],
            "nov_stem": cms_ext["nov_stem"][:, -L_sb:],
            "k0_steps": tuple(t[:, -L_sb:] for t in cms_ext["k0_steps"]),
            "k1_steps": tuple(t[:, -L_sb:] for t in cms_ext["k1_steps"]),
            "k2_ext_steps": tuple(t[:, -L_sb:] for t in cms_ext["k2_ext_steps"]),
            "aux_f": cms_ext["aux_f"][:, -L_sb:],
            "rep_pen": cms_ext["rep_pen"][:, :, -L_sb:],
        }
        x0_sb = x0_ext[:, -L_sb:]
        nov4_seg = cms_ext["nov4"][:, :, -S:]

        alpha_ema = float(model.ema_alpha)
        gamma_ema = 1.0 - alpha_ema
        ema_bc_seg = compute_ema_bias_correction(S, dev, alpha=alpha_ema, start_pos=T_0)
        ema_carry_pow = torch.pow(
            torch.tensor(gamma_ema, device=dev, dtype=torch.float64),
            torch.arange(1, S + 1, device=dev, dtype=torch.float64),
        ).to(torch.float32).view(1, 1, S, 1)

        H_x0n = min(T_0, 2)
        x0_for_lex = x0_ext[:, -(H_x0n + S) :]

        for l_idx, blk in enumerate(model.layers):
            attn: VAULTv2_5Attention = blk.attn
            ls = cache.layers[l_idx]
            h_in = blk.norm1(x)

            n_keep_g = attn.n_keep_g
            n_keep_h = n_keep_g * attn.heads_per_group
            G = attn.n_vault_groups
            D = attn.head_dim
            q_dim = attn.n_q_heads * D
            kv_dim = attn.kv_dim

            w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = attn._get_v2_5_packed_weights(h_in)
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
            sal_c = 2 * sb.ind_dim + 1 + 2 * sb.sal_hidden
            qsh_c = attn.n_ind_groups * (attn.heads_per_group - 1) * D
            vind_c = attn.n_ind_groups * D
            ib_c = attn.n_ind_groups

            iso_proj_sb = F.linear(sh_sb["x0_iso"], w_iso_all)
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

            m_shared_2 = (0.20 / 8.0) * torch.tanh(attn.ind_shared_proj.float())
            m_shared_g2 = m_shared_2[1:2] + (0.20 / 8.0) * torch.tanh(attn.ind_shared_proj_g2.float())
            m_shared_3 = torch.cat([m_shared_2, m_shared_g2], dim=0)

            ent_ur = torch.stack([ent_u, ent_r], dim=3)
            eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
            ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, m_shared_3)
            ur_ent_rms = ur_ent * torch.rsqrt(ur_ent.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            w_ext = 2.0 * torch.tanh(attn.ext_span_alpha.float())
            ur_comb = (
                ur_ent_rms[:, :, 0:1, :, :]
                + ur_ent_rms[:, :, 1:2, :, :]
                + w_ext * ur_ent_rms[:, :, 2:3, :, :]
            )
            ur_ent_3 = torch.cat(
                [
                    ur_ent_rms[:, :, :2, :, :],
                    ur_comb * torch.rsqrt(ur_comb.pow(2).mean(dim=-1, keepdim=True) + 1e-6),
                ],
                dim=2,
            )

            w_ir = attn.ind_read_proj.weight
            lg2_c = attn.lex_router_g2.weight.size(0)
            lg3_c = attn.lex_router_g3.weight.size(0)
            ir_c = w_ir.size(0)
            s0_c = lg2_c + lg3_c + ir_c

            x0n_proj_ext = F.linear(x0_for_lex, w_x0n_all)
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
                p=2,
                dim=-1,
            )
            pk_keep = F.normalize(
                torch.exp(bk[:, :n_keep_g] * torch.clamp(ks_f[:, :n_keep_g] * ks_rms[:, :n_keep_g], -3.0, 3.0)),
                p=2,
                dim=-1,
            )
            zq_ind = q_ind_9 - q_ind_9.mean(dim=-1, keepdim=True)
            zk_ind = k_ind_3 - k_ind_3.mean(dim=-1, keepdim=True)
            phi_q = torch.cat([pq_keep, F.normalize(zq_ind, p=2, dim=-1)], dim=1).to(dtype)
            pk_ind = F.normalize(zk_ind, p=2, dim=-1)

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

            if T_0 == 0:
                y_syntax = attn._chunk_local_rope_syntax(q_norm, k_norm, v)
            else:
                q_pad = torch.cat([torch.zeros_like(q_norm[:, :, :C, :]), q_norm], dim=2)
                k_pad = torch.cat([ls.local_k_win[:, :, C:, :], k_norm], dim=2)
                v_pad = torch.cat([ls.local_v_win[:, :, C:, :], v], dim=2)
                y_syntax = attn._chunk_local_rope_syntax(q_pad, k_pad, v_pad)[:, :, C:, :]
            ls.local_k_win[:, :, C:, :].copy_(k_norm[:, :, -C:, :])
            ls.local_v_win[:, :, C:, :].copy_(v[:, :, -C:, :])

            # Stationary Causal Leaky-EMA GTRM across arbitrary segments!
            u_ema_seg = _parallel_decay_scan_dim2(v.float(), decay=gamma_ema, step_scale=alpha_ema, block_len=C)
            m_seg = ls.v_ema * ema_carry_pow + u_ema_seg
            ls.v_ema.copy_(m_seg[:, :, -1:, :])
            v_mean = (m_seg * ema_bc_seg).to(dtype)
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
                k_dim = 2 * sb.ind_dim
                sh_dim = sb.sal_hidden
                L_tail = min(L_sb, C + 5)
                ps_t = pre_sal_sb[:, -L_tail:]
                x0_iso_t = sh_sb["x0_iso"][:, -L_tail:]
                nov_s_t = sh_sb["nov_stem"][:, -L_tail:]
                aux_t = sh_sb["aux_f"][:, -L_tail:]
                rep_t = sh_sb["rep_pen"][:, :, -L_tail:]
                k0_t = tuple(kt[:, -C:] for kt in sh_sb["k0_steps"])
                k1_t = tuple(kt[:, -C:] for kt in sh_sb["k1_steps"])
                k2e_t = tuple(kt[:, -C:] for kt in sh_sb["k2_ext_steps"])

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
                ls.a_raw_ring.copy_(a_raw_c[:, -11:])
                ls.a_left_ring.copy_(a_left_c[:, -11:])

                e0_pad = F.pad(a_raw_c[:, :, 0], (0, 0, 0, 0, 0, 4))
                e1_pad = F.pad(a_raw_c[:, :, 1], (0, 0, 0, 0, 0, 11))
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
                tail2 = (
                    tail1
                    + 2.55 * k2e_t[0] * torch.roll(e1_pad[:, 7 : C + 7], shifts=19, dims=-1)
                    + 2.65 * k2e_t[1] * torch.roll(e1_pad[:, 8 : C + 8], shifts=29, dims=-1)
                    + 2.75 * k2e_t[2] * torch.roll(e1_pad[:, 9 : C + 9], shifts=43, dims=-1)
                    + 2.85 * k2e_t[3] * torch.roll(e1_pad[:, 10 : C + 10], shifts=59, dims=-1)
                )
                a_left_3c = torch.cat([a_left_c, a_left_c[:, :, 1:2]], dim=2)
                a_ctx_c = (
                    fft_bind_with_id(a_left_3c, torch.stack([tail0, tail1, tail2], dim=2), id_weight=sb._id_weights_3g)
                    .reshape(B, C, 3, 2 * D)
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
                sal_2c = 4.0 * torch.tanh(raw_sal_c / 4.0) - rep_t[:, :, -C:]
                sal_c_out = torch.cat([sal_2c, sal_2c[:, 1:2, :]], dim=1)

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
            cache.cc_ring.copy_(cone_cos_ext[:, -11:, :])
            cache.bf_ring.copy_(bf_ext[:, -11:, :])
            nov1_ext = torch.sigmoid(12.0 * (0.62 - bf_ext[:, -3:, :])).pow(2).transpose(1, 2)
            cache.nov1_ring.copy_(nov1_ext)
            cls._sync_cms_decode_state_from_hist(cache, b0, x0_ext[:, -min(x0_ext.size(1), 1058) :], T_end)

        return model.norm_f(x)

    @staticmethod
    @torch.no_grad()
    def _get_decode_static_pack(model: VAULTLMv2_5, comp_dt: torch.dtype, dev: torch.device) -> Dict:
        ver_sum = sum(
            blk.attn.ext_span_alpha._version
            + blk.attn.ind_shared_proj_g2._version
            + blk.attn.q_proj.weight._version
            + blk.mlp.w1.weight._version
            + blk.mlp.w2.weight._version
            + blk.mlp.w3.weight._version
            for blk in model.layers
        )
        pack_key = (dev, comp_dt, tuple(model.prime_banks), ver_sum)
        cached = getattr(model, "_decode_static_pack_v2_5", None)
        if cached is not None and cached[0] == pack_key:
            return cached[1]

        d_m = model.d_model
        D = model.head_dim
        assert D == 64
        b0 = model.layers[0].attn.salience_binder
        cm = b0.x0_cone_mean.detach().float().view(1, 1, d_m).contiguous()
        cone_norm = cm.norm(dim=-1, keepdim=True)
        cone_strength = (cone_norm / ((d_m ** 0.5) * 0.30)).clamp(max=1.0).contiguous()
        cone_u = (cm / cone_norm.clamp(min=1e-6)).contiguous()
        cms_proj_t = b0.cms_proj.detach().float().T.contiguous()

        max_cms = max(b0.BANKS)
        cms_gather = []
        cms_mask = torch.zeros((1, 3, max_cms), device=dev, dtype=torch.float32)
        off_c = 0
        for r_i, p_c in enumerate(b0.BANKS):
            cms_gather.extend([off_c + min(j, p_c - 1) for j in range(max_cms)])
            if p_c < max_cms:
                cms_mask[0, r_i, p_c:] = -1e9
            off_c += p_c
        cms_gather_idx = torch.tensor(cms_gather, device=dev, dtype=torch.long)
        cms_banks_p = torch.tensor([[b0.BANKS[0], b0.BANKS[1], b0.BANKS[2]]], device=dev, dtype=torch.long)

        ar_d = torch.arange(D, device=dev, dtype=torch.long)
        roll_idx_0 = (
            torch.stack([(ar_d - s) % D for s in (3, 7, 13)], dim=0)
            .view(1, 3, 1, D)
            .expand(1, 3, 2, D)
            .contiguous()
        )
        roll_idx_1 = (
            torch.stack([(ar_d - s) % D for s in (3, 7, 13, 23, 37, 47)], dim=0)
            .view(1, 6, 1, D)
            .expand(1, 6, 2, D)
            .contiguous()
        )
        roll_idx_2 = (
            torch.stack([(ar_d - s) % D for s in (3, 7, 13, 23, 37, 47, 19, 29, 43, 59)], dim=0)
            .view(1, 10, 1, D)
            .expand(1, 10, 2, D)
            .contiguous()
        )
        w0_3 = torch.tensor([1.40, 1.50, 1.60], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        w1_6 = torch.tensor([1.10, 1.25, 1.45, 1.75, 2.10, 2.40], device=dev, dtype=torch.float32).view(1, 6, 1, 1)
        w2_10 = torch.tensor(
            [1.10, 1.25, 1.45, 1.75, 2.10, 2.40, 2.55, 2.65, 2.75, 2.85], device=dev, dtype=torch.float32
        ).view(1, 10, 1, 1)

        # Precompute exact FP64 -> FP32 `64 x 64` zero-padded RFFT & IRFFT twiddle tables (`16 KB` total!)
        n_64 = torch.arange(64, device=dev, dtype=torch.float64).unsqueeze(1)   # (64, 1)
        k_33 = torch.arange(33, device=dev, dtype=torch.float64).unsqueeze(0)   # (1, 33)
        ang = (2.0 * math.pi / 64.0) * (n_64 * k_33)                            # (64, 33)
        f_cos = torch.zeros((64, 64), device=dev, dtype=torch.float32)
        f_sin = torch.zeros((64, 64), device=dev, dtype=torch.float32)
        f_cos[:, :33] = torch.cos(ang).to(torch.float32)
        f_sin[:, :33] = (-torch.sin(ang)).to(torch.float32)

        inv_w = torch.full((33, 1), 2.0 / 64.0, device=dev, dtype=torch.float64)
        inv_w[0, 0] = 1.0 / 64.0
        inv_w[32, 0] = 1.0 / 64.0
        ang_inv = ang.T  # (33, 64)
        i_cos = torch.zeros((64, 64), device=dev, dtype=torch.float32)
        i_sin = torch.zeros((64, 64), device=dev, dtype=torch.float32)
        i_cos[:33, :] = (inv_w * torch.cos(ang_inv)).to(torch.float32)
        i_sin[:33, :] = (-inv_w * torch.sin(ang_inv)).to(torch.float32)
        id_w_6 = torch.tensor([1.50, 1.50, 1.20, 1.20, 1.15, 1.15], device=dev, dtype=torch.float32)
        zero_6 = torch.zeros((6,), device=dev, dtype=torch.float32)

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
            attn: VAULTv2_5Attention = blk.attn
            sb = attn.salience_binder
            n_keep_g = attn.n_keep_g
            n_keep_h = n_keep_g * attn.heads_per_group
            w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = attn._get_v2_5_packed_weights(dummy_h)
            wk_raw = attn.kenyon_proj.weight.detach().to(comp_dt)
            wk_th = 0.25 * wk_raw.abs().mean()
            wk_tern = torch.where(
                wk_raw > wk_th,
                torch.ones_like(wk_raw),
                torch.where(wk_raw < -wk_th, -torch.ones_like(wk_raw), torch.zeros_like(wk_raw)),
            ).contiguous()

            m_sh2 = (0.20 / 8.0) * torch.tanh(attn.ind_shared_proj.float())
            m_sh_g2 = m_sh2[1:2] + (0.20 / 8.0) * torch.tanh(attn.ind_shared_proj_g2.float())
            m_sh3 = torch.cat([m_sh2, m_sh_g2], dim=0).contiguous()

            occ_2 = F.softplus(sb.occ_scale.float()).squeeze(2).squeeze(2)
            rec_2 = F.softplus(sb.recency_param.float()).squeeze(2).squeeze(2)
            occ_3 = torch.cat([occ_2, occ_2[:, 1:2]], dim=1).contiguous()
            rec_3 = torch.cat([rec_2, rec_2[:, 1:2]], dim=1).contiguous()

            mlp_w12 = torch.cat([blk.mlp.w1.weight, blk.mlp.w2.weight], dim=0).to(comp_dt).contiguous()
            mlp_w3 = blk.mlp.w3.weight.to(comp_dt).contiguous()

            layer_packs.append(
                {
                    "w_x_all": w_x_all,
                    "w_qkv_t": qkv_w.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),
                    "qkv_b_f": qkv_b.float().view(1, 1, -1).contiguous(),
                    "w_iso_all": w_iso_all,
                    "w_x0n_all": w_x0n_all,
                    "w_ib_rms": w_ib_rms,
                    "w_ac": sb.anchor_ent_conv.weight.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),
                    "w_sc": sb.sal_span_conv.weight.squeeze(1).float().transpose(0, 1).unsqueeze(0).contiguous(),
                    "w_si_span": sb.sal_in.weight[:, :d_m].to(comp_dt).contiguous(),
                    "w_si_aux": sb.sal_in.weight[:, 3 * d_m :].to(comp_dt).contiguous(),
                    "occ_sc_3": occ_3,                                                  # (1, 3, 1)
                    "rec_sc_3": rec_3,                                                  # (1, 3, 1)
                    "occ_sc_vec": occ_3.view(3).contiguous(),                           # (3,)
                    "rec_sc_vec": rec_3.view(3).contiguous(),                           # (3,)
                    "m_shared_3": m_sh3,                                                # (3, D, D)
                    "w_ext": (2.0 * torch.tanh(attn.ext_span_alpha.float())).contiguous(),
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
                    "cos_q_flat": attn.rope_cos_q.reshape(64, 32).float().contiguous(),
                    "sin_q_flat": attn.rope_sin_q.reshape(64, 32).float().contiguous(),
                    "cos_k_flat": attn.rope_cos_k.reshape(128, 32).float().contiguous(),
                    "sin_k_flat": attn.rope_sin_k.reshape(128, 32).float().contiguous(),
                    "local_bias_flat": attn._local_attn_bias.reshape(12, 64, 128).float().contiguous(),
                    "mlp_w12": mlp_w12,
                    "mlp_w3": mlp_w3,
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
            "roll_idx_2": roll_idx_2,
            "w0_3": w0_3,
            "w1_6": w1_6,
            "w2_10": w2_10,
            "f_cos": f_cos.contiguous(),
            "f_sin": f_sin.contiguous(),
            "i_cos": i_cos.contiguous(),
            "i_sin": i_sin.contiguous(),
            "id_w_6": id_w_6.contiguous(),
            "zero_6": zero_6.contiguous(),
            "bank_gather_idx": bank_gather_idx,
            "bank_valid_mask": b_mask,
            "bank_offsets": bank_offsets,
            "max_p": max_p,
            "layers": layer_packs,
        }
        model._decode_static_pack_v2_5 = (pack_key, pack)
        return pack

    @classmethod
    @torch.no_grad()
    def _decode_step_static(
        cls,
        model: VAULTLMv2_5,
        token_id: torch.Tensor,
        cache: VAULTv2_5StreamCache,
        step_ratio: float = 1.0,
        occ_alpha: float = 0.5,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        100% static-shape, zero-host-sync, zero-`rocFFT` O(1) recurrent step.
        Uses fused Triton decode kernels on CUDA/ROCm and exact vectorized PyTorch fallback on CPU.
        """
        B = token_id.size(0)
        C = model.chunk_size
        dev = token_id.device
        comp_dt = cache.comp_dt
        use_triton_gpu = (
            _HAS_TRITON
            and (dev.type == "cuda")
            and (comp_dt == torch.float32 or (comp_dt == torch.bfloat16 and _gpu_supports_bf16_triton(dev)))
        )
        d_m = model.d_model
        pack = cls._get_decode_static_pack(model, comp_dt, dev)

        t_buf = cache.t_pos_buf
        q_in_t = t_buf % C
        is_t0 = t_buf == 0
        do_roll = (q_in_t == 0) & (t_buf > 0)

        x = model.embed_tokens(token_id)
        if x.dtype != comp_dt:
            x = x.to(comp_dt)
        x0_t = model.compute_x0_norm(x)
        cache.x0_hist.index_copy_(1, t_buf % 1185, x0_t.to(cache.x0_hist.dtype))

        # 1. O(1) Incremental Windowed Co-Prime CMS (`[t-1056 .. t-32)`)
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
        ib_t = proj_3.argmax(dim=-1)
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
        ).clamp(max=15.0).transpose(1, 2)

        nov1_t = torch.sigmoid(12.0 * (0.62 - bf_t)).pow(2).transpose(1, 2)
        nov4_win = torch.cat([cache.nov1_ring, nov1_t], dim=-1)
        nov4_t = nov4_win.amax(dim=-1, keepdim=True)
        cache.nov1_ring.copy_(nov4_win[:, :, 1:])

        cc_12 = torch.cat([cache.cc_ring, cone_cos_t], dim=1)  # [t-11 .. t]
        bf_12 = torch.cat([cache.bf_ring, bf_t], dim=1)        # [t-11 .. t]
        cache.cc_ring.copy_(cc_12[:, 1:, :])
        cache.bf_ring.copy_(bf_12[:, 1:, :])

        # Vectorized right-suffix `step_keep` weights for `g0` (`m=1..3`), `g1` (`m=1..6`), `g2` (`m=1..10`)
        sig_bf_1_11 = torch.sigmoid(8.0 * (0.68 - bf_12[:, 1:11, :]))
        sig_nx_2_12 = torch.sigmoid(8.0 * (0.72 - bf_12[:, 2:12, :]))
        prod_bf_10 = sig_bf_1_11 * sig_nx_2_12
        np_0 = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_12[:, 8:11, :] + 0.32))
        np_2 = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_12[:, 1:11, :] + 0.36))
        k0_w = (prod_bf_10[:, 7:10, :] * np_0).unsqueeze(2) * pack["w0_3"]        # (B, 3, 1, 1)
        k1_w = (prod_bf_10[:, 4:10, :] * np_2[:, 4:10, :]).unsqueeze(2) * pack["w1_6"]  # (B, 6, 1, 1)
        k2_w = (prod_bf_10 * np_2).unsqueeze(2) * pack["w2_10"]                   # (B, 10, 1, 1)

        # Stationary EMA GTRM step constants (`expm1` avoids cancellation at small `t`, `0` FP64 ops on GPU!)
        alpha_ema = float(model.ema_alpha)
        gamma_ema = 1.0 - alpha_ema
        log_gamma = math.log1p(-alpha_ema)
        inv_ema_bc = (1.0 / (-torch.expm1(log_gamma * (t_f + 1.0))).clamp(min=1e-12)).view(1, 1, 1, 1)

        inv_tau = 1.0 / max(0.18, 0.45 * (1.0 - 0.60 * step_ratio))
        inv_sqrt_d = 1.0 / math.sqrt(model.head_dim)
        q_sh_zero = torch.zeros(
            (B, 1, model.layers[0].attn.n_ind_groups, 1, model.head_dim), device=dev, dtype=torch.float32
        )
        y_vault_buf = torch.empty((B, model.n_q_heads, 1, model.head_dim), device=dev, dtype=comp_dt)
        y_syn_buf = torch.empty((B, model.n_q_heads, 1, model.head_dim), device=dev, dtype=comp_dt)
        actx_buf = torch.empty((B, 3, 1, 2 * model.head_dim), device=dev, dtype=comp_dt)
        ent_out_buf = torch.empty((B, 1, 3, 2 * model.head_dim), device=dev, dtype=torch.float32)
        ur_ind_buf = torch.empty((B, 1, 3, 2, model.head_dim), device=dev, dtype=torch.float32)
        occ_alpha_c = round(float(occ_alpha), 4)

        roll_idx_0 = pack["roll_idx_0"].expand(B, -1, -1, -1)
        roll_idx_1 = pack["roll_idx_1"].expand(B, -1, -1, -1)
        roll_idx_2 = pack["roll_idx_2"].expand(B, -1, -1, -1)
        bank_gather_idx = pack["bank_gather_idx"]
        bank_valid_mask = pack["bank_valid_mask"]
        bank_offsets = pack["bank_offsets"]
        max_p = pack["max_p"]
        f_cos = pack["f_cos"]
        f_sin = pack["f_sin"]
        i_cos = pack["i_cos"]
        i_sin = pack["i_sin"]
        id_w_6 = pack["id_w_6"]
        zero_6 = pack["zero_6"]

        cond_lt_c = (t_buf < C).view(1, 1, 1, 1)
        cond_lt_c_3d = (t_buf < C).view(1, 1, 1)

        if not use_triton_gpu:
            idx_sal = (C + q_in_t).view(1, 1, 1).expand(B, 3, 1)
            idx_k0 = (C - 4) + q_in_t
            idx_k1 = (C - 7) + q_in_t
            idx_k2 = (C - 11) + q_in_t
            idx_k012_actx = torch.cat([idx_k0, idx_k1, idx_k2], dim=0).view(1, 3, 1, 1).expand(B, 3, 1, 2 * model.head_dim)
            idx_k012_cs = torch.cat([idx_k0, idx_k1, idx_k2], dim=0).view(1, 3, 1).expand(B, 3, 1)
            cond_k012 = torch.stack([t_buf >= 4, t_buf >= 7, t_buf >= 11], dim=1).view(1, 3, 1, 1)

        # 2. Layer-by-layer O(1) recurrent step
        for l_idx, blk in enumerate(model.layers):
            attn: VAULTv2_5Attention = blk.attn
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
            sal_2t = 4.0 * torch.tanh(raw_sal_t / 4.0) - rep_pen_t
            sal_t = torch.cat([sal_2t, sal_2t[:, 1:2, :]], dim=1)  # (B, 3, 1)

            ar_g = a_raw_t.view(B, 1, 2, 2, D)
            al_g = a_left_t.view(B, 1, 2, 2, D)
            ar_12 = torch.cat([ls.a_raw_ring, ar_g], dim=1)
            al_12 = torch.cat([ls.a_left_ring, al_g], dim=1)
            ls.a_raw_ring.copy_(ar_12[:, 1:])
            ls.a_left_ring.copy_(al_12[:, 1:])

            tail0 = (ar_12[:, 8:11, 0].gather(-1, roll_idx_0) * k0_w).sum(dim=1, keepdim=True)
            tail1 = (ar_12[:, 5:11, 1].gather(-1, roll_idx_1) * k1_w).sum(dim=1, keepdim=True)
            tail2 = (ar_12[:, 1:11, 1].gather(-1, roll_idx_2) * k2_w).sum(dim=1, keepdim=True)
            al_012 = torch.stack([al_12[:, 7, 0], al_12[:, 4, 1], al_12[:, 0, 1]], dim=1).unsqueeze(1).contiguous()
            tail_012 = torch.stack([tail0[:, 0], tail1[:, 0], tail2[:, 0]], dim=1).unsqueeze(1).contiguous()

            if use_triton_gpu:
                _vault_v2_5_fft_bind_64_kernel[(B * 6,)](
                    al_012,
                    tail_012,
                    id_w_6,
                    f_cos,
                    f_sin,
                    i_cos,
                    i_sin,
                    actx_buf,
                    PER_BIN_UNIT_PHASE=True,
                    POST_NORM_UR=False,
                    OUT_BF16=(dtype == torch.bfloat16),
                    num_warps=2,
                )
                _vault_v2_5_salience_pool_decode_kernel[(B * 3,)](
                    ls.sal_win,
                    ls.a_ctx_win,
                    sal_t.contiguous(),
                    actx_buf,
                    t_buf,
                    lp["occ_sc_vec"],
                    lp["rec_sc_vec"],
                    ent_out_buf,
                    GAP_GUARD=sb.gap_guard,
                    num_warps=4,
                )
                ent_both = ent_out_buf
            else:
                ra = torch.cat([ls.a_ctx_win[:, :, C:, :], torch.zeros_like(ls.a_ctx_win[:, :, :C, :])], dim=2)
                rs = torch.cat([ls.sal_win[:, :, C:], torch.full_like(ls.sal_win[:, :, :C], -1e4)], dim=2)
                ls.a_ctx_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), ra, ls.a_ctx_win))
                ls.sal_win.copy_(torch.where(do_roll.view(1, 1, 1), rs, ls.sal_win))
                ls.sal_win.scatter_(2, idx_sal, sal_t)

                actx_012 = (
                    fft_bind_with_id(al_012, tail_012, id_weight=sb._id_weights_3g).reshape(B, 3, 1, 2 * D).to(dtype)
                )
                cur_actx = ls.a_ctx_win.gather(2, idx_k012_actx)
                ls.a_ctx_win.scatter_(2, idx_k012_actx, torch.where(cond_k012, actx_012, cur_actx))

                s_w = ls.sal_win
                p_w = torch.where(s_w > -5e3, torch.sigmoid(s_w), torch.zeros_like(s_w))
                cs = torch.cumsum(p_w, dim=-1)
                cs_new = cs.gather(2, idx_k012_cs)
                gg = sb.gap_guard
                cs_grd = torch.cat([cs[..., gg:], cs[..., -1:].expand(-1, -1, gg)], dim=-1) if gg > 0 else cs
                prior_occ = (cs_new - cs_grd).clamp(min=0.0)

                lag_m = sb._sal_lag_mask_3g[:, :, 0].index_select(2, q_in_t).squeeze(2)
                lag_d = sb.sal_lag_dist[:, :, 0].index_select(2, q_in_t).squeeze(2)
                valid = lag_m & (s_w > -5e3)
                eff_logits = torch.where(
                    valid, s_w - lp["occ_sc_3"] * prior_occ - lp["rec_sc_3"] * lag_d, torch.full_like(s_w, -1e4)
                )
                any_val = valid.any(dim=-1, keepdim=True)
                attn_w = torch.where(any_val, F.softmax(eff_logits, dim=-1), torch.zeros_like(eff_logits))
                ent_both = torch.matmul(attn_w.to(dtype).unsqueeze(-2), ls.a_ctx_win).squeeze(-2).unsqueeze(1).float()
                ent_both = torch.where(is_t0.view(1, 1, 1, 1), torch.zeros_like(ent_both), ent_both)

            ent_u, ent_r = ent_both[..., :D], ent_both[..., D:]
            ent_ur = torch.stack([ent_u, ent_r], dim=3)
            eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
            ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, lp["m_shared_3"])
            ur_ent_rms = F.rms_norm(ur_ent, (D,), eps=1e-6)
            ur_comb = (
                ur_ent_rms[:, :, 0:1, :, :]
                + ur_ent_rms[:, :, 1:2, :, :]
                + lp["w_ext"] * ur_ent_rms[:, :, 2:3, :, :]
            )
            ur_ent_3 = torch.cat([ur_ent_rms[:, :, :2, :, :], F.rms_norm(ur_comb, (D,), eps=1e-6)], dim=2).contiguous()

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
            lex_rms = F.rms_norm(lex_zm, (D,), eps=1e-6).contiguous()

            if use_triton_gpu:
                _vault_v2_5_fft_bind_64_kernel[(B * 6,)](
                    ur_ent_3,
                    lex_rms,
                    zero_6,
                    f_cos,
                    f_sin,
                    i_cos,
                    i_sin,
                    ur_ind_buf,
                    PER_BIN_UNIT_PHASE=False,
                    POST_NORM_UR=True,
                    OUT_BF16=False,
                    num_warps=2,
                )
                u_ind = ur_ind_buf[:, :, :, 0, :]
                r_all = ur_ind_buf[:, :, :, 1, :]
            else:
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

            qk_unit = torch.cat([qr_unit, kr_unit], dim=1)
            n_q_u = qr_unit.size(1)
            k_lin = F.linear(qk_unit, lp["w_kenyon_tern"]).float()
            k_inh = torch.clamp(k_lin.abs().mean(dim=-1, keepdim=True) * lp["apl_gain_f"], 0.0, 0.90)
            kc_qk = F.relu(k_lin * (1.0 - k_inh)) * inv_tau
            kc_s4 = (
                kc_qk.squeeze(2).index_select(2, bank_gather_idx).view(B, qk_unit.size(1), 4, max_p) + bank_valid_mask
            )
            cell_qk = (kc_s4.argmax(dim=-1) + bank_offsets).unsqueeze(-1)
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
                p=2,
                dim=-1,
            )
            pk_keep = F.normalize(
                torch.exp(lp["bk_keep"] * torch.clamp(ks_norm_f[:, :n_keep_g], -3.0, 3.0)),
                p=2,
                dim=-1,
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

            phi_k_t = torch.cat([pk_keep, pk_ind_w], dim=1).to(dtype).float().squeeze(2).contiguous()
            v_vault_t = torch.cat([v_keep_w, v_ind_w], dim=1).float().squeeze(2).contiguous()
            beta_t = torch.cat([beta_keep_f, beta_ind_f], dim=1).to(dtype).float().squeeze(2).contiguous()
            ck_t = torch.cat([cell_k_raw[:, :n_keep_g], ck_ind_w], dim=1).squeeze(-1).contiguous()
            pq_t = phi_q.float().squeeze(2).contiguous()

            if use_triton_gpu and shared_group_q and attn.heads_per_group == 3:
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
                        y_banks = torch.einsum("bgrvd,bghd->bghrv", S_q_grp, pq_grp)
                        if occ_alpha == 0.5:
                            w_banks = torch.where(n_q_grp > 1e-3, torch.rsqrt(1.0 + n_q_grp), torch.zeros_like(n_q_grp))
                        else:
                            w_banks = torch.where(
                                n_q_grp > 1e-3, (1.0 + n_q_grp).pow(-occ_alpha), torch.zeros_like(n_q_grp)
                            )
                        w_sum = w_banks.sum(dim=-1, keepdim=True)
                        w_norm = torch.where(w_sum > 1e-6, w_banks / w_sum.clamp(min=1e-6), torch.full_like(w_banks, 0.25))
                        y_vault = (
                            (y_banks * w_norm.unsqueeze(2).unsqueeze(-1))
                            .sum(dim=3)
                            .reshape(B, attn.n_q_heads, 1, D)
                            .to(dtype)
                        )
                    else:
                        cell_q = torch.cat(
                            [cq_u[:, :n_keep_h], cq_u[:, n_keep_h:].repeat_interleave(attn.heads_per_group, dim=1)],
                            dim=1,
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

            # 3. Fused Chunk-Local `64 x 128` RoPE Syntax Attention
            if use_triton_gpu and attn.heads_per_group == 3 and G == 4:
                _vault_v2_5_local_syntax_decode_kernel[(B * G,)](
                    ls.local_k_win,
                    ls.local_v_win,
                    q_norm.squeeze(2).contiguous(),
                    k_norm.squeeze(2).contiguous(),
                    v.squeeze(2).contiguous(),
                    lp["cos_q_flat"],
                    lp["sin_q_flat"],
                    lp["cos_k_flat"],
                    lp["sin_k_flat"],
                    lp["local_bias_flat"],
                    t_buf,
                    y_syn_buf,
                    OUT_BF16=(dtype == torch.bfloat16),
                    num_warps=4,
                )
                y_syntax = y_syn_buf if dtype == comp_dt else y_syn_buf.to(dtype)
            else:
                rk = torch.cat([ls.local_k_win[:, :, C:, :], torch.zeros_like(ls.local_k_win[:, :, :C, :])], dim=2)
                rv = torch.cat([ls.local_v_win[:, :, C:, :], torch.zeros_like(ls.local_v_win[:, :, :C, :])], dim=2)
                ls.local_k_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), rk, ls.local_k_win))
                ls.local_v_win.copy_(torch.where(do_roll.view(1, 1, 1, 1), rv, ls.local_v_win))
                idx_loc = (C + q_in_t).view(1, 1, 1, 1).expand(B, G, 1, D)
                ls.local_k_win.scatter_(2, idx_loc, k_norm)
                ls.local_v_win.scatter_(2, idx_loc, v)
                cos_q_t = attn.rope_cos_q.index_select(3, q_in_t)
                sin_q_t = attn.rope_sin_q.index_select(3, q_in_t)
                q_rot = _apply_local_rope(q_norm.unsqueeze(2), cos_q_t, sin_q_t).squeeze(2)
                k_rot = _apply_local_rope(ls.local_k_win.unsqueeze(2), attn.rope_cos_k, attn.rope_sin_k).squeeze(2)
                q_gm = q_rot.view(B, G, attn.heads_per_group, D)
                logits_loc = (torch.matmul(q_gm, k_rot.transpose(-1, -2)).float() * inv_sqrt_d).view(
                    B, attn.n_q_heads, 1, 2 * C
                )
                mask_loc = attn.local_decay_mask[:, :, 0].index_select(2, q_in_t) > 0.0
                bias_loc = attn._local_attn_bias[:, :, 0].index_select(2, q_in_t)
                logits_loc = torch.where(mask_loc, logits_loc + bias_loc, torch.full_like(logits_loc, -1e9))
                probs_loc = F.softmax(logits_loc, dim=-1).to(dtype).view(B, G, attn.heads_per_group, 2 * C)
                y_syntax = torch.matmul(probs_loc, ls.local_v_win).view(B, attn.n_q_heads, 1, D)

            # 4. O(1) Stationary Bias-Corrected Causal Leaky-EMA GTRM
            ls.v_ema.mul_(gamma_ema).add_(v.float(), alpha=alpha_ema)
            v_mean = (ls.v_ema * inv_ema_bc).to(dtype)
            v_macro = attn.gtrm_proj(v_mean).repeat_interleave(attn.heads_per_group, dim=1)

            y_keep = (
                lp["v_gate_keep"] * y_vault[:, :n_keep_h]
                + lp["s_gate_keep"] * y_syntax[:, :n_keep_h]
                + lp["g_gate_keep"] * v_macro[:, :n_keep_h]
            )
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
            x_mlp = x_n2.to(comp_dt) if x_n2.dtype != comp_dt else x_n2
            mlp_h1, mlp_h2 = F.linear(x_mlp, lp["mlp_w12"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(mlp_h1) * mlp_h2, lp["mlp_w3"])

        t_buf.add_(1)
        h_out = F.rms_norm(x.float(), (d_m,), eps=1e-6).to(x.dtype) * model.norm_f.weight
        return model.lm_head(h_out) if return_logits else h_out

    @classmethod
    @torch.no_grad()
    def decode_step(
        cls,
        model: VAULTLMv2_5,
        token_id: torch.Tensor,
        cache: VAULTv2_5StreamCache,
        step_ratio: float = 1.0,
        return_logits: bool = True,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
    ) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        B, L_one = token_id.shape
        assert L_one == 1 and B == cache.batch_size

        dev = token_id.device
        if use_cudagraph and _HAS_TRITON and dev.type == "cuda" and not torch.is_grad_enabled():
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
            runners = getattr(model, "_decode_graph_runners_v2_5", None)
            if runners is None:
                runners = {}
                model._decode_graph_runners_v2_5 = runners
            stale_keys = [k for k in runners if k != g_key]
            for sk in stale_keys:
                runners.pop(sk).release_owner(banks)
            runner = runners.get(g_key)
            if runner is None:
                runner = _DecodeCUDAGraphRunnerV2_5(
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
        model: VAULTLMv2_5,
        input_ids: torch.Tensor,
        segment_size: int = 4096,
        cache: Optional[VAULTv2_5StreamCache] = None,
        step_ratio: float = 1.0,
        tail_logits_len: int = 1,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
        populate_decode_rings: bool = True,
    ) -> Tuple[torch.Tensor, VAULTv2_5StreamCache]:
        if model.training:
            model.eval()
        B, L = input_ids.shape
        assert L >= 1
        C = model.chunk_size
        assert segment_size >= C and segment_size % C == 0
        if cache is None:
            cache = cls.init_cache(model, batch_size=B)

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
            is_final_aligned = nxt_end == seg_end_all
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
        model: VAULTLMv2_5,
        prefix_ids: List[int],
        max_new_tokens: int,
        segment_size: int = 4096,
        stop_ids: Tuple[int, ...] = (),
        tok=None,
        min_chars_stop: int = 0,
        occ_alpha: float = 0.5,
        use_cudagraph: bool = True,
    ) -> Tuple[List[int], VAULTv2_5StreamCache]:
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
