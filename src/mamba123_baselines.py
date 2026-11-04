# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Verified Baseline Wrappers for VAULT v2.2 Head-to-Head Evaluation:
#   1. Mamba-1 (130M, state-spaces/mamba-130m-hf) with exact parallel associative scan (zero Python token loops)
#   2. Mamba-2 (130M, state-spaces/mamba2-130m) via vectorized SSD Mamba2ForCausalLM
#   3. Mamba-3 MIMO (187M, state-spaces/mamba3-mimo-187m) with exact Exponential-Trapezoidal +
#      Complex Data-Dependent 4-Slice RoPE + Rank-4 MIMO Chunkwise SSD (Lahoti, Kolter, Dao & Gu, 2026)
#   4. Transformer (135M, HuggingFaceTB/SmolLM-135M) with ROCm Flash/Mem-Efficient SDPA & KV-cache telemetry

import math
import os
import types
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Mamba2Config, Mamba2ForCausalLM


import triton
import triton.language as tl


# ==============================================================================
# 1A. MAMBA-1 FUSED TRITON REGISTER SCAN (FWD + EXACT REVERSE-ADJOINT BWD)
# ==============================================================================

@triton.jit
def _mamba1_fwd_kernel(
    u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
    y_ptr, h_ckpt_ptr,
    L, D_in, num_ckpts,
    SAVE_CKPT: tl.constexpr,
    CKPT_STEP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_in
    offs_n = tl.arange(0, 16)

    A_dn = tl.load(A_ptr + offs_d[:, None] * 16 + offs_n[None, :], mask=mask_d[:, None], other=0.0).to(tl.float32)
    D_d = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    h = tl.zeros([BLOCK_D, 16], dtype=tl.float32)

    base_bld = b_idx * L * D_in
    base_bln = b_idx * L * 16

    for t in range(0, L):
        if SAVE_CKPT:
            if (t % CKPT_STEP) == 0:
                c_idx = t // CKPT_STEP
                ckpt_off = ((b_idx * num_ckpts + c_idx) * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
                tl.store(h_ckpt_ptr + ckpt_off, h, mask=mask_d[:, None])

        off_ld = base_bld + t * D_in + offs_d
        off_ln = base_bln + t * 16 + offs_n

        u_t = tl.load(u_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        z_t = tl.load(z_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        B_t = tl.load(B_ptr + off_ln).to(tl.float32)
        C_t = tl.load(C_ptr + off_ln).to(tl.float32)

        a_t = tl.exp(tl.minimum(dt_t[:, None] * A_dn, 0.0))
        h = a_t * h + (dt_t * u_t)[:, None] * B_t[None, :]
        s_t = tl.sum(h * C_t[None, :], axis=1) + D_d * u_t
        sig_z = tl.sigmoid(z_t)
        y_t = s_t * (z_t * sig_z)
        tl.store(y_ptr + off_ld, y_t, mask=mask_d)


@triton.jit
def _mamba1_replay_chunk_kernel(
    u_ptr, dt_ptr, A_ptr, B_ptr, h_ckpt_ptr, h_prev_buf_ptr,
    L, D_in, num_ckpts, c_idx, chunk_start, chunk_len,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_in
    offs_n = tl.arange(0, 16)

    A_dn = tl.load(A_ptr + offs_d[:, None] * 16 + offs_n[None, :], mask=mask_d[:, None], other=0.0).to(tl.float32)
    ckpt_off = ((b_idx * num_ckpts + c_idx) * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
    h = tl.load(h_ckpt_ptr + ckpt_off, mask=mask_d[:, None], other=0.0).to(tl.float32)

    base_bld = b_idx * L * D_in
    base_bln = b_idx * L * 16

    for rel_t in range(0, chunk_len):
        buf_off = ((b_idx * chunk_len + rel_t) * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
        tl.store(h_prev_buf_ptr + buf_off, h, mask=mask_d[:, None])

        t = chunk_start + rel_t
        off_ld = base_bld + t * D_in + offs_d
        off_ln = base_bln + t * 16 + offs_n
        u_t = tl.load(u_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        B_t = tl.load(B_ptr + off_ln).to(tl.float32)
        a_t = tl.exp(tl.minimum(dt_t[:, None] * A_dn, 0.0))
        h = a_t * h + (dt_t * u_t)[:, None] * B_t[None, :]


@triton.jit
def _mamba1_bwd_chunk_kernel(
    u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, dy_ptr,
    h_prev_buf_ptr, dh_carry_ptr,
    du_ptr, ddt_ptr, dA_part_ptr, dB_part_ptr, dC_part_ptr, dD_part_ptr, dz_ptr,
    B_sz, L, D_in, chunk_start, chunk_len,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(0)
    pid_d = tl.program_id(1)
    num_d_blocks = tl.num_programs(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_in
    offs_n = tl.arange(0, 16)

    A_dn = tl.load(A_ptr + offs_d[:, None] * 16 + offs_n[None, :], mask=mask_d[:, None], other=0.0).to(tl.float32)
    D_d = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    carry_off = (b_idx * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
    dh = tl.load(dh_carry_ptr + carry_off, mask=mask_d[:, None], other=0.0).to(tl.float32)

    dA_acc = tl.zeros([BLOCK_D, 16], dtype=tl.float32)
    dD_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    base_bld = b_idx * L * D_in
    base_bln = b_idx * L * 16
    base_part_bln = (pid_d * B_sz + b_idx) * L * 16

    for rev_i in range(0, chunk_len):
        rel_t = chunk_len - 1 - rev_i
        t = chunk_start + rel_t
        off_ld = base_bld + t * D_in + offs_d
        off_ln = base_bln + t * 16 + offs_n

        u_t = tl.load(u_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        z_t = tl.load(z_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        dy_t = tl.load(dy_ptr + off_ld, mask=mask_d, other=0.0).to(tl.float32)
        B_t = tl.load(B_ptr + off_ln).to(tl.float32)
        C_t = tl.load(C_ptr + off_ln).to(tl.float32)

        buf_off = ((b_idx * chunk_len + rel_t) * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
        h_prev = tl.load(h_prev_buf_ptr + buf_off, mask=mask_d[:, None], other=0.0).to(tl.float32)

        a_t = tl.exp(tl.minimum(dt_t[:, None] * A_dn, 0.0))
        h_t = a_t * h_prev + (dt_t * u_t)[:, None] * B_t[None, :]
        s_t = tl.sum(h_t * C_t[None, :], axis=1) + D_d * u_t

        sig_z = tl.sigmoid(z_t)
        silu_z = z_t * sig_z
        ds_t = dy_t * silu_z
        dz_t = dy_t * s_t * sig_z * (1.0 + z_t * (1.0 - sig_z))

        dD_acc += ds_t * u_t
        dC_n = tl.sum(tl.where(mask_d[:, None], ds_t[:, None] * h_t, 0.0), axis=0)
        tl.store(dC_part_ptr + base_part_bln + t * 16 + offs_n, dC_n)

        dh = dh + ds_t[:, None] * C_t[None, :]
        da_t = dh * h_prev
        d_adt = da_t * a_t
        dA_acc += d_adt * dt_t[:, None]

        db_scalar = tl.sum(dh * B_t[None, :], axis=1)
        ddt_t = tl.sum(d_adt * A_dn, axis=1) + db_scalar * u_t
        du_t = ds_t * D_d + db_scalar * dt_t
        dB_n = tl.sum(tl.where(mask_d[:, None], dh * (dt_t * u_t)[:, None], 0.0), axis=0)
        tl.store(dB_part_ptr + base_part_bln + t * 16 + offs_n, dB_n)

        tl.store(du_ptr + off_ld, du_t, mask=mask_d)
        tl.store(ddt_ptr + off_ld, ddt_t, mask=mask_d)
        tl.store(dz_ptr + off_ld, dz_t, mask=mask_d)

        dh = dh * a_t

    tl.store(dh_carry_ptr + carry_off, dh, mask=mask_d[:, None])

    # Accumulate chunk contributions to per-batch A & D partials
    dA_off = (b_idx * D_in + offs_d[:, None]) * 16 + offs_n[None, :]
    prev_dA = tl.load(dA_part_ptr + dA_off, mask=mask_d[:, None], other=0.0)
    tl.store(dA_part_ptr + dA_off, prev_dA + dA_acc, mask=mask_d[:, None])

    dD_off = b_idx * D_in + offs_d
    prev_dD = tl.load(dD_part_ptr + dD_off, mask=mask_d, other=0.0)
    tl.store(dD_part_ptr + dD_off, prev_dD + dD_acc, mask=mask_d)


class _Mamba1TritonScanFn(torch.autograd.Function):
    CKPT_STEP = 512
    BLOCK_D = 64

    @staticmethod
    def forward(ctx, u, dt, A, B_mat, C_mat, D_skip, gate):
        # u, dt, gate: (B, L, D_in); A: (D_in, 16); B_mat, C_mat: (B, L, 16); D_skip: (D_in,)
        orig_dtype = u.dtype
        B_sz, L, D_in = u.shape
        u_c = u.contiguous()
        dt_c = dt.contiguous()
        A_c = A.float().contiguous()
        B_c = B_mat.contiguous()
        C_c = C_mat.contiguous()
        D_c = D_skip.float().contiguous()
        z_c = gate.contiguous()

        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        num_ckpts = (L + _Mamba1TritonScanFn.CKPT_STEP - 1) // _Mamba1TritonScanFn.CKPT_STEP
        h_ckpt = (
            torch.empty((B_sz, num_ckpts, D_in, 16), device=u.device, dtype=torch.float32)
            if need_grad
            else torch.empty((1,), device=u.device, dtype=torch.float32)
        )
        y = torch.empty((B_sz, L, D_in), device=u.device, dtype=orig_dtype)

        grid = (B_sz, triton.cdiv(D_in, _Mamba1TritonScanFn.BLOCK_D))
        _mamba1_fwd_kernel[grid](
            u_c, dt_c, A_c, B_c, C_c, D_c, z_c,
            y, h_ckpt,
            L, D_in, num_ckpts,
            SAVE_CKPT=need_grad,
            CKPT_STEP=_Mamba1TritonScanFn.CKPT_STEP,
            BLOCK_D=_Mamba1TritonScanFn.BLOCK_D,
        )
        if need_grad:
            ctx.save_for_backward(u_c, dt_c, A_c, B_c, C_c, D_c, z_c, h_ckpt)
            ctx.orig_dtype = orig_dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        u_c, dt_c, A_c, B_c, C_c, D_c, z_c, h_ckpt = ctx.saved_tensors
        orig_dtype = ctx.orig_dtype
        B_sz, L, D_in = u_c.shape
        ckpt_step = _Mamba1TritonScanFn.CKPT_STEP
        block_d = _Mamba1TritonScanFn.BLOCK_D
        num_ckpts = h_ckpt.size(1)
        num_d_blocks = triton.cdiv(D_in, block_d)

        dy_c = dy.contiguous()
        du = torch.empty((B_sz, L, D_in), device=u_c.device, dtype=torch.float32)
        ddt = torch.empty((B_sz, L, D_in), device=u_c.device, dtype=torch.float32)
        dz = torch.empty((B_sz, L, D_in), device=u_c.device, dtype=torch.float32)
        dA_part = torch.zeros((B_sz, D_in, 16), device=u_c.device, dtype=torch.float32)
        dD_part = torch.zeros((B_sz, D_in), device=u_c.device, dtype=torch.float32)
        dB_part = torch.empty((num_d_blocks, B_sz, L, 16), device=u_c.device, dtype=torch.float32)
        dC_part = torch.empty((num_d_blocks, B_sz, L, 16), device=u_c.device, dtype=torch.float32)
        dh_carry = torch.zeros((B_sz, D_in, 16), device=u_c.device, dtype=torch.float32)
        h_prev_buf = torch.empty((B_sz, min(L, ckpt_step), D_in, 16), device=u_c.device, dtype=torch.float32)

        grid = (B_sz, num_d_blocks)
        for c_idx in range(num_ckpts - 1, -1, -1):
            c_start = c_idx * ckpt_step
            c_len = min(ckpt_step, L - c_start)
            _mamba1_replay_chunk_kernel[grid](
                u_c, dt_c, A_c, B_c, h_ckpt, h_prev_buf,
                L, D_in, num_ckpts, c_idx, c_start, c_len,
                BLOCK_D=block_d,
            )
            _mamba1_bwd_chunk_kernel[grid](
                u_c, dt_c, A_c, B_c, C_c, D_c, z_c, dy_c,
                h_prev_buf, dh_carry,
                du, ddt, dA_part, dB_part, dC_part, dD_part, dz,
                B_sz, L, D_in, c_start, c_len,
                BLOCK_D=block_d,
            )

        dA = dA_part.sum(dim=0).to(A_c.dtype)
        dD = dD_part.sum(dim=0).to(D_c.dtype)
        dB = dB_part.sum(dim=0).to(B_c.dtype)
        dC = dC_part.sum(dim=0).to(C_c.dtype)
        return du.to(orig_dtype), ddt.to(dt_c.dtype), dA, dB, dC, dD, dz.to(z_c.dtype)


def mamba1_triton_scan(
    hidden_states: torch.Tensor,  # (B, D_in, L)
    dt: torch.Tensor,             # (B, D_in, L)
    A: torch.Tensor,              # (D_in, 16)
    B_mat: torch.Tensor,          # (B, 16, L)
    C_mat: torch.Tensor,          # (B, 16, L)
    D_skip: torch.Tensor,         # (D_in,)
    gate: torch.Tensor,           # (B, D_in, L)
) -> torch.Tensor:
    if not hidden_states.is_cuda or A.size(1) != 16:
        return mamba1_parallel_scan(hidden_states, dt, A, B_mat, C_mat, D_skip, gate)
    y = _Mamba1TritonScanFn.apply(
        hidden_states.transpose(1, 2),
        dt.transpose(1, 2),
        A,
        B_mat.transpose(1, 2),
        C_mat.transpose(1, 2),
        D_skip,
        gate.transpose(1, 2),
    )
    return y.transpose(1, 2)


# ==============================================================================
# 1B. REFERENCE MAMBA-1 BLOCK-PARALLEL ASSOCIATIVE SCAN (CPU / PARITY CHECK)
# ==============================================================================

def mamba1_parallel_scan(
    hidden_states: torch.Tensor,  # (B, D_in, L) after SiLU(conv1d)
    dt: torch.Tensor,             # (B, D_in, L) positive float32/bf16
    A: torch.Tensor,              # (D_in, N) negative float32
    B_mat: torch.Tensor,          # (B, N, L)
    C_mat: torch.Tensor,          # (B, N, L)
    D_skip: torch.Tensor,         # (D_in,)
    gate: torch.Tensor,           # (B, D_in, L) pre-SiLU z
    block_len: int = 1024,
) -> torch.Tensor:
    orig_dtype = hidden_states.dtype
    B_sz, D_in, L = hidden_states.shape
    N = A.size(1)

    u_f = hidden_states.float().transpose(1, 2)
    dt_f = dt.float().transpose(1, 2)
    B_f = B_mat.float().transpose(1, 2)
    C_f = C_mat.float().transpose(1, 2)
    A_f = A.float().view(1, 1, D_in, N)

    h_prev = torch.zeros((B_sz, 1, D_in, N), device=hidden_states.device, dtype=torch.float32)
    y_blocks = []

    for start in range(0, L, block_len):
        end = min(start + block_len, L)
        cur_len = end - start
        dt_blk = dt_f[:, start:end]
        u_blk = u_f[:, start:end]
        B_blk = B_f[:, start:end]
        C_blk = C_f[:, start:end]

        a = torch.exp((dt_blk.unsqueeze(-1) * A_f).clamp(max=0.0))
        b = (dt_blk * u_blk).unsqueeze(-1) * B_blk.unsqueeze(2)

        b0 = b[:, :1] + a[:, :1] * h_prev
        b = torch.cat([b0, b[:, 1:]], dim=1) if cur_len > 1 else b0

        n_steps = int(math.ceil(math.log2(max(1, cur_len))))
        for k in range(n_steps):
            stride = 1 << k
            if stride >= cur_len:
                break
            a_hi = a[:, stride:]
            b_hi = b[:, stride:]
            b = torch.cat([b[:, :stride], a_hi * b[:, :-stride] + b_hi], dim=1)
            a = torch.cat([a[:, :stride], a_hi * a[:, :-stride]], dim=1)

        h_prev = b[:, -1:]
        y_blk = torch.einsum("bldn,bln->bld", b, C_blk) + u_blk * D_skip.float().view(1, 1, D_in)
        y_blocks.append(y_blk)

    y_all = torch.cat(y_blocks, dim=1).transpose(1, 2)
    y_gated = y_all * F.silu(gate.float())
    return y_gated.to(orig_dtype)


def patch_hf_mamba1_fast_scan(model: nn.Module) -> nn.Module:
    """Replaces HuggingFace MambaMixer's slow Python token-loop fallback with fused `mamba1_triton_scan`."""
    for mod in model.modules():
        if mod.__class__.__name__ == "MambaMixer":
            def _fast_mixer_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
                batch_size, seq_len, _ = hidden_states.shape
                projected = self.in_proj(hidden_states).transpose(1, 2)  # (B, 2*D_in, L)
                h_in, gate = projected.chunk(2, dim=1)
                if attention_mask is not None:
                    h_in = h_in * attention_mask.unsqueeze(1)
                conv_out = self.act(self.conv1d(h_in)[..., :seq_len])     # (B, D_in, L)
                if attention_mask is not None:
                    conv_out = conv_out * attention_mask.unsqueeze(1)

                ssm_params = self.x_proj(conv_out.transpose(1, 2))        # (B, L, dt_rank + 2*N)
                dt_raw, B_mat, C_mat = torch.split(
                    ssm_params, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
                )
                dt = F.softplus(self.dt_proj(dt_raw).float()).transpose(1, 2)  # (B, D_in, L)
                A = -torch.exp(self.A_log.float())                             # (D_in, N)

                scan_out = mamba1_triton_scan(
                    conv_out, dt, A, B_mat.transpose(1, 2), C_mat.transpose(1, 2), self.D, gate
                )
                return self.out_proj(scan_out.transpose(1, 2))

            mod.forward = types.MethodType(_fast_mixer_forward, mod)
    return model


# ==============================================================================
# 1C. MAMBA-2 TENSOR-CORE GEMM SSD PATCH (REPLACES HF 7D FP32 ELEMENTWISE SCAN)
# ==============================================================================

def mamba2_fast_ssd(
    hidden_states: torch.Tensor,  # (B, L, H, P)
    dt: torch.Tensor,             # (B, L, H) pre-bias, pre-softplus
    A: torch.Tensor,              # (H,) negative float32
    B_mat: torch.Tensor,          # (B, L, G, N)
    C_mat: torch.Tensor,          # (B, L, G, N)
    chunk_size: int,
    D: torch.Tensor,              # (H,)
    dt_bias: torch.Tensor,        # (H,)
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
) -> torch.Tensor:
    B_sz, L, H, P = hidden_states.shape
    G, N = B_mat.size(2), B_mat.size(3)
    gemm_dtype = hidden_states.dtype

    dt_f = F.softplus(dt.float() + dt_bias.float()).clamp(min=dt_limit[0], max=dt_limit[1])  # (B, L, H)
    adt = dt_f * A.float().view(1, 1, H)                                                     # (B, L, H)

    x_dt = (hidden_states.float() * dt_f.unsqueeze(-1)).to(gemm_dtype)                       # (B, L, H, P)
    B_h = B_mat.repeat_interleave(H // G, dim=2).to(gemm_dtype)                              # (B, L, H, N)
    C_h = C_mat.repeat_interleave(H // G, dim=2).to(gemm_dtype)                              # (B, L, H, N)

    pad_L = (chunk_size - L % chunk_size) % chunk_size
    if pad_L > 0:
        x_dt = F.pad(x_dt, (0, 0, 0, 0, 0, pad_L))
        B_h = F.pad(B_h, (0, 0, 0, 0, 0, pad_L))
        C_h = F.pad(C_h, (0, 0, 0, 0, 0, pad_L))
        adt = F.pad(adt, (0, 0, 0, pad_L))

    Lp = x_dt.size(1)
    nc = Lp // chunk_size

    adt_c = adt.permute(0, 2, 1).reshape(B_sz, H, nc, chunk_size)                            # (B, H, nc, C)
    dA_cs = torch.cumsum(adt_c, dim=-1)
    L_intra = torch.exp(_segsum_causal(adt_c)).to(gemm_dtype)                                # (B, H, nc, C, C)

    q_c = C_h.permute(0, 2, 1, 3).reshape(B_sz, H, nc, chunk_size, N)
    k_c = B_h.permute(0, 2, 1, 3).reshape(B_sz, H, nc, chunk_size, N)
    v_c = x_dt.permute(0, 2, 1, 3).reshape(B_sz, H, nc, chunk_size, P)

    y_intra = ((q_c @ k_c.transpose(-1, -2)) * L_intra) @ v_c                                # (B, H, nc, C, P)

    decay_states = torch.exp((dA_cs[..., -1:] - dA_cs).clamp(max=0.0))
    k_to_end = (k_c.float() * decay_states.unsqueeze(-1)).to(gemm_dtype)
    chunk_states = k_to_end.transpose(-1, -2) @ v_c                                          # (B, H, nc, N, P)

    cs_chunks = torch.cumsum(dA_cs[..., -1], dim=-1)
    cs_prev = F.pad(cs_chunks[..., :-1], (1, 0), value=0.0)
    strict_tril = torch.tril(torch.ones(nc, nc, dtype=torch.bool, device=hidden_states.device), diagonal=-1)
    inter_decay = torch.exp((cs_prev[..., :, None] - cs_chunks[..., None, :]).masked_fill(~strict_tril, -torch.inf)).to(gemm_dtype)
    entering_states = torch.einsum("bhck,bhknp->bhcnp", inter_decay, chunk_states)

    q_from_start = (q_c.float() * torch.exp(dA_cs.clamp(max=0.0)).unsqueeze(-1)).to(gemm_dtype)
    y_inter = q_from_start @ entering_states

    y = (y_intra + y_inter).float().reshape(B_sz, H, Lp, P)[:, :, :L, :].permute(0, 2, 1, 3)
    y = y + D.float().view(1, 1, H, 1) * hidden_states.float()
    return y


def patch_hf_mamba2_fast_ssd(model: nn.Module) -> nn.Module:
    """Replaces HuggingFace Mamba2Mixer's slow 7D FP32 broadcast scan with Tensor-Core GEMM `mamba2_fast_ssd`."""
    for mod in model.modules():
        if mod.__class__.__name__ == "Mamba2Mixer":
            def _fast_mamba2_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                batch_size, seq_len, _ = hidden_states.shape
                dtype = hidden_states.dtype
                if attention_mask is not None:
                    hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                projected_states = self.in_proj(hidden_states)
                gate, hidden_states_B_C, dt = projected_states.split(
                    [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )
                conv_out = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len]).transpose(1, 2)
                if attention_mask is not None:
                    conv_out = conv_out * attention_mask.unsqueeze(-1)
                x_ssm, B_ssm, C_ssm = torch.split(
                    conv_out,
                    [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
                    dim=-1,
                )
                A = -torch.exp(self.A_log.float())
                scan_out = mamba2_fast_ssd(
                    x_ssm.view(batch_size, seq_len, self.num_heads, self.head_dim),
                    dt,
                    A,
                    B_ssm.view(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                    C_ssm.view(batch_size, seq_len, self.n_groups, self.ssm_state_size),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    dt_bias=self.dt_bias,
                    dt_limit=self.time_step_limit,
                ).reshape(batch_size, seq_len, -1)
                normed = self.norm(scan_out, gate)
                return self.out_proj(normed.to(dtype))

            mod.forward = types.MethodType(_fast_mamba2_forward, mod)
    return model



# ==============================================================================
# 2. OFFICIAL MAMBA-3 MIMO (187M) MODULE & CHUNKWISE SSD ENGINE
# ==============================================================================

class Mamba3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight.float()
        return out.to(x.dtype)


def _heavy_tail_neg_A(dd_A: torch.Tensor, a_floor: float = 1e-4) -> torch.Tensor:
    af = dd_A.float()
    af_neg = torch.clamp(af, max=0.0)
    ht = torch.where(af >= 0.0, 1.0 + af, 1.0 / (1.0 - af_neg))
    return torch.clamp(-ht, max=-a_floor)


def _segsum_causal(x: torch.Tensor) -> torch.Tensor:
    T = x.size(-1)
    cs = torch.cumsum(x, dim=-1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    diff = torch.clamp(cs[..., :, None] - cs[..., None, :], max=0.0)
    return torch.where(mask, diff, torch.full_like(diff, -1e9))


def _rotate_4slice_mimo(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Official TileLang MIMO 4-slice RoPE (`rotary_dim_divisor=4`): rotates slice 0 against slice 2."""
    x4 = x.view(*x.shape[:-1], 4, x.size(-1) // 4)
    s0, s1, s2, s3 = x4.unbind(dim=-2)
    r0 = s0 * cos - s2 * sin
    r2 = s0 * sin + s2 * cos
    return torch.stack([r0, s1, r2, s3], dim=-2).reshape_as(x)


class Mamba3MIMOMixer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        mimo_rank: int = 4,
        rope_fraction: float = 0.5,
        a_floor: float = 1e-4,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.mimo_rank = mimo_rank
        self.n_angles = int(d_state * rope_fraction) // 2
        self.a_floor = a_floor
        self.chunk_size = chunk_size

        d_in_proj = (
            2 * self.d_inner
            + 2 * mimo_rank * ngroups * d_state
            + 3 * self.nheads
            + self.n_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.B_norm = Mamba3RMSNorm(d_state)
        self.C_norm = Mamba3RMSNorm(d_state)
        self.mimo_x = nn.Parameter(torch.full((self.nheads, mimo_rank, headdim), 1.0 / mimo_rank))
        self.mimo_z = nn.Parameter(torch.ones(self.nheads, mimo_rank, headdim))
        self.mimo_o = nn.Parameter(torch.full((self.nheads, mimo_rank, headdim), 1.0 / mimo_rank))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor, seg_len: int = 2048) -> torch.Tensor:
        B_sz, orig_L, _ = u.shape
        H, P, R, G, N, S = (
            self.nheads,
            self.headdim,
            self.mimo_rank,
            self.ngroups,
            self.d_state,
            self.n_angles,
        )
        C_sz = self.chunk_size

        proj = self.in_proj(u)
        gemm_dtype = torch.float32 if self.training else proj.dtype
        z_all, xv_all, B_raw_all, C_raw_all, dd_dt, dd_A, trap_raw, ang_raw = torch.split(
            proj, [self.d_inner, self.d_inner, R * G * N, R * G * N, H, H, H, S], dim=-1
        )

        dt = F.softplus(dd_dt.float() + self.dt_bias.float()).permute(0, 2, 1)  # (B, H, L)
        A = _heavy_tail_neg_A(dd_A, self.a_floor).permute(0, 2, 1)              # (B, H, L)
        adt = A * dt
        trap = trap_raw.float().permute(0, 2, 1)                                # (B, H, L)
        ang = ang_raw.float().unsqueeze(2).expand(B_sz, orig_L, H, S)           # (B, L, H, S)

        pad_L = (C_sz - orig_L % C_sz) % C_sz
        if pad_L > 0:
            z_all = F.pad(z_all, (0, 0, 0, pad_L))
            xv_all = F.pad(xv_all, (0, 0, 0, pad_L))
            B_raw_all = F.pad(B_raw_all, (0, 0, 0, pad_L))
            C_raw_all = F.pad(C_raw_all, (0, 0, 0, pad_L))
            dt = F.pad(dt, (0, pad_L))
            adt = F.pad(adt, (0, pad_L))
            trap = F.pad(trap, (0, pad_L))
            ang = F.pad(ang, (0, 0, 0, 0, 0, pad_L))

        Lp = dt.size(-1)
        dtheta = torch.tanh(ang) * (dt.permute(0, 2, 1).unsqueeze(-1) * math.pi)
        ang_cs_all = torch.cumsum(dtheta, dim=1)                                # (B, Lp, H, S)

        lam = torch.sigmoid(trap)
        shifted_gamma_all = F.pad(dt[:, :, 1:] * (1.0 - lam[:, :, 1:]), (0, 1), value=0.0)
        factor_all = dt * lam + shifted_gamma_all                               # (B, H, Lp)

        ssm_state = torch.zeros((B_sz, H, N, P), device=u.device, dtype=torch.float32)
        seg_outputs = []

        for s_start in range(0, Lp, seg_len):
            s_end = min(s_start + seg_len, Lp)
            s_len = s_end - s_start
            nc = s_len // C_sz
            bhn = B_sz * H * nc
            CR = C_sz * R

            z = z_all[:, s_start:s_end].view(B_sz, s_len, H, P).to(gemm_dtype)
            xv = xv_all[:, s_start:s_end].view(B_sz, s_len, H, P).to(gemm_dtype)
            B_n = self.B_norm(B_raw_all[:, s_start:s_end].view(B_sz, s_len, R, G, N)).to(gemm_dtype)
            C_n = self.C_norm(C_raw_all[:, s_start:s_end].view(B_sz, s_len, R, G, N)).to(gemm_dtype)

            q = C_n.repeat_interleave(H // G, dim=3).permute(0, 1, 3, 2, 4) + self.C_bias.to(gemm_dtype)[None, None]
            k = B_n.repeat_interleave(H // G, dim=3).permute(0, 1, 3, 2, 4) + self.B_bias.to(gemm_dtype)[None, None]

            ang_seg = ang_cs_all[:, s_start:s_end]
            cos_a = torch.cos(ang_seg).unsqueeze(3).to(gemm_dtype)
            sin_a = torch.sin(ang_seg).unsqueeze(3).to(gemm_dtype)
            q_rot = _rotate_4slice_mimo(q, cos_a, sin_a)
            k_rot = _rotate_4slice_mimo(k, cos_a, sin_a)

            factor = factor_all[:, :, s_start:s_end]
            shifted_gamma = shifted_gamma_all[:, :, s_start:s_end]
            k_fact = (k_rot.float() * factor.permute(0, 2, 1)[:, :, :, None, None]).to(gemm_dtype)

            v_up = xv.unsqueeze(-2) * self.mimo_x.to(gemm_dtype)[None, None]
            z_up = z.unsqueeze(-2) * self.mimo_z.to(gemm_dtype)[None, None]

            adt_c = adt[:, :, s_start:s_end].view(B_sz, H, nc, C_sz)
            dA_cs = torch.cumsum(adt_c, dim=-1).clamp(max=0.0)
            dA_rev = (torch.cumsum(adt_c.flip(-1), dim=-1).flip(-1) - adt_c).clamp(max=0.0)

            L_intra = torch.exp(_segsum_causal(adt_c)).repeat_interleave(R, -1).repeat_interleave(R, -2).to(gemm_dtype).reshape(bhn, CR, CR)
            q_c = q_rot.permute(0, 2, 1, 3, 4).reshape(bhn, CR, N).contiguous()
            k_c = k_fact.permute(0, 2, 1, 3, 4).reshape(bhn, CR, N).contiguous()
            v_c = v_up.permute(0, 2, 1, 3, 4).reshape(bhn, CR, P).contiguous()
            attn_intra = torch.bmm(q_c, k_c.transpose(1, 2).contiguous()) * L_intra
            y_intra = torch.bmm(attn_intra, v_c).view(B_sz, H, nc, CR, P)

            k_to_end = (
                k_fact.permute(0, 2, 1, 3, 4).view(B_sz, H, nc, C_sz, R, N).float()
                * torch.exp(dA_rev)[..., None, None]
            ).to(gemm_dtype).reshape(bhn, CR, N).contiguous()
            chunk_states = torch.bmm(k_to_end.transpose(1, 2).contiguous(), v_c).view(B_sz, H, nc, N, P)

            cs_chunks = torch.cumsum(dA_cs[..., -1], dim=-1)                    # (B, H, nc)
            cs_prev = F.pad(cs_chunks[..., :-1], (1, 0), value=0.0)
            strict_tril = torch.tril(torch.ones(nc, nc, dtype=torch.bool, device=u.device), diagonal=-1)
            inter_diff = torch.clamp(cs_prev[..., :, None] - cs_chunks[..., None, :], max=0.0)
            inter_decay = torch.exp(torch.where(strict_tril, inter_diff, torch.full_like(inter_diff, -1e9))).to(gemm_dtype)
            entering_states = torch.bmm(
                inter_decay.reshape(B_sz * H, nc, nc).contiguous(),
                chunk_states.reshape(B_sz * H, nc, N * P).contiguous(),
            ).view(B_sz, H, nc, N, P).float()
            if s_start > 0:
                entering_states = entering_states + ssm_state.unsqueeze(2) * torch.exp(cs_prev)[..., None, None]

            q_from_start = (
                q_c.view(B_sz, H, nc, CR, N).float() * torch.exp(dA_cs.repeat_interleave(R, dim=-1)).unsqueeze(-1)
            ).to(gemm_dtype).reshape(bhn, CR, N).contiguous()
            y_inter = torch.bmm(q_from_start, entering_states.to(gemm_dtype).reshape(bhn, N, P).contiguous()).view(B_sz, H, nc, CR, P)

            # Update segment carry state `ssm_state` for next segment (if any)
            if s_end < Lp:
                end_decay = torch.exp((cs_chunks[..., -1:] - cs_chunks).clamp(max=0.0))
                ssm_state = (
                    ssm_state * torch.exp(cs_chunks[..., -1])[..., None, None]
                    + (chunk_states.float() * end_decay[..., None, None]).sum(dim=2)
                )

            y = (y_intra + y_inter).view(B_sz, H, s_len, R, P).float()
            vh = v_up.permute(0, 2, 1, 3, 4).float()

            # Unroll R=4 trapezoidal diagonal correction (avoids >65535-batch rocBLAS gridDim.z bug on RDNA4!)
            qf = q.permute(0, 2, 1, 3, 4).float()
            kf = k.permute(0, 2, 1, 3, 4).float()
            diag_corr = torch.zeros_like(vh)
            for r_in in range(R):
                # dot over N -> (B, H, s_len, R)
                qk_r = (qf * kf[:, :, :, r_in : r_in + 1, :]).sum(dim=-1)
                diag_corr = diag_corr + qk_r.unsqueeze(-1) * vh[:, :, :, r_in : r_in + 1, :]
            y = y - diag_corr * shifted_gamma[:, :, :, None, None]

            y = y + self.D.float()[None, :, None, None, None] * vh
            y = y * F.silu(z_up.permute(0, 2, 1, 3, 4).float())
            seg_out = (
                (y * self.mimo_o.float()[None, :, None, :, :])
                .sum(dim=-2)
                .permute(0, 2, 1, 3)
                .to(gemm_dtype)
                .reshape(B_sz, s_len, self.d_inner)
            )
            seg_outputs.append(seg_out)

        m_out = (seg_outputs[0] if len(seg_outputs) == 1 else torch.cat(seg_outputs, dim=1))[:, :orig_L]
        return self.out_proj(m_out).to(u.dtype)


class Mamba3GatedMLP(nn.Module):
    def __init__(self, d_model: int = 768, d_ff: int = 1280):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class Mamba3Block(nn.Module):
    def __init__(self, d_model: int = 768, d_ff: int = 1280, chunk_size: int = 32):
        super().__init__()
        self.norm = Mamba3RMSNorm(d_model)
        self.mixer = Mamba3MIMOMixer(d_model=d_model, chunk_size=chunk_size)
        self.norm2 = Mamba3RMSNorm(d_model)
        self.mlp = Mamba3GatedMLP(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Mamba3Backbone(nn.Module):
    def __init__(self, vocab_size: int = 128256, d_model: int = 768, n_layer: int = 12, d_ff: int = 1280, chunk_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Mamba3Block(d_model=d_model, d_ff=d_ff, chunk_size=chunk_size) for _ in range(n_layer)])
        self.norm_f = Mamba3RMSNorm(d_model)


class Mamba3MIMOForCausalLM(nn.Module):
    """Exact architectural & parameter match for official `state-spaces/mamba3-mimo-187m`."""
    def __init__(
        self,
        vocab_size: int = 128256,
        d_model: int = 768,
        n_layer: int = 12,
        d_intermediate: int = 1264,
        chunk_size: int = 32,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        d_ff = ((d_intermediate + 127) // 128) * 128
        self.d_model = d_model
        self.n_layer = n_layer
        self.grad_checkpoint = grad_checkpoint
        self.backbone = Mamba3Backbone(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, d_ff=d_ff, chunk_size=chunk_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

    def count_parameters(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def encode_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.backbone.embedding(input_ids)
        for blk in self.backbone.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = blk(x)
        return self.backbone.norm_f(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.encode_hidden(input_ids))


# ==============================================================================
# 3. CLEAN MODEL + TOKENIZER LOADERS & PERSISTENT STATE FOOTPRINT CALCULATOR
# ==============================================================================

def load_mamba1_130m(device: str = "cuda:0", grad_checkpoint: bool = False) -> Tuple[nn.Module, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-130m-hf", local_files_only=True, dtype=torch.float32
    )
    patch_hf_mamba1_fast_scan(model)
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model.to(device), tok


def load_mamba2_130m(device: str = "cuda:0", grad_checkpoint: bool = False, fast_ssd: bool = True) -> Tuple[nn.Module, AutoTokenizer]:
    snap_root = "/home/phil/.cache/huggingface/hub/models--state-spaces--mamba2-130m/snapshots"
    snap_dir = os.path.join(snap_root, os.listdir(snap_root)[0])
    ckpt_path = os.path.join(snap_dir, "pytorch_model.bin")
    tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", local_files_only=True)

    cfg = Mamba2Config(
        vocab_size=50288,
        hidden_size=768,
        state_size=128,
        num_hidden_layers=24,
        num_heads=24,
        head_dim=64,
        expand=2,
        conv_kernel=4,
        n_groups=1,
        chunk_size=64,
        tie_word_embeddings=True,
        use_bias=False,
        use_conv_bias=True,
        residual_in_fp32=True,
        layer_norm_epsilon=1e-5,
        time_step_limit=(0.0, float("inf")),
    )
    raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hf_sd = {k.replace("backbone.embedding.weight", "backbone.embeddings.weight"): v.float() for k, v in raw_sd.items()}
    model = Mamba2ForCausalLM(cfg)
    model.load_state_dict(hf_sd, strict=True)
    if fast_ssd:
        patch_hf_mamba2_fast_ssd(model)
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model.to(device), tok


def load_mamba3_mimo_187m(device: str = "cuda:0", grad_checkpoint: bool = False) -> Tuple[Mamba3MIMOForCausalLM, AutoTokenizer]:
    bin_p = hf_hub_download("state-spaces/mamba3-mimo-187m", "pytorch_model.bin", local_files_only=True)
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B", local_files_only=True)
    raw_sd = torch.load(bin_p, map_location="cpu", weights_only=True)
    sd_fp32 = {k: v.float() for k, v in raw_sd.items()}
    model = Mamba3MIMOForCausalLM(
        vocab_size=128256, d_model=768, n_layer=12, d_intermediate=1264, chunk_size=32, grad_checkpoint=grad_checkpoint
    )
    model.load_state_dict(sd_fp32, strict=True)
    model.lm_head.weight = model.backbone.embedding.weight
    return model.to(device), tok


def load_transformer_smollm_135m(device: str = "cuda:0", grad_checkpoint: bool = False) -> Tuple[nn.Module, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M", local_files_only=True, dtype=torch.float32
    )
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model.to(device), tok


def forward_tail_logits(model: nn.Module, input_ids: torch.Tensor, tail_len: Optional[int] = None) -> torch.Tensor:
    """
    Computes backbone hidden states over the full `input_ids` sequence and projects only the final `tail_len`
    positions through `lm_head` (or all positions if `tail_len is None`), preventing multi-GB `32K x V` unused
    logit allocations during long-context prefill and NIAH evaluation across all 5 architectures.
    """
    if hasattr(model, "_encode_hidden"):
        # VAULTLMv2
        h = model._encode_hidden(input_ids, step_ratio=1.0)
        if tail_len is not None:
            h = h[:, -tail_len:, :]
        return model.lm_head(h)
    if hasattr(model, "encode_hidden"):
        # Mamba3MIMOForCausalLM
        h = model.encode_hidden(input_ids)
        if tail_len is not None:
            h = h[:, -tail_len:, :]
        return model.lm_head(h)
    if hasattr(model, "backbone") and hasattr(model, "lm_head"):
        # MambaForCausalLM (Mamba-1) and Mamba2ForCausalLM (Mamba-2)
        out = model.backbone(input_ids, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if tail_len is not None:
            h = h[:, -tail_len:, :]
        return model.lm_head(h)
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        # LlamaForCausalLM (SmolLM-135M Transformer)
        out = model.model(input_ids, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if tail_len is not None:
            h = h[:, -tail_len:, :]
        return model.lm_head(h)
    res = model(input_ids)
    logits = res.logits if hasattr(res, "logits") else res
    return logits[:, -tail_len:, :] if tail_len is not None else logits


def persistent_decode_state_mb(arch: str, seq_len: int) -> float:
    """
    Exact BF16/FP32 persistent autoregressive decode cache size (in MB) at context length `seq_len` (`B=1`).
    """
    if arch == "VAULT_v2_2_180M":
        # 16 layers: 4 vault groups x 88 cells x 64x64 (BF16) + 88 occ + 64-tok local syntax KV & W=64 induction delay ring + GTRM + conv
        per_layer_bytes = (
            4 * 88 * 64 * 64 * 2          # Fast-weight vaults S (2.75 MB/layer)
            + 4 * 88 * 4                  # Cell occupancy
            + 64 * (4 * 64 * 2) * 2       # 64-tok local RoPE K,V ring
            + 64 * (3 * (64 + 64 + 2)) * 2  # W_delay=64 induction write ring
            + 1024 * 4                    # Windowed CMS ring
        )
        return round(16 * per_layer_bytes / (1024 ** 2), 2)
    elif arch in ("VAULT_v2_3_180M", "VAULT_v2_4_180M"):
        # 16 layers: 4 vault groups x 220 cells x 64x64 (BF16) + 220 occ + 64-tok local syntax KV & W=64 induction delay ring + Windowed CMS ring
        per_layer_bytes = (
            4 * 220 * 64 * 64 * 2         # Fast-weight vaults S (6.875 MB/layer)
            + 4 * 220 * 4                 # Cell occupancy
            + 64 * (4 * 64 * 2) * 2       # 64-tok local RoPE K,V ring
            + 64 * (3 * (64 + 64 + 2)) * 2  # W_delay=64 induction write ring
            + 1024 * 4                    # Windowed CMS ring
        )
        return round(16 * per_layer_bytes / (1024 ** 2), 2)
    elif arch == "Mamba1_130M":
        # 24 layers: conv state (1536 x 4 x 2B) + SSM state (1536 x 16 x 4B FP32)
        per_layer_bytes = 1536 * 4 * 2 + 1536 * 16 * 4
        return round(24 * per_layer_bytes / (1024 ** 2), 2)
    elif arch == "Mamba2_130M":
        # 24 layers: conv state (1792 x 4 x 2B) + SSD state (24 x 64 x 128 x 4B FP32)
        per_layer_bytes = 1792 * 4 * 2 + 24 * 64 * 128 * 4
        return round(24 * per_layer_bytes / (1024 ** 2), 2)
    elif arch == "Mamba3_MIMO_187M":
        # 12 layers: SSM state (24 x 64 x 128 x 4B) + angle_state (24 x 32 x 4B) + trapezoidal k_state (4 x 24 x 128 x 2B) + v_state (24 x 64 x 2B)
        per_layer_bytes = (24 * 64 * 128 * 4) + (24 * 32 * 4) + (4 * 24 * 128 * 2) + (24 * 64 * 2)
        return round(12 * per_layer_bytes / (1024 ** 2), 2)
    elif arch == "Transformer_SmolLM_135M":
        # 30 layers, 3 KV heads x 64 head_dim x 2 (K & V) x 2 bytes (BF16) x seq_len -> O(L) linear growth!
        total_bytes = 30 * 2 * 3 * 64 * 2 * seq_len
        return round(total_bytes / (1024 ** 2), 2)
    elif arch == "Transformer_Matched_180M":
        # 16 layers, 4 KV heads x 64 head_dim x 2 (K & V) x 2 bytes (BF16) x seq_len -> O(L) linear growth!
        total_bytes = 16 * 2 * 4 * 64 * 2 * seq_len
        return round(total_bytes / (1024 ** 2), 2)
    raise ValueError(arch)
