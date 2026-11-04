# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 (`VAULTLMv2_5`): Post-10K Long-Horizon & Multi-Subword Upgrade of VAULT v2.4
# Built strictly from Philip Bartsch's owned IP (`vault`, `PRIME-Moment-Attention`, `BioNPC`,
# `gist-memory`, `buildbit-os`, `mamba1and2-to-3`) + standard PyTorch primitives.
#
# Architectural Upgrades over VAULT v2.4 (100% Isolated from Active v2.4 Files):
#   1. Stationary Causal Leaky-EMA GTRM (`alpha = 2^-11 = 1/2048`, `gamma = 2047/2048`):
#      Replaces `cumsum(v) / t` (which collapses by 64x at `t = 131,072` vs `t = 2,048`) with a
#      bias-corrected stationary causal exponential moving average computed via O(L) parallel 64x64
#      block matrix scans in prefill/training and O(1) register updates in streaming decode.
#   2. Dedicated 11-BPE-Subword (`m = 1..10`, `diff >= 11`) Extended `SalienceEntityBinderV2_5` Channel:
#      Preserves `Vault Group 1` (`1..4` subwords) and `Vault Group 2` (`1..7` subwords) bit-for-bit,
#      while routing `g2` (`m = 1..10`, odd-prime circular shifts `19, 29, 43, 59`) exclusively into
#      `Vault Group 3` via `ext_span_alpha`, resolving `8..11`-subword identical-prefix sibling entities
#      (e.g., `IonThruster-T21560` vs `IonThruster-T21904`) with strict write/read causality (`q - k >= 11`).
#   3. Shared-Prefix / Suffix Contrastive Choice-Margin Heads (`forward_mc_contrastive_loss` &
#      `forward_wg_contrastive_loss`) for zero-leakage multiple-choice calibration during SFT.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.flash_vault_v2_triton import causal_depthwise_conv1d, flash_vault_v2_chunkwise
from src.vault_lm_150m import FusedAPLNorm, RMSNorm, SwiGLUMLP, TernarySTEProjection
from src.vault_v2_model import (
    CausalShortConv1D,
    _apply_local_rope,
    _safe_orthogonal_,
    _shift_right_zero,
    compute_windowed_cms_aux,
    fft_bind_with_id,
    fft_circular_bind,
)

GTRM_EMA_ALPHA_DEFAULT: float = 1.0 / 2048.0  # Exact IEEE-754 power of two (2^-11)


def _vault_v2_5_cpu_chunkwise_autograd(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cell_q: torch.Tensor,
    cell_k: torch.Tensor,
    chunk_size: int = 64,
    v2_2_beta_occ: bool = True,
    prime_banks: Tuple[int, ...] = (47, 53, 59, 61),
) -> torch.Tensor:
    """
    Pure-PyTorch autograd-compatible Compact-WY Co-Prime chunkwise scan for `CPU` unit tests & smoke runs
    (`CUDA_VISIBLE_DEVICES=""`), mathematically bit-exact to `flash_vault_v2_chunkwise` (`v2_1_consensus=True`,
    `v2_2_beta_occ=True`). On GPU (`phi_q.is_cuda`), `VAULTv2_5Attention` dispatches directly to `flash_vault_v2_chunkwise`.
    """
    orig_dtype = v.dtype
    B, H, L_orig, D = phi_q.shape
    _, G, R, _ = cell_k.shape
    heads_per_group = H // G
    C = chunk_size
    total_cells = sum(int(p) for p in prime_banks)

    rem = L_orig % C
    pad_l = (C - rem) if rem != 0 else 0
    if pad_l > 0:
        phi_q = F.pad(phi_q, (0, 0, 0, pad_l))
        phi_k = F.pad(phi_k, (0, 0, 0, pad_l))
        v = F.pad(v, (0, 0, 0, pad_l))
        beta = F.pad(beta, (0, pad_l), value=0.0)
        cell_q = F.pad(cell_q, (0, pad_l), value=0)
        cell_k = F.pad(cell_k, (0, pad_l), value=0)

    L = phi_q.size(2)
    M = L // C
    pq = phi_q.float().view(B, H, M, C, D)
    pk = phi_k.float().view(B, G, M, C, D)
    vf = v.float().view(B, G, M, C, D)
    bf = beta.float().view(B, G, M, C)
    cq = cell_q.view(B, H, R, M, C)
    ck = cell_k.view(B, G, R, M, C)

    dev = v.device
    idx = torch.arange(C, device=dev)
    strict_lower = idx[:, None] > idx[None, :]
    causal = idx[:, None] >= idx[None, :]
    eye_c = torch.eye(C, device=dev, dtype=torch.float32)

    b_g = torch.arange(B, device=dev)[:, None, None, None]
    g_g = torch.arange(G, device=dev)[None, :, None, None]
    b_h = torch.arange(B, device=dev)[:, None, None, None]
    g_for_h = (torch.arange(H, device=dev) // heads_per_group)[None, :, None, None]

    S_state = torch.zeros((B, G, total_cells, D, D), device=dev, dtype=torch.float32)
    occ_state = torch.zeros((B, G, total_cells), device=dev, dtype=torch.float32)
    out_chunks = []

    for m in range(M):
        km = pk[:, :, m]
        vm = vf[:, :, m]
        bm = bf[:, :, m]
        ckm = ck[:, :, :, m]
        cqm = cq[:, :, :, m]
        qm = pq[:, :, m]

        kk = torch.matmul(km, km.transpose(-1, -2)).unsqueeze(2)
        same_cell = (ckm.unsqueeze(-1) == ckm.unsqueeze(-2)) & strict_lower
        T_mat = eye_c + torch.where(same_cell, bm.unsqueeze(2).unsqueeze(-1) * kk, 0.0)
        bv = (bm.unsqueeze(-1) * vm).unsqueeze(2).expand(B, G, R, C, D)
        bk = (bm.unsqueeze(-1) * km).unsqueeze(2).expand(B, G, R, C, D)
        uw = torch.linalg.solve_triangular(T_mat, torch.cat([bv, bk], dim=-1), upper=False, unitriangular=True)
        u_tile, w_tile = uw[..., :D], uw[..., D:]

        S_ck = S_state[b_g, g_g, ckm]
        delta_tile = u_tile - torch.einsum("bgrcvd,bgrcd->bgrcv", S_ck, w_tile)

        S_cq = S_state[b_h, g_for_h, cqm]
        occ_prior = occ_state.detach()[b_h, g_for_h, cqm]
        y_cross = torch.einsum("bhrcvd,bhcd->bhrcv", S_cq, qm)

        km_h = km.repeat_interleave(heads_per_group, dim=1)
        ckm_h = ckm.repeat_interleave(heads_per_group, dim=1)
        bm_h = bm.detach().repeat_interleave(heads_per_group, dim=1)
        delta_h = delta_tile.repeat_interleave(heads_per_group, dim=1)

        qk_raw = torch.matmul(qm, km_h.transpose(-1, -2)).unsqueeze(2)
        same_qk = (cqm.unsqueeze(-1) == ckm_h.unsqueeze(-2)) & causal
        y_intra = torch.matmul(torch.where(same_qk, qk_raw, 0.0), delta_h)

        occ_intra = torch.where(same_qk, bm_h.unsqueeze(2).unsqueeze(-2), 0.0).sum(dim=-1)
        n_tot = occ_prior + occ_intra
        w_r = torch.where(n_tot > 1e-3, torch.rsqrt(1.0 + n_tot), torch.zeros_like(n_tot))
        w_sum = w_r.sum(dim=2, keepdim=True)
        w_norm = torch.where(w_sum > 1e-6, w_r / w_sum.clamp(min=1e-6), torch.full_like(w_r, 1.0 / float(R)))
        out_chunks.append((w_norm.unsqueeze(-1) * (y_cross + y_intra)).sum(dim=2))

        upd_flat = torch.einsum("bgrcv,bgrcd->bgrcvd", delta_tile, km.unsqueeze(2)).reshape(B * G, R * C, D * D)
        ck_idx = ckm.reshape(B * G, R * C).unsqueeze(-1).expand(-1, -1, D * D)
        S_next_flat = S_state.view(B * G, total_cells, D * D).scatter_add(1, ck_idx, upd_flat)
        S_state = S_next_flat.view(B, G, total_cells, D, D)

        b_add = (
            bm.detach().unsqueeze(2).expand(B, G, R, C)
            if v2_2_beta_occ
            else torch.ones_like(ckm, dtype=torch.float32)
        ).reshape(B * G, R * C)
        occ_state = occ_state.view(B * G, total_cells).scatter_add(1, ckm.reshape(B * G, R * C), b_add).view(
            B, G, total_cells
        )

    out = torch.cat(out_chunks, dim=2)
    return out[:, :, :L_orig, :].to(orig_dtype)


_EMA_KERNEL_CACHE: Dict[Tuple[torch.device, int, float, float], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_ema_block_kernels(
    device: torch.device, block_len: int, decay: float, step_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, int(block_len), float(decay), float(step_scale))
    cached = _EMA_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(block_len, device=device, dtype=torch.float64)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (C, C)
    t_mat = torch.where(
        diff >= 0,
        float(step_scale) * torch.pow(torch.tensor(decay, device=device, dtype=torch.float64), diff.clamp(min=0)),
        torch.zeros_like(diff),
    ).to(torch.float32)
    d_vec = torch.pow(
        torch.tensor(decay, device=device, dtype=torch.float64),
        torch.arange(1, block_len + 1, device=device, dtype=torch.float64),
    ).to(torch.float32).view(block_len, 1)
    _EMA_KERNEL_CACHE[key] = (t_mat, d_vec)
    return t_mat, d_vec


def _parallel_decay_scan_dim2(
    x_f: torch.Tensor,
    decay: float,
    step_scale: float = 1.0,
    block_len: int = 64,
) -> torch.Tensor:
    """
    Exact O(L) parallel hierarchical block matrix scan along `dim=2` for `(B, G, L, D)` FP32 tensors:
        y[:, :, t, :] = decay * y[:, :, t - 1, :] + step_scale * x_f[:, :, t, :]   (with y[:, :, -1, :] = 0)
    Uses zero token-level Python loops across arbitrary sequence lengths (`2K` to `256K+`).
    """
    B, G, L, D = x_f.shape
    if L == 0:
        return x_f
    if L <= block_len:
        t_mat, _ = _get_ema_block_kernels(x_f.device, L, decay, step_scale)
        return torch.matmul(t_mat, x_f)

    rem = L % block_len
    pad_r = (block_len - rem) if rem != 0 else 0
    xp = F.pad(x_f, (0, 0, 0, pad_r)) if pad_r > 0 else x_f
    M = xp.size(2) // block_len
    xb = xp.view(B, G, M, block_len, D)

    t_mat, d_vec = _get_ema_block_kernels(x_f.device, block_len, decay, step_scale)
    u_loc = torch.matmul(t_mat, xb)  # (B, G, M, block_len, D)
    if M > 1:
        s_end = u_loc[:, :, :-1, -1, :]  # (B, G, M - 1, D)
        chunk_decay = float(decay) ** block_len
        e_prev = _parallel_decay_scan_dim2(s_end, decay=chunk_decay, step_scale=1.0, block_len=block_len)
        carry = F.pad(e_prev, (0, 0, 1, 0)).unsqueeze(3)  # (B, G, M, 1, D)
        u_loc = u_loc + carry * d_vec
    out = u_loc.reshape(B, G, M * block_len, D)
    return out[:, :, :L, :] if pad_r > 0 else out


def compute_ema_bias_correction(
    length: int,
    device: torch.device,
    alpha: float = GTRM_EMA_ALPHA_DEFAULT,
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Computes `1 / (1 - (1 - alpha)^(t + 1))` in FP64 -> FP32 for `t in [start_pos, start_pos + length)`.
    Returns shape `(1, 1, length, 1)`.
    """
    gamma = 1.0 - float(alpha)
    steps = torch.arange(start_pos + 1, start_pos + length + 1, device=device, dtype=torch.float64)
    denom = (1.0 - torch.pow(gamma, steps)).clamp(min=1e-12)
    return (1.0 / denom).to(torch.float32).view(1, 1, length, 1)


def chunked_causal_ema_dim2(
    v_f: torch.Tensor,
    alpha: float = GTRM_EMA_ALPHA_DEFAULT,
    block_len: int = 64,
    ema_bias_corr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Bias-corrected stationary causal EMA along `dim=2` for `(B, G, L, D)` FP32 tensors:
        m_t = (1 - alpha) * m_{t-1} + alpha * v_t
        v_hat_t = m_t / (1 - (1 - alpha)^(t + 1))
    Matches `cumsum(v) / (t + 1)` at `t = 0` (`v_hat_0 == v_0`) while remaining 100% stationary at `128K+`.
    """
    L = v_f.size(2)
    gamma = 1.0 - float(alpha)
    m_seq = _parallel_decay_scan_dim2(v_f, decay=gamma, step_scale=float(alpha), block_len=block_len)
    if ema_bias_corr is None or ema_bias_corr.size(2) != L:
        ema_bias_corr = compute_ema_bias_correction(L, v_f.device, alpha=alpha, start_pos=0)
    return m_seq * ema_bias_corr


@torch.no_grad()
def compute_v2_5_shared_aux(
    x0_norm: torch.Tensor,
    binder: "SalienceEntityBinderV2_5",
    ema_alpha: float = GTRM_EMA_ALPHA_DEFAULT,
) -> Dict[str, object]:
    """
    Precomputes all layer-invariant `x0_norm` / Windowed Co-Prime CMS / novelty / `1..11`-subword step-keep
    and stationary `GTRM` EMA bias-correction quantities once per forward pass across all 16 layers.
    """
    B, L, d_model = x0_norm.shape
    dtype = x0_norm.dtype
    cone_cos, uf, bf, hh, cone_strength = compute_windowed_cms_aux(
        x0_norm, binder.x0_cone_mean, binder.cms_proj, banks=binder.BANKS
    )
    x0_iso = (
        F.normalize(x0_norm.float() - binder.x0_cone_mean.float(), p=2, dim=-1, eps=1e-6) * (d_model ** 0.5)
    ).to(dtype)

    cone_gate_l = 1.0 - cone_strength + cone_strength * torch.sigmoid(8.0 * (cone_cos - 0.02))
    nov_stem = torch.sigmoid(6.0 * (0.75 - uf)) * cone_gate_l

    # Pad to +11 so `m = 1..10` (`bf_next` at `m + 1 <= 11`) is strictly bounded for `diff >= 11`!
    bf_pad = F.pad(bf, (0, 0, 0, 11))
    cc_pad = F.pad(cone_cos, (0, 0, 0, 11))

    def step_keep(m: int, cc_bias: float) -> torch.Tensor:
        cc_m = cc_pad[:, m : L + m, :]
        bf_m = bf_pad[:, m : L + m, :]
        bf_next = bf_pad[:, m + 1 : L + m + 1, :]
        non_punct = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_m + cc_bias))
        return (torch.sigmoid(8.0 * (0.68 - bf_m)) * torch.sigmoid(8.0 * (0.72 - bf_next)) * non_punct).unsqueeze(2)

    k0_steps = (step_keep(1, 0.32), step_keep(2, 0.32), step_keep(3, 0.32))
    k1_steps = tuple(step_keep(m, 0.36) for m in range(1, 7))
    k2_ext_steps = tuple(step_keep(m, 0.36) for m in range(7, 11))  # m = 7, 8, 9, 10 (up to 11 subwords total!)

    aux_f = torch.cat([cone_cos, uf.clamp(max=6.0), bf.clamp(max=6.0), hh], dim=-1).to(dtype)
    rep_pen = (
        9.0 * F.relu(uf - 0.42)
        + 7.0 * F.relu(bf - 0.35)
        + (8.0 * cone_strength) * F.relu(0.08 - cone_cos)
    ).clamp(max=15.0).transpose(1, 2)

    nov1 = torch.sigmoid(12.0 * (0.62 - bf)).pow(2).transpose(1, 2)
    nov4 = F.max_pool1d(F.pad(nov1, (3, 0)), kernel_size=4, stride=1)
    ema_bias_corr = compute_ema_bias_correction(L, x0_norm.device, alpha=ema_alpha, start_pos=0)

    return {
        "cms": (cone_cos, uf, bf, hh, cone_strength),
        "shape": (B, L),
        "v2_5": True,
        "x0_iso": x0_iso,
        "nov_stem": nov_stem,
        "k0_steps": k0_steps,
        "k1_steps": k1_steps,
        "k2_ext_steps": k2_ext_steps,
        "aux_f": aux_f,
        "rep_pen": rep_pen,
        "nov4": nov4,
        "ema_bias_corr": ema_bias_corr,
    }


class SalienceEntityBinderV2_5(nn.Module):
    """
    Tri-Horizon Strictly Causal Voronoi-Decoupled Entity Selector (`VAULT v2.5`):
      - Branch `g0` (`diff >= 4`, `m = 1..3`): 1..4-subword entities (100% preserved from `v2.4` -> `Vault Group 1`).
      - Branch `g1` (`diff >= 7`, `m = 1..6`): 5..7-subword entities (100% preserved from `v2.4` -> `Vault Group 2`).
      - Branch `g2` (`diff >= 11`, `m = 1..10`): 8..11-subword entities using odd-prime circular shifts
        `(3, 7, 13, 23, 37, 47, 19, 29, 43, 59)` -> dedicated to `Vault Group 3`, resolving identical-7-token-prefix
        sibling identifiers (e.g. `IonThruster-T21560` vs `IonThruster-T21904`) with zero write/read asymmetry!
    """

    BANKS: Tuple[int, ...] = (503, 509, 521)
    EXT_SHIFTS: Tuple[int, int, int, int] = (19, 29, 43, 59)
    EXT_GAINS: Tuple[float, float, float, float] = (2.55, 2.65, 2.75, 2.85)

    def __init__(
        self,
        d_model: int,
        n_ind_groups: int = 2,
        head_dim: int = 64,
        window_size: int = 64,
        chunk_size: int = 64,
        sal_hidden: int = 128,
        gap_guard: int = 2,
    ):
        super().__init__()
        assert n_ind_groups == 2 and window_size == chunk_size
        self.d_model = d_model
        self.n_ind_groups = n_ind_groups
        self.head_dim = head_dim
        self.ind_dim = n_ind_groups * head_dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.sal_hidden = sal_hidden
        self.gap_guard = gap_guard

        self.register_buffer("x0_cone_mean", torch.zeros(1, 1, d_model, dtype=torch.float32))
        g = torch.Generator().manual_seed(20261031 + d_model)
        self.register_buffer("cms_proj", F.normalize(torch.randn(sum(self.BANKS), d_model, generator=g), p=2, dim=-1))

        self.ent_key_proj = nn.Linear(d_model, 2 * self.ind_dim, bias=False)
        _safe_orthogonal_(self.ent_key_proj.weight)

        self.ent_tok_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.ent_tok_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ent_tok_gate.bias, 1.20)

        self.anchor_ent_conv = nn.Conv1d(
            in_channels=2 * self.ind_dim,
            out_channels=2 * self.ind_dim,
            kernel_size=6,
            padding=5,
            groups=2 * self.ind_dim,
            bias=False,
        )
        with torch.no_grad():
            k6 = torch.tensor([0.25, 0.35, 0.45, 0.55, 0.65, 0.00], dtype=torch.float32)
            k6 = k6 / k6.norm()
            self.anchor_ent_conv.weight.copy_(k6.view(1, 1, 6).expand(2 * self.ind_dim, 1, 6))

        self.sal_span_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=6,
            padding=5,
            groups=d_model,
            bias=False,
        )
        with torch.no_grad():
            ks = torch.tensor([0.25, 0.35, 0.45, 0.50, 0.50, 0.45], dtype=torch.float32)
            ks = ks / ks.norm()
            self.sal_span_conv.weight.copy_(ks.view(1, 1, 6).expand(d_model, 1, 6))

        self.sal_in = nn.Linear(3 * d_model + 4, sal_hidden, bias=True)
        self.sal_out = nn.Linear(sal_hidden, n_ind_groups, bias=True)
        nn.init.normal_(self.sal_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sal_in.bias)
        nn.init.normal_(self.sal_out.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.sal_out.bias, 2.20)

        self.occ_scale = nn.Parameter(torch.full((1, n_ind_groups, 1, 1, 1), 2.80))
        self.recency_param = nn.Parameter(torch.full((1, n_ind_groups, 1, 1, 1), -3.20))

        idx_q = torch.arange(chunk_size) + chunk_size
        idx_k = torch.arange(2 * chunk_size)
        diff = idx_q[:, None] - idx_k[None, :]
        mask_g0 = (diff >= 4) & (diff <= window_size)
        mask_g1 = (diff >= 7) & (diff <= window_size)
        mask_g2 = (diff >= 11) & (diff <= window_size)
        # Keep persistent `sal_lag_mask` shape `(1, 2, 1, C, 2C)` identical to `v2.4` checkpoints!
        lag_mask_2g = torch.stack([mask_g0, mask_g1], dim=0).unsqueeze(0).unsqueeze(2)
        lag_mask_3g = torch.stack([mask_g0, mask_g1, mask_g2], dim=0).unsqueeze(0).unsqueeze(2)
        self.register_buffer("sal_lag_mask", lag_mask_2g)
        self.register_buffer("_sal_lag_mask_3g", lag_mask_3g, persistent=False)
        self.register_buffer("sal_lag_dist", diff.clamp(min=0).float().view(1, 1, 1, chunk_size, 2 * chunk_size))
        self.register_buffer(
            "_id_weights_3g",
            torch.tensor([1.50, 1.20, 1.15], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        x_rms: torch.Tensor,
        x0_norm: torch.Tensor,
        cms_aux=None,
        pre_x0_iso_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_orig, d_m = x0_norm.shape
        D, C = self.head_dim, self.chunk_size
        dtype = x0_norm.dtype

        if isinstance(cms_aux, dict) and cms_aux.get("shape") == (B, L_orig) and cms_aux.get("v2_5"):
            sh = cms_aux
        else:
            sh = compute_v2_5_shared_aux(x0_norm, self)

        x0_iso = sh["x0_iso"]
        nov_stem = sh["nov_stem"]
        k0_1, k0_2, k0_3 = sh["k0_steps"]
        k1_1, k1_2, k1_3, k1_4, k1_5, k1_6 = sh["k1_steps"]
        k2_7, k2_8, k2_9, k2_10 = sh["k2_ext_steps"]
        aux_f = sh["aux_f"]
        rep_pen = sh["rep_pen"]

        w_si = self.sal_in.weight
        k_dim = 2 * self.ind_dim  # 256
        sh_dim = self.sal_hidden  # 128
        if pre_x0_iso_sal is None:
            w_sal_cat = torch.cat(
                [self.ent_key_proj.weight, self.ent_tok_gate.weight, w_si[:, d_m : 2 * d_m], w_si[:, 2 * d_m : 3 * d_m]],
                dim=0,
            )
            pre_x0_iso_sal = F.linear(x0_iso, w_sal_cat)

        a_raw = pre_x0_iso_sal[..., :k_dim].float()
        g_tok = torch.sigmoid((pre_x0_iso_sal[..., k_dim : k_dim + 1] + self.ent_tok_gate.bias).float()) * nov_stem
        sal_iso_p1 = _shift_right_zero(pre_x0_iso_sal[..., k_dim + 1 : k_dim + 1 + sh_dim], 1)
        sal_iso_0 = pre_x0_iso_sal[..., k_dim + 1 + sh_dim :]

        a_stem = causal_depthwise_conv1d(
            (g_tok * a_raw).to(dtype), self.anchor_ent_conv.weight, None, apply_silu=True if False else False
        ).float()
        a_left = a_raw * (0.65 + 0.85 * torch.tanh(1.5 * a_stem)) + 0.45 * a_stem

        a_raw_g = a_raw.view(B, L_orig, 2, 2, D)
        a_left_g = a_left.view(B, L_orig, 2, 2, D)
        e0_pad = F.pad(a_raw_g[:, :, 0], (0, 0, 0, 0, 0, 4))
        e1_pad = F.pad(a_raw_g[:, :, 1], (0, 0, 0, 0, 0, 11))

        tail0 = (
            1.40 * k0_1 * torch.roll(e0_pad[:, 1 : L_orig + 1], shifts=3, dims=-1)
            + 1.50 * k0_2 * torch.roll(e0_pad[:, 2 : L_orig + 2], shifts=7, dims=-1)
            + 1.60 * k0_3 * torch.roll(e0_pad[:, 3 : L_orig + 3], shifts=13, dims=-1)
        )
        tail1 = (
            1.10 * k1_1 * torch.roll(e1_pad[:, 1 : L_orig + 1], shifts=3, dims=-1)
            + 1.25 * k1_2 * torch.roll(e1_pad[:, 2 : L_orig + 2], shifts=7, dims=-1)
            + 1.45 * k1_3 * torch.roll(e1_pad[:, 3 : L_orig + 3], shifts=13, dims=-1)
            + 1.75 * k1_4 * torch.roll(e1_pad[:, 4 : L_orig + 4], shifts=23, dims=-1)
            + 2.10 * k1_5 * torch.roll(e1_pad[:, 5 : L_orig + 5], shifts=37, dims=-1)
            + 2.40 * k1_6 * torch.roll(e1_pad[:, 6 : L_orig + 6], shifts=47, dims=-1)
        )
        tail2 = (
            tail1
            + 2.55 * k2_7 * torch.roll(e1_pad[:, 7 : L_orig + 7], shifts=19, dims=-1)
            + 2.65 * k2_8 * torch.roll(e1_pad[:, 8 : L_orig + 8], shifts=29, dims=-1)
            + 2.75 * k2_9 * torch.roll(e1_pad[:, 9 : L_orig + 9], shifts=43, dims=-1)
            + 2.85 * k2_10 * torch.roll(e1_pad[:, 10 : L_orig + 10], shifts=59, dims=-1)
        )

        a_left_3 = torch.cat([a_left_g, a_left_g[:, :, 1:2]], dim=2)  # (B, L_orig, 3, 2, D)
        tail_3 = torch.stack([tail0, tail1, tail2], dim=2)            # (B, L_orig, 3, 2, D)
        a_ctx = (
            fft_bind_with_id(a_left_3, tail_3, id_weight=self._id_weights_3g)
            .reshape(B, L_orig, 3, 2 * D)
            .to(dtype)
            .permute(0, 2, 1, 3)
        )  # (B, 3, L_orig, 128)

        x0_span = causal_depthwise_conv1d(x0_iso, self.sal_span_conv.weight, None, apply_silu=False)
        sal_h = (
            F.linear(x0_span, w_si[:, :d_m], self.sal_in.bias)
            + sal_iso_p1
            + sal_iso_0
            + F.linear(aux_f, w_si[:, 3 * d_m :])
        )
        raw_sal = self.sal_out(F.silu(sal_h)).float().transpose(1, 2)
        sal_2 = 4.0 * torch.tanh(raw_sal / 4.0) - rep_pen
        sal = torch.cat([sal_2, sal_2[:, 1:2, :]], dim=1)  # (B, 3, L_orig)

        rem = L_orig % C
        pad_r = (C - rem) if rem != 0 else 0
        if pad_r > 0:
            a_ctx = F.pad(a_ctx, (0, 0, 0, pad_r))
            sal = F.pad(sal, (0, pad_r), value=-1e4)

        L = a_ctx.size(2)
        a_pad = F.pad(a_ctx, (0, 0, C, 0))
        a_win = a_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)  # (B, 3, M, 2C, 128)
        s_pad = F.pad(sal, (C, 0), value=-1e4)
        s_win = s_pad.unfold(dimension=2, size=2 * C, step=C)                         # (B, 3, M, 2C)

        p_win = torch.where(s_win > -5e3, torch.sigmoid(s_win), torch.zeros_like(s_win))
        cs = torch.cumsum(p_win, dim=-1)
        cs_g0 = cs[:, 0:1, :, (C - 4) : (2 * C - 4)]
        cs_g1 = cs[:, 1:2, :, (C - 7) : (2 * C - 7)]
        cs_g2 = cs[:, 2:3, :, (C - 11) : (2 * C - 11)]
        cs_newest = torch.cat([cs_g0, cs_g1, cs_g2], dim=1).unsqueeze(-1)
        gg = self.gap_guard
        cs_guard = torch.cat([cs[..., gg:], cs[..., -1:].expand(-1, -1, -1, gg)], dim=-1) if gg > 0 else cs
        prior_occ = (cs_newest - cs_guard.unsqueeze(-2)).clamp(min=0.0)

        occ_3 = torch.cat([self.occ_scale.float(), self.occ_scale[:, 1:2].float()], dim=1)
        rec_3 = torch.cat([self.recency_param.float(), self.recency_param[:, 1:2].float()], dim=1)

        valid = self._sal_lag_mask_3g & (s_win.unsqueeze(-2) > -5e3)
        eff_logits = (
            s_win.unsqueeze(-2)
            - F.softplus(occ_3) * prior_occ
            - F.softplus(rec_3) * self.sal_lag_dist
        )
        eff_logits = torch.where(valid, eff_logits, torch.full_like(eff_logits, -1e4))
        any_valid = valid.any(dim=-1, keepdim=True)
        attn_w = torch.where(any_valid, F.softmax(eff_logits, dim=-1), torch.zeros_like(eff_logits))

        out_blk = torch.matmul(attn_w.to(dtype), a_win)                                 # (B, 3, M, C, 128) WMMA!
        ent_3 = out_blk.reshape(B, 3, L, 2 * D)[:, :, :L_orig, :].permute(0, 2, 1, 3).float()
        if L_orig > 0:
            ent_3 = torch.cat([torch.zeros_like(ent_3[:, :1]), ent_3[:, 1:]], dim=1)
        return ent_3[..., :D], ent_3[..., D:]  # (ent_u, ent_r), each (B, L_orig, 3, D) float32


class VAULTv2_5Attention(nn.Module):
    """
    True 3:1 Grouped-Vault Attention (GVA) Layer for `VAULTLMv2_5`.
    Strictly O(L) compute & O(1) stationary inference state across `2K -> 128K+` sequence lengths.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_q_heads: int = 12,
        n_vault_groups: int = 4,
        head_dim: int = 64,
        prime_banks: Tuple[int, ...] = (47, 53, 59, 61),
        chunk_size: int = 64,
        conv_kernel: int = 4,
        layer_idx: int = 0,
        ckpt_interval: int = 8,
        ema_alpha: float = GTRM_EMA_ALPHA_DEFAULT,
    ):
        super().__init__()
        assert n_q_heads % n_vault_groups == 0 and n_vault_groups >= 4
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_vault_groups = n_vault_groups
        self.heads_per_group = n_q_heads // n_vault_groups
        self.head_dim = head_dim
        self.kv_dim = n_vault_groups * head_dim
        self.prime_banks = tuple(int(p) for p in prime_banks)
        self.num_banks = len(self.prime_banks)
        self.total_cells = sum(self.prime_banks)
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.v2_2_mode = True
        self.v2_1_mode = True
        self.ckpt_interval = ckpt_interval
        self.ema_alpha = float(ema_alpha)

        self.qk_group_anchor = nn.Linear(d_model, self.kv_dim, bias=False)
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_delta = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_delta.weight, std=0.02)

        self.n_keep_g = 1
        self.n_ind_groups = 3
        self.n_ent_groups = 2

        self.lex_router_g2 = nn.Linear(2 * d_model, 2 * self.n_ent_groups * head_dim, bias=False)
        self.lex_router_g3 = nn.Linear(3 * d_model, 2 * head_dim, bias=False)
        _safe_orthogonal_(self.lex_router_g2.weight)
        _safe_orthogonal_(self.lex_router_g3.weight)

        self.salience_binder = SalienceEntityBinderV2_5(
            d_model,
            n_ind_groups=self.n_ent_groups,
            head_dim=head_dim,
            window_size=chunk_size,
            chunk_size=chunk_size,
        )
        self.ind_shared_proj = nn.Parameter(torch.zeros(self.n_ent_groups, head_dim, head_dim))
        # New v2.5 parameters (zero-initialized -> exact Step-0 bit parity on `m = 1..6` when `ext_span_alpha == 0`)
        self.ind_shared_proj_g2 = nn.Parameter(torch.zeros(1, head_dim, head_dim))
        self.ext_span_alpha = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))

        self.q_subhead_delta = nn.Linear(
            d_model, self.n_ind_groups * (self.heads_per_group - 1) * head_dim, bias=False
        )
        nn.init.normal_(self.q_subhead_delta.weight, mean=0.0, std=0.01)

        self.ind_beta_proj = nn.Linear(4 * d_model, self.n_ind_groups, bias=False)
        nn.init.normal_(self.ind_beta_proj.weight, mean=0.0, std=0.01)

        self.v_ind_proj = nn.Linear(d_model, self.n_ind_groups * head_dim, bias=False)
        self.ind_out_proj = nn.Linear(self.n_ind_groups * head_dim, d_model, bias=False)
        _safe_orthogonal_(self.v_ind_proj.weight)
        _safe_orthogonal_(self.ind_out_proj.weight)

        self.ind_read_proj = nn.Linear(3 * d_model, self.n_ind_groups * self.heads_per_group, bias=True)
        nn.init.normal_(self.ind_read_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.ind_read_proj.bias, -1.80)

        self.q_conv = CausalShortConv1D(n_q_heads * head_dim, kernel_size=conv_kernel)
        self.k_conv = CausalShortConv1D(self.kv_dim, kernel_size=conv_kernel)
        self.v_conv = CausalShortConv1D(self.kv_dim, kernel_size=conv_kernel)

        self.beta_proj = nn.Linear(d_model, n_vault_groups, bias=True)
        nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_proj.bias, -0.15)

        self.kenyon_proj = TernarySTEProjection(head_dim, self.total_cells, seed=100 + layer_idx)
        self.apl_norm = FusedAPLNorm(self.total_cells)

        anchors = []
        for p in self.prime_banks:
            anchors.extend([1.0 / float(p)] * p)
        self.register_buffer("harmonic_anchors", torch.tensor(anchors, dtype=torch.float32))

        self.beta_sem_q = nn.Parameter(torch.full((1, n_q_heads, 1, 1), 1.85))
        self.beta_sem_k = nn.Parameter(torch.full((1, n_vault_groups, 1, 1), 1.85))

        log_one_minus = torch.linspace(math.log(0.34), math.log(0.008), n_q_heads)
        head_decays = 1.0 - torch.exp(log_one_minus)
        idx_q = torch.arange(chunk_size, dtype=torch.float32) + chunk_size
        idx_k = torch.arange(2 * chunk_size, dtype=torch.float32)
        diff = idx_q.unsqueeze(-1) - idx_k.unsqueeze(-2)
        valid = (diff >= 0).float().unsqueeze(0)
        decay_3d = torch.pow(head_decays.view(n_q_heads, 1, 1), diff.clamp(min=0.0).unsqueeze(0)) * valid
        self.register_buffer("local_decay_mask", decay_3d.view(1, n_q_heads, 1, chunk_size, 2 * chunk_size))
        attn_bias = torch.where(decay_3d > 0.0, torch.log(decay_3d.clamp(min=1e-8)), torch.full_like(decay_3d, -1e9))
        self.register_buffer(
            "_local_attn_bias", attn_bias.view(1, n_q_heads, 1, chunk_size, 2 * chunk_size), persistent=False
        )

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos_2c = torch.arange(2 * chunk_size, dtype=torch.float32)
        freqs = torch.outer(pos_2c, inv_freq)
        self.register_buffer("rope_cos_k", torch.cos(freqs).view(1, 1, 1, 2 * chunk_size, head_dim // 2))
        self.register_buffer("rope_sin_k", torch.sin(freqs).view(1, 1, 1, 2 * chunk_size, head_dim // 2))
        self.register_buffer("rope_cos_q", torch.cos(freqs[chunk_size:]).view(1, 1, 1, chunk_size, head_dim // 2))
        self.register_buffer("rope_sin_q", torch.sin(freqs[chunk_size:]).view(1, 1, 1, chunk_size, head_dim // 2))

        self.vault_gate = nn.Parameter(torch.full((1, n_q_heads, 1, 1), 0.50))
        self.syntax_gate = nn.Parameter(torch.full((1, n_q_heads, 1, 1), 0.50))
        self.gtrm_gate = nn.Parameter(torch.full((1, n_q_heads, 1, 1), 0.08))
        self.gtrm_proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.normal_(self.gtrm_proj.weight, std=0.02)

        self.head_norm = RMSNorm(head_dim)
        self.out_gate = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        self._eval_w_cache = None
        self.register_load_state_dict_post_hook(lambda m, _: setattr(m, "_eval_w_cache", None))

    def train(self, mode: bool = True):
        self._eval_w_cache = None
        return super().train(mode)

    def _get_v2_5_packed_weights(self, x: torch.Tensor):
        d_m = self.d_model
        sb = self.salience_binder
        w_si = sb.sal_in.weight
        w_ib = self.ind_beta_proj.weight
        w_lg2 = self.lex_router_g2.weight
        w_lg3 = self.lex_router_g3.weight
        w_ir = self.ind_read_proj.weight

        if not self.training and not torch.is_grad_enabled():
            comp_dt = torch.get_autocast_dtype("cuda") if (x.is_cuda and torch.is_autocast_enabled("cuda")) else x.dtype
            key = (
                x.device,
                comp_dt,
                self.q_proj.weight.data_ptr(),
                self.q_proj.weight._version,
                self.v_ind_proj.weight._version,
            )
            if self._eval_w_cache is None or self._eval_w_cache[0] != key:
                w_x_all = torch.cat(
                    [
                        self.qk_group_anchor.weight,
                        self.q_proj.weight,
                        self.k_delta.weight,
                        self.v_proj.weight,
                        self.beta_proj.weight,
                        self.out_gate.weight,
                    ],
                    dim=0,
                ).to(dtype=comp_dt).contiguous()
                qkv_w = torch.cat(
                    [self.q_conv.conv.weight, self.k_conv.conv.weight, self.v_conv.conv.weight], dim=0
                ).contiguous()
                qkv_b = torch.cat(
                    [self.q_conv.conv.bias, self.k_conv.conv.bias, self.v_conv.conv.bias], dim=0
                ).contiguous()
                w_iso_all = torch.cat(
                    [
                        sb.ent_key_proj.weight,
                        sb.ent_tok_gate.weight,
                        w_si[:, d_m : 2 * d_m],
                        w_si[:, 2 * d_m : 3 * d_m],
                        self.q_subhead_delta.weight,
                        self.v_ind_proj.weight,
                        w_ib[:, 2 * d_m : 3 * d_m],
                        w_ib[:, 3 * d_m :],
                    ],
                    dim=0,
                ).to(dtype=comp_dt).contiguous()
                w_x0n_all = torch.cat(
                    [
                        w_lg2[:, d_m:],
                        w_lg3[:, 2 * d_m :],
                        w_ir[:, 2 * d_m :],
                        w_lg2[:, :d_m],
                        w_lg3[:, d_m : 2 * d_m],
                        w_ir[:, :d_m],
                        w_lg3[:, :d_m],
                    ],
                    dim=0,
                ).to(dtype=comp_dt).contiguous()
                w_ib_rms = torch.cat([w_ib[:, :d_m], w_ib[:, d_m : 2 * d_m]], dim=0).to(dtype=comp_dt).contiguous()
                self._eval_w_cache = (key, (w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms))
            return self._eval_w_cache[1]

        w_x_all = torch.cat(
            [
                self.qk_group_anchor.weight,
                self.q_proj.weight,
                self.k_delta.weight,
                self.v_proj.weight,
                self.beta_proj.weight,
                self.out_gate.weight,
            ],
            dim=0,
        )
        qkv_w = torch.cat([self.q_conv.conv.weight, self.k_conv.conv.weight, self.v_conv.conv.weight], dim=0)
        qkv_b = torch.cat([self.q_conv.conv.bias, self.k_conv.conv.bias, self.v_conv.conv.bias], dim=0)
        w_iso_all = torch.cat(
            [
                sb.ent_key_proj.weight,
                sb.ent_tok_gate.weight,
                w_si[:, d_m : 2 * d_m],
                w_si[:, 2 * d_m : 3 * d_m],
                self.q_subhead_delta.weight,
                self.v_ind_proj.weight,
                w_ib[:, 2 * d_m : 3 * d_m],
                w_ib[:, 3 * d_m :],
            ],
            dim=0,
        )
        w_x0n_all = torch.cat(
            [
                w_lg2[:, d_m:],
                w_lg3[:, 2 * d_m :],
                w_ir[:, 2 * d_m :],
                w_lg2[:, :d_m],
                w_lg3[:, d_m : 2 * d_m],
                w_ir[:, :d_m],
                w_lg3[:, :d_m],
            ],
            dim=0,
        )
        w_ib_rms = torch.cat([w_ib[:, :d_m], w_ib[:, d_m : 2 * d_m]], dim=0)
        return w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms

    def _route_cells_v2(
        self, q_unit: torch.Tensor, k_unit: torch.Tensor, step_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        orig_dtype = q_unit.dtype
        n_q = q_unit.size(1)
        inv_tau = 1.0 / max(0.18, 0.45 * (1.0 - 0.60 * step_ratio))
        need_ste = torch.is_grad_enabled() and (
            q_unit.requires_grad
            or k_unit.requires_grad
            or self.kenyon_proj.weight.requires_grad
            or self.apl_norm.gain.requires_grad
        )

        qk_unit = torch.cat([q_unit, k_unit], dim=1)
        kc_qk = F.relu(self.apl_norm(self.kenyon_proj(qk_unit)).float()) * inv_tau

        cqk_list = []
        gqk_list = []
        offset = 0
        for p in self.prime_banks:
            s = kc_qk[..., offset : offset + p]
            idx = torch.argmax(s, dim=-1)
            if need_ste:
                pr = F.softmax(s, dim=-1)
                gqk_list.append(pr.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
            cqk_list.append(idx + offset)
            offset += p

        cell_qk = torch.stack(cqk_list, dim=2)
        cell_q = cell_qk[:, :n_q]
        cell_k = cell_qk[:, n_q:]
        if not need_ste:
            return cell_q, cell_k, None, None

        gqk_mean = torch.stack(gqk_list, dim=2).mean(dim=2)
        ste_qk = (1.0 + (gqk_mean - gqk_mean.detach())).unsqueeze(-1).to(orig_dtype)
        return cell_q, cell_k, ste_qk[:, :n_q], ste_qk[:, n_q:]

    def _chunk_local_rope_syntax(
        self, q_norm: torch.Tensor, k_norm: torch.Tensor, v_grp: torch.Tensor
    ) -> torch.Tensor:
        B, H, L_orig, D = q_norm.shape
        G = self.n_vault_groups
        H_g = self.heads_per_group
        C = self.chunk_size

        rem = L_orig % C
        pad_l = (C - rem) if rem != 0 else 0
        if pad_l > 0:
            q_norm = F.pad(q_norm, (0, 0, 0, pad_l))
            k_norm = F.pad(k_norm, (0, 0, 0, pad_l))
            v_grp = F.pad(v_grp, (0, 0, 0, pad_l))

        L = q_norm.size(2)
        M = L // C

        q_blk = _apply_local_rope(q_norm.view(B, H, M, C, D), self.rope_cos_q, self.rope_sin_q)
        k_pad = F.pad(k_norm, (0, 0, C, 0))
        v_pad = F.pad(v_grp, (0, 0, C, 0))
        k_win = _apply_local_rope(
            k_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3),
            self.rope_cos_k,
            self.rope_sin_k,
        )
        v_win = v_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)

        q_gm = q_blk.view(B, G, H_g, M, C, D).permute(0, 1, 3, 2, 4, 5).reshape(B, G, M, H_g * C, D)
        scale = 1.0 / math.sqrt(D)
        logits_gm = torch.matmul(q_gm, k_win.transpose(-1, -2)).float() * scale
        logits = logits_gm.view(B, G, M, H_g, C, 2 * C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, M, C, 2 * C)
        logits = torch.where(self.local_decay_mask > 0.0, logits + self._local_attn_bias, torch.full_like(logits, -1e9))
        probs = F.softmax(logits, dim=-1).to(v_grp.dtype)
        probs_gm = probs.view(B, G, H_g, M, C, 2 * C).permute(0, 1, 3, 2, 4, 5).reshape(B, G, M, H_g * C, 2 * C)
        out_gm = torch.matmul(probs_gm, v_win)
        out_blk = out_gm.view(B, G, M, H_g, C, D).permute(0, 1, 3, 2, 4, 5).reshape(B, H, L, D)
        return out_blk[:, :, :L_orig, :]

    def forward(
        self,
        x: torch.Tensor,
        step_ratio: float = 1.0,
        return_beta_mean: bool = False,
        x0_norm: Optional[torch.Tensor] = None,
        return_ent_div: bool = False,
        cms_aux=None,
    ):
        B, L, d_m = x.shape
        ent_div_aux = torch.zeros((), device=x.device, dtype=torch.float32)

        if x0_norm is None:
            x0_norm = x.detach()
        n_keep_g = self.n_keep_g
        n_keep_h = n_keep_g * self.heads_per_group
        G = self.n_vault_groups
        D = self.head_dim
        q_dim = self.n_q_heads * D
        kv_dim = self.kv_dim

        # 1. Fused single-GEMM `x` projection + single-launch `Q/K/V` CausalShortConv1D
        w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = self._get_v2_5_packed_weights(x)
        x_proj = F.linear(x, w_x_all)
        anchor = x_proj[..., :kv_dim]
        q_raw = x_proj[..., kv_dim : kv_dim + q_dim]
        k_raw = x_proj[..., kv_dim + q_dim : 2 * kv_dim + q_dim]
        v_raw = x_proj[..., 2 * kv_dim + q_dim : 3 * kv_dim + q_dim]
        beta_raw = x_proj[..., 3 * kv_dim + q_dim : 3 * kv_dim + q_dim + G] + self.beta_proj.bias
        gate_raw = x_proj[..., 3 * kv_dim + q_dim + G :].contiguous()

        anchor_q = (
            anchor.view(B, L, G, 1, D)
            .expand(B, L, G, self.heads_per_group, D)
            .reshape(B, L, q_dim)
        )
        qkv_in = torch.cat([anchor_q + q_raw, anchor + k_raw, v_raw], dim=-1)
        del x_proj, anchor, anchor_q, q_raw, k_raw, v_raw, w_x_all

        qkv_out = causal_depthwise_conv1d(qkv_in, qkv_w, qkv_b, apply_silu=True)
        del qkv_in, qkv_w, qkv_b

        q_syn_raw = qkv_out[..., :q_dim].view(B, L, self.n_q_heads, D).transpose(1, 2)
        k_syn_raw = qkv_out[..., q_dim : q_dim + kv_dim].view(B, L, G, D).transpose(1, 2)
        v = qkv_out[..., q_dim + kv_dim :].contiguous().view(B, L, G, D).transpose(1, 2)
        dtype = v.dtype

        qs_f, ks_f = q_syn_raw.float(), k_syn_raw.float()
        del qkv_out, q_syn_raw, k_syn_raw
        qs_rms = torch.rsqrt(torch.mean(qs_f * qs_f, dim=-1, keepdim=True) + 1e-6)
        ks_rms = torch.rsqrt(torch.mean(ks_f * ks_f, dim=-1, keepdim=True) + 1e-6)
        q_norm = (qs_f * qs_rms).to(dtype)
        k_norm = (ks_f * ks_rms).to(dtype)

        x_f = x.float()
        x_rms = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
        del x_f

        if isinstance(cms_aux, dict) and cms_aux.get("shape") == (B, L) and cms_aux.get("v2_5"):
            sh = cms_aux
        else:
            sh = compute_v2_5_shared_aux(x0_norm, self.salience_binder, ema_alpha=self.ema_alpha)

        x0_iso = sh["x0_iso"]
        nov4 = sh["nov4"]
        ema_bias_corr = sh["ema_bias_corr"]

        # 2. Fused single-GEMM `x0_iso` projection across SalienceEntityBinderV2_5 + Induction Q/V/Beta
        sb = self.salience_binder
        sal_c = 2 * sb.ind_dim + 1 + 2 * sb.sal_hidden  # 513
        qsh_c = self.n_ind_groups * (self.heads_per_group - 1) * D  # 384
        vind_c = self.n_ind_groups * D  # 192
        ib_c = self.n_ind_groups  # 3
        iso_proj = F.linear(x0_iso, w_iso_all)
        del w_iso_all
        pre_sal = iso_proj[..., :sal_c]
        qsh_proj = iso_proj[..., sal_c : sal_c + qsh_c]
        vind_proj = iso_proj[..., sal_c + qsh_c : sal_c + qsh_c + vind_c]
        ib_iso_p1 = _shift_right_zero(
            iso_proj[..., sal_c + qsh_c + vind_c : sal_c + qsh_c + vind_c + ib_c], 1
        )
        ib_iso_0 = iso_proj[..., sal_c + qsh_c + vind_c + ib_c :]

        # Tri-Horizon Voronoi-Decoupled Entity Subspaces (`ent_u`, `ent_r` each `(B, L, 3, D)`)
        ent_u, ent_r = sb(x_rms, x0_norm, cms_aux=sh, pre_x0_iso_sal=pre_sal)
        del pre_sal
        m_shared_2 = (0.20 / 8.0) * torch.tanh(self.ind_shared_proj.float())
        m_shared_g2 = m_shared_2[1:2] + (0.20 / 8.0) * torch.tanh(self.ind_shared_proj_g2.float())
        m_shared_3 = torch.cat([m_shared_2, m_shared_g2], dim=0)  # (3, D, D)

        ent_ur = torch.stack([ent_u, ent_r], dim=3)  # (B, L, 3, 2, D)
        del ent_u, ent_r
        eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
        del ent_ur
        ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, m_shared_3)
        del eur_zm
        ur_ent_rms = ur_ent * torch.rsqrt(ur_ent.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        w_ext = 2.0 * torch.tanh(self.ext_span_alpha.float())
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
        )  # (B, L, 3, 2, D)
        del ur_ent_rms, ur_comb

        # Fused single-GEMM `x0_norm` projection across Lexical Routers + Readout Gate
        w_ir = self.ind_read_proj.weight
        lg2_c = self.lex_router_g2.weight.size(0)  # 256
        lg3_c = self.lex_router_g3.weight.size(0)  # 128
        ir_c = w_ir.size(0)  # 9
        s0_c = lg2_c + lg3_c + ir_c  # 393
        x0n_proj = F.linear(x0_norm, w_x0n_all)
        del w_x0n_all
        s01 = x0n_proj[..., :s0_c] + _shift_right_zero(x0n_proj[..., s0_c : 2 * s0_c], 1)
        s2_lg3 = _shift_right_zero(x0n_proj[..., 2 * s0_c :], 2)
        del x0n_proj

        lex_12 = s01[..., :lg2_c].float().view(B, L, 2, 2, D)
        lex_3 = (s01[..., lg2_c : lg2_c + lg3_c] + s2_lg3).float().view(B, L, 1, 2, D)
        ir_x0 = s01[..., lg2_c + lg3_c :] + self.ind_read_proj.bias
        del s01, s2_lg3

        lex_both = torch.cat([lex_12, lex_3], dim=2)
        del lex_12, lex_3
        lex_zm = lex_both - lex_both.mean(dim=-1, keepdim=True)
        del lex_both
        lex_rms = lex_zm * torch.rsqrt(lex_zm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        del lex_zm

        ur_ind_raw = fft_circular_bind(ur_ent_3, lex_rms)
        del ur_ent_3, lex_rms
        u_ind_raw = ur_ind_raw[:, :, :, 0, :]
        r_ind_raw = ur_ind_raw[:, :, :, 1, :]
        u_ind = u_ind_raw * torch.rsqrt(u_ind_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        r_all = F.normalize(r_ind_raw, p=2, dim=-1, eps=1e-6)
        del ur_ind_raw, u_ind_raw, r_ind_raw
        r_all_prev = F.pad(r_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0)) if L > 1 else torch.zeros_like(r_all)

        shared_group_q = self.prime_banks != (17, 19, 23, 29)
        lex_q_3 = r_all.transpose(1, 2)
        lex_k_3 = r_all_prev.transpose(1, 2)
        kr_f = torch.cat([ks_f[:, :n_keep_g], lex_k_3], dim=1)
        inv_sqrt_d = 1.0 / math.sqrt(D)
        kr_unit = (kr_f * (torch.rsqrt(kr_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)

        if shared_group_q:
            qr_4_f = torch.cat([qs_f[:, :n_keep_h].mean(dim=1, keepdim=True), lex_q_3], dim=1)
            qr_unit = (qr_4_f * (torch.rsqrt(qr_4_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)
            del qr_4_f
            cq_4, cell_k, ste_q_4, ste_k = self._route_cells_v2(qr_unit, kr_unit, step_ratio)
            cell_q = cq_4.repeat_interleave(self.heads_per_group, dim=1)
            ste_q = ste_q_4.repeat_interleave(self.heads_per_group, dim=1) if ste_q_4 is not None else None
            del cq_4, ste_q_4
        else:
            qr_6_f = torch.cat([qs_f[:, :n_keep_h], lex_q_3], dim=1)
            qr_unit = (qr_6_f * (torch.rsqrt(qr_6_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)
            del qr_6_f
            cq_6, cell_k, ste_q_6, ste_k = self._route_cells_v2(qr_unit, kr_unit, step_ratio)
            cell_q = torch.cat(
                [cq_6[:, :n_keep_h], cq_6[:, n_keep_h:].repeat_interleave(self.heads_per_group, dim=1)], dim=1
            )
            if ste_q_6 is not None:
                ste_q = torch.cat(
                    [ste_q_6[:, :n_keep_h], ste_q_6[:, n_keep_h:].repeat_interleave(self.heads_per_group, dim=1)],
                    dim=1,
                )
            else:
                ste_q = None
            del cq_6, ste_q_6
        del r_all, r_all_prev, lex_q_3, lex_k_3, kr_f, qr_unit, kr_unit

        u_ind_prev = F.pad(u_ind[:, :-1, :, :], (0, 0, 0, 0, 1, 0)) if L > 1 else torch.zeros_like(u_ind)
        k_ind_3 = u_ind_prev.transpose(1, 2)

        q_sh_rest = 0.04 * torch.tanh(qsh_proj.float()).view(
            B, L, self.n_ind_groups, self.heads_per_group - 1, D
        )
        q_sh_zero = torch.zeros(
            (B, L, self.n_ind_groups, 1, D), device=x.device, dtype=torch.float32
        )
        q_ind_9 = (
            (u_ind.unsqueeze(3) + torch.cat([q_sh_zero, q_sh_rest], dim=3))
            .reshape(B, L, self.n_ind_groups * self.heads_per_group, D)
            .transpose(1, 2)
        )
        del u_ind, u_ind_prev, q_sh_rest, q_sh_zero

        bq = torch.clamp(self.beta_sem_q.float(), 1.2, 3.5)
        bk = torch.clamp(self.beta_sem_k.float(), 1.2, 3.5)
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
        del qs_f, ks_f, qs_rms, ks_rms

        zq_ind = q_ind_9 - q_ind_9.mean(dim=-1, keepdim=True)
        zk_ind = k_ind_3 - k_ind_3.mean(dim=-1, keepdim=True)
        del q_ind_9, k_ind_3
        phi_q = torch.cat([pq_keep, F.normalize(zq_ind, p=2, dim=-1)], dim=1).to(dtype)
        pk_ind = F.normalize(zk_ind, p=2, dim=-1)
        del pq_keep, zq_ind, zk_ind

        v_ind = vind_proj.contiguous().view(B, L, self.n_ind_groups, D).transpose(1, 2)
        if ste_k is not None:
            v_keep_w = v[:, :n_keep_g] * ste_k[:, :n_keep_g]
            v_ind_raw = (0.05 * v[:, n_keep_g:] + v_ind) * ste_k[:, n_keep_g:]
        else:
            v_keep_w = v[:, :n_keep_g]
            v_ind_raw = 0.05 * v[:, n_keep_g:] + v_ind
        del v_ind, ste_k

        beta_keep_f = 0.96 * torch.sigmoid(beta_raw.float()).transpose(1, 2)[:, :n_keep_g, :]
        del beta_raw

        x_rms_zm = x_rms - x_rms.mean(dim=-1, keepdim=True)
        ib_rms_both = F.linear(x_rms_zm, w_ib_rms)
        del x_rms_zm, w_ib_rms
        ib_lin = (
            _shift_right_zero(ib_rms_both[..., :ib_c], 1)
            + ib_rms_both[..., ib_c:]
            + ib_iso_p1
            + ib_iso_0
        )
        del iso_proj, qsh_proj, vind_proj, ib_iso_p1, ib_iso_0, ib_rms_both
        z_beta = 3.4 * torch.tanh(ib_lin.float() / 3.4)
        g_beta = torch.sigmoid(z_beta).transpose(1, 2)
        del ib_lin, z_beta
        beta_ind_raw_f = 0.00005 + nov4 * (0.140 * g_beta)
        del g_beta
        beta_aux_mean = beta_ind_raw_f.mean()

        if return_ent_div and L >= 256:
            m_cnt = L // 64
            l_tr = m_cnt * 64
            x0_tr = x0_norm[:, :l_tr]
            x0_p1_tr = _shift_right_zero(x0_tr, 1)
            u_ent = ur_ent[:, :, :, 0, :]
            tok_val = ((x0_tr - x0_p1_tr).float().abs().amax(dim=-1) > 1e-5).view(B, 1, m_cnt, 64)
            chk_val = tok_val.squeeze(1).any(dim=-1)
            rev_cum = chk_val.flip(dims=[1]).long().cumsum(dim=1).flip(dims=[1])
            hay_chk = chk_val & (rev_cum > 3)
            b_blk = beta_ind_raw_f[:, :, :l_tr].detach().view(B, 3, m_cnt, 64)
            w_blk = F.softmax(torch.where(tok_val, 25.0 * b_blk, torch.full_like(b_blk, -1e4)), dim=-1)
            u_prev_3 = F.pad(u_ent[:, :-1, :, :], (0, 0, 0, 0, 1, 0))[:, :l_tr, :, :]
            u_blk = u_prev_3.permute(0, 2, 1, 3).reshape(B, 3, m_cnt, 64, D)
            u_pool = (w_blk.unsqueeze(-1) * u_blk).sum(dim=3)
            u_pool = u_pool - u_pool.mean(dim=-1, keepdim=True)
            u_hat = F.normalize(u_pool, p=2, dim=-1, eps=1e-6)
            gram = torch.matmul(u_hat, u_hat.transpose(-1, -2))
            c_act = torch.where(tok_val, b_blk, torch.zeros_like(b_blk)).amax(dim=-1)
            c_w = (c_act / c_act.mean(dim=-1, keepdim=True).clamp(min=1e-4)).clamp(max=2.5) * hay_chk.unsqueeze(1).float()
            idx_m = torch.arange(m_cnt, device=x.device)
            far_m = ((idx_m.unsqueeze(1) - idx_m.unsqueeze(0)).abs() >= 2).view(1, 1, m_cnt, m_cnt).float()
            pair_w = (c_w.unsqueeze(3) * c_w.unsqueeze(2)) * far_m
            ent_div_aux = (gram.pow(2) * pair_w).sum() / pair_w.sum().clamp(min=1.0)
        del ur_ent

        W = self.chunk_size
        ck_ind_src = cell_k[:, n_keep_g:]
        if L > W:
            pk_ind_w = F.pad(pk_ind[:, :, :-W, :], (0, 0, W, 0))
            v_ind_w = F.pad(v_ind_raw[:, :, :-W, :], (0, 0, W, 0))
            beta_ind_f = F.pad(beta_ind_raw_f[:, :, :-W], (W, 0), value=0.0)
            ck_pad = ck_ind_src[:, :, :, :1].expand(-1, -1, -1, W)
            ck_ind_w = torch.cat([ck_pad, ck_ind_src[:, :, :, :-W]], dim=3)
        else:
            pk_ind_w = torch.zeros_like(pk_ind)
            v_ind_w = torch.zeros_like(v_ind_raw)
            beta_ind_f = torch.zeros_like(beta_ind_raw_f)
            ck_ind_w = ck_ind_src
        del pk_ind, v_ind_raw, beta_ind_raw_f, ck_ind_src

        phi_k = torch.cat([pk_keep, pk_ind_w], dim=1).to(dtype)
        v_vault = torch.cat([v_keep_w, v_ind_w], dim=1)
        beta_write = torch.cat([beta_keep_f, beta_ind_f], dim=1).to(dtype)
        cell_k = torch.cat([cell_k[:, :n_keep_g], ck_ind_w], dim=1)
        del pk_keep, pk_ind_w, v_keep_w, v_ind_w, beta_keep_f, beta_ind_f, ck_ind_w

        if phi_q.is_cuda:
            y_vault = flash_vault_v2_chunkwise(
                phi_q,
                phi_k,
                v_vault,
                beta_write,
                cell_q,
                cell_k,
                chunk_size=self.chunk_size,
                v2_1_consensus=True,
                ckpt_interval=self.ckpt_interval,
                v2_2_beta_occ=True,
                prime_banks=self.prime_banks,
                shared_group_cell_q=shared_group_q,
            )
        else:
            y_vault = _vault_v2_5_cpu_chunkwise_autograd(
                phi_q,
                phi_k,
                v_vault,
                beta_write,
                cell_q,
                cell_k,
                chunk_size=self.chunk_size,
                v2_2_beta_occ=True,
                prime_banks=self.prime_banks,
            )
        del phi_q, phi_k, v_vault, beta_write, cell_q, cell_k

        # 3. O(L) Chunk-Local RoPE + Multi-Scale Decay Syntax Branch
        y_syntax = self._chunk_local_rope_syntax(q_norm, k_norm, v)

        # 4. O(L) Stationary Bias-Corrected Causal Leaky-EMA GTRM Summary (`2K -> 128K+` invariant!)
        v_mean = chunked_causal_ema_dim2(
            v.float(), alpha=self.ema_alpha, block_len=self.chunk_size, ema_bias_corr=ema_bias_corr
        ).to(dtype)
        v_macro = self.gtrm_proj(v_mean).repeat_interleave(self.heads_per_group, dim=1)

        y_keep = (
            self.vault_gate[:, :n_keep_h].to(dtype) * y_vault[:, :n_keep_h]
            + self.syntax_gate[:, :n_keep_h].to(dtype) * y_syntax[:, :n_keep_h]
            + self.gtrm_gate[:, :n_keep_h].to(dtype) * v_macro[:, :n_keep_h]
        )
        y_base_ind = (
            self.syntax_gate[:, n_keep_h:].to(dtype) * y_syntax[:, n_keep_h:]
            + self.gtrm_gate[:, n_keep_h:].to(dtype) * v_macro[:, n_keep_h:]
        )
        y_norm_12 = self.head_norm(torch.cat([y_keep, y_base_ind], dim=1))
        if ste_q is not None:
            y_norm_12 = y_norm_12 * ste_q
        y_base_all = y_norm_12.transpose(1, 2).reshape(B, L, self.n_q_heads * self.head_dim)
        gate = torch.sigmoid(gate_raw).to(y_base_all.dtype)
        out_base = self.o_proj(y_base_all * gate)

        yv_ind_src = y_vault[:, n_keep_h:] * ste_q[:, n_keep_h:] if ste_q is not None else y_vault[:, n_keep_h:]
        yv_ind_f = yv_ind_src.transpose(1, 2).float()
        yv_soft_f = yv_ind_f * torch.rsqrt(yv_ind_f.detach().pow(2).mean(dim=-1, keepdim=True) + 2.5e-5)
        ir_lin = ir_x0 + F.linear(x_rms, w_ir[:, d_m : 2 * d_m])
        sig_r = torch.sigmoid(ir_lin.float()).unsqueeze(-1)
        g_read_f = 2.60 * sig_r
        yv_grp = (
            (g_read_f * yv_soft_f)
            .view(B, L, self.n_ind_groups, self.heads_per_group, self.head_dim)
            .mean(dim=3)
            .reshape(B, L, self.n_ind_groups * self.head_dim)
            .to(dtype)
        )
        out = out_base + self.ind_out_proj(yv_grp)

        if return_beta_mean and return_ent_div:
            return out, beta_aux_mean, ent_div_aux
        if return_beta_mean:
            return out, beta_aux_mean
        return out


class VAULTv2_5Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_vault_groups: int,
        head_dim: int,
        d_ff: int,
        chunk_size: int,
        layer_idx: int,
        ckpt_interval: int = 8,
        prime_banks: Tuple[int, ...] = (47, 53, 59, 61),
        ema_alpha: float = GTRM_EMA_ALPHA_DEFAULT,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = VAULTv2_5Attention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_vault_groups=n_vault_groups,
            head_dim=head_dim,
            prime_banks=prime_banks,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            ckpt_interval=ckpt_interval,
            ema_alpha=ema_alpha,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        step_ratio: float = 1.0,
        x0_norm: Optional[torch.Tensor] = None,
        cms_aux=None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), step_ratio=step_ratio, x0_norm=x0_norm, cms_aux=cms_aux)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_with_aux(
        self,
        x: torch.Tensor,
        step_ratio: float = 1.0,
        x0_norm: Optional[torch.Tensor] = None,
        return_ent_div: bool = False,
        cms_aux=None,
    ):
        if return_ent_div:
            attn_out, beta_mean, ent_div = self.attn(
                self.norm1(x),
                step_ratio=step_ratio,
                return_beta_mean=True,
                x0_norm=x0_norm,
                return_ent_div=True,
                cms_aux=cms_aux,
            )
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, beta_mean, ent_div
        attn_out, beta_mean = self.attn(
            self.norm1(x), step_ratio=step_ratio, return_beta_mean=True, x0_norm=x0_norm, cms_aux=cms_aux
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, beta_mean


class VAULTLMv2_5(nn.Module):
    """
    VAULTLM-v2.5 (~180.6M Parameters):
      - Stationary Causal Leaky-EMA GTRM (`alpha = 2^-11`, invariant across `2K -> 128K+`)
      - Tri-Horizon `1..11`-BPE-Subword `SalienceEntityBinderV2_5` (`Vault Group 3` extended sibling channel)
      - Lossless Step-0 checkpoint inheritance from `VAULT v2.4` + zero-shot Co-Prime Voronoi Mitosis
        (`220 -> 508 -> 942 -> 2,002` cells).
    """

    def __init__(
        self,
        vocab_size: int = 49152,
        d_model: int = 768,
        n_layers: int = 16,
        n_q_heads: int = 12,
        n_vault_groups: int = 4,
        head_dim: int = 64,
        d_ff: int = 2048,
        chunk_size: int = 64,
        grad_checkpoint=False,
        ckpt_interval: Optional[int] = None,
        prime_banks: Tuple[int, ...] = (47, 53, 59, 61),
        ema_alpha: float = GTRM_EMA_ALPHA_DEFAULT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_q_heads = n_q_heads
        self.n_vault_groups = n_vault_groups
        self.head_dim = head_dim
        self.d_ff = d_ff
        self.chunk_size = chunk_size
        self.grad_checkpoint = grad_checkpoint
        self.v2_2_mode = True
        self.v2_1_mode = True
        self.v2_5_mode = True
        self.prime_banks = tuple(int(p) for p in prime_banks)
        self.ckpt_interval = ckpt_interval if ckpt_interval is not None else 8
        self.ema_alpha = float(ema_alpha)

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        self.x0_rms_norm = RMSNorm(d_model)

        self.layers = nn.ModuleList(
            [
                VAULTv2_5Block(
                    d_model=d_model,
                    n_q_heads=n_q_heads,
                    n_vault_groups=n_vault_groups,
                    head_dim=head_dim,
                    d_ff=d_ff,
                    chunk_size=chunk_size,
                    layer_idx=i,
                    ckpt_interval=self.ckpt_interval,
                    prime_banks=self.prime_banks,
                    ema_alpha=self.ema_alpha,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        res_std = 0.02 / math.sqrt(2.0 * n_layers)
        for blk in self.layers:
            nn.init.normal_(blk.attn.o_proj.weight, mean=0.0, std=res_std)
            nn.init.normal_(blk.mlp.w3.weight, mean=0.0, std=res_std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compute_x0_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.x0_rms_norm(x.detach())

    def compute_cms_aux(self, x0_norm: torch.Tensor):
        b0 = self.layers[0].attn.salience_binder
        return compute_v2_5_shared_aux(x0_norm, b0, ema_alpha=self.ema_alpha)

    def _encode_hidden_with_aux(
        self, input_ids: torch.Tensor, step_ratio: float = 1.0, return_ent_div: bool = False
    ):
        x = self.embed_tokens(input_ids)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        x0_norm = self.compute_x0_norm(x)
        cms_aux = self.compute_cms_aux(x0_norm)

        beta_acc = torch.zeros((), device=x.device, dtype=torch.float32)
        ent_div_acc = torch.zeros((), device=x.device, dtype=torch.float32)
        for idx, layer in enumerate(self.layers):
            use_ckpt = False
            if self.training and torch.is_grad_enabled():
                if self.grad_checkpoint in (False, None, "none"):
                    use_ckpt = False
                elif self.grad_checkpoint == "quarter":
                    use_ckpt = idx % 4 == 0
                elif self.grad_checkpoint == "half":
                    use_ckpt = idx % 2 == 0
                elif self.grad_checkpoint == "three_quarter":
                    use_ckpt = idx % 4 != 3
                else:
                    use_ckpt = True
            if return_ent_div:
                if use_ckpt:
                    x, b_m, e_d = torch.utils.checkpoint.checkpoint(
                        layer.forward_with_aux, x, step_ratio, x0_norm, True, cms_aux, use_reentrant=False
                    )
                else:
                    x, b_m, e_d = layer.forward_with_aux(
                        x, step_ratio=step_ratio, x0_norm=x0_norm, return_ent_div=True, cms_aux=cms_aux
                    )
                ent_div_acc = ent_div_acc + e_d
            else:
                if use_ckpt:
                    x, b_m = torch.utils.checkpoint.checkpoint(
                        layer.forward_with_aux, x, step_ratio, x0_norm, False, cms_aux, use_reentrant=False
                    )
                else:
                    x, b_m = layer.forward_with_aux(x, step_ratio=step_ratio, x0_norm=x0_norm, cms_aux=cms_aux)
            beta_acc = beta_acc + b_m
        inv_l = 1.0 / float(len(self.layers))
        if return_ent_div:
            return self.norm_f(x), beta_acc * inv_l, ent_div_acc * inv_l
        return self.norm_f(x), beta_acc * inv_l

    def _encode_hidden(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        return self._encode_hidden_with_aux(input_ids, step_ratio=step_ratio)[0]

    def forward(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        return self.lm_head(self._encode_hidden(input_ids, step_ratio=step_ratio))

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        step_ratio: float = 1.0,
        logit_chunk_tokens: int = 2048,
        return_beta_mean: bool = False,
        return_ent_div: bool = False,
    ):
        if return_ent_div:
            h_raw, beta_mean, ent_div_mean = self._encode_hidden_with_aux(
                input_ids, step_ratio=step_ratio, return_ent_div=True
            )
        else:
            h_raw, beta_mean = self._encode_hidden_with_aux(input_ids, step_ratio=step_ratio)
            ent_div_mean = torch.zeros((), device=h_raw.device, dtype=torch.float32)
        h = h_raw.reshape(-1, self.d_model)
        t_flat = targets.reshape(-1)
        w_flat = loss_weights.reshape(-1) if loss_weights is not None else None
        n_tok = h.size(0)

        def _chunk_loss_sum(h_sub: torch.Tensor, t_sub: torch.Tensor, w_sub: Optional[torch.Tensor]) -> torch.Tensor:
            logits_sub = self.lm_head(h_sub)
            if w_sub is not None:
                ce = F.cross_entropy(logits_sub.float(), t_sub, ignore_index=-100, reduction="none")
                return (ce * w_sub).sum()
            return F.cross_entropy(logits_sub.float(), t_sub, ignore_index=-100, reduction="sum")

        total_loss = torch.zeros((), device=h.device, dtype=torch.float32)
        if w_flat is not None:
            denom = w_flat.sum().clamp(min=1.0)
        else:
            denom = (t_flat != -100).float().sum().clamp(min=1.0)

        for start in range(0, n_tok, logit_chunk_tokens):
            end = min(start + logit_chunk_tokens, n_tok)
            hs = h[start:end]
            ts = t_flat[start:end]
            ws = w_flat[start:end] if w_flat is not None else None
            if self.training and torch.is_grad_enabled() and n_tok > logit_chunk_tokens:
                total_loss = total_loss + torch.utils.checkpoint.checkpoint(
                    _chunk_loss_sum, hs, ts, ws, use_reentrant=False
                )
            else:
                total_loss = total_loss + _chunk_loss_sum(hs, ts, ws)

        ce_loss = total_loss / denom
        if return_beta_mean and return_ent_div:
            return ce_loss, beta_mean, ent_div_mean
        if return_beta_mean:
            return ce_loss, beta_mean
        return ce_loss

    def forward_mc_contrastive_loss(
        self,
        prompt_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        choice_ids: torch.Tensor,
        choice_lens: torch.Tensor,
        choice_bytes: torch.Tensor,
        n_choices: torch.Tensor,
        gold_idx: torch.Tensor,
        eos_id: int = 0,
        step_ratio: float = 1.0,
        byte_temp: float = 8.0,
        tok_temp: float = 2.2,
    ) -> Tuple[torch.Tensor, float]:
        """
        Memory-sparse 2..4-way multiple-choice contrastive choice-margin loss:
          - Evaluates `lm_head` strictly on candidate continuation positions (`~250..400` tokens total per batch),
            avoiding full-sequence `(B*4*L, V)` vocab GEMMs!
          - Optimizes `0.65 * CE(byte_temp * sum_logp / bytes, gold) + 0.35 * CE(tok_temp * sum_logp / n_tok, gold)`,
            jointly calibrating `acc_norm` (HellaSwag, ARC-C/E, OBQA) and `acc` (PIQA, SciQ, BoolQ).
        """
        dev = prompt_ids.device
        B_mc = prompt_ids.size(0)
        seqs = []
        meta = []  # (item_idx, cand_idx, p_len, c_len)
        max_len = 0
        for i in range(B_mc):
            p_l = int(prompt_lens[i].item())
            k_n = int(n_choices[i].item())
            p_slice = prompt_ids[i, :p_l]
            for j in range(k_n):
                c_l = int(choice_lens[i, j].item())
                c_slice = choice_ids[i, j, :c_l]
                cat_ij = torch.cat([p_slice, c_slice], dim=0)
                seqs.append(cat_ij)
                meta.append((i, j, p_l, c_l))
                if cat_ij.numel() > max_len:
                    max_len = cat_ij.numel()

        pad_len = ((max_len + 63) // 64) * 64
        batch_ids = torch.full((len(seqs), pad_len), int(eos_id), device=dev, dtype=torch.long)
        for row_idx, s in enumerate(seqs):
            batch_ids[row_idx, : s.numel()] = s.long()

        h_batch = self._encode_hidden(batch_ids, step_ratio=step_ratio)  # (N_seq, pad_len, d_model)

        # Gather ONLY choice-predicting hidden vectors across all candidates into a single tiny tensor!
        h_gather = []
        tgt_gather = []
        seg_slices = []
        cur = 0
        for row_idx, (i, j, p_l, c_l) in enumerate(meta):
            h_gather.append(h_batch[row_idx, (p_l - 1) : (p_l - 1 + c_l), :])
            tgt_gather.append(batch_ids[row_idx, p_l : (p_l + c_l)])
            seg_slices.append((i, j, cur, cur + c_l))
            cur += c_l

        h_cat = torch.cat(h_gather, dim=0)    # (~320, d_model)
        t_cat = torch.cat(tgt_gather, dim=0)  # (~320,)
        tok_nll = F.cross_entropy(self.lm_head(h_cat).float(), t_cat, reduction="none")  # -log p(tok)

        scores_byte = torch.full((B_mc, 4), -1e9, device=dev, dtype=torch.float32)
        scores_tok = torch.full((B_mc, 4), -1e9, device=dev, dtype=torch.float32)
        for i, j, s0, s1 in seg_slices:
            sum_logp = -tok_nll[s0:s1].sum()
            b_ij = choice_bytes[i, j].float().clamp(min=1.0)
            c_ij = float(max(1, s1 - s0))
            scores_byte[i, j] = byte_temp * (sum_logp / b_ij)
            scores_tok[i, j] = tok_temp * (sum_logp / c_ij)

        g_long = gold_idx.long()
        loss = 0.65 * F.cross_entropy(scores_byte, g_long) + 0.35 * F.cross_entropy(scores_tok, g_long)
        with torch.no_grad():
            acc = float((scores_byte.argmax(dim=-1) == g_long).float().mean().item())
        return loss, acc

    def forward_wg_contrastive_loss(
        self,
        seq_ids: torch.Tensor,
        suf_start: torch.Tensor,
        suf_len: torch.Tensor,
        gold_idx: torch.Tensor,
        step_ratio: float = 1.0,
        suf_temp: float = 2.6,
    ) -> Tuple[torch.Tensor, float]:
        """
        Exact Winogrande suffix-scoring contrastive loss: scores identical suffix tokens conditioned on
        `prefix + option1` vs `prefix + option2` using sparse hidden-state gathering.
        """
        dev = seq_ids.device
        B_wg = seq_ids.size(0)
        flat_ids = seq_ids.reshape(B_wg * 2, seq_ids.size(-1)).long()
        h_batch = self._encode_hidden(flat_ids, step_ratio=step_ratio)

        h_gather = []
        t_gather = []
        seg_slices = []
        cur = 0
        for i in range(B_wg):
            for j in range(2):
                row_idx = 2 * i + j
                st = int(suf_start[i, j].item())
                ln = int(suf_len[i, j].item())
                h_gather.append(h_batch[row_idx, (st - 1) : (st - 1 + ln), :])
                t_gather.append(flat_ids[row_idx, st : (st + ln)])
                seg_slices.append((i, j, cur, cur + ln))
                cur += ln

        h_cat = torch.cat(h_gather, dim=0)
        t_cat = torch.cat(t_gather, dim=0)
        tok_nll = F.cross_entropy(self.lm_head(h_cat).float(), t_cat, reduction="none")

        scores = torch.zeros((B_wg, 2), device=dev, dtype=torch.float32)
        for i, j, s0, s1 in seg_slices:
            sum_logp = -tok_nll[s0:s1].sum()
            scores[i, j] = suf_temp * (sum_logp / float(max(1, s1 - s0)))

        g_long = gold_idx.long()
        loss = F.cross_entropy(scores, g_long)
        with torch.no_grad():
            acc = float((scores.argmax(dim=-1) == g_long).float().mean().item())
        return loss, acc


def _unwrap_checkpoint_state_dict(ckpt_or_sd: Dict[str, object]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "model", "state_dict"):
        sub = ckpt_or_sd.get(key)
        if isinstance(sub, dict):
            return { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sub.items() }
    return { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ckpt_or_sd.items() }


@torch.no_grad()
def load_v2_4_into_v2_5(
    model_v2_5: VAULTLMv2_5,
    v2_4_state: Dict[str, torch.Tensor],
    init_ext_span_alpha: float = 0.0,
) -> Dict[str, object]:
    """
    Loads a `VAULT v2.3 / v2.4` checkpoint (`state_dict` or wrapper dict with `'model_state'`/`'model'`)
    into `VAULTLMv2_5` losslessly, setting `ext_span_alpha` to `init_ext_span_alpha` (`0.0` for exact Step-0
    parity tests, `0.55` when launching SFT so `Vault Group 3` immediately separates `8..11`-subword siblings).
    """
    clean_sd = _unwrap_checkpoint_state_dict(v2_4_state)

    incompatible = model_v2_5.load_state_dict(clean_sd, strict=False)
    allowed_missing = {"ind_shared_proj_g2", "ext_span_alpha"}
    unexpected_missing = [
        k for k in incompatible.missing_keys if not any(k.endswith("." + s) for s in allowed_missing)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint key mismatch loading v2.4 -> v2.5! "
            f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
        )

    for blk in model_v2_5.layers:
        blk.attn.ind_shared_proj_g2.zero_()
        blk.attn.ext_span_alpha.fill_(float(init_ext_span_alpha))

    return {
        "loaded_tensors": len(clean_sd),
        "new_v2_5_tensors": len(incompatible.missing_keys),
        "init_ext_span_alpha": float(init_ext_span_alpha),
    }


@torch.no_grad()
def upgrade_v2_5_to_wider_primes(
    model_dst: VAULTLMv2_5,
    src_state: Dict[str, torch.Tensor],
    old_primes: Tuple[int, ...] = (47, 53, 59, 61),
    split_angle_rad: float = 0.52,
    k_nz: int = 52,
) -> Dict[str, object]:
    """
    Zero-Shot Voronoi Cell Mitosis for `VAULTLMv2_5` (`220 -> 508 -> 942 -> 2,002` cells):
    Identical simplex-cone daughter splitting and uniform `k_nz = 52` ternary L1 equalization as `v2.3/v2.4`,
    preserving `ind_shared_proj_g2` and `ext_span_alpha` across all 16 layers.
    """
    v2_5_sd = _unwrap_checkpoint_state_dict(src_state)

    new_primes = model_dst.prime_banks
    assert len(old_primes) == len(new_primes) == 4
    old_total = sum(old_primes)
    new_total = sum(new_primes)

    dst_sd = model_dst.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    copied_exact = 0
    mitosis_layers = 0
    cos_t = math.cos(split_angle_rad)
    sin_t = math.sin(split_angle_rad)

    for k, dst_t in dst_sd.items():
        if k.endswith(".attn.harmonic_anchors"):
            new_sd[k] = dst_t.clone()
            continue
        if k not in v2_5_sd:
            if k.endswith(".ind_shared_proj_g2") or k.endswith(".ext_span_alpha"):
                new_sd[k] = dst_t.clone()
                continue
            raise KeyError(f"Missing expected v2.5 checkpoint key during mitosis: {k}")
        src_t = v2_5_sd[k]
        if src_t.shape == dst_t.shape:
            new_sd[k] = src_t.to(device=dst_t.device, dtype=dst_t.dtype)
            copied_exact += 1
            continue

        if not k.endswith(".attn.kenyon_proj.weight"):
            raise ValueError(f"Unexpected shape mismatch on {k}: src={src_t.shape} vs dst={dst_t.shape}")

        assert src_t.shape[0] == old_total and dst_t.shape[0] == new_total
        D = dst_t.shape[1]
        layer_idx = int(k.split(".")[1])
        w_old = src_t.detach().float().cpu()
        old_mean_abs = float(w_old.abs().mean().item())

        rows_out = []
        off_old = 0
        for r_idx, (p_old, p_new) in enumerate(zip(old_primes, new_primes)):
            base_m = p_new // p_old
            rem_m = p_new % p_old
            assert base_m >= 1
            bank_rows = []
            for c_idx in range(p_old):
                m_c = base_m + (1 if c_idx < rem_m else 0)
                u_c = F.normalize(w_old[off_old + c_idx], p=2, dim=0, eps=1e-6)
                if m_c == 1:
                    dirs = [u_c]
                else:
                    gen = torch.Generator().manual_seed(20261101 + 10000 * layer_idx + 1000 * r_idx + c_idx)
                    z = torch.randn((m_c, D), generator=gen, dtype=torch.float32)
                    z = z - (z @ u_c).unsqueeze(-1) * u_c.unsqueeze(0)
                    z = z - z.mean(dim=0, keepdim=True)
                    z = z - (z @ u_c).unsqueeze(-1) * u_c.unsqueeze(0)
                    v_simp = F.normalize(z, p=2, dim=-1, eps=1e-6)
                    dirs = [F.normalize(cos_t * u_c + sin_t * v_simp[j], p=2, dim=0, eps=1e-6) for j in range(m_c)]

                for d_vec in dirs:
                    signs = torch.where(d_vec >= 0.0, torch.ones_like(d_vec), -torch.ones_like(d_vec))
                    mags = d_vec.abs()
                    vals_sorted, _ = torch.sort(mags, descending=True)
                    kth = vals_sorted[k_nz - 1]
                    top_m = mags >= kth
                    if int(top_m.sum().item()) > k_nz:
                        top_idx = torch.topk(mags, k=k_nz, largest=True).indices
                        top_m = torch.zeros_like(top_m)
                        top_m[top_idx] = True
                    m_max = vals_sorted[0]
                    scaled_top = 0.45 + 1.05 * (mags - kth).clamp(min=0.0) / (m_max - kth + 1e-6)
                    scaled_bot = 0.11 * (mags / (kth + 1e-6)).clamp(max=1.0)
                    row_eq = signs * torch.where(top_m, scaled_top, scaled_bot)
                    bank_rows.append(row_eq)
            assert len(bank_rows) == p_new
            rows_out.extend(bank_rows)
            off_old += p_old

        w_new = torch.stack(rows_out, dim=0)
        w_new = w_new * (old_mean_abs / w_new.abs().mean().clamp(min=1e-6))
        thresh_new = 0.25 * w_new.abs().mean()
        nz_counts = (w_new.abs() > thresh_new).sum(dim=-1)
        assert int(nz_counts.min().item()) == k_nz and int(nz_counts.max().item()) == k_nz
        new_sd[k] = w_new.to(device=dst_t.device, dtype=dst_t.dtype)
        mitosis_layers += 1

    model_dst.load_state_dict(new_sd, strict=True)
    return {
        "copied_exact_tensors": copied_exact,
        "mitosis_kenyon_layers": mitosis_layers,
        "old_primes": old_primes,
        "new_primes": new_primes,
        "old_cells": old_total,
        "new_cells": new_total,
        "crt_capacity": new_primes[0] * new_primes[1] * new_primes[2] * new_primes[3],
    }
