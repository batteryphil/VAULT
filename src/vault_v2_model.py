# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2 (`VAULTLMv2`): Second-Generation Co-Prime Associative Fast-Weight Language Model
# Built strictly from Philip Bartsch's owned IP (`vault`, `PRIME-Moment-Attention`, `BioNPC`,
# `gist-memory`, `buildbit-os`, `mamba1and2-to-3`) + standard PyTorch primitives.
#
# Architectural Advances over VAULT v1:
#   1. True 3:1 Grouped-Vault Attention (`n_q_heads=12`, `n_vault_groups=4` shared 88-cell Co-Prime Vaults):
#      3x smaller fast-weight state & HBM traffic while allowing 3 independent query heads per group
#      to probe 3 distinct Co-Prime cells concurrently.
#   2. Causal Multi-Token `ShortConv1D(kernel_size=4)` + SiLU on Q, K, V:
#      Binds multi-subword identifiers (e.g. `ShieldPod-A7`, `reverse_i32`) into single key/query vectors
#      from Layer 0, eliminating v1's 1-token `k_prev` blind spot.
#   3. Token-Selective Write/Erase Gate `beta_{b,g,t} = sigmoid(W_beta(x_t))`:
#      Skips overwriting Co-Prime cells on filler/boilerplate tokens while committing salient key-value
#      bindings via Compact WY Sherman-Morrison orthogonalization.
#   4. Chunk-Invariant Relative RoPE + Multi-Scale Geometric Decay (`gamma_h in [0.65, 0.992]`)
#      inside the strictly O(L) `64 x 128` local syntactic branch (half-lives spanning 1.6 to 86 tokens).
#   5. Per-Head Sub-Layer RMSNorm + Data-Dependent Output Gate before `o_proj`.

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vault_lm_150m import RMSNorm, FusedAPLNorm, TernarySTEProjection, SwiGLUMLP
from src.kernels.flash_vault_v2_triton import flash_vault_v2_chunkwise, causal_depthwise_conv1d


def _safe_orthogonal_(weight: torch.Tensor) -> None:
    if weight.device.type == "cpu" and weight.dtype in (torch.bfloat16, torch.float16):
        tmp = torch.empty(weight.shape, dtype=torch.float32, device="cpu")
        nn.init.orthogonal_(tmp)
        with torch.no_grad():
            weight.copy_(tmp.to(weight.dtype))
    else:
        nn.init.orthogonal_(weight)


class CausalShortConv1D(nn.Module):
    """
    Depthwise causal 1D convolution (`kernel_size=4`) + SiLU activation.
    Gives every query, key, and value vector native 4-token n-gram reception in O(L) time.
    """
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=channels,
            bias=True,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> causal (B, L, C) via fused contiguous Triton kernel (zero transpose copies)
        return causal_depthwise_conv1d(x, self.conv.weight, self.conv.bias, apply_silu=True)


def _apply_local_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies chunk-relative rotary phase to `(..., seq, D)` with zero long-range RoPE decay."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class MultiOffsetSpanBinder(nn.Module):
    """
    Strictly O(L) Causal Multi-Scale Span Key/Query Binder (`VAULT v2.1`).
    Tiles `[t-38 .. t-1]` with 7 contiguous non-overlapping causal spans:
      (1,2), (3,5), (6,9), (10,14), (15,20), (21,28), (29,38)
    with ZERO blind spots across 1..38 token entity-to-value gaps and O(1) 38-token ring-buffer state.
    Pools the pre-trained `qk_group_anchor` metric space with separate data-dependent write (`k`)
    and query (`q`) gates, using zero-initialized per-group / per-head output projections so Step-0
    forward outputs match the Stage-1 VAULT v2 checkpoint while waking up gradients immediately on Step 1!
    """
    SPANS: Tuple[Tuple[int, int], ...] = (
        (1, 2),
        (3, 5),
        (6, 9),
        (10, 14),
        (15, 20),
        (21, 28),
        (29, 38),
    )

    def __init__(self, d_model: int, n_q_heads: int, n_vault_groups: int, head_dim: int):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_vault_groups = n_vault_groups
        self.heads_per_group = n_q_heads // n_vault_groups
        self.head_dim = head_dim
        self.n_spans = len(self.SPANS)

        self.k_gate = nn.Linear(d_model, self.n_spans * n_vault_groups, bias=True)
        self.q_gate = nn.Linear(d_model, self.n_spans * n_q_heads, bias=True)
        nn.init.normal_(self.k_gate.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.q_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.k_gate.bias, -0.5)
        nn.init.constant_(self.q_gate.bias, -1.0)

        self.k_bind_proj = nn.Parameter(torch.zeros(n_vault_groups, head_dim, head_dim))
        self.q_bind_proj = nn.Parameter(torch.zeros(n_q_heads, head_dim, head_dim))

    def forward(self, x: torch.Tensor, anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, kv_dim = anchor.shape
        dtype = anchor.dtype
        a_t = anchor.float().transpose(1, 2)  # (B, G*D, L)

        span_list = []
        for d_near, d_far in self.SPANS:
            w = d_far - d_near + 1
            padded = F.pad(a_t, (d_far, 0))[..., : L + w - 1]
            pooled = F.avg_pool1d(padded, kernel_size=w, stride=1)  # (B, G*D, L)
            span_list.append(pooled.transpose(1, 2).view(B, L, self.n_vault_groups, self.head_dim))

        spans = torch.stack(span_list, dim=2)  # (B, L, S, G, D)

        gk = torch.sigmoid(self.k_gate(x).float()).view(B, L, self.n_spans, self.n_vault_groups, 1)
        k_ctx = (spans * gk).sum(dim=2).to(dtype)  # (B, L, G, D)
        k_delta_bind = torch.einsum("blgd,gde->blge", k_ctx, self.k_bind_proj.to(dtype)).reshape(B, L, kv_dim)

        gq = torch.sigmoid(self.q_gate(x).float()).view(
            B, L, self.n_spans, self.n_vault_groups, self.heads_per_group, 1
        )
        q_ctx = (spans.unsqueeze(4) * gq).sum(dim=2).reshape(B, L, self.n_q_heads, self.head_dim).to(dtype)
        q_delta_bind = torch.einsum("blhd,hde->blhe", q_ctx, self.q_bind_proj.to(dtype)).reshape(
            B, L, self.n_q_heads * self.head_dim
        )
        return q_delta_bind, k_delta_bind


def fft_circular_bind(u: torch.Tensor, lex: torch.Tensor) -> torch.Tensor:
    """Isometric circular convolution in R^(64) via RFFT unit-phase modulation."""
    u_f = u.float()
    l_f = lex.float()
    b_phase = F.normalize(torch.fft.rfft(l_f, dim=-1), p=2, dim=-1, eps=1e-6)
    return torch.fft.irfft(torch.fft.rfft(u_f, dim=-1) * b_phase, n=u_f.size(-1), dim=-1)


def fft_bind_with_id(u: torch.Tensor, tail: torch.Tensor, id_weight=1.5) -> torch.Tensor:
    """Identity-anchored RFFT phase binding (`tail == 0 -> exact identity`, `tail != 0 -> all-bin phase rotation`)."""
    u_f = u.float()
    t_f = tail.float()
    uf_fft = torch.fft.rfft(u_f, dim=-1)
    tf_fft = id_weight + torch.fft.rfft(t_f, dim=-1)
    phase = tf_fft / (tf_fft.abs() + 1e-6)
    return torch.fft.irfft(uf_fft * phase, n=u_f.size(-1), dim=-1)


def _shift_right_zero(t: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """Strictly causal right-shift along `dim=1` (`(B, L, C)`) padded with zeros on the left."""
    if t.size(1) <= steps:
        return torch.zeros_like(t)
    return F.pad(t[:, :-steps, :], (0, 0, steps, 0))


def _chunked_cumsum_dim2(v_f: torch.Tensor, block_len: int = 256) -> torch.Tensor:
    """
    2-level parallel block prefix sum along `dim=2` for `(B, G, L, D)` FP32 tensors.
    Eliminates ROCm sequential strided scan bottleneck at `L=32K+` (`154 ms -> <8 ms`) with exact FP32 accuracy.
    """
    L = v_f.size(2)
    if L <= block_len:
        return torch.cumsum(v_f, dim=2)
    rem = L % block_len
    pad_r = (block_len - rem) if rem != 0 else 0
    vp = F.pad(v_f, (0, 0, 0, pad_r)) if pad_r > 0 else v_f
    M = vp.size(2) // block_len
    vb = vp.view(v_f.size(0), v_f.size(1), M, block_len, v_f.size(3))
    loc = torch.cumsum(vb, dim=3)
    if M > 1:
        totals = loc[:, :, :-1, -1, :]
        pref = F.pad(torch.cumsum(totals, dim=2), (0, 0, 1, 0)).unsqueeze(3)
        loc = loc + pref
    out = loc.reshape(v_f.size(0), v_f.size(1), M * block_len, v_f.size(3))
    return out[:, :, :L, :] if pad_r > 0 else out


@torch.no_grad()
def compute_windowed_cms_aux(
    x0_norm: torch.Tensor,
    cone_mu: torch.Tensor,
    cms_proj: torch.Tensor,
    banks: Tuple[int, ...] = (503, 509, 521),
    lag: int = 32,
    win: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Strictly causal O(L) Co-Prime `(503, 509, 521)` exact `int32` Windowed Count-Min Sketch (`[t-1056 .. t-32]`).
    Computes unigram (`uf`) and bigram (`bf`) local document frequency normalized per ~110-token span,
    plus anisotropic embedding-cone alignment (`cone_cos`, `cone_strength`), invariant from `2K` to `32K+`.
    """
    B, L, d_model = x0_norm.shape
    dev = x0_norm.device
    x0_f = x0_norm.detach().float()
    cm = cone_mu.detach().float().view(1, 1, d_model)
    cone_norm = cm.norm(dim=-1, keepdim=True)
    cone_strength = (cone_norm / ((d_model ** 0.5) * 0.30)).clamp(max=1.0)
    cone_u = cm / cone_norm.clamp(min=1e-6)
    cone_cos = torch.sum(F.normalize(x0_f, p=2, dim=-1, eps=1e-6) * cone_u, dim=-1, keepdim=True)

    x0_c = F.normalize(x0_f - cm, p=2, dim=-1, eps=1e-6)
    proj = torch.matmul(x0_c, cms_proj.detach().float().T)

    idx = torch.arange(L, device=dev)
    hi = (idx - lag).clamp(min=0)
    lo = (idx - lag - win).clamp(min=0)
    span = (hi - lo).float().clamp(min=1.0).unsqueeze(0)
    n_sec_est = ((span - 15.0) / 110.0).clamp(min=1.0)
    b_idx = torch.arange(B, device=dev).unsqueeze(1)
    hi_2d = hi.unsqueeze(0)
    lo_2d = lo.unsqueeze(0)

    min_u, min_b = None, None
    off = 0
    for p in banks:
        ib = proj[..., off : off + p].argmax(dim=-1)
        ib_p1 = F.pad(ib[:, :-1], (1, 0), value=0)
        bg = (ib_p1 * 131 + ib) % p

        ohu = F.one_hot(ib, num_classes=p).to(torch.int32)
        csu = torch.cat([torch.zeros((B, 1, p), device=dev, dtype=torch.int32), torch.cumsum(ohu, dim=1)], dim=1)
        cu = (csu[b_idx, hi_2d, ib] - csu[b_idx, lo_2d, ib]).float()

        ohb = F.one_hot(bg, num_classes=p).to(torch.int32)
        if L > 0:
            ohb[:, 0, :] = 0
        csb = torch.cat([torch.zeros((B, 1, p), device=dev, dtype=torch.int32), torch.cumsum(ohb, dim=1)], dim=1)
        cb = (csb[b_idx, hi_2d, bg] - csb[b_idx, lo_2d, bg]).float()

        min_u = cu if min_u is None else torch.minimum(min_u, cu)
        min_b = cb if min_b is None else torch.minimum(min_b, cb)
        off += p

    hh_1d = (hi - lo >= 45).float().unsqueeze(0)
    uf = (min_u / n_sec_est * hh_1d).unsqueeze(-1)
    bf = (min_b / n_sec_est * hh_1d).unsqueeze(-1)
    hh = hh_1d.unsqueeze(-1).expand(B, L, 1)
    return cone_cos, uf, bf, hh, cone_strength


@torch.no_grad()
def compute_v2_2_shared_aux(x0_norm: torch.Tensor, binder: "SalienceEntityBinder") -> Dict[str, object]:
    """
    Precomputes all 100% layer-invariant `x0_norm` / CMS / novelty / causal step-keep quantities
    once per forward pass instead of recomputing them across all 16 layers (and again in grad checkpointing).
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

    bf_pad = F.pad(bf, (0, 0, 0, 7))
    cc_pad = F.pad(cone_cos, (0, 0, 0, 7))

    def step_keep(m: int, cc_bias: float) -> torch.Tensor:
        cc_m = cc_pad[:, m : L + m, :]
        bf_m = bf_pad[:, m : L + m, :]
        bf_next = bf_pad[:, m + 1 : L + m + 1, :]
        non_punct = 1.0 - cone_strength + cone_strength * torch.sigmoid(10.0 * (cc_m + cc_bias))
        return (torch.sigmoid(8.0 * (0.68 - bf_m)) * torch.sigmoid(8.0 * (0.72 - bf_next)) * non_punct).unsqueeze(2)

    k0_steps = (step_keep(1, 0.32), step_keep(2, 0.32), step_keep(3, 0.32))
    k1_steps = tuple(step_keep(m, 0.36) for m in range(1, 7))

    aux_f = torch.cat([cone_cos, uf.clamp(max=6.0), bf.clamp(max=6.0), hh], dim=-1).to(dtype)
    rep_pen = (
        9.0 * F.relu(uf - 0.42)
        + 7.0 * F.relu(bf - 0.35)
        + (8.0 * cone_strength) * F.relu(0.08 - cone_cos)
    ).clamp(max=15.0).transpose(1, 2)

    nov1 = torch.sigmoid(12.0 * (0.62 - bf)).pow(2).transpose(1, 2)
    nov4 = F.max_pool1d(F.pad(nov1, (3, 0)), kernel_size=4, stride=1)
    inv_counts_f = (1.0 / torch.arange(1, L + 1, device=x0_norm.device, dtype=torch.float32)).view(1, 1, L, 1)

    return {
        "cms": (cone_cos, uf, bf, hh, cone_strength),
        "shape": (B, L),
        "x0_iso": x0_iso,
        "nov_stem": nov_stem,
        "k0_steps": k0_steps,
        "k1_steps": k1_steps,
        "aux_f": aux_f,
        "rep_pen": rep_pen,
        "nov4": nov4,
        "inv_counts_f": inv_counts_f,
    }


class SalienceEntityBinder(nn.Module):
    """
    Strictly O(L), Mode- & Length-Invariant Voronoi-Decoupled Causal Entity Selector (`VAULT v2.2` / `v2.3`):
      1. `ent_key_proj` (`768 -> 256` orthogonal): `[:128]` builds continuous subspace entity keys `ent_u` while
         `[128:256]` builds statistically orthogonal Level-1 Voronoi cell-router entity keys `ent_r`, eliminating
         within-cell ternary Kenyon selection bias across `8K -> 32K` manifests.
      2. Dual-horizon strictly causal `fft_bind_with_id` right-suffix phase binding:
         - Group 0 (`diff >= 4`, `m = 1..3`): exact 1..4-subword entity capture without delimiter overshoot.
         - Group 1 (`diff >= 7`, `m = 1..6`): full 5..7-subword multi-digit suffix disambiguation (`96..396+` sections).
      3. Windowed Co-Prime `(503, 509, 521)` CMS novelty suppression (`rep_pen`) + `64 x 128` WMMA block pooling.
    """
    BANKS: Tuple[int, ...] = (503, 509, 521)

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

        # Orthogonal 256D projection: [:128] = continuous entity key `ent_u`, [128:256] = Voronoi router `ent_r`
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
        # Strict causality: G0 `diff >= 4` (`m=1..3 -> k+4 <= q`), G1 `diff >= 7` (`m=1..6 -> k+7 <= q`)
        mask_g0 = (diff >= 4) & (diff <= window_size)
        mask_g1 = (diff >= 7) & (diff <= window_size)
        lag_mask = torch.stack([mask_g0, mask_g1], dim=0).unsqueeze(0).unsqueeze(2)
        self.register_buffer("sal_lag_mask", lag_mask)
        self.register_buffer("sal_lag_dist", diff.clamp(min=0).float().view(1, 1, 1, chunk_size, 2 * chunk_size))
        self.register_buffer("_id_weights_2g", torch.tensor([1.50, 1.20], dtype=torch.float32).view(1, 1, 2, 1, 1), persistent=False)

    def forward(
        self,
        x_rms: torch.Tensor,
        x0_norm: torch.Tensor,
        cms_aux=None,
        pre_x0_iso_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_orig, d_m = x0_norm.shape
        G, D, C = self.n_ind_groups, self.head_dim, self.chunk_size
        dtype = x0_norm.dtype

        if isinstance(cms_aux, dict) and cms_aux.get("shape") == (B, L_orig):
            sh = cms_aux
        else:
            sh = compute_v2_2_shared_aux(x0_norm, self)

        x0_iso = sh["x0_iso"]
        nov_stem = sh["nov_stem"]
        k0_1, k0_2, k0_3 = sh["k0_steps"]
        k1_1, k1_2, k1_3, k1_4, k1_5, k1_6 = sh["k1_steps"]
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
            (g_tok * a_raw).to(dtype), self.anchor_ent_conv.weight, None, apply_silu=False
        ).float()
        a_left = a_raw * (0.65 + 0.85 * torch.tanh(1.5 * a_stem)) + 0.45 * a_stem

        a_raw_g = a_raw.view(B, L_orig, 2, 2, D)   # (B, L, G=2, ur=2, D=64)
        a_left_g = a_left.view(B, L_orig, 2, 2, D)
        e0_pad = F.pad(a_raw_g[:, :, 0], (0, 0, 0, 0, 0, 7))
        e1_pad = F.pad(a_raw_g[:, :, 1], (0, 0, 0, 0, 0, 7))

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

        tail_both = torch.stack([tail0, tail1], dim=2)
        a_ctx = (
            fft_bind_with_id(a_left_g, tail_both, id_weight=self._id_weights_2g)
            .reshape(B, L_orig, 2, 2 * D)
            .to(dtype)
            .permute(0, 2, 1, 3)
        )

        x0_span = causal_depthwise_conv1d(x0_iso, self.sal_span_conv.weight, None, apply_silu=False)
        sal_h = (
            F.linear(x0_span, w_si[:, :d_m], self.sal_in.bias)
            + sal_iso_p1
            + sal_iso_0
            + F.linear(aux_f, w_si[:, 3 * d_m :])
        )
        raw_sal = self.sal_out(F.silu(sal_h)).float().transpose(1, 2)
        sal = 4.0 * torch.tanh(raw_sal / 4.0) - rep_pen

        rem = L_orig % C
        pad_r = (C - rem) if rem != 0 else 0
        if pad_r > 0:
            a_ctx = F.pad(a_ctx, (0, 0, 0, pad_r))
            sal = F.pad(sal, (0, pad_r), value=-1e4)

        L = a_ctx.size(2)
        a_pad = F.pad(a_ctx, (0, 0, C, 0))
        a_win = a_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)  # (B, 2, M, 2C, 128)
        s_pad = F.pad(sal, (C, 0), value=-1e4)
        s_win = s_pad.unfold(dimension=2, size=2 * C, step=C)                         # (B, 2, M, 2C)

        p_win = torch.where(s_win > -5e3, torch.sigmoid(s_win), torch.zeros_like(s_win))
        cs = torch.cumsum(p_win, dim=-1)
        cs_g0 = cs[:, 0:1, :, (C - 4) : (2 * C - 4)]
        cs_g1 = cs[:, 1:2, :, (C - 7) : (2 * C - 7)]
        cs_newest = torch.cat([cs_g0, cs_g1], dim=1).unsqueeze(-1)
        gg = self.gap_guard
        cs_guard = torch.cat([cs[..., gg:], cs[..., -1:].expand(-1, -1, -1, gg)], dim=-1) if gg > 0 else cs
        prior_occ = (cs_newest - cs_guard.unsqueeze(-2)).clamp(min=0.0)

        valid = self.sal_lag_mask & (s_win.unsqueeze(-2) > -5e3)
        eff_logits = (
            s_win.unsqueeze(-2)
            - F.softplus(self.occ_scale.float()) * prior_occ
            - F.softplus(self.recency_param.float()) * self.sal_lag_dist
        )
        eff_logits = torch.where(valid, eff_logits, torch.full_like(eff_logits, -1e4))
        any_valid = valid.any(dim=-1, keepdim=True)
        attn_w = torch.where(any_valid, F.softmax(eff_logits, dim=-1), torch.zeros_like(eff_logits))

        out_blk = torch.matmul(attn_w.to(dtype), a_win)                                 # (B, 2, M, C, 128) WMMA!
        ent_both = out_blk.reshape(B, G, L, 2 * D)[:, :, :L_orig, :].permute(0, 2, 1, 3).float()
        if L_orig > 0:
            ent_both = torch.cat([torch.zeros_like(ent_both[:, :1]), ent_both[:, 1:]], dim=1)
        return ent_both[..., :D], ent_both[..., D:]  # (ent_u, ent_r), each (B, L_orig, 2, D) float32


class VAULTv2Attention(nn.Module):
    """
    True 3:1 Grouped-Vault Attention (GVA) Layer for VAULTLM-v2 / v2.1 / v2.2.
    Strictly O(L) compute & O(1) inference state across all sequence lengths.
    """
    def __init__(
        self,
        d_model: int = 768,
        n_q_heads: int = 12,
        n_vault_groups: int = 4,
        head_dim: int = 64,
        prime_banks: Tuple[int, ...] = (17, 19, 23, 29),
        chunk_size: int = 64,
        conv_kernel: int = 4,
        layer_idx: int = 0,
        v2_1_mode: bool = False,
        ckpt_interval: int = 1,
        v2_2_mode: bool = False,
    ):
        super().__init__()
        assert n_q_heads % n_vault_groups == 0
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
        self.v2_2_mode = v2_2_mode
        self.v2_1_mode = v2_1_mode or v2_2_mode
        self.ckpt_interval = ckpt_interval

        # Shared metric anchor on vault groups so Q and K align in Co-Prime cells from Step 0
        self.qk_group_anchor = nn.Linear(d_model, self.kv_dim, bias=False)
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_delta = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_delta.weight, std=0.02)

        if v2_1_mode and not v2_2_mode:
            self.span_binder = MultiOffsetSpanBinder(d_model, n_q_heads, n_vault_groups, head_dim)

        if v2_2_mode:
            assert n_vault_groups >= 4, "v2_2_mode expects Group 0 semantic + Groups 1..3 Voronoi-decoupled induction"
            self.n_keep_g = 1
            self.n_ind_groups = 3
            self.n_ent_groups = 2

            # Voronoi-Decoupled 2x-wide Orthogonal Lexical Projections (`[:half]` -> continuous key `u_ind`, `[half:]` -> cell router `r_all`)
            self.lex_router_g2 = nn.Linear(2 * d_model, 2 * self.n_ent_groups * head_dim, bias=False)
            self.lex_router_g3 = nn.Linear(3 * d_model, 2 * head_dim, bias=False)
            _safe_orthogonal_(self.lex_router_g2.weight)
            _safe_orthogonal_(self.lex_router_g3.weight)

            self.salience_binder = SalienceEntityBinder(
                d_model,
                n_ind_groups=self.n_ent_groups,
                head_dim=head_dim,
                window_size=chunk_size,
                chunk_size=chunk_size,
            )
            self.ind_shared_proj = nn.Parameter(torch.zeros(self.n_ent_groups, head_dim, head_dim))

            # Sub-head 0 stays strictly symmetric (`0`); sub-heads 1..2 explore complementary subspace directions
            self.q_subhead_delta = nn.Linear(
                d_model, self.n_ind_groups * (self.heads_per_group - 1) * head_dim, bias=False
            )
            nn.init.normal_(self.q_subhead_delta.weight, mean=0.0, std=0.01)

            # Collapse-proof 33x-dynamic-range novelty-modulated write gate (`[0.0044, 0.1356]` on novel spans, `0.00005` on filler)
            self.ind_beta_proj = nn.Linear(4 * d_model, self.n_ind_groups, bias=False)
            nn.init.normal_(self.ind_beta_proj.weight, mean=0.0, std=0.01)

            # Isotropic `x0_iso` SVD token-copy value & unclamped residual readout pair (`init_v2_2_from_pretrained_backbone`)
            self.v_ind_proj = nn.Linear(d_model, self.n_ind_groups * head_dim, bias=False)
            self.ind_out_proj = nn.Linear(self.n_ind_groups * head_dim, d_model, bias=False)
            _safe_orthogonal_(self.v_ind_proj.weight)
            _safe_orthogonal_(self.ind_out_proj.weight)

            self.ind_read_proj = nn.Linear(3 * d_model, self.n_ind_groups * self.heads_per_group, bias=True)
            nn.init.normal_(self.ind_read_proj.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.ind_read_proj.bias, -1.80)

        # Causal Multi-Token ShortConv1D on Q, K, V (solves multi-token identifier binding!)
        self.q_conv = CausalShortConv1D(n_q_heads * head_dim, kernel_size=conv_kernel)
        self.k_conv = CausalShortConv1D(self.kv_dim, kernel_size=conv_kernel)
        self.v_conv = CausalShortConv1D(self.kv_dim, kernel_size=conv_kernel)

        # Token-Selective Write/Erase Gate `beta_{b,g,t} in (0, 1)`
        self.beta_proj = nn.Linear(d_model, n_vault_groups, bias=True)
        nn.init.normal_(self.beta_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_proj.bias, -0.15)  # Initial write gate ~0.46, selective per token!

        # Co-Prime Ternary Kenyon Router + BioNPC Homeostatic APL Inhibition
        self.kenyon_proj = TernarySTEProjection(head_dim, self.total_cells, seed=100 + layer_idx)
        self.apl_norm = FusedAPLNorm(self.total_cells)

        anchors = []
        for p in self.prime_banks:
            anchors.extend([1.0 / float(p)] * p)
        self.register_buffer("harmonic_anchors", torch.tensor(anchors, dtype=torch.float32))

        # High-gain PRIME positive kernel temperatures
        self.beta_sem_q = nn.Parameter(torch.full((1, n_q_heads, 1, 1), 1.85))
        self.beta_sem_k = nn.Parameter(torch.full((1, n_vault_groups, 1, 1), 1.85))

        # Multi-Scale Geometric Decay Spectrum + Chunk-Relative RoPE for O(L) Local Syntax (`64 x 128`)
        # Spans half-lives from ~1.7 tokens (sharp token adjacency) to ~86 tokens (clause/paragraph binding!)
        log_one_minus = torch.linspace(math.log(0.34), math.log(0.008), n_q_heads)
        head_decays = 1.0 - torch.exp(log_one_minus)  # [0.66 .. 0.992]
        idx_q = torch.arange(chunk_size, dtype=torch.float32) + chunk_size      # [C .. 2C-1]
        idx_k = torch.arange(2 * chunk_size, dtype=torch.float32)               # [0 .. 2C-1]
        diff = idx_q.unsqueeze(-1) - idx_k.unsqueeze(-2)                        # (C, 2C)
        valid = (diff >= 0).float().unsqueeze(0)                                # (1, C, 2C)
        decay_3d = torch.pow(head_decays.view(n_q_heads, 1, 1), diff.clamp(min=0.0).unsqueeze(0)) * valid
        self.register_buffer("local_decay_mask", decay_3d.view(1, n_q_heads, 1, chunk_size, 2 * chunk_size))
        attn_bias = torch.where(decay_3d > 0.0, torch.log(decay_3d.clamp(min=1e-8)), torch.full_like(decay_3d, -1e9))
        self.register_buffer(
            "_local_attn_bias", attn_bias.view(1, n_q_heads, 1, chunk_size, 2 * chunk_size), persistent=False
        )

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos_2c = torch.arange(2 * chunk_size, dtype=torch.float32)
        freqs = torch.outer(pos_2c, inv_freq)  # (2C, D/2)
        self.register_buffer("rope_cos_k", torch.cos(freqs).view(1, 1, 1, 2 * chunk_size, head_dim // 2))
        self.register_buffer("rope_sin_k", torch.sin(freqs).view(1, 1, 1, 2 * chunk_size, head_dim // 2))
        self.register_buffer("rope_cos_q", torch.cos(freqs[chunk_size:]).view(1, 1, 1, chunk_size, head_dim // 2))
        self.register_buffer("rope_sin_q", torch.sin(freqs[chunk_size:]).view(1, 1, 1, chunk_size, head_dim // 2))

        # Branch mixing gates + GTRM macro-summary + Gated Head Norm + Output projection
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
        if mode or self.training:
            self._eval_w_cache = None
        return super().train(mode)

    def _get_v2_2_packed_weights(self, x: torch.Tensor):
        d_m = self.d_model
        sb = self.salience_binder
        w_si = sb.sal_in.weight
        w_ib = self.ind_beta_proj.weight
        w_lg2 = self.lex_router_g2.weight
        w_lg3 = self.lex_router_g3.weight
        w_ir = self.ind_read_proj.weight

        if not self.training and not torch.is_grad_enabled():
            comp_dt = torch.get_autocast_dtype("cuda") if (x.is_cuda and torch.is_autocast_enabled("cuda")) else x.dtype
            key = (x.device, comp_dt, self.q_proj.weight.data_ptr(), self.q_proj.weight._version, self.v_ind_proj.weight._version)
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
                qkv_w = torch.cat([self.q_conv.conv.weight, self.k_conv.conv.weight, self.v_conv.conv.weight], dim=0).contiguous()
                qkv_b = torch.cat([self.q_conv.conv.bias, self.k_conv.conv.bias, self.v_conv.conv.bias], dim=0).contiguous()
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
        qkv_w = torch.cat([self.q_conv.conv.weight, self.k_conv.conv.weight, self.v_conv.conv.bias * 0 + self.v_conv.conv.weight], dim=0) if False else torch.cat([self.q_conv.conv.weight, self.k_conv.conv.weight, self.v_conv.conv.weight], dim=0)
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
        """
        Routes query heads (`cell_q`: `(B, H_q, R, L)`) and vault groups (`cell_k`: `(B, G, R, L)`)
        into Co-Prime cells `[0, total_cells)` via a single batched Top-1 `argmax` pass.
        In inference (`not torch.is_grad_enabled()`), skips `F.softmax` + `.gather` since `ste == 1.0`.
        """
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
        if self.v2_1_mode:
            kc_qk = F.relu(self.apl_norm(self.kenyon_proj(qk_unit)).float()) * inv_tau
        else:
            anchors = self.harmonic_anchors.to(orig_dtype)
            kc_qk = (F.relu(self.apl_norm(self.kenyon_proj(qk_unit))) * anchors) * inv_tau

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

        cell_qk = torch.stack(cqk_list, dim=2)  # (B, H_q + G, R, L)
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
        """
        Strictly O(L) `64 x 128` sliding block local attention with chunk-invariant RoPE
        and multi-scale geometric head decay (`gamma_h in [0.66, 0.992]`), evaluated at
        grouped (`G=4`) `k_win`/`v_win` resolution (`192 x 64 @ 64 x 128` packed WMMA tiles).
        """
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
        )  # (B, G, M, 2C, D)
        v_win = v_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)  # (B, G, M, 2C, D)

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

        if self.v2_2_mode:
            if x0_norm is None:
                x0_norm = x.detach()
            n_keep_g = self.n_keep_g
            n_keep_h = n_keep_g * self.heads_per_group
            G = self.n_vault_groups
            D = self.head_dim
            q_dim = self.n_q_heads * D
            kv_dim = self.kv_dim

            # 1. Fused single-GEMM `x` projection + single-launch `Q/K/V` CausalShortConv1D
            w_x_all, qkv_w, qkv_b, w_iso_all, w_x0n_all, w_ib_rms = self._get_v2_2_packed_weights(x)
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

            if isinstance(cms_aux, dict) and cms_aux.get("shape") == (B, L):
                sh = cms_aux
            else:
                sh = compute_v2_2_shared_aux(x0_norm, self.salience_binder)

            x0_iso = sh["x0_iso"]
            nov4 = sh["nov4"]
            inv_counts_f = sh["inv_counts_f"]

            # 2. Fused single-GEMM `x0_iso` projection across SalienceEntityBinder + Induction Q/V/Beta
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

            # Voronoi-Decoupled Entity Subspaces (`ent_u`, `ent_r`) + Batched Shared-Projection & RFFT Binding
            ent_u, ent_r = sb(x_rms, x0_norm, cms_aux=sh, pre_x0_iso_sal=pre_sal)
            del pre_sal
            m_shared = (0.20 / 8.0) * torch.tanh(self.ind_shared_proj.float())

            ent_ur = torch.stack([ent_u, ent_r], dim=3)  # (B, L, 2, 2, D)
            del ent_u, ent_r
            eur_zm = ent_ur - ent_ur.mean(dim=-1, keepdim=True)
            del ent_ur
            ur_ent = eur_zm + torch.einsum("blgud,gde->blgue", eur_zm, m_shared)
            del eur_zm
            ur_ent_rms = ur_ent * torch.rsqrt(ur_ent.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            ur_comb = ur_ent_rms[:, :, 0:1, :, :] + ur_ent_rms[:, :, 1:2, :, :]
            ur_ent_3 = torch.cat(
                [ur_ent_rms, ur_comb * torch.rsqrt(ur_comb.pow(2).mean(dim=-1, keepdim=True) + 1e-6)], dim=2
            )  # (B, L, 3, 2, D)
            del ur_ent_rms, ur_comb

            # Fused single-GEMM `x0_norm` projection across Lexical Routers + Readout Gate (shift low-dim outputs!)
            w_ir = self.ind_read_proj.weight
            lg2_c = self.lex_router_g2.weight.size(0)  # 256
            lg3_c = self.lex_router_g3.weight.size(0)  # 128
            ir_c = w_ir.size(0)    # 9
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

            lex_both = torch.cat([lex_12, lex_3], dim=2)  # (B, L, 3, 2, D)
            del lex_12, lex_3
            lex_zm = lex_both - lex_both.mean(dim=-1, keepdim=True)
            del lex_both
            lex_rms = lex_zm * torch.rsqrt(lex_zm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            del lex_zm

            # Single batched RFFT Circular-Convolution for both Level-2 `u_ind` and Level-1 `r_ind`!
            ur_ind_raw = fft_circular_bind(ur_ent_3, lex_rms)  # (B, L, 3, 2, D)
            del ur_ent_3, lex_rms
            u_ind_raw = ur_ind_raw[:, :, :, 0, :]
            r_ind_raw = ur_ind_raw[:, :, :, 1, :]
            u_ind = u_ind_raw * torch.rsqrt(u_ind_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            r_all = F.normalize(r_ind_raw, p=2, dim=-1, eps=1e-6)
            del ur_ind_raw, u_ind_raw, r_ind_raw
            r_all_prev = F.pad(r_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0)) if L > 1 else torch.zeros_like(r_all)

            # Route unique Q streams + 4 K groups in 1 pass; wider Co-Prime banks (`v2.3`) share Group-0 cell Q!
            shared_group_q = self.prime_banks != (17, 19, 23, 29)
            lex_q_3 = r_all.transpose(1, 2)
            lex_k_3 = r_all_prev.transpose(1, 2)
            kr_f = torch.cat([ks_f[:, :n_keep_g], lex_k_3], dim=1)  # (B, 4, L, D)
            inv_sqrt_d = 1.0 / math.sqrt(D)
            kr_unit = (kr_f * (torch.rsqrt(kr_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)

            if shared_group_q:
                qr_4_f = torch.cat([qs_f[:, :n_keep_h].mean(dim=1, keepdim=True), lex_q_3], dim=1)  # (B, 4, L, D)
                qr_unit = (qr_4_f * (torch.rsqrt(qr_4_f.pow(2).mean(-1, keepdim=True) + 1e-6) * inv_sqrt_d)).to(dtype)
                del qr_4_f
                cq_4, cell_k, ste_q_4, ste_k = self._route_cells_v2(qr_unit, kr_unit, step_ratio)
                cell_q = cq_4.repeat_interleave(self.heads_per_group, dim=1)
                ste_q = ste_q_4.repeat_interleave(self.heads_per_group, dim=1) if ste_q_4 is not None else None
                del cq_4, ste_q_4
            else:
                qr_6_f = torch.cat([qs_f[:, :n_keep_h], lex_q_3], dim=1)  # (B, 6, L, D)
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
                b_blk = beta_ind_raw_f[:, :2, :l_tr].detach().view(B, 2, m_cnt, 64)
                w_blk = F.softmax(torch.where(tok_val, 25.0 * b_blk, torch.full_like(b_blk, -1e4)), dim=-1)
                u_prev_2 = F.pad(u_ent[:, :-1, :, :], (0, 0, 0, 0, 1, 0))[:, :l_tr, :, :]
                u_blk = u_prev_2.permute(0, 2, 1, 3).reshape(B, 2, m_cnt, 64, D)
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
            del phi_q, phi_k, v_vault, beta_write, cell_q, cell_k
        else:
            anchor = self.qk_group_anchor(x)
            anchor_q = anchor.view(B, L, self.n_vault_groups, 1, self.head_dim).expand(
                B, L, self.n_vault_groups, self.heads_per_group, self.head_dim
            ).reshape(B, L, self.n_q_heads * self.head_dim)
            if self.v2_1_mode:
                q_bind, k_bind = self.span_binder(x, anchor)
                q = self.q_conv(anchor_q + self.q_proj(x) + q_bind).view(B, L, self.n_q_heads, self.head_dim).transpose(1, 2)
                k = self.k_conv(anchor + self.k_delta(x) + k_bind).view(B, L, self.n_vault_groups, self.head_dim).transpose(1, 2)
            else:
                q = self.q_conv(anchor_q + self.q_proj(x)).view(B, L, self.n_q_heads, self.head_dim).transpose(1, 2)
                k = self.k_conv(anchor + self.k_delta(x)).view(B, L, self.n_vault_groups, self.head_dim).transpose(1, 2)

            v = self.v_conv(self.v_proj(x)).view(B, L, self.n_vault_groups, self.head_dim).transpose(1, 2)
            dtype = v.dtype

            q_f, k_f = q.float(), k.float()
            q_rms = torch.rsqrt(torch.mean(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
            k_rms = torch.rsqrt(torch.mean(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
            q_norm = (q_f * q_rms).to(dtype)
            k_norm = (k_f * k_rms).to(dtype)
            q_unit = (q_f * (q_rms / math.sqrt(self.head_dim))).to(dtype)
            k_unit = (k_f * (k_rms / math.sqrt(self.head_dim))).to(dtype)

            cell_q, cell_k, ste_q, ste_k = self._route_cells_v2(q_unit, k_unit, step_ratio)
            bq = torch.clamp(self.beta_sem_q.float(), 1.2, 3.5)
            bk = torch.clamp(self.beta_sem_k.float(), 1.2, 3.5)

            if self.v2_1_mode:
                phi_q = F.normalize(torch.exp(bq * torch.clamp(q_f * q_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype)
                phi_k = F.normalize(torch.exp(bk * torch.clamp(k_f * k_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype)
                v_vault = v * ste_k if ste_k is not None else v
                beta_write_f = 0.96 * torch.sigmoid(self.beta_proj(x).float()).transpose(1, 2)
                beta_aux_mean = beta_write_f.mean()
                beta_write = beta_write_f.to(dtype)
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
                    prime_banks=self.prime_banks,
                )
            else:
                phi_q = F.normalize(torch.exp(bq * torch.clamp(q_f * q_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype)
                phi_k = F.normalize(torch.exp(bk * torch.clamp(k_f * k_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype)
                if ste_q is not None:
                    phi_q = phi_q * ste_q
                if ste_k is not None:
                    phi_k = phi_k * ste_k
                beta_write_f = 0.02 + 0.94 * torch.sigmoid(self.beta_proj(x).float()).transpose(1, 2)
                beta_aux_mean = beta_write_f.mean()
                beta_write = beta_write_f.to(dtype)
                y_vault = flash_vault_v2_chunkwise(
                    phi_q, phi_k, v, beta_write, cell_q, cell_k, chunk_size=self.chunk_size, prime_banks=self.prime_banks
                )
            inv_counts_f = (1.0 / torch.arange(1, L + 1, device=x.device, dtype=torch.float32)).view(1, 1, L, 1)

        # 3. O(L) Chunk-Local RoPE + Multi-Scale Decay Syntax Branch
        y_syntax = self._chunk_local_rope_syntax(q_norm, k_norm, v)

        # 4. O(L) GTRM Macro-Horizon Summary Branch (2-level block scan avoids sequential ROCm scan at 32K)
        if self.v2_1_mode:
            v_mean = (_chunked_cumsum_dim2(v.float()) * inv_counts_f).to(dtype)
        else:
            v_mean = (_chunked_cumsum_dim2(v.float()) * inv_counts_f).to(dtype)
        v_macro = self.gtrm_proj(v_mean).repeat_interleave(self.heads_per_group, dim=1)

        if self.v2_2_mode:
            n_keep_h = self.n_keep_g * self.heads_per_group  # 3 heads (Group 0)
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

            # Unclamped SVD token-copy residual readout bypassing `yb_rms` radial clamp
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
        else:
            y = (
                self.vault_gate.to(dtype) * y_vault
                + self.syntax_gate.to(dtype) * y_syntax
                + self.gtrm_gate.to(dtype) * v_macro
            )
            if self.v2_1_mode and ste_q is not None:
                y_norm = (self.head_norm(y) * ste_q).transpose(1, 2).reshape(
                    B, L, self.n_q_heads * self.head_dim
                )
            else:
                y_norm = self.head_norm(y).transpose(1, 2).reshape(B, L, self.n_q_heads * self.head_dim)
            gate = torch.sigmoid(self.out_gate(x)).to(y_norm.dtype)
            out = self.o_proj(y_norm * gate)
        if return_beta_mean and return_ent_div:
            return out, beta_aux_mean, ent_div_aux
        if return_beta_mean:
            return out, beta_aux_mean
        return out


class VAULTv2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_vault_groups: int,
        head_dim: int,
        d_ff: int,
        chunk_size: int,
        layer_idx: int,
        v2_1_mode: bool = False,
        ckpt_interval: int = 1,
        v2_2_mode: bool = False,
        prime_banks: Tuple[int, ...] = (17, 19, 23, 29),
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = VAULTv2Attention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_vault_groups=n_vault_groups,
            head_dim=head_dim,
            prime_banks=prime_banks,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            v2_1_mode=v2_1_mode,
            ckpt_interval=ckpt_interval,
            v2_2_mode=v2_2_mode,
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


class VAULTLMv2(nn.Module):
    """
    VAULTLM-v2 / v2.1 / v2.2 / v2.3 (~151.3M - 180.5M Parameters):
      - vocab_size     = 49,152 (SmolLM BPE with native `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<file_sep>`)
      - d_model        = 768
      - n_layers       = 16
      - n_q_heads      = 12, n_vault_groups = 4 (`head_dim = 64`, True 3:1 Grouped-Vault Attention)
      - d_ff           = 2048 (`SwiGLU`)
      - v2.2 / v2.3: Voronoi-Decoupled Two-Level Associative Induction (`SalienceEntityBinder` + Windowed Co-Prime CMS
        + RFFT Circular-Convolution Binding + Unclamped Isotropic SVD Token-Copy Readout);
        `v2.3` expands Co-Prime Voronoi banks `(17,19,23,29)` -> `(47,53,59,61)` (`88 -> 220` cells, `8.97M` CRT addresses).
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
        grad_checkpoint: bool = False,
        v2_1_mode: bool = False,
        ckpt_interval: Optional[int] = None,
        v2_2_mode: bool = False,
        prime_banks: Tuple[int, ...] = (17, 19, 23, 29),
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
        self.v2_2_mode = v2_2_mode
        self.v2_1_mode = v2_1_mode or v2_2_mode
        self.prime_banks = tuple(int(p) for p in prime_banks)
        self.ckpt_interval = ckpt_interval if ckpt_interval is not None else (8 if self.v2_1_mode else 1)

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        if v2_2_mode:
            self.x0_rms_norm = RMSNorm(d_model)

        self.layers = nn.ModuleList([
            VAULTv2Block(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_vault_groups=n_vault_groups,
                head_dim=head_dim,
                d_ff=d_ff,
                chunk_size=chunk_size,
                layer_idx=i,
                v2_1_mode=v2_1_mode,
                ckpt_interval=self.ckpt_interval,
                v2_2_mode=v2_2_mode,
                prime_banks=self.prime_banks,
            )
            for i in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # GPT-2 / Llama residual depth scaling `1 / sqrt(2 * n_layers)` on residual output projections
        res_std = 0.02 / math.sqrt(2.0 * n_layers)
        for blk in self.layers:
            nn.init.normal_(blk.attn.o_proj.weight, mean=0.0, std=res_std)
            nn.init.normal_(blk.mlp.w3.weight, mean=0.0, std=res_std)

        if v2_2_mode:
            self.init_v2_2_from_pretrained_backbone()

            def _v2_2_post_load_hook(module: "VAULTLMv2", incompatible_keys):
                if module.v2_2_mode and any("v_ind_proj.weight" in k for k in incompatible_keys.missing_keys):
                    module.init_v2_2_from_pretrained_backbone()

            self.register_load_state_dict_post_hook(_v2_2_post_load_hook)

    @torch.no_grad()
    def init_v2_2_from_pretrained_backbone(self) -> None:
        """
        Initializes `x0_cone_mean` and complementary 192-dim principal SVD slices `(v_ind_proj, ind_out_proj)`
        from the centered isotropic token-embedding spectrum (`e_iso = normalize(x0_rms_norm(E) - mu)`).
        Because `lm_head.weight` is tied to `embed_tokens.weight`, complementary layer slices tile all `768`
        directions (`16 * 48 = 768`), yielding high-margin open-vocabulary token copy from Step 0 when forking
        from Stage-1 `vault_v2_pretrain_best.pt` while leaving loaded `v2_2` checkpoints untouched!
        """
        ew_rms = self.x0_rms_norm(self.embed_tokens.weight.detach().float())
        cone_mu = ew_rms.mean(dim=0, keepdim=True)  # (1, d_model)
        e_iso = F.normalize(ew_rms - cone_mu, p=2, dim=-1, eps=1e-6)
        _, _, vh = torch.linalg.svd(e_iso, full_matrices=False)  # (d_model, d_model)
        vh_rep = torch.cat([vh, vh], dim=0)
        cone_mu_3d = cone_mu.unsqueeze(0)  # (1, 1, d_model)

        stride = max(1, self.d_model // max(1, len(self.layers)))
        for li, blk in enumerate(self.layers):
            attn = blk.attn
            if not attn.v2_2_mode:
                continue
            attn.salience_binder.x0_cone_mean.copy_(cone_mu_3d)
            ind_w = attn.n_ind_groups * attn.head_dim
            st = (li * stride) % self.d_model
            w_slice = vh_rep[st : st + ind_w, :]
            attn.v_ind_proj.weight.copy_(w_slice.to(attn.v_ind_proj.weight.dtype))
            attn.ind_out_proj.weight.copy_((2.80 * w_slice.T).to(attn.ind_out_proj.weight.dtype))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compute_router_aux_loss(self) -> torch.Tensor:
        """
        Lightweight O(1)-in-sequence-length Co-Prime Kenyon Router independence & anti-collapse regularizer.
        Encourages the 4 Co-Prime banks to hash orthogonal subspaces while preventing intra-bank cell collapse.
        """
        loss = torch.zeros((), device=self.embed_tokens.weight.device, dtype=torch.float32)
        for blk in self.layers:
            attn = blk.attn
            w = F.normalize(attn.kenyon_proj.weight.float(), p=2, dim=-1)
            gram = torch.matmul(w, w.transpose(0, 1))
            slices = []
            off = 0
            for p in attn.prime_banks:
                slices.append((off, off + p))
                sub = gram[off : off + p, off : off + p]
                eye = torch.eye(p, device=w.device, dtype=torch.float32)
                loss = loss + ((sub - eye) ** 2).mean()
                off += p
            for i in range(len(slices)):
                for j in range(i + 1, len(slices)):
                    s_i0, s_i1 = slices[i]
                    s_j0, s_j1 = slices[j]
                    cross = gram[s_i0:s_i1, s_j0:s_j1]
                    loss = loss + (cross ** 2).mean()
        return loss / float(len(self.layers))

    def compute_x0_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.x0_rms_norm(x.detach())

    def compute_cms_aux(self, x0_norm: torch.Tensor):
        b0 = self.layers[0].attn.salience_binder
        return compute_v2_2_shared_aux(x0_norm, b0)

    def _encode_hidden_with_aux(
        self, input_ids: torch.Tensor, step_ratio: float = 1.0, return_ent_div: bool = False
    ):
        x = self.embed_tokens(input_ids)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        if self.v2_2_mode:
            x0_norm = self.compute_x0_norm(x)
            cms_aux = self.compute_cms_aux(x0_norm)
        else:
            x0_norm = None
            cms_aux = None
        beta_acc = torch.zeros((), device=x.device, dtype=torch.float32)
        ent_div_acc = torch.zeros((), device=x.device, dtype=torch.float32)
        for idx, layer in enumerate(self.layers):
            use_ckpt = False
            if self.training and torch.is_grad_enabled():
                if self.grad_checkpoint in (False, None, "none"):
                    use_ckpt = False
                elif self.grad_checkpoint == "quarter":
                    use_ckpt = (idx % 4 == 0)
                elif self.grad_checkpoint == "half":
                    use_ckpt = (idx % 2 == 0)
                elif self.grad_checkpoint == "three_quarter":
                    use_ckpt = (idx % 4 != 3)
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
        """
        Memory-optimal chunked vocabulary cross-entropy: avoids allocating multi-GB `(B*L, V)` FP32
        logit buffers while computing the exact weighted cross-entropy loss and gradients.
        """
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


@torch.no_grad()
def upgrade_v2_2_to_wider_primes(
    model_v2_3: VAULTLMv2,
    v2_2_state: Dict[str, torch.Tensor],
    old_primes: Tuple[int, ...] = (17, 19, 23, 29),
    split_angle_rad: float = 0.52,
    k_nz: int = 52,
) -> Dict[str, object]:
    """
    Lossless Voronoi Cell Mitosis (`88 -> 220` cells, e.g. `(17,19,23,29)` -> `(47,53,59,61)`):
      - Copies 99.98% of `VAULT v2.2` weights (`embed_tokens`, `qk_group_anchor`, `q/k/v_proj`, `SwiGLU`,
        `salience_binder`, `v_ind_proj`, `ind_out_proj`, `lex_router_g2/g3`, etc.) bit-for-bit.
      - Splits each parent Co-Prime cell `u_c` into `2..3` non-degenerate daughter cones in `u_c^\\perp`
        via zero-sum simplex directions (`cos(theta)*u_c + sin(theta)*v_{c,j}`) and equalizes ternary
        `L1` norm (`||w_tern||_1 == k_nz` across all 220 cells) so zero Voronoi black-hole cells can form.
    """
    new_primes = model_v2_3.prime_banks
    assert len(old_primes) == len(new_primes) == 4
    old_total = sum(old_primes)
    new_total = sum(new_primes)

    dst_sd = model_v2_3.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    copied_exact = 0
    mitosis_layers = 0
    cos_t = math.cos(split_angle_rad)
    sin_t = math.sin(split_angle_rad)

    for k, dst_t in dst_sd.items():
        if k.endswith(".attn.harmonic_anchors"):
            new_sd[k] = dst_t.clone()
            continue
        if k not in v2_2_state:
            raise KeyError(f"Missing expected v2.2 checkpoint key during mitosis: {k}")
        src_t = v2_2_state[k]
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
                    # Break exact ties deterministically if any
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

        w_new = torch.stack(rows_out, dim=0)  # (new_total, D)
        w_new = w_new * (old_mean_abs / w_new.abs().mean().clamp(min=1e-6))
        thresh_new = 0.25 * w_new.abs().mean()
        nz_counts = (w_new.abs() > thresh_new).sum(dim=-1)
        assert int(nz_counts.min().item()) == k_nz and int(nz_counts.max().item()) == k_nz, (
            f"Expected uniform k_nz={k_nz}, got [{nz_counts.min().item()}, {nz_counts.max().item()}]"
        )
        new_sd[k] = w_new.to(device=dst_t.device, dtype=dst_t.dtype)
        mitosis_layers += 1

    model_v2_3.load_state_dict(new_sd, strict=True)
    return {
        "copied_exact_tensors": copied_exact,
        "mitosis_kenyon_layers": mitosis_layers,
        "old_primes": old_primes,
        "new_primes": new_primes,
        "old_cells": old_total,
        "new_cells": new_total,
        "crt_capacity": new_primes[0] * new_primes[1] * new_primes[2] * new_primes[3],
    }
