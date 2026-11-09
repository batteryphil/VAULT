# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Archived Co-Prime Kenyon Cell Associative Memory (PyTorch reference).
#
# Preserved standalone module from VAULT v2.5 / VAULT v3 Stage-A for future repurposing:
#   - Fly-olfaction (Drosophila Mushroom Body) sparse ternary random projection LSH
#   - Chinese Remainder Theorem (CRT) co-prime bank routing (e.g., P = (13, 17, 19) -> 4,199 triplets)
#   - Chunkwise delta-rule associative write + homogeneous cell rescaling (|z| <= 65,536 for infinite streams)
#   - Bank-averaged normalized associative read
#
# Candidate future repurposing targets in VAULT:
#   1. Semantic / cross-lingual co-prime router for k_seg (routing mid-layer hidden states to segment bitmasks)
#   2. Mushroom-Body Output Neuron (MBON) long-horizon familiarity / semantic deduplication sketch
#   3. Final-layer tied-embedding (lm_head == embed_tokens) O(1) kNN logit cache
#   4. O(1) CRT self-speculative CPU draft continuation table

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VAULT_Z_RESCALE_MAX = 65536.0
FAST_VAULT_MATMUL = False  # Set True in trainers for BF16 tensor-core matmuls while keeping state in FP32


def make_kenyon_route(
    dim: int,
    primes: Tuple[int, ...] = (13, 17, 19),
    seed: int = 20261105,
    density: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Fixed random ternary projection (dim, sum(primes)) in {-1, 0, +1}: fly-olfaction style LSH."""
    g = torch.Generator().manual_seed(int(seed))
    n_cells = int(sum(primes))
    u = torch.rand((dim, n_cells), generator=g)
    half = density / 2.0
    route = torch.zeros((dim, n_cells))
    route[u < half] = 1.0
    route[(u >= half) & (u < density)] = -1.0
    for c in range(n_cells):  # guarantee every column has at least one non-zero entry
        if not route[:, c].any():
            route[int(torch.randint(0, dim, (1,), generator=g)), c] = 1.0
    return route


def unit_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def kenyon_route_cells(x: torch.Tensor, route: torch.Tensor, primes: Tuple[int, ...]) -> torch.Tensor:
    """x: (..., D) -> (..., n_banks) global cell index (argmax within each co-prime bank)."""
    logits = x @ route
    idx = []
    off = 0
    for p in primes:
        idx.append(logits[..., off : off + p].argmax(dim=-1) + off)
        off += p
    return torch.stack(idx, dim=-1)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if FAST_VAULT_MATMUL and a.is_cuda:
        return (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float()
    return a @ b


def kenyon_write(
    M: torch.Tensor,
    z: torch.Tensor,
    kt: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    route: torch.Tensor,
    primes: Tuple[int, ...],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Delta-rule write of one evicted chunk (all tokens evaluate residuals against pre-chunk state).
    Includes homogeneous cell rescaling when z_max > 65536 so FP32 accumulators never saturate on >100M-token streams.
    """
    B, G, C, D, _ = M.shape
    T = kt.shape[2]
    NB = len(primes)
    phk = elu_plus_one(kt)
    cells = kenyon_route_cells(kt, route, primes)                         # (B, G, T, NB)
    m_flat = M.permute(0, 1, 3, 2, 4).reshape(B, G, D, C * D)             # [d, c*D + e] = M_c[d, e]
    num_all = _mm(phk, m_flat).view(B, G, T, C, D)
    den_all = torch.einsum("bgtd,bgcd->bgtc", phk, z)
    num = num_all.gather(3, cells[..., None].expand(B, G, T, NB, D))
    den = den_all.gather(3, cells)
    ok = den > eps
    retr = torch.where(ok[..., None], num / den.clamp_min(eps)[..., None], torch.zeros_like(num))
    delta = beta[..., None, None] * (v[..., None, :] - retr)              # (B, G, T, NB, D)
    dense = torch.zeros((B, G, T, C, D), device=M.device, dtype=M.dtype).scatter(
        3, cells[..., None].expand(B, G, T, NB, D), delta
    )
    upd = _mm(phk.transpose(-1, -2), dense.view(B, G, T, C * D))          # (B, G, D, C*D)
    M_new = M + upd.view(B, G, D, C, D).permute(0, 1, 3, 2, 4)
    w_cells = torch.zeros((B, G, T, C), device=M.device, dtype=M.dtype).scatter(
        3, cells, beta[..., None].expand(B, G, T, NB)
    )
    z_new = z + torch.einsum("bgtc,bgtd->bgcd", w_cells, phk)
    if not torch.is_grad_enabled():
        big = z_new.amax(dim=-1) > VAULT_Z_RESCALE_MAX                    # (B, G, C)
        if big.any():
            scale = torch.where(big, 0.5, 1.0).to(M_new.dtype)
            M_new = M_new * scale[..., None, None]
            z_new = z_new * scale[..., None]
    return M_new, z_new


def kenyon_read(
    M: torch.Tensor,
    z: torch.Tensor,
    qt: torch.Tensor,
    route: torch.Tensor,
    primes: Tuple[int, ...],
    eps: float = 1e-6,
) -> torch.Tensor:
    """qt: (B, G, R, T, D) -> (B, G, R, T, D): mean over co-prime banks of normalized cell reads."""
    B, G, C, D, _ = M.shape
    R, T = qt.shape[2], qt.shape[3]
    phq = elu_plus_one(qt)
    cells = kenyon_route_cells(qt, route, primes)                         # (B, G, R, T, NB)
    m_flat = M.permute(0, 1, 3, 2, 4).reshape(B, G, 1, D, C * D)
    num_all = _mm(phq, m_flat).view(B, G, R, T, C, D)
    den_all = torch.einsum("bgrtd,bgcd->bgrtc", phq, z)
    num = num_all.gather(4, cells[..., None].expand(B, G, R, T, len(primes), D))
    den = den_all.gather(4, cells)
    ok = den > eps
    out = torch.where(ok[..., None], num / den.clamp_min(eps)[..., None], torch.zeros_like(num))
    return out.mean(dim=-2)


class KenyonVault(nn.Module):
    """Per-layer trainable Kenyon associative cell adapter (a_k, a_q, beta, alpha)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.a_k = nn.Parameter(torch.eye(head_dim).repeat(n_kv_heads, 1, 1))
        self.a_q = nn.Parameter(torch.eye(head_dim).repeat(n_kv_heads, 1, 1))
        self.beta = nn.Linear(d_model, n_kv_heads, bias=True)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.alpha = nn.Parameter(torch.zeros(n_heads))

    def keys(self, k: torch.Tensor) -> torch.Tensor:
        """k: (B, G, L, D) pre-RoPE -> adapted float32 keys."""
        return torch.einsum("bgld,gde->bgle", unit_rms(k.float()), self.a_k.float())

    def queries(self, q: torch.Tensor, group: int) -> torch.Tensor:
        """q: (B, H, L, D) pre-RoPE -> (B, G, R, L, D) adapted float32 queries."""
        B, H, L, D = q.shape
        qg = unit_rms(q.float()).view(B, H // group, group, L, D)
        return torch.einsum("bgrld,gde->bgrle", qg, self.a_q.float())

    def write_strength(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d_model) normed layer input -> (B, G, L) float32 in (0, 1)."""
        return torch.sigmoid(F.linear(h.float(), self.beta.weight.float(), self.beta.bias.float())).transpose(1, 2)

    def gate(self) -> torch.Tensor:
        return torch.tanh(self.alpha.float())


def save_kenyon_params(module: nn.Module, path: str, extra: Optional[dict] = None) -> None:
    sd = {k: v.detach().cpu() for k, v in module.state_dict().items() if ".vault." in k or k == "vault_route"}
    payload = {"format": "vault_kenyon_archive", "vault_state": sd}
    if extra:
        payload.update(extra)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
