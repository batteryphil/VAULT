# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# FlashVAULTAttention: Strictly O(L) Compute / O(1) State Memory Multi-Head VAULT Layer
# Weight-compatible with VAULTAttention150M (`src/vault_lm_150m.py`), replacing all O(L^2)
# dense matrices with:
#   1. FlashVAULT Chunkwise Compact WY Triton 3.8 Kernel (88 Co-Prime Fast-Weight Cells)
#   2. Exact Chunk-Local Syntactic Flow Window (0.92^128 = 2.3e-5 -> exact to BF16 precision in O(L))
#   3. O(L) Dyadic GTRM Prefix Cascade + Straight-Through Router Gradient Coupling

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vault_lm_150m import FusedAPLNorm, TernarySTEProjection
from src.kernels.flash_vault_triton import flash_vault_chunkwise


class FlashVAULTAttention(nn.Module):
    """
    Drop-in O(L) Chunkwise Training & Long-Context Module for VAULTLM.
    Supports both per-head routing (n_vault_groups = n_heads) and
    Grouped-Vault Attention (GVA, e.g. n_vault_groups = 4).
    """
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        head_dim: int = 64,
        n_vault_groups: int = 4,
        prime_banks: Tuple[int, ...] = (17, 19, 23, 29),
        chunk_size: int = 64,
        layer_idx: int = 0,
    ):
        super().__init__()
        assert n_heads % n_vault_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_vault_groups = n_vault_groups
        self.heads_per_group = n_heads // n_vault_groups
        self.prime_banks = prime_banks
        self.num_banks = len(prime_banks)
        self.total_cells = sum(prime_banks)
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        self.qk_shared = nn.Linear(d_model, d_model, bias=False)
        self.q_delta = nn.Linear(d_model, d_model, bias=False)
        self.k_delta = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.q_delta.weight, std=0.02)
        nn.init.normal_(self.k_delta.weight, std=0.02)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.kenyon_proj = TernarySTEProjection(head_dim, self.total_cells, seed=42 + layer_idx)
        self.apl_norm = FusedAPLNorm(self.total_cells)

        anchors = []
        offsets = []
        cur = 0
        for p in prime_banks:
            anchors.extend([1.0 / float(p)] * p)
            offsets.append(cur)
            cur += p
        self.register_buffer("harmonic_anchors", torch.tensor(anchors, dtype=torch.float32))
        self.register_buffer("bank_offsets", torch.tensor(offsets, dtype=torch.int64))

        self.beta_sem = nn.Parameter(torch.full((1, n_heads, 1, 1), 1.8))
        self.beta_syn = nn.Parameter(torch.full((1, n_heads, 1, 1), 2.0))
        self.write_beta_logit = nn.Parameter(torch.full((1, n_heads, 1), 0.62))  # sigmoid(0.62) ~= 0.65

        self.vault_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.45))
        self.syntax_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.55))
        self.gtrm_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.10))
        self.gtrm_proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.normal_(self.gtrm_proj.weight, std=0.02)

        # Precompute 2-chunk local causal decay mask (C x 2C) for O(L) Syntactic Branch
        idx_q = torch.arange(chunk_size, dtype=torch.float32) + chunk_size      # [C .. 2C-1]
        idx_k = torch.arange(2 * chunk_size, dtype=torch.float32)               # [0 .. 2C-1]
        diff = idx_q.unsqueeze(-1) - idx_k.unsqueeze(-2)
        local_decay = torch.where(diff >= 0, torch.pow(0.92, diff.clamp(min=0.0)), torch.zeros_like(diff))
        self.register_buffer("local_decay_mask", local_decay.view(1, 1, 1, chunk_size, 2 * chunk_size))

    def _route_global_cells(
        self, q_unit: torch.Tensor, k_unit: torch.Tensor, step_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routes queries and keys to global Co-Prime cell IDs in [0, 88) across `n_vault_groups`
        and returns unit-forward STE router gates so `kenyon_proj` & `apl_norm` receive gradients.
        """
        B, H, L, D = q_unit.shape
        G = self.n_vault_groups
        orig_dtype = q_unit.dtype
        inv_tau = 1.0 / max(0.18, 0.5 * (1.0 - 0.64 * step_ratio))

        if G < H:
            q_grp = q_unit.view(B, G, self.heads_per_group, L, D).mean(dim=2)
            k_grp = k_unit.view(B, G, self.heads_per_group, L, D).mean(dim=2)
        else:
            q_grp, k_grp = q_unit, k_unit

        anchors = self.harmonic_anchors.to(orig_dtype)
        kc_q = (F.relu(self.apl_norm(self.kenyon_proj(q_grp))) * anchors) * inv_tau
        kc_k = (F.relu(self.apl_norm(self.kenyon_proj(k_grp))) * anchors) * inv_tau

        cq_list, ck_list = [], []
        gq_list, gk_list = [], []
        offset = 0
        for r, p in enumerate(self.prime_banks):
            qs = kc_q[..., offset : offset + p]
            ks = kc_k[..., offset : offset + p]
            pq_r = F.softmax(qs, dim=-1)
            pk_r = F.softmax(ks, dim=-1)

            iq = torch.argmax(qs, dim=-1)  # (B, G, L)
            ik = torch.argmax(ks, dim=-1)  # (B, G, L)
            gq_list.append(pq_r.gather(-1, iq.unsqueeze(-1)).squeeze(-1))
            gk_list.append(pk_r.gather(-1, ik.unsqueeze(-1)).squeeze(-1))

            cq_list.append(iq + offset)
            ck_list.append(ik + offset)
            offset += p

        cell_q = torch.stack(cq_list, dim=2)  # (B, G, R, L)
        cell_k = torch.stack(ck_list, dim=2)  # (B, G, R, L)

        # Mean bank confidence -> Straight-Through Estimator gate (identically 1.0 in forward!)
        gq_mean = torch.stack(gq_list, dim=2).mean(dim=2)  # (B, G, L)
        gk_mean = torch.stack(gk_list, dim=2).mean(dim=2)  # (B, G, L)
        if G < H:
            gq_mean = gq_mean.repeat_interleave(self.heads_per_group, dim=1)
            gk_mean = gk_mean.repeat_interleave(self.heads_per_group, dim=1)

        ste_q = (1.0 + (gq_mean - gq_mean.detach())).unsqueeze(-1).to(orig_dtype)
        ste_k = (1.0 + (gk_mean - gk_mean.detach())).unsqueeze(-1).to(orig_dtype)
        return cell_q, cell_k, ste_q, ste_k

    def _chunk_local_syntax(self, psi_q: torch.Tensor, psi_k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Strictly O(L) Syntactic Local Flow via 2-chunk sliding causal blocks (64 x 128).
        Since 0.92^64 = 0.0048 and 0.92^128 = 2.3e-5, this matches full O(L^2) decay to BF16 precision!
        """
        B, H, L_orig, D = v.shape
        C = self.chunk_size
        rem = L_orig % C
        pad_l = (C - rem) if rem != 0 else 0
        if pad_l > 0:
            psi_q = F.pad(psi_q, (0, 0, 0, pad_l))
            psi_k = F.pad(psi_k, (0, 0, 0, pad_l))
            v = F.pad(v, (0, 0, 0, pad_l))

        L = psi_q.size(2)
        M = L // C

        q_blk = psi_q.view(B, H, M, C, D)
        # Prepend one zero chunk so chunk m attends to [chunk m-1, chunk m] -> (B, H, M, 2C, D)
        k_pad = F.pad(psi_k, (0, 0, C, 0))
        v_pad = F.pad(v, (0, 0, C, 0))
        k_win = k_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)  # (B, H, M, 2C, D)
        v_win = v_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)

        scores = (torch.matmul(q_blk, k_win.transpose(-1, -2)) ** 2) * self.local_decay_mask.to(v.dtype)
        out_blk = torch.matmul(scores, v_win) / scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        return out_blk.reshape(B, H, L, D)[:, :, :L_orig, :]

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, D = x.shape

        base_qk = self.qk_shared(x)
        q = (base_qk + self.q_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k_raw = (base_qk + self.k_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        dtype = v.dtype

        k_prev = F.pad(k_raw[:, :, :-1, :], (0, 0, 1, 0))
        k = torch.cat([k_raw[:, : self.n_heads // 2], k_prev[:, self.n_heads // 2 :]], dim=1)

        q_f, k_f = q.float(), k.float()
        q_rms_inv = torch.rsqrt(torch.mean(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
        k_rms_inv = torch.rsqrt(torch.mean(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
        q_unit = (q_f * (q_rms_inv / math.sqrt(self.head_dim))).to(dtype)
        k_unit = (k_f * (k_rms_inv / math.sqrt(self.head_dim))).to(dtype)
        q_clamped = torch.clamp(q_f * q_rms_inv, -3.0, 3.0)
        k_clamped = torch.clamp(k_f * k_rms_inv, -3.0, 3.0)

        # 1. O(L) FlashVAULT Chunkwise Compact WY Co-Prime Branch
        cell_q, cell_k, ste_q, ste_k = self._route_global_cells(q_unit, k_unit, step_ratio)
        b_sem = torch.clamp(self.beta_sem.float(), 1.5, 3.5)
        phi_q = F.normalize(torch.exp(b_sem * q_clamped), p=2, dim=-1).to(dtype) * ste_q
        phi_k = F.normalize(torch.exp(b_sem * k_clamped), p=2, dim=-1).to(dtype) * ste_k
        beta_write = torch.sigmoid(self.write_beta_logit).expand(B, self.n_heads, L).to(dtype)

        y_vault = flash_vault_chunkwise(phi_q, phi_k, v, beta_write, cell_q, cell_k, chunk_size=self.chunk_size)

        # 2. O(L) Chunk-Local Syntactic Decay Branch
        b_syn = torch.clamp(self.beta_syn.float(), 1.5, 3.5)
        psi_q = F.normalize(torch.exp(b_syn * q_clamped), p=2, dim=-1).to(dtype)
        psi_k = F.normalize(torch.exp(b_syn * k_clamped), p=2, dim=-1).to(dtype)
        y_syntax = self._chunk_local_syntax(psi_q, psi_k, v)

        # 3. O(L) Dyadic GTRM Macro-Cascade Branch
        inv_counts = (1.0 / torch.arange(1, L + 1, device=x.device, dtype=torch.float32)).view(1, 1, L, 1).to(dtype)
        y_gtrm = self.gtrm_proj(torch.cumsum(v, dim=2) * inv_counts)

        y = self.vault_gate.to(dtype) * y_vault + self.syntax_gate.to(dtype) * y_syntax + self.gtrm_gate.to(dtype) * y_gtrm
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, L, D))
