# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULTLM-150M: 151.2M Parameter Native VAULT Language Model
# Built strictly from Philip Bartsch's owned repositories:
#   - vault/src/vault.py
#   - PRIME-Moment-Attention/src/prime_moment_attention/primelm_50m.py
#   - BioNPC/src/bionpc/model.py
#   - gist-memory/src/gist_memory/core.py

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = torch.mean(x_f * x_f, dim=-1, keepdim=True)
        return (x_f * torch.rsqrt(var + self.eps)).to(orig_dtype) * self.weight


class FusedAPLNorm(nn.Module):
    """Anterior Paired Lateral (APL) Homeostatic Inhibition (from BioNPC)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.decay_logit = nn.Parameter(torch.tensor(2.0))
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = F.relu(self.gain).to(x.dtype)
        activity = x.abs().mean(dim=-1, keepdim=True)
        inh = torch.clamp(activity * gain, 0.0, 0.90)
        return x * (1.0 - inh)


class TernarySTEProjection(nn.Module):
    """1.58-Bit Ternary {-1, 0, +1} Linear Projection (from buildbit-os & synapse-os)."""
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        gen = torch.Generator()
        gen.manual_seed(seed)
        w = torch.randn(out_features, in_features, generator=gen) / math.sqrt(in_features)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        thresh = 0.25 * w.abs().mean()
        w_tern = torch.where(w > thresh, torch.ones_like(w), torch.where(w < -thresh, -torch.ones_like(w), torch.zeros_like(w)))
        w_ste = w + (w_tern - w).detach()
        return F.linear(x, w_ste)


def apply_prime_kernel_bf16(x: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    PRIME-Net discovered high-gain positive kernel exp(beta * RMSNorm(x))
    projected onto the unit L2 hypersphere, safe in bfloat16/float32.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    rms = torch.rsqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + 1e-6)
    x_norm = torch.clamp(x_f * rms, -3.0, 3.0)
    phi = torch.exp(temp.float() * x_norm)
    return F.normalize(phi, p=2, dim=-1).to(orig_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class VAULTAttention150M(nn.Module):
    """
    Multi-Head Native VAULT Attention Block for VAULTLM-150M.
    Unites:
      1. Shared-Metric Q/K Initialization + 50/50 1-Step Induction Head Split
      2. Co-Prime Ternary Router (P = {17, 19, 23, 29}) with Soft-to-Hard Warmup STE
      3. Row-Normalized 2nd-Order Neumann Sherman-Morrison Nullspace Orthogonalization
      4. PRIME-Net Variance-Weighted Consensus Readout
      5. Syntactic Exponential Decay Branch + Dyadic GTRM Macro-Cascade
    """
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        head_dim: int = 64,
        prime_banks: Tuple[int, ...] = (17, 19, 23, 29),
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.prime_banks = prime_banks
        self.num_banks = len(prime_banks)
        self.total_cells = sum(prime_banks)
        self.layer_idx = layer_idx

        # Shared metric anchor so Q and K align in Co-Prime cells at Step 0
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
        for p in prime_banks:
            anchors.extend([1.0 / float(p)] * p)
        self.register_buffer("harmonic_anchors", torch.tensor(anchors, dtype=torch.float32))

        self.beta_sem = nn.Parameter(torch.full((1, n_heads, 1, 1), 1.8))
        self.beta_syn = nn.Parameter(torch.full((1, n_heads, 1, 1), 2.0))

        self.vault_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.45))
        self.syntax_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.55))
        self.gtrm_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.10))
        self.gtrm_proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.normal_(self.gtrm_proj.weight, std=0.02)
        self._mask_cache: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, ...]] = {}

    def _get_masks(self, L: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (L, device, dtype)
        if key not in self._mask_cache:
            strict_tril = torch.tril(torch.ones(L, L, device=device, dtype=dtype), diagonal=-1).view(1, 1, L, L)
            causal_tril = torch.tril(torch.ones(L, L, device=device, dtype=dtype)).view(1, 1, L, L)
            idx = torch.arange(L, device=device, dtype=torch.float32)
            diff = idx.unsqueeze(-1) - idx.unsqueeze(-2)
            decay_mat = torch.where(diff >= 0, torch.pow(0.92, diff.clamp(min=0.0)), torch.zeros_like(diff)).to(dtype).view(1, 1, L, L)
            counts = (1.0 / torch.arange(1, L + 1, device=device, dtype=torch.float32)).view(1, 1, L, 1).to(dtype)
            self._mask_cache[key] = (strict_tril, causal_tril, decay_mat, counts)
        return self._mask_cache[key]

    def _route_coprime_affinity(
        self, q_norm: torch.Tensor, k_norm: torch.Tensor, step_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = q_norm.dtype
        inv_tau = 1.0 / max(0.18, 0.5 * (1.0 - 0.64 * step_ratio))
        hard_blend = min(0.85, step_ratio * 0.85)

        anchors = self.harmonic_anchors.to(orig_dtype)
        kc_q = (F.relu(self.apl_norm(self.kenyon_proj(q_norm))) * anchors) * inv_tau
        kc_k = (F.relu(self.apl_norm(self.kenyon_proj(k_norm))) * anchors) * inv_tau

        p_q_list = []
        p_k_list = []
        idx_q_list = []
        idx_k_list = []
        offset = 0
        for p in self.prime_banks:
            q_slice = kc_q[..., offset : offset + p]
            k_slice = kc_k[..., offset : offset + p]
            p_q_list.append(F.softmax(q_slice, dim=-1))
            p_k_list.append(F.softmax(k_slice, dim=-1))
            idx_q_list.append(torch.argmax(q_slice, dim=-1))
            idx_k_list.append(torch.argmax(k_slice, dim=-1))
            offset += p

        # Single fused 88-dim Tensor-Core GEMM across all 4 Co-Prime banks
        p_q_cat = torch.cat(p_q_list, dim=-1)
        p_k_cat = torch.cat(p_k_list, dim=-1)
        soft_qk = 0.25 * torch.matmul(p_q_cat, p_k_cat.transpose(-1, -2))
        soft_kk = 0.25 * torch.matmul(p_k_cat, p_k_cat.transpose(-1, -2))

        # Vectorized 4-bank hard cell match
        idx_q_stack = torch.stack(idx_q_list, dim=-1)  # (B, H, L, 4)
        idx_k_stack = torch.stack(idx_k_list, dim=-1)  # (B, H, L, 4)
        hard_qk = (idx_q_stack.unsqueeze(-2) == idx_k_stack.unsqueeze(-3)).to(orig_dtype).mean(dim=-1)
        hard_kk = (idx_k_stack.unsqueeze(-2) == idx_k_stack.unsqueeze(-3)).to(orig_dtype).mean(dim=-1)

        ste_qk = hard_qk - soft_qk.detach() + soft_qk
        ste_kk = hard_kk - soft_kk.detach() + soft_kk
        qk_cell = hard_blend * ste_qk + (1.0 - hard_blend) * soft_qk
        kk_cell = hard_blend * ste_kk + (1.0 - hard_blend) * soft_kk
        return kk_cell, qk_cell

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype

        base_qk = self.qk_shared(x)
        q = (base_qk + self.q_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k_raw = (base_qk + self.k_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # 50/50 split between standard key and 1-step shifted induction key
        k_prev = F.pad(k_raw[:, :, :-1, :], (0, 0, 1, 0))
        k = torch.cat([k_raw[:, :self.n_heads // 2], k_prev[:, self.n_heads // 2:]], dim=1)

        strict_tril, causal_tril, decay_mat, inv_counts = self._get_masks(L, device, dtype)

        # Shared unit-hypersphere + clamped RMS normalization for Q and K
        q_f = q.float()
        k_f = k.float()
        q_rms_inv = torch.rsqrt(torch.mean(q_f * q_f, dim=-1, keepdim=True) + 1e-6)
        k_rms_inv = torch.rsqrt(torch.mean(k_f * k_f, dim=-1, keepdim=True) + 1e-6)
        q_unit = (q_f * (q_rms_inv / math.sqrt(self.head_dim))).to(dtype)
        k_unit = (k_f * (k_rms_inv / math.sqrt(self.head_dim))).to(dtype)
        q_clamped = torch.clamp(q_f * q_rms_inv, -3.0, 3.0)
        k_clamped = torch.clamp(k_f * k_rms_inv, -3.0, 3.0)

        # 1. Co-Prime Sherman-Morrison Vault Branch (Fused 4D Tensor-Core Operator)
        kk_cell, qk_cell = self._route_coprime_affinity(q_unit, k_unit, step_ratio)
        b_sem = torch.clamp(self.beta_sem.float(), 1.5, 3.5)
        phi_q = F.normalize(torch.exp(b_sem * q_clamped), p=2, dim=-1).to(dtype)
        phi_k = F.normalize(torch.exp(b_sem * k_clamped), p=2, dim=-1).to(dtype)

        gram_kk = torch.matmul(phi_k, phi_k.transpose(-1, -2)) ** 2
        gram_qk = torch.matmul(phi_q, phi_k.transpose(-1, -2)) ** 2

        raw_M = kk_cell * strict_tril * gram_kk
        M_b = (0.65 * raw_M) / (1.0 + raw_M.sum(dim=-1, keepdim=True))
        Mv = torch.matmul(M_b, v)
        E_all = v - Mv + torch.matmul(M_b, Mv)

        K_qk_all = qk_cell * causal_tril * gram_qk
        den_all = K_qk_all.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        y_vault = torch.matmul(K_qk_all, E_all) / den_all

        # 2. Syntactic Local Flow Branch
        b_syn = torch.clamp(self.beta_syn.float(), 1.5, 3.5)
        psi_q = F.normalize(torch.exp(b_syn * q_clamped), p=2, dim=-1).to(dtype)
        psi_k = F.normalize(torch.exp(b_syn * k_clamped), p=2, dim=-1).to(dtype)
        syn_scores = (torch.matmul(psi_q, psi_k.transpose(-1, -2)) ** 2) * decay_mat
        y_syntax = torch.matmul(syn_scores, v) / syn_scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)

        # 3. Dyadic GTRM Macro-Cascade Branch
        macro_v = torch.cumsum(v, dim=2) * inv_counts
        y_gtrm = self.gtrm_proj(macro_v)

        v_g = self.vault_gate.to(dtype)
        s_g = self.syntax_gate.to(dtype)
        g_g = self.gtrm_gate.to(dtype)
        y = v_g * y_vault + s_g * y_syntax + g_g * y_gtrm

        return self.o_proj(y.transpose(1, 2).contiguous().view(B, L, D))


class VAULTDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        d_ff: int,
        layer_idx: int,
        use_flash_vault: bool = False,
        n_vault_groups: int = 12,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if use_flash_vault:
            from src.flash_vault_module import FlashVAULTAttention
            self.attn = FlashVAULTAttention(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                n_vault_groups=n_vault_groups,
                chunk_size=chunk_size,
                layer_idx=layer_idx,
            )
        else:
            self.attn = VAULTAttention150M(d_model=d_model, n_heads=n_heads, head_dim=head_dim, layer_idx=layer_idx)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), step_ratio=step_ratio)
        x = x + self.mlp(self.norm2(x))
        return x


class VAULTLM150M(nn.Module):
    """
    151.2M / 160.6M Parameter Native VAULT Language Model.
      - vocab_size = 49,152 (SmolLM / Cosmo2 BPE tokenizer)
      - d_model    = 768
      - n_layers   = 16
      - n_heads    = 12 (head_dim = 64)
      - d_ff       = 2048
      - Supports both eager O(L^2) `VAULTAttention150M` and O(L) `FlashVAULTAttention` (`use_flash_vault=True`).
    """
    def __init__(
        self,
        vocab_size: int = 49152,
        d_model: int = 768,
        n_layers: int = 16,
        n_heads: int = 12,
        head_dim: int = 64,
        d_ff: int = 2048,
        use_flash_vault: bool = False,
        n_vault_groups: int = 12,
        chunk_size: int = 64,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_flash_vault = use_flash_vault
        self.n_vault_groups = n_vault_groups
        self.chunk_size = chunk_size
        self.grad_checkpoint = grad_checkpoint
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)

        self.layers = nn.ModuleList([
            VAULTDecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                d_ff=d_ff,
                layer_idx=i,
                use_flash_vault=use_flash_vault,
                n_vault_groups=n_vault_groups,
                chunk_size=chunk_size,
            )
            for i in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie input and output embeddings
        self.lm_head.weight = self.embed_tokens.weight

    def load_pretrained_vault_state(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, list]:
        """
        Loads a `VAULTAttention150M` or `FlashVAULTAttention` checkpoint with strict verification
        that zero unexpected keys exist and all 160.6M core parameters match 100%.
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading VAULT checkpoint: {unexpected}")
        allowed_suffixes = ("attn.write_beta_logit", "attn.bank_offsets", "attn.local_decay_mask")
        bad_missing = [k for k in missing if not k.endswith(allowed_suffixes)]
        if bad_missing:
            raise RuntimeError(f"Missing core parameters when loading VAULT checkpoint: {bad_missing}")
        return len(state_dict), missing

    def forward(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        for layer in self.layers:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer, x, step_ratio, use_reentrant=False)
            else:
                x = layer(x, step_ratio=step_ratio)
        x = self.norm_f(x)
        return self.lm_head(x)

