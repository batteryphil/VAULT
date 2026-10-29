# MIT License
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
from src.kernels.flash_vault_v2_triton import flash_vault_v2_chunkwise


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
        # x: (B, L, C) -> causal (B, L, C)
        L = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        return F.silu(y)


def _apply_local_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies chunk-relative rotary phase to `(..., seq, D)` with zero long-range RoPE decay."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class VAULTv2Attention(nn.Module):
    """
    True 3:1 Grouped-Vault Attention (GVA) Layer for VAULTLM-v2.
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
    ):
        super().__init__()
        assert n_q_heads % n_vault_groups == 0
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_vault_groups = n_vault_groups
        self.heads_per_group = n_q_heads // n_vault_groups
        self.head_dim = head_dim
        self.kv_dim = n_vault_groups * head_dim
        self.prime_banks = prime_banks
        self.num_banks = len(prime_banks)
        self.total_cells = sum(prime_banks)
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Shared metric anchor on vault groups so Q and K align in Co-Prime cells from Step 0
        self.qk_group_anchor = nn.Linear(d_model, self.kv_dim, bias=False)
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_delta = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_delta.weight, std=0.02)

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
        for p in prime_banks:
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

    def _route_cells_v2(
        self, q_unit: torch.Tensor, k_unit: torch.Tensor, step_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routes `H` query heads (`cell_q`: `(B, H, R, L)`) and `G` vault groups (`cell_k`: `(B, G, R, L)`)
        into Co-Prime cells `[0, 88)` with unit-forward Straight-Through Estimator router gradients.
        """
        orig_dtype = q_unit.dtype
        inv_tau = 1.0 / max(0.18, 0.45 * (1.0 - 0.60 * step_ratio))
        anchors = self.harmonic_anchors.to(orig_dtype)

        kc_q = (F.relu(self.apl_norm(self.kenyon_proj(q_unit))) * anchors) * inv_tau  # (B, H, L, 88)
        kc_k = (F.relu(self.apl_norm(self.kenyon_proj(k_unit))) * anchors) * inv_tau  # (B, G, L, 88)

        cq_list, ck_list = [], []
        gq_list, gk_list = [], []
        offset = 0
        for p in self.prime_banks:
            qs = kc_q[..., offset : offset + p]
            ks = kc_k[..., offset : offset + p]
            pq_r = F.softmax(qs, dim=-1)
            pk_r = F.softmax(ks, dim=-1)

            iq = torch.argmax(qs, dim=-1)  # (B, H, L)
            ik = torch.argmax(ks, dim=-1)  # (B, G, L)
            gq_list.append(pq_r.gather(-1, iq.unsqueeze(-1)).squeeze(-1))
            gk_list.append(pk_r.gather(-1, ik.unsqueeze(-1)).squeeze(-1))

            cq_list.append(iq + offset)
            ck_list.append(ik + offset)
            offset += p

        cell_q = torch.stack(cq_list, dim=2)  # (B, H, R, L)
        cell_k = torch.stack(ck_list, dim=2)  # (B, G, R, L)

        gq_mean = torch.stack(gq_list, dim=2).mean(dim=2)  # (B, H, L)
        gk_mean = torch.stack(gk_list, dim=2).mean(dim=2)  # (B, G, L)

        ste_q = (1.0 + (gq_mean - gq_mean.detach())).unsqueeze(-1).to(orig_dtype)
        ste_k = (1.0 + (gk_mean - gk_mean.detach())).unsqueeze(-1).to(orig_dtype)
        return cell_q, cell_k, ste_q, ste_k

    def _chunk_local_rope_syntax(
        self, q_norm: torch.Tensor, k_norm: torch.Tensor, v_grp: torch.Tensor
    ) -> torch.Tensor:
        """
        Strictly O(L) `64 x 128` sliding block local attention with chunk-invariant RoPE
        and multi-scale geometric head decay (`gamma_h in [0.66, 0.992]`).
        """
        B, H, L_orig, D = q_norm.shape
        C = self.chunk_size
        k_exp = k_norm.repeat_interleave(self.heads_per_group, dim=1)  # (B, H, L, D)
        v_exp = v_grp.repeat_interleave(self.heads_per_group, dim=1)   # (B, H, L, D)

        rem = L_orig % C
        pad_l = (C - rem) if rem != 0 else 0
        if pad_l > 0:
            q_norm = F.pad(q_norm, (0, 0, 0, pad_l))
            k_exp = F.pad(k_exp, (0, 0, 0, pad_l))
            v_exp = F.pad(v_exp, (0, 0, 0, pad_l))

        L = q_norm.size(2)
        M = L // C

        q_blk = _apply_local_rope(q_norm.view(B, H, M, C, D), self.rope_cos_q, self.rope_sin_q)
        k_pad = F.pad(k_exp, (0, 0, C, 0))
        v_pad = F.pad(v_exp, (0, 0, C, 0))
        k_win = _apply_local_rope(
            k_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3),
            self.rope_cos_k,
            self.rope_sin_k,
        )
        v_win = v_pad.unfold(dimension=2, size=2 * C, step=C).permute(0, 1, 2, 4, 3)

        # Scaled dot-product softmax modulated by multi-scale causal decay -> exact relative position precision!
        scale = 1.0 / math.sqrt(D)
        logits = torch.matmul(q_blk, k_win.transpose(-1, -2)).float() * scale
        decay = self.local_decay_mask  # (1, H, 1, C, 2C)
        logits = torch.where(decay > 0.0, logits + torch.log(decay.clamp(min=1e-8)), torch.full_like(logits, -1e9))
        probs = F.softmax(logits, dim=-1).to(v_exp.dtype)
        out_blk = torch.matmul(probs, v_win)
        return out_blk.reshape(B, H, L, D)[:, :, :L_orig, :]

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, _ = x.shape

        # 1. Shared-Metric Anchor + Multi-Token Causal ShortConv1D(k=4)
        anchor = self.qk_group_anchor(x)  # (B, L, G*D)
        anchor_q = anchor.view(B, L, self.n_vault_groups, 1, self.head_dim).expand(
            B, L, self.n_vault_groups, self.heads_per_group, self.head_dim
        ).reshape(B, L, self.n_q_heads * self.head_dim)

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

        # 2. True 3:1 Grouped-Vault Co-Prime Associative Memory (`O(L)` Chunkwise Compact WY)
        cell_q, cell_k, ste_q, ste_k = self._route_cells_v2(q_unit, k_unit, step_ratio)
        bq = torch.clamp(self.beta_sem_q.float(), 1.2, 3.5)
        bk = torch.clamp(self.beta_sem_k.float(), 1.2, 3.5)
        phi_q = F.normalize(torch.exp(bq * torch.clamp(q_f * q_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype) * ste_q
        phi_k = F.normalize(torch.exp(bk * torch.clamp(k_f * k_rms, -3.0, 3.0)), p=2, dim=-1).to(dtype) * ste_k

        # Token-selective write/erase gate `beta_{b,g,t} in (0.02, 0.96)`
        beta_write = (0.02 + 0.94 * torch.sigmoid(self.beta_proj(x).transpose(1, 2))).to(dtype)
        y_vault = flash_vault_v2_chunkwise(phi_q, phi_k, v, beta_write, cell_q, cell_k, chunk_size=self.chunk_size)

        # 3. O(L) Chunk-Local RoPE + Multi-Scale Decay Syntax Branch
        y_syntax = self._chunk_local_rope_syntax(q_norm, k_norm, v)

        # 4. O(L) GTRM Macro-Horizon Summary Branch
        inv_counts = (1.0 / torch.arange(1, L + 1, device=x.device, dtype=torch.float32)).view(1, 1, L, 1).to(dtype)
        v_macro = self.gtrm_proj(torch.cumsum(v, dim=2) * inv_counts).repeat_interleave(self.heads_per_group, dim=1)

        y = (
            self.vault_gate.to(dtype) * y_vault
            + self.syntax_gate.to(dtype) * y_syntax
            + self.gtrm_gate.to(dtype) * v_macro
        )

        # 5. Per-Head Sub-Layer RMSNorm + Data-Dependent Sigmoid Output Gate
        y_norm = self.head_norm(y.transpose(1, 2)).reshape(B, L, self.n_q_heads * self.head_dim)
        gate = torch.sigmoid(self.out_gate(x)).to(y_norm.dtype)
        return self.o_proj(y_norm * gate)


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
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = VAULTv2Attention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_vault_groups=n_vault_groups,
            head_dim=head_dim,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), step_ratio=step_ratio)
        x = x + self.mlp(self.norm2(x))
        return x


class VAULTLMv2(nn.Module):
    """
    VAULTLM-v2 (~152M Parameters):
      - vocab_size     = 49,152 (SmolLM BPE with native `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<file_sep>`)
      - d_model        = 768
      - n_layers       = 16
      - n_q_heads      = 12, n_vault_groups = 4 (`head_dim = 64`, True 3:1 Grouped-Vault Attention)
      - d_ff           = 2048 (`SwiGLU`)
      - Causal `ShortConv1D(k=4)` + Token-Selective `beta_t` + Chunk-Relative RoPE Multi-Scale Syntax
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

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            VAULTv2Block(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_vault_groups=n_vault_groups,
                head_dim=head_dim,
                d_ff=d_ff,
                chunk_size=chunk_size,
                layer_idx=i,
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _encode_hidden(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        for idx, layer in enumerate(self.layers):
            use_ckpt = False
            if self.training and torch.is_grad_enabled():
                if self.grad_checkpoint == "half":
                    use_ckpt = (idx % 2 == 0)
                elif self.grad_checkpoint == "three_quarter":
                    use_ckpt = (idx % 4 != 3)
                elif bool(self.grad_checkpoint):
                    use_ckpt = True
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(layer, x, step_ratio, use_reentrant=False)
            else:
                x = layer(x, step_ratio=step_ratio)
        return self.norm_f(x)

    def forward(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        return self.lm_head(self._encode_hidden(input_ids, step_ratio=step_ratio))

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        step_ratio: float = 1.0,
        logit_chunk_tokens: int = 2048,
    ) -> torch.Tensor:
        """
        Memory-optimal chunked vocabulary cross-entropy: avoids allocating multi-GB `(B*L, V)` FP32
        logit buffers while computing the exact weighted cross-entropy loss and gradients.
        """
        h = self._encode_hidden(input_ids, step_ratio=step_ratio).reshape(-1, self.d_model)
        t_flat = targets.reshape(-1)
        w_flat = loss_weights.reshape(-1) if loss_weights is not None else None
        n_tok = h.size(0)

        def _chunk_loss_sum(h_sub: torch.Tensor, t_sub: torch.Tensor, w_sub: Optional[torch.Tensor]) -> torch.Tensor:
            logits_sub = self.lm_head(h_sub)
            ce = F.cross_entropy(logits_sub.float(), t_sub, ignore_index=-100, reduction="none")
            if w_sub is not None:
                return (ce * w_sub).sum()
            return ce.sum()

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

        return total_loss / denom
