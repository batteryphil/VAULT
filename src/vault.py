"""
vault.py — The VAULT Foundational Sequence Architecture
=======================================================
VAULT: Variance-Weighted Associative Unitary Lattice Tensor
100% Original Architecture by Philip Bartsch (@batteryphil) — MIT License.

Unifies 7 foundational mechanisms from Philip Bartsch's owned repositories:
  1. Bit-Sliced 1.58-Bit Ternary Projections (`buildbit-os`, `synapse-os`)
  2. Prime Reciprocal Harmonic Anchors {1/p} (`prime-revisited`)
  3. Sparse Connectome Routing + FusedAPLNorm Homeostatic Inhibition (`BioNPC`)
  4. PRIME-Net Discovered High-Gain Kernel exp(2 * RMSNorm(x)) (`gist-memory`, `PRIME-Net`)
  5. Sherman-Morrison Orthogonal Nullspace Recurrence (`PRIME-Moment-Attention`)
  6. Dyadic 256-Token GTRM / ConceptPerceptron Multi-Scale Cascade (`prime_gtrm.h`, `thalamic-bloom`)
  7. Subspace-Factored Co-Prime Routing & Variance-Weighted Consensus Readout (`PRIME-Net`)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


PRIME_ANCHORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
DEFAULT_SUBSPACE_PRIMES = (83, 89, 97, 101)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class FusedAPLNorm(nn.Module):
    """
    Anterior Paired Lateral (APL) Recurrent Homeostatic Inhibition
    (from BioNPC/src/bionpc/model.py).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.decay_logit = nn.Parameter(torch.tensor(2.0))
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward_step(
        self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay_logit)
        gain = F.relu(self.gain)
        activity = x.abs().mean(dim=-1)
        if prev_state is not None:
            new_state = torch.clamp(prev_state * decay + activity * gain, -0.95, 0.95)
        else:
            new_state = torch.clamp(activity * gain, -0.95, 0.95)
        return x * (1.0 - new_state.unsqueeze(-1)), new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_step(x, None)
        return out


class TernarySTEProjection(nn.Module):
    """
    1.58-Bit Ternary {-1, 0, +1} Linear Projection (from buildbit-os & synapse-os).
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        gen = torch.Generator()
        gen.manual_seed(seed)
        w = torch.randn(out_features, in_features, generator=gen) / math.sqrt(in_features)
        self.weight = nn.Parameter(w)

    def quantize_ternary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-5)
        w_trit = torch.clamp(torch.round(w / scale), -1.0, 1.0)
        w_ste = w + (w_trit * scale - w).detach()
        return w_ste, w_trit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, _ = self.quantize_ternary()
        return F.linear(x, w_ste)


class VaultCoPrimeRouter(nn.Module):
    """
    Sparse Connectome Router over Co-Prime Vault Banks (BioNPC + prime-revisited).
    Maps a key or query vector into 1 resonant vault cell index per Co-Prime bank P_h,
    using sparse k=4 projections modulated by Prime Reciprocal Harmonics {1/p}.
    """
    def __init__(
        self,
        d_key: int,
        coprime_banks: Tuple[int, ...] = DEFAULT_SUBSPACE_PRIMES,
        k_sparse: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.d_key = d_key
        self.coprime_banks = coprime_banks
        self.num_banks = len(coprime_banks)
        self.total_cells = sum(coprime_banks)
        self.k_sparse = min(k_sparse, d_key)

        gen = torch.Generator()
        gen.manual_seed(seed)

        W_vault = torch.zeros(self.total_cells, d_key)
        for c_idx in range(self.total_cells):
            cols = torch.randperm(d_key, generator=gen)[: self.k_sparse]
            for j, col in enumerate(cols):
                prime_p = PRIME_ANCHORS[(c_idx + j) % len(PRIME_ANCHORS)]
                sign = 1.0 if ((c_idx * 31 + j * 17) % 2 == 0) else -1.0
                W_vault[c_idx, col] = sign * (1.0 + 1.0 / float(prime_p))

        W_vault = F.normalize(W_vault, p=2, dim=-1)
        self.register_buffer("W_vault", W_vault)

    def route(self, k_vec: torch.Tensor) -> List[torch.Tensor]:
        scores = F.linear(F.normalize(k_vec, p=2, dim=-1), self.W_vault)
        indices: List[torch.Tensor] = []
        offset = 0
        for P_h in self.coprime_banks:
            bank_scores = scores[..., offset : offset + P_h]
            c_h = torch.argmax(bank_scores, dim=-1)
            indices.append(c_h)
            offset += P_h
        return indices


def apply_prime_kernel(x: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
    """
    PRIME-Net discovered high-gain positive kernel (gist-memory/src/gist_memory/core.py)
    stabilized onto the unit L2 hypersphere for exact Sherman-Morrison conditioning.
    """
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    x_norm = torch.clamp(x * rms, -3.0, 3.0)
    phi = torch.exp(temp * x_norm)
    return F.normalize(phi, p=2, dim=-1)


class VaultAssociativeMemory:
    """
    VAULT: Variance-Weighted Associative Unitary Lattice Tensor
    Stateful O(1) Co-Prime Nullspace + Dyadic GTRM Memory Engine.
    """
    def __init__(
        self,
        d_key: int = 32,
        d_val: int = 64,
        coprime_banks: Tuple[int, ...] = DEFAULT_SUBSPACE_PRIMES,
        subspace_factored: bool = True,
        ridge_mu: float = 1e-2,
        innovation_threshold: float = 1e-3,
        min_free_rank_ratio: float = 0.05,
        rejuvenation_rate: float = 0.12,
        use_prime_kernel: bool = True,
        device: str = "cpu",
    ):
        self.d_key = d_key
        self.d_val = d_val
        self.coprime_banks = coprime_banks
        self.num_banks = len(coprime_banks)
        self.subspace_factored = subspace_factored and (d_val % self.num_banks == 0)
        self.d_sub = (d_val // self.num_banks) if self.subspace_factored else d_val
        self.ridge_mu = ridge_mu
        self.innovation_threshold = innovation_threshold
        self.min_free_rank_ratio = min_free_rank_ratio
        self.rejuvenation_rate = rejuvenation_rate
        self.use_prime_kernel = use_prime_kernel
        self.device = device

        self.router = VaultCoPrimeRouter(
            d_key=d_key, coprime_banks=coprime_banks
        ).to(device)

        self.S: List[torch.Tensor] = []
        self.P: List[torch.Tensor] = []
        self.cell_load: List[torch.Tensor] = []

        eye_init = torch.eye(d_key, device=device, dtype=torch.float32) / ridge_mu
        for P_h in coprime_banks:
            self.S.append(torch.zeros(P_h, self.d_sub, d_key, device=device, dtype=torch.float32))
            self.P.append(eye_init.unsqueeze(0).repeat(P_h, 1, 1).clone())
            self.cell_load.append(torch.zeros(P_h, device=device, dtype=torch.float32))

        self.num_dyadic_levels = 12
        self.gtrm_chunk_size = 256
        self.gtrm_current = torch.zeros(d_key, d_val, device=device, dtype=torch.float32)
        self.gtrm_cascade = torch.zeros(
            self.num_dyadic_levels, d_key, d_val, device=device, dtype=torch.float32
        )
        self.step_count = 0
        self.skipped_redundant_writes = 0
        self.executed_novel_writes = 0

    @property
    def state_bytes(self) -> int:
        total_cells = sum(self.coprime_banks)
        micro_floats = total_cells * (self.d_sub * self.d_key + self.d_key * self.d_key + 1)
        macro_floats = (1 + self.num_dyadic_levels) * (self.d_key * self.d_val)
        return (micro_floats + macro_floats) * 4

    def reset(self):
        eye_init = torch.eye(self.d_key, device=self.device, dtype=torch.float32) / self.ridge_mu
        for h, P_h in enumerate(self.coprime_banks):
            self.S[h].zero_()
            self.P[h].copy_(eye_init.unsqueeze(0).repeat(P_h, 1, 1))
            self.cell_load[h].zero_()
        self.gtrm_current.zero_()
        self.gtrm_cascade.zero_()
        self.step_count = 0
        self.skipped_redundant_writes = 0
        self.executed_novel_writes = 0

    def _transform_key(self, k: torch.Tensor) -> torch.Tensor:
        if self.use_prime_kernel:
            return apply_prime_kernel(k)
        return F.normalize(k, p=2, dim=-1)

    @torch.no_grad()
    def write_step(self, k_in: torch.Tensor, v_in: torch.Tensor) -> bool:
        self.step_count += 1
        c_indices = self.router.route(k_in)
        phi_k = self._transform_key(k_in)

        wrote_any = False
        max_trace = float(self.d_key) / self.ridge_mu

        for h in range(self.num_banks):
            c = int(c_indices[h].item())
            S_cell = self.S[h][c]
            P_cell = self.P[h][c]

            v_target = (
                v_in[h * self.d_sub : (h + 1) * self.d_sub]
                if self.subspace_factored
                else v_in
            )

            v_hat = torch.mv(S_cell, phi_k)
            error_t = v_target - v_hat
            err_norm = float(torch.norm(error_t).item())

            if err_norm < self.innovation_threshold:
                continue

            wrote_any = True

            current_trace = float(torch.trace(P_cell).item())
            if current_trace < self.min_free_rank_ratio * max_trace:
                eye = torch.eye(self.d_key, device=self.device, dtype=torch.float32) / self.ridge_mu
                P_cell.mul_(1.0 - self.rejuvenation_rate).add_(eye * self.rejuvenation_rate)
                self.cell_load[h][c] *= (1.0 - self.rejuvenation_rate)

            u = torch.mv(P_cell, phi_k)
            k_dot_u = max(float(torch.dot(phi_k, u).item()), 1e-6)
            gamma_s = 1.0 / k_dot_u

            S_cell.add_(torch.outer(error_t, u), alpha=gamma_s)
            if k_dot_u > 2.0:
                gamma_p = 1.0 / (1.0 + k_dot_u)
                P_cell.sub_(torch.outer(u, u), alpha=gamma_p)
                self.cell_load[h][c] += 1.0

        if wrote_any:
            self.executed_novel_writes += 1
        else:
            self.skipped_redundant_writes += 1

        self.gtrm_current.mul_(0.996).add_(torch.outer(phi_k, v_in), alpha=0.004)
        if (self.step_count % self.gtrm_chunk_size) == 0:
            chunk_idx = self.step_count // self.gtrm_chunk_size
            tz = (chunk_idx & -chunk_idx).bit_length() - 1
            tz = min(tz, self.num_dyadic_levels - 1)
            for lvl in range(tz + 1):
                if lvl == 0:
                    self.gtrm_cascade[0].mul_(0.5).add_(self.gtrm_current, alpha=0.5)
                else:
                    self.gtrm_cascade[lvl].mul_(0.5).add_(self.gtrm_cascade[lvl - 1], alpha=0.5)

        return wrote_any

    @torch.no_grad()
    def read_step(self, q_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        c_indices = self.router.route(q_in)
        phi_q = self._transform_key(q_in)

        bank_preds = []
        bank_loads = []

        for h in range(self.num_banks):
            c = int(c_indices[h].item())
            S_cell = self.S[h][c]
            v_h = torch.mv(S_cell, phi_q)
            load_h = float(self.cell_load[h][c].item())
            bank_preds.append(v_h)
            bank_loads.append(load_h)

        loads_t = torch.tensor(bank_loads, device=self.device, dtype=torch.float32)

        if self.subspace_factored:
            v_out = torch.cat(bank_preds, dim=0)
            telemetry = {
                "mean_bank_load": float(loads_t.mean().item()),
                "min_bank_load": float(loads_t.min().item()),
            }
            return v_out, telemetry

        V_stack = torch.stack(bank_preds, dim=0)
        sorted_V, _ = torch.sort(V_stack, dim=0)
        if self.num_banks >= 4:
            v_median = 0.5 * (sorted_V[1] + sorted_V[2])
        else:
            v_median = torch.median(V_stack, dim=0).values

        dev2 = torch.mean((V_stack - v_median.unsqueeze(0)) ** 2, dim=-1)
        norm_load = loads_t / (torch.mean(loads_t) + 1.0)
        harm_weights = 1.0 / (dev2 + 0.01 * norm_load + 1e-4)
        harm_weights = harm_weights / torch.sum(harm_weights)
        v_consensus = torch.mv(V_stack.t(), harm_weights)

        telemetry = {
            "mean_bank_load": float(loads_t.mean().item()),
            "min_bank_load": float(loads_t.min().item()),
        }
        return v_consensus, telemetry


# Backwards-compatible alias
KenyonPrimeAssociativeMemory = VaultAssociativeMemory
