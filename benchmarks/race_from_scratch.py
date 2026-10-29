#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# From-Scratch Head-to-Head Pretraining Race:
#   1. Standard Transformer (Causal Softmax Attention)
#   2. Linear / SSM Recurrence (Exponential Decay Kernel)
#   3. VAULT (Co-Prime Nullspace Lattice + PRIME Kernel + Dyadic GTRM)
#
# Evaluates:
#   - Round 1: Real Language Pretraining (TinyStories, 200 steps from random init)
#   - Round 2: From-Scratch In-Context Associative Recall (Passkey/KV lookup under distractors)

import os
import sys
import time
import math
import json
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vault import FusedAPLNorm, TernarySTEProjection, apply_prime_kernel, RMSNorm


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 3):
        super().__init__()
        hidden = d_model * mult
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SoftmaxAttnLayer(nn.Module):
    """Contender 1: Standard Quadratic Softmax Attention."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_raw, v = qkv[0], qkv[1], qkv[2]
        k_prev = F.pad(k_raw[:, :, :-1, :], (0, 0, 1, 0))
        k = torch.cat([k_raw[:, :self.n_heads // 2], k_prev[:, self.n_heads // 2:]], dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, L, D))


class LinearSSMLayer(nn.Module):
    """Contender 2: Mamba/Linear Attention Style Decaying State-Space Recurrence."""
    def __init__(self, d_model: int, n_heads: int, decay: float = 0.95):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.decay = decay
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_raw, k_raw, v = qkv[0], qkv[1], qkv[2]
        k_prev = F.pad(k_raw[:, :, :-1, :], (0, 0, 1, 0))
        k_shifted = torch.cat([k_raw[:, :self.n_heads // 2], k_prev[:, self.n_heads // 2:]], dim=1)

        q = F.normalize(F.elu(q_raw) + 1.0, p=2, dim=-1)
        k = F.normalize(F.elu(k_shifted) + 1.0, p=2, dim=-1)

        idx = torch.arange(L, device=x.device, dtype=torch.float32)
        diff = idx.unsqueeze(-1) - idx.unsqueeze(-2)
        decay_mat = torch.where(diff >= 0, torch.pow(self.decay, diff.clamp(min=0.0)), torch.zeros_like(diff)).view(1, 1, L, L)

        scores = torch.matmul(q, k.transpose(-1, -2)) * decay_mat
        den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        out = torch.matmul(scores, v) / den
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, L, D))


class NativeVAULTLayer(nn.Module):
    """
    Contender 3: Native From-Scratch VAULT Layer.
    Key From-Scratch Principle:
      Because Co-Prime routing hashes Q and K into discrete cells c_b(q) == c_b(k),
      Q and K must share a common metric anchor at initialization (rather than
      uncorrelated random Gaussian matrices that miss 95% of cells at Step 0),
      paired with smooth soft-to-hard temperature annealing.
    """
    def __init__(self, d_model: int, n_heads: int, prime_banks=(17, 19, 23, 29)):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.prime_banks = prime_banks
        self.num_banks = len(prime_banks)
        self.total_cells = sum(prime_banks)

        # Shared metric base for Q/K + learnable asymmetric offsets so c_b(q) == c_b(k) at Step 0
        self.qk_shared = nn.Linear(d_model, d_model, bias=False)
        self.q_delta = nn.Linear(d_model, d_model, bias=False)
        self.k_delta = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.q_delta.weight, std=0.02)
        nn.init.normal_(self.k_delta.weight, std=0.02)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.kenyon_proj = TernarySTEProjection(self.head_dim, self.total_cells)
        self.apl_norm = FusedAPLNorm(self.total_cells)
        anchors = []
        for p in prime_banks:
            anchors.extend([1.0 / float(p)] * p)
        self.register_buffer("harmonic_anchors", torch.tensor(anchors, dtype=torch.float32))

        self.beta_sem = nn.Parameter(torch.full((1, n_heads, 1, 1), 1.6))
        self.beta_syn = nn.Parameter(torch.full((1, n_heads, 1, 1), 2.0))

        self.vault_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.45))
        self.syntax_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.55))
        self.gtrm_gate = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.10))
        self.gtrm_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        nn.init.normal_(self.gtrm_proj.weight, std=0.02)

    def _route_coprime_affinity(self, q: torch.Tensor, k: torch.Tensor, step_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L, D = k.shape
        tau = max(0.10, 0.5 * (1.0 - 0.80 * step_ratio))
        hard_blend = min(1.0, step_ratio * 1.2)

        kc_q = F.relu(self.apl_norm(self.kenyon_proj(F.normalize(q, dim=-1)))) * self.harmonic_anchors
        kc_k = F.relu(self.apl_norm(self.kenyon_proj(F.normalize(k, dim=-1)))) * self.harmonic_anchors

        kk_matches = []
        qk_matches = []
        offset = 0
        for p in self.prime_banks:
            q_slice = kc_q[..., offset : offset + p]
            k_slice = kc_k[..., offset : offset + p]

            p_q = F.softmax(q_slice / tau, dim=-1)
            p_k = F.softmax(k_slice / tau, dim=-1)
            soft_qk = torch.matmul(p_q, p_k.transpose(-1, -2))
            soft_kk = torch.matmul(p_k, p_k.transpose(-1, -2))

            idx_q = torch.argmax(q_slice, dim=-1)
            idx_k = torch.argmax(k_slice, dim=-1)
            hard_qk = (idx_q.unsqueeze(-1) == idx_k.unsqueeze(-2)).float()
            hard_kk = (idx_k.unsqueeze(-1) == idx_k.unsqueeze(-2)).float()

            ste_qk = hard_qk - soft_qk.detach() + soft_qk
            ste_kk = hard_kk - soft_kk.detach() + soft_kk
            qk_matches.append(hard_blend * ste_qk + (1.0 - hard_blend) * soft_qk)
            kk_matches.append(hard_blend * ste_kk + (1.0 - hard_blend) * soft_kk)
            offset += p

        return torch.stack(kk_matches, dim=2), torch.stack(qk_matches, dim=2)

    def forward(self, x: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        B, L, D = x.shape
        device = x.device
        base_qk = self.qk_shared(x)
        q = (base_qk + self.q_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k_raw = (base_qk + self.k_delta(x)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        k_prev = F.pad(k_raw[:, :, :-1, :], (0, 0, 1, 0))
        k = torch.cat([k_raw[:, :self.n_heads // 2], k_prev[:, self.n_heads // 2:]], dim=1)

        # 1. Co-Prime Sherman-Morrison Vault Branch
        kk_cell, qk_cell = self._route_coprime_affinity(q, k, step_ratio)
        phi_q = apply_prime_kernel(q, torch.clamp(self.beta_sem, 1.5, 3.5))
        phi_k = apply_prime_kernel(k, torch.clamp(self.beta_sem, 1.5, 3.5))

        # Sharpened 2nd-order PRIME Moment intra-cell kernel
        gram_kk = (torch.matmul(phi_k, phi_k.transpose(-1, -2)) ** 2).unsqueeze(2)
        gram_qk = (torch.matmul(phi_q, phi_k.transpose(-1, -2)) ** 2).unsqueeze(2)

        strict_tril = torch.tril(torch.ones(L, L, device=device), diagonal=-1).view(1, 1, 1, L, L)
        causal_tril = torch.tril(torch.ones(L, L, device=device)).view(1, 1, 1, L, L)

        # Row-normalized Sherman-Morrison interference operator (spectral radius strictly <= 0.65)
        raw_M = kk_cell * strict_tril * gram_kk
        M_b = 0.65 * raw_M / (1.0 + raw_M.sum(dim=-1, keepdim=True))
        v_exp = v.unsqueeze(2).expand(B, self.n_heads, self.num_banks, L, self.head_dim)
        Mv = torch.matmul(M_b, v_exp)
        E_all = v_exp - Mv + torch.matmul(M_b, Mv)

        K_qk_all = qk_cell * causal_tril * gram_qk
        den_all = K_qk_all.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        v_all = torch.matmul(K_qk_all, E_all) / den_all

        cell_occ = (qk_cell * causal_tril).sum(dim=-1, keepdim=True)
        sigma_sq = F.relu(cell_occ - 1.0) / float(self.head_dim)
        prec_w = 1.0 / (sigma_sq + 1e-3)
        y_vault = (prec_w * v_all).sum(dim=2) / prec_w.sum(dim=2).clamp(min=1e-6)

        # 2. Syntactic Local Flow Branch
        psi_q = apply_prime_kernel(q, torch.clamp(self.beta_syn, 1.5, 3.5))
        psi_k = apply_prime_kernel(k, torch.clamp(self.beta_syn, 1.5, 3.5))
        idx = torch.arange(L, device=device, dtype=torch.float32)
        diff = idx.unsqueeze(-1) - idx.unsqueeze(-2)
        decay_mat = torch.where(diff >= 0, torch.pow(0.92, diff.clamp(min=0.0)), torch.zeros_like(diff)).view(1, 1, L, L)
        syn_scores = (torch.matmul(psi_q, psi_k.transpose(-1, -2)) ** 2) * decay_mat
        y_syntax = torch.matmul(syn_scores, v) / syn_scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)

        # 3. Dyadic GTRM Macro-Cascade Branch
        counts = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, 1, L, 1)
        y_gtrm = self.gtrm_proj(torch.cumsum(v, dim=2) / counts)

        y = self.vault_gate * y_vault + self.syntax_gate * y_syntax + self.gtrm_gate * y_gtrm
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, L, D))


class ScratchLM(nn.Module):
    def __init__(self, arch: str, vocab_size: int = 4096, d_model: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.arch = arch
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if arch == "transformer":
                mixer = SoftmaxAttnLayer(d_model, n_heads)
            elif arch == "linear_ssm":
                mixer = LinearSSMLayer(d_model, n_heads)
            elif arch == "vault":
                mixer = NativeVAULTLayer(d_model, n_heads)
            else:
                raise ValueError(arch)
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(d_model),
                "mixer": mixer,
                "norm2": RMSNorm(d_model),
                "mlp": SwiGLUMLP(d_model, mult=2),
            }))
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.layers:
            x = x + blk["mixer"](blk["norm1"](x), step_ratio=step_ratio)
            x = x + blk["mlp"](blk["norm2"](x))
        return self.lm_head(self.norm_f(x))


def load_tiny_batches(vocab_size: int = 4096, seq_len: int = 128, batch_size: int = 8, num_train: int = 150, num_eval: int = 20, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    train_arrow = os.path.join(hf_cache, "datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-train-00000-of-00004.arrow")
    val_arrow = os.path.join(hf_cache, "datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-validation.arrow")
    train_ds = Dataset.from_file(train_arrow)
    val_ds = Dataset.from_file(val_arrow)

    def pack(ds, count):
        buf = []
        idx = 0
        needed = count * batch_size * seq_len
        while len(buf) < needed and idx < len(ds):
            txt = ds[idx]["text"]
            idx += 1
            if not txt or len(txt) < 40:
                continue
            toks = [t % vocab_size for t in tokenizer.encode(txt, add_special_tokens=False)]
            buf.extend(toks)
        batches = []
        for b in range(count):
            s = b * batch_size * seq_len
            batches.append(torch.tensor(buf[s : s + batch_size * seq_len], dtype=torch.long, device=device).view(batch_size, seq_len))
        return batches

    return pack(train_ds, num_train), pack(val_ds, num_eval)


def generate_associative_kv_batch(batch_size: int, seq_len: int, num_pairs: int, vocab_size: int, device: str):
    """
    Generates synthetic In-Context Associative Recall sequences:
      [k1, v1, k2, v2, ..., k_m, v_m, <distractor noise tokens...>, q_key, target_val]
    where keys are in [1..32], values are in [33..64] (random chance = 3.1%),
    and distractors are in [100..500], testing long-range in-context lookup.
    """
    seqs = torch.randint(100, 500, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(32, device=device)[:num_pairs] + 1
        vals = torch.randint(33, 65, (num_pairs,), device=device)
        for p in range(num_pairs):
            seqs[b, 2 * p] = keys[p]
            seqs[b, 2 * p + 1] = vals[p]
        q_idx = int(torch.randint(0, num_pairs // 2, (1,)).item())
        seqs[b, -2] = keys[q_idx]
        seqs[b, -1] = vals[q_idx]
        targets[b] = vals[q_idx]
    return seqs, targets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 94, flush=True)
    print("FROM-SCRATCH HEAD-TO-HEAD RACE: TRANSFORMER vs. LINEAR-SSM vs. VAULT", flush=True)
    print("=" * 94, flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n", flush=True)

    vocab_size = 4096
    train_batches, eval_batches = load_tiny_batches(
        vocab_size=vocab_size, seq_len=128, batch_size=8, num_train=150, num_eval=16, device=device
    )

    contenders = ["transformer", "linear_ssm", "vault"]
    round1_results = {}

    print("ROUND 1: REAL LANGUAGE PRETRAINING FROM RANDOM INITIALIZATION (150 Steps, B=8, L=128)", flush=True)
    print("-" * 94, flush=True)
    print(f"{'Architecture':<16} | {'Params':<10} | {'Step 0 PPL':<12} | {'Step 50 PPL':<12} | {'Step 100 PPL':<13} | {'Final PPL':<11} | {'Time':<8}", flush=True)
    print("-" * 94, flush=True)

    for arch in contenders:
        torch.manual_seed(1337)
        model = ScratchLM(arch, vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

        def eval_ppl():
            model.eval()
            tot_loss, tot_tok = 0.0, 0
            with torch.no_grad():
                for b_ids in eval_batches:
                    logits = model(b_ids, step_ratio=1.0)[:, :-1, :].contiguous()
                    labels = b_ids[:, 1:].contiguous()
                    tot_loss += F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction="sum").item()
                    tot_tok += labels.numel()
            model.train()
            return math.exp(min(tot_loss / tot_tok, 20.0))

        ppl_0 = eval_ppl()
        ppls = {0: ppl_0}
        t0 = time.perf_counter()

        for step, b_ids in enumerate(train_batches, start=1):
            opt.zero_grad()
            logits = model(b_ids, step_ratio=step / float(len(train_batches)))[:, :-1, :].contiguous()
            labels = b_ids[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (50, 100, 150):
                ppls[step] = eval_ppl()

        dt = time.perf_counter() - t0
        print(f"{arch.upper():<16} | {n_params/1e6:>6.2f}M   | {ppls[0]:<12.1f} | {ppls[50]:<12.2f} | {ppls[100]:<13.2f} | {ppls[150]:<11.2f} | {dt:>5.1f}s", flush=True)
        round1_results[arch] = {
            "params": n_params,
            "ppl_0": round(ppls[0], 2),
            "ppl_50": round(ppls[50], 2),
            "ppl_100": round(ppls[100], 2),
            "ppl_150": round(ppls[150], 2),
            "time_sec": round(dt, 2),
        }

    print("\n" + "=" * 94, flush=True)
    print("ROUND 2: IN-CONTEXT LONG-RANGE ASSOCIATIVE RECALL FROM SCRATCH (16 KV Pairs + Distractors)", flush=True)
    print("-" * 94, flush=True)
    print(f"{'Architecture':<16} | {'Step 50 Acc':<15} | {'Step 100 Acc':<15} | {'Step 200 Acc':<15} | {'Final Query Loss':<18}", flush=True)
    print("-" * 94, flush=True)

    # Pre-generate 200 training batches and 20 eval batches of associative recall
    torch.manual_seed(999)
    kv_train = [generate_associative_kv_batch(16, 128, 16, vocab_size, device) for _ in range(200)]
    kv_eval = [generate_associative_kv_batch(16, 128, 16, vocab_size, device) for _ in range(20)]

    round2_results = {}
    for arch in contenders:
        torch.manual_seed(2026)
        model = ScratchLM(arch, vocab_size=vocab_size, d_model=256, n_layers=2, n_heads=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

        def eval_kv():
            model.eval()
            correct, total, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for seqs, tgt in kv_eval:
                    logits = model(seqs, step_ratio=1.0)
                    q_logits = logits[:, -2, :]  # Predicts token at -1 (the target value!)
                    loss_sum += F.cross_entropy(q_logits, tgt).item()
                    preds = torch.argmax(q_logits, dim=-1)
                    correct += int((preds == tgt).sum().item())
                    total += tgt.numel()
            model.train()
            return 100.0 * correct / total, loss_sum / len(kv_eval)

        accs = {}
        final_loss = 0.0
        for step, (seqs, tgt) in enumerate(kv_train, start=1):
            opt.zero_grad()
            logits = model(seqs, step_ratio=step / 200.0)
            q_logits = logits[:, -2, :]
            loss = F.cross_entropy(q_logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (50, 100, 200):
                acc, fl = eval_kv()
                accs[step] = acc
                final_loss = fl

        print(f"{arch.upper():<16} | {accs[50]:>10.1f}%    | {accs[100]:>10.1f}%    | {accs[200]:>10.1f}%    | {final_loss:<18.4f}", flush=True)
        round2_results[arch] = {
            "acc_50": round(accs[50], 1),
            "acc_100": round(accs[100], 1),
            "acc_200": round(accs[200], 1),
            "final_query_loss": round(final_loss, 4),
        }

    print("=" * 94, flush=True)
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "from_scratch_race_results.json")
    with open(out_file, "w") as f:
        json.dump({"round1_language": round1_results, "round2_associative_recall": round2_results}, f, indent=2)
    print(f"Saved head-to-head race results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
