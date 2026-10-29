#!/usr/bin/env python3
"""
test_kenyon_prime_rigorous.py — Strict Grounded-Truth Audit & Multi-Regime Falsification
========================================================================================
Tests Kenyon-PRIME under adversarial conditions where naive tricks fail:
  1. Pure Non-Repeating Novel Stream (0% repeats — every token is brand-new Gaussian noise,
     so the Innovation Gate cannot skip redundant tokens; tests pure Co-Prime Nullspace
     capacity up to N = 3,072 unique facts).
  2. Dyadic 256-Token GTRM Macro-Horizon Retrieval (retrieving coarse semantic gist
     across 4,096 tokens where flat GistLayer washes out).
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vault import (
    VaultAssociativeMemory as KenyonPrimeAssociativeMemory,
    apply_prime_kernel,
)


def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def test_1_pure_non_repeating_capacity():
    print("\n" + "=" * 96)
    print("TEST 1: Adversarial 100% Non-Repeating Stream (Zero Redundancy, Pure Orthogonal Capacity)")
    print("=" * 96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_key, d_val = 32, 64

    unique_counts = [32, 64, 128, 256, 512, 1024, 2048, 3072]
    n_probes = 32

    print(
        f"{'Unique Facts N':<14} | {'Linear(ELU)':<11} | {'Dense-RLS':<11} | "
        f"{'Replicated 4-Bank':<19} | {'Subspace Co-Prime P={83,89,97,101}':<32}"
    )
    print("-" * 96)

    results_t1 = []
    for N in unique_counts:
        torch.manual_seed(2026)
        keys = F.normalize(torch.randn(N, d_key, device=device), p=2, dim=-1)
        vals = F.normalize(torch.randn(N, d_val, device=device), p=2, dim=-1)

        # 1. Subspace-Factored Co-Prime Kenyon-PRIME (P = {83, 89, 97, 101}, d_sub = 16)
        kp_subspace = KenyonPrimeAssociativeMemory(
            d_key=d_key,
            d_val=d_val,
            coprime_banks=(83, 89, 97, 101),
            subspace_factored=True,
            device=device,
        )

        # 2. Replicated 4-Bank Co-Prime (P = {17, 19, 23, 29}, d_sub = 64)
        kp_repl = KenyonPrimeAssociativeMemory(
            d_key=d_key,
            d_val=d_val,
            coprime_banks=(17, 19, 23, 29),
            subspace_factored=False,
            device=device,
        )

        # 3. Single-Matrix Dense RLS
        S_rls = torch.zeros(d_val, d_key, device=device)
        P_rls = torch.eye(d_key, device=device) * 100.0

        # 4. Flat Linear Attention
        M_lin = torch.zeros(d_val, d_key, device=device)
        Z_lin = torch.zeros(d_key, device=device)

        for i in range(N):
            k_i = keys[i]
            v_i = vals[i]

            kp_subspace.write_step(k_i, v_i)
            kp_repl.write_step(k_i, v_i)

            # Dense RLS write
            phi_k = apply_prime_kernel(k_i)
            if torch.trace(P_rls).item() < 0.05 * d_key * 100.0:
                P_rls = 0.88 * P_rls + 12.0 * torch.eye(d_key, device=device)
            err = v_i - torch.mv(S_rls, phi_k)
            u = torch.mv(P_rls, phi_k)
            gamma = 1.0 / (1.0 + max(float(torch.dot(phi_k, u).item()), 0.0))
            S_rls.add_(torch.outer(err, u), alpha=gamma)
            P_rls.sub_(torch.outer(u, u), alpha=gamma)

            # Linear ELU+1 write
            phi_elu = F.elu(k_i) + 1.0
            M_lin.mul_(0.9995).add_(torch.outer(v_i, phi_elu))
            Z_lin.mul_(0.9995).add_(phi_elu)

        # Evaluate recall on the VERY FIRST n_probes facts (oldest memories buried under N - n_probes writes)
        cos_lin, cos_rls, cos_repl, cos_sub = 0.0, 0.0, 0.0, 0.0
        top1_repl, top1_sub, top1_rls = 0, 0, 0

        for i in range(n_probes):
            q_i = keys[i]
            v_true = vals[i]

            # Linear
            phi_elu = F.elu(q_i) + 1.0
            v_lin = torch.mv(M_lin, phi_elu) / (torch.dot(Z_lin, phi_elu) + 1e-4)
            cos_lin += cosine_sim_batch(v_lin, v_true)

            # Dense RLS
            phi_q = apply_prime_kernel(q_i)
            v_rls = torch.mv(S_rls, phi_q)
            cos_rls += cosine_sim_batch(v_rls, v_true)
            sims_rls = torch.mv(vals, F.normalize(v_rls, p=2, dim=-1))
            if int(torch.argmax(sims_rls).item()) == i:
                top1_rls += 1

            # Replicated 4-Bank
            v_repl, _ = kp_repl.read_step(q_i)
            cos_repl += cosine_sim_batch(v_repl, v_true)
            sims_repl = torch.mv(vals, F.normalize(v_repl, p=2, dim=-1))
            if int(torch.argmax(sims_repl).item()) == i:
                top1_repl += 1

            # Subspace-Factored Co-Prime Kenyon-PRIME
            v_sub, _ = kp_subspace.read_step(q_i)
            cos_sub += cosine_sim_batch(v_sub, v_true)
            sims_sub = torch.mv(vals, F.normalize(v_sub, p=2, dim=-1))
            if int(torch.argmax(sims_sub).item()) == i:
                top1_sub += 1

        c_lin = cos_lin / n_probes
        c_rls = cos_rls / n_probes
        c_repl = cos_repl / n_probes
        c_sub = cos_sub / n_probes
        acc_repl = 100.0 * top1_repl / n_probes
        acc_sub = 100.0 * top1_sub / n_probes

        print(
            f"{N:<14d} | {c_lin:>11.4f} | {c_rls:>11.4f} | "
            f"{c_repl:>8.4f} ({acc_repl:>5.1f}%)   | {c_sub:>10.4f} ({acc_sub:>5.1f}% Top-1)"
        )
        results_t1.append({
            "N_unique": N,
            "linear_elu_cos": c_lin,
            "dense_rls_cos": c_rls,
            "replicated_4bank_cos": c_repl,
            "replicated_4bank_top1_pct": acc_repl,
            "subspace_coprime_cos": c_sub,
            "subspace_coprime_top1_pct": acc_sub,
        })

    return results_t1


def test_2_dyadic_gtrm_macro_cascade():
    print("\n" + "=" * 96)
    print("TEST 2: Dyadic 256-Token GTRM Macro-Cascade (Multi-Timescale Context Retention)")
    print("=" * 96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_key, d_val = 32, 64
    torch.manual_seed(77)

    kp = KenyonPrimeAssociativeMemory(d_key=d_key, d_val=d_val, device=device)

    n_epochs = 4
    tokens_per_epoch = 1024
    topic_keys = F.normalize(torch.randn(n_epochs, d_key, device=device), p=2, dim=-1)
    topic_vals = F.normalize(torch.randn(n_epochs, d_val, device=device), p=2, dim=-1)

    flat_gist = torch.zeros(d_key, d_val, device=device)

    for ep in range(n_epochs):
        for t in range(tokens_per_epoch):
            k_t = F.normalize(topic_keys[ep] + 0.5 * torch.randn(d_key, device=device), p=2, dim=-1)
            v_t = F.normalize(topic_vals[ep] + 0.5 * torch.randn(d_val, device=device), p=2, dim=-1)
            kp.write_step(k_t, v_t)
            phi_k = apply_prime_kernel(k_t)
            flat_gist.mul_(0.996).add_(torch.outer(phi_k, v_t), alpha=0.004)

    print(f"After streaming {n_epochs * tokens_per_epoch} tokens across {n_epochs} distinct Topic Epochs:")
    epoch_results = []
    for ep in range(n_epochs):
        phi_q = apply_prime_kernel(topic_keys[ep])
        v_flat = torch.mv(flat_gist.t(), phi_q)
        cos_flat = cosine_sim_batch(v_flat, topic_vals[ep])

        best_cos_dyadic = -1.0
        for lvl in range(4):
            v_lvl = torch.mv(kp.gtrm_cascade[lvl].t(), phi_q)
            c_lvl = cosine_sim_batch(v_lvl, topic_vals[ep])
            if c_lvl > best_cos_dyadic:
                best_cos_dyadic = c_lvl

        dist_ago = (n_epochs - ep) * tokens_per_epoch
        print(f"  Epoch {ep} ({dist_ago:>4d} tokens ago) | Flat Gist Cosine: {cos_flat:>7.4f} | Dyadic GTRM Cascade: {best_cos_dyadic:>7.4f}")
        epoch_results.append({
            "epoch": ep,
            "tokens_ago": dist_ago,
            "flat_gist_cos": cos_flat,
            "dyadic_gtrm_cos": best_cos_dyadic,
        })

    return epoch_results


def main():
    r1 = test_1_pure_non_repeating_capacity()
    r2 = test_2_dyadic_gtrm_macro_cascade()

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "rigorous_audit_results.json")
    with open(out_file, "w") as f:
        json.dump({"test_1_non_repeating_capacity": r1, "test_2_dyadic_gtrm_cascade": r2}, f, indent=2)
    print(f"\nSaved rigorous audit results to {out_file}")


if __name__ == "__main__":
    main()
