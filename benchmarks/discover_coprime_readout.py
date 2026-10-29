#!/usr/bin/env python3
"""
discover_coprime_readout.py — Phase 2: Symbolic Discovery of the Co-Prime Readout Rule
======================================================================================
Uses Philip Bartsch's PRIME-Net (AFPO + MDL Symbolic Regression Engine) to discover
the optimal closed-form mathematical readout operator across H=4 Co-Prime Kenyon
Associative Banks under heavy collision noise.

Problem Formulation:
- A target key-value pair (k*, v*) is stored in H=4 co-prime banks P = {17, 19, 23, 29}.
- Over thousands of subsequent tokens, unrelated writes collide with the target cell
  in individual banks (Poisson collision load c_h), perturbing each bank's readout:
      r_h = v* + eps_h,   where Var(eps_h) ~ sigma_h^2 (tracked by cell load / residual energy)
- We provide PRIME-Net with sorted/precision-weighted bank readouts and inverse-load
  weights to discover the minimum-MDL invariant that recovers v* while rejecting
  collided outlier banks.
"""

import os
import sys
import json
import numpy as np

# Add PRIME-Net to path (100% owned by Philip Bartsch / batteryphil)
PRIME_NET_PATH = os.environ.get("PRIME_NET_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PRIME-Net")))
sys.path.insert(0, PRIME_NET_PATH)

from prime_core import run_prime_engine


def generate_coprime_collision_telemetry(n_samples: int = 2000, seed: int = 42):
    """
    Simulate exact Co-Prime Bank readouts (P = 17, 19, 23, 29) after T = 25,000 steps.
    By the Chinese Remainder Theorem, a burst of interfering tokens that collides with
    our target key in 1 or 2 banks will NOT collide in the remaining co-prime banks.
    """
    rng = np.random.default_rng(seed)

    # True stored scalar value component v* in [-2, 2]
    v_true = rng.uniform(-2.0, 2.0, size=n_samples)

    # Simulate 4 co-prime banks where 1 or 2 banks suffer heavy collision spikes
    # while the other co-prime banks have low background noise.
    r_banks = np.zeros((n_samples, 4), dtype=np.float64)
    s_banks = np.zeros((n_samples, 4), dtype=np.float64)  # tracked collision energy per cell

    for i in range(n_samples):
        # Number of heavily collided banks (1 or 2 out of 4)
        n_collided = rng.choice([1, 2], p=[0.65, 0.35])
        collided_idx = rng.choice(4, size=n_collided, replace=False)

        for h in range(4):
            if h in collided_idx:
                # Heavy collision in this bank
                sigma = rng.uniform(0.8, 2.5)
                noise = rng.normal(0.0, sigma)
            else:
                # Uncollided co-prime bank (clean retention)
                sigma = rng.uniform(0.01, 0.08)
                noise = rng.normal(0.0, sigma)

            r_banks[i, h] = v_true[i] + noise
            s_banks[i, h] = sigma ** 2

    # Construct candidate invariant features for PRIME-Net:
    # X0: Simple mean of the 4 banks (standard linear ensemble)
    mean_r = np.mean(r_banks, axis=1)

    # X1, X2: The two middle-ranked readouts across the 4 co-prime banks (Order-Statistic / Median pair)
    sorted_r = np.sort(r_banks, axis=1)
    r_mid1 = sorted_r[:, 1]
    r_mid2 = sorted_r[:, 2]

    # X3: Inverse-variance (Sherman-Morrison / Energy precision) weighted readout
    prec = 1.0 / (s_banks + 1e-3)
    prec_norm = prec / np.sum(prec, axis=1, keepdims=True)
    r_prec = np.sum(prec_norm * r_banks, axis=1)

    # X4: Harmonic Consensus Readout (combines precision with distance from median)
    med_r = 0.5 * (r_mid1 + r_mid2)[:, None]
    dev2 = (r_banks - med_r) ** 2
    harm_w = 1.0 / (s_banks + dev2 + 1e-3)
    harm_w /= np.sum(harm_w, axis=1, keepdims=True)
    r_harm = np.sum(harm_w * r_banks, axis=1)

    # X5: Disagreement spread between banks
    spread = sorted_r[:, 3] - sorted_r[:, 0]

    X = np.column_stack([mean_r, r_mid1, r_mid2, r_prec, r_harm, spread])
    return X, v_true, r_banks


def main():
    print("=" * 78)
    print("PRIME-Net Symbolic Discovery: Optimal Co-Prime Bank Consensus Readout")
    print("=" * 78)

    X_train, y_train, r_train = generate_coprime_collision_telemetry(n_samples=1500, seed=42)
    X_test, y_test, r_test = generate_coprime_collision_telemetry(n_samples=500, seed=99)

    # Baseline MSEs
    mse_single_bank = float(np.mean((r_test[:, 0] - y_test) ** 2))
    mse_linear_mean = float(np.mean((X_test[:, 0] - y_test) ** 2))
    mse_median_pair = float(np.mean((0.5 * (X_test[:, 1] + X_test[:, 2]) - y_test) ** 2))
    mse_prec_weight = float(np.mean((X_test[:, 3] - y_test) ** 2))
    mse_harm_cons   = float(np.mean((X_test[:, 4] - y_test) ** 2))

    print(f"Baseline Single Bank MSE            : {mse_single_bank:.6f}")
    print(f"Baseline Linear Mean (4 Banks) MSE  : {mse_linear_mean:.6f}")
    print(f"Baseline Order-Median Pair MSE      : {mse_median_pair:.6f}")
    print(f"Baseline Precision-Weighted MSE     : {mse_prec_weight:.6f}")
    print(f"Baseline Harmonic-Consensus MSE     : {mse_harm_cons:.6f}")
    print("-" * 78)

    # Run PRIME-Net AFPO + MDL Symbolic Regression
    res = run_prime_engine(
        X_train, y_train,
        X_test, y_test,
        pop_size=512,
        max_generations=400,
        timeout_sec=20.0,
        lambda_penalty=0.002,
        seed=42,
    )

    print(f"[PRIME-Net] Discovered Symbolic Invariant : {res['best_sympy']}")
    print(f"[PRIME-Net] Train R^2                     : {res['train_r2']:.6f}")
    print(f"[PRIME-Net] Test  R^2                     : {res['test_r2']:.6f}")
    print(f"[PRIME-Net] Test  MSE                     : {res['test_mse']:.6f}")
    print(f"[PRIME-Net] Noise Reduction vs Single Bank: {mse_single_bank / max(res['test_mse'], 1e-9):.1f}x")
    print(f"[PRIME-Net] Noise Reduction vs Linear Mean: {mse_linear_mean / max(res['test_mse'], 1e-9):.1f}x")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "primenet_readout_discovery.json")
    with open(out_path, "w") as f:
        json.dump({
            "mse_single_bank": mse_single_bank,
            "mse_linear_mean": mse_linear_mean,
            "mse_median_pair": mse_median_pair,
            "mse_prec_weight": mse_prec_weight,
            "mse_harm_cons": mse_harm_cons,
            "primenet_best_sympy": str(res["best_sympy"]),
            "primenet_test_r2": float(res["test_r2"]),
            "primenet_test_mse": float(res["test_mse"]),
            "snr_gain_vs_single_bank": float(mse_single_bank / max(res["test_mse"], 1e-9)),
            "snr_gain_vs_linear_mean": float(mse_linear_mean / max(res["test_mse"], 1e-9)),
        }, f, indent=2)
    print(f"Saved symbolic discovery telemetry to {out_path}")


if __name__ == "__main__":
    main()
