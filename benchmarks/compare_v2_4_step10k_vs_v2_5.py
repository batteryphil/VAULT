#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Automated Side-by-Side Comparison:
#   VAULT v2.4 @ Step 10,000 (Pre-SFT Full Suite) vs VAULT v2.5 (Post-SFT Full Suite)
#   Compares ONLY our model across all 4 benchmark pillars + verbatim decoded outputs.

import argparse
import json
import os
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _fmt_delta(v25: float, v24: float, lower_is_better: bool = False, unit: str = "") -> str:
    d = v25 - v24
    sign = "+" if d >= 0 else ""
    good = (d <= 0) if lower_is_better else (d >= 0)
    tag = "IMPROVED" if abs(d) > 1e-6 and good else ("SAME" if abs(d) <= 1e-6 else "REGRESSED")
    return f"{sign}{d:.2f}{unit} ({tag})"


def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Compare VAULT v2.4 Step 10k vs VAULT v2.5 Post-SFT")
    parser.add_argument(
        "--trad_json",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "results_vault_traditional.json"),
    )
    parser.add_argument(
        "--pre_suite_json",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "results_vault_v2_4_step10k_suite.json"),
    )
    parser.add_argument(
        "--post_suite_json",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "results_vault_v2_5_post10k.json"),
    )
    parser.add_argument(
        "--pre_8dom_json",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "results_vault_v2_4_step10k_8domain.json"),
    )
    parser.add_argument(
        "--post_8dom_json",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "results_vault_v2_5_8domain.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(root_dir, "benchmarks", "comparison_v2_4_step10k_vs_v2_5_sft.json"),
    )
    args = parser.parse_args()

    trad = _load(args.trad_json).get("models", {})
    t_pre = trad.get("VAULT_v2_4_Step10000_180M", {})
    t_post = trad.get("VAULT_v2_5_180M", {})
    s_pre = _load(args.pre_suite_json)
    s_post = _load(args.post_suite_json)
    d_pre = _load(args.pre_8dom_json)
    d_post = _load(args.post_8dom_json)

    comp: Dict[str, Any] = {
        "traditional_10task": {},
        "language_modeling": {},
        "decode_latency_ms_per_tok": {},
        "controlled_2k_gaps": {},
        "adversarial_long_sibling_niah": {},
        "extreme_128k_and_mitosis": {},
        "domain8_verbatim_side_by_side": [],
    }

    print("\n" + "=" * 112)
    print("HEAD-TO-HEAD COMPARISON: VAULT v2.4 @ Step 10,000 (Pre-SFT) vs VAULT v2.5 (Post-SFT)")
    print("=" * 112)

    # 1. Language Modeling & Traditional 10-Task Suite
    bpb_pre = s_pre.get("language_modeling", {}).get("holdout_english_bpb", {}).get("bpb")
    bpb_post = s_post.get("language_modeling", {}).get("holdout_english_bpb", {}).get("bpb")
    ppl_pre = s_pre.get("language_modeling", {}).get("unseen_edu_prose_2k", {}).get("ppl")
    ppl_post = s_post.get("language_modeling", {}).get("unseen_edu_prose_2k", {}).get("ppl")
    if bpb_pre is not None and bpb_post is not None:
        comp["language_modeling"]["holdout_bpb"] = {
            "v2_4_step10k": bpb_pre,
            "v2_5_sft": bpb_post,
            "delta": round(bpb_post - bpb_pre, 4),
        }
        print(f"Holdout English BPB      : {bpb_pre:.4f} -> {bpb_post:.4f} ({bpb_post - bpb_pre:+.4f})")
    if ppl_pre is not None and ppl_post is not None:
        comp["language_modeling"]["unseen_edu_prose_ppl"] = {
            "v2_4_step10k": ppl_pre,
            "v2_5_sft": ppl_post,
            "delta": round(ppl_post - ppl_pre, 2),
        }
        print(f"Unseen Edu Prose PPL     : {ppl_pre:.2f} -> {ppl_post:.2f} ({_fmt_delta(ppl_post, ppl_pre, True)})")

    if t_pre and t_post:
        wt_pre = t_pre["wikitext2"]["word_ppl"]
        wt_post = t_post["wikitext2"]["word_ppl"]
        lam_pre = t_pre["lambada"]["acc"]
        lam_post = t_post["lambada"]["acc"]
        comp["traditional_10task"]["wikitext2_word_ppl"] = {"v2_4_step10k": wt_pre, "v2_5_sft": wt_post, "delta": round(wt_post - wt_pre, 2)}
        comp["traditional_10task"]["lambada_acc"] = {"v2_4_step10k": lam_pre, "v2_5_sft": lam_post, "delta": round(lam_post - lam_pre, 2)}
        print(f"WikiText-2 Word PPL      : {wt_pre:.2f} -> {wt_post:.2f} ({_fmt_delta(wt_post, wt_pre, True)})")
        print(f"LAMBADA Exact Acc        : {lam_pre:.2f}% -> {lam_post:.2f}% ({_fmt_delta(lam_post, lam_pre, False, '%')})")

        mc_specs = [
            ("hellaswag", "acc_norm", "HellaSwag (acc_norm)"),
            ("piqa", "acc_norm", "PIQA (acc_norm)"),
            ("arc_easy", "acc_norm", "ARC-Easy (acc_norm)"),
            ("arc_challenge", "acc_norm", "ARC-Challenge (acc_norm)"),
            ("sciq_closedbook", "acc_norm", "SciQ Closed-Book"),
            ("sciq_rc", "acc_norm", "SciQ Open-Book (+Psg)"),
            ("winogrande", "acc", "Winogrande (acc)"),
            ("boolq", "acc", "BoolQ (acc)"),
        ]
        for k, metric_k, label in mc_specs:
            v0 = t_pre["multiple_choice"][k][metric_k]
            v1 = t_post["multiple_choice"][k][metric_k]
            comp["traditional_10task"][k] = {"metric": metric_k, "v2_4_step10k": v0, "v2_5_sft": v1, "delta": round(v1 - v0, 2)}
            print(f"{label:<25s}: {v0:6.2f}% -> {v1:6.2f}% ({_fmt_delta(v1, v0, False, '%')})")

        avg0 = t_pre.get("zero_shot_9task_avg_pct", 0.0)
        avg1 = t_post.get("zero_shot_9task_avg_pct", 0.0)
        comp["traditional_10task"]["zero_shot_9task_avg_pct"] = {"v2_4_step10k": avg0, "v2_5_sft": avg1, "delta": round(avg1 - avg0, 2)}
        print(f"{'9-Task Zero-Shot Avg':<25s}: {avg0:6.2f}% -> {avg1:6.2f}% ({_fmt_delta(avg1, avg0, False, '%')})")

    # 2. O(1) Decode Latency
    lat_pre = s_pre.get("gpu_kernel_parity_and_latency", {})
    lat_post = s_post.get("gpu_kernel_parity_and_latency", {})
    if lat_pre and lat_post:
        print("\n--- O(1) Streaming Decode Latency (ms/tok) ---")
        for bank_k in ("scaling_220c", "scaling_2002c"):
            comp["decode_latency_ms_per_tok"][bank_k] = {}
            for lk, r0 in lat_pre.get(bank_k, {}).items():
                r1 = lat_post.get(bank_k, {}).get(lk)
                if r1:
                    m0 = r0["decode_ms_per_tok"]
                    m1 = r1["decode_ms_per_tok"]
                    sp = round(m0 / max(m1, 1e-6), 2)
                    comp["decode_latency_ms_per_tok"][bank_k][lk] = {
                        "v2_4_step10k_ms": m0,
                        "v2_5_sft_ms": m1,
                        "speedup_x": sp,
                    }
                    print(f"  [{bank_k} @ {lk:>4s}] {m0:6.2f} ms/tok -> {m1:6.2f} ms/tok ({sp:.2f}x speedup)")

    # 3. Controlled 2K Gaps & Adversarial >=8-Subword Sibling NIAH
    c_pre = s_pre.get("controlled_2k_gaps", {})
    c_post = s_post.get("controlled_2k_gaps", {})
    if c_pre and c_post:
        print("\n--- Controlled 2K Gaps (15t / 3t / 24t) ---")
        for m_tag in ("oneshot_base", "chatml_vault"):
            comp["controlled_2k_gaps"][m_tag] = {}
            for gk in ("15tok_boilerplate", "3tok_compact", "24tok_wide"):
                a0 = c_pre.get(m_tag, {}).get(gk, {}).get("cue_exact_acc", 0.0) * 100.0
                a1 = c_post.get(m_tag, {}).get(gk, {}).get("cue_exact_acc", 0.0) * 100.0
                comp["controlled_2k_gaps"][m_tag][gk] = {"v2_4_step10k_pct": round(a0, 1), "v2_5_sft_pct": round(a1, 1), "delta_pct": round(a1 - a0, 1)}
                print(f"  [{m_tag:<12s} | {gk:<18s}] {a0:5.1f}% -> {a1:5.1f}% ({a1 - a0:+.1f}%)")

    sib_pre = s_pre.get("adversarial_long_sibling_niah", {}).get("primary", {})
    sib_post = s_post.get("adversarial_long_sibling_niah", {}).get("primary", {})
    if sib_pre and sib_post:
        print("\n--- Held-Out Adversarial >=8-Token Sibling Disentanglement (2K & 4K) ---")
        for m_tag in ("oneshot_base", "chatml_vault"):
            comp["adversarial_long_sibling_niah"][m_tag] = {}
            for hk in ("2K", "4K"):
                r0 = sib_pre.get(m_tag, {}).get(hk, {})
                r1 = sib_post.get(m_tag, {}).get(hk, {})
                a0 = r0.get("cue_exact_acc", 0.0) * 100.0
                a1 = r1.get("cue_exact_acc", 0.0) * 100.0
                comp["adversarial_long_sibling_niah"][m_tag][hk] = {
                    "v2_4_step10k_pct": round(a0, 1),
                    "v2_5_sft_pct": round(a1, 1),
                    "delta_pct": round(a1 - a0, 1),
                    "v2_4_nll_bits": r0.get("mean_code_nll_bits"),
                    "v2_5_nll_bits": r1.get("mean_code_nll_bits"),
                }
                print(
                    f"  [{m_tag:<12s} | {hk}] {a0:5.1f}% (NLL={r0.get('mean_code_nll_bits', 0):.2f}b) -> "
                    f"{a1:5.1f}% (NLL={r1.get('mean_code_nll_bits', 0):.2f}b) [{a1 - a0:+.1f}%]"
                )

    # 4. Extreme 2K -> 128K & Zero-Shot Mitosis
    ex_pre = s_pre.get("extreme_niah_2k_to_128k", {})
    ex_post = s_post.get("extreme_niah_2k_to_128k", {})
    if ex_pre and ex_post:
        print("\n--- Extreme 2K -> 128K Extrapolation & Zero-Shot Mitosis ---")
        for cfg_k, d0 in ex_pre.items():
            d1 = ex_post.get(cfg_k, {})
            comp["extreme_128k_and_mitosis"][cfg_k] = {}
            for lk, r0 in d0.items():
                r1 = d1.get(lk, {})
                if not r1:
                    continue
                a0 = r0.get("cue_exact_acc", 0.0) * 100.0
                a1 = r1.get("cue_exact_acc", 0.0) * 100.0
                n0 = r0.get("mean_code_nll_bits", 0.0)
                n1 = r1.get("mean_code_nll_bits", 0.0)
                comp["extreme_128k_and_mitosis"][cfg_k][lk] = {
                    "v2_4_step10k_pct": round(a0, 1),
                    "v2_5_sft_pct": round(a1, 1),
                    "delta_pct": round(a1 - a0, 1),
                    "v2_4_nll_bits": n0,
                    "v2_5_nll_bits": n1,
                }
                print(f"  [{cfg_k:<22s} | {lk:>4s}] {a0:5.1f}% ({n0:5.2f}b) -> {a1:5.1f}% ({n1:5.2f}b) [{a1 - a0:+.1f}%]")

    # 5. Side-by-Side 8-Domain Verbatim Decoded Text
    ev0 = d_pre.get("evaluations", [])
    ev1 = d_post.get("evaluations", [])
    if ev0 and ev1:
        print("\n--- 8-Domain Verbatim Decoded Outputs (Step 10k Pre-SFT vs v2.5 Post-SFT) ---")
        for x0, x1 in zip(ev0, ev1):
            dom = x1.get("domain", x0.get("domain", ""))
            comp["domain8_verbatim_side_by_side"].append(
                {
                    "domain": dom,
                    "prompt_tail": x1.get("prompt_tail", ""),
                    "v2_4_step10k_output": x0.get("output", ""),
                    "v2_5_sft_output": x1.get("output", ""),
                }
            )
            print(f"\n[{dom}]")
            print(f"  Pre-SFT (10k): {repr(x0.get('output', '')[:220])}")
            print(f"  Post-SFT(v25): {repr(x1.get('output', '')[:220])}")

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(comp, f, indent=2)
    os.replace(tmp, args.out)
    print(f"\n[+] Saved full head-to-head comparison report -> {args.out}")


if __name__ == "__main__":
    main()
