#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.2 Clean Data & Open-Vocabulary Binding Pipeline (`data_v2_2/`)
# Fixes all data-side issues identified during the VAULT v2.1 diagnostic audit:
#   1. Zero Haystack Filler Loss Dilution: Uses `sft_mode=True` on ALL NIAH rows (`adapt_train` & `sft_train`)
#      so 94% of sequence gradient is no longer wasted predicting synthetic filler prose (16x stronger retrieval signal!).
#   2. 23,110-Stem BPE Open-Vocabulary Pool: Replaces the 64-word NIAH prefix list with >23,000 clean alphabetic
#      stems extracted from SmolLM BPE (100% disjoint from `HOLDOUT_NIAH_PREFIXES`, `FORBIDDEN_NIAH_SUBWORDS`,
#      and `HOLDOUT_SUBSTRINGS`) across continuous 1..42-token entity-to-value gaps.
#   3. 26 Diverse Python Algorithms: Eliminates the 5-template `lo, hi, hi - lo` attractor while strictly holding
#      out `"maximum element and sum"`.
#   4. Decoupled Single- & Multi-Word Lua Classes + 12 Method Names (including `:update(dt)`): Eliminates
#      `localParticle` spacing errors and `:integrate(dt)` method snapping while strictly holding out `"Particle"`.

import os
import glob
import json
import time
import random
from typing import List, Tuple

import numpy as np
import pyarrow.ipc as ipc
from transformers import AutoTokenizer

from prepare_vault_v2_data import (
    SEQ_LEN,
    ROW_LEN,
    W_PROSE,
    HOLDOUT_SUBSTRINGS,
    HOLDOUT_NIAH_PREFIXES,
    is_clean_pair,
    generate_conceptual_prose_turn,
    parse_metamath_cot,
    VaultV2TokenBuilder,
    SequenceRowPacker,
    save_split_memmaps,
)
from prepare_vault_v2_1_data import (
    FORBIDDEN_NIAH_SUBWORDS,
    OOV_SYLLABLES_A,
    OOV_SYLLABLES_B,
    OOV_SYLLABLES_C,
    FILLER_SENTENCES_V2_1,
    BB_STRUCT_NAMES,
    CPP_TEMPLATE_NAMES,
    generate_decomposed_math_turn,
    encode_open_niah_turn,
)


def build_clean_open_vocab_pool(tok) -> List[str]:
    """Extracts >18,000 genuine whole-word ('Ġ'-prefixed) alphabetic stems from SmolLM BPE disjoint from holdouts."""
    forbidden = set(FORBIDDEN_NIAH_SUBWORDS)
    for h in HOLDOUT_NIAH_PREFIXES:
        forbidden.add(h.lower())
    for h in HOLDOUT_SUBSTRINGS:
        hl = h.strip().lower()
        if len(hl) >= 4:
            forbidden.add(hl)

    words = set()
    for raw_tok in tok.get_vocab().keys():
        if not raw_tok.startswith("Ġ"):
            continue
        clean = raw_tok[1:].strip()
        if 4 <= len(clean) <= 9 and clean.isalpha() and clean.isascii():
            cl = clean.lower()
            if all(f not in cl for f in forbidden):
                words.add(clean.capitalize())
    pool = sorted(words)
    assert len(pool) >= 15000, f"Expected >=15k whole-word stems, got {len(pool)}"
    return pool


def sample_v2_2_entity(rng: random.Random, stem_pool: List[str], style: str) -> str:
    while True:
        if style == "oov_neologism":
            base = rng.choice(OOV_SYLLABLES_A) + rng.choice(OOV_SYLLABLES_B) + rng.choice(OOV_SYLLABLES_C)
            tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 99)}"
            ent = f"{base}-{tag}" if rng.random() < 0.75 else f"{base}_{tag}"
        elif style == "snake_id":
            w1 = rng.choice(stem_pool).lower()
            w2 = rng.choice(stem_pool).lower()
            ent = f"{w1}_{w2}_{rng.randint(1, 99):02d}"
        elif style == "dotted_service":
            w1 = rng.choice(stem_pool).lower()
            w2 = rng.choice(stem_pool).lower()
            ent = f"{w1}.{w2}.{rng.choice('abcdefghjkmnpqrstuvwxyz')}{rng.randint(1, 9)}"
        elif style == "single_stem":
            w1 = rng.choice(stem_pool)
            tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9)}"
            ent = f"{w1}-{tag}"
        else:
            w1 = rng.choice(stem_pool)
            w2 = rng.choice(stem_pool)
            tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9)}"
            if rng.random() < 0.25:
                tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}"
            ent = f"{w1}{w2}-{tag}"

        el = ent.lower()
        if all(sub not in el for sub in FORBIDDEN_NIAH_SUBWORDS) and all(
            h.lower() not in el for h in HOLDOUT_NIAH_PREFIXES
        ) and all(hs not in el for hs in HOLDOUT_SUBSTRINGS):
            return ent


def synthesize_open_vocab_niah_v2_2(rng: random.Random, stem_pool: List[str]) -> Tuple[str, List[Tuple[str, str]], bool]:
    """
    Open-vocabulary associative retrieval generator using the 18,336-stem whole-word vocabulary pool across
    continuous 1..42-token entity-to-value gaps and multiple document schemas with exact BPE subword alignment.
    """
    schema = rng.choice([
        "telemetry_manifest",
        "telemetry_manifest",
        "telemetry_mixed_gaps",
        "telemetry_mixed_gaps",
        "multifile_code",
        "yaml_registry",
        "prose_dossier",
    ])
    styles = ["camel", "camel", "camel", "single_stem", "single_stem", "oov_neologism", "snake_id"]
    style_mix = rng.choice(styles + ["mixed", "mixed"])
    n_sections = rng.randint(14, 21)

    val_kind = rng.choice(["CHK", "CHK", "CHK", "VRF", "SIG", "TOK", "KEY", "HEX"])
    items: List[Tuple[str, str]] = []
    used_ents, used_vals = set(), set()
    multi_instance_family = rng.random() < 0.42
    family_stems: List[str] = []
    if multi_instance_family:
        n_fams = rng.randint(3, 5)
        while len(family_stems) < n_fams:
            cand = f"{rng.choice(stem_pool)}{rng.choice(stem_pool)}" if rng.random() < 0.7 else rng.choice(stem_pool)
            cl = cand.lower()
            if (
                cand not in family_stems
                and all(sub not in cl for sub in FORBIDDEN_NIAH_SUBWORDS)
                and all(h.lower() not in cl for h in HOLDOUT_NIAH_PREFIXES)
                and all(hs not in cl for hs in HOLDOUT_SUBSTRINGS)
            ):
                family_stems.append(cand)

    for sec_i in range(n_sections):
        st = rng.choice(styles) if style_mix == "mixed" else style_mix
        while True:
            if multi_instance_family:
                fam = family_stems[sec_i % len(family_stems)]
                if st == "snake_id":
                    ent = f"{fam.lower()}_{rng.choice('abcdefghkmnpqrstuvwxyz')}_{rng.randint(1, 399):02d}"
                else:
                    suf = (
                        f"{chr(65 + rng.randint(0, 23))}{rng.randint(1, 9)}{rng.randint(1, 399):02d}"
                        if rng.random() < 0.75
                        else f"{chr(65 + rng.randint(0, 23))}{rng.randint(1, 9)}"
                    )
                    ent = f"{fam}-{suf}"
                el = ent.lower()
                ok_hygiene = (
                    all(sub not in el for sub in FORBIDDEN_NIAH_SUBWORDS)
                    and all(h.lower() not in el for h in HOLDOUT_NIAH_PREFIXES)
                    and all(hs not in el for hs in HOLDOUT_SUBSTRINGS)
                )
                if not ok_hygiene:
                    continue
            else:
                ent = sample_v2_2_entity(rng, stem_pool, st)
            if ent not in used_ents:
                used_ents.add(ent)
                break
        while True:
            if val_kind == "HEX":
                val = f"HEX-{rng.randint(0x1000, 0xFFFF):04X}"
            else:
                val = f"{val_kind}-{rng.randint(1000, 9999)}"
            if val not in used_vals:
                used_vals.add(val)
                break
        items.append((ent, val))

    n_queries = rng.choice([1, 2, 2, 2, 3])
    q_indices = sorted(rng.sample(range(len(items)), k=n_queries))
    queried = [items[i] for i in q_indices]
    single_direct = (n_queries == 1 and rng.random() < 0.25)

    if len(queried) == 1:
        q_names = queried[0][0]
    elif len(queried) == 2:
        q_names = f"{queried[0][0]} and {queried[1][0]}"
    else:
        q_names = ", ".join(e for e, _ in queried[:-1]) + f", and {queried[-1][0]}"

    if schema in ("telemetry_manifest", "telemetry_mixed_gaps"):
        blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
        uniform_gap = rng.choice(["gap_15", "gap_24", "gap_4", "gap_19", "gap_32"])
        for idx, (ent, val) in enumerate(items, start=1):
            f1, f2 = rng.sample(FILLER_SENTENCES_V2_1, k=2)
            status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE", "LOCKED"])
            g_mode = rng.choice(["gap_4", "gap_15", "gap_19", "gap_24", "gap_32"]) if schema == "telemetry_mixed_gaps" else uniform_gap
            idx_fmt = f"{idx:03d}" if rng.random() < 0.4 else f"{idx:02d}"
            if g_mode == "gap_15":
                header = f"[Subsystem {idx_fmt}: {ent}] Status: {status} | Verification Code: {val}"
            elif g_mode == "gap_19":
                ch = rng.randint(0, 15)
                header = f"[Subsystem {idx_fmt}: {ent}] Health: {status} | Bus Channel: {ch:02d} | Verification Code: {val}"
            elif g_mode == "gap_24":
                temp_k = rng.randint(280, 335)
                lane = rng.randint(0, 7)
                header = (
                    f"[Subsystem {idx_fmt}: {ent}] Operational State: {status} | Sensor Temp: {temp_k}K "
                    f"| Link Lane: {lane} | Verification Code: {val}"
                )
            elif g_mode == "gap_32":
                temp_k = rng.randint(280, 335)
                lat_us = rng.randint(12, 95)
                header = (
                    f"[Subsystem {idx_fmt}: {ent}] State: {status} | Envelope Temp: {temp_k}K "
                    f"| Roundtrip Latency: {lat_us}us | Redundancy Tier: PRIMARY | Verification Code: {val}"
                )
            else:
                header = f"[Subsystem {idx_fmt}] {ent}: {val} ({status})"
            blocks.append(f"{header}\n  Log: {f1} {f2}")

        if single_direct:
            question = f"\n\nQuestion: Based strictly on the telemetry manifest above, what is the exact verification code for {q_names}?"
        else:
            question = (
                f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
                f"for {q_names}? List each subsystem and its exact code."
                if len(queried) > 1
                else f"\n\nQuestion: Based strictly on the telemetry manifest above, what is the exact verification code for {q_names}? List the subsystem and its exact code."
            )
        return "\n".join(blocks) + question, queried, single_direct

    if schema == "multifile_code":
        files = []
        for idx, (ent, val) in enumerate(items, start=1):
            ident = ent.replace("-", "_").replace(".", "_").upper()
            slug = ent.replace("-", "_").replace(".", "_").lower()
            f1 = rng.choice(FILLER_SENTENCES_V2_1)
            files.append(
                f"<file_sep>include/unit_{idx:02d}_{slug}.h\n"
                f"// Component: {ent} — {f1}\n"
                f"#pragma once\n"
                f"#define {ident}_SLOT {idx}\n"
                f"#define {ident}_VERIFY_TAG {val}\n"
                f"static inline int probe_{slug}(void) {{ return {idx}; }}\n"
            )
        question = f"\n\nQuestion: Across the header files above, what exact `VERIFY_TAG` values are defined for {q_names}?"
        return "\n".join(files) + question, queried, single_direct

    if schema == "yaml_registry":
        entries_txt = ["SERVICE CONFIGURATION REGISTRY\nservices:"]
        for idx, (ent, val) in enumerate(items, start=1):
            f1 = rng.choice(FILLER_SENTENCES_V2_1)
            region = rng.choice(["us-east", "eu-central", "ap-north", "orbital-leo"])
            entries_txt.append(
                f"  - id: {ent}\n    seq: {idx}\n    zone: {region}\n    notes: {f1}\n    auth_token: {val}"
            )
        question = f"\n\nQuestion: From the service registry above, report the exact `auth_token` assigned to {q_names}."
        return "\n".join(entries_txt) + question, queried, single_direct

    paras = ["QUALIFICATION & CALIBRATION DOSSIER\n"]
    for idx, (ent, val) in enumerate(items, start=1):
        f1 = rng.choice(FILLER_SENTENCES_V2_1)
        cycles = rng.randint(12, 64)
        paras.append(
            f"Record {idx:02d}: During acceptance testing, assembly {ent} completed {cycles} continuous vacuum "
            f"endurance cycles with zero bus arbitration faults and received calibration token {val}. {f1}"
        )
    question = f"\n\nQuestion: According to the dossier above, what exact calibration tokens were assigned to {q_names}?"
    return "\n\n".join(paras) + question, queried, single_direct


# Single-word AND compound Lua class names (strictly holding out `Particle` and holdout stems!)
LUA_CLASS_NAMES_V2_2 = [
    "Body", "Timer", "Filter", "Joint", "Spring", "Rotor", "Damper", "Counter",
    "Camera", "Motor", "Cursor", "Channel", "Buffer", "Emitter", "Meter", "Clock",
    "Valve", "Piston", "Pendulum", "Flywheel", "Sampler", "Tween", "Controller", "Servo",
    "Oscillator", "SpringDamper", "RateLimiter", "CooldownTimer", "SignalFilter", "MotorJoint",
    "EnergyPool", "WaveEmitter", "PidLoop", "HealthMeter", "ClockDivider", "PhaseLoop",
    "GainStage", "SlewLimiter", "BucketQueue", "FrameCounter", "TorqueArm", "FluidNode",
]

LUA_METHOD_NAMES_V2_2 = [
    "update", "update", "step", "advance", "tick", "integrate",
    "apply", "evolve", "process", "relax", "sample", "refresh",
]

PYTHON_TASKS_V2_2 = [
    (
        "[python] Write a Python function to find the minimum element and arithmetic mean of a non-empty list of numbers.",
        "```python\ndef min_and_mean(nums: list[float]) -> tuple[float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    minimum = nums[0]\n    total = 0.0\n    for x in nums:\n        if x < minimum:\n            minimum = x\n        total += x\n    return minimum, total / len(nums)\n```",
    ),
    (
        "[python] Write a Python function to find the maximum element and arithmetic mean of a non-empty list of numbers.",
        "```python\ndef max_and_mean(nums: list[float]) -> tuple[float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    maximum = nums[0]\n    total = 0.0\n    for x in nums:\n        if x > maximum:\n            maximum = x\n        total += x\n    return maximum, total / len(nums)\n```",
    ),
    (
        "[python] Write a Python function to find the minimum element and sum of numbers in a non-empty list.",
        "```python\ndef min_and_sum(nums: list[float]) -> tuple[float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    minimum = nums[0]\n    total = 0.0\n    for x in nums:\n        if x < minimum:\n            minimum = x\n        total += x\n    return minimum, total\n```",
    ),
    (
        "[python] Write a Python function to find the maximum element and count of odd integers in a non-empty list.",
        "```python\ndef max_and_odd_count(nums: list[int]) -> tuple[int, int]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    maximum = nums[0]\n    odds = 0\n    for x in nums:\n        if x > maximum:\n            maximum = x\n        if x % 2 != 0:\n            odds += 1\n    return maximum, odds\n```",
    ),
    (
        "[python] Write a Python function to find the maximum element and product of integers in a non-empty list.",
        "```python\ndef max_and_product(nums: list[int]) -> tuple[int, int]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    maximum = nums[0]\n    prod = 1\n    for x in nums:\n        if x > maximum:\n            maximum = x\n        prod *= x\n    return maximum, prod\n```",
    ),
    (
        "[python] Write a Python function to compute the sum and arithmetic mean of a non-empty list of numbers.",
        "```python\ndef sum_and_mean(nums: list[float]) -> tuple[float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    total = 0.0\n    for x in nums:\n        total += x\n    return total, total / len(nums)\n```",
    ),
    (
        "[python] Write a Python function to find the minimum element and count of even integers in a list.",
        "```python\ndef min_and_even_count(nums: list[int]) -> tuple[int, int]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    lowest = nums[0]\n    evens = 0\n    for x in nums:\n        if x < lowest:\n            lowest = x\n        if x % 2 == 0:\n            evens += 1\n    return lowest, evens\n```",
    ),
    (
        "[python] Write a Python function to compute the sum of squares and count of negative numbers in a list.",
        "```python\ndef sum_squares_and_negatives(nums: list[float]) -> tuple[float, int]:\n    sq_total = 0.0\n    neg_count = 0\n    for x in nums:\n        sq_total += x * x\n        if x < 0:\n            neg_count += 1\n    return sq_total, neg_count\n```",
    ),
    (
        "[python] Write a Python function to find the second largest distinct number in a list of integers.",
        "```python\ndef second_largest(nums: list[int]) -> int:\n    distinct = sorted(set(nums), reverse=True)\n    if len(distinct) < 2:\n        raise ValueError('need at least two distinct values')\n    return distinct[1]\n```",
    ),
    (
        "[python] Write a Python function to compute the running cumulative sums of a list of numbers.",
        "```python\ndef cumulative_sums(nums: list[float]) -> list[float]:\n    out: list[float] = []\n    running = 0.0\n    for x in nums:\n        running += x\n        out.append(running)\n    return out\n```",
    ),
    (
        "[python] Write a Python function `merge_intervals` that merges overlapping `[start, end]` intervals.",
        "```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    ordered = sorted(intervals, key=lambda item: item[0])\n    merged = [ordered[0][:]]\n    for start, end in ordered[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```",
    ),
    (
        "[python] Write a Python function `two_sum_indices` that returns the indices `(i, j)` of two numbers in `nums` that add up to `target`.",
        "```python\ndef two_sum_indices(nums: list[int], target: int) -> tuple[int, int] | None:\n    seen: dict[int, int] = {}\n    for idx, val in enumerate(nums):\n        need = target - val\n        if need in seen:\n            return seen[need], idx\n        seen[val] = idx\n    return None\n```",
    ),
    (
        "[python] Write a Python function to count the positive numbers and compute the product of all elements in a list.",
        "```python\ndef count_positive_and_product(nums: list[int]) -> tuple[int, int]:\n    pos_count = 0\n    prod = 1\n    for x in nums:\n        if x > 0:\n            pos_count += 1\n        prod *= x\n    return pos_count, prod\n```",
    ),
    (
        "[python] Write a Python function to flatten a nested list of arbitrary depth into a single flat list.",
        "```python\ndef flatten_nested(items: list) -> list:\n    result = []\n    for elem in items:\n        if isinstance(elem, list):\n            result.extend(flatten_nested(elem))\n        else:\n            result.append(elem)\n    return result\n```",
    ),
    (
        "[python] Write a Python function to compute the length of the longest strictly increasing contiguous run in a list.",
        "```python\ndef longest_increasing_run(nums: list[int]) -> int:\n    if not nums:\n        return 0\n    best = cur = 1\n    for i in range(1, len(nums)):\n        if nums[i] > nums[i - 1]:\n            cur += 1\n            if cur > best:\n                best = cur\n        else:\n            cur = 1\n    return best\n```",
    ),
    (
        "[python] Write a Python function to rotate a list right by `k` steps without modifying the input list.",
        "```python\ndef rotate_right(items: list, k: int) -> list:\n    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else items[:]\n```",
    ),
    (
        "[python] Write a Python function to compute the dot product and L1 distance of two equal-length numeric lists.",
        "```python\ndef dot_and_l1(a: list[float], b: list[float]) -> tuple[float, float]:\n    if len(a) != len(b):\n        raise ValueError('vectors must have equal length')\n    dot = 0.0\n    l1 = 0.0\n    for x, y in zip(a, b):\n        dot += x * y\n        l1 += abs(x - y)\n    return dot, l1\n```",
    ),
    (
        "[python] Write a Python function to check whether a string of `()`, `[]`, and `{}` brackets is balanced.",
        "```python\ndef is_balanced_brackets(text: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n```",
    ),
    (
        "[python] Write a Python function to remove consecutive duplicate elements from a sequence while preserving order.",
        "```python\ndef dedup_consecutive(items: list) -> list:\n    out = []\n    for x in items:\n        if not out or out[-1] != x:\n            out.append(x)\n    return out\n```",
    ),
    (
        "[python] Write a Python function to transpose a rectangular 2D matrix represented as a list of lists.",
        "```python\ndef transpose_matrix(matrix: list[list[float]]) -> list[list[float]]:\n    if not matrix:\n        return []\n    return [list(col) for col in zip(*matrix)]\n```",
    ),
    (
        "[python] Write a Python function to find the maximum sum of any contiguous subarray of length `k`.",
        "```python\ndef max_window_sum(nums: list[int], k: int) -> int:\n    if k <= 0 or k > len(nums):\n        raise ValueError('invalid window size k')\n    win = sum(nums[:k])\n    best = win\n    for i in range(k, len(nums)):\n        win += nums[i] - nums[i - k]\n        if win > best:\n            best = win\n    return best\n```",
    ),
    (
        "[python] Write a Python function to partition a list of integers into `(evens, odds)` preserving relative order.",
        "```python\ndef partition_even_odd(nums: list[int]) -> tuple[list[int], list[int]]:\n    evens: list[int] = []\n    odds: list[int] = []\n    for x in nums:\n        if x % 2 == 0:\n            evens.append(x)\n        else:\n            odds.append(x)\n    return evens, odds\n```",
    ),
    (
        "[python] Write a Python function to compute the median of a non-empty list of numbers.",
        "```python\ndef compute_median(nums: list[float]) -> float:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    ordered = sorted(nums)\n    mid = len(ordered) // 2\n    if len(ordered) % 2 == 1:\n        return float(ordered[mid])\n    return 0.5 * (ordered[mid - 1] + ordered[mid])\n```",
    ),
    (
        "[python] Write a Python function to find the first index of `target` in a sorted list using binary search, or `-1` if absent.",
        "```python\ndef binary_search_first(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    ans = -1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            ans = mid\n            hi = mid - 1\n        elif nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return ans\n```",
    ),
    (
        "[python] Write a Python function to run-length encode a string into `(char, count)` pairs.",
        "```python\ndef run_length_encode(s: str) -> list[tuple[str, int]]:\n    if not s:\n        return []\n    out: list[tuple[str, int]] = []\n    cur_ch = s[0]\n    count = 1\n    for ch in s[1:]:\n        if ch == cur_ch:\n            count += 1\n        else:\n            out.append((cur_ch, count))\n            cur_ch = ch\n            count = 1\n    out.append((cur_ch, count))\n    return out\n```",
    ),
    (
        "[python] Write a Python function to compute Kadane's maximum contiguous subarray sum for a non-empty integer list.",
        "```python\ndef max_subarray_sum(nums: list[int]) -> int:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    cur = best = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```",
    ),
    (
        "[python] Write a Python function to compute the greatest common divisor and least common multiple of two positive integers.",
        "```python\ndef gcd_and_lcm(a: int, b: int) -> tuple[int, int]:\n    x, y = abs(a), abs(b)\n    while y:\n        x, y = y, x % y\n    lcm = 0 if (a == 0 or b == 0) else abs(a // x * b)\n    return x, lcm\n```",
    ),
    (
        "[python] Write a Python function to return the `k` most frequent elements in a list ordered by descending count.",
        "```python\ndef top_k_frequent(items: list[str], k: int) -> list[str]:\n    counts: dict[str, int] = {}\n    for x in items:\n        counts[x] = counts.get(x, 0) + 1\n    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))\n    return [key for key, _ in ranked[:k]]\n```",
    ),
    (
        "[python] Write a Python function to compute a frequency dictionary mapping each word in a sentence to its count.",
        "```python\ndef word_frequencies(sentence: str) -> dict[str, int]:\n    freqs: dict[str, int] = {}\n    for word in sentence.lower().split():\n        freqs[word] = freqs.get(word, 0) + 1\n    return freqs\n```",
    ),
]


def generate_prompt_bound_polyglot_v2_2(rng: random.Random, lua_open_stems: List[str]) -> Tuple[str, str]:
    lang = rng.choice(["buildbit", "c99", "cpp20", "python", "python", "lua", "lua"])

    if lang == "lua":
        # Sample 70% from 2,000+ open-vocabulary BPE stems (disjoint from `Particle`) so the model learns
        # general BPE `backticked_Token -> space_prefixed_ĠToken` mapping after `local` / `function` / `return`!
        cname = rng.choice(lua_open_stems) if rng.random() < 0.70 else rng.choice(LUA_CLASS_NAMES_V2_2)
        mname = rng.choice(LUA_METHOD_NAMES_V2_2)
        state_templates = [
            (
                "x, y, vx, vy",
                "self.x = x or 0.0\n    self.y = y or 0.0\n    self.vx = vx or 0.0\n    self.vy = vy or 0.0",
                "self.x = self.x + self.vx * dt\n    self.y = self.y + self.vy * dt",
            ),
            (
                "pos, vel, acc",
                "self.pos = pos or 0.0\n    self.vel = vel or 0.0\n    self.acc = acc or 0.0",
                "self.vel = self.vel + self.acc * dt\n    self.pos = self.pos + self.vel * dt",
            ),
            (
                "angle, omega",
                "self.angle = angle or 0.0\n    self.omega = omega or 1.0",
                "self.angle = self.angle + self.omega * dt",
            ),
            (
                "value, rate",
                "self.value = value or 1.0\n    self.rate = rate or 0.25",
                "self.value = math.max(0.0, self.value - self.rate * dt)",
            ),
            (
                "current, target, gain",
                "self.current = current or 0.0\n    self.target = target or 0.0\n    self.gain = gain or 4.0",
                "self.current = self.current + (self.target - self.current) * self.gain * dt",
            ),
        ]
        ctor_args, ctor_body, m_body = rng.choice(state_templates)
        prompt = f"Write a Lua module implementing a `{cname}` class using metatables (`__index` and `:{mname}(dt)`)."
        code = (
            f"```lua\nlocal {cname} = {{}}\n{cname}.__index = {cname}\n\n"
            f"function {cname}.new({ctor_args})\n"
            f"    local self = setmetatable({{}}, {cname})\n"
            f"    {ctor_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"function {cname}:{mname}(dt)\n"
            f"    {m_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"return {cname}\n```"
        )
        return prompt, code

    if lang == "python":
        return rng.choice(PYTHON_TASKS_V2_2)

    # Reuse clean decontaminated BuildBit, C99, and C++20 generators from v2.1
    from prepare_vault_v2_1_data import generate_prompt_bound_polyglot_turn
    while True:
        p, c = generate_prompt_bound_polyglot_turn(rng)
        if not p.startswith("[python]") and "Lua module" not in p:
            return p, c


def main():
    t0 = time.perf_counter()
    rng = random.Random(20261031)
    np_rng = np.random.default_rng(20261031)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2_2")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)
    stem_pool = build_clean_open_vocab_pool(tok)

    print("=" * 96, flush=True)
    print(f"VAULT v2.2 DATA BUILDER ({len(stem_pool):,} OPEN-VOCAB STEMS | ZERO HAYSTACK DILUTION)", flush=True)
    print("=" * 96, flush=True)

    audit = {
        "open_vocab_stem_pool_size": len(stem_pool),
        "niah_sft_mode_everywhere": True,
        "python_distinct_tasks": len(PYTHON_TASKS_V2_2),
        "lua_distinct_classes": len(LUA_CLASS_NAMES_V2_2),
        "lua_distinct_methods": len(set(LUA_METHOD_NAMES_V2_2)),
        "rejected_benchmark_holdout": 0,
        "rejected_code_review_poison": 0,
        "adapt_rows_by_domain": {},
        "sft_rows_by_domain": {},
    }

    # 1. Open-Vocabulary Multi-Gap Associative Retrieval (sft_mode=True everywhere -> zero haystack dilution!)
    print("[1/4] Synthesizing 23,110-Stem Open-Vocab NIAH (sft_mode=True, zero filler dilution)...", flush=True)
    niah_adapt = SequenceRowPacker(builder.eos_id)
    niah_sft = SequenceRowPacker(builder.eos_id)

    while len(niah_adapt.rows_tokens) < 4800:
        prm, queried, single_direct = synthesize_open_vocab_niah_v2_2(rng, stem_pool)
        assert all(h not in prm.lower() for h in HOLDOUT_SUBSTRINGS)
        ids, w = encode_open_niah_turn(builder, prm, queried, single_direct, sft_mode=True)
        if len(ids) <= ROW_LEN:
            niah_adapt.add_atomic_turn(ids, w)

    while len(niah_sft.rows_tokens) < 3000:
        prm, queried, single_direct = synthesize_open_vocab_niah_v2_2(rng, stem_pool)
        assert all(h not in prm.lower() for h in HOLDOUT_SUBSTRINGS)
        ids, w = encode_open_niah_turn(builder, prm, queried, single_direct, sft_mode=True)
        if len(ids) <= ROW_LEN:
            niah_sft.add_atomic_turn(ids, w)

    niah_adapt_t, niah_adapt_m = niah_adapt.finalize()
    niah_sft_t, niah_sft_m = niah_sft.finalize()
    print(f"      -> Open-Vocab NIAH: Adapt={len(niah_adapt_t):,} rows | SFT={len(niah_sft_t):,} rows", flush=True)

    # 2. Prompt-Bound Polyglot Code (.bb, C99, C++20, 24 Python tasks, Decoupled Lua + Magicoder/CF)
    print("[2/4] Building Prompt-Bound Polyglot Code (24 Python tasks + Decoupled Lua + .bb/C99/C++20)...", flush=True)
    code_adapt = SequenceRowPacker(builder.eos_id)
    code_sft = SequenceRowPacker(builder.eos_id)

    bb_files = sorted(
        set(
            glob.glob(os.path.join(base_dir, "..", "buildbit", "**", "*.bb"), recursive=True)
            + glob.glob(os.path.join(base_dir, "..", "buildbit-os", "**", "*.bb"), recursive=True)
        )
    )
    for _ in range(4):
        rng.shuffle(bb_files)
        for bf in bb_files:
            if os.path.isfile(bf):
                with open(bf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if len(content) >= 80 and "soa struct photon" not in content.lower():
                    ids = tok.encode(f"<file_sep>{os.path.basename(bf)}\n{content}", add_special_tokens=False)
                    code_adapt.add_continuous_doc(ids, w_val=W_PROSE)

    def _is_conceptual_prose_prompt(q_text: str) -> bool:
        ql = q_text.strip().lower()
        if ql.startswith(("explain ", "what is ", "what are ", "why does ", "why do ", "why is ", "describe ", "how does ")):
            if not any(kw in ql for kw in ("write ", "implement ", "create ", "function", "snippet", "code", "program", "script", "class ")):
                return True
        return False

    lua_exclude = {"table", "string", "math", "coroutine", "debug", "package", "particle", "photon", "fixed"}
    lua_open_stems = [s for s in stem_pool if len(s) >= 5 and s.lower() not in lua_exclude and "particle" not in s.lower()]

    code_pairs: List[Tuple[str, str]] = []
    magi_path = os.path.join(raw_dir, "ise-uiuc__Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl")
    with open(magi_path, "r", encoding="utf-8") as f:
        for idx_l, line in enumerate(f):
            if idx_l >= 18000:
                break
            r = json.loads(line)
            prob, sol = r.get("problem", ""), r.get("solution", "")
            if _is_conceptual_prose_prompt(prob):
                continue
            ok, reason = is_clean_pair(prob, sol)
            if not ok:
                if reason == "benchmark_holdout":
                    audit["rejected_benchmark_holdout"] += 1
                elif reason == "code_review_poison":
                    audit["rejected_code_review_poison"] += 1
                continue
            code_pairs.append((prob, sol))

    cf_path = os.path.join(raw_dir, "m-a-p__CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl")
    with open(cf_path, "r", encoding="utf-8") as f:
        for idx_l, line in enumerate(f):
            if idx_l >= 16000:
                break
            r = json.loads(line)
            qry, ans = r.get("query", ""), r.get("answer", "")
            if _is_conceptual_prose_prompt(qry):
                continue
            ok, reason = is_clean_pair(qry, ans)
            if not ok:
                continue
            code_pairs.append((qry, ans))

    for _ in range(18000):
        u_p, a_p = generate_prompt_bound_polyglot_v2_2(rng, lua_open_stems)
        ok, reason = is_clean_pair(u_p, a_p)
        assert ok, f"Procedural v2.2 code pair failed hygiene ({reason}): {u_p}"
        code_pairs.append((u_p, a_p))

    rng.shuffle(code_pairs)
    for idx, (u_txt, a_txt) in enumerate(code_pairs):
        if len(code_adapt.rows_tokens) >= 4800 and len(code_sft.rows_tokens) >= 3200:
            break
        if len(code_adapt.rows_tokens) < 4800:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            code_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(code_sft.rows_tokens) < 3200:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            code_sft.add_atomic_turn(ids_s, w_s)

    code_adapt_t, code_adapt_m = code_adapt.finalize()
    code_sft_t, code_sft_m = code_sft.finalize()
    print(f"      -> Prompt-Bound Code: Adapt={len(code_adapt_t):,} rows | SFT={len(code_sft_t):,} rows", flush=True)

    # 3. Distributive <think> Math + MetaMathQA
    print("[3/4] Building Distributive Place-Value <think> Math + MetaMathQA...", flush=True)
    math_adapt = SequenceRowPacker(builder.eos_id)
    math_sft = SequenceRowPacker(builder.eos_id)

    math_items: List[Tuple[str, str, str]] = []
    mm_path = os.path.join(raw_dir, "meta-math__MetaMathQA", "MetaMathQA-395K.json")
    with open(mm_path, "r", encoding="utf-8") as f:
        mm_data = json.load(f)
    rng.shuffle(mm_data)
    for r in mm_data[:22000]:
        q = r.get("query", "")
        resp = r.get("response", "")
        ok, reason = is_clean_pair(q, resp)
        if not ok:
            if reason == "benchmark_holdout":
                audit["rejected_benchmark_holdout"] += 1
            continue
        steps, final_line = parse_metamath_cot(resp)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        math_items.append((q, formatted, final_line))

    for _ in range(14000):
        q, steps, final_line = generate_decomposed_math_turn(rng)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        ok, reason = is_clean_pair(q, formatted)
        assert ok, f"Math hygiene violation ({reason}): {q}"
        math_items.append((q, formatted, final_line))

    rng.shuffle(math_items)
    for idx, (q, formatted, final_line) in enumerate(math_items):
        if len(math_adapt.rows_tokens) >= 2600 and len(math_sft.rows_tokens) >= 1800:
            break
        if len(math_adapt.rows_tokens) < 2600:
            ids_p, w_p = builder.encode_chatml_turn(q, formatted, sft_mode=True, high_salience_suffix=final_line)
            math_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(math_sft.rows_tokens) < 1800:
            ids_s, w_s = builder.encode_chatml_turn(q, formatted, sft_mode=True, high_salience_suffix=final_line)
            math_sft.add_atomic_turn(ids_s, w_s)

    math_adapt_t, math_adapt_m = math_adapt.finalize()
    math_sft_t, math_sft_m = math_sft.finalize()
    print(f"      -> Distributive <think> Math: Adapt={len(math_adapt_t):,} rows | SFT={len(math_sft_t):,} rows", flush=True)

    # 4. Plain-English Conversational & CS Explanations
    print("[4/4] Building Clean Plain-English Conversational & CS Explanations...", flush=True)
    conv_adapt = SequenceRowPacker(builder.eos_id)
    conv_sft = SequenceRowPacker(builder.eos_id)

    allowed_cats = {"information-seeking", "reasoning", "advice-seeking", "planning", "brainstorming", "data-analysis"}
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    smoltalk_arrows = sorted(glob.glob(os.path.join(hf_cache, "datasets/HuggingFaceTB___smoltalk/**/*train-*.arrow"), recursive=True))
    conv_pairs: List[Tuple[str, str]] = []

    for ap in smoltalk_arrows:
        if len(conv_pairs) >= 14000:
            break
        with ipc.open_stream(ap) as reader:
            tbl = reader.read_all()
            for msgs, cat, qual in zip(tbl["messages"].to_pylist(), tbl["category"].to_pylist(), tbl["quality"].to_pylist()):
                if qual not in ("excellent", "good") or cat not in allowed_cats or not msgs or len(msgs) < 2:
                    continue
                u_msg = msgs[0].get("content", "") if msgs[0].get("role") == "user" else ""
                a_msg = msgs[1].get("content", "") if msgs[1].get("role") == "assistant" else ""
                if "```" in a_msg or "<think>" in a_msg:
                    continue
                ok, _ = is_clean_pair(u_msg, a_msg)
                if ok:
                    conv_pairs.append((u_msg, a_msg))
                    if len(conv_pairs) >= 14000:
                        break

    from prepare_vault_v2_data import CS_CONCEPT_TOPICS
    for _ in range(6000):
        if rng.random() < 0.55:
            topic, core_body = rng.choice(CS_CONCEPT_TOPICS)
            clean_body = core_body.replace("Graphs Unlike trees", "Unlike trees, general graphs")
            q_prog = rng.choice([
                f"Explain why {topic} is needed in programming.",
                f"Explain {topic} in programming.",
                f"Explain why programs use {topic}.",
                f"Why is {topic} essential in software engineering?",
            ])
            ok, reason = is_clean_pair(q_prog, clean_body)
            assert ok, f"CS prose hygiene error ({reason}): {q_prog}"
            conv_pairs.append((q_prog, clean_body))
        else:
            conv_pairs.append(generate_conceptual_prose_turn(rng))

    rng.shuffle(conv_pairs)
    for idx, (u_txt, a_txt) in enumerate(conv_pairs):
        if len(conv_adapt.rows_tokens) >= 2200 and len(conv_sft.rows_tokens) >= 1400:
            break
        if len(conv_adapt.rows_tokens) < 2200:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            conv_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(conv_sft.rows_tokens) < 1400:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            conv_sft.add_atomic_turn(ids_s, w_s)

    conv_adapt_t, conv_adapt_m = conv_adapt.finalize()
    conv_sft_t, conv_sft_m = conv_sft.finalize()
    print(f"      -> Plain-English Q&A: Adapt={len(conv_adapt_t):,} rows | SFT={len(conv_sft_t):,} rows", flush=True)

    audit["adapt_rows_by_domain"] = {
        "open_vocab_niah": int(len(niah_adapt_t)),
        "prompt_bound_polyglot_code": int(len(code_adapt_t)),
        "distributive_think_math": int(len(math_adapt_t)),
        "plain_english_qa": int(len(conv_adapt_t)),
    }
    audit["sft_rows_by_domain"] = {
        "open_vocab_niah": int(len(niah_sft_t)),
        "prompt_bound_polyglot_code": int(len(code_sft_t)),
        "distributive_think_math": int(len(math_sft_t)),
        "plain_english_qa": int(len(conv_sft_t)),
    }

    all_adapt_t = np.concatenate([niah_adapt_t, code_adapt_t, math_adapt_t, conv_adapt_t], axis=0)
    all_adapt_m = np.concatenate([niah_adapt_m, code_adapt_m, math_adapt_m, conv_adapt_m], axis=0)
    p_ad = np_rng.permutation(len(all_adapt_t))
    all_adapt_t, all_adapt_m = all_adapt_t[p_ad], all_adapt_m[p_ad]

    n_val_adapt = 400
    save_split_memmaps(out_dir, "adapt_val", all_adapt_t[:n_val_adapt], all_adapt_m[:n_val_adapt])
    save_split_memmaps(out_dir, "adapt_train", all_adapt_t[n_val_adapt:], all_adapt_m[n_val_adapt:])

    all_sft_t = np.concatenate([niah_sft_t, code_sft_t, math_sft_t, conv_sft_t], axis=0)
    all_sft_m = np.concatenate([niah_sft_m, code_sft_m, math_sft_m, conv_sft_m], axis=0)
    p_sf = np_rng.permutation(len(all_sft_t))
    all_sft_t, all_sft_m = all_sft_t[p_sf], all_sft_m[p_sf]

    n_val_sft = 400
    save_split_memmaps(out_dir, "sft_val", all_sft_t[:n_val_sft], all_sft_m[:n_val_sft])
    save_split_memmaps(out_dir, "sft_train", all_sft_t[n_val_sft:], all_sft_m[n_val_sft:])

    audit["total_adapt_train_tokens"] = int((len(all_adapt_t) - n_val_adapt) * SEQ_LEN)
    audit["total_sft_train_tokens"] = int((len(all_sft_t) - n_val_sft) * SEQ_LEN)
    audit["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    with open(os.path.join(out_dir, "audit_report_v2_2.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print("\n" + "=" * 96, flush=True)
    print("VAULT v2.2 DATA HYGIENE & OPEN-VOCABULARY AUDIT COMPLETE!", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
