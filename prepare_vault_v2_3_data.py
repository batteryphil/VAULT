#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.3 Continual Pretraining + High-Density Associative Retrieval Corpus Builder (`data_v2_3/`)
#   1. Stream 1 (`prose_train` / `prose_val`): ~92.2M completely unseen raw plain-text educational & textbook tokens
#      from row-groups `rg_idx >= 100` of `Cosmopedia-v2` + `FineWeb-Edu-Dedup` (Stage-1 only read `rg_idx < 70`),
#      strictly decontaminated against all 8-domain holdouts, NIAH prefixes, and BPB evaluation passages.
#   2. Stream 2 (`niah_train` / `niah_val`): High-density open-vocabulary associative retrieval (`14..38` sections
#      within `L = 2048`, `3..32`-token gaps) across 18,336 disjoint BPE stems in both ChatML (`75%`) and
#      plain-text 1-shot / QA (`25%`) formats, with zero haystack filler loss dilution.

import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_vault_v2_data import (
    HOLDOUT_NIAH_PREFIXES,
    HOLDOUT_SUBSTRINGS,
    ROW_LEN,
    SEQ_LEN,
    W_ASSIST,
    W_IGNORE,
    W_NEEDLE,
    W_PROSE,
    SequenceRowPacker,
    VaultV2TokenBuilder,
    save_split_memmaps,
)
from prepare_vault_v2_1_data import (
    FILLER_SENTENCES_V2_1,
    FORBIDDEN_NIAH_SUBWORDS,
    encode_open_niah_turn,
)
from prepare_vault_v2_2_data import (
    build_clean_open_vocab_pool,
    sample_v2_2_entity,
    synthesize_open_vocab_niah_v2_2,
)
from train_vault_v2 import UNSEEN_FILLER, UNSEEN_NIAH_PREFIXES
from benchmarks.bench_vault_vs_mamba123_transformer import EVAL_BPB_PASSAGES, REPLAY_PROSE_PASSAGES


def build_decontamination_signatures() -> List[str]:
    sigs = set()
    for s in HOLDOUT_SUBSTRINGS:
        sl = s.strip().lower()
        if len(sl) >= 5:
            sigs.add(sl)
    for p in list(HOLDOUT_NIAH_PREFIXES) + list(UNSEEN_NIAH_PREFIXES):
        sigs.add(p.strip().lower() + "-")
    sigs.add(UNSEEN_FILLER[:40].lower())
    for passage in list(EVAL_BPB_PASSAGES) + list(REPLAY_PROSE_PASSAGES):
        pl = passage.strip().lower()
        sigs.add(pl[:48])
        sigs.add(pl[-48:])
    return sorted(sigs)


def synthesize_dense_niah_v2_3(
    rng: random.Random, stem_pool: List[str]
) -> Tuple[str, List[Tuple[str, str]], Tuple[str, str], bool]:
    """
    High-density (`22..38` sections in `<= 1,920` tokens) multi-gap open-vocabulary manifest generator
    teaching the 220-cell Co-Prime Voronoi router sharp multi-entity separation under dense crowding.
    Returns `(prompt_body_plus_question, queried_pairs, first_example_pair, single_direct)`.
    """
    n_sections = rng.randint(22, 36)
    styles = ["camel", "camel", "single_stem", "oov_neologism", "snake_id"]
    style_mix = rng.choice(styles + ["mixed", "mixed"])
    val_kind = rng.choice(["CHK", "CHK", "CHK", "CHK", "VRF", "SIG", "TOK", "HEX"])

    items: List[Tuple[str, str]] = []
    used_ents, used_vals = set(), set()
    multi_fam = rng.random() < 0.50
    fams: List[str] = []
    if multi_fam:
        n_f = rng.randint(4, 7)
        while len(fams) < n_f:
            cand = rng.choice(stem_pool) if rng.random() < 0.55 else f"{rng.choice(stem_pool)}{rng.choice(stem_pool)}"
            cl = cand.lower()
            if (
                cand not in fams
                and all(sub not in cl for sub in FORBIDDEN_NIAH_SUBWORDS)
                and all(h.lower() not in cl for h in HOLDOUT_NIAH_PREFIXES)
                and all(u.lower() not in cl for u in UNSEEN_NIAH_PREFIXES)
                and all(hs not in cl for hs in HOLDOUT_SUBSTRINGS)
            ):
                fams.append(cand)

    for sec_i in range(n_sections):
        st = rng.choice(styles) if style_mix == "mixed" else style_mix
        while True:
            if multi_fam:
                fam = fams[sec_i % len(fams)]
                suf = f"{chr(65 + (sec_i % 24))}{rng.randint(1, 9)}{sec_i + 1:02d}"
                ent = f"{fam}-{suf}" if st != "snake_id" else f"{fam.lower()}_{chr(97 + (sec_i % 20))}_{sec_i + 1:02d}"
            else:
                ent = sample_v2_2_entity(rng, stem_pool, st)
            el = ent.lower()
            if (
                ent not in used_ents
                and all(u.lower() not in el for u in UNSEEN_NIAH_PREFIXES)
                and all(h.lower() not in el for h in HOLDOUT_NIAH_PREFIXES)
            ):
                used_ents.add(ent)
                break
        while True:
            val = (
                f"HEX-{rng.randint(0x1000, 0xFFFF):04X}"
                if val_kind == "HEX"
                else f"{val_kind}-{rng.randint(1000, 9999)}"
            )
            if val not in used_vals:
                used_vals.add(val)
                break
        items.append((ent, val))

    # Reserve items[0] as optional non-overlapping 1-shot example; sample queries strictly from items[1:]
    n_q = rng.choice([2, 2, 2, 3])
    q_indices = sorted(rng.sample(range(1, len(items)), k=n_q))
    queried = [items[i] for i in q_indices]
    if len(queried) == 2:
        q_names = f"{queried[0][0]} and {queried[1][0]}"
    else:
        q_names = ", ".join(e for e, _ in queried[:-1]) + f", and {queried[-1][0]}"

    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    uniform_gap = rng.choice(["gap_15", "gap_15", "gap_3", "gap_24", "mixed"])
    for idx, (ent, val) in enumerate(items, start=1):
        status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE"])
        gm = rng.choice(["gap_3", "gap_15", "gap_19", "gap_24"]) if uniform_gap == "mixed" else uniform_gap
        if gm == "gap_3":
            hdr = f"[Subsystem {idx:03d}] {ent}: {val} ({status})"
        elif gm == "gap_19":
            hdr = f"[Subsystem {idx:03d}: {ent}] Health: {status} | Bus Channel: {idx % 16:02d} | Verification Code: {val}"
        elif gm == "gap_24":
            hdr = (
                f"[Subsystem {idx:03d}: {ent}] Operational State: {status} | Sensor Temp: {285 + (idx % 40)}K "
                f"| Link Lane: {idx % 8} | Verification Code: {val}"
            )
        else:
            hdr = f"[Subsystem {idx:03d}: {ent}] Status: {status} | Verification Code: {val}"
        f1 = rng.choice(FILLER_SENTENCES_V2_1)
        blocks.append(f"{hdr}\n  Log: {f1}")

    question = (
        f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
        f"for {q_names}? List each subsystem and its exact code."
    )
    return "\n".join(blocks) + question, queried, items[0], False


def encode_plaintext_niah_row(
    tok,
    eos_id: int,
    prompt_body_q: str,
    queried: List[Tuple[str, str]],
    ex0: Tuple[str, str],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """
    Encodes a plain-text (non-ChatML) NIAH manifest in either 1-shot summary completion (`Verification Summary:`)
    or direct `Answer:` completion, supervising strictly the retrieved answer tokens (`W_IGNORE` on haystack).
    """
    use_oneshot = rng.random() < 0.65
    if use_oneshot:
        prefix = f"{prompt_body_q}\nVerification Summary:\n- {ex0[0]}: {ex0[1]}\n"
    else:
        prefix = f"{prompt_body_q}\nAnswer:\n"

    ids = list(tok.encode(prefix, add_special_tokens=False))
    w = [W_IGNORE] * len(ids)

    for idx_q, (ent, val) in enumerate(queried):
        lead = f"- {ent}: "
        lead_ids = tok.encode(lead, add_special_tokens=False)
        ids.extend(lead_ids)
        w.extend([W_ASSIST] * len(lead_ids))

        v_str = val + ("\n" if idx_q + 1 < len(queried) else "")
        v_ids = tok.encode(v_str, add_special_tokens=False)
        ids.extend(v_ids)
        w.extend([W_NEEDLE] * len(v_ids))

    ids.append(eos_id)
    w.append(W_ASSIST)
    return ids, w


def main():
    t0 = time.perf_counter()
    rng = random.Random(20261101)
    np_rng = np.random.default_rng(20261101)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2_3")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)
    stem_pool = build_clean_open_vocab_pool(tok)
    bad_sigs = build_decontamination_signatures()

    print("=" * 98, flush=True)
    print(f"VAULT v2.3 UNSEEN PROSE + DENSE OPEN-VOCAB NIAH BUILDER (L = {SEQ_LEN}, stems = {len(stem_pool):,})", flush=True)
    print("=" * 98, flush=True)

    # --------------------------------------------------------------------------
    # 1. UNSEEN RAW EDUCATIONAL PROSE (`rg_idx >= 100` in Cosmopedia-v2 + FineWeb-Edu)
    # --------------------------------------------------------------------------
    prose_packer = SequenceRowPacker(builder.eos_id)
    rejected_holdout = 0

    cosmo_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "cosmopedia-v2", "train-00000-of-00104.parquet")
    pf_cosmo = pq.ParquetFile(cosmo_path)
    cosmo_target_rows = 25000  # ~51.2M unseen textbook tokens
    cosmo_last_rg = 100
    for rg_idx in range(100, pf_cosmo.num_row_groups):
        if len(prose_packer.rows_tokens) >= cosmo_target_rows:
            break
        cosmo_last_rg = rg_idx
        batch_texts = pf_cosmo.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()
        clean_texts = []
        for tx in batch_texts:
            if not tx or len(tx) < 220:
                continue
            tl = tx.lower()
            if any(sig in tl for sig in bad_sigs):
                rejected_holdout += 1
                continue
            clean_texts.append(tx)
        for i in range(0, len(clean_texts), 1024):
            if len(prose_packer.rows_tokens) >= cosmo_target_rows:
                break
            enc = tok(clean_texts[i : i + 1024], add_special_tokens=False)["input_ids"]
            for ids in enc:
                if len(ids) >= 64:
                    prose_packer.add_continuous_doc(ids, w_val=W_PROSE)

    cosmo_rows = len(prose_packer.rows_tokens)
    print(
        f"[1/2] Unseen Cosmopedia-v2 (rg 100..{cosmo_last_rg}): {cosmo_rows:,} rows ({cosmo_rows * SEQ_LEN / 1e6:.2f}M tok)",
        flush=True,
    )

    fw_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "fineweb-edu-dedup", "train-00000-of-00234.parquet")
    pf_fw = pq.ParquetFile(fw_path)
    total_prose_target = cosmo_rows + 20500  # + ~42.0M unseen FineWeb-Edu tokens = ~93.2M total
    fw_last_rg = 100
    for rg_idx in range(100, pf_fw.num_row_groups):
        if len(prose_packer.rows_tokens) >= total_prose_target:
            break
        fw_last_rg = rg_idx
        batch_texts = pf_fw.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()
        clean_texts = []
        for tx in batch_texts:
            if not tx or len(tx) < 250:
                continue
            tl = tx.lower()
            if any(sig in tl for sig in bad_sigs):
                rejected_holdout += 1
                continue
            clean_texts.append(tx)
        for i in range(0, len(clean_texts), 1024):
            if len(prose_packer.rows_tokens) >= total_prose_target:
                break
            enc = tok(clean_texts[i : i + 1024], add_special_tokens=False)["input_ids"]
            for ids in enc:
                if len(ids) >= 80:
                    prose_packer.add_continuous_doc(ids, w_val=W_PROSE)

    prose_toks, prose_msks = prose_packer.finalize()
    perm_p = np_rng.permutation(len(prose_toks))
    prose_toks = prose_toks[perm_p]
    prose_msks = prose_msks[perm_p]

    n_prose_val = 500
    save_split_memmaps(out_dir, "prose_val", prose_toks[:n_prose_val], prose_msks[:n_prose_val])
    save_split_memmaps(out_dir, "prose_train", prose_toks[n_prose_val:], prose_msks[n_prose_val:])

    # --------------------------------------------------------------------------
    # 2. HIGH-DENSITY DUAL-FORMAT OPEN-VOCAB NIAH (`niah_train` / `niah_val`)
    # --------------------------------------------------------------------------
    print("[2/2] Building High-Density ChatML (75%) + Plain-Text 1-Shot/QA (25%) Open-Vocab NIAH...", flush=True)
    niah_rows_t: List[np.ndarray] = []
    niah_rows_m: List[np.ndarray] = []
    n_niah_total = 4400
    chatml_cnt = 0
    plain_cnt = 0

    while len(niah_rows_t) < n_niah_total:
        use_chatml = rng.random() < 0.75
        use_dense = rng.random() < 0.45

        if use_dense or not use_chatml:
            prm, queried, ex0, s_dir = synthesize_dense_niah_v2_3(rng, stem_pool)
        else:
            prm, queried, s_dir = synthesize_open_vocab_niah_v2_2(rng, stem_pool)
            ex0 = queried[0]

        pl = prm.lower()
        if any(sig in pl for sig in bad_sigs):
            rejected_holdout += 1
            continue

        if use_chatml:
            ids, w = encode_open_niah_turn(builder, prm, queried, single_direct=s_dir, sft_mode=True)
        else:
            ids, w = encode_plaintext_niah_row(tok, builder.eos_id, prm, queried, ex0, rng)

        if 384 <= len(ids) <= ROW_LEN:
            pad_n = ROW_LEN - len(ids)
            if pad_n > 0:
                ids = ids + [builder.eos_id] * pad_n
                w = w + [W_IGNORE] * pad_n
            arr_t = np.asarray(ids, dtype=np.uint16)
            arr_m = np.asarray(w[1:], dtype=np.uint8)
            if (arr_m > 0).sum() >= 8:
                niah_rows_t.append(arr_t)
                niah_rows_m.append(arr_m)
                if use_chatml:
                    chatml_cnt += 1
                else:
                    plain_cnt += 1

    nt_arr = np.stack(niah_rows_t, axis=0)
    nm_arr = np.stack(niah_rows_m, axis=0)
    perm_n = np_rng.permutation(len(nt_arr))
    nt_arr = nt_arr[perm_n]
    nm_arr = nm_arr[perm_n]

    n_niah_val = 200
    save_split_memmaps(out_dir, "niah_val", nt_arr[:n_niah_val], nm_arr[:n_niah_val])
    save_split_memmaps(out_dir, "niah_train", nt_arr[n_niah_val:], nm_arr[n_niah_val:])

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "seq_len": SEQ_LEN,
        "cosmopedia_rg_range": [100, cosmo_last_rg],
        "fineweb_edu_rg_range": [100, fw_last_rg],
        "prose_train_rows": int(len(prose_toks) - n_prose_val),
        "prose_train_tokens": int((len(prose_toks) - n_prose_val) * SEQ_LEN),
        "prose_val_rows": n_prose_val,
        "niah_train_rows": int(len(nt_arr) - n_niah_val),
        "niah_val_rows": n_niah_val,
        "niah_chatml_rows": chatml_cnt,
        "niah_plaintext_rows": plain_cnt,
        "rejected_holdout_docs": rejected_holdout,
        "decontamination_signatures_checked": len(bad_sigs),
    }
    with open(os.path.join(out_dir, "manifest_v2_3.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved data_v2_3 manifest ({manifest['elapsed_sec']}s): {json.dumps(manifest)}", flush=True)


if __name__ == "__main__":
    main()
