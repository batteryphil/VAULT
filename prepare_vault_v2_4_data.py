#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.4 Large-Scale Continual Pretraining Corpus Builder (`data_v2_4/`):
#   1. `prose_train_tokens.bin` / `prose_val_tokens.bin`: ~614M completely unseen, strictly decontaminated
#      English textbook & STEM educational tokens (`rg_idx >= 70` of `Cosmopedia-v2` + `FineWeb-Edu-Dedup`),
#      streamed directly to uint16 NVMe memmap (~1.23 GB disk, <500 MB RAM).
#   2. `niah_train_{tokens,mask}.bin` (`10,000` rows = `20.5M` tokens): Wide-diversity `14..38`-section
#      open-vocabulary associative recall (`55%` ChatML + `45%` plain-text 1-shot/QA) across 17,904 stems
#      so multi-hour continual pretraining preserves and sharpens `2K -> 32K` recall without NIAH overfitting.

import argparse
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
    W_IGNORE,
    VaultV2TokenBuilder,
    save_split_memmaps,
)
from prepare_vault_v2_1_data import encode_open_niah_turn
from prepare_vault_v2_2_data import build_clean_open_vocab_pool, synthesize_open_vocab_niah_v2_2
from prepare_vault_v2_3_data import encode_plaintext_niah_row, synthesize_dense_niah_v2_3
from train_vault_v2 import UNSEEN_FILLER, UNSEEN_NIAH_PREFIXES
from benchmarks.bench_vault_vs_mamba123_transformer import EVAL_BPB_PASSAGES, REPLAY_PROSE_PASSAGES


def build_strict_decontam_signatures() -> List[str]:
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
        mid = len(pl) // 2
        sigs.add(pl[:44])
        sigs.add(pl[max(0, mid - 22) : mid + 22])
        sigs.add(pl[-44:])
    return sorted(sigs)


def main():
    parser = argparse.ArgumentParser(description="Build ~614M-token unseen continual pretraining corpus for VAULT v2.4")
    parser.add_argument("--target_prose_rows", type=int, default=301000, help="Total L=2048 rows (301k = 616.4M tokens)")
    parser.add_argument("--val_prose_rows", type=int, default=1000, help="Held-out validation rows (1k = 2.05M tokens)")
    parser.add_argument("--niah_rows", type=int, default=10200, help="Diverse open-vocab NIAH replay rows")
    args = parser.parse_args()

    t0 = time.perf_counter()
    rng = random.Random(20261102)
    np_rng = np.random.default_rng(20261102)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2_4")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)
    eos_id = int(builder.eos_id)
    stem_pool = build_clean_open_vocab_pool(tok)
    bad_sigs = build_strict_decontam_signatures()

    cosmo_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "cosmopedia-v2", "train-00000-of-00104.parquet")
    fw_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "fineweb-edu-dedup", "train-00000-of-00234.parquet")
    pf_cosmo = pq.ParquetFile(cosmo_path)
    pf_fw = pq.ParquetFile(fw_path)

    # Strictly unseen row-groups (`rg >= 70`; Stage-1 only touched `rg < 70`)
    sources: List[Tuple[str, int]] = []
    for rg in range(70, pf_cosmo.num_row_groups):
        sources.append(("cosmo", rg))
    for rg in range(70, pf_fw.num_row_groups):
        sources.append(("fw", rg))
    rng.shuffle(sources)

    target_total = args.target_prose_rows
    n_val = args.val_prose_rows
    n_train = target_total - n_val

    print("=" * 100, flush=True)
    print(
        f"VAULT v2.4 STREAMING CORPUS BUILDER: Target={target_total:,} rows ({target_total * SEQ_LEN / 1e6:.1f}M unseen tokens)",
        flush=True,
    )
    print(f"  Candidate Unseen Row-Groups: {len(sources)} (Cosmopedia-v2 rg>=70 + FineWeb-Edu-Dedup rg>=70)", flush=True)
    print("=" * 100, flush=True)

    train_tok_path = os.path.join(out_dir, "prose_train_tokens.bin")
    val_tok_path = os.path.join(out_dir, "prose_val_tokens.bin")
    mm_train = np.memmap(train_tok_path + ".tmp", dtype=np.uint16, mode="w+", shape=(n_train, ROW_LEN))
    mm_val = np.memmap(val_tok_path + ".tmp", dtype=np.uint16, mode="w+", shape=(n_val, ROW_LEN))

    buf = np.empty(0, dtype=np.uint16)
    written_val = 0
    written_train = 0
    rejected_holdout = 0
    cosmo_rgs_used = 0
    fw_rgs_used = 0

    for src_idx, (src_kind, rg_idx) in enumerate(sources):
        if written_train >= n_train and written_val >= n_val:
            break
        pf = pf_cosmo if src_kind == "cosmo" else pf_fw
        min_chars = 220 if src_kind == "cosmo" else 240
        batch_texts = pf.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()
        rng.shuffle(batch_texts)

        clean_texts: List[str] = []
        for tx in batch_texts:
            if not tx or len(tx) < min_chars:
                continue
            tl = tx.lower()
            if any(sig in tl for sig in bad_sigs):
                rejected_holdout += 1
                continue
            clean_texts.append(tx)

        if src_kind == "cosmo":
            cosmo_rgs_used += 1
        else:
            fw_rgs_used += 1

        chunks_u16: List[np.ndarray] = [buf] if buf.size > 0 else []
        for i in range(0, len(clean_texts), 2048):
            enc = tok(clean_texts[i : i + 2048], add_special_tokens=False)["input_ids"]
            flat: List[int] = []
            for ids in enc:
                if len(ids) >= 64:
                    flat.extend(ids)
                    flat.append(eos_id)
            if flat:
                chunks_u16.append(np.asarray(flat, dtype=np.uint16))

        if not chunks_u16:
            continue
        stream = np.concatenate(chunks_u16)
        n_avail = stream.size // ROW_LEN
        if n_avail == 0:
            buf = stream
            continue

        rows_2d = stream[: n_avail * ROW_LEN].reshape(n_avail, ROW_LEN)
        buf = stream[n_avail * ROW_LEN :].copy()

        # Route ~1 in 250 rows to validation until `n_val` filled, rest to `mm_train`
        cursor = 0
        while cursor < n_avail and (written_train < n_train or written_val < n_val):
            if written_val < n_val and (written_train == n_train or (cursor % 250 == 0)):
                mm_val[written_val] = rows_2d[cursor]
                written_val += 1
                cursor += 1
            else:
                take = min(n_avail - cursor, n_train - written_train)
                if written_val < n_val:
                    take = min(take, 249)
                mm_train[written_train : written_train + take] = rows_2d[cursor : cursor + take]
                written_train += take
                cursor += take

        if (src_idx + 1) % 40 == 0 or (written_train >= n_train and written_val >= n_val):
            dt = time.perf_counter() - t0
            tot_tok_m = (written_train + written_val) * SEQ_LEN / 1e6
            print(
                f"  [Prose {src_idx+1:4d}/{len(sources)}] train={written_train:,}/{n_train:,} | val={written_val:,}/{n_val:,} "
                f"| {tot_tok_m:6.1f}M tok ({tot_tok_m / max(dt, 1e-3):.2f}M tok/s) | rejected={rejected_holdout}",
                flush=True,
            )

    mm_train.flush()
    mm_val.flush()
    del mm_train, mm_val

    # If candidate pool yielded slightly fewer than target_total, truncate cleanly to exact written count
    if written_train < n_train:
        os.truncate(train_tok_path + ".tmp", written_train * ROW_LEN * 2)
        n_train = written_train
    if written_val < n_val:
        os.truncate(val_tok_path + ".tmp", written_val * ROW_LEN * 2)
        n_val = written_val

    os.replace(train_tok_path + ".tmp", train_tok_path)
    os.replace(val_tok_path + ".tmp", val_tok_path)

    # --------------------------------------------------------------------------
    # 2. WIDE-DIVERSITY DUAL-FORMAT OPEN-VOCAB NIAH (`niah_train` / `niah_val`)
    # --------------------------------------------------------------------------
    print(f"[2/2] Synthesizing {args.niah_rows:,} Open-Vocab NIAH Replay Rows (55% ChatML / 45% Plain-Text)...", flush=True)
    niah_rows_t: List[np.ndarray] = []
    niah_rows_m: List[np.ndarray] = []
    chatml_cnt = 0
    plain_cnt = 0

    while len(niah_rows_t) < args.niah_rows:
        use_chatml = rng.random() < 0.55
        use_dense = rng.random() < 0.55
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
            ids, w = encode_plaintext_niah_row(tok, eos_id, prm, queried, ex0, rng)

        if 384 <= len(ids) <= ROW_LEN:
            pad_n = ROW_LEN - len(ids)
            if pad_n > 0:
                ids = ids + [eos_id] * pad_n
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
        "prose_train_rows": int(n_train),
        "prose_train_tokens": int(n_train * SEQ_LEN),
        "prose_val_rows": int(n_val),
        "prose_val_tokens": int(n_val * SEQ_LEN),
        "cosmopedia_rgs_used": cosmo_rgs_used,
        "fineweb_edu_rgs_used": fw_rgs_used,
        "niah_train_rows": int(len(nt_arr) - n_niah_val),
        "niah_val_rows": n_niah_val,
        "niah_chatml_rows": chatml_cnt,
        "niah_plaintext_rows": plain_cnt,
        "rejected_holdout_docs": rejected_holdout,
        "decontamination_signatures_checked": len(bad_sigs),
    }
    with open(os.path.join(out_dir, "manifest_v2_4.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved data_v2_4 manifest ({manifest['elapsed_sec']}s): {json.dumps(manifest)}", flush=True)


if __name__ == "__main__":
    main()
