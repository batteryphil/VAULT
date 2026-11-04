#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 Programmatic Decontamination Engine & Post-Generation Verifier
# Enforces 0% evaluation overlap (exact normalized stem hashes + 13-word shingle hashes
# + holdout entity/substring filters) across all 4 VAULT benchmark suites:
#   1. Traditional 10-Task Suite (`bench_vault_traditional.py`: WikiText-2 test, LAMBADA test,
#      ARC-Easy test, ARC-Challenge test, SciQ test, PIQA val, HellaSwag val, Winogrande val, BoolQ val)
#   2. 5-Architecture Extrapolation & BPB Suite (`EVAL_BPB_PASSAGES`, `UNSEEN_NIAH_PREFIXES`, `UNSEEN_FILLER`)
#   3. Extreme 128K Mitosis Suite (`bench_vault_extreme_128k.py`)
#   4. 8-Domain Unseen ChatML Suite (`HOLDOUT_SUBSTRINGS`, `HOLDOUT_NIAH_PREFIXES`, 8-domain prompts)

import argparse
import glob
import hashlib
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from prepare_vault_v2_data import (
    HOLDOUT_NIAH_PREFIXES,
    HOLDOUT_SUBSTRINGS,
    POISON_CRITIQUE_PHRASES,
)
from prepare_vault_v2_1_data import FORBIDDEN_NIAH_SUBWORDS
from train_vault_v2 import UNSEEN_FILLER, UNSEEN_NIAH_PREFIXES
from benchmarks.bench_vault_vs_mamba123_transformer import (
    EVAL_BPB_PASSAGES,
    REPLAY_PROSE_PASSAGES,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def hash64_str(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest(), "little")


def extract_13gram_hashes(words: List[str], n: int = 13) -> List[int]:
    if len(words) < n:
        return []
    return [hash64_str(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)]


class DecontaminationGuardV2_5:
    """
    Loads 100% of evaluation items across all 4 benchmark suites offline on CPU and provides
    O(L) exact-stem + 13-gram shingle + holdout-substring filtering and post-build verification.
    """

    def __init__(self, verbose: bool = True):
        t0 = time.perf_counter()
        self.exact_stem_hashes: Set[int] = set()
        self.shingle_13_hashes: Set[int] = set()
        self.eval_counts: Dict[str, int] = {}

        # Lowercase forbidden substrings (8-domain holdouts + NIAH eval prefixes + unseen filler)
        forb = set()
        for s in HOLDOUT_SUBSTRINGS:
            sl = s.strip().lower()
            if len(sl) >= 5:
                forb.add(sl)
        for p in set(HOLDOUT_NIAH_PREFIXES) | set(UNSEEN_NIAH_PREFIXES):
            pl = p.strip().lower()
            forb.add(pl + "-")
            forb.add(pl + "_")
            forb.add(" " + pl)
        for sub in FORBIDDEN_NIAH_SUBWORDS:
            if len(sub) >= 5:
                forb.add(sub.lower())
        forb.add(UNSEEN_FILLER[:40].strip().lower())
        self.forbidden_substrings: Tuple[str, ...] = tuple(sorted(forb))

        self._ingest_all_eval_suites()
        self.build_seconds = round(time.perf_counter() - t0, 2)
        if verbose:
            print(
                f"[DecontamGuard v2.5] Indexed {sum(self.eval_counts.values()):,} eval items -> "
                f"{len(self.exact_stem_hashes):,} exact stems | {len(self.shingle_13_hashes):,} 13-grams "
                f"| {len(self.forbidden_substrings)} forbidden substrings ({self.build_seconds:.2f}s)",
                flush=True,
            )

    def _register_eval_text(self, text: str, register_stem: bool = True) -> None:
        if not text:
            return
        words = normalize_words(text)
        if register_stem and len(words) >= 5:
            self.exact_stem_hashes.add(hash64_str(" ".join(words)))
        for h in extract_13gram_hashes(words, n=13):
            self.shingle_13_hashes.add(h)

    def _ingest_all_eval_suites(self) -> None:
        # 1. BPB & Replay Passages + 8-Domain Unseen Prompts
        eight_dom_prompts = [
            "Write a BuildBit (.bb) function that defines `soa struct Photon` with `f32` fields `x`, `vx`, `ax`, allocates 4096 elements in a 16 MB `gpu_arena`, and updates positions in parallel.",
            "Write a C99 function to reverse an array of 32-bit integers in place.",
            "Write a C++20 generic class template `FixedStack<T, 128>` with `push`, `pop`, and `size` methods.",
            "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
            "Write a Lua module implementing a `Particle` class using metatables (`__index` and `:update(dt)`).",
            "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?",
            "Explain why recursion needs a base case in programming.",
        ]
        for p in list(EVAL_BPB_PASSAGES) + list(REPLAY_PROSE_PASSAGES) + eight_dom_prompts:
            self._register_eval_text(p, register_stem=True)
        self.eval_counts["bpb_and_8domain"] = len(EVAL_BPB_PASSAGES) + len(REPLAY_PROSE_PASSAGES) + len(eight_dom_prompts)

        # 2. WikiText-2 (test)
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        wt_cnt = 0
        for r in wt:
            tx = (r.get("text") or "").strip()
            if len(tx) >= 40:
                self._register_eval_text(tx, register_stem=True)
                wt_cnt += 1
        self.eval_counts["wikitext2_test"] = wt_cnt

        # 3. LAMBADA (test = 5,153)
        lam = load_dataset("EleutherAI/lambada_openai", split="test")
        for r in lam:
            self._register_eval_text(r["text"], register_stem=True)
        self.eval_counts["lambada_test"] = len(lam)

        # 4. ARC-Easy & ARC-Challenge (test)
        for sub, key in [("ARC-Easy", "arc_easy_test"), ("ARC-Challenge", "arc_challenge_test")]:
            ds = load_dataset("allenai/ai2_arc", sub, split="test")
            for r in ds:
                q = r["question"]
                self._register_eval_text(q, register_stem=True)
                for c in r["choices"]["text"]:
                    self._register_eval_text(f"{q} {c}", register_stem=False)
            self.eval_counts[key] = len(ds)

        # 5. SciQ (test)
        sciq = load_dataset("allenai/sciq", split="test")
        for r in sciq:
            self._register_eval_text(r["question"], register_stem=True)
            if r.get("support"):
                self._register_eval_text(r["support"], register_stem=True)
        self.eval_counts["sciq_test"] = len(sciq)

        # 6. PIQA (validation)
        piqa = load_dataset("baber/piqa", split="validation")
        for r in piqa:
            self._register_eval_text(r["goal"], register_stem=True)
            self._register_eval_text(f"{r['goal']} {r['sol1']}", register_stem=False)
            self._register_eval_text(f"{r['goal']} {r['sol2']}", register_stem=False)
        self.eval_counts["piqa_validation"] = len(piqa)

        # 7. HellaSwag (validation)
        hs = load_dataset("Rowan/hellaswag", split="validation")
        for r in hs:
            stem = f"{r['activity_label']}: {r['ctx_a']} {r['ctx_b']}"
            self._register_eval_text(stem, register_stem=True)
            self._register_eval_text(f"{r['ctx_a']} {r['ctx_b']}", register_stem=True)
            for end in r["endings"]:
                self._register_eval_text(f"{r['ctx_a']} {r['ctx_b']} {end}", register_stem=False)
        self.eval_counts["hellaswag_validation"] = len(hs)

        # 8. Winogrande (validation)
        wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        for r in wg:
            self._register_eval_text(r["sentence"], register_stem=True)
        self.eval_counts["winogrande_validation"] = len(wg)

        # 9. BoolQ (validation)
        bq = load_dataset("google/boolq", split="validation")
        for r in bq:
            self._register_eval_text(r["question"], register_stem=True)
            # Note: BoolQ passages come from Wikipedia; register question+passage 13-grams for question,
            # and full passage stem + 13-grams so we never train on BoolQ validation passages either!
            self._register_eval_text(r["passage"], register_stem=True)
        self.eval_counts["boolq_validation"] = len(bq)

    def check_text(
        self,
        text: str,
        stem_texts: Optional[Iterable[str]] = None,
        check_poison: bool = False,
    ) -> Tuple[bool, str]:
        """
        Returns `(is_clean, reason)`.
        Rejects if `text` contains any forbidden holdout substring, critique poison (when enabled),
        exact normalized stem collision, or ANY 13-word shingle collision with any eval split.
        """
        if not text:
            return False, "empty"
        tl = text.lower()
        for sub in self.forbidden_substrings:
            if sub in tl:
                return False, f"forbidden_substring:{sub}"
        if check_poison:
            for p in POISON_CRITIQUE_PHRASES:
                if p in tl:
                    return False, "critique_poison"

        if stem_texts is not None:
            for st in stem_texts:
                sw = normalize_words(st)
                if len(sw) >= 5 and hash64_str(" ".join(sw)) in self.exact_stem_hashes:
                    return False, "exact_eval_stem_collision"
        else:
            words_full = normalize_words(text)
            if len(words_full) >= 5 and hash64_str(" ".join(words_full)) in self.exact_stem_hashes:
                return False, "exact_eval_stem_collision"

        words = normalize_words(text)
        for h in extract_13gram_hashes(words, n=13):
            if h in self.shingle_13_hashes:
                return False, "eval_13gram_collision"

        return True, "ok"


def audit_data_v2_5_directory(data_dir: str, out_json: str) -> Dict:
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    guard = DecontaminationGuardV2_5(verbose=True)

    report: Dict[str, object] = {
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": data_dir,
        "eval_item_counts": guard.eval_counts,
        "indexed_exact_stems": len(guard.exact_stem_hashes),
        "indexed_13gram_shingles": len(guard.shingle_13_hashes),
        "files": {},
        "total_rows_scanned": 0,
        "total_leaks_found": 0,
    }

    bin_files = sorted(glob.glob(os.path.join(data_dir, "*_tokens.bin")))
    for bf in bin_files:
        fname = os.path.basename(bf)
        row_len = 4097 if "4k" in fname else 2049
        n_rows = os.path.getsize(bf) // (row_len * 2)
        mm = np.memmap(bf, dtype=np.uint16, mode="r", shape=(n_rows, row_len))
        leaks = 0
        leak_reasons: Dict[str, int] = {}

        eos_id = int(tok.convert_tokens_to_ids("<|endoftext|>"))
        # Decode packed segments (split on `<|endoftext|>`) in batches for exact per-item verification
        for st in range(0, n_rows, 128):
            en = min(n_rows, st + 128)
            seg_ids_batch: List[List[int]] = []
            seg_row_owner: List[int] = []
            for r_off, row_arr in enumerate(mm[st:en]):
                cuts = np.where(row_arr == eos_id)[0]
                prev = 0
                for c_idx in cuts:
                    if c_idx > prev + 2:
                        seg_ids_batch.append(row_arr[prev:c_idx].tolist())
                        seg_row_owner.append(r_off)
                    prev = int(c_idx) + 1
                if len(row_arr) > prev + 2:
                    seg_ids_batch.append(row_arr[prev:].tolist())
                    seg_row_owner.append(r_off)

            texts = tok.batch_decode(seg_ids_batch, skip_special_tokens=True) if seg_ids_batch else []
            leaked_rows_in_batch: Set[int] = set()
            for r_off, tx in zip(seg_row_owner, texts):
                if r_off in leaked_rows_in_batch:
                    continue
                tl = tx.lower()
                bad_reason = None
                for sub in guard.forbidden_substrings:
                    if sub in tl:
                        bad_reason = f"forbidden_substring:{sub}"
                        break
                if bad_reason is None:
                    words = normalize_words(tx)
                    for h in extract_13gram_hashes(words, n=13):
                        if h in guard.shingle_13_hashes:
                            bad_reason = "eval_13gram_collision"
                            break
                if bad_reason is not None:
                    leaked_rows_in_batch.add(r_off)
                    leaks += 1
                    leak_reasons[bad_reason] = leak_reasons.get(bad_reason, 0) + 1

        report["files"][fname] = {
            "rows": int(n_rows),
            "row_len": int(row_len),
            "leaks": int(leaks),
            "leak_reasons": leak_reasons,
        }
        report["total_rows_scanned"] = int(report["total_rows_scanned"]) + int(n_rows)
        report["total_leaks_found"] = int(report["total_leaks_found"]) + int(leaks)
        print(f"  [Audit] {fname}: {n_rows:,} rows (L={row_len-1}) -> leaks={leaks}", flush=True)

    mc_files = sorted(glob.glob(os.path.join(data_dir, "mc_contrastive_*.npz")))
    for mf in mc_files:
        fname = os.path.basename(mf)
        d = np.load(mf)
        p_ids = d["prompt_ids"]
        p_lens = d["prompt_lens"]
        c_ids = d["choice_ids"]
        c_lens = d["choice_lens"]
        n_ch = d["n_choices"]
        leaks = 0
        for i in range(len(p_ids)):
            ptxt = tok.decode(p_ids[i, : int(p_lens[i])].tolist(), skip_special_tokens=True)
            ok, _ = guard.check_text(ptxt, stem_texts=[ptxt])
            if not ok:
                leaks += 1
                continue
            for j in range(int(n_ch[i])):
                ctxt = tok.decode(c_ids[i, j, : int(c_lens[i, j])].tolist(), skip_special_tokens=True)
                ok_c, _ = guard.check_text(ptxt + " " + ctxt)
                if not ok_c:
                    leaks += 1
                    break
        report["files"][fname] = {"items": int(len(p_ids)), "leaks": int(leaks)}
        report["total_rows_scanned"] = int(report["total_rows_scanned"]) + int(len(p_ids))
        report["total_leaks_found"] = int(report["total_leaks_found"]) + int(leaks)
        print(f"  [Audit] {fname}: {len(p_ids):,} MC tuples -> leaks={leaks}", flush=True)

    wg_files = sorted(glob.glob(os.path.join(data_dir, "wg_contrastive_*.npz")))
    for wf in wg_files:
        fname = os.path.basename(wf)
        d = np.load(wf)
        seq_ids = d["seq_ids"]
        suf_start = d["suf_start"]
        suf_len = d["suf_len"]
        leaks = 0
        for i in range(len(seq_ids)):
            l0 = int(suf_start[i, 0]) + int(suf_len[i, 0])
            l1 = int(suf_start[i, 1]) + int(suf_len[i, 1])
            t0_s = tok.decode(seq_ids[i, 0, :l0].tolist(), skip_special_tokens=True)
            t1_s = tok.decode(seq_ids[i, 1, :l1].tolist(), skip_special_tokens=True)
            ok0, _ = guard.check_text(t0_s, stem_texts=[t0_s])
            ok1, _ = guard.check_text(t1_s, stem_texts=[t1_s])
            if not (ok0 and ok1):
                leaks += 1
        report["files"][fname] = {"items": int(len(seq_ids)), "leaks": int(leaks)}
        report["total_rows_scanned"] = int(report["total_rows_scanned"]) + int(len(seq_ids))
        report["total_leaks_found"] = int(report["total_leaks_found"]) + int(leaks)
        print(f"  [Audit] {fname}: {len(seq_ids):,} WG pairs -> leaks={leaks}", flush=True)

    report["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    if int(report["total_leaks_found"]) != 0:
        raise RuntimeError(f"DECONTAMINATION FAILURE: {report['total_leaks_found']} leaks detected! See {out_json}")
    print(
        f"[PASS] Zero-Contamination Audit Verified: {report['total_rows_scanned']:,} rows/items scanned, "
        f"0 leaks ({report['elapsed_sec']:.1f}s) -> {out_json}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v2_5"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "decontamination_audit_v2_5.json"),
    )
    args = parser.parse_args()
    audit_data_v2_5_directory(args.data_dir, args.out)
