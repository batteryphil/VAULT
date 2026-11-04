#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.5 Post-Step-10,000 Decontaminated 5-Stream SFT & Long-Horizon Corpus Builder (`data_v2_5/`)
#
# Builds 100% offline-cached, zero-leakage NVMe memmaps & contrastive choice arrays:
#   1. `sft_skill_{train,val}_{tokens,mask}.bin` (`L = 2,048`):
#        - Stream 1B (40%): Dual-format (`55%` plain-text continuation + `45%` ChatML) STEM, Reading Comp,
#          and Commonsense (`sciq`, `ai2_arc`, `openbookqa`, `commonsense_qa`, `piqa`, `hellaswag`,
#          `winogrande_xl`, and 50/50 class-balanced `boolq` — official `train` splits ONLY).
#        - Stream 3  (35%): `<think>` quantitative math (`gsm8k train`, `MetaMathQA`, place-value arithmetic)
#          + strictly `<think>`-free direct conceptual CS/science ChatML (`smoltalk` + `CS_CONCEPT_TOPICS`).
#        - Stream 4  (25%): From-scratch polyglot code (`Magicoder`, `CodeFeedback`, `python_code_alpaca`,
#          authentic `buildbit/*.bb` & `buildbit-os/*.bb`, and diverse C99 / C++20 / Python / Lua modules).
#   2. `mc_contrastive_{train,val}.npz` & `wg_contrastive_{train,val}.npz`:
#        - Shared-prefix 2..4-way multiple-choice tuples + Winogrande suffix-scoring pairs from decontaminated
#          `train` splits for the Byte-Normalized + Raw-Prior Contrastive Choice-Margin Loss (`L_MC`).
#   3. `prose_climax_{train,val}_{tokens,mask}.bin` (`L = 2,048`) & `prose_climax_4k_{train,val}_{tokens,mask}.bin` (`L = 4,096`):
#        - Unseen textbook & narrative prose (`Cosmopedia-v2` + `FineWeb-Edu` high row-groups `rg >= 165`,
#          100% disjoint from `v2.0` & `v2.4`, plus `TinyStories train`) with `3.5x` (`code = 35`) salience
#          boost on clause/sentence-final repeated words & named entities (`LAMBADA` sharpener with 0% leakage).
#   4. `niah_2k_{train,val}_{tokens,mask}.bin` (`L = 2,048`) & `niah_4k_{train,val}_{tokens,mask}.bin` (`L = 4,096`):
#        - Dual-format (`55%` ChatML / `45%` plain-text 1-shot & QA) associative retrieval featuring `55%`
#          adversarial `8..11`-BPE-subword sibling entities sharing identical `7`-token prefixes across
#          `3..38`-token syntax gaps (`16..34` sections at `2K`, `42..60` dense sections at `4K`).
#   5. Automatic end-to-end verification via `audit_data_v2_5_directory` (`total_leaks_found == 0`).

import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

from audit_decontamination_v2_5 import DecontaminationGuardV2_5, audit_data_v2_5_directory
from benchmarks.bench_vault_traditional import hellaswag_preprocess
from prepare_vault_v2_data import (
    HOLDOUT_NIAH_PREFIXES,
    HOLDOUT_SUBSTRINGS,
    SEQ_LEN,
    W_ASSIST,
    W_IGNORE,
    W_NEEDLE,
    W_PROSE,
    VaultV2TokenBuilder,
    generate_conceptual_prose_turn,
    is_clean_pair,
    parse_metamath_cot,
)
from prepare_vault_v2_1_data import (
    FILLER_SENTENCES_V2_1,
    FORBIDDEN_NIAH_SUBWORDS,
    encode_open_niah_turn,
    generate_decomposed_math_turn,
)
from prepare_vault_v2_2_data import (
    build_clean_open_vocab_pool,
    generate_prompt_bound_polyglot_v2_2,
    sample_v2_2_entity,
)
from prepare_vault_v2_3_data import encode_plaintext_niah_row
from train_vault_v2 import UNSEEN_NIAH_PREFIXES

W_CLIMAX = 35  # 3.5x salience on sentence-final repeated entities/nouns in unseen prose
W_MC_GOLD = 22  # 2.2x salience on plain-text continuation gold target spans


class FlexRowPacker:
    """Packs atomic or continuous sequences into `(N, seq_len + 1)` uint16 tokens + `(N, seq_len)` uint8 masks."""

    def __init__(self, eos_id: int, seq_len: int = 2048):
        self.eos_id = int(eos_id)
        self.seq_len = int(seq_len)
        self.row_len = self.seq_len + 1
        self.rows_tokens: List[np.ndarray] = []
        self.rows_masks: List[np.ndarray] = []
        self.cur_ids: List[int] = []
        self.cur_w: List[int] = []

    def _flush(self) -> None:
        if not self.cur_ids:
            return
        if sum(1 for w in self.cur_w[1:] if w > 0) < 6:
            self.cur_ids.clear()
            self.cur_w.clear()
            return
        pad_n = self.row_len - len(self.cur_ids)
        if pad_n > 0:
            self.cur_ids.extend([self.eos_id] * pad_n)
            self.cur_w.extend([W_IGNORE] * pad_n)
        self.rows_tokens.append(np.asarray(self.cur_ids[: self.row_len], dtype=np.uint16))
        self.rows_masks.append(np.asarray(self.cur_w[1 : self.row_len], dtype=np.uint8))
        self.cur_ids.clear()
        self.cur_w.clear()

    def add_atomic(self, ids: List[int], w: List[int]) -> bool:
        if len(ids) < 8 or len(ids) > self.row_len:
            return False
        need = len(ids) + (1 if self.cur_ids else 0)
        if len(self.cur_ids) + need > self.row_len:
            self._flush()
        if self.cur_ids:
            self.cur_ids.append(self.eos_id)
            self.cur_w.append(W_IGNORE)
        w_adj = list(w)
        w_adj[0] = W_IGNORE
        self.cur_ids.extend(ids)
        self.cur_w.extend(w_adj)
        if len(self.cur_ids) >= self.row_len - 16:
            self._flush()
        return True

    def add_continuous(self, ids: List[int], w: List[int]) -> None:
        pos = 0
        n = len(ids)
        while pos < n:
            space = self.row_len - len(self.cur_ids)
            take = min(space, n - pos)
            chunk_i = ids[pos : pos + take]
            chunk_w = list(w[pos : pos + take])
            if pos == 0 and take > 0:
                chunk_w[0] = W_IGNORE
            self.cur_ids.extend(chunk_i)
            self.cur_w.extend(chunk_w)
            pos += take
            if len(self.cur_ids) == self.row_len:
                self.rows_tokens.append(np.asarray(self.cur_ids, dtype=np.uint16))
                self.rows_masks.append(np.asarray(self.cur_w[1:], dtype=np.uint8))
                self.cur_ids.clear()
                self.cur_w.clear()
        if self.cur_ids and self.cur_ids[-1] != self.eos_id and len(self.cur_ids) < self.row_len:
            self.cur_ids.append(self.eos_id)
            self.cur_w.append(W_IGNORE)
            if len(self.cur_ids) == self.row_len:
                self.rows_tokens.append(np.asarray(self.cur_ids, dtype=np.uint16))
                self.rows_masks.append(np.asarray(self.cur_w[1:], dtype=np.uint8))
                self.cur_ids.clear()
                self.cur_w.clear()

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        self._flush()
        if not self.rows_tokens:
            return (
                np.zeros((0, self.row_len), dtype=np.uint16),
                np.zeros((0, self.seq_len), dtype=np.uint8),
            )
        return np.stack(self.rows_tokens, axis=0), np.stack(self.rows_masks, axis=0)


def save_memmaps_flex(out_dir: str, prefix: str, tokens_arr: np.ndarray, masks_arr: np.ndarray) -> None:
    tok_path = os.path.join(out_dir, f"{prefix}_tokens.bin")
    msk_path = os.path.join(out_dir, f"{prefix}_mask.bin")
    fp_t = np.memmap(tok_path, dtype=np.uint16, mode="w+", shape=tokens_arr.shape)
    fp_t[:] = tokens_arr[:]
    fp_t.flush()
    fp_m = np.memmap(msk_path, dtype=np.uint8, mode="w+", shape=masks_arr.shape)
    fp_m[:] = masks_arr[:]
    fp_m.flush()
    seq_l = masks_arr.shape[1]
    print(
        f"  [+] Wrote {prefix}: {tokens_arr.shape[0]:,} rows x {seq_l} ({tokens_arr.shape[0] * seq_l / 1e6:.2f}M tok) "
        f"| {os.path.getsize(tok_path)/1e6:.1f} MB + {os.path.getsize(msk_path)/1e6:.1f} MB",
        flush=True,
    )


def encode_plain_continuation(
    tok, eos_id: int, ctx: str, gold_cont: str, gold_weight: int = W_MC_GOLD
) -> Tuple[List[int], List[int]]:
    p_ids = tok.encode(ctx, add_special_tokens=False)
    c_ids = tok.encode(gold_cont, add_special_tokens=False)
    if not p_ids or not c_ids:
        return [], []
    ids = p_ids + c_ids + [eos_id]
    w = [W_IGNORE] * len(p_ids) + [gold_weight] * len(c_ids) + [W_ASSIST]
    return ids, w


def build_climax_salience_mask(
    ids: List[int],
    word_starts: Set[int],
    alpha_tokens: Set[int],
    punct_end_tokens: Set[int],
    min_prior_dist: int = 44,
) -> List[int]:
    """
    Assigns `W_PROSE = 10` (`1.0x`) across a clean natural passage, boosting to `W_CLIMAX = 35` (`3.5x`)
    whenever a whole-word alphabetic entity/noun span (`[s .. e)`) occurs right before clause-ending
    punctuation (`.`, `!`, `?`, `"`, `\\n`) AND that exact multi-token word span previously appeared
    at least `min_prior_dist` tokens earlier in the passage — mirroring LAMBADA's discourse-climax
    recall mechanics purely on unseen natural prose (`0%` LAMBADA exposure).
    """
    n = len(ids)
    w = [W_PROSE] * n
    first_seen: Dict[Tuple[int, ...], int] = {}
    i = 0
    while i < n:
        tid = ids[i]
        if tid in word_starts and tid in alpha_tokens:
            j = i + 1
            while j < n and (ids[j] not in word_starts) and (ids[j] in alpha_tokens) and (j - i < 6):
                j += 1
            span = tuple(ids[i:j])
            prev_pos = first_seen.get(span)
            if prev_pos is None:
                first_seen[span] = i
            elif i - prev_pos >= min_prior_dist and j < n and ids[j] in punct_end_tokens:
                for k in range(i, j):
                    w[k] = W_CLIMAX
            i = j
        else:
            i += 1
    return w


def synthesize_long_sibling_niah_v2_5(
    rng: random.Random,
    tok,
    stem_pool: List[str],
    target_4k: bool = False,
) -> Tuple[str, List[Tuple[str, str]], Tuple[str, str], bool]:
    """
    Synthesizes associative retrieval manifests where `55%` of manifests contain adversarial
    `8..11`-BPE-subword sibling identifiers sharing identical `7`-BPE-token prefixes
    (e.g. `FalconRotor-T21408` vs `FalconRotor-T21953`, `cobalt_bus_lane_4108` vs `cobalt_bus_lane_4195`)
    and `45%` use standard `2..7`-subword open-vocabulary identifiers, across `3..36`-token gaps.
    """
    n_sections = rng.randint(33, 47) if target_4k else rng.randint(14, 24)
    long_sibling_mode = rng.random() < 0.58
    val_kind = rng.choice(["CHK", "CHK", "CHK", "CHK", "VRF", "SIG", "TOK", "HEX"])

    items: List[Tuple[str, str]] = []
    used_ents, used_vals = set(), set()

    if long_sibling_mode:
        n_fams = rng.randint(4, 7) if target_4k else rng.randint(3, 5)
        fam_bases: List[Tuple[str, str]] = []
        while len(fam_bases) < n_fams:
            w1 = rng.choice(stem_pool)
            w2 = rng.choice(stem_pool)
            w3 = rng.choice(stem_pool) if rng.random() < 0.35 else ""
            style = rng.choice(["camel_5dig", "camel_5dig", "camel_tier", "snake_5dig"])
            tier_ch = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
            tier_hi = rng.randint(10, 89)  # Shared 2-digit prefix so first 7 BPE tokens collide across siblings!
            if style == "snake_5dig":
                base_str = f"{w1}_{w2}" + (f"_{w3}" if w3 else "")
                probe_ent = f"{base_str.lower()}_{tier_ch.lower()}{tier_hi}408"
            elif style == "camel_tier":
                base_str = f"{w1}{w2}{w3}"
                probe_ent = f"{base_str}-Unit{tier_hi}408"
            else:
                base_str = f"{w1}{w2}{w3}"
                probe_ent = f"{base_str}-{tier_ch}{tier_hi}408"
            cand_check = probe_ent.lower()
            probe_len = len(tok.encode(" " + probe_ent, add_special_tokens=False))
            if (
                8 <= probe_len <= 11
                and all(sub not in cand_check for sub in FORBIDDEN_NIAH_SUBWORDS)
                and all(h.lower() not in cand_check for h in HOLDOUT_NIAH_PREFIXES)
                and all(u.lower() not in cand_check for u in UNSEEN_NIAH_PREFIXES)
                and all(hs not in cand_check for hs in HOLDOUT_SUBSTRINGS)
            ):
                fam_bases.append((base_str, f"{style}:{tier_ch}:{tier_hi}"))

        for sec_i in range(n_sections):
            base_name, spec = fam_bases[sec_i % len(fam_bases)]
            style, tier_ch, tier_hi_s = spec.split(":")
            ent = ""
            for attempt in range(40):
                tail_3 = rng.randint(100, 999) if attempt < 30 else rng.randint(1000, 9999)
                if style == "snake_5dig":
                    cand = f"{base_name.lower()}_{tier_ch.lower()}{tier_hi_s}{tail_3}"
                elif style == "camel_tier":
                    cand = f"{base_name}-Unit{tier_hi_s}{tail_3}"
                else:
                    cand = f"{base_name}-{tier_ch}{tier_hi_s}{tail_3}"
                if cand in used_ents:
                    continue
                e_len = len(tok.encode(" " + cand, add_special_tokens=False))
                if 7 <= e_len <= 11 or attempt >= 25:
                    ent = cand
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
    else:
        styles = ["camel", "camel", "single_stem", "oov_neologism", "snake_id"]
        for _ in range(n_sections):
            while True:
                ent = sample_v2_2_entity(rng, stem_pool, rng.choice(styles))
                el = ent.lower()
                if ent not in used_ents and all(u.lower() not in el for u in UNSEEN_NIAH_PREFIXES):
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

    # Prefer querying sibling pairs from the SAME prefix family when in `long_sibling_mode`!
    n_q = rng.choice([2, 2, 2, 3])
    if long_sibling_mode and len(items) >= 8 and rng.random() < 0.75:
        fam0_indices = [i for i in range(1, len(items)) if (i % len(fam_bases)) == (1 % len(fam_bases))]
        if len(fam0_indices) >= 2:
            picked = rng.sample(fam0_indices, k=min(n_q, len(fam0_indices)))
            while len(picked) < n_q:
                extra = rng.randint(1, len(items) - 1)
                if extra not in picked:
                    picked.append(extra)
            q_indices = sorted(picked)
        else:
            q_indices = sorted(rng.sample(range(1, len(items)), k=n_q))
    else:
        q_indices = sorted(rng.sample(range(1, len(items)), k=n_q))

    queried = [items[i] for i in q_indices]
    q_names = (
        f"{queried[0][0]} and {queried[1][0]}"
        if len(queried) == 2
        else ", ".join(e for e, _ in queried[:-1]) + f", and {queried[-1][0]}"
    )

    schema = rng.choice(["telemetry", "telemetry", "telemetry_mixed", "yaml_registry", "dossier"])
    if schema in ("telemetry", "telemetry_mixed"):
        blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
        ugap = rng.choice(["gap_15", "gap_15", "gap_3", "gap_24", "gap_32"])
        for idx, (ent, val) in enumerate(items, start=1):
            status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE"])
            gm = rng.choice(["gap_3", "gap_15", "gap_19", "gap_24", "gap_32"]) if schema == "telemetry_mixed" else ugap
            if gm == "gap_3":
                hdr = f"[Subsystem {idx:03d}] {ent}: {val} ({status})"
            elif gm == "gap_19":
                hdr = f"[Subsystem {idx:03d}: {ent}] Health: {status} | Bus Channel: {idx % 16:02d} | Verification Code: {val}"
            elif gm == "gap_24":
                hdr = (
                    f"[Subsystem {idx:03d}: {ent}] Operational State: {status} | Sensor Temp: {285 + (idx % 40)}K "
                    f"| Link Lane: {idx % 8} | Verification Code: {val}"
                )
            elif gm == "gap_32":
                hdr = (
                    f"[Subsystem {idx:03d}: {ent}] State: {status} | Envelope Temp: {288 + (idx % 35)}K "
                    f"| Roundtrip Latency: {14 + (idx % 50)}us | Redundancy Tier: PRIMARY | Verification Code: {val}"
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

    if schema == "yaml_registry":
        entries_txt = ["SERVICE CONFIGURATION REGISTRY\nservices:"]
        for idx, (ent, val) in enumerate(items, start=1):
            f1 = rng.choice(FILLER_SENTENCES_V2_1)
            region = rng.choice(["us-east", "eu-central", "ap-north", "orbital-leo"])
            entries_txt.append(f"  - id: {ent}\n    seq: {idx}\n    zone: {region}\n    notes: {f1}\n    auth_token: {val}")
        question = f"\n\nQuestion: From the service registry above, report the exact `auth_token` assigned to {q_names}."
        return "\n".join(entries_txt) + question, queried, items[0], False

    paras = ["QUALIFICATION & CALIBRATION DOSSIER\n"]
    for idx, (ent, val) in enumerate(items, start=1):
        f1 = rng.choice(FILLER_SENTENCES_V2_1)
        paras.append(
            f"Record {idx:03d}: During acceptance testing, assembly {ent} completed {16 + (idx % 40)} continuous "
            f"endurance cycles with zero arbitration faults and received calibration token {val}. {f1}"
        )
    question = f"\n\nQuestion: According to the dossier above, what exact calibration tokens were assigned to {q_names}?"
    return "\n\n".join(paras) + question, queried, items[0], False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild_sft_only",
        action="store_true",
        help="Rebuild SFT Skill (2K/4K) + MC/WG Contrastive shards while keeping existing clean prose_climax & niah shards, then re-audit ALL files.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    rng = random.Random(20261103)
    np_rng = np.random.default_rng(20261103)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2_5")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)
    eos_id = int(builder.eos_id)
    stem_pool = build_clean_open_vocab_pool(tok)
    lua_open_stems = [w for w in stem_pool if w.lower() != "particle" and 5 <= len(w) <= 8]
    guard = DecontaminationGuardV2_5(verbose=True)

    stats: Dict[str, int] = {
        "rejected_decontam": 0,
        "mc_tuples_kept": 0,
        "wg_pairs_kept": 0,
        "s1_academic_items": 0,
        "s3_math_items": 0,
        "s3_concept_items": 0,
        "s4_code_items": 0,
    }

    print("=" * 100, flush=True)
    print("VAULT v2.5 POST-10K DECONTAMINATED 5-STREAM SFT & LONG-HORIZON BUILDER", flush=True)
    print("=" * 100, flush=True)

    # ==========================================================================
    # 1. STREAM 1: DECONTAMINATED ACADEMIC TRAIN SPLITS (DUAL-FORMAT + MC ARRAYS)
    # ==========================================================================
    print("[1/5] Processing Official Train Splits (SciQ, ARC-E/C, OBQA, CSQA, PIQA, HellaSwag, Winogrande, BoolQ)...", flush=True)

    # Tuple schema for `mc_items`: (prompt_str, list_of_choice_strs[2..4], gold_idx)
    mc_items: List[Tuple[str, List[str], int]] = []
    # Atomic sequence pairs `(ids, w)` for `sft_skill` Stream 1B
    s1_atomic_seqs: List[Tuple[List[int], List[int]]] = []

    def is_bpe_seq_clean(seq_ids: List[int], check_poison: bool = False) -> bool:
        dec_txt = tok.decode(seq_ids, skip_special_tokens=True)
        ok_d, _ = guard.check_text(dec_txt, check_poison=check_poison)
        return ok_d

    def try_add_mc_and_s1(
        ctx_plain: str,
        choices_plain: List[str],
        gold_idx: int,
        stem_check_list: List[str],
        chat_user: Optional[str] = None,
        chat_assist: Optional[str] = None,
        add_to_mc: bool = True,
    ) -> bool:
        p_ids_chk = tok.encode(ctx_plain, add_special_tokens=False)
        if not p_ids_chk:
            return False
        ptxt_dec = tok.decode(p_ids_chk, skip_special_tokens=True)
        ok_p, _ = guard.check_text(ctx_plain, stem_texts=list(stem_check_list) + [ctx_plain, ptxt_dec])
        if not ok_p:
            stats["rejected_decontam"] += 1
            return False
        ok_pd, _ = guard.check_text(ptxt_dec, stem_texts=[ptxt_dec])
        if not ok_pd:
            stats["rejected_decontam"] += 1
            return False

        for ch_cand in choices_plain:
            ok_raw, _ = guard.check_text(ctx_plain + " " + ch_cand)
            if not ok_raw:
                stats["rejected_decontam"] += 1
                return False
            c_ids_chk = tok.encode(ch_cand, add_special_tokens=False)
            ctxt_dec = tok.decode(c_ids_chk, skip_special_tokens=True)
            ok_dec1, _ = guard.check_text(ptxt_dec + " " + ctxt_dec)
            ok_dec2, _ = guard.check_text(tok.decode(p_ids_chk + c_ids_chk, skip_special_tokens=True))
            if not (ok_dec1 and ok_dec2):
                stats["rejected_decontam"] += 1
                return False

        if chat_user and chat_assist:
            ok2, _ = guard.check_text(chat_user + " " + chat_assist, check_poison=True)
            if not ok2:
                stats["rejected_decontam"] += 1
                return False

        # Dual-Format S1 supervision: 55% plain-text continuation, 45% clean ChatML
        if chat_user and chat_assist and rng.random() < 0.45:
            ids, w = builder.encode_chatml_turn(chat_user, chat_assist, sft_mode=True)
        else:
            ids, w = encode_plain_continuation(tok, eos_id, ctx_plain, choices_plain[gold_idx], W_MC_GOLD)
        if not is_bpe_seq_clean(ids):
            stats["rejected_decontam"] += 1
            return False

        if add_to_mc and 2 <= len(choices_plain) <= 4:
            mc_items.append((ctx_plain, choices_plain, gold_idx))

        if 12 <= len(ids) <= 1900:
            s1_atomic_seqs.append((ids, w))
            stats["s1_academic_items"] += 1
        return True

    # 1.1 ARC-Easy & ARC-Challenge (`train`)
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        ds_arc = load_dataset("allenai/ai2_arc", cfg, split="train")
        for r in ds_arc:
            labels = list(r["choices"]["label"])
            texts = [t.strip() for t in r["choices"]["text"]]
            if r["answerKey"] not in labels or len(texts) < 2:
                continue
            g_idx = labels.index(r["answerKey"])
            if len(texts) > 4:
                gold_t = texts[g_idx]
                dists = [t for i, t in enumerate(texts) if i != g_idx][:3]
                cands = dists + [gold_t]
                rng.shuffle(cands)
                texts = cands
                g_idx = texts.index(gold_t)
            q = r["question"].strip()
            ctx = f"Question: {q}\nAnswer:"
            ch = [" " + t for t in texts]
            opts_str = "\n".join(f"{chr(65+i)}. {t}" for i, t in enumerate(texts))
            c_u = f"Answer the following science question concisely.\n\n{q}\n{opts_str}"
            c_a = f"{chr(65+g_idx)}. {texts[g_idx]}"
            # Oversample scarce ARC-Challenge & ARC-Easy 2x so STEM reasoning has strong weight
            for _ in range(2):
                try_add_mc_and_s1(ctx, ch, g_idx, [q], c_u, c_a, add_to_mc=True)

    # 1.2 OpenBookQA (`main` `train` = 4,957)
    ds_obqa = load_dataset("allenai/openbookqa", "main", split="train")
    for r in ds_obqa:
        labels = list(r["choices"]["label"])
        texts = [t.strip() for t in r["choices"]["text"]]
        if r["answerKey"] not in labels or len(texts) != 4:
            continue
        g_idx = labels.index(r["answerKey"])
        q = r["question_stem"].strip()
        ctx = f"Question: {q}\nAnswer:"
        ch = [" " + t for t in texts]
        opts_str = "\n".join(f"{chr(65+i)}. {t}" for i, t in enumerate(texts))
        c_u = f"Select the correct scientific completion:\n\n{q}\n{opts_str}"
        c_a = f"{chr(65+g_idx)}. {texts[g_idx]}"
        try_add_mc_and_s1(ctx, ch, g_idx, [q], c_u, c_a, add_to_mc=True)

    # 1.3 SciQ (`train` = 11,679: both Reading-Comp w/ support AND Closed-Book!)
    ds_sciq = load_dataset("allenai/sciq", split="train")
    sciq_concept_pairs: List[Tuple[str, str]] = []
    for idx_s, r in enumerate(ds_sciq):
        q = r["question"].strip()
        sup = (r.get("support") or "").strip()
        gold_t = r["correct_answer"].strip()
        cands_raw = [r["distractor1"].strip(), r["distractor2"].strip(), r["distractor3"].strip(), gold_t]
        rng.shuffle(cands_raw)
        g_idx = cands_raw.index(gold_t)
        ch = [" " + c for c in cands_raw]
        use_rc = bool(sup) and (idx_s % 2 == 0)
        ctx = f"{sup}\nQuestion: {q}\nAnswer:" if use_rc else f"Question: {q}\nAnswer:"
        stems_chk = [q, sup] if sup else [q]
        c_u = f"Context: {sup}\n\nQuestion: {q}" if use_rc else q
        c_a = f"The correct answer is {gold_t}." if rng.random() < 0.4 else gold_t
        try_add_mc_and_s1(ctx, ch, g_idx, stems_chk, c_u, c_a, add_to_mc=(idx_s < 8000))
        if sup and 120 <= len(sup) <= 650 and len(sciq_concept_pairs) < 2500:
            ok_sup, _ = guard.check_text(q + " " + sup, stem_texts=[q, sup], check_poison=True)
            if ok_sup:
                sciq_concept_pairs.append((q, f"{gold_t}. {sup}"))

    # 1.4 CommonsenseQA (`train`, take 4,500 clean items, convert 5 choices -> gold + 3 distractors)
    ds_csqa = load_dataset("tau/commonsense_qa", split="train")
    for idx_c, r in enumerate(ds_csqa):
        if idx_c >= 4500:
            break
        labels = list(r["choices"]["label"])
        texts = [t.strip() for t in r["choices"]["text"]]
        if r["answerKey"] not in labels or len(texts) < 4:
            continue
        g_orig = labels.index(r["answerKey"])
        gold_t = texts[g_orig]
        dists = [t for i, t in enumerate(texts) if i != g_orig]
        rng.shuffle(dists)
        sub_t = dists[:3] + [gold_t]
        rng.shuffle(sub_t)
        g_idx = sub_t.index(gold_t)
        q = r["question"].strip()
        ctx = f"Question: {q}\nAnswer:"
        ch = [" " + t for t in sub_t]
        try_add_mc_and_s1(ctx, ch, g_idx, [q], q, gold_t, add_to_mc=True)

    # 1.5 PIQA (`train`, take 8,000 clean items)
    ds_piqa = load_dataset("baber/piqa", split="train")
    piqa_indices = list(range(len(ds_piqa)))
    rng.shuffle(piqa_indices)
    for idx_p in piqa_indices[:8000]:
        r = ds_piqa[idx_p]
        gold = int(r["label"])
        if gold not in (0, 1):
            continue
        goal = r["goal"].strip()
        s1, s2 = r["sol1"].strip(), r["sol2"].strip()
        ctx = f"Question: {goal}\nAnswer:"
        ch = [" " + s1, " " + s2]
        gold_sol = s1 if gold == 0 else s2
        c_u = f"Goal: {goal}\nWhat is the most practical physical solution?"
        try_add_mc_and_s1(ctx, ch, gold, [goal], c_u, gold_sol, add_to_mc=(len(mc_items) < 28000))

    # 1.6 HellaSwag (`train`, take 9,500 clean items using exact `hellaswag_preprocess`)
    ds_hs = load_dataset("Rowan/hellaswag", split="train")
    hs_indices = list(range(len(ds_hs)))
    rng.shuffle(hs_indices)
    hs_kept = 0
    for idx_h in hs_indices:
        if hs_kept >= 9500:
            break
        r = ds_hs[idx_h]
        lbl = str(r.get("label", "")).strip()
        if lbl not in ("0", "1", "2", "3"):
            continue
        g_idx = int(lbl)
        raw_stem = r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
        ctx = hellaswag_preprocess(raw_stem)
        ends = [" " + hellaswag_preprocess(e) for e in r["endings"]]
        if any(len(e.strip()) < 3 for e in ends):
            continue
        if try_add_mc_and_s1(
            ctx,
            ends,
            g_idx,
            [raw_stem, f"{r['ctx_a']} {r['ctx_b']}"],
            chat_user=None,
            chat_assist=None,
            add_to_mc=(hs_kept < 6500),
        ):
            hs_kept += 1

    # 1.7 BoolQ (`train` = 9,427 -> strictly class-balanced 50% `" yes"` / 50% `" no"`!)
    ds_bq = load_dataset("google/boolq", split="train")
    bq_yes, bq_no = [], []
    for r in ds_bq:
        q = r["question"].strip()
        if not q.endswith("?"):
            q = q + "?"
        psg = r["passage"].strip()
        if len(psg) > 2200:
            continue
        ans = bool(r["answer"])
        (bq_yes if ans else bq_no).append((psg, q, 1 if ans else 0))
    rng.shuffle(bq_yes)
    rng.shuffle(bq_no)
    n_bq_bal = min(len(bq_yes), len(bq_no), 3200)
    bq_balanced = []
    for y_it, n_it in zip(bq_yes[:n_bq_bal], bq_no[:n_bq_bal]):
        bq_balanced.append(y_it)
        bq_balanced.append(n_it)
    for psg, q, g_idx in bq_balanced:
        ctx = f"{psg}\nQuestion: {q}\nAnswer:"
        ch = [" no", " yes"]
        ans_word = "yes" if g_idx == 1 else "no"
        c_u = f"Passage: {psg}\n\nQuestion: {q}\nAnswer strictly yes or no."
        try_add_mc_and_s1(ctx, ch, g_idx, [q, psg], c_u, ans_word, add_to_mc=True)

    # 1.8 Winogrande (`winogrande_xl` `train`, take 8,500 clean items -> both suffix continuation & `wg_contrastive`!)
    ds_wg = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    wg_indices = list(range(len(ds_wg)))
    rng.shuffle(wg_indices)
    wg_pairs: List[Tuple[List[int], List[int], int, int, int, int, int]] = []
    for idx_w in wg_indices:
        if len(wg_pairs) >= 7500:
            break
        r = ds_wg[idx_w]
        sent = r["sentence"].strip()
        if "_" not in sent or r["answer"] not in ("1", "2"):
            continue
        ok_w, _ = guard.check_text(sent, stem_texts=[sent])
        if not ok_w:
            stats["rejected_decontam"] += 1
            continue
        u_pos = sent.index("_")
        pref = sent[:u_pos]
        suf = sent[u_pos + 1 :]
        o1, o2 = r["option1"], r["option2"]
        gold = int(r["answer"]) - 1

        p1_ids = tok.encode(pref + o1, add_special_tokens=False)
        p2_ids = tok.encode(pref + o2, add_special_tokens=False)
        s_ids = tok.encode(suf, add_special_tokens=False)
        if not p1_ids or not p2_ids or len(s_ids) < 2:
            continue
        seq1 = p1_ids + s_ids
        seq2 = p2_ids + s_ids
        if len(seq1) <= 96 and len(seq2) <= 96:
            t1_dec = tok.decode(seq1, skip_special_tokens=True)
            t2_dec = tok.decode(seq2, skip_special_tokens=True)
            ok_w1, _ = guard.check_text(t1_dec, stem_texts=[t1_dec])
            ok_w2, _ = guard.check_text(t2_dec, stem_texts=[t2_dec])
            if not (ok_w1 and ok_w2):
                stats["rejected_decontam"] += 1
                continue
            wg_pairs.append((seq1, seq2, len(p1_ids), len(s_ids), len(p2_ids), len(s_ids), gold))
            # Also add gold suffix-prediction row to `s1_atomic_seqs`
            g_pref = pref + (o1 if gold == 0 else o2)
            ids_wg, w_wg = encode_plain_continuation(tok, eos_id, g_pref, suf, W_MC_GOLD)
            if 8 <= len(ids_wg) <= 256 and is_bpe_seq_clean(ids_wg):
                s1_atomic_seqs.append((ids_wg, w_wg))
                stats["s1_academic_items"] += 1

    # Save pre-tokenized `mc_contrastive_{train,val}.npz` (shared prompt up to 512 tokens, 4 choices up to 64 tokens)
    rng.shuffle(mc_items)
    MAX_P_TOK = 512
    MAX_C_TOK = 64
    p_arr_list, plen_list, c_arr_list, clen_list, cbytes_list, nch_list, gold_list = [], [], [], [], [], [], []
    for ctx_s, ch_s, g_i in mc_items:
        p_ids = tok.encode(ctx_s, add_special_tokens=False)
        if len(p_ids) < 3 or len(p_ids) > MAX_P_TOK:
            continue
        c_toks = [tok.encode(c, add_special_tokens=False) for c in ch_s]
        if any(len(ct) == 0 or len(ct) > MAX_C_TOK for ct in c_toks):
            continue
        p_row = np.full((MAX_P_TOK,), eos_id, dtype=np.uint16)
        p_row[: len(p_ids)] = p_ids
        c_row = np.full((4, MAX_C_TOK), eos_id, dtype=np.uint16)
        cl_row = np.zeros((4,), dtype=np.int16)
        cb_row = np.ones((4,), dtype=np.float32)
        for j, (ct, cs_raw) in enumerate(zip(c_toks, ch_s)):
            c_row[j, : len(ct)] = ct
            cl_row[j] = len(ct)
            cb_row[j] = float(max(1, len(cs_raw.encode("utf-8"))))
        p_arr_list.append(p_row)
        plen_list.append(len(p_ids))
        c_arr_list.append(c_row)
        clen_list.append(cl_row)
        cbytes_list.append(cb_row)
        nch_list.append(len(ch_s))
        gold_list.append(g_i)

    stats["mc_tuples_kept"] = len(p_arr_list)
    n_mc_val = min(600, max(100, len(p_arr_list) // 25))
    for split_name, sl in [("val", slice(0, n_mc_val)), ("train", slice(n_mc_val, None))]:
        np.savez_compressed(
            os.path.join(out_dir, f"mc_contrastive_{split_name}.npz"),
            prompt_ids=np.stack(p_arr_list[sl], axis=0),
            prompt_lens=np.asarray(plen_list[sl], dtype=np.int16),
            choice_ids=np.stack(c_arr_list[sl], axis=0),
            choice_lens=np.stack(clen_list[sl], axis=0),
            choice_bytes=np.stack(cbytes_list[sl], axis=0),
            n_choices=np.asarray(nch_list[sl], dtype=np.int8),
            gold_idx=np.asarray(gold_list[sl], dtype=np.int8),
        )

    # Save `wg_contrastive_{train,val}.npz`
    rng.shuffle(wg_pairs)
    MAX_WG_TOK = 96
    wg_seq = np.full((len(wg_pairs), 2, MAX_WG_TOK), eos_id, dtype=np.uint16)
    wg_s_start = np.zeros((len(wg_pairs), 2), dtype=np.int16)
    wg_s_len = np.zeros((len(wg_pairs), 2), dtype=np.int16)
    wg_gold = np.zeros((len(wg_pairs),), dtype=np.int8)
    for idx_i, (sq1, sq2, st1, ln1, st2, ln2, g_w) in enumerate(wg_pairs):
        wg_seq[idx_i, 0, : len(sq1)] = sq1
        wg_seq[idx_i, 1, : len(sq2)] = sq2
        wg_s_start[idx_i, 0] = st1
        wg_s_start[idx_i, 1] = st2
        wg_s_len[idx_i, 0] = ln1
        wg_s_len[idx_i, 1] = ln2
        wg_gold[idx_i] = g_w
    stats["wg_pairs_kept"] = len(wg_pairs)
    n_wg_val = min(300, max(64, len(wg_pairs) // 20))
    for split_name, sl in [("val", slice(0, n_wg_val)), ("train", slice(n_wg_val, None))]:
        np.savez_compressed(
            os.path.join(out_dir, f"wg_contrastive_{split_name}.npz"),
            seq_ids=wg_seq[sl],
            suf_start=wg_s_start[sl],
            suf_len=wg_s_len[sl],
            gold_idx=wg_gold[sl],
        )
    print(
        f"      -> MC Contrastive: {stats['mc_tuples_kept']:,} tuples | WG Contrastive: {stats['wg_pairs_kept']:,} pairs "
        f"| S1 Academic Seqs: {len(s1_atomic_seqs):,}",
        flush=True,
    )

    # ==========================================================================
    # 2. STREAM 3 (`<think>` MATH + DIRECT CONCEPTUAL PROSE) & STREAM 4 (CODE)
    # ==========================================================================
    print("[2/5] Building Stream 3 (<think> Math + Direct Conceptual Prose) & Stream 4 (Polyglot Code)...", flush=True)
    s3_atomic_seqs: List[Tuple[List[int], List[int]]] = []
    s4_atomic_seqs: List[Tuple[List[int], List[int]]] = []

    # 2A. `<think>` Quantitative Math (`gsm8k train` + `MetaMathQA` + place-value decomposed arithmetic)
    ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds_gsm:
        q = r["question"].strip()
        steps, final_ans = parse_metamath_cot(r["answer"])
        ok_m, _ = guard.check_text(q + " " + steps + " " + final_ans, stem_texts=[q], check_poison=True)
        if not ok_m:
            stats["rejected_decontam"] += 1
            continue
        ids, w = builder.encode_chatml_turn(
            q, f"<think>\n{steps}\n</think>\n{final_ans}", sft_mode=True, high_salience_suffix=final_ans
        )
        if 16 <= len(ids) <= 1800:
            s3_atomic_seqs.append((ids, w))
            stats["s3_math_items"] += 1

    mm_path = os.path.join(raw_dir, "meta-math__MetaMathQA", "MetaMathQA-395K.json")
    if os.path.exists(mm_path):
        with open(mm_path, "r", encoding="utf-8") as f:
            mm_rows = json.load(f)
        rng.shuffle(mm_rows)
        mm_added = 0
        for r in mm_rows:
            if mm_added >= 10000:
                break
            q = (r.get("query") or "").strip()
            resp = (r.get("response") or "").strip()
            if not q or not resp or len(resp) > 1600:
                continue
            ok_pair, _ = is_clean_pair(q, resp)
            if not ok_pair:
                continue
            ok_m, _ = guard.check_text(q + " " + resp, stem_texts=[q], check_poison=True)
            if not ok_m:
                stats["rejected_decontam"] += 1
                continue
            steps, final_ans = parse_metamath_cot(resp)
            ids, w = builder.encode_chatml_turn(
                q, f"<think>\n{steps}\n</think>\n{final_ans}", sft_mode=True, high_salience_suffix=final_ans
            )
            if 16 <= len(ids) <= 1800:
                s3_atomic_seqs.append((ids, w))
                stats["s3_math_items"] += 1
                mm_added += 1

    for _ in range(3200):
        q_ar, st_ar, fa_ar = generate_decomposed_math_turn(rng)
        ok_m, _ = guard.check_text(q_ar + " " + st_ar + " " + fa_ar, check_poison=True)
        if ok_m:
            ids, w = builder.encode_chatml_turn(
                q_ar, f"<think>\n{st_ar}\n</think>\n{fa_ar}", sft_mode=True, high_salience_suffix=fa_ar
            )
            if 16 <= len(ids) <= 1024:
                s3_atomic_seqs.append((ids, w))
                stats["s3_math_items"] += 1

    # 2B. Strictly `<think>`-free Direct Conceptual ChatML (`smoltalk` + `CS_CONCEPT_TOPICS` + `sciq` concepts)
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    smoltalk_arrows = sorted(
        glob.glob(os.path.join(hf_cache, "datasets/HuggingFaceTB___smoltalk/**/*train-*.arrow"), recursive=True)
    )
    allowed_cats = {"information-seeking", "reasoning", "advice-seeking", "planning", "data-analysis"}
    smol_added = 0
    for ap in smoltalk_arrows:
        if smol_added >= 7500:
            break
        with ipc.open_stream(ap) as reader:
            tbl = reader.read_all()
            for msgs, cat, qual in zip(
                tbl["messages"].to_pylist(), tbl["category"].to_pylist(), tbl["quality"].to_pylist()
            ):
                if smol_added >= 7500:
                    break
                if qual not in ("excellent", "good") or cat not in allowed_cats or not msgs or len(msgs) < 2:
                    continue
                u_msg = (msgs[0].get("content") or "").strip() if msgs[0].get("role") == "user" else ""
                a_msg = (msgs[1].get("content") or "").strip() if msgs[1].get("role") == "assistant" else ""
                if not u_msg or not a_msg or len(a_msg) > 1800 or "```" in a_msg or "<think>" in a_msg:
                    continue
                ok_p, _ = is_clean_pair(u_msg, a_msg)
                if not ok_p:
                    continue
                ok_g, _ = guard.check_text(u_msg + " " + a_msg, stem_texts=[u_msg], check_poison=True)
                if not ok_g:
                    stats["rejected_decontam"] += 1
                    continue
                ids, w = builder.encode_chatml_turn(u_msg, a_msg, sft_mode=True)
                if 20 <= len(ids) <= 1850:
                    s3_atomic_seqs.append((ids, w))
                    stats["s3_concept_items"] += 1
                    smol_added += 1

    for _ in range(1800):
        u_c, a_c = generate_conceptual_prose_turn(rng)
        ok_g, _ = guard.check_text(u_c + " " + a_c, check_poison=True)
        if ok_g:
            ids, w = builder.encode_chatml_turn(u_c, a_c, sft_mode=True)
            s3_atomic_seqs.append((ids, w))
            stats["s3_concept_items"] += 1

    for u_sc, a_sc in sciq_concept_pairs:
        ids, w = builder.encode_chatml_turn(u_sc, a_sc, sft_mode=True)
        if 16 <= len(ids) <= 1200:
            s3_atomic_seqs.append((ids, w))
            stats["s3_concept_items"] += 1

    # 2C. Stream 4: Polyglot Code (`Magicoder` + `CodeFeedback` + `python_alpaca` + `.bb` + procedural C99/C++20/Py/Lua)
    magi_path = os.path.join(raw_dir, "ise-uiuc__Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl")
    if os.path.exists(magi_path):
        with open(magi_path, "r", encoding="utf-8") as f:
            magi_lines = f.readlines()
        rng.shuffle(magi_lines)
        m_cnt = 0
        for ln in magi_lines:
            if m_cnt >= 5500:
                break
            r = json.loads(ln)
            pr = (r.get("problem") or "").strip()
            so = (r.get("solution") or "").strip()
            if len(so) > 2600 or "<think>" in so:
                continue
            ok_p, _ = is_clean_pair(pr, so)
            if not ok_p:
                continue
            ok_g, _ = guard.check_text(pr + " " + so, stem_texts=[pr], check_poison=True)
            if not ok_g:
                stats["rejected_decontam"] += 1
                continue
            ids, w = builder.encode_chatml_turn(pr, so, sft_mode=True)
            if 24 <= len(ids) <= 1950:
                s4_atomic_seqs.append((ids, w))
                stats["s4_code_items"] += 1
                m_cnt += 1

    cf_path = os.path.join(raw_dir, "m-a-p__CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl")
    if os.path.exists(cf_path):
        with open(cf_path, "r", encoding="utf-8") as f:
            cf_lines = f.readlines()
        rng.shuffle(cf_lines)
        cf_cnt = 0
        for ln in cf_lines:
            if cf_cnt >= 3500:
                break
            r = json.loads(ln)
            pr = (r.get("query") or "").strip()
            so = (r.get("answer") or "").strip()
            if len(so) > 2600 or "<think>" in so:
                continue
            ok_p, _ = is_clean_pair(pr, so)
            if not ok_p:
                continue
            ok_g, _ = guard.check_text(pr + " " + so, stem_texts=[pr], check_poison=True)
            if not ok_g:
                stats["rejected_decontam"] += 1
                continue
            ids, w = builder.encode_chatml_turn(pr, so, sft_mode=True)
            if 24 <= len(ids) <= 1950:
                s4_atomic_seqs.append((ids, w))
                stats["s4_code_items"] += 1
                cf_cnt += 1

    ds_py_alp = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    py_indices = list(range(len(ds_py_alp)))
    rng.shuffle(py_indices)
    py_cnt = 0
    for idx_py in py_indices:
        if py_cnt >= 1800:
            break
        r = ds_py_alp[idx_py]
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out_py = (r.get("output") or "").strip()
        if not instr or not out_py or len(out_py) > 2200 or "<think>" in out_py:
            continue
        pr = f"{instr}\n\nInput:\n{inp}" if inp else instr
        so = out_py if "```" in out_py else f"```python\n{out_py}\n```"
        ok_p, _ = is_clean_pair(pr, so)
        if not ok_p:
            continue
        ok_g, _ = guard.check_text(pr + " " + so, stem_texts=[instr], check_poison=True)
        if not ok_g:
            stats["rejected_decontam"] += 1
            continue
        ids, w = builder.encode_chatml_turn(pr, so, sft_mode=True)
        if 20 <= len(ids) <= 1850:
            s4_atomic_seqs.append((ids, w))
            stats["s4_code_items"] += 1
            py_cnt += 1

    bb_files = sorted(
        set(
            glob.glob(os.path.join(base_dir, "..", "buildbit", "**", "*.bb"), recursive=True)
            + glob.glob(os.path.join(base_dir, "..", "buildbit-os", "**", "*.bb"), recursive=True)
        )
    )
    for bf in bb_files:
        if os.path.isfile(bf):
            with open(bf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if 80 <= len(content) <= 4500 and "soa struct photon" not in content.lower():
                doc_bb = f"<file_sep>{os.path.basename(bf)}\n{content}"
                ok_bb, _ = guard.check_text(doc_bb)
                if ok_bb:
                    ids_bb = tok.encode(doc_bb, add_special_tokens=False) + [eos_id]
                    if 24 <= len(ids_bb) <= 1900:
                        w_bb = [W_PROSE] * len(ids_bb)
                        for _ in range(4):
                            s4_atomic_seqs.append((list(ids_bb), list(w_bb)))
                            stats["s4_code_items"] += 1

    for _ in range(4200):
        pr_p, co_p = generate_prompt_bound_polyglot_v2_2(rng, lua_open_stems)
        ok_p, _ = is_clean_pair(pr_p, co_p)
        if not ok_p:
            continue
        ok_g, _ = guard.check_text(pr_p + " " + co_p, check_poison=True)
        if ok_g:
            ids, w = builder.encode_chatml_turn(pr_p, co_p, sft_mode=True)
            if 20 <= len(ids) <= 1800:
                s4_atomic_seqs.append((ids, w))
                stats["s4_code_items"] += 1

    # Pack `sft_skill_{train,val}` with balanced interleaving across S1 (Academic), S3 (Math+Concept), S4 (Code)
    rng.shuffle(s1_atomic_seqs)
    rng.shuffle(s3_atomic_seqs)
    rng.shuffle(s4_atomic_seqs)
    raw_skill_seqs = s1_atomic_seqs + s3_atomic_seqs + s4_atomic_seqs
    all_skill_seqs: List[Tuple[List[int], List[int]]] = []
    for st_b in range(0, len(raw_skill_seqs), 512):
        chunk_sq = raw_skill_seqs[st_b : st_b + 512]
        dec_batch = tok.batch_decode([x[0] for x in chunk_sq], skip_special_tokens=True)
        for sq_item, dec_s in zip(chunk_sq, dec_batch):
            ok_sq, _ = guard.check_text(dec_s)
            if ok_sq:
                all_skill_seqs.append(sq_item)
            else:
                stats["rejected_decontam"] += 1
    rng.shuffle(all_skill_seqs)

    skill_packer = FlexRowPacker(eos_id, seq_len=SEQ_LEN)
    for ids, w in all_skill_seqs:
        skill_packer.add_atomic(ids, w)
    sk_toks, sk_msks = skill_packer.finalize()
    perm_sk = np_rng.permutation(len(sk_toks))
    sk_toks, sk_msks = sk_toks[perm_sk], sk_msks[perm_sk]
    n_sk_val = min(300, max(100, len(sk_toks) // 40))
    save_memmaps_flex(out_dir, "sft_skill_val", sk_toks[:n_sk_val], sk_msks[:n_sk_val])
    save_memmaps_flex(out_dir, "sft_skill_train", sk_toks[n_sk_val:], sk_msks[n_sk_val:])

    pr2_train_path = os.path.join(out_dir, "prose_climax_train_tokens.bin")
    pr4_train_path = os.path.join(out_dir, "prose_climax_4k_train_tokens.bin")
    n2_train_path = os.path.join(out_dir, "niah_2k_train_tokens.bin")
    n4_train_path = os.path.join(out_dir, "niah_4k_train_tokens.bin")
    can_reuse_prose_niah = args.rebuild_sft_only and all(
        os.path.exists(p) for p in (pr2_train_path, pr4_train_path, n2_train_path, n4_train_path)
    )

    if can_reuse_prose_niah:
        print("[3-4/5] --rebuild_sft_only: Reusing existing prose_climax & niah shards (will re-audit ALL files in [5/5])...", flush=True)
        pr2_train_n = os.path.getsize(pr2_train_path) // (2049 * 2)
        pr4_train_n = os.path.getsize(pr4_train_path) // (4097 * 2)
        n2_train_n = os.path.getsize(n2_train_path) // (2049 * 2)
        n4_train_n = os.path.getsize(n4_train_path) // (4097 * 2)
    else:
        # ==========================================================================
        # 3. STREAM 2: UNSEEN TEXTBOOK & TERMINAL-ENTITY CLIMAX REPLAY (2K & 4K)
        # ==========================================================================
        print("[3/5] Building Stream 2 Unseen Prose + Terminal-Entity Climax Salience (2K & 4K)...", flush=True)
        word_starts: Set[int] = set()
        alpha_tokens: Set[int] = set()
        punct_end_tokens: Set[int] = set()
        for raw_t, tid in tok.get_vocab().items():
            clean_t = raw_t.lstrip("ĠĊ")
            if raw_t.startswith("Ġ"):
                word_starts.add(tid)
            if clean_t and clean_t.isalpha() and clean_t.isascii():
                alpha_tokens.add(tid)
            if any(ch in raw_t for ch in (".", "!", "?", '"', "Ċ")):
                punct_end_tokens.add(tid)

        cosmo_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "cosmopedia-v2", "train-00000-of-00104.parquet")
        fw_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "fineweb-edu-dedup", "train-00000-of-00234.parquet")
        pf_cosmo = pq.ParquetFile(cosmo_path)
        pf_fw = pq.ParquetFile(fw_path)

        rg_sources: List[Tuple[str, int]] = []
        for rg in range(max(75, int(0.80 * pf_cosmo.num_row_groups)), pf_cosmo.num_row_groups):
            rg_sources.append(("cosmo", rg))
        for rg in range(max(75, int(0.80 * pf_fw.num_row_groups)), pf_fw.num_row_groups):
            rg_sources.append(("fw", rg))
        rng.shuffle(rg_sources)

        ds_ts = load_dataset("roneneldan/tiny_stories", split="train")
        ts_cursor = 250000

        prose_2k_packer = FlexRowPacker(eos_id, seq_len=2048)
        prose_4k_packer = FlexRowPacker(eos_id, seq_len=4096)
        target_2k_rows = 8200
        target_4k_rows = 1260

        for src_kind, rg_idx in rg_sources:
            if len(prose_2k_packer.rows_tokens) >= target_2k_rows and len(prose_4k_packer.rows_tokens) >= target_4k_rows:
                break
            pf = pf_cosmo if src_kind == "cosmo" else pf_fw
            texts = pf.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()

            n_ts_blend = max(1, len(texts) // 4)
            if ts_cursor + n_ts_blend < len(ds_ts):
                texts.extend(ds_ts[ts_cursor : ts_cursor + n_ts_blend]["text"])
                ts_cursor += n_ts_blend
            rng.shuffle(texts)

            clean_docs: List[str] = []
            for tx in texts:
                if not tx or len(tx) < 260:
                    continue
                ok_g, _ = guard.check_text(tx)
                if not ok_g:
                    stats["rejected_decontam"] += 1
                    continue
                clean_docs.append(tx)
                if len(clean_docs) >= 3600:
                    break

            for i in range(0, len(clean_docs), 512):
                if len(prose_2k_packer.rows_tokens) >= target_2k_rows and len(prose_4k_packer.rows_tokens) >= target_4k_rows:
                    break
                enc_batch = tok(clean_docs[i : i + 512], add_special_tokens=False)["input_ids"]
                for doc_idx, ids in enumerate(enc_batch):
                    if len(ids) < 80:
                        continue
                    w_climax = build_climax_salience_mask(ids, word_starts, alpha_tokens, punct_end_tokens)
                    if doc_idx % 5 == 0 and len(prose_4k_packer.rows_tokens) < target_4k_rows:
                        prose_4k_packer.add_continuous(ids, w_climax)
                    elif len(prose_2k_packer.rows_tokens) < target_2k_rows:
                        prose_2k_packer.add_continuous(ids, w_climax)

        pr2_t, pr2_m = prose_2k_packer.finalize()
        perm_p2 = np_rng.permutation(len(pr2_t))
        pr2_t, pr2_m = pr2_t[perm_p2], pr2_m[perm_p2]
        save_memmaps_flex(out_dir, "prose_climax_val", pr2_t[:200], pr2_m[:200])
        save_memmaps_flex(out_dir, "prose_climax_train", pr2_t[200:], pr2_m[200:])
        pr2_train_n = len(pr2_t) - 200

        pr4_t, pr4_m = prose_4k_packer.finalize()
        perm_p4 = np_rng.permutation(len(pr4_t))
        pr4_t, pr4_m = pr4_t[perm_p4], pr4_m[perm_p4]
        save_memmaps_flex(out_dir, "prose_climax_4k_val", pr4_t[:60], pr4_m[:60])
        save_memmaps_flex(out_dir, "prose_climax_4k_train", pr4_t[60:], pr4_m[60:])
        pr4_train_n = len(pr4_t) - 60

        # ==========================================================================
        # 4. STREAM 5: LONG-SPAN (`1..11` SUBWORDS) DUAL-FORMAT NIAH (`2K` & `4K`)
        # ==========================================================================
        print("[4/5] Synthesizing Long-Sibling (`8..11`-Subword) Dual-Format NIAH (`2K` & `4K`)...", flush=True)

        def build_niah_split(n_target: int, seq_l: int, target_4k: bool) -> Tuple[np.ndarray, np.ndarray]:
            row_l = seq_l + 1
            min_l = 2100 if target_4k else 420
            rows_t, rows_m = [], []
            while len(rows_t) < n_target:
                use_chatml = rng.random() < 0.55
                prm, queried, ex0, s_dir = synthesize_long_sibling_niah_v2_5(rng, tok, stem_pool, target_4k=target_4k)
                ok_n, _ = guard.check_text(prm)
                if not ok_n:
                    stats["rejected_decontam"] += 1
                    continue
                if use_chatml:
                    ids, w = encode_open_niah_turn(builder, prm, queried, single_direct=s_dir, sft_mode=True)
                else:
                    ids, w = encode_plaintext_niah_row(tok, eos_id, prm, queried, ex0, rng)
                if min_l <= len(ids) <= row_l:
                    pad_n = row_l - len(ids)
                    if pad_n > 0:
                        ids = ids + [eos_id] * pad_n
                        w = w + [W_IGNORE] * pad_n
                    arr_t = np.asarray(ids, dtype=np.uint16)
                    arr_m = np.asarray(w[1:], dtype=np.uint8)
                    if (arr_m > 0).sum() >= 8:
                        rows_t.append(arr_t)
                        rows_m.append(arr_m)
            t_out = np.stack(rows_t, axis=0)
            m_out = np.stack(rows_m, axis=0)
            perm = np_rng.permutation(len(t_out))
            return t_out[perm], m_out[perm]

        n2_t, n2_m = build_niah_split(4700, seq_l=2048, target_4k=False)
        save_memmaps_flex(out_dir, "niah_2k_val", n2_t[:200], n2_m[:200])
        save_memmaps_flex(out_dir, "niah_2k_train", n2_t[200:], n2_m[200:])
        n2_train_n = len(n2_t) - 200

        n4_t, n4_m = build_niah_split(1300, seq_l=4096, target_4k=True)
        save_memmaps_flex(out_dir, "niah_4k_val", n4_t[:100], n4_m[:100])
        save_memmaps_flex(out_dir, "niah_4k_train", n4_t[100:], n4_m[100:])
        n4_train_n = len(n4_t) - 100

    # Also build `sft_skill_4k_{train,val}` (`1,260` packed `4,096`-token skill rows for Stage-2 Horizon Cooldown!)
    skill_4k_packer = FlexRowPacker(eos_id, seq_len=4096)
    rng.shuffle(all_skill_seqs)
    for ids, w in all_skill_seqs:
        if len(skill_4k_packer.rows_tokens) >= 1260:
            break
        skill_4k_packer.add_atomic(ids, w)
    sk4_t, sk4_m = skill_4k_packer.finalize()
    perm_sk4 = np_rng.permutation(len(sk4_t))
    sk4_t, sk4_m = sk4_t[perm_sk4], sk4_m[perm_sk4]
    save_memmaps_flex(out_dir, "sft_skill_4k_val", sk4_t[:60], sk4_m[:60])
    save_memmaps_flex(out_dir, "sft_skill_4k_train", sk4_t[60:], sk4_m[60:])

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_sec": round(time.perf_counter() - t0, 2),
        "stats": stats,
        "splits": {
            "sft_skill_train_2k": int(len(sk_toks) - n_sk_val),
            "sft_skill_4k_train": int(len(sk4_t) - 60),
            "prose_climax_train_2k": int(pr2_train_n),
            "prose_climax_4k_train": int(pr4_train_n),
            "niah_2k_train": int(n2_train_n),
            "niah_4k_train": int(n4_train_n),
            "mc_contrastive_train": int(stats["mc_tuples_kept"] - n_mc_val),
            "wg_contrastive_train": int(stats["wg_pairs_kept"] - n_wg_val),
        },
    }
    with open(os.path.join(out_dir, "manifest_v2_5.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ==========================================================================
    # 5. INDEPENDENT POST-GENERATION DECONTAMINATION AUDIT (HARD 0-LEAK ASSERT!)
    # ==========================================================================
    print("[5/5] Running Independent Post-Generation Decontamination Verification...", flush=True)
    audit_path = os.path.join(base_dir, "benchmarks", "decontamination_audit_v2_5.json")
    audit_data_v2_5_directory(out_dir, audit_path)
    print(f"[DONE] VAULT v2.5 Dataset Ready & Verified Clean in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
