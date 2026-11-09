#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT-1B (`1,029.27M` Parameters = `931.95M` Dense + `97.32M` Persistent Cortical Prior Vaults):
# Infinite-Stream Pretraining & Associative Memory Consolidation Engine
#
# Architecture & Memory Consolidation Highlights:
#   - Backbone: 27 layers, d_model=1536, n_q_heads=24, n_vault_groups=4 (6:1 GVA), head_dim=64,
#     d_ff=4096, chunk_size=64, Co-Prime Kenyon banks P=(47, 53, 59, 61) (220 cells/group,
#     zero-shot Mitosis expandable to 508 -> 942 -> 2,002 cells).
#   - Multi-Timescale GTRM: Geometric per-layer `ema_alpha_schedule` spanning `2^-10` (Layer 0)
#     down to `2^-16` (Layer 26, ~65K characteristic token horizon) to eliminate 64K+ decay ceilings.
#   - Zero-Ghost-Weight Cortical Consolidation: Uses VAULT's own associative memory state (`S_stream`
#     + persistent `s_prior`) with `dS_init` gradient updates, slow hippocampal trace EMA, 25% cold-start
#     prior dropout, and periodic blank-vault Sleep-Consolidation KL pulses (`forward_sleep_kl_loss`).
#   - Balanced Frontier Infinite Stream (`streaming=True` from HuggingFace Hub, zero disk bloat):
#       55% FineWeb-Edu (`sample-100BT`)
#       20% Cosmopedia-v2 (`smollm-corpus`)
#       15% Clean Code (`codeparrot-clean` + `Magicoder-OSS-Instruct-75K` polyglot)
#        7% FineMath (`finemath-4plus`)
#        3% Procedural Multi-Chunk Long-Range NIAH (`2.4K-6.4K` tokens) & Quantitative Derivations

import argparse
import gc
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

# Guard against ROCm 7.2 RDNA4 (`gfx1201`) `expandable_segments` allocator bug before importing torch
if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vault_v2_5_model import VAULTLMv2_5, build_geometric_ema_schedule
from src.vault_v2_5_streaming import VAULTv2_5StreamingEngine


# ==============================================================================
# 1. STRICT DECONTAMINATION: HELD-OUT EVALUATION STEMS & PROCEDURAL SYNTHESIS
# ==============================================================================

# Strictly held-out evaluation prefixes (never generated in procedural training)
FORBIDDEN_EVAL_SUBSTRINGS = (
    "shieldpod",
    "cryolock",
    "warpcore",
    "ionthruster",
    "plasmagrid",
    "startracker",
    "beaconmux",
    "gravsensor",
    "solarwing",
    "hullradar",
    "thermalsink",
    "dockclamp",
    "quasar-lock",
    "pulsar-grid",
    "zenith-core",
)

TRAIN_ENTITY_DOMAINS = [
    ("ChronoShard", "ReplicaNorth", "LSN"),
    ("ChronoShard", "ReplicaSouth", "LSN"),
    ("AeroTurbine", "StatorBay", "CAL"),
    ("AeroTurbine", "RotorBay", "CAL"),
    ("BioSample", "CryoRack", "LOT"),
    ("BioSample", "AssayPlate", "LOT"),
    ("NetFabric", "SpineLeaf", "VLAN"),
    ("NetFabric", "Edgeplink", "VLAN"),
    ("CargoLedger", "BerthWest", "SEAL"),
    ("CargoLedger", "BerthEast", "SEAL"),
    ("CompilerSym", "ModuleIR", "ADDR"),
    ("CompilerSym", "LinkerGOT", "ADDR"),
    ("GeoArray", "Borehole", "PIN"),
    ("PhotonBus", "Waveguide", "KEY"),
    ("HydroValve", "Manifold", "TAG"),
    ("KubeNode", "WorkerPool", "HASH"),
]

TRAIN_FILLER_CLAUSES = [
    "Periodic parity scrub completed across all redundant lanes with zero uncorrectable symbol errors.",
    "Upstream backpressure remained below threshold during bursty asynchronous batch commits.",
    "Thermal equilibrium stabilized within nominal millikelvin drift bounds across adjacent sensors.",
    "Deterministic replay checksums matched the primary write-ahead log across three quorum peers.",
    "Pipeline hazard interlocks cleared all speculative store buffers prior to epoch boundary rollover.",
    "Calibration polynomials converged in four Gauss-Newton iterations without residual outliers.",
    "Packet pacing timers maintained strict FIFO ordering across cross-rack optical transceivers.",
    "Memory controller refresh scheduler interleaved bank activations without stalling read queues.",
]


def generate_procedural_document(rng: random.Random) -> str:
    """
    Generates either:
      (a) A multi-chunk (`~2,400 - 6,000` token) open-vocabulary associative manifest containing
          `36..85` keyed records (including sibling identifiers sharing `6..9` BPE prefix tokens
          and variable `2..36`-token syntax lags), followed by 50% plain-text / 50% ChatML
          multi-needle extraction spanning across `2..3` consecutive `2,048`-token chunks; OR
      (b) A clean step-by-step quantitative / symbolic derivation.
    Zero overlap with `FORBIDDEN_EVAL_SUBSTRINGS`.
    """
    mode = rng.random()
    if mode < 0.70:
        n_records = rng.randint(38, 86)  # ~2,500 to ~5,800 tokens -> spans 2 to 3+ chunks!
        use_chatml = rng.random() < 0.50
        header_titles = [
            "DISTRIBUTED STATE & AUDIT MANIFEST",
            "FACILITY INSTRUMENTATION REGISTRY",
            "CLUSTER CONFIGURATION & CHECKSUM LEDGER",
            "LABORATORY BATCH TRACEABILITY LOG",
        ]
        lines = [rng.choice(header_titles)]
        records: List[Tuple[str, str, str]] = []
        used_ids = set()
        used_vals = set()

        idx = 0
        while idx < n_records:
            dom, sub, v_prefix = rng.choice(TRAIN_ENTITY_DOMAINS)
            # 35% chance to emit an adversarial sibling pair sharing identical 7+ subword prefix!
            make_sibling = (rng.random() < 0.35) and (idx + 1 < n_records)
            count_here = 2 if make_sibling else 1
            shared_cluster = rng.randint(10, 99)
            for _ in range(count_here):
                for _attempt in range(20):
                    unit_num = rng.randint(1000, 9999)
                    ent_id = f"{dom}-{sub}{shared_cluster}-Unit{unit_num}"
                    if ent_id not in used_ids:
                        used_ids.add(ent_id)
                        break
                for _attempt in range(20):
                    if rng.random() < 0.5:
                        val = f"{v_prefix}-{rng.randint(10000, 99999)}"
                    else:
                        hex_tag = "".join(rng.choices("0123456789ABCDEF", k=6))
                        val = f"{v_prefix}-{hex_tag}"
                    if val not in used_vals:
                        used_vals.add(val)
                        break

                records.append((ent_id, v_prefix, val))
                gap_style = rng.randint(0, 2)
                f1 = rng.choice(TRAIN_FILLER_CLAUSES)
                f2 = rng.choice(TRAIN_FILLER_CLAUSES)
                if gap_style == 0:
                    entry = f"[Record {idx + 1:03d} | {ent_id}] Status: ACTIVE | Assigned Token: {val}\n  Notes: {f1}"
                elif gap_style == 1:
                    entry = (
                        f"Section {idx + 1:03d} — Identifier: {ent_id} (health=OK, epoch={rng.randint(100, 999)}, "
                        f"latency={rng.randint(2, 48)}us) -> Committed Reference: {val}.\n  Detail: {f1} {f2}"
                    )
                else:
                    entry = (
                        f"* Component `{ent_id}` passed automated preflight diagnostics across {rng.randint(3, 12)} "
                        f"channels without contention; authoritative token recorded as `{val}`.\n  Audit: {f1}"
                    )
                lines.append(entry)
                idx += 1

        # Pick 2 to 4 needles weighted toward the first half of the manifest so cross-chunk recall is exercised!
        early_pool = records[: max(4, len(records) // 2)]
        all_pool = records
        k_needles = rng.randint(2, 4)
        chosen = [rng.choice(early_pool)]
        for _ in range(k_needles - 1):
            cand = rng.choice(all_pool)
            if cand not in chosen:
                chosen.append(cand)

        target_names = ", ".join(c[0] for c in chosen)
        manifest_body = "\n".join(lines)
        if use_chatml:
            ans_lines = "\n".join(f"- {c[0]}: {c[2]}" for c in chosen)
            doc = (
                f"<|im_start|>user\n{manifest_body}\n\n"
                f"Question: Extract the exact committed token for each of the following identifiers: "
                f"{target_names}.<|im_end|>\n"
                f"<|im_start|>assistant\n{ans_lines}<|im_end|>"
            )
        else:
            ans_lines = "; ".join(f"{c[0]} = {c[2]}" for c in chosen)
            doc = (
                f"{manifest_body}\n\n"
                f"Summary Query ({target_names}):\n"
                f"Verified Tokens: {ans_lines}."
            )
        low = doc.lower()
        assert not any(bad in low for bad in FORBIDDEN_EVAL_SUBSTRINGS)
        return doc

    # Quantitative / multi-step symbolic derivation (50% plain textbook worked example, 50% ChatML <think>)
    a = rng.randint(18, 490)
    b = rng.randint(12, 96)
    c = rng.randint(15, 750)
    b_tens, b_ones = (b // 10) * 10, b % 10
    p1 = a * b_tens
    p2 = a * b_ones
    prod = p1 + p2
    total = prod + c
    if rng.random() < 0.50:
        return (
            f"Worked Problem: Evaluate ({a} * {b}) + {c} step by step using place-value decomposition.\n"
            f"Solution:\n"
            f"1. Split {b} into {b_tens} + {b_ones}.\n"
            f"2. Multiply {a} * {b_tens} = {p1}, and {a} * {b_ones} = {p2}.\n"
            f"3. Sum partial products: {p1} + {p2} = {prod}.\n"
            f"4. Add the offset {c}: {prod} + {c} = {total}.\n"
            f"Answer: {total}"
        )
    return (
        f"<|im_start|>user\nCompute ({a} * {b}) + {c} exactly.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
        f"Decompose {b} = {b_tens} + {b_ones}.\n"
        f"{a} * {b_tens} = {p1}\n"
        f"{a} * {b_ones} = {p2}\n"
        f"{p1} + {p2} = {prod}\n"
        f"{prod} + {c} = {total}\n"
        f"</think>\nThe exact result is {total}.<|im_end|>"
    )


# ==============================================================================
# 2. INFINITE HUGGINGFACE STREAMING + STATEFUL MULTI-CHUNK PACKER
# ==============================================================================

STREAM_SPECS = [
    # (name, weight, hf_repo, hf_config, formatter)
    ("fineweb_edu", 0.55, "HuggingFaceFW/fineweb-edu", "sample-100BT", lambda r: r.get("text", "")),
    ("cosmopedia_v2", 0.20, "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", lambda r: r.get("text", "")),
    ("codeparrot", 0.11, "codeparrot/codeparrot-clean", None, lambda r: r.get("content", "")),
    (
        "magicoder",
        0.04,
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        None,
        lambda r: f"Problem ({r.get('lang', 'code')}):\n{r.get('problem', '').strip()}\n\nSolution:\n{r.get('solution', '').strip()}",
    ),
    ("finemath", 0.07, "HuggingFaceTB/finemath", "finemath-4plus", lambda r: r.get("text", "")),
]
PROCEDURAL_WEIGHT = 0.03


def make_resilient_hf_iterator(
    repo_id: str,
    config_name: Optional[str],
    formatter: Callable[[dict], str],
    initial_skip: int = 0,
) -> Iterator[str]:
    """
    Streams documents infinitely from HuggingFace Hub (`streaming=True`) with zero local disk bloat.
    Automatically reconnects with exponential backoff on transient HTTP/socket timeouts and resumes
    at the exact `seen_docs` offset, wrapping around cleanly if an epoch completes.
    """
    seen_docs = int(initial_skip)
    backoff = 2.0
    while True:
        try:
            if config_name is not None:
                ds = load_dataset(repo_id, config_name, split="train", streaming=True)
            else:
                ds = load_dataset(repo_id, split="train", streaming=True)
            if seen_docs > 0:
                ds = ds.skip(seen_docs)
            YieldedAny = False
            for row in ds:
                txt = formatter(row)
                seen_docs += 1
                backoff = 2.0
                YieldedAny = True
                if isinstance(txt, str) and len(txt.strip()) >= 64:
                    low_head = txt[:512].lower()
                    if not any(bad in low_head for bad in FORBIDDEN_EVAL_SUBSTRINGS):
                        yield txt
            # Reached end of finite stream (e.g. Magicoder 75K) -> wrap to next epoch!
            seen_docs = 0
            if not YieldedAny:
                time.sleep(1.0)
        except Exception as exc:
            print(
                f"[Stream Warning] Transient error on {repo_id} (at doc {seen_docs}): {exc!r}. "
                f"Retrying in {backoff:.1f}s...",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(60.0, backoff * 1.8)


class InfiniteStatefulStreamPacker:
    """
    Background producer packing documents into `batch_size` continuous stateful token lanes of length
    `seq_len` (`x = slice[:-1], y = slice[1:]`).
    Tracks `reset_mask[b]`:
      - `True`: Lane `b` starts cleanly at a fresh document boundary (`S_init[b] = s_prior`).
      - `False`: Lane `b` continues a multi-chunk document from the previous chunk (`S_init[b] = 0.95*S_stream + 0.05*s_prior`),
                 bounded to `max_chain_chunks` (`6` chunks = `12,288` tokens) so occupancy stays stationary.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int = 2,
        seq_len: int = 2048,
        seed: int = 20261104,
        initial_doc_counts: Optional[Dict[str, int]] = None,
        max_chain_chunks: int = 6,
        queue_size: int = 24,
    ):
        self.tok = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.row_len = seq_len + 1
        self.eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
        self.max_chain_chunks = max_chain_chunks
        self.rng = random.Random(seed)
        self.proc_rng = random.Random(seed + 7919)

        self.doc_counts: Dict[str, int] = {name: 0 for name, *_ in STREAM_SPECS}
        self.doc_counts["procedural"] = 0
        if initial_doc_counts:
            for k, v in initial_doc_counts.items():
                if k in self.doc_counts:
                    self.doc_counts[k] = int(v)

        self.names = [s[0] for s in STREAM_SPECS] + ["procedural"]
        self.weights = [s[1] for s in STREAM_SPECS] + [PROCEDURAL_WEIGHT]

        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Tuple[np.ndarray, np.ndarray, List[bool], Dict[str, int]]]" = queue.Queue(
            maxsize=queue_size
        )
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="VAULT1BStreamProducer")
        self._worker.start()

    def close(self) -> None:
        self._stop_event.set()

    def _sample_next_doc_tokens(self, iterators: Dict[str, Iterator[str]]) -> List[int]:
        while not self._stop_event.is_set():
            src = self.rng.choices(self.names, weights=self.weights, k=1)[0]
            if src == "procedural":
                txt = generate_procedural_document(self.proc_rng)
            else:
                txt = next(iterators[src])
            self.doc_counts[src] += 1
            ids = self.tok.encode(txt, add_special_tokens=False)
            if len(ids) >= 24:
                ids.append(self.eos_id)
                return ids
        return [self.eos_id] * self.row_len

    def _run_loop(self) -> None:
        iterators: Dict[str, Iterator[str]] = {}
        for name, _w, repo, cfg, fmt in STREAM_SPECS:
            # Wrap finite datasets like magicoder with modulo skip if resumed past dataset length
            init_skip = self.doc_counts.get(name, 0)
            if name == "magicoder":
                init_skip = init_skip % 75000
            iterators[name] = make_resilient_hf_iterator(repo, cfg, fmt, initial_skip=init_skip)
        for _ in range(self.doc_counts.get("procedural", 0)):
            # Fast-forward procedural RNG deterministically on resume (cap at 50k steps of fast-forward)
            if _ < 50000:
                self.proc_rng.random()

        active_long: List[List[int]] = [[] for _ in range(self.batch_size)]
        chain_len: List[int] = [0 for _ in range(self.batch_size)]

        while not self._stop_event.is_set():
            x_np = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
            y_np = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
            resets: List[bool] = [True] * self.batch_size

            for b in range(self.batch_size):
                if len(active_long[b]) >= self.row_len and chain_len[b] < self.max_chain_chunks:
                    row = active_long[b][: self.row_len]
                    rem = active_long[b][self.seq_len :]
                    resets[b] = False
                    chain_len[b] += 1
                    active_long[b] = rem if len(rem) >= 256 else []
                else:
                    buf: List[int] = []
                    while len(buf) < self.row_len:
                        buf.extend(self._sample_next_doc_tokens(iterators))
                    row = buf[: self.row_len]
                    rem = buf[self.seq_len :]
                    resets[b] = True
                    chain_len[b] = 1
                    active_long[b] = rem if len(rem) >= 256 else []

                x_np[b, :] = row[:-1]
                y_np[b, :] = row[1:]

            snap_counts = dict(self.doc_counts)
            while not self._stop_event.is_set():
                try:
                    self._queue.put((x_np, y_np, resets, snap_counts), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[bool], Dict[str, int]]:
        x_np, y_np, resets, counts = self._queue.get()
        bx = torch.from_numpy(x_np).to(device, non_blocking=True)
        by = torch.from_numpy(y_np).to(device, non_blocking=True)
        return bx, by, resets, counts


# ==============================================================================
# 3. VALIDATION HOLDOUT & UNSEEN STREAMING NIAH PROBES
# ==============================================================================

UNSEEN_NIAH_PREFIXES = [
    "ShieldPod", "CryoLock", "WarpCore", "IonThruster", "PlasmaGrid", "StarTracker",
    "BeaconMux", "GravSensor", "SolarWing", "HullRadar", "ThermalSink", "DockClamp",
]

UNSEEN_FILLER = (
    "All primary telemetry channels maintained nominal signal-to-noise margin across "
    "redundant optical transceivers during orbital attitude stabilization."
)


def build_unseen_2k_niah_probe() -> Tuple[str, List[Tuple[str, str]]]:
    rng = random.Random(2026102999)
    entries: List[Tuple[str, str]] = []
    blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
    used_codes = set()
    for idx in range(1, 19):
        stem = UNSEEN_NIAH_PREFIXES[(idx - 1) % len(UNSEEN_NIAH_PREFIXES)]
        ent = f"{stem}-{chr(64 + idx)}{(idx % 9) + 1}"
        while True:
            code = f"CHK-{rng.randint(1000, 9999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        entries.append((ent, code))
        blocks.append(
            f"[Subsystem {idx:02d}: {ent}] Status: NOMINAL | Verification Code: {code}\n"
            f"  Log: {UNSEEN_FILLER} Packet frame {100 + idx} committed without retransmission."
        )
    targets = [entries[4], entries[13]]
    prompt = (
        "\n".join(blocks)
        + f"\n\nSummary Query ({targets[0][0]}, {targets[1][0]}):\nVerified Tokens: {targets[0][0]} = CHK-"
    )
    return prompt, targets


def load_fixed_val_batches(
    tokenizer, seq_len: int = 2048, n_batches: int = 6, batch_size: int = 2
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[int]]:
    """
    Loads deterministic held-out validation batches from `data_v2/val_tokens.bin` if present,
    computing exact UTF-8 byte counts per batch for `Bits-Per-Byte (BPB)` reporting.
    """
    val_bin = os.path.join(ROOT_DIR, "data_v2", "val_tokens.bin")
    row_len = seq_len + 1
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    byte_counts: List[int] = []
    if os.path.isfile(val_bin):
        n_rows = os.path.getsize(val_bin) // (row_len * 2)
        mm = np.memmap(val_bin, dtype=np.uint16, mode="r", shape=(n_rows, row_len))
        limit = min(n_rows, n_batches * batch_size)
        for start in range(0, limit, batch_size):
            rows = np.asarray(mm[start : start + batch_size], dtype=np.int64)
            x = torch.from_numpy(rows[:, :-1].copy())
            y = torch.from_numpy(rows[:, 1:].copy())
            n_bytes = sum(len(tokenizer.decode(r[1:].tolist(), skip_special_tokens=True).encode("utf-8")) for r in rows)
            batches.append((x, y))
            byte_counts.append(max(1, n_bytes))
        return batches, byte_counts

    # Fallback deterministic validation passages if `data_v2/` is absent
    rng = random.Random(99991)
    for _ in range(n_batches):
        xs, ys = [], []
        b_cnt = 0
        for _b in range(batch_size):
            ids: List[int] = []
            while len(ids) < row_len:
                ids.extend(tokenizer.encode(generate_procedural_document(rng), add_special_tokens=False) + [0])
            row = ids[:row_len]
            xs.append(row[:-1])
            ys.append(row[1:])
            b_cnt += len(tokenizer.decode(row[1:], skip_special_tokens=True).encode("utf-8"))
        batches.append((torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)))
        byte_counts.append(max(1, b_cnt))
    return batches, byte_counts


@torch.no_grad()
def run_validation(
    model: VAULTLMv2_5,
    tokenizer,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_bytes: List[int],
    device: torch.device,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()

    tot_nll_prior = 0.0
    tot_nll_blank = 0.0
    tot_tokens = 0
    tot_bytes = sum(val_bytes)

    for (vx_cpu, vy_cpu), _nb in zip(val_batches, val_bytes):
        vx = vx_cpu.to(device, non_blocking=True)
        vy = vy_cpu.to(device, non_blocking=True)
        n_tok = vx.numel()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ce_prior = model.forward_loss(vx, vy, step_ratio=1.0, logit_chunk_tokens=1024, disable_prior=False)
            ce_blank = model.forward_loss(vx, vy, step_ratio=1.0, logit_chunk_tokens=1024, disable_prior=True)
        tot_nll_prior += float(ce_prior.item()) * n_tok
        tot_nll_blank += float(ce_blank.item()) * n_tok
        tot_tokens += n_tok

    val_ce = tot_nll_prior / max(1, tot_tokens)
    val_ce_blank = tot_nll_blank / max(1, tot_tokens)
    val_bpb = (tot_nll_prior / math.log(2.0)) / max(1, tot_bytes)
    val_ppl = math.exp(min(20.0, val_ce))
    val_ppl_blank = math.exp(min(20.0, val_ce_blank))

    # Unseen 2K 1-shot plain-text NIAH probe via O(1) Streaming Engine
    niah_prompt, niah_targets = build_unseen_2k_niah_probe()
    niah_ids = tokenizer.encode(niah_prompt, add_special_tokens=False)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_niah, _c1 = VAULTv2_5StreamingEngine.generate_greedy(
            model, niah_ids, max_new_tokens=32, stop_ids=(0, 2), use_cudagraph=False
        )
    del _c1
    niah_out = "CHK-" + tokenizer.decode(gen_niah, skip_special_tokens=True).split("\n")[0].strip()
    expected_d0 = niah_targets[0][1]
    niah_hit = expected_d0 in niah_out

    # Short open-ended English continuation probe
    prose_prefix = "The fundamental advantage of associative memory in recurrent sequence models is"
    p_ids = tokenizer.encode(prose_prefix, add_special_tokens=False)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_p, _c2 = VAULTv2_5StreamingEngine.generate_greedy(
            model, p_ids, max_new_tokens=40, stop_ids=(0, 2), use_cudagraph=False
        )
    del _c2
    prose_out = tokenizer.decode(gen_p, skip_special_tokens=True).replace("\n", " ").strip()

    # Cortical `s_prior` norm telemetry across layers
    prior_cell_norms = []
    for blk in model.layers:
        if blk.attn.s_prior is not None:
            cn = blk.attn.s_prior.detach().float().norm(dim=(-2, -1))
            prior_cell_norms.append((float(cn.mean().item()), float(cn.max().item())))
    mean_cell_f = sum(x[0] for x in prior_cell_norms) / max(1, len(prior_cell_norms))
    max_cell_f = max((x[1] for x in prior_cell_norms), default=0.0)

    model._decode_static_pack_v2_5 = None
    model._decode_graph_runners_v2_5 = None
    if was_training:
        model.train()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_ce": round(val_ce, 4),
        "val_ce_blank": round(val_ce_blank, 4),
        "prior_ce_gain": round(val_ce_blank - val_ce, 5),
        "val_bpb": round(val_bpb, 4),
        "val_ppl": round(val_ppl, 2),
        "val_ppl_blank": round(val_ppl_blank, 2),
        "prior_mean_cell_fnorm": round(mean_cell_f, 4),
        "prior_max_cell_fnorm": round(max_cell_f, 4),
        "unseen_2k_niah_target": expected_d0,
        "unseen_2k_niah_hit": bool(niah_hit),
        "unseen_2k_niah_out": niah_out[:96],
        "sample_prose": prose_out[:180],
    }


# ==============================================================================
# 4. KAHAN-COMPENSATED 8-BIT BF16 ADAMW, WSD SCHEDULE, & ATOMIC CHECKPOINTING
# ==============================================================================

_FP32_BUFFER_NAMES = {
    "harmonic_anchors",
    "local_decay_mask",
    "_local_attn_bias",
    "rope_cos_k",
    "rope_sin_k",
    "rope_cos_q",
    "rope_sin_q",
    "x0_cone_mean",
    "cms_proj",
    "sal_lag_dist",
    "_id_weights_3g",
}


def cast_model_bf16_keep_fp32_buffers_(model: VAULTLMv2_5) -> Dict[int, torch.Tensor]:
    """
    Casts model parameters to `torch.bfloat16` (unlocking 100% BF16 WMMA without autocast weight
    duplication and saving ~3.5 GiB VRAM across weights + grads) while returning exact initial
    FP32-minus-BF16 Kahan residuals `{id(p): res_bf16}` and preserving `_FP32_BUFFER_NAMES` in FP32.
    """
    init_residuals: Dict[int, torch.Tensor] = {}
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.is_floating_point():
            if p.requires_grad:
                p_f32 = p.data.float()
                p_bf16 = p_f32.to(torch.bfloat16)
                res_bf16 = (p_f32 - p_bf16.float()).to(torch.bfloat16)
                p.data = p_bf16
                init_residuals[id(p)] = res_bf16
            elif p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)

    for mod in model.modules():
        for b_name, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_floating_point():
                if b_name in _FP32_BUFFER_NAMES:
                    if buf.dtype != torch.float32:
                        mod._buffers[b_name] = buf.to(torch.float32)
                else:
                    if buf.dtype != torch.bfloat16:
                        mod._buffers[b_name] = buf.to(torch.bfloat16)
    return init_residuals


@triton.jit
def _kahan_adamw_8bit_bf16_kernel(
    p_ptr,
    res_ptr,
    g_ptr,
    mq_ptr,
    ms_ptr,
    rq_ptr,
    rs_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    bc1_inv,
    sqrt_bc2_inv,
    weight_decay,
    eps,
    inv_grad_scale,
    BLOCK: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    p_hi = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p_lo = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_grad_scale

    ms = tl.load(ms_ptr + pid).to(tl.float32)
    rs = tl.load(rs_ptr + pid).to(tl.float32)
    mq = tl.load(mq_ptr + offs, mask=mask, other=0).to(tl.float32)
    rq = tl.load(rq_ptr + offs, mask=mask, other=0).to(tl.float32)

    m = mq * ms
    r_old = rq * rs
    v_old = r_old * r_old

    p_f32 = p_hi + p_lo
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v_old + (1.0 - beta2) * (g * g)
    r_new = tl.sqrt(v_new)

    m_hat = m_new * bc1_inv
    denom = r_new * sqrt_bc2_inv + eps

    p_upd = p_f32 * (1.0 - lr * weight_decay) - lr * (m_hat / denom)
    new_hi_bf16 = p_upd.to(tl.bfloat16)
    new_lo_bf16 = (p_upd - new_hi_bf16.to(tl.float32)).to(tl.bfloat16)

    # Blockwise symmetric int8 quantization for first moment `m_new`
    m_absmax = tl.max(tl.abs(m_new), axis=0)
    new_ms = m_absmax / 127.0
    inv_ms = tl.where(new_ms > 0.0, 1.0 / new_ms, 0.0)
    mq_f = m_new * inv_ms
    mq_round = tl.where(mq_f >= 0.0, tl.floor(mq_f + 0.5), tl.ceil(mq_f - 0.5))
    mq_i8 = tl.minimum(tl.maximum(mq_round, -127.0), 127.0).to(tl.int8)

    # Blockwise unsigned uint8 quantization for root-second-moment `r_new = sqrt(v_new)`
    r_max = tl.max(r_new, axis=0)
    new_rs = r_max / 255.0
    inv_rs = tl.where(new_rs > 0.0, 1.0 / new_rs, 0.0)
    rq_round = tl.floor(r_new * inv_rs + 0.5)
    rq_u8 = tl.minimum(tl.maximum(rq_round, 0.0), 255.0).to(tl.uint8)

    tl.store(p_ptr + offs, new_hi_bf16, mask=mask)
    tl.store(res_ptr + offs, new_lo_bf16, mask=mask)
    tl.store(mq_ptr + offs, mq_i8, mask=mask)
    tl.store(ms_ptr + pid, new_ms)
    tl.store(rq_ptr + offs, rq_u8, mask=mask)
    tl.store(rs_ptr + pid, new_rs)


class Kahan8BitBF16AdamW:
    """
    Fused Triton Kahan-compensated BF16 master-weight AdamW with per-256-block 8-bit `m` (`int8`)
    and `sqrt(v)` (`uint8`) states. Achieves `~4e-5` FP32 master-weight fidelity (`44x` finer than
    1 BF16 ULP) with zero BF16 stagnation while fitting `VAULT-1B` comfortably in `<13.5 GiB` VRAM.
    """

    BLOCK = 256

    def __init__(
        self,
        model: VAULTLMv2_5,
        lr: float = 2.5e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        init_residuals: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.step_count = 0

        decay_entries = []
        nodecay_entries = []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            wd = weight_decay if (p.ndim == 2 and "conv" not in name and "beta_proj" not in name) else 0.0
            if wd > 0.0:
                decay_entries.append((name, p))
            else:
                nodecay_entries.append((name, p))

        self.param_groups = [
            {
                "params": [p for _, p in decay_entries],
                "names": [n for n, _ in decay_entries],
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            },
            {
                "params": [p for _, p in nodecay_entries],
                "names": [n for n, _ in nodecay_entries],
                "lr": float(lr),
                "weight_decay": 0.0,
            },
        ]

        self.state: Dict[str, Dict[str, torch.Tensor]] = {}
        for pg in self.param_groups:
            for name, p in zip(pg["names"], pg["params"]):
                n_el = p.numel()
                n_blk = (n_el + self.BLOCK - 1) // self.BLOCK
                if init_residuals and id(p) in init_residuals:
                    res = init_residuals[id(p)].to(device=p.device, dtype=torch.bfloat16).view_as(p)
                else:
                    res = torch.zeros_like(p, dtype=torch.bfloat16)
                self.state[name] = {
                    "res": res,
                    "mq": torch.zeros(n_el, device=p.device, dtype=torch.int8),
                    "ms": torch.zeros(n_blk, device=p.device, dtype=torch.float32),
                    "rq": torch.zeros(n_el, device=p.device, dtype=torch.uint8),
                    "rs": torch.zeros(n_blk, device=p.device, dtype=torch.float32),
                }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for pg in self.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    @torch.no_grad()
    def step_with_clip(self, max_norm: float = 1.0) -> Tuple[float, bool]:
        sq_sum = None
        for pg in self.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    if sq_sum is None:
                        sq_sum = torch.zeros((), device=p.device, dtype=torch.float32)
                    g_f = p.grad.detach().float()
                    sq_sum = sq_sum + torch.sum(g_f * g_f)
        if sq_sum is None:
            return 0.0, True
        grad_norm = float(sq_sum.sqrt().item())
        if not math.isfinite(grad_norm):
            return grad_norm, False
        inv_scale = min(1.0, float(max_norm) / (grad_norm + 1e-6)) if max_norm > 0 else 1.0

        self.step_count += 1
        b1, b2 = self.betas
        bc1_inv = 1.0 / (1.0 - (b1 ** self.step_count))
        sqrt_bc2_inv = 1.0 / math.sqrt(1.0 - (b2 ** self.step_count))

        for pg in self.param_groups:
            lr = float(pg["lr"])
            wd = float(pg["weight_decay"])
            for name, p in zip(pg["names"], pg["params"]):
                if p.grad is None:
                    continue
                g = p.grad if p.grad.is_contiguous() else p.grad.contiguous()
                st = self.state[name]
                n_el = p.numel()
                n_blk = (n_el + self.BLOCK - 1) // self.BLOCK
                _kahan_adamw_8bit_bf16_kernel[(n_blk,)](
                    p,
                    st["res"],
                    g,
                    st["mq"],
                    st["ms"],
                    st["rq"],
                    st["rs"],
                    n_el,
                    lr,
                    b1,
                    b2,
                    bc1_inv,
                    sqrt_bc2_inv,
                    wd,
                    self.eps,
                    inv_scale,
                    BLOCK=self.BLOCK,
                )
        return grad_norm, True

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "betas": self.betas,
            "eps": self.eps,
            "group_lrs": [pg["lr"] for pg in self.param_groups],
            "group_wds": [pg["weight_decay"] for pg in self.param_groups],
            "state": {
                name: {k: v.detach().cpu() for k, v in st.items()}
                for name, st in self.state.items()
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step_count = int(sd["step_count"])
        self.betas = tuple(sd.get("betas", self.betas))
        self.eps = float(sd.get("eps", self.eps))
        for pg, lr in zip(self.param_groups, sd.get("group_lrs", [])):
            pg["lr"] = float(lr)
        saved_state = sd["state"]
        for name, st in self.state.items():
            if name in saved_state:
                src = saved_state[name]
                for k in ("res", "mq", "ms", "rq", "rs"):
                    st[k].copy_(src[k].to(device=st[k].device, dtype=st[k].dtype))


def get_infinite_wsd_lr(
    step: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
    cooldown_start_step: int = 0,
    cooldown_steps: int = 0,
) -> float:
    if step <= warmup_steps:
        return peak_lr * (step / float(max(1, warmup_steps)))
    if cooldown_steps > 0 and cooldown_start_step > 0 and step > cooldown_start_step:
        prog = min(1.0, (step - cooldown_start_step) / float(max(1, cooldown_steps)))
        # Smooth cosine WSD crystallization decay
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
    return peak_lr


def compute_chunked_ce(
    model: VAULTLMv2_5, h_raw: torch.Tensor, targets: torch.Tensor, logit_chunk_tokens: int = 1024
) -> torch.Tensor:
    h = h_raw.reshape(-1, model.d_model)
    t_flat = targets.reshape(-1)
    n_tok = h.size(0)
    denom = (t_flat != -100).float().sum().clamp(min=1.0)
    total_loss = torch.zeros((), device=h.device, dtype=torch.float32)

    def _chunk_fn(hs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(model.lm_head(hs).float(), ts, ignore_index=-100, reduction="sum")

    use_ckpt = model.training and torch.is_grad_enabled() and (model.grad_checkpoint not in (False, None, "none"))
    for start in range(0, n_tok, logit_chunk_tokens):
        end = min(start + logit_chunk_tokens, n_tok)
        if use_ckpt:
            c_sum = torch.utils.checkpoint.checkpoint(_chunk_fn, h[start:end], t_flat[start:end], use_reentrant=False)
        else:
            c_sum = _chunk_fn(h[start:end], t_flat[start:end])
        total_loss = total_loss + c_sum
    return total_loss / denom


def atomic_torch_save(obj: dict, target_path: str) -> None:
    tmp_path = target_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def save_latest_checkpoint(
    path: str,
    model: VAULTLMv2_5,
    optimizer: Kahan8BitBF16AdamW,
    step: int,
    tokens_seen: int,
    best_val_ce: float,
    S_stream: List[torch.Tensor],
    occ_stream: List[torch.Tensor],
    s_prior_res: List[torch.Tensor],
    cone_ema: torch.Tensor,
    doc_counts: Dict[str, int],
    vault_cfg_dict: dict,
) -> None:
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_ce": best_val_ce,
        "vault_config": vault_cfg_dict,
        "prime_banks": list(model.prime_banks),
        "ema_alpha": float(model.ema_alpha),
        "ema_alpha_schedule": list(model.ema_alpha_schedule),
        "use_cortical_prior": bool(model.use_cortical_prior),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "S_stream": [s.detach().cpu() for s in S_stream],
        "occ_stream": [o.detach().cpu() for o in occ_stream],
        "s_prior_res": [r.detach().cpu() for r in s_prior_res],
        "cone_ema": cone_ema.detach().cpu(),
        "doc_counts": dict(doc_counts),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_torch_save(payload, path)


def save_best_inference_checkpoint(
    path: str,
    model: VAULTLMv2_5,
    step: int,
    tokens_seen: int,
    val_ce: float,
    vault_cfg_dict: dict,
    optimizer: Optional[Kahan8BitBF16AdamW] = None,
    s_prior_res: Optional[List[torch.Tensor]] = None,
) -> None:
    """
    Saves a compact BF16 inference-ready checkpoint (`~2.0 GB`) folding Kahan master residuals
    (`p.float() + res.float()` and `s_prior + s_prior_res`) and preserving exact FP32 precision buffers.
    """
    bf16_sd = {}
    for k, v in model.state_dict().items():
        leaf = k.rsplit(".", 1)[-1]
        if v.is_floating_point() and leaf not in _FP32_BUFFER_NAMES:
            if optimizer is not None and k in optimizer.state:
                res = optimizer.state[k]["res"].detach().float()
                bf16_sd[k] = (v.detach().float() + res).to(torch.bfloat16).cpu()
            elif optimizer is not None and k == "lm_head.weight" and "embed_tokens.weight" in optimizer.state:
                res = optimizer.state["embed_tokens.weight"]["res"].detach().float()
                bf16_sd[k] = (v.detach().float() + res).to(torch.bfloat16).cpu()
            elif s_prior_res is not None and leaf == "s_prior" and k.startswith("layers."):
                l_idx = int(k.split(".")[1])
                bf16_sd[k] = (v.detach().float() + s_prior_res[l_idx].detach().float()).to(torch.bfloat16).cpu()
            else:
                bf16_sd[k] = v.detach().to(torch.bfloat16).cpu()
        else:
            bf16_sd[k] = v.detach().cpu()
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "val_ce": val_ce,
        "vault_config": vault_cfg_dict,
        "prime_banks": list(model.prime_banks),
        "ema_alpha": float(model.ema_alpha),
        "ema_alpha_schedule": list(model.ema_alpha_schedule),
        "use_cortical_prior": bool(model.use_cortical_prior),
        "model_state": bf16_sd,
    }
    atomic_torch_save(payload, path)


# ==============================================================================
# 5. MAIN INFINITE-STREAM TRAINING & CONSOLIDATION LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="VAULT-1B (1.03B) Infinite-Stream Pretraining Engine")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--peak_lr", type=float, default=2.5e-4)
    parser.add_argument("--min_lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    parser.add_argument("--router_anneal_steps", type=int, default=3000)
    parser.add_argument("--cooldown_start_step", type=int, default=0)
    parser.add_argument("--cooldown_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = infinite stream; >0 = stop at max_steps")

    # Cortical Prior & Sleep-Consolidation hyperparameters
    parser.add_argument("--prior_lr", type=float, default=4.0e-3)
    parser.add_argument("--prior_trace_ema", type=float, default=5.0e-4)
    parser.add_argument("--prior_max_cell_norm", type=float, default=2.0)
    parser.add_argument("--cold_start_prob", type=float, default=0.25)
    parser.add_argument("--stream_carry", type=float, default=0.95)
    parser.add_argument("--sleep_every", type=int, default=16)
    parser.add_argument("--sleep_kl_weight", type=float, default=0.35)
    parser.add_argument("--sleep_ce_weight", type=float, default=0.15)
    parser.add_argument("--sleep_pulse_scale", type=float, default=0.25)
    parser.add_argument("--sleep_temp", type=float, default=1.5)

    # Logging, Eval, Checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--bench_dir", type=str, default="benchmarks")
    parser.add_argument("--run_prefix", type=str, default="vault_1b_stream")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(ROOT_DIR, args.ckpt_dir)
    bench_dir = os.path.join(ROOT_DIR, args.bench_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    latest_ckpt = os.path.join(ckpt_dir, f"{args.run_prefix}_latest.pt")
    best_ckpt = os.path.join(ckpt_dir, f"{args.run_prefix}_best.pt")
    cooled_ckpt = os.path.join(ckpt_dir, f"{args.run_prefix}_cooled_final.pt")
    control_file = os.path.join(ckpt_dir, f"{args.run_prefix}_control.json")
    telemetry_path = os.path.join(bench_dir, f"{args.run_prefix}_telemetry.json")

    hf_dist_dir = os.path.join(ROOT_DIR, "hf_dist")
    try:
        if os.path.isdir(hf_dist_dir) and os.path.isfile(os.path.join(hf_dist_dir, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(hf_dist_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("batteryphil/VAULT-v2.5-180M", trust_remote_code=True)
    assert len(tokenizer) == 49152, f"Expected vocab_size=49152, got {len(tokenizer)}"

    n_layers = 27
    d_model = 1536
    n_q_heads = 24
    n_vault_groups = 4
    head_dim = 64
    d_ff = 4096
    chunk_size = 64
    prime_banks = (47, 53, 59, 61)
    total_cells = sum(prime_banks)
    ema_sched = build_geometric_ema_schedule(n_layers, alpha_fast=2.0**-10, alpha_slow=2.0**-16)

    vault_cfg_dict = {
        "vocab_size": len(tokenizer),
        "d_model": d_model,
        "n_layers": n_layers,
        "n_q_heads": n_q_heads,
        "n_vault_groups": n_vault_groups,
        "head_dim": head_dim,
        "d_ff": d_ff,
        "chunk_size": chunk_size,
        "prime_banks": list(prime_banks),
        "ema_alpha": float(ema_sched[0]),
        "ema_alpha_schedule": list(ema_sched),
        "use_cortical_prior": True,
        "tie_word_embeddings": True,
    }

    model = VAULTLMv2_5(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_vault_groups=n_vault_groups,
        head_dim=head_dim,
        d_ff=d_ff,
        chunk_size=chunk_size,
        grad_checkpoint=True,
        prime_banks=prime_banks,
        ema_alpha=float(ema_sched[0]),
        ema_alpha_schedule=ema_sched,
        use_cortical_prior=True,
    )
    # Initialize Vault Group 3 extended sibling binder (`ext_span_alpha = 0.35`) and gentle orthogonal induction out
    with torch.no_grad():
        for blk in model.layers:
            blk.attn.ext_span_alpha.data.fill_(0.35)
            blk.attn.ind_out_proj.weight.data.mul_(0.25)

    init_residuals = cast_model_bf16_keep_fp32_buffers_(model)
    model = model.to(device)
    optimizer = Kahan8BitBF16AdamW(
        model,
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        init_residuals=init_residuals,
    )
    del init_residuals

    B = args.micro_batch
    S_stream = [
        torch.zeros((B, n_vault_groups, total_cells, head_dim, head_dim), device=device, dtype=torch.bfloat16)
        for _ in range(n_layers)
    ]
    occ_stream = [
        torch.zeros((B, n_vault_groups, total_cells), device=device, dtype=torch.float32)
        for _ in range(n_layers)
    ]
    s_prior_grad_acc = [
        torch.zeros((1, n_vault_groups, total_cells, head_dim, head_dim), device=device, dtype=torch.float32)
        for _ in range(n_layers)
    ]
    s_prior_res = [
        torch.zeros((1, n_vault_groups, total_cells, head_dim, head_dim), device=device, dtype=torch.bfloat16)
        for _ in range(n_layers)
    ]
    cone_ema = torch.zeros((1, 1, d_model), device=device, dtype=torch.float32)

    start_step = 0
    tokens_seen = 0
    best_val_ce = float("inf")
    initial_doc_counts: Optional[Dict[str, int]] = None

    if args.resume and os.path.isfile(latest_ckpt):
        print(f"[*] Resuming VAULT-1B Infinite Pretraining from {latest_ckpt} ...", flush=True)
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        cast_model_bf16_keep_fp32_buffers_(model)
        optimizer.load_state_dict(ck["optimizer_state"])
        start_step = int(ck.get("step", 0))
        tokens_seen = int(ck.get("tokens_seen", start_step * B * args.grad_accum * args.seq_len))
        best_val_ce = float(ck.get("best_val_ce", float("inf")))
        initial_doc_counts = ck.get("doc_counts", None)
        if "S_stream" in ck and len(ck["S_stream"]) == n_layers:
            for l_i in range(n_layers):
                S_stream[l_i].copy_(ck["S_stream"][l_i].to(device=device, dtype=torch.bfloat16))
        if "occ_stream" in ck and len(ck["occ_stream"]) == n_layers:
            for l_i in range(n_layers):
                occ_stream[l_i].copy_(ck["occ_stream"][l_i].to(device=device, dtype=torch.float32))
        if "s_prior_res" in ck and len(ck["s_prior_res"]) == n_layers:
            for l_i in range(n_layers):
                s_prior_res[l_i].copy_(ck["s_prior_res"][l_i].to(device=device, dtype=torch.bfloat16))
        if "cone_ema" in ck and ck["cone_ema"] is not None:
            cone_ema.copy_(ck["cone_ema"].to(device=device, dtype=torch.float32))
        print(
            f"[*] Resumed at step {start_step:,} ({tokens_seen / 1e6:.2f}M tokens, best_val_ce={best_val_ce:.4f})",
            flush=True,
        )

    telemetry = {"config": vault_cfg_dict, "args": vars(args), "history": [], "evals": []}
    if args.resume and os.path.isfile(telemetry_path):
        try:
            with open(telemetry_path, "r") as f:
                old_tel = json.load(f)
            if isinstance(old_tel, dict):
                telemetry["history"] = old_tel.get("history", [])
                telemetry["evals"] = old_tel.get("evals", [])
        except Exception:
            pass

    val_batches, val_bytes = load_fixed_val_batches(tokenizer, seq_len=args.seq_len, n_batches=6, batch_size=B)
    packer = InfiniteStatefulStreamPacker(
        tokenizer,
        batch_size=B,
        seq_len=args.seq_len,
        seed=args.seed + start_step,
        initial_doc_counts=initial_doc_counts,
    )

    dense_params = model.count_dense_parameters()
    prior_params = model.count_prior_parameters()
    tot_params = model.count_parameters(trainable_only=False)
    tok_per_step = B * args.grad_accum * args.seq_len
    hw_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 106, flush=True)
    print("VAULT-1B (`1.03B`) INFINITE-STREAM PRETRAINING + CORTICAL PRIOR VAULT CONSOLIDATION", flush=True)
    print("=" * 106, flush=True)
    print(f"  Hardware & Precision : {device} ({hw_name}) | BF16 WMMA + Triton Kahan Master Residual + 8-bit AdamW", flush=True)
    print(
        f"  Parameters           : {tot_params:,} total ({dense_params:,} dense backbone + {prior_params:,} sparse s_prior)",
        flush=True,
    )
    print(
        f"  Geometry             : {n_layers}L x {d_model}d | H_q={n_q_heads}, G_vault={n_vault_groups} (6:1 GVA) | "
        f"P={prime_banks} ({total_cells} cells)",
        flush=True,
    )
    print(
        f"  Multi-Timescale GTRM : Geometric EMA alpha in [{ema_sched[-1]:.2e} .. {ema_sched[0]:.2e}] "
        f"(char. horizons 1,024 -> 65,536 tok)",
        flush=True,
    )
    print(
        f"  Batch & Schedule     : B={B} x accum={args.grad_accum} x L={args.seq_len} = {tok_per_step:,} tok/step | "
        f"Warmup={args.warmup_steps} -> Peak LR={args.peak_lr:.2e}",
        flush=True,
    )
    print(
        f"  Stream Mixture       : 55% FineWeb-Edu | 20% Cosmopedia-v2 | 15% Clean Code | 7% FineMath | 3% Multi-Chunk NIAH",
        flush=True,
    )
    print("=" * 106, flush=True)

    stop_requested = False

    def _handle_sig(signum, _frame):
        nonlocal stop_requested
        print(f"\n[Signal {signum}] Graceful stop requested — will finish current step and save checkpoint...", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    dropout_rng = random.Random(args.seed + 900001 + start_step)
    cooldown_start = int(args.cooldown_start_step)
    cooldown_steps = int(args.cooldown_steps)
    stop_after_step = int(args.max_steps)

    model.train()
    t_window_start = time.perf_counter()
    toks_in_window = 0
    latest_doc_counts: Dict[str, int] = dict(packer.doc_counts)
    last_sleep_kl = 0.0
    last_sleep_ce = 0.0
    consec_skipped = 0
    skipped_steps = 0

    step = start_step
    completed_step = start_step
    try:
        while True:
            if stop_requested:
                break
            if stop_after_step > 0 and step >= stop_after_step:
                break

            step += 1

            # Optional live control file check (`checkpoints/vault_1b_stream_control.json`) every 25 steps
            if step % 25 == 0 and os.path.isfile(control_file):
                try:
                    with open(control_file, "r") as cf:
                        ctrl = json.load(cf)
                    if isinstance(ctrl, dict):
                        if "cooldown_start_step" in ctrl:
                            cooldown_start = int(ctrl["cooldown_start_step"])
                        if "cooldown_steps" in ctrl:
                            cooldown_steps = int(ctrl["cooldown_steps"])
                        if "stop_after_step" in ctrl:
                            stop_after_step = int(ctrl["stop_after_step"])
                except Exception:
                    pass

            lr = get_infinite_wsd_lr(
                step,
                warmup_steps=args.warmup_steps,
                peak_lr=args.peak_lr,
                min_lr=args.min_lr,
                cooldown_start_step=cooldown_start,
                cooldown_steps=cooldown_steps,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            step_ratio = min(1.0, step / float(max(1, args.router_anneal_steps)))

            optimizer.zero_grad(set_to_none=True)
            for g_acc in s_prior_grad_acc:
                g_acc.zero_()

            loss_accum = 0.0
            cont_lanes = 0
            cold_lanes = 0
            batch_x0_mean_acc = torch.zeros((1, 1, d_model), device=device, dtype=torch.float32)

            for m_idx in range(args.grad_accum):
                bx, by, reset_mask, latest_doc_counts = packer.next_batch(device)
                with torch.no_grad():
                    x0_mb = model.compute_x0_norm(model.embed_tokens(bx)).float().mean(dim=(0, 1), keepdim=True)
                    batch_x0_mean_acc.add_(x0_mb, alpha=(1.0 / float(args.grad_accum)))

                cold_flags = [dropout_rng.random() < args.cold_start_prob for _ in range(B)]
                prior_weights = [0.0] * B
                for b_i in range(B):
                    if cold_flags[b_i]:
                        cold_lanes += 1
                        prior_weights[b_i] = 0.0
                    elif reset_mask[b_i]:
                        prior_weights[b_i] = 1.0
                    else:
                        cont_lanes += 1
                        prior_weights[b_i] = 1.0 - args.stream_carry

                need_prior_grad = any(w > 0.0 for w in prior_weights)
                s_inits: List[torch.Tensor] = []
                occ_inits: List[torch.Tensor] = []
                with torch.no_grad():
                    for l_i, blk in enumerate(model.layers):
                        sp = blk.attn.s_prior[0]          # (G, C, D, D) bfloat16
                        op = blk.attn.get_occ_prior()[0]  # (G, C) float32
                        s_lanes = []
                        o_lanes = []
                        for b_i in range(B):
                            if cold_flags[b_i]:
                                s_lanes.append(torch.zeros_like(sp))
                                o_lanes.append(torch.zeros_like(op))
                            elif reset_mask[b_i]:
                                s_lanes.append(sp.clone())
                                o_lanes.append(op.clone())
                            else:
                                c_w = args.stream_carry
                                p_w = 1.0 - c_w
                                s_mix = (c_w * S_stream[l_i][b_i].float() + p_w * sp.float()).to(torch.bfloat16)
                                o_mix = c_w * occ_stream[l_i][b_i] + p_w * op
                                s_lanes.append(s_mix)
                                o_lanes.append(o_mix)
                        s_in_l = torch.stack(s_lanes, dim=0)
                        o_in_l = torch.stack(o_lanes, dim=0)
                        if need_prior_grad:
                            s_in_l.requires_grad_(True)
                        s_inits.append(s_in_l)
                        occ_inits.append(o_in_l)

                do_sleep = (
                    (args.sleep_every > 0)
                    and (step % args.sleep_every == 0)
                    and (m_idx == args.grad_accum - 1)
                )

                if do_sleep:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        h_t, _beta_m, fin_states = model._encode_hidden_with_aux(
                            bx,
                            step_ratio=step_ratio,
                            s_inits=s_inits,
                            occ_inits=occ_inits,
                            return_final_states=True,
                        )
                        h_teacher_det = h_t.detach()
                        mb_ce = compute_chunked_ce(model, h_t, by, logit_chunk_tokens=1024)
                        scaled_loss = mb_ce / float(args.grad_accum)
                    del h_t
                    scaled_loss.backward()
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        mb_ce, fin_states = model.forward_loss(
                            bx,
                            by,
                            step_ratio=step_ratio,
                            logit_chunk_tokens=1024,
                            s_inits=s_inits,
                            occ_inits=occ_inits,
                            return_final_states=True,
                        )
                        scaled_loss = mb_ce / float(args.grad_accum)
                    scaled_loss.backward()

                loss_accum += float(scaled_loss.item())
                tokens_seen += bx.numel()
                toks_in_window += bx.numel()

                with torch.no_grad():
                    for l_i in range(n_layers):
                        S_stream[l_i].copy_(fin_states[l_i][0])
                        occ_stream[l_i].copy_(fin_states[l_i][1])
                        if need_prior_grad and s_inits[l_i].grad is not None:
                            g_s = s_inits[l_i].grad
                            for b_i in range(B):
                                w_b = prior_weights[b_i]
                                if w_b > 0.0:
                                    s_prior_grad_acc[l_i][0].add_(
                                        g_s[b_i].float(), alpha=(w_b / float(args.grad_accum))
                                    )
                del s_inits, occ_inits, fin_states

                if do_sleep:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s_tot, s_ce, s_kl = model.forward_sleep_kl_loss(
                            bx,
                            by,
                            h_teacher=h_teacher_det,
                            step_ratio=step_ratio,
                            temperature=args.sleep_temp,
                            kl_weight=args.sleep_kl_weight,
                            ce_weight=args.sleep_ce_weight,
                            logit_chunk_tokens=1024,
                        )
                        sleep_scaled = args.sleep_pulse_scale * s_tot
                    del h_teacher_det
                    sleep_scaled.backward()
                    last_sleep_kl = float(s_kl.item())
                    last_sleep_ce = float(s_ce.item())

            if not math.isfinite(loss_accum):
                raise RuntimeError(f"Non-finite training loss {loss_accum} at step {step}")

            grad_norm, step_ok = optimizer.step_with_clip(max_norm=1.0)
            if not step_ok:
                consec_skipped += 1
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for l_i in range(n_layers):
                        S_stream[l_i].zero_()
                        occ_stream[l_i].zero_()
                print(
                    f"[Warning] Non-finite grad_norm={grad_norm} at step {step}; skipped optimizer update "
                    f"& reset S_stream (consec={consec_skipped}, total_skipped={skipped_steps})",
                    flush=True,
                )
                if consec_skipped >= 5:
                    raise RuntimeError(f"5 consecutive non-finite gradient steps at step {step}")
                continue
            consec_skipped = 0
            completed_step = step

            # Update embedding anisotropy cone buffer (`x0_cone_mean`) + persistent Cortical Prior Vaults (`s_prior`)
            with torch.no_grad():
                cone_ema.mul_(0.995).add_(batch_x0_mean_acc, alpha=0.005)
                warmup_scale = min(1.0, step / float(max(1, args.warmup_steps)))
                lr_prior = args.prior_lr * (lr / max(args.peak_lr, 1e-12))
                trace_alpha = args.prior_trace_ema * warmup_scale

                for l_i, blk in enumerate(model.layers):
                    blk.attn.salience_binder.x0_cone_mean.copy_(cone_ema)
                    sp = blk.attn.s_prior
                    if sp is None:
                        continue
                    g_acc = s_prior_grad_acc[l_i]  # (1, G, C, D, D)
                    c_fnorm = g_acc.norm(dim=(-2, -1), keepdim=True)
                    g_unit = g_acc / (c_fnorm + 1e-6)
                    sp_f = (sp.data.float() + s_prior_res[l_i].float()) - lr_prior * g_unit
                    # Hippocampal -> cortical slow EMA trace consolidation from live rolling `S_stream`
                    s_trace = S_stream[l_i].float().mean(dim=0, keepdim=True)
                    sp_f.mul_(1.0 - trace_alpha).add_(s_trace, alpha=trace_alpha)
                    # Soft Frobenius norm bound per 64x64 cell (`max_norm = 2.0`)
                    new_norm = sp_f.norm(dim=(-2, -1), keepdim=True)
                    cap = args.prior_max_cell_norm
                    scale_c = torch.where(new_norm > cap, cap / new_norm.clamp(min=1e-6), torch.ones_like(new_norm))
                    sp_upd = sp_f * scale_c
                    sp_hi = sp_upd.to(torch.bfloat16)
                    sp_lo = (sp_upd - sp_hi.float()).to(torch.bfloat16)
                    sp.data.copy_(sp_hi)
                    s_prior_res[l_i].copy_(sp_lo)

            if step == 1 or step % args.log_every == 0:
                dt_win = max(1e-6, time.perf_counter() - t_window_start)
                tps = toks_in_window / dt_win
                t_window_start = time.perf_counter()
                toks_in_window = 0
                vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                p0_norm = float(
                    (model.layers[0].attn.s_prior.detach().float() + s_prior_res[0].float()).norm(dim=(-2, -1)).mean().item()
                )
                p_last_norm = float(
                    (model.layers[-1].attn.s_prior.detach().float() + s_prior_res[-1].float()).norm(dim=(-2, -1)).mean().item()
                )
                tot_slots = B * args.grad_accum

                print(
                    f"[Step {step:6d} | {tokens_seen / 1e6:8.2f}M tok] "
                    f"ce={loss_accum:.4f} (PPL={math.exp(min(15.0, loss_accum)):6.2f}) | "
                    f"sleep_kl={last_sleep_kl:.4f} | prior_F=({p0_norm:.5f}..{p_last_norm:.5f}) | "
                    f"cont={cont_lanes}/{tot_slots} cold={cold_lanes}/{tot_slots} | "
                    f"gnorm={grad_norm:.2f} | lr={lr:.2e} | {tps:,.0f} tok/s | peak={vram_gb:.2f}GiB",
                    flush=True,
                )

                telemetry["history"].append(
                    {
                        "step": step,
                        "tokens_m": round(tokens_seen / 1e6, 3),
                        "train_ce": round(loss_accum, 4),
                        "sleep_kl": round(last_sleep_kl, 4),
                        "sleep_ce": round(last_sleep_ce, 4),
                        "prior_l0_fnorm": round(p0_norm, 6),
                        "prior_l26_fnorm": round(p_last_norm, 6),
                        "grad_norm": round(grad_norm, 3),
                        "skipped_steps": skipped_steps,
                        "lr": lr,
                        "tok_s": round(tps, 1),
                        "peak_vram_gib": round(vram_gb, 2),
                    }
                )
                # Keep history bounded in JSON if running for hundreds of thousands of steps
                if len(telemetry["history"]) > 5000:
                    telemetry["history"] = telemetry["history"][-4000:]
                with open(telemetry_path + ".tmp", "w") as tf:
                    json.dump(telemetry, tf, indent=2)
                os.replace(telemetry_path + ".tmp", telemetry_path)

            if step % args.eval_every == 0:
                t_ev0 = time.perf_counter()
                ev = run_validation(model, tokenizer, val_batches, val_bytes, device)
                ev["step"] = step
                ev["tokens_m"] = round(tokens_seen / 1e6, 3)
                ev["eval_sec"] = round(time.perf_counter() - t_ev0, 2)
                telemetry["evals"].append(ev)
                with open(telemetry_path + ".tmp", "w") as tf:
                    json.dump(telemetry, tf, indent=2)
                os.replace(telemetry_path + ".tmp", telemetry_path)

                print(
                    f"  >>> [Eval @ {step:6d}] val_ce={ev['val_ce']:.4f} (PPL={ev['val_ppl']:.2f} vs "
                    f"blank={ev['val_ppl_blank']:.2f}, gain={ev['prior_ce_gain']:+.5f}) | "
                    f"BPB={ev['val_bpb']:.4f} | prior_F={ev['prior_mean_cell_fnorm']:.5f} | "
                    f"2K_NIAH={ev['unseen_2k_niah_hit']} ({ev['unseen_2k_niah_out'][:42]})",
                    flush=True,
                )
                print(f"  >>> [Sample] {ev['sample_prose']}", flush=True)

                if float(ev["val_ce"]) < best_val_ce:
                    best_val_ce = float(ev["val_ce"])
                    save_best_inference_checkpoint(
                        best_ckpt,
                        model,
                        step,
                        tokens_seen,
                        best_val_ce,
                        vault_cfg_dict,
                        optimizer=optimizer,
                        s_prior_res=s_prior_res,
                    )
                    print(f"  >>> [Saved Best BF16 Checkpoint] {best_ckpt} (val_ce={best_val_ce:.4f})", flush=True)

                # Reset window throughput timer so eval time doesn't skew training tok/s
                t_window_start = time.perf_counter()
                toks_in_window = 0

            if step % args.save_every == 0:
                save_latest_checkpoint(
                    latest_ckpt,
                    model,
                    optimizer,
                    step,
                    tokens_seen,
                    best_val_ce,
                    S_stream,
                    occ_stream,
                    s_prior_res,
                    cone_ema,
                    latest_doc_counts,
                    vault_cfg_dict,
                )
                print(f"  >>> [Saved Latest Resumable Checkpoint] {latest_ckpt} (step={step:,})", flush=True)
                t_window_start = time.perf_counter()
                toks_in_window = 0

            if cooldown_steps > 0 and cooldown_start > 0 and step >= cooldown_start + cooldown_steps:
                save_best_inference_checkpoint(
                    cooled_ckpt,
                    model,
                    step,
                    tokens_seen,
                    best_val_ce,
                    vault_cfg_dict,
                    optimizer=optimizer,
                    s_prior_res=s_prior_res,
                )
                print(f"[*] WSD Cooldown finished at step {step:,}! Saved {cooled_ckpt}.", flush=True)
                break

    finally:
        packer.close()
        if completed_step > start_step:
            print(f"[*] Saving final resumable state at step {completed_step:,} -> {latest_ckpt} ...", flush=True)
            save_latest_checkpoint(
                latest_ckpt,
                model,
                optimizer,
                completed_step,
                tokens_seen,
                best_val_ce,
                S_stream,
                occ_stream,
                s_prior_res,
                cone_ema,
                latest_doc_counts,
                vault_cfg_dict,
            )
            print("[*] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
