#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v3 — Stage A (ARCHIVED): Kenyon Vault memory recovery trainer + LongSequencePacker.
#
# ARCHIVE NOTE:
#   Active VAULT v3 removed the per-layer Kenyon fast-weight matrices (saving ~245 MB of C99
#   FP32 inference arena and 141 checkpoint tensors while retaining 100% long-context accuracy
#   via SegmentScorer v2 + 2-tier shingle resonance + Hop-2/Rollback + Auto-Pinned Sinks).
#   The standalone Co-Prime Kenyon Cell implementations are preserved in the repository for
#   future repurposing:
#     - PyTorch module : src/kenyon_cells.py
#     - Freestanding C : runtime/kenyon_cells_archive.h
#     - Trained weights: vault_params/archive/vault_v3_stage_a_kenyon.pt
#   Note: LongSequencePacker and DEFAULT_QWEN in this file remain used by evaluation harnesses.
#
# Original Stage A setup:
# The Qwen2.5 backbone is frozen. Only the Vault tensors (~1.9M params) train.
# Teacher = the same frozen weights with FULL causal attention (KV cache grows with context).
# Student = block sliding-window attention (fixed 2W KV ring) + Kenyon Vault (fixed cells).
# Loss    = KL(teacher || student) on positions >= 2W, i.e. exactly where the window can no
#           longer see the context and the Vault has to carry it.
#
# Metric that decides whether VAULT is helping:
#   recovery = (CE_swa_only - CE_vault) / (CE_swa_only - CE_full)
#   0 = vault adds nothing over a plain sliding window, 1 = matches full attention.

import argparse
import hashlib
import json
import math
import os
import queue
import random
import signal
import string
import sys
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

if "expandable_segments" in os.environ.get("PYTORCH_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
from src.vault_v3_model import (  # noqa: E402
    VAULTv3Config,
    VAULTv3ForCausalLM,
    load_qwen2_into_vault,
    save_vault_params,
)

DEFAULT_QWEN = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


# ==============================================================================
# Data: long documents (EN/FR Wikipedia, code) + synthetic past-window retrieval
# ==============================================================================

SOURCES = [
    # name, weight, repo, config, text key, min tokens
    ("wiki_en", 0.30, "wikimedia/wikipedia", "20231101.en", "text", 2500),
    ("wiki_fr", 0.25, "wikimedia/wikipedia", "20231101.fr", "text", 2500),
    ("code", 0.10, "codeparrot/codeparrot-clean", None, "content", 2000),
]
RETRIEVAL_WEIGHT = 0.35

SYLL = ["ka", "lo", "vir", "en", "tor", "sa", "mi", "dra", "ul", "zen", "pha", "rek", "os", "bel", "qui", "nor", "ta", "vex", "ri", "mon"]
ITEMS_EN = [("access code", "code"), ("storage locker", "number"), ("home city", "city"), ("project codename", "word")]
ITEMS_FR = [("code d'accès", "code"), ("casier", "number"), ("ville natale", "city"), ("nom de code du projet", "word")]
CITIES = ["Valmont", "Kirovia", "Port-Elan", "Sarneth", "Dunhaven", "Lisbre", "Ostria", "Marcelline", "Tavora", "Belcourt", "Quimper-Sud", "Arvinne"]
WORDS = ["Heliotrope", "Bastion", "Cormorant", "Lanterne", "Obsidian", "Mistral", "Cascade", "Ferrule", "Galion", "Saffron", "Quartzite", "Ombrelle"]


def _doc_is_eval(text: str) -> bool:
    """Deterministic held-out split: ~2% of documents are reserved for evaluation."""
    return int(hashlib.md5(text[:256].encode("utf-8", "ignore")).hexdigest(), 16) % 50 == 0


def _stream_docs(repo: str, cfg: Optional[str], key: str, seed: int, want_eval: bool) -> Iterator[str]:
    backoff = 2.0
    while True:
        try:
            ds = load_dataset(repo, cfg, split="train", streaming=True) if cfg else load_dataset(repo, split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=2000)
            for row in ds:
                txt = row.get(key) or ""
                if len(txt) < 4000:
                    continue
                if _doc_is_eval(txt) != want_eval:
                    continue
                backoff = 2.0
                yield txt[:80000]
            seed += 1
        except Exception as exc:  # transient network errors
            print(f"[stream] {repo}:{cfg} error {exc!r}; retry in {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(60.0, backoff * 1.8)


class _Rng:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def name(self):
        n = self.r.randint(2, 3)
        a = "".join(self.r.choice(SYLL) for _ in range(n)).capitalize()
        b = "".join(self.r.choice(SYLL) for _ in range(self.r.randint(2, 3))).capitalize()
        return f"{a} {b}"

    def value(self, kind):
        if kind == "code":
            return f"{self.r.randint(1000, 9999)}-{''.join(self.r.choice(string.ascii_uppercase) for _ in range(2))}"
        if kind == "number":
            return str(self.r.randint(100, 999))
        if kind == "city":
            return self.r.choice(CITIES)
        return self.r.choice(WORDS)


class LongSequencePacker:
    """Background producer of (seq_len + 1)-token rows. Retrieval rows also return answer spans."""

    def __init__(self, tok, seq_len: int, seed: int, eval_split: bool = False, queue_size: int = 8):
        self.tok = tok
        self.L = seq_len + 1
        self.eos = tok.convert_tokens_to_ids("<|endoftext|>")
        self.rng = _Rng(seed)
        self.pick = random.Random(seed + 1)
        self.eval_split = eval_split
        self.iters = {n: _stream_docs(r, c, k, seed + i * 101, eval_split) for i, (n, _w, r, c, k, _m) in enumerate(SOURCES)}
        self.min_tok = {n: m for (n, _w, _r, _c, _k, m) in SOURCES}
        self.names = [s[0] for s in SOURCES] + ["retrieval"]
        self.weights = [s[1] for s in SOURCES] + [RETRIEVAL_WEIGHT]
        self.q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.counts = {n: 0 for n in self.names}
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def close(self):
        self._stop.set()

    def _long_doc_ids(self, src: str) -> List[int]:
        while True:
            ids = self.tok(next(self.iters[src]), add_special_tokens=False).input_ids
            if len(ids) >= self.min_tok[src]:
                return ids

    def _plain_row(self, src: str) -> List[int]:
        row: List[int] = []
        while len(row) < self.L:
            row.extend(self._long_doc_ids(src))
            row.append(self.eos)
        return row[: self.L]

    def _retrieval_row(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        fr = self.pick.random() < 0.45
        chat = self.pick.random() < 0.5
        items = ITEMS_FR if fr else ITEMS_EN
        n_facts = self.pick.randint(6, 12)
        facts = []
        used = set()
        while len(facts) < n_facts:
            nm = self.rng.name()
            if nm in used:
                continue
            used.add(nm)
            item, kind = self.pick.choice(items)
            facts.append((nm, item, self.rng.value(kind)))
        if fr:
            fact_lines = [f"Note : le {it} de {nm} est {val}." for nm, it, val in facts]
            head = "Registre confidentiel. Mémorisez les informations suivantes.\n"
        else:
            fact_lines = [f"Note: the {it} of {nm} is {val}." for nm, it, val in facts]
            head = "Confidential register. Memorize the following records.\n"
        filler = self._long_doc_ids("wiki_fr" if fr else "wiki_en")
        asked = self.pick.sample(facts, k=min(len(facts), self.pick.randint(2, 4)))

        # Facts are split into two groups placed early in the sequence (always > 2W before the questions).
        g1_facts, g2_facts = facts[: len(facts) // 2], facts[len(facts) // 2:]
        g1, g2 = fact_lines[: len(fact_lines) // 2], fact_lines[len(fact_lines) // 2:]
        ids_head = self.tok(head + "\n".join(g1) + "\n\n", add_special_tokens=False).input_ids
        ids_mid = self.tok("\n\n" + "\n".join(g2) + "\n\n", add_special_tokens=False).input_ids

        def _build_qa(qfacts):
            q_ids: List[int] = []
            sp_rel: List[Tuple[int, int]] = []
            for nm, it, val in qfacts:
                if chat:
                    q = (f"Quel est le {it} de {nm} ?" if fr else f"What is the {it} of {nm}?")
                    pre = self.tok(f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
                else:
                    q = (f"\nQuestion : quel est le {it} de {nm} ?\nRéponse : " if fr else f"\nQuestion: What is the {it} of {nm}?\nAnswer: ")
                    pre = self.tok(q, add_special_tokens=False).input_ids
                ans = self.tok(f"{val}", add_special_tokens=False).input_ids
                q_ids.extend(pre)
                sp_rel.append((len(q_ids), len(q_ids) + len(ans)))
                q_ids.extend(ans)
            if chat:
                q_ids.extend(self.tok("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False).input_ids)
            return q_ids, sp_rel

        if chat:
            sys_pre = self.tok(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", add_special_tokens=False
            ).input_ids
            ids_head = sys_pre + ids_head

        qa_ids, spans_rel = _build_qa(asked)
        tail = self.tok("\n", add_special_tokens=False).input_ids

        if self.eval_split or self.L < 6000 or not g1_facts:
            n_fill = self.L - len(ids_head) - len(ids_mid) - len(qa_ids) - len(tail)
            while len(filler) < n_fill:
                filler = filler + [self.eos] + self._long_doc_ids("wiki_fr" if fr else "wiki_en")
            cut = max(1, int(n_fill * self.pick.uniform(0.05, 0.20)))
            row = ids_head + filler[:cut] + ids_mid + filler[cut:n_fill] + qa_ids + tail
            row = row[: self.L]
            q0 = len(ids_head) + n_fill + len(ids_mid)
            spans = [(q0 + a, q0 + b) for a, b in spans_rel if q0 + b <= self.L]
            return row, spans

        # Training curriculum: also place a mid-sequence QA block ~2.4K..3.8K tokens after g1 (1-2 evicted chunks)
        mid_asked = self.pick.sample(g1_facts, k=min(len(g1_facts), self.pick.randint(2, 3)))
        qa_mid, sp_mid_rel = _build_qa(mid_asked)
        n_fill = max(2600, self.L - len(ids_head) - len(ids_mid) - len(qa_mid) - len(qa_ids) - len(tail))
        while len(filler) < n_fill:
            filler = filler + [self.eos] + self._long_doc_ids("wiki_fr" if fr else "wiki_en")
        cut1 = max(64, int(n_fill * self.pick.uniform(0.05, 0.14)))
        cut2 = min(n_fill - 256, max(cut1 + 2200, int(n_fill * self.pick.uniform(0.42, 0.56))))
        p1 = ids_head + filler[:cut1] + ids_mid + filler[cut1:cut2]
        qm0 = len(p1)
        p2 = p1 + qa_mid + filler[cut2:n_fill]
        q0 = len(p2)
        row = (p2 + qa_ids + tail)[: self.L]
        spans = [(qm0 + a, qm0 + b) for a, b in sp_mid_rel if qm0 + b <= self.L]
        spans.extend((q0 + a, q0 + b) for a, b in spans_rel if q0 + b <= self.L)
        return row, spans

    def _run(self):
        while not self._stop.is_set():
            src = self.pick.choices(self.names, weights=self.weights, k=1)[0]
            try:
                if src == "retrieval":
                    row, spans = self._retrieval_row()
                else:
                    row, spans = self._plain_row(src), []
            except StopIteration:
                continue
            self.counts[src] += 1
            self.q.put((torch.tensor(row, dtype=torch.long), spans, src))

    def next(self):
        return self.q.get()


# ==============================================================================
# Loss / metrics
# ==============================================================================

def distill_loss_backward(
    model, h_s, h_t, y, start: int, slice_len: int, temp: float,
    kl_weight: float, ce_weight: float, grad_scale: float = 1.0,
    spans: Optional[List[Tuple[int, int]]] = None, span_boost: float = 1.0,
):
    """Chunked KL(teacher||student) + CE over [start, L) accumulating gradient directly onto h_s without recomputation."""
    L = h_s.shape[1]
    use_kl = h_t is not None and kl_weight > 0
    w = torch.ones(L, device=h_s.device, dtype=torch.float32)
    if spans and span_boost != 1.0:
        for s0, s1 in spans:
            lo, hi = max(start, s0 - 1), min(L, s1 - 1)
            if hi > lo:
                w[lo:hi] = span_boost
    w_sum = max(1.0, float(w[start:L].sum().item()))
    hs_det = h_s.detach().requires_grad_(True)
    kl_sum = ce_sum = ce_t_sum = 0.0
    n = 0
    inv_t = 1.0 / temp
    t2 = temp * temp
    for a in range(start, L, slice_len):
        b = min(L, a + slice_len)
        ls = model.lm_logits(hs_det[0, a:b]).float()
        ysl = y[0, a:b]
        wsl = w[a:b]
        if not use_kl:
            ce_vec = F.cross_entropy(ls, ysl, reduction="none")
            sl_loss = (wsl * (ce_weight * ce_vec)).sum() * (grad_scale / w_sum)
            sl_loss.backward()
            ce_sum += float(ce_vec.detach().sum().item())
        else:
            with torch.no_grad():
                lt = model.lm_logits(h_t[0, a:b]).float()
                logp_t = F.log_softmax(lt * inv_t if temp != 1.0 else lt, dim=-1)
                ce_t = (
                    float((-logp_t.gather(-1, ysl[:, None]).squeeze(-1)).sum().item())
                    if temp == 1.0
                    else float(F.cross_entropy(lt, ysl, reduction="sum").item())
                )
            if temp == 1.0:
                logp_s = F.log_softmax(ls, dim=-1)
                ce_vec = -logp_s.gather(-1, ysl[:, None]).squeeze(-1)
            else:
                logp_s = F.log_softmax(ls * inv_t, dim=-1)
                ce_vec = F.cross_entropy(ls, ysl, reduction="none")
            kl_vec = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1) * t2
            sl_loss = (wsl * (kl_weight * kl_vec + ce_weight * ce_vec)).sum() * (grad_scale / w_sum)
            sl_loss.backward()
            kl_sum += float(kl_vec.detach().sum().item())
            ce_sum += float(ce_vec.detach().sum().item())
            ce_t_sum += ce_t
        n += b - a
    h_s.backward(hs_det.grad)
    return kl_sum / n, ce_sum / n, ce_t_sum / n


_BASELINE_CACHE: Dict[str, float] = {}


@torch.no_grad()
def run_eval(model, eval_rows, W: int) -> Dict[str, float]:
    """CE on positions >= 2W and exact-match retrieval of answers placed > 2W after their facts.
    full / swa baselines are computed once (frozen backbone) and cached."""
    model.eval()
    tags = [("vault", "block", True)]
    if not _BASELINE_CACHE:
        tags = [("full", "full", False), ("swa", "block", False)] + tags
    res = {t[0]: [0.0, 0] for t in tags}
    acc = {t[0]: [0, 0] for t in tags}
    for row, spans, _src in eval_rows:
        x = row[:-1].unsqueeze(0).cuda()
        y = row[1:].cuda()
        for tag, mode, von in tags:
            h = model.forward_hidden(x, mode=mode, vault_on=von)
            for a in range(2 * W, h.shape[1], 1024):
                b = min(h.shape[1], a + 1024)
                lg = model.lm_logits(h[0, a:b]).float()
                res[tag][0] += float(F.cross_entropy(lg, y[a:b], reduction="sum"))
                res[tag][1] += b - a
            for s0, s1 in spans:  # answer tokens: predicted at positions s0-1 .. s1-2 in x-coordinates
                lg = model.lm_logits(h[0, s0 - 1: s1 - 1]).float()
                pred = lg.argmax(-1)
                acc[tag][0] += int((pred == row[s0:s1].cuda()).all())
                acc[tag][1] += 1
            del h
    out = {f"ce_{k}": v[0] / max(1, v[1]) for k, v in res.items()}
    out.update({f"acc_{k}": v[0] / max(1, v[1]) for k, v in acc.items()})
    if not _BASELINE_CACHE:
        _BASELINE_CACHE.update({k: out[k] for k in ("ce_full", "ce_swa", "acc_full", "acc_swa")})
    out.update(_BASELINE_CACHE)
    gap = out["ce_swa"] - out["ce_full"]
    out["recovery"] = (out["ce_swa"] - out["ce_vault"]) / gap if gap > 1e-6 else float("nan")
    model.train()
    return out


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="VAULT v3 Stage A — Kenyon Vault memory recovery")
    ap.add_argument("--qwen_dir", default=DEFAULT_QWEN)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_alpha", type=float, default=1.5e-2, help="LR for scalar Vault output gates alpha")
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl_weight", type=float, default=1.0, help=">0 adds KL to full-attention teacher (fast with AOTriton)")
    ap.add_argument("--ce_weight", type=float, default=0.5)
    ap.add_argument("--span_boost", type=float, default=30.0, help="Loss weight multiplier on past-window retrieval answer spans")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--n_eval", type=int, default=12)
    ap.add_argument("--run_prefix", default="vault_v3_stage_a")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    import src.vault_v3_model as v3mod
    v3mod.FAST_VAULT_MATMUL = True
    ck_dir = os.path.join(ROOT_DIR, "checkpoints")
    bench_dir = os.path.join(ROOT_DIR, "benchmarks")
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)
    latest = os.path.join(ck_dir, f"{args.run_prefix}_latest.pt")
    best = os.path.join(ck_dir, f"{args.run_prefix}_best.pt")
    tel_path = os.path.join(bench_dir, f"{args.run_prefix}_telemetry.json")

    with open(os.path.join(args.qwen_dir, "config.json")) as f:
        qc = json.load(f)
    cfg = VAULTv3Config.from_qwen_config(qc, window=args.window)
    tok = AutoTokenizer.from_pretrained(args.qwen_dir)
    model = VAULTv3ForCausalLM(cfg)
    load_qwen2_into_vault(model, args.qwen_dir)
    model = model.to(dev, torch.bfloat16)
    for blk in model.layers:
        blk.self_attn.vault.float()
    model.freeze_backbone()
    model.grad_checkpoint = True
    model.train()

    alpha_params = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".vault.alpha")]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".vault.alpha")]
    vparams = alpha_params + other_params
    alpha_mult = args.lr_alpha / max(1e-8, args.lr)
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr, "lr_mult": 1.0},
         {"params": alpha_params, "lr": args.lr_alpha, "lr_mult": alpha_mult}],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8,
    )
    step = 0
    best_rec = -1e9
    tel = {"config": cfg.to_dict(), "args": vars(args), "history": [], "evals": []}
    if args.resume and os.path.isfile(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["vault_state"], strict=False)
        if len(ck["optimizer"].get("param_groups", [])) == len(opt.param_groups):
            opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        best_rec = float(ck.get("best_recovery", best_rec))
        if os.path.isfile(tel_path):
            with open(tel_path) as f:
                old = json.load(f)
            tel["history"], tel["evals"] = old.get("history", []), old.get("evals", [])
        print(f"[*] resumed at step {step} (best_score={best_rec:+.3f})", flush=True)

    n_train = sum(p.numel() for p in vparams)
    n_total = sum(p.numel() for p in model.parameters())
    print("=" * 100, flush=True)
    print("VAULT v3 STAGE A — Kenyon Vault memory recovery (backbone frozen; " +
          ("CE + KL to full-attention teacher)" if args.kl_weight > 0 else "CE on positions beyond the window)"), flush=True)
    print(f"  params: {n_total:,} total | {n_train:,} trainable (Vault) | window W={cfg.window} | cells={cfg.n_cells} {cfg.vault_primes}", flush=True)
    print(f"  seq_len={args.seq_len} x accum={args.grad_accum} = {args.seq_len*args.grad_accum:,} tok/step | "
          f"lr={args.lr:g} (alpha={args.lr_alpha:g}) | kl={args.kl_weight:g} ce={args.ce_weight:g} boost={args.span_boost:g}", flush=True)
    print("=" * 100, flush=True)

    eval_packer = LongSequencePacker(tok, args.seq_len, seed=777, eval_split=True)
    eval_rows = []
    seen_src = {"wiki_en": 0, "wiki_fr": 0, "code": 0, "retrieval": 0}
    quota = {"wiki_en": 2, "wiki_fr": 2, "code": 1, "retrieval": max(1, args.n_eval - 5)}
    while len(eval_rows) < sum(quota.values()):
        row, spans, src = eval_packer.next()
        if seen_src[src] < quota[src]:
            seen_src[src] += 1
            eval_rows.append((row, spans, src))
    eval_packer.close()
    print(f"[*] eval set: {len(eval_rows)} rows {seen_src}", flush=True)

    packer = LongSequencePacker(tok, args.seq_len, seed=args.seed + step)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    if step == 0:
        ev0 = run_eval(model, eval_rows, cfg.window)
        ev0["step"] = 0
        tel["evals"].append(ev0)
        print(f"  >>> [Eval @ {step}] ce full={ev0['ce_full']:.4f} swa={ev0['ce_swa']:.4f} vault={ev0['ce_vault']:.4f} | "
              f"recovery={ev0['recovery']:+.3f} | retrieval acc full={ev0['acc_full']:.2f} swa={ev0['acc_swa']:.2f} vault={ev0['acc_vault']:.2f}", flush=True)

    t0 = time.perf_counter()
    toks = 0
    start_pos = 2 * cfg.window
    while step < args.max_steps and not stop["flag"]:
        step += 1
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            pr = (step - args.warmup) / max(1, args.max_steps - args.warmup)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, pr)))
        for g in opt.param_groups:
            g["lr"] = lr * float(g.get("lr_mult", 1.0))
        opt.zero_grad(set_to_none=True)
        m_kl = m_ce = m_cet = 0.0
        for _ in range(args.grad_accum):
            row, _spans, _src = packer.next()
            x = row[:-1].unsqueeze(0).to(dev)
            y = row[1:].unsqueeze(0).to(dev)
            h_t = None
            if args.kl_weight > 0:
                with torch.no_grad():
                    h_t = model.forward_hidden(x, mode="full", vault_on=False)
            h_s = model.forward_hidden(x, mode="block", vault_on=True)
            kl, ce_s, ce_t = distill_loss_backward(
                model, h_s, h_t, y, start_pos, 1024, args.temp,
                args.kl_weight, args.ce_weight, grad_scale=1.0 / args.grad_accum,
                spans=_spans, span_boost=args.span_boost,
            )
            m_kl += kl / args.grad_accum
            m_ce += ce_s / args.grad_accum
            m_cet += ce_t / args.grad_accum
            toks += x.numel()
            del h_t, h_s
        gn = torch.nn.utils.clip_grad_norm_(vparams, 1.0)
        if not torch.isfinite(gn):
            print(f"[warn] non-finite grad norm at step {step}; skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        if step % args.log_every == 0 or step == 1:
            dt = time.perf_counter() - t0
            gates = torch.stack([blk.self_attn.vault.gate().detach().abs().mean() for blk in model.layers])
            rec = {
                "step": step, "kl": round(m_kl, 5), "ce_student": round(m_ce, 4), "ce_teacher": round(m_cet, 4),
                "gate_mean": round(float(gates.mean().item()), 4), "gate_max": round(float(gates.max().item()), 4),
                "grad_norm": round(float(gn.item() if torch.is_tensor(gn) else gn), 4), "lr": lr, "tok_s": round(toks / dt, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            tel["history"].append(rec)
            print(f"[Step {step:5d}] kl={m_kl:.4f} | ce student={m_ce:.4f} teacher={m_cet:.4f} | |gate| mean={rec['gate_mean']:.3f} "
                  f"max={rec['gate_max']:.3f} | gn={rec['grad_norm']:.3f} | lr={lr:.2e} | {rec['tok_s']:,.0f} tok/s | peak={rec['peak_gib']:.2f}GiB", flush=True)
            t0 = time.perf_counter()
            toks = 0
            with open(tel_path + ".tmp", "w") as f:
                json.dump(tel, f, indent=1)
            os.replace(tel_path + ".tmp", tel_path)

        if step % args.eval_every == 0:
            ev = run_eval(model, eval_rows, cfg.window)
            ev["step"] = step
            tel["evals"].append(ev)
            score = ev["recovery"] + 0.5 * ev["acc_vault"]
            print(f"  >>> [Eval @ {step}] ce full={ev['ce_full']:.4f} swa={ev['ce_swa']:.4f} vault={ev['ce_vault']:.4f} | "
                  f"recovery={ev['recovery']:+.3f} | retrieval acc full={ev['acc_full']:.2f} swa={ev['acc_swa']:.2f} vault={ev['acc_vault']:.2f}", flush=True)
            if score == score and score > best_rec:
                best_rec = score
                save_vault_params(model, best, extra={"step": step, "eval": ev, "score": score})
                print(f"  >>> [Saved best Vault params] {best} (recovery={ev['recovery']:+.3f}, acc={ev['acc_vault']:.2f})", flush=True)
            t0 = time.perf_counter()
            toks = 0

        if step % args.save_every == 0:
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".vault." in k or k == "vault_route"}
            torch.save({"vault_state": sd, "optimizer": opt.state_dict(), "step": step, "best_recovery": best_rec,
                        "config": cfg.to_dict()}, latest + ".tmp")
            os.replace(latest + ".tmp", latest)

    packer.close()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".vault." in k or k == "vault_route"}
    torch.save({"vault_state": sd, "optimizer": opt.state_dict(), "step": step, "best_recovery": best_rec,
                "config": cfg.to_dict()}, latest + ".tmp")
    os.replace(latest + ".tmp", latest)
    print(f"[*] stopped at step {step}; saved {latest}", flush=True)


if __name__ == "__main__":
    main()
