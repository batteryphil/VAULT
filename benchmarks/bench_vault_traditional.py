#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Traditional Academic Zero-Shot LM Benchmark Suite (EleutherAI lm-evaluation-harness Protocol):
#   Evaluates the full VAULT continual evolution (v2.0 Pretrain -> v2.2 -> v2.3 -> v2.4) head-to-head
#   against Mamba-1 (130M), Mamba-2 (130M), Mamba-3 MIMO (187M), and Transformer (SmolLM-135M) across:
#     1. WikiText-2 (wikitext-2-raw-v1, test) : Tokenizer-invariant Bits-Per-Byte (BPB) & Word PPL
#     2. LAMBADA (lambada_openai, en, test)   : Final-word greedy exact-match Acc & Target-Word PPL
#     3. ARC-Easy (ai2_arc, test, n=2376)     : Grade-school STEM science QA (acc & acc_norm)
#     4. ARC-Challenge (ai2_arc, test, n=1172): Middle-school STEM reasoning QA (acc & acc_norm)
#     5. SciQ-RC (sciq w/ support, n=1000)    : In-context STEM textbook reading comprehension (acc_norm)
#     6. SciQ-ClosedBook (sciq no support)    : Parametric STEM exam QA (isolates in-context synthesis gain)
#     7. PIQA (validation, n=1838)            : Physical commonsense interaction QA (acc & acc_norm)
#     8. HellaSwag (validation)               : Commonsense narrative completion (acc & acc_norm)
#     9. Winogrande (winogrande_xl, n=1267)   : Pronoun coreference resolution via suffix log-likelihood (acc)
#    10. BoolQ (validation)                   : Passage reading comprehension Yes/No QA (acc)

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.vault_v2_model import VAULTLMv2
from src.mamba123_baselines import (
    load_mamba1_130m,
    load_mamba2_130m,
    load_mamba3_mimo_187m,
    load_transformer_smollm_135m,
)


def atomic_save_json(data: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ==============================================================================
# 1. UNIFIED CAUSAL BACKBONE + CONTINUATION-ONLY LM_HEAD SCORER
# ==============================================================================

def extract_backbone_hidden(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Runs strictly causal backbone forward pass without materializing `(B, T_pad, V)` full-sequence logits.
    Right-padding up to a multiple of 64 is mathematically bit-exact across VAULT, Mamba-1/2/3, and Transformer.
    """
    if hasattr(model, "_encode_hidden"):
        return model._encode_hidden(input_ids, step_ratio=1.0)
    if hasattr(model, "encode_hidden"):
        return model.encode_hidden(input_ids)
    if hasattr(model, "backbone") and hasattr(model, "lm_head"):
        out = model.backbone(input_ids, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        out = model.model(input_ids, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    raise TypeError(f"Unsupported model class for hidden extraction: {type(model)}")


def encode_pair_lm_eval(
    tok, context: str, continuation: str, max_seq_len: int = 2048
) -> Tuple[List[int], int, int]:
    """
    Exact EleutherAI `lm-evaluation-harness` `_encode_pair`:
    moves trailing spaces from `context` onto `continuation`, encodes `context + continuation`,
    and isolates continuation token span `(whole_ids, cont_len, cont_byte_len)`.
    """
    n_spaces = len(context) - len(context.rstrip(" "))
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    cont_byte_len = max(1, len(continuation.encode("utf-8")))
    ctx_ids = tok.encode(context, add_special_tokens=False)
    whole_ids = tok.encode(context + continuation, add_special_tokens=False)

    cont_len = len(whole_ids) - len(ctx_ids)
    if cont_len <= 0:
        cont_only = tok.encode(continuation, add_special_tokens=False)
        if not cont_only:
            cont_only = [tok.eos_token_id or 0]
        whole_ids = ctx_ids + cont_only
        cont_len = len(cont_only)

    if len(whole_ids) < 2:
        bos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
        whole_ids = [bos] + whole_ids
        cont_len = min(cont_len, len(whole_ids) - 1)

    if len(whole_ids) > max_seq_len:
        whole_ids = whole_ids[-max_seq_len:]
        cont_len = min(cont_len, max_seq_len - 1)

    cont_len = max(1, min(cont_len, len(whole_ids) - 1))
    return whole_ids, cont_len, cont_byte_len


@torch.no_grad()
def score_pairs_batched(
    model: nn.Module,
    tok,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 32,
    max_seq_len: int = 2048,
    device: str = "cuda:0",
) -> List[Tuple[float, float, bool, int]]:
    """
    Scores `(context, continuation)` pairs in length-sorted 64-aligned causal right-padded batches,
    projecting ONLY continuation positions through `model.lm_head` for 20x-100x lower logit memory & FLOPs.
    Returns list aligned with `pairs`: `(sum_logprob, byte_norm_logprob, greedy_exact, cont_tok_len)`.
    """
    model.eval()
    n = len(pairs)
    if n == 0:
        return []

    encoded = []
    for idx, (ctx, cont) in enumerate(pairs):
        w_ids, c_len, b_len = encode_pair_lm_eval(tok, ctx, cont, max_seq_len=max_seq_len)
        encoded.append((idx, w_ids, c_len, b_len))

    # Sort by total sequence length to minimize right-padding overhead
    encoded.sort(key=lambda x: len(x[1]))
    results: List[Optional[Tuple[float, float, bool, int]]] = [None] * n
    pad_id = int(tok.eos_token_id if tok.eos_token_id is not None else 0)
    use_cuda_amp = device.startswith("cuda")

    for st in range(0, n, batch_size):
        chunk = encoded[st : st + batch_size]
        max_len = max(len(item[1]) for item in chunk)
        t_pad = max(64, ((max_len + 63) // 64) * 64)

        # Dynamically cap batch size if `t_pad` approaches 1024-2048 so peak VRAM stays < 4 GB
        sub_bs = batch_size
        if t_pad >= 1024:
            sub_bs = min(batch_size, 8)
        elif t_pad >= 512:
            sub_bs = min(batch_size, 16)

        for sub_st in range(0, len(chunk), sub_bs):
            sub = chunk[sub_st : sub_st + sub_bs]
            sub_max = max(len(item[1]) for item in sub)
            sub_t_pad = max(64, ((sub_max + 63) // 64) * 64)

            inp = torch.full((len(sub), sub_t_pad), pad_id, dtype=torch.long, device=device)
            for b, (_, w_ids, _, _) in enumerate(sub):
                inp[b, : len(w_ids)] = torch.tensor(w_ids, dtype=torch.long, device=device)

            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h = extract_backbone_hidden(model, inp)
                    h_slices = []
                    targets = []
                    spans = []
                    cursor = 0
                    for b, (_, w_ids, c_len, _) in enumerate(sub):
                        L = len(w_ids)
                        c_start = L - c_len
                        h_slices.append(h[b, c_start - 1 : L - 1])
                        targets.extend(w_ids[c_start:L])
                        spans.append((cursor, cursor + c_len))
                        cursor += c_len
                    h_cat = torch.cat(h_slices, dim=0)
                    logits_cat = model.lm_head(h_cat).float()
            else:
                h = extract_backbone_hidden(model, inp)
                h_slices = []
                targets = []
                spans = []
                cursor = 0
                for b, (_, w_ids, c_len, _) in enumerate(sub):
                    L = len(w_ids)
                    c_start = L - c_len
                    h_slices.append(h[b, c_start - 1 : L - 1])
                    targets.extend(w_ids[c_start:L])
                    spans.append((cursor, cursor + c_len))
                    cursor += c_len
                h_cat = torch.cat(h_slices, dim=0)
                logits_cat = model.lm_head(h_cat).float()

            log_probs_cat = F.log_softmax(logits_cat, dim=-1)
            tgt_tensor = torch.tensor(targets, dtype=torch.long, device=device)
            tok_lp = log_probs_cat.gather(-1, tgt_tensor.unsqueeze(-1)).squeeze(-1)
            greedy_ids = log_probs_cat.argmax(dim=-1)

            for b, (orig_idx, _, c_len, b_len) in enumerate(sub):
                s0, s1 = spans[b]
                sum_lp = float(tok_lp[s0:s1].sum().item())
                exact = bool((greedy_ids[s0:s1] == tgt_tensor[s0:s1]).all().item())
                results[orig_idx] = (sum_lp, sum_lp / float(b_len), exact, c_len)

    return results  # type: ignore[return-value]


# ==============================================================================
# 2. MULTIPLE-CHOICE, LAMBADA & WIKITEXT-2 EVALUATORS
# ==============================================================================

def evaluate_mc_items(
    model: nn.Module,
    tok,
    mc_items: Sequence[Tuple[List[Tuple[str, str]], int]],
    batch_size: int = 32,
    device: str = "cuda:0",
) -> Dict:
    """
    Evaluates multiple-choice items where each item is `(candidate_pairs, gold_idx)`.
    Supports both standard shared-context MC and Winogrande suffix-scoring MC.
    """
    t0 = time.perf_counter()
    flat_pairs: List[Tuple[str, str]] = []
    item_offsets: List[Tuple[int, int, int]] = []
    for cand_pairs, gold_idx in mc_items:
        s0 = len(flat_pairs)
        flat_pairs.extend(cand_pairs)
        item_offsets.append((s0, len(flat_pairs), gold_idx))

    scored = score_pairs_batched(model, tok, flat_pairs, batch_size=batch_size, device=device)
    acc_hits = 0
    norm_hits = 0
    for s0, s1, gold in item_offsets:
        cands = scored[s0:s1]
        pred_raw = max(range(len(cands)), key=lambda i: cands[i][0])
        pred_norm = max(range(len(cands)), key=lambda i: cands[i][1])
        if pred_raw == gold:
            acc_hits += 1
        if pred_norm == gold:
            norm_hits += 1

    n = max(1, len(mc_items))
    dt = time.perf_counter() - t0
    return {
        "n_samples": len(mc_items),
        "acc": round(100.0 * acc_hits / n, 2),
        "acc_norm": round(100.0 * norm_hits / n, 2),
        "eval_sec": round(dt, 2),
    }


def evaluate_lambada(
    model: nn.Module,
    tok,
    texts: Sequence[str],
    batch_size: int = 32,
    device: str = "cuda:0",
) -> Dict:
    t0 = time.perf_counter()
    pairs: List[Tuple[str, str]] = []
    for txt in texts:
        t_clean = txt.strip()
        if " " not in t_clean:
            continue
        ctx, last_word = t_clean.rsplit(" ", 1)
        pairs.append((ctx, " " + last_word))

    scored = score_pairs_batched(model, tok, pairs, batch_size=batch_size, device=device)
    n = max(1, len(scored))
    exact_hits = sum(1 for _, _, ex, _ in scored if ex)
    mean_word_nll = -sum(lp for lp, _, _, _ in scored) / n
    word_ppl = math.exp(min(20.0, mean_word_nll))
    dt = time.perf_counter() - t0
    return {
        "n_samples": len(scored),
        "acc": round(100.0 * exact_hits / n, 2),
        "target_word_ppl": round(word_ppl, 2),
        "eval_sec": round(dt, 2),
    }


@torch.no_grad()
def evaluate_wikitext2(
    model: nn.Module,
    tok,
    full_text: str,
    seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda:0",
) -> Dict:
    """
    Evaluates full `WikiText-2` raw test split (`wikitext-2-raw-v1`) in `L=2048` rolling chunks,
    reporting tokenizer-invariant `WikiText2_BPB` (Bits-Per-Byte) and Merity word-level `Word_PPL`.
    """
    t0 = time.perf_counter()
    model.eval()
    ids = tok.encode(full_text, add_special_tokens=False)
    total_bytes = len(full_text.encode("utf-8"))
    total_words = len(re.findall(r"\S+", full_text))

    # Build contiguous (seq_len + 1) windows covering the corpus cleanly
    step = seq_len
    windows: List[List[int]] = []
    for pos in range(0, len(ids) - 1, step):
        seg = ids[pos : pos + seq_len + 1]
        if len(seg) < 65:
            continue
        windows.append(seg)

    total_nll_nats = 0.0
    total_pred_tokens = 0
    pad_id = int(tok.eos_token_id if tok.eos_token_id is not None else 0)
    use_cuda_amp = device.startswith("cuda")

    for st in range(0, len(windows), batch_size):
        sub = windows[st : st + batch_size]
        max_x = max(len(w) - 1 for w in sub)
        t_pad = max(64, ((max_x + 63) // 64) * 64)
        bx = torch.full((len(sub), t_pad), pad_id, dtype=torch.long, device=device)
        for b, w in enumerate(sub):
            bx[b, : len(w) - 1] = torch.tensor(w[:-1], dtype=torch.long, device=device)

        if use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = extract_backbone_hidden(model, bx)
                for b, w in enumerate(sub):
                    L_pred = len(w) - 1
                    logits_b = model.lm_head(h[b, :L_pred]).float()
                    yb = torch.tensor(w[1:], dtype=torch.long, device=device)
                    total_nll_nats += float(F.cross_entropy(logits_b, yb, reduction="sum").item())
                    total_pred_tokens += L_pred
        else:
            h = extract_backbone_hidden(model, bx)
            for b, w in enumerate(sub):
                L_pred = len(w) - 1
                logits_b = model.lm_head(h[b, :L_pred]).float()
                yb = torch.tensor(w[1:], dtype=torch.long, device=device)
                total_nll_nats += float(F.cross_entropy(logits_b, yb, reduction="sum").item())
                total_pred_tokens += L_pred

    # Scale byte/word denominators precisely by fraction of corpus tokens scored
    cov = total_pred_tokens / float(max(1, len(ids) - 1))
    scored_bytes = max(1.0, total_bytes * cov)
    scored_words = max(1.0, total_words * cov)

    bpb = (total_nll_nats / math.log(2.0)) / scored_bytes
    word_ppl = math.exp(min(20.0, total_nll_nats / scored_words))
    tok_ppl = math.exp(min(20.0, total_nll_nats / max(1, total_pred_tokens)))
    dt = time.perf_counter() - t0
    return {
        "bpb": round(bpb, 4),
        "word_ppl": round(word_ppl, 2),
        "native_tok_ppl": round(tok_ppl, 2),
        "evaluated_tokens": total_pred_tokens,
        "eval_sec": round(dt, 2),
    }


# ==============================================================================
# 3. CANONICAL LM-EVAL DATASET BUILDERS (LOADED ONCE ON CPU)
# ==============================================================================

def hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_traditional_benchmark_tasks(max_large_split: int = 1500) -> Dict:
    """Loads and formats all 10 traditional academic evaluation splits once on CPU."""
    print("[*] Loading traditional academic benchmark splits from local HuggingFace cache...", flush=True)

    # 1. WikiText-2 (full raw test split)
    wt_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wikitext2_raw = "\n\n".join(x for x in wt_ds["text"] if x)

    # 2. LAMBADA OpenAI (test)
    lam_ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    lambada_texts = list(lam_ds["text"])
    if max_large_split > 0:
        lambada_texts = lambada_texts[:max_large_split]

    # 3 & 4. ARC-Easy & ARC-Challenge (full test splits)
    def _build_arc(cfg: str):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        items = []
        for r in ds:
            labels = list(r["choices"]["label"])
            texts = list(r["choices"]["text"])
            if r["answerKey"] not in labels or len(texts) < 2:
                continue
            gold = labels.index(r["answerKey"])
            ctx = f"Question: {r['question'].strip()}\nAnswer:"
            items.append(([(ctx, " " + t.strip()) for t in texts], gold))
        return items

    arc_easy = _build_arc("ARC-Easy")
    arc_challenge = _build_arc("ARC-Challenge")

    # 5 & 6. SciQ (full test=1000: both official w/ support passage AND closed-book)
    sciq_ds = load_dataset("allenai/sciq", split="test")
    sciq_rc = []
    sciq_cb = []
    for r in sciq_ds:
        q = r["question"].strip()
        sup = (r.get("support") or "").strip()
        cands = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        ctx_cb = f"Question: {q}\nAnswer:"
        ctx_rc = f"{sup}\nQuestion: {q}\nAnswer:" if sup else ctx_cb
        sciq_rc.append(([(ctx_rc, " " + c.strip()) for c in cands], 3))
        sciq_cb.append(([(ctx_cb, " " + c.strip()) for c in cands], 3))

    # 7. PIQA (full validation=1838)
    piqa_ds = load_dataset("baber/piqa", split="validation")
    piqa_items = []
    for r in piqa_ds:
        gold = int(r["label"])
        if gold not in (0, 1):
            continue
        ctx = f"Question: {r['goal'].strip()}\nAnswer:"
        piqa_items.append(([(ctx, " " + r["sol1"].strip()), (ctx, " " + r["sol2"].strip())], gold))

    # 8. HellaSwag (validation)
    hs_ds = load_dataset("Rowan/hellaswag", split="validation")
    hellaswag_items = []
    n_hs = len(hs_ds) if max_large_split <= 0 else min(len(hs_ds), max_large_split)
    for i in range(n_hs):
        r = hs_ds[i]
        ctx_raw = r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
        ctx = hellaswag_preprocess(ctx_raw)
        endings = [" " + hellaswag_preprocess(e) for e in r["endings"]]
        gold = int(r["label"])
        hellaswag_items.append(([(ctx, e) for e in endings], gold))

    # 9. Winogrande (full validation=1267, exact suffix log-likelihood scoring)
    wg_ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    winogrande_items = []
    for r in wg_ds:
        sent = r["sentence"]
        if "_" not in sent:
            continue
        idx = sent.index("_")
        prefix = sent[:idx]
        suffix = sent[idx + 1 :]
        gold = int(r["answer"]) - 1
        winogrande_items.append(
            ([(prefix + r["option1"], suffix), (prefix + r["option2"], suffix)], gold)
        )

    # 10. BoolQ (validation)
    bq_ds = load_dataset("google/boolq", split="validation")
    boolq_items = []
    n_bq = len(bq_ds) if max_large_split <= 0 else min(len(bq_ds), max_large_split)
    for i in range(n_bq):
        r = bq_ds[i]
        q = r["question"].strip()
        if not q.endswith("?"):
            q = q + "?"
        ctx = f"{r['passage'].strip()}\nQuestion: {q}\nAnswer:"
        gold = 1 if bool(r["answer"]) else 0
        boolq_items.append(([(ctx, " no"), (ctx, " yes")], gold))

    print(
        f"[+] Ready: WikiText-2 ({len(wikitext2_raw):,} chars), LAMBADA ({len(lambada_texts)}), "
        f"ARC-E ({len(arc_easy)}), ARC-C ({len(arc_challenge)}), SciQ-RC/CB ({len(sciq_rc)}), "
        f"PIQA ({len(piqa_items)}), HellaSwag ({len(hellaswag_items)}), "
        f"Winogrande ({len(winogrande_items)}), BoolQ ({len(boolq_items)})",
        flush=True,
    )
    return {
        "wikitext2_raw": wikitext2_raw,
        "lambada": lambada_texts,
        "arc_easy": arc_easy,
        "arc_challenge": arc_challenge,
        "sciq_rc": sciq_rc,
        "sciq_closedbook": sciq_cb,
        "piqa": piqa_items,
        "hellaswag": hellaswag_items,
        "winogrande": winogrande_items,
        "boolq": boolq_items,
    }


# ==============================================================================
# 4. MODEL LOADERS (VAULT v2.0 -> v2.2 -> v2.3 -> v2.4 + EXTERNAL BASELINES)
# ==============================================================================

def load_vault_ckpt(ckpt_name: str, v2_2_mode: bool, default_primes: Tuple[int, ...], device: str) -> Tuple[VAULTLMv2, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    primes = tuple(ckpt.get("prime_banks") or default_primes)
    model = VAULTLMv2(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint="none",
        v2_2_mode=v2_2_mode,
        prime_banks=primes,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), tok


def load_vault_v2_5_ckpt(ckpt_name: str, default_primes: Tuple[int, ...], device: str):
    from src.vault_v2_5_model import VAULTLMv2_5, _unwrap_checkpoint_state_dict

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    primes = tuple(ckpt.get("prime_banks") or default_primes)
    ema_alpha = float(ckpt.get("ema_alpha", 2.0**-11))
    model = VAULTLMv2_5(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_q_heads=12,
        n_vault_groups=4,
        head_dim=64,
        d_ff=2048,
        chunk_size=64,
        grad_checkpoint="none",
        prime_banks=primes,
        ema_alpha=ema_alpha,
    )
    model.load_state_dict(_unwrap_checkpoint_state_dict(ckpt), strict=True)
    return model.to(device).eval(), tok


def _probe_ckpt_meta(ckpt_path: str, base_tok_b: float = 0.275, default_step: int = 0) -> Tuple[float, int]:
    if not os.path.exists(ckpt_path):
        return base_tok_b, default_step
    try:
        _m = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        tot_seen = float(_m.get("total_tokens_seen", 0))
        # Note: v2.4 stores tokens since v2.3 (`0.275B + tot_seen`); v2.5 stores cumulative `base_tokens_seen + sft_tokens_seen`
        if str(_m.get("version", "")).startswith("vault_v2_5"):
            tok_b = round((0.275 + tot_seen / 1e9) if tot_seen < 0.45e9 else (tot_seen / 1e9), 3)
        else:
            tok_b = round(base_tok_b + tot_seen / 1e9, 3)
        st = int(_m.get("step", default_step))
        del _m
        return tok_b, st
    except Exception:
        return base_tok_b, default_step


def main():
    parser = argparse.ArgumentParser(description="Traditional Academic Zero-Shot LM Benchmark Suite")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "benchmarks", "results_vault_traditional.json"),
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated model keys (e.g. VAULT_v2_5_180M) to evaluate & merge into --out",
    )
    parser.add_argument(
        "--max_large_split",
        type=int,
        default=1500,
        help="Deterministic cap on >3K splits (LAMBADA, HellaSwag, BoolQ); 0 = uncapped full split",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    only_set = {x.strip() for x in args.only.split(",") if x.strip()}
    tasks = build_traditional_benchmark_tasks(max_large_split=args.max_large_split)

    v2_4_tok_b, v2_4_step = _probe_ckpt_meta(os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_best.pt"))
    v2_4_10k_tok_b, v2_4_10k_step = _probe_ckpt_meta(
        os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_step10000_best.pt"), base_tok_b=0.275, default_step=10000
    )
    v2_5_tok_b, v2_5_step = _probe_ckpt_meta(
        os.path.join(ROOT_DIR, "checkpoints", "vault_v2_5_best.pt"), base_tok_b=0.630, default_step=1800
    )

    specs = [
        {
            "key": "VAULT_v2_0_Pretrain_180M",
            "title": "VAULT v2.0 Pretrain (180.4M, 88c, 0.24B tok)",
            "pretrain_tokens_B": 0.244,
            "loader": lambda: load_vault_ckpt("vault_v2_pretrain_best.pt", False, (17, 19, 23, 29), device),
        },
        {
            "key": "VAULT_v2_2_180M",
            "title": "VAULT v2.2 (180.4M, 88c, 0.26B tok)",
            "pretrain_tokens_B": 0.260,
            "loader": lambda: load_vault_ckpt("vault_v2_2_best.pt", True, (17, 19, 23, 29), device),
        },
        {
            "key": "VAULT_v2_3_180M",
            "title": "VAULT v2.3 (180.5M, 220c, 0.28B tok)",
            "pretrain_tokens_B": 0.275,
            "loader": lambda: load_vault_ckpt("vault_v2_3_best.pt", True, (47, 53, 59, 61), device),
        },
        {
            "key": "VAULT_v2_4_180M",
            "title": f"VAULT v2.4 (180.5M, 220c, {v2_4_tok_b:.3f}B tok, Step {v2_4_step})",
            "pretrain_tokens_B": v2_4_tok_b,
            "loader": lambda: load_vault_ckpt("vault_v2_4_best.pt", True, (47, 53, 59, 61), device),
        },
        {
            "key": "VAULT_v2_4_Step10000_180M",
            "title": f"VAULT v2.4 Step 10k (180.5M, 220c, {v2_4_10k_tok_b:.3f}B tok, Step {v2_4_10k_step})",
            "pretrain_tokens_B": v2_4_10k_tok_b,
            "loader": lambda: load_vault_ckpt("vault_v2_4_step10000_best.pt", True, (47, 53, 59, 61), device),
        },
        {
            "key": "VAULT_v2_5_180M",
            "title": f"VAULT v2.5 SFT (180.6M, 220c, {v2_5_tok_b:.3f}B tok, Step {v2_5_step})",
            "pretrain_tokens_B": v2_5_tok_b,
            "loader": lambda: load_vault_v2_5_ckpt("vault_v2_5_best.pt", (47, 53, 59, 61), device),
        },
        {
            "key": "Mamba1_130M",
            "title": "Mamba-1 (130.9M, 300B tok)",
            "pretrain_tokens_B": 300.0,
            "loader": lambda: load_mamba1_130m(device=device, grad_checkpoint=False),
        },
        {
            "key": "Mamba2_130M",
            "title": "Mamba-2 (129.7M, 300B tok)",
            "pretrain_tokens_B": 300.0,
            "loader": lambda: load_mamba2_130m(device=device, grad_checkpoint=False, fast_ssd=True),
        },
        {
            "key": "Mamba3_MIMO_187M",
            "title": "Mamba-3 MIMO (187.3M, 100B tok)",
            "pretrain_tokens_B": 100.0,
            "loader": lambda: load_mamba3_mimo_187m(device=device, grad_checkpoint=False),
        },
        {
            "key": "Transformer_SmolLM_135M",
            "title": "Transformer SmolLM-135M (134.5M, 600B tok)",
            "pretrain_tokens_B": 600.0,
            "loader": lambda: load_transformer_smollm_135m(device=device, grad_checkpoint=False),
        },
    ]

    report: Dict = {
        "benchmark": "Traditional Academic Zero-Shot LM Benchmark Suite (EleutherAI lm-eval Protocol)",
        "hardware": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU",
        "max_large_split": args.max_large_split,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {},
    }
    if os.path.exists(args.out) and only_set:
        with open(args.out, "r") as f:
            report = json.load(f)

    mc_suite = [
        ("arc_easy", "ARC-Easy (25% rand)", tasks["arc_easy"], "acc_norm"),
        ("arc_challenge", "ARC-Challenge (25% rand)", tasks["arc_challenge"], "acc_norm"),
        ("sciq_rc", "SciQ w/ Passage (25% rand)", tasks["sciq_rc"], "acc_norm"),
        ("sciq_closedbook", "SciQ Closed-Book (25% rand)", tasks["sciq_closedbook"], "acc_norm"),
        ("piqa", "PIQA (50% rand)", tasks["piqa"], "acc_norm"),
        ("hellaswag", "HellaSwag (25% rand)", tasks["hellaswag"], "acc_norm"),
        ("winogrande", "Winogrande (50% rand)", tasks["winogrande"], "acc"),
        ("boolq", "BoolQ (50% rand)", tasks["boolq"], "acc"),
    ]

    for sp in specs:
        key = sp["key"]
        if only_set and key not in only_set:
            continue

        print("\n" + "=" * 108, flush=True)
        print(f"TRADITIONAL ZERO-SHOT SUITE: {sp['title']} [{key}]", flush=True)
        print("=" * 108, flush=True)

        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_m0 = time.perf_counter()
        model, tok = sp["loader"]()
        n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())

        wt2 = evaluate_wikitext2(model, tok, tasks["wikitext2_raw"], seq_len=2048, batch_size=4, device=device)
        print(
            f"  [WikiText-2     ] BPB={wt2['bpb']:.4f} | Word_PPL={wt2['word_ppl']:6.2f} "
            f"| NativeTok_PPL={wt2['native_tok_ppl']:6.2f} ({wt2['eval_sec']:.1f}s)",
            flush=True,
        )

        lam = evaluate_lambada(model, tok, tasks["lambada"], batch_size=args.batch_size, device=device)
        print(
            f"  [LAMBADA        ] Acc={lam['acc']:5.2f}% | TargetWord_PPL={lam['target_word_ppl']:6.2f} "
            f"(n={lam['n_samples']}, {lam['eval_sec']:.1f}s)",
            flush=True,
        )

        mc_res: Dict[str, Dict] = {}
        primary_scores: List[float] = [lam["acc"]]
        for m_key, m_label, m_items, primary_metric in mc_suite:
            r_mc = evaluate_mc_items(model, tok, m_items, batch_size=args.batch_size, device=device)
            mc_res[m_key] = r_mc
            primary_scores.append(r_mc[primary_metric])
            print(
                f"  [{m_label:<26s}] primary({primary_metric:8s})={r_mc[primary_metric]:5.2f}% "
                f"| acc={r_mc['acc']:5.2f}% | acc_norm={r_mc['acc_norm']:5.2f}% "
                f"(n={r_mc['n_samples']}, {r_mc['eval_sec']:.1f}s)",
                flush=True,
            )

        sciq_in_context_gain = round(mc_res["sciq_rc"]["acc_norm"] - mc_res["sciq_closedbook"]["acc_norm"], 2)
        avg_9task = round(sum(primary_scores) / len(primary_scores), 2)
        total_sec = round(time.perf_counter() - t_m0, 1)
        print(
            f"  --> 9-Task Zero-Shot Average = {avg_9task:.2f}% | SciQ In-Context Gain = {sciq_in_context_gain:+.2f}% "
            f"| Total = {total_sec:.1f}s",
            flush=True,
        )

        report["models"][key] = {
            "title": sp["title"],
            "params": n_params,
            "params_M": round(n_params / 1e6, 2),
            "pretrain_tokens_B": sp["pretrain_tokens_B"],
            "wikitext2": wt2,
            "lambada": lam,
            "multiple_choice": mc_res,
            "sciq_in_context_gain_pct": sciq_in_context_gain,
            "zero_shot_9task_avg_pct": avg_9task,
            "total_eval_sec": total_sec,
        }
        report["timestamp_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_save_json(report, args.out)
        print(f"[+] Saved incremental traditional benchmark telemetry -> {args.out}", flush=True)

        del model, tok
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 108, flush=True)
    print(f"TRADITIONAL ACADEMIC BENCHMARK SUITE COMPLETE -> {args.out}", flush=True)
    print("=" * 108, flush=True)


if __name__ == "__main__":
    main()
