# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v3 reference model.
#
#   Backbone : Qwen2-compatible decoder (GQA + RoPE + SwiGLU + RMSNorm, tied embeddings).
#              Initialised from Qwen2.5-1.5B-Instruct (Apache-2.0, see NOTICE).
#   Attention: block sliding window (chunk size W; active ring 2W) + dynamic system-prompt pinning
#              (automatically pins ChatML <|im_start|>system...<|im_end|> turns up to max_sink)
#              + fixed-budget Segment-Evicted KV Memory (exact softmax over top-B learned-scored
#              segments in contiguous re-based RoPE space, APL causal EMA habituation, 2-tier
#              shingle resonance, Hop-2/Rollback bridging, and CAAI output governance).
#
# Note: The Co-Prime Kenyon Cell fast-weight modules are preserved for future repurposing in
# src/kenyon_cells.py and runtime/kenyon_cells_archive.h.
#
# Everything here is built from matmul, argmax/topk, elementwise and norm ops so that the
# exact same math runs in the freestanding C99 runtime (runtime/vault3.c).

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class VAULTv3Config:
    vocab_size: int = 151936
    d_model: int = 1536
    n_layers: int = 28
    n_heads: int = 12
    n_kv_heads: int = 2
    head_dim: int = 128
    d_ff: int = 8960
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_embeddings: bool = True
    window: int = 1024                       # chunk size W (attention span is W..2W tokens)
    n_sink: int = 4                          # minimum attention-sink tokens kept visible forever
    max_sink: int = 512                      # maximum dynamically pinned ChatML system-prompt tokens
    auto_pin_system: bool = True             # auto-pin <|im_start|>system ... <|im_end|>\n prefix
    im_start_id: int = 151644
    im_end_id: int = 151645
    system_tok_id: int = 8948
    # Fixed-budget Segment-Evicted KV Memory (0 = SWA+sinks only; >0 = top-B exact KV segment bank)
    seg_budget: int = 0
    seg_hidden: int = 256
    seg_min: int = 16
    seg_max: int = 48
    seg_layers: Tuple[int, ...] = ()

    @property
    def n_cells(self) -> int:
        return 0

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def has_seg_scorer(self) -> bool:
        return len(self.seg_layers) > 0 and self.seg_budget > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seg_layers"] = list(self.seg_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VAULTv3Config":
        d = {k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__}
        if "seg_layers" in d:
            d["seg_layers"] = tuple(int(l) for l in d["seg_layers"])
        return cls(**d)

    @classmethod
    def from_qwen_config(cls, qc: dict, **overrides) -> "VAULTv3Config":
        cfg = cls(
            vocab_size=qc["vocab_size"],
            d_model=qc["hidden_size"],
            n_layers=qc["num_hidden_layers"],
            n_heads=qc["num_attention_heads"],
            n_kv_heads=qc["num_key_value_heads"],
            head_dim=qc["hidden_size"] // qc["num_attention_heads"],
            d_ff=qc["intermediate_size"],
            rms_eps=qc["rms_norm_eps"],
            rope_theta=qc["rope_theta"],
            tie_embeddings=qc.get("tie_word_embeddings", True),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ==============================================================================
# Token byte & ChatML security helpers (exact parity with VTOK in runtime/vault3.c)
# ==============================================================================

_DEFAULT_QWEN_SNAP = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
_RAW_BYTES_CACHE: Dict[Tuple[str, int], List[bytes]] = {}
_TIER_LUT_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}
_BRK_MASK_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _seg_break_tier_bytes(b: bytes) -> int:
    """
    Exact 3-tier structural + JSON/pipe/TSV + CSV/query delimiter rule shared 1:1 with runtime/vault3.c:seg_break_tier.
      Tier 1 (break at >= seg_min): '\\n' anywhere, or trailing '.', '!', '?', ';' (ignoring trailing ASCII ws)
      Tier 2 (break at >= max(seg_min, 2*seg_max//3)): '}', ']', '|', '\\t', or '",' (minified JSON / pipe / TSV)
      Tier 3 (break at >= max(seg_min, seg_max - 8)): ',' or '&' (CSV / query soft fallback before hard seg_max)
    """
    if not b:
        return 0
    has_t2 = False
    has_t3 = False
    bl = len(b)
    for i, ch in enumerate(b):
        if ch == 0x0A:
            return 1
        if ch in (0x7D, 0x5D, 0x7C, 0x09):  # '}', ']', '|', '\t'
            has_t2 = True
        elif i + 1 < bl and ch == 0x22 and b[i + 1] == 0x2C:  # '",'
            has_t2 = True
        elif ch in (0x2C, 0x26):  # ',', '&'
            has_t3 = True
    s = b.rstrip(b" \t\r\x0b\x0c")
    if s and s[-1] in (0x2E, 0x21, 0x3F, 0x3B):  # '.', '!', '?', ';'
        return 1
    if has_t2:
        return 2
    return 3 if has_t3 else 0


def _is_seg_break_bytes(b: bytes) -> bool:
    return _seg_break_tier_bytes(b) == 1


def _resolve_tokenizer_json(qwen_dir: str) -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(qwen_dir, "tokenizer.json"),
        os.path.join(here, "tokenizer.json"),
        os.path.join(os.path.dirname(here), "tokenizer.json"),
        os.path.dirname(here) + "/dist/hf_vault_v3/tokenizer.json",
    ):
        if os.path.isfile(cand):
            return cand
    return None


def build_token_raw_bytes(vocab_size: int, qwen_dir: str = _DEFAULT_QWEN_SNAP) -> List[bytes]:
    key = (qwen_dir, vocab_size)
    if key in _RAW_BYTES_CACHE:
        return _RAW_BYTES_CACHE[key]
    raw_lut = [b""] * vocab_size
    tj_path = _resolve_tokenizer_json(qwen_dir)
    if tj_path is not None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        u2b = {chr(c): b for b, c in zip(bs, cs)}
        with open(tj_path, encoding="utf-8") as f:
            tj = json.load(f)
        for s, idx in tj["model"]["vocab"].items():
            if idx < vocab_size:
                raw_lut[idx] = bytes(u2b[ch] for ch in s)
        for a in tj.get("added_tokens", []):
            idx = a["id"]
            if idx < vocab_size:
                raw_lut[idx] = a["content"].encode("utf-8")
    _RAW_BYTES_CACHE[key] = raw_lut
    return raw_lut


def build_token_tier_lut(vocab_size: int, qwen_dir: str = _DEFAULT_QWEN_SNAP) -> torch.Tensor:
    key = (qwen_dir, vocab_size)
    if key in _TIER_LUT_CACHE:
        return _TIER_LUT_CACHE[key]
    raw_lut = build_token_raw_bytes(vocab_size, qwen_dir)
    tiers = torch.tensor([_seg_break_tier_bytes(b) for b in raw_lut], dtype=torch.uint8)
    _TIER_LUT_CACHE[key] = tiers
    return tiers


def build_token_break_mask(vocab_size: int, qwen_dir: str = _DEFAULT_QWEN_SNAP) -> torch.Tensor:
    key = (qwen_dir, vocab_size)
    if key in _BRK_MASK_CACHE:
        return _BRK_MASK_CACHE[key]
    mask = build_token_tier_lut(vocab_size, qwen_dir) == 1
    _BRK_MASK_CACHE[key] = mask
    return mask


def _norm_shingle_bytes_from_raw(raw: bytes, cap: int = 4096) -> bytes:
    """Exact byte-level normalizer shared 1:1 with runtime/vault3.c:norm_append_raw_bytes."""
    out = bytearray(b" ")
    for c in raw:
        if 0x41 <= c <= 0x5A:
            c += 32
        if (0x30 <= c <= 0x39) or (0x61 <= c <= 0x7A) or c in (0x5F, 0x2D):
            if len(out) + 2 < cap:
                out.append(c)
        elif out[-1] != 0x20:
            if len(out) + 2 < cap:
                out.append(0x20)
    if out[-1] != 0x20 and len(out) < cap:
        out.append(0x20)
    return bytes(out)


def _norm_shingle_bytes_from_tokens(tok_ids: List[int], raw_lut: List[bytes], cap: int = 8192) -> bytes:
    """Exact byte-level identifier normalizer shared 1:1 with runtime/vault3.c:norm_append_tok_id."""
    out = bytearray(b" ")
    n_lut = len(raw_lut)
    for tid in tok_ids:
        b = raw_lut[tid] if 0 <= tid < n_lut else b""
        for c in b:
            if 0x41 <= c <= 0x5A:
                c += 32
            if (0x30 <= c <= 0x39) or (0x61 <= c <= 0x7A) or c in (0x5F, 0x2D):
                if len(out) + 2 < cap:
                    out.append(c)
            elif out[-1] != 0x20:
                if len(out) + 2 < cap:
                    out.append(0x20)
    if out[-1] != 0x20 and len(out) < cap:
        out.append(0x20)
    return bytes(out)


_ALPHA_STOPWORDS = frozenset({
    b"what", b"with", b"only", b"from", b"that", b"this", b"into", b"find",
    b"which", b"where", b"there", b"their", b"about", b"after", b"under", b"above",
    b"state", b"reply", b"exact", b"value", b"query", b"using", b"token", b"first",
    b"other", b"these", b"those", b"would", b"could",
    b"before", b"string", b"single", b"should", b"active",
    b"current", b"through", b"between",
    b"original", b"assigned",
})


def _fnv1a_word_framed(w: bytes) -> int:
    h = ((2166136261 ^ 32) * 16777619) & 0xFFFFFFFF
    for b in w:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return ((h ^ 32) * 16777619) & 0xFFFFFFFF


def _extract_norm_shingles(norm: bytes, max_out: int = 2048) -> List[int]:
    """
    Exact 2-tier 2-bit-tagged FNV-1a shingle extractor matching runtime/vault3.c:extract_norm_shingles:
      (h & 3) == 1 : Tier-A 4-byte strict-ID sub-shingle
      (h & 3) == 3 : Tier-A whole-identifier / [t=N] tag hash (eligible for Hop-2 / Rollback bridging)
      (h & 3) == 0 : Tier-B pure-alphabetic content whole-word hash (fallback when query has no ID hits)
    """
    nl = len(norm)
    if nl < 4 or max_out <= 0:
        return []
    out: List[int] = []
    i = 0
    while i < nl and len(out) < max_out:
        if norm[i] == 0x20:
            i += 1
            continue
        tok_l = i
        has_alpha = False
        has_num_us = False
        has_hyph = False
        while i < nl and norm[i] != 0x20:
            c = norm[i]
            if 0x61 <= c <= 0x7A:
                has_alpha = True
            elif (0x30 <= c <= 0x39) or c == 0x5F:
                has_num_us = True
            elif c == 0x2D:
                has_hyph = True
            i += 1
        tok_r = i
        tlen = tok_r - tok_l
        if tlen >= 3 and has_alpha and has_num_us:
            has_trail = tok_r < nl
            if has_trail or tlen >= 4:
                k_end = (tok_r - 3) if has_trail else (tok_r - 4)
                k = tok_l
                while k <= k_end and len(out) < max_out:
                    b0, b1, b2, b3 = norm[k], norm[k + 1], norm[k + 2], norm[k + 3]
                    if (
                        (0x30 <= b0 <= 0x39 or b0 in (0x5F, 0x2D))
                        or (0x30 <= b1 <= 0x39 or b1 in (0x5F, 0x2D))
                        or (0x30 <= b2 <= 0x39 or b2 in (0x5F, 0x2D))
                        or (0x30 <= b3 <= 0x39 or b3 in (0x5F, 0x2D))
                    ):
                        h = 2166136261
                        h = ((h ^ b0) * 16777619) & 0xFFFFFFFF
                        h = ((h ^ b1) * 16777619) & 0xFFFFFFFF
                        h = ((h ^ b2) * 16777619) & 0xFFFFFFFF
                        h = ((h ^ b3) * 16777619) & 0xFFFFFFFF
                        out.append((h | 1) & 0xFFFFFFFD)
                    k += 1
            if len(out) < max_out:
                out.append((_fnv1a_word_framed(norm[tok_l:tok_r]) | 3) & 0xFFFFFFFF)
        elif tlen == 1 and has_alpha and not has_num_us and tok_r + 1 < nl:
            n_l = tok_r + 1
            while n_l < nl and norm[n_l] == 0x20:
                n_l += 1
            n_r = n_l
            all_dig = True
            while n_r < nl and norm[n_r] != 0x20:
                if not (0x30 <= norm[n_r] <= 0x39):
                    all_dig = False
                n_r += 1
            if all_dig and 1 <= (n_r - n_l) <= 3 and n_l == tok_r + 1 and len(out) < max_out:
                out.append((_fnv1a_word_framed(norm[tok_l:n_r]) | 3) & 0xFFFFFFFF)
        elif not has_num_us and has_alpha and tlen >= 5:
            w = norm[tok_l:tok_r]
            if not has_hyph:
                if w not in _ALPHA_STOPWORDS and len(out) < max_out:
                    out.append(_fnv1a_word_framed(w) & 0xFFFFFFFC)
            else:
                parts = w.split(b"-")
                all_sub_ok = True
                for sub in parts:
                    slen = len(sub)
                    if slen < 4 or sub in _ALPHA_STOPWORDS:
                        all_sub_ok = False
                    if slen >= 5 and sub not in _ALPHA_STOPWORDS and len(out) < max_out:
                        out.append(_fnv1a_word_framed(sub) & 0xFFFFFFFC)
                if len(parts) >= 2 and all_sub_ok and len(out) < max_out:
                    out.append(_fnv1a_word_framed(w) & 0xFFFFFFFC)
    if len(out) <= 1:
        return out
    return sorted(set(out))


_HOMOGLYPH_MAP: Dict[int, int] = {
    0x0391: ord("a"), 0x03B1: ord("a"), 0x0392: ord("b"), 0x03B2: ord("b"),
    0x0395: ord("e"), 0x03B5: ord("e"), 0x0396: ord("z"), 0x0397: ord("h"),
    0x0399: ord("i"), 0x03B9: ord("i"), 0x039A: ord("k"), 0x03BA: ord("k"),
    0x039C: ord("m"), 0x039D: ord("n"), 0x03BD: ord("n"), 0x039F: ord("o"), 0x03BF: ord("o"),
    0x03A1: ord("p"), 0x03C1: ord("p"), 0x03A3: ord("s"), 0x03C3: ord("s"), 0x03C2: ord("s"),
    0x03A4: ord("t"), 0x03C4: ord("t"), 0x03A5: ord("y"), 0x03C5: ord("y"),
    0x03A7: ord("x"), 0x03C7: ord("x"),
    0x0410: ord("a"), 0x0430: ord("a"), 0x0412: ord("b"), 0x0432: ord("b"),
    0x0421: ord("c"), 0x0441: ord("c"), 0x0415: ord("e"), 0x0435: ord("e"),
    0x041D: ord("h"), 0x043D: ord("h"), 0x0406: ord("i"), 0x0456: ord("i"),
    0x0408: ord("j"), 0x0458: ord("j"), 0x041A: ord("k"), 0x043A: ord("k"),
    0x041C: ord("m"), 0x043C: ord("m"), 0x041E: ord("o"), 0x043E: ord("o"),
    0x0420: ord("p"), 0x0440: ord("p"), 0x0405: ord("s"), 0x0455: ord("s"),
    0x0422: ord("t"), 0x0442: ord("t"), 0x0425: ord("x"), 0x0445: ord("x"),
    0x0423: ord("y"), 0x0443: ord("y"),
}
_STRIP_ASCII = {ord("'"), ord('"'), ord("`"), ord("+"), ord("\\"), ord("_"), ord("-")}
_STRIP_CPS = {0x00AD, 0x034F, 0x061C, 0x180E, 0xFEFF}
_HEX_SEPS = frozenset(b" \\xX:,-_\r\n")


def _gov_b64_val(c: int) -> int:
    if 0x41 <= c <= 0x5A:
        return c - 0x41
    if 0x61 <= c <= 0x7A:
        return c - 0x61 + 26
    if 0x30 <= c <= 0x39:
        return c - 0x30 + 52
    if c in (0x2B, 0x2D):
        return 62
    if c in (0x2F, 0x5F):
        return 63
    return -1


def _gov_hex_val(c: int) -> int:
    if 0x30 <= c <= 0x39:
        return c - 0x30
    if 0x61 <= c <= 0x66:
        return c - 0x61 + 10
    if 0x41 <= c <= 0x46:
        return c - 0x41 + 10
    return -1


class CAAIGovernor:
    """
    Constitutional output governor (inspired by Operating-Organism CAAI & ACI) shared 1:1 with
    runtime/vault3.c:v3_governor. Blocks forbidden syscall/action signatures across arbitrary BPE splits,
    concatenated string smuggling ('SYS' + 'CALL_'), separator mutations ('SYS_CALL_EXEC'), quotes,
    zero-width chars, Greek/Cyrillic homoglyphs, reversed output, and streaming Base64 / Hex encodings.
    """

    def __init__(self, raw_lut: List[bytes], special_stop_ids: Tuple[int, ...] = (151643, 151644, 151645)):
        self.raw_lut = raw_lut
        self.special_stop_ids = set(special_stop_ids)
        self.rules: List[bytes] = []
        self.hist = bytearray()
        self.utf8_rem = bytearray()
        self.raw_tail = bytearray()

    @property
    def active(self) -> bool:
        return len(self.rules) > 0

    def reset_stream(self) -> None:
        self.hist.clear()
        self.utf8_rem.clear()
        self.raw_tail.clear()

    @staticmethod
    def _fold_cp(cp: int) -> int:
        if 0xFF01 <= cp <= 0xFF5E:
            cp -= 0xFEE0
        if cp < 0x80:
            if cp <= 0x20 or cp == 0x7F or cp in _STRIP_ASCII:
                return 0
            if 0x41 <= cp <= 0x5A:
                return cp + 32
            return cp
        if (
            cp in _STRIP_CPS
            or (0x200B <= cp <= 0x200F)
            or (0x2018 <= cp <= 0x201F)
            or (0x202A <= cp <= 0x202E)
            or (0x2060 <= cp <= 0x206F)
        ):
            return 0
        return _HOMOGLYPH_MAP.get(cp, ord("?"))

    @classmethod
    def _feed_bytes(cls, hist: bytearray, rem: bytearray, b: bytes, cap: int = 64) -> None:
        for c in b:
            if rem:
                if (c & 0xC0) == 0x80:
                    rem.append(c)
                    lead = rem[0]
                    need = 2 if (lead & 0xE0) == 0xC0 else (3 if (lead & 0xF0) == 0xE0 else 4)
                    if len(rem) == need:
                        try:
                            cp = ord(bytes(rem).decode("utf-8"))
                        except UnicodeDecodeError:
                            cp = -1
                        rem.clear()
                        if cp >= 0:
                            fc = cls._fold_cp(cp)
                            if fc:
                                if len(hist) >= cap:
                                    del hist[0]
                                hist.append(fc)
                    continue
                rem.clear()
            if c < 0x80:
                fc = cls._fold_cp(c)
                if fc:
                    if len(hist) >= cap:
                        del hist[0]
                    hist.append(fc)
            elif (c & 0xE0) == 0xC0 or (c & 0xF0) == 0xE0 or (c & 0xF8) == 0xF0:
                rem.append(c)

    def add_rule(self, pat: str | bytes) -> None:
        raw = pat.encode("utf-8") if isinstance(pat, str) else pat
        canon = bytearray()
        rem = bytearray()
        self._feed_bytes(canon, rem, raw, cap=32)
        if len(canon) < 4:
            return
        rule = bytes(canon)
        if rule not in self.rules and len(self.rules) < 8:
            self.rules.append(rule)

    @classmethod
    def from_system_prompt(
        cls,
        sys_text: str,
        vocab_size: int = 151936,
        qwen_dir: str = _DEFAULT_QWEN_SNAP,
        raw_lut: Optional[List[bytes]] = None,
    ) -> "CAAIGovernor":
        lut = raw_lut if raw_lut is not None else build_token_raw_bytes(vocab_size, qwen_dir)
        gov = cls(lut)
        low = sys_text.lower()
        if not any(k in low for k in ("never", "forbid", "prohibit", "must not", "do not output", "do not emit")):
            return gov
        if "syscall_exec" in low:
            gov.add_rule("syscall_exec")
            gov.add_rule("syscallexec")
        i, n = 0, len(sys_text)
        while i < n:
            q = sys_text[i]
            if q in ("'", '"', "`"):
                j = sys_text.find(q, i + 1)
                if j != -1 and 5 <= (j - i - 1) < 32:
                    lit = sys_text[i + 1 : j]
                    if "_" in lit or "(" in lit:
                        gov.add_rule(lit)
                        base = lit[:-1] if lit.endswith("(") else lit
                        if len(base) >= 5:
                            gov.add_rule(base)
                        stripped = base.replace("_", "")
                        if len(stripped) >= 5:
                            gov.add_rule(stripped)
                    i = j + 1
                    continue
            i += 1
        return gov

    def _violates(self, hist: bytearray, prev_hlen: int) -> bool:
        hlen = len(hist)
        for r in self.rules:
            rl = len(r)
            if hlen < rl:
                continue
            start_min = max(0, prev_hlen - rl + 1)
            if hist.find(r, start_min) != -1 or hist.find(r[::-1], start_min) != -1:
                return True
        return False

    def _encoded_tail_violates(self, tb: bytes) -> bool:
        tb_take = tb[:64]
        has_b64 = any(_gov_b64_val(b) >= 0 for b in tb_take)
        has_hex = any(_gov_hex_val(b) >= 0 for b in tb_take)
        if not has_b64 and not has_hex:
            return False
        comb = bytes(self.raw_tail) + tb_take
        clen = len(comb)
        bound = len(tb_take) + 3

        if has_b64:
            end = clen
            while end > 0 and comb[end - 1] in (0x0D, 0x0A):
                end -= 1
            eq = 0
            while end > 0 and comb[end - 1] == 0x3D and eq < 2:
                end -= 1
                eq += 1
            start = end
            cnt_b64 = 0
            while start > 0 and cnt_b64 < 128:
                ch = comb[start - 1]
                if ch in (0x0D, 0x0A):
                    start -= 1
                elif _gov_b64_val(ch) >= 0:
                    start -= 1
                    cnt_b64 += 1
                else:
                    break
            if cnt_b64 >= 8:
                vals = [_gov_b64_val(comb[i]) for i in range(start, end) if _gov_b64_val(comb[i]) >= 0][:128]
                nb64 = len(vals)
                for off in range(4):
                    if off + 6 > nb64:
                        break
                    rem_n = nb64 - off
                    p = off
                    dec = bytearray()
                    while rem_n >= 4 and len(dec) + 3 <= 96:
                        v = (vals[p] << 18) | (vals[p + 1] << 12) | (vals[p + 2] << 6) | vals[p + 3]
                        dec.extend(((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF))
                        p += 4
                        rem_n -= 4
                    if rem_n == 3 and len(dec) + 2 <= 96:
                        v = (vals[p] << 18) | (vals[p + 1] << 12) | (vals[p + 2] << 6)
                        dec.extend(((v >> 16) & 0xFF, (v >> 8) & 0xFF))
                    elif rem_n == 2 and len(dec) + 1 <= 96:
                        v = (vals[p] << 18) | (vals[p + 1] << 12)
                        dec.append((v >> 16) & 0xFF)
                    dhist = bytearray()
                    drem = bytearray()
                    self._feed_bytes(dhist, drem, bytes(dec), cap=64)
                    prev_dh = max(0, len(dhist) - bound)
                    if self._violates(dhist, prev_dh):
                        return True

        if has_hex:
            start = clen
            cnt_hex = 0
            while start > 0 and cnt_hex < 192:
                ch = comb[start - 1]
                if _gov_hex_val(ch) >= 0:
                    start -= 1
                    cnt_hex += 1
                elif ch in _HEX_SEPS:
                    start -= 1
                else:
                    break
            if cnt_hex >= 8:
                nibs: List[int] = []
                i = start
                while i < clen and len(nibs) < 192:
                    if (
                        comb[i] == 0x30
                        and i + 2 < clen
                        and comb[i + 1] in (0x78, 0x58)
                        and _gov_hex_val(comb[i + 2]) >= 0
                    ):
                        i += 2
                        continue
                    v = _gov_hex_val(comb[i])
                    if v >= 0:
                        nibs.append(v)
                    i += 1
                nnib = len(nibs)
                if nnib >= 8:
                    for off in range(2):
                        if off + 8 > nnib:
                            break
                        dec = bytearray()
                        p = off
                        while p + 1 < nnib and len(dec) < 96:
                            dec.append((nibs[p] << 4) | nibs[p + 1])
                            p += 2
                        dhist = bytearray()
                        drem = bytearray()
                        self._feed_bytes(dhist, drem, bytes(dec), cap=64)
                        prev_dh = max(0, len(dhist) - bound)
                        if self._violates(dhist, prev_dh):
                            return True
        return False

    def mask_logits(self, logits: torch.Tensor, top_k_guard: int = 64) -> torch.Tensor:
        if not self.active:
            return logits
        view_1d = logits.view(-1)
        k = min(max(1, top_k_guard), 64, int(view_1d.numel()))
        base_hist = self.hist[-31:] if len(self.hist) > 31 else self.hist[:]
        prev_hlen = len(base_hist)
        n_lut = len(self.raw_lut)
        for _ in range(4):
            vals, idxs = torch.topk(view_1d, k)
            any_masked = False
            for v, tid in zip(vals.tolist(), idxs.tolist()):
                if v <= -3.0e38:
                    continue
                tb = self.raw_lut[tid] if 0 <= tid < n_lut else b""
                if not tb:
                    continue
                thist = bytearray(base_hist)
                trem = bytearray(self.utf8_rem)
                self._feed_bytes(thist, trem, tb, cap=64)
                if self._violates(thist, prev_hlen) or self._encoded_tail_violates(tb):
                    view_1d[tid] = -3.4e38
                    any_masked = True
            if not any_masked:
                break
        return logits

    def commit(self, token_id: int) -> None:
        if not self.active:
            return
        if token_id in self.special_stop_ids:
            self.reset_stream()
            return
        if 0 <= token_id < len(self.raw_lut):
            tb = self.raw_lut[token_id]
            self._feed_bytes(self.hist, self.utf8_rem, tb, cap=64)
            if tb:
                self.raw_tail.extend(tb)
                if len(self.raw_tail) > 96:
                    del self.raw_tail[: len(self.raw_tail) - 96]


def encode_ordinary(tok, text: str) -> List[int]:
    """
    Encode untrusted text WITHOUT allowing literal special-token strings (<|im_start|>, <|im_end|>,
    <|endoftext|>, etc.) to be parsed as ChatML control tokens. Mirrors v3_encode_ex(..., allow_special=0).
    """
    if not text:
        return []
    if "<" not in text:
        return tok.encode(text, add_special_tokens=False)
    added_dec = getattr(tok, "added_tokens_decoder", None) or {}
    specials = sorted(
        {str(v.content) for v in added_dec.values()} | set(getattr(tok, "all_special_tokens", []) or []),
        key=len,
        reverse=True,
    )
    specials = [s for s in specials if len(s) >= 3]
    if not specials:
        return tok.encode(text, add_special_tokens=False)

    out: List[int] = []
    chunk_start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "<":
            matched = None
            for sp in specials:
                if text.startswith(sp, i):
                    matched = sp
                    break
            if matched is not None:
                # Every Qwen2 added token (<|name|> or </name>) transitions from cat-0 punctuation ('<|', '</')
                # at index 2 to a cat-1 ASCII letter, which is a natural Qwen2 pre-tokenizer boundary!
                cut = i + 2
                out.extend(tok.encode(text[chunk_start:cut], add_special_tokens=False))
                chunk_start = cut
                i = cut
                continue
        i += 1
    if chunk_start < n:
        out.extend(tok.encode(text[chunk_start:], add_special_tokens=False))
    return out


def build_chatml_ids(tok, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> List[int]:
    """Safe ChatML formatter immune to user/log special-token delimiter injection."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    nl = tok.encode("\n", add_special_tokens=False)
    sys_ids = [im_start] + encode_ordinary(tok, f"system\n{system_prompt}") + [im_end] + nl
    usr_ids = [im_start] + encode_ordinary(tok, f"user\n{user_prompt}") + [im_end] + nl
    ast_ids = [im_start] + encode_ordinary(tok, "assistant\n")
    return sys_ids + usr_ids + ast_ids


def detect_pinned_prefix_len(ids_1d: torch.Tensor, cfg: VAULTv3Config) -> int:
    """
    Determine exact pinned sink count S for a sequence.
    - Non-chat or systemless ChatML streams: returns cfg.n_sink (default 4).
    - ChatML streams starting with <|im_start|>system: pins the opening system turn through
      <|im_end|> (plus immediately following '\\n' token 198 if present), clamping to
      min(max_sink, window) if the system prompt exceeds max_sink.
    """
    L = int(ids_1d.numel())
    base = min(cfg.n_sink, cfg.window)
    if not cfg.auto_pin_system or L < 2:
        return base
    cap = min(cfg.max_sink, cfg.window)
    if cap < 2:
        return base
    if int(ids_1d[0].item()) != cfg.im_start_id or int(ids_1d[1].item()) != cfg.system_tok_id:
        return base
    limit = min(L, cap)
    # Scan for first im_end_id within [2, limit)
    sub = ids_1d[2:limit]
    hits = (sub == cfg.im_end_id).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return max(base, cap) if L >= cap else base
    end_idx = int(hits[0].item()) + 2  # position of <|im_end|>
    pin = end_idx + 1
    if pin < limit and int(ids_1d[pin].item()) == 198:  # '\n' in Qwen2 BPE
        pin += 1
    return max(base, min(pin, cap))


# ==============================================================================
# Basic ops (match HF Qwen2 & runtime/vault3.c numerics)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dt)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: (L,) int64 -> cos, sin: (L, head_dim) float32 (rotate-half layout)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=positions.device).float() / head_dim))
    ang = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, D); cos/sin: (L, D) float32."""
    c = cos.to(x.dtype)[None, None]
    s = sin.to(x.dtype)[None, None]
    return x * c + _rotate_half(x) * s


def _expand_kv(x: torch.Tensor, group: int) -> torch.Tensor:
    """(B, G, ..., L, D) -> (B, G*group, ..., L, D); kv head g serves q heads g*group..g*group+group-1."""
    if group == 1:
        return x
    B, G = x.shape[:2]
    return x.unsqueeze(2).expand(B, G, group, *x.shape[2:]).reshape(B, G * group, *x.shape[2:])


def sink_positions(chunk_idx: torch.Tensor, window: int, n_sink: int) -> torch.Tensor:
    """Virtual RoPE positions of the sink tokens for a query chunk i >= 2: just before the visible window,
    i.e. (i-1)*W - n_sink + j. Keeps query-to-sink distances in [W+1, 2W+n_sink] at any context length."""
    j = torch.arange(n_sink, device=chunk_idx.device)
    return ((chunk_idx[:, None] - 1) * window - n_sink + j[None, :]).clamp(min=0)


def block_sliding_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
    k_raw: Optional[torch.Tensor] = None, n_sink: int = 0, head_dim: int = 0, rope_theta: float = 0.0,
) -> torch.Tensor:
    """
    q: (B, H, L, D) post-RoPE, k/v: (B, G, L, D) post-RoPE, k_raw: (B, G, L, D) pre-RoPE keys (for sinks).
    Token in chunk i attends to chunk i-1 (all) + chunk i (causal) and, for i >= 2, to the first n_sink
    tokens of the sequence (attention sinks). Returns (B, H, L, D).
    """
    B, H, L, D = q.shape
    G = k.shape[1]
    W = window
    S = min(n_sink, L) if (k_raw is not None and n_sink > 0) else 0
    sink_k = k_raw[:, :, :S] if S else None
    sink_v = v[:, :, :S] if S else None
    pad = (-L) % W
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    Lp = L + pad
    n = Lp // W
    ke = _expand_kv(k, H // G)
    ve = _expand_kv(v, H // G)
    qc = q.view(B, H, n, W, D)

    outs = [F.scaled_dot_product_attention(qc[:, :, 0], ke[:, :, :W], ve[:, :, :W], is_causal=True).unsqueeze(2)]
    if n >= 2:
        outs.append(
            F.scaled_dot_product_attention(
                qc[:, :, 1], ke[:, :, :2 * W], ve[:, :, :2 * W], attn_mask=causal_lower_right(W, 2 * W)
            ).unsqueeze(2)
        )
    if n >= 3:
        m = n - 2
        kc = k[:, :, W:].unfold(2, 2 * W, W).transpose(-1, -2)[:, :, :m]  # (B, G, m, 2W, D)
        vc = v[:, :, W:].unfold(2, 2 * W, W).transpose(-1, -2)[:, :, :m]
        if S:
            pos = sink_positions(torch.arange(2, n, device=q.device), W, S).reshape(-1)
            cos, sin = rope_cos_sin(pos, D, rope_theta)
            ks = sink_k.unsqueeze(2).expand(B, G, m, S, D).reshape(B, G, m * S, D)
            ks = apply_rope(ks, cos, sin).view(B, G, m, S, D)
            vs = sink_v.unsqueeze(2).expand(B, G, m, S, D)
            kc = torch.cat([ks, kc], dim=3)
            vc = torch.cat([vs, vc], dim=3)
        KL = S + 2 * W
        k2 = _expand_kv(kc, H // G).reshape(B, H * m, KL, D)
        v2 = _expand_kv(vc, H // G).reshape(B, H * m, KL, D)
        q2 = qc[:, :, 2:].reshape(B, H * m, W, D)
        o2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=causal_lower_right(W, KL)).reshape(B, H, m, W, D)
        outs.append(o2)
    return torch.cat(outs, dim=2).reshape(B, H, Lp, D)[:, :, :L]


# ==============================================================================
# Segment-Evicted KV Scorer & Bank Selection (matches runtime/vault3.c)
# ==============================================================================

class SegmentScorer(nn.Module):
    """Learned MLP scoring evicted segments from causal multi-layer hidden states."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, eps=1e-5)
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat.float()
        h = F.layer_norm(x, (self.ln.normalized_shape[0],), self.ln.weight.float(), self.ln.bias.float(), self.ln.eps)
        h = F.gelu(F.linear(h, self.fc1.weight.float(), self.fc1.bias.float()), approximate="none")
        return F.linear(h, self.fc2.weight.float(), self.fc2.bias.float()).squeeze(-1)


def chunk_segment_scores(
    tok_scores: torch.Tensor, chunk_ids: torch.Tensor, brk_mask: torch.Tensor,
    seg_min: int, seg_max: int, seg_id_offset: int = 0,
) -> Tuple[torch.Tensor, int]:
    """
    Compute self-contained segment-broadcast scores for one chunk of length T (<= W).
    Accepts either a uint8 3-tier LUT (tiers 0..3) or legacy bool break mask (treated as tier 1).
    """
    T = int(chunk_ids.numel())
    if T == 0:
        return tok_scores, seg_id_offset
    tiers = brk_mask[chunk_ids.cpu()].to(torch.uint8).tolist()
    t2 = max(seg_min, (seg_max * 2) // 3)
    t3 = max(seg_min, seg_max - 8 if seg_max > 8 else seg_min)
    seg = torch.empty(T, dtype=torch.long)
    s_loc, n = 0, 0
    for j in range(T):
        seg[j] = s_loc
        n += 1
        tr = tiers[j]
        brk = (tr == 1 and n >= seg_min) or (tr == 2 and n >= t2) or (tr == 3 and n >= t3)
        if j == T - 1 or brk or n >= seg_max:
            s_loc += 1
            n = 0
    seg_dev = seg.to(tok_scores.device)
    mx = torch.full((s_loc,), float("-inf"), device=tok_scores.device, dtype=torch.float32)
    mx.scatter_reduce_(0, seg_dev, tok_scores.float(), "amax")
    global_seg = (seg_dev + seg_id_offset) % 16384
    out = mx[seg_dev] - 1e-6 * global_seg.float()
    return out, seg_id_offset + s_loc


def merge_segment_bank_indices(
    mem_pos: torch.Tensor, mem_score: torch.Tensor,
    cand_pos: torch.Tensor, cand_score: torch.Tensor, budget: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact incremental top-B selection matching runtime/vault3.c:
      Order key: (score DESC, pos ASC). Winners returned in chronological order (pos ASC).
      Returns: (keep_old_idx, add_cand_idx, new_mem_pos, new_mem_score).
    """
    dev = mem_pos.device
    if budget <= 0 or (mem_pos.numel() == 0 and cand_pos.numel() == 0):
        e_l = torch.empty(0, dtype=torch.long, device=dev)
        e_f = torch.empty(0, dtype=torch.float32, device=dev)
        return e_l, e_l, e_l, e_f
    M = int(mem_pos.numel())
    all_pos = torch.cat([mem_pos, cand_pos], dim=0)
    all_sc = torch.cat([mem_score, cand_score], dim=0)
    tot = int(all_pos.numel())
    if tot <= budget:
        return (
            torch.arange(M, dtype=torch.long, device=dev),
            torch.arange(tot - M, dtype=torch.long, device=dev),
            all_pos,
            all_sc,
        )
    # Since all_pos is already strictly ascending, stable argsort of (-all_sc) breaks exact float ties by smaller pos!
    order = torch.argsort(-all_sc, stable=True)[:budget]
    winners = torch.sort(order).values  # chronological order!
    keep_mask = winners < M
    keep_old = winners[keep_mask]
    add_cand = winners[~keep_mask] - M
    return keep_old, add_cand, all_pos[winners], all_sc[winners]


# ==============================================================================
# Streaming state (fixed size, statically allocatable)
# ==============================================================================

class LayerState:
    def __init__(self, cfg: VAULTv3Config, batch: int, device, dtype, seg_budget: int = 0):
        W, G, D = cfg.window, cfg.n_kv_heads, cfg.head_dim
        z = lambda *s, dt=dtype: torch.zeros(s, device=device, dtype=dt)
        # attention ring: [prev chunk | current chunk], post-RoPE keys (in re-based [0, 2W) frame), raw keys, and values
        self.k_prev, self.v_prev = z(batch, G, W, D), z(batch, G, W, D)
        self.k_cur, self.v_cur = z(batch, G, W, D), z(batch, G, W, D)
        self.kraw_prev, self.kraw_cur = z(batch, G, W, D), z(batch, G, W, D)
        # attention sinks: pre-RoPE raw keys and values up to max_sink (rotated when active)
        S_cap = max(1, cfg.n_sink, cfg.max_sink if cfg.auto_pin_system else cfg.n_sink)
        self.k_sink_raw, self.v_sink = z(batch, G, S_cap, D), z(batch, G, S_cap, D)
        self.k_sink_rot = z(batch, G, S_cap, D)
        # fixed-budget Segment-Evicted KV Memory bank (pre-RoPE raw keys, rotated keys, values)
        B_cap = max(1, seg_budget)
        self.k_seg_raw, self.v_seg = z(batch, G, B_cap, D), z(batch, G, B_cap, D)
        self.k_seg_rot = z(batch, G, B_cap, D)


class VAULTv3State:
    def __init__(self, cfg: VAULTv3Config, batch: int, device, dtype=torch.bfloat16, seg_budget: Optional[int] = None):
        self.cfg = cfg
        self.pos = 0
        self.seg_budget = cfg.seg_budget if seg_budget is None else int(seg_budget)
        self.n_pin = min(cfg.n_sink, cfg.window)
        self.pin_locked = not cfg.auto_pin_system
        self.saw_im_start = False
        self.awaiting_sys_nl = False
        # Segment scorer, APL causal EMA habituation & shingle resonance state (batch=1 for variable segment bank)
        W = cfg.window
        feat_dim = max(1, len(cfg.seg_layers) * cfg.d_model)
        self.ema_ready = False
        self.seg_feat_ema = torch.zeros((feat_dim,), device=device, dtype=torch.float32)
        self.tok_prev = torch.zeros((batch, W), device=device, dtype=torch.long)
        self.tok_cur = torch.zeros((batch, W), device=device, dtype=torch.long)
        self.sc_prev = torch.zeros((batch, W), device=device, dtype=torch.float32)
        self.sc_cur = torch.zeros((batch, W), device=device, dtype=torch.float32)
        self.seg_id_offset = 0
        self.seg_len = 0
        self.cur_seg_id = 0
        self.seg_prev_ids = torch.zeros((batch, W), device=device, dtype=torch.long)
        self.seg_cur_ids = torch.zeros((batch, W), device=device, dtype=torch.long)
        self.mem_pos = torch.empty(0, dtype=torch.long, device=device)
        self.mem_score = torch.empty(0, dtype=torch.float32, device=device)
        self.mem_tok = torch.empty(0, dtype=torch.long, device=device)
        self.mem_seg = torch.empty(0, dtype=torch.long, device=device)
        self.n_uniq_segs = 0
        self.seg_inv: List[int] = []
        self.sh_df: Dict[int, int] = {}
        self.sh_postings: Dict[int, List[Tuple[int, int]]] = {}
        self.seg_bias_mode: int = 0
        self.seg_bias_active = False
        self.seg_head_bias: Optional[torch.Tensor] = None       # (1, H, 1, M_cnt) float32 (all layers or late layers)
        self.seg_head_bias_hop1: Optional[torch.Tensor] = None  # (1, H, 1, M_cnt) float32 (early layers when mode == 2)
        self.layers = [LayerState(cfg, batch, device, dtype, seg_budget=self.seg_budget) for _ in range(cfg.n_layers)]
        # Precomputed -W RoPE shift for re-basing k_prev at chunk boundaries (identical to C cos_w/sin_w)
        self.cos_neg_w, self.sin_neg_w = rope_cos_sin(torch.tensor([-W], device=device), cfg.head_dim, cfg.rope_theta)

    def set_pinned(self, n_pin: int) -> None:
        cap = min(self.cfg.window, self.layers[0].k_sink_raw.shape[2])
        self.n_pin = max(0, min(int(n_pin), cap))
        self.pin_locked = True

    def nbytes(self) -> int:
        tot = 0
        for ls in self.layers:
            for t in vars(ls).values():
                if isinstance(t, torch.Tensor):
                    tot += t.numel() * t.element_size()
        return tot


# ==============================================================================
# Transformer
# ==============================================================================

class VAULTv3Attention(nn.Module):
    def __init__(self, cfg: VAULTv3Config):
        super().__init__()
        self.cfg = cfg
        H, G, D, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_model
        self.q_proj = nn.Linear(d, H * D, bias=True)
        self.k_proj = nn.Linear(d, G * D, bias=True)
        self.v_proj = nn.Linear(d, G * D, bias=True)
        self.o_proj = nn.Linear(H * D, d, bias=False)

    def _qkv(self, h: torch.Tensor):
        B, L, _ = h.shape
        cfg = self.cfg
        q = self.q_proj(h).view(B, L, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        mode: str = "block", collect: Optional[dict] = None, n_pin: Optional[int] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, L, _ = h.shape
        W = cfg.window
        S = cfg.n_sink if n_pin is None else n_pin
        q, k, v = self._qkv(h)
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        group = cfg.group_size

        if mode == "full" or L <= 2 * W:
            attn = F.scaled_dot_product_attention(qr, _expand_kv(kr, group), _expand_kv(v, group), is_causal=True)
        else:
            attn = block_sliding_attention(qr, kr, v, W, k_raw=k, n_sink=S, head_dim=cfg.head_dim, rope_theta=cfg.rope_theta)

        if collect is not None:
            collect.update(dict(kr=kr, v=v, k=k))

        out = attn.transpose(1, 2).reshape(B, L, cfg.n_heads * cfg.head_dim)
        return self.o_proj(out)

    @torch.no_grad()
    def step(
        self, h: torch.Tensor, ls: LayerState, st: VAULTv3State,
        seg_active: bool = False, layer_idx: int = 0,
    ) -> torch.Tensor:
        """One-token recurrent step in re-based [-(S+M)..2W) RoPE coordinates (mirrors C v3_step)."""
        cfg = self.cfg
        pos = st.pos
        W, D, group = cfg.window, cfg.head_dim, cfg.group_size
        c = pos % W
        eff = (W + c) if pos >= W else c

        q, k, v = self._qkv(h)                                   # (B, H|G, 1, D)
        cos, sin = rope_cos_sin(torch.tensor([eff], device=h.device), D, cfg.rope_theta)
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        ls.k_cur[:, :, c] = kr[:, :, 0].to(ls.k_cur.dtype)
        ls.v_cur[:, :, c] = v[:, :, 0].to(ls.v_cur.dtype)
        ls.kraw_cur[:, :, c] = k[:, :, 0].to(ls.kraw_cur.dtype)

        S_cap = ls.k_sink_raw.shape[2]
        if pos < S_cap:
            ls.k_sink_raw[:, :, pos] = k[:, :, 0].to(ls.k_sink_raw.dtype)
            ls.v_sink[:, :, pos] = v[:, :, 0].to(ls.v_sink.dtype)

        M_cnt = 0
        S = 0
        if pos >= W:
            keys = torch.cat([ls.k_prev, ls.k_cur[:, :, : c + 1]], dim=2)
            vals = torch.cat([ls.v_prev, ls.v_cur[:, :, : c + 1]], dim=2)
            if pos >= 2 * W:
                M_cnt = int(st.mem_pos.numel()) if seg_active else 0
                if M_cnt > 0:
                    keys = torch.cat([ls.k_seg_rot[:, :, :M_cnt], keys], dim=2)
                    vals = torch.cat([ls.v_seg[:, :, :M_cnt], vals], dim=2)
                S = st.n_pin
                if S > 0:
                    keys = torch.cat([ls.k_sink_rot[:, :, :S], keys], dim=2)
                    vals = torch.cat([ls.v_sink[:, :, :S], vals], dim=2)
        else:
            keys, vals = ls.k_cur[:, :, : c + 1], ls.v_cur[:, :, : c + 1]

        kk_e = _expand_kv(keys.to(qr.dtype), group)
        vv_e = _expand_kv(vals.to(qr.dtype), group)
        l_lo = cfg.n_layers // 4
        l_mid = cfg.n_layers // 2
        l_hi = cfg.n_layers - cfg.n_layers // 14
        if (
            seg_active
            and st.seg_bias_active
            and st.seg_head_bias is not None
            and M_cnt > 0
            and l_lo <= layer_idx < l_hi
        ):
            hb = (
                st.seg_head_bias_hop1
                if (st.seg_bias_mode == 2 and layer_idx < l_mid and st.seg_head_bias_hop1 is not None)
                else st.seg_head_bias
            )
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(qr, kk_e.transpose(-1, -2)).float() * scale
            scores[:, :, :, S : S + M_cnt] = scores[:, :, :, S : S + M_cnt] + hb
            probs = F.softmax(scores, dim=-1).to(qr.dtype)
            attn = torch.matmul(probs, vv_e)
        else:
            attn = F.scaled_dot_product_attention(qr, kk_e, vv_e)

        B = h.shape[0]
        return self.o_proj(attn.transpose(1, 2).reshape(B, 1, cfg.n_heads * D))


class VAULTv3MLP(nn.Module):
    def __init__(self, cfg: VAULTv3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VAULTv3Block(nn.Module):
    def __init__(self, cfg: VAULTv3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = VAULTv3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = VAULTv3MLP(cfg)

    def forward(self, x, cos, sin, mode="block", collect=None, n_pin=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mode=mode, collect=collect, n_pin=n_pin)
        return x + self.mlp(self.post_attention_layernorm(x))


class VAULTv3ForCausalLM(nn.Module):
    def __init__(self, cfg: VAULTv3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([VAULTv3Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        if len(cfg.seg_layers) > 0:
            self.seg_scorer: Optional[SegmentScorer] = SegmentScorer(len(cfg.seg_layers) * cfg.d_model, cfg.seg_hidden)
        else:
            self.seg_scorer = None
        self.grad_checkpoint = False
        self._brk_mask: Optional[torch.Tensor] = None
        self._tier_lut: Optional[torch.Tensor] = None
        self._raw_bytes: Optional[List[bytes]] = None
        self._ema_mats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def _get_raw_bytes(self) -> List[bytes]:
        if self._raw_bytes is None:
            self._raw_bytes = build_token_raw_bytes(self.cfg.vocab_size)
        return self._raw_bytes

    def _get_tier_lut(self) -> torch.Tensor:
        if self._tier_lut is None:
            self._tier_lut = build_token_tier_lut(self.cfg.vocab_size)
        return self._tier_lut

    def _get_brk_mask(self) -> torch.Tensor:
        if self._brk_mask is None:
            self._brk_mask = build_token_break_mask(self.cfg.vocab_size)
        return self._brk_mask

    def _get_ema_mats(self, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W = self.cfg.window
        if self._ema_mats is None or self._ema_mats[0].device != dev or self._ema_mats[0].shape[0] != W:
            idx = torch.arange(W, device=dev, dtype=torch.float32)
            diff = idx[:, None] - idx[None, :]
            L_mat = torch.where(diff >= 0, 0.015 * (0.985 ** diff.clamp(min=0)), torch.zeros_like(diff))
            L_init0 = L_mat.clone()
            L_init0[:, 0] = 0.985 ** idx
            pow_lam = 0.985 ** (idx + 1.0)
            self._ema_mats = (L_mat, L_init0, pow_lam)
        return self._ema_mats

    # ---- parameter groups ----
    def scorer_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith("seg_scorer.")]

    def backbone_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("seg_scorer.")]

    def freeze_backbone(self) -> None:
        for p in self.backbone_parameters():
            p.requires_grad_(False)
        for p in self.scorer_parameters():
            p.requires_grad_(True)

    # ---- APL habituation, shingle resonance & authority-spoofing quarantine helpers ----
    def _score_chunk_habituated(self, st: VAULTv3State, f_cat: torch.Tensor) -> torch.Tensor:
        """Exact vectorized APL causal feature habituation matching runtime/vault3.c token-by-token."""
        assert self.seg_scorer is not None
        clen = int(f_cat.shape[0])
        if clen == 0:
            return torch.empty(0, device=f_cat.device, dtype=torch.float32)
        L_mat, L_init0, pow_lam = self._get_ema_mats(f_cat.device)
        raw_sc = self.seg_scorer(f_cat)
        if not st.ema_ready:
            e_after = L_init0[:clen, :clen] @ f_cat
            e_before = torch.cat([f_cat[:1], e_after[:-1]], dim=0)
            hab_sc = self.seg_scorer(f_cat - 0.65 * e_before)
            sc = raw_sc + 0.35 * F.relu(hab_sc - raw_sc)
            sc[0] = raw_sc[0]
            st.seg_feat_ema.copy_(e_after[-1])
            st.ema_ready = True
        else:
            ema0 = st.seg_feat_ema
            e_after = pow_lam[:clen, None] * ema0[None, :] + (L_mat[:clen, :clen] @ f_cat)
            e_before = torch.cat([ema0[None, :], e_after[:-1]], dim=0)
            hab_sc = self.seg_scorer(f_cat - 0.65 * e_before)
            sc = raw_sc + 0.35 * F.relu(hab_sc - raw_sc)
            st.seg_feat_ema.copy_(e_after[-1])
        return sc

    def _rebuild_shingle_index(self, st: VAULTv3State) -> None:
        """Rebuild 2-tier tagged FNV-1a shingle index over Segment Memory (matches runtime/vault3.c)."""
        M_cnt = int(st.mem_pos.numel())
        st.seg_bias_mode = 0
        st.seg_bias_active = False
        st.seg_head_bias = None
        st.seg_head_bias_hop1 = None
        st.sh_df = {}
        st.sh_postings = {}
        if M_cnt == 0:
            st.n_uniq_segs = 0
            st.seg_inv = []
            return
        mem_seg_l = st.mem_seg.tolist()
        mem_tok_l = st.mem_tok.tolist()
        seg_inv: List[int] = []
        n_s = 0
        for j in range(M_cnt):
            if j > 0 and mem_seg_l[j] != mem_seg_l[j - 1]:
                n_s += 1
            seg_inv.append(n_s)
        N_s = n_s + 1
        st.n_uniq_segs = N_s
        st.seg_inv = seg_inv
        if N_s <= 2:
            return

        raw_lut = self._get_raw_bytes()
        seg_sh: List[List[int]] = []
        df: Dict[int, int] = {}
        j = 0
        for sid in range(N_s):
            s_toks: List[int] = []
            while j < M_cnt and seg_inv[j] == sid:
                s_toks.append(mem_tok_l[j])
                j += 1
            shs = _extract_norm_shingles(_norm_shingle_bytes_from_tokens(s_toks, raw_lut, cap=8192), max_out=2048)
            seg_sh.append(shs)
            for sh in shs:
                df[sh] = df.get(sh, 0) + 1

        postings: Dict[int, List[Tuple[int, int]]] = {}
        for sid, shs in enumerate(seg_sh):
            for sh in shs:
                d = df[sh]
                is_id = (sh & 1) != 0
                keep = (
                    (((sh & 3) == 3 and d == 2) or (d * 4 <= N_s + 6))
                    if is_id
                    else (d <= 16 and d * 5 <= N_s * 2 + 4)
                )
                if keep:
                    lg = math.log((N_s + 1.0) / (d + 0.5))
                    mult = 2048.0 if (sh & 3) == 3 else (1024.0 if is_id else 768.0)
                    w_int = int(mult * lg * lg + 0.5)
                    postings.setdefault(sh, []).append((sid, w_int))
        st.sh_df = df
        st.sh_postings = postings

    @staticmethod
    def _find_rollback_cue_pos(raw_lc: bytes) -> int:
        for kw in (b"rollback", b"revert", b"restore"):
            p = raw_lc.find(kw)
            if p >= 0:
                return p
        p = raw_lc.find(b"roll ")
        if p >= 0 and raw_lc.find(b" back", p) >= 0:
            return p
        return -1

    def _eval_query_shingles_to_tok_biases(
        self, st: VAULTv3State, q_sh: List[int],
    ) -> Optional[Tuple[List[float], List[float], Optional[List[float]], int]]:
        if len(q_sh) < 2 or not st.sh_postings or st.n_uniq_segs <= 2:
            return None
        N_s = st.n_uniq_segs
        postings = st.sh_postings
        df_map = st.sh_df
        n_q_id = sum(1 for sh in q_sh if (sh & 1) != 0)
        n_q_alpha = len(q_sh) - n_q_id
        whole_flag = 1 << 63
        qbits_mask = whole_flag - 1

        scores = [0] * N_s
        hits = [0] * N_s
        q_mask = [0] * N_s
        mx = 0
        matched: List[int] = []
        used_tier_a = False
        max_hits = 0

        # Tier A: strict identifier shingles take priority whenever the query contains ID tokens
        if n_q_id >= 2:
            bit_i = 0
            for sh in q_sh:
                if (sh & 1) == 0:
                    continue
                bit = (1 << (bit_i % 63)) | (whole_flag if (sh & 3) == 3 else 0)
                bit_i += 1
                plist = postings.get(sh)
                if plist is not None:
                    for sid, w_int in plist:
                        ns = scores[sid] + w_int
                        scores[sid] = ns
                        hits[sid] += 1
                        q_mask[sid] |= bit
                        if ns > mx:
                            mx = ns
            if mx >= 2048:
                n_at_mx = 0
                best_whole = 0
                best_qbits = 0
                for sid in range(N_s):
                    if hits[sid] >= 2 and scores[sid] == mx:
                        n_at_mx += 1
                        best_whole |= (q_mask[sid] & whole_flag)
                        best_qbits |= (q_mask[sid] & qbits_mask)
                for sid in range(N_s):
                    if hits[sid] >= 2 and scores[sid] * 2 >= mx:
                        ms = q_mask[sid]
                        if (
                            n_at_mx == 1
                            and best_whole != 0
                            and (ms & whole_flag) == 0
                            and ((ms & qbits_mask) & ~best_qbits) == 0
                        ):
                            continue
                        matched.append(sid)
                if matched:
                    used_tier_a = True

        # Tier B: pure-alphabetic content whole-word fallback when Tier A yields zero matches
        if not matched and n_q_alpha >= 2:
            scores = [0] * N_s
            hits = [0] * N_s
            q_mask = [0] * N_s
            mx = 0
            bit_i = 0
            for sh in q_sh:
                if (sh & 3) != 0:
                    continue
                bit = 1 << (bit_i % 63)
                bit_i += 1
                plist = postings.get(sh)
                if plist is not None:
                    for sid, w_int in plist:
                        ns = scores[sid] + w_int
                        scores[sid] = ns
                        hits[sid] += 1
                        q_mask[sid] |= bit
                        if ns > mx:
                            mx = ns
            if mx >= 2500:
                n_at_mx = 0
                best_qbits = 0
                hits_at_mx = 0
                for sid in range(N_s):
                    if hits[sid] >= 2 and scores[sid] == mx:
                        n_at_mx += 1
                        best_qbits |= (q_mask[sid] & qbits_mask)
                        if hits[sid] > hits_at_mx:
                            hits_at_mx = hits[sid]
                for sid in range(N_s):
                    cond = (scores[sid] * 4 >= mx * 3) if hits_at_mx >= 3 else (scores[sid] * 2 >= mx)
                    if hits[sid] >= 2 and cond:
                        ms = q_mask[sid] & qbits_mask
                        if n_at_mx == 1 and scores[sid] < mx and (ms & ~best_qbits) == 0:
                            continue
                        matched.append(sid)
                        if hits[sid] > max_hits:
                            max_hits = hits[sid]

        if not matched:
            return None

        seg_inv = st.seg_inv
        mem_tok_l = st.mem_tok.tolist()
        mem_seg_l = st.mem_seg.tolist()
        M_cnt = len(mem_tok_l)
        bg = (
            0.0
            if (used_tier_a or max_hits < 3)
            else -(1.0 + (math.log(M_cnt / 1024.0) if M_cnt > 1024 else 0.0))
        )
        b_arch_s = [bg] * N_s
        b_sup_s = [bg] * N_s
        b_hop1_s = [0.0] * N_s
        mode = 1

        s_pivot = matched[-1]
        raw_lut = self._get_raw_bytes()
        n_lut = len(raw_lut)
        raw_buf = bytearray()
        for tid, sid in zip(mem_tok_l, seg_inv):
            if sid != s_pivot:
                continue
            tb = raw_lut[tid] if 0 <= tid < n_lut else b""
            for ch in tb:
                if len(raw_buf) >= 4096:
                    break
                raw_buf.append(ch + 32 if 0x41 <= ch <= 0x5A else ch)
        raw_lc = bytes(raw_buf)
        q_sh_set = set(q_sh)

        # Case 1: Rollback pointer resolution ("rollback / revert / restore / roll ... back" -> [t=K])
        cue_pos = self._find_rollback_cue_pos(raw_lc)
        s_rb = -1
        if cue_pos >= 0 and s_pivot > 0:
            to_abs = raw_lc.find(b" to ", cue_pos)
            starts = [to_abs, cue_pos] if to_abs >= 0 else [cue_pos]
            for st_pos in starts:
                if s_rb >= 0:
                    break
                piv_sh = _extract_norm_shingles(_norm_shingle_bytes_from_raw(raw_lc[st_pos:], cap=4096), max_out=1024)
                for kh in piv_sh:
                    if (kh & 3) != 3 or kh in q_sh_set:
                        continue
                    plist = postings.get(kh)
                    if plist is None or df_map.get(kh, 99) > 3:
                        continue
                    for csid, _ in plist:
                        if csid < s_pivot and csid > s_rb:
                            s_rb = csid

        # Case 2: Bounded 1-step Hop-2 relational join when Tier-A Hop-1 isolates s0 and s0 has a df==2 bridge ID
        hop2: List[int] = []
        if s_rb < 0 and len(matched) == 1 and used_tier_a:
            s0 = matched[0]
            piv_sh = _extract_norm_shingles(_norm_shingle_bytes_from_raw(raw_lc, cap=4096), max_out=1024)
            overflow_h2 = False
            for kh in piv_sh:
                if (kh & 3) != 3 or kh in q_sh_set:
                    continue
                plist = postings.get(kh)
                if plist is None or df_map.get(kh, 0) != 2:
                    continue
                for csid, _ in plist:
                    if csid == s0 or csid in hop2:
                        continue
                    if len(hop2) < 2:
                        hop2.append(csid)
                    else:
                        overflow_h2 = True
            if overflow_h2:
                hop2.clear()

        has_sys_pin = st.n_pin > max(16, self.cfg.n_sink)
        if s_rb >= 0:
            mode = 2
            b_arch_s = [0.0] * N_s
            b_sup_s = [0.0] * N_s
            for sid in matched:
                b_hop1_s[sid] = -0.3
                b_arch_s[sid] = 0.4
                b_sup_s[sid] = -0.5
            b_hop1_s[s_pivot] = 2.0
            b_hop1_s[s_rb] = 2.0
            b_arch_s[s_pivot] = 1.2
            b_arch_s[s_rb] = 2.2
            b_sup_s[s_pivot] = 0.8
            b_sup_s[s_rb] = 2.8
        elif hop2:
            mode = 2
            b_arch_s = [0.0] * N_s
            b_sup_s = [0.0] * N_s
            s0 = matched[0]
            b_hop1_s[s0] = 2.4
            b_arch_s[s0] = 1.0
            b_sup_s[s0] = 0.6
            for s2 in hop2:
                b_hop1_s[s2] = 1.0
                b_arch_s[s2] = 2.4
                b_sup_s[s2] = 2.8
        elif len(matched) == 1:
            s0 = matched[0]
            if not used_tier_a:
                ba, bs = (2.2, 2.5) if max_hits >= 3 else (0.5, 0.7)
                if has_sys_pin:
                    ba = min(ba, 1.5)
                    bs = min(bs, 1.9)
                b_arch_s[s0] = ba
                b_sup_s[s0] = bs
            else:
                b_arch_s[s0] = 1.5 if has_sys_pin else 2.2
                b_sup_s[s0] = 1.9 if has_sys_pin else 2.5
        else:
            s_last = matched[-1]
            last_mask = q_mask[s_last] & qbits_mask
            inv_mx = 1.0 / float(mx)
            scale_tb = 1.0 if (used_tier_a or max_hits >= 3) else 0.5
            for sid in matched:
                rel = float(scores[sid]) * inv_mx * scale_tb
                if sid == s_last:
                    b_arch_s[sid] = 1.5 * rel
                    b_sup_s[sid] = 2.5 * rel
                elif (q_mask[sid] & last_mask) != 0:
                    b_arch_s[sid] = 1.5 * rel
                    b_sup_s[sid] = -0.4
                else:
                    b_arch_s[sid] = 1.8 * rel
                    b_sup_s[sid] = 2.0 * rel

        arch_tok = [b_arch_s[sid] for sid in seg_inv]
        sup_tok = [b_sup_s[sid] for sid in seg_inv]
        hop1_tok = [b_hop1_s[sid] for sid in seg_inv] if mode == 2 else None

        # O(1) sub-segment clause refinement for mode == 1 when a matched segment packs >=2 competing self-contained records
        if mode == 1 and len(matched) <= 2:
            active_q = {sh for sh in q_sh if ((sh & 1) != 0 if used_tier_a else (sh & 3) == 0) and sh in postings}
            for sid in matched:
                ks = [k for k in range(M_cnt) if seg_inv[k] == sid]
                if len(ks) < 8 or not active_q:
                    continue
                clauses: List[List[int]] = []
                c0 = 0
                for idx_k in range(len(ks)):
                    tb = raw_lut[mem_tok_l[ks[idx_k]]] if 0 <= mem_tok_l[ks[idx_k]] < n_lut else b""
                    is_delim = any(ch in tb for ch in (0x0A, 0x7C, 0x09, 0x3B))
                    if is_delim and (idx_k - c0 + 1) >= 4 and (len(ks) - (idx_k + 1)) >= 4 and len(clauses) + 1 < 32:
                        clauses.append(ks[c0 : idx_k + 1])
                        c0 = idx_k + 1
                if c0 < len(ks) and len(clauses) < 32:
                    clauses.append(ks[c0:])
                if len(clauses) < 2:
                    continue
                c_scores: List[int] = []
                c_self_cont: List[bool] = []
                for cl in clauses:
                    ctoks = [mem_tok_l[k] for k in cl]
                    csh = _extract_norm_shingles(_norm_shingle_bytes_from_tokens(ctoks, raw_lut, cap=2048), max_out=512)
                    tier_sh = [sh for sh in csh if ((sh & 1) != 0 if used_tier_a else (sh & 3) == 0)]
                    sc = 0
                    for sh in tier_sh:
                        if sh in active_q:
                            for psid, pw in postings.get(sh, ()):
                                if psid == sid:
                                    sc += pw
                                    break
                    raw_cl = b"".join(raw_lut[t] for t in ctoks if 0 <= t < n_lut)[:2048]
                    stripped = raw_cl.rstrip(b" \t\r\n|,;")
                    has_kv = ((b"=" in stripped[:-1]) or (b":" in stripped[:-1])) if len(stripped) > 2 else False
                    self_cont = (len(cl) >= 9 or has_kv) and (len(tier_sh) >= 2)
                    c_scores.append(sc)
                    c_self_cont.append(self_cont)
                mx_c = max(c_scores)
                if mx_c > 0 and any(sc < mx_c and sc_ok for sc, sc_ok in zip(c_scores, c_self_cont)):
                    sib_val = min(0.0, bg * 0.5)
                    for cl, sc, sc_ok in zip(clauses, c_scores, c_self_cont):
                        if sc * 4 < mx_c * 3 and sc_ok:
                            for k in cl:
                                arch_tok[k] = sib_val
                                sup_tok[k] = sib_val

        # O(1) Orphan-Value Continuation Halo: if single matched segment s0 ends at '\n' and contiguous
        # pre-eviction segment s0+1 starts with a short (<10-tok, <2 alpha-word) orphan value line, keep it coupled.
        if mode == 1 and len(matched) == 1:
            s0 = matched[0]
            if s0 + 1 < N_s:
                ks0 = [k for k in range(M_cnt) if seg_inv[k] == s0]
                ks1 = [k for k in range(M_cnt) if seg_inv[k] == s0 + 1]
                if ks0 and ks1 and mem_seg_l[ks1[0]] == mem_seg_l[ks0[0]] + 1:
                    last_tb = raw_lut[mem_tok_l[ks0[-1]]] if 0 <= mem_tok_l[ks0[-1]] < n_lut else b""
                    if 0x0A in last_tb and arch_tok[ks0[-1]] > 0.0:
                        nl_idx = -1
                        for idx1 in range(min(10, len(ks1))):
                            tb1 = raw_lut[mem_tok_l[ks1[idx1]]] if 0 <= mem_tok_l[ks1[idx1]] < n_lut else b""
                            if 0x0A in tb1:
                                nl_idx = idx1
                                break
                        if 0 <= nl_idx < 10:
                            pref_toks = [mem_tok_l[ks1[j]] for j in range(nl_idx + 1)]
                            p_sh = _extract_norm_shingles(_norm_shingle_bytes_from_tokens(pref_toks, raw_lut, cap=512), max_out=64)
                            if sum(1 for sh in p_sh if (sh & 3) == 0) < 2:
                                ba0 = arch_tok[ks0[-1]]
                                bs0 = sup_tok[ks0[-1]]
                                for j in range(nl_idx + 1):
                                    arch_tok[ks1[j]] = ba0
                                    sup_tok[ks1[j]] = bs0
        return arch_tok, sup_tok, hop1_tok, mode

    def _pack_head_bias(self, arch_2d: torch.Tensor, sup_2d: torch.Tensor) -> torch.Tensor:
        """arch_2d, sup_2d: (Q, M_cnt) float32 -> (1, H, Q, M_cnt) float32 (3-way free/arch/sup head split)."""
        cfg = self.cfg
        H, grp = cfg.n_heads, cfg.group_size
        free_h = grp // 3
        sup_h = (grp - free_h) // 2
        arch_end = grp - sup_h
        z_2d = torch.zeros_like(arch_2d)
        cols = []
        for h in range(H):
            hr = h % grp
            cols.append(z_2d if hr < free_h else (arch_2d if hr < arch_end else sup_2d))
        return torch.stack(cols, dim=0).unsqueeze(0)

    def _assistant_query_span(
        self, hist_toks: List[int], hi: int, pos: int, n_pin: int, raw_lut: List[bytes],
    ) -> Optional[Tuple[int, int, int]]:
        """
        Exact ChatML turn-gated query span shared 1:1 with runtime/vault3.c:update_query_resonance.
        Fires ONLY when a non-pinned <|im_end|> appears within the last 38 tokens, and bounds the
        shingle lookback strictly to the user's final question line (stopping at <|im_start|> or '\\n').
        """
        cfg = self.cfg
        im_end = cfg.im_end_id
        im_start = cfg.im_start_id
        search_lo = max(0, hi - 127)
        p_end = -1
        for p in range(hi, search_lo - 1, -1):
            if hist_toks[p] == im_end:
                p_end = p
                break
        if p_end <= 0:
            return None
        gen_dist = hi - p_end
        if gen_dist >= 38:
            return None
        if pos - gen_dist < n_pin:
            return None
        scan_from = p_end - 1
        q_stop = max(0, scan_from - 48)
        lo_t = q_stop
        n_lut = len(raw_lut)
        for p in range(scan_from, q_stop - 1, -1):
            tid = hist_toks[p]
            if tid == im_start:
                lo_t = p + 1
                break
            if p + 4 <= scan_from:
                tb = raw_lut[tid] if 0 <= tid < n_lut else b""
                if 0x0A in tb:
                    lo_t = p + 1
                    break
        if lo_t > scan_from:
            return None
        return (lo_t, scan_from, gen_dist)

    def _update_step_resonance(self, st: VAULTv3State, c: int) -> None:
        W = self.cfg.window
        M_cnt = int(st.mem_pos.numel())
        st.seg_bias_mode = 0
        st.seg_bias_active = False
        st.seg_head_bias = None
        st.seg_head_bias_hop1 = None
        if M_cnt == 0 or st.n_uniq_segs <= 2 or not st.sh_postings:
            return
        raw_lut = self._get_raw_bytes()
        hist_toks = st.tok_prev[0].tolist() + st.tok_cur[0, : c + 1].tolist()
        span = self._assistant_query_span(hist_toks, W + c, st.pos, st.n_pin, raw_lut)
        if span is None:
            return
        lo_t, scan_from, gen_dist = span
        atten = 1.0 if gen_dist <= 18 else ((38 - gen_dist) * 0.05)
        q_toks = hist_toks[lo_t : scan_from + 1]
        q_sh = _extract_norm_shingles(_norm_shingle_bytes_from_tokens(q_toks, raw_lut, cap=4096), max_out=1024)
        res = self._eval_query_shingles_to_tok_biases(st, q_sh)
        if res is None:
            return
        arch_tok, sup_tok, hop1_tok, mode = res
        dev = self.embed_tokens.weight.device
        atten_f = float(torch.tensor(atten, dtype=torch.float32).item())
        a_t = torch.tensor([arch_tok], dtype=torch.float32, device=dev) * atten_f
        s_t = torch.tensor([sup_tok], dtype=torch.float32, device=dev) * atten_f
        st.seg_head_bias = self._pack_head_bias(a_t, s_t)
        if mode == 2 and hop1_tok is not None:
            h1_t = torch.tensor([hop1_tok], dtype=torch.float32, device=dev) * atten_f
            st.seg_head_bias_hop1 = self._pack_head_bias(h1_t, h1_t)
        st.seg_bias_mode = mode
        st.seg_bias_active = True

    def _build_chunk_resonance_bias(
        self, st: VAULTv3State, r0: int, tok_c_1d: List[int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Exact chunk-level query resonance matching _update_step_resonance at every position 0..clen-1."""
        M_cnt = int(st.mem_pos.numel())
        clen = len(tok_c_1d)
        if M_cnt == 0 or clen == 0 or st.n_uniq_segs <= 2 or not st.sh_postings:
            return None
        W = self.cfg.window
        im_end = self.cfg.im_end_id
        tail_prev = st.tok_prev[0, max(0, W - 127) : W].tolist()
        if (im_end not in tok_c_1d) and (im_end not in tail_prev):
            return None

        hist_toks = st.tok_prev[0].tolist() + tok_c_1d
        raw_lut = self._get_raw_bytes()
        zero_row = [0.0] * M_cnt
        early_arch_rows: List[List[float]] = []
        early_sup_rows: List[List[float]] = []
        late_arch_rows: List[List[float]] = []
        late_sup_rows: List[List[float]] = []
        any_active = False
        any_mode2 = False
        prev_key: Optional[Tuple[int, int]] = (-2, -2)
        prev_res: Optional[Tuple[List[float], List[float], Optional[List[float]], int]] = None

        for j in range(clen):
            span = self._assistant_query_span(hist_toks, W + j, r0 + j, st.n_pin, raw_lut)
            if span is None:
                res = None
                gen_dist = 0
            else:
                lo_t, scan_from, gen_dist = span
                key = (lo_t, scan_from)
                if key == prev_key:
                    res = prev_res
                else:
                    prev_key = key
                    q_toks = hist_toks[lo_t : scan_from + 1]
                    q_sh = _extract_norm_shingles(_norm_shingle_bytes_from_tokens(q_toks, raw_lut, cap=4096), max_out=1024)
                    res = self._eval_query_shingles_to_tok_biases(st, q_sh)
                    prev_res = res
            if res is None:
                early_arch_rows.append(zero_row)
                early_sup_rows.append(zero_row)
                late_arch_rows.append(zero_row)
                late_sup_rows.append(zero_row)
            else:
                any_active = True
                arch_tok, sup_tok, hop1_tok, mode = res
                atten = 1.0 if gen_dist <= 18 else ((38 - gen_dist) * 0.05)
                la = [v * atten for v in arch_tok] if atten != 1.0 else arch_tok
                ls = [v * atten for v in sup_tok] if atten != 1.0 else sup_tok
                late_arch_rows.append(la)
                late_sup_rows.append(ls)
                if mode == 2 and hop1_tok is not None:
                    any_mode2 = True
                    h1 = [v * atten for v in hop1_tok] if atten != 1.0 else hop1_tok
                    early_arch_rows.append(h1)
                    early_sup_rows.append(h1)
                else:
                    early_arch_rows.append(la)
                    early_sup_rows.append(ls)

        if not any_active:
            return None
        dev = self.embed_tokens.weight.device
        la_2d = torch.tensor(late_arch_rows, dtype=torch.float32, device=dev)
        ls_2d = torch.tensor(late_sup_rows, dtype=torch.float32, device=dev)
        late_bias = self._pack_head_bias(la_2d, ls_2d)
        if not any_mode2:
            return (late_bias, late_bias)
        ea_2d = torch.tensor(early_arch_rows, dtype=torch.float32, device=dev)
        es_2d = torch.tensor(early_sup_rows, dtype=torch.float32, device=dev)
        early_bias = self._pack_head_bias(ea_2d, es_2d)
        return (early_bias, late_bias)

    def _check_spoof_step(self, st: VAULTv3State, pos: int, c: int) -> bool:
        cfg = self.cfg
        if st.n_pin <= cfg.n_sink or pos < st.n_pin:
            return False
        W = cfg.window
        post_pin = pos - st.n_pin + 1
        lb = min(24, post_pin)
        avail_cur = c + 1
        avail_tot = (W + avail_cur) if pos >= W else avail_cur
        lb = min(lb, avail_tot)
        need_prev = (lb - avail_cur) if avail_cur < lb else 0
        cur_start = (avail_cur - lb) if avail_cur > lb else 0
        toks = (st.tok_prev[0, W - need_prev : W].tolist() if need_prev > 0 else []) + st.tok_cur[0, cur_start : c + 1].tolist()
        raw_lut = self._get_raw_bytes()
        n_lut = len(raw_lut)
        buf = bytearray()
        for tid in toks:
            b = raw_lut[tid] if 0 <= tid < n_lut else b""
            for ch in b:
                if len(buf) >= 4096:
                    break
                buf.append(ch + 32 if 0x41 <= ch <= 0x5A else ch)
        return (b"opi_signed_directive" in buf) or (b"urgent opi delegation" in buf)

    def _apply_spoof_chunk(self, st: VAULTv3State, r0: int, tok_c_1d: List[int], seg_active: bool) -> None:
        if not seg_active:
            return
        cfg = self.cfg
        clen = len(tok_c_1d)
        if st.n_pin <= cfg.n_sink or r0 + clen <= st.n_pin:
            return
        W = cfg.window
        raw_lut = self._get_raw_bytes()
        n_lut = len(raw_lut)
        prev_l = st.tok_prev[0].tolist() if r0 >= W else []
        all_lc = b"".join((raw_lut[tid].lower() if 0 <= tid < n_lut else b"") for tid in (prev_l + tok_c_1d))
        if (b"opi_signed_directive" not in all_lc) and (b"urgent opi delegation" not in all_lc):
            return
        hit_idx: List[int] = []
        for j in range(clen):
            if self._check_spoof_step(st, r0 + j, j):
                hit_idx.append(j)
        if hit_idx:
            idx_t = torch.tensor(hit_idx, dtype=torch.long, device=st.sc_cur.device)
            st.sc_cur[0, idx_t] = -1.0e30

    # ---- streaming boundary rotation & eviction helpers ----
    def _refresh_prefix_rotations(self, st: VAULTv3State, seg_active: bool) -> None:
        cfg = self.cfg
        dev = self.embed_tokens.weight.device
        M_cnt = int(st.mem_pos.numel()) if seg_active else 0
        S = st.n_pin
        if M_cnt > 0:
            p_seg = torch.arange(-M_cnt, 0, device=dev)
            cs_m, sn_m = rope_cos_sin(p_seg, cfg.head_dim, cfg.rope_theta)
            for ls in st.layers:
                ls.k_seg_rot[:, :, :M_cnt] = apply_rope(ls.k_seg_raw[:, :, :M_cnt], cs_m, sn_m)
        if S > 0:
            p_sk = torch.arange(-(M_cnt + S), -M_cnt, device=dev)
            cs_s, sn_s = rope_cos_sin(p_sk, cfg.head_dim, cfg.rope_theta)
            for ls in st.layers:
                ls.k_sink_rot[:, :, :S] = apply_rope(ls.k_sink_raw[:, :, :S], cs_s, sn_s)

    def _advance_chunk_boundary(self, st: VAULTv3State, seg_active: bool) -> None:
        """Called at c == 0, pos > 0 BEFORE running token at st.pos."""
        cfg = self.cfg
        W = cfg.window
        pos = st.pos
        dev = self.embed_tokens.weight.device

        if pos >= 2 * W and seg_active and st.seg_budget > 0:
            ev_chunk_idx = pos // W - 2
            bad_tok = st.sc_prev[0] <= -1.0e29
            if bool(bad_tok.any().item()):
                seg_ids_q = st.seg_prev_ids[0]
                n_loc_q = int(seg_ids_q.max().item()) + 1 if seg_ids_q.numel() else 0
                bad_seg = torch.zeros((max(1, n_loc_q),), device=dev, dtype=torch.bool)
                bad_seg[seg_ids_q[bad_tok]] = True
                st.sc_prev[0, bad_seg[seg_ids_q]] = -1.0e30

            # Segment-Evicted KV Memory update (batch=1)
            seg_ids_ev = st.seg_prev_ids[0]
            n_loc = int(seg_ids_ev.max().item()) + 1 if seg_ids_ev.numel() else 0
            mx = torch.full((max(1, n_loc),), float("-inf"), device=dev, dtype=torch.float32)
            mx.scatter_reduce_(0, seg_ids_ev, st.sc_prev[0], "amax")
            full_seg = seg_ids_ev + st.seg_id_offset
            g_seg = full_seg % 16384
            cand_all = mx[seg_ids_ev] - 1e-6 * g_seg.float()
            st.seg_id_offset += n_loc

            t_start = st.n_pin if ev_chunk_idx == 0 else 0
            if t_start < W:
                c_pos_all = torch.arange(ev_chunk_idx * W + t_start, (ev_chunk_idx + 1) * W, device=dev, dtype=torch.long)
                c_sc_all = cand_all[t_start:]
                c_tok_all = st.tok_prev[0, t_start:]
                c_seg_all = full_seg[t_start:]
                c_src_all = torch.arange(t_start, W, device=dev, dtype=torch.long)
                valid = c_sc_all > -1.0e29
                c_pos = c_pos_all[valid]
                c_sc = c_sc_all[valid]
                c_tok = c_tok_all[valid]
                c_seg = c_seg_all[valid]
                c_src = c_src_all[valid]
                keep_old, add_cand, st.mem_pos, st.mem_score = merge_segment_bank_indices(
                    st.mem_pos, st.mem_score, c_pos, c_sc, st.seg_budget
                )
                n_keep = int(keep_old.numel())
                n_add = int(add_cand.numel())
                parts_tok = []
                parts_seg = []
                if n_keep > 0:
                    parts_tok.append(st.mem_tok[keep_old])
                    parts_seg.append(st.mem_seg[keep_old])
                if n_add > 0:
                    parts_tok.append(c_tok[add_cand])
                    parts_seg.append(c_seg[add_cand])
                st.mem_tok = torch.cat(parts_tok, dim=0) if parts_tok else torch.empty(0, dtype=torch.long, device=dev)
                st.mem_seg = torch.cat(parts_seg, dim=0) if parts_seg else torch.empty(0, dtype=torch.long, device=dev)
                src_new = c_src[add_cand]
                for ls in st.layers:
                    if n_keep > 0:
                        ls.k_seg_raw[:, :, :n_keep] = ls.k_seg_raw[:, :, keep_old].clone()
                        ls.v_seg[:, :, :n_keep] = ls.v_seg[:, :, keep_old].clone()
                    if n_add > 0:
                        ls.k_seg_raw[:, :, n_keep : n_keep + n_add] = ls.kraw_prev[:, :, src_new]
                        ls.v_seg[:, :, n_keep : n_keep + n_add] = ls.v_prev[:, :, src_new]
                self._rebuild_shingle_index(st)

        # Shift cur -> prev and re-base k_prev by -W
        for ls in st.layers:
            ls.k_prev.copy_(ls.k_cur); ls.v_prev.copy_(ls.v_cur); ls.kraw_prev.copy_(ls.kraw_cur)
            if pos >= 2 * W:
                ls.k_prev.copy_(apply_rope(ls.k_prev, st.cos_neg_w.expand(W, -1), st.sin_neg_w.expand(W, -1)))
        st.tok_prev.copy_(st.tok_cur)
        st.sc_prev.copy_(st.sc_cur)
        st.seg_prev_ids.copy_(st.seg_cur_ids)
        st.cur_seg_id = 0
        st.seg_len = 0

        if pos >= 2 * W:
            self._refresh_prefix_rotations(st, seg_active)

    # ---- single-pass chunk-recurrent forward for Segment Memory ----
    def _forward_chunk_recurrent(
        self, input_ids: torch.Tensor, seg_budget: Optional[int] = None,
        return_state: bool = False, last_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[VAULTv3State]]:
        cfg = self.cfg
        B, L = input_ids.shape
        assert B == 1, "Segment-Evicted KV Memory requires batch_size=1"
        W, D, group = cfg.window, cfg.head_dim, cfg.group_size
        dev = input_ids.device
        dt = self.embed_tokens.weight.dtype
        bud = cfg.seg_budget if seg_budget is None else int(seg_budget)
        seg_active = (self.seg_scorer is not None) and (bud > 0)

        st = VAULTv3State(cfg, B, dev, dt, seg_budget=bud)
        limit_pin = min(cfg.max_sink, W, st.layers[0].k_sink_raw.shape[2])
        for p_i in range(min(L, limit_pin + 1)):
            if st.pin_locked:
                break
            tid_p = int(input_ids[0, p_i].item())
            if p_i == 0:
                if tid_p == cfg.im_start_id and limit_pin >= 2:
                    st.saw_im_start = True
                else:
                    st.pin_locked = True
            elif p_i == 1 and st.saw_im_start:
                if tid_p != cfg.system_tok_id:
                    st.saw_im_start = False
                    st.pin_locked = True
            elif st.saw_im_start and p_i < limit_pin:
                if st.awaiting_sys_nl:
                    st.n_pin = max(st.n_pin, p_i + 1 if tid_p == 198 else p_i)
                    st.pin_locked = True
                    st.awaiting_sys_nl = False
                elif tid_p == cfg.im_end_id:
                    st.n_pin = max(st.n_pin, p_i + 1)
                    if p_i + 1 < limit_pin:
                        st.awaiting_sys_nl = True
                    else:
                        st.pin_locked = True
            elif p_i >= limit_pin:
                if st.saw_im_start and not st.awaiting_sys_nl:
                    st.n_pin = max(st.n_pin, limit_pin)
                st.pin_locked = True
                st.awaiting_sys_nl = False

        tier_lut = self._get_tier_lut() if seg_active else None
        t2 = max(cfg.seg_min, (cfg.seg_max * 2) // 3)
        t3 = max(cfg.seg_min, cfg.seg_max - 8 if cfg.seg_max > 8 else cfg.seg_min)
        feat_set = set(cfg.seg_layers) if seg_active else set()
        feat_map = {li: idx for idx, li in enumerate(cfg.seg_layers)} if seg_active else {}
        scale = 1.0 / math.sqrt(D)
        l_lo = cfg.n_layers // 4
        l_mid = cfg.n_layers // 2
        l_hi = cfg.n_layers - cfg.n_layers // 14

        n_chunks = (L + W - 1) // W
        out_chunks = []
        last_x = None

        for ci in range(n_chunks):
            r0, r1 = ci * W, min(L, (ci + 1) * W)
            clen = r1 - r0
            st.pos = r0
            if ci > 0:
                self._advance_chunk_boundary(st, seg_active=seg_active)

            tok_c = input_ids[:, r0:r1]
            tok_c_1d = tok_c[0].tolist()
            st.tok_cur[:, :clen] = tok_c
            # Segment IDs inside this chunk (j == W - 1 closes full chunks while preserving mid-chunk state on partial prefill)
            if seg_active and tier_lut is not None:
                tiers = tier_lut[tok_c[0].cpu()].tolist()
                s_loc, slen = 0, 0
                seg_c = torch.empty(clen, dtype=torch.long)
                for j in range(clen):
                    seg_c[j] = s_loc
                    slen += 1
                    tr = tiers[j]
                    brk = (tr == 1 and slen >= cfg.seg_min) or (tr == 2 and slen >= t2) or (tr == 3 and slen >= t3)
                    if j == W - 1 or brk or slen >= cfg.seg_max:
                        s_loc += 1
                        slen = 0
                st.seg_cur_ids[0, :clen] = seg_c.to(dev)
                st.cur_seg_id = s_loc
                st.seg_len = slen
                feats = [None] * len(cfg.seg_layers)

            eff_pos = torch.arange(W if ci >= 1 else 0, (W if ci >= 1 else 0) + clen, device=dev)
            cos_q, sin_q = rope_cos_sin(eff_pos, D, cfg.rope_theta)

            x = self.embed_tokens(tok_c)
            M_cnt = int(st.mem_pos.numel()) if (seg_active and ci >= 2) else 0
            S_cnt = st.n_pin if ci >= 2 else 0
            prefix_len = (S_cnt + M_cnt + W) if ci >= 2 else (W if ci == 1 else 0)
            if prefix_len > 0:
                q_loc = torch.arange(clen, device=dev)[:, None]
                k_loc = torch.arange(prefix_len + clen, device=dev)[None, :]
                cmask = (k_loc <= (prefix_len + q_loc))[None, None]
            else:
                cmask = None

            chunk_seg_bias = (
                self._build_chunk_resonance_bias(st, r0, tok_c_1d)
                if (seg_active and ci >= 2 and M_cnt > 0)
                else None
            )

            for li, (blk, ls) in enumerate(zip(self.layers, st.layers)):
                if li in feat_set:
                    feats[feat_map[li]] = x[0].float()

                u = blk.input_layernorm(x)
                a = blk.self_attn
                q, k, v = a._qkv(u)
                qr, kr = apply_rope(q, cos_q, sin_q), apply_rope(k, cos_q, sin_q)

                ls.k_cur[:, :, :clen] = kr.to(ls.k_cur.dtype)
                ls.v_cur[:, :, :clen] = v.to(ls.v_cur.dtype)
                ls.kraw_cur[:, :, :clen] = k.to(ls.kraw_cur.dtype)

                S_cap = ls.k_sink_raw.shape[2]
                if r0 < S_cap:
                    take = min(clen, S_cap - r0)
                    ls.k_sink_raw[:, :, r0 : r0 + take] = k[:, :, :take].to(ls.k_sink_raw.dtype)
                    ls.v_sink[:, :, r0 : r0 + take] = v[:, :, :take].to(ls.v_sink.dtype)

                if ci == 0:
                    attn = F.scaled_dot_product_attention(qr, _expand_kv(kr, group), _expand_kv(v, group), is_causal=True)
                elif ci == 1:
                    kk = torch.cat([ls.k_prev, kr.to(ls.k_prev.dtype)], dim=2).to(qr.dtype)
                    vv = torch.cat([ls.v_prev, v.to(ls.v_prev.dtype)], dim=2).to(qr.dtype)
                    attn = F.scaled_dot_product_attention(qr, _expand_kv(kk, group), _expand_kv(vv, group), attn_mask=cmask)
                else:
                    parts_k = []
                    parts_v = []
                    if S_cnt > 0:
                        parts_k.append(ls.k_sink_rot[:, :, :S_cnt])
                        parts_v.append(ls.v_sink[:, :, :S_cnt])
                    if M_cnt > 0:
                        parts_k.append(ls.k_seg_rot[:, :, :M_cnt])
                        parts_v.append(ls.v_seg[:, :, :M_cnt])
                    parts_k.extend([ls.k_prev, kr.to(ls.k_prev.dtype)])
                    parts_v.extend([ls.v_prev, v.to(ls.v_prev.dtype)])
                    kk = torch.cat(parts_k, dim=2).to(qr.dtype)
                    vv = torch.cat(parts_v, dim=2).to(qr.dtype)
                    kk_e = _expand_kv(kk, group)
                    vv_e = _expand_kv(vv, group)
                    if chunk_seg_bias is not None and l_lo <= li < l_hi:
                        cb = chunk_seg_bias[0] if li < l_mid else chunk_seg_bias[1]
                        scores = torch.matmul(qr, kk_e.transpose(-1, -2)).float() * scale
                        scores[:, :, :, S_cnt : S_cnt + M_cnt] = scores[:, :, :, S_cnt : S_cnt + M_cnt] + cb
                        scores = scores.masked_fill(~cmask, float("-inf"))
                        probs = F.softmax(scores, dim=-1).to(qr.dtype)
                        attn = torch.matmul(probs, vv_e)
                    else:
                        attn = F.scaled_dot_product_attention(qr, kk_e, vv_e, attn_mask=cmask)

                x = x + a.o_proj(attn.transpose(1, 2).reshape(B, clen, cfg.n_heads * D))
                x = x + blk.mlp(blk.post_attention_layernorm(x))

            if seg_active:
                st.sc_cur[0, :clen] = self._score_chunk_habituated(st, torch.cat(feats, dim=-1))

            self._apply_spoof_chunk(st, r0, tok_c_1d, seg_active)

            if last_only:
                last_x = x[:, -1:]
            else:
                out_chunks.append(x)

        # Populate st resonance fields & st.pos for the final token so st matches step-by-step prefill identically
        if return_state and seg_active and L > 2 * W:
            st.pos = L - 1
            self._update_step_resonance(st, clen - 1)
        st.pos = L
        if last_only:
            return self.norm(last_x), (st if return_state else None)
        h_all = self.norm(torch.cat(out_chunks, dim=1))
        return h_all, (st if return_state else None)

    # ---- full-sequence forward ----
    def forward_hidden(
        self, input_ids: torch.Tensor, mode: str = "block", vault_on: bool = False, pos0: int = 0,
        seg_budget: Optional[int] = None,
    ) -> torch.Tensor:
        _ = vault_on
        B, L = input_ids.shape
        bud = self.cfg.seg_budget if seg_budget is None else int(seg_budget)
        not_train = not (self.training and torch.is_grad_enabled())
        use_chunk_rec = (
            (mode == "block")
            and (B == 1)
            and (pos0 == 0)
            and not_train
            and (((self.seg_scorer is not None) and (bud > 0)) or (detect_pinned_prefix_len(input_ids[0], self.cfg) > self.cfg.n_sink))
        )
        if use_chunk_rec:
            h, _ = self._forward_chunk_recurrent(input_ids, seg_budget=bud, return_state=False)
            return h

        n_pin = detect_pinned_prefix_len(input_ids[0], self.cfg) if B == 1 else self.cfg.n_sink
        x = self.embed_tokens(input_ids)
        pos = torch.arange(pos0, pos0 + L, device=input_ids.device)
        cos, sin = rope_cos_sin(pos, self.cfg.head_dim, self.cfg.rope_theta)
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, mode, None, n_pin, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mode=mode, n_pin=n_pin)
        return self.norm(x)

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.embed_tokens.weight)

    def forward(
        self, input_ids: torch.Tensor, mode: str = "block", vault_on: bool = False, seg_budget: Optional[int] = None,
    ) -> torch.Tensor:
        return self.lm_logits(self.forward_hidden(input_ids, mode=mode, vault_on=vault_on, seg_budget=seg_budget))

    # ---- streaming ----
    def new_state(self, batch: int = 1, seg_budget: Optional[int] = None) -> VAULTv3State:
        p = self.embed_tokens.weight
        return VAULTv3State(self.cfg, batch, p.device, p.dtype, seg_budget=seg_budget)

    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor, vault_on: bool = False, seg_budget: Optional[int] = None,
    ) -> Tuple[torch.Tensor, VAULTv3State]:
        """Parallel prefill from position 0. Returns last-token logits (B, V) and exact streaming state."""
        _ = vault_on
        B, L = input_ids.shape
        bud = self.cfg.seg_budget if seg_budget is None else int(seg_budget)
        if B == 1:
            h, st = self._forward_chunk_recurrent(
                input_ids, seg_budget=bud, return_state=True, last_only=True,
            )
            return self.lm_logits(h)[:, 0], st

        # Batch > 1 sliding-window + sink prefill
        cfg = self.cfg
        W = cfg.window
        st = self.new_state(B, seg_budget=0)
        st.set_pinned(cfg.n_sink)
        x = self.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(torch.arange(L, device=input_ids.device), cfg.head_dim, cfg.rope_theta)
        last = (L - 1) // W
        cnt = L - last * W
        st.tok_cur[:, :cnt] = input_ids[:, last * W : L]
        if last >= 1:
            st.tok_prev.copy_(input_ids[:, (last - 1) * W : last * W])
        eff_cur = torch.arange(W if last >= 1 else 0, (W if last >= 1 else 0) + cnt, device=input_ids.device)
        cos_c, sin_c = rope_cos_sin(eff_cur, cfg.head_dim, cfg.rope_theta)
        cos_p, sin_p = rope_cos_sin(torch.arange(W, device=input_ids.device), cfg.head_dim, cfg.rope_theta)
        for layer, ls in zip(self.layers, st.layers):
            col: Dict[str, torch.Tensor] = {}
            x = layer(x, cos, sin, mode="block", collect=col, n_pin=st.n_pin)
            ns = min(st.n_pin, L)
            if ns > 0:
                ls.k_sink_raw[:, :, :ns] = col["k"][:, :, :ns].to(ls.k_sink_raw.dtype)
                ls.v_sink[:, :, :ns] = col["v"][:, :, :ns].to(ls.v_sink.dtype)
            cur = slice(last * W, L)
            ls.k_cur[:, :, :cnt] = apply_rope(col["k"][:, :, cur], cos_c, sin_c).to(ls.k_cur.dtype)
            ls.v_cur[:, :, :cnt] = col["v"][:, :, cur].to(ls.v_cur.dtype)
            ls.kraw_cur[:, :, :cnt] = col["k"][:, :, cur].to(ls.kraw_cur.dtype)
            if last >= 1:
                prv = slice((last - 1) * W, last * W)
                ls.k_prev.copy_(apply_rope(col["k"][:, :, prv], cos_p, sin_p).to(ls.k_prev.dtype))
                ls.v_prev.copy_(col["v"][:, :, prv])
                ls.kraw_prev.copy_(col["k"][:, :, prv].to(ls.kraw_prev.dtype))
        st.pos = L
        if L >= 2 * W:
            self._refresh_prefix_rotations(st, seg_active=False)
        h = self.norm(x[:, -1:])
        return self.lm_logits(h)[:, 0], st

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor, st: VAULTv3State, vault_on: bool = False) -> torch.Tensor:
        """token_ids: (B,) -> logits (B, V). Fixed-size state; no growth with context."""
        _ = vault_on
        cfg = self.cfg
        W = cfg.window
        pos = st.pos
        c = pos % W
        seg_active = (self.seg_scorer is not None) and (st.seg_budget > 0) and (token_ids.shape[0] == 1)

        # Online ChatML system-prompt auto-pinning (matches C v3_step token-by-token!)
        if not st.pin_locked and token_ids.shape[0] == 1:
            tid = int(token_ids[0].item())
            limit = min(cfg.max_sink, W, st.layers[0].k_sink_raw.shape[2])
            if pos == 0:
                if tid == cfg.im_start_id and limit >= 2:
                    st.saw_im_start = True
                else:
                    st.pin_locked = True
            elif pos == 1 and st.saw_im_start:
                if tid != cfg.system_tok_id:
                    st.saw_im_start = False
                    st.pin_locked = True
            elif st.saw_im_start and pos < limit:
                if st.awaiting_sys_nl:
                    st.n_pin = max(st.n_pin, pos + 1 if tid == 198 else pos)
                    st.pin_locked = True
                    st.awaiting_sys_nl = False
                elif tid == cfg.im_end_id:
                    st.n_pin = max(st.n_pin, pos + 1)
                    if pos + 1 < limit:
                        st.awaiting_sys_nl = True
                    else:
                        st.pin_locked = True
            elif pos >= limit:
                if st.saw_im_start and not st.awaiting_sys_nl:
                    st.n_pin = max(st.n_pin, limit)
                st.pin_locked = True
                st.awaiting_sys_nl = False

        if c == 0 and pos > 0:
            self._advance_chunk_boundary(st, seg_active=seg_active)

        st.tok_cur[:, c] = token_ids
        if seg_active:
            tid = int(token_ids[0].item())
            tr = int(self._get_tier_lut()[tid].item())
            t2 = max(cfg.seg_min, (cfg.seg_max * 2) // 3)
            t3 = max(cfg.seg_min, cfg.seg_max - 8 if cfg.seg_max > 8 else cfg.seg_min)
            st.seg_cur_ids[0, c] = st.cur_seg_id
            st.seg_len += 1
            brk = (tr == 1 and st.seg_len >= cfg.seg_min) or (tr == 2 and st.seg_len >= t2) or (tr == 3 and st.seg_len >= t3)
            if c == W - 1 or brk or st.seg_len >= cfg.seg_max:
                st.cur_seg_id += 1
                st.seg_len = 0
            feats = [None] * len(cfg.seg_layers)
            feat_map = {li: idx for idx, li in enumerate(cfg.seg_layers)}
            if pos >= 2 * W and int(st.mem_pos.numel()) > 0:
                self._update_step_resonance(st, c)
            else:
                st.seg_bias_mode = 0
                st.seg_bias_active = False
                st.seg_head_bias = None
                st.seg_head_bias_hop1 = None
        else:
            feats, feat_map = [], {}
            st.seg_bias_mode = 0
            st.seg_bias_active = False
            st.seg_head_bias = None
            st.seg_head_bias_hop1 = None

        x = self.embed_tokens(token_ids)[:, None]
        for li, (layer, ls) in enumerate(zip(self.layers, st.layers)):
            if seg_active and li in feat_map:
                feats[feat_map[li]] = x[0, 0].float()
            x = x + layer.self_attn.step(
                layer.input_layernorm(x), ls, st, seg_active=seg_active, layer_idx=li
            )
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        if seg_active:
            f_vec = torch.cat(feats, dim=-1)
            raw_sc = self.seg_scorer(f_vec[None, :])[0]
            sc = raw_sc
            if not st.ema_ready:
                st.seg_feat_ema.copy_(f_vec)
                st.ema_ready = True
            else:
                hab_sc = self.seg_scorer((f_vec - 0.65 * st.seg_feat_ema)[None, :])[0]
                st.seg_feat_ema.mul_(0.985).add_(f_vec, alpha=0.015)
                sc = raw_sc + 0.35 * F.relu(hab_sc - raw_sc)
            st.sc_cur[0, c] = sc

            if token_ids.shape[0] == 1 and self._check_spoof_step(st, pos, c):
                st.sc_cur[0, c] = -1.0e30

        st.pos = pos + 1
        return self.lm_logits(self.norm(x))[:, 0]


# ==============================================================================
# Qwen2 weight loading / checkpoint IO
# ==============================================================================

def load_qwen2_into_vault(model: VAULTv3ForCausalLM, qwen_dir: str) -> None:
    """Copy Qwen2 safetensors weights into the VAULT v3 backbone (scorer params keep their init)."""
    import glob
    from safetensors.torch import load_file

    sd = {}
    for f in sorted(glob.glob(os.path.join(qwen_dir, "*.safetensors"))):
        sd.update(load_file(f))
    mapped = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue  # tied
        nk = k[len("model."):] if k.startswith("model.") else k
        mapped[nk] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if not m.startswith("seg_scorer.")]
    if missing or unexpected:
        raise RuntimeError(f"Qwen2 load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")


def save_vault_v3(model: VAULTv3ForCausalLM, path: str, extra: Optional[dict] = None) -> None:
    payload = {"format": "vault_v3", "config": model.cfg.to_dict(), "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        payload.update(extra)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_vault_params(model: VAULTv3ForCausalLM, path: str, extra: Optional[dict] = None) -> None:
    """Small checkpoint with only the trainable VAULT v3 SegmentScorer tensors."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("seg_scorer.")}
    payload = {"format": "vault_v3_scorer_only", "config": model.cfg.to_dict(), "vault_state": sd}
    if extra:
        payload.update(extra)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_vault_params(model: VAULTv3ForCausalLM, path: str) -> dict:
    """Load trainable VAULT v3 auxiliary parameters (e.g. SegmentScorer) from a checkpoint into model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = ck.get("vault_state", ck.get("model_state", ck.get("state", ck)))
    remap = {
        "net.0.weight": "seg_scorer.ln.weight", "net.0.bias": "seg_scorer.ln.bias",
        "net.1.weight": "seg_scorer.fc1.weight", "net.1.bias": "seg_scorer.fc1.bias",
        "net.3.weight": "seg_scorer.fc2.weight", "net.3.bias": "seg_scorer.fc2.bias",
        "ln.weight": "seg_scorer.ln.weight", "ln.bias": "seg_scorer.ln.bias",
        "fc1.weight": "seg_scorer.fc1.weight", "fc1.bias": "seg_scorer.fc1.bias",
        "fc2.weight": "seg_scorer.fc2.weight", "fc2.bias": "seg_scorer.fc2.bias",
    }
    state = {remap.get(k, k): v for k, v in raw.items() if ".vault." not in k and k != "vault_route"}
    _, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    return {k: v for k, v in ck.items() if k not in ("vault_state", "model_state", "state", "optimizer")}
