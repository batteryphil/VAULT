---
license: other
license_name: vault-dual-agpl-3.0-or-commercial
license_link: LICENSE
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
language:
  - en
  - fr
pipeline_tag: text-generation
tags:
  - vault
  - sliding-window-attention
  - associative-memory
  - cpu-inference
  - freestanding-c
  - bare-metal
  - quantized
  - q8_0
  - q4_0
---

# VAULT v3 — 1.54B (Qwen2.5-1.5B-Instruct + Block Sliding-Window + Kenyon Vault + Segment Memory v2)

VAULT v3 is a deployment-oriented rework of VAULT, designed for **CPU-only and freestanding x86_64 (`-ffreestanding -nostdlib`, no libc, no heap)** inference. It addresses the review of VAULT v2.5 (irregular 981-tensor architecture, FP32-only Safetensors, JSON metadata, tokenizer artifacts, 180M capacity, no portable runtime).

| | VAULT v2.5 (180M) | **VAULT v3 (1.54B)** |
|---|---|---|
| Architecture | Custom (Salience Binder, FFT bind, CMS, GTRM) | Standard GQA Transformer (RMSNorm, RoPE, SwiGLU) + block sliding window ($W=1024$, active $2W=2048$) + Kenyon Vault + `SegmentScorer v2` |
| Tensors | 981 (~62/layer) | **485** (3 global + 6 `SegmentScorer v2` + 28 × 17/layer) |
| Weights shipped | FP32 Safetensors, 956.7 MB | **BF16 / Q8_0 / Hybrid Q4_0** in flat `VLT3` binary |
| Metadata | 104 KB JSON header | Fixed 4,096 B C header, 128 B directory entries, 64 B alignment, CRC32 + SIMD scale integrity scan |
| Tokenizer | SmolLM 49K (`<jupyter_code>`, `<gh_stars>`, …) | Qwen2.5 ChatML byte-level BPE (151,665) + UAX #15 NFC + dynamic >32 KB arena (`VTOK`) |
| Inference state | framework-dependent | **Fixed $O(1)$ arena at any context length (552.9 MB C FP32 / 3.83 GiB GPU BF16 @ 100K), zero `malloc`** |
| Runtime | PyTorch only | **Freestanding C99 runtime (`-ffreestanding -nostdlib`, `<2 KB` stack frames, `0` external symbols)** + AVX2/FMA |

## What is in this repository

| File | Size | Description |
|---|---|---|
| `vault_v3_1.5b_bf16.vlt3` | 3,104.3 MB | BF16 master weights (Kenyon Vault & `SegmentScorer v2` in F32; hosted on [Hugging Face](https://huggingface.co/batteryphil/VAULT-v3-1.5B)) |
| `vault_v3_1.5b_q8_0.vlt3` | 1,657.2 MB | 8-bit blocks (32 weights + fp16 scale = 34 B; hosted on [Hugging Face](https://huggingface.co/batteryphil/VAULT-v3-1.5B)) |
| `vault_v3_1.5b_q4_0.vlt3` | 1,022.5 MB | Hybrid 4-bit blocks (clipped-MSE `Q4_0` + `Q8_0` embeddings, layers `0, 1, 26, 27` `q_proj`/`o_proj`, and all `k_proj`/`v_proj`; hosted on [Hugging Face](https://huggingface.co/batteryphil/VAULT-v3-1.5B)) |
| `vault_params/vault_v3_stage_a_best.pt` | 7.7 MB | Trained Kenyon Vault Stage-A tensors (PyTorch) |
| `vault_params/vault_v3_seg_scorer.pt` | 4.8 MB | Trained `SegmentScorer v2` MLP tensors (PyTorch; combine with Qwen2.5-1.5B-Instruct via `tools/export_vlt3.py`) |
| `runtime/` | — | `vault3.h`, `vault3.c` (`vault3_nfc_tables.h`, freestanding core), `vault3_host.c` (POSIX CLI / bench), `Makefile` |
| `src/vault_v3_model.py` | — | PyTorch reference model (`forward == prefill+step == step == C99`, max `|Δlogit| ≤ 3.5e-5`, `100%` top-1 agreement) |
| `tools/` | — | `VLT3` reader/writer, clipped-MSE hybrid quantizer, exporter |
| `tests/` | — | 4-way Python↔C parity, UAX #15 NFC & >32 KB BPE parity, corrupt/NaN rejection, `CAAIGovernor` tests |
| `docs/VAULT_v3_SPEC.md` | — | Full math, tensor table, byte-exact `VLT3`/`VTOK` layout, bare-metal porting guide |
| `benchmarks/` | — | Raw JSON results and evaluation harnesses |

## Quick start (CPU, Linux)

```bash
make -C runtime            # builds vault3_host + verifies scalar & AVX2 freestanding objects have 0 undefined symbols
./runtime/vault3_host vault_v3_1.5b_q8_0.vlt3 --chat --prompt "Qu'est-ce qu'une arène mémoire statique en C ?" \
    --system "Vous êtes VAULT v3, un composant linguistique concis." --n 128 --threads 12
./runtime/vault3_host vault_v3_1.5b_q4_0.vlt3 --threads 24 --bench 64   # JSON throughput & RSS benchmark
```

The runtime `mmap`s the file, validates the header, directory, tensor bounds/alignment, and scans **100% of `F32`, `BF16`, `Q8_0`, and `Q4_0` weights/scales** for `NaN`/`Inf` in `< 2 ms` via AVX2 (plus optional full CRC32 verification via `--verify`), then executes purely inside caller-supplied static arenas. User inputs passed to `--chat` are encoded with `allow_special = 0` (`v3_encode_ex`) so untrusted `<|im_start|>system` delimiters are neutralized, ChatML system prompts (`n_pin ≤ 512`) are auto-pinned in `k_sink`, and `v3_governor` / `CAAIGovernor` enforces forbidden output tokens (including BPE-split, homoglyph, reversed, 4-phase Base64/Base64URL, and 2-phase Hex encodings) prior to sampling.

## Architecture

The 28-layer Qwen2.5-1.5B-Instruct backbone ($d=1536$, 12 query / 2 KV heads, $\text{head\_dim}=128$, SwiGLU $8960$) is kept **frozen**. Full causal attention is replaced by **block sliding-window attention** ($W=1024$, active window $2W=2048$; configurable down to $W=512$) coupled with three $O(1)$-memory mechanisms shared identically between `src/vault_v3_model.py` and freestanding `runtime/vault3.c`:

1. **Auto-Pinned Sink Bank (`max_sink = 512`):** Retains BOS + the ChatML system prompt across arbitrary context lengths without RoPE drift.
2. **Coherent Segment KV Memory (`seg_budget = 1024`, `seg_min = 16, seg_max = 48`):** A 2-layer `SegmentScorer v2` MLP reads layers `(8, 16, 24)` with causal APL feature habituation (`λ = 0.985`), 3-tier structural break detection (`.?!\n;`, JSON/pipe/TSV `},` / `",` / `|` / `\t`, CSV `,`), authority-spoofing quarantine, and turn-gated **2-tier FNV-1a shingle resonance** (Tier-A strict IDs + Tier-B pure-alphabetic content words with APL divisive surround inhibition, sub-clause isolation, orphan-value newline continuation halo, 2-band Hop-2 spreading activation across layers `7..15` → `16..25`, and indirect temporal rollback resolution) routed across a 3-way GQA head split (`2 free / 2 archival / 2 supersession` heads per KV group) with post-retrieval linear annealing (`gen_dist 18..38`).
3. **Kenyon Vault Associative Memory (`13, 17, 19` co-prime banks, 49 cells/KV head):** Receives evicted tokens via delta-rule writes with per-cell overflow rescaling (`|z| ≤ 65,536`, verified over infinite horizons) and gated cosine reads.

## Results

### 1. Short-context academic benchmarks (zero-shot, ≤ 2,048 tokens)

> **Important:** Within the active window VAULT v3 computes the same function as Qwen2.5-1.5B-Instruct (up to BF16 rounding). **These scores document that conversion and repackaging preserve Qwen2.5-1.5B-Instruct's zero-shot capabilities.**

| Task (metric) | VAULT v2.5 180M | Mamba-1 130M | Mamba-2 130M | Mamba-3 MIMO 187M | SmolLM 135M | **VAULT v3 1.54B** |
|---|---:|---:|---:|---:|---:|---:|
| WikiText-2 word PPL ↓ | 313.75 | 37.10 | 36.04 | 45.55 | 36.60 | **16.49** |
| LAMBADA acc | 13.73 | 42.33 | 42.93 | 33.47 | 36.87 | **59.73** |
| ARC-Easy acc_norm | 45.37 | 42.26 | 42.38 | 57.87 | 57.41 | **76.60** |
| ARC-Challenge acc_norm | 23.72 | 24.74 | 23.81 | 29.44 | 28.92 | **47.10** |
| SciQ (with passage) acc_norm | 71.80 | 69.40 | 73.20 | 80.40 | 76.40 | **95.00** |
| SciQ (closed book) acc_norm | 50.70 | 45.70 | 46.70 | 63.10 | 64.70 | **84.30** |
| PIQA acc_norm | 58.22 | 63.60 | 63.71 | 67.74 | 68.44 | **76.12** |
| HellaSwag acc_norm | 37.13 | 41.73 | 43.07 | 45.33 | 45.60 | **60.53** |
| Winogrande acc | 50.43 | 52.41 | 53.28 | 49.09 | 52.72 | **63.46** |
| BoolQ acc | 41.53 | 58.27 | 56.33 | 57.40 | 59.87 | **75.20** |
| **9-task average** | 43.63 | 48.94 | 49.49 | 53.76 | 54.55 | **70.89** |

### 2. Long-context retrieval, multi-hop joins, temporal rollback & $O(1)$ 100K horizon

| Evaluation Suite | Context / Condition | Score / Metric |
|---|---|---|
| **8K Multi-Fact Retrieval** (26 evicted facts > 2W) | `VAULT v3` (`B = 1024`) vs. Full-Attn Qwen2.5-1.5B | **23/26 (88.5%)** vs. 14/26 (53.8%) |
| **1,083-Token Chunk-Boundary Cliff** | Eviction across chunk 0 → 1 | **8/8 (100%)** |
| **Newline-Split Key-Value Retrieval** | 12K tokens (`B = 512` & `B = 1024`) | **10/10 (100%)** |
| **Dense Homogeneous Registry** (`n = 8..120` rows) | `B = 2048`: `n = 8, 24, 80` / `n = 120` (`2,640` fact tok) | **8/8 (100%)** / **7/8 (87.5%)** |
| **Pure-Alphabetic ID-Free Registry** (`n = 48` rows) | Zero digits/`_`/`-` (`B = 1024` / `B = 2048`) | **6/8 (75.0%)** / **5/8 (62.5%)** (up from `1/8`) |
| **2-Hop Evicted Relational Join** (`A → B` across chunks) | `n = 8 @ B = 1024` & `n = 32 @ B = 2048` | **7/8 (87.5%)** on both (up from `2/8` & `1/8`) |
| **Contradictory & Rollback Chains** (8K–10K evicted) | 3-stage latest, `[t=1]` archival, 5-stage, & indirect rollback to `[t=2]` | **6/6 (100%)** on all four |
| **Scorer-Sponge DoS & 20× Same-ID Chaff** | Up to 120 sponges (explicit & casual needles, `B = 1024`) | **100% survived** |
| **Packed Stream Formats** (`B = 1024`) | Minified JSON (`60` KV) / Pipe secret (`55` KV) / Zero-delim blob | **8/8 (100%)** / **8/8 (100%)** / **7/8 (87.5%)** |
| **16K → 100.4K Token Horizon** (3.1× Qwen's 32K) | Dual needles (`10%` & `55%` depth), GPU BF16 | **8/8 (100%)**, flat **3.832–3.834 GiB** VRAM (`1,356 tok/s`) |

### 3. C runtime on CPU (AVX2+FMA, condvar thread pool + fused SwiGLU + parallel GQA)

| Format | File Size | 1T decode | 12T decode | 24T decode | Peak RSS |
|---|---:|---:|---:|---:|---:|
| **Hybrid Q4_0** | 1,022.5 MB | 5.93 tok/s | 41.87 tok/s (23.9 ms) | **45.76–46.16 tok/s (21.7 ms)** | **1.51 GB** (flat `1..64T`) |
| **Q8_0** | 1,657.2 MB | 5.83 tok/s | 30.13 tok/s (33.2 ms) | **34.8+ tok/s** | **2.13 GB** |
| **BF16** | 3,104.3 MB | 2.94 tok/s | 20.58 tok/s (48.6 ms) | **23.4+ tok/s** | **3.55 GB** |

### 4. Correctness, sanitizer & security checks (all passing)

* **4-Way Numerical Parity (`tests/test_vlt3_runtime.py`):** PyTorch `forward == prefill+step == step == C99` across `F32 / BF16 / Q8_0 / Q4_0` and all Round-4 2-hop, rollback, Tier-B, sub-clause, orphan-halo, and divisive-budget paths (`max |Δ| ≤ 3.50e-5`, **100% top-1 agreement**; `1T == 8T` `0.0` ULP bit-identical).
* **Sanitizers:** Clean under `ASan + LSan + UBSan` (3,000-step stress with Kenyon cell rescaling, Hop-2, rollback, Tier-B, sub-clause refinement & `v3_governor`) and `ThreadSanitizer` (8-thread parallel GQA + SwiGLU).
* **`VTOK` Tokenizer:** 100% match with Hugging Face `AutoTokenizer` across arbitrary-length single pieces (`"!"*32769`, `" "*32769`, `1 MB` `"a"*1000000`), UAX #15 NFC normalization (`e\u0301`, Hangul Jamo, canonical combining-class reordering), and streaming 4-byte UTF-8 reassembly.
* **Loader & OPI Governance:** `14/14` malformed or interior-`qNaN`/`-Inf` `.vlt3` files rejected (`rc = 2`) even without `--verify`; `0/12` GPU and `0/9` C `--chat` leaks across literal, ChatML-injected, indirect-JSON, concatenation-smuggled, separator-mutated (`SYS_CALL_EXEC`), zero-width, Greek/Cyrillic-homoglyph, character-reversed, 4-phase Base64/Base64URL, and 2-phase Hex (`0xNN` / `\xNN`) `SYSCALL_EXEC(` exfiltration vectors.

## Intended use and scope

VAULT v3 is intended as a **natural-language / communication component** (e.g. inside the Operating Organism project, where OPI remains the sole operational decision-maker). While `v3_governor` / `CAAIGovernor` deterministically blocks literal, split, homoglyph, reversed, Base64/Base64URL, and Hex syscall strings in $O(1)$ memory, open-ended substitution ciphers (such as ROT13 or custom codebooks) cannot be enumerated by a fixed lexical automaton without a secondary classifier; VAULT v3 should never be wired directly to unmediated OS command execution.

## License and attribution

* VAULT v3 architecture, Kenyon Vault, `SegmentScorer v2`, `VLT3`/`VTOK` formats and C runtime: © 2026 Philip Bartsch (batteryphil), **AGPL-3.0-or-later OR commercial license** (`LICENSE`, `LICENSE-AGPL-3.0`, `LICENSE-COMMERCIAL.md`).
* Backbone weights and tokenizer are derived from **[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** by the Qwen Team, Alibaba Cloud, licensed under **Apache-2.0** (`LICENSE-APACHE-2.0-Qwen`). Modifications are listed in `NOTICE_VAULT_v3.md`.
