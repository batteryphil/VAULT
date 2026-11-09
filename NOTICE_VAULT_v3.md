# VAULT v3 — Provenance, Licensing & Operational Scope Notice

## 1. VAULT v3 Architecture, Segment Memory & Freestanding Runtime
- **Copyright:** Copyright (c) 2026 Philip Bartsch (`batteryphil`)
- **License:** Dual Licensed under `AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial` (see `LICENSE-AGPL-3.0` and `LICENSE-COMMERCIAL.md`).
- **Covered Components:**
  - `src/vault_v3_model.py` — VAULT v3 architecture (Block Sliding-Window Attention with bounded RoPE re-basing + Auto-Pinned System Sinks + `SegmentScorer v2` Coherent Segment KV Memory + 2-tier Shingle Resonance / Hop-2 / Rollback + `CAAIGovernor`).
  - `src/kenyon_cells.py`, `runtime/kenyon_cells_archive.h`, `train_vault_v3_stage_a.py` — Archived Co-Prime Kenyon Cell associative memory implementations (PyTorch + freestanding C99) preserved in the repository for future repurposing.
  - `tools/vlt3.py`, `tools/export_vlt3.py` — `VLT3` flat binary format specification, `Q8_0` / Clipped-MSE Hybrid `Q4_0` / `BF16` quantizer, and `VTOK` binary BPE serializer.
  - `runtime/vault3.h`, `runtime/vault3.c`, `runtime/vault3_nfc_tables.h`, `runtime/vault3_host.c`, `runtime/Makefile` — Zero-dependency freestanding C99 inference runtime and hosted POSIX CLI/benchmark harness.

## 2. Backbone Weights & BPE Vocabulary Attribution (`Apache-2.0`)
VAULT v3 (`1.54B` parameters) initializes its frozen `28`-layer Grouped-Query Attention + SwiGLU Transformer backbone (`1,543,714,304` parameters) and `151,665`-entry byte-level BPE ChatML vocabulary from **`Qwen/Qwen2.5-1.5B-Instruct`**:
- **Upstream Source:** Qwen Team, Alibaba Cloud (`https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`)
- **Upstream License:** Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`)
- **Modifications Made in VAULT v3:**
  1. Replaced unbounded $O(L)$ causal KV-cache attention with **Block Sliding-Window Attention** (`W = 1024`, active ring `2W = 2048`) with chunk-boundary RoPE re-basing and dynamic ChatML system-prompt auto-pinning (`max_sink = 512`), bounding active KV state strictly regardless of stream length.
  2. Added **Coherent Segment KV Memory (`SegmentScorer v2`)** (`1,190,657` trained parameters across `6` global float32 tensors; `344` total tensors in `VLT3`) with causal APL feature habituation, 3-tier structural segmentation, authority-spoofing quarantine, turn-gated 2-tier FNV-1a shingle resonance, 2-hop spreading activation, and indirect temporal rollback resolution.
  3. Repackaged weights and byte-level BPE merge/pre-tokenization tables into the flat, `64`-byte aligned, CRC32-verified **`VLT3`** binary container (`BF16`, `Q8_0`, and Clipped-MSE Hybrid `Q4_0` variants) with zero JSON metadata and deterministic `v3_governor` / `CAAIGovernor` output enforcement.

## 3. Intended Role in Operating Organism (`OO`)
VAULT v3 is strictly a **stateless-interface communication and natural-language processing component**. It does not invoke system calls, execute shell commands, allocate dynamic heap memory, or make autonomous operational decisions. Within Operating Organism (`OO`), **`OPI`** remains the sole operational decision-maker governing all system actions.
