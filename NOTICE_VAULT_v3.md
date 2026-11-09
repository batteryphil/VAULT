# VAULT v3 — Provenance, Licensing & Operational Scope Notice

## 1. VAULT v3 Architecture, Runtime & Kenyon Vault Memory
- **Copyright:** Copyright (c) 2026 Philip Bartsch (`batteryphil`)
- **License:** Dual Licensed under `AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial` (see `LICENSE-AGPL-3.0` and `LICENSE-COMMERCIAL.md`).
- **Covered Components:**
  - `src/vault_v3_model.py` — VAULT v3 architecture (Block Sliding-Window Attention with bounded RoPE re-basing + Attention Sinks + 5-tensor-per-layer Kenyon Vault associative memory).
  - `train_vault_v3_stage_a.py` — Stage A Kenyon Vault long-context memory & past-window retrieval trainer.
  - `tools/vlt3.py`, `tools/export_vlt3.py` — `VLT3` flat binary format specification, `Q8_0`/`Q4_0`/`BF16` quantizer, and `VTOK` binary BPE serializer.
  - `runtime/vault3.h`, `runtime/vault3.c`, `runtime/vault3_host.c`, `runtime/Makefile` — Zero-dependency freestanding C99 inference runtime and hosted POSIX CLI/benchmark harness.

## 2. Backbone Weights & BPE Vocabulary Attribution (`Apache-2.0`)
VAULT v3 (`1.54B` parameters) initializes its frozen `28`-layer Grouped-Query Attention + SwiGLU Transformer backbone (`1,543,714,304` parameters) and `151,665`-entry byte-level BPE ChatML vocabulary from **`Qwen/Qwen2.5-1.5B-Instruct`**:
- **Upstream Source:** Qwen Team, Alibaba Cloud (`https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`)
- **Upstream License:** Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`)
- **Modifications Made in VAULT v3:**
  1. Replaced unbounded $O(L)$ causal KV-cache attention with **Block Sliding-Window Attention** (`W = 1024`, `n_sink = 4` StreamingLLM attention sinks) with chunk-boundary RoPE re-basing (`[0, 2W)`), bounding KV-ring state strictly regardless of stream length.
  2. Added **Kenyon Vault** co-prime LSH associative memory (`P = (13, 17, 19)`, `49` cells/KV-head/layer, `1,921,416` trained parameters across `5` regular float32 tensors per layer plus `1` shared ternary routing matrix `vault_route`), trained to absorb evicted KV chunks via the Delta Rule and restore past-window context.
  3. Repackaged weights and byte-level BPE merge/pre-tokenization tables into the flat, `64`-byte aligned, CRC32-verified **`VLT3`** binary container (`BF16`, `Q8_0`, and `Q4_0` variants) with zero JSON metadata.

## 3. Intended Role in Operating Organism (`OO`)
VAULT v3 is strictly a **stateless-interface communication and natural-language processing component**. It does not invoke system calls, execute shell commands, allocate dynamic heap memory, or make autonomous operational decisions. Within Operating Organism (`OO`), **`OPI`** remains the sole operational decision-maker governing all system actions.
