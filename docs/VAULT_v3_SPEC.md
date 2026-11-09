# VAULT v3 — Architecture, `VLT3` Binary Format & Freestanding `x86_64` Porting Specification

- **Model Family:** `VAULT-v3-1.5B` (`1,546,826,389` parameters; `485` tensors total: `3` global + `6` `SegmentScorer v2` + `28 × 17` per layer)
- **Target Environments:** Local CPU-only hosts, GPU PyTorch hosts, and freestanding `x86_64` bare-metal kernels (`-std=c99 -ffreestanding -nostdlib`, zero `libc`, zero heap allocation, `< 2 KB` stack frames, deterministic static memory arenas)
- **Operational Role (Operating Organism / `OO`):** Stateless-interface natural-language and communication component. **`OPI` remains the sole operational decision-maker**; `VAULT v3` never invokes system calls or governs autonomous OS actions, and `v3_governor` / `CAAIGovernor` enforces forbidden output patterns deterministically prior to token sampling.
- **License:** Dual-licensed `AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial` (`Qwen2.5-1.5B-Instruct` backbone & ChatML BPE vocabulary under `Apache-2.0`; see `NOTICE_VAULT_v3.md`).

---

## 1. Design Summary: `VAULT v2.5` vs `VAULT v3`

| Dimension | `VAULT v2.5 (180M / 1B)` | `VAULT v3 (1.54B)` |
| :--- | :--- | :--- |
| **Parameters** | `180M` (eval) / `1.0B` (experimental) | **`1,546,826,389` (`1.55B`)** (`1,543.7M` backbone + `1.92M` Kenyon Vault + `1.19M` `SegmentScorer v2`) |
| **Tensor Regularity** | `981` tensors (`~61–62` per layer: Salience Binder, FFT bind, CMS sketches, GTRM scans) | **`485` tensors (`3` global + `6` `SegmentScorer v2` + `28 × 17` per layer)** — 100% regular dense matrices/vectors |
| **Attention & Memory** | Custom multi-gate scan soup | **6:1 GQA Block Sliding-Window (`W=1024`, active `2W=2048`, auto-pinned `n_pin <= 512`) + `SegmentScorer v2` (`B=1024`) + 49-Cell Co-Prime Kenyon Vault** |
| **Inference State Scaling** | Dynamic / framework-dependent | **Strictly $O(1)$ fixed (`552.9 MB` C FP32 arena / `3.83 GiB` GPU BF16 at `2K`, `32K`, `100K`, or `1M+` tokens; zero `malloc`)** |
| **Checkpoint Container** | `956.7 MB` FP32 Safetensors (`104 KB` JSON header) | **Flat `VLT3` binary (`4,096 B` fixed C header, `128 B` directory entries, `64 B` alignment, CRC32 + `< 2 ms` AVX2 finite scan)** |
| **Weight Releases** | FP32 only | **`BF16` master (`3,104.3 MB`), `Q8_0` (`1,657.2 MB`), Clipped-MSE Hybrid `Q4_0` (`1,022.5 MB`)** |
| **Tokenizer** | `SmolLM-135M` (`49,152` vocab; `<jupyter_code>`, `<gh_stars>` artifacts; JSON parser required) | **Embedded `VTOK` binary ChatML BPE (`151,665` vocab, UAX #15 NFC, `>32 KB` dynamic window, zero JSON parsing)** |
| **Freestanding Runtime** | None (PyTorch / HF Python only) | **`runtime/vault3.c` + `vault3.h` + `vault3_nfc_tables.h` (`-std=c99 -ffreestanding -nostdlib`, `0` external symbols)** |

---

## 2. Mathematical Architecture Specification

### 2.1 Hyper-Parameters (`VAULT-v3-1.5B`)

| Symbol | Header Field | Value | Meaning |
| :--- | :--- | ---: | :--- |
| $V$ | `vocab_size` | `151,936` | Padded embedding/LM-head rows (`151,665` active BPE + ChatML tokens) |
| $d$ | `d_model` | `1,536` | Residual stream hidden dimension ($H \cdot D = 1536$) |
| $L_{\text{layers}}$ | `n_layers` | `28` | Identical Transformer + Segment Memory + Kenyon Vault blocks |
| $H$ | `n_heads` | `12` | Query attention heads |
| $G$ | `n_kv_heads` | `2` | Key/Value & Kenyon Vault heads (`6:1` GQA group size $R = H / G = 6$) |
| $D$ | `head_dim` | `128` | Per-head dimension |
| $F$ | `d_ff` | `8,960` | SwiGLU intermediate dimension |
| $W$ | `window` | `1,024` | Block sliding-window chunk size (active KV ring holds $2W = 2,048$ tokens max; configurable down to $W=512$) |
| $S$ | `n_sink` | `4` | Base StreamingLLM attention-sink tokens (`auto_pin` expands up to `V3_MAX_SINK = 512` for ChatML system prompts) |
| $B$ | `seg_budget` | `1,024` | Fixed evicted Segment Memory KV slots (`seg_min = 16`, `seg_max = 48`, `seg_layers = (8, 16, 24)`, `seg_hidden = 256`) |
| $\mathcal{P}$ | `primes` | `(13, 17, 19)` | Co-prime Kenyon routing banks ($N_B = 3$, total cells $C = 13 + 17 + 19 = 49$) |
| $\varepsilon_{\text{rms}}$ | `rms_eps` | $10^{-6}$ | RMSNorm epsilon |
| $\theta$ | `rope_theta` | $10^{6}$ | RoPE base period |
| $\varepsilon_{\text{vault}}$ | `vault_eps` | $10^{-4}$ | Kenyon cell denominator threshold |

### 2.2 Normalization & Projections
For input residual vector $x \in \mathbb{R}^d$ at layer $\ell \in \{0, \dots, 27\}$:
$$\mathrm{RMSNorm}(x; \gamma) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1} x_i^2 + \varepsilon_{\text{rms}}}}$$
Let $u = \mathrm{RMSNorm}(x; \gamma_{\text{ln1}})$. The raw projections are:
$$q^{\text{raw}} = W_q u + b_q \in \mathbb{R}^{H \times D}, \quad k^{\text{raw}} = W_k u + b_k \in \mathbb{R}^{G \times D}, \quad v = W_v u + b_v \in \mathbb{R}^{G \times D}$$
Notice $W_q, W_k, W_v$ carry explicit float32 biases $(b_q, b_k, b_v)$, while $W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ are bias-free.

### 2.3 Constant-Precision Re-Based RoPE, Auto-Pinned Sinks & Segment KV Memory
Inverse frequencies for $i \in \{0, \dots, D/2 - 1\}$ are $\omega_i = \theta^{-2i/D}$. Using the rotate-half pairing $(i, i + D/2)$, rotation $R_p(z)$ at integer position $p$ maps:
$$R_p(z)_i = z_i \cos(p\omega_i) - z_{i + D/2}\sin(p\omega_i), \qquad R_p(z)_{i + D/2} = z_{i + D/2}\cos(p\omega_i) + z_i \sin(p\omega_i)$$

In streaming inference at absolute token index $t \ge 0$ ($c = t \bmod W$), RoPE inner products depend only on relative distance, so `VAULT v3` keeps all effective RoPE indices strictly bounded in $[-S_{\text{pin}} - B, 2W - 1]$ at $10^6+$ token horizons:
- **Current & Previous Chunk (`k_prev`, `k_cur`):** Current token sits at $p_{\text{eff}}(t) = c$ if $t < W$, else $W + c \in [W, 2W - 1]$; at every chunk rollover $t = mW$ ($m \ge 2$), `k_prev` is shifted by $R_{-W}$ to $[0, W - 1]$.
- **Auto-Pinned Sink Prefix (`k_sink`, $n_{\text{pin}} \le 512$):** Captures BOS + the initial `<|im_start|>system ... <|im_end|>\n` prefix in raw pre-RoPE form (`k_sink_raw`) and rotates them to contiguous negative indices immediately preceding Segment Memory (`-M_{\text{cnt}} - n_{\text{pin}} \dots -M_{\text{cnt}} - 1$).
- **Coherent Segment KV Memory (`k_seg`, $M_{\text{cnt}} \le B$):**
  - **Scoring & Habituation (`SegmentScorer v2`):** Concatenates normalized hidden states from layers `(8, 16, 24)` (`3d = 4,608` dims) with causal APL feature habituation ($e_t = 0.985\, e_{t-1} + 0.015\, |f_t|$, $h_t = f_t / (1 + 0.35\, e_{t-1})$) through `LayerNorm(4608) -> Linear(4608, 256) -> SiLU -> Linear(256, 1)`.
  - **3-Tier Structural Segmentation & Quarantine:** Segments break cleanly on `[16, 48]` boundaries preferring `.?!\n;` (Tier 1), JSON/pipe/TSV `},` / `",` / `|` / `\t` (Tier 2), or CSV `,` (Tier 3), while demoting non-pinned authority-spoofing sponges (`[CRITICAL`, `OVERRIDE`, `SYSTEM DIRECTIVE`, etc.) by `-6.0`.
  - **Turn-Gated 2-Tier FNV-1a Shingle Resonance, 2-Hop Joins & Temporal Rollback:** Evicted segments are indexed into a static open-addressed FNV-1a shingle table (Tier-A strict alphanumeric IDs + Tier-B pure-alphabetic words $\ge 4$ chars). At user-query turn boundaries, `VAULT v3` computes:
    - **APL Divisive Surround Inhibition:** Non-matching background segments receive budget-invariant suppression $\mathrm{bg} = -(1.0 + \max(0, \ln(M_{\text{cnt}} / 1024)))$, while matched segments receive `+5.0` resonance, sub-clause isolation on packed JSON/pipe/CSV lines, and an orphan-value newline continuation halo (`s_0 \to s_0 + 1`).
    - **3-Way GQA Head Split (`2 Free / 2 Archival / 2 Supersession` per KV group):** Free heads (`h % 6 in {0,1}`) remain un-gated; archival heads (`{2,3}`) preserve first-mention `[t=1]` anchors; supersession heads (`{4,5}`) apply `Chronos` log-temporal gap bias $+2.5 \ln(1 + \Delta\text{pos}/64)$ (or exact rollback target boosts when revert cues + anchor/ordinal cues appear).
    - **2-Band Hop-2 Spreading Activation:** Early layers (`7..15`) attend to Hop-1 bridge segments while mid/late layers (`16..25`) attend to Hop-2 target segments discovered via bridge shingles, linearly annealing over `gen_dist = 18..38` so generation fluency is unaffected.

### 2.4 Kenyon Vault Co-Prime Associative Memory
When $t$ crosses $t = mW$ ($m \ge 2$), the evicted chunk is also compressed into per-layer, per-KV-head associative Kenyon cells $M_{g} \in \mathbb{R}^{C \times D \times D}$ and normalizers $z_{g} \in \mathbb{R}^{C \times D}$ ($C = 49$, co-prime banks $\mathcal{P} = (13, 17, 19)$, `4,199` combinatorial triplets):
1. **Key/Query Adaptation & Write Gate:**
   $$\tilde{k}_{g,t} = \mathrm{UnitRMS}(k^{\text{raw}}_{g,t})\, A^{(k)}_g, \qquad \tilde{q}_{h,t} = \mathrm{UnitRMS}(q^{\text{raw}}_{h,t})\, A^{(q)}_g, \qquad \beta_{g,t} = \sigma(w^{(\beta)}_g \cdot u_t + b^{(\beta)}_g)$$
   with positive feature map $\phi(z) = \mathrm{elu}(z) + 1 > 0$.
2. **Chunkwise Delta-Rule Write & Overflow Rescaling:**
   $$\hat{v}_{\tau, b} = \frac{\phi(\tilde{k}_\tau)^\top M_{c_b(\tilde{k}_\tau)}}{\phi(\tilde{k}_\tau)^\top z_{c_b(\tilde{k}_\tau)}}, \qquad \delta_{\tau, b} = \beta_\tau (v_\tau - \hat{v}_{\tau, b}), \qquad M_c \mathrel{+}= \phi(\tilde{k}_\tau)\delta_{\tau, b}^\top, \qquad z_c \mathrel{+}= \beta_\tau \phi(\tilde{k}_\tau)$$
   Whenever $\max_i z_{c, i} > 65,536$, both $(M_c, z_c)$ are multiplied by $0.5$, preserving the exact retrieval ratio $\phi(\tilde{q})^\top M_c / \phi(\tilde{q})^\top z_c$ while guaranteeing zero `float32` overflow across infinite token streams.
3. **Bank-Averaged Read & Gated Residual Injection ($t \ge 2W$):**
   $$o_h = o^{\text{swa+seg}}_h + \tanh(\alpha_h)\, m_{h, t}, \qquad x' = x + W_o \, \mathrm{concat}(o_0, \dots, o_{H-1})$$

---

## 3. Complete Regular Tensor Table (`485` Tensors)

Every tensor is identified in `VLT3` by a deterministic 32-bit integer `id` — **no string parsing or hash tables are used by the loader**:
- Global backbone & routing tensors (`3`): `id = 1..3`.
- Global `SegmentScorer v2` tensors (`6`): `id = 4..9` (`0` in Kenyon-only files when `seg_n_feat == 0`).
- Per-layer tensors (`28 × 17 = 476`): `id = ((layer + 1) << 8) | kind` for `layer in 0..27` and `kind in 1..17`.

| Scope | `id / kind` | Canonical Name | Shape (`dims`) | `BF16` Dtype | `Q8_0` Dtype | Hybrid `Q4_0` Dtype |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| Global | `1` | `embed_tokens.weight` (tied LM head) | `[151936, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(kept 8-bit)* |
| Global | `2` | `norm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Global | `3` | `vault_route` | `[128, 49]` | `F32` | `F32` | `F32` |
| Global | `4` | `seg_scorer.ln.weight` | `[4608]` | `F32` | `F32` | `F32` |
| Global | `5` | `seg_scorer.ln.bias` | `[4608]` | `F32` | `F32` | `F32` |
| Global | `6` | `seg_scorer.fc1.weight` | `[256, 4608]` | `F32` | `F32` | `F32` |
| Global | `7` | `seg_scorer.fc1.bias` | `[256]` | `F32` | `F32` | `F32` |
| Global | `8` | `seg_scorer.fc2.weight` | `[1, 256]` | `F32` | `F32` | `F32` |
| Global | `9` | `seg_scorer.fc2.bias` | `[1]` | `F32` | `F32` | `F32` |
| Layer `l` | `1` | `layers.{l}.input_layernorm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `2` | `layers.{l}.self_attn.q_proj.weight` | `[1536, 1536]` | `BF16` | `Q8_0` | `Q8_0` if `l ∈ {0,1,26,27}` else `Q4_0` |
| Layer `l` | `3` | `layers.{l}.self_attn.q_proj.bias` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `4` | `layers.{l}.self_attn.k_proj.weight` | `[256, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(6:1 GQA KV kept 8-bit)* |
| Layer `l` | `5` | `layers.{l}.self_attn.k_proj.bias` | `[256]` | `F32` | `F32` | `F32` |
| Layer `l` | `6` | `layers.{l}.self_attn.v_proj.weight` | `[256, 1536]` | `BF16` | `Q8_0` | `Q8_0` *(6:1 GQA KV kept 8-bit)* |
| Layer `l` | `7` | `layers.{l}.self_attn.v_proj.bias` | `[256]` | `F32` | `F32` | `F32` |
| Layer `l` | `8` | `layers.{l}.self_attn.o_proj.weight` | `[1536, 1536]` | `BF16` | `Q8_0` | `Q8_0` if `l ∈ {0,1,26,27}` else `Q4_0` |
| Layer `l` | `9` | `layers.{l}.post_attention_layernorm.weight` | `[1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `10` | `layers.{l}.mlp.gate_proj.weight` | `[8960, 1536]` | `BF16` | `Q8_0` | `Q4_0` |
| Layer `l` | `11` | `layers.{l}.mlp.up_proj.weight` | `[8960, 1536]` | `BF16` | `Q8_0` | `Q4_0` |
| Layer `l` | `12` | `layers.{l}.mlp.down_proj.weight` | `[1536, 8960]` | `BF16` | `Q8_0` | `Q4_0` |
| Layer `l` | `13` | `layers.{l}.self_attn.vault.a_k` | `[2, 128, 128]` | `F32` | `F32` | `F32` |
| Layer `l` | `14` | `layers.{l}.self_attn.vault.a_q` | `[2, 128, 128]` | `F32` | `F32` | `F32` |
| Layer `l` | `15` | `layers.{l}.self_attn.vault.beta.weight` | `[2, 1536]` | `F32` | `F32` | `F32` |
| Layer `l` | `16` | `layers.{l}.self_attn.vault.beta.bias` | `[2]` | `F32` | `F32` | `F32` |
| Layer `l` | `17` | `layers.{l}.self_attn.vault.alpha` | `[12]` | `F32` | `F32` | `F32` |

---

## 4. `VLT3` Flat Binary Container & Quantization Layouts

All integers and floats in `VLT3` are **little-endian**. Every major section (`tensor_dir`, `tokenizer`, and every individual tensor payload) starts at an offset divisible by `64` bytes (cache-line & AVX-512 aligned), so the file can be mapped directly into physical/virtual memory and read in-place with zero copies.

### 4.1 Fixed `4096`-Byte File Header

| Byte Offset | Type | Field | Validation Rule in `v3_load()` |
| ---: | :--- | :--- | :--- |
| `0..3` | `char[4]` | `magic` | Must equal `"VLT3"` (`0x56, 0x4C, 0x54, 0x33`) |
| `4..7` | `uint32` | `version` | Must equal `1` |
| `8..11` | `uint32` | `header_size` | Must equal `4096` |
| `12..15` | `uint32` | `endian_tag` | Must equal `0x01020304` |
| `16..23` | `uint64` | `file_size` | Must equal exact buffer byte length |
| `24..27` | `uint32` | `alignment` | Must equal `64` |
| `28..31` | `uint32` | `n_tensors` | Number of `128`-byte entries in directory (`485`, or `479` without scorer) |
| `32..39` | `uint64` | `tensor_dir_offset` | `>= 4096`, bounded within `file_size` |
| `40..47` | `uint64` | `tensor_dir_size` | Must equal `n_tensors * 128` |
| `48..55` | `uint64` | `tokenizer_offset` | `>= 4096`, `64`-byte aligned, bounded |
| `56..63` | `uint64` | `tokenizer_size` | `>= 128`, bounded within `file_size` |
| `64..71` | `uint64` | `data_offset` | `>= 4096`, `64`-byte aligned, bounded |
| `72..79` | `uint64` | `data_size` | Bounded within `file_size` |
| `80..115` | `9 × uint32` | `vocab_size .. n_banks` | Non-zero; `n_layers <= 64`, `head_dim <= 256` even, `n_heads % n_kv_heads == 0` |
| `116..147` | `8 × uint32` | `primes[8]` | First `n_banks` entries `> 0`; sum must equal `n_cells` |
| `148..151` | `uint32` | `n_cells` | Sum of active co-prime banks (`49`) |
| `152..163` | `3 × float32` | `rms_eps, rope_theta, vault_eps` | Finite; `rms_eps > 0`, `rope_theta > 1`, `vault_eps > 0` |
| `164..183` | `5 × uint32` | `bos_id, eos_id, im_start_id, im_end_id, pad_id` | Special token IDs (`< vocab_size`) |
| `184..191` | `2 × uint32` | `weight_format, flags` | Format tag (`0,1,2,8,9`); bitmask (`1=tied`, `2=VTOK`, `4=seg_scorer`) |
| `192..255` | `char[64]` | `model_name` | NUL-padded UTF-8 string |
| `256..259` | `uint32` | `n_sink` | `<= 512` and `<= window` (`4`) |
| `260..279` | `5 × uint32` | `seg_budget, seg_hidden, seg_min, seg_max, seg_n_feat` | `seg_budget <= 16384`, `seg_hidden <= 1024`, `seg_min <= seg_max <= window`, `seg_n_feat <= 4` |
| `280..295` | `4 × uint32` | `seg_layers[4]` | Layer indices `< n_layers` (`[8, 16, 24, 0]`) |
| `296..4091` | `uint8[3796]` | `reserved` | Zero-filled |
| `4092..4095` | `uint32` | `header_crc32` | IEEE 802.3 CRC32 of bytes `0..4091` (verified **before** reading fields) |

**Unconditional `< 2 ms` SIMD Finite Integrity Scan:** Even when `verify_crc = 0` (`mmap` fast startup), `v3_load()` unconditionally scans **100%** of `F32`, `F16`, `BF16`, `Q8_0`, and `Q4_0` weights and block scales for `NaN`/`Inf` using AVX2 strided masks (`V3_E_NAN = -12`).

### 4.2 Clipped-MSE Hybrid Quantization (`QK = 32`)
1. **`Q8_0` (`dtype = 8`, `34` bytes / `32` weights = `8.50 bits/weight`):** `float16` scale $d = \max_j |w_j| / 127$ + `int8_t q[32]`.
2. **Clipped-MSE Hybrid `Q4_0` (`dtype = 9`, `18` bytes / `32` weights = `4.50 bits/weight`):** Evaluates 8 candidate clipping ratios $r \in \{-1.00/8, -0.98/8, -0.96/8, -0.94/8, -0.92/8, -0.89/8, +1.00/7.5, +0.96/7.5\}$ per 32-weight block to minimize reconstruction MSE $\|\hat{w} - w\|_2^2$, while retaining `Q8_0` on `embed_tokens.weight`, all 28 layers' `6:1` GQA `k_proj`/`v_proj`, and boundary layers `0, 1, 26, 27` `q_proj`/`o_proj` (`1,022.5 MB` total).

Export command (`tools/export_vlt3.py`):
```bash
python tools/export_vlt3.py --vault_ckpt vault_params/vault_v3_stage_a_best.pt --seg_ckpt vault_params/vault_v3_seg_scorer.pt --format bf16 --out dist/vault_v3_1.5b_bf16.vlt3
python tools/export_vlt3.py --vault_ckpt vault_params/vault_v3_stage_a_best.pt --seg_ckpt vault_params/vault_v3_seg_scorer.pt --format q8_0 --out dist/vault_v3_1.5b_q8_0.vlt3
python tools/export_vlt3.py --vault_ckpt vault_params/vault_v3_stage_a_best.pt --seg_ckpt vault_params/vault_v3_seg_scorer.pt --format q4_0 --out dist/vault_v3_1.5b_q4_0.vlt3
```

---

## 5. `VTOK` Binary Tokenizer, UAX #15 NFC & `v3_governor`

- **Binary Layout (`4.02 MB`, Zero JSON):** `tok_table` (`n_vocab × 8 B`), `tok_pool`, `tok_merges` (`151,387 × 12 B`), `tok_special` (`22 × 4 B`), `tok_ranges` (`L*`/`N*` intervals), and `tok_bytes` (`256 × 4 B`).
- **Built-in UAX #15 NFC + Arbitrary-Length BPE:** `v3_encode_ex()` fast-paths pure ASCII/pre-composed UTF-8 (`< 0xCC`) with zero copies, and automatically applies full canonical decomposition + Hangul composition + canonical combining-class sorting + canonical composition (`runtime/vault3_nfc_tables.h`) whenever combining marks or Hangul Jamo appear. Single pre-tokenizer pieces exceeding `V3_TOK_MAX_PIECE = 32,768` bytes use an overlapping `32 KB` BPE merge window (`100%` identical to HuggingFace `AutoTokenizer` up to `1 MB+` single pieces).
- **Delimiter Injection Prevention (`allow_special = 0`):** Untrusted user/tool strings passed via `v3_encode_ex(..., allow_special=0)` cannot inject `<|im_start|>system` or `<|im_end|>`.
- **Constitutional Output Governor (`v3_governor` / `CAAIGovernor`):** Deterministic $O(1)$-arena pre-sampling logit filter (`top_k_guard = 64`) that normalizes zero-width characters, separators (`_`, `-`, `.`), Greek/Cyrillic homoglyphs, character reversal, 4-phase Base64/Base64URL streams, and 2-phase Hex (`0xNN` / `\xNN`) byte streams, masking any token that completes a forbidden `SYSCALL_EXEC(` directive to `-1e30f`.

---

## 6. Freestanding `x86_64` Porting Guide (`runtime/vault3.h`, `runtime/vault3.c`)

### 6.1 Dependencies & Compiler Flags
`runtime/vault3.c` includes **only** `<stddef.h>`, `<stdint.h>`, `"vault3_nfc_tables.h"`, and (when `__AVX2__` is defined) `<immintrin.h>`. Every function compiles under `-Wframe-larger-than=2048` (`< 2 KB` stack frames) and `nm -u` confirms **zero undefined external symbols**:
```bash
# Portable scalar freestanding object (no SIMD required):
gcc -std=c99 -O2 -ffreestanding -fno-math-errno -fno-stack-protector -nostdlib -mno-red-zone -Wall -Wextra -Wframe-larger-than=2048 -c runtime/vault3.c -o vault3_scalar.o

# x86_64 AVX2 + FMA + F16C freestanding object (recommended on Haswell+ / Zen+ bare metal):
gcc -std=c99 -O2 -ffreestanding -fno-math-errno -fno-stack-protector -nostdlib -mno-red-zone -mavx2 -mfma -mf16c -Wall -Wextra -Wframe-larger-than=2048 -c runtime/vault3.c -o vault3_avx2.o
```

### 6.2 Deterministic Memory Layout & Static Arena Sizing
All mutable memory is owned by the caller and passed in as **`64`-byte aligned** buffers:
1. **Read-Only Model Buffer (`buf`, `64`-byte aligned):**
   - `BF16`: `3,104.3 MB` | `Q8_0`: `1,657.2 MB` | Hybrid `Q4_0`: `1,022.5 MB`
2. **Inference State Arena (`v3_state_arena_size(&m)` = `552.9 MB` for `W=1024, seg_budget=1024, sink_cap=512`):**
   - Strictly constant at `128` tokens, `8,192` tokens, `100,400` tokens, or `10,000,000` tokens (`v3_state_arena_size_ex(&m, max_seg_budget)` sizes custom budgets).
3. **Tokenizer Arena (`v3_tok_arena_size(&m)` = `9.7 MB`):**
   - Holds the `524,288`-slot merge hash table, $O(n \log n)$ BPE priority heap, `>32 KB` sliding-window scratch, and UAX #15 NFC buffers.

---

## 7. Verified Benchmark Summary (Round 4)

Raw JSON outputs are stored in `benchmarks/vault_v3_full_benchmark.json`, `benchmarks/vault_v3_round4_c_verification.json`, and `benchmarks/vault_v3_round4_gpu_verification.json`.

- **4-Way Numerical Parity (`tests/test_vlt3_runtime.py`):** PyTorch `forward == prefill+step == step == C99` across `F32 / BF16 / Q8_0 / Q4_0` and all Round-4 2-hop, rollback, Tier-B, sub-clause, orphan-halo, and divisive-budget paths (`max |Δlogit| <= 3.50e-5`, **`100.0%` top-1 agreement**, `1T == 8T` `0.0` ULP bit-identical).
- **8K Multi-Fact Retrieval (26 evicted facts > 2W):** **`23/26 (88.5%)`** at `B = 1024` vs `14/26 (53.8%)` for full-attention Qwen2.5-1.5B-Instruct.
- **Adversarial Long-Context Suite:** `8/8` Chunk-0 Cliff, `10/10` Newline-Split KV (`B = 512` & `1024`), `8/8` Dense Homogeneous Registry (`n = 8, 24, 80 @ B = 2048`) & `7/8` (`n = 120`), `6/8` Pure-Alphabetic ID-Free Registry (`B = 1024`), `7/8` 2-Hop Evicted Relational Join (`n = 8 @ B = 1024` & `n = 32 @ B = 2048`), `6/6` across all four Contradictory & Indirect Rollback chains, `100%` Scorer-Sponge (`0..120`) & `20x` Chaff survival, `8/8` Minified JSON + `8/8` Pipe secret, and **`8/8 (100%)` at `100.4K` tokens** (`3.832–3.834 GiB` flat GPU VRAM, `1,356 tok/s`).
- **Freestanding C99 CPU Performance & Sanitizers:** Hybrid `Q4_0` reaches **`45.76–46.16 tok/s`** (`21.7 ms/tok`) at `24` threads with **`1.51 GB`** flat RSS across `1..64` threads; `ASan + LSan + UBSan` (`3,000` steps) and `TSan` (`8T`) `0` errors; `14/14` corrupt/NaN `.vlt3` inputs rejected (`rc = 2`); `0/12` GPU and `0/9` C `--chat` OPI governance leaks.
