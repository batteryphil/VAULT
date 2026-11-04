# VAULT & VAULT v2 (`FlashVAULT v2`): Technical Dossier for Independent AI Peer Review

**Author & Architect:** Philip Bartsch (`@batteryphil`)
**Repository:** https://github.com/batteryphil/VAULT (Dual License: AGPL-3.0-or-later OR Commercial)
**Snapshot Status:** Live Stage-1 WSD Pretraining at Step 5,900 / 7,500 (193.33M / 245.76M `L=2048` tokens; Holdout `Val PPL = 11.06`, `gnorm = 0.62`, `2,699 tok/s` on a single 16 GB AMD Radeon RX 9060 XT GPU)

---

## 0. Instructions for the Reviewing AI
Please review this architecture, kernel mathematics, training methodology, and empirical evidence critically and adversarially. Do **not** flatter the design. Specifically evaluate:
1. Mathematical soundness of the **Cell-Masked Chunkwise Compact WY Householder/Delta recurrence** and its **3:1 Grouped-Vault Attention (GVA)** formulation.
2. Potential gradient bottlenecks in the **Straight-Through Estimator (STE) discrete Co-Prime cell router** and **occupancy-based harmonic consensus readout**.
3. Validity of what is **proven** vs. **still unproven** in the empirical tables below, including our root-cause diagnosis of the open-vocabulary entity-copy bottleneck observed during live `2K` NIAH probes.
4. Concrete architectural or training failure modes that may surface at `8K–32K` length extrapolation or `1B–7B` parameter scale.

---

## 1. Problem Statement & Core Architectural Thesis

Sequence models face a fundamental memory-compute trilemma:
* **Softmax Transformers** store an explicit KV table ($O(L)$ state memory, $O(L^2)$ prefill compute), yielding exact token lookup at the cost of unbounded KV cache growth.
* **Uniformly Decaying Linear RNNs / SSMs** (Mamba, Mamba-2, GLA, RetNet) compress all history into a single fixed matrix $S_t = \gamma_t S_{t-1} + k_t v_t^\top$. Exponential decay ($\gamma_t < 1$) washes out early facts over $10^4–10^5$ tokens, while dense rank accumulation causes cross-talk once unique writes approach head dimension $D$.
* **Single-Matrix DeltaNet / Gated DeltaNet** improves single-matrix capacity via orthogonal Householder-like erasure $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, but every token still mutates the *same* $D \times D$ matrix per head ($O(D)$ rank ceiling per head) and typically maintains 1:1 per-head fast-weight state ($H \times D \times D$).

### What VAULT Does Differently
**VAULT** (**V**ariance-Weighted **A**ssociative **U**nitary **L**attice **T**ensor) partitions fast-weight associative memory per group into $C_{\text{tot}} = \sum_{r=1}^{R} P_r$ **isolated Co-Prime Orthogonal Nullspace Vault Cells** (in `VAULT v2`, $P = \{17, 19, 23, 29\} \implies 88$ matrices of shape $D \times D$ per vault group, $R = 4$ banks):
1. **Zero Passive Decay on Untouched Cells ($\lambda = 1.0$):** At any token $t$, a vault group writes to only $R = 4$ cells (one cell per co-prime bank). The remaining $84$ cells in that group undergo **strictly zero mutation and zero decay** ($S_c^{(t)} = S_c^{(t-1)}$).
2. **Chinese Remainder Decorrelation Across Collisions:** Because bank moduli $\{17, 19, 23, 29\}$ are pairwise co-prime, two distinct keys that collide in Bank 1 ($p_1 = 17$) map to identical 4-tuples across all 4 banks with probability only $1 / (17 \times 19 \times 23 \times 29) = 1 / 216,611$ under uniform routing.
3. **Orthogonal Delta Projection Inside Active Cells + Selective Write Gate:** Within an addressed cell $c$, writes apply the exact error-correcting projection $S_c \leftarrow S_c + \beta_t (v_t - S_c \phi(k_t)) \phi(k_t)^\top$ with $\|\phi(k_t)\|_2 = 1$ and learned token-selective write/erase gate $\beta_{b,g,t} \in (0.02, 0.96)$, suppressing boilerplate writes while locking salient bindings into orthogonal subspaces.
4. **True 3:1 Grouped-Vault Attention (`GVA`):** Analogous to Grouped-Query Attention (`GQA`) for KV caches—but applied to *recurrent fast-weight vaults*—`VAULT v2` pairs $H = 12$ independent query heads (each with its own query vector $\phi(q_h)$ and its own routed cell index `cell_q[h]`) over $G = 4$ shared Co-Prime vaults. Three query heads per group can simultaneously read *three completely different cells* out of the same shared 88-cell vault while cutting recurrent state memory and HBM checkpoint bandwidth **3x**.

---

## 2. Exact Mathematical Specification of `VAULTLMv2` (151,329,440 Parameters)

### 2.1 Model Hyperparameters
* **Vocabulary:** `49,152` (`HuggingFaceTB/SmolLM-135M` BPE with single-token `<|endoftext|>=0`, `<|im_start|>=1`, `<|im_end|>=2`, `<file_sep>=5`; tied input/output embeddings)
* **Backbone:** `n_layers = 16`, `d_model = 768`, Pre-`RMSNorm`, `SwiGLU` MLP (`d_ff = 2048`), GPT-2/Llama residual output scaling $\sigma = 0.02 / \sqrt{2 \cdot n_{\text{layers}}}$
* **Attention Geometry:** `n_q_heads (H) = 12`, `n_vault_groups (G) = 4` (`heads_per_group = 3`), `head_dim (D) = 64`, `chunk_size (C) = 64`
* **Co-Prime Vault Topology:** $P = (17, 19, 23, 29) \implies R = 4$ banks, $C_{\text{tot}} = 88$ cells per group (`88 * 64 * 64 * 4 = 1,441,792` floats = `5.50 MB` FP32 state across all 4 groups per layer, or `1.375 MB` per group)

### 2.2 Step-by-Step Layer Forward Pass (`src/vault_v2_model.py`)
Given pre-normalized hidden input $x \in \mathbb{R}^{B \times L \times 768}$:

1. **Shared-Metric Group Anchor + Causal `ShortConv1D(kernel_size=4)` + `SiLU`:**
   To align $Q$ and $K$ in the same metric space from Step 0 while binding 4-token local n-grams (subword identifiers, numbers, punctuation) prior to routing:
   $$a = W_{\text{anchor}} x \in \mathbb{R}^{B \times L \times (G \cdot D)}$$
   $$q = \text{SiLU}(\text{CausalConv1D}_4(\text{repeat}_3(a) + W_q x)) \in \mathbb{R}^{B \times H \times L \times D}$$
   $$k = \text{SiLU}(\text{CausalConv1D}_4(a + W_{k,\Delta} x)) \in \mathbb{R}^{B \times G \times L \times D}$$
   $$v = \text{SiLU}(\text{CausalConv1D}_4(W_v x)) \in \mathbb{R}^{B \times G \times L \times D}$$

2. **Co-Prime Ternary Kenyon Routing with Straight-Through Estimator (`_route_cells_v2`):**
   Using 1.58-bit ternary STE weights $W_{\text{kenyon}} \in \{-1, 0, +1\}^{88 \times D}$, homeostatic center-surround `FusedAPLNorm`, prime reciprocal anchors $\alpha_c = 1/p_r$, and annealed inverse temperature $\tau^{-1}$ (`step_ratio` $0 \to 1$ over the first 30% of pretraining):
   $$z_q = \text{ReLU}(\text{APLNorm}(\hat{q} W_{\text{kenyon}}^\top)) \odot \alpha \cdot \tau^{-1} \in \mathbb{R}^{B \times H \times L \times 88}$$
   $$z_k = \text{ReLU}(\text{APLNorm}(\hat{k} W_{\text{kenyon}}^\top)) \odot \alpha \cdot \tau^{-1} \in \mathbb{R}^{B \times G \times L \times 88}$$
   Within each co-prime slice $r \in \{1..4\}$ of width $p_r$:
   $$c_{q,r} = \text{offset}_r + \arg\max(z_{q,r}), \quad c_{k,r} = \text{offset}_r + \arg\max(z_{k,r})$$
   Router gradients flow into $q, k, W_{\text{kenyon}}$ via unit-forward multiplicative STE factors $\text{ste}_q = 1 + (\bar{p}_q - \text{sg}(\bar{p}_q))$, where $\bar{p}_q = \frac{1}{R}\sum_r \max \text{softmax}(z_{q,r})$.

3. **Bounded Positive `PRIME` Feature Kernel & Selective Write Gate:**
   $$\phi(q) = \frac{\exp(\beta_q \cdot \text{clamp}(\text{RMSNorm}(q), -3, 3))}{\|\exp(\beta_q \cdot \text{clamp}(\text{RMSNorm}(q), -3, 3))\|_2} \cdot \text{ste}_q, \quad \beta_q \in [1.2, 3.5]$$
   $$\phi(k) = \frac{\exp(\beta_k \cdot \text{clamp}(\text{RMSNorm}(k), -3, 3))}{\|\exp(\beta_k \cdot \text{clamp}(\text{RMSNorm}(k), -3, 3))\|_2} \cdot \text{ste}_k, \quad \|\phi(k_{b,g,t})\|_2 = 1$$
   $$\beta_{b,g,t} = 0.02 + 0.94 \cdot \sigma(W_\beta x_{b,t} + b_\beta)_g \in (0.02, 0.96)$$

4. **Branch 1 — `FlashVAULT v2` Co-Prime Associative Recurrence:**
   For each token $t = 0 \dots L-1$ and bank $r \in \{0..3\}$, group $g$ updates its addressed cell $c_k = c_{k,r}(b,g,t)$:
   $$S_{b,g,c_k}^{(t)} = S_{b,g,c_k}^{(t-1)} + \beta_{b,g,t}\left(v_{b,g,t} - S_{b,g,c_k}^{(t-1)} \phi(k_{b,g,t})\right)\phi(k_{b,g,t})^\top, \quad n_{b,g,c_k}^{(t)} = n_{b,g,c_k}^{(t-1)} + 1$$
   Each query head $h$ (mapped to group $g = \lfloor h / 3 \rfloor$) reads its $R = 4$ addressed cells $c_{q,r} = c_{q,r}(b,h,t)$ and combines them via occupancy-variance precision weighting:
   $$y_{h,r,t} = S_{b,g,c_{q,r}}^{(t)} \phi(q_{b,h,t}), \quad \sigma_r^2 = \frac{\max(0, n_{b,g,c_{q,r}}^{(t)} - 1)}{D}, \quad w_r = \frac{(\sigma_r^2 + 0.01)^{-1}}{\sum_{j=1}^R (\sigma_j^2 + 0.01)^{-1}}$$
   $$y_{\text{vault}}(b,h,t) = \sum_{r=1}^{R} w_r \, y_{h,r,t}$$

5. **Branch 2 — Strictly $O(L)$ Chunk-Local RoPE + Multi-Scale Decay Syntax Window (`64 x 128`):**
   Input is tiled into non-overlapping chunks of $C = 64$, attending causally over `[prev_chunk, curr_chunk]` ($2C = 128$ tokens) with **chunk-relative RoPE** (positions $0 \dots 127$ inside every window, so zero long-range RoPE phase drift at arbitrary $L$) plus fixed multi-scale geometric head decays $\gamma_h \in [0.66, 0.992]$ (half-lives spanning $1.7$ to $86$ tokens across the 12 heads).

6. **Branch 3 — $O(L)$ Running Macro-Summary (`GTRM`) + Gated Sub-Layer Norm:**
   $$y = \lambda_{\text{vault}} y_{\text{vault}} + \lambda_{\text{syntax}} y_{\text{syntax}} + \lambda_{\text{gtrm}} W_{\text{gtrm}}\left(\frac{1}{t}\sum_{i \le t} v_i\right)$$
   $$\text{out} = W_o \Big( \text{RMSNorm}_{\text{head}}(y) \odot \sigma(W_{\text{gate}} x) \Big)$$

---

## 3. `FlashVAULT v2` Chunkwise Parallel Triton 3.8 Kernel (`src/kernels/flash_vault_v2_triton.py`)

Sequential execution of the 88-cell recurrence in PyTorch is memory-bound and unusable for LLM training. `FlashVAULT v2` reformulates the multi-cell recurrence into exact $64 \times 64$ chunkwise parallel matrix operations:

1. **Cell-Masked Intra-Chunk Compact WY Factorization (`_vault_v2_wy_fwd`):**
   Within chunk $m$ of length $C = 64$ and bank $r$, token $j < i$ mutates the same matrix as token $i$ **if and only if** $c_{k,r,i} == c_{k,r,j}$. Defining the strictly lower-triangular cell-equivalence mask $M_{ij} = \mathbb{I}[i > j \land c_{k,r,i} = c_{k,r,j}]$, the strictly lower-triangular coupling matrix is:
   $$L_{\text{strict}} = \big(\text{diag}(\beta) \, (K K^\top) \big) \odot M \in \mathbb{R}^{64 \times 64}$$
   Because $\|\phi(k_t)\|_2 = 1$ and $\beta_t \in (0.02, 0.96)$, $(I + L_{\text{strict}})$ is unit lower-triangular and unconditionally invertible. Let $A = (I + L_{\text{strict}})^{-1}$; then the chunk's Compact WY factors are:
   $$U = A \, (\text{diag}(\beta) V) \in \mathbb{R}^{64 \times D}, \quad W = A \, (\text{diag}(\beta) K) \in \mathbb{R}^{64 \times D}$$
2. **6-Step Block-Recursive Triangular Solver (`_invert_unit_lower_tri_64`):**
   Instead of sequential row substitution (64 serial steps) or truncated Neumann series (which explodes on collinear repeated tokens), `FlashVAULT v2` computes $(I - A_{\text{neg}})^{-1}$ where $A_{\text{neg}} = -L_{\text{strict}}$ in **6 exact block-doubling steps** (`BS = 1, 2, 4, 8, 16, 32` via `X <- X + (X @ A_sub) @ X`), keeping intermediate values bounded in $[-1, 1]$ under BF16 WMMA.
3. **Inter-Chunk State Scan + Fused Intra/Inter Readout (`_vault_v2_fwd_scan_and_read`):**
   Checkpoints $S_{\text{ckpt}}$ at chunk boundaries (`G = 4` resolution) while fusing intra-chunk causal attention $\mathbb{I}[i \ge j \land c_{q,r,i} = c_{k,r,j}] (Q_h K_g^\top) V_{\text{eff}}$ with inter-chunk vault state readout across all $H = 12$ query heads.
4. **Zero Global Atomics in Forward or Backward:**
   In `_vault_v2_bwd_state_and_grads` and `_vault_v2_wy_bwd`, gradients across the 3 query heads per group are accumulated in SRAM registers inside each `(bg, dv_block)` program instance before writing `dK, dV, dBeta`, completely eliminating global `atomic_add` contention.

---

## 4. Complete Empirical Evidence Matrix

### 4.1 Native C99 100,000-Token Associative Horizon (`c/bench_vault.c`)
*85% Zipfian repeated stream + 15% novel key-value vectors; querying earliest keys written at $t = 0$ after $T$ intervening tokens (`D = 64`):*

| Horizon $T$ | Linear (`ELU+1`) | `GistLayer` (`exp`) | `PRIME-Moment` | Single-Matrix `Dense-RLS` | **VAULT (C99)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 0.1123 | 0.2356 | 0.0806 | -0.0073 | **1.0000 (100% Top-1)** |
| 1,000 | 0.0361 | 0.0473 | 0.0313 | 0.0123 | **1.0000 (100% Top-1)** |
| 8,000 | 0.0295 | 0.0309 | 0.0281 | -0.0386 | **0.9929 (100% Top-1)** |
| 32,000 | 0.0310 | 0.0296 | 0.0303 | -0.0285 | **0.9731 (100% Top-1)** |
| 64,000 | 0.0296 | 0.0282 | 0.0290 | -0.0048 | **0.9623 (95.8% Top-1)** |
| 100,000 | 0.0248 | 0.0244 | 0.0246 | -0.0146 | **0.9645 (100% Top-1)** |

* **Single-Core CPU Speed:** `179,565 tok/s` (`5.57 us/tok`).

### 4.2 Adversarial Falsification Suites (`tests/test_vault_rigorous.py` & `benchmarks/torture_test_vault.py`)
1. **100% Non-Repeating Gaussian Stream (Innovation Gate Disabled — Every Token Writes):**
   Querying oldest 32 memories buried under $N$ unique random facts:
   * At $N = 1,024$: `ELU+1 = 0.0007`, `Dense-RLS = 0.0057`, Replicated 4-Bank ($P=\{17..29\}$) = `0.9354 (100%)`, Subspace-Factored VAULT ($P=\{83..101\}$) = **`0.9929 (100% Top-1)`**.
   * At $N = 3,072$: `ELU+1 = -0.0153`, `Dense-RLS = 0.0184`, Replicated 4-Bank = `0.1353 (0% Top-1)`, Subspace-Factored VAULT = **`0.8849 (100% Top-1)`**.
2. **Adversarial Near-Collinear Key Attack ($k_i = \rho k_{\text{base}} + \sqrt{1-\rho^2}\epsilon_i$):**
   * Holds `100.0%` Top-1 up to $\rho = 0.90$, `96.9%` at $\rho = 0.95$, `90.6%` at $\rho = 0.98$, then degrades to `62.5%` at $\rho = 0.99$ and `12.5%` at $\rho = 0.999$.

### 4.3 Small-Scale Matched From-Scratch Race (`benchmarks/race_from_scratch.py`)
*Matched ~3.7M–3.9M parameter models trained identically on TinyStories (`Round 1`) and Multi-Pair Associative Recall (`Round 2`):*
* **Language Val PPL (150 steps):** Transformer = `590.75` | Linear SSM = `625.53` | **VAULT = `461.45`**
* **Associative Recall Accuracy (200 steps):** Linear SSM = `1.6%` (chance) | **VAULT = `85.3%`** | Softmax Transformer = `99.4%`

### 4.4 Surgical Adaptation of Pretrained `SmolLM-135M` (`benchmarks/run_smollm_vault_path2.py`)
*Replaced 24 of 30 (`80%`) Softmax attention layers in `HuggingFaceTB/SmolLM-135M` (`9.01` teacher PPL), freezing `126.7M` params and aligning `8.06M` VAULT adapter params for `100` steps (`53.6s`):*
* **Zero-Shot Transplant PPL (0 steps):** Moment-only (`PRIME-Moment`) = `485,165,195` (collapsed) vs. **VAULT = `245.80`**.
* **53.6-Second Aligned PPL (100 steps):** **`35.20 PPL`** with `80%` of Transformer KV cache eliminated.
* **4,096-Token Passkey Cosine in Real `SmolLM-135M` $Q/K$ Space:** `ELU+1` = `0.0172` vs. **VAULT = `0.7321`** (`42.5x` higher signal retention; `0.8070` at `L=2048`).
* **Takeaway:** Proved real LLM $Q/K$ compatibility, while the gap (`35.20` vs `9.01` teacher PPL) proved frozen Softmax projections are suboptimal for Householder nullspace routing—motivating native from-scratch pretraining (`VAULTLMv2`).

### 4.5 `FlashVAULT v2` Kernel Verification & Single-GPU Throughput (`tests/test_vault_v2.py`)
*Hardware: Single consumer `AMD Radeon RX 9060 XT` (16 GB VRAM, ROCm 7.2, Triton 3.8):*
* **FP32 Forward/Backward Cosine Parity (`L=200` non-chunk-aligned, `H=12, G=4, D=64`):**
  `y = 0.9999999` | `dQ = 1.0000000` | `dK = 0.9999999` | `dV = 0.9999999` | `dBeta = 1.0000000`
* **BF16 WMMA Forward/Backward Cosine Parity (`L=200`):**
  `y = 0.9999908` | `dQ = 0.9999918` | `dK = 0.9999911` | `dV = 0.9999920` | `dBeta = 0.9999931`
* **256-Identical-Token BF16 Padding Torture (`beta=0.85`):** `0` `NaN`s / `0` `Inf`s in forward or backward.
* **Chunked Vocabulary Cross-Entropy (`forward_loss`):** Exact `0.00e+00` absolute diff vs. dense FP32 CE while saving >4 GB VRAM.
* **`VAULTLMv2` (151.33M) Full `L=2048` Training Speed:**
  * `B=2, grad_ckpt=False`: **`3,060.0 tok/s`** (`14.79 GB` peak VRAM)
  * `B=2, grad_ckpt="half"`: **`2,749.3 tok/s`** (**`9.26 GB`** peak VRAM)
  * `B=8, grad_ckpt=True`: **`2,389.8 tok/s`** (`11.01 GB` peak VRAM)

---

## 5. Live `VAULTLMv2` From-Scratch `L = 2048` Training Design & Telemetry

### 5.1 Decontaminated Dataset (`data_v2/audit_report.json` — `284.2M` Pre-Tokenized `L=2048` Tokens)
Built from `SmolLM-Corpus` (`cosmopedia-v2`, `fineweb-edu-dedup`), `SmolTalk` (filtered plain-English prose), `Magicoder-OSS-Instruct-75K`, `CodeFeedback-Filtered-Instruction`, `EvolInstruct-Code-80k`, `MetaMathQA`, `MathInstruct`, procedural polyglot systems (`BuildBit .bb`, `C99`, `C++20`, `Lua`), and multi-span `2K` NIAH (`1,350..1,920` tokens spanning `20..31` chunks with `1..35` token key-value boilerplate gaps):
* **Purged Contamination:** `7,157` code-review items (`"given code"`, `"revised code"`, `"code you provided"`) removed so the model writes code from scratch; `0` overlap with unseen 8-domain evaluation targets.
* **Document-Aware Row Packing (`ROW_LEN = 2049`):** Atomic QA and `2K` NIAH turns never split across row boundaries (preventing orphaned answers whose haystacks fell in the prior row).
* **Per-Token Salience Loss Masking:**
  * Stage 1 Pretrain (`248.37M` train / `3.07M` val tokens): `0.0x` pad, `0.3x` user prompt/haystack, `1.0x` raw textbooks/code (`50.7%` prose/QA, `24.0%` polyglot code, `12.6%` `<think>` math, `12.6%` `2K` NIAH), `1.5x` assistant body, `3.0x` exact needle values + final math answers + `<|im_end|>`.
  * Stage 2 Clean ChatML SFT (`31.96M` train / `0.82M` val tokens): Strict `0.0x` mask on every prompt token right up to `<|im_start|>assistant\n`, mixed with `25%` Stage-1 replay during SFT to prevent catastrophic forgetting.

### 5.2 Schedule & Live Unweighted Holdout Perplexity Curve (`Steps 1 -> 5,900 / 7,500`)
* **Schedule:** Stage 1 Warmup-Stable-Decay (`WSD`, `7,500` steps $\times$ `32,768` tok/step = `245.76M` tokens): linear warmup `300` steps $\to$ stable plateau `6.0e-4` up to Step `5,400` $\to$ cosine decay `6.0e-4 -> 3.0e-5` across Steps `5,400..7,500`; followed by Stage 2 SFT (`750` steps = `24.58M` tokens at `4.5e-5 -> 5.0e-6`). Co-Prime router anneals over Steps `1..2,250`.

| Pretrain Step | `L=2048` Tokens | LR | Unweighted `val_ce` | Holdout `Val PPL` | Grad Norm | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 0.03M | `2.00e-6` | 10.8584 | 51,969.80 | 8.59 | Random initialization (`Step 0`) |
| 100 | 3.28M | `2.00e-4` | 5.9410 | 380.30 | 2.65 | Warmup |
| 500 | 16.38M | `6.00e-4` | 3.7070 | 40.73 | 1.19 | Early WSD plateau |
| 1,000 | 32.77M | `6.00e-4` | 3.1009 | 22.22 | 0.94 | `<think>...</think>` math structure learned |
| 2,000 | 65.54M | `6.00e-4` | 2.7582 | 15.77 | 0.79 | Compilable C++20 `BoundedBuffer<T,128>` & C99 |
| 2,250 | 73.73M | `6.00e-4` | 2.7059 | 14.97 | 0.68 | Router `step_ratio = 1.0` locked (zero loss spike) |
| 3,000 | 98.30M | `6.00e-4` | 2.6278 | 13.84 | 0.65 | Full `BuildBit (.bb)` `soa struct` + `gpu_arena` |
| 4,000 | 131.07M | `6.00e-4` | 2.5498 | 12.80 | 0.69 | Clean Python + BuildBit |
| 5,000 | 163.84M | `6.00e-4` | 2.4777 | 11.91 | 0.60 | Clean Lua metatable OOP (`Vec2.__index`) |
| 5,400 | 176.95M | `6.00e-4` | 2.4451 | 11.53 | 0.70 | End of WSD plateau (`vault_v2_wsd_stable.pt`) |
| **5,900** *(live)* | **193.33M** | **`5.24e-4`** | **2.4035** | **11.06** | **0.62** | **First 500 steps of WSD cosine decay (`-0.47` PPL)** |

---

## 6. Candid Failure Modes, Open Bottlenecks & What Remains Unproven

1. **The "36-Stem Lexical Prior Trap" on Open-Vocabulary `2K` NIAH (Observed Live):**
   * Unseen NIAH evaluation queries `PlasmaGrid-E6` (`CHK-3213`) and `CryoLock-N6` (`CHK-4811`) buried across a `2,000`-token manifest with a `+15`-token boilerplate gap (`[Subsystem 05: PlasmaGrid-E6] Status: NOMINAL | Verification Code: CHK-3213`).
   * At Step `1,600`, the model output `- VoltRail-N6: CHK-3206 / - CryoPump-N6: CHK-4815` (retrieving `32xx` and `48xx` across 14 chunks). However, because our procedural NIAH generator sampled training entity names from a closed list of only **36 stems** (`TRAIN_ENTITY_STEMS`), at constant `LR = 6.0e-4` (Steps `3,600–5,200`) `lm_head` overfit to emitting one of those 36 stems (`- QuantumMux-N6:` instead of copying `- PlasmaGrid-E6:` from the question 20 tokens earlier). Once bullet 1 emits `QuantumMux-N6:`, autoregressive decoding queries the vault for `QuantumMux-N6` rather than `PlasmaGrid-E6`!
   * *Live update during WSD decay:* At Step `5,800` (`LR = 5.50e-4`), the probe copied `- PlasmaGrid-E6:` verbatim for the first time. Still, full `2/2` open-vocabulary NIAH accuracy across `+15..28` token gaps remains **unproven** until WSD decay + Stage 2 SFT complete (and may benefit from open-vocabulary random entity stems in SFT).
2. **No Iso-Compute / Iso-Data `151M` Baseline Race Yet:**
   * While `race_from_scratch.py` proved `VAULT > Transformer > Linear SSM` at `3.8M` params, we have **not yet** trained an identical `151M` `Transformer++` or `Mamba-2` / `Gated DeltaNet` on the exact same `270M`-token `data_v2` memmaps to compare `lm-eval` (`HellaSwag`, `ARC`, `PIQA`, `SciQ`, `HumanEval`) head-to-head.
3. **Zero-Shot Length Extrapolation (`Train @ 2K -> Eval @ 8K–32K`) Not Yet Tested on `VAULTLMv2`:**
   * Chunk-relative RoPE has zero global position drift, and C99/PyTorch synthetic tests hold `100K` recall when redundant writes are gated, but static vault recall dips past ~`4,096` *unique* writes per group (`88 * 64 = 5,632` total rank slots per group) if $\beta_{b,g,t}$ fails to filter boilerplate on real `32K` documents.
4. **Routing & Consensus Design Questions Open for Critique:**
   * `_route_cells_v2` uses hard `argmax` per co-prime bank with unit-forward multiplicative STE (`1 + p - sg(p)`), meaning non-selected cells receive zero direct readout gradient on a given token.
   * Multi-bank consensus weights $w_r \propto (\sigma_r^2 + 0.01)^{-1}$ use deterministic cell occupancy count $n_r$ rather than a content-dependent query-key match confidence or learned residual error estimator.
   * In Layer 0, `CausalShortConv1D(k=4)` spans 4 tokens; binding a key (`PlasmaGrid-E6`) to a value 15 tokens later (`CHK-3213`) relies on Layer 0's `64x128` local RoPE attention fusing the entity into the value's hidden state so Layers `1..15` write the bound pair into their Co-Prime vaults.

---

## 7. Specific Questions We Want the Reviewing AI to Answer
1. **Mathematical & Kernel Audit:** Do you see any subtle flaw in the Cell-Masked Chunkwise Compact WY equivalence ($M_{ij} = \mathbb{I}[i > j \land c_{k,i} = c_{k,j}]$) or the 6-step block-recursive unit lower-triangular inverse under BF16 WMMA?
2. **Router & Multi-Token Binding Critique:** Is relying on Layer 0's `64x128` local attention + Layers `1..15` vaults sufficient for `+15..35` token key-value binding, or would key-value delay convolution / cross-token key modulation strengthen single-layer associative binding?
3. **Router Gradient & Collapse Risk:** Could `argmax` + multiplicative STE (`ste_q`, `ste_k`) suffer bank cell under-utilization on skewed real-world token distributions despite `FusedAPLNorm` center-surround inhibition, and should we log per-bank cell entropy across the 88 cells?
4. **Benchmarking & Publication Roadmap:** If you were a skeptical NeurIPS / ICML reviewer, what exact 3 ablation or baseline experiments on this `151M` model would completely convince you?
