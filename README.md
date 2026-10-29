# VAULT: Variance-Weighted Associative Unitary Lattice Tensor
### `VAULT` & `VAULT v2` (`FlashVAULT v2` True Grouped-Vault Attention) — Post-Transformer, Post-SSM Sequence Modeling

**Author & Architect:** Philip Bartsch (`@batteryphil`)  
**License:** MIT License (`Copyright (c) 2026 Philip Bartsch`)

---

## 🔬 Overview

* **Transformers** rely on an unbounded **KV Cache** ($\mathcal{O}(T)$ memory, $\mathcal{O}(T^2)$ prefill compute).
* **Mamba / Linear RNNs** compress all history into a single dense **Decaying State** ($h_t = A_t h_{t-1} + B_t x_t$), losing exact multi-span retrieval to $\lambda^T$ exponential washout and rank interference.
* **VAULT** (**V**ariance-Weighted **A**ssociative **U**nitary **L**attice **T**ensor) locks novel memories into **isolated Co-Prime Orthogonal Nullspace Vaults** via chunkwise parallel Householder / Delta-Rule projection, where untouched vault cells never decay ($\lambda = 1.0$) and redundant boilerplate is bypassed by a token-selective write/erase gate.

**VAULT** unifies core primitives across Philip Bartsch's (`@batteryphil`) open-source research stack:
1. **Bit-Sliced 1.58-Bit Ternary `{pos_mask, neg_mask}` `POPCNT` Router** ([`buildbit-os`](https://github.com/batteryphil/buildbit-os) & [`SynapseOS`](https://github.com/batteryphil/SynapseOS))
2. **Sparse Connectome Routing + `FusedAPLNorm` Homeostatic Inhibition** ([`BioNPC`](https://github.com/batteryphil/BioNPC))
3. **Subspace-Factored Co-Prime Residue Banks & Prime Reciprocal Harmonics $\{1/p\}$** ([`prime-revisited`](https://github.com/batteryphil/prime-revisited))
4. **`PRIME-Net` Discovered High-Gain Exponential Kernel $\phi(x) = \text{norm}(\exp(\beta \cdot \text{RMSNorm}(x)))$** ([`gist-memory`](https://github.com/batteryphil/gist-memory) & [`PRIME-Net`](https://github.com/batteryphil/PRIME-Net))
5. **Sherman-Morrison / Gated Delta-PRIME Orthogonal Nullspace Vault Cells** ([`PRIME-Moment-Attention`](https://github.com/batteryphil/PRIME-Moment-Attention))
6. **Dyadic 256-Token `GTRM` / `ConceptPerceptron` Multi-Scale Cascade** ([`PRIME-Moment-Attention`](https://github.com/batteryphil/PRIME-Moment-Attention) & [`thalamic-bloom`](https://github.com/batteryphil/thalamic-bloom))
7. **`PRIME-Net` Symbolic Variance-Weighted Harmonic Consensus Readout** ([`PRIME-Net`](https://github.com/batteryphil/PRIME-Net))

---

## 🚀 `VAULT v2` (`VAULTLMv2` + `FlashVAULT v2` Triton 3.8 Kernel)

`VAULT v2` scales Co-Prime Nullspace Vaults into a hardware-optimal **151.3M-parameter** native $\mathcal{O}(L)$ language model (`d_model=768`, `n_layers=16`, `H_q=12`, `G_vault=4`, `d_head=64`, `d_ff=2048`):

### Architectural Innovations in `VAULT v2`
1. **True 3:1 Grouped-Vault Attention (`GVA`) with Zero Global Atomics (`src/kernels/flash_vault_v2_triton.py`):**
   - `H = 12` independent query heads (each with its own learned routing query projection) share `G = 4` `88`-cell Co-Prime vaults ($P = \{17, 19, 23, 29\}$).
   - Cuts recurrent state traffic **3x** (`1.41 MB` vs `4.22 MB` per layer) and eliminates all global atomic contention in both forward and backward passes.
   - Uses a 6-step register/SRAM block-recursive triangular solver (`_invert_unit_lower_tri_64`) for exact chunkwise parallel WY / Delta-Rule updates ($C = 64$).
2. **Token-Selective Write / Erase Gate $\beta_{b,g,t} \in (0.02, 0.96)$ with Analytical `dBeta`:**
   - Dynamically suppresses boilerplate tokens ($\beta \to 0.02$, preserving vault rank) while locking high-salience entities, identifiers, and verification codes into orthogonal nullspace cells ($\beta \to 0.96$).
3. **`CausalShortConv1D(kernel_size=4)` + `SiLU` before `Q, K, V` (`src/vault_v2_model.py`):**
   - Binds multi-token key-value phrases (`"Verification Code: CHK-4811"`) prior to Co-Prime routing so multi-word spans are addressed and retrieved across `+15..35` token syntactic gaps.
4. **Chunk-Relative RoPE + Multi-Scale Geometric Decay ($\gamma_h \in [0.66, 0.992]$) in the $\mathcal{O}(L)$ Local Syntax Branch:**
   - Exact $64 \times 128$ overlapping causal window (`prev_chunk + curr_chunk`) provides crisp positional syntax for indentation, brackets, and multi-digit arithmetic while keeping global complexity strictly $\mathcal{O}(L)$.
5. **Per-Head `RMSNorm` + Sigmoid Output Gate & Chunked Vocabulary Cross-Entropy:**
   - Stabilizes hybrid local + vault fusion (`gnorm ~ 1.0`) and eliminates multi-GB dense logit allocations during full `L = 2048` training (`abs_diff = 0.00e+00` vs dense cross-entropy).

### Verified `FlashVAULT v2` Kernel Parity & Single-GPU Throughput
*Measured on a single consumer 16 GB GPU (`AMD Radeon RX 9060 XT`, ROCm 7.2, Triton 3.8, `tests/test_vault_v2.py`):*

| Verification / Benchmark Suite | Configuration | Result |
| :--- | :--- | :--- |
| **FP32 Forward + Backward Parity** | `B=2, L=200 (non-multiple of 64), H=12, G=4, D=64` | `y=0.9999999`, `dQ=1.0000000`, `dK=0.9999999`, `dV=0.9999999`, `dBeta=1.0000000` |
| **BF16 WMMA Forward + Backward Parity** | `B=2, L=200, H=12, G=4, D=64` | `y=0.9999908`, `dQ=0.9999918`, `dK=0.9999911`, `dV=0.9999920`, `dBeta=0.9999931` |
| **Collinear Degeneracy Torture** | `256` identical BF16 tokens at high write gain ($\beta=0.85$) | **`0` `NaN`s / `0` `Inf`s** (finite forward & backward) |
| **Chunked Vocabulary CE Parity** | `VAULTLMv2.forward_loss` vs dense FP32 CE | **`abs_diff = 0.00e+00`** |
| **`VAULTLMv2` (151.3M) Train Throughput** | `L=2048, B=2, grad_ckpt=False` | **`3,060.0 tok/s`** (`14.79 GB` peak VRAM) |
| **`VAULTLMv2` (151.3M) Train Throughput** | `L=2048, B=2, grad_ckpt="half"` | **`2,749.3 tok/s`** (**`9.26 GB`** peak VRAM) |
| **`VAULTLMv2` (151.3M) Train Throughput** | `L=2048, B=4, grad_ckpt="three_quarter"` | **`2,514.7 tok/s`** (`10.72 GB` peak VRAM) |
| **`VAULTLMv2` (151.3M) Train Throughput** | `L=2048, B=8, grad_ckpt=True` | **`2,389.8 tok/s`** (`11.01 GB` peak VRAM) |

---

## 📊 Foundational Associative Memory Benchmarks (`VAULT` Core)

### 1. 100,000-Token Associative Horizon Benchmark (Native C99 — `c/bench_vault.c`)
*Stream of $T$ intervening tokens (85% Zipfian repeated language stream + 15% novel facts); measuring cosine similarity & exact Top-1 dictionary retrieval of the earliest target keys written at $t = 0$:*

| Horizon $T$ (Tokens) | Linear Attention (`ELU+1`) | Standalone `GistLayer` (`exp`) | Standalone `PRIME-Moment` ($S_0, S_1, S_2$) | Single-Matrix `Dense-RLS` | **VAULT (C99)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | `0.1123` | `0.2356` | `0.0806` | `-0.0073` | **`1.0000` (`100.0%` Top-1)** |
| **500** | `0.0501` | `0.0716` | `0.0443` | `0.0108` | **`1.0000` (`100.0%` Top-1)** |
| **1,000** | `0.0361` | `0.0473` | `0.0313` | `0.0123` | **`1.0000` (`100.0%` Top-1)** |
| **4,000** | `0.0327` | `0.0291` | `0.0319` | `-0.0433` | **`0.9990` (`100.0%` Top-1)** |
| **8,000** | `0.0295` | `0.0309` | `0.0281` | `-0.0386` | **`0.9929` (`100.0%` Top-1)** |
| **16,000** | `0.0228` | `0.0208` | `0.0225` | `0.0019` | **`0.9800` (`100.0%` Top-1)** |
| **32,000** | `0.0310` | `0.0296` | `0.0303` | `-0.0285` | **`0.9731` (`100.0%` Top-1)** |
| **64,000** | `0.0296` | `0.0282` | `0.0290` | `-0.0048` | **`0.9623` (`95.8%` Top-1)** |
| **100,000** | `0.0248` | `0.0244` | `0.0246` | `-0.0146` | **`0.9645` (`100.0%` Top-1)** |

* **C99 Single-Core Throughput:** **`179,565 tok/s`** (`5.569 us/token`)
* **Innovation Gate Efficiency:** **`99.96%`** of redundant Zipfian writes bypassed automatically.

### 2. Adversarial 100% Non-Repeating Stream (`tests/test_vault_rigorous.py`)
*Zero redundancy—every single token is brand-new Gaussian noise, so the Innovation Gate cannot skip any write. Queries the oldest 32 memories buried under $N$ unique facts:*

| Unique Facts $N$ | Linear (`ELU+1`) | Single-Matrix `Dense-RLS` | Replicated 4-Bank ($P=\{17,19,23,29\}$) | **Subspace-Factored VAULT ($P=\{83,89,97,101\}$)** |
| :---: | :---: | :---: | :---: | :---: |
| **32** | `0.1643` | `0.9552` | `1.0000` (`100.0%`) | **`1.0000` (`100.0%` Top-1)** |
| **64** | `0.1227` | `0.3253` | `1.0000` (`100.0%`) | **`1.0000` (`100.0%` Top-1)** |
| **256** | `0.0495` | `0.0113` | `0.9997` (`100.0%`) | **`0.9997` (`100.0%` Top-1)** |
| **512** | `0.0042` | `0.0020` | `0.9933` (`100.0%`) | **`0.9981` (`100.0%` Top-1)** |
| **1,024** | `0.0007` | `0.0057` | `0.9354` (`100.0%`) | **`0.9929` (`100.0%` Top-1)** |
| **2,048** | `0.0013` | `0.0073` | `0.4660` (`50.0%`) | **`0.9587` (`100.0%` Top-1)** |
| **3,072** | `-0.0153` | `0.0184` | `0.1353` (`0.0%`) | **`0.8849` (`100.0%` Top-1)** |

### 3. Dyadic 256-Token GTRM Macro-Cascade (`tests/test_vault_rigorous.py`)
*Streaming 4,096 tokens across 4 distinct Topic Epochs (1,024 tokens per epoch):*

| Historical Topic Epoch | Distance Ago | Flat `GistLayer` Cosine | **VAULT Dyadic 256-Token GTRM Cascade** |
| :--- | :---: | :---: | :---: |
| **Epoch 0 (Oldest)** | `4,096` tokens ago | `-0.2332` (Washed out) | **`0.6087`** |
| **Epoch 1** | `3,072` tokens ago | `0.2496` | **`0.8671`** |
| **Epoch 2** | `2,048` tokens ago | `0.2091` | **`0.8676`** |
| **Epoch 3 (Most Recent)** | `1,024` tokens ago | `0.9781` | **`0.9831`** |

---

## 📂 Repository Structure

```text
VAULT/
├── c/                                    # Native C99 VAULT engine & 100K-token benchmarks
│   ├── vault.h / vault.c                 # Core Co-Prime Nullspace Vault C99 implementation
│   ├── bench_vault.c                     # 100,000-token head-to-head horizon benchmark
│   └── torture_vault.c                   # Collinear & rank-saturation C99 torture suite
├── src/
│   ├── kernels/
│   │   ├── flash_vault_v2_triton.py      # FlashVAULT v2: True 3:1 GVA Triton 3.8 kernel (zero atomics)
│   │   └── flash_vault_triton.py         # FlashVAULT v1: Chunkwise WY Triton kernel
│   ├── vault_v2_model.py                 # VAULTLMv2 (151.3M): ShortConv1D, RoPE local, GVA, chunked CE
│   ├── vault_lm_150m.py                  # VAULT-150M v1 backbone & shared RMSNorm/router primitives
│   ├── flash_vault_module.py             # FlashVAULT v1 PyTorch wrapper
│   ├── vault.py                          # PyTorch reference VAULT associative memory & Dyadic GTRM
│   └── vault_surgery.py                  # Transformer-to-VAULT attention surgery utilities
├── tests/
│   ├── test_vault_v2.py                  # FP32/BF16 parity, 256-token torture, chunked CE & VRAM suite
│   ├── test_flash_vault_grad.py          # FlashVAULT v1 gradient verification
│   └── test_vault_rigorous.py            # Non-repeating capacity & Dyadic GTRM falsification tests
├── benchmarks/
│   ├── eval_vault_v2_standalone.py       # 8-domain unseen benchmark + 2K..32K O(L) scaling audit
│   └── *.py / *.json                     # Empirical benchmark scripts & recorded JSON audits
├── data_v2/
│   └── audit_report.json                 # Decontamination & token distribution audit (284.2M L=2048 tokens)
├── prepare_vault_v2_data.py              # Decontaminated L=2048 NVMe memmap packer (WSD + ChatML SFT)
├── train_vault_v2.py                     # Full L=2048 WSD Pretraining + Clean ChatML SFT engine
├── Makefile                              # Build, verification, data prep, training & eval targets
├── requirements.txt                      # Python dependencies
└── LICENSE                               # MIT License — Copyright (c) 2026 Philip Bartsch (batteryphil)
```

---

## ⚡ Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# 1. Verify FlashVAULT v2 Triton 3.8 kernel parity, torture stability, and L=2048 throughput
make test-v2

# 2. Build decontaminated L=2048 NVMe memmap shards (248.4M pretrain + 32.0M ChatML SFT tokens)
make data-v2

# 3. Train VAULTLMv2 (151.3M) from scratch (Stage 1 WSD @ L=2048 -> Stage 2 Clean ChatML SFT)
make train-v2

# 4. Run standalone 8-domain unseen evaluation + 2K..32K O(L) prefill scaling audit
make eval-v2

# 5. Run foundational C99 & PyTorch associative memory falsification tests
make test
```
