# VAULT: Variance-Weighted Associative Unitary Lattice Tensor
### `VAULT v2.5` (`180.6M` Co-Prime Kenyon Cell LM & Zero-`rocFFT` $\mathcal{O}(1)$ Streaming Engine) — Post-Transformer, Post-SSM Sequence Modeling

**Author & Architect:** Philip Bartsch ([`@batteryphil`](https://github.com/batteryphil))  
**Pretrained & SFT Weights (HuggingFace Hub):** [**`batteryphil/VAULT-v2.5-180M`**](https://huggingface.co/batteryphil/VAULT-v2.5-180M)  
**License:** Dual License — [**GNU AGPL-3.0**](LICENSE-AGPL-3.0) (Free / Non-Profit / Open-Source) **or** [**Commercial License**](LICENSE-COMMERCIAL.md) (Closed-Source Business Use) · `Copyright (c) 2026 Philip Bartsch`

---

## ⚡ Download & Run in 30 Seconds

### Option A — 5 Lines of Python (`transformers`, Zero Repo Clone Needed)
```bash
pip install torch transformers safetensors huggingface_hub
# Optional on Linux CUDA / ROCm for 48.8 tok/s hardware-fused O(1) Triton kernels:
# pip install triton
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("batteryphil/VAULT-v2.5-180M", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "batteryphil/VAULT-v2.5-180M", trust_remote_code=True
).to("cuda" if __import__("torch").cuda.is_available() else "cpu")

print(model.chat(tok, "Explain why recursion needs a base case in programming.", stream=True))
```

**Zero-Shot Elastic Co-Prime Mitosis (`220 -> 2,002` Kenyon Cells in 1 Line):**
Expand associative fast-weight capacity **9.1x** (`8.97M -> 62.7B` Chinese Remainder Theorem addresses) at inference time for `32K+` streaming retrieval with **zero change in per-token decode latency (`20.59 ms/tok`)**:
```python
model_2002c = model.expand_mitosis(2002)  # Presets: 220, 508, 942, 2002
reply, stats = model_2002c.chat(
    tok,
    "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
    return_stats=True,
)
print(reply, "\n", stats)
```

### Option B — 1-Command CLI & Interactive Streaming REPL
`run_vault.py` automatically downloads weights from [`batteryphil/VAULT-v2.5-180M`](https://huggingface.co/batteryphil/VAULT-v2.5-180M) if local `./checkpoints/` are not present, and falls back gracefully to pure PyTorch on macOS / Windows / CPU:
```bash
git clone https://github.com/batteryphil/VAULT.git
cd VAULT
pip install -r requirements.txt

# 1. Run the 4-domain live streaming showcase (Python, CS prose, <think> CoT, 8-subsystem NIAH):
python run_vault.py --demo

# 2. Interactive streaming Chat REPL (supports /mitosis 2002, /demo, /temp <float>):
python run_vault.py

# 3. Single prompt (or Pre-SFT base model via --variant base --raw):
python run_vault.py --cells 2002 --prompt "Write a C++20 RAII file descriptor wrapper."
```

### 💻 Cross-Vendor Hardware Support (`NVIDIA CUDA` • `AMD ROCm` • `Apple Silicon` • `CPU`)
* **NVIDIA CUDA GPUs (`Ampere` RTX 30xx / A100, `Ada` RTX 40xx / L4, `Hopper` H100, `Blackwell` RTX 50xx):** Full native acceleration via vendor-neutral **Triton PTX TensorCore** chunkwise prefill (`tile_p <= 64`, `<= 32 KB` shared memory across all `220c..2,002c` Mitosis presets), fused register-DFT Triton decode kernels, and `torch.cuda.CUDAGraph`.
* **Pre-Ampere NVIDIA GPUs (`Turing` Colab T4 / RTX 20xx, `Volta` V100):** Automatic `FP32` Triton / `FP16` fallback when hardware `bfloat16` TensorCores (`SM < 8.0`) are unavailable.
* **AMD ROCm GPUs (`RDNA3` / `RDNA4` / `CDNA`):** Full native Triton `BF16` WMMA + HIPGraph acceleration.
* **Apple Silicon (`mps`) & `CPU`:** Zero-dependency vectorized Compact-WY PyTorch fallback (`run_vault.py` auto-detects `cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`).

---

## 🏆 `VAULT v2.5` (`180.6M`) — Architecture, $\mathcal{O}(1)$ Streaming & Empirical Benchmarks

Trained from scratch on a **single consumer 16 GB GPU** (`AMD Radeon RX 9060 XT`) on **`0.594B` total tokens** (`0.575B` base continual pretraining + `19.7M` precision SFT tokens), `VAULT v2.5` pairs **4 Co-Prime Kenyon Cell banks** ($P = \{47, 53, 59, 61\} = 220$ cells) with **Tri-Horizon Salience Phase Binding** (`g0` local + `g1` phrase + `g2` wide `m=1..10` subword span binder), **Zero-Decay Unaddressed Cells ($\gamma = 1.0$)**, and a **Zero-`rocFFT` Fused Triton Decode Engine** (`_vault_v2_5_fft_bind_64_kernel` + CUDAGraph/HIPGraph).

### 1. True $\mathcal{O}(1)$ Streaming Decode Latency & Mitosis Scaling (`2K -> 128K`)
*Measured on `AMD Radeon RX 9060 XT` (ROCm 7.2, BF16/FP32). Step-wise streaming matches monolithic FP32 forward across chunk rollovers (`t=0..255`) with `5.91e-5` prefill max diff, `2.46e-5` decode median diff, and **`100.00%` Top-1 token agreement**:*

| Bank Configuration | Context Length | Prefill Throughput | Persistent State | $\mathcal{O}(1)$ Decode Latency | Decode Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`220c` Default** ($P=\{47,53,59,61\}$) | `2K` (`2,048`) | `10,214 tok/s` | `227.3 MB` | **`20.61 ms/tok`** | **`48.5 tok/s`** |
| **`220c` Default** ($P=\{47,53,59,61\}$) | `32K` (`32,768`) | `13,077 tok/s` | `227.3 MB` | **`20.55 ms/tok`** | **`48.7 tok/s`** |
| **`220c` Default** ($P=\{47,53,59,61\}$) | `128K` (`131,072`) | `13,119 tok/s` | `227.3 MB` | **`20.53 ms/tok`** | **`48.7 tok/s`** |
| **`2,002c` Zero-Shot Mitosis** ($P=\{491,499,503,509\}$) | `32K` (`32,768`) | `2,803 tok/s` | `2,009.7 MB` | **`20.58 ms/tok`** | **`48.6 tok/s`** |
| **`2,002c` Zero-Shot Mitosis** ($P=\{491,499,503,509\}$) | `128K` (`131,072`) | `2,798 tok/s` | `2,009.7 MB` | **`20.59 ms/tok`** | **`48.6 tok/s`** |

### 2. Zero-Shot Academic Suite (`0.594B` Tokens vs. `100B–600B` External Baselines)
*Full score sheets and verbatim outputs are archived in [`benchmarks/vault_scores_audit.txt`](benchmarks/vault_scores_audit.txt) and [`benchmarks/comparison_v2_4_step10k_vs_v2_5_sft.json`](benchmarks/comparison_v2_4_step10k_vs_v2_5_sft.json):*

| Benchmark / Metric | VAULT v2.4 Base (`0.575B` tok) | **VAULT v2.5 SFT (`0.594B` tok)** | SFT Delta | Reference: Mamba-2 130M (`300B` tok) | Reference: SmolLM 135M (`600B` tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SciQ w/ Passage (`acc_norm`)** | `50.40%` | **`71.80%`** | **`+21.40%`** | `73.20%` (`81.90%` raw) | `76.40%` (`83.20%` raw) |
| **SciQ Closed-Book (`acc_norm`)** | `37.00%` | **`50.70%`** | **`+13.70%`** | `46.70%` | `64.70%` |
| **SciQ In-Context Working-Memory Gain** | `+13.40%` | **`+21.10%`** | **`+7.70%`** | `+26.50%` | `+11.70%` |
| **ARC-Easy (`acc_norm`)** | `37.71%` (`41.37%` raw) | **`45.37%` (`46.34%` raw)** | **`+7.66%`** | `42.38%` | `57.41%` |
| **ARC-Challenge (`acc_norm`)** | `23.12%` | **`23.72%`** | **`+0.60%`** | `23.81%` | `28.92%` |
| **PIQA (`acc_norm`)** | `57.02%` (`58.92%` raw) | **`58.22%` (`59.09%` raw)** | **`+1.20%`** | `63.71%` | `68.44%` |
| **HellaSwag (`acc_norm`)** | `35.73%` | **`37.13%`** | **`+1.40%`** | `43.07%` | `45.60%` |
| **Winogrande (`acc`)** | `49.64%` | **`50.43%`** | **`+0.79%`** | `53.28%` | `52.72%` |
| **BoolQ (`acc` / `acc_norm`)** | `56.93%` / `62.93%` | `41.53%` / `51.33%` | `-15.40%` | `56.33%` / `61.40%` | `59.87%` / `63.20%` |
| **LAMBADA (`acc`)** | `12.27%` | **`13.73%`** | **`+1.46%`** | `42.93%` | `36.87%` |
| **WikiText-2 (`BPB` / `TokPPL`)** | `1.5447` / `93.97` | **`1.5466` / `94.51`** | `+0.0019` BPB | `0.9645` / `19.98` | `0.9686` / `17.26` |
| **9-Task Zero-Shot Average** | `39.98%` | **`43.63%`** | **`+3.65%`** | `49.49%` | `54.55%` |

### 3. Multi-Needle Retrieval & `16x` Zero-Shot Context Extrapolation (`Trained Strictly at 2K`)
* **Controlled `2K` Syntactic Gaps:** **`100.0%`** (`+3` tok), **`93.8%`** (`+15` tok boilerplate), **`87.5%`** (`+24` tok wide).
* **Adversarial `>=8`-Subword Sibling Collisions (`>=7` identical prefix BPE tokens):** **`81.2%`** at `2K ChatML` (`+6.2%` vs v2.4) and **`75.0%`** at `4K ChatML` (`+16.7%` vs v2.4, `-0.70` bits NLL via Tri-Horizon `g2` span binding).
* **`32K` Zero-Shot Co-Prime Mitosis (`2,002c`, `16x` beyond `2K` training context):** **`83.3%` exact needle recall (`91.7%` digit accuracy)** in 1-shot streaming and **`66.7%` exact (`83.3%` digit accuracy)** in zero-shot ChatML without a single long-context training step.
* **Honest Compute Wall (`0.594B` Tokens):** Closed-book world trivia (`LAMBADA`, open-domain `WikiText-2` PPL) remains strictly token-budget bound at `<0.6B` tokens (`~1/500th` of a 300B-token run), and `64K–128K` exact string readout requires rescaling the Leaky-EMA horizon / short post-mitosis continual adaptation.

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
├── configuration_vault.py                # HuggingFace AutoConfig (`VAULTConfig`, Mitosis presets 220..2002)
├── modeling_vault.py                     # HuggingFace AutoModelForCausalLM (`VAULTForCausalLM`, chat, mitosis)
├── run_vault.py                          # 1-command runner, 4-domain live demo & streaming Chat REPL
├── export_and_upload_hf.py               # Self-contained HuggingFace Hub + safetensors exporter & clean-room test
├── c/                                    # Native C99 VAULT engine & 100K-token benchmarks
│   ├── vault.h / vault.c                 # Core Co-Prime Nullspace Vault C99 implementation
│   ├── bench_vault.c                     # 100,000-token head-to-head horizon benchmark
│   └── torture_vault.c                   # Collinear & rank-saturation C99 torture suite
├── src/
│   ├── kernels/
│   │   ├── flash_vault_v2_triton.py      # FlashVAULT v2: True 3:1 GVA + O(1) decode Triton kernels (CPU fallback)
│   │   └── flash_vault_triton.py         # FlashVAULT v1: Chunkwise WY Triton kernel
│   ├── vault_v2_5_model.py               # VAULTLMv2_5 (180.6M): Co-Prime Kenyon Cells, Tri-Horizon Binder & Mitosis
│   ├── vault_v2_5_streaming.py           # Zero-rocFFT O(1) streaming engine (fused radix-8 FFT + CUDAGraph decode)
│   ├── vault_v2_streaming.py             # VAULT v2.4 O(1) streaming engine & shared decode cache primitives
│   ├── vault_v2_model.py                 # VAULTLMv2 (151.3M..180.5M): ShortConv1D, RoPE local, GVA, chunked CE
│   ├── vault_lm_150m.py                  # VAULT-150M v1 backbone & shared RMSNorm/router primitives
│   ├── flash_vault_module.py             # FlashVAULT v1 PyTorch wrapper
│   ├── vault.py                          # PyTorch reference VAULT associative memory & Dyadic GTRM
│   └── vault_surgery.py                  # Transformer-to-VAULT attention surgery utilities
├── tests/
│   ├── test_vault_v2_5_cpu.py            # VAULT v2.5 monolithic vs O(1) streaming parity & Zero-rocFFT audit
│   ├── test_flash_vault_v2_invariants.py # FlashVAULT v2 mathematical & numerical invariant suite
│   ├── test_vault_v2.py                  # FP32/BF16 parity, 256-token torture, chunked CE & VRAM suite
│   ├── test_flash_vault_grad.py          # FlashVAULT v1 gradient verification
│   └── test_vault_rigorous.py            # Non-repeating capacity & Dyadic GTRM falsification tests
├── benchmarks/
│   ├── vault_scores_audit.txt            # Complete plain-text score sheet & verbatim outputs (v2.4 10k vs v2.5)
│   ├── comparison_v2_4_step10k_vs_v2_5_sft.json # Head-to-head Pre-SFT vs Post-SFT JSON report
│   ├── bench_vault_v2_5_post10k_suite.py # Full O(1) latency, 128K Mitosis, sibling NIAH & 8-domain suite
│   ├── eval_vault_v2_standalone.py       # 8-domain unseen benchmark + 2K..32K O(L) scaling audit
│   └── *.py / *.json                     # Empirical benchmark scripts & recorded JSON audits
├── data_v2/
│   └── audit_report.json                 # Decontamination & token distribution audit (284.2M L=2048 tokens)
├── prepare_vault_v2_data.py              # Decontaminated L=2048 NVMe memmap packer (WSD + ChatML SFT)
├── prepare_vault_v2_5_sft_data.py        # Precision 4.92M-token SFT dataset builder (Python/Lua/prose/sibling NIAH)
├── train_vault_v2.py                     # Stage 1 WSD Pretraining + Clean ChatML SFT engine
├── train_vault_v2_4.py                   # Stage 2 Co-Prime Kenyon Cell continual pretraining (to Step 10,000)
├── train_vault_v2_5_sft.py               # Stage 3 VAULT v2.5 precision SFT trainer (EMA + Val Gate)
├── Makefile                              # Build, demo, REPL, verification, data prep, training & eval targets
├── requirements.txt                      # Cross-platform Python dependencies
├── LICENSE                               # Dual License summary + full GNU AGPL-3.0 text
├── LICENSE-AGPL-3.0                      # Verbatim GNU Affero General Public License v3.0
└── LICENSE-COMMERCIAL.md                 # Commercial License terms & contact details
```

---

## 🛠️ Make Targets & Developer Workflow

```bash
# Install Python dependencies
pip install -r requirements.txt

# 1. Auto-download VAULT v2.5 from HuggingFace Hub & run 4-domain live streaming demo
make demo

# 2. Launch interactive streaming Chat REPL (with live /mitosis 2002 switching)
make run

# 3. Verify VAULT v2.5 O(1) Zero-rocFFT streaming parity & FlashVAULT v2 Triton kernels
make test-v2-5
make test-v2

# 4. Run full VAULT v2.5 Post-10k benchmark suite (2K..128K O(1) scaling, NIAH, 9 zero-shot tasks)
make bench-v2-5

# 5. Reproduce training pipeline from scratch (Data -> WSD -> v2.4 Continual -> v2.5 SFT)
make data-v2
make train-v2

# 6. Run foundational C99 & PyTorch associative memory falsification tests
make test
```

---

## ⚖️ Dual License (`AGPL-3.0-or-later` OR Commercial License)

Copyright (c) 2026 **Philip Bartsch** ([`@batteryphil`](https://github.com/batteryphil)). All rights reserved.

This repository is **dual-licensed** under:

1. **[GNU Affero General Public License v3.0 (`AGPL-3.0-or-later`)](LICENSE-AGPL-3.0)** — **Free & Open Source:**
   - Free for **non-profit organizations, universities, academic researchers, students, educators, and personal projects**.
   - Free for **open-source projects and businesses** that release the complete source code of any software, derivative work, or network/API/SaaS service using VAULT under **GNU AGPL-3.0** (including Section 13 network-interaction obligations).
2. **[VAULT Commercial License](LICENSE-COMMERCIAL.md)** — **Closed-Source Business Use:**
   - Required for **for-profit businesses and commercial entities** wishing to embed, integrate, distribute, or host VAULT in proprietary/closed-source software, hardware, internal enterprise deployments, APIs, or SaaS offerings **without** open-sourcing their own software under AGPL-3.0.
   - For commercial licensing inquiries, see [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md) or contact `@batteryphil` at [`https://github.com/batteryphil`](https://github.com/batteryphil).

