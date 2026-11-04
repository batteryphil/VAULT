#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Exports a 100% self-contained HuggingFace `transformers` (`trust_remote_code=True`) + `safetensors`
# model repository for `VAULT v2.5` (`batteryphil/VAULT-v2.5-180M`), verifies clean-room loading with
# `PYTHONPATH=""` outside the repo, and uploads to HuggingFace Hub.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configuration_vault import VAULTConfig
from src.vault_v2_5_model import _unwrap_checkpoint_state_dict

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

HF_MODEL_CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - vault
  - post-transformer
  - recurrent
  - linear-attention
  - state-space-model
  - triton
  - long-context
  - kenyon-cells
  - co-prime-mitosis
---

# VAULT v2.5 (180.6M) — Co-Prime Kenyon Cell Architecture & Zero-`rocFFT` O(1) Streaming Engine

**Author & Architect:** Philip Bartsch ([`@batteryphil`](https://github.com/batteryphil))  
**GitHub Repository:** [`https://github.com/batteryphil/VAULT`](https://github.com/batteryphil/VAULT)  
**License:** MIT  

**VAULT** (**V**ariance-Weighted **A**ssociative **U**nitary **L**attice **T**ensor) is a 180.6M-parameter post-Transformer, post-SSM recurrent sequence architecture (`d_model=768`, `n_layers=16`, `H_q=12`, `G_vault=4`, `d_head=64`, `d_ff=2048`) trained from scratch on a single consumer 16 GB GPU (`AMD Radeon RX 9060 XT`) on **`0.594B` total tokens** (`0.575B` base continual pretraining + `19.7M` precision SFT tokens).

Instead of an unbounded $\\mathcal{O}(T)$ KV-cache (Transformers) or a single exponentially decaying state matrix (Mamba / linear RNNs), VAULT routes high-salience entities into **4 Co-Prime Kenyon Cell fast-weight banks** ($P = \\{47, 53, 59, 61\\} = 220$ cells, $8.97\\text{M}$ CRT addresses) via phase-bound circular convolution, where **unaddressed memory cells have zero passive temporal decay ($\\gamma = 1.0$)** and capacity can be expanded **zero-shot at inference time** up to **$2,002$ cells ($P = \\{491, 499, 503, 509\\}$, $62.7\\text{B}$ CRT addresses)** via **Elastic Co-Prime Mitosis** without changing per-token decode latency.

---

## ⚡ Super-Quick Start (5 Lines — Zero Repo Clone Needed)

```bash
pip install torch transformers safetensors huggingface_hub
# Optional on Linux CUDA / ROCm for 48.8 tok/s hardware-fused O(1) kernels:
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

### Zero-Shot Elastic Co-Prime Mitosis (`220 -> 2,002` Cells in 1 Line)
Expand the associative memory state **9.1x** (`220 -> 2,002` Kenyon cells) at inference time for `32K+` streaming retrieval (`83.3%` exact needle recall at `32K`, trained strictly at `2K`!) with **identical `20.59 ms/tok` O(1) decode latency**:

```python
model_2002c = model.expand_mitosis(2002)  # Supports 220, 508, 942, or 2002 cells
reply, stats = model_2002c.chat(
    tok,
    "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
    return_stats=True,
)
print(reply)
print(stats)
```

### One-Command CLI & Interactive REPL (via GitHub Repo)
```bash
git clone https://github.com/batteryphil/VAULT.git
cd VAULT
pip install -r requirements.txt

# Auto-downloads weights from HuggingFace Hub & runs 4 live streaming demos:
python run_vault.py --demo

# Interactive streaming Chat REPL (supports /mitosis 2002, /demo, /temp):
python run_vault.py
```

### Hardware Compatibility (`NVIDIA CUDA` • `AMD ROCm` • `Apple Silicon` • `CPU`)
- **NVIDIA CUDA (`RTX 30xx/40xx/50xx`, `L4`, `A100`, `H100`):** Native OpenAI/NVIDIA **Triton PTX TensorCore** chunkwise prefill (`<= 32 KB` shared memory across all `220c..2,002c` Mitosis presets), zero-external-FFT register-DFT Triton decode kernels, and `torch.cuda.CUDAGraph`.
- **Older NVIDIA GPUs (`T4`, `RTX 20xx`, `V100`):** Automatic `FP32` Triton & `FP16` support on `SM < 8.0`.
- **AMD ROCm (`RDNA3/RDNA4/CDNA`):** Native Triton `BF16` WMMA + HIPGraph acceleration.
- **Apple Silicon (`mps`) & `CPU`:** Automatic pure-PyTorch Compact-WY fallback when Triton/CUDA is not present.

---

## 📊 Empirical Benchmarks (`VAULT v2.4 @ Step 10k` Base vs. `VAULT v2.5` SFT)

*All evaluations run offline under identical EleutherAI `lm-eval` and streaming protocols on a single 16 GB AMD Radeon RX 9060 XT:*

### 1. Zero-Shot Academic Suite (`0.594B` Tokens vs. `100B–600B` External Baselines)
| Benchmark / Metric | VAULT v2.4 Base (`0.575B` tok) | **VAULT v2.5 SFT (`0.594B` tok)** | Delta (SFT vs Base) | Reference: Mamba-2 130M (`300B` tok) | Reference: SmolLM 135M (`600B` tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SciQ w/ Passage (`acc_norm`)** | `50.40%` | **`71.80%`** | **`+21.40%`** | `73.20%` | `76.40%` |
| **SciQ Closed-Book (`acc_norm`)** | `37.00%` | **`50.70%`** | **`+13.70%`** | `46.70%` | `64.70%` |
| **SciQ In-Context Gain** | `+13.40%` | **`+21.10%`** | **`+7.70%`** | `+26.50%` | `+11.70%` |
| **ARC-Easy (`acc_norm`)** | `37.71%` (`41.37%` raw) | **`45.37%` (`46.34%` raw)** | **`+7.66%`** | `42.38%` | `57.41%` |
| **ARC-Challenge (`acc_norm`)** | `23.12%` | **`23.72%`** | **`+0.60%`** | `23.81%` | `28.92%` |
| **PIQA (`acc_norm`)** | `57.02%` (`58.92%` raw) | **`58.22%` (`59.09%` raw)** | **`+1.20%`** | `63.71%` | `68.44%` |
| **HellaSwag (`acc_norm`)** | `35.73%` | **`37.13%`** | **`+1.40%`** | `43.07%` | `45.60%` |
| **Winogrande (`acc`)** | `49.64%` | **`50.43%`** | **`+0.79%`** | `53.28%` | `52.72%` |
| **BoolQ (`acc` / `acc_norm`)** | `56.93%` / `62.93%` | `41.53%` / `51.33%` | `-15.40%` | `56.33%` / `61.40%` | `59.87%` / `63.20%` |
| **LAMBADA (`acc`)** | `12.27%` | **`13.73%`** | **`+1.46%`** | `42.93%` | `36.87%` |
| **WikiText-2 (`BPB` / `TokPPL`)** | `1.5447` / `93.97` | **`1.5466` / `94.51`** | `+0.0019` BPB | `0.9645` / `19.98` | `0.9686` / `17.26` |
| **9-Task Zero-Shot Average** | `39.98%` | **`43.63%`** | **`+3.65%`** | `49.49%` | `54.55%` |

### 2. True O(1) Streaming Decode Latency & State Scaling (`2K -> 128K` & `220c -> 2,002c`)
*Zero-`rocFFT` Fused Triton Decode (`_vault_v2_5_fft_bind_64_kernel` + `_vault_v2_5_salience_pool_decode_kernel` + `_vault_v2_5_local_syntax_decode_kernel` + `_vault_v2_decode_fw_kernel`):*
- **FP32 Monolithic vs. Streaming Parity (`t = 0..255` across rollovers):** Prefill max diff `5.91e-5`, 128-step decode median diff `2.46e-5`, **`100.00%` Top-1 token agreement**.
- **`220c` @ `2K` (`2,048` tok):** `10,214 tok/s` prefill | `227.3 MB` state | **`20.61 ms/tok` (`48.5 tok/s`)**
- **`220c` @ `128K` (`131,072` tok):** `13,119 tok/s` prefill | `227.3 MB` state | **`20.53 ms/tok` (`48.7 tok/s`)**
- **`2,002c` Mitosis @ `128K` (`131,072` tok):** `2,798 tok/s` prefill | `2,009.7 MB` state | **`20.59 ms/tok` (`48.6 tok/s`)**

### 3. In-Context Multi-Needle Retrieval & `16x` Zero-Shot Extrapolation (`Trained at 2K Only`)
- **Controlled `2K` Syntactic Gaps:** `100.0%` (`+3` tok), `93.8%` (`+15` tok boilerplate), `87.5%` (`+24` tok wide).
- **Adversarial `>=8`-Subword Sibling Collision Stress Test (`>=7` shared prefix BPE tokens):**
  - `2K ChatML`: **`81.2%`** (`+6.2%` vs v2.4)
  - `4K ChatML`: **`75.0%`** (`+16.7%` vs v2.4, `-0.70` bits NLL via Tri-Horizon `g2` `m=1..10` binder)
  - `4K 1-Shot`: **`75.0%`** (`+8.3%` vs v2.4, `-0.64` bits NLL)
- **`32K` Zero-Shot Co-Prime Mitosis (`2,002c`, `16x` beyond `2K` training horizon):** **`83.3%` exact (`1-shot`, `91.7%` digit acc)** / **`66.7%` exact (`ChatML`, `83.3%` digit acc)**.

---

## 🔍 Honest Capabilities & Known Compute Limits
- **Strengths:** Strong in-context reading comprehension (`71.80%` `SciQ +Passage`), exact multi-needle key-value routing, `16x` zero-shot length extrapolation to `32K` via Co-Prime Mitosis, clean Python/C++20/CoT/CS-prose generation, and flat `O(1)` memory & latency up to `128K`.
- **Compute Limits (`0.594B` Tokens):** Closed-book trivia and rare fiction idioms (`LAMBADA`, raw `WikiText-2` PPL) remain strictly compute-bound by training on `<0.6B` tokens (`~1/500th` of a 300B-token run). At `64K–128K`, zero-shot `2,002c` mitosis lowers target NLL by `-2.7b`, retrieves the leading digit (`33%–50%`), and requires post-mitosis continual adaptation to orthogonalize daughter Voronoi cones past `occ > 0.90`.

## 📦 Included Checkpoints in This Repo
- `model.safetensors`: **`VAULT v2.5` SFT** (`180.58M` params, promoted `ema_step_600`, lossless FP32) — loaded automatically by `AutoModelForCausalLM`.
- `base/vault_v2_4_step10k.safetensors`: **`VAULT v2.4` Step 10,000 Base** (`180.51M` params, Pre-SFT checkpoint, lossless FP32) — loadable via `python run_vault.py --variant base`.
"""


HF_MODULE_MAP = {
    "src.kernels.flash_vault_v2_triton": "hf_flash_vault_v2_triton",
    "src.vault_lm_150m": "hf_vault_lm_150m",
    "src.vault_v2_model": "hf_vault_v2_model",
    "src.vault_v2_streaming": "hf_vault_v2_streaming",
    "src.vault_v2_5_model": "hf_vault_v2_5_model",
    "src.vault_v2_5_streaming": "hf_vault_v2_5_streaming",
}


MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 Philip Bartsch (batteryphil)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _normalize_hf_mit_header(code: str) -> str:
    return code.replace(
        "# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial\n"
        "# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License\n",
        "# MIT License\n",
    )


def _rewrite_for_hf_flat(src_path: str, dst_path: str) -> None:
    import ast

    with open(src_path, "r", encoding="utf-8") as f:
        code = _normalize_hf_mit_header(f.read())
    # Guard legacy v1 optional import inside vault_lm_150m.py (never called by v2.5)
    code = code.replace(
        "from src.flash_vault_module import FlashVAULTAttention",
        "raise RuntimeError('Legacy v1 FlashVAULTAttention is not used in VAULT v2.5')",
    )
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in HF_MODULE_MAP:
            assert node.end_lineno is not None
            targets.append((node.lineno, node.end_lineno, node.col_offset, node.module, HF_MODULE_MAP[node.module]))
    # Replace bottom-up so line numbers remain valid
    for start_ln, end_ln, col, old_mod, new_mod in sorted(targets, key=lambda t: t[0], reverse=True):
        orig_chunk = "".join(lines[start_ln - 1 : end_ln])
        indent = " " * col
        stripped_lines = [
            (ln[col:] if ln.startswith(indent) else ln) for ln in orig_chunk.splitlines(keepends=True)
        ]
        stripped_text = "".join(stripped_lines)
        rel_text = re.sub(
            rf"^from\s+{re.escape(old_mod)}\s+import",
            f"from .{new_mod} import",
            stripped_text,
            count=1,
        )
        flat_text = re.sub(
            rf"^from\s+{re.escape(old_mod)}\s+import",
            f"from {new_mod} import",
            stripped_text,
            count=1,
        )
        rel_indented = "".join((f"{indent}    {ln}" if ln.strip() else ln) for ln in rel_text.splitlines(keepends=True))
        flat_indented = "".join(
            (f"{indent}    {ln}" if ln.strip() else ln) for ln in flat_text.splitlines(keepends=True)
        )
        if not rel_indented.endswith("\n"):
            rel_indented += "\n"
        if not flat_indented.endswith("\n"):
            flat_indented += "\n"
        replacement = f"{indent}try:\n{rel_indented}{indent}except ImportError:\n{flat_indented}"
        lines[start_ln - 1 : end_ln] = [replacement]
    final_code = "".join(lines)
    ast.parse(final_code)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(final_code)


def _save_clean_safetensors(ckpt_pt_path: str, dst_safetensors_path: str, clone_lm_head: bool = True) -> Dict:
    raw = torch.load(ckpt_pt_path, map_location="cpu", weights_only=False)
    sd = _unwrap_checkpoint_state_dict(raw)
    tensors = {}
    for k, v in sd.items():
        tensors[k] = v.detach().cpu().contiguous().clone()
    if clone_lm_head and "lm_head.weight" not in tensors and "embed_tokens.weight" in tensors:
        tensors["lm_head.weight"] = tensors["embed_tokens.weight"].clone()
    os.makedirs(os.path.dirname(os.path.abspath(dst_safetensors_path)), exist_ok=True)
    save_file(tensors, dst_safetensors_path, metadata={"format": "pt"})
    meta = {k: v for k, v in raw.items() if k != "model_state" and isinstance(v, (int, float, str, list, tuple, dict))}
    return meta


def build_hf_export_dir(out_dir: str) -> None:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/5] Saving bundled SmolLM BPE tokenizer + ChatML template -> {out_dir}")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    tok.chat_template = CHATML_TEMPLATE
    tok.save_pretrained(out_dir)
    tc_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        tc.pop("local_files_only", None)
        tc.pop("is_local", None)
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2)

    print("[2/5] Exporting lossless `model.safetensors` (VAULT v2.5 SFT) & `base/vault_v2_4_step10k.safetensors`...")
    v25_pt = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_5_best.pt")
    v24_pt = os.path.join(ROOT_DIR, "checkpoints", "vault_v2_4_step10000_best.pt")
    v25_meta = _save_clean_safetensors(v25_pt, os.path.join(out_dir, "model.safetensors"), clone_lm_head=True)
    v24_meta = _save_clean_safetensors(
        v24_pt, os.path.join(out_dir, "base", "vault_v2_4_step10k.safetensors"), clone_lm_head=True
    )

    print("[3/5] Writing `config.json`, `generation_config.json`, and standalone `trust_remote_code` modules...")
    cfg = VAULTConfig(
        vocab_size=len(tok),
        prime_banks=list(v25_meta.get("prime_banks", [47, 53, 59, 61])),
        ema_alpha=float(v25_meta.get("ema_alpha", 2.0**-11)),
        occ_alpha=0.5,
        tie_word_embeddings=False,
    )
    cfg.auto_map = {
        "AutoConfig": "configuration_vault.VAULTConfig",
        "AutoModelForCausalLM": "modeling_vault.VAULTForCausalLM",
    }
    cfg.architectures = ["VAULTForCausalLM"]
    cfg.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 0,
                "eos_token_id": [0, 2],
                "pad_token_id": 0,
                "max_new_tokens": 256,
                "use_cache": False,
            },
            f,
            indent=2,
        )

    module_map = {
        os.path.join(ROOT_DIR, "configuration_vault.py"): os.path.join(out_dir, "configuration_vault.py"),
        os.path.join(ROOT_DIR, "modeling_vault.py"): os.path.join(out_dir, "modeling_vault.py"),
        os.path.join(ROOT_DIR, "src", "kernels", "flash_vault_v2_triton.py"): os.path.join(
            out_dir, "hf_flash_vault_v2_triton.py"
        ),
        os.path.join(ROOT_DIR, "src", "vault_lm_150m.py"): os.path.join(out_dir, "hf_vault_lm_150m.py"),
        os.path.join(ROOT_DIR, "src", "vault_v2_model.py"): os.path.join(out_dir, "hf_vault_v2_model.py"),
        os.path.join(ROOT_DIR, "src", "vault_v2_streaming.py"): os.path.join(out_dir, "hf_vault_v2_streaming.py"),
        os.path.join(ROOT_DIR, "src", "vault_v2_5_model.py"): os.path.join(out_dir, "hf_vault_v2_5_model.py"),
        os.path.join(ROOT_DIR, "src", "vault_v2_5_streaming.py"): os.path.join(out_dir, "hf_vault_v2_5_streaming.py"),
    }
    for s_p, d_p in module_map.items():
        _rewrite_for_hf_flat(s_p, d_p)

    # Include `run_vault.py` (with flat fallback import) + full benchmark score sheets + MIT LICENSE for HF Hub
    with open(os.path.join(ROOT_DIR, "run_vault.py"), "r", encoding="utf-8") as f:
        rv_code = _normalize_hf_mit_header(f.read()).replace(
            "from src.vault_v2_5_model import _unwrap_checkpoint_state_dict, load_v2_4_into_v2_5",
            "try:\n    from src.vault_v2_5_model import _unwrap_checkpoint_state_dict, load_v2_4_into_v2_5\n"
            "except ImportError:\n    from hf_vault_v2_5_model import _unwrap_checkpoint_state_dict, load_v2_4_into_v2_5",
        )
    with open(os.path.join(out_dir, "run_vault.py"), "w", encoding="utf-8") as f:
        f.write(rv_code)

    with open(os.path.join(out_dir, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(MIT_LICENSE_TEXT)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)
    for b_file in (
        "vault_scores_audit.txt",
        "comparison_v2_4_step10k_vs_v2_5_sft.json",
        "results_vault_traditional.json",
        "results_vault_v2_5_8domain.json",
    ):
        src_b = os.path.join(ROOT_DIR, "benchmarks", b_file)
        if os.path.exists(src_b):
            shutil.copy2(src_b, os.path.join(out_dir, "benchmarks", b_file))

    with open(os.path.join(out_dir, "checkpoint_metadata.json"), "w") as f:
        json.dump({"vault_v2_5_sft": v25_meta, "vault_v2_4_step10k_base": v24_meta}, f, indent=2)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(HF_MODEL_CARD)


def verify_cleanroom_load(out_dir: str) -> None:
    print("[4/5] Running clean-room subprocess test (`cwd=/tmp`, `PYTHONPATH=''`) to verify zero local-repo dependency...")
    test_code = f"""
import importlib.util
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = {out_dir!r}
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True).to(dev).eval()
out, stats = model.chat(
    tok,
    "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
    max_new_tokens=96,
    return_stats=True,
)
print("Clean-room output:\\n" + out)
print("Clean-room stats:", stats)
assert "return maximum, total" in out and "len(nums)" not in out, f"Unexpected output: {{out}}"

# Multi-call CUDAGraph + AMP regression check (back-to-back NIAH + model.generate)
spec = importlib.util.spec_from_file_location("hf_run_vault", os.path.join(p, "run_vault.py"))
rv_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv_mod)
niah_prm = rv_mod.build_live_niah_demo_prompt()
for k in range(2):
    r_niah = model.chat(tok, niah_prm, max_new_tokens=64, temperature=0.0)
    assert "CHK-3213" in r_niah and "CHK-4811" in r_niah, f"Multi-call NIAH {{k}} failed: {{r_niah}}"
ids_g = tok("<|im_start|>user\\nWrite a short Python function `square(x)` that returns x * x.<|im_end|>\\n<|im_start|>assistant\\n", return_tensors="pt").to(dev)
gen_ids = model.generate(**ids_g, max_new_tokens=32, do_sample=False)
gen_txt = tok.decode(gen_ids[0, ids_g["input_ids"].size(1):], skip_special_tokens=True)
assert "def square" in gen_txt, f"model.generate() failed: {{gen_txt}}"

m_wide = model.expand_mitosis(508)
assert sum(m_wide.prime_banks) == 508
print("Clean-room multi-call CUDAGraph + 508c Mitosis verified!")
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run([sys.executable, "-c", test_code], cwd="/tmp", env=env, stdin=subprocess.DEVNULL, check=True)


def upload_to_hf(out_dir: str, repo_id: str) -> str:
    print(f"[5/5] Uploading folder `{out_dir}` to HuggingFace Hub `{repo_id}`...")
    api = HfApi()
    url = api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release VAULT v2.5 (180.6M) + v2.4 Step 10k Base, Zero-rocFFT O(1) Streaming Engine & AutoModelForCausalLM support",
    )
    print(f"[+] Uploaded successfully: {url} (commit: {commit_info.oid if hasattr(commit_info, 'oid') else commit_info})")
    return str(url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "hf_dist"))
    parser.add_argument("--repo_id", type=str, default="batteryphil/VAULT-v2.5-180M")
    parser.add_argument("--skip_upload", action="store_true")
    args = parser.parse_args()

    build_hf_export_dir(args.out_dir)
    verify_cleanroom_load(args.out_dir)
    if not args.skip_upload:
        upload_to_hf(args.out_dir, args.repo_id)


if __name__ == "__main__":
    main()
