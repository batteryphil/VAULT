#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Comprehensive Unseen Evaluation Suite for VAULTLM-150M (263.3M Tokens)

import os
import sys
import json
import math
import shutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vault_lm_150m import VAULTLM150M


@torch.no_grad()
def generate_eval(model, tok, prompt: str, max_new: int = 160, temperature: float = 0.25, device: str = "cuda:0") -> str:
    model.eval()
    full_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
    ids = tok.encode(full_prompt, return_tensors="pt").to(device)
    prompt_len = ids.size(1)
    eos_id = tok.eos_token_id

    for _ in range(max_new):
        ctx = ids[:, -256:]
        logits = model(ctx, step_ratio=1.0)[:, -1, :] / max(temperature, 1e-4)
        for prev_id in set(ids[0, prompt_len:].tolist()):
            if logits[0, prev_id] > 0:
                logits[0, prev_id] /= 1.12
            else:
                logits[0, prev_id] *= 1.12
        top_v, top_i = torch.topk(logits, k=12, dim=-1)
        probs = F.softmax(top_v.float(), dim=-1)
        nxt = top_i.gather(-1, torch.multinomial(probs, 1))
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break

    out = tok.decode(ids[0, prompt_len:], skip_special_tokens=True).strip()
    if "### User:" in out:
        out = out.split("### User:")[0].strip()
    return out


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_path = os.path.join(root, "checkpoints", "vault_150m_latest.pt")
    backup_path = os.path.join(root, "checkpoints", "vault_150m_polyglot_263m.pt")
    if not os.path.exists(backup_path):
        shutil.copy2(ckpt_path, backup_path)
        print(f"[*] Backed up 263M-token polyglot checkpoint -> {backup_path}")

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    model = VAULTLM150M(
        vocab_size=len(tok), d_model=768, n_layers=16, n_heads=12, head_dim=64, d_ff=2048
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    step = ckpt.get("step", 0)
    tokens = ckpt.get("tokens_pushed", 0)
    print(f"[*] Loaded VAULTLM-150M @ Step {step:,} ({tokens:,} tokens)")

    unseen_suite = [
        ("1. BuildBit (.bb) GPU SoA", "Write a BuildBit (.bb) function that defines `soa struct Photon` with `f32` fields `x`, `vx`, `ax`, allocates 4096 elements in a 16 MB `gpu_arena`, and updates positions in parallel."),
        ("2. C99 Systems Code", "Write a C99 function to reverse an array of 32-bit integers in place."),
        ("3. C++20 Generic Template", "Write a C++20 generic class template `FixedStack<T, 128>` with `push`, `pop`, and `size` methods."),
        ("4. Python Algorithm", "[python] Write a Python function to find the maximum element and sum of a list of numbers."),
        ("5. Lua Metatable Module", "Write a Lua module implementing a `Particle` class using metatables (`__index` and `:update(dt)`)."),
        ("6. Chain-of-Thought Math (<think>)", "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?"),
        ("7. Conversational Explanation", "Explain why recursion needs a base case in programming."),
    ]

    results = {"step": step, "tokens_pushed": tokens, "evaluations": []}
    for title, prompt in unseen_suite:
        out = generate_eval(model, tok, prompt, max_new=150, temperature=0.2, device=device)
        print(f"\n==================== {title} ====================")
        print(f"Prompt: {prompt}")
        print(f"Output:\n{out}")
        results["evaluations"].append({"domain": title, "prompt": prompt, "output": out})

    out_json = os.path.join(root, "benchmarks", "eval_vault_150m_263m.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved evaluation results -> {out_json}")


if __name__ == "__main__":
    main()
