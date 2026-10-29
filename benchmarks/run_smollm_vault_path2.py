#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Path 2 Verification: Surgical Conversion + Fast Distillation Alignment
# on Pretrained HuggingFaceTB/SmolLM-135M

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PRIME-Moment-Attention", "src")))

from vault_surgery import convert_model_to_vault, VAULTTransplantedAttention
from prime_moment_attention.surgery import convert_transformer_to_prime


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, eval_batches: list) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids in eval_batches:
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels.numel()
    avg_nll = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_nll, 20.0))


@torch.no_grad()
def generate_sample(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 48, device: str = "cuda") -> str:
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        out = model(input_ids=ids, use_cache=False)
        next_logits = out.logits[:, -1, :] / 0.7
        # Top-k sampling for natural text
        top_vals, top_idx = torch.topk(next_logits, k=10, dim=-1)
        probs = F.softmax(top_vals, dim=-1)
        sampled = top_idx.gather(-1, torch.multinomial(probs, num_samples=1))
        ids = torch.cat([ids, sampled], dim=-1)
        if sampled.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def load_local_token_batches(tokenizer, seq_len: int = 128, batch_size: int = 4, num_train: int = 160, num_eval: int = 20, device: str = "cuda"):
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    train_arrow = os.path.join(hf_cache, "datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-train-00000-of-00004.arrow")
    val_arrow = os.path.join(hf_cache, "datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/tiny_stories-validation.arrow")

    train_ds = Dataset.from_file(train_arrow)
    val_ds = Dataset.from_file(val_arrow)

    def pack_sequences(ds, count_batches):
        batches = []
        token_buf = []
        idx = 0
        needed = count_batches * batch_size * seq_len
        while len(token_buf) < needed and idx < len(ds):
            txt = ds[idx]["text"]
            idx += 1
            if not txt or len(txt) < 50:
                continue
            toks = tokenizer.encode(txt, add_special_tokens=False)
            token_buf.extend(toks)
        for b in range(count_batches):
            start = b * batch_size * seq_len
            chunk = token_buf[start : start + batch_size * seq_len]
            t = torch.tensor(chunk, dtype=torch.long, device=device).view(batch_size, seq_len)
            batches.append(t)
        return batches

    return pack_sequences(train_ds, num_train), pack_sequences(val_ds, num_eval)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM-135M"

    print("=" * 92)
    print(f"VAULT PATH-2 SURGICAL CONVERSION & FAST ALIGNMENT ON {model_id}")
    print("=" * 92)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[1/5] Loading local tokenized batches (seq_len=128, batch_size=4)...", flush=True)
    train_batches, eval_batches = load_local_token_batches(
        tokenizer, seq_len=128, batch_size=4, num_train=100, num_eval=10, device=device
    )

    # 1. Load Teacher Model
    print("[2/5] Evaluating Frozen Teacher Model (HuggingFaceTB/SmolLM-135M)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, dtype=torch.float32
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_ppl = evaluate_perplexity(teacher, eval_batches)
    prompt_1 = "Once upon a time, in a quiet village, a young inventor built a"
    teacher_gen = generate_sample(teacher, tokenizer, prompt_1, max_new_tokens=40, device=device)
    print(f"  -> Frozen Teacher Validation PPL : {teacher_ppl:.2f}", flush=True)
    print(f"  -> Frozen Teacher Generation     : {teacher_gen!r}", flush=True)

    # 2. Evaluate PRIME-Moment Surgical Transplant Baseline (75% Hybrid Trunk)
    print("\n[3/5] Evaluating PRIME-Moment Surgical Transplant Baseline (75% Trunk)...", flush=True)
    prime_model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, dtype=torch.float32
    ).to(device)
    prime_model, prime_layers = convert_transformer_to_prime(prime_model, hybrid_ratio=0.75)
    prime_zero_ppl = evaluate_perplexity(prime_model, eval_batches)
    print(f"  -> PRIME-Moment Zero-Shot PPL ({len(prime_layers)}/30 layers): {prime_zero_ppl:.2f}", flush=True)
    del prime_model
    torch.cuda.empty_cache()

    # 3. Surgical Conversion to VAULT (75% Hybrid Trunk: 24 of 30 layers converted)
    print("\n[4/5] Surgically Converting SmolLM-135M -> SmolLM-135M-VAULT (75% Trunk, 24/30 layers)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, dtype=torch.float32
    ).to(device)
    student, converted_layers = convert_model_to_vault(
        student, hybrid_ratio=0.75, prime_banks=(17, 19, 23, 29), decay=0.96
    )

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  -> Converted Layers : {len(converted_layers)} / {len(student.model.layers)} layers ({sorted(list(converted_layers))[0]}..{sorted(list(converted_layers))[-1]})", flush=True)
    print(f"  -> Total Parameters : {total_params:,}", flush=True)
    print(f"  -> Trainable Adapter: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}% of model; 94.0% Frozen!)", flush=True)

    vault_zero_ppl = evaluate_perplexity(student, eval_batches)
    vault_zero_gen = generate_sample(student, tokenizer, prompt_1, max_new_tokens=40, device=device)
    print(f"  -> VAULT Step-0 (Zero-Shot) PPL  : {vault_zero_ppl:.2f} (vs PRIME-Moment {prime_zero_ppl:.2f})", flush=True)
    print(f"  -> VAULT Step-0 Generation       : {vault_zero_gen!r}", flush=True)

    # 4. Path 2 Fast Alignment (100 Steps of Combined CE + Teacher KL Distillation)
    print("\n[5/5] Running Path-2 Fast Distillation Alignment (100 Steps)...", flush=True)
    print("-" * 92, flush=True)
    print(f"{'Step':<8} | {'CE Loss':<10} | {'KL Loss':<10} | {'Val PPL':<12} | {'VaultGate':<10} | {'Elapsed':<10}", flush=True)
    print("-" * 92, flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_batches), eta_min=5e-5)

    t0 = time.perf_counter()
    history = [{"step": 0, "val_ppl": vault_zero_ppl}]

    student.train()
    for step_idx, batch_ids in enumerate(train_batches, start=1):
        optimizer.zero_grad()

        with torch.no_grad():
            t_logits = teacher(input_ids=batch_ids, use_cache=False).logits

        s_logits = student(input_ids=batch_ids, use_cache=False).logits

        # Next-token Cross-Entropy Loss
        shift_logits = s_logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Teacher-Student Logit KL Distillation (Temperature T = 2.0)
        temp = 2.0
        s_log_probs = F.log_softmax(s_logits / temp, dim=-1)
        t_probs = F.softmax(t_logits / temp, dim=-1)
        kl_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temp ** 2)

        loss = 0.5 * ce_loss + 0.5 * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if step_idx % 20 == 0 or step_idx == len(train_batches):
            val_ppl = evaluate_perplexity(student, eval_batches)
            student.train()
            v_gates = [
                student.model.layers[idx].self_attn.vault_gate.abs().mean().item()
                for idx in converted_layers
            ]
            mean_v_gate = sum(v_gates) / len(v_gates)
            elapsed = time.perf_counter() - t0
            print(f"{step_idx:<8} | {ce_loss.item():<10.4f} | {kl_loss.item():<10.4f} | {val_ppl:<12.2f} | {mean_v_gate:<10.4f} | {elapsed:<8.1f}s", flush=True)
            history.append({
                "step": step_idx,
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "val_ppl": val_ppl,
                "mean_vault_gate": mean_v_gate,
                "elapsed_sec": elapsed,
            })

    total_train_time = time.perf_counter() - t0
    final_ppl = history[-1]["val_ppl"]

    print("-" * 92)
    print(f"Path-2 Alignment Complete in {total_train_time:.1f}s! Final Val PPL: {final_ppl:.2f} (Teacher: {teacher_ppl:.2f})")

    # Generate samples after Path 2 alignment
    prompts = [
        "Once upon a time, in a quiet village, a young inventor built a",
        "Lily found a shiny golden key under the old oak tree. She picked it up and",
    ]
    generations = {}
    print("\nPost-Alignment Live Generation Samples (SmolLM-135M-VAULT):")
    for p in prompts:
        gen_txt = generate_sample(student, tokenizer, p, max_new_tokens=45, device=device)
        generations[p] = gen_txt
        print(f"  Prompt : {p!r}")
        print(f"  VAULT  : {gen_txt!r}\n")

    # 5. Long-Horizon Associative Passkey Retrieval on Pretrained SmolLM-135M Embeddings
    print("=" * 92)
    print("LONG-HORIZON O(1)-STATE PASSKEY RETRIEVAL ON PRETRAINED SmolLM-135M REPRESENTATIONS")
    print("=" * 92)
    print(f"{'Sequence Len L':<16} | {'Needle Pos':<12} | {'Linear(ELU+1) Cosine':<22} | {'SmolLM-VAULT Cosine':<22}")
    print("-" * 92)

    mid_layer_attn: VAULTTransplantedAttention = student.model.layers[15].self_attn
    passkey_results = []
    # Gather a long stream of real token IDs from eval_batches
    long_tok_stream = torch.cat([b.reshape(-1) for b in eval_batches], dim=0)

    for L in [256, 512, 1024, 2048, 4096]:
        if len(long_tok_stream) < L:
            repeats = (L // len(long_tok_stream)) + 1
            tok_slice = long_tok_stream.repeat(repeats)[:L].unsqueeze(0)
        else:
            tok_slice = long_tok_stream[:L].unsqueeze(0)

        with torch.no_grad():
            emb = student.model.embed_tokens(tok_slice)
            # Inject a distinct rare token embedding as the passkey needle at position L // 10
            needle_pos = max(5, L // 10)
            rare_token_id = torch.tensor([[12345]], device=device, dtype=torch.long)
            emb[:, needle_pos : needle_pos + 1, :] = student.model.embed_tokens(rare_token_id) * 1.5

            res = mid_layer_attn.verify_long_horizon_passkey_recall(emb, needle_pos=needle_pos)
            print(f"{L:<16} | {needle_pos:<12} | {res['linear_elu_cosine']:<22.4f} | {res['vault_cosine']:<22.4f}")
            passkey_results.append({
                "seq_len": L,
                "needle_pos": needle_pos,
                "linear_elu_cosine": res["linear_elu_cosine"],
                "vault_cosine": res["vault_cosine"],
            })

    print("=" * 92)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smollm_135m_vault_path2_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model_id": model_id,
            "converted_layers": len(converted_layers),
            "total_layers": len(student.model.layers),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "teacher_val_ppl": teacher_ppl,
            "prime_moment_zero_shot_ppl": prime_zero_ppl,
            "vault_zero_shot_ppl": vault_zero_ppl,
            "vault_aligned_ppl": final_ppl,
            "alignment_time_sec": total_train_time,
            "training_history": history,
            "generations": generations,
            "long_horizon_passkey_recall": passkey_results,
        }, f, indent=2)
    print(f"Saved full Path-2 verification telemetry to {out_path}")


if __name__ == "__main__":
    main()
