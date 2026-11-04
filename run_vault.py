#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# One-Command VAULT v2.5 Runner (Auto-Downloads from HuggingFace Hub if Local Checkpoint Not Found)
#
# Usage Examples:
#   python run_vault.py --demo
#   python run_vault.py --prompt "Explain why recursion needs a base case in programming."
#   python run_vault.py --cells 2002 --prompt "Write a Python function to find the maximum element and sum of a list of numbers."
#   python run_vault.py                       # Interactive streaming Chat REPL

import argparse
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configuration_vault import MITOSIS_PRESETS, VAULTConfig
from modeling_vault import VAULTForCausalLM
from src.vault_v2_5_model import _unwrap_checkpoint_state_dict, load_v2_4_into_v2_5

DEFAULT_HF_REPO = "batteryphil/VAULT-v2.5-180M"


def load_vault_easy(
    repo_or_path: str = DEFAULT_HF_REPO,
    local_ckpt: str = "",
    variant: str = "sft",
    cells: int = 220,
    device: str = "",
):
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    default_local = os.path.join(
        ROOT_DIR,
        "checkpoints",
        "vault_v2_5_best.pt" if variant == "sft" else "vault_v2_4_step10000_best.pt",
    )
    ckpt_file = local_ckpt or (default_local if os.path.exists(default_local) else "")

    if ckpt_file and os.path.isfile(ckpt_file):
        print(f"[VAULT] Loading local checkpoint: {ckpt_file} ...", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)
        raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        sd = _unwrap_checkpoint_state_dict(raw)
        cfg = VAULTConfig(
            vocab_size=len(tok),
            prime_banks=list(raw.get("prime_banks", [47, 53, 59, 61])),
            ema_alpha=float(raw.get("ema_alpha", 2.0**-11)),
        )
        model = VAULTForCausalLM(cfg)
        if any("ext_span_alpha" in k for k in sd.keys()):
            sd_load = dict(sd)
            if not cfg.tie_word_embeddings and "lm_head.weight" in sd_load:
                sd_load["lm_head.weight"] = sd_load["lm_head.weight"].clone()
            model.load_state_dict(sd_load, strict=True)
        else:
            load_v2_4_into_v2_5(model, raw, init_ext_span_alpha=0.55)  # type: ignore[arg-type]
    elif os.path.isdir(repo_or_path):
        print(f"[VAULT] Loading from directory: {repo_or_path} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo_or_path, trust_remote_code=True)
    else:
        print(f"[VAULT] Downloading / loading from HuggingFace Hub: {repo_or_path} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)
        if variant == "base":
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            cfg = AutoConfig.from_pretrained(repo_or_path, trust_remote_code=True)
            model = VAULTForCausalLM(cfg)
            base_st = hf_hub_download(repo_id=repo_or_path, filename="base/vault_v2_4_step10k.safetensors")
            load_v2_4_into_v2_5(model, load_file(base_st), init_ext_span_alpha=0.0)  # type: ignore[arg-type]
        else:
            model = AutoModelForCausalLM.from_pretrained(repo_or_path, trust_remote_code=True)

    model = model.to(device).eval()
    if cells != sum(model.prime_banks):
        print(f"[VAULT] Expanding Co-Prime Kenyon Cell banks via Zero-Shot Mitosis: {sum(model.prime_banks)} -> {cells} cells...", flush=True)
        model = model.expand_mitosis(cells)

    hw_name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"
    crt_cap = 1
    for p in model.prime_banks:
        crt_cap *= int(p)
    total_p = sum(p.numel() for p in model.parameters())
    emb_p = model.embed_tokens.weight.numel() if hasattr(model, "embed_tokens") else 0
    uniq_p = (
        total_p - emb_p
        if hasattr(model, "lm_head")
        and model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
        and torch.equal(model.lm_head.weight.data, model.embed_tokens.weight.data)
        else total_p
    )
    print(
        f"[VAULT] Ready on {hw_name} ({device}) | Params: {uniq_p/1e6:.2f}M | "
        f"Banks: {tuple(model.prime_banks)} ({sum(model.prime_banks)} cells, {crt_cap:,} CRT addresses)",
        flush=True,
    )
    return model, tok


def build_live_niah_demo_prompt() -> str:
    filler = (
        "Telemetry packet validated across dual-channel optical bus during attitude stabilization; "
        "thermal loop nominal and frame checksum committed without retransmission."
    )
    subsystems = [
        ("NavBeacon-A1", "CHK-1904"),
        ("CryoPump-B4", "CHK-7742"),
        ("PlasmaGrid-E6", "CHK-3213"),
        ("GyroArray-F2", "CHK-6058"),
        ("SolarBus-H9", "CHK-8420"),
        ("ThermalLouvre-K3", "CHK-5197"),
        ("CryoLock-N6", "CHK-4811"),
        ("StarTracker-Q8", "CHK-9365"),
    ]
    lines = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST"]
    for idx, (name, code) in enumerate(subsystems, start=1):
        lines.append(f"[Subsystem {idx:03d}: {name}] Status: NOMINAL | Verification Code: {code}")
        lines.append(f"  Log: {filler} Frame {100 + idx} ok.")
    lines.append(
        "\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
        "for PlasmaGrid-E6 and CryoLock-N6? List each subsystem and its exact code."
    )
    return "\n".join(lines)


def run_showcase_demo(model, tok) -> None:
    demos = [
        (
            "1. Python Code Generation",
            "[python] Write a Python function to find the maximum element and sum of a list of numbers.",
        ),
        (
            "2. Direct Computer Science Prose",
            "Explain why recursion needs a base case in programming.",
        ),
        (
            "3. Chain-of-Thought Reasoning (<think>)",
            "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?",
        ),
        (
            "4. Multi-Needle In-Context Retrieval (Expected: PlasmaGrid-E6=CHK-3213, CryoLock-N6=CHK-4811)",
            build_live_niah_demo_prompt(),
        ),
    ]
    # Warmup 1 short call so Triton / CUDAGraph compilation is excluded from steady-state demo speeds
    _ = model.chat(tok, "Say hello.", max_new_tokens=8)

    for title, prm in demos:
        print("\n" + "=" * 96)
        print(f"DEMO {title}")
        print("-" * 96)
        _, stats = model.chat(tok, prm, max_new_tokens=192, temperature=0.0, stream=True, return_stats=True)
        print(
            f"\n[Stats] prompt={stats['prompt_tokens']}t | gen={stats['generated_tokens']}t | "
            f"state={stats['persistent_state_mb']:.1f}MB ({stats['total_cells']}c) | "
            f"prefill={stats['prefill_tok_s']:,.0f} tok/s | "
            f"O(1) decode={stats['decode_ms_per_tok']:.2f} ms/tok ({stats['decode_tok_s']:.1f} tok/s)"
        )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & Run VAULT v2.5 (180.6M Co-Prime Kenyon Cell LM)")
    parser.add_argument("--repo", type=str, default=DEFAULT_HF_REPO, help="HuggingFace repo ID or local export dir")
    parser.add_argument("--ckpt", type=str, default="", help="Optional path to local .pt checkpoint")
    parser.add_argument("--variant", choices=["sft", "base"], default="sft", help="Load v2.5 SFT (default) or v2.4 Step 10k Base")
    parser.add_argument("--cells", type=int, choices=sorted(MITOSIS_PRESETS.keys()), default=220, help="Co-Prime Kenyon Cell bank size (220, 508, 942, 2002)")
    parser.add_argument("--device", type=str, default="", help="Device (default: cuda:0 if available else cpu)")
    parser.add_argument("--prompt", type=str, default="", help="Run a single prompt and exit")
    parser.add_argument("--raw", action="store_true", help="Use raw text completion instead of ChatML template")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--demo", action="store_true", help="Run the 4-domain live streaming showcase")
    args = parser.parse_args()

    model, tok = load_vault_easy(
        repo_or_path=args.repo,
        local_ckpt=args.ckpt,
        variant=args.variant,
        cells=args.cells,
        device=args.device,
    )

    if args.demo:
        run_showcase_demo(model, tok)
        return

    if args.prompt:
        if args.raw:
            inp = torch.tensor([tok.encode(args.prompt, add_special_tokens=False)], dtype=torch.long)
            printed = 0
            buf = []

            def _cb(tid: int) -> None:
                nonlocal printed
                buf.append(tid)
                s = tok.decode(buf, skip_special_tokens=True)
                if len(s) > printed:
                    print(s[printed:], end="", flush=True)
                    printed = len(s)

            print(args.prompt, end="", flush=True)
            _, stats = model.stream_generate(
                inp,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_ids=(tok.eos_token_id or 0,),
                on_token=_cb,
                return_stats=True,
            )
            print(
                f"\n\n[Stats] state={stats['persistent_state_mb']:.1f}MB ({stats['total_cells']}c) | "
                f"O(1) decode={stats['decode_ms_per_tok']:.2f} ms/tok ({stats['decode_tok_s']:.1f} tok/s)"
            )
        else:
            _, stats = model.chat(
                tok,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                return_stats=True,
            )
            print(
                f"\n[Stats] state={stats['persistent_state_mb']:.1f}MB ({stats['total_cells']}c) | "
                f"O(1) decode={stats['decode_ms_per_tok']:.2f} ms/tok ({stats['decode_tok_s']:.1f} tok/s)"
            )
        return

    # Interactive Streaming REPL
    print("\nInteractive VAULT v2.5 Chat REPL")
    print("Commands: /demo | /mitosis <220|508|942|2002> | /temp <float> | /quit\n")
    temp = args.temperature
    while True:
        try:
            user_in = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "exit", "quit"):
            break
        if user_in == "/demo":
            run_showcase_demo(model, tok)
            continue
        if user_in.startswith("/mitosis"):
            parts = user_in.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) in MITOSIS_PRESETS:
                target_c = int(parts[1])
                model = model.expand_mitosis(target_c)
                print(f"[+] Mitosis complete -> {sum(model.prime_banks)} Kenyon cells {tuple(model.prime_banks)}")
            else:
                print(f"Usage: /mitosis <{'|'.join(str(k) for k in sorted(MITOSIS_PRESETS))}>")
            continue
        if user_in.startswith("/temp"):
            parts = user_in.split()
            if len(parts) == 2:
                temp = float(parts[1])
                print(f"[+] Temperature set to {temp}")
            continue

        print("VAULT > ", end="", flush=True)
        _, stats = model.chat(
            tok,
            user_in,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            top_p=args.top_p,
            stream=True,
            return_stats=True,
        )
        print(
            f"  [{stats['decode_ms_per_tok']:.2f} ms/tok ({stats['decode_tok_s']:.1f} tok/s) | "
            f"state={stats['persistent_state_mb']:.1f}MB ({stats['total_cells']}c)]\n"
        )


if __name__ == "__main__":
    main()
