# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# HuggingFace `AutoModelForCausalLM` wrapper for VAULT v2.5:
#   - 1-to-1 lossless `safetensors` compatibility with `VAULTLMv2_5`
#   - Full `GenerationMixin` (`model.generate(...)` & `transformers.pipeline("text-generation")`)
#   - Native `float32`, `bfloat16`, `float16`, `device_map="auto"`, and CPU fallback support
#   - True O(1)-state streaming prefill & Zero-rocFFT Triton / CUDAGraph decode (`stream_generate`, `chat`)
#   - 1-line Zero-Shot Elastic Co-Prime Mitosis (`expand_mitosis(2002)`)

from contextlib import nullcontext
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_vault import VAULTConfig, resolve_prime_banks
except ImportError:
    from configuration_vault import VAULTConfig, resolve_prime_banks

from src.kernels.flash_vault_v2_triton import _gpu_supports_bf16_triton
from src.vault_v2_5_model import (
    VAULTLMv2_5,
    compute_v2_5_shared_aux,
    upgrade_v2_5_to_wider_primes,
)
from src.vault_v2_5_streaming import VAULTv2_5StreamingEngine


_FP32_BUFFER_SUFFIXES = (
    "harmonic_anchors",
    "local_decay_mask",
    "_local_attn_bias",
    "rope_cos_k",
    "rope_sin_k",
    "rope_cos_q",
    "rope_sin_q",
    "x0_cone_mean",
    "cms_proj",
    "sal_lag_dist",
    "_id_weights_3g",
)


class VAULTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VAULTConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["VAULTv2_5Block"]
    _supports_cache_class = False

    def __init__(self, config: VAULTConfig):
        super().__init__(config)
        # Materialize on CPU in FP32 first even if `transformers >= 5.0` enters `__init__` under
        # `with torch.device("meta"):` or `dtype=torch.bfloat16 / float16`, then cast parameters cleanly.
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.device("cpu"):
                core = VAULTLMv2_5(
                    vocab_size=config.vocab_size,
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    n_q_heads=config.n_q_heads,
                    n_vault_groups=config.n_vault_groups,
                    head_dim=config.head_dim,
                    d_ff=config.d_ff,
                    chunk_size=config.chunk_size,
                    grad_checkpoint="none",
                    prime_banks=tuple(config.prime_banks),
                    ema_alpha=float(config.ema_alpha),
                )
                if not config.tie_word_embeddings:
                    core.lm_head.weight = nn.Parameter(core.embed_tokens.weight.detach().clone())
                if orig_dtype != torch.float32:
                    for p in core.parameters():
                        if p.is_floating_point():
                            p.data = p.data.to(orig_dtype)
        finally:
            torch.set_default_dtype(orig_dtype)

        for name, mod in core._modules.items():
            self._modules[name] = mod
        for name, par in core._parameters.items():
            self._parameters[name] = par
        for name, buf in core._buffers.items():
            self._buffers[name] = buf
        for k, v in core.__dict__.items():
            if not k.startswith("_"):
                setattr(self, k, v)

        self.post_init()
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.use_cache = False

    def _keep_precision_buffers_fp32_(self) -> None:
        if not hasattr(self, "layers"):
            return
        for mod in self.modules():
            for b_name, buf in list(mod._buffers.items()):
                if buf is not None and buf.device.type != "meta" and buf.is_floating_point():
                    if b_name in _FP32_BUFFER_SUFFIXES and buf.dtype != torch.float32:
                        mod._buffers[b_name] = buf.to(torch.float32)
        for blk in self.layers:
            blk.attn._eval_w_cache = None
        if hasattr(self, "_decode_static_pack_v2_5"):
            delattr(self, "_decode_static_pack_v2_5")
        if hasattr(self, "_cudagraph_decode_v2_5"):
            delattr(self, "_cudagraph_decode_v2_5")

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse=recurse)
        self._keep_precision_buffers_fp32_()
        self._restore_non_persistent_buffers_()
        return out

    @torch.no_grad()
    def _restore_non_persistent_buffers_(self) -> None:
        """
        Re-populates deterministic `persistent=False` buffers (`_local_attn_bias`, `_sal_lag_mask_3g`,
        `_id_weights_3g`) in exact FP32 after `transformers >= 5.0` runs `model.to_empty()` or `.to(dtype)`.
        """
        if not hasattr(self, "layers"):
            return
        for blk in self.layers:
            attn = blk.attn
            dev = attn.local_decay_mask.device
            if dev.type != "meta":
                C = int(attn.chunk_size)
                log_one_minus = torch.linspace(
                    math.log(0.34), math.log(0.008), int(attn.n_q_heads), device=dev, dtype=torch.float32
                )
                head_decays = 1.0 - torch.exp(log_one_minus)
                idx_q = torch.arange(C, device=dev, dtype=torch.float32) + C
                idx_k = torch.arange(2 * C, device=dev, dtype=torch.float32)
                diff_f = idx_q.unsqueeze(-1) - idx_k.unsqueeze(-2)
                valid = (diff_f >= 0).float().unsqueeze(0)
                decay_3d = torch.pow(head_decays.view(int(attn.n_q_heads), 1, 1), diff_f.clamp(min=0.0).unsqueeze(0)) * valid
                attn_bias = torch.where(
                    decay_3d > 0.0, torch.log(decay_3d.clamp(min=1e-8)), torch.full_like(decay_3d, -1e9)
                )
                attn._buffers["_local_attn_bias"] = attn_bias.view(1, int(attn.n_q_heads), 1, C, 2 * C)

                sb = attn.salience_binder
                W = int(sb.window_size)
                iq = torch.arange(C, device=dev) + C
                ik = torch.arange(2 * C, device=dev)
                df = iq[:, None] - ik[None, :]
                lag_3g = torch.stack(
                    [(df >= 4) & (df <= W), (df >= 7) & (df <= W), (df >= 11) & (df <= W)], dim=0
                ).unsqueeze(0).unsqueeze(2)
                sb._buffers["_sal_lag_mask_3g"] = lag_3g
                sb._buffers["_id_weights_3g"] = torch.tensor(
                    [1.50, 1.20, 1.15], device=dev, dtype=torch.float32
                ).view(1, 1, 3, 1, 1)
            attn._eval_w_cache = None

    def _init_weights(self, module: nn.Module) -> None:
        if module is self:
            self._keep_precision_buffers_fp32_()
            self._restore_non_persistent_buffers_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._keep_precision_buffers_fp32_()
        model._restore_non_persistent_buffers_()
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
        return model

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def compute_x0_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.x0_rms_norm(x.detach())

    def compute_cms_aux(self, x0_norm: torch.Tensor):
        b0 = self.layers[0].attn.salience_binder
        return compute_v2_5_shared_aux(x0_norm, b0, ema_alpha=self.ema_alpha)

    def _encode_hidden_with_aux(
        self, input_ids: torch.Tensor, step_ratio: float = 1.0, return_ent_div: bool = False
    ):
        return VAULTLMv2_5._encode_hidden_with_aux(
            self, input_ids, step_ratio=step_ratio, return_ent_div=return_ent_div
        )

    def _encode_hidden(self, input_ids: torch.Tensor, step_ratio: float = 1.0) -> torch.Tensor:
        return self._encode_hidden_with_aux(input_ids, step_ratio=step_ratio)[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        step_ratio: float = 1.0,
        return_dict: bool = True,
        **kwargs,
    ):
        del attention_mask, kwargs
        h = self._encode_hidden(input_ids, step_ratio=step_ratio)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        **kwargs,
    ):
        """
        HuggingFace `model.generate(...)` entry point:
          - Automatically accelerates standard `batch_size == 1` greedy / temperature / top-p calls via
            VAULT's hardware-fused O(1) `stream_generate` engine (~49 tok/s on GPU).
          - Seamlessly delegates `transformers.pipeline("text-generation")`, `batch_size > 1`, beam search,
            streamers, and custom `GenerationMixin` processors to `super().generate(..., use_cache=False)`.
        """
        inp = inputs if inputs is not None else kwargs.get("input_ids", None)
        exotic_keys = {
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
            "custom_generate",
        }
        has_exotic = any(kwargs.get(k) is not None for k in exotic_keys)
        num_beams = int(kwargs.get("num_beams", getattr(generation_config, "num_beams", 1) or 1))
        num_ret = int(
            kwargs.get("num_return_sequences", getattr(generation_config, "num_return_sequences", 1) or 1)
        )
        ret_dict = bool(
            kwargs.get("return_dict_in_generate", getattr(generation_config, "return_dict_in_generate", False))
        )
        out_scores = bool(
            kwargs.get("output_scores", getattr(generation_config, "output_scores", False))
            or kwargs.get("output_logits", getattr(generation_config, "output_logits", False))
            or kwargs.get("output_hidden_states", getattr(generation_config, "output_hidden_states", False))
        )
        top_k = kwargs.get("top_k", None)

        if (
            isinstance(inp, torch.Tensor)
            and inp.dim() in (1, 2)
            and (inp.dim() == 1 or inp.size(0) == 1)
            and generation_config is None
            and not has_exotic
            and num_beams == 1
            and num_ret == 1
            and not ret_dict
            and not out_scores
            and top_k is None
        ):
            max_new = kwargs.get("max_new_tokens", None)
            if max_new is None:
                max_len = kwargs.get("max_length", None)
                seq_len = inp.size(-1)
                max_new = max(1, int(max_len) - seq_len) if max_len is not None else 256
            do_sample = bool(kwargs.get("do_sample", False))
            temp = float(kwargs.get("temperature", 1.0)) if do_sample else 0.0
            top_p = float(kwargs.get("top_p", 1.0)) if do_sample else 1.0
            rep_pen = float(kwargs.get("repetition_penalty", 1.0))
            eos_arg = kwargs.get("eos_token_id", [0, 2])
            eos_ids = [eos_arg] if isinstance(eos_arg, int) else list(eos_arg if eos_arg is not None else [0, 2])
            return self.stream_generate(
                inp,
                max_new_tokens=int(max_new),
                temperature=temp,
                top_p=top_p,
                repetition_penalty=rep_pen,
                eos_token_ids=eos_ids,
            )

        kwargs["use_cache"] = False
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

    @torch.no_grad()
    def expand_mitosis(
        self,
        cells_or_primes: Union[int, Sequence[int]] = 2002,
        split_angle_rad: float = 0.26,
    ) -> "VAULTForCausalLM":
        """
        Zero-Shot Elastic Co-Prime Mitosis (`220 -> 508 / 942 / 2002` Kenyon cells).
        Expands Voronoi associative capacity up to `62.7B` CRT addresses while preserving O(1) decode speed.
        """
        target_banks = resolve_prime_banks(cells_or_primes)
        if tuple(target_banks) == tuple(self.prime_banks):
            return self

        dev = self.embed_tokens.weight.device
        dt = self.embed_tokens.weight.dtype
        new_cfg = VAULTConfig.from_dict(self.config.to_dict())
        new_cfg.prime_banks = list(target_banks)
        if sum(target_banks) > 220 and float(getattr(new_cfg, "occ_alpha", 0.5)) == 0.5:
            new_cfg.occ_alpha = 1.5

        wide = VAULTForCausalLM(new_cfg)
        src_primes = tuple(int(p) for p in self.prime_banks)
        src_sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        upgrade_v2_5_to_wider_primes(
            wide,  # type: ignore[arg-type]
            src_sd,
            old_primes=src_primes,
            split_angle_rad=split_angle_rad,
        )
        return wide.to(device=dev, dtype=dt).eval()

    @staticmethod
    def _sample_next_token(
        logits_1d: torch.Tensor,
        generated: Sequence[int],
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> int:
        lg = logits_1d.float().clone()
        if repetition_penalty != 1.0 and generated:
            seen = torch.tensor(list(set(generated)), device=lg.device, dtype=torch.long)
            vals = lg.index_select(0, seen)
            vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            lg.scatter_(0, seen, vals)

        if temperature <= 1e-5:
            return int(torch.argmax(lg).item())

        probs = F.softmax(lg / float(temperature), dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum().clamp(min=1e-8)
            choice = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx.gather(0, choice).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_ids: Sequence[int] = (0, 2),
        occ_alpha: Optional[float] = None,
        segment_size: int = 4096,
        use_cudagraph: bool = True,
        use_bf16_amp: Optional[bool] = None,
        on_token: Optional[Callable[[int], None]] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        True O(1)-state streaming prefill + constant-latency autoregressive decode via `VAULTv2_5StreamingEngine`.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.size(0) == 1, "stream_generate currently supports batch_size=1"

        dev = self.embed_tokens.weight.device
        inp = input_ids.to(dev)
        if occ_alpha is not None:
            occ = float(occ_alpha)
        else:
            cfg_occ = float(getattr(self.config, "occ_alpha", 0.5))
            occ = 1.5 if (cfg_occ == 0.5 and (sum(self.prime_banks) > 220 or inp.size(1) > 4096)) else cfg_occ

        is_cuda = dev.type == "cuda" and torch.cuda.is_available()
        w_dt = self.embed_tokens.weight.dtype
        bf16_hw = is_cuda and _gpu_supports_bf16_triton(dev)
        enable_bf16_amp = bool(
            use_bf16_amp if use_bf16_amp is not None else (bf16_hw and w_dt == torch.float32)
        )
        stop_set = set(int(x) for x in eos_token_ids if x is not None and int(x) >= 0)
        cg_ok = bool(
            use_cudagraph
            and is_cuda
            and (w_dt == torch.float32 or (w_dt == torch.bfloat16 and bf16_hw))
        )

        if is_cuda:
            amp_guard = (
                torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
                if enable_bf16_amp
                else torch.amp.autocast("cuda", enabled=False)
            )
        else:
            amp_guard = nullcontext()
        with amp_guard:
            if is_cuda:
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            seg = max(64, (int(segment_size) // 64) * 64)
            logits_last, cache = VAULTv2_5StreamingEngine.prefill_streaming(
                self,  # type: ignore[arg-type]
                inp,
                segment_size=seg,
                tail_logits_len=1,
                occ_alpha=occ,
                use_cudagraph=cg_ok,
            )
            if is_cuda:
                torch.cuda.synchronize(dev)
            prefill_sec = max(1e-6, time.perf_counter() - t0)
            state_mb = round(cache.state_bytes() / (1024 ** 2), 2)

            gen_tokens = []
            nxt_id = self._sample_next_token(
                logits_last[0, -1],
                gen_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            t_dec0 = time.perf_counter()
            n_decoded = 0
            if nxt_id not in stop_set and max_new_tokens > 0:
                gen_tokens.append(nxt_id)
                if on_token is not None:
                    on_token(nxt_id)

                for _ in range(max_new_tokens - 1):
                    tok_t = torch.tensor([[nxt_id]], device=dev, dtype=torch.long)
                    step_logits = VAULTv2_5StreamingEngine.decode_step(
                        self,  # type: ignore[arg-type]
                        tok_t,
                        cache,
                        occ_alpha=occ,
                        use_cudagraph=cg_ok,
                    )
                    n_decoded += 1
                    nxt_id = self._sample_next_token(
                        step_logits[0, -1],
                        gen_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    if nxt_id in stop_set:
                        break
                    gen_tokens.append(nxt_id)
                    if on_token is not None:
                        on_token(nxt_id)

            if is_cuda:
                torch.cuda.synchronize(dev)
            decode_sec = max(1e-6, time.perf_counter() - t_dec0)

        out_ids = (
            torch.cat([inp, torch.tensor([gen_tokens], device=dev, dtype=torch.long)], dim=1)
            if gen_tokens
            else inp
        )
        if not return_stats:
            return out_ids

        steps = max(1, n_decoded)
        stats = {
            "prompt_tokens": int(inp.size(1)),
            "generated_tokens": len(gen_tokens),
            "total_cells": sum(self.prime_banks),
            "persistent_state_mb": state_mb,
            "prefill_ms": round(prefill_sec * 1000.0, 2),
            "prefill_tok_s": round(inp.size(1) / prefill_sec, 1),
            "decode_ms_per_tok": round((decode_sec * 1000.0) / steps, 2),
            "decode_tok_s": round(steps / decode_sec, 1),
        }
        return out_ids, stats

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        prompt: Union[str, Sequence[Dict[str, str]]],
        system: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        occ_alpha: Optional[float] = None,
        use_think: bool = False,
        stream: bool = False,
        return_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        One-line ChatML generation helper using `VAULTv2_5StreamingEngine`.
        Accepts either a plain prompt `str` or an OpenAI/HF `messages` list (`[{"role": ..., "content": ...}]`),
        and supports `use_think=True` to trigger explicit `<think>...</think>` Chain-of-Thought reasoning.
        """
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        ids = []
        if isinstance(prompt, str):
            if system:
                ids.extend([im_start] + tokenizer.encode(f"system\n{system.strip()}", add_special_tokens=False) + [im_end])
                ids.extend(nl_ids)
            ids.extend([im_start] + tokenizer.encode(f"user\n{prompt.strip()}", add_special_tokens=False) + [im_end])
            ids.extend(nl_ids)
        else:
            if system and (not prompt or prompt[0].get("role") != "system"):
                ids.extend([im_start] + tokenizer.encode(f"system\n{system.strip()}", add_special_tokens=False) + [im_end])
                ids.extend(nl_ids)
            for msg in prompt:
                role = str(msg.get("role", "user")).strip()
                content = str(msg.get("content", "")).strip()
                ids.extend([im_start] + tokenizer.encode(f"{role}\n{content}", add_special_tokens=False) + [im_end])
                ids.extend(nl_ids)

        assistant_prefix = "assistant\n<think>\n" if use_think else "assistant\n"
        ids.extend([im_start] + tokenizer.encode(assistant_prefix, add_special_tokens=False))

        inp = torch.tensor([ids], dtype=torch.long, device=self.embed_tokens.weight.device)
        stream_buf = []
        printed_len = 0
        seed_prefix = "<think>\n" if use_think else ""
        if stream and seed_prefix:
            print(seed_prefix, end="", flush=True)

        def _printer(tid: int) -> None:
            nonlocal printed_len
            stream_buf.append(tid)
            txt = tokenizer.decode(stream_buf, skip_special_tokens=True)
            if len(txt) > printed_len:
                print(txt[printed_len:], end="", flush=True)
                printed_len = len(txt)

        res = self.stream_generate(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_ids=(im_end, eos_id),
            occ_alpha=occ_alpha,
            on_token=_printer if stream else None,
            return_stats=return_stats,
        )
        if stream:
            print(flush=True)

        if return_stats:
            out_ids, stats = res  # type: ignore[misc]
            gen_ids = out_ids[0, inp.size(1) :].tolist()
            text = (seed_prefix + tokenizer.decode(gen_ids, skip_special_tokens=True)).strip()
            return text, stats

        gen_ids = res[0, inp.size(1) :].tolist()  # type: ignore[index]
        return (seed_prefix + tokenizer.decode(gen_ids, skip_special_tokens=True)).strip()
