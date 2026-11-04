# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-VAULT-Commercial
# Dual License: GNU AGPL-3.0-or-later (Free / Non-Profit / Open-Source) OR Commercial License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# HuggingFace `AutoConfig` definition for VAULT v2.5 (180.6M, Co-Prime Kenyon Cell Architecture).

from typing import Dict, List, Sequence, Tuple, Union
from transformers import PretrainedConfig

MITOSIS_PRESETS: Dict[int, Tuple[int, int, int, int]] = {
    220: (47, 53, 59, 61),
    508: (113, 127, 131, 137),
    942: (229, 233, 239, 241),
    2002: (491, 499, 503, 509),
}


def resolve_prime_banks(cells_or_primes: Union[int, Sequence[int]]) -> Tuple[int, int, int, int]:
    if isinstance(cells_or_primes, int):
        if cells_or_primes not in MITOSIS_PRESETS:
            raise ValueError(
                f"Unknown cell preset {cells_or_primes}. Supported presets: {sorted(MITOSIS_PRESETS.keys())}"
            )
        return MITOSIS_PRESETS[cells_or_primes]
    banks = tuple(int(p) for p in cells_or_primes)
    if len(banks) != 4:
        raise ValueError(f"Expected 4 co-prime bank sizes, got {banks}")
    return banks  # type: ignore[return-value]


class VAULTConfig(PretrainedConfig):
    model_type = "vault_v2_5"

    def __init__(
        self,
        vocab_size: int = 49152,
        d_model: int = 768,
        n_layers: int = 16,
        n_q_heads: int = 12,
        n_vault_groups: int = 4,
        head_dim: int = 64,
        d_ff: int = 2048,
        chunk_size: int = 64,
        prime_banks: List[int] = None,
        ema_alpha: float = 0.00048828125,
        occ_alpha: float = 0.5,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        im_start_token_id: int = 1,
        im_end_token_id: int = 2,
        **kwargs,
    ):
        if prime_banks is None:
            prime_banks = [47, 53, 59, 61]
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_q_heads = int(n_q_heads)
        self.n_vault_groups = int(n_vault_groups)
        self.head_dim = int(head_dim)
        self.d_ff = int(d_ff)
        self.chunk_size = int(chunk_size)
        self.prime_banks = [int(p) for p in prime_banks]
        self.ema_alpha = float(ema_alpha)
        self.occ_alpha = float(occ_alpha)
        self.im_start_token_id = int(im_start_token_id)
        self.im_end_token_id = int(im_end_token_id)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
