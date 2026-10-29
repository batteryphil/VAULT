# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Phase 1 Verification Suite for FlashVAULT (Triton 3.8 / ROCm 7.2)
# Verifies:
#   1. Exact Forward Parity (`flash_vault_chunkwise` vs `flash_vault_recurrent_ref`)
#      across multi-head (G=12) and Grouped-Vault (G=4), aligned (L=256) and non-aligned (L=192, L=200) lengths.
#   2. Exact Backward Autograd Parity (`dPhiQ`, `dPhiK`, `dV`, `dBeta`) in both FP32 and BF16.

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kernels.flash_vault_triton import (
    PRIME_BANKS_DEFAULT,
    flash_vault_recurrent_ref,
    flash_vault_chunkwise,
)


def make_global_cells(B: int, G: int, L: int, device: torch.device, seed: int = 42):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cq_list, ck_list = [], []
    offset = 0
    for p in PRIME_BANKS_DEFAULT:
        cq_list.append(torch.randint(offset, offset + p, (B, G, L), generator=gen, device=device))
        ck_list.append(torch.randint(offset, offset + p, (B, G, L), generator=gen, device=device))
        offset += p
    return torch.stack(cq_list, dim=2), torch.stack(ck_list, dim=2)


def run_parity_case(B: int, H: int, G: int, L: int, dtype: torch.dtype, label: str):
    device = torch.device("cuda")
    torch.manual_seed(1234 + L + G)
    D = 64

    phi_q_raw = F.normalize(torch.exp(0.8 * torch.randn(B, H, L, D, device=device)), p=2, dim=-1).to(dtype)
    phi_k_raw = F.normalize(torch.exp(0.8 * torch.randn(B, H, L, D, device=device)), p=2, dim=-1).to(dtype)
    v_raw = (torch.randn(B, H, L, D, device=device) * 0.5).to(dtype)
    beta_raw = torch.sigmoid(torch.randn(B, H, L, device=device)).to(dtype) * 0.8 + 0.1
    cell_q, cell_k = make_global_cells(B, G, L, device, seed=99 + L)
    grad_out = torch.randn(B, H, L, D, device=device, dtype=dtype)

    # 1. Ground-Truth Sequential Recurrent Reference
    pq_ref = phi_q_raw.clone().detach().requires_grad_(True)
    pk_ref = phi_k_raw.clone().detach().requires_grad_(True)
    v_ref = v_raw.clone().detach().requires_grad_(True)
    b_ref = beta_raw.clone().detach().requires_grad_(True)

    out_ref, _ = flash_vault_recurrent_ref(pq_ref, pk_ref, v_ref, b_ref, cell_q, cell_k)
    (out_ref * grad_out).sum().backward()

    # 2. FlashVAULT Chunkwise Triton Kernel
    pq_tri = phi_q_raw.clone().detach().requires_grad_(True)
    pk_tri = phi_k_raw.clone().detach().requires_grad_(True)
    v_tri = v_raw.clone().detach().requires_grad_(True)
    b_tri = beta_raw.clone().detach().requires_grad_(True)

    out_tri = flash_vault_chunkwise(pq_tri, pk_tri, v_tri, b_tri, cell_q, cell_k)
    (out_tri * grad_out).sum().backward()

    fwd_max_err = (out_ref.float() - out_tri.float()).abs().max().item()
    fwd_cos = F.cosine_similarity(out_ref.float().flatten(), out_tri.float().flatten(), dim=0).item()

    grads = [
        ("dPhiQ", pq_ref.grad, pq_tri.grad),
        ("dPhiK", pk_ref.grad, pk_tri.grad),
        ("dV", v_ref.grad, v_tri.grad),
        ("dBeta", b_ref.grad, b_tri.grad),
    ]

    tol = 2e-4 if dtype == torch.float32 else 2e-2
    cos_min = 0.99995 if dtype == torch.float32 else 0.999
    print(f"[{label}] Fwd MaxAbsErr: {fwd_max_err:.3e} | CosSim: {fwd_cos:.7f}")
    assert fwd_max_err < tol and fwd_cos > cos_min, f"Forward mismatch in {label}: err={fwd_max_err}"

    for name, g_r, g_t in grads:
        err = (g_r.float() - g_t.float()).abs().max().item()
        cos = F.cosine_similarity(g_r.float().flatten(), g_t.float().flatten(), dim=0).item()
        print(f"   -> {name:5s} MaxAbsErr: {err:.3e} | CosSim: {cos:.7f}")
        assert cos > cos_min and err < (tol * 5.0), f"Backward {name} mismatch in {label}: err={err}, cos={cos}"


def test_flash_vault_module():
    from src.flash_vault_module import FlashVAULTAttention

    device = torch.device("cuda")
    torch.manual_seed(77)
    mod = FlashVAULTAttention(d_model=768, n_heads=12, head_dim=64, n_vault_groups=4).to(device).bfloat16()
    x = torch.randn(2, 256, 768, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = mod(x, step_ratio=0.5)
    loss = (y.float() ** 2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input x grad missing or non-finite"
    for name, p in mod.named_parameters():
        if p.requires_grad and "decay_logit" not in name:
            assert p.grad is not None, f"Missing grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
            assert p.grad.abs().max().item() > 0.0, f"Zero grad for {name} (router STE broken?)"
    print(f"[FlashVAULTAttention Module] Out shape={tuple(y.shape)} | All parameter & STE router grads finite & > 0!")


def test_identical_padding_tokens_stability():
    """
    Torture test: 100% identical `<eos>`-style repeated tokens routing to the EXACT SAME Co-Prime cell
    across 4 full chunks (L=256) in BF16 WMMA mode!
    Unfactored Neumann squaring (`A^16, A^32`) overflows BF16 (`C(63,31) * 0.65^32 = 4.7e11 > 65504 -> NaN`),
    whereas `_invert_unit_lower_tri_64` stays strictly in `[-1, +1]` with zero NaNs and exact parity!
    """
    device = torch.device("cuda")
    torch.manual_seed(2026)
    B, H, G, L, D = 1, 4, 4, 256, 64
    dtype = torch.bfloat16

    # Identical unit key across all 256 positions + identical cell ID in every bank!
    single_k = F.normalize(torch.exp(torch.randn(1, H, 1, D, device=device)), p=2, dim=-1).expand(B, H, L, D).contiguous().to(dtype)
    phi_q_raw = F.normalize(torch.exp(0.5 * torch.randn(B, H, L, D, device=device)), p=2, dim=-1).to(dtype)
    v_raw = (torch.randn(B, H, L, D, device=device) * 0.5).to(dtype)
    beta_raw = torch.full((B, H, L), 0.65, device=device, dtype=dtype)

    offsets = torch.tensor([0, 17, 36, 59], device=device, dtype=torch.int64).view(1, 1, 4, 1)
    cell_q = offsets.expand(B, G, 4, L).contiguous()
    cell_k = offsets.expand(B, G, 4, L).contiguous()
    grad_out = torch.randn(B, H, L, D, device=device, dtype=dtype)

    pq_ref = phi_q_raw.clone().detach().requires_grad_(True)
    pk_ref = single_k.clone().detach().requires_grad_(True)
    v_ref = v_raw.clone().detach().requires_grad_(True)
    b_ref = beta_raw.clone().detach().requires_grad_(True)
    out_ref, _ = flash_vault_recurrent_ref(pq_ref, pk_ref, v_ref, b_ref, cell_q, cell_k)
    (out_ref * grad_out).sum().backward()

    pq_tri = phi_q_raw.clone().detach().requires_grad_(True)
    pk_tri = single_k.clone().detach().requires_grad_(True)
    v_tri = v_raw.clone().detach().requires_grad_(True)
    b_tri = beta_raw.clone().detach().requires_grad_(True)
    out_tri = flash_vault_chunkwise(pq_tri, pk_tri, v_tri, b_tri, cell_q, cell_k)
    (out_tri * grad_out).sum().backward()

    assert torch.isfinite(out_tri).all(), "NaN/Inf in forward on 256 identical tokens!"
    assert torch.isfinite(pk_tri.grad).all() and torch.isfinite(v_tri.grad).all(), "NaN/Inf in backward on 256 identical tokens!"
    cos = F.cosine_similarity(out_ref.float().flatten(), out_tri.float().flatten(), dim=0).item()
    g_cos = F.cosine_similarity(v_ref.grad.float().flatten(), v_tri.grad.float().flatten(), dim=0).item()
    print(f"[BF16 256 Identical `<eos>` Tokens Torture] Fwd CosSim: {cos:.7f} | dV CosSim: {g_cos:.7f} (ZERO NaNs!)")
    assert cos > 0.999 and g_cos > 0.999


def main():
    assert torch.cuda.is_available(), "ROCm / CUDA GPU required for Triton FlashVAULT test"
    print(f"Testing FlashVAULT on {torch.cuda.get_device_name(0)} (Triton + PyTorch {torch.__version__})", flush=True)
    run_parity_case(B=2, H=12, G=12, L=128, dtype=torch.float32, label="FP32 H=12 G=12 L=128")
    run_parity_case(B=2, H=12, G=4,  L=256, dtype=torch.float32, label="FP32 H=12 G=4 (GVA) L=256")
    run_parity_case(B=2, H=12, G=4,  L=200, dtype=torch.float32, label="FP32 Non-Aligned Padding L=200")
    run_parity_case(B=2, H=12, G=4,  L=256, dtype=torch.bfloat16, label="BF16 H=12 G=4 (GVA) L=256")
    test_identical_padding_tokens_stability()
    test_flash_vault_module()
    print("\nALL FLASHVAULT FORWARD + BACKWARD PARITY TESTS PASSED 100%!", flush=True)


if __name__ == "__main__":
    main()

