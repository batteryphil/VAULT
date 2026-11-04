# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# Rigorous Forward + Backward Parity & Speed Verification for FlashVAULT v2 & VAULTLMv2

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kernels.flash_vault_v2_triton import (
    flash_vault_v2_chunkwise,
    flash_vault_v2_recurrent_reference,
)
from src.vault_v2_model import VAULTLMv2


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_flash_vault_v2_parity():
    dev = "cuda:0"
    torch.manual_seed(2026)

    for dtype, name, tol_cos in [(torch.float32, "FP32", 0.99999), (torch.bfloat16, "BF16", 0.99990)]:
        B, H, G, R, L, D = 2, 12, 4, 4, 200, 64  # Non-chunk-aligned L=200 tests padding too!
        pq1 = F.normalize(torch.randn(B, H, L, D, device=dev, dtype=dtype), dim=-1).detach().requires_grad_(True)
        pk1 = F.normalize(torch.randn(B, G, L, D, device=dev, dtype=dtype), dim=-1).detach().requires_grad_(True)
        v1 = (torch.randn(B, G, L, D, device=dev, dtype=dtype) * 0.5).detach().requires_grad_(True)
        b1 = torch.rand(B, G, L, device=dev, dtype=dtype).clamp(0.05, 0.95).detach().requires_grad_(True)

        pq2 = pq1.detach().clone().requires_grad_(True)
        pk2 = pk1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        offs = [0, 17, 36, 59]
        sizes = [17, 19, 23, 29]
        cq = torch.stack([torch.randint(offs[r], offs[r] + sizes[r], (B, H, L), device=dev) for r in range(R)], dim=2)
        ck = torch.stack([torch.randint(offs[r], offs[r] + sizes[r], (B, G, L), device=dev) for r in range(R)], dim=2)
        dout = torch.randn(B, H, L, D, device=dev, dtype=dtype)

        y_ref = flash_vault_v2_recurrent_reference(pq1, pk1, v1, b1, cq, ck)
        (y_ref * dout).sum().backward()

        y_tri = flash_vault_v2_chunkwise(pq2, pk2, v2, b2, cq, ck, chunk_size=64)
        (y_tri * dout).sum().backward()

        cs_y = cos_sim(y_ref, y_tri)
        cs_dq = cos_sim(pq1.grad, pq2.grad)
        cs_dk = cos_sim(pk1.grad, pk2.grad)
        cs_dv = cos_sim(v1.grad, v2.grad)
        cs_db = cos_sim(b1.grad, b2.grad)
        print(f"[FlashVAULT v2 {name} H=12 G=4 L={L}] y={cs_y:.7f} | dQ={cs_dq:.7f} | dK={cs_dk:.7f} | dV={cs_dv:.7f} | dBeta={cs_db:.7f}")
        assert min(cs_y, cs_dq, cs_dk, cs_dv, cs_db) >= tol_cos

    # 256 identical tokens padding torture test in BF16
    B, H, G, R, L, D = 1, 12, 4, 4, 256, 64
    q_one = F.normalize(torch.ones(B, H, 1, D, device=dev, dtype=torch.bfloat16), dim=-1).expand(B, H, L, D).clone().requires_grad_(True)
    k_one = F.normalize(torch.ones(B, G, 1, D, device=dev, dtype=torch.bfloat16), dim=-1).expand(B, G, L, D).clone().requires_grad_(True)
    v_one = torch.ones(B, G, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    b_one = torch.full((B, G, L), 0.85, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cq0 = torch.tensor([0, 17, 36, 59], device=dev).view(1, 1, 4, 1).expand(B, H, 4, L).contiguous()
    ck0 = torch.tensor([0, 17, 36, 59], device=dev).view(1, 1, 4, 1).expand(B, G, 4, L).contiguous()
    y_id = flash_vault_v2_chunkwise(q_one, k_one, v_one, b_one, cq0, ck0)
    y_id.sum().backward()
    assert torch.isfinite(y_id).all() and torch.isfinite(v_one.grad).all()
    print("[FlashVAULT v2 256-Identical-Token BF16 Torture] ZERO NaNs!")


    # Verify FlashVAULT v2.2 beta-weighted consensus (`v2_1_consensus=True, v2_2_beta_occ=True`) in FP32 & BF16
    for dtype, name, tol_cos in [(torch.float32, "FP32-v2.2", 0.99999), (torch.bfloat16, "BF16-v2.2", 0.99990)]:
        B, H, G, R, L, D = 2, 12, 4, 4, 200, 64
        pq1 = F.normalize(torch.randn(B, H, L, D, device=dev, dtype=dtype), dim=-1).detach().requires_grad_(True)
        pk1 = F.normalize(torch.randn(B, G, L, D, device=dev, dtype=dtype), dim=-1).detach().requires_grad_(True)
        v1 = (torch.randn(B, G, L, D, device=dev, dtype=dtype) * 0.5).detach().requires_grad_(True)
        b1 = torch.rand(B, G, L, device=dev, dtype=dtype).clamp(0.02, 0.96).detach().requires_grad_(True)

        pq2 = pq1.detach().clone().requires_grad_(True)
        pk2 = pk1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        offs = [0, 17, 36, 59]
        sizes = [17, 19, 23, 29]
        cq = torch.stack([torch.randint(offs[r], offs[r] + sizes[r], (B, H, L), device=dev) for r in range(R)], dim=2)
        ck = torch.stack([torch.randint(offs[r], offs[r] + sizes[r], (B, G, L), device=dev) for r in range(R)], dim=2)
        dout = torch.randn(B, H, L, D, device=dev, dtype=dtype)

        y_ref = flash_vault_v2_recurrent_reference(
            pq1, pk1, v1, b1, cq, ck, v2_1_consensus=True, v2_2_beta_occ=True
        )
        (y_ref * dout).sum().backward()

        y_tri = flash_vault_v2_chunkwise(
            pq2, pk2, v2, b2, cq, ck, chunk_size=64, v2_1_consensus=True, ckpt_interval=8, v2_2_beta_occ=True
        )
        (y_tri * dout).sum().backward()

        cs_y = cos_sim(y_ref, y_tri)
        cs_dq = cos_sim(pq1.grad, pq2.grad)
        cs_dk = cos_sim(pk1.grad, pk2.grad)
        cs_dv = cos_sim(v1.grad, v2.grad)
        cs_db = cos_sim(b1.grad, b2.grad)
        print(f"[FlashVAULT {name} H=12 G=4 L={L}] y={cs_y:.7f} | dQ={cs_dq:.7f} | dK={cs_dk:.7f} | dV={cs_dv:.7f} | dBeta={cs_db:.7f}")
        assert min(cs_y, cs_dq, cs_dk, cs_dv, cs_db) >= tol_cos


def bench_vault_lm_v2():
    dev = "cuda:0"
    for v2_2_flag, label in [(False, "VAULTLMv2.0"), (True, "VAULTLMv2.2")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = VAULTLMv2(
            vocab_size=49152,
            d_model=768,
            n_layers=16,
            n_q_heads=12,
            n_vault_groups=4,
            d_ff=2048,
            grad_checkpoint="half",
            v2_2_mode=v2_2_flag,
        ).to(dev)
        print(f"[{label}] Parameters: {model.count_parameters():,}")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x_chk = torch.randint(0, 49152, (2, 128), device=dev)
            y_chk = torch.randint(0, 49152, (2, 128), device=dev)
            w_chk = torch.rand(2, 128, device=dev) + 0.2
            logits_full = model(x_chk, step_ratio=1.0)
            ce_full = (
                F.cross_entropy(logits_full.float().view(-1, 49152), y_chk.view(-1), reduction="none") * w_chk.view(-1)
            ).sum() / w_chk.sum()
            ce_chunked = model.forward_loss(x_chk, y_chk, loss_weights=w_chk, step_ratio=1.0, logit_chunk_tokens=64)
            diff_loss = abs(ce_full.item() - ce_chunked.item())
        print(f"[{label} Chunked-CE Parity] abs_diff = {diff_loss:.2e}")
        assert diff_loss < 1e-5

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        B = 2
        x = torch.randint(0, 49152, (B, 2048), device=dev)
        y = torch.randint(0, 49152, (B, 2048), device=dev)
        w = torch.ones((B, 2048), device=dev, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.forward_loss(x, y, loss_weights=w, step_ratio=1.0)
        loss.backward()
        zero_grads = [
            n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
            and "decay_logit" not in n
        ]
        assert len(zero_grads) == 0, f"Unexpected zero-grad params in {label}: {zero_grads[:8]}"
        opt.step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model.forward_loss(x, y, loss_weights=w, step_ratio=1.0)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 2.0
        print(
            f"[{label} B={B} L=2048 gc=half] {B * 2048 / dt:,.1f} tok/s ({dt * 1000:.0f} ms/step) "
            f"| Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        del model, opt, x, y, w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_flash_vault_v2_parity()
    bench_vault_lm_v2()
    print("\nALL VAULT V2.0 & V2.2 PARITY, GRADIENT & THROUGHPUT CHECKS PASSED 100%!")

