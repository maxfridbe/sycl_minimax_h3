#!/usr/bin/env python3
"""Parity and speed of h3sycl.int8_linear against comfy-kitchen's eager int8_linear, on the card.
Random weights and activations at the DiT's real shapes; with and without the ConvRot rotation."""
import sys, time, torch
sys.path.insert(0, "/work/kernels")
import h3sycl
from comfy_kitchen.backends import eager
from comfy_kitchen.backends.eager.quantization import _build_hadamard


def exact(x, w, ws, bias, convrot, g):
    """The same arithmetic in float32 throughout (rotation, row scale, rounding, product): what both should approach."""
    xf = x.float()
    if convrot:
        h = _build_hadamard(g, device=x.device, dtype=torch.float32)
        xf = (xf.view(xf.shape[0], -1, g) @ h).reshape(xf.shape)
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = torch.round(xf / scale).clamp(-128, 127)
    return (q @ w.float().T) * (scale * ws.float().reshape(1, -1)) + bias.float()


dev = torch.device("xpu")
torch.manual_seed(0)
ok = True
def sync(): torch.xpu.synchronize()
for name, K, N in (("qkv", 5376, 21504), ("fc1", 5376, 28672), ("fc2", 14336, 5376), ("out", 7168, 5376)):
    for M in (1024, 16384):
        for convrot in (False, True):
            x = (torch.randn(M, K, device=dev) * 0.7).to(torch.bfloat16)
            w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
            # the checkpoint's form (one scale per output row, [N, 1]) with the rotation; a scalar without
            ws = (torch.rand(N, 1, device=dev) * 0.004 + 0.001) if convrot else torch.tensor(0.0031, device=dev)
            bias = torch.randn(N, device=dev).to(torch.bfloat16)
            kw = dict(bias=bias, out_dtype=torch.bfloat16, convrot=convrot, convrot_groupsize=256)
            ref = eager.int8_linear(x, w, ws, **kw).float(); sync()
            got = h3sycl.int8_linear(x, w, ws, **kw).float(); sync()
            rel = float((got - ref).norm() / ref.norm())
            cos = float(torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0))
            if M == 1024:
                ex = exact(x, w, ws, bias, convrot, 256); sync()
                print(f"{name} M={M:5d} convrot={int(convrot)}  vs exact float32: eager rel err {float((ref - ex).norm() / ex.norm()):.2e}"
                      f"  h3sycl rel err {float((got - ex).norm() / ex.norm()):.2e}", flush=True)
                del ex
            # timing: 3 calls each, synchronized
            def t(fn):
                fn(); sync(); t0 = time.time()
                for _ in range(3): fn()
                sync(); return (time.time() - t0) / 3 * 1e3
            te = t(lambda: eager.int8_linear(x, w, ws, **kw)); ts = t(lambda: h3sycl.int8_linear(x, w, ws, **kw))
            tb = None
            if not convrot:
                wf = w.to(torch.bfloat16)
                tb = t(lambda: torch.nn.functional.linear(x, wf, bias))
                del wf
            good = rel < 2e-2 and cos > 0.999
            ok &= good
            print(f"{name} M={M:5d} convrot={int(convrot)}  rel err {rel:.2e}  cosine {cos:.6f}  eager {te:8.1f} ms  h3sycl {ts:7.1f} ms"
                  + (f"  bf16 linear {tb:7.1f} ms" if tb else "") + ("" if good else "   <-- MISMATCH"), flush=True)
            del x, w, bias, ref, got
            torch.xpu.empty_cache()
print("int8_linear parity:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
