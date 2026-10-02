#!/usr/bin/env python3
"""h2d_bench.py - why does the denoiser reach the card at 0.4 GB/s?  Times host -> device copies in PyTorch XPU
from different kinds of host memory, in one 2 GiB piece and in 64 MiB pieces."""
import os, time, numpy as np, torch
dev = torch.device("xpu")
GB = 1 << 30
def rate(label, fn, nbytes):
    torch.xpu.synchronize(); t0 = time.time(); out = fn(); torch.xpu.synchronize(); dt = time.time() - t0
    print(f"{label:58s} {nbytes/GB/dt:6.2f} GB/s  ({dt:5.2f} s)", flush=True); return out
torch.empty(1 << 20, dtype=torch.uint8).to(dev)   # warm up the device
n = 2 * GB
a = torch.empty(n, dtype=torch.uint8); a.fill_(7)                      # ordinary (pageable) RAM, already touched
rate("pageable RAM, one 2 GiB tensor", lambda: a.to(dev), n)
parts = list(a.split(64 << 20))
rate("pageable RAM, 32 x 64 MiB tensors", lambda: [p.to(dev) for p in parts], n)
rate("pageable RAM, 32 x 64 MiB, non_blocking", lambda: [p.to(dev, non_blocking=True) for p in parts], n)
try:
    b = a.pin_memory()
    rate("pin_memory() copy, one 2 GiB tensor", lambda: b.to(dev), n)
    rate("pin_memory(), non_blocking", lambda: b.to(dev, non_blocking=True), n)
except Exception as e:
    print("pin_memory():", type(e).__name__, str(e)[:120], flush=True)
f = "/models/engines/minimax_h3_fl2va_pruned-Q8_0.gguf"
m = np.memmap(f, dtype=np.uint8, mode="r", offset=4 * GB, shape=(n,))
t = torch.from_numpy(m)
rate("file mmap (first touch: page cache or SSD), 2 GiB", lambda: t.to(dev), n)
rate("file mmap again (now cached), 2 GiB", lambda: t.to(dev), n)
# plain read() into RAM, for the file's own speed
fd = os.open(f, os.O_RDONLY); buf = bytearray(256 << 20); t0 = time.time(); got = 0
os.lseek(fd, 8 * GB, 0)
while got < n: got += os.readv(fd, [buf])
print(f"{'plain read() of 2 GiB of the file (another region)':58s} {n/GB/(time.time()-t0):6.2f} GB/s", flush=True)
