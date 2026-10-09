"""Real-ESRGAN release weights (.pth) -> .safetensors with the same tensor names, for the engine's pixel upscaler
(engine/h3-core/src/esrgan.rs). Writes the format directly (an 8-byte header length, a JSON header, the raw
little-endian float32 data), so only PyTorch is needed to read the .pth.

    python esrgan_to_safetensors.py realesr-animevideov3.pth realesr-animevideov3.safetensors
"""
import json
import struct
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)
sd = sd.get("params_ema", sd.get("params", sd))
header, blobs, off = {}, [], 0
for name in sorted(sd):
    t = sd[name].detach().to(torch.float32).contiguous()
    b = t.numpy().tobytes()
    header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
    blobs.append(b)
    off += len(b)
header["__metadata__"] = {"source": src.rsplit("/", 1)[-1]}
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * ((8 - len(h) % 8) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(h)))
    f.write(h)
    for b in blobs:
        f.write(b)
print(f"{dst}: {len(sd)} tensors, {off / 1e6:.1f} MB")
