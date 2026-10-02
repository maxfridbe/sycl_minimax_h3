#!/usr/bin/env python3
"""compare.py A B --out PREFIX [--labels "Q8_0" "int8"]

Two clips made from the same prompt, seed and settings by two denoisers (tags A and B in /out):
  - PREFIX.sbs.mp4    the two videos side by side (A's audio)
  - PREFIX.grid.png   four matching frames, A on top, B below
  - PREFIX.json       per-step distance between the two denoisers' estimates (from the H3X_DUMP_STEPS dumps),
                      per-frame PSNR of the finished videos
Runs on the CPU inside the production image (torch + PyAV are there). No GPU.

How to read the numbers: "cosine" is the direction agreement of the two step outputs (1.0 = identical);
PSNR is the usual picture difference in dB (above ~35 dB two frames look the same; 20-25 dB is clearly different
pictures - which is normal when a sampler takes a different path, it does not by itself mean worse).
"""
import argparse, glob, json, os, sys
import numpy as np, torch, av

def frames_of(path):
    c = av.open(path)
    out = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return out

def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 12 + 9 * len(text), 26], fill=(0, 0, 0)); d.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(im)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs=2, default=None); ap.add_argument("--dir", default="/out")
    a = ap.parse_args()
    la, lb = a.labels or (a.a, a.b)
    res = {"a": a.a, "b": a.b, "steps": [], "frames": {}}
    # ---- per-step estimates
    for fa in sorted(glob.glob(f"{a.dir}/{a.a}.step*.pt")):
        fb = fa.replace(f"/{a.a}.step", f"/{a.b}.step")
        if not os.path.exists(fb): continue
        ta, tb = torch.load(fa), torch.load(fb)
        row = {"step": int(fa[-5:-3])}
        for name, x, y in zip(("video", "audio"), ta, tb):
            x, y = x.flatten().double(), y.flatten().double()
            row[name + "_cosine"] = round(float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-30)), 5)
            row[name + "_rel_err"] = round(float((x - y).norm() / (x.norm() + 1e-30)), 5)
        res["steps"].append(row)
    # ---- finished videos
    A, B = frames_of(f"{a.dir}/{a.a}.mp4"), frames_of(f"{a.dir}/{a.b}.mp4")
    n = min(len(A), len(B))
    psnr = []
    for i in range(n):
        mse = np.mean((A[i].astype(np.float64) - B[i].astype(np.float64)) ** 2)
        psnr.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    res["frames"] = {"count": n, "size": list(A[0].shape[:2][::-1]), "psnr_mean_db": round(float(np.mean(psnr)), 2),
                     "psnr_min_db": round(float(np.min(psnr)), 2), "psnr_first_db": round(float(psnr[0]), 2),
                     "psnr_last_db": round(float(psnr[-1]), 2)}
    json.dump(res, open(f"{a.out}.json", "w"), indent=1)
    # ---- grid: 4 matching frames
    from PIL import Image
    idx = [int(n * f) for f in (0.05, 0.35, 0.65, 0.95)]
    top = np.concatenate([label(A[i], f"{la}  frame {i}") for i in idx], axis=1)
    bot = np.concatenate([label(B[i], f"{lb}  frame {i}") for i in idx], axis=1)
    grid = Image.fromarray(np.concatenate([top, bot], axis=0))
    grid.thumbnail((2400, 2400)); grid.save(f"{a.out}.grid.png")
    # ---- side by side, with A's audio
    src = av.open(f"{a.dir}/{a.a}.mp4"); vin = src.streams.video[0]
    out = av.open(f"{a.out}.sbs.mp4", "w")
    vs = out.add_stream("libx264", rate=vin.average_rate); vs.width = A[0].shape[1] * 2; vs.height = A[0].shape[0]
    vs.pix_fmt = "yuv420p"; vs.options = {"crf": "18"}
    astream = None
    if src.streams.audio:
        ain = src.streams.audio[0]
        astream = out.add_stream("aac", rate=ain.rate)
    for i in range(n):
        f = av.VideoFrame.from_ndarray(np.concatenate([label(A[i], la), label(B[i], lb)], axis=1), format="rgb24")
        for p in vs.encode(f): out.mux(p)
    for p in vs.encode(): out.mux(p)
    if astream is not None:
        for fr in src.decode(audio=0):
            fr.pts = None
            for p in astream.encode(fr): out.mux(p)
        for p in astream.encode(): out.mux(p)
    out.close(); src.close()
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
