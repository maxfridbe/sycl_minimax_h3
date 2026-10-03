#!/usr/bin/env python3
"""h3 - MiniMax H3 video generation CLI for the Intel Arc Pro B70.

Standalone CLI. ComfyUI never runs as an application - its H3 model code
(comfy/ldm/minimax/*) and the ComfyUI-GGUF dequant ops are imported as
libraries. No server, no node graph, no browser.

Runs inside the h3-xpu image because the HOST has no Intel level-zero runtime
(torch.xpu sees 0 devices there); the image carries Intel 26.31 userspace.
"""
import argparse, os, sys, time

COMFY = "/comfy"
MODELS = "/models"

# ComfyUI-GGUF uses relative imports, so it must be imported as a PACKAGE.
# Its real directory name contains a hyphen (illegal in a module name), so the
# wrapper bind-mounts it at /pkgs/comfyui_gguf.
sys.path.insert(0, COMFY)
sys.path.insert(0, "/pkgs")


def _t(msg, t0):
    """Stage timing line. Appends VRAM (current / peak-since-last-checkpoint /
    reserved) and host peak RSS so every run records its memory profile."""
    extra = ""
    try:
        import torch, resource
        if torch.xpu.is_available():
            g = 1024 ** 3
            a = torch.xpu.memory_allocated() / g
            p = torch.xpu.max_memory_allocated() / g
            r = torch.xpu.memory_reserved() / g
            torch.xpu.reset_peak_memory_stats()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
            extra = f"   vram {a:5.2f}G peak {p:5.2f}G resv {r:5.2f}G  rss {rss:4.1f}G"
    except Exception:
        pass
    print(f"  {msg:<34} {time.time()-t0:7.1f}s{extra}", flush=True)


def cmd_info(args):
    import torch
    print("=== device ===")
    print("  torch:", torch.__version__)
    ok = torch.xpu.is_available()
    print("  xpu available:", ok)
    if ok:
        p = torch.xpu.get_device_properties(0)
        print("  device:", torch.xpu.get_device_name(0))
        print("  VRAM:", round(p.total_memory / 1024**3, 2), "GiB")
    import comfy.model_management as mm
    print("  comfy device:", mm.get_torch_device())

    print("\n=== models ===")
    for label, path in _paths().items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / 1024**3 if exists and os.path.isfile(path) else 0
        print(f"  {'OK ' if exists else 'MISSING'} {label:<12} {size:6.2f} GB  {path}")

    print("\n=== GGUF headers ===")
    import gguf
    from collections import Counter
    for label in ("dit", "te"):
        p = _paths()[label]
        if not os.path.exists(p):
            continue
        r = gguf.GGUFReader(p)
        arch = None
        for f in r.fields.values():
            if f.name == "general.architecture":
                arch = bytes(f.parts[f.data[0]]).decode()
        print(f"  {label}: arch={arch}  tensors={len(r.tensors)}")
        if label == "dit":
            aud = [t.name for t in r.tensors if "audio" in t.name.lower()]
            print(f"       audio tensors: {len(aud)} -> {aud}")


def _paths():
    return {
        "dit":       f"{MODELS}/MiniStack/diffusion_models/MiniMax-H3-FL2VA-pruned-Q4_K_M.gguf",
        "te":        f"{MODELS}/MiniStack/text_encoders/qwen3vl-4b-h3student-Q4_K_M.gguf",
        "te_teacher": f"{MODELS}/teacher/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "adapter":   f"{MODELS}/MiniStack/text_encoders/te_adapter_v1.safetensors",
        "tokenizer": f"{MODELS}/MiniStack/text_encoders/h3_tokenizer",
        "vae_video": f"{MODELS}/Comfy-Org-MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors",
        "vae_audio": f"{MODELS}/Comfy-Org-MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors",
    }


def cmd_loadtest(args):
    """Load the DiT (and optionally the VAEs) to verify the GGUF path works."""
    import torch, comfy.sd, comfy.model_management as mm
    from comfyui_gguf.loader import gguf_sd_loader
    from comfyui_gguf.ops import GGMLOps
    P = _paths()

    print("=== DiT via gguf_sd_loader ===")
    t0 = time.time()
    dit_path = getattr(args, "dit", None) or P["dit"]
    print(f"  engine : {os.path.basename(dit_path)} ({os.path.getsize(dit_path)/2**30:.2f} GiB)", flush=True)
    sd, extra = _dit_state_dict(dit_path)
    _t(f"read {len(sd)} tensors", t0)
    print("  arch_str:", extra.get("arch_str"))

    print("\n=== instantiate model ===")
    t0 = time.time()
    ops = GGMLOps()
    ops.Linear.dequant_dtype = None
    ops.Linear.patch_dtype = None
    import inspect
    kwargs = {}
    if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters:
        kwargs["metadata"] = extra.get("metadata", {})
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}, **kwargs)
    if model is None:
        print("  *** model type NOT detected - unsupported UNET ***")
        sys.exit(2)
    _t("model built", t0)
    print("  model class:", type(model.model).__name__)
    print("  diffusion model:", type(model.model.diffusion_model).__name__)
    print("  dtype:", model.model.get_dtype())



def _load_gguf_native(path):
    """Read a GGUF whose keys are already comfy-native (arch-less files quantized
    with the ComfyUI-patched llama.cpp, e.g. the unsloth teacher TE). Mirrors the
    tensor-wrapping loop of ComfyUI-GGUF's gguf_sd_loader, minus its
    architecture detection, which only knows diffusion/llama layouts."""
    import gguf, torch
    from comfyui_gguf.ops import GGMLTensor
    from comfyui_gguf.dequant import dequantize_tensor
    reader = gguf.GGUFReader(path)
    sd, qtypes, biggest, big_n = {}, {}, None, -1
    for t in reader.tensors:
        tt = torch.from_numpy(t.data)                       # mmap-backed
        shape = torch.Size(tuple(int(v) for v in reversed(t.shape)))
        if t.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            tt = tt.view(*shape)
        sd[t.name] = GGMLTensor(tt, tensor_type=t.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and t.tensor_type == gguf.GGMLQuantizationType.BF16:
            sd[t.name] = dequantize_tensor(sd[t.name], dtype=torch.float32)
        n = 1
        for v in shape:
            n *= int(v)
        if n > big_n:
            big_n, biggest = n, t.name
        k = getattr(t.tensor_type, "name", repr(t.tensor_type))
        qtypes[k] = qtypes.get(k, 0) + 1
    if biggest:
        sd[biggest].is_largest_weight = True
    print("  gguf qtypes:", ", ".join(f"{k} ({v})" for k, v in qtypes.items()), flush=True)
    return sd


def _student_te_classes():
    """Rebuild the missing `h3_small_te` custom node.

    Per the MiniStack README the loader must: load the GGUF via ComfyUI-GGUF's
    gguf_clip_loader into comfy-native Llama2_(Qwen3VL_4BConfig) with GGMLOps,
    keep the H3 HF tokenizer + an Identity final-norm, and apply the
    2560->4096->5120 adapter. Nobody ships this node, so we reconstruct it.
    """
    import comfy.sd1_clip
    import comfy.text_encoders.minimax as mm

    class StudentQwen3VL(mm.MiniMaxQwen3VL):
        model_type = "qwen3vl_4b"        # student, not the 32B teacher

    class StudentClipModel(mm.MiniMaxH3ClipModel):
        def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
            comfy.sd1_clip.SDClipModel.__init__(
                self, device=device, layer="last", layer_idx=None, textmodel_json_config={},
                dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                model_class=StudentQwen3VL, enable_attention_masks=False,
                return_attention_masks=False, model_options=model_options)

    class StudentTEModel(comfy.sd1_clip.SD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                             clip_model=StudentClipModel, model_options=model_options)

    return StudentTEModel


def cmd_teload(args):
    """Load the distilled 4B student text encoder + adapter."""
    import torch, comfy.sd1_clip
    from comfyui_gguf.loader import gguf_clip_loader
    from comfyui_gguf.ops import GGMLOps
    from safetensors.torch import load_file
    P = _paths()

    print("=== student GGUF ===")
    t0 = time.time()
    sd = gguf_clip_loader(P["te"])
    if isinstance(sd, tuple):
        sd = sd[0]
    _t(f"read {len(sd)} tensors", t0)
    ks = list(sd.keys())
    print("  sample keys:", ks[:4])

    print("\n=== adapter ===")
    ad = load_file(P["adapter"])
    for k, v in ad.items():
        print(f"  {k:<40} {tuple(v.shape)}  {v.dtype}")

    print("\n=== build student TE ===")
    t0 = time.time()
    ops = GGMLOps()
    ops.Linear.dequant_dtype = None
    ops.Linear.patch_dtype = None
    TE = _student_te_classes()
    model_options = {"custom_operations": ops,
                     "model_config": {"final_norm": False}}
    try:
        te = TE(device="cpu", dtype=torch.float16, model_options=model_options)
        _t("constructed", t0)
        missing, unexpected = te.load_sd(sd)
        print(f"  missing: {len(missing)}  unexpected: {len(unexpected)}")
        if missing[:5]:   print("   missing[:5]:", missing[:5])
        if unexpected[:5]: print("   unexpected[:5]:", unexpected[:5])
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  BUILD FAILED:", type(e).__name__, e)



def _load_frame(path):
    """Image (png/jpg) or video (.mp4 etc -> its LAST frame) -> float tensor [1,H,W,3] in 0..1."""
    import numpy as np, torch
    if path.lower().endswith((".mp4", ".mkv", ".mov", ".webm")):
        import av
        last = None
        with av.open(path) as c:
            for fr in c.decode(video=0):
                last = fr
        if last is None:
            raise ValueError(f"no video frames in {path}")
        arr = last.to_ndarray(format="rgb24")
    else:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr.astype(np.float32) / 255.0)[None]


def _canon_audio(w):
    """-> [B, C, L]. The H3 audio VAE works in [B, L, C] (its decode prints (1, 482400, 2)),
    comfy AUDIO dicts are [B, C, L]; tell them apart by which axis is 1 or 2 wide."""
    while w.ndim > 3:
        w = w[0]
    if w.ndim == 1:
        w = w[None]
    if w.ndim == 2:
        w = w[None]
    if w.shape[-1] in (1, 2) and w.shape[-2] > 8:
        w = w.movedim(-1, 1)
    return w


def _load_audio_tail(path, seconds):
    """Last `seconds` of a media file's first audio stream as a float tensor [1, C, L] + rate."""
    import av, numpy as np, torch
    c = av.open(path)
    st = next(s for s in c.streams if s.type == "audio")
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=st.rate)
    chunks = []
    for fr in c.decode(st):
        for r in rs.resample(fr):
            chunks.append(r.to_ndarray())        # [C, n]
    c.close()
    a = np.concatenate(chunks, axis=1)
    a = a[:, -int(seconds * st.rate):]
    return torch.from_numpy(a)[None].float(), int(st.rate)


def _load_audio_full(path):
    """Whole first audio stream of a file as a comfy AUDIO dict."""
    w, sr = _load_audio_tail(path, 1e9)
    return {"waveform": w, "sample_rate": sr}


def _ref_cache_key(args):
    """Identity of the reference set, for the conditioning cache. Path plus mtime plus size:
    re-encoding is minutes, stat-ing three files is free."""
    import os
    parts = []
    for p in list(getattr(args, "ref_image", None) or []) + list(getattr(args, "ref_audio", None) or []):
        try:
            st = os.stat(p)
            parts.append(f"{p}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(f"{p}:missing")
    return "|".join(parts) + f"|{getattr(args, 'ref_image_size', 'match')}"


def _build_refs(args, P):
    """ref2va conditioning: reference images and reference audio.

    This is NOT the same mechanism as --first-frame / --first-audio. A keyframe pins real
    pixels or real audio at frame 0, so the shot has to start there and the pinned audio is
    audible at the head of the clip. A REFERENCE is presented to the text encoder as
    <Picture i> / <Audio j> and its latent rides through every sampling step, conditioning
    identity and voice without dictating any frame. Cloning a voice from a recording is this
    path, not the keyframe one.

    Mirrors MiniMaxH3ReferenceToVideo in comfy_extras/nodes_minimax_h3.py. Order matters:
    images first, then audio, because the tokenizer numbers the tags in the order given.
    """
    import math, comfy.sd, comfy.model_management as mm
    import comfy_extras.nodes_minimax_h3 as h3nodes
    from safetensors.torch import load_file

    imgs = list(getattr(args, "ref_image", None) or [])[:9]
    auds = list(getattr(args, "ref_audio", None) or [])[:3]
    if not (imgs or auds):
        return None, None

    # KNOWN LIMITATION. Reference IMAGES go through the Qwen3-VL vision tower, and on this
    # build that dies inside preprocess_embed with
    #     mat1 and mat2 shapes cannot be multiplied (1728x1152 and 3456x1152)
    # 3456 = 3 x 1152: the merger wants three concatenated deepstack levels and gets one, so
    # the teacher GGUF we run is missing the deepstack projections the vision path needs.
    # Reference AUDIO is unaffected: the tokenizer only emits an "<Audio j>" text tag for it
    # and the latent goes straight to the DiT, never touching the vision tower. Since voice
    # is what references are for here, audio works today and images wait for a TE build with
    # the vision weights. Use --first-frame for identity in the meantime.
    if imgs and not os.environ.get("H3_ALLOW_REF_IMAGE"):
        raise SystemExit(
            "--ref-image is not supported by this text encoder build (its vision tower is "
            "missing the deepstack projections; see _build_refs). Use --first-frame for "
            "identity, or set H3_ALLOW_REF_IMAGE=1 to try anyway.")

    items, blocks = [], []
    vvae = avae = None
    dev = mm.get_torch_device()
    cm = h3nodes.CANVAS_MULTIPLE

    for i, path in enumerate(imgs, 1):
        frame = _load_frame(path)
        h, w = frame.shape[1], frame.shape[2]
        if getattr(args, "ref_image_size", "match") == "match":
            scale = min(1.0, math.sqrt((args.width * args.height) / (w * h)))
        else:
            # the reference pipeline's own short edge: best identity, several times slower,
            # because reference tokens are re-read at every sampling step
            scale = min(1.0, h3nodes.REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(cm, round(w * scale / cm) * cm)
        th = max(cm, round(h * scale / cm) * cm)
        resized = h3nodes._resize(frame[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        if vvae is None:
            vvae = comfy.sd.VAE(sd=load_file(P["vae_video"]), device=dev)
            vvae.disable_offload = True
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vvae.encode(resized)})
        print(f"  <Picture {i}>: {path} -> {tw}x{th} reference latent", flush=True)

    for j, path in enumerate(auds, 1):
        items.append({"type": "audio"})
        if avae is None:
            avae = comfy.sd.VAE(sd=load_file(P["vae_audio"]), device=dev)
            avae.disable_offload = True
        z, t = h3nodes._encode_ref_audio(avae, _load_audio_full(path))
        blocks.append({"kind": "audio", "ref_audio_t": t, "audio_latent": z})
        print(f"  <Audio {j}>: {path} -> reference audio latent t={t}", flush=True)

    del vvae, avae
    mm.soft_empty_cache()
    return items, blocks


def _dit_state_dict(path):
    """gguf_sd_loader needs the file to declare an architecture so it can map keys.
    The unsloth engine GGUFs are arch-less with keys that are already comfy-native
    (verified: same 532 tensor names and shapes as the MiniStack build), so those
    go through _load_gguf_native, exactly as the teacher text encoder does."""
    import gguf
    from comfyui_gguf.loader import gguf_sd_loader
    arch = None
    try:
        r = gguf.GGUFReader(path)
        f = r.fields.get("general.architecture")
        if f is not None and getattr(f, "parts", None):
            arch = bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        del r
    except Exception as e:
        print("  engine: header probe failed:", type(e).__name__, e, flush=True)
    if arch:
        return gguf_sd_loader(path)
    print("  engine: arch-less GGUF -> native loader", flush=True)
    return _load_gguf_native(path), {}


def _prefetch(path, threads=8, block=32 << 20):
    """h3x: read `path` ahead into the page cache on several threads, in the background.
    One stream of this SSD is ~0.4 GB/s and that is what a lazy or mmap load gets; eight streams are ~1.5 GB/s and
    the copy onto the card is 5-6 GB/s, so the loader (which walks the file in order) finds its pages already there.
    Returns the thread list; nothing waits for it."""
    import threading
    size = os.path.getsize(path)
    nblk = (size + block - 1) // block
    nxt = [0]; lock = threading.Lock(); t0 = time.time()
    def work():
        fd = os.open(path, os.O_RDONLY)
        buf = bytearray(block)
        try:
            while True:
                with lock:
                    i = nxt[0]; nxt[0] += 1
                if i >= nblk: break
                os.preadv(fd, [buf], i * block)
        finally:
            os.close(fd)
    ts = [threading.Thread(target=work, daemon=True) for _ in range(threads)]
    for t in ts: t.start()
    def report():
        for t in ts: t.join()
        dt = time.time() - t0
        print(f"  prefetch: {size/2**30:.1f} GiB of {os.path.basename(path)} read ahead in {dt:.1f}s ({size/2**30/dt:.2f} GiB/s, {threads} threads)", flush=True)
    threading.Thread(target=report, daemon=True).start()
    return ts


def _align(n):
    while n % 17 != 5:
        n += 1
    return n


def _graceful_sigterm():
    """docker stop sends SIGTERM; unhandled, Python dies 30 s later by SIGKILL with GPU
    contexts still open. 2026-09-22 06:01 that left a GuC exec queue hanging, xe's reset
    failed and the device went wedged (every job after that: "XPU device count is zero")
    until a reboot. Turn SIGTERM into SystemExit so torch/xe teardown runs."""
    import signal, sys

    def _bye(signum, frame):
        print("  SIGTERM: shutting down cleanly (releasing XPU contexts)", flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, _bye)


def _dump_block_install(path):
    """h3x: on the first denoiser call, save everything the Rust engine needs to check a ported block against this
    pipeline: block 0's input, every intermediate of block 0 (computed again here, step by step, as DiTBlock.forward
    does it), and the last block's output. One .safetensors file."""
    import torch
    import comfy.ldm.minimax.model as M
    import comfy.ops as ops
    import comfy.model_management as mm
    import comfy.quant_ops as qo
    from safetensors.torch import save_file
    orig = M.DiTBlock.forward
    st = {"d": {}, "done": False}

    def cpu(t):
        return t.detach().to("cpu").contiguous()

    def fwd(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        idx = transformer_options.get("block_index")
        if st["done"] or attention is not None:
            return orig(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, attention=attention)
        d = st["d"]
        if idx == 0:
            layout = transformer_options["minimax_h3_layout"]
            xc = x.clone()
            d["x_in"] = cpu(xc)
            d["t_emb"] = cpu(t_emb.float())
            rows = torch.empty(x.shape[0], dtype=torch.int32)
            for a, b, row in mod_segments:
                rows[a:b] = row.to("cpu", torch.int32) if torch.is_tensor(row) else int(row)
            d["mod_rows"] = rows
            d["position_ids"] = cpu(layout.position_ids)
            d["rope_table"] = cpu(rope_freqs.float())
            names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
            mods = self.adaln_proj(t_emb)
            for n, m in zip(names, mods):
                d["b0." + n] = cpu(m.float())
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
            h1 = M._mod_scale_shift(self.norm1(xc), shift_msa, scale_msa, mod_segments)
            d["b0.h1"] = cpu(h1)
            at = self.attn
            s = h1.shape[0]
            qkv = at.qkv_proj(h1)
            d["b0.qkv"] = cpu(qkv)
            q, k, v = qkv.split(at.heads * at.head_dim, dim=-1)
            v = v.view(s, at.heads, at.head_dim)
            q = q.view(1, s, at.heads, at.head_dim)
            k = k.view(1, s, at.heads, at.head_dim)
            qw = mm.cast_to(at.q_norm.weight, device=x.device)
            kw = mm.cast_to(at.k_norm.weight, device=x.device)
            qo.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=at.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
            d["b0.q_rope"] = cpu(q[0])
            d["b0.k_rope"] = cpu(k[0])
            att = M.optimized_attention(M.AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0)),
                                        M.AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0)),
                                        M.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0)),
                                        at.heads, mask=None, skip_reshape=True, transformer_options=transformer_options).squeeze(0)
            d["b0.att"] = cpu(att)
            attn_out = at.out_proj(att)
            d["b0.attn_out"] = cpu(attn_out)
            x1 = M._mod_gate(xc.clone(), gate_msa, attn_out, mod_segments)
            d["b0.x1"] = cpu(x1)
            h2 = M._mod_scale_shift(self.norm2(x1), shift_mlp, scale_mlp, mod_segments)
            d["b0.h2"] = cpu(h2)
            fc1 = self.mlp.fc1(h2)
            d["b0.fc1"] = cpu(fc1)
            mlp = ops.linear_input_act(self.mlp.fc2, fc1, "swiglu")
            d["b0.mlp"] = cpu(mlp)
            d["b0.x2"] = cpu(M._mod_gate(x1.clone(), gate_mlp, mlp, mod_segments))
            st["meta"] = {"heads": str(at.heads), "head_dim": str(at.head_dim), "qk_eps": repr(float(at.q_norm.eps)),
                          "norm_eps": repr(float(self.norm1.eps)), "rot_dim": str(rope_freqs.shape[-3] * 2),
                          "segments": repr([(a, b, kind) for a, b, kind in layout.segments])}
        out = orig(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, attention=attention)
        if idx == 0:
            rel = float((out.float().cpu() - d["b0.x2"].float()).norm() / out.float().norm())
            print(f"  block dump: block 0 recomputed step by step differs from the real call by {rel:.2e}", flush=True)
        if idx in (0, 1, 24):
            d[f"out.{idx}"] = cpu(out)
        if idx is not None and idx >= 40:          # whichever block runs last
            st["last"] = (idx, cpu(out))
        return out

    def finish():
        if st["done"] or "x_in" not in st["d"]:
            return
        d = st["d"]
        if st.get("last"):
            d["out.last"] = st["last"][1]
            st["meta"]["last_block"] = str(st["last"][0])
        save_file(d, path, metadata=st["meta"])
        st["done"] = True
        print(f"  block dump -> {path}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in d.items() if k in ("x_in", "b0.qkv", "out.last")), flush=True)

    M.DiTBlock.forward = fwd
    final = M.FinalLayer.forward

    def final_fwd(self, *a, **k):
        finish()
        return final(self, *a, **k)
    M.FinalLayer.forward = final_fwd


def _profile_install():
    """h3x: where a denoiser step goes. Wraps the pieces of the DiT with a device sync on both sides and returns
    {name: [calls, seconds]}; the names nest (model > block > attention/mlp > their linears), so they do not add up."""
    import torch
    import comfy.ldm.minimax.model as M
    import comfy.quant_ops as qo
    stats = {}

    def timed(name, fn):
        def w(*a, **k):
            torch.xpu.synchronize(); t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.xpu.synchronize()
            st = stats.setdefault(name, [0, 0.0]); st[0] += 1; st[1] += time.perf_counter() - t0
            return r
        return w

    M.optimized_attention = timed("attention (scores+softmax)", M.optimized_attention)
    for cls, meth, name in ((M.MiniMaxH3Model, "forward", "model forward"), (M.DiTBlock, "forward", "DiT block"),
                            (M.RefinerBlock, "forward", "refiner block"), (M.Attention, "forward", "attention module"),
                            (M.FinalLayer, "forward", "final layer"),
                            (M.AdalnProj, "forward", "adaln projection")):
        setattr(cls, meth, timed(name, getattr(cls, meth)))
    for fn in ("rms_rope_split_half_",):
        if hasattr(qo.ck, fn):
            setattr(qo.ck, fn, timed("ck." + fn, getattr(qo.ck, fn)))
    M._mod_scale_shift = timed("scale+shift", M._mod_scale_shift)
    M._mod_gate = timed("gated residual add", M._mod_gate)
    import comfy.ops as ops
    M.MLP.forward = timed("mlp module", lambda self, x: timed("fc2 (swiglu + linear)", ops.linear_input_act)(
        self.fc2, timed("fc1 linear", self.fc1)(x), "swiglu"))
    return stats


def cmd_gen(args):
    _graceful_sigterm()
    import torch, comfy.sd, comfy.sample, comfy.samplers, comfy.utils
    import comfy.model_management as mm
    import comfy.text_encoders.minimax as mmte
    from comfyui_gguf.loader import gguf_sd_loader, gguf_clip_loader
    from comfyui_gguf.ops import GGMLOps
    from safetensors.torch import load_file
    import comfy_extras.nodes_minimax_h3 as h3nodes
    P = _paths()
    T0 = time.time()

    frames = _align(max(5, round(args.seconds * 24)))
    print(f"=== h3 gen ===")
    print(f"  prompt : {args.prompt[:70]}")
    print(f"  {args.width}x{args.height}  {frames} frames @24fps (~{frames/24:.2f}s)  steps={args.steps}  te={args.te}")

    # ---- (2) conditioning cache: same prompt + encoder => identical cond ----
    import hashlib
    ckey = hashlib.sha256(
        f"{args.te}|{args.te_file or ''}|{args.prompt}|{_ref_cache_key(args)}".encode()
    ).hexdigest()[:16]
    cdir = "/cache/cond"; cpath = f"{cdir}/{ckey}.pt"
    positive = negative = None
    if os.path.isdir("/cache") and os.path.exists(cpath) and not args.no_cond_cache:
        t0 = time.time()
        try:
            _c = torch.load(cpath, map_location="cpu")
            positive, negative = _c["positive"], _c["negative"]
            _t("conditioning (CACHED, TE skipped)", t0)
            print("  cond dim:", positive[0][0].shape)
        except Exception as e:
            print("  cond cache unreadable, re-encoding:", e, flush=True); positive = negative = None

    # ---- text encoder: student (4B + adapter) or teacher (32B, native 5120-d) ----
    t0 = time.time()
    if positive is not None:
        te = None; adapter = None
    ops2 = GGMLOps(); ops2.Linear.dequant_dtype = None; ops2.Linear.patch_dtype = None
    if positive is not None:
        pass
    elif args.te == "teacher":
        adapter = None
        # Stock comfy H3 encoder: Qwen3-VL-32B conditioning checkpoint (50 layers,
        # no final norm, no lm_head) -> 5120-d directly, no adapter. This is the
        # encoder H3 was trained with; the student loses phoneme-level detail.
        tf = args.te_file or P["te_teacher"]
        if not os.path.exists(tf):
            sys.exit(f"teacher TE not found: {tf} (download still running?)")
        # The unsloth teacher GGUF is ARCH-LESS (general.architecture=None) with
        # comfy-native key names - it was quantized with the ComfyUI-patched
        # llama.cpp like the DiT. gguf_clip_loader refuses arch-less files, but
        # the diffusion-style loader returns the native keys untouched. Only the
        # token embedding must be a plain tensor (nn.Embedding can't index a
        # GGMLTensor), so dequantize that one, as the student path does.
        from comfyui_gguf.dequant import dequantize_tensor
        te_sd = _load_gguf_native(tf)
        emb_k = "model.embed_tokens.weight"
        if emb_k in te_sd and hasattr(te_sd[emb_k], "tensor_type"):
            te_sd[emb_k] = dequantize_tensor(te_sd[emb_k], dtype=torch.float16)
        # Drop the vision tower: unused for text-to-video, and its patch-embed is
        # stored flattened ([3456,2,16,16] vs comfy's [1152,3,2,16,16]) which is
        # the only shape that fails strict-shape loading. Text side matches.
        nvis = sum(1 for k in te_sd if k.startswith("visual."))
        te_sd = {k: v for k, v in te_sd.items() if not k.startswith("visual.")}
        print(f"  teacher TE: {len(te_sd)} text tensors read (dropped {nvis} visual)", flush=True)
        te = mmte.MiniMaxH3TEModel(device="cpu", dtype=torch.float16,
                                   model_options={"custom_operations": ops2})
        missing, unexpected = te.load_sd(te_sd)
        nonvis = [k for k in missing if not k.startswith("visual.")]
        print(f"  teacher TE: missing non-visual={len(nonvis)} unexpected={len(unexpected)}", flush=True)
        _t("teacher TE (32B) loaded", t0)
    else:
        adapter = None
        te_sd = gguf_clip_loader(P["te"])
        if isinstance(te_sd, tuple):
            te_sd = te_sd[0]
        TE = _student_te_classes()
        te = TE(device="cpu", dtype=torch.float16,
                model_options={"custom_operations": ops2, "model_config": {"final_norm": False}})
        te.load_sd(te_sd)
        ad = load_file(P["adapter"])
        adapter = torch.nn.Sequential(
            torch.nn.Linear(2560, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 5120))
        with torch.no_grad():
            adapter[0].weight.copy_(ad["net.0.weight"]); adapter[0].bias.copy_(ad["net.0.bias"])
            adapter[2].weight.copy_(ad["net.2.weight"]); adapter[2].bias.copy_(ad["net.2.bias"])
        adapter = adapter.to(torch.float32).eval()
        try:
            adapter = adapter.to(mm.get_torch_device())
        except Exception:
            pass
        _t("student TE + adapter", t0)

    # ---- tokenize + encode, then map 2560 -> 5120 ----
    t0 = time.time()
    if positive is None:
      tok = mmte.MiniMaxH3Tokenizer()
      clip = comfy.sd.CLIP(no_init=True)
      clip.cond_stage_model = te
      clip.tokenizer = tok
      # Force the TE onto the XPU. mm.text_encoder_device() picks CPU on this box,
      # which makes encoding run single-threaded for many minutes.
      _dev = mm.get_torch_device()
      print("  TE device:", _dev, flush=True)
      te.cond_stage_model = getattr(te, "cond_stage_model", None)
      clip.patcher = comfy.model_patcher.ModelPatcher(te, load_device=_dev,
                                                      offload_device=torch.device("cpu"))
      clip.layer_idx = None
      clip.use_clip_schedule = False
      clip.apply_hooks_to_conds = None
      clip.tokenizer_options = {}

      ref_items, ref_blocks = _build_refs(args, P)

      def _enc(prompt, refs=None):
          # refs go on the positive only; cfg is 1.0 here so the negative is inert anyway
          tokens = clip.tokenize(prompt, minimax_ref_items=refs) if refs else clip.tokenize(prompt)
          c = clip.encode_from_tokens_scheduled(tokens)
          for entry in c:
              emb = entry[0]
              if adapter is not None and emb.shape[-1] == 2560:
                  dev = next(adapter.parameters()).device
                  entry[0] = adapter(emb.to(dev, torch.float32)).to(emb.dtype)
          return c

      print("  encoding positive...", flush=True)
      positive = _enc(args.prompt, ref_items)
      print("  encoding negative...", flush=True)
      negative = _enc("")
      if ref_blocks:
          import node_helpers
          positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": ref_blocks})
          print(f"  {len(ref_blocks)} reference block(s) attached to the positive conditioning", flush=True)
      _t("conditioning", t0)
      if os.path.isdir("/cache") and not args.no_cond_cache:
          os.makedirs(cdir, exist_ok=True)
          def _cpu(c):
              return [[e[0].detach().cpu(), {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in e[1].items()}] for e in c]
          torch.save({"positive": _cpu(positive), "negative": _cpu(negative), "prompt": args.prompt, "te": args.te}, cpath)
          print(f"  cond cached -> {cpath}", flush=True)
    print("  cond dim:", positive[0][0].shape)

    # ---- free the TE, then load the DiT ----
    # STAGED RESIDENCY: the card is 32 GiB and ComfyUI sizes the DiT by its
    # DEQUANTIZED footprint (18.7B params), so TE + DiT resident together
    # overruns VRAM (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY). Encode first, drop
    # the encoder, then bring in the denoiser.
    import gc
    try:
        del te_sd
    except NameError:
        pass
    try:
        mm.unload_all_models(); mm.soft_empty_cache()
    except Exception:
        pass
    try:
        del te, clip
    except NameError:
        pass
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    _t("TE unloaded", time.time())

    if getattr(args, "first_latent", None):
        # CHAIN FROM THE LATENT. The previous clip's last latent frame is already in
        # the space vae.encode() produces, so it can be injected as the keyframe with
        # no decode/encode at all. Feeding the finished mp4 instead costs a VAE round
        # trip plus an h264 round trip per hop, and those losses compound down a chain.
        import node_helpers
        t0 = time.time()
        _kf = torch.load(args.first_latent, map_location="cpu")
        kf_latent = _kf["latent"] if isinstance(_kf, dict) else _kf
        kf_latent = kf_latent.to(mm.get_torch_device())
        positive = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": kf_latent}]})
        print(f"  first latent: {args.first_latent} -> keyframe latent {tuple(kf_latent.shape)} (no VAE, no codec)", flush=True)
        _t("first latent loaded", t0)
    elif getattr(args, "first_frame", None) or getattr(args, "last_frame", None):
        # KEYFRAMES (fl2va = first/last frame to video+audio): anchor frame 0 and/or the
        # final frame by injecting each image's VAE latent as a condition block that rides
        # through every step and is never denoised - exactly what comfy's
        # MiniMaxH3ImageToVideo does with its first_frame / last_frame inputs. The text
        # encoder does not see the images; the latent anchors are the strong signal. With
        # hub anchoring first == last == clip 1's frame, so both sides of every cut are the
        # same pixels and the join needs no crossfade to hide.
        import node_helpers
        t0 = time.time()
        vvae = None
        kfs = list(positive[0][1].get("minimax_keyframes", []))
        _ref = None
        if getattr(args, "first_frame_ref", None) and os.path.exists(args.first_frame_ref):
            _ref = _load_frame(args.first_frame_ref)
        _y = lambda t: float((0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]).mean())

        def _anchor(path, idx, what):
            nonlocal vvae
            frame = _load_frame(path)
            if _ref is not None:
                _a, _b = _y(frame), _y(_ref)
                if _a > 1e-4:
                    _g = min(max(_b / _a, 0.75), 1.35)      # clamp: correct drift, never re-grade the shot
                    frame = (frame * _g).clamp(0, 1)
                    print(f"  exposure match ({what}): anchor Y {_a*255:.1f} -> ref Y {_b*255:.1f}, gain {_g:.3f}", flush=True)
            img = h3nodes._resize(frame, args.width, args.height, "disabled")
            if vvae is None:
                vvae = comfy.sd.VAE(sd=load_file(P["vae_video"]), device=mm.get_torch_device())
                vvae.disable_offload = True
            lat = vvae.encode(img)
            kfs.append({"resolved_frame_index": idx, "latent": lat})
            print(f"  {what}: {path} {tuple(frame.shape[1:3])} -> keyframe latent {tuple(lat.shape)} at frame {idx}", flush=True)

        if getattr(args, "first_frame", None):
            _anchor(args.first_frame, 0, "first frame")
        if getattr(args, "last_frame", None):
            _anchor(args.last_frame, frames - 1, "last frame")
        del vvae; gc.collect()
        try:
            mm.unload_all_models(); mm.soft_empty_cache()
        except Exception:
            pass
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": kfs})
        _t("keyframes encoded", t0)

    if getattr(args, "first_audio", None):
        # AUDIO KEYFRAME (comfy's MiniMaxH3AddGuide with an audio input): the last
        # --first-audio-s seconds of the source are encoded with the audio VAE and pinned at
        # frame 0, so this clip opens on the previous clip's closing room tone / voice tail
        # and the join has no audio seam. Source: a <clip>.lastaud.pt (model-native level,
        # preferred) or any media file (decoded with PyAV).
        import node_helpers
        t0 = time.time()
        # the audio VAE's conv stack needs >= ~0.6 s of input (0.5 s died with "padded input
        # size per channel: (6). Kernel size: (7)"); 1.0 s still sits inside the ~1.2 s lead
        # the model leaves before the first word
        _sec = max(1.0, float(getattr(args, "first_audio_s", 1.0) or 1.0))
        if args.first_audio.endswith(".pt"):
            _d = torch.load(args.first_audio, map_location="cpu")
            wav, sr = _d["waveform"].float(), int(_d.get("sr", 32000))
        else:
            wav, sr = _load_audio_tail(args.first_audio, _sec + 1.0)
        wav = _canon_audio(wav)                  # [B, C, L]
        wav = wav[..., -int(_sec * sr):]
        print(f"  audio anchor waveform {tuple(wav.shape)} @ {sr} Hz ({wav.shape[-1]/sr:.2f}s)", flush=True)
        avae = comfy.sd.VAE(sd=load_file(P["vae_audio"]), device=mm.get_torch_device())
        vae_sr = int(getattr(avae, "audio_sample_rate", 32000))
        if sr != vae_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, vae_sr)
        z = avae.encode(wav[:1].movedim(1, -1).to(mm.get_torch_device()))     # VAE wants [B, L, C] -> [1, 32, 2, T]
        del avae; gc.collect()
        try:
            mm.unload_all_models(); mm.soft_empty_cache()
        except Exception:
            pass
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        kfs = list(positive[0][1].get("minimax_keyframes", []))
        for kf in kfs:                           # share the frame-0 entry with the image anchor, as AddGuide does
            if kf.get("resolved_frame_index") == 0 and "audio_latent" not in kf:
                kf["audio_latent"] = z; break
        else:
            kfs.append({"resolved_frame_index": 0, "audio_latent": z})
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": kfs})
        print(f"  first audio: {args.first_audio} last {_sec:.2f}s -> audio keyframe latent {tuple(z.shape)} at frame 0", flush=True)
        _t("audio keyframe encoded", t0)

    if getattr(args, "cond_noise_aug", None):
        # tell the denoiser the anchor is approximate; stops it faithfully
        # reproducing the artefacts of a many-generations-old keyframe
        import node_helpers
        positive = node_helpers.conditioning_set_values(
            positive, {"minimax_visual_cond_noise_aug": float(args.cond_noise_aug)})
        print(f"  keyframe noise aug: {args.cond_noise_aug}", flush=True)

    t0 = time.time()
    ops = GGMLOps(); ops.Linear.dequant_dtype = None; ops.Linear.patch_dtype = None
    dit_path = getattr(args, "dit", None) or P["dit"]
    print(f"  engine : {os.path.basename(dit_path)} ({os.path.getsize(dit_path)/2**30:.2f} GiB)", flush=True)
    if os.environ.get("H3X_PREFETCH"):
        _prefetch(dit_path, threads=int(os.environ["H3X_PREFETCH"]))
    if dit_path.endswith(".safetensors"):
        # h3x: a comfy-native checkpoint (bf16 / fp8 / int8_convrot / w6a8): comfy's own loader and quantized ops,
        # which call comfy-kitchen (int8_linear etc.) - no GGUF ops, no GGUF patcher
        if os.environ.get("H3X_SYCL") or os.environ.get("H3X_INT8_NATIVE"):
            # h3x: ComfyUI answers "no int8 compute" for an Intel GPU (model_management.supports_int8_compute), and
            # then turns every int8 weight back into bf16 on each call. Say yes, so the linears reach comfy-kitchen.
            comfy.model_management.supports_int8_compute = lambda device=None: True
            print("  int8 linears : native (comfy-kitchen int8_linear)", flush=True)
        if os.environ.get("H3X_SYCL"):
            # h3x: our kernels (kernels/kitchen_sycl.py) ahead of comfy-kitchen's own backends
            sys.path.insert(0, os.environ.get("H3X_SYCL_DIR", "/work/kernels"))
            import kitchen_sycl
            print("  sycl backend :", kitchen_sycl.register(sync=bool(os.environ.get("H3X_SYCL_SYNC"))) or "NOT AVAILABLE", flush=True)
        model = comfy.sd.load_diffusion_model(dit_path)
        sd = None
        try:
            import comfy.quant_ops as _qo
            print("  kitchen backends:", {k: (v.get("available") if isinstance(v, dict) else v) for k, v in _qo.ck.list_backends().items()}, flush=True)
        except Exception as e:
            print("  kitchen backends: n/a", e, flush=True)
    else:
        sd, extra = _dit_state_dict(dit_path)
        import inspect
        kw = {}
        if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters:
            kw["metadata"] = extra.get("metadata", {})
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops}, **kw)
        from comfyui_gguf.nodes import GGUFModelPatcher
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = False
    # LoRAs: keys are diffusion_model.*.lora_A/lora_B, comfy's standard layout for this model.
    # They ride on the GGUF patcher, which applies patches as it dequantises each weight, so a
    # LoRA costs its own size in host RAM and nothing extra on the card.
    for spec in (getattr(args, "lora", None) or []):
        lp, _, sw = spec.rpartition(":")
        if not lp or not os.path.exists(lp):        # no ":strength" given
            lp, strength = spec, 1.0
        else:
            strength = float(sw)
        t0 = time.time()
        import comfy.sd
        model, _ = comfy.sd.load_lora_for_models(model, None, load_file(lp), strength, 0.0)
        print(f"  lora: {os.path.basename(lp)} @ {strength} ({os.path.getsize(lp)/2**20:.0f} MB)", flush=True)
        _t("lora applied", t0)
    _t("DiT loaded", t0)
    if getattr(args, "fused_q4k", False):
        # Marlin-style fused dequant+matmul (kernel/q4k_fused.py). Opt-in; see kernel/PLAN.md.
        try:
            sys.path.insert(0, "/work/kernel")
            import q4k_fused
            q4k_fused.install(force=True)
        except Exception as e:
            print("  fused-q4k: not installed:", type(e).__name__, e, flush=True)
    if not args.no_compile:
        # (1) fuse the per-step Q4_K dequant elementwise chain. Measured 18.1 ->
        # 16.8 s/step (7%). First compile ~90 s per latent shape; the launcher
        # mounts TORCHINDUCTOR_CACHE_DIR so later runs start warm.
        try:
            dm = model.model.diffusion_model
            dm.forward = torch.compile(dm.forward, backend="inductor", dynamic=False)
            print("  DiT: torch.compile enabled (inductor, cache=%s)" % os.environ.get("TORCHINDUCTOR_CACHE_DIR", "ephemeral"), flush=True)
        except Exception as e:
            print("  DiT: torch.compile unavailable, eager:", e, flush=True)

    if os.environ.get("H3X_PRELOAD"):
        # h3x: put the denoiser's weights on the card NOW, timed on its own, so step 1 no longer contains the load
        t0 = time.time()
        mm.load_models_gpu([model])
        if torch.xpu.is_available():
            torch.xpu.synchronize()
        _t("DiT weights on device (preload)", t0)
    # ---- latent ----
    latent, frame_count = h3nodes._empty_av_latent(args.width, args.height, frames)
    print("  latent:", type(latent["samples"]).__name__, " frames:", frame_count)

    # ---- sample ----
    t0 = time.time()
    torch.manual_seed(args.seed)
    noise = comfy.sample.prepare_noise(latent["samples"], args.seed)
    # seed= must be passed: model_base sets payload["seed"] = kwargs.get("seed", 0)
    # and the sampler's own default is None, which blows up in _cond_video_rows.
    # Deterministic progress lines (tqdm output does not reliably reach
    # `docker logs` mid-run). The front end parses "  step N/M".
    _t0s = time.time()
    _dump = os.environ.get("H3X_DUMP_STEPS")
    _run = os.environ.get("H3X_DUMP_RUN")     # h3x: a whole sampling run for the Rust engine (see below)
    _run_x0 = {}
    def _cb(step, x0, x, total):
        print(f"  step {step+1}/{total}  {time.time()-_t0s:6.1f}s", flush=True)
        _tt = getattr(x0, "tensors", None)          # video + audio travel as a nested pair
        _tt = list(_tt) if _tt is not None else [x0]
        if _dump:   # h3x: the denoised estimate after every step, for step-by-step comparisons between engines
            torch.save([t.detach().float().cpu() for t in _tt], f"{_dump}.step{step+1:02d}.pt")
        if _run:
            _run_x0[f"x0.{step+1:02d}.video"] = _tt[0].detach().float().cpu().contiguous()
            if len(_tt) > 1:
                _run_x0[f"x0.{step+1:02d}.audio"] = _tt[1].detach().float().cpu().contiguous()
    if os.environ.get("H3X_DUMP_BLOCK"):
        _dump_block_install(os.environ["H3X_DUMP_BLOCK"])
    _prof = _profile_install() if os.environ.get("H3X_PROFILE") else None
    samples = comfy.sample.sample(model, noise, args.steps, 1.0, "euler", "simple",
                                  positive, negative, latent["samples"], denoise=1.0,
                                  seed=args.seed, callback=_cb)
    _t(f"sampled {args.steps} steps", t0)
    if _run:
        # h3x: what the Rust engine needs to run the same sampling and be checked against it, in one safetensors file:
        # the text conditioning as it enters the model (before its projection and refiner), the starting noise (the
        # sampler starts at sigma 1, so this is the first x), the sigmas, every step's denoised estimate (in the
        # sampler's space: the audio stream carried x audio_scale), and the final latents (back in the model's space)
        from safetensors.torch import save_file
        import comfy.samplers
        _ms = model.get_model_object("model_sampling")
        _sig = comfy.samplers.calculate_sigmas(_ms, "simple", args.steps).float().cpu().contiguous()
        _nt = list(getattr(noise, "tensors", None) or [noise])
        _st2 = list(getattr(samples, "tensors", None) or [samples])
        _ctx = positive[0][0]
        _d = {"context": _ctx.detach().to("cpu").contiguous(), "sigmas": _sig,
              "noise.video": _nt[0].float().contiguous(), "noise.audio": _nt[1].float().contiguous(),
              "samples.video": _st2[0].detach().float().cpu().contiguous(), "samples.audio": _st2[1].detach().float().cpu().contiguous()}
        _tags = positive[0][1].get("minimax_token_tags")
        if _tags is not None:
            _d["token_tags"] = _tags.detach().to("cpu").to(torch.int32).contiguous()
        _d.update(_run_x0)
        save_file(_d, _run, metadata={"seed": str(args.seed), "steps": str(args.steps), "frames": str(frame_count),
                                      "width": str(args.width), "height": str(args.height),
                                      "audio_scale": repr(float(getattr(_ms, "audio_scale", 1.0))),
                                      "shift": repr(float(_ms.shift)), "audio_shift": repr(float(_ms.audio_shift or 0)),
                                      "keyframes": str(bool(positive[0][1].get("minimax_keyframes")))})
        print(f"  run dump -> {_run}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in _d.items() if not k.startswith("x0.")), flush=True)
    if _prof:
        print("  profile (seconds, synchronized; slower than a normal run):", flush=True)
        for k, (n, sec) in sorted(_prof.items(), key=lambda kv: -kv[1][1]):
            print(f"    {k:28s} {n:6d} calls {sec:8.1f}s", flush=True)
    if "kitchen_sycl" in sys.modules:
        print("  sycl backend calls:", {k: (v[0], round(v[1], 1)) for k, v in sys.modules["kitchen_sycl"].STATS.items()}, flush=True)
    # comfy's own accounting: was the DiT fully resident or partially streamed?
    try:
        ls = model.loaded_size() / 1024**3
        lv = getattr(model.model, "model_lowvram", None)
        print(f"  DiT residency: loaded_size {ls:.2f}G  lowvram={lv}  "
              f"model_size {model.model_size()/1024**3:.2f}G", flush=True)
    except Exception as e:
        print("  DiT residency: n/a", e, flush=True)

    # ---- hand off to a fresh process for decode ----
    # 23 GiB host: after sampling, this heap has held the 10.6 GB DiT and glibc
    # does not return that to the OS, so loading the 4.85 GB video VAE here gets
    # SIGKILLed. Save the latents and decode in a clean address space.
    import gc, subprocess, pickle
    # NestedTensor.__getitem__ broadcasts into EACH stream - use .tensors to
    # select the video/audio streams themselves.
    _st = getattr(samples, "tensors", None)
    if _st is None:
        _st = [samples[0], samples[1]]
    vid_lat, aud_lat = _st[0], _st[1]
    lat_path = (args.out[:-4] if args.out.endswith(".mp4") else args.out) + ".latents.pt"   # h3x: per clip, so a rerun of the decode is possible
    torch.save({"video": vid_lat.detach().cpu(),
                "audio": aud_lat.detach().cpu(),
                "frames": frame_count}, lat_path)
    print(f"  latents saved ({os.path.getsize(lat_path)/1024**2:.0f} MB)", flush=True)
    try:   # the next clip in a chain wants only the final latent frame
        _base = args.out[:-4] if args.out.endswith(".mp4") else args.out
        torch.save({"latent": vid_lat[:, :, -1:].detach().cpu()}, f"{_base}.lastlat.pt")
        print(f"  last latent -> {_base}.lastlat.pt {tuple(vid_lat[:, :, -1:].shape)}", flush=True)
    except Exception as e:
        print("  last-latent save failed:", e, flush=True)
    del model, sd, samples, vid_lat, aud_lat
    gc.collect()

    argv = [sys.executable, "-u", __file__, "decode", "--latents", lat_path,
            "--out", args.out, "--chunk", str(args.chunk)]
    if args.no_audio:
        argv.append("--no-audio")
    if getattr(args, "no_normalize", False):
        argv.append("--no-normalize")
    if getattr(args, "upscale", None) and float(args.upscale) > 1.0:
        argv += ["--upscale", str(args.upscale), "--upscale-model", args.upscale_model]
    print("  -> decoding in a fresh process", flush=True)
    rc = subprocess.call(argv)
    print(f"\n  TOTAL {time.time()-T0:.1f}s  (decode rc={rc})")


def cmd_decode(args):
    """Decode saved latents in a clean process (see cmd_gen for why)."""
    _graceful_sigterm()
    import torch, comfy.sd, gc
    import comfy.model_management as mm
    from safetensors.torch import load_file as _lf
    P = _paths()
    T0 = time.time()
    if args.latents.endswith(".safetensors"):   # h3x: latents from the Rust engine (h3d denoise --out)
        d = _lf(args.latents)
        vid_lat, aud_lat = d["samples.video"], d["samples.audio"]
    else:
        d = torch.load(args.latents, map_location="cpu")
        vid_lat, aud_lat = d["video"], d["audio"]
    print("=== decode ===")
    print("  inference_mode:", torch.is_inference_mode_enabled(), flush=True)
    print("  video latent:", tuple(vid_lat.shape), " audio latent:", tuple(aud_lat.shape))
    _dev = mm.get_torch_device()

    # ---- optional latent upscale, between sampling and the video VAE -------------
    # A trained 3D resizer on the 24-channel H3 latent (LBH-123-AI/Minimax_h3_latent_Upscaler):
    # it buys resolution without paying the DiT's quadratic cost, because sampling already
    # happened at the small canvas. Nothing else in the pipeline changes - the anchors, the
    # audio and the frame count are untouched.
    if getattr(args, "upscale", None) and float(args.upscale) > 1.0:
        t0 = time.time()
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        # two backbones from the same pack, same checkpoint: 3d = fully 3D convolutions,
        # temporally coherent; 2d = 2D ResBlocks with temporal convs, faster
        if getattr(args, "upscale_node", "3d") == "2d":
            import h3_upscaler_2d as U
        else:
            import h3_upscaler as U
        sc = float(args.upscale)
        B, C, T, H, W = vid_lat.shape
        h_out, w_out = int(round(H * sc)), int(round(W * sc))
        print(f"  latent upscale: {W}x{H} -> {w_out}x{h_out} latent  "
              f"({W*16}x{H*16} -> {w_out*16}x{h_out*16} pixels), scale {sc}", flush=True)
        dtype = torch.bfloat16
        model = U.load_model(args.upscale_model, _dev, "bf16")
        mean, std = U._make_norm_tensors(_dev, dtype)
        x = vid_lat.to(device=_dev, dtype=dtype)
        with torch.inference_mode():
            x = (x - mean) / std
            if getattr(args, "upscale_node", "3d") == "2d":
                x = model(x, scale=sc, target_hw=(h_out, w_out))
            else:
                x = model(x, scale=sc, target_size=(T, h_out, w_out), enable_chunking=True)
            x = x * std + mean
        vid_lat = x.to(device="cpu", dtype=vid_lat.dtype)
        del x, model, mean, std
        U.MODEL_CACHE.clear()
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        try:
            mm.unload_all_models(); mm.soft_empty_cache()
        except Exception:
            pass
        print(f"  upscaled latent: {tuple(vid_lat.shape)}", flush=True)
        _t("latent upscaled", t0)

    audio = None
    if not args.no_audio:
        t0 = time.time()
        try:
            avae = comfy.sd.VAE(sd=_lf(P["vae_audio"]), device=_dev)
            audio = avae.decode(aud_lat)
            _t("AUDIO decoded", t0)
            _w = audio["waveform"] if isinstance(audio, dict) else audio
            _w = _w.detach()
            print("  audio:", tuple(_w.shape), " rms:", round(float(_w.float().pow(2).mean().sqrt()), 5))
            del avae
        except Exception as e:
            print("  AUDIO DECODE FAILED:", type(e).__name__, e)
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    t0 = time.time()
    # fp16 explicitly: comfy upcasts VAEs to fp32 by default, which turns the
    # 4.85 GB fp16 checkpoint into ~9.7 GB resident on a 23 GiB host.
    # Mirror the measured-fast path EXACTLY (vaegrad.py: 0.23 s/frame, 5.3 GiB):
    # no explicit dtype, and disable_offload=True so comfy's memory manager
    # does not re-plan / partially stream the VAE on every chunk. Also make
    # sure nothing from the audio pass is still resident.
    try:
        mm.unload_all_models(); mm.soft_empty_cache()
    except Exception:
        pass
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    _vsd = _lf(P["vae_video"])
    vvae = comfy.sd.VAE(sd=_vsd, device=_dev)
    vvae.disable_offload = True
    del _vsd; gc.collect()
    print("  video VAE loaded", flush=True)
    # TEMPORAL CHUNKING: decode_tiled's tile_x/tile_y are 2D and OOM on this 3D
    # video VAE; a full decode of all latent frames SIGKILLs the 23 GiB host.
    # Decode a few latent frames at a time instead (latent is B,C,T,H,W).
    T = vid_lat.shape[2]
    # chunk<=0 means ONE pass over all latent frames. Chunked causal decoding
    # drops frames at every chunk boundary (2-frame chunks gave 91 of 124);
    # with inference_mode the full 37-frame latent decodes at ~5-10 GiB, so
    # single-pass is the correct default. Use --chunk N only if VRAM demands.
    chunk = int(args.chunk) if int(args.chunk) > 0 else T
    outs = []
    for i in range(0, T, chunk):
        part = vid_lat[:, :, i:i + chunk]
        img = vvae.decode(part)
        outs.append(img.detach().cpu())
        del img, part
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        print(f"    decoded latent frames {i}-{min(i+chunk,T)-1}/{T-1}", flush=True)
    images = torch.cat(outs, dim=0) if outs[0].ndim == 4 else torch.cat(outs, dim=1)
    del outs
    _t("video decoded", t0)
    print("  frames:", tuple(images.shape))
    del vvae; gc.collect()

    _fbytes = images.numel() * images.element_size()
    try:
        if _fbytes > 2 * 2**30:      # >2 GiB: the remux shortcut is not worth an OOM
            raise MemoryError(f"frame buffer {_fbytes/2**30:.1f} GiB, skipping the remux cache")
        torch.save({"images": images.detach().cpu(),
                    "audio": (audio["waveform"] if isinstance(audio, dict) else audio).detach().cpu()
                             if audio is not None else None,
                    "sr": audio.get("sample_rate", 32000) if isinstance(audio, dict) else 32000},
                   "/out/.frames.pt")
        print("  frames cached to /out/.frames.pt", flush=True)
    except Exception as e:
        print("  frame cache failed:", e)
    try:   # lossless anchor for the next clip: no h264 generation loss
        from PIL import Image
        import numpy as _np
        _last = images[-1] if images.ndim == 4 else images[0][-1]
        _arr = (_last.clamp(0, 1).cpu().numpy() * 255).astype(_np.uint8)
        _base = args.out[:-4] if args.out.endswith(".mp4") else args.out
        Image.fromarray(_arr).save(f"{_base}.last.png")
        print(f"  last frame -> {_base}.last.png (lossless)", flush=True)
    except Exception as e:
        print("  last-frame save failed:", e, flush=True)
    try:   # last second of audio at the model's own level (the mp4 is peak-normalised): the
           # audio anchor for the next clip, so its opening room tone is this clip's closing one
        if audio is not None:
            _w = _canon_audio((audio["waveform"] if isinstance(audio, dict) else audio).detach().cpu().float())
            _sr = int(audio.get("sample_rate", 32000)) if isinstance(audio, dict) else 32000
            _keep = int(2 * _sr)
            torch.save({"waveform": _w[..., -_keep:].clone(), "sr": _sr}, f"{_base}.lastaud.pt")
            print(f"  last audio -> {_base}.lastaud.pt ({min(_w.shape[-1], _keep) / _sr:.2f}s @ {_sr} Hz)", flush=True)
    except Exception as e:
        print("  last-audio save failed:", e, flush=True)

    t0 = time.time()
    _write_video(images, audio, args.out, fps=24)
    _t("muxed", t0)
    print(f"  OUT   {args.out}")
    print(f"  decode total {time.time()-T0:.1f}s")


NORMALIZE = True


def _write_video(images, audio, path, fps=24):
    """images: [T,H,W,C] float 0..1 ; audio: waveform tensor or dict"""
    import numpy as np, torch, av, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(images, "movedim") and images.ndim == 5:
        images = images[0]
    T, H, W = images.shape[0], images.shape[1], images.shape[2]
    container = av.open(path, mode="w")
    vs = container.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"

    astream = None
    wav = None
    if audio is not None:
        w = audio["waveform"] if isinstance(audio, dict) else audio
        sr = audio.get("sample_rate", 32000) if isinstance(audio, dict) else 32000
        w = w.detach().cpu().float()
        while w.ndim > 2:
            w = w[0]
        wav = w.numpy()          # (samples, channels) or (channels, samples)
        _a = w.detach().cpu().float().numpy()
        while _a.ndim > 2:
            _a = _a[0]
        if _a.ndim == 2 and _a.shape[0] in (1, 2) and _a.shape[1] > 8:
            _a = _a.T
        _ch = _a.shape[1] if _a.ndim == 2 else 1
        astream = container.add_stream("aac", rate=int(sr),
                                       layout="stereo" if _ch == 2 else "mono")

    for i in range(T):
        # per frame, not the whole clip: a 1536x1152x362 uint8 copy is 1.9 GB and the float
        # source another 7.7 GB, which is more host RAM than this box has to spare
        fr = (images[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(fr, format="rgb24")
        for p in vs.encode(frame):
            container.mux(p)
        del fr
    for p in vs.encode():
        container.mux(p)

    if astream is not None and wav is not None:
        try:
            # packed s16 wants (1, nb_samples*channels), interleaved.
            a = wav
            if a.ndim == 2 and a.shape[0] in (1, 2) and a.shape[1] > 8:
                a = a.T                       # (channels, samples) -> (samples, channels)
            ch = a.shape[1] if a.ndim == 2 else 1
            # A/V LENGTH LOCK: the video VAE lands on the 17k+5 frame grid at 24 fps
            # and the audio VAE on a 40 Hz latent grid, so decoded audio runs ~30 ms
            # longer than the video. Inaudible in one clip; concatenating 103 of them
            # drifted 3.5 s and put the narration visibly off the lips. Pin the audio
            # to exactly T/fps so every clip is a sync-neutral building block.
            want = int(round(T / fps * sr))
            if a.shape[0] > want:
                a = a[:want]
            elif a.shape[0] < want:
                pad = np.zeros((want - a.shape[0],) + a.shape[1:], dtype=a.dtype)
                a = np.concatenate([a, pad], axis=0)
            # Loudness: H3 output for quiet scenes sits around -40..-60 dBFS
            # (a whispered line is inaudible on a laptop). Peak-normalize to
            # -3 dBFS unless disabled; report the gain applied.
            if NORMALIZE:
                peak = float(np.abs(a).max()) if a.size else 0.0
                if peak > 1e-6:
                    g = (10 ** (-3 / 20)) / peak
                    a = a * g
                    print(f"  audio normalized: peak {20*np.log10(peak):.1f} dBFS -> -3 dBFS (gain x{g:.1f})", flush=True)
            pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
            af = av.AudioFrame.from_ndarray(pcm, format="s16",
                                            layout="stereo" if ch == 2 else "mono")
            af.sample_rate = int(astream.rate)
            for p in astream.encode(af):
                container.mux(p)
            for p in astream.encode():
                container.mux(p)
            print("  audio muxed (%d ch)" % ch, flush=True)
        except Exception as e:
            print("  AUDIO MUX FAILED (video still written):", type(e).__name__, e, flush=True)
    container.close()


def cmd_remux(args):
    """Re-mux from cached frames - seconds, no GPU, no decode."""
    import torch
    d = torch.load(args.frames, map_location="cpu")
    imgs = d["images"]
    aud = d.get("audio")
    audio = None if aud is None else {"waveform": aud, "sample_rate": d.get("sr", 32000)}
    print("  frames:", tuple(imgs.shape), " audio:", None if aud is None else tuple(aud.shape))
    t0 = time.time()
    _write_video(imgs, audio, args.out, fps=24)
    _t("muxed", t0)
    print("  OUT", args.out)


def main():
    ap = argparse.ArgumentParser(prog="h3", description="MiniMax H3 on Intel Arc Pro B70")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="show device + model inventory")
    sub.add_parser("loadtest", help="load the DiT and report (no generation)")
    sub.add_parser("teload", help="load the student text encoder + adapter")
    g = sub.add_parser("gen", help="generate a video")
    g.add_argument("--prompt", default=None)
    g.add_argument("--prompt-file", default=None, help="read the prompt from a file (avoids shell quoting)")
    g.add_argument("--dry-run", action="store_true", help="resolve args and exit without generating")
    g.add_argument("--te", choices=("student", "teacher"), default="teacher", help="text encoder: 4B student (fast) or 32B teacher (better dialogue)")
    g.add_argument("--te-file", default=None, help="override teacher TE GGUF path")
    g.add_argument("--dit", default=None, help="denoiser GGUF to use (default: the MiniStack Q4_K_M); pick a different quantization to trade VRAM for fidelity")
    g.add_argument("--ref-image", action="append", default=None, metavar="PATH",
                   help="NOT YET WORKING on this TE build (vision tower lacks the deepstack "
                        "projections) - use --first-frame instead. "
                        "reference image for identity (repeatable, up to 9). Unlike --first-frame this "
                        "does NOT pin frame 0: the encoder sees it as <Picture i> and its latent rides "
                        "every sampling step, so the shot is free to start anywhere.")
    g.add_argument("--ref-audio", action="append", default=None, metavar="PATH",
                   help="reference audio for VOICE (repeatable, up to 3). Presented as <Audio j> and "
                        "encoded by the audio VAE, so the model matches that voice. Unlike --first-audio "
                        "nothing is pinned at frame 0 and none of the reference is audible in the output.")
    g.add_argument("--ref-image-size", default="match", choices=("match", "max"),
                   help="'match' scales each reference to the generation's pixel area; 'max' uses the "
                        "reference pipeline's 2048px short edge for best identity, several times slower")
    g.add_argument("--first-frame", default=None, help="image, or a video whose LAST frame, anchors frame 0 (fl2va keyframe); chains clips")
    g.add_argument("--last-frame", default=None,
                   help="image, or a video whose LAST frame, anchors the FINAL frame (fl2va end keyframe); with "
                        "--first-frame set to the same image every cut is pixel-identical on both sides")
    g.add_argument("--lora", action="append", default=None, metavar="PATH[:STRENGTH]",
                   help="LoRA to apply to the denoiser, repeatable. Strength defaults to 1.0. A 4-step turbo "
                        "LoRA also wants --steps 4; a trigger-word LoRA wants its token in the prompt")
    g.add_argument("--upscale", type=float, default=None,
                   help="upscale the video latent by this factor before the VAE, so the DiT never pays for the "
                        "extra pixels (1.0-4.0; the checkpoint was trained mostly at 2.0)")
    g.add_argument("--upscale-model", default="/models/upscaler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                   help="latent upscaler checkpoint, as seen inside the container")
    g.add_argument("--upscale-node", default="3d", choices=("3d", "2d"),
                   help="which backbone from the upscaler pack. 3d = fully 3D convs, the default and the only "
                        "one that works with the published 3d_conv_v1 checkpoint. 2d loads that checkpoint "
                        "without complaint and decodes to a checkerboard of noise - it needs a 2D-trained "
                        "checkpoint, which the HF repo does not currently ship (tested 2026-09-24)")
    g.add_argument("--first-audio", default=None,
                   help="<clip>.lastaud.pt or any media file: its last --first-audio-s seconds are pinned as an audio "
                        "keyframe at frame 0 (fl2va audio guide), so the clip opens on that room tone / voice tail")
    g.add_argument("--first-audio-s", type=float, default=1.0, help="seconds of audio to anchor (min/default 1.0)")
    g.add_argument("--first-frame-ref", default=None,
                   help="reference image whose exposure the keyframe is matched to. Each decode/encode hop "
                        "darkens the anchor by ~1 luminance unit, which cost 19%% over a 19-clip chain; "
                        "rescaling to a fixed reference cancels the drift instead of diluting it")
    g.add_argument("--first-latent", default=None,
                   help="chain from a saved last-latent (<clip>.lastlat.pt) instead of a picture: skips the VAE "
                        "round trip AND the h264 round trip, which is what degrades a long chain")
    g.add_argument("--cond-noise-aug", type=float, default=None,
                   help="0..1 noise added to the keyframe latent so the denoiser trusts a degraded anchor less "
                        "(model_base: minimax_visual_cond_noise_aug). ~0.02-0.10 is the useful range")
    g.add_argument("--no-compile", action="store_true", help="skip torch.compile of the DiT")
    g.add_argument("--no-cond-cache", action="store_true", help="always re-encode the prompt")
    g.add_argument("--fused-q4k", action="store_true", help="EXPERIMENTAL: Triton fused Q4_K dequant+matmul (kernel/)")
    g.add_argument("--seconds", type=float, default=5.0)
    g.add_argument("--steps", type=int, default=8)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--no-normalize", action="store_true", help="keep raw audio level")
    g.add_argument("--chunk", type=int, default=0, help="latent frames per decode pass; 0 = all (default)")
    g.add_argument("--out", default="/out/h3.mp4")
    d = sub.add_parser("decode", help="decode saved latents (used internally)")
    d.add_argument("--latents", required=True)
    d.add_argument("--upscale", type=float, default=None,
                   help="upscale the video latent by this factor before the VAE (1.0-4.0, trained mostly at 2.0)")
    d.add_argument("--upscale-model", default="/models/upscaler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                   help="the latent upscaler checkpoint, inside the container")
    d.add_argument("--upscale-node", default="3d", choices=("3d", "2d"),
                   help="which backbone from the upscaler pack. 3d = fully 3D convs, the default and the only "
                        "one that works with the published 3d_conv_v1 checkpoint. 2d loads that checkpoint "
                        "without complaint and decodes to a checkerboard of noise - it needs a 2D-trained "
                        "checkpoint, which the HF repo does not currently ship (tested 2026-09-24)")
    d.add_argument("--out", default="/out/h3.mp4")
    d.add_argument("--no-audio", action="store_true")
    d.add_argument("--no-normalize", action="store_true", help="keep raw audio level")
    d.add_argument("--chunk", type=int, default=0, help="latent frames per decode pass; 0 = all (default)")
    r = sub.add_parser("remux", help="re-mux from cached frames")
    r.add_argument("--frames", default="/out/.frames.pt")
    r.add_argument("--out", default="/out/h3.mp4")
    r.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()
    global NORMALIZE
    NORMALIZE = not getattr(args, "no_normalize", False)
    if getattr(args, "prompt_file", None):
        with open(args.prompt_file) as _pf:
            args.prompt = _pf.read().strip()
    if args.cmd == "gen":
        if not args.prompt:
            ap.error("gen needs --prompt or --prompt-file (file was empty or missing)")
        if getattr(args, "dry_run", False):
            print("DRY RUN ok\n  prompt chars:", len(args.prompt), "\n  first line:", args.prompt.splitlines()[0][:100])
            print("  seconds:", args.seconds, " steps:", args.steps, " seed:", args.seed, " te:", args.te, " out:", args.out, " first_frame:", args.first_frame, " dit:", args.dit)
            return
    # inference_mode for EVERYTHING. Measured on the video VAE: grad-enabled
    # decode was 23.2 s/frame at a 70 GiB "VRAM" peak (spilling over PCIe on a
    # 32 GB card); inference_mode is 0.23 s/frame at 5.3 GiB. 100x. This one
    # line was the 35-minute decode, the OOM-kills, and the wedged xe engine.
    # NOTE: must be a `with` block. `torch.inference_mode().__enter__()` on a
    # temporary gets garbage-collected at once and its guard destructor turns
    # inference mode back OFF - which silently re-enabled autograd here and
    # reproduced the 70 GiB / full-card stall while the benchmarks were fast.
    import torch
    # HARD CAP on GPU allocations. Without it the xe driver lets a process
    # oversubscribe the 32 GB card by evicting VRAM buffers into HOST RAM (TTM).
    # Those pages are unswappable and not charged to any process, so the host
    # OOM-killer finds nothing to kill and the whole box livelocks (three
    # hard hangs on 2026-09-16/17, "[TTM] Buffer eviction failed" in the log).
    # With the cap torch raises an OOM error instead. H3_MEM_FRACTION overrides.
    if torch.xpu.is_available():
        frac = float(os.environ.get("H3_MEM_FRACTION", "0.95"))
        torch.xpu.set_per_process_memory_fraction(frac)
        print(f"  xpu allocator cap: {frac:.2f} of {torch.xpu.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
    with torch.inference_mode():
        assert torch.is_inference_mode_enabled(), "inference mode did not take"
        {"info": cmd_info, "loadtest": cmd_loadtest, "teload": cmd_teload,
         "gen": cmd_gen, "decode": cmd_decode, "remux": cmd_remux}[args.cmd](args)


if __name__ == "__main__":
    main()
