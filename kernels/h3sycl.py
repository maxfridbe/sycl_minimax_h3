"""h3sycl.py - Python side of libh3sycl: SYCL kernels for MiniMax H3 called on PyTorch-XPU tensors in place.

    import h3sycl
    y = h3sycl.int8_linear(x, weight_int8, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=True)

The library runs on PyTorch's own SYCL queue (torch.xpu.current_stream().sycl_queue), so its work is ordered with
PyTorch's and nothing is copied: tensors are passed as device pointers.
"""
import ctypes, os
import torch

_DT = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_lib = None
_ctx = None


def _load():
    global _lib, _ctx
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    lib = ctypes.CDLL(os.path.join(here, "libh3sycl.so"))
    lib.h3s_last_error.restype = ctypes.c_char_p
    lib.h3s_create.restype = ctypes.c_void_p
    lib.h3s_create.argtypes = [ctypes.c_void_p]
    lib.h3s_int8_linear.restype = ctypes.c_int
    lib.h3s_int8_linear.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cap = torch.xpu.current_stream().sycl_queue              # a PyCapsule around sycl::queue*
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    qptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, ctypes.pythonapi.PyCapsule_GetName(cap))
    ctx = lib.h3s_create(qptr)
    if not ctx:
        raise RuntimeError("h3sycl: " + lib.h3s_last_error().decode())
    _lib, _ctx = lib, ctx
    return lib


def available() -> bool:
    try:
        return torch.xpu.is_available() and _load() is not None
    except Exception:
        return False


def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False, convrot_groupsize=256):
    """comfy-kitchen's int8_linear on the matrix engine. x [..., K] float, weight int8 [N, K]."""
    lib = _load()
    if x.dtype not in _DT or out_dtype not in _DT:
        raise TypeError(f"h3sycl.int8_linear: unsupported dtype {x.dtype} -> {out_dtype}")
    dev = x.device
    K = x.shape[-1]; N = weight.shape[0]
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    w = weight if (weight.device == dev and weight.is_contiguous()) else weight.to(dev).contiguous()
    ws = weight_scale.to(device=dev, dtype=torch.float32).reshape(-1).contiguous()
    b = None if bias is None else bias.to(device=dev, dtype=out_dtype).contiguous()
    out = torch.empty((M, N), device=dev, dtype=out_dtype)
    rc = lib.h3s_int8_linear(_ctx, x2.data_ptr(), _DT[x.dtype], M, K, w.data_ptr(), N, ws.data_ptr(), ws.numel(),
                             0 if b is None else b.data_ptr(), out.data_ptr(), _DT[out_dtype],
                             int(convrot_groupsize) if convrot else 0)
    if rc != 0:
        raise RuntimeError("h3sycl: " + lib.h3s_last_error().decode())
    # x2 / w / ws / b must outlive the queued kernels: tie them to the result
    out._h3s_keep = (x2, w, ws, b)
    return out.reshape(*x.shape[:-1], N)
