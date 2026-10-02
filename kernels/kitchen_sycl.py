"""kitchen_sycl.py - a comfy-kitchen backend named "sycl": libh3sycl's kernels behind comfy-kitchen's function names.

    import kitchen_sycl
    kitchen_sycl.register()        # "sycl" goes first in comfy-kitchen's priority, for tensors on the Intel GPU

comfy-kitchen picks a backend per call from the registered constraints, so anything this backend does not cover
(other functions, other devices, other dtypes) keeps going to the backends that were there before.
"""
import sys
import time

import torch

import h3sycl

STATS = {}       # function name -> [calls, seconds]  (seconds only with sync=True; the queue is asynchronous)
_SYNC = False


def _input_act(x, input_act, act_weight, act_eps):
    # the activation the caller folded into this linear (comfy_kitchen/backends/_activations.py)
    if input_act in (None, "none"):
        return x
    if input_act == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if input_act == "swiglu":
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)
    if input_act == "rms_norm":
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=act_weight.to(x.dtype), eps=act_eps)
    raise ValueError(f"kitchen_sycl: unsupported input_act {input_act!r}")


def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False, convrot_groupsize=256,
                input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
    t0 = time.perf_counter() if _SYNC else 0.0
    x = _input_act(x, input_act, input_act_weight, input_act_eps)
    out = h3sycl.int8_linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize)
    if residual is not None:
        out = torch.addcmul(residual.to(out.dtype), out, residual_scale.to(out.dtype))
    s = STATS.setdefault("int8_linear", [0, 0.0])
    s[0] += 1
    if _SYNC:
        torch.xpu.synchronize()
        s[1] += time.perf_counter() - t0
    return out


def register(first=True, sync=False):
    """Registers the backend; returns the list of functions it took over (empty when the library is not usable)."""
    global _SYNC
    from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint
    from comfy_kitchen.registry import registry
    if not h3sycl.available():
        return []
    _SYNC = sync
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    caps = {
        "int8_linear": FunctionConstraints(
            params={"x": ParamConstraint(dtypes=floats),
                    "weight": ParamConstraint(dtypes=frozenset({torch.int8}))},
            default_devices=frozenset({"xpu"})),
    }
    registry.register("sycl", sys.modules[__name__], caps)
    rest = [b for b in registry._priority if b != "sycl"]
    registry.set_priority((["sycl"] + rest) if first else (rest + ["sycl"]))
    return sorted(caps)
