"""Work around a reproduced LLVM SLP crash in Qwen3.5 SP8 backward.

Torch 2.9.1 / Triton 3.5.1 on SM90 aborts while compiling the (2, 64, 2) candidate of this BF16 multi-output reduction.
Keep the other candidates and all numerical operations unchanged. Remove after validating the reproducer against a
fixed compiler. Generated imports also cover spawn compile workers.
"""

import functools
import logging
import re

import torch


logger = logging.getLogger(__name__)
_KERNEL = re.compile(r"triton_(?:red|unk)_fused__to_copy_add_mul_rms_norm_native_backward_select_slice_sum_[0-9]+")


def _supported_versions():
    try:
        import triton
    except ImportError:
        return False
    return torch.__version__.split("+")[0] == "2.9.1" and triton.__version__ == "3.5.1"


def _affected(size_hints, triton_meta, inductor_meta):
    device = triton_meta.get("device")
    signature = triton_meta.get("signature", {})
    expected = {f"in_ptr{i}": "*fp32" if i == 5 else "*bf16" for i in range(9)}
    expected.update({f"out_ptr{i}": "*bf16" for i in range(10)})
    pointers = {k: v for k, v in signature.items() if k.startswith(("in_ptr", "out_ptr"))}
    return (
        getattr(device, "type", None) == "cuda"
        and getattr(device, "cc", None) == 90
        and size_hints == {"x": 256, "r0_": 4096}
        and _KERNEL.fullmatch(inductor_meta.get("kernel_name", "")) is not None
        and pointers == expected
    )


def _unsafe(config):
    return config.kwargs.get("XBLOCK") == 2 and config.kwargs.get("R0_BLOCK") == 64 and config.num_warps == 2


def reduction(*args, **kwargs):
    """Filter after cache lookup, so cached candidates cannot bypass the
    guard."""
    from torch._inductor.runtime import triton_heuristics

    decorate = triton_heuristics.reduction(*args, **kwargs)

    def guarded(fn):
        tuner = decorate(fn)
        if not _supported_versions() or not _affected(
            tuner.size_hints, {**tuner.triton_meta, "device": tuner.device_props}, tuner.inductor_meta
        ):
            return tuner
        safe = [c for c in tuner.configs if not _unsafe(c)]
        if len(safe) != len(tuner.configs):
            if not safe:
                # A stale cache may contain only the excluded candidate.
                safe = [
                    c
                    for c in triton_heuristics._reduction_configs(
                        size_hints=tuner.size_hints,
                        inductor_meta=tuner.inductor_meta,
                        num_dynamic=sum("ks" in k for k in tuner.triton_meta["signature"]),
                    )
                    if not _unsafe(c)
                ]
                tuner.save_cache_hook = None
            if not safe:
                raise RuntimeError("No safe Qwen3.5 reduction candidates remain")
            logger.warning(
                "Qwen3.5 LLVM SLP workaround: excluding XBLOCK=2, R0_BLOCK=64, num_warps=2 for %s", fn.__name__
            )
            tuner.configs = safe
        # Dynamic register-pressure tuning / coordinate descent can introduce
        # candidates later. Redirect only the same unsafe tuple to a safe one.
        tuner._qwen35_safe_config = safe[0]
        return tuner

    return guarded


def _safe_config(tuner, config):
    fallback = getattr(tuner, "_qwen35_safe_config", None)
    return fallback if fallback is not None and _unsafe(config) else config


def _precompile_config(self, config, *args, **kwargs):
    return _ORIGINAL_PRECOMPILE(self, _safe_config(self, config), *args, **kwargs)


def patch_inductor_qwen35_reduction():
    if torch.__version__.split("+")[0] != "2.9.1":
        return
    try:
        import triton
    except ImportError:
        return
    if triton.__version__ != "3.5.1":
        return
    from torch._inductor.codegen.triton import TritonKernel
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    global _ORIGINAL_PRECOMPILE
    if CachingAutotuner._precompile_config is not _precompile_config:
        _ORIGINAL_PRECOMPILE = CachingAutotuner._precompile_config
        CachingAutotuner._precompile_config = _precompile_config

    original = TritonKernel.codegen_kernel
    if getattr(original, "_xtuner_qwen35_slp_guard", False):
        return

    @functools.wraps(original)
    def codegen(self, *args, **kwargs):
        source = original(self, *args, **kwargs)
        marker = "@triton_heuristics.reduction("
        if marker in source:
            source = source.replace(
                marker,
                "from xtuner.v1.patch.inductor_qwen35_reduction import reduction as _qwen35_reduction\n"
                "@_qwen35_reduction(",
                1,
            )
        return source

    codegen._xtuner_qwen35_slp_guard = True
    TritonKernel.codegen_kernel = codegen


patch_inductor_qwen35_reduction()
