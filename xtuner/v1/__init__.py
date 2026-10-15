import inspect
import os
from typing import Any, cast


def _patch_triton_autotune_for_determinism() -> None:
    """Pin kernel selection before importing FLA or other Triton operators."""
    import triton
    from triton.runtime.autotuner import Autotuner

    # FLA 0.5's CachedAutotuner constructs an Autotuner subclass directly,
    # bypassing triton.autotune. Pin the shared constructor as well, before
    # importing FLA: SP changes head counts and therefore its tuning keys.
    # FLA's on-disk config lookup can replace even a single pinned candidate.
    os.environ["FLA_CACHE_MODE"] = "disabled"
    original_init = Autotuner.__init__
    if not getattr(original_init, "_xtuner_deterministic_patched", False):
        supports_cache_results = "cache_results" in inspect.signature(original_init).parameters

        def deterministic_init(
            self: Any, fn: Any, arg_names: list[str], configs: list[Any], *args: Any, **kwargs: Any
        ) -> None:
            if configs:
                configs = configs[:1]
            if supports_cache_results:
                kwargs["cache_results"] = False
            else:
                kwargs.pop("cache_results", None)
            original_init(self, fn, arg_names, configs, *args, **kwargs)

        patched_init = cast(Any, deterministic_init)
        patched_init._xtuner_deterministic_patched = True
        patched_init._xtuner_original_init = original_init
        Autotuner.__init__ = patched_init

    original_autotune = triton.autotune
    if getattr(original_autotune, "_xtuner_deterministic_patched", False):
        return

    supports_autotune_cache = "cache_results" in inspect.signature(original_autotune).parameters

    def deterministic_autotune(configs: list[Any], *args: Any, **kwargs: Any) -> Any:
        if configs:
            configs = configs[:1]
        if supports_autotune_cache:
            kwargs["cache_results"] = False
        else:
            kwargs.pop("cache_results", None)
        return original_autotune(configs, *args, **kwargs)

    patched = cast(Any, deterministic_autotune)
    patched._xtuner_deterministic_patched = True
    patched._xtuner_original_autotune = original_autotune
    triton.autotune = deterministic_autotune


if os.getenv("XTUNER_DETERMINISTIC") == "true":
    _patch_triton_autotune_for_determinism()

from . import patch  # noqa: E402,F401
