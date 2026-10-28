"""Read, validate and compare training environment snapshots without mutation.

Defaults belong to the shell launchers. Export all required settings before
preparing a recipe or importing packages that consume import-time switches.
"""

import os
import sys
import warnings
from importlib import import_module

from packaging.version import InvalidVersion, Version


# Names to validate and record, without Python-side defaults.
TRAINING_ENV_NAMES = (
    "XTUNER_USE_FA3",
    "OMP_NUM_THREADS",
    "XTUNER_DETERMINISTIC",
    "NCP_Q35V2_CACHE_CROSS_SOURCE_RMS",
    "NCP_Q35V2_NATIVE_RMS_BACKWARD",
    "NCP_Q35V2_TRITON_SOURCE_SCALE",
    "XTUNER_CHUNK_GRAD_ACCUMULATION",
    "XTUNER_GDN_CUSTOM_OP",
    "XTUNER_GDN_FUSED_QK_L2",
    "XTUNER_GDN_FUSED_NORM_GATE",
    "NCP_Q35V2_TORCH_RMS",
    "NCP_Q35V2_MUDD_RMS",
    "NCP_Q35V2_GDN_TORCH_INPUT_NORM",
    "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD",
)


def capture_training_environment() -> dict[str, str]:
    """Validate and record explicitly exported training settings.

    Returns:
        dict[str, str]: Effective kernel switches and CPU thread count.
    """
    missing = [name for name in TRAINING_ENV_NAMES if name not in os.environ]
    if missing:
        raise ValueError("Missing training environment; use a shell launcher or export: " + ", ".join(missing))
    effective = {name: os.environ[name] for name in TRAINING_ENV_NAMES}
    for name, value in effective.items():
        if name == "OMP_NUM_THREADS":
            if not value.isascii() or not value.isdecimal() or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        else:
            textual = name.startswith("XTUNER_") and name != "XTUNER_USE_FA3"
            allowed = ("true", "false") if textual else ("0", "1")
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return effective


def validate_training_environment(snapshot: dict[str, str] | None) -> None:
    """Reject runtime drift without changing import-time environment settings.

    Args:
        snapshot (dict[str, str] | None): Environment recorded during preparation.
    """
    if snapshot is None:
        warnings.warn(
            "Legacy recipe has no kernel_env snapshot; runtime environment cannot be verified.", stacklevel=2
        )
        return
    if "XTUNER_USE_FA3" not in snapshot:
        warnings.warn("Legacy kernel_env does not record FA3; FA3 cannot be verified.", stacklevel=2)
    differences = [
        f"{name}: recorded={value!r}, runtime={os.environ.get(name)!r}"
        for name, value in snapshot.items()
        if os.environ.get(name) != value
    ]
    if differences:
        raise ValueError(
            "Training environment differs from kernel_env; export the recorded values before launching "
            "or prepare a new recipe:\n" + "\n".join(differences)
        )


MIN_FLA_VERSION = Version("0.5.1")


def require_fla_version() -> str:
    """Validate the version of the FLA module selected by this Python environment.

    Returns:
        str: Version of the imported FLA module.
    """
    try:
        fla = import_module("fla")
    except Exception as error:
        raise RuntimeError(
            f"Qwen3.5 requires FLA >= {MIN_FLA_VERSION}; cannot import fla: {error}; Python={sys.executable}"
        ) from error

    actual = getattr(fla, "__version__", None)
    context = f"found={actual!r}; module={getattr(fla, '__file__', '<unknown>')}; Python={sys.executable}"
    try:
        installed = Version(str(actual))
    except InvalidVersion as error:
        raise RuntimeError(
            f"Qwen3.5 requires FLA >= {MIN_FLA_VERSION}; cannot determine a valid FLA version; {context}"
        ) from error
    if installed < MIN_FLA_VERSION:
        raise RuntimeError(f"Qwen3.5 requires FLA >= {MIN_FLA_VERSION}; {context}")
    print(f"FLA_VERSION_CHECK_OK {context}", flush=True)
    return str(actual)


if __name__ == "__main__":
    try:
        for name, value in capture_training_environment().items():
            print(f"{name}={value}")
        require_fla_version()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
