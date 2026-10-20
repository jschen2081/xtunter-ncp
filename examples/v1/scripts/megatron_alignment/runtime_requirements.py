"""Runtime requirements shared by Qwen3.5 replay and independent training."""

import sys
from importlib import import_module

from packaging.version import InvalidVersion, Version

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
        require_fla_version()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
