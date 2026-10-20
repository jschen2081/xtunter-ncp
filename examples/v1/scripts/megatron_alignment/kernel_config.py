"""Kernel policy shared by the native ConceptLM training launchers.

This module imports only the standard library. Apply it before importing FLA
or xTuner, including before the standard CLI initializes its package.
"""

import os


TRAINING_KERNEL_ENV = {
    "XTUNER_DETERMINISTIC": "true",
    "XTUNER_GDN_CUSTOM_OP": "true",
    "XTUNER_GDN_FUSED_QK_L2": "true",
    "XTUNER_GDN_FUSED_NORM_GATE": "true",
    "NCP_Q35V2_TORCH_RMS": "1",
    "NCP_Q35V2_MUDD_RMS": "1",
    "NCP_Q35V2_GDN_TORCH_INPUT_NORM": "1",
    "NCP_Q35V2_TRITON_DUAL_CROSS_MUDD": "0",
}


def configure_training_kernels() -> None:
    """Apply the common differentiable training kernel policy."""
    os.environ.update(TRAINING_KERNEL_ENV)
    os.environ.pop("XTUNER_HF_IMPL", None)


if __name__ == "__main__":
    # Plain assignments let shell launchers export the same policy without eval.
    for name, value in TRAINING_KERNEL_ENV.items():
        print(f"{name}={value}")
