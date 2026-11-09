"""Qwen3.5-35B-A3B speedtest with OLMo indexed tokens and mixed Muon/AdamW.

Data loading follows jyhuang/conceptlm-moe-stable at b07c9e9a.

For the full no-EP 8192-token benchmark on 140 GiB H200 GPUs, enable local
expert packing with INTRA_LAYER_MICRO_BATCH=2 and RECOMPUTE_RATIO=0.75.
"""

import os
from pathlib import Path

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.train import TrainerConfig


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0, 1, false, or true")
    return raw in {"1", "true"}


work_dir = Path(os.environ["WORK_DIR"])
seq_len = int(os.environ.get("SEQ_LEN", "8192"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "4096"))
# This groups sequence contexts inside each layer, not a dense batch dimension.
intra_layer_micro_batch = int(os.environ.get("INTRA_LAYER_MICRO_BATCH", "1"))
world_size = int(os.environ.get("WORLD_SIZE", "8"))
total_step = int(os.environ.get("TOTAL_STEP", "100"))
lr = float(os.environ.get("LR", "1e-4"))
lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", "10"))
lr_min_ratio = float(os.environ.get("LR_MIN_RATIO", "0.1"))
model_compile = _bool_env("MODEL_COMPILE", False)
recompute_ratio = float(os.environ.get("RECOMPUTE_RATIO", "0.0"))
profile_steps = [int(step) for step in os.environ.get("PROFILE_STEP", "").split(",") if step.strip()]
if min(global_batch_size, world_size, total_step, intra_layer_micro_batch) < 1 or seq_len < 2:
    raise ValueError("Batch sizes, world size, and steps must be positive; SEQ_LEN must be >= 2")
if lr <= 0 or not 0 <= lr_min_ratio <= 1:
    raise ValueError("LR must be positive and LR_MIN_RATIO must be between 0 and 1")
if not 0 <= lr_warmup_steps < total_step:
    raise ValueError("LR_WARMUP_STEPS must be nonnegative and smaller than TOTAL_STEP")
if global_batch_size % world_size:
    raise ValueError("GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE (SP=TP=EP=1)")
if (global_batch_size // world_size) % intra_layer_micro_batch:
    raise ValueError("GLOBAL_BATCH_SIZE / WORLD_SIZE must be divisible by INTRA_LAYER_MICRO_BATCH")
if not 0 <= recompute_ratio <= 1:
    raise ValueError("RECOMPUTE_RATIO must be between 0 and 1")
if any(step < 0 or step >= total_step for step in profile_steps):
    raise ValueError("PROFILE_STEP uses zero-based steps and must be smaller than TOTAL_STEP")

# Keep the official 40-layer, 256-expert/top-8 text tower intact.
model_cfg = Qwen3_5_VLTextMoE35BA3BConfig(
    compile_cfg=model_compile,
    ep_size=1,
    dispatcher=None,
    expert_backend=os.environ.get("EXPERT_BACKEND", "grouped_gemm"),
    mtp_config=None,
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=None,
    tokenizer_path=os.environ["TOKENIZER_PATH"],
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(
                    name="olmo_500b_qwen35",
                    anno_path=os.environ["DATA_PATH"],
                    class_name="MegatronIndexedDataset",
                    sequence_length=seq_len + 1,  # Causal shifting consumes one source token.
                    cache_dir=work_dir / "data_cache",
                ),
                "tokenize_fn": PretrainTokenizeFunctionConfig(),
            }
        ],
        # Exact indexed L+1 blocks need no repacking. This collator shifts to L
        # without padding while satisfying the trainer's fixed-shape compile check.
        pack_level="none",
        pack_max_length=seq_len + 1,
        pack_to_max_length=True,
        group_by_length=False,
        collator="single_sequence_sft_llm_collator",
        num_workers=2,
    ),
    # Native MuonConfig routes embeddings, LM head, norms, biases and conv1d to AdamW.
    # Embeddings, LM head and 1D parameters use zero weight decay.
    optim_cfg=MuonConfig(
        lr=lr,
        momentum=0.9,
        nesterov=True,
        adjust_lr="rms_norm",
        weight_decay=0.1,
        embedding_weight_decay=0.0,
        weight_decay_overrides={r"(?:^|\.)lm_head\.": 0.0},
        one_d_weight_decay=0.0,
        betas=(0.9, 0.95),
        clip_grad_mode="all",
        max_grad_norm=1.0,
    ),
    lr_cfg=LRConfig(
        lr_type="cosine",
        warmup_steps=lr_warmup_steps,
        decay_steps=total_step,
        lr_min=lr * lr_min_ratio,
    ),
    loss_cfg=CELossConfig(mode="chunk", chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "128"))),
    fsdp_cfg=FSDPConfig(
        tp_size=1,
        ep_size=1,
        cpu_offload=False,
        reshard_after_forward=_bool_env("RESHARD_AFTER_FORWARD", True),
        gradient_accumulation_no_sync=_bool_env("FSDP_GRAD_ACCUMULATION_NO_SYNC", False),
        recompute_ratio=recompute_ratio,
        checkpoint_preserve_rng_state=False,
        torch_compile=model_compile,
    ),
    sp_size=1,
    global_batch_size=global_batch_size,
    intra_layer_micro_batch=intra_layer_micro_batch,
    total_step=total_step,
    checkpoint_interval=None,
    hf_interval=None,
    auto_resume=False,
    debug_skip_save=True,
    profile_step=profile_steps or None,
    profile_memory=False,
    exp_tracker="both",
    work_dir=work_dir,
    seed=42,
)
