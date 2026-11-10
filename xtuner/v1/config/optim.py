import os
import re
from abc import abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Annotated

from xtuner.v1.optim import ConceptMuon, LocalDTensorAdamW, Muon, SwapAdamW
from xtuner.v1.utils import get_logger


logger = get_logger()


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0
    skip_grad_norm_threshold: Annotated[
        float | None, Parameter(help="Gradient norm threshold for skipping optimizer step.")
    ] = None
    fused_grad_clip: Annotated[
        bool,
        Parameter(help="Fold gradient clipping and finite checks into a fused optimizer step."),
    ] = False

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8
    embedding_weight_decay: Annotated[
        Optional[float],
        Parameter(help="Weight decay for embedding parameters. None uses weight_decay."),
    ] = None
    one_d_weight_decay: Annotated[
        Optional[float],
        Parameter(help="Weight decay for 1D / bias parameters. None uses weight_decay."),
    ] = None
    foreach: Annotated[Optional[bool], Parameter(help="Use foreach implementation for AdamW")] = None
    fused: Annotated[Optional[bool], Parameter(help="Use fused implementation for AdamW")] = None
    local_dtensor_step: Annotated[
        bool,
        Parameter(help="Dispatch fused AdamW directly on DTensor local shards."),
    ] = False
    swap_optimizer: Annotated[Optional[bool], Parameter(help="Swap optimizer states to host memory.")] = False

    def build(self, model):
        if self.swap_optimizer and self.local_dtensor_step:
            raise ValueError("swap_optimizer and local_dtensor_step are mutually exclusive")

        params = [p for p in model.parameters() if p.requires_grad]

        trainable_parameters_names = model.trainable_parameters()
        trainable_names = [name for name, _ in trainable_parameters_names]
        untrainable_names = []
        num_total_requires_grad = 0
        num_total = 0
        for name, params_ in model.named_parameters():
            num_total += params_.numel()
            num_total_requires_grad += params_.numel() if name in trainable_names else 0
            if name not in trainable_names:
                untrainable_names.append(name)

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad / 1e6:.2f}M, total parameters: {num_total / 1e6:.2f}M"
            )
            logger.info(f"Untrainable parameters names: {untrainable_names}")
        if self.embedding_weight_decay is not None or self.one_d_weight_decay is not None:
            if self.swap_optimizer or self.local_dtensor_step:
                raise ValueError(
                    "embedding/1D weight decay split is not supported with swap_optimizer or local_dtensor_step"
                )
            embed_wd = self.weight_decay if self.embedding_weight_decay is None else self.embedding_weight_decay
            one_d_wd = self.weight_decay if self.one_d_weight_decay is None else self.one_d_weight_decay
            decay_params, embed_params, one_d_params = [], [], []
            for name, param in model.named_parameters():
                if name not in trainable_names:
                    continue
                is_embed = "embed_tokens" in name or name.endswith("embeddings.weight")
                is_1d = param.ndim < 2 or param.numel() in param.shape
                if is_embed:
                    embed_params.append(param)
                elif is_1d:
                    one_d_params.append(param)
                else:
                    decay_params.append(param)
            if dist.get_rank() == 0:
                logger.info(
                    "AdamW groups: "
                    f"decay={sum(param.numel() for param in decay_params)} wd={self.weight_decay}, "
                    f"embed={sum(param.numel() for param in embed_params)} wd={embed_wd}, "
                    f"one_d={sum(param.numel() for param in one_d_params)} wd={one_d_wd}"
                )
            param_groups = []
            if decay_params:
                param_groups.append({"params": decay_params, "weight_decay": self.weight_decay})
            if embed_params:
                param_groups.append({"params": embed_params, "weight_decay": embed_wd})
            if one_d_params:
                param_groups.append({"params": one_d_params, "weight_decay": one_d_wd})
            return torch.optim.AdamW(
                param_groups,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                foreach=self.foreach,
                fused=self.fused,
            )
        if self.swap_optimizer:
            if self.fused:
                raise ValueError("fused AdamW is not supported together with optimizer-state swapping")
            return SwapAdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                foreach=self.foreach,
            )
        if self.local_dtensor_step and self.fused is not True:
            raise ValueError("local_dtensor_step requires fused=True")
        optimizer_cls = LocalDTensorAdamW if self.local_dtensor_step else torch.optim.AdamW
        return optimizer_cls(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            foreach=self.foreach,
            fused=self.fused,
        )


class ParamOverride(BaseModel):
    """Algorithm and weight-decay settings for one regex-selected parameter set.

    ``MuonConfig.param_overrides`` maps ordered regex patterns to these settings. It only affects trainable parameters
    handled by ``MuonConfig``; it does not apply to ``AdamWConfig``. A pattern that matches no trainable parameter is
    silently ignored, and an entry whose fields are both ``None`` leaves the inherited settings unchanged.

    Rule ordering:
        For each parameter name, the first matching pattern wins; later patterns are not evaluated for that name.
        Put exact rules before broad rules. Repeating the exact same key in a Python dict is different: Python keeps
        only the last value before ``MuonConfig`` can validate it.

    Inheritance and precedence:
        ``algorithm=None`` inherits the model's ``param_options`` algorithm when ``concept_muon=True`` and the model
        provides it. Otherwise the default shape/name classification is used: non-degenerate 2D parameters other than
        embeddings and the LM head use Muon; all other parameters use AdamW.

        ``weight_decay=None`` inherits the first available value in this highest-to-lowest priority order:

        1. The ``weight_decay`` in the matching ``param_overrides`` entry.
        2. The model's ``param_options`` decay when ``concept_muon=True``.
        3. ``embedding_weight_decay`` for embeddings or ``one_d_weight_decay`` for 1D/bias parameters in AdamW groups.
        4. The global ``weight_decay`` (default ``0.1``).

        Different values at different priority levels are intentional overrides, not errors. In particular, a user
        override may replace the model or category decay without changing the algorithm.

    Shared parameters:
        Aliases of one tied parameter are resolved separately and then merged field by field. One alias may supply the
        algorithm while another supplies decay. Conflicting non-None values for either field raise ``ValueError``;
        aliases cannot select independent algorithms or decay values for the same parameter object.

    Validation errors:
        ``ValueError`` (including Pydantic validation errors) is raised for the following configurations:

        - Nonempty legacy ``weight_decay_overrides`` together with nonempty ``param_overrides``. Legacy overrides used
          alone are normalized to ``param_overrides`` with ``algorithm=None``.
        - Conflicting algorithm or decay values from aliases of a shared parameter.
        - A user algorithm that differs from the model's ``param_options`` algorithm.
        - An AdamW parameter that matches both embedding and 1D categories when their explicitly configured decay
          values differ.
        - An invalid regex, an algorithm other than ``"muon"`` or ``"adamw"``, or unknown override fields.
        - An explicit Muon assignment for a parameter with fewer than two dimensions. ``ConceptMuon`` additionally
          requires 2D parameters and rejects higher-rank tensors during optimizer construction.

    Example:
        .. code-block:: python

            optim_cfg = MuonConfig(
                weight_decay=0.1,
                embedding_weight_decay=0.0,
                one_d_weight_decay=0.0,
                param_overrides={
                    r"^projection\\.weight$": {"algorithm": "adamw", "weight_decay": 0.03},
                    r"^projection\\.": {"weight_decay": 0.2},
                },
            )

        Here ``projection.weight`` uses AdamW with decay ``0.03``. Other projection parameters, including its bias,
        inherit their default algorithm and use decay ``0.2``. Parameters outside ``projection`` are unchanged.

    Inspecting effective settings:
        Configuration validation does not print the resolved groups. To verify the actual algorithm and decay after
        all inheritance and overrides, inspect ``optimizer.param_groups`` directly:

        .. code-block:: python

            optimizer = optim_cfg.build(model)
            groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
            for name, param in model.named_parameters():
                group = groups[id(param)]
                print(name, group["algorithm"], group["weight_decay"])
    """

    model_config = ConfigDict(extra="forbid")
    algorithm: Annotated[
        Literal["muon", "adamw"] | None,
        Parameter(help="Optimizer algorithm. None inherits model metadata or default shape/name classification."),
    ] = None
    weight_decay: Annotated[
        float | None,
        Parameter(help="Weight decay override. None inherits model, category, then global defaults."),
    ] = None


class MuonConfig(OptimConfig):
    concept_muon: Annotated[
        bool,
        Parameter(
            help="Use EMA momentum, quintic Newton-Schulz and model-owned projection split "
            "layouts (ConceptMuon); FP32 optimizer state is kept while parameters stay low-precision"
        ),
    ] = False
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.1
    momentum: Annotated[float, Parameter(help="Momentum coefficients for Muon optimizer")] = 0.95
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for AdamW optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Muon optimizer")] = 1e-8
    adjust_lr: Annotated[
        Literal["rms_norm", "spectral_norm", "none"], Parameter(help="Method for adjusting lr in Muon")
    ] = "rms_norm"
    extra_scale_factor: Annotated[
        float,
        Parameter(help="Multiplier for rms_norm: scale * sqrt(max(rows, cols)); 0.2 keeps the historical RMS ratio."),
    ] = 0.2
    enable_all2all: Annotated[
        bool, Parameter(help="Allow all-to-all comm strategy; set False on topologies where all-to-all is unreliable")
    ] = True
    clip_grad_mode: Annotated[
        Literal["all", "adamw_only"],
        Parameter(help="Gradient clipping policy: clip all parameters or only the AdamW parameter groups"),
    ] = "adamw_only"
    adamw_foreach: Annotated[
        bool, Parameter(help="Use foreach updates for parameters handled by Muon's AdamW fallback")
    ] = True
    nesterov: Annotated[bool, Parameter(help="Use Nesterov momentum in Muon updates")] = True
    embedding_weight_decay: Annotated[
        float | None,
        Parameter(help="Default weight decay for embedding parameters; param_overrides takes precedence."),
    ] = None
    one_d_weight_decay: Annotated[
        float | None,
        Parameter(help="Default weight decay for 1D / bias parameters; param_overrides takes precedence."),
    ] = None
    param_overrides: Annotated[
        dict[str, ParamOverride],
        Parameter(
            help="Ordered regex-to-algorithm/weight-decay overrides. First match per name wins; "
            "see ParamOverride for inheritance, conflict rules, and resolved-group inspection."
        ),
    ] = {}

    @model_validator(mode="before")
    @classmethod
    def _normalize_config(cls, data):
        if isinstance(data, dict):
            legacy_overrides = data.get("weight_decay_overrides")
            if legacy_overrides:
                if data.get("param_overrides"):
                    raise ValueError("weight_decay_overrides and param_overrides cannot be used together")
                data = {
                    **data,
                    "param_overrides": {
                        pattern: {"weight_decay": decay} for pattern, decay in legacy_overrides.items()
                    },
                }
            if "weight_decay_overrides" in data:
                data = {key: value for key, value in data.items() if key != "weight_decay_overrides"}
            # The split-projection layout expects Muon-group gradients to be clipped
            # together with the AdamW groups; keep the historical default otherwise.
            if data.get("concept_muon") and "clip_grad_mode" not in data:
                data = {**data, "clip_grad_mode": "all"}
        return data

    @field_validator("param_overrides")
    @classmethod
    def _validate_param_override_patterns(cls, overrides):
        for pattern in overrides:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"Invalid parameter override pattern {pattern!r}: {error}") from error
        return overrides

    def _resolve_param_overrides(self, model, trainable_params):
        resolved = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            if param not in trainable_params:
                continue
            for pattern, override in self.param_overrides.items():
                if not re.search(pattern, name):
                    continue
                previous = resolved.get(param)
                if previous is None:
                    resolved[param] = override
                    break
                updates = {}
                for field in ("algorithm", "weight_decay"):
                    value = getattr(override, field)
                    previous_value = getattr(previous, field)
                    if value is not None and previous_value is not None and value != previous_value:
                        raise ValueError(
                            f"Conflicting parameter overrides for shared parameter at {name!r}: "
                            f"{field}={previous_value!r} versus {value!r}"
                        )
                    if previous_value is None:
                        updates[field] = value
                resolved[param] = previous.model_copy(update=updates)
                break
        return resolved

    def build(self, model):
        trainable_parameters_names = model.trainable_parameters()
        trainable_names = {name for name, _ in trainable_parameters_names}
        trainable_params = {param for _, param in trainable_parameters_names}

        muon_split_sizes = {}
        muon_qkv_split_shapes = {}
        metadata = {}
        if self.concept_muon:
            provider = getattr(model, "get_muon_metadata", None)
            if callable(provider):
                metadata = provider()
                muon_split_sizes.update(metadata.get("split_sizes", {}))
        param_options = metadata.get("param_options", {})
        param_overrides = self._resolve_param_overrides(model, trainable_params)

        explicit_adamw_params = {}
        for module in model.modules():
            get_muon_split_sizes = getattr(module, "get_muon_split_sizes", None)
            if callable(get_muon_split_sizes):
                muon_split_sizes.update(get_muon_split_sizes())
            get_muon_qkv_split_shapes = getattr(module, "get_muon_qkv_split_shapes", None)
            if callable(get_muon_qkv_split_shapes):
                muon_qkv_split_shapes.update(get_muon_qkv_split_shapes())

        untrainable_names = []
        num_total = 0
        num_total_requires_grad = 0
        num_muon_regular = 0
        num_muon_moe = 0
        num_adamw = 0

        num_experts: int
        if hasattr(model.config, "text_config"):
            num_experts = getattr(model.config.text_config, "n_routed_experts", 1) or 1
        else:
            num_experts = getattr(model.config, "n_routed_experts", 1) or 1

        is_moe_model = num_experts > 1

        # Expert parameter patterns for MoE models
        # Note: fused_w1w3 contains both w1 and w3 weights, so num_experts = 2 * n_routed_experts
        fused_w1w3_patterns = ("fused_w1w3",)
        other_expert_patterns = ("fused_w2", "fused_w1", "fused_w3")

        # Separate Muon params into regular and MoE expert params
        # fused_w1w3 has 2 * num_experts (w1 and w3 each have num_experts)
        # other expert params have num_experts
        muon_params_regular = []
        muon_params_moe_fused_w1w3 = []  # num_experts = 2 * n_routed_experts
        muon_params_moe_other = []  # num_experts = n_routed_experts
        adamw_params = []
        adamw_embed_params = []
        adamw_1d_params = []

        weight_decays = {}
        for name, p in model.named_parameters():
            n = p.numel()
            num_total += n
            if name not in trainable_names:
                untrainable_names.append(name)
                continue

            num_total_requires_grad += n
            options = param_options.get(p, {})
            algorithm = options.get("algorithm")
            if algorithm not in (None, "muon", "adamw"):
                raise ValueError(f"Invalid optimizer algorithm {algorithm!r} for {name!r} in model metadata")
            override = param_overrides.get(p)
            if override is not None and override.algorithm is not None:
                if algorithm is not None and override.algorithm != algorithm:
                    raise ValueError(
                        f"Conflicting optimizer algorithms for {name!r}: model metadata specifies "
                        f"{algorithm!r}, parameter override specifies {override.algorithm!r}"
                    )
                algorithm = override.algorithm

            decay = options.get("weight_decay")
            if override is not None and override.weight_decay is not None:
                decay = override.weight_decay
            if decay is not None:
                weight_decays[p] = decay

            if algorithm == "muon":
                if p.ndim < 2:
                    raise ValueError(
                        f"Cannot use Muon for {name!r} with shape {tuple(p.shape)}; "
                        "Muon requires at least 2D parameters"
                    )
                is_muon_tensor = True
            elif algorithm == "adamw":
                is_muon_tensor = False
            else:
                # By default, Muon only applies to non-degenerate 2D matrices. Higher-rank
                # tensors and effectively-1D tensors use AdamW unless explicitly overridden.
                is_muon_tensor = (
                    p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name and p.numel() not in p.shape
                )

            is_embed = "embed_tokens" in name or name.endswith("embeddings.weight")
            is_1d = p.ndim < 2 or p.numel() in p.shape
            if (
                not is_muon_tensor
                and is_embed
                and is_1d
                and self.embedding_weight_decay is not None
                and self.one_d_weight_decay is not None
                and self.embedding_weight_decay != self.one_d_weight_decay
            ):
                raise ValueError(
                    f"Conflicting weight decay defaults for {name!r}: it matches both embedding "
                    f"({self.embedding_weight_decay}) and 1D ({self.one_d_weight_decay}) defaults"
                )

            if p in param_options and algorithm == "adamw":
                if decay is None:
                    if self.embedding_weight_decay is not None and is_embed:
                        decay = self.embedding_weight_decay
                    elif self.one_d_weight_decay is not None and is_1d:
                        decay = self.one_d_weight_decay
                    else:
                        decay = self.weight_decay
                explicit_adamw_params.setdefault(decay, []).append(p)
                num_adamw += n
                continue

            if is_muon_tensor:
                if is_moe_model and any(pattern in name for pattern in fused_w1w3_patterns):
                    muon_params_moe_fused_w1w3.append(p)
                    num_muon_moe += n
                    if dist.get_rank() == 0:
                        logger.info(f"Muon moe param: [{name}]: shape = {p.shape}")
                elif is_moe_model and any(pattern in name for pattern in other_expert_patterns):
                    muon_params_moe_other.append(p)
                    num_muon_moe += n
                    if dist.get_rank() == 0:
                        logger.info(f"Muon moe param: [{name}]: shape = {p.shape}")
                else:
                    muon_params_regular.append(p)
                    num_muon_regular += n
                    if dist.get_rank() == 0:
                        logger.info(f"Muon regular param: [{name}]: shape = {p.shape}")
            else:
                if self.embedding_weight_decay is not None and is_embed:
                    adamw_embed_params.append(p)
                elif self.one_d_weight_decay is not None and is_1d:
                    adamw_1d_params.append(p)
                else:
                    adamw_params.append(p)
                num_adamw += n
                if dist.get_rank() == 0:
                    logger.info(f"AdamW param: [{name}]: shape = {p.shape}")

        # Build parameter groups
        clip_muon_grad = self.clip_grad_mode == "all"
        param_groups = []
        if muon_params_regular:
            param_groups.append(dict(params=muon_params_regular, clip_grad=clip_muon_grad))
        # fused_w1w3: w1 and w3 are fused, so num_experts = 2 * n_routed_experts
        if muon_params_moe_fused_w1w3:
            param_groups.append(
                dict(
                    params=muon_params_moe_fused_w1w3,
                    num_experts=2 * num_experts,
                    clip_grad=clip_muon_grad,
                )
            )
        # Other expert params: num_experts = n_routed_experts
        if muon_params_moe_other:
            param_groups.append(dict(params=muon_params_moe_other, num_experts=num_experts, clip_grad=clip_muon_grad))
        if adamw_embed_params:
            param_groups.append(
                dict(
                    params=adamw_embed_params,
                    algorithm="adamw",
                    clip_grad=True,
                    adamw_foreach=self.adamw_foreach,
                    weight_decay=self.embedding_weight_decay,
                )
            )
        if adamw_1d_params:
            param_groups.append(
                dict(
                    params=adamw_1d_params,
                    algorithm="adamw",
                    clip_grad=True,
                    adamw_foreach=self.adamw_foreach,
                    weight_decay=self.one_d_weight_decay,
                )
            )
        if adamw_params:
            param_groups.append(
                dict(params=adamw_params, algorithm="adamw", clip_grad=True, adamw_foreach=self.adamw_foreach)
            )

        # Sanity check: ensure all trainable parameters are assigned to optimizer
        for decay, parameters in explicit_adamw_params.items():
            param_groups.append(
                dict(
                    params=parameters,
                    algorithm="adamw",
                    clip_grad=True,
                    adamw_foreach=self.adamw_foreach,
                    weight_decay=decay,
                )
            )

        if weight_decays:
            resolved_param_groups = []
            for group in param_groups:
                params_by_decay = {}
                for p in group["params"]:
                    decay = weight_decays.get(p, group.get("weight_decay", self.weight_decay))
                    params_by_decay.setdefault(decay, []).append(p)
                for decay, parameters in params_by_decay.items():
                    resolved_param_groups.append({**group, "params": parameters, "weight_decay": decay})
            param_groups = resolved_param_groups

        total_assigned = sum(len(group["params"]) for group in param_groups)
        total_trainable = len(trainable_names)
        assert total_assigned == total_trainable, (
            f"Parameter assignment mismatch: {total_assigned} assigned vs {total_trainable} trainable. "
            f"Some parameters may have their gradients not zeroed by optimizer.zero_grad()."
        )

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad / 1e6:.2f}M, total parameters: {num_total / 1e6:.2f}M"
            )
            if is_moe_model:
                logger.info(
                    f"Muon params: {(num_muon_regular + num_muon_moe) / 1e6:.2f}M "
                    f"(regular: {num_muon_regular / 1e6:.2f}M, MoE expert: {num_muon_moe / 1e6:.2f}M), "
                    f"AdamW params: {num_adamw / 1e6:.2f}M (counts by numel)"
                )
                logger.info(
                    f"Detected MoE model with {num_experts} routed experts, "
                    f"fused_w1w3 uses num_experts={2 * num_experts} (w1+w3), "
                    f"other expert params use num_experts={num_experts}"
                )
            else:
                logger.info(
                    f"Muon params: {num_muon_regular / 1e6:.2f}M, AdamW params: {num_adamw / 1e6:.2f}M (counts by numel)"
                )
            logger.info(f"Untrainable parameters names: {untrainable_names}")

        optimizer_cls = ConceptMuon if self.concept_muon else Muon
        optimizer_kwargs = {}
        if self.concept_muon:
            optimizer_kwargs["muon_adamw_ranges"] = metadata.get("adamw_ranges", {})
        optimizer = optimizer_cls(
            param_groups,
            lr=self.lr,
            mu=self.momentum,
            betas=self.betas,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            flatten=True,
            adjust_lr=self.adjust_lr,
            extra_scale_factor=self.extra_scale_factor,
            use_triton=os.getenv("XTUNER_MUON_USE_TRITON", "0").lower() in ("1", "true", "yes", "on"),
            epsilon=self.eps,
            enable_all2all=self.enable_all2all,
            muon_split_sizes=muon_split_sizes,
            muon_qkv_split_shapes=muon_qkv_split_shapes,
            **optimizer_kwargs,
        )

        return optimizer


class LRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr_type: Annotated[
        Literal["cosine", "linear", "constant", "wsd"], Parameter(help="Type of learning rate schedule")
    ] = "constant"
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6

    warmup_steps: Annotated[int | None, Parameter(help="Explicit warmup updates; overrides warmup_ratio")] = None
    decay_steps: Annotated[int | None, Parameter(help="Full LR schedule updates, independent of run length")] = None
    wsd_decay_steps: Annotated[int | None, Parameter(help="Number of updates in the WSD decay tail")] = None
