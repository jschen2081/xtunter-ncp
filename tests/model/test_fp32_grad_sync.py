"""CPU coverage of ignored-FP32 registration and real distributed reductions."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense, Qwen3_5_VLTextDenseConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE, Qwen3_5_VLTextMoEConfig
from xtuner.v1.ops.gated_deltanet.dtensor import dtensor_full_tensor, dtensor_to_local


def _model(family, kind, mesh, ep):
    cls = Qwen3_5_VLTextMoE if family == "moe" else Qwen3_5_VLTextDense
    config_cls = Qwen3_5_VLTextMoEConfig if family == "moe" else Qwen3_5_VLTextDenseConfig
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hf_save_cfg=config_cls.model_fields["hf_save_cfg"].default,
        expert_tp_size=1,
        layers_type=["linear_attention"],
        tie_word_embeddings=False,
        n_routed_experts=2,
    )
    model._world_mesh = mesh
    model.ep_mesh = mesh["default.ep"] if family == "moe" and "default.ep" in (mesh.mesh_dim_names or ()) else None
    model.expert_tp_mesh = None
    model.layers = nn.ModuleDict({"0": nn.Module()})
    attn = nn.Module()
    attn.A_log = nn.Parameter(torch.zeros(4))
    attn.dt_bias = nn.Parameter(torch.zeros(4))
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.ones(4))
    setattr(attn, "norm" if kind == "gdn" else "o_norm", norm)
    model.layers["0"].self_attn = attn
    if family == "moe" and ep > 1:
        model._replicate_other_params(model)
    model.load_spec_mapping = {}
    for name, _ in model.named_parameters():
        keys = model.to_hf_key_list(name)
        if family == "dense":
            keys = [key.replace("model.", "model.language_model.", 1) for key in keys]
        model.load_spec_mapping[name] = SimpleNamespace(global_hf_keys=keys)
    ignored = set()

    def capture_fsdp(target, **kwargs):
        ignored.update(kwargs.get("ignored_params") or ())

    # Stub only accelerator-only FSDP wrapping; registration and DTensor autograd stay real.
    with patch("xtuner.v1.model.base.fully_shard", capture_fsdp):
        for _ in range(2):
            previous = dict(model.named_parameters())
            model._fully_shard(
                mesh=mesh["default.fsdp"]
                if family == "moe" and "default.fsdp" in (mesh.mesh_dim_names or ())
                else mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
                reshard_after_forward=True,
                offload_policy=None,
                module=model.layers["0"],
            )
            if _ == 1:
                assert all(param is previous[name] for name, param in model.named_parameters())
    return model, {name: param for name, param in model.named_parameters() if param in ignored}


def _check_rank(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        dense_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("default.fsdp",))
        moe_meshes = {
            ep: init_device_mesh("cpu", (4 // ep, ep), mesh_dim_names=("default.fsdp", "default.ep")) for ep in (1, 2)
        }
        for family, ep in (("dense", 1), ("moe", 1), ("moe", 2)):
            for kind in ("gdn", "kda"):
                mesh = dense_mesh if family == "dense" else moe_meshes[ep]
                model, ignored = _model(family, kind, mesh, ep)
                required_ignored = {"layers.0.self_attn.A_log"}
                if kind == "gdn":
                    required_ignored.add("layers.0.self_attn.norm.weight")
                assert required_ignored <= ignored.keys()
                assert all(param.requires_grad for param in ignored.values())
                assert all(param.device_mesh == mesh for param in ignored.values())
                assert all(all(isinstance(p, Replicate) for p in param.placements) for param in ignored.values())
                # Model an ordinary FSDP shard: Dense must skip it; MoE must retain EP-only averaging.
                placements = [Shard(0)] if family == "dense" else [Shard(0), Replicate()]
                model.control = nn.Parameter(distribute_tensor(torch.ones(4), mesh, placements))
                materialize = dtensor_to_local if kind == "gdn" else dtensor_full_tensor
                loss = sum(materialize(param).sum() * (rank + 1) for param in ignored.values())
                control_coefficient = rank + 1 if family == "dense" else rank % ep + 1
                loss = loss + model.control.to_local().sum() * control_coefficient
                loss.backward()
                calls = []
                original_all_reduce = dist.all_reduce

                def counted_all_reduce(tensor, *args, **kwargs):
                    calls.append((tensor.data_ptr(), dist.get_world_size(kwargs["group"])))
                    return original_all_reduce(tensor, *args, **kwargs)

                with patch("torch.distributed.all_reduce", counted_all_reduce):
                    model.scale_and_reduce_grad()
                for name, param in ignored.items():
                    grad = param.grad.to_local()
                    torch.testing.assert_close(grad, torch.full_like(grad, 2.5))
                    assert calls.count((grad.data_ptr(), 4)) == 1, (family, kind, ep, name, calls)
                    before = param.to_local().detach().clone()
                    with torch.no_grad():
                        param.to_local().add_(grad, alpha=-0.1)
                    torch.testing.assert_close(param.to_local(), before - 0.25)
                control_grad = model.control.grad.to_local()
                expected = control_coefficient if family == "dense" else (ep + 1) / 2
                torch.testing.assert_close(control_grad, torch.full_like(control_grad, expected))
                assert calls.count((control_grad.data_ptr(), ep)) == (family == "moe")
                if family == "dense":
                    assert not any(ptr == control_grad.data_ptr() for ptr, _ in calls)
                if rank == 0:
                    print(f"PASS: {family} {kind} EP={ep}: global mean, one reduction, shard control", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ignored_fp32_gradients_sync_across_dense_and_moe_ranks(tmp_path):
    mp.start_processes(
        _check_rank, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, start_method="fork", join=True
    )


def _check_hsdp_rank(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        prototype = Qwen3_5_VLTextMoE.__new__(Qwen3_5_VLTextMoE)
        nn.Module.__init__(prototype)
        prototype.config = SimpleNamespace(expert_tp_size=1, mesh_prefix="default")
        with patch("xtuner.v1.model.moe.moe.DEVICE", "cpu"):
            prototype._init_device_mesh(FSDPConfig(hsdp_sharding_size=2))
        assert prototype.world_mesh is prototype.hsdp_mesh
        assert prototype.world_mesh.mesh_dim_names == ("default.hsdp_replicate", "default.hsdp_shard")
        for kind in ("gdn", "kda"):
            model, ignored = _model("moe", kind, prototype.world_mesh, 1)
            materialize = dtensor_to_local if kind == "gdn" else dtensor_full_tensor
            sum(materialize(parameter).sum() * (rank + 1) for parameter in ignored.values()).backward()
            calls = []
            original_all_reduce = dist.all_reduce

            def counted_all_reduce(tensor, *args, **kwargs):
                calls.append(dist.get_world_size(kwargs["group"]))
                return original_all_reduce(tensor, *args, **kwargs)

            with patch("torch.distributed.all_reduce", counted_all_reduce):
                model.scale_and_reduce_grad()
            for parameter in ignored.values():
                assert parameter.device_mesh is prototype.world_mesh
                torch.testing.assert_close(parameter.grad.to_local(), torch.full((4,), 2.5))
            assert calls == [4] * len(ignored)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestHSDPIgnoredFP32:
    def test_named_world_mesh_and_global_gradient_mean(self, tmp_path) -> None:
        mp.start_processes(
            _check_hsdp_rank,
            args=(f"file://{tmp_path / 'hsdp-rendezvous'}",),
            nprocs=4,
            start_method="fork",
            join=True,
        )
