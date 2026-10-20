"""Standard-engine accumulation must synchronize replicated gradients before clipping."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, distribute_tensor

from xtuner.v1.config import FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.base import BaseModel


class _Output:
    model_fields = ("loss",)

    def __init__(self, loss):
        self.loss = loss

    def free_nongrad_feature(self):
        pass


class _ReplicatedModel(nn.Module):
    scale_and_reduce_grad = BaseModel.scale_and_reduce_grad
    cal_grad_norm = BaseModel.cal_grad_norm
    _clean_param_name = BaseModel._clean_param_name

    def __init__(self, mesh, accumulation):
        super().__init__()
        placements = [Replicate() for _ in range(mesh.ndim)]
        self.weight = nn.Parameter(distribute_tensor(torch.ones(4), mesh, placements))
        self.unused = nn.Parameter(distribute_tensor(torch.ones(4), mesh, placements))
        self._fsdp_ignored_param_names = {"weight", "unused"}
        self.accumulation = accumulation

    def trainable_parameters(self):
        return list(self.named_parameters())

    def pre_micro_batch_forward(self, _batches):
        return {}

    def post_micro_batch_forward(self, _outputs):
        return {}

    def forward(self, *, seq_ctx, loss_ctx):
        local_loss = (self.weight.to_local() * seq_ctx).sum() / (dist.get_world_size() * self.accumulation)
        return _Output(dist_nn.all_reduce(local_loss))


def _check_engine(rank, init_method, mesh_shape, accumulation):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=tuple(f"dim{i}" for i in range(len(mesh_shape))))
        model = _ReplicatedModel(mesh, accumulation)
        engine = object.__new__(TrainEngine)
        engine.model = model
        engine.fsdp_cfg = FSDPConfig()
        engine.optim_cfg = SimpleNamespace(max_grad_norm=1.0, fused_grad_clip=False, skip_grad_norm_threshold=None)
        engine.optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        engine.intra_layer_micro_batch = 1
        engine._count = 0
        engine._maybe_precompute_float8_dynamic_scale_for_fsdp = lambda: None
        batches = [
            {"seq_ctx": torch.full((4,), float((rank + 1) * (i + 1))), "loss_ctx": None} for i in range(accumulation)
        ]
        expected_grad = 1.5 * (accumulation + 1) / 2
        with patch("xtuner.v1.engine.train_engine.DEVICE", torch.device("cpu")):
            info = engine.train_step(batches)
            assert info["total_loss"] == 4 * expected_grad
            norm = engine.clip_grad_norm()
            torch.testing.assert_close(norm, torch.tensor(2 * expected_grad), rtol=0, atol=0)
            expected_clipped = expected_grad / (2 * expected_grad + 1e-6)
            torch.testing.assert_close(model.weight.grad.to_local(), torch.full((4,), expected_clipped))
            assert model.unused.grad is None
            engine.step_optimizer(norm)
        torch.testing.assert_close(model.weight.to_local(), torch.full((4,), 1 - 0.25 * expected_clipped))
        torch.testing.assert_close(model.unused.to_local(), torch.ones(4), rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mesh_shape", [(2,), (1, 2)])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_standard_engine_synchronizes_ignored_gradients(tmp_path, mesh_shape, accumulation):
    mp.spawn(_check_engine, args=((tmp_path / "rendezvous").as_uri(), mesh_shape, accumulation), nprocs=2, join=True)


class _TokenModel(_ReplicatedModel):
    def build_loss_ctx_batch(self, data_batch, sp_mesh=None):
        labels = [data["shifted_labels"] for data in data_batch]
        if sp_mesh is not None:
            labels = [label.chunk(sp_mesh.size(), dim=1)[sp_mesh.get_local_rank()] for label in labels]
        denominator = sum(label.ne(-100).sum() for label in labels).float()
        dist.all_reduce(denominator)
        return [
            {"lm": SimpleNamespace(loss_kwargs=SimpleNamespace(shifted_labels=label), denominator=denominator)}
            for label in labels
        ]

    def forward(self, *, seq_ctx, loss_ctx):
        lm = loss_ctx["lm"]
        valid = lm.loss_kwargs.shifted_labels.ne(-100)
        loss = (seq_ctx.input_ids.float() * valid * self.weight.to_local()[0]).sum() / lm.denominator
        return _Output(dist_nn.all_reduce(loss))


class _Sequence:
    def __init__(self, tokens):
        self.input_ids = tokens

    @classmethod
    def from_input_ids(cls, documents, device=None):
        return cls(torch.cat(documents, dim=1))

    def split(self, mesh):
        return _Sequence(self.input_ids.chunk(mesh.size(), dim=1)[mesh.get_local_rank()])


class _Writer:
    def __init__(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    add_scalar = flush = close = add_text


def _check_alignment_script(rank, init_method, directory):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        path = Path(__file__).resolve().parents[2] / "examples/v1/scripts/megatron_alignment/align_conceptlm_qwen35_v2_sp_training.py"
        spec = importlib.util.spec_from_file_location("sp_alignment", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        tensorboard = ModuleType("torch.utils.tensorboard")
        tensorboard.SummaryWriter = _Writer
        args = SimpleNamespace(
            seed=0,
            steps=2,
            lengths=[5, 3],
            max_grad_norm=1.0,
            output=Path(directory) / "alignment.json",
            tensorboard_dir=Path(directory) / "events",
        )
        batches = [
            [torch.arange(1 + i, 1 + i + length).view(1, -1) for i in range(2) for length in args.lengths]
            for _ in range(args.steps)
        ]
        tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return tensor(*args, **kwargs)

        reference = initial = None
        for sp in (1, 2):
            mesh = init_device_mesh("cpu", (2 // sp, sp), mesh_dim_names=("dp", "sp"))
            model = _TokenModel(mesh, sp)
            engine = object.__new__(TrainEngine)
            engine.model = model
            engine.fsdp_cfg = FSDPConfig()
            engine.optim_cfg = SimpleNamespace(max_grad_norm=1.0, fused_grad_clip=False, skip_grad_norm_threshold=None)
            engine.optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
            engine.intra_layer_micro_batch = 1
            engine._count = 0
            engine._maybe_precompute_float8_dynamic_scale_for_fsdp = lambda: None
            with (
                patch.object(script, "_load_engine", return_value=engine),
                patch("xtuner.v1.data_proto.SequenceContext", _Sequence),
                patch("xtuner.v1.engine.train_engine.DEVICE", torch.device("cpu")),
                patch.dict(sys.modules, {"torch.utils.tensorboard": tensorboard}),
                patch.object(torch, "tensor", side_effect=cpu_tensor),
                patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self),
                patch.object(torch.cuda, "reset_peak_memory_stats"),
                patch.object(torch.cuda, "synchronize"),
                patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                patch.object(engine, "train_step", wraps=engine.train_step) as train,
                patch.object(engine, "clip_grad_norm", wraps=engine.clip_grad_norm) as clip,
                patch.object(engine, "step_optimizer", wraps=engine.step_optimizer) as step,
            ):
                metrics, weights = script._run_training(args, mesh["sp"], rank, 2, batches, initial)
                assert train.call_count == clip.call_count == step.call_count == args.steps
                assert all(len(call.args[0]) == sp for call in train.call_args_list)
            if reference is None:
                reference, initial = metrics, weights
            else:
                for actual, expected in zip(metrics, reference, strict=True):
                    for name in ("loss", "grad_norm"):
                        torch.testing.assert_close(tensor(actual[name]), tensor(expected[name]), rtol=1e-6, atol=1e-6)
                    assert actual["valid_targets"] == expected["valid_targets"] == 12
                    assert actual["batch_sha256"] == expected["batch_sha256"]
    finally:
        dist.destroy_process_group()


def test_alignment_script_uses_standard_engine_with_sp_accumulation(tmp_path):
    mp.spawn(_check_alignment_script, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
