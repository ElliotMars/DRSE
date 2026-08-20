from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import exp.exp_multi_expert as multi_expert
from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.online_diagnostics import StreamingTSBGradientDiagnostics


class _TinyExpertBase(nn.Module):
    expert_type = "base"

    def __init__(self, args, device) -> None:
        super().__init__()
        del device
        self.output_dim = args.pred_len * args.c_out
        self.representation = nn.Parameter(torch.zeros(320))

    def forward(self, x, x_mark, return_repr=False):
        del x_mark
        output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        representation = self.representation.to(x.device).expand(x.shape[0], -1)
        if return_repr:
            return output, representation
        return output

    def store_grad(self) -> None:
        pass


class _TinyFSNet(_TinyExpertBase):
    expert_type = "fsnet"


class _TinyFSNetTime(_TinyExpertBase):
    expert_type = "fsnet_time"


class _TinyRoutingFeatureEncoder(nn.Module):
    def __init__(self, args, hidden_dim, device) -> None:
        super().__init__()
        del args, device
        self.hidden_dim = hidden_dim

    def forward(self, x, x_mark):
        del x_mark
        sequence = torch.zeros(
            x.shape[0], x.shape[1], self.hidden_dim, device=x.device
        )
        return sequence, sequence[:, -1]


def _model_args(composition="mixed") -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=4,
        expert_composition=composition,
        top_k=4,
        c_out=2,
        pred_len=3,
        enc_in=2,
        dropout=0.0,
        router_temperature=1.0,
        router_granularity="channel",
        capability_sketch_dim=16,
        capability_sketch_seed=31,
    )


def _patch_tiny_model(monkeypatch) -> None:
    monkeypatch.setattr(multi_expert, "ExpertNet", _TinyFSNet)
    monkeypatch.setattr(multi_expert, "FSNetTimeExpertNet", _TinyFSNetTime)
    monkeypatch.setattr(
        multi_expert, "RoutingFeatureEncoder", _TinyRoutingFeatureEncoder
    )


@pytest.mark.parametrize(
    "composition, expected_fsnet, expected_time, expected_types",
    [
        ("fsnet", 4, 0, [_TinyFSNet] * 4),
        ("fsnet_time", 0, 4, [_TinyFSNetTime] * 4),
        ("mixed", 2, 2, [_TinyFSNet] * 2 + [_TinyFSNetTime] * 2),
    ],
)
def test_expert_composition_preserves_model_shapes(
    monkeypatch, composition, expected_fsnet, expected_time, expected_types
) -> None:
    _patch_tiny_model(monkeypatch)
    model = multi_expert.net(_model_args(composition), torch.device("cpu"))

    assert model.num_fsnet_experts == expected_fsnet
    assert model.num_time_experts == expected_time
    assert [type(expert) for expert in model.experts] == expected_types

    x = torch.randn(2, 5, 2)
    x_mark = torch.zeros(2, 5, 7)
    outputs, representations = model.forward_experts(x, x_mark, return_repr=True)
    gates = model._compute_gates(x, x_mark)
    sketches = model.compute_capability_sketch(representations)

    assert outputs.shape == (2, 4, 6)
    assert representations.shape == (2, 4, 320)
    assert gates.shape == (2, 2, 4)
    assert sketches.shape == (2, 4, 16)
    assert model.capability_projection.shape == (4, 320, 16)

    model.set_router_mode(True)
    assert model(x, x_mark).shape == (2, 6)


def test_omitted_composition_is_strictly_checkpoint_compatible_with_mixed(
    monkeypatch,
) -> None:
    _patch_tiny_model(monkeypatch)
    legacy_args = _model_args()
    del legacy_args.expert_composition
    legacy = multi_expert.net(legacy_args, torch.device("cpu"))
    explicit = multi_expert.net(_model_args("mixed"), torch.device("cpu"))

    assert legacy.num_fsnet_experts == explicit.num_fsnet_experts == 2
    assert legacy.num_time_experts == explicit.num_time_experts == 2
    assert [type(expert) for expert in legacy.experts] == [
        type(expert) for expert in explicit.experts
    ]
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.equal(
        legacy.capability_projection, explicit.capability_projection
    )


@pytest.mark.parametrize(
    "smoothing, filtering, expected, expected_conflict_rate",
    [
        (False, False, [-2.0, 2.0], 0.0),
        (True, False, [-1.25, 1.5], 0.0),
        (False, True, [0.0, 2.0], 1.0),
        (True, True, [0.0, 1.5], 1.0),
    ],
)
def test_tsb_smoothing_and_conflict_filter_are_independent(
    smoothing, filtering, expected, expected_conflict_rate
) -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    parameter = nn.Parameter(torch.zeros(2))
    experiment.expert_params = [parameter]
    experiment.tsb_eps = 1e-8
    experiment.tsb_smoothing_enabled = smoothing
    experiment.tsb_conflict_filter_enabled = filtering

    diagnostics = experiment._apply_tsb_gradient_filter(
        current=[torch.tensor([-2.0, 2.0])],
        reference=[torch.tensor([1.0, 0.0])],
        alpha=0.25,
    )

    assert torch.allclose(
        parameter.grad, torch.tensor(expected), atol=1e-6
    )
    assert diagnostics["tsb_smoothing_enabled"] is smoothing
    assert diagnostics["tsb_conflict_filter_enabled"] is filtering
    assert diagnostics["tsb_conflict_rate"] == expected_conflict_rate


def test_tsb_reference_is_not_computed_when_both_operations_are_disabled() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.tsb_reference_enabled = False
    experiment.online_buffer = deque([object()])

    assert experiment._compute_tsb_reference_gradient() is None


@pytest.mark.parametrize(
    "reference, expected_cosine, expected_conflict, expected_gradient",
    [
        ([1.0, -2.0], 1.0, False, [1.0, -2.0]),
        ([-1.0, 2.0], -1.0, True, [0.0, 0.0]),
    ],
)
def test_tsb_whole_gradient_cosine_and_projection_are_deterministic(
    reference, expected_cosine, expected_conflict, expected_gradient
) -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    parameter = nn.Parameter(torch.zeros(2))
    experiment.expert_params = [parameter]
    experiment.tsb_eps = 1e-8
    experiment.tsb_smoothing_enabled = False
    experiment.tsb_conflict_filter_enabled = True

    diagnostics = experiment._apply_tsb_gradient_filter(
        current=[torch.tensor([1.0, -2.0])],
        reference=[torch.tensor(reference)],
        alpha=0.5,
    )

    assert diagnostics["tsb_grad_cosine"] == pytest.approx(expected_cosine)
    assert diagnostics["tsb_gradient_conflict"] is expected_conflict
    assert torch.allclose(
        parameter.grad, torch.tensor(expected_gradient), atol=1e-6
    )
    expected_ratio = 1.0 if expected_conflict else 0.0
    assert diagnostics["tsb_modification_ratio"] == pytest.approx(
        expected_ratio, abs=1e-6
    )
    if expected_conflict:
        assert diagnostics["tsb_projection_removal_ratio"] == pytest.approx(
            1.0, abs=1e-6
        )
    else:
        assert diagnostics["tsb_projection_removal_ratio"] is None


def test_tsb_without_reference_is_an_exact_noop_with_explicit_diagnostics() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    parameter = nn.Parameter(torch.zeros(2))
    experiment.expert_params = [parameter]
    experiment.tsb_eps = 1e-8
    experiment.tsb_smoothing_enabled = True
    experiment.tsb_conflict_filter_enabled = True
    current = torch.tensor([3.0, 4.0])

    diagnostics = experiment._apply_tsb_gradient_filter(
        current=[current], reference=None, alpha=0.5
    )

    assert torch.equal(parameter.grad, current)
    assert diagnostics["tsb_reference_available"] is False
    assert diagnostics["tsb_grad_cosine"] is None
    assert diagnostics["tsb_gradient_conflict"] is None
    assert diagnostics["tsb_modification_ratio"] == 0.0
    assert diagnostics["tsb_projection_removal_ratio"] is None


def test_tsb_streaming_diagnostics_use_constant_scalar_state() -> None:
    diagnostics = StreamingTSBGradientDiagnostics()
    initial_keys = set(diagnostics.__dict__)

    for index in range(200):
        diagnostics.update(
            grad_cosine=-1.0 if index % 2 else 1.0,
            conflict=bool(index % 2),
            modification_ratio=0.5,
            projection_removal_ratio=1.0 if index % 2 else None,
        )

    metrics = diagnostics.metrics()
    assert set(diagnostics.__dict__) == initial_keys
    assert all(
        isinstance(value, (int, float))
        for value in diagnostics.__dict__.values()
    )
    assert metrics["mean_grad_cosine"] == 0.0
    assert metrics["conflict_rate"] == 0.5
    assert metrics["mean_tsb_modification_ratio"] == 0.5
    assert metrics["mean_tsb_projection_removal_ratio"] == 1.0


def _controller(
    mode, *, smoothing, previous_mse=4.0, ema=1.0
) -> Exp_TS2VecSupervised:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.adaptive_controller = mode
    experiment.base_learning_rate_expert = 0.02
    experiment.base_learning_rate_router = 0.01
    experiment.tsb_alpha = 0.5
    experiment.tsb_eps = 1e-8
    experiment.tsb_smoothing_enabled = smoothing
    experiment.prev_online_mse = previous_mse
    experiment.online_mse_ema = ema
    return experiment


def test_dynamic_controller_adjusts_lr_without_tsb() -> None:
    experiment = _controller("dynamic", smoothing=False)

    expert_lr, router_lr, alpha = experiment._adaptive_online_hparams()

    assert expert_lr == pytest.approx(0.01)
    assert router_lr == pytest.approx(0.005)
    assert alpha == 0.5
    assert expert_lr <= experiment.base_learning_rate_expert
    assert router_lr <= experiment.base_learning_rate_router


def test_fixed_controller_keeps_base_lr_with_tsb() -> None:
    experiment = _controller("fixed", smoothing=True)

    expert_lr, router_lr, alpha = experiment._adaptive_online_hparams()

    assert expert_lr == experiment.base_learning_rate_expert
    assert router_lr == experiment.base_learning_rate_router
    assert alpha == experiment.tsb_alpha


@pytest.mark.parametrize("previous_mse", [0.25, 1.0, 4.0, 100.0])
def test_dynamic_controller_never_exceeds_base_lr(previous_mse) -> None:
    experiment = _controller(
        "dynamic", smoothing=True, previous_mse=previous_mse, ema=1.0
    )

    expert_lr, router_lr, _ = experiment._adaptive_online_hparams()

    assert 0.0 < expert_lr <= experiment.base_learning_rate_expert
    assert 0.0 < router_lr <= experiment.base_learning_rate_router


def test_error_spike_reduces_dynamic_lr() -> None:
    stable = _controller(
        "dynamic", smoothing=False, previous_mse=1.0, ema=1.0
    )
    spike = _controller(
        "dynamic", smoothing=False, previous_mse=9.0, ema=1.0
    )

    stable_expert_lr, stable_router_lr, _ = stable._adaptive_online_hparams()
    spike_expert_lr, spike_router_lr, _ = spike._adaptive_online_hparams()

    assert spike_expert_lr < stable_expert_lr
    assert spike_router_lr < stable_router_lr


def test_structural_ablation_cli_defaults_and_flags(monkeypatch) -> None:
    import sys

    from main import parse_args

    monkeypatch.setattr(sys, "argv", ["main.py"])
    defaults = parse_args()
    assert defaults.expert_composition == "mixed"
    assert defaults.disable_tsb_smoothing is False
    assert defaults.disable_tsb_conflict_filter is False
    assert defaults.adaptive_controller == "dynamic"
    assert defaults.pretrain_mode == "retrain"
    assert defaults.dynamic_comparator is False
    assert defaults.dynamic_comparator_max_switches == 1
    assert defaults.dynamic_comparator_max_points == 64

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--expert_composition",
            "fsnet_time",
            "--disable_tsb_smoothing",
            "--disable_tsb_conflict_filter",
            "--pretrain_mode",
            "none",
            "--adaptive_controller",
            "fixed",
            "--dynamic_comparator",
            "--dynamic_comparator_max_switches",
            "2",
            "--dynamic_comparator_max_points",
            "32",
        ],
    )
    configured = parse_args()
    assert configured.expert_composition == "fsnet_time"
    assert configured.disable_tsb_smoothing is True
    assert configured.disable_tsb_conflict_filter is True
    assert configured.adaptive_controller == "fixed"
    assert configured.dynamic_comparator is True
    assert configured.dynamic_comparator_max_switches == 2
    assert configured.dynamic_comparator_max_points == 32
    assert configured.pretrain_mode == "none"
