import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.comparator_evaluation import StaticComparatorAccumulator
from utils.expert_memory import ExpertMemoryManager
from utils.evaluation_diagnostics import SpecializationDiagnosticsAggregator
from utils.online_diagnostics import OnlineDiagnosticsRecorder
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)
from utils.subspace_protection import RegressorSubspaceProtector


def _credit_record(origin: int, js: float, alignment: list[float]) -> dict:
    experts = len(alignment)
    responsibility = [1.0 / experts] * experts
    prediction_mse = [float(origin + index) for index in range(experts)]
    mixture_mse = float(origin) + 0.5
    best_mse = min(prediction_mse)
    return {
        "origin": origin,
        "horizon_delay": 2,
        "pred_len": 2,
        "prediction_responsibility": responsibility,
        "current_responsibility": responsibility,
        "prediction_expert_mse": prediction_mse,
        "current_expert_mse": prediction_mse,
        "js_divergence": js,
        "ranking_reversal": bool(origin % 2),
        "capability_alignment": alignment,
        "capability_l2_distance": [1.0 - value for value in alignment],
        "sample_confidence": 0.5,
        "expert_update_delta": origin,
        "global_expert_update_count": origin + 2,
        "prediction_mixture_mse": mixture_mse,
        "best_prediction_expert_mse": best_mse,
        "router_gap": mixture_mse - best_mse,
        "oracle_hard_expert_id": int(np.argmin(prediction_mse)),
        "oracle_hard_mse": best_mse,
        "oracle_top2_mse": best_mse,
        "oracle_top2_pair": [0, 1] if experts > 1 else [0, 0],
        "oracle_top2_alpha": 1.0,
        "router_mse": mixture_mse,
        "gap_to_hard_oracle": mixture_mse - best_mse,
        "gap_to_top2_oracle": mixture_mse - best_mse,
    }


def test_credit_diagnostics_are_bounded_and_persisted(tmp_path) -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.diagnostics = OnlineDiagnosticsRecorder(2, interval=1)
    experiment.diagnostics.update(online_mse=1.0)
    experiment.diagnostics.maybe_record(0)
    experiment.credit_diagnostics = deque(maxlen=2)
    for origin, js in enumerate((0.1, 0.2, 0.4)):
        experiment.credit_diagnostics.append(
            _credit_record(origin, js, [0.8, 0.6])
        )
    experiment.credit_diagnostic_total_count = 3
    experiment.specialization_diagnostics = SpecializationDiagnosticsAggregator(
        pred_len=2, c_out=1, num_experts=2
    )
    assert len(experiment.credit_diagnostics) == 2

    _, summary_path = experiment.save_online_diagnostics(str(tmp_path))

    credit_path = tmp_path / "credit_diagnostics.npz"
    assert credit_path.exists()
    arrays = np.load(credit_path)
    assert arrays["origin"].tolist() == [1, 2]
    assert arrays["prediction_responsibility"].shape == (2, 2)
    assert arrays["current_responsibility"].shape == (2, 2)
    assert arrays["prediction_expert_mse"].shape == (2, 2)
    assert arrays["current_expert_mse"].shape == (2, 2)
    assert arrays["capability_alignment_existing"].shape == (2, 2)
    assert arrays["capability_alignment"].shape == (2, 2)
    assert np.array_equal(
        arrays["capability_alignment_existing"],
        arrays["capability_alignment"],
    )
    assert arrays["capability_l2_distance"].shape == (2, 2)
    assert arrays["expert_update_delta"].tolist() == [1, 2]
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)["credit_diagnostics"]
    assert summary["total_records"] == 3
    assert summary["retained_records"] == 2
    assert summary["estimated_overwritten"] == 1
    assert abs(summary["mean_js_divergence"] - 0.3) < 1e-12
    assert abs(summary["ranking_reversal_rate"] - 0.5) < 1e-12
    assert abs(summary["mean_alignment"] - 0.7) < 1e-12
    assert abs(summary["min_alignment"] - 0.6) < 1e-12


def test_progressive_reset_clears_stream_state_without_changing_parameters() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    expert_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    router_parameter = torch.nn.Parameter(torch.tensor([2.0]))
    expert_parameter.grad = torch.ones_like(expert_parameter)
    router_parameter.grad = torch.ones_like(router_parameter)
    experiment.opt_expert = torch.optim.Adam([expert_parameter], lr=0.1)
    experiment.opt_router = torch.optim.Adam([router_parameter], lr=0.1)
    parameter_snapshots = (
        expert_parameter.detach().clone(),
        router_parameter.detach().clone(),
    )
    experiment.online_buffer = deque([object()], maxlen=2)
    experiment.online_step = 9
    experiment.prev_online_mse = 3.0
    experiment.online_mse_ema = 2.0
    experiment.fallback_count = 4
    experiment.fallback_channel_count = 5
    experiment.progressive_origin = 7
    experiment.credit_diagnostics = deque([_credit_record(0, 0.1, [0.9])], maxlen=2)
    experiment.credit_diagnostic_total_count = 4
    experiment.expert_update_count = 3
    experiment.completed_record_count = 2
    experiment.last_expert_update_diagnostics = {"gradient": 1.0}
    experiment.routing_correction = OnlineRoutingCorrection(
        pred_len=2,
        c_out=1,
        num_experts=1,
        device=torch.device("cpu"),
        correction_lr=0.1,
        correction_decay=0.1,
        correction_grad_clip=1.0,
        correction_logit_clip=1.0,
    )
    experiment.routing_correction.z.fill_(0.7)
    experiment.memory_manager = ExpertMemoryManager(
        num_experts=1,
        stable_capacity=1,
        recovery_capacity=1,
        responsibility_threshold=0.0,
        alignment_threshold=0.5,
        duplicate_threshold=0.9,
        failure_penalty=0.5,
        max_recovery_attempts=1,
    )
    experiment.subspace_protector = RegressorSubspaceProtector(
        num_experts=1,
        feature_dim=2,
        rank=1,
        max_rank=1,
        energy_threshold=0.9,
        min_samples=1,
    )
    experiment.subspace_protector.states[0].basis = torch.ones(2, 1)
    experiment.subspace_protector.states[0].effective_rank = 1
    experiment.diagnostics = OnlineDiagnosticsRecorder(1, interval=1)
    experiment.diagnostics.update(online_mse=2.0)
    experiment.diagnostics.maybe_record(1)
    manager = ProgressiveFeedbackManager(pred_len=2, c_out=1)
    manager._last_release_origin = 5

    experiment._reset_progressive_online_state(manager)

    assert len(experiment.online_buffer) == 0
    assert experiment.online_step == 0
    assert experiment.prev_online_mse is None
    assert experiment.online_mse_ema is None
    assert experiment.fallback_count == 0
    assert experiment.fallback_channel_count == 0
    assert manager._last_release_origin is None
    assert torch.count_nonzero(experiment.routing_correction.z) == 0
    assert experiment.subspace_protector.states[0].effective_rank == 0
    assert experiment.progressive_origin == 0
    assert len(experiment.credit_diagnostics) == 0
    assert experiment.credit_diagnostic_total_count == 0
    assert experiment.expert_update_count == 0
    assert experiment.completed_record_count == 0
    assert experiment.last_expert_update_diagnostics == {}
    assert expert_parameter.grad is None and router_parameter.grad is None
    assert torch.equal(expert_parameter, parameter_snapshots[0])
    assert torch.equal(router_parameter, parameter_snapshots[1])


class _DiagnosticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.current_outputs = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        self.current_representations = torch.nn.Parameter(torch.eye(2))

    def forward_experts(self, x, x_mark, return_repr=False):
        del x_mark
        outputs = self.current_outputs.view(1, 2, 1).expand(x.shape[0], -1, -1)
        representations = self.current_representations.unsqueeze(0).expand(
            x.shape[0], -1, -1
        )
        return (outputs, representations) if return_repr else outputs

    def compute_capability_sketch(self, representations):
        return torch.nn.functional.normalize(representations, dim=-1)


def test_completed_record_diagnostic_uses_prediction_snapshot_and_version_delta(
    tmp_path,
) -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.device = torch.device("cpu")
    experiment.model = _DiagnosticModel()
    experiment.args = SimpleNamespace(pred_len=1, c_out=1)
    experiment.sample_credit_temperature = 1.0
    experiment.min_credit_eps = 1e-8
    experiment.expert_update_count = 12
    experiment.credit_diagnostics = deque(maxlen=4)
    experiment.credit_diagnostic_total_count = 0
    experiment.diagnostics = OnlineDiagnosticsRecorder(2, interval=1)
    experiment.specialization_diagnostics = SpecializationDiagnosticsAggregator(
        pred_len=1, c_out=1, num_experts=2
    )

    record = ProgressiveForecastRecord(
        origin=3,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        expert_predictions=torch.tensor([[[0.0, 2.0]]]),
        router_prior=torch.full((1, 1, 2), 0.5),
        router_weights=torch.full((1, 1, 2), 0.5),
        mixture_prediction=torch.tensor([[1.5]]),
        capability_sketch=torch.eye(2),
    )
    record.matured_targets[0] = torch.tensor([0.0])
    record.sample_responsibility.copy_(torch.tensor([0.9, 0.1]))
    record.sample_confidence = 0.7
    record.metadata["expert_update_count_at_prediction"] = 5

    experiment._evaluate_completed_record(record, current_origin=4)

    diagnostic = experiment.credit_diagnostics[-1]
    assert diagnostic["expert_update_delta"] == 7
    assert diagnostic["capability_alignment_existing"] == [1.0, 1.0]
    assert diagnostic["capability_alignment_existing"] == diagnostic[
        "capability_alignment"
    ]
    assert diagnostic["prediction_expert_mse"] == [0.0, 4.0]
    assert diagnostic["current_expert_mse"] == [1.0, 9.0]
    assert diagnostic["prediction_mixture_mse"] == 2.25
    assert diagnostic["best_prediction_expert_mse"] == 0.0
    assert diagnostic["router_gap"] == 2.25
    assert diagnostic["oracle_hard_expert_id"] == 0
    assert diagnostic["oracle_hard_mse"] == 0.0
    assert diagnostic["oracle_top2_mse"] == 0.0
    assert diagnostic["oracle_top2_pair"] == [0, 1]
    assert diagnostic["gap_to_hard_oracle"] == 2.25
    assert diagnostic["gap_to_top2_oracle"] == 2.25
    assert all(parameter.grad is None for parameter in experiment.model.parameters())

    experiment.save_online_diagnostics(str(tmp_path))
    arrays = np.load(tmp_path / "credit_diagnostics.npz")
    assert arrays["prediction_responsibility"].shape == (1, 2)
    assert arrays["current_responsibility"].shape == (1, 2)
    assert arrays["prediction_expert_mse"].shape == (1, 2)
    assert arrays["current_expert_mse"].shape == (1, 2)
    assert arrays["capability_alignment_existing"].shape == (1, 2)
    assert arrays["capability_alignment"].shape == (1, 2)
    assert arrays["capability_l2_distance"].shape == (1, 2)
    assert arrays["expert_update_delta"].tolist() == [7]
    assert arrays["router_gap"].tolist() == [2.25]
    assert arrays["oracle_hard_expert_id"].tolist() == [0]
    assert arrays["oracle_top2_pair"].shape == (1, 2)
    assert arrays["gap_to_top2_oracle"].tolist() == [2.25]


def test_empty_credit_diagnostics_keep_two_dimensional_expert_fields(tmp_path) -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.model = SimpleNamespace(num_experts=2)
    experiment.diagnostics = OnlineDiagnosticsRecorder(2, interval=1)
    experiment.credit_diagnostics = deque(maxlen=2)
    experiment.credit_diagnostic_total_count = 0
    experiment.specialization_diagnostics = SpecializationDiagnosticsAggregator(
        pred_len=2, c_out=1, num_experts=2
    )
    experiment.comparator_diagnostics = StaticComparatorAccumulator(2)

    experiment.save_online_diagnostics(str(tmp_path))

    arrays = np.load(tmp_path / "credit_diagnostics.npz")
    assert arrays["prediction_responsibility"].shape == (0, 2)
    assert arrays["current_expert_mse"].shape == (0, 2)
    assert arrays["capability_alignment_existing"].shape == (0, 2)
    assert arrays["capability_alignment"].shape == (0, 2)
    assert arrays["oracle_top2_pair"].shape == (0, 2)
    specialization = np.load(tmp_path / "specialization_diagnostics.npz")
    assert specialization["mean_router_weight_by_horizon"].shape == (2, 2)
    assert specialization["mean_router_weight_by_channel"].shape == (1, 2)
    comparator = np.load(tmp_path / "comparator_diagnostics.npz")
    assert comparator["num_origins"].item() == 0
    assert comparator["static_comparator_weights"].shape == (2,)
    with open(
        tmp_path / "comparator_diagnostics.json", "r", encoding="utf-8"
    ) as handle:
        comparator_json = json.load(handle)
    assert comparator_json["comparator_type"] == "static_fixed_convex_mixture"
