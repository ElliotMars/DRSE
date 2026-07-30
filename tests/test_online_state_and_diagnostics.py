import json
from collections import deque

import numpy as np
import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.expert_memory import ExpertMemoryManager
from utils.online_diagnostics import OnlineDiagnosticsRecorder
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import ProgressiveFeedbackManager
from utils.subspace_protection import RegressorSubspaceProtector


def _credit_record(origin: int, js: float, alignment: list[float]) -> dict:
    return {
        "origin": origin,
        "horizon_delay": 2,
        "js_divergence": js,
        "ranking_reversal": bool(origin % 2),
        "capability_alignment": alignment,
        "sample_confidence": 0.5,
        "expert_update_count": origin,
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
    assert len(experiment.credit_diagnostics) == 2

    _, summary_path = experiment.save_online_diagnostics(str(tmp_path))

    credit_path = tmp_path / "credit_diagnostics.npz"
    assert credit_path.exists()
    arrays = np.load(credit_path)
    assert arrays["origin"].tolist() == [1, 2]
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
