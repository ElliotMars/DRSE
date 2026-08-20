from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from models.ts2vec.fsnet_ import SamePadConv
from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem
from utils.online_diagnostics import OnlineDiagnosticsRecorder
from utils.recovery_learning import recovery_replay_objective


def test_prediction_time_sketch_is_stop_gradient() -> None:
    prediction = torch.zeros(2, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    current = torch.randn(2, 4, requires_grad=True)
    historical = torch.randn(2, 4, requires_grad=True)

    loss, _ = recovery_replay_objective(
        prediction,
        target,
        current,
        historical,
        responsibility=torch.ones(2),
        sketch_weight=1.0,
    )
    loss.backward()

    assert current.grad is not None
    assert current.grad.abs().sum() > 0
    assert historical.grad is None


def test_recovery_responsibility_directly_scales_loss() -> None:
    common = dict(
        prediction=torch.ones(1, 1),
        target=torch.zeros(1, 1),
        current_sketch=torch.zeros(1, 2),
        prediction_time_sketch=torch.zeros(1, 2),
        sketch_weight=0.0,
    )
    low, _ = recovery_replay_objective(
        **common, responsibility=torch.tensor([0.1])
    )
    high, _ = recovery_replay_objective(
        **common, responsibility=torch.tensor([0.9])
    )

    assert torch.allclose(high, 9.0 * low)


def test_recovery_objective_reduces_sketch_distance() -> None:
    torch.manual_seed(0)
    current = nn.Parameter(torch.randn(3, 5))
    historical = torch.randn(3, 5)
    optimizer = torch.optim.SGD([current], lr=0.5)
    before = (current.detach() - historical).pow(2).mean()

    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = recovery_replay_objective(
            prediction=torch.zeros(3, 1),
            target=torch.zeros(3, 1),
            current_sketch=current,
            prediction_time_sketch=historical,
            responsibility=torch.ones(3),
            sketch_weight=1.0,
        )
        loss.backward()
        optimizer.step()

    after = (current.detach() - historical).pow(2).mean()
    assert after < before


def test_read_only_fsnet_forward_preserves_state_but_allows_gradients() -> None:
    layer = SamePadConv(
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        device=torch.device("cpu"),
    )
    layer.eval()
    layer.trigger.fill_(True)
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.model = nn.Sequential(layer)
    before_q = layer.q_ema.clone()
    before_trigger = layer.trigger.clone()
    before_memory = layer.W.detach().clone()

    x = torch.randn(2, 2, 8)
    with experiment._fsnet_state_updates(False):
        loss = layer(x).pow(2).mean()
    loss.backward()

    assert torch.equal(layer.q_ema, before_q)
    assert torch.equal(layer.trigger, before_trigger)
    assert torch.equal(layer.W, before_memory)
    assert layer.conv.weight.grad is not None


def test_recovery_lifecycle_and_buffer_tensors_stay_on_cpu() -> None:
    manager = ExpertMemoryManager(
        num_experts=1,
        stable_capacity=2,
        recovery_capacity=2,
        responsibility_threshold=0.2,
        alignment_threshold=0.8,
        duplicate_threshold=0.99,
        failure_penalty=0.5,
        max_recovery_attempts=1,
        storage_dtype="fp16",
        promote_alignment_threshold=0.9,
        promote_loss_threshold=0.2,
    )
    item = VersionedMemoryItem(
        sample_id=1,
        origin=1,
        expert_id=0,
        x=torch.randn(1, 3, 2),
        x_mark=torch.randn(1, 3, 7),
        target=torch.randn(1, 2, 2),
        prediction_capability_sketch=torch.randn(4),
        normalized_sketch=torch.randn(4),
        sample_responsibility=0.9,
        last_alignment=0.2,
        stable_credit=0.18,
        recovery_credit=0.72,
        timestamp=1,
    )

    assert manager.add_candidate(item) == "recovery"
    sampled = manager.sample_recovery(0, 2)
    assert len(sampled) == 1
    for tensor in (
        sampled[0].x,
        sampled[0].x_mark,
        sampled[0].target,
        sampled[0].prediction_capability_sketch,
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float16

    assert manager.update_recovery_result(0, 1, 0.95, 0.1) == "promoted"
    assert manager.stable_buffers[0].contains(1)


def test_sampled_recovery_weight_is_confidence_aware_credit() -> None:
    manager = ExpertMemoryManager(
        num_experts=1, stable_capacity=1, recovery_capacity=1,
        responsibility_threshold=0.0, alignment_threshold=0.8,
        duplicate_threshold=0.99, failure_penalty=0.5,
        max_recovery_attempts=2, storage_dtype="fp16",
    )
    confidence, responsibility, alignment = 0.25, 0.8, 0.2
    item = VersionedMemoryItem(
        sample_id=9, origin=9, expert_id=0,
        x=torch.zeros(1, 2, 1), x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 1, 1),
        prediction_capability_sketch=torch.tensor([1.0, 0.0]),
        normalized_sketch=torch.tensor([1.0, 0.0]),
        sample_responsibility=responsibility,
        sample_confidence=confidence, last_alignment=alignment,
        stable_credit=confidence * responsibility * alignment,
        recovery_credit=confidence * responsibility * (1.0 - alignment),
        timestamp=9,
    )
    assert manager.add_candidate(item) == "recovery"
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.recovery_enabled = True
    experiment.recovery_batch_size = 1
    experiment.recovery_loss_weight = 1.0
    experiment.memory_manager = manager
    experiment.model = SimpleNamespace(num_experts=1)
    experiment.device = torch.device("cpu")

    batches = experiment._sample_recovery_batches()

    assert len(batches) == 1
    expected = confidence * responsibility * (1.0 - alignment)
    assert torch.allclose(batches[0]["responsibility"], torch.tensor([expected]))


class _ReplayDiagnosticExpert(nn.Module):
    def forward(self, x, x_mark, return_repr=False):
        del x_mark
        prediction = torch.zeros(x.shape[0], 1)
        representation = torch.ones(x.shape[0], 2)
        return (prediction, representation) if return_repr else prediction


class _ReplayDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_ReplayDiagnosticExpert()])
        self.register_buffer(
            "capability_projection", torch.eye(2).unsqueeze(0)
        )


def test_attempt_exhaustion_does_not_count_as_recovery_eviction() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.device = torch.device("cpu")
    experiment.model = _ReplayDiagnosticModel()
    experiment.min_credit_eps = 1e-8
    experiment.memory_manager = SimpleNamespace(
        update_recovery_result=lambda *args: "dropped"
    )
    experiment.diagnostics = OnlineDiagnosticsRecorder(1, interval=1)
    normalized = torch.nn.functional.normalize(
        torch.ones(1, 2), dim=-1
    )
    batch = {
        "expert_id": 0,
        "items": [SimpleNamespace(sample_id=1)],
        "x": torch.zeros(1, 2, 1),
        "x_mark": torch.zeros(1, 2, 7),
        "target": torch.zeros(1, 1),
        "historical_sketch": normalized,
    }

    experiment._finalize_recovery_replay([batch])

    counters = experiment.diagnostics.counters
    assert counters["recovery_evicted"] == 0
    assert counters["recovery_attempt_exhausted"] == 1
    assert counters["recovery_failed"] == 1
    assert counters["drop_count"] == 1
