from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.expert_memory import (
    ExpertMemoryManager,
    VersionedMemoryItem,
    classify_capability_evolution,
)
from utils.online_diagnostics import OnlineDiagnosticsRecorder
from utils.progressive_feedback import ProgressiveForecastRecord


def _record() -> ProgressiveForecastRecord:
    record = ProgressiveForecastRecord(
        origin=7,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        expert_predictions=torch.zeros(1, 1, 1),
        router_prior=torch.ones(1, 1, 1),
        router_weights=torch.ones(1, 1, 1),
        mixture_prediction=torch.zeros(1, 1),
        capability_sketch=torch.tensor([[1.0, 0.0]]),
    )
    record.matured_targets[0] = torch.tensor([0.0])
    record.sample_responsibility.copy_(torch.tensor([1.0]))
    record.sample_confidence = 1.0
    return record


def _builder_experiment(
    *,
    directional: bool = True,
    version: bool = True,
    recovery: bool = True,
    margin: float = 0.0,
) -> Exp_TS2VecSupervised:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.credit_top_k = 1
    experiment.responsibility_threshold = 0.0
    experiment.alignment_threshold = 0.8
    experiment.min_credit_eps = 1e-8
    experiment.recovery_degradation_margin = margin
    experiment.directional_recovery_enabled = directional and version and recovery
    experiment.version_awareness_enabled = version
    experiment.recovery_enabled = recovery
    experiment.diagnostics = OnlineDiagnosticsRecorder(1, interval=1)
    return experiment


def _manager(
    *,
    directional: bool = True,
    version: bool = True,
    stable_capacity: int = 2,
    margin: float = 0.0,
) -> ExpertMemoryManager:
    return ExpertMemoryManager(
        num_experts=1,
        stable_capacity=stable_capacity,
        recovery_capacity=2,
        responsibility_threshold=0.0,
        alignment_threshold=0.8,
        duplicate_threshold=1.1,
        failure_penalty=0.5,
        max_recovery_attempts=3,
        storage_dtype="fp32",
        promote_alignment_threshold=0.9,
        promote_loss_threshold=0.2,
        directional_recovery_enabled=directional,
        version_awareness_enabled=version,
        recovery_degradation_margin=margin,
    )


def _memory_item(
    sample_id: int,
    *,
    alignment: float,
    prediction_loss: float = 1.0,
    category: str = "retained",
    recovery_eligible: bool = True,
) -> VersionedMemoryItem:
    sketch = torch.tensor([1.0, 0.0])
    return VersionedMemoryItem(
        sample_id=sample_id,
        origin=sample_id,
        expert_id=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 1, 1),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=1.0,
        last_alignment=alignment,
        stable_credit=alignment,
        recovery_credit=1.0 - alignment,
        timestamp=sample_id,
        recent_prediction_loss=prediction_loss,
        prediction_time_loss=prediction_loss,
        last_observed_alignment=alignment,
        recovery_eligible=recovery_eligible,
        capability_evolution=category,
    )


@pytest.mark.parametrize(
    ("alignment", "prediction", "current", "expected"),
    [
        (0.2, 1.0, 1.2, "harmful_drift"),
        (0.2, 1.0, 0.8, "beneficial_or_neutral_evolution"),
        (0.2, 1.0, 1.0, "beneficial_or_neutral_evolution"),
        (0.9, 1.0, 2.0, "retained"),
    ],
)
def test_capability_evolution_categories(
    alignment: float,
    prediction: float,
    current: float,
    expected: str,
) -> None:
    decision = classify_capability_evolution(
        alignment=alignment,
        prediction_loss=prediction,
        current_loss=current,
        alignment_threshold=0.8,
    )

    assert decision.category == expected
    assert decision.relative_degradation == pytest.approx(
        (current - prediction) / (prediction + 1e-8)
    )


def test_directional_candidate_admission_uses_alignment_and_loss_direction() -> None:
    experiment = _builder_experiment()
    record = _record()

    harmful = experiment._build_memory_candidates(
        record,
        alignment=torch.tensor([0.2]),
        prediction_loss=torch.tensor([1.0]),
        current_loss=torch.tensor([1.2]),
        timestamp=8,
    )
    beneficial = experiment._build_memory_candidates(
        record,
        alignment=torch.tensor([0.2]),
        prediction_loss=torch.tensor([1.0]),
        current_loss=torch.tensor([0.8]),
        timestamp=8,
    )
    equal = experiment._build_memory_candidates(
        record,
        alignment=torch.tensor([0.2]),
        prediction_loss=torch.tensor([1.0]),
        current_loss=torch.tensor([1.0]),
        timestamp=8,
    )
    retained = experiment._build_memory_candidates(
        record,
        alignment=torch.tensor([0.9]),
        prediction_loss=torch.tensor([1.0]),
        current_loss=torch.tensor([2.0]),
        timestamp=8,
    )

    assert len(harmful) == 1
    assert harmful[0].capability_evolution == "harmful_drift"
    assert harmful[0].recovery_credit == pytest.approx(0.8)
    manager = _manager()
    assert manager.add_candidate(harmful[0]) == "recovery"
    assert beneficial == []
    assert equal == []
    assert len(retained) == 1
    assert retained[0].capability_evolution == "retained"
    assert manager.add_candidate(retained[0]) == "stable"


def test_recovery_degradation_margin_changes_directional_gate() -> None:
    experiment = _builder_experiment(margin=0.1)
    record = _record()

    within_margin = experiment._build_memory_candidates(
        record,
        torch.tensor([0.2]),
        torch.tensor([1.0]),
        timestamp=8,
        current_loss=torch.tensor([1.05]),
    )
    beyond_margin = experiment._build_memory_candidates(
        record,
        torch.tensor([0.2]),
        torch.tensor([1.0]),
        timestamp=8,
        current_loss=torch.tensor([1.11]),
    )

    assert within_margin == []
    assert len(beyond_margin) == 1
    assert beyond_margin[0].capability_evolution == "harmful_drift"


def test_directional_and_version_ablations_restore_expected_decisions() -> None:
    record = _record()
    alignment = torch.tensor([0.2])
    prediction = torch.tensor([1.0])
    improved = torch.tensor([0.5])

    alignment_only = _builder_experiment(directional=False)
    old_candidates = alignment_only._build_memory_candidates(
        record,
        alignment,
        prediction,
        timestamp=8,
        current_loss=improved,
    )
    assert len(old_candidates) == 1
    assert old_candidates[0].recovery_eligible
    assert _manager(directional=False).add_candidate(
        old_candidates[0]
    ) == "recovery"

    version_disabled = _builder_experiment(version=False)
    ignored_candidates = version_disabled._build_memory_candidates(
        record,
        alignment,
        prediction,
        timestamp=8,
        current_loss=torch.tensor([2.0]),
    )
    assert len(ignored_candidates) == 1
    assert ignored_candidates[0].last_alignment == 1.0
    assert _manager(
        directional=True, version=False
    ).add_candidate(ignored_candidates[0]) == "stable"


def test_periodic_refresh_demotes_only_harmful_drift() -> None:
    beneficial_manager = _manager()
    beneficial = _memory_item(1, alignment=0.9)
    assert beneficial_manager.add_candidate(beneficial) == "stable"
    current_sketch = torch.tensor(
        [0.0, 1.0], requires_grad=True
    )

    beneficial_stats = beneficial_manager.refresh(
        lambda expert_id, item: (0.2, 0.8, current_sketch),
        timestamp=2,
        count_recovery_attempts=False,
    )

    assert beneficial_stats["stable_to_recovery"] == 0
    assert beneficial_stats["recovery_skipped_non_degraded"] == 1
    assert beneficial_stats["capability_rebase"] == 1
    stored = beneficial_manager.stable_buffers[0].get(1)
    assert stored is not None
    assert not stored.prediction_capability_sketch.requires_grad
    assert torch.equal(
        stored.prediction_capability_sketch,
        torch.tensor([0.0, 1.0]),
    )
    assert stored.reference_capability_loss == pytest.approx(0.8)
    assert stored.prediction_time_loss == pytest.approx(1.0)
    assert stored.recovery_credit == 0.0

    harmful_manager = _manager()
    harmful = _memory_item(2, alignment=0.9)
    assert harmful_manager.add_candidate(harmful) == "stable"
    harmful_stats = harmful_manager.refresh(
        lambda expert_id, item: (
            0.2,
            1.2,
            torch.tensor([0.0, 1.0]),
        ),
        timestamp=3,
        count_recovery_attempts=False,
    )
    assert harmful_stats["stable_to_recovery"] == 1
    assert harmful_stats["harmful_drift"] == 1
    assert harmful_manager.recovery_buffers[0].contains(2)


def test_performance_recovery_stops_replay_and_rebases_low_alignment_item() -> None:
    manager = _manager()
    recovering = _memory_item(
        3,
        alignment=0.2,
        category="harmful_drift",
        recovery_eligible=True,
    )
    assert manager.add_candidate(recovering) == "recovery"
    current_sketch = torch.tensor(
        [0.0, 2.0], requires_grad=True
    )

    status = manager.update_recovery_result(
        0,
        3,
        alignment=0.2,
        prediction_loss=0.9,
        current_sketch=current_sketch,
    )

    assert status == "promoted_performance"
    assert manager.sample_recovery(0, batch_size=1) == ()
    stored = manager.stable_buffers[0].get(3)
    assert stored is not None
    assert not stored.prediction_capability_sketch.requires_grad
    assert torch.equal(
        stored.prediction_capability_sketch,
        torch.tensor([0.0, 1.0]),
    )
    assert stored.reference_capability_loss == pytest.approx(0.9)
    assert stored.prediction_time_loss == pytest.approx(1.0)

    def evaluate_rebased(expert_id, item):
        del expert_id
        current = torch.tensor([0.0, 1.0])
        cosine = torch.dot(item.normalized_sketch.float(), current)
        alignment = float(((cosine.clamp(-1.0, 1.0) + 1.0) / 2.0).item())
        return alignment, 0.9, current

    next_stats = manager.refresh(
        evaluate_rebased,
        timestamp=4,
        count_recovery_attempts=False,
    )
    assert next_stats["stable_to_recovery"] == 0
    assert manager.stable_buffers[0].contains(3)


class _StatefulDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 1
        self.state_updates_enabled = True
        self.output = nn.Parameter(torch.tensor([1.0]))
        self.representation = nn.Parameter(torch.tensor([0.0, 1.0]))
        self.register_buffer("persistent_updates", torch.zeros((), dtype=torch.long))

    def forward_experts(self, x, x_mark, return_repr=False):
        del x_mark
        if self.state_updates_enabled:
            self.persistent_updates.add_(1)
        outputs = self.output.reshape(1, 1, 1).expand(x.shape[0], -1, -1)
        representations = self.representation.reshape(
            1, 1, 2
        ).expand(x.shape[0], -1, -1)
        return (outputs, representations) if return_repr else outputs

    def compute_capability_sketch(self, representations):
        return torch.nn.functional.normalize(representations, dim=-1)


def test_direction_diagnostics_are_read_only_even_when_recovery_disabled() -> None:
    experiment = _builder_experiment(recovery=False)
    experiment.device = torch.device("cpu")
    experiment.model = _StatefulDiagnosticModel()
    experiment.args = SimpleNamespace(pred_len=1, c_out=1)
    experiment.sample_credit_temperature = 1.0
    experiment.expert_update_count = 0
    experiment.credit_diagnostics = deque(maxlen=4)
    experiment.credit_diagnostic_total_count = 0
    record = _record()
    record.metadata["expert_update_count_at_prediction"] = 0
    before = experiment.model.persistent_updates.clone()

    alignment, _, _, prediction_loss = experiment._evaluate_completed_record(
        record, current_origin=8
    )

    assert torch.equal(experiment.model.persistent_updates, before)
    assert experiment.model.state_updates_enabled
    assert all(
        parameter.grad is None
        for parameter in experiment.model.parameters()
    )
    diagnostic = experiment.credit_diagnostics[-1]
    assert diagnostic["capability_evolution"] == ["harmful_drift"]
    assert diagnostic["harmful_drift"] == [True]
    assert diagnostic["prediction_expert_mse"] == [0.0]
    assert diagnostic["current_expert_mse"] == [1.0]
    assert experiment.diagnostics.counters["harmful_drift_count"] == 1.0
    assert experiment._build_memory_candidates(
        record,
        alignment,
        prediction_loss,
        timestamp=8,
        current_loss=torch.tensor(
            diagnostic["current_expert_mse"]
        ),
    ) == []


def _alignment_to_reference(
    item: VersionedMemoryItem, current_sketch: torch.Tensor
) -> float:
    current = torch.nn.functional.normalize(
        current_sketch.float(), dim=-1
    )
    reference = item.normalized_sketch.float()
    cosine = torch.dot(reference, current).clamp(-1.0, 1.0)
    return float(((cosine + 1.0) / 2.0).item())


def test_rebased_loss_reference_detects_next_harmful_drift() -> None:
    manager = _manager()
    item = _memory_item(10, alignment=0.9, prediction_loss=1.0)
    assert manager.add_candidate(item) == "stable"

    first_sketch = torch.tensor([0.0, 1.0])
    first = manager.refresh(
        lambda expert_id, stored: (
            _alignment_to_reference(stored, first_sketch),
            0.5,
            first_sketch,
        ),
        timestamp=11,
        count_recovery_attempts=False,
    )

    stored = manager.stable_buffers[0].get(10)
    assert stored is not None
    assert first["beneficial_evolution"] == 1
    assert first["stable_to_recovery"] == 0
    assert stored.reference_capability_loss == pytest.approx(0.5)
    assert stored.prediction_time_loss == pytest.approx(1.0)

    second_sketch = torch.tensor([1.0, 0.0])
    second = manager.refresh(
        lambda expert_id, stored: (
            _alignment_to_reference(stored, second_sketch),
            0.8,
            second_sketch,
        ),
        timestamp=12,
        count_recovery_attempts=False,
    )

    recovering = manager.recovery_buffers[0].get(10)
    assert second["harmful_drift"] == 1
    assert second["stable_to_recovery"] == 1
    assert recovering is not None
    assert recovering.reference_capability_loss == pytest.approx(0.5)
    assert recovering.prediction_time_loss == pytest.approx(1.0)


def test_consecutive_beneficial_rebases_preserve_original_loss() -> None:
    manager = _manager()
    item = _memory_item(11, alignment=0.9, prediction_loss=1.0)
    assert manager.add_candidate(item) == "stable"

    for timestamp, loss, sketch in (
        (12, 0.7, torch.tensor([0.0, 1.0])),
        (13, 0.5, torch.tensor([1.0, 0.0])),
    ):
        stats = manager.refresh(
            lambda expert_id, stored, current=sketch, current_loss=loss: (
                _alignment_to_reference(stored, current),
                current_loss,
                current,
            ),
            timestamp=timestamp,
            count_recovery_attempts=False,
        )
        stored = manager.stable_buffers[0].get(11)
        assert stats["beneficial_evolution"] == 1
        assert stats["stable_to_recovery"] == 0
        assert stored is not None
        assert stored.reference_capability_loss == pytest.approx(loss)
        assert stored.prediction_time_loss == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("current_loss", "expected_kind"),
    [(0.54, "stable"), (0.56, "recovery")],
)
def test_degradation_margin_uses_latest_rebased_loss(
    current_loss: float, expected_kind: str
) -> None:
    manager = _manager(margin=0.1)
    item = _memory_item(12, alignment=0.9, prediction_loss=1.0)
    assert manager.add_candidate(item) == "stable"

    first_sketch = torch.tensor([0.0, 1.0])
    manager.refresh(
        lambda expert_id, stored: (
            _alignment_to_reference(stored, first_sketch),
            0.5,
            first_sketch,
        ),
        timestamp=13,
        count_recovery_attempts=False,
    )
    stored = manager.stable_buffers[0].get(12)
    assert stored is not None
    assert stored.reference_capability_loss == pytest.approx(0.5)

    second_sketch = torch.tensor([1.0, 0.0])
    manager.refresh(
        lambda expert_id, stored: (
            _alignment_to_reference(stored, second_sketch),
            current_loss,
            second_sketch,
        ),
        timestamp=14,
        count_recovery_attempts=False,
    )

    if expected_kind == "stable":
        stored = manager.stable_buffers[0].get(12)
        assert stored is not None
        assert stored.reference_capability_loss == pytest.approx(0.54)
        assert not manager.recovery_buffers[0].contains(12)
    else:
        stored = manager.recovery_buffers[0].get(12)
        assert stored is not None
        assert stored.reference_capability_loss == pytest.approx(0.5)
        assert not manager.stable_buffers[0].contains(12)
    assert stored.prediction_time_loss == pytest.approx(1.0)


def test_beneficial_new_candidate_uses_rebased_stable_reference() -> None:
    experiment = _builder_experiment()
    experiment.memory_manager = _manager()
    record = _record()
    current_sketch = torch.tensor(
        [[0.0, 1.0]], requires_grad=True
    )

    candidates = experiment._build_memory_candidates(
        record,
        alignment=torch.tensor([0.2]),
        prediction_loss=torch.tensor([1.0]),
        current_loss=torch.tensor([0.5]),
        current_sketch=current_sketch,
        timestamp=8,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.capability_evolution == (
        "beneficial_or_neutral_evolution"
    )
    assert candidate.last_alignment == pytest.approx(1.0)
    assert candidate.last_observed_alignment == pytest.approx(0.2)
    assert candidate.reference_capability_loss == pytest.approx(0.5)
    assert candidate.prediction_time_loss == pytest.approx(1.0)
    assert not candidate.prediction_capability_sketch.requires_grad

    experiment._commit_memory_candidates(candidates)
    stored = experiment.memory_manager.stable_buffers[0].get(7)
    assert stored is not None
    assert not experiment.memory_manager.recovery_buffers[0].contains(7)
    assert stored.reference_capability_loss == pytest.approx(0.5)
    assert stored.prediction_time_loss == pytest.approx(1.0)
    assert torch.equal(
        stored.prediction_capability_sketch,
        torch.tensor([0.0, 1.0]),
    )
    assert experiment.diagnostics.counters[
        "recovery_skipped_non_degraded_count"
    ] == 1.0
    assert experiment.diagnostics.counters[
        "capability_rebase_count"
    ] == 1.0


def test_alignment_only_ablation_ignores_reference_loss_gate() -> None:
    manager = _manager(directional=False)
    item = _memory_item(13, alignment=0.9, prediction_loss=1.0)
    item.reference_capability_loss = 0.5
    assert manager.add_candidate(item) == "stable"

    stats = manager.refresh(
        lambda expert_id, stored: (
            0.2,
            0.4,
            torch.tensor([0.0, 1.0]),
        ),
        timestamp=14,
        count_recovery_attempts=False,
    )

    assert stats["stable_to_recovery"] == 1
    assert manager.recovery_buffers[0].contains(13)

def test_legacy_item_falls_back_to_original_prediction_loss() -> None:
    item = _memory_item(14, alignment=0.9, prediction_loss=1.0)
    del item.reference_capability_loss

    assert item.get_reference_capability_loss() == pytest.approx(1.0)
    assert item.reference_capability_loss == pytest.approx(1.0)
