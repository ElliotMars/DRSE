import numpy as np
import pytest
import torch

from data_provider.synthetic_drift import SyntheticDriftDataset
from utils.drift_metrics import (
    DriftMetricTracker,
    MemoryTimelineRecorder,
    compute_drift_metrics,
    recovery_time,
)
from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem


def _dataset(kind: str, **kwargs) -> SyntheticDriftDataset:
    return SyntheticDriftDataset(
        seq_len=6,
        pred_len=2,
        channels=3,
        total_length=60,
        seed=11,
        noise_std=0.02,
        drift_type=kind,
        **kwargs,
    )


def test_abrupt_change_point_and_metadata_are_exact() -> None:
    dataset = _dataset("abrupt")

    assert dataset.intervals == {"A": (0, 30), "B": (30, 60)}
    assert np.all(dataset.regime_id[:30] == 0)
    assert np.all(dataset.regime_id[30:] == 1)
    assert dataset.events[0].timestamp == 30
    assert dataset.events[0].origin == 24


def test_gradual_alpha_is_monotone_and_has_fixed_endpoints() -> None:
    dataset = _dataset("gradual", transition_window=12)
    start, end = dataset.intervals["transition"]
    transition = dataset.transition_alpha[start:end]

    assert transition[0] == 0.0
    assert transition[-1] == 1.0
    assert np.all(np.diff(transition) >= 0.0)
    assert np.all(dataset.transition_alpha[:start] == 0.0)
    assert np.all(dataset.transition_alpha[end:] == 1.0)


def test_recurring_metadata_is_A_then_B_then_A() -> None:
    dataset = _dataset("recurring")

    assert dataset.intervals["first_A"] == (0, 20)
    assert dataset.intervals["B"] == (20, 40)
    assert dataset.intervals["recurring_A"] == (40, 60)
    assert dataset.events[0].from_regime == "A"
    assert dataset.events[0].to_regime == "B"
    assert dataset.events[1].from_regime == "B"
    assert dataset.events[1].to_regime == "A"
    assert np.array_equal(
        dataset.regime_id,
        np.asarray([0] * 20 + [1] * 20 + [0] * 20),
    )


def test_seed_reproducibility_and_channel_local_drift() -> None:
    first = _dataset("abrupt", drift_channels=(1,))
    repeated = _dataset("abrupt", drift_channels=(1,))
    different = SyntheticDriftDataset(
        seq_len=6,
        pred_len=2,
        channels=3,
        total_length=60,
        seed=12,
        noise_std=0.02,
        drift_type="abrupt",
        drift_channels=(1,),
    )
    no_drift_channels = _dataset("abrupt", drift_channels=())

    assert np.array_equal(first.series, repeated.series)
    assert not np.array_equal(first.series, different.series)
    np.testing.assert_array_equal(first.series[:, 0], no_drift_channels.series[:, 0])
    np.testing.assert_array_equal(first.series[:, 2], no_drift_channels.series[:, 2])
    assert not np.array_equal(first.series[30:, 1], no_drift_channels.series[30:, 1])


def test_rolling_origin_context_does_not_contain_future_target() -> None:
    dataset = _dataset("shock", shock_duration=5)
    index = 7
    context = dataset.context_at(index)
    target = dataset.evaluation_target_at(index)

    np.testing.assert_array_equal(
        context.numpy(), dataset.series[index : index + dataset.seq_len]
    )
    np.testing.assert_array_equal(
        target.numpy(),
        dataset.series[
            index + dataset.seq_len : index + dataset.seq_len + dataset.pred_len
        ],
    )
    target.fill_(999.0)
    assert not np.any(dataset.series == 999.0)
    assert context[-1].equal(torch.from_numpy(dataset.series[index + 5]))


def test_recovery_time_toy_case_and_no_recovery_sentinel() -> None:
    timeline = [1.0, 1.0, 4.0, 3.0, 1.1, 1.0, 1.05]

    assert recovery_time(
        timeline,
        drift_origin=2,
        pre_drift_error=1.0,
        recovery_window=2,
        recovery_tolerance=0.1,
    ) == 2
    assert recovery_time(
        [1.0, 1.0, 4.0, 3.0, 2.0],
        drift_origin=2,
        pre_drift_error=1.0,
        recovery_window=2,
        recovery_tolerance=0.1,
    ) is None

    metrics = compute_drift_metrics(
        [1.0, 1.0, 4.0, 3.0, 2.0],
        [{"name": "change", "kind": "abrupt", "origin": 2}],
        pre_window=2,
        early_window=2,
        recovery_window=2,
        recovery_tolerance=0.1,
    )
    assert metrics["events"][0]["recovery_time"] is None


def _memory_item() -> VersionedMemoryItem:
    sketch = torch.tensor([1.0, 0.0])
    return VersionedMemoryItem(
        sample_id=1,
        origin=1,
        expert_id=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 1, 1),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=1.0,
        last_alignment=1.0,
        stable_credit=1.0,
        recovery_credit=0.0,
        timestamp=1,
    )


def test_metric_and_memory_recorders_reset_cleanly() -> None:
    metric = DriftMetricTracker()
    metric.update(1.5)
    manager = ExpertMemoryManager(
        num_experts=1,
        stable_capacity=2,
        recovery_capacity=2,
        responsibility_threshold=0.0,
        alignment_threshold=0.5,
        duplicate_threshold=1.1,
        failure_penalty=0.5,
        max_recovery_attempts=2,
    )
    manager.add_candidate(_memory_item())
    recorder = MemoryTimelineRecorder(1)
    recorder.record(
        3,
        manager,
        transition_stats={"stable_to_recovery": 1},
        sample_regimes={1: 0},
    )

    metric.reset()
    recorder.reset()

    assert metric.mse == []
    assert recorder.arrays()["stable_occupancy"].shape == (0, 1)
    assert recorder.summary()["num_steps"] == 0
    assert all(value == 0 for value in recorder.transitions.values())
