import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from utils.synthetic_drift_benchmark import (
    SyntheticBenchmarkConfig,
    SyntheticProgressiveBenchmark,
    result_directory,
)
from utils.progressive_feedback import ProgressiveForecastRecord



def test_tiny_synthetic_progressive_pipeline_finishes_and_saves(
    tmp_path, monkeypatch
) -> None:
    config = SyntheticBenchmarkConfig(
        seq_len=6,
        pred_len=2,
        channels=2,
        total_length=38,
        seed=5,
        noise_std=0.01,
        drift_type="recurring",
        drift_channels=(1,),
        num_experts=2,
        strategy="hybrid",
        stable_capacity=6,
        recovery_capacity=6,
        memory_refresh_interval=1,
        pre_window=4,
        early_window=3,
        recovery_window=2,
    )
    benchmark = SyntheticProgressiveBenchmark(config)
    original_target = benchmark.dataset.evaluation_target_at
    original_predict = benchmark._predict_record
    prediction_calls = []

    def counted_predict(origin):
        prediction_calls.append(origin)
        return original_predict(origin)

    def guarded_target(index):
        assert benchmark.protocol_trace[-1] == (index, "evaluate")
        return original_target(index)

    def forbidden_dataset_item(instance, index):
        raise AssertionError("runner must not materialize future target before predict")

    monkeypatch.setattr(benchmark, "_predict_record", counted_predict)
    monkeypatch.setattr(benchmark.dataset, "evaluation_target_at", guarded_target)
    monkeypatch.setattr(type(benchmark.dataset), "__getitem__", forbidden_dataset_item)
    result = benchmark.run(str(tmp_path))
    origins = len(benchmark.dataset)

    assert result.predictions.shape == (origins, 2, 2)
    assert result.targets.shape == result.predictions.shape
    assert result.mse_timeline.shape == (origins,)
    assert np.isfinite(result.mse_timeline).all()
    assert result.completed_records == origins - config.pred_len
    assert prediction_calls == list(range(origins))
    assert result.memory_arrays["stable_occupancy"].shape == (origins, 2)
    assert "recurring_mode" in result.drift_metrics

    for origin in range(origins):
        labels = [label for step, label in result.protocol_trace if step == origin]
        assert labels.index("release") < labels.index("predict")
        assert labels.index("predict") < labels.index("evaluate")

    for filename in (
        "predictions_and_mse.npz",
        "drift_events.json",
        "drift_metrics.json",
        "memory_diagnostics.json",
        "regime_diagnostics.json",
        "synthetic_timeline.npz",
        "timeline.npz",
    ):
        assert (tmp_path / filename).exists()
    assert json.loads((tmp_path / "drift_metrics.json").read_text())["events"]
    with np.load(tmp_path / "synthetic_timeline.npz") as timeline:
        assert timeline["series"].shape == (config.total_length, config.channels)
        assert timeline["regime_id"].shape == (config.total_length,)
        assert timeline["transition_alpha"].shape == (config.total_length,)
        assert timeline["stable_occupancy_by_regime"].shape[0] == origins
        assert timeline["recovery_occupancy_by_regime"].shape[0] == origins
    with np.load(tmp_path / "timeline.npz") as timeline:
        required = {
            "origin",
            "regime_id",
            "phase",
            "mse",
            "rolling_mse",
            "stable_size",
            "recovery_size",
            "harmful_drift_events",
            "beneficial_evolution_events",
            "capability_rebase_events",
            "stable_to_recovery_events",
            "recovery_to_stable_events",
        }
        assert required.issubset(timeline.files)
        assert timeline["origin"].shape == (origins,)
        phases = timeline["phase"].tolist()
        assert phases[:6] == ["A1"] * 6
        assert phases[6:19] == ["B"] * 13
        assert phases[19:] == ["A2"] * (origins - 19)
        np.testing.assert_array_equal(
            timeline["regime_id"],
            np.asarray([0] * 6 + [1] * 13 + [0] * (origins - 19)),
        )


def test_recovery_disabled_has_zero_recovery_occupancy() -> None:
    config = SyntheticBenchmarkConfig(
        seq_len=5,
        pred_len=2,
        channels=2,
        total_length=30,
        seed=2,
        drift_type="abrupt",
        num_experts=2,
        strategy="plain",
        disable_recovery=True,
        memory_refresh_interval=1,
        pre_window=3,
        early_window=2,
        recovery_window=2,
    )

    result = SyntheticProgressiveBenchmark(config).run()

    assert not result.memory_arrays["recovery_occupancy"].any()
    assert all(
        value == 0
        for value in result.memory_diagnostics["transitions"].values()
    )


def _completed_record(
    benchmark: SyntheticProgressiveBenchmark,
    prediction_loss: float,
) -> ProgressiveForecastRecord:
    experts = benchmark.config.num_experts
    horizon = benchmark.config.pred_len
    channels = benchmark.config.channels
    prediction = torch.full(
        (horizon, channels, experts), prediction_loss ** 0.5
    )
    record = ProgressiveForecastRecord(
        origin=0,
        x=torch.zeros(1, benchmark.config.seq_len, channels),
        x_mark=torch.zeros(1, benchmark.config.seq_len, 7),
        expert_predictions=prediction,
        router_prior=torch.full(
            (horizon, channels, experts), 1.0 / experts
        ),
        router_weights=torch.full(
            (horizon, channels, experts), 1.0 / experts
        ),
        mixture_prediction=prediction.mean(dim=-1),
        capability_sketch=benchmark._capability_sketch(),
    )
    record.matured_targets.zero_()
    record.matured_mask.fill_(True)
    record.num_matured = horizon
    record.sample_responsibility.zero_()
    record.sample_responsibility[0] = 1.0
    record.sample_confidence = 1.0
    return record


def _force_low_alignment(benchmark: SyntheticProgressiveBenchmark) -> None:
    with torch.no_grad():
        benchmark.expert_weight[0].mul_(-1.0)
        benchmark.expert_bias[0].add_(0.5)


def test_recurring_benchmark_defaults_to_direction_awareness() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(drift_type="recurring")
    )

    assert benchmark.memory.version_awareness_enabled
    assert benchmark.memory.direction_awareness_enabled
    assert benchmark.memory.capability_rebase_enabled


def test_no_direction_restores_alignment_only_recovery() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(
            disable_directional_recovery=True,
            pred_len=1,
            channels=2,
            num_experts=2,
        )
    )
    record = _completed_record(benchmark, prediction_loss=100.0)
    _force_low_alignment(benchmark)
    candidate = benchmark._candidate_for_record(record, timestamp=1)

    assert not benchmark.memory.direction_awareness_enabled
    assert not benchmark.memory.capability_rebase_enabled
    assert candidate.last_alignment < benchmark.memory.alignment_threshold
    assert benchmark.memory.add_candidate(candidate) == "recovery"


def test_beneficial_evolution_rebases_to_stable() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(pred_len=1, channels=2, num_experts=2)
    )
    record = _completed_record(benchmark, prediction_loss=100.0)
    original_sketch = record.capability_sketch[0].clone()
    _force_low_alignment(benchmark)
    candidate = benchmark._candidate_for_record(record, timestamp=1)

    assert candidate.capability_evolution == "beneficial_or_neutral_evolution"
    assert candidate.prediction_time_loss == 100.0
    assert candidate.reference_capability_loss < candidate.prediction_time_loss
    assert torch.allclose(
        candidate.prediction_capability_sketch, original_sketch
    )
    assert not torch.allclose(candidate.normalized_sketch, original_sketch)
    assert benchmark.memory.add_candidate(candidate) == "stable"


def test_only_harmful_drift_enters_recovery() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(pred_len=1, channels=2, num_experts=2)
    )
    record = _completed_record(benchmark, prediction_loss=0.0)
    prediction_sketch = record.capability_sketch[0].clone()
    _force_low_alignment(benchmark)
    candidate = benchmark._candidate_for_record(record, timestamp=1)

    assert candidate.capability_evolution == "harmful_drift"
    assert candidate.recovery_eligible
    assert candidate.prediction_time_loss == 0.0
    assert candidate.reference_capability_loss == 0.0
    assert torch.allclose(
        candidate.prediction_capability_sketch, prediction_sketch
    )
    assert benchmark.memory.add_candidate(candidate) == "recovery"


def test_no_recovery_keeps_direction_classification_without_lifecycle() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(
            pred_len=1,
            channels=2,
            num_experts=2,
            disable_recovery=True,
        )
    )
    record = _completed_record(benchmark, prediction_loss=0.0)
    _force_low_alignment(benchmark)
    harmful = benchmark._candidate_for_record(record, timestamp=1)

    assert benchmark.memory.direction_awareness_enabled
    assert harmful.capability_evolution == "harmful_drift"
    assert not harmful.recovery_eligible
    assert benchmark.memory.add_candidate(harmful) is None
    assert all(not buffer.items for buffer in benchmark.memory.recovery_buffers)

    benchmark.reset()
    record = _completed_record(benchmark, prediction_loss=100.0)
    _force_low_alignment(benchmark)
    beneficial = benchmark._candidate_for_record(record, timestamp=1)

    assert beneficial.capability_evolution == "beneficial_or_neutral_evolution"
    assert benchmark.memory.add_candidate(beneficial) == "stable"
    assert all(not buffer.items for buffer in benchmark.memory.recovery_buffers)


@pytest.fixture(scope="module")
def controlled_recurring_results(tmp_path_factory):
    output = tmp_path_factory.mktemp("controlled_recurring")
    base = SyntheticBenchmarkConfig(
        seq_len=6,
        pred_len=2,
        channels=2,
        total_length=90,
        seed=0,
        noise_std=0.05,
        drift_type="recurring",
        regime_separation=1.0,
        a1_length=30,
        b_length=30,
        a2_length=30,
        num_experts=2,
        strategy="subspace",
        stable_capacity=16,
        recovery_capacity=16,
        memory_refresh_interval=2,
        pre_window=8,
        early_window=4,
        recovery_window=3,
        recovery_hold_steps=2,
        rolling_window=3,
    )
    configs = {
        "full": base,
        "no_direction": replace(
            base, disable_directional_recovery=True
        ),
        "no_recovery": replace(base, disable_recovery=True),
    }
    return {
        name: SyntheticProgressiveBenchmark(config).run(
            str(output / name)
        )
        for name, config in configs.items()
    }


def test_controlled_full_activates_harmful_drift_and_recovery(
    controlled_recurring_results,
) -> None:
    result = controlled_recurring_results["full"]
    counts = result.memory_diagnostics["diagnostic_counts"]

    assert counts["harmful_drift_count"] > 0
    assert (
        counts["recovery_admission_count"] > 0
        or counts["stable_to_recovery_count"] > 0
    )
    assert set(result.regime_diagnostics["regimes"]) == {"A1", "B", "A2"}
    for phase in result.regime_diagnostics["regimes"].values():
        assert "mean_mse" in phase
        assert "mean_stable_occupancy" in phase
        assert "mean_recovery_occupancy" in phase
        assert "events" in phase


def test_controlled_no_recovery_keeps_direction_diagnostics_only(
    controlled_recurring_results,
) -> None:
    counts = controlled_recurring_results["no_recovery"].memory_diagnostics[
        "diagnostic_counts"
    ]

    assert counts["harmful_drift_count"] > 0
    assert counts["beneficial_evolution_count"] > 0
    assert counts["capability_rebase_count"] > 0
    assert counts["recovery_admission_count"] == 0
    assert counts["recovery_attempt_count"] == 0
    assert counts["recovery_to_stable_count"] == 0
    assert counts["stable_to_recovery_count"] == 0


def test_controlled_no_direction_finishes_with_alignment_only_semantics(
    controlled_recurring_results,
) -> None:
    result = controlled_recurring_results["no_direction"]
    counts = result.memory_diagnostics["diagnostic_counts"]

    assert result.completed_records > 0
    assert counts["capability_rebase_count"] == 0
    assert np.isfinite(result.mse_timeline).all()


def test_prediction_loss_stays_immutable_across_beneficial_rebase() -> None:
    benchmark = SyntheticProgressiveBenchmark(
        SyntheticBenchmarkConfig(pred_len=1, channels=2, num_experts=2)
    )
    record = _completed_record(benchmark, prediction_loss=100.0)
    _force_low_alignment(benchmark)
    candidate = benchmark._candidate_for_record(record, timestamp=1)
    original_prediction_loss = candidate.prediction_time_loss
    assert benchmark.memory.add_candidate(candidate) == "stable"
    stored = benchmark.memory.stable_buffers[0].get(candidate.sample_id)
    assert stored is not None

    current_sketch = -stored.normalized_sketch.float()
    current_loss = stored.reference_capability_loss * 0.5
    stats = benchmark.memory.refresh(
        lambda expert_id, item: (0.0, current_loss, current_sketch),
        timestamp=2,
        count_recovery_attempts=False,
    )

    assert stats["capability_rebase"] == 1
    assert stored.prediction_time_loss == original_prediction_loss
    assert stored.reference_capability_loss == pytest.approx(current_loss)
    assert torch.allclose(
        stored.normalized_sketch.float(),
        torch.nn.functional.normalize(current_sketch, dim=0),
    )


def test_variant_output_directories_do_not_overlap() -> None:
    base = SyntheticBenchmarkConfig(
        drift_type="recurring", strategy="subspace", seed=1
    )
    paths = {
        result_directory("/tmp/results", base),
        result_directory(
            "/tmp/results",
            replace(base, disable_directional_recovery=True),
        ),
        result_directory(
            "/tmp/results", replace(base, disable_recovery=True)
        ),
    }

    assert len(paths) == 3
    assert any(path.endswith("_no_direction") for path in paths)
    assert any(path.endswith("_no_recovery") for path in paths)
