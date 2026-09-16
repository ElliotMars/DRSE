import json

import numpy as np
import torch

from utils.synthetic_drift_benchmark import (
    SyntheticBenchmarkConfig,
    SyntheticProgressiveBenchmark,
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

    def guarded_target(index):
        assert benchmark.protocol_trace[-1] == (index, "evaluate")
        return original_target(index)

    def forbidden_dataset_item(instance, index):
        raise AssertionError("runner must not materialize future target before predict")

    monkeypatch.setattr(benchmark.dataset, "evaluation_target_at", guarded_target)
    monkeypatch.setattr(type(benchmark.dataset), "__getitem__", forbidden_dataset_item)
    result = benchmark.run(str(tmp_path))
    origins = len(benchmark.dataset)

    assert result.predictions.shape == (origins, 2, 2)
    assert result.targets.shape == result.predictions.shape
    assert result.mse_timeline.shape == (origins,)
    assert np.isfinite(result.mse_timeline).all()
    assert result.completed_records == origins - config.pred_len
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
        "synthetic_timeline.npz",
    ):
        assert (tmp_path / filename).exists()
    assert json.loads((tmp_path / "drift_metrics.json").read_text())["events"]
    with np.load(tmp_path / "synthetic_timeline.npz") as timeline:
        assert timeline["series"].shape == (config.total_length, config.channels)
        assert timeline["regime_id"].shape == (config.total_length,)
        assert timeline["transition_alpha"].shape == (config.total_length,)
        assert timeline["stable_occupancy_by_regime"].shape[0] == origins
        assert timeline["recovery_occupancy_by_regime"].shape[0] == origins


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
