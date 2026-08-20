import json

import numpy as np

from utils.synthetic_drift_benchmark import (
    SyntheticBenchmarkConfig,
    SyntheticProgressiveBenchmark,
)



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
