import json

import numpy as np

from utils.iteration_diagnostics import (
    aggregate_online_diagnostics,
    annotate_iteration_summary,
    save_prediction_results,
)
from utils.online_diagnostics import OnlineDiagnosticsRecorder


def _write_iteration(
    root,
    index: int,
    seed: int,
    mse: float,
    tsb_conflict: float | None,
):
    directory = root / f"itr_{index}"
    recorder = OnlineDiagnosticsRecorder(num_experts=2, interval=1)
    values = {"online_mse": mse, "router_gap": mse / 10.0}
    if tsb_conflict is not None:
        values["tsb_conflict_rate"] = tsb_conflict
    recorder.update(**values)
    recorder.maybe_record(0)
    _, summary_path = recorder.save(str(directory))
    np.savez_compressed(directory / "credit_diagnostics.npz", origin=[index])
    save_prediction_results(
        str(directory),
        [mse] * 6,
        np.asarray([mse]),
        np.asarray([0.0]),
        np.asarray([mse]),
        np.asarray([mse]),
    )
    annotate_iteration_summary(
        summary_path,
        {
            "index": index,
            "seed": seed,
            "early_ended": False,
            "strict_checks_passed": True,
            "completed_record_count": 1,
        },
    )
    return summary_path


def test_two_iterations_are_isolated_and_aggregated(tmp_path) -> None:
    first = _write_iteration(tmp_path, 0, 11, 1.0, None)
    second = _write_iteration(tmp_path, 1, 12, 3.0, 0.4)

    output = tmp_path / "aggregate_diagnostics_summary.json"
    aggregate = aggregate_online_diagnostics([first, second], str(output))

    assert (tmp_path / "itr_0" / "online_diagnostics.npz").exists()
    assert (tmp_path / "itr_1" / "online_diagnostics.npz").exists()
    assert np.load(tmp_path / "itr_0" / "metrics.npy")[0] == 1.0
    assert np.load(tmp_path / "itr_1" / "metrics.npy")[0] == 3.0
    assert [item["seed"] for item in aggregate["iterations"]] == [11, 12]
    assert aggregate["metrics"]["online_mse"]["mean"] == 2.0
    assert aggregate["metrics"]["online_mse"]["std"] == 1.0
    assert aggregate["metrics"]["tsb_conflict_rate"]["count"] == 1
    assert aggregate["metrics"]["tsb_conflict_rate"]["mean"] == 0.4
    assert json.loads(output.read_text()) == aggregate


def test_tsb_streaming_summary_metrics_are_aggregated() -> None:
    aggregate = aggregate_online_diagnostics(
        [
            {
                "tsb_gradient_diagnostics": {
                    "mean_grad_cosine": 0.5,
                    "conflict_rate": 0.25,
                    "mean_tsb_modification_ratio": 0.1,
                }
            },
            {
                "tsb_gradient_diagnostics": {
                    "mean_grad_cosine": -0.5,
                    "conflict_rate": 0.75,
                    "mean_tsb_modification_ratio": 0.3,
                }
            },
        ]
    )

    assert aggregate["metrics"]["mean_grad_cosine"]["mean"] == 0.0
    assert aggregate["metrics"]["gradient_conflict_rate"]["mean"] == 0.5
    assert aggregate["metrics"]["mean_tsb_modification_ratio"]["mean"] == 0.2


def test_single_iteration_keeps_legacy_prediction_filenames(tmp_path) -> None:
    summary_path = _write_iteration(tmp_path, 0, 7, 2.5, None)
    aggregate = aggregate_online_diagnostics([summary_path])

    directory = tmp_path / "itr_0"
    for name in ("metrics.npy", "preds.npy", "trues.npy", "mae.npy", "mse.npy"):
        assert (directory / name).exists()
    assert aggregate["num_iterations"] == 1
    assert aggregate["metrics"]["online_mse"]["std"] == 0.0
    assert aggregate["metrics"]["tsb_conflict_rate"] is None

