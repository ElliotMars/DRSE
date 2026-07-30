import copy
import json

import numpy as np

from scripts.check_smoke_results import check_smoke_results, main


def _valid_summary():
    metric = lambda value: {"mean": value, "last": value}
    return {
        "num_records": 2,
        "counters": {},
        "metrics": {
            "online_mse": metric(1.0),
            "raw_prior_entropy": metric(0.9),
            "effective_routing_entropy": metric(0.7),
            "z_norm": metric(0.2),
            "capability_alignment": metric([0.8, 0.9]),
            "responsibility_js_divergence": metric(0.1),
            "ranking_reversal": metric(0.5),
            "subspace_captured_energy": metric([0.0, 0.95]),
        },
        "iteration": {
            "index": 0,
            "seed": 0,
            "pred_len": 2,
            "processed_origins": 5,
            "completed_record_count": 3,
            "strict_checks_passed": True,
            "strict_check_failure_count": 0,
            "stable_buffer_capacity": 2,
            "recovery_buffer_capacity": 1,
            "subspace_max_rank": 2,
        },
    }


def _write_valid(root):
    directory = root / "itr_0"
    directory.mkdir(parents=True)
    np.savez_compressed(
        directory / "online_diagnostics.npz",
        online_mse=[1.0, 0.8],
        raw_prior_entropy=[0.9, 0.8],
        effective_routing_entropy=[0.7, 0.6],
        z_norm=[0.1, 0.2],
        stable_buffer_size=[[1, 2], [2, 2]],
        recovery_buffer_size=[[0, 1], [1, 1]],
        subspace_rank=[[0, 1], [1, 2]],
        subspace_captured_energy=[[0.0, 0.9], [0.8, 0.95]],
    )
    np.savez_compressed(
        directory / "credit_diagnostics.npz",
        origin=[0],
        js_divergence=[0.1],
        mean_alignment=[0.85],
    )
    (directory / "online_diagnostics_summary.json").write_text(
        json.dumps(_valid_summary()), encoding="utf-8"
    )
    return directory


def test_valid_smoke_result_passes(tmp_path, capsys) -> None:
    _write_valid(tmp_path)
    assert check_smoke_results(tmp_path) == []
    assert main([str(tmp_path)]) == 0
    assert "Smoke test diagnostics: PASS" in capsys.readouterr().out


def test_nan_fails(tmp_path) -> None:
    directory = _write_valid(tmp_path)
    np.savez_compressed(
        directory / "online_diagnostics.npz",
        online_mse=[np.nan],
    )
    errors = check_smoke_results(tmp_path)
    assert any("NaN or Inf" in error for error in errors)


def test_capacity_violation_fails(tmp_path) -> None:
    directory = _write_valid(tmp_path)
    with np.load(directory / "online_diagnostics.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["stable_buffer_size"] = np.asarray([[3, 0]])
    np.savez_compressed(directory / "online_diagnostics.npz", **arrays)
    errors = check_smoke_results(tmp_path)
    assert any("stable_buffer_size exceeds" in error for error in errors)


def test_missing_file_fails(tmp_path) -> None:
    directory = _write_valid(tmp_path)
    (directory / "credit_diagnostics.npz").unlink()
    errors = check_smoke_results(tmp_path)
    assert any("missing file credit_diagnostics.npz" in error for error in errors)


def test_strict_failure_fails(tmp_path) -> None:
    directory = _write_valid(tmp_path)
    summary = copy.deepcopy(_valid_summary())
    summary["iteration"]["strict_check_failure_count"] = 1
    summary["iteration"]["strict_checks_passed"] = False
    (directory / "online_diagnostics_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    errors = check_smoke_results(tmp_path)
    assert any("strict check failure count" in error for error in errors)
