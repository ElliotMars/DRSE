#!/usr/bin/env python3
"""Validate progressive smoke-test diagnostics without loading model state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = (
    "online_diagnostics.npz",
    "credit_diagnostics.npz",
    "online_diagnostics_summary.json",
)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _numeric(value: Any, field: str, errors: list[str]) -> np.ndarray | None:
    if value is None:
        errors.append(f"missing field {field}")
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f"{field} is not numeric")
        return None
    if array.size == 0:
        errors.append(f"{field} is empty")
        return None
    if not np.isfinite(array).all():
        errors.append(f"{field} contains NaN or Inf")
        return None
    return array


def _iteration_directories(result_directory: Path) -> list[Path]:
    if all((result_directory / name).is_file() for name in REQUIRED_FILES):
        return [result_directory]
    return sorted(
        (
            child
            for child in result_directory.glob("itr_*")
            if child.is_dir()
        ),
        key=lambda path: int(path.name.split("_", 1)[1]),
    )


def _check_npz(path: Path, errors: list[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = array
                if array.dtype.kind not in "biuf":
                    errors.append(f"{path.name}:{name} is not numeric")
                elif not np.isfinite(array).all():
                    errors.append(
                        f"{path.name}:{name} contains NaN or Inf"
                    )
    except (OSError, ValueError, EOFError) as exc:
        errors.append(f"cannot load {path.name}: {exc}")
    return arrays


def _check_range(
    value: Any,
    field: str,
    lower: float,
    upper: float | None,
    errors: list[str],
) -> None:
    array = _numeric(value, field, errors)
    if array is None:
        return
    if (array < lower).any() or (
        upper is not None and (array > upper).any()
    ):
        bound = f"[{lower},{upper}]" if upper is not None else f">={lower}"
        errors.append(f"{field} is outside {bound}")


def _check_iteration(directory: Path) -> list[str]:
    prefix = directory.name
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            errors.append(f"{prefix}: missing file {name}")
    if errors:
        return errors

    try:
        with (directory / REQUIRED_FILES[2]).open(
            "r", encoding="utf-8"
        ) as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{prefix}: invalid summary JSON: {exc}"]

    online = _check_npz(directory / REQUIRED_FILES[0], errors)
    credit = _check_npz(directory / REQUIRED_FILES[1], errors)
    metrics = summary.get("metrics", {})
    metadata = summary.get("iteration", {})

    for field in (
        "online_mse",
        "raw_prior_entropy",
        "effective_routing_entropy",
        "z_norm",
    ):
        _numeric(
            _nested(metrics, field, "mean"),
            f"metrics.{field}.mean",
            errors,
        )
    _check_range(
        _nested(metrics, "capability_alignment", "mean"),
        "metrics.capability_alignment.mean",
        0.0,
        1.0,
        errors,
    )
    _check_range(
        _nested(metrics, "responsibility_js_divergence", "mean"),
        "metrics.responsibility_js_divergence.mean",
        0.0,
        None,
        errors,
    )
    _check_range(
        _nested(metrics, "ranking_reversal", "mean"),
        "metrics.ranking_reversal.mean",
        0.0,
        1.0,
        errors,
    )
    _check_range(
        _nested(metrics, "subspace_captured_energy", "mean"),
        "metrics.subspace_captured_energy.mean",
        0.0,
        1.0,
        errors,
    )

    stable_capacity = metadata.get("stable_buffer_capacity")
    recovery_capacity = metadata.get("recovery_buffer_capacity")
    max_rank = metadata.get("subspace_max_rank")
    for field, capacity in (
        ("stable_buffer_size", stable_capacity),
        ("recovery_buffer_size", recovery_capacity),
    ):
        values = _numeric(online.get(field), field, errors)
        numeric_capacity = _numeric(
            capacity, f"iteration.{field}_capacity", errors
        )
        if values is not None and numeric_capacity is not None:
            if (values < 0).any() or (
                values > float(numeric_capacity.max())
            ).any():
                errors.append(f"{field} exceeds configured capacity")
    ranks = _numeric(online.get("subspace_rank"), "subspace_rank", errors)
    numeric_max_rank = _numeric(
        max_rank, "iteration.subspace_max_rank", errors
    )
    if ranks is not None and numeric_max_rank is not None:
        if (ranks < 0).any() or (ranks > float(numeric_max_rank.max())).any():
            errors.append("subspace_rank exceeds configured maximum")

    num_records = int(summary.get("num_records", 0))
    if num_records <= 0:
        errors.append("diagnostics contain no records")
    if credit.get("origin", np.asarray([])).size <= 0:
        errors.append("credit diagnostics contain no completed record")

    strict_failures = metadata.get("strict_check_failure_count")
    strict_array = _numeric(
        strict_failures,
        "iteration.strict_check_failure_count",
        errors,
    )
    if strict_array is not None and (strict_array != 0).any():
        errors.append("strict check failure count is non-zero")
    if metadata.get("strict_checks_passed") is not True:
        errors.append("strict checks did not pass")

    completed = _numeric(
        metadata.get("completed_record_count"),
        "iteration.completed_record_count",
        errors,
    )
    processed = _numeric(
        metadata.get("processed_origins"),
        "iteration.processed_origins",
        errors,
    )
    pred_len = _numeric(
        metadata.get("pred_len"), "iteration.pred_len", errors
    )
    if completed is not None and processed is not None and pred_len is not None:
        theoretical = max(
            0,
            int(processed.max()) - int(pred_len.max()),
        )
        if int(completed.max()) > theoretical:
            errors.append(
                "completed_record_count exceeds theoretical mature records"
            )

    return [f"{prefix}: {error}" for error in errors]


def check_smoke_results(result_directory: str | os.PathLike[str]) -> list[str]:
    """Return validation errors; an empty list means PASS."""

    root = Path(result_directory)
    if not root.is_dir():
        return [f"result directory does not exist: {root}"]
    iterations = _iteration_directories(root)
    if not iterations:
        return [f"no diagnostics or itr_* directory found in {root}"]
    errors: list[str] = []
    for directory in iterations:
        errors.extend(_check_iteration(directory))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    args = parser.parse_args(argv)
    errors = check_smoke_results(args.result_directory)
    if errors:
        for error in errors:
            print(f"Smoke test diagnostics: FAIL: {error}", file=sys.stderr)
        return 1
    print("Smoke test diagnostics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
