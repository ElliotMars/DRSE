#!/usr/bin/env python3
"""Aggregate controlled recurring-drift runs and draw paper-ready curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


_VARIANTS = {
    "full": "PACE",
    "no_direction": "w/o Direction Awareness",
    "no_recovery": "w/o Recovery",
}


def _variant(name: str) -> str | None:
    if name.endswith("_no_direction"):
        return "no_direction"
    if name.endswith("_no_recovery"):
        return "no_recovery"
    if "_no_version" not in name and "_no_z" not in name:
        return "full"
    return None


def _load_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {
        key: [] for key in _VARIANTS
    }
    for directory in sorted(root.glob("recurring_*_seed*")):
        variant = _variant(directory.name)
        timeline_path = directory / "timeline.npz"
        metrics_path = directory / "drift_metrics.json"
        memory_path = directory / "memory_diagnostics.json"
        if (
            variant is None
            or not timeline_path.exists()
            or not metrics_path.exists()
            or not memory_path.exists()
        ):
            continue
        with np.load(timeline_path) as payload:
            timeline = {key: payload[key].copy() for key in payload.files}
        runs[variant].append(
            {
                "directory": directory,
                "timeline": timeline,
                "metrics": json.loads(metrics_path.read_text()),
                "memory": json.loads(memory_path.read_text()),
            }
        )
    missing = [name for name, values in runs.items() if not values]
    if missing:
        raise ValueError(
            "missing recurring results for variants: " + ", ".join(missing)
        )
    return runs


def _value(run: dict[str, Any], name: str) -> float:
    recurring = run["metrics"]["recurring_mode"]
    counts = run["memory"]["diagnostic_counts"]
    if name == "overall_mse":
        return float(np.mean(run["timeline"]["mse"]))
    if name == "harmful_drift_rate":
        numerator = counts["harmful_drift_count"]
    elif name == "beneficial_evolution_rate":
        numerator = counts["beneficial_evolution_count"]
    elif name == "recovery_attempts":
        return float(counts["recovery_attempt_count"])
    elif name == "capability_rebase_rate":
        numerator = counts["capability_rebase_count"]
    else:
        value = recurring.get(name)
        return float("nan") if value is None else float(value)
    denominator = max(1, counts["matured_records_count"])
    return float(numerator / denominator)


def _write_summary(
    runs: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    metrics = (
        "overall_mse",
        "A1_reference_mse",
        "B_early_mse",
        "B_late_mse",
        "A2_early_mse",
        "A2_late_mse",
        "normalized_recurring_degradation",
        "reacquisition_time",
        "cumulative_excess_error",
        "harmful_drift_rate",
        "beneficial_evolution_rate",
        "recovery_attempts",
        "capability_rebase_rate",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "metric", "mean", "std", "num_seeds"),
        )
        writer.writeheader()
        for variant, variant_runs in runs.items():
            for metric in metrics:
                values = np.asarray(
                    [_value(run, metric) for run in variant_runs],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                writer.writerow(
                    {
                        "variant": _VARIANTS[variant],
                        "metric": metric,
                        "mean": (
                            float(finite.mean()) if finite.size else ""
                        ),
                        "std": (
                            float(finite.std(ddof=0)) if finite.size else ""
                        ),
                        "num_seeds": len(variant_runs),
                    }
                )


def _aligned(
    variant_runs: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(run["timeline"][key]) for run in variant_runs)
    values = np.stack(
        [run["timeline"][key][:length] for run in variant_runs], axis=0
    )
    return values.mean(axis=0), values.std(axis=0)


def _boundaries(run: dict[str, Any]) -> tuple[int, int]:
    phases = run["timeline"]["phase"].astype(str)
    b_start = int(np.flatnonzero(phases == "B")[0])
    a2_start = int(np.flatnonzero(phases == "A2")[0])
    return b_start, a2_start


def _plot(
    runs: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required only for recurring-drift plotting"
        ) from exc

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for variant, variant_runs in runs.items():
        mean, std = _aligned(variant_runs, "rolling_mse")
        origin = np.arange(mean.size)
        axis.plot(origin, mean, label=_VARIANTS[variant])
        axis.fill_between(origin, mean - std, mean + std, alpha=0.18)
    b_start, a2_start = _boundaries(runs["full"][0])
    axis.axvline(b_start, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(a2_start, color="black", linestyle="--", linewidth=0.8)
    axis.text(b_start / 2, 0.98, "A1", transform=axis.get_xaxis_transform())
    axis.text(
        (b_start + a2_start) / 2,
        0.98,
        "B",
        transform=axis.get_xaxis_transform(),
    )
    axis.text(
        (a2_start + len(mean)) / 2,
        0.98,
        "A2",
        transform=axis.get_xaxis_transform(),
    )
    axis.set_xlabel("Online origin")
    axis.set_ylabel("Rolling MSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "recurring_mse.png", dpi=180)
    fig.savefig(output / "recurring_mse.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for variant, variant_runs in runs.items():
        stable_mean, stable_std = _aligned(variant_runs, "stable_size")
        recovery_mean, recovery_std = _aligned(
            variant_runs, "recovery_size"
        )
        origin = np.arange(stable_mean.size)
        axes[0].plot(origin, stable_mean, label=_VARIANTS[variant])
        axes[0].fill_between(
            origin,
            stable_mean - stable_std,
            stable_mean + stable_std,
            alpha=0.18,
        )
        axes[1].plot(origin, recovery_mean, label=_VARIANTS[variant])
        axes[1].fill_between(
            origin,
            recovery_mean - recovery_std,
            recovery_mean + recovery_std,
            alpha=0.18,
        )
    axes[0].set_ylabel("Stable size")
    axes[1].set_ylabel("Recovery size")
    axes[1].set_xlabel("Online origin")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "recurring_memory.png", dpi=180)
    fig.savefig(output / "recurring_memory.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", default="result/recurring_drift_maintext"
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="write summary.csv without importing matplotlib",
    )
    args = parser.parse_args()
    root = Path(args.input_dir)
    output = Path(args.output_dir) if args.output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    _write_summary(runs, output / "summary.csv")
    if not args.skip_plots:
        _plot(runs, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
