#!/usr/bin/env python3
"""Summarize progressive-baseline fairness-control result directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(result_root: Path) -> tuple[list[dict], list[dict]]:
    runs: list[dict] = []
    protocols: list[dict] = []
    for config_path in sorted(result_root.rglob("run_config.json")):
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if (
            config.get("feedback_protocol")
            != "progressive_baseline_control"
            and not config.get("progressive_baseline_fb", False)
        ):
            continue

        metrics_path = config_path.with_name("metrics.npy")
        if not metrics_path.exists():
            continue
        metrics = np.asarray(np.load(metrics_path), dtype=np.float64).reshape(-1)
        if metrics.size < 2:
            continue
        row = {
            "method": config.get("method"),
            "dataset": config.get("data"),
            "pred_len": int(config.get("pred_len")),
            "seed": int(config.get("seed")),
            "mse": float(metrics[1]),
            "mae": float(metrics[0]),
            "result_dir": str(config_path.parent),
        }
        runs.append(row)

        diagnostic_path = config_path.with_name("protocol_diagnostics.json")
        if diagnostic_path.exists():
            with diagnostic_path.open(encoding="utf-8") as handle:
                diagnostic = json.load(handle)
            protocols.append(
                {
                    "method": row["method"],
                    "dataset": row["dataset"],
                    "pred_len": row["pred_len"],
                    "seed": row["seed"],
                    "released_event_count": diagnostic.get(
                        "released_event_count"
                    ),
                    "optimizer_step_count": diagnostic.get(
                        "optimizer_step_count"
                    ),
                    "pending_records_at_end": diagnostic.get(
                        "pending_records_at_end"
                    ),
                    "max_optimizer_steps_per_origin": diagnostic.get(
                        "max_optimizer_steps_per_origin"
                    ),
                    "future_target_leakage_detected": diagnostic.get(
                        "future_target_leakage_detected"
                    ),
                    "result_dir": row["result_dir"],
                }
            )
    return runs, protocols


def add_aggregates(runs: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        groups[(row["method"], row["dataset"], row["pred_len"])].append(row)
    for rows in groups.values():
        mse = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        mae = np.asarray([row["mae"] for row in rows], dtype=np.float64)
        for row in rows:
            row.update(
                {
                    "mse_mean": float(mse.mean()),
                    "mse_std": float(mse.std(ddof=0)),
                    "mae_mean": float(mae.mean()),
                    "mae_std": float(mae.std(ddof=0)),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, default=Path("result"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/progressive_baseline_control"),
    )
    args = parser.parse_args()

    runs, protocols = collect(args.result_root)
    add_aggregates(runs)
    summary_fields = [
        "method",
        "dataset",
        "pred_len",
        "seed",
        "mse",
        "mae",
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "result_dir",
    ]
    protocol_fields = [
        "method",
        "dataset",
        "pred_len",
        "seed",
        "released_event_count",
        "optimizer_step_count",
        "pending_records_at_end",
        "max_optimizer_steps_per_origin",
        "future_target_leakage_detected",
        "result_dir",
    ]
    _write_csv(args.output_dir / "summary.csv", runs, summary_fields)
    _write_csv(
        args.output_dir / "protocol_summary.csv",
        protocols,
        protocol_fields,
    )
    print(
        f"Wrote {len(runs)} runs and {len(protocols)} protocol summaries "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
