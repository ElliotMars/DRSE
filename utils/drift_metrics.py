"""Metrics and lightweight memory timelines for known synthetic drift events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_TRANSITION_KEYS = (
    "stable_to_recovery",
    "recovery_to_stable",
    "recovery_dropped_after_success",
    "recovery_attempt_exhausted",
)


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    converter = getattr(event, "to_dict", None)
    if converter is None:
        raise TypeError("drift events must be mappings or expose to_dict()")
    return dict(converter())


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def recovery_time(
    mse_timeline: Sequence[float],
    *,
    drift_origin: int,
    pre_drift_error: float,
    recovery_window: int,
    recovery_tolerance: float,
    search_end: int | None = None,
) -> int | None:
    """Return the first offset whose W-origin mean reaches baseline tolerance."""

    values = np.asarray(mse_timeline, dtype=np.float64).reshape(-1)
    if recovery_window <= 0:
        raise ValueError("recovery_window must be positive")
    if recovery_tolerance < 0.0 or not np.isfinite(recovery_tolerance):
        raise ValueError("recovery_tolerance must be finite and non-negative")
    if not np.isfinite(pre_drift_error) or pre_drift_error < 0.0:
        raise ValueError("pre_drift_error must be finite and non-negative")
    start = max(0, int(drift_origin))
    end = len(values) if search_end is None else min(len(values), int(search_end))
    threshold = float(pre_drift_error) * (1.0 + float(recovery_tolerance))
    last_start = end - int(recovery_window)
    for candidate in range(start, last_start + 1):
        window = values[candidate : candidate + recovery_window]
        if np.isfinite(window).all() and float(window.mean()) <= threshold:
            return candidate - start
    return None


def compute_drift_metrics(
    mse_timeline: Sequence[float],
    events: Sequence[Any],
    *,
    pre_window: int = 16,
    early_window: int = 8,
    recovery_window: int = 4,
    recovery_tolerance: float = 0.2,
    recurring_intervals: Mapping[str, Sequence[int]] | None = None,
    seq_len: int = 0,
) -> dict[str, Any]:
    """Compute interpretable event and recurring-mode recovery statistics."""

    values = np.asarray(mse_timeline, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("mse_timeline must contain finite non-negative values")
    if pre_window <= 0 or early_window <= 0:
        raise ValueError("pre_window and early_window must be positive")
    event_dicts = [_event_dict(event) for event in events]
    event_results: list[dict[str, Any]] = []
    for index, event in enumerate(event_dicts):
        start = min(len(values), max(0, int(event["origin"])))
        next_start = (
            min(len(values), max(start, int(event_dicts[index + 1]["origin"])))
            if index + 1 < len(event_dicts)
            else len(values)
        )
        baseline = _mean_or_none(values[max(0, start - pre_window) : start])
        early = _mean_or_none(values[start : min(next_start, start + early_window)])
        segment = values[start:next_start]
        peak = float(segment.max()) if segment.size else None
        if baseline is None:
            excess = None
            recovered = None
        else:
            excess = float(np.maximum(segment - baseline, 0.0).sum())
            recovered = recovery_time(
                values,
                drift_origin=start,
                pre_drift_error=baseline,
                recovery_window=recovery_window,
                recovery_tolerance=recovery_tolerance,
                search_end=next_start,
            )
        event_results.append(
            {
                "name": event.get("name", f"event_{index}"),
                "kind": event.get("kind"),
                "origin": start,
                "pre_drift_error": baseline,
                "early_post_drift_error": early,
                "peak_error": peak,
                "recovery_time": recovered,
                "cumulative_excess_error": excess,
            }
        )

    result: dict[str, Any] = {
        "definition": {
            "pre_window": int(pre_window),
            "early_window": int(early_window),
            "recovery_window": int(recovery_window),
            "recovery_tolerance": float(recovery_tolerance),
            "no_recovery_sentinel": None,
        },
        "events": event_results,
    }
    required = {"first_A", "B", "recurring_A"}
    if recurring_intervals is not None and required.issubset(recurring_intervals):
        origin_intervals = {
            name: (
                max(0, int(bounds[0]) - int(seq_len)),
                max(0, int(bounds[1]) - int(seq_len)),
            )
            for name, bounds in recurring_intervals.items()
        }
        first_start, first_end = origin_intervals["first_A"]
        recurring_start, recurring_end = origin_intervals["recurring_A"]
        first_end = min(first_end, len(values))
        recurring_start = min(recurring_start, len(values))
        recurring_end = min(recurring_end, len(values))
        first_error = _mean_or_none(values[first_start:first_end])
        recurring_early = _mean_or_none(
            values[recurring_start : min(recurring_end, recurring_start + early_window)]
        )
        reacquisition = (
            recovery_time(
                values,
                drift_origin=recurring_start,
                pre_drift_error=first_error,
                recovery_window=recovery_window,
                recovery_tolerance=recovery_tolerance,
                search_end=recurring_end,
            )
            if first_error is not None
            else None
        )
        recovered_error = None
        if reacquisition is not None:
            recovered_start = recurring_start + reacquisition
            recovered_error = _mean_or_none(
                values[recovered_start : recovered_start + recovery_window]
            )
        retention_ratio = (
            recurring_early / first_error
            if first_error is not None
            and first_error > 0.0
            and recurring_early is not None
            else None
        )
        result["recurring_mode"] = {
            "first_A_error": first_error,
            "recurring_A_early_error": recurring_early,
            "recurring_A_recovered_error": recovered_error,
            "reacquisition_time": reacquisition,
            "old_mode_retention_error_ratio": retention_ratio,
            "old_mode_retention_definition": (
                "recurring_A_early_error / first_A_error; 1 is perfect retention"
            ),
        }
    return result


class DriftMetricTracker:
    """Resettable holder for an online MSE timeline."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mse: list[float] = []

    def update(self, mse: float) -> None:
        value = float(mse)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("MSE must be finite and non-negative")
        self.mse.append(value)


class MemoryTimelineRecorder:
    """Record small occupancy arrays and aggregate lifecycle transitions."""

    def __init__(self, num_experts: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = int(num_experts)
        self.reset()

    def reset(self) -> None:
        self.steps: list[int] = []
        self.stable_occupancy: list[list[int]] = []
        self.recovery_occupancy: list[list[int]] = []
        self.mean_recovery_attempts: list[float] = []
        self.stable_regime_occupancy: list[dict[int, int]] = []
        self.recovery_regime_occupancy: list[dict[int, int]] = []
        self.transitions = {key: 0 for key in _TRANSITION_KEYS}

    def record(
        self,
        step: int,
        memory_manager: Any,
        *,
        transition_stats: Mapping[str, int] | None = None,
        sample_regimes: Mapping[int, int] | None = None,
    ) -> None:
        stable_sizes, recovery_sizes = memory_manager.buffer_sizes()
        if len(stable_sizes) != self.num_experts:
            raise ValueError("memory manager Expert count mismatch")
        self.steps.append(int(step))
        self.stable_occupancy.append([int(value) for value in stable_sizes])
        self.recovery_occupancy.append([int(value) for value in recovery_sizes])
        recovery_items = [
            item
            for buffer in memory_manager.recovery_buffers
            for item in buffer.items
        ]
        self.mean_recovery_attempts.append(
            float(np.mean([item.recovery_attempts for item in recovery_items]))
            if recovery_items
            else 0.0
        )
        sample_regimes = sample_regimes or {}
        stable_counts: dict[int, int] = {}
        recovery_counts: dict[int, int] = {}
        for buffers, destination in (
            (memory_manager.stable_buffers, stable_counts),
            (memory_manager.recovery_buffers, recovery_counts),
        ):
            for buffer in buffers:
                for item in buffer.items:
                    regime = int(sample_regimes.get(item.sample_id, -1))
                    destination[regime] = destination.get(regime, 0) + 1
        self.stable_regime_occupancy.append(stable_counts)
        self.recovery_regime_occupancy.append(recovery_counts)
        if transition_stats is not None:
            for key in _TRANSITION_KEYS:
                self.transitions[key] += int(transition_stats.get(key, 0))

    def arrays(self) -> dict[str, np.ndarray]:
        regime_labels = sorted(
            {
                regime
                for timeline in (
                    self.stable_regime_occupancy,
                    self.recovery_regime_occupancy,
                )
                for counts in timeline
                for regime in counts
            }
        )

        def regime_array(records: list[dict[int, int]]) -> np.ndarray:
            return np.asarray(
                [
                    [counts.get(regime, 0) for regime in regime_labels]
                    for counts in records
                ],
                dtype=np.int64,
            ).reshape(len(records), len(regime_labels))

        return {
            "step": np.asarray(self.steps, dtype=np.int64),
            "stable_occupancy": np.asarray(
                self.stable_occupancy, dtype=np.int64
            ).reshape(-1, self.num_experts),
            "recovery_occupancy": np.asarray(
                self.recovery_occupancy, dtype=np.int64
            ).reshape(-1, self.num_experts),
            "mean_recovery_attempts": np.asarray(
                self.mean_recovery_attempts, dtype=np.float64
            ),
            "regime_labels": np.asarray(regime_labels, dtype=np.int64),
            "stable_occupancy_by_regime": regime_array(
                self.stable_regime_occupancy
            ),
            "recovery_occupancy_by_regime": regime_array(
                self.recovery_regime_occupancy
            ),
        }

    def summary(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "num_steps": len(self.steps),
            "transitions": dict(self.transitions),
            "mean_recovery_attempts_definition": (
                "mean attempts among current Recovery items, averaged over origins"
            ),
            "mean_recovery_attempts": (
                float(arrays["mean_recovery_attempts"].mean())
                if self.steps
                else 0.0
            ),
            "final_stable_occupancy": (
                arrays["stable_occupancy"][-1].tolist() if self.steps else []
            ),
            "final_recovery_occupancy": (
                arrays["recovery_occupancy"][-1].tolist() if self.steps else []
            ),
            "stable_regime_occupancy_over_time": self.stable_regime_occupancy,
            "recovery_regime_occupancy_over_time": self.recovery_regime_occupancy,
        }
