"""Causal, deterministic synthetic time series with known concept drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_DRIFT_TYPES = {"abrupt", "gradual", "recurring", "shock"}


@dataclass(frozen=True)
class DriftEvent:
    """One known change, expressed in both timestamps and forecast origins."""

    name: str
    kind: str
    timestamp: int
    end_timestamp: int
    origin: int
    end_origin: int
    from_regime: str
    to_regime: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SyntheticDriftDataset(Dataset):
    """Rolling-origin view over a separately generated synthetic stream.

    At item ``i``, the context ends at timestamp ``i + seq_len - 1`` and the
    first evaluation target is timestamp ``i + seq_len``. Adjacent items move
    by exactly one timestamp, matching the real progressive protocol.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        channels: int,
        total_length: int,
        seed: int = 0,
        noise_std: float = 0.05,
        drift_type: str = "recurring",
        transition_window: int | None = None,
        shock_duration: int | None = None,
        drift_channels: Sequence[int] | None = None,
        label_len: int = 0,
    ) -> None:
        if seq_len <= 0 or pred_len <= 0 or channels <= 0:
            raise ValueError("seq_len, pred_len, and channels must be positive")
        if total_length < seq_len + pred_len + 3:
            raise ValueError("total_length is too short for rolling forecasts")
        if label_len < 0 or label_len > seq_len:
            raise ValueError("label_len must be in [0, seq_len]")
        if not np.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        drift_type = str(drift_type).lower()
        if drift_type not in _DRIFT_TYPES:
            raise ValueError(f"drift_type must be one of {sorted(_DRIFT_TYPES)}")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.total_length = int(total_length)
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        self.drift_type = drift_type
        self.label_len = int(label_len)
        selected = (
            tuple(range(self.channels))
            if drift_channels is None
            else tuple(sorted({int(index) for index in drift_channels}))
        )
        if any(index < 0 or index >= self.channels for index in selected):
            raise ValueError("synthetic drift channel index is out of range")
        self.drift_channels = selected

        alpha, regime_id, events, intervals, shock_mask = self._schedule(
            transition_window=transition_window,
            shock_duration=shock_duration,
        )
        self.transition_alpha = alpha
        self.regime_id = regime_id
        self.events = tuple(events)
        self.intervals = intervals
        self.series = self._generate_series(alpha, shock_mask)
        self.time_marks = self._time_marks(self.total_length)
        self.metadata = {
            "drift_type": self.drift_type,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "channels": self.channels,
            "total_length": self.total_length,
            "seed": self.seed,
            "noise_std": self.noise_std,
            "drift_channels": list(self.drift_channels),
            "events": [event.to_dict() for event in self.events],
            "intervals": {
                name: [int(bounds[0]), int(bounds[1])]
                for name, bounds in self.intervals.items()
            },
            "mechanisms": {
                "A": "positive AR(1), daily sinusoid, weak positive mixing",
                "B": "negative AR(1), faster sinusoid, signed channel mixing",
                "shock": "temporary level and innovation-amplitude shift",
            },
        }

    def _origin(self, timestamp: int) -> int:
        return max(0, int(timestamp) - self.seq_len)

    def _event(
        self,
        name: str,
        kind: str,
        start: int,
        end: int,
        source: str,
        target: str,
    ) -> DriftEvent:
        return DriftEvent(
            name=name,
            kind=kind,
            timestamp=int(start),
            end_timestamp=int(end),
            origin=self._origin(start),
            end_origin=self._origin(end),
            from_regime=source,
            to_regime=target,
        )

    def _schedule(
        self,
        *,
        transition_window: int | None,
        shock_duration: int | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[DriftEvent],
        dict[str, tuple[int, int]],
        np.ndarray,
    ]:
        length = self.total_length
        alpha = np.zeros(length, dtype=np.float64)
        regime = np.zeros(length, dtype=np.int64)
        shock_mask = np.zeros(length, dtype=bool)
        events: list[DriftEvent] = []
        intervals: dict[str, tuple[int, int]] = {}

        if self.drift_type == "abrupt":
            point = length // 2
            alpha[point:] = 1.0
            regime[point:] = 1
            intervals = {"A": (0, point), "B": (point, length)}
            events.append(self._event("A_to_B", "abrupt", point, length, "A", "B"))
        elif self.drift_type == "gradual":
            width = int(transition_window or max(4, length // 6))
            if width < 2 or width >= length // 2:
                raise ValueError("transition_window must be in [2, total_length/2)")
            start = length // 2 - width // 2
            end = start + width
            alpha[start:end] = np.linspace(0.0, 1.0, width, endpoint=True)
            alpha[end:] = 1.0
            regime[alpha >= 0.5] = 1
            intervals = {
                "A": (0, start),
                "transition": (start, end),
                "B": (end, length),
            }
            events.append(self._event("A_to_B", "gradual", start, end, "A", "B"))
        elif self.drift_type == "recurring":
            first_end = length // 3
            second_end = 2 * length // 3
            alpha[first_end:second_end] = 1.0
            regime[first_end:second_end] = 1
            intervals = {
                "first_A": (0, first_end),
                "B": (first_end, second_end),
                "recurring_A": (second_end, length),
            }
            events.extend(
                [
                    self._event(
                        "A_to_B", "recurring", first_end, second_end, "A", "B"
                    ),
                    self._event(
                        "B_to_recurring_A",
                        "recurring",
                        second_end,
                        length,
                        "B",
                        "A",
                    ),
                ]
            )
        else:
            duration = int(shock_duration or max(3, length // 10))
            if duration <= 0 or duration >= length // 2:
                raise ValueError("shock_duration must be in [1, total_length/2)")
            start = length // 2
            end = start + duration
            shock_mask[start:end] = True
            regime[start:end] = 2
            intervals = {
                "pre_shock_A": (0, start),
                "shock": (start, end),
                "recovered_A": (end, length),
            }
            events.extend(
                [
                    self._event(
                        "shock_start", "shock", start, end, "A", "shock"
                    ),
                    self._event(
                        "shock_end", "shock_recovery", end, length, "shock", "A"
                    ),
                ]
            )
        return alpha, regime, events, intervals, shock_mask

    def _generate_series(
        self, alpha: np.ndarray, shock_mask: np.ndarray
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        length, channels = self.total_length, self.channels
        innovation_a = rng.normal(0.0, 0.16, size=(length, channels))
        innovation_b = rng.normal(0.0, 0.16, size=(length, channels))
        state_a = np.zeros((length, channels), dtype=np.float64)
        state_b = np.zeros((length, channels), dtype=np.float64)
        phases = np.linspace(0.0, np.pi / 2.0, channels, endpoint=True)
        time = np.arange(length, dtype=np.float64)
        seasonal_a = np.sin(2.0 * np.pi * time[:, None] / 24.0 + phases)
        seasonal_b = np.sin(2.0 * np.pi * time[:, None] / 9.0 + phases[::-1])
        trend = (time[:, None] / max(1.0, length - 1.0) - 0.5)

        for timestamp in range(length):
            previous_a = state_a[timestamp - 1] if timestamp else 0.0
            previous_b = state_b[timestamp - 1] if timestamp else 0.0
            state_a[timestamp] = (
                0.72 * previous_a
                + 0.32 * seasonal_a[timestamp]
                + 0.08 * trend[timestamp]
                + innovation_a[timestamp]
            )
            state_b[timestamp] = (
                -0.28 * previous_b
                + 0.62 * seasonal_b[timestamp]
                - 0.18 * trend[timestamp]
                + innovation_b[timestamp]
            )

        identity = np.eye(channels)
        neighbor = np.roll(identity, 1, axis=1)
        mixing_a = 0.85 * identity + 0.15 * neighbor
        mixing_b = 0.65 * identity - 0.35 * neighbor
        process_a = state_a @ mixing_a.T
        process_b = state_b @ mixing_b.T
        values = process_a.copy()
        if self.drift_channels:
            indices = np.asarray(self.drift_channels, dtype=np.int64)
            blend = alpha[:, None]
            values[:, indices] = (
                (1.0 - blend) * process_a[:, indices]
                + blend * process_b[:, indices]
            )
            if shock_mask.any():
                shock_noise = rng.normal(
                    0.0,
                    max(0.25, 4.0 * self.noise_std),
                    size=(int(shock_mask.sum()), len(indices)),
                )
                values[np.ix_(shock_mask, indices)] += 2.0 + shock_noise
        values += rng.normal(0.0, self.noise_std, size=values.shape)
        return values.astype(np.float32)

    @staticmethod
    def _time_marks(length: int) -> np.ndarray:
        timestamp = np.arange(length, dtype=np.float64)
        scale = timestamp / max(1.0, length - 1.0)
        marks = np.stack(
            [
                scale,
                np.sin(2.0 * np.pi * timestamp / 24.0),
                np.cos(2.0 * np.pi * timestamp / 24.0),
                np.sin(2.0 * np.pi * timestamp / 168.0),
                np.cos(2.0 * np.pi * timestamp / 168.0),
                np.sin(2.0 * np.pi * timestamp / 12.0),
                np.cos(2.0 * np.pi * timestamp / 12.0),
            ],
            axis=1,
        )
        return marks.astype(np.float32)

    def context_at(self, index: int) -> torch.Tensor:
        self._validate_index(index)
        return torch.from_numpy(self.series[index : index + self.seq_len].copy())

    def context_marks_at(self, index: int) -> torch.Tensor:
        self._validate_index(index)
        return torch.from_numpy(
            self.time_marks[index : index + self.seq_len].copy()
        )

    def observation_at(self, index: int) -> torch.Tensor:
        """Return only the vector observable before forecast origin ``index``."""

        return self.context_at(index)[-1].clone()

    def evaluation_target_at(self, index: int) -> torch.Tensor:
        """Return future truth for evaluation, never for progressive updates."""

        self._validate_index(index)
        start = index + self.seq_len
        return torch.from_numpy(self.series[start : start + self.pred_len].copy())

    def target_regime_at(self, index: int) -> int:
        self._validate_index(index)
        return int(self.regime_id[index + self.seq_len])

    def _validate_index(self, index: int) -> None:
        if not 0 <= int(index) < len(self):
            raise IndexError("synthetic rolling origin is out of range")

    def __getitem__(self, index: int):
        self._validate_index(index)
        x_start = int(index)
        x_end = x_start + self.seq_len
        y_start = x_end - self.label_len
        y_end = x_end + self.pred_len
        return (
            torch.from_numpy(self.series[x_start:x_end].copy()),
            torch.from_numpy(self.series[y_start:y_end].copy()),
            torch.from_numpy(self.time_marks[x_start:x_end].copy()),
            torch.from_numpy(self.time_marks[y_start:y_end].copy()),
        )

    def __len__(self) -> int:
        return self.total_length - self.seq_len - self.pred_len + 1


def parse_drift_channels(value: str | Iterable[int] | None) -> tuple[int, ...] | None:
    """Parse comma-separated CLI channel indices without special casing loaders."""

    if value is None or value == "":
        return None
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(index) for index in value)
