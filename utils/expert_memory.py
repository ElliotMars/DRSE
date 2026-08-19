import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F


MemoryKind = Literal["stable", "recovery"]


def _storage_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError("buffer_storage_dtype must be fp16 or fp32")


@dataclass
class VersionedMemoryItem:
    sample_id: int
    origin: int
    expert_id: int
    x: torch.Tensor
    x_mark: torch.Tensor
    target: torch.Tensor
    prediction_capability_sketch: torch.Tensor
    sample_responsibility: float
    last_alignment: float
    stable_credit: float
    recovery_credit: float
    timestamp: int
    age: int = 0
    recovery_attempts: int = 0
    recent_prediction_loss: float = float("inf")
    normalized_sketch: Optional[torch.Tensor] = None
    sample_confidence: float = 1.0

    def __post_init__(self) -> None:
        self.sample_confidence = float(self.sample_confidence)
        if not math.isfinite(self.sample_confidence):
            raise FloatingPointError("sample_confidence contains NaN or Inf")
        if not 0.0 <= self.sample_confidence <= 1.0:
            raise ValueError("sample_confidence must be in [0,1]")
        self.refresh_credit()

    def refresh_credit(self) -> None:
        """Recompute confidence-aware long-term credit at current alignment."""

        self.stable_credit = (
            self.sample_confidence
            * self.sample_responsibility
            * self.last_alignment
        )
        self.recovery_credit = (
            self.sample_confidence
            * self.sample_responsibility
            * (1.0 - self.last_alignment)
        )
        if not (
            math.isfinite(self.stable_credit)
            and math.isfinite(self.recovery_credit)
        ):
            raise FloatingPointError("memory credit contains NaN or Inf")

    def to_cpu_storage(self, dtype: torch.dtype) -> "VersionedMemoryItem":
        def snapshot(tensor: torch.Tensor) -> torch.Tensor:
            stored = tensor.detach().to(device="cpu").clone()
            return stored.to(dtype=dtype) if stored.is_floating_point() else stored

        self.x = snapshot(self.x)
        self.x_mark = snapshot(self.x_mark)
        self.target = snapshot(self.target)
        self.prediction_capability_sketch = snapshot(
            self.prediction_capability_sketch
        )
        sketch = (
            self.prediction_capability_sketch
            if self.normalized_sketch is None
            else snapshot(self.normalized_sketch)
        )
        self.normalized_sketch = F.normalize(
            sketch.float(), p=2, dim=-1, eps=1e-8
        ).to(dtype=dtype)
        return self

    @property
    def stable_score(self) -> float:
        return self.stable_credit

    def recovery_score(self, failure_penalty: float) -> float:
        return (
            self.recovery_credit
            / (1.0 + failure_penalty * self.recovery_attempts)
        )


class FixedCapacityExpertBuffer:
    def __init__(
        self,
        capacity: int,
        kind: MemoryKind,
        duplicate_threshold: float,
        failure_penalty: float,
    ) -> None:
        if capacity < 0:
            raise ValueError("buffer capacity must be non-negative")
        if kind not in {"stable", "recovery"}:
            raise ValueError("invalid buffer kind")
        self.capacity = int(capacity)
        self.kind = kind
        self.duplicate_threshold = float(duplicate_threshold)
        self.failure_penalty = float(failure_penalty)
        self._items: List[VersionedMemoryItem] = []

    def _score(self, item: VersionedMemoryItem) -> float:
        if self.kind == "stable":
            return item.stable_score
        return item.recovery_score(self.failure_penalty)

    def add(self, item: VersionedMemoryItem) -> bool:
        if self.capacity == 0:
            return False

        same_sample = next(
            (i for i, old in enumerate(self._items) if old.sample_id == item.sample_id),
            None,
        )
        if same_sample is not None:
            if self._score(item) > self._score(self._items[same_sample]):
                self._items[same_sample] = item
                return True
            return False

        if self.kind == "stable" and self._items:
            query = item.normalized_sketch.float()
            similarities = torch.tensor(
                [
                    float(
                        torch.dot(query, old.normalized_sketch.float()).clamp(-1, 1)
                    )
                    for old in self._items
                ]
            )
            duplicate_index = int(similarities.argmax().item())
            if float(similarities[duplicate_index].item()) > self.duplicate_threshold:
                if self._score(item) > self._score(self._items[duplicate_index]):
                    self._items[duplicate_index] = item
                    return True
                return False

        if len(self._items) < self.capacity:
            self._items.append(item)
            return True
        lowest_index = min(
            range(len(self._items)), key=lambda index: self._score(self._items[index])
        )
        if self._score(item) > self._score(self._items[lowest_index]):
            self._items[lowest_index] = item
            return True
        return False

    def remove(self, sample_id: int) -> Optional[VersionedMemoryItem]:
        for index, item in enumerate(self._items):
            if item.sample_id == sample_id:
                return self._items.pop(index)
        return None

    def contains(self, sample_id: int) -> bool:
        return any(item.sample_id == sample_id for item in self._items)

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> Tuple[VersionedMemoryItem, ...]:
        return tuple(self._items)

    def get(self, sample_id: int) -> Optional[VersionedMemoryItem]:
        for item in self._items:
            if item.sample_id == sample_id:
                return item
        return None

    def __len__(self) -> int:
        return len(self._items)


class ExpertMemoryManager:
    def __init__(
        self,
        num_experts: int,
        stable_capacity: int,
        recovery_capacity: int,
        responsibility_threshold: float,
        alignment_threshold: float,
        duplicate_threshold: float,
        failure_penalty: float,
        max_recovery_attempts: int,
        storage_dtype: str = "fp16",
        promote_alignment_threshold: float = 0.9,
        promote_loss_threshold: float = 1.0,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        self.num_experts = int(num_experts)
        self.responsibility_threshold = float(responsibility_threshold)
        self.alignment_threshold = float(alignment_threshold)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.storage_dtype = _storage_dtype(storage_dtype)
        self.promote_alignment_threshold = float(promote_alignment_threshold)
        self.promote_loss_threshold = float(promote_loss_threshold)
        self.stable_buffers = [
            FixedCapacityExpertBuffer(
                stable_capacity, "stable", duplicate_threshold, failure_penalty
            )
            for _ in range(num_experts)
        ]
        self.recovery_buffers = [
            FixedCapacityExpertBuffer(
                recovery_capacity, "recovery", duplicate_threshold, failure_penalty
            )
            for _ in range(num_experts)
        ]

    def add_candidate(self, item: VersionedMemoryItem) -> Optional[MemoryKind]:
        expert_id = int(item.expert_id)
        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        if item.sample_responsibility < self.responsibility_threshold:
            return None
        item.to_cpu_storage(self.storage_dtype)
        kind: MemoryKind = (
            "stable"
            if item.last_alignment >= self.alignment_threshold
            else "recovery"
        )
        target = (
            self.stable_buffers[expert_id]
            if kind == "stable"
            else self.recovery_buffers[expert_id]
        )
        other = (
            self.recovery_buffers[expert_id]
            if kind == "stable"
            else self.stable_buffers[expert_id]
        )
        if target.add(item):
            other.remove(item.sample_id)
            return kind
        return None

    def refresh(
        self,
        evaluator: Callable[
            [int, VersionedMemoryItem], Tuple[float, float]
        ],
        timestamp: int,
        count_recovery_attempts: bool = True,
    ) -> dict[str, int]:
        """Re-evaluate snapshots, then migrate using container snapshots."""

        stats = {
            "stable_to_recovery": 0,
            "recovery_to_stable": 0,
            "recovery_dropped_after_success": 0,
            "recovery_failed": 0,
            "recovery_evicted": 0,
            "recovery_attempt_exhausted": 0,
            "evicted": 0,
        }
        stable_snapshots = [
            (expert_id, item)
            for expert_id, buffer in enumerate(self.stable_buffers)
            for item in buffer.items
        ]
        recovery_snapshots = [
            (expert_id, item)
            for expert_id, buffer in enumerate(self.recovery_buffers)
            for item in buffer.items
        ]

        demotions: List[Tuple[int, VersionedMemoryItem]] = []
        for expert_id, item in stable_snapshots:
            if not self.stable_buffers[expert_id].contains(item.sample_id):
                continue
            alignment, prediction_loss = evaluator(expert_id, item)
            item.last_alignment = float(alignment)
            item.recent_prediction_loss = float(prediction_loss)
            item.age = max(0, int(timestamp) - item.timestamp)
            item.refresh_credit()
            if item.last_alignment < self.alignment_threshold:
                demotions.append((expert_id, item))

        # Remove first, then migrate.  If Recovery rejects the item, it is
        # evicted rather than incorrectly remaining in Stable.
        for expert_id, item in demotions:
            removed = self.stable_buffers[expert_id].remove(item.sample_id)
            if removed is None:
                continue
            self.recovery_buffers[expert_id].remove(item.sample_id)
            if self.recovery_buffers[expert_id].add(removed):
                stats["stable_to_recovery"] += 1
            else:
                stats["evicted"] += 1

        demoted_keys = {
            (expert_id, item.sample_id) for expert_id, item in demotions
        }
        for expert_id, item in recovery_snapshots:
            if (expert_id, item.sample_id) in demoted_keys:
                continue
            if not self.recovery_buffers[expert_id].contains(item.sample_id):
                continue
            alignment, prediction_loss = evaluator(expert_id, item)
            item.last_alignment = float(alignment)
            item.recent_prediction_loss = float(prediction_loss)
            item.age = max(0, int(timestamp) - item.timestamp)
            if count_recovery_attempts:
                item.recovery_attempts += 1
            item.refresh_credit()
            if (
                item.last_alignment >= self.promote_alignment_threshold
                and item.recent_prediction_loss <= self.promote_loss_threshold
            ):
                removed = self.recovery_buffers[expert_id].remove(
                    item.sample_id
                )
                if removed is None:
                    continue
                if self.stable_buffers[expert_id].add(removed):
                    stats["recovery_to_stable"] += 1
                else:
                    stats["recovery_dropped_after_success"] += 1
                    stats["evicted"] += 1
                continue
            if count_recovery_attempts:
                stats["recovery_failed"] += 1
            if (
                count_recovery_attempts
                and item.recovery_attempts >= self.max_recovery_attempts
            ):
                self.recovery_buffers[expert_id].remove(item.sample_id)
                stats["recovery_attempt_exhausted"] += 1
                stats["recovery_evicted"] += 1
                stats["evicted"] += 1
        return stats

    def sample_recovery(
        self,
        expert_id: int,
        batch_size: int,
        excluded_sample_ids: Optional[Set[int]] = None,
    ) -> Tuple[VersionedMemoryItem, ...]:
        """Select high-priority Recovery items without replacement."""

        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        if batch_size <= 0:
            return ()
        excluded = excluded_sample_ids if excluded_sample_ids is not None else set()
        buffer = self.recovery_buffers[expert_id]
        candidates = [
            item for item in buffer.items if item.sample_id not in excluded
        ]
        candidates.sort(
            key=lambda item: item.recovery_score(buffer.failure_penalty),
            reverse=True,
        )
        selected = tuple(candidates[:batch_size])
        excluded.update(item.sample_id for item in selected)
        return selected

    def update_recovery_result(
        self,
        expert_id: int,
        sample_id: int,
        alignment: float,
        prediction_loss: float,
    ) -> Literal[
        "recovery",
        "promoted",
        "dropped",
        "dropped_after_recovery",
        "missing",
    ]:
        """Apply one post-update Recovery result and lifecycle transition."""

        if not 0 <= expert_id < self.num_experts:
            raise IndexError("invalid expert_id")
        buffer = self.recovery_buffers[expert_id]
        item = buffer.get(sample_id)
        if item is None:
            return "missing"
        values = torch.tensor([alignment, prediction_loss], dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("Recovery result contains NaN or Inf")
        item.recovery_attempts += 1
        item.last_alignment = float(alignment)
        item.recent_prediction_loss = float(prediction_loss)
        item.refresh_credit()
        if (
            item.last_alignment >= self.promote_alignment_threshold
            and item.recent_prediction_loss <= self.promote_loss_threshold
        ):
            removed = buffer.remove(sample_id)
            if removed is None:
                return "missing"
            if self.stable_buffers[expert_id].add(removed):
                return "promoted"
            return "dropped_after_recovery"
        if item.recovery_attempts >= self.max_recovery_attempts:
            buffer.remove(sample_id)
            return "dropped"
        return "recovery"

    def buffer_sizes(self) -> tuple[list[int], list[int]]:
        """Return Stable and Recovery sizes for every Expert."""

        return (
            [len(buffer) for buffer in self.stable_buffers],
            [len(buffer) for buffer in self.recovery_buffers],
        )

    def all_items(self) -> Iterable[VersionedMemoryItem]:
        for buffers in (self.stable_buffers, self.recovery_buffers):
            for buffer in buffers:
                yield from buffer.items

    def clear(self) -> None:
        for buffers in (self.stable_buffers, self.recovery_buffers):
            for buffer in buffers:
                buffer.clear()
