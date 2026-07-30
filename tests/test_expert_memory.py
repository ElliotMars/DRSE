import torch

from utils.expert_memory import (
    ExpertMemoryManager,
    FixedCapacityExpertBuffer,
    VersionedMemoryItem,
)


def _item(
    sample_id: int,
    responsibility: float,
    alignment: float,
    sketch: torch.Tensor,
    attempts: int = 0,
) -> VersionedMemoryItem:
    return VersionedMemoryItem(
        sample_id=sample_id,
        origin=sample_id,
        expert_id=0,
        x=torch.ones(1, 3, 2) * sample_id,
        x_mark=torch.zeros(1, 3, 7),
        target=torch.zeros(1, 2, 2),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=responsibility,
        last_alignment=alignment,
        stable_credit=responsibility * alignment,
        recovery_credit=responsibility * (1.0 - alignment),
        timestamp=sample_id,
        recovery_attempts=attempts,
        recent_prediction_loss=1.0,
    ).to_cpu_storage(torch.float16)


def _manager(max_attempts: int = 2) -> ExpertMemoryManager:
    return ExpertMemoryManager(
        num_experts=1,
        stable_capacity=2,
        recovery_capacity=2,
        responsibility_threshold=0.3,
        alignment_threshold=0.7,
        duplicate_threshold=0.95,
        failure_penalty=0.5,
        max_recovery_attempts=max_attempts,
        storage_dtype="fp16",
        promote_alignment_threshold=0.85,
        promote_loss_threshold=0.2,
    )


def test_stable_and_recovery_admission_are_mutually_exclusive() -> None:
    manager = _manager()
    stable = _item(1, 0.8, 0.9, torch.tensor([1.0, 0.0]))
    recovery = _item(2, 0.8, 0.4, torch.tensor([0.0, 1.0]))

    assert manager.add_candidate(stable) == "stable"
    assert manager.add_candidate(recovery) == "recovery"
    assert manager.stable_buffers[0].contains(1)
    assert not manager.recovery_buffers[0].contains(1)
    assert manager.recovery_buffers[0].contains(2)
    assert not manager.stable_buffers[0].contains(2)


def test_full_buffer_replaces_only_lower_score() -> None:
    buffer = FixedCapacityExpertBuffer(
        capacity=2,
        kind="stable",
        duplicate_threshold=1.1,
        failure_penalty=0.5,
    )
    buffer.add(_item(1, 0.5, 0.8, torch.tensor([1.0, 0.0, 0.0])))
    buffer.add(_item(2, 0.6, 0.9, torch.tensor([0.0, 1.0, 0.0])))

    assert buffer.add(_item(3, 0.9, 0.9, torch.tensor([0.0, 0.0, 1.0])))
    assert not buffer.contains(1)
    assert buffer.contains(2)
    assert buffer.contains(3)
    assert not buffer.add(_item(4, 0.1, 0.5, torch.tensor([1.0, 1.0, 0.0])))


def test_duplicate_keeps_higher_stable_score() -> None:
    buffer = FixedCapacityExpertBuffer(
        capacity=3,
        kind="stable",
        duplicate_threshold=0.95,
        failure_penalty=0.5,
    )
    buffer.add(_item(1, 0.5, 0.8, torch.tensor([1.0, 0.0])))

    assert buffer.add(_item(2, 0.9, 0.9, torch.tensor([0.99, 0.01])))
    assert not buffer.contains(1)
    assert buffer.contains(2)
    assert not buffer.add(_item(3, 0.2, 0.8, torch.tensor([1.0, 0.0])))
    assert buffer.contains(2)


def test_stable_recovery_migrations() -> None:
    manager = _manager()
    manager.add_candidate(_item(1, 0.9, 0.9, torch.tensor([1.0, 0.0])))

    first = manager.refresh(
        lambda expert_id, item: (0.4, 0.8), timestamp=10
    )
    assert first["stable_to_recovery"] == 1
    assert manager.recovery_buffers[0].contains(1)
    assert not manager.stable_buffers[0].contains(1)

    second = manager.refresh(
        lambda expert_id, item: (0.95, 0.1), timestamp=11
    )
    assert second["recovery_to_stable"] == 1
    assert manager.stable_buffers[0].contains(1)
    assert not manager.recovery_buffers[0].contains(1)


def test_recovery_item_is_evicted_after_attempt_limit() -> None:
    manager = _manager(max_attempts=1)
    manager.add_candidate(_item(1, 0.9, 0.2, torch.tensor([1.0, 0.0])))

    first = manager.refresh(
        lambda expert_id, item: (0.2, 1.0), timestamp=2
    )
    second = manager.refresh(
        lambda expert_id, item: (0.2, 1.0), timestamp=3
    )

    assert first["evicted"] == 0
    assert second["evicted"] == 1
    assert len(manager.recovery_buffers[0]) == 0
