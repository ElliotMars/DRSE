import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
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


def test_failed_stable_demotion_is_evicted_not_left_stable() -> None:
    manager = _manager()
    manager.add_candidate(_item(1, 0.9, 0.9, torch.tensor([1.0, 0.0, 0.0])))
    manager.add_candidate(_item(2, 0.9, 0.1, torch.tensor([0.0, 1.0, 0.0])))
    manager.add_candidate(_item(3, 0.8, 0.1, torch.tensor([0.0, 0.0, 1.0])))

    stats = manager.refresh(
        lambda expert_id, item: (0.6, 1.0) if item.sample_id == 1 else (0.1, 1.0),
        timestamp=10,
        count_recovery_attempts=False,
    )

    assert stats["stable_to_recovery"] == 0
    assert stats["evicted"] == 1
    assert not manager.stable_buffers[0].contains(1)
    assert not manager.recovery_buffers[0].contains(1)


def test_successful_stable_demotion_has_no_duplicate_sample_id() -> None:
    manager = _manager()
    manager.add_candidate(_item(1, 0.9, 0.9, torch.tensor([1.0, 0.0])))

    stats = manager.refresh(
        lambda expert_id, item: (0.4, 1.0),
        timestamp=10,
        count_recovery_attempts=False,
    )

    ids = [item.sample_id for item in manager.all_items()]
    assert stats["stable_to_recovery"] == 1
    assert ids.count(1) == 1
    assert not manager.stable_buffers[0].contains(1)
    assert manager.recovery_buffers[0].contains(1)


def test_recovery_replay_deduplicates_sample_across_experts() -> None:
    manager = ExpertMemoryManager(
        num_experts=2, stable_capacity=1, recovery_capacity=2,
        responsibility_threshold=0.0, alignment_threshold=0.7,
        duplicate_threshold=0.95, failure_penalty=0.5,
        max_recovery_attempts=2, storage_dtype="fp16",
    )
    first = _item(7, 0.9, 0.2, torch.tensor([1.0, 0.0]))
    second = _item(7, 0.8, 0.3, torch.tensor([0.0, 1.0]))
    first.expert_id = 0
    second.expert_id = 1
    assert manager.recovery_buffers[0].add(first)
    assert manager.recovery_buffers[1].add(second)
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.recovery_enabled = True
    experiment.recovery_batch_size = 1
    experiment.recovery_loss_weight = 1.0
    experiment.memory_manager = manager
    experiment.model = type("Model", (), {"num_experts": 2})()
    experiment.device = torch.device("cpu")

    batches = experiment._sample_recovery_batches()
    sampled_ids = [
        item.sample_id for batch in batches for item in batch["items"]
    ]

    assert sampled_ids == [7]
