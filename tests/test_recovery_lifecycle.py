import torch

from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem


def _manager(max_attempts: int = 2) -> ExpertMemoryManager:
    return ExpertMemoryManager(
        num_experts=1,
        stable_capacity=2,
        recovery_capacity=2,
        responsibility_threshold=0.0,
        alignment_threshold=0.7,
        duplicate_threshold=0.95,
        failure_penalty=0.5,
        max_recovery_attempts=max_attempts,
        promote_alignment_threshold=0.85,
        promote_loss_threshold=0.2,
    )


def _item(
    sample_id: int,
    responsibility: float,
    alignment: float,
    sketch: torch.Tensor,
) -> VersionedMemoryItem:
    return VersionedMemoryItem(
        sample_id=sample_id,
        origin=sample_id,
        expert_id=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 2, 1),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=responsibility,
        last_alignment=alignment,
        stable_credit=responsibility * alignment,
        recovery_credit=responsibility * (1.0 - alignment),
        timestamp=sample_id,
    ).to_cpu_storage(torch.float16)


def test_recovery_success_moves_to_stable_when_accepted() -> None:
    manager = _manager()
    item = _item(20, 0.8, 0.2, torch.tensor([0.0, 0.0, 1.0]))
    assert manager.add_candidate(item) == "recovery"

    status = manager.update_recovery_result(0, 20, 0.95, 0.1)

    assert status == "promoted"
    assert manager.stable_buffers[0].contains(20)
    assert not manager.recovery_buffers[0].contains(20)
    assert [item.sample_id for item in manager.all_items()].count(20) == 1


def test_recovery_success_is_discarded_when_stable_rejects() -> None:
    manager = _manager()
    manager.add_candidate(
        _item(21, 1.0, 0.99, torch.tensor([1.0, 0.0, 0.0]))
    )
    manager.add_candidate(
        _item(22, 1.0, 0.98, torch.tensor([0.0, 1.0, 0.0]))
    )
    recovering = _item(23, 0.8, 0.2, torch.tensor([0.0, 0.0, 1.0]))
    assert manager.add_candidate(recovering) == "recovery"

    status = manager.update_recovery_result(0, 23, 0.9, 0.1)

    assert status == "dropped_after_recovery"
    assert not manager.stable_buffers[0].contains(23)
    assert not manager.recovery_buffers[0].contains(23)
    assert manager.sample_recovery(0, batch_size=2) == ()


def test_unresolved_recovery_stays_until_attempt_limit() -> None:
    manager = _manager(max_attempts=2)
    item = _item(24, 0.8, 0.2, torch.tensor([1.0, 0.0]))
    assert manager.add_candidate(item) == "recovery"

    status = manager.update_recovery_result(0, 24, 0.5, 0.8)

    assert status == "recovery"
    assert manager.recovery_buffers[0].contains(24)
    assert manager.recovery_buffers[0].get(24).recovery_attempts == 1
    assert not manager.stable_buffers[0].contains(24)

    second_status = manager.update_recovery_result(0, 24, 0.5, 0.8)
    assert second_status == "dropped"
    assert not manager.recovery_buffers[0].contains(24)


def test_refresh_promotes_or_discards_successful_recovery() -> None:
    accepted = _manager()
    accepted.add_candidate(
        _item(30, 0.8, 0.2, torch.tensor([0.0, 0.0, 1.0]))
    )
    accepted_stats = accepted.refresh(
        lambda expert_id, item: (0.95, 0.1),
        timestamp=31,
        count_recovery_attempts=False,
    )
    assert accepted_stats["recovery_to_stable"] == 1
    assert accepted.stable_buffers[0].contains(30)
    assert not accepted.recovery_buffers[0].contains(30)

    rejected = _manager()
    rejected.add_candidate(
        _item(31, 1.0, 0.99, torch.tensor([1.0, 0.0, 0.0]))
    )
    rejected.add_candidate(
        _item(32, 1.0, 0.98, torch.tensor([0.0, 1.0, 0.0]))
    )
    rejected.add_candidate(
        _item(33, 0.8, 0.2, torch.tensor([0.0, 0.0, 1.0]))
    )
    rejected_stats = rejected.refresh(
        lambda expert_id, item: (
            (0.9, 0.1)
            if item.sample_id == 33
            else (item.last_alignment, 1.0)
        ),
        timestamp=34,
        count_recovery_attempts=False,
    )
    assert rejected_stats["recovery_to_stable"] == 0
    assert rejected_stats["recovery_dropped_after_success"] == 1
    assert rejected_stats["recovery_evicted"] == 0
    assert rejected_stats["recovery_attempt_exhausted"] == 0
    assert rejected_stats["evicted"] == 1
    assert not rejected.stable_buffers[0].contains(33)
    assert not rejected.recovery_buffers[0].contains(33)
