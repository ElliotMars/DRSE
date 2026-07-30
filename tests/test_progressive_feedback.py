from dataclasses import fields

import torch

from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)


def _record(origin: int, horizon: int = 3, channels: int = 2, experts: int = 2):
    expert_predictions = torch.arange(
        horizon * channels * experts, dtype=torch.float32
    ).reshape(horizon, channels, experts)
    prior = torch.full((horizon, channels, experts), 1.0 / experts)
    mixture = (expert_predictions * prior).sum(dim=-1)
    return ProgressiveForecastRecord(
        origin=origin,
        x=torch.tensor([[[float(origin), -float(origin)]]]),
        x_mark=torch.zeros(1, 1, 7),
        expert_predictions=expert_predictions,
        router_prior=prior,
        router_weights=prior,
        mixture_prediction=mixture,
    )


def test_record_contains_no_unmatured_future_target() -> None:
    record = _record(origin=0)
    field_names = {item.name for item in fields(record)}

    assert "batch_y" not in field_names
    assert torch.isnan(record.matured_targets).all()
    assert not record.matured_mask.any()
    assert record.x.device.type == "cpu"
    assert record.x_mark.device.type == "cpu"


def test_origin_horizon_mapping_and_exactly_once_completion() -> None:
    manager = ProgressiveFeedbackManager(pred_len=3, c_out=2)
    release_counts = {0: 0, 1: 0}
    completed_origins = []

    manager.release(0, torch.tensor([100.0, 101.0]))
    record0 = _record(origin=0)
    manager.add_record(record0)

    events, completed = manager.release(1, torch.tensor([1.0, 11.0]))
    assert [(event.record.origin, event.horizon_index) for event in events] == [(0, 0)]
    assert completed == []
    release_counts[0] += 1
    assert torch.equal(record0.matured_targets[0], torch.tensor([1.0, 11.0]))
    assert torch.isnan(record0.matured_targets[1:]).all()
    record1 = _record(origin=1)
    manager.add_record(record1)

    events, completed = manager.release(2, torch.tensor([2.0, 12.0]))
    assert [(event.record.origin, event.horizon_index) for event in events] == [
        (0, 1),
        (1, 0),
    ]
    assert completed == []
    for event in events:
        release_counts[event.record.origin] += 1

    events, completed = manager.release(3, torch.tensor([3.0, 13.0]))
    assert [(event.record.origin, event.horizon_index) for event in events] == [
        (0, 2),
        (1, 1),
    ]
    for event in events:
        release_counts[event.record.origin] += 1
    completed_origins.extend(record.origin for record in completed)
    assert completed_origins == [0]
    assert release_counts[0] == 3
    assert record0.num_matured == 3
    assert record0.matured_mask.all()

    events, completed = manager.release(4, torch.tensor([4.0, 14.0]))
    assert [(event.record.origin, event.horizon_index) for event in events] == [(1, 2)]
    release_counts[1] += 1
    completed_origins.extend(record.origin for record in completed)
    assert completed_origins == [0, 1]
    assert release_counts[1] == 3

    events, completed = manager.release(5, torch.tensor([5.0, 15.0]))
    assert events == []
    assert completed == []
    assert completed_origins.count(0) == 1
    assert completed_origins.count(1) == 1


def test_last_incomplete_records_are_not_flushed() -> None:
    manager = ProgressiveFeedbackManager(pred_len=3, c_out=2)
    manager.release(0, torch.zeros(2))
    manager.add_record(_record(origin=0))
    manager.release(1, torch.ones(2))

    assert len(manager) == 1
    assert manager.pending_records[0].num_matured == 1

