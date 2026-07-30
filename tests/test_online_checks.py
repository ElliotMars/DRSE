from types import SimpleNamespace

import pytest
import torch

from utils.online_checks import StrictOnlineChecker


def test_strict_checker_accepts_valid_online_state_and_order() -> None:
    checker = StrictOnlineChecker(True)
    prior = torch.tensor([[[0.7, 0.3]]])
    checker.prediction_bundle(
        prediction=torch.tensor([[1.0]]),
        prior=prior,
        effective=prior,
        correction=torch.zeros(1, 1, 2),
        origin=3,
    )
    checker.begin_completed(record_origin=1, origin=3)
    checker.expert_updated(record_origin=1, origin=3)
    checker.before_memory_commit(record_origin=1, origin=3)


def test_strict_checker_reports_context_for_future_target_and_bad_order() -> None:
    checker = StrictOnlineChecker(True)
    record = SimpleNamespace(
        num_matured=0,
        matured_mask=torch.zeros(2, dtype=torch.bool),
        matured_targets=torch.tensor([[float("nan")], [2.0]]),
        metadata={},
    )

    with pytest.raises(RuntimeError, match=r"origin=4"):
        checker.new_record(record, origin=4)
    with pytest.raises(RuntimeError, match="before Expert update"):
        checker.before_memory_commit(record_origin=2, origin=5)
