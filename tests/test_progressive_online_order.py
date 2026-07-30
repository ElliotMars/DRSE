from types import MethodType, SimpleNamespace

import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)


def test_expert_update_strategy_call_order() -> None:
    expected = {
        "plain": ["raw"],
        "tsb": ["tsb"],
        "subspace": ["raw", "subspace"],
        "hybrid": ["tsb", "subspace"],
    }
    for strategy, expected_order in expected.items():
        experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
        experiment.expert_update_strategy = strategy
        calls = []

        experiment._assign_raw_expert_gradients = MethodType(
            lambda self, current: calls.append("raw"), experiment
        )
        experiment._apply_tsb_gradient_filter = MethodType(
            lambda self, current, reference, alpha: (
                calls.append("tsb") or {"tsb_conflict_rate": 0.0}
            ),
            experiment,
        )
        experiment._apply_subspace_gradient_filter = MethodType(
            lambda self, current_lr: (
                calls.append("subspace")
                or {
                    "parallel_gradient_norm": [],
                    "perpendicular_gradient_norm": [],
                    "subspace_gamma": [],
                }
            ),
            experiment,
        )
        experiment.model = SimpleNamespace(num_experts=0)

        experiment._apply_expert_update_strategy([], None, 0.5, 1e-3)
        assert calls == expected_order


def test_completed_sample_learns_before_memory_commit() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.completed_record_count = 0
    calls = []
    record = SimpleNamespace(
        matured_targets=torch.tensor([[1.0], [2.0]]),
        x=torch.zeros(1, 3, 1),
        x_mark=torch.zeros(1, 3, 7),
    )

    experiment._evaluate_completed_record = MethodType(
        lambda self, completed, origin: (
            calls.append("evaluate")
            or (
                torch.ones(1),
                torch.zeros(1),
                torch.ones(1),
                torch.zeros(1),
            )
        ),
        experiment,
    )
    experiment._build_memory_candidates = MethodType(
        lambda self, completed, alignment, prediction_loss, timestamp: (
            calls.append("classify") or ["candidate"]
        ),
        experiment,
    )

    def learn(self, dataset, x, target, x_mark):
        assert "commit" not in calls
        assert torch.equal(target[0], record.matured_targets)
        calls.append("learn")

    experiment._ol_one_batch = MethodType(learn, experiment)
    experiment._commit_memory_candidates = MethodType(
        lambda self, candidates: calls.append("commit"), experiment
    )
    experiment._periodic_memory_and_subspace_refresh = MethodType(
        lambda self, timestamp: calls.append("refresh"), experiment
    )

    experiment._learn_then_commit_completed_record(None, record, origin=2)
    assert calls == ["evaluate", "classify", "learn", "commit", "refresh"]


def test_progressive_release_never_exposes_unmatured_future() -> None:
    horizon, channels, experts = 2, 1, 2
    predictions = torch.zeros(horizon, channels, experts)
    weights = torch.full_like(predictions, 0.5)
    record = ProgressiveForecastRecord(
        origin=0,
        x=torch.zeros(1, 3, channels),
        x_mark=torch.zeros(1, 3, 7),
        expert_predictions=predictions,
        router_prior=weights,
        router_weights=weights,
        mixture_prediction=torch.zeros(horizon, channels),
    )
    manager = ProgressiveFeedbackManager(pred_len=horizon, c_out=channels)
    manager.release(0, torch.tensor([0.0]))
    manager.add_record(record)

    events, completed = manager.release(1, torch.tensor([4.0]))

    assert len(events) == 1
    assert completed == []
    assert torch.equal(record.matured_targets[0], torch.tensor([4.0]))
    assert torch.isnan(record.matured_targets[1])
