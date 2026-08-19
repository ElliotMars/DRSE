import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.comparator_evaluation import (
    StaticComparatorAccumulator,
    evaluate_dynamic_comparator,
    evaluate_static_comparator,
    fixed_mixture_cumulative_loss,
    save_comparator_diagnostics,
)
from utils.evaluation_diagnostics import compute_router_oracle_diagnostics
from utils.online_diagnostics import OnlineDiagnosticsRecorder


def _origin(target_weights, router_offset=0.0):
    expert_predictions = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float64
    )
    weights = torch.as_tensor(target_weights, dtype=torch.float64)
    target = torch.sum(expert_predictions * weights, dim=-1)
    router_prediction = target + router_offset
    return expert_predictions, router_prediction, target


def _accumulator(num_origins=3, router_offset=0.0):
    accumulator = StaticComparatorAccumulator(num_experts=2)
    for _ in range(num_origins):
        expert_predictions, router_prediction, target = _origin(
            [0.25, 0.75], router_offset=router_offset
        )
        accumulator.update(expert_predictions, router_prediction, target)
    return accumulator


def test_router_matching_best_fixed_comparator_has_zero_regret() -> None:
    result = evaluate_static_comparator(_accumulator())

    assert result.comparator_loss == pytest.approx(0.0, abs=1e-12)
    assert result.router_cumulative_loss == pytest.approx(0.0, abs=1e-12)
    assert result.regret == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        result.comparator_weights, [0.25, 0.75], atol=1e-9
    )


def test_static_comparator_is_no_worse_than_given_fixed_mixture() -> None:
    accumulator = _accumulator(num_origins=4)
    candidate_loss = fixed_mixture_cumulative_loss(
        accumulator, np.asarray([0.5, 0.5])
    )

    result = evaluate_static_comparator(accumulator)

    assert result.comparator_loss <= candidate_loss + 1e-10
    assert result.comparator_loss == pytest.approx(0.0, abs=1e-12)


def test_average_static_regret_is_cumulative_regret_over_origins() -> None:
    result = evaluate_static_comparator(
        _accumulator(num_origins=4, router_offset=1.0)
    )

    assert result.router_cumulative_loss == pytest.approx(4.0)
    assert result.comparator_loss == pytest.approx(0.0, abs=1e-12)
    assert result.regret == pytest.approx(4.0)
    assert result.average_regret == pytest.approx(result.regret / 4)
    values = result.as_dict()
    assert values["average_static_regret"] == pytest.approx(
        values["static_regret"] / values["num_origins"]
    )


def test_oracle_gap_and_static_regret_use_distinct_fields() -> None:
    accumulator = _accumulator()
    comparator_fields = evaluate_static_comparator(accumulator).as_dict()
    expert_predictions, router_prediction, target = _origin([0.25, 0.75])
    oracle_fields = compute_router_oracle_diagnostics(
        expert_predictions, router_prediction, target
    )

    assert "static_regret" in comparator_fields
    assert "gap_to_hard_oracle" not in comparator_fields
    assert "gap_to_top2_oracle" not in comparator_fields
    assert "gap_to_hard_oracle" in oracle_fields
    assert "gap_to_top2_oracle" in oracle_fields
    assert "static_regret" not in oracle_fields


def test_dynamic_comparator_interface_does_not_fabricate_regret() -> None:
    with pytest.raises(NotImplementedError, match="comparator class"):
        evaluate_dynamic_comparator(_accumulator())


def test_comparator_state_is_streaming_and_files_are_plot_ready(tmp_path) -> None:
    accumulator = StaticComparatorAccumulator(num_experts=2)
    initial_nbytes = accumulator.state_nbytes
    for _ in range(100):
        expert_predictions, router_prediction, target = _origin(
            [0.25, 0.75], router_offset=0.5
        )
        accumulator.update(expert_predictions, router_prediction, target)
    assert accumulator.state_nbytes == initial_nbytes
    assert not any(
        isinstance(value, list) for value in accumulator.__dict__.values()
    )

    result = evaluate_static_comparator(accumulator)
    npz_path, json_path = save_comparator_diagnostics(result, str(tmp_path))

    with np.load(npz_path) as arrays:
        assert arrays["router_cumulative_loss"].item() == pytest.approx(25.0)
        assert arrays["static_comparator_loss"].item() == pytest.approx(
            result.comparator_loss
        )
        assert arrays["average_static_regret"].item() == pytest.approx(
            result.average_regret
        )
        assert arrays["static_comparator_weights"].shape == (2,)
    with open(json_path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    assert values["comparator_type"] == "static_fixed_convex_mixture"
    assert values["static_regret"] == pytest.approx(result.regret)


def test_prediction_diagnostics_updates_comparator_without_gradients() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.args = SimpleNamespace(pred_len=1, c_out=2)
    experiment.min_credit_eps = 1e-8
    experiment.specialization_diagnostics = None
    experiment.comparator_diagnostics = StaticComparatorAccumulator(2)
    experiment.routing_correction = SimpleNamespace(z=torch.zeros(1, 2, 2))
    experiment.diagnostics = OnlineDiagnosticsRecorder(2, interval=1)
    expert_predictions, router_prediction, target = _origin([0.25, 0.75])
    expert_predictions.requires_grad_()
    router_prediction.requires_grad_()
    target.requires_grad_()
    prior = torch.full((1, 2, 2), 0.5, dtype=torch.float64)

    experiment._update_prediction_diagnostics(
        true=target.reshape(-1),
        expert_hce=expert_predictions,
        prior_hce=prior,
        weights_hce=prior,
        mixture_hc=router_prediction,
    )

    assert experiment.comparator_diagnostics.num_origins == 1
    assert expert_predictions.grad is None
    assert router_prediction.grad is None
    assert target.grad is None
