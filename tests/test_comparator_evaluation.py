import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.comparator_evaluation import (
    StaticComparatorAccumulator,
    KSwitchComparatorAccumulator,
    evaluate_horizon_channel_static_comparator,
    evaluate_k_switch_dynamic_comparator,
    evaluate_path_length_comparator,
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
    expert_predictions, router_prediction, target = _origin(
        [0.25, 0.75], router_offset=0.5
    )
    accumulator.update(expert_predictions, router_prediction, target)
    initial_nbytes = accumulator.state_nbytes
    for _ in range(99):
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
    assert values["comparator_type"] == "global_static_convex_comparator"
    assert values["static_regret"] == pytest.approx(result.regret)

def test_horizon_channel_static_comparator_recovers_known_weights(tmp_path) -> None:
    accumulator = StaticComparatorAccumulator(2, horizon=2, channels=2)
    expert_predictions = torch.tensor(
        [
            [[0.0, 2.0], [1.0, 3.0]],
            [[2.0, 0.0], [4.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    known_weights = torch.tensor(
        [
            [[0.2, 0.8], [0.7, 0.3]],
            [[0.4, 0.6], [0.9, 0.1]],
        ],
        dtype=torch.float64,
    )
    target = (expert_predictions * known_weights).sum(dim=-1)
    uniform_prediction = expert_predictions.mean(dim=-1)
    prediction_snapshot = expert_predictions.clone()
    for _ in range(5):
        accumulator.update(expert_predictions, uniform_prediction, target)

    result = evaluate_horizon_channel_static_comparator(accumulator)
    uniform_loss = 5.0 * float(
        (uniform_prediction - target).pow(2).mean().item()
    )

    assert result.comparator_weights.shape == (2, 2, 2)
    np.testing.assert_allclose(
        result.comparator_weights, known_weights.numpy(), atol=1e-7
    )
    assert result.comparator_loss == pytest.approx(0.0, abs=1e-10)
    assert result.comparator_loss <= uniform_loss + 1e-10
    assert torch.equal(expert_predictions, prediction_snapshot)
    assert expert_predictions.grad is None

    global_result = evaluate_static_comparator(accumulator)
    npz_path, json_path = save_comparator_diagnostics(
        global_result, str(tmp_path), hc_result=result
    )
    with np.load(npz_path) as arrays:
        assert arrays["hc_static_comparator_weights"].shape == (2, 2, 2)
        assert "hc_static_comparator_loss" in arrays
        assert "hc_static_regret" in arrays
        assert "hc_average_static_regret" in arrays
    with open(json_path, "r", encoding="utf-8") as handle:
        fields = json.load(handle)
    assert fields["hc_comparator_type"] == (
        "horizon_channel_static_convex_comparator"
    )


def _switch_accumulators(num_origins=8, max_points=16, router_matches=False):
    dynamic = KSwitchComparatorAccumulator(2, max_points=max_points)
    static = StaticComparatorAccumulator(2)
    for origin in range(num_origins):
        weights = [0.2, 0.8] if origin < num_origins // 2 else [0.8, 0.2]
        expert_predictions = torch.tensor(
            [[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float64
        )
        target = (
            expert_predictions
            * torch.tensor(weights, dtype=torch.float64)
        ).sum(dim=-1)
        router = target.clone() if router_matches else expert_predictions.mean(dim=-1)
        dynamic.update(expert_predictions, router, target)
        static.update(expert_predictions, router, target)
    return dynamic, static


def test_k_zero_matches_global_and_more_switches_do_not_increase_loss(
    tmp_path,
) -> None:
    dynamic, static = _switch_accumulators()
    global_result = evaluate_static_comparator(static)
    zero_switch = evaluate_k_switch_dynamic_comparator(dynamic, max_switches=0)
    one_switch = evaluate_k_switch_dynamic_comparator(dynamic, max_switches=1)
    two_switches = evaluate_k_switch_dynamic_comparator(dynamic, max_switches=2)

    assert zero_switch.comparator_loss == pytest.approx(
        global_result.comparator_loss, abs=1e-9
    )
    assert one_switch.comparator_loss <= zero_switch.comparator_loss + 1e-10
    assert two_switches.comparator_loss <= one_switch.comparator_loss + 1e-10
    assert one_switch.switch_points == (4,)
    assert one_switch.segment_weights.shape == (2, 2)

    npz_path, json_path = save_comparator_diagnostics(
        global_result, str(tmp_path), dynamic_result=one_switch
    )
    with np.load(npz_path) as arrays:
        assert arrays["dynamic_comparator_loss"].item() == pytest.approx(0.0)
        assert arrays["switch_points"].tolist() == [4]
        assert arrays["segment_weights"].shape == (2, 2)
    with open(json_path, "r", encoding="utf-8") as handle:
        fields = json.load(handle)
    assert fields["dynamic_comparator_type"] == (
        "empirical_k_switch_convex_comparator"
    )


def test_router_matching_k_switch_comparator_has_zero_dynamic_regret() -> None:
    dynamic, _ = _switch_accumulators(router_matches=True)

    result = evaluate_k_switch_dynamic_comparator(dynamic, max_switches=1)

    assert result.comparator_loss == pytest.approx(0.0, abs=1e-10)
    assert result.router_cumulative_loss == pytest.approx(0.0, abs=1e-12)
    assert result.regret == pytest.approx(0.0, abs=1e-10)


def test_dynamic_max_points_bounds_quadratic_problem_size() -> None:
    dynamic, _ = _switch_accumulators(num_origins=80, max_points=7)

    result = evaluate_k_switch_dynamic_comparator(dynamic, max_switches=3)

    assert len(dynamic.blocks) <= 7
    assert result.num_comparator_points <= 7
    assert sum(block.statistics.num_origins for block in dynamic.blocks) == 80


def test_oracle_dynamic_and_static_fields_remain_strictly_distinct() -> None:
    dynamic, static = _switch_accumulators()
    dynamic_fields = evaluate_k_switch_dynamic_comparator(
        dynamic, max_switches=1
    ).as_dict()
    static_fields = evaluate_static_comparator(static).as_dict()
    expert_predictions, router_prediction, target = _origin([0.25, 0.75])
    oracle_fields = compute_router_oracle_diagnostics(
        expert_predictions, router_prediction, target
    )

    assert "dynamic_regret" in dynamic_fields
    assert "global_static_regret" in static_fields
    assert "gap_to_all_expert_oracle" in oracle_fields
    assert "gap_to_all_expert_oracle" not in dynamic_fields
    assert "dynamic_regret" not in oracle_fields


def test_path_length_interface_remains_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="path-length"):
        evaluate_path_length_comparator()


def test_prediction_diagnostics_updates_comparator_without_gradients() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.args = SimpleNamespace(pred_len=1, c_out=2)
    experiment.min_credit_eps = 1e-8
    experiment.specialization_diagnostics = None
    experiment.comparator_diagnostics = StaticComparatorAccumulator(2)
    experiment.dynamic_comparator_diagnostics = KSwitchComparatorAccumulator(
        2, max_points=4
    )
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
    assert experiment.dynamic_comparator_diagnostics.num_origins == 1
    assert expert_predictions.grad is None
    assert router_prediction.grad is None
    assert target.grad is None
