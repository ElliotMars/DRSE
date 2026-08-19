import numpy as np
import pytest
import torch

from utils.credit_diagnostic_analysis import analyze_credit_diagnostics
from utils.evaluation_diagnostics import (
    SpecializationDiagnosticsAggregator,
    compute_router_oracle_diagnostics,
)


def _oracle_inputs():
    expert_predictions = torch.tensor(
        [[[0.0, 2.0, 0.0], [0.0, 2.0, 2.0]]]
    )
    mixture_prediction = torch.tensor([[0.5, 1.5]])
    target = torch.tensor([[1.0, 1.0]])
    return expert_predictions, mixture_prediction, target


def test_hard_and_top2_oracles_bound_single_expert_mse() -> None:
    expert_predictions, mixture_prediction, target = _oracle_inputs()

    result = compute_router_oracle_diagnostics(
        expert_predictions, mixture_prediction, target
    )

    individual_mse = (
        (expert_predictions - target.unsqueeze(-1)).pow(2).mean(dim=(0, 1))
    )
    assert result["oracle_hard_mse"] <= float(individual_mse.min()) + 1e-12
    assert all(
        result["oracle_hard_mse"] <= float(value) + 1e-12
        for value in individual_mse
    )
    assert result["oracle_top2_mse"] <= result["oracle_hard_mse"] + 1e-7
    assert result["oracle_top2_pair"] == (0, 1)
    assert result["oracle_top2_alpha"] == pytest.approx(0.5)
    assert result["oracle_top2_mse"] == pytest.approx(0.0, abs=1e-12)
    assert result["router_mse"] == pytest.approx(0.25)


def test_oracle_evaluation_detaches_target_and_predictions() -> None:
    expert_predictions, mixture_prediction, target = _oracle_inputs()
    expert_predictions.requires_grad_()
    mixture_prediction.requires_grad_()
    target.requires_grad_()
    prediction_snapshot = expert_predictions.detach().clone()
    target_snapshot = target.detach().clone()

    result = compute_router_oracle_diagnostics(
        expert_predictions, mixture_prediction, target
    )

    assert result["expert_mse"].requires_grad is False
    assert expert_predictions.grad is None
    assert mixture_prediction.grad is None
    assert target.grad is None
    assert torch.equal(expert_predictions.detach(), prediction_snapshot)
    assert torch.equal(target.detach(), target_snapshot)


def _specialization_inputs():
    weights = torch.tensor(
        [
            [[0.8, 0.2], [0.6, 0.4]],
            [[0.1, 0.9], [0.3, 0.7]],
        ]
    )
    predictions = torch.tensor(
        [
            [[0.0, 2.0], [1.0, 3.0]],
            [[2.0, 0.0], [3.0, 1.0]],
        ]
    )
    target = torch.tensor([[0.0, 2.0], [1.0, 2.0]])
    return weights, predictions, target


def test_specialization_streaming_shapes_values_and_save(tmp_path) -> None:
    aggregator = SpecializationDiagnosticsAggregator(2, 2, 2)
    weights, predictions, target = _specialization_inputs()

    aggregator.update(weights, predictions, target)
    aggregator.update(weights, predictions, target)
    arrays = aggregator.arrays()

    assert arrays["mean_router_weight_by_horizon"].shape == (2, 2)
    assert arrays["mean_router_weight_by_channel"].shape == (2, 2)
    assert arrays["expert_mse_by_horizon"].shape == (2, 2)
    assert arrays["expert_mse_by_channel"].shape == (2, 2)
    assert arrays["winning_expert_rate_by_horizon"].shape == (2, 2)
    assert arrays["winning_expert_rate_by_channel"].shape == (2, 2)
    np.testing.assert_allclose(
        arrays["mean_router_weight_by_horizon"],
        np.asarray([[0.7, 0.3], [0.2, 0.8]]),
    )
    np.testing.assert_allclose(
        arrays["mean_router_weight_by_channel"],
        np.asarray([[0.45, 0.55], [0.45, 0.55]]),
    )
    assert arrays["num_updates"].item() == 2

    output_path = tmp_path / "specialization_diagnostics.npz"
    aggregator.save(str(output_path))
    with np.load(output_path) as saved:
        assert saved["expert_mse_by_horizon"].shape == (2, 2)
        assert saved["expert_mse_by_channel"].shape == (2, 2)


def test_specialization_reset_and_memory_are_constant_in_stream_length() -> None:
    aggregator = SpecializationDiagnosticsAggregator(2, 2, 2)
    weights, predictions, target = _specialization_inputs()
    initial_nbytes = aggregator.state_nbytes
    initial_shapes = {
        name: value.shape
        for name, value in aggregator.__dict__.items()
        if isinstance(value, np.ndarray)
    }

    for _ in range(200):
        aggregator.update(weights, predictions, target)

    assert aggregator.state_nbytes == initial_nbytes
    assert {
        name: value.shape
        for name, value in aggregator.__dict__.items()
        if isinstance(value, np.ndarray)
    } == initial_shapes
    assert not any(
        isinstance(value, list) for value in aggregator.__dict__.values()
    )
    assert aggregator.num_updates == 200

    aggregator.reset()
    assert aggregator.num_updates == 0
    assert all(
        not value.any()
        for value in aggregator.__dict__.values()
        if isinstance(value, np.ndarray)
    )


def test_credit_analysis_reads_toy_npz_and_outputs_correct_bins(tmp_path) -> None:
    input_path = tmp_path / "credit_diagnostics.npz"
    output_path = tmp_path / "credit_analysis.npz"
    np.savez_compressed(
        input_path,
        expert_update_delta=np.asarray([0, 1, 3, 9]),
        js_divergence=np.asarray([0.1, 0.3, 0.5, 0.7]),
        ranking_reversal=np.asarray([0, 1, 1, 0]),
        capability_alignment=np.asarray(
            [[0.9, 0.9], [0.6, 0.8], [0.4, 0.6], [0.2, 0.4]]
        ),
        router_gap=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )

    results = analyze_credit_diagnostics(
        str(input_path),
        str(output_path),
        update_delta_bins=[0, 2, 5, 10],
        alignment_bins=[0, 0.5, 0.75, 1.0],
    )

    assert output_path.exists()
    assert results["update_delta_count"].tolist() == [2, 1, 1]
    np.testing.assert_allclose(
        results["update_delta_mean_js"], [0.2, 0.5, 0.7]
    )
    np.testing.assert_allclose(
        results["update_delta_reversal_rate"], [0.5, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        results["update_delta_mean_alignment"], [0.8, 0.5, 0.3]
    )
    np.testing.assert_allclose(
        results["update_delta_mean_router_gap"], [1.5, 3.0, 4.0]
    )
    assert results["alignment_count"].tolist() == [1, 2, 1]
    np.testing.assert_allclose(
        results["alignment_mean_js"], [0.7, 0.4, 0.1]
    )
    np.testing.assert_allclose(
        results["alignment_reversal_rate"], [0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        results["alignment_mean_router_gap"], [4.0, 2.5, 1.0]
    )

    with np.load(output_path) as saved:
        assert saved["num_records"].item() == 4
        assert saved["update_delta_count"].tolist() == [2, 1, 1]
