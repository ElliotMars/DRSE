import torch

from utils.subspace_protection import RegressorSubspaceProtector


def _protector(**kwargs) -> RegressorSubspaceProtector:
    defaults = dict(
        num_experts=1,
        feature_dim=4,
        rank=2,
        max_rank=4,
        energy_threshold=0.9,
        min_samples=2,
        eps=1e-8,
        subspace_lambda=100.0,
        gamma_min=0.0,
        gamma_max=1.0,
    )
    defaults.update(kwargs)
    return RegressorSubspaceProtector(**defaults)


def test_covariance_shape_basis_orthogonality_and_fixed_rank() -> None:
    torch.manual_seed(0)
    protector = _protector()
    features = torch.randn(20, 4)
    weights = torch.linspace(0.2, 1.0, 20)

    refreshed = protector.refresh_expert(0, features, weights, 20, step=7)
    state = protector.states[0]

    assert refreshed
    assert protector.last_covariance_shape == (4, 4)
    assert state.effective_rank == 2
    assert torch.allclose(
        state.basis.T @ state.basis, torch.eye(2), atol=1e-5
    )


def test_energy_threshold_selects_smallest_sufficient_rank() -> None:
    protector = _protector(rank=0, max_rank=3, energy_threshold=0.8)
    features = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, -0.2, 0.0],
        ]
    )

    assert protector.refresh_expert(
        0, features, torch.ones(6), sample_count=6, step=1
    )
    assert protector.states[0].effective_rank == 1
    assert protector.states[0].captured_energy > 0.8


def test_projection_scales_parallel_component_and_gamma_extremes() -> None:
    torch.manual_seed(1)
    gradient = torch.randn(3, 4)
    basis = torch.eye(4)[:, :2]

    filtered, parallel, perpendicular = (
        RegressorSubspaceProtector.project_with_basis(
            gradient, basis, gamma=0.25
        )
    )
    filtered_parallel = (filtered @ basis) @ basis.T
    assert torch.allclose(filtered_parallel, 0.25 * parallel, atol=1e-6)
    assert torch.allclose(filtered - filtered_parallel, perpendicular, atol=1e-6)

    hard, _, _ = RegressorSubspaceProtector.project_with_basis(
        gradient, basis, gamma=0.0
    )
    identity, _, _ = RegressorSubspaceProtector.project_with_basis(
        gradient, basis, gamma=1.0
    )
    assert torch.allclose(hard @ basis, torch.zeros(3, 2), atol=1e-6)
    assert torch.allclose(identity, gradient, atol=1e-6)


def test_hard_protection_makes_delta_weight_orthogonal() -> None:
    torch.manual_seed(2)
    gradient = torch.randn(5, 4)
    basis, _ = torch.linalg.qr(torch.randn(4, 2))
    filtered, _, _ = RegressorSubspaceProtector.project_with_basis(
        gradient, basis, gamma=0.0
    )
    delta_weight = -0.01 * filtered

    assert torch.linalg.vector_norm(delta_weight @ basis) < 1e-6


def test_ecl_channel_features_only_build_320_covariance() -> None:
    torch.manual_seed(3)
    protector = RegressorSubspaceProtector(
        num_experts=1,
        feature_dim=320,
        rank=4,
        max_rank=8,
        energy_threshold=0.95,
        min_samples=2,
        eps=1e-8,
        subspace_lambda=100.0,
        gamma_min=0.0,
        gamma_max=1.0,
    )
    channels = 321
    features = torch.randn(2 * channels, 320)
    weights = torch.full((2 * channels,), 1.0 / channels)

    assert protector.refresh_expert(
        0, features, weights, sample_count=2, step=1
    )
    assert protector.last_covariance_shape == (320, 320)
    assert protector.states[0].effective_rank == 4
