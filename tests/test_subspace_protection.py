from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem
from utils.online_diagnostics import OnlineDiagnosticsRecorder
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


def test_identical_features_keep_their_activation_direction() -> None:
    protector = _protector(rank=0, energy_threshold=0.99)
    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    features = direction.repeat(6, 1)

    assert protector.refresh_expert(
        0, features, torch.ones(6), sample_count=6, step=1
    )
    state = protector.states[0]
    normalized = direction / direction.norm()
    assert state.effective_rank >= 1
    assert torch.abs(torch.dot(state.basis[:, 0], normalized)) > 0.999


def test_equal_orthogonal_directions_retain_two_dimensional_energy() -> None:
    protector = _protector(rank=0, energy_threshold=0.99)
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    ).repeat(4, 1)

    assert protector.refresh_expert(
        0, features, torch.ones(8), sample_count=8, step=2
    )
    state = protector.states[0]
    assert state.effective_rank == 2
    assert torch.allclose(state.basis.T @ state.basis, torch.eye(2), atol=1e-6)
    assert abs(state.captured_energy - 1.0) < 1e-6


def test_weighted_second_moment_prioritizes_high_evidence_direction() -> None:
    protector = _protector(rank=1)
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    weights = torch.tensor([9.0, 1.0])

    assert protector.refresh_expert(
        0, features, weights, sample_count=2, step=3
    )
    first = protector.states[0].basis[:, 0]
    assert torch.abs(first[0]) > 0.999
    assert torch.abs(first[1]) < 1e-6


def test_evidence_mass_saturates_and_gamma_is_monotonic() -> None:
    protector = _protector(
        rank=1, min_samples=1, subspace_lambda=100.0,
        evidence_mass_scale=2.0,
    )
    features = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    weights = torch.ones(1)

    assert protector.refresh_expert(
        0, features, weights, sample_count=1, step=0,
        stable_evidence_mass=0.0,
    )
    assert protector.states[0].stable_evidence_mass == 0.0
    assert protector.states[0].protection_mass == 0.0
    assert protector.gamma(0, current_lr=0.01) == 1.0

    assert protector.refresh_expert(
        0, features, weights, sample_count=1, step=1,
        stable_evidence_mass=1.0,
    )
    low_mass = protector.states[0].protection_mass
    low_gamma = protector.gamma(0, current_lr=0.01)
    assert abs(low_mass - 1.0 / 3.0) < 1e-7

    assert protector.refresh_expert(
        0, features, weights, sample_count=1, step=2,
        stable_evidence_mass=8.0,
    )
    high_mass = protector.states[0].protection_mass
    high_gamma = protector.gamma(0, current_lr=0.01)
    metrics = protector.metrics()

    assert high_mass == 0.8
    assert high_mass > low_mass
    assert high_gamma < low_gamma
    assert metrics["subspace_evidence_mass"] == [8.0]
    assert metrics["subspace_protection_mass"] == [0.8]
    assert metrics["subspace_gamma"] == [high_gamma]


def test_failed_small_sample_refresh_updates_strength_and_retains_basis() -> None:
    protector = _protector(
        rank=1, min_samples=4, subspace_lambda=100.0,
        evidence_mass_scale=2.0,
    )
    initial_features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]]
    ).repeat(4, 1)
    assert protector.refresh_expert(
        0, initial_features, torch.ones(4), sample_count=4, step=1,
        stable_evidence_mass=8.0, current_lr=0.01,
    )
    state = protector.states[0]
    old_basis = state.basis.clone()
    old_gamma = state.gamma

    refreshed = protector.refresh_expert(
        0, initial_features[:1], torch.tensor([0.5]),
        sample_count=1, step=2, stable_evidence_mass=0.5,
        current_lr=0.01,
    )

    assert not refreshed
    assert torch.equal(state.basis, old_basis)
    assert state.stable_evidence_mass == 0.5
    assert state.protection_mass == 0.2
    assert abs(state.gamma - (1.0 / 1.2)) < 1e-7
    assert state.gamma > old_gamma
    assert state.gamma != old_gamma


def test_zero_weight_refresh_disables_cached_basis_protection() -> None:
    protector = _protector(
        rank=1, min_samples=2, subspace_lambda=100.0,
        evidence_mass_scale=1.0, gamma_max=0.8,
    )
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    )
    assert protector.refresh_expert(
        0, features, torch.ones(2), sample_count=2, step=1,
        current_lr=0.01,
    )
    state = protector.states[0]
    old_basis = state.basis.clone()

    refreshed = protector.refresh_expert(
        0, features, torch.zeros(2), sample_count=2, step=2,
        current_lr=0.01,
    )

    assert not refreshed
    assert torch.equal(state.basis, old_basis)
    assert state.stable_evidence_mass == 0.0
    assert state.protection_mass == 0.0
    assert state.gamma == 1.0
    gradient = torch.randn(3, 4)
    filtered, stats = protector.filter_gradient(
        0, gradient, current_lr=0.01
    )
    assert torch.equal(filtered, gradient)
    assert stats.gamma == 1.0


def test_eigh_failure_updates_strength_without_corrupting_basis(
    monkeypatch,
) -> None:
    protector = _protector(
        rank=1, min_samples=2, subspace_lambda=100.0,
        evidence_mass_scale=1.0,
    )
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    assert protector.refresh_expert(
        0, features, torch.ones(2), sample_count=2, step=1,
        stable_evidence_mass=4.0, current_lr=0.01,
    )
    state = protector.states[0]
    old_basis = state.basis.clone()
    old_eigenvalues = state.eigenvalues.clone()

    def fail_eigh(covariance):
        del covariance
        raise RuntimeError("synthetic eigendecomposition failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail_eigh)
    refreshed = protector.refresh_expert(
        0, features, torch.tensor([0.25, 0.25]),
        sample_count=2, step=2, stable_evidence_mass=0.5,
        current_lr=0.01,
    )

    assert not refreshed
    assert torch.equal(state.basis, old_basis)
    assert torch.equal(state.eigenvalues, old_eigenvalues)
    assert state.stable_evidence_mass == 0.5
    assert state.protection_mass == 1.0 / 3.0
    assert abs(state.gamma - 0.75) < 1e-7
    scalars = torch.tensor(
        [state.stable_evidence_mass, state.protection_mass, state.gamma]
    )
    assert bool(torch.isfinite(scalars).all())


def test_nonfinite_observation_weights_fail_fast() -> None:
    protector = _protector(min_samples=1)
    features = torch.ones(1, 4)
    try:
        protector.refresh_expert(
            0, features, torch.tensor([float("nan")]),
            sample_count=1, step=1, current_lr=0.01,
        )
    except ValueError as error:
        assert "observation_weights must be finite" in str(error)
    else:
        raise AssertionError("non-finite weights must fail fast")


def test_evidence_mass_scale_must_be_positive_and_finite() -> None:
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        try:
            _protector(evidence_mass_scale=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evidence mass scale must fail")


def test_empty_stable_refresh_deactivates_cached_basis_protection() -> None:
    protector = _protector(
        rank=1,
        min_samples=1,
        subspace_lambda=100.0,
        evidence_mass_scale=1.0,
    )
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    )
    assert protector.refresh_expert(
        0,
        features,
        torch.ones(2),
        sample_count=2,
        step=1,
        stable_evidence_mass=2.0,
    )
    state = protector.states[0]
    cached_basis = state.basis.clone()
    assert state.protection_mass > 0.0
    assert protector.gamma(0, current_lr=0.01) < 1.0

    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.memory_manager = SimpleNamespace(
        stable_buffers=[SimpleNamespace(items=())]
    )
    experiment.subspace_protector = protector
    experiment.diagnostics = SimpleNamespace(update=lambda **kwargs: None)
    experiment._refresh_subspaces(step=2)

    assert state.stable_evidence_mass == 0.0
    assert state.protection_mass == 0.0
    assert state.gamma == 1.0
    assert state.last_refresh_step == 2
    assert state.effective_rank == 1
    assert torch.equal(state.basis, cached_basis)

    gradient = torch.randn(3, 4)
    filtered, stats = protector.filter_gradient(
        0, gradient, current_lr=0.01
    )
    assert torch.equal(filtered, gradient)
    assert stats.gamma == 1.0
    assert stats.rank == 1


def test_memory_refresh_immediately_deactivates_empty_stable_buffer() -> None:
    protector = _protector(
        rank=1,
        min_samples=1,
        subspace_lambda=100.0,
        evidence_mass_scale=1.0,
    )
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    )
    assert protector.refresh_expert(
        0,
        features,
        torch.ones(2),
        sample_count=2,
        step=1,
        stable_evidence_mass=2.0,
    )
    assert protector.gamma(0, current_lr=0.01) < 1.0

    manager = ExpertMemoryManager(
        num_experts=1,
        stable_capacity=1,
        recovery_capacity=1,
        responsibility_threshold=0.0,
        alignment_threshold=0.7,
        duplicate_threshold=0.95,
        failure_penalty=0.5,
        max_recovery_attempts=2,
    )
    item = VersionedMemoryItem(
        sample_id=1,
        origin=1,
        expert_id=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 1, 1),
        prediction_capability_sketch=torch.tensor([1.0, 0.0]),
        normalized_sketch=torch.tensor([1.0, 0.0]),
        sample_responsibility=1.0,
        last_alignment=0.9,
        stable_credit=0.9,
        recovery_credit=0.1,
        timestamp=1,
    )
    assert manager.add_candidate(item) == "stable"

    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.memory_manager = manager
    experiment.subspace_protector = protector
    experiment.diagnostics = OnlineDiagnosticsRecorder(1, interval=1)
    experiment.completed_record_count = 1
    experiment.memory_refresh_interval = 1
    experiment.subspace_refresh_interval = 10
    experiment.online_checker = None
    experiment._refresh_expert_memory = lambda timestamp: manager.refresh(
        lambda expert_id, stored: (0.0, 1.0),
        timestamp=timestamp,
        count_recovery_attempts=False,
    )
    experiment._refresh_subspaces = lambda step: (_ for _ in ()).throw(
        AssertionError("subspace refresh must not run")
    )

    experiment._periodic_memory_and_subspace_refresh(timestamp=2)

    state = protector.states[0]
    assert len(manager.stable_buffers[0]) == 0
    assert manager.recovery_buffers[0].contains(1)
    assert state.stable_evidence_mass == 0.0
    assert state.protection_mass == 0.0
    assert state.gamma == 1.0
    gradient = torch.randn(3, 4)
    filtered, _ = protector.filter_gradient(0, gradient, current_lr=0.01)
    assert torch.equal(filtered, gradient)


class _HeadFeatureExpert(nn.Module):
    def forward(self, x):
        return x

    def extract_head_features(self, x, x_mark):
        del x_mark
        return torch.ones(x.shape[0], 2, 4)


class _CaptureProtector:
    def refresh_expert(self, **kwargs):
        self.kwargs = kwargs
        return True


    def metrics(self):
        return {}


def test_refresh_subspaces_uses_stable_credit_and_splits_channel_weight() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.device = torch.device("cpu")
    experiment.model = nn.Module()
    experiment.model.experts = nn.ModuleList([_HeadFeatureExpert()])
    items = (
        SimpleNamespace(
            x=torch.zeros(1, 2, 2), x_mark=torch.zeros(1, 2, 7),
            stable_credit=0.2,
        ),
        SimpleNamespace(
            x=torch.zeros(1, 2, 2), x_mark=torch.zeros(1, 2, 7),
            stable_credit=0.8,
        ),
    )
    experiment.memory_manager = SimpleNamespace(
        stable_buffers=[SimpleNamespace(items=items)]
    )
    experiment.credit_weighted_subspace = True
    experiment.subspace_protector = _CaptureProtector()
    experiment.diagnostics = SimpleNamespace(update=lambda **kwargs: None)

    experiment._refresh_subspaces(step=3)

    captured = experiment.subspace_protector.kwargs
    assert torch.allclose(
        captured["observation_weights"],
        torch.tensor([0.1, 0.1, 0.4, 0.4]),
    )
    assert captured["stable_evidence_mass"] == 1.0
    assert captured["observation_weights"].sum() == 1.0
