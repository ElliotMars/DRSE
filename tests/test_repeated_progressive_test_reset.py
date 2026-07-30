from collections import deque

import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from models.ts2vec.fsnet_ import SamePadConv
from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem
from utils.online_checks import StrictOnlineChecker
from utils.online_diagnostics import OnlineDiagnosticsRecorder
from utils.online_routing import OnlineRoutingCorrection
from utils.subspace_protection import RegressorSubspaceProtector


class _StateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fsnet = SamePadConv(
            2, 2, kernel_size=3, device=torch.device("cpu")
        )
        self.router = nn.Linear(2, 2)


def _assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_restore_recovers_model_fsnet_and_optimizer_state() -> None:
    torch.manual_seed(8)
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.model = _StateModel()
    experiment.opt_expert = torch.optim.Adam(
        experiment.model.fsnet.parameters(), lr=0.01
    )
    experiment.opt_router = torch.optim.Adam(
        experiment.model.router.parameters(), lr=0.01
    )
    experiment._invalidate_test_start_state()

    experiment.opt_expert.zero_grad()
    experiment.opt_router.zero_grad()
    encoded = experiment.model.fsnet(torch.randn(1, 2, 8))
    routed = experiment.model.router(torch.randn(1, 2))
    (encoded.square().mean() + routed.square().mean()).backward()
    experiment.model.fsnet.store_grad()
    experiment.opt_expert.step()
    experiment.opt_router.step()
    experiment._capture_test_start_state()

    expected_model = Exp_TS2VecSupervised._state_to_cpu(
        experiment.model.state_dict()
    )
    expected_optimizers = {
        "expert": Exp_TS2VecSupervised._state_to_cpu(
            experiment.opt_expert.state_dict()
        ),
        "router": Exp_TS2VecSupervised._state_to_cpu(
            experiment.opt_router.state_dict()
        ),
    }
    for parameter in experiment.model.parameters():
        parameter.data.add_(3.0)
    experiment.model.fsnet.grads.fill_(4.0)
    experiment.model.fsnet.f_grads.fill_(5.0)
    experiment.model.fsnet.q_ema.fill_(6.0)
    experiment.model.fsnet.trigger.fill_(True)
    experiment.model.fsnet.W.data.fill_(7.0)
    for optimizer in (experiment.opt_expert, experiment.opt_router):
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.add_(9.0)

    experiment._restore_test_start_state()

    _assert_nested_equal(experiment.model.state_dict(), expected_model)
    _assert_nested_equal(
        experiment.opt_expert.state_dict(), expected_optimizers["expert"]
    )
    _assert_nested_equal(
        experiment.opt_router.state_dict(), expected_optimizers["router"]
    )
    for value in experiment._test_start_model_state.values():
        assert value.device.type == "cpu"


class _ToyOnlineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expert = nn.Linear(1, 1, bias=False)
        self.router = nn.Linear(1, 2, bias=False)


def _memory_item(origin: int) -> VersionedMemoryItem:
    sketch = torch.tensor([1.0, 0.0])
    return VersionedMemoryItem(
        sample_id=origin,
        origin=origin,
        expert_id=0,
        x=torch.zeros(1, 1, 1),
        x_mark=torch.zeros(1, 1, 7),
        target=torch.zeros(1, 1, 1),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=1.0,
        last_alignment=1.0,
        stable_credit=1.0,
        recovery_credit=0.0,
        timestamp=origin,
    )


def _runtime_experiment() -> Exp_TS2VecSupervised:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.model = _ToyOnlineModel()
    experiment.opt_expert = torch.optim.Adam(
        experiment.model.expert.parameters(), lr=0.02
    )
    experiment.opt_router = torch.optim.Adam(
        experiment.model.router.parameters(), lr=0.01
    )
    experiment.online_buffer = deque(maxlen=2)
    experiment.online_step = 0
    experiment.prev_online_mse = None
    experiment.online_mse_ema = None
    experiment.fallback_count = 0
    experiment.fallback_channel_count = 0
    experiment.progressive_origin = 0
    experiment.credit_diagnostics = deque(maxlen=4)
    experiment.credit_diagnostic_total_count = 0
    experiment.expert_update_count = 0
    experiment.completed_record_count = 0
    experiment.last_expert_update_diagnostics = {}
    experiment.routing_correction = OnlineRoutingCorrection(
        pred_len=1,
        c_out=1,
        num_experts=2,
        device=torch.device("cpu"),
        correction_lr=0.1,
        correction_decay=0.0,
        correction_grad_clip=10.0,
        correction_logit_clip=5.0,
    )
    experiment.memory_manager = ExpertMemoryManager(
        num_experts=2,
        stable_capacity=8,
        recovery_capacity=1,
        responsibility_threshold=0.0,
        alignment_threshold=0.5,
        duplicate_threshold=1.1,
        failure_penalty=0.5,
        max_recovery_attempts=1,
    )
    experiment.subspace_protector = RegressorSubspaceProtector(
        num_experts=2,
        feature_dim=1,
        rank=1,
        max_rank=1,
        energy_threshold=0.9,
        min_samples=1,
    )
    experiment.diagnostics = OnlineDiagnosticsRecorder(2, interval=1)
    experiment.online_checker = StrictOnlineChecker(False)
    experiment._invalidate_test_start_state()
    return experiment


def _run_deterministic_stream(experiment) -> dict[str, object]:
    predictions = []
    router_weights = []
    for origin, value in enumerate((0.2, -0.4, 0.7, 0.1)):
        x = torch.tensor([[value]])
        target = torch.tensor([value * 0.25])
        experiment.routing_correction.begin_origin(origin)
        prior = torch.softmax(experiment.model.router(x), dim=-1).reshape(
            1, 1, 2
        )
        effective = experiment.routing_correction.effective_weights(prior)
        base = experiment.model.expert(x).reshape(1)
        expert_predictions = torch.stack([base, -base], dim=-1)
        prediction = (effective[0] * expert_predictions).sum()
        predictions.append(prediction.detach().clone())
        router_weights.append(effective.detach().clone())

        experiment.opt_expert.zero_grad()
        experiment.opt_router.zero_grad()
        (prediction - target).pow(2).backward()
        experiment.opt_expert.step()
        experiment.opt_router.step()
        experiment.routing_correction.update(
            0,
            expert_predictions.detach(),
            prediction.detach().reshape(1),
            target,
        )
        experiment.memory_manager.add_candidate(_memory_item(origin))

    return {
        "predictions": torch.stack(predictions),
        "z": experiment.routing_correction.z.clone(),
        "router_weights": torch.stack(router_weights),
        "expert_parameter": experiment.model.expert.weight.detach().clone(),
        "memory_sizes": experiment.memory_manager.buffer_sizes(),
    }


def test_same_exp_repeated_stream_is_deterministic_after_restore() -> None:
    torch.manual_seed(12)
    experiment = _runtime_experiment()
    experiment._prepare_test_start_state()
    experiment._reset_progressive_online_state(None)
    first = _run_deterministic_stream(experiment)

    experiment._prepare_test_start_state()
    experiment._reset_progressive_online_state(None)
    second = _run_deterministic_stream(experiment)

    assert torch.equal(first["predictions"], second["predictions"])
    assert torch.equal(first["z"], second["z"])
    assert torch.equal(first["router_weights"], second["router_weights"])
    assert torch.equal(first["expert_parameter"], second["expert_parameter"])
    assert first["memory_sizes"] == second["memory_sizes"]
