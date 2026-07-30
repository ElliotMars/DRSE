from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from utils.credit_assignment import (
    compute_local_credit,
    compute_sample_credit,
    partial_router_objective,
)
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import (
    ProgressiveFeedbackEvent,
    ProgressiveForecastRecord,
)


class _PartialRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.logits = nn.Parameter(torch.zeros(3, 1, 2))

    def _compute_prior(
        self, x: torch.Tensor, x_mark: torch.Tensor
    ) -> torch.Tensor:
        del x_mark
        return torch.softmax(self.logits, dim=-1).unsqueeze(0).expand(
            x.shape[0], -1, -1, -1
        )


def test_local_responsibility_prefers_lower_error_and_normalizes() -> None:
    prediction = torch.tensor([[0.0, 2.0, 4.0], [3.0, 1.0, -2.0]])
    target = torch.tensor([0.5, 1.0])

    credit = compute_local_credit(prediction, target, temperature=0.5)

    assert torch.allclose(credit.responsibility.sum(dim=-1), torch.ones(2))
    assert credit.responsibility[0, 0] > credit.responsibility[0, 1]
    assert credit.responsibility[1, 1] > credit.responsibility[1, 0]


def test_sample_responsibility_uses_mean_error() -> None:
    first = compute_sample_credit(
        accumulated_loss=torch.tensor([1.0, 3.0]),
        num_matured_values=2,
        matured_horizons=1,
        total_horizons=4,
        temperature=1.0,
    )
    scaled = compute_sample_credit(
        accumulated_loss=torch.tensor([4.0, 12.0]),
        num_matured_values=8,
        matured_horizons=1,
        total_horizons=4,
        temperature=1.0,
    )

    assert torch.allclose(first.mean_loss, scaled.mean_loss)
    assert torch.allclose(first.responsibility, scaled.responsibility)


def test_sample_confidence_scales_with_maturity_at_fixed_entropy() -> None:
    partial = compute_sample_credit(
        accumulated_loss=torch.tensor([0.0, 2.0]),
        num_matured_values=2,
        matured_horizons=1,
        total_horizons=4,
        temperature=1.0,
    )
    mature = compute_sample_credit(
        accumulated_loss=torch.tensor([0.0, 2.0]),
        num_matured_values=2,
        matured_horizons=4,
        total_horizons=4,
        temperature=1.0,
    )

    assert mature.confidence == partial.confidence * 4


def test_uniform_responsibility_has_zero_confidence() -> None:
    credit = compute_sample_credit(
        accumulated_loss=torch.tensor([3.0, 3.0, 3.0, 3.0]),
        num_matured_values=4,
        matured_horizons=2,
        total_horizons=2,
        temperature=1.0,
    )

    assert abs(credit.confidence) < 1e-6


def test_partial_router_objective_touches_only_selected_horizon() -> None:
    logits = torch.nn.Parameter(torch.zeros(3, 1, 2))
    selected_horizon = 1
    prior = torch.softmax(logits[selected_horizon].unsqueeze(0), dim=-1)
    loss, _ = partial_router_objective(
        current_prior=prior,
        correction=torch.zeros_like(prior),
        expert_prediction=torch.tensor([[[0.0, 2.0]]]),
        target=torch.tensor([[0.0]]),
        local_responsibility=torch.tensor([[[0.9, 0.1]]]),
        local_confidence=torch.tensor([[1.0]]),
        local_credit_weight=1.0,
        entropy_weight=0.0,
    )
    loss.backward()

    assert logits.grad[0].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0
    assert logits.grad[2].abs().sum() == 0


def test_sample_responsibility_does_not_modify_z() -> None:
    correction = OnlineRoutingCorrection(
        pred_len=2,
        c_out=1,
        num_experts=2,
        device=torch.device("cpu"),
        correction_lr=0.1,
        correction_decay=0.0,
        correction_grad_clip=10.0,
        correction_logit_clip=5.0,
    )
    correction.z.copy_(torch.tensor([[[0.3, -0.3]], [[0.2, -0.2]]]))
    before = correction.z.clone()

    compute_sample_credit(
        accumulated_loss=torch.tensor([1.0, 2.0]),
        num_matured_values=2,
        matured_horizons=1,
        total_horizons=2,
        temperature=1.0,
    )

    assert torch.equal(correction.z, before)


def test_partial_router_update_changes_only_matured_horizon_and_not_z() -> None:
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.device = torch.device("cpu")
    experiment.model = _PartialRouter()
    experiment.args = SimpleNamespace(pred_len=3, c_out=1)
    experiment.router_params = [experiment.model.logits]
    experiment.router_grad_clip = 10.0
    experiment.router_entropy_weight = 0.0
    experiment.local_credit_weight = 1.0
    experiment.min_credit_eps = 1e-8
    experiment.online_step = 0
    experiment.opt_router = torch.optim.SGD(experiment.router_params, lr=0.5)
    unused_expert_parameter = nn.Parameter(torch.zeros(()))
    experiment.opt_expert = torch.optim.SGD(
        [unused_expert_parameter], lr=0.1
    )
    experiment.routing_correction = OnlineRoutingCorrection(
        pred_len=3,
        c_out=1,
        num_experts=2,
        device=torch.device("cpu"),
        correction_lr=0.1,
        correction_decay=0.0,
        correction_grad_clip=10.0,
        correction_logit_clip=5.0,
    )
    expert_prediction = torch.tensor(
        [[[1.0, 2.0]], [[0.0, 2.0]], [[3.0, 4.0]]]
    )
    prior = torch.full((3, 1, 2), 0.5)
    record = ProgressiveForecastRecord(
        origin=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        expert_predictions=expert_prediction,
        router_prior=prior,
        router_weights=prior,
        mixture_prediction=expert_prediction.mean(dim=-1),
    )
    event = ProgressiveFeedbackEvent(
        record=record,
        horizon_index=1,
        target=torch.tensor([0.0]),
        local_loss=torch.tensor([[0.0, 4.0]]),
        local_responsibility=torch.tensor([[0.9, 0.1]]),
        local_confidence=torch.tensor([1.0]),
    )
    logits_before = experiment.model.logits.detach().clone()
    z_before = experiment.routing_correction.z.clone()

    experiment._update_router_from_partial_feedback([event])

    assert torch.equal(experiment.model.logits[0], logits_before[0])
    assert not torch.equal(experiment.model.logits[1], logits_before[1])
    assert torch.equal(experiment.model.logits[2], logits_before[2])
    assert torch.equal(experiment.routing_correction.z, z_before)

