import torch

from utils.online_routing import OnlineRoutingCorrection


def _correction(decay: float = 0.0) -> OnlineRoutingCorrection:
    return OnlineRoutingCorrection(
        pred_len=2,
        c_out=1,
        num_experts=2,
        device=torch.device("cpu"),
        correction_lr=0.5,
        correction_decay=decay,
        correction_grad_clip=100.0,
        correction_logit_clip=100.0,
    )


def test_zero_correction_preserves_prior() -> None:
    correction = _correction()
    prior = torch.tensor(
        [[[0.25, 0.75]], [[0.60, 0.40]]], dtype=torch.float32
    )

    effective = correction.effective_weights(prior)

    assert torch.allclose(effective, prior, atol=1e-7)


def test_update_moves_weight_toward_better_expert() -> None:
    correction = _correction()
    prior = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
    before = correction.effective_weights(prior)[0, 0].clone()

    # Mixture is 1, target is 0. Expert 0 predicts 0 (better), expert 1
    # predicts 2 (worse), so centered EG must increase expert 0's logit.
    gradient = correction.update(
        horizon_index=0,
        expert_prediction=torch.tensor([[0.0, 2.0]]),
        mixture_prediction=torch.tensor([1.0]),
        target=torch.tensor([0.0]),
    )
    after = correction.effective_weights(prior)[0, 0]

    assert gradient[0, 0] < 0
    assert gradient[0, 1] > 0
    assert after[0] > before[0]
    assert after[1] < before[1]
    assert torch.allclose(correction.z.mean(dim=-1), torch.zeros(2, 1))


def test_decay_happens_only_once_per_origin() -> None:
    correction = _correction(decay=0.25)
    correction.z.copy_(torch.tensor([[[2.0, -2.0]], [[1.0, -1.0]]]))

    correction.begin_origin(7)
    after_first = correction.z.clone()
    correction.begin_origin(7)
    after_second = correction.z.clone()
    correction.begin_origin(8)

    assert torch.equal(after_first, after_second)
    assert torch.allclose(after_first, torch.tensor([[[1.5, -1.5]], [[0.75, -0.75]]]))
    assert torch.allclose(correction.z, after_first * 0.75)



def test_correction_precedes_top_k_and_can_change_selected_expert() -> None:
    correction = OnlineRoutingCorrection(
        pred_len=1, c_out=1, num_experts=3, device=torch.device("cpu"),
        correction_lr=0.1, correction_decay=0.0,
        correction_grad_clip=10.0, correction_logit_clip=10.0,
    )
    prior = torch.tensor([[[0.80, 0.15, 0.05]]])
    correction.z.copy_(torch.tensor([[[0.0, 0.0, 5.0]]]))

    effective = correction.effective_weights(prior, top_k=1)

    assert effective.argmax(dim=-1).item() == 2
    assert torch.equal((effective > 0).sum(dim=-1), torch.ones(1, 1, dtype=torch.long))
    assert torch.allclose(effective.sum(dim=-1), torch.ones(1, 1))


def test_zero_correction_top_k_matches_prior_ranking() -> None:
    correction = OnlineRoutingCorrection(
        pred_len=1, c_out=1, num_experts=3, device=torch.device("cpu"),
        correction_lr=0.1, correction_decay=0.0,
        correction_grad_clip=10.0, correction_logit_clip=10.0,
    )
    prior = torch.tensor([[[0.80, 0.15, 0.05]]])

    effective = correction.effective_weights(prior, top_k=2)

    assert torch.equal(effective > 0, torch.tensor([[[True, True, False]]]))
    assert torch.allclose(effective.sum(dim=-1), torch.ones(1, 1))
    assert bool((effective >= 0).all())
    assert bool(torch.isfinite(effective).all())
