from types import SimpleNamespace

import torch
import torch.nn as nn

import exp.exp_multi_expert as multi_expert


class _TinyExpert(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        del device
        self.output_dim = args.pred_len * args.c_out
        self.representation = nn.Parameter(torch.randn(320))

    def forward(self, x, x_mark, return_repr=False):
        del x_mark
        batch = x.shape[0]
        output = torch.zeros(batch, self.output_dim, device=x.device)
        representation = self.representation.to(x.device).expand(batch, -1)
        if return_repr:
            return output, representation
        return output

    def store_grad(self) -> None:
        pass


def _args(seed: int = 17) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=2,
        top_k=2,
        c_out=2,
        pred_len=3,
        enc_in=2,
        dropout=0.0,
        router_temperature=1.0,
        router_granularity="horizon_channel",
        capability_sketch_dim=16,
        capability_sketch_seed=seed,
    )


def _model(monkeypatch, seed: int = 17):
    monkeypatch.setattr(multi_expert, "ExpertNet", _TinyExpert)
    monkeypatch.setattr(multi_expert, "FSNetTimeExpertNet", _TinyExpert)
    return multi_expert.net(_args(seed), torch.device("cpu"))


def test_projection_is_fixed_and_saved_in_state_dict(monkeypatch) -> None:
    first = _model(monkeypatch, seed=23)
    second = _model(monkeypatch, seed=23)
    different = _model(monkeypatch, seed=24)

    assert torch.equal(first.capability_projection, second.capability_projection)
    assert not torch.equal(first.capability_projection, different.capability_projection)
    assert "capability_projection" in first.state_dict()

    second.load_state_dict(first.state_dict())
    assert torch.equal(first.capability_projection, second.capability_projection)


def test_alignment_is_one_for_same_state_and_drops_after_expert_change(
    monkeypatch,
) -> None:
    model = _model(monkeypatch)
    x = torch.randn(1, 5, 2)
    x_mark = torch.zeros(1, 5, 7)
    _, representation_before = model.forward_experts(
        x, x_mark, return_repr=True
    )
    sketch_before = model.compute_capability_sketch(representation_before)
    same_alignment = ((sketch_before * sketch_before).sum(dim=-1) + 1.0) / 2.0

    with torch.no_grad():
        model.experts[0].representation.mul_(-1)
    _, representation_after = model.forward_experts(
        x, x_mark, return_repr=True
    )
    sketch_after = model.compute_capability_sketch(representation_after)
    changed_alignment = (
        (sketch_before * sketch_after).sum(dim=-1).clamp(-1, 1) + 1.0
    ) / 2.0

    assert torch.allclose(same_alignment, torch.ones_like(same_alignment), atol=1e-6)
    assert changed_alignment[0, 0] < 0.1
    assert changed_alignment[0, 1] > 0.99

