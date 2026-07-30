import torch
import torch.nn as nn

from exp.exp_multi_expert import net


class _FeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor):
        del x_mark
        feature = x[..., : self.hidden_dim].mean(dim=1)
        return x, feature


def _router(granularity: str, top_k: int = 3) -> net:
    model = net.__new__(net)
    nn.Module.__init__(model)
    model.num_experts = 3
    model.top_k = top_k
    model.hidden_dim = 4
    model.c_out = 2
    model.pred_len = 3
    model.router_temperature = 1.0
    model.router_granularity = granularity
    model.feature_encoder = _FeatureEncoder(model.hidden_dim)
    model.router = nn.Linear(model.hidden_dim, model.c_out * model.num_experts)
    if granularity == "horizon_channel":
        model.horizon_head = nn.Linear(
            model.hidden_dim, model.pred_len * model.num_experts
        )
        model.horizon_bias = nn.Parameter(
            torch.zeros(model.pred_len, model.num_experts)
        )
    else:
        model.horizon_head = None
        model.register_parameter("horizon_bias", None)
    return model


def test_horizon_channel_prior_shape_and_top_k() -> None:
    torch.manual_seed(0)
    model = _router("horizon_channel", top_k=2)
    x = torch.randn(5, 7, 4)
    x_mark = torch.zeros(5, 7, 7)

    prior = model._compute_prior(x, x_mark)

    assert prior.shape == (5, 3, 2, 3)
    assert torch.equal((prior > 0).sum(dim=-1), torch.full((5, 3, 2), 2))
    assert torch.allclose(prior.sum(dim=-1), torch.ones(5, 3, 2))


def test_horizon_channel_aggregation_matches_manual_result() -> None:
    model = _router("horizon_channel")
    outputs = torch.arange(1 * 3 * 3 * 2, dtype=torch.float32).reshape(1, 3, 6)
    gates = torch.tensor(
        [
            [
                [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
                [[0.6, 0.1, 0.3], [0.4, 0.4, 0.2]],
                [[0.3, 0.3, 0.4], [0.8, 0.1, 0.1]],
            ]
        ]
    )

    actual = model.aggregate_with_gates(gates, outputs).reshape(1, 3, 2)
    output_ehc = outputs.reshape(1, 3, 3, 2)
    expected = (gates.permute(0, 3, 1, 2) * output_ehc).sum(dim=1)

    assert torch.allclose(actual, expected)


def test_legacy_channel_router_still_runs() -> None:
    torch.manual_seed(1)
    model = _router("channel", top_k=3)
    x = torch.randn(2, 5, 4)
    x_mark = torch.zeros(2, 5, 7)
    outputs = torch.randn(2, 3, 6)

    gates = model._compute_gates(x, x_mark)
    prediction = model.aggregate_with_gates(gates, outputs)

    assert gates.shape == (2, 2, 3)
    assert prediction.shape == (2, 6)
    manual = (
        gates.permute(0, 2, 1).unsqueeze(2)
        * outputs.reshape(2, 3, 3, 2)
    ).sum(dim=1)
    assert torch.allclose(prediction.reshape(2, 3, 2), manual)
