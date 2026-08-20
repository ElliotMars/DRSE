from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from exp.exp_multi_expert import Exp_TS2VecSupervised
from main import prepare_experiment_for_run


class _CallTrackingExperiment:
    def __init__(self) -> None:
        self.calls = []

    def train(self, setting):
        self.calls.append(("train", setting))

    def load_pretrained(self, checkpoint):
        self.calls.append(("load", checkpoint))

    def prepare_without_pretraining(self):
        self.calls.append(("none", None))


def _args(mode: str, checkpoint: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        pretrain_mode=mode,
        pretrained_checkpoint=checkpoint,
    )


def test_none_skips_offline_train_and_checkpoint_load() -> None:
    experiment = _CallTrackingExperiment()

    prepare_experiment_for_run(experiment, _args("none"), "setting")

    assert experiment.calls == [("none", None)]


def test_retrain_and_load_dispatch_are_unchanged() -> None:
    retrain = _CallTrackingExperiment()
    loaded = _CallTrackingExperiment()

    prepare_experiment_for_run(retrain, _args("retrain"), "setting")
    prepare_experiment_for_run(loaded, _args("load", "/tmp/model.pth"), "setting")

    assert retrain.calls == [("train", "setting")]
    assert loaded.calls == [("load", "/tmp/model.pth")]
    with pytest.raises(ValueError, match="pretrained_checkpoint"):
        prepare_experiment_for_run(
            _CallTrackingExperiment(), _args("load"), "setting"
        )


class _RandomInitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.Linear(2, 2)
        self.router = nn.Linear(2, 2)
        self.router_mode = False

    def set_router_mode(self, enabled: bool) -> None:
        self.router_mode = bool(enabled)


def test_none_snapshot_restores_the_same_random_initial_state() -> None:
    torch.manual_seed(37)
    experiment = Exp_TS2VecSupervised.__new__(Exp_TS2VecSupervised)
    experiment.model = _RandomInitModel()
    experiment.expert_params = list(experiment.model.experts.parameters())
    experiment.router_params = list(experiment.model.router.parameters())
    experiment.args = SimpleNamespace(
        learning_rate_expert=1e-3,
        learning_rate_router=2e-3,
        weight_decay=0.0,
    )
    experiment.base_learning_rate_expert = 1e-4
    experiment.base_learning_rate_router = 1e-5
    experiment._invalidate_test_start_state()

    experiment.prepare_without_pretraining()
    initial = {
        name: value.detach().clone()
        for name, value in experiment.model.state_dict().items()
    }
    experiment._prepare_test_start_state()
    for parameter in experiment.model.parameters():
        parameter.data.add_(5.0)
    experiment._prepare_test_start_state()

    assert experiment.model.router_mode is True
    assert all(
        torch.equal(value, initial[name])
        for name, value in experiment.model.state_dict().items()
    )
