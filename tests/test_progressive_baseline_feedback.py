from collections import deque
from dataclasses import fields
from types import MethodType, SimpleNamespace
import os
import subprocess

import pytest
import torch
import torch.nn as nn

from exp.exp_dyname import Exp_TS2VecSupervised as ExpDynaME
from exp.exp_stream_baselines import ExpStreamBaseline
from utils.progressive_baseline_feedback import (
    ProgressiveBaselineDiagnostics,
    ProgressiveBaselineEvent,
    ProgressiveBaselineFeedbackManager,
    ProgressiveBaselineRecord,
    progressive_partial_mse,
)
from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    matured_horizon_index,
)
from utils.run_config import build_run_config, validate_online_feedback_protocol


def _record(origin, pred_len=3, method="fsnet"):
    kwargs = {}
    if method == "dyname":
        kwargs.update(
            representation=torch.zeros(1, 4, 1),
            expert_predictions=torch.zeros(1, 2, pred_len, 1),
            blend=0.25,
        )
    return ProgressiveBaselineRecord(
        origin=origin,
        x=torch.full((1, 2, 1), float(origin)),
        x_mark=None if method == "dyname" else torch.zeros(1, 2, 7),
        method=method,
        **kwargs,
    )


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class HorizonModel(nn.Module):
    def __init__(self, pred_len=3, channels=1, calls=None):
        super().__init__()
        self.values = nn.Parameter(torch.zeros(pred_len, channels))
        self.calls = calls if calls is not None else []
        self.store_grad_count = 0

    def forward(self, x, mark=None):
        del mark
        self.calls.append("predict")
        return self.values.unsqueeze(0).expand(x.shape[0], -1, -1) + 0.0

    def store_grad(self):
        self.store_grad_count += 1


class FakeDynaME(nn.Module):
    def __init__(self, pred_len=3, channels=1):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.pred_len = pred_len
        self.channels = channels
        self.forward_count = 0
        self.combine_blends = []
        self.signal_updates = []

    def gate_parameters(self):
        return [self.logit]

    def combine_cached(self, rep, stacked, blend=None):
        del rep
        self.combine_blends.append(float(blend))
        weight = torch.sigmoid(self.logit)
        pred = (1.0 - weight) * stacked[:, 0] + weight * stacked[:, 1]
        return pred, weight, blend

    def forward(self, x, past=None, return_details=False):
        del past
        self.forward_count += 1
        batch = x.shape[0]
        rep = torch.zeros(batch, 4, self.channels, device=x.device)
        first = torch.zeros(
            batch, self.pred_len, self.channels, device=x.device
        )
        second = torch.ones_like(first)
        stacked = torch.stack([first, second], dim=1)
        pred, weight, blend = self.combine_cached(rep, stacked, blend=0.25)
        if return_details:
            return pred, first, weight, blend, rep, stacked
        return pred

    def update_signal(self, mse):
        self.signal_updates.append(float(mse))


def _stream_exp(method, pred_len=3):
    exp = ExpStreamBaseline.__new__(ExpStreamBaseline)
    exp.method = method
    exp.args = SimpleNamespace(features="M", c_out=1, pred_len=pred_len)
    exp.device = torch.device("cpu")
    exp.model = HorizonModel(pred_len)
    exp.opt = CountingSGD(exp.model.parameters())
    exp.progressive_protocol_diagnostics = ProgressiveBaselineDiagnostics(
        pred_len
    )
    return exp


def _dyname_exp(pred_len=3):
    exp = ExpDynaME.__new__(ExpDynaME)
    exp.args = SimpleNamespace(
        features="M",
        c_out=1,
        pred_len=pred_len,
        seq_len=2,
        dyname_past_num=4,
    )
    exp.device = torch.device("cpu")
    exp.model = FakeDynaME(pred_len)
    exp.opt_gate = CountingSGD(exp.model.gate_parameters())
    exp.past = None
    exp.progressive_protocol_diagnostics = ProgressiveBaselineDiagnostics(
        pred_len
    )
    return exp


def test_exact_maturity_matches_pace_schedule():
    assert matured_horizon_index(7, 7, 3) is None
    assert matured_horizon_index(7, 8, 3) == 0
    assert matured_horizon_index(7, 9, 3) == 1
    assert matured_horizon_index(7, 10, 3) == 2
    assert matured_horizon_index(7, 11, 3) is None

    baseline = ProgressiveBaselineFeedbackManager(pred_len=3, c_out=1)
    pace = ProgressiveFeedbackManager(pred_len=3, c_out=1)
    baseline.release(0, torch.tensor([0.0]))
    pace.release(0, torch.tensor([0.0]))
    baseline.add_record(_record(0, pred_len=3))

    released = []
    for origin in (1, 2, 3):
        events = baseline.release(origin, torch.tensor([float(origin)]))
        released.extend(
            (event.record.origin, event.horizon_index, origin)
            for event in events
        )
    assert released == [(0, 0, 1), (0, 1, 2), (0, 2, 3)]


def test_progressive_record_contains_no_future_target_fields():
    record = _record(0, method="dyname")
    names = {item.name for item in fields(record)}
    assert "batch_y" not in names
    assert "target" not in names
    assert "targets" not in names
    assert record.x.device.type == "cpu"
    assert record.representation.device.type == "cpu"
    assert record.expert_predictions.device.type == "cpu"


def test_release_and_update_happen_before_current_prediction():
    order = []

    class LoggingManager(ProgressiveBaselineFeedbackManager):
        def release(self, origin, observation):
            order.append(("release", origin))
            return super().release(origin, observation)

    exp = _stream_exp("fsnet", pred_len=2)
    exp.model.calls = order

    def update(self, events):
        order.append(("update", len(events)))

    exp._progressive_feedback_update = MethodType(update, exp)
    manager = LoggingManager(pred_len=2, c_out=1)
    x = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1)
    y = torch.zeros(2, 2, 1)
    mark = torch.zeros(2, 2, 7)
    exp._progressive_online_batch(manager, x, y, mark)

    assert order == [
        ("release", 0),
        ("update", 0),
        "predict",
        ("release", 1),
        ("update", 1),
        "predict",
    ]


def test_multiple_events_produce_one_optimizer_step():
    exp = _stream_exp("fsnet", pred_len=3)
    events = [
        ProgressiveBaselineEvent(_record(0), 1, torch.tensor([1.0])),
        ProgressiveBaselineEvent(_record(1), 0, torch.tensor([2.0])),
    ]
    exp.progressive_protocol_diagnostics.begin_origin(2, events)
    exp._progressive_feedback_update(events)

    assert exp.opt.step_count == 1
    assert exp.progressive_protocol_diagnostics.optimizer_step_count == 1
    assert (
        exp.progressive_protocol_diagnostics.max_optimizer_steps_per_origin
        == 1
    )


def test_unknown_horizons_are_excluded_from_loss_and_denominator():
    predictions = torch.nn.Parameter(torch.tensor([[[1.0], [2.0], [3.0]]]))
    loss = progressive_partial_mse(
        predictions,
        torch.tensor([1]),
        torch.tensor([[0.0]]),
    )
    loss.backward()
    assert loss.item() == pytest.approx(4.0)
    assert predictions.grad[0, 0, 0].item() == 0.0
    assert predictions.grad[0, 1, 0].item() == pytest.approx(4.0)
    assert predictions.grad[0, 2, 0].item() == 0.0


def test_short_stream_is_not_flushed_at_end():
    manager = ProgressiveBaselineFeedbackManager(pred_len=3, c_out=1)
    manager.release(0, torch.tensor([0.0]))
    manager.add_record(_record(0))
    manager.release(1, torch.tensor([1.0]))
    manager.add_record(_record(1))
    assert len(manager) == 2
    assert manager.pending_records[0].released_horizons.tolist() == [
        True,
        False,
        False,
    ]


def test_horizon_one_matches_one_step_delayed_availability_and_steps():
    x = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    y = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    mark = torch.zeros(3, 1, 7)

    progressive = _stream_exp("fsnet", pred_len=1)
    manager = ProgressiveBaselineFeedbackManager(pred_len=1, c_out=1)
    progressive_pred, _ = progressive._progressive_online_batch(
        manager, x, y, mark
    )

    native = _stream_exp("fsnet", pred_len=1)
    native.student = None
    native.buffer = deque(maxlen=8)
    queue = deque()
    native_pred, _ = native._delayed_online_batch(queue, x, y, mark)

    assert progressive.opt.step_count == native.opt.step_count == 2
    assert len(manager) == len(queue) == 1
    assert torch.allclose(progressive_pred, native_pred)
    assert torch.allclose(progressive.model.values, native.model.values)


@pytest.mark.parametrize("method", ["fsnet", "onenet"])
def test_stream_baseline_progressive_smoke(method):
    exp = _stream_exp(method, pred_len=3)
    manager = ProgressiveBaselineFeedbackManager(pred_len=3, c_out=1)
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    y = torch.zeros(4, 3, 1)
    mark = torch.zeros(4, 2, 7)

    pred, true = exp._progressive_online_batch(manager, x, y, mark)
    exp.progressive_protocol_diagnostics.finish(len(manager))
    summary = exp.progressive_protocol_diagnostics.as_dict()

    assert pred.shape == true.shape == (4, 3, 1)
    assert summary["released_event_count"] == 6
    assert summary["optimizer_step_count"] == 3
    assert summary["max_optimizer_steps_per_origin"] == 1
    assert summary["pending_records_at_end"] == 3
    assert summary["future_target_leakage_detected"] is False


def test_dyname_progressive_smoke_preserves_cached_gate_semantics():
    exp = _dyname_exp(pred_len=3)
    manager = ProgressiveBaselineFeedbackManager(pred_len=3, c_out=1)
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    y = torch.zeros(4, 3, 1)

    pred, true = exp._progressive_online_batch(manager, x, y)
    exp.progressive_protocol_diagnostics.finish(len(manager))
    summary = exp.progressive_protocol_diagnostics.as_dict()

    assert pred.shape == true.shape == (4, 3)
    assert exp.model.forward_count == 4
    assert exp.opt_gate.step_count == 3
    assert len(exp.model.signal_updates) == 3
    assert exp.model.combine_blends == [0.25] * 10
    assert summary["released_event_count"] == 6
    assert summary["optimizer_step_count"] == 3
    assert summary["max_optimizer_steps_per_origin"] == 1
    assert summary["pending_records_at_end"] == 3


@pytest.mark.parametrize("method", ["fsnet", "onenet"])
def test_native_delayed_stream_baseline_behavior_is_unchanged(method):
    exp = _stream_exp(method, pred_len=2)
    exp.student = None
    exp.buffer = deque(maxlen=8)
    queue = deque()
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
    y = torch.zeros(3, 2, 1)
    mark = torch.zeros(3, 2, 7)

    pred, true = exp._delayed_online_batch(queue, x, y, mark)

    assert pred.shape == true.shape == (3, 2, 1)
    assert exp.opt.step_count == 1
    assert len(queue) == 2
    assert queue[0][1].shape == (1, 2, 1)


def test_native_delayed_dyname_behavior_is_unchanged():
    exp = _dyname_exp(pred_len=2)
    queue = deque()
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
    y = torch.zeros(3, 2, 1)

    pred, true = exp._delayed_online_batch(queue, x, y)

    assert pred.shape == true.shape == (3, 2)
    assert exp.opt_gate.step_count == 1
    assert len(exp.model.signal_updates) == 1
    assert len(queue) == 2
    assert queue[0][3].shape == (1, 2, 1)


def test_progressive_protocol_run_config():
    args = SimpleNamespace(
        online_learning="full",
        progressive_fb=False,
        progressive_baseline_fb=True,
        delay_fb=False,
        method="fsnet",
        data="ETTm1",
        pred_len=24,
        seed=2,
        use_gpu=False,
    )
    assert (
        validate_online_feedback_protocol(args)
        == "progressive_baseline_control"
    )
    config = build_run_config(args)
    assert config["feedback_protocol"] == "progressive_baseline_control"
    assert config["progressive_baseline_fb"] is True
    assert config["seed"] == 2


def test_native_run_config_does_not_gain_progressive_control_fields():
    args = SimpleNamespace(
        online_learning="full",
        progressive_fb=False,
        progressive_baseline_fb=False,
        delay_fb=True,
        method="fsnet",
        data="ETTm1",
        pred_len=24,
        seed=0,
        use_gpu=False,
    )
    config = build_run_config(args)
    assert config["causal_feedback_protocol"] == "legacy_delayed"
    assert "feedback_protocol" not in config
    assert "progressive_baseline_fb" not in config


def test_progressive_runner_explicitly_propagates_seed():
    env = os.environ.copy()
    env.update(
        {
            "METHODS": "fsnet",
            "SETTINGS": "ETTm1:1",
            "SEED": "2",
            "EXECUTE": "0",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/run_progressive_baseline_control.sh"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "--seed 2" in result.stdout
    assert "--progressive_baseline_fb" in result.stdout
