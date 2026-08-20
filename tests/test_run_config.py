import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils import run_config
from utils.run_config import (
    build_run_config,
    save_run_config,
    validate_online_feedback_protocol,
)


def _args(**overrides):
    values = {
        "online_learning": "full",
        "progressive_fb": True,
        "delay_fb": False,
        "method": "onenet_fsnet",
        "data": "ETTh2",
        "data_path": Path("ETTh2.csv"),
        "seq_len": 60,
        "pred_len": 24,
        "features": "M",
        "num_experts": 4,
        "expert_composition": "mixed",
        "router_granularity": "horizon_channel",
        "top_k": np.int64(2),
        "router_temperature": 2.0,
        "disable_online_correction": False,
        "correction_lr": 0.1,
        "correction_decay": 0.01,
        "pretrain_mode": "load",
        "expert_update_strategy": "hybrid",
        "online_lr_expert": 1e-4,
        "online_lr_router": 1e-5,
        "adaptive_controller": "dynamic",
        "disable_tsb": False,
        "disable_tsb_smoothing": False,
        "disable_tsb_conflict_filter": False,
        "tsb_alpha": 0.5,
        "stable_buffer_size": 32,
        "recovery_buffer_size": 32,
        "responsibility_threshold": 0.3,
        "credit_top_k": 1,
        "buffer_duplicate_threshold": 0.98,
        "recovery_failure_penalty": 0.5,
        "max_recovery_attempts": 3,
        "recovery_batch_size": 2,
        "recovery_loss_weight": 0.1,
        "recovery_sketch_weight": 1.0,
        "disable_recovery": False,
        "capability_sketch_dim": 32,
        "capability_sketch_seed": 2025,
        "subspace_scope": "regressor",
        "subspace_rank": 16,
        "subspace_energy_threshold": 0.95,
        "subspace_max_rank": 32,
        "subspace_lambda": 1e4,
        "subspace_evidence_mass_scale": 8.0,
        "subspace_gamma_min": 0.0,
        "subspace_gamma_max": 1.0,
        "disable_credit_weighted_subspace": False,
        "credit_diagnostic_buffer_size": 10000,
        "dynamic_comparator": True,
        "dynamic_comparator_max_switches": 2,
        "dynamic_comparator_max_points": 64,
        "seed": 7,
        "finetune_model_seed": 8,
        "itr": 2,
        "gpu": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_noncausal_online_protocol_fails_fast() -> None:
    args = _args(progressive_fb=False, delay_fb=False)
    with pytest.raises(ValueError, match="Full-window immediate online"):
        validate_online_feedback_protocol(args)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"progressive_fb": True, "delay_fb": False}, "progressive"),
        ({"progressive_fb": False, "delay_fb": True}, "legacy_delayed"),
        (
            {
                "online_learning": "none",
                "progressive_fb": False,
                "delay_fb": False,
            },
            "none_offline",
        ),
    ],
)
def test_causal_and_offline_protocols_pass(overrides, expected) -> None:
    args = _args(**overrides)
    assert validate_online_feedback_protocol(args) == expected
    config = build_run_config(args)
    assert config["causal_feedback_protocol"] == expected
    if expected == "none_offline":
        assert config["online_correction_enabled"] is False
        assert config["tsb_enabled"] is False
        assert config["recovery_enabled"] is False


def test_metadata_builder_preserves_paper_critical_values() -> None:
    config = build_run_config(_args(), iteration_index=1)

    assert config["causal_feedback_protocol"] == "progressive"
    assert config["online_learning"] == "full"
    assert config["pretrain_mode"] == "load"
    assert config["expert_composition"] == "mixed"
    assert config["adaptive_controller"] == "dynamic"
    assert config["effective_expert_update_strategy"] == "hybrid"
    assert config["tsb_enabled"] is True
    assert config["credit_weighted_subspace"] is True
    assert config["subspace_rank_mode"] == "fixed"
    assert config["device"] == "cpu"
    assert config["top_k"] == 2
    assert isinstance(config["top_k"], int)
    assert config["data_path"] == "ETTh2.csv"
    assert config["iteration_index"] == 1
    assert "git_commit" in config
    assert "git_dirty" in config


def test_run_config_json_round_trip_preserves_scalar_types(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_config,
        "read_git_metadata",
        lambda: {"git_commit": None, "git_dirty": None},
    )
    path = save_run_config(_args(), tmp_path, iteration_index=0)
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert Path(path) == tmp_path / "run_config.json"
    assert loaded["progressive_fb"] is True
    assert loaded["pred_len"] == 24
    assert isinstance(loaded["pred_len"], int)
    assert loaded["online_lr_expert"] == 1e-4
    assert isinstance(loaded["online_lr_expert"], float)
    assert loaded["method"] == "onenet_fsnet"
    assert loaded["git_commit"] is None


def test_git_metadata_failure_is_nonfatal(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        del args, kwargs
        raise OSError("git unavailable")

    monkeypatch.setattr(run_config.subprocess, "run", fail_run)
    assert run_config.read_git_metadata() == {
        "git_commit": None,
        "git_dirty": None,
    }
