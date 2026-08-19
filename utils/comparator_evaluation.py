"""Empirical static-comparator evaluation for progressive routing."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ComparatorStatistics:
    """Streaming sufficient statistics for cumulative mixture MSE."""

    gram: np.ndarray
    expert_target_cross: np.ndarray
    target_square_sum: float
    router_cumulative_loss: float
    num_origins: int


@dataclass(frozen=True)
class ComparatorResult:
    """Common result interface for empirical comparator evaluations."""

    comparator_type: str
    router_cumulative_loss: float
    comparator_loss: float
    regret: float
    average_regret: float
    num_origins: int
    comparator_weights: np.ndarray | None
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "comparator_type": self.comparator_type,
            "router_cumulative_loss": self.router_cumulative_loss,
            "comparator_loss": self.comparator_loss,
            "regret": self.regret,
            "average_regret": self.average_regret,
            "num_origins": self.num_origins,
            "comparator_weights": (
                None
                if self.comparator_weights is None
                else self.comparator_weights.tolist()
            ),
            "converged": self.converged,
            "iterations": self.iterations,
        }
        if self.comparator_type == "static_fixed_convex_mixture":
            result.update(
                {
                    "static_comparator_loss": self.comparator_loss,
                    "static_regret": self.regret,
                    "average_static_regret": self.average_regret,
                    "static_comparator_weights": result["comparator_weights"],
                }
            )
        return result


class StaticComparatorAccumulator:
    """Accumulate O(E^2) statistics without retaining per-origin tensors."""

    def __init__(self, num_experts: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = int(num_experts)
        self.reset()

    def reset(self) -> None:
        self.gram = np.zeros(
            (self.num_experts, self.num_experts), dtype=np.float64
        )
        self.expert_target_cross = np.zeros(
            self.num_experts, dtype=np.float64
        )
        self.target_square_sum = 0.0
        self.router_cumulative_loss = 0.0
        self.num_origins = 0

    @property
    def state_nbytes(self) -> int:
        return int(self.gram.nbytes + self.expert_target_cross.nbytes)

    @torch.no_grad()
    def update(
        self,
        expert_predictions: torch.Tensor,
        router_prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if expert_predictions.ndim != 3:
            raise ValueError("expert_predictions must have shape [H,C,E]")
        horizon, channels, experts = expert_predictions.shape
        if experts != self.num_experts:
            raise ValueError("Expert dimension does not match accumulator")
        expected_hc = (horizon, channels)
        if tuple(router_prediction.shape) != expected_hc:
            raise ValueError("router_prediction must have shape [H,C]")
        if tuple(target.shape) != expected_hc:
            raise ValueError("target must have shape [H,C]")

        predictions = (
            expert_predictions.detach().double().cpu().numpy().reshape(-1, experts)
        )
        router = router_prediction.detach().double().cpu().numpy().reshape(-1)
        target_value = target.detach().double().cpu().numpy().reshape(-1)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Expert predictions contain NaN or Inf")
        if not np.isfinite(router).all():
            raise FloatingPointError("Router prediction contains NaN or Inf")
        if not np.isfinite(target_value).all():
            raise FloatingPointError("evaluation target contains NaN or Inf")

        num_values = target_value.size
        if num_values <= 0:
            raise ValueError("an origin must contain at least one target value")
        scale = 1.0 / num_values
        self.gram += scale * (predictions.T @ predictions)
        self.expert_target_cross += scale * (predictions.T @ target_value)
        self.target_square_sum += scale * float(target_value @ target_value)
        self.router_cumulative_loss += float(
            np.mean((router - target_value) ** 2)
        )
        self.num_origins += 1

    def statistics(self) -> ComparatorStatistics:
        return ComparatorStatistics(
            gram=self.gram.copy(),
            expert_target_cross=self.expert_target_cross.copy(),
            target_square_sum=float(self.target_square_sum),
            router_cumulative_loss=float(self.router_cumulative_loss),
            num_origins=int(self.num_origins),
        )


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    candidates = sorted_values - (cumulative - 1.0) / (
        np.arange(vector.size) + 1.0
    )
    positive = np.flatnonzero(candidates > 0)
    if positive.size == 0:
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    rho = int(positive[-1])
    threshold = (cumulative[rho] - 1.0) / (rho + 1.0)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / projected.sum()


def _validated_statistics(
    source: ComparatorStatistics | StaticComparatorAccumulator,
) -> ComparatorStatistics:
    statistics = (
        source.statistics()
        if isinstance(source, StaticComparatorAccumulator)
        else source
    )
    gram = np.asarray(statistics.gram, dtype=np.float64)
    cross = np.asarray(statistics.expert_target_cross, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must have shape [E,E]")
    if cross.shape != (gram.shape[0],):
        raise ValueError("expert_target_cross must have shape [E]")
    scalars = np.asarray(
        [
            statistics.target_square_sum,
            statistics.router_cumulative_loss,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise FloatingPointError("comparator statistics contain NaN or Inf")
    if not np.isfinite(scalars).all():
        raise FloatingPointError("comparator scalar statistics contain NaN or Inf")
    if statistics.num_origins < 0:
        raise ValueError("num_origins must be non-negative")
    return ComparatorStatistics(
        gram=(gram + gram.T) / 2.0,
        expert_target_cross=cross,
        target_square_sum=float(statistics.target_square_sum),
        router_cumulative_loss=float(statistics.router_cumulative_loss),
        num_origins=int(statistics.num_origins),
    )


def fixed_mixture_cumulative_loss(
    source: ComparatorStatistics | StaticComparatorAccumulator,
    weights: np.ndarray,
) -> float:
    statistics = _validated_statistics(source)
    mixture = np.asarray(weights, dtype=np.float64)
    experts = statistics.gram.shape[0]
    if mixture.shape != (experts,):
        raise ValueError(f"weights must have shape {(experts,)}")
    if not np.isfinite(mixture).all():
        raise FloatingPointError("mixture weights contain NaN or Inf")
    if np.any(mixture < -1e-12) or not np.isclose(
        mixture.sum(), 1.0, atol=1e-10
    ):
        raise ValueError("weights must belong to the probability simplex")
    loss = float(
        mixture @ statistics.gram @ mixture
        - 2.0 * statistics.expert_target_cross @ mixture
        + statistics.target_square_sum
    )
    return max(loss, 0.0)


def evaluate_static_comparator(
    source: ComparatorStatistics | StaticComparatorAccumulator,
    max_iterations: int = 20000,
    tolerance: float = 1e-12,
    eps: float = 1e-12,
) -> ComparatorResult:
    """Find the best fixed convex Expert mixture over the evaluation stream."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    statistics = _validated_statistics(source)
    experts = statistics.gram.shape[0]
    weights = np.full(experts, 1.0 / experts, dtype=np.float64)

    if statistics.num_origins == 0:
        comparator_loss = 0.0
        converged = True
        iterations = 0
    else:
        largest_eigenvalue = max(
            float(np.linalg.eigvalsh(statistics.gram)[-1]), 0.0
        )
        step_size = 1.0 / max(2.0 * largest_eigenvalue, eps)
        converged = False
        iterations = max_iterations
        for iteration in range(1, max_iterations + 1):
            gradient = 2.0 * (
                statistics.gram @ weights
                - statistics.expert_target_cross
            )
            updated = _project_simplex(weights - step_size * gradient)
            if np.linalg.norm(updated - weights, ord=2) <= tolerance:
                weights = updated
                converged = True
                iterations = iteration
                break
            weights = updated
        comparator_loss = fixed_mixture_cumulative_loss(statistics, weights)

    router_loss = statistics.router_cumulative_loss
    regret = router_loss - comparator_loss
    average_regret = (
        regret / statistics.num_origins if statistics.num_origins > 0 else 0.0
    )
    return ComparatorResult(
        comparator_type="static_fixed_convex_mixture",
        router_cumulative_loss=router_loss,
        comparator_loss=comparator_loss,
        regret=regret,
        average_regret=average_regret,
        num_origins=statistics.num_origins,
        comparator_weights=weights,
        converged=converged,
        iterations=iterations,
    )


def evaluate_dynamic_comparator(
    source: ComparatorStatistics | StaticComparatorAccumulator,
    comparator_class: str | None = None,
) -> ComparatorResult:
    """Reserved interface; no dynamic comparator is claimed in this version."""

    del source
    raise NotImplementedError(
        "TODO: define a dynamic comparator class and reliable optimizer first "
        "(for example <=K switches or a path-length constraint); "
        f"received comparator_class={comparator_class!r}"
    )


def save_comparator_diagnostics(
    result: ComparatorResult, directory: str
) -> tuple[str, str]:
    os.makedirs(directory, exist_ok=True)
    values = result.as_dict()
    npz_path = os.path.join(directory, "comparator_diagnostics.npz")
    json_path = os.path.join(directory, "comparator_diagnostics.json")
    np.savez_compressed(
        npz_path,
        comparator_type=np.asarray(result.comparator_type),
        router_cumulative_loss=np.asarray(result.router_cumulative_loss),
        static_comparator_loss=np.asarray(result.comparator_loss),
        static_regret=np.asarray(result.regret),
        average_static_regret=np.asarray(result.average_regret),
        num_origins=np.asarray(result.num_origins, dtype=np.int64),
        static_comparator_weights=np.asarray(
            result.comparator_weights, dtype=np.float64
        ),
        converged=np.asarray(result.converged),
        iterations=np.asarray(result.iterations, dtype=np.int64),
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2)
    return npz_path, json_path
