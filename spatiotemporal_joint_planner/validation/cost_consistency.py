from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem
from spatiotemporal_joint_planner.cost import CostFunction
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class CostConsistencyConfig:
    sample_count: int = 128
    seed: int = 0
    relative_tolerance: float = 1e-2
    absolute_tolerance: float = 1e-3


@dataclass(frozen=True)
class CostConsistencyResult:
    model_name: str
    sample_count: int
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    worst_index: int
    worst_parameters: np.ndarray
    worst_scalar_cost: float
    worst_batch_cost: float
    passed: bool

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "sample_count": int(self.sample_count),
            "max_absolute_error": float(self.max_absolute_error),
            "mean_absolute_error": float(self.mean_absolute_error),
            "max_relative_error": float(self.max_relative_error),
            "mean_relative_error": float(self.mean_relative_error),
            "worst_index": int(self.worst_index),
            "worst_parameters": np.asarray(self.worst_parameters, dtype=float).tolist(),
            "worst_scalar_cost": float(self.worst_scalar_cost),
            "worst_batch_cost": float(self.worst_batch_cost),
            "passed": bool(self.passed),
        }


def evaluate_cost_consistency(
    trajectory_model: TrajectoryModel,
    cost_function: CostFunction,
    problem: PlanningProblem,
    config: CostConsistencyConfig | None = None,
) -> CostConsistencyResult:
    cfg = config or CostConsistencyConfig()
    if not hasattr(trajectory_model, "decode_batch_arrays"):
        raise ValueError(f"{trajectory_model.name} does not provide decode_batch_arrays.")
    if not hasattr(cost_function, "evaluate_batch"):
        raise ValueError(f"{cost_function.name} does not provide evaluate_batch.")

    low, high = trajectory_model.bounds(problem)
    rng = np.random.default_rng(int(cfg.seed))
    parameters = rng.uniform(
        np.asarray(low, dtype=float),
        np.asarray(high, dtype=float),
        size=(max(int(cfg.sample_count), 1), np.asarray(low).size),
    )
    reference = np.asarray(trajectory_model.reference_parameters(problem), dtype=float).reshape(1, -1)
    parameters[0] = np.clip(reference[0], low, high)

    batch_arrays = trajectory_model.decode_batch_arrays(parameters, problem)
    batch_costs = np.asarray(cost_function.evaluate_batch(batch_arrays, problem), dtype=float).reshape(-1)
    scalar_costs = np.asarray(
        [
            float(cost_function.evaluate(trajectory_model.decode(theta, problem), problem).total)
            for theta in parameters
        ],
        dtype=float,
    )
    if batch_costs.shape != scalar_costs.shape:
        raise ValueError(f"Scalar/batch cost shape mismatch: {scalar_costs.shape} != {batch_costs.shape}")

    absolute = np.abs(batch_costs - scalar_costs)
    relative = absolute / np.maximum(np.abs(scalar_costs), 1.0)
    finite = np.isfinite(absolute) & np.isfinite(relative)
    if not np.any(finite):
        worst_index = 0
        max_absolute = float("inf")
        mean_absolute = float("inf")
        max_relative = float("inf")
        mean_relative = float("inf")
    else:
        finite_indices = np.where(finite)[0]
        worst_index = int(finite_indices[np.argmax(relative[finite_indices])])
        max_absolute = float(np.max(absolute[finite]))
        mean_absolute = float(np.mean(absolute[finite]))
        max_relative = float(np.max(relative[finite]))
        mean_relative = float(np.mean(relative[finite]))
    passed = bool(
        np.all(finite)
        and np.all(
            (absolute <= float(cfg.absolute_tolerance))
            | (relative <= float(cfg.relative_tolerance))
        )
    )
    return CostConsistencyResult(
        model_name=str(trajectory_model.name),
        sample_count=int(parameters.shape[0]),
        max_absolute_error=max_absolute,
        mean_absolute_error=mean_absolute,
        max_relative_error=max_relative,
        mean_relative_error=mean_relative,
        worst_index=worst_index,
        worst_parameters=parameters[worst_index].copy(),
        worst_scalar_cost=float(scalar_costs[worst_index]),
        worst_batch_cost=float(batch_costs[worst_index]),
        passed=passed,
    )
