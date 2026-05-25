from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import (
    CostResult,
    OptimizationProblem,
    PlannerResult,
    PlanningProblem,
    Trajectory,
)
from spatiotemporal_joint_planner.cost import CostFunction
from spatiotemporal_joint_planner.optimizer import CMAESConfig, CMAESOptimizer, Optimizer
from spatiotemporal_joint_planner.planner.base import Planner
from spatiotemporal_joint_planner.planner.warm_start import (
    WarmStartContext,
    WarmStartGenerator,
    default_parametric_warm_start_generator,
)
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class ParametricPlannerConfig:
    candidate_limit: int = 16
    warm_start: bool = True
    max_initial_anchors: int = 96


class ParametricPlanner(Planner):
    """Planner that optimizes finite-dimensional trajectory parameters."""

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        cost_function: CostFunction,
        optimizer: Optional[Optimizer] = None,
        config: Optional[ParametricPlannerConfig] = None,
        warm_start_generator: Optional[WarmStartGenerator] = None,
    ):
        self.trajectory_model = trajectory_model
        self.cost_function = cost_function
        self.optimizer = optimizer or CMAESOptimizer(CMAESConfig())
        self.config = config or ParametricPlannerConfig()
        self.warm_start_generator = warm_start_generator or default_parametric_warm_start_generator()
        self._last_parameters: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "parametric_planner"

    def plan(self, problem: PlanningProblem) -> PlannerResult:
        low, high = self.trajectory_model.bounds(problem)
        seeds = self._initial_population(problem, low, high)

        def objective(parameters: np.ndarray) -> float:
            try:
                trajectory = self.trajectory_model.decode(parameters, problem)
                cost = self.cost_function.evaluate(trajectory, problem)
            except Exception:
                return float("inf")
            if not np.isfinite(cost.total):
                return float("inf")
            return float(cost.total)

        optimization_problem = OptimizationProblem(
            objective=objective,
            initial_population=seeds,
            lower_bound=low,
            upper_bound=high,
            metadata={
                "planner": self.name,
                "trajectory_model": self.trajectory_model.name,
                "cost_function": self.cost_function.name,
            },
        )
        optimization = self.optimizer.optimize(optimization_problem)
        best_trajectory, best_cost = self._decode_and_evaluate(optimization.best_position, problem)
        candidates = self._candidate_trajectories(optimization.population, optimization.values, problem)

        status = "success"
        if best_trajectory is None or best_cost is None:
            status = "no_valid_trajectory"
        elif not best_cost.feasible:
            status = "infeasible_best"

        if best_trajectory is not None and self.config.warm_start:
            self._last_parameters = np.asarray(optimization.best_position, dtype=float).copy()

        return PlannerResult(
            trajectory=best_trajectory,
            cost=best_cost,
            status=status,
            candidates=candidates,
            optimization=optimization,
            metadata={
                "planner": self.name,
                "trajectory_model": self.trajectory_model.name,
                "cost_function": self.cost_function.name,
                "optimizer": self.optimizer.name,
                "parameter_dim": int(self.trajectory_model.parameter_dim(problem)),
            },
        )

    def reset(self) -> None:
        self._last_parameters = None
        self.optimizer.reset()

    def _initial_population(self, problem: PlanningProblem, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        previous = self._last_parameters if self.config.warm_start else None
        context = WarmStartContext(
            problem=problem,
            trajectory_model=self.trajectory_model,
            lower_bound=np.asarray(low, dtype=float),
            upper_bound=np.asarray(high, dtype=float),
            previous_parameters=previous,
            max_count=int(self.config.max_initial_anchors),
        )
        if not self.warm_start_generator.supports(context):
            self.warm_start_generator = default_parametric_warm_start_generator()
        seeds = self.warm_start_generator.generate(context)
        if seeds.size == 0:
            raise ValueError(f"No valid seeds for trajectory model {self.trajectory_model.name}")
        return seeds

    def _decode_and_evaluate(
        self,
        parameters: np.ndarray,
        problem: PlanningProblem,
    ) -> tuple[Optional[Trajectory], Optional[CostResult]]:
        try:
            trajectory = self.trajectory_model.decode(parameters, problem)
            cost = self.cost_function.evaluate(trajectory, problem)
        except Exception:
            return None, None
        return trajectory, cost

    def _candidate_trajectories(
        self,
        population: np.ndarray,
        values: np.ndarray,
        problem: PlanningProblem,
    ) -> Sequence[Trajectory]:
        if population.size == 0 or values.size == 0:
            return []

        order = np.argsort(np.asarray(values, dtype=float))
        candidates = []
        seen = set()
        for idx in order:
            if len(candidates) >= int(self.config.candidate_limit):
                break
            parameters = np.asarray(population[int(idx)], dtype=float)
            key = tuple(np.round(parameters, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            trajectory, cost = self._decode_and_evaluate(parameters, problem)
            if trajectory is None:
                continue
            metadata = dict(trajectory.metadata)
            metadata["candidate_cost"] = None if cost is None else float(cost.total)
            trajectory.metadata = metadata
            candidates.append(trajectory)
        return candidates

    @staticmethod
    def _dedupe_rows(values: np.ndarray) -> np.ndarray:
        rows = []
        seen = set()
        for row in values:
            key = tuple(np.round(np.asarray(row, dtype=float), 9).tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(np.asarray(row, dtype=float))
        return np.asarray(rows, dtype=float)
