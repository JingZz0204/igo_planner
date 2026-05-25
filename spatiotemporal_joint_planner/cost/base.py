from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from spatiotemporal_joint_planner.common import CostResult, PlanningProblem, Trajectory


class CostFunction(ABC):
    """Base interface for trajectory-level spatiotemporal costs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable cost name for logging and debug output."""

    @abstractmethod
    def evaluate(self, trajectory: Trajectory, problem: PlanningProblem) -> CostResult:
        """Evaluate one trajectory against the planning problem."""

    def evaluate_batch(self, trajectories: Sequence[Trajectory], problem: PlanningProblem) -> list[CostResult]:
        """Evaluate multiple trajectories; subclasses can override for vectorization."""
        return [self.evaluate(trajectory, problem) for trajectory in trajectories]
