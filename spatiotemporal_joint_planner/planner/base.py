from __future__ import annotations

from abc import ABC, abstractmethod

from spatiotemporal_joint_planner.common import PlannerResult, PlanningProblem


class Planner(ABC):
    """Base interface for closed-loop spatiotemporal planners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable planner name for CLI selection and experiment output."""

    @abstractmethod
    def plan(self, problem: PlanningProblem) -> PlannerResult:
        """Plan one cycle from the current planning problem."""

    def reset(self) -> None:
        """Clear planner warm-start state if the implementation keeps any."""
