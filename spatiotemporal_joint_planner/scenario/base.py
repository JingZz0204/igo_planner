from __future__ import annotations

from abc import ABC, abstractmethod

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem


class Scenario(ABC):
    """Base interface for deterministic and interactive planning scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable scenario name for CLI selection and result folders."""

    @abstractmethod
    def initial_state(self) -> EgoState:
        """Return the ego state at scenario start."""

    @abstractmethod
    def build_problem(self, ego: EgoState, t: float = 0.0) -> PlanningProblem:
        """Create a planning problem for the current ego state and time."""

    def actors_at(self, t: float) -> list[ActorPrediction]:
        """Return actor predictions at time t; static scenarios may ignore t."""
        return list(self.build_problem(self.initial_state(), t).actors)
