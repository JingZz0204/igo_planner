from __future__ import annotations

from abc import ABC, abstractmethod

from spatiotemporal_joint_planner.common import OptimizationProblem, OptimizationResult


class Optimizer(ABC):
    """Base interface for CMA-ES, SVGD, and future trajectory optimizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable optimizer name for configs and logs."""

    @abstractmethod
    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        """Run optimization and return the best solution plus debug state."""

    def reset(self) -> None:
        """Clear optimizer warm-start state if the implementation keeps any."""
