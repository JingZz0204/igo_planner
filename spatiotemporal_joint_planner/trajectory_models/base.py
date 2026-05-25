from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory


class TrajectoryModel(ABC):
    """Base interface for fixed-horizon trajectory parameterizations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model name for configs, logs, and debug output."""

    @abstractmethod
    def parameter_dim(self, problem: PlanningProblem) -> int:
        """Return the optimizer parameter dimension for this problem."""

    @abstractmethod
    def bounds(self, problem: PlanningProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper parameter bounds."""

    @abstractmethod
    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        """Return a nominal parameter vector suitable for seeding optimizers."""

    @abstractmethod
    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        """Convert optimizer parameters into a fixed-horizon trajectory."""

    def decode_batch(self, parameters: Sequence[np.ndarray], problem: PlanningProblem) -> list[Trajectory]:
        """Decode a batch of parameter vectors; subclasses can override for speed."""
        return [self.decode(np.asarray(theta, dtype=float), problem) for theta in parameters]
