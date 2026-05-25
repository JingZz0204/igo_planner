from __future__ import annotations

import numpy as np
from typing import Optional

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.lattice_trajectory import (
    LatticeTrajectoryConfig,
    LatticeTrajectoryModel,
)


class SvgdParticleTrajectoryModel(LatticeTrajectoryModel):
    """SVGD particle trajectory model with particles z = [l_end, v_end]."""

    def __init__(self, config: Optional[LatticeTrajectoryConfig] = None):
        super().__init__(config=config)

    @property
    def name(self) -> str:
        return "svgd_particle_trajectory"

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        trajectory = super().decode(parameters, problem)
        metadata = dict(trajectory.metadata)
        metadata["model"] = self.name
        metadata["particle_parameterization"] = "terminal_l_end_v_end"
        metadata["fixed_horizon"] = float(problem.horizon)
        trajectory.metadata = metadata
        return trajectory

    def decode_particles(self, particles: np.ndarray, problem: PlanningProblem) -> list[Trajectory]:
        particles = np.asarray(particles, dtype=float)
        if particles.ndim != 2 or particles.shape[1] != 2:
            raise ValueError(f"{self.name} expects particle array shape (n, 2), got {particles.shape}")
        return [self.decode(particle, problem) for particle in particles]
