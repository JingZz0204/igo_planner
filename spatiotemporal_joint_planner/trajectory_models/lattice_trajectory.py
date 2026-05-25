from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.common import (
    fixed_time_grid,
    quartic_profile,
    quintic_profile,
    trajectory_from_sl,
)


@dataclass(frozen=True)
class LatticeTrajectoryConfig:
    min_terminal_speed: float = 0.5
    max_terminal_speed: float = 15.0
    ego_width: float = 2.70
    road_edge_margin: float = 0.0
    terminal_lateral_speed: float = 0.0
    terminal_lateral_accel: float = 0.0
    terminal_longitudinal_accel: float = 0.0


class LatticeTrajectoryModel(TrajectoryModel):
    """Fixed-horizon terminal-state trajectory: theta = [l_end, v_end]."""

    def __init__(self, config: Optional[LatticeTrajectoryConfig] = None):
        self.config = config or LatticeTrajectoryConfig()

    @property
    def name(self) -> str:
        return "lattice_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        return 2

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        low_l = min(float(problem.road_boundary.right_l), float(problem.road_boundary.left_l))
        high_l = max(float(problem.road_boundary.right_l), float(problem.road_boundary.left_l))
        half_width = max(0.0, 0.5 * float(self.config.ego_width))
        road_edge_margin = max(0.0, float(self.config.road_edge_margin))
        center_low_l = low_l + half_width + road_edge_margin
        center_high_l = high_l - half_width - road_edge_margin
        if center_low_l > center_high_l:
            center_mid_l = 0.5 * (low_l + high_l)
            center_low_l = center_mid_l
            center_high_l = center_mid_l
        low = np.array([center_low_l, float(self.config.min_terminal_speed)], dtype=float)
        high = np.array([center_high_l, float(self.config.max_terminal_speed)], dtype=float)
        return low, high

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        low, high = self.bounds(problem)
        return np.array(
            [
                float(np.clip(problem.ego.l, low[0], high[0])),
                float(np.clip(problem.ego.s_v, low[1], high[1])),
            ],
            dtype=float,
        )

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != (2,):
            raise ValueError(f"{self.name} expects theta shape (2,), got {theta.shape}")

        low, high = self.bounds(problem)
        l_end, v_end = np.clip(theta, low, high)
        t = fixed_time_grid(problem.horizon, problem.dt)
        s, s_v, s_a = quartic_profile(
            problem.ego.s,
            problem.ego.s_v,
            problem.ego.s_a,
            float(v_end),
            t,
            a1=float(self.config.terminal_longitudinal_accel),
        )
        l, l_v, l_a = quintic_profile(
            problem.ego.l,
            problem.ego.l_v,
            problem.ego.l_a,
            float(l_end),
            float(self.config.terminal_lateral_speed),
            float(self.config.terminal_lateral_accel),
            t,
        )
        return trajectory_from_sl(
            problem,
            t,
            s,
            l,
            s_v=s_v,
            l_v=l_v,
            s_a=s_a,
            l_a=l_a,
            metadata={
                "model": self.name,
                "parameterization": "terminal_l_end_v_end",
                "fixed_horizon": float(problem.horizon),
                "theta": theta.copy(),
            },
        )
