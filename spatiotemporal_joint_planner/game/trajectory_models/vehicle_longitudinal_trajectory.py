from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.common import fixed_time_grid, quintic_profile, trajectory_from_sl


@dataclass(frozen=True)
class VehicleLongitudinalTrajectoryConfig:
    min_terminal_speed: float = 0.0
    max_terminal_speed: float = 15.0
    min_terminal_s_offset: float = -10.0
    max_terminal_s_offset: float = 25.0
    terminal_longitudinal_accel: float = 0.0


class VehicleLongitudinalTrajectoryModel(TrajectoryModel):
    """Longitudinal target-lane vehicle trajectory: theta = [s_end, v_end]."""

    def __init__(self, config: Optional[VehicleLongitudinalTrajectoryConfig] = None):
        self.config = config or VehicleLongitudinalTrajectoryConfig()

    @property
    def name(self) -> str:
        return "vehicle_longitudinal_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        return 2

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        horizon = max(float(problem.horizon), 1e-3)
        nominal_s_end = float(problem.ego.s) + max(float(problem.ego.s_v), 0.0) * horizon
        low = np.array(
            [
                nominal_s_end + float(self.config.min_terminal_s_offset),
                float(self.config.min_terminal_speed),
            ],
            dtype=float,
        )
        high = np.array(
            [
                nominal_s_end + float(self.config.max_terminal_s_offset),
                float(self.config.max_terminal_speed),
            ],
            dtype=float,
        )
        return low, high

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        low, high = self.bounds(problem)
        horizon = max(float(problem.horizon), 1e-3)
        return np.array(
            [
                float(np.clip(float(problem.ego.s) + float(problem.ego.s_v) * horizon, low[0], high[0])),
                float(np.clip(float(problem.ego.s_v), low[1], high[1])),
            ],
            dtype=float,
        )

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != (2,):
            raise ValueError(f"{self.name} expects theta shape (2,), got {theta.shape}")
        low, high = self.bounds(problem)
        s_end, v_end = np.clip(theta, low, high)
        t = fixed_time_grid(problem.horizon, problem.dt)
        s, s_v, s_a = quintic_profile(
            problem.ego.s,
            problem.ego.s_v,
            problem.ego.s_a,
            float(s_end),
            float(v_end),
            float(self.config.terminal_longitudinal_accel),
            t,
        )
        l = np.full_like(t, float(problem.ego.l), dtype=float)
        l_v = np.zeros_like(t, dtype=float)
        l_a = np.zeros_like(t, dtype=float)
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
                "parameterization": "terminal_s_end_v_end",
                "fixed_horizon": float(problem.horizon),
                "theta": theta.copy(),
            },
        )

    def decode_batch_arrays(self, parameters_batch: np.ndarray, problem: PlanningProblem) -> dict:
        theta = np.asarray(parameters_batch, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError(f"{self.name} expects theta batch shape (B, 2), got {theta.shape}")

        low, high = self.bounds(problem)
        theta_clipped = np.clip(theta, low[None, :], high[None, :])
        s_end = theta_clipped[:, 0]
        v_end = theta_clipped[:, 1]
        t = fixed_time_grid(problem.horizon, problem.dt)
        s, s_v, s_a = self._quintic_profile_batch(
            p0=problem.ego.s,
            v0=problem.ego.s_v,
            a0=problem.ego.s_a,
            p1=s_end,
            v1=v_end,
            a1=float(self.config.terminal_longitudinal_accel),
            t=t,
        )
        l = np.full_like(s, float(problem.ego.l), dtype=float)
        l_v = np.zeros_like(s, dtype=float)
        l_a = np.zeros_like(s, dtype=float)
        return {
            "model": self.name,
            "t": t,
            "theta": theta.copy(),
            "theta_clipped": theta_clipped,
            "s": s,
            "l": l,
            "s_v": s_v,
            "l_v": l_v,
            "s_a": s_a,
            "l_a": l_a,
            "v": np.abs(s_v),
            "a": np.abs(s_a),
            "x": None,
            "y": None,
            "yaw": None,
            "kappa": None,
        }

    @staticmethod
    def _quintic_profile_batch(
        p0: float,
        v0: float,
        a0: float,
        p1: np.ndarray,
        v1: np.ndarray,
        a1: float,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        v1 = np.asarray(v1, dtype=float).reshape(-1)
        horizon = max(float(t[-1]), 1e-3)
        c0 = float(p0)
        c1 = float(v0)
        c2 = 0.5 * float(a0)
        matrix = np.array(
            [
                [horizon**3, horizon**4, horizon**5],
                [3.0 * horizon**2, 4.0 * horizon**3, 5.0 * horizon**4],
                [6.0 * horizon, 12.0 * horizon**2, 20.0 * horizon**3],
            ],
            dtype=float,
        )
        rhs = np.column_stack(
            [
                p1 - c0 - c1 * horizon - c2 * horizon**2,
                v1 - c1 - 2.0 * c2 * horizon,
                np.full_like(p1, float(a1) - 2.0 * c2),
            ]
        )
        c3_c4_c5 = np.linalg.solve(matrix, rhs.T).T
        c3 = c3_c4_c5[:, 0:1]
        c4 = c3_c4_c5[:, 1:2]
        c5 = c3_c4_c5[:, 2:3]
        tt = t[None, :]
        position = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4 + c5 * tt**5
        velocity = c1 + 2.0 * c2 * tt + 3.0 * c3 * tt**2 + 4.0 * c4 * tt**3 + 5.0 * c5 * tt**4
        accel = 2.0 * c2 + 6.0 * c3 * tt + 12.0 * c4 * tt**2 + 20.0 * c5 * tt**3
        return position, velocity, accel
