from __future__ import annotations

import math
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

    def decode_batch_arrays(self, parameters_batch: np.ndarray, problem: PlanningProblem) -> dict:
        """Vectorized lattice decode for optimizer-side batch cost evaluation."""

        theta = np.asarray(parameters_batch, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError(f"{self.name} expects theta batch shape (B, 2), got {theta.shape}")

        low, high = self.bounds(problem)
        theta_clipped = np.clip(theta, low[None, :], high[None, :])
        l_end = theta_clipped[:, 0]
        v_end = theta_clipped[:, 1]
        t = fixed_time_grid(problem.horizon, problem.dt)

        s, s_v, s_a = self._quartic_profile_batch(
            s0=problem.ego.s,
            v0=problem.ego.s_v,
            a0=problem.ego.s_a,
            v1=v_end,
            t=t,
            a1=float(self.config.terminal_longitudinal_accel),
        )
        l, l_v, l_a = self._quintic_profile_batch(
            p0=problem.ego.l,
            v0=problem.ego.l_v,
            a0=problem.ego.l_a,
            p1=l_end,
            v1=float(self.config.terminal_lateral_speed),
            a1=float(self.config.terminal_lateral_accel),
            t=t,
        )
        x, y = self._xy_from_sl_batch(problem, s, l)
        yaw = None
        kappa = None
        if x is not None and y is not None:
            yaw = np.arctan2(self._finite_difference_batch(y, t), self._finite_difference_batch(x, t))
            kappa = self._curvature_from_xy_batch(x, y, t)

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
            "x": x,
            "y": y,
            "yaw": yaw,
            "v": np.hypot(s_v, l_v),
            "a": np.hypot(s_a, l_a),
            "kappa": kappa,
        }

    @staticmethod
    def _quartic_profile_batch(
        s0: float,
        v0: float,
        a0: float,
        v1: np.ndarray,
        t: np.ndarray,
        a1: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        v1 = np.asarray(v1, dtype=float).reshape(-1)
        horizon = max(float(t[-1]), 1e-3)
        c0 = float(s0)
        c1 = float(v0)
        c2 = 0.5 * float(a0)
        matrix = np.array(
            [
                [3.0 * horizon**2, 4.0 * horizon**3],
                [6.0 * horizon, 12.0 * horizon**2],
            ],
            dtype=float,
        )
        rhs = np.column_stack(
            [
                v1 - c1 - 2.0 * c2 * horizon,
                np.full_like(v1, float(a1) - 2.0 * c2),
            ]
        )
        c3_c4 = np.linalg.solve(matrix, rhs.T).T
        c3 = c3_c4[:, 0:1]
        c4 = c3_c4[:, 1:2]
        tt = t[None, :]
        position = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4
        velocity = c1 + 2.0 * c2 * tt + 3.0 * c3 * tt**2 + 4.0 * c4 * tt**3
        accel = 2.0 * c2 + 6.0 * c3 * tt + 12.0 * c4 * tt**2
        return position, velocity, accel

    @staticmethod
    def _quintic_profile_batch(
        p0: float,
        v0: float,
        a0: float,
        p1: np.ndarray,
        v1: float,
        a1: float,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
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
                np.full_like(p1, float(v1) - c1 - 2.0 * c2 * horizon),
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

    @classmethod
    def _xy_from_sl_batch(cls, problem: PlanningProblem, s_values: np.ndarray, l_values: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ref_path = problem.ref_path
        s_values = np.asarray(s_values, dtype=float)
        l_values = np.asarray(l_values, dtype=float)
        if not hasattr(ref_path, "calc_position") or not hasattr(ref_path, "calc_yaw"):
            return s_values.copy(), l_values.copy()

        route_end = cls._route_end_s(ref_path)
        x_values = np.empty_like(s_values, dtype=float)
        y_values = np.empty_like(l_values, dtype=float)
        for index in np.ndindex(s_values.shape):
            s_clamped = float(np.clip(float(s_values[index]), 0.0, route_end))
            xy_ref = ref_path.calc_position(s_clamped)
            if xy_ref is None or xy_ref[0] is None or xy_ref[1] is None:
                return None, None
            yaw = float(ref_path.calc_yaw(s_clamped))
            x_values[index] = float(xy_ref[0]) + float(l_values[index]) * math.cos(yaw + math.pi / 2.0)
            y_values[index] = float(xy_ref[1]) + float(l_values[index]) * math.sin(yaw + math.pi / 2.0)
        return x_values, y_values

    @staticmethod
    def _finite_difference_batch(values: np.ndarray, t: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        t = np.asarray(t, dtype=float)
        if values.shape[-1] <= 1:
            return np.zeros_like(values)
        edge_order = 2 if values.shape[-1] >= 3 else 1
        return np.gradient(values, t, axis=-1, edge_order=edge_order)

    @classmethod
    def _curvature_from_xy_batch(cls, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        vx = cls._finite_difference_batch(np.asarray(x, dtype=float), t)
        vy = cls._finite_difference_batch(np.asarray(y, dtype=float), t)
        ax = cls._finite_difference_batch(vx, t)
        ay = cls._finite_difference_batch(vy, t)
        denom = np.maximum((vx * vx + vy * vy) ** 1.5, 1e-6)
        return (vx * ay - vy * ax) / denom

    @staticmethod
    def _route_end_s(ref_path) -> float:
        if hasattr(ref_path, "s"):
            values = np.asarray(ref_path.s, dtype=float)
            if values.size:
                return max(float(values[-1]), 1e-3)
        return 1.0e6
