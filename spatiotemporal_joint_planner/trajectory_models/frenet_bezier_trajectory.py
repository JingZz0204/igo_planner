from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.common import fixed_time_grid, trajectory_from_sl, xy_from_sl
from spatiotemporal_joint_planner.trajectory_models.frenet_via_bspline_trajectory import (
    FrenetViaBSplineTrajectoryModel,
)


@dataclass(frozen=True)
class FrenetBezierTrajectoryConfig:
    min_terminal_speed: float = 0.5
    max_terminal_speed: float = 15.0
    ego_width: float = 2.70
    road_edge_margin: float = 0.0
    terminal_longitudinal_accel: float = 0.0
    clip_control_to_road: bool = True


class FrenetBezierTrajectoryModel(FrenetViaBSplineTrajectoryModel):
    """Frenet quadratic Bezier trajectory with three lateral control points.

    P0 is fixed by the current ego lateral state. Theta optimizes:

    theta = [l_ctrl, l_end, v_end]

    Compared with ``frenet_via_bspline_trajectory``, ``l_ctrl`` is the actual
    middle Bezier control point rather than an actual via point on the curve.
    This gives the optimizer a more direct shape handle for temporary nudge or
    overtake-and-return behavior.
    """

    def __init__(self, config: Optional[FrenetBezierTrajectoryConfig] = None):
        self.config = config or FrenetBezierTrajectoryConfig()

    @property
    def name(self) -> str:
        return "frenet_bezier_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        return 3

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        lateral_low, lateral_high = self._lateral_center_bounds(problem)
        low = np.array(
            [
                lateral_low,
                lateral_low,
                max(float(self.config.min_terminal_speed), 0.0),
            ],
            dtype=float,
        )
        high = np.array(
            [
                lateral_high,
                lateral_high,
                max(float(self.config.max_terminal_speed), float(self.config.min_terminal_speed)),
            ],
            dtype=float,
        )
        return low, high

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        low, high = self.bounds(problem)
        target_l = float(np.clip(self._target_l(problem), low[1], high[1]))
        current_l = float(problem.ego.l)
        l_ctrl = 0.5 * (current_l + target_l)
        target_speed = float(np.clip(self._target_speed(problem), low[2], high[2]))
        return np.clip(np.array([l_ctrl, target_l, target_speed], dtype=float), low, high)

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        arrays = self.decode_batch_arrays(np.asarray(parameters, dtype=float).reshape(1, -1), problem)
        metadata = self._metadata_from_arrays(np.asarray(parameters, dtype=float), arrays, problem)
        return trajectory_from_sl(
            problem,
            arrays["t"],
            arrays["s"][0],
            arrays["l"][0],
            s_v=arrays["s_v"][0],
            l_v=arrays["l_v"][0],
            s_a=arrays["s_a"][0],
            l_a=arrays["l_a"][0],
            metadata=metadata,
        )

    def decode_batch_arrays(self, parameters_batch: np.ndarray, problem: PlanningProblem) -> dict:
        theta = np.asarray(parameters_batch, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[1] != 3:
            raise ValueError(f"{self.name} expects theta batch shape (B, 3), got {theta.shape}")

        low, high = self.bounds(problem)
        theta_clipped = np.clip(theta, low[None, :], high[None, :])
        l_ctrl = theta_clipped[:, 0]
        l_end = theta_clipped[:, 1]
        v_end = theta_clipped[:, 2]
        if bool(self.config.clip_control_to_road):
            lateral_low, lateral_high = self._lateral_center_bounds(problem)
            l_ctrl = np.clip(l_ctrl, lateral_low, lateral_high)

        t = fixed_time_grid(problem.horizon, problem.dt)
        horizon = max(float(t[-1]), 1e-6)
        u = np.clip(t / horizon, 0.0, 1.0)
        l0 = float(problem.ego.l)
        l = (
            (1.0 - u)[None, :] ** 2 * l0
            + 2.0 * u[None, :] * (1.0 - u)[None, :] * l_ctrl[:, None]
            + u[None, :] ** 2 * l_end[:, None]
        )
        l_v = self._gradient_batch(l, t)
        l_a = self._gradient_batch(l_v, t)
        s, s_v, s_a = self._quartic_profile_batch(
            s0=float(problem.ego.s),
            v0=float(problem.ego.s_v),
            a0=float(problem.ego.s_a),
            v1=v_end,
            t=t,
            a1=float(self.config.terminal_longitudinal_accel),
        )
        kappa = self._frenet_lateral_curvature_batch(s, l, t)
        return {
            "model": self.name,
            "t": t,
            "theta": theta.copy(),
            "theta_clipped": theta_clipped,
            "l_ctrl": np.column_stack([np.full_like(l_end, l0), l_ctrl, l_end]),
            "l_end": l_end,
            "v_end": v_end,
            "s_ctrl": np.column_stack([s[:, 0], 0.5 * (s[:, 0] + s[:, -1]), s[:, -1]]),
            "s": s,
            "l": l,
            "s_v": s_v,
            "l_v": l_v,
            "s_a": s_a,
            "l_a": l_a,
            "x": None,
            "y": None,
            "yaw": None,
            "v": np.hypot(s_v, l_v),
            "a": np.hypot(s_a, l_a),
            "kappa": kappa,
        }

    def _metadata_from_arrays(self, parameters: np.ndarray, arrays: dict, problem: PlanningProblem) -> dict:
        control_s = np.asarray(arrays["s_ctrl"][0], dtype=float)
        control_l = np.asarray(arrays["l_ctrl"][0], dtype=float)
        control_x, control_y = xy_from_sl(problem, control_s, control_l)
        return {
            "model": self.name,
            "parameterization": "frenet_quadratic_bezier_l_ctrl_l_end_v_end",
            "fixed_horizon": float(problem.horizon),
            "theta": np.asarray(parameters, dtype=float).copy(),
            "bspline_control_t": np.array([0.0, 0.5 * float(problem.horizon), float(problem.horizon)], dtype=float),
            "bspline_control_s": control_s,
            "bspline_control_l": control_l,
            "bspline_control_x": None if control_x is None else np.asarray(control_x, dtype=float),
            "bspline_control_y": None if control_y is None else np.asarray(control_y, dtype=float),
        }
