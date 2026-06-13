from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.common import fixed_time_grid, quartic_profile, trajectory_from_sl, xy_from_sl


@dataclass(frozen=True)
class FrenetViaBSplineTrajectoryConfig:
    min_terminal_speed: float = 0.5
    max_terminal_speed: float = 15.0
    ego_width: float = 2.70
    road_edge_margin: float = 0.0
    min_mid_time: float = 1.2
    max_mid_time: float = 4.0
    min_mid_s_offset: float = 12.0
    min_mid_speed_floor: float = 3.0
    min_mid_time_ratio: float = 0.25
    max_mid_time_ratio: float = 0.85
    terminal_time_buffer: float = 0.5
    terminal_longitudinal_accel: float = 0.0
    monotonic_lateral: bool = False
    clip_control_to_road: bool = True


class FrenetViaBSplineTrajectoryModel(TrajectoryModel):
    """Frenet quadratic B-spline/Bezier with one semantic via point.

    theta = [t_mid, l_mid, l_end, v_end]

    The lateral curve is evaluated as a quadratic Bezier in normalized time.
    ``l_mid`` is treated as an actual via point at ``t_mid`` rather than as a
    raw control point, and the internal control point is solved from it. This
    keeps the parameter meaning close to terminal-state lattice sampling while
    adding one extra lateral-shape degree of freedom.
    """

    def __init__(self, config: Optional[FrenetViaBSplineTrajectoryConfig] = None):
        self.config = config or FrenetViaBSplineTrajectoryConfig()

    @property
    def name(self) -> str:
        return "frenet_via_bspline_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        return 4

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        lateral_low, lateral_high = self._lateral_center_bounds(problem)
        t_low, t_high = self._mid_time_bounds(problem)
        low = np.array(
            [
                t_low,
                lateral_low,
                lateral_low,
                max(float(self.config.min_terminal_speed), 0.0),
            ],
            dtype=float,
        )
        high = np.array(
            [
                t_high,
                lateral_high,
                lateral_high,
                max(float(self.config.max_terminal_speed), float(self.config.min_terminal_speed)),
            ],
            dtype=float,
        )
        return low, high

    def _mid_time_bounds(self, problem: PlanningProblem) -> tuple[float, float]:
        horizon = max(float(problem.horizon), 1e-3)
        speed_floor = max(float(self.config.min_mid_speed_floor), 1e-3)
        ego_speed = max(abs(float(problem.ego.s_v)), speed_floor)
        min_s_time = max(float(self.config.min_mid_s_offset), 0.0) / ego_speed

        t_low = max(
            float(self.config.min_mid_time),
            min_s_time,
            float(self.config.min_mid_time_ratio) * horizon,
        )
        t_high = min(
            float(self.config.max_mid_time),
            float(self.config.max_mid_time_ratio) * horizon,
            horizon - max(float(self.config.terminal_time_buffer), 0.0),
        )
        t_high = float(np.clip(t_high, 0.1 * horizon, 0.95 * horizon))
        if t_low >= t_high:
            t_low = max(0.05 * horizon, t_high - min(0.5, 0.1 * horizon))
        return float(t_low), float(t_high)

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        low, high = self.bounds(problem)
        target_l = float(np.clip(self._target_l(problem), low[2], high[2]))
        current_l = float(problem.ego.l)
        t_mid = 0.5 * (low[0] + high[0])
        l_mid = 0.5 * (current_l + target_l)
        target_speed = float(np.clip(self._target_speed(problem), low[3], high[3]))
        return np.clip(np.array([t_mid, l_mid, target_l, target_speed], dtype=float), low, high)

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        arrays = self.decode_batch_arrays(np.asarray(parameters, dtype=float).reshape(1, -1), problem)
        metadata = self._metadata_from_arrays(np.asarray(parameters, dtype=float), arrays, problem)
        trajectory = trajectory_from_sl(
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
        trajectory.kappa = np.asarray(arrays["kappa"][0], dtype=float).copy()
        return trajectory

    def decode_batch_arrays(self, parameters_batch: np.ndarray, problem: PlanningProblem) -> dict:
        theta = np.asarray(parameters_batch, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if theta.ndim != 2 or theta.shape[1] != 4:
            raise ValueError(f"{self.name} expects theta batch shape (B, 4), got {theta.shape}")

        low, high = self.bounds(problem)
        theta_clipped = np.clip(theta, low[None, :], high[None, :])
        t_mid = theta_clipped[:, 0]
        l_mid = theta_clipped[:, 1]
        l_end = theta_clipped[:, 2]
        v_end = theta_clipped[:, 3]
        t = fixed_time_grid(problem.horizon, problem.dt)

        l0 = float(problem.ego.l)
        if bool(self.config.monotonic_lateral):
            l_mid = np.clip(l_mid, np.minimum(l0, l_end), np.maximum(l0, l_end))

        horizon = max(float(t[-1]), 1e-6)
        u_mid = np.clip(t_mid / horizon, 1e-3, 1.0 - 1e-3)
        denom = np.maximum(2.0 * u_mid * (1.0 - u_mid), 1e-6)
        l_ctrl = (l_mid - (1.0 - u_mid) ** 2 * l0 - u_mid**2 * l_end) / denom
        if bool(self.config.monotonic_lateral):
            l_ctrl = np.clip(l_ctrl, np.minimum(l0, l_end), np.maximum(l0, l_end))
        if bool(self.config.clip_control_to_road):
            lateral_low, lateral_high = self._lateral_center_bounds(problem)
            l_ctrl = np.clip(l_ctrl, lateral_low, lateral_high)

        u = np.clip(t / horizon, 0.0, 1.0)
        l = (
            (1.0 - u)[None, :] ** 2 * l0
            + 2.0 * u[None, :] * (1.0 - u)[None, :] * l_ctrl[:, None]
            + u[None, :] ** 2 * l_end[:, None]
        )
        actual_l_mid = np.asarray([np.interp(float(tm), t, row) for tm, row in zip(t_mid, l)], dtype=float)
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
            "t_mid": t_mid,
            "l_mid": actual_l_mid,
            "l_mid_desired": l_mid,
            "l_end": l_end,
            "v_end": v_end,
            "l_ctrl": np.column_stack([np.full_like(l_end, l0), l_ctrl, l_end]),
            "s_ctrl": self._control_s_batch(s, t, t_mid),
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
        via_s = float(np.interp(float(arrays["t_mid"][0]), arrays["t"], arrays["s"][0]))
        via_l = float(arrays["l_mid"][0])
        via_x, via_y = xy_from_sl(problem, np.array([via_s], dtype=float), np.array([via_l], dtype=float))
        return {
            "model": self.name,
            "parameterization": "frenet_quadratic_bspline_via_t_mid_l_mid_l_end_v_end",
            "fixed_horizon": float(problem.horizon),
            "theta": np.asarray(parameters, dtype=float).copy(),
            "bspline_control_t": np.array([0.0, float(arrays["t_mid"][0]), float(problem.horizon)], dtype=float),
            "bspline_control_s": control_s,
            "bspline_control_l": control_l,
            "bspline_control_x": None if control_x is None else np.asarray(control_x, dtype=float),
            "bspline_control_y": None if control_y is None else np.asarray(control_y, dtype=float),
            "bspline_via_t": float(arrays["t_mid"][0]),
            "bspline_via_s": via_s,
            "bspline_via_l": via_l,
            "bspline_via_x": None if via_x is None else float(via_x[0]),
            "bspline_via_y": None if via_y is None else float(via_y[0]),
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
    def _gradient_batch(values: np.ndarray, t: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        t = np.asarray(t, dtype=float)
        if values.shape[1] <= 1:
            return np.zeros_like(values)
        return np.gradient(values, t, axis=1, edge_order=2 if values.shape[1] >= 3 else 1)

    @classmethod
    def _frenet_lateral_curvature_batch(cls, s: np.ndarray, l: np.ndarray, t: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        l = np.asarray(l, dtype=float)
        s_t = cls._gradient_batch(s, t)
        l_t = cls._gradient_batch(l, t)
        s_tt = cls._gradient_batch(s_t, t)
        l_tt = cls._gradient_batch(l_t, t)
        dl_ds = l_t / np.maximum(np.abs(s_t), 1e-3)
        d2l_ds2 = (l_tt * s_t - l_t * s_tt) / np.maximum(np.abs(s_t) ** 3, 1e-3)
        return d2l_ds2 / np.maximum((1.0 + dl_ds * dl_ds) ** 1.5, 1e-6)

    @staticmethod
    def _control_s_batch(s: np.ndarray, t: np.ndarray, t_mid: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        t_mid = np.asarray(t_mid, dtype=float).reshape(-1)
        mid_s = np.asarray([np.interp(float(tm), t, row) for tm, row in zip(t_mid, s)], dtype=float)
        return np.column_stack([s[:, 0], mid_s, s[:, -1]])

    def _lateral_center_bounds(self, problem: PlanningProblem) -> tuple[float, float]:
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        half_width = max(0.0, 0.5 * float(self.config.ego_width))
        edge_margin = max(0.0, float(self.config.road_edge_margin))
        low = right + half_width + edge_margin
        high = left - half_width - edge_margin
        if low > high:
            center = 0.5 * (left + right)
            low = center
            high = center
        return float(low), float(high)

    @staticmethod
    def _target_l(problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("reference_l", "target_lane_l", "target_l", "preferred_l"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {key} metadata: {metadata[key]!r}") from exc
        return float(problem.ego.l)

    def _target_speed(self, problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    return min(float(metadata[key]), float(self.config.max_terminal_speed))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {key} metadata: {metadata[key]!r}") from exc
        return min(max(float(problem.ego.s_v), float(self.config.min_terminal_speed)), float(self.config.max_terminal_speed))
