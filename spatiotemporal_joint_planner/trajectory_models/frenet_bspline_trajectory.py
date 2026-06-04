from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.common import fixed_time_grid, trajectory_from_sl, xy_from_sl


@dataclass(frozen=True)
class FrenetBSplineTrajectoryConfig:
    degree: int = 2
    num_control_points: int = 3
    min_speed: float = 0.5
    max_speed: float = 15.0
    ego_width: float = 2.70
    road_edge_margin: float = 0.0


class FrenetBSplineTrajectoryModel(TrajectoryModel):
    """Structured Frenet B-spline trajectory.

    theta = [l_ctrl_mid, l_end, v_end]

    The actual lateral B-spline control polygon is exactly
    [ego_l, l_ctrl_mid, l_end]. The actual speed control polygon is exactly
    [ego_v, 0.5 * (ego_v + v_end), v_end]. No hidden control-point expansion is
    used, so the parameterization, visualization, and trajectory generation are
    aligned.
    """

    def __init__(self, config: Optional[FrenetBSplineTrajectoryConfig] = None):
        self.config = config or FrenetBSplineTrajectoryConfig()

    @property
    def name(self) -> str:
        return "frenet_bspline_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        return 3

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        lateral_low, lateral_high = self._lateral_center_bounds(problem)
        low = np.array(
            [
                lateral_low,
                lateral_low,
                max(float(self.config.min_speed), 0.0),
            ],
            dtype=float,
        )
        high = np.array(
            [
                lateral_high,
                lateral_high,
                max(float(self.config.max_speed), float(self.config.min_speed)),
            ],
            dtype=float,
        )
        return low, high

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        low, high = self.bounds(problem)
        target_l = float(np.clip(self._target_l(problem), low[1], high[1]))
        current_l = float(np.clip(problem.ego.l, low[0], high[0]))
        l_ctrl_mid = 0.5 * (current_l + target_l)
        target_speed = float(np.clip(self._target_speed(problem), low[2], high[2]))
        return np.clip(np.array([l_ctrl_mid, target_l, target_speed], dtype=float), low, high)

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        arrays = self.decode_batch_arrays(np.asarray(parameters, dtype=float).reshape(1, -1), problem)
        control_u = self._control_abscissae(self._num_control_points(), self._degree())
        control_t = control_u * max(float(problem.horizon), 1e-6)
        control_s = np.interp(control_t, arrays["t"], arrays["s"][0], left=arrays["s"][0, 0], right=arrays["s"][0, -1])
        control_l = np.asarray(arrays["l_ctrl"][0], dtype=float)
        control_x, control_y = xy_from_sl(problem, control_s, control_l)
        semantic_control_t = np.array([0.0, 0.5 * float(problem.horizon), float(problem.horizon)], dtype=float)
        semantic_control_s = np.interp(
            semantic_control_t,
            arrays["t"],
            arrays["s"][0],
            left=arrays["s"][0, 0],
            right=arrays["s"][0, -1],
        )
        semantic_control_l = np.array(
            [
                float(problem.ego.l),
                float(arrays["l_ctrl_mid"][0]),
                float(arrays["l_end"][0]),
            ],
            dtype=float,
        )
        semantic_control_x, semantic_control_y = xy_from_sl(problem, semantic_control_s, semantic_control_l)
        metadata = {
            "model": self.name,
            "parameterization": "frenet_structured_bspline_l_ctrl_mid_l_end_v_end",
            "fixed_horizon": float(problem.horizon),
            "theta": np.asarray(parameters, dtype=float).copy(),
            "semantic_control_t": semantic_control_t,
            "semantic_control_s": semantic_control_s,
            "semantic_control_l": semantic_control_l,
            "semantic_control_x": None if semantic_control_x is None else np.asarray(semantic_control_x, dtype=float),
            "semantic_control_y": None if semantic_control_y is None else np.asarray(semantic_control_y, dtype=float),
            "degree": int(self._degree()),
            "num_control_points": int(self._num_control_points()),
            "bspline_control_u": control_u,
            "bspline_control_t": control_t,
            "bspline_control_s": control_s,
            "bspline_control_l": control_l,
            "bspline_control_v": np.asarray(arrays["v_ctrl"][0], dtype=float),
            "bspline_control_x": None if control_x is None else np.asarray(control_x, dtype=float),
            "bspline_control_y": None if control_y is None else np.asarray(control_y, dtype=float),
        }
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
        dim = self.parameter_dim(problem)
        if theta.ndim != 2 or theta.shape[1] != dim:
            raise ValueError(f"{self.name} expects theta batch shape (B, {dim}), got {theta.shape}")

        low, high = self.bounds(problem)
        theta_clipped = np.clip(theta, low[None, :], high[None, :])
        n_ctrl = self._num_control_points()
        t = fixed_time_grid(problem.horizon, problem.dt)
        basis = self._basis_matrix(n_ctrl, self._degree(), t / max(float(t[-1]), 1e-6))

        l_ctrl_mid = theta_clipped[:, 0]
        l_end = theta_clipped[:, 1]
        v_end = theta_clipped[:, 2]
        ego_speed = max(float(problem.ego.s_v), 0.0)
        v_ctrl = np.column_stack(
            [
                np.full_like(v_end, ego_speed),
                0.5 * ego_speed + 0.5 * v_end,
                v_end,
            ]
        )
        l_ctrl = np.column_stack(
            [
                np.full_like(l_end, float(problem.ego.l)),
                l_ctrl_mid,
                l_end,
            ]
        )

        s_v = np.maximum(v_ctrl @ basis.T, 0.0)
        l = l_ctrl @ basis.T
        s = float(problem.ego.s) + self._integrate_speed_batch(s_v, t)
        l_v = self._gradient_batch(l, t)
        s_a = self._gradient_batch(s_v, t)
        l_a = self._gradient_batch(l_v, t)
        kappa = self._frenet_lateral_curvature_batch(s, l, t)
        return {
            "model": self.name,
            "t": t,
            "theta": theta.copy(),
            "theta_clipped": theta_clipped,
            "l_ctrl_mid": l_ctrl_mid,
            "l_end": l_end,
            "v_end": v_end,
            "v_ctrl": v_ctrl,
            "l_ctrl": l_ctrl,
            "s": s,
            "l": l,
            "s_v": s_v,
            "l_v": l_v,
            "s_a": s_a,
            "l_a": l_a,
            "v": np.hypot(s_v, l_v),
            "a": np.hypot(s_a, l_a),
            "x": None,
            "y": None,
            "yaw": None,
            "kappa": kappa,
        }

    def _num_control_points(self) -> int:
        return 3

    def _degree(self) -> int:
        return 2

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
    def _integrate_speed_batch(speed: np.ndarray, t: np.ndarray) -> np.ndarray:
        speed = np.asarray(speed, dtype=float)
        t = np.asarray(t, dtype=float)
        if speed.shape[1] <= 1:
            return np.zeros_like(speed)
        dt = np.diff(t)
        increments = 0.5 * (speed[:, :-1] + speed[:, 1:]) * dt[None, :]
        return np.column_stack([np.zeros((speed.shape[0],), dtype=float), np.cumsum(increments, axis=1)])

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

    @classmethod
    def _basis_matrix(cls, num_control_points: int, degree: int, u_values: np.ndarray) -> np.ndarray:
        knots = cls._clamped_uniform_knots(num_control_points, degree)
        u = np.asarray(u_values, dtype=float).reshape(-1)
        basis = np.zeros((u.size, num_control_points), dtype=float)
        for i in range(num_control_points):
            basis[:, i] = cls._basis_function(i, degree, knots, u)
        if u.size:
            basis[u >= 1.0 - 1e-12, :] = 0.0
            basis[u >= 1.0 - 1e-12, num_control_points - 1] = 1.0
        row_sum = np.sum(basis, axis=1)
        valid = row_sum > 1e-12
        basis[valid] /= row_sum[valid, None]
        return basis

    @classmethod
    def _control_abscissae(cls, num_control_points: int, degree: int) -> np.ndarray:
        """Return Greville abscissae for plotting spline control points."""

        n = int(num_control_points)
        p = max(int(degree), 1)
        knots = cls._clamped_uniform_knots(n, p)
        values = np.zeros((n,), dtype=float)
        for i in range(n):
            values[i] = float(np.mean(knots[i + 1 : i + p + 1]))
        if n:
            values[0] = 0.0
            values[-1] = 1.0
        return np.clip(values, 0.0, 1.0)

    @staticmethod
    def _clamped_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
        n = int(num_control_points)
        p = int(degree)
        internal_count = n - p - 1
        if internal_count <= 0:
            internal = np.empty((0,), dtype=float)
        else:
            internal = np.linspace(0.0, 1.0, num=internal_count + 2, dtype=float)[1:-1]
        return np.concatenate([np.zeros((p + 1,), dtype=float), internal, np.ones((p + 1,), dtype=float)])

    @classmethod
    def _basis_function(cls, i: int, degree: int, knots: np.ndarray, u: np.ndarray) -> np.ndarray:
        if degree == 0:
            return ((knots[i] <= u) & (u < knots[i + 1])).astype(float)
        left_den = knots[i + degree] - knots[i]
        right_den = knots[i + degree + 1] - knots[i + 1]
        left = np.zeros_like(u, dtype=float)
        right = np.zeros_like(u, dtype=float)
        if left_den > 1e-12:
            left = ((u - knots[i]) / left_den) * cls._basis_function(i, degree - 1, knots, u)
        if right_den > 1e-12:
            right = ((knots[i + degree + 1] - u) / right_den) * cls._basis_function(i + 1, degree - 1, knots, u)
        return left + right

    @staticmethod
    def _target_l(problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("reference_l", "target_lane_l", "target_l", "preferred_l"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        return float(problem.ego.l)

    def _target_speed(self, problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    return min(float(metadata[key]), float(self.config.max_speed))
                except (TypeError, ValueError):
                    pass
        return min(max(float(problem.ego.s_v), float(self.config.min_speed)), float(self.config.max_speed))
