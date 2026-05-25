from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.common import (
    curvature_from_xy,
    finite_difference,
    fixed_time_grid,
    project_xy_to_sl,
)


@dataclass(frozen=True)
class BezierTrajectoryConfig:
    path_length: float = 50.0
    bezier_order: int = 7
    fixed_start_controls: int = 2
    start_tangent_length: float = 4.0
    xy_longitudinal_margin: float = 10.0
    xy_lateral_margin: float = 8.0
    projection_ds: float = 0.25


class BezierTrajectoryModel(TrajectoryModel):
    """Fixed-horizon ego-local XY Bezier trajectory.

    The horizon comes from PlanningProblem.horizon. Theta only contains the
    free Bezier control points P_fixed..P_order, matching the existing
    bezier_l_path behavior.
    """

    def __init__(self, config: Optional[BezierTrajectoryConfig] = None):
        self.config = config or BezierTrajectoryConfig()

    @property
    def name(self) -> str:
        return "bezier_trajectory"

    def parameter_dim(self, problem: PlanningProblem) -> int:
        degree = self._degree()
        fixed = self._fixed_controls()
        return 2 * (degree + 1 - fixed)

    def bounds(self, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        nominal = self.reference_parameters(problem)
        nominal_xy = nominal.reshape(-1, 2)
        margin = np.array(
            [
                max(float(self.config.xy_longitudinal_margin), 1.0),
                max(float(self.config.xy_lateral_margin), 1.0),
            ],
            dtype=float,
        )
        low = nominal_xy - margin
        high = nominal_xy + margin
        return low.reshape(-1), high.reshape(-1)

    def reference_parameters(self, problem: PlanningProblem) -> np.ndarray:
        pose = self._start_pose(problem)
        x0, y0, yaw0 = pose
        control_s = self._control_s()
        world_points = []
        ref_path = problem.ref_path
        if hasattr(ref_path, "calc_position") and hasattr(ref_path, "calc_yaw"):
            route_end = self._route_end_s(ref_path)
            for rel_s in control_s:
                abs_s = float(np.clip(problem.ego.s + float(rel_s), 0.0, route_end))
                xy_ref = ref_path.calc_position(abs_s)
                if xy_ref is None or xy_ref[0] is None or xy_ref[1] is None:
                    return np.column_stack([control_s, np.zeros_like(control_s)]).reshape(-1)
                world_points.append([float(xy_ref[0]), float(xy_ref[1])])
        else:
            world_points = np.column_stack([problem.ego.s + control_s, np.full_like(control_s, problem.ego.l)]).tolist()

        local_points = self._world_to_local(np.asarray(world_points, dtype=float), x0, y0, yaw0)
        return local_points.reshape(-1)

    def decode(self, parameters: np.ndarray, problem: PlanningProblem) -> Trajectory:
        theta = np.asarray(parameters, dtype=float)
        expected = self.parameter_dim(problem)
        if theta.shape != (expected,):
            raise ValueError(f"{self.name} expects theta shape ({expected},), got {theta.shape}")

        control_local, x0, y0, yaw0 = self._build_control_points(theta, problem)
        t = fixed_time_grid(problem.horizon, problem.dt)
        horizon = max(float(problem.horizon), 1e-3)
        u = np.clip(t / horizon, 0.0, 1.0)
        local_points = self._bezier_values(control_local, u)
        world_points = self._local_to_world(local_points, x0, y0, yaw0)
        d1_local = self._bezier_first_derivative(control_local, u)
        d2_local = self._bezier_second_derivative(control_local, u)
        d1_world = self._rotate_local_vector(d1_local, yaw0)
        d2_world = self._rotate_local_vector(d2_local, yaw0)
        velocity_xy = d1_world / horizon
        accel_xy = d2_world / (horizon * horizon)
        speed = np.linalg.norm(velocity_xy, axis=1)
        accel = np.linalg.norm(accel_xy, axis=1)
        yaw = np.arctan2(d1_world[:, 1], d1_world[:, 0])
        kappa = curvature_from_xy(world_points[:, 0], world_points[:, 1], t)
        s, l = project_xy_to_sl(problem, world_points[:, 0], world_points[:, 1], self.config.projection_ds)
        s_v = finite_difference(s, t)
        l_v = finite_difference(l, t)
        s_a = finite_difference(s_v, t)
        l_a = finite_difference(l_v, t)
        return Trajectory(
            t=t,
            s=s,
            l=l,
            s_v=s_v,
            l_v=l_v,
            s_a=s_a,
            l_a=l_a,
            x=world_points[:, 0],
            y=world_points[:, 1],
            yaw=yaw,
            v=speed,
            a=accel,
            kappa=kappa,
            metadata={
                "model": self.name,
                "parameterization": "ego_local_xy_bezier",
                "fixed_horizon": float(problem.horizon),
                "theta": theta.copy(),
            },
        )

    def _degree(self) -> int:
        return max(int(self.config.bezier_order), 1)

    def _fixed_controls(self) -> int:
        degree = self._degree()
        return int(np.clip(int(self.config.fixed_start_controls), 1, degree))

    def _control_s(self) -> np.ndarray:
        degree = self._degree()
        fixed = self._fixed_controls()
        indices = np.arange(fixed, degree + 1, dtype=float)
        return float(self.config.path_length) * indices / float(degree)

    def _build_control_points(self, theta: np.ndarray, problem: PlanningProblem):
        x0, y0, yaw0 = self._start_pose(problem)
        degree = self._degree()
        fixed = self._fixed_controls()
        control = np.zeros((degree + 1, 2), dtype=float)
        control[0] = [0.0, 0.0]
        if fixed >= 2:
            speed0 = max(float(problem.ego.s_v), 0.0)
            tangent_len = speed0 * float(problem.horizon) / max(float(degree), 1.0)
            control[1] = [max(tangent_len, float(self.config.start_tangent_length), 0.2), 0.0]
        opt_points = theta.reshape(-1, 2)
        control[fixed:] = opt_points
        if fixed == 1:
            control[1:] = opt_points
        return control, x0, y0, yaw0

    def _start_pose(self, problem: PlanningProblem) -> tuple[float, float, float]:
        ref_path = problem.ref_path
        if not hasattr(ref_path, "calc_position") or not hasattr(ref_path, "calc_yaw"):
            return float(problem.ego.s), float(problem.ego.l), float(problem.ego.yaw or 0.0)

        route_end = self._route_end_s(ref_path)
        s0 = float(np.clip(problem.ego.s, 0.0, route_end))
        xy_ref = ref_path.calc_position(s0)
        if xy_ref is None or xy_ref[0] is None or xy_ref[1] is None:
            return float(problem.ego.s), float(problem.ego.l), float(problem.ego.yaw or 0.0)

        yaw_ref = float(ref_path.calc_yaw(s0))
        x0 = float(xy_ref[0]) + float(problem.ego.l) * math.cos(yaw_ref + math.pi / 2.0)
        y0 = float(xy_ref[1]) + float(problem.ego.l) * math.sin(yaw_ref + math.pi / 2.0)
        if problem.ego.yaw is not None:
            yaw0 = float(problem.ego.yaw)
        else:
            s_speed = max(abs(float(problem.ego.s_v)), 1e-3)
            start_slope = float(np.clip(float(problem.ego.l_v) / s_speed, -0.8, 0.8))
            yaw0 = yaw_ref + math.atan(start_slope)
        return x0, y0, yaw0

    @staticmethod
    def _bernstein_basis(degree: int, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        basis = np.empty((u.size, degree + 1), dtype=float)
        one_minus_u = 1.0 - u
        for i in range(degree + 1):
            basis[:, i] = math.comb(degree, i) * (one_minus_u ** (degree - i)) * (u**i)
        return basis

    @classmethod
    def _bezier_values(cls, control: np.ndarray, u: np.ndarray) -> np.ndarray:
        degree = int(control.shape[0] - 1)
        return cls._bernstein_basis(degree, u) @ control

    @classmethod
    def _bezier_first_derivative(cls, control: np.ndarray, u: np.ndarray) -> np.ndarray:
        degree = int(control.shape[0] - 1)
        if degree <= 0:
            return np.zeros((np.asarray(u).size, control.shape[1]), dtype=float)
        return cls._bezier_values(degree * np.diff(control, axis=0), u)

    @classmethod
    def _bezier_second_derivative(cls, control: np.ndarray, u: np.ndarray) -> np.ndarray:
        degree = int(control.shape[0] - 1)
        if degree <= 1:
            return np.zeros((np.asarray(u).size, control.shape[1]), dtype=float)
        return cls._bezier_values(degree * (degree - 1) * np.diff(control, n=2, axis=0), u)

    @staticmethod
    def _local_to_world(local_points: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
        c = math.cos(float(yaw0))
        s = math.sin(float(yaw0))
        x = float(x0) + local_points[:, 0] * c - local_points[:, 1] * s
        y = float(y0) + local_points[:, 0] * s + local_points[:, 1] * c
        return np.column_stack([x, y])

    @staticmethod
    def _world_to_local(world_points: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
        dx = world_points[:, 0] - float(x0)
        dy = world_points[:, 1] - float(y0)
        c = math.cos(float(yaw0))
        s = math.sin(float(yaw0))
        x = dx * c + dy * s
        y = -dx * s + dy * c
        return np.column_stack([x, y])

    @staticmethod
    def _rotate_local_vector(local_vectors: np.ndarray, yaw0: float) -> np.ndarray:
        c = math.cos(float(yaw0))
        s = math.sin(float(yaw0))
        x = local_vectors[:, 0] * c - local_vectors[:, 1] * s
        y = local_vectors[:, 0] * s + local_vectors[:, 1] * c
        return np.column_stack([x, y])

    @staticmethod
    def _route_end_s(ref_path) -> float:
        if hasattr(ref_path, "s"):
            values = np.asarray(ref_path.s, dtype=float)
            if values.size:
                return max(float(values[-1]), 1e-3)
        return 1.0e6
