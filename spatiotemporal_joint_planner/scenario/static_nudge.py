from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.scenario.base import Scenario


@dataclass(frozen=True)
class StaticObstacleSpec:
    actor_id: str
    actor_type: str
    s: float
    l: float
    length: float
    width: float


class PolylineReferencePath:
    """Lightweight reference path with the methods used by trajectory models."""

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        if points.shape[0] < 2:
            raise ValueError("PolylineReferencePath requires at least two points.")
        deltas = np.diff(points, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-9
        if not np.all(valid):
            points = np.vstack([points[0], points[1:][valid]])
            deltas = np.diff(points, axis=0)
            lengths = np.linalg.norm(deltas, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("PolylineReferencePath contains degenerate segments.")

        self.x = points[:, 0]
        self.y = points[:, 1]
        self.points = points
        self.segment_lengths = lengths
        self.s = np.concatenate([[0.0], np.cumsum(lengths)])
        self.segment_yaw = np.arctan2(deltas[:, 1], deltas[:, 0])

    def calc_position(self, s_query: float) -> tuple[float, float]:
        idx, ratio = self._segment_at(s_query)
        point = self.points[idx] + ratio * (self.points[idx + 1] - self.points[idx])
        return float(point[0]), float(point[1])

    def calc_yaw(self, s_query: float) -> float:
        idx, _ = self._segment_at(s_query)
        return float(self.segment_yaw[idx])

    def _segment_at(self, s_query: float) -> tuple[int, float]:
        s_clamped = float(np.clip(float(s_query), 0.0, float(self.s[-1])))
        idx = int(np.searchsorted(self.s, s_clamped, side="right") - 1)
        idx = min(max(idx, 0), self.segment_lengths.size - 1)
        ratio = (s_clamped - float(self.s[idx])) / max(float(self.segment_lengths[idx]), 1e-9)
        return idx, float(ratio)


class CubicSpline1D:
    """Natural cubic spline for one-dimensional interpolation."""

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self.x = np.asarray(x, dtype=float)
        self.a = np.asarray(y, dtype=float)
        if self.x.ndim != 1 or self.a.ndim != 1 or self.x.size != self.a.size:
            raise ValueError("CubicSpline1D requires one-dimensional x/y arrays with the same length.")
        if self.x.size < 2:
            raise ValueError("CubicSpline1D requires at least two samples.")

        h = np.diff(self.x)
        if np.any(h <= 1e-9):
            raise ValueError("CubicSpline1D x values must be strictly increasing.")

        matrix = self._build_matrix(h)
        rhs = self._build_rhs(h)
        self.c = np.linalg.solve(matrix, rhs)
        self.b = np.empty(self.x.size - 1, dtype=float)
        self.d = np.empty(self.x.size - 1, dtype=float)
        for idx in range(self.x.size - 1):
            self.d[idx] = (self.c[idx + 1] - self.c[idx]) / (3.0 * h[idx])
            self.b[idx] = (self.a[idx + 1] - self.a[idx]) / h[idx] - h[idx] * (
                self.c[idx + 1] + 2.0 * self.c[idx]
            ) / 3.0

    def calc(self, t: float) -> float:
        idx, dx = self._index_and_dx(t)
        return float(self.a[idx] + self.b[idx] * dx + self.c[idx] * dx**2 + self.d[idx] * dx**3)

    def calcd(self, t: float) -> float:
        idx, dx = self._index_and_dx(t)
        return float(self.b[idx] + 2.0 * self.c[idx] * dx + 3.0 * self.d[idx] * dx**2)

    def calcdd(self, t: float) -> float:
        idx, dx = self._index_and_dx(t)
        return float(2.0 * self.c[idx] + 6.0 * self.d[idx] * dx)

    def _index_and_dx(self, t: float) -> tuple[int, float]:
        t_clamped = float(np.clip(float(t), float(self.x[0]), float(self.x[-1])))
        idx = int(np.searchsorted(self.x, t_clamped, side="right") - 1)
        idx = min(max(idx, 0), self.x.size - 2)
        return idx, float(t_clamped - self.x[idx])

    def _build_matrix(self, h: np.ndarray) -> np.ndarray:
        size = self.x.size
        matrix = np.zeros((size, size), dtype=float)
        matrix[0, 0] = 1.0
        matrix[-1, -1] = 1.0
        for idx in range(size - 2):
            matrix[idx + 1, idx] = h[idx]
            matrix[idx + 1, idx + 1] = 2.0 * (h[idx] + h[idx + 1])
            matrix[idx + 1, idx + 2] = h[idx + 1]
        return matrix

    def _build_rhs(self, h: np.ndarray) -> np.ndarray:
        rhs = np.zeros(self.x.size, dtype=float)
        for idx in range(self.x.size - 2):
            rhs[idx + 1] = 3.0 * (self.a[idx + 2] - self.a[idx + 1]) / h[idx + 1] - 3.0 * (
                self.a[idx + 1] - self.a[idx]
            ) / h[idx]
        return rhs


class SmoothReferencePath:
    """Spline reference path with continuous position, yaw, and curvature."""

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        if points.shape[0] < 2:
            raise ValueError("SmoothReferencePath requires at least two points.")

        deltas = np.diff(points, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-9
        if not np.all(valid):
            points = np.vstack([points[0], points[1:][valid]])
            deltas = np.diff(points, axis=0)
            lengths = np.linalg.norm(deltas, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("SmoothReferencePath contains degenerate segments.")

        self.x = points[:, 0]
        self.y = points[:, 1]
        self.points = points
        self.s = np.concatenate([[0.0], np.cumsum(lengths)])
        self.sx = CubicSpline1D(self.s, self.x)
        self.sy = CubicSpline1D(self.s, self.y)

    def calc_position(self, s_query: float) -> tuple[float, float]:
        s_clamped = self._clamp_s(s_query)
        return self.sx.calc(s_clamped), self.sy.calc(s_clamped)

    def calc_yaw(self, s_query: float) -> float:
        s_clamped = self._clamp_s(s_query)
        return float(math.atan2(self.sy.calcd(s_clamped), self.sx.calcd(s_clamped)))

    def calc_curvature(self, s_query: float) -> float:
        s_clamped = self._clamp_s(s_query)
        dx = self.sx.calcd(s_clamped)
        dy = self.sy.calcd(s_clamped)
        ddx = self.sx.calcdd(s_clamped)
        ddy = self.sy.calcdd(s_clamped)
        denom = max((dx * dx + dy * dy) ** 1.5, 1e-9)
        return float((dx * ddy - dy * ddx) / denom)

    def sample_xy(self, ds: float = 0.25) -> tuple[list[float], list[float]]:
        points = [self.calc_position(float(s)) for s in self.sample_s(ds)]
        values = np.asarray(points, dtype=float)
        return values[:, 0].tolist(), values[:, 1].tolist()

    def sample_s(self, ds: float = 0.25) -> np.ndarray:
        step = max(float(ds), 1e-3)
        values = np.arange(0.0, float(self.s[-1]) + 0.5 * step, step, dtype=float)
        if values.size == 0 or values[-1] < float(self.s[-1]) - 1e-9:
            values = np.concatenate([values, [float(self.s[-1])]])
        else:
            values[-1] = float(self.s[-1])
        return values

    def _clamp_s(self, s_query: float) -> float:
        return float(np.clip(float(s_query), 0.0, float(self.s[-1])))


class StaticNudgeScenario(Scenario):
    """Static obstacle nudge scenario based on the old static_bypass setup."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        road_width: float = 8.0,
        lane_width: float = 3.6,
        default_start_l: float = 2.0,
        target_speed: float = 30.0 / 3.6,
        obstacle_specs: Sequence[StaticObstacleSpec] | None = None,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.road_width = float(road_width)
        self.lane_width = float(lane_width)
        self.default_start_l = float(default_start_l)
        self.target_speed = float(target_speed)
        self.ref_waypoints = self._design_reference_line()
        self.ref_path = SmoothReferencePath(*self.ref_waypoints)
        self.ref_line = self.ref_path.sample_xy(ds=0.25)
        self.lane_markings = (
            self.offset_curve(self.lane_width, ds=0.25),
            self.offset_curve(-self.lane_width, ds=0.25),
        )
        self.obstacle_specs = tuple(obstacle_specs) if obstacle_specs is not None else self._default_obstacles()

    @property
    def name(self) -> str:
        return "static_nudge"

    def initial_state(self) -> EgoState:
        return EgoState(s=0.0, l=self.default_start_l, s_v=self.target_speed)

    def build_problem(self, ego: EgoState, t: float = 0.0) -> PlanningProblem:
        times = np.arange(0.0, self.horizon + 0.5 * self.dt, self.dt, dtype=float)
        if times.size == 0 or times[-1] < self.horizon - 1e-9:
            times = np.concatenate([times, [self.horizon]])
        else:
            times[-1] = self.horizon

        actors = [self._actor_prediction(spec, times, float(t)) for spec in self.obstacle_specs]
        return PlanningProblem(
            ego=ego,
            ref_path=self.ref_path,
            road_boundary=RoadBoundary(left_l=self.road_width, right_l=-self.road_width),
            horizon=self.horizon,
            dt=self.dt,
            actors=actors,
            metadata={
                "scenario": self.name,
                "ref_line": self.ref_line,
                "ref_waypoints": self.ref_waypoints,
                "lane_markings": self.lane_markings,
                "road_width": self.road_width,
                "target_speed": self.target_speed,
                "reference_l": 0.0,
                "source": "gmm_nva_path_planner.static_bypass",
            },
        )

    def actors_at(self, t: float) -> list[ActorPrediction]:
        times = np.array([0.0], dtype=float)
        return [self._actor_prediction(spec, times, float(t)) for spec in self.obstacle_specs]

    def _actor_prediction(self, spec: StaticObstacleSpec, times: np.ndarray, start_time: float) -> ActorPrediction:
        x, y, yaw = self._pose_from_sl(spec.s, spec.l)
        relative_times = np.asarray(times, dtype=float)
        times = relative_times + float(start_time)
        half_length = 0.5 * float(spec.length)
        half_width = 0.5 * float(spec.width)
        temporal_blocked_range = {
            "t": relative_times,
            "s_min": np.full(relative_times.shape, float(spec.s) - half_length, dtype=float),
            "s_max": np.full(relative_times.shape, float(spec.s) + half_length, dtype=float),
            "l_min": np.full(relative_times.shape, float(spec.l) - half_width, dtype=float),
            "l_max": np.full(relative_times.shape, float(spec.l) + half_width, dtype=float),
        }
        return ActorPrediction(
            actor_id=spec.actor_id,
            actor_type=spec.actor_type,
            times=times,
            x=np.full(times.shape, x, dtype=float),
            y=np.full(times.shape, y, dtype=float),
            yaw=np.full(times.shape, yaw, dtype=float),
            length=float(spec.length),
            width=float(spec.width),
            metadata={
                "s": float(spec.s),
                "l": float(spec.l),
                "blocked_s_min": float(spec.s) - half_length,
                "blocked_s_max": float(spec.s) + half_length,
                "blocked_l_min": float(spec.l) - half_width,
                "blocked_l_max": float(spec.l) + half_width,
                "temporal_blocked_range": temporal_blocked_range,
                "static": True,
            },
        )

    def _pose_from_sl(self, s: float, l: float) -> tuple[float, float, float]:
        x_ref, y_ref = self.ref_path.calc_position(float(s))
        yaw = self.ref_path.calc_yaw(float(s))
        x = float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0)
        y = float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0)
        return x, y, yaw

    def offset_curve(self, offset: float, ds: float = 0.25) -> tuple[list[float], list[float]]:
        shifted = []
        for s in self.ref_path.sample_s(ds):
            x_ref, y_ref = self.ref_path.calc_position(float(s))
            yaw = self.ref_path.calc_yaw(float(s))
            shifted.append(
                [
                    float(x_ref) + float(offset) * math.cos(yaw + math.pi / 2.0),
                    float(y_ref) + float(offset) * math.sin(yaw + math.pi / 2.0),
                ]
            )
        shifted_array = np.asarray(shifted, dtype=float)
        return shifted_array[:, 0].tolist(), shifted_array[:, 1].tolist()

    def _default_obstacles(self) -> tuple[StaticObstacleSpec, ...]:
        # return [StaticObstacleSpec("stopped_box_truck", "vehicle", 58.0, 0.0, 7.0, 2.4)]
        return (
            StaticObstacleSpec("stopped_box_truck", "vehicle", 58.0, 0.0, 7.0, 2.4),
            StaticObstacleSpec("service_van", "vehicle", 68.0, -3.8, 5.2, 2.1),
            StaticObstacleSpec("road_worker", "pedestrian", 64.0, -5.4, 0.8, 0.8),
            StaticObstacleSpec("parked_suv_curve", "vehicle", 128.0, 3.2, 4.8, 2.0),
            StaticObstacleSpec("maintenance_truck", "vehicle", 176.0, -3.1, 6.4, 2.3),
            StaticObstacleSpec("lane_blocker_curve", "vehicle", 87.0, 0.0, 4.7, 2.0),
            StaticObstacleSpec("parked_sedan_outer", "vehicle", 110.0, 4.2, 4.6, 1.9),
            StaticObstacleSpec("disabled_hatchback", "vehicle", 148.0, -3.4, 4.3, 1.8),
            StaticObstacleSpec("roadside_pickup", "vehicle", 206.0, 4.8, 5.0, 2.0),
            StaticObstacleSpec("stalled_suv_return", "vehicle", 224.0, -5.0, 4.9, 2.1),
            StaticObstacleSpec("stalled_suv_return", "vehicle", 224.0, 5.0, 4.9, 2.1),
            StaticObstacleSpec("support_van_return", "vehicle", 240.0, 0.0, 5.3, 2.1),
        )

    @staticmethod
    def _design_reference_line() -> tuple[list[float], list[float]]:
        rx, ry = [], []
        step_curve = 0.1 * math.pi
        step_line = 4

        cx, cy, cr = 30, 30, 20
        for theta in np.arange(math.pi, math.pi * 1.5, step_curve):
            rx.append(cx + cr * math.cos(theta))
            ry.append(cy + cr * math.sin(theta))

        for ix in np.arange(30, 80, step_line):
            rx.append(float(ix))
            ry.append(10.0)

        cx, cy, cr = 80, 25, 15
        for theta in np.arange(-math.pi / 2.0, math.pi / 2.0, step_curve):
            rx.append(cx + cr * math.cos(theta))
            ry.append(cy + cr * math.sin(theta))

        for ix in np.arange(80, 60, -step_line):
            rx.append(float(ix))
            ry.append(40.0)

        cx, cy, cr = 60, 60, 20
        for theta in np.arange(-math.pi / 2.0, -math.pi, -step_curve):
            rx.append(cx + cr * math.cos(theta))
            ry.append(cy + cr * math.sin(theta))

        cx, cy, cr = 25, 60, 15
        for theta in np.arange(0.0, math.pi, step_curve):
            rx.append(cx + cr * math.cos(theta))
            ry.append(cy + cr * math.sin(theta))

        for iy in np.arange(60, 30, -step_line):
            rx.append(10.0)
            ry.append(float(iy))

        return rx, ry

    @staticmethod
    def _offset_polyline(ref_line: tuple[Sequence[float], Sequence[float]], offset: float) -> tuple[list[float], list[float]]:
        x_values, y_values = ref_line
        points = np.column_stack([np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)])
        shifted = []
        for idx, point in enumerate(points):
            if idx == 0:
                tangent = points[1] - point
            elif idx == len(points) - 1:
                tangent = point - points[idx - 1]
            else:
                tangent = points[idx + 1] - points[idx - 1]
            yaw = math.atan2(float(tangent[1]), float(tangent[0]))
            shifted.append(
                [
                    float(point[0] + float(offset) * math.cos(yaw + math.pi / 2.0)),
                    float(point[1] + float(offset) * math.sin(yaw + math.pi / 2.0)),
                ]
            )
        shifted_array = np.asarray(shifted, dtype=float)
        return shifted_array[:, 0].tolist(), shifted_array[:, 1].tolist()
