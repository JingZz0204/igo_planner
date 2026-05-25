from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.scenario.base import Scenario
from spatiotemporal_joint_planner.scenario.static_nudge import SmoothReferencePath


@dataclass(frozen=True)
class LaneChangeActorSpec:
    actor_id: str
    actor_type: str
    s: float
    l: float
    length: float
    width: float


class LaneChangeScenario(Scenario):
    """Two-lane lane-change scenario with current and target lane reference lines."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        lane_width: float = 3.6,
        current_lane_l: float | None = None,
        target_lane_l: float = 0.0,
        target_speed: float = 12.0,
        route_length: float = 260.0,
        road_side_margin: float = 1.0,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.lane_width = float(lane_width)
        self.target_lane_l = float(target_lane_l)
        self.current_lane_l = float(current_lane_l) if current_lane_l is not None else -float(lane_width)
        self.target_speed = float(target_speed)
        self.route_length = float(route_length)
        self.road_side_margin = max(float(road_side_margin), 0.0)

        self.ref_waypoints = self._design_reference_line()
        self.ref_path = SmoothReferencePath(*self.ref_waypoints)
        self.current_ref_line = self.offset_curve(self.current_lane_l, ds=0.25)
        self.target_ref_line = self.offset_curve(self.target_lane_l, ds=0.25)
        self.ref_line = self.target_ref_line
        self.reference_lines = (
            {
                "name": "current_lane_reference",
                "role": "current",
                "l": self.current_lane_l,
                "line": self.current_ref_line,
            },
            {
                "name": "target_lane_reference",
                "role": "target",
                "l": self.target_lane_l,
                "line": self.target_ref_line,
            },
        )

        left_lane_l = max(self.current_lane_l, self.target_lane_l)
        right_lane_l = min(self.current_lane_l, self.target_lane_l)
        self.road_left_l = left_lane_l + 0.5 * self.lane_width + self.road_side_margin
        self.road_right_l = right_lane_l - 0.5 * self.lane_width - self.road_side_margin
        lane_separator_l = 0.5 * (self.current_lane_l + self.target_lane_l)
        self.lane_markings = (
            self.offset_curve(lane_separator_l, ds=0.25),
        )
        self.actor_specs = self._default_actors()

    @property
    def name(self) -> str:
        return "lane_change"

    def initial_state(self) -> EgoState:
        return EgoState(s=0.0, l=self.current_lane_l, s_v=self.target_speed)

    def build_problem(self, ego: EgoState, t: float = 0.0) -> PlanningProblem:
        times = np.arange(0.0, self.horizon + 0.5 * self.dt, self.dt, dtype=float)
        if times.size == 0 or times[-1] < self.horizon - 1e-9:
            times = np.concatenate([times, [self.horizon]])
        else:
            times[-1] = self.horizon

        actors = [self._actor_prediction(spec, times, float(t)) for spec in self.actor_specs]
        return PlanningProblem(
            ego=ego,
            ref_path=self.ref_path,
            road_boundary=RoadBoundary(left_l=self.road_left_l, right_l=self.road_right_l),
            horizon=self.horizon,
            dt=self.dt,
            actors=actors,
            metadata={
                "scenario": self.name,
                "maneuver": "lane_change_left" if self.current_lane_l < self.target_lane_l else "lane_change_right",
                "ref_line": self.ref_line,
                "ref_waypoints": self.ref_waypoints,
                "current_ref_line": self.current_ref_line,
                "target_ref_line": self.target_ref_line,
                "reference_lines": self.reference_lines,
                "active_reference_role": "target",
                "lane_markings": self.lane_markings,
                "lane_width": self.lane_width,
                "road_side_margin": self.road_side_margin,
                "lane_centers": (self.current_lane_l, self.target_lane_l),
                "current_lane_l": self.current_lane_l,
                "target_lane_l": self.target_lane_l,
                "reference_l": self.target_lane_l,
                "target_speed": self.target_speed,
                "source": "spatiotemporal_joint_planner.lane_change",
            },
        )

    def actors_at(self, t: float) -> list[ActorPrediction]:
        times = np.array([float(t)], dtype=float)
        return [self._actor_prediction(spec, times, float(t)) for spec in self.actor_specs]

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
        values = np.asarray(shifted, dtype=float)
        return values[:, 0].tolist(), values[:, 1].tolist()

    def _actor_prediction(self, spec: LaneChangeActorSpec, times: np.ndarray, start_time: float) -> ActorPrediction:
        x, y, yaw = self._pose_from_sl(spec.s, spec.l)
        times = np.asarray(times, dtype=float) + float(start_time)
        half_length = 0.5 * float(spec.length)
        half_width = 0.5 * float(spec.width)
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
                "static": True,
            },
        )

    def _pose_from_sl(self, s: float, l: float) -> tuple[float, float, float]:
        x_ref, y_ref = self.ref_path.calc_position(float(s))
        yaw = self.ref_path.calc_yaw(float(s))
        x = float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0)
        y = float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0)
        return x, y, yaw

    def _default_actors(self) -> tuple[LaneChangeActorSpec, ...]:
        return (
            LaneChangeActorSpec("slow_vehicle_current_lane", "vehicle", 52.0, self.current_lane_l, 5.2, 2.1),
            LaneChangeActorSpec("target_lane_lead_vehicle", "vehicle", 80.0, self.target_lane_l, 4.8, 2.0),
        )

    def _design_reference_line(self) -> tuple[list[float], list[float]]:
        x_values = [0.0, 60.0, 130.0, self.route_length]
        y_values = [0.0, 0.0, 2.5, 2.5]
        return x_values, y_values
