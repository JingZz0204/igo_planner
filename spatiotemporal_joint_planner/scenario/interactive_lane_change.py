from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.scenario.base import Scenario
from spatiotemporal_joint_planner.scenario.static_nudge import SmoothReferencePath


@dataclass(frozen=True)
class InteractiveLaneChangeActorSpec:
    actor_id: str
    actor_type: str
    s: float
    l: float
    length: float
    width: float
    s_v: float
    s_a: float = 0.0
    l_v: float = 0.0
    l_a: float = 0.0


class InteractiveLaneChangeScenario(Scenario):
    """Lane-change execution scenario after target-lane reference selection."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        lane_width: float = 3.6,
        current_lane_l: float | None = None,
        target_lane_l: float = 0.0,
        ego_speed: float = 9.0,
        target_speed: float = 12.0,
        route_length: float = 520.0,
        road_side_margin: float = 1.0,
        interaction_mode: str = "keep",
        target_lead_s: float = 45.0,
        target_lead_v: float = 8.5,
        target_rear_s: float = -12.0,
        target_rear_v: float = 12.0,
        current_slow_s: float = 24.0,
        current_slow_v: float = 4.5,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.lane_width = float(lane_width)
        self.current_lane_l = float(current_lane_l) if current_lane_l is not None else -float(lane_width)
        self.target_lane_l = float(target_lane_l)
        self.ego_speed = float(ego_speed)
        self.target_speed = float(target_speed)
        self.route_length = float(route_length)
        self.road_side_margin = max(float(road_side_margin), 0.0)
        self.interaction_mode = str(interaction_mode).lower()
        self.target_lead_s = float(target_lead_s)
        self.target_lead_v = float(target_lead_v)
        self.target_rear_s = float(target_rear_s)
        self.target_rear_v = float(target_rear_v)
        self.current_slow_s = float(current_slow_s)
        self.current_slow_v = float(current_slow_v)

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
        self.lane_markings = (self.offset_curve(lane_separator_l, ds=0.25),)
        self.actor_specs = self._default_actors()

    @property
    def name(self) -> str:
        return "interactive_lane_change"

    def initial_state(self) -> EgoState:
        return EgoState(s=0.0, l=self.current_lane_l, s_v=self.ego_speed)

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
                "interaction_mode": self.interaction_mode,
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
                "current_slow_s": self.current_slow_s,
                "current_slow_v": self.current_slow_v,
                "source": "spatiotemporal_joint_planner.interactive_lane_change",
            },
        )

    def actors_at(self, t: float) -> list[ActorPrediction]:
        times = np.array([0.0], dtype=float)
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

    def _actor_prediction(
        self,
        spec: InteractiveLaneChangeActorSpec,
        relative_times: np.ndarray,
        start_time: float,
    ) -> ActorPrediction:
        relative_times = np.asarray(relative_times, dtype=float)
        times = relative_times + float(start_time)
        s_values, s_v_values = self._rollout_axis(
            p0=float(spec.s),
            v0=float(spec.s_v),
            a=float(spec.s_a),
            start_time=float(start_time),
            relative_times=relative_times,
            min_velocity=0.0,
        )
        l_values, l_v_values = self._rollout_axis(
            p0=float(spec.l),
            v0=float(spec.l_v),
            a=float(spec.l_a),
            start_time=float(start_time),
            relative_times=relative_times,
            min_velocity=None,
        )
        del s_v_values, l_v_values

        poses = np.asarray([self._pose_from_sl(float(s), float(l)) for s, l in zip(s_values, l_values)], dtype=float)
        half_length = 0.5 * float(spec.length)
        half_width = 0.5 * float(spec.width)
        temporal_blocked_range = {
            "t": relative_times,
            "s_min": s_values - half_length,
            "s_max": s_values + half_length,
            "l_min": l_values - half_width,
            "l_max": l_values + half_width,
        }
        return ActorPrediction(
            actor_id=spec.actor_id,
            actor_type=spec.actor_type,
            times=times,
            x=poses[:, 0],
            y=poses[:, 1],
            yaw=poses[:, 2],
            length=float(spec.length),
            width=float(spec.width),
            metadata={
                "s": float(s_values[0]),
                "l": float(l_values[0]),
                "s_v": float(spec.s_v),
                "s_a": float(spec.s_a),
                "l_v": float(spec.l_v),
                "l_a": float(spec.l_a),
                "blocked_s_min": float(s_values[0]) - half_length,
                "blocked_s_max": float(s_values[0]) + half_length,
                "blocked_l_min": float(l_values[0]) - half_width,
                "blocked_l_max": float(l_values[0]) + half_width,
                "temporal_blocked_range": temporal_blocked_range,
                "static": False,
            },
        )

    def _pose_from_sl(self, s: float, l: float) -> tuple[float, float, float]:
        x_ref, y_ref = self.ref_path.calc_position(float(s))
        yaw = self.ref_path.calc_yaw(float(s))
        x = float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0)
        y = float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0)
        return x, y, yaw

    def _default_actors(self) -> tuple[InteractiveLaneChangeActorSpec, ...]:
        rear_accel_by_mode = {
            "yield": -1.5,
            "keep": 0.0,
            "block": 0.8,
        }
        rear_accel = rear_accel_by_mode.get(self.interaction_mode, 0.0)
        return (
            InteractiveLaneChangeActorSpec(
                actor_id="current_lane_slow_vehicle",
                actor_type="vehicle",
                s=self.current_slow_s,
                l=self.current_lane_l,
                length=5.0,
                width=2.0,
                s_v=self.current_slow_v,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_lead_vehicle",
                actor_type="vehicle",
                s=self.target_lead_s,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=self.target_lead_v,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_rear_vehicle",
                actor_type="vehicle",
                s=self.target_rear_s,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=self.target_rear_v,
                s_a=rear_accel,
            ),
        )

    @staticmethod
    def _rollout_axis(
        p0: float,
        v0: float,
        a: float,
        start_time: float,
        relative_times: np.ndarray,
        min_velocity: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_times = float(start_time) + np.asarray(relative_times, dtype=float)
        if min_velocity is None or float(a) >= 0.0:
            position = float(p0) + float(v0) * query_times + 0.5 * float(a) * query_times**2
            velocity = float(v0) + float(a) * query_times
            return position, velocity

        min_v = float(min_velocity)
        if float(v0) <= min_v:
            return float(p0) + min_v * query_times, np.full_like(query_times, min_v, dtype=float)

        stop_time = max((min_v - float(v0)) / float(a), 0.0)
        stop_position = float(p0) + float(v0) * stop_time + 0.5 * float(a) * stop_time**2
        before_stop = query_times <= stop_time
        free_position = float(p0) + float(v0) * query_times + 0.5 * float(a) * query_times**2
        position = np.where(before_stop, free_position, stop_position + min_v * (query_times - stop_time))
        velocity = np.where(before_stop, float(v0) + float(a) * query_times, min_v)
        return position, velocity

    def _design_reference_line(self) -> tuple[list[float], list[float]]:
        route_length = max(float(self.route_length), 320.0)
        x_values = [0.0, 60.0, 130.0, 220.0, 320.0]
        y_values = [0.0, 0.0, 2.5, 2.5, 5.0]
        if route_length > x_values[-1] + 1e-6:
            x_values.append(route_length)
            y_values.append(y_values[-1])
        return x_values, y_values
