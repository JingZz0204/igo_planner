from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.scenario.base import Scenario
from spatiotemporal_joint_planner.scenario.static_nudge import SmoothReferencePath


@dataclass(frozen=True)
class LeftTurnActorSpec:
    actor_id: str
    route_id: str
    s: float
    speed: float
    acceleration: float = 0.0
    length: float = 4.8
    width: float = 2.0


class UnprotectedLeftTurnScenario(Scenario):
    """Ego unprotected left turn against oncoming and left-crossing traffic."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        route_extent: float = 55.0,
        lane_offset: float = 1.8,
        road_half_width: float = 4.2,
        turn_entry_y: float = -5.0,
        ego_s: float = 18.0,
        ego_speed: float = 8.0,
        target_speed: float = 9.0,
        oncoming_s: float = 28.0,
        oncoming_speed: float = 8.0,
        left_crossing_s: float = 28.0,
        left_crossing_speed: float = 7.5,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.route_extent = max(float(route_extent), 35.0)
        self.lane_offset = float(lane_offset)
        self.road_half_width = max(float(road_half_width), 3.0)
        self.turn_entry_y = min(float(turn_entry_y), -2.5)
        self.ego_s = float(ego_s)
        self.ego_speed = float(ego_speed)
        self.target_speed = float(target_speed)

        self.route_paths = {
            "ego_left_turn": self._left_turn_path(),
            "oncoming_straight": SmoothReferencePath(
                [-self.lane_offset, -self.lane_offset],
                [self.route_extent, -self.route_extent],
            ),
            "left_crossing_straight": SmoothReferencePath(
                [-self.route_extent, self.route_extent],
                [-self.lane_offset, -self.lane_offset],
            ),
        }
        self.ref_path = self.route_paths["ego_left_turn"]
        self.ref_line = self.ref_path.sample_xy(0.2)
        self.reference_lines = (
            {"name": "ego left turn", "role": "ego", "line": self.ref_line},
            {
                "name": "oncoming straight",
                "role": "interaction",
                "line": self.route_paths["oncoming_straight"].sample_xy(0.25),
            },
            {
                "name": "left crossing straight",
                "role": "interaction",
                "line": self.route_paths["left_crossing_straight"].sample_xy(0.25),
            },
        )
        self.road_left_l = self.road_half_width
        self.road_right_l = -self.road_half_width
        extent = self.route_extent
        half = self.road_half_width
        self.road_boundaries = (
            ([-half, -half], [-extent, -half]),
            ([-half, -half], [half, extent]),
            ([half, half], [-extent, -half]),
            ([half, half], [half, extent]),
            ([-extent, -half], [-half, -half]),
            ([half, extent], [-half, -half]),
            ([-extent, -half], [half, half]),
            ([half, extent], [half, half]),
        )
        self.lane_markings = (
            ([0.0, 0.0], [-extent, -half]),
            ([0.0, 0.0], [half, extent]),
            ([-extent, -half], [0.0, 0.0]),
            ([half, extent], [0.0, 0.0]),
        )
        self.visual_bounds = (-34.0, 34.0, -34.0, 34.0)
        self.actor_specs = (
            LeftTurnActorSpec("oncoming_vehicle", "oncoming_straight", oncoming_s, oncoming_speed),
            LeftTurnActorSpec(
                "left_crossing_vehicle",
                "left_crossing_straight",
                left_crossing_s,
                left_crossing_speed,
            ),
        )

    @property
    def name(self) -> str:
        return "unprotected_left_turn"

    def initial_state(self) -> EgoState:
        return EgoState(s=self.ego_s, l=0.0, s_v=self.ego_speed)

    def build_problem(self, ego: EgoState, t: float = 0.0) -> PlanningProblem:
        times = np.arange(0.0, self.horizon + 0.5 * self.dt, self.dt, dtype=float)
        times[-1] = self.horizon
        actors = tuple(self._actor_prediction(spec, times, float(t)) for spec in self.actor_specs)
        return PlanningProblem(
            ego=ego,
            ref_path=self.ref_path,
            road_boundary=RoadBoundary(left_l=self.road_left_l, right_l=self.road_right_l),
            horizon=self.horizon,
            dt=self.dt,
            actors=actors,
            metadata={
                "scenario": self.name,
                "maneuver": "unprotected_left_turn",
                "target_speed": self.target_speed,
                "reference_l": 0.0,
                "reference_lines": self.reference_lines,
                "game_actor_ids": tuple(spec.actor_id for spec in self.actor_specs),
                "source": "spatiotemporal_joint_planner.unprotected_left_turn",
            },
        )

    def actors_at(self, t: float) -> list[ActorPrediction]:
        return [self._actor_prediction(spec, np.array([0.0]), float(t)) for spec in self.actor_specs]

    def offset_curve(self, offset: float, ds: float = 0.25) -> tuple[list[float], list[float]]:
        return self._offset_curve(self.ref_path, offset, ds)

    def _left_turn_path(self) -> SmoothReferencePath:
        extent = self.route_extent
        offset = self.lane_offset
        entry_y = self.turn_entry_y
        radius = offset - entry_y
        center_x = offset - radius
        center_y = entry_y
        approach_y = np.linspace(-extent, entry_y, num=24, endpoint=False)
        approach = np.column_stack([np.full_like(approach_y, offset), approach_y])
        angles = np.linspace(0.0, 0.5 * math.pi, num=25)
        arc = np.column_stack(
            [
                center_x + radius * np.cos(angles),
                center_y + radius * np.sin(angles),
            ]
        )
        departure_x = np.linspace(center_x, -extent, num=24)[1:]
        departure = np.column_stack([departure_x, np.full_like(departure_x, offset)])
        points = np.vstack([approach, arc, departure])
        return SmoothReferencePath(points[:, 0], points[:, 1])

    def _actor_prediction(self, spec: LeftTurnActorSpec, relative_times: np.ndarray, start_time: float) -> ActorPrediction:
        query_times = float(start_time) + np.asarray(relative_times, dtype=float)
        s = float(spec.s) + float(spec.speed) * query_times + 0.5 * float(spec.acceleration) * query_times**2
        speed = np.maximum(float(spec.speed) + float(spec.acceleration) * query_times, 0.0)
        path = self.route_paths[spec.route_id]
        poses = np.asarray([self._pose_from_path(path, value, 0.0) for value in s], dtype=float)
        half_length = 0.5 * float(spec.length)
        half_width = 0.5 * float(spec.width)
        temporal = {
            "t": np.asarray(relative_times, dtype=float),
            "s_min": s - half_length,
            "s_max": s + half_length,
            "l_min": np.full_like(s, -half_width),
            "l_max": np.full_like(s, half_width),
        }
        return ActorPrediction(
            actor_id=spec.actor_id,
            actor_type="vehicle",
            times=query_times,
            x=poses[:, 0],
            y=poses[:, 1],
            yaw=poses[:, 2],
            length=float(spec.length),
            width=float(spec.width),
            metadata={
                "s": float(s[0]),
                "l": 0.0,
                "s_v": float(speed[0]),
                "s_a": float(spec.acceleration),
                "route_id": spec.route_id,
                "ref_path": path,
                "road_boundary": RoadBoundary(left_l=self.road_half_width, right_l=-self.road_half_width),
                "temporal_blocked_range": temporal,
                "static": False,
            },
        )

    @staticmethod
    def _pose_from_path(path: SmoothReferencePath, s: float, l: float) -> tuple[float, float, float]:
        x_ref, y_ref = path.calc_position(float(s))
        yaw = path.calc_yaw(float(s))
        return (
            float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0),
            float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0),
            float(yaw),
        )

    @staticmethod
    def _offset_curve(path: SmoothReferencePath, offset: float, ds: float) -> tuple[list[float], list[float]]:
        values = np.asarray(
            [UnprotectedLeftTurnScenario._pose_from_path(path, float(s), float(offset))[:2] for s in path.sample_s(ds)],
            dtype=float,
        )
        return values[:, 0].tolist(), values[:, 1].tolist()
