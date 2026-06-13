from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.scenario.base import Scenario
from spatiotemporal_joint_planner.scenario.static_nudge import SmoothReferencePath


@dataclass(frozen=True)
class IntersectionActorSpec:
    actor_id: str
    route_id: str
    s: float
    speed: float
    acceleration: float = 0.0
    length: float = 4.8
    width: float = 2.0


class UnprotectedIntersectionScenario(Scenario):
    """Three-vehicle unprotected intersection with independent actor routes."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        route_extent: float = 55.0,
        lane_offset: float = 1.8,
        road_half_width: float = 4.2,
        ego_s: float = 18.0,
        ego_speed: float = 9.0,
        target_speed: float = 10.0,
        north_actor_s: float = 22.0,
        north_actor_speed: float = 7.5,
        south_actor_s: float = 15.0,
        south_actor_speed: float = 8.5,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.route_extent = max(float(route_extent), 30.0)
        self.lane_offset = float(lane_offset)
        self.road_half_width = max(float(road_half_width), 3.0)
        self.ego_s = float(ego_s)
        self.ego_speed = float(ego_speed)
        self.target_speed = float(target_speed)
        extent = self.route_extent
        offset = self.lane_offset
        self.route_paths = {
            "ego_west_east": SmoothReferencePath([-extent, extent], [-offset, -offset]),
            "north_south": SmoothReferencePath([offset, offset], [extent, -extent]),
            "south_north": SmoothReferencePath([-offset, -offset], [-extent, extent]),
        }
        self.ref_path = self.route_paths["ego_west_east"]
        self.ref_line = self.ref_path.sample_xy(0.25)
        self.reference_lines = tuple(
            {
                "name": route_id,
                "role": "ego" if route_id == "ego_west_east" else "interaction",
                "line": path.sample_xy(0.25),
            }
            for route_id, path in self.route_paths.items()
        )
        self.road_left_l = self.road_half_width
        self.road_right_l = -self.road_half_width
        self.lane_markings = (
            ([-extent, extent], [0.0, 0.0]),
            ([0.0, 0.0], [-extent, extent]),
        )
        self.extra_road_boundaries = (
            ([-self.road_half_width, -self.road_half_width], [-extent, extent]),
            ([self.road_half_width, self.road_half_width], [-extent, extent]),
        )
        self.visual_bounds = (-32.0, 32.0, -32.0, 32.0)
        self.actor_specs = (
            IntersectionActorSpec("north_interaction_vehicle", "north_south", north_actor_s, north_actor_speed),
            IntersectionActorSpec("south_interaction_vehicle", "south_north", south_actor_s, south_actor_speed),
        )

    @property
    def name(self) -> str:
        return "unprotected_intersection"

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
                "maneuver": "cross_unprotected_intersection",
                "target_speed": self.target_speed,
                "reference_l": 0.0,
                "reference_lines": self.reference_lines,
                "game_actor_ids": tuple(spec.actor_id for spec in self.actor_specs),
                "source": "spatiotemporal_joint_planner.unprotected_intersection",
            },
        )

    def actors_at(self, t: float) -> list[ActorPrediction]:
        return [self._actor_prediction(spec, np.array([0.0]), float(t)) for spec in self.actor_specs]

    def offset_curve(self, offset: float, ds: float = 0.25) -> tuple[list[float], list[float]]:
        return self._offset_curve(self.ref_path, offset, ds)

    def _actor_prediction(self, spec: IntersectionActorSpec, relative_times: np.ndarray, start_time: float) -> ActorPrediction:
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
            [UnprotectedIntersectionScenario._pose_from_path(path, float(s), float(offset))[:2] for s in path.sample_s(ds)],
            dtype=float,
        )
        return values[:, 0].tolist(), values[:, 1].tolist()
