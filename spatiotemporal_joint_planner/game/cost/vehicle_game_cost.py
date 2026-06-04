from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import CostBreakdown, CostResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.cost.parametric_trajectory_cost import pseudo_huber, saturate_cost, shaped_hinge


@dataclass(frozen=True)
class VehicleGameCostConfig:
    vehicle_front: float = 2.4
    vehicle_rear: float = 2.4
    vehicle_width: float = 2.0
    obstacle_s_buffer: float = 0.5
    obstacle_l_buffer: float = 0.2
    road_edge_buffer: float = 0.4
    max_speed: float = 15.0
    speed_tracking_comfort: float = 2.5
    acceleration_comfort: float = 1.5
    jerk_comfort: float = 2.0
    lane_keep_comfort: float = 0.35
    prior_speed_comfort: float = 2.0
    min_follow_gap: float = 6.0
    time_headway: float = 1.2
    headway_comfort: float = 4.0
    collision_score_scale: float = 0.25
    road_score_scale: float = 0.25
    speed_score_scale: float = 1.0
    comfort_score_scale: float = 1.0
    lane_score_scale: float = 1.0
    prior_score_scale: float = 1.0
    headway_score_scale: float = 1.0


class VehicleGameCost:
    """Cost for an optimized non-ego vehicle in the game planner."""

    def __init__(self, config: Optional[VehicleGameCostConfig] = None):
        self.config = config or VehicleGameCostConfig()

    @property
    def name(self) -> str:
        return "vehicle_game_cost"

    def evaluate(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        opponent_trajectories: Sequence[Trajectory] = (),
    ) -> CostResult:
        blocked_ranges = self._blocked_ranges(problem, opponent_trajectories)
        collision_running, collision_overlap = self._collision_terms(trajectory, blocked_ranges)
        road_running, road_violation = self._road_terms(trajectory, problem)
        speed_running, speed_violation = self._speed_terms(trajectory, problem)
        comfort_running = self._comfort_terms(trajectory)
        lane_running = self._lane_keep_terms(trajectory, problem)
        prior_running = self._prior_terms(trajectory, problem)
        headway_running = self._headway_terms(trajectory, opponent_trajectories)

        collision_cost = self._topk_max(collision_running, 0.15)
        road_cost = self._topk_max(road_running, 0.15)
        speed_cost = float(np.mean(speed_running)) if speed_running.size else 4.0
        comfort_cost = float(np.mean(comfort_running)) if comfort_running.size else 4.0
        lane_cost = float(np.mean(lane_running)) if lane_running.size else 4.0
        prior_cost = float(np.mean(prior_running)) if prior_running.size else 4.0
        headway_cost = self._topk_max(headway_running, 0.25)

        collision_score = saturate_cost(collision_cost, self.config.collision_score_scale)
        road_score = saturate_cost(road_cost, self.config.road_score_scale)
        speed_score = saturate_cost(speed_cost, self.config.speed_score_scale)
        comfort_score = saturate_cost(comfort_cost, self.config.comfort_score_scale)
        lane_score = saturate_cost(lane_cost, self.config.lane_score_scale)
        prior_score = saturate_cost(prior_cost, self.config.prior_score_scale)
        headway_score = saturate_cost(headway_cost, self.config.headway_score_scale)

        total = float(
            1.0e9 * collision_score
            + 1.0e8 * road_score
            + 1.0e4 * headway_score
            + 1.0e3 * speed_score
            + 1.0e2 * prior_score
            + 1.0e2 * lane_score
            + 1.0e1 * comfort_score
        )
        terms = {
            "collision_flag": float(np.max(collision_overlap) > 0.0) if collision_overlap.size else 0.0,
            "collision_cost": float(collision_cost),
            "collision_score": float(collision_score),
            "road_flag": float(np.max(road_violation) > 0.0) if road_violation.size else 0.0,
            "road_cost": float(road_cost),
            "road_score": float(road_score),
            "speed_flag": float(np.max(speed_violation) > 0.0) if speed_violation.size else 0.0,
            "speed_cost": float(speed_cost),
            "speed_score": float(speed_score),
            "comfort_cost": float(comfort_cost),
            "comfort_score": float(comfort_score),
            "lane_keep_cost": float(lane_cost),
            "lane_keep_score": float(lane_score),
            "prior_cost": float(prior_cost),
            "prior_score": float(prior_score),
            "headway_cost": float(headway_cost),
            "headway_score": float(headway_score),
        }
        hard_violation = bool(terms["collision_flag"] > 0.0 or terms["road_flag"] > 0.0 or terms["speed_flag"] > 0.0)
        return CostResult(
            total=total,
            breakdown=CostBreakdown(terms=terms, hard_violation=hard_violation),
            feasible=not hard_violation,
            metadata={"cost": self.name, "blocked_ranges": blocked_ranges},
        )

    def _blocked_ranges(self, problem: PlanningProblem, opponents: Sequence[Trajectory]) -> list[dict]:
        ranges = [self._blocked_range_from_actor(actor) for actor in problem.actors]
        ranges = [item for item in ranges if item is not None]
        for idx, trajectory in enumerate(opponents):
            ranges.append(self._blocked_range_from_trajectory(trajectory, f"opponent_{idx}"))
        return [self._inflate_range(item) for item in ranges]

    def _blocked_range_from_actor(self, actor) -> Optional[dict]:
        metadata = dict(actor.metadata or {})
        temporal = metadata.get("temporal_blocked_range") or metadata.get("temporal_blocked_ranges")
        if isinstance(temporal, Mapping) and all(key in temporal for key in ("t", "s_min", "s_max", "l_min", "l_max")):
            arrays = {key: np.asarray(temporal[key], dtype=float).reshape(-1) for key in ("t", "s_min", "s_max", "l_min", "l_max")}
            n = min(value.size for value in arrays.values())
            if n > 0:
                arrays = {key: value[:n] for key, value in arrays.items()}
                return {
                    "s_min": float(arrays["s_min"][0]),
                    "s_max": float(arrays["s_max"][0]),
                    "l_min": float(arrays["l_min"][0]),
                    "l_max": float(arrays["l_max"][0]),
                    "temporal": arrays,
                    "actor_id": actor.actor_id,
                }
        keys = ("blocked_s_min", "blocked_s_max", "blocked_l_min", "blocked_l_max")
        if all(key in metadata for key in keys):
            return {
                "s_min": float(metadata["blocked_s_min"]),
                "s_max": float(metadata["blocked_s_max"]),
                "l_min": float(metadata["blocked_l_min"]),
                "l_max": float(metadata["blocked_l_max"]),
                "actor_id": actor.actor_id,
            }
        if "s" in metadata and "l" in metadata:
            return {
                "s_min": float(metadata["s"]) - 0.5 * float(actor.length),
                "s_max": float(metadata["s"]) + 0.5 * float(actor.length),
                "l_min": float(metadata["l"]) - 0.5 * float(actor.width),
                "l_max": float(metadata["l"]) + 0.5 * float(actor.width),
                "actor_id": actor.actor_id,
            }
        return None

    def _blocked_range_from_trajectory(self, trajectory: Trajectory, actor_id: str) -> dict:
        s = np.asarray(trajectory.s, dtype=float)
        l = np.asarray(trajectory.l, dtype=float)
        t = np.asarray(trajectory.t, dtype=float)
        n = min(s.size, l.size, t.size)
        front = float(self.config.vehicle_front)
        rear = float(self.config.vehicle_rear)
        half_width = 0.5 * float(self.config.vehicle_width)
        temporal = {
            "t": t[:n],
            "s_min": s[:n] - rear,
            "s_max": s[:n] + front,
            "l_min": l[:n] - half_width,
            "l_max": l[:n] + half_width,
        }
        return {
            "s_min": float(temporal["s_min"][0]),
            "s_max": float(temporal["s_max"][0]),
            "l_min": float(temporal["l_min"][0]),
            "l_max": float(temporal["l_max"][0]),
            "actor_id": actor_id,
            "temporal": temporal,
        }

    def _inflate_range(self, blocked: dict) -> dict:
        inflated = dict(blocked)
        s_buffer = max(float(self.config.obstacle_s_buffer), 0.0)
        l_buffer = max(float(self.config.obstacle_l_buffer), 0.0)
        inflated["s_min"] = float(blocked["s_min"]) - s_buffer
        inflated["s_max"] = float(blocked["s_max"]) + s_buffer
        inflated["l_min"] = float(blocked["l_min"]) - l_buffer
        inflated["l_max"] = float(blocked["l_max"]) + l_buffer
        temporal = blocked.get("temporal")
        if temporal:
            inflated_temporal = dict(temporal)
            inflated_temporal["s_min"] = np.asarray(temporal["s_min"], dtype=float) - s_buffer
            inflated_temporal["s_max"] = np.asarray(temporal["s_max"], dtype=float) + s_buffer
            inflated_temporal["l_min"] = np.asarray(temporal["l_min"], dtype=float) - l_buffer
            inflated_temporal["l_max"] = np.asarray(temporal["l_max"], dtype=float) + l_buffer
            inflated["temporal"] = inflated_temporal
        return inflated

    def _collision_terms(self, trajectory: Trajectory, blocked_ranges: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(trajectory.s, dtype=float)
        l = np.asarray(trajectory.l, dtype=float)
        t = np.asarray(trajectory.t, dtype=float)
        n = min(s.size, l.size)
        running = np.zeros((n,), dtype=float)
        overlap = np.zeros((n,), dtype=float)
        if n == 0 or not blocked_ranges:
            return running, overlap
        if t.size < n:
            t = np.linspace(0.0, float(n - 1), num=n, dtype=float)
        ego_s_min = s[:n] - float(self.config.vehicle_rear)
        ego_s_max = s[:n] + float(self.config.vehicle_front)
        half_width = 0.5 * float(self.config.vehicle_width)
        ego_l_min = l[:n] - half_width
        ego_l_max = l[:n] + half_width
        for blocked in blocked_ranges:
            b_s_min = self._blocked_at(blocked, "s_min", t[:n])
            b_s_max = self._blocked_at(blocked, "s_max", t[:n])
            b_l_min = self._blocked_at(blocked, "l_min", t[:n])
            b_l_max = self._blocked_at(blocked, "l_max", t[:n])
            mask = (ego_s_min <= b_s_max) & (b_s_min <= ego_s_max) & (ego_l_min <= b_l_max) & (b_l_min <= ego_l_max)
            if not np.any(mask):
                continue
            s_overlap = np.minimum(ego_s_max, b_s_max) - np.maximum(ego_s_min, b_s_min)
            l_overlap = np.minimum(ego_l_max, b_l_max) - np.maximum(ego_l_min, b_l_min)
            penetration = np.maximum(np.minimum(s_overlap, l_overlap), 0.0)
            sample_cost = 1.0 + shaped_hinge(penetration, safe=0.0, soft=0.6, tail_gain=0.35, cap=3.0)
            running = np.maximum(running, np.where(mask, sample_cost, 0.0))
            overlap = np.maximum(overlap, mask.astype(float))
        return running, overlap

    def _road_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        l = np.asarray(trajectory.l, dtype=float)
        if l.size == 0:
            return np.array([4.0]), np.array([1.0])
        half_width = 0.5 * float(self.config.vehicle_width)
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        l_min = l - half_width
        l_max = l + half_width
        excess = np.maximum(np.maximum(l_max - left, 0.0), np.maximum(right - l_min, 0.0))
        clearance = np.minimum(left - l_max, l_min - right)
        edge_pressure = np.maximum(float(self.config.road_edge_buffer) - clearance, 0.0)
        running = shaped_hinge(excess, safe=0.0, soft=0.6, tail_gain=0.25, cap=3.0) + 0.2 * shaped_hinge(
            edge_pressure,
            safe=0.0,
            soft=max(float(self.config.road_edge_buffer), 1e-3),
            tail_gain=0.1,
            cap=1.5,
        )
        return np.asarray(running, dtype=float), np.asarray(excess > 1e-6, dtype=float)

    def _speed_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> tuple[np.ndarray, np.ndarray]:
        if trajectory.s_v is None:
            return np.array([4.0]), np.array([1.0])
        s_v = np.asarray(trajectory.s_v, dtype=float)
        max_speed = max(float(self.config.max_speed), 1e-3)
        target = float(dict(problem.metadata or {}).get("target_speed", max_speed))
        comfort = max(float(self.config.speed_tracking_comfort), 1e-3)
        target_cost = pseudo_huber((s_v - min(target, max_speed)) / comfort, delta=1.0)
        reverse = np.maximum(-s_v, 0.0)
        excess = np.maximum(s_v - max_speed, 0.0)
        limit_cost = shaped_hinge(reverse, safe=0.0, soft=0.5, tail_gain=0.35, cap=3.0) + shaped_hinge(
            excess,
            safe=0.0,
            soft=1.0,
            tail_gain=0.35,
            cap=3.0,
        )
        return np.asarray(target_cost + limit_cost, dtype=float), np.asarray((reverse > 1e-6) | (excess > 1e-6), dtype=float)

    def _comfort_terms(self, trajectory: Trajectory) -> np.ndarray:
        t = np.asarray(trajectory.t, dtype=float)
        if trajectory.s_a is None:
            return np.array([4.0])
        s_a = np.asarray(trajectory.s_a, dtype=float)
        accel_cost = pseudo_huber(s_a / max(float(self.config.acceleration_comfort), 1e-3), delta=1.0)
        if s_a.size >= 2 and t.size >= s_a.size:
            jerk = np.gradient(s_a[: t.size], t[: s_a.size], edge_order=2 if s_a.size >= 3 else 1)
            jerk_cost = pseudo_huber(jerk / max(float(self.config.jerk_comfort), 1e-3), delta=1.0)
            n = min(accel_cost.size, jerk_cost.size)
            return np.asarray(0.6 * accel_cost[:n] + 0.4 * jerk_cost[:n], dtype=float)
        return np.asarray(accel_cost, dtype=float)

    def _lane_keep_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> np.ndarray:
        l = np.asarray(trajectory.l, dtype=float)
        target_l = float(dict(problem.metadata or {}).get("reference_l", problem.ego.l))
        return np.asarray(pseudo_huber((l - target_l) / max(float(self.config.lane_keep_comfort), 1e-3), delta=1.0), dtype=float)

    def _prior_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> np.ndarray:
        if trajectory.s_v is None:
            return np.array([4.0])
        prior_speed = float(dict(problem.metadata or {}).get("prior_speed", problem.ego.s_v))
        s_v = np.asarray(trajectory.s_v, dtype=float)
        return np.asarray(pseudo_huber((s_v - prior_speed) / max(float(self.config.prior_speed_comfort), 1e-3), delta=1.0), dtype=float)

    def _headway_terms(self, trajectory: Trajectory, opponents: Sequence[Trajectory]) -> np.ndarray:
        s = np.asarray(trajectory.s, dtype=float)
        l = np.asarray(trajectory.l, dtype=float)
        if s.size == 0:
            return np.array([0.0], dtype=float)
        s_v = np.zeros_like(s, dtype=float) if trajectory.s_v is None else np.asarray(trajectory.s_v, dtype=float)
        n = min(s.size, l.size, s_v.size)
        running = np.zeros((n,), dtype=float)
        if n == 0:
            return running
        agent_front = s[:n] + float(self.config.vehicle_front)
        desired_gap = float(self.config.min_follow_gap) + np.maximum(s_v[:n], 0.0) * float(self.config.time_headway)
        lateral_gate = max(float(self.config.vehicle_width), 1e-3)
        for opponent in opponents:
            opp_s = np.asarray(opponent.s, dtype=float)
            opp_l = np.asarray(opponent.l, dtype=float)
            m = min(n, opp_s.size, opp_l.size)
            if m == 0:
                continue
            opp_rear = opp_s[:m] - float(self.config.vehicle_rear)
            gap = opp_rear - agent_front[:m]
            same_lane = np.abs(opp_l[:m] - l[:m]) <= lateral_gate
            ahead = gap >= -float(self.config.vehicle_front + self.config.vehicle_rear)
            pressure = np.maximum(desired_gap[:m] - gap, 0.0)
            sample_cost = pseudo_huber(pressure / max(float(self.config.headway_comfort), 1e-3), delta=1.0)
            running[:m] = np.maximum(running[:m], np.where(same_lane & ahead, sample_cost, 0.0))
        return running

    @staticmethod
    def _blocked_at(blocked: dict, key: str, times: np.ndarray) -> np.ndarray:
        temporal = blocked.get("temporal")
        if not temporal:
            return np.full(times.shape, float(blocked[key]), dtype=float)
        source_t = np.asarray(temporal.get("t", []), dtype=float).reshape(-1)
        source_v = np.asarray(temporal.get(key, []), dtype=float).reshape(-1)
        n = min(source_t.size, source_v.size)
        if n == 0:
            return np.full(times.shape, float(blocked[key]), dtype=float)
        return np.interp(np.asarray(times, dtype=float), source_t[:n], source_v[:n], left=source_v[0], right=source_v[n - 1])

    @staticmethod
    def _topk_max(values: np.ndarray, fraction: float) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        k = max(1, int(np.ceil(values.size * float(fraction))))
        top = np.partition(values, -k)[-k:]
        return float(0.7 * np.mean(top) + 0.3 * np.max(values))
