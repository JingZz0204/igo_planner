from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction
from spatiotemporal_joint_planner.planner.warm_start.base import WarmStartContext, WarmStartGenerator, finalize_warm_starts
from spatiotemporal_joint_planner.trajectory_models.common import fixed_time_grid, quartic_profile, quintic_profile, xy_from_sl


@dataclass(frozen=True)
class BezierTrajectoryWarmStartConfig:
    lateral_offsets: tuple[float, ...] = (0.0, 1.8, -1.8, 3.6, -3.6)
    ramp_start: float = 0.35
    ramp_end: float = 1.0
    lane_width: float = 3.6
    ego_width: float = 2.7
    planning_obstacle_s_buffer: float = 0.5
    planning_obstacle_l_buffer: float = 0.2
    road_edge_margin: float = 0.4
    relevant_s_buffer: float = 25.0
    lateral_grid_count: int = 9
    speed_grid_count: int = 7
    speed_lower_scale: float = 0.3
    speed_upper_scale: float = 2.0
    min_terminal_speed: float = 0.5
    max_terminal_speed: float = 15.0
    lateral_finish_times: tuple[float, ...] = (1.5, 2.5, 3.5, 5.0)
    min_lateral_finish_time: float = 1.0
    fit_dt: float = 0.25
    terminal_position_weight: float = 2.0
    terminal_derivative_weight: float = 4.0
    max_semantic_seeds: int = 88
    target_speed_offsets: tuple[float, ...] = (0.0, -2.0, 2.0)


class BezierTrajectoryWarmStartGenerator(WarmStartGenerator):
    """Roadmap-aware warm starts for ego-local XY Bezier control points."""

    def __init__(self, config: Optional[BezierTrajectoryWarmStartConfig] = None):
        self.config = config or BezierTrajectoryWarmStartConfig()

    @property
    def name(self) -> str:
        return "bezier_trajectory_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return context.trajectory_model.name == "bezier_trajectory" and context.parameter_dim % 2 == 0

    def generate(self, context: WarmStartContext) -> np.ndarray:
        rows = []
        if context.previous_parameters is not None:
            rows.append(np.asarray(context.previous_parameters, dtype=float))

        try:
            nominal = np.asarray(context.trajectory_model.reference_parameters(context.problem), dtype=float)
        except Exception:
            nominal = 0.5 * (context.lower_bound + context.upper_bound)
        rows.append(nominal)

        rows.extend(self._semantic_rows(context, nominal))
        rows.extend(self._shifted_nominal_rows(nominal))

        rows.append(0.5 * (context.lower_bound + context.upper_bound))
        return finalize_warm_starts(rows, context)

    def _semantic_rows(self, context: WarmStartContext, nominal: np.ndarray) -> list[np.ndarray]:
        if not self._has_bezier_helpers(context):
            return []

        candidates = self._candidate_tuples(context)
        rows = []
        for l_target, v_end, t_l_finish in candidates[: max(int(self.config.max_semantic_seeds), 1)]:
            fitted = self._fit_semantic_seed(context, nominal, float(l_target), float(v_end), float(t_l_finish))
            if fitted is not None:
                rows.append(fitted)
        return rows

    def _fit_semantic_seed(
        self,
        context: WarmStartContext,
        nominal: np.ndarray,
        l_target: float,
        v_end: float,
        t_l_finish: float,
    ) -> Optional[np.ndarray]:
        model = context.trajectory_model
        problem = context.problem
        try:
            fixed_control, x0, y0, yaw0 = self._fixed_control_points(context, nominal)
            t, s_values, world_points = self._semantic_world_points(context, l_target, v_end, t_l_finish)
            local_points = model._world_to_local(world_points, x0, y0, yaw0)
            degree = int(model._degree())
            fixed = int(model._fixed_controls())
            horizon = max(float(problem.horizon), 1e-3)
            u_values = np.clip(t / horizon, 0.0, 1.0)
            basis = model._bernstein_basis(degree, u_values)
            return self._fit_free_controls(
                context=context,
                basis=basis,
                fixed_control=fixed_control,
                local_points=local_points,
                s_end=float(s_values[-1]),
                v_end=float(v_end),
                yaw0=float(yaw0),
            )
        except Exception:
            return None

    def _semantic_world_points(
        self,
        context: WarmStartContext,
        l_target: float,
        v_end: float,
        t_l_finish: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        problem = context.problem
        horizon = max(float(problem.horizon), 1e-3)
        t = fixed_time_grid(horizon, float(self.config.fit_dt))
        s_values, _, _ = quartic_profile(
            problem.ego.s,
            problem.ego.s_v,
            problem.ego.s_a,
            float(v_end),
            t,
            a1=0.0,
        )

        finish = float(np.clip(float(t_l_finish), float(self.config.min_lateral_finish_time), horizon))
        lateral_t = np.minimum(t, finish)
        l_values, _, _ = quintic_profile(
            problem.ego.l,
            problem.ego.l_v,
            problem.ego.l_a,
            float(l_target),
            0.0,
            0.0,
            lateral_t,
        )
        x_values, y_values = xy_from_sl(problem, s_values, l_values)
        if x_values is None or y_values is None:
            raise ValueError("Unable to project semantic Frenet seed to XY.")
        return t, np.asarray(s_values, dtype=float), np.column_stack([x_values, y_values])

    def _fit_free_controls(
        self,
        context: WarmStartContext,
        basis: np.ndarray,
        fixed_control: np.ndarray,
        local_points: np.ndarray,
        s_end: float,
        v_end: float,
        yaw0: float,
    ) -> np.ndarray:
        model = context.trajectory_model
        degree = int(model._degree())
        fixed = int(model._fixed_controls())
        n_free = degree + 1 - fixed
        if n_free <= 0:
            raise ValueError("Bezier model has no free controls.")

        fixed_part = basis[:, :fixed] @ fixed_control
        matrix = basis[:, fixed:].copy()
        rhs = np.asarray(local_points, dtype=float) - fixed_part

        terminal_position_weight = max(float(self.config.terminal_position_weight), 0.0)
        if terminal_position_weight > 0.0:
            matrix = np.vstack([matrix, terminal_position_weight * basis[-1, fixed:]])
            rhs = np.vstack([rhs, terminal_position_weight * rhs[-1]])

        derivative_row = self._terminal_derivative_row(context, s_end, v_end, yaw0, degree, fixed, n_free)
        if derivative_row is not None:
            row, desired_delta = derivative_row
            weight = max(float(self.config.terminal_derivative_weight), 0.0)
            if weight > 0.0:
                matrix = np.vstack([matrix, weight * row])
                rhs = np.vstack([rhs, weight * desired_delta])

        sol_x, *_ = np.linalg.lstsq(matrix, rhs[:, 0], rcond=None)
        sol_y, *_ = np.linalg.lstsq(matrix, rhs[:, 1], rcond=None)
        free_controls = np.column_stack([sol_x, sol_y])
        return free_controls.reshape(-1)

    def _terminal_derivative_row(
        self,
        context: WarmStartContext,
        s_end: float,
        v_end: float,
        yaw0: float,
        degree: int,
        fixed: int,
        n_free: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if degree <= 0:
            return None
        idx_last = degree - fixed
        idx_prev = degree - 1 - fixed
        if idx_last < 0 or idx_last >= n_free or idx_prev < 0 or idx_prev >= n_free:
            return None

        yaw_ref = self._reference_yaw(context, s_end)
        world_velocity = np.array([float(v_end) * math.cos(yaw_ref), float(v_end) * math.sin(yaw_ref)], dtype=float)
        c = math.cos(float(yaw0))
        s = math.sin(float(yaw0))
        local_velocity = np.array(
            [
                world_velocity[0] * c + world_velocity[1] * s,
                -world_velocity[0] * s + world_velocity[1] * c,
            ],
            dtype=float,
        )
        desired_delta = local_velocity * max(float(context.problem.horizon), 1e-3) / max(float(degree), 1.0)
        row = np.zeros((n_free,), dtype=float)
        row[idx_prev] = -1.0
        row[idx_last] = 1.0
        return row, desired_delta

    def _fixed_control_points(self, context: WarmStartContext, nominal: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        model = context.trajectory_model
        control, x0, y0, yaw0 = model._build_control_points(np.asarray(nominal, dtype=float), context.problem)
        fixed = int(model._fixed_controls())
        return np.asarray(control[:fixed], dtype=float), float(x0), float(y0), float(yaw0)

    def _candidate_tuples(self, context: WarmStartContext) -> list[tuple[float, float, float]]:
        lateral_values = self._lateral_values(context)
        speed_values = self._speed_values(context)
        finish_times = self._finish_times(context)
        target_l = self._target_lateral(context)
        target_speed = self._target_speed(context)
        preferred_time = self._preferred_lateral_finish_time(context)

        candidates = []
        for lateral in lateral_values:
            for speed in speed_values:
                for finish_time in finish_times:
                    candidates.append((float(lateral), float(speed), float(finish_time)))

        def score(item: tuple[float, float, float]) -> tuple[float, float, float]:
            lateral, speed, finish_time = item
            lateral_targets = [float(context.problem.ego.l)]
            if target_l is not None:
                lateral_targets.append(float(target_l))
            for nudge in self._obstacle_nudges(context):
                lateral_targets.append(float(nudge))
            lateral_score = min(abs(float(lateral) - value) for value in lateral_targets) / max(float(self.config.lane_width), 1.0)
            speed_score = min(abs(float(speed) - float(context.problem.ego.s_v)), abs(float(speed) - target_speed)) / max(target_speed, 1.0)
            time_score = abs(float(finish_time) - preferred_time) / max(float(context.problem.horizon), 1.0)
            return lateral_score, 0.35 * speed_score, 0.20 * time_score

        candidates.sort(key=score)
        return self._unique_tuples(candidates)

    def _lateral_values(self, context: WarmStartContext) -> list[float]:
        safe_right, safe_left = self._safe_lateral_bounds(context)
        grid_count = max(int(self.config.lateral_grid_count), 2)
        values = np.linspace(safe_right, safe_left, num=grid_count, dtype=float).tolist()
        values.extend([float(context.problem.ego.l), 0.0, float(self.config.lane_width), -float(self.config.lane_width)])
        values.extend([safe_right, safe_left])
        target_l = self._target_lateral(context)
        if target_l is not None:
            values.extend([float(target_l), float(target_l) - 0.3, float(target_l) + 0.3])
        values.extend(self._obstacle_nudges(context))
        return self._ordered_unique([float(np.clip(value, safe_right, safe_left)) for value in values], context.problem.ego.l)

    def _speed_values(self, context: WarmStartContext) -> list[float]:
        current_speed = max(float(context.problem.ego.s_v), 0.0)
        min_speed = max(float(self.config.min_terminal_speed), 0.0)
        max_speed = max(float(self.config.max_terminal_speed), min_speed + 1e-3)
        target_speed = float(np.clip(self._target_speed(context), min_speed, max_speed))

        low = float(np.clip(float(self.config.speed_lower_scale) * current_speed, min_speed, max_speed))
        high = float(np.clip(max(float(self.config.speed_upper_scale) * current_speed, target_speed), min_speed, max_speed))
        if high < low:
            low, high = high, low

        values = np.linspace(low, high, num=max(int(self.config.speed_grid_count), 2), dtype=float).tolist()
        values.extend([min_speed, max_speed, current_speed, target_speed])
        values.extend(target_speed + offset for offset in self.config.target_speed_offsets)
        clipped = [float(np.clip(value, min_speed, max_speed)) for value in values]
        return self._ordered_unique(clipped, current_speed)

    def _finish_times(self, context: WarmStartContext) -> list[float]:
        horizon = max(float(context.problem.horizon), 1e-3)
        low = min(max(float(self.config.min_lateral_finish_time), 0.2), horizon)
        values = [float(np.clip(value, low, horizon)) for value in self.config.lateral_finish_times]
        blocker = self._nearest_relevant_block(context)
        if blocker is not None:
            distance = max(float(blocker["s_min"]) - float(context.problem.ego.s), 0.0)
            arrival_time = distance / max(float(context.problem.ego.s_v), 1.0)
            values.extend([0.5 * arrival_time, 0.7 * arrival_time, arrival_time - 0.5])
        return self._ordered_unique([float(np.clip(value, low, horizon)) for value in values], self._preferred_lateral_finish_time(context))

    def _preferred_lateral_finish_time(self, context: WarmStartContext) -> float:
        horizon = max(float(context.problem.horizon), 1e-3)
        blocker = self._nearest_relevant_block(context)
        if blocker is None:
            return min(0.6 * horizon, horizon)
        distance = max(float(blocker["s_min"]) - float(context.problem.ego.s), 0.0)
        return float(np.clip(0.7 * distance / max(float(context.problem.ego.s_v), 1.0), float(self.config.min_lateral_finish_time), horizon))

    def _target_lateral(self, context: WarmStartContext) -> Optional[float]:
        metadata = dict(context.problem.metadata or {})
        for key in ("target_lane_l", "target_l", "lane_change_target_l"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        return None

    def _target_speed(self, context: WarmStartContext) -> float:
        metadata = dict(context.problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        return float(self.config.max_terminal_speed)

    def _obstacle_nudges(self, context: WarmStartContext) -> list[float]:
        blocker = self._nearest_relevant_block(context)
        if blocker is None:
            return []
        safe_right, safe_left = self._safe_lateral_bounds(context)
        half_width = 0.5 * float(self.config.ego_width)
        contact_eps = 1e-6
        left_nudge = float(blocker["l_max"]) + half_width + contact_eps
        right_nudge = float(blocker["l_min"]) - half_width - contact_eps
        return [
            float(np.clip(left_nudge, safe_right, safe_left)),
            float(np.clip(right_nudge, safe_right, safe_left)),
        ]

    def _nearest_relevant_block(self, context: WarmStartContext) -> Optional[dict]:
        blocks = self._blocked_ranges(context.problem.actors)
        if not blocks:
            return None
        ego_s = float(context.problem.ego.s)
        reachable_s = ego_s + max(float(context.problem.ego.s_v), 1.0) * float(context.problem.horizon)
        reachable_s += float(self.config.relevant_s_buffer)
        relevant = [
            block
            for block in blocks
            if float(block["s_max"]) >= ego_s - 5.0 and float(block["s_min"]) <= reachable_s
        ]
        if not relevant:
            return None
        relevant.sort(key=lambda block: max(float(block["s_min"]) - ego_s, 0.0))
        return relevant[0]

    def _safe_lateral_bounds(self, context: WarmStartContext) -> tuple[float, float]:
        left = max(float(context.problem.road_boundary.left_l), float(context.problem.road_boundary.right_l))
        right = min(float(context.problem.road_boundary.left_l), float(context.problem.road_boundary.right_l))
        half_width = 0.5 * float(self.config.ego_width)
        margin = float(self.config.road_edge_margin)
        safe_left = left - half_width - margin
        safe_right = right + half_width + margin
        if safe_right > safe_left:
            center = 0.5 * (right + left)
            return center, center
        return safe_right, safe_left

    def _reference_yaw(self, context: WarmStartContext, s_value: float) -> float:
        ref_path = context.problem.ref_path
        if not hasattr(ref_path, "calc_yaw"):
            return float(context.problem.ego.yaw or 0.0)
        route_end = self._route_end_s(ref_path)
        return float(ref_path.calc_yaw(float(np.clip(s_value, 0.0, route_end))))

    def _shifted_nominal_rows(self, nominal: np.ndarray) -> list[np.ndarray]:
        rows = []
        nominal_xy = np.asarray(nominal, dtype=float).reshape(-1, 2)
        n_points = nominal_xy.shape[0]
        ramp = np.linspace(float(self.config.ramp_start), float(self.config.ramp_end), num=n_points, dtype=float)
        for offset in self.config.lateral_offsets:
            if abs(float(offset)) <= 1e-9:
                continue
            shifted = nominal_xy.copy()
            shifted[:, 1] += float(offset) * ramp
            rows.append(shifted.reshape(-1))
        return rows

    def _blocked_ranges(self, actors: Sequence[ActorPrediction]) -> list[dict]:
        ranges = []
        for actor in actors:
            metadata = dict(actor.metadata or {})
            if all(key in metadata for key in ("blocked_s_min", "blocked_s_max", "blocked_l_min", "blocked_l_max")):
                ranges.append(
                    self._inflate_blocked_range(
                        {
                            "s_min": float(metadata["blocked_s_min"]),
                            "s_max": float(metadata["blocked_s_max"]),
                            "l_min": float(metadata["blocked_l_min"]),
                            "l_max": float(metadata["blocked_l_max"]),
                            "actor_id": actor.actor_id,
                        }
                    )
                )
            elif "s" in metadata and "l" in metadata:
                half_length = 0.5 * float(actor.length)
                half_width = 0.5 * float(actor.width)
                ranges.append(
                    self._inflate_blocked_range(
                        {
                            "s_min": float(metadata["s"]) - half_length,
                            "s_max": float(metadata["s"]) + half_length,
                            "l_min": float(metadata["l"]) - half_width,
                            "l_max": float(metadata["l"]) + half_width,
                            "actor_id": actor.actor_id,
                        }
                    )
                )
        return ranges

    def _inflate_blocked_range(self, blocked: dict) -> dict:
        s_buffer = max(float(self.config.planning_obstacle_s_buffer), 0.0)
        l_buffer = max(float(self.config.planning_obstacle_l_buffer), 0.0)
        inflated = dict(blocked)
        inflated["s_min"] = float(blocked["s_min"]) - s_buffer
        inflated["s_max"] = float(blocked["s_max"]) + s_buffer
        inflated["l_min"] = float(blocked["l_min"]) - l_buffer
        inflated["l_max"] = float(blocked["l_max"]) + l_buffer
        return inflated

    @staticmethod
    def _ordered_unique(values: Sequence[float], preferred: float) -> list[float]:
        unique = []
        seen = set()
        for value in sorted(values, key=lambda item: abs(float(item) - float(preferred))):
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(value))
        return unique

    @staticmethod
    def _unique_tuples(values: Sequence[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        unique = []
        seen = set()
        for lateral, speed, finish_time in values:
            key = (round(float(lateral), 6), round(float(speed), 6), round(float(finish_time), 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append((float(lateral), float(speed), float(finish_time)))
        return unique

    @staticmethod
    def _route_end_s(ref_path) -> float:
        if hasattr(ref_path, "s"):
            values = np.asarray(ref_path.s, dtype=float)
            if values.size:
                return max(float(values[-1]), 1e-3)
        return 1.0e6

    @staticmethod
    def _has_bezier_helpers(context: WarmStartContext) -> bool:
        model = context.trajectory_model
        return all(
            hasattr(model, name)
            for name in (
                "_build_control_points",
                "_degree",
                "_fixed_controls",
                "_bernstein_basis",
                "_world_to_local",
            )
        )
