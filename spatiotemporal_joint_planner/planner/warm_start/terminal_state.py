from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction
from spatiotemporal_joint_planner.planner.warm_start.base import WarmStartContext, WarmStartGenerator, finalize_warm_starts


@dataclass(frozen=True)
class TerminalStateWarmStartConfig:
    lane_width: float = 3.6
    ego_width: float = 2.7
    planning_obstacle_s_buffer: float = 0.5
    planning_obstacle_l_buffer: float = 0.2
    road_edge_margin: float = 0.4
    relevant_s_buffer: float = 25.0
    max_terminal_speed: float = 15.0
    min_terminal_speed: float = 0.5
    lateral_grid_count: int = 9
    speed_grid_count: int = 7
    speed_lower_scale: float = 0.3
    speed_upper_scale: float = 2.0
    slow_speeds: tuple[float, ...] = (0.5, 2.0, 4.0, 6.0)
    cruise_speed_offsets: tuple[float, ...] = (0.0, -2.0, 2.0)
    target_speed_offsets: tuple[float, ...] = (0.0, -2.0, 2.0)


class TerminalStateWarmStartGenerator(WarmStartGenerator):
    """Warm starts for theta = [l_end, v_end] terminal-state models."""

    terminal_model_names = {"lattice_trajectory"}

    def __init__(self, config: Optional[TerminalStateWarmStartConfig] = None):
        self.config = config or TerminalStateWarmStartConfig()

    @property
    def name(self) -> str:
        return "terminal_state_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return context.trajectory_model.name in self.terminal_model_names and context.parameter_dim == 2

    def generate(self, context: WarmStartContext) -> np.ndarray:
        rows = []
        if context.previous_parameters is not None:
            rows.append(np.asarray(context.previous_parameters, dtype=float))

        rows.append(context.trajectory_model.reference_parameters(context.problem))
        rows.extend(self._obstacle_aware_rows(context))
        rows.extend(self._lane_rows(context))
        rows.extend(self._grid_rows(context))
        return finalize_warm_starts(rows, context)

    def _grid_rows(self, context: WarmStartContext) -> list[np.ndarray]:
        safe_right, safe_left = self._safe_lateral_bounds(context)
        lateral_values = np.linspace(
            safe_right,
            safe_left,
            num=max(min(int(self.config.lateral_grid_count), int(context.max_count)), 2),
            dtype=float,
        )
        speed_values = self._grid_speed_values(
            context,
            max(min(int(self.config.speed_grid_count), int(context.max_count)), 2),
        )
        current_l = float(context.problem.ego.l)
        lateral_values = sorted(lateral_values.tolist(), key=lambda value: abs(float(value) - current_l))
        speed_values = sorted(speed_values, key=lambda value: abs(float(value) - float(context.problem.ego.s_v)))
        count = min(len(lateral_values), len(speed_values), max(int(context.max_count), 1))
        return [
            np.array([float(lateral_values[index]), float(speed_values[index])], dtype=float)
            for index in range(count)
        ]

    def _grid_speed_values(self, context: WarmStartContext, count: int) -> list[float]:
        current_speed = max(float(context.problem.ego.s_v), 0.0)
        low = float(self.config.speed_lower_scale) * current_speed
        high = float(self.config.speed_upper_scale) * current_speed

        min_speed = max(float(self.config.min_terminal_speed), float(context.lower_bound[1]), 0.0)
        max_speed = max(min(float(self.config.max_terminal_speed), float(context.upper_bound[1])), min_speed)
        target_speed = float(np.clip(self._target_speed(context), min_speed, max_speed))
        high = max(high, target_speed)

        low = float(np.clip(low, min_speed, max_speed))
        high = float(np.clip(high, min_speed, max_speed))
        if high < low:
            low, high = high, low

        values = np.linspace(low, high, num=max(int(count), 2), dtype=float).tolist()
        values.extend([min_speed, max_speed, target_speed, current_speed])
        return self._clip_speeds(values)

    def _obstacle_aware_rows(self, context: WarmStartContext) -> list[np.ndarray]:
        blocker = self._nearest_relevant_block(context)
        if blocker is None:
            return []

        safe_right, safe_left = self._safe_lateral_bounds(context)
        ego_half_width = 0.5 * float(self.config.ego_width)
        contact_eps = 1e-6
        left_nudge = np.clip(float(blocker["l_max"]) + ego_half_width + contact_eps, safe_right, safe_left)
        right_nudge = np.clip(float(blocker["l_min"]) - ego_half_width - contact_eps, safe_right, safe_left)

        current_l = float(context.problem.ego.l)
        current_overlap = self._ranges_overlap(
            current_l - ego_half_width,
            current_l + ego_half_width,
            float(blocker["l_min"]),
            float(blocker["l_max"]),
        )
        speeds = self._obstacle_speeds(context, current_overlap)
        rows = []

        for l_target in self._ordered_unique([left_nudge, right_nudge], prefer_positive=current_l >= 0.0):
            for speed in speeds:
                rows.append(np.array([l_target, speed], dtype=float))

        if current_overlap:
            for speed in speeds[:2]:
                rows.append(np.array([current_l, speed], dtype=float))
                rows.append(np.array([0.0, speed], dtype=float))
        return rows

    def _lane_rows(self, context: WarmStartContext) -> list[np.ndarray]:
        safe_right, safe_left = self._safe_lateral_bounds(context)
        current_l = float(context.problem.ego.l)
        lane_width = float(self.config.lane_width)
        lane_centers = self._lane_centers(context)
        lateral_values = [
            current_l,
            0.0,
            lane_width,
            -lane_width,
            safe_left,
            safe_right,
        ]
        lateral_values.extend(lane_centers)
        speed_values = self._cruise_speeds(context)
        rows = []
        for lateral in self._ordered_unique(lateral_values, prefer_positive=current_l >= 0.0):
            l_target = float(np.clip(lateral, safe_right, safe_left))
            for speed in speed_values:
                rows.append(np.array([l_target, speed], dtype=float))
        return rows

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
        edge_margin = float(self.config.road_edge_margin)
        safe_left = left - half_width - edge_margin
        safe_right = right + half_width + edge_margin
        if safe_right > safe_left:
            center = 0.5 * (right + left)
            safe_right = center
            safe_left = center
        return safe_right, safe_left

    def _obstacle_speeds(self, context: WarmStartContext, current_overlap: bool) -> list[float]:
        current_speed = float(context.problem.ego.s_v)
        target_speed = self._target_speed(context)
        if current_overlap:
            speeds = list(self.config.slow_speeds) + [0.6 * current_speed, target_speed]
        else:
            speeds = [0.6 * current_speed, current_speed, 0.8 * target_speed, target_speed]
        return self._clip_speeds(speeds)

    def _cruise_speeds(self, context: WarmStartContext) -> list[float]:
        current_speed = float(context.problem.ego.s_v)
        target_speed = self._target_speed(context)
        speeds = [current_speed + offset for offset in self.config.cruise_speed_offsets]
        speeds.append(0.8 * current_speed)
        speeds.extend(target_speed + offset for offset in self.config.target_speed_offsets)
        return self._clip_speeds(speeds)

    @staticmethod
    def _lane_centers(context: WarmStartContext) -> list[float]:
        metadata = dict(context.problem.metadata or {})
        lane_centers = metadata.get("lane_centers")
        if lane_centers is None:
            return []
        try:
            return [float(value) for value in lane_centers]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid lane_centers metadata: {lane_centers!r}") from exc

    def _target_speed(self, context: WarmStartContext) -> float:
        metadata = dict(context.problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {key} metadata: {metadata[key]!r}") from exc
        return float(self.config.max_terminal_speed)

    def _clip_speeds(self, speeds: Sequence[float]) -> list[float]:
        low = max(float(self.config.min_terminal_speed), 0.0)
        high = max(float(self.config.max_terminal_speed), low + 1e-3)
        return self._ordered_unique([float(np.clip(speed, low, high)) for speed in speeds], prefer_positive=True)

    def _blocked_ranges(self, actors: Sequence[ActorPrediction]) -> list[dict]:
        ranges = []
        keys = ("blocked_s_min", "blocked_s_max", "blocked_l_min", "blocked_l_max")
        for actor in actors:
            metadata = dict(actor.metadata or {})
            if any(key in metadata for key in keys) and not all(key in metadata for key in keys):
                missing = [key for key in keys if key not in metadata]
                raise ValueError(f"Actor {actor.actor_id!r} warm-start blocked range is missing fields: {missing}.")
            if all(key in metadata for key in keys):
                blocked = {
                    "s_min": float(metadata["blocked_s_min"]),
                    "s_max": float(metadata["blocked_s_max"]),
                    "l_min": float(metadata["blocked_l_min"]),
                    "l_max": float(metadata["blocked_l_max"]),
                    "actor_id": actor.actor_id,
                }
            elif ("s" in metadata) != ("l" in metadata):
                raise ValueError(f"Actor {actor.actor_id!r} warm-start metadata must provide both s and l.")
            elif "s" in metadata and "l" in metadata:
                half_length = 0.5 * float(actor.length)
                half_width = 0.5 * float(actor.width)
                blocked = {
                    "s_min": float(metadata["s"]) - half_length,
                    "s_max": float(metadata["s"]) + half_length,
                    "l_min": float(metadata["l"]) - half_width,
                    "l_max": float(metadata["l"]) + half_width,
                    "actor_id": actor.actor_id,
                }
            else:
                raise ValueError(
                    f"Actor {actor.actor_id!r} must provide blocked_s/l bounds or an s/l pose for warm start."
                )
            self._validate_blocked_range(blocked)
            ranges.append(self._inflate_blocked_range(blocked))
        return ranges

    @staticmethod
    def _validate_blocked_range(blocked: dict) -> None:
        values = np.asarray(
            [blocked["s_min"], blocked["s_max"], blocked["l_min"], blocked["l_max"]],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Actor {blocked['actor_id']!r} warm-start blocked range must be finite.")
        if float(blocked["s_min"]) > float(blocked["s_max"]) or float(blocked["l_min"]) > float(blocked["l_max"]):
            raise ValueError(f"Actor {blocked['actor_id']!r} warm-start blocked range has inverted bounds.")

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
    def _ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
        return float(a_min) <= float(b_max) and float(b_min) <= float(a_max)

    @staticmethod
    def _ordered_unique(values: Sequence[float], prefer_positive: bool) -> list[float]:
        unique = []
        seen = set()
        ordered = sorted(values, key=lambda value: (0 if (value >= 0.0) == prefer_positive else 1, abs(value)))
        for value in ordered:
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(value))
        return unique


class SvgdParticleWarmStartGenerator(TerminalStateWarmStartGenerator):
    """Dense deterministic particles for z = [l_end, v_end] SVGD trajectories."""

    @property
    def name(self) -> str:
        return "svgd_particle_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return context.trajectory_model.name == "svgd_particle_trajectory" and context.parameter_dim == 2

    def generate(self, context: WarmStartContext) -> np.ndarray:
        return super().generate(context)
