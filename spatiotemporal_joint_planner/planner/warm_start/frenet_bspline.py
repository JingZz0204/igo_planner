from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.planner.warm_start.base import WarmStartContext, WarmStartGenerator, finalize_warm_starts


@dataclass(frozen=True)
class FrenetBSplineWarmStartConfig:
    control_progress: tuple[float, ...] = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    terminal_progress: tuple[float, ...] = (0.0, 0.35, 0.6, 0.8, 1.0)
    speed_scales: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
    target_speed_offsets: tuple[float, ...] = (-2.0, 0.0, 2.0)


class FrenetBSplineWarmStartGenerator(WarmStartGenerator):
    """Warm starts for theta = [l_ctrl_mid, l_end, v_end]."""

    def __init__(self, config: Optional[FrenetBSplineWarmStartConfig] = None):
        self.config = config or FrenetBSplineWarmStartConfig()

    @property
    def name(self) -> str:
        return "frenet_bspline_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return context.trajectory_model.name == "frenet_bspline_trajectory" and context.parameter_dim == 3

    def generate(self, context: WarmStartContext) -> np.ndarray:
        rows = []
        if context.previous_parameters is not None:
            rows.append(np.asarray(context.previous_parameters, dtype=float))
        try:
            rows.append(context.trajectory_model.reference_parameters(context.problem))
        except Exception:
            pass

        low = np.asarray(context.lower_bound, dtype=float)
        high = np.asarray(context.upper_bound, dtype=float)
        current_l = float(context.problem.ego.l)
        target_l = float(np.clip(self._target_l(context), low[1], high[1]))
        current_speed = max(float(context.problem.ego.s_v), 0.0)
        target_speed = self._target_speed(context)

        terminal_l_values = self._terminal_l_values(context, current_l, target_l, low[1], high[1])
        speed_values = self._speed_values(current_speed, target_speed, low[2], high[2])
        global_control_values = self._ordered_unique(
            [current_l, target_l, low[0], high[0], *self._lane_centers(context)]
        )

        for l_end in terminal_l_values:
            direction_span = float(l_end) - current_l
            control_values = []
            for progress in self.config.control_progress:
                control_values.append(current_l + float(progress) * direction_span)
            control_values.extend(global_control_values)
            for l_ctrl_mid in self._ordered_unique(
                [float(np.clip(value, low[0], high[0])) for value in control_values]
            ):
                for speed in speed_values:
                    rows.append(np.array([float(l_ctrl_mid), float(l_end), float(speed)], dtype=float))

        rows.append(0.5 * (low + high))
        return finalize_warm_starts(rows, context)

    def _terminal_l_values(self, context: WarmStartContext, current_l: float, target_l: float, low: float, high: float) -> list[float]:
        values = [current_l]
        for progress in self.config.terminal_progress:
            values.append(current_l + float(progress) * (target_l - current_l))
        values.append(target_l)
        values.extend(self._lane_centers(context))
        values.extend([low, high])
        return self._ordered_unique([float(np.clip(value, low, high)) for value in values])

    def _speed_values(self, current_speed: float, target_speed: float, low: float, high: float) -> list[float]:
        values = [current_speed, target_speed, low, high]
        values.extend(float(scale) * current_speed for scale in self.config.speed_scales)
        values.extend(target_speed + float(offset) for offset in self.config.target_speed_offsets)
        return self._ordered_unique([float(np.clip(value, low, high)) for value in values])

    @staticmethod
    def _target_l(context: WarmStartContext) -> float:
        metadata = dict(context.problem.metadata or {})
        for key in ("reference_l", "target_lane_l", "target_l", "preferred_l"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        return float(context.problem.ego.l)

    @staticmethod
    def _target_speed(context: WarmStartContext) -> float:
        metadata = dict(context.problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        return float(context.problem.ego.s_v)

    @staticmethod
    def _lane_centers(context: WarmStartContext) -> list[float]:
        metadata = dict(context.problem.metadata or {})
        lane_centers = metadata.get("lane_centers")
        if lane_centers is None:
            return []
        try:
            return [float(value) for value in lane_centers]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _ordered_unique(values: Sequence[float]) -> list[float]:
        output = []
        seen = set()
        for value in values:
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            output.append(float(value))
        return output
