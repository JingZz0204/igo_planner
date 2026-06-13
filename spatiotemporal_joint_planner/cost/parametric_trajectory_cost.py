from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, CostBreakdown, CostResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.cost.base import CostFunction


def shaped_hinge(value, safe: float = 0.0, soft: float = 1.0, tail_gain: float = 0.25, cap: float = 4.0):
    value = np.maximum(np.asarray(value, dtype=float), 0.0)
    soft_span = max(float(soft) - float(safe), 1e-6)
    z = np.maximum((value - float(safe)) / soft_span, 0.0)
    smooth = z * z * (3.0 - 2.0 * np.minimum(z, 1.0))
    tail = 1.0 + float(tail_gain) * np.log1p(np.maximum(z - 1.0, 0.0))
    return np.minimum(np.where(z <= 1.0, smooth, tail), float(cap))


def pseudo_huber(x, delta: float = 1.0):
    x = np.asarray(x, dtype=float) / max(float(delta), 1e-6)
    return np.sqrt(1.0 + x * x) - 1.0


def saturate_cost(value: float, scale: float) -> float:
    value = max(float(value), 0.0)
    scale = max(float(scale), 1e-6)
    return float(value / (value + scale))


def topk_mean(values, fraction: float = 0.2) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(float(values.size) * float(fraction))))
    return float(np.mean(np.partition(values, -k)[-k:]))


def ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return float(a_min) <= float(b_max) and float(b_min) <= float(a_max)


@dataclass(frozen=True)
class ParametricTrajectoryCostConfig:
    ego_front: float = 4.05
    ego_rear: float = 0.90
    ego_width: float = 2.70
    planning_obstacle_s_buffer: float = 0.5
    planning_obstacle_l_buffer: float = 0.2
    road_edge_buffer: float = 1.0
    min_lateral_accel: float = -2.5
    max_lateral_accel: float = 2.5
    lateral_accel_zero_comfort: float = 1.2
    lateral_accel_zero_weight: float = 0.35
    min_kappa: float = -0.20
    max_kappa: float = 0.20
    kappa_zero_comfort: float = 0.04
    kappa_zero_weight: float = 0.50
    min_dkappa: float = -0.08
    max_dkappa: float = 0.08
    dkappa_zero_comfort: float = 0.04
    dkappa_zero_weight: float = 0.50
    min_lateral_jerk: float = -3.0
    max_lateral_jerk: float = 3.0
    lateral_jerk_zero_comfort: float = 1.0
    lateral_jerk_zero_weight: float = 0.35
    max_longitudinal_speed: float = 15.0
    speed_tracking_comfort: float = 2.5
    collision_score_scale: float = 0.25
    road_score_scale: float = 0.25
    dynamic_score_scale: float = 0.10
    speed_score_scale: float = 0.30
    efficiency_score_scale: float = 1.0
    comfort_score_scale: float = 1.0
    reference_score_scale: float = 1.0
    trajectory_certificate_enabled: bool = True
    trajectory_certificate_score_scale: float = 1.0
    terminal_future_feasibility_weight: float = 0.45
    terminal_recoverability_weight: float = 0.25
    terminal_progress_weight: float = 0.20
    terminal_speed_weight: float = 0.10
    terminal_tail_time: float = 1.0
    terminal_future_preview_time: float = 3.0
    terminal_future_min_preview_distance: float = 15.0
    terminal_future_max_preview_distance: float = 50.0
    terminal_future_step_s: float = 1.0
    terminal_future_slope_delta: float = 0.08
    terminal_future_min_free_width: float = 1.2
    terminal_dl_ds_comfort: float = 0.12
    terminal_kappa_comfort: float = 0.04
    terminal_dkappa_comfort: float = 0.04
    terminal_lateral_accel_comfort: float = 1.2
    terminal_lateral_jerk_comfort: float = 1.0
    terminal_boundary_clearance_comfort: float = 0.8
    terminal_speed_comfort: float = 2.5
    collision_decay_s: float = 12.0
    efficiency_progress_comfort: float = 8.0
    reference_lateral_comfort: float = 1.0
    reference_lateral_time_weight_start: float = 0.3
    reference_lateral_time_weight_end: float = 1.0
    reference_obstacle_relief_buffer: float = 12.0
    reference_obstacle_relief_weight: float = 0.2


class ParametricTrajectoryCost(CostFunction):
    """Hierarchical cost for fixed-horizon parameterized trajectories.

    The scalar is intentionally level-coded: collision terms dominate road
    boundary terms, road terms dominate dynamic terms, and dynamic terms
    dominate soft speed tracking.
    """

    def __init__(self, config: Optional[ParametricTrajectoryCostConfig] = None):
        self.config = config or ParametricTrajectoryCostConfig()

    @property
    def name(self) -> str:
        return "parametric_trajectory_cost"

    @staticmethod
    def _validate_trajectory(trajectory: Trajectory) -> None:
        required = ("t", "s", "l", "s_v", "l_v", "l_a", "kappa")
        arrays = {}
        for field_name in required:
            value = getattr(trajectory, field_name)
            if value is None:
                raise ValueError(f"Parametric trajectory cost requires trajectory.{field_name}.")
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size == 0:
                raise ValueError(f"Parametric trajectory cost requires non-empty trajectory.{field_name}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Parametric trajectory cost requires finite trajectory.{field_name}.")
            arrays[field_name] = array

        expected = arrays["t"].size
        mismatched = {name: values.size for name, values in arrays.items() if values.size != expected}
        if mismatched:
            raise ValueError(
                f"Parametric trajectory fields must share length {expected}; mismatched lengths: {mismatched}."
            )
        if expected < 2:
            raise ValueError("Parametric trajectory cost requires at least two trajectory samples.")
        if np.any(np.diff(arrays["t"]) <= 0.0):
            raise ValueError("Parametric trajectory timestamps must be strictly increasing.")

    @staticmethod
    def _validate_trajectory_batch(trajectory_batch: dict) -> None:
        required = ("t", "s", "l", "s_v", "l_v", "l_a", "kappa")
        missing = [name for name in required if name not in trajectory_batch or trajectory_batch[name] is None]
        if missing:
            raise ValueError(f"Vectorized parametric trajectory cost is missing batch fields: {missing}.")

        sample_shape = None
        for field_name in ("s", "l", "s_v", "l_v", "l_a", "kappa"):
            values = np.asarray(trajectory_batch[field_name], dtype=float)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
                raise ValueError(
                    f"Vectorized trajectory field {field_name!r} must have shape [batch, samples>=2], "
                    f"got {values.shape}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Vectorized trajectory field {field_name!r} must contain only finite values.")
            if sample_shape is None:
                sample_shape = values.shape
            elif values.shape != sample_shape:
                raise ValueError(
                    f"Vectorized trajectory fields must share shape {sample_shape}; "
                    f"{field_name!r} has shape {values.shape}."
                )

        t = np.asarray(trajectory_batch["t"], dtype=float)
        if t.ndim == 1:
            if t.size != sample_shape[1]:
                raise ValueError(
                    f"Vectorized trajectory timestamps must have {sample_shape[1]} samples, got {t.size}."
                )
            if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
                raise ValueError("Vectorized trajectory timestamps must be finite and strictly increasing.")
        elif t.ndim == 2:
            if t.shape != sample_shape:
                raise ValueError(
                    f"Vectorized trajectory timestamp shape must be {sample_shape}, got {t.shape}."
                )
            if not np.all(np.isfinite(t)) or np.any(np.diff(t, axis=1) <= 0.0):
                raise ValueError("Every vectorized trajectory timestamp row must be finite and strictly increasing.")
        else:
            raise ValueError(f"Vectorized trajectory timestamps must be one- or two-dimensional, got {t.shape}.")

    def evaluate(self, trajectory: Trajectory, problem: PlanningProblem) -> CostResult:
        self._validate_trajectory(trajectory)
        blocked_ranges = self._blocked_ranges(problem)
        running_terms = self._markov_running_terms(trajectory, problem, blocked_ranges)
        global_terms = self._global_hierarchy_terms(running_terms)
        certificate_terms = self._trajectory_level_certificate_terms(trajectory, problem, blocked_ranges)

        collision_flag = global_terms["collision_flag"]
        collision_cost = global_terms["collision_cost"]
        road_flag = global_terms["road_flag"]
        road_cost = global_terms["road_cost"]
        lateral_accel_flag = global_terms["lateral_accel_flag"]
        lateral_accel_limit_cost = global_terms["lateral_accel_limit_cost"]
        lateral_accel_zero_cost = global_terms["lateral_accel_zero_cost"]
        lateral_accel_score = global_terms["lateral_accel_score"]
        lateral_accel_hard_score = global_terms["lateral_accel_hard_score"]
        lateral_accel_soft_score = global_terms["lateral_accel_soft_score"]
        kappa_flag = global_terms["kappa_flag"]
        kappa_limit_cost = global_terms["kappa_limit_cost"]
        kappa_zero_cost = global_terms["kappa_zero_cost"]
        kappa_score = global_terms["kappa_score"]
        kappa_hard_score = global_terms["kappa_hard_score"]
        kappa_soft_score = global_terms["kappa_soft_score"]
        dkappa_flag = global_terms["dkappa_flag"]
        dkappa_limit_cost = global_terms["dkappa_limit_cost"]
        dkappa_zero_cost = global_terms["dkappa_zero_cost"]
        dkappa_score = global_terms["dkappa_score"]
        dkappa_hard_score = global_terms["dkappa_hard_score"]
        dkappa_soft_score = global_terms["dkappa_soft_score"]
        lateral_jerk_flag = global_terms["lateral_jerk_flag"]
        lateral_jerk_limit_cost = global_terms["lateral_jerk_limit_cost"]
        lateral_jerk_zero_cost = global_terms["lateral_jerk_zero_cost"]
        lateral_jerk_score = global_terms["lateral_jerk_score"]
        lateral_jerk_hard_score = global_terms["lateral_jerk_hard_score"]
        lateral_jerk_soft_score = global_terms["lateral_jerk_soft_score"]
        speed_flag = global_terms["speed_flag"]
        speed_limit_cost = global_terms["speed_limit_cost"]
        speed_deviation_cost = global_terms["speed_deviation_cost"]
        speed_score = global_terms["speed_score"]
        dynamic_score = global_terms["dynamic_score"]
        efficiency_cost = global_terms["efficiency_cost"]
        efficiency_score = global_terms["efficiency_score"]
        comfort_score = global_terms["comfort_score"]
        reference_lateral_target = global_terms["reference_lateral_target"]
        reference_lateral_cost = global_terms["reference_lateral_cost"]
        reference_score = global_terms["reference_score"]
        trajectory_certificate_score = certificate_terms["trajectory_certificate_score"]
        terminal_value_score = certificate_terms["terminal_value_score"]

        collision_score = saturate_cost(collision_cost, self.config.collision_score_scale)
        road_score = saturate_cost(road_cost, self.config.road_score_scale)
        hard_hierarchy_cost = float(
            1.0e9 * collision_score
            + 1.0e8 * road_score
            + 1.0e7 * kappa_hard_score
            + 1.0e6 * dkappa_hard_score
            + 1.0e5 * lateral_accel_hard_score
            + 1.0e4 * lateral_jerk_hard_score
        )
        soft_hierarchy_cost = float(
            1.0e3 * efficiency_score
            + 1.0e2 * reference_score
            + 1.0e1 * comfort_score
        )
        if bool(self.config.trajectory_certificate_enabled):
            soft_hierarchy_cost = float(
                1.0e3 * efficiency_score
                + 1.0e2 * trajectory_certificate_score
                + 1.0e2 * reference_score
                + 1.0e1 * comfort_score
            )

        total = float(hard_hierarchy_cost + soft_hierarchy_cost)
        terms = {
            "collision_flag": float(collision_flag),
            "collision_cost": float(collision_cost),
            "collision_score": float(collision_score),
            "road_flag": float(road_flag),
            "road_cost": float(road_cost),
            "road_score": float(road_score),
            "lateral_accel_flag": float(lateral_accel_flag),
            "lateral_accel_limit_cost": float(lateral_accel_limit_cost),
            "lateral_accel_zero_cost": float(lateral_accel_zero_cost),
            "lateral_accel_hard_cost": float(lateral_accel_limit_cost),
            "lateral_accel_soft_cost": float(lateral_accel_zero_cost),
            "lateral_accel_hard_score": float(lateral_accel_hard_score),
            "lateral_accel_soft_score": float(lateral_accel_soft_score),
            "lateral_accel_score": float(lateral_accel_score),
            "kappa_flag": float(kappa_flag),
            "kappa_limit_cost": float(kappa_limit_cost),
            "kappa_zero_cost": float(kappa_zero_cost),
            "kappa_hard_cost": float(kappa_limit_cost),
            "kappa_soft_cost": float(kappa_zero_cost),
            "kappa_hard_score": float(kappa_hard_score),
            "kappa_soft_score": float(kappa_soft_score),
            "kappa_score": float(kappa_score),
            "dkappa_flag": float(dkappa_flag),
            "dkappa_limit_cost": float(dkappa_limit_cost),
            "dkappa_zero_cost": float(dkappa_zero_cost),
            "dkappa_hard_cost": float(dkappa_limit_cost),
            "dkappa_soft_cost": float(dkappa_zero_cost),
            "dkappa_hard_score": float(dkappa_hard_score),
            "dkappa_soft_score": float(dkappa_soft_score),
            "dkappa_score": float(dkappa_score),
            "lateral_jerk_flag": float(lateral_jerk_flag),
            "lateral_jerk_limit_cost": float(lateral_jerk_limit_cost),
            "lateral_jerk_zero_cost": float(lateral_jerk_zero_cost),
            "lateral_jerk_hard_cost": float(lateral_jerk_limit_cost),
            "lateral_jerk_soft_cost": float(lateral_jerk_zero_cost),
            "lateral_jerk_hard_score": float(lateral_jerk_hard_score),
            "lateral_jerk_soft_score": float(lateral_jerk_soft_score),
            "lateral_jerk_score": float(lateral_jerk_score),
            "speed_flag": float(speed_flag),
            "speed_limit_cost": float(speed_limit_cost),
            "speed_deviation_cost": float(speed_deviation_cost),
            "speed_max_deviation_cost": float(speed_deviation_cost),
            "speed_hard_cost": float(speed_limit_cost),
            "speed_soft_cost": float(speed_deviation_cost),
            "speed_score": float(speed_score),
            "speed_tracking_cost": float(speed_deviation_cost),
            "efficiency_cost": float(efficiency_cost),
            "efficiency_score": float(efficiency_score),
            "comfort_score": float(comfort_score),
            "reference_lateral_cost": float(reference_lateral_cost),
            "reference_lateral_target": float(reference_lateral_target),
            "reference_score": float(reference_score),
            "dynamic_score": float(dynamic_score),
            "trajectory_certificate_score": float(trajectory_certificate_score),
            "trajectory_certificate_cost": float(certificate_terms["trajectory_certificate_cost"]),
            "terminal_value_score": float(terminal_value_score),
            "terminal_value_cost": float(certificate_terms["terminal_value_cost"]),
            "terminal_future_feasibility_score": float(certificate_terms["terminal_future_feasibility_score"]),
            "terminal_recoverability_score": float(certificate_terms["terminal_recoverability_score"]),
            "terminal_progress_score": float(certificate_terms["terminal_progress_score"]),
            "terminal_speed_score": float(certificate_terms["terminal_speed_score"]),
            "terminal_s": float(certificate_terms["terminal_s"]),
            "terminal_l": float(certificate_terms["terminal_l"]),
            "terminal_v": float(certificate_terms["terminal_v"]),
            "terminal_dl_ds": float(certificate_terms["terminal_dl_ds"]),
            "terminal_future_preview_distance": float(certificate_terms["terminal_future_preview_distance"]),
            "terminal_future_feasible_progress": float(certificate_terms["terminal_future_feasible_progress"]),
            "occupation_style_score": float(certificate_terms["occupation_style_score"]),
            "certificate_slack_score": float(certificate_terms["certificate_slack_score"]),
            "running_cost": float(global_terms["running_cost"]),
            "hard_hierarchy_cost": float(hard_hierarchy_cost),
            "soft_hierarchy_cost": float(soft_hierarchy_cost),
            "global_hierarchy_cost": float(hard_hierarchy_cost + soft_hierarchy_cost),
        }
        hard_violation = bool(
            collision_flag > 0.0
            or road_flag > 0.0
            or lateral_accel_flag > 0.0
            or kappa_flag > 0.0
            or dkappa_flag > 0.0
            or lateral_jerk_flag > 0.0
            or speed_flag > 0.0
        )
        return CostResult(
            total=total,
            breakdown=CostBreakdown(terms=terms, hard_violation=hard_violation),
            feasible=not hard_violation,
            metadata={
                "cost": self.name,
                "blocked_ranges": blocked_ranges,
                "sort_key": (
                    round(float(collision_score), 6),
                    round(float(road_score), 6),
                    round(float(kappa_hard_score), 6),
                    round(float(dkappa_hard_score), 6),
                    round(float(lateral_accel_hard_score), 6),
                    round(float(lateral_jerk_hard_score), 6),
                    round(float(trajectory_certificate_score), 6),
                    round(float(efficiency_score), 6),
                    round(float(comfort_score), 6),
                    round(float(reference_score), 6),
                ),
            },
        )

    def _markov_running_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        """Return per-sample Markov-style running costs.

        These arrays are pointwise costs of the current trajectory state and
        environment at each sample. The hierarchical part of ``evaluate`` may
        still aggregate them with max/top-k for safety dominance, but the soft
        objectives are available as additive running costs.
        """

        terms = {}
        terms.update(self._collision_running_terms(trajectory, blocked_ranges))
        terms.update(self._road_running_terms(trajectory, problem))
        terms.update(self._lateral_accel_running_terms(trajectory))
        terms.update(self._shape_running_terms(trajectory))
        terms.update(self._speed_running_terms(trajectory, problem))
        terms.update(self._reference_lateral_running_terms(trajectory, problem, blocked_ranges))
        return terms

    def evaluate_lattice_batch(self, trajectory_batch: dict, problem: PlanningProblem) -> np.ndarray:
        """Vectorized total-cost evaluation for lattice_trajectory batches."""

        return self.evaluate_batch(trajectory_batch, problem)

    def evaluate_bezier_batch(self, trajectory_batch: dict, problem: PlanningProblem) -> np.ndarray:
        """Vectorized total-cost evaluation for bezier_trajectory batches."""

        return self.evaluate_batch(trajectory_batch, problem)

    def evaluate_batch(self, trajectory_batch: dict, problem: PlanningProblem) -> np.ndarray:
        """Vectorized total-cost evaluation for trajectory array batches."""

        self._validate_trajectory_batch(trajectory_batch)
        blocked_ranges = self._blocked_ranges(problem)
        terms = self._lattice_batch_running_terms(trajectory_batch, problem, blocked_ranges)
        scores = self._lattice_batch_hierarchy_scores(terms)
        certificate_score = self._lattice_batch_certificate_score(trajectory_batch, problem, blocked_ranges)

        collision_score = self._saturate_array(scores["collision_cost"], self.config.collision_score_scale)
        road_score = self._saturate_array(scores["road_cost"], self.config.road_score_scale)
        hard_hierarchy_cost = (
            1.0e9 * collision_score
            + 1.0e8 * road_score
            + 1.0e7 * scores["kappa_hard_score"]
            + 1.0e6 * scores["dkappa_hard_score"]
            + 1.0e5 * scores["lateral_accel_hard_score"]
            + 1.0e4 * scores["lateral_jerk_hard_score"]
        )
        soft_hierarchy_cost = (
            1.0e3 * scores["efficiency_score"]
            + 1.0e2 * scores["reference_score"]
            + 1.0e1 * scores["comfort_score"]
        )
        if bool(self.config.trajectory_certificate_enabled):
            soft_hierarchy_cost = (
                1.0e3 * scores["efficiency_score"]
                + 1.0e2 * certificate_score
                + 1.0e2 * scores["reference_score"]
                + 1.0e1 * scores["comfort_score"]
            )
        total = np.asarray(hard_hierarchy_cost + soft_hierarchy_cost, dtype=float)
        if not np.all(np.isfinite(total)):
            raise FloatingPointError("Vectorized parametric trajectory cost produced non-finite values.")
        return total

    def _lattice_batch_running_terms(
        self,
        trajectory_batch: dict,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        s = np.asarray(trajectory_batch["s"], dtype=float)
        l = np.asarray(trajectory_batch["l"], dtype=float)
        t = np.asarray(trajectory_batch["t"], dtype=float)
        s_v = np.asarray(trajectory_batch["s_v"], dtype=float)
        l_a = np.asarray(trajectory_batch["l_a"], dtype=float)
        kappa_values = np.asarray(trajectory_batch["kappa"], dtype=float)
        dkappa = self._batch_dkappa_values(trajectory_batch, kappa_values)
        lateral_jerk = self._batch_gradient(l_a, t)

        terms = {}
        terms.update(self._lattice_batch_collision_terms(s, l, t, blocked_ranges))
        terms.update(self._lattice_batch_road_terms(l, problem))
        terms.update(
            self._lattice_batch_bounded_zero_terms(
                "lateral_accel",
                l_a,
                min_value=float(self.config.min_lateral_accel),
                max_value=float(self.config.max_lateral_accel),
                zero_comfort=float(self.config.lateral_accel_zero_comfort),
                zero_weight=float(self.config.lateral_accel_zero_weight),
            )
        )
        terms.update(
            self._lattice_batch_bounded_zero_terms(
                "kappa",
                kappa_values,
                min_value=float(self.config.min_kappa),
                max_value=float(self.config.max_kappa),
                zero_comfort=float(self.config.kappa_zero_comfort),
                zero_weight=float(self.config.kappa_zero_weight),
            )
        )
        terms.update(
            self._lattice_batch_bounded_zero_terms(
                "dkappa",
                dkappa,
                min_value=float(self.config.min_dkappa),
                max_value=float(self.config.max_dkappa),
                zero_comfort=float(self.config.dkappa_zero_comfort),
                zero_weight=float(self.config.dkappa_zero_weight),
            )
        )
        terms.update(
            self._lattice_batch_bounded_zero_terms(
                "lateral_jerk",
                lateral_jerk,
                min_value=float(self.config.min_lateral_jerk),
                max_value=float(self.config.max_lateral_jerk),
                zero_comfort=float(self.config.lateral_jerk_zero_comfort),
                zero_weight=float(self.config.lateral_jerk_zero_weight),
            )
        )
        terms.update(self._lattice_batch_speed_terms(s_v, problem))
        terms.update(self._lattice_batch_reference_terms(s, l, problem, blocked_ranges))
        return terms

    def _lattice_batch_hierarchy_scores(self, terms: dict) -> dict:
        collision_cost = self._batch_topk_max_cost(terms["collision_running"], fraction=0.15)
        road_cost = self._batch_topk_max_cost(terms["road_running"], fraction=0.15)
        lateral_accel_limit_cost = np.mean(terms["lateral_accel_limit_running"], axis=1)
        kappa_limit_cost = np.mean(terms["kappa_limit_running"], axis=1)
        dkappa_limit_cost = np.mean(terms["dkappa_limit_running"], axis=1)
        lateral_jerk_limit_cost = np.mean(terms["lateral_jerk_limit_running"], axis=1)
        speed_limit_cost = np.mean(terms["speed_limit_running"], axis=1)
        lateral_accel_zero_cost = np.mean(terms["lateral_accel_zero_running"], axis=1)
        kappa_zero_cost = np.mean(terms["kappa_zero_running"], axis=1)
        dkappa_zero_cost = np.mean(terms["dkappa_zero_running"], axis=1)
        lateral_jerk_zero_cost = np.mean(terms["lateral_jerk_zero_running"], axis=1)
        speed_deviation_cost = np.mean(terms["speed_max_deviation_running"], axis=1)
        efficiency_cost = np.mean(terms["efficiency_running"], axis=1)
        reference_lateral_cost = np.mean(terms["reference_lateral_running"], axis=1)

        lateral_accel_hard_score = self._saturate_array(lateral_accel_limit_cost, self.config.dynamic_score_scale)
        kappa_hard_score = self._saturate_array(kappa_limit_cost, self.config.dynamic_score_scale)
        dkappa_hard_score = self._saturate_array(dkappa_limit_cost, self.config.dynamic_score_scale)
        lateral_jerk_hard_score = self._saturate_array(lateral_jerk_limit_cost, self.config.dynamic_score_scale)
        lateral_accel_soft_score = self._saturate_array(lateral_accel_zero_cost, self.config.comfort_score_scale)
        kappa_soft_score = self._saturate_array(kappa_zero_cost, self.config.comfort_score_scale)
        dkappa_soft_score = self._saturate_array(dkappa_zero_cost, self.config.comfort_score_scale)
        lateral_jerk_soft_score = self._saturate_array(lateral_jerk_zero_cost, self.config.comfort_score_scale)
        return {
            "collision_cost": collision_cost,
            "road_cost": road_cost,
            "lateral_accel_hard_score": lateral_accel_hard_score,
            "kappa_hard_score": kappa_hard_score,
            "dkappa_hard_score": dkappa_hard_score,
            "lateral_jerk_hard_score": lateral_jerk_hard_score,
            "efficiency_score": self._saturate_array(
                efficiency_cost + speed_deviation_cost, self.config.efficiency_score_scale
            ),
            "comfort_score": np.clip(
                np.mean(
                    np.vstack(
                        [
                            lateral_accel_soft_score,
                            kappa_soft_score,
                            dkappa_soft_score,
                            lateral_jerk_soft_score,
                        ]
                    ),
                    axis=0,
                ),
                0.0,
                1.0,
            ),
            "reference_score": self._saturate_array(reference_lateral_cost, self.config.reference_score_scale),
            "speed_score": self._saturate_array(speed_limit_cost + speed_deviation_cost, self.config.speed_score_scale),
        }

    def _lattice_batch_collision_terms(
        self,
        s: np.ndarray,
        l: np.ndarray,
        t: np.ndarray,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        running = np.zeros_like(s, dtype=float)
        overlap = np.zeros_like(s, dtype=float)
        if not blocked_ranges:
            return {"collision_running": running, "collision_overlap": overlap}

        t_grid = self._batch_time_grid(t, s.shape)
        ego_s_min = s - float(self.config.ego_rear)
        ego_s_max = s + float(self.config.ego_front)
        half_width = 0.5 * float(self.config.ego_width)
        ego_l_min = l - half_width
        ego_l_max = l + half_width
        decay = 0.2 + 0.8 * np.exp(-np.maximum(s - s[:, :1], 0.0) / max(float(self.config.collision_decay_s), 1e-3))
        for blocked in blocked_ranges:
            blocked_s_min = self._blocked_value_at_times(blocked, "s_min", t_grid)
            blocked_s_max = self._blocked_value_at_times(blocked, "s_max", t_grid)
            blocked_l_min = self._blocked_value_at_times(blocked, "l_min", t_grid)
            blocked_l_max = self._blocked_value_at_times(blocked, "l_max", t_grid)
            s_mask = (ego_s_min <= blocked_s_max) & (blocked_s_min <= ego_s_max)
            l_mask = (ego_l_min <= blocked_l_max) & (blocked_l_min <= ego_l_max)
            mask = s_mask & l_mask
            if not np.any(mask):
                continue
            s_overlap = np.minimum(ego_s_max, blocked_s_max) - np.maximum(ego_s_min, blocked_s_min)
            l_overlap = np.minimum(ego_l_max, blocked_l_max) - np.maximum(ego_l_min, blocked_l_min)
            penetration = np.maximum(np.minimum(s_overlap, l_overlap), 0.0)
            sample_cost = decay * (
                1.0 + shaped_hinge(penetration, safe=0.0, soft=0.6, tail_gain=0.35, cap=3.0)
            )
            running = np.maximum(running, np.where(mask, sample_cost, 0.0))
            overlap = np.maximum(overlap, mask.astype(float))
        return {"collision_running": running, "collision_overlap": overlap}

    def _lattice_batch_road_terms(self, l: np.ndarray, problem: PlanningProblem) -> dict:
        half_width = 0.5 * float(self.config.ego_width)
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        ego_l_min = l - half_width
        ego_l_max = l + half_width
        left_excess = np.maximum(ego_l_max - left, 0.0)
        right_excess = np.maximum(right - ego_l_min, 0.0)
        excess = np.maximum(left_excess, right_excess)
        edge_clearance = np.minimum(left - ego_l_max, ego_l_min - right)
        edge_pressure = np.maximum(float(self.config.road_edge_buffer) - edge_clearance, 0.0)
        violation_cost = shaped_hinge(excess, safe=0.0, soft=0.6, tail_gain=0.25, cap=3.0)
        edge_cost = 0.35 * shaped_hinge(
            edge_pressure,
            safe=0.0,
            soft=max(float(self.config.road_edge_buffer), 1e-3),
            tail_gain=0.1,
            cap=1.5,
        )
        return {
            "road_running": np.asarray(violation_cost + edge_cost, dtype=float),
            "road_violation": np.asarray(excess > 1e-6, dtype=float),
        }

    @staticmethod
    def _batch_time_grid(t: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        time = np.asarray(t, dtype=float)
        if time.shape == target_shape:
            return time
        if time.ndim == 1 and len(target_shape) == 2 and time.size == target_shape[1]:
            return np.broadcast_to(time[None, :], target_shape)
        raise ValueError(f"Trajectory timestamp shape {time.shape} cannot align with batch shape {target_shape}.")

    @staticmethod
    def _blocked_value_at_times(blocked: dict, key: str, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        temporal = blocked.get("temporal")
        if not temporal:
            return np.full(times.shape, float(blocked[key]), dtype=float)

        source_t = np.asarray(temporal["t"], dtype=float).reshape(-1)
        source_v = np.asarray(temporal[key], dtype=float).reshape(-1)
        if source_t.size != source_v.size or source_t.size == 0:
            raise ValueError(f"Temporal blocked range field {key!r} must align with non-empty timestamps.")
        flat = np.interp(times.reshape(-1), source_t, source_v, left=source_v[0], right=source_v[-1])
        return flat.reshape(times.shape)

    @staticmethod
    def _blocked_scalar_at_time(blocked: dict, key: str, time: float) -> float:
        value = ParametricTrajectoryCost._blocked_value_at_times(
            blocked,
            key,
            np.array([float(time)], dtype=float),
        )
        return float(value[0])

    def _lattice_batch_bounded_zero_terms(
        self,
        prefix: str,
        values: np.ndarray,
        min_value: float,
        max_value: float,
        zero_comfort: float,
        zero_weight: float,
    ) -> dict:
        values = np.asarray(values, dtype=float)
        low = min(float(min_value), float(max_value))
        high = max(float(min_value), float(max_value))
        lower_excess = np.maximum(low - values, 0.0)
        upper_excess = np.maximum(values - high, 0.0)
        excess = np.maximum(lower_excess, upper_excess)
        limit_scale = max(0.25 * max(abs(low), abs(high)), 1e-6)
        return {
            f"{prefix}_limit_running": shaped_hinge(excess, safe=0.0, soft=limit_scale, tail_gain=0.35, cap=3.0),
            f"{prefix}_zero_running": float(zero_weight)
            * pseudo_huber(values / max(float(zero_comfort), 1e-6), delta=1.0),
            f"{prefix}_violation": np.asarray(excess > 1e-6, dtype=float),
        }

    def _lattice_batch_speed_terms(self, s_v: np.ndarray, problem: PlanningProblem) -> dict:
        s_v = np.asarray(s_v, dtype=float)
        max_speed = max(float(self.config.max_longitudinal_speed), 1e-3)
        comfort = max(float(self.config.speed_tracking_comfort), 1e-3)
        speed_tracking = pseudo_huber((s_v - max_speed) / comfort, delta=1.0)
        target_speed = self._target_speed(problem)
        low_speed_shortfall = np.maximum(float(target_speed) - s_v, 0.0)
        efficiency_running = pseudo_huber(low_speed_shortfall / comfort, delta=1.0)
        reverse_excess = np.maximum(-s_v, 0.0)
        speed_excess = np.maximum(s_v - max_speed, 0.0)
        speed_pressure = shaped_hinge(speed_excess, safe=0.0, soft=1.0, tail_gain=0.35, cap=3.0)
        reverse_pressure = shaped_hinge(reverse_excess, safe=0.0, soft=0.5, tail_gain=0.35, cap=3.0)
        return {
            "speed_limit_running": speed_pressure + reverse_pressure,
            "speed_tracking_running": speed_tracking,
            "speed_max_deviation_running": speed_tracking,
            "efficiency_running": efficiency_running,
            "speed_violation": np.asarray((reverse_excess > 1e-6) | (speed_excess > 1e-6), dtype=float),
        }

    def _lattice_batch_reference_terms(
        self,
        s: np.ndarray,
        l: np.ndarray,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        target = self._reference_lateral_target(problem)
        comfort = max(float(self.config.reference_lateral_comfort), 1e-6)
        deviation_cost = pseudo_huber((l - float(target)) / comfort, delta=1.0)
        n = l.shape[1]
        time_weight = np.linspace(
            float(self.config.reference_lateral_time_weight_start),
            float(self.config.reference_lateral_time_weight_end),
            num=n,
            dtype=float,
        )
        relief = np.ones_like(l, dtype=float)
        buffer = max(float(self.config.reference_obstacle_relief_buffer), 0.0)
        relief_weight = float(np.clip(float(self.config.reference_obstacle_relief_weight), 0.0, 1.0))
        for blocked in blocked_ranges:
            near = (s >= float(blocked["s_min"]) - buffer) & (s <= float(blocked["s_max"]) + buffer)
            relief[near] = np.minimum(relief[near], relief_weight)
        return {
            "reference_lateral_target": float(target),
            "reference_lateral_running": time_weight[None, :] * relief * deviation_cost,
        }

    def _lattice_batch_certificate_score(
        self,
        trajectory_batch: dict,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> np.ndarray:
        s = np.asarray(trajectory_batch["s"], dtype=float)
        if not bool(self.config.trajectory_certificate_enabled):
            return np.zeros((s.shape[0],), dtype=float)

        l = np.asarray(trajectory_batch["l"], dtype=float)
        s_v = np.asarray(trajectory_batch["s_v"], dtype=float)
        l_v = np.asarray(trajectory_batch["l_v"], dtype=float)
        l_a = np.asarray(trajectory_batch["l_a"], dtype=float)
        t = np.asarray(trajectory_batch["t"], dtype=float)
        kappa_values = np.asarray(trajectory_batch["kappa"], dtype=float)
        dkappa = self._batch_dkappa_values(trajectory_batch, kappa_values)
        lateral_jerk = self._batch_gradient(l_a, t)

        s_t = s[:, -1]
        l_t = l[:, -1]
        v_t = np.maximum(s_v[:, -1], 0.0)
        dl_ds_t = self._batch_terminal_dl_ds(s, l, s_v, l_v, t)
        preview_distance = np.clip(
            v_t * max(float(self.config.terminal_future_preview_time), 0.0),
            max(float(self.config.terminal_future_min_preview_distance), 0.0),
            max(float(self.config.terminal_future_max_preview_distance), 1e-6),
        )
        future_score, feasible_progress = self._lattice_batch_terminal_future_rollout_score(
            s_t=s_t,
            l_t=l_t,
            dl_ds_t=dl_ds_t,
            preview_distance=preview_distance,
            problem=problem,
            blocked_ranges=blocked_ranges,
        )
        recoverability_score = self._lattice_batch_terminal_recoverability_score(
            l_t=l_t,
            dl_ds_t=dl_ds_t,
            kappa_t=kappa_values[:, -1],
            dkappa_t=dkappa[:, -1],
            lateral_accel_t=l_a[:, -1],
            lateral_jerk_t=lateral_jerk[:, -1],
            problem=problem,
        )
        progress_score = self._unit_hinge_array(
            np.maximum(preview_distance - feasible_progress, 0.0), soft=np.maximum(preview_distance, 1.0)
        )
        speed_score = self._unit_pseudo_huber_array(
            np.maximum(float(self._target_speed(problem)) - v_t, 0.0),
            comfort=float(self.config.terminal_speed_comfort),
        )
        terminal_value = self._weighted_unit_score_array(
            [
                (future_score, self.config.terminal_future_feasibility_weight),
                (recoverability_score, self.config.terminal_recoverability_weight),
                (progress_score, self.config.terminal_progress_weight),
                (speed_score, self.config.terminal_speed_weight),
            ]
        )
        return np.clip(
            terminal_value / max(float(self.config.trajectory_certificate_score_scale), 1e-6),
            0.0,
            1.0,
        )

    def _lattice_batch_terminal_recoverability_score(
        self,
        l_t: np.ndarray,
        dl_ds_t: np.ndarray,
        kappa_t: np.ndarray,
        dkappa_t: np.ndarray,
        lateral_accel_t: np.ndarray,
        lateral_jerk_t: np.ndarray,
        problem: PlanningProblem,
    ) -> np.ndarray:
        center_right, center_left = self._center_lateral_interval(problem)
        boundary_clearance = np.minimum(l_t - center_right, center_left - l_t)
        boundary_score = self._unit_hinge_array(
            float(self.config.terminal_boundary_clearance_comfort) - boundary_clearance,
            soft=max(float(self.config.terminal_boundary_clearance_comfort), 1e-6),
        )
        return self._weighted_unit_score_array(
            [
                (self._unit_pseudo_huber_array(np.abs(dl_ds_t), self.config.terminal_dl_ds_comfort), 1.0),
                (self._unit_pseudo_huber_array(np.abs(kappa_t), self.config.terminal_kappa_comfort), 1.0),
                (self._unit_pseudo_huber_array(np.abs(dkappa_t), self.config.terminal_dkappa_comfort), 1.0),
                (
                    self._unit_pseudo_huber_array(
                        np.abs(lateral_accel_t), self.config.terminal_lateral_accel_comfort
                    ),
                    1.0,
                ),
                (
                    self._unit_pseudo_huber_array(
                        np.abs(lateral_jerk_t), self.config.terminal_lateral_jerk_comfort
                    ),
                    1.0,
                ),
                (boundary_score, 1.0),
            ]
        )

    def _lattice_batch_terminal_future_rollout_score(
        self,
        s_t: np.ndarray,
        l_t: np.ndarray,
        dl_ds_t: np.ndarray,
        preview_distance: np.ndarray,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = s_t.shape[0]
        step_s = max(float(self.config.terminal_future_step_s), 0.25)
        max_preview = max(float(np.max(preview_distance)), step_s)
        n_steps = max(1, int(math.ceil(max_preview / step_s)))
        fractions = np.linspace(1.0 / float(n_steps), 1.0, num=n_steps, dtype=float)
        slope_delta = max(float(self.config.terminal_future_slope_delta), 0.0)
        primitive_targets = [
            dl_ds_t,
            np.zeros_like(dl_ds_t),
            dl_ds_t + slope_delta,
            dl_ds_t - slope_delta,
        ]

        best_score = np.ones((batch_size,), dtype=float)
        best_feasible_progress = np.zeros((batch_size,), dtype=float)
        for target_slope in primitive_targets:
            distances = preview_distance[:, None] * fractions[None, :]
            s_hat = s_t[:, None] + distances
            l_hat = l_t[:, None] + dl_ds_t[:, None] * distances + 0.5 * (
                target_slope[:, None] - dl_ds_t[:, None]
            ) * (distances * distances / np.maximum(preview_distance[:, None], 1e-6))
            sample_score, hard = self._lattice_batch_terminal_future_sample_score(
                s_hat=s_hat,
                l_hat=l_hat,
                problem=problem,
                blocked_ranges=blocked_ranges,
            )
            primitive_score = self._batch_topk_max_cost(sample_score, fraction=0.2)
            has_hard = np.any(hard, axis=1)
            first_hard = np.argmax(hard, axis=1)
            row = np.arange(batch_size)
            prev_index = np.maximum(first_hard - 1, 0)
            progress_if_blocked = np.where(first_hard > 0, distances[row, prev_index], 0.0)
            feasible_progress = np.where(has_hard, progress_if_blocked, preview_distance)
            best_score = np.minimum(best_score, primitive_score)
            best_feasible_progress = np.maximum(best_feasible_progress, feasible_progress)
        return np.clip(best_score, 0.0, 1.0), np.clip(best_feasible_progress, 0.0, preview_distance)

    def _lattice_batch_terminal_future_sample_score(
        self,
        s_hat: np.ndarray,
        l_hat: np.ndarray,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        center_right, center_left = self._center_lateral_interval(problem)
        if center_left <= center_right:
            return np.ones_like(s_hat, dtype=float), np.ones_like(s_hat, dtype=bool)

        hard = (l_hat < center_right) | (l_hat > center_left)
        interval_min = np.full_like(l_hat, center_right, dtype=float)
        interval_max = np.full_like(l_hat, center_left, dtype=float)
        half_width = 0.5 * float(self.config.ego_width)
        ego_s_min = s_hat - float(self.config.ego_rear)
        ego_s_max = s_hat + float(self.config.ego_front)
        ego_l_min = l_hat - half_width
        ego_l_max = l_hat + half_width
        for blocked in blocked_ranges:
            active_s = (ego_s_min <= float(blocked["s_max"])) & (float(blocked["s_min"]) <= ego_s_max)
            hard |= active_s & (ego_l_min <= float(blocked["l_max"])) & (float(blocked["l_min"]) <= ego_l_max)

            blocked_l_min = float(blocked["l_min"]) - half_width
            blocked_l_max = float(blocked["l_max"]) + half_width
            inside_center_block = active_s & (l_hat >= blocked_l_min) & (l_hat <= blocked_l_max)
            hard |= inside_center_block
            left_side = active_s & (l_hat < blocked_l_min)
            right_side = active_s & (l_hat > blocked_l_max)
            interval_max = np.where(left_side, np.minimum(interval_max, blocked_l_min), interval_max)
            interval_min = np.where(right_side, np.maximum(interval_min, blocked_l_max), interval_min)

        containing_width = np.maximum(interval_max - interval_min, 0.0)
        hard |= containing_width <= 1e-6
        road_width = max(center_left - center_right, 1e-6)
        desired_width = min(float(self.config.terminal_future_min_free_width), max(0.3 * road_width, 0.5))
        space_score = self._unit_hinge_array(desired_width - containing_width, soft=max(desired_width, 1e-6))
        edge_clearance = np.minimum(l_hat - center_right, center_left - l_hat)
        edge_score = 0.2 * self._unit_hinge_array(
            float(self.config.terminal_boundary_clearance_comfort) - edge_clearance,
            soft=max(float(self.config.terminal_boundary_clearance_comfort), 1e-6),
        )
        score = np.maximum(space_score, edge_score)
        return np.where(hard, 1.0, np.clip(score, 0.0, 1.0)), hard

    def _batch_terminal_dl_ds(
        self,
        s: np.ndarray,
        l: np.ndarray,
        s_v: np.ndarray,
        l_v: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        del s_v, l_v
        tail_time = max(float(self.config.terminal_tail_time), 0.0)
        mask = t >= float(t[-1]) - tail_time
        if np.count_nonzero(mask) < 2:
            raise ValueError("Terminal slope estimation requires at least two samples inside terminal_tail_time.")
        s_tail = s[:, mask]
        l_tail = l[:, mask]
        s_mean = np.mean(s_tail, axis=1, keepdims=True)
        l_mean = np.mean(l_tail, axis=1, keepdims=True)
        denom = np.sum((s_tail - s_mean) ** 2, axis=1)
        if np.any(denom <= 1e-9):
            raise ValueError("Terminal slope estimation requires longitudinal motion inside terminal_tail_time.")
        numer = np.sum((s_tail - s_mean) * (l_tail - l_mean), axis=1)
        return numer / denom

    def _batch_dkappa_values(self, trajectory_batch: dict, kappa: np.ndarray) -> np.ndarray:
        x = trajectory_batch.get("x")
        y = trajectory_batch.get("y")
        if x is not None and y is not None:
            x_values = np.asarray(x, dtype=float)
            y_values = np.asarray(y, dtype=float)
            ds = np.hypot(np.diff(x_values, axis=1), np.diff(y_values, axis=1))
            coordinate = np.concatenate([np.zeros((x_values.shape[0], 1), dtype=float), np.cumsum(ds, axis=1)], axis=1)
        else:
            coordinate = np.asarray(trajectory_batch["s"], dtype=float)
        return self._batch_gradient(np.asarray(kappa, dtype=float), coordinate)

    @staticmethod
    def _batch_gradient(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        coordinate = np.asarray(coordinate, dtype=float)
        if values.shape[-1] <= 1:
            return np.zeros_like(values)
        if coordinate.ndim == 1:
            edge_order = 2 if values.shape[-1] >= 3 else 1
            return np.gradient(values, coordinate, axis=-1, edge_order=edge_order)

        out = np.zeros_like(values, dtype=float)
        for row in range(values.shape[0]):
            row_values = values[row]
            row_coordinate = coordinate[row]
            if not np.all(np.isfinite(row_values)) or not np.all(np.isfinite(row_coordinate)):
                raise ValueError("Trajectory derivative inputs must contain only finite values.")
            if np.any(np.diff(row_coordinate) <= 1e-6):
                raise ValueError("Trajectory derivative coordinate must be strictly increasing.")
            edge_order = 2 if row_values.size >= 3 else 1
            out[row] = np.gradient(row_values, row_coordinate, edge_order=edge_order)
        return out

    @staticmethod
    def _batch_topk_max_cost(values: np.ndarray, fraction: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] == 0:
            return np.zeros((values.shape[0],), dtype=float)
        k = max(1, int(math.ceil(values.shape[1] * float(fraction))))
        top = np.partition(values, -k, axis=1)[:, -k:]
        return 0.7 * np.mean(top, axis=1) + 0.3 * np.max(values, axis=1)

    @staticmethod
    def _saturate_array(value, scale: float) -> np.ndarray:
        value = np.maximum(np.asarray(value, dtype=float), 0.0)
        scale = max(float(scale), 1e-6)
        return value / (value + scale)

    @staticmethod
    def _unit_hinge_array(value, soft) -> np.ndarray:
        value = np.maximum(np.asarray(value, dtype=float), 0.0)
        soft = np.maximum(np.asarray(soft, dtype=float), 1e-6)
        z = np.maximum(value / soft, 0.0)
        smooth = z * z * (3.0 - 2.0 * np.minimum(z, 1.0))
        tail = 1.0 + 0.25 * np.log1p(np.maximum(z - 1.0, 0.0))
        return np.clip(np.where(z <= 1.0, smooth, tail), 0.0, 1.0)

    @staticmethod
    def _unit_pseudo_huber_array(value, comfort: float) -> np.ndarray:
        raw = pseudo_huber(np.asarray(value, dtype=float) / max(float(comfort), 1e-6), delta=1.0)
        return np.clip(raw / (raw + 1.0), 0.0, 1.0)

    @staticmethod
    def _weighted_unit_score_array(weighted_scores: Sequence[tuple[np.ndarray, float]]) -> np.ndarray:
        total_weight = 0.0
        total_score = None
        for score, weight in weighted_scores:
            weight = max(float(weight), 0.0)
            if weight <= 0.0:
                continue
            values = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
            total_score = values * weight if total_score is None else total_score + values * weight
            total_weight += weight
        if total_score is None or total_weight <= 1e-9:
            first = np.asarray(weighted_scores[0][0], dtype=float)
            return np.zeros_like(first)
        return np.clip(total_score / total_weight, 0.0, 1.0)

    def _trajectory_level_certificate_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        """Return non-Markov trajectory-level certificate terms.

        This is intentionally a single entry point so the whole certificate
        layer can be disabled without touching the Markov safety/dynamics costs.
        """

        if not bool(self.config.trajectory_certificate_enabled):
            return self._zero_trajectory_certificate_terms()

        terminal_terms = self._terminal_value_terms(trajectory, problem, blocked_ranges)
        occupation_terms = self._occupation_style_terms(trajectory, problem, blocked_ranges)
        slack_terms = self._certificate_slack_terms(trajectory, problem, blocked_ranges)

        certificate_cost = float(
            terminal_terms["terminal_value_score"]
            + occupation_terms["occupation_style_score"]
            + slack_terms["certificate_slack_score"]
        )
        certificate_score = float(
            np.clip(certificate_cost / max(float(self.config.trajectory_certificate_score_scale), 1e-6), 0.0, 1.0)
        )
        terms = {
            "trajectory_certificate_cost": certificate_cost,
            "trajectory_certificate_score": certificate_score,
        }
        terms.update(terminal_terms)
        terms.update(occupation_terms)
        terms.update(slack_terms)
        return terms

    @staticmethod
    def _zero_trajectory_certificate_terms() -> dict:
        return {
            "trajectory_certificate_cost": 0.0,
            "trajectory_certificate_score": 0.0,
            "terminal_value_cost": 0.0,
            "terminal_value_score": 0.0,
            "terminal_future_feasibility_score": 0.0,
            "terminal_recoverability_score": 0.0,
            "terminal_progress_score": 0.0,
            "terminal_speed_score": 0.0,
            "terminal_s": 0.0,
            "terminal_l": 0.0,
            "terminal_v": 0.0,
            "terminal_dl_ds": 0.0,
            "terminal_future_preview_distance": 0.0,
            "terminal_future_feasible_progress": 0.0,
            "occupation_style_score": 0.0,
            "certificate_slack_score": 0.0,
        }

    def _terminal_value_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        terminal = self._terminal_state_features(trajectory)
        s_t = float(terminal["s"])
        l_t = float(terminal["l"])
        v_t = max(float(terminal["v"]), 0.0)
        dl_ds_t = float(terminal["dl_ds"])
        preview_distance = float(
            np.clip(
                v_t * max(float(self.config.terminal_future_preview_time), 0.0),
                max(float(self.config.terminal_future_min_preview_distance), 0.0),
                max(float(self.config.terminal_future_max_preview_distance), 1e-6),
            )
        )

        future_score, feasible_progress = self._terminal_future_rollout_score(
            s_t=s_t,
            l_t=l_t,
            dl_ds_t=dl_ds_t,
            preview_distance=preview_distance,
            problem=problem,
            blocked_ranges=blocked_ranges,
        )
        recoverability_score = self._terminal_recoverability_score(terminal, problem)
        progress_shortfall = max(preview_distance - feasible_progress, 0.0)
        progress_score = self._unit_hinge_score(progress_shortfall, soft=max(preview_distance, 1.0))
        target_speed = self._target_speed(problem)
        speed_shortfall = max(float(target_speed) - v_t, 0.0)
        speed_score = self._unit_pseudo_huber_score(speed_shortfall, comfort=float(self.config.terminal_speed_comfort))

        terminal_value_score = self._weighted_unit_score(
            [
                (future_score, self.config.terminal_future_feasibility_weight),
                (recoverability_score, self.config.terminal_recoverability_weight),
                (progress_score, self.config.terminal_progress_weight),
                (speed_score, self.config.terminal_speed_weight),
            ]
        )
        return {
            "terminal_value_cost": float(terminal_value_score),
            "terminal_value_score": float(terminal_value_score),
            "terminal_future_feasibility_score": float(future_score),
            "terminal_recoverability_score": float(recoverability_score),
            "terminal_progress_score": float(progress_score),
            "terminal_speed_score": float(speed_score),
            "terminal_s": s_t,
            "terminal_l": l_t,
            "terminal_v": v_t,
            "terminal_dl_ds": dl_ds_t,
            "terminal_future_preview_distance": preview_distance,
            "terminal_future_feasible_progress": float(feasible_progress),
        }

    def _occupation_style_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        del trajectory, problem, blocked_ranges
        return {"occupation_style_score": 0.0}

    def _certificate_slack_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        del trajectory, problem, blocked_ranges
        return {"certificate_slack_score": 0.0}

    def _terminal_state_features(self, trajectory: Trajectory) -> dict:
        s_values = np.asarray(trajectory.s, dtype=float).reshape(-1)
        l_values = np.asarray(trajectory.l, dtype=float).reshape(-1)
        idx = int(s_values.size - 1)
        s_t = float(s_values[idx])
        l_t = float(l_values[idx])
        dl_ds_t = self._terminal_dl_ds(trajectory, idx)
        v_t = float(np.asarray(trajectory.s_v, dtype=float).reshape(-1)[idx])
        kappa_t = float(np.asarray(trajectory.kappa, dtype=float).reshape(-1)[idx])
        dkappa_t = float(self._dkappa_values(trajectory)[-1])
        lateral_accel_t = float(np.asarray(trajectory.l_a, dtype=float).reshape(-1)[idx])
        lateral_jerk_t = float(self._lateral_jerk_values(trajectory)[-1])
        return {
            "s": s_t,
            "l": l_t,
            "dl_ds": dl_ds_t,
            "v": v_t,
            "kappa": kappa_t,
            "dkappa": dkappa_t,
            "lateral_accel": lateral_accel_t,
            "lateral_jerk": lateral_jerk_t,
        }

    def _terminal_dl_ds(self, trajectory: Trajectory, terminal_idx: int) -> float:
        s_values = np.asarray(trajectory.s, dtype=float).reshape(-1)
        l_values = np.asarray(trajectory.l, dtype=float).reshape(-1)
        n = min(s_values.size, l_values.size, int(terminal_idx) + 1)
        if n < 2:
            raise ValueError("Terminal slope estimation requires at least two trajectory samples.")

        indices = np.arange(n, dtype=int)
        if trajectory.t is not None:
            t_values = np.asarray(trajectory.t, dtype=float).reshape(-1)
            if t_values.size >= n and np.isfinite(t_values[terminal_idx]):
                tail_time = max(float(self.config.terminal_tail_time), 0.0)
                indices = indices[t_values[:n] >= float(t_values[terminal_idx]) - tail_time]
        if indices.size < 2:
            raise ValueError("Terminal slope estimation requires at least two samples inside terminal_tail_time.")

        s_tail = s_values[indices]
        l_tail = l_values[indices]
        if float(np.max(s_tail) - np.min(s_tail)) <= 1e-6:
            raise ValueError("Terminal slope estimation requires longitudinal motion inside terminal_tail_time.")
        return float(np.polyfit(s_tail, l_tail, deg=1)[0])

    def _terminal_recoverability_score(self, terminal: dict, problem: PlanningProblem) -> float:
        center_right, center_left = self._center_lateral_interval(problem)
        boundary_clearance = min(float(terminal["l"]) - center_right, center_left - float(terminal["l"]))
        boundary_score = self._unit_hinge_score(
            float(self.config.terminal_boundary_clearance_comfort) - boundary_clearance,
            soft=max(float(self.config.terminal_boundary_clearance_comfort), 1e-6),
        )
        return self._weighted_unit_score(
            [
                (
                    self._unit_pseudo_huber_score(abs(float(terminal["dl_ds"])), self.config.terminal_dl_ds_comfort),
                    1.0,
                ),
                (
                    self._unit_pseudo_huber_score(abs(float(terminal["kappa"])), self.config.terminal_kappa_comfort),
                    1.0,
                ),
                (
                    self._unit_pseudo_huber_score(abs(float(terminal["dkappa"])), self.config.terminal_dkappa_comfort),
                    1.0,
                ),
                (
                    self._unit_pseudo_huber_score(
                        abs(float(terminal["lateral_accel"])), self.config.terminal_lateral_accel_comfort
                    ),
                    1.0,
                ),
                (
                    self._unit_pseudo_huber_score(
                        abs(float(terminal["lateral_jerk"])), self.config.terminal_lateral_jerk_comfort
                    ),
                    1.0,
                ),
                (boundary_score, 1.0),
            ]
        )

    def _terminal_future_rollout_score(
        self,
        s_t: float,
        l_t: float,
        dl_ds_t: float,
        preview_distance: float,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> tuple[float, float]:
        preview_distance = max(float(preview_distance), 0.0)
        if preview_distance <= 1e-6:
            return 0.0, 0.0

        step_s = max(float(self.config.terminal_future_step_s), 0.25)
        distances = np.arange(step_s, preview_distance + 0.5 * step_s, step_s, dtype=float)
        if distances.size == 0:
            distances = np.array([preview_distance], dtype=float)

        slope_delta = max(float(self.config.terminal_future_slope_delta), 0.0)
        slope_targets = [float(dl_ds_t), 0.0, float(dl_ds_t) + slope_delta, float(dl_ds_t) - slope_delta]
        slope_targets = list(dict.fromkeys(round(value, 6) for value in slope_targets))

        best_score = 1.0
        best_feasible_progress = 0.0
        for target_slope in slope_targets:
            sample_scores = []
            feasible_progress = preview_distance
            hard_blocked = False
            for distance in distances:
                d = min(float(distance), preview_distance)
                s_hat = float(s_t) + d
                l_hat = float(l_t) + float(dl_ds_t) * d + 0.5 * (float(target_slope) - float(dl_ds_t)) * (
                    d * d / max(preview_distance, 1e-6)
                )
                sample_score, hard_violation = self._terminal_future_sample_score(
                    s_hat=s_hat,
                    l_hat=l_hat,
                    problem=problem,
                    blocked_ranges=blocked_ranges,
                )
                sample_scores.append(sample_score)
                if hard_violation and not hard_blocked:
                    feasible_progress = max(d - step_s, 0.0)
                    hard_blocked = True
            primitive_score = self._topk_max_cost(np.asarray(sample_scores, dtype=float), fraction=0.2)
            best_score = min(best_score, primitive_score)
            best_feasible_progress = max(best_feasible_progress, feasible_progress)

        return float(np.clip(best_score, 0.0, 1.0)), float(np.clip(best_feasible_progress, 0.0, preview_distance))

    def _terminal_future_sample_score(
        self,
        s_hat: float,
        l_hat: float,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> tuple[float, bool]:
        center_right, center_left = self._center_lateral_interval(problem)
        if center_left <= center_right:
            return 1.0, True

        if float(l_hat) < center_right or float(l_hat) > center_left:
            return 1.0, True

        hard_collision = self._center_overlaps_blocked(float(s_hat), float(l_hat), blocked_ranges)
        if hard_collision:
            return 1.0, True

        intervals = self._free_lateral_intervals_at_s(float(s_hat), problem, blocked_ranges)
        containing_width = self._containing_interval_width(intervals, float(l_hat))
        if containing_width <= 0.0:
            return 1.0, True

        road_width = max(center_left - center_right, 1e-6)
        desired_width = min(float(self.config.terminal_future_min_free_width), max(0.3 * road_width, 0.5))
        space_score = self._unit_hinge_score(desired_width - containing_width, soft=max(desired_width, 1e-6))

        edge_clearance = min(float(l_hat) - center_right, center_left - float(l_hat))
        edge_score = 0.2 * self._unit_hinge_score(
            float(self.config.terminal_boundary_clearance_comfort) - edge_clearance,
            soft=max(float(self.config.terminal_boundary_clearance_comfort), 1e-6),
        )
        return float(np.clip(max(space_score, edge_score), 0.0, 1.0)), False

    def _center_lateral_interval(self, problem: PlanningProblem) -> tuple[float, float]:
        half_width = 0.5 * float(self.config.ego_width)
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        return right + half_width, left - half_width

    def _free_lateral_intervals_at_s(
        self,
        s_value: float,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> list[tuple[float, float]]:
        center_right, center_left = self._center_lateral_interval(problem)
        if center_left <= center_right:
            return []

        intervals = [(center_right, center_left)]
        ego_s_min = float(s_value) - float(self.config.ego_rear)
        ego_s_max = float(s_value) + float(self.config.ego_front)
        half_width = 0.5 * float(self.config.ego_width)
        for blocked in blocked_ranges:
            if not ranges_overlap(ego_s_min, ego_s_max, float(blocked["s_min"]), float(blocked["s_max"])):
                continue
            blocked_l_min = float(blocked["l_min"]) - half_width
            blocked_l_max = float(blocked["l_max"]) + half_width
            next_intervals = []
            for interval_min, interval_max in intervals:
                if not ranges_overlap(interval_min, interval_max, blocked_l_min, blocked_l_max):
                    next_intervals.append((interval_min, interval_max))
                    continue
                if blocked_l_min > interval_min:
                    next_intervals.append((interval_min, min(blocked_l_min, interval_max)))
                if blocked_l_max < interval_max:
                    next_intervals.append((max(blocked_l_max, interval_min), interval_max))
            intervals = [(a, b) for a, b in next_intervals if b - a > 1e-6]
            if not intervals:
                break
        return intervals

    def _center_overlaps_blocked(self, s_value: float, l_value: float, blocked_ranges: Sequence[dict]) -> bool:
        ego_s_min, ego_s_max, ego_l_min, ego_l_max = self._ego_frenet_range(float(s_value), float(l_value))
        for blocked in blocked_ranges:
            if ranges_overlap(ego_s_min, ego_s_max, blocked["s_min"], blocked["s_max"]) and ranges_overlap(
                ego_l_min, ego_l_max, blocked["l_min"], blocked["l_max"]
            ):
                return True
        return False

    @staticmethod
    def _containing_interval_width(intervals: Sequence[tuple[float, float]], l_value: float) -> float:
        for interval_min, interval_max in intervals:
            if float(interval_min) <= float(l_value) <= float(interval_max):
                return float(interval_max) - float(interval_min)
        return 0.0

    @staticmethod
    def _unit_hinge_score(value: float, soft: float) -> float:
        return float(np.clip(shaped_hinge(max(float(value), 0.0), safe=0.0, soft=max(float(soft), 1e-6), cap=1.0), 0.0, 1.0))

    @staticmethod
    def _unit_pseudo_huber_score(value: float, comfort: float) -> float:
        raw = float(pseudo_huber(float(value) / max(float(comfort), 1e-6), delta=1.0))
        return float(np.clip(raw / (raw + 1.0), 0.0, 1.0))

    @staticmethod
    def _weighted_unit_score(weighted_scores: Sequence[tuple[float, float]]) -> float:
        total_weight = 0.0
        total_score = 0.0
        for score, weight in weighted_scores:
            weight = max(float(weight), 0.0)
            if weight <= 0.0:
                continue
            total_weight += weight
            total_score += weight * float(np.clip(float(score), 0.0, 1.0))
        if total_weight <= 1e-9:
            return 0.0
        return float(np.clip(total_score / total_weight, 0.0, 1.0))

    def _global_hierarchy_terms(self, running_terms: dict) -> dict:
        collision_running = self._finite_array(running_terms.get("collision_running"))
        collision_overlap = self._finite_array(running_terms.get("collision_overlap"))
        road_running = self._finite_array(running_terms.get("road_running"))
        road_violation = self._finite_array(running_terms.get("road_violation"))
        lateral_limit = self._finite_array(running_terms.get("lateral_accel_limit_running"))
        lateral_zero = self._finite_array(running_terms.get("lateral_accel_zero_running"))
        lateral_violation = self._finite_array(running_terms.get("lateral_accel_violation"))
        kappa_limit = self._finite_array(running_terms.get("kappa_limit_running"))
        kappa_zero = self._finite_array(running_terms.get("kappa_zero_running"))
        kappa_violation = self._finite_array(running_terms.get("kappa_violation"))
        dkappa_limit = self._finite_array(running_terms.get("dkappa_limit_running"))
        dkappa_zero = self._finite_array(running_terms.get("dkappa_zero_running"))
        dkappa_violation = self._finite_array(running_terms.get("dkappa_violation"))
        lateral_jerk_limit = self._finite_array(running_terms.get("lateral_jerk_limit_running"))
        lateral_jerk_zero = self._finite_array(running_terms.get("lateral_jerk_zero_running"))
        lateral_jerk_violation = self._finite_array(running_terms.get("lateral_jerk_violation"))
        speed_limit = self._finite_array(running_terms.get("speed_limit_running"))
        speed_tracking = self._finite_array(running_terms.get("speed_max_deviation_running"))
        if speed_tracking.size == 0:
            speed_tracking = self._finite_array(running_terms.get("speed_tracking_running"))
        speed_efficiency = self._finite_array(running_terms.get("efficiency_running"))
        speed_violation = self._finite_array(running_terms.get("speed_violation"))
        reference_lateral = self._finite_array(running_terms.get("reference_lateral_running"))

        collision_cost = self._topk_max_cost(collision_running, fraction=0.15)
        road_cost = self._topk_max_cost(road_running, fraction=0.15)
        lateral_accel_limit_cost = float(np.mean(lateral_limit)) if lateral_limit.size else 4.0
        kappa_limit_cost = float(np.mean(kappa_limit)) if kappa_limit.size else 4.0
        dkappa_limit_cost = float(np.mean(dkappa_limit)) if dkappa_limit.size else 4.0
        lateral_jerk_limit_cost = float(np.mean(lateral_jerk_limit)) if lateral_jerk_limit.size else 4.0
        speed_limit_cost = float(np.mean(speed_limit)) if speed_limit.size else 4.0
        lateral_accel_zero_cost = float(np.mean(lateral_zero)) if lateral_zero.size else 4.0
        kappa_zero_cost = float(np.mean(kappa_zero)) if kappa_zero.size else 4.0
        dkappa_zero_cost = float(np.mean(dkappa_zero)) if dkappa_zero.size else 4.0
        lateral_jerk_zero_cost = float(np.mean(lateral_jerk_zero)) if lateral_jerk_zero.size else 4.0
        speed_deviation_cost = float(np.mean(speed_tracking)) if speed_tracking.size else 4.0
        efficiency_cost = float(np.mean(speed_efficiency)) if speed_efficiency.size else 4.0
        reference_lateral_cost = float(np.mean(reference_lateral)) if reference_lateral.size else 0.0
        lateral_accel_hard_score = saturate_cost(lateral_accel_limit_cost, self.config.dynamic_score_scale)
        kappa_hard_score = saturate_cost(kappa_limit_cost, self.config.dynamic_score_scale)
        dkappa_hard_score = saturate_cost(dkappa_limit_cost, self.config.dynamic_score_scale)
        lateral_jerk_hard_score = saturate_cost(lateral_jerk_limit_cost, self.config.dynamic_score_scale)
        lateral_accel_soft_score = saturate_cost(lateral_accel_zero_cost, self.config.comfort_score_scale)
        kappa_soft_score = saturate_cost(kappa_zero_cost, self.config.comfort_score_scale)
        dkappa_soft_score = saturate_cost(dkappa_zero_cost, self.config.comfort_score_scale)
        lateral_jerk_soft_score = saturate_cost(lateral_jerk_zero_cost, self.config.comfort_score_scale)
        speed_score = saturate_cost(speed_limit_cost + speed_deviation_cost, self.config.speed_score_scale)
        efficiency_score = saturate_cost(efficiency_cost + speed_deviation_cost, self.config.efficiency_score_scale)
        comfort_score = self._mean_score(
            [lateral_accel_soft_score, kappa_soft_score, dkappa_soft_score, lateral_jerk_soft_score]
        )
        reference_score = saturate_cost(reference_lateral_cost, self.config.reference_score_scale)
        lateral_accel_score = self._mean_score([lateral_accel_hard_score, lateral_accel_soft_score])
        kappa_score = self._mean_score([kappa_hard_score, kappa_soft_score])
        dkappa_score = self._mean_score([dkappa_hard_score, dkappa_soft_score])
        lateral_jerk_score = self._mean_score([lateral_jerk_hard_score, lateral_jerk_soft_score])
        dynamic_score = float(
            kappa_hard_score + dkappa_hard_score + lateral_accel_hard_score + lateral_jerk_hard_score
        )

        running_cost = float(
            1.0e7 * kappa_hard_score
            + 1.0e6 * dkappa_hard_score
            + 1.0e5 * lateral_accel_hard_score
            + 1.0e4 * lateral_jerk_hard_score
            + 1.0e3 * efficiency_score
            + 1.0e2 * reference_score
            + 1.0e1 * comfort_score
        )
        return {
            "collision_flag": float(np.max(collision_overlap) > 0.0) if collision_overlap.size else 0.0,
            "collision_cost": collision_cost,
            "road_flag": float(np.max(road_violation) > 0.0) if road_violation.size else 0.0,
            "road_cost": road_cost,
            "lateral_accel_flag": float(np.max(lateral_violation) > 0.0) if lateral_violation.size else 0.0,
            "lateral_accel_limit_cost": lateral_accel_limit_cost,
            "lateral_accel_zero_cost": lateral_accel_zero_cost,
            "lateral_accel_hard_score": lateral_accel_hard_score,
            "lateral_accel_soft_score": lateral_accel_soft_score,
            "lateral_accel_score": lateral_accel_score,
            "kappa_flag": float(np.max(kappa_violation) > 0.0) if kappa_violation.size else 0.0,
            "kappa_limit_cost": kappa_limit_cost,
            "kappa_zero_cost": kappa_zero_cost,
            "kappa_hard_score": kappa_hard_score,
            "kappa_soft_score": kappa_soft_score,
            "kappa_score": kappa_score,
            "dkappa_flag": float(np.max(dkappa_violation) > 0.0) if dkappa_violation.size else 0.0,
            "dkappa_limit_cost": dkappa_limit_cost,
            "dkappa_zero_cost": dkappa_zero_cost,
            "dkappa_hard_score": dkappa_hard_score,
            "dkappa_soft_score": dkappa_soft_score,
            "dkappa_score": dkappa_score,
            "lateral_jerk_flag": float(np.max(lateral_jerk_violation) > 0.0) if lateral_jerk_violation.size else 0.0,
            "lateral_jerk_limit_cost": lateral_jerk_limit_cost,
            "lateral_jerk_zero_cost": lateral_jerk_zero_cost,
            "lateral_jerk_hard_score": lateral_jerk_hard_score,
            "lateral_jerk_soft_score": lateral_jerk_soft_score,
            "lateral_jerk_score": lateral_jerk_score,
            "speed_flag": float(np.max(speed_violation) > 0.0) if speed_violation.size else 0.0,
            "speed_limit_cost": speed_limit_cost,
            "speed_deviation_cost": speed_deviation_cost,
            "speed_score": speed_score,
            "dynamic_score": dynamic_score,
            "efficiency_cost": efficiency_cost,
            "efficiency_score": efficiency_score,
            "comfort_score": comfort_score,
            "reference_lateral_target": float(running_terms.get("reference_lateral_target", 0.0)),
            "reference_lateral_cost": reference_lateral_cost,
            "reference_score": reference_score,
            "running_cost": running_cost,
        }

    def _collision_running_terms(self, trajectory: Trajectory, blocked_ranges: Sequence[dict]) -> dict:
        s_values = np.asarray(trajectory.s, dtype=float)
        l_values = np.asarray(trajectory.l, dtype=float)
        n = min(s_values.size, l_values.size)
        running = np.zeros((n,), dtype=float)
        overlap = np.zeros((n,), dtype=float)
        if n == 0 or not blocked_ranges:
            return {"collision_running": running, "collision_overlap": overlap}

        t_values = np.asarray(trajectory.t, dtype=float)
        if t_values.size < n:
            t_values = np.linspace(0.0, float(n - 1), num=n, dtype=float)
        start_s = float(s_values[0])
        for idx, (s, l) in enumerate(zip(s_values[:n], l_values[:n])):
            ego_s_min, ego_s_max, ego_l_min, ego_l_max = self._ego_frenet_range(float(s), float(l))
            sample_t = float(t_values[idx])
            decay = self._range_decay(float(s) - start_s)
            sample_cost = 0.0
            for blocked in blocked_ranges:
                blocked_s_min = self._blocked_scalar_at_time(blocked, "s_min", sample_t)
                blocked_s_max = self._blocked_scalar_at_time(blocked, "s_max", sample_t)
                blocked_l_min = self._blocked_scalar_at_time(blocked, "l_min", sample_t)
                blocked_l_max = self._blocked_scalar_at_time(blocked, "l_max", sample_t)
                if ranges_overlap(ego_s_min, ego_s_max, blocked_s_min, blocked_s_max) and ranges_overlap(
                    ego_l_min, ego_l_max, blocked_l_min, blocked_l_max
                ):
                    overlap[idx] = 1.0
                    s_overlap = min(ego_s_max, blocked_s_max) - max(ego_s_min, blocked_s_min)
                    l_overlap = min(ego_l_max, blocked_l_max) - max(ego_l_min, blocked_l_min)
                    penetration = max(min(float(s_overlap), float(l_overlap)), 0.0)
                    sample_cost = max(
                        sample_cost,
                        decay * (1.0 + float(shaped_hinge(penetration, safe=0.0, soft=0.6, tail_gain=0.35, cap=3.0))),
                    )
            running[idx] = sample_cost
        return {"collision_running": running, "collision_overlap": overlap}

    def _road_running_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> dict:
        l_values = np.asarray(trajectory.l, dtype=float)
        if l_values.size == 0:
            return {"road_running": np.array([4.0]), "road_violation": np.array([1.0])}

        half_width = 0.5 * float(self.config.ego_width)
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        ego_l_min = l_values - half_width
        ego_l_max = l_values + half_width
        left_excess = np.maximum(ego_l_max - left, 0.0)
        right_excess = np.maximum(right - ego_l_min, 0.0)
        excess = np.maximum(left_excess, right_excess)

        edge_clearance = np.minimum(left - ego_l_max, ego_l_min - right)
        edge_pressure = np.maximum(float(self.config.road_edge_buffer) - edge_clearance, 0.0)
        violation_cost = shaped_hinge(excess, safe=0.0, soft=0.6, tail_gain=0.25, cap=3.0)
        edge_cost = 0.35 * shaped_hinge(
            edge_pressure,
            safe=0.0,
            soft=max(float(self.config.road_edge_buffer), 1e-3),
            tail_gain=0.1,
            cap=1.5,
        )
        return {
            "road_running": np.asarray(violation_cost + edge_cost, dtype=float),
            "road_violation": np.asarray(excess > 1e-6, dtype=float),
        }

    def _lateral_accel_running_terms(self, trajectory: Trajectory) -> dict:
        lat_accel = self._lateral_acceleration_values(trajectory)
        return self._bounded_zero_running_terms(
            "lateral_accel",
            lat_accel,
            min_value=float(self.config.min_lateral_accel),
            max_value=float(self.config.max_lateral_accel),
            zero_comfort=float(self.config.lateral_accel_zero_comfort),
            zero_weight=float(self.config.lateral_accel_zero_weight),
        )

    def _shape_running_terms(self, trajectory: Trajectory) -> dict:
        terms = {}
        terms.update(
            self._bounded_zero_running_terms(
                "kappa",
                self._kappa_values(trajectory),
                min_value=float(self.config.min_kappa),
                max_value=float(self.config.max_kappa),
                zero_comfort=float(self.config.kappa_zero_comfort),
                zero_weight=float(self.config.kappa_zero_weight),
            )
        )
        terms.update(
            self._bounded_zero_running_terms(
                "dkappa",
                self._dkappa_values(trajectory),
                min_value=float(self.config.min_dkappa),
                max_value=float(self.config.max_dkappa),
                zero_comfort=float(self.config.dkappa_zero_comfort),
                zero_weight=float(self.config.dkappa_zero_weight),
            )
        )
        terms.update(
            self._bounded_zero_running_terms(
                "lateral_jerk",
                self._lateral_jerk_values(trajectory),
                min_value=float(self.config.min_lateral_jerk),
                max_value=float(self.config.max_lateral_jerk),
                zero_comfort=float(self.config.lateral_jerk_zero_comfort),
                zero_weight=float(self.config.lateral_jerk_zero_weight),
            )
        )
        return terms

    def _bounded_zero_running_terms(
        self,
        prefix: str,
        values,
        min_value: float,
        max_value: float,
        zero_comfort: float,
        zero_weight: float,
    ) -> dict:
        values = self._finite_array(values)
        if values.size == 0:
            return {
                f"{prefix}_limit_running": np.array([4.0]),
                f"{prefix}_zero_running": np.array([4.0]),
                f"{prefix}_violation": np.array([1.0]),
            }

        low = min(float(min_value), float(max_value))
        high = max(float(min_value), float(max_value))
        lower_excess = np.maximum(low - values, 0.0)
        upper_excess = np.maximum(values - high, 0.0)
        excess = np.maximum(lower_excess, upper_excess)
        limit_scale = max(0.25 * max(abs(low), abs(high)), 1e-6)
        limit_running = shaped_hinge(excess, safe=0.0, soft=limit_scale, tail_gain=0.35, cap=3.0)
        zero_running = float(zero_weight) * pseudo_huber(values / max(float(zero_comfort), 1e-6), delta=1.0)
        return {
            f"{prefix}_limit_running": np.asarray(limit_running, dtype=float),
            f"{prefix}_zero_running": np.asarray(zero_running, dtype=float),
            f"{prefix}_violation": np.asarray(excess > 1e-6, dtype=float),
        }

    def _speed_running_terms(self, trajectory: Trajectory, problem: PlanningProblem) -> dict:
        s_v = np.asarray(trajectory.s_v, dtype=float)

        max_speed = max(float(self.config.max_longitudinal_speed), 1e-3)
        comfort = max(float(self.config.speed_tracking_comfort), 1e-3)
        speed_tracking = pseudo_huber((s_v - max_speed) / comfort, delta=1.0)
        target_speed = self._target_speed(problem)
        low_speed_shortfall = np.maximum(float(target_speed) - s_v, 0.0)
        efficiency_running = pseudo_huber(low_speed_shortfall / comfort, delta=1.0)

        reverse_excess = np.maximum(-s_v, 0.0)
        speed_excess = np.maximum(s_v - max_speed, 0.0)
        speed_pressure = shaped_hinge(speed_excess, safe=0.0, soft=1.0, tail_gain=0.35, cap=3.0)
        reverse_pressure = shaped_hinge(reverse_excess, safe=0.0, soft=0.5, tail_gain=0.35, cap=3.0)
        return {
            "speed_limit_running": np.asarray(speed_pressure + reverse_pressure, dtype=float),
            "speed_tracking_running": np.asarray(speed_tracking, dtype=float),
            "speed_max_deviation_running": np.asarray(speed_tracking, dtype=float),
            "efficiency_running": np.asarray(efficiency_running, dtype=float),
            "speed_violation": np.asarray((reverse_excess > 1e-6) | (speed_excess > 1e-6), dtype=float),
        }

    def _reference_lateral_running_terms(
        self,
        trajectory: Trajectory,
        problem: PlanningProblem,
        blocked_ranges: Sequence[dict],
    ) -> dict:
        l_values = np.asarray(trajectory.l, dtype=float)
        target = self._reference_lateral_target(problem)
        if l_values.size == 0:
            return {"reference_lateral_target": float(target), "reference_lateral_running": np.empty((0,), dtype=float)}

        comfort = max(float(self.config.reference_lateral_comfort), 1e-6)
        deviation_cost = pseudo_huber((l_values - float(target)) / comfort, delta=1.0)
        n = l_values.size
        time_weight = np.linspace(
            float(self.config.reference_lateral_time_weight_start),
            float(self.config.reference_lateral_time_weight_end),
            num=n,
            dtype=float,
        )
        time_weight = np.maximum(time_weight, 0.0)

        relief = np.ones((n,), dtype=float)
        s_values = np.asarray(trajectory.s, dtype=float)
        if s_values.size:
            m = min(n, s_values.size)
            buffer = max(float(self.config.reference_obstacle_relief_buffer), 0.0)
            relief_weight = float(np.clip(float(self.config.reference_obstacle_relief_weight), 0.0, 1.0))
            for blocked in blocked_ranges:
                near = (s_values[:m] >= float(blocked["s_min"]) - buffer) & (
                    s_values[:m] <= float(blocked["s_max"]) + buffer
                )
                near_indices = np.where(near)[0]
                if near_indices.size:
                    relief[near_indices] = np.minimum(relief[near_indices], relief_weight)

        return {
            "reference_lateral_target": float(target),
            "reference_lateral_running": np.asarray(time_weight * relief * deviation_cost, dtype=float),
        }

    @staticmethod
    def _finite_array(values) -> np.ndarray:
        if values is None:
            return np.empty((0,), dtype=float)
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _topk_max_cost(values: np.ndarray, fraction: float) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0.0
        return float(0.7 * topk_mean(values, fraction) + 0.3 * np.max(values))

    @staticmethod
    def _mean_score(scores: Sequence[float]) -> float:
        values = np.asarray(scores, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        return float(np.clip(np.mean(values), 0.0, 1.0))

    @staticmethod
    def _reference_lateral_target(problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("preferred_l", "reference_l", "target_lane_l", "target_l", "lane_change_target_l"):
            if key in metadata:
                try:
                    value = float(metadata[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Planning metadata {key!r} must be a finite numeric lateral target.") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Planning metadata {key!r} must be a finite numeric lateral target.")
                return value
        return 0.0

    def _target_speed(self, problem: PlanningProblem) -> float:
        metadata = dict(problem.metadata or {})
        for key in ("target_speed", "desired_speed", "speed_limit"):
            if key in metadata:
                try:
                    value = float(metadata[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Planning metadata {key!r} must be a finite numeric speed.") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Planning metadata {key!r} must be a finite numeric speed.")
                return min(value, float(self.config.max_longitudinal_speed))
        return float(self.config.max_longitudinal_speed)

    def _blocked_ranges(self, problem: PlanningProblem) -> list[dict]:
        ranges = []
        for actor in problem.actors:
            blocked = self._blocked_range_from_metadata(actor)
            if blocked is None:
                raise ValueError(
                    f"Actor {actor.actor_id!r} must explicitly provide temporal_blocked_range, "
                    "blocked_s/l bounds, or an s/l pose in metadata."
                )
            ranges.append(self._inflate_blocked_range(blocked))
        return ranges

    def _inflate_blocked_range(self, blocked: dict) -> dict:
        s_buffer = max(float(self.config.planning_obstacle_s_buffer), 0.0)
        l_buffer = max(float(self.config.planning_obstacle_l_buffer), 0.0)
        inflated = dict(blocked)
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
        inflated["planning_s_buffer"] = s_buffer
        inflated["planning_l_buffer"] = l_buffer
        return inflated

    def _blocked_range_from_metadata(self, actor: ActorPrediction) -> Optional[dict]:
        metadata = dict(actor.metadata or {})
        temporal = self._temporal_blocked_range_from_metadata(metadata, actor_id=actor.actor_id)
        if temporal is not None:
            return {
                "s_min": float(temporal["s_min"][0]),
                "s_max": float(temporal["s_max"][0]),
                "l_min": float(temporal["l_min"][0]),
                "l_max": float(temporal["l_max"][0]),
                "actor_id": actor.actor_id,
                "temporal": temporal,
            }
        keys = ("blocked_s_min", "blocked_s_max", "blocked_l_min", "blocked_l_max")
        if any(key in metadata for key in keys) and not all(key in metadata for key in keys):
            missing = [key for key in keys if key not in metadata]
            raise ValueError(f"Actor {actor.actor_id!r} static blocked range is missing fields: {missing}.")
        if all(key in metadata for key in keys):
            blocked = {
                "s_min": float(metadata["blocked_s_min"]),
                "s_max": float(metadata["blocked_s_max"]),
                "l_min": float(metadata["blocked_l_min"]),
                "l_max": float(metadata["blocked_l_max"]),
                "actor_id": actor.actor_id,
            }
            self._validate_blocked_bounds(blocked, actor.actor_id)
            return blocked
        if ("s" in metadata) != ("l" in metadata):
            raise ValueError(f"Actor {actor.actor_id!r} metadata must provide both s and l.")
        if "s" in metadata and "l" in metadata:
            half_length = 0.5 * float(actor.length)
            half_width = 0.5 * float(actor.width)
            blocked = {
                "s_min": float(metadata["s"]) - half_length,
                "s_max": float(metadata["s"]) + half_length,
                "l_min": float(metadata["l"]) - half_width,
                "l_max": float(metadata["l"]) + half_width,
                "actor_id": actor.actor_id,
            }
            self._validate_blocked_bounds(blocked, actor.actor_id)
            return blocked
        return None

    @staticmethod
    def _validate_blocked_bounds(blocked: dict, actor_id: str) -> None:
        values = np.asarray(
            [blocked["s_min"], blocked["s_max"], blocked["l_min"], blocked["l_max"]],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Actor {actor_id!r} blocked range must contain only finite bounds.")
        if float(blocked["s_min"]) > float(blocked["s_max"]) or float(blocked["l_min"]) > float(blocked["l_max"]):
            raise ValueError(f"Actor {actor_id!r} blocked range has inverted min/max bounds.")

    @classmethod
    def _temporal_blocked_range_from_metadata(cls, metadata: dict, actor_id: str = "<unknown>") -> Optional[dict]:
        raw = metadata.get("temporal_blocked_range")
        if raw is None:
            raw = metadata.get("temporal_blocked_ranges")
        if raw is None:
            return None

        if isinstance(raw, dict):
            source = raw
        else:
            source = {}
            try:
                rows = list(raw)
            except TypeError as exc:
                raise ValueError(f"Actor {actor_id!r} temporal blocked range must be a mapping or iterable.") from exc
            if not rows:
                raise ValueError(f"Actor {actor_id!r} temporal blocked range must not be empty.")
            for key in ("t", "s_min", "s_max", "l_min", "l_max"):
                values = []
                for row in rows:
                    if isinstance(row, dict):
                        values.append(row.get(key))
                    else:
                        values.append(getattr(row, key, None))
                source[key] = values

        required = ("t", "s_min", "s_max", "l_min", "l_max")
        if not all(key in source for key in required):
            missing = [key for key in required if key not in source]
            raise ValueError(f"Actor {actor_id!r} temporal blocked range is missing fields: {missing}.")

        arrays = {key: np.asarray(source[key], dtype=float).reshape(-1) for key in required}
        lengths = {key: arr.size for key, arr in arrays.items()}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) <= 0:
            raise ValueError(f"Actor {actor_id!r} temporal blocked range fields need equal non-zero lengths: {lengths}.")
        if not all(np.all(np.isfinite(arr)) for arr in arrays.values()):
            raise ValueError(f"Actor {actor_id!r} temporal blocked range must contain only finite values.")
        if np.any(np.diff(arrays["t"]) < 0.0):
            raise ValueError(f"Actor {actor_id!r} temporal blocked range timestamps must be non-decreasing.")
        if np.any(arrays["s_min"] > arrays["s_max"]) or np.any(arrays["l_min"] > arrays["l_max"]):
            raise ValueError(f"Actor {actor_id!r} temporal blocked range has inverted min/max bounds.")
        return arrays

    def _ego_frenet_range(self, s: float, l: float) -> tuple[float, float, float, float]:
        return (
            float(s) - float(self.config.ego_rear),
            float(s) + float(self.config.ego_front),
            float(l) - 0.5 * float(self.config.ego_width),
            float(l) + 0.5 * float(self.config.ego_width),
        )

    def _range_decay(self, s_rel: float) -> float:
        return float(0.2 + 0.8 * math.exp(-max(float(s_rel), 0.0) / max(float(self.config.collision_decay_s), 1e-3)))

    @staticmethod
    def _kappa_values(trajectory: Trajectory) -> np.ndarray:
        kappa = np.asarray(trajectory.kappa, dtype=float).reshape(-1)
        return kappa[np.isfinite(kappa)]

    def _dkappa_values(self, trajectory: Trajectory) -> np.ndarray:
        kappa = np.asarray(trajectory.kappa, dtype=float).reshape(-1)
        coordinate = self._path_coordinate_values(trajectory, kappa.size)
        return self._gradient_values(kappa, coordinate)

    def _lateral_jerk_values(self, trajectory: Trajectory) -> np.ndarray:
        lateral_accel = np.asarray(trajectory.l_a, dtype=float).reshape(-1)
        t_values = np.asarray(trajectory.t, dtype=float).reshape(-1)
        return self._gradient_values(lateral_accel, t_values)

    @staticmethod
    def _path_coordinate_values(trajectory: Trajectory, size: int) -> np.ndarray:
        size = max(int(size), 0)
        s_values = np.asarray(trajectory.s, dtype=float).reshape(-1)
        if size < 2 or s_values.size != size:
            raise ValueError(f"dkappa calculation requires aligned s/kappa samples, got s={s_values.size}, kappa={size}.")
        return s_values

    @staticmethod
    def _gradient_values(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
        n = min(values.size, coordinate.size)
        if n < 2:
            return np.empty((0,), dtype=float)
        values = values[:n]
        coordinate = coordinate[:n]
        if np.any(np.diff(coordinate) <= 1e-6):
            raise ValueError("Trajectory derivative coordinate must be strictly increasing.")
        edge_order = 2 if n >= 3 else 1
        gradient = np.gradient(values, coordinate, edge_order=edge_order)
        return np.asarray(gradient, dtype=float)[np.isfinite(gradient)]

    @staticmethod
    def _lateral_acceleration_values(trajectory: Trajectory) -> np.ndarray:
        return np.asarray(trajectory.l_a, dtype=float).reshape(-1)
