from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, Trajectory


def execution_index(trajectory: Trajectory, planning_dt: float) -> int:
    t_values = np.asarray(trajectory.t, dtype=float)
    if t_values.size < 2:
        return 0
    idx = int(np.searchsorted(t_values, float(planning_dt), side="left"))
    return min(max(idx, 1), t_values.size - 1)


def ego_from_trajectory(trajectory: Trajectory, index: int, absolute_t: float) -> EgoState:
    s = float(np.asarray(trajectory.s, dtype=float)[index])
    l = float(np.asarray(trajectory.l, dtype=float)[index])
    s_v = array_value(trajectory.s_v, index, 0.0)
    l_v = array_value(trajectory.l_v, index, 0.0)
    s_a = array_value(trajectory.s_a, index, 0.0)
    l_a = array_value(trajectory.l_a, index, 0.0)
    yaw = None if trajectory.yaw is None else float(np.asarray(trajectory.yaw, dtype=float)[index])
    return EgoState(s=s, l=l, s_v=s_v, l_v=l_v, s_a=s_a, l_a=l_a, yaw=yaw, t=float(absolute_t))


def ego_speed(ego: EgoState) -> float:
    return float(math.hypot(float(ego.s_v), float(ego.l_v)))


def trajectory_speed_values(trajectory: Trajectory) -> np.ndarray:
    if trajectory.v is not None:
        values = np.asarray(trajectory.v, dtype=float)
        if values.size:
            return values
    if trajectory.s_v is not None and trajectory.l_v is not None:
        s_v = np.asarray(trajectory.s_v, dtype=float)
        l_v = np.asarray(trajectory.l_v, dtype=float)
        n = min(s_v.size, l_v.size)
        if n:
            return np.hypot(s_v[:n], l_v[:n])
    if trajectory.s_v is not None:
        values = np.asarray(trajectory.s_v, dtype=float)
        if values.size:
            return np.abs(values)
    return np.empty((0,), dtype=float)


def trajectory_speed_at(trajectory: Trajectory, index: int, default: float = 0.0) -> float:
    values = trajectory_speed_values(trajectory)
    if values.size == 0:
        return float(default)
    return float(values[min(max(int(index), 0), values.size - 1)])


def trajectory_delta_s(trajectory: Trajectory) -> float:
    s_values = np.asarray(trajectory.s, dtype=float)
    if s_values.size < 2:
        return 0.0
    return float(s_values[-1] - s_values[0])


def array_value(values, index: int, default: float) -> float:
    if values is None:
        return float(default)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(arr[min(max(int(index), 0), arr.size - 1)])


def find_actor(problem: PlanningProblem, actor_id: str | None) -> Optional[ActorPrediction]:
    if not actor_id:
        return None
    for actor in problem.actors:
        if str(actor.actor_id) == str(actor_id):
            return actor
    return None


def actor_keep_baseline_xy(
    scenario,
    actor: ActorPrediction,
    horizon: float,
    dt: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    metadata = dict(actor.metadata or {})
    if "s" not in metadata or "l" not in metadata:
        return None
    times = np.arange(0.0, float(horizon) + 0.5 * float(dt), float(dt), dtype=float)
    if times.size == 0:
        return None
    s0 = float(metadata.get("s", 0.0))
    l0 = float(metadata.get("l", 0.0))
    s_v = float(metadata.get("s_v", 0.0))
    s_a = float(metadata.get("s_a", 0.0))
    l_v = float(metadata.get("l_v", 0.0))
    l_a = float(metadata.get("l_a", 0.0))
    s_values = s0 + s_v * times + 0.5 * s_a * times**2
    l_values = l0 + l_v * times + 0.5 * l_a * times**2
    poses = np.asarray(
        [pose_from_sl(scenario, float(s), float(l), actor=actor) for s, l in zip(s_values, l_values)],
        dtype=float,
    )
    if poses.ndim != 2 or poses.shape[1] < 2:
        return None
    return poses[:, 0], poses[:, 1]


def apply_game_actor_states(
    scenario,
    problem: PlanningProblem,
    actor_states: Mapping[str, EgoState],
    sim_time: float,
    controlled_actor_ids: Sequence[str] = (),
) -> PlanningProblem:
    controlled = {str(actor_id) for actor_id in controlled_actor_ids}
    if not actor_states and not controlled:
        return problem
    actors = []
    for actor in problem.actors:
        state = actor_states.get(actor.actor_id)
        if state is None and actor.actor_id in controlled:
            state = state_from_actor_prediction(actor)
        actors.append(actor if state is None else actor_prediction_from_state(scenario, actor, state, float(sim_time)))
    return PlanningProblem(
        ego=problem.ego,
        ref_path=problem.ref_path,
        road_boundary=problem.road_boundary,
        horizon=problem.horizon,
        dt=problem.dt,
        actors=tuple(actors),
        metadata=dict(problem.metadata or {}),
    )


def state_from_actor_prediction(actor: ActorPrediction) -> EgoState:
    metadata = dict(actor.metadata or {})
    return EgoState(
        s=float(metadata.get("s", 0.0)),
        l=float(metadata.get("l", 0.0)),
        s_v=float(metadata.get("s_v", 0.0)),
        l_v=float(metadata.get("l_v", 0.0)),
        s_a=float(metadata.get("s_a", 0.0)),
        l_a=float(metadata.get("l_a", 0.0)),
    )


def actor_prediction_from_state(
    scenario,
    template: ActorPrediction,
    state: EgoState,
    sim_time: float,
) -> ActorPrediction:
    x, y, yaw = pose_from_sl(scenario, float(state.s), float(state.l), actor=template)
    half_length = 0.5 * float(template.length)
    half_width = 0.5 * float(template.width)
    return ActorPrediction(
        actor_id=template.actor_id,
        actor_type=template.actor_type,
        times=np.array([float(sim_time)], dtype=float),
        x=np.array([x], dtype=float),
        y=np.array([y], dtype=float),
        yaw=np.array([yaw], dtype=float),
        length=float(template.length),
        width=float(template.width),
        metadata={
            **dict(template.metadata or {}),
            "s": float(state.s),
            "l": float(state.l),
            "s_v": float(state.s_v),
            "s_a": float(state.s_a),
            "l_v": float(state.l_v),
            "l_a": float(state.l_a),
            "blocked_s_min": float(state.s) - half_length,
            "blocked_s_max": float(state.s) + half_length,
            "blocked_l_min": float(state.l) - half_width,
            "blocked_l_max": float(state.l) + half_width,
            "temporal_blocked_range": {
                "t": np.array([0.0], dtype=float),
                "s_min": np.array([float(state.s) - half_length], dtype=float),
                "s_max": np.array([float(state.s) + half_length], dtype=float),
                "l_min": np.array([float(state.l) - half_width], dtype=float),
                "l_max": np.array([float(state.l) + half_width], dtype=float),
            },
            "static": False,
            "optimized_by_game": True,
        },
    )


def pose_from_sl(
    scenario,
    s: float,
    l: float,
    actor: Optional[ActorPrediction] = None,
) -> tuple[float, float, float]:
    actor_path = None if actor is None else dict(actor.metadata or {}).get("ref_path")
    if actor_path is not None:
        x_ref, y_ref = actor_path.calc_position(float(s))
        yaw = float(actor_path.calc_yaw(float(s)))
        return (
            float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0),
            float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0),
            yaw,
        )
    if hasattr(scenario, "_pose_from_sl"):
        return scenario._pose_from_sl(float(s), float(l))
    ref_path = scenario.ref_path
    x_ref, y_ref = ref_path.calc_position(float(s))
    yaw = float(ref_path.calc_yaw(float(s)))
    x = float(x_ref) + float(l) * math.cos(yaw + math.pi / 2.0)
    y = float(y_ref) + float(l) * math.sin(yaw + math.pi / 2.0)
    return x, y, yaw
