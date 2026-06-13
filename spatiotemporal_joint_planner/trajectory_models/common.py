from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem, Trajectory


def fixed_time_grid(horizon: float, dt: float) -> np.ndarray:
    horizon = max(float(horizon), 1e-3)
    dt = max(float(dt), 1e-3)
    values = np.arange(0.0, horizon + 0.5 * dt, dt, dtype=float)
    if values.size == 0:
        return np.array([0.0, horizon], dtype=float)
    if values[-1] < horizon - 1e-9:
        values = np.concatenate([values, [horizon]])
    else:
        values[-1] = horizon
    return values


def quartic_profile(
    s0: float,
    v0: float,
    a0: float,
    v1: float,
    t: np.ndarray,
    a1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    horizon = max(float(t[-1]), 1e-3)
    c0 = float(s0)
    c1 = float(v0)
    c2 = 0.5 * float(a0)
    matrix = np.array(
        [
            [3.0 * horizon**2, 4.0 * horizon**3],
            [6.0 * horizon, 12.0 * horizon**2],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            float(v1) - c1 - 2.0 * c2 * horizon,
            float(a1) - 2.0 * c2,
        ],
        dtype=float,
    )
    c3, c4 = np.linalg.solve(matrix, rhs)
    position = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4
    velocity = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3
    accel = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2
    return position, velocity, accel


def quintic_profile(
    p0: float,
    v0: float,
    a0: float,
    p1: float,
    v1: float,
    a1: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    horizon = max(float(t[-1]), 1e-3)
    c0 = float(p0)
    c1 = float(v0)
    c2 = 0.5 * float(a0)
    matrix = np.array(
        [
            [horizon**3, horizon**4, horizon**5],
            [3.0 * horizon**2, 4.0 * horizon**3, 5.0 * horizon**4],
            [6.0 * horizon, 12.0 * horizon**2, 20.0 * horizon**3],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            float(p1) - c0 - c1 * horizon - c2 * horizon**2,
            float(v1) - c1 - 2.0 * c2 * horizon,
            float(a1) - 2.0 * c2,
        ],
        dtype=float,
    )
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    position = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    velocity = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
    accel = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
    return position, velocity, accel


def finite_difference(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    edge_order = 2 if values.size >= 3 else 1
    return np.gradient(values, t, edge_order=edge_order)


def xy_from_sl(problem: PlanningProblem, s_values: np.ndarray, l_values: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ref_path = problem.ref_path
    if not hasattr(ref_path, "calc_position") or not hasattr(ref_path, "calc_yaw"):
        raise TypeError("SL-to-XY conversion requires ref_path.calc_position and ref_path.calc_yaw.")

    x_values = []
    y_values = []
    route_end = _route_end_s(ref_path)
    for s, l in zip(s_values, l_values):
        s_clamped = float(np.clip(float(s), 0.0, route_end))
        xy_ref = ref_path.calc_position(s_clamped)
        if xy_ref is None or xy_ref[0] is None or xy_ref[1] is None:
            raise ValueError(f"Reference path returned no position at s={s_clamped:.3f}.")
        yaw = float(ref_path.calc_yaw(s_clamped))
        x_values.append(float(xy_ref[0]) + float(l) * math.cos(yaw + math.pi / 2.0))
        y_values.append(float(xy_ref[1]) + float(l) * math.sin(yaw + math.pi / 2.0))
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def project_xy_to_sl(
    problem: PlanningProblem,
    x_values: np.ndarray,
    y_values: np.ndarray,
    projection_ds: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    ref_path = problem.ref_path
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if not hasattr(ref_path, "calc_position") or not hasattr(ref_path, "calc_yaw"):
        raise TypeError("XY-to-SL projection requires ref_path.calc_position and ref_path.calc_yaw.")

    route_end = _route_end_s(ref_path)
    ds = max(float(projection_ds), 0.05)
    ref_s = np.arange(0.0, route_end + 0.5 * ds, ds, dtype=float)
    ref_xy = []
    ref_yaw = []
    for s in ref_s:
        xy = ref_path.calc_position(float(s))
        if xy is None or xy[0] is None or xy[1] is None:
            continue
        ref_xy.append([float(xy[0]), float(xy[1])])
        ref_yaw.append(float(ref_path.calc_yaw(float(s))))

    if not ref_xy:
        raise ValueError("Reference path returned no valid samples for XY-to-SL projection.")

    ref_xy = np.asarray(ref_xy, dtype=float)
    ref_yaw = np.asarray(ref_yaw, dtype=float)
    ref_s = ref_s[: ref_xy.shape[0]]
    projected_s = []
    projected_l = []
    for x, y in zip(x_values, y_values):
        diff = ref_xy - np.array([float(x), float(y)], dtype=float)
        idx = int(np.argmin(np.sum(diff * diff, axis=1)))
        yaw = float(ref_yaw[idx])
        nx = math.cos(yaw + math.pi / 2.0)
        ny = math.sin(yaw + math.pi / 2.0)
        lateral = float((float(x) - ref_xy[idx, 0]) * nx + (float(y) - ref_xy[idx, 1]) * ny)
        projected_s.append(float(ref_s[idx]))
        projected_l.append(lateral)
    return np.asarray(projected_s, dtype=float), np.asarray(projected_l, dtype=float)


def trajectory_from_sl(
    problem: PlanningProblem,
    t: np.ndarray,
    s: np.ndarray,
    l: np.ndarray,
    s_v: Optional[np.ndarray] = None,
    l_v: Optional[np.ndarray] = None,
    s_a: Optional[np.ndarray] = None,
    l_a: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> Trajectory:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    l = np.asarray(l, dtype=float)
    if s_v is None:
        s_v = finite_difference(s, t)
    if l_v is None:
        l_v = finite_difference(l, t)
    if s_a is None:
        s_a = finite_difference(np.asarray(s_v, dtype=float), t)
    if l_a is None:
        l_a = finite_difference(np.asarray(l_v, dtype=float), t)

    x, y = xy_from_sl(problem, s, l)
    yaw = None
    kappa = None
    speed = np.hypot(np.asarray(s_v, dtype=float), np.asarray(l_v, dtype=float))
    accel = np.hypot(np.asarray(s_a, dtype=float), np.asarray(l_a, dtype=float))
    if x is not None and y is not None:
        yaw = np.arctan2(finite_difference(y, t), finite_difference(x, t))
        kappa = curvature_from_xy(x, y, t)

    return Trajectory(
        t=t,
        s=s,
        l=l,
        s_v=np.asarray(s_v, dtype=float),
        l_v=np.asarray(l_v, dtype=float),
        s_a=np.asarray(s_a, dtype=float),
        l_a=np.asarray(l_a, dtype=float),
        x=x,
        y=y,
        yaw=yaw,
        v=speed,
        a=accel,
        kappa=kappa,
        metadata=metadata or {},
    )


def curvature_from_xy(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    vx = finite_difference(np.asarray(x, dtype=float), t)
    vy = finite_difference(np.asarray(y, dtype=float), t)
    ax = finite_difference(vx, t)
    ay = finite_difference(vy, t)
    denom = np.maximum((vx * vx + vy * vy) ** 1.5, 1e-6)
    return (vx * ay - vy * ax) / denom


def _route_end_s(ref_path) -> float:
    if not hasattr(ref_path, "s"):
        raise TypeError("Reference path must expose cumulative arc-length samples through .s.")
    values = np.asarray(ref_path.s, dtype=float)
    if not values.size:
        raise ValueError("Reference path .s must not be empty.")
    return max(float(values[-1]), 1e-3)
