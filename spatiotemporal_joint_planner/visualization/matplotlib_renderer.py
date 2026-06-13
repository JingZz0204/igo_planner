from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, PlannerResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.simulation.execution import (
    actor_keep_baseline_xy,
    ego_speed,
    find_actor,
    trajectory_delta_s,
    trajectory_speed_at,
)


def is_interactive_visual_scenario(problem: PlanningProblem) -> bool:
    return str(problem.metadata.get("scenario", "")) in {
        "interactive_lane_change",
        "dense_target_lane_change",
        "unprotected_intersection",
        "unprotected_left_turn",
    }


def is_intersection_visual_scenario(problem: PlanningProblem) -> bool:
    return str(problem.metadata.get("scenario", "")) in {"unprotected_intersection", "unprotected_left_turn"}


def create_live_figure(scenario):
    import matplotlib.pyplot as plt

    plt.ion()
    if scenario.name in {"unprotected_intersection", "unprotected_left_turn"}:
        return plt.subplots(figsize=(9, 9))
    if scenario.name in {"interactive_lane_change", "dense_target_lane_change"}:
        return plt.subplots(figsize=(14, 6))
    return plt.subplots(figsize=(10, 7))


def plot_frame(
    scenario,
    problem: PlanningProblem,
    result: PlannerResult,
    step: int,
    exec_idx: int,
    save_frame: Optional[str],
    show: bool,
    tight: bool = True,
    pause_s: float = 0.08,
    figure_axes=None,
) -> None:
    if not show and figure_axes is None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    trajectory = result.trajectory
    if trajectory is None:
        return

    owns_figure = figure_axes is None
    if owns_figure:
        if is_intersection_visual_scenario(problem):
            fig, ax = plt.subplots(figsize=(9, 9))
        elif is_interactive_visual_scenario(problem):
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = figure_axes
        ax.clear()

    plot_reference_lines(ax, scenario, problem)
    plot_road(ax, scenario, problem)
    plot_actors(ax, problem.actors)
    for candidate in result.candidates:
        if candidate.x is not None and candidate.y is not None:
            ax.plot(candidate.x, candidate.y, color="#7E8AA2", alpha=0.25, linewidth=1.2)

    if trajectory.x is not None and trajectory.y is not None:
        ax.plot(trajectory.x, trajectory.y, color="#F28E2B", linewidth=3.0, label="planned")
        plot_control_points(ax, trajectory)
        idx = min(max(exec_idx, 0), len(trajectory.x) - 1)
        ax.scatter([trajectory.x[idx]], [trajectory.y[idx]], s=45, color="#D55E00", zorder=8)
        yaw = 0.0 if trajectory.yaw is None else float(np.asarray(trajectory.yaw, dtype=float)[idx])
        ax.add_patch(
            Polygon(
                box_corners(float(trajectory.x[idx]), float(trajectory.y[idx]), yaw, 4.95, 2.70),
                closed=True,
                facecolor="#F28E2B",
                edgecolor="black",
                alpha=0.92,
                zorder=7,
            )
        )

    target_actor_id = result.metadata.get("game_target_actor_id") if result.metadata else None
    target_actor = find_actor(problem, str(target_actor_id) if target_actor_id else None)
    baseline = None if target_actor is None else actor_keep_baseline_xy(scenario, target_actor, problem.horizon, problem.dt)
    if baseline is not None:
        ax.plot(
            baseline[0],
            baseline[1],
            color="#5FAE7D",
            linewidth=1.8,
            linestyle=(0, (4, 4)),
            alpha=0.9,
            label=f"{target_actor_id} keep",
        )

    game_trajectories = dict(result.metadata.get("game_actor_trajectories", {})) if result.metadata else {}
    colors = ("#2CA02C", "#8E44AD", "#1F77B4", "#D62728")
    for actor_index, (actor_id, actor_trajectory) in enumerate(game_trajectories.items()):
        if actor_trajectory is None or actor_trajectory.x is None or actor_trajectory.y is None:
            continue
        plot_game_trajectory(ax, actor_id, actor_trajectory, exec_idx, colors[actor_index % len(colors)])

    set_title_and_status(ax, scenario, problem, result, trajectory, step, exec_idx)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="upper right" if is_intersection_visual_scenario(problem) else "lower right")
    if tight:
        apply_visual_bounds(fig, ax, scenario, problem)
    if save_frame:
        os.makedirs(os.path.dirname(os.path.abspath(save_frame)), exist_ok=True)
        fig.savefig(save_frame, dpi=140)
    if show:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(max(float(pause_s), 0.001))
    if owns_figure:
        plt.close(fig)


def plot_road(ax, scenario, problem: PlanningProblem) -> None:
    road_boundaries = getattr(scenario, "road_boundaries", None)
    if road_boundaries:
        for boundary_x, boundary_y in road_boundaries:
            ax.plot(boundary_x, boundary_y, color="black", linewidth=2.0)
    else:
        left_l = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right_l = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        for boundary_x, boundary_y in (
            scenario.offset_curve(left_l, ds=0.25),
            scenario.offset_curve(right_l, ds=0.25),
            *getattr(scenario, "extra_road_boundaries", ()),
        ):
            ax.plot(boundary_x, boundary_y, color="black", linewidth=2.0)
    for lane_x, lane_y in scenario.lane_markings:
        ax.plot(lane_x, lane_y, color="0.65", linewidth=1.2, linestyle=(0, (6, 5)))


def plot_reference_lines(ax, scenario, problem: PlanningProblem) -> None:
    reference_lines = getattr(scenario, "reference_lines", None)
    if reference_lines is None:
        reference_lines = problem.metadata.get("reference_lines") if problem.metadata else None
    if not reference_lines:
        rx, ry = scenario.ref_line
        ax.plot(rx, ry, "--", color="0.55", linewidth=1.8, label="reference")
        return
    for reference in reference_lines:
        if isinstance(reference, dict):
            line = reference.get("line")
            role = str(reference.get("role", "reference"))
            name = str(reference.get("name", role))
        else:
            line, role, name = reference, "reference", "reference"
        if not line:
            continue
        rx, ry = line
        style = {
            "target": ("--", "#2F80ED", 2.2, "target reference"),
            "current": ("-.", "0.45", 1.7, "current reference"),
            "ego": ("-.", "#F28E2B", 1.8, name),
            "interaction": ("--", "#2F80ED", 1.6, name),
        }.get(role, ("--", "0.55", 1.8, "reference"))
        ax.plot(rx, ry, style[0], color=style[1], linewidth=style[2], label=style[3])


def plot_actors(ax, actors: Sequence[ActorPrediction]) -> None:
    from matplotlib.patches import Polygon

    for actor in actors:
        x_values = np.asarray(actor.x, dtype=float)
        y_values = np.asarray(actor.y, dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            continue
        yaw_values = np.asarray(actor.yaw, dtype=float)
        yaw = float(yaw_values[0]) if yaw_values.size else 0.0
        metadata = dict(actor.metadata or {})
        face, edge = (
            ("#2CA02C", "#0F5132")
            if bool(metadata.get("optimized_by_game", False))
            else ("#D95F02", "#7F2704")
            if actor.actor_type == "pedestrian"
            else ("#4C566A", "#1F2937")
        )
        if not bool(metadata.get("static", False)) and x_values.size > 1 and y_values.size > 1:
            times = np.asarray(actor.times, dtype=float)
            stride = (
                max(1, int(round(0.5 / max(abs(float(np.median(np.diff(times)))), 1e-3))))
                if times.size >= 2
                else max(1, x_values.size // 10)
            )
            for idx in range(stride, min(x_values.size, y_values.size), stride):
                future_yaw = float(yaw_values[idx]) if yaw_values.size > idx else yaw
                ax.add_patch(
                    Polygon(
                        box_corners(float(x_values[idx]), float(y_values[idx]), future_yaw, actor.length, actor.width),
                        closed=True,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=0.6,
                        alpha=0.12,
                        zorder=4,
                    )
                )
        ax.add_patch(
            Polygon(
                box_corners(float(x_values[0]), float(y_values[0]), yaw, actor.length, actor.width),
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
                alpha=0.92,
                zorder=6,
            )
        )


def plot_game_trajectory(ax, actor_id: str, trajectory: Trajectory, exec_idx: int, color: str) -> None:
    from matplotlib.patches import Polygon

    x = np.asarray(trajectory.x, dtype=float)
    y = np.asarray(trajectory.y, dtype=float)
    if x.size == 0 or y.size == 0:
        return
    ax.plot(x, y, color=color, linewidth=2.4, label=f"{actor_id} game")
    times = np.asarray(trajectory.t, dtype=float)
    stride = (
        max(1, int(round(0.5 / max(abs(float(np.median(np.diff(times)))), 1e-3))))
        if times.size >= 2
        else max(1, x.size // 10)
    )
    yaw_values = None if trajectory.yaw is None else np.asarray(trajectory.yaw, dtype=float)
    for future_idx in range(stride, min(x.size, y.size), stride):
        future_yaw = 0.0 if yaw_values is None else float(yaw_values[future_idx])
        ax.add_patch(
            Polygon(
                box_corners(float(x[future_idx]), float(y[future_idx]), future_yaw, 4.8, 2.0),
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.14,
                zorder=5,
            )
        )
    idx = min(max(exec_idx, 0), x.size - 1)
    yaw = 0.0 if yaw_values is None else float(yaw_values[idx])
    ax.add_patch(
        Polygon(
            box_corners(float(x[idx]), float(y[idx]), yaw, 4.8, 2.0),
            closed=True,
            facecolor=color,
            edgecolor="black",
            alpha=0.65,
            zorder=7,
        )
    )


def plot_control_points(ax, trajectory: Trajectory) -> None:
    metadata = dict(trajectory.metadata or {})
    model = metadata.get("model")
    if model == "frenet_bspline_trajectory":
        x_values, y_values, label, prefix = (
            metadata.get("semantic_control_x"),
            metadata.get("semantic_control_y"),
            "semantic controls",
            "S",
        )
    elif model in {"frenet_via_bspline_trajectory", "frenet_bezier_trajectory"}:
        x_values, y_values, label, prefix = (
            metadata.get("bspline_control_x"),
            metadata.get("bspline_control_y"),
            "Bezier controls" if model == "frenet_bezier_trajectory" else "B-spline controls",
            "C",
        )
    else:
        return
    if x_values is None or y_values is None:
        return
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    n = min(x.size, y.size)
    finite = np.isfinite(x[:n]) & np.isfinite(y[:n])
    x, y = x[:n][finite], y[:n][finite]
    if x.size == 0:
        return
    ax.plot(
        x,
        y,
        color="#6F4CC3",
        linewidth=1.8,
        linestyle=(0, (2, 3)),
        marker="D",
        markersize=5.5,
        markerfacecolor="#FFFFFF",
        markeredgecolor="#6F4CC3",
        label=label,
        zorder=9,
    )
    for idx, (cx, cy) in enumerate(zip(x, y)):
        ax.text(float(cx), float(cy), f"{prefix}{idx}", color="#4B2E83", fontsize=8, ha="left", va="bottom", zorder=10)


def set_title_and_status(ax, scenario, problem, result, trajectory, step: int, exec_idx: int) -> None:
    terms = {} if result.cost is None else result.cost.breakdown.terms
    current_speed = ego_speed(problem.ego)
    exec_speed = trajectory_speed_at(trajectory, exec_idx, current_speed)
    end_speed = trajectory_speed_at(trajectory, len(np.asarray(trajectory.t, dtype=float)) - 1, exec_speed)
    ax.set_title(
        f"{problem.metadata.get('scenario', scenario.name)} | {result.metadata.get('trajectory_model', '')} | "
        f"step={step:03d} status={result.status} cost={0.0 if result.cost is None else result.cost.total:.1f}\n"
        f"collision={terms.get('collision_flag', 0.0):.0f} road={terms.get('road_flag', 0.0):.0f} "
        f"lat_acc={terms.get('lateral_accel_flag', 0.0):.0f} k={terms.get('kappa_flag', 0.0):.0f} "
        f"dk={terms.get('dkappa_flag', 0.0):.0f} jerk={terms.get('lateral_jerk_flag', 0.0):.0f} "
        f"speed={terms.get('speed_flag', 0.0):.0f} eff={terms.get('efficiency_cost', 0.0):.2f} "
        f"ref_l={terms.get('reference_lateral_cost', 0.0):.2f} dyn={terms.get('dynamic_score', 0.0):.2f} "
        f"tv={terms.get('terminal_value_score', 0.0):.2f}"
    )
    target_trajectory = result.metadata.get("game_target_trajectory") if result.metadata else None
    target_status = ""
    if target_trajectory is not None:
        target_exec_speed = trajectory_speed_at(target_trajectory, exec_idx, 0.0)
        target_end_speed = trajectory_speed_at(
            target_trajectory,
            len(np.asarray(target_trajectory.t, dtype=float)) - 1,
            target_exec_speed,
        )
        target_status = f"\ntarget_exec_v: {target_exec_speed:.2f} m/s\ntarget_end_v: {target_end_speed:.2f} m/s"
    ax.text(
        0.015,
        0.985,
        f"ego_v: {current_speed:.2f} m/s ({current_speed * 3.6:.1f} km/h)\n"
        f"exec_v: {exec_speed:.2f} m/s ({exec_speed * 3.6:.1f} km/h)\n"
        f"end_v: {end_speed:.2f} m/s ({end_speed * 3.6:.1f} km/h)\n"
        f"plan_ds: {trajectory_delta_s(trajectory):.1f} m / {problem.horizon:.1f} s"
        f"{target_status}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        zorder=20,
    )


def apply_visual_bounds(fig, ax, scenario, problem: PlanningProblem) -> None:
    if hasattr(scenario, "visual_bounds"):
        x_min, x_max, y_min, y_max = scenario.visual_bounds
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))
    elif is_interactive_visual_scenario(problem):
        ego_s = float(problem.ego.s)
        route_end = float(getattr(scenario.ref_path, "s", [260.0])[-1])
        left = max(-5.0, ego_s - 18.0)
        right = min(route_end + 10.0, ego_s + 92.0)
        if right - left < 90.0:
            right = min(route_end + 10.0, left + 90.0)
        ax.set_xlim(left, right)
        ax.set_ylim(-10, 10)
    elif problem.metadata.get("scenario") == "lane_change":
        ax.set_xlim(-5, 145)
        ax.set_ylim(-10, 10)
    else:
        ax.set_xlim(-8, 108)
        ax.set_ylim(-4, 88)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0) if is_interactive_visual_scenario(problem) else None)


def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    local = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=float,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.column_stack([x + local[:, 0] * c - local[:, 1] * s, y + local[:, 0] * s + local[:, 1] * c])
