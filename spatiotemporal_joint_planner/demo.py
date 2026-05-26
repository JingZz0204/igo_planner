from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlannerResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig
from spatiotemporal_joint_planner.optimizer import CMAESConfig, CMAESOptimizer
from spatiotemporal_joint_planner.planner import ParametricPlanner, ParametricPlannerConfig
from spatiotemporal_joint_planner.scenario import LaneChangeScenario, StaticNudgeScenario
from spatiotemporal_joint_planner.trajectory_models import (
    BezierTrajectoryModel,
    LatticeTrajectoryConfig,
    LatticeTrajectoryModel,
    SvgdParticleTrajectoryModel,
)


def _prepare_mp4_frame_dir(mp4_path: str) -> tuple[str, str]:
    output_path = os.path.abspath(mp4_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root, _ = os.path.splitext(output_path)
    base_frame_dir = root + "_frames"
    frame_dir = base_frame_dir
    suffix = 1
    while os.path.isdir(frame_dir) and any(name.startswith("frame_") and name.endswith(".png") for name in os.listdir(frame_dir)):
        frame_dir = f"{base_frame_dir}_{suffix:03d}"
        suffix += 1
    os.makedirs(frame_dir, exist_ok=True)
    return output_path, frame_dir


def _mp4_frame_path(frame_dir: str, frame_index: int) -> str:
    return os.path.join(frame_dir, f"frame_{frame_index:05d}.png")


def _find_ffmpeg_executable() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _encode_mp4_from_frames(frame_dir: str, output_path: str, fps: float) -> bool:
    first_frame = _mp4_frame_path(frame_dir, 0)
    if not os.path.exists(first_frame):
        print(f"MP4 skipped: no frames were written to {frame_dir}")
        return False

    ffmpeg = _find_ffmpeg_executable()
    if ffmpeg is None:
        print(f"MP4 skipped: ffmpeg/imageio-ffmpeg unavailable. Frames are kept in {frame_dir}")
        return False

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{max(float(fps), 1.0):g}",
        "-i",
        os.path.join(frame_dir, "frame_%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-8:])
        print(f"MP4 encoding failed. Frames are kept in {frame_dir}")
        if tail:
            print(tail)
        return False

    print(f"MP4 saved: {output_path}")
    return True


def _build_scenario(args):
    if args.scenario == "static_nudge":
        return StaticNudgeScenario(
            horizon=args.horizon,
            dt=args.traj_dt,
            road_width=args.road_width,
            lane_width=args.lane_width,
            default_start_l=args.start_l,
            target_speed=args.target_speed,
        )
    if args.scenario == "lane_change":
        return LaneChangeScenario(
            horizon=args.horizon,
            dt=args.traj_dt,
            lane_width=args.lane_width,
            current_lane_l=args.lane_change_current_l,
            target_lane_l=args.lane_change_target_l,
            target_speed=args.target_speed,
            route_length=args.lane_change_route_length,
            road_side_margin=args.lane_change_side_margin,
        )
    raise ValueError(f"Unsupported scenario: {args.scenario}")


def _build_trajectory_model(args):
    if args.trajectory_model == "lattice_trajectory":
        return LatticeTrajectoryModel(
            LatticeTrajectoryConfig(
                min_terminal_speed=args.min_terminal_speed,
                max_terminal_speed=args.max_terminal_speed,
            )
        )
    if args.trajectory_model == "svgd_particle_trajectory":
        return SvgdParticleTrajectoryModel(
            LatticeTrajectoryConfig(
                min_terminal_speed=args.min_terminal_speed,
                max_terminal_speed=args.max_terminal_speed,
            )
        )
    if args.trajectory_model == "bezier_trajectory":
        return BezierTrajectoryModel()
    raise ValueError(f"Unsupported trajectory model: {args.trajectory_model}")


def _build_planner(args, trajectory_model) -> ParametricPlanner:
    optimizer = CMAESOptimizer(
        CMAESConfig(
            n_components=args.components,
            n_samples=args.samples,
            n_iterations=args.iters,
            elite_fraction=args.elite,
            init_std=args.init_std,
            seed=args.seed,
            early_stop=not args.disable_early_stop,
            min_iterations=args.min_iters,
            convergence_window=args.convergence_window,
            cost_window_tol=args.cost_window_tol,
            theta_window_tol=args.theta_window_tol,
            component_sigma_tol=args.component_sigma_tol,
            component_weight_tol=args.component_weight_tol,
        )
    )
    cost = ParametricTrajectoryCost(
        ParametricTrajectoryCostConfig(
            road_edge_buffer=args.road_edge_buffer,
            min_lateral_accel=args.min_lateral_accel,
            max_lateral_accel=args.max_lateral_accel,
            lateral_accel_zero_comfort=args.lateral_accel_zero_comfort,
            min_kappa=args.min_kappa,
            max_kappa=args.max_kappa,
            kappa_zero_comfort=args.kappa_zero_comfort,
            min_dkappa=args.min_dkappa,
            max_dkappa=args.max_dkappa,
            dkappa_zero_comfort=args.dkappa_zero_comfort,
            min_lateral_jerk=args.min_lateral_jerk,
            max_lateral_jerk=args.max_lateral_jerk,
            lateral_jerk_zero_comfort=args.lateral_jerk_zero_comfort,
            max_longitudinal_speed=args.max_speed,
            speed_tracking_comfort=args.speed_tracking_comfort,
            efficiency_progress_comfort=args.efficiency_progress_comfort,
            reference_lateral_comfort=args.reference_lateral_comfort,
            trajectory_certificate_enabled=not args.disable_trajectory_certificate,
        )
    )
    return ParametricPlanner(
        trajectory_model=trajectory_model,
        cost_function=cost,
        optimizer=optimizer,
        config=ParametricPlannerConfig(
            candidate_limit=args.mode_paths,
            warm_start=not args.no_warm_start,
            max_initial_anchors=args.max_initial_anchors,
            objective_mode=args.objective_mode,
        ),
    )


def _execution_index(trajectory: Trajectory, planning_dt: float) -> int:
    t_values = np.asarray(trajectory.t, dtype=float)
    if t_values.size < 2:
        return 0
    idx = int(np.searchsorted(t_values, float(planning_dt), side="left"))
    return min(max(idx, 1), t_values.size - 1)


def _ego_from_trajectory(trajectory: Trajectory, index: int, absolute_t: float) -> EgoState:
    s = float(np.asarray(trajectory.s, dtype=float)[index])
    l = float(np.asarray(trajectory.l, dtype=float)[index])
    s_v = _array_value(trajectory.s_v, index, 0.0)
    l_v = _array_value(trajectory.l_v, index, 0.0)
    s_a = _array_value(trajectory.s_a, index, 0.0)
    l_a = _array_value(trajectory.l_a, index, 0.0)
    yaw = None if trajectory.yaw is None else float(np.asarray(trajectory.yaw, dtype=float)[index])
    return EgoState(s=s, l=l, s_v=s_v, l_v=l_v, s_a=s_a, l_a=l_a, yaw=yaw, t=float(absolute_t))


def _ego_speed(ego: EgoState) -> float:
    return float(math.hypot(float(ego.s_v), float(ego.l_v)))


def _trajectory_speed_values(trajectory: Trajectory) -> np.ndarray:
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


def _trajectory_speed_at(trajectory: Trajectory, index: int, default: float = 0.0) -> float:
    values = _trajectory_speed_values(trajectory)
    if values.size == 0:
        return float(default)
    return float(values[min(max(int(index), 0), values.size - 1)])


def _trajectory_delta_s(trajectory: Trajectory) -> float:
    s_values = np.asarray(trajectory.s, dtype=float)
    if s_values.size < 2:
        return 0.0
    return float(s_values[-1] - s_values[0])


def _array_value(values, index: int, default: float) -> float:
    if values is None:
        return float(default)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(arr[min(max(int(index), 0), arr.size - 1)])


def _plot_frame(
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
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = figure_axes
        ax.clear()

    _plot_reference_lines(ax, scenario, problem)

    left_l = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
    right_l = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
    bx_left, by_left = scenario.offset_curve(left_l, ds=0.25)
    bx_right, by_right = scenario.offset_curve(right_l, ds=0.25)
    ax.plot(bx_left, by_left, color="black", linewidth=2.0)
    ax.plot(bx_right, by_right, color="black", linewidth=2.0)
    for lane_x, lane_y in scenario.lane_markings:
        ax.plot(lane_x, lane_y, color="0.65", linewidth=1.2, linestyle=(0, (6, 5)))

    _plot_actors(ax, problem.actors)

    for candidate in result.candidates[: max(0, len(result.candidates))]:
        if candidate.x is None or candidate.y is None:
            continue
        ax.plot(candidate.x, candidate.y, color="#7E8AA2", alpha=0.25, linewidth=1.2)

    if trajectory.x is not None and trajectory.y is not None:
        ax.plot(trajectory.x, trajectory.y, color="#F28E2B", linewidth=3.0, label="planned")
        idx = min(max(exec_idx, 0), len(trajectory.x) - 1)
        ax.scatter([trajectory.x[idx]], [trajectory.y[idx]], s=45, color="#D55E00", zorder=8)
        yaw = None if trajectory.yaw is None else float(np.asarray(trajectory.yaw, dtype=float)[idx])
        ax.add_patch(
            Polygon(
                _box_corners(float(trajectory.x[idx]), float(trajectory.y[idx]), yaw or 0.0, 4.95, 2.70),
                closed=True,
                facecolor="#F28E2B",
                edgecolor="black",
                alpha=0.92,
                zorder=7,
            )
        )

    terms = {} if result.cost is None else result.cost.breakdown.terms
    ego_v = _ego_speed(problem.ego)
    exec_v = _trajectory_speed_at(trajectory, exec_idx, ego_v)
    end_v = _trajectory_speed_at(trajectory, len(np.asarray(trajectory.t, dtype=float)) - 1, exec_v)
    plan_ds = _trajectory_delta_s(trajectory)
    title = (
        f"{problem.metadata.get('scenario', scenario.name)} | {result.metadata.get('trajectory_model', '')} | "
        f"step={step:03d} status={result.status} cost={0.0 if result.cost is None else result.cost.total:.1f}\n"
        f"collision={terms.get('collision_flag', 0.0):.0f} road={terms.get('road_flag', 0.0):.0f} "
        f"lat_acc={terms.get('lateral_accel_flag', 0.0):.0f} k={terms.get('kappa_flag', 0.0):.0f} "
        f"dk={terms.get('dkappa_flag', 0.0):.0f} jerk={terms.get('lateral_jerk_flag', 0.0):.0f} "
        f"speed={terms.get('speed_flag', 0.0):.0f} "
        f"eff={terms.get('efficiency_cost', 0.0):.2f} "
        f"ref_l={terms.get('reference_lateral_cost', 0.0):.2f} dyn={terms.get('dynamic_score', 0.0):.2f} "
        f"tv={terms.get('terminal_value_score', 0.0):.2f}"
    )
    ax.set_title(title)
    ax.text(
        0.015,
        0.985,
        (
            f"ego_v: {ego_v:.2f} m/s ({ego_v * 3.6:.1f} km/h)\n"
            f"exec_v: {exec_v:.2f} m/s ({exec_v * 3.6:.1f} km/h)\n"
            f"end_v: {end_v:.2f} m/s ({end_v * 3.6:.1f} km/h)\n"
            f"plan_ds: {plan_ds:.1f} m / {problem.horizon:.1f} s"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        zorder=20,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(loc="upper right")
    if tight:
        if problem.metadata.get("scenario") == "lane_change":
            ax.set_xlim(-5, 145)
            ax.set_ylim(-10, 10)
        else:
            ax.set_xlim(-8, 108)
            ax.set_ylim(-4, 88)
        fig.tight_layout()

    if save_frame:
        os.makedirs(os.path.dirname(os.path.abspath(save_frame)), exist_ok=True)
        fig.savefig(save_frame, dpi=140)
    if show:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(max(float(pause_s), 0.001))
    if owns_figure:
        plt.close(fig)


def _plot_reference_lines(ax, scenario, problem: PlanningProblem) -> None:
    reference_lines = getattr(scenario, "reference_lines", None)
    if reference_lines is None:
        reference_lines = problem.metadata.get("reference_lines") if problem.metadata else None

    if reference_lines:
        for reference in reference_lines:
            if isinstance(reference, dict):
                line = reference.get("line")
                role = str(reference.get("role", "reference"))
            else:
                line = reference
                role = "reference"
            if not line:
                continue
            rx, ry = line
            if role == "target":
                ax.plot(rx, ry, "--", color="#2F80ED", linewidth=2.2, label="target reference")
            elif role == "current":
                ax.plot(rx, ry, "-.", color="0.45", linewidth=1.7, label="current reference")
            else:
                ax.plot(rx, ry, "--", color="0.55", linewidth=1.8, label="reference")
        return

    rx, ry = scenario.ref_line
    ax.plot(rx, ry, "--", color="0.55", linewidth=1.8, label="reference")


def _plot_actors(ax, actors: Sequence[ActorPrediction]) -> None:
    from matplotlib.patches import Polygon

    for actor in actors:
        if np.asarray(actor.x).size == 0 or np.asarray(actor.y).size == 0:
            continue
        x = float(np.asarray(actor.x, dtype=float)[0])
        y = float(np.asarray(actor.y, dtype=float)[0])
        yaw = float(np.asarray(actor.yaw, dtype=float)[0]) if np.asarray(actor.yaw).size else 0.0
        if actor.actor_type == "pedestrian":
            face = "#D95F02"
            edge = "#7F2704"
        else:
            face = "#4C566A"
            edge = "#1F2937"
        ax.add_patch(
            Polygon(
                _box_corners(x, y, yaw, actor.length, actor.width),
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
                alpha=0.92,
                zorder=6,
            )
        )


def _box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=float,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.column_stack([x + local[:, 0] * c - local[:, 1] * s, y + local[:, 0] * s + local[:, 1] * c])


def main_simulation(args) -> str:
    scenario = _build_scenario(args)
    trajectory_model = _build_trajectory_model(args)
    planner = _build_planner(args, trajectory_model)

    ego = scenario.initial_state()
    sim_time = 0.0
    status = "max_steps"
    mp4_output_path = None
    mp4_frame_dir = None
    mp4_frame_count = 0
    if args.save_mp4:
        mp4_output_path, mp4_frame_dir = _prepare_mp4_frame_dir(args.save_mp4)

    live_figure_axes = None
    if args.show:
        import matplotlib.pyplot as plt

        plt.ion()
        live_figure_axes = plt.subplots(figsize=(10, 7))

    for step in range(int(args.max_steps)):
        problem = scenario.build_problem(ego, sim_time)
        step_start = time.perf_counter()
        result = planner.plan(problem)
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0

        if result.trajectory is None or result.cost is None:
            print("No valid trajectory found.")
            status = "no_path"
            break

        exec_idx = _execution_index(result.trajectory, args.planning_dt)
        if step % max(int(args.log_every), 1) == 0:
            terms = result.cost.breakdown.terms
            theta = np.asarray(result.optimization.best_position, dtype=float) if result.optimization is not None else np.array([])
            opt_metadata = {} if result.optimization is None else result.optimization.metadata
            executed_iters = int(opt_metadata.get("executed_iterations", 0))
            max_iters = int(opt_metadata.get("n_iterations", 0))
            stop_reason = str(opt_metadata.get("stop_reason", ""))
            print(
                f"step={step:03d} time={elapsed_ms:8.2f} ms status={result.status:>15s} "
                f"iter={executed_iters:02d}/{max_iters:02d} stop={stop_reason:<36s} "
                f"cost={result.cost.total:10.2f} theta={np.round(theta, 3).tolist()} "
                f"ego_v={_ego_speed(problem.ego):5.2f} exec_v={_trajectory_speed_at(result.trajectory, exec_idx):5.2f} "
                f"end_v={_trajectory_speed_at(result.trajectory, len(np.asarray(result.trajectory.t, dtype=float)) - 1):5.2f} "
                f"ds={_trajectory_delta_s(result.trajectory):5.1f} "
                f"collision={terms.get('collision_flag', 0.0):.0f} road={terms.get('road_flag', 0.0):.0f} "
                f"lat_acc={terms.get('lateral_accel_flag', 0.0):.0f} k={terms.get('kappa_flag', 0.0):.0f} "
                f"dk={terms.get('dkappa_flag', 0.0):.0f} jerk={terms.get('lateral_jerk_flag', 0.0):.0f} "
                f"speed={terms.get('speed_flag', 0.0):.0f} "
                f"eff={terms.get('efficiency_cost', 0.0):.3f} "
                f"ref_l={terms.get('reference_lateral_cost', 0.0):.3f} dyn={terms.get('dynamic_score', 0.0):.3f} "
                f"tv={terms.get('terminal_value_score', 0.0):.3f} "
                f"tf={terms.get('terminal_future_feasibility_score', 0.0):.3f} "
                f"tr={terms.get('terminal_recoverability_score', 0.0):.3f} "
                f"tp={terms.get('terminal_progress_score', 0.0):.3f} "
                f"ts={terms.get('terminal_speed_score', 0.0):.3f}"
            )

        video_frame = None
        if mp4_frame_dir:
            video_frame = _mp4_frame_path(mp4_frame_dir, mp4_frame_count)
            mp4_frame_count += 1
        if args.show or args.save_frame or video_frame:
            _plot_frame(
                scenario,
                problem,
                result,
                step,
                exec_idx,
                save_frame=video_frame or args.save_frame,
                show=args.show,
                tight=video_frame is None,
                pause_s=args.pause,
                figure_axes=live_figure_axes,
            )

        sim_time += float(np.asarray(result.trajectory.t, dtype=float)[exec_idx])
        ego = _ego_from_trajectory(result.trajectory, exec_idx, sim_time)
        if ego.s >= float(scenario.ref_path.s[-1]) - 2.0:
            status = "goal"
            break

    if args.show:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    if mp4_output_path and mp4_frame_dir:
        _encode_mp4_from_frames(mp4_frame_dir, mp4_output_path, args.mp4_fps)

    print(f"simulation_status={status}")
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatiotemporal joint planner demo")
    parser.add_argument("--scenario", choices=["static_nudge", "lane_change"], default="static_nudge")
    parser.add_argument(
        "--trajectory-model",
        choices=["lattice_trajectory", "bezier_trajectory", "svgd_particle_trajectory"],
        default="lattice_trajectory",
    )
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--planning-dt", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--traj-dt", type=float, default=0.1)
    parser.add_argument("--road-width", type=float, default=8.0)
    parser.add_argument("--lane-width", type=float, default=3.6)
    parser.add_argument("--road-edge-buffer", type=float, default=1.0)
    parser.add_argument("--start-l", type=float, default=2.0)
    parser.add_argument("--lane-change-current-l", type=float, default=None)
    parser.add_argument("--lane-change-target-l", type=float, default=0.0)
    parser.add_argument("--lane-change-route-length", type=float, default=260.0)
    parser.add_argument("--lane-change-side-margin", type=float, default=1.0)
    parser.add_argument("--target-speed", type=float, default=30.0 / 3.6)
    parser.add_argument("--min-terminal-speed", type=float, default=0.5)
    parser.add_argument("--max-terminal-speed", type=float, default=15.0)
    parser.add_argument("--max-speed", type=float, default=15.0)
    parser.add_argument("--min-lateral-accel", type=float, default=-2.5)
    parser.add_argument("--max-lateral-accel", type=float, default=2.5)
    parser.add_argument("--lateral-accel-zero-comfort", type=float, default=1.2)
    parser.add_argument("--min-kappa", type=float, default=-0.20)
    parser.add_argument("--max-kappa", type=float, default=0.20)
    parser.add_argument("--kappa-zero-comfort", type=float, default=0.04)
    parser.add_argument("--min-dkappa", type=float, default=-0.08)
    parser.add_argument("--max-dkappa", type=float, default=0.08)
    parser.add_argument("--dkappa-zero-comfort", type=float, default=0.04)
    parser.add_argument("--min-lateral-jerk", type=float, default=-3.0)
    parser.add_argument("--max-lateral-jerk", type=float, default=3.0)
    parser.add_argument("--lateral-jerk-zero-comfort", type=float, default=1.0)
    parser.add_argument("--speed-tracking-comfort", type=float, default=2.5)
    parser.add_argument("--efficiency-progress-comfort", type=float, default=8.0)
    parser.add_argument("--reference-lateral-comfort", type=float, default=1.0)
    parser.add_argument("--disable-trajectory-certificate", action="store_true")
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--elite", type=float, default=0.25)
    parser.add_argument("--init-std", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--min-iters", type=int, default=3)
    parser.add_argument("--convergence-window", type=int, default=5)
    parser.add_argument("--cost-window-tol", type=float, default=1e-3)
    parser.add_argument("--theta-window-tol", type=float, default=2e-2)
    parser.add_argument("--component-sigma-tol", type=float, default=0.08)
    parser.add_argument("--component-weight-tol", type=float, default=0.15)
    parser.add_argument("--max-initial-anchors", type=int, default=96)
    parser.add_argument("--mode-paths", type=int, default=8)
    parser.add_argument("--objective-mode", choices=["auto", "vectorized", "scalar"], default="auto")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--save-frame", type=str, default=None)
    parser.add_argument("--save-mp4", type=str, default=None)
    parser.add_argument("--mp4-fps", type=float, default=10.0)
    parser.add_argument("--pause", type=float, default=0.08)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_show:
        args.show = False
    main_simulation(args)


if __name__ == "__main__":
    main()
