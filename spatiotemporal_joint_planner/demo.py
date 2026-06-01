from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlannerResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig
from spatiotemporal_joint_planner.optimizer import CMAESConfig, CMAESOptimizer
from spatiotemporal_joint_planner.planner import ParametricPlanner, ParametricPlannerConfig
from spatiotemporal_joint_planner.scenario import (
    InteractiveLaneChangeScenario,
    LaneChangeActorSpec,
    LaneChangeScenario,
    StaticNudgeScenario,
    StaticObstacleSpec,
)
from spatiotemporal_joint_planner.trajectory_models import (
    BezierTrajectoryModel,
    LatticeTrajectoryConfig,
    LatticeTrajectoryModel,
    SvgdParticleTrajectoryModel,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEMO_CONFIG_DIR = PACKAGE_ROOT / "config" / "demo"
SCENARIO_CONFIG_DIR = PACKAGE_ROOT / "config" / "scenario"


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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML config files. Install dependency `pyyaml`.") from exc

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return dict(data)


def _scenario_config_path(args) -> Path:
    if args.scenario_config:
        return Path(args.scenario_config).expanduser().resolve()
    return SCENARIO_CONFIG_DIR / f"{args.scenario}.yaml"


def _demo_config_path(args) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    return DEMO_CONFIG_DIR / "default.yaml"


def _load_demo_config(args) -> dict[str, Any]:
    path = _demo_config_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Demo config not found: {path}")
    raw = _load_yaml_mapping(path)
    config = raw.get("demo", raw)
    if not isinstance(config, Mapping):
        raise ValueError(f"`demo` entry must be a mapping: {path}")
    config = dict(config)
    for override in args.set or []:
        _apply_config_override(config, override)
    args.config_path = str(path)
    return config


def _load_scenario_config(args) -> dict[str, Any]:
    path = _scenario_config_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    raw = _load_yaml_mapping(path)
    config = raw.get("scenario", raw)
    if not isinstance(config, Mapping):
        raise ValueError(f"`scenario` entry must be a mapping: {path}")
    config = dict(config)
    config_type = config.get("type")
    if config_type is not None and str(config_type) != str(args.scenario):
        raise ValueError(f"Scenario config type `{config_type}` does not match CLI scenario `{args.scenario}`: {path}")

    for override in args.scenario_set or []:
        _apply_config_override(config, override)
    args.scenario_config_path = str(path)
    return config


def _apply_config_override(config: dict[str, Any], override: str) -> None:
    if "=" not in str(override):
        raise ValueError(f"Config override must be key=value, got: {override}")
    key, raw_value = str(override).split("=", 1)
    keys = [part for part in key.strip().split(".") if part]
    if not keys:
        raise ValueError(f"Config override has empty key: {override}")

    try:
        import yaml

        value = yaml.safe_load(raw_value)
    except Exception:
        value = raw_value

    cursor = config
    for part in keys[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def _scenario_value(args, config: Mapping[str, Any], legacy_name: str, config_name: str, default: Any) -> Any:
    cli_value = getattr(args, legacy_name, None)
    if cli_value is not None:
        return cli_value
    return config.get(config_name, default)


def _config_section(args, section_name: str) -> Mapping[str, Any]:
    config = getattr(args, "demo_config_values", {})
    section = config.get(section_name, {}) if isinstance(config, Mapping) else {}
    return section if isinstance(section, Mapping) else {}


def _config_value(args, section_name: str, legacy_name: str, config_name: str, default: Any) -> Any:
    cli_value = getattr(args, legacy_name, None)
    if cli_value is not None:
        return cli_value
    return _config_section(args, section_name).get(config_name, default)


def _resolve_demo_runtime_args(args) -> None:
    args.max_steps = _config_value(args, "runtime", "max_steps", "max_steps", 150)
    args.log_every = _config_value(args, "runtime", "log_every", "log_every", 5)
    args.planning_dt = _config_value(args, "runtime", "planning_dt", "planning_dt", 0.25)
    args.save_frame = _config_value(args, "visualization", "save_frame", "save_frame", None)
    args.save_mp4 = _config_value(args, "visualization", "save_mp4", "save_mp4", None)
    args.mp4_fps = _config_value(args, "visualization", "mp4_fps", "mp4_fps", 10.0)
    args.pause = _config_value(args, "visualization", "pause", "pause", 0.08)

    show = _config_value(args, "visualization", "show", "show", False)
    if args.no_show is not None:
        show = not args.no_show
    args.show = bool(show)


def _resolve_lane_l(value: Any, current_lane_l: float, target_lane_l: float) -> float:
    if isinstance(value, str):
        if value == "current_lane":
            return float(current_lane_l)
        if value == "target_lane":
            return float(target_lane_l)
    return float(value)


def _static_obstacles_from_config(config: Mapping[str, Any]) -> Optional[tuple[StaticObstacleSpec, ...]]:
    obstacles = config.get("obstacles")
    if obstacles is None:
        return None
    specs = []
    for item in obstacles:
        specs.append(
            StaticObstacleSpec(
                actor_id=str(item.get("actor_id", "obstacle")),
                actor_type=str(item.get("actor_type", "vehicle")),
                s=float(item["s"]),
                l=float(item["l"]),
                length=float(item["length"]),
                width=float(item["width"]),
            )
        )
    return tuple(specs)


def _lane_change_actors_from_config(
    config: Mapping[str, Any],
    current_lane_l: float,
    target_lane_l: float,
) -> Optional[tuple[LaneChangeActorSpec, ...]]:
    actors = config.get("actors")
    if actors is None:
        return None
    specs = []
    for item in actors:
        specs.append(
            LaneChangeActorSpec(
                actor_id=str(item.get("actor_id", "actor")),
                actor_type=str(item.get("actor_type", "vehicle")),
                s=float(item["s"]),
                l=_resolve_lane_l(item.get("l", target_lane_l), current_lane_l, target_lane_l),
                length=float(item["length"]),
                width=float(item["width"]),
                s_v=float(item.get("s_v", item.get("v", 0.0))),
                l_v=float(item.get("l_v", 0.0)),
            )
        )
    return tuple(specs)


def _actor_config_value(
    config: Mapping[str, Any],
    actor_name: str,
    field: str,
    default: Any,
    aliases: Sequence[str] = (),
) -> Any:
    actors = config.get("actors", {})
    actor = actors.get(actor_name, {}) if isinstance(actors, Mapping) else {}
    for key in (field, *aliases):
        if isinstance(actor, Mapping) and key in actor:
            return actor[key]
    return default


def _build_scenario(args):
    config = getattr(args, "scenario_config_values", {})
    if args.scenario == "static_nudge":
        target_speed = _scenario_value(args, config, "target_speed", "target_speed", 30.0 / 3.6)
        return StaticNudgeScenario(
            horizon=_scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=_scenario_value(args, config, "traj_dt", "dt", 0.1),
            road_width=_scenario_value(args, config, "road_width", "road_width", 8.0),
            lane_width=_scenario_value(args, config, "lane_width", "lane_width", 3.6),
            default_start_l=_scenario_value(args, config, "start_l", "default_start_l", 2.0),
            target_speed=target_speed,
            obstacle_specs=_static_obstacles_from_config(config),
        )
    if args.scenario == "lane_change":
        lane_width = float(_scenario_value(args, config, "lane_width", "lane_width", 3.6))
        current_lane_l = _scenario_value(args, config, "lane_change_current_l", "current_lane_l", None)
        current_lane_l = None if current_lane_l is None else float(current_lane_l)
        target_lane_l = float(_scenario_value(args, config, "lane_change_target_l", "target_lane_l", 0.0))
        resolved_current_lane_l = current_lane_l if current_lane_l is not None else -lane_width
        return LaneChangeScenario(
            horizon=_scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=_scenario_value(args, config, "traj_dt", "dt", 0.1),
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            target_speed=_scenario_value(args, config, "target_speed", "target_speed", 12.0),
            route_length=_scenario_value(args, config, "lane_change_route_length", "route_length", 260.0),
            road_side_margin=_scenario_value(args, config, "lane_change_side_margin", "road_side_margin", 1.0),
            actor_specs=_lane_change_actors_from_config(config, resolved_current_lane_l, target_lane_l),
        )
    if args.scenario == "interactive_lane_change":
        lane_width = float(_scenario_value(args, config, "lane_width", "lane_width", 3.6))
        current_lane_l = _scenario_value(args, config, "lane_change_current_l", "current_lane_l", None)
        current_lane_l = None if current_lane_l is None else float(current_lane_l)
        target_lane_l = float(_scenario_value(args, config, "lane_change_target_l", "target_lane_l", 0.0))
        route_length = float(_scenario_value(args, config, "lane_change_route_length", "route_length", 520.0))
        return InteractiveLaneChangeScenario(
            horizon=_scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=_scenario_value(args, config, "traj_dt", "dt", 0.1),
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            ego_speed=_scenario_value(args, config, "interactive_ego_speed", "ego_speed", 9.0),
            target_speed=_scenario_value(args, config, "target_speed", "target_speed", 30.0 / 3.6),
            route_length=max(route_length, 520.0),
            road_side_margin=_scenario_value(args, config, "lane_change_side_margin", "road_side_margin", 1.0),
            interaction_mode=_scenario_value(args, config, "interaction_mode", "interaction_mode", "keep"),
            target_lead_s=_scenario_value(
                args, config, "target_lead_s", "target_lead_s", _actor_config_value(config, "target_lead", "s", 45.0)
            ),
            target_lead_v=_scenario_value(
                args,
                config,
                "target_lead_v",
                "target_lead_v",
                _actor_config_value(config, "target_lead", "s_v", 8.5, aliases=("v",)),
            ),
            target_rear_s=_scenario_value(
                args, config, "target_rear_s", "target_rear_s", _actor_config_value(config, "target_rear", "s", -18.0)
            ),
            target_rear_v=_scenario_value(
                args,
                config,
                "target_rear_v",
                "target_rear_v",
                _actor_config_value(config, "target_rear", "s_v", 15.0, aliases=("v",)),
            ),
            current_slow_s=_scenario_value(
                args,
                config,
                "current_slow_s",
                "current_slow_s",
                _actor_config_value(config, "current_slow", "s", 24.0),
            ),
            current_slow_v=_scenario_value(
                args,
                config,
                "current_slow_v",
                "current_slow_v",
                _actor_config_value(config, "current_slow", "s_v", 4.5, aliases=("v",)),
            ),
        )
    raise ValueError(f"Unsupported scenario: {args.scenario}")


def _build_trajectory_model(args):
    trajectory_model = _config_value(args, "trajectory_model", "trajectory_model", "type", "lattice_trajectory")
    if trajectory_model == "lattice_trajectory":
        return LatticeTrajectoryModel(
            LatticeTrajectoryConfig(
                min_terminal_speed=_config_value(
                    args, "trajectory_model", "min_terminal_speed", "min_terminal_speed", 0.5
                ),
                max_terminal_speed=_config_value(
                    args, "trajectory_model", "max_terminal_speed", "max_terminal_speed", 15.0
                ),
            )
        )
    if trajectory_model == "svgd_particle_trajectory":
        return SvgdParticleTrajectoryModel(
            LatticeTrajectoryConfig(
                min_terminal_speed=_config_value(
                    args, "trajectory_model", "min_terminal_speed", "min_terminal_speed", 0.5
                ),
                max_terminal_speed=_config_value(
                    args, "trajectory_model", "max_terminal_speed", "max_terminal_speed", 15.0
                ),
            )
        )
    if trajectory_model == "bezier_trajectory":
        return BezierTrajectoryModel()
    raise ValueError(f"Unsupported trajectory model: {trajectory_model}")


def _build_planner(args, trajectory_model) -> ParametricPlanner:
    early_stop = _config_section(args, "optimizer").get("early_stop", True)

    trajectory_certificate_enabled = _config_section(args, "cost").get("trajectory_certificate_enabled", True)
    if args.scenario == "interactive_lane_change":
        trajectory_certificate_enabled = False

    warm_start = _config_section(args, "planner").get("warm_start", True)

    optimizer = CMAESOptimizer(
        CMAESConfig(
            n_components=_config_value(args, "optimizer", "components", "components", 2),
            n_samples=_config_value(args, "optimizer", "samples", "samples", 48),
            n_iterations=_config_value(args, "optimizer", "iters", "iterations", 50),
            elite_fraction=_config_value(args, "optimizer", "elite", "elite_fraction", 0.25),
            init_std=_config_value(args, "optimizer", "init_std", "init_std", 0.22),
            seed=_config_value(args, "optimizer", "seed", "seed", 0),
            early_stop=bool(early_stop),
            min_iterations=_config_value(args, "optimizer", "min_iters", "min_iterations", 3),
            convergence_window=_config_value(args, "optimizer", "convergence_window", "convergence_window", 5),
            cost_window_tol=_config_value(args, "optimizer", "cost_window_tol", "cost_window_tol", 1e-3),
            theta_window_tol=_config_value(args, "optimizer", "theta_window_tol", "theta_window_tol", 2e-2),
            component_sigma_tol=_config_value(args, "optimizer", "component_sigma_tol", "component_sigma_tol", 0.08),
            component_weight_tol=_config_value(args, "optimizer", "component_weight_tol", "component_weight_tol", 0.15),
        )
    )
    cost = ParametricTrajectoryCost(
        ParametricTrajectoryCostConfig(
            road_edge_buffer=_config_value(args, "cost", "road_edge_buffer", "road_edge_buffer", 1.0),
            min_lateral_accel=_config_value(args, "cost", "min_lateral_accel", "min_lateral_accel", -2.5),
            max_lateral_accel=_config_value(args, "cost", "max_lateral_accel", "max_lateral_accel", 2.5),
            lateral_accel_zero_comfort=_config_value(
                args, "cost", "lateral_accel_zero_comfort", "lateral_accel_zero_comfort", 1.2
            ),
            min_kappa=_config_value(args, "cost", "min_kappa", "min_kappa", -0.20),
            max_kappa=_config_value(args, "cost", "max_kappa", "max_kappa", 0.20),
            kappa_zero_comfort=_config_value(args, "cost", "kappa_zero_comfort", "kappa_zero_comfort", 0.04),
            min_dkappa=_config_value(args, "cost", "min_dkappa", "min_dkappa", -0.08),
            max_dkappa=_config_value(args, "cost", "max_dkappa", "max_dkappa", 0.08),
            dkappa_zero_comfort=_config_value(args, "cost", "dkappa_zero_comfort", "dkappa_zero_comfort", 0.04),
            min_lateral_jerk=_config_value(args, "cost", "min_lateral_jerk", "min_lateral_jerk", -3.0),
            max_lateral_jerk=_config_value(args, "cost", "max_lateral_jerk", "max_lateral_jerk", 3.0),
            lateral_jerk_zero_comfort=_config_value(
                args, "cost", "lateral_jerk_zero_comfort", "lateral_jerk_zero_comfort", 1.0
            ),
            max_longitudinal_speed=_config_value(args, "cost", "max_speed", "max_longitudinal_speed", 15.0),
            speed_tracking_comfort=_config_value(
                args, "cost", "speed_tracking_comfort", "speed_tracking_comfort", 2.5
            ),
            efficiency_progress_comfort=_config_value(
                args, "cost", "efficiency_progress_comfort", "efficiency_progress_comfort", 8.0
            ),
            reference_lateral_comfort=_config_value(
                args, "cost", "reference_lateral_comfort", "reference_lateral_comfort", 1.0
            ),
            trajectory_certificate_enabled=trajectory_certificate_enabled,
        )
    )
    return ParametricPlanner(
        trajectory_model=trajectory_model,
        cost_function=cost,
        optimizer=optimizer,
        config=ParametricPlannerConfig(
            candidate_limit=_config_value(args, "planner", "mode_paths", "mode_paths", 8),
            warm_start=bool(warm_start),
            max_initial_anchors=_config_value(args, "planner", "max_initial_anchors", "max_initial_anchors", 96),
            objective_mode=_config_value(args, "planner", "objective_mode", "objective_mode", "auto"),
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
        if problem.metadata.get("scenario") == "interactive_lane_change":
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
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
    if problem.metadata.get("scenario") == "interactive_lane_change":
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper right")
    if tight:
        if problem.metadata.get("scenario") == "interactive_lane_change":
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
        if problem.metadata.get("scenario") == "interactive_lane_change":
            fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
        else:
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
        x_values = np.asarray(actor.x, dtype=float)
        y_values = np.asarray(actor.y, dtype=float)
        yaw_values = np.asarray(actor.yaw, dtype=float)
        x = float(x_values[0])
        y = float(y_values[0])
        yaw = float(yaw_values[0]) if yaw_values.size else 0.0
        if actor.actor_type == "pedestrian":
            face = "#D95F02"
            edge = "#7F2704"
        else:
            face = "#4C566A"
            edge = "#1F2937"
        metadata = dict(actor.metadata or {})
        if not bool(metadata.get("static", False)) and x_values.size > 1 and y_values.size > 1:
            times = np.asarray(actor.times, dtype=float)
            if times.size >= 2:
                dt = float(np.median(np.diff(times)))
                stride = max(1, int(round(0.5 / max(abs(dt), 1e-3))))
            else:
                stride = max(1, x_values.size // 10)
            for idx in range(stride, min(x_values.size, y_values.size), stride):
                future_yaw = float(yaw_values[idx]) if yaw_values.size > idx else yaw
                ax.add_patch(
                    Polygon(
                        _box_corners(float(x_values[idx]), float(y_values[idx]), future_yaw, actor.length, actor.width),
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
                tight=True,
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
    parser.add_argument("--scenario", choices=["static_nudge", "lane_change", "interactive_lane_change"], default="static_nudge")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML demo config path. Defaults to spatiotemporal_joint_planner/config/demo/default.yaml.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a demo YAML value with dot path, e.g. --set optimizer.samples=32.",
    )
    parser.add_argument(
        "--scenario-config",
        type=str,
        default=None,
        help="YAML scenario config path. Defaults to spatiotemporal_joint_planner/config/scenario/<scenario>.yaml.",
    )
    parser.add_argument(
        "--scenario-set",
        action="append",
        default=[],
        help="Override a scenario YAML value with dot path, e.g. --scenario-set actors.current_slow.s=30.",
    )
    parser.add_argument(
        "--trajectory-model",
        choices=["lattice_trajectory", "bezier_trajectory", "svgd_particle_trajectory"],
        default=None,
        help="Override trajectory_model.type from demo config.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override runtime.max_steps from demo config.")
    parser.add_argument("--save-frame", type=str, default=None, help="Save one frame to image path.")
    parser.add_argument("--save-mp4", type=str, default=None, help="Save an MP4 rollout to this path.")
    parser.add_argument("--show", action="store_true", default=None, help="Show live matplotlib visualization.")
    parser.add_argument("--no-show", action="store_true", default=None, help="Disable visualization even if config enables it.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.demo_config_values = _load_demo_config(args)
    args.scenario_config_values = _load_scenario_config(args)
    _resolve_demo_runtime_args(args)
    main_simulation(args)


if __name__ == "__main__":
    main()
