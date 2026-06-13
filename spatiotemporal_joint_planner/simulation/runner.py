from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from spatiotemporal_joint_planner.application import build_planner, build_scenario, build_trajectory_model
from spatiotemporal_joint_planner.common import EgoState
from spatiotemporal_joint_planner.simulation.execution import (
    apply_game_actor_states,
    ego_from_trajectory,
    ego_speed,
    execution_index,
    trajectory_delta_s,
    trajectory_speed_at,
)
from spatiotemporal_joint_planner.visualization import (
    create_live_figure,
    encode_mp4_from_frames,
    mp4_frame_path,
    plot_frame,
    prepare_mp4_frame_dir,
)


def run_simulation(args) -> str:
    scenario = build_scenario(args)
    trajectory_model = build_trajectory_model(args)
    planner = build_planner(args, trajectory_model)

    ego = scenario.initial_state()
    sim_time = 0.0
    status = "max_steps"
    mp4_output_path = None
    mp4_frame_dir = None
    mp4_frame_count = 0
    if args.save_mp4:
        mp4_output_path, mp4_frame_dir = prepare_mp4_frame_dir(args.save_mp4)

    live_figure_axes = create_live_figure(scenario) if args.show else None
    game_actor_states: dict[str, EgoState] = {}
    controlled_game_actor_ids = _controlled_game_actor_ids(planner)

    for step in range(int(args.max_steps)):
        problem = apply_game_actor_states(
            scenario,
            scenario.build_problem(ego, sim_time),
            game_actor_states,
            sim_time,
            controlled_actor_ids=controlled_game_actor_ids,
        )
        step_start = time.perf_counter()
        result = planner.plan(problem)
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0

        if result.trajectory is None or result.cost is None:
            print("No valid trajectory found.")
            status = "no_path"
            break

        exec_idx = execution_index(result.trajectory, args.planning_dt)
        if step % max(int(args.log_every), 1) == 0:
            print(format_step_log(step, elapsed_ms, problem, result, exec_idx))

        video_frame = None
        if mp4_frame_dir:
            video_frame = mp4_frame_path(mp4_frame_dir, mp4_frame_count)
            mp4_frame_count += 1
        if args.show or args.save_frame or video_frame:
            plot_frame(
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

        executed_dt = float(np.asarray(result.trajectory.t, dtype=float)[exec_idx])
        next_sim_time = sim_time + executed_dt
        ego = ego_from_trajectory(result.trajectory, exec_idx, next_sim_time)
        for actor_id, actor_trajectory in dict(result.metadata.get("game_actor_trajectories", {})).items():
            if actor_trajectory is None:
                continue
            actor_idx = min(max(exec_idx, 0), len(np.asarray(actor_trajectory.t, dtype=float)) - 1)
            game_actor_states[str(actor_id)] = ego_from_trajectory(actor_trajectory, actor_idx, next_sim_time)
        sim_time = next_sim_time
        if ego.s >= float(scenario.ref_path.s[-1]) - 2.0:
            status = "goal"
            break

    if args.show:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()
    if mp4_output_path and mp4_frame_dir:
        encode_mp4_from_frames(mp4_frame_dir, mp4_output_path, args.mp4_fps)
    print(f"simulation_status={status}")
    return status


def _controlled_game_actor_ids(planner) -> tuple[str, ...]:
    if not hasattr(planner, "config"):
        return ()
    if hasattr(planner.config, "target_actor_ids") and planner.config.target_actor_ids:
        return tuple(str(value) for value in planner.config.target_actor_ids)
    if hasattr(planner.config, "target_actor_id"):
        return (str(planner.config.target_actor_id),)
    return ()


def format_step_log(step: int, elapsed_ms: float, problem, result, exec_idx: int) -> str:
    terms = result.cost.breakdown.terms
    game_metadata = dict(result.metadata.get("game_metadata", {})) if result.metadata else {}
    game_parameters = dict(result.metadata.get("game_best_parameters", {})) if result.metadata else {}
    theta = (
        np.asarray(result.optimization.best_position, dtype=float)
        if result.optimization is not None
        else np.array([])
    )
    theta_repr = (
        {str(name): np.round(np.asarray(value, dtype=float), 3).tolist() for name, value in game_parameters.items()}
        if game_parameters
        else np.round(theta, 3).tolist()
    )
    opt_metadata = (
        game_metadata
        if result.optimization is None and game_metadata
        else {} if result.optimization is None else result.optimization.metadata
    )
    nash_summary = _nash_summary(opt_metadata)
    target_summary = _target_summary(result, exec_idx)
    belief_summary = _belief_summary(result)
    executed_iters = int(opt_metadata.get("executed_iterations", 0))
    max_iters = int(opt_metadata.get("n_iterations", 0))
    stop_reason = str(opt_metadata.get("stop_reason", ""))
    trajectory = result.trajectory
    return (
        f"step={step:03d} time={elapsed_ms:8.2f} ms status={result.status:>15s} "
        f"iter={executed_iters:02d}/{max_iters:02d} stop={stop_reason:<36s} "
        f"cost={result.cost.total:10.2f} theta={theta_repr} "
        f"ego_v={ego_speed(problem.ego):5.2f} exec_v={trajectory_speed_at(trajectory, exec_idx):5.2f} "
        f"end_v={trajectory_speed_at(trajectory, len(np.asarray(trajectory.t, dtype=float)) - 1):5.2f} "
        f"ds={trajectory_delta_s(trajectory):5.1f} "
        f"collision={terms.get('collision_flag', 0.0):.0f} road={terms.get('road_flag', 0.0):.0f} "
        f"lat_acc={terms.get('lateral_accel_flag', 0.0):.0f} k={terms.get('kappa_flag', 0.0):.0f} "
        f"dk={terms.get('dkappa_flag', 0.0):.0f} jerk={terms.get('lateral_jerk_flag', 0.0):.0f} "
        f"speed={terms.get('speed_flag', 0.0):.0f} eff={terms.get('efficiency_cost', 0.0):.3f} "
        f"ref_l={terms.get('reference_lateral_cost', 0.0):.3f} dyn={terms.get('dynamic_score', 0.0):.3f} "
        f"tv={terms.get('terminal_value_score', 0.0):.3f} "
        f"tf={terms.get('terminal_future_feasibility_score', 0.0):.3f} "
        f"tr={terms.get('terminal_recoverability_score', 0.0):.3f} "
        f"tp={terms.get('terminal_progress_score', 0.0):.3f} "
        f"ts={terms.get('terminal_speed_score', 0.0):.3f} "
        f"ic={terms.get('implicit_contingency_score', 0.0):.3f}"
        f"{nash_summary}{target_summary}{belief_summary}"
    )


def _nash_summary(metadata: Mapping) -> str:
    if metadata.get("bayesian_equilibrium_check", False):
        target_regrets = [
            float(value)
            for key, value in metadata.items()
            if str(key).endswith("_bayesian_regret")
            and str(key) not in {"max_bayesian_regret", "ego_bayesian_regret"}
        ]
        return (
            f" bayes_eq={float(metadata.get('bayesian_equilibrium_converged', 0.0)):.0f}"
            f" bayes_reg={float(metadata.get('max_bayesian_regret', float('inf'))):.3f}"
            f" ego_reg={float(metadata.get('ego_bayesian_regret', float('inf'))):.3f}"
            f" target_reg={max(target_regrets) if target_regrets else float('inf'):.3f}"
            f" local_checks={int(metadata.get('local_nash_check_count', 0))}"
            f" joint_shift={float(metadata.get('joint_theta_window_shift', float('inf'))):.3f}"
        )
    if metadata.get("nash_check", False):
        target_regrets = [
            float(value)
            for key, value in metadata.items()
            if str(key).startswith("target_rear") and str(key).endswith("_nash_regret")
        ]
        return (
            f" feas={float(metadata.get('selected_joint_feasible', 0.0)):.0f}"
            f" nash={float(metadata.get('max_nash_regret', float('inf'))):.3f}"
            f" ego_reg={float(metadata.get('ego_nash_regret', float('inf'))):.3f}"
            f" target_reg={max(target_regrets) if target_regrets else float('inf'):.3f}"
            f" joint_shift={float(metadata.get('joint_theta_window_shift', float('inf'))):.3f}"
        )
    return ""


def _target_summary(result, exec_idx: int) -> str:
    target_cost = result.metadata.get("game_target_cost") if result.metadata else None
    if target_cost is None:
        return ""
    target_terms = target_cost.breakdown.terms
    trajectory = result.metadata.get("game_target_trajectory")
    if trajectory is None:
        speed_summary = ""
    else:
        exec_speed = trajectory_speed_at(trajectory, exec_idx, 0.0)
        end_speed = trajectory_speed_at(trajectory, len(np.asarray(trajectory.t, dtype=float)) - 1, exec_speed)
        speed_summary = f" target_exec_v={exec_speed:.2f} target_end_v={end_speed:.2f}"
    return (
        f" target_cost={target_cost.total:.1f} "
        f"target_collision={target_terms.get('collision_flag', 0.0):.0f}{speed_summary}"
    )


def _belief_summary(result) -> str:
    belief = result.metadata.get("game_target_type_belief") if result.metadata else None
    if not isinstance(belief, Mapping):
        return ""
    return " belief=" + str({str(name): round(float(value), 3) for name, value in belief.items()})
