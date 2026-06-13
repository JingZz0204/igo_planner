from __future__ import annotations

from typing import Mapping

from spatiotemporal_joint_planner.application.configuration import (
    actor_type_profiles_from_config,
    config_section,
    config_value,
)
from spatiotemporal_joint_planner.belief import BayesianTypeFilter, BayesianTypeFilterConfig
from spatiotemporal_joint_planner.contingency import (
    BeliefContinuationConfig,
    BeliefContinuationEvaluator,
    RiskAggregator,
    RiskAggregatorConfig,
)
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig
from spatiotemporal_joint_planner.game.bayesian_game_parametric_planner import (
    BayesianGameParametricPlanner,
    BayesianGameParametricPlannerConfig,
)
from spatiotemporal_joint_planner.game.bayesian_igo_optimizer import BayesianIGOConfig, BayesianIGOOptimizer
from spatiotemporal_joint_planner.game.game_parametric_planner import GameParametricPlanner, GameParametricPlannerConfig
from spatiotemporal_joint_planner.game.igo_game_optimizer import GameIGOConfig, GameIGOOptimizer
from spatiotemporal_joint_planner.optimizer import CMAESConfig, CMAESOptimizer
from spatiotemporal_joint_planner.planner import ParametricPlanner, ParametricPlannerConfig


def _trajectory_cost(args) -> ParametricTrajectoryCost:
    certificate_enabled = config_section(args, "cost").get("trajectory_certificate_enabled", True)
    if args.scenario in {
        "interactive_lane_change",
        "dense_target_lane_change",
        "unprotected_intersection",
        "unprotected_left_turn",
    }:
        certificate_enabled = False
    return ParametricTrajectoryCost(
        ParametricTrajectoryCostConfig(
            road_edge_buffer=config_value(args, "cost", "road_edge_buffer", "road_edge_buffer", 1.0),
            min_lateral_accel=config_value(args, "cost", "min_lateral_accel", "min_lateral_accel", -2.5),
            max_lateral_accel=config_value(args, "cost", "max_lateral_accel", "max_lateral_accel", 2.5),
            lateral_accel_zero_comfort=config_value(args, "cost", "lateral_accel_zero_comfort", "lateral_accel_zero_comfort", 1.2),
            min_kappa=config_value(args, "cost", "min_kappa", "min_kappa", -0.20),
            max_kappa=config_value(args, "cost", "max_kappa", "max_kappa", 0.20),
            kappa_zero_comfort=config_value(args, "cost", "kappa_zero_comfort", "kappa_zero_comfort", 0.04),
            min_dkappa=config_value(args, "cost", "min_dkappa", "min_dkappa", -0.08),
            max_dkappa=config_value(args, "cost", "max_dkappa", "max_dkappa", 0.08),
            dkappa_zero_comfort=config_value(args, "cost", "dkappa_zero_comfort", "dkappa_zero_comfort", 0.04),
            min_lateral_jerk=config_value(args, "cost", "min_lateral_jerk", "min_lateral_jerk", -3.0),
            max_lateral_jerk=config_value(args, "cost", "max_lateral_jerk", "max_lateral_jerk", 3.0),
            lateral_jerk_zero_comfort=config_value(args, "cost", "lateral_jerk_zero_comfort", "lateral_jerk_zero_comfort", 1.0),
            max_longitudinal_speed=config_value(args, "cost", "max_speed", "max_longitudinal_speed", 15.0),
            speed_tracking_comfort=config_value(args, "cost", "speed_tracking_comfort", "speed_tracking_comfort", 2.5),
            efficiency_progress_comfort=config_value(args, "cost", "efficiency_progress_comfort", "efficiency_progress_comfort", 8.0),
            reference_lateral_comfort=config_value(args, "cost", "reference_lateral_comfort", "reference_lateral_comfort", 1.0),
            trajectory_certificate_enabled=certificate_enabled,
        )
    )


def _game_optimizer_config(args) -> GameIGOConfig:
    return GameIGOConfig(
        n_components=config_value(args, "optimizer", "components", "components", 2),
        n_samples=config_value(args, "optimizer", "samples", "samples", 48),
        n_iterations=config_value(args, "optimizer", "iters", "iterations", 50),
        elite_fraction=config_value(args, "optimizer", "elite", "elite_fraction", 0.25),
        init_std=config_value(args, "optimizer", "init_std", "init_std", 0.22),
        seed=config_value(args, "optimizer", "seed", "seed", 0),
        early_stop=bool(config_section(args, "optimizer").get("early_stop", True)),
        min_iterations=config_value(args, "optimizer", "min_iters", "min_iterations", 3),
        convergence_window=config_value(args, "optimizer", "convergence_window", "convergence_window", 5),
        cost_window_tol=config_value(args, "optimizer", "cost_window_tol", "cost_window_tol", 1e-3),
        theta_window_tol=config_value(args, "optimizer", "theta_window_tol", "theta_window_tol", 2e-2),
        component_sigma_tol=config_value(args, "optimizer", "component_sigma_tol", "component_sigma_tol", 0.08),
        component_weight_tol=config_value(args, "optimizer", "component_weight_tol", "component_weight_tol", 0.15),
        opponent_rank_gate=config_value(args, "optimizer", "opponent_rank_gate", "opponent_rank_gate", 0.35),
        nash_check=bool(config_value(args, "optimizer", "nash_check", "nash_check", True)),
        nash_regret_tol=config_value(args, "optimizer", "nash_regret_tol", "nash_regret_tol", 0.02),
        nash_candidate_limit=config_value(args, "optimizer", "nash_candidate_limit", "nash_candidate_limit", 48),
        nash_perturbation=config_value(args, "optimizer", "nash_perturbation", "nash_perturbation", 0.04),
        joint_theta_window_tol=config_value(args, "optimizer", "joint_theta_window_tol", "joint_theta_window_tol", 0.03),
        joint_cost_window_tol=config_value(args, "optimizer", "joint_cost_window_tol", "joint_cost_window_tol", 0.002),
    )


def _common_game_config(args, game_section: Mapping) -> dict:
    return {
        "target_actor_id": game_section.get("target_actor_id", "target_lane_rear_vehicle"),
        "candidate_limit": config_value(args, "planner", "mode_paths", "mode_paths", 8),
        "max_initial_anchors": config_value(args, "planner", "max_initial_anchors", "max_initial_anchors", 8),
        "target_min_terminal_speed": game_section.get("target_min_terminal_speed", 0.0),
        "target_max_terminal_speed": game_section.get(
            "target_max_terminal_speed",
            config_value(args, "trajectory_model", "max_terminal_speed", "max_terminal_speed", 15.0),
        ),
        "target_min_terminal_s_offset": game_section.get("target_min_terminal_s_offset", -10.0),
        "target_max_terminal_s_offset": game_section.get("target_max_terminal_s_offset", 25.0),
    }


def build_planner(args, trajectory_model):
    cost = _trajectory_cost(args)
    planner_type = str(config_section(args, "planner").get("type", "parametric")).lower()
    if planner_type == "parametric":
        return _build_parametric_planner(args, trajectory_model, cost)
    if planner_type not in {"game", "bayesian_game"}:
        raise ValueError(f"Unsupported planner.type: {planner_type}")

    optimizer_config = _game_optimizer_config(args)
    game_section = config_section(args, "game")
    common_config = _common_game_config(args, game_section)
    if planner_type == "game":
        return GameParametricPlanner(
            ego_trajectory_model=trajectory_model,
            ego_cost_function=cost,
            optimizer=GameIGOOptimizer(optimizer_config),
            config=GameParametricPlannerConfig(**common_config),
        )
    return _build_bayesian_game_planner(args, trajectory_model, cost, optimizer_config, game_section, common_config)


def _build_parametric_planner(args, trajectory_model, cost):
    optimizer = CMAESOptimizer(
        CMAESConfig(
            n_components=config_value(args, "optimizer", "components", "components", 2),
            n_samples=config_value(args, "optimizer", "samples", "samples", 48),
            n_iterations=config_value(args, "optimizer", "iters", "iterations", 50),
            elite_fraction=config_value(args, "optimizer", "elite", "elite_fraction", 0.25),
            init_std=config_value(args, "optimizer", "init_std", "init_std", 0.22),
            seed=config_value(args, "optimizer", "seed", "seed", 0),
            early_stop=bool(config_section(args, "optimizer").get("early_stop", True)),
            min_iterations=config_value(args, "optimizer", "min_iters", "min_iterations", 3),
            convergence_window=config_value(args, "optimizer", "convergence_window", "convergence_window", 5),
            cost_window_tol=config_value(args, "optimizer", "cost_window_tol", "cost_window_tol", 1e-3),
            theta_window_tol=config_value(args, "optimizer", "theta_window_tol", "theta_window_tol", 2e-2),
            component_sigma_tol=config_value(args, "optimizer", "component_sigma_tol", "component_sigma_tol", 0.08),
            component_weight_tol=config_value(args, "optimizer", "component_weight_tol", "component_weight_tol", 0.15),
        )
    )
    return ParametricPlanner(
        trajectory_model=trajectory_model,
        cost_function=cost,
        optimizer=optimizer,
        config=ParametricPlannerConfig(
            candidate_limit=config_value(args, "planner", "mode_paths", "mode_paths", 8),
            warm_start=bool(config_section(args, "planner").get("warm_start", True)),
            max_initial_anchors=config_value(args, "planner", "max_initial_anchors", "max_initial_anchors", 8),
            objective_mode=config_value(args, "planner", "objective_mode", "objective_mode", "vectorized"),
        ),
    )


def _build_bayesian_game_planner(args, trajectory_model, cost, optimizer_config, game_section, common_config):
    risk_values = game_section.get("risk", {})
    risk_values = risk_values if isinstance(risk_values, Mapping) else {}
    risk_aggregator = RiskAggregator(
        RiskAggregatorConfig(
            expected_weight=float(risk_values.get("expected_weight", 1.0)),
            cvar_weight=float(risk_values.get("cvar_weight", 0.35)),
            cvar_alpha=float(risk_values.get("cvar_alpha", 0.25)),
        )
    )
    equilibrium = game_section.get("equilibrium", {})
    equilibrium = equilibrium if isinstance(equilibrium, Mapping) else {}
    continuation = game_section.get("continuation", {})
    continuation = continuation if isinstance(continuation, Mapping) else {}
    filter_values = game_section.get("belief_filter", {})
    filter_values = filter_values if isinstance(filter_values, Mapping) else {}
    return BayesianGameParametricPlanner(
        ego_trajectory_model=trajectory_model,
        ego_cost_function=cost,
        optimizer=BayesianIGOOptimizer(
            BayesianIGOConfig(
                **vars(optimizer_config),
                equilibrium_check_interval=int(equilibrium.get("check_interval", 3)),
                equilibrium_regret_tol=float(equilibrium.get("regret_tol", 0.1)),
                material_type_probability=float(equilibrium.get("material_type_probability", 0.05)),
                local_nash_samples=int(equilibrium.get("local_nash_samples", 16)),
                local_nash_perturbation=float(equilibrium.get("local_nash_perturbation", 0.05)),
                local_nash_seed=int(equilibrium.get("local_nash_seed", 1701)),
            )
        ),
        config=BayesianGameParametricPlannerConfig(
            **common_config,
            target_actor_ids=tuple(str(value) for value in game_section.get("target_actor_ids", ())),
            simulated_target_type=str(game_section.get("simulated_target_type", "normal")),
            simulated_actor_types=dict(game_section.get("simulated_actor_types", {})),
            min_type_probability_for_feasibility=float(game_section.get("min_type_probability_for_feasibility", 0.05)),
            max_exact_hypotheses=int(game_section.get("max_exact_hypotheses", 81)),
        ),
        type_profiles=actor_type_profiles_from_config(args),
        risk_aggregator=risk_aggregator,
        continuation_evaluator=BeliefContinuationEvaluator(
            risk_aggregator,
            BeliefContinuationConfig(
                enabled=bool(continuation.get("enabled", True)),
                observation_time=float(continuation.get("observation_time", 0.8)),
                position_sigma=float(continuation.get("position_sigma", 2.0)),
                speed_sigma=float(continuation.get("speed_sigma", 1.0)),
                acceleration_sigma=float(continuation.get("acceleration_sigma", 1.0)),
                score_scale=float(continuation.get("score_scale", 500.0)),
                cost_weight=float(continuation.get("cost_weight", 100.0)),
            ),
        ),
        belief_filter=BayesianTypeFilter(
            BayesianTypeFilterConfig(
                position_sigma=float(filter_values.get("position_sigma", 0.8)),
                speed_sigma=float(filter_values.get("speed_sigma", 0.5)),
                acceleration_sigma=float(filter_values.get("acceleration_sigma", 0.6)),
                displacement_sigma=float(filter_values.get("displacement_sigma", 0.5)),
                speed_delta_sigma=float(filter_values.get("speed_delta_sigma", 0.35)),
                observation_window_seconds=float(filter_values.get("observation_window_seconds", 1.5)),
                evidence_gain=float(filter_values.get("evidence_gain", 0.4)),
                probability_floor=float(filter_values.get("probability_floor", 0.02)),
                forgetting_factor=float(filter_values.get("forgetting_factor", 0.005)),
            )
        ),
    )
