from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Mapping, Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.belief import (
    ActorTypeBelief,
    ActorTypeProfile,
    BayesianTypeFilter,
    BayesianTypeFilterConfig,
    LongitudinalObservation,
    LongitudinalTypePrediction,
    default_actor_type_profiles,
)
from spatiotemporal_joint_planner.common import CostBreakdown, CostResult, PlannerResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.contingency import BeliefContinuationConfig, BeliefContinuationEvaluator, RiskAggregator, RiskAggregatorConfig
from spatiotemporal_joint_planner.cost import CostFunction
from spatiotemporal_joint_planner.game.base import GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.bayesian_igo_optimizer import (
    BayesianBatchEvaluation,
    BayesianGameOptimizationProblem,
    BayesianIGOConfig,
    BayesianIGOOptimizer,
    BayesianPhysicalActor,
)
from spatiotemporal_joint_planner.game.cost import VehicleGameCost
from spatiotemporal_joint_planner.game.game_parametric_planner import GameParametricPlanner, GameParametricPlannerConfig
from spatiotemporal_joint_planner.game.trajectory_models import VehicleLongitudinalTrajectoryConfig, VehicleLongitudinalTrajectoryModel
from spatiotemporal_joint_planner.planner.warm_start import WarmStartGenerator
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class BayesianGameParametricPlannerConfig(GameParametricPlannerConfig):
    target_actor_ids: tuple[str, ...] = ()
    simulated_target_type: str = "normal"
    simulated_actor_types: Mapping[str, str] = field(default_factory=dict)
    min_type_probability_for_feasibility: float = 0.05
    max_exact_hypotheses: int = 81


class BayesianGameParametricPlanner(GameParametricPlanner):
    """Multi-actor Bayesian game planner with one shared ego strategy."""

    def __init__(
        self,
        ego_trajectory_model: TrajectoryModel,
        ego_cost_function: CostFunction,
        optimizer: Optional[BayesianIGOOptimizer] = None,
        config: Optional[BayesianGameParametricPlannerConfig] = None,
        target_cost_function: Optional[VehicleGameCost] = None,
        warm_start_generator: Optional[WarmStartGenerator] = None,
        type_profiles: Optional[Sequence[ActorTypeProfile]] = None,
        risk_aggregator: Optional[RiskAggregator] = None,
        belief_filter: Optional[BayesianTypeFilter] = None,
        continuation_evaluator: Optional[BeliefContinuationEvaluator] = None,
    ):
        risk_aggregator = risk_aggregator or RiskAggregator(RiskAggregatorConfig())
        super().__init__(
            ego_trajectory_model=ego_trajectory_model,
            ego_cost_function=ego_cost_function,
            optimizer=optimizer or BayesianIGOOptimizer(BayesianIGOConfig()),
            config=config or BayesianGameParametricPlannerConfig(),
            target_cost_function=target_cost_function,
            warm_start_generator=warm_start_generator,
        )
        self.config = config or BayesianGameParametricPlannerConfig()
        self.type_profiles = tuple(type_profiles or default_actor_type_profiles())
        names = [profile.name for profile in self.type_profiles]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Invalid target actor type names: {names}")
        self.risk_aggregator = risk_aggregator
        self.continuation_evaluator = continuation_evaluator or BeliefContinuationEvaluator(
            self.risk_aggregator, BeliefContinuationConfig()
        )
        filter_config = (belief_filter or BayesianTypeFilter(BayesianTypeFilterConfig())).config
        self._filter_config = filter_config
        self.beliefs: dict[str, ActorTypeBelief] = {}
        self.belief_filters: dict[str, BayesianTypeFilter] = {}
        self._last_target_parameters_by_actor_type: dict[tuple[str, str], np.ndarray] = {}
        self._last_type_predictions_by_actor: dict[str, dict[str, Trajectory]] = {}
        self._last_observations_by_actor: dict[str, LongitudinalObservation] = {}
        self._last_optimization_problem: Optional[BayesianGameOptimizationProblem] = None

    @property
    def name(self) -> str:
        return "multi_actor_bayesian_game_parametric_planner"

    @property
    def last_optimization_problem(self) -> Optional[BayesianGameOptimizationProblem]:
        return self._last_optimization_problem

    def plan(self, problem: PlanningProblem) -> PlannerResult:
        actor_ids = self._controlled_actor_ids(problem)
        target_actors = {actor.actor_id: actor for actor in problem.actors if actor.actor_id in actor_ids}
        missing = tuple(actor_id for actor_id in actor_ids if actor_id not in target_actors)
        if missing:
            return PlannerResult(
                trajectory=None,
                cost=None,
                status="missing_game_target_actor",
                metadata={"planner": self.name, "missing_actor_ids": missing},
            )

        type_names = tuple(profile.name for profile in self.type_profiles)
        for actor_id, actor in target_actors.items():
            self._ensure_actor_belief(actor_id)
            self._update_type_belief(actor_id, actor)
        probabilities_by_actor = {
            actor_id: self.beliefs[actor_id].vector(type_names)
            for actor_id in actor_ids
        }
        hypotheses = self._type_hypotheses(actor_ids, type_names, probabilities_by_actor)

        exogenous_actors = tuple(actor for actor in problem.actors if actor.actor_id not in actor_ids)
        ego_problem = PlanningProblem(
            ego=problem.ego,
            ref_path=problem.ref_path,
            road_boundary=problem.road_boundary,
            horizon=problem.horizon,
            dt=problem.dt,
            actors=exogenous_actors,
            metadata=dict(problem.metadata or {}),
        )
        ego_player = self._ego_player(ego_problem)
        target_problems: dict[tuple[str, str], PlanningProblem] = {}
        target_players: dict[tuple[str, str], GamePlayer] = {}
        target_costs: dict[tuple[str, str], VehicleGameCost] = {}
        physical_actors: dict[str, BayesianPhysicalActor] = {}
        for actor_id in actor_ids:
            players_by_type = {}
            for profile in self.type_profiles:
                key = (actor_id, profile.name)
                target_problems[key] = self._target_problem_for_type(
                    problem, target_actors[actor_id], exogenous_actors, profile
                )
                target_players[key] = self._target_player_for_type(actor_id, target_problems[key], profile)
                target_costs[key] = self._target_cost_for_type(profile)
                players_by_type[profile.name] = target_players[key]
            physical_actors[actor_id] = BayesianPhysicalActor(
                actor_id=actor_id,
                players_by_type=players_by_type,
                type_probabilities={
                    type_name: float(probabilities_by_actor[actor_id][index])
                    for index, type_name in enumerate(type_names)
                },
            )
        target_key_by_player_name = {player.name: key for key, player in target_players.items()}

        def decode_joint(parameters_by_player: Mapping[str, np.ndarray]) -> JointTrajectory:
            trajectories = {
                ego_player.name: self.ego_trajectory_model.decode(
                    np.asarray(parameters_by_player[ego_player.name], dtype=float), ego_problem
                )
            }
            for key, player in target_players.items():
                trajectories[player.name] = player.trajectory_model.decode(
                    np.asarray(parameters_by_player[player.name], dtype=float), target_problems[key]
                )
            return JointTrajectory(trajectories=trajectories)

        def evaluate_players_batch(parameters_by_player: Mapping[str, np.ndarray]) -> BayesianBatchEvaluation:
            ego_batch = self.ego_trajectory_model.decode_batch_arrays(
                np.asarray(parameters_by_player[ego_player.name], dtype=float), ego_problem
            )
            target_batches = {
                key: target_players[key].trajectory_model.decode_batch_arrays(
                    np.asarray(parameters_by_player[target_players[key].name], dtype=float),
                    target_problems[key],
                )
                for key in target_players
            }
            ego_values_by_hypothesis = []
            hypothesis_features = []
            target_weighted_values = {
                player.name: np.zeros((np.asarray(parameters_by_player[player.name]).shape[0],), dtype=float)
                for player in target_players.values()
            }
            target_probability_mass = {player.name: 0.0 for player in target_players.values()}
            hypothesis_probabilities = np.asarray([item[1] for item in hypotheses], dtype=float)

            for types_by_actor, probability in hypotheses:
                active = [
                    (
                        actor_id,
                        type_name,
                        target_batches[(actor_id, type_name)],
                        target_problems[(actor_id, type_name)],
                        target_costs[(actor_id, type_name)],
                    )
                    for actor_id, type_name in types_by_actor.items()
                ]
                ego_values_by_hypothesis.append(
                    self._evaluate_ego_multi_game_batch(
                        ego_batch,
                        [(item[2], item[3]) for item in active],
                        ego_problem,
                    )
                )
                features = np.asarray([self._continuation_features_from_batch(item[2]) for item in active], dtype=float)
                hypothesis_features.append(np.mean(features, axis=0))
                for actor_id, type_name, target_batch, target_problem, target_cost in active:
                    opponents = [(ego_batch, ego_problem)] + [
                        (other_batch, other_problem)
                        for other_actor_id, _other_type, other_batch, other_problem, _other_cost in active
                        if other_actor_id != actor_id
                    ]
                    player_name = target_players[(actor_id, type_name)].name
                    target_weighted_values[player_name] += float(probability) * self._evaluate_target_multi_game_batch(
                        target_batch, opponents, target_problem, target_cost
                    )
                    target_probability_mass[player_name] += float(probability)

            ego_cost_matrix = np.asarray(ego_values_by_hypothesis, dtype=float)
            aggregate_ego = self.risk_aggregator.aggregate_batch(ego_cost_matrix, hypothesis_probabilities)
            continuation = self.continuation_evaluator.evaluate_batch(
                ego_cost_matrix,
                np.asarray(hypothesis_features, dtype=float),
                hypothesis_probabilities,
            )
            aggregate_ego += np.asarray(continuation["implicit_contingency_cost"], dtype=float)
            player_values = {ego_player.name: aggregate_ego}
            for player_name, values in target_weighted_values.items():
                player_values[player_name] = values / max(float(target_probability_mass[player_name]), 1e-12)
            return BayesianBatchEvaluation(
                player_values=player_values,
                diagnostics={**continuation, "hypothesis_count": np.asarray([len(hypotheses)], dtype=float)},
            )

        def evaluate_unilateral_batch(
            player_name: str,
            candidate_parameters: np.ndarray,
            fixed_parameters: Mapping[str, np.ndarray],
        ) -> np.ndarray:
            candidates = np.asarray(candidate_parameters, dtype=float)
            batch_size = int(candidates.shape[0])
            if player_name != ego_player.name:
                actor_id, type_name = target_key_by_player_name[player_name]
                target_problem = target_problems[(actor_id, type_name)]
                candidate_batch = target_players[(actor_id, type_name)].trajectory_model.decode_batch_arrays(
                    candidates, target_problem
                )
                ego_batch = self.ego_trajectory_model.decode_batch_arrays(
                    np.repeat(
                        np.asarray(fixed_parameters[ego_player.name], dtype=float).reshape(1, -1),
                        batch_size,
                        axis=0,
                    ),
                    ego_problem,
                )
                other_batches = {
                    key: target_players[key].trajectory_model.decode_batch_arrays(
                        np.repeat(
                            np.asarray(fixed_parameters[target_players[key].name], dtype=float).reshape(1, -1),
                            batch_size,
                            axis=0,
                        ),
                        target_problems[key],
                    )
                    for key in target_players
                    if key[0] != actor_id
                }
                weighted = np.zeros((batch_size,), dtype=float)
                probability_mass = 0.0
                for types_by_actor, probability in hypotheses:
                    if types_by_actor[actor_id] != type_name:
                        continue
                    opponents = [(ego_batch, ego_problem)] + [
                        (
                            other_batches[(other_actor_id, other_type)],
                            target_problems[(other_actor_id, other_type)],
                        )
                        for other_actor_id, other_type in types_by_actor.items()
                        if other_actor_id != actor_id
                    ]
                    weighted += float(probability) * self._evaluate_target_multi_game_batch(
                        candidate_batch,
                        opponents,
                        target_problem,
                        target_costs[(actor_id, type_name)],
                    )
                    probability_mass += float(probability)
                return weighted / max(probability_mass, 1e-12)
            parameters = {
                name: (
                    candidates
                    if name == player_name
                    else np.repeat(np.asarray(value, dtype=float).reshape(1, -1), batch_size, axis=0)
                )
                for name, value in fixed_parameters.items()
            }
            return np.asarray(evaluate_players_batch(parameters).player_values[player_name], dtype=float)

        def evaluate_joint(joint: JointTrajectory) -> dict[str, CostResult]:
            parameters = {
                name: np.asarray(trajectory.metadata.get("theta"), dtype=float).reshape(1, -1)
                for name, trajectory in joint.trajectories.items()
            }
            values = evaluate_players_batch(parameters).player_values
            return {
                name: self._cost_result_from_total(float(np.asarray(player_values, dtype=float)[0]))
                for name, player_values in values.items()
            }

        game_problem = BayesianGameOptimizationProblem(
            ego_player=ego_player,
            physical_actors=physical_actors,
            evaluate_players_batch=evaluate_players_batch,
            evaluate_unilateral_batch=evaluate_unilateral_batch,
            decode_joint=decode_joint,
            evaluate_joint=evaluate_joint,
            metadata={
                "planner": self.name,
                "target_actor_ids": actor_ids,
                "hypothesis_count": len(hypotheses),
                "scenario": problem.metadata.get("scenario") if problem.metadata else None,
            },
        )
        self._last_optimization_problem = game_problem
        optimization = self.optimizer.optimize(game_problem)
        ego_trajectory = optimization.best_joint_trajectory.trajectories.get(ego_player.name)
        ego_cost = optimization.player_costs.get(ego_player.name)

        actor_type_trajectories = {
            actor_id: {
                type_name: optimization.best_joint_trajectory.trajectories.get(target_players[(actor_id, type_name)].name)
                for type_name in type_names
            }
            for actor_id in actor_ids
        }
        simulated_types = {
            actor_id: self._simulated_target_type(actor_id, type_names)
            for actor_id in actor_ids
        }
        actor_trajectories = {
            actor_id: actor_type_trajectories[actor_id].get(simulated_types[actor_id])
            for actor_id in actor_ids
        }
        actor_costs = {
            actor_id: optimization.player_costs.get(target_players[(actor_id, simulated_types[actor_id])].name)
            for actor_id in actor_ids
        }
        for actor_id in actor_ids:
            for type_name in type_names:
                player = target_players[(actor_id, type_name)]
                if player.name in optimization.best_parameters:
                    self._last_target_parameters_by_actor_type[(actor_id, type_name)] = np.asarray(
                        optimization.best_parameters[player.name], dtype=float
                    )
            self._last_type_predictions_by_actor[actor_id] = {
                type_name: trajectory
                for type_name, trajectory in actor_type_trajectories[actor_id].items()
                if trajectory is not None
            }
            self._last_observations_by_actor[actor_id] = self._observation_from_actor(target_actors[actor_id])
        if ego_trajectory is not None:
            self._last_ego_parameters = np.asarray(optimization.best_parameters[ego_player.name], dtype=float)

        first_actor_id = actor_ids[0]
        status = "success"
        if ego_trajectory is None or ego_cost is None:
            status = "no_valid_trajectory"
        elif not ego_cost.feasible:
            status = "infeasible_best"
        return PlannerResult(
            trajectory=ego_trajectory,
            cost=ego_cost,
            status=status,
            metadata={
                "planner": self.name,
                "trajectory_model": self.ego_trajectory_model.name,
                "optimizer": self.optimizer.name,
                "game_optimizer": optimization,
                "game_actor_ids": actor_ids,
                "game_actor_trajectories": actor_trajectories,
                "game_actor_costs": actor_costs,
                "game_actor_type_trajectories": actor_type_trajectories,
                "game_actor_type_beliefs": {
                    actor_id: dict(self.beliefs[actor_id].probabilities) for actor_id in actor_ids
                },
                "game_simulated_actor_types": simulated_types,
                "game_target_trajectory": actor_trajectories[first_actor_id],
                "game_target_actor_id": first_actor_id,
                "game_target_cost": actor_costs[first_actor_id],
                "game_target_type_trajectories": actor_type_trajectories[first_actor_id],
                "game_target_type_belief": dict(self.beliefs[first_actor_id].probabilities),
                "game_best_parameters": optimization.best_parameters,
                "game_metadata": optimization.metadata,
                "parameter_dim": int(self.ego_trajectory_model.parameter_dim(ego_problem)),
            },
        )

    def reset(self) -> None:
        super().reset()
        self._last_target_parameters_by_actor_type.clear()
        self._last_type_predictions_by_actor.clear()
        self._last_observations_by_actor.clear()
        self._last_optimization_problem = None
        for actor_id, belief in self.beliefs.items():
            belief.update({profile.name: profile.prior_probability for profile in self.type_profiles})
            self.belief_filters[actor_id].reset()

    def _controlled_actor_ids(self, problem: PlanningProblem) -> tuple[str, ...]:
        configured = tuple(str(value) for value in self.config.target_actor_ids if str(value))
        metadata_ids = tuple(str(value) for value in dict(problem.metadata or {}).get("game_actor_ids", ()))
        values = configured or metadata_ids or (str(self.config.target_actor_id),)
        return tuple(dict.fromkeys(values))

    def _ensure_actor_belief(self, actor_id: str) -> None:
        if actor_id not in self.beliefs:
            self.beliefs[actor_id] = ActorTypeBelief(
                actor_id=actor_id,
                probabilities={profile.name: profile.prior_probability for profile in self.type_profiles},
            )
            self.belief_filters[actor_id] = BayesianTypeFilter(self._filter_config)

    def _type_hypotheses(self, actor_ids, type_names, probabilities_by_actor):
        count = len(type_names) ** len(actor_ids)
        if count > int(self.config.max_exact_hypotheses):
            raise ValueError(
                f"Exact Bayesian hypothesis count {count} exceeds max_exact_hypotheses="
                f"{self.config.max_exact_hypotheses}."
            )
        hypotheses = []
        for choices in product(type_names, repeat=len(actor_ids)):
            types_by_actor = dict(zip(actor_ids, choices))
            probability = float(np.prod([
                probabilities_by_actor[actor_id][type_names.index(types_by_actor[actor_id])]
                for actor_id in actor_ids
            ]))
            hypotheses.append((types_by_actor, probability))
        total = max(float(sum(item[1] for item in hypotheses)), 1e-12)
        return tuple((types, probability / total) for types, probability in hypotheses)

    def _target_player_for_type(self, actor_id: str, problem: PlanningProblem, profile: ActorTypeProfile) -> GamePlayer:
        shared_min_offset = min(
            float(self.config.target_min_terminal_s_offset),
            *(float(item.min_terminal_s_offset) for item in self.type_profiles),
        )
        shared_max_offset = max(
            float(self.config.target_max_terminal_s_offset),
            *(float(item.max_terminal_s_offset) for item in self.type_profiles),
        )
        model = VehicleLongitudinalTrajectoryModel(
            VehicleLongitudinalTrajectoryConfig(
                min_terminal_speed=max(float(self.config.target_min_terminal_speed), float(profile.min_terminal_speed)),
                max_terminal_speed=min(float(self.config.target_max_terminal_speed), float(profile.max_terminal_speed)),
                min_terminal_s_offset=shared_min_offset,
                max_terminal_s_offset=shared_max_offset,
            )
        )
        low, high = model.bounds(problem)
        horizon = max(float(problem.horizon), 1e-3)
        nominal_s_end = float(problem.ego.s) + float(problem.ego.s_v) * horizon
        desired_speed = float(np.clip(dict(problem.metadata or {})["target_speed"], low[1], high[1]))
        middle_offset = 0.5 * (float(profile.min_terminal_s_offset) + float(profile.max_terminal_s_offset))
        seeds = []
        previous = self._last_target_parameters_by_actor_type.get((actor_id, profile.name))
        if previous is not None:
            seeds.append(np.clip(previous, low, high))
        seeds.extend(
            [
                np.array([nominal_s_end + middle_offset, desired_speed], dtype=float),
                model.reference_parameters(problem),
                np.array(
                    [
                        nominal_s_end + float(profile.min_terminal_s_offset),
                        max(float(low[1]), 0.8 * desired_speed),
                    ],
                    dtype=float,
                ),
                np.array(
                    [
                        nominal_s_end + float(profile.max_terminal_s_offset),
                        min(float(high[1]), 1.1 * desired_speed),
                    ],
                    dtype=float,
                ),
            ]
        )
        return GamePlayer(
            name=self._target_player_name(actor_id, profile.name),
            role=f"interaction_actor:{actor_id}:{profile.name}",
            problem=problem,
            trajectory_model=model,
            lower_bound=low,
            upper_bound=high,
            initial_population=self._dedupe_rows(
                np.clip(np.asarray(seeds, dtype=float), low[None, :], high[None, :])
            )[: max(int(self.config.max_initial_anchors), 1)],
        )

    def _target_problem_for_type(self, problem, actor, exogenous_actors, profile):
        base = self._target_problem(problem, actor, exogenous_actors)
        metadata = dict(base.metadata or {})
        desired_speed = float(np.clip(
            max(float(base.ego.s_v), 0.0) * float(profile.desired_speed_scale),
            float(profile.min_terminal_speed),
            min(float(profile.max_terminal_speed), float(self.config.target_max_terminal_speed)),
        ))
        metadata.update({"target_speed": desired_speed, "prior_speed": desired_speed, "actor_type_hypothesis": profile.name})
        return PlanningProblem(
            ego=base.ego, ref_path=base.ref_path, road_boundary=base.road_boundary,
            horizon=base.horizon, dt=base.dt, actors=base.actors, metadata=metadata
        )

    def _target_cost_for_type(self, profile: ActorTypeProfile) -> VehicleGameCost:
        return VehicleGameCost(replace(
            self.target_cost_function.config,
            min_follow_gap=float(profile.min_follow_gap),
            time_headway=float(profile.time_headway),
            headway_comfort=float(profile.headway_comfort),
            speed_tracking_comfort=float(profile.speed_tracking_comfort),
            prior_speed_comfort=float(profile.prior_speed_comfort),
            max_speed=min(float(profile.max_terminal_speed), float(self.config.target_max_terminal_speed)),
        ))

    def _evaluate_ego_multi_game_batch(self, ego_batch, target_opponents, problem):
        cost = self.ego_cost_function
        blocked_ranges = cost._blocked_ranges(problem)
        terms = cost._lattice_batch_running_terms(ego_batch, problem, blocked_ranges)
        for target_batch, target_problem in target_opponents:
            collision = self._pairwise_game_collision_batch(
                ego_batch, problem, target_batch, target_problem,
                float(cost.config.ego_front), float(cost.config.ego_rear), float(cost.config.ego_width),
                float(self.target_cost_function.config.vehicle_front) + float(cost.config.planning_obstacle_s_buffer),
                float(self.target_cost_function.config.vehicle_rear) + float(cost.config.planning_obstacle_s_buffer),
                float(self.target_cost_function.config.vehicle_width) + 2.0 * float(cost.config.planning_obstacle_l_buffer),
                decay_s=float(cost.config.collision_decay_s),
            )
            terms["collision_running"] = np.maximum(terms["collision_running"], collision["collision_running"])
            terms["collision_overlap"] = np.maximum(terms["collision_overlap"], collision["collision_overlap"])
        scores = cost._lattice_batch_hierarchy_scores(terms)
        certificate_score = np.zeros_like(scores["efficiency_score"])
        if bool(cost.config.trajectory_certificate_enabled):
            certificate_score = cost._lattice_batch_certificate_score(ego_batch, problem, blocked_ranges)
        total = (
            1.0e9 * self._saturate_array(scores["collision_cost"], cost.config.collision_score_scale)
            + 1.0e8 * self._saturate_array(scores["road_cost"], cost.config.road_score_scale)
            + 1.0e7 * scores["kappa_hard_score"]
            + 1.0e6 * scores["dkappa_hard_score"]
            + 1.0e5 * scores["lateral_accel_hard_score"]
            + 1.0e4 * scores["lateral_jerk_hard_score"]
            + 1.0e3 * scores["efficiency_score"]
            + (1.0e2 * certificate_score if bool(cost.config.trajectory_certificate_enabled) else 0.0)
            + 1.0e2 * scores["reference_score"]
            + 1.0e1 * scores["comfort_score"]
        )
        return self._finite_array(total)

    def _evaluate_target_multi_game_batch(self, target_batch, opponents, problem, target_cost):
        cfg = target_cost.config
        s = np.asarray(target_batch["s"], dtype=float)
        l = np.asarray(target_batch["l"], dtype=float)
        t = np.asarray(target_batch["t"], dtype=float)
        blocked_ranges = target_cost._blocked_ranges(problem, opponents=())
        collision = self._agent_blocked_collision_batch(s, l, t, blocked_ranges, cfg.vehicle_front, cfg.vehicle_rear, cfg.vehicle_width)
        collision_running = collision["collision_running"]
        collision_overlap = collision["collision_overlap"]
        headway_running = np.zeros_like(s)
        for opponent_batch, opponent_problem in opponents:
            pairwise_collision = self._pairwise_game_collision_batch(
                target_batch, problem, opponent_batch, opponent_problem,
                cfg.vehicle_front, cfg.vehicle_rear, cfg.vehicle_width,
                cfg.vehicle_front + cfg.obstacle_s_buffer, cfg.vehicle_rear + cfg.obstacle_s_buffer,
                cfg.vehicle_width + 2.0 * cfg.obstacle_l_buffer,
                decay_s=1.0e9,
            )
            collision_running = np.maximum(collision_running, pairwise_collision["collision_running"])
            collision_overlap = np.maximum(collision_overlap, pairwise_collision["collision_overlap"])
            if problem.ref_path is opponent_problem.ref_path:
                headway_running = np.maximum(headway_running, self._target_headway_terms_batch(
                    target_s=s, target_l=l, target_s_v=np.asarray(target_batch["s_v"], dtype=float),
                    ego_s=np.asarray(opponent_batch["s"], dtype=float), ego_l=np.asarray(opponent_batch["l"], dtype=float),
                    target_cost_function=target_cost,
                ))
        road_running, _ = self._target_road_terms_batch(l, problem, target_cost)
        speed_running, _ = self._target_speed_terms_batch(np.asarray(target_batch["s_v"], dtype=float), problem, target_cost)
        comfort_running = self._target_comfort_terms_batch(np.asarray(target_batch["s_a"], dtype=float), t, target_cost)
        lane_running = self._target_lane_terms_batch(l, problem, target_cost)
        prior_running = self._target_prior_terms_batch(np.asarray(target_batch["s_v"], dtype=float), problem, target_cost)
        total = (
            1.0e9 * self._saturate_array(self._batch_topk_max(collision_running, 0.15), cfg.collision_score_scale)
            + 1.0e8 * self._saturate_array(self._batch_topk_max(road_running, 0.15), cfg.road_score_scale)
            + 1.0e4 * self._saturate_array(self._batch_topk_max(headway_running, 0.25), cfg.headway_score_scale)
            + 1.0e3 * self._saturate_array(np.mean(speed_running, axis=1), cfg.speed_score_scale)
            + 1.0e2 * self._saturate_array(np.mean(prior_running, axis=1), cfg.prior_score_scale)
            + 1.0e2 * self._saturate_array(np.mean(lane_running, axis=1), cfg.lane_score_scale)
            + 1.0e1 * self._saturate_array(np.mean(comfort_running, axis=1), cfg.comfort_score_scale)
        )
        return self._finite_array(total)

    def _pairwise_game_collision_batch(
        self,
        agent,
        agent_problem,
        other,
        other_problem,
        agent_front,
        agent_rear,
        agent_width,
        other_front,
        other_rear,
        other_width,
        decay_s,
    ):
        if agent_problem.ref_path is other_problem.ref_path:
            return self._pairwise_collision_batch(
                agent_s=agent["s"],
                agent_l=agent["l"],
                other_s=other["s"],
                other_l=other["l"],
                agent_front=agent_front,
                agent_rear=agent_rear,
                agent_width=agent_width,
                other_front=other_front,
                other_rear=other_rear,
                other_width=other_width,
                decay_s=decay_s,
            )
        return self._pairwise_xy_collision_batch(
            self._with_batch_xy(agent, agent_problem),
            self._with_batch_xy(other, other_problem),
            agent_front,
            agent_rear,
            agent_width,
            other_front,
            other_rear,
            other_width,
        )

    @staticmethod
    def _pairwise_xy_collision_batch(agent, other, agent_front, agent_rear, agent_width, other_front, other_rear, other_width):
        ax = np.asarray(agent["x"], dtype=float)
        ay = np.asarray(agent["y"], dtype=float)
        ox = np.asarray(other["x"], dtype=float)
        oy = np.asarray(other["y"], dtype=float)
        ayaw = np.zeros_like(ax) if agent.get("yaw") is None else np.asarray(agent["yaw"], dtype=float)
        oyaw = np.zeros_like(ox) if other.get("yaw") is None else np.asarray(other["yaw"], dtype=float)
        n, m = min(ax.shape[0], ox.shape[0]), min(ax.shape[1], ox.shape[1])
        ax, ay, ayaw, ox, oy, oyaw = (value[:n, :m] for value in (ax, ay, ayaw, ox, oy, oyaw))
        dx, dy = ox - ax, oy - ay
        afx, afy = np.cos(ayaw), np.sin(ayaw)
        alx, aly = -afy, afx
        ofx, ofy = np.cos(oyaw), np.sin(oyaw)
        olx, oly = -ofy, ofx
        ahl, ahw = 0.5 * (float(agent_front) + float(agent_rear)), 0.5 * float(agent_width)
        ohl, ohw = 0.5 * (float(other_front) + float(other_rear)), 0.5 * float(other_width)
        axes = ((afx, afy), (alx, aly), (ofx, ofy), (olx, oly))
        margins = []
        for ux, uy in axes:
            separation = np.abs(dx * ux + dy * uy)
            agent_radius = ahl * np.abs(afx * ux + afy * uy) + ahw * np.abs(alx * ux + aly * uy)
            other_radius = ohl * np.abs(ofx * ux + ofy * uy) + ohw * np.abs(olx * ux + oly * uy)
            margins.append(agent_radius + other_radius - separation)
        penetration = np.minimum.reduce(margins)
        overlap = penetration >= 0.0
        running = np.where(overlap, 1.0 + np.minimum(np.maximum(penetration, 0.0), 3.0), 0.0)
        return {"collision_running": running, "collision_overlap": overlap.astype(float)}

    @staticmethod
    def _with_batch_xy(batch, problem):
        if batch.get("x") is not None and batch.get("y") is not None:
            return batch
        values = dict(batch)
        s = np.asarray(values["s"], dtype=float)
        l = np.asarray(values["l"], dtype=float)
        ref_path = problem.ref_path
        if not hasattr(ref_path, "calc_position") or not hasattr(ref_path, "calc_yaw"):
            raise TypeError("Cross-route XY collision requires a geometric reference path.")
        if not hasattr(ref_path, "s") or not np.asarray(ref_path.s, dtype=float).size:
            raise ValueError("Cross-route XY collision requires non-empty ref_path.s.")
        route_end = float(np.asarray(ref_path.s, dtype=float)[-1])
        flat_s = np.clip(s.reshape(-1), 0.0, route_end)
        flat_l = l.reshape(-1)
        pose_rows = []
        for s_value in flat_s:
            xy = ref_path.calc_position(float(s_value))
            if xy is None or xy[0] is None or xy[1] is None:
                raise ValueError(f"Reference path returned no position at s={float(s_value):.3f}.")
            pose_rows.append((float(xy[0]), float(xy[1]), float(ref_path.calc_yaw(float(s_value)))))
        poses = np.asarray(pose_rows, dtype=float)
        values["x"] = (poses[:, 0] + flat_l * np.cos(poses[:, 2] + np.pi / 2.0)).reshape(s.shape)
        values["y"] = (poses[:, 1] + flat_l * np.sin(poses[:, 2] + np.pi / 2.0)).reshape(s.shape)
        l_v = np.asarray(values.get("l_v", np.zeros_like(s)), dtype=float)
        s_v = np.asarray(values.get("s_v", np.zeros_like(s)), dtype=float)
        values["yaw"] = poses[:, 2].reshape(s.shape) + np.arctan2(l_v, np.maximum(np.abs(s_v), 1e-6))
        return values

    def _continuation_features_from_batch(self, trajectory_batch):
        t = np.asarray(trajectory_batch["t"], dtype=float).reshape(-1)
        index = 0 if t.size == 0 else int(np.argmin(np.abs(t - float(self.continuation_evaluator.config.observation_time))))
        return np.column_stack([
            np.asarray(trajectory_batch["s"], dtype=float)[:, index],
            np.asarray(trajectory_batch["s_v"], dtype=float)[:, index],
            np.asarray(trajectory_batch["s_a"], dtype=float)[:, index],
        ])

    def _update_type_belief(self, actor_id: str, actor) -> None:
        predictions_by_type = self._last_type_predictions_by_actor.get(actor_id)
        previous = self._last_observations_by_actor.get(actor_id)
        if not predictions_by_type or previous is None:
            return
        observation = self._observation_from_actor(actor)
        elapsed = max(float(observation.time) - float(previous.time), 0.0)
        predictions = {}
        for type_name, trajectory in predictions_by_type.items():
            t = np.asarray(trajectory.t, dtype=float)
            s = np.asarray(trajectory.s, dtype=float)
            s_v = np.asarray(trajectory.s_v, dtype=float)
            s_a = np.asarray(trajectory.s_a, dtype=float)
            predictions[type_name] = LongitudinalTypePrediction(
                s=float(np.interp(elapsed, t, s)),
                speed=float(np.interp(elapsed, t, s_v)),
                acceleration=float(np.interp(elapsed, t, s_a)),
                start_s=float(s[0]),
                start_speed=float(s_v[0]),
            )
        self.belief_filters[actor_id].update(self.beliefs[actor_id], observation, predictions, previous)

    def _simulated_target_type(self, actor_id: str, type_names: tuple[str, ...]) -> str:
        configured = str(self.config.simulated_actor_types.get(actor_id, self.config.simulated_target_type))
        if configured not in type_names:
            raise ValueError(
                f"Invalid simulated actor type {configured!r} for actor {actor_id!r}; "
                f"expected one of {type_names}."
            )
        return configured

    @staticmethod
    def _target_player_name(actor_id: str, type_name: str) -> str:
        return f"{actor_id}:{type_name}"

    @staticmethod
    def _actor_time(actor) -> float:
        times = np.asarray(actor.times, dtype=float).reshape(-1)
        if not times.size:
            raise ValueError(f"Actor {actor.actor_id!r} has no prediction timestamps.")
        return float(times[0])

    @classmethod
    def _observation_from_actor(cls, actor) -> LongitudinalObservation:
        metadata = dict(actor.metadata or {})
        required_state = ("s", "s_v", "s_a")
        missing = tuple(key for key in required_state if key not in metadata)
        if missing:
            raise ValueError(f"Actor {actor.actor_id!r} is missing observation metadata: {missing}")
        return LongitudinalObservation(
            time=cls._actor_time(actor),
            s=float(metadata["s"]),
            speed=float(metadata["s_v"]),
            acceleration=float(metadata["s_a"]),
        )

    @staticmethod
    def _cost_result_from_total(total: float) -> CostResult:
        feasible = bool(np.isfinite(total) and total < 1.0e8)
        return CostResult(
            total=float(total),
            breakdown=CostBreakdown(terms={"collision_flag": float(not feasible)}, hard_violation=not feasible),
            feasible=feasible,
            metadata={"cost": "multi_actor_bayesian_game_batch"},
        )

    @staticmethod
    def _finite_array(values):
        result = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Bayesian game batch cost produced non-finite values.")
        return result
