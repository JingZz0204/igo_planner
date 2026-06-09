from __future__ import annotations

from dataclasses import dataclass, replace
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
from spatiotemporal_joint_planner.contingency import (
    BeliefContinuationConfig,
    BeliefContinuationEvaluator,
    RiskAggregator,
    RiskAggregatorConfig,
)
from spatiotemporal_joint_planner.cost import CostFunction
from spatiotemporal_joint_planner.game.base import GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.bayesian_igo_optimizer import (
    BayesianBatchEvaluation,
    BayesianGameOptimizationProblem,
    BayesianIGOConfig,
    BayesianIGOOptimizer,
)
from spatiotemporal_joint_planner.game.cost import VehicleGameCost
from spatiotemporal_joint_planner.game.game_parametric_planner import GameParametricPlanner, GameParametricPlannerConfig
from spatiotemporal_joint_planner.game.trajectory_models import (
    VehicleLongitudinalTrajectoryConfig,
    VehicleLongitudinalTrajectoryModel,
)
from spatiotemporal_joint_planner.planner.warm_start import WarmStartGenerator
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class BayesianGameParametricPlannerConfig(GameParametricPlannerConfig):
    simulated_target_type: str = "normal"
    min_type_probability_for_feasibility: float = 0.05


class BayesianGameParametricPlanner(GameParametricPlanner):
    """Implicit-contingency game planner with a shared ego action across actor types."""

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
        if not self.type_profiles:
            raise ValueError("Bayesian game planner requires at least one target actor type.")
        names = [profile.name for profile in self.type_profiles]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate target actor type names: {names}")
        self.risk_aggregator = risk_aggregator
        self.continuation_evaluator = continuation_evaluator or BeliefContinuationEvaluator(
            self.risk_aggregator,
            BeliefContinuationConfig(),
        )
        self.belief_filter = belief_filter or BayesianTypeFilter(BayesianTypeFilterConfig())
        self.belief = ActorTypeBelief(
            actor_id=self.config.target_actor_id,
            probabilities={profile.name: profile.prior_probability for profile in self.type_profiles},
        )
        self._last_target_parameters_by_type: dict[str, np.ndarray] = {}
        self._last_type_predictions: dict[str, Trajectory] = {}
        self._last_observation: Optional[LongitudinalObservation] = None

    @property
    def name(self) -> str:
        return "bayesian_game_parametric_planner"

    def plan(self, problem: PlanningProblem) -> PlannerResult:
        target_actor = self._target_actor(problem)
        if target_actor is None:
            return PlannerResult(
                trajectory=None,
                cost=None,
                status="missing_game_target_actor",
                metadata={"planner": self.name, "target_actor_id": self.config.target_actor_id},
            )

        self._update_type_belief(target_actor)
        type_names = tuple(profile.name for profile in self.type_profiles)
        probabilities = self.belief.vector(type_names)
        exogenous_actors = tuple(actor for actor in problem.actors if actor.actor_id != self.config.target_actor_id)
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
        target_problems = {
            profile.name: self._target_problem_for_type(problem, target_actor, exogenous_actors, profile)
            for profile in self.type_profiles
        }
        target_players = {
            profile.name: self._target_player_for_type(target_problems[profile.name], profile)
            for profile in self.type_profiles
        }
        target_costs = {
            profile.name: self._target_cost_for_type(profile)
            for profile in self.type_profiles
        }

        def decode_joint(parameters_by_player: Mapping[str, np.ndarray]) -> JointTrajectory:
            trajectories: dict[str, Trajectory] = {
                self.config.ego_player_name: self.ego_trajectory_model.decode(
                    np.asarray(parameters_by_player[self.config.ego_player_name], dtype=float),
                    ego_problem,
                )
            }
            for type_name in type_names:
                player = target_players[type_name]
                trajectories[player.name] = player.trajectory_model.decode(
                    np.asarray(parameters_by_player[player.name], dtype=float),
                    target_problems[type_name],
                )
            return JointTrajectory(trajectories=trajectories)

        def evaluate_joint(joint: JointTrajectory) -> dict[str, CostResult]:
            ego_trajectory = joint.trajectories[self.config.ego_player_name]
            ego_costs_by_type: dict[str, CostResult] = {}
            results: dict[str, CostResult] = {}
            target_features = []
            for type_name in type_names:
                player = target_players[type_name]
                target_trajectory = joint.trajectories[player.name]
                optimized_target_actor = self._actor_from_trajectory(target_actor, target_trajectory)
                ego_eval_problem = PlanningProblem(
                    ego=problem.ego,
                    ref_path=problem.ref_path,
                    road_boundary=problem.road_boundary,
                    horizon=problem.horizon,
                    dt=problem.dt,
                    actors=tuple(exogenous_actors) + (optimized_target_actor,),
                    metadata=dict(problem.metadata or {}),
                )
                ego_costs_by_type[type_name] = self.ego_cost_function.evaluate(ego_trajectory, ego_eval_problem)
                results[player.name] = target_costs[type_name].evaluate(
                    target_trajectory,
                    target_problems[type_name],
                    opponent_trajectories=(ego_trajectory,),
                )
                target_features.append(self._continuation_features_from_trajectory(target_trajectory))
            continuation = self.continuation_evaluator.evaluate_batch(
                np.asarray([[ego_costs_by_type[name].total] for name in type_names], dtype=float),
                np.asarray(target_features, dtype=float)[:, None, :],
                probabilities,
            )
            results[self.config.ego_player_name] = self._aggregate_ego_costs(
                ego_costs_by_type,
                type_names,
                probabilities,
                continuation_diagnostics={
                    name: float(np.asarray(value, dtype=float).reshape(-1)[0])
                    for name, value in continuation.items()
                },
            )
            return results

        def evaluate_batch(
            ego_parameters: np.ndarray,
            target_parameters_by_type: Mapping[str, np.ndarray],
        ) -> BayesianBatchEvaluation:
            if not hasattr(self.ego_trajectory_model, "decode_batch_arrays"):
                raise ValueError("Bayesian game batch evaluation requires a batch-decodable ego trajectory model.")
            ego_batch = self.ego_trajectory_model.decode_batch_arrays(np.asarray(ego_parameters, dtype=float), ego_problem)
            ego_values_by_type: dict[str, np.ndarray] = {}
            target_values_by_type: dict[str, np.ndarray] = {}
            target_features = []
            for type_name in type_names:
                player = target_players[type_name]
                if not hasattr(player.trajectory_model, "decode_batch_arrays"):
                    raise ValueError("Bayesian game batch evaluation requires batch-decodable target models.")
                target_parameters = np.asarray(target_parameters_by_type[type_name], dtype=float)
                target_batch = player.trajectory_model.decode_batch_arrays(
                    target_parameters,
                    target_problems[type_name],
                )
                ego_values_by_type[type_name] = self._evaluate_ego_lattice_game_batch(
                    ego_batch,
                    target_batch,
                    ego_problem,
                    target_cost_function=target_costs[type_name],
                )
                target_values_by_type[type_name] = self._evaluate_target_game_batch(
                    target_batch,
                    ego_batch,
                    target_problems[type_name],
                    target_cost_function=target_costs[type_name],
                )
                target_features.append(self._continuation_features_from_batch(target_batch))
            type_cost_matrix = np.asarray([ego_values_by_type[name] for name in type_names], dtype=float)
            continuation = self.continuation_evaluator.evaluate_batch(
                type_cost_matrix,
                np.asarray(target_features, dtype=float),
                probabilities,
            )
            aggregate = self.risk_aggregator.aggregate_batch(type_cost_matrix, probabilities)
            aggregate = aggregate + np.asarray(continuation["implicit_contingency_cost"], dtype=float)
            return BayesianBatchEvaluation(
                aggregate_ego_values=aggregate,
                ego_values_by_type=ego_values_by_type,
                target_values_by_type=target_values_by_type,
                diagnostics=continuation,
            )

        def evaluate_type_batch(
            type_name: str,
            ego_parameters: np.ndarray,
            target_parameters: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            player = target_players[type_name]
            ego_batch = self.ego_trajectory_model.decode_batch_arrays(np.asarray(ego_parameters, dtype=float), ego_problem)
            target_batch = player.trajectory_model.decode_batch_arrays(
                np.asarray(target_parameters, dtype=float),
                target_problems[type_name],
            )
            return (
                self._evaluate_ego_lattice_game_batch(
                    ego_batch,
                    target_batch,
                    ego_problem,
                    target_cost_function=target_costs[type_name],
                ),
                self._evaluate_target_game_batch(
                    target_batch,
                    ego_batch,
                    target_problems[type_name],
                    target_cost_function=target_costs[type_name],
                ),
            )

        game_problem = BayesianGameOptimizationProblem(
            ego_player=ego_player,
            target_players_by_type=target_players,
            type_probabilities={name: float(probabilities[idx]) for idx, name in enumerate(type_names)},
            evaluate_batch=evaluate_batch,
            evaluate_type_batch=evaluate_type_batch,
            decode_joint=decode_joint,
            evaluate_joint=evaluate_joint,
            metadata={
                "planner": self.name,
                "target_actor_id": self.config.target_actor_id,
                "scenario": problem.metadata.get("scenario") if problem.metadata else None,
                "target_type_belief": dict(self.belief.probabilities),
            },
        )
        optimization = self.optimizer.optimize(game_problem)
        ego_trajectory = optimization.best_joint_trajectory.trajectories.get(self.config.ego_player_name)
        ego_cost = optimization.player_costs.get(self.config.ego_player_name)
        target_trajectories = {
            type_name: optimization.best_joint_trajectory.trajectories.get(target_players[type_name].name)
            for type_name in type_names
        }
        target_type_costs = {
            type_name: optimization.player_costs.get(target_players[type_name].name)
            for type_name in type_names
        }
        simulated_type = self._simulated_target_type(type_names)
        target_trajectory = target_trajectories.get(simulated_type)
        target_cost = target_type_costs.get(simulated_type)

        status = "success"
        if ego_trajectory is None or ego_cost is None:
            status = "no_valid_trajectory"
        elif not ego_cost.feasible:
            status = "infeasible_best"

        if ego_trajectory is not None:
            self._last_ego_parameters = np.asarray(optimization.best_parameters[self.config.ego_player_name], dtype=float)
        for type_name in type_names:
            player_name = target_players[type_name].name
            if player_name in optimization.best_parameters:
                self._last_target_parameters_by_type[type_name] = np.asarray(
                    optimization.best_parameters[player_name],
                    dtype=float,
                )
        self._last_type_predictions = {
            name: trajectory for name, trajectory in target_trajectories.items() if trajectory is not None
        }
        self._last_observation = self._observation_from_actor(target_actor)

        game_metadata = dict(optimization.metadata)
        game_metadata.update(
            {
                "target_type_belief": dict(self.belief.probabilities),
                "simulated_target_type": simulated_type,
                "risk_expected_weight": float(self.risk_aggregator.config.expected_weight),
                "risk_cvar_weight": float(self.risk_aggregator.config.cvar_weight),
                "risk_cvar_alpha": float(self.risk_aggregator.config.cvar_alpha),
                "implicit_contingency_enabled": bool(self.continuation_evaluator.config.enabled),
                "implicit_contingency_observation_time": float(
                    self.continuation_evaluator.config.observation_time
                ),
                "implicit_contingency_cost_weight": float(self.continuation_evaluator.config.cost_weight),
            }
        )
        return PlannerResult(
            trajectory=ego_trajectory,
            cost=ego_cost,
            status=status,
            candidates=(),
            optimization=None,
            metadata={
                "planner": self.name,
                "trajectory_model": self.ego_trajectory_model.name,
                "optimizer": self.optimizer.name,
                "game_optimizer": optimization,
                "game_target_trajectory": target_trajectory,
                "game_target_actor_id": self.config.target_actor_id,
                "game_target_cost": target_cost,
                "game_target_type_trajectories": target_trajectories,
                "game_target_type_costs": target_type_costs,
                "game_target_type_belief": dict(self.belief.probabilities),
                "game_simulated_target_type": simulated_type,
                "game_best_parameters": optimization.best_parameters,
                "game_metadata": game_metadata,
                "parameter_dim": int(self.ego_trajectory_model.parameter_dim(ego_problem)),
            },
        )

    def reset(self) -> None:
        super().reset()
        self._last_target_parameters_by_type.clear()
        self._last_type_predictions.clear()
        self._last_observation = None
        self.belief_filter.reset()
        self.belief.update({profile.name: profile.prior_probability for profile in self.type_profiles})

    def _target_player_for_type(self, problem: PlanningProblem, profile: ActorTypeProfile) -> GamePlayer:
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
        desired_speed = float(dict(problem.metadata or {}).get("target_speed", problem.ego.s_v))
        speed_values = np.linspace(
            max(float(low[1]), 0.65 * desired_speed),
            min(float(high[1]), 1.20 * max(desired_speed, 1e-3)),
            num=5,
            dtype=float,
        )
        offset_values = np.linspace(
            float(profile.min_terminal_s_offset),
            float(profile.max_terminal_s_offset),
            num=6,
            dtype=float,
        )
        seeds = []
        previous = self._last_target_parameters_by_type.get(profile.name)
        if previous is not None:
            seeds.append(np.clip(previous, low, high))
        seeds.append(model.reference_parameters(problem))
        for offset in offset_values:
            for speed in speed_values:
                seeds.append(np.array([nominal_s_end + float(offset), float(speed)], dtype=float))
        return GamePlayer(
            name=self._target_player_name(profile.name),
            role=f"target_rear:{profile.name}",
            problem=problem,
            trajectory_model=model,
            lower_bound=low,
            upper_bound=high,
            initial_population=self._dedupe_rows(np.clip(np.asarray(seeds, dtype=float), low[None, :], high[None, :])),
        )

    def _target_problem_for_type(
        self,
        problem: PlanningProblem,
        actor,
        exogenous_actors: Sequence,
        profile: ActorTypeProfile,
    ) -> PlanningProblem:
        base = self._target_problem(problem, actor, exogenous_actors)
        metadata = dict(base.metadata or {})
        desired_speed = float(np.clip(
            max(float(base.ego.s_v), 0.0) * float(profile.desired_speed_scale),
            float(profile.min_terminal_speed),
            min(float(profile.max_terminal_speed), float(self.config.target_max_terminal_speed)),
        ))
        metadata.update(
            {
                "target_speed": desired_speed,
                "prior_speed": desired_speed,
                "actor_type_hypothesis": profile.name,
            }
        )
        return PlanningProblem(
            ego=base.ego,
            ref_path=base.ref_path,
            road_boundary=base.road_boundary,
            horizon=base.horizon,
            dt=base.dt,
            actors=base.actors,
            metadata=metadata,
        )

    def _target_cost_for_type(self, profile: ActorTypeProfile) -> VehicleGameCost:
        config = replace(
            self.target_cost_function.config,
            min_follow_gap=float(profile.min_follow_gap),
            time_headway=float(profile.time_headway),
            headway_comfort=float(profile.headway_comfort),
            speed_tracking_comfort=float(profile.speed_tracking_comfort),
            prior_speed_comfort=float(profile.prior_speed_comfort),
            max_speed=min(float(profile.max_terminal_speed), float(self.config.target_max_terminal_speed)),
        )
        return VehicleGameCost(config)

    def _aggregate_ego_costs(
        self,
        costs_by_type: Mapping[str, CostResult],
        type_names: tuple[str, ...],
        probabilities: np.ndarray,
        continuation_diagnostics: Optional[Mapping[str, float]] = None,
    ) -> CostResult:
        costs = np.asarray([costs_by_type[name].total for name in type_names], dtype=float)
        diagnostics = self.risk_aggregator.diagnostics(costs, probabilities)
        terms: dict[str, float] = dict(diagnostics)
        continuation_diagnostics = dict(continuation_diagnostics or {})
        terms.update({name: float(value) for name, value in continuation_diagnostics.items()})
        all_term_names = set().union(*(costs_by_type[name].breakdown.terms.keys() for name in type_names))
        for term_name in all_term_names:
            values = np.asarray(
                [float(costs_by_type[name].breakdown.terms.get(term_name, 0.0)) for name in type_names],
                dtype=float,
            )
            if term_name.endswith("_flag"):
                terms[term_name] = float(np.max(values))
            else:
                terms[term_name] = float(np.sum(probabilities * values))
        for index, type_name in enumerate(type_names):
            terms[f"contingency_probability_{type_name}"] = float(probabilities[index])
            terms[f"contingency_cost_{type_name}"] = float(costs[index])

        considered = probabilities >= float(self.config.min_type_probability_for_feasibility)
        feasible_values = np.asarray([costs_by_type[name].feasible for name in type_names], dtype=bool)
        feasible = bool(np.all(feasible_values[considered])) if np.any(considered) else bool(np.all(feasible_values))
        total = float(diagnostics["contingency_aggregate_cost"]) + float(
            continuation_diagnostics.get("implicit_contingency_cost", 0.0)
        )
        return CostResult(
            total=total,
            breakdown=CostBreakdown(terms=terms, hard_violation=not feasible),
            feasible=feasible,
            metadata={
                "cost": "bayesian_implicit_contingency",
                "type_probabilities": {name: float(probabilities[idx]) for idx, name in enumerate(type_names)},
                "type_costs": {name: costs_by_type[name] for name in type_names},
            },
        )

    def _continuation_features_from_batch(self, trajectory_batch: Mapping[str, object]) -> np.ndarray:
        t = np.asarray(trajectory_batch["t"], dtype=float).reshape(-1)
        if t.size == 0:
            batch_size = int(np.asarray(trajectory_batch["s"], dtype=float).shape[0])
            return np.zeros((batch_size, 3), dtype=float)
        index = int(np.argmin(np.abs(t - float(self.continuation_evaluator.config.observation_time))))
        return np.column_stack(
            [
                np.asarray(trajectory_batch["s"], dtype=float)[:, index],
                np.asarray(trajectory_batch["s_v"], dtype=float)[:, index],
                np.asarray(trajectory_batch["s_a"], dtype=float)[:, index],
            ]
        )

    def _continuation_features_from_trajectory(self, trajectory: Trajectory) -> np.ndarray:
        t = np.asarray(trajectory.t, dtype=float).reshape(-1)
        if t.size == 0:
            return np.zeros((3,), dtype=float)
        index = int(np.argmin(np.abs(t - float(self.continuation_evaluator.config.observation_time))))
        s = np.asarray(trajectory.s, dtype=float).reshape(-1)
        s_v = np.zeros_like(s) if trajectory.s_v is None else np.asarray(trajectory.s_v, dtype=float).reshape(-1)
        s_a = np.zeros_like(s) if trajectory.s_a is None else np.asarray(trajectory.s_a, dtype=float).reshape(-1)
        index = min(index, s.size - 1, s_v.size - 1, s_a.size - 1)
        return np.asarray([s[index], s_v[index], s_a[index]], dtype=float)

    def _update_type_belief(self, actor) -> None:
        if not self._last_type_predictions or self._last_observation is None:
            return
        observation = self._observation_from_actor(actor)
        elapsed = max(float(observation.time) - float(self._last_observation.time), 0.0)
        predictions = {}
        for type_name, trajectory in self._last_type_predictions.items():
            t = np.asarray(trajectory.t, dtype=float)
            s = np.asarray(trajectory.s, dtype=float)
            if t.size == 0 or s.size == 0:
                continue
            n = min(t.size, s.size)
            predicted_s = float(np.interp(elapsed, t[:n], s[:n], left=s[0], right=s[n - 1]))
            if trajectory.s_v is None:
                predicted_speed = float(self._last_observation.speed)
                start_speed = float(self._last_observation.speed)
            else:
                s_v = np.asarray(trajectory.s_v, dtype=float)
                m = min(t.size, s_v.size)
                if m:
                    predicted_speed = float(np.interp(elapsed, t[:m], s_v[:m], left=s_v[0], right=s_v[m - 1]))
                    start_speed = float(s_v[0])
                else:
                    predicted_speed = float(self._last_observation.speed)
                    start_speed = float(self._last_observation.speed)
            if trajectory.s_a is None:
                predicted_acceleration = 0.0
            else:
                s_a = np.asarray(trajectory.s_a, dtype=float)
                m = min(t.size, s_a.size)
                predicted_acceleration = (
                    float(np.interp(elapsed, t[:m], s_a[:m], left=s_a[0], right=s_a[m - 1]))
                    if m
                    else 0.0
                )
            predictions[type_name] = LongitudinalTypePrediction(
                s=predicted_s,
                speed=predicted_speed,
                acceleration=predicted_acceleration,
                start_s=float(s[0]),
                start_speed=start_speed,
            )
        if predictions:
            self.belief_filter.update(
                self.belief,
                observation=observation,
                predictions=predictions,
                previous_observation=self._last_observation,
            )

    def _simulated_target_type(self, type_names: tuple[str, ...]) -> str:
        configured = str(self.config.simulated_target_type)
        if configured in type_names:
            return configured
        probabilities = self.belief.vector(type_names)
        return type_names[int(np.argmax(probabilities))]

    def _target_player_name(self, type_name: str) -> str:
        return f"{self.config.target_player_name}_{type_name}"

    @staticmethod
    def _actor_time(actor) -> float:
        times = np.asarray(actor.times, dtype=float).reshape(-1)
        return float(times[0]) if times.size else 0.0

    @classmethod
    def _observation_from_actor(cls, actor) -> LongitudinalObservation:
        metadata = dict(actor.metadata or {})
        return LongitudinalObservation(
            time=cls._actor_time(actor),
            s=float(metadata.get("s", 0.0)),
            speed=float(metadata.get("s_v", 0.0)),
            acceleration=float(metadata.get("s_a", 0.0)),
        )
