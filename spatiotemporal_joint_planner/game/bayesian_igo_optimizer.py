from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

from spatiotemporal_joint_planner.game.base import GameOptimizationResult, GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.igo_game_optimizer import GameIGOConfig, GameIGOOptimizer
from spatiotemporal_joint_planner.optimizer.cma_es_optimizer import denormalize_parameters, normalize_parameters


@dataclass(frozen=True)
class BayesianIGOConfig(GameIGOConfig):
    """Configuration for synchronous type-conditioned Bayesian IGO flow."""

    equilibrium_check_interval: int = 3
    equilibrium_regret_tol: float = 0.1
    material_type_probability: float = 0.05
    local_nash_samples: int = 16
    local_nash_perturbation: float = 0.05
    local_nash_seed: int = 1701


@dataclass(frozen=True)
class BayesianBatchEvaluation:
    player_values: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class BayesianPhysicalActor:
    actor_id: str
    players_by_type: Mapping[str, GamePlayer]
    type_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class BayesianGameOptimizationProblem:
    ego_player: GamePlayer
    physical_actors: Mapping[str, BayesianPhysicalActor]
    evaluate_players_batch: Callable[[Mapping[str, np.ndarray]], BayesianBatchEvaluation]
    evaluate_unilateral_batch: Callable[[str, np.ndarray, Mapping[str, np.ndarray]], np.ndarray]
    decode_joint: Callable[[Mapping[str, np.ndarray]], JointTrajectory]
    evaluate_joint: Callable[[JointTrajectory], Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def strategy_players(self) -> tuple[GamePlayer, ...]:
        return (
            self.ego_player,
            *(
                player
                for actor in self.physical_actors.values()
                for player in actor.players_by_type.values()
            ),
        )

    @property
    def material_probabilities_by_player(self) -> dict[str, float]:
        result = {self.ego_player.name: 1.0}
        for actor in self.physical_actors.values():
            type_names = tuple(actor.players_by_type)
            probabilities = BayesianIGOOptimizer._normalized_type_probabilities(actor.type_probabilities, type_names)
            result.update(
                {
                    actor.players_by_type[type_name].name: float(probabilities[index])
                    for index, type_name in enumerate(type_names)
                }
            )
        return result


class BayesianIGOOptimizer(GameIGOOptimizer):
    """Synchronous Bayesian IGO flow for any number of uncertain actors."""

    def __init__(self, config: Optional[BayesianIGOConfig] = None):
        super().__init__(config or BayesianIGOConfig())
        self.config = config or BayesianIGOConfig()

    @property
    def name(self) -> str:
        return "synchronous_bayesian_igo_flow"

    def optimize(self, problem: BayesianGameOptimizationProblem) -> GameOptimizationResult:
        ego = problem.ego_player
        players = problem.strategy_players
        if len(players) <= 1:
            raise ValueError("Bayesian IGO requires at least one physical interaction actor.")
        player_names = [player.name for player in players]
        material_probabilities = problem.material_probabilities_by_player
        bounds = {
            player.name: (np.asarray(player.lower_bound, dtype=float), np.asarray(player.upper_bound, dtype=float))
            for player in players
        }
        anchors = {
            player.name: self._clip_positions_to_bounds(
                self._as_anchor_matrix(player.initial_population),
                bounds[player.name],
            )
            for player in players
        }
        states = {
            player.name: self._initial_state(player.name, bounds[player.name], anchors[player.name])
            for player in players
        }

        player_positions = {name: np.empty((0, bounds[name][0].size), dtype=float) for name in player_names}
        player_values = {name: np.empty((0,), dtype=float) for name in player_names}
        profile_value_history: dict[str, list[float]] = {name: [] for name in player_names}
        profile_unit_history: dict[str, list[np.ndarray]] = {name: [] for name in player_names}
        joint_unit_history: list[np.ndarray] = []
        joint_cost_history: list[np.ndarray] = []
        history: list[dict] = []

        best_parameters = self._representative_profile(states, bounds, players)
        best_evaluation = self._evaluate_profile(problem, best_parameters)
        best_selection = self._profile_selection(best_evaluation, ego.name)
        stop_reason = "max_iterations"
        local_nash_diagnostics = self._local_nash_not_evaluated(player_names)
        last_flow_diagnostics: dict[str, float] = {}
        local_nash_check_count = 0
        flow_batch_evaluations = 0

        for iteration in range(max(int(self.config.n_iterations), 0)):
            unit_samples, component_ids, parameter_samples = self._sample_all(states, bounds, players)
            evaluation = self._evaluate_samples(problem, parameter_samples)
            flow_batch_evaluations += 1
            self._record_flow_batch(
                parameter_samples,
                evaluation,
                player_positions,
                player_values,
            )
            values = self._player_values(evaluation)
            states = {
                player.name: self._update_distribution(
                    states[player.name],
                    unit_samples[player.name],
                    component_ids[player.name],
                    values[player.name],
                    iteration,
                )
                for player in players
            }

            best_parameters = self._representative_profile(states, bounds, players)
            best_evaluation = self._evaluate_profile(problem, best_parameters)
            best_selection = self._profile_selection(best_evaluation, ego.name)
            profile_values = self._player_values(best_evaluation)
            for player in players:
                profile_value_history[player.name].append(float(profile_values[player.name][0]))
                profile_unit_history[player.name].append(
                    normalize_parameters(best_parameters[player.name], bounds[player.name])
                )
            joint_unit_history.append(self._joint_unit_vector(best_parameters, bounds, player_names))
            joint_cost_history.append(
                np.asarray([profile_values[name][0] for name in player_names], dtype=float)
            )

            search_diagnostics = self._early_stop_diagnostics(profile_value_history, profile_unit_history, states)
            joint_diagnostics = self._joint_stop_diagnostics(joint_unit_history, joint_cost_history)
            flow_stable = self._flow_stable(search_diagnostics, joint_diagnostics, iteration + 1)
            interval = max(int(self.config.equilibrium_check_interval), 1)
            can_check = (
                bool(self.config.nash_check)
                and flow_stable
                and (iteration + 1) % interval == 0
            )
            if can_check:
                local_nash_check_count += 1
                local_nash_diagnostics = self._local_nash_check(
                    problem=problem,
                    selected_parameters=best_parameters,
                    players=players,
                    material_probabilities=material_probabilities,
                    check_index=local_nash_check_count,
                )

            stop_diagnostics = {
                **search_diagnostics,
                **joint_diagnostics,
                **local_nash_diagnostics,
                "bayesian_equilibrium_converged": float(
                    local_nash_diagnostics.get("local_nash_converged", 0.0)
                ),
                "max_bayesian_regret": float(
                    local_nash_diagnostics.get("local_max_bayesian_regret", float("inf"))
                ),
                "ego_bayesian_regret": float(
                    local_nash_diagnostics.get("local_ego_regret", float("inf"))
                ),
                **{
                    f"{name}_bayesian_regret": float(
                        local_nash_diagnostics.get(f"local_{name}_regret", float("inf"))
                    )
                    for name in player_names
                    if name != ego.name
                },
            }
            last_flow_diagnostics = dict(stop_diagnostics)
            should_stop = (
                flow_stable
                and bool(self.config.early_stop)
                and (
                    not bool(self.config.nash_check)
                    or float(local_nash_diagnostics.get("local_nash_converged", 0.0)) > 0.5
                )
            )
            if should_stop:
                stop_reason = (
                    "early_stop:local_bayesian_nash"
                    if bool(self.config.nash_check)
                    else "early_stop:bayesian_igo_flow_stable"
                )

            history.append(
                {
                    "iteration": int(iteration),
                    "stop_reason": stop_reason if should_stop else None,
                    "flow_stable": float(flow_stable),
                    "ego_profile_value": float(profile_values[ego.name][0]),
                    "ego_mean_sigma": self._mean_sigma(states[ego.name]),
                    **{f"{name}_profile_value": float(profile_values[name][0]) for name in player_names if name != ego.name},
                    **{f"{name}_mean_sigma": self._mean_sigma(states[name]) for name in player_names if name != ego.name},
                    **stop_diagnostics,
                }
            )
            if should_stop:
                break

        if bool(self.config.nash_check) and local_nash_check_count == 0:
            local_nash_check_count += 1
            local_nash_diagnostics = self._local_nash_check(
                problem=problem,
                selected_parameters=best_parameters,
                players=players,
                material_probabilities=material_probabilities,
                check_index=local_nash_check_count,
            )

        best_joint = problem.decode_joint(best_parameters)
        best_costs = dict(problem.evaluate_joint(best_joint))
        final_diagnostics = {
            **last_flow_diagnostics,
            **local_nash_diagnostics,
            "bayesian_equilibrium_converged": float(
                local_nash_diagnostics.get("local_nash_converged", 0.0)
            ),
            "max_bayesian_regret": float(
                local_nash_diagnostics.get("local_max_bayesian_regret", float("inf"))
            ),
            "ego_bayesian_regret": float(
                local_nash_diagnostics.get("local_ego_regret", float("inf"))
            ),
            **{
                f"{name}_bayesian_regret": float(
                    local_nash_diagnostics.get(f"local_{name}_regret", float("inf"))
                )
                for name in player_names
                if name != ego.name
            },
        }
        return GameOptimizationResult(
            best_parameters=best_parameters,
            best_joint_trajectory=best_joint,
            player_costs=best_costs,
            status="success",
            history=history,
            metadata={
                "optimizer": self.name,
                "solver_formulation": "synchronous_type_conditioned_bayesian_igo_flow",
                "n_components": int(self.config.n_components),
                "n_samples": int(self.config.n_samples),
                "n_iterations": int(self.config.n_iterations),
                "executed_iterations": int(len(history)),
                "flow_batch_evaluations": int(flow_batch_evaluations),
                "stop_reason": stop_reason,
                "early_stop": bool(self.config.early_stop),
                "physical_players": (ego.name, *problem.physical_actors.keys()),
                "strategy_players": tuple(player_names),
                "actor_type_probabilities": {
                    actor_id: dict(actor.type_probabilities)
                    for actor_id, actor in problem.physical_actors.items()
                },
                "bayesian_equilibrium_check": bool(self.config.nash_check),
                "equilibrium_regret_tol": float(self.config.equilibrium_regret_tol),
                "equilibrium_check_interval": int(self.config.equilibrium_check_interval),
                "local_nash_check_count": int(local_nash_check_count),
                "local_nash_samples": int(self.config.local_nash_samples),
                "local_nash_perturbation": float(self.config.local_nash_perturbation),
                "best_response_feedback_count": 0,
                "candidate_pool": "final_iteration",
                "nash_check": bool(self.config.nash_check),
                **best_selection,
                **final_diagnostics,
            },
            player_populations=player_positions,
            player_values=player_values,
            joint_merit=None,
        )

    def _sample_all(self, states, bounds, players):
        unit_samples = {}
        component_ids = {}
        parameter_samples = {}
        for player in players:
            units, ids = self._sample_population(states[player.name])
            unit_samples[player.name] = units
            component_ids[player.name] = ids
            parameter_samples[player.name] = denormalize_parameters(units, bounds[player.name])
        return unit_samples, component_ids, parameter_samples

    @staticmethod
    def _evaluate_samples(problem, parameters):
        return problem.evaluate_players_batch(parameters)

    def _evaluate_profile(self, problem, parameters):
        batched = {
            name: np.asarray(value, dtype=float).reshape(1, -1)
            for name, value in parameters.items()
        }
        return self._evaluate_samples(problem, batched)

    @staticmethod
    def _player_values(evaluation):
        return {
            name: np.asarray(values, dtype=float).reshape(-1)
            for name, values in evaluation.player_values.items()
        }

    def _record_flow_batch(
        self,
        parameters,
        evaluation,
        player_positions,
        player_values,
    ) -> None:
        values = self._player_values(evaluation)
        for name, samples in parameters.items():
            player_positions[name] = np.asarray(samples, dtype=float)
            player_values[name] = self._finite_values(values[name])

    def _representative_profile(self, states, bounds, players) -> dict[str, np.ndarray]:
        profile = {}
        for player in players:
            state = states[player.name]
            weights = np.asarray(state.weights, dtype=float)
            radii = self._component_radii(state)
            eligible = np.where(weights >= float(self.config.component_weight_tol))[0]
            candidates = eligible if eligible.size else np.arange(weights.size)
            component_index = min(
                (int(index) for index in candidates),
                key=lambda index: (-float(weights[index]), float(radii[index])),
            )
            unit = np.clip(state.components[component_index].mean, 0.0, 1.0)
            profile[player.name] = denormalize_parameters(unit, bounds[player.name])
        return profile

    @staticmethod
    def _profile_selection(evaluation, ego_name: str) -> dict[str, object]:
        return {
            "selection_index": -1,
            "selection_mode": "bayesian_igo_flow_representative_profile",
            "selection_ego_value": float(np.asarray(evaluation.player_values[ego_name], dtype=float)[0]),
            **{
                f"selection_{name}_value": float(np.asarray(values, dtype=float)[0])
                for name, values in evaluation.player_values.items()
                if name != ego_name
            },
        }

    def _flow_stable(self, search_diagnostics, joint_diagnostics, iteration_count: int) -> bool:
        if int(iteration_count) < max(int(self.config.min_iterations), 0):
            return False
        search_ok = float(search_diagnostics.get("all_players_search_converged", 0.0)) > 0.5
        joint_ok = (
            float(joint_diagnostics.get("joint_theta_window_converged", 0.0)) > 0.5
            or float(joint_diagnostics.get("joint_cost_window_converged", 0.0)) > 0.5
        )
        return bool(search_ok and joint_ok)

    def _local_nash_check(
        self,
        problem: BayesianGameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
        players: tuple[GamePlayer, ...],
        material_probabilities: Mapping[str, float],
        check_index: int,
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        material_regrets = []
        for index, player in enumerate(players):
            selected = np.asarray(selected_parameters[player.name], dtype=float)
            candidates = self._local_perturbation_candidates(player, selected, check_index, index)
            values = self._finite_values(problem.evaluate_unilateral_batch(player.name, candidates, selected_parameters))
            best_index = int(np.argmin(values))
            best_cost = float(values[best_index])
            regret = self._normalized_regret(np.asarray([values[0], best_cost], dtype=float))
            prefix = "ego" if player.name == problem.ego_player.name else player.name
            diagnostics[f"local_{prefix}_current_cost"] = float(values[0])
            diagnostics[f"local_{prefix}_best_response_cost"] = best_cost
            diagnostics[f"local_{prefix}_regret"] = float(regret)
            diagnostics[f"local_{prefix}_candidate_count"] = float(candidates.shape[0])
            if float(material_probabilities.get(player.name, 1.0)) >= float(self.config.material_type_probability):
                material_regrets.append(regret)

        max_regret = float(np.max(material_regrets)) if material_regrets else float("inf")
        diagnostics["local_max_bayesian_regret"] = max_regret
        diagnostics["local_nash_converged"] = float(
            np.isfinite(max_regret) and max_regret <= float(self.config.equilibrium_regret_tol)
        )
        return diagnostics

    def _local_perturbation_candidates(
        self,
        player: GamePlayer,
        selected: np.ndarray,
        check_index: int,
        seed_offset: int,
    ) -> np.ndarray:
        low = np.asarray(player.lower_bound, dtype=float)
        high = np.asarray(player.upper_bound, dtype=float)
        selected = np.clip(np.asarray(selected, dtype=float), low, high)
        selected_unit = np.clip(normalize_parameters(selected, (low, high)), 0.0, 1.0)
        sample_count = max(int(self.config.local_nash_samples), 0)
        rng = np.random.default_rng(int(self.config.local_nash_seed) + 1009 * int(check_index) + int(seed_offset))
        perturbation = max(float(self.config.local_nash_perturbation), 0.0)
        random_units = np.clip(
            selected_unit[None, :] + rng.normal(0.0, perturbation, size=(sample_count, selected_unit.size)),
            0.0,
            1.0,
        )
        units = np.vstack([selected_unit, random_units])
        return self._dedupe_rows(denormalize_parameters(units, (low, high)), (low, high))

    @staticmethod
    def _local_nash_not_evaluated(player_names: list[str]) -> dict[str, float]:
        return {
            "local_ego_regret": float("inf"),
            **{f"local_{name}_regret": float("inf") for name in player_names if name != "ego"},
            "local_max_bayesian_regret": float("inf"),
            "local_nash_converged": 0.0,
        }

    @staticmethod
    def _normalized_type_probabilities(probabilities: Mapping[str, float], type_names: tuple[str, ...]) -> np.ndarray:
        values = np.asarray([max(float(probabilities.get(name, 0.0)), 0.0) for name in type_names], dtype=float)
        total = float(np.sum(values))
        if not np.all(np.isfinite(values)) or total <= 1e-12:
            raise ValueError(f"Bayesian type probabilities must contain positive finite mass: {probabilities}.")
        return values / total

    @staticmethod
    def _finite_values(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Bayesian IGO evaluator produced non-finite values.")
        return result

    @staticmethod
    def _normalized_regret(values: np.ndarray) -> float:
        costs = np.asarray(values, dtype=float).reshape(-1)
        finite = costs[np.isfinite(costs)]
        if finite.size == 0:
            return float("inf")
        current = float(finite[0])
        best = float(np.min(finite))
        return max(current - best, 0.0) / max(abs(current), 1.0)
