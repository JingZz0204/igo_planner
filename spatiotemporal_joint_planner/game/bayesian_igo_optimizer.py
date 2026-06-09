from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

from spatiotemporal_joint_planner.game.base import GameOptimizationResult, GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.igo_game_optimizer import GameIGOConfig, GameIGOOptimizer
from spatiotemporal_joint_planner.optimizer.cma_es_optimizer import denormalize_parameters


@dataclass(frozen=True)
class BayesianIGOConfig(GameIGOConfig):
    equilibrium_check_interval: int = 3
    equilibrium_regret_tol: float = 0.08
    material_type_probability: float = 0.05
    equilibrium_polish_rounds: int = 2


@dataclass(frozen=True)
class BayesianBatchEvaluation:
    aggregate_ego_values: np.ndarray
    ego_values_by_type: Mapping[str, np.ndarray]
    target_values_by_type: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class BayesianGameOptimizationProblem:
    ego_player: GamePlayer
    target_players_by_type: Mapping[str, GamePlayer]
    type_probabilities: Mapping[str, float]
    evaluate_batch: Callable[[np.ndarray, Mapping[str, np.ndarray]], BayesianBatchEvaluation]
    evaluate_type_batch: Callable[[str, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    decode_joint: Callable[[Mapping[str, np.ndarray]], JointTrajectory]
    evaluate_joint: Callable[[JointTrajectory], Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)


class BayesianIGOOptimizer(GameIGOOptimizer):
    """IGO solver for one physical opponent with type-conditioned policies."""

    def __init__(self, config: Optional[BayesianIGOConfig] = None):
        super().__init__(config or BayesianIGOConfig())
        self.config = config or BayesianIGOConfig()

    @property
    def name(self) -> str:
        return "bayesian_igo_game"

    def optimize(self, problem: BayesianGameOptimizationProblem) -> GameOptimizationResult:
        ego = problem.ego_player
        type_names = tuple(problem.target_players_by_type.keys())
        if not type_names:
            raise ValueError("Bayesian IGO requires at least one target type.")
        probabilities = self._normalized_type_probabilities(problem.type_probabilities, type_names)
        strategy_players = (ego, *(problem.target_players_by_type[name] for name in type_names))
        strategy_names = [player.name for player in strategy_players]
        bounds = {
            player.name: (np.asarray(player.lower_bound, dtype=float), np.asarray(player.upper_bound, dtype=float))
            for player in strategy_players
        }
        anchors = {
            player.name: self._clip_positions_to_bounds(self._as_anchor_matrix(player.initial_population), bounds[player.name])
            for player in strategy_players
        }
        states = {
            player.name: self._initial_state(player.name, bounds[player.name], anchors[player.name])
            for player in strategy_players
        }

        joint_batches: list[dict[str, np.ndarray]] = []
        ego_value_batches: list[np.ndarray] = []
        target_value_batches: dict[str, list[np.ndarray]] = {name: [] for name in type_names}
        player_positions: dict[str, list[np.ndarray]] = {name: [] for name in strategy_names}
        player_values: dict[str, list[np.ndarray]] = {name: [] for name in strategy_names}
        best_value_history: dict[str, list[float]] = {name: [] for name in strategy_names}
        best_unit_history: dict[str, list[np.ndarray]] = {name: [] for name in strategy_names}
        selected_unit_history: list[np.ndarray] = []
        selected_cost_history: list[np.ndarray] = []
        history: list[dict] = []
        stop_reason = "max_iterations"
        stop_diagnostics: dict[str, float] = self._equilibrium_not_evaluated(type_names)

        initial_samples = self._initial_paired_samples(strategy_players, anchors, bounds)
        if initial_samples:
            initial_evaluation = problem.evaluate_batch(
                initial_samples[ego.name],
                {
                    name: initial_samples[problem.target_players_by_type[name].name]
                    for name in type_names
                },
            )
            self._record_batch(
                initial_samples,
                initial_evaluation,
                ego,
                type_names,
                problem.target_players_by_type,
                joint_batches,
                ego_value_batches,
                target_value_batches,
                player_positions,
                player_values,
            )

        best_parameters: Optional[dict[str, np.ndarray]] = None
        best_selection: dict[str, object] = {}
        for iteration in range(max(int(self.config.n_iterations), 0)):
            unit_samples: dict[str, np.ndarray] = {}
            component_ids: dict[str, np.ndarray] = {}
            parameter_samples: dict[str, np.ndarray] = {}
            for player in strategy_players:
                units, ids = self._sample_population(states[player.name])
                unit_samples[player.name] = units
                component_ids[player.name] = ids
                parameter_samples[player.name] = denormalize_parameters(units, bounds[player.name])

            evaluation = problem.evaluate_batch(
                parameter_samples[ego.name],
                {
                    name: parameter_samples[problem.target_players_by_type[name].name]
                    for name in type_names
                },
            )
            self._record_batch(
                parameter_samples,
                evaluation,
                ego,
                type_names,
                problem.target_players_by_type,
                joint_batches,
                ego_value_batches,
                target_value_batches,
                player_positions,
                player_values,
            )

            iteration_values = {ego.name: np.asarray(evaluation.aggregate_ego_values, dtype=float)}
            for type_name in type_names:
                player_name = problem.target_players_by_type[type_name].name
                iteration_values[player_name] = np.asarray(evaluation.target_values_by_type[type_name], dtype=float)
            for player in strategy_players:
                values = iteration_values[player.name]
                best_idx = int(np.argmin(values))
                best_value_history[player.name].append(float(values[best_idx]))
                best_unit_history[player.name].append(np.asarray(unit_samples[player.name][best_idx], dtype=float).copy())
                states[player.name] = self._update_distribution(
                    states[player.name],
                    unit_samples[player.name],
                    component_ids[player.name],
                    values,
                    iteration,
                )

            best_parameters, best_selection = self._select_bayesian_joint_candidate(
                joint_batches,
                ego_value_batches,
                target_value_batches,
                ego,
                type_names,
                problem.target_players_by_type,
                probabilities,
            )
            selected_unit_history.append(self._joint_unit_vector(best_parameters, bounds, strategy_names))
            selected_cost_history.append(
                np.asarray(
                    [
                        float(best_selection.get("selection_ego_value", float("inf"))),
                        *[
                            float(best_selection.get(f"selection_{type_name}_target_value", float("inf")))
                            for type_name in type_names
                        ],
                    ],
                    dtype=float,
                )
            )
            search_diagnostics = self._early_stop_diagnostics(best_value_history, best_unit_history, states)
            joint_diagnostics = self._joint_stop_diagnostics(selected_unit_history, selected_cost_history)
            equilibrium_diagnostics = self._equilibrium_not_evaluated(type_names)
            search_stable = float(search_diagnostics.get("all_players_search_converged", 0.0)) > 0.5
            interval = max(int(self.config.equilibrium_check_interval), 1)
            if (
                bool(self.config.nash_check)
                and search_stable
                and int(iteration + 1) >= max(int(self.config.min_iterations), 0)
                and int(iteration + 1) % interval == 0
            ):
                best_parameters, equilibrium_diagnostics = self._polish_bayesian_equilibrium(
                    problem=problem,
                    selected_parameters=best_parameters,
                    current_samples=parameter_samples,
                    current_values=iteration_values,
                    anchors=anchors,
                    states=states,
                    ego=ego,
                    type_names=type_names,
                    probabilities=probabilities,
                )
            stop_diagnostics = {
                **search_diagnostics,
                **joint_diagnostics,
                **equilibrium_diagnostics,
            }
            early_stop_trigger = self._bayesian_early_stop_trigger(iteration + 1, stop_diagnostics)
            should_stop = early_stop_trigger is not None
            if should_stop:
                stop_reason = f"early_stop:{early_stop_trigger}"
            history.append(
                {
                    "iteration": int(iteration),
                    "stop_reason": stop_reason if should_stop else None,
                    "ego_best_value": float(np.min(iteration_values[ego.name])),
                    "ego_mean_sigma": self._mean_sigma(states[ego.name]),
                    **{
                        f"{type_name}_target_best_value": float(
                            np.min(iteration_values[problem.target_players_by_type[type_name].name])
                        )
                        for type_name in type_names
                    },
                    **{
                        f"{type_name}_target_mean_sigma": self._mean_sigma(
                            states[problem.target_players_by_type[type_name].name]
                        )
                        for type_name in type_names
                    },
                    **stop_diagnostics,
                }
            )
            if should_stop:
                break

        if best_parameters is None:
            if not joint_batches:
                raise RuntimeError("Bayesian IGO did not evaluate any joint samples.")
            best_parameters, best_selection = self._select_bayesian_joint_candidate(
                joint_batches,
                ego_value_batches,
                target_value_batches,
                ego,
                type_names,
                problem.target_players_by_type,
                probabilities,
            )

        if bool(self.config.nash_check) and not np.isfinite(float(stop_diagnostics.get("max_bayesian_regret", np.inf))):
            best_parameters, final_equilibrium = self._polish_bayesian_equilibrium(
                problem=problem,
                selected_parameters=best_parameters,
                current_samples={
                    name: np.vstack(chunks) if chunks else np.empty((0, bounds[name][0].size), dtype=float)
                    for name, chunks in player_positions.items()
                },
                current_values={
                    name: np.concatenate(chunks) if chunks else np.empty((0,), dtype=float)
                    for name, chunks in player_values.items()
                },
                anchors=anchors,
                states=states,
                ego=ego,
                type_names=type_names,
                probabilities=probabilities,
            )
            stop_diagnostics = {
                **stop_diagnostics,
                **final_equilibrium,
            }

        best_joint = problem.decode_joint(best_parameters)
        best_costs = dict(problem.evaluate_joint(best_joint))
        self._last_positions = {name: np.asarray(value, dtype=float).copy() for name, value in best_parameters.items()}
        populations = {
            name: np.vstack(chunks) if chunks else np.empty((0, bounds[name][0].size), dtype=float)
            for name, chunks in player_positions.items()
        }
        values = {
            name: np.concatenate(chunks) if chunks else np.empty((0,), dtype=float)
            for name, chunks in player_values.items()
        }
        return GameOptimizationResult(
            best_parameters=best_parameters,
            best_joint_trajectory=best_joint,
            player_costs=best_costs,
            status="success",
            history=history,
            metadata={
                "optimizer": self.name,
                "n_components": int(self.config.n_components),
                "n_samples": int(self.config.n_samples),
                "n_iterations": int(self.config.n_iterations),
                "executed_iterations": int(len(history)),
                "stop_reason": stop_reason,
                "early_stop": bool(self.config.early_stop),
                "physical_players": (ego.name, "target_rear"),
                "target_types": type_names,
                "type_probabilities": {name: float(probabilities[idx]) for idx, name in enumerate(type_names)},
                "bayesian_equilibrium_check": bool(self.config.nash_check),
                "equilibrium_regret_tol": float(self.config.equilibrium_regret_tol),
                "equilibrium_check_interval": int(self.config.equilibrium_check_interval),
                "equilibrium_polish_rounds": int(self.config.equilibrium_polish_rounds),
                "nash_check": False,
                **best_selection,
                **stop_diagnostics,
            },
            player_populations=populations,
            player_values=values,
            joint_merit=None,
        )

    def _record_batch(
        self,
        samples: Mapping[str, np.ndarray],
        evaluation: BayesianBatchEvaluation,
        ego: GamePlayer,
        type_names: tuple[str, ...],
        target_players: Mapping[str, GamePlayer],
        joint_batches: list[dict[str, np.ndarray]],
        ego_value_batches: list[np.ndarray],
        target_value_batches: dict[str, list[np.ndarray]],
        player_positions: dict[str, list[np.ndarray]],
        player_values: dict[str, list[np.ndarray]],
    ) -> None:
        joint_batches.append({name: np.asarray(value, dtype=float) for name, value in samples.items()})
        ego_values = self._finite_values(evaluation.aggregate_ego_values)
        ego_value_batches.append(ego_values)
        player_positions[ego.name].append(np.asarray(samples[ego.name], dtype=float))
        player_values[ego.name].append(ego_values)
        for type_name in type_names:
            player_name = target_players[type_name].name
            values = self._finite_values(evaluation.target_values_by_type[type_name])
            target_value_batches[type_name].append(values)
            player_positions[player_name].append(np.asarray(samples[player_name], dtype=float))
            player_values[player_name].append(values)

    def _initial_paired_samples(
        self,
        players: tuple[GamePlayer, ...],
        anchors: Mapping[str, np.ndarray],
        bounds: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        count = min(
            max(int(self.config.max_anchor_samples), 1),
            max((int(value.shape[0]) for value in anchors.values()), default=1),
        )
        result = {}
        for player in players:
            values = anchors[player.name]
            if values.size == 0:
                values = player.trajectory_model.reference_parameters(player.problem).reshape(1, -1)
            rows = [values[idx % values.shape[0]] for idx in range(count)]
            result[player.name] = self._clip_positions_to_bounds(np.asarray(rows, dtype=float), bounds[player.name])
        return result

    def _select_bayesian_joint_candidate(
        self,
        joint_batches: list[dict[str, np.ndarray]],
        ego_value_batches: list[np.ndarray],
        target_value_batches: Mapping[str, list[np.ndarray]],
        ego: GamePlayer,
        type_names: tuple[str, ...],
        target_players: Mapping[str, GamePlayer],
        probabilities: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        parameters = {
            ego.name: np.vstack([batch[ego.name] for batch in joint_batches]),
            **{
                target_players[name].name: np.vstack([batch[target_players[name].name] for batch in joint_batches])
                for name in type_names
            },
        }
        ego_values = np.concatenate(ego_value_batches)
        target_values = {name: np.concatenate(target_value_batches[name]) for name in type_names}
        count = int(ego_values.size)
        finite = np.isfinite(ego_values)
        for name in type_names:
            finite &= np.isfinite(target_values[name])
        if not np.any(finite):
            index = 0
            mode = "fallback_first_no_finite"
        else:
            target_ranks = {name: self._normalized_ranks(target_values[name]) for name in type_names}
            gate = finite.copy()
            for idx, name in enumerate(type_names):
                if float(probabilities[idx]) >= float(self.config.material_type_probability):
                    gate &= target_ranks[name] <= float(self.config.opponent_rank_gate)
            candidates = np.where(gate if np.any(gate) else finite)[0]
            weighted_target_rank = np.sum(
                np.asarray([probabilities[idx] * target_ranks[name][candidates] for idx, name in enumerate(type_names)]),
                axis=0,
            )
            order = np.lexsort((weighted_target_rank, ego_values[candidates]))
            index = int(candidates[int(order[0])])
            mode = "ego_min_with_type_conditioned_target_gate" if np.any(gate) else "fallback_ego_min"
        selected = {name: values[index].copy() for name, values in parameters.items()}
        diagnostics: dict[str, object] = {
            "selection_index": int(index),
            "selection_mode": mode,
            "selection_ego_value": float(ego_values[index]),
        }
        for name in type_names:
            diagnostics[f"selection_{name}_target_value"] = float(target_values[name][index])
        return selected, diagnostics

    def _bayesian_equilibrium_diagnostics(
        self,
        problem: BayesianGameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
        current_samples: Mapping[str, np.ndarray],
        current_values: Mapping[str, np.ndarray],
        anchors: Mapping[str, np.ndarray],
        states,
        ego: GamePlayer,
        type_names: tuple[str, ...],
        probabilities: np.ndarray,
    ) -> dict[str, float]:
        selected_targets = {
            name: np.asarray(selected_parameters[problem.target_players_by_type[name].name], dtype=float)
            for name in type_names
        }
        ego_candidates = self._nash_candidate_matrix(
            ego.name,
            selected_parameters,
            current_samples.get(ego.name),
            current_values.get(ego.name),
            anchors.get(ego.name),
            states[ego.name],
            (np.asarray(ego.lower_bound, dtype=float), np.asarray(ego.upper_bound, dtype=float)),
        )
        target_batches = {
            name: np.repeat(selected_targets[name].reshape(1, -1), ego_candidates.shape[0], axis=0)
            for name in type_names
        }
        ego_values = self._finite_values(problem.evaluate_batch(ego_candidates, target_batches).aggregate_ego_values)
        ego_regret = self._normalized_regret(ego_values)
        diagnostics: dict[str, float] = {"ego_bayesian_regret": ego_regret}
        regrets = [ego_regret]
        material_flags = [np.isfinite(ego_regret)]
        for idx, type_name in enumerate(type_names):
            player = problem.target_players_by_type[type_name]
            candidates = self._nash_candidate_matrix(
                player.name,
                selected_parameters,
                current_samples.get(player.name),
                current_values.get(player.name),
                anchors.get(player.name),
                states[player.name],
                (np.asarray(player.lower_bound, dtype=float), np.asarray(player.upper_bound, dtype=float)),
            )
            repeated_ego = np.repeat(np.asarray(selected_parameters[ego.name], dtype=float).reshape(1, -1), candidates.shape[0], axis=0)
            _ego_type_values, target_values = problem.evaluate_type_batch(type_name, repeated_ego, candidates)
            regret = self._normalized_regret(target_values)
            diagnostics[f"{type_name}_target_bayesian_regret"] = regret
            if float(probabilities[idx]) >= float(self.config.material_type_probability):
                regrets.append(regret)
                material_flags.append(np.isfinite(regret))
        max_regret = float(np.max(regrets)) if regrets else float("inf")
        diagnostics["max_bayesian_regret"] = max_regret
        diagnostics["bayesian_equilibrium_converged"] = float(
            all(material_flags) and max_regret <= float(self.config.equilibrium_regret_tol)
        )
        return diagnostics

    def _polish_bayesian_equilibrium(
        self,
        problem: BayesianGameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
        current_samples: Mapping[str, np.ndarray],
        current_values: Mapping[str, np.ndarray],
        anchors: Mapping[str, np.ndarray],
        states,
        ego: GamePlayer,
        type_names: tuple[str, ...],
        probabilities: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        selected = {name: np.asarray(value, dtype=float).copy() for name, value in selected_parameters.items()}
        rounds = max(int(self.config.equilibrium_polish_rounds), 0)
        for _ in range(rounds):
            for type_name in type_names:
                player = problem.target_players_by_type[type_name]
                candidates = self._nash_candidate_matrix(
                    player.name,
                    selected,
                    current_samples.get(player.name),
                    current_values.get(player.name),
                    anchors.get(player.name),
                    states[player.name],
                    (np.asarray(player.lower_bound, dtype=float), np.asarray(player.upper_bound, dtype=float)),
                )
                repeated_ego = np.repeat(selected[ego.name].reshape(1, -1), candidates.shape[0], axis=0)
                _ego_values, target_values = problem.evaluate_type_batch(type_name, repeated_ego, candidates)
                finite_values = self._finite_values(target_values)
                selected[player.name] = candidates[int(np.argmin(finite_values))].copy()

            ego_candidates = self._nash_candidate_matrix(
                ego.name,
                selected,
                current_samples.get(ego.name),
                current_values.get(ego.name),
                anchors.get(ego.name),
                states[ego.name],
                (np.asarray(ego.lower_bound, dtype=float), np.asarray(ego.upper_bound, dtype=float)),
            )
            target_batches = {
                name: np.repeat(
                    selected[problem.target_players_by_type[name].name].reshape(1, -1),
                    ego_candidates.shape[0],
                    axis=0,
                )
                for name in type_names
            }
            ego_values = self._finite_values(problem.evaluate_batch(ego_candidates, target_batches).aggregate_ego_values)
            selected[ego.name] = ego_candidates[int(np.argmin(ego_values))].copy()

        diagnostics = self._bayesian_equilibrium_diagnostics(
            problem=problem,
            selected_parameters=selected,
            current_samples=current_samples,
            current_values=current_values,
            anchors=anchors,
            states=states,
            ego=ego,
            type_names=type_names,
            probabilities=probabilities,
        )
        return selected, diagnostics

    def _bayesian_early_stop_trigger(self, iteration_count: int, diagnostics: Mapping[str, float]) -> Optional[str]:
        if not bool(self.config.early_stop) or int(iteration_count) < max(int(self.config.min_iterations), 0):
            return None
        search_ok = float(diagnostics.get("all_players_search_converged", 0.0)) > 0.5
        joint_ok = (
            float(diagnostics.get("joint_theta_window_converged", 0.0)) > 0.5
            or float(diagnostics.get("joint_cost_window_converged", 0.0)) > 0.5
        )
        if not search_ok or not joint_ok:
            return None
        if bool(self.config.nash_check):
            if float(diagnostics.get("bayesian_equilibrium_converged", 0.0)) > 0.5:
                return "bayesian_equilibrium_search_stable"
            return None
        return "bayesian_search_stable"

    def _equilibrium_not_evaluated(self, type_names: tuple[str, ...]) -> dict[str, float]:
        return {
            "ego_bayesian_regret": float("inf"),
            **{f"{name}_target_bayesian_regret": float("inf") for name in type_names},
            "max_bayesian_regret": float("inf"),
            "bayesian_equilibrium_converged": 0.0,
        }

    @staticmethod
    def _normalized_type_probabilities(probabilities: Mapping[str, float], type_names: tuple[str, ...]) -> np.ndarray:
        values = np.asarray([max(float(probabilities.get(name, 0.0)), 0.0) for name in type_names], dtype=float)
        total = float(np.sum(values))
        return values / total if total > 1e-12 else np.full((len(type_names),), 1.0 / float(len(type_names)))

    @staticmethod
    def _finite_values(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(-1).copy()
        result[~np.isfinite(result)] = float("inf")
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
