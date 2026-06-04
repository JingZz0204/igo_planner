from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from spatiotemporal_joint_planner.game.base import GameOptimizationProblem, GameOptimizationResult, JointTrajectory
from spatiotemporal_joint_planner.optimizer.cma_es_optimizer import denormalize_parameters, normalize_parameters


@dataclass(frozen=True)
class GameIGOConfig:
    n_components: int = 2
    n_samples: int = 48
    n_iterations: int = 50
    elite_fraction: float = 0.25
    init_std: float = 0.22
    min_std: float = 0.035
    max_std: float = 0.45
    weight_floor: float = 1e-3
    weight_lr: float = 0.55
    mean_lr: float = 0.85
    sigma_lr: float = 0.65
    seed: int = 0
    early_stop: bool = True
    min_iterations: int = 3
    convergence_window: int = 5
    cost_window_tol: float = 1e-3
    theta_window_tol: float = 2e-2
    component_sigma_tol: float = 0.08
    component_weight_tol: float = 0.15
    max_anchor_samples: int = 192
    opponent_rank_gate: float = 0.35
    nash_check: bool = True
    nash_regret_tol: float = 0.02
    nash_candidate_limit: int = 48
    nash_perturbation: float = 0.04
    joint_theta_window_tol: float = 0.03
    joint_cost_window_tol: float = 2e-3


@dataclass
class _CMAComponentState:
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    p_sigma: np.ndarray
    p_c: np.ndarray


@dataclass
class _DistributionState:
    weights: np.ndarray
    components: list[_CMAComponentState]


class GameIGOOptimizer:
    """Paired-sample IGO-style optimizer for general-sum trajectory games."""

    def __init__(self, config: Optional[GameIGOConfig] = None):
        self.config = config or GameIGOConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._last_positions: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "igo_game"

    def reset(self) -> None:
        self._last_positions.clear()

    def optimize(self, problem: GameOptimizationProblem) -> GameOptimizationResult:
        if not problem.players:
            raise ValueError("Game optimizer requires at least one player.")

        player_names = [player.name for player in problem.players]
        bounds = {
            player.name: (np.asarray(player.lower_bound, dtype=float), np.asarray(player.upper_bound, dtype=float))
            for player in problem.players
        }
        states = {
            player.name: self._initial_state(
                player.name,
                bounds[player.name],
                self._as_anchor_matrix(player.initial_population),
            )
            for player in problem.players
        }

        history: list[dict] = []
        all_positions = {name: [] for name in player_names}
        all_values = {name: [] for name in player_names}
        best_parameters: Optional[dict[str, np.ndarray]] = None
        best_joint: Optional[JointTrajectory] = None
        best_costs = None
        best_selection = {}
        best_value_history = {name: [] for name in player_names}
        best_unit_history = {name: [] for name in player_names}
        selected_unit_history: list[np.ndarray] = []
        selected_cost_history: list[np.ndarray] = []
        stop_reason = "max_iterations"
        stop_diagnostics: dict[str, float] = {}
        anchors_by_player = {
            player.name: self._clip_positions_to_bounds(
                self._as_anchor_matrix(player.initial_population),
                bounds[player.name],
            )
            for player in problem.players
        }
        anchor_samples = self._initial_joint_samples(problem, anchors_by_player, bounds)
        if anchor_samples:
            joint_results = self._evaluate_joint_samples(problem, anchor_samples)
            values = joint_results["values"]
            for name in player_names:
                all_positions[name].append(np.asarray(anchor_samples[name], dtype=float))
                all_values[name].append(np.asarray(values[name], dtype=float))
            best_parameters, best_selection = self._select_global_joint_candidate(all_positions, all_values, player_names)
            best_joint = None
            best_costs = None

        for iteration in range(max(int(self.config.n_iterations), 0)):
            unit_samples = {}
            component_ids = {}
            parameter_samples = {}
            for player in problem.players:
                units, ids = self._sample_population(states[player.name])
                unit_samples[player.name] = units
                component_ids[player.name] = ids
                parameter_samples[player.name] = denormalize_parameters(units, bounds[player.name])

            joint_results = self._evaluate_joint_samples(problem, parameter_samples)
            values = joint_results["values"]
            for name in player_names:
                all_positions[name].append(np.asarray(parameter_samples[name], dtype=float))
                all_values[name].append(np.asarray(values[name], dtype=float))
                best_idx = int(np.argmin(values[name]))
                best_value_history[name].append(float(values[name][best_idx]))
                best_unit_history[name].append(np.asarray(unit_samples[name][best_idx], dtype=float).copy())
                states[name] = self._update_distribution(
                    states[name],
                    unit_samples[name],
                    component_ids[name],
                    values[name],
                    iteration,
                )
            best_parameters, best_selection = self._select_global_joint_candidate(all_positions, all_values, player_names)
            best_joint = None
            best_costs = None

            selected_unit = self._joint_unit_vector(best_parameters, bounds, player_names)
            selected_cost = self._selection_cost_vector(best_selection, player_names)
            selected_unit_history.append(selected_unit)
            selected_cost_history.append(selected_cost)

            player_diagnostics = self._early_stop_diagnostics(best_value_history, best_unit_history, states)
            joint_diagnostics = self._joint_stop_diagnostics(selected_unit_history, selected_cost_history)
            nash_diagnostics = self._nash_diagnostics_not_evaluated(player_names)
            selected_feasibility = self._selected_feasibility_not_evaluated(player_names)
            if int(iteration + 1) >= max(int(self.config.min_iterations), 0):
                selected_feasibility = self._selected_joint_feasibility(problem, best_parameters, player_names)
                if bool(self.config.nash_check):
                    candidate_sets = {
                        name: self._nash_candidate_matrix(
                            player_name=name,
                            selected_parameters=best_parameters,
                            current_samples=parameter_samples.get(name),
                            current_values=values.get(name),
                            anchors=anchors_by_player.get(name),
                            state=states[name],
                            bounds=bounds[name],
                        )
                        for name in player_names
                    }
                    nash_diagnostics = self._approximate_nash_diagnostics(
                        problem,
                        best_parameters,
                        candidate_sets,
                        player_names,
                    )

            stop_diagnostics = {
                **player_diagnostics,
                **joint_diagnostics,
                **nash_diagnostics,
                **selected_feasibility,
            }
            early_stop_trigger = self._early_stop_trigger(iteration + 1, stop_diagnostics)
            should_stop = early_stop_trigger is not None
            if should_stop:
                stop_reason = f"early_stop:{early_stop_trigger}"
            history.append(
                {
                    "iteration": int(iteration),
                    "joint_merit": float(best_selection.get("selection_merit", float("inf"))),
                    "selection_index": int(best_selection.get("selection_index", -1)),
                    "selection_mode": str(best_selection.get("selection_mode", "")),
                    "stop_reason": stop_reason if should_stop else None,
                    **{
                        f"{name}_best_value": float(np.min(values[name]))
                        for name in player_names
                    },
                    **{
                        f"{name}_mean_sigma": self._mean_sigma(states[name])
                        for name in player_names
                    },
                    **stop_diagnostics,
                }
            )
            if should_stop:
                break

        if best_parameters is None:
            raise RuntimeError("Game IGO did not evaluate any joint samples.")
        if best_joint is None or best_costs is None:
            best_joint = problem.decode_joint(best_parameters)
            best_costs = dict(problem.evaluate_joint(best_joint))

        self._last_positions = {name: value.copy() for name, value in best_parameters.items()}
        player_populations = {
            name: np.vstack(chunks) if chunks else np.empty((0, bounds[name][0].size), dtype=float)
            for name, chunks in all_positions.items()
        }
        player_values = {
            name: np.concatenate(chunks) if chunks else np.empty((0,), dtype=float)
            for name, chunks in all_values.items()
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
                "player_names": tuple(player_names),
                "joint_selection": "ego_prioritized_opponent_rank_gate",
                "opponent_rank_gate": float(self.config.opponent_rank_gate),
                "max_anchor_samples": int(self.config.max_anchor_samples),
                "nash_check": bool(self.config.nash_check),
                "nash_regret_tol": float(self.config.nash_regret_tol),
                "nash_candidate_limit": int(self.config.nash_candidate_limit),
                "nash_perturbation": float(self.config.nash_perturbation),
                "joint_theta_window_tol": float(self.config.joint_theta_window_tol),
                "joint_cost_window_tol": float(self.config.joint_cost_window_tol),
                **best_selection,
                **stop_diagnostics,
            },
            player_populations=player_populations,
            player_values=player_values,
            joint_merit=None,
        )

    def _evaluate_joint_samples(self, problem: GameOptimizationProblem, samples: Mapping[str, np.ndarray]) -> dict:
        player_names = list(samples.keys())
        sample_count = min(np.asarray(samples[name]).shape[0] for name in player_names)
        if problem.evaluate_joint_batch is not None:
            try:
                raw_values = problem.evaluate_joint_batch(samples)
                values = {}
                for name in player_names:
                    value = np.asarray(raw_values[name], dtype=float).reshape(-1)
                    if value.shape != (sample_count,):
                        raise ValueError(f"Batch value shape mismatch for {name}: {value.shape} != {(sample_count,)}")
                    value = value.copy()
                    value[~np.isfinite(value)] = float("inf")
                    values[name] = value
                return {"values": values}
            except Exception:
                pass
        values = {name: np.full((sample_count,), float("inf"), dtype=float) for name in player_names}
        trajectories = []
        costs = []
        for idx in range(sample_count):
            parameters = {name: np.asarray(samples[name][idx], dtype=float) for name in player_names}
            try:
                joint = problem.decode_joint(parameters)
                result_costs = dict(problem.evaluate_joint(joint))
            except Exception:
                joint = JointTrajectory(trajectories={})
                result_costs = {}
            trajectories.append(joint)
            costs.append(result_costs)
            for name in player_names:
                cost = result_costs.get(name)
                value = float("inf") if cost is None else float(cost.total)
                values[name][idx] = value if np.isfinite(value) else float("inf")
        return {"values": values, "trajectories": trajectories, "costs": costs}

    def _initial_joint_samples(
        self,
        problem: GameOptimizationProblem,
        anchors_by_player: Mapping[str, np.ndarray],
        bounds: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        player_names = [player.name for player in problem.players]
        if not player_names:
            return {}

        references = {}
        for player in problem.players:
            try:
                reference = player.trajectory_model.reference_parameters(player.problem)
            except Exception:
                reference = 0.5 * (bounds[player.name][0] + bounds[player.name][1])
            references[player.name] = np.clip(np.asarray(reference, dtype=float), bounds[player.name][0], bounds[player.name][1])

        max_samples = max(int(self.config.max_anchor_samples), 0)
        rows: list[dict[str, np.ndarray]] = []
        seen = set()

        def add_row(row: Mapping[str, np.ndarray]) -> None:
            if len(rows) >= max_samples:
                return
            clipped = {}
            key_values = []
            for name in player_names:
                low, high = bounds[name]
                value = np.clip(np.asarray(row[name], dtype=float), low, high)
                clipped[name] = value
                key_values.extend(np.round(value, 8).tolist())
            key = tuple(key_values)
            if key in seen:
                return
            seen.add(key)
            rows.append(clipped)

        add_row(references)
        for name in player_names:
            anchors = anchors_by_player.get(name, np.empty((0, bounds[name][0].size), dtype=float))
            for anchor in anchors:
                row = {other_name: references[other_name] for other_name in player_names}
                row[name] = np.asarray(anchor, dtype=float)
                add_row(row)

        max_anchor_count = max((anchors.shape[0] for anchors in anchors_by_player.values() if anchors.size), default=0)
        for idx in range(max_anchor_count):
            add_row(
                {
                    name: (
                        anchors_by_player[name][idx % anchors_by_player[name].shape[0]]
                        if anchors_by_player[name].size
                        else references[name]
                    )
                    for name in player_names
                }
            )

        if not rows:
            add_row(references)
        return {name: np.asarray([row[name] for row in rows], dtype=float) for name in player_names}

    def _initial_state(
        self,
        player_name: str,
        bounds: tuple[np.ndarray, np.ndarray],
        anchors: np.ndarray,
    ) -> _DistributionState:
        dim = int(bounds[0].size)
        n_components = max(int(self.config.n_components), 1)
        means = []
        if player_name in self._last_positions and self._last_positions[player_name].shape == (dim,):
            means.append(np.clip(normalize_parameters(self._last_positions[player_name], bounds), 0.0, 1.0))
        for anchor in anchors:
            if anchor.shape == (dim,):
                means.append(np.clip(normalize_parameters(anchor, bounds), 0.0, 1.0))
        while len(means) < n_components:
            if means:
                base = means[len(means) % len(means)]
                means.append(np.clip(base + self._rng.normal(0.0, 0.2, size=(dim,)), 0.0, 1.0))
            else:
                means.append(self._rng.uniform(0.0, 1.0, size=(dim,)))
        init_std = float(np.clip(self.config.init_std, self.config.min_std, self.config.max_std))
        components = [
            _CMAComponentState(
                mean=np.asarray(mean, dtype=float).copy(),
                sigma=init_std,
                covariance=np.eye(dim, dtype=float),
                p_sigma=np.zeros((dim,), dtype=float),
                p_c=np.zeros((dim,), dtype=float),
            )
            for mean in means[:n_components]
        ]
        return _DistributionState(
            weights=np.full((n_components,), 1.0 / float(n_components), dtype=float),
            components=components,
        )

    def _sample_population(self, state: _DistributionState) -> tuple[np.ndarray, np.ndarray]:
        total_samples = max(int(self.config.n_samples), int(state.weights.size))
        counts = self._component_counts(state.weights, total_samples)
        samples = []
        ids = []
        for component_id, (component, count) in enumerate(zip(state.components, counts)):
            for sample in self._sample_component(component, int(count), include_mean=True):
                samples.append(sample)
                ids.append(component_id)
        return np.asarray(samples[:total_samples], dtype=float), np.asarray(ids[:total_samples], dtype=int)

    def _sample_component(self, component: _CMAComponentState, count: int, include_mean: bool) -> list[np.ndarray]:
        dim = int(component.mean.size)
        count = max(int(count), 1)
        samples = []
        if include_mean:
            samples.append(component.mean.copy())
        eigvals, eigvecs = self._eigendecomposition(component.covariance)
        transform = eigvecs @ np.diag(np.sqrt(eigvals))
        while len(samples) < count:
            z = self._rng.normal(0.0, 1.0, size=(dim,))
            samples.append(np.clip(component.mean + float(component.sigma) * (transform @ z), 0.0, 1.0))
        return samples[:count]

    def _update_distribution(
        self,
        state: _DistributionState,
        samples: np.ndarray,
        component_ids: np.ndarray,
        values: np.ndarray,
        iteration: int,
    ) -> _DistributionState:
        new_components = []
        for component_id, component in enumerate(state.components):
            local = np.where(component_ids == component_id)[0]
            if local.size == 0:
                new_components.append(component)
                continue
            order = local[np.argsort(values[local])]
            elite_n = max(1, min(order.size, int(np.ceil(order.size * float(self.config.elite_fraction)))))
            selected = samples[order[:elite_n]]
            weights = self._rank_weights(elite_n)
            new_components.append(self._update_component(component, selected, weights, iteration))
        return _DistributionState(
            weights=self._update_component_weights(state.weights, component_ids, values),
            components=new_components,
        )

    def _update_component(
        self,
        component: _CMAComponentState,
        selected: np.ndarray,
        weights: np.ndarray,
        iteration: int,
    ) -> _CMAComponentState:
        dim = int(component.mean.size)
        mu_eff = 1.0 / float(np.sum(weights**2))
        params = self._strategy_parameters(dim, mu_eff)

        old_mean = component.mean.copy()
        sigma = max(float(component.sigma), 1e-12)
        y_selected = (np.asarray(selected, dtype=float) - old_mean[None, :]) / sigma
        y_w = np.sum(y_selected * weights[:, None], axis=0)
        mean = np.clip(old_mean + sigma * y_w, 0.0, 1.0)

        eigvals, eigvecs = self._eigendecomposition(component.covariance)
        invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        cs = params["cs"]
        cc = params["cc"]
        c1 = params["c1"]
        cmu = params["cmu"]
        chi_n = params["chi_n"]

        p_sigma = (1.0 - cs) * component.p_sigma + np.sqrt(cs * (2.0 - cs) * mu_eff) * (invsqrt @ y_w)
        norm_p_sigma = float(np.linalg.norm(p_sigma))
        denom = np.sqrt(max(1.0 - (1.0 - cs) ** (2.0 * (iteration + 1)), 1e-12)) * chi_n
        h_sigma = float(norm_p_sigma / max(denom, 1e-12) < 1.4 + 2.0 / (dim + 1.0))
        p_c = (1.0 - cc) * component.p_c + h_sigma * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_w

        rank_mu = np.zeros_like(component.covariance)
        for weight, y in zip(weights, y_selected):
            rank_mu += float(weight) * np.outer(y, y)

        covariance = (
            (1.0 - c1 - cmu + (1.0 - h_sigma) * c1 * cc * (2.0 - cc)) * component.covariance
            + c1 * np.outer(p_c, p_c)
            + cmu * rank_mu
        )
        covariance = self._repair_covariance(covariance)
        sigma = sigma * np.exp((cs / params["damps"]) * (norm_p_sigma / max(chi_n, 1e-12) - 1.0))
        sigma = float(np.clip(sigma, float(self.config.min_std), float(self.config.max_std)))
        return _CMAComponentState(mean=mean, sigma=sigma, covariance=covariance, p_sigma=p_sigma, p_c=p_c)

    def _update_component_weights(self, old_weights: np.ndarray, component_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        k = int(old_weights.size)
        elite_n = max(1, int(np.ceil(len(values) * float(self.config.elite_fraction))))
        order = np.asarray(sorted(range(len(values)), key=lambda idx: float(values[idx])), dtype=int)
        utility = np.zeros((len(values),), dtype=float)
        rank_weights = self._rank_weights(elite_n)
        for rank, idx in enumerate(order[:elite_n]):
            utility[idx] = rank_weights[rank]

        target = np.zeros((k,), dtype=float)
        for component_id in range(k):
            target[component_id] = float(np.sum(utility[component_ids == component_id]))

        floor = float(self.config.weight_floor)
        if float(np.sum(target)) <= 1e-12:
            target = np.asarray(old_weights, dtype=float)
        target = np.maximum(target, floor)
        target /= np.sum(target)

        weights = (1.0 - float(self.config.weight_lr)) * np.asarray(old_weights, dtype=float) + float(self.config.weight_lr) * target
        weights = np.maximum(weights, floor)
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def _strategy_parameters(dim: int, mu_eff: float) -> dict:
        dim = max(int(dim), 1)
        cc = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chi_n = np.sqrt(float(dim)) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        return {
            "cc": float(cc),
            "cs": float(cs),
            "c1": float(c1),
            "cmu": float(max(cmu, 0.0)),
            "damps": float(damps),
            "chi_n": float(chi_n),
        }

    @staticmethod
    def _eigendecomposition(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        covariance = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        return np.clip(eigvals, 1e-10, 1e3), eigvecs

    @classmethod
    def _repair_covariance(cls, covariance: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = cls._eigendecomposition(covariance)
        repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (repaired + repaired.T)

    @staticmethod
    def _component_sigmas(state: _DistributionState) -> np.ndarray:
        return np.asarray([component.sigma for component in state.components], dtype=float)

    @classmethod
    def _component_radii(cls, state: _DistributionState) -> np.ndarray:
        radii = []
        for component in state.components:
            eigvals, _ = cls._eigendecomposition(component.covariance)
            radii.append(float(component.sigma) * float(np.sqrt(np.mean(eigvals))))
        return np.asarray(radii, dtype=float)

    @classmethod
    def _mean_sigma(cls, state: _DistributionState) -> float:
        sigmas = cls._component_sigmas(state)
        weights = np.asarray(state.weights, dtype=float).reshape(-1)
        if sigmas.size == 0:
            return 0.0
        if weights.size != sigmas.size or float(np.sum(weights)) <= 1e-12:
            return float(np.mean(sigmas))
        weights = weights / np.sum(weights)
        return float(np.sum(weights * sigmas))

    def _early_stop_diagnostics(
        self,
        best_value_history: Mapping[str, list[float]],
        best_unit_history: Mapping[str, list[np.ndarray]],
        states: Mapping[str, _DistributionState],
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        player_cost_converged = []
        player_theta_converged = []
        player_sigma_converged = []
        window = max(int(self.config.convergence_window), 1)
        for name, values in best_value_history.items():
            cost_improvement = float("inf")
            theta_shift = float("inf")
            if len(values) >= window:
                previous = float(values[-window])
                current = float(values[-1])
                if np.isfinite(previous) and np.isfinite(current):
                    cost_improvement = max(0.0, previous - current) / max(abs(previous), 1.0)
            theta_values = best_unit_history.get(name, [])
            if len(theta_values) >= window:
                theta_matrix = np.asarray(theta_values[-window:], dtype=float)
                if theta_matrix.ndim == 2 and theta_matrix.shape[0] >= 2:
                    deltas = np.linalg.norm(np.diff(theta_matrix, axis=0), axis=1) / np.sqrt(
                        float(theta_matrix.shape[1])
                    )
                    theta_shift = float(np.max(deltas)) if deltas.size else 0.0
            state = states[name]
            eligible = np.where(state.weights >= float(self.config.component_weight_tol))[0]
            sigmas = self._component_sigmas(state)
            radii = self._component_radii(state)
            sigma = float("inf")
            radius = float("inf")
            weight = 0.0
            if eligible.size:
                best_idx = int(eligible[np.argmin(radii[eligible])])
                sigma = float(sigmas[best_idx])
                radius = float(radii[best_idx])
                weight = float(state.weights[best_idx])
            cost_ok = float(np.isfinite(cost_improvement) and cost_improvement <= float(self.config.cost_window_tol))
            theta_ok = float(np.isfinite(theta_shift) and theta_shift <= float(self.config.theta_window_tol))
            sigma_ok = float(radius <= float(self.config.component_sigma_tol))
            diagnostics[f"{name}_cost_window_improvement"] = float(cost_improvement)
            diagnostics[f"{name}_theta_window_shift"] = float(theta_shift)
            diagnostics[f"{name}_qualified_component_sigma"] = float(sigma)
            diagnostics[f"{name}_qualified_component_radius"] = float(radius)
            diagnostics[f"{name}_qualified_component_weight"] = float(weight)
            diagnostics[f"{name}_cost_window_converged"] = cost_ok
            diagnostics[f"{name}_theta_window_converged"] = theta_ok
            diagnostics[f"{name}_component_sigma_converged"] = sigma_ok
            player_cost_converged.append(cost_ok)
            player_theta_converged.append(theta_ok)
            player_sigma_converged.append(sigma_ok)
        diagnostics["all_players_cost_window_converged"] = float(all(value > 0.5 for value in player_cost_converged))
        diagnostics["all_players_theta_window_converged"] = float(all(value > 0.5 for value in player_theta_converged))
        diagnostics["all_players_component_sigma_converged"] = float(all(value > 0.5 for value in player_sigma_converged))
        diagnostics["all_players_search_converged"] = float(
            diagnostics["all_players_cost_window_converged"] > 0.5
            or diagnostics["all_players_theta_window_converged"] > 0.5
            or diagnostics["all_players_component_sigma_converged"] > 0.5
        )
        return diagnostics

    def _early_stop_trigger(self, iteration_count: int, diagnostics: Mapping[str, float]) -> Optional[str]:
        if not bool(self.config.early_stop):
            return None
        if int(iteration_count) < max(int(self.config.min_iterations), 0):
            return None
        if float(diagnostics.get("selected_joint_feasible", 0.0)) <= 0.5:
            return None
        joint_theta_ok = float(diagnostics.get("joint_theta_window_converged", 0.0)) > 0.5
        joint_cost_ok = float(diagnostics.get("joint_cost_window_converged", 0.0)) > 0.5
        player_search_ok = float(diagnostics.get("all_players_search_converged", 0.0)) > 0.5
        if bool(self.config.nash_check):
            nash_ok = float(diagnostics.get("nash_converged", 0.0)) > 0.5
            if nash_ok and joint_theta_ok:
                return "nash_joint_theta"
            if nash_ok and joint_cost_ok and player_search_ok:
                return "nash_joint_cost_player_search"
            return None
        if joint_theta_ok and player_search_ok:
            return "joint_theta_player_search"
        if joint_cost_ok and player_search_ok:
            return "joint_cost_player_search"
        return None

    def _selected_joint_feasibility(
        self,
        problem: GameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
        player_names: list[str],
    ) -> dict[str, float]:
        diagnostics = {}
        feasible_flags = []
        try:
            joint = problem.decode_joint(selected_parameters)
            costs = dict(problem.evaluate_joint(joint))
        except Exception:
            costs = {}
        for name in player_names:
            cost = costs.get(name)
            feasible = bool(cost.feasible) if cost is not None else False
            total = float(cost.total) if cost is not None else float("inf")
            diagnostics[f"{name}_selected_feasible"] = float(feasible)
            diagnostics[f"{name}_selected_total"] = total
            feasible_flags.append(feasible)
        diagnostics["selected_joint_feasible"] = float(all(feasible_flags) if feasible_flags else False)
        return diagnostics

    @staticmethod
    def _selected_feasibility_not_evaluated(player_names: list[str]) -> dict[str, float]:
        diagnostics: dict[str, float] = {"selected_joint_feasible": 0.0}
        for name in player_names:
            diagnostics[f"{name}_selected_feasible"] = 0.0
            diagnostics[f"{name}_selected_total"] = float("inf")
        return diagnostics

    def _joint_stop_diagnostics(
        self,
        selected_unit_history: list[np.ndarray],
        selected_cost_history: list[np.ndarray],
    ) -> dict[str, float]:
        window = max(int(self.config.convergence_window), 1)
        theta_shift = float("inf")
        cost_change = float("inf")

        if len(selected_unit_history) >= window:
            theta_matrix = np.asarray(selected_unit_history[-window:], dtype=float)
            if theta_matrix.ndim == 2 and theta_matrix.shape[0] >= 2 and theta_matrix.shape[1] > 0:
                deltas = np.linalg.norm(np.diff(theta_matrix, axis=0), axis=1) / np.sqrt(float(theta_matrix.shape[1]))
                theta_shift = float(np.max(deltas)) if deltas.size else 0.0

        if len(selected_cost_history) >= window:
            costs = np.asarray(selected_cost_history[-window:], dtype=float)
            if costs.ndim == 2 and costs.shape[0] >= 2 and costs.shape[1] > 0:
                previous = costs[0]
                current = costs[-1]
                finite = np.isfinite(previous) & np.isfinite(current)
                if np.any(finite):
                    denom = np.maximum(np.abs(previous[finite]), 1.0)
                    cost_change = float(np.max(np.abs(current[finite] - previous[finite]) / denom))

        return {
            "joint_theta_window_shift": float(theta_shift),
            "joint_cost_window_change": float(cost_change),
            "joint_theta_window_converged": float(
                np.isfinite(theta_shift) and theta_shift <= float(self.config.joint_theta_window_tol)
            ),
            "joint_cost_window_converged": float(
                np.isfinite(cost_change) and cost_change <= float(self.config.joint_cost_window_tol)
            ),
        }

    def _approximate_nash_diagnostics(
        self,
        problem: GameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
        candidate_sets: Mapping[str, np.ndarray],
        player_names: list[str],
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        regrets = []
        finite_flags = []
        for name in player_names:
            candidates = np.asarray(candidate_sets.get(name, np.empty((0,))), dtype=float)
            if candidates.ndim == 1:
                candidates = candidates.reshape(1, -1)
            if candidates.size == 0:
                diagnostics[f"{name}_nash_current_cost"] = float("inf")
                diagnostics[f"{name}_best_response_cost"] = float("inf")
                diagnostics[f"{name}_nash_regret"] = float("inf")
                diagnostics[f"{name}_best_response_index"] = -1.0
                regrets.append(float("inf"))
                finite_flags.append(False)
                continue
            sample_count = int(candidates.shape[0])
            unilateral_samples = {}
            for other_name in player_names:
                if other_name == name:
                    unilateral_samples[other_name] = candidates
                else:
                    fixed = np.asarray(selected_parameters[other_name], dtype=float).reshape(1, -1)
                    unilateral_samples[other_name] = np.repeat(fixed, sample_count, axis=0)
            values = self._evaluate_joint_samples(problem, unilateral_samples)["values"]
            own_values = np.asarray(values[name], dtype=float).reshape(-1)
            if own_values.size == 0:
                current_cost = float("inf")
                best_response_cost = float("inf")
                best_response_index = -1
            else:
                current_cost = float(own_values[0])
                finite = np.isfinite(own_values)
                if np.any(finite):
                    finite_indices = np.where(finite)[0]
                    local = int(finite_indices[np.argmin(own_values[finite_indices])])
                    best_response_index = local
                    best_response_cost = float(own_values[local])
                else:
                    best_response_index = -1
                    best_response_cost = float("inf")
            regret = max(0.0, current_cost - best_response_cost) if np.isfinite(current_cost) else float("inf")
            normalized_regret = regret / max(abs(current_cost), 1.0) if np.isfinite(regret) else float("inf")
            diagnostics[f"{name}_nash_current_cost"] = float(current_cost)
            diagnostics[f"{name}_best_response_cost"] = float(best_response_cost)
            diagnostics[f"{name}_nash_regret"] = float(normalized_regret)
            diagnostics[f"{name}_best_response_index"] = float(best_response_index)
            regrets.append(float(normalized_regret))
            finite_flags.append(bool(np.isfinite(normalized_regret)))
        max_regret = float(np.max(regrets)) if regrets else float("inf")
        diagnostics["max_nash_regret"] = max_regret
        diagnostics["nash_converged"] = float(
            all(finite_flags) and max_regret <= float(self.config.nash_regret_tol)
        )
        return diagnostics

    @staticmethod
    def _nash_diagnostics_not_evaluated(player_names: list[str]) -> dict[str, float]:
        diagnostics: dict[str, float] = {
            "max_nash_regret": float("inf"),
            "nash_converged": 0.0,
        }
        for name in player_names:
            diagnostics[f"{name}_nash_current_cost"] = float("inf")
            diagnostics[f"{name}_best_response_cost"] = float("inf")
            diagnostics[f"{name}_nash_regret"] = float("inf")
            diagnostics[f"{name}_best_response_index"] = -1.0
        return diagnostics

    def _nash_candidate_matrix(
        self,
        player_name: str,
        selected_parameters: Mapping[str, np.ndarray],
        current_samples: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        anchors: Optional[np.ndarray],
        state: _DistributionState,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        low, high = bounds
        selected = np.clip(np.asarray(selected_parameters[player_name], dtype=float), low, high)
        rows = [selected]

        perturb = max(float(self.config.nash_perturbation), 0.0)
        if perturb > 0.0:
            selected_unit = np.clip(normalize_parameters(selected, bounds), 0.0, 1.0)
            for dim in range(int(selected_unit.size)):
                for sign in (-1.0, 1.0):
                    unit = selected_unit.copy()
                    unit[dim] = np.clip(unit[dim] + sign * perturb, 0.0, 1.0)
                    rows.append(denormalize_parameters(unit, bounds))

        for component in state.components:
            rows.append(denormalize_parameters(np.clip(component.mean, 0.0, 1.0), bounds))

        current = self._ranked_rows(current_samples, current_values)
        if current.size:
            rows.extend(current)

        anchor_values = self._clip_positions_to_bounds(
            np.empty((0, low.size), dtype=float) if anchors is None else np.asarray(anchors, dtype=float),
            bounds,
        )
        if anchor_values.size:
            rows.extend(anchor_values)

        matrix = self._dedupe_rows(np.asarray(rows, dtype=float), bounds)
        limit = max(int(self.config.nash_candidate_limit), 1)
        if matrix.shape[0] <= limit:
            return matrix
        return matrix[:limit]

    @staticmethod
    def _ranked_rows(samples: Optional[np.ndarray], values: Optional[np.ndarray]) -> np.ndarray:
        if samples is None:
            return np.empty((0, 0), dtype=float)
        rows = np.asarray(samples, dtype=float)
        if rows.size == 0:
            return np.empty((0, rows.shape[-1] if rows.ndim else 0), dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if values is None:
            return rows
        raw_values = np.asarray(values, dtype=float).reshape(-1)
        n = min(rows.shape[0], raw_values.size)
        if n <= 0:
            return rows
        finite_values = np.where(np.isfinite(raw_values[:n]), raw_values[:n], np.inf)
        order = np.argsort(finite_values)
        return rows[:n][order]

    @staticmethod
    def _dedupe_rows(values: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        low, high = bounds
        matrix = np.asarray(values, dtype=float)
        if matrix.size == 0:
            return np.empty((0, low.size), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        rows = []
        seen = set()
        for row in matrix:
            clipped = np.clip(np.asarray(row, dtype=float), low, high)
            key = tuple(np.round(clipped, 8).tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(clipped)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _joint_unit_vector(
        parameters: Mapping[str, np.ndarray],
        bounds: Mapping[str, tuple[np.ndarray, np.ndarray]],
        player_names: list[str],
    ) -> np.ndarray:
        chunks = []
        for name in player_names:
            chunks.append(np.clip(normalize_parameters(np.asarray(parameters[name], dtype=float), bounds[name]), 0.0, 1.0))
        return np.concatenate(chunks) if chunks else np.empty((0,), dtype=float)

    @staticmethod
    def _selection_cost_vector(selection: Mapping[str, object], player_names: list[str]) -> np.ndarray:
        values = []
        for idx, name in enumerate(player_names):
            key = "selection_ego_value" if idx == 0 else f"selection_{name}_value"
            values.append(float(selection.get(key, float("inf"))))
        return np.asarray(values, dtype=float)

    @staticmethod
    def _joint_rank_merit(values: Mapping[str, np.ndarray], player_names: list[str]) -> np.ndarray:
        sample_count = min(np.asarray(values[name]).size for name in player_names)
        merit = np.zeros((sample_count,), dtype=float)
        for name in player_names:
            raw = np.asarray(values[name], dtype=float)[:sample_count]
            finite_raw = np.where(np.isfinite(raw), raw, np.inf)
            order = np.argsort(finite_raw)
            ranks = np.empty((sample_count,), dtype=float)
            ranks[order] = np.linspace(0.0, 1.0, num=sample_count, dtype=float)
            merit += ranks
        return merit / max(float(len(player_names)), 1.0)

    def _select_global_joint_candidate(
        self,
        all_positions: Mapping[str, list[np.ndarray]],
        all_values: Mapping[str, list[np.ndarray]],
        player_names: list[str],
    ) -> tuple[dict[str, np.ndarray], dict]:
        positions = {
            name: np.vstack(all_positions[name]) if all_positions.get(name) else np.empty((0, 0), dtype=float)
            for name in player_names
        }
        values = {
            name: np.concatenate(all_values[name]) if all_values.get(name) else np.empty((0,), dtype=float)
            for name in player_names
        }
        sample_count = min((values[name].size for name in player_names), default=0)
        if sample_count <= 0:
            raise RuntimeError("No evaluated game samples are available for selection.")
        finite = np.ones((sample_count,), dtype=bool)
        for name in player_names:
            finite &= np.isfinite(values[name][:sample_count])
        if not np.any(finite):
            idx = 0
            return {name: positions[name][idx].copy() for name in player_names}, {
                "selection_index": int(idx),
                "selection_mode": "fallback_first_no_finite",
                "selection_merit": float("inf"),
            }

        ranks = {name: self._normalized_ranks(values[name][:sample_count]) for name in player_names}
        ego_name = player_names[0]
        opponent_names = player_names[1:]
        candidate_mask = finite.copy()
        for name in opponent_names:
            candidate_mask &= ranks[name] <= float(self.config.opponent_rank_gate)

        if np.any(candidate_mask):
            candidate_indices = np.where(candidate_mask)[0]
            ego_values = values[ego_name][candidate_indices]
            ego_ranks = ranks[ego_name][candidate_indices]
            tie_break = np.sum(np.vstack([ranks[name][candidate_indices] for name in player_names]), axis=0)
            order = np.lexsort((tie_break, ego_ranks, ego_values))
            idx = int(candidate_indices[order[0]])
            mode = "ego_min_with_opponent_rank_gate"
        else:
            candidate_indices = np.where(finite)[0]
            merit = np.sum(np.vstack([ranks[name][candidate_indices] for name in player_names]), axis=0)
            order = np.lexsort((values[ego_name][candidate_indices], merit))
            idx = int(candidate_indices[order[0]])
            mode = "fallback_min_sum_rank"

        selection_merit = float(np.sum([ranks[name][idx] for name in player_names]) / max(len(player_names), 1))
        diagnostics = {
            "selection_index": int(idx),
            "selection_mode": mode,
            "selection_merit": selection_merit,
            "selection_ego_value": float(values[ego_name][idx]),
            "selection_ego_rank": float(ranks[ego_name][idx]),
        }
        for name in opponent_names:
            diagnostics[f"selection_{name}_value"] = float(values[name][idx])
            diagnostics[f"selection_{name}_rank"] = float(ranks[name][idx])
        return {name: positions[name][idx].copy() for name in player_names}, diagnostics

    @staticmethod
    def _normalized_ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        n = int(values.size)
        ranks = np.ones((n,), dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return ranks
        finite_indices = np.where(finite)[0]
        order = finite_indices[np.argsort(values[finite_indices])]
        denom = max(order.size - 1, 1)
        ranks[order] = np.arange(order.size, dtype=float) / float(denom)
        return ranks

    @staticmethod
    def _component_counts(weights: np.ndarray, total_samples: int) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        total_samples = max(int(total_samples), int(weights.size))
        raw = weights * float(total_samples)
        counts = np.maximum(np.floor(raw).astype(int), 1)
        while int(np.sum(counts)) < total_samples:
            counts[int(np.argmax(raw - counts))] += 1
        while int(np.sum(counts)) > total_samples:
            reducible = np.where(counts > 1)[0]
            if reducible.size == 0:
                break
            idx = int(reducible[np.argmin(raw[reducible] - counts[reducible])])
            counts[idx] -= 1
        return counts

    @staticmethod
    def _rank_weights(count: int) -> np.ndarray:
        count = max(int(count), 1)
        ranks = np.arange(1, count + 1, dtype=float)
        weights = np.log(float(count) + 0.5) - np.log(ranks)
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return np.full((count,), 1.0 / float(count), dtype=float)
        return weights / total

    @staticmethod
    def _as_anchor_matrix(initial_population: np.ndarray) -> np.ndarray:
        anchors = np.asarray(initial_population, dtype=float)
        if anchors.ndim == 1:
            return anchors.reshape(1, -1)
        if anchors.ndim != 2:
            raise ValueError(f"initial_population must be 1D or 2D, got shape {anchors.shape}")
        return anchors

    @staticmethod
    def _clip_positions_to_bounds(positions: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.size == 0:
            return np.empty((0, bounds[0].size), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        low, high = bounds
        return np.clip(values, low[None, :], high[None, :])
