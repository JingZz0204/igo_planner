from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spatiotemporal_joint_planner.common import OptimizationProblem, OptimizationResult
from spatiotemporal_joint_planner.optimizer.base import Optimizer


def normalize_parameters(parameters: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    low, high = bounds
    return (np.asarray(parameters, dtype=float) - low) / np.maximum(high - low, 1e-12)


def denormalize_parameters(unit_parameters: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    low, high = bounds
    return low + np.asarray(unit_parameters, dtype=float) * (high - low)


@dataclass(frozen=True)
class CMAESConfig:
    n_components: int = 4
    n_samples: int = 64
    n_iterations: int = 10
    elite_fraction: float = 0.25
    init_std: float = 0.22
    min_std: float = 0.035
    max_std: float = 0.45
    weight_floor: float = 1e-3
    weight_lr: float = 0.55
    seed: int = 0
    early_stop: bool = True
    min_iterations: int = 3
    convergence_window: int = 5
    cost_window_tol: float = 1e-3
    theta_window_tol: float = 2e-2
    component_sigma_tol: float = 0.08
    component_weight_tol: float = 0.15


@dataclass
class _CMAComponentState:
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    p_sigma: np.ndarray
    p_c: np.ndarray


@dataclass
class _MixtureState:
    weights: np.ndarray
    components: list[_CMAComponentState]


class CMAESOptimizer(Optimizer):
    """Multi-modal CMA-ES optimizer in normalized parameter space."""

    def __init__(self, config: Optional[CMAESConfig] = None):
        self.config = config or CMAESConfig()
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def name(self) -> str:
        return "cma_es"

    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        anchors = self._as_anchor_matrix(problem.initial_population)
        bounds = self._resolve_bounds(problem, anchors)

        best_position = None
        best_value = float("inf")
        candidate_positions = np.empty((0, bounds[0].size), dtype=float)
        candidate_values = np.empty((0,), dtype=float)
        history = []
        best_value_history = []
        best_unit_history = []
        stop_reason = "max_iterations"
        stop_diagnostics = {}

        anchor_positions = self._clip_positions_to_bounds(anchors, bounds)
        if anchor_positions.size:
            anchor_values = self._evaluate_many(problem, anchor_positions)
            anchor_best_idx = int(np.argmin(anchor_values))
            if float(anchor_values[anchor_best_idx]) < best_value:
                best_value = float(anchor_values[anchor_best_idx])
                best_position = np.asarray(anchor_positions[anchor_best_idx], dtype=float).copy()
            anchors = self._ranked_seed_anchors(anchor_positions, anchor_values, int(self.config.n_components))

        state = self._initial_state(bounds, anchors)

        for iteration in range(max(int(self.config.n_iterations), 0)):
            unit_samples, component_ids = self._sample_population(state)
            positions = denormalize_parameters(unit_samples, bounds)
            values = self._evaluate_many(problem, positions)
            candidate_positions = positions
            candidate_values = values

            iter_best_idx = int(np.argmin(values))
            iter_best_value = float(values[iter_best_idx])
            if iter_best_value < best_value:
                best_value = iter_best_value
                best_position = np.asarray(positions[iter_best_idx], dtype=float).copy()

            state = self._update_mixture(state, unit_samples, component_ids, values, iteration)
            if best_position is not None:
                best_value_history.append(float(best_value))
                best_unit_history.append(np.clip(normalize_parameters(best_position, bounds), 0.0, 1.0))
            stop_diagnostics = self._early_stop_diagnostics(best_value_history, best_unit_history, state)
            early_stop_trigger = self._early_stop_trigger(iteration + 1, stop_diagnostics)
            should_stop = early_stop_trigger is not None
            if should_stop:
                stop_reason = f"early_stop:{early_stop_trigger}"
            history.append(
                self._history_item(
                    iteration=iteration,
                    state=state,
                    iteration_best_value=iter_best_value,
                    global_best_value=best_value,
                    diagnostics=stop_diagnostics,
                    stop_reason=stop_reason if should_stop else None,
                )
            )
            if should_stop:
                break

        if best_position is None:
            raise RuntimeError("CMA-ES did not evaluate any samples.")

        if candidate_positions.size == 0:
            population = best_position.reshape(1, -1)
            values = np.asarray([best_value], dtype=float)
        else:
            is_best_present = np.any(np.all(np.isclose(candidate_positions, best_position[None, :]), axis=1))
            if is_best_present:
                population = candidate_positions
                values = candidate_values
            else:
                population = np.vstack([best_position, candidate_positions])
                values = np.concatenate([[best_value], candidate_values])
        return OptimizationResult(
            best_position=best_position,
            best_value=best_value,
            population=population,
            values=values,
            history=history,
            metadata={
                "optimizer": self.name,
                "bounds": bounds,
                "n_components": int(self.config.n_components),
                "n_samples": int(self.config.n_samples),
                "n_iterations": int(self.config.n_iterations),
                "executed_iterations": int(len(history)),
                "stop_reason": stop_reason,
                "early_stop": bool(self.config.early_stop),
                "min_iterations": int(self.config.min_iterations),
                "convergence_window": int(self.config.convergence_window),
                "cost_window_tol": float(self.config.cost_window_tol),
                "theta_window_tol": float(self.config.theta_window_tol),
                "component_sigma_tol": float(self.config.component_sigma_tol),
                "component_weight_tol": float(self.config.component_weight_tol),
                "warm_start_seed_count": int(anchor_positions.shape[0]),
                "candidate_pool": "best_plus_final_iteration",
                **stop_diagnostics,
            },
        )

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def _initial_state(self, bounds: tuple[np.ndarray, np.ndarray], anchors: np.ndarray) -> _MixtureState:
        dim = int(bounds[0].size)
        n_components = max(int(self.config.n_components), 1)
        init_std = float(np.clip(self.config.init_std, self.config.min_std, self.config.max_std))

        seed_means = []
        for anchor in anchors:
            if anchor.shape == (dim,):
                seed_means.append(np.clip(normalize_parameters(anchor, bounds), 0.0, 1.0))

        while len(seed_means) < n_components:
            if seed_means:
                base = seed_means[len(seed_means) % len(seed_means)]
                seed_means.append(np.clip(base + self._rng.normal(0.0, 0.20, size=(dim,)), 0.0, 1.0))
            else:
                seed_means.append(self._rng.uniform(0.0, 1.0, size=(dim,)))

        components = [
            _CMAComponentState(
                mean=np.asarray(mean, dtype=float).copy(),
                sigma=init_std,
                covariance=np.eye(dim, dtype=float),
                p_sigma=np.zeros((dim,), dtype=float),
                p_c=np.zeros((dim,), dtype=float),
            )
            for mean in seed_means[:n_components]
        ]
        weights = np.full((n_components,), 1.0 / float(n_components), dtype=float)
        return _MixtureState(weights=weights, components=components)

    def _sample_population(self, state: _MixtureState) -> tuple[np.ndarray, np.ndarray]:
        total_samples = max(int(self.config.n_samples), len(state.components))
        counts = self._component_counts(state.weights, total_samples)
        samples = []
        component_ids = []

        for component_id, (component, count) in enumerate(zip(state.components, counts)):
            for sample in self._sample_component(component, int(count), include_mean=True):
                samples.append(sample)
                component_ids.append(component_id)

        return np.asarray(samples, dtype=float), np.asarray(component_ids, dtype=int)

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

    def _update_mixture(
        self,
        state: _MixtureState,
        samples: np.ndarray,
        component_ids: np.ndarray,
        values: np.ndarray,
        iteration: int,
    ) -> _MixtureState:
        new_components = []
        for component_id, component in enumerate(state.components):
            local_indices = np.where(component_ids == component_id)[0]
            if local_indices.size == 0:
                new_components.append(component)
                continue
            local_order = sorted(local_indices.tolist(), key=lambda idx: float(values[idx]))
            elite_n = max(1, min(len(local_order), int(np.ceil(len(local_order) * float(self.config.elite_fraction)))))
            selected = samples[local_order[:elite_n]]
            weights = self._recombination_weights(elite_n)
            new_components.append(self._update_component(component, selected, weights, iteration))

        return _MixtureState(
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
        order = sorted(range(len(values)), key=lambda idx: float(values[idx]))
        utility = np.zeros((len(values),), dtype=float)
        rank_weights = self._recombination_weights(elite_n)
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

    def _history_item(
        self,
        iteration: int,
        state: _MixtureState,
        iteration_best_value: float,
        global_best_value: float,
        diagnostics: dict,
        stop_reason: Optional[str],
    ) -> dict:
        return {
            "iteration": int(iteration),
            "best_value": float(iteration_best_value),
            "iteration_best_value": float(iteration_best_value),
            "global_best_value": float(global_best_value),
            "weights": state.weights.copy(),
            "mean_sigma": self._mean_sigma(state),
            "stop_reason": stop_reason,
            **diagnostics,
        }

    @staticmethod
    def _mean_sigma(state: _MixtureState) -> float:
        if not state.components:
            return 0.0
        sigmas = np.asarray([component.sigma for component in state.components], dtype=float)
        weights = np.asarray(state.weights, dtype=float).reshape(-1)
        if weights.size != sigmas.size or float(np.sum(weights)) <= 1e-12:
            return float(np.mean(sigmas))
        weights = weights / np.sum(weights)
        return float(np.sum(weights * sigmas))

    def _early_stop_diagnostics(
        self,
        best_value_history: list[float],
        best_unit_history: list[np.ndarray],
        state: _MixtureState,
    ) -> dict:
        cost_window_improvement = float("inf")
        theta_window_shift = float("inf")
        window = max(int(self.config.convergence_window), 1)

        if len(best_value_history) >= window and len(best_unit_history) >= window:
            previous_value = float(best_value_history[-window])
            current_value = float(best_value_history[-1])
            if np.isfinite(previous_value) and np.isfinite(current_value):
                denom = max(abs(previous_value), 1.0)
                cost_window_improvement = max(0.0, previous_value - current_value) / denom

            theta_window = [np.asarray(theta, dtype=float) for theta in best_unit_history[-window:]]
            if theta_window and all(theta.shape == theta_window[0].shape for theta in theta_window):
                theta_matrix = np.asarray(theta_window, dtype=float)
                if theta_matrix.ndim == 2 and theta_matrix.shape[1] > 0:
                    theta_deltas = np.linalg.norm(np.diff(theta_matrix, axis=0), axis=1) / np.sqrt(
                        float(theta_matrix.shape[1])
                    )
                    theta_window_shift = float(np.max(theta_deltas)) if theta_deltas.size else 0.0

        component_sigma = float("inf")
        component_weight = 0.0
        sigmas = np.asarray([component.sigma for component in state.components], dtype=float)
        weights = np.asarray(state.weights, dtype=float).reshape(-1)
        if sigmas.size and weights.shape == sigmas.shape:
            eligible = np.where(weights >= float(self.config.component_weight_tol))[0]
            if eligible.size:
                best_idx = int(eligible[np.argmin(sigmas[eligible])])
                component_sigma = float(sigmas[best_idx])
                component_weight = float(weights[best_idx])

        return {
            "cost_window_improvement": float(cost_window_improvement),
            "theta_window_shift": float(theta_window_shift),
            "qualified_component_sigma": float(component_sigma),
            "qualified_component_weight": float(component_weight),
            "cost_window_converged": float(
                np.isfinite(cost_window_improvement) and cost_window_improvement <= float(self.config.cost_window_tol)
            ),
            "theta_window_converged": float(
                np.isfinite(theta_window_shift) and theta_window_shift <= float(self.config.theta_window_tol)
            ),
            "component_sigma_converged": float(component_sigma <= float(self.config.component_sigma_tol)),
        }

    def _early_stop_trigger(self, iteration_count: int, diagnostics: dict) -> Optional[str]:
        if not bool(self.config.early_stop):
            return None
        if int(iteration_count) < max(int(self.config.min_iterations), 0):
            return None
        if float(diagnostics.get("cost_window_converged", 0.0)) > 0.5:
            return "cost_window"
        if float(diagnostics.get("theta_window_converged", 0.0)) > 0.5:
            return "theta_window"
        if float(diagnostics.get("component_sigma_converged", 0.0)) > 0.5:
            return "component_sigma"
        return None

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
    def _recombination_weights(mu: int) -> np.ndarray:
        mu = max(int(mu), 1)
        ranks = np.arange(1, mu + 1, dtype=float)
        weights = np.log(float(mu) + 0.5) - np.log(ranks)
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-12:
            return np.full((mu,), 1.0 / float(mu), dtype=float)
        return weights / total

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
    def _as_anchor_matrix(initial_population: np.ndarray) -> np.ndarray:
        anchors = np.asarray(initial_population, dtype=float)
        if anchors.ndim == 1:
            return anchors.reshape(1, -1)
        if anchors.ndim != 2:
            raise ValueError(f"initial_population must be 1D or 2D, got shape {anchors.shape}")
        return anchors

    @staticmethod
    def _resolve_bounds(problem: OptimizationProblem, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del anchors
        if problem.lower_bound is None or problem.upper_bound is None:
            raise ValueError("CMA-ES requires explicit lower_bound and upper_bound.")
        low = np.asarray(problem.lower_bound, dtype=float)
        high = np.asarray(problem.upper_bound, dtype=float)

        if low.shape != high.shape:
            raise ValueError(f"Bounds shape mismatch: low={low.shape}, high={high.shape}")
        span = high - low
        if np.any(span <= 1e-12):
            raise ValueError(f"CMA-ES bounds must have positive span in every dimension: low={low}, high={high}")
        return low.astype(float), high.astype(float)

    @staticmethod
    def _clip_positions_to_bounds(positions: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.size == 0:
            return np.empty((0, bounds[0].size), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        low, high = bounds
        return np.clip(values, low[None, :], high[None, :])

    @staticmethod
    def _ranked_seed_anchors(positions: np.ndarray, values: np.ndarray, max_count: int) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        values = np.asarray(values, dtype=float)
        if positions.size == 0 or values.size == 0:
            return positions
        finite = np.where(np.isfinite(values))[0]
        if finite.size == 0:
            return positions[: max(int(max_count), 1)]
        order = finite[np.argsort(values[finite])]
        return positions[order[: max(int(max_count), 1)]]

    @staticmethod
    def _evaluate(problem: OptimizationProblem, position: np.ndarray) -> float:
        value = float(problem.objective(np.asarray(position, dtype=float)))
        if not np.isfinite(value):
            raise FloatingPointError("CMA-ES scalar objective produced a non-finite value.")
        return value

    @classmethod
    def _evaluate_many(cls, problem: OptimizationProblem, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            return np.empty((0,), dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        if problem.objective_batch is not None:
            values = np.asarray(problem.objective_batch(positions), dtype=float).reshape(-1)
            expected_shape = (positions.shape[0],)
            if values.shape != expected_shape:
                raise ValueError(f"Batch objective shape mismatch: got {values.shape}, expected {expected_shape}.")
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("CMA-ES batch objective produced non-finite values.")
            return values

        return np.asarray([cls._evaluate(problem, position) for position in positions], dtype=float)
