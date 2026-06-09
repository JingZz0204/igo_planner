from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatiotemporal_joint_planner.contingency.risk_aggregator import RiskAggregator


@dataclass(frozen=True)
class BeliefContinuationConfig:
    enabled: bool = True
    observation_time: float = 0.8
    position_sigma: float = 2.0
    speed_sigma: float = 1.0
    acceleration_sigma: float = 1.0
    score_scale: float = 500.0
    cost_weight: float = 100.0


class BeliefContinuationEvaluator:
    """Estimate the residual risk after a future type observation.

    The returned surcharge is zero when the observation makes the target type
    perfectly identifiable, and grows when costly type outcomes remain
    difficult to distinguish.
    """

    def __init__(
        self,
        risk_aggregator: RiskAggregator,
        config: BeliefContinuationConfig | None = None,
    ):
        self.risk_aggregator = risk_aggregator
        self.config = config or BeliefContinuationConfig()

    def evaluate_batch(
        self,
        type_costs: np.ndarray,
        type_features: np.ndarray,
        prior_probabilities: np.ndarray,
    ) -> dict[str, np.ndarray]:
        costs = np.asarray(type_costs, dtype=float)
        features = np.asarray(type_features, dtype=float)
        if costs.ndim == 1:
            costs = costs.reshape(-1, 1)
        if features.ndim == 2:
            features = features[:, None, :]
        if costs.ndim != 2 or features.ndim != 3:
            raise ValueError(
                f"Expected costs (types, samples) and features (types, samples, dims), "
                f"got {costs.shape} and {features.shape}"
            )
        if costs.shape != features.shape[:2]:
            raise ValueError(f"Cost/feature batch mismatch: {costs.shape} != {features.shape[:2]}")

        type_count, sample_count = costs.shape
        prior = RiskAggregator._normalized_probabilities(prior_probabilities, type_count)
        safe_costs = np.where(np.isfinite(costs), costs, np.finfo(float).max / 1e6)
        sigmas = np.asarray(
            [
                max(float(self.config.position_sigma), 1e-3),
                max(float(self.config.speed_sigma), 1e-3),
                max(float(self.config.acceleration_sigma), 1e-3),
            ],
            dtype=float,
        )
        if features.shape[2] != sigmas.size:
            raise ValueError(f"Expected {sigmas.size} continuation features, got {features.shape[2]}")

        posterior_risk = np.zeros((sample_count,), dtype=float)
        posterior_entropy = np.zeros((sample_count,), dtype=float)
        risk_factor = float(self.risk_aggregator.config.expected_weight) + float(
            self.risk_aggregator.config.cvar_weight
        )
        clairvoyant_risk = risk_factor * np.sum(prior[:, None] * safe_costs, axis=0)

        for true_type in range(type_count):
            residual = (features - features[true_type][None, :, :]) / sigmas[None, None, :]
            log_likelihood = -0.5 * np.sum(residual * residual, axis=2)
            likelihood = np.exp(log_likelihood - np.max(log_likelihood, axis=0, keepdims=True))
            posterior = prior[:, None] * likelihood
            posterior /= np.maximum(np.sum(posterior, axis=0, keepdims=True), 1e-12)
            posterior_risk += float(prior[true_type]) * self._aggregate_with_sample_probabilities(
                safe_costs,
                posterior,
            )
            posterior_entropy += float(prior[true_type]) * (
                -np.sum(posterior * np.log(np.maximum(posterior, 1e-12)), axis=0)
            )

        surcharge = np.maximum(posterior_risk - clairvoyant_risk, 0.0)
        scale = max(float(self.config.score_scale), 1e-9)
        score = 1.0 - np.exp(-surcharge / scale)
        weighted_cost = float(self.config.cost_weight) * score if bool(self.config.enabled) else np.zeros_like(score)
        return {
            "implicit_contingency_surcharge": surcharge,
            "implicit_contingency_score": score,
            "implicit_contingency_cost": weighted_cost,
            "expected_posterior_entropy": posterior_entropy,
        }

    def _aggregate_with_sample_probabilities(self, costs: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        expected = np.sum(probabilities * costs, axis=0)
        alpha = float(np.clip(self.risk_aggregator.config.cvar_alpha, 1e-6, 1.0))
        order = np.argsort(costs, axis=0)[::-1]
        sorted_costs = np.take_along_axis(costs, order, axis=0)
        sorted_probabilities = np.take_along_axis(probabilities, order, axis=0)
        cumulative_before = np.cumsum(sorted_probabilities, axis=0) - sorted_probabilities
        take = np.clip(alpha - cumulative_before, 0.0, sorted_probabilities)
        cvar = np.sum(take * sorted_costs, axis=0) / alpha
        return (
            float(self.risk_aggregator.config.expected_weight) * expected
            + float(self.risk_aggregator.config.cvar_weight) * cvar
        )
