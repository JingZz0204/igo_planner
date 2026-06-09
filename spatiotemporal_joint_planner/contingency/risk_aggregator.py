from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskAggregatorConfig:
    expected_weight: float = 1.0
    cvar_weight: float = 0.35
    cvar_alpha: float = 0.25


class RiskAggregator:
    """Aggregate type-conditioned costs into one implicit-contingency objective."""

    def __init__(self, config: RiskAggregatorConfig | None = None):
        self.config = config or RiskAggregatorConfig()

    def aggregate(self, costs: np.ndarray, probabilities: np.ndarray) -> float:
        values = self.aggregate_batch(np.asarray(costs, dtype=float).reshape(-1, 1), probabilities)
        return float(values[0])

    def aggregate_batch(self, costs: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(costs, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(f"costs must have shape (types, samples), got {values.shape}")
        probabilities = self._normalized_probabilities(probabilities, values.shape[0])
        finite_values = np.where(np.isfinite(values), values, np.finfo(float).max / 1e6)
        expected = np.sum(probabilities[:, None] * finite_values, axis=0)
        cvar = self._upper_tail_cvar(finite_values, probabilities, float(self.config.cvar_alpha))
        return (
            float(self.config.expected_weight) * expected
            + float(self.config.cvar_weight) * cvar
        )

    def diagnostics(self, costs: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
        values = np.asarray(costs, dtype=float).reshape(-1)
        probabilities = self._normalized_probabilities(probabilities, values.size)
        finite_values = np.where(np.isfinite(values), values, np.finfo(float).max / 1e6)
        expected = float(np.sum(probabilities * finite_values))
        cvar = float(self._upper_tail_cvar(finite_values.reshape(-1, 1), probabilities, self.config.cvar_alpha)[0])
        return {
            "contingency_expected_cost": expected,
            "contingency_cvar_cost": cvar,
            "contingency_aggregate_cost": float(
                float(self.config.expected_weight) * expected + float(self.config.cvar_weight) * cvar
            ),
        }

    @staticmethod
    def _normalized_probabilities(probabilities: np.ndarray, count: int) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if values.size != int(count):
            raise ValueError(f"probability count {values.size} does not match cost type count {count}")
        values = np.maximum(values, 0.0)
        total = float(np.sum(values))
        if total <= 1e-12:
            return np.full((count,), 1.0 / max(count, 1), dtype=float)
        return values / total

    @staticmethod
    def _upper_tail_cvar(costs: np.ndarray, probabilities: np.ndarray, alpha: float) -> np.ndarray:
        values = np.asarray(costs, dtype=float)
        alpha = float(np.clip(alpha, 1e-6, 1.0))
        order = np.argsort(values, axis=0)[::-1]
        result = np.zeros((values.shape[1],), dtype=float)
        for sample in range(values.shape[1]):
            remaining = alpha
            weighted = 0.0
            for type_index in order[:, sample]:
                take = min(float(probabilities[int(type_index)]), remaining)
                if take > 0.0:
                    weighted += take * float(values[int(type_index), sample])
                    remaining -= take
                if remaining <= 1e-12:
                    break
            if remaining > 1e-12:
                weighted += remaining * float(values[int(order[-1, sample]), sample])
            result[sample] = weighted / alpha
        return result
