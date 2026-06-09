from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass
class ActorTypeBelief:
    """Normalized categorical belief over one actor's latent behavior type."""

    actor_id: str
    probabilities: dict[str, float]

    def __post_init__(self) -> None:
        self.probabilities = self._normalized(self.probabilities)

    def vector(self, type_names: tuple[str, ...]) -> np.ndarray:
        values = np.asarray([self.probabilities.get(name, 0.0) for name in type_names], dtype=float)
        total = float(np.sum(values))
        if total <= 1e-12:
            return np.full((len(type_names),), 1.0 / max(len(type_names), 1), dtype=float)
        return values / total

    def update(self, probabilities: Mapping[str, float]) -> None:
        self.probabilities = self._normalized(probabilities)

    @staticmethod
    def _normalized(probabilities: Mapping[str, float]) -> dict[str, float]:
        values = {str(name): max(float(value), 0.0) for name, value in probabilities.items()}
        total = float(sum(values.values()))
        if total <= 1e-12:
            count = max(len(values), 1)
            return {name: 1.0 / float(count) for name in values}
        return {name: value / total for name, value in values.items()}


@dataclass(frozen=True)
class BayesianTypeFilterConfig:
    position_sigma: float = 0.8
    speed_sigma: float = 0.5
    acceleration_sigma: float = 0.6
    displacement_sigma: float = 0.5
    speed_delta_sigma: float = 0.35
    observation_window_seconds: float = 1.5
    evidence_gain: float = 0.4
    probability_floor: float = 0.02
    forgetting_factor: float = 0.005


@dataclass(frozen=True)
class LongitudinalObservation:
    time: float
    s: float
    speed: float
    acceleration: float = 0.0


@dataclass(frozen=True)
class LongitudinalTypePrediction:
    s: float
    speed: float
    acceleration: float = 0.0
    start_s: Optional[float] = None
    start_speed: Optional[float] = None


class BayesianTypeFilter:
    """Fixed-window Bayesian filter using longitudinal motion evidence."""

    def __init__(self, config: BayesianTypeFilterConfig | None = None):
        self.config = config or BayesianTypeFilterConfig()
        self._base_prior: Optional[dict[str, float]] = None
        self._log_likelihood_window: deque[tuple[float, dict[str, float]]] = deque()

    def reset(self) -> None:
        self._base_prior = None
        self._log_likelihood_window.clear()

    def update(
        self,
        belief: ActorTypeBelief,
        observation: LongitudinalObservation,
        predictions: Mapping[str, LongitudinalTypePrediction],
        previous_observation: Optional[LongitudinalObservation] = None,
    ) -> dict[str, float]:
        names = tuple(belief.probabilities.keys())
        if not names:
            return {}

        if self._base_prior is None or set(self._base_prior) != set(names):
            self._base_prior = dict(belief.probabilities)
            self._log_likelihood_window.clear()

        position_sigma = max(float(self.config.position_sigma), 1e-3)
        speed_sigma = max(float(self.config.speed_sigma), 1e-3)
        acceleration_sigma = max(float(self.config.acceleration_sigma), 1e-3)
        displacement_sigma = max(float(self.config.displacement_sigma), 1e-3)
        speed_delta_sigma = max(float(self.config.speed_delta_sigma), 1e-3)
        evidence_gain = max(float(self.config.evidence_gain), 0.0)
        current_log_likelihoods: dict[str, float] = {}
        for name in names:
            predicted = predictions.get(name)
            if predicted is None:
                log_likelihood = 0.0
            else:
                errors = [
                    (float(observation.s) - float(predicted.s)) / position_sigma,
                    (float(observation.speed) - float(predicted.speed)) / speed_sigma,
                    (float(observation.acceleration) - float(predicted.acceleration)) / acceleration_sigma,
                ]
                if (
                    previous_observation is not None
                    and predicted.start_s is not None
                    and predicted.start_speed is not None
                ):
                    observed_displacement = float(observation.s) - float(previous_observation.s)
                    predicted_displacement = float(predicted.s) - float(predicted.start_s)
                    observed_speed_delta = float(observation.speed) - float(previous_observation.speed)
                    predicted_speed_delta = float(predicted.speed) - float(predicted.start_speed)
                    errors.extend(
                        [
                            (observed_displacement - predicted_displacement) / displacement_sigma,
                            (observed_speed_delta - predicted_speed_delta) / speed_delta_sigma,
                        ]
                    )
                log_likelihood = -0.5 * evidence_gain * float(np.dot(errors, errors))
            current_log_likelihoods[name] = float(log_likelihood)

        current_time = float(observation.time)
        self._log_likelihood_window.append((current_time, current_log_likelihoods))
        window_seconds = max(float(self.config.observation_window_seconds), 0.0)
        while self._log_likelihood_window and current_time - self._log_likelihood_window[0][0] > window_seconds:
            self._log_likelihood_window.popleft()

        log_weights = []
        for name in names:
            prior = max(float(self._base_prior.get(name, 0.0)), 1e-12)
            window_log_likelihood = sum(values.get(name, 0.0) for _, values in self._log_likelihood_window)
            log_weights.append(np.log(prior) + float(window_log_likelihood))

        weights = np.exp(np.asarray(log_weights, dtype=float) - float(np.max(log_weights)))
        weights /= max(float(np.sum(weights)), 1e-12)
        forgetting = float(np.clip(self.config.forgetting_factor, 0.0, 1.0))
        weights = (1.0 - forgetting) * weights + forgetting / float(len(names))
        floor = float(np.clip(self.config.probability_floor, 0.0, 1.0 / float(len(names))))
        weights = (1.0 - floor * float(len(names))) * weights + floor
        updated = {name: float(value) for name, value in zip(names, weights)}
        belief.update(updated)
        return dict(belief.probabilities)
