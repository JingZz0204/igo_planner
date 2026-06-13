import unittest

import numpy as np

from spatiotemporal_joint_planner.belief import (
    ActorTypeBelief,
    BayesianTypeFilter,
    BayesianTypeFilterConfig,
    LongitudinalObservation,
    LongitudinalTypePrediction,
    default_actor_type_profiles,
)
from spatiotemporal_joint_planner.contingency import (
    BeliefContinuationEvaluator,
    RiskAggregator,
    RiskAggregatorConfig,
)


class BeliefFilterTest(unittest.TestCase):
    def test_each_synthetic_type_becomes_most_likely(self):
        profiles = default_actor_type_profiles()
        names = tuple(profile.name for profile in profiles)
        speeds = {"yielding": 7.0, "normal": 10.0, "aggressive": 13.0}
        for true_type in names:
            with self.subTest(true_type=true_type):
                belief = ActorTypeBelief(
                    "target",
                    {profile.name: profile.prior_probability for profile in profiles},
                )
                filter_ = BayesianTypeFilter(BayesianTypeFilterConfig())
                previous = None
                for step in range(15):
                    time = 0.25 * step
                    predictions = {
                        name: LongitudinalTypePrediction(
                            s=speeds[name] * time,
                            speed=speeds[name],
                            acceleration=0.0,
                            start_s=speeds[name] * max(time - 0.25, 0.0),
                            start_speed=speeds[name],
                        )
                        for name in names
                    }
                    observation = LongitudinalObservation(
                        time=time,
                        s=speeds[true_type] * time,
                        speed=speeds[true_type],
                        acceleration=0.0,
                    )
                    filter_.update(belief, observation, predictions, previous)
                    previous = observation
                self.assertGreater(belief.probabilities[true_type], 0.75, belief.probabilities)
                self.assertAlmostEqual(sum(belief.probabilities.values()), 1.0, places=12)

    def test_uninformative_observation_keeps_belief_close_to_prior(self):
        profiles = default_actor_type_profiles()
        prior = {profile.name: profile.prior_probability for profile in profiles}
        belief = ActorTypeBelief("target", prior)
        filter_ = BayesianTypeFilter(BayesianTypeFilterConfig())
        observation = LongitudinalObservation(time=0.0, s=0.0, speed=10.0, acceleration=0.0)
        predictions = {
            profile.name: LongitudinalTypePrediction(s=0.0, speed=10.0, acceleration=0.0)
            for profile in profiles
        }
        filter_.update(belief, observation, predictions)
        for name, probability in prior.items():
            self.assertLess(abs(belief.probabilities[name] - probability), 0.01, belief.probabilities)


class BeliefContinuationTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = BeliefContinuationEvaluator(
            RiskAggregator(RiskAggregatorConfig(expected_weight=1.0, cvar_weight=0.35, cvar_alpha=0.25))
        )
        self.prior = np.array([0.5, 0.5], dtype=float)
        self.different_costs = np.array([[10.0], [100.0]], dtype=float)

    def score(self, costs, features):
        return float(self.evaluator.evaluate_batch(costs, features, self.prior)["implicit_contingency_score"][0])

    def test_separable_types_have_zero_residual_uncertainty_cost(self):
        features = np.array([[[0.0, 0.0, 0.0]], [[100.0, 20.0, 10.0]]], dtype=float)
        self.assertAlmostEqual(self.score(self.different_costs, features), 0.0, places=10)

    def test_indistinguishable_types_with_different_costs_are_penalized(self):
        features = np.zeros((2, 1, 3), dtype=float)
        self.assertGreater(self.score(self.different_costs, features), 0.0)

    def test_indistinguishable_types_with_same_cost_have_zero_surcharge(self):
        features = np.zeros((2, 1, 3), dtype=float)
        same_costs = np.array([[10.0], [10.0]], dtype=float)
        self.assertAlmostEqual(self.score(same_costs, features), 0.0, places=10)

    def test_score_decreases_as_types_become_more_separable(self):
        scores = []
        for separation in (0.0, 1.0, 3.0, 10.0):
            features = np.array([[[0.0, 0.0, 0.0]], [[separation, 0.0, 0.0]]], dtype=float)
            scores.append(self.score(self.different_costs, features))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(scores, scores[1:])), scores)


if __name__ == "__main__":
    unittest.main()
