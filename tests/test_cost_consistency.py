import unittest

from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig
from spatiotemporal_joint_planner.scenario import InteractiveLaneChangeScenario
from spatiotemporal_joint_planner.trajectory_models import (
    FrenetBezierTrajectoryModel,
    FrenetBSplineTrajectoryModel,
    FrenetViaBSplineTrajectoryModel,
    LatticeTrajectoryModel,
)
from spatiotemporal_joint_planner.validation import CostConsistencyConfig, evaluate_cost_consistency


class CostConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scenario = InteractiveLaneChangeScenario(route_length=800.0)
        cls.problem = scenario.build_problem(scenario.initial_state())
        cls.cost = ParametricTrajectoryCost(
            ParametricTrajectoryCostConfig(trajectory_certificate_enabled=False)
        )

    def test_lattice_scalar_batch_costs_are_close(self):
        result = evaluate_cost_consistency(
            LatticeTrajectoryModel(),
            self.cost,
            self.problem,
            CostConsistencyConfig(sample_count=48, seed=7, relative_tolerance=1e-2),
        )
        self.assertTrue(result.passed, result.as_dict())
        self.assertLess(result.max_relative_error, 1e-2, result.as_dict())

    def test_frenet_bspline_scalar_batch_costs_are_close(self):
        result = evaluate_cost_consistency(
            FrenetBSplineTrajectoryModel(),
            self.cost,
            self.problem,
            CostConsistencyConfig(sample_count=48, seed=7, relative_tolerance=1e-4),
        )
        self.assertTrue(result.passed, result.as_dict())
        self.assertLess(result.max_relative_error, 1e-4, result.as_dict())

    def test_other_frenet_scalar_batch_costs_are_close(self):
        for model in (FrenetBezierTrajectoryModel(), FrenetViaBSplineTrajectoryModel()):
            with self.subTest(model=model.name):
                result = evaluate_cost_consistency(
                    model,
                    self.cost,
                    self.problem,
                    CostConsistencyConfig(sample_count=32, seed=7, relative_tolerance=1e-6, absolute_tolerance=1e-4),
                )
                self.assertTrue(result.passed, result.as_dict())


if __name__ == "__main__":
    unittest.main()
