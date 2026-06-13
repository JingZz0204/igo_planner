import unittest

import numpy as np

from spatiotemporal_joint_planner.common import CostBreakdown, CostResult, EgoState, PlanningProblem, RoadBoundary
from spatiotemporal_joint_planner.game.base import GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.bayesian_igo_optimizer import (
    BayesianBatchEvaluation,
    BayesianGameOptimizationProblem,
    BayesianIGOConfig,
    BayesianIGOOptimizer,
    BayesianPhysicalActor,
)
from spatiotemporal_joint_planner.trajectory_models import LatticeTrajectoryModel
from spatiotemporal_joint_planner.validation import BayesianRegretOracle, BayesianRegretOracleConfig


class BayesianRegretOracleTest(unittest.TestCase):
    def _problem(self, initial=0.0):
        planning_problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=5.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=2.0, right_l=-2.0),
            horizon=1.0,
            dt=0.1,
        )
        model = LatticeTrajectoryModel()

        def player(name):
            return GamePlayer(
                name=name,
                role=name,
                problem=planning_problem,
                trajectory_model=model,
                lower_bound=np.array([0.0]),
                upper_bound=np.array([1.0]),
                initial_population=np.array([[initial]]),
            )

        ego = player("ego")
        targets = {"yielding": player("target:yielding"), "aggressive": player("target:aggressive")}
        optima = {"ego": 0.2, "target:yielding": 0.25, "target:aggressive": 0.75}

        def evaluate(parameters):
            return BayesianBatchEvaluation(
                player_values={
                    name: 100.0 * (np.asarray(values)[:, 0] - optima[name]) ** 2
                    for name, values in parameters.items()
                }
            )

        def unilateral(name, candidates, fixed):
            del fixed
            return 100.0 * (np.asarray(candidates)[:, 0] - optima[name]) ** 2

        return BayesianGameOptimizationProblem(
            ego_player=ego,
            physical_actors={
                "target": BayesianPhysicalActor("target", targets, {"yielding": 0.5, "aggressive": 0.5})
            },
            evaluate_players_batch=evaluate,
            evaluate_unilateral_batch=unilateral,
            decode_joint=lambda _: JointTrajectory(trajectories={}),
            evaluate_joint=lambda _: {"ego": CostResult(0.0, CostBreakdown({}))},
        )

    def test_oracle_finds_independent_unilateral_improvements(self):
        problem = self._problem()
        selected = {
            "ego": np.array([0.8]),
            "target:yielding": np.array([0.9]),
            "target:aggressive": np.array([0.1]),
        }
        report = BayesianRegretOracle(
            BayesianRegretOracleConfig(uniform_samples=128, axis_samples_per_dimension=21, seed=4)
        ).evaluate(problem, selected)
        self.assertGreater(report.ego.normalized_regret, 0.9)
        self.assertGreater(report.targets_by_type["target:yielding"].normalized_regret, 0.9)
        self.assertGreater(report.targets_by_type["target:aggressive"].normalized_regret, 0.9)
        self.assertLess(abs(report.ego.best_response_parameters[0] - 0.2), 0.06)

    def test_online_local_nash_check_finds_nearby_improvements(self):
        problem = self._problem(initial=0.3)
        players = problem.strategy_players
        optimizer = BayesianIGOOptimizer(
            BayesianIGOConfig(
                equilibrium_regret_tol=0.1,
                local_nash_samples=128,
                local_nash_perturbation=0.2,
                local_nash_seed=4,
            )
        )
        selected = {"ego": np.array([0.3]), "target:yielding": np.array([0.35]), "target:aggressive": np.array([0.65])}
        diagnostics = optimizer._local_nash_check(
            problem=problem,
            selected_parameters=selected,
            players=players,
            material_probabilities=problem.material_probabilities_by_player,
            check_index=1,
        )
        self.assertGreater(diagnostics["local_max_bayesian_regret"], 0.4)
        self.assertEqual(diagnostics["local_nash_converged"], 0.0)
        self.assertLessEqual(diagnostics["local_ego_candidate_count"], 129.0)


if __name__ == "__main__":
    unittest.main()
