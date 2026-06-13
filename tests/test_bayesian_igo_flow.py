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


class BayesianIGOFlowTest(unittest.TestCase):
    def test_multiple_physical_actors_update_in_one_synchronous_flow(self):
        planning_problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=1.0),
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
                lower_bound=np.array([0.0], dtype=float),
                upper_bound=np.array([1.0], dtype=float),
                initial_population=np.array([[0.1], [0.5], [0.9]], dtype=float),
            )

        ego = player("ego")
        actor_a = {"yielding": player("actor_a:yielding"), "aggressive": player("actor_a:aggressive")}
        actor_b = {"yielding": player("actor_b:yielding"), "aggressive": player("actor_b:aggressive")}
        optima = {
            "ego": 0.2,
            "actor_a:yielding": 0.35,
            "actor_a:aggressive": 0.75,
            "actor_b:yielding": 0.45,
            "actor_b:aggressive": 0.85,
        }

        def evaluate(parameters):
            return BayesianBatchEvaluation(
                player_values={
                    name: (np.asarray(values)[:, 0] - optima[name]) ** 2
                    for name, values in parameters.items()
                }
            )

        def unilateral(player_name, candidates, fixed):
            del fixed
            return (np.asarray(candidates)[:, 0] - optima[player_name]) ** 2

        game_problem = BayesianGameOptimizationProblem(
            ego_player=ego,
            physical_actors={
                "actor_a": BayesianPhysicalActor("actor_a", actor_a, {"yielding": 0.5, "aggressive": 0.5}),
                "actor_b": BayesianPhysicalActor("actor_b", actor_b, {"yielding": 0.5, "aggressive": 0.5}),
            },
            evaluate_players_batch=evaluate,
            evaluate_unilateral_batch=unilateral,
            decode_joint=lambda _: JointTrajectory(trajectories={}),
            evaluate_joint=lambda _: {"ego": CostResult(0.0, CostBreakdown({}))},
        )
        optimizer = BayesianIGOOptimizer(
            BayesianIGOConfig(
                n_components=1,
                n_samples=32,
                n_iterations=30,
                init_std=0.3,
                min_std=0.01,
                early_stop=False,
                nash_check=False,
                seed=3,
            )
        )
        result = optimizer.optimize(game_problem)

        for player_name, optimum in optima.items():
            self.assertAlmostEqual(result.best_parameters[player_name][0], optimum, places=3)
        self.assertEqual(result.metadata["physical_players"], ("ego", "actor_a", "actor_b"))
        self.assertEqual(result.metadata["best_response_feedback_count"], 0)
        self.assertEqual(result.metadata["flow_batch_evaluations"], 30)


if __name__ == "__main__":
    unittest.main()
