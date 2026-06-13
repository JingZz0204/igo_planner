import unittest
from unittest.mock import patch

import numpy as np

from spatiotemporal_joint_planner import demo
from spatiotemporal_joint_planner.application import (
    build_planner,
    build_scenario,
    build_trajectory_model,
    load_demo_config,
    load_scenario_config,
    resolve_demo_runtime_args,
)
from spatiotemporal_joint_planner.game.bayesian_game_parametric_planner import BayesianGameParametricPlanner


class MultiActorIntersectionTest(unittest.TestCase):
    def test_same_route_lane_change_does_not_project_game_batches_to_xy(self):
        args = demo.build_arg_parser().parse_args(
            [
                "--scenario",
                "interactive_lane_change",
                "--trajectory-model",
                "lattice_trajectory",
                "--set",
                "planner.type=bayesian_game",
                "--set",
                "optimizer.samples=3",
                "--set",
                "optimizer.iterations=1",
                "--set",
                "optimizer.nash_check=false",
                "--no-show",
            ]
        )
        args.demo_config_values = load_demo_config(args)
        args.scenario_config_values = load_scenario_config(args)
        resolve_demo_runtime_args(args)
        scenario = build_scenario(args)
        problem = scenario.build_problem(scenario.initial_state())
        planner = build_planner(args, build_trajectory_model(args))

        with patch.object(
            BayesianGameParametricPlanner,
            "_with_batch_xy",
            side_effect=AssertionError("same-route lane change must stay in SL"),
        ):
            result = planner.plan(problem)

        self.assertIsNotNone(result.trajectory)

    def test_unprotected_left_turn_game_uses_independent_conflict_routes(self):
        args = demo.build_arg_parser().parse_args(
            [
                "--scenario",
                "unprotected_left_turn",
                "--trajectory-model",
                "lattice_trajectory",
                "--set",
                "planner.type=bayesian_game",
                "--set",
                "optimizer.samples=3",
                "--set",
                "optimizer.iterations=1",
                "--set",
                "optimizer.nash_check=false",
                "--no-show",
            ]
        )
        args.demo_config_values = load_demo_config(args)
        args.scenario_config_values = load_scenario_config(args)
        resolve_demo_runtime_args(args)
        scenario = build_scenario(args)
        problem = scenario.build_problem(scenario.initial_state())
        planner = build_planner(args, build_trajectory_model(args))
        result = planner.plan(problem)

        self.assertIsNotNone(result.trajectory)
        self.assertEqual(set(result.metadata["game_actor_ids"]), {
            "oncoming_vehicle",
            "left_crossing_vehicle",
        })
        self.assertEqual(len(result.metadata["game_actor_trajectories"]), 2)
        self.assertEqual(result.metadata["game_metadata"]["physical_players"], (
            "ego",
            "oncoming_vehicle",
            "left_crossing_vehicle",
        ))
        self.assertIsNot(problem.actors[0].metadata["ref_path"], problem.ref_path)
        self.assertIsNot(problem.actors[0].metadata["ref_path"], problem.actors[1].metadata["ref_path"])
        ego_xy = np.asarray(problem.ref_path.sample_xy(0.1), dtype=float).T
        for actor in problem.actors:
            actor_xy = np.asarray(actor.metadata["ref_path"].sample_xy(0.1), dtype=float).T
            pairwise_distance = np.linalg.norm(ego_xy[:, None, :] - actor_xy[None, :, :], axis=2)
            self.assertLess(float(np.min(pairwise_distance)), 0.1)


if __name__ == "__main__":
    unittest.main()
