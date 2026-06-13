import unittest

import numpy as np

from spatiotemporal_joint_planner.common import (
    ActorPrediction,
    EgoState,
    OptimizationProblem,
    PlanningProblem,
    RoadBoundary,
    Trajectory,
)
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost
from spatiotemporal_joint_planner.game.bayesian_game_parametric_planner import (
    BayesianGameParametricPlanner,
    BayesianGameParametricPlannerConfig,
)
from spatiotemporal_joint_planner.optimizer import CMAESConfig, CMAESOptimizer
from spatiotemporal_joint_planner.planner.warm_start import WarmStartContext, default_parametric_warm_start_generator
from spatiotemporal_joint_planner.trajectory_models import LatticeTrajectoryModel
from spatiotemporal_joint_planner.trajectory_models.base import TrajectoryModel


class _UnsupportedTrajectoryModel(TrajectoryModel):
    @property
    def name(self):
        return "unsupported_trajectory"

    def parameter_dim(self, problem):
        return 1

    def bounds(self, problem):
        return np.array([0.0]), np.array([1.0])

    def reference_parameters(self, problem):
        return np.array([0.5])

    def decode(self, parameters, problem):
        raise NotImplementedError


class StrictExecutionTest(unittest.TestCase):
    def test_cma_es_does_not_fallback_when_batch_objective_fails(self):
        problem = OptimizationProblem(
            objective=lambda _: 0.0,
            objective_batch=lambda _: (_ for _ in ()).throw(RuntimeError("batch path failed")),
            initial_population=np.array([[0.5]], dtype=float),
            lower_bound=np.array([0.0], dtype=float),
            upper_bound=np.array([1.0], dtype=float),
        )
        optimizer = CMAESOptimizer(CMAESConfig(n_components=1, n_samples=2, n_iterations=1, early_stop=False))
        with self.assertRaisesRegex(RuntimeError, "batch path failed"):
            optimizer.optimize(problem)

    def test_cma_es_rejects_non_finite_batch_costs(self):
        problem = OptimizationProblem(
            objective=lambda _: 0.0,
            objective_batch=lambda values: np.full((np.asarray(values).shape[0],), np.nan, dtype=float),
            initial_population=np.array([[0.5]], dtype=float),
            lower_bound=np.array([0.0], dtype=float),
            upper_bound=np.array([1.0], dtype=float),
        )
        optimizer = CMAESOptimizer(CMAESConfig(n_components=1, n_samples=2, n_iterations=1, early_stop=False))
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            optimizer.optimize(problem)

    def test_cma_es_candidate_pool_contains_only_best_and_final_iteration(self):
        anchors = np.linspace(0.0, 1.0, num=20, dtype=float).reshape(-1, 1)
        problem = OptimizationProblem(
            objective=lambda x: float((x[0] - 0.25) ** 2),
            objective_batch=lambda x: (np.asarray(x, dtype=float)[:, 0] - 0.25) ** 2,
            initial_population=anchors,
            lower_bound=np.array([0.0], dtype=float),
            upper_bound=np.array([1.0], dtype=float),
        )
        result = CMAESOptimizer(
            CMAESConfig(n_components=2, n_samples=4, n_iterations=1, early_stop=False)
        ).optimize(problem)
        self.assertEqual(result.metadata["warm_start_seed_count"], 20)
        self.assertEqual(result.metadata["candidate_pool"], "best_plus_final_iteration")
        self.assertLessEqual(result.population.shape[0], 5)

    def test_unknown_trajectory_model_has_no_generic_warm_start_fallback(self):
        planning_problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=1.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=2.0, right_l=-2.0),
            horizon=1.0,
            dt=0.1,
        )
        context = WarmStartContext(
            problem=planning_problem,
            trajectory_model=_UnsupportedTrajectoryModel(),
            lower_bound=np.array([0.0]),
            upper_bound=np.array([1.0]),
        )
        generator = default_parametric_warm_start_generator()
        self.assertFalse(generator.supports(context))
        with self.assertRaisesRegex(ValueError, "must match exactly one"):
            generator.generate(context)

    def test_invalid_simulated_type_is_rejected(self):
        planner = BayesianGameParametricPlanner(
            ego_trajectory_model=LatticeTrajectoryModel(),
            ego_cost_function=ParametricTrajectoryCost(),
            config=BayesianGameParametricPlannerConfig(simulated_target_type="unknown"),
        )
        with self.assertRaisesRegex(ValueError, "Invalid simulated actor type"):
            planner._simulated_target_type("target", ("yielding", "normal", "aggressive"))

    def test_parametric_cost_rejects_missing_required_trajectory_field(self):
        trajectory = Trajectory(
            t=np.array([0.0, 0.1], dtype=float),
            s=np.array([0.0, 1.0], dtype=float),
            l=np.zeros((2,), dtype=float),
            s_v=np.ones((2,), dtype=float),
            l_v=np.zeros((2,), dtype=float),
            l_a=np.zeros((2,), dtype=float),
            kappa=None,
        )
        problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=1.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=2.0, right_l=-2.0),
            horizon=1.0,
            dt=0.1,
        )
        with self.assertRaisesRegex(ValueError, r"requires trajectory\.kappa"):
            ParametricTrajectoryCost().evaluate(trajectory, problem)

    def test_parametric_cost_rejects_malformed_temporal_blocked_range(self):
        actor = ActorPrediction(
            actor_id="bad_actor",
            actor_type="vehicle",
            times=np.array([0.0, 0.1], dtype=float),
            x=np.zeros((2,), dtype=float),
            y=np.zeros((2,), dtype=float),
            yaw=np.zeros((2,), dtype=float),
            length=4.0,
            width=2.0,
            metadata={
                "temporal_blocked_range": {
                    "t": np.array([0.0, 0.1], dtype=float),
                    "s_min": np.array([0.0], dtype=float),
                    "s_max": np.array([4.0, 4.1], dtype=float),
                    "l_min": np.array([-1.0, -1.0], dtype=float),
                    "l_max": np.array([1.0, 1.0], dtype=float),
                }
            },
        )
        problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=1.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=2.0, right_l=-2.0),
            horizon=1.0,
            dt=0.1,
            actors=(actor,),
        )
        with self.assertRaisesRegex(ValueError, "equal non-zero lengths"):
            ParametricTrajectoryCost()._blocked_ranges(problem)


if __name__ == "__main__":
    unittest.main()
