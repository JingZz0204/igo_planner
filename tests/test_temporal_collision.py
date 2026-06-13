import unittest

import numpy as np

from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary, Trajectory
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig


def actor_with_temporal_s(s_values):
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    s = np.asarray(s_values, dtype=float)
    half_length = 2.0
    half_width = 1.0
    return ActorPrediction(
        actor_id="moving_actor",
        actor_type="vehicle",
        times=t,
        x=s.copy(),
        y=np.zeros_like(s),
        yaw=np.zeros_like(s),
        length=2.0 * half_length,
        width=2.0 * half_width,
        metadata={
            "s": float(s[0]),
            "l": 0.0,
            "temporal_blocked_range": {
                "t": t,
                "s_min": s - half_length,
                "s_max": s + half_length,
                "l_min": np.full_like(s, -half_width),
                "l_max": np.full_like(s, half_width),
            },
        },
    )


class TemporalCollisionTest(unittest.TestCase):
    def setUp(self):
        self.cost = ParametricTrajectoryCost(
            ParametricTrajectoryCostConfig(trajectory_certificate_enabled=False)
        )
        self.trajectory = Trajectory(
            t=np.array([0.0, 1.0, 2.0], dtype=float),
            s=np.array([0.0, 10.0, 20.0], dtype=float),
            l=np.zeros((3,), dtype=float),
        )

    def collision_overlap(self, actor):
        problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=10.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=10.0, right_l=-10.0),
            horizon=2.0,
            dt=1.0,
            actors=(actor,),
        )
        blocked = self.cost._blocked_ranges(problem)
        scalar = self.cost._collision_running_terms(self.trajectory, blocked)["collision_overlap"]
        batch = self.cost._lattice_batch_collision_terms(
            self.trajectory.s.reshape(1, -1),
            self.trajectory.l.reshape(1, -1),
            self.trajectory.t,
            blocked,
        )["collision_overlap"][0]
        np.testing.assert_array_equal(scalar, batch)
        return scalar

    def test_same_spatial_path_at_different_times_is_not_collision(self):
        overlap = self.collision_overlap(actor_with_temporal_s([20.0, 20.0, 10.0]))
        self.assertEqual(float(np.max(overlap)), 0.0)

    def test_same_time_overlap_is_collision(self):
        overlap = self.collision_overlap(actor_with_temporal_s([20.0, 10.0, 0.0]))
        self.assertEqual(overlap.tolist(), [0.0, 1.0, 0.0])

    def test_final_sample_overlap_is_detected(self):
        overlap = self.collision_overlap(actor_with_temporal_s([40.0, 30.0, 20.0]))
        self.assertEqual(overlap.tolist(), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
