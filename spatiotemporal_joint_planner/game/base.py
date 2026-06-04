from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import CostResult, PlanningProblem, Trajectory
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class GamePlayer:
    name: str
    role: str
    problem: PlanningProblem
    trajectory_model: TrajectoryModel
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    initial_population: np.ndarray


@dataclass(frozen=True)
class JointTrajectory:
    trajectories: Mapping[str, Trajectory]


@dataclass(frozen=True)
class GameOptimizationProblem:
    players: Sequence[GamePlayer]
    decode_joint: Callable[[Mapping[str, np.ndarray]], JointTrajectory]
    evaluate_joint: Callable[[JointTrajectory], Mapping[str, CostResult]]
    evaluate_joint_batch: Optional[Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class GameOptimizationResult:
    best_parameters: Mapping[str, np.ndarray]
    best_joint_trajectory: JointTrajectory
    player_costs: Mapping[str, CostResult]
    status: str
    history: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)
    player_populations: Mapping[str, np.ndarray] = field(default_factory=dict)
    player_values: Mapping[str, np.ndarray] = field(default_factory=dict)
    joint_merit: Optional[np.ndarray] = None
