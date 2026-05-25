from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EgoState:
    s: float
    l: float
    s_v: float
    l_v: float = 0.0
    s_a: float = 0.0
    l_a: float = 0.0
    yaw: Optional[float] = None
    t: float = 0.0


@dataclass(frozen=True)
class RoadBoundary:
    left_l: float
    right_l: float


@dataclass(frozen=True)
class ActorPrediction:
    actor_id: str
    actor_type: str
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    length: float
    width: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningProblem:
    ego: EgoState
    ref_path: Any
    road_boundary: RoadBoundary
    horizon: float
    dt: float
    actors: Sequence[ActorPrediction] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    t: np.ndarray
    s: np.ndarray
    l: np.ndarray
    s_v: Optional[np.ndarray] = None
    l_v: Optional[np.ndarray] = None
    s_a: Optional[np.ndarray] = None
    l_a: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    yaw: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    a: Optional[np.ndarray] = None
    kappa: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostBreakdown:
    terms: Mapping[str, float]
    hard_violation: bool = False


@dataclass(frozen=True)
class CostResult:
    total: float
    breakdown: CostBreakdown
    feasible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationProblem:
    objective: Callable[[np.ndarray], float]
    initial_population: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best_position: np.ndarray
    best_value: float
    population: np.ndarray
    values: np.ndarray
    history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PlannerResult:
    trajectory: Optional[Trajectory]
    cost: Optional[CostResult]
    status: str
    candidates: Sequence[Trajectory] = field(default_factory=tuple)
    optimization: Optional[OptimizationResult] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
