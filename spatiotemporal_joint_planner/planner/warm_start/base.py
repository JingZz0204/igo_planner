from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import PlanningProblem
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class WarmStartContext:
    problem: PlanningProblem
    trajectory_model: TrajectoryModel
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    previous_parameters: Optional[np.ndarray] = None
    max_count: int = 8

    @property
    def parameter_dim(self) -> int:
        return int(self.trajectory_model.parameter_dim(self.problem))


class WarmStartGenerator(ABC):
    """Base interface for optimizer anchors or SVGD initial particles."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable generator name for logging and debugging."""

    @abstractmethod
    def supports(self, context: WarmStartContext) -> bool:
        """Return whether this generator is applicable to the context."""

    @abstractmethod
    def generate(self, context: WarmStartContext) -> np.ndarray:
        """Return a 2D array of initial parameters or particles."""


class ModelWarmStartGenerator(WarmStartGenerator):
    """Select exactly one model-specific warm-start generator."""

    def __init__(self, generators: Sequence[WarmStartGenerator]):
        self.generators = list(generators)

    @property
    def name(self) -> str:
        return "model_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return sum(generator.supports(context) for generator in self.generators) == 1

    def generate(self, context: WarmStartContext) -> np.ndarray:
        matched = [generator for generator in self.generators if generator.supports(context)]
        if len(matched) != 1:
            names = [generator.name for generator in matched]
            raise ValueError(
                f"Trajectory model {context.trajectory_model.name!r} must match exactly one warm-start "
                f"generator, matched={names}"
            )
        generated = matched[0].generate(context)
        if generated.size == 0:
            raise ValueError(f"Warm-start generator {matched[0].name!r} returned no seeds.")
        return finalize_warm_starts(generated, context)


def default_parametric_warm_start_generator() -> WarmStartGenerator:
    from .bezier import BezierTrajectoryWarmStartGenerator
    from .frenet_bezier import FrenetBezierWarmStartGenerator
    from .frenet_bspline import FrenetBSplineWarmStartGenerator
    from .frenet_via_bspline import FrenetViaBSplineWarmStartGenerator
    from .terminal_state import SvgdParticleWarmStartGenerator, TerminalStateWarmStartGenerator

    return ModelWarmStartGenerator(
        [
            SvgdParticleWarmStartGenerator(),
            TerminalStateWarmStartGenerator(),
            BezierTrajectoryWarmStartGenerator(),
            FrenetBezierWarmStartGenerator(),
            FrenetBSplineWarmStartGenerator(),
            FrenetViaBSplineWarmStartGenerator(),
        ]
    )


def finalize_warm_starts(rows: Sequence[np.ndarray], context: WarmStartContext) -> np.ndarray:
    dim = context.parameter_dim
    low = np.asarray(context.lower_bound, dtype=float)
    high = np.asarray(context.upper_bound, dtype=float)
    if low.shape != (dim,) or high.shape != (dim,):
        raise ValueError(f"Warm start bounds shape mismatch for dim {dim}: low={low.shape}, high={high.shape}")

    output = []
    seen = set()
    for index, row in enumerate(rows):
        value = np.asarray(row, dtype=float).reshape(-1)
        if value.shape != (dim,):
            raise ValueError(
                f"Warm-start seed {index} for trajectory model {context.trajectory_model.name!r} "
                f"has shape {value.shape}, expected {(dim,)}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"Warm-start seed {index} for trajectory model {context.trajectory_model.name!r} "
                "contains non-finite values."
            )
        value = np.clip(value, low, high)
        key = tuple(np.round(value, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= int(context.max_count):
            break

    if not output:
        raise ValueError(f"No valid warm-start seeds for trajectory model {context.trajectory_model.name!r}.")
    return np.asarray(output, dtype=float)
