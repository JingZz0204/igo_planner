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
    max_count: int = 32

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


class DefaultWarmStartGenerator(WarmStartGenerator):
    """Generic fallback: previous best, model reference, and bounds center."""

    @property
    def name(self) -> str:
        return "default_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return True

    def generate(self, context: WarmStartContext) -> np.ndarray:
        rows = []
        if context.previous_parameters is not None:
            rows.append(np.asarray(context.previous_parameters, dtype=float))

        try:
            rows.append(np.asarray(context.trajectory_model.reference_parameters(context.problem), dtype=float))
        except Exception:
            pass

        rows.append(0.5 * (np.asarray(context.lower_bound, dtype=float) + np.asarray(context.upper_bound, dtype=float)))
        return finalize_warm_starts(rows, context)


class CompositeWarmStartGenerator(WarmStartGenerator):
    """Combine several generators in priority order."""

    def __init__(self, generators: Sequence[WarmStartGenerator]):
        self.generators = list(generators)

    @property
    def name(self) -> str:
        return "composite_warm_start"

    def supports(self, context: WarmStartContext) -> bool:
        return any(generator.supports(context) for generator in self.generators)

    def generate(self, context: WarmStartContext) -> np.ndarray:
        rows = []
        for generator in self.generators:
            if not generator.supports(context):
                continue
            generated = generator.generate(context)
            if generated.size:
                rows.extend(np.asarray(generated, dtype=float))
        return finalize_warm_starts(rows, context)


def default_parametric_warm_start_generator() -> WarmStartGenerator:
    from .bezier import BezierTrajectoryWarmStartGenerator
    from .terminal_state import SvgdParticleWarmStartGenerator, TerminalStateWarmStartGenerator

    return CompositeWarmStartGenerator(
        [
            SvgdParticleWarmStartGenerator(),
            TerminalStateWarmStartGenerator(),
            BezierTrajectoryWarmStartGenerator(),
            DefaultWarmStartGenerator(),
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
    for row in rows:
        value = np.asarray(row, dtype=float).reshape(-1)
        if value.shape != (dim,) or not np.all(np.isfinite(value)):
            continue
        value = np.clip(value, low, high)
        key = tuple(np.round(value, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= int(context.max_count):
            break

    if not output:
        center = 0.5 * (low + high)
        output.append(center)
    return np.asarray(output, dtype=float)
