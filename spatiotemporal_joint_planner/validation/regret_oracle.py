from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from spatiotemporal_joint_planner.game.bayesian_igo_optimizer import BayesianGameOptimizationProblem


@dataclass(frozen=True)
class BayesianRegretOracleConfig:
    uniform_samples: int = 512
    axis_samples_per_dimension: int = 9
    include_initial_population: bool = True
    seed: int = 101


@dataclass(frozen=True)
class PlayerRegret:
    player_name: str
    current_cost: float
    best_response_cost: float
    normalized_regret: float
    best_response_parameters: np.ndarray
    candidate_count: int

    def as_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "current_cost": float(self.current_cost),
            "best_response_cost": float(self.best_response_cost),
            "normalized_regret": float(self.normalized_regret),
            "best_response_parameters": np.asarray(self.best_response_parameters, dtype=float).tolist(),
            "candidate_count": int(self.candidate_count),
        }


@dataclass(frozen=True)
class BayesianRegretReport:
    ego: PlayerRegret
    targets_by_type: Mapping[str, PlayerRegret]

    @property
    def max_regret(self) -> float:
        return float(max([self.ego.normalized_regret, *(item.normalized_regret for item in self.targets_by_type.values())]))

    def as_dict(self) -> dict:
        return {
            "ego": self.ego.as_dict(),
            "targets_by_type": {name: value.as_dict() for name, value in self.targets_by_type.items()},
            "max_regret": self.max_regret,
        }


class BayesianRegretOracle:
    """Independent high-coverage unilateral best-response checker.

    The oracle never reuses optimizer populations or internal convergence
    diagnostics. It builds a fresh deterministic candidate set from bounds,
    axis sweeps, and optional warm-start anchors.
    """

    def __init__(self, config: BayesianRegretOracleConfig | None = None):
        self.config = config or BayesianRegretOracleConfig()

    def evaluate(
        self,
        problem: BayesianGameOptimizationProblem,
        selected_parameters: Mapping[str, np.ndarray],
    ) -> BayesianRegretReport:
        ego = problem.ego_player
        selected_ego = np.asarray(selected_parameters[ego.name], dtype=float)
        ego_candidates = self._candidate_matrix(ego, selected_ego, seed_offset=0)
        ego_costs = np.asarray(
            problem.evaluate_unilateral_batch(ego.name, ego_candidates, selected_parameters),
            dtype=float,
        )
        ego_report = self._player_report(ego.name, ego_candidates, ego_costs)

        target_reports = {}
        for index, player in enumerate(problem.strategy_players[1:]):
            selected = np.asarray(selected_parameters[player.name], dtype=float)
            candidates = self._candidate_matrix(player, selected, seed_offset=index + 1)
            costs = problem.evaluate_unilateral_batch(player.name, candidates, selected_parameters)
            target_reports[player.name] = self._player_report(player.name, candidates, costs)
        return BayesianRegretReport(ego=ego_report, targets_by_type=target_reports)

    def _candidate_matrix(self, player, selected: np.ndarray, seed_offset: int) -> np.ndarray:
        low = np.asarray(player.lower_bound, dtype=float)
        high = np.asarray(player.upper_bound, dtype=float)
        selected = np.clip(np.asarray(selected, dtype=float), low, high)
        rows = [selected]
        axis_count = max(int(self.config.axis_samples_per_dimension), 2)
        for dimension in range(low.size):
            for value in np.linspace(low[dimension], high[dimension], num=axis_count, dtype=float):
                row = selected.copy()
                row[dimension] = value
                rows.append(row)
        if bool(self.config.include_initial_population):
            anchors = np.asarray(player.initial_population, dtype=float)
            if anchors.ndim == 1:
                anchors = anchors.reshape(1, -1)
            rows.extend(anchors)
        random_count = max(int(self.config.uniform_samples), 0)
        if random_count:
            rng = np.random.default_rng(int(self.config.seed) + int(seed_offset))
            rows.extend(rng.uniform(low, high, size=(random_count, low.size)))
        return self._dedupe_rows(np.clip(np.asarray(rows, dtype=float), low, high))

    @staticmethod
    def _player_report(player_name: str, candidates: np.ndarray, costs: np.ndarray) -> PlayerRegret:
        values = np.asarray(costs, dtype=float).reshape(-1)
        safe = np.where(np.isfinite(values), values, np.inf)
        best_index = int(np.argmin(safe))
        current = float(safe[0])
        best = float(safe[best_index])
        regret = max(current - best, 0.0) / max(abs(current), 1.0) if np.isfinite(current) else float("inf")
        return PlayerRegret(
            player_name=str(player_name),
            current_cost=current,
            best_response_cost=best,
            normalized_regret=float(regret),
            best_response_parameters=np.asarray(candidates[best_index], dtype=float).copy(),
            candidate_count=int(candidates.shape[0]),
        )

    @staticmethod
    def _dedupe_rows(values: np.ndarray) -> np.ndarray:
        rows = []
        seen = set()
        for row in np.asarray(values, dtype=float):
            key = tuple(np.round(row, 10).tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        return np.asarray(rows, dtype=float)
