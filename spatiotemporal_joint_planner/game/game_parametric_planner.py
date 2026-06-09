from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from spatiotemporal_joint_planner.common import (
    ActorPrediction,
    CostResult,
    EgoState,
    PlannerResult,
    PlanningProblem,
    Trajectory,
)
from spatiotemporal_joint_planner.cost import CostFunction
from spatiotemporal_joint_planner.cost.parametric_trajectory_cost import pseudo_huber, shaped_hinge
from spatiotemporal_joint_planner.game.base import GameOptimizationProblem, GamePlayer, JointTrajectory
from spatiotemporal_joint_planner.game.cost import VehicleGameCost, VehicleGameCostConfig
from spatiotemporal_joint_planner.game.igo_game_optimizer import GameIGOConfig, GameIGOOptimizer
from spatiotemporal_joint_planner.game.trajectory_models import (
    VehicleLongitudinalTrajectoryConfig,
    VehicleLongitudinalTrajectoryModel,
)
from spatiotemporal_joint_planner.planner.base import Planner
from spatiotemporal_joint_planner.planner.warm_start import (
    WarmStartContext,
    WarmStartGenerator,
    default_parametric_warm_start_generator,
)
from spatiotemporal_joint_planner.trajectory_models import TrajectoryModel


@dataclass(frozen=True)
class GameParametricPlannerConfig:
    target_actor_id: str = "target_lane_rear_vehicle"
    candidate_limit: int = 8
    max_initial_anchors: int = 96
    ego_player_name: str = "ego"
    target_player_name: str = "target_rear"
    target_min_terminal_speed: float = 0.0
    target_max_terminal_speed: float = 15.0
    target_min_terminal_s_offset: float = -10.0
    target_max_terminal_s_offset: float = 25.0


class GameParametricPlanner(Planner):
    """Two-player parametric planner using IGO-style game updates."""

    def __init__(
        self,
        ego_trajectory_model: TrajectoryModel,
        ego_cost_function: CostFunction,
        optimizer: Optional[GameIGOOptimizer] = None,
        config: Optional[GameParametricPlannerConfig] = None,
        target_cost_function: Optional[VehicleGameCost] = None,
        warm_start_generator: Optional[WarmStartGenerator] = None,
    ):
        self.ego_trajectory_model = ego_trajectory_model
        self.ego_cost_function = ego_cost_function
        self.optimizer = optimizer or GameIGOOptimizer(GameIGOConfig())
        self.config = config or GameParametricPlannerConfig()
        self.target_cost_function = target_cost_function or VehicleGameCost(VehicleGameCostConfig())
        self.warm_start_generator = warm_start_generator or default_parametric_warm_start_generator()
        self._last_ego_parameters: Optional[np.ndarray] = None
        self._last_target_parameters: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "game_parametric_planner"

    def plan(self, problem: PlanningProblem) -> PlannerResult:
        target_actor = self._target_actor(problem)
        if target_actor is None:
            return PlannerResult(
                trajectory=None,
                cost=None,
                status="missing_game_target_actor",
                metadata={"planner": self.name, "target_actor_id": self.config.target_actor_id},
            )

        exogenous_actors = tuple(actor for actor in problem.actors if actor.actor_id != self.config.target_actor_id)
        ego_problem = PlanningProblem(
            ego=problem.ego,
            ref_path=problem.ref_path,
            road_boundary=problem.road_boundary,
            horizon=problem.horizon,
            dt=problem.dt,
            actors=exogenous_actors,
            metadata=dict(problem.metadata or {}),
        )
        target_problem = self._target_problem(problem, target_actor, exogenous_actors)
        ego_player = self._ego_player(ego_problem)
        target_player = self._target_player(target_problem)

        def decode_joint(parameters_by_player) -> JointTrajectory:
            ego_trajectory = self.ego_trajectory_model.decode(
                np.asarray(parameters_by_player[self.config.ego_player_name], dtype=float),
                ego_problem,
            )
            target_trajectory = target_player.trajectory_model.decode(
                np.asarray(parameters_by_player[self.config.target_player_name], dtype=float),
                target_problem,
            )
            return JointTrajectory(
                trajectories={
                    self.config.ego_player_name: ego_trajectory,
                    self.config.target_player_name: target_trajectory,
                }
            )

        def evaluate_joint(joint: JointTrajectory) -> dict[str, CostResult]:
            ego_trajectory = joint.trajectories[self.config.ego_player_name]
            target_trajectory = joint.trajectories[self.config.target_player_name]
            optimized_target_actor = self._actor_from_trajectory(target_actor, target_trajectory)
            ego_eval_problem = PlanningProblem(
                ego=problem.ego,
                ref_path=problem.ref_path,
                road_boundary=problem.road_boundary,
                horizon=problem.horizon,
                dt=problem.dt,
                actors=tuple(exogenous_actors) + (optimized_target_actor,),
                metadata=dict(problem.metadata or {}),
            )
            ego_cost = self.ego_cost_function.evaluate(ego_trajectory, ego_eval_problem)
            target_cost = self.target_cost_function.evaluate(
                target_trajectory,
                target_problem,
                opponent_trajectories=(ego_trajectory,),
            )
            return {
                self.config.ego_player_name: ego_cost,
                self.config.target_player_name: target_cost,
            }

        def evaluate_joint_batch(parameters_by_player) -> dict[str, np.ndarray]:
            if not hasattr(self.ego_trajectory_model, "decode_batch_arrays") or not hasattr(
                target_player.trajectory_model,
                "decode_batch_arrays",
            ):
                raise ValueError("Game batch evaluation requires batch-decodable trajectory models.")
            ego_parameters = np.asarray(parameters_by_player[self.config.ego_player_name], dtype=float)
            target_parameters = np.asarray(parameters_by_player[self.config.target_player_name], dtype=float)
            ego_batch = self.ego_trajectory_model.decode_batch_arrays(ego_parameters, ego_problem)
            target_batch = target_player.trajectory_model.decode_batch_arrays(target_parameters, target_problem)
            return {
                self.config.ego_player_name: self._evaluate_ego_lattice_game_batch(
                    ego_batch,
                    target_batch,
                    ego_problem,
                ),
                self.config.target_player_name: self._evaluate_target_game_batch(
                    target_batch,
                    ego_batch,
                    target_problem,
                ),
            }

        game_problem = GameOptimizationProblem(
            players=(ego_player, target_player),
            decode_joint=decode_joint,
            evaluate_joint=evaluate_joint,
            evaluate_joint_batch=evaluate_joint_batch,
            metadata={
                "planner": self.name,
                "target_actor_id": self.config.target_actor_id,
                "scenario": problem.metadata.get("scenario") if problem.metadata else None,
            },
        )
        optimization = self.optimizer.optimize(game_problem)
        ego_trajectory = optimization.best_joint_trajectory.trajectories.get(self.config.ego_player_name)
        target_trajectory = optimization.best_joint_trajectory.trajectories.get(self.config.target_player_name)
        ego_cost = optimization.player_costs.get(self.config.ego_player_name)
        target_cost = optimization.player_costs.get(self.config.target_player_name)

        status = "success"
        if ego_trajectory is None or ego_cost is None:
            status = "no_valid_trajectory"
        elif not ego_cost.feasible:
            status = "infeasible_best"

        if ego_trajectory is not None:
            self._last_ego_parameters = np.asarray(optimization.best_parameters[self.config.ego_player_name], dtype=float)
        if target_trajectory is not None:
            self._last_target_parameters = np.asarray(
                optimization.best_parameters[self.config.target_player_name], dtype=float
            )

        return PlannerResult(
            trajectory=ego_trajectory,
            cost=ego_cost,
            status=status,
            candidates=(),
            optimization=None,
            metadata={
                "planner": self.name,
                "trajectory_model": self.ego_trajectory_model.name,
                "optimizer": self.optimizer.name,
                "game_optimizer": optimization,
                "game_target_trajectory": target_trajectory,
                "game_target_actor_id": self.config.target_actor_id,
                "game_target_cost": target_cost,
                "game_best_parameters": optimization.best_parameters,
                "game_metadata": optimization.metadata,
                "parameter_dim": int(self.ego_trajectory_model.parameter_dim(ego_problem)),
            },
        )

    def _evaluate_ego_lattice_game_batch(
        self,
        ego_batch: dict,
        target_batch: dict,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        cost = self.ego_cost_function
        target_cost = target_cost_function or self.target_cost_function
        required = ("_blocked_ranges", "_lattice_batch_running_terms", "_lattice_batch_hierarchy_scores")
        if not all(hasattr(cost, name) for name in required):
            raise ValueError("Ego cost function does not expose lattice batch helpers.")

        blocked_ranges = cost._blocked_ranges(problem)
        terms = cost._lattice_batch_running_terms(ego_batch, problem, blocked_ranges)
        game_collision = self._pairwise_collision_batch(
            agent_s=np.asarray(ego_batch["s"], dtype=float),
            agent_l=np.asarray(ego_batch["l"], dtype=float),
            other_s=np.asarray(target_batch["s"], dtype=float),
            other_l=np.asarray(target_batch["l"], dtype=float),
            agent_front=float(cost.config.ego_front),
            agent_rear=float(cost.config.ego_rear),
            agent_width=float(cost.config.ego_width),
            other_front=float(target_cost.config.vehicle_front) + float(cost.config.planning_obstacle_s_buffer),
            other_rear=float(target_cost.config.vehicle_rear) + float(cost.config.planning_obstacle_s_buffer),
            other_width=float(target_cost.config.vehicle_width)
            + 2.0 * float(cost.config.planning_obstacle_l_buffer),
            decay_s=float(cost.config.collision_decay_s),
        )
        terms["collision_running"] = np.maximum(terms["collision_running"], game_collision["collision_running"])
        terms["collision_overlap"] = np.maximum(terms["collision_overlap"], game_collision["collision_overlap"])

        scores = cost._lattice_batch_hierarchy_scores(terms)
        certificate_score = np.zeros_like(scores["efficiency_score"], dtype=float)
        if bool(cost.config.trajectory_certificate_enabled) and hasattr(cost, "_lattice_batch_certificate_score"):
            certificate_score = cost._lattice_batch_certificate_score(ego_batch, problem, blocked_ranges)

        collision_score = self._saturate_array(scores["collision_cost"], cost.config.collision_score_scale)
        road_score = self._saturate_array(scores["road_cost"], cost.config.road_score_scale)
        hard_hierarchy_cost = (
            1.0e9 * collision_score
            + 1.0e8 * road_score
            + 1.0e7 * scores["kappa_hard_score"]
            + 1.0e6 * scores["dkappa_hard_score"]
            + 1.0e5 * scores["lateral_accel_hard_score"]
            + 1.0e4 * scores["lateral_jerk_hard_score"]
        )
        if bool(cost.config.trajectory_certificate_enabled):
            soft_hierarchy_cost = (
                1.0e3 * scores["efficiency_score"]
                + 1.0e2 * certificate_score
                + 1.0e2 * scores["reference_score"]
                + 1.0e1 * scores["comfort_score"]
            )
        else:
            soft_hierarchy_cost = (
                1.0e3 * scores["efficiency_score"]
                + 1.0e2 * scores["reference_score"]
                + 1.0e1 * scores["comfort_score"]
            )
        total = np.asarray(hard_hierarchy_cost + soft_hierarchy_cost, dtype=float)
        total[~np.isfinite(total)] = float("inf")
        return total

    def _evaluate_target_game_batch(
        self,
        target_batch: dict,
        ego_batch: dict,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        target_cost = target_cost_function or self.target_cost_function
        cfg = target_cost.config
        s = np.asarray(target_batch["s"], dtype=float)
        l = np.asarray(target_batch["l"], dtype=float)
        t = np.asarray(target_batch["t"], dtype=float)
        blocked_ranges = target_cost._blocked_ranges(problem, opponents=())
        exo_collision = self._agent_blocked_collision_batch(
            s=s,
            l=l,
            t=t,
            blocked_ranges=blocked_ranges,
            front=float(cfg.vehicle_front),
            rear=float(cfg.vehicle_rear),
            width=float(cfg.vehicle_width),
        )
        ego_collision = self._pairwise_collision_batch(
            agent_s=s,
            agent_l=l,
            other_s=np.asarray(ego_batch["s"], dtype=float),
            other_l=np.asarray(ego_batch["l"], dtype=float),
            agent_front=float(cfg.vehicle_front),
            agent_rear=float(cfg.vehicle_rear),
            agent_width=float(cfg.vehicle_width),
            other_front=float(getattr(self.ego_cost_function.config, "ego_front", cfg.vehicle_front))
            + float(cfg.obstacle_s_buffer),
            other_rear=float(getattr(self.ego_cost_function.config, "ego_rear", cfg.vehicle_rear))
            + float(cfg.obstacle_s_buffer),
            other_width=float(getattr(self.ego_cost_function.config, "ego_width", cfg.vehicle_width))
            + 2.0 * float(cfg.obstacle_l_buffer),
            decay_s=1.0e9,
        )
        collision_running = np.maximum(exo_collision["collision_running"], ego_collision["collision_running"])
        collision_overlap = np.maximum(exo_collision["collision_overlap"], ego_collision["collision_overlap"])
        road_running, _road_violation = self._target_road_terms_batch(
            l,
            problem,
            target_cost_function=target_cost,
        )
        speed_running, _speed_violation = self._target_speed_terms_batch(
            np.asarray(target_batch["s_v"], dtype=float),
            problem,
            target_cost_function=target_cost,
        )
        comfort_running = self._target_comfort_terms_batch(
            np.asarray(target_batch["s_a"], dtype=float),
            t,
            target_cost_function=target_cost,
        )
        lane_running = self._target_lane_terms_batch(l, problem, target_cost_function=target_cost)
        prior_running = self._target_prior_terms_batch(
            np.asarray(target_batch["s_v"], dtype=float),
            problem,
            target_cost_function=target_cost,
        )
        headway_running = self._target_headway_terms_batch(
            target_s=s,
            target_l=l,
            target_s_v=np.asarray(target_batch["s_v"], dtype=float),
            ego_s=np.asarray(ego_batch["s"], dtype=float),
            ego_l=np.asarray(ego_batch["l"], dtype=float),
            target_cost_function=target_cost,
        )

        collision_cost = self._batch_topk_max(collision_running, 0.15)
        road_cost = self._batch_topk_max(road_running, 0.15)
        speed_cost = np.mean(speed_running, axis=1)
        comfort_cost = np.mean(comfort_running, axis=1)
        lane_cost = np.mean(lane_running, axis=1)
        prior_cost = np.mean(prior_running, axis=1)
        headway_cost = self._batch_topk_max(headway_running, 0.25)
        collision_score = self._saturate_array(collision_cost, cfg.collision_score_scale)
        road_score = self._saturate_array(road_cost, cfg.road_score_scale)
        speed_score = self._saturate_array(speed_cost, cfg.speed_score_scale)
        comfort_score = self._saturate_array(comfort_cost, cfg.comfort_score_scale)
        lane_score = self._saturate_array(lane_cost, cfg.lane_score_scale)
        prior_score = self._saturate_array(prior_cost, cfg.prior_score_scale)
        headway_score = self._saturate_array(headway_cost, cfg.headway_score_scale)
        total = (
            1.0e9 * collision_score
            + 1.0e8 * road_score
            + 1.0e4 * headway_score
            + 1.0e3 * speed_score
            + 1.0e2 * prior_score
            + 1.0e2 * lane_score
            + 1.0e1 * comfort_score
        )
        total = np.asarray(total, dtype=float)
        total[~np.isfinite(total)] = float("inf")
        return total

    def reset(self) -> None:
        self._last_ego_parameters = None
        self._last_target_parameters = None
        self.optimizer.reset()

    def _ego_player(self, problem: PlanningProblem) -> GamePlayer:
        low, high = self.ego_trajectory_model.bounds(problem)
        context = WarmStartContext(
            problem=problem,
            trajectory_model=self.ego_trajectory_model,
            lower_bound=np.asarray(low, dtype=float),
            upper_bound=np.asarray(high, dtype=float),
            previous_parameters=self._last_ego_parameters,
            max_count=int(self.config.max_initial_anchors),
        )
        seeds = self.warm_start_generator.generate(context)
        if seeds.size == 0:
            seeds = self.ego_trajectory_model.reference_parameters(problem).reshape(1, -1)
        return GamePlayer(
            name=self.config.ego_player_name,
            role="ego",
            problem=problem,
            trajectory_model=self.ego_trajectory_model,
            lower_bound=low,
            upper_bound=high,
            initial_population=seeds,
        )

    def _target_player(self, problem: PlanningProblem) -> GamePlayer:
        model = VehicleLongitudinalTrajectoryModel(
            VehicleLongitudinalTrajectoryConfig(
                min_terminal_speed=float(self.config.target_min_terminal_speed),
                max_terminal_speed=float(self.config.target_max_terminal_speed),
                min_terminal_s_offset=float(self.config.target_min_terminal_s_offset),
                max_terminal_s_offset=float(self.config.target_max_terminal_s_offset),
            )
        )
        low, high = model.bounds(problem)
        seeds = self._target_initial_population(model, problem, low, high)
        return GamePlayer(
            name=self.config.target_player_name,
            role="target_rear",
            problem=problem,
            trajectory_model=model,
            lower_bound=low,
            upper_bound=high,
            initial_population=seeds,
        )

    def _target_initial_population(
        self,
        model: VehicleLongitudinalTrajectoryModel,
        problem: PlanningProblem,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        horizon = max(float(problem.horizon), 1e-3)
        nominal_s_end = float(problem.ego.s) + float(problem.ego.s_v) * horizon
        speed_values = np.linspace(
            max(float(low[1]), 0.5 * float(problem.ego.s_v)),
            min(float(high[1]), 1.25 * max(float(problem.ego.s_v), 1e-3)),
            num=5,
            dtype=float,
        )
        offset_values = np.array([-8.0, -4.0, 0.0, 4.0, 8.0, 14.0], dtype=float)
        seeds = []
        if self._last_target_parameters is not None:
            seeds.append(np.clip(self._last_target_parameters, low, high))
        seeds.append(model.reference_parameters(problem))
        for offset in offset_values:
            for speed in speed_values:
                seeds.append(np.array([nominal_s_end + float(offset), float(speed)], dtype=float))
        values = np.asarray(seeds, dtype=float)
        values = np.clip(values, low[None, :], high[None, :])
        return self._dedupe_rows(values)

    @staticmethod
    def _pairwise_collision_batch(
        agent_s: np.ndarray,
        agent_l: np.ndarray,
        other_s: np.ndarray,
        other_l: np.ndarray,
        agent_front: float,
        agent_rear: float,
        agent_width: float,
        other_front: float,
        other_rear: float,
        other_width: float,
        decay_s: float,
    ) -> dict:
        agent_s = np.asarray(agent_s, dtype=float)
        agent_l = np.asarray(agent_l, dtype=float)
        other_s = np.asarray(other_s, dtype=float)
        other_l = np.asarray(other_l, dtype=float)
        n = min(agent_s.shape[0], other_s.shape[0])
        m = min(agent_s.shape[1], other_s.shape[1])
        agent_s = agent_s[:n, :m]
        agent_l = agent_l[:n, :m]
        other_s = other_s[:n, :m]
        other_l = other_l[:n, :m]
        a_s_min = agent_s - float(agent_rear)
        a_s_max = agent_s + float(agent_front)
        a_l_min = agent_l - 0.5 * float(agent_width)
        a_l_max = agent_l + 0.5 * float(agent_width)
        o_s_min = other_s - float(other_rear)
        o_s_max = other_s + float(other_front)
        o_l_min = other_l - 0.5 * float(other_width)
        o_l_max = other_l + 0.5 * float(other_width)
        mask = (a_s_min <= o_s_max) & (o_s_min <= a_s_max) & (a_l_min <= o_l_max) & (o_l_min <= a_l_max)
        s_overlap = np.minimum(a_s_max, o_s_max) - np.maximum(a_s_min, o_s_min)
        l_overlap = np.minimum(a_l_max, o_l_max) - np.maximum(a_l_min, o_l_min)
        penetration = np.maximum(np.minimum(s_overlap, l_overlap), 0.0)
        decay = 0.2 + 0.8 * np.exp(-np.maximum(agent_s - agent_s[:, :1], 0.0) / max(float(decay_s), 1e-3))
        running = np.where(
            mask,
            decay * (1.0 + shaped_hinge(penetration, safe=0.0, soft=0.6, tail_gain=0.35, cap=3.0)),
            0.0,
        )
        return {"collision_running": np.asarray(running, dtype=float), "collision_overlap": mask.astype(float)}

    def _agent_blocked_collision_batch(
        self,
        s: np.ndarray,
        l: np.ndarray,
        t: np.ndarray,
        blocked_ranges: Sequence[dict],
        front: float,
        rear: float,
        width: float,
    ) -> dict:
        s = np.asarray(s, dtype=float)
        l = np.asarray(l, dtype=float)
        running = np.zeros_like(s, dtype=float)
        overlap = np.zeros_like(s, dtype=float)
        if not blocked_ranges:
            return {"collision_running": running, "collision_overlap": overlap}
        s_min = s - float(rear)
        s_max = s + float(front)
        l_min = l - 0.5 * float(width)
        l_max = l + 0.5 * float(width)
        for blocked in blocked_ranges:
            b_s_min = self._blocked_values_for_batch(blocked, "s_min", t, s.shape)
            b_s_max = self._blocked_values_for_batch(blocked, "s_max", t, s.shape)
            b_l_min = self._blocked_values_for_batch(blocked, "l_min", t, s.shape)
            b_l_max = self._blocked_values_for_batch(blocked, "l_max", t, s.shape)
            mask = (s_min <= b_s_max) & (b_s_min <= s_max) & (l_min <= b_l_max) & (b_l_min <= l_max)
            if not np.any(mask):
                continue
            s_overlap = np.minimum(s_max, b_s_max) - np.maximum(s_min, b_s_min)
            l_overlap = np.minimum(l_max, b_l_max) - np.maximum(l_min, b_l_min)
            penetration = np.maximum(np.minimum(s_overlap, l_overlap), 0.0)
            sample_cost = 1.0 + shaped_hinge(penetration, safe=0.0, soft=0.6, tail_gain=0.35, cap=3.0)
            running = np.maximum(running, np.where(mask, sample_cost, 0.0))
            overlap = np.maximum(overlap, mask.astype(float))
        return {"collision_running": running, "collision_overlap": overlap}

    def _target_road_terms_batch(
        self,
        l: np.ndarray,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = (target_cost_function or self.target_cost_function).config
        half_width = 0.5 * float(cfg.vehicle_width)
        left = max(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        right = min(float(problem.road_boundary.left_l), float(problem.road_boundary.right_l))
        l_min = l - half_width
        l_max = l + half_width
        excess = np.maximum(np.maximum(l_max - left, 0.0), np.maximum(right - l_min, 0.0))
        clearance = np.minimum(left - l_max, l_min - right)
        edge_pressure = np.maximum(float(cfg.road_edge_buffer) - clearance, 0.0)
        running = shaped_hinge(excess, safe=0.0, soft=0.6, tail_gain=0.25, cap=3.0) + 0.2 * shaped_hinge(
            edge_pressure,
            safe=0.0,
            soft=max(float(cfg.road_edge_buffer), 1e-3),
            tail_gain=0.1,
            cap=1.5,
        )
        return np.asarray(running, dtype=float), np.asarray(excess > 1e-6, dtype=float)

    def _target_speed_terms_batch(
        self,
        s_v: np.ndarray,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = (target_cost_function or self.target_cost_function).config
        max_speed = max(float(cfg.max_speed), 1e-3)
        target = float(dict(problem.metadata or {}).get("target_speed", max_speed))
        comfort = max(float(cfg.speed_tracking_comfort), 1e-3)
        target_cost = pseudo_huber((s_v - min(target, max_speed)) / comfort, delta=1.0)
        reverse = np.maximum(-s_v, 0.0)
        excess = np.maximum(s_v - max_speed, 0.0)
        limit_cost = shaped_hinge(reverse, safe=0.0, soft=0.5, tail_gain=0.35, cap=3.0) + shaped_hinge(
            excess,
            safe=0.0,
            soft=1.0,
            tail_gain=0.35,
            cap=3.0,
        )
        return np.asarray(target_cost + limit_cost, dtype=float), np.asarray((reverse > 1e-6) | (excess > 1e-6), dtype=float)

    def _target_comfort_terms_batch(
        self,
        s_a: np.ndarray,
        t: np.ndarray,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        cfg = (target_cost_function or self.target_cost_function).config
        accel_cost = pseudo_huber(s_a / max(float(cfg.acceleration_comfort), 1e-3), delta=1.0)
        if s_a.shape[1] >= 2:
            edge_order = 2 if s_a.shape[1] >= 3 else 1
            jerk = np.gradient(s_a, np.asarray(t, dtype=float), axis=1, edge_order=edge_order)
            jerk_cost = pseudo_huber(jerk / max(float(cfg.jerk_comfort), 1e-3), delta=1.0)
            return np.asarray(0.6 * accel_cost + 0.4 * jerk_cost, dtype=float)
        return np.asarray(accel_cost, dtype=float)

    def _target_lane_terms_batch(
        self,
        l: np.ndarray,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        cfg = (target_cost_function or self.target_cost_function).config
        target_l = float(dict(problem.metadata or {}).get("reference_l", problem.ego.l))
        return np.asarray(pseudo_huber((l - target_l) / max(float(cfg.lane_keep_comfort), 1e-3), delta=1.0), dtype=float)

    def _target_prior_terms_batch(
        self,
        s_v: np.ndarray,
        problem: PlanningProblem,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        cfg = (target_cost_function or self.target_cost_function).config
        prior_speed = float(dict(problem.metadata or {}).get("prior_speed", problem.ego.s_v))
        return np.asarray(pseudo_huber((s_v - prior_speed) / max(float(cfg.prior_speed_comfort), 1e-3), delta=1.0), dtype=float)

    def _target_headway_terms_batch(
        self,
        target_s: np.ndarray,
        target_l: np.ndarray,
        target_s_v: np.ndarray,
        ego_s: np.ndarray,
        ego_l: np.ndarray,
        target_cost_function: Optional[VehicleGameCost] = None,
    ) -> np.ndarray:
        cfg = (target_cost_function or self.target_cost_function).config
        target_s = np.asarray(target_s, dtype=float)
        target_l = np.asarray(target_l, dtype=float)
        target_s_v = np.asarray(target_s_v, dtype=float)
        ego_s = np.asarray(ego_s, dtype=float)
        ego_l = np.asarray(ego_l, dtype=float)
        n = min(target_s.shape[0], ego_s.shape[0])
        m = min(target_s.shape[1], ego_s.shape[1])
        if n <= 0 or m <= 0:
            return np.empty((0, 0), dtype=float)
        target_s = target_s[:n, :m]
        target_l = target_l[:n, :m]
        target_s_v = target_s_v[:n, :m]
        ego_s = ego_s[:n, :m]
        ego_l = ego_l[:n, :m]
        target_front = target_s + float(cfg.vehicle_front)
        ego_rear = ego_s - float(getattr(self.ego_cost_function.config, "ego_rear", cfg.vehicle_rear))
        gap = ego_rear - target_front
        desired_gap = float(cfg.min_follow_gap) + np.maximum(target_s_v, 0.0) * float(cfg.time_headway)
        lateral_gate = max(float(cfg.vehicle_width), 1e-3)
        same_lane = np.abs(ego_l - target_l) <= lateral_gate
        ahead = gap >= -float(cfg.vehicle_front + cfg.vehicle_rear)
        pressure = np.maximum(desired_gap - gap, 0.0)
        cost = pseudo_huber(pressure / max(float(cfg.headway_comfort), 1e-3), delta=1.0)
        return np.asarray(np.where(same_lane & ahead, cost, 0.0), dtype=float)

    @staticmethod
    def _blocked_values_for_batch(blocked: dict, key: str, t: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        temporal = blocked.get("temporal")
        if not temporal:
            return np.full(target_shape, float(blocked[key]), dtype=float)
        source_t = np.asarray(temporal.get("t", []), dtype=float).reshape(-1)
        source_v = np.asarray(temporal.get(key, []), dtype=float).reshape(-1)
        n = min(source_t.size, source_v.size)
        if n == 0:
            return np.full(target_shape, float(blocked[key]), dtype=float)
        values = np.interp(np.asarray(t, dtype=float), source_t[:n], source_v[:n], left=source_v[0], right=source_v[n - 1])
        return np.broadcast_to(values.reshape(1, -1), target_shape)

    @staticmethod
    def _batch_topk_max(values: np.ndarray, fraction: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.empty((0,), dtype=float)
        k = max(1, int(np.ceil(values.shape[1] * float(fraction))))
        top = np.partition(values, -k, axis=1)[:, -k:]
        return np.asarray(0.7 * np.mean(top, axis=1) + 0.3 * np.max(values, axis=1), dtype=float)

    @staticmethod
    def _saturate_array(values: np.ndarray, scale: float) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=float), 0.0)
        scale = max(float(scale), 1e-6)
        return values / (values + scale)

    def _target_problem(
        self,
        problem: PlanningProblem,
        actor: ActorPrediction,
        exogenous_actors: Sequence[ActorPrediction],
    ) -> PlanningProblem:
        metadata = dict(actor.metadata or {})
        s = float(metadata.get("s", 0.0))
        l = float(metadata.get("l", dict(problem.metadata or {}).get("target_lane_l", 0.0)))
        s_v = float(metadata.get("s_v", 0.0))
        s_a = float(metadata.get("s_a", 0.0))
        target_metadata = dict(problem.metadata or {})
        target_metadata.update(
            {
                "reference_l": l,
                "target_speed": s_v,
                "prior_speed": s_v,
                "source_actor_id": actor.actor_id,
            }
        )
        return PlanningProblem(
            ego=EgoState(s=s, l=l, s_v=s_v, s_a=s_a),
            ref_path=problem.ref_path,
            road_boundary=problem.road_boundary,
            horizon=problem.horizon,
            dt=problem.dt,
            actors=tuple(exogenous_actors),
            metadata=target_metadata,
        )

    def _target_actor(self, problem: PlanningProblem) -> Optional[ActorPrediction]:
        for actor in problem.actors:
            if actor.actor_id == self.config.target_actor_id:
                return actor
        return None

    @staticmethod
    def _actor_from_trajectory(template: ActorPrediction, trajectory: Trajectory) -> ActorPrediction:
        t = np.asarray(trajectory.t, dtype=float)
        s = np.asarray(trajectory.s, dtype=float)
        l = np.asarray(trajectory.l, dtype=float)
        n = min(t.size, s.size, l.size)
        half_length = 0.5 * float(template.length)
        half_width = 0.5 * float(template.width)
        temporal = {
            "t": t[:n],
            "s_min": s[:n] - half_length,
            "s_max": s[:n] + half_length,
            "l_min": l[:n] - half_width,
            "l_max": l[:n] + half_width,
        }
        x = np.asarray(trajectory.x, dtype=float)[:n] if trajectory.x is not None else s[:n]
        y = np.asarray(trajectory.y, dtype=float)[:n] if trajectory.y is not None else l[:n]
        yaw = np.asarray(trajectory.yaw, dtype=float)[:n] if trajectory.yaw is not None else np.zeros((n,), dtype=float)
        s_v = trajectory.s_v
        return ActorPrediction(
            actor_id=template.actor_id,
            actor_type=template.actor_type,
            times=t[:n],
            x=x,
            y=y,
            yaw=yaw,
            length=float(template.length),
            width=float(template.width),
            metadata={
                "s": float(s[0]) if s.size else 0.0,
                "l": float(l[0]) if l.size else 0.0,
                "s_v": float(np.asarray(s_v, dtype=float)[0]) if s_v is not None and np.asarray(s_v).size else 0.0,
                "blocked_s_min": float(temporal["s_min"][0]) if n else 0.0,
                "blocked_s_max": float(temporal["s_max"][0]) if n else 0.0,
                "blocked_l_min": float(temporal["l_min"][0]) if n else 0.0,
                "blocked_l_max": float(temporal["l_max"][0]) if n else 0.0,
                "temporal_blocked_range": temporal,
                "static": False,
                "optimized_by_game": True,
            },
        )

    @staticmethod
    def _dedupe_rows(values: np.ndarray) -> np.ndarray:
        rows = []
        seen = set()
        for row in values:
            key = tuple(np.round(np.asarray(row, dtype=float), 9).tolist())
            if key in seen:
                continue
            seen.add(key)
            rows.append(np.asarray(row, dtype=float))
        return np.asarray(rows, dtype=float)
