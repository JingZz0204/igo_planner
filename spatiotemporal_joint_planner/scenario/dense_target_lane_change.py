from __future__ import annotations

from typing import Sequence

from spatiotemporal_joint_planner.scenario.interactive_lane_change import (
    InteractiveLaneChangeActorSpec,
    InteractiveLaneChangeScenario,
)


class DenseTargetLaneChangeScenario(InteractiveLaneChangeScenario):
    """Interactive lane-change scenario with dense target-lane traffic."""

    def __init__(
        self,
        horizon: float = 5.0,
        dt: float = 0.1,
        lane_width: float = 3.6,
        current_lane_l: float | None = None,
        target_lane_l: float = 0.0,
        ego_speed: float = 9.0,
        target_speed: float = 12.0,
        route_length: float = 650.0,
        road_side_margin: float = 1.0,
        interaction_mode: str = "keep",
        actor_specs: Sequence[InteractiveLaneChangeActorSpec] | None = None,
        current_slow_s: float = 34.0,
        current_slow_v: float = 4.5,
    ):
        super().__init__(
            horizon=horizon,
            dt=dt,
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            ego_speed=ego_speed,
            target_speed=target_speed,
            route_length=route_length,
            road_side_margin=road_side_margin,
            interaction_mode=interaction_mode,
            current_slow_s=current_slow_s,
            current_slow_v=current_slow_v,
        )
        self.actor_specs = tuple(actor_specs) if actor_specs is not None else self._default_dense_actors()

    @property
    def name(self) -> str:
        return "dense_target_lane_change"

    def _default_dense_actors(self) -> tuple[InteractiveLaneChangeActorSpec, ...]:
        rear_accel_by_mode = {
            "yield": -1.2,
            "keep": 0.0,
            "block": 0.8,
        }
        game_rear_accel = rear_accel_by_mode.get(self.interaction_mode, 0.0)
        return (
            InteractiveLaneChangeActorSpec(
                actor_id="current_lane_slow_vehicle",
                actor_type="vehicle",
                s=self.current_slow_s,
                l=self.current_lane_l,
                length=5.0,
                width=2.0,
                s_v=self.current_slow_v,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_back_vehicle",
                actor_type="vehicle",
                s=-42.0,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=11.2,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_rear_vehicle",
                actor_type="vehicle",
                s=-22.0,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=11.8,
                s_a=game_rear_accel,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_gap_front_vehicle",
                actor_type="vehicle",
                s=2.0,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=10.8,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_mid_vehicle",
                actor_type="vehicle",
                s=25.0,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=10.6,
            ),
            InteractiveLaneChangeActorSpec(
                actor_id="target_lane_lead_vehicle",
                actor_type="vehicle",
                s=49.0,
                l=self.target_lane_l,
                length=4.8,
                width=2.0,
                s_v=11.0,
            ),
        )
