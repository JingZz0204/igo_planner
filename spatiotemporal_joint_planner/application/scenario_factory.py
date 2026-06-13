from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from spatiotemporal_joint_planner.application.configuration import scenario_value
from spatiotemporal_joint_planner.scenario import (
    DenseTargetLaneChangeScenario,
    InteractiveLaneChangeActorSpec,
    InteractiveLaneChangeScenario,
    LaneChangeActorSpec,
    LaneChangeScenario,
    StaticNudgeScenario,
    StaticObstacleSpec,
    UnprotectedIntersectionScenario,
    UnprotectedLeftTurnScenario,
)


def _resolve_lane_l(value: Any, current_lane_l: float, target_lane_l: float) -> float:
    if isinstance(value, str):
        if value == "current_lane":
            return float(current_lane_l)
        if value == "target_lane":
            return float(target_lane_l)
    return float(value)


def _static_obstacles(config: Mapping[str, Any]) -> Optional[tuple[StaticObstacleSpec, ...]]:
    obstacles = config.get("obstacles")
    if obstacles is None:
        return None
    return tuple(
        StaticObstacleSpec(
            actor_id=str(item.get("actor_id", "obstacle")),
            actor_type=str(item.get("actor_type", "vehicle")),
            s=float(item["s"]),
            l=float(item["l"]),
            length=float(item["length"]),
            width=float(item["width"]),
        )
        for item in obstacles
    )


def _lane_change_actors(
    config: Mapping[str, Any],
    current_lane_l: float,
    target_lane_l: float,
) -> Optional[tuple[LaneChangeActorSpec, ...]]:
    actors = config.get("actors")
    if actors is None:
        return None
    return tuple(
        LaneChangeActorSpec(
            actor_id=str(item.get("actor_id", "actor")),
            actor_type=str(item.get("actor_type", "vehicle")),
            s=float(item["s"]),
            l=_resolve_lane_l(item.get("l", target_lane_l), current_lane_l, target_lane_l),
            length=float(item["length"]),
            width=float(item["width"]),
            s_v=float(item.get("s_v", item.get("v", 0.0))),
            l_v=float(item.get("l_v", 0.0)),
        )
        for item in actors
    )


def _iter_actor_items(actors: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if actors is None:
        return []
    if isinstance(actors, Mapping):
        return [(str(name), item) for name, item in actors.items() if isinstance(item, Mapping)]
    return [(str(index), item) for index, item in enumerate(actors) if isinstance(item, Mapping)]


def _interactive_actors(
    config: Mapping[str, Any],
    current_lane_l: float,
    target_lane_l: float,
) -> Optional[tuple[InteractiveLaneChangeActorSpec, ...]]:
    items = _iter_actor_items(config.get("actors"))
    if not items:
        return None
    return tuple(
        InteractiveLaneChangeActorSpec(
            actor_id=str(item.get("actor_id", name)),
            actor_type=str(item.get("actor_type", "vehicle")),
            s=float(item["s"]),
            l=_resolve_lane_l(item.get("l", target_lane_l), current_lane_l, target_lane_l),
            length=float(item.get("length", 4.8)),
            width=float(item.get("width", 2.0)),
            s_v=float(item.get("s_v", item.get("v", 0.0))),
            s_a=float(item.get("s_a", item.get("a", 0.0))),
            l_v=float(item.get("l_v", 0.0)),
            l_a=float(item.get("l_a", 0.0)),
        )
        for name, item in items
    )


def _actor_value(
    config: Mapping[str, Any],
    actor_name: str,
    field: str,
    default: Any,
    aliases: Sequence[str] = (),
) -> Any:
    actors = config.get("actors", {})
    actor = actors.get(actor_name, {}) if isinstance(actors, Mapping) else {}
    for key in (field, *aliases):
        if isinstance(actor, Mapping) and key in actor:
            return actor[key]
    return default


def build_scenario(args):
    config = getattr(args, "scenario_config_values", {})
    if args.scenario == "static_nudge":
        return StaticNudgeScenario(
            horizon=scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=scenario_value(args, config, "traj_dt", "dt", 0.1),
            road_width=scenario_value(args, config, "road_width", "road_width", 8.0),
            lane_width=scenario_value(args, config, "lane_width", "lane_width", 3.6),
            default_start_l=scenario_value(args, config, "start_l", "default_start_l", 2.0),
            target_speed=scenario_value(args, config, "target_speed", "target_speed", 30.0 / 3.6),
            obstacle_specs=_static_obstacles(config),
        )
    if args.scenario == "lane_change":
        lane_width = float(scenario_value(args, config, "lane_width", "lane_width", 3.6))
        current_lane_l = scenario_value(args, config, "lane_change_current_l", "current_lane_l", None)
        current_lane_l = None if current_lane_l is None else float(current_lane_l)
        target_lane_l = float(scenario_value(args, config, "lane_change_target_l", "target_lane_l", 0.0))
        return LaneChangeScenario(
            horizon=scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=scenario_value(args, config, "traj_dt", "dt", 0.1),
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            target_speed=scenario_value(args, config, "target_speed", "target_speed", 12.0),
            route_length=scenario_value(args, config, "lane_change_route_length", "route_length", 260.0),
            road_side_margin=scenario_value(args, config, "lane_change_side_margin", "road_side_margin", 1.0),
            actor_specs=_lane_change_actors(
                config,
                current_lane_l if current_lane_l is not None else -lane_width,
                target_lane_l,
            ),
        )
    if args.scenario == "interactive_lane_change":
        lane_width = float(scenario_value(args, config, "lane_width", "lane_width", 3.6))
        current_lane_l = scenario_value(args, config, "lane_change_current_l", "current_lane_l", None)
        current_lane_l = None if current_lane_l is None else float(current_lane_l)
        target_lane_l = float(scenario_value(args, config, "lane_change_target_l", "target_lane_l", 0.0))
        route_length = float(scenario_value(args, config, "lane_change_route_length", "route_length", 520.0))
        return InteractiveLaneChangeScenario(
            horizon=scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=scenario_value(args, config, "traj_dt", "dt", 0.1),
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            ego_speed=scenario_value(args, config, "interactive_ego_speed", "ego_speed", 9.0),
            target_speed=scenario_value(args, config, "target_speed", "target_speed", 30.0 / 3.6),
            route_length=max(route_length, 520.0),
            road_side_margin=scenario_value(args, config, "lane_change_side_margin", "road_side_margin", 1.0),
            interaction_mode=scenario_value(args, config, "interaction_mode", "interaction_mode", "keep"),
            target_lead_s=scenario_value(
                args, config, "target_lead_s", "target_lead_s", _actor_value(config, "target_lead", "s", 45.0)
            ),
            target_lead_v=scenario_value(
                args, config, "target_lead_v", "target_lead_v", _actor_value(config, "target_lead", "s_v", 8.5, ("v",))
            ),
            target_rear_s=scenario_value(
                args, config, "target_rear_s", "target_rear_s", _actor_value(config, "target_rear", "s", -18.0)
            ),
            target_rear_v=scenario_value(
                args, config, "target_rear_v", "target_rear_v", _actor_value(config, "target_rear", "s_v", 15.0, ("v",))
            ),
            current_slow_s=scenario_value(
                args, config, "current_slow_s", "current_slow_s", _actor_value(config, "current_slow", "s", 24.0)
            ),
            current_slow_v=scenario_value(
                args, config, "current_slow_v", "current_slow_v", _actor_value(config, "current_slow", "s_v", 4.5, ("v",))
            ),
        )
    if args.scenario == "dense_target_lane_change":
        lane_width = float(scenario_value(args, config, "lane_width", "lane_width", 3.6))
        current_lane_l = scenario_value(args, config, "lane_change_current_l", "current_lane_l", None)
        current_lane_l = None if current_lane_l is None else float(current_lane_l)
        target_lane_l = float(scenario_value(args, config, "lane_change_target_l", "target_lane_l", 0.0))
        route_length = float(scenario_value(args, config, "lane_change_route_length", "route_length", 650.0))
        return DenseTargetLaneChangeScenario(
            horizon=scenario_value(args, config, "horizon", "horizon", 5.0),
            dt=scenario_value(args, config, "traj_dt", "dt", 0.1),
            lane_width=lane_width,
            current_lane_l=current_lane_l,
            target_lane_l=target_lane_l,
            ego_speed=scenario_value(args, config, "interactive_ego_speed", "ego_speed", 9.0),
            target_speed=scenario_value(args, config, "target_speed", "target_speed", 12.0),
            route_length=max(route_length, 650.0),
            road_side_margin=scenario_value(args, config, "lane_change_side_margin", "road_side_margin", 1.0),
            interaction_mode=scenario_value(args, config, "interaction_mode", "interaction_mode", "keep"),
            actor_specs=_interactive_actors(
                config,
                current_lane_l if current_lane_l is not None else -lane_width,
                target_lane_l,
            ),
            current_slow_s=scenario_value(
                args, config, "current_slow_s", "current_slow_s", _actor_value(config, "current_slow", "s", 34.0)
            ),
            current_slow_v=scenario_value(
                args, config, "current_slow_v", "current_slow_v", _actor_value(config, "current_slow", "s_v", 4.5, ("v",))
            ),
        )
    if args.scenario == "unprotected_intersection":
        return UnprotectedIntersectionScenario(
            horizon=float(config.get("horizon", 5.0)),
            dt=float(config.get("dt", 0.1)),
            route_extent=float(config.get("route_extent", 55.0)),
            lane_offset=float(config.get("lane_offset", 1.8)),
            road_half_width=float(config.get("road_half_width", 4.2)),
            ego_s=float(config.get("ego_s", 18.0)),
            ego_speed=float(config.get("ego_speed", 9.0)),
            target_speed=float(config.get("target_speed", 10.0)),
            north_actor_s=float(_actor_value(config, "north", "s", 22.0)),
            north_actor_speed=float(_actor_value(config, "north", "s_v", 7.5, ("v",))),
            south_actor_s=float(_actor_value(config, "south", "s", 15.0)),
            south_actor_speed=float(_actor_value(config, "south", "s_v", 8.5, ("v",))),
        )
    if args.scenario == "unprotected_left_turn":
        return UnprotectedLeftTurnScenario(
            horizon=float(config.get("horizon", 5.0)),
            dt=float(config.get("dt", 0.1)),
            route_extent=float(config.get("route_extent", 55.0)),
            lane_offset=float(config.get("lane_offset", 1.8)),
            road_half_width=float(config.get("road_half_width", 4.2)),
            turn_entry_y=float(config.get("turn_entry_y", -5.0)),
            ego_s=float(config.get("ego_s", 18.0)),
            ego_speed=float(config.get("ego_speed", 8.0)),
            target_speed=float(config.get("target_speed", 9.0)),
            oncoming_s=float(_actor_value(config, "oncoming", "s", 28.0)),
            oncoming_speed=float(_actor_value(config, "oncoming", "s_v", 8.0, ("v",))),
            left_crossing_s=float(_actor_value(config, "left_crossing", "s", 28.0)),
            left_crossing_speed=float(_actor_value(config, "left_crossing", "s_v", 7.5, ("v",))),
        )
    raise ValueError(f"Unsupported scenario: {args.scenario}")
