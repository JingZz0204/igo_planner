from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorTypeProfile:
    """Behavior hypothesis used to construct a type-conditioned game player."""

    name: str
    prior_probability: float
    desired_speed_scale: float = 1.0
    min_follow_gap: float = 6.0
    time_headway: float = 1.2
    headway_comfort: float = 4.0
    speed_tracking_comfort: float = 2.5
    prior_speed_comfort: float = 2.0
    min_terminal_speed: float = 0.0
    max_terminal_speed: float = 15.0
    min_terminal_s_offset: float = -10.0
    max_terminal_s_offset: float = 25.0


def default_actor_type_profiles() -> tuple[ActorTypeProfile, ...]:
    return (
        ActorTypeProfile(
            name="yielding",
            prior_probability=0.30,
            desired_speed_scale=0.72,
            min_follow_gap=8.0,
            time_headway=1.7,
            headway_comfort=3.0,
            speed_tracking_comfort=2.0,
            prior_speed_comfort=3.0,
            min_terminal_s_offset=-18.0,
            max_terminal_s_offset=12.0,
        ),
        ActorTypeProfile(
            name="normal",
            prior_probability=0.45,
            desired_speed_scale=1.0,
            min_follow_gap=6.0,
            time_headway=1.2,
            headway_comfort=4.0,
            speed_tracking_comfort=2.5,
            prior_speed_comfort=2.0,
            min_terminal_s_offset=-10.0,
            max_terminal_s_offset=25.0,
        ),
        ActorTypeProfile(
            name="aggressive",
            prior_probability=0.25,
            desired_speed_scale=1.18,
            min_follow_gap=3.5,
            time_headway=0.7,
            headway_comfort=5.0,
            speed_tracking_comfort=2.0,
            prior_speed_comfort=3.0,
            min_terminal_s_offset=-2.0,
            max_terminal_s_offset=36.0,
        ),
    )
