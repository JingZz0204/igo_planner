from __future__ import annotations

import argparse

from spatiotemporal_joint_planner.application import (
    load_demo_config,
    load_scenario_config,
    resolve_demo_runtime_args,
)
from spatiotemporal_joint_planner.simulation import run_simulation


SCENARIO_CHOICES = (
    "static_nudge",
    "lane_change",
    "interactive_lane_change",
    "dense_target_lane_change",
    "unprotected_intersection",
    "unprotected_left_turn",
)

TRAJECTORY_MODEL_CHOICES = (
    "lattice_trajectory",
    "bezier_trajectory",
    "frenet_bezier_trajectory",
    "svgd_particle_trajectory",
    "frenet_bspline_trajectory",
    "frenet_via_bspline_trajectory",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatiotemporal joint planner demo")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="static_nudge")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML demo config path. Defaults to spatiotemporal_joint_planner/config/demo/default.yaml.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a demo YAML value with dot path, e.g. --set optimizer.samples=32.",
    )
    parser.add_argument(
        "--scenario-config",
        type=str,
        default=None,
        help="YAML scenario config path. Defaults to spatiotemporal_joint_planner/config/scenario/<scenario>.yaml.",
    )
    parser.add_argument(
        "--scenario-set",
        action="append",
        default=[],
        help="Override a scenario YAML value with dot path, e.g. --scenario-set actors.current_slow.s=30.",
    )
    parser.add_argument(
        "--trajectory-model",
        choices=TRAJECTORY_MODEL_CHOICES,
        default=None,
        help="Override trajectory_model.type from demo config.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override runtime.max_steps from demo config.")
    parser.add_argument("--save-frame", type=str, default=None, help="Save one frame to image path.")
    parser.add_argument("--save-mp4", type=str, default=None, help="Save an MP4 rollout to this path.")
    parser.add_argument("--show", action="store_true", default=None, help="Show live matplotlib visualization.")
    parser.add_argument("--no-show", action="store_true", default=None, help="Disable visualization even if config enables it.")
    return parser


def resolve_args(args):
    args.demo_config_values = load_demo_config(args)
    args.scenario_config_values = load_scenario_config(args)
    resolve_demo_runtime_args(args)
    return args


def main() -> None:
    run_simulation(resolve_args(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
