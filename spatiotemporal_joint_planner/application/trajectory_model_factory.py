from __future__ import annotations

from spatiotemporal_joint_planner.application.configuration import config_value
from spatiotemporal_joint_planner.trajectory_models import (
    BezierTrajectoryModel,
    FrenetBezierTrajectoryConfig,
    FrenetBezierTrajectoryModel,
    FrenetBSplineTrajectoryConfig,
    FrenetBSplineTrajectoryModel,
    FrenetViaBSplineTrajectoryConfig,
    FrenetViaBSplineTrajectoryModel,
    LatticeTrajectoryConfig,
    LatticeTrajectoryModel,
    SvgdParticleTrajectoryModel,
)


def build_trajectory_model(args):
    model_name = config_value(args, "trajectory_model", "trajectory_model", "type", "lattice_trajectory")
    min_speed = config_value(args, "trajectory_model", "min_terminal_speed", "min_terminal_speed", 0.5)
    max_speed = config_value(args, "trajectory_model", "max_terminal_speed", "max_terminal_speed", 15.0)
    if model_name == "lattice_trajectory":
        return LatticeTrajectoryModel(LatticeTrajectoryConfig(min_terminal_speed=min_speed, max_terminal_speed=max_speed))
    if model_name == "svgd_particle_trajectory":
        return SvgdParticleTrajectoryModel(LatticeTrajectoryConfig(min_terminal_speed=min_speed, max_terminal_speed=max_speed))
    if model_name == "bezier_trajectory":
        return BezierTrajectoryModel()
    if model_name == "frenet_bezier_trajectory":
        return FrenetBezierTrajectoryModel(
            FrenetBezierTrajectoryConfig(min_terminal_speed=min_speed, max_terminal_speed=max_speed)
        )
    if model_name == "frenet_bspline_trajectory":
        return FrenetBSplineTrajectoryModel(
            FrenetBSplineTrajectoryConfig(
                degree=config_value(args, "trajectory_model", "bspline_degree", "degree", 3),
                num_control_points=config_value(args, "trajectory_model", "bspline_control_points", "num_control_points", 6),
                min_speed=config_value(args, "trajectory_model", "min_terminal_speed", "min_speed", 0.5),
                max_speed=config_value(args, "trajectory_model", "max_terminal_speed", "max_speed", 15.0),
            )
        )
    if model_name == "frenet_via_bspline_trajectory":
        return FrenetViaBSplineTrajectoryModel(
            FrenetViaBSplineTrajectoryConfig(
                min_terminal_speed=min_speed,
                max_terminal_speed=max_speed,
                min_mid_time=config_value(args, "trajectory_model", "via_bspline_mid_time_min", "min_mid_time", 1.2),
                max_mid_time=config_value(args, "trajectory_model", "via_bspline_mid_time_max", "max_mid_time", 4.0),
                min_mid_s_offset=config_value(args, "trajectory_model", "via_bspline_min_mid_s_offset", "min_mid_s_offset", 12.0),
                min_mid_speed_floor=config_value(args, "trajectory_model", "via_bspline_min_mid_speed_floor", "min_mid_speed_floor", 3.0),
                min_mid_time_ratio=config_value(args, "trajectory_model", "via_bspline_min_mid_time_ratio", "min_mid_time_ratio", 0.25),
                max_mid_time_ratio=config_value(args, "trajectory_model", "via_bspline_max_mid_time_ratio", "max_mid_time_ratio", 0.85),
                terminal_time_buffer=config_value(args, "trajectory_model", "via_bspline_terminal_time_buffer", "terminal_time_buffer", 0.5),
                monotonic_lateral=config_value(args, "trajectory_model", "via_bspline_monotonic_lateral", "monotonic_lateral", False),
            )
        )
    raise ValueError(f"Unsupported trajectory model: {model_name}")
