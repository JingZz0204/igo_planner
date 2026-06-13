from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from spatiotemporal_joint_planner.belief import ActorTypeProfile, default_actor_type_profiles


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG_DIR = PACKAGE_ROOT / "config" / "demo"
SCENARIO_CONFIG_DIR = PACKAGE_ROOT / "config" / "scenario"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML config files. Install dependency `pyyaml`.") from exc

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return dict(data)


def scenario_config_path(args) -> Path:
    if args.scenario_config:
        return Path(args.scenario_config).expanduser().resolve()
    return SCENARIO_CONFIG_DIR / f"{args.scenario}.yaml"


def demo_config_path(args) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    return DEMO_CONFIG_DIR / "default.yaml"


def load_demo_config(args) -> dict[str, Any]:
    path = demo_config_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Demo config not found: {path}")
    raw = load_yaml_mapping(path)
    config = raw.get("demo", raw)
    if not isinstance(config, Mapping):
        raise ValueError(f"`demo` entry must be a mapping: {path}")
    config = dict(config)
    for override in args.set or []:
        apply_config_override(config, override)
    args.config_path = str(path)
    return config


def load_scenario_config(args) -> dict[str, Any]:
    path = scenario_config_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    raw = load_yaml_mapping(path)
    config = raw.get("scenario", raw)
    if not isinstance(config, Mapping):
        raise ValueError(f"`scenario` entry must be a mapping: {path}")
    config = dict(config)
    config_type = config.get("type")
    if config_type is not None and str(config_type) != str(args.scenario):
        raise ValueError(f"Scenario config type `{config_type}` does not match CLI scenario `{args.scenario}`: {path}")

    for override in args.scenario_set or []:
        apply_config_override(config, override)
    args.scenario_config_path = str(path)
    return config


def apply_config_override(config: dict[str, Any], override: str) -> None:
    if "=" not in str(override):
        raise ValueError(f"Config override must be key=value, got: {override}")
    key, raw_value = str(override).split("=", 1)
    keys = [part for part in key.strip().split(".") if part]
    if not keys:
        raise ValueError(f"Config override has empty key: {override}")

    import yaml

    value = yaml.safe_load(raw_value)
    cursor = config
    for part in keys[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def scenario_value(args, config: Mapping[str, Any], legacy_name: str, config_name: str, default: Any) -> Any:
    cli_value = getattr(args, legacy_name, None)
    if cli_value is not None:
        return cli_value
    return config.get(config_name, default)


def config_section(args, section_name: str) -> Mapping[str, Any]:
    config = getattr(args, "demo_config_values", {})
    section = config.get(section_name, {}) if isinstance(config, Mapping) else {}
    return section if isinstance(section, Mapping) else {}


def config_value(args, section_name: str, legacy_name: str, config_name: str, default: Any) -> Any:
    cli_value = getattr(args, legacy_name, None)
    if cli_value is not None:
        return cli_value
    return config_section(args, section_name).get(config_name, default)


def actor_type_profiles_from_config(args) -> tuple[ActorTypeProfile, ...]:
    raw_profiles = config_section(args, "game").get("type_profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        return default_actor_type_profiles()
    defaults = {profile.name: profile for profile in default_actor_type_profiles()}
    profiles = []
    for name, raw in raw_profiles.items():
        values = dict(raw) if isinstance(raw, Mapping) else {}
        base = defaults.get(str(name), ActorTypeProfile(name=str(name), prior_probability=1.0))
        profiles.append(
            ActorTypeProfile(
                name=str(name),
                prior_probability=float(values.get("prior_probability", base.prior_probability)),
                desired_speed_scale=float(values.get("desired_speed_scale", base.desired_speed_scale)),
                min_follow_gap=float(values.get("min_follow_gap", base.min_follow_gap)),
                time_headway=float(values.get("time_headway", base.time_headway)),
                headway_comfort=float(values.get("headway_comfort", base.headway_comfort)),
                speed_tracking_comfort=float(values.get("speed_tracking_comfort", base.speed_tracking_comfort)),
                prior_speed_comfort=float(values.get("prior_speed_comfort", base.prior_speed_comfort)),
                min_terminal_speed=float(values.get("min_terminal_speed", base.min_terminal_speed)),
                max_terminal_speed=float(values.get("max_terminal_speed", base.max_terminal_speed)),
                min_terminal_s_offset=float(values.get("min_terminal_s_offset", base.min_terminal_s_offset)),
                max_terminal_s_offset=float(values.get("max_terminal_s_offset", base.max_terminal_s_offset)),
            )
        )
    return tuple(profiles)


def resolve_demo_runtime_args(args) -> None:
    args.max_steps = config_value(args, "runtime", "max_steps", "max_steps", 150)
    args.log_every = config_value(args, "runtime", "log_every", "log_every", 5)
    args.planning_dt = config_value(args, "runtime", "planning_dt", "planning_dt", 0.25)
    args.save_frame = config_value(args, "visualization", "save_frame", "save_frame", None)
    args.save_mp4 = config_value(args, "visualization", "save_mp4", "save_mp4", None)
    args.mp4_fps = config_value(args, "visualization", "mp4_fps", "mp4_fps", 10.0)
    args.pause = config_value(args, "visualization", "pause", "pause", 0.08)

    show = config_value(args, "visualization", "show", "show", False)
    if args.no_show is not None:
        show = not args.no_show
    args.show = bool(show)
