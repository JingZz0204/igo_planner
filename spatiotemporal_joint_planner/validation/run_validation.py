from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spatiotemporal_joint_planner import demo
from spatiotemporal_joint_planner.application import (
    build_planner,
    build_scenario,
    build_trajectory_model,
    load_demo_config,
    load_scenario_config,
    resolve_demo_runtime_args,
)
from spatiotemporal_joint_planner.belief import (
    ActorTypeBelief,
    BayesianTypeFilter,
    BayesianTypeFilterConfig,
    LongitudinalObservation,
    LongitudinalTypePrediction,
    default_actor_type_profiles,
)
from spatiotemporal_joint_planner.contingency import BeliefContinuationEvaluator, RiskAggregator, RiskAggregatorConfig
from spatiotemporal_joint_planner.common import ActorPrediction, EgoState, PlanningProblem, RoadBoundary, Trajectory
from spatiotemporal_joint_planner.cost import ParametricTrajectoryCost, ParametricTrajectoryCostConfig
from spatiotemporal_joint_planner.scenario import InteractiveLaneChangeScenario
from spatiotemporal_joint_planner.trajectory_models import (
    FrenetBezierTrajectoryModel,
    FrenetBSplineTrajectoryModel,
    FrenetViaBSplineTrajectoryModel,
    LatticeTrajectoryModel,
)
from spatiotemporal_joint_planner.validation.cost_consistency import (
    CostConsistencyConfig,
    evaluate_cost_consistency,
)
from spatiotemporal_joint_planner.validation.regret_oracle import (
    BayesianRegretOracle,
    BayesianRegretOracleConfig,
)


def build_report(oracle_samples: int = 1024) -> dict:
    scenario = InteractiveLaneChangeScenario(route_length=800.0)
    problem = scenario.build_problem(scenario.initial_state())
    cost = ParametricTrajectoryCost(ParametricTrajectoryCostConfig(trajectory_certificate_enabled=False))
    consistency = {}
    for model in (
        LatticeTrajectoryModel(),
        FrenetBSplineTrajectoryModel(),
        FrenetBezierTrajectoryModel(),
        FrenetViaBSplineTrajectoryModel(),
    ):
        result = evaluate_cost_consistency(
            model,
            cost,
            problem,
            CostConsistencyConfig(
                sample_count=128,
                seed=7,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-4,
            ),
        )
        consistency[model.name] = result.as_dict()

    belief = _belief_validation()
    continuation = _continuation_validation()
    temporal_collision = _temporal_collision_validation()
    game = _bayesian_game_validation(oracle_samples)
    checks = {
        "scalar_batch_cost_consistency": all(item["passed"] for item in consistency.values()),
        "belief_identification": all(item["true_type_probability"] >= 0.75 for item in belief.values()),
        "continuation_properties": bool(continuation["passed"]),
        "temporal_collision_alignment": bool(temporal_collision["passed"]),
        "flow_profile_bayesian_regret": float(game["flow_profile_max_bayesian_regret"]) <= 0.1,
        "online_local_bayesian_regret": float(game["online_local_max_bayesian_regret"]) <= 0.1,
        "independent_ego_regret": float(game["oracle"]["ego"]["normalized_regret"]) <= 0.1,
        "independent_target_regret": max(
            float(item["normalized_regret"]) for item in game["oracle"]["targets_by_type"].values()
        )
        <= 0.1,
    }
    return {
        "summary": {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "important_finding": (
                "The Bayesian IGO flow profile still has a material unilateral improvement."
                if not checks["independent_ego_regret"] or not checks["independent_target_regret"]
                else "The synchronous Bayesian IGO flow profile passed the independent regret audit."
            ),
        },
        "cost_consistency": consistency,
        "belief_filter": belief,
        "implicit_contingency": continuation,
        "temporal_collision": temporal_collision,
        "bayesian_game_regret": game,
    }


def _belief_validation() -> dict:
    profiles = default_actor_type_profiles()
    names = tuple(profile.name for profile in profiles)
    speeds = {"yielding": 7.0, "normal": 10.0, "aggressive": 13.0}
    result = {}
    for true_type in names:
        belief = ActorTypeBelief("target", {profile.name: profile.prior_probability for profile in profiles})
        filter_ = BayesianTypeFilter(BayesianTypeFilterConfig())
        previous = None
        for step in range(15):
            current_time = 0.25 * step
            predictions = {
                name: LongitudinalTypePrediction(
                    s=speeds[name] * current_time,
                    speed=speeds[name],
                    acceleration=0.0,
                    start_s=speeds[name] * max(current_time - 0.25, 0.0),
                    start_speed=speeds[name],
                )
                for name in names
            }
            observation = LongitudinalObservation(
                time=current_time,
                s=speeds[true_type] * current_time,
                speed=speeds[true_type],
                acceleration=0.0,
            )
            filter_.update(belief, observation, predictions, previous)
            previous = observation
        result[true_type] = {
            "true_type_probability": float(belief.probabilities[true_type]),
            "probabilities": dict(belief.probabilities),
        }
    return result


def _continuation_validation() -> dict:
    evaluator = BeliefContinuationEvaluator(
        RiskAggregator(RiskAggregatorConfig(expected_weight=1.0, cvar_weight=0.35, cvar_alpha=0.25))
    )
    prior = np.array([0.5, 0.5], dtype=float)
    different_costs = np.array([[10.0], [100.0]], dtype=float)
    same_costs = np.array([[10.0], [10.0]], dtype=float)

    def score(costs, features):
        return float(evaluator.evaluate_batch(costs, features, prior)["implicit_contingency_score"][0])

    separable = score(different_costs, np.array([[[0.0, 0.0, 0.0]], [[100.0, 20.0, 10.0]]]))
    indistinguishable = score(different_costs, np.zeros((2, 1, 3), dtype=float))
    same_cost = score(same_costs, np.zeros((2, 1, 3), dtype=float))
    monotonic_scores = [
        score(different_costs, np.array([[[0.0, 0.0, 0.0]], [[separation, 0.0, 0.0]]]))
        for separation in (0.0, 1.0, 3.0, 10.0)
    ]
    passed = (
        abs(separable) <= 1e-9
        and indistinguishable > 0.0
        and abs(same_cost) <= 1e-9
        and all(a >= b - 1e-12 for a, b in zip(monotonic_scores, monotonic_scores[1:]))
    )
    return {
        "passed": bool(passed),
        "separable_different_cost_score": separable,
        "indistinguishable_different_cost_score": indistinguishable,
        "indistinguishable_same_cost_score": same_cost,
        "separability_monotonic_scores": monotonic_scores,
    }


def _temporal_collision_validation() -> dict:
    cost = ParametricTrajectoryCost(ParametricTrajectoryCostConfig(trajectory_certificate_enabled=False))
    trajectory = Trajectory(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        s=np.array([0.0, 10.0, 20.0], dtype=float),
        l=np.zeros((3,), dtype=float),
    )

    def overlap(actor_s):
        t = trajectory.t.copy()
        s = np.asarray(actor_s, dtype=float)
        actor = ActorPrediction(
            actor_id="moving_actor",
            actor_type="vehicle",
            times=t,
            x=s.copy(),
            y=np.zeros_like(s),
            yaw=np.zeros_like(s),
            length=4.0,
            width=2.0,
            metadata={
                "temporal_blocked_range": {
                    "t": t,
                    "s_min": s - 2.0,
                    "s_max": s + 2.0,
                    "l_min": np.full_like(s, -1.0),
                    "l_max": np.full_like(s, 1.0),
                }
            },
        )
        problem = PlanningProblem(
            ego=EgoState(s=0.0, l=0.0, s_v=10.0),
            ref_path=None,
            road_boundary=RoadBoundary(left_l=10.0, right_l=-10.0),
            horizon=2.0,
            dt=1.0,
            actors=(actor,),
        )
        blocked = cost._blocked_ranges(problem)
        return cost._collision_running_terms(trajectory, blocked)["collision_overlap"].tolist()

    different_times = overlap([20.0, 20.0, 10.0])
    same_time = overlap([20.0, 10.0, 0.0])
    final_sample = overlap([40.0, 30.0, 20.0])
    passed = different_times == [0.0, 0.0, 0.0] and same_time == [0.0, 1.0, 0.0] and final_sample == [0.0, 0.0, 1.0]
    return {
        "passed": bool(passed),
        "different_times_overlap": different_times,
        "same_time_overlap": same_time,
        "final_sample_overlap": final_sample,
    }


def _bayesian_game_validation(oracle_samples: int) -> dict:
    args = demo.build_arg_parser().parse_args(
        [
            "--scenario",
            "interactive_lane_change",
            "--trajectory-model",
            "frenet_bspline_trajectory",
            "--set",
            "planner.type=bayesian_game",
            "--set",
            "game.simulated_target_type=aggressive",
            "--set",
            "optimizer.samples=18",
            "--set",
            "optimizer.iterations=30",
            "--set",
            "planner.mode_paths=4",
            "--no-show",
        ]
    )
    args.demo_config_values = load_demo_config(args)
    args.scenario_config_values = load_scenario_config(args)
    resolve_demo_runtime_args(args)
    scenario = build_scenario(args)
    model = build_trajectory_model(args)
    planner = build_planner(args, model)
    problem = scenario.build_problem(scenario.initial_state())

    start = time.perf_counter()
    result = planner.plan(problem)
    plan_ms = 1000.0 * (time.perf_counter() - start)
    optimization_problem = planner.last_optimization_problem
    if optimization_problem is None:
        raise RuntimeError("Bayesian planner did not expose its last optimization problem.")
    parameters = result.metadata["game_best_parameters"]
    start = time.perf_counter()
    oracle = BayesianRegretOracle(
        BayesianRegretOracleConfig(
            uniform_samples=max(int(oracle_samples), 0),
            axis_samples_per_dimension=21,
            include_initial_population=True,
            seed=101,
        )
    ).evaluate(optimization_problem, parameters)
    oracle_ms = 1000.0 * (time.perf_counter() - start)
    internal = dict(result.metadata.get("game_metadata", {}))
    return {
        "planner_status": result.status,
        "plan_time_ms": plan_ms,
        "oracle_time_ms": oracle_ms,
        "flow_profile_max_bayesian_regret": float(internal.get("max_bayesian_regret", float("inf"))),
        "flow_profile_ego_bayesian_regret": float(internal.get("ego_bayesian_regret", float("inf"))),
        "online_local_max_bayesian_regret": float(
            internal.get("local_max_bayesian_regret", float("inf"))
        ),
        "online_local_equilibrium_converged": float(
            internal.get("local_nash_converged", 0.0)
        ),
        "online_local_check_count": int(internal.get("local_nash_check_count", 0)),
        "best_response_feedback_count": int(internal.get("best_response_feedback_count", 0)),
        "internal_stop_reason": str(internal.get("stop_reason", "")),
        "oracle": oracle.as_dict(),
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# 初始算法正确性验证报告",
        "",
        f"总体结果：**{'通过' if summary['passed'] else '未通过'}**",
        "",
        f"关键发现：{summary['important_finding']}",
        "",
        "## 验收检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")

    lines.extend(["", "## Scalar / Batch Cost 一致性", "", "| 模型 | 最大相对误差 | 最大绝对误差 | 结果 |", "|---|---:|---:|---:|"])
    for name, item in report["cost_consistency"].items():
        lines.append(
            f"| `{name}` | {item['max_relative_error']:.3e} | {item['max_absolute_error']:.3e} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )

    game = report["bayesian_game_regret"]
    oracle = game["oracle"]
    lines.extend(
        [
            "",
            "## Bayesian Regret",
            "",
            f"- 规划状态：`{game['planner_status']}`",
            f"- 规划耗时：{game['plan_time_ms']:.2f} ms",
            f"- Flow profile 最大 Bayesian regret：{game['flow_profile_max_bayesian_regret']:.6f}",
            f"- 在线局部最大 Bayesian regret：{game['online_local_max_bayesian_regret']:.6f}",
            f"- 在线局部扰动检查次数：{game['online_local_check_count']}",
            f"- 最佳响应反馈次数：{game['best_response_feedback_count']}",
            f"- 独立 oracle 自车 regret：{oracle['ego']['normalized_regret']:.6f}",
            f"- 独立 oracle 最大 regret：{oracle['max_regret']:.6f}",
            "",
            "| 玩家 | 当前 cost | 最佳响应 cost | 独立 regret | 候选数 |",
            "|---|---:|---:|---:|---:|",
            (
                f"| ego | {oracle['ego']['current_cost']:.3f} | {oracle['ego']['best_response_cost']:.3f} | "
                f"{oracle['ego']['normalized_regret']:.6f} | {oracle['ego']['candidate_count']} |"
            ),
        ]
    )
    for type_name, item in oracle["targets_by_type"].items():
        lines.append(
            f"| target:{type_name} | {item['current_cost']:.3f} | {item['best_response_cost']:.3f} | "
            f"{item['normalized_regret']:.6f} | {item['candidate_count']} |"
        )
    conclusion = (
        [
            "基础数值模块与独立最佳响应验证均已通过。",
            "当前早停结果可以在 regret 阈值 0.1 下解释为近似 Bayesian-Nash 均衡。",
        ]
        if summary["passed"]
        else [
            "基础数值模块已经通过首轮验证，但当前内部 Bayesian equilibrium 检查仍存在候选集覆盖不足。",
            "在独立 oracle 验证通过前，不能将当前早停结果解释为近似 Bayesian-Nash 均衡。",
        ]
    )
    lines.extend(["", "## 当前结论", "", *conclusion, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the initial correctness validation suite.")
    parser.add_argument("--oracle-samples", type=int, default=1024)
    parser.add_argument("--output", type=str, default="docs/validation/initial_validation_report.md")
    parser.add_argument("--json-output", type=str, default="docs/validation/initial_validation_report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build_report(oracle_samples=int(args.oracle_samples))
    markdown_path = Path(args.output)
    json_path = Path(args.json_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_markdown(report))
    print(f"markdown_report={markdown_path}")
    print(f"json_report={json_path}")
    if bool(args.strict) and not bool(report["summary"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
