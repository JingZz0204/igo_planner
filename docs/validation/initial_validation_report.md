# 初始算法正确性验证报告

总体结果：**通过**

关键发现：The synchronous Bayesian IGO flow profile passed the independent regret audit.

## 验收检查

| 检查项 | 结果 |
|---|---:|
| `scalar_batch_cost_consistency` | PASS |
| `belief_identification` | PASS |
| `continuation_properties` | PASS |
| `temporal_collision_alignment` | PASS |
| `flow_profile_bayesian_regret` | PASS |
| `online_independent_bayesian_regret` | PASS |
| `independent_ego_regret` | PASS |
| `independent_target_regret` | PASS |

## Scalar / Batch Cost 一致性

| 模型 | 最大相对误差 | 最大绝对误差 | 结果 |
|---|---:|---:|---:|
| `lattice_trajectory` | 9.257e-11 | 2.957e-05 | PASS |
| `frenet_bspline_trajectory` | 1.808e-13 | 5.960e-07 | PASS |
| `frenet_bezier_trajectory` | 2.400e-14 | 2.265e-06 | PASS |
| `frenet_via_bspline_trajectory` | 4.866e-15 | 3.338e-06 | PASS |

## Bayesian Regret

- 规划状态：`success`
- 规划耗时：786.91 ms
- Flow profile 最大 Bayesian regret：0.048878
- 在线独立最大 Bayesian regret：0.048878
- 在线独立检查次数：1
- 最佳响应反馈次数：0
- 独立 oracle 自车 regret：0.000000
- 独立 oracle 最大 regret：0.038406

| 玩家 | 当前 cost | 最佳响应 cost | 独立 regret | 候选数 |
|---|---:|---:|---:|---:|
| ego | 543.752 | 543.752 | 0.000000 | 1183 |
| target:yielding | 6606.837 | 6520.093 | 0.013129 | 1098 |
| target:normal | 2290.640 | 2202.665 | 0.038406 | 1098 |
| target:aggressive | 358.288 | 358.288 | 0.000000 | 1098 |

## 当前结论

基础数值模块与独立最佳响应验证均已通过。
当前早停结果可以在 regret 阈值 0.1 下解释为近似 Bayesian-Nash 均衡。
