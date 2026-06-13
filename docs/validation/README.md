# 算法正确性验证

当前验证体系用于回答三个问题：

1. 优化器使用的 batch cost 是否与最终 scalar cost 一致。
2. belief、时间碰撞和隐式 contingency 是否满足基本性质。
3. Bayesian IGO 输出是否真的接近类型条件均衡。

## 运行单元测试

```powershell
python -B -m unittest discover -s tests -v
```

## 生成完整验证报告

```powershell
python -B -m spatiotemporal_joint_planner.validation.run_validation
```

默认输出：

- `docs/validation/initial_validation_report.md`
- `docs/validation/initial_validation_report.json`

在 CI 中可以使用严格模式。只要任何验收项失败，命令就返回非零状态：

```powershell
python -B -m spatiotemporal_joint_planner.validation.run_validation --strict
```

## Regret Oracle

独立 regret oracle 不读取 IGO 的内部候选集、component 或收敛诊断。它重新构造：

- 参数边界内均匀采样；
- 每个参数维度的轴向扫描；
- 初始 warm-start anchors。

随后分别固定其他玩家，重新搜索每个玩家的单边最佳响应。只有独立 oracle 的 regret 足够小，当前解才可以被解释为近似 Bayesian-Nash 均衡。
