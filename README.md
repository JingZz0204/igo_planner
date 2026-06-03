# Spatiotemporal Joint Planner

`spatiotemporal_joint_planner` 是一个面向自动驾驶局部规划实验的时空联合决策规划原型项目。项目目标是：在每一帧规划中输出固定未来时域的轨迹，例如未来 5 秒轨迹，同时支持不同轨迹参数化方式、不同优化器、不同场景和可插拔的 cost 设计。

当前项目重点关注：

- 固定时域的时空联合轨迹规划。
- 参数化轨迹优化，当前主要使用 CMA-ES。
- `lattice_trajectory` 与 `bezier_trajectory` 两类轨迹模型。
- `static_nudge` 与 `lane_change` 场景。
- 分层 cost，包括碰撞、道路边界、动态约束、效率、参考线吸引和 trajectory-level certificate cost。
- warm start 机制，用于提升采样优化的稳定性和收敛速度。

## 项目特点

### 固定未来时域规划

规划问题使用固定 horizon，例如：

```text
T = 5.0 s
```

每一帧都输出未来 `T` 秒轨迹。轨迹模型可以优化横向终点、终端速度、Bezier 控制点等变量，但不通过改变 `T` 来规避问题。

### 可扩展工程分层

项目将规划系统拆成几个独立模块：

```text
scenario            负责构建场景和 PlanningProblem
trajectory_models   负责轨迹参数化和解码
planner             负责规划流程和 warm start 管理
optimizer           负责数值优化
cost                负责轨迹评价
common              负责共享数据结构
```

每个核心模块都有抽象基类，便于后续扩展新的场景、优化器、轨迹模型和 cost。

### 支持分层 cost

当前 `ParametricTrajectoryCost` 使用显式等级编码，高等级 cost dominate 低等级 cost：

```text
1e9 * collision_score
+ 1e8 * road_score
+ 1e7 * kappa_hard_score
+ 1e6 * dkappa_hard_score
+ 1e5 * lateral_accel_hard_score
+ 1e4 * lateral_jerk_hard_score
+ 1e3 * efficiency_score
+ 1e2 * trajectory_certificate_score
+ 1e1 * comfort_score
+ 1e0 * reference_score
```

其中 `trajectory_certificate_score` 是轨迹级非马尔可夫 cost 的统一入口，目前已经实现 terminal value，后续可继续扩展 occupation-style feature 和 certificate slack。

## 可视化效果

以下 GIF 使用 `static_nudge`、`lane_change`、`interactive_lane_change` 场景，以及 `lattice_trajectory`、`bezier_trajectory` 两种参数化模式生成，用于快速观察规划器在静态绕行、换道和交互换道任务中的轨迹行为。

| 场景 | 轨迹模型 | 可视化 |
| --- | --- | --- |
| `static_nudge` | `lattice_trajectory` | ![static_nudge lattice](docs/assets/static_nudge_lattice.gif) |
| `lane_change` | `lattice_trajectory` | ![lane_change lattice](docs/assets/lane_change_lattice.gif) |
| `static_nudge` | `bezier_trajectory` | ![static_nudge bezier](docs/assets/static_nudge_bezier.gif) |
| `lane_change` | `bezier_trajectory` | ![lane_change bezier](docs/assets/lane_change_bezier.gif) |

### 交互换道 Keep 模式对比

以下三组均使用 `interactive_lane_change` 场景，并保持目标车道后车 `interaction_mode=keep`。这里只改变目标车道后车的初始位置和速度，用来观察不同后车压力下的换道规划行为。

当前 `config/scenario/interactive_lane_change.yaml` 配置下的慢速 GIF：

<img src="docs/assets/interactive_lane_change_current_config.gif" width="860" alt="interactive lane change current config">

| 后车配置 | 可视化 |
| --- | --- |
| `s=-28 m`, `v=10 m/s` | <img src="docs/assets/interactive_lane_change_rear_far_slow.gif" width="860" alt="interactive lane change rear far slow"> |
| `s=-18 m`, `v=15 m/s` | <img src="docs/assets/interactive_lane_change_rear_nominal.gif" width="860" alt="interactive lane change rear nominal"> |
| `s=-8 m`, `v=18 m/s` | <img src="docs/assets/interactive_lane_change_rear_close_fast.gif" width="860" alt="interactive lane change rear close fast"> |

## 安装

建议使用 Python 3.9 或更高版本。

```bash
pip install -r requirements.txt
```

当前基础依赖：

```text
numpy
matplotlib
pyyaml
```

如果需要保存 MP4，系统需要有 `ffmpeg`，或者 Python 环境中安装 `imageio-ffmpeg`。

## 快速运行

### Static Nudge 场景

使用 lattice trajectory：

```bash
python -B -m spatiotemporal_joint_planner.demo --scenario static_nudge --trajectory-model lattice_trajectory --show
```

使用 Bezier trajectory：

```bash
python -B -m spatiotemporal_joint_planner.demo --scenario static_nudge --trajectory-model bezier_trajectory --show
```

### Lane Change 场景

```bash
python -B -m spatiotemporal_joint_planner.demo --scenario lane_change --trajectory-model lattice_trajectory --scenario-set target_speed=12.0 --show
```

### 无可视化 smoke test

```bash
python -B -m spatiotemporal_joint_planner.demo --scenario static_nudge --trajectory-model lattice_trajectory --max-steps 3 --set optimizer.samples=24 --set optimizer.iterations=3 --set optimizer.components=4 --no-show
```

### 保存 MP4

```bash
python -B -m spatiotemporal_joint_planner.demo --scenario lane_change --trajectory-model lattice_trajectory --max-steps 80 --save-mp4 demo_outputs/lane_change.mp4 --set visualization.mp4_fps=10 --no-show
```

## 目录结构

```text
spatiotemporal_joint_planner/
  __init__.py
  demo.py

  common/
    types.py
    __init__.py

  config/
    demo/
      default.yaml
    scenario/
      static_nudge.yaml
      lane_change.yaml
      interactive_lane_change.yaml

  cost/
    base.py
    parametric_trajectory_cost.py
    __init__.py

  optimizer/
    base.py
    cma_es_optimizer.py
    __init__.py

  planner/
    base.py
    parametric_planner.py
    __init__.py
    warm_start/
      base.py
      terminal_state.py
      bezier.py
      __init__.py

  scenario/
    base.py
    static_nudge.py
    lane_change.py
    interactive_lane_change.py
    __init__.py

  trajectory_models/
    base.py
    common.py
    lattice_trajectory.py
    bezier_trajectory.py
    svgd_particle_trajectory.py
    __init__.py
```

## 核心数据结构

核心数据结构位于：

```text
spatiotemporal_joint_planner/common/types.py
```

主要类型包括：

### EgoState

描述自车当前 Frenet 状态：

```text
s      纵向位置
l      横向位置
s_v    纵向速度
l_v    横向速度
s_a    纵向加速度
l_a    横向加速度
yaw    航向角
t      当前时间
```

### PlanningProblem

每一帧规划的完整输入：

```text
ego              自车状态
ref_path         参考线
road_boundary    道路边界
horizon          规划时域
dt               轨迹采样时间间隔
actors           障碍物预测
metadata         场景附加信息
```

### Trajectory

规划器输出的轨迹：

```text
t        时间序列
s, l     Frenet 位置
s_v,l_v  Frenet 速度
s_a,l_a  Frenet 加速度
x,y,yaw  世界坐标轨迹
v,a      速度和加速度
kappa    曲率
```

### CostResult

cost 计算结果：

```text
total       总 cost
breakdown   分项 cost
feasible    是否满足硬约束
metadata    附加信息
```

## 场景模块

场景模块位于：

```text
spatiotemporal_joint_planner/scenario/
```

所有场景继承自：

```python
Scenario
```

场景负责提供：

```text
initial_state()
build_problem(ego, t)
actors_at(t)
```

### StaticNudgeScenario

`static_nudge` 用于测试静态障碍物绕行能力。它构造一条道路参考线、道路边界、车道线和静态障碍物，并在每一帧生成 `PlanningProblem`。

该场景主要用于测试：

- 静态障碍物绕行。
- 道路边界约束。
- 参考线吸引。
- 终端状态 warm start。
- trajectory-level certificate cost 是否会影响轨迹选择。

### LaneChangeScenario

`lane_change` 用于测试换道规划能力。该场景包含：

- 双车道道路。
- 当前车道中心线。
- 目标车道中心线。
- 当前车道慢车。
- 目标车道前车。

场景中 `ref_path` 会使用目标车道中心线，以测试规划器在行驶过程中向目标车道规划的能力。

## 轨迹模型

轨迹模型位于：

```text
spatiotemporal_joint_planner/trajectory_models/
```

所有轨迹模型继承自：

```python
TrajectoryModel
```

主要接口：

```python
parameter_dim(problem)
bounds(problem)
reference_parameters(problem)
decode(parameters, problem)
```

### LatticeTrajectoryModel

`lattice_trajectory` 是 Frenet 终端状态参数化模型。

当前参数为：

```text
theta = [l_end, v_end]
```

其中：

```text
l_end  规划终点横向位置
v_end  规划终点纵向速度
T      固定，由 PlanningProblem.horizon 给出
```

横向轨迹使用固定时域多项式连接当前横向状态与终端横向状态。纵向轨迹使用固定时域速度 profile 连接当前速度与终端速度。

### BezierTrajectoryModel

`bezier_trajectory` 使用 ego-local Bezier 控制点表示轨迹形状。

它适合表达比简单终端状态模型更复杂的中间轨迹形态，例如绕行过程中间点的横向偏移。

### SvgdParticleTrajectoryModel

`svgd_particle_trajectory` 当前继承 lattice 终端状态模型，用于后续非参数化 / 粒子化优化方向的接口预留。

## Planner 模块

规划器位于：

```text
spatiotemporal_joint_planner/planner/
```

### ParametricPlanner

`ParametricPlanner` 是当前主要规划器。其流程为：

```text
1. 从 trajectory_model 获取参数 bounds
2. 生成 warm start initial population
3. 构造 optimization objective
4. 使用 optimizer 优化参数
5. decode 最优参数为 trajectory
6. 重新 evaluate 最优 trajectory
7. 返回 PlannerResult
```

整体数据流：

```text
PlanningProblem
  -> WarmStartGenerator
  -> initial_population
  -> CMAESOptimizer
  -> best_parameters
  -> TrajectoryModel.decode()
  -> ParametricTrajectoryCost.evaluate()
  -> PlannerResult
```

## Warm Start

warm start 位于：

```text
spatiotemporal_joint_planner/planner/warm_start/
```

设计目标是：在采样优化之前提供高质量初始解，降低 CMA-ES 的随机性，提高稳定性和收敛速度。

### TerminalStateWarmStartGenerator

用于：

```text
lattice_trajectory
svgd_particle_trajectory
```

它会采样：

```text
l_end: 道路左右可行边界之间的横向终点
v_end: 当前速度 * 0.3 到 当前速度 * 2.0 之间的速度样本
```

同时考虑：

- 终端速度最小值。
- 终端速度最大值。
- 道路边界。
- 自车宽度。
- 静态障碍物 blocked range。
- 车道中心线候选。

### BezierTrajectoryWarmStartGenerator

用于：

```text
bezier_trajectory
```

它先生成语义级 Frenet seed，例如不同横向目标、终端速度、横向完成时间，然后将这些 seed 拟合成 Bezier 控制点。

这种方式能给 Bezier 模型更好的初始形状，而不是完全依赖随机采样。

## Optimizer 模块

优化器位于：

```text
spatiotemporal_joint_planner/optimizer/
```

所有优化器继承自：

```python
Optimizer
```

### CMAESOptimizer

当前主要优化器是多模态 CMA-ES。

输入：

```text
OptimizationProblem
  objective
  initial_population
  lower_bound
  upper_bound
```

输出：

```text
OptimizationResult
  best_position
  best_value
  population
  values
  history
```

CMA-ES 会先评估 warm start anchors，再用 anchors 初始化多个 component，之后迭代采样、评估、排序、更新分布。

## Cost 模块

cost 位于：

```text
spatiotemporal_joint_planner/cost/
```

所有 cost 继承自：

```python
CostFunction
```

### ParametricTrajectoryCost

当前主要 cost 是 `ParametricTrajectoryCost`。

它分为两层：

```text
Markov running terms
Trajectory-level certificate terms
```

### Markov Running Terms

这些 cost 可以理解为每个轨迹点局部状态的代价：

```text
collision_running
road_running
lateral_accel_limit_running
lateral_accel_zero_running
kappa_limit_running
kappa_zero_running
dkappa_limit_running
dkappa_zero_running
lateral_jerk_limit_running
lateral_jerk_zero_running
speed_limit_running
efficiency_running
reference_lateral_running
```

其中硬约束包括：

- 与障碍物真实 overlap。
- 自车 box 出道路边界。
- 曲率超限。
- 曲率变化率超限。
- 横向加速度超限。
- 横向 jerk 超限。
- 速度超限或倒车。

软约束包括：

- 横向加速度向 0 偏差。
- 横向 jerk 向 0 偏差。
- 曲率向 0 偏差。
- 曲率变化率向 0 偏差。
- 速度向最大速度 / 目标速度靠近。
- 参考线吸引。

### Trajectory-Level Certificate Cost

统一入口：

```python
_trajectory_level_certificate_terms(...)
```

当前已实现：

```text
terminal_value_score
```

预留接口：

```text
occupation_style_score
certificate_slack_score
```

#### Terminal Value

terminal value 用于评价：

```text
这条 5 秒轨迹的终点，是否是一个后续容易继续规划的状态
```

当前包含：

```text
terminal_future_feasibility_score
terminal_recoverability_score
terminal_progress_score
terminal_speed_score
```

其中：

- `terminal_future_feasibility_score` 从轨迹终点向未来 rollout 多条简单 continuation primitive，评估终点之后的可行性。
- `terminal_recoverability_score` 评价终点的 `dl/ds`、`kappa`、`dkappa`、横向加速度、横向 jerk 和边界距离。
- `terminal_progress_score` 评价终点之后是否还能保持足够前向推进。
- `terminal_speed_score` 评价终点速度是否过低。

该模块可以通过 demo 配置关闭：

```bash
--set cost.trajectory_certificate_enabled=false
```

## Demo 可视化

入口：

```text
spatiotemporal_joint_planner/demo.py
```

常用入口：

```text
--scenario                         static_nudge / lane_change / interactive_lane_change
--config                           demo YAML 配置文件路径
--set                              覆盖 demo YAML 配置，例如 optimizer.samples=32
--scenario-config                  scenario YAML 配置文件路径
--scenario-set                     覆盖 scenario YAML 配置，例如 actors.current_slow.s=30
--trajectory-model                 lattice_trajectory / bezier_trajectory / svgd_particle_trajectory
--max-steps                        覆盖 runtime.max_steps
--show / --no-show                 开启或关闭动画
--save-frame                       保存单帧图片
--save-mp4                         保存 MP4
```

调参项默认写在：

```text
spatiotemporal_joint_planner/config/demo/default.yaml
spatiotemporal_joint_planner/config/scenario/<scenario>.yaml
```

日志中包含：

```text
collision  碰撞硬约束 flag
road       道路边界硬约束 flag
lat_acc    横向加速度硬约束 flag
k          曲率硬约束 flag
dk         曲率变化率硬约束 flag
jerk       横向 jerk 硬约束 flag
speed      速度硬约束 flag
eff        效率 cost
ref_l      参考线横向 cost
dyn        动态硬约束诊断分数
tv         terminal value score
tf         terminal future feasibility score
tr         terminal recoverability score
tp         terminal progress score
ts         terminal speed score
```

## 当前限制

当前项目仍是原型阶段，主要限制包括：

- 目前主要实现参数化 planner，非参数化 SVGD planner 仍在预留阶段。
- `lattice_trajectory` 目前参数维度较低，只包含 `l_end` 与 `v_end`。
- `trajectory-level certificate cost` 目前只实现 terminal value，occupation-style feature 和 certificate slack 仍是空实现。
- batch/vectorized 样本评估已经支持 `lattice_trajectory` 和 `bezier_trajectory`，但参考线投影仍是主要性能瓶颈。
- 动态障碍物交互 cost 尚未完整实现。

## 后续计划

比较明确的后续方向：

- 为 `lattice_trajectory` 增加更强的中间形态表达能力。
- 实现 occupation-style trajectory feature。
- 实现 certificate slack cost。
- 优化参考线投影、candidate 解码和 batch decode 性能。
- 实现真正的 SVGD 非参数化粒子轨迹优化。
- 增加动态障碍物场景和交互 cost。
- 增加单元测试和 benchmark 场景。

## 发布建议

如果只发布当前项目，建议 GitHub 仓库根目录保持如下结构：

```text
README.md
requirements.txt
pyproject.toml
.gitignore
spatiotemporal_joint_planner/
```

不要把 `spatiotemporal_joint_planner/` 内部文件摊平到根目录，因为项目使用包级绝对导入：

```python
from spatiotemporal_joint_planner.common import ...
```

生成的图片、视频和缓存文件不要提交到仓库。
