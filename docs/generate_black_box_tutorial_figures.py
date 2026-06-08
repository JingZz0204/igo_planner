"""Generate diagrams for the black-box trajectory-planning tutorial."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


OUTPUT_DIR = Path(__file__).resolve().parent / "assets" / "black_box_tutorial"

COLORS = {
    "navy": "#23405A",
    "blue": "#3B82C4",
    "cyan": "#4CA7A5",
    "green": "#58A66A",
    "orange": "#E98A3A",
    "red": "#D95D5D",
    "purple": "#8266A5",
    "yellow": "#E8C547",
    "gray": "#718096",
    "light": "#F5F7FA",
    "ink": "#1F2933",
}


plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 11,
        "figure.dpi": 140,
    }
)


def new_canvas(title: str, subtitle: str = "", figsize: tuple[float, float] = (12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.35, 5.65, title, fontsize=20, weight="bold", color=COLORS["ink"], va="top")
    if subtitle:
        ax.text(0.37, 5.27, subtitle, fontsize=10.5, color=COLORS["gray"], va="top")
    return fig, ax


def rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = COLORS["navy"],
    textcolor: str = COLORS["ink"],
    fontsize: float = 11,
    linewidth: float = 1.5,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.45,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["gray"],
    width: float = 1.5,
    connectionstyle: str = "arc3",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=width,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    return patch


def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_road(ax, y0=1.0, y1=4.4):
    ax.plot([0.6, 11.4], [y0, y0], color=COLORS["ink"], lw=2)
    ax.plot([0.6, 11.4], [y1, y1], color=COLORS["ink"], lw=2)
    ax.plot([0.6, 11.4], [(y0 + y1) / 2] * 2, color="#B7BEC7", lw=1.5, ls="--")


def vehicle(ax, x, y, color, label=""):
    rounded_box(
        ax,
        (x - 0.34, y - 0.17),
        0.68,
        0.34,
        "",
        facecolor=color,
        edgecolor=COLORS["ink"],
        linewidth=1.0,
    )
    if label:
        ax.text(x, y - 0.34, label, ha="center", va="top", fontsize=8.5, color=COLORS["ink"])


def figure_01_problem():
    fig, ax = new_canvas(
        "轨迹规划本质：从大量可行未来中选择一条",
        "规划器并不是“画一条线”，而是在动力学、道路与交互约束下优化未来状态序列",
    )
    draw_road(ax, 0.8, 4.5)
    vehicle(ax, 1.3, 1.7, COLORS["orange"], "当前自车")
    vehicle(ax, 6.4, 1.7, COLORS["gray"], "障碍物")
    vehicle(ax, 7.8, 3.6, COLORS["purple"], "他车")

    x = np.linspace(1.7, 10.5, 120)
    candidates = [
        (1.7 + 0.12 * np.sin((x - 1.7) / 1.5), COLORS["gray"], "--", 1.2),
        (1.7 + 1.85 / (1 + np.exp(-(x - 5.5) * 1.5)), COLORS["blue"], "-", 2.7),
        (1.7 + 1.2 * np.exp(-((x - 6.4) / 1.4) ** 2), COLORS["red"], "--", 1.2),
    ]
    for y, color, style, lw in candidates:
        ax.plot(x, y, color=color, ls=style, lw=lw, alpha=0.95)

    rounded_box(
        ax,
        (8.85, 3.65),
        2.4,
        1.25,
        "选择依据\n安全 · 合规 · 效率\n舒适 · 交互 · 长期价值",
        facecolor=COLORS["light"],
        edgecolor=COLORS["green"],
        fontsize=10,
    )
    arrow(ax, (8.45, 3.45), (9.15, 3.65), color=COLORS["green"])
    ax.text(4.55, 4.75, r"$\tau^*=\arg\min_{\tau\in\mathcal{T}} J(\tau)$", fontsize=16, color=COLORS["navy"])
    save(fig, "01_trajectory_optimization_problem.png")


def figure_02_ilqr_vs_blackbox():
    fig, ax = new_canvas(
        "两类优化思路：局部梯度优化与黑箱搜索",
        "两者不是高下之分，而是依赖的假设、擅长的问题形态不同",
    )
    rounded_box(
        ax,
        (0.55, 0.55),
        5.15,
        4.35,
        "",
        facecolor="#F5F9FC",
        edgecolor=COLORS["blue"],
    )
    rounded_box(
        ax,
        (6.3, 0.55),
        5.15,
        4.35,
        "",
        facecolor="#FBF7F1",
        edgecolor=COLORS["orange"],
    )
    ax.text(3.12, 4.55, "iLQR：沿局部几何向下走", ha="center", fontsize=15, weight="bold", color=COLORS["blue"])
    ax.text(8.88, 4.55, "黑箱优化：比较候选解并更新搜索分布", ha="center", fontsize=15, weight="bold", color=COLORS["orange"])

    ax.text(
        0.95,
        3.95,
        "• 围绕名义轨迹线性化动力学\n• 二次近似局部 cost\n• 利用梯度 / Hessian 结构\n• 收敛快，适合平滑 Markov 问题",
        va="top",
        linespacing=1.75,
        color=COLORS["ink"],
    )
    ax.text(
        6.7,
        3.95,
        "• 只要求能生成轨迹并计算 cost\n• 可容纳碰撞、排序、分段与轨迹级指标\n• 同时保留多个搜索模态\n• 代价是样本量与评估耗时",
        va="top",
        linespacing=1.75,
        color=COLORS["ink"],
    )
    ax.plot([1.2, 2.0, 2.8, 3.45, 3.8], [1.6, 1.35, 1.2, 1.12, 1.1], "-o", color=COLORS["blue"], lw=2, ms=4)
    ax.text(2.45, 0.95, "依赖局部近似质量", ha="center", fontsize=9.5, color=COLORS["gray"])
    rng = np.random.default_rng(4)
    pts = rng.normal([8.8, 1.55], [1.0, 0.36], size=(28, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=18, color=COLORS["orange"], alpha=0.6)
    ax.scatter([9.25], [1.35], s=90, marker="*", color=COLORS["red"], zorder=3)
    ax.text(8.85, 0.95, "依赖候选评估与分布更新", ha="center", fontsize=9.5, color=COLORS["gray"])
    save(fig, "02_ilqr_vs_blackbox.png")


def figure_03_black_box_loop():
    fig, ax = new_canvas(
        "黑箱轨迹优化的最小闭环",
        "优化器无需理解道路、碰撞或曲线，只需提出参数并接收一个可比较的评价",
    )
    boxes = [
        ((0.5, 2.35), 1.65, 1.0, "搜索分布\n$p(\\theta)$", COLORS["purple"]),
        ((2.65, 2.35), 1.65, 1.0, "候选参数\n$\\theta_i$", COLORS["blue"]),
        ((4.8, 2.35), 1.9, 1.0, "轨迹解码器\n$D(\\theta_i)$", COLORS["cyan"]),
        ((7.2, 2.35), 1.65, 1.0, "候选轨迹\n$\\tau_i$", COLORS["green"]),
        ((9.35, 2.35), 2.0, 1.0, "代价评估\n$J(\\tau_i)$", COLORS["orange"]),
    ]
    for xy, w, h, text, color in boxes:
        rounded_box(ax, xy, w, h, text, facecolor="white", edgecolor=color, fontsize=12)
    for x1, x2 in [(2.15, 2.65), (4.3, 4.8), (6.7, 7.2), (8.85, 9.35)]:
        arrow(ax, (x1, 2.85), (x2, 2.85))
    arrow(ax, (10.35, 2.25), (1.35, 2.22), color=COLORS["purple"], connectionstyle="arc3,rad=-0.26")
    ax.text(5.9, 1.0, "依据 cost / 排名更新分布，再产生下一代候选", ha="center", color=COLORS["purple"], fontsize=11)
    ax.text(5.9, 4.2, "核心接口", ha="center", fontsize=10, color=COLORS["gray"])
    ax.text(5.9, 3.82, r"$\theta \longrightarrow \tau \longrightarrow J$", ha="center", fontsize=19, color=COLORS["navy"])
    save(fig, "03_black_box_loop.png")


def figure_04_representations():
    fig, ax = new_canvas(
        "轨迹表示决定优化问题的难度上限",
        "自由度越低越稳定，但表达能力越有限；自由度越高越灵活，但搜索与约束更困难",
    )
    panels = [
        (0.45, "终端状态\nLattice", 2, COLORS["blue"]),
        (3.35, "中间点 + 终端\nBezier / B-spline", 3, COLORS["green"]),
        (6.25, "多个控制点\n高维参数化", 7, COLORS["orange"]),
        (9.15, "逐时刻粒子\n近似非参数化", 20, COLORS["purple"]),
    ]
    for x0, title, dim, color in panels:
        rounded_box(ax, (x0, 0.65), 2.45, 3.95, "", facecolor=COLORS["light"], edgecolor=color)
        ax.text(x0 + 1.225, 4.28, title, ha="center", va="top", fontsize=12.5, weight="bold", color=color)
        x = np.linspace(x0 + 0.25, x0 + 2.2, 70)
        if dim == 2:
            y = 1.55 + 1.3 * (3 * ((x - x.min()) / (x.max() - x.min())) ** 2 - 2 * ((x - x.min()) / (x.max() - x.min())) ** 3)
            controls = [(x[0], y[0]), (x[-1], y[-1])]
        elif dim == 3:
            u = (x - x.min()) / (x.max() - x.min())
            y = (1 - u) ** 2 * 1.55 + 2 * (1 - u) * u * 2.9 + u**2 * 2.65
            controls = [(x[0], y[0]), (x[len(x) // 2], 2.9), (x[-1], y[-1])]
        elif dim == 7:
            y = 2.05 + 0.65 * np.sin(np.linspace(-1.0, 2.3, len(x)))
            controls = list(zip(x[::12], y[::12]))
        else:
            y = 2.05 + 0.55 * np.sin(np.linspace(-0.8, 2.4, len(x)))
            controls = list(zip(x[::4], y[::4]))
        ax.plot(x, y, color=color, lw=2)
        cx, cy = zip(*controls)
        ax.scatter(cx, cy, s=22, color=color, edgecolor="white", zorder=3)
        ax.text(x0 + 1.225, 1.0, f"优化维度示意：{dim}", ha="center", color=COLORS["ink"], fontsize=10)
    ax.text(1.7, 0.28, "稳定、易搜索", ha="center", color=COLORS["blue"])
    arrow(ax, (2.7, 0.3), (9.2, 0.3), color=COLORS["gray"])
    ax.text(10.2, 0.28, "灵活、难搜索", ha="center", color=COLORS["purple"])
    save(fig, "04_trajectory_representations.png")


def figure_05_cma_es():
    fig, ax = new_canvas(
        "CMA-ES：让搜索分布自己学会朝哪里收缩",
        "均值描述当前搜索中心，协方差描述方向与尺度；优秀样本共同塑造下一代分布",
    )
    gx, gy = np.meshgrid(np.linspace(0.5, 11.4, 180), np.linspace(0.5, 4.85, 120))
    z = ((gx - 8.8) / 1.35) ** 2 + ((gy - 1.5) / 0.65) ** 2 + 0.28 * np.sin(gx * 1.8) * np.cos(gy * 2.1)
    ax.contour(gx, gy, z, levels=12, colors="#D8DEE6", linewidths=0.8)
    rng = np.random.default_rng(7)
    means = [np.array([2.2, 3.8]), np.array([5.4, 2.8]), np.array([7.7, 1.9]), np.array([8.75, 1.52])]
    covs = [
        np.array([[1.2, -0.35], [-0.35, 0.5]]),
        np.array([[0.85, -0.28], [-0.28, 0.32]]),
        np.array([[0.42, -0.12], [-0.12, 0.16]]),
        np.array([[0.14, -0.03], [-0.03, 0.06]]),
    ]
    colors = [COLORS["purple"], COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for i, (mean, cov, color) in enumerate(zip(means, covs, colors)):
        pts = rng.multivariate_normal(mean, cov, size=22)
        ax.scatter(pts[:, 0], pts[:, 1], s=15, color=color, alpha=0.45)
        eigval, eigvec = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvec[1, -1], eigvec[0, -1]))
        ell = Ellipse(mean, 3.2 * np.sqrt(eigval[-1]), 3.2 * np.sqrt(eigval[0]), angle=angle, fill=False, lw=2, ec=color)
        ax.add_patch(ell)
        ax.text(mean[0], mean[1] + 0.55, f"第 {i + 1} 代", ha="center", color=color, fontsize=9.5, weight="bold")
        if i < len(means) - 1:
            arrow(ax, tuple(mean), tuple(means[i + 1]), color=color, width=1.3)
    ax.scatter([8.8], [1.5], marker="*", s=160, color=COLORS["red"], zorder=5)
    ax.text(9.05, 1.18, "低 cost 区域", color=COLORS["red"], fontsize=10)
    save(fig, "05_cma_es_update.png")


def figure_06_igo():
    fig, ax = new_canvas(
        "IGO：用排名定义“向哪些样本学习”",
        "IGO 将具体搜索分布与目标函数解耦；CMA-ES 可理解为高斯分布族上的一种自然梯度更新",
    )
    steps = [
        ((0.45, 2.2), 2.0, "从 $p_\\eta(\\theta)$\n采样候选", COLORS["purple"]),
        ((3.05, 2.2), 2.0, "解码轨迹并\n计算 $J(\\theta)$", COLORS["blue"]),
        ((5.65, 2.2), 2.0, "按 cost 排名\n得到效用权重", COLORS["orange"]),
        ((8.25, 2.2), 3.0, "沿分布参数的自然梯度\n更新 $\\eta=(\\mu,\\Sigma)$", COLORS["green"]),
    ]
    for xy, width, text, color in steps:
        rounded_box(ax, xy, width, 1.15, text, facecolor="white", edgecolor=color, fontsize=11)
    for start, end in [((2.45, 2.78), (3.05, 2.78)), ((5.05, 2.78), (5.65, 2.78)), ((7.65, 2.78), (8.25, 2.78))]:
        arrow(ax, start, end)
    arrow(ax, (9.75, 2.05), (1.45, 2.05), color=COLORS["purple"], connectionstyle="arc3,rad=-0.27")
    bars_x = np.arange(6)
    heights = np.array([1.0, 0.72, 0.42, 0.18, 0.05, 0.0])
    ax.bar(4.8 + bars_x * 0.35, heights * 0.72, width=0.24, bottom=0.72, color=COLORS["orange"], alpha=0.8)
    ax.text(5.65, 0.48, "绝对 cost 尺度可以很怪，排名仍然可用", ha="center", color=COLORS["gray"], fontsize=10)
    ax.text(6.0, 4.25, r"$\eta \leftarrow \eta + \alpha\,\widetilde{\nabla}_{\eta}\;\mathbb{E}_{\theta\sim p_\eta}[W(J(\theta))]$", ha="center", fontsize=16, color=COLORS["navy"])
    save(fig, "06_igo_rank_update.png")


def figure_07_svgd():
    fig, ax = new_canvas(
        "SVGD：粒子既被低 cost 区域吸引，也彼此排斥",
        "它擅长保留多模态，但经典 SVGD 仍需要目标密度梯度；非光滑 cost 需要额外的梯度估计或平滑策略",
    )
    gx, gy = np.meshgrid(np.linspace(0.4, 11.5, 160), np.linspace(0.45, 4.85, 100))
    density = np.exp(-((gx - 3.5) ** 2 / 1.4 + (gy - 1.8) ** 2 / 0.45)) + 0.9 * np.exp(
        -((gx - 8.6) ** 2 / 1.1 + (gy - 3.4) ** 2 / 0.5)
    )
    ax.contourf(gx, gy, density, levels=15, cmap="YlGnBu", alpha=0.42)
    rng = np.random.default_rng(10)
    pts = rng.uniform([1.3, 1.0], [10.7, 4.35], size=(16, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=65, color=COLORS["purple"], edgecolor="white", zorder=4)
    targets = np.where(pts[:, :1] < 6.0, np.array([3.5, 1.8]), np.array([8.6, 3.4]))
    for p, t in zip(pts[::2], targets[::2]):
        delta = 0.35 * (t - p)
        arrow(ax, tuple(p), tuple(p + delta), color=COLORS["green"], width=1.2)
    p0, p1 = pts[5], pts[8]
    arrow(ax, tuple((p0 + p1) / 2), tuple((p0 + p1) / 2 + np.array([0.0, -0.55])), color=COLORS["orange"], width=1.4)
    ax.text(2.0, 4.55, "吸引项：靠近高概率 / 低 cost 区域", color=COLORS["green"], fontsize=10.5)
    ax.text(7.1, 0.65, "排斥项：防止所有粒子塌缩到同一个解", color=COLORS["orange"], fontsize=10.5)
    save(fig, "07_svgd_particles.png")


def figure_08_receding_horizon():
    fig, ax = new_canvas(
        "滚动时域规划：每帧都看未来 5 秒，但只执行最前面一小段",
        "固定预测时域不代表一次规划执行到底；环境更新后，下一帧会重新优化",
    )
    y_levels = [4.3, 3.25, 2.2, 1.15]
    starts = [1.1, 2.25, 3.4, 4.55]
    for i, (y, start) in enumerate(zip(y_levels, starts)):
        ax.plot([start, start + 6.2], [y, y], color="#D2D8E0", lw=8, solid_capstyle="round")
        ax.plot([start, start + 1.15], [y, y], color=COLORS["orange"], lw=8, solid_capstyle="round")
        ax.plot([start + 1.15, start + 6.2], [y, y], color=COLORS["blue"], lw=8, solid_capstyle="round")
        ax.text(0.45, y, f"帧 {i}", va="center", color=COLORS["ink"])
        ax.text(start + 0.57, y + 0.27, "执行", ha="center", fontsize=8.5, color=COLORS["orange"])
        ax.text(start + 3.6, y + 0.27, "本帧规划的未来 5 s", ha="center", fontsize=8.5, color=COLORS["blue"])
        ax.plot([start, start], [y - 0.18, y + 0.18], color=COLORS["ink"], lw=1.2)
    ax.text(8.9, 4.7, "上一帧最优解\n可作为下一帧 warm start", ha="center", color=COLORS["purple"], fontsize=10.5)
    arrow(ax, (8.4, 4.48), (5.1, 3.45), color=COLORS["purple"], connectionstyle="arc3,rad=0.15")
    ax.text(9.15, 0.75, "时间向前推进", color=COLORS["gray"])
    arrow(ax, (8.2, 0.82), (10.8, 0.82), color=COLORS["gray"])
    save(fig, "08_receding_horizon.png")


def figure_09_prediction_vs_game():
    fig, ax = new_canvas(
        "从“固定预测”到“联合博弈”",
        "固定预测把他车当成环境；联合规划则让双方轨迹相互影响、共同被搜索",
    )
    rounded_box(ax, (0.45, 0.7), 5.15, 4.0, "", facecolor="#F5F9FC", edgecolor=COLORS["blue"])
    rounded_box(ax, (6.4, 0.7), 5.15, 4.0, "", facecolor="#FBF7F1", edgecolor=COLORS["orange"])
    ax.text(3.02, 4.38, "固定预测", ha="center", fontsize=15, weight="bold", color=COLORS["blue"])
    ax.text(8.98, 4.38, "联合轨迹博弈", ha="center", fontsize=15, weight="bold", color=COLORS["orange"])

    vehicle(ax, 1.35, 1.55, COLORS["orange"], "自车")
    vehicle(ax, 1.35, 3.35, COLORS["purple"], "他车")
    ax.plot([1.75, 4.8], [3.35, 3.35], color=COLORS["purple"], lw=2, ls="--")
    ax.plot([1.75, 4.8], [1.55, 2.75], color=COLORS["blue"], lw=2)
    ax.text(3.1, 3.62, "预测先固定", ha="center", fontsize=9.5, color=COLORS["purple"])
    ax.text(3.1, 1.25, "只优化自车响应", ha="center", fontsize=9.5, color=COLORS["blue"])

    vehicle(ax, 7.25, 1.55, COLORS["orange"], "自车")
    vehicle(ax, 7.25, 3.35, COLORS["purple"], "他车")
    ax.plot([7.65, 10.8], [1.55, 2.8], color=COLORS["orange"], lw=2.3)
    ax.plot([7.65, 10.8], [3.35, 3.65], color=COLORS["purple"], lw=2.3)
    arrow(ax, (8.55, 2.05), (8.8, 3.25), color=COLORS["green"], connectionstyle="arc3,rad=-0.25")
    arrow(ax, (9.4, 3.25), (9.55, 2.45), color=COLORS["green"], connectionstyle="arc3,rad=-0.25")
    ax.text(9.15, 1.15, "双方策略耦合，分别拥有自己的 cost", ha="center", fontsize=9.5, color=COLORS["ink"])
    save(fig, "09_single_prediction_vs_game.png")


def figure_10_nash():
    fig, ax = new_canvas(
        "近似 Nash / regret 检查：任何一方单独改轨迹都不再明显获益",
        "这比“联合 cost 不再变化”更接近博弈收敛的语义",
    )
    rounded_box(
        ax,
        (4.25, 2.25),
        3.5,
        1.15,
        "当前联合解\n$(\\theta_{ego}^*,\\theta_{rear}^*)$",
        facecolor=COLORS["light"],
        edgecolor=COLORS["navy"],
        fontsize=12,
    )
    rounded_box(
        ax,
        (0.55, 3.85),
        3.05,
        1.0,
        "固定后车\n仅让自车单边偏离",
        facecolor="white",
        edgecolor=COLORS["orange"],
    )
    rounded_box(
        ax,
        (8.4, 3.85),
        3.05,
        1.0,
        "固定自车\n仅让后车单边偏离",
        facecolor="white",
        edgecolor=COLORS["purple"],
    )
    rounded_box(
        ax,
        (0.55, 0.75),
        3.05,
        1.0,
        r"$r_{ego}=J_{ego}^*-J_{ego}^{BR}$",
        facecolor="#FBF7F1",
        edgecolor=COLORS["orange"],
    )
    rounded_box(
        ax,
        (8.4, 0.75),
        3.05,
        1.0,
        r"$r_{rear}=J_{rear}^*-J_{rear}^{BR}$",
        facecolor="#F7F3FA",
        edgecolor=COLORS["purple"],
    )
    arrow(ax, (4.3, 3.2), (3.45, 3.85), color=COLORS["orange"])
    arrow(ax, (7.7, 3.2), (8.55, 3.85), color=COLORS["purple"])
    arrow(ax, (2.05, 3.85), (2.05, 1.75), color=COLORS["orange"])
    arrow(ax, (9.95, 3.85), (9.95, 1.75), color=COLORS["purple"])
    ax.text(6.0, 1.25, "若所有玩家 regret 都足够小\n则当前联合解可视为近似 Nash 均衡", ha="center", va="center", fontsize=12, color=COLORS["green"], weight="bold")
    save(fig, "10_nash_regret_check.png")


def main():
    figure_01_problem()
    figure_02_ilqr_vs_blackbox()
    figure_03_black_box_loop()
    figure_04_representations()
    figure_05_cma_es()
    figure_06_igo()
    figure_07_svgd()
    figure_08_receding_horizon()
    figure_09_prediction_vs_game()
    figure_10_nash()
    print(f"Generated tutorial figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
