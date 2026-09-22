"""从 results.csv 渲染「最终验证损失 vs 峰值学习率」的 U 形曲线。

重新生成：
    uv run python experiments/lr_sweep/plot_sweep.py

样式刻意对齐 W&B 导出的收敛曲线（白底、仅横向网格、无上/右边框），
这样它能和三张 loss-vs-step 图并排放在同一份报告里而不显得突兀。
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
OUT = HERE.parent / "wandb" / "lr_sweep_u_shape.png"

# W&B 风格的图表元素配色
GRID = "#e8e8e8"
AXIS = "#cfcfcf"
TEXT = "#3b3b3b"
MUTED = "#8a8a8a"
COARSE_C = "#4a90d9"
FINE_C = "#e07b39"
CTRL_C = "#c0392b"

rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))


def lr_text(v):
    """写成 2e-3 而不是 0.002，与报告正文的记号保持一致。"""
    exp = math.floor(math.log10(v))
    return f"{v / 10 ** exp:g}e{exp}"


def points(stage):
    return [
        (float(r["peak_lr"]), float(r["final_val_loss"]))
        for r in rows
        if r["stage"] == stage
    ]


coarse, fine = points("coarse"), points("fine")
control = points("schedule_control")[0]

# 除 schedule 对照外的 11 个 run 共享同一套 cosine 调度，因此可以连成一条曲线
curve = sorted(coarse + fine)
xs, ys = zip(*curve)
best_lr, best_loss = min(curve, key=lambda p: p[1])

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 曲线本身
ax.plot(xs, ys, color=COARSE_C, linewidth=1.7, zorder=2, solid_capstyle="round")
ax.scatter(
    *zip(*coarse), s=52, color=COARSE_C, zorder=3,
    label="coarse screen (7 runs)",
)
ax.scatter(
    *zip(*fine), s=56, color=FINE_C, marker="s", zorder=3,
    label="fine screen (4 runs)",
)

# 最优点的水平参考线
ax.axhline(best_loss, color=FINE_C, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)

# schedule 对照：同为 3e-3 峰值，但不属于这条曲线，单独标出
ax.scatter(
    [control[0]], [control[1]], s=95, color=CTRL_C, marker="X", zorder=4,
    label="constant-3e-3 control",
)

# 放在最小值的左上方，引导线全程走在曲线之上，不会横穿曲线
ax.annotate(
    f"best tested peak LR  {lr_text(best_lr)}\nvalidation loss {best_loss:.4f}",
    xy=(best_lr, best_loss),
    xytext=(best_lr * 0.40, best_loss + 0.66),
    ha="center", va="bottom", color=TEXT, fontsize=9.5, linespacing=1.6,
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.0,
                    shrinkA=3, shrinkB=4),
)
ax.annotate(
    "constant schedule  2.5767",
    xy=control, xytext=(control[0] * 1.30, control[1] + 0.03),
    ha="left", va="center", color=CTRL_C, fontsize=9,
    arrowprops=dict(arrowstyle="-", color=CTRL_C, linewidth=0.9,
                    shrinkA=2, shrinkB=5),
)
ax.annotate(
    "severe under-training",
    xy=(1e-5, 6.7139), xytext=(1.15e-5, 6.05),
    ha="left", va="top", color=MUTED, fontsize=9,
)
ax.annotate(
    "degradation",
    xy=(1e-1, 4.0082), xytext=(1e-1, 4.42),
    ha="right", va="bottom", color=MUTED, fontsize=9,
)

# 坐标轴与网格
ax.set_xscale("log")
ax.set_xlim(5.5e-6, 2.4e-1)
ax.set_ylim(2.05, 7.15)
ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ax.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"])
ax.minorticks_off()

ax.set_xlabel("peak learning rate", color=TEXT, fontsize=10.5, labelpad=8)
ax.set_ylabel("final validation loss at step 500", color=TEXT, fontsize=10.5,
              labelpad=8)

ax.yaxis.grid(True, color=GRID, linewidth=0.9)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

ax.set_title("Final validation loss is U-shaped in log learning rate",
             color="#1a1a1a", fontsize=13, fontweight="bold", pad=34)

ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False,
    fontsize=9, labelcolor=TEXT, handletextpad=0.5, columnspacing=1.8,
)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT.relative_to(HERE.parent.parent)}  "
      f"({OUT.stat().st_size / 1024:.0f} KB)")
