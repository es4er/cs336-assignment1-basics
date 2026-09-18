"""Render a dependency-free SVG summary of final sweep endpoints."""

from __future__ import annotations

import csv
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROWS = list(csv.DictReader((HERE / "results.csv").open(encoding="utf-8")))
ROWS = [row for row in ROWS if int(row["max_iters"]) == 200]

WIDTH, HEIGHT = 900, 520
LEFT, RIGHT, TOP, BOTTOM = 82, 30, 50, 72
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM
X_MIN, X_MAX = -7.0, -1.0
Y_MIN, Y_MAX = 2.0, 10.0


def x_pos(lr: float) -> float:
    return LEFT + (math.log10(lr) - X_MIN) / (X_MAX - X_MIN) * PLOT_W


def y_pos(loss: float) -> float:
    return TOP + (Y_MAX - loss) / (Y_MAX - Y_MIN) * PLOT_H


def polyline(key: str) -> str:
    points = " ".join(
        f"{x_pos(float(row['max_lr'])):.1f},{y_pos(float(row[key])):.1f}"
        for row in ROWS
    )
    return points


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
    '<rect width="100%" height="100%" fill="#fbfbfd"/>',
    '<style>text{font-family:Inter,Segoe UI,sans-serif;fill:#172033}.axis{stroke:#748094;stroke-width:1}.grid{stroke:#dfe3ea;stroke-width:1}.train{stroke:#2563eb;fill:none;stroke-width:3}.val{stroke:#dc2626;fill:none;stroke-width:3}.dot-t{fill:#2563eb}.dot-v{fill:#dc2626}</style>',
    '<text x="82" y="28" font-size="20" font-weight="700">Learning-rate sweep: final loss at 200 steps</text>',
]

for exponent in range(-7, 0):
    x = x_pos(10**exponent)
    parts += [
        f'<line class="grid" x1="{x}" y1="{TOP}" x2="{x}" y2="{TOP + PLOT_H}"/>',
        f'<text x="{x}" y="{TOP + PLOT_H + 28}" text-anchor="middle" font-size="13">1e{exponent}</text>',
    ]
for loss in range(2, 11):
    y = y_pos(loss)
    parts += [
        f'<line class="grid" x1="{LEFT}" y1="{y}" x2="{LEFT + PLOT_W}" y2="{y}"/>',
        f'<text x="{LEFT - 14}" y="{y + 5}" text-anchor="end" font-size="13">{loss}</text>',
    ]

parts += [
    f'<line class="axis" x1="{LEFT}" y1="{TOP + PLOT_H}" x2="{LEFT + PLOT_W}" y2="{TOP + PLOT_H}"/>',
    f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + PLOT_H}"/>',
    f'<polyline class="train" points="{polyline("final_train_loss")}"/>',
    f'<polyline class="val" points="{polyline("final_val_loss")}"/>',
]
for row in ROWS:
    x = x_pos(float(row["max_lr"]))
    parts.append(f'<circle class="dot-t" cx="{x}" cy="{y_pos(float(row["final_train_loss"]))}" r="4"/>')
    parts.append(f'<circle class="dot-v" cx="{x}" cy="{y_pos(float(row["final_val_loss"]))}" r="4"/>')
parts += [
    f'<text x="{LEFT + PLOT_W / 2}" y="{HEIGHT - 18}" text-anchor="middle" font-size="14">Peak learning rate (log scale)</text>',
    f'<text x="20" y="{TOP + PLOT_H / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {TOP + PLOT_H / 2})">Loss</text>',
    '<line class="train" x1="650" y1="25" x2="690" y2="25"/><text x="700" y="30" font-size="13">train</text>',
    '<line class="val" x1="770" y1="25" x2="810" y2="25"/><text x="820" y="30" font-size="13">validation</text>',
    '</svg>',
]

(HERE / "lr_sweep_endpoints.svg").write_text("\n".join(parts), encoding="utf-8")

