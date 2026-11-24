"""Render the six-metric manuscript heatmap from the frozen CSV table."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
table = pd.read_csv(ROOT / "results" / "manuscript_table.csv", index_col=0)
values = table.to_numpy(float)
masked = np.ma.masked_invalid(values)

fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
image = ax.imshow(masked, vmin=0, vmax=1, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
ax.set_yticks(range(len(table.index)), table.index)
ax.set_xlabel("Evaluation dimension (higher is better)")
ax.set_ylabel("Method")
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        label = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.3f}"
        color = "white" if not np.isfinite(values[i, j]) or values[i, j] < 0.55 else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)
bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
bar.set_label("Mean score")
fig.suptitle("Positive synthetic-trajectory benchmark (n = 25 datasets)")
(ROOT / "figures").mkdir(exist_ok=True)
for suffix in ("png", "svg", "pdf"):
    fig.savefig(ROOT / "figures" / f"metric_heatmap.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)

negative = pd.read_csv(ROOT / "results" / "spatial_null_negative_control.csv")
fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
colors = ["#2b8cbe" if method == "MORPH" else "#bdbdbd" for method in negative["method"]]
bars = ax.bar(negative["method"], negative["rejection_rate"], color=colors, edgecolor="black", linewidth=0.6)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Rejection rate (1 - output coverage)")
ax.set_xlabel("Method")
ax.set_title("Spatial-null negative control (n = 5 datasets)")
for bar, value in zip(bars, negative["rejection_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
for suffix in ("png", "svg", "pdf"):
    fig.savefig(ROOT / "figures" / f"spatial_null_rejection.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)

