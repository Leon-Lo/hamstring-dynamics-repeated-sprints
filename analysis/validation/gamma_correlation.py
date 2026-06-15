"""
Correlation between optimised gamma and mean EMG (BFLH, SEMIMEM).
"""

# %% Setup
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats

# %% Load data
df = pd.read_csv("data/alpha_beta_gamma.csv")
df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
df["BFLH_mean"] = pd.to_numeric(df["BFLH_mean"], errors="coerce")
df["SEMIMEM_mean"] = pd.to_numeric(df["SEMIMEM_mean"], errors="coerce")
df = df.dropna(subset=["gamma", "BFLH_mean", "SEMIMEM_mean"])

# Classify run type
def _classify(run: str) -> str:
    if "baselineA" in run:
        return "Baseline A"
    if "baselineB" in run:
        return "Baseline B"
    return "Loaded"

df["run_type"] = df["run"].apply(_classify)

RUN_TYPES = ["Baseline A", "Baseline B", "Loaded"]
COLORS = {"Baseline A": "#2196F3", "Baseline B": "#FF9800", "Loaded": "#E91E63"}
MARKERS = {"Baseline A": "o", "Baseline B": "s", "Loaded": "^"}

# %% Plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

targets = [
    ("BFLH_mean",    "BFLH mean EMG",    axes[0]),
    ("SEMIMEM_mean", "SEMIMEM mean EMG", axes[1]),
]

for col, ylabel, ax in targets:
    # Scatter by run type
    for rt in RUN_TYPES:
        sub = df[df["run_type"] == rt]
        ax.scatter(
            sub["gamma"], sub[col],
            color=COLORS[rt], marker=MARKERS[rt],
            alpha=0.75, edgecolors="white", linewidths=0.4,
            s=60, label=rt, zorder=3,
        )

    # Overall regression line
    x, y = df["gamma"].values, df[col].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="#333333", linewidth=1.8,
            linestyle="--", zorder=4)

    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.set_title(f"γ vs {ylabel}\nr = {r:.3f},  {p_str}", fontsize=12)
    ax.set_xlabel("Gamma (γ)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

# Shared legend
legend_handles = [
    mlines.Line2D([], [], color=COLORS[rt], marker=MARKERS[rt],
                  linestyle="None", markersize=7, label=rt)
    for rt in RUN_TYPES
]
legend_handles.append(
    mlines.Line2D([], [], color="#333333", linewidth=1.8,
                  linestyle="--", label="Overall regression")
)
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.08))

fig.suptitle("Gamma vs Mean EMG — BFLH & SEMIMEM", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = Path("data/plot_results/validation/gamma_emg_correlation.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
