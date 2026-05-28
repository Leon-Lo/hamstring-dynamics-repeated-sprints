"""
EMG Validation — Measured vs CEINMS AdjustedEmgs

Two complementary views of EMG-CEINMS agreement:
  1. R² bar plots — one bar per trial, per subject
  2. Waveform overlay — all trials overlaid per subject, R² and RMSE annotated

Produces four output figures (BFLH and SEMIMEM × both plot types).
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SIM_BASE = Path("data/simulations")
OUT_DIR = Path("data/plot_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLORS = {"CON": "black", "FAIM": "darkorange", "FAIS": "red"}
TRIAL_COLORS = {
    "Run_baselineA1": "#1f77b4",
    "Run_baselineB1": "#ff7f0e",
    "RunL1": "#2ca02c",
    "RunM1": "#d62728",
    "RunN1": "#9467bd",
    "RunO1": "#8c564b",
    "RunP1": "#e377c2",
}
TRIAL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


# =============================================================================
# HELPERS
# =============================================================================

def _parse_file(path: Path) -> pd.DataFrame:
    """Parse whitespace-separated .mot / .sto with an endheader block."""
    lines = path.read_text().splitlines()
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "endheader")
    cols = lines[end + 1].split()
    rows = [list(map(float, ln.split())) for ln in lines[end + 2:] if ln.split()]
    return pd.DataFrame(rows, columns=cols)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r² — standard for EMG waveform comparison."""
    if len(y_true) < 2:
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r ** 2) if not np.isnan(r) else float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =============================================================================
# DATA LOADING
# =============================================================================

pub = pd.read_csv("data/subject_demographics_public.csv")
pub["subj_id"] = pub["Subject"].astype(int).apply(lambda x: f"{x:03d}")
pub["leg"] = pub["Measured Leg"].str.strip().str.lower()

demo_raw = pd.read_csv("data/subject_demographics.csv", usecols=["Subject", "Group"])
demo_raw["subj_num"] = pd.to_numeric(
    demo_raw["Subject"].astype(str).str.split().str[0], errors="coerce"
)
demo_raw = demo_raw.dropna(subset=["subj_num"])
demo_raw["subj_id"] = demo_raw["subj_num"].astype(int).apply(lambda x: f"{x:03d}")
demo = demo_raw[["subj_id", "Group"]].drop_duplicates("subj_id")

info = pub.merge(demo, on="subj_id", how="left").sort_values("subj_id")

store = {}  # subj_id -> {"group": ..., "trials": [...]}

for _, row in info.iterrows():
    subj = row["subj_id"]
    leg = row["leg"]
    group = row.get("Group", None)

    subj_dir = SIM_BASE / subj / "pre"
    if not subj_dir.exists():
        continue

    bflh_col = f"bflh_{leg}"
    semi_col = f"semimem_{leg}"

    trials = []
    for trial_dir in sorted(subj_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        emg_path = trial_dir / "emg.mot"
        adj_path = trial_dir / "Optimised" / "AdjustedEmgs.sto"
        if not (emg_path.exists() and adj_path.exists()):
            continue

        try:
            emg_df = _parse_file(emg_path)
            adj_df = _parse_file(adj_path)
        except Exception:
            continue

        missing = [c for c in ["BFLH", "SEMIMEM"] if c not in emg_df.columns]
        missing += [c for c in [bflh_col, semi_col] if c not in adj_df.columns]
        if missing:
            continue

        t_emg = emg_df["time"].values
        t_adj = adj_df["time"].values
        t_lo = max(t_emg[0], t_adj[0])
        t_hi = min(t_emg[-1], t_adj[-1])
        if t_hi - t_lo < 0.05:
            continue

        mask = (t_adj >= t_lo) & (t_adj <= t_hi)
        t_common = t_adj[mask]
        pct = (t_common - t_common[0]) / (t_common[-1] - t_common[0]) * 100

        emg_bflh = np.interp(t_common, t_emg, emg_df["BFLH"].values)
        emg_semi = np.interp(t_common, t_emg, emg_df["SEMIMEM"].values)
        adj_bflh = adj_df.loc[mask, bflh_col].values
        adj_semi = adj_df.loc[mask, semi_col].values

        trials.append({
            "name": trial_dir.name,
            "pct": pct,
            "emg_bflh": emg_bflh,
            "adj_bflh": adj_bflh,
            "r2_bflh": _r2(emg_bflh, adj_bflh),
            "rmse_bflh": _rmse(emg_bflh, adj_bflh),
            "emg_semi": emg_semi,
            "adj_semi": adj_semi,
            "r2_semi": _r2(emg_semi, adj_semi),
            "rmse_semi": _rmse(emg_semi, adj_semi),
        })

    if trials:
        store[subj] = {"group": group, "trials": trials}

valid_subjects = [s for s in info["subj_id"] if s in store]
n_subj = len(valid_subjects)
NCOLS = 6
NROWS = (n_subj + NCOLS - 1) // NCOLS

plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# PLOT 1: R² Bar Plots
# =============================================================================

MUSCLES_BAR = [("BFLH", "r2_bflh"), ("SEMIMEM", "r2_semi")]

for muscle, r2_key in MUSCLES_BAR:
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 2.8, NROWS * 2.6),
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = axes.flatten()

    for idx, subj in enumerate(valid_subjects):
        ax = axes_flat[idx]
        entry = store[subj]
        group = entry["group"]

        trials = entry["trials"]
        names = [t["name"] for t in trials]
        r2_values = [t[r2_key] for t in trials]
        colors = [TRIAL_COLORS.get(n, "#7f7f7f") for n in names]

        x = np.arange(len(trials))
        bars = ax.bar(x, r2_values, color=colors, edgecolor="white",
                      linewidth=0.6, width=0.65, zorder=3)

        for bar, val in zip(bars, r2_values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(val + 0.03, 0.97),
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=4.5, color="dimgray",
                )

        ax.set_xlim(-0.5, len(trials) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace("Run_baseline", "BL-").replace("Run", "") for n in names],
            fontsize=5, rotation=30, ha="right",
        )
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylabel("R²", fontsize=6)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"S{subj}", fontsize=8,
                     color=GROUP_COLORS.get(group, "dimgray"),
                     fontweight="bold", pad=3)

    for idx in range(n_subj, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    trial_handles = [mpatches.Patch(color=c, label=n) for n, c in TRIAL_COLORS.items()]
    group_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10,
                        hspace=0.60, wspace=0.45)
    fig.legend(handles=trial_handles + group_handles,
               loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), frameon=True,
               title="Trial name  |  Group", title_fontsize=8)
    fig.suptitle(f"{muscle}  —  R²  (EMG measured vs CEINMS AdjustedEmgs)",
                 fontsize=13, fontweight="bold")

    out = OUT_DIR / f"r2_barplot_{muscle.lower()}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# =============================================================================
# PLOT 2: Waveform Overlay with R² and RMSE
# =============================================================================

MUSCLES_WAVE = [
    ("BFLH",    "emg_bflh", "adj_bflh", "r2_bflh",  "rmse_bflh"),
    ("SEMIMEM", "emg_semi", "adj_semi", "r2_semi",  "rmse_semi"),
]

for muscle, ek, ak, r2k, rmsek in MUSCLES_WAVE:
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 3.8, NROWS * 2.8),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, subj in enumerate(valid_subjects):
        ax = axes_flat[idx]
        entry = store[subj]
        group = entry["group"]

        for ti, trial in enumerate(entry["trials"]):
            c = TRIAL_PALETTE[ti % len(TRIAL_PALETTE)]
            lbl = (f"{trial['name']}\n"
                   f"R²={trial[r2k]:.2f}  RMSE={trial[rmsek]:.3f}")
            ax.plot(trial["pct"], trial[ek], color=c, lw=1.1, alpha=0.85)
            ax.plot(trial["pct"], trial[ak], color=c, lw=1.1, ls="--",
                    alpha=0.70, label=lbl)

        ax.set_title(f"S{subj}", fontsize=8,
                     color=GROUP_COLORS.get(group, "dimgray"),
                     fontweight="bold", pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("Gait cycle (%)", fontsize=5.5)
        ax.set_ylabel("EMG (norm.)", fontsize=5.5)
        leg = ax.legend(fontsize=4.0, loc="upper right",
                        framealpha=0.6, handlelength=1.2,
                        labelspacing=0.2, borderpad=0.4)
        for txt in leg.get_texts():
            txt.set_color("dimgray")

    for idx in range(n_subj, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    style_handles = [
        Line2D([0], [0], color="gray", lw=1.4, label="EMG measured (emg.mot)"),
        Line2D([0], [0], color="gray", lw=1.4, ls="--", label="Adjusted EMG (CEINMS)"),
    ]
    group_handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=style_handles + group_handles,
               loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True)
    fig.suptitle(f"{muscle}  —  EMG measured vs CEINMS AdjustedEmgs",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = OUT_DIR / f"emg_comparison_{muscle.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)
