# -*- coding: utf-8 -*-
"""Standalone script: generate combined Figure 2 + 3 + 4.

Rows 0–1: Hip/Knee/Ankle joint angles and moments (Overall group, SPM)
Rows 2–3: BFLH/ST/SM muscle forces and normalised fibre lengths (Overall group, SPM)
Row  4  : Eccentric load-sharing stacked bars (peak force, impulse, work)

Output: data/plot_results/final_plot_set/Figure_combined_2_3_4.png
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as t_dist
import spm1d as _spm1d

from utils import Analysis_utils
from analysis.core.biomechanics import eccentric_mask as _eccentric_mask, METRICS as _ALL_METRICS, compute_metric_per_subject

analysis_utils = Analysis_utils()

# ── Data ──────────────────────────────────────────────────────────────────────
pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()
subject_ids = sorted(pre_df["subject_id"].unique())

_ct_df = analysis_utils.get_initial_foot_contact_time_df(time_normalized=True)
_ct_pct_pre = _ct_df["initial_contact_time_pre"].mean()
_ct_pct_post = _ct_df["initial_contact_time_post"].mean()

_dem = pd.read_csv("data/subject_demographics.csv")
_grp_map = (
    _dem[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)

_METRICS = {
    "Eccentric Impulse (N·s)": _ALL_METRICS["Eccentric Impulse (N·s)"],
    "Eccentric Work (J)": _ALL_METRICS["Eccentric Work (J)"],
}


# ── Eccentric % contribution ──────────────────────────────────────────────────
_muscles_ecc = ["bflh", "semiten", "semimem"]
_MUSCLE_ORDER_PCT = _muscles_ecc
_MUSCLE_COLORS_PCT = {"bflh": "#4C72B0", "semiten": "#DD8452", "semimem": "#55A868"}
_MUSCLE_NAMES_PCT = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}


def _peak_force_ecc(grp: pd.DataFrame, muscle: str) -> float:
    mask = _eccentric_mask(grp)
    vals = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    return float(np.nanmax(vals[mask])) if mask.any() else float(np.nanmax(vals))


def _compute_pct_contribution(df: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for subj, grp in df.groupby("subject_id"):
        peaks = {m: _peak_force_ecc(grp, m) for m in _muscles_ecc}
        total = sum(peaks.values())
        records[subj] = {m: 100.0 * peaks[m] / total if total > 0 else np.nan for m in _muscles_ecc}
    return pd.DataFrame(records).T


def _to_pct_df(raw: dict, common_idx) -> pd.DataFrame:
    records = {}
    for subj in common_idx:
        peaks = {m: float(raw[m].loc[subj]) for m in _muscles_ecc}
        total = sum(peaks.values())
        records[subj] = {m: 100.0 * peaks[m] / total if total != 0 else np.nan for m in _muscles_ecc}
    return pd.DataFrame(records).T


print("Computing eccentric % contributions …")
_pct_pre = _compute_pct_contribution(pre_df)
_pct_post = _compute_pct_contribution(post_df)

_imp_fn = _METRICS["Eccentric Impulse (N·s)"]
_imp_pre_raw = {m: compute_metric_per_subject(pre_df, m, _imp_fn).reindex(subject_ids) for m in _muscles_ecc}
_imp_post_raw = {m: compute_metric_per_subject(post_df, m, _imp_fn).reindex(subject_ids) for m in _muscles_ecc}
_common_imp = (
    _imp_pre_raw[_muscles_ecc[0]].dropna().index
    .intersection(_imp_post_raw[_muscles_ecc[0]].dropna().index)
)
_imp_pct_pre = _to_pct_df(_imp_pre_raw, _common_imp)
_imp_pct_post = _to_pct_df(_imp_post_raw, _common_imp)

_work_fn = _METRICS["Eccentric Work (J)"]
_work_pre_raw = {m: compute_metric_per_subject(pre_df, m, _work_fn).reindex(subject_ids) for m in _muscles_ecc}
_work_post_raw = {m: compute_metric_per_subject(post_df, m, _work_fn).reindex(subject_ids) for m in _muscles_ecc}
_common_work = (
    _work_pre_raw[_muscles_ecc[0]].dropna().index
    .intersection(_work_post_raw[_muscles_ecc[0]].dropna().index)
)
_work_pct_pre = _to_pct_df(_work_pre_raw, _common_work)
_work_pct_post = _to_pct_df(_work_post_raw, _common_work)


# ── SPM pre/post matrix builder ───────────────────────────────────────────────
def _build_pp_matrices(col: str, subset=None):
    common = sorted(set(pre_df["subject_id"].unique()) & set(post_df["subject_id"].unique()))
    if subset is not None:
        common = [s for s in common if s in subset]
    pre_rows, post_rows = [], []
    for s in common:
        pg = pre_df[pre_df["subject_id"] == s].sort_values("index")
        qg = post_df[post_df["subject_id"] == s].sort_values("index")
        if col not in pg.columns or col not in qg.columns:
            continue
        if len(pg) == 101 and len(qg) == 101:
            pre_rows.append(pg[col].values)
            post_rows.append(qg[col].values)
    if len(pre_rows) < 2:
        return None, None
    return np.array(pre_rows), np.array(post_rows)


# ── Stacked-bar panel ─────────────────────────────────────────────────────────
def _filter_pct_df(pct_df: pd.DataFrame, group_code=None) -> pd.DataFrame:
    if group_code is None:
        return pct_df
    subj_str = pct_df.index.astype(str).str.lstrip("0").str.strip()
    mask = subj_str.map(_grp_map) == group_code
    return pct_df.loc[mask]


def _draw_ecc_stacked_panel(ax, pre_pdf: pd.DataFrame, post_pdf: pd.DataFrame, title: str) -> None:
    _bar_means: dict = {}
    for _xi, (_lbl, _pdf) in enumerate([("Pre", pre_pdf), ("Post", post_pdf)]):
        _common = _pdf[_MUSCLE_ORDER_PCT].dropna().index
        _bot = 0.0
        _bar_means[_xi] = {}
        for _m in _MUSCLE_ORDER_PCT:
            _mean = float(_pdf.loc[_common, _m].mean()) if len(_common) else 0.0
            _bar_means[_xi][_m] = _mean
            ax.bar(_xi, _mean, bottom=_bot, color=_MUSCLE_COLORS_PCT[_m],
                   width=0.5, edgecolor="white", linewidth=1.5)
            if _mean > 8:
                ax.text(_xi, _bot + _mean / 2, f"{_MUSCLE_NAMES_PCT[_m]}\n{_mean:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            elif _mean > 3:
                ax.text(_xi, _bot + _mean / 2,
                        f"{_MUSCLE_NAMES_PCT[_m]} {_mean:.1f}%",
                        ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            _bot += _mean
    _b0, _b1 = 0.0, 0.0
    for _m in _MUSCLE_ORDER_PCT[:-1]:
        _b0 += _bar_means[0][_m]
        _b1 += _bar_means[1][_m]
        ax.plot([0.25, 0.75], [_b0, _b1], color="black", linewidth=0.9,
                linestyle="--", alpha=0.5, zorder=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


# ── Combined figure ───────────────────────────────────────────────────────────
_ALPHA_PP = 0.05 / 3
_FINAL_PLOT_DIR = Path("data/plot_results/final_plot_set")

_waveform_rows = [
    ("Joint Angle (°)", [
        ("Hip", "hip_flexion"),
        ("Knee", "knee_angle"),
        ("Ankle", "ankle_angle"),
    ]),
    ("Joint Moment (Nm/kg)", [
        ("Hip", "hip_flexion_moment"),
        ("Knee", "knee_angle_moment"),
        ("Ankle", "ankle_angle_moment"),
    ]),
    ("Muscle Force (N/kg BW)", [
        ("M. Biceps Femoris\nLong Head", "bflh_force_per_bw"),
        ("M. Semitendinosus", "semiten_force_per_bw"),
        ("M. Semimembranosus", "semimem_force_per_bw"),
    ]),
    (r"Norm. Fiber Length ($l/l_0$)", [
        ("M. Biceps Femoris\nLong Head", "bflh_fiber_lengths"),
        ("M. Semitendinosus", "semiten_fiber_lengths"),
        ("M. Semimembranosus", "semimem_fiber_lengths"),
    ]),
]

print("Building combined figure …")

fig, axes = plt.subplots(
    5, 3,
    figsize=(15, 21),
    gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.6]},
    constrained_layout=True,
)

_IDX = np.arange(101)

# Waveform rows 0–3
for row_idx, (ylabel, specs) in enumerate(_waveform_rows):
    for col_idx, (col_title, col_name) in enumerate(specs):
        ax = axes[row_idx, col_idx]
        pre_mat, post_mat = _build_pp_matrices(col_name, None)

        if pre_mat is None:
            ax.text(0.5, 0.5, "n < 2", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.spines[["top", "right"]].set_visible(False)
            continue

        n = pre_mat.shape[0]
        t_crit = t_dist.ppf(0.975, df=n - 1)

        for mat, color in [(pre_mat, "#2196F3"), (post_mat, "#FF9800")]:
            mn = mat.mean(axis=0)
            ci = t_crit * mat.std(axis=0, ddof=1) / np.sqrt(n)
            ax.plot(_IDX, mn, color=color, linewidth=1.8)
            ax.fill_between(_IDX, mn - ci, mn + ci, color=color, alpha=0.2)

        try:
            _t_spm = _spm1d.stats.ttest_paired(pre_mat, post_mat)
            _ti_spm = _t_spm.inference(_ALPHA_PP, two_tailed=True)
            _sig_spm = np.abs(_ti_spm.z) > _ti_spm.zstar
            ax.fill_between(
                _IDX, 0, 1, where=_sig_spm, color="red", alpha=0.15,
                transform=ax.get_xaxis_transform(), zorder=0,
            )
        except Exception as _e:
            print(f"  SPM skipped {col_name}: {_e}")

        if row_idx in (0, 2):
            ax.set_title(col_title, fontweight="bold", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row_idx == 3:
            ax.set_xlabel("Sprint Cycle (%)", fontsize=9)

        ax.axvline(_ct_pct_pre, color="#2196F3", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.axvline(_ct_pct_post, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        ax.text(0.97, 0.97, f"n={n}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray")
        ax.set_xlim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

# Stacked-bar row 4
_ecc_specs = [
    ("Peak eccentric force", _filter_pct_df(_pct_pre, None), _filter_pct_df(_pct_post, None)),
    ("Eccentric impulse", _filter_pct_df(_imp_pct_pre, None), _filter_pct_df(_imp_pct_post, None)),
    ("Eccentric work", _filter_pct_df(_work_pct_pre, None), _filter_pct_df(_work_pct_post, None)),
]
_n_overall = len(_filter_pct_df(_pct_pre, None))

for col_idx, (col_title, pre_g, post_g) in enumerate(_ecc_specs):
    ax = axes[4, col_idx]
    _draw_ecc_stacked_panel(ax, pre_g, post_g, col_title)
    if col_idx == 0:
        ax.set_ylabel(f"Overall (n={_n_overall})\n% of total", fontsize=9)
    else:
        ax.tick_params(axis="y", left=False)

# Shared legend (waveform panels only)
_legend_h = [
    Line2D([0], [0], color="#2196F3", linewidth=2, label="Pre"),
    Line2D([0], [0], color="#FF9800", linewidth=2, label="Post"),
    Patch(facecolor="#2196F3", alpha=0.2, label="95% CI (Pre)"),
    Patch(facecolor="#FF9800", alpha=0.2, label="95% CI (Post)"),
    Patch(facecolor="red", alpha=0.2, label=f"Sig. (SPM, α={_ALPHA_PP:.3f})"),
    Line2D([0], [0], color="#2196F3", linestyle="--", linewidth=1.5, label="Mean Contact Pre"),
    Line2D([0], [0], color="#FF9800", linestyle="--", linewidth=1.5, label="Mean Contact Post"),
]
fig.legend(
    handles=_legend_h,
    loc="upper right",
    ncol=7,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(1.0, 1.02),
)

_out = str(_FINAL_PLOT_DIR / "Figure_combined_2_3_4.png")
plt.savefig(_out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {_out}")
