"""
Eccentric loading analysis — pre/post hamstring mechanics during late-swing sprinting.

Covers:
  - Paired pre/post plots per metric per muscle (Figure 3)
  - Δ change score bar charts (Figure 3b)
  - Eccentric window quality-control plot (Figure 4)
  - Peak force, eccentric impulse, and eccentric work % contribution (pre vs post)
  - ANCOVA on % contribution change scores (Δspeed as covariate)
  - ANCOVA on absolute change scores (Δspeed as covariate)
  - Mixed ANOVA: Time × Group for eccentric metrics
  - Eccentric redistribution stacked-bar figures (Figure 6, Figure 6a)
  - Summary sentences for combined tables

All metrics are computed within the second negative knee joint-power phase
(eccentric window = late-swing hamstring lengthening).
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pingouin as pg
import statsmodels.formula.api as smf
from scipy.stats import shapiro, t as t_dist, ttest_rel, wilcoxon

from utils import Analysis_utils
from analysis.core.biomechanics import eccentric_mask, METRICS, compute_metric_per_subject
from analysis.core.output_helpers import _save_word_table, _extract_latex_caption

analysis_utils = Analysis_utils()

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()
subject_ids = sorted(pre_df["subject_id"].unique())
muscles = ["bflh", "semiten", "semimem"]
MUSCLE_MAP = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}

_PLOT_DIR = Path("data/plot_results/eccentric_loading")
_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_TABLE_DIR = Path("data/tables/eccentric_loading")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)
_FINAL_PLOT_DIR = Path("data/plot_results/final_plot_set")
_FINAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Re-run RM ANOVA to get significance flags for plot annotations
# =============================================================================
def _row_pval(effect_row: pd.Series) -> tuple:
    eps = effect_row.get("eps", 1.0)
    gg_corrected = eps is not None and float(eps) < 1.0
    p_gg = effect_row.get("p_GG_corr", float("nan"))
    p_unc = effect_row.get("p_unc", float("nan"))
    if gg_corrected and not pd.isna(p_gg):
        return float(p_gg), True
    return float(p_unc), False


interaction_sig_by_metric = {}
posthoc_pvals = {}

for metric_name, metric_fn in METRICS.items():
    rows_long = []
    for muscle in muscles:
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        for subj in subject_ids:
            rows_long.append({"subject": subj, "time": "pre", "muscle": muscle, "value": pre_vals.get(subj)})
            rows_long.append({"subject": subj, "time": "post", "muscle": muscle, "value": post_vals.get(subj)})
    long_df = pd.DataFrame(rows_long)
    complete = long_df.groupby("subject")["value"].apply(lambda x: x.notna().all())
    long_df = long_df[long_df["subject"].isin(complete[complete].index)].dropna()
    aov = pg.rm_anova(data=long_df, dv="value", within=["time", "muscle"], subject="subject", correction=True)
    interaction_row = aov[aov["Source"] == "time * muscle"].iloc[0]
    p_interaction, _ = _row_pval(interaction_row)
    interaction_sig_by_metric[metric_name] = p_interaction < 0.05
    for muscle in muscles:
        muscle_df = long_df[long_df["muscle"] == muscle]
        ph = pg.pairwise_tests(data=muscle_df, dv="value", within="time", subject="subject", parametric=True)
        posthoc_pvals[(metric_name, muscle)] = min(float(ph["p_unc"].iloc[0]) * len(muscles), 1.0)

# =============================================================================
# Paired pre/post plots — Figure 3
# =============================================================================
MUSCLE_LABELS = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
METRIC_NAMES = list(METRICS.keys())

fig, axes = plt.subplots(len(METRICS), len(muscles), figsize=(10, 3 * len(METRICS)))

for j, (metric_name, metric_fn) in enumerate(METRICS.items()):
    for i, muscle in enumerate(muscles):
        ax = axes[j, i]
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        common = pre_vals.dropna().index.intersection(post_vals.dropna().index)
        pre_plot = pre_vals.loc[common]
        post_plot = post_vals.loc[common]

        for subj in common:
            ax.plot([0, 1], [pre_plot[subj], post_plot[subj]], color="gray", alpha=0.35, linewidth=0.8)

        ax.scatter([0] * len(common), pre_plot.values, color="blue", zorder=3, s=20)
        ax.scatter([1] * len(common), post_plot.values, color="orange", zorder=3, s=20)

        n = len(common)
        t_crit = t_dist.ppf(0.975, df=n - 1)
        pre_center, post_center = float(pre_plot.mean()), float(post_plot.mean())
        pre_err = float(t_crit * pre_plot.std() / np.sqrt(n))
        post_err = float(t_crit * post_plot.std() / np.sqrt(n))

        ax.errorbar(
            [0, 1], [pre_center, post_center], yerr=[pre_err, post_err],
            fmt="D", color="black", capsize=4, zorder=4, markersize=7, mfc="white", mec="black",
        )

        p_bonf = posthoc_pvals.get((metric_name, muscle), 1.0)
        interaction_sig = interaction_sig_by_metric.get(metric_name, False)
        if interaction_sig and p_bonf < 0.05:
            ax.text(1, post_center + post_err, "*", ha="center", va="bottom", fontsize=13, color="black")

        if "EMG" in metric_name:
            ax.set_ylim(bottom=0)
        if i == 0:
            ax.set_ylabel(metric_name)
        if j == len(METRICS) - 1:
            ax.set_xlabel(MUSCLE_LABELS[muscle], fontsize=10, fontweight="bold")
        else:
            ax.set_xlabel("")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_xlim(-0.5, 1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.97])
_tmp_fig, _tmp_ax = plt.subplots()
_eb_handle = _tmp_ax.errorbar(
    [0], [0], yerr=[1], fmt="D", color="black", capsize=4, markersize=7,
    mfc="white", mec="black", label="Mean ± 95% CI",
)
plt.close(_tmp_fig)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=7, label="Pre"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=7, label="Post"),
    _eb_handle,
]
fig.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.0))
plt.savefig(str(_PLOT_DIR / "Figure3.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

_fig_caption = "\n".join([
    r"\begin{figure}",
    r"    \centering",
    r"    \includegraphics[width=0.8\linewidth]{images/Figure3.png}",
    r"    \caption{Pre- and post-repeated sprints paired comparisons for BFLH, SM, and ST across",
    r"    five biomechanical variables (rows), all computed within the eccentric window",
    r"    (knee joint power $<$ 0): eccentric muscle impulse,",
    r"    eccentric knee moment contribution, mean and peak adjusted EMG, and peak normalized",
    r"    fiber length ($l/l_0$). Blue/orange dots are individual subject values",
    r"    pre-/post-repeated sprints; grey lines connect pairs. Open diamonds show group mean",
    r"    $\pm$ 95\% CI. $^*p < 0.05$ after Bonferroni correction for three muscles.}",
    r"    \label{fig:biomechanics-pre-post}",
    r"\end{figure}",
])
(_TABLE_DIR / "biomechanics_fig_caption.tex").write_text(_fig_caption, encoding="utf-8")
(_TABLE_DIR / "biomechanics_fig_caption.txt").write_text(_extract_latex_caption(_fig_caption), encoding="utf-8")
print(f"Saved {_PLOT_DIR / 'Figure3.png'}")

# =============================================================================
# Δ change score plots — Figure 3b
# =============================================================================
_rng = np.random.default_rng(42)
_metric_list = list(METRICS.items())
_n_metrics = len(_metric_list)
_ncols_cs, _bar_colors = 3, {"bflh": "#e15759", "semiten": "#f28e2b", "semimem": "#59a14f"}
_nrows_cs = (_n_metrics + _ncols_cs - 1) // _ncols_cs
_x_positions = np.arange(len(muscles))
_bar_width, _jitter_spread = 0.55, 0.12
_muscle_display = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}

fig_cs, axes_cs = plt.subplots(_nrows_cs, _ncols_cs, figsize=(12, 5 * _nrows_cs))
axes_cs_flat = axes_cs.flatten()

for _j, (metric_name, metric_fn) in enumerate(_metric_list):
    ax = axes_cs_flat[_j]
    _interaction_sig = interaction_sig_by_metric.get(metric_name, False)

    for _i, muscle in enumerate(muscles):
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        common = pre_vals.dropna().index.intersection(post_vals.dropna().index)
        delta = (post_vals.loc[common] - pre_vals.loc[common]).dropna()
        mean_d, sd_d, n_d = float(delta.mean()), float(delta.std()), len(delta)

        color = _bar_colors[muscle]
        ax.bar(_x_positions[_i], mean_d, width=_bar_width, color=color, alpha=0.75, zorder=2)
        ax.errorbar(_x_positions[_i], mean_d, yerr=sd_d, fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3)
        jitter = _rng.uniform(-_jitter_spread, _jitter_spread, size=n_d)
        ax.scatter(_x_positions[_i] + jitter, delta.values, color=color, edgecolors="white",
                   linewidths=0.4, s=30, alpha=0.85, zorder=4)

        _p_bonf = posthoc_pvals.get((metric_name, muscle), 1.0)
        if _interaction_sig and _p_bonf < 0.05:
            _top = mean_d + sd_d
            _star_y = _top + abs(_top) * 0.08 + abs(float(delta.abs().max())) * 0.05
            ax.text(_x_positions[_i], _star_y, "*", ha="center", va="bottom", fontsize=14, color="black")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(_x_positions)
    ax.set_xticklabels([_muscle_display[m] for m in muscles], fontsize=10)
    ax.set_ylabel(f"Δ {metric_name}", fontsize=9)
    ax.set_xlim(-0.6, len(muscles) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for _k in range(_n_metrics, len(axes_cs_flat)):
    axes_cs_flat[_k].set_visible(False)

fig_cs.suptitle("Change scores (post − pre) per muscle and metric", fontsize=12, y=1.01)
fig_cs.tight_layout()
plt.savefig(str(_PLOT_DIR / "Figure3b_change_scores.png"), dpi=300, bbox_inches="tight")
plt.close(fig_cs)
print(f"Saved {_PLOT_DIR / 'Figure3b_change_scores.png'}")

# =============================================================================
# Eccentric window QC — Figure 4
# =============================================================================
MUSCLE_COLORS = {"bflh": "#e15759", "semiten": "#f28e2b", "semimem": "#59a14f"}
MUSCLE_ABBREV = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
n_cols = 9
n_rows = (len(subject_ids) + n_cols - 1) // n_cols
fig_chk, axes_chk = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), constrained_layout=True)
axes_chk_flat = axes_chk.flatten()

for ax, subj in zip(axes_chk_flat, subject_ids):
    grp = pre_df[pre_df["subject_id"] == subj]
    time = np.asarray(grp["time"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), time)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)
    mask = eccentric_mask(grp)
    idx = grp["index"].values
    ax.plot(idx, power, color="steelblue", lw=1.2)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.fill_between(idx, power, 0, where=mask, color="steelblue", alpha=0.25)
    ax2 = ax.twinx()
    for muscle, color in MUSCLE_COLORS.items():
        force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
        ax2.plot(idx, force, color=color, lw=1.0, alpha=0.8)
    ax2.tick_params(labelsize=5, colors="grey")
    ax2.spines["top"].set_visible(False)
    ax.set_title(f"{subj}", fontsize=8)
    ax.set_xlabel("Sprint cycle (%)", fontsize=6)
    ax.set_ylabel("Knee joint power (W)", fontsize=6, color="steelblue")
    ax2.set_ylabel("Muscle fiber force (N)", fontsize=6, color="grey")
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes_chk_flat[len(subject_ids):]:
    ax.set_visible(False)

legend_handles = [
    Line2D([0], [0], color="steelblue", lw=1.5, label="Knee power"),
    Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Eccentric window"),
] + [Line2D([0], [0], color=c, lw=1.5, label=MUSCLE_ABBREV[m]) for m, c in MUSCLE_COLORS.items()]
fig_chk.tight_layout(rect=[0, 0, 1, 0.97])
fig_chk.legend(handles=legend_handles, loc="upper right", ncol=len(legend_handles),
               fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.0))
plt.savefig(str(_PLOT_DIR / "Figure4.png"), dpi=150, bbox_inches="tight")
print(f"Saved {_PLOT_DIR / 'Figure4.png'}")

# =============================================================================
# Per-subject eccentric window plots
# =============================================================================
_PER_SUBJ_DIR = _PLOT_DIR / "per_subject"
_PER_SUBJ_DIR.mkdir(parents=True, exist_ok=True)

_per_subj_legend = [
    Line2D([0], [0], color="steelblue", lw=1.5, label="Knee power"),
    Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Eccentric window"),
] + [Line2D([0], [0], color=c, lw=1.5, label=MUSCLE_ABBREV[m]) for m, c in MUSCLE_COLORS.items()]

for subj in subject_ids:
    grp = pre_df[pre_df["subject_id"] == subj]
    time = np.asarray(grp["time"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), time)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)
    mask = eccentric_mask(grp)
    idx = grp["index"].values

    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    ax_s.plot(idx, power, color="steelblue", lw=1.5)
    ax_s.axhline(0, color="black", lw=0.8, ls="--")
    ax_s.fill_between(idx, power, 0, where=mask, color="steelblue", alpha=0.25)
    ax2_s = ax_s.twinx()
    for muscle, color in MUSCLE_COLORS.items():
        force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
        ax2_s.plot(idx, force, color=color, lw=1.2, alpha=0.85, label=MUSCLE_ABBREV[muscle])
    ax2_s.tick_params(labelsize=9, colors="grey")
    ax2_s.spines["top"].set_visible(False)
    ax_s.set_xlabel("Sprint cycle (%)", fontsize=10)
    ax_s.set_ylabel("Knee joint power (W)", fontsize=10, color="steelblue")
    ax2_s.set_ylabel("Muscle fiber force (N)", fontsize=10, color="grey")
    ax_s.tick_params(labelsize=9)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    fig_s.tight_layout(rect=[0, 0, 1, 0.90])
    fig_s.legend(handles=_per_subj_legend, loc="upper right", ncol=len(_per_subj_legend),
                 fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.0), borderpad=1.0, labelspacing=0.6)
    _out_path = _PER_SUBJ_DIR / f"eccentric_window_subject_{subj}.png"
    plt.savefig(str(_out_path), dpi=150, bbox_inches="tight")
    plt.close(fig_s)

print(f"Saved {len(subject_ids)} per-subject eccentric window plots to {_PER_SUBJ_DIR}")

# =============================================================================
# Eccentric % contribution — peak force, impulse, work
# =============================================================================
def _peak_force_ecc(grp: pd.DataFrame, muscle: str) -> float:
    mask = eccentric_mask(grp)
    vals = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    return float(np.nanmax(vals[mask])) if mask.any() else float(np.nanmax(vals))


def _compute_pct_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject % of total (BFLH+ST+SM) peak force in the eccentric window."""
    records = {}
    for subj, grp in df.groupby("subject_id"):
        peaks = {m: _peak_force_ecc(grp, m) for m in muscles}
        total = sum(peaks.values())
        records[subj] = {m: 100.0 * peaks[m] / total if total > 0 else np.nan for m in muscles}
    return pd.DataFrame(records).T


_pct_pre = _compute_pct_contribution(pre_df)
_pct_post = _compute_pct_contribution(post_df)

_force_fn_ecc = _peak_force_ecc
_force_pre_raw = {m: compute_metric_per_subject(pre_df, m, _force_fn_ecc).reindex(subject_ids) for m in muscles}
_force_post_raw = {m: compute_metric_per_subject(post_df, m, _force_fn_ecc).reindex(subject_ids) for m in muscles}

_impulse_fn = METRICS["Eccentric Impulse (N·s)"]
_imp_pre_raw = {m: compute_metric_per_subject(pre_df, m, _impulse_fn).reindex(subject_ids) for m in muscles}
_imp_post_raw = {m: compute_metric_per_subject(post_df, m, _impulse_fn).reindex(subject_ids) for m in muscles}
_common_imp = _imp_pre_raw[muscles[0]].dropna().index.intersection(_imp_post_raw[muscles[0]].dropna().index)

_moment_fn = METRICS["Eccentric Knee Moment (N·m·s)"]
_mom_pre_raw = {m: compute_metric_per_subject(pre_df, m, _moment_fn).reindex(subject_ids) for m in muscles}
_mom_post_raw = {m: compute_metric_per_subject(post_df, m, _moment_fn).reindex(subject_ids) for m in muscles}
_common_mom = _mom_pre_raw[muscles[0]].dropna().index.intersection(_mom_post_raw[muscles[0]].dropna().index)

_work_fn = METRICS["Eccentric Work (J)"]
_work_pre_raw = {m: compute_metric_per_subject(pre_df, m, _work_fn).reindex(subject_ids) for m in muscles}
_work_post_raw = {m: compute_metric_per_subject(post_df, m, _work_fn).reindex(subject_ids) for m in muscles}
_common_work = _work_pre_raw[muscles[0]].dropna().index.intersection(_work_post_raw[muscles[0]].dropna().index)


def _to_pct_df(raw: dict, common_idx) -> pd.DataFrame:
    records = {}
    for subj in common_idx:
        peaks = {m: float(raw[m].loc[subj]) for m in muscles}
        total = sum(peaks.values())
        records[subj] = {m: 100.0 * peaks[m] / total if total != 0 else np.nan for m in muscles}
    return pd.DataFrame(records).T


_imp_pct_pre = _to_pct_df(_imp_pre_raw, _common_imp)
_imp_pct_post = _to_pct_df(_imp_post_raw, _common_imp)
_mom_pct_pre = _to_pct_df(_mom_pre_raw, _common_mom)
_mom_pct_post = _to_pct_df(_mom_post_raw, _common_mom)
_work_pct_pre = _to_pct_df(_work_pre_raw, _common_work)
_work_pct_post = _to_pct_df(_work_post_raw, _common_work)

# ── Paired % contribution plots ───────────────────────────────────────────────
_muscle_labels_ecc = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
_ecc_legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=7, label="Pre"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=7, label="Post"),
]


def _draw_ecc_paired_panel(ax, pre_vals: pd.Series, post_vals: pd.Series, ylabel: str, muscle_title: str) -> None:
    common = pre_vals.dropna().index.intersection(post_vals.dropna().index)
    pv = pre_vals.loc[common]
    qv = post_vals.loc[common]
    for subj in common:
        ax.plot([0, 1], [pv[subj], qv[subj]], color="gray", alpha=0.35, linewidth=0.8)
    ax.scatter([0] * len(common), pv.values, color="blue", zorder=3, s=20)
    ax.scatter([1] * len(common), qv.values, color="orange", zorder=3, s=20)
    n = len(common)
    t_crit = t_dist.ppf(0.975, df=n - 1)
    ax.errorbar(
        [0, 1],
        [float(pv.mean()), float(qv.mean())],
        yerr=[float(t_crit * pv.std() / np.sqrt(n)), float(t_crit * qv.std() / np.sqrt(n))],
        fmt="D", color="black", capsize=4, zorder=4, markersize=7, mfc="white", mec="black",
    )
    ax.set_title(muscle_title, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_xlim(-0.5, 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)


def _save_ecc_figure(pre_data, post_data, ylabel: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True, constrained_layout=True)
    for i, muscle in enumerate(muscles):
        _draw_ecc_paired_panel(axes[i], pre_data[muscle], post_data[muscle],
                               ylabel if i == 0 else "", _muscle_labels_ecc[muscle])
    fig.legend(handles=_ecc_legend_elements, loc="upper right", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.06))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# =============================================================================
# ANCOVA on % contribution change scores
# =============================================================================
_ancova_raw = pd.read_csv("data/spatiotemporal_stats.csv")[["subject_id", "pre_speed_m_s", "post_speed_m_s"]].copy()
_ancova_speed_df = _ancova_raw.rename(columns={"subject_id": "subject"})
_ancova_speed_df["delta_speed"] = _ancova_speed_df["post_speed_m_s"] - _ancova_speed_df["pre_speed_m_s"]
_ancova_speed_mean = float(_ancova_speed_df["delta_speed"].mean())
_ancova_speed_df["speed_c"] = _ancova_speed_df["delta_speed"] - _ancova_speed_mean
_ancova_alpha = 0.05 / len(muscles)


def _fmt_tp(t, p, alpha):
    if pd.isna(t):
        return "—", "—"
    p_str = ("<0.001" if p < 0.001 else f"{p:.3f}") + ("*" if p < alpha else "")
    return f"{t:.2f}", p_str


_pct_ancova_metrics = [
    ("Peak Eccentric Force", _pct_pre, _pct_post),
    ("Eccentric Impulse", _imp_pct_pre, _imp_pct_post),
    ("Eccentric Work", _work_pct_pre, _work_pct_post),
]

_pct_ancova_rows = []
for _metric_label, _pre_pct_df, _post_pct_df in _pct_ancova_metrics:
    for _muscle in muscles:
        _common_pct = _pre_pct_df.index.intersection(_post_pct_df.index)
        _cs_pct = []
        for _subj in _common_pct:
            _pv = _pre_pct_df.loc[_subj, _muscle]
            _ov = _post_pct_df.loc[_subj, _muscle]
            if pd.notna(_pv) and pd.notna(_ov):
                _cs_pct.append({"subject": _subj, "delta": float(_ov) - float(_pv)})
        if len(_cs_pct) < 4:
            continue
        _cs_pct_df = pd.DataFrame(_cs_pct).merge(_ancova_speed_df[["subject", "speed_c"]], on="subject")
        _fit_pct = smf.ols("delta ~ speed_c", data=_cs_pct_df).fit()
        _ti, _pi = _fmt_tp(float(_fit_pct.tvalues["Intercept"]), float(_fit_pct.pvalues["Intercept"]), _ancova_alpha)
        _ts, _ps = _fmt_tp(float(_fit_pct.tvalues["speed_c"]), float(_fit_pct.pvalues["speed_c"]), 0.05)
        _pct_ancova_rows.append({
            "Metric": _metric_label,
            "Muscle": _muscle_labels_ecc[_muscle],
            "Intercept t": _ti, "Intercept p": _pi,
            "Δspeed t": _ts, "Δspeed p": _ps,
        })

_pct_ancova_df = pd.DataFrame(_pct_ancova_rows)
print(_pct_ancova_df.to_string(index=False))
_save_word_table(
    _pct_ancova_df,
    str(_TABLE_DIR / "ancova_pct_contribution_speed.docx"),
    title="ANCOVA on % Contribution Change Scores: Controlling for Δspeed",
    note=f"Each muscle's % share of total hamstring eccentric load (BFLH + ST + SM = 100%). "
    f"Δ_pct = post % − pre %. OLS regression on Δspeed (mean-centered; "
    f"group mean = {_ancova_speed_mean:.2f} m/s). "
    f"* Intercept p < {_ancova_alpha:.3f} (Bonferroni for 3 muscles); * Δspeed p < 0.05.",
)
print(f"Saved {_TABLE_DIR / 'ancova_pct_contribution_speed.docx'}")

# ── ANCOVA on absolute change scores ─────────────────────────────────────────
_abs_ancova_metrics = [
    ("Peak Eccentric Force (N)", _force_pre_raw, _force_post_raw),
    ("Eccentric Impulse (N·s)", _imp_pre_raw, _imp_post_raw),
    ("Eccentric Work (J)", _work_pre_raw, _work_post_raw),
]

_abs_ancova_rows = []
for _metric_name, _pre_raw, _post_raw in _abs_ancova_metrics:
    for _muscle in muscles:
        _pre_v = _pre_raw[_muscle].dropna()
        _post_v = _post_raw[_muscle].dropna()
        _idx = _pre_v.index.intersection(_post_v.index)
        _cs_abs = [{"subject": s, "delta": float(_post_v[s]) - float(_pre_v[s])} for s in _idx]
        if len(_cs_abs) < 4:
            continue
        _cs_abs_df = pd.DataFrame(_cs_abs).merge(_ancova_speed_df[["subject", "speed_c"]], on="subject")
        try:
            _fit_abs = smf.ols("delta ~ speed_c", data=_cs_abs_df).fit()
            _ti, _pi = _fmt_tp(float(_fit_abs.tvalues["Intercept"]), float(_fit_abs.pvalues["Intercept"]), _ancova_alpha)
            _ts, _ps = _fmt_tp(float(_fit_abs.tvalues["speed_c"]), float(_fit_abs.pvalues["speed_c"]), 0.05)
        except Exception as _e:
            print(f"  ANCOVA failed {_metric_name}/{_muscle}: {_e}")
            _ti = _pi = _ts = _ps = "—"
        _abs_ancova_rows.append({
            "Metric": _metric_name, "Muscle": _muscle_labels_ecc[_muscle],
            "Intercept t": _ti, "Intercept p": _pi, "Δspeed t": _ts, "Δspeed p": _ps,
        })

_save_word_table(
    pd.DataFrame(_abs_ancova_rows),
    str(_TABLE_DIR / "ancova_absolute_speed.docx"),
    title="ANCOVA on Absolute Change Scores: Controlling for Δspeed",
    note=f"OLS regression of Δ_metric (post − pre) on Δspeed, mean-centered "
    f"(group mean = {_ancova_speed_mean:.2f} m/s). "
    f"* Intercept p < {_ancova_alpha:.3f} (Bonferroni for 3 muscles); * Δspeed p < 0.05.",
)
print(f"Saved {_TABLE_DIR / 'ancova_absolute_speed.docx'}")

# =============================================================================
# Mixed ANOVA: Time × Group (CON / FAIS / FAIM) per eccentric metric per muscle
# =============================================================================
_dem_grp_mix = pd.read_csv("data/subject_demographics.csv")
_grp_map_mix = (
    _dem_grp_mix[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)

_MIXED_METRICS = [
    ("Peak eccentric force (N)", _force_pre_raw, _force_post_raw),
    ("Eccentric impulse (N·s)", _imp_pre_raw, _imp_post_raw),
    ("Eccentric work (J)", _work_pre_raw, _work_post_raw),
]
_ALPHA_MIXED = 0.05 / len(muscles)
_MUSCLE_LABELS_MIX = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
_MIXED_GROUP_LABELS = {"CON": "Healthy", "FAIS": "FAIS", "FAIM": "CAM"}


def _get_group(subj_id) -> str | None:
    s = str(subj_id).lstrip("0").strip()
    return _grp_map_mix.get(s)


def _build_mixed_long(pre_raw: dict, post_raw: dict, muscle: str) -> pd.DataFrame:
    rows = []
    for subj in subject_ids:
        grp = _get_group(subj)
        if grp not in ("CON", "FAIS", "FAIM"):
            continue
        pv = pre_raw[muscle].get(subj)
        ov = post_raw[muscle].get(subj)
        if pd.notna(pv) and pd.notna(ov):
            rows += [
                {"subject": subj, "time": "pre", "group": grp, "value": float(pv)},
                {"subject": subj, "time": "post", "group": grp, "value": float(ov)},
            ]
    return pd.DataFrame(rows)


def _fmt_effect_label(src: str) -> str:
    return src.replace("time", "Time").replace("group", "Group").replace("*", "×").replace("  ", " ").strip()


_mixed_aov_results = {}
_posthoc_mix_rows = []

print("\n" + "=" * 70)
print("MIXED ANOVA: Time × Group — Eccentric Metrics")
print("=" * 70)

for _mn, _ppre, _ppost in _MIXED_METRICS:
    for _mus in muscles:
        _ldf = _build_mixed_long(_ppre, _ppost, _mus)
        if _ldf.empty or _ldf["group"].nunique() < 2:
            continue
        complete = _ldf.groupby("subject")["value"].apply(lambda x: x.notna().all())
        _ldf = _ldf[_ldf["subject"].isin(complete[complete].index)].dropna()
        try:
            _aov = pg.mixed_anova(
                data=_ldf, dv="value", within="time", between="group", subject="subject"
            )
            _mixed_aov_results[(_mn, _mus)] = _aov
        except Exception as _e:
            print(f"  Mixed ANOVA failed {_mn}/{_mus}: {_e}")
            continue
        print(f"\n  {_mn} | {_MUSCLE_LABELS_MIX[_mus]}")
        for _, _row in _aov.iterrows():
            _src = str(_row["Source"])
            _p_val = _row.get("p-unc", _row.get("p_unc", np.nan))
            _F_val = _row.get("F", np.nan)
            print(f"    {_src:30s}  F={_F_val:.2f}  p={_p_val:.3f}{'*' if not pd.isna(_p_val) and _p_val < _ALPHA_MIXED else ''}")

# ── Post-hoc where interaction significant ────────────────────────────────────
for _mn, _mus in list(_mixed_aov_results.keys()):
    _aov = _mixed_aov_results[(_mn, _mus)]
    _int_row = _aov[_aov["Source"].str.contains("Interaction", case=False, na=False)]
    if _int_row.empty:
        continue
    _p_int = _int_row.iloc[0].get("p-unc", _int_row.iloc[0].get("p_unc", 1.0))
    if pd.isna(_p_int) or _p_int >= _ALPHA_MIXED:
        continue

    _, _ppre_ph, _ppost_ph = next(s for s in _MIXED_METRICS if s[0] == _mn)
    _ldf_ph = _build_mixed_long(_ppre_ph, _ppost_ph, _mus)
    for _g in ("CON", "FAIS", "FAIM"):
        _sub = _ldf_ph[_ldf_ph["group"] == _g]
        _pv = _sub[_sub["time"] == "pre"].set_index("subject")["value"]
        _qv = _sub[_sub["time"] == "post"].set_index("subject")["value"]
        _idx = _pv.index.intersection(_qv.index)
        if len(_idx) < 3:
            continue
        _d = _qv.loc[_idx] - _pv.loc[_idx]
        _, _swp_ph = shapiro(_d)
        if _swp_ph > 0.05:
            _stat_ph, _p_raw_ph = ttest_rel(_pv.loc[_idx], _qv.loc[_idx])
            _test_ph = "t"
        else:
            _stat_ph, _p_raw_ph = wilcoxon(_pv.loc[_idx], _qv.loc[_idx])
            _test_ph = "W"
        _p_bonf_ph = min(_p_raw_ph * 9, 1.0)
        _sig_ph = " *" if _p_bonf_ph < 0.05 else ""
        _gl = _MIXED_GROUP_LABELS[_g]
        print(f"  {_mn} | {_MUSCLE_LABELS_MIX[_mus]} | {_gl:<8}  {_test_ph}={_stat_ph:.3f}, p_bonf={_p_bonf_ph:.3f}{_sig_ph}")
        _posthoc_mix_rows.append({
            "Metric": _mn, "Muscle": _MUSCLE_LABELS_MIX[_mus], "Group": _gl, "n": len(_idx),
            "Pre (mean±SD)": f"{float(_pv.loc[_idx].mean()):.1f}±{float(_pv.loc[_idx].std()):.1f}",
            "Post (mean±SD)": f"{float(_qv.loc[_idx].mean()):.1f}±{float(_qv.loc[_idx].std()):.1f}",
            "Test": _test_ph, "Statistic": round(float(_stat_ph), 3),
            "p (Bonf.)": round(_p_bonf_ph, 3), "Sig.": _sig_ph.strip(),
        })

# ── Save mixed ANOVA Word table ───────────────────────────────────────────────
_mix_word_rows = []
for (_mn, _mus), _aov in _mixed_aov_results.items():
    for _, _row in _aov.iterrows():
        _p_val_w = _row.get("p-unc", _row.get("p_unc", np.nan))
        _mix_word_rows.append({
            "Metric": _mn, "Muscle": _MUSCLE_LABELS_MIX[_mus],
            "Effect": _fmt_effect_label(str(_row["Source"])),
            "F": round(float(_row.get("F", np.nan)), 2) if not pd.isna(_row.get("F", np.nan)) else "---",
            "df": f"{int(_row.get('DF1', 0))}, {int(_row.get('DF2', 0))}"
                  if not (pd.isna(_row.get("DF1", np.nan)) or pd.isna(_row.get("DF2", np.nan))) else "---",
            "p": ("<0.001 *" if _p_val_w < 0.001 else (f"{_p_val_w:.3f} *" if _p_val_w < _ALPHA_MIXED else f"{_p_val_w:.3f}"))
                 if not pd.isna(_p_val_w) else "---",
            "η²p": round(float(_row.get("np2", np.nan)), 3) if not pd.isna(_row.get("np2", np.nan)) else "---",
        })
_save_word_table(
    pd.DataFrame(_mix_word_rows),
    str(_TABLE_DIR / "mixed_anova_redistribution.docx"),
    title="2×3 Mixed ANOVA: Time (pre/post) × Group (Healthy/FAIS/CAM) — Eccentric Metrics per Muscle",
    note=f"Bonferroni-corrected α = 0.05/3 muscles = {_ALPHA_MIXED:.4f}. "
    "* p < Bonferroni-corrected α.",
)
print(f"Saved {_TABLE_DIR / 'mixed_anova_redistribution.docx'}")

# =============================================================================
# Eccentric redistribution stacked-bar figures — Figure 6, Figure 6a
# =============================================================================
_MUSCLE_ORDER_PCT = ["bflh", "semiten", "semimem"]
_MUSCLE_COLORS_PCT = {"bflh": "#4C72B0", "semiten": "#DD8452", "semimem": "#55A868"}
_MUSCLE_NAMES_PCT = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}


def _filter_pct_df(pct_df: pd.DataFrame, group_code: str | None) -> pd.DataFrame:
    if group_code is None:
        return pct_df
    grp_col = pct_df.index.astype(str).str.lstrip("0").str.strip().map(_grp_map_mix)
    return pct_df[grp_col == group_code]


def _draw_ecc_stacked_panel(ax, pre_g: pd.DataFrame, post_g: pd.DataFrame, col_title: str) -> None:
    common = pre_g.index.intersection(post_g.index)
    pre_m = pre_g.loc[common][_MUSCLE_ORDER_PCT].mean()
    post_m = post_g.loc[common][_MUSCLE_ORDER_PCT].mean()
    x, width = np.array([0, 1]), 0.5
    for i, m in enumerate(_MUSCLE_ORDER_PCT):
        bottom_pre = sum(pre_m[_MUSCLE_ORDER_PCT[j]] for j in range(i))
        bottom_post = sum(post_m[_MUSCLE_ORDER_PCT[j]] for j in range(i))
        ax.bar(x[0], pre_m[m], width, bottom=bottom_pre,
               color=_MUSCLE_COLORS_PCT[m], label=_MUSCLE_NAMES_PCT[m] if i == 0 else "_")
        ax.bar(x[1], post_m[m], width, bottom=bottom_post, color=_MUSCLE_COLORS_PCT[m], alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_title(col_title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 105)


# Figure 6: overall
_ecc_specs_overall = [
    ("Peak eccentric force", _pct_pre, _pct_post),
    ("Eccentric impulse", _imp_pct_pre, _imp_pct_post),
    ("Eccentric work", _work_pct_pre, _work_pct_post),
]

fig6, axes6 = plt.subplots(1, 3, figsize=(9, 4), constrained_layout=True)
_n_overall = len(_pct_pre)
for _ci, (_title, _pre_g, _post_g) in enumerate(_ecc_specs_overall):
    _draw_ecc_stacked_panel(axes6[_ci], _filter_pct_df(_pre_g, None), _filter_pct_df(_post_g, None), _title)
    if _ci == 0:
        axes6[_ci].set_ylabel(f"Overall (n={_n_overall})\n% of total", fontsize=9)
    else:
        axes6[_ci].tick_params(axis="y", left=False)

_leg_handles = [Patch(facecolor=_MUSCLE_COLORS_PCT[m], label=_MUSCLE_NAMES_PCT[m]) for m in _MUSCLE_ORDER_PCT]
_leg_handles += [Patch(facecolor="gray", alpha=0.3, label="Post (lighter)")]
fig6.legend(handles=_leg_handles, loc="upper right", ncol=len(_leg_handles), fontsize=8, frameon=False)

_fig6_out = str(_FINAL_PLOT_DIR / "Figure6_eccentric_redistribution_by_group.png")
plt.savefig(_fig6_out, dpi=300, bbox_inches="tight")
plt.close(fig6)
print(f"Saved {_fig6_out}")

# Figure 6a: clinical groups
_GROUP_ROWS_6a = [("Healthy", "CON"), ("FAIS", "FAIS"), ("CAM", "FAIM")]
fig6a, axes6a = plt.subplots(3, 3, figsize=(9, 10), constrained_layout=True)

for _ri, (_group_label, _group_code) in enumerate(_GROUP_ROWS_6a):
    for _ci, (_title, _pre_g, _post_g) in enumerate(_ecc_specs_overall):
        _ax = axes6a[_ri, _ci]
        _pre_grp = _filter_pct_df(_pre_g, _group_code)
        _post_grp = _filter_pct_df(_post_g, _group_code)
        _n_grp = len(_pre_grp.index.intersection(_post_grp.index))
        _draw_ecc_stacked_panel(_ax, _pre_grp, _post_grp, _title if _ri == 0 else "")
        if _ci == 0:
            _ax.set_ylabel(f"{_group_label} (n={_n_grp})\n% of total", fontsize=9)

fig6a.legend(handles=_leg_handles, loc="upper right", ncol=len(_leg_handles), fontsize=8, frameon=False)
_fig6a_out = str(_FINAL_PLOT_DIR / "Figure6a_eccentric_redistribution_by_group.png")
plt.savefig(_fig6a_out, dpi=300, bbox_inches="tight")
plt.close(fig6a)
print(f"Saved {_fig6a_out}")
