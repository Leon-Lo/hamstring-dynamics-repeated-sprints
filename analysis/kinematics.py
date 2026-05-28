"""
Joint kinematics and kinetics — pre/post SPM waveforms and RM ANOVA.

Covers:
  - Pre/post SPM waveform grids: joint angles (Figure 2) and joint moments (Figure 3)
  - Overall combined SPM figure (angles + moments in one panel)
  - Overall RM ANOVA: Time × Joint for peak angle and peak moment

SPM alpha = 0.05/3 (Bonferroni across 3 joints).
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
from scipy.stats import t as t_dist
import pingouin as pg
import spm1d

from utils import Analysis_utils
from analysis.core.output_helpers import _save_word_table, _extract_latex_caption

analysis_utils = Analysis_utils()

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()

_FINAL_PLOT_DIR = Path("data/plot_results/final_plot_set")
_FINAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_TABLE_DIR = Path("data/tables/kinematics")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)

_ALPHA_PP = 0.05 / 3

# Contact timing for vertical lines
_ct_df = analysis_utils.get_initial_foot_contact_time_df(time_normalized=True)
_ct_pct_pre = _ct_df["initial_contact_time_pre"].mean()
_ct_pct_post = _ct_df["initial_contact_time_post"].mean()

# =============================================================================
# Pre/post SPM helpers
# =============================================================================
_GROUP_ROWS_PP = [
    ("Overall", None),
    ("Healthy", "CON"),
    ("FAIS", "FAIS"),
    ("CAM", "FAIM"),
]

_dem_grp = pd.read_csv("data/subject_demographics.csv")
_grp_map = (
    _dem_grp[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)


def _pp_group_subjects(df_: pd.DataFrame, group_code: str) -> pd.Series:
    subj_str = df_["subject_id"].astype(str).str.lstrip("0").str.strip().map(_grp_map)
    return subj_str == group_code


def _build_pp_matrices(col_name: str, subset) -> tuple:
    pre_rows, post_rows = [], []
    for subj in sorted(pre_df["subject_id"].unique()):
        if subset is not None and not subset.loc[pre_df[pre_df["subject_id"] == subj].index].any():
            continue
        pg_ = pre_df[pre_df["subject_id"] == subj].sort_values("index")
        qg_ = post_df[post_df["subject_id"] == subj].sort_values("index")
        if col_name not in pg_.columns or col_name not in qg_.columns:
            continue
        pv, qv = pg_[col_name].values, qg_[col_name].values
        if len(pv) == 101 and len(qv) == 101 and not (np.isnan(pv).any() or np.isnan(qv).any()):
            pre_rows.append(pv)
            post_rows.append(qv)
    if len(pre_rows) < 2:
        return None, None
    return np.array(pre_rows), np.array(post_rows)


def _draw_prepost_spm_grid(variable_specs: list, ylabel: str, alpha: float, out_name: str, caption_tex: str = "") -> None:
    n_rows, n_cols = len(_GROUP_ROWS_PP), len(variable_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows), sharex=True, constrained_layout=True)
    _IDX = np.arange(101)

    for row_idx, (group_label, group_code) in enumerate(_GROUP_ROWS_PP):
        subset = (
            None if group_code is None
            else (_pp_group_subjects(pre_df, group_code) & _pp_group_subjects(post_df, group_code))
        )
        for col_idx, (col_title, col_name) in enumerate(variable_specs):
            ax = axes[row_idx, col_idx]
            pre_mat, post_mat = _build_pp_matrices(col_name, subset)

            if pre_mat is None:
                ax.text(0.5, 0.5, "n < 2", ha="center", va="center", transform=ax.transAxes, color="gray")
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
                _t_spm = spm1d.stats.ttest_paired(pre_mat, post_mat)
                _ti_spm = _t_spm.inference(alpha, two_tailed=True)
                _sig = np.abs(_ti_spm.z) > _ti_spm.zstar
                ax.fill_between(_IDX, 0, 1, where=_sig, color="red", alpha=0.15,
                                transform=ax.get_xaxis_transform(), zorder=0)
            except Exception as _e_spm:
                print(f"  SPM failed {col_name}/{group_label}: {_e_spm}")

            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{group_label}\n{ylabel}", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Sprint Cycle (%)", fontsize=9)

            ax.axvline(_ct_pct_pre, color="#2196F3", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
            ax.axvline(_ct_pct_post, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
            ax.text(0.97, 0.97, f"n={n}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray")
            ax.set_xlim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    _legend_h = [
        Line2D([0], [0], color="#2196F3", linewidth=2, label="Pre"),
        Line2D([0], [0], color="#FF9800", linewidth=2, label="Post"),
        Patch(facecolor="#2196F3", alpha=0.2, label="95% CI (Pre)"),
        Patch(facecolor="#FF9800", alpha=0.2, label="95% CI (Post)"),
        Patch(facecolor="red", alpha=0.2, label=f"Sig. (SPM, α={alpha:.3f})"),
        Line2D([0], [0], color="#2196F3", linestyle="--", linewidth=1.5, label="Mean Contact Pre"),
        Line2D([0], [0], color="#FF9800", linestyle="--", linewidth=1.5, label="Mean Contact Post"),
    ]
    fig.legend(handles=_legend_h, loc="upper right", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.04))
    _out = str(_FINAL_PLOT_DIR / f"{out_name}.png")
    plt.savefig(_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_out}")
    if caption_tex:
        (_TABLE_DIR / f"{out_name}_caption.tex").write_text(caption_tex)
        (_TABLE_DIR / f"{out_name}_caption.txt").write_text(_extract_latex_caption(caption_tex))
        print(f"Saved {_TABLE_DIR / f'{out_name}_caption.tex'}")


# =============================================================================
# Figure 2: joint angles
# =============================================================================
_draw_prepost_spm_grid(
    [("Hip", "hip_flexion"), ("Knee", "knee_angle"), ("Ankle", "ankle_angle")],
    "Joint Angle (°)",
    _ALPHA_PP,
    "Figure2_prepost_spm_joint_angles",
    caption_tex="\n".join([
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{images/Figure2_prepost_spm_joint_angles.png}",
        r"    \caption{Ensemble averages of hip, knee, and ankle joint flexion angles ($^\circ$)",
        r"    pre (blue) and post (orange) the intervention. Rows represent clinical groups;",
        r"    columns represent joints. Red shading = significant pre-to-post difference",
        r"    (SPM, Bonferroni $\alpha \approx 0.017$, two-tailed).}",
        r"    \label{fig:prepost-spm-joint-angles}",
        r"\end{figure}",
    ]),
)

# =============================================================================
# Figure 3: joint moments
# =============================================================================
_draw_prepost_spm_grid(
    [("Hip", "hip_flexion_moment"), ("Knee", "knee_angle_moment"), ("Ankle", "ankle_angle_moment")],
    "Joint Moment (Nm/kg)",
    _ALPHA_PP,
    "Figure3_prepost_spm_joint_moments",
    caption_tex="\n".join([
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{images/Figure3_prepost_spm_joint_moments.png}",
        r"    \caption{Ensemble averages of hip, knee, and ankle joint flexion moments (Nm/kg)",
        r"    pre (blue) and post (orange) the intervention. Rows represent clinical groups;",
        r"    columns represent joints. Red shading = significant pre-to-post difference",
        r"    (SPM, Bonferroni $\alpha \approx 0.017$, two-tailed).}",
        r"    \label{fig:prepost-spm-joint-moments}",
        r"\end{figure}",
    ]),
)

# =============================================================================
# Overall combined SPM: angles + moments in two rows
# =============================================================================
def _draw_overall_combined_spm(row_specs: list, row_ylabels: list, alpha: float, out_name: str) -> None:
    n_rows = len(row_specs)
    n_cols = len(row_specs[0])
    _IDX = np.arange(101)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (specs, ylabel) in enumerate(zip(row_specs, row_ylabels)):
        for col_idx, (col_title, col_name) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            pre_mat, post_mat = _build_pp_matrices(col_name, None)

            if pre_mat is None:
                ax.text(0.5, 0.5, "n < 2", ha="center", va="center", transform=ax.transAxes, color="gray")
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
                _t_spm = spm1d.stats.ttest_paired(pre_mat, post_mat)
                _ti_spm = _t_spm.inference(alpha, two_tailed=True)
                _sig_spm = np.abs(_ti_spm.z) > _ti_spm.zstar
                ax.fill_between(_IDX, 0, 1, where=_sig_spm, color="red", alpha=0.15,
                                transform=ax.get_xaxis_transform(), zorder=0)
            except Exception as _e_spm:
                print(f"  SPM failed {col_name}: {_e_spm}")

            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Sprint Cycle (%)", fontsize=9)

            ax.axvline(_ct_pct_pre, color="#2196F3", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
            ax.axvline(_ct_pct_post, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
            ax.text(0.97, 0.97, f"n={n}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray")
            ax.set_xlim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    _legend_h = [
        Line2D([0], [0], color="#2196F3", linewidth=2, label="Pre"),
        Line2D([0], [0], color="#FF9800", linewidth=2, label="Post"),
        Patch(facecolor="#2196F3", alpha=0.2, label="95% CI (Pre)"),
        Patch(facecolor="#FF9800", alpha=0.2, label="95% CI (Post)"),
        Patch(facecolor="red", alpha=0.2, label=f"Sig. (SPM, α={alpha:.3f})"),
        Line2D([0], [0], color="#2196F3", linestyle="--", linewidth=1.5, label="Mean Contact Pre"),
        Line2D([0], [0], color="#FF9800", linestyle="--", linewidth=1.5, label="Mean Contact Post"),
    ]
    fig.legend(handles=_legend_h, loc="upper right", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.04))
    _out = str(_FINAL_PLOT_DIR / f"{out_name}.png")
    plt.savefig(_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_out}")


_draw_overall_combined_spm(
    row_specs=[
        [("Hip", "hip_flexion"), ("Knee", "knee_angle"), ("Ankle", "ankle_angle")],
        [("Hip", "hip_flexion_moment"), ("Knee", "knee_angle_moment"), ("Ankle", "ankle_angle_moment")],
    ],
    row_ylabels=["Joint Angle (°)", "Joint Moment (Nm/kg)"],
    alpha=_ALPHA_PP,
    out_name="Figure_overall_angles_moments",
)

# =============================================================================
# Overall RM ANOVA: Time × Joint — peak angle and peak moment
# =============================================================================
_JOINT_LIST = ["hip", "knee", "ankle"]
_JOINT_COLS = {
    "hip": ("hip_flexion", "hip_flexion_moment"),
    "knee": ("knee_angle", "knee_angle_moment"),
    "ankle": ("ankle_angle", "ankle_angle_moment"),
}
_overall_subject_ids = sorted(set(pre_df["subject_id"].unique()) & set(post_df["subject_id"].unique()))


def _peak_per_subject(df_: pd.DataFrame, col: str) -> pd.Series:
    return df_.groupby("subject_id")[col].max().reindex(_overall_subject_ids)


def _overall_rm_anova_block(metric_name: str, col_map: dict, factor_name: str, factor_levels: list) -> dict:
    rows_long = []
    for level in factor_levels:
        col = col_map[level]
        pre_vals = _peak_per_subject(pre_df, col)
        post_vals = _peak_per_subject(post_df, col)
        for subj in _overall_subject_ids:
            rows_long.append({"subject": subj, "time": "pre", factor_name: level, "value": pre_vals.get(subj)})
            rows_long.append({"subject": subj, "time": "post", factor_name: level, "value": post_vals.get(subj)})
    long_df = pd.DataFrame(rows_long)
    complete = long_df.groupby("subject")["value"].apply(lambda x: x.notna().all())
    long_df = long_df[long_df["subject"].isin(complete[complete].index)].dropna()
    aov = pg.rm_anova(data=long_df, dv="value", within=["time", factor_name], subject="subject", correction=True)

    def _pval(row):
        eps = row.get("eps", 1.0)
        gg = eps is not None and float(eps) < 1.0
        p_gg = row.get("p_GG_corr", float("nan"))
        p_unc = row.get("p_unc", float("nan"))
        return (float(p_gg), True) if (gg and not pd.isna(p_gg)) else (float(p_unc), False)

    interaction_src = f"time * {factor_name}"
    interaction_row = aov[aov["Source"] == interaction_src].iloc[0]
    p_interaction, _ = _pval(interaction_row)
    ph_pvals = {}
    for level in factor_levels:
        level_df = long_df[long_df[factor_name] == level]
        ph = pg.pairwise_tests(data=level_df, dv="value", within="time", subject="subject", parametric=True)
        ph_pvals[level] = min(float(ph["p_unc"].iloc[0]) * len(factor_levels), 1.0)
    return {"aov": aov, "interaction_sig": p_interaction < 0.05, "posthoc_pvals": ph_pvals,
            "alpha_ph": 0.05 / len(factor_levels)}


_KIN_METRICS = {
    "Peak Joint Angle (°)": {j: _JOINT_COLS[j][0] for j in _JOINT_LIST},
    "Peak Joint Moment (Nm/kg)": {j: _JOINT_COLS[j][1] for j in _JOINT_LIST},
}

print("\n" + "=" * 65 + "\nOVERALL RM ANOVA — Kinematics/Kinetics (time × joint)\n" + "=" * 65)
_kin_anova_results = {}
for _metric_name, _col_map in _KIN_METRICS.items():
    _res = _overall_rm_anova_block(_metric_name, _col_map, "joint", _JOINT_LIST)
    _kin_anova_results[_metric_name] = _res
    print(f"\n{_metric_name}")
    for _, _row in _res["aov"].iterrows():
        _eps = _row.get("eps", 1.0)
        _gg = _eps is not None and float(_eps) < 1.0
        _p_gg = _row.get("p_GG_corr", float("nan"))
        _p_unc = _row.get("p_unc", float("nan"))
        _p = float(_p_gg) if (_gg and not pd.isna(_p_gg)) else float(_p_unc)
        print(
            f"  {_row['Source']:25s}  F={_row['F']:.2f}  "
            f"df=({int(_row['ddof1'])},{int(_row['ddof2'])})  "
            f"p={_p:.3f}{'*' if _p < 0.05 else ''}"
        )
    if _res["interaction_sig"]:
        print(f"  Post-hoc (Bonferroni α={_res['alpha_ph']:.3f}):")
        for _lv, _pp in _res["posthoc_pvals"].items():
            print(f"    {_lv}: p={_pp:.3f}{'*' if _pp < 0.05 else ''}")

# ── Save Word table ───────────────────────────────────────────────────────────
_EFFECT_LABELS_OV = {
    "time": "Time", "joint": "Joint", "time * joint": "Time × Joint",
}
_kin_word_rows = []
for _mname, _res in _kin_anova_results.items():
    _first = True
    for _, _effect_row in _res["aov"].iterrows():
        _src = _effect_row["Source"]
        _eps = _effect_row.get("eps", 1.0)
        _gg = _eps is not None and float(_eps) < 1.0
        _p_gg = _effect_row.get("p_GG_corr", float("nan"))
        _p_unc = _effect_row.get("p_unc", float("nan"))
        _p_val = float(_p_gg) if (_gg and not pd.isna(_p_gg)) else float(_p_unc)
        _p_str = ("<0.001" if _p_val < 0.001 else f"{_p_val:.3f}") + ("*" if _p_val < 0.05 else "") + ("‡" if _gg else "")
        _eta2 = _effect_row.get("ng2", float("nan"))
        _kin_word_rows.append({
            "Metric": _mname if _first else "",
            "Effect": _EFFECT_LABELS_OV.get(_src, _src),
            "F": f"{_effect_row['F']:.2f}",
            "df₁, df₂": f"{int(_effect_row['ddof1'])}, {int(_effect_row['ddof2'])}",
            "p": _p_str,
            "η²g": f"{_eta2:.3f}" if not pd.isna(_eta2) else "—",
        })
        _first = False

_save_word_table(
    pd.DataFrame(_kin_word_rows),
    str(_TABLE_DIR / "rm_anova_overall_kinematics.docx"),
    title="Two-Way RM ANOVA (Time × Joint): Overall Group — Kinematics and Kinetics",
    note="Peak value over the full sprint cycle per subject. "
    "‡ Greenhouse-Geisser correction. η²g = generalised eta-squared. * p < 0.05. "
    "Post-hoc: Bonferroni-corrected paired t-tests (α = 0.05/3 ≈ 0.017), "
    "reported only where the interaction was significant.",
)
print(f"Saved {_TABLE_DIR / 'rm_anova_overall_kinematics.docx'}")
