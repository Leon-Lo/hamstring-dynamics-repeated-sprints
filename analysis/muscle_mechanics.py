"""
Muscle mechanics — fiber length trajectories and SPM pre/post waveforms.

Covers:
  - Normalized fiber length curves pre vs post (Figure 5, standalone no-SPM version)
  - Interactive peak fiber length plot (Plotly)
  - Pre/post SPM waveform grids: muscle forces (Figure 4) and fiber lengths (Figure 5)
  - Overall RM ANOVA: Time × Muscle for peak force/BW and peak fiber length

SPM alpha = 0.05/3 (Bonferroni across 3 muscles).
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
from scipy import stats as _stats
from scipy.stats import t as t_dist
import pingouin as pg
import spm1d

from utils import Analysis_utils
from analysis.core.output_helpers import _save_word_table, _extract_latex_caption

analysis_utils = Analysis_utils()

df = pd.read_csv("data/master_results.csv")
pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()
subject_ids = sorted(pre_df["subject_id"].unique())

_PLOT_DIR = Path("data/plot_results/muscle_mechanics")
_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_TABLE_DIR = Path("data/tables/muscle_mechanics")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)
_FINAL_PLOT_DIR = Path("data/plot_results/final_plot_set")
_FINAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

_ALPHA_SPM = 0.05 / 3

# Contact timing (used for vertical lines on waveform plots)
_ct_df = analysis_utils.get_initial_foot_contact_time_df(time_normalized=True)
_ct_pct_pre = _ct_df["initial_contact_time_pre"].mean()
_ct_pct_post = _ct_df["initial_contact_time_post"].mean()

# =============================================================================
# Normalized fiber length curves — pre vs post (no SPM, Figure 5 standalone)
# =============================================================================
_fl_pre_trials = ["Run_baselineA1", "Run_baselineB1"]
_fl_post_trial_order = ["RunP1", "RunO1", "RunN1", "RunM1", "RunL1"]
_fl_cols = ["bflh_fiber_lengths", "semiten_fiber_lengths", "semimem_fiber_lengths"]

_fl_pre = df[df["trial"].isin(_fl_pre_trials)].copy()
_fl_pre["trial"] = "pre"
_fl_pre_agg = _fl_pre.groupby(["subject_id", "index"])[_fl_cols].mean().reset_index()
_fl_pre_agg["trial"] = "pre"

_fl_post_rows = []
for _subj, _subj_df in df.groupby("subject_id"):
    for _t in _fl_post_trial_order:
        if _t in _subj_df["trial"].values:
            _fl_post_rows.append(_subj_df[_subj_df["trial"] == _t])
            break
_fl_post = pd.concat(_fl_post_rows, ignore_index=True).copy() if _fl_post_rows else pd.DataFrame(columns=df.columns)
_fl_post["trial"] = "post"

_fl_combined = pd.concat([_fl_pre_agg, _fl_post], ignore_index=True)
_fl_mean = _fl_combined.groupby(["trial", "index"])[_fl_cols].mean().reset_index()
_fl_std = _fl_combined.groupby(["trial", "index"])[_fl_cols].std().reset_index()
_fl_n = _fl_combined.groupby(["trial", "index"])[_fl_cols].count().reset_index()

_fl_ci = _fl_std.copy()
for _c in _fl_cols:
    _se = _fl_std[_c] / np.sqrt(_fl_n[_c])
    _t = _fl_n[_c].apply(lambda n: _stats.t.ppf(0.975, n - 1) if n > 1 else _stats.norm.ppf(0.975))
    _fl_ci[_c] = _se * _t

_fl_muscle_labels = {
    "bflh_fiber_lengths": "M. Biceps Femoris Long Head",
    "semiten_fiber_lengths": "M. Semitendinosus",
    "semimem_fiber_lengths": "M. Semimembranosus",
}

_fl_spm = analysis_utils.get_spm_sig_index(p=0.05 / 3)

fig_fl, axes_fl = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for _i, _col in enumerate(_fl_cols):
    _ax = axes_fl[_i]
    for _trial, _color in [("pre", "blue"), ("post", "orange")]:
        _m = _fl_mean[_fl_mean["trial"] == _trial]
        _ci = _fl_ci[_fl_ci["trial"] == _trial]
        _ax.plot(_m["index"], _m[_col], color=_color, linewidth=1.8)
        _ax.fill_between(_m["index"], _m[_col] - _ci[_col], _m[_col] + _ci[_col], color=_color, alpha=0.2)
    _ax.set_title(_fl_muscle_labels[_col])
    _ax.set_xlabel("Sprint Cycle (%)")
    _ax.set_xlim(0, 100)
    _ax.set_ylim(0.5, 1.2)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

    _sig = _fl_spm.get(_col)
    if _sig is not None and len(_sig) == 101:
        _ax.fill_between(range(101), 0, 1, where=_sig, color="red", alpha=0.15,
                         transform=_ax.get_xaxis_transform(), zorder=0)

    _ax.axvline(_ct_pct_pre, color="blue", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
    _ax.axvline(_ct_pct_post, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)

axes_fl[0].set_ylabel(r"normalized Fiber Length ($l/l_0$)")
_fl_legend = [
    Line2D([0], [0], color="blue", linewidth=2, label="Pre"),
    Line2D([0], [0], color="orange", linewidth=2, label="Post"),
    Patch(facecolor="blue", alpha=0.2, label="95% CI (Pre)"),
    Patch(facecolor="orange", alpha=0.2, label="95% CI (Post)"),
    Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5, label="Mean Contact Pre"),
    Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="Mean Contact Post"),
    Patch(facecolor="red", alpha=0.15, label="Significant (SPM)"),
]
fig_fl.legend(handles=_fl_legend, loc="upper right", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.06))
fig_fl.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(str(_PLOT_DIR / "Figure5.png"), dpi=300, bbox_inches="tight")
plt.close(fig_fl)
print(f"Saved {_PLOT_DIR / 'Figure5.png'}")

# Caption files
_fl_caption = "\n".join([
    r"\begin{figure}",
    r"    \centering",
    r"    \includegraphics[width=0.9\linewidth]{images/Figure5_fiber_length.png}",
    r"    \caption{Ensemble-averaged normalized fiber length ($l/l_0$) over the sprint cycle",
    r"    for BFLH (left), ST (centre), and SM (right). Solid lines show group means;",
    r"    shaded regions are 95\% confidence intervals. Blue = pre, orange = post.",
    r"    Red shading = significant pre-to-post difference (SPM, Bonferroni $\alpha \approx 0.017$).}",
    r"    \label{fig:fiber-length-curves}",
    r"\end{figure}",
])
(_TABLE_DIR / "fiber_length_fig_caption.tex").write_text(_fl_caption)
(_TABLE_DIR / "fiber_length_fig_caption.txt").write_text(_extract_latex_caption(_fl_caption))
print(f"Saved {_TABLE_DIR / 'fiber_length_fig_caption.tex'}")

# =============================================================================
# Interactive peak fiber length plot (Plotly)
# =============================================================================
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _MUSCLES_INT = ["bflh", "semiten", "semimem"]
    _MUSCLE_TITLES_INT = {
        "bflh": "M. Biceps Femoris Long Head",
        "semiten": "M. Semitendinosus",
        "semimem": "M. Semimembranosus",
    }
    _combined_int = pd.concat([pre_df, post_df], ignore_index=True)

    def _eccentric_peak_int(grp: pd.DataFrame, col: str, min_len: int = 5) -> float:
        from analysis.core.biomechanics import eccentric_mask as _em
        mask = _em(grp)
        vals = np.asarray(grp[col], dtype=float)
        return float(np.nanmax(vals[mask]) if mask.any() else np.nanmax(vals))

    _peak_int = (
        _combined_int.groupby(["subject_id", "trial"])
        .apply(lambda g: pd.Series({f"{m}_peak": _eccentric_peak_int(g, f"{m}_fiber_lengths") for m in _MUSCLES_INT}))
        .reset_index()
    )

    _pre_peak = _peak_int[_peak_int["trial"] == "Run_baseline"].set_index("subject_id")
    _post_peak = _peak_int[_peak_int["trial"] == "RunL1"].set_index("subject_id")
    _subjects_int = sorted(set(_pre_peak.index) & set(_post_peak.index))

    fig_int = make_subplots(rows=1, cols=3, subplot_titles=[_MUSCLE_TITLES_INT[m] for m in _MUSCLES_INT])
    for col_idx, muscle in enumerate(_MUSCLES_INT, start=1):
        pcol = f"{muscle}_peak"
        for subj in _subjects_int:
            pre_val = _pre_peak.loc[subj, pcol]
            post_val = _post_peak.loc[subj, pcol]
            color = "green" if post_val < pre_val else "red"
            fig_int.add_trace(
                go.Scatter(
                    x=["Pre", "Post"], y=[pre_val, post_val], mode="lines+markers",
                    line=dict(color=color, width=1.5), marker=dict(size=7),
                    name=f"Subject {subj}", legendgroup=f"subj_{subj}",
                    showlegend=(col_idx == 1),
                    hovertemplate=f"<b>Subject {subj}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
                ),
                row=1, col=col_idx,
            )
    fig_int.update_layout(
        title="Peak normalized Fiber Length — Eccentric Window (Pre vs Post)",
        height=500, hovermode="closest",
        legend=dict(title="Subject", font=dict(size=10)),
    )
    for col_idx in range(1, 4):
        fig_int.update_yaxes(title_text="Peak norm. fiber length (fl/ofl)" if col_idx == 1 else "", row=1, col=col_idx)
    fig_int.write_html(str(_PLOT_DIR / "Figure5_interactive.html"))
    print(f"Saved {_PLOT_DIR / 'Figure5_interactive.html'}")
except ImportError:
    print("plotly not installed — skipping interactive figure")

# =============================================================================
# Pre/post SPM waveform helpers (shared by Figure 4, 5 grids)
# =============================================================================
_GROUP_ROWS_PP = [
    ("Overall", None),
    ("Healthy", "CON"),
    ("FAIS", "FAIS"),
    ("CAM", "FAIM"),
]

_dem_grp_pp = pd.read_csv("data/subject_demographics.csv")
_grp_map_pp = (
    _dem_grp_pp[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)


def _pp_group_subjects(df_: pd.DataFrame, group_code: str) -> pd.Series:
    subj_str = df_["subject_id"].astype(str).str.lstrip("0").str.strip().map(_grp_map_pp)
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
    from scipy.stats import t as _t_pp
    from matplotlib.patches import Patch as _Patch_pp

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
            t_crit = _t_pp.ppf(0.975, df=n - 1)
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
        _Patch_pp(facecolor="#2196F3", alpha=0.2, label="95% CI (Pre)"),
        _Patch_pp(facecolor="#FF9800", alpha=0.2, label="95% CI (Post)"),
        _Patch_pp(facecolor="red", alpha=0.2, label=f"Sig. (SPM, α={alpha:.3f})"),
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


# ── Figure 4: muscle forces ───────────────────────────────────────────────────
_draw_prepost_spm_grid(
    [
        ("M. Biceps Femoris\nLong Head", "bflh_force_per_bw"),
        ("M. Semitendinosus", "semiten_force_per_bw"),
        ("M. Semimembranosus", "semimem_force_per_bw"),
    ],
    r"Muscle Force (N/kg BW)",
    _ALPHA_SPM,
    "Figure4_prepost_spm_muscle_forces",
    caption_tex="\n".join([
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{images/Figure4_prepost_spm_muscle_forces.png}",
        r"    \caption{Ensemble averages of hamstring muscle forces (N/kg BW)",
        r"    pre (blue) and post (orange) the intervention, across clinical groups.",
        r"    Red shading = significant pre-to-post difference (SPM, Bonferroni $\alpha \approx 0.017$).}",
        r"    \label{fig:prepost-spm-muscle-forces}",
        r"\end{figure}",
    ]),
)

# ── Figure 5: fiber lengths ───────────────────────────────────────────────────
_draw_prepost_spm_grid(
    [
        ("M. Biceps Femoris\nLong Head", "bflh_fiber_lengths"),
        ("M. Semitendinosus", "semiten_fiber_lengths"),
        ("M. Semimembranosus", "semimem_fiber_lengths"),
    ],
    r"Norm. Fiber Length ($l/l_0$)",
    _ALPHA_SPM,
    "Figure5_prepost_spm_fiber_lengths",
    caption_tex="\n".join([
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{images/Figure5_prepost_spm_fiber_lengths.png}",
        r"    \caption{Ensemble averages of normalized fiber lengths ($l/l_0$)",
        r"    pre (blue) and post (orange) the intervention, across clinical groups.",
        r"    Red shading = significant pre-to-post difference (SPM, Bonferroni $\alpha \approx 0.017$).}",
        r"    \label{fig:prepost-spm-fiber-lengths}",
        r"\end{figure}",
    ]),
)

# =============================================================================
# Overall RM ANOVA: Time × Muscle — peak force/BW and peak fiber length
# =============================================================================
_MUSCLE_LIST = ["bflh", "semiten", "semimem"]
_MUSCLE_LABEL_MAP = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
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


_MUS_METRICS = {
    "Peak Force/BW (N/kg)": {m: f"{m}_force_per_bw" for m in _MUSCLE_LIST},
    "Peak Norm. Fiber Length (l/l₀)": {m: f"{m}_fiber_lengths" for m in _MUSCLE_LIST},
}

print("\n" + "=" * 65 + "\nOVERALL RM ANOVA — Muscle (time × muscle)\n" + "=" * 65)
_mus_anova_results = {}
for _metric_name, _col_map in _MUS_METRICS.items():
    _res = _overall_rm_anova_block(_metric_name, _col_map, "muscle", _MUSCLE_LIST)
    _mus_anova_results[_metric_name] = _res
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
            _lbl = _MUSCLE_LABEL_MAP.get(_lv, _lv)
            print(f"    {_lbl}: p={_pp:.3f}{'*' if _pp < 0.05 else ''}")

# ── Save Word table ───────────────────────────────────────────────────────────
_EFFECT_LABELS_OV = {
    "time": "Time", "muscle": "Muscle", "time * muscle": "Time × Muscle",
}
_mus_word_rows = []
for _mname, _res in _mus_anova_results.items():
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
        _mus_word_rows.append({
            "Metric": _mname if _first else "",
            "Effect": _EFFECT_LABELS_OV.get(_src, _src),
            "F": f"{_effect_row['F']:.2f}",
            "df₁, df₂": f"{int(_effect_row['ddof1'])}, {int(_effect_row['ddof2'])}",
            "p": _p_str,
            "η²g": f"{_eta2:.3f}" if not pd.isna(_eta2) else "—",
        })
        _first = False

_save_word_table(
    pd.DataFrame(_mus_word_rows),
    str(_TABLE_DIR / "rm_anova_overall_muscle.docx"),
    title="Two-Way RM ANOVA (Time × Muscle): Overall Group — Muscle Forces and Fiber Lengths",
    note="Peak value over the full sprint cycle per subject. "
    "‡ Greenhouse-Geisser correction. η²g = generalised eta-squared. * p < 0.05. "
    "Post-hoc: Bonferroni-corrected paired t-tests (α = 0.05/3 ≈ 0.017), "
    "reported only where the interaction was significant.",
)
print(f"Saved {_TABLE_DIR / 'rm_anova_overall_muscle.docx'}")
