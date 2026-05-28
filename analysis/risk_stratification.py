"""
Risk stratification — responder classification and group comparisons.

Covers:
  - Classify subjects as risk-prone (positive BFLH peak fiber length increase)
  - Change scores (Δ = post − pre) compared between risk-prone and low-risk groups:
      spatio-temporal parameters, mean joint angles
  - SPM independent t-test (Welch) group comparisons:
      joint angles, joint moments (Figures spm_group_hip/knee/ankle)
      muscle forces (Figure spm_group_muscle_forces)
      adjusted EMG (Figure spm_group_adjusted_emg)
  - Word tables for group comparison results

SPM alpha = 0.05/3 (Bonferroni across 3 joints for kinematic figures).
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
from scipy.stats import shapiro, t as t_dist, mannwhitneyu, ttest_ind
import spm1d

from utils import Analysis_utils
from analysis.core.output_helpers import _save_word_table, _extract_latex_caption

analysis_utils = Analysis_utils()

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()
subject_ids = sorted(pre_df["subject_id"].unique())

_PLOT_DIR = Path("data/plot_results/risk_stratification")
_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_TABLE_DIR = Path("data/tables/risk_stratification")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Responder classification: BFLH peak fiber length increase → risk-prone
# =============================================================================
_resp_dict = analysis_utils.get_norm_fiber_length_responders_ids(muscles=["bflh"])
_responder_ids = {int(s) for s in _resp_dict["bflh"]}
_bflh_resp = analysis_utils.get_norm_fiber_length_responders_ids(muscles=["bflh"])
_risk_ids = set(int(s) for s in _bflh_resp["bflh"])
_all_spm_subj = sorted(set(int(s) for s in pre_df["subject_id"]) | set(int(s) for s in post_df["subject_id"]))
_nonrisk_ids = set(_all_spm_subj) - _risk_ids

print(f"Risk prone  (BFLH responders): n={len(_risk_ids)},  IDs={sorted(_risk_ids)}")
print(f"Low risk:                      n={len(_nonrisk_ids)}, IDs={sorted(_nonrisk_ids)}")

# =============================================================================
# Change score comparison: spatio-temporal parameters
# =============================================================================
def _group_comparison(resp_vals: np.ndarray, nonresp_vals: np.ndarray) -> dict | None:
    if len(resp_vals) < 3 or len(nonresp_vals) < 3:
        return None
    sw_r_p = shapiro(resp_vals).pvalue
    sw_nr_p = shapiro(nonresp_vals).pvalue
    if sw_r_p > 0.05 and sw_nr_p > 0.05:
        res = ttest_ind(resp_vals, nonresp_vals)
        return {"test": "t", "stat": float(res.statistic), "p": float(res.pvalue)}
    res = mannwhitneyu(resp_vals, nonresp_vals, alternative="two-sided")
    return {"test": "U", "stat": float(res.statistic), "p": float(res.pvalue)}


def _fmt_delta(vals: np.ndarray) -> str:
    return f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) else "n/a"


_st_df = pd.read_csv("data/spatiotemporal_stats.csv")
_st_df["_resp"] = _st_df["subject_id"].apply(lambda x: int(x) in _responder_ids)
_n_resp_st = int(_st_df["_resp"].sum())
_n_nonresp_st = int((~_st_df["_resp"]).sum())

_ST_VARS = {
    "Duration (s)": ("pre_duration", "post_duration"),
    "Step Length (m)": ("pre_distance_m", "post_distance_m"),
    "Speed (m/s)": ("pre_speed_m_s", "post_speed_m_s"),
    "Contact time (s)": ("contact_time_pre", "contact_time_post"),
}

print("=" * 70)
print("RISK PRONE vs. LOW RISK – Spatio-Temporal Parameters")
print(f"  Risk prone n={_n_resp_st},  Low risk n={_n_nonresp_st}")
print("=" * 70)

_resp_st_rows = []
for _vl, (_pre_c, _post_c) in _ST_VARS.items():
    _delta = _st_df[_post_c] - _st_df[_pre_c]
    _r_vals = _delta[_st_df["_resp"]].dropna().values
    _nr_vals = _delta[~_st_df["_resp"]].dropna().values
    _gc = _group_comparison(_r_vals, _nr_vals)
    if _gc is None:
        continue
    _sig = "SIGNIFICANT" if _gc["p"] < 0.05 else "ns"
    print(f"\n  {_vl}")
    print(f"    Risk prone  Δ: {_fmt_delta(_r_vals)}")
    print(f"    Low risk    Δ: {_fmt_delta(_nr_vals)}")
    print(f"    {_gc['test']}={_gc['stat']:.3f}, p={_gc['p']:.3f}  [{_sig}]")
    _resp_st_rows.append({
        "Variable": _vl,
        "Risk Prone Δ": _fmt_delta(_r_vals), "Low Risk Δ": _fmt_delta(_nr_vals),
        "Test": _gc["test"], "Stat": f"{_gc['stat']:.3f}",
        "p": f"{_gc['p']:.3f}", "Sig": "*" if _gc["p"] < 0.05 else "",
    })

_st_df.drop(columns=["_resp"], inplace=True)

# =============================================================================
# Change score comparison: kinematics
# =============================================================================
_KIN_COLS = {
    "Hip flexion (°)": "hip_flexion",
    "Knee angle (°)": "knee_angle",
    "Ankle angle (°)": "ankle_angle",
    "Hip adduction (°)": "hip_adduction",
    "Hip rotation (°)": "hip_rotation",
}

_pre_mean_ang = pre_df.groupby("subject_id")[list(_KIN_COLS.values())].mean()
_post_mean_ang = post_df.groupby("subject_id")[list(_KIN_COLS.values())].mean()
_common_kin = _pre_mean_ang.index.intersection(_post_mean_ang.index)
_delta_ang = _post_mean_ang.loc[_common_kin] - _pre_mean_ang.loc[_common_kin]
_is_resp_kin = pd.Series({s: int(s) in _responder_ids for s in _common_kin})
_n_resp_kin = int(_is_resp_kin.sum())
_n_nonresp_kin = int((~_is_resp_kin).sum())

print("\n" + "=" * 70)
print("RISK PRONE vs. LOW RISK – Mean Joint Angle Change Scores")
print(f"  Risk prone n={_n_resp_kin},  Low risk n={_n_nonresp_kin}")
print("=" * 70)

_resp_kin_rows = []
for _kl, _kc in _KIN_COLS.items():
    _r_vals = _delta_ang.loc[_is_resp_kin[_is_resp_kin].index, _kc].dropna().values
    _nr_vals = _delta_ang.loc[_is_resp_kin[~_is_resp_kin].index, _kc].dropna().values
    _gc = _group_comparison(_r_vals, _nr_vals)
    if _gc is None:
        continue
    _sig = "SIGNIFICANT" if _gc["p"] < 0.05 else "ns"
    print(f"\n  {_kl}")
    print(f"    Risk prone  Δ: {_fmt_delta(_r_vals)}°")
    print(f"    Low risk    Δ: {_fmt_delta(_nr_vals)}°")
    print(f"    {_gc['test']}={_gc['stat']:.3f}, p={_gc['p']:.3f}  [{_sig}]")
    _resp_kin_rows.append({
        "Variable": _kl,
        "Risk Prone Δ": _fmt_delta(_r_vals), "Low Risk Δ": _fmt_delta(_nr_vals),
        "Test": _gc["test"], "Stat": f"{_gc['stat']:.3f}",
        "p": f"{_gc['p']:.3f}", "Sig": "*" if _gc["p"] < 0.05 else "",
    })

# ── Save tables ───────────────────────────────────────────────────────────────
if _resp_st_rows:
    _save_word_table(
        pd.DataFrame(_resp_st_rows).drop(columns=["Sig"], errors="ignore"),
        str(_TABLE_DIR / "responder_spatiotemporal.docx"),
        title="Risk Prone vs. Low Risk: Spatio-Temporal Change Scores",
        note=f"Values are mean ± SD of change scores (Δ = post − pre). "
        f"Risk prone (n={_n_resp_st}) = positive % increase in peak normalized BFLH fiber length. "
        f"Test = t (independent t-test) or U (Mann-Whitney U). * p < 0.05.",
    )
    print(f"Saved {_TABLE_DIR / 'responder_spatiotemporal.docx'}")

if _resp_kin_rows:
    _save_word_table(
        pd.DataFrame(_resp_kin_rows).drop(columns=["Sig"], errors="ignore"),
        str(_TABLE_DIR / "responder_kinematics.docx"),
        title="Risk Prone vs. Low Risk: Mean Joint Angle Change Scores",
        note=f"Values are mean ± SD of change in sprint-cycle mean joint angle (Δ = post − pre, degrees). "
        f"Risk prone (n={_n_resp_kin}); low risk n={_n_nonresp_kin}. * p < 0.05.",
    )
    print(f"Saved {_TABLE_DIR / 'responder_kinematics.docx'}")

# =============================================================================
# SPM group comparisons — waveform figures (risk-prone vs low-risk)
# =============================================================================
_SPM_IDX = np.arange(101)
_N_SPM_JOINTS = 3
_ALPHA_SPM = 0.05 / _N_SPM_JOINTS
_ALPHA_SPM_MF = 0.05 / 3
_ALPHA_SPM_EMG = 0.05 / 3

# Build post − pre difference arrays per subject
_diff_subjs = sorted(set(int(s) for s in pre_df["subject_id"]) & set(int(s) for s in post_df["subject_id"]))
_diff_cols = [c for c in pre_df.columns if c not in ("subject_id", "index", "trial", "time", "leg")]
_diff_records = []
for _s in _diff_subjs:
    _pg = pre_df[pre_df["subject_id"] == _s].sort_values("index").reset_index(drop=True)
    _qg = post_df[post_df["subject_id"] == _s].sort_values("index").reset_index(drop=True)
    if len(_pg) == 101 and len(_qg) == 101:
        _valid_cols = [_c for _c in _diff_cols if _c in _qg.columns]
        _diff_data = {_c: _qg[_c].values - _pg[_c].values for _c in _valid_cols}
        _d = pd.concat([_pg[["subject_id", "index"]].reset_index(drop=True), pd.DataFrame(_diff_data)], axis=1)
        _diff_records.append(_d)
_diff_df_spm = pd.concat(_diff_records, ignore_index=True) if _diff_records else pd.DataFrame()

_COL_SPECS = [("Pre", pre_df), ("Post", post_df), ("Difference", _diff_df_spm)]


def _subj_series(df_: pd.DataFrame, subj: int, col: str) -> "np.ndarray | None":
    g = df_[df_["subject_id"] == subj].sort_values("index")
    if col not in g.columns or g.empty:
        return None
    vals = g[col].values
    return vals if len(vals) == 101 else None


def _group_matrix(df_: pd.DataFrame, col: str, ids: set) -> np.ndarray:
    rows = [v for s in sorted(ids) if (v := _subj_series(df_, s, col)) is not None]
    return np.array(rows) if rows else np.empty((0, 101))


def _draw_spm_figure(fspec, alpha, out_name):
    n_rows = len(fspec["rows"])
    _fig, _axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows), sharex=True, constrained_layout=True)
    if n_rows == 1:
        _axes = _axes[np.newaxis, :]

    for _row_idx, (_param, _ylabel) in enumerate(fspec["rows"]):
        for _col_idx, (_cond_label, _cdf) in enumerate(_COL_SPECS):
            _ax = _axes[_row_idx, _col_idx]

            if isinstance(_cdf, pd.DataFrame) and (_cdf.empty or _param not in _cdf.columns):
                _ax.set_visible(False)
                continue

            _risk_mat = _group_matrix(_cdf, _param, _risk_ids)
            _nr_mat = _group_matrix(_cdf, _param, _nonrisk_ids)

            for _mat, _color, _ls in [(_nr_mat, "green", "-"), (_risk_mat, "red", "--")]:
                if _mat.shape[0] < 2:
                    continue
                _mn = _mat.mean(axis=0)
                _n = _mat.shape[0]
                _ci = t_dist.ppf(0.975, _n - 1) * _mat.std(axis=0, ddof=1) / np.sqrt(_n)
                _ax.plot(_SPM_IDX, _mn, color=_color, linestyle=_ls, linewidth=1.8)
                _ax.fill_between(_SPM_IDX, _mn - _ci, _mn + _ci, color=_color, alpha=0.2)

            if _risk_mat.shape[0] >= 2 and _nr_mat.shape[0] >= 2:
                try:
                    _t2 = spm1d.stats.ttest2(_risk_mat, _nr_mat, equal_var=False)
                    _ti2 = _t2.inference(alpha, two_tailed=True)
                    _sig = np.abs(_ti2.z) > _ti2.zstar
                    _ax.fill_between(_SPM_IDX, 0, 1, where=_sig, color="pink", alpha=0.35,
                                     transform=_ax.get_xaxis_transform(), zorder=0)
                except Exception as _e:
                    print(f"  SPM failed for {_param} ({_cond_label}): {_e}")

            if _row_idx == 0:
                _ax.set_title(_cond_label, fontweight="bold")
            if _col_idx == 0:
                _ax.set_ylabel(_ylabel)
            if _row_idx == n_rows - 1:
                _ax.set_xlabel("Sprint Cycle (%)")
            _ax.set_xlim(0, 100)
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)

    _legend_h = [
        Line2D([0], [0], color="green", linestyle="-", linewidth=2, label="Low Risk"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Risk Prone"),
        plt.matplotlib.patches.Patch(facecolor="pink", alpha=0.5, label="Significant (SPM)"),
    ]
    _fig.legend(handles=_legend_h, loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.06))
    _spm_out = str(_PLOT_DIR / f"{out_name}.png")
    plt.savefig(_spm_out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(_fig)
    print(f"Saved {_spm_out}")


_FIGURE_SPECS = [
    {"name": "hip", "title": "Hip",
     "rows": [("hip_flexion", "Angle (°)"), ("hip_flexion_moment", "Moment (Nm/kg)")]},
    {"name": "knee", "title": "Knee",
     "rows": [("knee_angle", "Angle (°)"), ("knee_angle_moment", "Moment (Nm/kg)")]},
    {"name": "ankle", "title": "Ankle",
     "rows": [("ankle_angle", "Angle (°)"), ("ankle_angle_moment", "Moment (Nm/kg)")]},
]

for _fspec in _FIGURE_SPECS:
    _draw_spm_figure(_fspec, _ALPHA_SPM, f"Figure_spm_group_{_fspec['name']}")

    _spm_row_labels = ", ".join(f"{r[1]}" for r in _fspec["rows"])
    _spm_cap_tex = "\n".join([
        r"\begin{figure}",
        r"    \centering",
        rf"    \includegraphics[width=\linewidth]{{images/Figure_spm_group_{_fspec['name']}.png}}",
        rf"    \caption{{SPM group comparison (Risk Prone vs.\ Low Risk) for {_fspec['title'].lower()} joint",
        rf"    biomechanics. Rows: {_spm_row_labels}. Columns: Pre, Post, Difference (post $-$ pre).",
        r"    Green solid = Low Risk; dashed red = Risk Prone. Pink = significant between-group",
        rf"    difference (SPM Welch $t$-test, Bonferroni $\alpha = 0.05/{_N_SPM_JOINTS} \approx {_ALPHA_SPM:.4f}$).}}",
        rf"    \label{{fig:spm-group-{_fspec['name']}}}",
        r"\end{figure}",
    ])
    (_TABLE_DIR / f"spm_group_{_fspec['name']}_caption.tex").write_text(_spm_cap_tex)
    (_TABLE_DIR / f"spm_group_{_fspec['name']}_caption.txt").write_text(_extract_latex_caption(_spm_cap_tex))
    print(f"Saved {_TABLE_DIR / f'spm_group_{_fspec[\"name\"]}_caption.tex'}")

# ── Muscle force SPM figure ───────────────────────────────────────────────────
_MUSCLE_FORCE_SPEC = {
    "name": "muscle_forces", "title": "Muscle Forces",
    "rows": [
        ("bflh_norm_force", "M. Biceps Femoris Long Head\nNorm. Force (F/Fmax)"),
        ("semiten_norm_force", "M. Semitendinosus\nNorm. Force (F/Fmax)"),
        ("semimem_norm_force", "M. Semimembranosus\nNorm. Force (F/Fmax)"),
    ],
}
_draw_spm_figure(_MUSCLE_FORCE_SPEC, _ALPHA_SPM_MF, "Figure_spm_group_muscle_forces")

_mf_cap_tex = "\n".join([
    r"\begin{figure}",
    r"    \centering",
    r"    \includegraphics[width=\linewidth]{images/Figure_spm_group_muscle_forces.png}",
    r"    \caption{Normalized hamstring muscle force waveforms ($F/F_\mathrm{max}$) comparing",
    r"    Risk Prone and Low Risk groups across the sprint cycle.",
    r"    Rows show BFLH (top), ST (middle), SM (bottom). Pink shading marks significant",
    rf"    between-group difference (SPM Welch $t$-test, Bonferroni $\alpha \approx {_ALPHA_SPM_MF:.3f}$).}}",
    r"    \label{fig:spm-group-muscle-forces}",
    r"\end{figure}",
])
(_TABLE_DIR / "spm_group_muscle_forces_caption.tex").write_text(_mf_cap_tex)
(_TABLE_DIR / "spm_group_muscle_forces_caption.txt").write_text(_extract_latex_caption(_mf_cap_tex))
print(f"Saved {_TABLE_DIR / 'spm_group_muscle_forces_caption.tex'}")

# ── Adjusted EMG SPM figure ───────────────────────────────────────────────────
_ADJUSTED_EMG_SPEC = {
    "name": "adjusted_emg", "title": "Adjusted EMG",
    "rows": [
        ("bflh_adjusted_emg", "M. Biceps Femoris Long Head\nAdjusted EMG"),
        ("semiten_adjusted_emg", "M. Semitendinosus\nAdjusted EMG"),
        ("semimem_adjusted_emg", "M. Semimembranosus\nAdjusted EMG"),
    ],
}
_draw_spm_figure(_ADJUSTED_EMG_SPEC, _ALPHA_SPM_EMG, "Figure_spm_group_adjusted_emg")

_emg_cap_tex = "\n".join([
    r"\begin{figure}",
    r"    \centering",
    r"    \includegraphics[width=\linewidth]{images/Figure_spm_group_adjusted_emg.png}",
    r"    \caption{Adjusted EMG waveforms comparing Risk Prone and Low Risk groups across the sprint cycle.",
    r"    Rows show BFLH (top), ST (middle), SM (bottom). Pink shading marks significant",
    rf"    between-group difference (SPM Welch $t$-test, Bonferroni $\alpha \approx {_ALPHA_SPM_EMG:.3f}$).}}",
    r"    \label{fig:spm-group-adjusted-emg}",
    r"\end{figure}",
])
(_TABLE_DIR / "spm_group_adjusted_emg_caption.tex").write_text(_emg_cap_tex)
(_TABLE_DIR / "spm_group_adjusted_emg_caption.txt").write_text(_extract_latex_caption(_emg_cap_tex))
print(f"Saved {_TABLE_DIR / 'spm_group_adjusted_emg_caption.tex'}")
