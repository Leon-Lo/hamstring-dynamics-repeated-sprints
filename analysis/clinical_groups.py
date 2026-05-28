"""
Clinical group comparisons — CON, FAIS, FAIM waveform descriptives.

Covers:
  - Group mean ± 95% CI waveforms: normalized muscle force and fiber length
    for Healthy (CON), CAM without pain (FAIM), CAM with pain (FAIS)
  - Pre-sprint baseline only (no SPM — F-statistic SPM for 3 groups out of scope)
  - Results paragraphs for combined tables (spatio-temporal, fiber length, muscle work)
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

from utils import Analysis_utils
from analysis.core.output_helpers import _extract_latex_caption
from analysis.tables.combined_tables import (
    build_spatiotemporal_paragraph as _build_spatiotemporal_paragraph,
    build_fiber_length_results_paragraph as _build_fiber_length_results_paragraph,
    build_muscle_work_results_paragraph as _build_muscle_work_results_paragraph,
)

analysis_utils = Analysis_utils()

df = pd.read_csv("data/master_results.csv")

_PLOT_DIR = Path("data/plot_results/clinical_groups")
_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_TABLE_DIR = Path("data/tables/clinical_groups")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Group waveforms — pre-sprint baseline (CON / FAIM / FAIS)
# =============================================================================
_dem_grp = pd.read_csv("data/subject_demographics.csv")
_grp_map = (
    _dem_grp[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)

_GROUP_DEFS = {"Healthy": "CON", "CAM no pain": "FAIM", "CAM with pain": "FAIS"}
_GROUP_COLORS = {"Healthy": "#2196F3", "CAM no pain": "#FF9800", "CAM with pain": "#E53935"}
_GROUP_LS = {"Healthy": "-", "CAM no pain": "--", "CAM with pain": ":"}

_grpwav_pre_trials = ["Run_baselineA1", "Run_baselineB1"]
_grpwav_pre = df[df["trial"].isin(_grpwav_pre_trials)].copy()
_grpwav_force_cols = ["bflh_norm_force", "semiten_norm_force", "semimem_norm_force"]
_grpwav_fl_cols = ["bflh_fiber_lengths", "semiten_fiber_lengths", "semimem_fiber_lengths"]
_grpwav_all_cols = _grpwav_force_cols + _grpwav_fl_cols

_grpwav_subj_mean = _grpwav_pre.groupby(["subject_id", "index"])[_grpwav_all_cols].mean().reset_index()
_grpwav_subj_mean["subject_str"] = _grpwav_subj_mean["subject_id"].astype(str).str.lstrip("0").str.strip()
_grpwav_subj_mean["group_code"] = _grpwav_subj_mean["subject_str"].map(_grp_map)
_valid_codes = set(_GROUP_DEFS.values())
_grpwav_subj_mean = _grpwav_subj_mean[_grpwav_subj_mean["group_code"].isin(_valid_codes)]
_code_to_label = {v: k for k, v in _GROUP_DEFS.items()}
_grpwav_subj_mean["group_label"] = _grpwav_subj_mean["group_code"].map(_code_to_label)


def _grpwav_ci(mat: np.ndarray) -> tuple:
    n = mat.shape[0]
    mn = mat.mean(axis=0)
    if n < 2:
        return mn, np.zeros_like(mn)
    se = mat.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = _stats.t.ppf(0.975, df=n - 1)
    return mn, t_crit * se


_GW_IDX = np.arange(101)
_MUSCLE_COLS_FORCE = {
    "M. Biceps Femoris Long Head": "bflh_norm_force",
    "M. Semitendinosus": "semiten_norm_force",
    "M. Semimembranosus": "semimem_norm_force",
}
_MUSCLE_COLS_FL = {
    "M. Biceps Femoris Long Head": "bflh_fiber_lengths",
    "M. Semitendinosus": "semiten_fiber_lengths",
    "M. Semimembranosus": "semimem_fiber_lengths",
}
_ROW_SPECS = [
    ("Norm. Muscle Force (F/F$_{max}$)", _MUSCLE_COLS_FORCE),
    ("Norm. Fiber Length ($l/l_0$)", _MUSCLE_COLS_FL),
]

fig_gw, axes_gw = plt.subplots(
    len(_ROW_SPECS), 3, figsize=(13, 4 * len(_ROW_SPECS)), sharex=True, constrained_layout=True
)

for _ri, (_ylabel, _col_map) in enumerate(_ROW_SPECS):
    for _ci, (_title, _col) in enumerate(_col_map.items()):
        _ax = axes_gw[_ri, _ci]
        for _label, _code in _GROUP_DEFS.items():
            _mask = _grpwav_subj_mean["group_label"] == _label
            _subj_ids = _grpwav_subj_mean.loc[_mask, "subject_id"].unique()
            _rows = []
            for _s in _subj_ids:
                _sg = _grpwav_subj_mean[(_grpwav_subj_mean["subject_id"] == _s) & _mask].sort_values("index")
                if len(_sg) == 101 and _col in _sg.columns:
                    _rows.append(_sg[_col].values)
            if not _rows:
                continue
            _mat = np.array(_rows)
            _mn, _ci_half = _grpwav_ci(_mat)
            _color = _GROUP_COLORS[_label]
            _ls = _GROUP_LS[_label]
            _ax.plot(_GW_IDX, _mn, color=_color, linestyle=_ls, linewidth=1.8, label=f"{_label} (n={len(_rows)})")
            _ax.fill_between(_GW_IDX, _mn - _ci_half, _mn + _ci_half, color=_color, alpha=0.18)

        if _ri == 0:
            _ax.set_title(_title, fontsize=10, fontweight="bold")
        if _ci == 0:
            _ax.set_ylabel(_ylabel, fontsize=9)
        if _ri == len(_ROW_SPECS) - 1:
            _ax.set_xlabel("Sprint Cycle (%)", fontsize=9)
        _ax.set_xlim(0, 100)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

_gw_legend = [
    Line2D([0], [0], color=_GROUP_COLORS[_l], linestyle=_GROUP_LS[_l], linewidth=2, label=_l)
    for _l in _GROUP_DEFS
] + [Patch(facecolor="gray", alpha=0.25, label="95% CI")]
fig_gw.legend(handles=_gw_legend, loc="upper right", ncol=len(_gw_legend),
              fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.04))

_gw_out = str(_PLOT_DIR / "Figure_group_waveforms.png")
plt.savefig(_gw_out, dpi=300, bbox_inches="tight")
plt.close(fig_gw)
print(f"Saved {_gw_out}")

_gw_caption_tex = "\n".join([
    r"\begin{figure}",
    r"    \centering",
    r"    \includegraphics[width=\linewidth]{images/Figure_group_waveforms.png}",
    r"    \caption{Group mean $\pm$ 95\% confidence interval waveforms over the sprint cycle",
    r"    for hamstring muscles (columns: BFLH, ST, SM) across two biomechanical outcomes",
    r"    (rows). \textit{Top row}: normalized muscle force ($F/F_\mathrm{max}$).",
    r"    \textit{Bottom row}: normalized fiber length ($l/l_0$).",
    r"    Groups: healthy controls (solid blue), asymptomatic CAM (dashed orange),",
    r"    symptomatic CAM (dotted red). Pre repeated sprints baseline only.",
    r"    No statistical inference — SPM $F$-statistic testing across three groups out of scope.}",
    r"    \label{fig:group-waveforms}",
    r"\end{figure}",
])
(_TABLE_DIR / "group_waveforms_caption.tex").write_text(_gw_caption_tex)
(_TABLE_DIR / "group_waveforms_caption.txt").write_text(_extract_latex_caption(_gw_caption_tex))
print(f"Saved {_TABLE_DIR / 'group_waveforms_caption.tex'}")

# =============================================================================
# Results paragraphs for combined output
# =============================================================================
_st_results_para = _build_spatiotemporal_paragraph()
(_TABLE_DIR / "spatiotemporal_results_paragraph.tex").write_text(_st_results_para)
print(f"Saved {_TABLE_DIR / 'spatiotemporal_results_paragraph.tex'}")
print(_st_results_para)

_fl_results_para = _build_fiber_length_results_paragraph()
(_TABLE_DIR / "fiber_length_results_paragraph.tex").write_text(_fl_results_para)
print(f"Saved {_TABLE_DIR / 'fiber_length_results_paragraph.tex'}")
print(_fl_results_para)

_mw_results_para = _build_muscle_work_results_paragraph()
(_TABLE_DIR / "muscle_work_results_paragraph.tex").write_text(_mw_results_para)
print(f"Saved {_TABLE_DIR / 'muscle_work_results_paragraph.tex'}")
print(_mw_results_para)
