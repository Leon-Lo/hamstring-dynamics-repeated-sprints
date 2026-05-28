"""
Participant demographics and trial-count descriptives.

Covers:
  - Baseline trial counts per subject
  - Peak fiber-length timing (pre vs post, paired test per muscle)
  - Proportion of subjects with increased peak normalized fiber length
  - Between-group equivalence of age and body weight (ANOVA / Kruskal-Wallis)
  - Final combined table assembly (run after all other topic scripts have written their files)
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, f_oneway, kruskal

from utils import Analysis_utils, Subject_utils
from analysis.core.output_helpers import _save_word_table, _extract_latex_caption
from analysis.tables.combined_tables import (
    build_combined_docx as _build_combined_docx,
    build_combined_tex as _build_combined_tex,
)

analysis_utils = Analysis_utils()
subject_utils = Subject_utils()

_TABLE_DIR = Path("data/tables/demographics")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Baseline trial counts
# =============================================================================
df = pd.read_csv("data/master_results.csv")
baseline_counts = df[df["trial"].str.contains("Run_baseline")].groupby("subject_id")["trial"].nunique()
baseline_table = pd.DataFrame({"subject_id": baseline_counts.index, "num_baseline_trials": baseline_counts.values})

(_TABLE_DIR / "baseline_counts.tex").write_text(baseline_table.to_latex(index=False), encoding="utf-8")
_save_word_table(baseline_table, str(_TABLE_DIR / "baseline_counts.docx"), title="Baseline Trial Counts per Subject")
print(f"Saved {_TABLE_DIR / 'baseline_counts.docx'}")

# =============================================================================
# Peak fiber-length timing — pre vs post
# =============================================================================
muscles = ["bflh", "semiten", "semimem"]

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()
subject_ids = sorted(pre_df["subject_id"].unique())


def get_peak_timing_series(df: pd.DataFrame, muscle: str) -> pd.Series:
    """Series (subject_id → gait-cycle %) of peak fiber-length timing."""
    col = f"{muscle}_fiber_lengths"
    result = {}
    for subj, grp in df.groupby("subject_id"):
        row_label = grp[col].idxmax()
        result[subj] = grp.loc[row_label, "index"]
    return pd.Series(result, name=muscle)


print("=" * 65)
print("PEAK MUSCLE FIBER LENGTH TIMING – Pre vs Post (gait-cycle %)")
print("=" * 65)

timing_rows = []
for muscle in muscles:
    pre_timing = get_peak_timing_series(pre_df, muscle).reindex(subject_ids)
    post_timing = get_peak_timing_series(post_df, muscle).reindex(subject_ids)
    change = (post_timing - pre_timing).dropna()
    pre_c = pre_timing.loc[change.index]
    post_c = post_timing.loc[change.index]

    sw = shapiro(change)
    stat_sw, p_sw = sw.statistic, sw.pvalue
    normal = p_sw > 0.05

    if normal:
        _res = ttest_rel(pre_c, post_c)
        stat_test, p_test = _res.statistic, _res.pvalue
        test_name, test_label = "Paired t-test", "t"
    else:
        _res = wilcoxon(pre_c, post_c)
        stat_test, p_test = _res.statistic, _res.pvalue
        test_name, test_label = "Wilcoxon", "W"

    sig = "SIGNIFICANT" if p_test < 0.05 else "ns"
    valid_pre = pre_c[pre_c != 0]
    pct_change = (change.loc[valid_pre.index] / valid_pre) * 100
    pct_str = f"{pct_change.mean():.1f} ± {pct_change.std():.1f}%" if len(pct_change) else "n/a"

    print(f"\n{muscle.upper()}")
    print(f"  Pre  timing: {pre_c.mean():.1f} ± {pre_c.std():.1f} %")
    print(f"  Post timing: {post_c.mean():.1f} ± {post_c.std():.1f} %")
    print(f"  Change:      {change.mean():.1f} ± {change.std():.1f} % ({pct_str})")
    print(f"  Shapiro-Wilk (change): W={stat_sw:.4f}, p={p_sw:.4f}  -> {'normal' if normal else 'non-normal'}")
    print(f"  {test_name}: stat={stat_test:.4f}, p={p_test:.4f}  [{sig}]")

    timing_rows.append(
        {
            "Muscle": muscle,
            "Pre mean±SD (%)": f"{pre_c.mean():.1f} ± {pre_c.std():.1f}",
            "Post mean±SD (%)": f"{post_c.mean():.1f} ± {post_c.std():.1f}",
            "Change mean±SD": f"{change.mean():.1f} ± {change.std():.1f}",
            "Change %": pct_str,
            "SW p": f"{p_sw:.3f}",
            "Test": test_label,
            "Stat": f"{stat_test:.3f}",
            "p": f"{p_test:.3f}",
            "Sig": "*" if p_test < 0.05 else "",
        }
    )

timing_summary_df = pd.DataFrame(timing_rows)
print("\n" + "=" * 65 + "\nSUMMARY TABLE\n" + "=" * 65)
print(timing_summary_df.to_string(index=False))

(_TABLE_DIR / "peak_timing.tex").write_text(timing_summary_df.to_latex(index=False), encoding="utf-8")
_save_word_table(
    timing_summary_df.drop(columns=["Sig"], errors="ignore"),
    str(_TABLE_DIR / "peak_timing.docx"),
    title="Peak Fiber Length Timing: Pre vs. Post Intervention",
    note="Values are gait-cycle percentage at which peak fiber length occurs (mean ± SD). "
    "SW p = Shapiro-Wilk p-value for the change score. "
    "Test = paired t-test (t) or Wilcoxon signed-rank (W). * p < 0.05.",
)
print(f"Saved {_TABLE_DIR / 'peak_timing.docx'}")

# =============================================================================
# Proportion with increased peak normalized fiber length
# =============================================================================
_fl_resp_dict = analysis_utils.get_norm_fiber_length_responders_ids(muscles=["bflh", "semimem"])
_n_total_fl = len(subject_ids)

print("\n" + "=" * 65)
print("PROPORTION WITH INCREASED PEAK NORM. FIBER LENGTH (post > pre)")
print("=" * 65)
for _muscle, _ids in _fl_resp_dict.items():
    _n = len(_ids)
    _label = {"bflh": "BFLH", "semimem": "SM"}.get(_muscle, _muscle.upper())
    print(f"  {_label}: {_n}/{_n_total_fl} ({100 * _n / _n_total_fl:.1f}%) increased")

# =============================================================================
# Demographics equivalence — Age & Body Weight across clinical groups
# =============================================================================
_dem_grp = pd.read_csv("data/subject_demographics.csv")
_dem_check = (
    _dem_grp[["Subject", "Age", "Weight", "Group"]]
    .dropna(subset=["Subject", "Group", "Age", "Weight"])
    .assign(
        Subject=lambda d: d["Subject"].astype(str).str.strip(),
        Group=lambda d: d["Group"].str.strip(),
    )
    .query("Group in ['CON', 'FAIM', 'FAIS']")
)

# Filter to subjects actually in the analysis
_grpwav_pre_trials = ["Run_baselineA1", "Run_baselineB1"]
_grpwav_pre = df[df["trial"].isin(_grpwav_pre_trials)].copy()
_grpwav_subj_mean = _grpwav_pre.groupby(["subject_id", "index"])[["bflh_norm_force"]].mean().reset_index()
_grpwav_subj_mean["subject_str"] = _grpwav_subj_mean["subject_id"].astype(str).str.lstrip("0").str.strip()
_analysis_subjects = set(_grpwav_subj_mean["subject_str"].unique())
_dem_check = _dem_check[_dem_check["Subject"].isin(_analysis_subjects)]

_group_label_map = {"CON": "Healthy", "FAIM": "CAM (no pain)", "FAIS": "CAM (with pain)"}

print("\n" + "=" * 60)
print("DEMOGRAPHICS EQUIVALENCE — Age & Body Weight")
print("=" * 60)

for _var, _col in [("Age (years)", "Age"), ("Body weight (kg)", "Weight")]:
    print(f"\n{_var}")
    _grp_vals = {}
    for _code, _label in _group_label_map.items():
        _vals = _dem_check.loc[_dem_check["Group"] == _code, _col].values
        _grp_vals[_label] = _vals
        print(f"  {_label:20s}: n={len(_vals)}, mean={_vals.mean():.1f}, SD={_vals.std(ddof=1):.1f}")

    _any_nonnormal = any(shapiro(v).pvalue <= 0.05 for v in _grp_vals.values() if len(v) >= 3)
    if _any_nonnormal:
        _stat, _p = kruskal(*_grp_vals.values())
        _test_name, _stat_label = "Kruskal-Wallis H", "H"
    else:
        _stat, _p = f_oneway(*_grp_vals.values())
        _test_name, _stat_label = "One-way ANOVA F", "F"

    _sig = "SIGNIFICANT" if _p < 0.05 else "ns"
    print(f"  {_test_name}: {_stat_label}={_stat:.3f}, p={_p:.3f}  [{_sig}]")

# =============================================================================
# Build combined output tables (run after all other topic scripts)
# NOTE: requires that all individual table files have been written first
# =============================================================================
_build_combined_docx("data/tables/combined_tables.docx")
_build_combined_tex("data/tables/combined_tables.tex")
print("Saved data/tables/combined_tables.docx / .tex")
