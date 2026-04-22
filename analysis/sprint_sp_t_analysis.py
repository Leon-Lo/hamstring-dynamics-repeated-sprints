"""
Sprint spatio-temporal analysis

=======================================
This script performs spatio-temporal analysis with statistics for all subjects in the dataset.

Variables analysed:
- Sprint duration (s)
- Average running speed (m/s)
- Distance covered (m)
- Ground contact time (s)
- Step length (m)

Statistical approach per variable:
1. Shapiro-Wilk normality test on the change score (post - pre).
2. If normal  → paired t-test.
If non-normal → Wilcoxon signed-rank test.
Significance threshold: p = 0.05.
"""

# %% Setup - Add project root to path
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)


import pandas as pd  # noqa: E402

from utils import Analysis_utils, Subject_utils  # noqa: E402

analysis_utils = Analysis_utils()
subject_utils = Subject_utils()


# %% Helper: paired statistics
def paired_stats(pre: pd.Series, post: pd.Series) -> None:
    """
    Print Shapiro-Wilk normality result for the change score, then run either a
    paired t-test (normal) or Wilcoxon signed-rank test (non-normal).
    """
    change = (post - pre).dropna()
    pre_clean = pre.loc[change.index]
    post_clean = post.loc[change.index]

    stat_sw, p_sw = shapiro(change)
    normal = p_sw > 0.05

    if normal:
        stat_test, p_test = ttest_rel(pre_clean, post_clean)
        test_name = "Paired t-test"
    else:
        stat_test, p_test = wilcoxon(pre_clean, post_clean)
        test_name = "Wilcoxon signed-rank"

    sig = "SIGNIFICANT" if p_test < 0.05 else "ns"
    print(f"  Shapiro-Wilk (change): W={stat_sw:.4f}, p={p_sw:.4f}  -> {'normal' if normal else 'non-normal'}")
    print(f"  {test_name}: stat={stat_test:.4f}, p={p_test:.4f}  [{sig}]")


# %% ── 1. SPRINT DURATION ─────────────────────────────────────────
speed_df = analysis_utils.get_speed_df()

print("=" * 60)
print("1. SPRINT DURATION (s)")
print("=" * 60)
duration_df = speed_df[["subject_id", "pre_duration", "post_duration", "duration_change_abs", "duration_change_%"]]
print(duration_df.drop(columns="subject_id").describe().round(3))
paired_stats(speed_df["pre_duration"], speed_df["post_duration"])
print()

# %% ── 2. DISTANCE COVERED ────────────────────────────────────────
print("=" * 60)
print("2. DISTANCE COVERED (m)")
print("=" * 60)
dist_cols = speed_df[["subject_id", "pre_distance_m", "post_distance_m", "distance_change_m"]]
print(dist_cols.drop(columns="subject_id").describe().round(3))
paired_stats(speed_df["pre_distance_m"], speed_df["post_distance_m"])
print()

# %% ── 3. AVERAGE RUNNING SPEED ───────────────────────────────────
print("=" * 60)
print("3. AVERAGE RUNNING SPEED (m/s)")
print("=" * 60)
speed_cols = speed_df[["subject_id", "pre_speed_m_s", "post_speed_m_s", "speed_change_m_s", "speed_change_%"]]
print(speed_cols.drop(columns="subject_id").describe().round(3))
paired_stats(speed_df["pre_speed_m_s"], speed_df["post_speed_m_s"])
print()

# %% ── 4. GROUND CONTACT TIME ─────────────────────────────────────
print("=" * 60)
print("4. GROUND CONTACT TIME (s)")
print("=" * 60)
contact_df = analysis_utils.get_contact_time_df()
print(contact_df.drop(columns="subject_id").describe().round(4))
paired_stats(contact_df["contact_time_pre"], contact_df["contact_time_post"])
print()

# %% ── 5. STEP LENGTH ─────────────────────────────────────────────
print("=" * 60)
print("5. STEP LENGTH (m)")
print("=" * 60)
step_df = analysis_utils.get_step_length_df()
print(step_df.drop(columns="subject_id").describe().round(3))
paired_stats(step_df["step_length_pre"], step_df["step_length_post"])
print()


# %% ── SUMMARY TABLE ──────────────────────────────────────────────
def summary_row(label: str, pre: pd.Series, post: pd.Series, unit: str = "") -> dict:
    """Return a single-row dict for the summary table."""
    change = (post - pre).dropna()
    pre_c = pre.loc[change.index]
    post_c = post.loc[change.index]

    pct_change = (change / pre_c) * 100

    stat_sw, p_sw = shapiro(change)
    normal = p_sw > 0.05

    if normal:
        stat_test, p_test = ttest_rel(pre_c, post_c)
        test_name = "t"

        def fmt(s: pd.Series) -> str:
            return f"{s.mean():.3f} ± {s.std():.3f}"

        fmt_pct = f"{pct_change.mean():.1f} ± {pct_change.std():.1f}"
    else:
        stat_test, p_test = wilcoxon(pre_c, post_c)
        test_name = "W"

        def fmt(s: pd.Series) -> str:
            q1, q3 = np.percentile(s, [25, 75])
            return f"{np.median(s):.3f} ± {(q3 - q1):.3f}"

        q1, q3 = np.percentile(pct_change, [25, 75])
        fmt_pct = f"{np.median(pct_change):.1f} ± {(q3 - q1):.1f}"

    return {
        "Variable": f"{label}{(' (' + unit + ')') if unit else ''}",
        "Pre": fmt(pre_c),
        "Post": fmt(post_c),
        "Change": fmt(change),
        "Change %": fmt_pct,
        "Normality (SW p)": f"{p_sw:.3f}",
        "Test": test_name,
        "Stat": f"{stat_test:.3f}",
        "p": f"{p_test:.3f}",
        "sig": p_test < 0.05,
        "normal": normal,
    }


rows = [
    summary_row("Sprint duration", speed_df["pre_duration"], speed_df["post_duration"], "s"),
    summary_row("Distance", speed_df["pre_distance_m"], speed_df["post_distance_m"], "m"),
    summary_row(r"Running speed ($\frac{m}{s}$)", speed_df["pre_speed_m_s"], speed_df["post_speed_m_s"]),
    summary_row("Contact time", contact_df["contact_time_pre"], contact_df["contact_time_post"], "s"),
    summary_row("Step length", step_df["step_length_pre"], step_df["step_length_post"], "m"),
]

summary_df = pd.DataFrame(rows).drop(columns=["sig", "normal"])
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("=" * 60)
print("SUMMARY TABLE – Pre vs Post (fatigue)")
print("=" * 60)
print(summary_df.to_string(index=False))

# %% ── LATEX TABLE ────────────────────────────────────────────────
# $^*$ = p < 0.05; $^\dagger$ = non-parametric (Wilcoxon), values are median [Q1--Q3]
for row in rows:
    change_str = row["Change"]
    if row["sig"]:
        change_str += "$^*$"
    if not row["normal"]:
        change_str += r"$^\dagger$"
    print(
        f"  {row['Variable']} & {row['Pre']} & {row['Post']} & {change_str} & {row['Change %']}\\% \\\\"
    )
# %%
