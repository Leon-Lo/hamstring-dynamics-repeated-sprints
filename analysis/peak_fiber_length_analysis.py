# %%
# =============================================================================
# PEAK NORMALIZED FIBER LENGTH ANALYSIS
# Pre vs. Post Intervention Comparison of Hamstring Muscle Peak Normalized Fiber Lengths
# =============================================================================
# This script analyses peak muscle normalized fiber lengths for three hamstring muscles:
#   - Biceps Femoris Long Head (bflh)
#   - Semitendinosus (semiten)
#   - Semimembranosus (semimem)
#
# The analysis is structured in four stages:
#   1. Data loading and % change calculation
#   2. Whole-group descriptive stats and significance testing
#   3. Responder-only significance testing
#   4. Running speed comparison (pre vs. post)
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats

from src import Simulation_analyzer
from utils import (
    Analysis_utils,
    Data_utils,
    File_utils,
    Plot_utils,
    Subject_utils,
    XML_utils,
)

plots = Plot_utils()
file_utils = File_utils()
subject_utils = Subject_utils()
data_utils = Data_utils()
analysis_utils = Analysis_utils()
tools = Simulation_analyzer()
xml_utils = XML_utils()

# Define the three hamstring muscles of interest
muscles = ["bflh", "semiten", "semimem"]


def direction_phrase(value, decimals=1):
    """Return a neutral signed change phrase for reporting percent changes."""
    return f"changed by {value:+.{decimals}f}%"


# %%
# =============================================================================
# STAGE 1: DATA LOADING AND % CHANGE CALCULATION
# =============================================================================

# Load raw peak normalized fiber length data (one row per subject, pre and post columns per muscle)
norm_fiber_length_df = analysis_utils.get_peak_norm_fiber_length_df()

# Calculate percentage change (post vs. pre) for each muscle
# Negative values indicate a reduction; positive values indicate an increase
for muscle in muscles:
    pre_col = f"{muscle}_peak_norm_fiber_length_pre"
    post_col = f"{muscle}_peak_norm_fiber_length_post"
    norm_fiber_length_df[f"{muscle}_%_change"] = (
        (norm_fiber_length_df[post_col] - norm_fiber_length_df[pre_col]) / norm_fiber_length_df[pre_col]
    ) * 100

# Build ordered list of column names for convenient indexing later:
# [pre, post, %change] x 3 muscles → indices 0-2 bflh, 3-5 semiten, 6-8 semimem
muscle_norm_fiber_length_names = []
for muscle in muscles:
    muscle_norm_fiber_length_names.extend(
        [
            f"{muscle}_peak_norm_fiber_length_pre",
            f"{muscle}_peak_norm_fiber_length_post",
            f"{muscle}_%_change",
        ]
    )

print("Peak normalized fiber length data loaded. Columns:", muscle_norm_fiber_length_names)

# %%
# =============================================================================
# STAGE 2: WHOLE-GROUP ANALYSIS (ALL SUBJECTS)
# =============================================================================
# First we look at the full cohort to understand the overall intervention effect
# before subsetting by responder status.

# --- Descriptive statistics (all subjects) ---
desc_stats = norm_fiber_length_df.describe()
desc_stats

# Summarise mean ± SD % change and absolute values for each muscle
pct_bflh_mean = desc_stats.loc["mean", muscle_norm_fiber_length_names[2]]
pct_bflh_std = desc_stats.loc["std", muscle_norm_fiber_length_names[2]]
pct_st_mean = desc_stats.loc["mean", muscle_norm_fiber_length_names[5]]
pct_st_std = desc_stats.loc["std", muscle_norm_fiber_length_names[5]]
pct_sm_mean = desc_stats.loc["mean", muscle_norm_fiber_length_names[8]]
pct_sm_std = desc_stats.loc["std", muscle_norm_fiber_length_names[8]]


def latex_pct_mean_sd(mean, std, decimals=1, signed=True):
    """Format mean±SD percentage values for direct LaTeX use."""
    mean_str = f"{mean:+.{decimals}f}" if signed else f"{mean:.{decimals}f}"
    return f"{mean_str}\\% $\\pm$ {std:.{decimals}f}\\%"


def latex_abs_mean_sd(mean, std, decimals=3):
    """Format non-percentage mean±SD values for direct LaTeX use."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


print(
    f"Peak normalized fiber lengths changed on average by "
    f"{latex_pct_mean_sd(pct_bflh_mean, pct_bflh_std)} for bflh "
    f"(pre: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[0]], desc_stats.loc['std', muscle_norm_fiber_length_names[0]])}, "
    f"post: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[1]], desc_stats.loc['std', muscle_norm_fiber_length_names[1]])}), "
    f"{latex_pct_mean_sd(pct_st_mean, pct_st_std)} for semitendinosus "
    f"(pre: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[3]], desc_stats.loc['std', muscle_norm_fiber_length_names[3]])}, "
    f"post: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[4]], desc_stats.loc['std', muscle_norm_fiber_length_names[4]])}), and "
    f"{latex_pct_mean_sd(pct_sm_mean, pct_sm_std)} for semimembranosus "
    f"(pre: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[6]], desc_stats.loc['std', muscle_norm_fiber_length_names[6]])}, "
    f"post: {latex_abs_mean_sd(desc_stats.loc['mean', muscle_norm_fiber_length_names[7]], desc_stats.loc['std', muscle_norm_fiber_length_names[7]])})."
)

# --- Paired t-tests (all subjects) ---
# Tests whether the pre-to-post change is statistically significant
# across the full cohort regardless of responder status.
bflh_t_all = stats.ttest_rel(
    norm_fiber_length_df[muscle_norm_fiber_length_names[0]],
    norm_fiber_length_df[muscle_norm_fiber_length_names[1]],
)
st_t_all = stats.ttest_rel(
    norm_fiber_length_df[muscle_norm_fiber_length_names[3]],
    norm_fiber_length_df[muscle_norm_fiber_length_names[4]],
)
sm_t_all = stats.ttest_rel(
    norm_fiber_length_df[muscle_norm_fiber_length_names[6]],
    norm_fiber_length_df[muscle_norm_fiber_length_names[7]],
)

print("\n--- Whole-group paired t-tests ---")
print(f"Biceps Femoris Long Head:  t={bflh_t_all.statistic:.3f}, p={bflh_t_all.pvalue:.3f}")
print(f"Semitendinosus:            t={st_t_all.statistic:.3f},   p={st_t_all.pvalue:.3f}")
print(f"Semimembranosus:           t={sm_t_all.statistic:.3f},   p={sm_t_all.pvalue:.3f}")

# --- Maximum % change per muscle (identify individual outliers) ---
# Useful for flagging subjects with unusually large responses
print("\n--- Top 3 maximum % change per muscle (subject, value) ---")
for muscle in muscles:
    pct_col = f"{muscle}_%_change"
    top3 = norm_fiber_length_df.nlargest(3, pct_col)[["subject_id", pct_col]]
    print(f"{muscle}:")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        print(f"  {rank}. subject {row['subject_id']}, {row[pct_col]:.1f}%")

# %%
# =============================================================================
# STAGE 3: RESPONDER-ONLY ANALYSIS
# =============================================================================
# Responders are subjects who showed a meaningful change in peak normalized fiber length for
# a given muscle. Each muscle has its own responder subset because a subject
# may respond in one muscle but not another.
#
# Note: subsetting by response direction increases the likelihood of finding
# significance — these results should be interpreted alongside the whole-group
# analysis above.

# %%
# --- Stage 3A: Load responder stats and IDs ---

# Load pre-computed summary stats and responder IDs
norm_fiber_length_summary = analysis_utils.get_peak_norm_fiber_length_summary_stats()
responders_norm_fiber_length_summary = analysis_utils.get_peak_norm_fiber_length_responders_summary_stats()
responders_id = analysis_utils.get_norm_fiber_length_responders_ids()

print("norm_fiber_length_summary:")
print(norm_fiber_length_summary)
print("responders_norm_fiber_length_summary:")
print(responders_norm_fiber_length_summary)
print("responders_id:")
print(responders_id)

total_subjects = norm_fiber_length_df["subject_id"].nunique()
responder_sets = {muscle: set(responders_id[muscle]) for muscle in muscles}

print("\n--- Responder status counts by muscle ---")
for muscle in muscles:
    n_responders = len(responder_sets[muscle])
    n_non_responders = total_subjects - n_responders
    print(f"{muscle}: n responders={n_responders}, n non-responders={n_non_responders}")

responders_all_muscles = set.intersection(*responder_sets.values())
print(f"Responders in all muscles: n={len(responders_all_muscles)}")

# %%
# --- Stage 3B: Top responder changes by muscle ---

# print top 3 max values
# Useful for flagging subjects with unusually large responses
print("\n--- Top 3 maximum % change per muscle (subject, value) ---")
for muscle in muscles:
    pct_col = f"{muscle}_%_change"
    top3 = norm_fiber_length_df.nlargest(3, pct_col)[["subject_id", pct_col]]
    print(f"{muscle}:")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        print(f"  {rank}. subject {row['subject_id']}, {row[pct_col]:.1f}%")

    # %%
    # --- Stage 3C: Responder-only paired t-tests ---


# Helper function to run and print a responder paired t-test for one muscle
def run_responder_ttest(muscle_label, responder_ids, pre_col, post_col):
    """Filter normalized fiber length dataframe to responders, run paired t-test, print result."""
    df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responder_ids)][["subject_id", pre_col, post_col]]
    t_result = stats.ttest_rel(df[pre_col], df[post_col])
    n_responders = len(df)
    n_non_responders = total_subjects - n_responders
    print(
        f"{muscle_label}: t={t_result.statistic:.3f}, p={t_result.pvalue:.3f} "
        f"(n responders={n_responders}, n non-responders={n_non_responders})"
    )
    return t_result


print("\n--- Responder-only paired t-tests ---")
bflh_t_resp = run_responder_ttest(
    "Biceps Femoris Long Head (responders)",
    responders_id["bflh"],
    muscle_norm_fiber_length_names[0],
    muscle_norm_fiber_length_names[1],
)
st_t_resp = run_responder_ttest(
    "Semitendinosus (responders)",
    responders_id["semiten"],
    muscle_norm_fiber_length_names[3],
    muscle_norm_fiber_length_names[4],
)
sm_t_resp = run_responder_ttest(
    "Semimembranosus (responders)",
    responders_id["semimem"],
    muscle_norm_fiber_length_names[6],
    muscle_norm_fiber_length_names[7],
)

# %%
# --- Stage 3D: Responder-only LaTeX summary sentence ---

# Responder-only LaTeX-ready descriptive sentence (same format as whole-group summary)
bflh_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["bflh"])]
bflh_resp_pct = (
    (bflh_resp_df[muscle_norm_fiber_length_names[1]] - bflh_resp_df[muscle_norm_fiber_length_names[0]])
    / bflh_resp_df[muscle_norm_fiber_length_names[0]]
) * 100

st_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["semiten"])]
st_resp_pct = (
    (st_resp_df[muscle_norm_fiber_length_names[4]] - st_resp_df[muscle_norm_fiber_length_names[3]])
    / st_resp_df[muscle_norm_fiber_length_names[3]]
) * 100

sm_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["semimem"])]
sm_resp_pct = (
    (sm_resp_df[muscle_norm_fiber_length_names[7]] - sm_resp_df[muscle_norm_fiber_length_names[6]])
    / sm_resp_df[muscle_norm_fiber_length_names[6]]
) * 100

bflh_resp_desc = bflh_resp_df.describe()
st_resp_desc = st_resp_df.describe()
sm_resp_desc = sm_resp_df.describe()

bflh_resp_pct_mean = bflh_resp_pct.mean()
bflh_resp_pct_std = bflh_resp_pct.std()
st_resp_pct_mean = st_resp_pct.mean()
st_resp_pct_std = st_resp_pct.std()
sm_resp_pct_mean = sm_resp_pct.mean()
sm_resp_pct_std = sm_resp_pct.std()

print(
    "Responder-only peak normalized fiber lengths changed on average by "
    f"{latex_pct_mean_sd(bflh_resp_pct_mean, bflh_resp_pct_std)} for bflh "
    f"(pre: {latex_abs_mean_sd(bflh_resp_desc.loc['mean', muscle_norm_fiber_length_names[0]], bflh_resp_desc.loc['std', muscle_norm_fiber_length_names[0]])}, "
    f"post: {latex_abs_mean_sd(bflh_resp_desc.loc['mean', muscle_norm_fiber_length_names[1]], bflh_resp_desc.loc['std', muscle_norm_fiber_length_names[1]])}), "
    f"{latex_pct_mean_sd(st_resp_pct_mean, st_resp_pct_std)} for semitendinosus "
    f"(pre: {latex_abs_mean_sd(st_resp_desc.loc['mean', muscle_norm_fiber_length_names[3]], st_resp_desc.loc['std', muscle_norm_fiber_length_names[3]])}, "
    f"post: {latex_abs_mean_sd(st_resp_desc.loc['mean', muscle_norm_fiber_length_names[4]], st_resp_desc.loc['std', muscle_norm_fiber_length_names[4]])}), and "
    f"{latex_pct_mean_sd(sm_resp_pct_mean, sm_resp_pct_std)} for semimembranosus "
    f"(pre: {latex_abs_mean_sd(sm_resp_desc.loc['mean', muscle_norm_fiber_length_names[6]], sm_resp_desc.loc['std', muscle_norm_fiber_length_names[6]])}, "
    f"post: {latex_abs_mean_sd(sm_resp_desc.loc['mean', muscle_norm_fiber_length_names[7]], sm_resp_desc.loc['std', muscle_norm_fiber_length_names[7]])})."
)

# %%
# =============================================================================
# STAGE 4: RUNNING SPEED COMPARISON (PRE vs. POST)
# =============================================================================
# Changes in peak normalized fiber length could be confounded by changes in running speed.
# Here we calculate average speed per subject pre and post intervention
# and test whether speed changed significantly.

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()

speed_records = []

for subject_id in pre_df["subject_id"].unique():
    pre_sub = pre_df[pre_df["subject_id"] == subject_id]
    post_sub = post_df[post_df["subject_id"] == subject_id]

    # Duration = time elapsed over the captured trial
    pre_duration = pre_sub["time"].iloc[-1] - pre_sub["time"].iloc[0]
    post_duration = post_sub["time"].iloc[-1] - post_sub["time"].iloc[0]

    # Distance = displacement of pelvis marker along the running direction (tx)
    pre_distance = pre_sub["pelvis_tx"].iloc[-1] - pre_sub["pelvis_tx"].iloc[0]
    post_distance = post_sub["pelvis_tx"].iloc[-1] - post_sub["pelvis_tx"].iloc[0]

    pre_speed = pre_distance / pre_duration
    post_speed = post_distance / post_duration

    speed_records.append(
        {
            "subject_id": subject_id,
            "pre_duration": pre_duration,
            "post_duration": post_duration,
            "duration_change_%": ((post_duration - pre_duration) / pre_duration) * 100,
            "pre_distance_m": pre_distance,
            "post_distance_m": post_distance,
            "distance_change_m": post_distance - pre_distance,
            "pre_speed_m_s": pre_speed,
            "post_speed_m_s": post_speed,
            "speed_change_%": ((post_speed - pre_speed) / pre_speed) * 100,
        }
    )

speed_df = pd.DataFrame(speed_records)

# Paired t-test on running speed
speed_t_test = stats.ttest_rel(speed_df["pre_speed_m_s"], speed_df["post_speed_m_s"])
avg_speed_change = speed_df["speed_change_%"].mean()
std_speed_change = speed_df["speed_change_%"].std()

print("\n--- Running speed: pre vs. post ---")
print(f"Average speed {direction_phrase(avg_speed_change, decimals=3)} ± {std_speed_change:.3f}%")
print(f"Paired t-test: t={speed_t_test.statistic:.3f}, p={speed_t_test.pvalue:.3f}")
# print step length
print(f"Average step length {direction_phrase(speed_df['distance_change_m'].mean(), decimals=3)} ± {speed_df['distance_change_m'].std():.3f} m")

# %%
# --- Stage 4 continued: Spatio-temporal statistics LaTeX table ---

# Load contact time using dedicated method (also reads from master_results.csv via GRF files)
contact_time_df = analysis_utils.get_contact_time_df()
contact_time_df["subject_id"] = contact_time_df["subject_id"].astype(int)

# All other metrics come from speed_df which is already built from prep_pre_df/prep_post_df
# (master_results.csv — consistent with all other analyses in this script)
spatiotemporal_df = speed_df[
    ["subject_id", "pre_duration", "post_duration", "pre_distance_m", "post_distance_m", "pre_speed_m_s", "post_speed_m_s"]
].merge(contact_time_df[["subject_id", "contact_time_pre", "contact_time_post"]], on="subject_id", how="inner")

# from pathlib import Path

# spatiotemporal_df.to_csv(Path("data/spatiotemporal_stats.csv"), index=False)

def select_paired_test(pre, post):
    """Paired t-test if delta is normal (Shapiro-Wilk α=0.05), otherwise Wilcoxon.
    Returns (pvalue, is_normal)."""
    delta = post - pre
    _, shapiro_p = stats.shapiro(delta)
    normal = shapiro_p > 0.05
    if normal:
        result = stats.ttest_rel(pre, post)
    else:
        result = stats.wilcoxon(pre, post)
    return result.pvalue, normal


def _fmt_central(s, normal: bool, decimals: int = 3) -> str:
    """Mean ± SD when normal, median ± IQR when non-normal."""
    if normal:
        return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"
    q1, q3 = np.percentile(s, [25, 75])
    return f"{np.median(s):.{decimals}f} ± {(q3 - q1):.{decimals}f}"


# Variable definitions: display name → (pre_col, post_col)
spatiotemporal_variables = [
    ("Sprint duration (s)", "pre_duration", "post_duration"),
    ("Step Length (m)", "pre_distance_m", "post_distance_m"),
    ("Running speed (m/s)", "pre_speed_m_s", "post_speed_m_s"),
    ("Contact time (s)", "contact_time_pre", "contact_time_post"),
]

print("\n--- Spatio-temporal descriptive stats ---")
spatiotemporal_rows = []
for var_name, pre_col, post_col in spatiotemporal_variables:
    pre = spatiotemporal_df[pre_col]
    post = spatiotemporal_df[post_col]
    delta = post - pre

    p, normal = select_paired_test(pre, post)
    sig = p < 0.05

    print(
        f"{var_name}: pre={pre.mean():.3f}±{pre.std():.3f}, post={post.mean():.3f}±{post.std():.3f}, "
        f"delta={delta.mean():.3f}±{delta.std():.3f}, p={p:.3f}{' *' if sig else ''}"
        f"{' [Wilcoxon]' if not normal else ''}"
    )
    spatiotemporal_rows.append((var_name, pre, post, delta, sig, normal))

# Build LaTeX table
latex_body_rows = []
for var_name, pre, post, delta, sig, normal in spatiotemporal_rows:
    pre_str = _fmt_central(pre, normal)
    post_str = _fmt_central(post, normal)
    delta_str = _fmt_central(delta, normal)
    if sig:
        delta_str += "$^*$"
    if not normal:
        delta_str += r"$^\dagger$"
    latex_body_rows.append(f"    {var_name} & {pre_str} & {post_str} & {delta_str} \\\\")

latex_table = "\n".join([
    "\\begin{table}",
    "    \\centering",
    "    \\begin{tabular}{cccc}",
    "    \\hline \\hline",
    "    Variable & Pre & Post & $\\Delta$ \\\\ ",
    "    \\hline",
    *latex_body_rows,
    "    \\hline",
    "    \\end{tabular}",
    "    \\caption{Pre- and post spatio-temporal statistics following a RS-protocoll (mean $\\pm$ SD). "
    "Normality was assessed with the Shapiro-Wilk test ($\\alpha = 0.05$), and parametric or non-parametric tests "
    "were selected accordingly; $\\Delta$ = post - pre; $^*p < 0.05$; $^\\dagger$non-parametric (Wilcoxon signed-rank), values are median $\\pm$ IQR.}",
    "    \\label{tab:Spatio-temporal stats}",
    "\\end{table}",
])

print("\n--- LaTeX table (spatio-temporal stats) ---")
print(latex_table)

# %%
# =============================================================================
# STAGE 5: RESULTS PARAGRAPH
# =============================================================================
# Assembles a written academic-style results paragraph from all computed values.
# The SPM-derived significance (joint angles/moments/forces) is hardcoded,
# as it was determined from 4x3 Bonferroni-corrected SPM plots (threshold p<0.1).
# All other values are derived dynamically from the data above.

N_TOTAL = 43  # Total number of subjects


# --- Significance label based on whole-group t-test p-values ---
def sig_label(p):
    if p < 0.01:
        return "p<0.01"
    elif p < 0.05:
        return "p<0.05"
    else:
        return f"p={p:.3f}"


# --- Responder counts and mean % change among responders ---
# Responders are identified externally and evaluated here using signed % change
bflh_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["bflh"])]
bflh_resp_pct = (
    (bflh_resp_df[muscle_norm_fiber_length_names[1]] - bflh_resp_df[muscle_norm_fiber_length_names[0]])
    / bflh_resp_df[muscle_norm_fiber_length_names[0]]
) * 100
bflh_n, bflh_pct_total = len(bflh_resp_df), (len(bflh_resp_df) / N_TOTAL) * 100
bflh_r_mean, bflh_r_std = bflh_resp_pct.mean(), bflh_resp_pct.std()

st_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["semiten"])]
st_resp_pct = (
    (st_resp_df[muscle_norm_fiber_length_names[4]] - st_resp_df[muscle_norm_fiber_length_names[3]])
    / st_resp_df[muscle_norm_fiber_length_names[3]]
) * 100
st_n, st_pct_total = len(st_resp_df), (len(st_resp_df) / N_TOTAL) * 100
st_r_mean, st_r_std = st_resp_pct.mean(), st_resp_pct.std()

sm_resp_df = norm_fiber_length_df[norm_fiber_length_df["subject_id"].isin(responders_id["semimem"])]
sm_resp_pct = (
    (sm_resp_df[muscle_norm_fiber_length_names[7]] - sm_resp_df[muscle_norm_fiber_length_names[6]])
    / sm_resp_df[muscle_norm_fiber_length_names[6]]
) * 100
sm_n, sm_pct_total = len(sm_resp_df), (len(sm_resp_df) / N_TOTAL) * 100
sm_r_mean, sm_r_std = sm_resp_pct.mean(), sm_resp_pct.std()

# --- Assemble paragraph ---
# Sentence 1: SPM-derived — hardcoded significance, dynamic speed values
spm_sentence = (
    f"After repeated sprints, we observed that running speed {direction_phrase(avg_speed_change, decimals=0)} "
    f"(p<0.01, \u00b1{std_speed_change:.0f}%), magnitude of sagittal plane "
    f"joint angles and moments, as well as semimembranosus muscle force, but no significant "
    f"change in forces of bflh and semitendinosus."
)

# Sentence 2: Whole-group peak normalized fiber length change and significance
norm_fiber_length_sentence = (
    f"Peak normalized fiber length significantly changed ({sig_label(bflh_t_all.pvalue)}) for bflh "
    f"({pct_bflh_mean:+.0f}%\u00b1{pct_bflh_std:.0f}%) and semitendinosus "
    f"({pct_st_mean:+.0f}%\u00b1{pct_st_std:.0f}%), but not semimembranosus "
    f"({pct_sm_mean:+.0f}%\u00b1{pct_sm_std:.0f}%)."
)

# Sentence 3: Responder counts and magnitude of change
responder_sentence = (
    f"Peak normalized fiber length changed for {bflh_n}({bflh_pct_total:.0f}%), {st_n}({st_pct_total:.0f}%), "
    f"and {sm_n}({sm_pct_total:.0f}%) participants in their bflh ({bflh_r_mean:.0f}%\u00b1{bflh_r_std:.0f}%), "
    f"semitendinosus ({st_r_mean:.0f}%\u00b1{st_r_std:.0f}%), "
    f"and semimembranosus ({sm_r_mean:.0f}%\u00b1{sm_r_std:.0f}%), respectively."
)

print(f"{spm_sentence} {norm_fiber_length_sentence} {responder_sentence}")
# %%
