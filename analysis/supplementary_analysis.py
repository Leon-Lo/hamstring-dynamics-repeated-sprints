"""
Quick script to add stats, tables, and figures to the thesis, requested by reviewers.

=======================================
"""

# %%
# %% Setup - Add project root to path
import os
import sys
from pathlib import Path


# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)


import pandas as pd  # noqa: E402
from scipy.stats import shapiro, t as t_dist, ttest_rel, wilcoxon  # noqa: E402

from utils import Analysis_utils, Subject_utils  # noqa: E402

analysis_utils = Analysis_utils()
subject_utils = Subject_utils()


# %%
# Which subjects have 1 or 2 baseline trials?
df = pd.read_csv("data/master_results.csv")
baseline_counts = df[df["trial"].str.contains("Run_baseline")].groupby("subject_id")["trial"].nunique()

# table for each subject, how many baseline trials they have
baseline_table = pd.DataFrame({
    "subject_id": baseline_counts.index,
    "num_baseline_trials": baseline_counts.values
})

save_path = "data/baseline_counts.tex"
baseline_table.to_latex(save_path, index=False)


# %%
# Analyse if statistic diff between peak muscle length timing pre vs post

muscles = ["bflh", "semiten", "semimem"]

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()

subject_ids = sorted(pre_df["subject_id"].unique())


def peak_timing(df: pd.DataFrame, muscle: str) -> pd.Series:
    """Return gait-cycle index (0-100) of peak fiber length per subject."""
    col = f"{muscle}_fiber_lengths"
    return df.groupby("subject_id")[col].apply(lambda x: x.idxmax()).map(
        df.set_index(df.index)["index"]
        if "index" in df.columns
        else df.groupby("subject_id").apply(lambda g: g.loc[g[col].idxmax(), "index"])
    )


# Build timing using idxmax on the full frame (preserves the 'index' column)
def get_peak_timing_series(df: pd.DataFrame, muscle: str) -> pd.Series:
    """Return a Series (subject_id → gait-cycle %) of peak fiber length timing."""
    col = f"{muscle}_fiber_lengths"
    # For each subject get the row where fiber_length is maximum, then extract 'index'
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
        stat_test, p_test = _res.statistic, _res.pvalue  # type: ignore[attr-defined]
        test_name = "Paired t-test"
        test_label = "t"
    else:
        _res = wilcoxon(pre_c, post_c)
        stat_test, p_test = _res.statistic, _res.pvalue  # type: ignore[attr-defined]
        test_name = "Wilcoxon"
        test_label = "W"

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

    timing_rows.append({
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
    })

timing_summary_df = pd.DataFrame(timing_rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(timing_summary_df.to_string(index=False))

# %%
# Muscle biomechanics summary table: impulse, avg/peak EMG, peak fiber length (pre vs post)

import numpy as np  # noqa: E402

METRICS = {
    "Impulse (N·s)": lambda grp, m: np.trapz(grp[f"{m}_fiber_force"], grp["time"]),
    "Avg EMG": lambda grp, m: grp[f"{m}_adjusted_emg"].mean(),
    "Peak EMG": lambda grp, m: grp[f"{m}_adjusted_emg"].max(),
    "Peak norm. fiber length (l/l₀)": lambda grp, m: grp[f"{m}_fiber_lengths"].max(),
}


def compute_metric_per_subject(df: pd.DataFrame, muscle: str, fn) -> pd.Series:
    result = {}
    for subj, grp in df.groupby("subject_id"):
        result[subj] = fn(grp, muscle)
    return pd.Series(result)


def bio_summary_row(label: str, pre: pd.Series, post: pd.Series) -> dict:
    change = (post - pre).dropna()
    pre_c = pre.loc[change.index]
    post_c = post.loc[change.index]
    normal = shapiro(change).pvalue > 0.05

    if normal:
        p_test = ttest_rel(pre_c, post_c).pvalue  # type: ignore[attr-defined]

        def fmt(s: pd.Series) -> str:
            return f"{s.mean():.3f} ± {s.std():.3f}"

    else:
        p_test = wilcoxon(pre_c, post_c).pvalue  # type: ignore[attr-defined]

        def fmt(s: pd.Series) -> str:
            q1, q3 = np.percentile(s, [25, 75])
            return f"{np.median(s):.3f} ± {(q3 - q1):.3f}"

    return {
        "Variable": label,
        "Pre": fmt(pre_c),
        "Post": fmt(post_c),
        "Change": fmt(change),
        "sig": p_test < 0.05,
        "normal": normal,
    }


bio_rows = []
for muscle in muscles:
    for metric_name, metric_fn in METRICS.items():
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        muscle_label = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}.get(muscle, muscle.upper())
        label = f"{muscle_label} -- {metric_name}"
        bio_rows.append(bio_summary_row(label, pre_vals, post_vals))

print(r"\begin{table}")
print(r"    \centering")
print(r"    \begin{tabular}{cccc}")
print(r"    \hline \hline")
print(r"    Variable & Pre & Post & $\Delta$ \\")
print(r"    \hline")
for row in bio_rows:
    delta = row["Change"]
    if row["sig"]:
        delta += "$^*$"
    if not row["normal"]:
        delta += r"$^\dagger$"
    print(f"    {row['Variable']} & {row['Pre']} & {row['Post']} & {delta} \\\\")
print(r"    \hline")
print(r"    \end{tabular}")
print(r"    \caption{Pre- and post-intervention muscle biomechanics (mean $\pm$ SD). Normality was assessed with the Shapiro-Wilk test ($\alpha = 0.05$); $\Delta$ = post $-$ pre; $^*p < 0.05$; $^\dagger$non-parametric (Wilcoxon signed-rank), values are median $\pm$ IQR.}")
print(r"    \label{tab:muscle-biomechanics}")
print(r"\end{table}")

# %%
# Paired pre/post plots – one figure, 4 rows (metrics) × 3 columns (muscles)

import matplotlib.pyplot as plt  # noqa: E402

MUSCLE_LABELS = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
METRIC_NAMES = list(METRICS.keys())

fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)

for j, (metric_name, metric_fn) in enumerate(METRICS.items()):
    for i, muscle in enumerate(muscles):
        ax = axes[j, i]
        row = bio_rows[i * len(METRICS) + j]

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
        if row["normal"]:
            t_crit = t_dist.ppf(0.975, df=n - 1)
            pre_center = pre_plot.mean()
            post_center = post_plot.mean()
            pre_err = float(t_crit * pre_plot.std() / np.sqrt(n))
            post_err = float(t_crit * post_plot.std() / np.sqrt(n))
        else:
            pre_center = float(np.median(pre_plot))
            post_center = float(np.median(post_plot))
            pre_err = float(np.percentile(pre_plot, 75) - np.percentile(pre_plot, 25))
            post_err = float(np.percentile(post_plot, 75) - np.percentile(post_plot, 25))

        ax.errorbar(
            [0, 1], [pre_center, post_center], yerr=[pre_err, post_err],
            fmt="D", color="black", capsize=4, zorder=4, markersize=7,
            mfc="white", mec="black",
        )

        if "EMG" in metric_name:
            ax.set_ylim(bottom=0)

        if row["sig"]:
            ax.text(1, post_center + post_err, "*", ha="center", va="bottom", fontsize=13, color="black")

        if j == 0:
            ax.set_title(MUSCLE_LABELS[muscle])
        if i == 0:
            ax.set_ylabel(metric_name)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_xlim(-0.5, 1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

plt.savefig("data/plot_results/biomechanics_pre_post.png", dpi=300, bbox_inches="tight")
plt.show()


print(
    r"\caption{Pre- and post-intervention paired comparisons for biceps femoris long head (BFLH), "
    r"semitendinosus (ST), and semimembranosus (SM). Each row shows one biomechanical variable: "
    r"muscle impulse (integral of fiber force over time, N$\cdot$s), mean adjusted EMG, "
    r"peak adjusted EMG, and peak normalized fiber length ($l/l_0$). "
    r"Blue and orange dots represent individual subject values pre- and post-intervention, "
    r"respectively; grey lines connect paired observations. "
    r"Open diamonds indicate the group mean $\pm$ 95\% CI, or median $\pm$ IQR for non-normally "
    r"distributed data (Shapiro-Wilk, $\alpha = 0.05$). $^*p < 0.05$ (paired $t$-test or "
    r"Wilcoxon signed-rank test).}"
)

# %%
