"""
Quick script to add stats, tables, and figures to the thesis, requested by reviewers.

=======================================
"""

# %% [markdown]
# ## Setup
# Adds the project root to `sys.path` and loads shared utilities.

# %%
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


# %% [markdown]
# ## Baseline trial counts
# Counts how many baseline trials each subject has and saves the result as a LaTeX table.
# Used to flag subjects with only one baseline trial in the thesis.

# %%
# Which subjects have 1 or 2 baseline trials?
df = pd.read_csv("data/master_results.csv")
baseline_counts = df[df["trial"].str.contains("Run_baseline")].groupby("subject_id")["trial"].nunique()

# table for each subject, how many baseline trials they have
baseline_table = pd.DataFrame({"subject_id": baseline_counts.index, "num_baseline_trials": baseline_counts.values})

save_path = "data/tables/baseline_counts.tex"
baseline_table.to_latex(save_path, index=False)


# %% [markdown]
# ## Peak fibre length timing — pre vs post
# Tests whether the gait-cycle percentage at which peak fibre length occurs shifted
# after the intervention. One paired t-test (or Wilcoxon if non-normal) per muscle,
# no correction applied here since these are separate outcome variables.

# %%
# Analyse if statistic diff between peak muscle length timing pre vs post

muscles = ["bflh", "semiten", "semimem"]

pre_df = analysis_utils.prep_pre_df()
post_df = analysis_utils.prep_post_df()

subject_ids = sorted(pre_df["subject_id"].unique())


def peak_timing(df: pd.DataFrame, muscle: str) -> pd.Series:
    """Return gait-cycle index (0-100) of peak fiber length per subject."""
    col = f"{muscle}_fiber_lengths"
    return (
        df.groupby("subject_id")[col]
        .apply(lambda x: x.idxmax())
        .map(
            df.set_index(df.index)["index"]
            if "index" in df.columns
            else df.groupby("subject_id").apply(lambda g: g.loc[g[col].idxmax(), "index"])
        )
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
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(timing_summary_df.to_string(index=False))

# %% [markdown]
# ## Biomechanics summary — compute per-subject values
# Computes four metrics per subject per muscle (impulse, avg EMG, peak EMG, peak normalised
# fibre length). Stores raw p-values for use in post-hoc testing after the RM ANOVA below.
# Significance flags (`sig`) are set later once the interaction p-values are known.

# %%
# Muscle biomechanics summary table: impulse, avg/peak EMG, peak fiber length (pre vs post)

import numpy as np  # noqa: E402


_MIN_REGION_LEN = 5  # ignore negative-power regions shorter than this many samples


def _eccentric_mask(grp: pd.DataFrame) -> np.ndarray:
    """Boolean mask for the second negative knee joint-power phase.

    Joint power = d(knee_angle)/dt × knee_moment. Negative power = energy
    absorption = eccentric. The second contiguous region (after filtering out
    spurious short regions < _MIN_REGION_LEN samples) is the late-swing
    hamstring eccentric window. Returns an all-False mask if fewer than two
    qualifying regions exist.
    """
    time = np.asarray(grp["time"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), time)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)

    neg = (power < 0).astype(int)
    diff = np.diff(neg, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # drop regions shorter than the minimum length
    valid = [(s, e) for s, e in zip(starts, ends) if (e - s) >= _MIN_REGION_LEN]

    mask = np.zeros(len(time), dtype=bool)
    if len(valid) >= 2:
        s, e = valid[1]
        mask[s:e] = True
    return mask


METRICS = {
    "Eccentric Impulse (N·s)": lambda grp, m: np.trapezoid(
        np.where(_eccentric_mask(grp), np.asarray(grp[f"{m}_fiber_force"], dtype=float), 0),
        np.asarray(grp["time"], dtype=float),
    ),
    "Eccentric Knee Moment (N·m·s)": lambda grp, m: np.trapezoid(
        np.where(
            _eccentric_mask(grp),
            np.asarray(grp[f"{m}_fiber_force"], dtype=float) * np.asarray(grp[f"{m}_knee_ma"], dtype=float),
            0,
        ),
        np.asarray(grp["time"], dtype=float),
    ),
    "Avg EMG": lambda grp, m: np.asarray(grp[f"{m}_adjusted_emg"], dtype=float)[_eccentric_mask(grp)].mean(),
    "Peak EMG": lambda grp, m: np.asarray(grp[f"{m}_adjusted_emg"], dtype=float)[_eccentric_mask(grp)].max(),
    "Peak norm. fiber length ($l/l_0$)": lambda grp, m: np.asarray(grp[f"{m}_fiber_lengths"], dtype=float)[
        _eccentric_mask(grp)
    ].max(),
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

    def fmt(s: pd.Series) -> str:
        return f"{s.mean():.3f} ± {s.std():.3f}"

    return {
        "Variable": label,
        "Pre": fmt(pre_c),
        "Post": fmt(post_c),
        "Change": fmt(change),
    }


bio_rows = []
for muscle in muscles:
    for metric_name, metric_fn in METRICS.items():
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        muscle_label = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}.get(muscle, muscle.upper())
        label = f"{muscle_label} -- {metric_name}"
        row = bio_summary_row(label, pre_vals, post_vals)
        row["metric"] = metric_name
        bio_rows.append(row)

# %% [markdown]
# ## Two-way repeated measures ANOVA
# Primary statistical test. For each metric, a 2×3 RM ANOVA (time × muscle, both within-subjects)
# tests whether the intervention affected muscles differently (time × muscle interaction).
# - Sphericity checked via Mauchly's test; Greenhouse-Geisser correction applied where `eps < 1`.
# - The interaction p-value gates post-hoc testing: paired comparisons are only interpreted
#   for metrics where the interaction is significant.
# - Post-hoc Bonferroni: α = 0.05 / 3 muscles ≈ 0.017 per metric.

# %%
# Two-way repeated measures ANOVA: time (pre/post) × muscle (BFLH/ST/SM) per metric.

import pingouin as pg  # noqa: E402

MUSCLE_MAP = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
EFFECT_LABELS = {
    "time": "Time",
    "muscle": "Muscle",
    "time * muscle": r"Time $\times$ Muscle",
}
alpha_posthoc = 0.05 / len(muscles)  # Bonferroni for 3 post-hoc comparisons per metric


def _row_pval(effect_row: pd.Series) -> tuple[float, bool]:
    """Return (p_value, gg_corrected) from a pingouin RM ANOVA row robustly."""
    eps = effect_row.get("eps", 1.0)
    gg_corrected = eps is not None and float(eps) < 1.0
    p_gg = effect_row.get("p_GG_corr", float("nan"))
    p_unc = effect_row.get("p_unc", float("nan"))
    if gg_corrected and not pd.isna(p_gg):
        return float(p_gg), True
    return float(p_unc), False


rm_anova_results = {}  # metric_name -> pingouin DataFrame
interaction_sig_by_metric = {}  # metric_name -> bool
posthoc_pvals = {}  # (metric_name, muscle) -> Bonferroni-corrected p

for j, (metric_name, metric_fn) in enumerate(METRICS.items()):
    rows_long = []
    for muscle in muscles:
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        for subj in subject_ids:
            rows_long.append({"subject": subj, "time": "pre", "muscle": muscle, "value": pre_vals.get(subj)})
            rows_long.append({"subject": subj, "time": "post", "muscle": muscle, "value": post_vals.get(subj)})

    long_df = pd.DataFrame(rows_long)
    # retain only subjects with complete data across all 6 cells
    complete = long_df.groupby("subject")["value"].apply(lambda x: x.notna().all())
    long_df = long_df[long_df["subject"].isin(complete[complete].index)].dropna()

    aov = pg.rm_anova(data=long_df, dv="value", within=["time", "muscle"], subject="subject", correction=True)
    rm_anova_results[metric_name] = aov

    interaction_row = aov[aov["Source"] == "time * muscle"].iloc[0]
    p_interaction, _ = _row_pval(interaction_row)
    interaction_sig = p_interaction < 0.05
    interaction_sig_by_metric[metric_name] = interaction_sig

    # Post-hoc: parametric paired t-test (pre vs post) per muscle, Bonferroni-corrected for
    # 3 comparisons. Uses pg.pairwise_tests to stay within the pingouin RM framework rather
    # than running separate independent tests.
    for i, muscle in enumerate(muscles):
        muscle_df = long_df[long_df["muscle"] == muscle]
        ph = pg.pairwise_tests(data=muscle_df, dv="value", within="time", subject="subject", parametric=True)
        p_bonf = min(float(ph["p_unc"].iloc[0]) * len(muscles), 1.0)
        posthoc_pvals[(metric_name, muscle)] = p_bonf
        bio_rows[i * len(METRICS) + j]["sig"] = interaction_sig and p_bonf < 0.05
        bio_rows[i * len(METRICS) + j]["p_bonf"] = p_bonf

# %% [markdown]
# ## LaTeX table — RM ANOVA (primary)
# Prints F, df, p, and η²_g for all three effects (time, muscle, time × muscle) per metric.
# GG-corrected p-values are used where sphericity was violated (marked ‡).

# %%
# RM ANOVA table
Path("data/tables").mkdir(parents=True, exist_ok=True)

lines = []
lines.append(r"\begin{table}")
lines.append(r"    \centering")
lines.append(r"    \resizebox{\textwidth}{!}{%")
lines.append(r"    \begin{tabular}{llcccc}")
lines.append(r"    \hline \hline")
lines.append(r"    Metric & Effect & $F$ & $df_1$, $df_2$ & $p$ & $\eta^2_g$ \\")
lines.append(r"    \hline")

for metric_name, aov in rm_anova_results.items():
    first = True
    for _, effect_row in aov.iterrows():
        source = effect_row["Source"]
        effect_label = EFFECT_LABELS.get(source, source)

        F = effect_row["F"]
        df1 = int(effect_row["ddof1"])
        df2 = int(effect_row["ddof2"])
        eta2 = effect_row.get("ng2", float("nan"))

        p_val, gg_corrected = _row_pval(effect_row)
        corr_flag = r"$^\ddagger$" if gg_corrected else ""

        sig_flag = r"$^*$" if p_val < 0.05 else ""
        p_str = (r"$<$0.001" if p_val < 0.001 else f"{p_val:.3f}") + sig_flag + corr_flag
        eta_str = f"{eta2:.3f}" if not pd.isna(eta2) else "—"

        prefix = metric_name if first else ""
        first = False
        lines.append(f"    {prefix} & {effect_label} & {F:.2f} & {df1}, {df2} & {p_str} & {eta_str} \\\\")
    lines.append(r"    \hline")

lines.append(r"    \end{tabular}}")
lines.append(
    r"    \caption{Two-way repeated measures ANOVA (time $\times$ muscle) for each biomechanical metric. "
    r"All metrics were computed within the second negative knee joint-power phase "
    r"(eccentric window: $\dot{\theta}_\text{knee} \times M_\text{knee} < 0$). "
    r"Sphericity was assessed with Mauchly's test; $^\ddagger$Greenhouse-Geisser correction applied where "
    r"violated. $\eta^2_g$ = generalised eta-squared. $^*p < 0.05$.}"
)
lines.append(r"    \label{tab:rm-anova}")
lines.append(r"\end{table}")

Path("data/tables/rm_anova.tex").write_text("\n".join(lines))
print("Saved data/tables/rm_anova.tex")

# %% [markdown]
# ## LaTeX table — post-hoc paired comparisons
# Prints pre/post values and Δ per muscle per metric. Asterisks only appear where
# the RM ANOVA interaction was significant **and** the paired comparison survives
# Bonferroni correction (α = 0.05 / 3 ≈ 0.017).

# %%
# Post-hoc paired comparisons table (shown only where interaction was significant)
lines = []
lines.append(r"\begin{table}")
lines.append(r"    \centering")
lines.append(r"    \resizebox{\textwidth}{!}{%")
lines.append(r"    \begin{tabular}{lccc}")
lines.append(r"    \hline \hline")
lines.append(r"    Variable & Pre & Post & $\Delta$ \\")
lines.append(r"    \hline")
for row in bio_rows:
    if not interaction_sig_by_metric.get(row["metric"], False):
        continue
    delta = row["Change"]
    if row["sig"]:
        delta += "$^*$"
    lines.append(f"    {row['Variable']} & {row['Pre']} & {row['Post']} & {delta} \\\\")
lines.append(r"    \hline")
lines.append(r"    \end{tabular}}")
lines.append(
    r"    \caption{Post-hoc paired comparisons per muscle and metric (mean $\pm$ SD). "
    r"All metrics were computed within the second negative knee joint-power phase (eccentric window). "
    r"Reported where the time $\times$ muscle interaction was significant (Table~\ref{tab:rm-anova}). "
    r"Post-hoc tests are parametric paired $t$-tests, consistent with the RM ANOVA framework. "
    r"$^*p < 0.05$ after Bonferroni correction for " + str(len(muscles)) + r" comparisons per metric "
    r"($\alpha_{\text{corr}} = 0.05/" + str(len(muscles)) + r" \approx " + f"{alpha_posthoc:.3f}" + r"$).}"
)
lines.append(r"    \label{tab:muscle-biomechanics}")
lines.append(r"\end{table}")

Path("data/tables/posthoc_comparisons.tex").write_text("\n".join(lines))
print("Saved data/tables/posthoc_comparisons.tex")

# %% [markdown]
# ## Paired pre/post plots
# 4 rows (metrics) × 3 columns (muscles). Each subplot shows individual subject trajectories
# (grey lines), pre/post scatter, and group mean ± 95% CI (or median ± IQR if non-normal).
# Asterisks follow the same significance rule as the post-hoc table above.

# %%
# Paired pre/post plots – one figure, 4 rows (metrics) × 3 columns (muscles)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MUSCLE_LABELS = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
METRIC_NAMES = list(METRICS.keys())

fig, axes = plt.subplots(len(METRICS), len(muscles), figsize=(10, 3 * len(METRICS)))

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
        t_crit = t_dist.ppf(0.975, df=n - 1)
        pre_center = float(pre_plot.mean())
        post_center = float(post_plot.mean())
        pre_err = float(t_crit * pre_plot.std() / np.sqrt(n))
        post_err = float(t_crit * post_plot.std() / np.sqrt(n))

        ax.errorbar(
            [0, 1],
            [pre_center, post_center],
            yerr=[pre_err, post_err],
            fmt="D",
            color="black",
            capsize=4,
            zorder=4,
            markersize=7,
            mfc="white",
            mec="black",
        )

        if "EMG" in metric_name:
            ax.set_ylim(bottom=0)

        if row["sig"]:
            ax.text(1, post_center + post_err, "*", ha="center", va="bottom", fontsize=13, color="black")

        if i == 0:
            ax.set_ylabel(metric_name)

        # muscle abbreviation as x-axis label on bottom row only
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
    [0], [0], yerr=[1], fmt="D", color="black", capsize=4, markersize=7, mfc="white", mec="black", label="Mean ± 95% CI"
)
plt.close(_tmp_fig)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=7, label="Pre"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=7, label="Post"),
    _eb_handle,
]
fig.legend(handles=legend_elements, loc="upper right", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(1.0, 1.0))

plt.savefig("data/plot_results/Figure3.png", dpi=300, bbox_inches="tight")
plt.close(fig)


_fig_caption = "\n".join(
    [
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=0.8\linewidth]{images/Figure3.png}",
        r"    \caption{Pre- and post-intervention paired comparisons for BFLH, SM, and ST across",
        r"    five biomechanical variables (rows), all computed within the eccentric window",
        r"    ($\dot{\theta}_\text{knee} \times M_\text{knee} < 0$): eccentric muscle impulse,",
        r"    eccentric knee moment contribution, mean and peak adjusted EMG, and peak normalised",
        r"    fibre length ($l/l_0$). Blue/orange dots are individual subject values",
        r"    pre-/post-intervention; grey lines connect pairs. Open diamonds show group mean",
        r"    $\pm$ 95\% CI. $^*p < 0.05$ after Bonferroni correction for three muscles.}",
        r"    \label{fig:biomechanics-pre-post}",
        r"\end{figure}",
    ]
)
Path("data/tables/biomechanics_fig_caption.tex").write_text(_fig_caption)
print("Saved data/tables/biomechanics_fig_caption.tex")

# %% [markdown]
# ## Sanity check — eccentric window per subject
# For each subject, plots knee joint power (pre baseline) with the second negative-power
# phase shaded in red. Allows visual verification that the eccentric window is correctly
# identified before trusting the metrics.

# %%
n_cols = 9
n_rows = (len(subject_ids) + n_cols - 1) // n_cols
fig_chk, axes_chk = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), constrained_layout=True)
axes_chk_flat = axes_chk.flatten()

MUSCLE_COLORS = {"bflh": "#e15759", "semiten": "#f28e2b", "semimem": "#59a14f"}
MUSCLE_ABBREV = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}

for ax, subj in zip(axes_chk_flat, subject_ids):
    grp = pre_df[pre_df["subject_id"] == subj]

    time = np.asarray(grp["time"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), time)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)
    mask = _eccentric_mask(grp)
    idx = grp["index"].values

    ax.plot(idx, power, color="steelblue", lw=1.2, label="knee power")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.fill_between(idx, power, 0, where=mask, color="steelblue", alpha=0.25)

    ax2 = ax.twinx()
    for muscle, color in MUSCLE_COLORS.items():
        force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
        ax2.plot(idx, force, color=color, lw=1.0, alpha=0.8, label=muscle.upper())
    ax2.tick_params(labelsize=5, colors="grey")
    ax2.spines["top"].set_visible(False)

    ax.set_title(f"{subj}", fontsize=8)
    ax.set_xlabel("Gait cycle (%)", fontsize=6)
    ax.set_ylabel("Knee joint power (W)", fontsize=6, color="steelblue")
    ax2.set_ylabel("Muscle fibre force (N)", fontsize=6, color="grey")
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes_chk_flat[len(subject_ids) :]:
    ax.set_visible(False)

legend_handles = [
    Line2D([0], [0], color="steelblue", lw=1.5, label="Knee power"),
    Line2D([0], [0], color="steelblue", lw=6, alpha=0.25, label="Eccentric window"),
] + [Line2D([0], [0], color=c, lw=1.5, label=MUSCLE_ABBREV[m]) for m, c in MUSCLE_COLORS.items()]
fig_chk.tight_layout(rect=[0, 0, 1, 0.97])
fig_chk.legend(
    handles=legend_handles,
    loc="upper right",
    ncol=len(legend_handles),
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(1.0, 1.0),
)
plt.savefig("data/plot_results/Figure4.png", dpi=150, bbox_inches="tight")
print("Saved data/plot_results/Figure4.png")

# %%
_sanity_caption = "\n".join(
    [
        r"\begin{figure}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{images/Figure4.png}",
        r"    \caption{Pre-intervention knee joint power (W, blue) over the normalised gait cycle (\%), with the",
        r"    eccentric window shaded. Right axis: muscle fibre forces (N) for BFLH (red), ST (orange),",
        r"    and SM (green).}",
        r"    \label{fig:eccentric-window-sanity}",
        r"\end{figure}",
    ]
)
Path("data/tables/eccentric_window_sanity_check_caption.tex").write_text(_sanity_caption)
print("Saved data/tables/eccentric_window_sanity_check_caption.tex")

# %% [markdown]
# ## Interactive peak normalised fiber length — eccentric window (last row only)
# Plotly scatter/line plot: each subject is a connecting line (Pre → Post).
# Hover shows subject ID. Green = decreased post, red = increased post.

# %%
import numpy as np  # noqa: E402, F811
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

_pre_df_int = analysis_utils.prep_pre_df()
_post_df_int = analysis_utils.prep_post_df()
_combined_int = pd.concat([_pre_df_int, _post_df_int], ignore_index=True)

_MUSCLES_INT = ["bflh", "semiten", "semimem"]
_MUSCLE_TITLES_INT = {
    "bflh": "M. Biceps Femoris Long Head",
    "semiten": "M. Semitendinosus",
    "semimem": "M. Semimembranosus",
}


def _eccentric_peak_int(grp: pd.DataFrame, col: str, min_len: int = 5) -> float:
    idx = np.asarray(grp["index"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), idx)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)
    neg = (power < 0).astype(int)
    diff = np.diff(neg, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    valid = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_len]
    mask = np.zeros(len(idx), dtype=bool)
    if len(valid) >= 2:
        s, e = valid[1]
        mask[s:e] = True
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
                x=["Pre", "Post"],
                y=[pre_val, post_val],
                mode="lines+markers",
                line=dict(color=color, width=1.5),
                marker=dict(size=7),
                name=f"Subject {subj}",
                legendgroup=f"subj_{subj}",
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>Subject {subj}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

fig_int.update_layout(
    title="Peak Normalised Fiber Length — Eccentric Window (Pre vs Post)",
    height=500,
    hovermode="closest",
    legend=dict(title="Subject", font=dict(size=10)),
)
for col_idx in range(1, 4):
    fig_int.update_yaxes(title_text="Peak norm. fiber length (fl/ofl)" if col_idx == 1 else "", row=1, col=col_idx)

fig_int.show()

# %%
