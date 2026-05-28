"""
Statistical analysis of sprint fatigue effects on hamstring eccentric mechanics.

Covers:
  - Biomechanics summary (pre/post per muscle per metric)
  - Two-way RM ANOVA: Time × Muscle per eccentric metric
  - One-way RM ANOVA: Time (pre/post) per metric per muscle (overall group)
  - ANCOVA on change scores: Δsprint speed as covariate
  - Post-hoc paired comparisons (Bonferroni-corrected)

All metrics are computed within the second negative knee joint-power phase
(eccentric window = late-swing hamstring lengthening during sprint acceleration).
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

_TABLE_DIR = Path("data/tables/fatigue_effects")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Biomechanics summary: pre/post per muscle per metric
# =============================================================================
MUSCLE_MAP = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}
EFFECT_LABELS = {
    "time": "Time",
    "muscle": "Muscle",
    "time * muscle": r"Time × Muscle",
}
alpha_posthoc = 0.05 / len(muscles)


def _row_pval(effect_row: pd.Series) -> tuple:
    eps = effect_row.get("eps", 1.0)
    gg_corrected = eps is not None and float(eps) < 1.0
    p_gg = effect_row.get("p_GG_corr", float("nan"))
    p_unc = effect_row.get("p_unc", float("nan"))
    if gg_corrected and not pd.isna(p_gg):
        return float(p_gg), True
    return float(p_unc), False


def bio_summary_row(label: str, pre: pd.Series, post: pd.Series) -> dict:
    change = (post - pre).dropna()
    pre_c = pre.loc[change.index]
    post_c = post.loc[change.index]

    def fmt(s: pd.Series) -> str:
        return f"{s.mean():.3f} ± {s.std():.3f}"

    return {"Variable": label, "Pre": fmt(pre_c), "Post": fmt(post_c), "Change": fmt(change)}


bio_rows = []
for muscle in muscles:
    for metric_name, metric_fn in METRICS.items():
        pre_vals = compute_metric_per_subject(pre_df, muscle, metric_fn).reindex(subject_ids)
        post_vals = compute_metric_per_subject(post_df, muscle, metric_fn).reindex(subject_ids)
        muscle_label = MUSCLE_MAP.get(muscle, muscle.upper())
        label = f"{muscle_label} -- {metric_name}"
        row = bio_summary_row(label, pre_vals, post_vals)
        row["metric"] = metric_name
        bio_rows.append(row)

# =============================================================================
# Two-way RM ANOVA: Time × Muscle per metric
# =============================================================================
rm_anova_results = {}
interaction_sig_by_metric = {}
posthoc_pvals = {}

for j, (metric_name, metric_fn) in enumerate(METRICS.items()):
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
    rm_anova_results[metric_name] = aov

    interaction_row = aov[aov["Source"] == "time * muscle"].iloc[0]
    p_interaction, _ = _row_pval(interaction_row)
    interaction_sig = p_interaction < 0.05
    interaction_sig_by_metric[metric_name] = interaction_sig

    for i, muscle in enumerate(muscles):
        muscle_df = long_df[long_df["muscle"] == muscle]
        ph = pg.pairwise_tests(data=muscle_df, dv="value", within="time", subject="subject", parametric=True)
        p_bonf = min(float(ph["p_unc"].iloc[0]) * len(muscles), 1.0)
        posthoc_pvals[(metric_name, muscle)] = p_bonf
        bio_rows[i * len(METRICS) + j]["sig"] = interaction_sig and p_bonf < 0.05
        bio_rows[i * len(METRICS) + j]["p_bonf"] = p_bonf

# ── LaTeX RM ANOVA table ──────────────────────────────────────────────────────
lines = [
    r"\begin{table}",
    r"    \centering",
    r"    \resizebox{\textwidth}{!}{%",
    r"    \begin{tabular}{llcccc}",
    r"    \hline \hline",
    r"    Metric & Effect & $F$ & $df_1$, $df_2$ & $p$ & $\eta^2_g$ \\",
    r"    \hline",
]
for metric_name, aov in rm_anova_results.items():
    first = True
    for _, effect_row in aov.iterrows():
        source = effect_row["Source"]
        effect_label = EFFECT_LABELS.get(source, source)
        F = effect_row["F"]
        df1, df2 = int(effect_row["ddof1"]), int(effect_row["ddof2"])
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
lines += [
    r"    \end{tabular}}",
    r"    \caption{Two-way repeated measures ANOVA (time $\times$ muscle) for each biomechanical metric. "
    r"All metrics were computed within the second negative knee joint-power phase. "
    r"Sphericity assessed with Mauchly's test; $^\ddagger$Greenhouse-Geisser correction applied. "
    r"$\eta^2_g$ = generalised eta-squared. $^*p < 0.05$.}",
    r"    \label{tab:rm-anova}",
    r"\end{table}",
]
(_TABLE_DIR / "rm_anova.tex").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {_TABLE_DIR / 'rm_anova.tex'}")

# ── Word RM ANOVA table ───────────────────────────────────────────────────────
_rmanova_word_rows = []
for _metric_name, _aov in rm_anova_results.items():
    _first = True
    for _, _effect_row in _aov.iterrows():
        _source = _effect_row["Source"]
        _effect_label = EFFECT_LABELS.get(_source, _source).replace(r"$\times$", "×")
        _F = _effect_row["F"]
        _df1, _df2 = int(_effect_row["ddof1"]), int(_effect_row["ddof2"])
        _eta2 = _effect_row.get("ng2", float("nan"))
        _p_val, _gg = _row_pval(_effect_row)
        _p_str = (
            ("<0.001" if _p_val < 0.001 else f"{_p_val:.3f}")
            + ("*" if _p_val < 0.05 else "")
            + ("‡" if _gg else "")
        )
        _rmanova_word_rows.append(
            {
                "Metric": _metric_name.replace(r"$l/l_0$", "l/l₀") if _first else "",
                "Effect": _effect_label,
                "F": f"{_F:.2f}",
                "df₁, df₂": f"{_df1}, {_df2}",
                "p": _p_str,
                "η²g": f"{_eta2:.3f}" if not pd.isna(_eta2) else "—",
            }
        )
        _first = False
_save_word_table(
    pd.DataFrame(_rmanova_word_rows),
    str(_TABLE_DIR / "rm_anova.docx"),
    title="Two-Way Repeated Measures ANOVA: Time × Muscle for Each Biomechanical Metric",
    note="All metrics computed within the eccentric window (dθ/dt_knee × M_knee < 0). "
    "‡ Greenhouse-Geisser correction. η²g = generalised eta-squared. * p < 0.05.",
)
print(f"Saved {_TABLE_DIR / 'rm_anova.docx'}")

# =============================================================================
# One-way RM ANOVA: Time per metric per muscle (overall group)
# =============================================================================
_ONEWAY_METRICS = {
    "Peak Eccentric Force (N)": lambda grp, m: np.asarray(grp[f"{m}_fiber_force"], dtype=float)[eccentric_mask(grp)].max(),
    "Eccentric Impulse (N·s)": METRICS["Eccentric Impulse (N·s)"],
    "Eccentric Work (J)": METRICS["Eccentric Work (J)"],
}
_ONEWAY_ALPHA = 0.05 / len(muscles)

_oneway_rows = []
for _metric_name, _metric_fn in _ONEWAY_METRICS.items():
    for _muscle in muscles:
        _pre_vals = compute_metric_per_subject(pre_df, _muscle, _metric_fn).reindex(subject_ids)
        _post_vals = compute_metric_per_subject(post_df, _muscle, _metric_fn).reindex(subject_ids)
        _muscle_long = []
        for _subj in subject_ids:
            if pd.notna(_pre_vals.get(_subj)) and pd.notna(_post_vals.get(_subj)):
                _muscle_long.append({"subject": _subj, "time": "pre", "value": float(_pre_vals[_subj])})
                _muscle_long.append({"subject": _subj, "time": "post", "value": float(_post_vals[_subj])})
        if len(_muscle_long) < 4:
            continue
        _mdf = pd.DataFrame(_muscle_long)
        _aov1 = pg.rm_anova(data=_mdf, dv="value", within="time", subject="subject", correction=True)
        _r = _aov1.iloc[0]
        _F, _df1, _df2 = float(_r["F"]), int(_r["ddof1"]), int(_r["ddof2"])
        _p = float(_r["p_unc"])
        _eta2p = (_F * _df1) / (_F * _df1 + _df2)
        _muscle_lbl = MUSCLE_MAP.get(_muscle, _muscle.upper())
        _oneway_rows.append(
            {
                "Metric": _metric_name,
                "Muscle": _muscle_lbl,
                "F": f"{_F:.2f}",
                "df": f"{_df1}, {_df2}",
                "p": ("<0.001" if _p < 0.001 else f"{_p:.3f}") + ("*" if _p < _ONEWAY_ALPHA else ""),
                "η²p": f"{_eta2p:.3f}",
            }
        )

_oneway_df = pd.DataFrame(_oneway_rows)
print(_oneway_df.to_string(index=False))
_save_word_table(
    _oneway_df,
    str(_TABLE_DIR / "rm_anova_time_only.docx"),
    title="Repeated Measures ANOVA: Time (pre/post) — Eccentric Metrics per Muscle (Overall Group)",
    note=f"One-way RM ANOVA with Time (pre/post) as the sole within-subject factor. "
    f"All metrics computed within the eccentric window. η²p = partial eta-squared. "
    f"* p < {_ONEWAY_ALPHA:.3f} (Bonferroni-corrected for 3 muscles).",
)
print(f"Saved {_TABLE_DIR / 'rm_anova_time_only.docx'}")

# =============================================================================
# ANCOVA on change scores — Δsprint speed as covariate
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


_ancova_rows = []
for _metric_name, _metric_fn in _ONEWAY_METRICS.items():
    for _muscle in muscles:
        _pre_v = compute_metric_per_subject(pre_df, _muscle, _metric_fn).reindex(subject_ids)
        _post_v = compute_metric_per_subject(post_df, _muscle, _metric_fn).reindex(subject_ids)
        _cs_rows = []
        for _subj in subject_ids:
            _pv, _ov = _pre_v.get(_subj), _post_v.get(_subj)
            if pd.notna(_pv) and pd.notna(_ov):
                _cs_rows.append({"subject": _subj, "delta": float(_ov) - float(_pv)})
        if len(_cs_rows) < 4:
            continue
        _cs_df = pd.DataFrame(_cs_rows).merge(_ancova_speed_df[["subject", "speed_c"]], on="subject")
        try:
            _fit = smf.ols("delta ~ speed_c", data=_cs_df).fit()
            _t_int, _p_int = float(_fit.tvalues["Intercept"]), float(_fit.pvalues["Intercept"])
            _t_spd, _p_spd = float(_fit.tvalues["speed_c"]), float(_fit.pvalues["speed_c"])
        except Exception as _exc:
            print(f"  ANCOVA failed {_metric_name} / {_muscle}: {_exc}")
            _t_int = _p_int = _t_spd = _p_spd = float("nan")

        _ti, _pi = _fmt_tp(_t_int, _p_int, _ancova_alpha)
        _ts, _ps = _fmt_tp(_t_spd, _p_spd, 0.05)
        _muscle_lbl = MUSCLE_MAP.get(_muscle, _muscle.upper())
        _ancova_rows.append(
            {
                "Metric": _metric_name,
                "Muscle": _muscle_lbl,
                "Intercept t": _ti,
                "Intercept p": _pi,
                "Δspeed t": _ts,
                "Δspeed p": _ps,
            }
        )

_ancova_df = pd.DataFrame(_ancova_rows)
print(_ancova_df.to_string(index=False))
_save_word_table(
    _ancova_df,
    str(_TABLE_DIR / "rm_ancova_speed.docx"),
    title="ANCOVA on Change Scores: Controlling for Change in Sprint Speed (Δspeed)",
    note=f"OLS regression of Δ_metric (post − pre) on Δspeed (post − pre sprint speed, m/s), "
    f"mean-centered (group mean Δspeed = {_ancova_speed_mean:.2f} m/s). "
    f"Intercept t: tests whether mean change is non-zero after controlling for speed reduction. "
    f"* Intercept p < {_ancova_alpha:.3f} (Bonferroni for 3 muscles); "
    f"* Δspeed p < 0.05 (uncorrected).",
)
print(f"Saved {_TABLE_DIR / 'rm_ancova_speed.docx'}")

# =============================================================================
# Post-hoc paired comparisons table
# =============================================================================
lines = [
    r"\begin{table}",
    r"    \centering",
    r"    \resizebox{\textwidth}{!}{%",
    r"    \begin{tabular}{lccc}",
    r"    \hline \hline",
    r"    Variable & Pre & Post & $\Delta$ \\",
    r"    \hline",
]
for row in bio_rows:
    if not interaction_sig_by_metric.get(row["metric"], False):
        continue
    delta = row["Change"]
    if row["sig"]:
        delta += "$^*$"
    lines.append(f"    {row['Variable']} & {row['Pre']} & {row['Post']} & {delta} \\\\")
lines += [
    r"    \hline",
    r"    \end{tabular}}",
    r"    \caption{Post-hoc paired comparisons per muscle and metric (mean $\pm$ SD). "
    r"Reported where the time $\times$ muscle interaction was significant. "
    r"$^*p < 0.05$ after Bonferroni correction for " + str(len(muscles)) + r" comparisons.}",
    r"    \label{tab:muscle-biomechanics}",
    r"\end{table}",
]
(_TABLE_DIR / "posthoc_comparisons.tex").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {_TABLE_DIR / 'posthoc_comparisons.tex'}")

_posthoc_word_rows = []
for _row in bio_rows:
    if not interaction_sig_by_metric.get(_row["metric"], False):
        continue
    _posthoc_word_rows.append(
        {
            "Variable": _row["Variable"].replace(r"$l/l_0$", "l/l₀"),
            "Pre": _row["Pre"],
            "Post": _row["Post"],
            "Δ": _row["Change"] + ("*" if _row["sig"] else ""),
        }
    )
_save_word_table(
    pd.DataFrame(_posthoc_word_rows),
    str(_TABLE_DIR / "posthoc_comparisons.docx"),
    title="Post-Hoc Paired Comparisons per Muscle and Metric",
    note=f"Values are mean ± SD. Reported only for metrics where the time × muscle interaction was "
    f"significant. Post-hoc: Bonferroni-corrected paired t-tests "
    f"(α = 0.05 / {len(muscles)} ≈ {alpha_posthoc:.3f}). * p < {alpha_posthoc:.3f}.",
)
print(f"Saved {_TABLE_DIR / 'posthoc_comparisons.docx'}")
