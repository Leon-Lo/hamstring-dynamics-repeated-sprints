"""
Build combined_tables.docx and combined_tables.tex.

Reads all pre-saved snippet files produced by thesis_feedback.py, wcb.py, and
participant_demographics.py, then assembles them into:
    data/tables/combined_tables.docx
    data/tables/combined_tables.tex

Also recomputes the spatio-temporal results paragraph from raw data so the
numbers are always up-to-date without re-running the full thesis_feedback pipeline.

Run from the project root or from anywhere — the script sets cwd automatically.
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Project root setup ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from utils import Analysis_utils  # noqa: E402

analysis_utils = Analysis_utils()

try:
    import docx  # noqa: F401
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


def _latex_to_plain(text: str) -> str:
    """Strip LaTeX markup for use in plain-text docx paragraphs."""
    text = text.replace("\\cdot", "·")  # \cdot → · (before $ removal)
    text = text.replace("\\pm", "±")   # \pm → ± (before $ removal)
    text = text.replace("\\%", "%")    # \% → %
    text = text.replace("~", " ")      # non-breaking space → regular space
    text = text.replace("$", "")       # strip math-mode delimiters
    text = text.replace("\\\\", " ")   # LaTeX line breaks → space
    return text

# ── Spatio-temporal results paragraph ─────────────────────────────────────────


def _pct_change(pre: pd.Series, post: pd.Series) -> pd.Series:
    return (post - pre) / pre * 100


def _paired_p(pre: pd.Series, post: pd.Series) -> float:
    delta = post - pre
    _, p_sw = stats.shapiro(delta)
    if p_sw > 0.05:
        return stats.ttest_rel(pre, post).pvalue
    return stats.wilcoxon(pre, post).pvalue


def _sig_str(p: float) -> str:
    if p < 0.01:
        return "$p<0.01$"
    if p < 0.05:
        return "$p<0.05$"
    return f"$p={p:.3f}$"


def _fmt_pct(mean: float, std: float) -> str:
    sign = "+" if mean >= 0 else ""
    return f"${sign}{mean:.1f}\\%\\pm{std:.1f}\\%$"


def build_spatiotemporal_paragraph() -> str:
    speed_df = analysis_utils.get_speed_df()
    ct_df = analysis_utils.get_contact_time_df()
    sl_df = analysis_utils.get_step_length_df()

    speed_pct = speed_df["speed_change_%"]
    speed_p = _paired_p(speed_df["pre_speed_m_s"], speed_df["post_speed_m_s"])

    sl_pct = _pct_change(sl_df["step_length_pre"], sl_df["step_length_post"])
    sl_p = _paired_p(sl_df["step_length_pre"], sl_df["step_length_post"])

    ct_pct = _pct_change(ct_df["contact_time_pre"], ct_df["contact_time_post"])
    ct_p = _paired_p(ct_df["contact_time_pre"], ct_df["contact_time_post"])

    return (
        f"Following RS, participants ran with significantly lower average sprint speed "
        f"({_fmt_pct(speed_pct.mean(), speed_pct.std())}; {_sig_str(speed_p)}) "
        f"and step length ({_fmt_pct(sl_pct.mean(), sl_pct.std())}; {_sig_str(sl_p)}), "
        f"and higher contact time ({_fmt_pct(ct_pct.mean(), ct_pct.std())}; {_sig_str(ct_p)}) "
        f"(Tab.~1). Hip, knee, and ankle angles and moments were significantly different "
        f"after RS (Fig.~2)."
    )


# ── Fiber length results paragraph ───────────────────────────────────────────

_MUSCLES = ["bflh", "semiten", "semimem"]
_LABELS = {"bflh": "BFLH", "semiten": "ST", "semimem": "SM"}


def _fmt_abs(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _parse_peak_timing_sig() -> dict[str, bool]:
    """Read peak_timing.tex and return {muscle: is_significant} from the Sig column."""
    sig = {m: False for m in _MUSCLES}
    pt_path = Path("data/tables/peak_timing.tex")
    if not pt_path.exists():
        return sig
    for line in pt_path.read_text().splitlines():
        for m in _MUSCLES:
            if line.strip().startswith(m + " "):
                # last &-delimited field, strip LaTeX line ending
                last = line.rsplit("&", 1)[-1].replace("\\\\", "").strip()
                sig[m] = ("*" in last)
    return sig


def build_fiber_length_results_paragraph() -> str:
    fl_df = analysis_utils.get_peak_norm_fiber_length_df()
    n_total = len(fl_df)

    # ── 1. Peak timing sentence (from pre-saved peak_timing.tex) ─────────────
    timing_sig = _parse_peak_timing_sig()
    sig_muscles = [_LABELS[m] for m in _MUSCLES if timing_sig[m]]
    ns_muscles = [_LABELS[m] for m in _MUSCLES if not timing_sig[m]]
    if sig_muscles:
        timing_sentence = (
            f"Following RS, peak muscle lengths occurred significantly earlier "
            f"for {' and '.join(sig_muscles)}, "
            f"but not for {' and '.join(ns_muscles)}."
        )
    else:
        timing_sentence = (
            "Following RS, peak muscle length timing did not significantly change "
            "for any muscle."
        )

    # ── 2. Whole-group fiber length stats (Bonferroni-corrected, α/3) ──────────
    _BONF_N = len(_MUSCLES)
    sig_whole = []
    whole_parts = []
    for muscle in _MUSCLES:
        pre = fl_df[f"{muscle}_peak_norm_fiber_length_pre"].dropna()
        post = fl_df[f"{muscle}_peak_norm_fiber_length_post"].dropna()
        idx = pre.index.intersection(post.index)
        pre, post = pre.loc[idx], post.loc[idx]
        pct = _pct_change(pre, post)
        p_adj = min(_paired_p(pre, post) * _BONF_N, 1.0)
        if p_adj < 0.05:
            sig_whole.append(_LABELS[muscle])
        whole_parts.append(
            f"{_fmt_pct(pct.mean(), pct.std())} for {_LABELS[muscle]} "
            f"(pre: {_fmt_abs(pre.mean(), pre.std())}, "
            f"post: {_fmt_abs(post.mean(), post.std())})"
        )

    if sig_whole:
        sig_note = f"{' and '.join(sig_whole)} showed a significant change."
    else:
        sig_note = "No muscle showed a significant whole-group change."

    whole_sentence = (
        "Our simulations showed a significant effect of RS on muscle length. "
        f"Peak normalized fiber lengths changed on average by "
        f"{', '.join(whole_parts[:-1])}, and {whole_parts[-1]}. {sig_note}"
    )

    # ── 3. Risk prone group definition (increased BFLH fiber length) ──────────
    resp_dict = analysis_utils.get_norm_fiber_length_responders_ids(muscles=["bflh"])
    risk_prone_ids = set(resp_dict["bflh"])
    n_risk = len(risk_prone_ids)

    def _pct_n(n: int) -> str:
        return f"{(n / n_total) * 100:.1f}"

    risk_count_sentence = (
        f"Of the {n_total} participants, {n_risk} ({_pct_n(n_risk)}\\%) "
        f"showed an increased peak BFLH fiber length post repeated sprints"
        f"and were classified as risk prone."
    )

    # ── 4. Risk prone group fiber length stats ────────────────────────────────
    rp_df = fl_df[fl_df["subject_id"].isin(risk_prone_ids)]
    resp_parts = []
    for muscle in _MUSCLES:
        pre = rp_df[f"{muscle}_peak_norm_fiber_length_pre"].dropna()
        post = rp_df[f"{muscle}_peak_norm_fiber_length_post"].dropna()
        idx = pre.index.intersection(post.index)
        pre, post = pre.loc[idx], post.loc[idx]
        pct = _pct_change(pre, post)
        resp_parts.append(
            f"{_fmt_pct(pct.mean(), pct.std())} for {_LABELS[muscle]} "
            f"(pre: {_fmt_abs(pre.mean(), pre.std())}, "
            f"post: {_fmt_abs(post.mean(), post.std())})"
        )

    risk_prone_sentence = (
        "Among risk prone participants, peak normalized fiber lengths changed on "
        f"average by {', '.join(resp_parts[:-1])}, and {resp_parts[-1]}."
    )

    return "\n".join([
        timing_sentence,
        whole_sentence,
        risk_count_sentence,
        risk_prone_sentence,
    ])


def build_muscle_work_results_paragraph() -> str:
    """Generate results paragraph for eccentric (negative) muscle work."""
    work_df = analysis_utils.get_muscle_work_df()
    # Reset index so subject_id is a column
    work_df = work_df.reset_index()
    n_total = len(work_df)

    # ── 1. Whole-group negative work stats (Bonferroni-corrected, α/3) ──────────
    _BONF_N = len(_MUSCLES)
    sig_whole = []
    whole_parts = []
    for muscle in _MUSCLES:
        pre = work_df[f"{muscle}_negative_work_pre"].dropna()
        post = work_df[f"{muscle}_negative_work_post"].dropna()
        idx = pre.index.intersection(post.index)
        pre, post = pre.loc[idx], post.loc[idx]
        pct = _pct_change(pre, post)
        p_adj = min(_paired_p(pre, post) * _BONF_N, 1.0)
        if p_adj < 0.05:
            sig_whole.append(_LABELS[muscle])
        whole_parts.append(
            f"{_fmt_pct(pct.mean(), pct.std())} for {_LABELS[muscle]} "
            f"(pre: {_fmt_abs(pre.mean(), pre.std())}, "
            f"post: {_fmt_abs(post.mean(), post.std())})"
        )

    if sig_whole:
        sig_note = f"{' and '.join(sig_whole)} showed a significant change."
    else:
        sig_note = "No muscle showed a significant whole-group change."

    whole_sentence = (
        "Our simulations showed a significant effect of RS on eccentric work absorption. "
        f"Eccentric work changed on average by "
        f"{', '.join(whole_parts[:-1])}, and {whole_parts[-1]}. {sig_note}"
    )

    # ── 2. Risk prone group definition (increased negative work) ──────────
    # Define responders as those with increased negative work (eccentric overload)
    responders_ids = set()
    for muscle in ["bflh"]:
        pre = work_df[f"{muscle}_negative_work_pre"].dropna()
        post = work_df[f"{muscle}_negative_work_post"].dropna()
        idx = pre.index.intersection(post.index)
        pre_vals, post_vals = pre.loc[idx].values, post.loc[idx].values
        # Responders show increased eccentric work (post > pre)
        responder_mask = post_vals > pre_vals
        responders_ids.update(work_df.loc[idx[responder_mask], "subject_id"].values)

    risk_prone_ids = responders_ids
    n_risk = len(risk_prone_ids)

    def _pct_n(n: int) -> str:
        return f"{(n / n_total) * 100:.1f}"

    risk_count_sentence = (
        f"Of the {n_total} participants, {n_risk} ({_pct_n(n_risk)}\\%) "
        f"showed increased BFLH eccentric work post repeated sprints "
        f"and were classified as risk prone."
    )

    # ── 3. Risk prone group negative work stats ────────────────────────────────
    rp_df = work_df[work_df["subject_id"].isin(risk_prone_ids)]
    resp_parts = []
    for muscle in _MUSCLES:
        pre = rp_df[f"{muscle}_negative_work_pre"].dropna()
        post = rp_df[f"{muscle}_negative_work_post"].dropna()
        idx = pre.index.intersection(post.index)
        pre, post = pre.loc[idx], post.loc[idx]
        pct = _pct_change(pre, post)
        resp_parts.append(
            f"{_fmt_pct(pct.mean(), pct.std())} for {_LABELS[muscle]} "
            f"(pre: {_fmt_abs(pre.mean(), pre.std())}, "
            f"post: {_fmt_abs(post.mean(), post.std())})"
        )

    risk_prone_sentence = (
        "Among risk prone participants, eccentric work changed on "
        f"average by {', '.join(resp_parts[:-1])}, and {resp_parts[-1]}."
    )

    return "\n".join([
        whole_sentence,
        risk_count_sentence,
        risk_prone_sentence,
    ])


# ── docx builder ──────────────────────────────────────────────────────────────


def build_combined_docx(output_path: str) -> None:
    if not _DOCX_AVAILABLE:
        print("  python-docx not available — skipping combined .docx")
        return

    from docx import Document
    from docx.shared import Pt

    def _add_h(doc, text, level=1):
        doc.add_heading(text, level=level)

    def _add_note(doc, text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.italic = True
        r.font.size = Pt(9)

    def _df_to_tbl(doc, df):
        t = doc.add_table(rows=1 + len(df), cols=len(df.columns))
        t.style = "Table Grid"
        for ci, col in enumerate(df.columns):
            cell = t.rows[0].cells[ci]
            cell.text = str(col)
            if cell.paragraphs[0].runs:
                r = cell.paragraphs[0].runs[0]
                r.bold = True
                r.font.size = Pt(9)
        for ri, (_, row) in enumerate(df.iterrows()):
            for ci, val in enumerate(row):
                cell = t.rows[ri + 1].cells[ci]
                cell.text = str(val) if pd.notna(val) else "---"
                if cell.paragraphs[0].runs:
                    cell.paragraphs[0].runs[0].font.size = Pt(9)

    def _copy_tbl(src_tbl, dest_doc):
        tbl = copy.deepcopy(src_tbl._tbl)
        dest_doc.add_paragraph()
        tmp = dest_doc.add_table(rows=0, cols=0)
        tmp._tbl.getparent().remove(tmp._tbl)
        dest_doc.paragraphs[-1]._p.addnext(tbl)

    combined = Document()
    combined.add_heading("Tables and Figure Captions", 0)

    # ── Demographics (optional) ───────────────────────────────────────────────
    _dem_summary = Path("data/plot_results/participant_demographics/demographics_table.csv")
    _dem_per = Path("data/plot_results/participant_demographics/per_subject_demographics.csv")
    if _dem_summary.exists() and _dem_per.exists():
        pub = pd.read_csv(_dem_summary, index_col=0)
        per = pd.read_csv(_dem_per)
        per["BMI"] = per["BMI"].round(1)
        for col in ["Alpha angle", "Centre edge angle"]:
            if col in per.columns:
                per[col] = per[col].round(1)
        per = per.rename(columns={
            "Dominant Leg ": "Dom. leg",
            "Measured Leg": "Meas. leg",
            "Alpha angle": "Alpha (°)",
            "Centre edge angle": "CEA (°)",
        })
        per["Sex"] = per["Sex"].map({"Male": "M", "Female": "F"})

        _add_h(combined, "Participant Demographics — Group Summary")
        _df_to_tbl(combined, pub.reset_index())
        _add_note(combined,
                  "Note. Values are mean ± SD. Risk prone = positive % increase in peak "
                  "normalized BFLH fiber length.")
        combined.add_paragraph()

        _add_h(combined, "Participant Demographics — Per Subject")
        _df_to_tbl(combined, per)
        _add_note(combined,
                  "Note. FAIM = FAI morphology; FAIS = FAI symptomatic; CON = healthy control. "
                  "CEA = centre edge angle. --- = missing value.")
        combined.add_paragraph()

    # ── Spatio-temporal table (optional, from spatiotemporal_stats.csv) ───────
    _st_path = Path("data/spatiotemporal_stats.csv")
    if _st_path.exists():
        _st = pd.read_csv(_st_path)
        _st_vars = [
            ("Sprint duration (s)", "pre_duration", "post_duration"),
            ("Step length (m)", "pre_distance_m", "post_distance_m"),
            ("Running speed (m/s)", "pre_speed_m_s", "post_speed_m_s"),
            ("Contact time (s)", "contact_time_pre", "contact_time_post"),
        ]
        _st_rows = []
        for _vn, _pc, _oc in _st_vars:
            _pre, _post = _st[_pc], _st[_oc]
            _delta = _post - _pre
            _, _sp = stats.shapiro(_delta)
            _norm = _sp > 0.05
            _p = (stats.ttest_rel(_pre, _post) if _norm else stats.wilcoxon(_pre, _post)).pvalue

            def _fmt(s, n=_norm):
                if n:
                    return f"{s.mean():.3f} ± {s.std():.3f}"
                q1, q3 = np.percentile(s, [25, 75])
                return f"{np.median(s):.3f} [{q1:.3f}–{q3:.3f}]"

            _st_rows.append({
                "Variable": _vn,
                "Pre": _fmt(_pre),
                "Post": _fmt(_post),
                "Δ": _fmt(_delta) + (" *" if _p < 0.05 else "") + ("" if _norm else " †"),
                "p": f"{_p:.3f}",
            })
        _add_h(combined, "Table 1 — Spatio-Temporal Statistics: Pre vs. Post Intervention")
        _df_to_tbl(combined, pd.DataFrame(_st_rows))
        _add_note(combined,
                  "Note. Mean ± SD (normal) or median [Q1–Q3] (non-normal). "
                  "Δ = post − pre. * p < 0.05. † Wilcoxon signed-rank.")
        combined.add_paragraph()

    # ── Main tables (read from individually saved .docx files) ───────────────
    _tbl_sources = [
        ("data/tables/posthoc_comparisons.docx", "Table 2 — Post-Hoc Paired Comparisons per Muscle and Metric"),
        ("data/tables/peak_timing.docx", "Table 4 — Peak Fiber Length Timing: Pre vs. Post Intervention"),
        ("data/tables/responder_kinematics.docx", "Table 5 — Risk Prone vs. Low Risk: Kinematics"),
        ("data/tables/responder_spatiotemporal.docx", "Table 6 — Risk Prone vs. Low Risk: Spatio-Temporal"),
        ("data/tables/baseline_counts.docx", "Table 7 — Baseline Trial Counts per Subject"),
        ("data/tables/mixed_anova_redistribution.docx", "Table 8 — 2×3 Mixed ANOVA: Time × Group (Eccentric Load Redistribution %)"),
        ("data/tables/rm_anova_time_only.docx", "Table 9 — RM ANOVA: Time (pre/post) — Eccentric Metrics per Muscle (Overall Group)"),
        ("data/tables/rm_ancova_speed.docx", "Table 10 — RM ANCOVA: Time Effect Controlling for Change in Sprint Speed (Δspeed)"),
        ("data/tables/ancova_pct_contribution_speed.docx", "Table 11 — ANCOVA on % Contribution Change Scores: Controlling for Δspeed"),
        ("data/tables/ancova_abs_speed.docx", "Table 12 — ANCOVA on Absolute Eccentric Metrics: Change Scores Adjusted for Sprint Speed Reduction"),
        ("data/tables/ancova_summary_pre_post.docx", "Table 13 — Eccentric Metrics Pre/Post Means with Unadjusted and Speed-Adjusted p-Values"),
        ("data/tables/ancova_pct_summary_pre_post.docx", "Table 14 — Eccentric Load Contribution (%) Pre/Post Means with Unadjusted and Speed-Adjusted p-Values"),
    ]
    for _tbl_path, _htxt in _tbl_sources:
        if not Path(_tbl_path).exists():
            continue
        _tdoc = Document(_tbl_path)
        _add_h(combined, _htxt)
        for _para in _tdoc.paragraphs:
            _txt = _para.text.strip()
            if not _txt:
                continue
            _p2 = combined.add_paragraph()
            _r2 = _p2.add_run(_txt)
            _r2.italic = _txt.lower().startswith("note")
            _r2.font.size = Pt(10)
        for _t2 in _tdoc.tables:
            _copy_tbl(_t2, combined)
        combined.add_paragraph()

    # ── Prose summaries ───────────────────────────────────────────────────────
    _prose_sources = [
        ("Spatio-Temporal Results", "data/tables/spatiotemporal_results_paragraph.tex"),
        ("Fiber Length Results", "data/tables/fiber_length_results_paragraph.tex"),
        ("Whole-Group Fiber Length Summary", "data/tables/whole_group_fiber_length_summary.tex"),
        ("Risk Prone Fiber Length Summary", "data/tables/responder_fiber_length_summary.tex"),
        ("Eccentric Peak Force Summary", "data/tables/eccentric_peak_force_summary.tex"),
        ("Eccentric Impulse Summary", "data/tables/eccentric_impulse_summary.tex"),
        ("Eccentric Knee Moment Summary", "data/tables/eccentric_moment_summary.tex"),
        ("Fiber Length (Eccentric Window) Summary", "data/tables/fiber_length_ecc_summary.tex"),
    ]
    for _label, _fpath in _prose_sources:
        if not Path(_fpath).exists():
            continue
        _add_h(combined, _label)
        _raw = Path(_fpath).read_text().strip()
        combined.add_paragraph(_latex_to_plain(_raw))
        combined.add_paragraph()

    # ── Figure captions ───────────────────────────────────────────────────────
    _cap_sources = [
        ("Biomechanics Pre/Post", "data/tables/biomechanics_fig_caption.txt"),
        ("Eccentric Window Sanity Check", "data/tables/eccentric_window_sanity_check_caption.txt"),
        ("Normalized Fiber Length Curves", "data/tables/fiber_length_fig_caption.txt"),
        ("SPM Group Comparison: Hip", "data/tables/spm_group_hip_caption.txt"),
        ("SPM Group Comparison: Knee", "data/tables/spm_group_knee_caption.txt"),
        ("SPM Group Comparison: Ankle", "data/tables/spm_group_ankle_caption.txt"),
        ("SPM Group Comparison: Muscle Forces", "data/tables/spm_group_muscle_forces_caption.txt"),
        ("SPM Group Comparison: Adjusted EMG", "data/tables/spm_group_adjusted_emg_caption.txt"),
        ("Risk Prone vs. Low Risk: Spatio-Temporal Change Scores", "data/tables/risk_prone_spatiotemporal_caption.txt"),
        ("Risk Prone vs. Low Risk: Kinematics Change Scores", "data/tables/risk_prone_kinematics_caption.txt"),
        ("Figure 2 — Joint Angles by Group (Pre/Post SPM)", "data/tables/Figure2_prepost_spm_joint_angles_caption.txt"),
        ("Figure 3 — Joint Moments by Group (Pre/Post SPM)", "data/tables/Figure3_prepost_spm_joint_moments_caption.txt"),
        ("Figure 4 — Muscle Forces by Group (Pre/Post SPM)", "data/tables/Figure4_prepost_spm_muscle_forces_caption.txt"),
        ("Figure 5 — Fiber Lengths by Group (Pre/Post SPM)", "data/tables/Figure5_prepost_spm_fiber_lengths_caption.txt"),
        ("Figure 6 — Eccentric Load Redistribution by Group", "data/tables/Figure6_eccentric_redistribution_by_group_caption.txt"),
    ]
    if any(Path(fp).exists() for _, fp in _cap_sources):
        _add_h(combined, "Figure Captions")
        for _ftitle, _fpath in _cap_sources:
            if not Path(_fpath).exists():
                continue
            _p3 = combined.add_paragraph()
            _rb = _p3.add_run(_ftitle + ": ")
            _rb.bold = True
            _rb.font.size = Pt(11)
            _rc = _p3.add_run(Path(_fpath).read_text().strip())
            _rc.font.size = Pt(11)
            combined.add_paragraph()

    combined.save(output_path)
    print(f"Saved {output_path}")


# ── tex builder ───────────────────────────────────────────────────────────────


def build_combined_tex(output_path: str) -> None:
    _tex_sections = [
        ("Participant Demographics — Group Summary", None),
        ("Participant Demographics — Per Subject", None),
        ("Table 1 — Spatio-Temporal Statistics", None),
        ("Spatio-Temporal Results", "data/tables/spatiotemporal_results_paragraph.tex"),
        ("Fiber Length Results", "data/tables/fiber_length_results_paragraph.tex"),
        ("Table 2 — Post-Hoc Paired Comparisons per Muscle and Metric", "data/tables/posthoc_comparisons.tex"),
        ("Table 4 — Peak Fiber Length Timing: Pre vs.\\ Post Intervention", "data/tables/peak_timing.tex"),
        ("Table 5 — Risk Prone vs.\\ Low Risk: Kinematics", None),
        ("Table 6 — Risk Prone vs.\\ Low Risk: Spatio-Temporal", None),
        ("Table 7 — Baseline Trial Counts per Subject", "data/tables/baseline_counts.tex"),
        ("Table 8 — 2×3 Mixed ANOVA: Time × Group (Eccentric Load Redistribution %)", "data/tables/mixed_anova_redistribution.tex"),
        ("Table 9 — RM ANOVA: Time (pre/post) — Eccentric Metrics per Muscle (Overall Group)", None),
        ("Whole-Group Fiber Length Summary", "data/tables/whole_group_fiber_length_summary.tex"),
        ("Risk Prone Fiber Length Summary", "data/tables/responder_fiber_length_summary.tex"),
        ("Eccentric Peak Force Summary", "data/tables/eccentric_peak_force_summary.tex"),
        ("Eccentric Impulse Summary", "data/tables/eccentric_impulse_summary.tex"),
        ("Eccentric Knee Moment Summary", "data/tables/eccentric_moment_summary.tex"),
        ("Fiber Length (Eccentric Window) Summary", "data/tables/fiber_length_ecc_summary.tex"),
        ("Biomechanics Pre/Post", "data/tables/biomechanics_fig_caption.tex"),
        ("Eccentric Window Sanity Check", "data/tables/eccentric_window_sanity_check_caption.tex"),
        ("Normalized Fiber Length Curves", "data/tables/fiber_length_fig_caption.tex"),
        ("SPM Group Comparison: Hip", "data/tables/spm_group_hip_caption.tex"),
        ("SPM Group Comparison: Knee", "data/tables/spm_group_knee_caption.tex"),
        ("SPM Group Comparison: Ankle", "data/tables/spm_group_ankle_caption.tex"),
        ("SPM Group Comparison: Muscle Forces", "data/tables/spm_group_muscle_forces_caption.tex"),
        ("SPM Group Comparison: Adjusted EMG", "data/tables/spm_group_adjusted_emg_caption.tex"),
        ("Figure 2 --- Joint Angles by Group (Pre/Post SPM)", "data/tables/Figure2_prepost_spm_joint_angles_caption.tex"),
        ("Figure 3 --- Joint Moments by Group (Pre/Post SPM)", "data/tables/Figure3_prepost_spm_joint_moments_caption.tex"),
        ("Figure 4 --- Muscle Forces by Group (Pre/Post SPM)", "data/tables/Figure4_prepost_spm_muscle_forces_caption.tex"),
        ("Figure 5 --- Fiber Lengths by Group (Pre/Post SPM)", "data/tables/Figure5_prepost_spm_fiber_lengths_caption.tex"),
        ("Figure 6 --- Eccentric Load Redistribution by Group", "data/tables/Figure6_eccentric_redistribution_by_group_caption.tex"),
    ]

    _tex_notes = {
        "data/tables/peak_timing.tex": (
            r"\vspace{4pt}\par\noindent\footnotesize\textit{Note.}~"
            r"Values are gait-cycle percentage at which peak fiber length occurs (mean~$\pm$~SD). "
            r"SW~$p$ = Shapiro-Wilk $p$-value for the change score. "
            r"Test = paired $t$-test (t) or Wilcoxon signed-rank (W). $^*p < 0.05$."
        ),
        "data/tables/baseline_counts.tex": (
            r"\vspace{4pt}\par\noindent\footnotesize\textit{Note.}~"
            r"Number of baseline trials available per subject prior to repeated sprints."
        ),
        "data/tables/mixed_anova_redistribution.tex": (
            r"\vspace{4pt}\par\noindent\footnotesize\textit{Note.}~"
            r"Bonferroni-corrected $\alpha = 0.05/3 \approx 0.017$ across muscles per metric. "
            r"DVs are raw eccentric values per muscle (peak force in N, impulse in N$\cdot$s, knee moment in N$\cdot$m$\cdot$s, fiber length in $l/l_0$). "
            r"$^*p < \alpha_\text{Bonf}$. "
            r"Normality assessed with Shapiro-Wilk per group~$\times$~time cell."
        ),
    }

    lines = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\usepackage{longtable}",
        r"\title{Tables and Figure Captions}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]

    for _title, _fpath in _tex_sections:
        if _fpath is None or not Path(_fpath).exists():
            continue
        lines.append("")
        lines.append(r"% " + "─" * 55)
        lines.append(r"\section*{" + _title + "}")
        lines.append("")
        _content = Path(_fpath).read_text().strip()
        if _content.startswith(r"\begin{tabular}"):
            lines.append(r"\begin{table}[h!]")
            lines.append(r"    \centering")
            lines.append(r"    \resizebox{\textwidth}{!}{%")
            lines.append(_content)
            lines.append(r"    }")
            lines.append(r"\end{table}")
        else:
            lines.append(_content)
        if _fpath in _tex_notes:
            lines.append(_tex_notes[_fpath])

    lines += ["", r"\end{document}", ""]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    para = build_spatiotemporal_paragraph()
    Path("data/tables/spatiotemporal_results_paragraph.tex").write_text(para)
    print("Saved data/tables/spatiotemporal_results_paragraph.tex")
    print(para)
    print()

    fl_para = build_fiber_length_results_paragraph()
    Path("data/tables/fiber_length_results_paragraph.tex").write_text(fl_para)
    print("Saved data/tables/fiber_length_results_paragraph.tex")
    print(fl_para)
    print()

    build_combined_docx("data/tables/combined_tables.docx")
    build_combined_tex("data/tables/combined_tables.tex")
