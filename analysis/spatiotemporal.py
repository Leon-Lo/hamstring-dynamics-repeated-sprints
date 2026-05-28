"""
Sprint spatio-temporal analysis

=======================================
This script performs spatio-temporal analysis with statistics for all subjects in the dataset
and for each clinical group (Healthy/CON, CAM/FAIM, FAIS).

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

try:
    from docx import Document as _DocxDocument
    from docx.oxml import OxmlElement as _OxmlElement
    from docx.oxml.ns import qn as _qn
    from docx.shared import Pt as _Pt
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

_APA_RULE = {"val": "single", "sz": "8", "color": "000000"}


def _set_cell_borders(cell, top=None, bottom=None):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(_qn("w:tcBorders")):
        tcPr.remove(old)
    tcBorders = _OxmlElement("w:tcBorders")
    for side, props in [("top", top), ("bottom", bottom), ("left", None), ("right", None)]:
        el = _OxmlElement(f"w:{side}")
        if props:
            for attr, val in props.items():
                el.set(_qn(f"w:{attr}"), val)
            el.set(_qn("w:space"), "0")
        else:
            el.set(_qn("w:val"), "none")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def _save_word_table(df: "pd.DataFrame", docx_path: str, title: str = "", note: str = "") -> None:
    if not _DOCX_AVAILABLE:
        html_path = docx_path.replace(".docx", ".html")
        parts = ['<div style="font-family:Times New Roman,serif;font-size:12pt;">']
        if title:
            parts.append(f'<p style="font-style:italic;">{title}</p>')
        parts.append('<table style="border-collapse:collapse;width:100%;">')
        parts.append('<thead><tr style="border-top:1px solid black;border-bottom:1px solid black;">')
        for col in df.columns:
            parts.append(f'<th style="padding:4px 10px;text-align:left;">{col}</th>')
        parts.append("</tr></thead><tbody>")
        for i, (_, row) in enumerate(df.iterrows()):
            border = "border-bottom:1px solid black;" if i == len(df) - 1 else ""
            parts.append(f'<tr style="{border}">')
            for val in row:
                parts.append(f'<td style="padding:4px 10px;">{val if val is not None else ""}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")
        if note:
            parts.append(f'<p><em>Note.</em> {note}</p>')
        parts.append("</div>")
        Path(html_path).write_text("\n".join(parts), encoding="utf-8")
        print(f"  (python-docx not found – saved HTML: {html_path})")
        return
    doc = _DocxDocument()
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.italic = True
        run.font.size = _Pt(12)
    tbl = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    tblPr = tbl._tbl.tblPr
    tblBorders = _OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = _OxmlElement(f"w:{side}")
        el.set(_qn("w:val"), "none")
        tblBorders.append(el)
    tblPr.append(tblBorders)
    hdr = tbl.rows[0]
    for ci, col in enumerate(df.columns):
        cell = hdr.cells[ci]
        cell.text = str(col)
        if cell.paragraphs[0].runs:
            r = cell.paragraphs[0].runs[0]
            r.bold = True
            r.font.size = _Pt(11)
        _set_cell_borders(cell, top=_APA_RULE, bottom=_APA_RULE)
    for ri, (_, row) in enumerate(df.iterrows()):
        is_last = ri == len(df) - 1
        for ci, val in enumerate(row):
            cell = tbl.rows[ri + 1].cells[ci]
            cell.text = str(val) if val is not None else ""
            if cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].font.size = _Pt(11)
            _set_cell_borders(cell, bottom=_APA_RULE if is_last else None)
    if note:
        p = doc.add_paragraph()
        p.add_run("Note. ").italic = True
        p.add_run(note).font.size = _Pt(11)
    doc.save(docx_path)


# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)


import pandas as pd  # noqa: E402

from utils import Analysis_utils, Subject_utils  # noqa: E402

analysis_utils = Analysis_utils()
subject_utils = Subject_utils()

_TABLE_DIR = Path("data/tables/spatiotemporal")
_TABLE_DIR.mkdir(parents=True, exist_ok=True)


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
_NA_ROW_DEFAULTS = {
    "Pre": "—", "Post": "—", "Change": "—", "Change %": "—",
    "Normality (SW p)": "—", "Test": "—", "Stat": "—", "p": "—",
    "sig": False, "normal": True,
}


def summary_row(label: str, pre: pd.Series, post: pd.Series, unit: str = "") -> dict:
    """Return a single-row dict for the summary table."""
    var_label = f"{label}{(' (' + unit + ')') if unit else ''}"
    change = (post - pre).dropna()
    pre_c = pre.loc[change.index]
    post_c = post.loc[change.index]

    if len(change) < 3:
        return {"Variable": var_label, **_NA_ROW_DEFAULTS}

    pct_change = (change / pre_c) * 100

    stat_sw, p_sw = shapiro(change)
    normal = bool(p_sw > 0.05) if not np.isnan(p_sw) else True

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
            return f"{np.median(s):.3f} [{q1:.3f}–{q3:.3f}]"

        q1, q3 = np.percentile(pct_change, [25, 75])
        fmt_pct = f"{np.median(pct_change):.1f} [{q1:.1f}–{q3:.1f}]"

    return {
        "Variable": var_label,
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


# %% ── GROUP MAPPING ──────────────────────────────────────────────
_dem = pd.read_csv("data/subject_demographics.csv")
_grp_map = (
    _dem[["Subject", "Group"]]
    .dropna(subset=["Subject", "Group"])
    .assign(Subject=lambda d: d["Subject"].astype(str).str.strip())
    .set_index("Subject")["Group"]
    .str.strip()
)

_GROUP_DEFS = {
    "Overall": None,          # None = no filter
    "Healthy": "CON",
    "CAM": "FAIM",
    "FAIS": "FAIS",
}


def _subject_ids_for_group(df: pd.DataFrame, code: str | None) -> pd.Index:
    """Return the subject_id index values belonging to *code* (or all if None)."""
    if code is None:
        return df.index
    subject_str = df.index.astype(str).str.lstrip("0").str.strip()
    mask = subject_str.map(_grp_map) == code
    return df.index[mask]


def _filter(series: pd.Series, keep_idx: pd.Index) -> pd.Series:
    """Filter a Series to only the subjects in *keep_idx*."""
    common = series.index.intersection(keep_idx)
    return series.loc[common]


def _rows_for_group(speed: pd.DataFrame, contact: pd.DataFrame, step: pd.DataFrame, keep_idx: pd.Index) -> list:
    return [
        summary_row("Sprint duration", _filter(speed["pre_duration"], keep_idx), _filter(speed["post_duration"], keep_idx), "s"),
        summary_row("Distance", _filter(speed["pre_distance_m"], keep_idx), _filter(speed["post_distance_m"], keep_idx), "m"),
        summary_row(r"Running speed ($\frac{m}{s}$)", _filter(speed["pre_speed_m_s"], keep_idx), _filter(speed["post_speed_m_s"], keep_idx)),
        summary_row("Contact time", _filter(contact["contact_time_pre"], keep_idx), _filter(contact["contact_time_post"], keep_idx), "s"),
        summary_row("Step length", _filter(step["step_length_pre"], keep_idx), _filter(step["step_length_post"], keep_idx), "m"),
    ]


# Align all dataframes on subject_id (normalize to int) so index-based filtering works
speed_df = speed_df.set_index("subject_id")
contact_df["subject_id"] = contact_df["subject_id"].astype(int)
contact_df = contact_df.set_index("subject_id")
step_df["subject_id"] = step_df["subject_id"].astype(int)
step_df = step_df.set_index("subject_id")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

all_group_rows: dict[str, list] = {}
group_ns: dict[str, int] = {}

for group_label, group_code in _GROUP_DEFS.items():
    keep = _subject_ids_for_group(speed_df, group_code)
    group_ns[group_label] = len(keep)
    all_group_rows[group_label] = _rows_for_group(speed_df, contact_df, step_df, keep)

# %% ── COMBINED SUMMARY TABLE ─────────────────────────────────────
# One row per (Variable, Group) pair; Distance excluded from display.
_VARS_ORDER = ["Sprint duration", "Running speed", "Contact time", "Step length"]
_VAR_DISPLAY = {
    "Sprint duration": "Sprint duration (s)",
    "Running speed": r"Running speed (m/s)",
    "Contact time": "Contact time (s)",
    "Step length": "Step length (m)",
}

combined_rows = []
for group_label in _GROUP_DEFS:
    for row in all_group_rows[group_label]:
        var_key = next((v for v in _VARS_ORDER if row["Variable"].startswith(v)), None)
        if var_key is None:
            continue
        combined_rows.append({
            "Variable": _VAR_DISPLAY[var_key],
            "Group": f"{group_label} (n={group_ns[group_label]})",
            "Pre": row["Pre"],
            "Post": row["Post"],
            "Change": row["Change"],
            "Change %": row["Change %"],
            "Test": row["Test"],
            "Stat": row["Stat"],
            "p": row["p"],
            "_sig": row["sig"],
            "_normal": row["normal"],
            "_var_key": var_key,
        })

combined_df = pd.DataFrame(combined_rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

display_df = (
    combined_df
    .drop(columns=["_sig", "_normal", "_var_key"])
    .sort_values(["Variable", "Group"], key=lambda s: s if s.name != "Variable" else s.map({v: i for i, v in enumerate(_VAR_DISPLAY.values())}))
)
print("=" * 80)
print("TABLE 1 – Spatio-Temporal Statistics: Pre vs Post by Group")
print("=" * 80)
print(display_df.to_string(index=False))
print()

# %% ── LATEX TABLE ────────────────────────────────────────────────
# Combined table: Variable column uses \multirow; one row per group.
# $^*$ = p < 0.05; $^\dagger$ = non-parametric (Wilcoxon), values are median [IQR]
n_groups = len(_GROUP_DEFS)
print("\n% Table 1 — Spatio-Temporal Statistics (Pre vs Post by Group)")
print(r"\begin{tabular}{llllllll}")
print(r"\toprule")
print(r"Variable & Group & Pre & Post & Change & Change\% & Test & \textit{p} \\")
print(r"\midrule")

for var_key, var_label in _VAR_DISPLAY.items():
    var_rows = combined_df[combined_df["_var_key"] == var_key].reset_index(drop=True)
    for i, (_, row) in enumerate(var_rows.iterrows()):
        change_str = row["Change"]
        if row["_sig"]:
            change_str += r"$^{*}$"
        if not row["_normal"]:
            change_str += r"$^{\dagger}$"
        var_cell = rf"\multirow{{{n_groups}}}{{*}}{{{var_label}}}" if i == 0 else ""
        test_stat = f"{row['Test']}={row['Stat']}"
        print(f"  {var_cell} & {row['Group']} & {row['Pre']} & {row['Post']} & {change_str} & {row['Change %']}\\% & {test_stat} & {row['p']} \\\\")
    print(r"  \midrule")

print(r"\bottomrule")
print(r"\end{tabular}")

# %% ── SAVE FILES ─────────────────────────────────────────────────
# Build thesis-style table: Metric | Group | Pre | Post | Δ
# Δ embeds significance markers (* p<0.05, † Wilcoxon)
_thesis_rows = []
for _, row in combined_df.iterrows():
    delta = row["Change"]
    if row["_sig"]:
        delta += " *"
    if not row["_normal"]:
        delta += " †"
    _thesis_rows.append({
        "Metric": row["Variable"],
        "Group": row["Group"],
        "Pre": row["Pre"],
        "Post": row["Post"],
        "Δ": delta,
    })
_thesis_df = pd.DataFrame(_thesis_rows)

# Word (.docx)
_docx_path = str(_TABLE_DIR / "spatiotemporal_table1.docx")
_save_word_table(
    _thesis_df,
    _docx_path,
    title="Table 1. Pre- and post-fatigue spatio-temporal characteristics by group following the repeated sprints protocol (mean ± SD).",
    note=(
        "Mean ± SD (normal) or median [Q1–Q3] (non-normal). "
        "Δ = post − pre. * significant. † Wilcoxon signed-rank. "
        "Groups: Healthy = CON; CAM = FAIM (FAI morphology, no pain); FAIS = FAI symptomatic."
    ),
)
print(f"Saved {_docx_path}")

# LaTeX (.tex)
_tex_path = str(_TABLE_DIR / "spatiotemporal_table1.tex")
_tex_lines = [
    r"\begin{tabular}{llllllll}",
    r"\toprule",
    r"Variable & Group & Pre & Post & Change & Change\% & Test & \textit{p} \\",
    r"\midrule",
]
for var_key, var_label in _VAR_DISPLAY.items():
    var_rows = combined_df[combined_df["_var_key"] == var_key].reset_index(drop=True)
    for i, (_, row) in enumerate(var_rows.iterrows()):
        change_str = row["Change"]
        if row["_sig"]:
            change_str += r"$^{*}$"
        if not row["_normal"]:
            change_str += r"$^{\dagger}$"
        var_cell = rf"\multirow{{{n_groups}}}{{*}}{{{var_label}}}" if i == 0 else ""
        test_stat = f"{row['Test']}={row['Stat']}"
        _tex_lines.append(
            f"  {var_cell} & {row['Group']} & {row['Pre']} & {row['Post']} & {change_str} & {row['Change %']}\\% & {test_stat} & {row['p']} \\\\"
        )
    _tex_lines.append(r"  \midrule")
_tex_lines += [
    r"\bottomrule",
    r"\end{tabular}",
]
Path(_tex_path).write_text("\n".join(_tex_lines) + "\n", encoding="utf-8")
print(f"Saved {_tex_path}")
# %%
