"""
Shared table output helpers for APA-style Word and HTML tables.

Import these in any analysis script that needs to save tables:

    from analysis.core.output_helpers import (
        _save_word_table,
        _save_html_table,
        _extract_latex_caption,
    )
"""

import re as _re
from pathlib import Path

try:
    from docx import Document as _DocxDocument
    from docx.oxml import OxmlElement as _OxmlElement
    from docx.oxml.ns import qn as _qn

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
    for side, props in [
        ("top", top),
        ("bottom", bottom),
        ("left", None),
        ("right", None),
        ("insideH", None),
        ("insideV", None),
    ]:
        el = _OxmlElement(f"w:{side}")
        if props:
            for attr, val in props.items():
                el.set(_qn(f"w:{attr}"), val)
            el.set(_qn("w:space"), "0")
        else:
            el.set(_qn("w:val"), "none")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def _save_html_table(df, path: str, title: str = "", note: str = "") -> None:
    """APA-style HTML table: italic title above, three horizontal rules, Note. below."""
    parts = ['<div style="font-family:Times New Roman,serif;font-size:12pt;max-width:800px;">']
    if title:
        parts.append(f'<p style="font-style:italic;margin:0 0 2px 0;">{title}</p>')
    parts.append('<table style="border-collapse:collapse;width:100%;">')
    parts.append('<thead><tr style="border-top:1px solid black;border-bottom:1px solid black;">')
    for col in df.columns:
        parts.append(f'<th style="padding:4px 10px;text-align:left;font-weight:bold;">{col}</th>')
    parts.append("</tr></thead><tbody>")
    for i, (_, row) in enumerate(df.iterrows()):
        border = "border-bottom:1px solid black;" if i == len(df) - 1 else ""
        parts.append(f'<tr style="{border}">')
        for val in row:
            parts.append(f'<td style="padding:4px 10px;">{val if val is not None else ""}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if note:
        parts.append(f'<p style="margin:4px 0 0 0;"><em>Note.</em> {note}</p>')
    parts.append("</div>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _save_word_table(df, docx_path: str, title: str = "", note: str = "") -> None:
    """Save a DataFrame as an APA-style .docx table; falls back to .html if python-docx absent."""
    if not _DOCX_AVAILABLE:
        html_path = docx_path.replace(".docx", ".html")
        _save_html_table(df, html_path, title, note)
        print(f"  (python-docx not found – saved HTML fallback: {html_path})")
        return
    from docx.shared import Pt

    doc = _DocxDocument()
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.italic = True
        run.font.size = Pt(12)
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
            r.font.size = Pt(12)
        _set_cell_borders(cell, top=_APA_RULE, bottom=_APA_RULE)
    for ri, (_, row) in enumerate(df.iterrows()):
        is_last = ri == len(df) - 1
        for ci, val in enumerate(row):
            cell = tbl.rows[ri + 1].cells[ci]
            cell.text = str(val) if val is not None else ""
            if cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].font.size = Pt(12)
            _set_cell_borders(cell, bottom=_APA_RULE if is_last else None)
    if note:
        p = doc.add_paragraph()
        p.add_run("Note. ").italic = True
        p.add_run(note).font.size = Pt(12)
    doc.save(docx_path)


def _extract_latex_caption(tex_str: str) -> str:
    """Extract plain-text caption content from a LaTeX figure block."""
    idx = tex_str.find(r"\caption{")
    if idx == -1:
        return tex_str
    start = tex_str.index("{", idx) + 1
    depth, end = 1, start
    while end < len(tex_str) and depth:
        if tex_str[end] == "{":
            depth += 1
        elif tex_str[end] == "}":
            depth -= 1
        end += 1
    raw = tex_str[start : end - 1]
    raw = _re.sub(r"\$([^$]*)\$", r"\1", raw)
    raw = _re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", raw)
    raw = _re.sub(r"\\[a-zA-Z]+\*?", "", raw)
    raw = _re.sub(r"[{}]", "", raw)
    raw = raw.replace(r"\%", "%")
    raw = raw.replace("--", "-")
    return _re.sub(r"\s+", " ", raw).strip()
