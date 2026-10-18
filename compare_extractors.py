"""Side-by-side check: PyMuPDF (this project's pipeline) vs Docling on the Hindi PDF.

It answers one question: does Docling give cleaner Devanagari than what we already have?
For every page it counts extraction damage in three versions of the text, shows the
selected pages side by side, and checks whether the facts the test queries need survive.

Setup (once):   pip install docling
Run:            python compare_extractors.py
                python compare_extractors.py --pages 3,9,16 --out my_report.md
Output:         compare_report.md  (open it in VS Code; it is UTF-8)
                docling_cache.json (Docling's output, so re-runs are instant)

Docling downloads its layout/table models from Hugging Face on the first run.
"""
import argparse
import json
import os
import re
import sys

import pymupdf

from hindi_rag import _DUP_MARKS, _DUP_PAIRS, DEFAULT_PDF, clean_text, extract_pages

# a vowel sign / anusvara / nukta directly after a space or at the start of a line is broken text
_ORPHAN_MARK = re.compile(r"(?:^|\s)[ँ-ः़ा-ौ]")
_BAD_CHARS = re.compile("[�-]")


# ---------------------------------------------------------------- extractors
def pymupdf_raw(pdf: str) -> dict[int, str]:
    """Plain PyMuPDF text, exactly as it comes out of the PDF (no repairs)."""
    return {p.number + 1: p.get_text() for p in pymupdf.open(pdf)}


def pymupdf_project(pdf: str) -> dict[int, str]:
    """PyMuPDF text after this project's own cleaning, paragraph rebuilding and table reading."""
    out = {}
    for pg in extract_pages(pdf):
        out[pg.number] = "\n".join(pg.paragraphs + [row for t in pg.tables for row in t])
    return out


def docling_pages(pdf: str, cache: str, use_ocr: bool) -> dict[int, str]:
    """Docling text grouped by page. Cached, because the conversion is slow."""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = use_ocr  # the PDF has a real text layer, so OCR is off unless asked for
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    print("Running Docling (first run downloads models and can take several minutes)...")
    doc = conv.convert(pdf).document

    pages: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        if type(item).__name__ == "TableItem":
            try:
                text = item.export_to_markdown(doc=doc)
            except TypeError:  # older Docling versions take no argument
                text = item.export_to_markdown()
        else:
            text = getattr(item, "text", "") or ""
        if text.strip():
            pages.setdefault(page_no, []).append(text.strip())

    result = {p: "\n".join(parts) for p, parts in pages.items()}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    return result


# ------------------------------------------------------------------- scoring
def damage(text: str) -> dict:
    return {
        "chars": len(text),
        "doubled": len(_DUP_MARKS.findall(text)) + len(_DUP_PAIRS.findall(text)),
        "orphan": len(_ORPHAN_MARK.findall(text)),
        "bad": len(_BAD_CHARS.findall(text)),
    }


def total(pages: dict[int, str]) -> dict:
    t = {"chars": 0, "doubled": 0, "orphan": 0, "bad": 0}
    for text in pages.values():
        for k, v in damage(text).items():
            t[k] += v
    return t


def expected_strings(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    items = json.load(open(path, encoding="utf-8"))
    return sorted({clean_text(e) for it in items for e in it["expect"]})


def facts_found(pages: dict[int, str], expect: list[str]) -> list[str]:
    blob = " ".join(clean_text(t) for t in pages.values())
    return [e for e in expect if e in blob]


# -------------------------------------------------------------------- report
def table_row(label: str, t: dict) -> str:
    return f"| {label} | {t['chars']} | {t['doubled']} | {t['orphan']} | {t['bad']} |"


def build_report(raw, proj, dock, selected, expect, max_chars) -> str:
    lines = ["# PyMuPDF vs Docling on the Hindi PDF", ""]

    lines += ["## 1. Damage counts, whole document", "",
              "Lower is cleaner. `doubled` = repeated vowel signs, `orphan` = a vowel sign "
              "stranded after a space, `bad` = replacement or private-use characters.", "",
              "| Version | Characters | Doubled | Orphan | Bad |", "| --- | --- | --- | --- | --- |",
              table_row("PyMuPDF, raw", total(raw)),
              table_row("PyMuPDF + this project's cleaning", total(proj)),
              table_row("Docling, raw", total(dock)),
              table_row("Docling + clean_text()", total({p: clean_text(t) for p, t in dock.items()})), ""]

    lines += ["## 2. Damage per page (raw PyMuPDF / project / raw Docling)", "",
              "| Page | PyMuPDF raw | Project | Docling raw |", "| --- | --- | --- | --- |"]
    for p in sorted(set(raw) | set(dock)):
        def cell(src):
            d = damage(src.get(p, ""))
            return f"{d['doubled']}d {d['orphan']}o {d['bad']}b ({d['chars']} chars)"
        lines.append(f"| {p} | {cell(raw)} | {cell(proj)} | {cell(dock)} |")
    lines.append("")

    if expect:
        lines += ["## 3. Do the facts the test queries need survive?", ""]
        for label, src in [("Project pipeline", proj), ("Docling raw", dock),
                           ("Docling + clean_text()", {p: clean_text(t) for p, t in dock.items()})]:
            ok = facts_found(src, expect)
            missing = [e for e in expect if e not in ok]
            lines.append(f"- **{label}:** {len(ok)} of {len(expect)} found."
                         + (f" Missing: {', '.join(missing)}" if missing else ""))
        lines.append("")

    lines += ["## 4. Selected pages side by side", "",
              "Read these yourself. Counts cannot tell you whether the Hindi *looks* right.", ""]
    for p in selected:
        lines += [f"### Page {p}", ""]
        for label, src in [("PyMuPDF, project pipeline", proj), ("Docling, raw", dock)]:
            text = src.get(p, "(nothing extracted for this page)")
            cut = text[:max_chars] + (" [...]" if len(text) > max_chars else "")
            lines += [f"**{label}**", "", "```text", cut, "```", ""]

    lines += ["## 5. How to decide", "",
              "- Prefer Docling (or a mix) only if its damage counts are clearly lower **and** "
              "the selected pages read better, tables included.",
              "- If the counts are similar, keep PyMuPDF: it is lighter, already tuned, and "
              "already passes retrieval 6 of 6.",
              "- Either way, put the outcome in the README as a design decision."]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # PowerShell-safe Hindi output
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--pages", default="3,4,9,16,17", help="pages to show side by side (1-based, comma separated)")
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--out", default="compare_report.md")
    ap.add_argument("--cache", default="docling_cache.json")
    ap.add_argument("--ocr", action="store_true", help="turn Docling OCR on (off by default)")
    ap.add_argument("--max-chars", type=int, default=1800, help="characters shown per page and extractor")
    args = ap.parse_args()

    selected = [int(x) for x in args.pages.split(",") if x.strip()]
    raw, proj = pymupdf_raw(args.pdf), pymupdf_project(args.pdf)
    dock = docling_pages(args.pdf, args.cache, args.ocr)

    report = build_report(raw, proj, dock, selected, expected_strings(args.queries), args.max_chars)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(report.split("## 2.")[0])  # the whole-document table and the explanation
    print(f"\nFull report with side-by-side pages: {args.out}")


if __name__ == "__main__":
    main()
