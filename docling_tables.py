"""Docling as a table reader. PyMuPDF keeps reading the body text; this module only supplies tables.

    python ingest.py --tables docling

Needs `pip install docling`. The first run downloads Docling's models from Hugging Face and can
take several minutes; the result is cached in docling_tables_cache.json so later runs are instant.
Delete that file to force a fresh conversion.
"""
from __future__ import annotations

import json
import os
import re

CACHE = "docling_tables_cache.json"
_SEPARATOR_ROW = re.compile(r"^\|[\s:\-|]+\|$")


def _markdown_rows(md: str) -> list[list[str]]:
    """Turn a Docling markdown table into a list of rows (the separator line is dropped)."""
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|") or _SEPARATOR_ROW.match(line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def _convert(pdf_path: str) -> dict[int, list[dict]]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # the PDF has a real text layer; OCR would only add errors on Devanagari
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    print("Docling: converting (first run downloads models, this can take several minutes)...")
    doc = conv.convert(pdf_path).document

    tables: dict[int, list[dict]] = {}
    for item, _level in doc.iterate_items():
        if type(item).__name__ != "TableItem" or not item.prov:
            continue
        prov = item.prov[0]
        try:
            md = item.export_to_markdown(doc=doc)
        except TypeError:  # older Docling versions take no argument
            md = item.export_to_markdown()
        rows = _markdown_rows(md)
        if not rows:
            continue
        bbox = None
        try:  # Docling's origin is bottom-left; PyMuPDF's is top-left, both in points
            page_h = doc.pages[prov.page_no].size.height
            b = prov.bbox.to_top_left_origin(page_h)
            bbox = [b.l, b.t, b.r, b.b]
        except Exception:
            pass  # without a box, body text inside this table is not excluded (see extract_pages)
        tables.setdefault(prov.page_no, []).append({"rows": rows, "bbox": bbox})
    return tables


def load_docling_tables(pdf_path: str, cache: str = CACHE) -> dict[int, list[dict]]:
    """{page number: [{"rows": [[cell, ...], ...], "bbox": [l, t, r, b] or None}, ...]}"""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    tables = _convert(pdf_path)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=1)
    found = sum(len(v) for v in tables.values())
    print(f"Docling: found {found} tables on pages {sorted(tables)}")
    return tables
