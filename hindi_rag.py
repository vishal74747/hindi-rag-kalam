"""Shared helpers for the Hindi RAG pipeline.

PDF extraction -> Devanagari clean-up -> sentence/paragraph-aware chunking -> embeddings.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pymupdf

DEFAULT_PDF = "data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
DEFAULT_MODEL = "BAAI/bge-m3"
COLLECTION = "kalam_hindi"

def force_utf8_output() -> None:
    """Make print() safe for Hindi on Windows consoles (PowerShell/cmd default to a legacy code page).
    The files on disk are always UTF-8; this only affects what the terminal shows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Devanagari clean-up
# --------------------------------------------------------------------------
# PyMuPDF returns this PDF's Devanagari with *duplicated* vowel signs
# (e.g. "उड़ाान" for "उड़ान", "अब्दुुल" for "अब्दुल"). A dependent vowel sign,
# anusvara/visarga or nukta can never legitimately repeat, so runs are collapsed.
_DUP_MARKS = re.compile(r"([ँ-ः़ा-ौ])\1+")
# ...and the same doubling can hit a vowel-sign + anusvara pair ("मूल्यांांकन")
_DUP_PAIRS = re.compile(r"([ा-ौ][ँ-ः])\1+")
_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_CLOSE = re.compile(r"\s+(?=[)\]।,;:?!])")
_HEADING = re.compile(r"^(\d{1,2})\s*·\s*(.+)$")
# Split on the Hindi danda and ? ! only. A plain "." is NOT a boundary because
# of abbreviations like "ए.पी.जे." and "Wings of Fire".
_SENT_END = re.compile(r"(?<=[।?!])\s+")
LINE_GAP_MAX = 6.5  # points; see extract_pages
_SENT_FINAL = tuple("।?!:;.\"”’)…")  # a paragraph ending in one of these is finished


# The extractor also leaves a stray space after a mid-word "ि" in a handful of
# words ("स्थि त", "मुस्लि म"). A general rule would break real word endings
# (अग्नि की, शक्ति का), so only the words observed in this document are repaired.
_SPLIT_WORDS = {
    "स्थि ": "स्थि",              # स्थित, स्थिति, स्थिर, उपस्थित ...
    "मुस्लि म": "मुस्लिम",
    "निष्क्रि य": "निष्क्रिय",
    "व्यक्ति गत": "व्यक्तिगत",
    "व्यक्ति यों": "व्यक्तियों",
    "शक्ति याँ": "शक्तियाँ",
    "दृष्टि कोण": "दृष्टिकोण",
}


# The same fault before the plural endings -यों / -याँ ("पंक्ति यों", "विद्यार्थि यों"). Unlike the
# general case this is safe to repair with a rule: no real word starts with "यों"/"याँ" right
# after a word ending in "ि" (note "प्रति यात्रा" must NOT match, so only these two endings).
_SPLIT_PLURAL = re.compile(r"ि (?=(?:यों|याँ)(?![\u0900-\u097F]))")


def clean_text(s: str) -> str:
    """NFC-normalise, drop invisible junk, collapse extraction artefacts."""
    s = unicodedata.normalize("NFC", s)
    # zero-width space, BOM, soft hyphen (keep ZWJ/ZWNJ: they matter in Devanagari)
    s = s.replace("​", "").replace("﻿", "").replace("­", "")
    s = _DUP_MARKS.sub(r"\1", s)
    s = _DUP_PAIRS.sub(r"\1", s)
    s = _SPACE.sub(" ", s).strip()
    for broken, fixed in _SPLIT_WORDS.items():
        s = s.replace(broken, fixed)
    s = _SPLIT_PLURAL.sub("ि", s)
    return _SPACE_BEFORE_CLOSE.sub("", s)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
@dataclass
class Page:
    number: int                      # 1-based PDF page number
    section_no: int                  # 0 = title page
    section: str
    paragraphs: list[str] = field(default_factory=list)
    tables: list[list[str]] = field(default_factory=list)  # each table -> serialised rows
    # lead-in sentence of each table (same order as `tables`, "" if none): the sentence ending in
    # ":" just above it. It is removed from `paragraphs` and re-attached to the table's chunks.
    table_intros: list[str] = field(default_factory=list)


def _overlap_area(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(w, 0.0) * max(h, 0.0)


def serialise_table(rows: list[list]) -> list[str]:
    """Turn a table into 'header: value | header: value' lines so each row is
    readable on its own (and embeds meaningfully)."""
    rows = [[clean_text(c or "") or "—" for c in r] for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return [" | ".join(r) for r in rows]
    header, body = rows[0], rows[1:]
    return [" | ".join(f"{h}: {v}" for h, v in zip(header, r)) for r in body]


def read_table(page: pymupdf.Page, table) -> list[list[str]]:
    """Read a table cell by cell with the same text path used for paragraphs.
    `table.extract()` mis-orders Devanagari vowel signs ("भषू ण"), whereas
    clipped `get_text` returns correct text."""
    rows = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            cells.append(page.get_text("text", clip=pymupdf.Rect(bbox)) if bbox else "")
        rows.append(cells)
    return rows


def _pull_table_intros(paras: list[str], ys: list[float], tops: list) -> list[str]:
    """For each table, find its lead-in: the sentence ending in ":" that sits above it.

    `paras`/`ys` are the page's paragraphs and their bottom edges, `tops` the tables' top edges
    (None when the extractor gave no geometry). A paragraph ending in ":" is paired with the first
    table below it. The sentence is removed from `paras` in place (the rest of a longer paragraph
    stays) and returned, so it can be stored with the table rather than as orphan body text.
    """
    intros = [""] * len(tops)
    for i in range(len(paras) - 1, -1, -1):  # backwards, so deleting keeps lower indices valid
        if not paras[i].rstrip().endswith(":"):
            continue
        if all(t is None for t in tops):
            below = [j for j in range(len(tops)) if not intros[j]] if len(tops) == 1 else []
        else:
            below = [j for j, t in sorted(enumerate(tops), key=lambda jt: jt[1] if jt[1] is not None else 1e9)
                     if t is not None and t >= ys[i] - 1 and not intros[j]]
        if not below:
            continue
        sentences = [x for x in _SENT_END.split(paras[i]) if x]
        intros[below[0]] = sentences[-1].strip()
        if len(sentences) > 1:
            paras[i] = " ".join(sentences[:-1])
        else:
            del paras[i], ys[i]
    return intros


def extract_pages(pdf_path: str = DEFAULT_PDF, tables: str = "pymupdf") -> list[Page]:
    """Body text always comes from PyMuPDF. `tables` picks who reads the tables:
    "pymupdf" (default) or "docling" (falls back to PyMuPDF on any page where Docling finds none)."""
    dl: dict = {}
    if tables == "docling":
        from docling_tables import load_docling_tables  # lazy: docling is optional
        dl = load_docling_tables(pdf_path)

    doc = pymupdf.open(pdf_path)
    pages: list[Page] = []
    section_no, section = 0, "शीर्षक पृष्ठ"
    for page in doc:
        found = page.find_tables().tables
        boxes = [pymupdf.Rect(t.bbox) for t in found]

        entries = dl.get(page.number + 1, [])
        if entries:
            # Docling's tables replace PyMuPDF's on this page. Its boxes are added to the
            # exclusion zones so table text is never also indexed as body text.
            boxes += [pymupdf.Rect(e["bbox"]) for e in entries if e.get("bbox")]
            page_tables = [serialise_table(e["rows"]) for e in entries]
            table_tops = [e["bbox"][1] if e.get("bbox") else None for e in entries]
            print(f"  page {page.number + 1}: {len(entries)} table(s) read by Docling")
        else:
            page_tables = [serialise_table(read_table(page, t)) for t in found]
            table_tops = [t.bbox[1] for t in found]

        # This PDF emits every visual line as its own block, so rebuild paragraphs:
        # lines of one paragraph are ~4-5pt apart, paragraphs/bullets >= 8pt apart.
        paras: list[str] = []
        para_y1: list[float] = []  # bottom edge of each paragraph, to pair table lead-ins with tables
        prev_y1 = prev_x0 = None
        for x0, y0, x1, y1, txt, _no, btype in page.get_text("blocks", sort=True):
            if btype != 0:
                continue
            r = pymupdf.Rect(x0, y0, x1, y1)
            area = r.get_area()
            # text that lives inside a detected table is handled via the table path
            if area and any(_overlap_area(r, b) > 0.5 * area for b in boxes):
                continue
            t = clean_text(txt)
            if not t:
                continue
            # Same paragraph when the next line is close below. Normally it must also start at the same
            # left edge, but a line that continues an unfinished sentence (no final punctuation) joins
            # even when indented: that is the hanging indent of a numbered item ("4. ..." / "में भी ...")
            # and the wrapped lines of the title-page note, which were split mid-sentence before.
            close = prev_y1 is not None and bool(paras) and y0 - prev_y1 < LINE_GAP_MAX
            same_para = close and (abs(x0 - prev_x0) < 3 or not paras[-1].endswith(_SENT_FINAL))
            if same_para:
                # a hyphen at the end of a wrapped line belongs to the word: no space after it
                paras[-1] += t if paras[-1].endswith("-") else " " + t
                para_y1[-1] = y1
            else:
                paras.append(t)
                para_y1.append(y1)
            prev_y1, prev_x0 = y1, x0

        if not paras and not page_tables:
            continue
        if paras and paras[0].startswith("विषय-सूची"):  # table of contents: pure noise for retrieval
            continue

        m = _HEADING.match(paras[0]) if paras else None
        if m:
            section_no, section = int(m.group(1)), m.group(2).strip()
            paras, para_y1 = paras[1:], para_y1[1:]  # heading is metadata, not body text

        intros = _pull_table_intros(paras, para_y1, table_tops)
        for intro in filter(None, intros):
            print(f"  page {page.number + 1}: table lead-in kept with its table: {intro[:60]}...")

        pages.append(Page(
            number=page.number + 1,
            section_no=section_no,
            section=section,
            paragraphs=paras,
            tables=page_tables,
            table_intros=intros,
        ))
    return pages


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    id: int
    page: int
    section_no: int
    section: str
    kind: str          # "text" | "table" | "questions"
    text: str
    subsection: str = ""   # the sub-heading above this chunk, if the page has one

    @property
    def embed_text(self) -> str:
        # The section title (and sub-heading) is prepended for embedding only (contextual chunk),
        # while `text` stays the clean passage that is shown and cited.
        head = f"{self.section}\n{self.subsection}" if self.subsection else self.section
        return f"{head}\n{self.text}"


# A sub-heading is a short, unpunctuated line of its own ("इस नगर ने उन्हें क्या दिया"). Left in
# the body it ends up glued to the end of a chunk. Body sentences end in a danda or "?", bullets
# contain a dash, so a short line with neither is a heading. The title page (section 0) is exempt.
SUBHEAD_MAX_CHARS = 60
SUBHEAD_MAX_WORDS = 10
_NOT_HEADING_END = tuple("।?!:;,.\"”’)")


def is_subheading(para: str) -> bool:
    p = para.strip()
    return (0 < len(p) <= SUBHEAD_MAX_CHARS and len(p.split()) <= SUBHEAD_MAX_WORDS
            and not p.endswith(_NOT_HEADING_END) and not any(c in p for c in "—|·"))


def pack(units: list[str], max_chars: int, overlap_units: int) -> list[list[str]]:
    """Greedily pack units (sentences / table rows) into chunks of <= max_chars,
    repeating the last `overlap_units` units at the start of the next chunk."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap_units:] if overlap_units else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(pages: list[Page], max_chars: int = 450, overlap: int = 1) -> list[Chunk]:
    """Chunks never cross a page or a sub-heading. Sub-headings are stored as metadata, not body
    text, and a table's lead-in sentence is stored with the table."""
    chunks: list[Chunk] = []

    def add(pg: Page, kind: str, text: str, sub: str = "") -> None:
        chunks.append(Chunk(len(chunks), pg.number, pg.section_no, pg.section, kind, text, sub))

    carry_section, carry_sub = None, ""
    for pg in pages:
        # split the page into segments, one per sub-heading. A heading that is the last thing on
        # a page still governs the start of the next page (same section), so it is carried over.
        sub = carry_sub if carry_section == pg.section_no else ""
        segments: list[tuple[str, list[str]]] = [(sub, [])]
        heading_is_last = False
        for para in pg.paragraphs:
            if pg.section_no > 0 and is_subheading(para):
                sub = para.strip()
                segments.append((sub, []))
                heading_is_last = True
                continue
            heading_is_last = False
            segments[-1][1].extend(s for s in _SENT_END.split(para) if s)
        carry_section, carry_sub = pg.section_no, (sub if heading_is_last else "")

        # A page that is mostly a list of questions (the document's own quiz page)
        # matches user queries almost word for word but holds no answers, so its
        # chunks are tagged and excluded from retrieval by default (see ask.retrieve).
        is_quiz = sum(p.count("?") for p in pg.paragraphs) >= 5
        for sub, sentences in segments:
            for group in pack(sentences, max_chars, overlap):
                add(pg, "questions" if is_quiz else "text", " ".join(group), sub)

        intros = list(pg.table_intros) + [""] * (len(pg.tables) - len(pg.table_intros))
        for rows, intro in zip(pg.tables, intros):
            budget = max(max_chars - (len(intro) + 1 if intro else 0), 120)
            for group in pack(rows, budget, 0):
                body = "\n".join(group)
                add(pg, "table", f"{intro}\n{body}" if intro else body)  # lead-in repeated per chunk
    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def _hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    """Character n-gram hashing. Offline *smoke-test* stand-in only
    (EMBED_MODEL=hash): it checks the plumbing, it is not cross-lingual."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = f" {t} "
        for n in (2, 3, 4):
            for j in range(len(t) - n + 1):
                h = int(hashlib.md5(t[j:j + n].encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


class Embedder:
    """Thin wrapper over a multilingual sentence-transformers model."""

    def __init__(self, name: str | None = None):
        self.name = name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
        self.is_e5 = "e5" in self.name.lower()  # e5 models need query:/passage: prefixes
        self.model = None
        if self.name != "hash":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
            self.model.max_seq_length = 512

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return _hash_embed(texts)
        return self.model.encode(texts, normalize_embeddings=True, batch_size=16,
                                 show_progress_bar=False)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts] if self.is_e5 else texts)

    def encode_query(self, query: str) -> np.ndarray:
        q = clean_text(query)  # same Unicode normalisation as the documents
        return self._encode([f"query: {q}" if self.is_e5 else q])[0]