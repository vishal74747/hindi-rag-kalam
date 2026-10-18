# Hindi RAG: Q&A over *अग्नि की उड़ान* (A.P.J. Abdul Kalam study aid)

Answers Hindi and English questions from the supplied PDF. Every answer carries a citation
(page, section, chunk id, score) read from the ChromaDB metadata, not written by hand.

```
PDF -> extract (PyMuPDF) -> clean (NFC + Devanagari repair) -> sentence-aware chunks
    -> BAAI/bge-m3 embeddings -> ChromaDB
question -> same embedder -> hybrid retrieval (dense + BM25, RRF) -> Groq LLM -> answer + sources
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=...        # free key, console.groq.com. PowerShell: $env:GROQ_API_KEY = "..."  (never commit it)

python ingest.py               # reads data/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf (or --pdf PATH); builds chroma_db/ + chunks.jsonl
python ask.py --queries queries.json --out results.md    # answers + citations for all queries -> results.md
python eval_retrieval.py --retrieval both                # is the right chunk in the top-k? (no LLM)
python eval_answers.py                                   # checks the answers in results.md
python lint_results.py                                   # flags typos / dropped spaces in the answers
python ask.py --query "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"   # one question
pip install -r requirements-dev.txt && python -m pytest tests    # unit tests
```
The first `ingest.py` downloads bge-m3 (~2 GB). Without `GROQ_API_KEY`, `ask.py` prints retrieval only.
**`results.md` is the required query/answer/citation log** (the 6 required queries are the first six).

## 1. Chunking
~450 characters (3-4 sentences) with a 1-sentence overlap. The document is fact-dense (a year, a name,
a place per sentence), so small chunks keep each embedding focused and each citation precise; the overlap
keeps pronouns ("उन्होंने …") attached to their subject. Sentences split on the danda `।` (and `?` `!`),
never on `.` ("ए.पी.जे."). Chunks never cross a page or sub-heading, so the page number is exact.
Tables are read cell by cell and stored as `header: value | header: value` rows (with the sentence that
introduces them). The section title is prepended to the text that is embedded, not to the cited text.
The PDF's own quiz page is tagged `kind="questions"` and excluded from retrieval (it repeats the user's
questions without answers). Values are reasoned from the document, not tuned on a benchmark.

## 2. Why ChromaDB
Embedded, persistent, no server: `pip install` plus one directory. Built-in metadata filtering is used for
the quiz exclusion and `--page`. At 76 chunks Qdrant's scaling strengths would be unused complexity.

## 3. Hindi handling
- **Embeddings:** `BAAI/bge-m3`, multilingual, so English queries retrieve Hindi chunks. The model name is stored in the collection, so queries can't silently use another one.
- **Normalization:** Unicode NFC; zero-width space/BOM/soft hyphen removed (ZWJ/ZWNJ kept); UTF-8 throughout, including console output on Windows.
- **PDF artefacts:** PyMuPDF returns doubled vowel signs ("उड़ाान"), stray spaces ("स्थि त") and broken table matras. Repeated vowel signs are collapsed (177 in the raw text, 0 after), a short word list and one narrow rule repair the split words, and table cells are read via clipped text. I compared Docling (`compare_extractors.py`): it left 253 stranded vowel signs and lost one answer string, so PyMuPDF is used for text and tables.
- **Segmentation:** danda-aware sentences; wrapped lines and hyphenated line breaks are rejoined.
- **Grounding:** the prompt allows only the numbered passages and asks for `[chunk_id]` citations; the answer language is decided in code from the question; the model says so when the passages lack the answer.

## 4. Results
Default retrieval is hybrid (dense + BM25 merged with Reciprocal Rank Fusion); English questions are also
translated to Hindi for the keyword side, because a keyword match cannot cross languages.
- 28 queries (6 required + 22 added: Hindi/English, table, multi-fact, false-premise, 6 unanswerable): **all 28 pass `eval_answers.py`** (language, expected facts, citation really supports the answer, no invented numbers, correct refusals).
- Retrieval, 20 answerable queries: hit@1/3/5 = 17/20, 19/20, 19/20 for both dense and hybrid. Hybrid did not beat dense on retrieval alone; the expansion is what fixes the English Padma Bhushan question at answer level.
- Measured and left **off** because they did not help: cross-encoder re-ranker, MMR, bge-m3 sparse scores (`hybrid-m3`), similarity threshold. Details and numbers: `EXPERIMENTS.md`.

## 5. What I would improve
- A larger hand-labelled query set; 28 queries is too few to tune on, and the expansion choice was tuned on two of them.
- The glossary chunk on page 20 still takes a top-4 slot for many queries; tagging glossary tables so they rank lower unless a term is asked about is the next change I would try.
- The word-repair list and the sub-heading / table lead-in rules are specific to this PDF; another document needs its own check (or OCR).
- `eval_answers.py` uses string rules; an LLM judge or manual review is still needed for subtle faithfulness.
- Free-tier Groq rate limits: errors are retried with backoff and a failed query is logged without losing the others.

## Files
`hindi_rag.py` extraction, cleaning, chunking, embedder · `ingest.py` index builder · `retrieval.py` dense / hybrid retrieval and options · `llm.py` Groq call · `ask.py` answers + citations + log · `eval_retrieval.py`, `eval_answers.py`, `lint_results.py` evaluation · `tests/` · `queries.json` queries + expected strings · `results.md` output log · `EXPERIMENTS.md` full measurements · `compare_extractors.py`, `docling_tables.py`, `compare_report.md`, `docling_cache.json` optional PyMuPDF-vs-Docling evidence (not needed to run).

Output format (hybrid shows the fused score and each retriever's rank):
```
Query : कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?
Answer: कलाम को भारत रत्न 1997 में प्राप्त हुआ।[58][69]
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 58 · rrf: 0.0313 (dense #5, bm25 #3)  <- cited in answer
```