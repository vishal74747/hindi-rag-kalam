# Experiments and measurements (detail behind README.md)

The short README is `README.md`. This file keeps the full reasoning and every measurement: chunking details, the PyMuPDF vs Docling comparison, dense vs hybrid vs hybrid-m3, re-ranker, MMR, threshold, evaluation and tests.

A small RAG pipeline over the supplied Hindi PDF (a study aid on A.P.J. Abdul Kalam).
Hindi and English questions are answered from the document, and every answer carries
a citation (page, section, chunk id, score) taken from the vector-DB metadata.

```
PDF ──extract──► clean (NFC + artefact repair) ──► sentence-aware chunks ──► multilingual embeddings
                                                                                    │
question ─► same embedder ─► ChromaDB top-k (cosine) ─► Groq LLM (grounded, cites [chunk_id]) ─► answer + sources
                              └─ optional: BM25 / bge-m3 sparse fusion, re-ranker, MMR, similarity threshold
```

## 1. Chunking strategy
- **Size: ~450 characters (about 3-4 sentences), overlap: 1 sentence.** The document is
  fact-dense: most questions are answered by a single sentence (a year, a name, a place).
  Small chunks keep the embedding focused and the citation precise; the one-sentence
  overlap protects pronoun references ("उन्होंने …") that straddle a boundary. The values
  are reasoned from the document's structure, not tuned against a benchmark.
- **Boundaries respect Hindi:** sentences split on the danda `।` (and `?`/`!`), never on
  `.`, because of "ए.पी.जे." and similar. Paragraphs are rebuilt from the PDF's line-level
  blocks using vertical gaps. A line that continues an unfinished sentence joins the paragraph
  even when it is indented (the hanging indent of a numbered item, the wrapped title-page note),
  and a hyphen at the end of a wrapped line is joined without a space ("रिट्रीवल-ऑगमेंटेड"), so chunks
  start at sentence boundaries. (Before this fix chunk starts like "में भी आशावान बनी रहती हैं।" appeared.)
- **Chunks never cross a page**, so the page number in the metadata is exact. They also
  never cross a sub-heading.
- **Sub-headings are metadata, not body text.** A short line with no sentence punctuation
  and no dash ("इस नगर ने उन्हें क्या दिया") is a sub-heading. Left in the body it ended up
  glued to the end of the previous chunk. It is now stored as the chunk's `subsection`,
  prepended to the embedded text like the section title, and shown to the LLM as
  `section › subsection`. A heading that is the last line of a page also governs the start
  of the next page. This is a text heuristic tuned on this document (it matched the two
  leaks found in the old chunks); a different PDF may need `is_subheading` adjusted.
- **Tables** (awards, timeline, missiles, glossary, fact sheet) are read cell by cell and
  stored as `header: value | header: value` rows, so each row makes sense on its own.
  The sentence that introduces a table (a paragraph ending in `:` just above it, e.g. the
  one before the missile table) is stored **with** the table, repeated in each table chunk,
  instead of sitting in a neighbouring text chunk.
- The **section title is prepended to the text that is embedded** (not to the text that
  is shown/cited), which helps short chunks that lack context on their own.
- The document's own **quiz page** (a list of questions with no answers) is tagged
  `kind="questions"` and excluded at retrieval by default. It repeats the user's
  questions almost verbatim, so it would otherwise outrank the passages that hold the
  answers. `--with-quiz` turns it back on.

## 2. Why ChromaDB
Embedded and persistent with no server, `pip install` and one directory on disk, so the
pipeline runs from a clean clone. Metadata filtering (`where`) is built in, which I use for
the quiz exclusion and an optional `--page` filter. At ~75 chunks the scaling strengths of
Qdrant would be unused complexity; hybrid search is done in `retrieval.py` (BM25 + RRF)
rather than by the database. I would reconsider Qdrant for a large corpus.

## 3. Hindi handling
- **Embeddings:** `BAAI/bge-m3` (default), a multilingual dense retriever that maps Hindi
  and English into one space, which is what makes English queries 4-6 retrieve Hindi chunks.
  Query and documents use the same model; the model name is stored in the collection, so
  `ask.py` can't silently use a different one. For a small machine set
  `EMBED_MODEL=intfloat/multilingual-e5-base` (the `query:`/`passage:` prefixes it needs are
  applied automatically).
- **Normalisation:** Unicode NFC on documents and queries; zero-width space/BOM/soft hyphen
  removed (ZWJ/ZWNJ kept, they matter in Devanagari); UTF-8 throughout. `ask.py`,
  `eval_retrieval.py` and `eval_answers.py` force UTF-8 console output, so Hindi is not
  garbled in the Windows PowerShell terminal.
- **PDF artefacts:** PyMuPDF returns this PDF's Devanagari with doubled vowel signs
  ("उड़ाान", "अब्दुुल", "मूल्यांांकन"), occasional stray spaces ("स्थि त") and, for tables,
  scrambled matras when read with `Table.extract()`. The repair collapses repeated dependent
  vowel signs (which can never legitimately repeat), fixes a short explicit list of split
  words, repairs the `ि` + space + `यों`/`याँ` fault with a rule ("पंक्ति यों" → "पंक्तियों"),
  and reads table cells through the clipped-text path instead of `Table.extract()`.
  The plural rule is deliberately narrow: "प्रति यात्रा" and "अग्नि की" must stay two words.
- **Extractor choice, measured** (`python compare_extractors.py`, results in
  `compare_report.md`): raw PyMuPDF has 177 doubled vowel signs over the document and the
  repair brings that to 0. Docling has no doubled signs but 253 stray vowel signs after
  spaces, and it lost one of the 12 answer strings the test queries need ("शिलांग"), where
  the PyMuPDF pipeline keeps all 12. So body text stays on PyMuPDF; Docling is available only
  as an optional table reader (`ingest.py --tables docling`). pdfplumber was **not**
  benchmarked in this repo, so no claim is made about it.
  Using Docling's tables (PyMuPDF body text) lowered hit@1 on the six original queries from 6/6 to 5/6
  (Q2 fell to rank 2) on the earlier index and added several minutes of CPU conversion, so it stays off.

## 4. Retrieval options
`ask.py` defaults to hybrid retrieval with English-to-Hindi query expansion (section below shows why); `--retrieval dense --no-expand-queries` is the plain baseline, and `eval_retrieval.py` runs dense and hybrid (both without expansion) by default. The other options are opt-in.

| Option | What it does |
| --- | --- |
| `--retrieval hybrid` | dense + BM25 keyword search, merged with Reciprocal Rank Fusion |
| `--retrieval hybrid-m3` | dense + bge-m3's own sparse (lexical) weights instead of BM25 (needs `pip install FlagEmbedding`; loads bge-m3 a second time) |
| `--bm25-weight W` | weight of the keyword ranking in the fusion (1.0 = plain RRF, 0 = dense only) |
| `--expand-queries` | translate English queries to Hindi with the LLM, for the keyword side only (a keyword match cannot cross languages, so BM25 otherwise does nothing for queries 4-6) |
| `--rerank` | re-score the top 20 candidates with a cross-encoder (`BAAI/bge-reranker-v2-m3`, or `--rerank-model` / `$RERANK_MODEL`) |
| `--mmr`, `--mmr-lambda L` | choose the final k by Maximal Marginal Relevance, so overlapping neighbour chunks (the three neighbouring chunks on page 8 for Q1) do not fill the context |
| `--min-score S` | drop passages with cosine similarity below S; if nothing is left, `ask.py` says the document has no relevant passage without calling the LLM |

**What has been measured, and what has not.**
- With bge-m3, hybrid retrieval, including `--bm25-weight 0.5`, gave hit@1 5/6 with Q2's
  correct chunk at rank 2, so plain hybrid did not beat dense (the expansion measured below
  did change that). My first explanation, that hybrid
  would mainly help the Hindi queries, was wrong. The "the short intro chunk above the awards table is the cause" explanation for
  Q2 comes from an offline test with a fake vector search, so it is an inference.
- The two original questions that are not in the document (Q7, Q8) are refused correctly, each in the language
  of the question (Q7 English, Q8 Hindi): `results.md`.
- **`--min-score` cannot replace the grounded prompt here.** From `results.md`, the top
  cosine score of the unanswerable Q8 (0.553) is higher than that of the answerable Q5 (0.551)
  and Q6 (0.533), so no single threshold separates them; it would catch only Q7 (0.409).
  That is why the threshold is off by default and refusal is left to the prompt.
- **`--mmr` was measured and gives no clear gain:** on the 20 retrieval queries, dense and hybrid both
  went from hit@1/3/5 = 17/20, 19/20, 19/20 to 17/20, 18/20, 20/20 with `--mmr`. It rescued Q10 into
  the top 5 but pushed Q22 from rank 1 to 5 in hybrid, so it trades one hit for another. It stays off.
- **`hybrid-m3` was measured and is worse here:** hit@1/3/5 = 7/10, 8/10, 8/10 against 8/10, 9/10, 9/10
  for both dense and BM25 hybrid (`python eval_retrieval.py --retrieval all`, no query expansion).
  It misses Q9 (the Hindi Padma Bhushan question) in the top 5 and puts Q12 at rank 3. It also
  needs `FlagEmbedding` and a second 2 GB copy of bge-m3. It stays off; I did not measure it with
  query expansion or at answer level.
- **Measured on the real index, 28 queries** (20 answerable ones with `expect` strings for
  retrieval; all 28 for answers). Final, after the paragraph-join fix:

  | setup | hit@1 | hit@3 | hit@5 | answers passing `eval_answers.py` |
  | --- | --- | --- | --- | --- |
  | dense | 17/20 | 19/20 | 19/20 | not run |
  | hybrid (no expansion) | 17/20 | 19/20 | 19/20 | not run |
  | dense or hybrid + `--mmr` | 17/20 | 18/20 | 20/20 | not run |
  | **hybrid + `--expand-queries` (the `ask.py` default)** | not run | not run | not run | **28/28** |

  Earlier measurements on the first 16 queries (10 with `expect` strings), before the 12 extra
  queries and the paragraph-join fix, are kept for the record: dense 8/10, 9/10, 9/10 with 14/16
  answers passing (Q10, Q13 failed); dense + `--rerank` 7/10 hit@1 and 10/10 hit@5; hybrid +
  `--expand-queries` 7/10, 10/10, 10/10 with 16/16 answers. `eval_retrieval.py` runs without query
  expansion by default, so the retrieval rows above do not include it.

  The English question "In which year did Kalam receive the Padma Bhushan?" (Q10) could not
  find the Hindi awards row with dense or plain hybrid retrieval (not in the top 5): the
  keyword side had no Hindi term to match. Translating English queries to Hindi for the
  keyword side puts it at rank 2 and the answer becomes correct. **Hybrid with expansion is
  therefore the default in `ask.py` and `eval_answers.py`** (`--retrieval dense
  --no-expand-queries` gives the old baseline). `eval_retrieval.py` runs dense and hybrid, both without
  expansion, so its numbers stay baselines. Expansion needs `GROQ_API_KEY`; without a key it is
  skipped with a note and Q10-type questions fail again.
- **Caveats.** Twenty-eight queries is still a small sample, and Q10/Q13 were the cases I tuned the
  expansion choice on, so the pass rate is partly in-sample (the 12 added queries were written
  after that choice and all passed). On the first 16 queries hit@1 fell from 8/10 to 7/10 with expansion (Q2, Q10, Q12 sat at
  rank 2); the generator reads 4 chunks, so hit@3 matters more here, but that is a judgement.
  `--rerank` **lowered** hit@1 on this document (7/10 vs 8/10; it moved the awards chunk from
  rank 3 to 5 for Q9) and stays off. `--mmr` is measured above and also stays off.
- **Chunk ids** shift by one after the title page because the paragraph-join fix (section 1) turned
  the title page into three chunks; the current `results.md` uses the new ids.
- **Known retrieval noise:** the glossary chunk on page 20 still takes a top-4 slot for many
  queries (7 of the first 16 with dense) and is sometimes the cited source (Q2, Q11) even though the
  awards chunks hold the answer. Overlapping neighbours (three on page 8 for Q1) share sentences.

## 5. Evaluation and tests
- `eval_retrieval.py`: hit@1/3/5, is a chunk that contains the answer strings in the top-k?
  Substring-based; no LLM. It first validates `queries.json`: every `expect` set must occur in
  one chunk, and no `absent` string of an unanswerable query may occur in any chunk, so a bad
  question is reported as a bad question, not as a retrieval miss. Queries without `expect`
  (multi-fact ones, checked only on their answer) are skipped.
- `eval_answers.py`: checks the **answers**, with deterministic rules and no LLM judge:
  answer language matches the question; the facts in `answer_must` appear; the answer cites a
  chunk, every cited id was actually retrieved, and a cited chunk contains the `expect`
  strings; every number in the answer appears in a cited chunk; the six unanswerable queries
  are refused. Numbers quoted from the question itself are exempt (so correcting a false
  premise is allowed), `answer_must_not` forbids strings, and a bare "नहीं है" no longer counts as a refusal. Queries carry a `type`
  and the summary is broken down by type. `python eval_answers.py` checks the existing `results.md` offline;
  `--live` re-runs retrieval and generation first. It exits non-zero on any failure.
  It cannot judge wording or subtle misreadings; a human read of `results.md` is still needed.
- `pytest` (`pip install -r requirements-dev.txt`): 55 offline tests (plus 3 regression tests that run only when the PDF is in `data/`) for `clean_text`, the
  BM25 tokenizer, citation parsing, sub-heading and table lead-in handling (including on a
  synthetic PDF), every retrieval option, the empty-retrieval answer, the Groq retry logic,
  the answer checker (fed deliberately bad answers), and the log being written after each query. No model download or API key is needed.

## 6. Limitations and what I would do next
- Tune chunk size/overlap, `k`, the MMR lambda and the re-ranker on a larger hand-labelled
  query set; twenty-eight queries are still too few to tune on.
- Measure `hybrid-m3` with query expansion, and the default setup's hit@k with expansion on (section 4).
- Page 15 of the PDF itself reads "… निधन हो गया। कि उन्होंने अपने अंतिम क्षण …" (checked against the
  rendered page), so the chunk that starts with "कि" reproduces the source, it is not an extraction error.
- The word-repair list and the sub-heading / table lead-in heuristics are specific to this
  document; a different PDF would need its own check (or a different extractor / OCR pass).
- `eval_answers.py` checks facts and grounding with string rules; an LLM-as-judge or a manual
  review would be needed for faithfulness beyond that.
- Free-tier Groq has rate limits and the hosted model catalogue changes; set `GROQ_MODEL`
  if the default is retired. Rate-limit errors are retried with backoff, and a query that
  still fails is logged as `[ERROR ...]` without losing the other answers.

## 7. How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=...            # free key from console.groq.com; never commit it
# optional: export GROQ_MODEL=...  (default openai/gpt-oss-120b)

python ingest.py                   # builds ./chroma_db and chunks.jsonl (downloads bge-m3, ~2 GB, first time)
python eval_retrieval.py           # hit@1/3/5 on the answerable queries with `expect` (no LLM needed)
python ask.py --queries queries.json --out results.md    # answers + citations for all 28 queries
python eval_answers.py             # checks those answers against queries.json
python ask.py --query "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"   # single question

python ask.py --queries queries.json --retrieval dense --no-expand-queries --out results_dense.md   # baseline
python ask.py --queries queries.json --rerank --mmr --out results_opts.md                           # options on top of the default
python -m pytest tests             # needs requirements-dev.txt
```
Google Colab: upload the folder, `!pip install -r requirements.txt`, set
`os.environ["GROQ_API_KEY"]` (Colab Secrets or a hidden prompt), then run the same commands
with `!python ...`.

Without `GROQ_API_KEY`, `ask.py` runs in retrieval-only mode (`--llm none`).
`EMBED_MODEL=hash` selects a tiny offline character-n-gram embedder used only to
smoke-test the plumbing (and by the tests) without downloading a model; it is not cross-lingual.

### Files
`hindi_rag.py` extraction, cleaning, chunking, embedder · `ingest.py` index builder ·
`retrieval.py` dense / hybrid retrieval and the optional re-ranker, MMR, threshold, query expansion ·
`llm.py` the Groq call (with retry) · `ask.py` generation + citations + log ·
`eval_retrieval.py` hit@k · `eval_answers.py` answer checks · `tests/` unit tests ·
`queries.json` the test queries (+ expected strings) · `results.md` produced by `ask.py --out` ·
`compare_extractors.py`, `docling_tables.py`, `compare_report.md`, `docling_cache.json`
the PyMuPDF-vs-Docling comparison and the optional table reader.

### Output format
```
Query : कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?
Answer: <answer with [chunk_id] markers>
Sources:
  - page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 57 · score: 0.83  <- cited in answer
```
Citations are generated from the stored metadata, not written by hand. Sources the answer
actually cites are flagged. Hybrid sources show `rrf: … (dense #1, bm25 #2)`, and re-ranked
ones add `rerank: …`. The first line of the log records the embedding model, the retrieval
options and the LLM used. `--out` rewrites the log after every query.
