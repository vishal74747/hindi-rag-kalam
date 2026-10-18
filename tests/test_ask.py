"""ask.py: empty-retrieval answer, log format, and the save-as-you-go behaviour."""
import json
import sys

import chromadb
import pytest

import ask
from hindi_rag import COLLECTION, Embedder


def test_no_passages_answers_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(ask.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))
    assert ask.generate("कलाम कौन थे?", [], "groq") == ask.NO_PASSAGES["Hindi"]
    assert ask.generate("Who was Kalam?", [], "groq") == ask.NO_PASSAGES["English"]


def test_format_result_marks_cited_sources_and_hybrid_ranks():
    hit = {"chunk_id": 7, "page": 3, "section": "खंड", "score": 0.0328, "text": "x",
           "dense_rank": 1, "sparse_rank": None, "sparse_name": "bm25"}
    out = ask.format_result("q?", "answer [7]", [hit])
    assert "chunk_id: 7" in out and "rrf: 0.0328 (dense #1, bm25 -)" in out and "<- cited in answer" in out
    assert "(none" in ask.format_result("q?", "answer", [])


def test_build_context_shows_the_subheading():
    h = {"chunk_id": 1, "page": 2, "section": "बचपन", "subsection": "इस नगर ने क्या दिया", "text": "पाठ"}
    assert "बचपन › इस नगर ने क्या दिया" in ask.build_context([h])


@pytest.fixture
def db(tmp_path):
    emb = Embedder("hash")
    texts = ["पहला अंश कलाम", "दूसरा अंश शिलांग", "तीसरा अंश रोहिणी"]
    coll = chromadb.PersistentClient(path=str(tmp_path / "db")).create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": "hash"})
    coll.add(ids=["0", "1", "2"], documents=texts, embeddings=emb.encode_docs(texts).tolist(),
             metadatas=[{"chunk_id": i, "page": 1, "section_no": 1, "section": "खंड", "subsection": "",
                         "kind": "text"} for i in range(3)])
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps([{"query": f"प्रश्न {w}"} for w in ("कलाम", "शिलांग", "रोहिणी")],
                                ensure_ascii=False), encoding="utf-8")
    return str(tmp_path / "db"), str(qfile), str(tmp_path / "out.md")


def run(monkeypatch, db, qfile, out):
    monkeypatch.setattr(sys, "argv", ["ask.py", "--db", db, "--queries", qfile, "--out", out, "--llm", "none",
                                      "--retrieval", "dense"])  # these tests are about logging, not retrieval
    return ask.main()


def test_log_is_rewritten_after_every_query(db, monkeypatch):
    sizes = []
    real = ask.write_log
    monkeypatch.setattr(ask, "write_log", lambda path, header, blocks: (sizes.append(len(blocks)),
                                                                        real(path, header, blocks)))
    assert run(monkeypatch, *db) == 0
    assert sizes == [1, 2, 3]


def test_a_failing_query_is_logged_and_the_run_continues(db, monkeypatch, capsys):
    calls = []

    def flaky(query, hits, llm_mode):
        calls.append(query)
        if len(calls) == 2:
            raise RuntimeError("rate limit")
        return "ठीक उत्तर [0]"
    monkeypatch.setattr(ask, "generate", flaky)
    assert run(monkeypatch, *db) == 1                              # non-zero exit signals a partial run
    log = open(db[2], encoding="utf-8").read()
    assert len(calls) == 3                                          # query 3 still ran
    assert log.count("Query :") == 3 and "[ERROR RuntimeError: rate limit]" in log
    assert log.count("ठीक उत्तर") == 2                              # the answers before/after are kept
    assert "retrieval: `dense`" in log and "llm: `none`" in log
