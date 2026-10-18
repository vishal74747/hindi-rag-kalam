"""Retrieval modes and options, on a tiny index built with the offline hash embedder.

The hash embedder is not cross-lingual, so these tests check the plumbing (filters, fusion,
threshold, MMR, re-ranking, query expansion) rather than retrieval quality. Real quality is
measured by eval_retrieval.py against the bge-m3 index.
"""
import chromadb
import pytest

import retrieval
from hindi_rag import COLLECTION, Embedder
from retrieval import Config, load_index, retrieve

CORPUS = [  # (text, kind)
    ("कलाम को भारत रत्न 1997 में प्रदान किया गया।", "text"),                                    # 0
    ("कलाम को भारत रत्न 1997 में प्रदान किया गया। यह सर्वोच्च नागरिक सम्मान है।", "text"),      # 1 near-copy of 0
    ("SLV-III ने रोहिणी उपग्रह को 1980 में कक्षा में स्थापित किया।", "text"),                    # 2
    ("27 जुलाई 2015 को कलाम का निधन शिलांग में हुआ।", "text"),                                  # 3
    ("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "questions"),                                    # 4 quiz page
]


@pytest.fixture
def index(tmp_path):
    emb = Embedder("hash")
    coll = chromadb.PersistentClient(path=str(tmp_path / "db")).create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": "hash"})
    texts = [t for t, _ in CORPUS]
    coll.add(ids=[str(i) for i in range(len(texts))], documents=texts,
             embeddings=emb.encode_docs(texts).tolist(),
             metadatas=[{"chunk_id": i, "page": 1 + i // 2, "section_no": 1, "section": "खंड",
                         "subsection": "", "kind": k} for i, (_, k) in enumerate(CORPUS)])
    return coll, emb


def ids(hits):
    return [h["chunk_id"] for h in hits]


def test_dense_finds_the_matching_chunk_and_skips_the_quiz(index):
    coll, emb = index
    hits = retrieve(coll, emb, "SLV-III रोहिणी उपग्रह 1980", 3)
    assert ids(hits)[0] == 2
    assert 4 not in ids(hits)                                       # quiz chunk excluded by default
    assert 4 in ids(retrieve(coll, emb, "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", 5, with_quiz=True))


def test_page_filter(index):
    coll, emb = index
    assert {h["page"] for h in retrieve(coll, emb, "कलाम", 5, page=2)} == {2}


def test_hybrid_reports_both_rankings(index):
    coll, emb = index
    hits = retrieve(coll, emb, "शिलांग निधन", 3, cfg=Config(mode="hybrid"))
    top = hits[0]
    assert top["chunk_id"] == 3 and top["dense_rank"] and top["sparse_rank"] == 1
    assert top["sparse_name"] == "bm25"


def test_hybrid_weight_zero_equals_dense_order(index):
    coll, emb = index
    q = "भारत रत्न सम्मान"
    dense = ids(retrieve(coll, emb, q, 3))
    assert ids(retrieve(coll, emb, q, 3, cfg=Config(mode="hybrid", bm25_weight=0.0))) == dense


def test_min_score_drops_weak_matches_and_can_return_nothing(index):
    coll, emb = index
    assert retrieve(coll, emb, "भारत रत्न 1997", 4, cfg=Config(min_score=0.99)) == []
    kept = retrieve(coll, emb, "भारत रत्न 1997", 4, cfg=Config(min_score=0.5))
    assert kept and all(h["cosine"] >= 0.5 for h in kept)


def test_mmr_replaces_a_near_duplicate_with_a_different_chunk(index):
    coll, emb = index
    q = "भारत रत्न 1997"
    plain = set(ids(retrieve(coll, emb, q, 2)))
    assert plain == {0, 1}                                           # the two overlapping chunks
    diverse = set(ids(retrieve(coll, emb, q, 2, cfg=Config(mmr=True, mmr_lambda=0.3))))
    assert len(diverse & {0, 1}) == 1 and len(diverse) == 2
    # lambda = 1 is pure relevance, i.e. the plain order
    assert ids(retrieve(coll, emb, q, 2, cfg=Config(mmr=True, mmr_lambda=1.0))) == ids(retrieve(coll, emb, q, 2))


def test_rerank_reorders_candidates(index, monkeypatch):
    class Fake:
        def predict(self, pairs):
            return [1.0 if "रोहिणी" in text else 0.0 for _, text in pairs]
    monkeypatch.setattr(retrieval, "_cross_encoder", lambda name: Fake())
    coll, emb = index
    hits = retrieve(coll, emb, "भारत रत्न 1997", 3, cfg=Config(rerank=True))
    assert hits[0]["chunk_id"] == 2 and hits[0]["rerank"] == 1.0


def test_query_expansion_gives_the_keyword_side_hindi_terms(index, monkeypatch):
    monkeypatch.setattr(retrieval.llm, "chat", lambda system, user, max_tokens=0: "कलाम का निधन कहाँ हुआ")
    retrieval._EXPANSIONS.clear()
    coll, emb = index
    q = "Where did he die?"
    plain = retrieve(coll, emb, q, 4, cfg=Config(mode="hybrid"))
    assert all(h["sparse_rank"] is None for h in plain)            # English shares no word with the Hindi text
    hits = retrieve(coll, emb, q, 4, cfg=Config(mode="hybrid", expand=True, llm="groq"))
    assert next(h for h in hits if h["chunk_id"] == 3)["sparse_rank"] == 1


def test_expansion_is_skipped_without_an_llm_or_for_hindi(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the LLM must not be called")
    monkeypatch.setattr(retrieval.llm, "chat", boom)
    assert retrieval.expand_query("Where did he die?", "none") == "Where did he die?"
    assert retrieval.expand_query("कलाम का निधन कहाँ हुआ?", "groq") == "कलाम का निधन कहाँ हुआ?"


def test_hybrid_m3_uses_the_sparse_scores(index, monkeypatch):
    class FakeM3:  # lexical weights = token -> 1.0; score = number of shared tokens
        def encode(self, texts, **kw):
            return {"lexical_weights": [{t: 1.0 for t in retrieval.tokenize(x)} for x in texts]}

        def compute_lexical_matching_score(self, a, b):
            return float(sum(a[t] * b[t] for t in a if t in b))
    monkeypatch.setattr(retrieval, "_m3_model", lambda: FakeM3())
    retrieval._M3.clear()
    coll, emb = index
    top = retrieve(coll, emb, "शिलांग निधन", 2, cfg=Config(mode="hybrid-m3"))[0]
    assert top["chunk_id"] == 3 and top["sparse_name"] == "m3" and top["sparse_rank"] == 1


def test_unknown_mode_is_rejected(index):
    coll, emb = index
    with pytest.raises(ValueError):
        retrieve(coll, emb, "x", 1, cfg=Config(mode="nope"))


def test_load_index_reads_the_embedder_name_from_the_index(index, tmp_path):
    coll, emb = load_index(str(tmp_path / "db"))
    assert emb.name == "hash" and coll.count() == len(CORPUS)
