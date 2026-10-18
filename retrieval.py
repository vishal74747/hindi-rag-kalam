"""Retrieval: dense, hybrid (dense + keyword, fused with RRF) and optional post-processing.

Modes (cfg.mode)
    dense       ChromaDB cosine search with the embedding model that built the index (default)
    hybrid      dense + BM25, merged with Reciprocal Rank Fusion (RRF)
    hybrid-m3   dense + bge-m3's own sparse (lexical) weights, merged with RRF
                (needs `pip install FlagEmbedding`; loads bge-m3 a second time)

Optional steps, each off by default so the baseline stays reproducible:
    expand      translate English queries to Hindi with the LLM for the keyword side (hybrid modes)
    min_score   drop passages whose cosine similarity to the query is below a threshold
    rerank      re-score candidates with a cross-encoder (default BAAI/bge-reranker-v2-m3)
    mmr         pick the final k by Maximal Marginal Relevance, so near-duplicate chunks
                (overlapping neighbours) do not fill the context
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

import chromadb
import numpy as np

import llm
from hindi_rag import COLLECTION, Embedder, clean_text

RRF_K = 60   # standard RRF constant: score = sum over retrievers of 1 / (RRF_K + rank)
POOL = 20    # candidates taken from each retriever before fusion
MODES = ("dense", "hybrid", "hybrid-m3")
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Config:
    mode: str = "dense"
    bm25_weight: float = 1.0      # weight of the keyword ranking (BM25 or m3 sparse) in the fusion
    expand: bool = False          # English query -> Hindi, for the keyword side of hybrid modes
    min_score: float = 0.0        # cosine threshold; 0 = off
    rerank: bool = False
    rerank_model: str = ""        # "" -> $RERANK_MODEL or DEFAULT_RERANKER
    rerank_pool: int = 20         # candidates passed to the re-ranker
    mmr: bool = False
    mmr_lambda: float = 0.7       # 1.0 = pure relevance, lower = more diversity
    llm: str = "none"             # "groq" lets `expand` call the LLM


def describe(cfg: Config) -> str:
    """Short label for logs, e.g. 'hybrid (bm25 weight 0.5) + expand + mmr 0.7'."""
    s = cfg.mode + (f" (keyword weight {cfg.bm25_weight})" if cfg.mode != "dense" else "")
    if cfg.expand and cfg.mode != "dense" and cfg.llm == "groq":  # expansion is skipped without the LLM
        s += " + query expansion"
    if cfg.min_score:
        s += f" + min-score {cfg.min_score}"
    if cfg.rerank:
        s += " + rerank"
    if cfg.mmr:
        s += f" + mmr {cfg.mmr_lambda}"
    return s


def add_retrieval_args(ap, expand_default: bool = False) -> None:
    """The retrieval options shared by ask.py and the eval scripts.

    expand_default: whether --expand-queries is on unless --no-expand-queries is given. ask.py and
    eval_answers.py pass True (measured best on this document); eval_retrieval.py keeps False so
    its dense / hybrid numbers stay plain baselines. Expansion needs the LLM and is skipped, with
    a note, when no GROQ_API_KEY is set.
    """
    ap.add_argument("--bm25-weight", type=float, default=1.0,
                    help="hybrid modes: weight of the keyword ranking in the fusion (1.0 = plain RRF)")
    ap.add_argument("--expand-queries", action=argparse.BooleanOptionalAction, default=expand_default,
                    help="hybrid modes: translate English queries to Hindi (LLM) for the keyword side "
                         f"(default: {'on' if expand_default else 'off'})")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop passages with cosine similarity below this (0 = off)")
    ap.add_argument("--rerank", action="store_true", help="re-rank candidates with a cross-encoder")
    ap.add_argument("--rerank-model", default="", help=f"default {DEFAULT_RERANKER} (or $RERANK_MODEL)")
    ap.add_argument("--rerank-pool", type=int, default=20, help="candidates given to the re-ranker")
    ap.add_argument("--mmr", action="store_true", help="diversify the final k (drops near-duplicate chunks)")
    ap.add_argument("--mmr-lambda", type=float, default=0.7, help="1.0 = pure relevance, lower = more diverse")


def config_from_args(args, mode: str) -> Config:
    return Config(mode=mode, bm25_weight=args.bm25_weight, expand=args.expand_queries,
                  min_score=args.min_score, rerank=args.rerank, rerank_model=args.rerank_model,
                  rerank_pool=args.rerank_pool, mmr=args.mmr, mmr_lambda=args.mmr_lambda,
                  llm=getattr(args, "llm", None) or ("groq" if llm.have_key() else "none"))


# --------------------------------------------------------------------------
# Language helpers
# --------------------------------------------------------------------------
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def question_language(query: str) -> str:
    """'Hindi' if the question contains Devanagari, else 'English' (decided in code, not by the model)."""
    return "Hindi" if _DEVANAGARI.search(query) else "English"


# Python's \w does not match Devanagari vowel signs (they are combining marks), so \w+ would
# cut every Hindi word at its matras. Match the Devanagari block explicitly instead, minus the
# danda (U+0964) and double danda (U+0965), which are punctuation.
_TOKEN = re.compile(r"[\u0900-\u0963\u0966-\u097F\w]+")
_STOPWORDS = set(
    "के में की है हैं को ने से का और एक था थे थी पर यह वह भी कि जो तो क्या किस कौन कहाँ कैसे कब "
    "लिए साथ ही हुआ हुई हुए किया गया थीं ये वे इस उस".split()
) | set("the a an of in on at to did do does and is was which who what how where when for by with".split())


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (Hindi + Latin), without stopwords."""
    return [t for t in _TOKEN.findall(clean_text(text).lower()) if t not in _STOPWORDS]


# --------------------------------------------------------------------------
# Index access
# --------------------------------------------------------------------------
def load_index(db: str):
    coll = chromadb.PersistentClient(path=db).get_collection(COLLECTION)
    # query with the same model that built the index
    return coll, Embedder(coll.metadata.get("embed_model"))


def _filters(page: int | None, with_quiz: bool):
    # skip the document's quiz page (questions without answers), optionally restrict to one page
    conds = [] if with_quiz else [{"kind": {"$ne": "questions"}}]
    if page:
        conds.append({"page": page})
    return {"$and": conds} if len(conds) > 1 else (conds[0] if conds else None)


def _corpus_rows(coll, page: int | None, with_quiz: bool) -> list[tuple[dict, str]]:
    data = coll.get(include=["documents", "metadatas"])
    return [(m, t) for t, m in zip(data["documents"], data["metadatas"])
            if (with_quiz or m.get("kind") != "questions") and (not page or m.get("page") == page)]


def retrieve_dense(coll, emb: Embedder, query: str, k: int, page: int | None = None,
                   with_quiz: bool = False) -> list[dict]:
    res = coll.query(query_embeddings=[emb.encode_query(query).tolist()], n_results=k,
                     where=_filters(page, with_quiz))
    return [{**meta, "text": text, "score": round(1 - dist, 3)}  # cosine similarity
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


# --------------------------------------------------------------------------
# Keyword side: BM25 or bge-m3 sparse
# --------------------------------------------------------------------------
def _keyword_text(meta: dict, text: str) -> str:
    # section title (+ sub-heading) + passage, like the embedded text
    return f"{meta['section']} {meta.get('subsection', '')} {text}"


def bm25_scores(rows: list[tuple[dict, str]], query: str) -> list[float]:
    from rank_bm25 import BM25Okapi  # lazy: only needed for hybrid mode

    q_tokens = tokenize(query)
    if not q_tokens:
        return [0.0] * len(rows)
    bm25 = BM25Okapi([tokenize(_keyword_text(m, t)) for m, t in rows])
    return list(bm25.get_scores(q_tokens))


_M3: dict = {}


def _m3_model():
    if "model" not in _M3:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise SystemExit("--retrieval hybrid-m3 needs FlagEmbedding: pip install FlagEmbedding") from e
        _M3["model"] = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _M3["model"]


def m3_scores(rows: list[tuple[dict, str]], query: str) -> list[float]:
    """bge-m3 lexical matching score: shared tokens weighted by the model's learned term weights."""
    model = _m3_model()
    docs = _M3.setdefault("docs", {})  # passage text -> lexical weights, so each passage is encoded once
    texts = [_keyword_text(m, t) for m, t in rows]
    missing = [t for t in dict.fromkeys(texts) if t not in docs]
    if missing:
        out = model.encode(missing, batch_size=16, return_dense=False, return_sparse=True,
                           return_colbert_vecs=False)
        docs.update(zip(missing, out["lexical_weights"]))
    q = model.encode([query], return_dense=False, return_sparse=True,
                     return_colbert_vecs=False)["lexical_weights"][0]
    return [float(model.compute_lexical_matching_score(q, docs[t])) for t in texts]


# --------------------------------------------------------------------------
# Query expansion (English -> Hindi, for the keyword side)
# --------------------------------------------------------------------------
_EXPANSIONS: dict[str, str] = {}
_WARNED: set[str] = set()
_TRANSLATE_PROMPT = ("Translate the user's question into Hindi (Devanagari script). "
                     "Output only the translation, nothing else.")


def expand_query(query: str, llm_mode: str) -> str:
    """For an English query return 'query + Hindi translation'; otherwise the query unchanged.

    The dense side keeps the original query (bge-m3 is cross-lingual); this only gives BM25 /
    m3-sparse Hindi terms to match, since a keyword match cannot cross languages.
    """
    if question_language(query) == "Hindi":
        return query
    if llm_mode != "groq":
        if "expand" not in _WARNED:
            print("note: --expand-queries needs the LLM (set GROQ_API_KEY); skipping expansion.",
                  file=sys.stderr)
            _WARNED.add("expand")
        return query
    if query not in _EXPANSIONS:
        _EXPANSIONS[query] = llm.chat(_TRANSLATE_PROMPT, query, max_tokens=300)
    return f"{query} {_EXPANSIONS[query]}"


# --------------------------------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------------------------------
def retrieve_hybrid(coll, emb: Embedder, query: str, k: int, page: int | None = None,
                    with_quiz: bool = False, pool: int = POOL, bm25_weight: float = 1.0,
                    sparse: str = "bm25", sparse_query: str | None = None) -> list[dict]:
    """Dense + keyword search, fused with Reciprocal Rank Fusion. Same filters as the dense path.

    bm25_weight scales the keyword list in the fusion (1.0 = plain RRF, 0 = dense only).
    `sparse` is "bm25" or "m3"; `sparse_query` is the text the keyword side sees (default: query).
    """
    rows = _corpus_rows(coll, page, with_quiz)
    if not rows:
        return []
    by_id = {m["chunk_id"]: (m, t) for m, t in rows}
    n_pool = min(max(pool, k), len(rows))

    # dense ranking (Chroma cosine)
    dense = retrieve_dense(coll, emb, query, n_pool, page, with_quiz)
    dense_rank = {h["chunk_id"]: r for r, h in enumerate(dense, 1)}

    # keyword ranking
    scores = (m3_scores if sparse == "m3" else bm25_scores)(rows, sparse_query or query)
    sparse_rank: dict[int, int] = {}
    for i in sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:n_pool]:
        if scores[i] > 0:  # a chunk sharing no query term is not a keyword match
            sparse_rank[rows[i][0]["chunk_id"]] = len(sparse_rank) + 1

    # Reciprocal Rank Fusion
    fused: dict[int, float] = {}
    for ranks, weight in ((dense_rank, 1.0), (sparse_rank, bm25_weight)):
        for cid, r in ranks.items():
            fused[cid] = fused.get(cid, 0.0) + weight / (RRF_K + r)
    order = sorted((c for c in fused if c in by_id),
                   key=lambda c: (-fused[c], dense_rank.get(c, 10**6)))[:k]  # ties: dense rank

    return [{**by_id[cid][0], "text": by_id[cid][1], "score": round(fused[cid], 4),
             "dense_rank": dense_rank.get(cid), "sparse_rank": sparse_rank.get(cid),
             "sparse_name": "m3" if sparse == "m3" else "bm25"} for cid in order]


# --------------------------------------------------------------------------
# Post-processing: similarity threshold, re-ranking, MMR
# --------------------------------------------------------------------------
def _cosines(coll, emb: Embedder, query: str, hits: list[dict]):
    """Chunk vectors from the index, and each hit's cosine similarity to the query."""
    got = coll.get(ids=[str(h["chunk_id"]) for h in hits], include=["embeddings"])
    vecs = {int(i): np.asarray(e, dtype=np.float32) for i, e in zip(got["ids"], got["embeddings"])}
    q = emb.encode_query(query)
    return vecs, {cid: float(v @ q) for cid, v in vecs.items()}  # vectors are L2-normalised


_RERANKERS: dict = {}


def _cross_encoder(name: str):
    if name not in _RERANKERS:
        from sentence_transformers import CrossEncoder  # lazy: only for --rerank
        _RERANKERS[name] = CrossEncoder(name, max_length=512)
    return _RERANKERS[name]


def rerank_hits(query: str, hits: list[dict], model_name: str = "") -> list[dict]:
    name = model_name or os.environ.get("RERANK_MODEL", DEFAULT_RERANKER)
    scores = _cross_encoder(name).predict([(query, h["text"]) for h in hits])
    for h, s in zip(hits, scores):
        h["rerank"] = round(float(s), 3)
    return sorted(hits, key=lambda h: -h["rerank"])


def mmr_select(hits: list[dict], vecs: dict, k: int, lam: float) -> list[dict]:
    """Maximal Marginal Relevance: repeatedly take the candidate with the best
    lam * relevance - (1 - lam) * (similarity to what is already selected)."""
    rel = {h["chunk_id"]: h.get("rerank", h.get("cosine", h["score"])) for h in hits}
    lo, hi = min(rel.values()), max(rel.values())
    rel = {c: (v - lo) / (hi - lo) if hi > lo else 1.0 for c, v in rel.items()}  # to [0, 1]
    chosen: list[dict] = []
    remaining = list(hits)
    while remaining and len(chosen) < k:
        def gain(h):
            red = max((float(vecs[h["chunk_id"]] @ vecs[s["chunk_id"]]) for s in chosen), default=0.0)
            return lam * rel[h["chunk_id"]] - (1 - lam) * red
        best = max(remaining, key=gain)
        chosen.append(best)
        remaining.remove(best)
    return chosen


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def retrieve(coll, emb: Embedder, query: str, k: int, page: int | None = None,
             with_quiz: bool = False, cfg: Config | None = None) -> list[dict]:
    """Top-k passages for `query`. May return fewer than k (or none) when cfg.min_score is set."""
    cfg = cfg or Config()
    if cfg.mode not in MODES:
        raise ValueError(f"unknown retrieval mode {cfg.mode!r}; choose from {MODES}")
    n = max(k, cfg.rerank_pool) if cfg.rerank else (k * 3 if cfg.mmr else k)  # candidate pool

    if cfg.mode == "dense":
        hits = retrieve_dense(coll, emb, query, n, page, with_quiz)
    else:
        sq = expand_query(query, cfg.llm) if cfg.expand else query
        hits = retrieve_hybrid(coll, emb, query, n, page, with_quiz, pool=max(POOL, n),
                               bm25_weight=cfg.bm25_weight,
                               sparse="m3" if cfg.mode == "hybrid-m3" else "bm25", sparse_query=sq)
    if not hits:
        return hits

    vecs: dict = {}
    if cfg.min_score or cfg.mmr:
        vecs, cos = _cosines(coll, emb, query, hits)
        for h in hits:
            h["cosine"] = round(cos[h["chunk_id"]], 3)
        if cfg.min_score:
            hits = [h for h in hits if h["cosine"] >= cfg.min_score]
    if cfg.rerank and hits:
        hits = rerank_hits(query, hits, cfg.rerank_model)
    if cfg.mmr and len(hits) > 1:
        hits = mmr_select(hits, vecs, k, cfg.mmr_lambda)
    return hits[:k]