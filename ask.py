"""Query the index: retrieve top-k chunks, generate a grounded answer, print answer + citations.

    python ask.py --query "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"
    python ask.py --queries queries.json --out results.md
    python ask.py --queries queries.json --out results.md --retrieval hybrid

Retrieval modes (see retrieval.py for details):
    hybrid     (default) dense + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF);
               English questions are also translated to Hindi for the keyword side (needs the LLM;
               --no-expand-queries turns that off)
    dense      ChromaDB cosine search with the embedding model that built the index
    hybrid-m3  dense + bge-m3 sparse (lexical) scores, merged with RRF (pip install FlagEmbedding)
Optional extras, off by default: --min-score, --rerank, --mmr (--rerank lowered hit@1 here).

With --out, the log file is rewritten after every query, so a rate-limit error part-way
through never loses the answers already obtained; a failed query is logged and the run goes on.
"""
import argparse
import json
import os
import re
import sys

import llm
from hindi_rag import force_utf8_output
from retrieval import (MODES, add_retrieval_args, config_from_args, describe, load_index,
                       question_language, retrieve)

SYSTEM_PROMPT = (
    "You answer questions about a Hindi document using ONLY the numbered context passages "
    "provided. Do not use outside knowledge.\n"
    "- Answer in the language of the question (Hindi question -> Hindi answer, English question -> English answer).\n"
    "- Be concise: 1-3 sentences, with the specific fact (name, year, place) stated directly.\n"
    "- After each claim, cite the supporting passage id in square brackets, e.g. [12].\n"
    "- The user message states the required answer language. Follow it exactly, even though the passages are in Hindi "
    "(for an English answer, translate the facts from the Hindi passages).\n"
    "- Cite only with plain ASCII square brackets such as [12] or [12][27]; never use other bracket styles.\n"
    "- If the passages do not contain the answer, say so plainly instead of guessing."
)

# Said without calling the LLM when retrieval returns nothing (e.g. everything fell below --min-score).
NO_PASSAGES = {
    "Hindi": "दस्तावेज़ में इस प्रश्न से संबंधित कोई जानकारी नहीं मिली।",
    "English": "The document does not contain information relevant to this question.",
}


def build_context(hits: list[dict]) -> str:
    def head(h):
        sub = f" › {h['subsection']}" if h.get("subsection") else ""
        return f"[{h['chunk_id']}] (page {h['page']}, section: {h['section']}{sub})"
    return "\n\n".join(f"{head(h)}\n{h['text']}" for h in hits)


# gpt-oss sometimes emits 【57】 or 【5†L1-L2】 instead of [57]; cited_ids() only understands [57]
_CITE = re.compile(r"[\[【]\s*(\d+)(?:\s*†[^\]】]*)?\s*[\]】]")


def normalize_answer(text: str) -> str:
    """Rewrite citation variants to [id] and replace the non-breaking hyphen the model sometimes emits."""
    return _CITE.sub(r"[\1]", text.replace("\u2011", "-"))


def generate(query: str, hits: list[dict], llm_mode: str) -> str:
    if not hits:
        return NO_PASSAGES[question_language(query)]
    if llm_mode == "none":
        return "(LLM disabled) top passage: " + hits[0]["text"].splitlines()[0][:300]
    user = (f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {query}"
            f"\n\nAnswer language: {question_language(query)}.")
    return normalize_answer(llm.chat(SYSTEM_PROMPT, user))


def cited_ids(answer: str) -> set[int]:
    ids = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        ids.update(int(x) for x in re.findall(r"\d+", group))
    return ids


def score_label(h: dict) -> str:
    if "dense_rank" in h:  # hybrid hit: the fused score and where each retriever ranked it
        d = f"#{h['dense_rank']}" if h["dense_rank"] else "-"
        s = f"#{h['sparse_rank']}" if h["sparse_rank"] else "-"
        label = f"rrf: {h['score']} (dense {d}, {h['sparse_name']} {s})"
    else:
        label = f"score: {h['score']}"
    if "rerank" in h:
        label += f" · rerank: {h['rerank']}"
    return label


def format_result(query: str, answer: str, hits: list[dict], answerable: bool = True) -> str:
    cited = cited_ids(answer)
    note = "" if answerable else "   [not in the document: a grounded answer says so]"
    lines = [f"Query : {query}{note}", f"Answer: {answer}", "Sources:"]
    for h in hits:
        tag = "  <- cited in answer" if h["chunk_id"] in cited else ""
        lines.append(f"  - page: {h['page']} · section: \"{h['section']}\" · "
                     f"chunk_id: {h['chunk_id']} · {score_label(h)}{tag}")
    if not hits:
        lines.append("  (none: no passage passed retrieval)")
    return "\n".join(lines)


def write_log(path: str, header: str, blocks: list[str]) -> None:
    """Rewrite the whole log; os.replace keeps the previous file intact if the write is interrupted."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(header + "```text\n" + "\n\n".join(blocks) + "\n```\n")
    os.replace(tmp, path)


def main() -> int:
    force_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query")
    ap.add_argument("--queries", help="JSON file of queries (see queries.json)")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    ap.add_argument("--retrieval", choices=MODES, default="hybrid",
                    help="hybrid (default) = vector + BM25 (RRF); dense = vector search only; "
                         "hybrid-m3 = vector + bge-m3 sparse (RRF)")
    add_retrieval_args(ap, expand_default=True)
    ap.add_argument("--page", type=int, help="restrict retrieval to one PDF page (metadata filter)")
    ap.add_argument("--with-quiz", action="store_true",
                    help="also search the document's quiz page (excluded by default)")
    ap.add_argument("--llm", choices=["groq", "none"], default="groq" if llm.have_key() else "none",
                    help="'none' prints retrieval only (default when GROQ_API_KEY is unset)")
    ap.add_argument("--out", help="also write the log to this markdown file (updated after every query)")
    args = ap.parse_args()

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [(q["query"], q.get("answerable", True)) for q in json.load(f)]
    elif args.query:
        queries = [(args.query, True)]
    else:
        ap.error("give --query or --queries")

    cfg = config_from_args(args, args.retrieval)
    coll, emb = load_index(args.db)
    header = (f"# Test query results\n\nembedding model: `{emb.name}` · retrieval: `{describe(cfg)}`"
              f" · llm: `{args.llm}` · top-k: {args.k}\n\n")
    blocks, failures = [], 0
    for q, answerable in queries:
        hits, answer = [], ""
        try:
            hits = retrieve(coll, emb, q, args.k, args.page, args.with_quiz, cfg=cfg)
            answer = generate(q, hits, args.llm)
        except Exception as e:  # one failed query (rate limit, network) must not lose the others
            failures += 1
            answer = f"[ERROR {type(e).__name__}: {e}]"
        block = format_result(q, answer, hits, answerable)
        print(block + "\n", flush=True)
        blocks.append(block)
        if args.out:
            write_log(args.out, header, blocks)

    if failures:
        print(f"{failures} of {len(queries)} queries failed; see [ERROR ...] in the output.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
