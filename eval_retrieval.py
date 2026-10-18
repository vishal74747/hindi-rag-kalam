"""Retrieval evaluation: is a chunk that actually contains the answer in the top-k?

For each query in queries.json, `expect` lists substrings that must all appear in one
chunk for it to count as the correct chunk. Reports hit@1/3/5 and the rank of the first hit.

    python eval_retrieval.py                          # dense and hybrid, with a side-by-side summary (default)
    python eval_retrieval.py --retrieval dense        # dense only
    python eval_retrieval.py --retrieval hybrid       # dense + BM25 + RRF
    python eval_retrieval.py --retrieval all          # dense, hybrid and hybrid-m3
    python eval_retrieval.py --retrieval dense --rerank --mmr    # any option from ask.py works too

Queries marked "answerable": false have no answer in the document, so there is nothing to
retrieve; they test the LLM's refusal and are read in results.md instead.
"""
import argparse
import json

from hindi_rag import clean_text, force_utf8_output
from retrieval import MODES, add_retrieval_args, config_from_args, load_index, retrieve


def evaluate(items, coll, emb, corpus, cfg) -> list:
    ranks = []
    for it in items:
        expect = [clean_text(e) for e in it["expect"]]
        if not any(all(e in doc for e in expect) for doc in corpus):
            print(f"  ! query {it['id']}: no chunk in the index contains {expect} - fix `expect`")
        hits = retrieve(coll, emb, it["query"], k=5, cfg=cfg)
        rank = next((i + 1 for i, h in enumerate(hits) if all(e in h["text"] for e in expect)), None)
        ranks.append(rank)
        print(f"Q{it['id']} [{it['lang']}] first correct chunk at rank: {rank or 'not in top-5'}")
    return ranks


def validate_queries(items, corpus) -> list[str]:
    """Problems in queries.json itself, so a typo cannot silently turn into a 'retrieval failure'.

    answerable query    every `expect` string must co-occur in at least one chunk
    unanswerable query  none of its `absent` strings may occur in any chunk (otherwise the
                        document does contain the answer and the question is not unanswerable)
    """
    problems = []
    for it in items:
        if it.get("answerable", True):
            expect = [clean_text(e) for e in it.get("expect", [])]
            if expect and not any(all(e in doc for e in expect) for doc in corpus):
                problems.append(f"Q{it['id']}: no chunk contains all of {expect}")
        else:
            for a in it.get("absent", []):
                if any(clean_text(a) in doc for doc in corpus):
                    problems.append(f"Q{it['id']}: unanswerable, but a chunk contains {a!r}")
    return problems


def hit_at(ranks: list, k: int) -> str:
    return f"{sum(1 for r in ranks if r and r <= k)}/{len(ranks)}"


def main() -> None:
    force_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--retrieval", choices=[*MODES, "both", "all"], default="both",
                    help="retrieval mode to evaluate; 'both' = dense + hybrid (default), 'all' = every mode")
    add_retrieval_args(ap)
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        all_items = json.load(f)
    # unanswerable queries have nothing to retrieve; a query with no `expect` strings (e.g. a
    # multi-fact question checked only on its answer) would count as a hit at rank 1 for any chunk
    items = [it for it in all_items if it.get("answerable", True) and it.get("expect")]
    skipped = len(all_items) - len(items)
    coll, emb = load_index(args.db)
    corpus = coll.get()["documents"]

    problems = validate_queries(all_items, corpus)
    for p in problems:
        print(f"  ! {p}")
    if problems:
        print("  fix queries.json (or drop the question) before trusting the numbers below\n")

    modes = {"both": ["dense", "hybrid"], "all": list(MODES)}.get(args.retrieval, [args.retrieval])
    results = {}
    for mode in modes:
        print(f"\n== {mode} ==")
        try:
            results[mode] = evaluate(items, coll, emb, corpus, config_from_args(args, mode))
        except SystemExit as e:  # e.g. hybrid-m3 without FlagEmbedding
            if args.retrieval != "all":
                raise  # an explicitly requested mode must fail loudly
            print(f"skipped: {e}")
            continue
        print(f"retrieval: {mode}")
        for k in (1, 3, 5):
            print(f"hit@{k}: {hit_at(results[mode], k)}")
    if skipped:
        print(f"\n({skipped} query/queries skipped: answerable=false or no `expect` strings)")
    done = list(results)  # modes that actually ran
    if len(done) > 1:
        print("\nsummary   " + "".join(f"{m:>10}" for m in done))
        for k in (1, 3, 5):
            print(f"hit@{k}     " + "".join(f"{hit_at(results[m], k):>10}" for m in done))


if __name__ == "__main__":
    main()