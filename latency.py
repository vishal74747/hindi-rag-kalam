"""Latency check: how long does retrieval take, and how long until the first answer token?

    python latency.py                        # retrieval timings + Groq streaming TTFT (needs GROQ_API_KEY)
    python latency.py --llm none             # retrieval timings only, no API calls
    python latency.py --llm-queries 10       # stream the first 10 queries instead of the first 6
    GROQ_MODEL=llama-3.1-8b-instant python latency.py    # compare another model

What is measured (wall-clock, milliseconds, this machine and this network):
  retrieval   retrieve() for every answerable query, after one untimed warm-up call, for
              dense, hybrid, and hybrid + query expansion (expansion makes an LLM call for English queries)
  llm ttft    Groq streaming request: time to the first ANSWER token (reasoning tokens of a
              reasoning model are counted separately as "first any token"), and time to the last token
  end to end  retrieval (hybrid + expansion, the ask.py default) + LLM first answer token

Start-up (importing libraries and loading bge-m3) is reported once and is not part of the per-query numbers.
The assignment portal shows a 420 ms TTFT target; the last section compares against it.
"""
import argparse
import json
import os
import statistics
import time

import llm
import retrieval
from ask import SYSTEM_PROMPT, build_context
from hindi_rag import force_utf8_output
from retrieval import Config, load_index, question_language, retrieve

TARGET_MS = 420.0


def ms(seconds: float) -> float:
    return seconds * 1000.0


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summary(label: str, values: list[float]) -> str:
    if not values:
        return f"{label:<34} (no data)"
    return (f"{label:<34} n={len(values):<3} median {statistics.median(values):7.0f} ms   "
            f"p95 {pct(values, 0.95):7.0f} ms   max {max(values):7.0f} ms")


def time_retrieval(coll, emb, items, cfg: Config, k: int) -> list[float]:
    times = []
    for it in items:
        retrieval._EXPANSIONS.clear()  # so an expansion call is timed every time, not served from cache
        t0 = time.perf_counter()
        retrieve(coll, emb, it["query"], k, cfg=cfg)
        times.append(ms(time.perf_counter() - t0))
    return times


def stream_ttft(system: str, user: str, max_tokens: int = 1500) -> dict:
    """One streaming Groq call. Returns ms to first any token, first answer token and last token."""
    import groq

    client = groq.Groq()
    model = os.environ.get("GROQ_MODEL", llm.DEFAULT_MODEL)
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens, stream=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    first_any = first_answer = None
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        now = ms(time.perf_counter() - t0)
        answer = getattr(d, "content", None)
        thinking = getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
        if (answer or thinking) and first_any is None:
            first_any = now
        if answer and first_answer is None:
            first_answer = now
    total = ms(time.perf_counter() - t0)
    return {"first_any": first_any, "first_answer": first_answer, "total": total, "model": model}


def main() -> int:
    force_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--llm", choices=["groq", "none"], default="groq" if llm.have_key() else "none")
    ap.add_argument("--llm-queries", type=int, default=6, help="how many queries to stream from the LLM")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        items = [q for q in json.load(f) if q.get("answerable", True)]

    t0 = time.perf_counter()
    coll, emb = load_index(args.db)
    startup = ms(time.perf_counter() - t0)
    retrieve(coll, emb, items[0]["query"], args.k, cfg=Config(mode="dense"))  # warm-up, not timed
    print(f"index + embedder load (once): {startup:.0f} ms   queries: {len(items)} answerable\n")

    use_llm = args.llm == "groq"
    modes = [("dense", Config(mode="dense")),
             ("hybrid", Config(mode="hybrid"))]
    if use_llm:
        modes.append(("hybrid + query expansion", Config(mode="hybrid", expand=True, llm="groq")))

    print("== retrieval only (no answer generation) ==")
    results = {}
    for label, cfg in modes:
        results[label] = time_retrieval(coll, emb, items, cfg, args.k)
        print(summary(label, results[label]))
    en = [i for i in items if question_language(i["query"]) == "English"]
    if use_llm and en:
        print(summary("  of which English queries", time_retrieval(coll, emb, en, modes[-1][1], args.k)))

    if not use_llm:
        print("\n(--llm none: no LLM timings. Set GROQ_API_KEY to measure time to first token.)")
        return 0

    print(f"\n== LLM streaming, first {args.llm_queries} queries ==")
    cfg = modes[-1][1]
    first_any, first_ans, total, e2e = [], [], [], []
    model = llm.DEFAULT_MODEL
    for it in items[:args.llm_queries]:
        retrieval._EXPANSIONS.clear()
        t0 = time.perf_counter()
        hits = retrieve(coll, emb, it["query"], args.k, cfg=cfg)
        r_ms = ms(time.perf_counter() - t0)
        user = (f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {it['query']}"
                f"\n\nAnswer language: {question_language(it['query'])}.")
        try:
            r = stream_ttft(SYSTEM_PROMPT, user)
        except Exception as e:  # rate limit etc.: report and carry on
            print(f"Q{it['id']}: skipped ({type(e).__name__}: {e})")
            continue
        model = r["model"]
        if r["first_answer"] is None:
            print(f"Q{it['id']}: no answer tokens (model spent max_tokens thinking)")
            continue
        first_any.append(r["first_any"])
        first_ans.append(r["first_answer"])
        total.append(r["total"])
        e2e.append(r_ms + r["first_answer"])
        print(f"Q{it['id']}: retrieval {r_ms:5.0f} ms | first token {r['first_any']:5.0f} ms | "
              f"first ANSWER token {r['first_answer']:5.0f} ms | done {r['total']:5.0f} ms")

    print(f"\nmodel: {model}")
    print(summary("LLM first token (any)", first_any))
    print(summary("LLM first answer token", first_ans))
    print(summary("LLM full reply", total))
    print(summary("end to end to first answer token", e2e))

    if e2e:
        med = statistics.median(e2e)
        verdict = "within" if med <= TARGET_MS else "above"
        print(f"\nmedian time to first answer token: {med:.0f} ms, {verdict} the {TARGET_MS:.0f} ms target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
