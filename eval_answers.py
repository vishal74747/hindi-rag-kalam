"""Answer evaluation: are the generated answers correct, grounded and in the right language?

eval_retrieval.py only asks "is a chunk with the answer in the top-k?". This script checks the
answers themselves, with deterministic rules (no LLM judge):

  language   a Hindi question gets a Devanagari answer, an English question a Latin one
  facts      every group in the query's `answer_must` has at least one of its strings in the answer
  citations  the answer cites at least one chunk, all cited ids were actually retrieved, and at least
             one cited chunk contains every `expect` string (so the citation really supports it)
  numbers    every number in the answer also appears in a cited chunk (catches invented years);
             numbers quoted from the question itself are exempt, so a correction of a false
             premise ("not 2005, but 1997") is not penalised for repeating the premise
  forbidden  no string in the query's optional `answer_must_not` list appears in the answer
  refusal    for "answerable": false queries the answer says the document does not contain it

Queries may carry a `type` (single, table, multi, false_premise, unanswerable); the summary is
then also broken down per type, so one weak category is not hidden by an overall pass rate.

    python eval_answers.py                      # check the answers already in results.md (offline)
    python eval_answers.py --results other.md
    python eval_answers.py --live               # run retrieval + the LLM now, then check (needs the index and a key)
    python eval_answers.py --live --retrieval dense --no-expand-queries

Exit status is 1 if any check fails, so it can gate a submission.
"""
import argparse
import json
import re
import sys

from ask import cited_ids, generate
from hindi_rag import clean_text, force_utf8_output

_DEV = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")
_NUMBER = re.compile(r"\d+")
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_REFUSAL = re.compile(
    r"do(?:es)? not (?:mention|contain|say|specify|provide|include)|not (?:mentioned|provided|stated|found)"
    r"|no (?:information|mention)|cannot (?:be )?(?:determine|find)"
    r"|उल्लेख नहीं|जानकारी नहीं|नहीं मिल|नहीं दी गई|नहीं बताया|उपलब्ध नहीं|उल्लिखित नहीं",
    re.IGNORECASE)
# (a bare "नहीं है" is deliberately not a refusal marker: it also matches ordinary negative facts)

_BLOCK = re.compile(
    r"^Query : (?P<q>.*?)(?:   \[not in the document[^\]]*\])?\nAnswer: (?P<a>.*?)\nSources:\n"
    r"(?P<s>.*?)(?=\n\nQuery : |\n```|\Z)", re.DOTALL | re.MULTILINE)


def parse_results(path: str) -> dict[str, dict]:
    """{query: {"answer": str, "listed": [chunk ids shown under Sources]}} from an ask.py log."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return {m["q"].strip(): {"answer": m["a"].strip(),
                             "listed": [int(x) for x in re.findall(r"chunk_id: (\d+)", m["s"])]}
            for m in _BLOCK.finditer(text)}


def load_chunk_texts(path: str) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["text"] for r in map(json.loads, f)}


def language_ok(question: str, answer: str) -> bool:
    body = _CITATION.sub("", answer)
    dev, lat = len(_DEV.findall(body)), len(_LATIN.findall(body))
    if _DEV.search(question):                       # Hindi question -> mostly Devanagari
        return dev > 0 and dev >= lat
    return lat > 0 and dev <= 0.1 * (dev + lat)     # English question -> (almost) no Devanagari


def check(item: dict, answer: str, listed: list[int], texts: dict[int, str]) -> dict[str, bool | None]:
    """Run every applicable check; None means 'not applicable' for this query."""
    res: dict[str, bool | None] = {"language": language_ok(item["query"], answer)}
    if not item.get("answerable", True):
        res["refusal"] = bool(_REFUSAL.search(answer))
        return res
    flat = clean_text(answer).lower()
    res["facts"] = all(any(clean_text(alt).lower() in flat for alt in group)
                       for group in item.get("answer_must", [])) if item.get("answer_must") else None
    cited = cited_ids(answer)
    expect = [clean_text(e) for e in item.get("expect", [])]
    res["cited"] = bool(cited) and cited <= set(listed)
    res["supported"] = any(all(e in texts.get(c, "") for e in expect) for c in cited) if expect else None
    cited_text = " ".join(texts.get(c, "") for c in cited)
    body = _CITATION.sub("", answer)
    quoted = set(_NUMBER.findall(item["query"]))
    res["numbers"] = all(n in cited_text or n in quoted for n in _NUMBER.findall(body)) if cited else False
    if item.get("answer_must_not"):
        res["forbidden"] = not any(clean_text(x).lower() in flat for x in item["answer_must_not"])
    return res


def main() -> int:
    force_utf8_output()
    from retrieval import MODES, add_retrieval_args, config_from_args  # after ask: shared CLI options
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--results", default="results.md", help="ask.py log to check (offline mode)")
    ap.add_argument("--chunks", default="chunks.jsonl", help="chunk dump written by ingest.py")
    ap.add_argument("--live", action="store_true", help="retrieve and generate now instead of reading --results")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--retrieval", choices=MODES, default="hybrid")
    add_retrieval_args(ap, expand_default=True)  # same defaults as ask.py
    ap.add_argument("--llm", choices=["groq", "none"], default=None)
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        items = json.load(f)
    texts = load_chunk_texts(args.chunks)

    if args.live:
        from retrieval import load_index, retrieve
        cfg = config_from_args(args, args.retrieval)
        coll, emb = load_index(args.db)
        answers = {}
        for it in items:
            hits = retrieve(coll, emb, it["query"], args.k, cfg=cfg)
            texts.update({h["chunk_id"]: h["text"] for h in hits})
            answers[it["query"]] = {"answer": generate(it["query"], hits, cfg.llm),
                                    "listed": [h["chunk_id"] for h in hits]}
        source = f"live run ({args.retrieval})"
    else:
        answers, source = parse_results(args.results), args.results

    failed = 0
    by_type: dict[str, list[bool]] = {}
    print(f"Answer check on {source}\n")
    for it in items:
        got = answers.get(it["query"])
        if got is None:
            print(f"Q{it['id']}: MISSING from the log"); failed += 1
            by_type.setdefault(it.get("type", "single"), []).append(False); continue
        res = check(it, got["answer"], got["listed"], texts)
        bad = [name for name, ok in res.items() if ok is False]
        failed += bool(bad)
        by_type.setdefault(it.get("type") or ("single" if it.get("answerable", True) else "unanswerable"),
                           []).append(not bad)
        shown = "  ".join(f"{n}={'ok' if v else ('-' if v is None else 'FAIL')}" for n, v in res.items())
        print(f"Q{it['id']}: {'PASS' if not bad else 'FAIL'}  {shown}")
    print(f"\n{len(items) - failed}/{len(items)} queries pass every check")
    if len(by_type) > 1:
        print("by type: " + ", ".join(f"{t} {sum(v)}/{len(v)}" for t, v in sorted(by_type.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
