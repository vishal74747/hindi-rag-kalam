"""Typo / spacing lint for the generated answers in results.md.

The LLM occasionally drops a space ("inShillong", "ऊर्जाविभाग") even when the passages are right, and
eval_answers.py cannot see that. For every answerable query this prints

  * Devanagari words in the answer that occur in no chunk and not in the question (a dropped space or
    a misspelling; a legitimately inflected word can also show up, so look at each one), and
  * Latin words with a capital letter glued inside them ("inShillong").

    python lint_results.py            # reads results.md, chunks.jsonl, queries.json
    python ask.py --queries queries.json --out results.md   # re-run if it reports a real typo

Exit status 1 if anything is flagged. Unanswerable queries are skipped (a refusal uses its own wording).
"""
import argparse
import json
import re
import sys

from eval_answers import load_chunk_texts, parse_results
from hindi_rag import clean_text, force_utf8_output

_DEV_WORD = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")
_GLUED = re.compile(r"\b[a-z]+[A-Z][a-z]+\b")


def lint_answer(answer: str, vocab: set[str]) -> dict:
    text = clean_text(answer)
    return {"unknown": sorted({w for w in _DEV_WORD.findall(text) if w not in vocab}),
            "glued": sorted(set(_GLUED.findall(text)))}


def main() -> int:
    force_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results.md")
    ap.add_argument("--queries", default="queries.json")
    ap.add_argument("--chunks", default="chunks.jsonl")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        items = json.load(f)
    answers = parse_results(args.results)
    chunk_text = " ".join(load_chunk_texts(args.chunks).values())
    flagged = 0
    for it in items:
        if not it.get("answerable", True):
            continue
        got = answers.get(it["query"])
        if got is None:
            print(f"Q{it['id']}: MISSING from the log")
            flagged += 1
            continue
        vocab = set(_DEV_WORD.findall(clean_text(chunk_text + " " + it["query"])))
        r = lint_answer(got["answer"], vocab)
        if r["unknown"] or r["glued"]:
            flagged += 1
            print(f"Q{it['id']}: check by eye  not in passages: {r['unknown'] or '-'}  glued: {r['glued'] or '-'}")
            print(f"      {got['answer'][:160]}")
    print(f"\n{flagged} of {sum(1 for i in items if i.get('answerable', True))} answerable answers flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())