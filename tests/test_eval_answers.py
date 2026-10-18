"""The answer checker must be able to fail: each test feeds it one specific kind of bad answer."""
import eval_answers as ea

TEXTS = {
    57: "वर्ष: 1997 | सम्मान: भारत रत्न",
    54: "27 जुलाई 2015 को कलाम शिलांग में गिर पड़े और 83 वर्ष की आयु में निधन हो गया।",
    8: "बचपन की कहानी।",
}
HINDI_Q = {"id": 2, "query": "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "expect": ["भारत रत्न", "1997"],
           "answer_must": [["1997"]]}
ENGLISH_Q = {"id": 6, "query": "How and where did Kalam die?", "expect": ["27 जुलाई 2015", "शिलांग"],
             "answer_must": [["27 July 2015"], ["Shillong"]]}
UNANSWERABLE = {"id": 7, "query": "What was his favourite food?", "answerable": False, "expect": []}


def failed(res):
    return sorted(k for k, v in res.items() if v is False)


def test_good_answers_pass():
    assert failed(ea.check(HINDI_Q, "कलाम को भारत रत्न 1997 में मिला [57]", [57, 8], TEXTS)) == []
    assert failed(ea.check(ENGLISH_Q, "He died on 27 July 2015 in Shillong at 83 [54]", [54], TEXTS)) == []
    assert failed(ea.check(UNANSWERABLE, "The passages do not mention his favourite food.", [8], TEXTS)) == []


def test_wrong_language_fails():
    assert "language" in failed(ea.check(HINDI_Q, "He received it in 1997 [57]", [57], TEXTS))
    assert "language" in failed(ea.check(ENGLISH_Q, "उनका निधन 27 जुलाई 2015 को शिलांग में हुआ [54]", [54], TEXTS))


def test_missing_fact_fails():
    assert "facts" in failed(ea.check(HINDI_Q, "कलाम को भारत रत्न मिला [57]", [57], TEXTS))
    assert "facts" in failed(ea.check(ENGLISH_Q, "He died on 27 July 2015 [54]", [54], TEXTS))


def test_missing_or_invented_citation_fails():
    assert "cited" in failed(ea.check(HINDI_Q, "कलाम को भारत रत्न 1997 में मिला", [57], TEXTS))       # none
    assert "cited" in failed(ea.check(HINDI_Q, "कलाम को भारत रत्न 1997 में मिला [99]", [57], TEXTS))  # not retrieved


def test_citation_that_does_not_support_the_answer_fails():
    res = ea.check(HINDI_Q, "कलाम को भारत रत्न 1997 में मिला [8]", [57, 8], TEXTS)
    assert "supported" in failed(res)


def test_number_missing_from_the_cited_chunk_fails():
    res = ea.check(HINDI_Q, "कलाम को भारत रत्न 1997 में, 1998 में नहीं, मिला [57]", [57], TEXTS)
    assert "numbers" in failed(res)


def test_unanswerable_query_needs_a_refusal():
    assert failed(ea.check(UNANSWERABLE, "His favourite food was biryani.", [8], TEXTS)) == ["refusal"]


def test_parse_results_reads_multiline_answers_and_sources(tmp_path):
    log = tmp_path / "r.md"
    log.write_text(
        "# Test query results\n\n```text\n"
        "Query : पहला प्रश्न?\nAnswer: पहली पंक्ति\nदूसरी पंक्ति [1]\nSources:\n"
        "  - page: 2 · section: \"क\" · chunk_id: 1 · score: 0.5  <- cited in answer\n"
        "  - page: 3 · section: \"ख\" · chunk_id: 2 · score: 0.4\n\n"
        "Query : What?   [not in the document: a grounded answer says so]\nAnswer: Not mentioned.\nSources:\n"
        "  - page: 1 · section: \"ग\" · chunk_id: 9 · rrf: 0.03 (dense #1, bm25 -)\n```\n", encoding="utf-8")
    got = ea.parse_results(str(log))
    assert got["पहला प्रश्न?"]["listed"] == [1, 2] and "दूसरी पंक्ति" in got["पहला प्रश्न?"]["answer"]
    assert got["What?"]["listed"] == [9]


def test_the_shipped_results_log_passes():
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not all(os.path.exists(os.path.join(root, f)) for f in ("results.md", "chunks.jsonl", "queries.json")):
        return
    answers = ea.parse_results(os.path.join(root, "results.md"))
    texts = ea.load_chunk_texts(os.path.join(root, "chunks.jsonl"))
    for it in json.load(open(os.path.join(root, "queries.json"), encoding="utf-8")):
        got = answers[it["query"]]
        assert failed(ea.check(it, got["answer"], got["listed"], texts)) == [], it["id"]


def test_number_quoted_from_a_false_premise_question_is_not_flagged():
    q = {"id": 9, "query": "कलाम को भारत रत्न 2005 में मिला था, है ना?", "expect": ["भारत रत्न", "1997"],
         "answer_must": [["1997"]]}
    assert failed(ea.check(q, "नहीं, भारत रत्न 2005 में नहीं बल्कि 1997 में मिला [57]", [57], TEXTS)) == []
    # an invented year that is not in the question is still caught
    assert "numbers" in failed(ea.check(q, "नहीं, भारत रत्न 1998 में नहीं बल्कि 1997 में मिला [57]", [57], TEXTS))


def test_forbidden_string_fails():
    q = {**HINDI_Q, "answer_must_not": ["पद्म भूषण"]}
    assert failed(ea.check(q, "कलाम को भारत रत्न 1997 में मिला [57]", [57], TEXTS)) == []
    assert "forbidden" in failed(ea.check(q, "कलाम को पद्म भूषण और भारत रत्न 1997 में मिला [57]", [57], TEXTS))


def test_a_bare_negative_is_not_a_refusal():
    # "नहीं है" alone used to count as a refusal, so a wrong answer containing it passed
    assert failed(ea.check(UNANSWERABLE, "His favourite food is not rice, it is idli.", [8], TEXTS)) == ["refusal"]
    hindi_unans = {"id": 8, "query": "कलाम की पत्नी का नाम क्या था?", "answerable": False, "expect": []}
    assert failed(ea.check(hindi_unans, "उनकी पत्नी का नाम यहाँ नहीं है और वे स्वस्थ थे।", [8], TEXTS)) == ["refusal"]
    assert failed(ea.check(hindi_unans, "दस्तावेज़ में इसकी जानकारी उपलब्ध नहीं है।", [8], TEXTS)) == []


def test_validate_queries_catches_bad_expect_and_bad_unanswerable():
    import eval_retrieval as er
    corpus = ["वर्ष: 1997 | सम्मान: भारत रत्न", "उनकी पत्नी का उल्लेख नहीं"]
    good = [{"id": 1, "query": "q", "expect": ["भारत रत्न", "1997"]},
            {"id": 2, "query": "q", "answerable": False, "absent": ["संतान"]},
            {"id": 3, "query": "q", "expect": []}]
    assert er.validate_queries(good, corpus) == []
    bad = [{"id": 4, "query": "q", "expect": ["भारत रत्न", "1998"]},
           {"id": 5, "query": "q", "answerable": False, "absent": ["पत्नी"]}]
    assert len(er.validate_queries(bad, corpus)) == 2


def test_expansion_defaults_differ_between_ask_and_eval_retrieval():
    import argparse
    import retrieval as r
    a = argparse.ArgumentParser(); r.add_retrieval_args(a, expand_default=True)
    b = argparse.ArgumentParser(); r.add_retrieval_args(b)
    assert a.parse_args([]).expand_queries is True
    assert a.parse_args(["--no-expand-queries"]).expand_queries is False
    assert b.parse_args([]).expand_queries is False          # eval_retrieval stays a plain baseline
    assert b.parse_args(["--expand-queries"]).expand_queries is True
