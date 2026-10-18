from lint_results import lint_answer

VOCAB = {"परमाणु", "ऊर्जा", "विभाग", "के", "वैज्ञानिकों"}


def test_flags_a_dropped_space_in_hindi():
    assert lint_answer("परमाणु ऊर्जाविभाग के वैज्ञानिकों", VOCAB)["unknown"] == ["ऊर्जाविभाग"]


def test_flags_glued_latin_words_and_passes_clean_text():
    assert lint_answer("a lecture inShillong, Meghalaya", VOCAB)["glued"] == ["inShillong"]
    clean = lint_answer("परमाणु ऊर्जा विभाग के वैज्ञानिकों, in Shillong", VOCAB)
    assert clean == {"unknown": [], "glued": []}