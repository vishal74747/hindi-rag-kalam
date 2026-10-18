"""clean_text, the BM25 tokenizer, citation parsing, sub-heading detection."""
from ask import cited_ids, normalize_answer, question_language
from hindi_rag import clean_text, is_subheading
from retrieval import tokenize


def test_doubled_vowel_signs_are_collapsed():
    assert clean_text("उड़ाान") == "उड़ान"
    assert clean_text("अब्दुुल") == "अब्दुल"
    assert clean_text("मूल्यांांकन") == "मूल्यांकन"


def test_listed_split_words_are_repaired():
    assert clean_text("मुस्लि म परिवार") == "मुस्लिम परिवार"
    assert clean_text("व्यक्ति गत") == "व्यक्तिगत"


def test_split_plural_endings_are_repaired():
    assert clean_text("राजनीतिक पंक्ति यों के आर-पार") == "राजनीतिक पंक्तियों के आर-पार"
    assert clean_text("विद्यार्थि यों को") == "विद्यार्थियों को"
    assert clean_text("शक्ति याँ") == "शक्तियाँ"


def test_real_word_boundaries_are_not_joined():
    # these look like the split-word fault but are two real words
    assert clean_text("अग्नि की उड़ान") == "अग्नि की उड़ान"
    assert clean_text("प्रति यात्रा") == "प्रति यात्रा"
    assert clean_text("बल्कि शक्ति का स्रोत") == "बल्कि शक्ति का स्रोत"


def test_invisible_characters_and_spacing():
    assert clean_text("क\u200bला\ufeffम  ") == "कलाम"
    assert clean_text("कलाम ।") == "कलाम।"


def test_tokenizer_keeps_matras_and_drops_danda_and_stopwords():
    toks = tokenize("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?")
    assert "भारत" in toks and "रत्न" in toks and "प्राप्त" in toks
    assert "को" not in toks and "किस" not in toks
    assert tokenize("पोखरण-II में।") == ["पोखरण", "ii"]
    assert tokenize("What did the Prime Minister do?") == ["prime", "minister"]


def test_cited_ids_formats():
    assert cited_ids("a [12] b [27][26]") == {12, 27, 26}
    assert cited_ids("b [3, 5]") == {3, 5}
    assert cited_ids("no citation here") == set()
    assert cited_ids("list [abc] and (12)") == set()


def test_normalize_answer_fixes_model_citation_styles():
    assert normalize_answer("fact 【57】.") == "fact [57]."
    assert normalize_answer("fact 【5†L1-L2】.") == "fact [5]."
    assert normalize_answer("non\u2011breaking") == "non-breaking"


def test_question_language():
    assert question_language("कलाम कौन थे?") == "Hindi"
    assert question_language("Who was Kalam?") == "English"


def test_subheading_detection():
    assert is_subheading("इस नगर ने उन्हें क्या दिया")
    assert is_subheading("आरंभिक वर्षों के सबक")
    assert not is_subheading("कलाम एक मुस्लिम परिवार में पले-बढ़े।")          # a sentence
    assert not is_subheading("ईमानदार श्रम का सम्मान — पिता की नावों से")      # a bullet (dash)
    assert not is_subheading("क्या यह सच है?")                               # a question
    assert not is_subheading("दर्शाता था:")                                  # a lead-in
    assert not is_subheading("एक " * 12)                                     # too long
