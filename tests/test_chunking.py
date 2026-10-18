"""Sub-headings, table lead-ins and page rules in chunking, plus extraction on a synthetic PDF."""
import make_pdf
from hindi_rag import Page, build_chunks, extract_pages

BODY1 = "कलाम का बचपन रामेश्वरम में बीता। वे अख़बार बेचते थे।"
BODY2 = "इस नगर ने उन्हें आस्था दी। उन्हें समुद्र से लगाव हुआ।"


def page(n, paras, section_no=4, tables=None, intros=None):
    return Page(number=n, section_no=section_no, section="बचपन", paragraphs=paras,
                tables=tables or [], table_intros=intros or [])


def test_subheading_is_metadata_not_body_text():
    chunks = build_chunks([page(1, [BODY1, "इस नगर ने उन्हें क्या दिया", BODY2])])
    assert all("क्या दिया" not in c.text for c in chunks)
    by_sub = {c.subsection: c.text for c in chunks}
    assert "रामेश्वरम" in by_sub[""]
    assert "आस्था" in by_sub["इस नगर ने उन्हें क्या दिया"]
    # the sub-heading is part of what is embedded
    sub_chunk = next(c for c in chunks if c.subsection)
    assert "क्या दिया" in sub_chunk.embed_text


def test_chunks_do_not_cross_a_subheading():
    chunks = build_chunks([page(1, [BODY1, "एक उपशीर्षक", BODY2])], overlap=1)
    assert not any("रामेश्वरम" in c.text and "आस्था" in c.text for c in chunks)


def test_heading_at_end_of_page_governs_start_of_next_page():
    pages = [page(1, [BODY1, "आरंभिक वर्षों के सबक"]), page(2, [BODY2])]
    chunks = build_chunks(pages)
    assert next(c for c in chunks if c.page == 2).subsection == "आरंभिक वर्षों के सबक"


def test_subheading_does_not_leak_into_a_new_section():
    pages = [page(1, [BODY1, "आरंभिक वर्षों के सबक"], section_no=4), page(2, [BODY2], section_no=5)]
    assert next(c for c in build_chunks(pages) if c.page == 2).subsection == ""


def test_title_page_lines_are_not_treated_as_subheadings():
    chunks = build_chunks([page(1, ["अग्नि की उड़ान", "अध्ययन सहायिका"], section_no=0)])
    assert "अग्नि की उड़ान" in chunks[0].text


def test_table_lead_in_is_stored_with_the_table():
    intro = "प्रत्येक मिसाइल का नाम किसी प्राकृतिक शक्ति को दर्शाता था:"
    rows = ["मिसाइल: पृथ्वी | अर्थ: धरती", "मिसाइल: अग्नि | अर्थ: आग"]
    chunks = build_chunks([page(1, [BODY1], tables=[rows], intros=[intro])])
    table = next(c for c in chunks if c.kind == "table")
    assert table.text.startswith(intro) and "पृथ्वी" in table.text
    assert not any(intro in c.text for c in chunks if c.kind == "text")


def test_long_table_repeats_lead_in_in_every_chunk():
    intro = "सूची:"
    rows = [f"क्रम: {i} | नाम: " + "अ" * 60 for i in range(12)]
    tables = [c for c in build_chunks([page(1, [], tables=[rows], intros=[intro])]) if c.kind == "table"]
    assert len(tables) > 1 and all(c.text.startswith(intro) for c in tables)


def test_extract_pages_on_a_synthetic_pdf(tmp_path):
    pdf = str(tmp_path / "s.pdf")
    make_pdf.build(pdf)
    (pg,) = extract_pages(pdf)
    assert (pg.section_no, pg.section) == (4, "Childhood")
    assert pg.table_intros == ["The programme built several systems, each named after a force:"]
    assert not any(p.endswith(":") for p in pg.paragraphs)        # lead-in moved out of the body
    chunks = build_chunks([pg])
    table = next(c for c in chunks if c.kind == "table")
    assert table.text.startswith("The programme built") and "Prithvi" in table.text
    assert any(c.subsection == "What this town gave him" for c in chunks)
    assert all("What this town gave him" not in c.text for c in chunks)
