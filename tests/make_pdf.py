"""Builds a tiny synthetic PDF (Latin text) that has the layout features the extractor cares
about: a "N · title" section heading, a short unpunctuated sub-heading, a paragraph ending in ':'
directly above a ruled table, and a paragraph below it. The real study-aid PDF is not needed."""
import pymupdf


def build(path: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    def text(y, s, size=11):
        page.insert_text((60, y), s, fontsize=size, fontname="helv")

    text(80, "4 \u00b7 Childhood")                       # section heading
    text(120, "He grew up by the sea. People were kind to him.")
    text(150, "What this town gave him")                  # sub-heading (no punctuation)
    text(180, "It gave him faith and curiosity about the sky.")
    text(230, "The programme built several systems, each named after a force:")  # table lead-in
    # ruled 3x3 table below it
    x0, y0, cw, rh = 60, 250, 150, 24
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    rows = [["Missile", "Role", "Meaning"], ["Prithvi", "short range", "earth"], ["Agni", "long range", "fire"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * cw + 6, y0 + r * rh + 16), cell, fontsize=10, fontname="helv")
    text(400, "Kalam stressed shared ownership of results.")
    doc.save(path)


if __name__ == "__main__":
    build("/tmp/synthetic.pdf")
