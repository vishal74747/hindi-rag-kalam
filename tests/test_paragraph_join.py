"""Regression tests on the real PDF (skipped when it is not in data/).

They guard the paragraph rebuild in extract_pages: a numbered item's indented continuation line, and
the wrapped title-page note, used to be cut into separate paragraphs, so chunks started mid-sentence.
"""
import os

import pytest

from hindi_rag import DEFAULT_PDF, build_chunks, extract_pages

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PDF), reason=f"{DEFAULT_PDF} not found")


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(extract_pages(DEFAULT_PDF))


def test_numbered_item_is_one_sentence(chunks):
    assert any("जो वर्षों की कठिनाई में भी आशावान बनी रहती हैं।" in c.text for c in chunks)


def test_no_chunk_starts_with_the_orphaned_continuation(chunks):
    assert not any(c.text.startswith("में भी आशावान") for c in chunks)


def test_wrapped_hyphen_is_joined(chunks):
    assert any("रिट्रीवल-ऑगमेंटेड" in c.text for c in chunks)
    assert not any("रिट्रीवल- ऑगमेंटेड" in c.text for c in chunks)