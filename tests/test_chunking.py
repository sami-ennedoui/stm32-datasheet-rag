"""Unit tests for chunking. No network or model needed."""
from app.ingest import chunk_pages, _clean


def test_clean_collapses_whitespace():
    assert _clean("a   b\n\n\n\nc") == "a b\n\nc"


def test_chunks_keep_one_page():
    pages = [(1, "x" * 3000), (2, "y" * 100)]
    chunks = chunk_pages(pages, chunk_chars=1000, overlap=100)
    # every chunk belongs to exactly one page
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}
    # page 1 is long enough to need several chunks
    page1 = [c for c in chunks if c.page == 1]
    assert len(page1) >= 3
    # ids are unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_short_pages_are_skipped():
    pages = [(1, "too short")]
    assert chunk_pages(pages) == []


def test_overlap_present():
    pages = [(1, "abcdefghij" * 30)]  # 300 chars
    chunks = chunk_pages(pages, chunk_chars=100, overlap=20)
    # second chunk should start 80 chars in, overlapping the first
    assert len(chunks) >= 3
    assert chunks[0].text[-20:] == chunks[1].text[:20]
