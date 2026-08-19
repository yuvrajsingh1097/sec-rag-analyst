"""
Unit Tests — SEC Filing Loader
================================
Run with: python -m pytest tests/test_sec_loader.py -v
"""

import pytest
import json
import os
from loaders.sec_loader import (
    Document, Filing,
    clean_filing_text, extract_sections, chunk_text,
    filing_to_documents, load_synthetic_filing,
    save_filing_json, load_filing_json,
    SYNTHETIC_FILINGS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aapl_filing():
    return load_synthetic_filing("AAPL")


@pytest.fixture
def msft_filing():
    return load_synthetic_filing("MSFT")


@pytest.fixture
def tsla_filing():
    return load_synthetic_filing("TSLA")


@pytest.fixture
def aapl_docs(aapl_filing):
    return filing_to_documents(aapl_filing, chunk_size=100, overlap=20)


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

class TestCleanFilingText:

    def test_removes_html(self):
        assert "<b>" not in clean_filing_text("<b>Revenue</b> grew")

    def test_normalises_whitespace(self):
        result = clean_filing_text("hello   world\n\nfoo")
        assert "   " not in result

    def test_returns_string(self):
        assert isinstance(clean_filing_text("test text"), str)

    def test_empty_string(self):
        assert clean_filing_text("") == ""

    def test_preserves_content(self):
        result = clean_filing_text("Revenue grew significantly")
        assert "Revenue" in result


# ---------------------------------------------------------------------------
# 2. Section extraction
# ---------------------------------------------------------------------------

class TestExtractSections:

    def test_returns_dict(self, aapl_filing):
        sections = extract_sections(aapl_filing.raw_text)
        assert isinstance(sections, dict)

    def test_empty_text_returns_empty(self):
        assert extract_sections("") == {}


# ---------------------------------------------------------------------------
# 3. Text chunker
# ---------------------------------------------------------------------------

class TestChunkText:

    def test_returns_list(self):
        chunks = chunk_text("hello world foo bar baz", chunk_size=3, overlap=1)
        assert isinstance(chunks, list)

    def test_single_chunk_short_text(self):
        chunks = chunk_text("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_multiple_chunks_long_text(self):
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_makes_more_chunks(self):
        text = " ".join(["word"] * 300)
        c1 = chunk_text(text, chunk_size=100, overlap=0)
        c2 = chunk_text(text, chunk_size=100, overlap=50)
        assert len(c2) >= len(c1)

    def test_no_empty_chunks(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        for c in chunks[:-1]:
            assert len(c.split()) <= 100


# ---------------------------------------------------------------------------
# 4. Synthetic filing loading
# ---------------------------------------------------------------------------

class TestLoadSyntheticFiling:

    def test_returns_filing(self, aapl_filing):
        assert isinstance(aapl_filing, Filing)

    def test_ticker_correct(self, aapl_filing):
        assert aapl_filing.ticker == "AAPL"

    def test_all_tickers_load(self):
        for ticker in ["AAPL", "MSFT", "TSLA"]:
            f = load_synthetic_filing(ticker)
            assert f is not None
            assert f.ticker == ticker

    def test_unknown_returns_none(self):
        assert load_synthetic_filing("ZZZZ") is None

    def test_has_sections(self, aapl_filing):
        assert len(aapl_filing.sections) > 0

    def test_has_raw_text(self, aapl_filing):
        assert len(aapl_filing.raw_text) > 0

    def test_sections_nonempty(self, aapl_filing):
        for section, text in aapl_filing.sections.items():
            assert len(text) > 0

    def test_form_is_10k(self, aapl_filing):
        assert aapl_filing.form == "10-K"

    def test_period_format(self, aapl_filing):
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}", aapl_filing.period)


# ---------------------------------------------------------------------------
# 5. Filing to documents
# ---------------------------------------------------------------------------

class TestFilingToDocuments:

    def test_returns_list(self, aapl_docs):
        assert isinstance(aapl_docs, list)

    def test_nonempty(self, aapl_docs):
        assert len(aapl_docs) > 0

    def test_document_structure(self, aapl_docs):
        d = aapl_docs[0]
        assert isinstance(d, Document)
        assert isinstance(d.text, str)
        assert isinstance(d.metadata, dict)

    def test_doc_id_unique(self, aapl_docs):
        ids = [d.doc_id for d in aapl_docs]
        assert len(ids) == len(set(ids))

    def test_metadata_keys(self, aapl_docs):
        for doc in aapl_docs:
            for key in ["ticker", "section", "form", "chunk_index"]:
                assert key in doc.metadata

    def test_ticker_in_doc(self, aapl_docs):
        for doc in aapl_docs:
            assert doc.ticker == "AAPL"

    def test_word_count(self, aapl_docs):
        for doc in aapl_docs:
            assert doc.word_count == len(doc.text.split())

    def test_char_count(self, aapl_docs):
        for doc in aapl_docs:
            assert doc.char_count == len(doc.text)

    def test_all_tickers_produce_docs(self):
        for ticker in ["AAPL", "MSFT", "TSLA"]:
            f    = load_synthetic_filing(ticker)
            docs = filing_to_documents(f)
            assert len(docs) > 0


# ---------------------------------------------------------------------------
# 6. JSON storage
# ---------------------------------------------------------------------------

class TestJSONStorage:

    def test_saves_file(self, aapl_filing, tmp_path):
        fpath = save_filing_json(aapl_filing, str(tmp_path))
        assert os.path.exists(fpath)

    def test_valid_json(self, aapl_filing, tmp_path):
        fpath = save_filing_json(aapl_filing, str(tmp_path))
        with open(fpath) as f:
            data = json.load(f)
        assert "ticker" in data
        assert "sections" in data
        assert "stats" in data

    def test_roundtrip(self, aapl_filing, tmp_path):
        fpath  = save_filing_json(aapl_filing, str(tmp_path))
        loaded = load_filing_json(fpath)
        assert loaded.ticker  == aapl_filing.ticker
        assert loaded.form    == aapl_filing.form
        assert loaded.period  == aapl_filing.period

    def test_sections_preserved(self, aapl_filing, tmp_path):
        fpath  = save_filing_json(aapl_filing, str(tmp_path))
        loaded = load_filing_json(fpath)
        for section in aapl_filing.sections:
            assert section in loaded.sections
