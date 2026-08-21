"""
Unit Tests — FAISS Vector Store
=================================
Run with: python -m pytest tests/test_vector_store.py -v
"""

import pytest
import numpy as np
from rag.vector_store import (
    TFIDFEmbedder, FAISSVectorStore,
    SearchResult, IndexStats, load_embedder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def filings_docs():
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    all_docs = []
    for t in ["AAPL", "MSFT", "TSLA"]:
        f    = load_synthetic_filing(t)
        docs = filing_to_documents(f, chunk_size=150, overlap=30)
        all_docs.extend(docs)
    return all_docs


@pytest.fixture(scope="module")
def built_store(filings_docs):
    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(filings_docs)
    return store


@pytest.fixture(scope="module")
def sample_texts():
    return [
        "Apple iPhone revenue grew significantly this year",
        "Microsoft Azure cloud services expanded rapidly",
        "Tesla electric vehicle deliveries increased substantially",
        "Risk factors include competition and supply chain issues",
        "Gross margin improved due to higher services revenue",
    ]


# ---------------------------------------------------------------------------
# 1. TF-IDF Embedder
# ---------------------------------------------------------------------------

class TestTFIDFEmbedder:

    def test_fit_and_encode(self, sample_texts):
        e = TFIDFEmbedder(n_components=32)
        e.fit(sample_texts)
        vecs = e.encode(sample_texts)
        assert vecs.shape[0] == len(sample_texts)

    def test_output_is_float32(self, sample_texts):
        e = TFIDFEmbedder(n_components=32)
        e.fit(sample_texts)
        vecs = e.encode(sample_texts)
        assert vecs.dtype == np.float32

    def test_l2_normalised(self, sample_texts):
        e = TFIDFEmbedder(n_components=32)
        e.fit(sample_texts)
        vecs = e.encode(sample_texts)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_before_fit_raises(self):
        e = TFIDFEmbedder()
        with pytest.raises(RuntimeError):
            e.encode(["hello world"])

    def test_single_text_encode(self, sample_texts):
        e = TFIDFEmbedder(n_components=16)
        e.fit(sample_texts)
        vec = e.encode(["Apple revenue"])
        assert vec.shape[0] == 1

    def test_dim_matches_n_components(self, sample_texts):
        e = TFIDFEmbedder(n_components=64)
        e.fit(sample_texts)
        vecs = e.encode(sample_texts)
        assert vecs.shape[1] == e.n_components


# ---------------------------------------------------------------------------
# 2. Vector store build
# ---------------------------------------------------------------------------

class TestVectorStoreBuild:

    def test_builds_successfully(self, filings_docs):
        store = FAISSVectorStore(embedding_model="tfidf")
        store.build(filings_docs)
        assert store._is_built

    def test_index_has_correct_count(self, built_store, filings_docs):
        assert built_store.index.ntotal == len(filings_docs)

    def test_documents_stored(self, built_store, filings_docs):
        assert len(built_store.documents) == len(filings_docs)

    def test_empty_docs_raises(self):
        store = FAISSVectorStore(embedding_model="tfidf")
        with pytest.raises(ValueError):
            store.build([])

    def test_search_before_build_raises(self):
        store = FAISSVectorStore(embedding_model="tfidf")
        with pytest.raises(RuntimeError):
            store.search("test query")


# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_returns_list(self, built_store):
        results = built_store.search("Apple risk factors", k=3)
        assert isinstance(results, list)

    def test_returns_k_results(self, built_store):
        results = built_store.search("revenue growth", k=3)
        assert len(results) <= 3

    def test_result_structure(self, built_store):
        r = built_store.search("Apple iPhone", k=1)[0]
        assert isinstance(r, SearchResult)
        assert isinstance(r.text, str)
        assert isinstance(r.score, float)

    def test_scores_descending(self, built_store):
        results = built_store.search("Apple revenue", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_positive(self, built_store):
        results = built_store.search("risk factors", k=5)
        assert all(r.score >= 0 for r in results)

    def test_filter_by_ticker(self, built_store):
        results = built_store.search("revenue growth", k=10,
                                     filter_ticker="AAPL")
        assert all(r.ticker == "AAPL" for r in results)

    def test_filter_by_section(self, built_store):
        results = built_store.search("risk factors", k=10,
                                     filter_section="risk_factors")
        assert all(r.section == "risk_factors" for r in results)

    def test_metadata_present(self, built_store):
        r = built_store.search("Apple", k=1)[0]
        assert "ticker" in r.metadata
        assert "section" in r.metadata

    def test_multi_query_search(self, built_store):
        queries = ["Apple risk factors", "AAPL regulatory risks"]
        results = built_store.search_multi_query(queries, k=5)
        assert isinstance(results, list)
        # All doc_ids should be unique
        ids = [r.doc_id for r in results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 4. Index stats
# ---------------------------------------------------------------------------

class TestIndexStats:

    def test_returns_stats(self, built_store):
        s = built_store.stats()
        assert isinstance(s, IndexStats)

    def test_n_documents_correct(self, built_store, filings_docs):
        assert built_store.stats().n_documents == len(filings_docs)

    def test_tickers_present(self, built_store):
        tickers = built_store.stats().n_tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_embedding_dim_positive(self, built_store):
        assert built_store.stats().embedding_dim > 0


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self, built_store, filings_docs, tmp_path):
        path = str(tmp_path / "test_index")
        built_store.save(path)

        store2 = FAISSVectorStore()
        store2.load(path)
        assert store2._is_built
        assert len(store2.documents) == len(filings_docs)

    def test_search_after_reload(self, built_store, tmp_path):
        path = str(tmp_path / "reload_index")
        built_store.save(path)
        store2 = FAISSVectorStore()
        store2.load(path)
        results = store2.search("Apple revenue", k=3)
        assert len(results) > 0

    def test_results_consistent_after_reload(self, built_store, tmp_path):
        path = str(tmp_path / "consistent_index")
        built_store.save(path)
        store2 = FAISSVectorStore()
        store2.load(path)
        r1 = built_store.search("Apple risk", k=1)
        r2 = store2.search("Apple risk", k=1)
        assert r1[0].doc_id == r2[0].doc_id


# ---------------------------------------------------------------------------
# 6. Retrieval evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_returns_dict(self, built_store):
        test_qs = [("Apple risk factors", "AAPL", "risk_factors")]
        result  = built_store.evaluate_retrieval(test_qs, k=5)
        assert isinstance(result, dict)

    def test_metric_keys(self, built_store):
        test_qs = [("Apple risk", "AAPL", "risk_factors")]
        result  = built_store.evaluate_retrieval(test_qs, k=5)
        assert "precision@5" in result
        assert "recall@5"    in result
        assert "mrr"         in result

    def test_metrics_in_range(self, built_store):
        test_qs = [
            ("Apple risk factors", "AAPL", "risk_factors"),
            ("Microsoft revenue",   "MSFT", "mda"),
        ]
        result = built_store.evaluate_retrieval(test_qs, k=5)
        assert 0 <= result["precision@5"] <= 1
        assert 0 <= result["recall@5"]    <= 1
        assert 0 <= result["mrr"]         <= 1

    def test_perfect_recall_exact_query(self, built_store):
        # Very specific query should retrieve correct doc
        test_qs = [("Apple risk factors competition", "AAPL", "risk_factors")]
        result  = built_store.evaluate_retrieval(test_qs, k=5)
        assert result["recall@5"] == 1.0
