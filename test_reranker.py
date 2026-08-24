"""
Unit Tests — Reranker & Hybrid Retrieval
==========================================
Run with: python -m pytest tests/test_reranker.py -v
"""

import pytest
import numpy as np
from rag.reranker import (
    BM25Retriever, reciprocal_rank_fusion,
    CrossEncoderReranker, HybridRetriever,
    RankedResult, precision_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_docs():
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    docs = []
    for t in ["AAPL", "MSFT", "TSLA"]:
        f = load_synthetic_filing(t)
        docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))
    return docs


@pytest.fixture(scope="module")
def store(all_docs):
    from rag.vector_store import FAISSVectorStore
    s = FAISSVectorStore(embedding_model="tfidf")
    s.build(all_docs)
    return s


@pytest.fixture(scope="module")
def bm25(all_docs):
    b = BM25Retriever()
    b.fit(all_docs)
    return b


@pytest.fixture(scope="module")
def hybrid(store, all_docs):
    return HybridRetriever(store, all_docs, use_reranker=False)


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker(use_model=False)


# ---------------------------------------------------------------------------
# 1. BM25
# ---------------------------------------------------------------------------

class TestBM25:

    def test_fit_returns_self(self, all_docs):
        b = BM25Retriever()
        result = b.fit(all_docs)
        assert result is b

    def test_is_fitted(self, bm25):
        assert bm25._fitted

    def test_retrieve_returns_list(self, bm25):
        results = bm25.retrieve("Apple risk factors", k=5)
        assert isinstance(results, list)

    def test_retrieve_k_results(self, bm25):
        results = bm25.retrieve("Apple", k=3)
        assert len(results) <= 3

    def test_scores_descending(self, bm25):
        results = bm25.retrieve("Apple risk", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_nonnegative(self, bm25):
        results = bm25.retrieve("Apple", k=5)
        assert all(s >= 0 for _, s in results)

    def test_returns_tuple_pairs(self, bm25):
        results = bm25.retrieve("revenue growth", k=3)
        for item in results:
            assert len(item) == 2
            idx, score = item
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_before_fit_raises(self):
        b = BM25Retriever()
        with pytest.raises(RuntimeError):
            b.retrieve("query")

    def test_score_before_fit_raises(self):
        b = BM25Retriever()
        with pytest.raises(RuntimeError):
            b.score("query", 0)

    def test_different_queries_different_results(self, bm25):
        r1 = bm25.retrieve("Apple iPhone revenue", k=3)
        r2 = bm25.retrieve("Tesla gross margin decline", k=3)
        assert r1[0][0] != r2[0][0] or r1[0][1] != r2[0][1]

    def test_idf_computed(self, bm25):
        assert len(bm25.idf) > 0

    def test_avgdl_positive(self, bm25):
        assert bm25.avgdl > 0


# ---------------------------------------------------------------------------
# 2. RRF fusion
# ---------------------------------------------------------------------------

class TestRRF:

    def test_returns_dict(self, store, bm25):
        dense  = store.search("Apple risk", k=5)
        sparse = bm25.retrieve("Apple risk", k=5)
        rrf    = reciprocal_rank_fusion(dense, sparse)
        assert isinstance(rrf, dict)

    def test_scores_positive(self, store, bm25):
        dense  = store.search("Apple risk", k=5)
        sparse = bm25.retrieve("Apple risk", k=5)
        rrf    = reciprocal_rank_fusion(dense, sparse)
        assert all(v >= 0 for v in rrf.values())

    def test_has_entries(self, store, bm25):
        dense  = store.search("Apple risk", k=5)
        sparse = bm25.retrieve("Apple risk", k=5)
        rrf    = reciprocal_rank_fusion(dense, sparse)
        assert len(rrf) > 0

    def test_weights_affect_scores(self, store, bm25):
        dense  = store.search("Apple risk", k=5)
        sparse = bm25.retrieve("Apple risk", k=5)
        rrf1 = reciprocal_rank_fusion(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
        rrf2 = reciprocal_rank_fusion(dense, sparse, dense_weight=0.1, sparse_weight=0.9)
        # Dense-heavy vs sparse-heavy should differ
        assert rrf1 != rrf2


# ---------------------------------------------------------------------------
# 3. Cross-encoder reranker (fallback)
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:

    def test_returns_list(self, reranker, store):
        candidates = store.search("Apple risk factors", k=10)
        results = reranker.rerank("Apple risk factors", candidates, top_k=5)
        assert isinstance(results, list)

    def test_top_k_respected(self, reranker, store):
        candidates = store.search("Apple risk", k=10)
        results = reranker.rerank("Apple risk", candidates, top_k=3)
        assert len(results) <= 3

    def test_returns_ranked_results(self, reranker, store):
        candidates = store.search("Apple risk", k=5)
        results = reranker.rerank("Apple risk", candidates, top_k=3)
        if results:
            assert isinstance(results[0], RankedResult)

    def test_rerank_scores_descending(self, reranker, store):
        candidates = store.search("Apple revenue", k=8)
        results = reranker.rerank("Apple revenue", candidates, top_k=5)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_final_ranks_assigned(self, reranker, store):
        candidates = store.search("Apple", k=5)
        results = reranker.rerank("Apple", candidates, top_k=5)
        if results:
            ranks = [r.final_rank for r in results]
            assert ranks[0] == 1

    def test_empty_candidates(self, reranker):
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_dense_score_preserved(self, reranker, store):
        candidates = store.search("Apple risk", k=5)
        results = reranker.rerank("Apple risk", candidates, top_k=3)
        if results and candidates:
            orig_scores = {r.doc_id: r.score for r in candidates}
            for r in results:
                assert r.dense_score == orig_scores.get(r.doc_id, 0.0)


# ---------------------------------------------------------------------------
# 4. HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:

    def test_retrieve_returns_list(self, hybrid):
        results = hybrid.retrieve("Apple risk factors", k=5)
        assert isinstance(results, list)

    def test_retrieve_k_results(self, hybrid):
        results = hybrid.retrieve("Apple risk", k=3)
        assert len(results) <= 3

    def test_returns_ranked_results(self, hybrid):
        results = hybrid.retrieve("Apple risk", k=3)
        if results:
            assert isinstance(results[0], RankedResult)

    def test_all_have_ticker(self, hybrid):
        results = hybrid.retrieve("revenue growth", k=5)
        assert all(r.ticker in {"AAPL","MSFT","TSLA"} for r in results)

    def test_filter_ticker(self, hybrid):
        results = hybrid.retrieve("risk factors", k=5, filter_ticker="AAPL")
        assert all(r.ticker == "AAPL" for r in results)

    def test_filter_section(self, hybrid):
        results = hybrid.retrieve("revenue growth", k=5, filter_section="mda")
        assert all(r.section == "mda" for r in results)

    def test_compare_methods_keys(self, hybrid):
        comp = hybrid.compare_methods("Apple risk", k=3)
        assert "dense_only"  in comp
        assert "sparse_only" in comp
        assert "hybrid"      in comp

    def test_compare_methods_lists(self, hybrid):
        comp = hybrid.compare_methods("Apple risk", k=3)
        assert isinstance(comp["hybrid"], list)

    def test_fusion_scores_populated(self, hybrid):
        results = hybrid.retrieve("Apple risk factors", k=5)
        assert all(r.fusion_score >= 0 for r in results)


# ---------------------------------------------------------------------------
# 5. Precision comparison
# ---------------------------------------------------------------------------

class TestPrecisionComparison:

    def test_returns_dict(self, store, hybrid):
        test_qs = [("Apple risk factors", "AAPL", "risk_factors")]
        result  = precision_comparison(store, hybrid, test_qs, k=5)
        assert isinstance(result, dict)

    def test_metric_keys(self, store, hybrid):
        test_qs = [("Apple risk", "AAPL", "risk_factors")]
        result  = precision_comparison(store, hybrid, test_qs, k=5)
        for key in ["dense_precision", "hybrid_precision", "n_queries"]:
            assert key in result

    def test_precision_in_range(self, store, hybrid):
        test_qs = [
            ("Apple risk factors", "AAPL", "risk_factors"),
            ("Microsoft revenue",   "MSFT", "mda"),
        ]
        result = precision_comparison(store, hybrid, test_qs, k=5)
        assert 0.0 <= result["dense_precision"]  <= 1.0
        assert 0.0 <= result["hybrid_precision"] <= 1.0

    def test_n_queries_correct(self, store, hybrid):
        test_qs = [("q1", "AAPL", "mda"), ("q2", "MSFT", "mda")]
        result  = precision_comparison(store, hybrid, test_qs, k=5)
        assert result["n_queries"] == 2
