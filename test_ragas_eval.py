"""
Unit Tests — RAG Evaluator & Financial Metric Extractor
=========================================================
Run with: python -m pytest tests/test_ragas_eval.py -v
"""

import pytest
import pandas as pd
from eval.ragas_eval import (
    EvalSample, EvalResult, EvalReport,
    score_context_precision, score_context_recall,
    score_faithfulness, score_answer_relevance,
    RAGEvaluator, FinancialMetricExtractor,
    build_test_cases, _token_overlap, _tokenise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain_and_store():
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    from rag.vector_store import FAISSVectorStore
    from rag.chain import RAGChain
    all_docs = []
    for t in ["AAPL","MSFT","TSLA"]:
        f = load_synthetic_filing(t)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))
    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)
    return RAGChain(store, llm="rule_based", k=5), store


@pytest.fixture(scope="module")
def filings():
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing
    return [load_synthetic_filing(t) for t in ["AAPL","MSFT","TSLA"]]


@pytest.fixture
def sample_chunks():
    return [
        "Apple reported total net sales of $383.3 billion in fiscal 2023, down 3 percent.",
        "iPhone revenue decreased 2 percent to $200.6 billion.",
        "Services revenue reached an all-time high of $85.2 billion, growing 9 percent.",
    ]


@pytest.fixture
def sample():
    return EvalSample(
        question="What was Apple's revenue in 2023?",
        ground_truth="Apple reported total net sales of $383.3 billion.",
        context_chunks=[
            "Apple reported total net sales of $383.3 billion in fiscal 2023.",
            "iPhone revenue was $200.6 billion. Services reached $85.2 billion.",
        ],
        generated_answer="Apple's revenue was $383.3 billion in fiscal 2023. [AAPL/mda]",
        latency_ms=12.5,
    )


# ---------------------------------------------------------------------------
# 1. Tokeniser utilities
# ---------------------------------------------------------------------------

class TestTokeniser:

    def test_returns_set(self):
        assert isinstance(_tokenise("hello world"), set)

    def test_removes_stopwords(self):
        tokens = _tokenise("the company is doing well")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_lowercases(self):
        tokens = _tokenise("Revenue GREW strongly")
        assert all(t == t.lower() for t in tokens)

    def test_overlap_same_text(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_overlap_no_overlap(self):
        assert _token_overlap("apple revenue billion", "tesla spacex") == 0.0

    def test_overlap_in_range(self):
        score = _token_overlap("Apple risk factors", "Apple competition risks")
        assert 0.0 <= score <= 1.0

    def test_empty_string(self):
        assert _token_overlap("", "hello") == 0.0


# ---------------------------------------------------------------------------
# 2. Individual metric scorers
# ---------------------------------------------------------------------------

class TestContextPrecision:

    def test_returns_float(self, sample_chunks):
        score = score_context_precision("Apple revenue", sample_chunks)
        assert isinstance(score, float)

    def test_in_range(self, sample_chunks):
        score = score_context_precision("Apple revenue", sample_chunks)
        assert 0.0 <= score <= 1.0

    def test_relevant_query_high_score(self, sample_chunks):
        score = score_context_precision("Apple iPhone Services revenue billion", sample_chunks)
        assert score > 0.5

    def test_irrelevant_query_low_score(self, sample_chunks):
        score = score_context_precision("quantum physics neutron star", sample_chunks)
        assert score < 0.5

    def test_empty_chunks(self):
        assert score_context_precision("Apple", []) == 0.0


class TestContextRecall:

    def test_returns_float(self, sample_chunks):
        score = score_context_recall("Apple revenue 383 billion", sample_chunks)
        assert isinstance(score, float)

    def test_in_range(self, sample_chunks):
        score = score_context_recall("Apple revenue growth", sample_chunks)
        assert 0.0 <= score <= 1.0

    def test_high_recall_matching_truth(self, sample_chunks):
        gt = "Apple revenue was 383.3 billion iPhone services"
        score = score_context_recall(gt, sample_chunks)
        assert score > 0.4

    def test_empty_chunks(self):
        assert score_context_recall("ground truth", []) == 0.0

    def test_empty_ground_truth(self, sample_chunks):
        assert score_context_recall("", sample_chunks) == 0.0


class TestFaithfulness:

    def test_returns_float(self, sample_chunks):
        score = score_faithfulness("Apple revenue was $383.3 billion.", sample_chunks)
        assert isinstance(score, float)

    def test_in_range(self, sample_chunks):
        score = score_faithfulness("Apple revenue grew this year.", sample_chunks)
        assert 0.0 <= score <= 1.0

    def test_grounded_answer_high(self, sample_chunks):
        answer = "Apple reported net sales of $383.3 billion. iPhone revenue was $200.6 billion."
        score  = score_faithfulness(answer, sample_chunks)
        assert score > 0.5

    def test_empty_answer(self, sample_chunks):
        score = score_faithfulness("", sample_chunks)
        assert score == 0.0

    def test_empty_chunks(self):
        score = score_faithfulness("Apple revenue grew", [])
        assert score == 0.0


class TestAnswerRelevance:

    def test_returns_float(self):
        score = score_answer_relevance("Apple revenue", "Apple reported $383 billion revenue.")
        assert isinstance(score, float)

    def test_in_range(self):
        score = score_answer_relevance("Apple risk factors", "Apple faces competition.")
        assert 0.0 <= score <= 1.0

    def test_relevant_answer_higher(self):
        q = "What is Apple's revenue?"
        relevant  = score_answer_relevance(q, "Apple's revenue was $383 billion in 2023.")
        irrelevant = score_answer_relevance(q, "Tesla makes electric vehicles.")
        assert relevant > irrelevant

    def test_empty_answer(self):
        assert score_answer_relevance("Apple", "") == 0.0

    def test_empty_question(self):
        assert score_answer_relevance("", "Apple revenue") == 0.0


# ---------------------------------------------------------------------------
# 3. RAGEvaluator
# ---------------------------------------------------------------------------

class TestRAGEvaluator:

    def test_evaluate_sample_returns_result(self, sample):
        ev = RAGEvaluator()
        r  = ev.evaluate_sample(sample)
        assert isinstance(r, EvalResult)

    def test_all_metrics_in_range(self, sample):
        ev = RAGEvaluator()
        r  = ev.evaluate_sample(sample)
        for score in [r.context_precision, r.context_recall,
                      r.faithfulness, r.answer_relevance, r.composite_score]:
            assert 0.0 <= score <= 1.0

    def test_composite_weighted(self, sample):
        ev = RAGEvaluator()
        r  = ev.evaluate_sample(sample)
        expected = (0.25 * r.context_precision + 0.25 * r.context_recall +
                    0.30 * r.faithfulness + 0.20 * r.answer_relevance)
        assert abs(r.composite_score - expected) < 0.01

    def test_evaluate_list(self, sample):
        ev = RAGEvaluator()
        report = ev.evaluate([sample, sample])
        assert isinstance(report, EvalReport)
        assert report.n_samples == 2

    def test_report_averages_in_range(self, sample):
        ev = RAGEvaluator()
        report = ev.evaluate([sample])
        for score in [report.avg_precision, report.avg_recall,
                      report.avg_faithfulness, report.avg_relevance,
                      report.avg_composite]:
            assert 0.0 <= score <= 1.0

    def test_to_dataframe(self, sample):
        ev = RAGEvaluator()
        report = ev.evaluate([sample])
        df = ev.to_dataframe(report)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "composite" in df.columns

    def test_evaluate_chain(self, chain_and_store):
        chain, _ = chain_and_store
        ev    = RAGEvaluator()
        cases = build_test_cases(["AAPL"])
        report = ev.evaluate_chain(chain, cases[:2])
        assert report.n_samples == 2
        assert 0.0 <= report.avg_composite <= 1.0


# ---------------------------------------------------------------------------
# 4. Financial metric extractor
# ---------------------------------------------------------------------------

class TestFinancialMetricExtractor:

    def test_extract_revenue(self):
        ex   = FinancialMetricExtractor()
        text = "Apple reported total net sales of $383.3 billion in fiscal 2023."
        r    = ex.extract_metric(text, "revenue")
        assert r is not None
        assert abs(r["value_raw"] - 383.3) < 0.1

    def test_extract_gross_margin(self):
        ex   = FinancialMetricExtractor()
        text = "Gross margin was 44.1 percent compared to 43.3 percent last year."
        r    = ex.extract_metric(text, "gross_margin")
        assert r is not None
        assert abs(r["value_raw"] - 44.1) < 0.1

    def test_extract_growth(self):
        ex   = FinancialMetricExtractor()
        text = "Revenue grew 9 percent year-over-year driven by Services growth."
        r    = ex.extract_metric(text, "growth")
        assert r is not None
        assert abs(r["value_raw"] - 9.0) < 0.5

    def test_returns_none_no_match(self):
        ex = FinancialMetricExtractor()
        r  = ex.extract_metric("No financial data here.", "revenue")
        assert r is None

    def test_extract_all_returns_dict(self, filings):
        ex = FinancialMetricExtractor()
        r  = ex.extract_all(filings[0].raw_text, "AAPL")
        assert isinstance(r, dict)
        assert "ticker" in r

    def test_comparison_table(self, filings):
        ex    = FinancialMetricExtractor()
        table = ex.build_comparison_table(filings)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3

    def test_comparison_table_tickers(self, filings):
        ex    = FinancialMetricExtractor()
        table = ex.build_comparison_table(filings)
        assert "AAPL" in table.index
        assert "MSFT" in table.index

    def test_yoy_change_positive(self):
        ex = FinancialMetricExtractor()
        r  = ex.yoy_change(100.0, 90.0)
        assert r["direction"] == "up"
        assert abs(r["change_pct"] - 11.11) < 0.1

    def test_yoy_change_negative(self):
        ex = FinancialMetricExtractor()
        r  = ex.yoy_change(80.0, 100.0)
        assert r["direction"] == "down"
        assert r["change_pct"] < 0

    def test_yoy_zero_prior(self):
        ex = FinancialMetricExtractor()
        r  = ex.yoy_change(100.0, 0.0)
        assert r["change_pct"] is None


# ---------------------------------------------------------------------------
# 5. Test case builder
# ---------------------------------------------------------------------------

class TestBuildTestCases:

    def test_returns_list(self):
        cases = build_test_cases()
        assert isinstance(cases, list)

    def test_tuple_structure(self):
        cases = build_test_cases()
        for c in cases:
            assert len(c) == 4
            q, gt, ticker, section = c
            assert isinstance(q, str)
            assert isinstance(gt, str)

    def test_ticker_filter(self):
        cases = build_test_cases(["AAPL"])
        tickers = [c[2] for c in cases]
        assert all(t == "AAPL" for t in tickers)

    def test_nonempty(self):
        assert len(build_test_cases()) > 0
