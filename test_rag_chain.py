"""
Unit Tests — RAG Chain
Run with: python -m pytest tests/test_rag_chain.py -v
"""
import pytest
from rag.chain import (
    RAGChain, RAGAnswer, Citation, ConversationTurn,
    RuleBasedSynthesiser, parse_citations, is_grounded,
    confidence_score, CITATION_RE,
)


@pytest.fixture(scope="module")
def built_chain():
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    from rag.vector_store import FAISSVectorStore
    all_docs = []
    for t in ["AAPL", "MSFT", "TSLA"]:
        f = load_synthetic_filing(t)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))
    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)
    return RAGChain(store, llm="rule_based", k=5)


@pytest.fixture(scope="module")
def sample_results(built_chain):
    return built_chain.store.search("Apple risk factors", k=5)


class TestSynthesiser:
    def test_returns_string(self, sample_results):
        assert isinstance(RuleBasedSynthesiser().synthesise("risks?", sample_results), str)
    def test_nonempty(self, sample_results):
        assert len(RuleBasedSynthesiser().synthesise("risk", sample_results)) > 20
    def test_empty_results(self):
        assert "do not contain" in RuleBasedSynthesiser().synthesise("q", []).lower()
    def test_has_citation(self, sample_results):
        r = RuleBasedSynthesiser().synthesise("Apple risk factors competition", sample_results)
        assert CITATION_RE.search(r)


class TestParseCitations:
    def test_parses(self, sample_results):
        assert len(parse_citations("[AAPL/risk_factors]", sample_results)) >= 1
    def test_structure(self, sample_results):
        cits = parse_citations("[AAPL/mda]", sample_results)
        if cits: assert isinstance(cits[0], Citation)
    def test_empty(self, sample_results):
        assert isinstance(parse_citations("no cites", sample_results), list)
    def test_dedup(self, sample_results):
        cits = parse_citations("[AAPL/risk_factors] [AAPL/risk_factors]", sample_results)
        keys = [(c.ticker, c.section) for c in cits]
        assert len(keys) == len(set(keys))


class TestIsGrounded:
    def test_grounded_citation(self, sample_results):
        assert is_grounded("[AAPL/risk_factors]", sample_results)
    def test_grounded_numbers(self, sample_results):
        assert is_grounded("Revenue was 383.3 billion", sample_results)
    def test_empty_results(self):
        assert not is_grounded("answer", [])
    def test_returns_bool(self, sample_results):
        assert isinstance(is_grounded("answer", sample_results), bool)


class TestConfidenceScore:
    def test_float(self, sample_results):
        assert isinstance(confidence_score("ans", [], sample_results), float)
    def test_in_range(self, sample_results):
        s = confidence_score("detailed answer about risk", [], sample_results)
        assert 0.0 <= s <= 1.0
    def test_zero_empty(self):
        assert confidence_score("ans", [], []) == 0.0
    def test_higher_with_cites(self, sample_results):
        c = Citation("AAPL", "risk_factors", "t", 0.8)
        s1 = confidence_score("a", [], sample_results)
        s2 = confidence_score("a", [c, c, c], sample_results)
        assert s2 >= s1


class TestRAGChainAsk:
    def test_returns_rag_answer(self, built_chain):
        assert isinstance(built_chain.ask("Apple risks?"), RAGAnswer)
    def test_nonempty_answer(self, built_chain):
        assert len(built_chain.ask("Apple risk factors").answer) > 10
    def test_model_used(self, built_chain):
        assert "rule_based" in built_chain.ask("Apple risk").model_used
    def test_n_chunks(self, built_chain):
        assert built_chain.ask("Apple risk factors").n_chunks > 0
    def test_confidence_range(self, built_chain):
        ans = built_chain.ask("Apple risk factors")
        assert 0.0 <= ans.confidence <= 1.0
    def test_is_grounded_bool(self, built_chain):
        assert isinstance(built_chain.ask("Apple risk").is_grounded, bool)
    def test_filter_ticker(self, built_chain):
        ans = built_chain.ask("risk factors", filter_ticker="AAPL")
        assert ans.n_chunks > 0
    def test_filter_section(self, built_chain):
        ans = built_chain.ask("revenue growth", filter_section="mda")
        assert ans.n_chunks > 0
    def test_citations_list(self, built_chain):
        assert isinstance(built_chain.ask("Apple risk factors").citations, list)
    def test_grounded_answer(self, built_chain):
        ans = built_chain.ask("Apple risk factors competition")
        assert ans.is_grounded


class TestHistory:
    def test_history_grows(self, built_chain):
        built_chain.reset_history()
        built_chain.ask("Apple risks?")
        assert len(built_chain.history) == 1
        built_chain.reset_history()
    def test_reset_clears(self, built_chain):
        built_chain.ask("Apple")
        built_chain.reset_history()
        assert len(built_chain.history) == 0
    def test_history_is_conversation_turns(self, built_chain):
        built_chain.reset_history()
        built_chain.ask("Apple risks?")
        assert isinstance(built_chain.history[0], ConversationTurn)
        built_chain.reset_history()


class TestBatchDecompose:
    def test_batch_list(self, built_chain):
        assert isinstance(built_chain.ask_batch(["Apple", "MSFT"]), list)
    def test_batch_all_rag(self, built_chain):
        assert all(isinstance(a, RAGAnswer) for a in built_chain.ask_batch(["Apple","MSFT"]))
    def test_decompose_simple(self, built_chain):
        assert isinstance(built_chain.decompose_query("What are Apple revenues?"), list)
    def test_decompose_compound(self, built_chain):
        parts = built_chain.decompose_query("What are revenues and what are risk factors?")
        assert len(parts) >= 2
