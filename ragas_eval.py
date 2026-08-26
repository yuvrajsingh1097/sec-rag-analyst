"""
RAG Evaluation — RAGAS-style Metrics + Financial Metric Extractor
==================================================================
Evaluates RAG answer quality using:
    1. Context Precision   — are retrieved chunks relevant to the question?
    2. Context Recall      — does the context cover the answer?
    3. Answer Faithfulness — is the answer grounded in the context?
    4. Answer Relevance    — does the answer address the question?
    5. Latency             — time to generate answer

Also includes a financial metric extractor:
    - Revenue, net income, EPS, gross margin, operating income
    - YoY comparison table builder
    - Multi-company metric comparison
"""

import re
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single evaluation sample with question, context, and answer."""
    question:        str
    ground_truth:    str        # expected answer (for recall)
    context_chunks:  list       # list of retrieved text chunks
    generated_answer:str
    latency_ms:      float = 0.0


@dataclass
class EvalResult:
    """Result of evaluating a single RAG sample."""
    question:          str
    context_precision: float    # 0–1
    context_recall:    float    # 0–1
    faithfulness:      float    # 0–1
    answer_relevance:  float    # 0–1
    latency_ms:        float
    composite_score:   float    # weighted average


@dataclass
class EvalReport:
    """Aggregated evaluation report over a test set."""
    results:             list       # list of EvalResult
    avg_precision:       float
    avg_recall:          float
    avg_faithfulness:    float
    avg_relevance:       float
    avg_latency_ms:      float
    avg_composite:       float
    n_samples:           int


# ---------------------------------------------------------------------------
# Tokeniser & overlap utilities
# ---------------------------------------------------------------------------

EVAL_STOPWORDS = {
    "the","a","an","is","are","was","were","of","in","to","for",
    "and","or","but","with","that","this","it","its","be","been",
    "by","from","as","at","on","not","have","has","had","do","does",
}

def _tokenise(text: str) -> set:
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {w for w in words if w not in EVAL_STOPWORDS and len(w) > 2}


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like token overlap between two texts."""
    a = _tokenise(text_a)
    b = _tokenise(text_b)
    if not a or not b:
        return 0.0
    intersection = a & b
    union        = a | b
    return len(intersection) / len(union)


def _contains_numbers(text: str) -> list:
    """Extract all numeric values from text."""
    return re.findall(r"\d+\.?\d*", text)


# ---------------------------------------------------------------------------
# Individual metric scorers
# ---------------------------------------------------------------------------

def score_context_precision(
    question: str,
    context_chunks: list,
    threshold: float = 0.05,
) -> float:
    """
    Context Precision: fraction of retrieved chunks relevant to the question.

    A chunk is relevant if its token overlap with the question > threshold.
    """
    if not context_chunks:
        return 0.0

    relevant = sum(
        1 for chunk in context_chunks
        if _token_overlap(question, chunk) > threshold
    )
    return round(relevant / len(context_chunks), 4)


def score_context_recall(
    ground_truth: str,
    context_chunks: list,
) -> float:
    """
    Context Recall: how much of the ground truth is covered by the context.

    Measures what fraction of ground truth tokens appear in the context.
    """
    if not context_chunks or not ground_truth:
        return 0.0

    gt_tokens      = _tokenise(ground_truth)
    context_text   = " ".join(context_chunks)
    context_tokens = _tokenise(context_text)

    if not gt_tokens:
        return 0.0

    covered = gt_tokens & context_tokens
    return round(len(covered) / len(gt_tokens), 4)


def score_faithfulness(
    generated_answer: str,
    context_chunks: list,
) -> float:
    """
    Faithfulness: fraction of answer claims supported by the context.

    Approach:
        1. Extract sentences from the answer
        2. For each sentence, check if key tokens appear in context
        3. Score = fraction of sentences that are contextually grounded
    """
    if not generated_answer or not context_chunks:
        return 0.0

    context_text   = " ".join(context_chunks)
    context_tokens = _tokenise(context_text)

    # Split answer into sentences
    sentences = re.split(r"(?<=[.!?])\s+", generated_answer.strip())
    sentences = [s for s in sentences if len(s.split()) >= 4]

    if not sentences:
        # Single short answer — check overall overlap
        return min(1.0, _token_overlap(generated_answer, context_text) * 3)

    grounded_count = 0
    for sent in sentences:
        sent_tokens = _tokenise(sent)
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap > 0.25:   # at least 25% of sentence tokens in context
            grounded_count += 1

        # Also check number overlap (numbers are strong grounding signals)
        sent_nums    = set(_contains_numbers(sent))
        context_nums = set(_contains_numbers(context_text))
        if sent_nums and sent_nums & context_nums:
            grounded_count = min(grounded_count + 0.5, len(sentences))

    return round(min(grounded_count / len(sentences), 1.0), 4)


def score_answer_relevance(
    question: str,
    generated_answer: str,
) -> float:
    """
    Answer Relevance: how well the answer addresses the question.

    Heuristics:
        - Token overlap between question and answer
        - Answer length (too short = likely irrelevant)
        - Presence of question keywords in answer
        - Financial number presence (finance questions expect numbers)
    """
    if not generated_answer or not question:
        return 0.0

    # Token overlap
    overlap = _token_overlap(question, generated_answer)

    # Length score
    n_words    = len(generated_answer.split())
    len_score  = min(n_words / 30, 1.0)

    # Question keyword coverage
    q_tokens   = _tokenise(question)
    ans_tokens = _tokenise(generated_answer)
    kw_coverage = len(q_tokens & ans_tokens) / max(len(q_tokens), 1)

    # Numeric presence for financial questions
    finance_kws = {"revenue","income","margin","growth","profit","billion","percent","eps"}
    is_finance  = bool(q_tokens & finance_kws)
    num_bonus   = 0.1 if is_finance and _contains_numbers(generated_answer) else 0.0

    score = 0.35 * overlap + 0.25 * len_score + 0.35 * kw_coverage + num_bonus
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates a RAG system on a test set of question–answer pairs.

    Weights for composite score:
        context_precision : 0.25
        context_recall    : 0.25
        faithfulness      : 0.30
        answer_relevance  : 0.20
    """

    WEIGHTS = {
        "context_precision": 0.25,
        "context_recall":    0.25,
        "faithfulness":      0.30,
        "answer_relevance":  0.20,
    }

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single RAG sample."""
        precision  = score_context_precision(sample.question, sample.context_chunks)
        recall     = score_context_recall(sample.ground_truth, sample.context_chunks)
        faithful   = score_faithfulness(sample.generated_answer, sample.context_chunks)
        relevance  = score_answer_relevance(sample.question, sample.generated_answer)

        composite = (
            self.WEIGHTS["context_precision"] * precision +
            self.WEIGHTS["context_recall"]    * recall    +
            self.WEIGHTS["faithfulness"]      * faithful  +
            self.WEIGHTS["answer_relevance"]  * relevance
        )

        return EvalResult(
            question=sample.question,
            context_precision=precision,
            context_recall=recall,
            faithfulness=faithful,
            answer_relevance=relevance,
            latency_ms=sample.latency_ms,
            composite_score=round(composite, 4),
        )

    def evaluate(self, samples: list) -> EvalReport:
        """
        Evaluate a list of EvalSample objects.
        Returns an EvalReport with aggregated metrics.
        """
        results = [self.evaluate_sample(s) for s in samples]

        return EvalReport(
            results=results,
            avg_precision=  round(float(np.mean([r.context_precision for r in results])), 4),
            avg_recall=     round(float(np.mean([r.context_recall     for r in results])), 4),
            avg_faithfulness=round(float(np.mean([r.faithfulness      for r in results])), 4),
            avg_relevance=  round(float(np.mean([r.answer_relevance   for r in results])), 4),
            avg_latency_ms= round(float(np.mean([r.latency_ms         for r in results])), 2),
            avg_composite=  round(float(np.mean([r.composite_score    for r in results])), 4),
            n_samples=len(results),
        )

    def evaluate_chain(
        self,
        chain,
        test_cases: list,
    ) -> EvalReport:
        """
        Run the full evaluation pipeline on a RAG chain.

        Parameters
        ----------
        chain      : RAGChain instance
        test_cases : list of (question, ground_truth, expected_ticker, expected_section)

        Returns EvalReport.
        """
        samples = []

        for question, ground_truth, exp_ticker, exp_section in test_cases:
            t0  = time.time()
            ans = chain.ask(question, filter_ticker=exp_ticker,
                            filter_section=exp_section)
            lat = (time.time() - t0) * 1000

            context_chunks = [r.text for r in chain.store.search(
                question, k=5, filter_ticker=exp_ticker, filter_section=exp_section
            )]

            samples.append(EvalSample(
                question=question,
                ground_truth=ground_truth,
                context_chunks=context_chunks,
                generated_answer=ans.answer,
                latency_ms=round(lat, 2),
            ))

        return self.evaluate(samples)

    def to_dataframe(self, report: EvalReport) -> pd.DataFrame:
        """Convert EvalReport to a summary DataFrame."""
        rows = []
        for r in report.results:
            rows.append({
                "question":          r.question[:50] + "...",
                "ctx_precision":     r.context_precision,
                "ctx_recall":        r.context_recall,
                "faithfulness":      r.faithfulness,
                "answer_relevance":  r.answer_relevance,
                "composite":         r.composite_score,
                "latency_ms":        r.latency_ms,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Financial metric extractor
# ---------------------------------------------------------------------------

# Patterns for common financial metrics
METRIC_PATTERNS = {
    "revenue": [
        r"(?:total\s+)?(?:net\s+)?(?:revenue|sales|net\s+sales)\s+(?:of|was|were|totaled?)\s+"
        r"[\$]?([\d,\.]+)\s*(billion|million|trillion)?",
        r"[\$]?([\d,\.]+)\s*(billion|million)?\s+in\s+(?:total\s+)?(?:revenue|sales)",
    ],
    "net_income": [
        r"net\s+income\s+(?:of|was|were)\s+[\$]?([\d,\.]+)\s*(billion|million)?",
        r"net\s+(?:earnings|profit)\s+(?:of|was)\s+[\$]?([\d,\.]+)\s*(billion|million)?",
    ],
    "gross_margin": [
        r"gross\s+margin\s+(?:of|was|were)\s+([\d,\.]+)\s*(?:percent|%)",
        r"gross\s+margin\s+(?:was\s+)?(\d+\.?\d*)\s*(?:percent|%)",
    ],
    "operating_income": [
        r"operating\s+income\s+(?:of|was|were)\s+[\$]?([\d,\.]+)\s*(billion|million)?",
        r"operating\s+(?:earnings|profit)\s+(?:of|was)\s+[\$]?([\d,\.]+)\s*(billion|million)?",
    ],
    "eps": [
        r"(?:diluted\s+)?(?:earnings|eps)\s+per\s+(?:diluted\s+)?share\s+(?:of|was|were)\s+"
        r"[\$]?([\d,\.]+)",
        r"\$([\d,\.]+)\s+per\s+(?:diluted\s+)?share",
    ],
    "growth": [
        r"(?:revenue|sales|income)\s+(?:grew|increased|grew)\s+([\d,\.]+)\s*(?:percent|%)",
        r"(?:up|grew|increased)\s+([\d,\.]+)\s*(?:percent|%)\s+year.over.year",
        r"(?:up|grew|increased)\s+([\d,\.]+)\s*(?:percent|%)",
    ],
}

UNIT_MULTIPLIERS = {
    "trillion": 1_000_000,   # in millions
    "billion":  1_000,
    "million":  1,
    None:       1,
    "":         1,
}


class FinancialMetricExtractor:
    """
    Extracts financial metrics from SEC filing text.

    Extracts: revenue, net income, gross margin, operating income, EPS, growth.
    Builds YoY comparison tables across tickers.
    """

    def _clean_number(self, num_str: str) -> float:
        """Parse a number string to float."""
        return float(num_str.replace(",", "").replace("$", ""))

    def extract_metric(self, text: str, metric: str) -> Optional[dict]:
        """
        Extract a single metric from text.

        Returns dict with value, unit, and raw match, or None.
        """
        if metric not in METRIC_PATTERNS:
            return None

        for pattern in METRIC_PATTERNS[metric]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    value = self._clean_number(groups[0])
                    unit  = groups[1].lower() if len(groups) > 1 and groups[1] else None
                    mult  = UNIT_MULTIPLIERS.get(unit, 1)
                    return {
                        "value_raw":    value,
                        "unit":         unit or "absolute",
                        "value_m":      value * mult,   # in millions
                        "raw_match":    match.group(0)[:80],
                    }
                except (ValueError, AttributeError):
                    continue
        return None

    def extract_all(self, text: str, ticker: str = "") -> dict:
        """Extract all financial metrics from a text block."""
        results = {"ticker": ticker}
        for metric in METRIC_PATTERNS:
            extracted = self.extract_metric(text, metric)
            if extracted:
                results[metric] = extracted
        return results

    def build_comparison_table(
        self,
        filings: list,
        metrics: list = None,
    ) -> pd.DataFrame:
        """
        Build a metric comparison table across multiple filings.

        Parameters
        ----------
        filings : list of Filing objects
        metrics : list of metric names to extract (default: all)

        Returns DataFrame: rows = tickers, columns = metrics.
        """
        if metrics is None:
            metrics = list(METRIC_PATTERNS.keys())

        rows = []
        for filing in filings:
            text = filing.raw_text
            row  = {
                "ticker":  filing.ticker,
                "period":  filing.period,
                "company": filing.company_name,
            }
            for metric in metrics:
                extracted = self.extract_metric(text, metric)
                if extracted:
                    row[metric] = extracted["value_raw"]
                    row[f"{metric}_unit"] = extracted["unit"]
                else:
                    row[metric] = None
            rows.append(row)

        return pd.DataFrame(rows).set_index("ticker")

    def yoy_change(
        self,
        current: float,
        prior: float,
    ) -> dict:
        """Compute year-over-year percentage change."""
        if prior == 0 or prior is None or current is None:
            return {"change_pct": None, "direction": "unknown"}
        change = (current - prior) / abs(prior) * 100
        return {
            "change_pct":  round(change, 2),
            "direction":   "up" if change > 0 else "down",
        }


# ---------------------------------------------------------------------------
# Test case builder
# ---------------------------------------------------------------------------

def build_test_cases(tickers: list = None) -> list:
    """
    Build a standard set of RAG evaluation test cases for SEC filings.

    Returns list of (question, ground_truth, ticker, section) tuples.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "TSLA"]

    cases = []

    if "AAPL" in tickers:
        cases += [
            (
                "What are Apple's main risk factors?",
                "Apple faces competition, supply chain risks, regulatory risks, "
                "FX exposure, and dependency on iPhone revenue.",
                "AAPL", "risk_factors",
            ),
            (
                "What was Apple's total revenue in fiscal 2023?",
                "Apple reported total net sales of $383.3 billion in fiscal 2023.",
                "AAPL", "mda",
            ),
            (
                "What is Apple's Services segment revenue?",
                "Apple Services revenue reached $85.2 billion, up 9 percent.",
                "AAPL", "mda",
            ),
        ]

    if "MSFT" in tickers:
        cases += [
            (
                "How fast did Microsoft Azure grow?",
                "Azure grew 29 percent in constant currency in fiscal 2023.",
                "MSFT", "mda",
            ),
            (
                "What are Microsoft's key cybersecurity risks?",
                "Microsoft faces sophisticated cyberattacks including nation-state actors.",
                "MSFT", "risk_factors",
            ),
        ]

    if "TSLA" in tickers:
        cases += [
            (
                "Why did Tesla's gross margin decline in 2023?",
                "Tesla's gross margin declined to 18.2 percent due to vehicle price reductions.",
                "TSLA", "mda",
            ),
            (
                "How many vehicles did Tesla deliver in 2023?",
                "Tesla delivered 1.81 million vehicles in 2023, up 38 percent.",
                "TSLA", "mda",
            ),
        ]

    return cases


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    from rag.vector_store import FAISSVectorStore
    from rag.chain import RAGChain

    print("=" * 60)
    print("RAG Evaluation + Financial Metric Extractor Demo")
    print("=" * 60)

    # Build system
    all_docs = []
    filings  = []
    for t in ["AAPL","MSFT","TSLA"]:
        f = load_synthetic_filing(t)
        filings.append(f)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))

    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)
    chain = RAGChain(store, llm="rule_based", k=5)

    # Evaluate
    evaluator  = RAGEvaluator()
    test_cases = build_test_cases()
    report     = evaluator.evaluate_chain(chain, test_cases)

    print(f"\n  Evaluation Results ({report.n_samples} samples):")
    print(f"  Context Precision : {report.avg_precision:.4f}")
    print(f"  Context Recall    : {report.avg_recall:.4f}")
    print(f"  Faithfulness      : {report.avg_faithfulness:.4f}")
    print(f"  Answer Relevance  : {report.avg_relevance:.4f}")
    print(f"  Composite Score   : {report.avg_composite:.4f}")
    print(f"  Avg Latency       : {report.avg_latency_ms:.1f}ms")

    df = evaluator.to_dataframe(report)
    print(f"\n  Per-question breakdown:")
    print(df[["question","composite","faithfulness"]].to_string(index=False))

    # Financial metric extraction
    print("\n  Financial Metric Extraction:")
    extractor = FinancialMetricExtractor()
    table = extractor.build_comparison_table(filings)
    print(table[["revenue","gross_margin","growth"]].to_string())
