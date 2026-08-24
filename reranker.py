"""
Reranker — Cross-Encoder + Hybrid BM25 Search
===============================================
Improves retrieval precision by:
    1. BM25 sparse retrieval      (keyword matching, fast)
    2. Dense FAISS retrieval      (semantic similarity)
    3. Hybrid fusion              (RRF — Reciprocal Rank Fusion)
    4. Cross-encoder reranker     (precision boost, slower)

Pipeline:
    query
      ├─ BM25 → top-K sparse results
      ├─ FAISS → top-K dense results
      └─ RRF fusion → merged ranked list
                         └─ Cross-encoder → final reranked list

Cross-encoder:
    Uses a BERT model that scores (query, passage) pairs jointly.
    Much more accurate than bi-encoder cosine similarity
    but slower — applied only to top-K candidates.

Fallback:
    If cross-encoder model unavailable → TF-IDF keyword overlap scorer.
"""

import re
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RankedResult:
    doc_id:       str
    ticker:       str
    section:      str
    text:         str
    dense_score:  float     # FAISS cosine similarity
    sparse_score: float     # BM25 score
    fusion_score: float     # RRF combined score
    rerank_score: float     # cross-encoder score (or fallback)
    final_rank:   int
    metadata:     dict


# ---------------------------------------------------------------------------
# BM25 sparse retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 (Best Match 25) sparse retrieval.

    BM25 score for document d given query q:
        score(d,q) = Σ IDF(t) · (tf(t,d)·(k1+1)) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

    Parameters:
        k1 : term frequency saturation (default 1.5)
        b  : length normalisation (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_tokens = []
        self.doc_ids       = []
        self.idf           = {}
        self.avgdl         = 0
        self._fitted       = False

    def _tokenise(self, text: str) -> list:
        """Simple whitespace + lowercase tokeniser."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 2]

    def fit(self, documents: list) -> "BM25Retriever":
        """
        Fit BM25 on a list of Document objects.
        Computes IDF for each term and avg document length.
        """
        self.corpus_tokens = []
        self.doc_ids       = []

        for doc in documents:
            tokens = self._tokenise(doc.text)
            self.corpus_tokens.append(tokens)
            self.doc_ids.append(doc.doc_id)

        n = len(self.corpus_tokens)
        self.avgdl = np.mean([len(t) for t in self.corpus_tokens]) if n > 0 else 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = {}
        for tokens in self.corpus_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)

        self._fitted = True
        return self

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        q_tokens = self._tokenise(query)
        d_tokens = self.corpus_tokens[doc_idx]
        dl       = len(d_tokens)

        # Term frequency in document
        tf_map = {}
        for t in d_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in q_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            tf_val  = tf_map.get(term, 0)
            num     = tf_val * (self.k1 + 1)
            den     = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score  += idf_val * (num / den)

        return float(score)

    def retrieve(self, query: str, k: int = 10) -> list:
        """
        Retrieve top-k documents by BM25 score.
        Returns list of (doc_idx, score) tuples sorted descending.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        scores = [(i, self.score(query, i)) for i in range(len(self.corpus_tokens))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    k: int = 60,
) -> dict:
    """
    Combine dense and sparse rankings via Reciprocal Rank Fusion.

    RRF score = dense_weight/(k + dense_rank) + sparse_weight/(k + sparse_rank)

    Parameters
    ----------
    dense_results  : list of SearchResult (from FAISS)
    sparse_results : list of (doc_idx, score) from BM25
    k              : RRF constant (default 60)

    Returns dict: {doc_id: rrf_score}
    """
    rrf_scores = {}

    # Dense results (by doc_id)
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result.doc_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0)
        rrf_scores[doc_id] += dense_weight / (k + rank)

    # Sparse results (by index position → use doc_id from BM25)
    # sparse_results is (doc_idx, score) list
    for rank, (doc_idx, score) in enumerate(sparse_results, start=1):
        # We'll tag by index; caller maps idx → doc_id
        tag = f"__sparse_{doc_idx}__"
        rrf_scores[tag] = rrf_scores.get(tag, 0.0)
        rrf_scores[tag] += sparse_weight / (k + rank)

    return rrf_scores


# ---------------------------------------------------------------------------
# Cross-encoder reranker (with TF-IDF fallback)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder.

    Cross-encoder: processes (query, passage) together → relevance score.
    More accurate than bi-encoder but slower (O(k) forward passes).

    Falls back to TF-IDF keyword overlap scoring if model unavailable.
    """

    CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, use_model: bool = True):
        self.model    = None
        self.use_model = use_model
        if use_model:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            print(f"  Loading cross-encoder ({self.CE_MODEL})...")
            self.model = CrossEncoder(self.CE_MODEL)
            print("  Cross-encoder loaded.")
        except Exception as e:
            print(f"  Cross-encoder unavailable ({e}). Using TF-IDF fallback.")
            self.model = None

    def _tfidf_score(self, query: str, passage: str) -> float:
        """
        Fast TF-IDF keyword overlap score as fallback.
        Computes weighted term overlap between query and passage.
        """
        q_terms = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
        p_terms = passage.lower().split()
        p_tf    = {}
        for t in p_terms:
            p_tf[t] = p_tf.get(t, 0) + 1

        score = 0.0
        for term in q_terms:
            if term in p_tf and len(term) > 2:
                # Boost for financial terms
                boost = 2.0 if len(term) > 5 else 1.0
                score += boost * math.log(1 + p_tf[term])

        # Normalise by query length
        return score / max(len(q_terms), 1)

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ) -> list:
        """
        Rerank candidate results for a query.

        Parameters
        ----------
        query      : user query string
        candidates : list of RankedResult or SearchResult objects
        top_k      : number of results to return after reranking

        Returns list of RankedResult sorted by rerank_score descending.
        """
        if not candidates:
            return []

        # Score each candidate
        if self.model is not None:
            pairs  = [(query, c.text) for c in candidates]
            scores = self.model.predict(pairs).tolist()
        else:
            scores = [self._tfidf_score(query, c.text) for c in candidates]

        # Build RankedResult list
        ranked = []
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            # Handle both SearchResult and RankedResult inputs
            dense_score  = getattr(cand, "dense_score",  getattr(cand, "score", 0.0))
            sparse_score = getattr(cand, "sparse_score", 0.0)
            fusion_score = getattr(cand, "fusion_score", dense_score)

            ranked.append(RankedResult(
                doc_id=cand.doc_id,
                ticker=cand.ticker,
                section=cand.section,
                text=cand.text,
                dense_score=dense_score,
                sparse_score=sparse_score,
                fusion_score=fusion_score,
                rerank_score=round(float(score), 6),
                final_rank=0,
                metadata=cand.metadata,
            ))

        # Sort by rerank score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(ranked):
            r.final_rank = i + 1

        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Hybrid retriever (BM25 + FAISS + RRF + Reranker)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
        BM25 + FAISS → RRF fusion → Cross-encoder reranker

    Usage:
        retriever = HybridRetriever(vector_store, documents)
        results = retriever.retrieve("Apple risk factors", k=5)
    """

    def __init__(
        self,
        vector_store,
        documents: list,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        use_reranker: bool = True,
    ):
        self.store         = vector_store
        self.documents     = documents
        self.dense_weight  = dense_weight
        self.sparse_weight = sparse_weight
        self.doc_id_to_doc = {d.doc_id: d for d in documents}

        # Fit BM25
        print("  Fitting BM25...")
        self.bm25 = BM25Retriever()
        self.bm25.fit(documents)

        # Build doc_idx → doc_id map
        self.idx_to_doc_id = {i: d.doc_id for i, d in enumerate(documents)}

        # Cross-encoder
        self.reranker = CrossEncoderReranker(use_model=use_reranker)

        print("  HybridRetriever ready.")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        filter_ticker: str = None,
        filter_section: str = None,
    ) -> list:
        """
        Retrieve documents using hybrid BM25 + FAISS + reranker.

        Parameters
        ----------
        query          : natural language query
        k              : final number of results
        fetch_k        : candidates to fetch before reranking
        filter_ticker  : restrict to ticker
        filter_section : restrict to section

        Returns list of RankedResult sorted by final rerank score.
        """
        # 1. Dense retrieval (FAISS)
        dense = self.store.search(query, k=fetch_k,
                                  filter_ticker=filter_ticker,
                                  filter_section=filter_section)

        # 2. Sparse retrieval (BM25)
        sparse_raw = self.bm25.retrieve(query, k=fetch_k)

        # 3. RRF fusion
        rrf = reciprocal_rank_fusion(dense, sparse_raw,
                                     self.dense_weight, self.sparse_weight)

        # 4. Collect unique candidates from both sources
        seen     = set()
        candidates = []

        # Add dense results
        for r in dense:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                from dataclasses import dataclass
                # Convert to RankedResult-compatible
                candidates.append(_wrap_search_result(r, rrf))

        # Add BM25-only results not in dense
        for doc_idx, bm25_score in sparse_raw:
            doc_id = self.idx_to_doc_id.get(doc_idx)
            if doc_id and doc_id not in seen:
                doc = self.doc_id_to_doc[doc_id]

                # Apply filters
                if filter_ticker and doc.ticker != filter_ticker:
                    continue
                if filter_section and doc.section != filter_section:
                    continue

                seen.add(doc_id)
                candidates.append(RankedResult(
                    doc_id=doc_id,
                    ticker=doc.ticker,
                    section=doc.section,
                    text=doc.text,
                    dense_score=0.0,
                    sparse_score=round(bm25_score, 4),
                    fusion_score=round(rrf.get(f"__sparse_{doc_idx}__", 0.0), 6),
                    rerank_score=0.0,
                    final_rank=0,
                    metadata=doc.metadata,
                ))

        # Sort by fusion score before reranking
        candidates.sort(key=lambda x: x.fusion_score, reverse=True)
        candidates = candidates[:fetch_k]

        # 5. Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=k)
        return reranked

    def compare_methods(
        self,
        query: str,
        k: int = 5,
    ) -> dict:
        """
        Compare dense-only vs BM25-only vs hybrid retrieval.
        Returns dict with results from each method.
        """
        dense_only  = self.store.search(query, k=k)
        sparse_only = [(self.idx_to_doc_id[i], s)
                       for i, s in self.bm25.retrieve(query, k=k)]
        hybrid      = self.retrieve(query, k=k)

        return {
            "dense_only":  [r.doc_id for r in dense_only],
            "sparse_only": [doc_id for doc_id, _ in sparse_only],
            "hybrid":      [r.doc_id for r in hybrid],
        }


def _wrap_search_result(r, rrf_scores: dict) -> RankedResult:
    """Convert SearchResult to RankedResult with fusion score."""
    return RankedResult(
        doc_id=r.doc_id,
        ticker=r.ticker,
        section=r.section,
        text=r.text,
        dense_score=r.score,
        sparse_score=0.0,
        fusion_score=round(rrf_scores.get(r.doc_id, 0.0), 6),
        rerank_score=0.0,
        final_rank=0,
        metadata=r.metadata,
    )


# ---------------------------------------------------------------------------
# Precision comparison utility
# ---------------------------------------------------------------------------

def precision_comparison(
    vector_store,
    hybrid_retriever,
    test_queries: list,
    k: int = 5,
) -> dict:
    """
    Compare precision@k of dense-only vs hybrid retrieval.

    test_queries: list of (query, expected_ticker, expected_section)
    """
    dense_hits, hybrid_hits = 0, 0

    for query, exp_ticker, exp_section in test_queries:
        dense_results  = vector_store.search(query, k=k)
        hybrid_results = hybrid_retriever.retrieve(query, k=k)

        dense_correct  = any(r.ticker == exp_ticker and r.section == exp_section
                             for r in dense_results)
        hybrid_correct = any(r.ticker == exp_ticker and r.section == exp_section
                             for r in hybrid_results)

        if dense_correct:  dense_hits  += 1
        if hybrid_correct: hybrid_hits += 1

    n = len(test_queries)
    return {
        "dense_precision":  round(dense_hits  / n, 4),
        "hybrid_precision": round(hybrid_hits / n, 4),
        "improvement":      round((hybrid_hits - dense_hits) / max(n, 1), 4),
        "n_queries":        n,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    from rag.vector_store import FAISSVectorStore

    print("=" * 60)
    print("Hybrid Retrieval + Reranker Demo")
    print("=" * 60)

    all_docs = []
    for t in ["AAPL", "MSFT", "TSLA"]:
        f = load_synthetic_filing(t)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))

    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)

    retriever = HybridRetriever(store, all_docs, use_reranker=False)

    queries = [
        "What are Apple's key risk factors?",
        "Microsoft Azure cloud revenue growth 2023",
        "Tesla gross margin decline vehicle prices",
    ]

    for q in queries:
        results = retriever.retrieve(q, k=3)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  #{r.final_rank} [{r.ticker}/{r.section}] "
                  f"dense={r.dense_score:.3f} rerank={r.rerank_score:.3f} | "
                  f"{r.text[:70]}...")

    # Method comparison
    print("\n--- Method Comparison ---")
    comparison = retriever.compare_methods("Apple risk factors competition", k=5)
    for method, doc_ids in comparison.items():
        print(f"  {method}: {len(doc_ids)} results")

    # Precision comparison
    test_qs = [
        ("Apple risk factors", "AAPL", "risk_factors"),
        ("Microsoft revenue growth", "MSFT", "mda"),
        ("Tesla margin decline", "TSLA", "mda"),
    ]
    precision = precision_comparison(store, retriever, test_qs, k=5)
    print(f"\n  Precision comparison:")
    for k, v in precision.items():
        print(f"    {k}: {v}")
