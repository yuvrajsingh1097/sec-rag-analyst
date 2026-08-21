"""
FAISS Vector Store
===================
Builds and queries a FAISS index over SEC filing chunks.

Embedding models supported:
    1. sentence-transformers/all-MiniLM-L6-v2  (fast, 384-dim)
    2. sentence-transformers/all-mpnet-base-v2  (accurate, 768-dim)
    3. TF-IDF fallback                          (no GPU needed)

Features:
    - Build index from Document list
    - Similarity search (top-k)
    - Persist index to disk
    - Load index from disk
    - Retrieval quality evaluation (top-k recall)
    - BM25 sparse retrieval (for hybrid search in Day 4)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    doc_id:    str
    ticker:    str
    section:   str
    text:      str
    score:     float        # cosine similarity ∈ [0, 1]
    metadata:  dict


@dataclass
class IndexStats:
    n_documents:  int
    n_tickers:    list
    n_sections:   list
    embedding_dim: int
    embedding_model: str
    index_type:   str


# ---------------------------------------------------------------------------
# TF-IDF fallback embedder
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    Fast TF-IDF based embedder.
    Used as fallback when sentence-transformers unavailable.
    Produces sparse-ish dense vectors via SVD truncation.
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.vectorizer   = None
        self.svd          = None
        self.is_fitted    = False

    def fit(self, texts: list) -> "TFIDFEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        from sklearn.pipeline import make_pipeline

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n = min(self.n_components, tfidf.shape[1] - 1, len(texts) - 1)
        self.svd = TruncatedSVD(n_components=n, random_state=42)
        self.svd.fit(tfidf)
        self.n_components = self.svd.n_components
        self.is_fitted = True
        return self

    def encode(self, texts: list, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before encode()")
        tfidf = self.vectorizer.transform(texts)
        vecs  = self.svd.transform(tfidf)
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

def load_embedder(model_name: str = "tfidf", texts: list = None):
    """
    Load embedding model.

    model_name options:
        'tfidf'                                 — fast fallback (default)
        'sentence-transformers/all-MiniLM-L6-v2' — 384-dim, fast
        'sentence-transformers/all-mpnet-base-v2' — 768-dim, accurate
    """
    if model_name == "tfidf":
        embedder = TFIDFEmbedder(n_components=128)
        if texts:
            embedder.fit(texts)
        return embedder, "tfidf", embedder.n_components

    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading {model_name}...")
        model = SentenceTransformer(model_name)
        dim   = model.get_sentence_embedding_dimension()
        print(f"  Loaded — embedding dim: {dim}")
        return model, model_name, dim
    except Exception as e:
        print(f"  SentenceTransformer failed ({e}), using TF-IDF fallback.")
        embedder = TFIDFEmbedder(n_components=128)
        if texts:
            embedder.fit(texts)
        return embedder, "tfidf", embedder.n_components


# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    FAISS-backed vector store for SEC filing chunks.

    Supports:
        - IndexFlatIP  (exact cosine similarity, small corpora)
        - IndexIVFFlat (approximate, large corpora > 10k docs)

    Usage:
        store = FAISSVectorStore()
        store.build(documents)
        results = store.search("What are Apple's risk factors?", k=5)
        store.save("models/faiss_index")
        store.load("models/faiss_index")
    """

    def __init__(
        self,
        embedding_model: str = "tfidf",
        index_type: str = "flat",
    ):
        self.embedding_model_name = embedding_model
        self.index_type  = index_type
        self.index       = None
        self.embedder    = None
        self.documents   = []       # original Document objects
        self.texts       = []       # text list aligned to index
        self.dim         = None
        self._is_built   = False

    def build(self, documents: list, batch_size: int = 64) -> "FAISSVectorStore":
        """
        Build FAISS index from a list of Document objects.

        Parameters
        ----------
        documents  : list of Document objects
        batch_size : embedding batch size

        Returns self for chaining.
        """
        import faiss

        self.documents = documents
        self.texts     = [d.text for d in documents]
        n              = len(self.texts)

        if n == 0:
            raise ValueError("No documents to index.")

        # Load embedder
        self.embedder, self.embedding_model_name, self.dim = load_embedder(
            self.embedding_model_name, self.texts
        )

        # Embed all documents
        print(f"  Embedding {n} documents (dim={self.dim})...")
        embeddings = self._embed_texts(self.texts, batch_size=batch_size)

        # Build FAISS index
        if self.index_type == "flat" or n < 1000:
            # Exact inner product (cosine after L2 norm)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
        else:
            # Approximate IVF index for large corpora
            n_clusters = min(int(np.sqrt(n)), 256)
            quantiser  = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantiser, self.dim, n_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(n_clusters, 10)

        self._is_built = True
        print(f"  Index built — {self.index.ntotal} vectors, type={self.index_type}")
        return self

    def _embed_texts(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs  = self.embedder.encode(batch, show_progress_bar=False)
            if isinstance(vecs, list):
                vecs = np.array(vecs)
            vecs  = vecs.astype(np.float32)
            # L2 normalise for cosine similarity via inner product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            all_embeddings.append(vecs / norms)
        return np.vstack(all_embeddings)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_ticker: str = None,
        filter_section: str = None,
    ) -> list:
        """
        Search the index for the most relevant chunks.

        Parameters
        ----------
        query          : natural language query
        k              : number of results to return
        filter_ticker  : restrict results to a specific ticker
        filter_section : restrict results to a specific section

        Returns list of SearchResult objects sorted by score.
        """
        if not self._is_built:
            raise RuntimeError("Call build() before search()")

        # Embed query
        q_vec = self.embedder.encode([query], show_progress_bar=False)
        if isinstance(q_vec, list):
            q_vec = np.array(q_vec)
        q_vec = q_vec.astype(np.float32)
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Search — retrieve more if filtering
        fetch_k = k * 5 if (filter_ticker or filter_section) else k
        fetch_k = min(fetch_k, len(self.documents))

        scores, indices = self.index.search(q_vec, fetch_k)
        scores   = scores[0]
        indices  = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]

            # Apply filters
            if filter_ticker and doc.ticker != filter_ticker:
                continue
            if filter_section and doc.section != filter_section:
                continue

            results.append(SearchResult(
                doc_id=doc.doc_id,
                ticker=doc.ticker,
                section=doc.section,
                text=doc.text,
                score=round(float(score), 6),
                metadata=doc.metadata,
            ))

            if len(results) >= k:
                break

        return results

    def search_multi_query(
        self,
        queries: list,
        k: int = 5,
    ) -> list:
        """
        Search with multiple query variants and deduplicate results.
        Useful for query expansion.
        """
        seen    = set()
        results = []
        for q in queries:
            for r in self.search(q, k=k):
                if r.doc_id not in seen:
                    seen.add(r.doc_id)
                    results.append(r)
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def stats(self) -> IndexStats:
        """Return index statistics."""
        tickers  = list(set(d.ticker  for d in self.documents))
        sections = list(set(d.section for d in self.documents))
        return IndexStats(
            n_documents=len(self.documents),
            n_tickers=tickers,
            n_sections=sections,
            embedding_dim=self.dim or 0,
            embedding_model=self.embedding_model_name,
            index_type=self.index_type,
        )

    def save(self, path: str) -> None:
        """Persist index and metadata to disk."""
        import faiss
        Path(path).mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(Path(path) / "index.faiss"))

        meta = {
            "embedding_model": self.embedding_model_name,
            "index_type":      self.index_type,
            "dim":             self.dim,
            "n_docs":          len(self.documents),
            "documents": [
                {
                    "doc_id":  d.doc_id,
                    "ticker":  d.ticker,
                    "form":    d.form,
                    "section": d.section,
                    "text":    d.text,
                    "metadata":d.metadata,
                }
                for d in self.documents
            ],
        }
        with open(Path(path) / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        with open(Path(path) / "embedder.pkl", "wb") as f:
            pickle.dump(self.embedder, f)

        print(f"  Saved index to {path}/")

    def load(self, path: str) -> "FAISSVectorStore":
        """Load index and metadata from disk."""
        import faiss
        from loaders.sec_loader import Document

        self.index = faiss.read_index(str(Path(path) / "index.faiss"))

        with open(Path(path) / "metadata.json") as f:
            meta = json.load(f)

        self.embedding_model_name = meta["embedding_model"]
        self.index_type = meta["index_type"]
        self.dim        = meta["dim"]
        self.documents  = [
            Document(
                doc_id=d["doc_id"], ticker=d["ticker"],
                form=d["form"], section=d["section"],
                text=d["text"], metadata=d["metadata"],
            )
            for d in meta["documents"]
        ]
        self.texts = [d.text for d in self.documents]

        with open(Path(path) / "embedder.pkl", "rb") as f:
            self.embedder = pickle.load(f)

        self._is_built = True
        print(f"  Loaded index from {path}/ ({len(self.documents)} docs)")
        return self

    def evaluate_retrieval(
        self,
        queries_with_answers: list,
        k: int = 5,
    ) -> dict:
        """
        Evaluate retrieval quality given gold-standard query-answer pairs.

        Parameters
        ----------
        queries_with_answers : list of (query, expected_ticker, expected_section) tuples

        Returns dict with precision@k, recall@k, MRR.
        """
        precisions, recalls, reciprocal_ranks = [], [], []

        for query, expected_ticker, expected_section in queries_with_answers:
            results = self.search(query, k=k)

            # Check if correct doc retrieved
            correct = [
                r for r in results
                if r.ticker == expected_ticker and r.section == expected_section
            ]

            precision = len(correct) / k
            recall    = 1.0 if correct else 0.0
            precisions.append(precision)
            recalls.append(recall)

            # MRR: position of first correct result
            rr = 0.0
            for rank, r in enumerate(results, 1):
                if r.ticker == expected_ticker and r.section == expected_section:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        return {
            f"precision@{k}": round(float(np.mean(precisions)), 4),
            f"recall@{k}":    round(float(np.mean(recalls)), 4),
            "mrr":            round(float(np.mean(reciprocal_ranks)), 4),
            "n_queries":      len(queries_with_answers),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents

    print("=" * 60)
    print("FAISS Vector Store Demo")
    print("=" * 60)

    # Load filings and build documents
    tickers  = ["AAPL", "MSFT", "TSLA"]
    all_docs = []
    for t in tickers:
        f    = load_synthetic_filing(t)
        docs = filing_to_documents(f, chunk_size=150, overlap=30)
        all_docs.extend(docs)
        print(f"  [{t}] {len(docs)} chunks")

    print(f"\n  Total documents: {len(all_docs)}")

    # Build index
    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)

    # Search
    queries = [
        "What are Apple's key risk factors?",
        "Microsoft Azure revenue growth",
        "Tesla gross margin decline reasons",
        "iPhone revenue fiscal 2023",
    ]

    print("\n  Search Results:")
    for q in queries:
        results = store.search(q, k=3)
        print(f"\n  Q: {q}")
        for r in results:
            print(f"    [{r.ticker}/{r.section}] score={r.score:.4f} | {r.text[:80]}...")

    # Stats
    s = store.stats()
    print(f"\n  Index stats:")
    print(f"    Documents : {s.n_documents}")
    print(f"    Tickers   : {s.n_tickers}")
    print(f"    Sections  : {s.n_sections}")
    print(f"    Emb dim   : {s.embedding_dim}")

    # Eval
    test_queries = [
        ("What are Apple risk factors?",    "AAPL", "risk_factors"),
        ("Microsoft Azure cloud revenue",   "MSFT", "mda"),
        ("Tesla vehicle deliveries 2023",   "TSLA", "mda"),
    ]
    eval_results = store.evaluate_retrieval(test_queries, k=5)
    print(f"\n  Retrieval Evaluation:")
    for k, v in eval_results.items():
        print(f"    {k}: {v}")

    # Save and reload
    store.save("models/faiss_index")
    store2 = FAISSVectorStore()
    store2.load("models/faiss_index")
    r = store2.search("Apple iPhone revenue", k=2)
    print(f"\n  After reload — top result: [{r[0].ticker}/{r[0].section}] score={r[0].score:.4f}")
