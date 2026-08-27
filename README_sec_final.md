# SEC RAG Analyst — Financial Document Q&A System

Retrieval-Augmented Generation (RAG) over SEC 10-K/10-Q filings. Ask natural language questions and get grounded answers with source citations, powered by FAISS + BM25 hybrid retrieval, cross-encoder reranking, and RAGAS evaluation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-183%20passing-brightgreen)](#testing)

---

## What this does

| Module | Description |
|--------|-------------|
| `loaders/sec_loader.py` | SEC EDGAR loader, section extractor, text chunker, JSON storage |
| `rag/vector_store.py` | FAISS index, TF-IDF/sentence-transformer embeddings, similarity search, persistence |
| `rag/chain.py` | RAG chain with rule-based synthesiser, citation parser, confidence scoring, conversation history |
| `rag/reranker.py` | BM25 sparse retrieval, RRF fusion, cross-encoder reranker, hybrid pipeline |
| `eval/ragas_eval.py` | RAGAS-style evaluation (precision, recall, faithfulness, relevance) + financial metric extractor |
| `app/chat_ui.py` | Streamlit chat interface with source panel and eval scores |

---

## Architecture

```
10-K Filing → Section Extractor → Text Chunker (400 words, 50 overlap)
                                          ↓
                              TF-IDF / Sentence-Transformer Embeddings
                                          ↓
                                   FAISS Index (IndexFlatIP)
                                          ↓
Query → [Dense FAISS] + [BM25 Sparse] → RRF Fusion → Cross-Encoder Rerank
                                                              ↓
                                         Prompt Template → LLM / Rule-based
                                                              ↓
                                         Answer + Citations + RAGAS Score
```

---

## Results

| Metric | Score |
|--------|-------|
| Context Precision | 0.429 |
| Context Recall | 0.860 |
| Faithfulness | 1.000 |
| Answer Relevance | 0.497 |
| Composite Score | 0.722 |
| Avg Latency | 0.7ms |

---

## Output Samples

![Dashboard Preview](outputs/dashboard_preview.png)
![RAGAS Evaluation](outputs/ragas_evaluation.png)
![Hybrid Retrieval](outputs/hybrid_retrieval.png)
![RAG Chain](outputs/rag_chain.png)
![Vector Store](outputs/vector_store.png)
![Filing Pipeline](outputs/filing_pipeline.png)

---

## Project Structure

```
sec-rag-analyst/
├── loaders/sec_loader.py
├── rag/
│   ├── vector_store.py
│   ├── chain.py
│   └── reranker.py
├── eval/ragas_eval.py
├── app/chat_ui.py
├── tests/  (183 tests)
└── outputs/
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/sec-rag-analyst
cd sec-rag-analyst
pip install -r requirements.txt
streamlit run app/chat_ui.py
```

---

## License
MIT
