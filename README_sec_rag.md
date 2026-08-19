# SEC RAG Analyst — Financial Document Q&A System

Retrieval-Augmented Generation (RAG) system over SEC 10-K/10-Q filings. Ask natural language questions like "What are Apple's key risk factors?" and get grounded answers with source citations, powered by FAISS vector search and LLM generation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-35%20passing-brightgreen)](#testing)

---

## What this does

| Module | Description |
|--------|-------------|
| `loaders/sec_loader.py` | SEC EDGAR loader, PDF parser, section extractor, text chunker |
| `rag/vector_store.py` | FAISS index build, embedding, similarity search, persistence |
| `rag/chain.py` | RAG chain with LLM, prompt engineering, source citation |
| `rag/reranker.py` | Cross-encoder reranker for improved retrieval precision |
| `eval/ragas_eval.py` | RAGAS evaluation: faithfulness, relevance, context recall |
| `app/chat_ui.py` | Streamlit chat interface with source panel |

---

## Architecture

```
10-K Filing → Section Extractor → Text Chunker → Embedding Model
                                                        ↓
User Query → Query Embedding → FAISS Search → Top-K Chunks
                                                        ↓
                              Reranker → LLM → Grounded Answer + Citations
```

---

## Sections Extracted

```
Item 1   : Business
Item 1A  : Risk Factors
Item 7   : MD&A (Management Discussion & Analysis)
Item 8   : Financial Statements
Item 9A  : Controls & Procedures
```

---

## Results

| Metric | Value |
|--------|-------|
| Retrieval precision (top-5) | TBD |
| Answer faithfulness (RAGAS) | TBD |
| Context relevance | TBD |
| Queries per second | TBD |

---

## Output Samples

![Filing Pipeline](outputs/filing_pipeline.png)

---

## Project Structure

```
sec-rag-analyst/
├── loaders/
│   └── sec_loader.py       # Filing loader, parser, chunker
├── rag/
│   ├── vector_store.py     # FAISS index + embedding
│   ├── chain.py            # RAG chain + LLM
│   └── reranker.py         # Cross-encoder reranker
├── eval/
│   └── ragas_eval.py       # RAG evaluation metrics
├── app/
│   └── chat_ui.py          # Streamlit chat interface
├── tests/
│   └── test_sec_loader.py
├── outputs/
├── data/
├── models/
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/sec-rag-analyst
cd sec-rag-analyst
pip install -r requirements.txt
python loaders/sec_loader.py
python -m pytest tests/ -v
streamlit run app/chat_ui.py
```

---

## License
MIT
