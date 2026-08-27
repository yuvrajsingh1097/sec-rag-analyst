"""
Streamlit Chat UI — SEC RAG Analyst
=====================================
Interactive chat interface over SEC 10-K filings.

Features:
    - Ticker selector (AAPL / MSFT / TSLA)
    - Natural language Q&A with source citations
    - Hybrid retrieval toggle (dense vs BM25+RRF)
    - Source panel with retrieved chunks
    - Financial metric extraction panel
    - RAGAS evaluation scores per answer
    - Conversation history

Run with: streamlit run app/chat_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loaders.sec_loader import load_synthetic_filing, filing_to_documents, SYNTHETIC_FILINGS
from rag.vector_store import FAISSVectorStore
from rag.chain import RAGChain
from rag.reranker import HybridRetriever
from eval.ragas_eval import (
    RAGEvaluator, EvalSample, FinancialMetricExtractor, build_test_cases
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SEC RAG Analyst",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .chat-user { background:#1c2128; border-left:3px solid #58a6ff;
                 padding:10px 14px; border-radius:6px; margin:8px 0; }
    .chat-bot  { background:#161b22; border-left:3px solid #3fb950;
                 padding:10px 14px; border-radius:6px; margin:8px 0; }
    .citation-box { background:#0d1117; border:1px solid #30363d;
                    border-radius:6px; padding:8px 12px; margin:4px 0;
                    font-size:12px; color:#8b949e; }
    .metric-pill { display:inline-block; background:#21262d; border:1px solid #30363d;
                   border-radius:20px; padding:2px 10px; font-size:11px;
                   color:#e6edf3; margin:2px; }
    h1,h2,h3 { color:#e6edf3; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def build_rag_system(tickers, use_hybrid):
    all_docs, filings = [], []
    for t in tickers:
        f = load_synthetic_filing(t)
        filings.append(f)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))

    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)

    if use_hybrid:
        retriever = HybridRetriever(store, all_docs, use_reranker=False)
        chain = RAGChain(store, llm="rule_based", k=5)
        chain._hybrid = retriever
    else:
        chain = RAGChain(store, llm="rule_based", k=5)

    return chain, store, filings, all_docs


@st.cache_data
def get_eval_report(_chain, tickers):
    evaluator  = RAGEvaluator()
    test_cases = build_test_cases(list(tickers))
    return evaluator.evaluate_chain(_chain, test_cases[:4])


@st.cache_data
def get_metric_table(_filings):
    extractor = FinancialMetricExtractor()
    return extractor.build_comparison_table(_filings)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📄 SEC RAG Analyst")
st.sidebar.markdown("---")

available = list(SYNTHETIC_FILINGS.keys())
selected_tickers = st.sidebar.multiselect(
    "Filings loaded", available, default=available
)
if not selected_tickers:
    selected_tickers = available

use_hybrid = st.sidebar.toggle("Hybrid retrieval (BM25 + FAISS)", value=False)
k_results  = st.sidebar.slider("Chunks retrieved (k)", 2, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sections indexed**")
for sec in ["business","risk_factors","mda","financials"]:
    st.sidebar.markdown(f"- `{sec}`")

st.sidebar.markdown("---")
page = st.sidebar.radio("View", [
    "💬 Chat",
    "📊 Evaluation",
    "💰 Financial Metrics",
    "🗂️ Filing Explorer",
])

# ---------------------------------------------------------------------------
# Load system
# ---------------------------------------------------------------------------

with st.spinner("Building RAG system..."):
    chain, store, filings, all_docs = build_rag_system(
        tuple(selected_tickers), use_hybrid
    )
    chain.k = k_results

evaluator = RAGEvaluator()
extractor = FinancialMetricExtractor()


# ---------------------------------------------------------------------------
# Page: Chat
# ---------------------------------------------------------------------------

if page == "💬 Chat":
    st.title("💬 Ask about SEC Filings")

    col1, col2 = st.columns([2, 1])
    with col1:
        filter_ticker  = st.selectbox("Filter by ticker (optional)",
                                      ["All"] + selected_tickers)
        filter_section = st.selectbox("Filter by section (optional)",
                                      ["All","business","risk_factors","mda","financials"])
    with col2:
        show_sources = st.toggle("Show source chunks", value=True)
        show_scores  = st.toggle("Show RAGAS scores",  value=True)

    ft = None if filter_ticker  == "All" else filter_ticker
    fs = None if filter_section == "All" else filter_section

    # Suggested questions
    st.markdown("**Suggested questions:**")
    suggestions = [
        "What are Apple's key risk factors?",
        "How fast did Microsoft Azure grow in 2023?",
        "Why did Tesla's gross margin decline?",
        "Compare Apple and Microsoft profitability",
    ]
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q[:35] + "...", use_container_width=True):
            st.session_state["pending_q"] = q

    st.markdown("---")

    # Chat history init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)
            if show_sources and msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="citation-box">'
                            f'<b>[{src["ticker"]}/{src["section"]}]</b> '
                            f'score={src["score"]:.4f}<br>{src["text"][:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            if show_scores and msg.get("scores"):
                s = msg["scores"]
                cols_s = st.columns(5)
                for col, (label, val) in zip(cols_s, s.items()):
                    col.metric(label, f"{val:.3f}")

    # Input
    pending = st.session_state.pop("pending_q", None)
    question = st.chat_input("Ask about any SEC filing...") or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-user">👤 {question}</div>',
                    unsafe_allow_html=True)

        with st.spinner("Retrieving and generating answer..."):
            t0  = time.time()
            ans = chain.ask(question, filter_ticker=ft, filter_section=fs)
            lat = (time.time() - t0) * 1000

        st.markdown(f'<div class="chat-bot">🤖 {ans.answer}</div>',
                    unsafe_allow_html=True)

        # Sources
        sources = []
        if show_sources:
            retrieved = store.search(question, k=k_results,
                                     filter_ticker=ft, filter_section=fs)
            sources = [{"ticker": r.ticker, "section": r.section,
                        "score": r.score, "text": r.text} for r in retrieved]
            with st.expander("📚 Sources", expanded=True):
                for src in sources:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<b>[{src["ticker"]}/{src["section"]}]</b> '
                        f'score={src["score"]:.4f}<br>{src["text"][:200]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # RAGAS scores
        scores = {}
        if show_scores:
            ctx_chunks = [r["text"] for r in sources] if sources else []
            sample = EvalSample(
                question=question,
                ground_truth=ans.answer,
                context_chunks=ctx_chunks,
                generated_answer=ans.answer,
                latency_ms=lat,
            )
            result = evaluator.evaluate_sample(sample)
            scores = {
                "Precision":   result.context_precision,
                "Recall":      result.context_recall,
                "Faithful":    result.faithfulness,
                "Relevance":   result.answer_relevance,
                "Composite":   result.composite_score,
            }
            cols_s = st.columns(5)
            for col, (label, val) in zip(cols_s, scores.items()):
                col.metric(label, f"{val:.3f}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": ans.answer,
            "sources": sources,
            "scores": scores,
        })

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        chain.reset_history()
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Evaluation
# ---------------------------------------------------------------------------

elif page == "📊 Evaluation":
    st.title("📊 RAG System Evaluation")

    with st.spinner("Running evaluation..."):
        report = get_eval_report(chain, tuple(selected_tickers))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Context Precision", f"{report.avg_precision:.3f}")
    col2.metric("Context Recall",    f"{report.avg_recall:.3f}")
    col3.metric("Faithfulness",      f"{report.avg_faithfulness:.3f}")
    col4.metric("Answer Relevance",  f"{report.avg_relevance:.3f}")
    col5.metric("Composite Score",   f"{report.avg_composite:.3f}")

    st.markdown("---")
    st.subheader("Per-Question Breakdown")
    df = evaluator.to_dataframe(report)
    st.dataframe(
        df.style.background_gradient(subset=["composite","faithfulness"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Metric Radar")
    metrics = ["Precision","Recall","Faithfulness","Relevance","Composite"]
    values  = [report.avg_precision, report.avg_recall,
               report.avg_faithfulness, report.avg_relevance, report.avg_composite]
    st.bar_chart(dict(zip(metrics, values)))


# ---------------------------------------------------------------------------
# Page: Financial Metrics
# ---------------------------------------------------------------------------

elif page == "💰 Financial Metrics":
    st.title("💰 Financial Metric Extraction")

    table = get_metric_table(filings)
    st.subheader("Cross-Ticker Metric Comparison")

    display_metrics = ["revenue","gross_margin","growth","net_income","operating_income"]
    available_cols  = [c for c in display_metrics if c in table.columns]
    if available_cols:
        display_df = table[available_cols].copy()
        display_df.columns = [c.replace("_"," ").title() for c in available_cols]
        st.dataframe(
            display_df.style.background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Extract Metrics from Custom Text")
    custom_text = st.text_area(
        "Paste any financial text:",
        value="Apple reported total net sales of $383.3 billion, with gross margin of 44.1 percent.",
        height=100,
    )
    if st.button("Extract Metrics"):
        result = extractor.extract_all(custom_text)
        found  = {k: v for k, v in result.items() if k != "ticker" and v is not None}
        if found:
            cols = st.columns(min(len(found), 4))
            for col, (metric, data) in zip(cols, found.items()):
                col.metric(
                    metric.replace("_"," ").title(),
                    f"{data['value_raw']:.1f} {data['unit']}",
                )
        else:
            st.info("No financial metrics detected.")


# ---------------------------------------------------------------------------
# Page: Filing Explorer
# ---------------------------------------------------------------------------

elif page == "🗂️ Filing Explorer":
    st.title("🗂️ Filing Explorer")

    sel_ticker = st.selectbox("Select ticker", selected_tickers)
    filing = next((f for f in filings if f.ticker == sel_ticker), None)

    if filing:
        col1, col2, col3 = st.columns(3)
        col1.metric("Company",    filing.company_name)
        col2.metric("Form",       filing.form)
        col3.metric("Period",     filing.period)

        st.markdown("---")
        st.subheader("Sections")
        for section, text in filing.sections.items():
            with st.expander(f"📄 {section.replace('_',' ').title()} "
                             f"({len(text.split())} words)"):
                st.write(text)

        st.markdown("---")
        st.subheader("Document Chunks")
        docs = filing_to_documents(filing, chunk_size=150, overlap=30)
        st.info(f"{len(docs)} chunks indexed from this filing")
        for i, doc in enumerate(docs[:5]):
            with st.expander(f"Chunk {i+1} — {doc.section} ({doc.word_count} words)"):
                st.write(doc.text)
                st.caption(f"ID: {doc.doc_id}")
