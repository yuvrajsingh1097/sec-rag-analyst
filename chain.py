"""
RAG Chain — Retrieval-Augmented Generation
============================================
Combines FAISS retrieval with LLM generation to answer
questions about SEC filings with grounded source citations.

Supports:
    1. OpenAI GPT-4 / GPT-3.5          (via OPENAI_API_KEY)
    2. Anthropic Claude                  (via ANTHROPIC_API_KEY)
    3. Rule-based answer synthesiser     (offline fallback)

Pipeline:
    query -> retrieve top-k chunks -> build prompt
    -> LLM generate -> parse answer + citations
"""

import re
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Citation:
    ticker:  str
    section: str
    text:    str
    score:   float


@dataclass
class RAGAnswer:
    question:    str
    answer:      str
    citations:   list
    confidence:  float
    model_used:  str
    n_chunks:    int
    is_grounded: bool


@dataclass
class ConversationTurn:
    question: str
    answer:   str


SYSTEM_PROMPT = """You are a financial analyst assistant specialising in SEC filings.
Answer questions ONLY from the provided context. Cite sources as [TICKER/SECTION]."""

QA_PROMPT_TEMPLATE = """{system_prompt}

CONTEXT:
{context}

HISTORY:
{history}

QUESTION: {question}

ANSWER (cite as [TICKER/SECTION]):"""

CONTEXT_CHUNK_TEMPLATE = """[{ticker} | {section} | {form} {period}]
{text}
---"""

CITATION_RE = re.compile(r"\[([A-Z]{1,5})/([a-z_]+)\]", re.IGNORECASE)

FINANCIAL_STOPWORDS = {"what","are","is","the","a","an","in","of","for",
                       "to","how","much","does","do","with","and","or","its"}
NUMBER_RE = re.compile(
    r"\$[\d,\.]+\s*(billion|million)?|[\d,\.]+\s*(percent|%|billion|million|bps)",
    re.I
)


class RuleBasedSynthesiser:
    """Offline synthesiser — no API key needed."""

    def _keywords(self, query: str) -> set:
        words = re.findall(r"\b[a-z]+\b", query.lower())
        return {w for w in words if w not in FINANCIAL_STOPWORDS and len(w) > 2}

    def _top_sentences(self, text: str, keywords: set, n: int = 3) -> list:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        scored = []
        for s in sentences:
            if len(s.split()) < 6:
                continue
            sl = s.lower()
            score = sum(1 for kw in keywords if kw in sl) * 2 + len(NUMBER_RE.findall(s))
            if score > 0:
                scored.append((score, s.strip()))
        scored.sort(reverse=True)
        return [s for _, s in scored[:n]]

    def synthesise(self, question: str, results: list) -> str:
        if not results:
            return "The provided filings do not contain information about this question."
        keywords = self._keywords(question)
        parts = []
        seen = set()
        for r in results:
            key = f"{r.ticker}/{r.section}"
            if key in seen:
                continue
            seen.add(key)
            sents = self._top_sentences(r.text, keywords, n=3)
            if sents:
                parts.append(f"{' '.join(sents)} [{r.ticker}/{r.section}]")
            if len(parts) >= 3:
                break
        if not parts:
            r = results[0]
            parts.append(f"{' '.join(r.text.split()[:60])}... [{r.ticker}/{r.section}]")
        return " ".join(parts)


def _call_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def parse_citations(answer: str, results: list) -> list:
    citations, seen = [], set()
    for m in CITATION_RE.finditer(answer):
        ticker, section = m.group(1).upper(), m.group(2).lower()
        key = f"{ticker}/{section}"
        if key in seen:
            continue
        seen.add(key)
        matching = [r for r in results if r.ticker == ticker and r.section == section]
        if matching:
            best = max(matching, key=lambda r: r.score)
            citations.append(Citation(ticker, section, best.text[:200]+"...", best.score))
    return citations


def is_grounded(answer: str, results: list) -> bool:
    if not results:
        return False
    if CITATION_RE.search(answer):
        return True
    answer_nums = set(re.findall(r"\d+\.?\d*", answer))
    for r in results:
        if answer_nums & set(re.findall(r"\d+\.?\d*", r.text)):
            return True
    return False


def confidence_score(answer: str, citations: list, results: list) -> float:
    if not results:
        return 0.0
    word_score = min(len(answer.split()) / 50, 1.0)
    cite_score = min(len(citations) / 3, 1.0)
    ret_score  = float(np.mean([r.score for r in results[:3]]))
    return round(float(0.3 * word_score + 0.3 * cite_score + 0.4 * ret_score), 4)


class RAGChain:
    """Full RAG pipeline: retrieve -> prompt -> generate -> cite."""

    def __init__(self, vector_store, llm: str = "rule_based",
                 k: int = 5, max_history: int = 3):
        self.store       = vector_store
        self.llm         = llm
        self.k           = k
        self.max_history = max_history
        self.history     = []
        self.synthesiser = RuleBasedSynthesiser()

    def _build_context(self, results: list) -> str:
        return "\n".join(
            CONTEXT_CHUNK_TEMPLATE.format(
                ticker=r.ticker, section=r.section,
                form=r.metadata.get("form","10-K"),
                period=r.metadata.get("period",""),
                text=r.text,
            )
            for r in results
        )

    def _build_history(self) -> str:
        if not self.history:
            return "None"
        return "\n".join(
            f"Q: {t.question}\nA: {t.answer[:150]}..."
            for t in self.history[-self.max_history:]
        )

    def ask(self, question: str,
            filter_ticker: str = None,
            filter_section: str = None) -> RAGAnswer:
        results = self.store.search(
            question, k=self.k,
            filter_ticker=filter_ticker,
            filter_section=filter_section,
        )
        if not results:
            return RAGAnswer(question, "No relevant information found.",
                             [], 0.0, self.llm, 0, False)

        if self.llm == "rule_based":
            answer_text = self.synthesiser.synthesise(question, results)
            model_used  = "rule_based_synthesiser"
        else:
            prompt = QA_PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                context=self._build_context(results),
                history=self._build_history(),
                question=question,
            )
            try:
                answer_text = (_call_openai(prompt) if self.llm == "openai"
                               else _call_anthropic(prompt))
                model_used  = self.llm
            except Exception as e:
                answer_text = self.synthesiser.synthesise(question, results)
                model_used  = f"rule_based_fallback ({e})"

        citations = parse_citations(answer_text, results)
        grounded  = is_grounded(answer_text, results)
        conf      = confidence_score(answer_text, citations, results)

        self.history.append(ConversationTurn(question, answer_text))

        return RAGAnswer(
            question=question, answer=answer_text,
            citations=citations, confidence=conf,
            model_used=model_used, n_chunks=len(results),
            is_grounded=grounded,
        )

    def ask_batch(self, questions: list) -> list:
        return [self.ask(q) for q in questions]

    def reset_history(self) -> None:
        self.history = []

    def decompose_query(self, question: str) -> list:
        parts = re.split(r"\band\b|\balso\b|\bfurthermore\b", question, flags=re.I)
        sub_qs = []
        for p in parts:
            p = p.strip()
            if len(p.split()) >= 3:
                if not p[0].isupper():
                    p = p[0].upper() + p[1:]
                if not p.endswith("?"):
                    p += "?"
                sub_qs.append(p)
        return sub_qs if len(sub_qs) > 1 else [question]


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from loaders.sec_loader import load_synthetic_filing, filing_to_documents
    from rag.vector_store import FAISSVectorStore

    print("=" * 60)
    print("RAG Chain Demo")
    print("=" * 60)

    all_docs = []
    for t in ["AAPL", "MSFT", "TSLA"]:
        f = load_synthetic_filing(t)
        all_docs.extend(filing_to_documents(f, chunk_size=150, overlap=30))

    store = FAISSVectorStore(embedding_model="tfidf")
    store.build(all_docs)
    chain = RAGChain(store, llm="rule_based", k=5)

    questions = [
        "What are Apple's key risk factors?",
        "How much revenue did Microsoft report in fiscal 2023?",
        "Why did Tesla's gross margin decline?",
    ]

    for q in questions:
        ans = chain.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {ans.answer[:200]}...")
        print(f"   Conf={ans.confidence:.3f} | Grounded={ans.is_grounded} | Citations={len(ans.citations)}")
