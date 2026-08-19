"""
SEC Filing Loader & Document Parser
=====================================
Loads SEC filings (10-K, 10-Q, 8-K) from:
    1. EDGAR full-text search API (live)
    2. Synthetic filing generator  (offline / testing)

Pipeline:
    fetch filing → parse text → extract sections
    → chunk text → return Document objects for RAG

Sections extracted:
    - Item 1  : Business
    - Item 1A : Risk Factors
    - Item 7  : MD&A (Management Discussion & Analysis)
    - Item 8  : Financial Statements
    - Item 9A : Controls & Procedures
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single text chunk ready for embedding and retrieval."""
    doc_id:   str
    ticker:   str
    form:     str           # 10-K | 10-Q | 8-K
    section:  str           # risk_factors | mda | business | financials
    text:     str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Filing:
    """A complete SEC filing."""
    ticker:       str
    company_name: str
    form:         str
    period:       str           # e.g. "2023-12-31"
    filed_date:   str
    sections:     dict = field(default_factory=dict)   # {section_name: text}
    raw_text:     str = ""
    source:       str = "synthetic"


# ---------------------------------------------------------------------------
# Section patterns for 10-K
# ---------------------------------------------------------------------------

SECTION_PATTERNS = {
    "business": [
        r"item\s+1[\.\s]+business",
        r"item\s+1\b",
    ],
    "risk_factors": [
        r"item\s+1a[\.\s]+risk\s+factors",
        r"risk\s+factors",
    ],
    "mda": [
        r"item\s+7[\.\s]+management.{0,30}discussion",
        r"management.{0,10}discussion\s+and\s+analysis",
    ],
    "financials": [
        r"item\s+8[\.\s]+financial\s+statements",
        r"financial\s+statements\s+and\s+supplementary",
    ],
    "controls": [
        r"item\s+9a[\.\s]+controls",
        r"disclosure\s+controls",
    ],
}

SECTION_RE = {
    name: re.compile("|".join(patterns), flags=re.IGNORECASE)
    for name, patterns in SECTION_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_filing_text(text: str) -> str:
    """Clean raw SEC filing text."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove EDGAR header boilerplate
    text = re.sub(r"UNITED STATES.*?SECURITIES AND EXCHANGE COMMISSION.*?\n", "", text, flags=re.DOTALL)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page numbers and headers
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    # Remove table of contents entries
    text = re.sub(r"\.\s*\.\s*\.\s*\.+\s*\d+", "", text)
    return text.strip()


def extract_sections(text: str) -> dict:
    """
    Split filing text into named sections using regex anchors.
    Returns dict of {section_name: text}.
    """
    sections = {}
    text_lower = text.lower()

    # Find positions of each section
    positions = {}
    for name, pattern in SECTION_RE.items():
        match = pattern.search(text_lower)
        if match:
            positions[name] = match.start()

    # Sort by position
    ordered = sorted(positions.items(), key=lambda x: x[1])

    # Extract text between consecutive section starts
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        section_text = text[start:end].strip()
        sections[name] = clean_filing_text(section_text)

    return sections


# ---------------------------------------------------------------------------
# Text chunker
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text       : input text
    chunk_size : words per chunk
    overlap    : words of overlap between consecutive chunks

    Returns list of text chunks.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def filing_to_documents(
    filing: Filing,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list:
    """
    Convert a Filing into a list of Document chunks ready for RAG.

    Each chunk gets metadata: ticker, form, section, chunk_index.
    """
    docs = []
    for section, text in filing.sections.items():
        if not text or len(text.split()) < 20:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            doc_id = f"{filing.ticker}_{filing.form}_{section}_{i:04d}"
            docs.append(Document(
                doc_id=doc_id,
                ticker=filing.ticker,
                form=filing.form,
                section=section,
                text=chunk,
                metadata={
                    "ticker":      filing.ticker,
                    "company":     filing.company_name,
                    "form":        filing.form,
                    "period":      filing.period,
                    "filed_date":  filing.filed_date,
                    "section":     section,
                    "chunk_index": i,
                    "source":      filing.source,
                },
            ))
    return docs


# ---------------------------------------------------------------------------
# Synthetic filing generator
# ---------------------------------------------------------------------------

SYNTHETIC_FILINGS = {
    "AAPL": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "form": "10-K",
        "period": "2023-09-30",
        "filed_date": "2023-11-03",
        "sections": {
            "business": (
                "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
                "tablets, wearables, and accessories worldwide. The company sells its products "
                "through its retail and online stores and direct sales force, as well as through "
                "third-party cellular network carriers, wholesalers, retailers, and resellers. "
                "Apple also sells a variety of related software, services, accessories, and "
                "third-party digital content and applications. The Company's products include "
                "iPhone, Mac, iPad, Apple Watch, AirPods, Apple TV, HomePod, and Beats products. "
                "Apple's Services segment includes advertising, AppleCare, cloud services, digital "
                "content, and payment services. The Company was incorporated in California in 1977. "
                "Apple operates through five geographic segments: Americas, Europe, Greater China, "
                "Japan, and Rest of Asia Pacific. The Company employs approximately 161,000 "
                "full-time equivalent employees. Apple is committed to innovation through research "
                "and development and focuses on privacy and security as core product values. "
                "The Company's retail stores serve as important points of contact with customers "
                "and provide a controlled environment for product demonstration and support."
            ),
            "risk_factors": (
                "Apple faces significant competition across all its product categories. "
                "The markets for its products and services are highly competitive and characterized "
                "by rapid technological change. The Company competes with global and regional "
                "competitors including Samsung, Google, Microsoft, and others. "
                "Global macroeconomic conditions including inflation, interest rate changes, "
                "and foreign exchange fluctuations significantly impact Apple's business. "
                "A substantial portion of the Company's revenue comes from a relatively small "
                "number of products, particularly iPhone. Any decline in iPhone demand could "
                "adversely affect financial results. The Company is subject to complex and "
                "changing laws and regulations globally including data privacy laws, antitrust "
                "regulations, and tax laws. Changes in these regulations could increase costs. "
                "Apple depends on third-party manufacturers primarily located in Asia. "
                "Supply chain disruptions, component shortages, or geopolitical tensions "
                "could impact the ability to manufacture and deliver products. "
                "The Company is exposed to credit risk, interest rate risk, and market risk. "
                "Cybersecurity threats and data breaches pose ongoing risks to the Company. "
                "Climate change and related regulations could increase operational costs."
            ),
            "mda": (
                "For fiscal year 2023, Apple reported total net sales of $383.3 billion compared "
                "to $394.3 billion in fiscal 2022, a decrease of 3 percent. iPhone net sales "
                "decreased 2 percent to $200.6 billion. Mac net sales decreased 27 percent to "
                "$29.4 billion reflecting weak consumer demand and challenging macro environment. "
                "iPad net sales decreased 3 percent to $28.3 billion. Wearables, Home and "
                "Accessories net sales decreased 3 percent to $39.8 billion. Services net sales "
                "increased 9 percent to $85.2 billion to a record high driven by growth in "
                "advertising, App Store, AppleCare, iCloud, and Apple Music. "
                "Gross margin was 44.1 percent compared to 43.3 percent in fiscal 2022. "
                "Products gross margin was 36.6 percent and Services gross margin was 70.8 percent. "
                "Operating expenses were $54.8 billion. Research and development expense was "
                "$29.9 billion. Net income was $97.0 billion, or $6.13 per diluted share. "
                "The Company generated operating cash flow of $114.0 billion. "
                "Capital expenditures were $10.7 billion. The Company returned over $89 billion "
                "to shareholders through dividends and share repurchases during fiscal 2023. "
                "Cash and marketable securities totaled $166.6 billion at year end. "
                "For fiscal 2024, the Company expects revenue growth in the low-single digits."
            ),
            "risk_factors_extended": (
                "The Company's operations involve the use of various raw materials including "
                "rare earth elements, aluminum, glass, and silicon. Shortages or price increases "
                "in these materials could adversely affect gross margins. "
                "Apple has significant exposure to foreign exchange risk given that a substantial "
                "portion of revenue is denominated in currencies other than the US dollar. "
                "Strengthening of the US dollar negatively impacts reported revenues and earnings. "
                "The Company faces significant litigation and regulatory risks across jurisdictions. "
                "The European Commission and various national competition authorities have ongoing "
                "investigations into the Company's business practices. Adverse outcomes could "
                "result in significant fines or changes to business practices."
            ),
        },
    },
    "MSFT": {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "form": "10-K",
        "period": "2023-06-30",
        "filed_date": "2023-07-27",
        "sections": {
            "business": (
                "Microsoft Corporation enables digital transformation for the era of an intelligent "
                "cloud and an intelligent edge. The Company's mission is to empower every person "
                "and every organization on the planet to achieve more. Microsoft operates through "
                "three segments: Productivity and Business Processes, Intelligent Cloud, and "
                "More Personal Computing. The Productivity and Business Processes segment includes "
                "Office 365 Commercial, Microsoft 365 Consumer, LinkedIn, and Dynamics 365. "
                "The Intelligent Cloud segment includes Azure and other cloud services, SQL Server, "
                "Windows Server, Visual Studio, and GitHub. Azure is the Company's comprehensive "
                "cloud computing platform offering over 200 products and cloud services. "
                "The More Personal Computing segment includes Windows OEM, Xbox content and "
                "services, Surface devices, and Search and news advertising. "
                "Microsoft employs approximately 238,000 employees globally. "
                "The Company invests heavily in artificial intelligence including partnerships "
                "with OpenAI and integration of AI across all products through Microsoft Copilot. "
                "Azure OpenAI Service provides enterprise customers access to GPT-4 and other "
                "large language models. GitHub Copilot has transformed software development."
            ),
            "risk_factors": (
                "Microsoft faces intense competition across all business segments from global "
                "technology companies including Amazon Web Services, Google Cloud, Apple, "
                "Salesforce, and many others. Cloud infrastructure competition is particularly "
                "intense with significant pricing pressure. "
                "Cybersecurity risks represent a material threat to Microsoft's business. "
                "The Company experiences sophisticated cyberattacks including nation-state actors. "
                "A significant breach could result in financial losses and reputational damage. "
                "Regulatory and legal risks continue to grow as governments worldwide increase "
                "scrutiny of large technology companies. Antitrust investigations in the EU "
                "and US could result in significant fines or operational changes. "
                "Microsoft's acquisition of Activision Blizzard is subject to regulatory approval "
                "in multiple jurisdictions with uncertain outcomes. "
                "AI technologies present new risks including model accuracy, bias, intellectual "
                "property concerns, and evolving regulatory requirements. "
                "The Company is exposed to foreign currency exchange rate fluctuations given "
                "that approximately 48 percent of revenue comes from outside the United States. "
                "Economic uncertainty, inflation, and interest rate changes impact customer "
                "spending on technology and cloud services."
            ),
            "mda": (
                "For fiscal year 2023, Microsoft reported total revenue of $211.9 billion, "
                "up 7 percent year over year. Intelligent Cloud revenue was $87.9 billion, "
                "up 19 percent driven by Azure growth of 29 percent. "
                "Productivity and Business Processes revenue was $69.3 billion, up 9 percent "
                "with Office 365 Commercial revenue growing 13 percent. "
                "More Personal Computing revenue decreased 9 percent to $54.7 billion reflecting "
                "weakness in PC market and gaming. "
                "Gross margin was 69 percent, up from 68 percent in fiscal 2022. "
                "Intelligent Cloud segment gross margin was 72 percent. "
                "Operating income was $88.5 billion, up 6 percent. "
                "Net income was $72.4 billion, up 20 percent year over year. "
                "Diluted earnings per share were $9.72, up 20 percent. "
                "The Company returned $12.4 billion to shareholders through dividends and "
                "buybacks in fiscal Q4. Capital expenditures were $28.1 billion for the year "
                "driven by cloud and AI infrastructure investment. "
                "Cash and equivalents were $111.3 billion at fiscal year end. "
                "Microsoft expects double-digit revenue and operating income growth in fiscal 2024 "
                "driven by continued Azure momentum and AI monetization across products."
            ),
        },
    },
    "TSLA": {
        "ticker": "TSLA",
        "company_name": "Tesla Inc.",
        "form": "10-K",
        "period": "2023-12-31",
        "filed_date": "2024-01-26",
        "sections": {
            "business": (
                "Tesla Inc. designs, develops, manufactures, leases, and sells electric vehicles, "
                "energy generation and storage systems, and related services. "
                "The Company operates through two segments: Automotive and Energy Generation "
                "and Storage. The Automotive segment includes design, development, manufacturing, "
                "sales, and leasing of electric vehicles including Model S, Model 3, Model X, "
                "Model Y, Cybertruck, and Tesla Semi. The Company also sells Full Self-Driving "
                "capability, over-the-air software updates, and vehicle insurance. "
                "Tesla operates Gigafactories in Fremont California, Austin Texas, Berlin Germany, "
                "and Shanghai China. The Company's Supercharger network is the largest fast "
                "charging network globally with over 50,000 connectors. "
                "Tesla's Energy segment includes Powerwall residential batteries, Megapack "
                "utility-scale storage systems, and Solar Roof products. "
                "The Company employs approximately 140,473 full-time employees globally. "
                "Tesla is developing autonomous driving technology through its neural net-based "
                "Full Self-Driving system trained on billions of miles of real-world data."
            ),
            "risk_factors": (
                "Tesla faces increasing competition from established automakers including "
                "Volkswagen Group, General Motors, Ford, and Chinese manufacturers including "
                "BYD and NIO who are rapidly expanding electric vehicle production. "
                "Pricing pressure and price reductions across the EV market could compress margins. "
                "The Company's growth depends on continued investment in manufacturing capacity, "
                "which requires significant capital expenditures. Construction delays or cost "
                "overruns at new Gigafactories could impact financial results. "
                "Tesla's Autopilot and Full Self-Driving systems face significant regulatory "
                "scrutiny from NHTSA and international regulators. Accidents involving these "
                "systems could result in litigation, recalls, or regulatory action. "
                "Battery cell supply is critical and the Company is investing in its own "
                "4680 battery cell production, which faces manufacturing challenges. "
                "The Company is dependent on raw materials including lithium, cobalt, nickel, "
                "and other minerals for battery production. Price volatility and supply "
                "constraints could adversely affect vehicle production and margins."
            ),
            "mda": (
                "For fiscal year 2023, Tesla reported total revenues of $96.8 billion, "
                "up 19 percent year over year. Automotive revenue was $82.4 billion. "
                "The Company delivered 1.81 million vehicles in 2023, up 38 percent. "
                "Model 3 and Model Y represented the majority of deliveries. "
                "Energy generation and storage revenue was $6.0 billion, up 54 percent "
                "driven by record Megapack deployments. "
                "Services and other revenue was $8.3 billion, up 37 percent. "
                "Gross margin decreased to 18.2 percent from 25.6 percent in 2022 "
                "primarily due to vehicle price reductions to stimulate demand. "
                "Automotive gross margin was 17.5 percent. "
                "Operating income was $8.9 billion, a decrease of 35 percent. "
                "Net income was $15.0 billion including a $5.9 billion income tax benefit. "
                "Adjusted EBITDA was $17.1 billion. "
                "The Company generated free cash flow of $4.4 billion. "
                "Capital expenditures were $8.9 billion for manufacturing and infrastructure. "
                "Cash and equivalents were $29.1 billion at year end. "
                "For 2024, Tesla expects vehicle volume growth to be notably lower than 2023 "
                "as the company focuses on next generation vehicle platforms."
            ),
        },
    },
}


def load_synthetic_filing(ticker: str) -> Optional[Filing]:
    """Load a synthetic 10-K filing for testing."""
    if ticker not in SYNTHETIC_FILINGS:
        return None
    data = SYNTHETIC_FILINGS[ticker]
    return Filing(
        ticker=data["ticker"],
        company_name=data["company_name"],
        form=data["form"],
        period=data["period"],
        filed_date=data["filed_date"],
        sections=data["sections"],
        raw_text=" ".join(data["sections"].values()),
        source="synthetic",
    )


def save_filing_json(filing: Filing, output_dir: str = "data") -> str:
    """Save filing to JSON for caching."""
    Path(output_dir).mkdir(exist_ok=True)
    fname = f"{filing.ticker}_{filing.form}_{filing.period}.json"
    fpath = Path(output_dir) / fname
    payload = {
        "ticker":       filing.ticker,
        "company_name": filing.company_name,
        "form":         filing.form,
        "period":       filing.period,
        "filed_date":   filing.filed_date,
        "source":       filing.source,
        "sections":     filing.sections,
        "stats": {
            "total_words":   len(filing.raw_text.split()),
            "n_sections":    len(filing.sections),
            "section_names": list(filing.sections.keys()),
        },
    }
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    return str(fpath)


def load_filing_json(fpath: str) -> Filing:
    """Load a cached filing from JSON."""
    with open(fpath) as f:
        data = json.load(f)
    return Filing(
        ticker=data["ticker"],
        company_name=data["company_name"],
        form=data["form"],
        period=data["period"],
        filed_date=data["filed_date"],
        sections=data["sections"],
        raw_text=" ".join(data["sections"].values()),
        source=data["source"],
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SEC Filing Loader Demo")
    print("=" * 60)

    for ticker in ["AAPL", "MSFT", "TSLA"]:
        filing = load_synthetic_filing(ticker)
        docs   = filing_to_documents(filing, chunk_size=200, overlap=30)

        print(f"\n[{ticker}] {filing.company_name} — {filing.form} {filing.period}")
        print(f"  Sections  : {list(filing.sections.keys())}")
        print(f"  Total words: {len(filing.raw_text.split()):,}")
        print(f"  Chunks    : {len(docs)}")
        print(f"  Sample chunk ({docs[0].section}):")
        print(f"    \"{docs[0].text[:120]}...\"")

        fpath = save_filing_json(filing)
        print(f"  Saved to  : {fpath}")
