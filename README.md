# SEC Filing Intelligence & Risk-Delta Agent

A LangGraph multi-agent pipeline that diffs a company's risk factors,
litigation exposure, and financial statements release-over-release — using
**free** SEC EDGAR data and the **free tier of Google Gemini**. Just type a
company name (typos fine); the agent resolves the ticker, pulls the two most
recent filings, and only spends LLM calls on sections that actually changed.

Full design write-up, diagrams, and reasoning: **[`DOCUMENTATION.md`](./DOCUMENTATION.md)**

## Setup

```bash
python -m venv venv && source venv/bin/activate   # (venv\Scripts\activate on Windows)
pip install -r requirements.txt
cp .env.example .env    # then edit .env with your free Gemini API key
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey. No key is
needed for the SEC EDGAR calls — just a descriptive User-Agent header, which
`.env.example` already sets up.

## Try it without the UI first

```bash
python scripts/demo_apple_example.py
```

This prints each pipeline stage for AAPL using **live** SEC data: CIK
resolution, the two most recent 10-Ks, extracted Risk Factors sections, and
the paragraph-level diff — no LLM key required for this part. Verified while
building this project: Apple's FY2025 10-K is accession `0000320193-25-000079`,
filed 2025-10-31, covering the period ended 2025-09-27.

## Free data sources used

| Source | Endpoint | Auth |
|---|---|---|
| SEC EDGAR submissions | `https://data.sec.gov/submissions/CIK{cik}.json` | none (User-Agent header only) |
| SEC EDGAR XBRL company facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` | none |
| SEC EDGAR XBRL concept | `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json` | none |
| SEC full text search | `https://efts.sec.gov/LATEST/search-index` | none |
| SEC ticker→CIK map | `https://www.sec.gov/files/company_tickers.json` | none |
| Google Gemini API | `gemini-2.5-flash` via `google-genai` | free API key |

Rate limit: SEC allows 10 req/sec per IP (we sleep ~0.12s between calls, see
`config.py`). Check https://ai.google.dev/gemini-api/docs/models for current
free-tier limits and model names — they change over time.

## Architecture

```
resolve_ticker -> fetch_filings -> parse_sections -> diff_all_sections -> orchestrate
      -> (conditional fan-out) -> {financial_agent, litigation_agent, sentiment_agent}
      -> reconcile -> Streamlit report
```

- **`resolve_ticker`** turns free-text company input ("Aple", "the iPhone
  company", or a real ticker) into a verified SEC ticker — local fuzzy match
  first, LLM only for informal names it can't handle, and the LLM's guess is
  always re-verified against real SEC data before being trusted.
- **`orchestrate`** inspects the diff summaries and only dispatches the
  litigation agent if Legal Proceedings materially changed, and the sentiment
  agent if Risk Factors materially changed — the financial agent always runs
  since XBRL pulls are cheap. This conditional routing is what LangGraph's
  `add_conditional_edges` gives you over a fixed pipeline.
- **`sentiment_agent`** queries long-term memory (Chroma) for risk language
  seen in earlier filings before it runs, so it can tell a genuinely new risk
  apart from a reworded restatement of one already on record.
- **`reconcile`** always writes the current filing's risk paragraphs back to
  memory, regardless of which agents ran, so every run grows the record.

## Folder structure

```
sec_risk_delta/
├── app.py                    # Streamlit frontend
├── config.py                 # all endpoints/models/paths in one place
├── requirements.txt
├── .env.example
├── scripts/
│   └── demo_apple_example.py # worked example, step-by-step
├── src/
│   ├── data/
│   │   ├── edgar_client.py     # submissions, full-text search, doc fetch, company index
│   │   ├── xbrl_client.py      # structured financials
│   │   └── ticker_resolver.py  # free-text company name -> verified ticker
│   ├── parsing/
│   │   └── filing_parser.py  # Item 1A/3/7 section extraction
│   ├── diffing/
│   │   └── diff_engine.py    # textual + fuzzy-semantic paragraph diff
│   ├── llm/
│   │   └── llm_gemini.py     # single-provider Gemini chat wrapper
│   ├── agents/
│   │   ├── state.py          # shared LangGraph state schema
│   │   ├── nodes.py          # every node incl. orchestrator + 3 specialists
│   │   └── graph.py          # graph assembly + conditional routing
│   ├── memory/
│   │   └── vector_store.py   # Chroma: cross-filing semantic risk search
│   └── report/
│       └── report_builder.py # final_report dict -> Markdown
```

## Document parsing: unstructured.io

`filing_parser.py` uses **unstructured.io** (`partition_html`) as its primary
parser, not plain BeautifulSoup. It classifies each block of the filing
(`NarrativeText`, `Title`, `Table`, ...) instead of returning one flat
string, so tables are kept as distinct elements rather than getting
flattened into surrounding prose — see `extract_sections_structured()` for
the element-level output, and `partition_filing()` for the raw element list.
A BeautifulSoup + regex fallback (`_extract_sections_bs4_fallback`) kicks in
automatically if `unstructured` isn't installed or throws on a malformed
document, so the pipeline never hard-fails on parsing alone.

Docling was the other option named in the original brief; it's not wired in
because it pulls in a heavier layout-model stack better suited to scanned
PDFs. For HTML filings, unstructured.io is the lighter free choice.

## Upgrade paths (for a stronger portfolio piece)

- Have `financial_agent` (in `src/agents/nodes.py`) read the `Table`
  elements from `extract_sections_structured()` directly, instead of only
  using XBRL — some numbers only appear in prose tables, not XBRL tags.
- Add a `mkdocs`/PDF export of `report_builder.to_markdown()`.
- Cache filing HTML to `data/cache/` keyed by accession number so re-runs
  don't re-hit EDGAR.
- Build a proper disambiguation UI for `ticker_resolver.py`'s "ambiguous"
  status instead of surfacing it as a plain error message.
