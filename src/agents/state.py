from typing import TypedDict, Optional
from src.diffing.diff_engine import SectionDiff


class RiskDeltaState(TypedDict, total=False):
    company_query: str                   # raw user input: "Apple", "Aple", "AAPL", etc.
    ticker: str                           # resolved by resolve_ticker node -- not user input
    resolved_company_title: str
    ticker_resolution_method: str         # "local_fuzzy" or "llm_assisted"
    cik: str
    form_type: str                       # "10-K" or "10-Q"

    old_filing_meta: dict
    new_filing_meta: dict
    old_html: str
    new_html: str

    old_sections: dict[str, str]
    new_sections: dict[str, str]

    risk_factors_diff: SectionDiff
    legal_diff: SectionDiff
    mdna_diff: SectionDiff

    route_to: list[str]                  # which specialist agents the orchestrator dispatches

    financial_findings: Optional[str]
    financial_provider: Optional[str]
    litigation_findings: Optional[str]
    litigation_provider: Optional[str]
    sentiment_findings: Optional[str]

    final_report: Optional[dict]
