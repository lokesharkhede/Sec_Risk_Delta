"""
Thin wrapper around SEC EDGAR's free JSON APIs.

Endpoints used (all free, no API key -- just a User-Agent header):
  - Company submissions history: https://data.sec.gov/submissions/CIK##########.json
  - Full text search (find filings by keyword): https://efts.sec.gov/LATEST/search-index
  - Ticker -> CIK map: https://www.sec.gov/files/company_tickers.json
  - Raw filing documents: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}

Rate limit: 10 requests/sec per SEC's fair access policy. We sleep briefly
between calls to stay well under it.
"""
import time
import functools
import requests

from config import (
    SEC_USER_AGENT,
    SEC_SUBMISSIONS_URL,
    SEC_TICKER_MAP_URL,
    SEC_FULLTEXT_SEARCH_URL,
    SEC_ARCHIVES_BASE,
    SEC_RATE_LIMIT_SLEEP_SEC,
)

_session = requests.Session()
_session.headers.update({"User-Agent": SEC_USER_AGENT})


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = _session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(SEC_RATE_LIMIT_SLEEP_SEC)
    return resp


@functools.lru_cache(maxsize=1)
def get_ticker_to_cik_map() -> dict:
    """Downloads SEC's full ticker->CIK mapping once and caches it in memory."""
    data = _get(SEC_TICKER_MAP_URL).json()
    # Raw shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


@functools.lru_cache(maxsize=1)
def get_company_index() -> list[dict]:
    """
    Same source file as get_ticker_to_cik_map(), but keeps the company TITLE
    too (the other function throws it away). Used by ticker_resolver.py to
    fuzzy-match a free-text company name against real SEC-registered names,
    not just exact tickers.
    """
    data = _get(SEC_TICKER_MAP_URL).json()
    return [
        {"ticker": v["ticker"].upper(), "cik": str(v["cik_str"]).zfill(10), "title": v["title"]}
        for v in data.values()
    ]


def cik_for_ticker(ticker: str) -> str:
    mapping = get_ticker_to_cik_map()
    cik = mapping.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker '{ticker}' not found in SEC's ticker map.")
    return cik


def get_submissions(cik: str) -> dict:
    """Full filing history + company metadata for a given 10-digit zero-padded CIK."""
    return _get(SEC_SUBMISSIONS_URL.format(cik=cik)).json()


def get_recent_filings_by_form(cik: str, form_type: str, limit: int = 2) -> list[dict]:
    """
    Returns the `limit` most recent filings of a given form type (e.g. "10-K", "10-Q"),
    each as {accessionNumber, filingDate, reportDate, primaryDocument}.
    """
    subs = get_submissions(cik)
    recent = subs["filings"]["recent"]
    n = len(recent["form"])
    hits = []
    for i in range(n):
        if recent["form"][i] == form_type:
            hits.append({
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": recent["reportDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
        if len(hits) >= limit:
            break
    return hits


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Builds the direct URL to a filing's primary HTML document."""
    acc_nodash = accession_number.replace("-", "")
    cik_nozero = str(int(cik))  # archive paths use the CIK without leading zeros
    return f"{SEC_ARCHIVES_BASE}/{cik_nozero}/{acc_nodash}/{primary_document}"


def fetch_filing_html(cik: str, accession_number: str, primary_document: str) -> str:
    url = filing_document_url(cik, accession_number, primary_document)
    return _get(url).text


def full_text_search(query: str, forms: str = "10-K", ciks: str | None = None) -> dict:
    """Keyword search across every filing's text since 2001 (efts.sec.gov)."""
    params = {"q": query, "forms": forms}
    if ciks:
        params["ciks"] = ciks
    return _get(SEC_FULLTEXT_SEARCH_URL, params=params).json()
