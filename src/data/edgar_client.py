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
    data = _get(SEC_TICKER_MAP_URL).json()
    # Raw shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


@functools.lru_cache(maxsize=1)
def get_company_index() -> list[dict]:
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
    return _get(SEC_SUBMISSIONS_URL.format(cik=cik)).json()


def get_recent_filings_by_form(cik: str, form_type: str, limit: int = 2) -> list[dict]:
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
