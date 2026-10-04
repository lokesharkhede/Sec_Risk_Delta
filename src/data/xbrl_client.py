"""
SEC XBRL API -- structured, machine-readable financial statement numbers
(revenue, net income, assets, etc.) tagged per the us-gaap taxonomy.
Free, no key. https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
"""
import time
import requests
import pandas as pd

from config import SEC_USER_AGENT, SEC_COMPANYFACTS_URL, SEC_COMPANYCONCEPT_URL, SEC_RATE_LIMIT_SLEEP_SEC

_session = requests.Session()
_session.headers.update({"User-Agent": SEC_USER_AGENT})


def get_company_facts(cik: str) -> dict:
    resp = _session.get(SEC_COMPANYFACTS_URL.format(cik=cik), timeout=20)
    resp.raise_for_status()
    time.sleep(SEC_RATE_LIMIT_SLEEP_SEC)
    return resp.json()


def get_concept_timeseries(cik: str, tag: str, unit: str = "USD") -> pd.DataFrame:
    """
    Pull one us-gaap concept (e.g. "Revenues", "NetIncomeLoss",
    "LongTermDebtNoncurrent") as a tidy DataFrame across all reported periods.
    """
    resp = _session.get(SEC_COMPANYCONCEPT_URL.format(cik=cik, tag=tag), timeout=20)
    resp.raise_for_status()
    time.sleep(SEC_RATE_LIMIT_SLEEP_SEC)
    data = resp.json()
    rows = data.get("units", {}).get(unit, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[["end", "val", "form", "fy", "fp", "filed", "accn"]]
    return df.sort_values("end")


def quarter_over_quarter_delta(cik: str, tag: str) -> dict | None:
    """
    Convenience helper for the financial-statement agent: returns the most
    recent value, the prior comparable value, and the % change for a concept.
    """
    df = get_concept_timeseries(cik, tag)
    if df.empty or len(df) < 2:
        return None
    df = df.drop_duplicates(subset="end", keep="last").sort_values("end")
    latest, prior = df.iloc[-1], df.iloc[-2]
    pct_change = None
    if prior["val"]:
        pct_change = round((latest["val"] - prior["val"]) / abs(prior["val"]) * 100, 2)
    return {
        "tag": tag,
        "latest_period_end": latest["end"],
        "latest_value": latest["val"],
        "prior_period_end": prior["end"],
        "prior_value": prior["val"],
        "pct_change": pct_change,
        "latest_accession": latest["accn"],
    }
