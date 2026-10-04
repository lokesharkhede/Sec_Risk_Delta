"""
Resolves free-text company input ("Aple", "microsoft", "the iPhone company",
or a plain ticker like "AAPL") into a verified SEC ticker + CIK.

Two-tier strategy, cheapest and safest first:

  1. LOCAL FUZZY MATCH (rapidfuzz) against SEC's own ~13,000-company
     ticker/title list. Free, instant, no LLM call. Handles typos, partial
     names, and tickers typed directly.

  2. LLM-ASSISTED NORMALIZATION (only if #1 is weak). Gemini is asked to
     guess the official company name for informal input a string-matcher
     can't handle ("Google's parent" -> "Alphabet"). Its guess is NEVER
     trusted directly as a ticker -- LLMs can hallucinate ticker symbols --
     it's re-run through the exact same local fuzzy match against the real
     SEC list. If that re-check is still weak, we return candidates for the
     user to pick from instead of silently guessing wrong.
"""
import json
import re
from rapidfuzz import process, fuzz

from src.data.edgar_client import get_company_index
from src.llm.llm_gemini import chat

LOCAL_MATCH_THRESHOLD = 85   # confident enough to skip the LLM entirely
LLM_MATCH_THRESHOLD = 70     # confident enough to trust after LLM normalization

# Corporate suffixes ("Inc.", "Corp", "Holdings", ...) drown out short typos
# when fuzzy-matching whole titles (e.g. "Aple" vs "Apple Inc." scores much
# lower than "Aple" vs "Apple" would). Strip them before scoring so the
# comparison is against the actual company name, not boilerplate.
_SUFFIX_PATTERN = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|holdings|group|the)\b\.?",
    re.IGNORECASE,
)


def _normalize(name: str) -> str:
    cleaned = _SUFFIX_PATTERN.sub("", name)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


_index_cache = None       # list[dict], built once
_norm_titles_cache = None # list[str], same order as _index_cache


def _get_index():
    global _index_cache, _norm_titles_cache
    if _index_cache is None:
        _index_cache = get_company_index()
        _norm_titles_cache = [_normalize(c["title"]) for c in _index_cache]
    return _index_cache, _norm_titles_cache


def _best_local_matches(query: str, n: int = 5) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    index, norm_titles = _get_index()
    norm_query = _normalize(query)

    # Score against normalized titles (handles typo'd/partial company names)
    # AND raw tickers (handles someone just typing "AAPL" directly), then
    # keep the best score per company across both.
    title_hits = process.extract(norm_query, norm_titles, scorer=fuzz.ratio, limit=n)
    tickers = [c["ticker"] for c in index]
    ticker_hits = process.extract(query.upper(), tickers, scorer=fuzz.ratio, limit=n)

    best_by_idx = {}
    #for _choice, score, idx in title_hits:
    #    best_by_idx[idx] = max(best_by_idx.get(idx, 0), score)
    #for _choice, score, idx in ticker_hits:
    #    best_by_idx[idx] = max(best_by_idx.get(idx, 0), score)

    for _choice, score, idx in title_hits:
        old_score = best_by_idx.get(idx, 0)
        if score > old_score:
            best_by_idx[idx] = score 

    for _choice, score, idx in ticker_hits:
        old_score = best_by_idx.get(idx, 0)
        if score > old_score:
            best_by_idx[idx] = score        

    ranked = sorted(best_by_idx.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{**index[idx], "score": score} for idx, score in ranked]


def _llm_normalize(query: str) -> dict:
    """Asks the LLM to guess the official company name for a casual query."""
    prompt = (
        f'A user typed this into a company search box: "{query}". '
        "It might be a nickname, typo, product name, or partial name. "
        'Respond with ONLY a JSON object, no other text, exactly like: '
        '{"guess_company_name": "...", "guess_ticker": "..."} '
        "for the most likely publicly-traded company they mean. "
        "Give your best guess even if unsure."
    )
    raw = chat(system_prompt="You output only valid JSON, nothing else.", user_prompt=prompt)
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        return {"guess_company_name": query, "guess_ticker": ""}


def resolve_ticker(query: str) -> dict:
    """
    Returns either:
      {"status": "resolved", "ticker", "cik", "title", "method", "score"}
      {"status": "ambiguous", "candidates": [{"ticker","title","cik","score"}, ...]}
    "method" is "local_fuzzy" or "llm_assisted", so callers can show the
    user (and the report) how the ticker was actually determined.
    """
    local_matches = _best_local_matches(query)
    if local_matches and local_matches[0]["score"] >= LOCAL_MATCH_THRESHOLD:
        best = local_matches[0]
        return {"status": "resolved", "ticker": best["ticker"], "cik": best["cik"],
                "title": best["title"], "method": "local_fuzzy", "score": best["score"]}

    llm_guess = _llm_normalize(query)
    combined = f"{llm_guess.get('guess_company_name', '')} {llm_guess.get('guess_ticker', '')}".strip()
    llm_matches = _best_local_matches(combined or query)

    if llm_matches and llm_matches[0]["score"] >= LLM_MATCH_THRESHOLD:
        best = llm_matches[0]
        return {"status": "resolved", "ticker": best["ticker"], "cik": best["cik"],
                "title": best["title"], "method": "llm_assisted", "score": best["score"]}

    # Neither step was confident -- don't guess, surface candidates instead.
    candidates = llm_matches if llm_matches else local_matches
    return {"status": "ambiguous", "candidates": candidates}
