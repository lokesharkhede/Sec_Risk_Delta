"""
Each function here is one LangGraph node. Nodes read/write the shared
RiskDeltaState dict. Keeping them as plain functions (rather than classes)
keeps the graph definition in graph.py easy to read.
"""
from src.agents.state import RiskDeltaState
from src.data import edgar_client, xbrl_client
from src.data.ticker_resolver import resolve_ticker as resolve_ticker_lookup
from src.parsing.filing_parser import extract_sections
from src.diffing.diff_engine import diff_sections, summarize_diff, split_paragraphs
from src.llm.llm_gemini import chat
from src.memory.vector_store import store_risk_paragraphs, find_similar_prior_risks


# ------------------------------------------------------- resolve_ticker ----
def resolve_ticker(state: RiskDeltaState) -> RiskDeltaState:
    """
    First node in the graph. Takes whatever the user typed -- a real ticker,
    a typo'd company name, or an informal name -- and resolves it to a real
    SEC ticker before anything else runs. See ticker_resolver.py for the
    local-fuzzy-match-first, LLM-only-if-needed strategy.
    """
    result = resolve_ticker_lookup(state["company_query"])
    if result["status"] == "ambiguous":
        candidate_list = ", ".join(
            f"{c['title']} ({c['ticker']})" for c in result["candidates"][:5]
        )
        raise ValueError(
            f"Couldn't confidently resolve '{state['company_query']}' to a ticker. "
            f"Closest matches: {candidate_list}"
        )
    return {
        "ticker": result["ticker"],
        "resolved_company_title": result["title"],
        "ticker_resolution_method": result["method"],
    }


# ---------------------------------------------------------------- fetch ----
def fetch_filings(state: RiskDeltaState) -> RiskDeltaState:
    cik = edgar_client.cik_for_ticker(state["ticker"])
    filings = edgar_client.get_recent_filings_by_form(cik, state["form_type"], limit=2)
    if len(filings) < 2:
        raise ValueError(f"Fewer than 2 {state['form_type']} filings found for {state['ticker']}")
    new_meta, old_meta = filings[0], filings[1]   # index 0 = most recent

    new_html = edgar_client.fetch_filing_html(cik, new_meta["accessionNumber"], new_meta["primaryDocument"])
    old_html = edgar_client.fetch_filing_html(cik, old_meta["accessionNumber"], old_meta["primaryDocument"])

    return {
        "cik": cik,
        "new_filing_meta": new_meta,
        "old_filing_meta": old_meta,
        "new_html": new_html,
        "old_html": old_html,
    }


# ---------------------------------------------------------------- parse ----
def parse_sections(state: RiskDeltaState) -> RiskDeltaState:
    return {
        "new_sections": extract_sections(state["new_html"]),
        "old_sections": extract_sections(state["old_html"]),
    }


def diff_all_sections(state: RiskDeltaState) -> RiskDeltaState:
    new_s, old_s = state["new_sections"], state["old_sections"]
    return {
        "risk_factors_diff": diff_sections(old_s.get("item_1a", ""), new_s.get("item_1a", "")),
        "legal_diff": diff_sections(old_s.get("item_3", ""), new_s.get("item_3", "")),
        "mdna_diff": diff_sections(old_s.get("item_7", ""), new_s.get("item_7", "")),
    }


# --------------------------------------------------------- orchestrator ----
def orchestrate(state: RiskDeltaState) -> RiskDeltaState:
    """
    Decides which specialist agents are worth invoking, based on where
    material change was actually detected. This is what makes the graph
    "agentic" rather than a fixed pipeline: a filing with an untouched
    litigation section skips the litigation agent entirely.
    """
    route_to = ["financial_agent"]  # financial deltas always checked (cheap, XBRL-based)

    if summarize_diff(state["legal_diff"])["material_change_signal"]:
        route_to.append("litigation_agent")

    if summarize_diff(state["risk_factors_diff"])["material_change_signal"]:
        route_to.append("sentiment_agent")

    return {"route_to": route_to}


def route_condition(state: RiskDeltaState) -> list[str]:
    """Used by the graph's conditional edge to fan out to the chosen agents."""
    return state["route_to"]


# ------------------------------------------------------- specialist agents
FINANCIAL_TAGS = ["Revenues", "NetIncomeLoss", "LongTermDebtNoncurrent", "OperatingIncomeLoss"]


def financial_agent(state: RiskDeltaState) -> RiskDeltaState:
    deltas = []
    for tag in FINANCIAL_TAGS:
        d = xbrl_client.quarter_over_quarter_delta(state["cik"], tag)
        if d:
            deltas.append(d)

    prompt = (
        "You are a financial-statement analyst. Given these XBRL period-over-period "
        f"deltas for {state['ticker']}, write a 3-4 sentence flag of anything a fund "
        f"analyst should know:\n\n{deltas}"
    )
    findings = chat(
        system_prompt="You are a precise, skeptical equity research analyst. No fluff.",
        user_prompt=prompt,
    )
    return {"financial_findings": findings}


def litigation_agent(state: RiskDeltaState) -> RiskDeltaState:
    diff = state["legal_diff"]
    prompt = (
        "Compare these Item 3 Legal Proceedings changes between two consecutive filings.\n\n"
        f"NEWLY ADDED paragraphs:\n{diff.added}\n\n"
        f"REMOVED paragraphs:\n{diff.removed}\n\n"
        f"REWORDED paragraphs (old -> new pairs):\n"
        f"{[(c.old_text[:200], c.new_text[:200]) for c in diff.reworded]}\n\n"
        "Summarize the litigation exposure delta in 3-4 sentences, flagging any new "
        "lawsuits, settlements, or escalation/de-escalation in tone."
    )
    findings = chat(
        system_prompt="You are a litigation-risk analyst reviewing SEC filings.",
        user_prompt=prompt,
    )
    return {"litigation_findings": findings}


def sentiment_agent(state: RiskDeltaState) -> RiskDeltaState:
    diff = state["risk_factors_diff"]
    ticker = state["ticker"]

    # For each genuinely new risk paragraph, check long-term memory for
    # semantically similar risks disclosed in earlier filings -- this is
    # what lets the agent say "this echoes something first seen in Q2 2024"
    # instead of only ever comparing against the single immediately-prior
    # filing that diff_sections() looked at.
    historical_context = []
    for paragraph in diff.added:
        hits = find_similar_prior_risks(ticker, paragraph, n_results=2)
        if hits:
            historical_context.append({"new_risk": paragraph[:150], "similar_prior": hits})

    prompt = (
        "Compare these Item 1A Risk Factors changes between two consecutive filings.\n\n"
        f"NEWLY ADDED risk paragraphs:\n{diff.added}\n\n"
        f"REMOVED risk paragraphs:\n{diff.removed}\n\n"
        f"HISTORICAL MEMORY -- similar risk language found in earlier filings "
        f"(empty if this is genuinely new or memory has no prior filings yet):\n"
        f"{historical_context}\n\n"
        "Identify: (1) any genuinely new risk category introduced (e.g. going concern, "
        "supply chain, new regulation) -- use the historical memory above to tell apart "
        "a TRULY new risk from a REWORDED restatement of something disclosed before, "
        "(2) whether existing risk language got more severe or was softened, "
        "(3) an overall risk-sentiment delta score from -5 (much riskier) to "
        "+5 (much safer). Be concise."
    )
    findings = chat(
        system_prompt="You are a risk-sentiment analyst specializing in 10-K risk factor language.",
        user_prompt=prompt,
    )
    return {"sentiment_findings": findings}

# --------------------------------------------------------------- report ----
def reconcile(state: RiskDeltaState) -> RiskDeltaState:
    report = {
        "ticker": state["ticker"],
        "resolved_company_title": state.get("resolved_company_title"),
        "ticker_resolution_method": state.get("ticker_resolution_method"),
        "form_type": state["form_type"],
        "compared_filings": {
            "new": state["new_filing_meta"],
            "old": state["old_filing_meta"],
        },
        "section_summaries": {
            "risk_factors": summarize_diff(state["risk_factors_diff"]),
            "legal_proceedings": summarize_diff(state["legal_diff"]),
            "mdna": summarize_diff(state["mdna_diff"]),
        },
        "agents_invoked": state["route_to"],
        "financial_findings": state.get("financial_findings"),
        "financial_provider": state.get("financial_provider"),
        "litigation_findings": state.get("litigation_findings"),
        "litigation_provider": state.get("litigation_provider"),
        "sentiment_findings": state.get("sentiment_findings")
    }

    # Grow long-term memory: persist this run's full risk-factor paragraph
    # set (not just the diff) so future runs -- for this ticker, any quarter
    # from now on -- can semantically search against everything ever seen,
    # not just the single prior filing that diff_sections() compared against.
    new_risk_text = state["new_sections"].get("item_1a", "")
    store_risk_paragraphs(
        ticker=state["ticker"],
        filing_date=state["new_filing_meta"]["filingDate"],
        accession=state["new_filing_meta"]["accessionNumber"],
        paragraphs=split_paragraphs(new_risk_text),
    )

    return {"final_report": report}
