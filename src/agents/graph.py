"""
Builds the LangGraph state machine:

  resolve_ticker -> fetch_filings -> parse_sections -> diff_all_sections -> orchestrate
        -> (conditional fan-out) -> {financial_agent, litigation_agent, sentiment_agent}
        -> reconcile -> END

resolve_ticker turns whatever the user typed (a real ticker, a typo'd
company name, or an informal name) into a verified SEC ticker before
anything else runs -- see ticker_resolver.py.

The conditional edge after `orchestrate` is what makes this "agentic": the
orchestrator decides at runtime which specialist agents actually run, based
on where it found material change.
"""
from langgraph.graph import StateGraph, END

from src.agents.state import RiskDeltaState
from src.agents import nodes


def build_graph():
    graph = StateGraph(RiskDeltaState)

    graph.add_node("resolve_ticker", nodes.resolve_ticker)
    graph.add_node("fetch_filings", nodes.fetch_filings)
    graph.add_node("parse_sections", nodes.parse_sections)
    graph.add_node("diff_all_sections", nodes.diff_all_sections)
    graph.add_node("orchestrate", nodes.orchestrate)
    graph.add_node("financial_agent", nodes.financial_agent)
    graph.add_node("litigation_agent", nodes.litigation_agent)
    graph.add_node("sentiment_agent", nodes.sentiment_agent)
    graph.add_node("reconcile", nodes.reconcile)

    graph.set_entry_point("resolve_ticker")
    graph.add_edge("resolve_ticker", "fetch_filings")
    graph.add_edge("fetch_filings", "parse_sections")
    graph.add_edge("parse_sections", "diff_all_sections")
    graph.add_edge("diff_all_sections", "orchestrate")

    # Conditional fan-out: only routes to agents orchestrate() selected.
    graph.add_conditional_edges(
        "orchestrate",
        nodes.route_condition,
        {
            "financial_agent": "financial_agent",
            "litigation_agent": "litigation_agent",
            "sentiment_agent": "sentiment_agent",
        },
    )

    for agent in ("financial_agent", "litigation_agent", "sentiment_agent"):
        graph.add_edge(agent, "reconcile")

    graph.add_edge("reconcile", END)

    return graph.compile()


def run_pipeline(company_query: str, form_type: str = "10-K") -> dict:
    app = build_graph()
    result = app.invoke({"company_query": company_query, "form_type": form_type})
    return result["final_report"]
