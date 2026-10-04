import streamlit as st

from src.agents.graph import run_pipeline
from src.report.report_builder import to_markdown

st.set_page_config(page_title="SEC Filing Risk-Delta Agent", layout="wide")
st.title("📑 SEC Filing Intelligence & Risk-Delta Agent")
st.caption(
    "Diffs risk factors, litigation exposure, and financial deltas across "
    "consecutive SEC filings using a LangGraph multi-agent pipeline."
)

with st.sidebar:
    st.header("Query")
    company_query = st.text_input(
        "Company name or ticker",
        value="Apple",
        help="Typos and informal names are fine -- e.g. 'Aple', 'the iPhone company', or just 'AAPL'.",
    ).strip()
    form_type = st.selectbox("Form type", ["10-K", "10-Q"], index=0)
    run_btn = st.button("Run Risk-Delta Analysis", type="primary")

if run_btn and company_query:
    with st.spinner(f"Resolving '{company_query}' and fetching {form_type} filings..."):
        try:
            report = run_pipeline(company_query, form_type)

            resolved_title = report.get("resolved_company_title")
            method = report.get("ticker_resolution_method")
            method_label = "typed directly / exact match" if method == "local_fuzzy" else "LLM-assisted match"
            st.success(
                f"Resolved to **{resolved_title} ({report['ticker']})** via {method_label}. Analysis complete."
            )
            st.markdown(to_markdown(report))

            with st.expander("Raw report JSON"):
                st.json(report)

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
else:
    st.info("Enter a company name or ticker (e.g. Apple, Aple, AAPL) and click **Run Risk-Delta Analysis**.")
    st.markdown(
        "**How it works:** the app first resolves your input to a verified SEC ticker "
        "(local fuzzy match first, LLM only for informal names), then the orchestrator "
        "fetches the two most recent filings of the chosen type, diffs Item 1A (Risk "
        "Factors), Item 3 (Legal Proceedings), and Item 7 (MD&A), then dispatches "
        "specialist agents (financial / litigation / sentiment) only for sections where "
        "it detects material change."
    )
