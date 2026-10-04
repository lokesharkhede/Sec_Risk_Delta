"""Formats the graph's final_report dict into a readable Markdown report."""


def to_markdown(report: dict) -> str:
    new_m, old_m = report["compared_filings"]["new"], report["compared_filings"]["old"]
    s = report["section_summaries"]

    lines = [
        f"# Risk-Delta Report: {report['ticker']} ({report['form_type']})",
        "",
        f"**Comparing:** filing dated {new_m['filingDate']} (period end {new_m['reportDate']}) "
        f"vs. filing dated {old_m['filingDate']} (period end {old_m['reportDate']})",
        "",
        f"**Specialist agents invoked:** {', '.join(report['agents_invoked'])}",
        "",
        "## Section Change Summary",
        "",
        "| Section | Added | Removed | Reworded | Material Change? |",
        "|---|---|---|---|---|",
    ]
    for name, key in [("Risk Factors (Item 1A)", "risk_factors"),
                       ("Legal Proceedings (Item 3)", "legal_proceedings"),
                       ("MD&A (Item 7)", "mdna")]:
        d = s[key]
        flag = "🚩 Yes" if d["material_change_signal"] else "No"
        lines.append(
            f"| {name} | {d['paragraphs_added']} | {d['paragraphs_removed']} | "
            f"{d['paragraphs_reworded']} | {flag} |"
        )

    if report.get("financial_findings"):
        lines += ["", "## Financial-Statement Agent", report["financial_findings"]]
    if report.get("litigation_findings"):
        lines += ["", "## Litigation Agent", report["litigation_findings"]]
    if report.get("sentiment_findings"):
        lines += ["", "## Risk-Sentiment Agent", report["sentiment_findings"]]

    return "\n".join(lines)
