"""
Splits a raw 10-K/10-Q HTML document into its named Item sections
(Item 1A Risk Factors, Item 3 Legal Proceedings, Item 7 MD&A, ...).

PRIMARY PARSER: unstructured.io (`unstructured.partition.html.partition_html`)
  - Classifies each block of the document (NarrativeText, Title, Table,
    ListItem, ...) instead of returning one flat string. This is what lets
    us keep tables as distinct, tagged elements rather than having table
    cells get silently flattened into surrounding paragraph text -- the
    "handling embedded tables" requirement from the project brief.
  - Free, pip-installable, no external service/API call, no vision model.

FALLBACK PARSER: BeautifulSoup + regex (`_extract_sections_bs4_fallback`)
  - Used automatically if `unstructured` isn't installed or throws on a
    malformed document, so the pipeline never hard-fails on parsing alone.
  - Cruder: flattens everything to plain text, so a table's cells just run
    together as text in reading order -- no table-vs-prose distinction.

Docling is the other option named in the brief. It's not wired in here
because it pulls in a heavier layout-model stack (better for scanned/complex
PDFs); for HTML filings, unstructured.io is a lighter free choice that
still satisfies the "real document-structure parsing" requirement. Swapping
in Docling later means writing one function -- `_partition_with_docling(html)
-> list[Element]` -- with the same shape as `_partition_with_unstructured`.
"""
import re
from bs4 import BeautifulSoup

try:
    from unstructured.partition.html import partition_html
    from unstructured.documents.elements import Table
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# 10-K item headers, in the order they appear in Part I / Part II.
ITEM_PATTERN = re.compile(
    r"^item\s+(1a|1b|1c|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15)\.?\s*[-—:]?\s*",
    re.IGNORECASE,
)


# ============================================================
# PRIMARY: unstructured.io element-level partitioning
# ============================================================

def partition_filing(html: str) -> list[dict]:
    """
    Runs the raw HTML through unstructured.io and returns a flat list of
    {"type": "NarrativeText"|"Table"|"Title"|..., "text": "..."} dicts in
    reading order. This is the layer that gives us real document structure
    instead of one flattened string.
    """
    elements = partition_html(text=html)
    out = []
    for el in elements:
        el_type = "Table" if isinstance(el, Table) else type(el).__name__
        out.append({"type": el_type, "text": str(el).strip()})
    return out


def extract_sections_structured(html: str) -> dict[str, list[dict]]:
    """
    Groups unstructured.io elements by Item section. Returns
    {"item_1a": [{"type": "NarrativeText", "text": "..."}, {"type": "Table", ...}], ...}
    so callers that care (e.g. a future table-aware financial agent) can
    tell prose apart from tables within a section. Keeps the LAST occurrence
    of each item header, same logic as the fallback, to skip the Table of
    Contents.
    """
    elements = partition_filing(html)

    header_positions = {}  # item_num -> index into `elements`
    for i, el in enumerate(elements):
        m = ITEM_PATTERN.match(el["text"])
        if m and el["type"] in ("Title", "Text", "NarrativeText"):
            header_positions[m.group(1).lower()] = i

    if not header_positions:
        return {}

    ordered = sorted(header_positions.items(), key=lambda kv: kv[1])
    sections: dict[str, list[dict]] = {}
    for idx, (item_num, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(elements)
        sections[f"item_{item_num}"] = elements[start:end]
    return sections


def _elements_to_text(elements: list[dict]) -> str:
    """
    Flattens a section's elements back to plain text for modules (diff_engine,
    the LLM agents) that just want prose. Tables are kept but clearly tagged,
    rather than silently merged into the surrounding paragraphs, so a table
    that changed between filings still shows up as a visible diff instead of
    disappearing into unrelated text.
    """
    lines = []
    for el in elements:
        if el["type"] == "Table":
            lines.append(f"[TABLE] {el['text']}")
        else:
            lines.append(el["text"])
    return "\n".join(lines)


def _extract_sections_unstructured(html: str) -> dict[str, str]:
    structured = extract_sections_structured(html)
    return {name: _elements_to_text(els) for name, els in structured.items()}


# ============================================================
# FALLBACK: BeautifulSoup + regex (used only if unstructured fails)
# ============================================================

def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_sections_bs4_fallback(html: str) -> dict[str, str]:
    text = html_to_clean_text(html)
    matches = list(re.finditer(
        r"item\s+(1a|1b|1c|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15)\.?\s*[-—:]?\s*",
        text, re.IGNORECASE,
    ))
    if not matches:
        return {}

    last_index_by_item = {}
    for m in matches:
        last_index_by_item[m.group(1).lower()] = m.start()

    ordered = sorted(last_index_by_item.items(), key=lambda kv: kv[1])
    sections = {}
    for idx, (item_num, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(text)
        sections[f"item_{item_num}"] = text[start:end].strip()
    return sections


# ============================================================
# PUBLIC API -- unchanged signatures, so nothing downstream needs to change
# ============================================================

def extract_sections(html: str) -> dict[str, str]:
    """
    Returns {"item_1a": "...", "item_3": "...", "item_7": "...", ...}.
    Tries unstructured.io first (keeps tables distinguishable via a
    "[TABLE] " prefix); falls back to a plain-text BeautifulSoup split if
    unstructured isn't installed or raises on a malformed document.
    """
    if UNSTRUCTURED_AVAILABLE:
        try:
            sections = _extract_sections_unstructured(html)
            if sections:
                return sections
        except Exception:
            pass  # fall through to the BS4 fallback below
    return _extract_sections_bs4_fallback(html)


def get_risk_factors(html: str) -> str:
    return extract_sections(html).get("item_1a", "")


def get_legal_proceedings(html: str) -> str:
    return extract_sections(html).get("item_3", "")


def get_mdna(html: str) -> str:
    return extract_sections(html).get("item_7", "")
