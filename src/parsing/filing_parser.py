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

def partition_filing(html: str) -> list[dict]:
    elements = partition_html(text=html)
    out = []
    for el in elements:
        el_type = "Table" if isinstance(el, Table) else type(el).__name__
        out.append({"type": el_type, "text": str(el).strip()})
    return out


def extract_sections_structured(html: str) -> dict[str, list[dict]]:
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

def extract_sections(html: str) -> dict[str, str]:

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
