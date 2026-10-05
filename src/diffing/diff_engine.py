import difflib
from dataclasses import dataclass, field
from rapidfuzz import fuzz


def split_paragraphs(text: str) -> list[str]:
    """Public: also used by nodes.py::reconcile to prep paragraphs for vector_store."""
    return [p.strip() for p in text.split("\n") if len(p.strip()) > 40]


# kept as an alias so existing internal call sites in this file don't need touching
_split_paragraphs = split_paragraphs


@dataclass
class ParagraphChange:
    kind: str            # "added" | "removed" | "reworded" | "unchanged"
    new_text: str = ""
    old_text: str = ""
    similarity: float = 0.0


@dataclass
class SectionDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    reworded: list[ParagraphChange] = field(default_factory=list)
    unchanged_count: int = 0


def diff_sections(old_text: str, new_text: str, reword_threshold: float = 60.0) -> SectionDiff:
    old_paras = _split_paragraphs(old_text)
    new_paras = _split_paragraphs(new_text)

    matcher = difflib.SequenceMatcher(a=old_paras, b=new_paras, autojunk=False)
    result = SectionDiff()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.unchanged_count += (i2 - i1)
        elif tag == "insert":
            result.added.extend(new_paras[j1:j2])
        elif tag == "delete":
            result.removed.extend(old_paras[i1:i2])
        elif tag == "replace":
            old_block = old_paras[i1:i2]
            new_block = new_paras[j1:j2]
            # Pair each new paragraph with its closest old paragraph by
            # fuzzy similarity to distinguish "reworded" from "genuinely new".
            for new_p in new_block:
                best_score, best_old = 0.0, None
                for old_p in old_block:
                    score = fuzz.token_sort_ratio(old_p, new_p)
                    if score > best_score:
                        best_score, best_old = score, old_p
                if best_old is not None and best_score >= reword_threshold:
                    result.reworded.append(ParagraphChange("reworded", new_p, best_old, best_score))
                else:
                    result.added.append(new_p)
            matched_old = {c.old_text for c in result.reworded}
            result.removed.extend([p for p in old_block if p not in matched_old])

    return result


def summarize_diff(diff: SectionDiff) -> dict:
    """Compact numeric summary used for quick anomaly flags and the report header."""
    return {
        "paragraphs_added": len(diff.added),
        "paragraphs_removed": len(diff.removed),
        "paragraphs_reworded": len(diff.reworded),
        "paragraphs_unchanged": diff.unchanged_count,
        "material_change_signal": len(diff.added) + len(diff.removed) > 3 or any(
            c.similarity < 80 for c in diff.reworded
        ),
    }
