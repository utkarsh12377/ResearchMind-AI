"""Bibliography extraction: locating the references section and parsing entries.

Reference lists are one of the least standardized parts of a paper — numbering
styles, author formats, and line wrapping all vary by venue. Rather than trying
to fully parse each entry, we split reliably into individual references and
pull out the identifiers that actually matter downstream (DOI, arXiv ID), which
Milestone 26 uses to link papers in the knowledge graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The heading that starts the bibliography. Anchored to a line start so a
# passing mention of "references" in body text doesn't trigger it.
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:(?:[IVXLC]+|\d+)[.)]?\s*)?"
    r"(REFERENCES|References|REFERENCE|Bibliography|BIBLIOGRAPHY|Works Cited)\s*$",
    re.MULTILINE,
)

# Sections that legitimately follow the bibliography and must not be swallowed.
_POST_REFERENCES_HEADING = re.compile(
    r"^\s*(?:(?:[IVXLC]+|\d+|[A-Z])[.)]?\s*)?"
    r"(APPENDIX|Appendix|APPENDICES|Supplementary|SUPPLEMENTARY)\b",
    re.MULTILINE,
)

# "[1] ..." / "1. ..." / "(1) ..." entry markers.
_NUMBERED_ENTRY = re.compile(r"(?m)^\s*(?:\[(\d{1,3})\]|\((\d{1,3})\)|(\d{1,3})\.)\s+")

_DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(
    r"\barXiv[:\s]\s*((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?))",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")

MIN_ENTRY_LENGTH = 20
MAX_ENTRY_LENGTH = 2000
MAX_REFERENCES = 500


@dataclass
class ParsedReference:
    order: int
    raw_text: str
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    title: str | None = None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_references_section(full_text: str) -> str | None:
    """Return the bibliography text, or None if no references heading is found.

    The *last* heading match wins: papers often cite the word in a section
    title ("we compare references") before the real bibliography.
    """
    matches = list(_REFERENCES_HEADING.finditer(full_text))
    if not matches:
        return None

    section = full_text[matches[-1].end() :]

    # Stop at an appendix so its prose isn't parsed as references.
    tail = _POST_REFERENCES_HEADING.search(section)
    if tail:
        section = section[: tail.start()]

    return section.strip() or None


def _trim_doi(doi: str) -> str:
    """Strip trailing punctuation that sentence-ends a DOI in running text."""
    return doi.rstrip(".,;)]}>")


def _split_numbered(section: str) -> list[str]:
    markers = list(_NUMBERED_ENTRY.finditer(section))
    if len(markers) < 2:
        return []

    entries = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(section)
        entries.append(section[marker.end() : end])
    return entries


def _split_unnumbered(section: str) -> list[str]:
    """Fall back to blank-line separated blocks for author-year bibliographies."""
    return [block for block in re.split(r"\n\s*\n", section) if block.strip()]


def parse_references(full_text: str) -> list[ParsedReference]:
    """Extract individual references with their identifiers."""
    section = find_references_section(full_text)
    if not section:
        return []

    raw_entries = _split_numbered(section) or _split_unnumbered(section)

    references: list[ParsedReference] = []
    for raw in raw_entries:
        text = _clean(raw)
        # Very short fragments are page furniture (headers, stray numbers);
        # very long ones mean the split failed and we'd store a whole page.
        if len(text) < MIN_ENTRY_LENGTH:
            continue
        text = text[:MAX_ENTRY_LENGTH]

        doi_match = _DOI_PATTERN.search(text)
        arxiv_match = _ARXIV_PATTERN.search(text)
        year_match = _YEAR_PATTERN.search(text)

        references.append(
            ParsedReference(
                order=len(references) + 1,
                raw_text=text,
                doi=_trim_doi(doi_match.group(1)) if doi_match else None,
                arxiv_id=arxiv_match.group(1) if arxiv_match else None,
                year=int(year_match.group(1)) if year_match else None,
                title=_guess_title(text),
            )
        )

        if len(references) >= MAX_REFERENCES:
            break

    return references


def _guess_title(entry: str) -> str | None:
    """Best-effort title: the first substantial segment that isn't authors or venue.

    Reference formats vary too much for this to be exact, so it is treated as a
    hint for display and fuzzy matching, never as an authoritative title.
    """
def _guess_title(entry: str) -> str | None:
    """Best-effort title: the first substantial segment that isn't authors or venue.

    Reference formats vary too much for this to be exact, so it is treated as a
    hint for display and fuzzy matching, never as an authoritative title.
    """
    # Identifiers and trailing venue noise would otherwise win the "first
    # substantial segment" contest, so strip them before splitting.
    text = _DOI_PATTERN.sub("", entry)
    text = _ARXIV_PATTERN.sub("", text)
    text = re.sub(r"\b(?:doi|arXiv|URL|https?)\S*", "", text, flags=re.IGNORECASE)

    for segment in re.split(r"(?<=[.?])\s+", text):
        candidate = _clean(segment).strip(" .,;:")
        if len(candidate) < 15:
            continue
        if _looks_like_authors(candidate) or _looks_like_venue(candidate):
            continue
        return candidate[:500]
    return None


def _looks_like_authors(segment: str) -> bool:
    """Author lists are dense in initials: "A. Vaswani, N. Shazeer"."""
    if len(re.findall(r"\b[A-Z]\.", segment)) >= 2:
        return True
    return bool(re.search(r"\bet al\.?", segment, re.IGNORECASE)) and len(segment) < 60


def _looks_like_venue(segment: str) -> bool:
    """Publication venues rather than titles."""
    return bool(
        re.match(
            r"^(In|Proceedings|Proc\.|Journal|Trans\.|Advances|IEEE|ACM|Vol\.?|pp\.?)\b",
            segment,
        )
    )
