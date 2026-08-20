"""Layout-aware hierarchical chunking.

Retrieval quality depends heavily on chunk boundaries. Splitting on a fixed
character count severs sentences and mixes unrelated sections into one vector,
so this splits on the paper's own structure first (sections), then packs
paragraphs into token-budgeted chunks, and only falls back to hard splitting
when a single paragraph exceeds the budget on its own.

Each chunk carries its section path, so a retrieved passage can be cited as
"3 Methods > 3.2 Training" rather than an anonymous offset, and so retrieval
can filter by section (e.g. only Results when comparing metrics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbered ("3.2 Methods"), roman ("IV. RESULTS"), or well-known unnumbered
# headings. Kept deliberately conservative: a false heading fragments a
# section, which hurts retrieval more than a missed heading does.
_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:"
    r"(?P<number>\d{1,2}(?:\.\d{1,2}){0,2})\.?[ \t]+(?P<numbered>[A-Z][^\n]{2,80})"
    r"|(?P<roman>[IVXLC]{1,5})\.[ \t]+(?P<roman_title>[A-Z][^\n]{2,80})"
    r"|(?P<known>ABSTRACT|INTRODUCTION|RELATED WORK|BACKGROUND|METHODS|METHOD|"
    r"METHODOLOGY|APPROACH|EXPERIMENTS|EXPERIMENT|EVALUATION|RESULTS|DISCUSSION|"
    r"CONCLUSIONS|CONCLUSION|LIMITATIONS|FUTURE WORK|ACKNOWLEDGMENTS)"
    r")[ \t]*$",
    re.MULTILINE,
)

# Roughly 4 characters per token for English prose. Exact tokenization is
# model-specific; this only needs to be close enough to keep chunks inside an
# embedding model's context window.
CHARS_PER_TOKEN = 4

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
MIN_CHUNK_CHARS = 80


@dataclass
class Section:
    title: str
    level: int
    text: str
    start_offset: int


@dataclass
class Chunk:
    index: int
    content: str
    section_path: str | None
    token_estimate: int
    page_number: int | None = None
    kind: str = "text"
    metadata: dict[str, str] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _heading_level(match: re.Match[str]) -> int:
    number = match.group("number")
    if number:
        # "3" -> level 1, "3.2" -> level 2, "3.2.1" -> level 3
        return number.count(".") + 1
    return 1


def _heading_title(match: re.Match[str]) -> str:
    if match.group("numbered"):
        return f"{match.group('number')} {match.group('numbered')}".strip()
    if match.group("roman_title"):
        return f"{match.group('roman')}. {match.group('roman_title')}".strip()
    return (match.group("known") or "").title()


def split_sections(text: str) -> list[Section]:
    """Split a document into sections on detected headings.

    Text before the first heading becomes a synthetic "Front Matter" section so
    the title block and abstract are never dropped.
    """
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        stripped = text.strip()
        if not stripped:
            return []
        return [Section(title="Document", level=1, text=stripped, start_offset=0)]

    sections: list[Section] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(title="Front Matter", level=1, text=preamble, start_offset=0))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue
        sections.append(
            Section(
                title=_heading_title(match),
                level=_heading_level(match),
                text=body,
                start_offset=match.start(),
            )
        )

    return sections


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split an oversized paragraph, preferring sentence boundaries.

    Used only when a single paragraph exceeds the budget; ending on a sentence
    keeps chunks from being cut mid-clause.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the whole budget still has to be cut.
        while len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.append(sentence[:max_chars].strip())
            sentence = sentence[max(1, max_chars - overlap_chars) :]

        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            # Carry a tail of the previous piece so context spans the boundary.
            current = current[-overlap_chars:] if overlap_chars else ""

        current = f"{current} {sentence}".strip()

    if current.strip():
        pieces.append(current.strip())

    return [piece for piece in pieces if piece]


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    section_path: str | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Pack paragraphs into token-budgeted chunks within a single section."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    parts: list[str] = []
    buffer = ""

    for paragraph in _split_paragraphs(text):
        if len(paragraph) > max_chars:
            if buffer:
                parts.append(buffer)
                buffer = ""
            parts.extend(_hard_split(paragraph, max_chars, overlap_chars))
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) > max_chars:
            parts.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate

    if buffer:
        parts.append(buffer)

    cleaned = [part.strip() for part in parts if part.strip()]
    # Drop slivers, unless the section is genuinely one short passage — losing
    # a one-line section entirely would be worse than keeping a short chunk.
    keep_all = len(cleaned) <= 1
    return [
        Chunk(
            index=start_index + offset,
            content=part,
            section_path=section_path,
            token_estimate=estimate_tokens(part),
        )
        for offset, part in enumerate(cleaned)
        if keep_all or len(part) >= MIN_CHUNK_CHARS
    ]


def chunk_document(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a full document, preserving section structure in every chunk."""
    chunks: list[Chunk] = []
    path_stack: list[tuple[int, str]] = []

    for section in split_sections(text):
        # Maintain a breadcrumb so subsections carry their parent heading.
        while path_stack and path_stack[-1][0] >= section.level:
            path_stack.pop()
        path_stack.append((section.level, section.title))
        section_path = " > ".join(title for _, title in path_stack)

        chunks.extend(
            chunk_text(
                section.text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                section_path=section_path,
                start_index=len(chunks),
            )
        )

    # Re-index so indices are contiguous across section boundaries.
    for position, chunk in enumerate(chunks):
        chunk.index = position
    return chunks


def chunk_asset(
    *,
    kind: str,
    content: str,
    caption: str | None,
    page_number: int,
    index: int,
) -> Chunk:
    """Turn a table or figure into its own retrievable chunk.

    Assets are kept whole rather than merged into surrounding prose: a table
    split across chunks loses its header row, and a caption retrieved next to
    its table is far more useful than either alone.
    """
    body = f"{caption}\n\n{content}".strip() if caption else content.strip()
    section_path = caption.split(":")[0] if caption and ":" in caption else None
    return Chunk(
        index=index,
        content=body,
        section_path=section_path,
        token_estimate=estimate_tokens(body),
        page_number=page_number,
        kind=kind,
    )
