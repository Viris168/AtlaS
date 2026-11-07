"""agent/rag/chunking.py — Turn loaded documents into chunks (Milestone 5B).

Ported from the earlier "Structure-based semantic chunking" project. Markdown (and HTML,
which the loader converts to Markdown headings) follows the document's structure:
split on ##, re-split long sections on ###, split at paragraphs to fit the token limit,
then merge chunks too small to rank well. Plain text and PDF pages have no headings,
so their paragraphs are packed up to the limit with a small overlap.

Pure Python: token counting is passed in, so tests never load a model.
"""
import math
import re
from dataclasses import dataclass, replace
from typing import Callable

TokenCounter = Callable[[str], int]

HEADING_PATH_SEPARATOR = " > "
SUBSECTION_JOINER = " / "
OVERLAP_FRACTION = 0.15  # plain chunks repeat about this much of the previous chunk

_FENCE = re.compile(r"^\s*(```|~~~)")
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_IMAGE = re.compile(r"!\[[^\]]*\](\([^)]*\)|\[[^\]]*\])")  # ![alt](url), ![alt][ref]
_LINK = re.compile(r"\[([^\]]*)\](\([^)]*\)|\[[^\]]*\])")  # [text](url) -> text


def approx_tokens(text: str) -> int:
    """About 4 characters per token, plus the model's start and end tokens. Used in tests."""
    return math.ceil(len(text) / 4) + 2


@dataclass(frozen=True)
class Document:
    """One loaded unit: a whole Markdown/HTML/TXT file, or one PDF page."""
    source: str  # path relative to knowledge/, e.g. "guides/setup.md"
    title: str
    text: str
    markdown: bool  # True: chunk by headings. False: pack paragraphs.
    page: int | None = None  # PDFs only, starting at 1


@dataclass(frozen=True)
class Chunk:
    id: str  # "{source}#{n}", n counts across the whole file
    source: str
    title: str
    section: str  # the ## heading ("" for text before the first one, or for plain text)
    subsection: str  # the ### heading(s), "" if the section wasn't split on ###
    text: str  # raw text as it appears in the source, without the heading path
    token_count: int  # of embed_text(), so it can be checked against the model's limit
    page: int | None = None

    @property
    def heading_path(self) -> str:
        return build_heading_path(self.title, self.section, self.subsection)

    def embed_text(self) -> str:
        """What the embedding model and BM25 see: the heading path, a blank line, the text."""
        return f"{self.heading_path}\n\n{self.text}"


@dataclass(frozen=True)
class ChunkSettings:
    max_tokens: int = 512  # whole embed text, heading path included
    min_tokens: int = 40
    h3_split_words: int = 400
    count_tokens: TokenCounter = approx_tokens

    @classmethod
    def from_config(cls, config, count_tokens: TokenCounter) -> "ChunkSettings":
        return cls(config.rag_max_chunk_tokens, config.rag_min_chunk_tokens, config.rag_h3_split_words, count_tokens)


# --- Markdown syntax -------------------------------------------------------------

def is_fence_line(line: str) -> bool:
    return bool(_FENCE.match(line))


def heading_pattern(level: int) -> re.Pattern[str]:
    """Matches headings of exactly this level: ## must not match ###."""
    return re.compile(rf"^#{{{level}}}\s+(.+?)\s*$")


def clean_heading(text: str) -> str:
    """Drop images and unwrap links, so badges and URLs don't end up in heading paths."""
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    return " ".join(text.split())


def has_content(text: str) -> bool:
    """True if there is anything besides headings and blank lines."""
    return any(line.strip() and not _ANY_HEADING.match(line) for line in text.splitlines())


def find_title(text: str, fallback: str) -> str:
    """The first # heading outside a code block, or the fallback."""
    pattern, in_fence = heading_pattern(1), False
    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        elif not in_fence and (match := pattern.match(line)):
            return clean_heading(match.group(1)) or fallback
    return fallback


def build_heading_path(*parts: str) -> str:
    """'Doc > Section > Subsection', skipping empty and repeated parts."""
    kept: list[str] = []
    for part in parts:
        if part and part not in kept:
            kept.append(part)
    return HEADING_PATH_SEPARATOR.join(kept)


def split_on_heading(text: str, level: int) -> list[tuple[str, str]]:
    """Split at every heading of exactly `level`, ignoring code blocks.

    Returns (heading, text) pairs in order; text before the first heading has heading "".
    """
    pattern = heading_pattern(level)
    sections: list[tuple[str, str]] = []
    heading, lines, in_fence = "", [], False
    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        match = None if in_fence else pattern.match(line)
        if match:
            sections.append((heading, "\n".join(lines).strip("\n")))
            heading, lines = clean_heading(match.group(1)), [line]
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines).strip("\n")))
    return [(h, t) for h, t in sections if t.strip()]


# --- Fitting text to the token limit ---------------------------------------------

def split_blocks(text: str) -> list[str]:
    """Split at blank lines, keeping fenced code blocks whole. Tables have no blank lines, so they stay whole too."""
    blocks, current, in_fence = [], [], False
    for line in text.splitlines():
        if is_fence_line(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(units: list[str], max_tokens: int, joiner: str, count: TokenCounter) -> list[str]:
    """Greedily group consecutive units so each group fits. A unit bigger than the limit stands alone."""
    groups, current, size = [], [], 0
    for unit in units:
        unit_size = count(unit)
        if current and size + unit_size > max_tokens:
            groups.append(joiner.join(current))
            current, size = [], 0
        current.append(unit)
        size += unit_size
    if current:
        groups.append(joiner.join(current))
    return groups


def _split_by_words(line: str, max_tokens: int, count: TokenCounter) -> list[str]:
    return _pack(line.split(" "), max_tokens, " ", count)


def _split_by_lines(block: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split one oversized block at lines (then words). A code block keeps its fences on every piece."""
    lines = block.splitlines()
    fenced = len(lines) >= 2 and is_fence_line(lines[0]) and is_fence_line(lines[-1])
    if fenced:
        opening, closing = lines[0], lines[-1]
        budget = max_tokens - count(f"{opening}\n{closing}")
        return [f"{opening}\n{body}\n{closing}" for body in _pack(lines[1:-1], budget, "\n", count)]
    units: list[str] = []
    for line in lines:
        units.extend(_split_by_words(line, max_tokens, count) if count(line) > max_tokens else [line])
    return _pack(units, max_tokens, "\n", count)


def _units(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Blocks small enough to pack, splitting any that are too big on their own."""
    units: list[str] = []
    for block in split_blocks(text):
        units.extend(_split_by_lines(block, max_tokens, count) if count(block) > max_tokens else [block])
    return units


def pack_paragraphs(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    if count(text) <= max_tokens:
        return [text]
    return _pack(_units(text, max_tokens, count), max_tokens, "\n\n", count)


def _fine_units(text: str, piece_tokens: int, count: TokenCounter) -> list[tuple[str, str]]:
    """(separator, text) units small enough to overlap with: whole paragraphs when they fit,
    otherwise single lines, and over-long lines cut into word groups. The separator
    ("\\n\\n", "\\n" or " ") is what originally stood before the unit, so joining restores the layout."""
    units: list[tuple[str, str]] = []
    for block in split_blocks(text):
        if count(block) <= piece_tokens:
            units.append(("\n\n", block))
            continue
        for n, line in enumerate(block.splitlines()):
            sep = "\n\n" if n == 0 else "\n"
            if count(line) <= piece_tokens:
                units.append((sep, line))
            else:
                for m, group in enumerate(_split_by_words(line, piece_tokens, count)):
                    units.append((sep if m == 0 else " ", group))
    return units


def _join(units: list[tuple[str, str]]) -> str:
    return "".join(sep + text for sep, text in units)[len(units[0][0]):] if units else ""


def pack_with_overlap(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Pack text up to max_tokens per group; each group starts by repeating the end of the one before (~15%)."""
    overlap_budget = max(1, int(max_tokens * OVERLAP_FRACTION))
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for unit in _fine_units(text, overlap_budget, count):
        if current and count(_join(current + [unit])) > max_tokens:
            groups.append(current)
            carried: list[tuple[str, str]] = []
            for previous in reversed(current):
                if count(_join([previous] + carried)) > overlap_budget:
                    break
                carried.insert(0, previous)
            current = carried if count(_join(carried + [unit])) <= max_tokens else []
        current.append(unit)
    if current:
        groups.append(current)
    return [_join(group) for group in groups]


# --- Chunking --------------------------------------------------------------------

def _join_subsections(first: str, second: str) -> str:
    parts: list[str] = []
    for label in (*first.split(SUBSECTION_JOINER), *second.split(SUBSECTION_JOINER)):
        if label and label not in parts:
            parts.append(label)
    return SUBSECTION_JOINER.join(parts)


def _try_combine(first: Chunk, second: Chunk, settings: ChunkSettings) -> Chunk | None:
    """Join two neighbours from the same ## section if the result still fits; otherwise None.

    Never across sections: the merged chunk would cite one section for another's text.
    """
    if (first.source, first.section, first.page) != (second.source, second.section, second.page):
        return None
    merged = replace(
        first,
        text=f"{first.text}\n\n{second.text}",
        subsection=_join_subsections(first.subsection, second.subsection),
    )
    size = settings.count_tokens(merged.embed_text())
    if size > settings.max_tokens:
        return None
    return replace(merged, token_count=size)


def merge_small_chunks(chunks: list[Chunk], settings: ChunkSettings) -> list[Chunk]:
    """Fold each chunk under min_tokens into the chunk before it, or else the one after it.

    A small chunk with no same-section neighbour (say, a one-line ## License) is kept as it is.
    """
    merged: list[Chunk] = []
    waiting: Chunk | None = None  # a small chunk that couldn't join the one before it
    for chunk in chunks:
        if waiting is not None:
            combined = _try_combine(waiting, chunk, settings)
            if combined is not None:
                chunk = combined
            else:
                merged.append(waiting)
            waiting = None
        # Judge "small" on the chunk's own text: a long heading path doesn't make it more useful.
        if settings.count_tokens(chunk.text) < settings.min_tokens:
            combined = _try_combine(merged[-1], chunk, settings) if merged else None
            if combined is not None:
                merged[-1] = combined
            else:
                waiting = chunk
            continue
        merged.append(chunk)
    if waiting is not None:
        merged.append(waiting)
    return merged


def _markdown_pieces(doc: Document, settings: ChunkSettings) -> list[tuple[str, str, str]]:
    """(section, subsection, text) pieces, each fitting the limit together with its heading path."""
    count = settings.count_tokens
    pieces = []
    for section, section_text in split_on_heading(doc.text, 2):
        if len(section_text.split()) > settings.h3_split_words:
            parts = split_on_heading(section_text, 3)
        else:
            parts = [("", section_text)]
        for subsection, text in parts:
            heading_path = build_heading_path(doc.title, section, subsection)
            budget = settings.max_tokens - count(heading_path + "\n\n")
            for part in pack_paragraphs(text, budget, count):
                if has_content(part):
                    pieces.append((section, subsection, part))
    return pieces


def _plain_pieces(doc: Document, settings: ChunkSettings) -> list[tuple[str, str, str]]:
    budget = settings.max_tokens - settings.count_tokens(doc.title + "\n\n")
    return [("", "", part) for part in pack_with_overlap(doc.text, budget, settings.count_tokens) if part.strip()]


def chunk_documents(docs: list[Document], settings: ChunkSettings = ChunkSettings()) -> list[Chunk]:
    """Chunk every Document of ONE file (one, or one per PDF page). Ids run 0, 1, 2 ... across the file."""
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = _markdown_pieces(doc, settings) if doc.markdown else _plain_pieces(doc, settings)
        doc_chunks = []
        for section, subsection, text in pieces:
            chunk = Chunk("", doc.source, doc.title, section, subsection, text, 0, doc.page)
            doc_chunks.append(replace(chunk, token_count=settings.count_tokens(chunk.embed_text())))
        chunks.extend(merge_small_chunks(doc_chunks, settings) if doc.markdown else doc_chunks)
    return [replace(chunk, id=f"{chunk.source}#{n}") for n, chunk in enumerate(chunks)]
