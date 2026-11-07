"""agent/tools/knowledge.py — search_knowledge, list_sources (Milestone 5B).

Agentic RAG: searching the user's documents is just another tool. The model decides when
to search, can search again with other words when the passages miss, and cites what it used.
Passages are wrapped in <passage> tags so the model can tell document text from instructions.

The retriever is created on the first call, so Atlas starts fast and the tests that never
search never open Chroma or load a model.
"""
from typing import Callable

from agent.rag.retriever import Hit, Retriever
from agent.tools.registry import Tool

MAX_K = 20
EMPTY_INDEX = (
    "The knowledge base is empty. Tell the user to put their documents (Markdown, text, HTML or PDF) "
    "in the knowledge/ folder and run /ingest."
)


def cite(hit: Hit) -> str:
    return f"{hit.source}, p. {hit.page}" if hit.page else hit.source


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]  # don't stop mid-word
    return cut + " …[passage cut]"


def _quote(text: str) -> str:
    # A document mustn't be able to close its own <passage> tag and talk outside it.
    return text.replace("</passage>", "<\\/passage>")


def render_hits(query: str, hits: list[Hit], passage_chars: int) -> str:
    lines = [f'{len(hits)} passages for "{query}". Cite them as [file, page], e.g. [{cite(hits[0])}].', ""]
    for n, hit in enumerate(hits, start=1):
        lines.append(f"[{n}] {cite(hit)} · {hit.heading_path}")
        lines.append("<passage>")
        lines.append(_quote(_trim(hit.text, passage_chars)))
        lines.append("</passage>")
        lines.append("")
    return "\n".join(lines).rstrip()


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_outline(outline: list[dict], max_chars: int = 2000, max_sections: int = 12) -> str:
    """One line per file: 'path: Title (Section, Section, …)'. Capped, so a big knowledge base
    can't bloat the system prompt; the rest is one list_sources call away."""
    lines: list[str] = []
    used = 0
    for n, entry in enumerate(outline):
        sections = [_short(s) for s in entry["sections"][:max_sections]]
        if len(entry["sections"]) > max_sections:
            sections.append(f"+{len(entry['sections']) - max_sections} more")
        line = f"- {entry['source']}: {_short(entry['title'])}" + (f" ({', '.join(sections)})" if sections else "")
        if used + len(line) > max_chars:
            lines.append(f"- … and {len(outline) - n} more files (call list_sources to see them all)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def knowledge_tools(get_retriever: Callable[[], Retriever], top_k: int = 8, passage_chars: int = 1000) -> list[Tool]:
    """search_knowledge and list_sources. Neither is risky: they only read the index."""
    opened: list[Retriever] = []

    def retriever() -> Retriever:
        if not opened:
            opened.append(get_retriever())
        return opened[0]

    def search_knowledge(query: str, k: int | None = None) -> str:
        query = str(query).strip()
        if not query:
            raise ValueError("'query' is empty; say what to look for")
        r = retriever()
        if r.store.count() == 0:
            return EMPTY_INDEX
        k = max(1, min(top_k if k is None else int(k), MAX_K))
        hits = r.search(query, k)
        if not hits:
            return f'No passages match "{query}". Try other words, or call list_sources to see what is indexed.'
        return render_hits(query, hits, passage_chars)

    def list_sources() -> str:
        sources = retriever().store.sources()
        if not sources:
            return EMPTY_INDEX
        total = sum(s["chunks"] for s in sources)
        lines = [f"{len(sources)} files indexed ({total} chunks):"]
        for s in sources:
            pages = f"{s['pages']} pages, " if s["pages"] else ""
            lines.append(f"- {s['source']} — {s['title']} ({pages}{s['chunks']} chunks)")
        return "\n".join(lines)

    return [
        Tool(
            name="search_knowledge",
            description=(
                "Search the user's own documents (the knowledge base built from the knowledge/ folder) "
                "and return the most relevant passages, numbered, each with its file, page and section. "
                "Use it for any question the user's documents might answer. Write the query in your own words; "
                "hybrid search matches both meaning and exact terms such as error codes or function names. "
                "If the passages don't answer the question, search again with different words. "
                f"Returns {top_k} passages by default (at most {MAX_K}), each cut to about {passage_chars} characters."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "k": {"type": "integer", "description": f"How many passages (1-{MAX_K}, default {top_k})."},
                },
                "required": ["query"],
            },
            fn=search_knowledge,
        ),
        Tool(
            name="list_sources",
            description="List the files in the knowledge base, with their titles and sizes, so you know what can be searched.",
            input_schema={"type": "object", "properties": {}, "required": []},
            fn=list_sources,
        ),
    ]
