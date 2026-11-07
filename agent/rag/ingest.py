"""agent/rag/ingest.py — Load, chunk, embed and index files from knowledge/ (Milestone 5B).

Step 2: loaders for Markdown, text, HTML and PDF, plus `--show` to read a file's chunks.
Step 3 adds the store and the incremental ingest.

    python -m agent.rag.ingest --show guides/setup.md
"""
import argparse
import re
import sys
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

from agent.config import Config, config as default_config
from agent.rag.chunking import Chunk, ChunkSettings, Document, TokenCounter, chunk_documents, find_title

MARKDOWN_SUFFIXES = {".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | HTML_SUFFIXES | {".txt", ".pdf"}


# --- Loaders ---------------------------------------------------------------------

class _HTMLToMarkdown(HTMLParser):
    """Keeps the visible text, turns h1-h3 into #-headings and block tags into blank lines,
    so HTML gets the same structure-based chunking as Markdown."""

    BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "br", "pre", "blockquote", "table", "ul", "ol", "h4", "h5", "h6"}
    SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
    HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### "}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP_TAGS:
            self._skip += 1
        elif tag in self.HEADINGS:
            self.parts.append("\n\n" + self.HEADINGS[tag])
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in self.HEADINGS or tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(" ".join(data.split()) if data.strip() else " ")

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_html(raw: str, source: str, fallback_title: str) -> Document:
    parser = _HTMLToMarkdown()
    parser.feed(raw)
    text = parser.markdown()
    title = " ".join(parser.title.split()) or find_title(text, fallback_title)
    return Document(source, title, text, markdown=True)


def _load_pdf(path: Path, source: str, fallback_title: str) -> list[Document]:
    from pypdf import PdfReader  # imported here so the agent loop never pays for it

    reader = PdfReader(path)
    title = ((reader.metadata.title if reader.metadata else None) or fallback_title).strip()
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # scanned pages have no text layer; there is nothing to index without OCR
            docs.append(Document(source, title, text, markdown=False, page=number))
    return docs


def load_file(path: Path, root: Path) -> list[Document]:
    """Load one file into Documents: one per file, or one per page for a PDF. Empty files give []."""
    source = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, source, path.stem)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        doc = _load_html(raw, source, path.stem)
    elif suffix in MARKDOWN_SUFFIXES:
        doc = Document(source, find_title(raw, path.stem), raw, markdown=True)
    elif suffix == ".txt":
        doc = Document(source, path.stem, raw, markdown=False)
    else:
        raise ValueError(f"unsupported file type '{suffix}'; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    return [doc] if doc.text.strip() else []


def find_files(root: Path) -> list[Path]:
    """Every supported file under root, sorted, skipping hidden files and folders."""
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


# --- Chunking a file -------------------------------------------------------------

@lru_cache(maxsize=2)
def model_token_counter(model_name: str) -> TokenCounter:
    """Count tokens exactly as the embedding model will, special tokens included. Loads the tokenizer once."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=True, verbose=False))


def chunk_file(path: Path, root: Path, settings: ChunkSettings) -> list[Chunk]:
    return chunk_documents(load_file(path, root), settings)


def show_chunks(chunks: list[Chunk], max_tokens: int) -> str:
    lines = []
    for chunk in chunks:
        page = f" · p. {chunk.page}" if chunk.page else ""
        flag = "  ⚠ over the limit, will be cut off" if chunk.token_count > max_tokens else ""
        lines.append(f"── {chunk.id}{page} · {chunk.heading_path} · {chunk.token_count} tokens{flag}")
        lines.append(chunk.text)
        lines.append("")
    lines.append(f"{len(chunks)} chunks")
    return "\n".join(lines)


def main(argv: list[str] | None = None, config: Config = default_config) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.rag.ingest", description=__doc__.split("\n")[0])
    parser.add_argument("--show", metavar="FILE", help="print a file's chunks (path relative to knowledge/) instead of indexing")
    args = parser.parse_args(argv)

    if not args.show:
        print("Indexing arrives in step 3. For now, use --show FILE to read a file's chunks.", file=sys.stderr)
        return 2

    root = config.knowledge_dir
    path = (root / args.show).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        print(f"Error: no file '{args.show}' in {root}", file=sys.stderr)
        return 1
    settings = ChunkSettings.from_config(config, model_token_counter(config.rag_embed_model))
    print(show_chunks(chunk_file(path, root, settings), settings.max_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
