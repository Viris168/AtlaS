"""scripts/fetch_docs.py — Download the FastAPI and Chroma docs into knowledge/ as plain Markdown.

    python scripts/fetch_docs.py             # refresh knowledge/fastapi/ and knowledge/chromadb/
    python scripts/fetch_docs.py --ingest    # ...then sync the RAG index

Both sites keep their docs in GitHub, in their own dialects: FastAPI uses mkdocs with code pulled
in from docs_src/ by `{* ... *}` includes, Chroma uses Mintlify MDX with JSX components and code
in several languages. This sparse-clones each repo into a temp folder, inlines the FastAPI code,
strips the JSX, keeps only Python/shell examples, and writes one .md per page. Each run replaces
both output folders, so pages removed upstream disappear here too.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `agent` imports when run as a script

from agent.config import config

FASTAPI_REPO = "https://github.com/fastapi/fastapi.git"
CHROMA_REPO = "https://github.com/chroma-core/chroma.git"
MIN_CHARS = 200  # pages shorter than this after conversion are index stubs

FASTAPI_ADVANCED = (
    "events", "settings", "middleware", "custom-response", "stream-data", "websockets",
    "async-tests", "testing-dependencies", "behind-a-proxy", "advanced-dependencies",
    "response-directly", "using-request-directly", "additional-responses", "sub-applications",
)
CHROMA_DIRS = ("docs/collections", "docs/embeddings", "docs/querying-collections", "docs/run-chroma",
               "guides/build", "reference/python")
CHROMA_PAGES = (
    "docs/overview/introduction", "docs/overview/getting-started", "docs/overview/troubleshooting",
    "docs/overview/migration", "guides/deploy/client-server-mode", "guides/deploy/docker",
    "guides/deploy/observability", "guides/deploy/python-thin-client",
    "guides/performance/single-node", "reference/server-env-vars",
)
CHROMA_CODE_LANGS = {"", "python", "py", "bash", "sh", "shell", "terminal", "text", "json", "yaml",
                     "dockerfile", "docker"}

INCLUDE = re.compile(r"\{\*\s+(\S+)(.*?)\*\}")
FENCE = re.compile(r"^```(\S*)[^\n]*\n.*?^```\s*$\n?", re.S | re.M)


def sparse_clone(url: str, dest: Path, paths: list[str]) -> None:
    """Clone only `paths` of the repo; git's progress output is shown only if a step fails."""
    for cmd in (["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "-q", url, str(dest)],
                ["git", "-C", str(dest), "sparse-checkout", "set", *paths]):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"{' '.join(cmd[:3])} failed:\n{result.stderr}")


# ---------- FastAPI ----------

def _slice(text: str, spec: str) -> str:
    """Apply an include's ln[a:b,c:d] line ranges (1-based, inclusive)."""
    m = re.search(r"ln\[([^\]]*)\]", spec)
    if not m:
        return text
    lines = text.splitlines()
    out = []
    for part in m.group(1).split(","):
        a, sep, b = part.partition(":")
        start = int(a) if a else 1
        end = (int(b) if b else len(lines)) if sep else start
        if out:
            out.append("# ...")
        out.extend(lines[start - 1:end])
    return "\n".join(out)


def convert_fastapi(path: Path, repo: Path) -> str:
    text = path.read_text()
    text = re.sub(r"\s*\{ #[\w-]+ \}", "", text)  # heading anchors

    def include(m):
        src = repo / "docs_src" / m.group(1).split("docs_src/", 1)[-1]
        if not src.exists():
            return ""
        return "```python\n" + _slice(src.read_text(), m.group(2)).rstrip() + "\n```"

    def admonition(m):
        kind, _, title = m.group(1).partition("|")
        return f"**{kind.strip().capitalize()}{f' ({title.strip()})' if title.strip() else ''}:**"

    text = INCLUDE.sub(include, text)
    text = re.sub(r"^//// tab \| (.*)$", r"**\1:**", text, flags=re.M)
    text = re.sub(r"^////\s*$\n?", "", text, flags=re.M)
    text = re.sub(r"^/// ?(\w[^\n]*)$", admonition, text, flags=re.M)
    text = re.sub(r"^///\s*$\n?", "", text, flags=re.M)
    text = re.sub(r"<abbr[^>]*>(.*?)</abbr>", r"\1", text)
    return text.strip()


def fetch_fastapi(work: Path, out: Path) -> int:
    repo = work / "fastapi"
    sparse_clone(FASTAPI_REPO, repo, ["docs/en/docs/tutorial", "docs/en/docs/advanced", "docs_src"])
    docs = repo / "docs/en/docs"
    pages = sorted((docs / "tutorial").rglob("*.md")) + [docs / "advanced" / f"{n}.md" for n in FASTAPI_ADVANCED]
    return write_pages(pages, docs, out, lambda p: convert_fastapi(p, repo),
                       "FastAPI docs (MIT License), https://fastapi.tiangolo.com")


# ---------- Chroma ----------

def convert_chroma(path: Path) -> str:
    text = path.read_text()
    title = ""
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if front:
        t = re.search(r'^title:\s*"?(.*?)"?\s*$', front.group(1), re.M)
        title = t.group(1) if t else ""
        text = text[front.end():]
    text = FENCE.sub(lambda m: m.group(0) if m.group(1).lower() in CHROMA_CODE_LANGS else "", text)
    # Park code blocks so the JSX stripping below can't touch them.
    blocks = []
    text = FENCE.sub(lambda m: blocks.append(m.group(0)) or f"\0{len(blocks) - 1}\0", text)
    text = re.sub(r"^(import|export) .*$", "", text, flags=re.M)
    text = re.sub(r"<(Tab|Accordion|Card|Step)\s+title=\"([^\"]*)\"[^>]*>", r"**\2**", text)
    text = re.sub(r"</?[A-Z][\w.]*[^>]*?/?>", "", text)
    text = re.sub(r"<img[^>]*>|</?(div|span|br|p)[^>]*>", "", text)
    text = re.sub(r"\0(\d+)\0", lambda m: blocks[int(m.group(1))], text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title and not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    return text


def fetch_chroma(work: Path, out: Path) -> int:
    repo = work / "chroma"
    sparse_clone(CHROMA_REPO, repo, ["docs/mintlify"])
    docs = repo / "docs/mintlify"
    pages = [p for d in CHROMA_DIRS for p in sorted((docs / d).glob("*.mdx"))]
    pages += [docs / f"{n}.mdx" for n in CHROMA_PAGES]
    return write_pages(pages, docs, out, convert_chroma, "Chroma docs (Apache-2.0), https://docs.trychroma.com")


# ---------- shared ----------

def write_pages(pages: list[Path], root: Path, out: Path, convert, source: str) -> int:
    """Convert each page to out/<path__with__underscores>.md; returns how many were written."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = 0
    for page in pages:
        if not page.exists():
            print(f"  missing upstream, skipped: {page.relative_to(root)}")
            continue
        body = convert(page)
        if len(body) > MIN_CHARS:
            name = page.relative_to(root).with_suffix(".md").as_posix().replace("/", "__")
            (out / name).write_text(f"{body}\n\nSource: {source}\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ingest", action="store_true", help="sync the RAG index afterwards")
    args = parser.parse_args()

    knowledge = config.knowledge_dir
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print(f"FastAPI: {fetch_fastapi(work, knowledge / 'fastapi')} pages -> {knowledge / 'fastapi'}")
        print(f"Chroma:  {fetch_chroma(work, knowledge / 'chromadb')} pages -> {knowledge / 'chromadb'}")

    if args.ingest:
        from agent.rag.ingest import run_ingest
        report = run_ingest(config)
        print(f"ingest: {len(report.added)} added, {len(report.updated)} updated, "
              f"{len(report.removed)} removed, {report.chunks_written} chunks written")


if __name__ == "__main__":
    main()
