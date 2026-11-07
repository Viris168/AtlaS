"""Milestone 5B. Offline: token counting is approximate and no model is ever loaded."""
from agent.rag.chunking import (
    ChunkSettings,
    Document,
    approx_tokens,
    chunk_documents,
    clean_heading,
    find_title,
    split_on_heading,
)
from agent.rag.ingest import find_files, load_file, show_chunks

SMALL = ChunkSettings(max_tokens=120, min_tokens=15, h3_split_words=60)


def words(n: int, word: str = "lorem") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def md(text: str, title: str = "Guide") -> Document:
    return Document("guide.md", title, text, markdown=True)


def make_pdf(path, pages: list[str]) -> None:
    """Write a minimal valid PDF with one line of text per page."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objects))
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))


# --- Markdown syntax ---------------------------------------------------------------

def test_split_on_heading_ignores_code_blocks_and_deeper_headings():
    text = "intro\n## One\na\n```\n## not a heading\n```\n### Deeper\nb\n## Two\nc"
    assert [h for h, _ in split_on_heading(text, 2)] == ["", "One", "Two"]


def test_headings_lose_badges_and_links():
    assert clean_heading("NVM [![Build](b.svg)][3] and [docs](http://x)") == "NVM and docs"
    assert find_title("```\n# not this\n```\n# Real Title\n", "fallback") == "Real Title"
    assert find_title("no heading", "fallback") == "fallback"


# --- Markdown chunking -------------------------------------------------------------

def test_one_chunk_per_section_with_heading_path():
    chunks = chunk_documents([md(f"# Guide\n\n## Install\n{words(30)}\n\n## Usage\n{words(30)}")], SMALL)
    assert [c.heading_path for c in chunks] == ["Guide > Install", "Guide > Usage"]
    assert [c.id for c in chunks] == ["guide.md#0", "guide.md#1"]
    assert chunks[0].embed_text().startswith("Guide > Install\n\n## Install")


def test_long_section_splits_on_h3_and_keeps_its_intro():
    text = f"## Setup\n{words(3, 'intro')}\n### Linux\n{words(40)}\n### Mac\n{words(40)}"
    chunks = chunk_documents([md(text)], SMALL)
    assert [c.subsection for c in chunks] == ["Linux", "Mac"]
    assert "intro0" in chunks[0].text  # the short intro merged into the first subsection


def test_short_section_is_not_split_on_h3():
    chunks = chunk_documents([md(f"## Setup\n{words(10)}\n### A\n{words(10)}\n### B\n{words(10)}")], SMALL)
    assert len(chunks) == 1 and chunks[0].subsection == ""


def test_every_chunk_fits_the_limit():
    text = "## Big\n\n" + "\n\n".join(words(40, f"p{i}x") for i in range(12))
    chunks = chunk_documents([md(text)], SMALL)
    assert len(chunks) > 1
    assert all(c.token_count <= SMALL.max_tokens for c in chunks)
    assert all(c.token_count == approx_tokens(c.embed_text()) for c in chunks)


def test_code_blocks_and_tables_are_never_split():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(8)) + "\n```"
    table = "| a | b |\n| - | - |\n" + "\n".join(f"| {i} | {i} |" for i in range(6))
    text = f"## Ref\n\n{words(50)}\n\n{code}\n\n{words(50)}\n\n{table}\n\n{words(50)}"
    chunks = chunk_documents([md(text)], SMALL)
    assert sum(code in c.text for c in chunks) == 1
    assert sum(table in c.text for c in chunks) == 1


def test_oversized_code_block_keeps_its_fences_on_every_piece():
    code = "```sh\n" + "\n".join(f"echo line number {i} of a very long script" for i in range(60)) + "\n```"
    chunks = chunk_documents([md(f"## Script\n\n{code}")], SMALL)
    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n")[-1]
        assert body.startswith("```sh") and body.endswith("```")


def test_tiny_chunks_merge_within_a_section_but_not_across():
    text = f"## A\n{words(40)}\n\n## B\nshort\n\n## C\n{words(40)}"
    chunks = chunk_documents([md(text)], SMALL)
    assert [c.section for c in chunks] == ["A", "B", "C"]  # B has no same-section neighbour, so it stays


def test_title_only_preamble_makes_no_chunk():
    chunks = chunk_documents([md(f"# Guide\n\n## Only\n{words(30)}")], SMALL)
    assert [c.section for c in chunks] == ["Only"]


# --- Plain chunking ----------------------------------------------------------------

def test_plain_text_packs_paragraphs_with_overlap():
    paragraphs = [words(4, f"para{i}w") for i in range(40)]
    doc = Document("notes.txt", "notes", "\n\n".join(paragraphs), markdown=False)
    chunks = chunk_documents([doc], SMALL)
    assert len(chunks) > 2
    assert all(c.token_count <= SMALL.max_tokens for c in chunks)
    for before, after in zip(chunks, chunks[1:]):
        assert before.text.endswith(after.text.split("\n\n")[0])  # the next chunk repeats the end
    assert all(c.heading_path == "notes" for c in chunks)
    assert "para0w0 para0w1 para0w2 para0w3\n\npara1w0" in chunks[0].text  # paragraph breaks kept


def test_pdf_style_text_without_blank_lines_still_overlaps():
    page = "\n".join(words(6, f"line{i}w") for i in range(40))  # one block, many short lines
    chunks = chunk_documents([Document("a.pdf", "A", page, markdown=False, page=1)], SMALL)
    assert len(chunks) > 1
    for before, after in zip(chunks, chunks[1:]):
        assert before.text.endswith(after.text.split("\n")[0])
    assert "\n\n" not in "".join(c.text for c in chunks)  # line breaks stay line breaks


def test_a_huge_single_line_still_fits():
    doc = Document("dump.txt", "dump", words(400), markdown=False)
    assert all(c.token_count <= SMALL.max_tokens for c in chunk_documents([doc], SMALL))


def test_pdf_pages_never_share_a_chunk_and_ids_run_across_pages():
    docs = [Document("a.pdf", "A", words(10, f"page{n}w"), markdown=False, page=n) for n in (1, 2)]
    chunks = chunk_documents(docs, SMALL)
    assert [(c.id, c.page) for c in chunks] == [("a.pdf#0", 1), ("a.pdf#1", 2)]


# --- Loaders -----------------------------------------------------------------------

def test_markdown_and_text_loaders(tmp_path):
    (tmp_path / "guide.md").write_text("# My Guide\n\n## Part\ntext")
    (tmp_path / "notes.txt").write_text("plain words")
    (tmp_path / "empty.md").write_text("  \n")
    [doc] = load_file(tmp_path / "guide.md", tmp_path)
    assert (doc.source, doc.title, doc.markdown) == ("guide.md", "My Guide", True)
    [doc] = load_file(tmp_path / "notes.txt", tmp_path)
    assert (doc.title, doc.markdown) == ("notes", False)
    assert load_file(tmp_path / "empty.md", tmp_path) == []


def test_html_becomes_markdown_without_scripts_or_styles(tmp_path):
    (tmp_path / "page.html").write_text(
        "<html><head><title>Help Page</title><style>body{color:red}</style></head><body>"
        "<script>alert('x')</script><h2>Install</h2><p>Run the &amp; installer.</p>"
        "<h3>Linux</h3><ul><li>one</li><li>two</li></ul></body></html>"
    )
    [doc] = load_file(tmp_path / "page.html", tmp_path)
    assert doc.title == "Help Page" and doc.markdown
    assert "## Install" in doc.text and "### Linux" in doc.text and "Run the & installer." in doc.text
    assert "alert" not in doc.text and "color" not in doc.text


def test_pdf_has_one_document_per_page(tmp_path):
    make_pdf(tmp_path / "manual.pdf", ["First page text", "Second page text"])
    docs = load_file(tmp_path / "manual.pdf", tmp_path)
    assert [(d.page, d.title, d.markdown) for d in docs] == [(1, "manual", False), (2, "manual", False)]
    assert "Second page" in docs[1].text


def test_find_files_skips_hidden_and_unsupported(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".git").mkdir()
    for name in ("a.md", "sub/b.PDF", "c.png", ".hidden.md", ".git/d.md"):
        (tmp_path / name).write_text("x")
    assert [p.relative_to(tmp_path).as_posix() for p in find_files(tmp_path)] == ["a.md", "sub/b.PDF"]


def test_show_chunks_flags_oversized_chunks():
    [chunk] = chunk_documents([md("## A\n" + words(30))], SMALL)
    assert "guide.md#0 · Guide > A" in show_chunks([chunk], SMALL.max_tokens)
    assert "over the limit" in show_chunks([chunk], 5)
