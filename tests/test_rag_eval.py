"""The retrieval eval's own logic, offline (fake embedder, approximate tokens)."""
import json

import pytest

from agent.config import Config
from agent.rag.chunking import Chunk
from agent.rag.retriever import Hit
from evals.rag_eval import DOCS, QUESTIONS, Question, Score, is_expected, load_questions, rank_of, run
from tests.test_rag import SMALL, FakeEmbedder, FakeReranker, words


def chunk(source="a.md", section="Setup", subsection="", text="body") -> Chunk:
    return Chunk(f"{source}#0", source, "A", section, subsection, text, 10)


def test_matching_rules():
    q = Question("?", "a.md", section="Setup", subsection="Linux")
    assert is_expected(q, chunk(subsection="Linux"))
    assert is_expected(q, chunk(subsection="Intro / Linux"))  # merged chunk
    assert is_expected(q, chunk(text="## Setup\n### Linux\nsteps"))  # section wasn't split on ###
    assert not is_expected(q, chunk(subsection="Mac"))
    assert not is_expected(q, chunk(source="b.md", subsection="Linux"))
    assert not is_expected(q, chunk(section="Usage", subsection="Linux"))

    preamble = Question("?", "a.md", section="")
    assert is_expected(preamble, chunk(section="")) and not is_expected(preamble, chunk())
    plain = Question("?", "notes.txt", contains="port 9000")
    assert is_expected(plain, chunk(source="notes.txt", section="", text="now port 9000"))
    assert not is_expected(plain, chunk(source="notes.txt", section="", text="port 8080"))


def test_rank_and_scores():
    q = Question("?", "a.md", section="Setup")
    hits = [Hit(chunk(section="Other"), 1.0), Hit(chunk(), 0.5)]
    assert rank_of(q, hits) == 2 and rank_of(q, hits[:1]) is None
    s = Score([1, 2, None, 4])
    assert s.hit_at(1) == 0.25 and s.hit_at(3) == 0.5 and s.hit_at(8) == 0.75
    assert s.mrr == pytest.approx((1 + 1 / 2 + 1 / 4) / 4)


def test_bad_question_files_are_rejected(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "?", "source": "a.md", "sectoin": "typo"}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)
    path.write_text("\n")
    with pytest.raises(ValueError, match="no questions"):
        load_questions(path)


def test_committed_questions_point_at_committed_docs():
    questions = load_questions(QUESTIONS)
    sources = {p.name for p in DOCS.iterdir()}
    assert len(questions) >= 20
    assert {q.source for q in questions} <= sources


def test_run_reports_every_setup(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "db.md").write_text("# DB\n\n## Collections\nUse get_or_create_collection to open the vector database. " + words(20))
    (docs / "ops.md").write_text("# Ops\n\n## Errors\nHTTP 429 means you hit the rate limit. " + words(20, "y"))
    questions = tmp_path / "q.jsonl"
    questions.write_text("\n".join(json.dumps(q) for q in [
        {"question": "get_or_create_collection", "source": "db.md", "section": "Collections"},
        {"question": "what does 429 mean", "source": "ops.md", "section": "Errors"},
    ]))
    config = Config(data_dir=tmp_path / "d", workspace=tmp_path / "w", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")

    out = run(config, docs, questions, k=2, embedder=FakeEmbedder(), settings=SMALL, reranker=FakeReranker("Errors"))
    assert out.startswith("2 questions, 2 chunks, top 2")
    for name in ("vector", "bm25", "hybrid", "hybrid+rerank"):
        assert name in out
    assert "hit@3" not in out  # only cutoffs up to k
    [hybrid_row] = [line for line in out.splitlines() if line.startswith("hybrid ")]
    assert hybrid_row.split() == ["hybrid", "100%", "100%", "1.00"]
