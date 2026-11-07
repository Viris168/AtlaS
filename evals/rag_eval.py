"""evals/rag_eval.py — Retrieval eval for the knowledge base (Milestone 5B, step 7).

Builds a throwaway index of evals/rag_docs/ and checks, for each question in
evals/rag_questions.jsonl, where the chunk holding the answer ranks under each search
setup. No LLM calls; it needs the embedding model (and the reranker, unless --no-rerank).

    python -m evals.rag_eval
    python -m evals.rag_eval --no-rerank --k 5 --details

Each question names where its answer lives: "source" (required), "section" (the ## heading,
"" for text before the first one), "subsection" (a ### heading) and/or "contains" (a phrase
the chunk must include, for files without headings).

Every question is scored in two forms: as a person asks it ("question"), and as the agent
would search for it ("query", keyword style). A query may use only words from the question
plus what the agent sees in its prompt (file titles and ## headings), never words that
appear only in the answer. "kind": "exact" marks questions built around an identifier,
error code or flag, which is where keyword search should help most.
"""
import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.config import Config, config as default_config
from agent.rag.chunking import Chunk, ChunkSettings
from agent.rag.ingest import ingest, model_token_counter
from agent.rag.retriever import CrossEncoderReranker, Hit, Reranker, Retriever
from agent.rag.store import Embedder, KnowledgeStore, SentenceTransformerEmbedder

EVALS = Path(__file__).resolve().parent
DOCS = EVALS / "rag_docs"
QUESTIONS = EVALS / "rag_questions.jsonl"

Search = Callable[[str, int], list[Hit]]


@dataclass(frozen=True)
class Question:
    question: str
    source: str
    section: str | None = None
    subsection: str | None = None
    contains: str | None = None
    query: str | None = None  # the agent-style search; falls back to the question
    kind: str = "natural"  # or "exact"

    def text(self, form: str) -> str:
        return self.query or self.question if form == "query" else self.question


FORMS = {"asked": "as asked", "query": "as the agent searches"}  # natural-language question / keyword-style query


def load_questions(path: Path) -> list[Question]:
    questions = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(Question(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"{path.name} line {n}: {e}") from e
    if not questions:
        raise ValueError(f"{path} has no questions")
    return questions


def is_expected(q: Question, chunk: Chunk) -> bool:
    """Does this chunk hold the answer's location? A ### subsection counts if the chunk is
    that subsection or still contains its heading (sections under 400 words aren't split)."""
    if chunk.source != q.source:
        return False
    if q.section is not None and chunk.section != q.section:
        return False
    if q.subsection and q.subsection not in chunk.subsection.split(" / ") and f"### {q.subsection}" not in chunk.text:
        return False
    return not q.contains or q.contains in chunk.text


def rank_of(q: Question, hits: list[Hit]) -> int | None:
    """1-based rank of the first chunk that holds the answer, or None if it's missing."""
    return next((rank for rank, hit in enumerate(hits, start=1) if is_expected(q, hit.chunk)), None)


@dataclass(frozen=True)
class Score:
    ranks: list[int | None]

    def hit_at(self, n: int) -> float:
        return sum(rank is not None and rank <= n for rank in self.ranks) / len(self.ranks)

    @property
    def mrr(self) -> float:
        """1.0 if always first, 0.5 if always second, 0 if missing."""
        return sum(1 / rank for rank in self.ranks if rank) / len(self.ranks)


def score(search: Search, questions: list[Question], k: int, form: str = "asked") -> Score:
    return Score([rank_of(q, search(q.text(form), k)) for q in questions])


def build_index(docs: Path, index_dir: Path, embedder: Embedder, settings: ChunkSettings, embed_model: str) -> KnowledgeStore:
    store = KnowledgeStore(index_dir, embedder)
    report = ingest(docs, store, settings, embed_model)
    if report.failed:
        raise RuntimeError(f"could not index: {report.failed}")
    return store


def setups(store: KnowledgeStore, config: Config, reranker: Reranker | None) -> list[tuple[str, Search]]:
    plain = Retriever(store, config.rag_candidates, config.rag_rrf_k)
    rows: list[tuple[str, Search]] = [
        ("vector", lambda q, k: plain.search(q, k, mode="vector")),
        ("bm25", lambda q, k: plain.search(q, k, mode="bm25")),
        ("hybrid", lambda q, k: plain.search(q, k)),
    ]
    if reranker is not None:
        reranked = Retriever(store, config.rag_candidates, config.rag_rrf_k, reranker)
        rows.append(("hybrid+rerank", lambda q, k: reranked.search(q, k)))
    return rows


def _table(title: str, results: dict[str, Score], k: int) -> list[str]:
    cutoffs = sorted({n for n in (1, 3, k) if n <= k})
    lines = [title, f"{'setup':15}" + "".join(f"{f'hit@{n}':>7}" for n in cutoffs) + f"{'MRR':>7}"]
    for name, s in results.items():
        lines.append(f"{name:15}" + "".join(f"{s.hit_at(n):>7.0%}" for n in cutoffs) + f"{s.mrr:>7.2f}")
    return lines


def _subset(results: dict[str, Score], keep: list[int]) -> dict[str, Score]:
    return {name: Score([s.ranks[i] for i in keep]) for name, s in results.items()}


def _details(title: str, questions: list[Question], results: dict[str, Score], form: str, k: int) -> list[str]:
    names = list(results)
    not_first = [i for i in range(len(questions)) if any(results[name].ranks[i] != 1 for name in names)]
    if not not_first:
        return []
    lines = ["", f"{title}: rank of the answer where any setup missed first place (- = not in top {k})"]
    lines.append(f"{'':60} " + " ".join(f"{name:>13}" for name in names))
    for i in not_first:
        text = questions[i].text(form)
        text = text if len(text) <= 58 else text[:57] + "…"
        ranks = (results[name].ranks[i] for name in names)
        lines.append(f"{text:60} " + " ".join(f"{(str(r) if r else '-'):>13}" for r in ranks))
    return lines


def report(questions: list[Question], by_form: dict[str, dict[str, Score]], k: int, chunks: int, details: bool = False) -> str:
    exact = [i for i, q in enumerate(questions) if q.kind == "exact"]
    lines = [f"{len(questions)} questions ({len(exact)} exact-term), {chunks} chunks, top {k}"]
    for form, results in by_form.items():
        lines += [""] + _table(FORMS[form].capitalize(), results, k)
    if exact:
        for form, results in by_form.items():
            lines += [""] + _table(f"Exact-term questions only, {FORMS[form]}", _subset(results, exact), k)
    if details:
        for form, results in by_form.items():
            lines += _details(FORMS[form].capitalize(), questions, results, form, k)
    return "\n".join(lines)


def run(
    config: Config = default_config,
    docs: Path = DOCS,
    questions_path: Path = QUESTIONS,
    k: int | None = None,
    embedder: Embedder | None = None,
    settings: ChunkSettings | None = None,
    reranker: Reranker | None = None,
    rerank: bool = True,
    details: bool = False,
) -> str:
    k = k or config.rag_top_k
    questions = load_questions(questions_path)
    embedder = embedder or SentenceTransformerEmbedder(config.rag_embed_model, config.rag_query_prefix)
    settings = settings or ChunkSettings.from_config(config, model_token_counter(config.rag_embed_model))
    if rerank and reranker is None:
        reranker = CrossEncoderReranker(config.rag_rerank_model)
    with tempfile.TemporaryDirectory(prefix="atlas-rag-eval-") as tmp:
        store = build_index(docs, Path(tmp), embedder, settings, config.rag_embed_model)
        rows = setups(store, config, reranker if rerank else None)
        by_form = {form: {name: score(search, questions, k, form) for name, search in rows} for form in FORMS}
        return report(questions, by_form, k, store.count(), details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.rag_eval", description="Retrieval eval for the knowledge base.")
    parser.add_argument("--k", type=int, help=f"results per search (default: rag_top_k, {default_config.rag_top_k})")
    parser.add_argument("--no-rerank", action="store_true", help="skip the hybrid+rerank row (no cross-encoder download)")
    parser.add_argument("--details", action="store_true", help="also show each question any setup didn't rank first")
    parser.add_argument("--docs", type=Path, default=DOCS, help="folder of documents to index")
    parser.add_argument("--questions", type=Path, default=QUESTIONS, help="JSONL file of questions")
    args = parser.parse_args(argv)
    print(run(k=args.k, docs=args.docs, questions_path=args.questions, rerank=not args.no_rerank, details=args.details))
    return 0


if __name__ == "__main__":
    sys.exit(main())
