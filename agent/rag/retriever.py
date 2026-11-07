"""agent/rag/retriever.py — Hybrid search (vector + BM25), RRF fusion, optional rerank (Milestone 5B).

Ported from the earlier RAG project. Vector search catches paraphrases ("cost" finds
"pricing"); BM25 catches exact terms (error codes, function names). Each returns its
top candidates, and Reciprocal Rank Fusion merges the two lists by rank, because
cosine and BM25 scores are on different scales and can't be added directly.
"""
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.rag.chunking import Chunk
from agent.rag.store import KnowledgeStore, Scored

Mode = Literal["hybrid", "vector", "bm25"]  # vector and bm25 alone exist for evals/rag_eval.py


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # RRF score, or the reranker's when it ran

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page(self) -> int | None:
        return self.chunk.page

    @property
    def heading_path(self) -> str:
        return self.chunk.heading_path

    @property
    def text(self) -> str:
        return self.chunk.text


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reads the query and each chunk side by side, so it can judge "answers this" rather than
    just "same topic". Too slow for every chunk, so it only re-scores the fused candidates."""

    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return [float(s) for s in self._model.predict([(query, text) for text in texts])]


def rrf_fuse(rankings: list[list[Scored]], k: int = 60) -> list[Hit]:
    """Each list adds 1 / (k + rank) to a chunk's score, rank starting at 1. Best fused score first."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            scores[scored.chunk.id] = scores.get(scored.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(scored.chunk.id, scored.chunk)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)  # stable: ties keep first-seen order
    return [Hit(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]


class Retriever:
    def __init__(self, store: KnowledgeStore, candidates: int = 20, rrf_k: int = 60, reranker: Reranker | None = None):
        self.store = store
        self.candidates = candidates  # from each of vector and BM25, and how many the reranker sees
        self.rrf_k = rrf_k
        self.reranker = reranker

    @classmethod
    def from_config(cls, config, store: KnowledgeStore) -> "Retriever":
        reranker = CrossEncoderReranker(config.rag_rerank_model) if config.rag_rerank else None
        return cls(store, config.rag_candidates, config.rag_rrf_k, reranker)

    def search(self, query: str, k: int = 8, mode: Mode = "hybrid") -> list[Hit]:
        """The k chunks most relevant to the query, best first."""
        query = query.strip()
        if not query or k < 1:
            return []
        rankings = []
        if mode in ("hybrid", "vector"):
            rankings.append(self.store.vector_search(query, self.candidates))
        if mode in ("hybrid", "bm25"):
            rankings.append(self.store.bm25_search(query, self.candidates))
        if not rankings:
            raise ValueError(f"unknown search mode '{mode}'; use hybrid, vector or bm25")

        hits = rrf_fuse(rankings, self.rrf_k)
        if self.reranker is None or not hits:
            return hits[:k]
        candidates = hits[: self.candidates]
        # The reranker sees what the embedding saw: heading path + text.
        scores = self.reranker.score(query, [hit.chunk.embed_text() for hit in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [Hit(hit.chunk, score) for score, hit in ranked[:k]]
