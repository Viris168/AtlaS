"""agent/rag/store.py — Persistent Chroma collection + BM25 keyword index (Milestone 5B).

Ported from the earlier RAG project. Chroma holds every chunk's raw text, its embedding
(of the heading path + text) and metadata; one record per chunk, replaced a whole file
at a time so an edited or deleted section never leaves a stale chunk behind.
BM25 is rebuilt from the same chunks whenever they change.

chromadb, sentence-transformers and rank-bm25 are imported only when first needed,
so the agent starts fast and the tests never load a model.
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.rag.chunking import Chunk

COLLECTION = "knowledge"
_BATCH = 1000  # Chroma rejects very large single adds
_TOKEN = re.compile(r"[a-z0-9_]+")


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """bge-small by default. Vectors are normalized so cosine distance behaves."""

    def __init__(self, model_name: str, query_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix  # bge wants an instruction on queries, never on documents
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25. Identifiers such as get_or_create stay whole
    (so the exact name matches strongly) and are also split on '_' (so 'create' matches too)."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


def _lucene_bm25(corpus: list[list[str]]):
    """BM25 with Lucene's idf, log(1 + (N - n + 0.5) / (n + 0.5)), which is always positive.

    rank-bm25's default (Okapi) idf is zero or below for a word in half the chunks or more,
    so in a small knowledge base a real keyword match could score 0 and be dropped."""
    from rank_bm25 import BM25Okapi

    class LuceneBM25(BM25Okapi):
        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    return LuceneBM25(corpus)


@dataclass(frozen=True)
class Scored:
    chunk: Chunk
    score: float


class KnowledgeStore:
    def __init__(self, index_dir: Path, embedder: Embedder):
        import chromadb
        from chromadb.config import Settings

        index_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))
        # We pass our own embeddings, so no embedding function: Chroma must not load a default model.
        self.collection = client.get_or_create_collection(
            COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder
        self._bm25: tuple[tuple, object, list[Chunk]] | None = None  # (what it was built from, index, chunks)

    # --- Per-file bookkeeping -----------------------------------------------------

    def file_hash(self, source: str) -> str | None:
        """The hash recorded when this file was last indexed, or None if it never was."""
        found = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        return str(found["metadatas"][0]["file_hash"]) if found["metadatas"] else None

    def replace_file(self, source: str, chunks: list[Chunk], file_hash: str) -> None:
        """Swap this file's chunks for new ones. Embeds first, so if that fails the old chunks are kept."""
        embeddings = self.embedder.embed_documents([c.embed_text() for c in chunks]) if chunks else []
        self.collection.delete(where={"source": source})
        for start in range(0, len(chunks), _BATCH):
            end = start + _BATCH
            self.collection.add(
                ids=[c.id for c in chunks[start:end]],
                documents=[c.text for c in chunks[start:end]],
                embeddings=embeddings[start:end],
                metadatas=[_metadata(c, file_hash) for c in chunks[start:end]],
            )

    def remove_missing(self, present: set[str]) -> list[str]:
        """Delete the chunks of every indexed file that's no longer on disk. Returns their paths."""
        removed = sorted({s["source"] for s in self.sources()} - present)
        for source in removed:
            self.collection.delete(where={"source": source})
        return removed

    def sources(self) -> list[dict]:
        """One entry per indexed file: source, title, chunk count and (PDFs) page count."""
        files: dict[str, dict] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            entry = files.setdefault(meta["source"], {"source": meta["source"], "title": meta["title"], "chunks": 0, "pages": set()})
            entry["chunks"] += 1
            if "page" in meta:
                entry["pages"].add(meta["page"])
        return [{**entry, "pages": len(entry["pages"])} for _, entry in sorted(files.items())]

    def count(self) -> int:
        return self.collection.count()

    # --- Search primitives (fused by agent/rag/retriever.py) ----------------------

    def vector_search(self, query: str, n: int) -> list[Scored]:
        """The n chunks closest in meaning. Score is cosine similarity."""
        n = min(n, self.count())
        if n == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        return [Scored(_chunk(i, text, meta), 1.0 - distance) for i, text, meta, distance in rows]

    def bm25_search(self, query: str, n: int) -> list[Scored]:
        """The n best keyword matches. Chunks sharing no word with the query are left out."""
        built = self._bm25_index()
        if built is None:
            return []
        _, index, chunks = built
        scores = index.get_scores(tokenize(query))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [Scored(chunks[i], float(scores[i])) for i in ranked[:n] if scores[i] > 0]

    def _bm25_index(self):
        """BM25 over every chunk's heading path + text, rebuilt only when the chunks change.

        An edited file can keep the same chunk ids, so the cache key is the ids plus each file's hash."""
        stored = self.collection.get(include=["documents", "metadatas"])
        if not stored["ids"]:
            self._bm25 = None
            return None
        key = (tuple(sorted(stored["ids"])), frozenset((m["source"], m["file_hash"]) for m in stored["metadatas"]))
        if self._bm25 is None or self._bm25[0] != key:
            chunks = [_chunk(i, t, m) for i, t, m in zip(stored["ids"], stored["documents"], stored["metadatas"])]
            self._bm25 = (key, _lucene_bm25([tokenize(c.embed_text()) for c in chunks]), chunks)
        return self._bm25


def _metadata(chunk: Chunk, file_hash: str) -> dict:
    meta = {
        "source": chunk.source,
        "title": chunk.title,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "token_count": chunk.token_count,
        "file_hash": file_hash,
    }
    if chunk.page is not None:  # Chroma metadata can't hold None
        meta["page"] = chunk.page
    return meta


def _chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id, meta["source"], meta["title"], meta["section"], meta["subsection"],
        text, int(meta["token_count"]), meta.get("page"),
    )
