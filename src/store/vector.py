"""ChromaDB vector store interface.

Implements: HLD 3.3
Satisfies: FR-04, FR-05, FR-06
"""
from dataclasses import dataclass, field

import chromadb

from ingestion.chunker import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalProvenance:
    """Stage-specific retrieval metadata preserved across fusion and reranking."""
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)


def clone_retrieval_provenance(provenance: RetrievalProvenance) -> RetrievalProvenance:
    """Return a copy of retrieval provenance without sharing mutable source lists."""
    return RetrievalProvenance(
        dense_rank=provenance.dense_rank,
        dense_score=provenance.dense_score,
        sparse_rank=provenance.sparse_rank,
        sparse_score=provenance.sparse_score,
        rrf_score=provenance.rrf_score,
        rerank_score=provenance.rerank_score,
        sources=list(provenance.sources),
    )


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with score and method metadata."""
    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    retrieval_method: str  # "dense" | "sparse" | "hybrid" | "reranked"
    provenance: RetrievalProvenance = field(default_factory=RetrievalProvenance)


class VectorStore:
    """ChromaDB interface — all reads and writes to the vector store go through here."""

    def __init__(self, collection_name: str, persist_directory: str = "", ephemeral: bool = False):
        self._collection_name = collection_name
        if ephemeral:
            self._client = chromadb.EphemeralClient()
            logger.info(f"VectorStore initialized (EPHEMERAL): collection='{collection_name}'")
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(
                f"VectorStore initialized (PERSISTENT): collection='{collection_name}', "
                f"path='{persist_directory}'"
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Existing count in collection: {self._collection.count()}")

    @property
    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._collection.count()

    def get_chunk_ids(self) -> list[str]:
        """Return all stored chunk IDs for index consistency checks."""
        return list(self._collection.get()["ids"])

    def reset(self) -> None:
        """Replace the collection with an empty collection for a full rebuild."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Reset vector collection '{self._collection_name}'")

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """
        Insert chunks with embeddings. Skips any chunk_id already present (idempotency).

        Returns:
            Number of chunks actually inserted (0 if all were duplicates)
        """
        # Filter out chunks that already exist
        new_chunks = []
        new_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if not self.exists(chunk.id):
                new_chunks.append(chunk)
                new_embeddings.append(embedding)

        if not new_chunks:
            logger.info("No new chunks to insert (all duplicates)")
            return 0

        self._collection.add(
            ids=[c.id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            embeddings=new_embeddings,
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in new_chunks
            ],
        )

        logger.info(
            f"Inserted {len(new_chunks)} chunks "
            f"(skipped {len(chunks) - len(new_chunks)} duplicates)"
        )
        return len(new_chunks)

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return top_k chunks by cosine similarity to query_embedding."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns cosine distance; convert to similarity
            score = 1.0 - results["distances"][0][i]
            metadata = results["metadatas"][0][i]

            retrieved.append(RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                source=metadata["source"],
                page=metadata["page"],
                score=score,
                retrieval_method="dense",
                provenance=RetrievalProvenance(
                    dense_rank=i + 1,
                    dense_score=score,
                    sources=["dense"],
                ),
            ))

        return retrieved

    def exists(self, chunk_id: str) -> bool:
        """Check if a chunk_id is already stored."""
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0
