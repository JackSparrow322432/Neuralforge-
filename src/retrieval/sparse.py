"""Sparse retrieval using BM25.

BM25 captures exact keyword matches that dense retrieval might miss,
especially for domain-specific terms, acronyms, and proper nouns.
"""

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.retrieval.dense import RetrievalResult


class BM25Index:
    """In-memory BM25 index built from database chunks."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunk_ids: list[int] = []
        self._contents: list[str] = []
        self._doc_names: list[str] = []

    async def build(self, session: AsyncSession):
        """Build BM25 index from all chunks in the database."""
        sql = text("""
            SELECT c.id, c.content, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.id
        """)
        result = await session.execute(sql)
        rows = result.fetchall()

        self._chunk_ids = [r.id for r in rows]
        self._contents = [r.content for r in rows]
        self._doc_names = [r.filename for r in rows]

        tokenized = [doc.lower().split() for doc in self._contents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if self._index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(
                chunk_id=self._chunk_ids[i],
                content=self._contents[i],
                score=float(scores[i]),
                document_name=self._doc_names[i],
                metadata={"retrieval_method": "bm25"},
            )
            for i in top_indices
            if scores[i] > 0
        ]


# Singleton index instance
_bm25_index = BM25Index()


async def sparse_search(
    query: str,
    session: AsyncSession,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """Search using BM25 sparse retrieval."""
    # Rebuild index if needed (in production, cache with invalidation)
    await _bm25_index.build(session)
    return _bm25_index.search(query, top_k)
