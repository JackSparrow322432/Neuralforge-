"""Dense retrieval via pgvector cosine similarity."""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.embedder import embed_query


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    score: float
    document_name: str
    metadata: dict


async def dense_search(
    query: str,
    session: AsyncSession,
    top_k: int = 20,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> list[RetrievalResult]:
    """Search using pgvector cosine similarity."""
    query_embedding = embed_query(query, model_name)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql = text("""
        SELECT
            c.id,
            c.content,
            1 - (c.embedding <=> :embedding::vector) AS score,
            d.filename
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> :embedding::vector
        LIMIT :top_k
    """)

    result = await session.execute(
        sql, {"embedding": embedding_str, "top_k": top_k}
    )

    return [
        RetrievalResult(
            chunk_id=row.id,
            content=row.content,
            score=float(row.score),
            document_name=row.filename,
            metadata={},
        )
        for row in result.fetchall()
    ]
