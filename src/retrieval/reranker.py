"""Cross-encoder reranking for precision refinement.

After hybrid retrieval + RRF fusion returns top candidates,
a cross-encoder jointly encodes (query, passage) pairs to produce
much more accurate relevance scores than bi-encoder similarity.

This is the "two-stage" retrieval pattern used in production systems.
"""

from sentence_transformers import CrossEncoder

from src.retrieval.dense import RetrievalResult

_reranker_cache: dict[str, CrossEncoder] = {}


def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoder:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(model_name, max_length=512)
    return _reranker_cache[model_name]


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RetrievalResult]:
    """Rerank retrieval results using cross-encoder.

    Args:
        query: The user's question.
        results: Candidate results from retrieval stage.
        top_k: Number of results to keep after reranking.

    Returns:
        Reranked results with updated scores.
    """
    if not results:
        return []

    reranker = get_reranker(model_name)

    # Create (query, passage) pairs
    pairs = [(query, r.content) for r in results]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach rerank scores and sort
    scored_results = []
    for result, score in zip(results, scores):
        scored_results.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=float(score),
                document_name=result.document_name,
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "rerank_score": float(score),
                },
            )
        )

    scored_results.sort(key=lambda x: x.score, reverse=True)
    return scored_results[:top_k]
