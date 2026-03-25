"""Reciprocal Rank Fusion (RRF) to combine multiple retrieval results.

RRF is a rank-based fusion method that combines results from different
retrieval strategies without needing score normalization. It uses the
formula: score = Σ 1 / (k + rank_i) for each document across all lists.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (2009)
"""

from collections import defaultdict

from src.retrieval.dense import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists using RRF.

    Args:
        result_lists: Multiple ranked result lists to fuse.
        k: RRF constant (default 60, as per original paper).
        top_k: Number of final results to return.

    Returns:
        Fused and re-ranked results.
    """
    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[int, float] = defaultdict(float)
    chunk_map: dict[int, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rrf_scores[result.chunk_id] += 1.0 / (k + rank + 1)
            # Keep the result with the highest original score
            if (
                result.chunk_id not in chunk_map
                or result.score > chunk_map[result.chunk_id].score
            ):
                chunk_map[result.chunk_id] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    return [
        RetrievalResult(
            chunk_id=chunk_map[cid].chunk_id,
            content=chunk_map[cid].content,
            score=rrf_scores[cid],
            document_name=chunk_map[cid].document_name,
            metadata={**chunk_map[cid].metadata, "rrf_score": rrf_scores[cid]},
        )
        for cid in sorted_ids
    ]
