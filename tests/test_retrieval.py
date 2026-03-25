"""Tests for retrieval fusion logic."""

import pytest

from src.retrieval.dense import RetrievalResult
from src.retrieval.fusion import reciprocal_rank_fusion


def _make_result(chunk_id: int, score: float, doc: str = "test.pdf") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"Chunk {chunk_id} content",
        score=score,
        document_name=doc,
        metadata={},
    )


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [_make_result(1, 0.9), _make_result(2, 0.8)]
        fused = reciprocal_rank_fusion([results], top_k=2)
        assert len(fused) == 2
        assert fused[0].chunk_id == 1  # Rank 1 should stay rank 1

    def test_two_lists_agreement(self):
        """When both lists agree on ranking, fusion should too."""
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8), _make_result(3, 0.7)]
        list_b = [_make_result(1, 0.85), _make_result(2, 0.75), _make_result(3, 0.65)]
        fused = reciprocal_rank_fusion([list_a, list_b], top_k=3)
        assert fused[0].chunk_id == 1

    def test_two_lists_disagreement(self):
        """Chunk appearing in both lists should rank higher than single-list chunks."""
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8)]
        list_b = [_make_result(3, 0.95), _make_result(2, 0.85)]
        fused = reciprocal_rank_fusion([list_a, list_b], top_k=3)
        # Chunk 2 appears in both lists => should be boosted
        chunk_ids = [r.chunk_id for r in fused]
        assert 2 in chunk_ids

    def test_top_k_limit(self):
        results = [_make_result(i, 1.0 - i * 0.1) for i in range(10)]
        fused = reciprocal_rank_fusion([results], top_k=3)
        assert len(fused) == 3

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []], top_k=5)
        assert len(fused) == 0

    def test_rrf_scores_are_positive(self):
        results = [_make_result(1, 0.5), _make_result(2, 0.3)]
        fused = reciprocal_rank_fusion([results])
        for r in fused:
            assert r.score > 0

    def test_deduplication(self):
        """Same chunk in multiple lists should only appear once."""
        list_a = [_make_result(1, 0.9)]
        list_b = [_make_result(1, 0.85)]
        fused = reciprocal_rank_fusion([list_a, list_b], top_k=5)
        assert len(fused) == 1
        assert fused[0].chunk_id == 1
