"""Tests for evaluation engine score parsing."""

import pytest

from src.evaluation.evaluator import _parse_score


class TestScoreParsing:
    def test_json_format(self):
        assert _parse_score('{"faithfulness": 0.85}', "faithfulness") == 0.85

    def test_colon_format(self):
        assert _parse_score("faithfulness: 0.92", "faithfulness") == 0.92

    def test_equals_format(self):
        assert _parse_score("relevance = 0.78", "relevance") == 0.78

    def test_integer_score(self):
        assert _parse_score('{"correctness": 1}', "correctness") == 1.0

    def test_clamp_above_one(self):
        assert _parse_score("faithfulness: 1.5", "faithfulness") == 1.0

    def test_clamp_below_zero(self):
        assert _parse_score("faithfulness: -0.3", "faithfulness") == 0.0

    def test_no_match_returns_default(self):
        assert _parse_score("I don't know", "faithfulness") == 0.5

    def test_case_insensitive(self):
        assert _parse_score("Faithfulness: 0.9", "faithfulness") == 0.9

    def test_with_surrounding_text(self):
        text = 'Based on my analysis, {"faithfulness": 0.88} is the score.'
        assert _parse_score(text, "faithfulness") == 0.88

    def test_hallucination_zero(self):
        assert _parse_score('{"hallucination": 0.0}', "hallucination") == 0.0
