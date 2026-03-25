"""Tests for chunking strategies."""

import pytest

from src.ingestion.chunker import (
    MarkdownChunker,
    RecursiveChunker,
    SlidingWindowChunker,
    get_chunker,
)

SAMPLE_TEXT = """
Artificial intelligence has transformed numerous industries over the past decade.
Machine learning, a subset of AI, enables systems to learn from data without
explicit programming. Deep learning, using neural networks with many layers,
has achieved remarkable results in computer vision and natural language processing.

The transformer architecture, introduced in 2017, revolutionized NLP.
Models like BERT and GPT demonstrated that pre-training on large text corpora
followed by fine-tuning could achieve state-of-the-art results across many tasks.
This paradigm shift led to the development of large language models.

Retrieval-Augmented Generation combines the strengths of retrieval systems
with generative models. Instead of relying solely on parametric knowledge,
RAG systems retrieve relevant documents and use them as context for generation.
This approach reduces hallucinations and improves factual accuracy.
""".strip()


SAMPLE_MARKDOWN = """
# Introduction to RAG

RAG stands for Retrieval-Augmented Generation. It combines search with AI generation.

## How It Works

The system first retrieves relevant documents from a knowledge base.
Then it uses these documents as context for the language model.

### Retrieval Stage

Documents are embedded as vectors and stored in a vector database.
When a query arrives, it is also embedded and similar vectors are found.

## Evaluation

Measuring RAG quality requires multiple metrics including faithfulness and relevance.
""".strip()


class TestRecursiveChunker:
    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
        assert all(c.strategy == "recursive" for c in chunks)

    def test_overlap(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # With overlap, later chunks should contain text from previous chunks
        if len(chunks) > 1:
            assert len(chunks[1].content) > len(chunks[0].content) * 0.1

    def test_empty_text(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_short_text(self):
        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."

    def test_token_count(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count == len(chunk.content) // 4


class TestMarkdownChunker:
    def test_respects_headings(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert len(chunks) >= 3  # At least 3 heading sections

    def test_heading_preserved(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        headings_found = [c for c in chunks if "#" in c.content]
        assert len(headings_found) > 0

    def test_strategy_label(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert all(c.strategy == "markdown" for c in chunks)


class TestSlidingWindowChunker:
    def test_window_generation(self):
        chunker = SlidingWindowChunker(window_size=50, stride=25)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1

    def test_overlap_between_windows(self):
        chunker = SlidingWindowChunker(window_size=100, stride=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        if len(chunks) >= 2:
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0  # Windows should overlap

    def test_strategy_label(self):
        chunker = SlidingWindowChunker()
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert all(c.strategy == "sliding_window" for c in chunks)


class TestChunkerFactory:
    def test_recursive(self):
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_markdown(self):
        chunker = get_chunker("markdown")
        assert isinstance(chunker, MarkdownChunker)

    def test_sliding_window(self):
        chunker = get_chunker("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_chunker("nonexistent")

    def test_custom_params(self):
        chunker = get_chunker("recursive", chunk_size=256, chunk_overlap=25)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25
