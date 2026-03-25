"""Chunking strategies for document splitting.

Supports multiple strategies to demonstrate understanding of
trade-offs in RAG systems:
- Recursive: simple character-based splitting with overlap
- Semantic: similarity-based splitting using embeddings
- Markdown-aware: respects document structure
- Sliding window: fixed-size with configurable overlap
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    index: int
    strategy: str
    token_count: int
    metadata: dict


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[TextChunk]:
        ...

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4


class RecursiveChunker(BaseChunker):
    """Split by paragraphs first, then sentences, then characters."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str) -> list[TextChunk]:
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        return [
            TextChunk(
                content=c.strip(),
                index=i,
                strategy="recursive",
                token_count=self._estimate_tokens(c),
                metadata={},
            )
            for i, c in enumerate(merged)
            if c.strip()
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text] if text else []

        sep = separators[0]
        parts = text.split(sep)
        result = []

        for part in parts:
            if self._estimate_tokens(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursive(part, separators[1:]))

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and add overlap between them."""
        merged = []
        current = ""

        for chunk in chunks:
            if self._estimate_tokens(current + chunk) <= self.chunk_size:
                current += ("\n" if current else "") + chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        # Add overlap
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped = [merged[0]]
            for i in range(1, len(merged)):
                prev_tail = merged[i - 1][-self.chunk_overlap * 4 :]
                overlapped.append(prev_tail + "\n" + merged[i])
            return overlapped

        return merged


class SemanticChunker(BaseChunker):
    """Split based on semantic similarity between sentences.

    Groups consecutive sentences that are semantically similar,
    creating natural topic-based chunks.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        similarity_threshold: float = 0.75,
        max_chunk_tokens: int = 512,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens

    def chunk(self, text: str) -> list[TextChunk]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [
                TextChunk(
                    content=text,
                    index=0,
                    strategy="semantic",
                    token_count=self._estimate_tokens(text),
                    metadata={},
                )
            ]

        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        groups = self._group_by_similarity(sentences, embeddings)

        return [
            TextChunk(
                content=g.strip(),
                index=i,
                strategy="semantic",
                token_count=self._estimate_tokens(g),
                metadata={"sentence_count": len(g.split(". "))},
            )
            for i, g in enumerate(groups)
            if g.strip()
        ]

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_by_similarity(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> list[str]:
        groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            current_text = " ".join(current_group + [sentences[i]])

            if (
                sim >= self.threshold
                and self._estimate_tokens(current_text) <= self.max_chunk_tokens
            ):
                current_group.append(sentences[i])
            else:
                groups.append(" ".join(current_group))
                current_group = [sentences[i]]

        if current_group:
            groups.append(" ".join(current_group))

        return groups


class MarkdownChunker(BaseChunker):
    """Respect markdown heading structure when chunking."""

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[TextChunk]:
        sections = re.split(r"\n(#{1,3}\s+.+)", text)
        chunks = []
        current_heading = ""
        current_content = ""

        for part in sections:
            if re.match(r"#{1,3}\s+", part):
                if current_content.strip():
                    chunks.append(f"{current_heading}\n{current_content}".strip())
                current_heading = part.strip()
                current_content = ""
            else:
                current_content += part

        if current_content.strip():
            chunks.append(f"{current_heading}\n{current_content}".strip())

        return [
            TextChunk(
                content=c,
                index=i,
                strategy="markdown",
                token_count=self._estimate_tokens(c),
                metadata={},
            )
            for i, c in enumerate(chunks)
            if c.strip()
        ]


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window with overlap."""

    def __init__(self, window_size: int = 512, stride: int = 256):
        self.window_size = window_size
        self.stride = stride

    def chunk(self, text: str) -> list[TextChunk]:
        words = text.split()
        chunks = []
        # Approximate words per token
        words_per_window = self.window_size * 3 // 4
        words_per_stride = self.stride * 3 // 4

        i = 0
        idx = 0
        while i < len(words):
            chunk_words = words[i : i + words_per_window]
            content = " ".join(chunk_words)
            chunks.append(
                TextChunk(
                    content=content,
                    index=idx,
                    strategy="sliding_window",
                    token_count=self._estimate_tokens(content),
                    metadata={"word_offset": i},
                )
            )
            i += words_per_stride
            idx += 1

        return chunks


def get_chunker(
    strategy: str, chunk_size: int = 512, chunk_overlap: int = 50
) -> BaseChunker:
    """Factory function to get chunker by strategy name."""
    chunkers = {
        "recursive": lambda: RecursiveChunker(chunk_size, chunk_overlap),
        "semantic": lambda: SemanticChunker(max_chunk_tokens=chunk_size),
        "markdown": lambda: MarkdownChunker(chunk_size),
        "sliding_window": lambda: SlidingWindowChunker(chunk_size, chunk_size // 2),
    }

    if strategy not in chunkers:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(chunkers.keys())}"
        )

    return chunkers[strategy]()
