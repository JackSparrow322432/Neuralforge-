"""Embedding generation using BGE-large-en-v1.5.

BGE models use a specific instruction prefix for queries vs. documents:
- Documents: embedded as-is
- Queries: prefixed with "Represent this sentence for searching relevant passages: "
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> SentenceTransformer:
    """Cached model loading to avoid reloading on every request."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_documents(
    texts: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
) -> np.ndarray:
    """Embed document chunks (no instruction prefix)."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return embeddings


def embed_query(
    query: str,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> np.ndarray:
    """Embed a search query with BGE instruction prefix."""
    model = get_embedding_model(model_name)
    instruction = (
        "Represent this sentence for searching relevant passages: " + query
    )
    embedding = model.encode(
        [instruction],
        normalize_embeddings=True,
    )
    return embedding[0]
