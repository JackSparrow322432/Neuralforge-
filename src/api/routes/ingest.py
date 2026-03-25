"""Document ingestion API routes."""

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.chunker import get_chunker
from src.ingestion.embedder import embed_documents
from src.ingestion.loader import DocumentLoader
from src.models.database import Chunk, Document, get_session
from src.models.schemas import IngestResponse, IngestURLRequest

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(default="recursive"),
    chunk_size: int = Form(default=512),
    chunk_overlap: int = Form(default=50),
    session: AsyncSession = Depends(get_session),
):
    """Ingest a PDF or Markdown file."""
    start = time.perf_counter()
    file_bytes = await file.read()

    # Detect file type and load
    if file.filename.endswith(".pdf"):
        doc_data = await DocumentLoader.load_pdf(file_bytes, file.filename)
    elif file.filename.endswith((".md", ".markdown")):
        doc_data = await DocumentLoader.load_markdown(file_bytes, file.filename)
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use PDF or Markdown."
        )

    # Check for duplicate
    from sqlalchemy import select

    existing = await session.execute(
        select(Document).where(Document.content_hash == doc_data["content_hash"])
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Document already ingested.")

    # Chunk the document
    chunker = get_chunker(chunking_strategy, chunk_size, chunk_overlap)
    chunks = chunker.chunk(doc_data["text"])

    # Generate embeddings in batch
    texts = [c.content for c in chunks]
    embeddings = embed_documents(texts)

    # Save document
    doc = Document(
        filename=doc_data["filename"],
        source_type=doc_data["source_type"],
        content_hash=doc_data["content_hash"],
        total_chunks=len(chunks),
        metadata_=doc_data["metadata"],
    )
    session.add(doc)
    await session.flush()

    # Save chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        db_chunk = Chunk(
            document_id=doc.id,
            content=chunk.content,
            chunk_index=chunk.index,
            chunking_strategy=chunk.strategy,
            token_count=chunk.token_count,
            embedding=embedding.tolist(),
            metadata_=chunk.metadata,
        )
        session.add(db_chunk)

    await session.commit()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return IngestResponse(
        document_id=doc.id,
        filename=doc.filename,
        total_chunks=len(chunks),
        chunking_strategy=chunking_strategy,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(
    request: IngestURLRequest,
    session: AsyncSession = Depends(get_session),
):
    """Ingest a web page by URL."""
    start = time.perf_counter()

    try:
        doc_data = await DocumentLoader.load_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # Check duplicate
    from sqlalchemy import select

    existing = await session.execute(
        select(Document).where(Document.content_hash == doc_data["content_hash"])
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Document already ingested.")

    chunker = get_chunker(
        request.chunking_strategy, request.chunk_size, request.chunk_overlap
    )
    chunks = chunker.chunk(doc_data["text"])
    texts = [c.content for c in chunks]
    embeddings = embed_documents(texts)

    doc = Document(
        filename=doc_data["filename"],
        source_type=doc_data["source_type"],
        content_hash=doc_data["content_hash"],
        total_chunks=len(chunks),
        metadata_=doc_data["metadata"],
    )
    session.add(doc)
    await session.flush()

    for chunk, embedding in zip(chunks, embeddings):
        db_chunk = Chunk(
            document_id=doc.id,
            content=chunk.content,
            chunk_index=chunk.index,
            chunking_strategy=chunk.strategy,
            token_count=chunk.token_count,
            embedding=embedding.tolist(),
            metadata_=chunk.metadata,
        )
        session.add(db_chunk)

    await session.commit()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return IngestResponse(
        document_id=doc.id,
        filename=doc.filename,
        total_chunks=len(chunks),
        chunking_strategy=request.chunking_strategy,
        processing_time_ms=round(elapsed_ms, 2),
    )
