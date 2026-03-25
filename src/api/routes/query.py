"""Query API route — orchestrates the full RAG pipeline."""

import time

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.evaluation.evaluator import evaluate_response
from src.models.database import QueryLog, get_session
from src.models.schemas import (
    EvaluationResult,
    PipelineMetrics,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from src.retrieval.dense import dense_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.sparse import sparse_search

OLLAMA_URL = "http://localhost:11434"

router = APIRouter()


SYSTEM_PROMPT = """You are a precise, helpful assistant. Answer the question
based ONLY on the provided context. If the context doesn't contain enough
information, say so clearly. Do not make up facts.

CONTEXT:
{context}"""


async def generate_answer(question: str, context: str) -> dict:
    """Generate answer using Ollama (Llama 3)."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3:8b",
                "system": SYSTEM_PROMPT.format(context=context),
                "prompt": question,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return {
            "answer": data["response"],
            "total_tokens": data.get("prompt_eval_count", 0)
            + data.get("eval_count", 0),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }


@router.post("/query", response_model=QueryResponse)
async def query_pipeline(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """Execute the full RAG pipeline: retrieve → fuse → rerank → generate → evaluate."""
    total_start = time.perf_counter()

    # --- Stage 1: Retrieval ---
    retrieval_start = time.perf_counter()

    if request.retrieval_mode == "dense":
        results = await dense_search(request.question, session, top_k=request.top_k * 4)
    elif request.retrieval_mode == "sparse":
        results = await sparse_search(request.question, session, top_k=request.top_k * 4)
    else:
        # Hybrid: dense + sparse + RRF fusion
        dense_results = await dense_search(
            request.question, session, top_k=request.top_k * 4
        )
        sparse_results = await sparse_search(
            request.question, session, top_k=request.top_k * 4
        )
        results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=request.top_k * 2,
        )

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    chunks_retrieved = len(results)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Ingest documents first.",
        )

    # --- Stage 2: Reranking ---
    rerank_start = time.perf_counter()
    reranked = rerank(request.question, results, top_k=request.top_k)
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    # --- Stage 3: Generation ---
    generation_start = time.perf_counter()
    context = "\n\n---\n\n".join(r.content for r in reranked)
    gen_result = await generate_answer(request.question, context)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    # --- Stage 4: Evaluation (optional) ---
    eval_start = time.perf_counter()
    eval_result = None

    if request.eval:
        context_chunks = [r.content for r in reranked]
        scores = await evaluate_response(
            request.question, gen_result["answer"], context_chunks
        )
        eval_result = EvaluationResult(
            faithfulness=round(scores.faithfulness, 3),
            relevance=round(scores.relevance, 3),
            answer_correctness=round(scores.answer_correctness, 3),
            hallucination_score=round(scores.hallucination_score, 3),
        )

    eval_ms = (time.perf_counter() - eval_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    # --- Log to database ---
    log = QueryLog(
        question=request.question,
        answer=gen_result["answer"],
        retrieved_chunk_ids=[r.chunk_id for r in results],
        reranked_chunk_ids=[r.chunk_id for r in reranked],
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        total_latency_ms=total_ms,
        total_tokens=gen_result["total_tokens"],
        prompt_tokens=gen_result["prompt_tokens"],
        completion_tokens=gen_result["completion_tokens"],
        faithfulness=eval_result.faithfulness if eval_result else None,
        relevance=eval_result.relevance if eval_result else None,
        answer_correctness=eval_result.answer_correctness if eval_result else None,
        hallucination_score=eval_result.hallucination_score if eval_result else None,
    )
    session.add(log)
    await session.commit()

    # --- Build response ---
    sources = [
        SourceChunk(
            chunk_id=r.chunk_id,
            content=r.content[:500],
            score=round(r.score, 4),
            document_name=r.document_name,
            rerank_score=r.metadata.get("rerank_score"),
        )
        for r in reranked
    ]

    pipeline_metrics = PipelineMetrics(
        retrieval_latency_ms=round(retrieval_ms, 2),
        rerank_latency_ms=round(rerank_ms, 2),
        generation_latency_ms=round(generation_ms, 2),
        evaluation_latency_ms=round(eval_ms, 2),
        total_latency_ms=round(total_ms, 2),
        total_tokens=gen_result["total_tokens"],
        prompt_tokens=gen_result["prompt_tokens"],
        completion_tokens=gen_result["completion_tokens"],
        chunks_retrieved=chunks_retrieved,
        chunks_after_rerank=len(reranked),
    )

    return QueryResponse(
        answer=gen_result["answer"],
        sources=sources,
        evaluation=eval_result,
        pipeline_metrics=pipeline_metrics,
    )
