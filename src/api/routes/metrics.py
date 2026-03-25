"""Metrics API routes for the observability dashboard."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import QueryLog, get_session
from src.models.schemas import MetricsSummary, MetricsTrend, TimeSeriesPoint

router = APIRouter()


@router.get("/metrics/summary", response_model=MetricsSummary)
async def get_metrics_summary(
    session: AsyncSession = Depends(get_session),
):
    """Aggregated evaluation metrics across all queries."""
    result = await session.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.avg(QueryLog.faithfulness).label("avg_faith"),
            func.avg(QueryLog.relevance).label("avg_rel"),
            func.avg(QueryLog.answer_correctness).label("avg_corr"),
            func.avg(QueryLog.hallucination_score).label("avg_hall"),
            func.avg(QueryLog.total_latency_ms).label("avg_lat"),
        )
    )
    row = result.one()

    # Percentile latencies
    latency_result = await session.execute(
        text("""
            SELECT
                percentile_cont(0.5) WITHIN GROUP (ORDER BY total_latency_ms) AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY total_latency_ms) AS p99
            FROM query_logs
        """)
    )
    lat_row = latency_result.one()

    return MetricsSummary(
        total_queries=row.total or 0,
        avg_faithfulness=round(float(row.avg_faith or 0), 3),
        avg_relevance=round(float(row.avg_rel or 0), 3),
        avg_correctness=round(float(row.avg_corr or 0), 3),
        avg_hallucination=round(float(row.avg_hall or 0), 3),
        avg_latency_ms=round(float(row.avg_lat or 0), 2),
        p50_latency_ms=round(float(lat_row.p50 or 0), 2),
        p95_latency_ms=round(float(lat_row.p95 or 0), 2),
        p99_latency_ms=round(float(lat_row.p99 or 0), 2),
    )


@router.get("/metrics/trends/{metric_name}", response_model=MetricsTrend)
async def get_metric_trend(
    metric_name: str,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
):
    """Time series data for a specific metric."""
    valid_metrics = {
        "faithfulness", "relevance", "answer_correctness",
        "hallucination_score", "total_latency_ms",
        "retrieval_latency_ms", "generation_latency_ms",
    }

    if metric_name not in valid_metrics:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Choose from: {valid_metrics}",
        )

    result = await session.execute(
        text(f"""
            SELECT created_at, {metric_name}
            FROM query_logs
            WHERE {metric_name} IS NOT NULL
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )

    rows = result.fetchall()

    return MetricsTrend(
        metric_name=metric_name,
        data_points=[
            TimeSeriesPoint(timestamp=row.created_at, value=float(row[1]))
            for row in reversed(rows)
        ],
    )


@router.get("/metrics/recent")
async def get_recent_queries(
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """Recent query logs with full metrics."""
    result = await session.execute(
        select(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "question": log.question,
            "answer": log.answer[:200] + "..." if len(log.answer) > 200 else log.answer,
            "faithfulness": log.faithfulness,
            "relevance": log.relevance,
            "correctness": log.answer_correctness,
            "hallucination": log.hallucination_score,
            "latency_ms": log.total_latency_ms,
            "tokens": log.total_tokens,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
