"""NeuralForge RAG — FastAPI Application Entrypoint."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import ingest, query, metrics
from src.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and models on startup."""
    await init_db()
    yield


app = FastAPI(
    title="NeuralForge RAG",
    description="Production-grade RAG pipeline with built-in evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_pipeline_metrics(request, call_next):
    """Track request latency for every endpoint."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Pipeline-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
