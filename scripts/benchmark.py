#!/usr/bin/env python3
"""Benchmark script — run evaluation on a curated Q&A dataset.

Usage:
    python scripts/benchmark.py --dataset data/benchmark_qa.json
    python scripts/benchmark.py --generate-report

Outputs evaluation metrics as a markdown table for README.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000/api/v1"

# Default benchmark dataset
DEFAULT_QA = [
    {
        "question": "What is retrieval-augmented generation?",
        "expected_keywords": ["retrieval", "generation", "context", "documents"],
    },
    {
        "question": "How does hybrid search combine dense and sparse retrieval?",
        "expected_keywords": ["dense", "sparse", "fusion", "BM25", "vector"],
    },
    {
        "question": "What metrics are used to evaluate RAG systems?",
        "expected_keywords": ["faithfulness", "relevance", "correctness"],
    },
]


async def run_benchmark(qa_pairs: list[dict]) -> list[dict]:
    """Run all Q&A pairs through the pipeline and collect metrics."""
    results = []

    async with httpx.AsyncClient(timeout=120) as client:
        for i, qa in enumerate(qa_pairs):
            print(f"  [{i+1}/{len(qa_pairs)}] {qa['question'][:60]}...")

            start = time.perf_counter()
            try:
                response = await client.post(
                    f"{API_URL}/query",
                    json={"question": qa["question"], "eval": True},
                )
                response.raise_for_status()
                data = response.json()

                results.append({
                    "question": qa["question"],
                    "answer_preview": data["answer"][:100],
                    "faithfulness": data["evaluation"]["faithfulness"],
                    "relevance": data["evaluation"]["relevance"],
                    "correctness": data["evaluation"]["answer_correctness"],
                    "hallucination": data["evaluation"]["hallucination_score"],
                    "latency_ms": data["pipeline_metrics"]["total_latency_ms"],
                    "tokens": data["pipeline_metrics"]["total_tokens"],
                    "status": "ok",
                })
            except Exception as e:
                results.append({
                    "question": qa["question"],
                    "status": "error",
                    "error": str(e),
                })

    return results


def generate_report(results: list[dict]) -> str:
    """Generate markdown report from benchmark results."""
    ok_results = [r for r in results if r["status"] == "ok"]

    if not ok_results:
        return "❌ No successful queries to report."

    metrics = {
        "Faithfulness": [r["faithfulness"] for r in ok_results],
        "Relevance": [r["relevance"] for r in ok_results],
        "Answer Correctness": [r["correctness"] for r in ok_results],
        "Hallucination Rate": [r["hallucination"] for r in ok_results],
    }

    latencies = [r["latency_ms"] for r in ok_results]

    report = "## Benchmark Results\n\n"
    report += f"**Queries:** {len(ok_results)} successful / {len(results)} total\n\n"
    report += "| Metric | Mean | Median | Min | Max |\n"
    report += "|--------|------|--------|-----|-----|\n"

    for name, values in metrics.items():
        report += (
            f"| {name} | {statistics.mean(values):.3f} "
            f"| {statistics.median(values):.3f} "
            f"| {min(values):.3f} | {max(values):.3f} |\n"
        )

    report += f"\n**Latency:** P50={statistics.median(latencies):.0f}ms "
    report += f"P95={sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms "
    report += f"Mean={statistics.mean(latencies):.0f}ms\n"

    report += f"\n**Avg Tokens:** {statistics.mean([r['tokens'] for r in ok_results]):.0f}\n"

    return report


async def main():
    parser = argparse.ArgumentParser(description="NeuralForge RAG Benchmark")
    parser.add_argument(
        "--dataset", type=str, help="Path to JSON file with Q&A pairs"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_report.md", help="Output report path"
    )
    args = parser.parse_args()

    if args.dataset:
        qa_pairs = json.loads(Path(args.dataset).read_text())
    else:
        qa_pairs = DEFAULT_QA

    print(f"🚀 Running benchmark with {len(qa_pairs)} queries...\n")
    results = await run_benchmark(qa_pairs)

    report = generate_report(results)
    print("\n" + report)

    Path(args.output).write_text(report)
    print(f"\n📄 Report saved to {args.output}")

    # Also save raw results
    raw_path = args.output.replace(".md", "_raw.json")
    Path(raw_path).write_text(json.dumps(results, indent=2))
    print(f"📊 Raw results saved to {raw_path}")


if __name__ == "__main__":
    asyncio.run(main())
