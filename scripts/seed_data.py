#!/usr/bin/env python3
"""Seed sample documents for demo purposes.

Downloads and ingests a few open-access documents to demonstrate
the pipeline with real content.
"""

import asyncio
import httpx

API_URL = "http://localhost:8000/api/v1"

SAMPLE_URLS = [
    {
        "url": "https://raw.githubusercontent.com/anthropics/courses/main/prompt_engineering_interactive_tutorial/README.md",
        "strategy": "markdown",
        "description": "Anthropic Prompt Engineering Tutorial",
    },
    {
        "url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "strategy": "semantic",
        "description": "Lilian Weng - LLM Powered Autonomous Agents",
    },
    {
        "url": "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "strategy": "recursive",
        "description": "Lilian Weng - Adversarial Attacks on LLMs",
    },
]


async def seed():
    print("🌱 Seeding sample documents...\n")

    async with httpx.AsyncClient(timeout=120) as client:
        for doc in SAMPLE_URLS:
            print(f"  📄 {doc['description']}...")
            try:
                response = await client.post(
                    f"{API_URL}/ingest/url",
                    json={
                        "url": doc["url"],
                        "chunking_strategy": doc["strategy"],
                    },
                )
                if response.status_code == 409:
                    print(f"     ⏩ Already ingested, skipping.")
                elif response.status_code == 200:
                    data = response.json()
                    print(
                        f"     ✅ {data['total_chunks']} chunks "
                        f"({data['processing_time_ms']:.0f}ms)"
                    )
                else:
                    print(f"     ❌ {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"     ❌ Error: {e}")

    print("\n🎉 Seeding complete! Try querying:")
    print('   curl -X POST http://localhost:8000/api/v1/query \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"question": "What are autonomous agents?"}\'')


if __name__ == "__main__":
    asyncio.run(seed())
