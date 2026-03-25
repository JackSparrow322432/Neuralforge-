"""RAG Evaluation Engine.

Automatically evaluates every query response across multiple dimensions.
This is the core differentiator — most RAG projects skip evaluation entirely.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Relevance: Are the retrieved chunks relevant to the question?
- Answer Correctness: Does the answer properly address the question?
- Hallucination Score: Are there claims not supported by context?

Uses LLM-as-judge pattern with structured output for consistent scoring.
"""

import json
import re
from dataclasses import dataclass

import httpx

OLLAMA_URL = "http://localhost:11434"


@dataclass
class EvalScores:
    faithfulness: float
    relevance: float
    answer_correctness: float
    hallucination_score: float


async def _llm_judge(prompt: str) -> str:
    """Call Ollama for LLM-as-judge evaluation."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 500},
            },
        )
        response.raise_for_status()
        return response.json()["response"]


def _parse_score(text: str, key: str) -> float:
    """Extract a numeric score from LLM output."""
    patterns = [
        rf'"{key}"\s*:\s*(\d+\.?\d*)',
        rf"{key}\s*:\s*(\d+\.?\d*)",
        rf"{key}\s*=\s*(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    return 0.5  # Default if parsing fails


async def evaluate_faithfulness(
    question: str, answer: str, context: str
) -> float:
    """Evaluate whether the answer is grounded in the retrieved context.

    Score: 1.0 = fully grounded, 0.0 = completely hallucinated.
    """
    prompt = f"""You are an expert evaluator for RAG systems.

TASK: Rate how faithfully the answer is supported by the given context.
A faithful answer ONLY contains information present in the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Rate faithfulness from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: The answer is entirely unsupported by the context

Respond ONLY with JSON: {{"faithfulness": <score>}}"""

    result = await _llm_judge(prompt)
    return _parse_score(result, "faithfulness")


async def evaluate_relevance(question: str, context: str) -> float:
    """Evaluate whether the retrieved chunks are relevant to the question.

    Score: 1.0 = all chunks highly relevant, 0.0 = no relevant chunks.
    """
    prompt = f"""You are an expert evaluator for RAG systems.

TASK: Rate how relevant the retrieved context is to answering the question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

Rate relevance from 0.0 to 1.0:
- 1.0: The context contains all information needed to answer the question
- 0.5: The context is partially relevant
- 0.0: The context is completely irrelevant to the question

Respond ONLY with JSON: {{"relevance": <score>}}"""

    result = await _llm_judge(prompt)
    return _parse_score(result, "relevance")


async def evaluate_correctness(question: str, answer: str) -> float:
    """Evaluate whether the answer correctly addresses the question.

    Score: 1.0 = comprehensive correct answer, 0.0 = wrong or no answer.
    """
    prompt = f"""You are an expert evaluator for RAG systems.

TASK: Rate how correctly and completely the answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate correctness from 0.0 to 1.0:
- 1.0: The answer fully and correctly addresses the question
- 0.5: The answer is partially correct or incomplete
- 0.0: The answer is wrong or doesn't address the question

Respond ONLY with JSON: {{"correctness": <score>}}"""

    result = await _llm_judge(prompt)
    return _parse_score(result, "correctness")


async def evaluate_hallucination(
    answer: str, context: str
) -> float:
    """Detect hallucinated claims not supported by context.

    Score: 0.0 = no hallucination (best), 1.0 = fully hallucinated (worst).
    """
    prompt = f"""You are a hallucination detection expert.

TASK: Identify what fraction of claims in the answer are NOT supported
by the context. A hallucination is any specific claim, fact, number,
or detail in the answer that cannot be found in the context.

CONTEXT:
{context}

ANSWER: {answer}

Rate hallucination from 0.0 to 1.0:
- 0.0: No hallucinations — every claim is in the context
- 0.5: About half the claims are hallucinated
- 1.0: The entire answer is hallucinated

Respond ONLY with JSON: {{"hallucination": <score>}}"""

    result = await _llm_judge(prompt)
    return _parse_score(result, "hallucination")


async def evaluate_response(
    question: str,
    answer: str,
    context_chunks: list[str],
) -> EvalScores:
    """Run full evaluation suite on a RAG response.

    Executes all four evaluation metrics concurrently for efficiency.
    """
    import asyncio

    context = "\n---\n".join(context_chunks)

    faithfulness, relevance, correctness, hallucination = await asyncio.gather(
        evaluate_faithfulness(question, answer, context),
        evaluate_relevance(question, context),
        evaluate_correctness(question, answer),
        evaluate_hallucination(answer, context),
    )

    return EvalScores(
        faithfulness=faithfulness,
        relevance=relevance,
        answer_correctness=correctness,
        hallucination_score=hallucination,
    )
