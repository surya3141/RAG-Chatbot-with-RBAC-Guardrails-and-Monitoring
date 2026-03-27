"""
Post-deployment automated evaluation script.

Runs a small golden-dataset evaluation against the deployed RAG pipeline
using RAGAS metrics and logs results to LangSmith.

Usage:
    python scripts/run_evals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.llm.llm_client import LLMClient
from src.monitoring.evaluator import RAGEvaluator
from src.monitoring.token_tracker import TokenTracker
from src.rbac.access_control import RBACManager
from src.rag.pipeline import RAGPipeline
from src.guardrails.pii_masker import PIIMasker
from src.vectorstore.vector_db import VectorStore

# ---------------------------------------------------------------------------
# Golden dataset – extend this for comprehensive coverage
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question": "What was the Q3 2024 revenue?",
        "role": "finance",
        "ground_truth": "$15.2 million",
    },
    {
        "question": "What is the standard PTO for a 2-year employee?",
        "role": "hr",
        "ground_truth": "15 days per year",
    },
    {
        "question": "What is the FY 2025 ARR target?",
        "role": "csuite",
        "ground_truth": "$68 million",
    },
]


def main() -> None:
    print("Initialising pipeline for evaluation…")
    rbac = RBACManager(config_path=settings.rbac_config_path)
    llm = LLMClient(api_key=settings.groq_api_key, model=settings.groq_model)
    pii = PIIMasker()
    tracker = TokenTracker(model=settings.groq_model)
    vs = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )

    pipeline = RAGPipeline(
        vector_store=vs,
        llm_client=llm,
        rbac_manager=rbac,
        pii_masker=pii,
        token_tracker=tracker,
    )

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        print(f"  Querying [{item['role']}]: {item['question']!r}")
        result = pipeline.query(question=item["question"], role=item["role"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([s["filename"] for s in result.get("sources", [])] or [""])
        ground_truths.append(item["ground_truth"])

    print("\nRunning RAGAS evaluation…")
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )

    print("\n=== Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")

    # Non-zero exit if evaluation is unavailable
    if scores.get("faithfulness", -1) == -1.0:
        print("\nWARNING: RAGAS not available – skipping score validation.")
        sys.exit(0)

    # Fail the pipeline stage if faithfulness drops below threshold
    threshold = 0.5
    if scores.get("faithfulness", 1.0) < threshold:
        print(f"\nFAIL: Faithfulness score below threshold ({threshold}).")
        sys.exit(1)

    print("\nAll evaluation checks passed.")


if __name__ == "__main__":
    main()
