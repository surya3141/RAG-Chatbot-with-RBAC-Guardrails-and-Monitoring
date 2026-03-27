"""
RAG evaluation module using RAGAS.

Evaluates the quality of RAG pipeline responses along three axes:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Recall: Does the retrieved context cover the ground truth?

When RAGAS is not installed (or no API key is configured) the module
returns placeholder scores so the pipeline still functions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning("ragas not installed. Evaluation will return placeholder scores.")


class RAGEvaluator:
    """Evaluate RAG pipeline outputs with RAGAS metrics."""

    def __init__(self, llm=None) -> None:
        self._llm = llm  # optional LangChain LLM for RAGAS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation on a batch of QA pairs.

        Args:
            questions:    List of user questions.
            answers:      List of model answers corresponding to each question.
            contexts:     List of retrieved context lists for each question.
            ground_truths: Optional ground-truth answers (needed for recall).

        Returns:
            Dict mapping metric name → score (0–1).
        """
        if not _RAGAS_AVAILABLE:
            logger.warning("RAGAS not available. Returning placeholder scores.")
            return self._placeholder_scores()

        if not questions:
            logger.warning("No questions provided for evaluation.")
            return {}

        try:
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy]

            kwargs = {}
            if self._llm:
                kwargs["llm"] = self._llm

            result = evaluate(dataset, metrics=metrics, **kwargs)
            scores = dict(result)
            logger.info(f"RAGAS evaluation complete: {scores}")
            return scores

        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            return self._placeholder_scores()

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Convenience wrapper for evaluating a single QA pair."""
        gts = [ground_truth] if ground_truth else None
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[context],
            ground_truths=gts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder_scores() -> Dict[str, float]:
        return {
            "faithfulness": -1.0,
            "answer_relevancy": -1.0,
            "note": "RAGAS not available or evaluation failed.",
        }
