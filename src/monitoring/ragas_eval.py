"""RAGAS-based evaluation of RAG pipeline quality."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available — evaluation will return dummy metrics.")


class RAGASEvaluator:
    """Evaluate RAG pipeline quality using RAGAS metrics."""

    def __init__(self) -> None:
        if not _RAGAS_AVAILABLE:
            logger.warning("Initialising RAGASEvaluator without RAGAS library.")

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation and return a metrics dictionary.

        Parameters
        ----------
        questions:
            The input questions posed to the RAG system.
        answers:
            The generated answers to evaluate.
        contexts:
            For each question, the list of retrieved context strings.
        ground_truths:
            Optional reference answers (used by some metrics).
        """
        if not _RAGAS_AVAILABLE:
            logger.warning("Returning stub metrics — install ragas to enable evaluation.")
            return self._stub_metrics(len(questions))

        if not questions:
            logger.warning("No questions provided for evaluation.")
            return {}

        data: Dict[str, List] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        try:
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            result = evaluate(dataset=dataset, metrics=metrics)
            scores = {
                "faithfulness": round(float(result["faithfulness"]), 4),
                "answer_relevancy": round(float(result["answer_relevancy"]), 4),
                "context_precision": round(float(result["context_precision"]), 4),
            }
            logger.info("RAGAS evaluation completed: %s", scores)
            return scores
        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)
            return {"error": str(exc)}

    def generate_report(self, eval_results: Dict[str, Any]) -> str:
        """Convert *eval_results* into a human-readable plain-text report."""
        if not eval_results:
            return "No evaluation results available."

        if "error" in eval_results:
            return f"Evaluation failed: {eval_results['error']}"

        lines = ["=" * 50, "RAG Pipeline Evaluation Report", "=" * 50]
        metric_descriptions = {
            "faithfulness": "Faithfulness     (factual consistency with context)",
            "answer_relevancy": "Answer Relevancy (relevance to the question)",
            "context_precision": "Context Precision (quality of retrieved context)",
        }
        for key, description in metric_descriptions.items():
            value = eval_results.get(key)
            if value is not None:
                bar = self._score_bar(float(value))
                lines.append(f"  {description}: {value:.4f}  {bar}")

        lines.append("=" * 50)
        avg = (
            sum(v for k, v in eval_results.items() if isinstance(v, float))
            / max(1, len([v for v in eval_results.values() if isinstance(v, float)]))
        )
        lines.append(f"  Overall average score: {avg:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    @staticmethod
    def _stub_metrics(n_questions: int) -> Dict[str, Any]:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "note": "RAGAS not installed — these are placeholder values.",
            "num_questions": n_questions,
        }
