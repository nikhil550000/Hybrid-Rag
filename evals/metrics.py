"""Evaluation metrics — DeepEval LLM-as-judge + citation accuracy.

Satisfies: FR-27, FR-28

Uses DeepEval with AnthropicModel as the evaluation judge to compute:
  - Faithfulness: is the answer grounded in the retrieved contexts?
  - Answer Relevancy: does the answer actually address the question?
  - Context Recall: do the retrieved contexts contain the ground truth?
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from deepeval.models import AnthropicModel
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One evaluation sample with RAG output and ground truth."""
    question: str
    ground_truth: str
    source: str
    generated_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    citations_valid: int = 0
    citations_invalid: int = 0
    refused: bool = False
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation results."""
    total_questions: int = 0
    answered: int = 0
    refused: int = 0
    avg_latency_ms: float = 0.0
    avg_citations_per_answer: float = 0.0
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_recall_score: float = 0.0
    citation_accuracy: float = 0.0
    samples: list[dict] = field(default_factory=list)

    def write(self, output_dir: Path = Path("evals")) -> Path:
        """Write report to evals/report_{timestamp}.json."""
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"report_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


# ─── DeepEval LLM-as-judge evaluation ────────────────────────────────────────

def _build_judge_model(provider: str, model: str) -> AnthropicModel:
    """Build the DeepEval judge model from settings."""
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set — required for evaluation")
        return AnthropicModel(model=model, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported eval provider: {provider}. "
            f"DeepEval supports: anthropic (native), openai, etc."
        )


def run_deepeval_evaluation(
    samples: list[EvalSample],
    provider: str,
    model: str,
) -> dict[str, float]:
    """
    Run DeepEval faithfulness + answer_relevancy + contextual_recall.
    Only evaluates non-refused samples that have retrieved contexts.

    Uses AnthropicModel as the LLM judge.
    """
    answered = [s for s in samples if not s.refused and s.retrieved_contexts]

    if not answered:
        logger.warning("No answered samples with contexts — skipping evaluation")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

    judge_model = _build_judge_model(provider, model)

    # Initialize metrics with the Anthropic judge
    faithfulness = FaithfulnessMetric(model=judge_model, threshold=0.5, include_reason=True)
    relevancy = AnswerRelevancyMetric(model=judge_model, threshold=0.5, include_reason=True)
    recall = ContextualRecallMetric(model=judge_model, threshold=0.5, include_reason=True)

    faithfulness_scores = []
    relevancy_scores = []
    recall_scores = []

    for i, s in enumerate(answered):
        logger.info(f"Judging [{i+1}/{len(answered)}] {s.question[:50]}...")

        test_case = LLMTestCase(
            input=s.question,
            actual_output=s.generated_answer,
            retrieval_context=s.retrieved_contexts,
            expected_output=s.ground_truth,
        )

        # Measure each metric
        try:
            faithfulness.measure(test_case)
            faithfulness_scores.append(faithfulness.score)
            logger.info(f"  Faithfulness: {faithfulness.score:.2f} — {faithfulness.reason}")
        except Exception as e:
            logger.warning(f"  Faithfulness failed: {e}")
            faithfulness_scores.append(0.0)

        try:
            relevancy.measure(test_case)
            relevancy_scores.append(relevancy.score)
            logger.info(f"  Relevancy:    {relevancy.score:.2f} — {relevancy.reason}")
        except Exception as e:
            logger.warning(f"  Relevancy failed: {e}")
            relevancy_scores.append(0.0)

        try:
            recall.measure(test_case)
            recall_scores.append(recall.score)
            logger.info(f"  Recall:       {recall.score:.2f} — {recall.reason}")
        except Exception as e:
            logger.warning(f"  Recall failed: {e}")
            recall_scores.append(0.0)

    scores = {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 4),
        "context_recall": round(sum(recall_scores) / len(recall_scores), 4),
    }
    logger.info(f"DeepEval scores: {scores}")
    return scores


def compute_citation_accuracy(samples: list[EvalSample]) -> float:
    """Simple metric: % of answered questions with ≥1 valid citation."""
    answered = [s for s in samples if not s.refused]
    if not answered:
        return 0.0
    return sum(1 for s in answered if s.citations_valid > 0) / len(answered)
