"""Evaluation metrics — DeepEval LLM-as-judge + retrieval/citation metrics.

Satisfies: FR-27, FR-28

Uses DeepEval with AnthropicModel as the evaluation judge to compute:
  - Faithfulness: is the answer grounded in the retrieved contexts?
  - Answer Relevancy: does the answer actually address the question?
  - Context Recall: do the retrieved contexts contain the ground truth?
"""
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from deepeval.models import AnthropicModel, GeminiModel
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
    pre_rerank_sources: list[str] = field(default_factory=list)
    post_rerank_sources: list[str] = field(default_factory=list)
    pre_rerank_chunk_ids: list[str] = field(default_factory=list)
    post_rerank_chunk_ids: list[str] = field(default_factory=list)
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
    citation_presence_rate: float = 0.0
    pre_rerank_source_recall_at_k: float = 0.0
    post_rerank_source_recall_at_k: float = 0.0
    pre_rerank_source_mrr: float = 0.0
    post_rerank_source_mrr: float = 0.0
    pre_rerank_source_ndcg: float = 0.0
    post_rerank_source_ndcg: float = 0.0
    reranker_source_win_rate: float = 0.0
    post_rerank_hit_rate_by_paper: dict[str, float] = field(default_factory=dict)
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

def _build_judge_model(provider: str, model: str):
    """Build the DeepEval judge model from settings.

    Supports:
        - anthropic: uses DeepEval's AnthropicModel
        - google: uses DeepEval's GeminiModel
    """
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set — required for evaluation")
        return AnthropicModel(model=model, api_key=api_key)
    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set — required for evaluation")
        return GeminiModel(model=model, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported eval provider: {provider}. "
            f"DeepEval supports: anthropic, google"
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


def _first_source_rank(sources: list[str], expected_source: str) -> int | None:
    """Return the one-indexed rank of the first chunk from the expected source."""
    for index, source in enumerate(sources, start=1):
        if source == expected_source:
            return index
    return None


def _mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / rank) if rank is not None else 0.0 for rank in ranks) / len(ranks)


def _mean_ndcg(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / math.log2(rank + 1)) if rank is not None else 0.0 for rank in ranks) / len(ranks)


def compute_citation_presence_rate(samples: list[EvalSample]) -> float:
    """Return the share of answered questions with at least one valid citation."""
    answered = [s for s in samples if not s.refused]
    if not answered:
        return 0.0
    return sum(1 for s in answered if s.citations_valid > 0) / len(answered)


def compute_retrieval_metrics(samples: list[EvalSample]) -> dict[str, float]:
    """Compute source-level retrieval metrics from the golden answer source."""
    evaluated = [s for s in samples if not s.refused]
    if not evaluated:
        return {
            "pre_rerank_source_recall_at_k": 0.0,
            "post_rerank_source_recall_at_k": 0.0,
            "pre_rerank_source_mrr": 0.0,
            "post_rerank_source_mrr": 0.0,
            "pre_rerank_source_ndcg": 0.0,
            "post_rerank_source_ndcg": 0.0,
            "reranker_source_win_rate": 0.0,
            "post_rerank_hit_rate_by_paper": {},
        }

    pre_ranks = [_first_source_rank(s.pre_rerank_sources, s.source) for s in evaluated]
    post_ranks = [_first_source_rank(s.post_rerank_sources, s.source) for s in evaluated]
    comparable = [
        (pre, post)
        for pre, post in zip(pre_ranks, post_ranks)
        if pre is not None and post is not None
    ]
    wins = sum(1 for pre, post in comparable if post < pre)
    per_paper_hits: dict[str, list[bool]] = {}
    for sample, rank in zip(evaluated, post_ranks):
        per_paper_hits.setdefault(sample.source, []).append(rank is not None)

    return {
        "pre_rerank_source_recall_at_k": round(sum(rank is not None for rank in pre_ranks) / len(pre_ranks), 4),
        "post_rerank_source_recall_at_k": round(sum(rank is not None for rank in post_ranks) / len(post_ranks), 4),
        "pre_rerank_source_mrr": round(_mean_reciprocal_rank(pre_ranks), 4),
        "post_rerank_source_mrr": round(_mean_reciprocal_rank(post_ranks), 4),
        "pre_rerank_source_ndcg": round(_mean_ndcg(pre_ranks), 4),
        "post_rerank_source_ndcg": round(_mean_ndcg(post_ranks), 4),
        "reranker_source_win_rate": round(wins / len(comparable), 4) if comparable else 0.0,
        "post_rerank_hit_rate_by_paper": {
            source: round(sum(hits) / len(hits), 4)
            for source, hits in sorted(per_paper_hits.items())
        },
    }
