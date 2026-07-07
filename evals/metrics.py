"""Evaluation metrics — Ragas faithfulness + citation accuracy.

Satisfies: FR-27, FR-28
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.logger import get_logger

logger = get_logger(__name__)


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


def get_ragas_llm(provider: str, model: str):
    """Build a LangChain LLM wrapper for Ragas evaluation."""
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set — required for Ragas eval")
        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.0)
    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set — required for Ragas eval")
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.0)
    else:
        raise ValueError(f"Unsupported Ragas eval provider: {provider}")
    return LangchainLLMWrapper(llm)


def run_ragas_evaluation(
    samples: list[EvalSample],
    provider: str,
    model: str,
) -> dict[str, float]:
    """
    Run Ragas faithfulness + answer_relevancy + context_recall.
    Only evaluates non-refused samples that have retrieved contexts.

    Uses the same LLM provider configured in settings.yaml as the evaluator judge.
    """
    answered = [s for s in samples if not s.refused and s.retrieved_contexts]

    if not answered:
        logger.warning("No answered samples with contexts — skipping Ragas eval")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

    ragas_samples = []
    for s in answered:
        ragas_samples.append(SingleTurnSample(
            user_input=s.question,
            response=s.generated_answer,
            retrieved_contexts=s.retrieved_contexts,
            reference=s.ground_truth,
        ))

    dataset = EvaluationDataset(samples=ragas_samples)
    evaluator_llm = get_ragas_llm(provider, model)

    logger.info(f"Running Ragas evaluation on {len(ragas_samples)} samples...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=evaluator_llm,
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_recall": round(result["context_recall"], 4),
    }
    logger.info(f"Ragas scores: {scores}")
    return scores


def compute_citation_accuracy(samples: list[EvalSample]) -> float:
    """Simple metric: % of answered questions with ≥1 valid citation."""
    answered = [s for s in samples if not s.refused]
    if not answered:
        return 0.0
    return sum(1 for s in answered if s.citations_valid > 0) / len(answered)
