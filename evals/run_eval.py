"""Evaluation runner — faithfulness gate for CI/CD.

Usage:
    python evals/run_eval.py

Exit codes:
    0 — PASSED (faithfulness >= threshold)
    1 — FAILED (faithfulness < threshold)
"""
import json
import sys
from pathlib import Path

# Add src/ to Python path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import load_settings
from utils.logger import setup_logging, new_correlation_id, get_logger
from utils.helpers import handle_exceptions
from llm.client import get_llm_client
from llm.embeddings import SentenceTransformerEmbedder
from store.vector import VectorStore
from store.bm25 import BM25Store
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.prompt import PromptBuilder
from generation.citations import CitationValidator
from generation.generator import Generator
from pipeline.query import QueryPipeline

# Import from evals/ (sibling directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import EvalSample, EvalReport, run_deepeval_evaluation, compute_citation_accuracy

logger = get_logger(__name__)


def build_pipeline(settings):
    """Construct the full query pipeline from settings."""
    embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
    )
    bm25_store = BM25Store()
    bm25_store.load(Path(settings.bm25_index_path))

    dense = DenseRetriever(vector_store, embedder, top_k=settings.retrieval_top_k)
    sparse = SparseRetriever(bm25_store, top_k=settings.retrieval_top_k)
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model, top_n=settings.reranked_top_n
    )

    llm = get_llm_client(settings.llm_provider, settings.llm_model, settings.llm_temperature)
    prompt_builder = PromptBuilder(prompt_version=settings.prompt_version)
    generator = Generator(llm, prompt_builder, CitationValidator())

    return QueryPipeline(dense, sparse, reranker, generator, rrf_k=settings.rrf_k)


@handle_exceptions
def main():
    settings = load_settings()
    log_file = setup_logging(log_level=settings.log_level)
    cid = new_correlation_id()
    logger.info(f"Eval started | correlation_id={cid} | log_file={log_file}")

    # Load golden dataset
    dataset_path = Path(settings.eval_dataset_path)
    with open(dataset_path) as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} evaluation questions from {dataset_path}")

    # Build pipeline
    pipeline = build_pipeline(settings)

    # Run each question through the pipeline
    samples: list[EvalSample] = []
    for i, item in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] {item['question'][:60]}...")

        result = pipeline.run(item["question"])

        sample = EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            source=item["source"],
            generated_answer=result.answer,
            retrieved_contexts=[c.passage for c in result.citations],
            citations_valid=len(result.citations),
            refused=result.refused,
            latency_ms=result.latency_ms,
        )
        samples.append(sample)

    # Compute simple citation accuracy
    citation_acc = compute_citation_accuracy(samples)
    logger.info(f"Citation accuracy: {citation_acc:.4f}")

    # Run DeepEval LLM-as-judge evaluation
    logger.info("Running DeepEval LLM-as-judge evaluation...")
    judge_scores = run_deepeval_evaluation(
        samples=samples,
        provider=settings.llm_provider,
        model=settings.llm_model,
    )

    # Aggregate stats
    answered_samples = [s for s in samples if not s.refused]

    report = EvalReport(
        total_questions=len(samples),
        answered=len(answered_samples),
        refused=len(samples) - len(answered_samples),
        avg_latency_ms=round(
            sum(s.latency_ms for s in samples) / len(samples), 1
        ) if samples else 0.0,
        avg_citations_per_answer=round(
            sum(s.citations_valid for s in answered_samples) / len(answered_samples), 2
        ) if answered_samples else 0.0,
        faithfulness_score=judge_scores["faithfulness"],
        answer_relevancy_score=judge_scores["answer_relevancy"],
        context_recall_score=judge_scores["context_recall"],
        citation_accuracy=round(citation_acc, 4),
        samples=[{
            "question": s.question,
            "ground_truth": s.ground_truth,
            "source": s.source,
            "generated_answer": s.generated_answer[:500],
            "citations_valid": s.citations_valid,
            "refused": s.refused,
            "latency_ms": s.latency_ms,
        } for s in samples],
    )

    # Write report
    report_path = report.write()
    logger.info(f"Eval report written to {report_path}")

    # Display results
    print(f"\n{'='*60}")
    print(f"  RAG Scholar Evaluation Report")
    print(f"{'='*60}")
    print(f"  Total questions:     {report.total_questions}")
    print(f"  Answered:            {report.answered}")
    print(f"  Refused:             {report.refused}")
    print(f"  Avg latency:         {report.avg_latency_ms}ms")
    print(f"  Avg citations/ans:   {report.avg_citations_per_answer}")
    print(f"{'─'*60}")
    print(f"  Citation accuracy:   {report.citation_accuracy:.4f}")
    print(f"  Faithfulness:        {report.faithfulness_score:.4f}")
    print(f"  Answer relevancy:    {report.answer_relevancy_score:.4f}")
    print(f"  Context recall:      {report.context_recall_score:.4f}")
    print(f"{'─'*60}")
    print(f"  Report saved:        {report_path}")
    print(f"{'='*60}")

    # Gate on faithfulness threshold
    threshold = settings.faithfulness_threshold
    if report.faithfulness_score < threshold:
        print(f"\n❌ EVAL FAILED: faithfulness {report.faithfulness_score:.3f} < threshold {threshold}")
        sys.exit(1)
    else:
        print(f"\n✅ EVAL PASSED: faithfulness {report.faithfulness_score:.3f} >= threshold {threshold}")
        sys.exit(0)


if __name__ == "__main__":
    main()
