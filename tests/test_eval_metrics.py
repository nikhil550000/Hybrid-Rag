import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from metrics import (
    EvalSample,
    compute_citation_presence_rate,
    compute_retrieval_metrics,
)


def test_citation_presence_rate_counts_answered_samples_with_valid_citations():
    samples = [
        EvalSample(question="q1", ground_truth="gt", source="a.pdf", citations_valid=1),
        EvalSample(question="q2", ground_truth="gt", source="b.pdf", citations_valid=0),
        EvalSample(question="q3", ground_truth="gt", source="c.pdf", citations_valid=0, refused=True),
    ]

    assert compute_citation_presence_rate(samples) == 0.5


def test_retrieval_metrics_use_expected_source_rank_before_and_after_rerank():
    samples = [
        EvalSample(
            question="q1",
            ground_truth="gt",
            source="paper-a.pdf",
            pre_rerank_sources=["other.pdf", "paper-a.pdf", "x.pdf"],
            post_rerank_sources=["paper-a.pdf", "other.pdf"],
        ),
        EvalSample(
            question="q2",
            ground_truth="gt",
            source="paper-b.pdf",
            pre_rerank_sources=["other.pdf", "paper-b.pdf"],
            post_rerank_sources=["other.pdf"],
        ),
    ]

    metrics = compute_retrieval_metrics(samples)

    assert metrics["pre_rerank_source_recall_at_k"] == 1.0
    assert metrics["post_rerank_source_recall_at_k"] == 0.5
    assert metrics["pre_rerank_source_mrr"] == 0.5
    assert metrics["post_rerank_source_mrr"] == 0.5
    assert metrics["pre_rerank_source_ndcg"] == 0.6309
    assert metrics["post_rerank_source_ndcg"] == 0.5
    assert metrics["reranker_source_win_rate"] == 1.0
    assert metrics["post_rerank_hit_rate_by_paper"] == {
        "paper-a.pdf": 1.0,
        "paper-b.pdf": 0.0,
    }
