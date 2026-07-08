"""In-memory metrics collector for request-level observability.

Implements: HLD 3.10
Satisfies: FR-29 (P50/P95 latency), FR-30 (cost-per-request),
           FR-31 (citation coverage), FR-32 (failure rate)
"""
import statistics
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsSummary:
    """Computed summary statistics across all recorded requests."""
    total_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    avg_latency_ms: float
    total_cost_usd: float
    avg_cost_usd: float
    citation_coverage_pct: float  # % of responses with ≥1 citation
    failure_rate_pct: float       # % of responses that were refused/errored


class MetricsCollector:
    """In-memory metrics store. Tracks per-request stats and computes summaries.

    Thread-safe enough for single-process use (list.append is atomic in CPython).
    Written to logs/ and reported in eval output.
    """

    def __init__(self):
        self._latencies: list[float] = []
        self._costs: list[float] = []
        self._has_citations: list[bool] = []
        self._failed: list[bool] = []

    def record_request(
        self,
        latency_ms: float,
        cost_usd: float,
        has_citations: bool,
        failed: bool,
    ) -> None:
        """Record metrics for a single completed request."""
        self._latencies.append(latency_ms)
        self._costs.append(cost_usd)
        self._has_citations.append(has_citations)
        self._failed.append(failed)

        logger.debug(
            f"Metrics recorded: latency={latency_ms:.1f}ms, "
            f"cost=${cost_usd:.6f}, citations={has_citations}, failed={failed}"
        )

    def get_summary(self) -> MetricsSummary:
        """Compute P50, P95 latency and aggregate stats over all recorded requests.

        Returns MetricsSummary with zeros if no requests have been recorded.
        """
        n = len(self._latencies)

        if n == 0:
            return MetricsSummary(
                total_requests=0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                avg_latency_ms=0.0,
                total_cost_usd=0.0,
                avg_cost_usd=0.0,
                citation_coverage_pct=0.0,
                failure_rate_pct=0.0,
            )

        # P50 and P95 via statistics.quantiles (requires n >= 2)
        if n >= 2:
            quantiles = statistics.quantiles(self._latencies, n=100)
            p50 = quantiles[49]  # 50th percentile
            p95 = quantiles[94]  # 95th percentile
        else:
            p50 = self._latencies[0]
            p95 = self._latencies[0]

        total_cost = sum(self._costs)
        cited_count = sum(1 for c in self._has_citations if c)
        failed_count = sum(1 for f in self._failed if f)

        return MetricsSummary(
            total_requests=n,
            p50_latency_ms=round(p50, 1),
            p95_latency_ms=round(p95, 1),
            avg_latency_ms=round(statistics.mean(self._latencies), 1),
            total_cost_usd=round(total_cost, 6),
            avg_cost_usd=round(total_cost / n, 6),
            citation_coverage_pct=round((cited_count / n) * 100, 1),
            failure_rate_pct=round((failed_count / n) * 100, 1),
        )

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._latencies.clear()
        self._costs.clear()
        self._has_citations.clear()
        self._failed.clear()
