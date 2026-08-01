import { CheckCircle2, Clock3, GitBranch, RefreshCw } from 'lucide-react';
import type { QueryResponse } from '../types';

interface PipelineDetailsProps {
  response: QueryResponse;
}

const STAGE_LABELS: Record<string, string> = {
  query_routing: 'Routing',
  query_rewriting: 'Query rewrite',
  query_embedding: 'Embedding',
  dense_retrieval: 'Dense retrieval',
  sparse_retrieval: 'BM25 retrieval',
  rrf: 'RRF fusion',
  reranking: 'Reranking',
  prompt_building: 'Prompt build',
  llm_call: 'LLM generation',
  citation_validation: 'Citation check',
};

const STAGE_ORDER = Object.keys(STAGE_LABELS);

const ROUTE_LABELS: Record<string, string> = {
  GREETING: 'Greeting · direct',
  APP_HELP: 'App help · direct',
  OUT_OF_SCOPE: 'Out of scope · direct',
  BASIC_NON_RAG: 'Basic query · direct',
  RAG_FACTUAL: 'Factual RAG',
  RAG_SUMMARY: 'Summary RAG',
  RAG_EXACT_KEYWORD_TABLE: 'Exact / table RAG',
  FOLLOW_UP: 'Follow-up RAG',
};

export function PipelineDetails({ response }: PipelineDetailsProps) {
  const timings = Object.entries(response.timings_ms)
    .filter(([name, value]) => name !== 'total_latency' && Number.isFinite(value))
    .sort(([left], [right]) => {
      const leftIndex = STAGE_ORDER.indexOf(left);
      const rightIndex = STAGE_ORDER.indexOf(right);
      return (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex)
        - (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex);
    });
  const maxTiming = Math.max(response.latency_ms, ...timings.map(([, value]) => value), 1);
  const hasParallelRetrieval = response.timings_ms.dense_retrieval !== undefined
    && response.timings_ms.sparse_retrieval !== undefined;

  return (
    <aside className="pipeline-panel" aria-label="Pipeline details">
      <div className="pipeline-title">
        <GitBranch size={17} aria-hidden="true" />
        <strong>Pipeline</strong>
      </div>

      <div className="route-summary">
        <CheckCircle2 size={15} aria-hidden="true" />
        <span>{ROUTE_LABELS[response.route] ?? response.route.replaceAll('_', ' ')}</span>
      </div>

      {response.query_rewritten && response.retrieval_query && (
        <div className="rewrite-summary">
          <span><RefreshCw size={14} aria-hidden="true" /> Rewritten retrieval query</span>
          <q>{response.retrieval_query}</q>
        </div>
      )}

      <div className="timing-list">
        {timings.map(([name, value]) => (
          <div className="timing-row" key={name}>
            <div className="timing-label">
              <span>{STAGE_LABELS[name] ?? name.replaceAll('_', ' ')}</span>
              <span>{value.toFixed(1)} ms</span>
            </div>
            <div className="timing-track" aria-hidden="true">
              <span style={{ width: `${Math.max((value / maxTiming) * 100, 2)}%` }} />
            </div>
          </div>
        ))}
      </div>

      {hasParallelRetrieval && (
        <p className="parallel-note">Dense and BM25 retrieval overlap; stage times are not additive.</p>
      )}

      <div className="total-timing">
        <Clock3 size={15} aria-hidden="true" />
        <span>Total</span>
        <strong>{response.latency_ms.toFixed(0)} ms</strong>
      </div>
    </aside>
  );
}
