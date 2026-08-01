import type { ReactNode } from 'react';
import { Bot, CircleDollarSign, MessageCircle, ShieldAlert, Sparkles } from 'lucide-react';
import { CitationCard } from './CitationCard';
import { PipelineDetails } from './PipelineDetails';
import type { ChatTurn, Citation, QueryResponse } from '../types';

interface ResultCardProps {
  turn: ChatTurn;
}

const DIRECT_ROUTES = new Set(['GREETING', 'APP_HELP', 'OUT_OF_SCOPE', 'BASIC_NON_RAG']);
const CITATION_PATTERN = /\[\s*SOURCE\s*:\s*([^\]]+?)\s*\]/gi;

function refusalHeading(response: QueryResponse): string {
  if (response.route === 'OUT_OF_SCOPE') return 'Outside the indexed corpus';
  if (response.route === 'BASIC_NON_RAG') return 'A more specific question is needed';
  return 'Insufficient source context';
}

function AnswerText({ answer, citations, turnId }: { answer: string; citations: Citation[]; turnId: string }) {
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  const pattern = new RegExp(CITATION_PATTERN);

  while ((match = pattern.exec(answer)) !== null) {
    if (match.index > cursor) nodes.push(answer.slice(cursor, match.index));

    const citationIndex = citations.findIndex((citation) => citation.chunk_id === match?.[1].trim());
    if (citationIndex >= 0) {
      nodes.push(
        <a
          className="inline-citation"
          href={`#${turnId}-citation-${citationIndex + 1}`}
          key={`${match.index}-${citationIndex}`}
          aria-label={`View source ${citationIndex + 1}`}
        >
          {citationIndex + 1}
        </a>,
      );
    } else {
      nodes.push(match[0]);
    }
    cursor = pattern.lastIndex;
  }

  if (cursor < answer.length) nodes.push(answer.slice(cursor));
  return <>{nodes}</>;
}

export function ResultCard({ turn }: ResultCardProps) {
  const { response } = turn;
  const isDirect = DIRECT_ROUTES.has(response.route);
  const AnswerIcon = response.refused ? ShieldAlert : isDirect ? MessageCircle : Bot;

  return (
    <article className="conversation-turn">
      <div className="user-message">{turn.query}</div>

      <div className="assistant-grid">
        <div className="answer-panel">
          <header className="answer-header">
            <span className={`answer-icon ${response.refused ? 'refused' : ''}`}>
              <AnswerIcon size={18} aria-hidden="true" />
            </span>
            <div>
              <strong>
                {response.refused
                  ? refusalHeading(response)
                  : isDirect
                    ? 'Direct response'
                    : 'Grounded answer'}
              </strong>
              <span>
                {isDirect
                  ? 'Retrieval and generation bypassed'
                  : `${response.citations.length} ${response.citations.length === 1 ? 'source' : 'sources'} validated`}
              </span>
            </div>
            {response.query_rewritten && (
              <span className="rewrite-badge">Follow-up rewritten</span>
            )}
          </header>

          <div className={`answer-content ${response.refused ? 'refused' : ''}`}>
            <AnswerText answer={response.answer} citations={response.citations} turnId={turn.id} />
          </div>

          <div className="answer-metrics">
            <span><CircleDollarSign size={15} aria-hidden="true" /> ${response.cost_usd.toFixed(6)}</span>
            <span><Sparkles size={15} aria-hidden="true" /> {response.route.replaceAll('_', ' ')}</span>
          </div>
        </div>

        <PipelineDetails response={response} />
      </div>

      {!response.refused && response.citations.length > 0 && (
        <section className="citations-section" aria-labelledby={`${turn.id}-sources-title`}>
          <h3 id={`${turn.id}-sources-title`}>Sources</h3>
          <div className="citation-grid">
            {response.citations.map((citation, index) => (
              <CitationCard
                key={citation.chunk_id}
                citation={citation}
                index={index + 1}
                targetId={`${turn.id}-citation-${index + 1}`}
              />
            ))}
          </div>
        </section>
      )}
    </article>
  );
}
