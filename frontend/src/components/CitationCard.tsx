import { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import type { Citation } from '../types';

interface CitationCardProps {
  citation: Citation;
  index: number;
  targetId: string;
}

export function CitationCard({ citation, index, targetId }: CitationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const sourceName = citation.source.split(/[\\/]/).pop() || citation.source;

  return (
    <article id={targetId} className="citation-card">
      <header className="citation-header">
        <span className="citation-number" aria-label={`Source ${index}`}>{index}</span>
        <div className="citation-source">
          <BookOpen size={16} aria-hidden="true" />
          <span title={citation.source}>{sourceName}</span>
        </div>
        <span className="citation-page">
          <FileText size={14} aria-hidden="true" />
          Page {citation.page}
        </span>
      </header>

      <p className={`citation-passage ${isExpanded ? 'expanded' : ''}`}>
        {citation.passage}
      </p>

      <footer className="citation-footer">
        <button type="button" onClick={() => setIsExpanded((expanded) => !expanded)}>
          {isExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          {isExpanded ? 'Show less' : 'Read passage'}
        </button>
        <code title={citation.chunk_id}>{citation.chunk_id}</code>
      </footer>
    </article>
  );
}
