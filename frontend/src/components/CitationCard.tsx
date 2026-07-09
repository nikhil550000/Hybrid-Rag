import { BookOpen } from 'lucide-react';
import type { Citation } from '../types';

interface CitationCardProps {
  citation: Citation;
}

export const CitationCard: React.FC<CitationCardProps> = ({ citation }) => {
  return (
    <div className="glass-panel glass-panel-interactive citation-card">
      <div className="citation-source">
        <BookOpen size={16} />
        <span>{citation.source}</span>
      </div>
      <div className="citation-passage">
        "{citation.passage}"
      </div>
      <div className="citation-id">
        ID: {citation.chunk_id}
      </div>
    </div>
  );
};
