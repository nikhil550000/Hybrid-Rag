import { Bot, Clock, DollarSign, ShieldAlert } from 'lucide-react';
import type { QueryResponse } from '../types';

interface ResultCardProps {
  response: QueryResponse;
}

export const ResultCard: React.FC<ResultCardProps> = ({ response }) => {
  if (response.refused) {
    return (
      <div className="glass-panel refusal-state">
        <ShieldAlert size={48} className="refusal-icon" />
        <h2 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Insufficient Context</h2>
        <p className="refusal-text">
          {response.answer || "I cannot answer this question based on the provided research papers."}
        </p>
      </div>
    );
  }

  return (
    <div className="glass-panel result-card">
      <div className="result-header">
        <Bot size={24} />
        <h2 style={{ margin: 0 }}>Answer</h2>
      </div>
      
      <div className="result-content">
        {response.answer}
      </div>

      <div className="metrics-bar">
        <div className="metric-item" title="Latency">
          <Clock size={16} />
          <span>{response.latency_ms.toFixed(0)} ms</span>
        </div>
        <div className="metric-item" title="Cost">
          <DollarSign size={16} />
          <span>${response.cost_usd.toFixed(6)}</span>
        </div>
      </div>
    </div>
  );
};
