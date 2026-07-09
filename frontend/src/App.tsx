import { useState } from 'react';
import { Loader2, AlertCircle, Sparkles } from 'lucide-react';
import { SearchBar } from './components/SearchBar';
import { ResultCard } from './components/ResultCard';
import { CitationCard } from './components/CitationCard';
import type { QueryResponse, ErrorResponse } from './types';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ErrorResponse | null>(null);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();

      if (!res.ok) {
        // Handle structured error from FastAPI
        setError({
          error: data.detail?.error || 'SERVER_ERROR',
          message: data.detail?.message || 'An unexpected error occurred.',
        });
      } else {
        setResponse(data as QueryResponse);
      }
    } catch (err: any) {
      setError({
        error: 'NETWORK_ERROR',
        message: 'Could not connect to the backend server. Is FastAPI running?',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">RAG Scholar</h1>
        <p className="subtitle">Interactive Q&A over Machine Learning Research</p>
      </header>

      <SearchBar onSearch={handleSearch} isLoading={isLoading} />

      {isLoading && (
        <div className="loading-container">
          <Loader2 size={48} className="spinner" />
          <span className="loading-text">Synthesizing research...</span>
        </div>
      )}

      {error && (
        <div className="glass-panel error-state">
          <AlertCircle size={48} color="var(--error-text)" style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{error.error}</h2>
          <p className="error-text">{error.message}</p>
        </div>
      )}

      {response && !isLoading && (
        <div className="results-container">
          <ResultCard response={response} />

          {!response.refused && response.citations.length > 0 && (
            <div className="citations-section">
              <h3 className="citations-title">
                <Sparkles size={20} className="text-accent" />
                Sources
              </h3>
              <div className="citation-grid">
                {response.citations.map((citation, idx) => (
                  <CitationCard key={idx} citation={citation} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
