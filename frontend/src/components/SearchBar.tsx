import { useState } from 'react';
import { Search, ArrowRight } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim());
    }
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-input-wrapper">
        <Search className="absolute left-6 text-slate-400" size={20} style={{ position: 'absolute', left: '1.5rem' }} />
        <input
          type="text"
          className="search-input"
          placeholder="Ask a question about the ML papers..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="search-button"
          disabled={!query.trim() || isLoading}
          aria-label="Submit search"
        >
          <ArrowRight size={20} />
        </button>
      </form>
    </div>
  );
};
