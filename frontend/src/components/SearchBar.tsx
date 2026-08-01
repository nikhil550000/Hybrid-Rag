import { useState, type FormEvent, type KeyboardEvent } from 'react';
import { LoaderCircle, Send } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void | Promise<void>;
  isLoading: boolean;
  disabled?: boolean;
  placeholder: string;
}

export function SearchBar({ onSearch, isLoading, disabled = false, placeholder }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const normalizedQuery = query.trim();
  const isDisabled = disabled || isLoading;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!normalizedQuery || isDisabled) return;

    void onSearch(normalizedQuery);
    setQuery('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="research-query">Research question</label>
      <textarea
        id="research-query"
        name="query"
        rows={1}
        maxLength={2000}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isDisabled}
        aria-describedby="query-limit"
      />
      <span id="query-limit" className={`query-limit ${query.length > 1800 ? 'visible' : ''}`}>
        {query.length}/2000
      </span>
      <button
        type="submit"
        className="send-button"
        disabled={!normalizedQuery || isDisabled}
        aria-label={isLoading ? 'Query in progress' : 'Send question'}
        title={isLoading ? 'Query in progress' : 'Send question'}
      >
        {isLoading
          ? <LoaderCircle size={19} className="spinner" aria-hidden="true" />
          : <Send size={19} aria-hidden="true" />}
      </button>
    </form>
  );
}
