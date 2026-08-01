import { useCallback, useEffect, useState } from 'react';
import {
  AlertCircle,
  BookOpen,
  Database,
  FileText,
  LibraryBig,
  LoaderCircle,
  MessageSquarePlus,
  RefreshCw,
  Server,
  Trash2,
  WifiOff,
  X,
} from 'lucide-react';
import {
  deleteDocumentSession,
  getHealth,
  submitQuery,
  toErrorResponse,
} from './api';
import { ResultCard } from './components/ResultCard';
import { SearchBar } from './components/SearchBar';
import { UploadZone } from './components/UploadZone';
import type {
  ConversationState,
  DataSource,
  ErrorResponse,
  HealthResponse,
  UploadResponse,
} from './types';
import './App.css';

type HealthState = 'checking' | 'ready' | 'not-ready' | 'unavailable';

const DEFAULT_EXAMPLES = [
  'What role does positional encoding play in the Transformer?',
  'Summarize the main contribution of the BERT paper.',
  'What accuracy is reported in table 2?',
];

const UPLOAD_EXAMPLES = [
  'Summarize the main argument of these documents.',
  'What limitations do the authors discuss?',
  'Compare the methods described in the uploaded papers.',
];

function emptyConversation(): ConversationState {
  return { conversationId: null, turns: [] };
}

function createTurnId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

interface ErrorNoticeProps {
  error: ErrorResponse;
  onDismiss: () => void;
}

function ErrorNotice({ error, onDismiss }: ErrorNoticeProps) {
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={20} aria-hidden="true" />
      <div className="error-notice-content">
        <strong>{error.error.replaceAll('_', ' ')}</strong>
        <span>{error.message}</span>
        {error.request_id && (
          <span className="request-id">
            Request ID: <code>{error.request_id}</code>
          </span>
        )}
      </div>
      <button className="icon-button" type="button" onClick={onDismiss} aria-label="Dismiss error">
        <X size={18} aria-hidden="true" />
      </button>
    </div>
  );
}

function App() {
  const [dataSource, setDataSource] = useState<DataSource>('default');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadSummary, setUploadSummary] = useState<UploadResponse | null>(null);
  const [conversations, setConversations] = useState<Record<DataSource, ConversationState>>({
    default: emptyConversation(),
    upload: emptyConversation(),
  });
  const [isQuerying, setIsQuerying] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isClearingSession, setIsClearingSession] = useState(false);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [error, setError] = useState<ErrorResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthState, setHealthState] = useState<HealthState>('checking');

  const refreshHealth = useCallback(async () => {
    setHealthState('checking');
    try {
      const nextHealth = await getHealth();
      setHealth(nextHealth);
      setHealthState(nextHealth.ready ? 'ready' : 'not-ready');
    } catch {
      setHealth(null);
      setHealthState('unavailable');
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    void getHealth()
      .then((nextHealth) => {
        if (cancelled) return;
        setHealth(nextHealth);
        setHealthState(nextHealth.ready ? 'ready' : 'not-ready');
      })
      .catch(() => {
        if (cancelled) return;
        setHealth(null);
        setHealthState('unavailable');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const activeConversation = conversations[dataSource];
  const isBusy = isQuerying || isUploading || isClearingSession;
  const isBackendReady = healthState === 'ready';
  const canQuery = dataSource === 'default' || sessionId !== null;
  const examples = dataSource === 'default' ? DEFAULT_EXAMPLES : UPLOAD_EXAMPLES;

  const handleSourceChange = (source: DataSource) => {
    if (isBusy || source === dataSource) return;
    setDataSource(source);
    setError(null);
  };

  const handleSearch = async (query: string) => {
    if (!isBackendReady || isBusy) return;

    const source = dataSource;
    const sourceConversation = conversations[source];
    const activeSessionId = source === 'upload' ? sessionId : null;
    if (source === 'upload' && !activeSessionId) return;

    setIsQuerying(true);
    setPendingQuery(query);
    setError(null);

    try {
      const response = await submitQuery({
        query,
        ...(activeSessionId ? { session_id: activeSessionId } : {}),
        ...(sourceConversation.conversationId
          ? { conversation_id: sourceConversation.conversationId }
          : {}),
      });

      setConversations((current) => {
        const conversation = current[source];
        return {
          ...current,
          [source]: {
            conversationId: response.conversation_id ?? conversation.conversationId,
            turns: [
              ...conversation.turns,
              { id: createTurnId(), query, response },
            ],
          },
        };
      });
    } catch (caughtError: unknown) {
      const apiError = toErrorResponse(caughtError);

      if (source === 'upload' && activeSessionId && apiError.status === 404) {
        setSessionId(null);
        setUploadSummary(null);
        setConversations((current) => ({
          ...current,
          upload: emptyConversation(),
        }));
        setError({
          ...apiError,
          error: 'SESSION_EXPIRED',
          message: 'This uploaded-document session is no longer available. Upload the PDFs again to continue.',
        });
      } else {
        setError(apiError);
      }
    } finally {
      setPendingQuery(null);
      setIsQuerying(false);
    }
  };

  const handleSessionCreated = (session: UploadResponse) => {
    setSessionId(session.session_id);
    setUploadSummary(session);
    setDataSource('upload');
    setConversations((current) => ({
      ...current,
      upload: emptyConversation(),
    }));
    setError(null);
  };

  const clearSession = async () => {
    if (!sessionId || isBusy) return;

    setIsClearingSession(true);
    setError(null);
    let shouldClearLocalSession = false;

    try {
      await deleteDocumentSession(sessionId);
      shouldClearLocalSession = true;
    } catch (caughtError: unknown) {
      const apiError = toErrorResponse(caughtError);
      if (apiError.status === 404) {
        shouldClearLocalSession = true;
      } else {
        setError(apiError);
      }
    } finally {
      setIsClearingSession(false);
    }

    if (shouldClearLocalSession) {
      setSessionId(null);
      setUploadSummary(null);
      setConversations((current) => ({
        ...current,
        upload: emptyConversation(),
      }));
    }
  };

  const startNewConversation = () => {
    if (isBusy) return;
    setConversations((current) => ({
      ...current,
      [dataSource]: emptyConversation(),
    }));
    setError(null);
  };

  const healthLabel = {
    checking: 'Checking API',
    ready: 'API ready',
    'not-ready': 'API initializing',
    unavailable: 'API unavailable',
  }[healthState];

  const HealthIcon = healthState === 'ready'
    ? Server
    : healthState === 'unavailable'
      ? WifiOff
      : RefreshCw;

  const failedHealthChecks = health
    ? Object.entries(health.checks)
      .filter(([, passed]) => !passed)
      .map(([name]) => name.replaceAll('_', ' '))
    : [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <BookOpen size={21} />
          </div>
          <div>
            <span className="brand-name">RAG Scholar</span>
            <span className="brand-context">Research workspace</span>
          </div>
        </div>

        <button
          type="button"
          className={`service-status service-status-${healthState}`}
          onClick={() => void refreshHealth()}
          disabled={healthState === 'checking'}
          aria-label={`${healthLabel}. Refresh backend status.`}
          title="Refresh backend status"
        >
          <HealthIcon
            size={15}
            className={healthState === 'checking' ? 'spinner' : undefined}
            aria-hidden="true"
          />
          <span>{healthLabel}</span>
        </button>
      </header>

      <main className="workspace">
        <section className="workspace-heading" aria-labelledby="workspace-title">
          <div>
            <p className="eyebrow">Grounded research assistant</p>
            <h1 id="workspace-title">Research paper Q&amp;A</h1>
          </div>
          {health && (
            <div className="corpus-count" title="Indexed chunks in the default corpus">
              <Database size={16} aria-hidden="true" />
              <span>{health.chunks_indexed.toLocaleString()} chunks indexed</span>
            </div>
          )}
        </section>

        <div className="source-selector" aria-label="Knowledge source">
          <button
            type="button"
            aria-pressed={dataSource === 'default'}
            className={dataSource === 'default' ? 'active' : undefined}
            onClick={() => handleSourceChange('default')}
            disabled={isBusy}
          >
            <LibraryBig size={17} aria-hidden="true" />
            <span>Default corpus</span>
            <span className="source-count">10 papers</span>
          </button>
          <button
            type="button"
            aria-pressed={dataSource === 'upload'}
            className={dataSource === 'upload' ? 'active' : undefined}
            onClick={() => handleSourceChange('upload')}
            disabled={isBusy}
          >
            <FileText size={17} aria-hidden="true" />
            <span>Custom uploads</span>
            {sessionId && <span className="session-dot" aria-label="Session ready" />}
          </button>
        </div>

        {healthState !== 'ready' && (
          <div className={`backend-notice backend-notice-${healthState}`} role="status">
            {healthState === 'unavailable' ? <WifiOff size={20} /> : <LoaderCircle size={20} className="spinner" />}
            <div>
              <strong>{healthLabel}</strong>
              <span>
                {healthState === 'unavailable'
                  ? 'Queries and uploads are paused until the backend can be reached.'
                  : failedHealthChecks.length > 0
                    ? `Waiting for: ${failedHealthChecks.join(', ')}.`
                    : 'Waiting for backend components to finish initializing.'}
              </span>
            </div>
            {healthState !== 'checking' && (
              <button type="button" className="secondary-button" onClick={() => void refreshHealth()}>
                <RefreshCw size={15} aria-hidden="true" />
                Retry
              </button>
            )}
          </div>
        )}

        {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}

        {dataSource === 'upload' && sessionId && (
          <section className="session-strip" aria-label="Active uploaded-document session">
            <div className="session-summary">
              <span className="session-icon"><Database size={18} aria-hidden="true" /></span>
              <div>
                <strong>Custom corpus ready</strong>
                <span>
                  {uploadSummary
                    ? `${uploadSummary.files_processed} ${uploadSummary.files_processed === 1 ? 'file' : 'files'} · ${uploadSummary.chunks_created.toLocaleString()} chunks`
                    : 'Uploaded documents are active'}
                </span>
              </div>
            </div>
            <button
              type="button"
              className="danger-button"
              onClick={() => void clearSession()}
              disabled={isBusy}
            >
              {isClearingSession
                ? <LoaderCircle size={16} className="spinner" aria-hidden="true" />
                : <Trash2 size={16} aria-hidden="true" />}
              Clear session
            </button>
          </section>
        )}

        {dataSource === 'upload' && !sessionId && (
          <UploadZone
            onSessionCreated={handleSessionCreated}
            onUploadStateChange={setIsUploading}
            disabled={!isBackendReady || isBusy}
          />
        )}

        {canQuery && (
          <section className="conversation-panel" aria-label="Research conversation">
            <div className="conversation-toolbar">
              <div>
                <span className="toolbar-label">
                  {dataSource === 'default' ? 'Default corpus' : 'Custom corpus'}
                </span>
                <span className="toolbar-meta">
                  {activeConversation.turns.length === 0
                    ? 'New conversation'
                    : `${activeConversation.turns.length} ${activeConversation.turns.length === 1 ? 'turn' : 'turns'}`}
                </span>
              </div>
              {activeConversation.turns.length > 0 && (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={startNewConversation}
                  disabled={isBusy}
                >
                  <MessageSquarePlus size={16} aria-hidden="true" />
                  New conversation
                </button>
              )}
            </div>

            <div className="transcript" aria-live="polite" aria-busy={isQuerying}>
              {activeConversation.turns.length === 0 && !pendingQuery && (
                <div className="empty-conversation">
                  <div className="empty-icon" aria-hidden="true">
                    {dataSource === 'default' ? <LibraryBig size={24} /> : <FileText size={24} />}
                  </div>
                  <h2>No questions yet</h2>
                  <div className="example-queries" aria-label="Example questions">
                    {examples.map((example) => (
                      <button
                        type="button"
                        key={example}
                        onClick={() => void handleSearch(example)}
                        disabled={!isBackendReady || isBusy}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {activeConversation.turns.map((turn) => (
                <ResultCard key={turn.id} turn={turn} />
              ))}

              {pendingQuery && (
                <div className="pending-turn" role="status">
                  <div className="user-message">{pendingQuery}</div>
                  <div className="pipeline-loading">
                    <LoaderCircle size={22} className="spinner" aria-hidden="true" />
                    <div>
                      <strong>Running the research pipeline</strong>
                      <span>Routing, retrieving, reranking, and generating a grounded answer.</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <SearchBar
              onSearch={handleSearch}
              isLoading={isQuerying}
              disabled={!isBackendReady || isClearingSession}
              placeholder={dataSource === 'default'
                ? 'Ask about the indexed ML papers'
                : 'Ask about your uploaded documents'}
            />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
