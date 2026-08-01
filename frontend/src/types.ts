export interface Citation {
  chunk_id: string;
  source: string;
  page: number;
  passage: string;
}

export type QueryRoute =
  | 'GREETING'
  | 'APP_HELP'
  | 'OUT_OF_SCOPE'
  | 'BASIC_NON_RAG'
  | 'RAG_FACTUAL'
  | 'RAG_SUMMARY'
  | 'RAG_EXACT_KEYWORD_TABLE'
  | 'FOLLOW_UP';

export interface QueryRequest {
  query: string;
  session_id?: string;
  conversation_id?: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  refused: boolean;
  latency_ms: number;
  cost_usd: number;
  timings_ms: Record<string, number>;
  route: QueryRoute;
  conversation_id: string | null;
  retrieval_query: string;
  query_rewritten: boolean;
}

export interface ErrorResponse {
  error: string;
  message: string;
  request_id?: string;
  status?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  chunks_indexed: number;
  ready: boolean;
  checks: Record<string, boolean>;
}

export interface UploadResponse {
  session_id: string;
  files_processed: number;
  chunks_created: number;
}

export type DataSource = 'default' | 'upload';

export interface ChatTurn {
  id: string;
  query: string;
  response: QueryResponse;
}

export interface ConversationState {
  conversationId: string | null;
  turns: ChatTurn[];
}
