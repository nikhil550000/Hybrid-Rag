export interface Citation {
  chunk_id: string;
  source: string;
  passage: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  refused: boolean;
  latency_ms: number;
  cost_usd: number;
}

export interface ErrorResponse {
  error: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  chunks_indexed: number;
}

export interface UploadResponse {
  session_id: string;
  files_processed: number;
  chunks_created: number;
}
