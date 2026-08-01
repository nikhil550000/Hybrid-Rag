import type {
  ErrorResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  UploadResponse,
} from './types';

const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();

export const API_URL = (configuredApiUrl || 'http://127.0.0.1:8000').replace(/\/$/, '');

type JsonRecord = Record<string, unknown>;

export class ApiError extends Error {
  readonly code: string;
  readonly status?: number;
  readonly requestId?: string;

  constructor(error: ErrorResponse) {
    super(error.message);
    this.name = 'ApiError';
    this.code = error.error;
    this.status = error.status;
    this.requestId = error.request_id;
  }

  toResponse(): ErrorResponse {
    return {
      error: this.code,
      message: this.message,
      request_id: this.requestId,
      status: this.status,
    };
  }
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function validationMessage(detail: unknown[]): string {
  const messages = detail
    .map((issue) => {
      if (!isRecord(issue) || typeof issue.msg !== 'string') return null;
      const location = Array.isArray(issue.loc)
        ? issue.loc.filter((part) => part !== 'body').join('.')
        : '';
      return location ? `${location}: ${issue.msg}` : issue.msg;
    })
    .filter((message): message is string => Boolean(message));

  return messages.length > 0 ? messages.join(' ') : 'The request was not valid.';
}

function errorFromResponse(response: Response, body: unknown): ApiError {
  const envelope = isRecord(body) ? body : null;
  const detail = envelope?.detail ?? body;
  const detailRecord = isRecord(detail) ? detail : null;
  const requestId =
    (typeof detailRecord?.request_id === 'string' ? detailRecord.request_id : undefined)
    ?? response.headers.get('X-Request-ID')
    ?? undefined;

  let code = `HTTP_${response.status}`;
  let message = response.statusText || 'The request could not be completed.';

  if (typeof detailRecord?.error === 'string') code = detailRecord.error;
  if (typeof detailRecord?.message === 'string') {
    message = detailRecord.message;
  } else if (typeof detail === 'string') {
    message = detail;
  } else if (Array.isArray(detail)) {
    code = 'VALIDATION_ERROR';
    message = validationMessage(detail);
  } else if (typeof body === 'string' && body.trim()) {
    message = body;
  }

  return new ApiError({
    error: code,
    message,
    request_id: requestId,
    status: response.status,
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError({
      error: 'NETWORK_ERROR',
      message: 'The API could not be reached. Check that the backend server is running.',
    });
  }

  const body = await readResponseBody(response);
  if (!response.ok) throw errorFromResponse(response, body);

  return body as T;
}

export function toErrorResponse(error: unknown): ErrorResponse {
  if (error instanceof ApiError) return error.toResponse();

  return {
    error: 'UNEXPECTED_ERROR',
    message: 'Something unexpected happened. Please try again.',
  };
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export function submitQuery(payload: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  return request<UploadResponse>('/upload', {
    method: 'POST',
    body: formData,
  });
}

export function deleteDocumentSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/session/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}
