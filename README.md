# RAG Scholar

A production-grade Retrieval-Augmented Generation system built over 10 foundational ML research papers. Takes natural language questions, retrieves the most relevant passages using hybrid search (BM25 + vector) with cross-encoder re-ranking, and generates cited answers strictly grounded in the paper corpus.

## What It Does

You ask a question about any of the 10 indexed papers (Attention Is All You Need, BERT, GPT, Seq2Seq, Word2Vec, Bahdanau Attention, Sentence-BERT, RAG, Toolformer, and one more). The system:

1. Embeds your query and searches ChromaDB (dense) + BM25 (sparse) in parallel
2. Fuses results using Reciprocal Rank Fusion (k=60)
3. Re-ranks the top candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
4. Passes the top 5 chunks to the LLM with strict citation instructions
5. Validates every `[SOURCE: chunk_id]` tag in the response against the actual retrieved context
6. Returns the answer with verified citations, latency, and cost

If the context doesn't support an answer, the system refuses rather than hallucinating.

## Evaluation Results

Evaluated against a golden dataset of 100 QA pairs (10 per paper) using DeepEval with `AnthropicModel` as the LLM judge:

| Metric | Score |
|--------|-------|
| Faithfulness | **0.974** |
| Answer Relevancy | **0.965** |
| Context Recall | **0.967** |
| Citation Accuracy | **0.928** |
| P50 Latency | 4,869ms |
| P90 Latency | 6,596ms |
| P95 Latency | 6,881ms |

Full report: `evals/report_20260708_015329.json`

## Architecture

```
                         ┌─────────────────────────────────────────────┐
  Offline (run once)     │         Ingestion Pipeline                  │
                         │                                             │
  data/papers/*.pdf ──►  │  PDFLoader → Chunker → Embedder ──► ChromaDB│
                         │                    └──► BM25Indexer ──► .pkl│
                         └─────────────────────────────────────────────┘

                         ┌─────────────────────────────────────────────┐
  Online (per request)   │          Query Pipeline                     │
                         │                                             │
  User Query ──►         │  ┌─── DenseRetriever (ChromaDB) ──┐        │
                         │  │                                 │        │
                         │  └─── SparseRetriever (BM25) ──────┤        │
                         │                                    ▼        │
                         │              Reciprocal Rank Fusion          │
                         │                      │                      │
                         │              CrossEncoderReranker            │
                         │                      │                      │
                         │         Generator (LLM + CitationValidator) │
                         │                      │                      │
                         │              Verified Answer + Citations     │
                         └─────────────────────────────────────────────┘

  Observability:  Langfuse TracerProtocol wraps retrieval + generation spans
```

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Package manager | `uv` | `uv sync` to install; `pyproject.toml` is source of truth |
| PDF parsing | `pymupdf` | Extracts text page-by-page, preserves page metadata |
| Text splitting | `langchain` | `RecursiveCharacterTextSplitter` (600 tokens, 100 overlap) |
| Vector store | `chromadb` | Local persistent client, cosine similarity |
| Sparse search | `rank-bm25` | BM25Okapi, serialized via pickle |
| Re-ranker | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no API key) |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` (local, no API key) |
| LLM | `anthropic` / `google-genai` | Configurable via `config/settings.yaml` (currently: `claude-haiku-4-5`) |
| Observability | `langfuse` v4 SDK | Injected via `TracerProtocol`; `NullTracer` fallback for tests |
| Evaluation | `deepeval` | Faithfulness + AnswerRelevancy + ContextRecall via `AnthropicModel` judge |
| API | `fastapi` + `uvicorn` | Thin wrapper around `QueryPipeline` |
| Frontend | React 18 + TypeScript + Vite | Glassmorphism dark theme, custom CSS |

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js v18+
- API key: `ANTHROPIC_API_KEY` (required) or `GOOGLE_API_KEY` (if using Google provider)
- Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (for observability)

### Backend

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run ingestion (once)
python scripts/ingest.py

# Start the API server
uvicorn src.api.routes:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### CLI (no server needed)

```bash
python src/main.py "What positional encoding does the Transformer use?"
```

### Run Evaluation

```bash
python evals/run_eval.py
```

Exits with code 1 if faithfulness drops below the threshold defined in `config/settings.yaml`. The CI pipeline (`.github/workflows/eval_gate.yml`) uses this to gate PRs.

## Repository Structure

```
.
├── src/
│   ├── main.py                  # CLI entry point
│   ├── api/                     # FastAPI routes + Pydantic schemas
│   ├── config/                  # settings.py (loads settings.yaml + .env)
│   ├── ingestion/               # loader.py, chunker.py, indexer.py
│   ├── retrieval/               # dense.py, sparse.py, hybrid.py, reranker.py
│   ├── generation/              # prompt.py, generator.py, citations.py
│   ├── llm/                     # client.py, providers.py, embeddings.py
│   ├── observability/           # tracer.py, metrics.py
│   ├── pipeline/                # ingest.py, query.py (orchestrators)
│   ├── store/                   # vector.py (ChromaDB), bm25.py
│   └── utils/                   # logger.py, helpers.py
├── frontend/                    # React + TypeScript + Vite
│   └── src/components/          # SearchBar, ResultCard, CitationCard, UploadZone
├── evals/
│   ├── run_eval.py              # Eval runner (CI gate)
│   ├── metrics.py               # DeepEval metric wrappers
│   └── dataset.json             # 100 golden QA pairs
├── config/
│   └── settings.yaml            # All tunable parameters
├── prompts/v2/                  # Versioned prompt templates
├── scripts/
│   └── ingest.py                # Ingestion CLI
├── data/papers/                 # 10 ML research PDFs
├── tests/                       # Unit tests (zero live API calls)
├── .github/workflows/
│   ├── tests.yml                # Unit tests on every push
│   └── eval_gate.yml            # Full eval on PRs to main
├── AGENT.md                     # Project context for AI assistants
└── BUILDLOG.md                  # Session-by-session development log
```

## Document Upload (V2 Feature)

Users can upload their own PDFs for isolated Q&A sessions via the web UI:

- Click **Custom Uploads** → drag-and-drop PDFs (max 5)
- The backend creates an ephemeral `ChromaDB.EphemeralClient()` + in-memory `BM25Store` scoped to a `session_id`
- Queries with that `session_id` only search the uploaded documents
- **Clear Session** deletes the ephemeral pipeline and frees memory
- Uploaded data never touches the persistent 10-paper corpus

## What's Next

This is the completed initial architecture. Current focus is on the **optimization phase**:

- Experimenting with semantic chunking strategies to improve retrieval fidelity
- Optimizing prompt templates for edge cases found during the 100-question eval
- Profiling and reducing average query latency
