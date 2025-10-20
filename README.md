# AI-Powered Knowledge Base — Backend

This repository contains the backend of an AI-powered knowledge base search and enrichment system. It's a FastAPI application that implements a Retrieval-Augmented Generation (RAG) pipeline: document ingestion → vector storage → semantic search → LLM-backed answer generation and auto-enrichment suggestions.

This README is written for a backend-only delivery (no frontend). It includes setup, configuration, how to run the API server with Uvicorn, and example requests you can send to the endpoints.

## Contents

- `app/` — FastAPI application code
  - `main.py` — API endpoints and service wiring
  - `document_processor.py` — file save, text extraction, chunking logic
  - `vector_store.py` — ChromaDB-backed vector store wrapper
  - `rag_pipeline.py` — Retrieval + LLM prompt creation + response parsing
  - `auto_enrichment.py` — optional enrichment from external sources (Wikipedia placeholder)
  - `models.py` — Pydantic models used by endpoints
  - `logger.py` — logging helpers
- `config.py` — application settings (Pydantic Settings, .env support)
- `requirements.txt` — Python dependencies for the backend
- `chroma_db/` — persistent ChromaDB files (on-disk vector DB)
- `uploads/` — (runtime) uploaded files are saved here during processing

## Quick overview of features

- Upload documents (PDF, DOCX, TXT) to extract text, chunk it, and store embeddings in ChromaDB.
- Natural-language search endpoint that:
  - Retrieves relevant chunks from vector store
  - Uses Google Generative AI (Gemini via `langchain_google_genai`) to generate answers
  - Assesses missing information and returns enrichment suggestions
  - Optionally attempts simple auto-enrichment (Wikipedia placeholder)
- Administrative endpoints to inspect or reset the knowledge base

## Requirements

- Python 3.10+ recommended
- The project dependencies are listed in `requirements.txt`.

## Environment variables

Create a `.env` file in the `backend/` folder (or provide environment variables by other means). Important variables:

- `GOOGLE_API_KEY` (required) — API key for Google Generative AI (Gemini) used by the RAG pipeline
- `CHROMA_DB_PATH` (optional) — path to persistent ChromaDB directory (default: `./chroma_db`)
- `UPLOAD_DIR` (optional) — path to store temporary uploaded files (default: `./uploads`)
- `API_HOST` (optional) — host to bind the FastAPI app (default: `0.0.0.0`)
- `API_PORT` (optional) — port to bind the FastAPI app (default: `8000`)
- `OPENAI_API_KEY` (optional) — used if other auto-enrichment sources are added
- `WIKIPEDIA_API_URL` (optional) — base URL for Wikipedia API (default in code)

Example `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
CHROMA_DB_PATH=./chroma_db
UPLOAD_DIR=./uploads
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

Note: Never commit secrets to source control.

## Install dependencies

Recommended: create and activate a virtual environment.

```bash

python3 -m venv .venv

Activation
Linux/Mac : source .venv/bin/activate
Windows :   .\myenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer (or your environment uses `pipx`/conda), adapt accordingly.

## Run the API (development)

From the `backend/` directory, run the FastAPI app with Uvicorn. This will start the HTTP server exposing the endpoints described below.

```bash
# activate virtualenv if you created one
# cd to the backend folder
cd backend

# run the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If you don't have `uvicorn` installed globally, use the virtualenv's `uvicorn` or run:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Notes:
- `--reload` enables auto-reload when code changes — useful for development but not recommended in production.
- The app also has an `if __name__ == "__main__"` block in `app/main.py` so you can run `python app/main.py` (equivalent to the command above).

## API endpoints

The service exposes the following main endpoints. Example uses are provided with `curl` for a backend-only workflow.

- GET `/` — simple health check (returns status and message)
- GET `/health` — detailed health check (includes number of document chunks)
- POST `/upload` — upload a document (multipart/form-data: `file` field)
- POST `/search` — search the knowledge base (JSON body)
- GET `/documents/count` — returns number of chunks stored
- GET `/documents/list` — lists documents/chunks metadata
- DELETE `/documents/reset` — deletes all documents from the knowledge base (destructive)

### Upload a document

Example curl (replace `file.pdf`):

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.pdf"
```

Response: `DocumentUploadResponse` with `chunks_created` and status.

### Search the knowledge base

Request schema (example):

```json
{
  "query": "How do I create a virtual environment in Python?",
  "top_k": 5,
  "include_auto_enrichment": false
}
```

Example curl:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","top_k":5,"include_auto_enrichment":false}'
```

Response: `SearchResponse` with answer, confidence, sources, missing info and enrichment suggestions.


## Notes, limitations and next steps

- Google Generative AI (Gemini) is used via `langchain_google_genai`. Make sure the `GOOGLE_API_KEY` you provide has the right access and quota.
- The auto-enrichment code contains placeholder integrations (Wikipedia and stubs). For production, integrate reliable external APIs (Google Custom Search, Bing Search, or domain-specific sources).
- The current embedding model uses `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`. For better quality or scale, consider managed embedding APIs.
- Security: add authentication (API keys, OAuth) and restrict CORS for client deployments.
- Scaling: ChromaDB is used with persistent on-disk storage. For large datasets or multi-node deployments, consider vector DB services (Pinecone, Milvus, Weaviate) or a dedicated Chroma server.

## Troubleshooting

- If Chroma fails to initialize, check `CHROMA_DB_PATH` permissions and ensure the process can create files there.
- If `GOOGLE_API_KEY` is missing or invalid, the RAG pipeline will fail when invoking the LLM. Check logs for detailed error messages.
- To clear data, call DELETE `/documents/reset` (this will remove the Chroma collection for `knowledge_base`).
