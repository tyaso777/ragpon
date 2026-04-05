# ragpon

`ragpon` is a Python project for RAG (Retrieval Augmented Generation) over Japanese documents. It includes document ingestion, chunking, BM25 and vector search, reranking, and application layers built with FastAPI and Streamlit.

## Features

- Ingest PDF, TXT, HTML, Word (`.docx`), and PowerPoint (`.pptx`) files
- Index pandas `DataFrame` objects directly
- Split Japanese text into chunks
- Combine BM25 and ChromaDB retrieval
- Improve ranking quality with rerankers
- Run a chat-oriented UI with FastAPI and Streamlit

## Project Structure

- `ragpon/`
  Core library code, including configuration, document readers, chunk processors, repositories, models, and services.
- `ragpon/apps/fastapi/`
  API server that integrates with OpenAI or Azure OpenAI and provides the RAG backend.
- `ragpon/apps/streamlit/`
  Streamlit-based frontend UI.
- `ragpon/examples/`
  Sample configuration files and usage examples.
- `tests/`
  Unit tests.
- `docs/`
  Sphinx documentation scaffold.

## Requirements

- Python `>=3.11,<3.14`
- Poetry or `pip`
- OpenAI or Azure OpenAI credentials
- Local paths for the embedding and reranker models you plan to use

Japanese chunking uses `ja_ginza`, so install dependencies including the related model packages during setup.

## Setup

### With Poetry

```bash
poetry install
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Sample configuration files:

- `ragpon/examples/sample_config.yml`
- `ragpon/apps/fastapi/config/sample_config.yml`

At minimum, review these keys:

- `MODELS.CL_NAGOYA_RURI_V3_MODEL_PATH`
- `MODELS.RURI_RERANKER_LARGE_PATH`
- `DATABASES.USE_BM25`
- `DATABASES.BM25_PATH`
- `DATABASES.USE_CHROMADB`
- `DATABASES.CHROMADB_COLLECTION_NAME`
- `DATABASES.CHROMADB_FOLDER_PATH`

LLM configuration is controlled through environment variables.

### Example: OpenAI

```bash
export OPENAI_TYPE=openai
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_MODEL=gpt-4o-mini
```

### Example: Azure OpenAI

```bash
export OPENAI_TYPE=azure
export AZURE_OPENAI_ENDPOINT=YOUR_ENDPOINT
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
export AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT
export AZURE_OPENAI_MODEL=gpt-4o-mini
```

## Library Usage

A minimal example is available in [`ragpon/examples/basic_usage.py`](/mnt/d/Users/AtsushiSuzuki/OneDrive/デスクトップ/test/ragpon/ragpon/examples/basic_usage.py).

```python
from ragpon import Config, DocumentProcessingService, JAGinzaChunkProcessor

config = Config("ragpon/examples/sample_config.yml")

service = DocumentProcessingService(
    config_or_config_path=config,
    chunk_processor=JAGinzaChunkProcessor(chunk_size=300),
)

service.process_file("path/to/document.pdf")
results = service.search(query="What is ragpon?")
print(results)
```

`DocumentProcessingService` provides these main operations:

- `process_file`: ingest a file
- `process_dataframe`: ingest a `DataFrame` as-is
- `process_dataframe_with_chunking`: chunk a text column in a `DataFrame` before ingesting
- `search`: run retrieval
- `enhance_search_results`: attach surrounding context to results
- `rerank_results`: rerank retrieved results
- `delete_by_ids` / `delete_by_metadata`: remove indexed data

## Running the Apps

### Local Run

FastAPI:

```bash
uvicorn ragpon.apps.fastapi.fast_api_app:app --host 0.0.0.0 --port 8006 --workers 4
```

Streamlit:

```bash
streamlit run ragpon/apps/streamlit/streamlit_app.py --server.port 8005 --server.address 0.0.0.0
```

Exposed ports:

- Streamlit: `http://localhost:8005`
- FastAPI: `http://localhost:8006`

Example environment variables:

- `OPENAI_API_KEY`
- `OPENAI_TYPE`
- `OPENAI_MODEL`
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`
- `MYSQL_POOL_NAME`
- `MYSQL_POOL_SIZE`
- `MYSQL_AUTOCOMMIT`
- `MYSQL_CHARSET`

### Using Compose

The root `compose.yml` can start `postgres`, `chromadb`, `fastapi`, and `streamlit` together. However, `fastapi` and `streamlit` currently assume prebuilt local images named `ragpon-fastapi` and `ragpon-streamlit`, so additional build/setup work is required before `docker compose up`.

Notes:

- `compose.yml` includes a PostgreSQL service, but the FastAPI implementation currently defaults to `DB_TYPE = "mysql"`.
- Verify that the database configuration used by the app matches the container setup before relying on the compose stack.

## Testing

```bash
pytest
```

Run a specific test file if needed:

```bash
pytest tests/test_config.py
```

## Documentation

Sphinx sources are located in `docs/source/`.

```bash
cd docs
make html
```

## Directory Overview

```text
.
├── ragpon
│   ├── apps
│   ├── config
│   ├── domain
│   ├── examples
│   ├── ml_models
│   ├── repository
│   └── service
├── tests
├── docs
├── visualization
├── pyproject.toml
└── compose.yml
```

## License

See [`LICENSE`](/mnt/d/Users/AtsushiSuzuki/OneDrive/デスクトップ/test/ragpon/LICENSE).
