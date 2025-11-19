# Personalised AI Chatbot (RAG)

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Database & Storage Strategy](#3-database--storage-strategy)
4. [Architecture & Project Flow](#4-architecture--project-flow)
5. [Setup Instructions](#5-setup-instructions)
6. [Usage Guide](#6-usage-guide)
7. [Testing](#7-testing)
8. [Deployment](#8-deployment)
9. [Contributing](#9-contributing)
10. [License & Credits](#10-license--credits)
11. [Additional Resources](#11-additional-resources)

---

## 1. Project Overview

The Personalised AI Chatbot is a Retrieval-Augmented Generation (RAG) assistant tailored for aviation domain knowledge. It allows maintenance engineers, student pilots, and aviation enthusiasts to upload authoritative PDFs (manuals, handbooks, notes) and receive grounded answers that cite the uploaded sources. The Streamlit-based interface delivers a low-friction way to ingest documents, inspect ingestion status, and ask contextualised questions, making it ideal for workshops, classroom demos, and on-the-job reference.

**Core capabilities**

- Upload single or multiple aviation PDFs and embed them into a local vector store.
- Ask natural-language questions and receive answers backed by citations pointing to the original documents.
- Switch between OpenAI and Google Gemini providers for both embeddings and generation without code changes.
- Persist embeddings and metadata locally so knowledge bases survive application restarts.

---

## 2. Technology Stack

| Layer         | Technology                                                                   | Purpose                                        | Rationale                                                            |
| ------------- | ---------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------- |
| Runtime       | Python 3.13 (managed via `uv`)                                               | Primary programming language                   | Latest language features, deterministic builds with `uv`.            |
| UI            | Streamlit (`[src/streamlit_app.py](src/streamlit_app.py)`)                   | Web UI for ingestion and Q&A                   | Fast prototyping, minimal boilerplate, supports interactive widgets. |
| Retrieval     | LangChain Core (`langchain`, `langchain-community`, `langchain-core`)        | Vector store orchestration, prompt chaining    | Rich ecosystem for RAG patterns and provider abstraction.            |
| PDF Loading   | `langchain_community.document_loaders.PyPDFLoader`                           | Convert PDFs into LangChain `Document` objects | Proven parser with page metadata retention.                          |
| Text Chunking | `langchain_text_splitters.RecursiveCharacterTextSplitter`                    | Context-aware splitting of long documents      | Maintains overlap to preserve semantic continuity.                   |
| Embeddings    | OpenAI (`OpenAIEmbeddings`) / Google Gemini (`GoogleGenerativeAIEmbeddings`) | Numerical representation of text for retrieval | Flexibility to leverage either vendor; user-selectable in UI.        |
| Generation    | OpenAI / Google Gemini chat models                                           | Factual answer synthesis                       | Aligns with chosen embedding provider for consistent quality.        |
| Storage       | LangChain `InMemoryVectorStore` with JSON persistence                        | Lightweight local vector index                 | Simple, file-based persistence without external DB dependency.       |
| Configuration | `python-dotenv`                                                              | Load secrets from `.env`                       | Simplified local configuration.                                      |
| PDF Hashing   | Python `hashlib`                                                             | Deduplicate uploads                            | Prevents redundant embedding operations.                             |

---

## 3. Database & Storage Strategy

The application does **not** rely on a traditional SQL/NoSQL database. Instead, it uses LangChain’s `InMemoryVectorStore`, persisted to JSON files on disk:

- Vector store payload: `[data/vector_store.json](data/vector_store.json)` for OpenAI or provider-specific files such as `[data/vector_store_gemini.json](data/vector_store_gemini.json)`.
- Metadata registry: `[data/vector_store_meta.json](data/vector_store_meta.json)` or provider-specific variants (e.g., `[data/vector_store_meta_gemini.json](data/vector_store_meta_gemini.json)`).
- Uploaded source PDFs are stored in `[data/uploads/](data/uploads/)`, named using SHA-256 hashes (`compute_file_hash`).

Each uploaded document’s metadata tracks:

- `file_name`, `path`, `doc_ids`, and `chunks`.
- Citations include file name and page numbers, enabling traceable answers.

Because the storage is local and file-based, deployments can swap to alternative vector stores (e.g., Chroma, Pinecone) by replacing the helper functions in `[src/streamlit_app.py](src/streamlit_app.py)`.

---

## 4. Architecture & Project Flow

The Streamlit script orchestrates the entire workflow, encapsulated in `[src/streamlit_app.py](src/streamlit_app.py)`:

1. **Startup**

   - Load environment variables via `load_env_file`.
   - Derive configuration (provider choice, chunk sizes, storage paths).
   - Warm up embedding + chat models (`get_embeddings`, `get_chat_model`) and vector store (`load_vector_store`).

2. **Document Ingestion**

   - Users upload PDFs through Streamlit’s file uploader.
   - Files are hash-deduplicated (`compute_file_hash`), saved locally, then parsed into `Document` objects using `PyPDFLoader`.
   - `RecursiveCharacterTextSplitter` chunks documents; metadata is enriched with source identifiers (`enrich_documents`).
   - Chunks are embedded and inserted into the vector store, with metadata persisted.

3. **Question Answering**
   - Queries trigger vector similarity search (`vector_store.similarity_search`).
   - Retrieved contexts feed a LangChain prompt template (`ChatPromptTemplate`) bound to the chosen chat model.
   - Answers are rendered with citations referencing the source documents (`show_citations`).

```mermaid
flowchart LR
    User[User in Browser] -->|Upload PDFs / Ask Question| Streamlit[Streamlit UI<br/>main()]
    subgraph Streamlit App
        Streamlit -->|selected provider| Config[Provider Config]
        Streamlit -->|ingest| Ingestion[ingest_pdfs()]
        Streamlit -->|ask| QA[answer_question()]
    end
    Ingestion -->|chunk & embed| VectorStore[(InMemoryVectorStore JSON)]
    Ingestion -->|update| Metadata[metadata JSON]
    QA -->|similarity_search| VectorStore
    QA -->|format citations| Response[Answer + Citations]
    Response -->|render| User
```

**Key modules/components**

- `main()` UI loop: `[streamlit_app.main()](src/streamlit_app.py:381)`
- Ingestion pipeline: `[streamlit_app.ingest_pdfs()](src/streamlit_app.py:179)`
- Answer pipeline: `[streamlit_app.answer_question()](src/streamlit_app.py:277)`
- State management: `[streamlit_app.bootstrap_state()](src/streamlit_app.py:315)`

---

## 5. Setup Instructions

### Prerequisites

- Python 3.13 (enforced via `uv`).
- `uv` package manager installed globally (`pip install uv` or follow official docs).
- OpenAI and/or Google Gemini API credentials.

### Environment Variables

Create a `.env` file (or copy `[.env.example](.env.example)`):

```
GOOGLE_API_KEY=<your-google-key>
OPENAI_API_KEY=<your-openai-key>
GEMINI_CHAT_MODEL=gemini-2.5-flash
OPENAI_CHAT_MODEL=openai:gpt-4.1
TOP_K=3
VECTOR_STORE_PATH=data/vector_store.json
VECTOR_STORE_META_PATH=data/vector_store_meta.json
UPLOAD_DIR=data/uploads
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> Only the provider-specific keys are required; optional fields fall back to sensible defaults.

### Installation

1. Clone the repository and enter the workspace.
2. Sync dependencies (installs and locks deterministic versions):
   ```bash
   uv sync
   ```
3. (Optional) Validate formatting and linting:
   ```bash
   uv run ruff format
   uv run ruff check
   ```

---

## 6. Usage Guide

### Launch the Streamlit App

```bash
uv run streamlit run src/streamlit_app.py
```

### Typical Workflow

1. Open the Streamlit UI (defaults to http://localhost:8501).
2. Choose your preferred model provider (OpenAI or Gemini).
3. Upload one or more aviation PDFs.
4. Click **“Add to knowledge base”** to embed documents.
5. Once ingestion succeeds, enter a question in the text area and click **“Get answer”**.
6. Review the answer and citations for traceability.

**CLI-only ingestion/testing**  
Future scripts can reuse the ingestion helpers in `[src/streamlit_app.py](src/streamlit_app.py)` (e.g., `ingest_pdfs` and `answer_question`) from `[src/rag-agent.py](src/rag-agent.py)` once populated.

---

## 7. Testing

- Framework: `pytest`
- Command:
  ```bash
  uv run pytest
  ```
- Current status: Formal tests are pending. Target smoke coverage for ingestion and an end-to-end retrieval test once fixtures are prepared under `[testing/](testing/)`.

---

## 8. Deployment

### Local / On-Prem

- Run the Streamlit app behind an internal reverse proxy (e.g., Nginx) if required.
- Persist the `data/` directory to retain embeddings between restarts.
- Set environment variables securely (systemd unit, Docker secrets, etc.).

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Configure a Streamlit app pointing to `[src/streamlit_app.py](src/streamlit_app.py)`.
3. Add required secrets (API keys) via Streamlit’s secrets manager.

### Containerisation (suggested pattern)

1. Package with a multi-stage Dockerfile installing `uv`.
2. Run `uv sync --frozen` during build.
3. Execute `uv run streamlit run src/streamlit_app.py --server.port 8501`.
4. Mount a persistent volume for `data/`.

### CI/CD Hooks (future)

- Add jobs to execute:
  - `uv sync --frozen`
  - `uv run ruff check`
  - `uv run pytest`
  - Optional Streamlit smoke test (e.g., `pytest --headless` with Playwright).

---

## 9. Contributing

We follow the workflow documented in `[AGENTS.md](AGENTS.md)`:

- Branch naming: `feature/<topic>`, `bugfix/<issue>`, etc.
- Keep commits small, imperative (`Add ingestion status alerts`), referencing issues (`Refs #42`).
- Code style: Python 3.13 with type hints and docstrings for public functions. Format with `uv run ruff format`; lint via `uv run ruff check`.
- Tests: Add or update `pytest` suites alongside new features. Use fixtures for large PDFs stored in `data/fixtures/`.
- Pull Requests must include:
  - Context paragraph and bullet summary.
  - Test evidence (`uv run pytest` output).
  - Screenshots/log snippets for Streamlit or notebook work.
  - Note any new dependencies and tagging of relevant reviewers.

---

## 10. License & Credits

- **License:** Currently unlicensed (all rights reserved). Contact the maintainers for reuse or distribution inquiries.
- **Credits:** Built with LangChain, Streamlit, OpenAI, and Google Generative AI SDKs. PDF parsing by `PyPDFLoader`. Inspired by best practices documented in `[AGENTS.md](AGENTS.md)`.

---

## 11. Additional Resources

- Repository guidelines and onboarding: `[AGENTS.md](AGENTS.md)`
- Issue tracker: Use GitHub Issues on the project repository to log bugs and feature requests.
- LangChain documentation: https://python.langchain.com/
- Streamlit documentation: https://docs.streamlit.io/
- `uv` package manager docs: https://docs.astral.sh/uv/
