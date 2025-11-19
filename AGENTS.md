# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/`, currently anchored by `rag-agent.py` for agent orchestration. Domain documents for retrieval live in `data/` (e.g., `AirplaneFlyingHandbook.pdf`), and exploratory notebooks belong in `testing/` (`rag.ipynb`). Keep scripts deterministic and make heavy document payloads configurable via paths so contributors can swap corpora without editing code.

## Build, Test, and Development Commands
- `uv sync` — install/lock dependencies declared in `pyproject.toml` and `uv.lock`.
- `uv run python src/rag-agent.py` — execute the agent entry point with the synced environment.
- `uv run pytest` — run automated tests once `testing/` houses formal suites; prefer colocating fixtures beside specs.
Use `uv` for everything (no `pip install .`), ensuring all contributors share the exact Python 3.13 toolchain.

## Coding Style & Naming Conventions
Write Python 3.13, using type hints and docstrings for every public function. Favor explicit factory helpers for LangChain components rather than inline configuration blocks. Name modules with hyphenless snake_case (`rag_agent.py` pattern) and constants in ALL_CAPS. Format with `uv run ruff format` and lint with `uv run ruff check`; fix style issues before pushing.

## Testing Guidelines
Adopt `pytest` with descriptive `test_<feature>.py` filenames. Mirror the runtime data shape using lightweight fixtures; large PDFs belong in `data/fixtures/` and should be referenced, not copied. Aim for smoke coverage of every ingestion pipeline plus at least one integration test that exercises embedding retrieval end to end. Run `uv run pytest -k "<scope>"` while iterating, then the full suite before submitting a PR.

## Commit & Pull Request Guidelines
Commits follow the existing imperative style (`Add RAG ingest helper`, `Fix pdf loader bug`). Keep them small and reference issues in the body (`Refs #42`). PRs must include: context paragraph, bullet summary of changes, explicit test evidence (`uv run pytest` output), and screenshots/log excerpts for notebook-driven work. Tag reviewers familiar with the touched modules and flag any new dependencies in the description.

## Security & Configuration Tips
Do not hard-code API keys; read from environment variables (`OPENAI_API_KEY`, etc.) and document new ones in the PR. Large PDFs or proprietary manuals belong in `data/` but should be git-ignored if sensitive. Validate that external calls respect rate limits and add retry/backoff helpers where appropriate.


# Additional Instructions

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.
