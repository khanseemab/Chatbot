"""Streamlit UI for the aviation RAG assistant."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_env_file(env_path: Path) -> None:
    """Populate os.environ with key/value pairs from a simple .env file."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store.json"))
VECTOR_STORE_META_PATH = Path(
    os.getenv("VECTOR_STORE_META_PATH", "data/vector_store_meta.json")
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "3"))
ENV_FILE = Path(os.getenv("ENV_FILE", ".env"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "openai:gpt-4.1")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
PROVIDER_OPTIONS = ("OpenAI", "Gemini")

load_env_file(ENV_FILE)


def provider_slug(provider: str) -> str:
    """Return a filesystem-friendly provider identifier."""
    return provider.lower().replace(" ", "_")


def resolve_storage_paths(provider: str) -> Tuple[Path, Path]:
    """Return provider-specific vector-store and metadata paths."""
    slug = provider_slug(provider)
    if slug == "openai":
        return VECTOR_STORE_PATH, VECTOR_STORE_META_PATH
    vec_name = f"{VECTOR_STORE_PATH.stem}_{slug}{VECTOR_STORE_PATH.suffix}"
    meta_name = f"{VECTOR_STORE_META_PATH.stem}_{slug}{VECTOR_STORE_META_PATH.suffix}"
    return VECTOR_STORE_PATH.with_name(vec_name), VECTOR_STORE_META_PATH.with_name(meta_name)


def ensure_directories(vector_path: Path, metadata_path: Path) -> None:
    """Create directories required for persistence."""
    for path in (vector_path.parent, metadata_path.parent, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_provider_keys(provider: str) -> None:
    """Validate API keys for the selected provider before continuing."""
    if provider == "Gemini" and not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Please set the GOOGLE_API_KEY environment variable to use Gemini models."
        )
        st.stop()
    if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable to continue.")
        st.stop()


def get_embeddings(provider: str) -> Embeddings:
    """Return the embedding client configured for the selected provider."""
    if provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def get_chat_model(provider: str) -> BaseChatModel:
    """Instantiate the chat model used for answering queries."""
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
    return init_chat_model(OPENAI_CHAT_MODEL)


def load_vector_store(
    embedding: Embeddings, vector_path: Path
) -> InMemoryVectorStore:
    """Load the persisted vector store or return a fresh instance."""
    if vector_path.exists():
        return InMemoryVectorStore.load(str(vector_path), embedding=embedding)
    return InMemoryVectorStore(embedding=embedding)


def dump_vector_store(store: InMemoryVectorStore, vector_path: Path) -> None:
    """Persist the current state of the vector store."""
    store.dump(str(vector_path))


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata describing which files have been indexed."""
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def persist_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """Persist ingestion metadata to disk."""
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def compute_file_hash(file_bytes: bytes) -> str:
    """Return a deterministic hash for deduplicating uploads."""
    return hashlib.sha256(file_bytes).hexdigest()


def save_pdf_to_disk(file_bytes: bytes, original_name: str, file_hash: str) -> Path:
    """Persist an uploaded PDF so future sessions can reload it."""
    safe_name = f"{file_hash}_{Path(original_name).name}"
    destination = UPLOAD_DIR / safe_name
    destination.write_bytes(file_bytes)
    return destination


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter used for chunking PDF pages."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )


def enrich_documents(
    documents: Sequence[Document],
    source_id: str,
    source_file: str,
    source_path: Path,
) -> List[Document]:
    """Attach source metadata so citations remain traceable."""
    enriched: List[Document] = []
    for doc in documents:
        metadata = {
            **doc.metadata,
            "source_id": source_id,
            "source_file": source_file,
            "source_path": str(source_path),
        }
        doc.metadata = metadata
        enriched.append(doc)
    return enriched


ProgressCallback = Callable[[str], None]


def ingest_pdfs(
    uploaded_files: Iterable[Any],
    vector_store: InMemoryVectorStore,
    metadata_registry: Dict[str, Any],
    vector_store_path: Path,
    metadata_path: Path,
    progress_fn: ProgressCallback | None = None,
) -> List[Dict[str, Any]]:
    """Ingest uploaded PDFs into the vector store and persist the results."""
    splitter = build_text_splitter()
    ingestion_report: List[Dict[str, Any]] = []
    updates_performed = False

    def notify(message: str) -> None:
        if progress_fn:
            progress_fn(message)

    for uploaded in uploaded_files:
        file_bytes = uploaded.getvalue()
        file_hash = compute_file_hash(file_bytes)
        notify(f"Processing {uploaded.name}...")

        if metadata_registry.get(file_hash):
            notify(f"Skipping {uploaded.name}; already embedded.")
            ingestion_report.append(
                {
                    "file": uploaded.name,
                    "status": "skipped (already indexed)",
                    "chunks": 0,
                }
            )
            continue

        stored_path = save_pdf_to_disk(file_bytes, uploaded.name, file_hash)
        notify(f"Loading {uploaded.name} into memory...")
        loader = PyPDFLoader(str(stored_path))
        documents = loader.load()
        if not documents:
            ingestion_report.append(
                {"file": uploaded.name, "status": "failed (empty PDF)", "chunks": 0}
            )
            notify(f"{uploaded.name} failed: no readable pages.")
            continue

        splits = splitter.split_documents(documents)
        if not splits:
            ingestion_report.append(
                {
                    "file": uploaded.name,
                    "status": "failed (no splits)",
                    "chunks": 0,
                }
            )
            notify(f"{uploaded.name} failed: could not split into chunks.")
            continue

        notify(
            f"{uploaded.name}: created {len(splits)} chunks. Generating embeddings..."
        )
        enriched_docs = enrich_documents(splits, file_hash, uploaded.name, stored_path)
        doc_ids = vector_store.add_documents(documents=enriched_docs)
        metadata_registry[file_hash] = {
            "file_name": uploaded.name,
            "path": str(stored_path),
            "doc_ids": doc_ids,
            "chunks": len(doc_ids),
        }
        ingestion_report.append(
            {"file": uploaded.name, "status": "indexed", "chunks": len(doc_ids)}
        )
        notify(f"{uploaded.name}: stored {len(doc_ids)} embedded chunks.")
        updates_performed = True

    if updates_performed:
        dump_vector_store(vector_store, vector_store_path)
        persist_metadata(metadata_registry, metadata_path)

    return ingestion_report


def build_context(docs: Sequence[Document]) -> Tuple[str, List[str]]:
    """Format retrieved documents into a prompt-ready context block."""
    context_blocks: List[str] = []
    citation_labels: List[str] = []

    for idx, doc in enumerate(docs, start=1):
        label = f"S{idx}"
        source_name = doc.metadata.get("source_file") or doc.metadata.get("source", "pdf")
        page = doc.metadata.get("page")
        prefix = f"[{label}] {source_name}"
        if page is not None:
            prefix += f" (page {page})"
        context_blocks.append(f"{prefix}\n{doc.page_content}")
        citation_labels.append(prefix)

    return "\n\n".join(context_blocks), citation_labels


def answer_question(
    question: str,
    vector_store: InMemoryVectorStore,
    model,
) -> Tuple[str, List[Document], List[str]]:
    """Answer a user question using the vector store and cite the supporting chunks."""
    retrieved_docs = vector_store.similarity_search(question, k=TOP_K)
    if not retrieved_docs:
        return (
            "I could not find any material in the knowledge base yet. "
            "Please upload an aviation PDF first.",
            [],
            [],
        )

    context, citations = build_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an aviation maintenance assistant. "
                    "Only answer with the provided context. "
                    "If the context lacks the answer, respond that the knowledge base "
                    "does not cover it. Always cite sources as [S1], [S2], etc."
                ),
            ),
            (
                "user",
                "Question: {question}\n\nContext:\n{context}",
            ),
        ]
    )
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer, retrieved_docs, citations


def bootstrap_state(
    provider: str,
) -> Tuple[InMemoryVectorStore, Dict[str, Any], Path, Path]:
    """Initialize reusable resources in the Streamlit session."""
    vector_path, metadata_path = resolve_storage_paths(provider)
    ensure_directories(vector_path, metadata_path)

    if st.session_state.get("active_provider") != provider:
        for key in ("embeddings", "vector_store", "store_metadata", "chat_model"):
            st.session_state.pop(key, None)
        st.session_state.active_provider = provider

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = get_embeddings(provider)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store(
            st.session_state.embeddings, vector_path
        )
    if "store_metadata" not in st.session_state:
        st.session_state.store_metadata = load_metadata(metadata_path)
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = get_chat_model(provider)

    st.session_state.vector_store_path = vector_path
    st.session_state.metadata_path = metadata_path

    return (
        st.session_state.vector_store,
        st.session_state.store_metadata,
        vector_path,
        metadata_path,
    )


def render_ingestion_report(report: Sequence[Dict[str, Any]]) -> None:
    """Render the ingestion summary in Streamlit."""
    if not report:
        return
    for item in report:
        status = item["status"]
        file_name = item["file"]
        chunks = item["chunks"]
        message = f"{file_name}: {status}"
        if chunks:
            message += f" ({chunks} chunks)"
        if status.startswith("indexed"):
            st.success(message)
        elif status.startswith("skipped"):
            st.info(message)
        else:
            st.warning(message)


def show_citations(docs: Sequence[Document]) -> None:
    """Display supporting citation snippets."""
    if not docs:
        return
    st.markdown("**Citations**")
    for idx, doc in enumerate(docs, start=1):
        source_name = doc.metadata.get("source_file") or doc.metadata.get("source", "pdf")
        page = doc.metadata.get("page", "n/a")
        snippet = doc.page_content[:240].strip().replace("\n", " ")
        st.markdown(f"[S{idx}] {source_name} — page {page}")
        st.caption(f"{snippet}...")


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Personalised AI Chatbot", layout="wide")
    st.title("Personalised AI Chatbot")
    st.caption(
        "Ask your questions based on the PDFs you upload. "
       
        "Answers rely solely on your indexed documents."
    )

    default_provider = st.session_state.get("provider_choice", PROVIDER_OPTIONS[0])
    if default_provider not in PROVIDER_OPTIONS:
        default_provider = PROVIDER_OPTIONS[0]
    provider_index = PROVIDER_OPTIONS.index(default_provider)
    provider = st.selectbox(
        "Model provider",
        PROVIDER_OPTIONS,
        index=provider_index,
        key="provider_choice",
        help="Use the same provider for embeddings and chat responses.",
    )
    require_provider_keys(provider)

    (
        vector_store,
        metadata_registry,
        vector_store_path,
        metadata_path,
    ) = bootstrap_state(provider)
    existing_chunks = len(vector_store.store)
    tracked_files = len(metadata_registry)
    st.write(
        f"Knowledge base contains **{existing_chunks}** chunks "
        f"from **{tracked_files}** file(s)."
    )
    st.caption(f"Using {provider} for embeddings and chat responses.")

    st.subheader("Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more related PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Add to knowledge base"):
        status = st.status("Preparing ingestion...", expanded=True)
        loader_placeholder = st.empty()

        def progress_update(message: str) -> None:
            loader_placeholder.info(message)
            status.write(message)

        with st.spinner("Computing embeddings..."):
            report = ingest_pdfs(
                uploaded_files,
                vector_store,
                metadata_registry,
                vector_store_path,
                metadata_path,
                progress_fn=progress_update,
            )
        loader_placeholder.empty()
        status.update(label="Embeddings complete", state="complete")
        render_ingestion_report(report)

    st.subheader("Ask a Question")
    question = st.text_area(
        "Enter your aviation question",
        placeholder="e.g., How should I service the landing gear struts?",
        height=120,
    )
    if st.button("Get answer"):
        if not question.strip():
            st.warning("Please enter a question before requesting an answer.")
        else:
            with st.spinner("Generating answer..."):
                answer, docs, _ = answer_question(
                    question.strip(), vector_store, st.session_state.chat_model
                )
            st.markdown("**Answer**")
            st.write(answer)
            show_citations(docs)


if __name__ == "__main__":
    main()
