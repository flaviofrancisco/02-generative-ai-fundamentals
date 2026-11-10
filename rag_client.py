"""
RAG client for the NASA mission chat.

Responsibilities:
- discover ChromaDB backends (persist directories + collections) in the project,
- open a collection,
- run semantic search (with optional mission filter) using an OpenAI query
  embedding that matches the embedding model used by embedding_pipeline.py,
- format the retrieved chunks into a clean, source-attributed context string.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

# Must match the model used in embedding_pipeline.py so query vectors live in
# the same space as the stored document vectors.
QUERY_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS_IN_CONTEXT = 1500
CHROMA_DIR_PREFIXES = ("chroma",)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory.

    Returns a dict keyed by "<directory>/<collection>" with the directory,
    collection name, a display name and the document count.
    """
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    # Any directory whose name starts with "chroma" (chroma_db, chroma_db_openai, ...)
    chroma_dirs = sorted(
        d for d in current_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith(CHROMA_DIR_PREFIXES)
    )

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for collection in collections:
                # chromadb >= 0.6 returns Collection objects; older versions return names
                name = collection.name if hasattr(collection, "name") else str(collection)
                key = f"{chroma_dir.name}/{name}"

                try:
                    doc_count = client.get_collection(name).count()
                except Exception:
                    doc_count = "?"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": name,
                    "display_name": f"{chroma_dir.name} / {name} ({doc_count} chunks)",
                    "document_count": str(doc_count),
                }

        except Exception as e:  # connection or access error
            key = f"{chroma_dir.name}/<unavailable>"
            err = str(e).replace("\n", " ")
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {err[:60]}{'...' if len(err) > 60 else ''})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[Optional[object], bool, Optional[str]]:
    """Open the ChromaDB collection used for retrieval.

    Returns (collection, success, error) so the caller can report problems
    without raising inside the UI.
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # The pipeline stores pre-computed OpenAI embeddings, so the collection
        # is opened without an embedding function and queries pass their own
        # vectors (see embed_query).
        collection = client.get_collection(collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def _openai_api_key() -> Optional[str]:
    return os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def embed_query(query: str, api_key: Optional[str] = None,
                model: str = QUERY_EMBEDDING_MODEL) -> List[float]:
    """Embed the user question with the same OpenAI model as the documents."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or _openai_api_key())
    response = client.embeddings.create(model=model, input=query)
    return response.data[0].embedding


def build_mission_filter(mission_filter: Optional[str]) -> Optional[Dict]:
    """Translate a UI mission choice into a ChromaDB metadata `where` clause."""
    if not mission_filter:
        return None
    value = mission_filter.strip().lower().replace(" ", "")
    if value in ("", "all", "allmissions", "none", "any"):
        return None
    return {"mission": value}


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve the top-k relevant chunks from ChromaDB with optional mission filter.

    Returns the raw ChromaDB query result (documents, metadatas, distances, ids).
    """
    if collection is None or not query or not query.strip():
        return None

    where_filter = build_mission_filter(mission_filter)

    query_kwargs = {
        "n_results": max(1, int(n_results)),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    try:
        # Preferred path: explicit OpenAI query embedding (same space as the stored vectors)
        query_kwargs["query_embeddings"] = [embed_query(query)]
    except Exception:
        # Fallback (e.g. no API key available): let Chroma embed with the
        # collection's own embedding function. Results may be lower quality.
        query_kwargs["query_texts"] = [query]

    return collection.query(**query_kwargs)


def _pretty(value: object, fallback: str) -> str:
    """'apollo13' -> 'Apollo 13', 'technical_transcript' -> 'Technical Transcript'."""
    text = str(value) if value not in (None, "") else fallback
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text.replace("_", " "))
    return text.title()


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved chunks into a single context string with source attribution.

    Chunks are ordered by similarity (lowest distance first) when distances are
    given, exact-duplicate texts are removed, and long chunks are truncated.
    """
    if not documents:
        return ""

    metadatas = list(metadatas or [])
    metadatas += [{}] * (len(documents) - len(metadatas))

    # Sort by score (ascending distance = most similar first) when available
    order = list(range(len(documents)))
    if distances and len(distances) == len(documents):
        order.sort(key=lambda i: distances[i])

    context_parts: List[str] = ["Retrieved NASA mission documents:"]
    seen_texts = set()
    source_index = 0

    for i in order:
        doc = (documents[i] or "").strip()
        if not doc:
            continue
        # Deduplicate identical snippets (overlapping chunks or repeated hits)
        fingerprint = " ".join(doc.split()).lower()
        if fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        source_index += 1

        metadata = metadatas[i] or {}
        mission = _pretty(metadata.get("mission"), "Unknown Mission")
        source = str(metadata.get("source") or metadata.get("file_name") or "Unknown Source")
        category = _pretty(metadata.get("document_category") or metadata.get("category"), "General")

        header = f"\n[Source {source_index}] Mission: {mission} | Document: {source} | Type: {category}"
        if distances and len(distances) == len(documents):
            header += f" | Distance: {distances[i]:.3f}"
        context_parts.append(header)

        if len(doc) > MAX_CHUNK_CHARS_IN_CONTEXT:
            doc = doc[:MAX_CHUNK_CHARS_IN_CONTEXT].rstrip() + " [...]"
        context_parts.append(doc)

    return "\n".join(context_parts)


def unique_documents(documents: List[str]) -> List[str]:
    """Return the retrieved chunks without exact duplicates, preserving order."""
    seen = set()
    result = []
    for doc in documents or []:
        key = " ".join((doc or "").split()).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(doc)
    return result
