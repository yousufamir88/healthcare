import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.rag.knowledge_base import (
    PROJECT_ROOT,
    KnowledgeBase,
    PolicyDocument,
)


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FIXED_CHUNK_SIZE = 100
FIXED_CHUNK_OVERLAP = 20

FIXED_SIZE_STRATEGY = "fixed_size_overlap"
SENTENCE_STRATEGY = "sentence"

DEFAULT_CHROMA_PATH = PROJECT_ROOT / "chroma_db"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class TextChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    topic: str
    title: str
    source: str
    strategy: str
    position: int = Field(ge=0)
    text: str = Field(min_length=1)


def fixed_size_chunks(
    text: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        elif normalized[next_start] != " ":
            previous_space = normalized.rfind(" ", start + 1, next_start)
            if previous_space != -1:
                next_start = previous_space + 1

        while next_start < length and normalized[next_start] == " ":
            next_start += 1

        if next_start <= start:
            break
        start = next_start

    return chunks


def sentence_chunks(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    return [
        sentence
        for sentence in _SENTENCE_BOUNDARY.split(normalized)
        if sentence
    ]


def chunk_documents(
    documents: list[PolicyDocument],
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> dict[str, list[TextChunk]]:
    return {
        FIXED_SIZE_STRATEGY: _chunk_with(
            documents,
            FIXED_SIZE_STRATEGY,
            lambda text: fixed_size_chunks(text, chunk_size, overlap),
        ),
        SENTENCE_STRATEGY: _chunk_with(
            documents,
            SENTENCE_STRATEGY,
            sentence_chunks,
        ),
    }


def embed_chunks(chunks: list[TextChunk], model) -> list[list[float]]:
    if not chunks:
        return []

    vectors = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = vectors.tolist()
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")
    return embeddings


def index_knowledge_base(
    directory: Path | None = None,
    persist_directory: Path | None = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
    model=None,
) -> dict[str, int]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    documents = KnowledgeBase(directory).load()
    grouped = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if model is None:
        model = SentenceTransformer(model_name)
    persist_directory = Path(persist_directory or DEFAULT_CHROMA_PATH)
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))

    counts: dict[str, int] = {}
    for strategy, chunks in grouped.items():
        if not chunks:
            raise ValueError(f"No chunks produced for {strategy}.")

        embeddings = embed_chunks(chunks, model)
        collection = _open_collection(
            client,
            strategy,
            _collection_metadata(
                strategy,
                model_name,
                chunk_size,
                overlap,
            ),
        )
        _upsert_chunks(collection, chunks, embeddings)
        counts[strategy] = collection.count()

    return counts


def _chunk_with(documents, strategy, splitter) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        pieces = splitter(document.text)
        chunks.extend(
            TextChunk(
                chunk_id=f"{document.doc_id}:{strategy}:{position}",
                doc_id=document.doc_id,
                topic=document.topic,
                title=document.title,
                source=document.source,
                strategy=strategy,
                position=position,
                text=piece,
            )
            for position, piece in enumerate(pieces)
        )
    return chunks


def _collection_metadata(strategy, model_name, chunk_size, overlap):
    metadata = {
        "hnsw:space": "cosine",
        "strategy": strategy,
        "embedding_model": model_name,
    }
    if strategy == FIXED_SIZE_STRATEGY:
        metadata["chunk_size"] = chunk_size
        metadata["overlap"] = overlap
    return metadata


def _open_collection(client, name: str, metadata: dict):
    existing = {
        collection.name for collection in client.list_collections()
    }
    if name in existing:
        client.delete_collection(name)
    return client.get_or_create_collection(name=name, metadata=metadata)


def _upsert_chunks(collection, chunks: list[TextChunk], embeddings) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk ids must be unique within a collection.")

    collection.upsert(
        ids=ids,
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "doc_id": chunk.doc_id,
                "topic": chunk.topic,
                "title": chunk.title,
                "source": chunk.source,
                "strategy": chunk.strategy,
                "position": chunk.position,
            }
            for chunk in chunks
        ],
    )


if __name__ == "__main__":
    indexed = index_knowledge_base()
    print("Indexed policy chunks.")
    for name, count in indexed.items():
        print(f"{name}: {count}")
