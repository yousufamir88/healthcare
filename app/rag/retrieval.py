"""Retrieve policy chunks and answer only from that retrieved text.

This module does not call a paid model. It embeds the question with the
local SentenceTransformers model, searches the ChromaDB collection, and
either quotes the chunks that clear the measured similarity threshold or
returns an explicit fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.rag.chunking import (
    DEFAULT_CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    FIXED_SIZE_STRATEGY,
    PROJECT_ROOT,
    SENTENCE_STRATEGY,
    index_knowledge_base,
)
from app.rag.knowledge_base import KnowledgeBase

TOP_K = 3
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
FALLBACK_ANSWER = (
    "I don't have enough verified information in the available support "
    "knowledge base to answer that question."
)

# Gold labels name the policy that contains the answer. They are not
# retrieval scores. Retrieval scores are measured when the index is queried.
EVAL_QUERIES: list[dict[str, object]] = [
    {
        "query": "How do I book a clinic appointment?",
        "expected_doc_ids": ["appointment-booking"],
        "in_scope": True,
    },
    {
        "query": "What is the cancellation and rescheduling policy?",
        "expected_doc_ids": ["cancellation-rescheduling"],
        "in_scope": True,
    },
    {
        "query": "What are the consultation fees by specialty?",
        "expected_doc_ids": ["consultation-fees"],
        "in_scope": True,
    },
    {
        "query": "How does the insurance claim process work?",
        "expected_doc_ids": ["insurance-claims"],
        "in_scope": True,
    },
    {
        "query": "What is the follow-up discount?",
        "expected_doc_ids": ["follow-up-discount"],
        "in_scope": True,
    },
    {
        "query": "Who is eligible for a telemedicine video visit?",
        "expected_doc_ids": ["telemedicine"],
        "in_scope": True,
    },
    {
        "query": (
            "What is the cardiology consultation fee and can a first "
            "cardiology visit be done by video?"
        ),
        "expected_doc_ids": ["consultation-fees", "telemedicine"],
        "in_scope": True,
    },
]

OUT_OF_SCOPE_QUERIES: list[dict[str, object]] = [
    {
        "query": "Who will win the next cricket world cup?",
        "expected_doc_ids": [],
        "in_scope": False,
    },
    {
        "query": "What is the capital of France?",
        "expected_doc_ids": [],
        "in_scope": False,
    },
]


class RetrievedChunk(BaseModel):
    """One nearest chunk, with the cosine similarity measured for this query."""

    chunk_id: str
    doc_id: str
    title: str
    source: str
    strategy: str
    text: str
    similarity: float
    distance: float | None = None


class GroundedAnswer(BaseModel):
    """A support answer that quotes retrieved policy text, or the fallback."""

    query: str
    strategy: str
    threshold: float
    fallback: bool
    answer: str = Field(min_length=1)
    chunks: list[RetrievedChunk]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two embedding vectors."""
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return float(np.dot(left_vector, right_vector) / denominator)


def select_threshold(
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
) -> dict[str, object]:
    """Choose a similarity cutoff from measured top-1 scores.

    When every in-scope score is above every out-of-scope score, the cutoff
    is the midpoint of that gap. The midpoint is calculated from those
    scores. It is not a preset such as 0.5, 0.6, or 0.7.
    """
    if len(in_scope_scores) < 3:
        raise ValueError(
            "Threshold calibration needs at least 3 in-scope top-1 scores. "
            f"Received {len(in_scope_scores)}."
        )
    if len(out_of_scope_scores) < 2:
        raise ValueError(
            "Threshold calibration needs at least 2 out-of-scope top-1 scores. "
            f"Received {len(out_of_scope_scores)}."
        )

    lowest_in_scope = min(in_scope_scores)
    highest_out_of_scope = max(out_of_scope_scores)
    if lowest_in_scope > highest_out_of_scope:
        threshold = (lowest_in_scope + highest_out_of_scope) / 2
        return {
            "threshold": threshold,
            "rule": "midpoint",
            "lowest_in_scope": lowest_in_scope,
            "highest_out_of_scope": highest_out_of_scope,
            "reason": (
                "Every measured in-scope top-1 similarity is above every "
                "measured out-of-scope top-1 similarity. The threshold is "
                "the midpoint between the lowest in-scope score "
                f"({lowest_in_scope:.6f}) and the highest out-of-scope "
                f"score ({highest_out_of_scope:.6f})."
            ),
        }

    candidates = sorted(set(in_scope_scores + out_of_scope_scores), reverse=True)
    best_threshold = candidates[0]
    best_correct = -1
    for candidate in candidates:
        correct = sum(score >= candidate for score in in_scope_scores)
        correct += sum(score < candidate for score in out_of_scope_scores)
        if correct > best_correct or (
            correct == best_correct and candidate > best_threshold
        ):
            best_threshold = candidate
            best_correct = correct
    return {
        "threshold": best_threshold,
        "rule": "best-separation",
        "lowest_in_scope": lowest_in_scope,
        "highest_out_of_scope": highest_out_of_scope,
        "correct_decisions": best_correct,
        "reason": (
            "In-scope and out-of-scope top-1 scores overlap, so a midpoint "
            "would not separate them. The threshold is the observed score "
            f"{best_threshold:.6f}, chosen because it correctly labels "
            f"{best_correct} of the calibration queries. Ties keep the "
            "higher score."
        ),
    }


def score_documents(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> dict[str, object]:
    """Score retrieval after collapsing chunks to their parent documents."""
    retrieved: list[str] = list(dict.fromkeys(retrieved_doc_ids))
    expected: list[str] = list(dict.fromkeys(expected_doc_ids))
    true_positives = len(set(retrieved) & set(expected))
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    return {
        "retrieved_doc_ids": retrieved,
        "expected_doc_ids": expected,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "precision_arithmetic": (
            f"{true_positives} / {len(retrieved)} = {precision}"
        ),
        "recall_arithmetic": (
            f"{true_positives} / {len(expected)} = {recall}"
        ),
    }


def render_grounded_answer(
    hits: list[RetrievedChunk],
    threshold: float,
) -> tuple[str, bool]:
    """Quote chunks at or above the threshold, or return the fallback."""
    supported: list[str] = []
    for hit in hits:
        if hit.similarity >= threshold and hit.text not in supported:
            supported.append(hit.text)
    if not supported:
        return FALLBACK_ANSWER, True
    return " ".join(supported), False


class PolicyRetriever:
    """Local embedder plus the two Chroma collections built from the policies."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_directory = Path(persist_directory or DEFAULT_CHROMA_PATH)
        self.model_name = model_name
        self._model = None
        self._client = None

    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def client(self):
        if self._client is None:
            import chromadb

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        return self._client

    def index(self, directory: Path | None = None) -> dict[str, int]:
        """Embed both chunking strategies and upsert them into Chroma."""
        counts = index_knowledge_base(
            directory=directory,
            persist_directory=self.persist_directory,
            model_name=self.model_name,
            model=self.model(),
        )
        self._client = None
        return counts

    def embed_query(self, query: str) -> list[float]:
        text = query.strip()
        if not text:
            raise ValueError("Query text is empty.")
        vector = self.model().encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    def search(self, query: str, strategy: str, k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the top-k chunks for one collection, nearest first."""
        if k < 1:
            raise ValueError(f"k must be at least 1. Received {k}.")
        if strategy not in {FIXED_SIZE_STRATEGY, SENTENCE_STRATEGY}:
            raise ValueError(
                f"Unknown chunking strategy {strategy!r}. "
                f"Use {FIXED_SIZE_STRATEGY!r} or {SENTENCE_STRATEGY!r}."
            )
        names = {item.name for item in self.client().list_collections()}
        if strategy not in names:
            raise ValueError(
                f"Collection {strategy!r} is not indexed. "
                "Index the knowledge base before searching."
            )

        query_vector = self.embed_query(query)
        result = self.client().get_collection(strategy).query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        embeddings = result.get("embeddings")
        embedding_rows = embeddings[0] if embeddings is not None else [None] * len(ids)

        hits: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance, embedding in zip(
            ids,
            documents,
            metadatas,
            distances,
            embedding_rows,
        ):
            if embedding is None:
                similarity = 1.0 - float(distance)
            else:
                similarity = cosine_similarity(query_vector, list(embedding))
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata["doc_id"],
                    title=metadata["title"],
                    source=metadata["source"],
                    strategy=metadata["strategy"],
                    text=document,
                    similarity=similarity,
                    distance=float(distance),
                )
            )
        hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return hits

    def answer(
        self,
        query: str,
        strategy: str,
        threshold: float,
        k: int = TOP_K,
    ) -> GroundedAnswer:
        hits = self.search(query, strategy, k=k)
        text, fallback = render_grounded_answer(hits, threshold)
        return GroundedAnswer(
            query=query,
            strategy=strategy,
            threshold=threshold,
            fallback=fallback,
            answer=text,
            chunks=hits,
        )


def measure_cosine_distance_scale(persist_directory: Path) -> dict[str, float]:
    """Record how this Chroma build reports cosine distance on unit vectors."""
    import chromadb

    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    name = "cosine_distance_probe"
    existing = {item.name for item in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    identical = [1.0, 0.0, 0.0]
    collection.upsert(
        ids=["identical", "orthogonal", "opposite"],
        embeddings=[identical, [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        documents=["identical", "orthogonal", "opposite"],
    )
    result = collection.query(
        query_embeddings=[identical],
        n_results=3,
        include=["distances"],
    )
    measured = {
        doc_id: float(distance)
        for doc_id, distance in zip(result["ids"][0], result["distances"][0])
    }
    client.delete_collection(name)
    return measured


def _top_similarity(retriever: PolicyRetriever, query: str, strategy: str) -> RetrievedChunk:
    hits = retriever.search(query, strategy, k=1)
    if not hits:
        raise ValueError(f"No chunks returned for {query!r}.")
    return hits[0]


def _macro_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_evaluation(
    persist_directory: Path | None = None,
    output_directory: Path | None = None,
    k: int = TOP_K,
) -> dict[str, object]:
    """Index both collections, measure retrieval, and write the reports."""
    retriever = PolicyRetriever(persist_directory=persist_directory)
    counts = retriever.index()
    distance_scale = measure_cosine_distance_scale(retriever.persist_directory)
    strategies = [FIXED_SIZE_STRATEGY, SENTENCE_STRATEGY]

    comparison_rows: list[dict[str, object]] = []
    for item in EVAL_QUERIES:
        for strategy in strategies:
            hits = retriever.search(str(item["query"]), strategy, k=k)
            scored = score_documents(
                [hit.doc_id for hit in hits],
                list(item["expected_doc_ids"]),
            )
            comparison_rows.append(
                {
                    "query": item["query"],
                    "strategy": strategy,
                    "hits": [hit.model_dump() for hit in hits],
                    **scored,
                }
            )

    by_strategy: dict[str, list[dict[str, object]]] = {
        strategy: [row for row in comparison_rows if row["strategy"] == strategy]
        for strategy in strategies
    }
    macro: dict[str, dict[str, float]] = {}
    for strategy, rows in by_strategy.items():
        precision = sum(float(row["precision"]) for row in rows) / len(rows)
        recall = sum(float(row["recall"]) for row in rows) / len(rows)
        macro[strategy] = {
            "precision": precision,
            "recall": recall,
            "f1": _macro_f1(precision, recall),
            "queries": float(len(rows)),
        }
    recommended = _recommend_strategy(macro)

    calibration_queries = [
        item for item in EVAL_QUERIES if item["in_scope"]
    ] + OUT_OF_SCOPE_QUERIES
    calibration_rows: list[dict[str, object]] = []
    for item in calibration_queries:
        hit = _top_similarity(retriever, str(item["query"]), recommended)
        calibration_rows.append(
            {
                "query": item["query"],
                "in_scope": item["in_scope"],
                "strategy": recommended,
                "top_similarity": hit.similarity,
                "top_distance": hit.distance,
                "doc_id": hit.doc_id,
                "source": hit.source,
                "chunk_id": hit.chunk_id,
            }
        )
    threshold_decision = select_threshold(
        [
            float(row["top_similarity"])
            for row in calibration_rows
            if row["in_scope"]
        ],
        [
            float(row["top_similarity"])
            for row in calibration_rows
            if not row["in_scope"]
        ],
    )
    threshold = float(threshold_decision["threshold"])

    demonstrations: list[dict[str, object]] = []
    demo_items = list(EVAL_QUERIES[:5]) + OUT_OF_SCOPE_QUERIES[:1]
    for item in demo_items:
        grounded = retriever.answer(str(item["query"]), recommended, threshold, k=k)
        demonstrations.append(grounded.model_dump())

    report = {
        "embedding_model": retriever.model_name,
        "collection_counts": counts,
        "top_k": k,
        "cosine_distance_probe": distance_scale,
        "comparison": comparison_rows,
        "macro": macro,
        "recommended_strategy": recommended,
        "recommendation_reason": _recommendation_reason(macro, recommended),
        "calibration": calibration_rows,
        "threshold": threshold_decision,
        "demonstrations": demonstrations,
        "document_count": len(KnowledgeBase().load()),
    }
    if output_directory is not None:
        _write_reports(Path(output_directory), report)
    return report


def _recommend_strategy(macro: dict[str, dict[str, float]]) -> str:
    def sort_key(strategy: str) -> tuple[float, float, float, int]:
        scores = macro[strategy]
        sentence_bonus = 1 if strategy == SENTENCE_STRATEGY else 0
        return (scores["f1"], scores["recall"], scores["precision"], sentence_bonus)

    return max(macro, key=sort_key)


def _recommendation_reason(
    macro: dict[str, dict[str, float]],
    recommended: str,
) -> str:
    other = (
        FIXED_SIZE_STRATEGY
        if recommended == SENTENCE_STRATEGY
        else SENTENCE_STRATEGY
    )
    chosen = macro[recommended]
    alternate = macro[other]
    tied = (
        chosen["f1"] == alternate["f1"]
        and chosen["recall"] == alternate["recall"]
        and chosen["precision"] == alternate["precision"]
    )
    summary = (
        f"{recommended} has macro F1 {chosen['f1']:.4f} "
        f"(precision {chosen['precision']:.4f}, recall {chosen['recall']:.4f}) "
        f"compared with {other} macro F1 {alternate['f1']:.4f} "
        f"(precision {alternate['precision']:.4f}, recall {alternate['recall']:.4f}). "
        "Each macro score is the unweighted mean across the evaluation queries. "
        "F1 is the harmonic mean of that macro precision and macro recall."
    )
    if tied and recommended == SENTENCE_STRATEGY:
        return (
            summary
            + " The measured scores are tied, so sentence chunking is selected "
            "because each chunk is one complete policy sentence."
        )
    return summary


def _write_reports(output_directory: Path, report: dict[str, object]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "retrieval_comparison.json").write_text(
        json.dumps(
            {
                "top_k": report["top_k"],
                "macro": report["macro"],
                "recommended_strategy": report["recommended_strategy"],
                "recommendation_reason": report["recommendation_reason"],
                "rows": [
                    {
                        key: row[key]
                        for key in (
                            "query",
                            "strategy",
                            "expected_doc_ids",
                            "retrieved_doc_ids",
                            "true_positives",
                            "precision",
                            "recall",
                            "precision_arithmetic",
                            "recall_arithmetic",
                        )
                    }
                    for row in report["comparison"]
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "retrieval_threshold_report.json").write_text(
        json.dumps(
            {
                "embedding_model": report["embedding_model"],
                "collection_counts": report["collection_counts"],
                "cosine_distance_probe": report["cosine_distance_probe"],
                "recommended_strategy": report["recommended_strategy"],
                "threshold": report["threshold"],
                "measurements": report["calibration"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "retrieval_comparison.md").write_text(
        _comparison_markdown(report),
        encoding="utf-8",
    )
    (output_directory / "rag_demo_transcript.txt").write_text(
        _transcript(report),
        encoding="utf-8",
    )


def _comparison_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Retrieval comparison",
        "",
        "Document-level precision and recall for the two chunking strategies.",
        "Multiple chunks from the same parent document count once.",
        "",
        f"Top-k chunks: {report['top_k']}",
        "",
        "| Query | Strategy | Expected | Retrieved | TP | Precision | Recall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["comparison"]:
        lines.append(
            "| {query} | {strategy} | {expected} | {retrieved} | {tp} | {precision:.4f} | {recall:.4f} |".format(
                query=row["query"],
                strategy=row["strategy"],
                expected=", ".join(row["expected_doc_ids"]),
                retrieved=", ".join(row["retrieved_doc_ids"]),
                tp=row["true_positives"],
                precision=row["precision"],
                recall=row["recall"],
            )
        )
    lines.extend(["", "## Arithmetic", ""])
    for row in report["comparison"]:
        lines.append(f"- **{row['strategy']}** — {row['query']}")
        lines.append(f"  - Precision: {row['precision_arithmetic']}")
        lines.append(f"  - Recall: {row['recall_arithmetic']}")
    lines.extend(["", "## Macro scores", ""])
    for strategy, scores in report["macro"].items():
        lines.append(
            f"- {strategy}: precision {scores['precision']:.4f}, "
            f"recall {scores['recall']:.4f}, F1 {scores['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Use **{report['recommended_strategy']}**.",
            "",
            str(report["recommendation_reason"]),
            "",
        ]
    )
    return "\n".join(lines)


def _transcript(report: dict[str, object]) -> str:
    threshold = report["threshold"]["threshold"]
    lines = [
        "Practo Domain Support Agent RAG demonstration",
        "Mode: MOCK_LLM",
        "Policies: fictional clinic policies, not official Practo policy and not real patient records.",
        f"Retrieval strategy: {report['recommended_strategy']}",
        f"Calibrated similarity threshold: {threshold:.6f}",
        f"Threshold rule: {report['threshold']['reason']}",
        "",
    ]
    for item in report["demonstrations"]:
        lines.append(f"Query: {item['query']}")
        lines.append(f"Retrieval strategy: {item['strategy']}")
        lines.append("Retrieved chunks:")
        for chunk in item["chunks"]:
            lines.append(
                "- {doc_id} ({source}) chunk_id={chunk_id} "
                "similarity={similarity:.6f} distance={distance:.6f}".format(
                    doc_id=chunk["doc_id"],
                    source=chunk["source"],
                    chunk_id=chunk["chunk_id"],
                    similarity=chunk["similarity"],
                    distance=chunk["distance"],
                )
            )
        lines.append(f"Final answer: {item['answer']}")
        lines.append(f"Fallback triggered: {str(item['fallback']).lower()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    report = run_evaluation(
        persist_directory=DEFAULT_CHROMA_PATH,
        output_directory=EVALUATION_DIR,
    )
    print(_transcript(report))
    print("Collection counts:")
    for name, count in report["collection_counts"].items():
        print(f"  {name}: {count}")
    print(f"Recommended strategy: {report['recommended_strategy']}")
    print(f"Threshold: {report['threshold']['threshold']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
