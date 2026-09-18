"""RAG ask orchestrator: screen -> embed -> ABAC-filtered retrieval -> citations.

Retrieval-only: no LLM call. The ABAC predicate is applied inside the SQL
query (pre-filter, deny-by-default), similarity is cosine in Python over the
permitted rows, and every retrieved chunk is re-scanned by the Prompt-Guard
classifier before it can enter the context (indirect-injection defense).
Audit rows go to ``flags.jsonl`` and only ever contain ids/counts/labels —
never chunk text or patient names.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from guard.abac import POLICY_VERSION, retrieval_predicate, subject_attributes
from guard.db import Chunk, Document, User
from guard.pipeline import DEFAULT_FLAG_LOG, screen
from guard.steps.embedding import embed
from guard.steps.prompt_guard import PromptGuardVerdict, classify

logger = logging.getLogger("guard.rag")

EVENT_RAG_QUERY = "RAG_QUERY"
RULE_RAG_INJECTION = "RAG_CONTEXT_INJECTION"
DEFAULT_TOP_K = 5

REJECT_MESSAGE = "Your request was blocked: jailbreak or prompt-injection detected."
EMPTY_MESSAGE = "no relevant permitted content found"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    ordinal: int
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class Citation:
    document_id: int
    chunk_id: int
    title: str
    score: float


@dataclass(frozen=True)
class RetrievalOutcome:
    """Result of the shared ABAC-filtered retrieval core (no screening, no audit row)."""

    chunks: list[RetrievedChunk]
    permitted: int
    dropped_chunk_ids: list[int]
    embedding_engine: str | None
    engine_mismatch: bool
    assembled_context: str
    citations: list[Citation]


@dataclass(frozen=True)
class AskResult:
    disposition: str
    message: str
    chunks: list[RetrievedChunk]
    assembled_context: str | None
    citations: list[Citation]
    policy_version: str
    embedding_engine: str | None
    engine_mismatch: bool


def get_top_k() -> int:
    return int(os.environ.get("GUARD_RAG_TOP_K", str(DEFAULT_TOP_K)))


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _permitted_rows(session: Session, predicate) -> list:
    statement: Select = (
        select(
            Chunk.id,
            Chunk.ordinal,
            Chunk.text,
            Chunk.embedding,
            Chunk.embedding_engine,
            Document.id.label("document_id"),
            Document.title,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(predicate)
    )
    return list(session.execute(statement))


def _flag_log_path() -> Path:
    return Path(os.environ.get("GUARD_FLAG_LOG", DEFAULT_FLAG_LOG))


def _append_audit(row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    path = _flag_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("rag | audit row %s -> %s", row.get("event"), path)


def _scan_chunk(row) -> PromptGuardVerdict:
    """Indirect-injection scan of one retrieved chunk (label + score only in logs)."""
    return classify(row.text)


def retrieve(
    session: Session,
    user: User,
    query_text: str,
    top_k: int | None = None,
) -> RetrievalOutcome:
    """Embed ``query_text`` (already masked by the caller) and retrieve chunks.

    Shared by ``ask()`` and the unified chat orchestrator. The ABAC predicate
    applies to ``user``'s DB row (deny-by-default), similarity is cosine over
    permitted rows, and every candidate chunk is re-scanned by the Prompt-Guard
    classifier (flagged chunks are dropped and audited). Does not write a
    ``RAG_QUERY`` audit row - callers own their own auditing.
    """
    if top_k is None:
        top_k = get_top_k()

    vector, engine, _model = embed(query_text)
    subject = subject_attributes(user)
    predicate = retrieval_predicate(subject)

    rows = _permitted_rows(session, predicate)
    permitted = len(rows)
    engine_rows = [row for row in rows if row.embedding_engine == engine and row.embedding]
    engine_mismatch = permitted > 0 and not engine_rows

    scored = [(row, _cosine(vector, row.embedding)) for row in engine_rows]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    kept = []
    dropped_chunk_ids: list[int] = []
    for row, score in scored[:top_k]:
        verdict = _scan_chunk(row)
        if verdict.flagged:
            dropped_chunk_ids.append(row.id)
            _append_audit(
                {
                    "event": RULE_RAG_INJECTION,
                    "document_id": row.document_id,
                    "chunk_id": row.id,
                    "label": verdict.label,
                    "suspicious_score": verdict.suspicious_score,
                }
            )
            logger.info(
                "rag | dropped chunk %d from context: injection suspected "
                "(label=%s score=%.4f engine=%s)",
                row.id,
                verdict.label,
                verdict.suspicious_score,
                verdict.engine,
            )
            continue
        kept.append((row, score))

    chunks = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            ordinal=row.ordinal,
            title=row.title,
            text=row.text,
            score=round(score, 4),
        )
        for row, score in kept
    ]
    assembled = "\n\n".join(
        f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks, start=1)
    )
    citations = [
        Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            score=chunk.score,
        )
        for chunk in chunks
    ]
    return RetrievalOutcome(
        chunks=chunks,
        permitted=permitted,
        dropped_chunk_ids=dropped_chunk_ids,
        embedding_engine=engine,
        engine_mismatch=engine_mismatch,
        assembled_context=assembled,
        citations=citations,
    )


def ask(
    session: Session,
    user: User,
    question: str,
    top_k: int | None = None,
) -> AskResult:
    """Answer one question with permitted, injection-screened chunks."""
    guarded = screen(question)
    if guarded.disposition == "REJECT":
        return AskResult(
            disposition="REJECT",
            message=REJECT_MESSAGE,
            chunks=[],
            assembled_context=None,
            citations=[],
            policy_version=POLICY_VERSION,
            embedding_engine=None,
            engine_mismatch=False,
        )

    query_text = guarded.masked_prompt or question
    outcome = retrieve(session, user, query_text, top_k)

    message = (
        f"Retrieved {len(outcome.chunks)} permitted chunk(s); context assembled with citations."
        if outcome.chunks
        else EMPTY_MESSAGE
    )
    if outcome.engine_mismatch:
        message += (
            f" [engine mismatch: query embedded with '{outcome.embedding_engine}' but no permitted "
            "chunks share that engine; re-ingest to re-index]"
        )

    subject = subject_attributes(user)
    _append_audit(
        {
            "event": EVENT_RAG_QUERY,
            "username": user.username,
            "role": subject.get("role"),
            "policy_version": POLICY_VERSION,
            "permitted_chunks": outcome.permitted,
            "top_chunk_ids": [chunk.chunk_id for chunk in outcome.chunks],
            "embedding_engine": outcome.embedding_engine,
        }
    )
    logger.info(
        "rag | ask complete: disposition=%s permitted=%d kept=%d engine=%s mismatch=%s",
        guarded.disposition,
        outcome.permitted,
        len(outcome.chunks),
        outcome.embedding_engine,
        outcome.engine_mismatch,
    )
    return AskResult(
        disposition=guarded.disposition,
        message=message,
        chunks=outcome.chunks,
        assembled_context=outcome.assembled_context,
        citations=outcome.citations,
        policy_version=POLICY_VERSION,
        embedding_engine=outcome.embedding_engine,
        engine_mismatch=outcome.engine_mismatch,
    )
