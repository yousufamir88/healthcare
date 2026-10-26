"""FastAPI routes for health, questions, new policies, and websocket chat.

Graded calls go through ``ask()``, which keeps MOCK_LLM, session memory,
and the guardrails. A disconnected websocket client is caught so the
server keeps running.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.api.documents import AddDocumentRequest, DocumentRejected, add_policy_document
from app.config import get_settings
from app.rag.chunking import DEFAULT_CHROMA_PATH
from app.rag.knowledge_base import DEFAULT_KNOWLEDGE_BASE
from app.session.memory import ask
from app.session.schema import SupportResponse

get_settings()

app = FastAPI(
    title="Practo Domain Support Agent",
    version="0.1.0",
)
app.state.knowledge_base_dir = DEFAULT_KNOWLEDGE_BASE
app.state.chroma_dir = DEFAULT_CHROMA_PATH


class AskRequest(BaseModel):
    """One support question in a named session."""

    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    question: str = Field(min_length=1)

    @field_validator("session_id", "question")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank.")
        return cleaned


@app.get("/health")
def health() -> dict[str, object]:
    """Report that the process is in the keyless demo mode."""
    settings = get_settings()
    return {
        "status": "ready",
        "mock_llm": settings.mock_llm,
        "api_key_required": settings.api_key_required,
    }


@app.post("/ask", response_model=SupportResponse)
def ask_question(body: AskRequest) -> SupportResponse:
    """Answer one question and keep the result in that session."""
    try:
        return ask(body.session_id, body.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/add-document")
def add_document(body: AddDocumentRequest) -> dict[str, object]:
    """Store a fictional policy and index it before returning."""
    try:
        return add_policy_document(
            body,
            directory=app.state.knowledge_base_dir,
            persist_directory=app.state.chroma_dir,
        )
    except DocumentRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    """Multi-turn chat. One client leaving does not stop the server."""
    await websocket.accept()
    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                await websocket.send_json({"error": "Invalid JSON message."})
                continue
            try:
                request = AskRequest.model_validate(payload)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "error": "Invalid chat message.",
                        "details": exc.errors(),
                    }
                )
                continue
            try:
                response = await asyncio.to_thread(
                    ask,
                    request.session_id,
                    request.question,
                )
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
                continue
            await websocket.send_json(response.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
