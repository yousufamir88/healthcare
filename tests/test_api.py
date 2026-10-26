"""Tests for the support API."""

import shutil

import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.rag.knowledge_base import DEFAULT_KNOWLEDGE_BASE
from app.rag.retrieval import PolicyRetriever


@pytest.fixture
def client():
    original_knowledge = app.state.knowledge_base_dir
    original_chroma = app.state.chroma_dir
    with TestClient(app) as test_client:
        yield test_client
    app.state.knowledge_base_dir = original_knowledge
    app.state.chroma_dir = original_chroma


def test_health_reports_mock_mode(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["mock_llm"] is True
    assert body["api_key_required"] is False


def test_ask_rejects_a_blank_question(client: TestClient):
    response = client.post(
        "/ask",
        json={"session_id": "demo-session", "question": "   "},
    )

    assert response.status_code == 422


def test_ask_rejects_an_invalid_session_id(client: TestClient):
    response = client.post(
        "/ask",
        json={"session_id": "", "question": "How do I book a clinic appointment?"},
    )

    assert response.status_code == 422


def test_add_document_rejects_a_one_sentence_policy(client: TestClient):
    response = client.post(
        "/add-document",
        json={
            "doc_id": "visitor-parking",
            "topic": "visitor-parking",
            "title": "Visitor Parking",
            "text": "Only one sentence here.",
        },
    )

    assert response.status_code == 422
    assert "1 sentences" in response.json()["detail"]
    assert not (DEFAULT_KNOWLEDGE_BASE / "visitor-parking.md").exists()


def test_websocket_rejects_an_invalid_message_and_survives_disconnect(client: TestClient):
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("not-json")
        invalid_json = websocket.receive_json()
        websocket.send_json({"session_id": "", "question": "hello"})
        invalid_session = websocket.receive_json()

    assert invalid_json["error"] == "Invalid JSON message."
    assert invalid_session["error"] == "Invalid chat message."
    health = client.get("/health")
    assert health.status_code == 200


@pytest.mark.integration
def test_http_and_websocket_share_a_session(client: TestClient):
    first = client.post(
        "/ask",
        json={
            "session_id": "api-session",
            "question": "What is the status of appointment APPT-1001?",
        },
    )
    assert first.status_code == 200
    assert first.json()["appointment_id"] == "APPT-1001"

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {
                "session_id": "api-session",
                "question": "What is the consultation fee for that appointment?",
            }
        )
        second = websocket.receive_json()

    assert second["appointment_id"] == "APPT-1001"
    assert "1883" in second["answer"]


@pytest.mark.integration
def test_added_document_is_indexed_before_the_response(client: TestClient, tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    chroma_dir = tmp_path / "chroma"
    shutil.copytree(DEFAULT_KNOWLEDGE_BASE, knowledge_dir)
    app.state.knowledge_base_dir = knowledge_dir
    app.state.chroma_dir = chroma_dir

    response = client.post(
        "/add-document",
        json={
            "doc_id": "visitor-parking",
            "topic": "visitor-parking",
            "title": "Visitor Parking",
            "text": (
                "Visitors park in the blue demo lot beside the fictional clinic. "
                "This parking rule is a fabricated example for the support agent."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] is True
    assert body["collection_counts"]["sentence"] > 0
    assert (knowledge_dir / "visitor-parking.md").is_file()

    retriever = PolicyRetriever(persist_directory=chroma_dir)
    found = retriever.client().get_collection("sentence").get(
        where={"doc_id": "visitor-parking"}
    )
    assert found["ids"]
