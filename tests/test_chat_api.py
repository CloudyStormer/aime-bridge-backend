from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, chat_store
from app.services.chat_store import ChatStore


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_send_message_returns_assistant_message(tmp_path: Path) -> None:
    app.dependency_overrides.clear()
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    client = TestClient(app)

    try:
        response = client.post(
            "/api/chat/message",
            json={"content": "今天有点累", "kind": "text"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["message"]["role"] == "assistant"
        assert body["message"]["kind"] == "text"
        assert body["message"]["content"]

        history = client.get("/api/chat/history").json()["messages"]
        assert [item["role"] for item in history] == ["wife", "assistant"]
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_ai_chat_compatible_endpoint() -> None:
    client = TestClient(app)

    response = client.post("/ai/chat", json={"user_id": "local", "message": "陪我聊一会"})

    assert response.status_code == 200
    assert response.json()["reply"]


def test_training_message_is_separate_from_chat_history(tmp_path: Path) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    client = TestClient(app)

    try:
        response = client.post(
            "/api/training/message",
            json={"content": "以后少说模板话，像我本人一点", "kind": "text"},
        )

        assert response.status_code == 200
        assert response.json()["message"]["mode"] == "training"
        assert client.get("/api/chat/history").json()["messages"] == []
        assert len(client.get("/api/training/history").json()["messages"]) == 2
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_conversation_review_filters_by_time_range(tmp_path: Path) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    client = TestClient(app)

    try:
        client.post("/api/chat/message", json={"content": "今天有点烦", "kind": "text"})

        response = client.post(
            "/api/conversation-review",
            json={
                "startAt": "2020-01-01T00:00:00Z",
                "endAt": "2099-01-01T00:00:00Z",
                "modes": ["chat"],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["stats"]["totalMessages"] == 2
        assert body["stats"]["wifeMessages"] == 1
        assert body["stats"]["assistantMessages"] == 1
        assert body["summary"]
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_chat_review_summary_endpoint_matches_frontend_contract(tmp_path: Path) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    client = TestClient(app)

    try:
        client.post("/api/chat/message", json={"content": "今天想被好好陪一会", "kind": "text"})

        response = client.get(
            "/api/chat/review",
            params={"startDate": "2020-01-01", "endDate": "2099-01-01"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["title"]
        assert body["rangeLabel"] == "2020-01-01 至 2099-01-01"
        assert body["aiSummary"]
        assert body["wifeSummary"]
        assert body["messageCount"] == 2
        assert body["moments"]
        assert body["suggestions"]
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max
