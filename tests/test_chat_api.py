from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app, chat_store
from app.services.ai_service import ai_service
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


def test_daily_chat_uses_training_memory(tmp_path: Path, monkeypatch) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 0
    captured = {}
    client = TestClient(app)

    def fake_reply(user_message, history, training_memory=None, image_url=None):
        captured["user_message"] = user_message
        captured["history"] = history
        captured["training_memory"] = training_memory or []
        captured["image_url"] = image_url
        return "我会按训练后的方式回复。"

    monkeypatch.setattr(ai_service, "reply", fake_reply)

    try:
        chat_store.append_user_message(
            content="以后少讲大道理，先接住情绪，再给一句很短的建议。",
            kind="text",
            mode="training",
        )
        chat_store.append_assistant_message("收到，我会把这个作为日常聊天的表达规则。", mode="training")

        response = client.post(
            "/api/chat/message",
            json={"content": "今天有点低落", "kind": "text"},
        )

        assert response.status_code == 200
        assert captured["user_message"] == "今天有点低落"
        assert any("少讲大道理" in item["content"] for item in captured["training_memory"])
        assert response.json()["message"]["content"] == "我会按训练后的方式回复。"
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_image_message_upload_is_saved_and_sent_to_ai(tmp_path: Path, monkeypatch) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(main_module, "IMAGE_UPLOAD_DIR", tmp_path / "uploads" / "images")
    captured = {}
    client = TestClient(app)

    def fake_reply(user_message, history, training_memory=None, image_url=None):
        captured["user_message"] = user_message
        captured["history"] = history
        captured["image_url"] = image_url
        return "我看到了这张图，会结合上下文继续说。"

    monkeypatch.setattr(ai_service, "reply", fake_reply)

    try:
        response = client.post(
            "/api/chat/message",
            data={"kind": "image", "content": "这张图也放进上下文。"},
            files={"image": ("photo.jpg", b"fake-image-bytes", "image/jpeg")},
        )

        assert response.status_code == 200
        assert captured["image_url"].endswith(".jpg")
        assert "/uploads/images/" in captured["image_url"]
        history = client.get("/api/chat/history").json()["messages"]
        assert history[0]["kind"] == "image"
        assert history[0]["imageUrl"] == captured["image_url"]
        assert (tmp_path / "uploads" / "images").exists()
        assert response.json()["message"]["content"] == "我看到了这张图，会结合上下文继续说。"
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_training_memory_prompt_shapes_daily_ai_identity() -> None:
    prompt = ai_service._training_memory_prompt(
        [
            {
                "role": "user",
                "content": "我今天很累，但我和她说话时会先接住情绪，再说自己的真实想法。",
            },
            {
                "role": "assistant",
                "content": "我会把这条沉淀成日常聊天里的身份、情绪和表达习惯。",
            },
        ]
    )
    training_prompt = ai_service._training_system_prompt()

    assert "你就是训练者" in prompt
    assert "身份延伸" in prompt
    assert "经历和心情可以作为你的经历和心情" in prompt
    assert "尽最大可能变成训练者本人" in training_prompt
    assert "以训练者的身份去回应用户" in training_prompt


def test_daily_reply_removes_stage_directions() -> None:
    text = ai_service._clean_daily_reply("（轻轻抱住你，低声说）别怕，我在。")

    assert text == "别怕，我在。"


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
        assert body["overview"]
        assert body["coreEvents"]
        assert body["userEmotionExpressions"]
        assert body["emotionalTrend"]
        assert body["aiResponsePattern"]
        assert body["importantDialogues"]
        assert body["followUpSuggestions"]
        assert body["aiSummary"]
        assert body["wifeSummary"]
        assert body["messageCount"] == 2
        assert body["moments"]
        assert body["suggestions"]
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max


def test_chat_review_follow_up_uses_selected_range(tmp_path: Path, monkeypatch) -> None:
    original_path = chat_store._file_path
    original_max = chat_store._max_history_messages
    chat_store._file_path = tmp_path / "chat.json"
    chat_store._max_history_messages = 20
    client = TestClient(app)
    captured = {}

    def fake_follow_up(messages, start_at, end_at, instruction):
        captured["messages"] = messages
        captured["instruction"] = instruction
        return "这段重点是她表达了累，我主要在接住情绪。"

    monkeypatch.setattr(ai_service, "review_follow_up", fake_follow_up)

    try:
        client.post("/api/chat/message", json={"content": "今天真的有点累，也有点委屈", "kind": "text"})

        response = client.post(
            "/api/chat/review/ask",
            json={
                "startDate": "2020-01-01",
                "endDate": "2099-01-01",
                "instruction": "只看情绪变化",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "这段重点是她表达了累，我主要在接住情绪。"
        assert body["rangeLabel"] == "2020-01-01 至 2099-01-01"
        assert body["relatedDialogues"]
        assert captured["instruction"] == "只看情绪变化"
        assert any("有点累" in item["content"] for item in captured["messages"])
    finally:
        chat_store._file_path = original_path
        chat_store._max_history_messages = original_max
