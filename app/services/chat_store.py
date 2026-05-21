import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.schemas import ChatMessage, ConversationMode, MessageKind


class ChatStore:
    def __init__(self, file_path: str, max_history_messages: int) -> None:
        self._file_path = Path(file_path)
        self._max_history_messages = max_history_messages
        self._lock = Lock()

    def history(
        self,
        mode: ConversationMode | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> list[ChatMessage]:
        with self._lock:
            messages = [ChatMessage.model_validate(item) for item in self._read()]
        return [
            item
            for item in messages
            if (mode is None or item.mode == mode)
            and (start_at is None or item.createdAt >= self._aware(start_at))
            and (end_at is None or item.createdAt <= self._aware(end_at))
        ]

    def history_page(
        self,
        mode: ConversationMode,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[ChatMessage]:
        messages = self.history(mode=mode)
        if before is not None:
            before_at = self._aware(before)
            messages = [item for item in messages if item.createdAt < before_at]
        return messages[-limit:]

    def append_user_message(
        self,
        content: str,
        kind: MessageKind,
        mode: ConversationMode = "chat",
        duration_seconds: float | None = None,
        image_url: str | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=self._new_id(),
            role="wife",
            kind=kind,
            content=content,
            createdAt=self._now(),
            mode=mode,
            durationSeconds=duration_seconds,
            imageUrl=image_url,
        )
        self.append(message)
        return message

    def append_assistant_message(self, content: str, mode: ConversationMode = "chat") -> ChatMessage:
        message = ChatMessage(
            id=self._new_id(),
            role="assistant",
            kind="text",
            content=content,
            createdAt=self._now(),
            mode=mode,
        )
        self.append(message)
        return message

    def append(self, message: ChatMessage) -> None:
        with self._lock:
            messages = self._read()
            messages.append(message.model_dump(mode="json", exclude_none=True))
            if self._max_history_messages > 0:
                messages = messages[-self._max_history_messages :]
            self._write(messages)

    def recent_for_ai(self, mode: ConversationMode = "chat", limit: int = 80) -> list[dict]:
        messages = self.history(mode=mode)[-limit:]
        role_map = {"wife": "user", "assistant": "assistant"}
        return [
            {
                "role": role_map[item.role],
                "content": item.content,
                **({"imageUrl": item.imageUrl} if item.imageUrl else {}),
            }
            for item in messages
            if item.content.strip() or item.imageUrl
        ]

    def recent_training_directives(self, limit: int = 120) -> list[dict]:
        messages = [item for item in self.history(mode="training") if item.role == "wife"]
        return [
            {
                "role": "user",
                "content": item.content,
                **({"imageUrl": item.imageUrl} if item.imageUrl else {}),
            }
            for item in messages[-limit:]
            if item.content.strip() or item.imageUrl
        ]

    def review_messages(
        self,
        start_at: datetime,
        end_at: datetime,
        modes: list[ConversationMode],
    ) -> list[ChatMessage]:
        selected_modes = set(modes)
        return [
            item
            for item in self.history(start_at=start_at, end_at=end_at)
            if item.mode in selected_modes
        ]

    def _read(self) -> list[dict]:
        if not self._file_path.exists():
            return []
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return data

    def _write(self, messages: list[dict]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _new_id(self) -> str:
        return uuid4().hex

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _aware(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
