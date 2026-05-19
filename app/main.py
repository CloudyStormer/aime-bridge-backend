from datetime import date, datetime, time, timedelta, timezone

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    AIStatusResponse,
    ChatRequest,
    ChatResponse,
    ConversationReviewRequest,
    ConversationReviewResponse,
    ConversationReviewStats,
    ChatHistoryResponse,
    ConversationMode,
    ReviewSummaryResponse,
    SendChatMessageRequest,
    SendChatMessageResponse,
    TrainingChatRequest,
)
from app.services.ai_service import ai_service
from app.services.chat_store import ChatStore


app = FastAPI(title=settings.app_name)
chat_store = ChatStore(
    file_path=settings.chat_store_path,
    max_history_messages=settings.chat_max_history_messages,
)
LOCAL_TIMEZONE = timezone(timedelta(hours=8))

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.get("/ai/status", response_model=AIStatusResponse)
def ai_status() -> dict:
    return ai_service.runtime_status()


@app.post("/ai/chat", response_model=ChatResponse)
def ai_chat(payload: ChatRequest) -> ChatResponse:
    reply = ai_service.chat(
        user_message=payload.message.strip(),
        history=chat_store.recent_for_ai(mode="chat"),
    )
    return ChatResponse(reply=reply)


@app.post("/ai/training-chat", response_model=ChatResponse)
def ai_training_chat(payload: TrainingChatRequest) -> ChatResponse:
    reply = ai_service.training_reply(
        user_message=payload.message.strip(),
        history=chat_store.recent_for_ai(mode="training"),
    )
    return ChatResponse(reply=reply)


@app.get("/api/chat/history", response_model=ChatHistoryResponse)
def chat_history() -> ChatHistoryResponse:
    return ChatHistoryResponse(messages=chat_store.history(mode="chat"))


@app.post("/api/chat/message", response_model=SendChatMessageResponse)
async def send_chat_message(request: Request) -> SendChatMessageResponse:
    message = await _send_message_for_mode(request=request, mode="chat")
    return SendChatMessageResponse(message=message)


@app.get("/api/training/history", response_model=ChatHistoryResponse)
def training_history() -> ChatHistoryResponse:
    return ChatHistoryResponse(messages=chat_store.history(mode="training"))


@app.post("/api/training/message", response_model=SendChatMessageResponse)
async def send_training_message(request: Request) -> SendChatMessageResponse:
    message = await _send_message_for_mode(request=request, mode="training")
    return SendChatMessageResponse(message=message)


@app.post("/api/conversation-review", response_model=ConversationReviewResponse)
def conversation_review(payload: ConversationReviewRequest) -> ConversationReviewResponse:
    messages = chat_store.review_messages(
        start_at=payload.startAt,
        end_at=payload.endAt,
        modes=payload.modes,
    )
    review_items = [
        {
            "role": item.role,
            "speaker": "老婆" if item.role == "wife" else "我",
            "kind": item.kind,
            "content": item.content,
            "createdAt": item.createdAt.isoformat(),
        }
        for item in messages
    ]
    summary = ai_service.conversation_review(
        messages=review_items,
        start_at=payload.startAt.isoformat(),
        end_at=payload.endAt.isoformat(),
    )
    stats = ConversationReviewStats(
        totalMessages=len(messages),
        wifeMessages=sum(1 for item in messages if item.role == "wife"),
        assistantMessages=sum(1 for item in messages if item.role == "assistant"),
        voiceMessages=sum(1 for item in messages if item.kind == "voice"),
    )
    return ConversationReviewResponse(
        startAt=payload.startAt,
        endAt=payload.endAt,
        modes=payload.modes,
        stats=stats,
        summary=summary,
        messages=messages,
    )


@app.post("/api/reviews/conversation", response_model=ConversationReviewResponse)
def conversation_review_alias(payload: ConversationReviewRequest) -> ConversationReviewResponse:
    return conversation_review(payload)


@app.get("/api/chat/review", response_model=ReviewSummaryResponse)
def chat_review_summary(
    startDate: date = Query(...),
    endDate: date = Query(...),
) -> ReviewSummaryResponse:
    start_at, end_at = _local_date_range_to_utc(startDate=startDate, endDate=endDate)
    messages = chat_store.review_messages(
        start_at=start_at,
        end_at=end_at,
        modes=["chat"],
    )
    return _build_review_summary(messages=messages, start_date=startDate, end_date=endDate)


async def _send_message_for_mode(request: Request, mode: ConversationMode):
    payload = await _parse_send_message_request(request)
    content = payload.content.strip()

    chat_store.append_user_message(
        content=content,
        kind=payload.kind,
        mode=mode,
        duration_seconds=payload.durationSeconds,
    )
    if mode == "training":
        reply = ai_service.training_reply(
            user_message=content,
            history=chat_store.recent_for_ai(mode="training"),
        )
    else:
        reply = ai_service.reply(
            user_message=content,
            history=chat_store.recent_for_ai(mode="chat"),
        )
    return chat_store.append_assistant_message(reply, mode=mode)


async def _parse_send_message_request(request: Request) -> SendChatMessageRequest:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        kind = str(form.get("kind") or "text")
        text = str(form.get("content") or "").strip()
        duration_value = form.get("durationSeconds")
        duration_seconds = float(duration_value) if duration_value not in (None, "") else None
        audio = form.get("audio")
        if kind == "voice" and not text:
            filename = getattr(audio, "filename", "") or "voice message"
            text = f"[收到一条语音消息：{filename}]"
        return SendChatMessageRequest(
            content=text,
            kind="voice" if kind == "voice" else "text",
            durationSeconds=duration_seconds,
        )

    return SendChatMessageRequest.model_validate(await request.json())


def _local_date_range_to_utc(startDate: date, endDate: date) -> tuple[datetime, datetime]:
    start = min(startDate, endDate)
    end = max(startDate, endDate)
    start_local = datetime.combine(start, time.min, tzinfo=LOCAL_TIMEZONE)
    end_local = datetime.combine(end, time.max, tzinfo=LOCAL_TIMEZONE)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _build_review_summary(
    messages: list,
    start_date: date,
    end_date: date,
) -> ReviewSummaryResponse:
    wife_messages = [item for item in messages if item.role == "wife"]
    assistant_messages = [item for item in messages if item.role == "assistant"]
    voice_count = sum(1 for item in messages if item.kind == "voice")
    range_label = f"{min(start_date, end_date).isoformat()} 至 {max(start_date, end_date).isoformat()}"

    return ReviewSummaryResponse(
        title="这段时间的对话回顾",
        rangeLabel=range_label,
        aiSummary=_summarize_side(
            messages=assistant_messages,
            empty_text="这段时间里 AI 还没有留下回复。",
            speaker="AI",
        ),
        wifeSummary=_summarize_side(
            messages=wife_messages,
            empty_text="这段时间里还没有她的消息。",
            speaker="她",
        ),
        moments=_build_review_moments(
            messages=messages,
            wife_count=len(wife_messages),
            assistant_count=len(assistant_messages),
            voice_count=voice_count,
        ),
        suggestions=_build_review_suggestions(messages=messages, voice_count=voice_count),
        messageCount=len(messages),
    )


def _summarize_side(messages: list, empty_text: str, speaker: str) -> str:
    if not messages:
        return empty_text

    samples = [item.content.strip() for item in messages if item.content.strip()][:3]
    sample_text = "；".join(samples)
    if not sample_text:
        return f"{speaker}这段时间主要留下了 {len(messages)} 条非文字消息，可以结合语音和时间点继续回看。"
    return f"{speaker}这段时间共有 {len(messages)} 条消息，重点围绕“{sample_text}”展开。"


def _build_review_moments(
    messages: list,
    wife_count: int,
    assistant_count: int,
    voice_count: int,
) -> list[str]:
    if not messages:
        return ["这个时间段暂时没有可回顾的对话。"]

    moments = [
        f"她发了 {wife_count} 条消息，AI 回复了 {assistant_count} 条。",
    ]
    first_text = next((item.content.strip() for item in messages if item.content.strip()), "")
    if first_text:
        moments.append(f"对话从“{first_text[:36]}”附近展开。")
    if voice_count:
        moments.append(f"其中有 {voice_count} 条语音消息，适合后续接入转写后做更细回顾。")
    return moments


def _build_review_suggestions(messages: list, voice_count: int) -> list[str]:
    if not messages:
        return ["先积累几轮日常聊天，再回来生成更有信息量的回顾。"]

    suggestions = [
        "继续保留微信式短句回复，少讲道理，多接住当下情绪。",
        "训练界面里可以补充她喜欢和不喜欢的回答样例。",
    ]
    if voice_count:
        suggestions.append("语音消息后续可以接 ASR 转写，让回顾更准确。")
    return suggestions
