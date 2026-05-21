from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

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
    ReviewFollowUpRequest,
    ReviewFollowUpResponse,
    SendChatMessageRequest,
    SendChatMessageResponse,
    TTSRequest,
    TrainingChatRequest,
    VoiceCloneResultResponse,
    VoiceCloneTrainResponse,
    VoiceCloneTrainTextResponse,
    VoiceTranscriptionResponse,
)
from app.services.ai_service import ai_service
from app.services.asr_service import transcribe_pcm16
from app.services.chat_store import ChatStore
from app.services.tts_service import synthesize_audio, synthesize_clone_audio
from app.services.voice_clone_service import create_voice_clone_task, get_train_text, get_voice_clone_result
from app.services.voice_clone_profile import load_active_clone_res_id, save_voice_clone_result


app = FastAPI(title=settings.app_name)
chat_store = ChatStore(
    file_path=settings.chat_store_path,
    max_history_messages=settings.chat_max_history_messages,
)
UPLOAD_ROOT = Path(settings.upload_dir)
IMAGE_UPLOAD_DIR = UPLOAD_ROOT / "images"
VOICE_CLONE_UPLOAD_DIR = UPLOAD_ROOT / "voice-clone"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_TIMEZONE = timezone(timedelta(hours=8))

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")


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
        history=chat_store.recent_for_ai(mode="chat", limit=settings.ai_context_messages),
        training_memory=chat_store.recent_for_ai(mode="training", limit=settings.ai_context_messages),
    )
    return ChatResponse(reply=reply)


@app.post("/ai/training-chat", response_model=ChatResponse)
def ai_training_chat(payload: TrainingChatRequest) -> ChatResponse:
    reply = ai_service.training_reply(
        user_message=payload.message.strip(),
        history=chat_store.recent_for_ai(mode="training", limit=settings.ai_context_messages),
    )
    return ChatResponse(reply=reply)


@app.get("/api/chat/history", response_model=ChatHistoryResponse)
def chat_history(
    limit: int = Query(default=50, ge=1, le=200),
    before: datetime | None = Query(default=None),
) -> ChatHistoryResponse:
    return ChatHistoryResponse(messages=chat_store.history_page(mode="chat", limit=limit, before=before))


@app.post("/api/chat/message", response_model=SendChatMessageResponse)
async def send_chat_message(request: Request) -> SendChatMessageResponse:
    message = await _send_message_for_mode(request=request, mode="chat")
    return SendChatMessageResponse(message=message)


@app.get("/api/training/history", response_model=ChatHistoryResponse)
def training_history(
    limit: int = Query(default=50, ge=1, le=200),
    before: datetime | None = Query(default=None),
) -> ChatHistoryResponse:
    return ChatHistoryResponse(messages=chat_store.history_page(mode="training", limit=limit, before=before))


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


@app.post("/api/voice/transcribe", response_model=VoiceTranscriptionResponse)
async def transcribe_voice(audio: UploadFile = File(...)) -> VoiceTranscriptionResponse:
    audio_bytes = await audio.read()
    try:
        text = await transcribe_pcm16(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VoiceTranscriptionResponse(text=text)


@app.post("/tts")
async def text_to_speech(payload: TTSRequest) -> Response:
    try:
        if payload.scene == "daily" and (settings.xfyun_clone_res_id or load_active_clone_res_id()):
            audio_bytes = await synthesize_clone_audio(payload.text.strip())
        else:
            audio_bytes = await synthesize_audio(payload.text.strip(), voice=payload.voice)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/api/voice-clone/train-text", response_model=VoiceCloneTrainTextResponse)
async def voice_clone_train_text(textId: int = Query(default=5001)) -> VoiceCloneTrainTextResponse:
    try:
        data = await get_train_text(text_id=textId)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VoiceCloneTrainTextResponse(
        textId=int(data.get("textId") or textId),
        textName=data.get("textName") or "",
        textSegs=data.get("textSegs") or [],
    )


@app.post("/api/voice-clone/train", response_model=VoiceCloneTrainResponse)
async def voice_clone_train(
    audio: UploadFile = File(...),
    textId: int = Form(default=5001),
    textSegId: int = Form(default=1),
    format: str = Form(default=""),
    resourceName: str = Form(default="aime-daily-voice"),
) -> VoiceCloneTrainResponse:
    format_suffix = f".{format.strip().lower().lstrip('.')}" if format else ""
    suffix = Path(audio.filename or "").suffix.lower() or format_suffix
    if suffix not in {".wav", ".mp3", ".m4a", ".pcm"}:
        suffix = ".wav"

    VOICE_CLONE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}{suffix}"
    file_path = VOICE_CLONE_UPLOAD_DIR / filename
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="录音文件为空，请重新录制")
    file_path.write_bytes(audio_bytes)
    audio_url = f"{settings.public_base_url.rstrip('/')}/uploads/voice-clone/{filename}"

    try:
        task_id = await create_voice_clone_task(
            audio_url=audio_url,
            text_id=textId,
            text_seg_id=textSegId,
            resource_name=resourceName,
            task_name=resourceName,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{exc}; audioUrl={audio_url}") from exc
    return VoiceCloneTrainResponse(taskId=task_id, audioUrl=audio_url)


@app.get("/api/voice-clone/result", response_model=VoiceCloneResultResponse)
async def voice_clone_result(taskId: str = Query(...)) -> VoiceCloneResultResponse:
    try:
        data = await get_voice_clone_result(taskId)
        save_voice_clone_result(taskId, data)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return VoiceCloneResultResponse(
        taskId=taskId,
        trainStatus=data.get("trainStatus"),
        assetId=data.get("assetId") or "",
        trainVid=data.get("trainVid") or "",
        failedDesc=data.get("failedDesc") or "",
        raw=data,
    )


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


@app.post("/api/chat/review/ask", response_model=ReviewFollowUpResponse)
def chat_review_follow_up(payload: ReviewFollowUpRequest) -> ReviewFollowUpResponse:
    try:
        start_date = date.fromisoformat(payload.startDate)
        end_date = date.fromisoformat(payload.endDate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date range") from exc

    start_at, end_at = _local_date_range_to_utc(startDate=start_date, endDate=end_date)
    messages = chat_store.review_messages(
        start_at=start_at,
        end_at=end_at,
        modes=["chat"],
    )
    range_label = f"{min(start_date, end_date).isoformat()} 至 {max(start_date, end_date).isoformat()}"
    related = _build_important_dialogues(messages=messages, limit=5)
    answer = ai_service.review_follow_up(
        messages=_review_items(messages),
        start_at=start_at.isoformat(),
        end_at=end_at.isoformat(),
        instruction=payload.instruction.strip(),
    )
    return ReviewFollowUpResponse(
        answer=answer,
        rangeLabel=range_label,
        relatedDialogues=related,
        messageCount=len(messages),
    )


async def _send_message_for_mode(request: Request, mode: ConversationMode):
    payload = await _parse_send_message_request(request)
    content = payload.content.strip()

    chat_store.append_user_message(
        content=content,
        kind=payload.kind,
        mode=mode,
        duration_seconds=payload.durationSeconds,
        image_url=payload.imageUrl,
    )
    if mode == "training":
        reply = ai_service.training_reply(
            user_message=content,
            history=chat_store.recent_for_ai(mode="training", limit=settings.ai_context_messages),
            image_url=payload.imageUrl,
        )
    else:
        reply = ai_service.reply(
            user_message=content,
            history=chat_store.recent_for_ai(mode="chat", limit=settings.ai_context_messages),
            training_memory=chat_store.recent_for_ai(mode="training", limit=settings.ai_context_messages),
            image_url=payload.imageUrl,
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
        image = form.get("image")
        image_url = None
        if hasattr(image, "filename") and hasattr(image, "read"):
            image_url = await _save_image_upload(request=request, image=image)
            kind = "image"
            if not text:
                text = "我发了一张图片。"
        if kind == "voice" and not text:
            filename = getattr(audio, "filename", "") or "voice message"
            text = f"[收到一条语音消息：{filename}]"
        return SendChatMessageRequest(
            content=text,
            kind=kind if kind in {"text", "voice", "image"} else "text",
            durationSeconds=duration_seconds,
            imageUrl=image_url,
        )

    return SendChatMessageRequest.model_validate(await request.json())


async def _save_image_upload(request: Request, image: UploadFile) -> str:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}:
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/heic": ".heic",
            "image/heif": ".heif",
        }.get(image.content_type, ".jpg")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large")

    IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex}{suffix}"
    file_path = IMAGE_UPLOAD_DIR / filename
    file_path.write_bytes(image_bytes)
    return _public_upload_url(request=request, path=f"/uploads/images/{filename}")


def _public_upload_url(request: Request, path: str) -> str:
    base = settings.public_base_url.rstrip("/")
    if not base:
        forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        forwarded_host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
        forwarded_prefix = request.headers.get("x-forwarded-prefix", "")
        base = f"{forwarded_proto}://{forwarded_host}{forwarded_prefix}".rstrip("/")
    return f"{base}{path}"


def _local_date_range_to_utc(startDate: date, endDate: date) -> tuple[datetime, datetime]:
    start = min(startDate, endDate)
    end = max(startDate, endDate)
    start_local = datetime.combine(start, time.min, tzinfo=LOCAL_TIMEZONE)
    end_local = datetime.combine(end, time.max, tzinfo=LOCAL_TIMEZONE)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _review_items(messages: list) -> list[dict[str, str]]:
    return [
        {
            "role": item.role,
            "speaker": "她" if item.role == "wife" else "我",
            "kind": item.kind,
            "content": item.content,
            "createdAt": item.createdAt.isoformat(),
        }
        for item in messages
    ]


def _build_review_summary(
    messages: list,
    start_date: date,
    end_date: date,
) -> ReviewSummaryResponse:
    wife_messages = [item for item in messages if item.role == "wife"]
    assistant_messages = [item for item in messages if item.role == "assistant"]
    voice_count = sum(1 for item in messages if item.kind == "voice")
    image_count = sum(1 for item in messages if item.kind == "image")
    range_label = f"{min(start_date, end_date).isoformat()} 至 {max(start_date, end_date).isoformat()}"
    core_events = _build_core_events(messages=messages)
    emotions = _build_user_emotions(messages=wife_messages)
    important_dialogues = _build_important_dialogues(messages=messages)
    overview = _build_review_overview(messages=messages, core_events=core_events, emotions=emotions)

    return ReviewSummaryResponse(
        title="这段时间的对话回顾",
        rangeLabel=range_label,
        overview=overview,
        coreEvents=core_events,
        userEmotionExpressions=emotions,
        emotionalTrend=_build_emotional_trend(wife_messages),
        aiResponsePattern=_build_ai_response_pattern(assistant_messages),
        importantDialogues=important_dialogues,
        followUpSuggestions=_build_review_follow_up_suggestions(
            messages=messages,
            voice_count=voice_count,
            image_count=image_count,
        ),
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


def _build_review_overview(messages: list, core_events: list[str], emotions: list[str]) -> str:
    if not messages:
        return "这个时间段里还没有可回顾的对话。"

    event_text = core_events[0] if core_events else "没有明显单一事件"
    emotion_text = emotions[0] if emotions else "情绪表达不明显"
    return f"这段时间共有 {len(messages)} 条对话，主要围绕“{event_text}”展开；用户情绪上更接近“{emotion_text}”。"


def _build_core_events(messages: list, limit: int = 3) -> list[str]:
    user_texts = [item.content.strip() for item in messages if item.role == "wife" and item.content.strip()]
    if not user_texts:
        return ["暂无明显核心事件。"] if messages else ["这个时间段暂时没有对话。"]

    events = []
    for text in user_texts:
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            continue
        if len(cleaned) > 42:
            cleaned = f"{cleaned[:42]}..."
        if cleaned not in events:
            events.append(cleaned)
        if len(events) >= limit:
            break
    return events or ["暂无明显核心事件。"]


def _build_user_emotions(messages: list, limit: int = 3) -> list[str]:
    emotion_keywords = [
        "开心", "高兴", "喜欢", "想你", "爱", "委屈", "难受", "累", "烦", "焦虑",
        "生气", "失落", "害怕", "担心", "崩溃", "孤独", "不安", "压力", "期待",
    ]
    matches = []
    for item in messages:
        text = item.content.strip()
        if not text:
            continue
        found = [word for word in emotion_keywords if word in text]
        if found:
            quote = text if len(text) <= 38 else f"{text[:38]}..."
            matches.append(f"{'、'.join(found[:3])}：{quote}")
        if len(matches) >= limit:
            break
    if matches:
        return matches
    return ["没有明显强烈情绪词，整体更像日常表达。"] if messages else ["暂无用户情绪表达。"]


def _build_emotional_trend(messages: list) -> str:
    if not messages:
        return "暂无可判断的情绪走势。"
    negative = ("委屈", "难受", "累", "烦", "焦虑", "生气", "失落", "害怕", "担心", "崩溃", "压力")
    positive = ("开心", "高兴", "喜欢", "想你", "爱", "期待", "舒服")
    first = " ".join(item.content for item in messages[: max(1, len(messages) // 2)])
    second = " ".join(item.content for item in messages[max(1, len(messages) // 2):])
    first_score = sum(first.count(word) for word in positive) - sum(first.count(word) for word in negative)
    second_score = sum(second.count(word) for word in positive) - sum(second.count(word) for word in negative)
    if second_score > first_score:
        return "情绪后段比前段更放松或更积极。"
    if second_score < first_score:
        return "情绪后段比前段更沉一些，需要继续留意。"
    return "情绪整体比较平稳，没有明显大幅转折。"


def _build_ai_response_pattern(messages: list) -> str:
    if not messages:
        return "这段时间里还没有我的回复。"
    samples = [item.content.strip() for item in messages if item.content.strip()][:2]
    if not samples:
        return f"我回复了 {len(messages)} 条，主要是非文字或空内容。"
    return f"我回复了 {len(messages)} 条，主要在承接情绪、顺着上下文回应；代表回复：“{'；'.join(samples)}”。"


def _build_important_dialogues(messages: list, limit: int = 4) -> list[str]:
    if not messages:
        return []

    important_words = ("委屈", "难受", "累", "烦", "焦虑", "生气", "想你", "爱", "重要", "记住", "别", "希望")
    selected = []
    for item in messages:
        text = item.content.strip()
        if not text:
            continue
        if any(word in text for word in important_words) or item.kind != "text":
            speaker = "她" if item.role == "wife" else "我"
            content = text if len(text) <= 60 else f"{text[:60]}..."
            selected.append(f"{speaker}：{content}")
        if len(selected) >= limit:
            break

    if selected:
        return selected

    for item in messages[:limit]:
        text = item.content.strip()
        if text:
            speaker = "她" if item.role == "wife" else "我"
            content = text if len(text) <= 60 else f"{text[:60]}..."
            selected.append(f"{speaker}：{content}")
    return selected


def _build_review_follow_up_suggestions(messages: list, voice_count: int, image_count: int) -> list[str]:
    if not messages:
        return ["可以换一个更长的时间段再回顾。"]

    suggestions = [
        "可以继续追问：这段时间她最在意什么？",
        "可以继续追问：帮我只看情绪变化。",
        "可以继续追问：把重要原话列出来。",
    ]
    if voice_count:
        suggestions.append("这段里有语音，必要时可以重点看语音转写内容。")
    if image_count:
        suggestions.append("这段里有图片，必要时可以围绕图片上下文继续追问。")
    return suggestions
