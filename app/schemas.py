from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ChatRole = Literal["wife", "assistant"]
MessageKind = Literal["text", "voice", "image"]
ConversationMode = Literal["chat", "training"]


class ChatMessage(BaseModel):
    id: str
    role: ChatRole
    kind: MessageKind
    content: str
    createdAt: datetime
    mode: ConversationMode = "chat"
    durationSeconds: float | None = None
    audioUrl: str | None = None
    imageUrl: str | None = None
    pending: bool | None = None
    failed: bool | None = None


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]


class SendChatMessageRequest(BaseModel):
    content: str = Field(default="")
    kind: MessageKind = "text"
    durationSeconds: float | None = Field(default=None, ge=0)
    imageUrl: str | None = None


class SendChatMessageResponse(BaseModel):
    message: ChatMessage


class ChatRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str


class TrainingChatRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1)
    message: str = Field(..., min_length=1)


class ConversationReviewRequest(BaseModel):
    startAt: datetime
    endAt: datetime
    modes: list[ConversationMode] = Field(default_factory=lambda: ["chat"])


class ConversationReviewStats(BaseModel):
    totalMessages: int
    wifeMessages: int
    assistantMessages: int
    voiceMessages: int


class ConversationReviewResponse(BaseModel):
    startAt: datetime
    endAt: datetime
    modes: list[ConversationMode]
    stats: ConversationReviewStats
    summary: str
    messages: list[ChatMessage]


class ReviewSummaryResponse(BaseModel):
    title: str
    rangeLabel: str
    overview: str = ""
    coreEvents: list[str] = Field(default_factory=list)
    userEmotionExpressions: list[str] = Field(default_factory=list)
    emotionalTrend: str = ""
    aiResponsePattern: str = ""
    importantDialogues: list[str] = Field(default_factory=list)
    followUpSuggestions: list[str] = Field(default_factory=list)
    aiSummary: str
    wifeSummary: str
    moments: list[str]
    suggestions: list[str]
    messageCount: int


class ReviewFollowUpRequest(BaseModel):
    startDate: str
    endDate: str
    instruction: str = Field(..., min_length=1)


class ReviewFollowUpResponse(BaseModel):
    answer: str
    rangeLabel: str
    relatedDialogues: list[str] = Field(default_factory=list)
    messageCount: int


class VoiceTranscriptionResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "x4_yezi"
    scene: str = "default"


class VoiceCloneTrainTextResponse(BaseModel):
    textId: int
    textName: str = ""
    textSegs: list[dict] = Field(default_factory=list)


class VoiceCloneTrainResponse(BaseModel):
    taskId: str
    audioUrl: str


class VoiceCloneResultResponse(BaseModel):
    taskId: str
    trainStatus: int | None = None
    assetId: str = ""
    trainVid: str = ""
    failedDesc: str = ""
    raw: dict = Field(default_factory=dict)


class AIStatusResponse(BaseModel):
    provider: str
    mode: str
    model: str
    base_url: str
    api_key_configured: bool
    init_error: str
