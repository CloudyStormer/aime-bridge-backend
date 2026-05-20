from typing import Any

from openai import OpenAI

from app.config import settings


class AIService:
    def __init__(self) -> None:
        self._provider = settings.llm_provider.lower().strip()
        self._model = settings.llm_model
        self._base_url = settings.llm_base_url
        self._api_key = settings.llm_api_key
        self._client: OpenAI | None = None
        self._init_error = ""

        self._apply_provider_overrides()
        self._initialize_client()

    def reply(self, user_message: str, history: list[dict[str, str]]) -> str:
        fallback = self._fallback_reply(user_message)
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_message})

        if self._client is None:
            return fallback

        try:
            completion: Any = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.75,
            )
            content = completion.choices[0].message.content
            return content.strip() if content else fallback
        except Exception:
            return fallback

    def chat(self, user_message: str, history: list[dict[str, str]] | None = None) -> str:
        return self.reply(user_message=user_message, history=history or [])

    def training_reply(self, user_message: str, history: list[dict[str, str]]) -> str:
        fallback = self._fallback_training_reply(user_message)
        messages = [{"role": "system", "content": self._training_system_prompt()}]
        messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_message})
        return self._invoke(messages=messages, fallback=fallback, temperature=0.45)

    def conversation_review(self, messages: list[dict[str, str]], start_at: str, end_at: str) -> str:
        if not messages:
            return "这个时间段里还没有可回顾的对话。"

        transcript = "\n".join(
            f"{item['createdAt']} {item['speaker']}：{item['content']}"
            for item in messages
            if item.get("content")
        )
        fallback = self._fallback_review(messages=messages, start_at=start_at, end_at=end_at)
        prompt = "\n".join(
            [
                "你负责做一段亲密对话的回顾总结。",
                "请用中文输出，语气克制、清楚、尊重隐私。",
                "重点说明：这段时间里她主要说了什么，我主要回应了什么，情绪走向是什么，可以后续留意什么。",
                "不要编造对话中没有出现的信息。",
                "控制在 4 到 8 句。",
            ]
        )
        return self._invoke(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"时间段：{start_at} 到 {end_at}\n\n对话记录：\n{transcript}",
                },
            ],
            fallback=fallback,
            temperature=0.35,
        )

    def runtime_status(self) -> dict[str, Any]:
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "provider": self._provider,
            "mode": "real" if self._client is not None else "mock",
            "model": self._model,
            "base_url": self._base_url or "",
            "api_key_configured": bool(self._api_key),
            "init_error": self._init_error,
        }

    def _invoke(self, messages: list[dict[str, str]], fallback: str, temperature: float) -> str:
        if self._client is None:
            return fallback

        try:
            completion: Any = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            content = completion.choices[0].message.content
            return content.strip() if content else fallback
        except Exception:
            return fallback

    def _apply_provider_overrides(self) -> None:
        if self._provider == "deepseek":
            self._api_key = settings.deepseek_api_key or self._api_key
            self._model = settings.deepseek_model or self._model
            self._base_url = settings.deepseek_base_url or self._base_url
        elif self._provider == "hunyuan":
            self._api_key = settings.hunyuan_api_key or self._api_key
            self._model = settings.hunyuan_model or self._model
            self._base_url = settings.hunyuan_base_url or self._base_url

    def _initialize_client(self) -> None:
        if self._provider == "mock" or not self._api_key:
            return
        if self._provider not in {"openai", "openai_compatible", "deepseek", "hunyuan"}:
            self._init_error = f"Unsupported provider: {self._provider}"
            return

        client_kwargs: dict[str, str] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = OpenAI(**client_kwargs)

    def _system_prompt(self) -> str:
        return "\n".join(
            [
                "你是这个产品里的“我”，不是英语老师，也不是第三方客服。",
                "你的说话对象是前端聊天里的亲密用户。你要像我本人一样自然回复：真诚、亲近、稳定、有分寸。",
                "不要进行英语教学，不要纠错，不要把对话带到学习场景，除非用户明确要求。",
                "默认使用中文。语气可以温柔一点，但不要油腻、不要模板化、不要反复解释你是 AI。",
                "回复保持简短，通常 2 到 5 句；一次最多问一个自然的问题。",
                "你可以承接情绪、陪她聊天、帮她整理想法，也可以在需要时给出实际的小建议。",
                "不要假装已经完成现实世界动作，不要编造你不知道的事实。",
                "如果用户表达明显危险、伤害自己或他人的意图，先稳定情绪，再建议立刻联系身边可信的人或当地紧急服务。",
            ]
        )

    def _training_system_prompt(self) -> str:
        return "\n".join(
            [
                "这是“训练我”的界面，不是普通陪聊界面。",
                "用户会告诉你：我应该怎么说话、哪些表达要保留、哪些表达不要用、遇到某类情况要如何回应。",
                "你的任务是把用户输入沉淀成清晰的回复规则，并用简短中文确认你学到了什么。",
                "不要假装已经永久写入模型；只说明本轮训练记录已经收到，后续会按这些规则调整。",
                "如果用户给了示例，请提炼语气、边界、禁用词和可复用表达。",
            ]
        )

    def _fallback_reply(self, user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "我在，你慢慢说。"
        if any(word in text for word in ("累", "烦", "难受", "委屈", "焦虑", "崩溃")):
            return "嗯，我听到了。先别急着把所有事都扛起来，告诉我现在最压着你的那一件事就好。"
        return "我听到了。你继续说，我在这边陪你把这件事慢慢理清楚。"

    def _fallback_training_reply(self, user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "我一直都在。你可以直接告诉我，哪些话要更像你，哪些话以后不要这么说。"
        ending = "" if text.endswith(("。", "！", "？", ".", "!", "?")) else "。"
        return f"收到，我会把这条当成训练规则：{text}{ending}后续回复会更贴近这个语气和边界。"

    def _fallback_review(self, messages: list[dict[str, str]], start_at: str, end_at: str) -> str:
        wife_count = sum(1 for item in messages if item.get("role") == "wife")
        assistant_count = sum(1 for item in messages if item.get("role") == "assistant")
        first_items = [item.get("content", "") for item in messages[:3] if item.get("content")]
        sample = "；".join(first_items)
        return (
            f"{start_at} 到 {end_at} 这段时间里，共有 {wife_count} 条她的消息和 "
            f"{assistant_count} 条我的回复。主要对话从“{sample}”附近展开。"
            "整体看，这段回顾适合继续结合具体消息做更细的情绪和主题整理。"
        )


ai_service = AIService()
