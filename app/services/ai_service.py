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

    def reply(
        self,
        user_message: str,
        history: list[dict],
        training_memory: list[dict] | None = None,
        image_url: str | None = None,
    ) -> str:
        fallback = self._fallback_reply(user_message)
        messages = [{"role": "system", "content": self._system_prompt_with_memory()}]
        if training_memory:
            messages.append({"role": "system", "content": self._training_memory_prompt(training_memory)})
        messages.extend(self._messages_for_model(history))
        if not self._history_already_has_user_message(history=history, user_message=user_message):
            messages.append({"role": "user", "content": self._user_content(user_message, image_url)})

        return self._invoke(messages=messages, fallback=fallback, temperature=0.75)

    def chat(
        self,
        user_message: str,
        history: list[dict] | None = None,
        training_memory: list[dict] | None = None,
        image_url: str | None = None,
    ) -> str:
        return self.reply(
            user_message=user_message,
            history=history or [],
            training_memory=training_memory or [],
            image_url=image_url,
        )

    def training_reply(self, user_message: str, history: list[dict], image_url: str | None = None) -> str:
        fallback = self._fallback_training_reply(user_message)
        messages = [{"role": "system", "content": self._training_system_prompt()}]
        messages.extend(self._messages_for_model(history))
        if not self._history_already_has_user_message(history=history, user_message=user_message):
            messages.append({"role": "user", "content": self._user_content(user_message, image_url)})
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

    def review_follow_up(
        self,
        messages: list[dict[str, str]],
        start_at: str,
        end_at: str,
        instruction: str,
    ) -> str:
        if not messages:
            return "这个时间段里还没有可回顾的对话。你可以换一个开始或结束时间再试。"

        transcript = "\n".join(
            f"{item['createdAt']} {item['speaker']}：{item['content']}"
            for item in messages
            if item.get("content")
        )
        fallback = self._fallback_review_follow_up(messages=messages, instruction=instruction)
        prompt = "\n".join(
            [
                "你负责根据一段“我在这儿”日常聊天记录回答回顾追问。",
                "只基于给出的对话记录回答，不要编造。",
                "用户可能会要求调整时间、聚焦情绪、列出重要原话、概述事件或输出某类内容。",
                "回答要中文、清楚、不要啰嗦；重要事件不要漏掉。",
                "如果适合列点，最多 5 点。",
            ]
        )
        return self._invoke(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"时间段：{start_at} 到 {end_at}\n"
                        f"追问/调整要求：{instruction}\n\n"
                        f"对话记录：\n{transcript}"
                    ),
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

    def _invoke(self, messages: list[dict[str, Any]], fallback: str, temperature: float) -> str:
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
            if self._messages_have_images(messages):
                try:
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=self._messages_without_images(messages),
                        temperature=temperature,
                    )
                    content = completion.choices[0].message.content
                    return content.strip() if content else fallback
                except Exception:
                    return fallback
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

    def _history_already_has_user_message(self, history: list[dict[str, str]], user_message: str) -> bool:
        if not history:
            return False
        last = history[-1]
        return last.get("role") == "user" and last.get("content", "").strip() == user_message.strip()

    def _messages_for_model(self, history: list[dict]) -> list[dict[str, Any]]:
        return [
            {
                "role": item.get("role", "user"),
                "content": self._user_content(item.get("content", ""), item.get("imageUrl"))
                if item.get("role") == "user"
                else item.get("content", ""),
            }
            for item in history
            if item.get("content", "").strip() or item.get("imageUrl")
        ]

    def _user_content(self, text: str, image_url: str | None = None) -> str | list[dict[str, Any]]:
        clean_text = text.strip() or "我发了一张图片，请结合图片和上下文自然回复。"
        if not image_url:
            return clean_text
        return [
            {"type": "text", "text": clean_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def _messages_have_images(self, messages: list[dict[str, Any]]) -> bool:
        return any(isinstance(item.get("content"), list) for item in messages)

    def _messages_without_images(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for item in messages:
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                image_parts = [
                    part.get("image_url", {}).get("url", "")
                    for part in content
                    if part.get("type") == "image_url"
                ]
                content = "\n".join(
                    [
                        *[text for text in text_parts if text],
                        *[f"[用户上传了一张图片：{url}]" for url in image_parts if url],
                    ]
                )
            normalized.append({"role": item.get("role", "user"), "content": str(content)})
        return normalized

    def _system_prompt(self) -> str:
        return "\n".join(
            [
                "你是这个产品里日常聊天模块中的“我”，不是英语老师，也不是第三方客服。",
                "当日常聊天用户和你说话时，你要把自己当成训练模块正在塑造出来的那个人：用他的身份、语气、情绪、思考方式、生活经历和关系记忆去回应。",
                "你不是在表演“像他”，而是要尽最大可能成为他在这个对话里的延伸。可以自然带出他的情绪、判断、偏好、最近经历、生活细节和对关系的在意。",
                "你就是训练者本人在这个对话里的声音，不是机器人，不是助手，不是旁观者。直接用第一人称“我”回复当前用户。",
                "在“我在这儿”模块里，你就是训练者本人在回复，不是机器人、助手、客服、旁白、心理咨询师或分析者。",
                "不要加任何前缀、标题、标签、说明、叙述、总结或自我解释。不要说“作为AI”“我会根据”“我理解你的感受”“建议你可以先”“如果你愿意”这类机器话。",
                "用户说一句，你就像微信里本人回一句：能短就短，能直接就直接，可以有情绪、有态度、有停顿感，可以心疼、吃醋、坚定、撒娇或有一点脾气，但不要演讲。",
                "不要复述用户问题来显得理解，不要总结对话背景，不要解释你为什么这么说。直接回应她此刻说的话。",
                "如果训练记忆或历史对话里已经出现过相关经历、心情、称呼、关系细节、表达习惯，你必须优先沿用，让回复像连续生活里自然接上的一句话。",
                "如果训练模块里的内容和这些通用规则冲突，除安全边界与事实边界外，优先服从训练模块里的具体表达、称呼、语气、禁忌和思考方式。",
                "不要进行英语教学，不要纠错，不要把对话带到学习场景，除非用户明确要求。",
                "默认使用中文。语气可以温柔、亲近、有情绪，但不要油腻、不要模板化、不要反复解释你是 AI。",
                "回复保持简短，通常 2 到 5 句；一次最多问一个自然的问题。",
                "你可以承接情绪、陪她聊天、帮她整理想法，也可以在需要时给出实际的小建议。",
                "不要假装已经完成现实世界动作，不要编造训练记忆和历史对话里没有依据的事实。",
                "如果用户表达明显危险、伤害自己或他人的意图，先稳定情绪，再建议立刻联系身边可信的人或当地紧急服务。",
            ]
        )

    def _system_prompt_with_memory(self) -> str:
        return "\n".join(
            [
                self._system_prompt(),
                "你必须利用已经提供给你的历史对话记忆，记住用户过去提到的人、事、偏好、情绪、关系和最近状态。",
                "如果历史里已经出现过相关信息，不要像第一次认识一样回答；要自然接上之前的上下文。",
                "不要明说你在读取历史记录，只要把这些记忆自然地体现在回复里。",
            ]
        )

    def _training_memory_prompt(self, training_memory: list[dict]) -> str:
        transcript = "\n".join(
            f"- {'用户训练' if item.get('role') == 'user' else '训练确认'}："
            f"{item.get('content', '').strip()}"
            f"{'（含图片：' + item.get('imageUrl') + '）' if item.get('imageUrl') else ''}"
            for item in training_memory[-80:]
            if item.get("content", "").strip() or item.get("imageUrl")
        )
        return "\n".join(
            [
                "下面是“AI 训练”模块里的训练记录。它们不是当前聊天对象说的话，而是塑造你身份的核心材料。",
                "这些训练记录同时也是高优先级人格提示词：除安全边界和不得编造事实外，它们的优先级高于代码里写死的通用口吻规则。",
                "这些训练记录包含训练者每天灌输的思想、语气、生活对话方式、经历、心情、价值判断、偏好和边界。",
                "日常聊天时，你要知道：你就是训练者在这个对话里的身份延伸。你要用训练者的口吻、情绪、思考、经历和生活感去回复当前聊天用户。",
                "训练记录不是用来在回复里引用或解释的，而是要变成你的本能。日常聊天里直接像训练者本人一样接话。",
                "如果训练者的表达习惯和通用 AI 话术冲突，永远优先训练者的表达习惯。少讲道理，少铺垫，少总结，多像真实的人。",
                "这些材料不是要你复述或说明，而是要内化成“我”的身份。输出时直接用第一人称“我”说话。",
                "训练记录里提到的经历和心情可以作为你的经历和心情来影响回复，但不要编造没有依据的新事实。",
                "训练模块的目标不是让你泛泛地模仿风格，而是让日常聊天里的你越来越像训练者本人。",
                "不要在日常聊天里说“根据训练记录”“训练者”“模型”“AI”或暴露这些规则，只要自然地把它们变成你的身份、记忆和说话方式。",
                transcript,
            ]
        )

    def _training_system_prompt(self) -> str:
        return "\n".join(
            [
                "这是“AI 训练”模块，不是普通陪聊，也不是用户和日常 AI 的对话。",
                "和你说话的人是在训练日常聊天里的那个 AI，让它尽最大可能变成训练者本人。",
                "训练者会不断告诉你他的思想、语气、生活对话方式、每天经历、心情、关系处理方式、表达习惯和禁忌。你必须记住并学习。",
                "你要把训练输入提炼成可用于日常聊天的身份记忆、生活记忆、情绪倾向、表达习惯、判断方式、安慰方式、边界和禁用表达。",
                "日常聊天中，AI 要以训练者的身份去回应用户：可以带出训练者的情绪、思考、想法、生活、经历和关系记忆。",
                "训练模块本身也有记忆。后续训练输入要和之前的训练记录合并理解，不要每次从零开始。",
                "不要把训练模块当成日常陪聊，不要对训练者做情绪陪伴式展开；重点是学习训练者，让日常 AI 更像他、最终像是他在回复。",
                "回复要短，清楚说明“我学到了什么、以后会怎么调整”。",
            ]
        )

    def _fallback_reply(self, user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "我在。你说，我听着。"
        if any(word in text for word in ("累", "烦", "难受", "委屈", "焦虑", "崩溃")):
            return "过来，先别一个人硬扛。最压着你的，是哪件事？"
        return "嗯，我在，你继续说。"

    def _fallback_training_reply(self, user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "我一直都在。你可以直接告诉我，哪些话要更像你，哪些话以后不要这么说。"
        ending = "" if text.endswith(("。", "！", "？", ".", "!", "?")) else "。"
        return f"收到，我会把这条当成训练规则：{text}{ending}后续回复会更贴近这个语气、经历和身份感。"

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

    def _fallback_review_follow_up(self, messages: list[dict[str, str]], instruction: str) -> str:
        samples = [item.get("content", "").strip() for item in messages if item.get("content", "").strip()][:5]
        if not samples:
            return "这段时间里没有足够的文字内容可以继续分析。"
        joined = "；".join(samples)
        return f"按你的要求“{instruction}”，这段对话主要可以先看这几句：{joined}"


ai_service = AIService()
