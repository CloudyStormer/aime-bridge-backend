import base64
import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone
from urllib.parse import quote

import websockets

from app.config import settings
from app.services.voice_clone_profile import load_active_clone_res_id


TTS_HOST = "tts-api.xfyun.cn"
TTS_PATH = "/v2/tts"
VOICE_CLONE_HOST = "cn-huabei-1.xf-yun.com"
VOICE_CLONE_PATH = "/v1/private/voice_clone"
TTS_CONNECT_TIMEOUT_SECONDS = 8
TTS_RECEIVE_TIMEOUT_SECONDS = 15


def _infer_voice_mood(text: str) -> dict[str, int | str]:
    lowered = text.lower()
    sad_words = ("累", "难受", "委屈", "想哭", "害怕", "低落", "崩溃", "失落", "心疼", "抱抱", "没事")
    warm_words = ("乖", "慢慢", "陪你", "我在", "别怕", "放心", "抱", "想你", "宝贝")
    happy_words = ("开心", "真好", "哈哈", "太棒", "喜欢", "期待", "厉害", "好呀", "爱你")
    urgent_words = ("别急", "马上", "赶紧", "现在", "一定", "必须")

    profile: dict[str, int | str] = {
        "speed": 46,
        "pitch": 49,
        "volume": 72,
        "rhy": 1,
        "style": settings.xfyun_clone_style or "chat",
        "impactFactor": 35,
    }
    if any(word in text for word in sad_words):
        profile.update({"speed": 40, "pitch": 45, "volume": 68, "impactFactor": 50})
    elif any(word in text for word in warm_words):
        profile.update({"speed": 42, "pitch": 47, "volume": 70, "impactFactor": 45})
    elif any(word in text for word in happy_words) or "!" in text or "！" in text:
        profile.update({"speed": 50, "pitch": 54, "volume": 76, "impactFactor": 55})
    elif any(word in lowered for word in urgent_words):
        profile.update({"speed": 52, "pitch": 51, "volume": 74, "impactFactor": 40})
    return profile


def _shape_clone_text(text: str) -> str:
    shaped = (
        text.replace("。", "。 ")
        .replace("，", "， ")
        .replace("？", "？ ")
        .replace("！", "！ ")
        .replace("；", "； ")
        .replace("：", "： ")
    )
    shaped = " ".join(shaped.split())
    if len(shaped) <= 80 and not shaped.endswith(("。", "！", "？", ".", "!", "?")):
        shaped = f"{shaped}。"
    return shaped


def _build_auth_url(host: str = TTS_HOST, path: str = TTS_PATH) -> str:
    date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(
            key=settings.xfyun_api_secret.encode("utf-8"),
            msg=sign_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    auth_origin = (
        f'api_key="{settings.xfyun_api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(auth_origin.encode("utf-8")).decode("utf-8")
    return (
        f"wss://{host}{path}"
        f"?authorization={quote(authorization)}"
        f"&date={quote(date)}"
        f"&host={host}"
    )


async def synthesize_audio(text: str, voice: str = "x4_yezi") -> bytes:
    if not settings.xfyun_app_id or not settings.xfyun_api_key or not settings.xfyun_api_secret:
        raise RuntimeError("讯飞 TTS 未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET")

    profile = _infer_voice_mood(text)
    shaped_text = _shape_clone_text(text)
    text_b64 = base64.b64encode(shaped_text.encode("utf-8")).decode("utf-8")
    chunks: list[bytes] = []

    async with websockets.connect(
        _build_auth_url(),
        open_timeout=TTS_CONNECT_TIMEOUT_SECONDS,
        close_timeout=2,
        ping_interval=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "common": {"app_id": settings.xfyun_app_id},
                    "business": {
                        "aue": "lame",
                        "sfl": 1,
                        "tte": "UTF8",
                        "vcn": voice,
                        "ent": "intp65",
                        "speed": 42,
                        "pitch": 50,
                        "volume": 80,
                        "bgs": 0,
                    },
                    "data": {"status": 2, "text": text_b64},
                },
                ensure_ascii=False,
            )
        )

        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=TTS_RECEIVE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("XFYUN TTS response timed out") from exc

            response = json.loads(raw)
            if response.get("code") != 0:
                raise RuntimeError(f"讯飞 TTS 错误 {response.get('code')}: {response.get('message')}")

            audio = response.get("data", {}).get("audio")
            if audio:
                chunks.append(base64.b64decode(audio))

            if response.get("data", {}).get("status") == 2:
                break

    return b"".join(chunks)


async def synthesize_clone_audio(text: str, res_id: str | None = None) -> bytes:
    if not settings.xfyun_app_id or not settings.xfyun_api_key or not settings.xfyun_api_secret:
        raise RuntimeError("讯飞 TTS 未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET")

    final_res_id = (res_id or settings.xfyun_clone_res_id or load_active_clone_res_id()).strip()
    if not final_res_id:
        raise RuntimeError("一句话复刻音色未配置，请设置 XFYUN_CLONE_RES_ID")

    text_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    chunks: list[bytes] = []

    async with websockets.connect(
        _build_auth_url(VOICE_CLONE_HOST, VOICE_CLONE_PATH),
        open_timeout=TTS_CONNECT_TIMEOUT_SECONDS,
        close_timeout=2,
        ping_interval=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "header": {
                        "app_id": settings.xfyun_app_id,
                        "status": 2,
                        "res_id": final_res_id,
                    },
                    "parameter": {
                        "tts": {
                            "vcn": settings.xfyun_clone_vcn or "x6_clone",
                            "volume": profile["volume"],
                            "rhy": profile["rhy"],
                            "pybuffer": 1,
                            "speed": profile["speed"],
                            "pitch": profile["pitch"],
                            "bgs": 0,
                            "reg": 0,
                            "rdn": 0,
                            "style": profile["style"],
                            "impactFactor": profile["impactFactor"],
                            "audio": {
                                "encoding": "lame",
                                "sample_rate": 24000,
                            },
                        },
                    },
                    "payload": {
                        "text": {
                            "encoding": "utf8",
                            "compress": "raw",
                            "format": "plain",
                            "status": 2,
                            "seq": 0,
                            "text": text_b64,
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=TTS_RECEIVE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("XFYUN voice clone TTS response timed out") from exc

            response = json.loads(raw)
            header = response.get("header", {})
            if header.get("code") not in (0, "0", None):
                raise RuntimeError(f"讯飞复刻 TTS 错误 {header.get('code')}: {header.get('message')}")

            audio = response.get("payload", {}).get("audio", {}).get("audio")
            if audio:
                chunks.append(base64.b64decode(audio))

            if response.get("payload", {}).get("audio", {}).get("status") == 2 or header.get("status") == 2:
                break

    return b"".join(chunks)
