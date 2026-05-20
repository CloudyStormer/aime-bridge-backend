import base64
import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone
from urllib.parse import quote

import websockets

from app.config import settings


TTS_HOST = "tts-api.xfyun.cn"
TTS_PATH = "/v2/tts"
TTS_CONNECT_TIMEOUT_SECONDS = 8
TTS_RECEIVE_TIMEOUT_SECONDS = 15


def _build_auth_url() -> str:
    date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_origin = f"host: {TTS_HOST}\ndate: {date}\nGET {TTS_PATH} HTTP/1.1"
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
        f"wss://{TTS_HOST}{TTS_PATH}"
        f"?authorization={quote(authorization)}"
        f"&date={quote(date)}"
        f"&host={TTS_HOST}"
    )


async def synthesize_audio(text: str, voice: str = "x4_yezi") -> bytes:
    if not settings.xfyun_app_id or not settings.xfyun_api_key or not settings.xfyun_api_secret:
        raise RuntimeError("讯飞 TTS 未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET")

    text_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
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
