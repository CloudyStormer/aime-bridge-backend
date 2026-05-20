import asyncio
import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from urllib.parse import quote

import websockets

from app.config import settings


IAT_HOST = "iat-api.xfyun.cn"
IAT_PATH = "/v2/iat"
IAT_CHUNK_SIZE = 1280
IAT_SEND_INTERVAL_SECONDS = 0.005
IAT_CONNECT_TIMEOUT_SECONDS = 8
IAT_RECEIVE_TIMEOUT_SECONDS = 12


def _build_auth_url() -> str:
    date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_origin = f"host: {IAT_HOST}\ndate: {date}\nGET {IAT_PATH} HTTP/1.1"
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
        f"wss://{IAT_HOST}{IAT_PATH}"
        f"?authorization={quote(authorization)}"
        f"&date={quote(date)}"
        f"&host={IAT_HOST}"
    )


def _parse_result_piece(result: dict) -> str:
    words = result.get("ws", [])
    return "".join("".join(candidate.get("w", "") for candidate in item.get("cw", [])) for item in words)


async def transcribe_pcm16(audio_bytes: bytes) -> str:
    if not settings.xfyun_app_id or not settings.xfyun_api_key or not settings.xfyun_api_secret:
        raise RuntimeError("讯飞 ASR 未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET")
    if not audio_bytes:
        return ""

    segments: dict[int, str] = {}

    async with websockets.connect(
        _build_auth_url(),
        open_timeout=IAT_CONNECT_TIMEOUT_SECONDS,
        close_timeout=2,
        ping_interval=None,
    ) as websocket:
        offset = 0
        total = len(audio_bytes)

        while offset < total:
            end = min(offset + IAT_CHUNK_SIZE, total)
            is_first = offset == 0
            is_end = end >= total
            status = 0 if is_first else 2 if is_end else 1
            chunk = audio_bytes[offset:end]
            frame = {
                "data": {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
            }

            if is_first:
                frame["common"] = {"app_id": settings.xfyun_app_id}
                frame["business"] = {
                    "language": "zh_cn",
                    "domain": "iat",
                    "accent": "mandarin",
                    "vad_eos": 8000,
                    "dwa": "wpgs",
                }

            await websocket.send(json.dumps(frame, ensure_ascii=False))
            offset = end
            await asyncio.sleep(IAT_SEND_INTERVAL_SECONDS)

        if total <= IAT_CHUNK_SIZE:
            await websocket.send(
                json.dumps(
                    {
                        "data": {
                            "status": 2,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": "",
                        }
                    },
                    ensure_ascii=False,
                )
            )

        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=IAT_RECEIVE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("XFYUN ASR response timed out") from exc

            response = json.loads(raw)
            if response.get("code") != 0:
                raise RuntimeError(f"讯飞 ASR 错误 {response.get('code')}: {response.get('message')}")

            result = response.get("data", {}).get("result")
            if result:
                sn = result.get("sn")
                if result.get("pgs") == "rpl" and result.get("rg"):
                    start, end = result["rg"]
                    for index in range(start, end + 1):
                        segments.pop(index, None)
                if sn is not None:
                    segments[int(sn)] = _parse_result_piece(result)

            if response.get("data", {}).get("status") == 2:
                break

    return "".join(segments[index] for index in sorted(segments)).strip()
