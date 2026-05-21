import hashlib
import json
import time
from typing import Any

import httpx

from app.config import settings


AUTH_URL = "http://avatar-hci.xfyousheng.com/aiauth/v1/token"
TRAIN_BASE_URL = "http://opentrain.xfyousheng.com/voice_train"
TOKEN_TIMEOUT_SECONDS = 15
TRAIN_TIMEOUT_SECONDS = 30


def _json_body(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _md5(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.md5(raw).hexdigest()


def _ensure_credentials() -> None:
    if not settings.xfyun_app_id or not settings.xfyun_api_key:
        raise RuntimeError("讯飞一句话复刻未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY")


async def _get_token(client: httpx.AsyncClient) -> str:
    timestamp = str(int(time.time() * 1000))
    body = _json_body(
        {
            "base": {
                "appid": settings.xfyun_app_id,
                "version": "v1",
                "timestamp": timestamp,
            },
            "model": "remote",
        }
    )
    key_sign = _md5(settings.xfyun_api_key + timestamp)
    authorization = _md5(key_sign + body)
    response = await client.post(
        AUTH_URL,
        content=body.encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": authorization},
        timeout=TOKEN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    token = data.get("accesstoken")
    if not token:
        raise RuntimeError(f"获取讯飞复刻 token 失败: {data}")
    return token


def _signed_headers(token: str, body: str) -> dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    return {
        "Content-Type": "application/json",
        "X-AppId": settings.xfyun_app_id,
        "X-Token": token,
        "X-Time": timestamp,
        "X-Sign": _md5(settings.xfyun_api_key + timestamp + _md5(body)),
    }


async def _post_json(client: httpx.AsyncClient, token: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = _json_body(payload)
    response = await client.post(
        f"{TRAIN_BASE_URL}{path}",
        content=body.encode("utf-8"),
        headers=_signed_headers(token, body),
        timeout=TRAIN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") not in (0, "0"):
        raise RuntimeError(f"讯飞复刻接口失败 {path}: {data}")
    return data


async def get_train_text(text_id: int = 5001) -> dict[str, Any]:
    _ensure_credentials()
    async with httpx.AsyncClient() as client:
        token = await _get_token(client)
        data = await _post_json(client, token, "/task/traintext", {"textId": text_id})
    return data.get("data") or {}


async def create_voice_clone_task(
    *,
    audio_url: str,
    text_id: int = 5001,
    text_seg_id: int = 1,
    resource_name: str = "aime-daily-voice",
    task_name: str = "aime-daily-voice",
) -> str:
    _ensure_credentials()
    async with httpx.AsyncClient() as client:
        token = await _get_token(client)
        task = await _post_json(
            client,
            token,
            "/task/add",
            {
                "engineVersion": "omni_v1",
                "taskName": task_name,
                "resourceName": resource_name,
                "resourceType": 12,
                "sex": 1,
                "ageGroup": 2,
                "denoise": 1,
            },
        )
        task_id = str(task.get("data") or "")
        if not task_id:
            raise RuntimeError(f"创建讯飞复刻训练任务失败: {task}")

        await _post_json(
            client,
            token,
            "/audio/v1/add",
            {
                "taskId": task_id,
                "audioUrl": audio_url,
                "textId": text_id,
                "textSegId": text_seg_id,
            },
        )
        await _post_json(client, token, "/task/submit", {"taskId": task_id})
    return task_id


async def get_voice_clone_result(task_id: str) -> dict[str, Any]:
    _ensure_credentials()
    async with httpx.AsyncClient() as client:
        token = await _get_token(client)
        data = await _post_json(client, token, "/task/result", {"taskId": task_id})
    return data.get("data") or {}
