import json
from pathlib import Path
from typing import Any

from app.config import settings


def _profile_path() -> Path:
    return Path(settings.voice_clone_profile_path)


def load_active_clone_res_id() -> str:
    path = _profile_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("assetId") or data.get("trainVid") or "").strip()


def save_voice_clone_result(task_id: str, result: dict[str, Any]) -> None:
    asset_id = str(result.get("assetId") or "").strip()
    train_vid = str(result.get("trainVid") or "").strip()
    if not asset_id and not train_vid:
        return

    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskId": task_id,
        "assetId": asset_id,
        "trainVid": train_vid,
        "trainStatus": result.get("trainStatus"),
        "failedDesc": result.get("failedDesc") or "",
        "raw": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
