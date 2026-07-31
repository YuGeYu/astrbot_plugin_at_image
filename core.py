import re
from collections.abc import Iterable
from typing import Any


IMAGE_INTENT_RE = re.compile(
    r"(?:生图|画图|生成(?:一张|张|个|幅)?(?:图片|图)?|"
    r"画(?:一张|张|个|幅)?(?:图片|图|画)?|绘制(?:一张|张|个|幅)?(?:图片|图)?|"
    r"做(?:一张|张|个|幅)?(?:图片|图))\s*[：:，,、]?\s*(.*)",
    re.IGNORECASE,
)


def extract_image_prompt(message: str, max_length: int = 500) -> str | None:
    """Return the prompt following an explicit image-generation intent."""
    match = IMAGE_INTENT_RE.search(message or "")
    if not match:
        return None
    prompt = match.group(1).strip().strip("。.!！?？")
    if not prompt:
        return ""
    return prompt[:max(1, max_length)]


def _raw_segments(message_obj: Any) -> list[dict[str, Any]]:
    raw = getattr(message_obj, "raw_message", None)
    if isinstance(raw, dict):
        segments = raw.get("message", [])
    else:
        segments = getattr(raw, "message", [])
    return segments if isinstance(segments, list) else []


def is_bot_mentioned(
    components: Iterable[Any], message_obj: Any, bot_id: str
) -> bool:
    target = str(bot_id)
    for component in components:
        qq = getattr(component, "qq", None)
        component_type = str(getattr(component, "type", "")).lower()
        if qq is not None and component_type.endswith("at") and str(qq) == target:
            return True

    for segment in _raw_segments(message_obj):
        if segment.get("type") != "at":
            continue
        data = segment.get("data") or {}
        if str(data.get("qq", "")) == target:
            return True
    return False

