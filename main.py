import asyncio
import base64
import json
import math
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.star import Context, Star, register

from .core import extract_image_prompt, is_bot_mentioned


class ImageGenerationError(RuntimeError):
    pass


@register(
    "astrbot_plugin_at_image",
    "local",
    "群聊艾特生图：仅在明确提出画图意图时生成并发送图片。",
    "1.0.1",
)
class AtImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.api_base = str(config.get("api_base", "https://api.llmgw.xyz")).rstrip("/")
        self.api_key = str(config.get("api_key", "")).strip()
        self.model = str(config.get("model", "gpt-image-2-[1.5k]")).strip()
        self.timeout_seconds = max(30, int(config.get("timeout_seconds", 120)))
        self.cooldown_seconds = max(0, int(config.get("cooldown_seconds", 60)))
        self.max_prompt_length = max(20, int(config.get("max_prompt_length", 500)))
        self.allowed_groups = {str(item) for item in config.get("allowed_groups", [])}
        self.send_progress = bool(config.get("send_progress", True))
        self._bot_id: str | None = None
        self._generation_semaphore = asyncio.Semaphore(1)
        self._guard_lock = asyncio.Lock()
        self._last_generation: dict[str, float] = {}
        self._seen_message_ids: set[str] = set()
        self._seen_order: deque[str] = deque(maxlen=200)

    async def _get_bot_id(self, event: AstrMessageEvent) -> str:
        if self._bot_id is None:
            login_info = await event.bot.api.call_action("get_login_info")
            self._bot_id = str(login_info.get("user_id", ""))
        return self._bot_id

    async def _reserve(self, user_id: str, message_id: str) -> int | None:
        async with self._guard_lock:
            if message_id and message_id in self._seen_message_ids:
                return -1

            now = time.monotonic()
            remaining = self.cooldown_seconds - (now - self._last_generation.get(user_id, 0))
            if remaining > 0:
                return math.ceil(remaining)

            self._last_generation[user_id] = now
            if message_id:
                if len(self._seen_order) == self._seen_order.maxlen:
                    oldest = self._seen_order.popleft()
                    self._seen_message_ids.discard(oldest)
                self._seen_order.append(message_id)
                self._seen_message_ids.add(message_id)
            return None

    def _validate_config(self) -> None:
        if not self.api_key:
            raise ImageGenerationError("插件尚未配置 API Key")
        parsed = urlparse(self.api_base)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ImageGenerationError("API 地址必须是有效的 HTTPS 地址")
        if not self.model:
            raise ImageGenerationError("生图模型不能为空")

    async def _read_image(self, session: aiohttp.ClientSession, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ImageGenerationError("上游返回了无效的图片地址")

        async with session.get(url, allow_redirects=True) as response:
            if response.status != 200:
                raise ImageGenerationError(f"下载图片失败（HTTP {response.status}）")
            content_length = int(response.headers.get("Content-Length", "0") or 0)
            if content_length > 10 * 1024 * 1024:
                raise ImageGenerationError("上游图片超过 10 MB")
            chunks = []
            total = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > 10 * 1024 * 1024:
                    raise ImageGenerationError("上游图片超过 10 MB")
                chunks.append(chunk)
            image_bytes = b"".join(chunks)
            if not image_bytes:
                raise ImageGenerationError("上游返回了空图片")
            return image_bytes

    async def _generate_image(self, prompt: str) -> bytes:
        self._validate_config()
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        endpoint = urljoin(self.api_base + "/", "v1/images/generations")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "low",
        }

        logger.info(
            "[at_image] 开始生图 model=%s prompt_chars=%d",
            self.model,
            len(prompt),
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                body = await response.text()
                logger.info("[at_image] 生图接口返回 status=%d", response.status)
                if response.status != 200:
                    try:
                        message = json.loads(body).get("error", {}).get("message", "")
                    except (json.JSONDecodeError, AttributeError):
                        message = ""
                    detail = message[:120] if message else f"HTTP {response.status}"
                    raise ImageGenerationError(f"生图接口请求失败（{detail}）")
                try:
                    data = json.loads(body).get("data", [])
                    item = data[0]
                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
                    raise ImageGenerationError("生图接口返回格式无效") from exc

                if item.get("b64_json"):
                    try:
                        image_bytes = base64.b64decode(item["b64_json"], validate=True)
                    except (ValueError, TypeError) as exc:
                        raise ImageGenerationError("生图接口返回了无效图片数据") from exc
                    if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
                        raise ImageGenerationError("上游图片为空或超过 10 MB")
                    logger.info("[at_image] 图片已解析 bytes=%d", len(image_bytes))
                    return image_bytes
                if item.get("url"):
                    image_bytes = await self._read_image(session, item["url"])
                    logger.info("[at_image] 图片已解析 bytes=%d", len(image_bytes))
                    return image_bytes
                raise ImageGenerationError("生图接口没有返回图片")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        group_id = str(event.get_group_id())
        if self.allowed_groups and group_id not in self.allowed_groups:
            return
        if not event.is_at_or_wake_command:
            return

        try:
            bot_id = await self._get_bot_id(event)
        except Exception:
            logger.exception("[at_image] 获取机器人账号失败")
            return
        if not bot_id or not is_bot_mentioned(event.get_messages(), event.message_obj, bot_id):
            return

        prompt = extract_image_prompt(event.message_str, self.max_prompt_length)
        if prompt is None:
            return

        event.stop_event()
        if not prompt:
            yield event.plain_result("请在艾特我后说明要画什么，例如：@我 画一只戴耳机的猫。")
            return

        user_id = str(event.get_sender_id())
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        wait_seconds = await self._reserve(user_id, message_id)
        if wait_seconds == -1:
            return
        if wait_seconds is not None:
            yield event.plain_result(f"生图冷却中，请等待 {wait_seconds} 秒。")
            return

        if self.send_progress:
            await self.context.send_message(
                event.unified_msg_origin,
                MessageChain().message("正在画，请稍等。"),
            )

        try:
            async with self._generation_semaphore:
                image_bytes = await self._generate_image(prompt)
        except ImageGenerationError as exc:
            logger.warning("[at_image] %s", exc)
            yield event.plain_result("生图失败，请稍后再试。")
            return
        except Exception:
            logger.exception("[at_image] 未预期的生图错误")
            yield event.plain_result("生图失败，请稍后再试。")
            return

        yield event.chain_result(
            [At(qq=event.get_sender_id()), Plain(" 画好了："), Image.fromBytes(image_bytes)]
        )

    async def terminate(self):
        logger.info("[at_image] 群聊艾特生图插件已卸载")
