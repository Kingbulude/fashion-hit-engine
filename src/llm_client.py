"""LLM/多模态客户端封装：百炼SDK多模型接入 + 限流 + 指数退避重试

设计原则：
1. 不直接依赖dashscope SDK的深层特性，用 dashscope 官方兼容OpenAI的Generation调用方式
2. 文本模型与多模态模型分开封装
3. 调用结果返回统一dict格式：{content, usage, model, error}
4. 异步并发由调用方控制，这里只提供同步调用 + 基础限流
"""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import APIConfig


# ========== 调用结果统一结构 ==========
@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.content != ""


# ========== Token/速率 限流器 ==========
class RateLimiter:
    """简单的分钟级 QPM 限流（令牌桶），对免费层够用"""

    def __init__(self, qpm: int):
        self.qpm = qpm
        self._tokens = qpm
        self._last_refill = time.time()
        self._lock = asyncio.Lock() if False else None  # 同步阶段只用 sleep

    def acquire(self, n: int = 1) -> None:
        now = time.time()
        # 每分钟补充 tokens
        elapsed = now - self._last_refill
        if elapsed >= 60:
            self._tokens = self.qpm
            self._last_refill = now
        else:
            self._tokens = min(self.qpm, self._tokens + self.qpm * elapsed / 60.0)
            # 只更新 if 超过了1秒精度
            if elapsed >= 1.0:
                self._last_refill = now

        if self._tokens >= n:
            self._tokens -= n
            return

        # 等待令牌补充
        need = n - self._tokens
        wait_s = (need / self.qpm) * 60.0 + 0.5
        time.sleep(wait_s)
        self._tokens = self.qpm - n
        self._last_refill = time.time()


# ========== 图片编码辅助 ==========
def encode_image(path: str | Path) -> str:
    """本地图片 -> base64 字符串（百炼多模态接受）"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========== 百炼SDK客户端 ==========
class BailianClient:
    """百炼平台统一客户端（Qwen3-VL / Qwen-Max / Doubao / DeepSeek 等都走百炼）"""

    def __init__(self, api_cfg: APIConfig):
        self.cfg = api_cfg
        if not api_cfg.dashscope_api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 为空。请复制 .env.example 为 .env 并填入百炼API Key"
            )
        try:
            import dashscope  # 延迟导入
        except ImportError as e:
            raise RuntimeError("dashscope未安装，请先 pip install -r requirements.txt") from e

        self._dashscope = dashscope
        dashscope.api_key = api_cfg.dashscope_api_key
        self._limiter = RateLimiter(api_cfg.qpm_limit)

    # ---- 文本生成（人设投票用）----
    def generate_text(
        self,
        prompt: str,
        *,
        model: str = "qwen3-max",
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._retry_loop(
            self._call_text,
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- 多模态图像理解（特征提取用）----
    def generate_multimodal(
        self,
        text_prompt: str,
        image_paths: list[str | Path],
        *,
        model: str = "qwen3-vl-plus",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # 构造内容 list：交替 text + image_url
        content: list[dict[str, Any]] = []

        # 先放图片，再放问题（百炼VL推荐顺序）
        for p in image_paths:
            img_b64 = encode_image(p)
            ext = Path(p).suffix.lower().lstrip(".") or "jpeg"
            mime = "image/png" if ext == "png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            })
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]
        return self._retry_loop(
            self._call_multimodal,
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- 内部调用 ----
    def _call_text(
        self, *, model: str, messages: list[dict],
        temperature: float, max_tokens: int,
    ) -> LLMResponse:
        self._limiter.acquire()
        Gen = self._dashscope.Generation
        resp = Gen.call(
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_response(resp, model)

    def _call_multimodal(
        self, *, model: str, messages: list[dict],
        temperature: float, max_tokens: int,
    ) -> LLMResponse:
        self._limiter.acquire()
        Gen = self._dashscope.MultiModalConversation
        resp = Gen.call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_response(resp, model)

    def _parse_response(self, resp: Any, model: str) -> LLMResponse:
        # dashscope 响应结构：resp.status_code / resp.output / resp.usage
        if getattr(resp, "status_code", None) != 200:
            return LLMResponse(
                content="", model=model,
                error=f"API错误 code={resp.status_code} msg={getattr(resp, 'message', str(resp))}",
            )
        output = getattr(resp, "output", None) or {}
        choices = output.get("choices", [])
        if not choices:
            return LLMResponse(content="", model=model, error="响应无choices")
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        # content可能是list of dict（多模态），取text字段
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
            content = "\n".join(text_parts)
        usage = dict(getattr(resp, "usage", {}) or {})
        return LLMResponse(content=content, model=model, usage=usage, raw=resp)

    # ---- 指数退避重试 ----
    def _retry_loop(self, fn, **kwargs) -> LLMResponse:
        last_err: LLMResponse | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                result = fn(**kwargs)
                if result.ok:
                    return result
                last_err = result
                # 429/限流 -> 等待
                if "429" in (result.error or "") or "rate" in (result.error or "").lower():
                    wait = 2 ** attempt + 3
                    time.sleep(wait)
                    continue
                # 其他错误，直接返回
                return result
            except Exception as e:
                last_err = LLMResponse(content="", model=kwargs.get("model", "?"), error=str(e))
                wait = 2 ** attempt + 1
                time.sleep(wait)
        return last_err or LLMResponse(content="", model="?", error="max_retries exceeded")
