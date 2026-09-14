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
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import APIConfig

log = logging.getLogger(__name__)


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


# ========== API 用量统计（钱花哪了，一眼可见）==========
class UsageTracker:
    """按模型累计 API 调用次数与 token 消耗（从响应 usage 字段取）。"""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_model: dict[str, dict[str, int]] = {}

    def record(self, model: str, usage: dict | None) -> None:
        u = usage or {}
        self.calls += 1
        it = int(u.get("input_tokens", 0) or 0)
        ot = int(u.get("output_tokens", 0) or 0)
        self.input_tokens += it
        self.output_tokens += ot
        m = self.by_model.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        m["calls"] += 1
        m["input_tokens"] += it
        m["output_tokens"] += ot

    def take(self) -> dict[str, Any]:
        """快照当前用量并清零（批次内按款切分用）。"""
        snap = {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_model": {k: dict(v) for k, v in self.by_model.items()},
        }
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_model = {}
        return snap


# ========== 致命错误识别（额度耗尽/欠费/鉴权失效 → 不重试，快速失败）==========
_FATAL_ERROR_PATTERNS = (
    "arrearage", "arrears", "quota", "额度", "insufficient balance",
    "invalid api key", "invalid_api_key", "unauthorized", "forbidden",
    # HTTP 401/403（智谱/百炼错误消息格式均为 "API错误 code=401 ..."）
    "code=401", "code=403",
    # 智谱中文鉴权错误（实测："令牌已过期或验证不正确"）
    "令牌已过期", "令牌不正确", "令牌无效", "api token",
)
_RETRYABLE_HINTS = ("429", "throttling", "rate limit")


def is_fatal_quota_error(msg: str | None) -> bool:
    """额度耗尽/欠费/鉴权失效类错误：重试无意义，应立即止损并提示充值。"""
    if not msg:
        return False
    low = msg.lower()
    # 限流（429/Throttling）是可重试的，不算致命
    if any(h in low for h in _RETRYABLE_HINTS):
        return False
    return any(p in low for p in _FATAL_ERROR_PATTERNS)


# ========== 图片编码辅助 ==========
def encode_image(path: str | Path) -> str:
    """本地图片 -> base64 字符串（百炼多模态接受）"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========== 智谱模型映射（品牌 YAML 里的百炼模型名 → 智谱免费模型）==========
# 品牌适配包/配置里写的是百炼模型名（qwen-vl-plus / qwen-max / deepseek-v3…），
# 切智谱后端时在此处做透明映射，品牌 YAML 一行不用改。
_ZHIPU_MODEL_MAP = {
    # VLM 特征提取 → GLM-4.6V-Flash（永久免费，128K 上下文，视觉同规模 SOTA）
    "qwen-vl-plus": "glm-4.6v-flash",
    "qwen-vl-max": "glm-4.6v-flash",
    "qwen3-vl-plus": "glm-4.6v-flash",
    "qwen2.5-vl-72b-instruct": "glm-4.6v-flash",
    # 文本人设投票 → GLM-4.7-Flash（永久免费，200K 上下文）
    "qwen-max": "glm-4.7-flash",
    "qwen-plus": "glm-4.7-flash",
    "qwen-turbo": "glm-4.7-flash",
    "deepseek-v3": "glm-4.7-flash",
    "deepseek-chat": "glm-4.7-flash",
    "deepseek-r1": "glm-4.7-flash",
}


def _map_zhipu_model(model: str) -> str:
    """百炼模型名 → 智谱模型名；glm-* 原样通过。"""
    return _ZHIPU_MODEL_MAP.get(model, model)


# 智谱免费层限流较保守（并发 1~30 视账号等级），QPM 默认压到 30
_ZHIPU_QPM_DEFAULT = 30


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
        self.usage_tracker = UsageTracker()

    # ---- 文本生成（人设投票用）----
    def generate_text(
        self,
        prompt: str,
        *,
        model: str = "qwen-max",
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
        model: str = "qwen-vl-plus",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # dashscope MultiModalConversation 要求格式：
        # content = [{"image": "data:..."}, {"text": "..."}]
        # key 是 "image"（不是 image_url），值是 data URL 字符串（不是嵌套 dict）
        content: list[dict[str, Any]] = []

        for p in image_paths:
            img_b64 = encode_image(p)
            ext = Path(p).suffix.lower().lstrip(".") or "jpeg"
            mime = "image/png" if ext == "png" else "image/jpeg"
            content.append({"image": f"data:{mime};base64,{img_b64}"})
        content.append({"text": text_prompt})

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
        parsed = self._parse_response(resp, model)
        self.usage_tracker.record(model, parsed.usage)
        return parsed

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
        parsed = self._parse_response(resp, model)
        self.usage_tracker.record(model, parsed.usage)
        return parsed

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
        # dashscope MultiModalConversation 返回 [{"text": "..."}] 无 type 字段
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    text_parts.append(str(c["text"]))
                elif isinstance(c, str):
                    text_parts.append(c)
            content = "\n".join(text_parts)
        usage = dict(getattr(resp, "usage", {}) or {})
        return LLMResponse(content=content, model=model, usage=usage, raw=resp)

    # ---- 指数退避重试 ----
    def _retry_loop(self, fn, **kwargs) -> LLMResponse:
        return retry_with_backoff(fn, max_retries=self.cfg.max_retries, **kwargs)


# ========== 通用指数退避重试（百炼/智谱客户端共用）==========
def retry_with_backoff(fn, *, max_retries: int, **kwargs) -> LLMResponse:
    last_err: LLMResponse | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(**kwargs)
            if result.ok:
                return result
            last_err = result
            # 额度耗尽/欠费/鉴权失效：重试无意义，立即快速失败
            if is_fatal_quota_error(result.error):
                log.error("致命 API 错误（不重试）: %s", result.error)
                return result
            # 429/限流 -> 等待
            if "429" in (result.error or "") or "rate" in (result.error or "").lower():
                wait = 2 ** attempt + 3
                time.sleep(wait)
                continue
            # 其他错误，直接返回
            return result
        except Exception as e:
            err_msg = str(e)
            # 额度耗尽等致命异常：立即返回，不做指数退避
            if is_fatal_quota_error(err_msg):
                log.error("致命 API 异常（不重试）: %s", err_msg)
                return LLMResponse(content="", model=kwargs.get("model", "?"), error=err_msg)
            last_err = LLMResponse(content="", model=kwargs.get("model", "?"), error=err_msg)
            wait = 2 ** attempt + 1
            time.sleep(wait)
    return last_err or LLMResponse(content="", model="?", error="max_retries exceeded")


# ========== 智谱AI客户端（OpenAI 兼容，免费模型）==========
class ZhipuClient:
    """智谱开放平台客户端（GLM-4.7-Flash 文本 / GLM-4.6V-Flash 视觉，均永久免费）。

    与 BailianClient 公开接口完全一致（generate_text / generate_multimodal /
    usage_tracker），pipeline 无需感知后端差异。

    接入方式：OpenAI 兼容 chat/completions（HTTP 直调 requests，无 SDK 依赖）
    - base_url: https://open.bigmodel.cn/api/paas/v4
    - key: open.bigmodel.cn 注册即得（无需信用卡）
    - 免费模型不消耗额度，无每日调用次数限制（仅并发限制，本管线串行调用无影响）

    模型名透明映射：品牌 YAML 里的百炼模型名（qwen-max 等）自动映射到
    对应的智谱免费模型（见 _ZHIPU_MODEL_MAP），品牌配置零改动。
    """

    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_cfg: APIConfig, api_key: str | None = None):
        self.cfg = api_cfg
        key = (api_key or api_cfg.zhipu_api_key or "").strip()
        if not key:
            raise RuntimeError(
                "ZHIPU_API_KEY 为空。请到 https://open.bigmodel.cn/usercenter/apikeys "
                "免费注册创建 Key，粘贴到侧边栏或写入 .env（ZHIPU_API_KEY=...）"
            )
        self._api_key = key
        import requests  # 延迟导入（requirements 已含）
        self._requests = requests
        self._limiter = RateLimiter(api_cfg.qpm_limit or _ZHIPU_QPM_DEFAULT)
        self.usage_tracker = UsageTracker()

    # ---- 文本生成（人设投票用）----
    def generate_text(
        self,
        prompt: str,
        *,
        model: str = "qwen-max",
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        glm_model = _map_zhipu_model(model)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._retry_loop(
            self._call_chat,
            model=glm_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- 多模态图像理解（特征提取用）----
    def generate_multimodal(
        self,
        text_prompt: str,
        image_paths: list[str | Path],
        *,
        model: str = "qwen-vl-plus",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        glm_model = _map_zhipu_model(model)
        # OpenAI 兼容多模态格式：content 数组 image_url(data URL) + text
        content: list[dict[str, Any]] = []
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
            self._call_chat,
            model=glm_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- 内部调用（单次 HTTP）----
    def _call_chat(
        self, *, model: str, messages: list[dict],
        temperature: float, max_tokens: int,
    ) -> LLMResponse:
        self._limiter.acquire()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # 评分任务不需要深度推理：关闭思考模式，输出更稳、更省 token
            "thinking": {"type": "disabled"},
        }
        try:
            resp = self._requests.post(
                self.BASE_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=120,
            )
        except Exception as e:
            return LLMResponse(content="", model=model, error=f"网络错误: {e}")

        if resp.status_code != 200:
            # 提取智谱错误 message（格式 {"error": {"code":…, "message":…}}）
            try:
                err_body = resp.json().get("error", {})
                err_msg = f"API错误 code={resp.status_code} {err_body.get('code', '')} msg={err_body.get('message', resp.text[:300])}"
            except Exception:
                err_msg = f"API错误 code={resp.status_code} msg={resp.text[:300]}"
            return LLMResponse(content="", model=model, error=err_msg)

        try:
            data = resp.json()
        except Exception as e:
            return LLMResponse(content="", model=model, error=f"响应解析失败: {e}")

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="", model=model, error="响应无choices")
        msg = choices[0].get("message", {})
        content_out = msg.get("content", "")
        if isinstance(content_out, list):
            # 兼容分段输出（智谱视觉模型可能返回数组）
            text_parts = []
            for c in content_out:
                if isinstance(c, dict) and "text" in c:
                    text_parts.append(str(c["text"]))
                elif isinstance(c, str):
                    text_parts.append(c)
            content_out = "\n".join(text_parts)
        # usage：OpenAI 兼容 prompt_tokens/completion_tokens → 统一 input/output
        u = data.get("usage", {}) or {}
        usage = {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }
        parsed = LLMResponse(content=content_out or "", model=model, usage=usage, raw=data)
        self.usage_tracker.record(model, parsed.usage)
        return parsed

    # ---- 指数退避重试（与百炼共用同一套逻辑）----
    def _retry_loop(self, fn, **kwargs) -> LLMResponse:
        return retry_with_backoff(fn, max_retries=self.cfg.max_retries, **kwargs)
