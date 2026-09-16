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
    """严格串行限流 —— 每次调用之间至少 min_interval 秒。

    智谱免费层官方限 1 并发，且低 RPM，不允许任何突发（token bucket 初始 QPM 个
    token → 可瞬间发 QPM 个请求 → 立即 429 风暴）。漏桶 + 硬间隔是唯一可靠的模式。

    支持跨实例共享（模块级 _SHARED_LIMITERS），防止"每次 new Client 就重置限流状态"。
    """

    # 模块级共享：key=identifier，防止多次点击测试连接/快速切换品牌时重置限流
    _SHARED_LIMITERS: dict[str, "RateLimiter"] = {}

    @classmethod
    def get_or_create(cls, identifier: str, qpm: int, min_interval: float) -> "RateLimiter":
        """获取或创建指定 identifier 的共享限流器（跨实例复用）。"""
        existing = cls._SHARED_LIMITERS.get(identifier)
        if existing is not None:
            return existing
        limiter = cls(qpm, min_interval)
        cls._SHARED_LIMITERS[identifier] = limiter
        return limiter

    @classmethod
    def reset(cls, identifier: str | None = None) -> None:
        """测试/调试用：重置指定或全部共享限流器。"""
        if identifier is None:
            cls._SHARED_LIMITERS.clear()
        else:
            cls._SHARED_LIMITERS.pop(identifier, None)

    def __init__(self, qpm: int, min_interval: float = 2.0):
        """
        qpm: 每分钟最多请求数（仅用于日志/告警，不作为突发上限）
        min_interval: 两次调用之间的最小间隔秒数（硬限速，核心保护）
        """
        self.qpm = qpm
        self.min_interval = min_interval
        self._last_call: float = 0.0

    def acquire(self, n: int = 1) -> None:
        now = time.time()
        elapsed = now - self._last_call
        # 硬限速：每次调用之间至少 min_interval 秒
        if elapsed < self.min_interval:
            wait = self.min_interval - elapsed
            time.sleep(wait)
        self._last_call = time.time()


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
    # ===== 百炼模型名 → 智谱 =====
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
    # ===== Ollama 本地模型名 → 智谱 =====
    # hybrid 模式下 VLM client 是 ZhipuClient，但 feature_extraction
    # 可能传入 Ollama 模型名（如 _resolve_models_from_inputs 返回的），
    # 必须在此处映射到智谱真实模型名，否则 API 报 "模型不存在"
    "qwen2.5vl:7b": "glm-4.6v-flash",
    "qwen2.5-vl:7b": "glm-4.6v-flash",
    "qwen2.5:7b": "glm-4.7-flash",
    "qwen2.5-coder:7b": "glm-4.7-flash",
}


def _map_zhipu_model(model: str) -> str:
    """百炼模型名 → 智谱模型名；glm-* 原样通过。"""
    return _ZHIPU_MODEL_MAP.get(model, model)


# ========== 模型名解析 + 去重（提速关键：映射后重复的只留一个）==========
def resolve_and_dedupe_models(
    client: Any,
    model_names: list[str],
    *,
    is_vlm: bool = False,
) -> list[str]:
    """把品牌 YAML 里的百炼模型名 → 映射到 client 实际可用名，并去重。

    典型场景：
      Ollama 本地: ["qwen-max", "deepseek-v3"] → 都映射到 "qwen2.5:7b" → 去重为 1 个
      智谱云端:   ["qwen-max", "deepseek-v3"] → 都映射到 "glm-4.7-flash" → 去重为 1 个
      百炼云端:   ["qwen-max", "deepseek-v3"] → 两个都是真实不同模型 → 保留 2 个

    不去重时每次人设投票会对同一个模型发多次完全相同的请求，
    本地 Ollama 单卡串行场景下直接浪费 2× 时间。
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for name in model_names:
        # 按 client 类型选映射逻辑
        if isinstance(client, OllamaClient):
            actual = client._resolve_model(name, is_vlm=is_vlm)
        elif isinstance(client, ZhipuClient):
            actual = _map_zhipu_model(name)
        else:
            # BailianClient 或其他：原样通过
            actual = name
        if actual not in seen:
            seen.add(actual)
            resolved.append(actual)
        else:
            log.debug("模型去重: %s → %s（已存在，跳过）", name, actual)
    if len(resolved) < len(model_names):
        log.info(
            "模型去重优化: %d → %d 个（client=%s, is_vlm=%s）",
            len(model_names), len(resolved),
            type(client).__name__, is_vlm,
        )
    return resolved


# 智谱免费层限流保守（1 并发 + 低 RPM），QPM 压到 10 防止 429 风暴
_ZHIPU_QPM_DEFAULT = 10


# ========== 图片压缩（多模态调用提速：payload 缩小 5-10 倍）==========
def downscale_image_data_url(path: str | Path, max_side: int = 1280, quality: int = 85) -> str:
    """图片 → 压缩后的 data URL（JPEG）。

    手机原图 2-5MB → base64 后 3-7MB，上传慢且 VLM 处理慢；
    压到 1280px/JPEG85 对服装细节识别足够，payload 缩小 5-10 倍。
    Pillow 不可用或压缩失败时回退原图。
    """
    raw = Path(path).read_bytes()
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) < len(raw):
                return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception as e:  # Pillow 缺失/图片损坏 → 回退原图
        log.debug("图片压缩回退原图 %s: %s", path, e)
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encode_image(path)}"


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
        self._limiter = RateLimiter.get_or_create(
            "bailian", api_cfg.qpm_limit, min_interval=1.0
        )
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
import random as _random

def retry_with_backoff(
    fn,
    *,
    max_retries: int,
    wait_429_base: float = 0,
    # 1305 是智谱模型级拥塞（全用户共享的模型过载），不是我们 R/QPS 超了
    # 这类错误也需要重试，而且要等更久（模型拥塞缓解比账户限流慢）
    extra_retryable_hints: tuple = (),
    # 连续多少次 1305 就提前 abort（模型真的挂了，再等也没用）
    abort_on_consecutive_1305: int = 3,
    **kwargs,
) -> LLMResponse:
    last_err: LLMResponse | None = None
    consecutive_1305 = 0
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

            err_text = (result.error or "").lower()
            is_1305 = "1305" in err_text
            if is_1305:
                consecutive_1305 += 1
                # 连续 N 次 1305 → 模型确实挂了，提前 abort 让上层 fallback
                if consecutive_1305 >= abort_on_consecutive_1305:
                    log.warning(
                        "[%s] 连续 %d 次 1305 模型拥塞，提前 abort（让上层 fallback）",
                        kwargs.get("model", "?"), consecutive_1305,
                    )
                    return result
            else:
                consecutive_1305 = 0

            is_retryable = (
                "429" in err_text
                or "rate" in err_text
                or any(h in err_text for h in _RETRYABLE_HINTS)
                or any(h.lower() in err_text for h in extra_retryable_hints)
                # 智谱 1305: "该模型当前访问量过大" — 模型级拥塞，必须重试
                or is_1305
                # 智谱 1304: 并发数超出
                or "1304" in err_text
            )

            if is_retryable:
                # 1305 模型拥塞 → 退避更激进（模型恢复比限流慢）
                if "1305" in err_text:
                    base = wait_429_base * 2 if wait_429_base > 0 else 20
                    label = "模型拥塞(1305)"
                elif wait_429_base > 0:
                    base = wait_429_base
                    label = "429限流"
                else:
                    base = 2
                    label = "限流"
                wait = base * (2 ** (attempt - 1))
                # +-30% jitter，避免多个并发请求同时重试形成风暴
                wait *= (0.7 + _random.random() * 0.6)
                wait = round(wait, 1)
                log.warning(
                    "[%s] %s，第 %d/%d 次重试，等待 %.1fs ... 错误: %s",
                    kwargs.get("model", "?"), label, attempt, max_retries, wait,
                    result.error[:120],
                )
                time.sleep(wait)
                continue
            # 其他不可重试错误，直接返回
            log.warning("[%s] 不可重试错误直接返回: %s", kwargs.get("model", "?"), result.error[:120])
            return result
        except Exception as e:
            err_msg = str(e)
            # 额度耗尽等致命异常：立即返回，不做指数退避
            if is_fatal_quota_error(err_msg):
                log.error("致命 API 异常（不重试）: %s", err_msg)
                return LLMResponse(content="", model=kwargs.get("model", "?"), error=err_msg)
            last_err = LLMResponse(content="", model=kwargs.get("model", "?"), error=err_msg)
            wait = (2 ** attempt) + _random.random() * 2
            log.warning("[%s] 异常重试 %d/%d: %s", kwargs.get("model", "?"), attempt, max_retries, err_msg[:120])
            time.sleep(wait)
    log.error("[%s] 重试耗尽（%d 次），最后错误: %s", kwargs.get("model", "?"), max_retries,
              (last_err.error if last_err else "unknown")[:200])
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
        # 免费层限速严格：1 并发 + 低 RPM（账户级 RPM 比模型级更紧）
        # 2.5s 硬间隔 + 指数退避重试（应对 1305 模型拥塞）
        self._limiter = RateLimiter.get_or_create(
            "zhipu", qpm=8, min_interval=2.5,
        )
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
        # 图片先压缩（payload 缩小 5-10 倍，上传+识别都提速）
        content: list[dict[str, Any]] = []
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": downscale_image_data_url(p)},
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
            # 带上真实 GLM 模型名：上层日志（人设投票/特征提取）显示的是配置别名
            # （qwen-max 等），这里自曝真实模型，排查限流/参数错误不歧义
            try:
                err_body = resp.json().get("error", {})
                err_msg = (
                    f"API错误 code={resp.status_code} {err_body.get('code', '')} "
                    f"模型={model} msg={err_body.get('message', resp.text[:300])}"
                )
            except Exception:
                err_msg = f"API错误 code={resp.status_code} 模型={model} msg={resp.text[:300]}"
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

    # ---- 指数退避重试（智谱 1305/429 都要扛过去，6 次足够了）----
    def _retry_loop(self, fn, **kwargs) -> LLMResponse:
        # 硬编码 6 次重试：智谱免费层 1305 模型拥塞 + 429 限流都可能来，
        # 3 次根本不够；Bailian 等其他后端可以继续用 config 默认值
        zhipu_retries = max(self.cfg.max_retries, 6)
        return retry_with_backoff(
            fn, max_retries=zhipu_retries, wait_429_base=15.0, **kwargs,
        )


# ========== Ollama 本地客户端（混合路线：本地 VLM + 云端文本）==========
class OllamaClient:
    """本地 Ollama 推理客户端（OpenAI 兼容 /chat/completions）。

    专为 RTX 3070 8GB 等消费级显卡设计：7B Q4_K_M 量化模型 ~4.5GB 显存，
    7B VLM ~6.5GB 显存，RTX 3070 能稳稳跑。

    支持两类场景：
    - **VLM 视觉**：qwen2.5-vl:7b / qwen2.5-vl:7b-instruct（图片特征提取）
    - **文本**：qwen2.5:7b / qwen2.5-coder:7b（人设投票，可选）

    Ollama 默认 http://localhost:11434，零 API Key、零成本、零限流。
    使用完自动释放显存（keep_alive 由 Ollama 服务控制，默认可设 10s）。
    """

    DEFAULT_BASE_URL = "http://localhost:11434/api"
    # 默认模型：先用 VLM 做特征提取
    DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
    DEFAULT_TEXT_MODEL = "qwen2.5:7b"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        vlm_model: str | None = None,
        text_model: str | None = None,
        num_ctx: int = 2048,
        keep_alive: str = "10s",
    ) -> None:
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.vlm_model = vlm_model or self.DEFAULT_VLM_MODEL
        self.text_model = text_model or self.DEFAULT_TEXT_MODEL
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.usage_tracker = UsageTracker()

        import requests  # 延迟导入
        self._requests = requests

    # ---- Ollama 健康检查（快速诊断服务是否在线）----
    def health_check(self) -> tuple[bool, str]:
        """返回 (是否在线, 诊断信息)。"""
        try:
            # 1. 先试 /api/tags（列出已下载的所有模型，不论是否在显存）
            base = self.base_url.replace("/api", "")
            resp = self._requests.get(f"{base}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code} from Ollama"

            # Ollama 新旧版本响应格式不同：
            #   旧: {"models": [{"name": "xxx"}, ...]}
            #   新: [{"name": "xxx"}, ...]  或  {"models": [{"model": "xxx"}]}
            data = resp.json()
            model_entries: list[dict]
            if isinstance(data, list):
                model_entries = data
            else:
                model_entries = data.get("models", [])

            disk_models = []
            for m in model_entries:
                name = m.get("name") or m.get("model") or m.get("Model", "")
                if name:
                    disk_models.append(name)

            # 2. 再查 /api/ps（当前正在显存里运行的模型）
            running_models: list[str] = []
            try:
                ps_resp = self._requests.get(f"{base}/api/ps", timeout=3)
                if ps_resp.status_code == 200:
                    ps_data = ps_resp.json()
                    ps_entries: list[dict]
                    if isinstance(ps_data, list):
                        ps_entries = ps_data
                    else:
                        ps_entries = ps_data.get("models", [])
                    for m in ps_entries:
                        name = m.get("name") or m.get("model") or ""
                        if name:
                            running_models.append(name)
            except Exception:
                pass  # /api/ps 不关键，失败不阻塞

            # 3. 拼结果
            if disk_models:
                msg = f"在线，磁盘上有 {len(disk_models)} 个模型: {', '.join(disk_models)}"
                if running_models:
                    msg += f"（当前显存加载中: {', '.join(running_models)}）"
                return True, msg
            else:
                # Ollama 在线但没找到任何模型 → 很可能是 OLLAMA_MODELS 配置问题
                return True, (
                    "Ollama 在线，但磁盘上没找到任何模型。\n"
                    "可能原因：OLLAMA_MODELS 环境变量没生效，或模型还在 C 盘。\n"
                    "请在本机 PowerShell 执行：ollama list 确认模型存在。"
                )
        except self._requests.ConnectionError:
            return False, "Ollama 未启动（localhost:11434 连接被拒）"
        except Exception as e:
            return False, f"检查失败: {e}"

    # ---- 内部：百炼/智谱模型名 → Ollama 模型名自动映射 ----
    _MODEL_ALIAS: dict[str, str] = {
        # 百炼 / 智谱 VLM → Ollama VLM
        "qwen-vl-plus": "qwen2.5vl:7b",
        "qwen-vl-max": "qwen2.5vl:7b",
        "qwen3-vl-plus": "qwen2.5vl:7b",
        "qwen2.5-vl-72b-instruct": "qwen2.5vl:7b",
        "glm-4v-flash": "qwen2.5vl:7b",
        "glm-4.6v-flash": "qwen2.5vl:7b",
        # 百炼 / 智谱文本 → Ollama 文本
        "qwen-max": "qwen2.5:7b",
        "qwen-plus": "qwen2.5:7b",
        "qwen-turbo": "qwen2.5:7b",
        "glm-4-flash": "qwen2.5:7b",
        "glm-4.7-flash": "qwen2.5:7b",
        # DeepSeek 系列（人设投票默认用 deepseek-v3）
        "deepseek-v3": "qwen2.5:7b",
        "deepseek-chat": "qwen2.5:7b",
        "deepseek-r1": "qwen2.5:7b",
    }

    def _resolve_model(self, model: str, *, is_vlm: bool) -> str:
        """把上游传来的百炼/智谱模型名映射为 Ollama 实际可用名。"""
        if not model:
            return self.vlm_model if is_vlm else self.text_model
        # 已经是 Ollama 模型名（含 : 或在本地 ollama list 里）
        if model in self._MODEL_ALIAS:
            return self._MODEL_ALIAS[model]
        return model

    # ---- 文本生成（人设投票用）----
    def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        target_model = self._resolve_model(model or self.text_model, is_vlm=False)
        return self._call_chat(
            model=target_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- 多模态图像理解（特征提取用）----
    def generate_multimodal(
        self,
        text_prompt: str,
        image_paths: list[str | Path],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        target_model = self._resolve_model(model or self.vlm_model, is_vlm=True)

        # Ollama /api/chat 多模态格式：
        #   messages.content = string（纯文本）
        #   payload.images = [base64, base64, ...]  ← 顶层字段
        images_b64 = [encode_image(p) for p in image_paths]
        messages = [{"role": "user", "content": text_prompt}]

        return self._call_chat(
            model=target_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            images=images_b64,
        )

    # ---- 内部调用 ----
    def _call_chat(
        self, *, model: str, messages: list[dict],
        temperature: float, max_tokens: int,
        images: list[str] | None = None,
    ) -> LLMResponse:
        # === Ollama 格式强制规范 ===
        # Ollama /api/chat 的硬约束：
        #   1. messages[].content 必须是 string（不能是 array）
        #   2. 图片走 payload 顶层 images=[base64, ...]
        # 不管上游是谁（BailianClient 签名、旧 feature_extraction 路径、
        # 还是有人手工构造 OpenAI array 格式），在这里统一转一次。
        normalized_messages: list[dict[str, Any]] = []
        extracted_images: list[str] = list(images) if images else []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                # 已经是 Ollama 格式
                normalized_messages.append({"role": msg.get("role", "user"), "content": content})
            elif isinstance(content, list):
                # OpenAI array 格式 → 提取 text + image_url
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(str(part["text"]))
                        elif part.get("type") == "image_url":
                            img_info = part.get("image_url", {})
                            url = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                            if url.startswith("data:"):
                                # data:image/jpeg;base64,xxxxx → 只取 base64
                                extracted_images.append(url.split(",", 1)[-1])
                            elif url:
                                # 纯 base64
                                extracted_images.append(url)
                        elif "image" in part:
                            img_val = part["image"]
                            if isinstance(img_val, str) and img_val.startswith("data:"):
                                extracted_images.append(img_val.split(",", 1)[-1])
                            elif isinstance(img_val, str):
                                extracted_images.append(img_val)
                normalized_messages.append({
                    "role": msg.get("role", "user"),
                    "content": "\n".join(text_parts) if text_parts else "(image-only request)",
                })
            else:
                normalized_messages.append({"role": msg.get("role", "user"), "content": str(content)})

        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
            "keep_alive": self.keep_alive,
        }
        if extracted_images:
            payload["images"] = extracted_images
        try:
            resp = self._requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=300,  # VLM 首 token 可能慢，留足时间
            )
        except Exception as e:
            return LLMResponse(content="", model=model, error=f"Ollama 网络错误: {e}")

        if resp.status_code != 200:
            err_text = resp.text[:300]
            if "model not found" in err_text.lower():
                hint = (
                    f"Ollama 未找到模型 `{model}`。"
                    f"请先在终端执行：ollama pull {model}"
                )
                return LLMResponse(content="", model=model, error=hint)
            return LLMResponse(
                content="", model=model,
                error=f"Ollama HTTP {resp.status_code}: {err_text}",
            )

        try:
            data = resp.json()
        except Exception as e:
            return LLMResponse(content="", model=model, error=f"Ollama 响应解析失败: {e}")

        # === 防御：Ollama 响应可能不是 dict（理论上不会，但以防万一）===
        if isinstance(data, list):
            # 偶尔 Ollama 会返回数组（不该发生）
            data = data[0] if data else {}
        elif not isinstance(data, dict):
            data = {"message": {"content": str(data)}}

        message_field = data.get("message", {})
        if isinstance(message_field, list):
            message_field = message_field[0] if message_field else {}
        elif not isinstance(message_field, dict):
            message_field = {"content": str(message_field) if message_field else ""}

        content_out = message_field.get("content", "") or ""
        if isinstance(content_out, list):
            text_parts = [
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content_out
            ]
            content_out = "\n".join(p for p in text_parts if p)

        # Ollama /api/chat 的 usage 是扁平字段（不是嵌套的 usage dict！）：
        #   { "prompt_eval_count": N, "eval_count": N, ... }
        # Bailian/Zhipu 才是嵌套 usage dict：
        #   { "usage": {"input_tokens": N, "output_tokens": N} }
        # 所以先试扁平，取不到再试嵌套
        input_tok = (
            int(data.get("prompt_eval_count", 0) or 0)
            or int(data.get("usage", {}).get("input_tokens", 0) or 0)
        )
        output_tok = (
            int(data.get("eval_count", 0) or 0)
            or int(data.get("usage", {}).get("output_tokens", 0) or 0)
        )
        usage_normalized = {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "total_tokens": input_tok + output_tok,
        }
        parsed = LLMResponse(
            content=content_out.strip(), model=model,
            usage=usage_normalized, raw=data,
        )
        self.usage_tracker.record(model, parsed.usage)
        return parsed
