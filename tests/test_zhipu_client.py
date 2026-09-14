"""智谱免费后端（ZhipuClient）单元测试。

验证（见 src/llm_client.py ZhipuClient）：
  1. 模型映射：品牌 YAML 的百炼模型名 → 智谱免费模型，glm-* 原样通过
  2. 构造：缺 Key 抛错并指引注册地址
  3. 消息格式：文本（system/user）+ 多模态（image_url data URL + text）
  4. usage 转换：OpenAI prompt_tokens/completion_tokens → input/output_tokens
  5. 错误处理：401 鉴权失效判致命（快速失败），429 限流可重试
  6. pipeline 分流：llm_backend=zhipu → 构造 ZhipuClient
  7. app.py 集成：三模式选择器 + Key 按后端分流
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import APIConfig  # noqa: E402
from src.llm_client import (  # noqa: E402
    LLMResponse,
    ZhipuClient,
    _map_zhipu_model,
    is_fatal_quota_error,
)

APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
PIPELINE_SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")


# ============================================================
# 1. 模型映射（品牌 YAML 零改动依赖此映射）
# ============================================================
def test_model_mapping_vlm():
    for m in ("qwen-vl-plus", "qwen-vl-max", "qwen3-vl-plus"):
        assert _map_zhipu_model(m) == "glm-4.6v-flash", m


def test_model_mapping_text():
    for m in ("qwen-max", "qwen-plus", "qwen-turbo", "deepseek-v3", "deepseek-chat"):
        assert _map_zhipu_model(m) == "glm-4.7-flash", m


def test_model_mapping_passthrough():
    for m in ("glm-4.6v-flash", "glm-4.7-flash", "glm-4.5-flash"):
        assert _map_zhipu_model(m) == m


# ============================================================
# 2. 构造校验
# ============================================================
def test_client_missing_key_raises():
    try:
        ZhipuClient(APIConfig(), api_key="")
        raise AssertionError("缺 Key 应抛 RuntimeError")
    except RuntimeError as e:
        assert "open.bigmodel.cn" in str(e), "错误信息要指引智谱注册地址"


def test_client_with_key_ok():
    c = ZhipuClient(APIConfig(zhipu_api_key="test-key"))
    assert c._api_key == "test-key"
    assert c.usage_tracker is not None


# ============================================================
# 3. 消息格式（mock requests，不发真实请求）
# ============================================================
def _mock_client(resp_json=None, status=200) -> tuple[ZhipuClient, list]:
    """构造 ZhipuClient，requests.post 用假对象替换；返回 (client, 调用记录)。"""
    c = ZhipuClient(APIConfig(max_retries=1, qpm_limit=10000), api_key="k")
    calls: list[dict] = []

    class _Resp:
        def __init__(self, j, code):
            self._j = j
            self.status_code = code
            self.text = str(j)

        def json(self):
            return self._j

    class _FakeRequests:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "headers": headers, "json": json})
            if resp_json is None:
                # 成功响应
                return _Resp({
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
                }, 200)
            return _Resp(resp_json, status)

    c._requests = _FakeRequests()
    return c, calls


def test_generate_text_message_format():
    c, calls = _mock_client()
    resp = c.generate_text("打分", model="qwen-max", system_prompt="你是人设")
    assert resp.ok and resp.content == "OK"
    assert resp.model == "glm-4.7-flash", "模型名应映射到智谱"
    payload = calls[0]["json"]
    assert payload["model"] == "glm-4.7-flash"
    assert payload["messages"][0] == {"role": "system", "content": "你是人设"}
    assert payload["messages"][1] == {"role": "user", "content": "打分"}
    assert payload["thinking"] == {"type": "disabled"}, "评分任务应关闭思考模式"
    assert calls[0]["headers"]["Authorization"] == "Bearer k"
    assert calls[0]["url"].endswith("/chat/completions")


def test_generate_multimodal_message_format(tmp_path):
    # 造一张最小 PNG（1x1 像素）
    png = tmp_path / "img.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    c, calls = _mock_client()
    resp = c.generate_multimodal("描述图片", [str(png)], model="qwen-vl-plus")
    assert resp.ok
    payload = calls[0]["json"]
    assert payload["model"] == "glm-4.6v-flash", "VLM 模型名应映射到 glm-4.6v-flash"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "描述图片"}


def test_usage_conversion_and_tracker():
    c, _ = _mock_client()
    c.generate_text("打分", model="qwen-max")
    assert c.usage_tracker.calls == 1
    assert c.usage_tracker.input_tokens == 100
    assert c.usage_tracker.output_tokens == 40
    assert c.usage_tracker.by_model["glm-4.7-flash"]["calls"] == 1


# ============================================================
# 4. 错误处理（智谱错误体格式）
# ============================================================
def test_auth_error_is_fatal():
    c, _ = _mock_client(
        resp_json={"error": {"code": "1002", "message": "Invalid api key"}},
        status=401,
    )
    resp = c.generate_text("打分", model="qwen-max")
    assert not resp.ok
    assert "Invalid api key" in resp.error
    assert is_fatal_quota_error(resp.error), "鉴权失效应判致命（快速失败）"


def test_rate_limit_error_is_retryable():
    c, _ = _mock_client(
        resp_json={"error": {"code": "1302", "message": "Requests rate limit exceeded (429)"}},
        status=429,
    )
    resp = c.generate_text("打分", model="qwen-max")
    assert not resp.ok
    assert not is_fatal_quota_error(resp.error), "限流应可重试，不算致命"


# ============================================================
# 5. pipeline / app 集成（源码级断言）
# ============================================================
def test_pipeline_routes_to_zhipu():
    assert 'if self.llm_backend == "zhipu":' in PIPELINE_SRC
    assert "ZhipuClient(api_cfg)" in PIPELINE_SRC
    assert "zhipu_api_key" in PIPELINE_SRC


def test_app_has_zhipu_mode_and_key_routing():
    # 三模式选择器，智谱免费为默认第一项
    assert '"智谱 GLM（免费）"' in APP_SRC
    assert '"百炼 API（阿里云·付费）"' in APP_SRC
    # Key 按后端分流
    assert 'st.session_state.api_key_for_backend = _key_for_backend' in APP_SRC
    assert "ZHIPU_API_KEY" in APP_SRC
    # 智谱模式缺 Key 的回退提示
    assert "智谱 GLM（免费）」，但未检测到智谱 API Key" in APP_SRC


if __name__ == "__main__":
    test_model_mapping_vlm()
    test_model_mapping_text()
    test_model_mapping_passthrough()
    test_client_missing_key_raises()
    test_client_with_key_ok()
    test_generate_text_message_format()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_generate_multimodal_message_format(Path(td))
    test_usage_conversion_and_tracker()
    test_auth_error_is_fatal()
    test_rate_limit_error_is_retryable()
    test_pipeline_routes_to_zhipu()
    test_app_has_zhipu_mode_and_key_routing()
    print("✅ 智谱免费后端测试全部通过")
