"""API 用量追踪 + 额度致命错误快速失败 单元测试。

验证（见 src/llm_client.py 领域定义）：
  1. UsageTracker：按模型累计 calls/input/output tokens，take() 快照并清零
  2. is_fatal_quota_error：额度耗尽/欠费/鉴权失效为致命；429/Throttling 限流不算
  3. _retry_loop 快速失败：致命错误不重试（1 次尝试即返回，无指数退避）
  4. 集成：run_one/run_batch 记录 api_usage；app.py 中止+用量面板+预估提示
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import APIConfig  # noqa: E402
from src.llm_client import (  # noqa: E402
    BailianClient,
    LLMResponse,
    UsageTracker,
    is_fatal_quota_error,
)

APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
PIPELINE_SRC = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")


# ============================================================
# 1. UsageTracker
# ============================================================
def test_usage_tracker_records_and_resets():
    t = UsageTracker()
    t.record("qwen-max", {"input_tokens": 100, "output_tokens": 50})
    t.record("qwen-max", {"input_tokens": 200, "output_tokens": 80})
    t.record("qwen-vl-plus", None)  # 失败调用：calls 计数、token 为 0
    assert t.calls == 3
    assert t.input_tokens == 300
    assert t.output_tokens == 130
    assert t.by_model["qwen-max"] == {
        "calls": 2, "input_tokens": 300, "output_tokens": 130,
    }
    assert t.by_model["qwen-vl-plus"]["calls"] == 1

    snap = t.take()
    assert snap["calls"] == 3
    assert snap["by_model"]["qwen-max"]["calls"] == 2
    # take 后清零（按款切分用）
    assert t.calls == 0 and t.input_tokens == 0 and t.by_model == {}
    snap2 = t.take()
    assert snap2["calls"] == 0


# ============================================================
# 2. is_fatal_quota_error
# ============================================================
def test_fatal_error_patterns():
    fatal_msgs = [
        "API错误 code=400 msg=Arrearage control, please recharge",
        "code=403 msg=Quota exceeded",
        "当前API额度已用完，请充值",
        "Invalid API key provided",
        "Unauthorized (401): api key not valid",
        # 智谱实测鉴权错误（中文 + HTTP 401）
        "API错误 code=401 401 msg=令牌已过期或验证不正确",
        "API错误 code=403 1113 msg=令牌不正确",
    ]
    for m in fatal_msgs:
        assert is_fatal_quota_error(m), f"应判致命: {m}"


def test_retryable_not_fatal():
    retryable = [
        "Requests rate limit exceeded, please retry later (429)",
        "code=429 msg=Throttling.RateQuota",
        "请求被限流，请稍后重试",
        "",  # 空
        "JSON解析失败",  # 普通错误
    ]
    for m in retryable:
        assert not is_fatal_quota_error(m), f"不应判致命: {m}"


# ============================================================
# 3. _retry_loop 快速失败（不依赖 dashscope，直接构造实例）
# ============================================================
def _bare_client(max_retries: int = 3) -> BailianClient:
    """绕过 __init__（避免依赖 dashscope import），只设 _retry_loop 需要的属性。"""
    c = BailianClient.__new__(BailianClient)
    c.cfg = APIConfig(dashscope_api_key="fake-key-for-test", max_retries=max_retries)
    return c


def test_retry_loop_fatal_no_retry():
    """致命错误 → 第 1 次尝试直接返回，不进入指数退避"""
    c = _bare_client(max_retries=3)
    attempts = []

    def _fn(**kwargs):
        attempts.append(1)
        return LLMResponse(content="", model="qwen-max",
                           error="API错误 code=400 msg=Arrearage control")

    t0 = time.time()
    resp = c._retry_loop(_fn, model="qwen-max")
    elapsed = time.time() - t0
    assert len(attempts) == 1, "致命错误不应重试"
    assert elapsed < 0.5, "致命错误不应有指数退避等待"
    assert not resp.ok and "Arrearage" in (resp.error or "")


def test_retry_loop_fatal_exception_no_retry():
    """致命异常（SDK 直接抛错）→ 立即返回错误，不重试不等待"""
    c = _bare_client(max_retries=3)
    attempts = []

    def _fn(**kwargs):
        attempts.append(1)
        raise RuntimeError("Status code 403, code: Arrearage, please recharge")

    t0 = time.time()
    resp = c._retry_loop(_fn, model="qwen-max")
    elapsed = time.time() - t0
    assert len(attempts) == 1
    assert elapsed < 0.5
    assert not resp.ok and "Arrearage" in (resp.error or "")


# ============================================================
# 4. 集成：源码级断言（repo 既有测试风格）
# ============================================================
def test_pipeline_records_api_usage():
    """run_one 真实路径 metadata 记录 api_usage；run_batch 按款 take + 批末汇总"""
    assert '"api_usage": self.client.usage_tracker.take()' in PIPELINE_SRC
    assert 'pred.metadata["api_usage"] = client.usage_tracker.take()' in PIPELINE_SRC
    assert "本批 API 用量" in PIPELINE_SRC


def test_pipeline_failfast_on_quota():
    """run_batch 额度致命错误 → break 中止批次（不再白跑剩余款）"""
    assert "is_fatal_quota_error(str(e))" in PIPELINE_SRC
    break_pos = PIPELINE_SRC.index("is_fatal_quota_error(str(e))")
    loop_break_pos = PIPELINE_SRC.index("break", break_pos)
    assert loop_break_pos - break_pos < 400, "break 应紧跟额度检测"


def test_feature_extraction_propagates_underlying_error():
    """「所有模型特征提取均失败」要带上底层错误，额度报错才能传到上层触发中止"""
    fe_src = (ROOT / "src" / "feature_extraction.py").read_text(encoding="utf-8")
    assert "所有模型特征提取均失败: {'; '.join(errors)" in fe_src


def test_app_has_usage_panel_and_abort_and_estimate():
    """app.py：用量仪表（指标+按模型分解）+ 额度中止 + 提交前预估"""
    assert "API 调用次数" in APP_SRC           # 用量指标
    assert "按模型分解" in APP_SRC             # per-model 分解
    assert "批次因 API 鉴权/额度问题中止" in APP_SRC   # 批次止损中止
    assert "预估本批 API 调用" in APP_SRC       # 提交前成本预估
    assert "from src.llm_client import is_fatal_quota_error" in APP_SRC


if __name__ == "__main__":
    test_usage_tracker_records_and_resets()
    test_fatal_error_patterns()
    test_retryable_not_fatal()
    test_retry_loop_fatal_no_retry()
    test_retry_loop_fatal_exception_no_retry()
    test_pipeline_records_api_usage()
    test_pipeline_failfast_on_quota()
    test_feature_extraction_propagates_underlying_error()
    test_app_has_usage_panel_and_abort_and_estimate()
    print("✅ 用量追踪 + 快速失败测试全部通过")
