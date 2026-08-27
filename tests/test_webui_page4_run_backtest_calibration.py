"""Web UI 页面4（回测校准）接入 run_backtest_calibration 测试（spec §9）。

验证页面4的 3Loop 按钮调用 PredictionPipeline.run_backtest_calibration，
而不是手动构造 history_df + 直接调 run_all_loops（旧路径，产物写错目录）。

关键点：
  1. app.py 源码应包含 run_backtest_calibration 调用
  2. app.py 源码不应包含手动构造 history_df（rows.append / persona_ids 那段）
  3. app.py 源码不应包含 out_dir = ROOT / "output" 作为 3Loop artifacts 目录
     （应通过 run_backtest_calibration 写到 brand_cfg.calibrated_dir）
  4. 功能性验证：page4 的调用签名（predictions + sales_lookup）能正常工作
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "app.py"
APP_SRC = APP_PATH.read_text(encoding="utf-8")


# ============================================================
# 1. 源码级：app.py 调用 run_backtest_calibration
# ============================================================
def test_app_calls_run_backtest_calibration():
    """app.py 页面4 的 3Loop 按钮应调用 PredictionPipeline.run_backtest_calibration，
    而非手动构造 history_df + 直接调 run_all_loops。
    """
    assert "run_backtest_calibration" in APP_SRC, (
        "app.py 应调用 run_backtest_calibration（spec §9 封装路径）"
    )
    # 应通过 PredictionPipeline 实例调用
    assert ".run_backtest_calibration(" in APP_SRC, (
        "应通过 PredictionPipeline 实例调用 run_backtest_calibration"
    )
    print("✅ app.py 调用 PredictionPipeline.run_backtest_calibration")


# ============================================================
# 2. 源码级：移除了手动 history_df 构造
# ============================================================
def test_app_removed_manual_history_df_construction():
    """app.py 页面4 不应再包含手动构造 history_df 的代码
    （persona_ids / feature_cols / persona_row 那段 40 行），
    这部分已封装到 build_history_df。

    注意：page2 批次总表的 rows=[] 是显示用，不在此检查范围。
    """
    # history_df 手动构造的特有标志（page2 summary 不会有这些）
    history_df_markers = [
        'persona_ids = [f"P{i:02d}"',   # 30 人设 ID 构造
        'feature_cols = [f"F{i:02d}"',   # 10 特征列构造
        "persona_row = {}",              # 人设投票行
    ]
    for marker in history_df_markers:
        assert marker not in APP_SRC, (
            f"app.py 不应再手动构造 history_df（已封装到 build_history_df），"
            f"但仍包含：{marker!r}"
        )
    print("✅ app.py 移除了手动 history_df 构造（persona_ids/feature_cols/persona_row）")


# ============================================================
# 3. 源码级：3Loop artifacts 不再写到 output/
# ============================================================
def test_app_no_longer_writes_3loop_to_output_dir():
    """app.py 不应再用 out_dir = ROOT / "output" 作为 run_all_loops 的
    prediction_artifacts_dir。产物应通过 run_backtest_calibration 写到
    brand_cfg.calibrated_dir（下次评估自动加载）。
    """
    # 旧的直接调用：out_dir = ROOT / "output" + run_all_loops(...prediction_artifacts_dir=out_dir)
    assert 'out_dir = ROOT / "output"' not in APP_SRC, (
        "app.py 不应再用 output/ 作为 3Loop artifacts 目录"
    )
    # 不应直接调 run_all_loops（应通过 run_backtest_calibration 间接调）
    # 注意：import 行可能还在，但不应有直接调用
    # 检查是否有 "run_all_loops(" 的直接调用（非 import 行）
    for line in APP_SRC.splitlines():
        stripped = line.strip()
        # 跳过 import 行和注释
        if stripped.startswith("#") or stripped.startswith("from ") or stripped.startswith("import "):
            continue
        if "run_all_loops(" in stripped and "run_backtest_calibration" not in stripped:
            pytest.fail(
                f"app.py 不应直接调 run_all_loops，应通过 run_backtest_calibration：\n"
                f"  {line}"
            )
    print("✅ app.py 不再直接调 run_all_loops，产物写 calibrated_dir")


# ============================================================
# 4. 源码级：成功消息准确（calibrated_dir）
# ============================================================
def test_app_success_message_mentions_calibrated_dir():
    """成功消息应说产物写入 brand_profiles/<brand>/calibrated/，
    这现在是真的（run_backtest_calibration 写到 calibrated_dir）。
    旧代码写 output/ 但消息说 calibrated/，是错的。
    """
    assert "brand_profiles" in APP_SRC and "calibrated/" in APP_SRC, (
        "成功消息应提到 brand_profiles/<brand>/calibrated/"
    )
    print("✅ app.py 成功消息准确（calibrated_dir）")


# ============================================================
# 5. 功能性：page4 的调用签名能正常工作
# ============================================================
def test_page4_call_signature_works():
    """验证 page4 使用的调用签名能正常工作：
    PredictionPipeline(brand_id=..., llm_backend="mock").run_backtest_calibration(
        predictions=preds, sales_lookup=truth_map
    )
    """
    from src.pipeline import PredictionPipeline

    brand_id = "tongzhuang-outdoor"
    pl = PredictionPipeline(brand_id=brand_id, llm_backend="mock")
    preds = pl.run_smoke_test_data(n=10)

    # 注入销量（page4 从 Excel 真实销售结果列解析出的 truth_map）
    sales_lookup = {p.info.style_id: float(200 + i * 30) for i, p in enumerate(preds)}

    result = pl.run_backtest_calibration(
        predictions=preds,
        sales_lookup=sales_lookup,
    )

    assert result is not None, (
        "page4 调用签名应返回有效结果（有销量信号）"
    )
    # 产物写到 calibrated_dir
    assert len(result.output_files) > 0, "应产出 ≥1 个 artifact"
    # 3Loop 三层结果都在
    assert result.loop1 is not None
    assert result.loop2 is not None
    assert result.loop3 is not None
    assert result.residual is not None

    print(f"✅ page4 调用签名工作正常：{len(result.output_files)} 个产物，"
          f"Loop1 applied={result.loop1.applied}, "
          f"Loop2 applied={result.loop2.applied}, "
          f"Loop3 applied={result.loop3.applied}")


# ============================================================
# 6. 功能性：无销量时返回 None（page4 显示跳过提示）
# ============================================================
def test_page4_no_sales_returns_none():
    """page4 调 run_backtest_calibration 时，若 sales_lookup 全 0，
    应返回 None（page4 显示"3Loop 跳过"提示，不抛异常）。
    """
    from src.pipeline import PredictionPipeline

    brand_id = "tongzhuang-outdoor"
    pl = PredictionPipeline(brand_id=brand_id, llm_backend="mock")
    preds = pl.run_smoke_test_data(n=10)

    # 全 0 销量
    sales_lookup = {p.info.style_id: 0.0 for p in preds}

    result = pl.run_backtest_calibration(
        predictions=preds,
        sales_lookup=sales_lookup,
    )

    assert result is None, (
        "全 0 销量应返回 None（page4 显示跳过提示，不抛异常）"
    )
    print("✅ 无销量返回 None（page4 显示跳过提示）")


if __name__ == "__main__":
    test_app_calls_run_backtest_calibration()
    test_app_removed_manual_history_df_construction()
    test_app_no_longer_writes_3loop_to_output_dir()
    test_app_success_message_mentions_calibrated_dir()
    test_page4_call_signature_works()
    test_page4_no_sales_returns_none()
    print("\n🎉 ALL Web UI PAGE 4 TESTS PASSED")
