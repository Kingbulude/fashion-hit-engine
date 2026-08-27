"""残差归一化测试（spec §9.1）。

验证：
1. _rank_percentile 把数值序列正确归一化到 [0, 1]
2. ResidualDecomposer 在归一化输入下能识别 over/underperformer
3. run_all_loops 端到端：构造含异常款的 history_df → 残差在 [-1, 1] 且能识别异常款
4. 回归保护：确认 μ 不再是千级（旧 bug）

Bug 背景：
  原先 run_all_loops 把原始销量（千级）与引擎集成分（0-10）直接相减，
  μ/σ 量级失真、±2σ 区间覆盖全数据 → 残差分离器实际失效。
  修复：spec §9.1 要求 y_i 与 ŷ_i 都 0-1 归一化，先做秩百分位归一化。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.optimization_kernel import (
    ResidualDecomposer,
    _rank_percentile,
    run_all_loops,
)


# ============================================================
# 1. _rank_percentile 单元测试
# ============================================================
def test_rank_percentile_basic():
    """[10, 20, 30] 应映射到 [0, 0.5, 1.0]"""
    result = _rank_percentile([10.0, 20.0, 30.0])
    assert len(result) == 3
    assert abs(result.iloc[0] - 0.0) < 1e-9, f"min 应为 0, got {result.iloc[0]}"
    assert abs(result.iloc[1] - 0.5) < 1e-9, f"mid 应为 0.5, got {result.iloc[1]}"
    assert abs(result.iloc[2] - 1.0) < 1e-9, f"max 应为 1.0, got {result.iloc[2]}"
    print("✅ test_rank_percentile_basic")


def test_rank_percentile_handles_ties():
    """并列值用 average 秩：[5, 5, 10] → [0.25, 0.25, 1.0]"""
    result = _rank_percentile([5.0, 5.0, 10.0])
    # ranks: 5.0 → (1+2)/2=1.5, 5.0 → 1.5, 10.0 → 3
    # (1.5-1)/2 = 0.25, (3-1)/2 = 1.0
    assert abs(result.iloc[0] - 0.25) < 1e-9, f"tie 应为 0.25, got {result.iloc[0]}"
    assert abs(result.iloc[1] - 0.25) < 1e-9, f"tie 应为 0.25, got {result.iloc[1]}"
    assert abs(result.iloc[2] - 1.0) < 1e-9, f"max 应为 1.0, got {result.iloc[2]}"
    print("✅ test_rank_percentile_handles_ties")


def test_rank_percentile_edge_cases():
    """空序列返回空；单元素序列返回 [0.5]"""
    assert len(_rank_percentile([])) == 0
    single = _rank_percentile([42.0])
    assert len(single) == 1
    assert abs(single.iloc[0] - 0.5) < 1e-9
    print("✅ test_rank_percentile_edge_cases")


# ============================================================
# 2. ResidualDecomposer 直接测试（归一化输入）
# ============================================================
def test_residual_decomposer_flags_outlier_when_normalized():
    """构造 1 个异常款（销量最高但预测最低），归一化后应被标为 overperformer"""
    n = 10
    # 9 款对齐：sales 与 pred 都升序
    # 1 款异常：sales 最高，pred 最低
    y_true = list(range(n))  # [0,1,2,...,9]
    y_pred = list(range(n - 1)) + [0]  # [0,1,...,8, 0] —— 最后一个 pred=0（最低）

    meta_df = pd.DataFrame({"style_id": [f"S{i+1:02d}" for i in range(n)]})

    y_true_norm = _rank_percentile(y_true)
    y_pred_norm = _rank_percentile(y_pred)
    result = ResidualDecomposer.decompose(y_true_norm, y_pred_norm, meta_df)

    # 所有残差在 [-1, 1]
    for r in result.residuals:
        assert -1.0 <= r <= 1.0, f"residual {r} 超出 [-1, 1]"

    # μ 应接近 0
    assert abs(result.residual_mean) < 0.3, f"μ={result.residual_mean} 过大"

    # σ 应 > 0（旧 bug 下因 μ/σ 失真，这里可能为 0）
    assert result.residual_std > 0, f"σ={result.residual_std} 应 > 0"

    # S10 应被标为 overperformer（卖得最高，预测最低）
    over_ids = [item["style_id"] for item in result.overperformers]
    assert "S10" in over_ids, f"S10 应为 overperformer, got {over_ids}"

    print(
        f"✅ test_residual_decomposer_flags_outlier_when_normalized: "
        f"μ={result.residual_mean:.3f}, σ={result.residual_std:.3f}, "
        f"over={len(result.overperformers)}, under={len(result.underperformers)}"
    )


# ============================================================
# 3. run_all_loops 端到端测试
# ============================================================
def test_run_all_loops_residual_normalization_end_to_end():
    """构造含异常款的 history_df → run_all_loops → 残差在 [-1,1]、能识别异常款"""
    tmp_dir = Path(tempfile.mkdtemp())
    n = 10
    feature_cols = [f"F{i:02d}" for i in range(1, 11)]
    persona_cols = [f"P{i:02d}" for i in range(1, 31)]

    rows = []
    for i in range(n):
        if i < n - 1:
            # 9 款对齐：sales 与 features 都升序
            sales = float((i + 1) * 1000)  # 1000..9000
            feat_score = 5.0 + i * 0.5  # 5.0..9.0
        else:
            # 异常款：sales 最高，features 最低
            sales = 10000.0
            feat_score = 1.0

        row: dict[str, float | str] = {"style_id": f"T{i+1:03d}", "sales": sales}
        for fc in feature_cols:
            row[fc] = feat_score
        for pc in persona_cols:
            row[pc] = feat_score
        row["persona_score"] = feat_score
        row["channel_score"] = feat_score
        row["price_value_score"] = feat_score
        row["natural_score"] = feat_score
        row["live_score"] = feat_score
        rows.append(row)

    history_df = pd.DataFrame(rows)

    # brand_cfg 传 Path → 成为 calibrated_dir
    calibrated_dir = tmp_dir / "calibrated"
    artifacts_dir = tmp_dir / "artifacts"

    result = run_all_loops(
        brand_cfg=calibrated_dir,
        history_df=history_df,
        prediction_artifacts_dir=artifacts_dir,
        sales_col="sales",
    )

    rd = result.residual

    # 关键回归：残差必须归一化到 [-1, 1]（旧 bug 下 μ=2779, σ=1459）
    for r in rd.residuals:
        assert -1.0 <= r <= 1.0, (
            f"residual {r} 超出 [-1, 1] —— 归一化修复未生效"
        )

    # μ 不再是千级
    assert abs(rd.residual_mean) < 0.5, (
        f"μ={rd.residual_mean} —— 旧 bug 仍存在（原 μ≈2779）"
    )

    # σ 在 (0, 1]（旧 bug 下 σ≈1459）
    assert 0 < rd.residual_std <= 1.0, (
        f"σ={rd.residual_std} 应在 (0, 1]，旧 bug 下 σ≈1459"
    )

    # T010（sales 最高、features 最低）应被标为 overperformer
    over_ids = [item["style_id"] for item in rd.overperformers]
    assert "T010" in over_ids, (
        f"T010 应为 overperformer (sales 最高/预测最低), got {over_ids}"
    )

    # 5 个产物文件全部生成
    assert len(result.output_files) == 5, (
        f"应有 5 个产物文件 (4 YAML + 1 MD), got {len(result.output_files)}"
    )

    # 关键产物文件存在
    assert (calibrated_dir / "residual_decompose.yaml").exists(), "残差 YAML 未生成"
    assert (calibrated_dir / "loop1_vlm_feature_biases.yaml").exists(), "Loop1 YAML 未生成"
    assert (calibrated_dir / "loop2_persona_distribution_weights.yaml").exists(), "Loop2 YAML 未生成"
    assert (calibrated_dir / "loop3_ensemble_weights.yaml").exists(), "Loop3 YAML 未生成"

    # calibration_report.md 在 prediction_artifacts_dir.parent / "calibration"
    report_md = tmp_dir / "calibration" / "calibration_report.md"
    assert report_md.exists(), f"calibration_report.md 未生成于 {report_md}"

    print(
        f"✅ test_run_all_loops_residual_normalization_end_to_end: "
        f"μ={rd.residual_mean:.4f}, σ={rd.residual_std:.4f}, "
        f"over={len(rd.overperformers)}, under={len(rd.underperformers)}, "
        f"files={len(result.output_files)}"
    )


# ============================================================
# 4. 回归测试：旧 bug 量级验证
# ============================================================
def test_old_bug_regression_check():
    """直接验证旧 bug 数据（千级销量 vs 0-10 预测分）现在能正确归一化"""
    y_true_raw = [5400.0, 5050.0, 3400.0, 3150.0, 2950.0]  # 千级销量
    y_pred_raw = [9.7, 9.5, 8.8, 8.6, 8.3]  # 0-10 引擎分

    # 直接相减（旧 bug 行为）
    raw_residuals = [yt - yp for yt, yp in zip(y_true_raw, y_pred_raw)]
    raw_mean = sum(raw_residuals) / len(raw_residuals)
    assert raw_mean > 1000, (
        f"sanity check: 原始 μ 应在千级, got {raw_mean}"
    )

    # 归一化后（新行为）
    y_true_norm = _rank_percentile(y_true_raw)
    y_pred_norm = _rank_percentile(y_pred_raw)
    norm_residuals = [yt - yp for yt, yp in zip(y_true_norm, y_pred_norm)]
    norm_mean = sum(norm_residuals) / len(norm_residuals)
    assert abs(norm_mean) < 0.5, (
        f"归一化后 μ 应 ~0, got {norm_mean}"
    )

    # 归一化残差都在 [-1, 1]
    for r in norm_residuals:
        assert -1.0 <= r <= 1.0, f"归一化残差 {r} 超出 [-1, 1]"

    print(
        f"✅ test_old_bug_regression_check: "
        f"raw μ={raw_mean:.1f} (旧 bug) → normalized μ={norm_mean:.4f} (修复)"
    )


if __name__ == "__main__":
    test_rank_percentile_basic()
    test_rank_percentile_handles_ties()
    test_rank_percentile_edge_cases()
    test_residual_decomposer_flags_outlier_when_normalized()
    test_run_all_loops_residual_normalization_end_to_end()
    test_old_bug_regression_check()
    print("\n🎉 ALL TESTS PASSED")
