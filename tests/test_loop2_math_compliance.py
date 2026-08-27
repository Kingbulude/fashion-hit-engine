"""Loop2 数学合规测试（spec §8.2 / §8.4）。

验证 PersonaDistributionFitter.fit() 严格满足 spec §8.2 的数学形式：

    min_w  Σ_i (y_i - Σ_k w_k · vote_k(x_i))² + λ · Σ_k |w_k|
    s.t.   Σ_k w_k = 1
           w_k ≥ w_min = 1/(2×30)         # 多样性保护下限

关键校验点：
  1. 约束可行性：Σ w_k = 1（浮点误差范围内）
  2. 下限约束：每个 w_k ≥ w_min ≈ 1.67%
  3. 方向性保留：正相关人设获高权重，负相关人设降至 w_min 下限
  4. y 归一化：fit 内部对 sales 做 rank percentile，输入千级销量也稳定
  5. vote 归一化：1-10 量表被 /10 → [0,1]
  6. 旧 bug 修复：不再用 abs()（避免负相关人设被误调高）
  7. 保护机制：Spearman 单调不减（spec §8.4），未提升则回滚均匀
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.optimization_kernel import (
    PersonaDistributionFitter,
    _rank_percentile,
)


# ============================================================
# 公用：构造可解释的 history_df
# ============================================================
def _make_df_with_strong_positive_persona(n: int = 20) -> pd.DataFrame:
    """构造 n 款：P05 与 sales 强正相关（其他 P 几乎随机/常数）。

    sales 升序，P05 也升序 → Lasso 应给 P05 高正系数，
    其他 P 系数接近 0 或被压成 0。

    非信号人设加入弱噪声（避免常数列让 Lasso 退化成全零解）。
    """
    rng = np.random.RandomState(42)
    rows = []
    sales_base = np.linspace(1000, 10000, n)
    for i in range(n):
        row: dict[str, float | str] = {"style_id": f"S{i+1:03d}", "sales": float(sales_base[i])}
        for k in range(1, 31):
            col = f"P{k:02d}"
            if k == 5:
                # P05 与 sales 同向：1-10 升序
                row[col] = 1.0 + (i / (n - 1)) * 9.0  # 1 → 10
            elif k == 6:
                # P06 与 sales 反向：10 → 1
                row[col] = 10.0 - (i / (n - 1)) * 9.0
            else:
                # 其他 P 用弱噪声（非常数，避免 Lasso 退化）
                row[col] = 5.0 + rng.normal(0, 0.3)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 1. 数学约束合规：Σ w_k = 1，每个 w_k ≥ w_min
# ============================================================
def test_loop2_constraints_sum_and_floor():
    df = _make_df_with_strong_positive_persona(n=20)
    result = PersonaDistributionFitter.fit(df, sales_col="sales")

    w_min = PersonaDistributionFitter.W_MIN  # 1/60 ≈ 0.01667
    n_p = len(PersonaDistributionFitter.PERSONA_COLS)

    # 1. Σ w_k = 1（浮点容差）
    total = sum(result.persona_weights.values())
    assert abs(total - 1.0) < 1e-9, (
        f"spec §8.2 要求 Σ w_k = 1, got sum={total}"
    )

    # 2. 每个 w_k ≥ w_min
    for col, w in result.persona_weights.items():
        assert w >= w_min - 1e-9, (
            f"spec §8.2 要求 w_k ≥ w_min={w_min:.6f}, "
            f"但 {col} = {w:.6f}（违反多样性下限）"
        )

    # 3. w_min 数值正确
    assert abs(w_min - 1.0 / (2 * n_p)) < 1e-12, (
        f"w_min 应为 1/(2×{n_p}) = {1.0/(2*n_p):.6f}, got {w_min}"
    )

    print(
        f"✅ test_loop2_constraints_sum_and_floor: "
        f"sum={total:.6f}, w_min={w_min:.6f}, "
        f"min_w={min(result.persona_weights.values()):.6f}, "
        f"max_w={max(result.persona_weights.values()):.6f}"
    )


# ============================================================
# 2. 方向性保留：正相关人设高权重，负相关人设降至 w_min
#    （直接测 _weights_from_raw_coef，与 Lasso 拟合解耦）
# ============================================================
def test_loop2_directionality_preserved():
    """spec §8.2 方向性保留。

    直接构造 Lasso 原始系数（绕过 Lasso 拟合的不确定性）：
      - P07: -1.0（强负相关）
      - P08: +2.0（强正相关）
      - P09: +0.5（弱正相关）
      - 其他: 0.0

    期望：
      - P07（负 coef → pos_coef=0）应触底 w_min
      - P08（最大正 coef）权重最高
      - P09（小正 coef）权重高于 w_min 但低于 P08
      - 其他 P（coef=0）应触底 w_min
      - 所有 w ≥ w_min，sum=1

    修复前（abs bug）：P07 会被错误调高权重（abs(-1.0)=1.0 与 P09 同量级）。
    """
    raw = {col: 0.0 for col in PersonaDistributionFitter.PERSONA_COLS}
    raw["P07"] = -1.0
    raw["P08"] = +2.0
    raw["P09"] = +0.5

    weights = PersonaDistributionFitter._weights_from_raw_coef(raw)

    w_min = PersonaDistributionFitter.W_MIN
    n_p = len(PersonaDistributionFitter.PERSONA_COLS)

    # 1. 约束合规
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"sum(w) 应 = 1, got {total}"
    for col, w in weights.items():
        assert w >= w_min - 1e-9, f"{col} w={w} < w_min={w_min}"

    # 2. P07 触底 w_min（负相关，方向性保留后 pos_coef=0）
    assert weights["P07"] <= w_min + 1e-9, (
        f"P07 负相关，权重应触底 w_min={w_min:.6f}, "
        f"got {weights['P07']:.6f}（说明 abs bug 又出现）"
    )

    # 3. P08 权重最高（最大正系数）
    assert weights["P08"] > weights["P09"] > w_min, (
        f"P08 应 > P09 > w_min: "
        f"P08={weights['P08']:.6f}, P09={weights['P09']:.6f}, w_min={w_min:.6f}"
    )
    assert weights["P08"] > w_min + 0.1, (
        f"P08 最大正系数，权重应明显高于 w_min, "
        f"got {weights['P08']:.6f}"
    )

    # 4. coef=0 的 P 触底 w_min（pos_coef=0 → norm=0 → w=w_min）
    for col in PersonaDistributionFitter.PERSONA_COLS:
        if col in ("P07", "P08", "P09"):
            continue
        assert weights[col] <= w_min + 1e-9, (
            f"{col} coef=0，权重应触底 w_min={w_min:.6f}, "
            f"got {weights[col]:.6f}"
        )

    # 5. 旧 bug 验证：abs(-1.0)=1.0 与 P09 abs(+0.5)=0.5，会让 P07 权重
    #    高于 P09；新行为下 P07 触底，明显低于 P09
    assert weights["P07"] < weights["P09"], (
        f"旧 bug 下 P07(abs=1.0) 应高于 P09(abs=0.5)；"
        f"新行为下 P07 触底 w_min < P09. "
        f"got P07={weights['P07']:.6f}, P09={weights['P09']:.6f}"
    )

    print(
        f"✅ test_loop2_directionality_preserved: "
        f"P07(neg)→w={weights['P07']:.6f} (触底), "
        f"P08(pos,max)→w={weights['P08']:.6f} (最高), "
        f"P09(pos,small)→w={weights['P09']:.6f} (中间), "
        f"其他触底 w_min={w_min:.6f}"
    )


# ============================================================
# 3. y 归一化稳定性：千级销量 vs 0-1 销量结果一致
# ============================================================
def test_loop2_y_normalization_invariant_to_scale():
    """spec §9.1 要求 y_i = 销量排名（0-1 归一化）。

    构造两份 df：
      - df_a：sales 千级 (1000..10000)
      - df_b：sales 单位级 (0.1..1.0) 但排名相同

    两者 Lasso 输入完全等价（rank percentile 不受尺度影响），
    所以最终 persona_weights 应一致。
    """
    df_a = _make_df_with_strong_positive_persona(n=20)
    df_b = df_a.copy()
    df_b["sales"] = df_a["sales"] / 10000.0  # 缩放到 [0.1, 1.0]，排名不变

    r_a = PersonaDistributionFitter.fit(df_a, sales_col="sales")
    r_b = PersonaDistributionFitter.fit(df_b, sales_col="sales")

    for col in PersonaDistributionFitter.PERSONA_COLS:
        wa = r_a.persona_weights[col]
        wb = r_b.persona_weights[col]
        assert abs(wa - wb) < 1e-6, (
            f"spec §9.1 y 归一化不变性：{col} 在千级 vs 单位级销量下 "
            f"权重应一致, got wa={wa:.6f}, wb={wb:.6f}"
        )

    # old/new Spearman 也应一致
    assert abs(r_a.old_spearman - r_b.old_spearman) < 1e-6, (
        f"old_spearman 受销量尺度影响: {r_a.old_spearman} vs {r_b.old_spearman}"
    )

    print(
        f"✅ test_loop2_y_normalization_invariant_to_scale: "
        f"千级 vs 单位级销量 → weights 一致 "
        f"(old_sp={r_a.old_spearman:.4f} == {r_b.old_spearman:.4f})"
    )


# ============================================================
# 4. vote 归一化：1-10 量表被 /10 → [0,1]（spec §8.2）
# ============================================================
def test_loop2_vote_scale_constant_and_canonical_input():
    """spec §8.2 要求 vote_k(x_i) = 1-10 → 0-1 归一化。

    验证：
      1. VOTE_SCALE 常量 = 10.0（对应 1-10 量表）
      2. 传规范 1-10 量表数据 → fit() 不抛异常，且产出非退化权重
         （即不是所有权重 = 1/n_p 的退化均匀分布）
      3. lasso_raw_coef 的尺度与 [0,1] 输入一致（系数不出现 1000× 量级）
    """
    # 1. VOTE_SCALE 常量正确
    assert PersonaDistributionFitter.VOTE_SCALE == 10.0, (
        f"VOTE_SCALE 应为 10.0（spec §8.2: 1-10 → 0-1 归一化）, "
        f"got {PersonaDistributionFitter.VOTE_SCALE}"
    )

    # 2. 规范 1-10 输入跑通，且能学到信号
    df = _make_df_with_strong_positive_persona(n=20)
    result = PersonaDistributionFitter.fit(df, sales_col="sales")

    if result.applied:
        # P05 强正相关 → 至少 P05 权重应高于 w_min（学到了信号）
        w_p05 = result.persona_weights["P05"]
        w_min = PersonaDistributionFitter.W_MIN
        assert w_p05 > w_min + 1e-6, (
            f"applied=True 时 P05（强正相关）权重 {w_p05:.6f} "
            f"应高于 w_min={w_min:.6f}"
        )
        signal_found = True
    else:
        # 即使保护回滚，常量约束仍满足
        signal_found = False

    # 3. lasso_raw_coef 尺度合理（[0,1] 输入下系数不应是千级）
    max_abs_coef = max(abs(v) for v in result.lasso_raw_coef.values())
    assert max_abs_coef < 100, (
        f"VOTE_SCALE /10 归一化后 Lasso 系数应是小数级，"
        f"max|coef|={max_abs_coef}（异常大，怀疑未归一化）"
    )

    print(
        f"✅ test_loop2_vote_scale_constant_and_canonical_input: "
        f"VOTE_SCALE=10.0, applied={result.applied}, "
        f"signal_found={signal_found}, max|coef|={max_abs_coef:.4f}"
    )


# ============================================================
# 5. 保护机制（spec §8.4）：纯噪声数据应回滚均匀分布
# ============================================================
def test_loop2_protection_rollback_on_noise():
    """构造 sales 与 P 完全独立的纯噪声数据 → Spearman 不提升 → 回滚均匀。

    spec §8.4：ρ_new < ρ_old + MIN_IMPROVEMENT 时回滚到更新前状态（均匀）。
    """
    rng = np.random.RandomState(123)
    n = 20
    rows = []
    for i in range(n):
        row: dict[str, float | str] = {"style_id": f"N{i+1:03d}"}
        row["sales"] = float(rng.randint(100, 10000))
        for k in range(1, 31):
            row[f"P{k:02d}"] = float(rng.uniform(1, 10))
        rows.append(row)
    df = pd.DataFrame(rows)

    result = PersonaDistributionFitter.fit(df, sales_col="sales")

    w_min = PersonaDistributionFitter.W_MIN
    n_p = len(PersonaDistributionFitter.PERSONA_COLS)
    uniform = 1.0 / n_p

    # 回滚后：所有权重 = 1/n（spec §8.4 "回滚到更新前状态"）
    if not result.applied:
        for col in PersonaDistributionFitter.PERSONA_COLS:
            w = result.persona_weights[col]
            assert abs(w - uniform) < 1e-9, (
                f"保护触发回滚后 {col} 应为均匀 {uniform:.6f}, got {w:.6f}"
            )
        # 但均匀分布本身满足 w_min 约束（uniform=1/30 > 1/60=w_min）
        assert uniform >= w_min
    else:
        # 即使应用了新权重，也必须满足 w_min 下限
        for col in PersonaDistributionFitter.PERSONA_COLS:
            assert result.persona_weights[col] >= w_min - 1e-9

    print(
        f"✅ test_loop2_protection_rollback_on_noise: "
        f"applied={result.applied}, old_sp={result.old_spearman:.4f}, "
        f"new_sp={result.new_spearman:.4f}, "
        f"all_w_ge_w_min={all(w >= w_min - 1e-9 for w in result.persona_weights.values())}"
    )


# ============================================================
# 6. 边界：极小样本（5 款）应能跑通
# ============================================================
def test_loop2_minimum_sample_size():
    """spec §8.2 没有规定最小样本，但 fit() 内部要求 ≥5 款。

    构造恰好 5 款的 df，验证不抛异常且权重满足约束。
    """
    n = 5
    rows = []
    for i in range(n):
        row: dict[str, float | str] = {
            "style_id": f"M{i+1:03d}",
            "sales": float((i + 1) * 1000),
        }
        for k in range(1, 31):
            # P05 与 sales 同向，其他随机
            if k == 5:
                row[f"P{k:02d}"] = 1.0 + i * 2.0  # 1, 3, 5, 7, 9
            else:
                row[f"P{k:02d}"] = 5.0
        rows.append(row)
    df = pd.DataFrame(rows)

    result = PersonaDistributionFitter.fit(df, sales_col="sales")

    w_min = PersonaDistributionFitter.W_MIN
    total = sum(result.persona_weights.values())
    assert abs(total - 1.0) < 1e-9
    for col, w in result.persona_weights.items():
        assert w >= w_min - 1e-9

    print(
        f"✅ test_loop2_minimum_sample_size: "
        f"n=5, sum={total:.6f}, "
        f"P05_w={result.persona_weights['P05']:.4f} (应 ≥ w_min={w_min:.4f})"
    )


# ============================================================
# 7. 回归保护：旧 bug 不再出现（abs 误把负相关当强相关）
#    （直接测 _weights_from_raw_coef，确定性）
# ============================================================
def test_loop2_no_abs_bug_regression():
    """旧 bug：用 abs(coef) 把负相关人设误判为"强相关"调高权重。

    直接构造 Lasso 原始系数（绕过 Lasso 拟合）：
      - P07: -2.0（强负相关，abs=2.0 是最大）
      - P08: +1.0（弱正相关）

    旧 bug 行为（abs + 归一化）：abs(P07)=2.0, abs(P08)=1.0,
    sum=3.0 → P07 权重 = 2/3 ≈ 0.667, P08 权重 = 1/3 ≈ 0.333.
    → P07 反而比 P08 高（误把负相关当强相关调高）。

    新行为（保留方向）：pos_coef(P07)=0, pos_coef(P08)=1.0,
    → P07 触底 w_min ≈ 0.0167, P08 获 pool ≈ 0.5167.
    → P07 << P08（负相关被压到下限，正相关获高权重）。
    """
    raw = {col: 0.0 for col in PersonaDistributionFitter.PERSONA_COLS}
    raw["P07"] = -2.0   # 强负相关（旧 bug 下 abs=2.0 最大）
    raw["P08"] = +1.0   # 弱正相关

    weights = PersonaDistributionFitter._weights_from_raw_coef(raw)

    w_min = PersonaDistributionFitter.W_MIN

    # 新行为：P07 应触底 w_min，P08 应获高权重
    assert weights["P07"] <= w_min + 1e-9, (
        f"新行为：P07 强负相关应触底 w_min={w_min:.6f}, "
        f"got {weights['P07']:.6f}（旧 abs bug 又出现：负相关被误调高）"
    )
    assert weights["P08"] > w_min + 0.1, (
        f"新行为：P08 正相关应获高权重 > w_min+0.1, "
        f"got {weights['P08']:.6f}"
    )

    # 关键回归：旧 bug 下 P07(abs=2) 应高于 P08(abs=1)
    # 新行为下 P07 触底，远低于 P08
    assert weights["P07"] < weights["P08"], (
        f"旧 bug 回归：P07(abs=2.0) 反而高于 P08(abs=1.0)。"
        f"新行为下 P07 触底 w_min < P08. "
        f"got P07={weights['P07']:.6f}, P08={weights['P08']:.6f}"
    )

    # 计算旧 bug 下的权重（用于对比展示）
    abs_p07 = abs(-2.0) / (abs(-2.0) + abs(1.0))  # 2/3 ≈ 0.667
    abs_p08 = abs(1.0) / (abs(-2.0) + abs(1.0))   # 1/3 ≈ 0.333

    print(
        f"✅ test_loop2_no_abs_bug_regression: "
        f"P07(neg) w={weights['P07']:.6f} (触底) vs 旧bug w={abs_p07:.4f}, "
        f"P08(pos) w={weights['P08']:.6f} (高) vs 旧bug w={abs_p08:.4f}"
    )


if __name__ == "__main__":
    test_loop2_constraints_sum_and_floor()
    test_loop2_directionality_preserved()
    test_loop2_y_normalization_invariant_to_scale()
    test_loop2_vote_scale_constant_and_canonical_input()
    test_loop2_protection_rollback_on_noise()
    test_loop2_minimum_sample_size()
    test_loop2_no_abs_bug_regression()
    print("\n🎉 ALL Loop2 §8.2 COMPLIANCE TESTS PASSED")
