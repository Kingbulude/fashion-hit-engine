"""盲测组（blind set）：防止系统「自己证明自己准」的对照机制。

领域定义（见 CONTEXT.md「盲测组」）：
- 每批次按品类分层随机抽取 10-20% 款式标记为盲测款
- 盲测款照常跑 AI 预测并完整落盘（系统内部可见），
  但批次报告/导出对运营打码（不给看 AI 结论 → 内审决策不受 AI 干扰）
- 销量回填后解锁对照分析：
  1. 盲测组 Spearman = AI 预测的「纯净」准确率（未被投放污染）
  2. 对照组（运营看过 AI 结论的款）Spearman 含投放放大效应
  3. AI vs 人工分级分歧款，按实际销量裁决谁对 → 验证系统真实价值
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .types import StyleInfo

DEFAULT_BLIND_RATIO = 0.15


# ========== 抽样 ==========
def select_blind_set(
    styles: list[StyleInfo],
    *,
    ratio: float = DEFAULT_BLIND_RATIO,
    seed: str = "batch",
) -> set[str]:
    """按品类分层随机抽取盲测款号（确定性：同 seed 同结果）。

    分层保证盲测组品类结构与整批一致，避免「盲测款恰好全是外套」偏差。
    每层至少保 1 款（层内款数≥1 且 ratio>0 时），不足按比例向下取整。
    """
    if not styles or ratio <= 0:
        return set()
    ratio = min(ratio, 0.5)
    rng = random.Random(seed)

    # 按品类分层
    strata: dict[str, list[str]] = {}
    for s in styles:
        strata.setdefault(s.category or "未分类", []).append(s.style_id)

    blind_ids: set[str] = set()
    for cat, ids in strata.items():
        n = round(len(ids) * ratio)
        # 层内期望值≥0.5款时至少保1款，避免小品类永远进不了盲测
        if n == 0 and len(ids) * ratio >= 0.5:
            n = 1
        n = min(n, len(ids))
        if n > 0:
            blind_ids.update(rng.sample(ids, n))
    return blind_ids


# ========== 对照分析 ==========
@dataclass
class BlindComparisonResult:
    """盲测组 vs 对照组对照分析结果"""
    # 样本量
    n_blind: int = 0
    n_control: int = 0
    # 纯净准确率（盲测组 Spearman：AI 分 vs 销量）
    blind_spearman: float | None = None
    # 对照组 Spearman（运营看过 AI 结论 → 含投放放大）
    control_spearman: float | None = None
    # AI vs 人工分歧裁决表（仅盲测款：人工没被 AI 带偏，分歧是真实分歧）
    disputes: list[dict[str, Any]] = field(default_factory=list)
    # 裁决汇总
    ai_wins: int = 0       # AI 对（AI 高分人工低分 → 实际卖得好；或 AI 低分人工高分 → 实际卖得差）
    human_wins: int = 0    # 人工对
    tie: int = 0           # 销量无法裁决（并列/销量缺失）
    # 数据是否足够出结论
    ready: bool = False
    note: str = ""


_GRADE_ORDER = {"S": 4, "A+": 3, "A": 2, "P": 1, "风险": 0}


def _spearman(x: list[float], y: list[float]) -> float | None:
    """小样本 Spearman（无 scipy 依赖；并列取平均秩）"""
    if len(x) != len(y) or len(x) < 3:
        return None

    def _ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based 平均秩
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def compare_blind_vs_control(
    history_df: pd.DataFrame,
    *,
    min_blind: int = 5,
) -> BlindComparisonResult:
    """对照分析：盲测组（纯净准确率）vs 对照组（含投放效应）。

    Args:
        history_df: 历史库 DataFrame，需含列
            is_blind / final_score / sales_qty / manual_grade / final_grade / style_id
        min_blind: 盲测款最小样本量（低于此不出 Spearman 结论）
    """
    res = BlindComparisonResult()
    if history_df.empty:
        res.note = "历史库为空"
        return res

    df = history_df.copy()
    df["is_blind"] = df["is_blind"].fillna(0).astype(int)
    # 只看有销量的款（销量回填后才算「解锁」）
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0)

    blind = df[(df["is_blind"] == 1) & (df["sales_qty"] > 0)]
    control = df[(df["is_blind"] == 0) & (df["sales_qty"] > 0)]

    res.n_blind = len(blind)
    res.n_control = len(control)

    if len(blind) >= min_blind:
        res.blind_spearman = _spearman(
            [float(x) for x in blind["final_score"]],
            [float(x) for x in blind["sales_qty"]],
        )
    else:
        res.note = f"盲测款有销量的仅 {len(blind)} 个（需≥{min_blind}），待销量回填后解锁"

    if len(control) >= 3:
        res.control_spearman = _spearman(
            [float(x) for x in control["final_score"]],
            [float(x) for x in control["sales_qty"]],
        )

    # ========== AI vs 人工分歧裁决（仅盲测款）==========
    sales_rank = blind["sales_qty"].rank(ascending=False, method="min").astype(int)
    median_sales = float(blind["sales_qty"].median()) if len(blind) else 0.0
    for (_, row), rank in zip(blind.iterrows(), sales_rank):
        ai_grade = str(row.get("final_grade", "") or "")
        human_grade = str(row.get("manual_grade", "") or "")
        if not human_grade or not ai_grade:
            continue
        if ai_grade == human_grade:
            continue
        ai_v = _GRADE_ORDER.get(ai_grade, -1)
        hu_v = _GRADE_ORDER.get(human_grade, -1)
        if ai_v < 0 or hu_v < 0:
            continue
        sales = float(row["sales_qty"])
        # 销量裁决：进入批次前 1/3 视为「卖得好」，后 1/3 视为「卖得差」
        third = max(1, len(blind) // 3)
        n = len(blind)
        if rank <= third:
            verdict = "卖得好"
        elif rank > n - third:
            verdict = "卖得差"
        else:
            res.tie += 1
            res.disputes.append({
                "style_id": row.get("style_id", "?"), "ai_grade": ai_grade,
                "human_grade": human_grade, "sales_qty": sales,
                "verdict": "中段（无法裁决）", "winner": "—",
            })
            continue
        # 分歧方向：AI 更乐观 or 人工更乐观
        ai_higher = ai_v > hu_v
        if verdict == "卖得好":
            winner = "AI" if ai_higher else "人工"
        else:  # 卖得差 → 谁低谁对
            winner = "人工" if ai_higher else "AI"
        if winner == "AI":
            res.ai_wins += 1
        else:
            res.human_wins += 1
        res.disputes.append({
            "style_id": row.get("style_id", "?"), "ai_grade": ai_grade,
            "human_grade": human_grade, "sales_qty": sales,
            "verdict": verdict, "winner": winner,
        })

    res.ready = res.blind_spearman is not None
    if not res.note:
        res.note = "对照已解锁" if res.ready else "盲测样本不足"
    return res
