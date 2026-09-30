"""三大/四大引擎合成器

v1.4.49+: 支持可选的第 4 引擎 grade_norm（内审分级归一化 [0,100]）。
  - 校准阶段 optimization_kernel 会自动从 StyleInfo.manual_grade 计算 grade_norm
  - Loop3 产物里可能带 4 个引擎权重（persona/channel/price_value/grade_norm）
  - 预测阶段如果 StyleInfo.manual_grade 非空 → 计算 grade_norm_score 并传入
  - 如果 manual_grade 为空（无内审流程） → 不传 grade_norm，保持 3 引擎

权重来源优先级：
1) Loop3 校准产物 engine_weights.yaml（通过 BrandConfig.engine_weights 注入）
2) BrandConfig.default_engine_weights / default_channel_split
3) 硬编码默认（persona=0.35, channel=0.30, price_value=0.35）
"""
from __future__ import annotations

from typing import Any


# grade 标签 → 归一化分数（0-100，与 optimization_kernel.build_history_df 一致）
_GRADE_TO_NORM: dict[str, float] = {
    "S": 100.0,
    "A+": 75.0,
    "A": 50.0,
    "P": 0.0,
}


def grade_norm_from_manual_grade(manual_grade: str) -> float | None:
    """把 StyleInfo.manual_grade (S/A+/A/P) 转成 [0,100] 的 grade_norm 分数。

    Returns:
        float [0,100] 如果 manual_grade 是已知标签
        None 如果 manual_grade 为空或未知（表示无内审）
    """
    if not manual_grade:
        return None
    return _GRADE_TO_NORM.get(manual_grade)


def synthesise_final_score(
    persona_score: float,
    channel_scores: dict[str, float],
    price_value_score: float,
    *,
    engine_weights: dict[str, float],
    channel_split: dict[str, float],
    grade_norm_score: float | None = None,
) -> tuple[float, dict[str, float]]:
    """合成最终爆款分

    Args:
        persona_score: 人设投票聚合分（0-10）
        channel_scores: 双渠道分，必须含 "natural" 和 "live_stream" 键
        price_value_score: 价格价值分（通常取 value_match 归一化到 0-10 或 perceived_value）
        engine_weights: 引擎权重字典 — 可能含 3 或 4 个 key:
            persona_voting / channel_scoring / price_value / (grade_norm)
        channel_split: 渠道内部加权 {"natural": w, "live_stream": w}
        grade_norm_score: 可选的第 4 引擎分（内审分级归一化 [0,100]）。
            如果传入且 engine_weights 里有 grade_norm 权重 → 启用 4 引擎合成。
            不传或为 None → 回退到原来的 3 引擎。

    Returns:
        (final_score, breakdown)
        final_score: 0-10 的合成总分
        breakdown: 各引擎贡献分字典
    """
    natural_sc = float(channel_scores.get("natural", 0.0))
    live_sc = float(channel_scores.get("live_stream", 0.0))

    split_natural = float(channel_split.get("natural", 0.50))
    split_live = float(channel_split.get("live_stream", 0.50))
    split_sum = split_natural + split_live
    if split_sum <= 0:
        split_natural, split_live = 0.50, 0.50
        split_sum = 1.0
    channel_final = (split_natural * natural_sc + split_live * live_sc) / split_sum

    # —— 收集有效引擎分 + 权重 ——
    engines: list[tuple[float, float, str]] = []  # (score, weight, name)

    w_persona = float(engine_weights.get("persona_voting", 0.35))
    w_channel = float(engine_weights.get("channel_scoring", 0.30))
    w_price = float(engine_weights.get("price_value", 0.35))

    # grade_norm: 仅当权重里有且预测时也提供了分数才启用
    w_grade_norm = float(engine_weights.get("grade_norm", 0.0))
    grade_norm_enabled = (w_grade_norm > 0.001) and (grade_norm_score is not None)

    engines.append((float(persona_score), w_persona, "persona"))
    engines.append((channel_final, w_channel, "channel"))
    engines.append((float(price_value_score), w_price, "price_value"))

    if grade_norm_enabled:
        # grade_norm_score 是 [0,100] 尺度 → 映射到 [0,10] 与其他引擎同尺度
        gn_0_10 = float(grade_norm_score) / 10.0
        engines.append((gn_0_10, w_grade_norm, "grade_norm"))

    w_sum = sum(w for _, w, _ in engines)
    if w_sum <= 0:
        # fallback: 恢复默认 3 引擎
        engines = [
            (float(persona_score), 0.35, "persona"),
            (channel_final, 0.30, "channel"),
            (float(price_value_score), 0.35, "price_value"),
        ]
        w_sum = 1.0

    final = sum(s * w for s, w, _ in engines) / w_sum

    breakdown: dict[str, float] = {
        "persona": float(persona_score) * w_persona / w_sum,
        "channel": channel_final * w_channel / w_sum,
        "price_value": float(price_value_score) * w_price / w_sum,
        "natural_channel": natural_sc * split_natural / split_sum,
        "live_channel": live_sc * split_live / split_sum,
    }
    if grade_norm_enabled:
        breakdown["grade_norm"] = (float(grade_norm_score) / 10.0) * w_grade_norm / w_sum

    return final, breakdown
