"""三大引擎合成器

把人设投票分 / 双渠道分 / 价格价值分 合成最终爆款分。
渠道内部再先做 自然流量 vs 直播带货 的加权合成。

权重来源优先级：
1) Loop3 校准产物 engine_weights.yaml（通过 BrandConfig.engine_weights 注入）
2) BrandConfig.default_engine_weights / default_channel_split
"""
from __future__ import annotations

from typing import Any


def synthesise_final_score(
    persona_score: float,
    channel_scores: dict[str, float],
    price_value_score: float,
    *,
    engine_weights: dict[str, float],
    channel_split: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """合成最终爆款分

    Args:
        persona_score: 人设投票聚合分（0-10）
        channel_scores: 双渠道分，必须含 "natural" 和 "live_stream" 键
        price_value_score: 价格价值分（通常取 value_match 归一化到 0-10 或 perceived_value）
        engine_weights: 三大引擎权重 {"persona_voting": w, "channel_scoring": w, "price_value": w}
        channel_split: 渠道内部加权 {"natural": w, "live_stream": w}

    Returns:
        (final_score, breakdown)
        final_score: 0-10 的合成总分
        breakdown: {"persona": v, "channel": v, "price_value": v,
                    "natural_channel": v, "live_channel": v}
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

    w_persona = float(engine_weights.get("persona_voting", 0.35))
    w_channel = float(engine_weights.get("channel_scoring", 0.30))
    w_price = float(engine_weights.get("price_value", 0.35))
    w_sum = w_persona + w_channel + w_price
    if w_sum <= 0:
        w_persona, w_channel, w_price = 0.35, 0.30, 0.35
        w_sum = 1.0

    final = (
        w_persona * float(persona_score)
        + w_channel * channel_final
        + w_price * float(price_value_score)
    ) / w_sum

    breakdown: dict[str, float] = {
        "persona": float(persona_score) * w_persona / w_sum,
        "channel": channel_final * w_channel / w_sum,
        "price_value": float(price_value_score) * w_price / w_sum,
        "natural_channel": natural_sc * split_natural / split_sum,
        "live_channel": live_sc * split_live / split_sum,
    }

    return final, breakdown
