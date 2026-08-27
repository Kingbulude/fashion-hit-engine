"""评审层：引擎二（双渠道评分） + 引擎三（价格价值模型）

两个都是"纯规则引擎"，完全基于 10个BARS特征 + 人设投票结果 计算。
不依赖LLM，所以可反复调整参数（在 scoring_weights.yaml 里）。
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from .config import AppConfig, load_brand_profile
from .types import (
    BrandConfig,
    ChannelScores,
    StyleFeatures,
    StyleInfo,
    VotingResult,
    clamp,
)

log = logging.getLogger(__name__)


def _resolve_scoring_cfg(
    brand_cfg: BrandConfig | None,
    cfg: AppConfig | None,
) -> dict[str, Any]:
    """优先 brand_cfg.scoring_weights；否则旧 AppConfig.scoring；否则默认品牌。"""
    if brand_cfg is not None and brand_cfg.scoring_weights:
        return brand_cfg.scoring_weights
    if cfg is not None and cfg.scoring:
        return cfg.scoring
    fallback = load_brand_profile("tongzhuang-outdoor")
    return fallback.scoring_weights


def _resolve_category_registry(
    brand_cfg: BrandConfig | None,
) -> dict[str, Any]:
    """优先从 brand_cfg.category_registry；否则默认品牌。"""
    if brand_cfg is not None and brand_cfg.category_registry:
        return brand_cfg.category_registry
    fallback = load_brand_profile("tongzhuang-outdoor")
    return fallback.category_registry


def _get_channel_formula_w(score_cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容 channel_formula key 与旧 scoring_formulas key。"""
    cf = score_cfg.get("channel_formula")
    if isinstance(cf, dict) and cf:
        return cf
    return score_cfg.get("scoring_formulas", {})


def _get_price_value_cfg(score_cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容 price_value_model key 与旧 price_value key。"""
    pv = score_cfg.get("price_value_model")
    if isinstance(pv, dict) and pv:
        return pv
    return score_cfg.get("price_value", {})


# ========== 品类价格百分位（新：从 category_registry 取 price_band）==========
def _match_category_band_from_registry(
    category_id: str | None,
    raw_category_name: str | None,
    category_registry: dict[str, Any],
    fallback_band: tuple[float, float] = (79.0, 519.0),
) -> tuple[float, float] | None:
    """从 category_registry.categories[].price_band 匹配价格带。

    匹配优先级：category_id 精确匹配 → name 精确匹配 → name 模糊包含
    """
    categories: list[dict[str, Any]] = category_registry.get("categories", []) or []

    if category_id:
        for cat in categories:
            if str(cat.get("id", "")).lower() == str(category_id).lower():
                pb = cat.get("price_band")
                if pb and len(pb) >= 2:
                    return float(pb[0]), float(pb[1])

    raw_name = (raw_category_name or "").strip()
    if raw_name:
        for cat in categories:
            if str(cat.get("name", "")).strip() == raw_name:
                pb = cat.get("price_band")
                if pb and len(pb) >= 2:
                    return float(pb[0]), float(pb[1])
        for cat in categories:
            cn = str(cat.get("name", "")).strip()
            if cn and (cn in raw_name or raw_name in cn):
                pb = cat.get("price_band")
                if pb and len(pb) >= 2:
                    return float(pb[0]), float(pb[1])

    return fallback_band


def _calc_price_percentile(
    price: float,
    band: tuple[float, float] | None,
) -> float:
    """价格百分位（min-max线性插值 clamp 0-1）"""
    if price <= 0:
        return 0.5
    if band is None:
        return 0.5
    lo, hi = band
    if hi <= lo:
        return 0.5
    pct = (price - lo) / (hi - lo)
    return max(0.0, min(1.0, pct))


# ========== 兼容旧接口：从 scoring_weights 的 category_price_bands 取 ==========
def _match_category_band_old(
    info: StyleInfo,
    pv_cfg: dict[str, Any],
) -> tuple[float, float] | None:
    bands: dict[str, list] = pv_cfg.get("category_price_bands", {})
    cat = (info.category or "").strip()
    if cat in bands:
        b = bands[cat]
        return float(b[0]), float(b[1])
    for key, b in bands.items():
        if key in cat or cat in key:
            return float(b[0]), float(b[1])
    fb = pv_cfg.get("fallback_band", [50, 500])
    return float(fb[0]), float(fb[1])


def _calc_price_percentile_old(info: StyleInfo, pv_cfg: dict[str, Any]) -> float:
    if info.price <= 0:
        return 0.5
    band = _match_category_band_old(info, pv_cfg)
    if band is None:
        return 0.5
    lo, hi = band
    if hi <= lo:
        return 0.5
    pct = (info.price - lo) / (hi - lo)
    return max(0.0, min(1.0, pct))


# ========== 引擎二：双渠道评分 ==========
def calc_natural_score(
    feats: StyleFeatures,
    formula_w: dict[str, float],
    bonus_cfg: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """自然流量分"""
    wearability = feats.features["F06_wearability"].score
    pairing = feats.features["F07_pairing"].score
    color_safety = 10.0 - feats.features["F03_color_risk"].score
    photogenic = feats.features["F05_photogenic"].score

    w_wear = formula_w.get("F06_wearability", formula_w.get("F06", 0.40))
    w_pair = formula_w.get("F07_pairing", formula_w.get("F07", 0.30))
    w_color = formula_w.get("F03_color_risk_inv", formula_w.get("F03_color_risk", 0.15))
    w_photo_natural = formula_w.get("F05_photogenic", 0.15)

    bonus_clean = (feats.features["F02_clean_look"].score - 5.0) * bonus_cfg.get("bonus_clean_sensitivity", 0.3)
    bonus_brand = (feats.features["F09_brand_tone"].score - 5.0) * bonus_cfg.get("bonus_brand_sensitivity", 0.25)

    raw = (
        wearability * w_wear
        + pairing * w_pair
        + color_safety * w_color
        + photogenic * w_photo_natural
    )
    raw += bonus_clean + bonus_brand
    score = clamp(raw, 1.0, 10.0)
    breakdown = {
        "wearability": wearability,
        "pairing": pairing,
        "color_safety": color_safety,
        "photogenic_for_natural": photogenic,
        "bonus_clean": bonus_clean,
        "bonus_brand": bonus_brand,
    }
    return score, breakdown


def calc_live_score(
    feats: StyleFeatures,
    formula_w: dict[str, float],
    live_cfg: dict[str, Any],
    voting: VotingResult,
) -> tuple[float, dict[str, float]]:
    """直播带货分（保留色彩双向调节逻辑）"""
    photogenic = feats.features["F05_photogenic"].score
    func_vis = feats.features["F04_function_visibility"].score
    uniqueness = feats.features["F10_uniqueness"].score
    silhouette = feats.features["F01_silhouette"].score
    color_risk = feats.features["F03_color_risk"].score

    w_ph = formula_w.get("F05_photogenic", formula_w.get("F05", 0.35))
    w_fv = formula_w.get("F04_function_visibility", formula_w.get("F04", 0.35))
    w_un = formula_w.get("F10_uniqueness", formula_w.get("F10", 0.30))
    w_si = formula_w.get("F01_silhouette", formula_w.get("F01", 0.0))

    raw = (
        photogenic * w_ph
        + func_vis * w_fv
        + uniqueness * w_un
        + silhouette * w_si
    )

    # 色彩吸睛奖励（双向调节保留）
    color_appeal_sensitivity = live_cfg.get("color_appeal_sensitivity")
    if color_appeal_sensitivity is None:
        sw = load_brand_profile("tongzhuang-outdoor").scoring_weights
        formula_wrapper = _get_channel_formula_w(sw)
        color_appeal_sensitivity = formula_wrapper.get("live_channel", {}).get("color_appeal_sensitivity", 0.5)
    color_appeal_bonus = max(0.0, (color_risk - 5.0)) * float(color_appeal_sensitivity)
    raw += color_appeal_bonus

    interact = (func_vis / 10.0) * (uniqueness / 10.0) * live_cfg.get("interact_factor", 2.0)
    raw += interact

    if silhouette >= 8.0 and photogenic >= 8.0:
        raw += live_cfg.get("bonus_fashion_show", 0.3)
    if func_vis < 4.0 and uniqueness < 4.0:
        raw -= live_cfg.get("penalty_boring", 0.8)
    opp_pen = voting.opposition_rate * live_cfg.get("opposition_penalty_factor", 3.0)
    raw -= opp_pen

    score = clamp(raw, 1.0, 10.0)
    breakdown = {
        "photogenic": photogenic,
        "func_visibility": func_vis,
        "uniqueness": uniqueness,
        "color_risk_raw": color_risk,
        "color_appeal_bonus": color_appeal_bonus,
        "silhouette": silhouette,
        "interact_term": interact,
        "opposition_penalty": opp_pen,
    }
    return score, breakdown


# ========== 新：独立 calculate_price_value_score ==========
def calculate_price_value_score(
    price: float,
    category_id: str | None,
    features_dict: dict[str, Any],
    *,
    brand_cfg: BrandConfig | None = None,
) -> dict[str, Any]:
    """独立的价格价值评分函数（v2.0 新接口）

    Args:
        price: 款式价格
        category_id: 品类ID（对应 category_registry.categories[].id）
        features_dict: 10特征字典，格式 {key: score} 或 {key: {score}}
        brand_cfg: 品牌配置（可选，None 时默认 tongzhuang-outdoor）

    Returns:
        dict 含:
          - perceived_value: 感知价值（1-10）
          - price_percentile: 价格百分位（0-1）
          - value_match: VM = 感知价值/10 - 百分位
          - risk_level: {"level": "低风险"/"中风险"/"高风险", "low_threshold":..., "high_threshold":...}
    """
    if brand_cfg is None:
        brand_cfg = load_brand_profile("tongzhuang-outdoor")

    score_cfg = brand_cfg.scoring_weights
    pv_cfg = _get_price_value_cfg(score_cfg)
    category_registry = _resolve_category_registry(brand_cfg)

    def _get_score(key: str) -> float:
        v = features_dict.get(key)
        if isinstance(v, dict):
            return float(v.get("score", 5.0))
        return clamp(float(v or 5.0), 1.0, 10.0)

    # 感知价值：0.30*F08面料 + 0.30*F04功能 + 0.25*F10独特 + 0.15*品牌(硬编码0.85系数) + 交互项
    f07_pairing = _get_score("F07_pairing")
    f08_fabric = _get_score("F08_fabric_perception")
    f04_func = _get_score("F04_function_visibility")
    f10_unique = _get_score("F10_uniqueness")
    f09_brand = _get_score("F09_brand_tone")

    # 感知价值按原公式（0.30F07/08相关 + 0.30F04 + 0.25F10 + 0.15品牌 + 交互项）
    pv_w = pv_cfg.get("perceived_value_weights", {})
    if pv_w:
        pv_raw = 0.0
        for feat_key, weight in pv_w.items():
            if feat_key == "F04_F08_interaction":
                continue
            if feat_key in ("F07_pairing", "F07"):
                pv_raw += f07_pairing * float(weight)
            elif feat_key in ("F08_fabric_perception", "F08"):
                pv_raw += f08_fabric * float(weight)
            elif feat_key in ("F04_function_visibility", "F04"):
                pv_raw += f04_func * float(weight)
            elif feat_key in ("F10_uniqueness", "F10"):
                pv_raw += f10_unique * float(weight)
            elif feat_key in ("F09_brand_tone", "F09"):
                pv_raw += f09_brand * float(weight)
            else:
                pv_raw += _get_score(feat_key) * float(weight)
        if "F04_F08_interaction" in pv_w:
            pv_raw += float(pv_w["F04_F08_interaction"]) * (f04_func * f08_fabric / 10.0)
    else:
        # 硬编码默认公式：0.30F08面料 + 0.30F04功能 + 0.25F10独特 + 0.15*F09品牌（品牌×0.85系数）+ 交互项
        pv_raw = (
            0.30 * f08_fabric
            + 0.30 * f04_func
            + 0.25 * f10_unique
            + 0.15 * (f09_brand * 0.85)
            + 1.0 * (f04_func * f08_fabric / 10.0)
        )

    # 隐式补充：面料×品牌的联合感知
    pv_raw += (f08_fabric / 10.0) * (f09_brand / 10.0) * 1.0
    perceived_value = clamp(pv_raw, 1.0, 10.0)

    # 价格百分位：从 category_registry 找 price_band
    band = _match_category_band_from_registry(
        category_id, None, category_registry,
        fallback_band=tuple(pv_cfg.get("fallback_band", [79.0, 519.0])),
    )
    price_percentile = _calc_price_percentile(price, band)

    # VM = PV/10 - 百分位
    value_match = (perceived_value / 10.0) - price_percentile
    value_match = clamp(value_match, -1.0, 1.0)

    # 风险等级
    thr = pv_cfg.get("thresholds", {"low_risk": 0.15, "high_risk": -0.15})
    low_risk_v = float(thr.get("low_risk", 0.15))
    high_risk_v = float(thr.get("high_risk", -0.15))
    if value_match <= high_risk_v:
        risk_level_name = "高风险"
    elif value_match >= low_risk_v:
        risk_level_name = "低风险"
    else:
        risk_level_name = "中风险"

    return {
        "perceived_value": perceived_value,
        "price_percentile": price_percentile,
        "value_match": value_match,
        "risk_level": {
            "level": risk_level_name,
            "low_threshold": low_risk_v,
            "high_threshold": high_risk_v,
        },
    }


# ========== v2.0 新主函数：calculate_channel_scores ==========
def calculate_channel_scores(
    info: StyleInfo,
    feats: StyleFeatures,
    voting: VotingResult,
    cfg: AppConfig | None = None,
    all_style_prices: list[float] | None = None,
    *,
    brand_cfg: BrandConfig | None = None,
    category_id: str | None = None,
) -> tuple[ChannelScores, dict[str, Any]]:
    """双渠道评分 + 价格价值评分（v2.0 BrandConfig 注入版）

    完全向后兼容：
    - 只传旧 cfg: AppConfig → 走旧 evaluate_channels 路径
    - 传 brand_cfg → 从 brand_cfg 取 channel_formula / price_value_model / category_registry
    - 都不传 → 默认 load_brand_profile('tongzhuang-outdoor')
    """
    score_cfg = _resolve_scoring_cfg(brand_cfg, cfg)
    channel_formulas = _get_channel_formula_w(score_cfg)
    pv_cfg = _get_price_value_cfg(score_cfg)
    category_registry = _resolve_category_registry(brand_cfg)

    nat_cfg = channel_formulas.get("natural_channel", {})
    live_cfg = channel_formulas.get("live_channel", {})

    channel_bonus = score_cfg.get("calibration", {}).get("channel_extra", {}) or {}
    nat_bonus_cfg = {
        "bonus_clean_sensitivity": channel_bonus.get("bonus_clean_sensitivity", 0.3),
        "bonus_brand_sensitivity": channel_bonus.get("bonus_brand_sensitivity", 0.25),
    }
    live_bonus_cfg = {
        "interact_factor": channel_bonus.get("live_interact_factor", 2.0),
        "bonus_fashion_show": channel_bonus.get("live_bonus_fashion_show", 0.3),
        "penalty_boring": channel_bonus.get("live_penalty_boring", 0.8),
        "opposition_penalty_factor": channel_bonus.get("live_opposition_penalty", 3.0),
        "color_appeal_sensitivity": (
            live_cfg.get("color_appeal_sensitivity")
            or channel_bonus.get("live_color_appeal_sensitivity", 0.5)
        ),
    }

    natural_score, nat_breakdown = calc_natural_score(
        feats, nat_cfg.get("formula", {}), nat_bonus_cfg,
    )
    live_score, live_breakdown = calc_live_score(
        feats, live_cfg.get("formula", {}), live_bonus_cfg, voting,
    )

    # 价格百分位 + 感知价值：优先用 category_registry（新路径）
    if brand_cfg is not None or category_id is not None:
        feats_dict = {k: f.score for k, f in feats.features.items()}
        pv_result = calculate_price_value_score(
            info.price, category_id, feats_dict, brand_cfg=brand_cfg,
        )
        perceived_value = pv_result["perceived_value"]
        pct = pv_result["price_percentile"]
        value_match = pv_result["value_match"]
        price_risk = pv_result["risk_level"]["level"]
    else:
        # 旧路径：从 scoring_weights.category_price_bands 取
        pct = _calc_price_percentile_old(info, pv_cfg)
        pv_w = pv_cfg.get("perceived_value_weights", {})
        pv_raw = 0.0
        for feat_key, weight in pv_w.items():
            if feat_key == "F04_F08_interaction":
                continue
            if feat_key in feats.features:
                pv_raw += feats.features[feat_key].score * float(weight)
        if "F04_F08_interaction" in pv_w and "F04_function_visibility" in feats.features and "F08_fabric_perception" in feats.features:
            f4 = feats.features["F04_function_visibility"].score
            f8 = feats.features["F08_fabric_perception"].score
            pv_raw += float(pv_w["F04_F08_interaction"]) * (f4 * f8 / 10.0)
        if "F08_fabric_perception" in feats.features and "F09_brand_tone" in feats.features:
            f8 = feats.features["F08_fabric_perception"].score
            f9 = feats.features["F09_brand_tone"].score
            pv_raw += (f8 / 10.0) * (f9 / 10.0) * 1.0
        perceived_value = clamp(pv_raw, 1.0, 10.0)
        value_match = (perceived_value / 10.0) - pct
        value_match = clamp(value_match, -1.0, 1.0)
        thr = pv_cfg.get("thresholds", {"low_risk": 0.15, "high_risk": -0.15})
        low_risk_v = float(thr.get("low_risk", 0.15))
        high_risk_v = float(thr.get("high_risk", -0.15))
        if value_match <= high_risk_v:
            price_risk = "高风险"
        elif value_match >= low_risk_v:
            price_risk = "低风险"
        else:
            price_risk = "中风险"

    channels = ChannelScores(
        style_id=info.style_id,
        natural_score=natural_score,
        live_score=live_score,
        perceived_value=perceived_value,
        price_percentile=pct,
        value_match=value_match,
        price_risk=price_risk,
    )
    debug = {
        "natural_breakdown": nat_breakdown,
        "live_breakdown": live_breakdown,
        "pv_raw": perceived_value,
    }
    return channels, debug
