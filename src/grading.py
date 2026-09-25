"""校准层 + 分级 + 报告生成

校准层策略：
- 阶段一（冷启动，历史数据<10款）：使用 spec.md 里的默认权重做加权合成，不跑回归
- 阶段二（≥10款历史数据）：使用 sklearn Lasso 稀疏回归拟合 → 把权重dump成YAML，覆盖默认权重

营销控制变量：是否主推、是否直播重点，允许在最终分±5%范围内调整
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig, load_brand_profile
from .types import (
    BrandConfig,
    ChannelScores,
    FullPrediction,
    GradeResult,
    StyleFeatures,
    StyleInfo,
    VotingResult,
    clamp,
)

log = logging.getLogger(__name__)


def _resolve_grade_thresholds_100(
    brand_cfg: BrandConfig | None,
    scoring_cfg: dict[str, Any],
) -> dict[str, float]:
    """解析最终分（0-100）阈值：优先从 brand_cfg.grading_thresholds 映射（×10），否则默认。

    BrandConfig.grading_thresholds 是 0-10 分制：{s:7.6, a_plus:6.6, a:5.2, p:0.0}
    转换为 0-100 分制：×10。
    """
    if brand_cfg is not None and brand_cfg.grading_thresholds:
        gt = brand_cfg.grading_thresholds
        return {
            "s": float(gt.get("s", 7.6)) * 10.0,
            "a_plus": float(gt.get("a_plus", 6.6)) * 10.0,
            "a": float(gt.get("a", 5.2)) * 10.0,
            "p": float(gt.get("p", 0.0)) * 10.0,
        }
    return scoring_cfg.get("grade_score_thresholds", _GRADE_SCORE_THRESHOLDS_DEFAULT)


def _resolve_grading_rules(
    brand_cfg: BrandConfig | None,
    scoring_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """解析 S/A+/A/P 分级规则：优先 scoring_weights.yaml.grading_thresholds（旧规则字段保持）。"""
    rules = scoring_cfg.get("grading_thresholds")
    if rules:
        return rules
    if brand_cfg is not None:
        return brand_cfg.scoring_weights.get("grading_thresholds")
    return None


# ========== 特征向量（用于校准层回归）==========
def build_feature_vector(
    info: StyleInfo,
    feats: StyleFeatures,
    voting: VotingResult,
    channels: ChannelScores,
    *,
    include_marketing_vars: bool = True,
) -> dict[str, float]:
    """把1款展开为回归特征向量（约50维）"""
    vec: dict[str, float] = {}
    # 10个基础特征
    for key, f in feats.features.items():
        vec[key] = f.score
    # 人设投票聚合
    vec["vote_weighted"] = voting.weighted_score
    vec["vote_support"] = voting.support_rate
    vec["vote_oppose"] = voting.opposition_rate
    vec["vote_std"] = voting.score_std
    # 双渠道
    vec["natural_score"] = channels.natural_score
    vec["live_score"] = channels.live_score
    # 价格价值
    vec["pv"] = channels.perceived_value
    vec["price_pct"] = channels.price_percentile
    vec["value_match"] = channels.value_match
    # 营销控制变量
    if include_marketing_vars:
        vec["is_main_push"] = 1.0 if info.is_main_push else 0.0
        vec["is_live_stream"] = 1.0 if info.is_live_stream else 0.0
    return vec


# ========== 默认权重合成（冷启动，未训练时用）==========
# 阶段一冷启动默认合成权重（4引擎 + 若干调节项）
_FINAL_SCORE_WEIGHTS_DEFAULT: dict[str, float] = {
    "engine1_persona":  0.40,   # 人设投票加权分
    "engine2_natural":  0.20,   # 自然流量分
    "engine2_live":     0.20,   # 直播带货分
    "engine3_value":    0.20,   # 价格价值匹配度
}

# 双维度分级矩阵阈值（0-100 制）
_DEMAND_POTENTIAL_THRESHOLD = 75.0   # ≥75 → 需求潜力高（大概率能走量）
_BRAND_PORTFOLIO_THRESHOLD = 70.0    # ≥70 → 品牌组合价值高（有品牌表达贡献）


# ========== 双维度分计算 ==========
def _calc_demand_potential(
    feats: StyleFeatures,
    voting: VotingResult,
    channels: ChannelScores,
    scoring_cfg: dict[str, Any],
) -> float:
    """需求潜力分（0-100）：代表这款的走量能力

    为什么和 final_score 独立：final_score 混了"品牌组合价值"（F09/F10 调性/独特性），
    而需求潜力只关心——假设品牌矩阵里这款是最普通的跑量定位，它能卖多少？

    权重：人设加权 + 自然流量 + 直播 + 价值匹配（和 _default_aggregate 基分相同）
    调节项：反对率惩罚 + 渠道 boost（和 final_score 完全一致）
    0-10 制 → 0-100 线性映射。
    """
    w = scoring_cfg.get("final_score_weights", _FINAL_SCORE_WEIGHTS_DEFAULT)

    e1 = clamp(voting.weighted_score, 0.0, 10.0)
    e2n = clamp(channels.natural_score, 0.0, 10.0)
    e2l = clamp(channels.live_score, 0.0, 10.0)
    e3 = (channels.value_match + 1.0) * 5.0

    base_0_10 = (
        e1 * w["engine1_persona"]
        + e2n * w["engine2_natural"]
        + e2l * w["engine2_live"]
        + e3 * w["engine3_value"]
    )

    opp_factor = scoring_cfg.get("opposition_penalty_per_10pct", 0.5)
    oppose_penalty_0_10 = voting.opposition_rate / 0.10 * opp_factor
    base_0_10 -= oppose_penalty_0_10

    best_channel = max(e2n, e2l)
    channel_boost_0_10 = max(0.0, (best_channel - 7.0)) * scoring_cfg.get("channel_boost_sensitivity", 0.3)
    base_0_10 += channel_boost_0_10

    return round(clamp(base_0_10, 0.0, 10.0) * 10.0, 1)


def _calc_brand_portfolio_value(
    feats: StyleFeatures,
    channels: ChannelScores,
    scoring_cfg: dict[str, Any] | None = None,
) -> float:
    """品牌组合价值分（0-100）：代表这款在品牌矩阵里的独特贡献

    为什么和需求潜力独立："品牌表达款"（P 款）可能走量不行但调性极强，
    和"难卖"（风险款）是两回事。品牌价值 = F09 品牌调性贡献 + F10 设计独特性，
    再叠加品类策略信号（价格处于本品牌上沿且 VM 为正 → 品牌溢价确认）。

    权重固定（不被校准覆盖），因为品牌价值判断是相对稳定的品类策略。
    """
    feat_scores = {k: f.score for k, f in feats.features.items()}
    f09 = feat_scores.get("F09_brand_tone", 5.0)
    f10 = feat_scores.get("F10_uniqueness", 5.0)

    base_0_10 = f09 * 0.5 + f10 * 0.5

    # 品类策略信号：价格百分位高 + VM 为正 → 品牌溢价款，加一点
    if channels.price_percentile >= 0.7 and channels.value_match >= 0.1:
        base_0_10 += 0.5

    # 兜底：如果 F09/F10 都在 feat 里找不到 → 安全回退 5.0
    base_0_10 = clamp(base_0_10, 0.0, 10.0)
    return round(base_0_10 * 10.0, 1)


def _grade_from_dual_dimension(
    demand_potential: float,
    brand_value: float,
    channels: ChannelScores,
    voting: VotingResult,
    *,
    demand_thr: float = _DEMAND_POTENTIAL_THRESHOLD,
    brand_thr: float = _BRAND_PORTFOLIO_THRESHOLD,
) -> tuple[str, str]:
    """双维度矩阵 → (grade, grade_note)

    矩阵：
      ┌────────────┬──────────────┬──────────────┐
      │            │ 品牌价值 ≥70  │ 品牌价值 <70  │
      ├────────────┼──────────────┼──────────────┤
      │ 需求 ≥75   │ S 主推款     │ A 跑量款      │
      │ 需求 <75   │ P 品牌表达款 │ 风险款       │
      └────────────┴──────────────┴──────────────┘

    注：需求和品牌价值是两个独立维度，故意不互斥。
    P 款 = 需求潜力低但品牌价值高（设计好看但颜色太花、受众窄），
    和"难卖/风险款"不再混为一档。
    """
    # 风险款绝对判定优先（双渠道极弱或反对率极高）
    risk_nat = channels.natural_score <= 5.0 and channels.live_score <= 5.0
    risk_oppose = voting.opposition_rate > 0.30
    if risk_nat or risk_oppose:
        if demand_potential >= demand_thr and brand_value >= brand_thr:
            # 极端矛盾：矩阵判 S 但双渠道极弱，保留矩阵结论 + 加备注
            return "S", "双渠道均低于5分但双维度综合仍高"
        if demand_potential >= demand_thr and brand_value < brand_thr:
            return "风险", "双渠道极弱或反对率高"
        if demand_potential < demand_thr and brand_value >= brand_thr:
            return "P", "双渠道极弱，品牌价值保留"
        return "风险", "双维度均偏低"

    if demand_potential >= demand_thr and brand_value >= brand_thr:
        return "S", "双维度均高 — 主推款 + 品牌标杆"
    if demand_potential >= demand_thr and brand_value < brand_thr:
        return "A", "需求潜力高但品牌价值低 — 跑量款"
    if demand_potential < demand_thr and brand_value >= brand_thr:
        return "P", "需求潜力低但品牌价值高 — 品牌表达款"
    return "风险", "双维度均偏低 — 谨慎推进"


def _default_aggregate(
    info: StyleInfo,
    feats: StyleFeatures,
    voting: VotingResult,
    channels: ChannelScores,
    scoring_cfg: dict[str, Any],
) -> tuple[float, list[str], list[str]]:
    """
    未训练时的兜底合成：按默认权重线性合成，再叠加反对率惩罚/渠道boost/营销调整。
    输出范围 0-100。

    权重来源：
    - 尝试读 scoring_cfg["final_score_weights"]（用户自定义覆盖）
    - 否则用 _FINAL_SCORE_WEIGHTS_DEFAULT
    """
    w = scoring_cfg.get("final_score_weights", _FINAL_SCORE_WEIGHTS_DEFAULT)

    # 4个引擎分：都归一化到 0-10 区间
    e1 = clamp(voting.weighted_score, 0.0, 10.0)
    e2n = clamp(channels.natural_score, 0.0, 10.0)
    e2l = clamp(channels.live_score, 0.0, 10.0)
    # VM: -1~1 → 0~10 线性映射（-1→0, 0→5, +1→10）
    e3 = (channels.value_match + 1.0) * 5.0

    base_0_10 = (
        e1 * w["engine1_persona"]
        + e2n * w["engine2_natural"]
        + e2l * w["engine2_live"]
        + e3 * w["engine3_value"]
    )

    # —— 调节项 ——
    # 1. 反对率惩罚：反对率每10%扣0.5分（满分10分制）
    opp_factor = scoring_cfg.get("opposition_penalty_per_10pct", 0.5)
    oppose_penalty_0_10 = voting.opposition_rate / 0.10 * opp_factor
    base_0_10 -= oppose_penalty_0_10

    # 2. 渠道boost：较强渠道高于7.0时加分（爆款的"强渠道杠杆"）
    best_channel = max(e2n, e2l)
    channel_boost_0_10 = max(0.0, (best_channel - 7.0)) * scoring_cfg.get("channel_boost_sensitivity", 0.3)
    base_0_10 += channel_boost_0_10

    # 3. 营销控制变量（±5%范围内）—— 先不碰主分，乘系数到最终0-100
    marketing_multiplier = 1.0
    max_adj = scoring_cfg.get("marketing_adj_max_pct", 0.05)
    if info.is_main_push:
        marketing_multiplier += max_adj * scoring_cfg.get("marketing_adj_main_push", 0.8)
    if info.is_live_stream:
        marketing_multiplier += max_adj * scoring_cfg.get("marketing_adj_live_stream", 0.6)

    final_0_10 = clamp(base_0_10, 0.0, 10.0)
    final = clamp(final_0_10 * 10.0 * marketing_multiplier, 0.0, 100.0)

    # —— 优劣势提取 ——
    strengths: list[str] = []
    weaknesses: list[str] = []
    for f in sorted(feats.features.values(), key=lambda x: -x.score)[:3]:
        if f.score >= 6.5:
            strengths.append(f"{f.name}（{f.score:.1f}/10）— {f.reason[:30]}")
    for f in feats.lowest_features(3):
        if f.score <= 5.0:
            weaknesses.append(f"{f.name}偏低（{f.score:.1f}/10）— {f.reason[:30]}")
    if voting.opposition_rate >= 0.3:
        weaknesses.append(f"人设反对率偏高（{voting.opposition_rate:.0%}）")
    if channels.value_match <= -0.15:
        weaknesses.append(f"价格价值不匹配（VM={channels.value_match:+.2f}，{channels.price_risk}）")
    for r in voting.top_buy_reasons[:2]:
        strengths.append(f"人设认可：{r[:40]}")
    for r in voting.top_oppose_reasons[:2]:
        weaknesses.append(f"人设顾虑：{r[:40]}")

    return final, strengths, weaknesses


# ========== S/A+/A/P 分级（默认阈值，与scoring_weights.yaml分级阈值双向参考）==========
_GRADE_SCORE_THRESHOLDS_DEFAULT = {
    "s": 82.0,       # 最终分≥82 → S
    "a_plus": 72.0,  # 72-82 → A+
    "a": 58.0,       # 58-72 → A
}


def assign_grade(
    final_score: float,
    confidence: float,
    scoring_cfg: dict[str, Any] | None = None,
    *,
    channels: "ChannelScores | None" = None,
    voting: "VotingResult | None" = None,
    feats: "StyleFeatures | None" = None,
    brand_cfg: BrandConfig | None = None,
) -> str:
    """
    分级策略：
    1. 先用 final_score 阈值给出基础分级（优先 brand_cfg.grading_thresholds）
    2. 再结合规则做修正（覆盖/降档）

    新接口建议：传 brand_cfg。不传时默认阈值与旧版保持一致（s=82/a+=72/a=58）。
    """
    if scoring_cfg is None:
        if brand_cfg is not None:
            scoring_cfg = brand_cfg.scoring_weights
        else:
            scoring_cfg = load_brand_profile("mipo").scoring_weights

    # —— 基础分档：优先 brand_cfg.grading_thresholds（0-10制×10） ——
    thr = _resolve_grade_thresholds_100(brand_cfg, scoring_cfg)
    base: str
    if final_score >= thr["s"]:
        base = "S"
    elif final_score >= thr["a_plus"]:
        base = "A+"
    elif final_score >= thr["a"]:
        base = "A"
    else:
        base = "P"

    # 置信度约束：S要求高置信度
    grading_cfg = scoring_cfg.get("grading", {})
    min_conf_s = grading_cfg.get("minimum_confidence_for_s", 0.60)
    if base == "S" and confidence < min_conf_s:
        base = "A+"

    # —— 规则修正：读 grading_thresholds 规则 ——
    rules = _resolve_grading_rules(brand_cfg, scoring_cfg)
    if rules and channels and voting and feats:
        s_rule = rules.get("s_grade", {})
        if base == "S":
            if channels.natural_score < float(s_rule.get("natural_score_min", 7.5)):
                pass
            if (channels.natural_score < float(s_rule.get("natural_score_min", 7.5))
                and channels.live_score < float(s_rule.get("live_score_min", 7.0))):
                base = "A+"
            if voting.opposition_rate > float(s_rule.get("opposition_max", 0.15)):
                base = "A+"
            vm_min = float(s_rule.get("value_match_min", 0.0))
            if channels.value_match < vm_min:
                base = "A+"

        if base in {"A", "P"}:
            ap_rule = rules.get("a_plus_grade", {})
            single_min = float(ap_rule.get("single_high_min", 7.5))
            other_min = float(ap_rule.get("other_low_min", 6.0))
            if (channels.natural_score >= single_min and channels.live_score >= other_min
                ) or (channels.live_score >= single_min and channels.natural_score >= other_min):
                default_ap = _GRADE_SCORE_THRESHOLDS_DEFAULT["a_plus"]
                threshold_ap = thr.get("a_plus", default_ap)
                if final_score >= threshold_ap - 5.0:
                    base = "A+"

        risk_rule = rules.get("risk_grade", {})
        if (channels.natural_score <= float(risk_rule.get("natural_max", 5.0))
            and channels.live_score <= float(risk_rule.get("live_max", 5.0))):
            base = "风险"
        if voting and voting.opposition_rate > float(risk_rule.get("opposition_max", 0.30)):
            base = "风险" if base in {"P", "A"} else "A"

    # 注意：P 款识别已移到 assign_relative_grades 末尾（后处理）
    # 相对分级会覆盖绝对档，P 款必须在相对分级之后应用

    return base


# ========== 批次内相对分级（冷启动对齐人工内审语义）==========
def assign_relative_grades(
    preds: list[FullPrediction],
    *,
    s_pct: float = 0.20,
    a_plus_pct: float = 0.20,
    a_pct: float = 0.35,
) -> dict[str, str]:
    """批次内相对分级：S=Top20%，A+=20-40%，A=40-75%，P=后25%。

    为什么需要（见 CONTEXT.md「相对分级」）：
    人工内审的 S/A/P 是批次内**相对排序**（这批里最好的打 S），
    而绝对阈值分级在冷启动时失效——VLM 绝对分尺度未校准
    （常见所有款 6.5-7.5 分→100 制全是 A 档），但 VLM 的**排序可靠**。
    相对分级直接消费排序信息，与人工语义对齐。

    规则：
    - 「风险」款保留绝对判定（反对率/双渠道规则），不参与分位
    - 同分并列进同一档（分数边界而非排名边界）
    - 幂等：绝对分档存 metadata['absolute_grade']，重复调用无副作用
    - 至少 1 款时仍可分级（单款=S）

    Returns: {style_id: 相对分级}，同时原地更新 p.grade.grade。
    """
    if not preds:
        return {}

    # 首次调用时快照绝对分档（幂等保护）
    for p in preds:
        md = p.metadata
        if "absolute_grade" not in md:
            md["absolute_grade"] = p.grade.grade

    # 风险款不参与分位（绝对规则已判）
    ranked = [p for p in preds if p.metadata.get("absolute_grade") != "风险"]
    result: dict[str, str] = {}
    for p in preds:
        if p.metadata.get("absolute_grade") == "风险":
            p.grade.grade = "风险"
            result[p.info.style_id] = "风险"

    if ranked:
        scores_sorted = sorted(
            (p.grade.final_score for p in ranked), reverse=True,
        )
        n = len(scores_sorted)
        n_s = max(1, round(n * s_pct))
        n_ap = max(1, round(n * a_plus_pct))
        n_a = max(0, round(n * a_pct))
        # 分数边界（同分并列自然进同档）
        s_line = scores_sorted[min(n_s, n) - 1]
        ap_line = scores_sorted[min(n_s + n_ap, n) - 1]
        a_line = scores_sorted[min(n_s + n_ap + n_a, n) - 1]

        for p in ranked:
            sc = p.grade.final_score
            if sc >= s_line:
                g = "S"
            elif sc >= ap_line:
                g = "A+"
            elif sc >= a_line:
                g = "A"
            else:
                g = "P"
            p.grade.grade = g
            result[p.info.style_id] = g

    # ========== P 款主动识别（后处理，在相对分级之后）==========
    # 数据验证：S款色彩安全度≈5.6，P款≈4.7 — 最强区分特征（色彩安全度越低越可能是P款）
    # 触发条件：
    #   (F03≤color_safety_low AND F09≥BRAND_TONE_HIGH) — 高调性 + 低色彩安全 → P款（设计好看但颜色太花）
    #   (F10≥UNIQUENESS_HIGH AND 双渠道弱) → 品牌展示款
    COLOR_SAFETY_LOW = 4.0   # F03 色彩安全度 ≤ 4 → 高饱和/撞色/高风险
    BRAND_TONE_HIGH = 7.5
    UNIQUENESS_HIGH = 7.5
    for p in preds:
        sid = p.info.style_id
        if result.get(sid) == "风险":
            continue  # 风险款不动
        feats = getattr(p, "features", None)
        channels = getattr(p, "channels", None)
        if not feats or not channels:
            continue
        feat_scores = {k: float(getattr(f, "score", 5.0))
                      for k, f in getattr(feats, "features", {}).items()}
        # 兼容新旧 key（F03_color_safety 新 / F03_color_risk 旧）
        f03 = feat_scores.get(
            "F03_color_safety",
            10.0 - feat_scores.get("F03_color_risk", 5.0),
        )
        f09 = feat_scores.get("F09_brand_tone", 5.0)
        f10 = feat_scores.get("F10_uniqueness", 5.0)
        p_showcase = (f09 >= BRAND_TONE_HIGH) and (f03 <= COLOR_SAFETY_LOW)
        channels_weak = (channels.natural_score < 6.5) or (channels.live_score < 6.0)
        p_unique = (f10 >= UNIQUENESS_HIGH) and channels_weak
        if p_showcase or p_unique:
            result[sid] = "P"
            p.grade.grade = "P"
            log.debug("P款识别：%s (F03_safety=%.1f F09=%.1f F10=%.1f → showcase=%s unique=%s)",
                      sid, f03, f09, f10, p_showcase, p_unique)

    log.info("相对分级完成：%s（绝对档快照存 metadata.absolute_grade）",
             {g: list(k for k, v in result.items() if v == g) for g in ("S", "A+", "A", "P")})
    return result


# ========== 渠道推荐 ==========
def _recommend_channel(channels: ChannelScores, info: StyleInfo) -> str:
    diff = channels.live_score - channels.natural_score
    if diff >= 1.5:
        return "直播带货优先"
    elif diff <= -1.5:
        return "自然流量优先"
    elif max(channels.live_score, channels.natural_score) >= 7:
        return "双渠道均衡"
    else:
        return "设计调性款（不做主推）"


# ========== 改款建议（基于扣分特征+BARS量表反查）==========
def _improvement_suggestions(
    feats: StyleFeatures,
    channels: ChannelScores,
    features_cfg: dict[str, Any],
    info_price: float,
) -> list[str]:
    suggestions: list[str] = []
    lowest = feats.lowest_features(4)
    for f in lowest:
        if f.score >= 5.0:
            continue
        fd = features_cfg["features"].get(f.key)
        if not fd:
            continue
        # 找当前档的上一档 label 作为建议
        anchors = list(fd["anchors"].items())  # [(lv, {range, label, desc})]
        # anchors 顺序是 L1(高分档)→L5(低分档)
        current_idx = len(anchors) - 1
        for idx, (_lv, a) in enumerate(anchors):
            rng = a["range"]
            if rng[0] <= f.score <= rng[1]:
                current_idx = idx
                break
        next_level_desc = ""
        if current_idx > 0:
            _next_lv, next_a = anchors[current_idx - 1]
            next_level_desc = f"提升到'{next_a['label']}'：{next_a['description'][:50]}"
        if next_level_desc:
            suggestions.append(f"{f.name}（{f.score:.1f}/10）：{next_level_desc}")
        else:
            suggestions.append(f"优化{f.name}：{f.reason[:40]}")

    if channels.value_match <= -0.15:
        suggestions.append(
            f"定价调整建议：当前价{info_price:.0f}元，感知价值{channels.perceived_value:.1f}/10，"
            f"在本批次价格百分位{channels.price_percentile:.0%}（{channels.price_risk}）。"
            f"建议：① 提升面料/功能感知（F04/F08）；② 或在大促给到8-9折强化VM"
        )
    return suggestions


def decide_grade(
    info: StyleInfo,
    feats: StyleFeatures,
    voting: VotingResult,
    channels: ChannelScores,
    cfg: AppConfig | None = None,
    *,
    calibrated_weights: dict[str, float] | None = None,
    brand_cfg: BrandConfig | None = None,
) -> GradeResult:
    """
    单款综合 → 双维度分 → 矩阵分级 → 改款建议 → 报告要素

    v1.4.42.2 起改为双维度分级：需求潜力（走量能力）× 品牌组合价值（品牌表达贡献），
    P 款（品牌表达款）不再和"难卖/风险款"混为一档。
    """
    if brand_cfg is None and cfg is None:
        fallback = load_brand_profile("mipo")
        brand_cfg = fallback
    if brand_cfg is not None:
        scoring_cfg = brand_cfg.scoring_weights
        features_cfg = brand_cfg.features_bars
    else:
        scoring_cfg = cfg.scoring  # type: ignore[union-attr]
        features_cfg = cfg.features  # type: ignore[union-attr]

    # —— 置信度估算 ——
    feat_divs = [f.divergence for f in feats.features.values()]
    avg_feat_div = sum(feat_divs) / max(1, len(feat_divs))
    confidence_raw = (
        0.35 * (1.0 - min(avg_feat_div / 3.0, 1.0))
        + 0.35 * (1.0 - min(voting.score_std / 3.0, 1.0))
        + 0.30 * (sum(f.confidence for f in feats.features.values()) / max(1, len(feats.features)))
    )
    confidence = clamp(confidence_raw, 0.0, 1.0)

    # —— 双维度分（0-100）——
    if calibrated_weights is not None:
        vec = build_feature_vector(info, feats, voting, channels)
        demand_potential = clamp(sum(vec.get(k, 0.0) * v for k, v in calibrated_weights.items()) * 10.0, 0.0, 100.0)
        _, strengths, weaknesses = _default_aggregate(info, feats, voting, channels, scoring_cfg)
    else:
        demand_potential = _calc_demand_potential(feats, voting, channels, scoring_cfg)
        _, strengths, weaknesses = _default_aggregate(info, feats, voting, channels, scoring_cfg)

    brand_value = _calc_brand_portfolio_value(feats, channels, scoring_cfg)

    # —— final_score 是双维度综合（0.7*demand + 0.3*brand）——
    # 向后兼容：0.7 权重因为走量是主要决策依据，品牌价值 0.3 是组合调优
    final_score = round(demand_potential * 0.7 + brand_value * 0.3, 1)

    # —— 双维度矩阵分级 ——
    grade, grade_note = _grade_from_dual_dimension(demand_potential, brand_value, channels, voting)

    improvements = _improvement_suggestions(feats, channels, features_cfg, info_price=info.price)

    insight_parts = []
    if voting.support_rate >= 0.4:
        insight_parts.append(f"约{voting.support_rate:.0%}的核心客群明确愿意购买")
    if voting.opposition_rate >= 0.2:
        insight_parts.append(f"约{voting.opposition_rate:.0%}的人设存在顾虑")
    if voting.top_buy_reasons:
        insight_parts.append("核心卖点：" + "；".join(voting.top_buy_reasons[:2]))
    if voting.top_oppose_reasons:
        insight_parts.append("主要顾虑：" + "；".join(voting.top_oppose_reasons[:2]))
    consumer_insights = "。".join(insight_parts) if insight_parts else "客群态度较为中性"

    return GradeResult(
        style_id=info.style_id,
        grade=grade,
        final_score=final_score,
        confidence=round(confidence, 2),
        demand_potential=demand_potential,
        brand_portfolio_value=brand_value,
        grade_note=grade_note,
        strengths=strengths,
        weaknesses=weaknesses,
        improvements=improvements,
        recommended_channel=_recommend_channel(channels, info),
        consumer_insights=consumer_insights,
    )
