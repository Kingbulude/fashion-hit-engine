"""通用决策层（非童装品牌）单元测试。

背景：persona 投票曾硬编码「妈妈/孩子」双层（mipo 童装事实泄漏到代码）。
重构后代码只认 decision_structure.layers（决策层通用），
「妈妈/孩子」语义仅存在于 mipo 品牌包 YAML。

用成人女装三层结构（无任何童装语义）验证：
  1. 层权重解析（age_weight_rules 通用 layer_weights 直配）
  2. 旧 YAML 字段名兼容（mom_weight/child_weight → 决策者层/影响层）
  3. 投票 prompt 按层动态渲染，不硬编码角色
  4. 否决层（role=veto）轴扫描通用化（persona_axes 按 persona_axis_key 取）
  5. aggregate_votes 对任意层数正确聚合
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.persona_voting import (  # noqa: E402
    VETO_PENALTY_FACTOR,
    _check_axis_veto,
    _layer_weights_for_age,
    _match_age_weight_rules,
    _render_persona_prompt,
    aggregate_votes,
)
from src.config import _parse_decision_structure  # noqa: E402
from src.types import (  # noqa: E402
    BrandConfig,
    FeatureScore,
    PersonaVote,
    StyleFeatures,
    StyleInfo,
)

# ========== 构造成人女装三层品牌（无童装语义）==========
LADY_RAW = {
    "type": "multi_layer",
    "layers": [
        {"id": "buyer_layer", "name": "女性买家层",
         "persona_axis_key": "identity_axes", "role": "decider", "default_weight": 0.60},
        {"id": "partner_layer", "name": "伴侣意见层",
         "persona_axis_key": "partner_axes", "role": "veto", "default_weight": 0.25},
        {"id": "bestie_layer", "name": "闺蜜种草层",
         "persona_axis_key": "bestie_axes", "role": "veto", "default_weight": 0.15},
    ],
    # 通用写法：按 layer_id 直配权重（不依赖 mom_weight/child_weight 字段名）
    "age_weight_rules": [
        {"age_range": [20, 35], "layer_weights": {
            "buyer_layer": 0.55, "partner_layer": 0.30, "bestie_layer": 0.15}},
        {"age_range": [36, 55], "layer_weights": {
            "buyer_layer": 0.40, "partner_layer": 0.40, "bestie_layer": 0.20}},
    ],
    "default_target_age": 28,
}

# 兼容写法：mipo 式 mom_weight/child_weight（决策者层/影响层）
LEGACY_RAW = {
    "type": "multi_layer",
    "layers": [
        {"id": "decider_layer", "name": "决策者层",
         "persona_axis_key": "identity_axes", "role": "decider", "default_weight": 0.60},
        {"id": "influencer_layer", "name": "影响层",
         "persona_axis_key": "influencer_axes", "role": "veto", "default_weight": 0.40},
    ],
    "age_weight_rules": [
        {"age_range": [0, 99], "mom_weight": 0.70, "child_weight": 0.30},
    ],
    "default_target_age": 28,
}


def _lady_brand() -> BrandConfig:
    ds = _parse_decision_structure(LADY_RAW)
    return BrandConfig(
        brand_id="ladywear",
        brand_name="测试女装品牌",
        decision_structure=ds,
        features_bars={},
        personas=[
            {"persona_id": "P01", "name": "通勤简约人设", "weight": 0.6,
             "fab_focus": ["通勤", "挺括"], "color_preference": ["藏青", "黑"]},
            {"persona_id": "P02", "name": "约会甜美人设", "weight": 0.4,
             "fab_focus": ["裙子", "收腰"], "color_preference": ["粉", "白"]},
        ],
        scoring_weights={},
        category_registry={},
        default_engine_weights={},
        default_channel_split={},
        grading_thresholds={},
        calibrated_dir="/tmp/ladywear_calibrated",
        # 通用轴数据：key=layer.persona_axis_key（与童装的 child_identity_axes 同构）
        persona_axes={
            "partner_axes": [
                {"age": 28, "gender": "male", "veto_when": ["太夸张的印花", "超短裙"]},
            ],
            "bestie_axes": [
                {"age": 28, "gender": "female", "veto_when": ["老气的颜色"]},
            ],
        },
    )


def _feats(fab: str = "") -> StyleFeatures:
    fs = StyleFeatures(style_id="L001")
    for k, (name, score) in {
        "F01_silhouette": ("廓形", 6.0),
        "F03_color_risk": ("颜色风险", 5.0),
    }.items():
        fs.features[k] = FeatureScore(
            key=k, name=name, category="外观", score=score,
            confidence=0.8, reason=fab or "常规",
        )
    return fs


def test_layer_weights_generic_direct_config():
    """三层结构用 layer_weights 直配：权重按 layer_id 正确解析并归一化"""
    brand = _lady_brand()
    w = _layer_weights_for_age(brand.decision_structure, 28)
    assert set(w) == {"buyer_layer", "partner_layer", "bestie_layer"}
    assert abs(w["buyer_layer"] - 0.55) < 1e-9
    assert abs(w["partner_layer"] - 0.30) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # 第二档年龄
    w2 = _layer_weights_for_age(brand.decision_structure, 45)
    assert abs(w2["buyer_layer"] - 0.40) < 1e-9


def test_layer_weights_legacy_mipo_fields():
    """旧 YAML 字段 mom_weight/child_weight 映射到决策者层/影响层（向后兼容）"""
    ds = _parse_decision_structure(LEGACY_RAW)
    w = _layer_weights_for_age(ds, 28)
    assert abs(w["decider_layer"] - 0.70) < 1e-9
    assert abs(w["influencer_layer"] - 0.30) < 1e-9


def test_prompt_renders_layers_dynamically():
    """投票 prompt 按层动态渲染：含各层名/输出字段，不含硬编码角色文案"""
    brand = _lady_brand()
    layers = list(brand.decision_structure.layers)
    age_rule = _match_age_weight_rules(brand.decision_structure.age_weight_rules, 28)
    sys_p, usr_p = _render_persona_prompt(
        brand.personas[0], StyleInfo(style_id="L001", price=299.0),
        _feats(), layers, _layer_weights_for_age(brand.decision_structure, 28),
        brand, age_rule,
    )
    for l in layers:
        assert l.name in usr_p, f"prompt 应包含层名 {l.name}"
        assert f'"{l.id}_score"' in usr_p, f"输出schema应含 {l.id}_score"
    # 影响层轴画像进入 prompt
    assert "太夸张的印花" in usr_p
    # 无童装语义硬编码
    for kw in ("妈妈", "孩子", "童装"):
        assert kw not in sys_p + usr_p, f"通用品牌 prompt 不应出现硬编码「{kw}」"


def test_axis_veto_generic_layers():
    """否决层轴扫描：命中 veto_when → 惩罚；未命中 → 原分"""
    brand = _lady_brand()
    layers = list(brand.decision_structure.layers)
    age_rule = _match_age_weight_rules(brand.decision_structure.age_weight_rules, 28)

    # FAB 含「超短裙」→ partner_layer 否决
    vetoed, reason = _check_axis_veto(brand, layers, age_rule, ["超短裙"])
    assert vetoed and "伴侣意见层" in reason

    vetoed2, _ = _check_axis_veto(brand, layers, age_rule, ["常规直筒裤"])
    assert not vetoed2


def test_aggregate_votes_three_layer_brand():
    """三层品牌聚合：基础分=人设加权；命中否决线 → ×VETO_PENALTY_FACTOR"""
    brand = _lady_brand()
    votes = [
        PersonaVote(persona_id="P01", persona_name="通勤简约人设",
                    layer_scores={"buyer_layer": 8.0, "partner_layer": 7.0, "bestie_layer": 6.0},
                    layer_reasons={"buyer_layer": "通勤百搭"},
                    final_score=8.0),
        PersonaVote(persona_id="P02", persona_name="约会甜美人设",
                    layer_scores={"buyer_layer": 6.0, "partner_layer": 6.0, "bestie_layer": 5.0},
                    layer_reasons={"buyer_layer": "约会可穿"},
                    final_score=6.0),
    ]
    # 无款式特征 → 无关键词扫描 → 基础分
    r1 = aggregate_votes(votes, brand_cfg=brand)
    expect_base = 8.0 * 0.6 + 6.0 * 0.4
    assert abs(r1.weighted_score - expect_base) < 1e-9

    # 款式含「超短裙」→ 伴侣否决惩罚
    info = StyleInfo(style_id="L001", price=299.0, fab_description="超短裙包臀")
    r2 = aggregate_votes(votes, brand_cfg=brand, feats=_feats("超短裙包臀"), info=info)
    assert abs(r2.weighted_score - expect_base * VETO_PENALTY_FACTOR) < 1e-9
    assert any("伴侣意见层" in x for x in r2.top_oppose_reasons) or r2.opposition_rate > 0


def test_legacy_aggregate_single_layer():
    """无 brand_cfg（无决策结构）时退化为单决策层人设加权，行为不变"""
    votes = [
        PersonaVote(persona_id="P01", persona_name="人设1",
                    layer_scores={"decider_layer": 8.0}, final_score=8.0),
        PersonaVote(persona_id="P02", persona_name="人设2",
                    layer_scores={"decider_layer": 6.0}, final_score=6.0),
    ]
    r = aggregate_votes(votes, personas_cfg={"personas": [
        {"id": "P01", "weight": 0.5}, {"id": "P02", "weight": 0.5}]})
    assert abs(r.weighted_score - 7.0) < 1e-9


if __name__ == "__main__":
    test_layer_weights_generic_direct_config()
    test_layer_weights_legacy_mipo_fields()
    test_prompt_renders_layers_dynamically()
    test_axis_veto_generic_layers()
    test_aggregate_votes_three_layer_brand()
    test_legacy_aggregate_single_layer()
    print("✅ 通用决策层（非童装）测试全部通过")
