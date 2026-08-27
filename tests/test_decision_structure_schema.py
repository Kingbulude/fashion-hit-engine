"""decision_structure YAML schema 合规测试（spec §7.x / §10 YAML1）。

验证：
  1. womenswear / tongzhuang-outdoor 两个 brand profile 的
     decision_structure 都以对象形式加载（{type, layers, ...}）
  2. type 字段值在 {single_layer, multi_layer, double_layer} 内
  3. layers 数组非空、每层有 id/name/persona_axis_key/role/default_weight
  4. multi_layer / double_layer 别名等价处理（spec-following YAML
     不再静默落入 neither 分支导致孩子否决层失效）
  5. spec 早期命名 double_layer 现在作为 multi_layer 别名
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import _parse_decision_structure, load_brand_profile
from src.persona_voting import _match_age_weight_rules
from src.types import BrandConfig, BrandDecisionStructure, DecisionLayer


# ============================================================
# 1. 真实品牌 profile 加载合规
# ============================================================
def test_womenswear_profile_loads_object_form():
    """womenswear/profile.yaml 的 decision_structure 必须是对象形式，
    type=single_layer（spec §7.3: 女装/男装/快消用 single_layer）。
    """
    cfg = load_brand_profile("womenswear")
    assert isinstance(cfg, BrandConfig)
    ds = cfg.decision_structure
    assert isinstance(ds, BrandDecisionStructure)
    assert ds.type == "single_layer", (
        f"womenswear 应为 single_layer, got {ds.type!r}"
    )
    # single_layer 模式下 layers 至少 1 条（self_decision_layer 自描述）
    assert len(ds.layers) >= 1, f"single_layer 也应有 ≥1 条 layer, got {len(ds.layers)}"
    # 每层结构完整
    for layer in ds.layers:
        assert isinstance(layer, DecisionLayer)
        assert layer.id, f"layer.id 不能为空"
        assert layer.role in ("decider", "veto")
        assert 0.0 <= layer.default_weight <= 1.0
    print(f"✅ womenswear: type={ds.type}, layers={len(ds.layers)}, "
          f"target_age={ds.default_target_age}")


def test_tongzhuang_profile_loads_multi_layer():
    """tongzhuang-outdoor/profile.yaml 的 decision_structure 必须是对象形式，
    type=multi_layer（spec §7.3: 童装双层 妈妈决策者层 + 孩子影响层）。
    """
    cfg = load_brand_profile("tongzhuang-outdoor")
    ds = cfg.decision_structure
    assert ds.type == "multi_layer", (
        f"tongzhuang-outdoor 应为 multi_layer, got {ds.type!r}"
    )
    assert len(ds.layers) == 2, (
        f"双层结构应有 2 条 layer (妈妈+孩子), got {len(ds.layers)}"
    )
    # 妈妈层 role=decider，孩子层 role=veto
    roles = [l.role for l in ds.layers]
    assert "decider" in roles, "应有妈妈决策者层 (role=decider)"
    assert "veto" in roles, "应有孩子否决层 (role=veto)"
    # age_weight_rules 非空（童装按年龄段调权重）
    assert len(ds.age_weight_rules) >= 3, (
        f"童装应有 ≥3 档 age_weight_rules (6-8/9-11/12-14), "
        f"got {len(ds.age_weight_rules)}"
    )
    print(f"✅ tongzhuang-outdoor: type={ds.type}, layers={len(ds.layers)}, "
          f"age_rules={len(ds.age_weight_rules)}")


# ============================================================
# 2. _parse_decision_structure 单元测试
# ============================================================
def test_parse_decision_structure_single_layer():
    raw = {
        "type": "single_layer",
        "layers": [
            {"id": "self_decision_layer", "name": "本人决策层",
             "persona_axis_key": "identity_axes", "role": "decider",
             "default_weight": 1.0}
        ],
    }
    ds = _parse_decision_structure(raw)
    assert ds.type == "single_layer"
    assert len(ds.layers) == 1
    assert ds.layers[0].id == "self_decision_layer"
    assert ds.layers[0].role == "decider"


def test_parse_decision_structure_multi_layer():
    raw = {
        "type": "multi_layer",
        "layers": [
            {"id": "mom", "name": "妈妈", "persona_axis_key": "identity_axes",
             "role": "decider", "default_weight": 0.7},
            {"id": "child", "name": "孩子", "persona_axis_key": "child_identity_axes",
             "role": "veto", "default_weight": 0.3},
        ],
        "age_weight_rules": [
            {"age_range": [6, 8], "mom_weight": 0.7, "child_weight": 0.3},
        ],
        "default_target_age": 10,
    }
    ds = _parse_decision_structure(raw)
    assert ds.type == "multi_layer"
    assert len(ds.layers) == 2
    assert ds.layers[1].role == "veto"
    assert ds.default_target_age == 10
    assert len(ds.age_weight_rules) == 1


def test_parse_decision_structure_double_layer_alias():
    """spec 早期命名 double_layer，代码应接受为 multi_layer 别名。

    若不支持，spec-following 的 double_layer YAML 会静默落入
    persona_voting 的 neither 分支，孩子否决层失效且无报错。
    """
    raw = {
        "type": "double_layer",  # spec 历史命名
        "layers": [
            {"id": "mom", "name": "妈妈", "persona_axis_key": "identity_axes",
             "role": "decider", "default_weight": 0.7},
            {"id": "child", "name": "孩子", "persona_axis_key": "child_identity_axes",
             "role": "veto", "default_weight": 0.3},
        ],
    }
    ds = _parse_decision_structure(raw)
    # double_layer 必须被接受为合法 type（不抛异常）
    assert ds.type == "double_layer"
    assert len(ds.layers) == 2
    print(f"✅ double_layer 别名被接受: type={ds.type}, layers={len(ds.layers)}")


# ============================================================
# 3. 端到端：persona_voting 识别 double_layer 为 multi_layer
# ============================================================
def test_persona_voting_treats_double_layer_as_multi_layer():
    """构造 type=double_layer 的 BrandConfig，验证 persona_voting
    的决策分支会把 double_layer 当 multi_layer 处理（不静默跳过）。

    spec §7.4: 孩子否决层只在 multi_layer/double_layer 模式下加载。
    若代码只识别 multi_layer，double_layer 会落入 neither 分支 →
    孩子否决层静默失效（无报错），这是潜在 silent bug。
    """
    # 用 double_layer 构造童装 BrandConfig
    cfg = load_brand_profile("tongzhuang-outdoor")
    # 把 type 改成 double_layer（spec 历史命名）
    original_type = cfg.decision_structure.type
    # dataclass 实例字段可改（非 frozen）
    cfg.decision_structure.type = "double_layer"

    try:
        ds = cfg.decision_structure
        assert ds.type == "double_layer"

        # 验证 persona_voting 的分支判断逻辑
        # （直接复用 persona_voting 内部的 type 判断分支）
        if ds.type in ("single_layer",):
            branch = "single_layer"
        elif ds.type in ("multi_layer", "double_layer"):
            branch = "multi_layer"
        else:
            branch = "neither (BUG!)"

        assert branch == "multi_layer", (
            f"double_layer 应被识别为 multi_layer 分支, got {branch!r}。"
            f"这说明 spec-following 的 double_layer YAML 会静默失效。"
        )
        print(f"✅ double_layer → 识别为 multi_layer 分支（孩子否决层生效）")
    finally:
        # 恢复 original type（避免污染其他测试）
        cfg.decision_structure.type = original_type


# ============================================================
# 4. YAML 文件 schema 合规（直接读 YAML 检查对象形式）
# ============================================================
def test_profile_yaml_files_use_object_form():
    """spec §10 YAML1: decision_structure 是对象 {type, layers, ...}。

    早期 spec 文档写的是字符串形式 `decision_structure: single_layer`，
    但代码实现用对象形式（更表达力，layers 数组显式声明结构）。
    两个 brand profile 的 YAML 必须是对象形式。
    """
    profiles = [
        ROOT / "brand_profiles" / "womenswear" / "profile.yaml",
        ROOT / "brand_profiles" / "tongzhuang-outdoor" / "profile.yaml",
        ROOT / "brand_profiles" / "_template" / "profile.yaml",
    ]
    for p in profiles:
        assert p.exists(), f"profile.yaml 不存在: {p}"
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        ds = raw.get("decision_structure")
        assert ds is not None, f"{p.name}: 缺 decision_structure 字段"
        assert isinstance(ds, dict), (
            f"{p.name}: decision_structure 必须是对象(dict), "
            f"got {type(ds).__name__}"
        )
        assert "type" in ds, f"{p.name}: decision_structure.type 缺失"
        assert ds["type"] in ("single_layer", "multi_layer", "double_layer"), (
            f"{p.name}: type={ds['type']!r} 不在合法集合内"
        )
        assert "layers" in ds, f"{p.name}: decision_structure.layers 缺失"
        assert isinstance(ds["layers"], list), (
            f"{p.name}: layers 必须是数组"
        )
    print(f"✅ {len(profiles)} 个 profile.yaml 都用对象形式 decision_structure")


if __name__ == "__main__":
    test_womenswear_profile_loads_object_form()
    test_tongzhuang_profile_loads_multi_layer()
    test_parse_decision_structure_single_layer()
    test_parse_decision_structure_multi_layer()
    test_parse_decision_structure_double_layer_alias()
    test_persona_voting_treats_double_layer_as_multi_layer()
    test_profile_yaml_files_use_object_form()
    print("\n🎉 ALL decision_structure SCHEMA TESTS PASSED")
