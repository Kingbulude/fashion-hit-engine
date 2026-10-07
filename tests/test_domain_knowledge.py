"""domain_knowledge 跨 segment 模板隔离测试。

核心假设：每个 industry_segment 的 base 模板只包含该品类的专属知识。
如果 children_outdoor 的"GB 20227"出现在 women_fashion 模板里，
那就是模板污染 —— 专家 prompt 会被无关国标/竞品信息干扰。

这些断言防的就是：某人改 _T_CHILDREN_OUTDOOR 时不小心把内容粘到了 _T_WOMEN_FASHION 里，
或者新增 segment 时复制粘贴漏了删。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.domain_knowledge import (
    SEGMENT_TEMPLATES,
    SEGMENT_EXPERT_CREDENTIALS,
    assemble_domain_knowledge,
    get_expert_credentials,
)
from src.config import load_brand_profile
from src.types import BrandConfig


# ──────────────────────────────────────────────────────────────
# 各 segment 的专属关键词（不应出现在其他 segment 模板里）
# ──────────────────────────────────────────────────────────────

_CHILDREN_ONLY_TERMS = [
    "GB 20227",          # 童装绳带安全国标（儿童专属，成人不需要）
    "探路者 kids",        # 童装户外竞品
    "绳带超 GB",         # 童装红线
    "荧光剂",            # 童装 A 类红线（成人也提但"A 类禁售"是童装独特监管）
    "帽绳必须可锁",
    "肩线前倾 15°",       # 儿童圆肩体态专属
    "尺码 110-180 码",    # 童装尺码体系（和成人 XS-XL 不同）
]

_WOMEN_ONLY_TERMS = [
    "梨形",              # 女性体型
    "苹果形",
    "沙漏形",
    "藏肉",              # 女装核心诉求
    "ZARA 快时尚",
    "优衣库基础款",
    "落肩/插肩",
    "通勤装",
    "社交/派对",
]

_SPORTSWEAR_ONLY_TERMS = [
    "压缩衣",            # 运动专属
    "压缩袜",
    "压力值 15-20mmHg",
    "撸铁/瑜伽",
    "运动 bra",
    "Lululemon",
]

_OUTDOOR_ONLY_TERMS = [
    "GORE-TEX Pro",      # 硬核户外专属
    "Primaloft",
    "3L GORE-TEX",
    "徒步露营",
    "登山专业级",
    "防水静水压 ≥5000mm",
    "腋下透气拉链",
]


def _assert_no_cross_contamination(template_text: str, forbidden_terms: list[str], segment_label: str) -> None:
    """断言 template_text 不含 forbidden_terms 中的任何一个。
    发现就列出来，让维护者一眼看到是哪个词串进来了。
    """
    hit = [t for t in forbidden_terms if t in template_text]
    assert not hit, (
        f"[{segment_label}] 模板污染！发现了不属于此 segment 的关键词: {hit}\n"
        f"如果这是你故意加的，请更新 forbidden_terms 列表。"
    )


# ──────────────────────────────────────────────────────────────
# 1. 每个 segment 的 base 模板都已注册（没有悬空的孤儿常量）
# ──────────────────────────────────────────────────────────────

def test_all_segments_have_templates():
    EXPECTED = {"children_outdoor", "children_fashion", "women_fashion",
                "men_fashion", "sportswear", "outdoor"}
    assert set(SEGMENT_TEMPLATES.keys()) == EXPECTED, (
        f"SEGMENT_TEMPLATES 应有 {EXPECTED}，实际 = {set(SEGMENT_TEMPLATES.keys())}"
    )


def test_all_templates_are_nonempty_strings():
    for seg, tpl in SEGMENT_TEMPLATES.items():
        assert isinstance(tpl, str) and len(tpl) > 50, (
            f"segment {seg} 模板不是有效长文本: {type(tpl).__name__} / len={len(tpl) if isinstance(tpl, str) else 'N/A'}"
        )


def test_all_segments_have_expert_credentials():
    """专家履历 credentials 也必须和模板一一对应，不能缺。"""
    missing = set(SEGMENT_TEMPLATES.keys()) - set(SEGMENT_EXPERT_CREDENTIALS.keys())
    assert not missing, f"SEGMENT_EXPERT_CREDENTIALS 缺少 segment: {missing}"


# ──────────────────────────────────────────────────────────────
# 2. 跨 segment 污染检测（核心断言）
# ──────────────────────────────────────────────────────────────

def test_women_fashion_template_no_children_terms():
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["women_fashion"],
        _CHILDREN_ONLY_TERMS,
        "women_fashion",
    )


def test_women_fashion_template_no_sports_terms():
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["women_fashion"],
        _SPORTSWEAR_ONLY_TERMS,
        "women_fashion",
    )


def test_women_fashion_template_no_outdoor_terms():
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["women_fashion"],
        _OUTDOOR_ONLY_TERMS,
        "women_fashion",
    )


def test_children_outdoor_template_no_women_terms():
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["children_outdoor"],
        _WOMEN_ONLY_TERMS,
        "children_outdoor",
    )


def test_sportswear_template_no_children_safety_terms():
    """运动品类不需要关心绳带 GB 20227 —— 那是童装红线。"""
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["sportswear"],
        _CHILDREN_ONLY_TERMS,
        "sportswear",
    )


def test_outdoor_template_no_women_terms():
    _assert_no_cross_contamination(
        SEGMENT_TEMPLATES["outdoor"],
        _WOMEN_ONLY_TERMS,
        "outdoor",
    )


# ──────────────────────────────────────────────────────────────
# 3. assemble_domain_knowledge 端到端：占位符正确填充 + overlay 叠加
# ──────────────────────────────────────────────────────────────

def _make_brand_cfg(segment: str, *, overlay: str | None = None) -> BrandConfig:
    """快速构造一个 BrandConfig（只填 domain_knowledge 需要的字段）。"""
    # 用 mipo 的 profile 作骨架，改几个字段
    cfg = load_brand_profile("mipo")
    cfg.industry_segment = segment
    cfg.brand_domain_knowledge = overlay
    if segment == "women_fashion":
        cfg.brand_name = "ZARA 快时尚"
        cfg.target_age_range = [20, 35]
        cfg.target_size_range = ["S", "L"]
    elif segment == "sportswear":
        cfg.brand_name = "MAIA ACTIVE"
        cfg.target_age_range = [18, 40]
        cfg.target_size_range = ["S", "XL"]
    return cfg


def test_assemble_women_fashion_placeholders():
    cfg = _make_brand_cfg("women_fashion")
    text = assemble_domain_knowledge(cfg)
    assert text is not None
    # 占位符必须被替换
    assert "{brand_name}" not in text
    assert "{target_age_str}" not in text
    assert "{target_size_str}" not in text
    assert "ZARA 快时尚" in text
    assert "20-35岁" in text
    # 内容正确
    assert "女装" in text
    assert "通勤装" in text


def test_assemble_children_outplaceholders():
    cfg = _make_brand_cfg("children_outdoor")
    text = assemble_domain_knowledge(cfg)
    assert text is not None
    assert "{brand_name}" not in text
    # 童装关键术语必须在
    assert "GB 20227" in text
    assert "U 型裆布" in text  # 模板里带空格


def test_assemble_no_segment_no_overlay_returns_none():
    cfg = _make_brand_cfg("")  # 空 segment
    cfg.brand_domain_knowledge = None
    result = assemble_domain_knowledge(cfg)
    assert result is None


def test_assemble_overlay_appended():
    cfg = _make_brand_cfg("women_fashion", overlay="""
【ZARA 专属补充】
· 我们家主打法式极简风
· 快反供应链 14 天上新
""")
    text = assemble_domain_knowledge(cfg)
    assert text is not None
    assert "ZARA 专属补充" in text
    # base 模板也在
    assert "通勤装" in text


def test_assemble_overlay_only_no_segment_template():
    """没有 segment 模板但有 brand overlay → 只返回 overlay（不应该返回 None）"""
    cfg = _make_brand_cfg("")  # 空 segment → 无 base 模板
    cfg.brand_domain_knowledge = "我有专属知识"
    text = assemble_domain_knowledge(cfg)
    assert text is not None
    assert "我有专属知识" in text


# ──────────────────────────────────────────────────────────────
# 4. get_expert_credentials 也按 segment 隔离
# ──────────────────────────────────────────────────────────────

def test_expert_credentials_women_no_children_refs():
    cfg = _make_brand_cfg("women_fashion")
    creds = get_expert_credentials(cfg)
    # 女装专家履历不该提"童装"、"探路者 kids"
    assert "探路者 kids" not in creds
    assert "童装" not in creds


def test_expert_credentials_children_outdoor_has_kids_refs():
    cfg = _make_brand_cfg("children_outdoor")
    creds = get_expert_credentials(cfg)
    assert "探路者 kids" in creds
    assert "童装" in creds


def test_expert_credentials_unknown_segment_defaults_safely():
    """未知 segment 走默认（women），不应抛异常。"""
    cfg = _make_brand_cfg("luxury_handbags")  # 未注册
    creds = get_expert_credentials(cfg)
    assert isinstance(creds, str) and len(creds) > 20
