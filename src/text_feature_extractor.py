"""版型/设计描述文本 → F11-F20 结构化特征规则引擎

Phase 0 baseline 已用 inline 代码验证过：AUC-ROC 0.78, Top-10 Recall 75%。
本模块将规则模块化，保持完全一致的规则逻辑，确保可复现该性能。

输入文本格式通常为三段式：
    面料：...
    版型：...
    颜色：...
每段内部可用中文冒号/英文冒号，段间用换行或分号分隔。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ============================================================
# 关键词词库（与 Phase 0 inline 代码完全一致）
# ============================================================

_F11_TECH_KEYWORDS = [
    "科技", "机能", "速干", "吸湿", "弹力", "SGS", "检测", "认证",
    "透气", "防水", "防风", "防晒", "防泼水", "抗静电", "三防",
]

_F12_EXPENSIVE_KEYWORDS = [
    "灯芯绒", "羊毛", "混纺", "科技面料", "锦纶", "聚酯纤维", "氨纶",
    "摇粒绒", "特氟龙", "塔丝隆", "抓绒", "羽绒",
]

_F12_CHEAP_KEYWORDS = ["棉", "涤"]

_F13_LOOSE_KEYWORDS = [
    "宽松", "落肩", "直筒", "阔腿", "束脚", "O型", "A型", "H型", "连帽",
]

_F13_TIGHT_KEYWORDS = ["修身", "紧身", "短款", "微收"]

_F14_CLASSIC_KEYWORDS = ["经典", "基础", "复古", "简约", "纯色"]

_F14_UNIQUE_KEYWORDS = [
    "独家", "立体", "茧型", "拼色", "撞色", "图案", "印花",
]

_F15_DETAIL_KEYWORDS = [
    "口袋", "抽绳", "弹性", "松紧", "印花", "不掉", "可拆卸",
    "调节", "反光", "拉链", "魔术扣", "拇指扣", "内里", "里料", "外层",
]

_F19_SAFE_KEYWORDS = [
    "黑", "灰", "白", "藏青", "深青", "墨绿", "深灰", "浅灰", "米白", "卡其",
]

_F19_RISKY_KEYWORDS = ["粉", "黄", "荧光", "芥末", "橙", "红"]

_F20_CERT_KEYWORDS = ["SGS", "ISO", "OEKO", "检测符合", "检测"]


# ============================================================
# 段提取正则
# ============================================================

_RE_FABRIC = re.compile(r"面料[：:](.+?)(?=版型[：:]|颜色[：:]|$)", re.DOTALL)
_RE_SILHOUETTE = re.compile(r"版型[：:](.+?)(?=面料[：:]|颜色[：:]|$)", re.DOTALL)
_RE_COLOR = re.compile(r"颜色[：:](.+?)(?=面料[：:]|版型[：:]|$)", re.DOTALL)


# ============================================================
# 工具函数
# ============================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _count_hits(text: str, keywords: list[str]) -> int:
    """关键词命中计数。同一段文本里同一关键词多次出现只算 1 次（去重）。"""
    hits = 0
    for kw in keywords:
        if kw in text:
            hits += 1
    return hits


def _safe_get_segment(pattern: re.Pattern, text: str) -> str:
    """用正则提取段落内容，失败返回空字符串。"""
    try:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _remove_parentheses(text: str) -> str:
    """移除中文/英文括号及其内容（颜色段落里常有「藏青（主推色）」）。"""
    return re.sub(r"[（(][^）)]*[）)]", "", text)


def _split_colors(color_seg: str) -> list[str]:
    """把颜色段落按分隔符拆成颜色列表。

    分隔符：/ 、, \n
    拆分后每条 strip 并移除括号内容；过滤掉"主推色"这类修饰词和空串。
    """
    if not color_seg:
        return []
    # 统一分隔符：/ 、 , （全半角） \n
    normalized = re.sub(r"[、，,\n]+", "/", color_seg)
    parts = [p.strip() for p in normalized.split("/")]
    result = []
    for p in parts:
        p = _remove_parentheses(p).strip()
        # 过滤纯修饰词（主推色 / 推荐色 / 新色 / 限量色 等）
        if not p:
            continue
        if p in {"主推色", "推荐色", "新色", "限量色", "经典色"}:
            continue
        # 去掉尾部的 '色' 字冗余（保留主体），但只有长度>1 才收
        if len(p) >= 1:
            result.append(p)
    return result


# ============================================================
# 单特征计算（每个函数独立，便于单测）
# ============================================================

def _calc_f11(text_full: str, fabric_seg: str) -> float:
    """F11 面料科技含量(0-10)：科技关键词命中数 ×1.5，cap 10.0。

    在整段文本 + 面料段都搜，更全。
    """
    haystack = f"{text_full}\n{fabric_seg}"
    hits = _count_hits(haystack, _F11_TECH_KEYWORDS)
    return round(_clamp(hits * 1.5, 0.0, 10.0), 2)


def _calc_f12(text_full: str, fabric_seg: str) -> float:
    """F12 面料成本感知(0-10)：expensive×1.2 − cheap×0.5，cap 10, floor 0。"""
    haystack = f"{text_full}\n{fabric_seg}"
    expensive = _count_hits(haystack, _F12_EXPENSIVE_KEYWORDS)
    cheap = _count_hits(haystack, _F12_CHEAP_KEYWORDS)
    score = expensive * 1.2 - cheap * 0.5
    return round(_clamp(score, 0.0, 10.0), 2)


def _calc_f13(silhouette_seg: str) -> float:
    """F13 版型宽松度(0-10)：loose×1.5 − tight×1.5 + 3，cap 10，floor 1.5。"""
    loose = _count_hits(silhouette_seg, _F13_LOOSE_KEYWORDS)
    tight = _count_hits(silhouette_seg, _F13_TIGHT_KEYWORDS)
    score = loose * 1.5 - tight * 1.5 + 3
    return round(_clamp(score, 1.5, 10.0), 2)


def _calc_f14(silhouette_seg: str) -> float:
    """F14 版型经典度(0-10)：classic×2 − unique×1 + 4，cap 10，floor 0。"""
    classic = _count_hits(silhouette_seg, _F14_CLASSIC_KEYWORDS)
    unique = _count_hits(silhouette_seg, _F14_UNIQUE_KEYWORDS)
    score = classic * 2 - unique * 1 + 4
    return round(_clamp(score, 0.0, 10.0), 2)


def _calc_f15(text_full: str, fabric_seg: str, silhouette_seg: str) -> int:
    """F15 功能细节数量(计数)：detail 关键词命中数。"""
    haystack = f"{fabric_seg}\n{silhouette_seg}\n{text_full}"
    return _count_hits(haystack, _F15_DETAIL_KEYWORDS)


def _calc_f16(silhouette_seg: str) -> int:
    """F16 版型描述长度(计数)：版型段的字符数（去空白）。"""
    return len(re.sub(r"\s+", "", silhouette_seg))


def _calc_f17(fabric_seg: str) -> int:
    """F17 面料描述长度(计数)：面料段的字符数（去空白）。"""
    return len(re.sub(r"\s+", "", fabric_seg))


def _calc_f18(color_seg: str) -> int:
    """F18 颜色数量(计数)：颜色段按 '/、,换行' 分割，去除修饰词。"""
    colors = _split_colors(color_seg)
    return len(colors)


def _calc_f19(color_seg: str) -> float:
    """F19 颜色安全度(0-10)：safe×1.5 − risky×2 + 5，cap 10，floor 1。"""
    colors = _split_colors(color_seg)
    haystack = " ".join(colors)
    safe = _count_hits(haystack, _F19_SAFE_KEYWORDS)
    risky = _count_hits(haystack, _F19_RISKY_KEYWORDS)
    score = safe * 1.5 - risky * 2 + 5
    return round(_clamp(score, 1.0, 10.0), 2)


def _calc_f20(text_full: str) -> int:
    """F20 检测认证数(计数)：cert 关键词命中数。"""
    return _count_hits(text_full, _F20_CERT_KEYWORDS)


# ============================================================
# 主入口
# ============================================================

def extract_text_features(text: Any) -> dict[str, Any]:
    """从版型/设计描述文本提取 F11-F20 共 10 个结构化特征。

    Args:
        text: 原始文本（通常是 Excel 的"版型/设计描述"列单元格值）。

    Returns:
        dict with keys:
            - F11..F17, F20: float 或 int（F15/F16/F17/F18/F20 为计数）
            - F18 / F19: 颜色相关（解析异常时走 fallback 默认值）
            - missing_flags: list[str]，标记哪些段/解析出了问题

    永不抛异常。
    """
    missing_flags: list[str] = []

    # --- 输入校验 ---
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()

    if not text:
        missing_flags.append("empty_text")
        return _default_result(missing_flags)

    # --- 段提取 ---
    fabric_seg = _safe_get_segment(_RE_FABRIC, text)
    silhouette_seg = _safe_get_segment(_RE_SILHOUETTE, text)
    color_seg = _safe_get_segment(_RE_COLOR, text)

    if not fabric_seg:
        missing_flags.append("fabric_segment_missing")
    if not silhouette_seg:
        missing_flags.append("silhouette_segment_missing")
    if not color_seg:
        missing_flags.append("color_segment_missing")

    # --- F18/F19 颜色解析 ---
    f18 = 0
    f19 = 5.0
    color_parse_error = False
    try:
        f18 = _calc_f18(color_seg)
        f19 = _calc_f19(color_seg)
    except Exception as e:
        log.warning("颜色段解析异常: %s", e)
        color_parse_error = True
        f18 = 0
        f19 = 5.0

    if color_parse_error:
        missing_flags.append("color_parse_error")

    # --- 组装结果 ---
    try:
        result: dict[str, Any] = {
            "F11": _calc_f11(text, fabric_seg),
            "F12": _calc_f12(text, fabric_seg),
            "F13": _calc_f13(silhouette_seg),
            "F14": _calc_f14(silhouette_seg),
            "F15": _calc_f15(text, fabric_seg, silhouette_seg),
            "F16": _calc_f16(silhouette_seg),
            "F17": _calc_f17(fabric_seg),
            "F18": f18,
            "F19": f19,
            "F20": _calc_f20(text),
            "missing_flags": missing_flags,
        }
    except Exception as e:
        # 防御兜底：任何特征计算异常 → 返回全默认
        log.error("extract_text_features 计算失败: %s", e)
        missing_flags.append("calc_error")
        return _default_result(missing_flags)

    return result


def _default_result(missing_flags: list[str]) -> dict[str, Any]:
    """全默认值结果（空文本或严重异常时）。

    默认值选择策略：
      - F11/F12/F13/F14/F19 取 5.0 中性值
      - F15/F16/F17/F18/F20 取 0（无信息）
    """
    return {
        "F11": 5.0, "F12": 5.0,
        "F13": 5.0, "F14": 5.0,
        "F15": 0, "F16": 0, "F17": 0,
        "F18": 0, "F19": 5.0,
        "F20": 0,
        "missing_flags": missing_flags,
    }


# ============================================================
# 批量接口
# ============================================================

_F_COLS = ["F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20"]


def batch_extract_text_features(texts: list[Any]) -> pd.DataFrame:
    """批量处理文本特征提取。

    Args:
        texts: 任意可迭代对象，每个元素为一段版型/设计描述文本。

    Returns:
        DataFrame，columns = F11..F20 + missing_flags。
        每行对应输入 texts[i]。
    """
    rows: list[dict[str, Any]] = []
    for t in texts:
        rows.append(extract_text_features(t))
    df = pd.DataFrame(rows)
    # 保证列顺序
    existing = [c for c in _F_COLS if c in df.columns]
    extras = [c for c in df.columns if c not in _F_COLS]
    return df[existing + extras]


# ============================================================
# 直接运行时的简单自检
# ============================================================

if __name__ == "__main__":
    sample = (
        "面料：面层 91.4%锦纶+8.6%聚酯纤维（藏青/紫色）、100%锦纶（芥黄/粉色）；"
        "里料92%聚酯纤维+8%氨纶（抗静电摇粒绒）；塔丝隆面料+特氟龙三防助剂\n"
        "版型：宽松H型连帽户外夹克（冲锋衣）；帽口弹力松紧、加高领口防风、袖口反光标志；"
        "外层防水、内里保暖抓绒\n"
        "颜色：藏青（主推色）/芥黄/粉色/紫色/蓝色/黄色"
    )
    import json

    print("=== sample ===")
    print(sample[:100], "...")
    r = extract_text_features(sample)
    print(json.dumps(r, ensure_ascii=False, indent=2))

    print("\n=== empty ===")
    print(json.dumps(extract_text_features(""), ensure_ascii=False))
    print("\n=== None ===")
    print(json.dumps(extract_text_features(None), ensure_ascii=False))
