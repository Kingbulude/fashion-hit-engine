"""数据IO层：读取 styles.xlsx / sales.xlsx，匹配图片文件
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from .config import load_brand_profile
from .types import BrandConfig, StyleInfo, safe_float

log = logging.getLogger(__name__)


def resolve_category(
    raw_category_name: str | None,
    brand_cfg: BrandConfig | None = None,
) -> str:
    """解析品类名 → 标准化品类ID或原名。

    匹配顺序：
      1) brand_cfg.category_registry["category_aliases"] 映射（别名→ID）
      2) category_registry.categories[].name 精确匹配 → ID
      3) category_registry.categories[].name 模糊包含 → ID
      4) 未识别：Warning + 返回 "_unknown"（若 raw_category_name 为空/None 则返回空字符串）

    Args:
        raw_category_name: Excel 原始品类名（可能是别名，如"外套"、"短袖"等）
        brand_cfg: 品牌配置（None时默认 tongzhuang-outdoor）

    Returns:
        category_id（标准化）或 "_unknown" 或 空字符串。
    """
    raw = (raw_category_name or "").strip()
    if not raw:
        return ""

    if brand_cfg is None:
        brand_cfg = load_brand_profile("tongzhuang-outdoor")
    registry = brand_cfg.category_registry or {}
    aliases: dict[str, Any] = registry.get("category_aliases", {}) or {}
    categories: list[dict[str, Any]] = registry.get("categories", []) or []

    # 1) 别名精确匹配 → ID
    if raw in aliases:
        return str(aliases[raw])

    # 2) categories[].name 精确匹配 → ID
    for cat in categories:
        if str(cat.get("name", "")).strip() == raw:
            cid = cat.get("id")
            if cid is not None:
                return str(cid)

    # 3) 模糊包含匹配
    for cat in categories:
        cn = str(cat.get("name", "")).strip()
        if cn and (cn in raw or raw in cn):
            cid = cat.get("id")
            if cid is not None:
                return str(cid)

    # 4) 都不认识 → Warning + _unknown
    warnings.warn(
        f"[resolve_category] 未识别品类名 '{raw}'，回退为 '_unknown'。"
        f"请在 brand_profiles/{brand_cfg.brand_id}/category_registry.yaml 的 "
        f"category_aliases 中补充映射。",
        UserWarning,
        stacklevel=2,
    )
    return "_unknown"


def _match_images(style_id: str, images_dir: Path) -> list[Path]:
    """模糊匹配图片文件：匹配 style_id 文件名前缀"""
    if not images_dir.exists():
        return []
    candidates = []
    sid = style_id.lower()
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        name = p.stem.lower()
        if sid in name or name.startswith(sid) or sid.startswith(name.replace("_", "")):
            candidates.append(p)
    return candidates


def read_styles_excel(
    styles_xlsx: Path,
    images_dir: Path,
    *,
    style_id_col: str = "款号",
    category_col: str | None = "品类",
    price_col: str | None = "售价",
    fab_col: str | None = "FAB描述",
    season_col: str | None = "季节",
    manual_grade_col: str | None = "内审分级",
    sales_col: str | None = "累计销量",
    sell_through_col: str | None = "售罄率",
    main_push_col: str | None = "是否主推",
    live_col: str | None = "是否直播重点",
    brand_cfg: BrandConfig | None = None,
) -> list[StyleInfo]:
    """从 styles.xlsx 读取所有款式信息（v2.0 支持 BrandConfig 品类解析）

    列名找不到时自动忽略该字段。category_col=None 表示没有品类列。

    新特性：若传 brand_cfg，则通过 resolve_category() 把原始品类名标准化为 category_id，
    同时保留原始名作为 category 字段的一部分（兼容旧代码）。
    """
    import pandas as pd

    if not styles_xlsx.exists():
        raise FileNotFoundError(f"款式表不存在：{styles_xlsx}")
    df = pd.read_excel(styles_xlsx)
    df.columns = [str(c).strip() for c in df.columns]

    styles: list[StyleInfo] = []
    for _, row in df.iterrows():
        def _get(col: str | None) -> Any:
            if col is None:
                return None
            if col not in df.columns:
                return None
            v = row[col]
            if pd.isna(v):
                return None
            return v

        sid_val = _get(style_id_col)
        if sid_val is None or str(sid_val).strip() == "":
            continue
        style_id = str(sid_val).strip()

        imgs = _match_images(style_id, images_dir)
        if not imgs:
            log.warning("款%s 未匹配到图片（请检查 %s 目录）", style_id, images_dir)

        grade_val = _get(manual_grade_col)
        if isinstance(grade_val, str):
            grade = grade_val.strip()
        elif grade_val is not None:
            grade = str(grade_val)
        else:
            grade = ""

        def _bool(v: Any) -> bool:
            if v is None: return False
            s = str(v).strip().lower()
            return s in {"1", "true", "yes", "是", "y", "t", "主推", "重点"}

        price = safe_float(_get(price_col), 0.0)
        sales = int(safe_float(_get(sales_col), 0.0))
        st = _get(sell_through_col)
        if isinstance(st, str) and st.endswith("%"):
            sell_through = safe_float(st.rstrip("%"), 0.0) / 100.0
        else:
            sell_through = safe_float(st, 0.0)

        # 品类：如果有 brand_cfg，就调用 resolve_category 标准化
        raw_cat_val = _get(category_col)
        raw_cat = (raw_cat_val or "").strip() if raw_cat_val is not None else ""
        if brand_cfg is not None and raw_cat:
            resolved_id = resolve_category(raw_cat, brand_cfg=brand_cfg)
            if resolved_id != "_unknown" and resolved_id != "":
                # 标准化后以ID为主，保留原始名在 category 字段
                category_field = resolved_id
            else:
                category_field = raw_cat  # 不识别时保留原始名，避免旧逻辑受影响
        else:
            category_field = raw_cat

        styles.append(StyleInfo(
            style_id=style_id,
            images=imgs,
            fab_description=(_get(fab_col) or "").strip(),
            category=category_field,
            price=price,
            manual_grade=grade,
            sales_qty=sales,
            sell_through_pct=sell_through,
            season=(_get(season_col) or "").strip(),
            is_main_push=_bool(_get(main_push_col)),
            is_live_stream=_bool(_get(live_col)),
        ))
    return styles


def load_styles_from_excel(
    styles_xlsx: Path,
    images_dir: Path,
    *,
    brand_cfg: BrandConfig | None = None,
    **kwargs: Any,
) -> list[StyleInfo]:
    """v2.0 规范命名的别名：等同于 read_styles_excel(..., brand_cfg=brand_cfg)"""
    return read_styles_excel(
        styles_xlsx, images_dir, brand_cfg=brand_cfg, **kwargs,
    )


def save_predictions_xlsx(
    predictions: list,
    out_path: Path,
) -> Path:
    """把批量预测结果导出为Excel（两Sheet：总表 / 逐款明细）"""
    import pandas as pd

    # Sheet 1：总表（每款一行，关键指标）
    summary_rows = [p.to_flat_dict() for p in predictions]
    df_summary = pd.DataFrame(summary_rows)

    # Sheet 2：逐款明细
    detail_rows = []
    for p in predictions:
        row = {
            "style_id": p.info.style_id,
            "grade": p.grade.grade,
            "final_score": p.grade.final_score,
            "confidence": p.grade.confidence,
            "recommended_channel": p.grade.recommended_channel,
            "consumer_insights": p.grade.consumer_insights,
            "strengths": "｜".join(p.grade.strengths),
            "weaknesses": "｜".join(p.grade.weaknesses),
            "improvements": "｜".join(p.grade.improvements),
        }
        detail_rows.append(row)
    df_detail = pd.DataFrame(detail_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="总表", index=False)
        df_detail.to_excel(writer, sheet_name="逐款明细", index=False)
    return out_path
