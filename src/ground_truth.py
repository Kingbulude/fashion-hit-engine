"""Ground truth 模块：从 mipo Excel 映射到爆款二分类标签.

核心逻辑（spec Q14）:
    is_bomb = (最终销量 ≥ 品类阈值) AND (30天标签 ∈ {爆, 旺})
    is_p_style = (内审分级 == 'P')
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_COLUMN_MAP = {
    '款式编号': 'style_id',
    '品类': 'category',
    '内审分级': 'internal_review_grade',
    '上架30天销售情况': 'label_30d',
    '版型/设计描述': 'design_text',
    '季节': 'season',
    '售价': 'price',
    '累计销量结果': 'sales_raw',
    '年份': 'year',
}

_LABEL_ORDINAL = {'爆': 4, '旺': 3, '平': 2, '滞': 1}
_BOMB_LABELS = {'爆', '旺'}


def _parse_sales(val):
    """销量数值化：处理 "9000+" / "25" / "1,200+" 等混合格式."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace('+', '').replace(',', '')
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def load_and_map_ground_truth(excel_path: str,
                              category_thresholds: dict | None = None) -> pd.DataFrame:
    """加载 Excel 并构建 ground truth DataFrame.

    Args:
        excel_path: mipo Excel 文件路径.
        category_thresholds: 可选的品类销量阈值 dict, 如 ``{'T恤': 1500, '长裤': 600}``.
            若为 None, 则自动用品类内 sales_num 的 top 10% quantile (0.9) 计算.

    Returns:
        DataFrame with columns:
            style_id, category, internal_review_grade, label_30d, design_text,
            season, price, year,
            sales_num (int),
            label_ordinal (int),        # 爆=4, 旺=3, 平=2, 滞=1
            category_threshold (int),
            is_bomb (bool),              # 爆款标签
            is_p_style (bool),           # P款
    """
    raw = pd.read_excel(excel_path)

    # 只保留预期中文字段（忽略 Unnamed:5 之类的空列）
    keep_cols = [c for c in _COLUMN_MAP if c in raw.columns]
    df = raw[keep_cols].rename(columns=_COLUMN_MAP)

    # 基础类型清理
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # 销量数值化
    df['sales_num'] = df['sales_raw'].apply(_parse_sales)

    # 标签 ordinal
    df['label_ordinal'] = df['label_30d'].map(_LABEL_ORDINAL)

    # 品类阈值
    if category_thresholds is not None:
        df['category_threshold'] = df['category'].map(category_thresholds)
    else:
        # 自动阈值 = max(品类内 top10% quantile, 全局 top10% quantile)
        # - 品类内 quantile 保留不同品类的销量差异
        # - 全局 quantile 作为 floor，防止小品类阈值过低（如本例卫衣品类内
        #   quantile 仅 1600，而全局 top10% 为 2100，全局 floor 把它抬到 2100）
        global_q90 = df['sales_num'].quantile(0.9)
        df['category_threshold'] = df.groupby('category')['sales_num'].transform(
            lambda x: max(x.quantile(0.9), global_q90)
        )

    # 爆款定义
    df['is_bomb'] = (
        (df['sales_num'] >= df['category_threshold']) &
        (df['label_30d'].isin(_BOMB_LABELS))
    )

    # P款
    df['is_p_style'] = (df['internal_review_grade'] == 'P')

    return df


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

def verify_ground_truth(df: pd.DataFrame) -> dict:
    """返回数据统计摘要, 用于 calibration 报告."""
    summary: dict = {
        'total_styles': len(df),
        'total_bombs': int(df['is_bomb'].sum()),
        'total_p_styles': int(df['is_p_style'].sum()),
        'bomb_rate': float(df['is_bomb'].mean()),
        'categories': df['category'].value_counts().to_dict(),
        'bomb_count_by_category': df.groupby('category')['is_bomb'].sum().to_dict(),
        'label_distribution': df['label_30d'].value_counts().to_dict(),
        'grade_distribution': df['internal_review_grade'].value_counts().to_dict(),
    }

    if df['sales_num'].notna().any():
        s = df['sales_num'].dropna()
        summary['sales_distribution'] = {
            'min': int(s.min()),
            'median': int(s.median()),
            'mean': int(s.mean()),
            'max': int(s.max()),
        }

    summary['actual_bombs'] = (
        df[df['is_bomb']][['style_id', 'category', 'sales_num', 'label_30d']]
        .to_dict('records')
    )

    return summary
