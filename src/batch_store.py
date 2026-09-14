"""跨批次持久化存储。

设计原则：
- 单文件 CSV，零依赖（只用 pandas），方便用户自己打开看
- 存原始预测数值 + 真实销量（如果有），校准权重是派生出来的，不存
- 自动去重：同一 style_id 同一批次只保留最新一次
- 启动时自动加载历史，回测校准时优先用累积所有批次

存储位置：brand_profiles/<brand>/calibrated/batches.csv
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from .types import FullPrediction

log = logging.getLogger(__name__)

# CSV 列 schema（版本化）
_COLUMNS = [
    "batch_id",        # UUID，一批一组
    "batch_name",      # 用户给的批次名，例如 "2026春第一批"
    "timestamp",       # ISO 时间
    "style_id",
    "category",
    "season",
    "price",
    "grade",           # S/A+/A/P
    "final_score",     # 0-100
    "confidence",
    "recommended_channel",
    "natural_score",   # 0-10
    "live_score",      # 0-10
    "perceived_value", # 0-10
    "value_match",     # -2 ~ +2
    "price_risk",
    "voting_weighted", # 0-10
    "voting_std",
    "opposition_rate",
    "support_rate",
    "sales_qty",       # 真实销量（如果有）
    "manual_grade",    # 人工分级（如果有）
    "sell_through_pct",
    "llm_backend",     # mock / aliyun_dashscope
]


def _batches_csv_path(calibrated_dir: Path | str) -> Path:
    """把 calibrated 目录转成 batches.csv 路径。"""
    calibrated_dir = Path(calibrated_dir) if calibrated_dir else Path(".")
    return calibrated_dir / "batches.csv"


def load_batches(calibrated_dir: Path | str) -> pd.DataFrame:
    """加载历史所有批次预测结果。空目录返回空 DataFrame。"""
    path = _batches_csv_path(calibrated_dir)
    if not path.exists():
        log.info("batches.csv 不存在，首次运行: %s", path)
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(path, dtype={"style_id": str})
        log.info("📥 加载历史批次: %d 条预测, %d 款有销量",
                 len(df), int(df["sales_qty"].notna().sum()) if "sales_qty" in df.columns else 0)
        return df
    except Exception as e:
        log.warning("加载 batches.csv 失败: %s，返回空", e)
        return pd.DataFrame(columns=_COLUMNS)


def append_batch(
    calibrated_dir: Path | str,
    predictions: list[FullPrediction],
    batch_name: str = "",
    llm_backend: str = "mock",
    batch_id: str | None = None,
) -> str:
    """把一批预测结果追加到 batches.csv。返回 batch_id。"""
    path = _batches_csv_path(calibrated_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    bid = batch_id or uuid.uuid4().hex[:8]
    ts = datetime.now().isoformat(timespec="seconds")
    rows = []
    for p in predictions:
        rows.append({
            "batch_id": bid,
            "batch_name": batch_name,
            "timestamp": ts,
            "style_id": p.info.style_id,
            "category": p.info.category or "",
            "season": p.info.season or "",
            "price": p.info.price,
            "grade": p.grade.grade,
            "final_score": p.grade.final_score,
            "confidence": p.grade.confidence,
            "recommended_channel": p.grade.recommended_channel or "",
            "natural_score": p.channels.natural_score,
            "live_score": p.channels.live_score,
            "perceived_value": p.channels.perceived_value,
            "value_match": p.channels.value_match,
            "price_risk": p.channels.price_risk,
            "voting_weighted": p.voting.weighted_score,
            "voting_std": p.voting.score_std,
            "opposition_rate": p.voting.opposition_rate,
            "support_rate": p.voting.support_rate,
            "sales_qty": p.info.sales_qty if p.info.sales_qty > 0 else None,
            "manual_grade": p.info.manual_grade or "",
            "sell_through_pct": p.info.sell_through_pct,
            "llm_backend": llm_backend,
        })

    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    # 去重：同一 style_id + 同一 batch_id 只留最新
    if path.exists():
        try:
            old_df = pd.read_csv(path, dtype={"style_id": str})
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["batch_id", "style_id"], keep="last"
            )
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    log.info("📤 批次 %s 已保存: %d 款 → %s (累计 %d 条)",
             bid, len(predictions), path, len(combined))
    return bid


def get_cumulative_sales_lookup(
    calibrated_dir: Path | str,
) -> dict[str, float]:
    """从历史所有批次里合并出 style_id → 销量 的映射。

    同一 style_id 可能出现在多个批次（不同季节、不同年份），
    用 最新一次的销量 覆盖之前的。
    """
    df = load_batches(calibrated_dir)
    if df.empty or "sales_qty" not in df.columns:
        return {}
    df = df[df["sales_qty"].notna()].copy()
    if df.empty:
        return {}
    df = df.sort_values("timestamp")
    # 每个 style_id 保留最新一条
    latest = df.drop_duplicates(subset="style_id", keep="last")
    return {
        row.style_id: float(row.sales_qty)
        for row in latest.itertuples(index=False)
    }


def get_cumulative_predictions(
    calibrated_dir: Path | str,
    llm_backend_filter: str | None = None,
) -> pd.DataFrame:
    """返回所有历史预测，可选按 LLM backend 过滤。"""
    df = load_batches(calibrated_dir)
    if df.empty:
        return df
    if llm_backend_filter and "llm_backend" in df.columns:
        df = df[df["llm_backend"] == llm_backend_filter]
    return df
