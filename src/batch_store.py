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
# v1.1 起追加 F01-F10 特征分 + P01-P30 人设分，
# 使 batches.csv 可直接重建 3Loop 校准所需的完整回归样本（跨批次记忆）。
_FEATURE_COLS = [f"F{i:02d}" for i in range(1, 11)]
_PERSONA_COLS = [f"P{i:02d}" for i in range(1, 31)]
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
    *_FEATURE_COLS,    # F01-F10 特征分（3Loop Loop1 回归特征）
    *_PERSONA_COLS,    # P01-P30 人设分（Loop2 回归特征）
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
        # 旧版 CSV 缺 F/P 列 → 补空列，保证下游列访问不炸
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = None
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
        # F01-F10：按 features 字典顺序取前 10 个；与 build_history_df 保持一致
        feat_scores = [f.score for f in list(p.features.features.values())[:10]]
        feat_scores += [None] * (10 - len(feat_scores))
        # P01-P30：votes 的 final_score；不足留空
        p_scores = [v.final_score for v in (p.voting.votes or [])[:30]]
        p_scores += [None] * (30 - len(p_scores))

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
            **dict(zip(_FEATURE_COLS, feat_scores)),
            **dict(zip(_PERSONA_COLS, p_scores)),
        })

    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    # 去重：同一 style_id + 同一 batch_id 只留最新
    if path.exists():
        try:
            old_df = pd.read_csv(path, dtype={"style_id": str})
            # 旧版缺列补齐，避免 concat 后列错位
            for col in _COLUMNS:
                if col not in old_df.columns:
                    old_df[col] = None
            combined = pd.concat(
                [old_df[_COLUMNS], new_df], ignore_index=True
            )
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


def build_cumulative_history_df(
    calibrated_dir: Path | str,
    sales_col: str = "sales",
    min_sales: float = 0.0,
) -> pd.DataFrame:
    """从 batches.csv 重建 3Loop 校准所需的完整 history_df（跨批次记忆核心）。

    输出列与 optimization_kernel.build_history_df 对齐：
        style_id, F01-F10, P01-P30,
        persona_score, channel_score, price_value_score,
        natural_score, live_score, sales

    规则：
    - 只保留 F01 非空的行（v1.1 之后写入的完整行；旧版摘要行无法参与回归）
    - 同一 style_id 跨批次只保留最新一条（时间戳排序后去重）
    - sales 取 sales_qty；<= min_sales 的行保留但销量视为 0（由调用方决定是否过滤）

    Returns:
        DataFrame；无可用历史时返回空 DataFrame（只含列头）。
    """
    out_cols = (
        ["style_id"] + _FEATURE_COLS + _PERSONA_COLS +
        ["persona_score", "channel_score", "price_value_score",
         "natural_score", "live_score", sales_col]
    )
    df = load_batches(calibrated_dir)
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # 只用完整行（有 F01）参与回归；旧版摘要行只有引擎分，不够
    if "F01" not in df.columns:
        return pd.DataFrame(columns=out_cols)
    df = df[df["F01"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # 同一 style_id 跨批次保留最新
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="style_id", keep="last")

    out = pd.DataFrame()
    out["style_id"] = df["style_id"].astype(str)
    for c in _FEATURE_COLS + _PERSONA_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["persona_score"] = pd.to_numeric(df["voting_weighted"], errors="coerce")
    natural = pd.to_numeric(df["natural_score"], errors="coerce")
    live = pd.to_numeric(df["live_score"], errors="coerce")
    out["channel_score"] = (natural + live) / 2
    out["price_value_score"] = pd.to_numeric(df["perceived_value"], errors="coerce")
    out["natural_score"] = natural
    out["live_score"] = live
    out[sales_col] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0.0)
    out.loc[out[sales_col] <= min_sales, sales_col] = 0.0

    # P 列缺失值用 persona_score 兜底（与 build_history_df 一致）
    for c in _PERSONA_COLS:
        out[c] = out[c].fillna(out["persona_score"])
    # F 列缺失用 5.0 兜底（与 build_history_df 一致）
    for c in _FEATURE_COLS:
        out[c] = out[c].fillna(5.0)

    log.info("📚 跨批次历史样本: %d 款（%d 款有销量）",
             len(out), int((out[sales_col] > 0).sum()))
    return out.reset_index(drop=True)
