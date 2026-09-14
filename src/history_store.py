"""历史批次持久化：SQLite + CSV 双写，跨批次累积特征分布。

设计原则
- 每跑一次上传/回测批次，自动把预测结果写入：
  1. output/history.db 的 SQLite 表 predictions（主库，方便聚合查询）
  2. output/history/<batch_id>.csv（人类可读，便于导出审计）
- 启动 pipeline 时加载历史数据，累积价格、特征、渠道、人设分分布，
  让后续评估的百分位计算真正反映“品牌历史款式”而不是冷启动空值。
- schema 扁平化：把 FullPrediction 拆成一行一款 + JSON 字段存列表/字典。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .types import FullPrediction


# ========== schema ==========
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    brand_id TEXT NOT NULL,
    style_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- 款式元信息
    category TEXT,
    price REAL,
    season TEXT,
    is_main_push INTEGER,
    is_live_stream INTEGER,
    is_blind INTEGER DEFAULT 0,
    manual_grade TEXT,
    sales_qty REAL,
    -- 评分结果
    final_score REAL,
    final_grade TEXT,
    confidence REAL,
    recommended_channel TEXT,
    -- 三大引擎分
    voting_weighted_score REAL,
    voting_support_rate REAL,
    voting_opposition_rate REAL,
    voting_score_std REAL,
    natural_score REAL,
    live_score REAL,
    perceived_value REAL,
    value_match REAL,
    -- JSON 字段
    feature_scores TEXT,          -- {F01: 7.2, ...}
    feature_confidences TEXT,     -- {F01: 0.85, ...}
    top_buy_reasons TEXT,         -- [str]
    top_oppose_reasons TEXT,      -- [str]
    strengths TEXT,               -- [str]
    weaknesses TEXT,              -- [str]
    improvements TEXT,            -- [str]
    consumer_insights TEXT,
    -- 运行元信息
    llm_backend TEXT,
    is_mock INTEGER,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_brand_created
    ON predictions(brand_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pred_batch
    ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS idx_pred_style
    ON predictions(style_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


class HistoryStore:
    """历史批次持久化入口。

    Args:
        db_path: SQLite 文件路径，默认 output/history.db
        csv_dir: CSV 备份目录，默认 output/history_csv
    """

    def __init__(
        self,
        db_path: Path | str = "output/history.db",
        csv_dir: Path | str = "output/history_csv",
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.csv_dir = Path(csv_dir).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_CREATE_SQL)
            # 迁移：旧库补 is_blind 列（默认0=非盲测）
            cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
            if "is_blind" not in cols:
                conn.execute(
                    "ALTER TABLE predictions ADD COLUMN is_blind INTEGER DEFAULT 0"
                )

    # ---------- 写入 ----------
    def append(
        self,
        predictions: list[FullPrediction],
        *,
        brand_id: str,
        llm_backend: str = "dashscope",
        is_mock: bool = False,
        notes: str = "",
        batch_id: str | None = None,
    ) -> str:
        """写入一批预测结果。返回 batch_id。"""
        batch_id = batch_id or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        created_at = _now()
        rows: list[tuple[Any, ...]] = []
        for p in predictions:
            info = p.info
            rows.append((
                batch_id,
                brand_id,
                info.style_id,
                created_at,
                info.category or "",
                float(info.price or 0),
                info.season or "",
                1 if info.is_main_push else 0,
                1 if info.is_live_stream else 0,
                1 if info.is_blind else 0,
                info.manual_grade or "",
                float(info.sales_qty or 0),
                float(p.grade.final_score),
                p.grade.grade,
                float(p.grade.confidence),
                p.grade.recommended_channel,
                float(p.voting.weighted_score),
                float(p.voting.support_rate),
                float(p.voting.opposition_rate),
                float(p.voting.score_std),
                float(p.channels.natural_score),
                float(p.channels.live_score),
                float(p.channels.perceived_value),
                float(p.channels.value_match),
                _to_json({k: round(v.score, 3) for k, v in p.features.features.items()}),
                _to_json({k: round(v.confidence, 3) for k, v in p.features.features.items()}),
                _to_json(p.voting.top_buy_reasons),
                _to_json(p.voting.top_oppose_reasons),
                _to_json(p.grade.strengths),
                _to_json(p.grade.weaknesses),
                _to_json(p.grade.improvements),
                p.grade.consumer_insights,
                llm_backend,
                1 if is_mock else 0,
                notes,
            ))

        cols = [
            "batch_id", "brand_id", "style_id", "created_at", "category", "price",
            "season", "is_main_push", "is_live_stream", "is_blind",
            "manual_grade", "sales_qty",
            "final_score", "final_grade", "confidence", "recommended_channel",
            "voting_weighted_score", "voting_support_rate", "voting_opposition_rate",
            "voting_score_std", "natural_score", "live_score", "perceived_value",
            "value_match", "feature_scores", "feature_confidences", "top_buy_reasons",
            "top_oppose_reasons", "strengths", "weaknesses", "improvements",
            "consumer_insights", "llm_backend", "is_mock", "notes",
        ]
        insert_sql = f"""
        INSERT INTO predictions ({', '.join(cols)})
        VALUES ({', '.join(['?'] * len(cols))})
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(insert_sql, rows)

        # CSV 备份
        if rows:
            df = pd.DataFrame(rows, columns=cols)
            csv_path = self.csv_dir / f"{batch_id}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return batch_id

    # ---------- 读取 ----------
    def load(
        self,
        brand_id: str | None = None,
        since_days: int | None = None,
        include_mock: bool = True,
    ) -> pd.DataFrame:
        """读取历史记录，返回 DataFrame。"""
        query = "SELECT * FROM predictions WHERE 1=1"
        params: list[Any] = []
        if brand_id:
            query += " AND brand_id = ?"
            params.append(brand_id)
        if since_days is not None:
            cutoff = (datetime.now() - timedelta(days=since_days)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)
        if not include_mock:
            query += " AND is_mock = 0"
        query += " ORDER BY created_at DESC, style_id"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def load_feature_scores(
        self,
        brand_id: str,
        since_days: int | None = None,
        include_mock: bool = False,
    ) -> dict[str, list[float]]:
        """返回每个特征的历史分数列表，用于累积分布/百分位。"""
        df = self.load(brand_id, since_days=since_days, include_mock=include_mock)
        result: dict[str, list[float]] = {}
        for _, row in df.iterrows():
            scores = _from_json(row.get("feature_scores"), {})
            for k, v in scores.items():
                result.setdefault(k, []).append(float(v))
        return result

    def load_prices(self, brand_id: str, since_days: int | None = None) -> list[float]:
        """返回历史价格列表。"""
        df = self.load(brand_id, since_days=since_days, include_mock=False)
        return [float(x) for x in df["price"].dropna().tolist()]

    def load_voting_scores(
        self,
        brand_id: str,
        since_days: int | None = None,
    ) -> dict[str, list[float]]:
        """返回人设/渠道历史分数列表。"""
        df = self.load(brand_id, since_days=since_days, include_mock=False)
        return {
            "voting_weighted_score": [float(x) for x in df["voting_weighted_score"].dropna()],
            "natural_score": [float(x) for x in df["natural_score"].dropna()],
            "live_score": [float(x) for x in df["live_score"].dropna()],
        }

    def load_blind_comparison(
        self,
        brand_id: str,
        include_mock: bool = False,
    ) -> pd.DataFrame:
        """返回盲测对照分析所需列（盲测组 + 对照组合并）。

        列：style_id / is_blind / final_score / final_grade / manual_grade /
            sales_qty / batch_id / created_at
        """
        with sqlite3.connect(self.db_path) as conn:
            query = (
                "SELECT style_id, is_blind, final_score, final_grade, "
                "manual_grade, sales_qty, batch_id, created_at "
                "FROM predictions WHERE brand_id = ?"
            )
            params: list[Any] = [brand_id]
            if not include_mock:
                query += " AND is_mock = 0"
            query += " ORDER BY created_at DESC"
            return pd.read_sql_query(query, conn, params=params)

    def get_summary(self, brand_id: str) -> dict[str, Any]:
        """返回某品牌历史统计摘要。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(DISTINCT batch_id), COUNT(*), MIN(created_at), MAX(created_at) "
                "FROM predictions WHERE brand_id = ?",
                (brand_id,),
            )
            n_batches, n_styles, first_at, last_at = cur.fetchone()
        return {
            "brand_id": brand_id,
            "batches": n_batches or 0,
            "styles": n_styles or 0,
            "first_at": first_at or "",
            "last_at": last_at or "",
            "db_path": str(self.db_path),
        }
