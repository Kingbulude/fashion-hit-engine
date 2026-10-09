"""品牌记忆库模块 - MemoryStore

从 calibration 产物（memory_db.parquet + feature_weights.json）加载记忆库，
提供加权相似度搜索接口。

用法：
    store = MemoryStore("brand_profiles/mipo/calibrated")
    result = store.search_neighbors({"F14": 5, "category": "外套", ...}, top_k=5)
    stats = store.get_stats()
    store.update_memory(new_styles_df)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# One-hot 特征 key → 原始列映射
_OH_PREFIX_MAP = {
    "cat": "category",
    "season": "season",
    "year": "year",
}

# boost/penalty 转化为 distance additive offset 时的缩放因子。
# offset 量级 = distance_std / SCALE_FACTOR，保证效应温和、不盖过特征相似度本身。
_OFFSET_SCALE_FACTOR = 6.0


class MemoryStore:
    """品牌记忆库：存储历史款式 + 加权相似度搜索。"""

    def __init__(self, calibrated_dir: str | Path):
        """从 calibration 产物加载记忆库。

        Args:
            calibrated_dir: calibration 产物目录，必须包含
                            memory_db.parquet 和 feature_weights.json。
        """
        calibrated_dir = Path(calibrated_dir)
        self._calibrated_dir = calibrated_dir
        self._parquet_path = calibrated_dir / "memory_db.parquet"
        self._weights_path = calibrated_dir / "feature_weights.json"

        if not self._parquet_path.exists():
            raise FileNotFoundError(f"记忆库 parquet 不存在: {self._parquet_path}")
        if not self._weights_path.exists():
            raise FileNotFoundError(f"feature_weights.json 不存在: {self._weights_path}")

        # 加载记忆库 DataFrame
        self.df = pd.read_parquet(self._parquet_path).reset_index(drop=True)

        # 加载特征权重
        with open(self._weights_path, "r", encoding="utf-8") as f:
            self.feature_weights: dict[str, float] = json.load(f)

        # 分离特征类型
        self._numeric_keys: list[str] = []  # 原始数值列名（含 price_pct → 原列 price）
        self._oh_keys: list[tuple[str, str, Any]] = []  # (onehot_key, raw_col, raw_value)

        for key, w in self.feature_weights.items():
            if w <= 0:
                continue  # 零权重特征跳过
            prefix, sep, suffix = key.partition("_")
            if prefix in _OH_PREFIX_MAP and sep == "_":
                raw_col = _OH_PREFIX_MAP[prefix]
                if prefix == "year":
                    raw_value: Any = int(suffix) if suffix.isdigit() else suffix
                else:
                    raw_value = suffix
                self._oh_keys.append((key, raw_col, raw_value))
            else:
                self._numeric_keys.append(key)

        # 数值特征归一化参数（min-max 到 [0,1]）
        self._norm_stats: dict[str, tuple[float, float]] = {}
        for col in self._numeric_keys:
            raw_col = col if col != "price_pct" else "price"
            if raw_col in self.df.columns:
                vmin = float(self.df[raw_col].min())
                vmax = float(self.df[raw_col].max())
                self._norm_stats[col] = (vmin, vmax)

        # 计算记忆库内部 pairwise distance 的 std，用作 boost/penalty
        # additive offset 的 base scale（保证效应温和、与记忆库数据量级匹配）
        self._distance_scale = self._estimate_distance_scale()

    # ------------------------------------------------------------------ utils

    def _estimate_distance_scale(self) -> float:
        """采样记忆库内若干对款式，估算 weighted distance 的 std。

        该值用作 boost/penalty additive offset 的基准：
            same_category:  distance -= (boost - 1) * scale / OFFSET_SCALE_FACTOR
            cross_category: distance += (1 - penalty) * scale / OFFSET_SCALE_FACTOR

        效应足够温和，不会盖过特征相似度排序本身。
        """
        n = len(self.df)
        if n < 2:
            return 0.1  # 兜底默认值

        # 向量化计算所有行的 feature vector
        row_vecs: list[dict[str, float]] = [
            self._build_row_vector(row) for _, row in self.df.iterrows()
        ]

        # 采样最多 200 对（避免 O(n^2) 过大）
        rng = np.random.default_rng(seed=42)
        n_pairs = min(200, n * (n - 1) // 2)
        dists = []
        tried = set()
        while len(dists) < n_pairs:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in tried:
                continue
            tried.add(pair)
            d, _ = self._weighted_distance(row_vecs[i], row_vecs[j], self.feature_weights)
            if d != float("inf"):
                dists.append(d)

        if not dists:
            return 0.1
        return float(np.std(dists))

    def _normalize_value(self, key: str, value: Any) -> float:
        """把 value 归一化到 [0,1]。缺失/无效值返回 0.5。"""
        if key not in self._norm_stats:
            return 0.5
        vmin, vmax = self._norm_stats[key]
        if vmax - vmin < 1e-9:
            return 0.0
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))

    def _build_query_vector(self, query_features: dict[str, Any]) -> dict[str, float]:
        """把 query_features（原始格式 dict）转换成对齐 feature_weights 的向量。

        数值特征自动归一化到 [0,1]；one-hot 特征展开为 0/1。
        """
        vec: dict[str, float] = {}

        for key in self._numeric_keys:
            raw_key = key if key != "price_pct" else "price"
            if raw_key in query_features:
                vec[key] = self._normalize_value(key, query_features[raw_key])

        for oh_key, raw_col, raw_value in self._oh_keys:
            if raw_col in query_features:
                q_val = query_features[raw_col]
                if raw_col == "year":
                    try:
                        q_val = int(q_val)
                    except (TypeError, ValueError):
                        pass
                vec[oh_key] = 1.0 if q_val == raw_value else 0.0

        return vec

    def _build_row_vector(self, row: pd.Series) -> dict[str, float]:
        """把记忆库一行转换成对齐 feature_weights 的向量。"""
        vec: dict[str, float] = {}

        for key in self._numeric_keys:
            raw_key = key if key != "price_pct" else "price"
            if raw_key in row.index and pd.notna(row[raw_key]):
                vec[key] = self._normalize_value(key, row[raw_key])

        for oh_key, raw_col, raw_value in self._oh_keys:
            if raw_col in row.index and pd.notna(row[raw_col]):
                r_val = row[raw_col]
                if raw_col == "year":
                    try:
                        r_val = int(r_val)
                    except (TypeError, ValueError):
                        pass
                vec[oh_key] = 1.0 if r_val == raw_value else 0.0

        return vec

    @staticmethod
    def _weighted_distance(v1: dict[str, float], v2: dict[str, float],
                           weights: dict[str, float]) -> tuple[float, float]:
        """计算两个向量间的加权欧氏距离（按有效权重归一化）。

        Returns:
            (distance, effective_weight_sum)
        """
        common_keys = set(v1.keys()) & set(v2.keys())
        if not common_keys:
            return float("inf"), 0.0

        total_w = 0.0
        sq_sum = 0.0
        for k in common_keys:
            w = weights.get(k, 0.0)
            if w <= 0:
                continue
            diff = v1.get(k, 0.0) - v2.get(k, 0.0)
            sq_sum += w * diff * diff
            total_w += w

        if total_w < 1e-12:
            return float("inf"), 0.0

        distance = np.sqrt(sq_sum / total_w)
        return float(distance), float(total_w)

    # ----------------------------------------------------------- public API

    def search_neighbors(
        self,
        query_features: dict[str, Any],
        top_k: int = 5,
        same_category_boost: float = 2.0,
        cross_category_penalty: float = 0.5,
    ) -> dict[str, Any]:
        """搜索与 query_features 最相似的 top-K 历史款式。

        相似度计算：
        1. 数值特征 + one-hot 特征 → 加权欧氏距离（权重来自 feature_weights）
        2. same_category_boost / cross_category_penalty 通过 distance 的加法偏移
           实现（而非直接乘 similarity），保证 similarity ∈ (0, 1] 且效应温和：
               same_category:  distance -= (boost - 1) * distance_scale / SCALE
               cross_category: distance += (1 - penalty) * distance_scale / SCALE
        3. similarity = 1 / (1 + adjusted_distance)
        4. 自动排除 query 自己（若 style_id 在记忆库中存在）

        Args:
            query_features: 新款式特征 dict，键可为原始列名
                            （'F11'~'F20', 'category', 'season', 'year', 'price' 等）。
            top_k: 返回邻居数量。
            same_category_boost: 同品类相似度提升因子（≥ 1.0）。
            cross_category_penalty: 不同品类相似度降低因子（≤ 1.0）。

        Returns:
            dict，含 neighbors 列表和汇总统计。
        """
        query_style_id = query_features.get("style_id")
        query_category = query_features.get("category")
        has_query_category = query_category is not None

        q_vec = self._build_query_vector(query_features)

        # boost/penalty → distance additive offset（温和效应）
        offset_base = self._distance_scale / _OFFSET_SCALE_FACTOR
        same_offset = -(same_category_boost - 1.0) * offset_base  # 减小距离 → 增大相似
        cross_offset = (1.0 - cross_category_penalty) * offset_base  # 增大距离 → 减小相似

        rows: list[dict[str, Any]] = []

        for idx, row in self.df.iterrows():
            if query_style_id is not None and row.get("style_id") == query_style_id:
                continue

            r_vec = self._build_row_vector(row)
            dist, eff_w = self._weighted_distance(q_vec, r_vec, self.feature_weights)

            if dist == float("inf"):
                rows.append({
                    "idx": idx,
                    "style_id": row.get("style_id"),
                    "category": row.get("category"),
                    "is_bomb": bool(row.get("is_bomb", False)),
                    "similarity": 0.0,
                    "sales_num": int(row.get("sales_num", 0)),
                    "label_30d": row.get("label_30d"),
                    "internal_review_grade": row.get("internal_review_grade"),
                })
                continue

            # category-aware distance offset
            row_category = row.get("category")
            if has_query_category and row_category == query_category:
                adj_dist = dist + same_offset
            elif has_query_category:
                adj_dist = dist + cross_offset
            else:
                adj_dist = dist

            # distance 下界保护，避免极端 offset 造成负距离
            adj_dist = max(1e-4, adj_dist)
            sim = 1.0 / (1.0 + adj_dist)

            rows.append({
                "idx": idx,
                "style_id": row.get("style_id"),
                "category": row.get("category"),
                "is_bomb": bool(row.get("is_bomb", False)),
                "similarity": sim,
                "sales_num": int(row.get("sales_num", 0)),
                "label_30d": row.get("label_30d"),
                "internal_review_grade": row.get("internal_review_grade"),
            })

        rows.sort(key=lambda r: r["similarity"], reverse=True)
        top = rows[:top_k]

        neighbors = []
        for r in top:
            neighbors.append({
                "style_id": r["style_id"],
                "category": r["category"],
                "is_bomb": r["is_bomb"],
                "similarity": round(r["similarity"], 4),
                "sales_num": r["sales_num"],
                "label_30d": r["label_30d"],
                "internal_review_grade": r["internal_review_grade"],
            })

        n = len(top)
        bomb_rate = sum(1 for r in top if r["is_bomb"]) / n if n else 0.0
        avg_sales = float(np.mean([r["sales_num"] for r in top])) if n else 0.0
        if has_query_category:
            cat_consistency = sum(
                1 for r in top if r["category"] == query_category
            ) / n if n else 0.0
        else:
            cat_consistency = None

        return {
            "neighbors": neighbors,
            "neighbor_bomb_rate": round(bomb_rate, 4),
            "neighbor_avg_sales": round(avg_sales, 2),
            "neighbor_category_consistency": round(cat_consistency, 4)
                if cat_consistency is not None else None,
        }

    # ----------------------------------------------------------- get_stats

    def get_stats(self) -> dict[str, Any]:
        """记忆库统计概览。"""
        df = self.df
        total = len(df)
        bombs = int(df["is_bomb"].sum()) if "is_bomb" in df.columns else 0
        p_styles = int(df["is_p_style"].sum()) if "is_p_style" in df.columns else 0

        categories: dict[str, int] = {}
        if "category" in df.columns:
            categories = {k: int(v) for k, v in df["category"].value_counts().items()}

        return {
            "total_styles": total,
            "bombs": bombs,
            "p_styles": p_styles,
            "bomb_rate": round(bombs / total, 4) if total else 0.0,
            "categories": categories,
        }

    # ------------------------------------------------------- update_memory

    def update_memory(self, new_styles_df: pd.DataFrame) -> None:
        """增量更新记忆库并保存回 parquet。

        Args:
            new_styles_df: 新款式 DataFrame，列需要与 memory_db.parquet 对齐。
        """
        if not isinstance(new_styles_df, pd.DataFrame) or new_styles_df.empty:
            return

        # 对齐列（缺失列用 NaN 填充）
        for col in self.df.columns:
            if col not in new_styles_df.columns:
                new_styles_df[col] = np.nan

        # 去重：以 style_id 为准，新数据覆盖旧数据
        merged = pd.concat([self.df, new_styles_df[self.df.columns]], ignore_index=True)
        if "style_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["style_id"], keep="last")
        merged = merged.reset_index(drop=True)

        self.df = merged

        # 更新数值特征归一化参数
        for col in self._numeric_keys:
            raw_key = col if col != "price_pct" else "price"
            if raw_key in self.df.columns:
                vmin = float(self.df[raw_key].min())
                vmax = float(self.df[raw_key].max())
                self._norm_stats[col] = (vmin, vmax)

        # 重算 distance scale
        self._distance_scale = self._estimate_distance_scale()

        # 保存回 parquet
        self.df.to_parquet(self._parquet_path, index=False)

    # -------------------------------------------------------------- private

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MemoryStore(total={stats['total_styles']}, bombs={stats['bombs']}, "
            f"categories={len(stats['categories'])})"
        )
