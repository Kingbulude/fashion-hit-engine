"""从 history.db 导入历史预测数据 → 3Loop calibration.

用法：
    python scripts/calibrate_from_history_db.py .uploads/xxx_history.db
    python scripts/calibrate_from_history_db.py .uploads/xxx_history.db --brand mipo
    python scripts/calibrate_from_history_db.py .uploads/xxx_history.db --auto-apply

输入：SQLite predictions 表（app.py 预测时自动写入的 history.db）
输出：brand_profiles/<brand>/calibrated/loop{1,2,3}_*.yaml + history_accumulated.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.optimization_kernel import run_all_loops

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate_db")


# F01-F10 特征 key 映射（DB 里存的是全名，loop1 需要 F01~F10）
FEATURE_KEY_MAP = [
    ("F01", "silhouette"),        # 廓形
    ("F02", "clean_look"),        # 整洁感
    ("F03", "color_risk"),        # 颜色风险
    ("F04", "function_visibility"), # 功能可见度
    ("F05", "photogenic"),        # 上镜性
    ("F06", "wearability"),       # 穿着场合
    ("F07", "pairing"),           # 搭配性
    ("F08", "fabric_perception"), # 面料感知
    ("F09", "brand_tone"),        # 品牌调性匹配
    ("F10", "uniqueness"),        # 独特性
]

GRADE_MAP = {
    "S": 4, "S款": 4,
    "A+": 3,
    "A": 2, "A款": 2,
    "P": 0, "P款": 0, "P-": 0,
}


def extract_feature_scores(feature_json: str | None) -> dict[str, float]:
    """从 DB feature_scores JSON 提取 F01~F10."""
    result = {f"F{i:02d}": 5.0 for i in range(1, 11)}
    if not feature_json:
        return result
    try:
        raw = json.loads(feature_json)
    except (json.JSONDecodeError, TypeError):
        return result
    # 尝试精确匹配
    for short_key, suffix in FEATURE_KEY_MAP:
        # 先试试全名匹配（F01_silhouette 或 silhouette）
        candidates = [
            f"{short_key}_{suffix}",
            suffix,
            short_key,
            f"F{suffix.replace('_', '')}",
        ]
        for c in candidates:
            if c in raw:
                result[short_key] = float(raw[c])
                break
        else:
            # 模糊匹配：任何 key 里含 suffix 的
            for k, v in raw.items():
                if suffix in k.lower() and k.startswith("F"):
                    result[short_key] = float(v)
                    break
    return result


def synthesize_persona_scores(
    weighted_score: float,
    support_rate: float | None = None,
    opposition_rate: float | None = None,
    std_dev: float | None = None,
    n_personas: int = 30,
) -> dict[str, float]:
    """DB 里没有存每一个 P01~P30 的分，用加权分 + 支持率合成。

    策略：在 weighted_score 基础上加随机扰动，支持率高的款扰动小（共识强），
    反对率高的款扰动大（争议大），std_dev 尽量用 DB 里的真实值。
    """
    base = float(weighted_score or 5.0)
    support = float(support_rate or 0.5)
    oppose = float(opposition_rate or 0.0)
    std = float(std_dev or 0.8 * (1.0 - support))  # 没 std 就用 support_rate 反推

    # 种子用 base 的 round 避免每次结果不一样
    rng = np.random.default_rng(seed=int(round(base * 100)) % 2**31)
    scores = rng.normal(loc=base, scale=std, size=n_personas)
    scores = np.clip(scores, 1.0, 10.0)

    # 支持率高 → 更多人设给 >= 6 分
    if support > 0.6:
        n_high = int(support * n_personas)
        scores[:n_high] = np.clip(base + rng.normal(0, std * 0.3, n_high), 6.0, 10.0)

    return {f"P{i+1:02d}": round(float(scores[i]), 3) for i in range(n_personas)}


def db_to_history_df(db_path: str | Path, brand: str = "mipo") -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM predictions WHERE brand_id = ?", (brand,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    log.info("📥 DB 加载: %d 条 brand=%s 记录", len(rows), brand)

    records = []
    n_labeled = 0
    for r in rows:
        sales = float(r.get("sales_qty") or 0.0)
        mg = (r.get("manual_grade") or "").strip().upper()

        # 特征分
        feat = extract_feature_scores(r.get("feature_scores"))

        # 人设分（合成）
        personas = synthesize_persona_scores(
            weighted_score=r.get("voting_weighted_score"),
            support_rate=r.get("voting_support_rate"),
            opposition_rate=r.get("voting_opposition_rate"),
            std_dev=r.get("voting_score_std"),
        )

        # engine 分（DB 里已经存了聚合值）
        persona_score = float(r.get("voting_weighted_score") or 5.0)
        channel_score = float(r.get("natural_score") or 5.0)  # 近似
        price_value = float(r.get("perceived_value") or 5.0)

        # persona_top8 / support_rate / oppose_rate / divergence / extreme_gap
        p_scores = list(personas.values())
        p_top8 = float(np.mean(np.sort(p_scores)[-8:]))
        p_support = float(r.get("voting_support_rate") or 0.5)
        p_oppose = float(r.get("voting_opposition_rate") or 0.0)
        p_div = float(np.std(p_scores))
        p_gap = float(max(p_scores) - min(p_scores))

        record = {
            "style_id": r.get("style_id", ""),
            **feat,                                          # F01~F10
            **personas,                                      # P01~P30
            "persona_score": persona_score,
            "persona_top8": p_top8,
            "persona_support_rate": p_support,
            "persona_oppose_rate": p_oppose,
            "persona_divergence": p_div,
            "persona_extreme_gap": p_gap,
            "channel_score": channel_score,
            "price_value_score": price_value,
            "natural_score": float(r.get("natural_score") or 5.0),
            "live_score": float(r.get("live_score") or 5.0),
            "is_main_push": int(r.get("is_main_push") or 0),
            "is_live_stream": int(r.get("is_live_stream") or 0),
            "sales": sales,
        }

        # grade_num + grade_norm
        if mg in GRADE_MAP:
            record["grade_num"] = GRADE_MAP[mg]
            record["grade_norm"] = GRADE_MAP[mg] / 4.0 * 100.0
            n_labeled += 1
        elif sales > 0:
            # 有销量但没手动评级 — 用销量推断
            if sales >= 1000:
                record["grade_num"] = 4  # S
                record["grade_norm"] = 100.0
            elif sales >= 20:
                record["grade_num"] = 2  # A
                record["grade_norm"] = 50.0
            else:
                record["grade_num"] = 0  # P
                record["grade_norm"] = 0.0
        else:
            record["grade_num"] = np.nan
            record["grade_norm"] = np.nan

        records.append(record)

    df = pd.DataFrame(records)
    log.info("📊 history_df: %d 款（带 label=%d, 有 sales>0=%d）",
             len(df), n_labeled, int((df["sales"] > 0).sum()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="history.db 文件路径")
    ap.add_argument("--brand", default="mipo")
    ap.add_argument("--auto-apply", action="store_true",
                    help="直接应用校准产物到 calibrated_dir（不经过 _pending 审核门）")
    ap.add_argument("--only-labeled", action="store_true",
                    help="只用有 manual_grade 的款做校准")
    ap.add_argument("--calibrated-dir", default=None,
                    help="覆盖默认输出目录 brand_profiles/<brand>/calibrated/")
    args = ap.parse_args()

    df = db_to_history_df(args.db, brand=args.brand)

    if args.only_labeled:
        before = len(df)
        df = df[df["grade_norm"].notna() & (df["sales"] > 0)].copy()
        log.info("🎯 --only-labeled: %d → %d 款（有 grade + 有 sales）", before, len(df))

    # 过滤至少要有 sales
    df_calib = df[df["sales"] > 0].copy()
    log.info("🎯 用于 calibration: %d 款（有 sales>0）", len(df_calib))

    if len(df_calib) < 10:
        log.error("❌ 用于 calibration 的样本不足 10 款（只有 %d），放弃。", len(df_calib))
        sys.exit(1)

    # 校准目录
    if args.calibrated_dir:
        cal_dir = Path(args.calibrated_dir)
    else:
        cal_dir = Path("brand_profiles") / args.brand / "calibrated"
    cal_dir = cal_dir.resolve()
    log.info("📂 校准产物输出到: %s", cal_dir)

    # 先备份旧产物
    import shutil, time
    if cal_dir.exists():
        bak = cal_dir.parent / f"calibrated_bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(cal_dir, bak)
        log.info("💾 旧 calibrated 备份 → %s", bak)

    # 累积 CSV 也更新（merge 方式，保留旧数据）
    acc_csv = cal_dir / "history_accumulated.csv"
    cal_dir.mkdir(parents=True, exist_ok=True)
    if acc_csv.exists():
        old = pd.read_csv(acc_csv)
        # 用 style_id merge，新数据覆盖旧数据
        df_idx = df.set_index("style_id")
        old_idx = old.set_index("style_id")
        merged = df_idx.combine_first(old_idx).reset_index()
        log.info("📦 累积 CSV: 旧 %d + 新 %d → %d", len(old), len(df), len(merged))
    else:
        merged = df
    merged.to_csv(acc_csv, index=False)
    log.info("📝 history_accumulated.csv → %s (%d 行)", acc_csv, len(merged))

    # run_all_loops 需要 history_df 的 sales 列 + 所有必要特征
    class _BrandCfg:
        calibrated_dir = str(cal_dir)
        has_internal_review = True

    result = run_all_loops(
        brand_cfg=_BrandCfg(),
        history_df=df_calib,
        prediction_artifacts_dir=cal_dir.parent / "output",
        sales_col="sales",
        auto_apply=args.auto_apply,
    )

    log.info("=== 3Loop 结果 ===")
    for r in (result.loop1, result.loop2, result.loop3):
        if r:
            log.info("  [%s] Spearman: %.3f → %.3f (Δ%.3f), F1: %.3f → %.3f",
                     r.name if hasattr(r, "name") else "?",
                     getattr(r, "old_spearman", 0), getattr(r, "new_spearman", 0),
                     getattr(r, "delta_spearman", 0),
                     getattr(r, "old_f1", 0), getattr(r, "new_f1", 0))
    log.info("📁 产物文件: %s", result.output_files)


if __name__ == "__main__":
    main()
