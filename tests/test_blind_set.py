"""盲测组（blind set）单元测试。

验证（见 src/blind_set.py 领域定义）：
  1. 分层抽样：品类结构一致、确定性（同 seed 同结果）、比例正确、小层保底
  2. 对照分析：盲测组纯净 Spearman vs 对照组
  3. AI vs 人工分歧裁决：按销量前/后 1/3 裁决谁对
  4. 样本不足时不出结论（ready=False）
  5. 历史库 is_blind 持久化 + 旧库迁移
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.blind_set import compare_blind_vs_control, select_blind_set  # noqa: E402
from src.types import FullPrediction, GradeResult, StyleInfo  # noqa: E402


def _mk_styles(n_per_cat: int = 10, cats: tuple[str, ...] = ("外套", "裤装")) -> list[StyleInfo]:
    return [
        StyleInfo(style_id=f"{cat}-{i:02d}", category=cat, price=199.0)
        for cat in cats for i in range(n_per_cat)
    ]


def test_select_blind_set_deterministic():
    """同 seed 同结果（批次名做 seed，可复现）"""
    styles = _mk_styles()
    a = select_blind_set(styles, ratio=0.2, seed="2026春第一批")
    b = select_blind_set(styles, ratio=0.2, seed="2026春第一批")
    c = select_blind_set(styles, ratio=0.2, seed="2026春第二批")
    assert a == b
    assert a != c  # 不同批次不同抽样
    assert 0 < len(a) <= len(styles)


def test_select_blind_set_stratified_ratio():
    """分层抽样：每层约 20%，盲测组品类结构与整批一致"""
    styles = _mk_styles(n_per_cat=10, cats=("外套", "裤装"))
    blind = select_blind_set(styles, ratio=0.2, seed="s")
    n_coat = sum(1 for s in blind if s.startswith("外套"))
    n_pants = sum(1 for s in blind if s.startswith("裤装"))
    assert n_coat == 2 and n_pants == 2  # 10×0.2 每层


def test_select_blind_set_small_stratum_floor():
    """小品类保底：层内期望≥0.5款时至少保 1 款，避免小品类永远进不了盲测"""
    styles = _mk_styles(n_per_cat=10, cats=("外套",))
    styles += [StyleInfo(style_id=f"配饰-0{i}", category="配饰", price=99.0)
               for i in range(3)]  # 3×0.2=0.6 ≥ 0.5 → 保底 1 款
    blind = select_blind_set(styles, ratio=0.2, seed="s")
    assert any(s.startswith("配饰") for s in blind)


def test_select_blind_set_zero_and_empty():
    """ratio=0 / 空列表 → 空集"""
    assert select_blind_set(_mk_styles(), ratio=0, seed="s") == set()
    assert select_blind_set([], ratio=0.2, seed="s") == set()


def _mk_hist_df() -> pd.DataFrame:
    """构造 10 款（5盲测+5对照）历史数据：分数与销量强相关 + 2 个 AI/人工分歧"""
    rows = []
    # 盲测组：AI 分数与销量完全同序
    blind_sales = [100, 80, 60, 40, 20]
    blind_scores = [9.0, 8.0, 6.0, 4.0, 3.0]
    blind_grades = ["S", "A+", "A", "P", "P"]
    human_grades = ["S", "A+", "A", "P", "S"]  # 末款：人工S AI=P → 实际卖得差 → AI对
    for i in range(5):
        rows.append({"style_id": f"B{i}", "is_blind": 1, "final_score": blind_scores[i],
                     "final_grade": blind_grades[i], "manual_grade": human_grades[i],
                     "sales_qty": blind_sales[i], "batch_id": "t", "created_at": ""})
    # 首款再加一个分歧：AI S 人工 P → 实际卖得好 → AI对
    rows[0]["manual_grade"] = "P"
    # 对照组：分数与销量弱相关
    ctrl_sales = [90, 70, 50, 30, 10]
    ctrl_scores = [7.0, 9.0, 5.0, 8.0, 4.0]
    for i in range(5):
        rows.append({"style_id": f"C{i}", "is_blind": 0, "final_score": ctrl_scores[i],
                     "final_grade": "A", "manual_grade": "A",
                     "sales_qty": ctrl_sales[i], "batch_id": "t", "created_at": ""})
    return pd.DataFrame(rows)


def test_compare_blind_vs_control():
    """盲测组强相关 Spearman=1.0；对照组弱；分歧裁决 AI 胜出"""
    df = _mk_hist_df()
    r = compare_blind_vs_control(df)
    assert r.n_blind == 5 and r.n_control == 5
    assert r.blind_spearman is not None and r.blind_spearman > 0.99
    assert r.control_spearman is not None and r.control_spearman < r.blind_spearman
    assert r.ready
    # 两个分歧款都裁决为 AI 对（B0: AI-S vs 人工-P 卖最好; B4: AI-P vs 人工-S 卖最差）
    assert r.ai_wins == 2 and r.human_wins == 0
    assert len(r.disputes) == 2


def test_compare_insufficient_blind_sample():
    """盲测有销量款不足 5 个 → 不出 Spearman 结论"""
    df = _mk_hist_df()
    df = df[df["style_id"].isin(["B0", "B1", "C0", "C1"])]  # 仅2个盲测款
    r = compare_blind_vs_control(df)
    assert r.blind_spearman is None
    assert not r.ready
    assert "不足" in r.note or "待" in r.note


def test_compare_sales_not_backfilled():
    """销量未回填（全0）→ 无有效盲测款"""
    df = _mk_hist_df()
    df["sales_qty"] = 0
    r = compare_blind_vs_control(df)
    assert r.n_blind == 0
    assert r.blind_spearman is None


def test_history_store_persists_is_blind():
    """历史库：is_blind 持久化 + 旧库自动迁移补列"""
    from src.history_store import HistoryStore
    from src.types import ChannelScores, StyleFeatures, VotingResult

    with tempfile.TemporaryDirectory() as td:
        store = HistoryStore(db_path=Path(td) / "h.db", csv_dir=Path(td) / "csv")
        # 构造带盲测标记的 FullPrediction
        info_b = StyleInfo(style_id="X1", category="外套", price=199.0, is_blind=True,
                           manual_grade="S", sales_qty=50)
        info_c = StyleInfo(style_id="X2", category="外套", price=199.0, is_blind=False)
        def _pred(info):
            return FullPrediction(
                info=info,
                features=StyleFeatures(style_id=info.style_id),
                voting=VotingResult(style_id=info.style_id),
                channels=ChannelScores(style_id=info.style_id),
                grade=GradeResult(style_id=info.style_id, final_score=8.0,
                                  grade="A+", confidence=0.8,
                                  recommended_channel="自然流量优先"),
            )
        store.append([_pred(info_b), _pred(info_c)], brand_id="mipo", is_mock=True)

        df_cmp = store.load_blind_comparison("mipo", include_mock=True)
        assert set(df_cmp["style_id"]) == {"X1", "X2"}
        blind_flags = dict(zip(df_cmp["style_id"], df_cmp["is_blind"]))
        assert blind_flags["X1"] == 1 and blind_flags["X2"] == 0

        # 旧库迁移：手工删列（模拟旧 schema）→ 重新打开自动补列
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("CREATE TABLE old_pred AS SELECT * FROM predictions")
            conn.execute("DROP TABLE predictions")
            conn.execute("ALTER TABLE old_pred RENAME TO predictions")
            conn.execute(
                "CREATE TABLE pred_no_blind AS "
                "SELECT batch_id, brand_id, style_id, created_at, category, price, "
                "season, is_main_push, is_live_stream, manual_grade, sales_qty, "
                "final_score, final_grade, confidence, recommended_channel, "
                "voting_weighted_score, voting_support_rate, voting_opposition_rate, "
                "voting_score_std, natural_score, live_score, perceived_value, "
                "value_match, feature_scores, feature_confidences, top_buy_reasons, "
                "top_oppose_reasons, strengths, weaknesses, improvements, "
                "consumer_insights, llm_backend, is_mock, notes FROM predictions"
            )
            conn.execute("DROP TABLE predictions")
            conn.execute("ALTER TABLE pred_no_blind RENAME TO predictions")

        store2 = HistoryStore(db_path=store.db_path, csv_dir=store.csv_dir)
        df2 = store2.load_blind_comparison("mipo", include_mock=True)
        assert "is_blind" in df2.columns
        assert (df2["is_blind"] == 0).all()  # 迁移默认0=非盲测


if __name__ == "__main__":
    test_select_blind_set_deterministic()
    test_select_blind_set_stratified_ratio()
    test_select_blind_set_small_stratum_floor()
    test_select_blind_set_zero_and_empty()
    test_compare_blind_vs_control()
    test_compare_insufficient_blind_sample()
    test_compare_sales_not_backfilled()
    test_history_store_persists_is_blind()
    print("✅ 盲测组测试全部通过")
