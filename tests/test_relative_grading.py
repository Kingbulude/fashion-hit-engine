"""批次内相对分级（assign_relative_grades）单元测试。

验证（见 src/grading.py 领域定义）：
  1. 分桶：S=Top20%，A+=20-40%，A=40-75%，P=后25%
  2. 幂等：绝对档快照存 metadata['absolute_grade']，重复调用无副作用
  3. 风险款保留绝对判定，不参与分位
  4. 同分并列进同一档（分数边界而非排名边界）
  5. 冷启动场景：分数全部扎堆 65-75（绝对档全 A）→ 相对分级恢复分布
  6. 边界：空批次 / 单款（单款=S）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.grading import assign_relative_grades  # noqa: E402
from src.types import (  # noqa: E402
    ChannelScores,
    FullPrediction,
    GradeResult,
    StyleFeatures,
    StyleInfo,
    VotingResult,
)


def _mk_pred(sid: str, score: float, grade: str = "A") -> FullPrediction:
    """构造最小 FullPrediction：只有 style_id / final_score / 初始档有意义。"""
    info = StyleInfo(style_id=sid, category="外套", price=199.0)
    return FullPrediction(
        info=info,
        features=StyleFeatures(style_id=sid),
        voting=VotingResult(style_id=sid),
        channels=ChannelScores(style_id=sid),
        grade=GradeResult(
            style_id=sid, final_score=score, grade=grade,
            confidence=0.8, recommended_channel="自然流量优先",
        ),
    )


def test_bucket_proportions_20_styles():
    """20 款分散分数（100..81）→ S=4 / A+=4 / A=7 / P=5"""
    preds = [_mk_pred(f"S{i:02d}", 100 - i) for i in range(20)]
    result = assign_relative_grades(preds)
    counts = {g: sum(1 for v in result.values() if v == g) for g in ("S", "A+", "A", "P")}
    assert counts == {"S": 4, "A+": 4, "A": 7, "P": 5}
    # 最高分是 S，最低分是 P
    assert result["S00"] == "S"
    assert result["S19"] == "P"


def test_idempotent_and_snapshot_preserved():
    """重复调用结果不变；绝对档快照不被第二次调用覆盖"""
    preds = [_mk_pred(f"S{i:02d}", 90 - i) for i in range(10)]
    first = assign_relative_grades(preds)
    snapshots = {p.info.style_id: p.metadata["absolute_grade"] for p in preds}

    # 人为污染当前档位，再调用一次 → 从 final_score 重算，恢复一致
    preds[0].grade.grade = "X"
    second = assign_relative_grades(preds)
    assert first == second
    assert preds[0].grade.grade == first[preds[0].info.style_id]
    # 快照仍是首次调用前的绝对档（A），不被污染值 X 覆盖
    assert all(
        p.metadata["absolute_grade"] == snapshots[p.info.style_id] for p in preds
    )


def test_risk_styles_excluded_from_percentile():
    """绝对档=风险 的款保留「风险」，不参与分位（绝对规则已判，优先级最高）"""
    preds = [_mk_pred(f"S{i:02d}", 90 - i) for i in range(9)]
    risk1 = _mk_pred("R01", 95.0, grade="风险")  # 高分但被风险规则判死
    risk2 = _mk_pred("R02", 10.0, grade="风险")
    preds += [risk1, risk2]
    result = assign_relative_grades(preds)
    assert result["R01"] == "风险" and result["R02"] == "风险"
    assert risk1.grade.grade == "风险" and risk2.grade.grade == "风险"
    # 9 款参与分位：S=round(9*0.2)=2, A+=2, A=round(9*0.35)=3, P=2
    ranked = [v for v in result.values() if v != "风险"]
    counts = {g: ranked.count(g) for g in ("S", "A+", "A", "P")}
    assert counts == {"S": 2, "A+": 2, "A": 3, "P": 2}


def test_ties_go_to_same_grade():
    """同分并列进同一档（分数边界，不是排名边界）：10款90分→全S"""
    scores = [90.0] * 10 + [80.0] * 5 + [70.0] * 3 + [60.0] * 2
    preds = [_mk_pred(f"T{i:02d}", s) for i, s in enumerate(scores)]
    result = assign_relative_grades(preds)
    by_score = {}
    for p in preds:
        by_score.setdefault(p.grade.final_score, set()).add(result[p.info.style_id])
    # 每个分数档内的所有款档位一致
    assert all(len(gs) == 1 for gs in by_score.values())
    assert by_score[90.0] == {"S"}     # 10 款 90 分全部并列 S
    assert by_score[80.0] == {"A"}
    assert by_score[70.0] == {"P"}
    assert by_score[60.0] == {"P"}


def test_cold_start_score_pileup():
    """冷启动：分数扎堆 65-75（VLM 尺度未校准，绝对档全 A）→ 相对分级恢复 S/A+/A/P 分布"""
    preds = [_mk_pred(f"C{i:02d}", 75.0 - i, grade="A") for i in range(10)]  # 75..66
    result = assign_relative_grades(preds)
    counts = {g: sum(1 for v in result.values() if v == g) for g in ("S", "A+", "A", "P")}
    # 10 款：S=2, A+=2, A=4（round(3.5)=4）, P=2 —— 分布恢复，不再全 A
    assert counts == {"S": 2, "A+": 2, "A": 4, "P": 2}
    assert result["C00"] == "S"   # 75 分最高 → S
    assert result["C09"] == "P"   # 66 分最低 → P
    # 绝对档快照保留原始 A 档（供总表对照列展示）
    assert all(p.metadata["absolute_grade"] == "A" for p in preds)


def test_empty_and_single_style():
    """空批次 → {}；单款 → S（批次内唯一款就是 Top）"""
    assert assign_relative_grades([]) == {}
    single = [_mk_pred("ONLY", 66.6, grade="A")]
    result = assign_relative_grades(single)
    assert result == {"ONLY": "S"}
    assert single[0].grade.grade == "S"


def test_pipeline_run_batch_calls_relative_grading():
    """run_batch 集成：批量处理完成后调用相对分级（报告/XLSX 前完成重定档）"""
    import inspect
    from src.pipeline import run_batch

    src = inspect.getsource(run_batch)
    assert "assign_relative_grades" in src
    # 报告生成在相对分级之后（保证单款报告展示最终相对档）
    rel_pos = src.index("assign_relative_grades(predictions)")
    report_pos = src.index("generate_report(pred,")
    assert rel_pos < report_pos


# ============================================================
# app.py 批次总表接入（源码级断言）
# ============================================================
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")


def test_app_summary_page_applies_relative_grading_once():
    """批次总表：批次完成后相对分级 + 历史库写入只执行一次
    （batch_finalized 守卫，防 Streamlit rerun 重复写库/重复分级）"""
    assert "from src.grading import assign_relative_grades" in APP_SRC
    assert 'st.session_state.get("batch_finalized")' in APP_SRC
    # 新批次提交时重置守卫
    assert "st.session_state.batch_finalized = False" in APP_SRC
    # 相对分级先于历史库写入（history 存的是相对档）
    finalize_pos = APP_SRC.index('st.session_state.get("batch_finalized")')
    grade_pos = APP_SRC.index("assign_relative_grades(preds)")
    history_pos = APP_SRC.index("pl.save_to_history(")
    assert finalize_pos < grade_pos < history_pos


def test_app_summary_page_absolute_grade_column_and_diagnostics():
    """批次总表：绝对档对照列 + 分数分布诊断（均值/σ/扎堆告警）"""
    assert '"绝对档"' in APP_SRC
    assert 'p.metadata.get("absolute_grade"' in APP_SRC
    assert "分级逻辑诊断" in APP_SRC
    assert "分数扎堆告警" in APP_SRC


if __name__ == "__main__":
    test_bucket_proportions_20_styles()
    test_idempotent_and_snapshot_preserved()
    test_risk_styles_excluded_from_percentile()
    test_ties_go_to_same_grade()
    test_cold_start_score_pileup()
    test_empty_and_single_style()
    test_pipeline_run_batch_calls_relative_grading()
    test_app_summary_page_applies_relative_grading_once()
    test_app_summary_page_absolute_grade_column_and_diagnostics()
    print("✅ 相对分级测试全部通过")
