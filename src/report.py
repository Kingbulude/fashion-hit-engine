"""报告生成：Markdown 单款报告
输出到 output/reports/<style_id>_grade.md
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import load_brand_profile
from .types import BrandConfig, FullPrediction


_IMPROVE_TEMPLATE = "  {idx}. {text}\n"


def _count_calibrated_yamls(brand_cfg: BrandConfig | None) -> int:
    """检查 calibrated_dir 下有多少个 yaml 文件作为校准轮次。"""
    if brand_cfg is None:
        return 0
    calibrated_dir_str = getattr(brand_cfg, "calibrated_dir", None)
    if not calibrated_dir_str:
        return 0
    calibrated_dir = Path(calibrated_dir_str)
    if not calibrated_dir.exists():
        return 0
    try:
        yamls = [p for p in calibrated_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
                 and not p.name.startswith(".")]
        return len(yamls)
    except OSError:
        return 0


def _resolve_brand_info(brand_cfg: BrandConfig | None) -> tuple[str, str, int]:
    """返回 (brand_name, brand_id, n_calibrated)。None 时用默认 tongzhuang-outdoor。"""
    if brand_cfg is None:
        brand_cfg = load_brand_profile("tongzhuang-outdoor")
    n = _count_calibrated_yamls(brand_cfg)
    return brand_cfg.brand_name, brand_cfg.brand_id, n


def _stars(n: float) -> str:
    n = round(n)
    return "★" * n + "☆" * (5 - n)


def _score_bar(value: float, max_val: float = 100) -> str:
    pct = int(round(min(value, max_val) / max_val * 20))
    return "█" * pct + "░" * (20 - pct) + f" {value:.1f}/{max_val}"


def generate_report(pred: FullPrediction, out_dir: Path, *, brand_cfg: BrandConfig | None = None) -> Path:
    """[兼容旧签名] 等同于 generate_markdown_report()。"""
    return generate_markdown_report(pred, out_dir, brand_cfg=brand_cfg)


def generate_markdown_report(
    pred: FullPrediction,
    out_dir: Path,
    *,
    brand_cfg: BrandConfig | None = None,
) -> Path:
    """生成 Markdown 单款爆款预测报告（v2.0 支持 BrandConfig 页脚）

    在报告页脚写入：
    「本报告基于品牌 {brand_name} ({brand_id}) 的品牌适配包生成 · 当前校准轮次：{n_calibrated或0冷启动}」
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pred.info.style_id}_grade.md"

    brand_name, brand_id, n_calibrated = _resolve_brand_info(brand_cfg)
    calibrate_label = "0（冷启动）" if n_calibrated == 0 else str(n_calibrated)

    grade_label = pred.grade.grade
    grade_cn = {"S": "S级主推款", "A+": "A+潜力款", "A": "A级常规款", "P": "P级设计款", "风险": "风险款"}
    grade_badge = grade_cn.get(grade_label, grade_label)

    lines = [
        f"# 款号 {pred.info.style_id} · 爆款预测报告",
        "",
        f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**内审分级**：{grade_badge} （最终分 **{pred.grade.final_score:.1f}/100** · 置信度 **{pred.grade.confidence:.0%}**）",
        f"**推荐渠道**：{pred.grade.recommended_channel}",
        f"**定价**：{pred.info.price}元（本批次价格百分位 {pred.channels.price_percentile:.0%}）",
        "",
        "---",
        "",
        "## 一、最终得分仪表盘",
        "",
        f"- 人设投票加权分：{_score_bar(pred.voting.weighted_score * 10, 100)}",
        f"- 自然流量潜力分：{_score_bar(pred.channels.natural_score * 10, 100)}",
        f"- 直播带货潜力分：{_score_bar(pred.channels.live_score * 10, 100)}",
        f"- 感知价值分：{_score_bar(pred.channels.perceived_value * 10, 100)}",
        f"- 价格价值匹配度（VM）：{pred.channels.value_match:+.2f} （**{pred.channels.price_risk}**）",
        f"- 综合最终分：{_score_bar(pred.grade.final_score, 100)}",
        "",
        "---",
        "",
        "## 二、消费者态度画像（30人设双重决策模拟）",
        "",
        f"- 客群明确支持率：**{pred.voting.support_rate:.0%}**",
        f"- 客群明确反对率：**{pred.voting.opposition_rate:.0%}**",
        f"- 评分分歧度（σ）：{pred.voting.score_std:.2f}",
        f"- 消费者洞察：{pred.grade.consumer_insights}",
        "",
        "### 核心买点（被反复提及的理由）",
    ]
    if pred.voting.top_buy_reasons:
        for i, r in enumerate(pred.voting.top_buy_reasons, 1):
            lines.append(f"  {i}. {r}")
    else:
        lines.append("  *无高频买点汇总*")
    lines += ["", "### 核心顾虑（被反复提及的理由）"]
    if pred.voting.top_oppose_reasons:
        for i, r in enumerate(pred.voting.top_oppose_reasons, 1):
            lines.append(f"  {i}. {r}")
    else:
        lines.append("  *无高频顾虑汇总*")

    lines += [
        "",
        "---",
        "",
        "## 三、10个设计特征BARS评分",
        "",
        "| 编号 | 特征 | 分数 | 档位 | 置信度 | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    level_map = {
        (8, 10.1): "非常高",
        (6.5, 8): "较高",
        (5, 6.5): "中等",
        (3.5, 5): "较低",
        (0, 3.5): "非常低",
    }
    for key, f in pred.features.features.items():
        lvl = ""
        for (a, b), name in level_map.items():
            if a <= f.score < b:
                lvl = name
                break
        review = " ⚠️ 需复核" if f.needs_review else ""
        lines.append(f"| {key} | {f.name} | {f.score:.1f} | {lvl} | {f.confidence:.0%} | {f.reason[:40]}{review} |")

    lines += [
        "",
        "---",
        "",
        "## 四、优势 vs 不足 vs 改款建议",
        "",
        "### 核心优势",
    ]
    if pred.grade.strengths:
        for s in pred.grade.strengths:
            lines.append(f"- ✅ {s}")
    else:
        lines.append("- 无显著优势")

    lines += ["", "### 关键短板"]
    if pred.grade.weaknesses:
        for w in pred.grade.weaknesses:
            lines.append(f"- ❌ {w}")
    else:
        lines.append("- 无显著短板")

    lines += ["", "### 改款建议（用于内审&下一轮企划迭代）"]
    if pred.grade.improvements:
        for s in pred.grade.improvements:
            s_filled = s.replace("[价格]", f"{pred.info.price}")
            lines.append(f"- 💡 {s_filled}")
    else:
        lines.append("- 暂无明确改款建议，保持现有设计")

    lines += [
        "",
        "---",
        "",
        "## 五、渠道投放建议",
        "",
        f"- 自然流量（搜索/推荐）潜力：{pred.channels.natural_score:.1f}/10 {_stars(pred.channels.natural_score / 2)}",
        f"- 直播带货潜力：{pred.channels.live_score:.1f}/10 {_stars(pred.channels.live_score / 2)}",
        f"- 推荐策略：**{pred.grade.recommended_channel}**",
        "",
        "---",
        "",
        f"*本报告基于品牌 {brand_name} ({brand_id}) 的品牌适配包生成 · 当前校准轮次：{calibrate_label}*",
        "",
        "*AI模拟30人设双重决策+视觉特征多模型交叉验证得出。为内部参考辅助工具，不替代专业判断。*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_backtest_summary(
    predictions: list[FullPrediction],
    spearman_grade: float,
    spearman_sales: float,
    accuracy_s_vs_not: float,
    out_path: Path,
    *,
    brand_cfg: BrandConfig | None = None,
) -> Path:
    """生成回测报告（v2.0 支持 BrandConfig 页脚）"""
    brand_name, brand_id, n_calibrated = _resolve_brand_info(brand_cfg)
    calibrate_label = "0（冷启动）" if n_calibrated == 0 else str(n_calibrated)

    lines = [
        "# 回测报告：爆款预测准确率验证",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 一、关键指标",
        "",
        f"| 指标 | 值 | 达标阈值 | 结论 |",
        "|---|---|---|---|",
        f"| Spearman(预测 vs 人工S/A/P分级) | {spearman_grade:+.3f} | ≥ 0.45 | {'✅ 达标' if spearman_grade >= 0.45 else '⚠️ 待优化'} |",
        f"| Spearman(预测 vs 真实销量) | {spearman_sales:+.3f} | ≥ 0.65 | {'✅ 达标' if spearman_sales >= 0.65 else '⚠️ 待优化'} |",
        f"| S款识别准确率(S vs 非S二分类) | {accuracy_s_vs_not:.0%} | ≥ 70% | {'✅ 达标' if accuracy_s_vs_not >= 0.7 else '⚠️ 待优化'} |",
        "",
        "## 二、逐款对比表",
        "",
        "| 款号 | 预测分级 | 人工分级 | 预测最终分 | 真实销量 | 价格 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in predictions:
        ok = ""
        if p.grade.grade == p.info.manual_grade:
            ok = "✅ 命中"
        elif {p.grade.grade, p.info.manual_grade} <= {"S", "A+"}:
            ok = "⬆️ 接近"
        elif {p.grade.grade, p.info.manual_grade} <= {"A", "P"}:
            ok = "⬇️ 接近"
        else:
            ok = "❌ 偏差"
        lines.append(
            f"| {p.info.style_id} | {p.grade.grade} | {p.info.manual_grade or '未标'} | "
            f"{p.grade.final_score:.1f} | {p.info.sales_qty} | {p.info.price}元 | {ok} |"
        )

    lines += [
        "",
        "## 三、改进方向",
        "",
        "- 若Spearman<0.65：进入训练模式，用本次数据对校准层Lasso回归重新拟合权重",
        "- 若S款识别偏差大：检查VLM模型是否对该款式品类存在审美偏差，可增加特征权重或补充特征维度",
        "- 若普遍高估销量：提高反对率惩罚因子，或重新拟合人设分布权重",
        "",
        "---",
        "",
        f"*本报告基于品牌 {brand_name} ({brand_id}) 的品牌适配包生成 · 当前校准轮次：{calibrate_label}*",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
