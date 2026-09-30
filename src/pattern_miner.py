"""PatternMiner — 从历史销售数据提炼可解释的爆款/败款模式

核心思想：ResidualDecomposer 只做了"四象限投放归因"，
没有回答 "款式本身的什么特征导致了爆/败"。
PatternMiner 用 Decision Tree 从历史数据自动提炼：
  "当一款同时具备 F06>7.2 + F09<4.8 + P15支持率>55% 时，
   有 82% 概率成为 S 款"

这些规则：
  1. 人类可读（给买手/运营直接看）
  2. 机器可用（存 YAML，下次评估时做 few-shot 注入）
  3. 自动衰减（每条规则带 season/confidence，过期自动降权）

设计原则：
  - 不依赖额外 LLM 调用（纯 sklearn + pandas）
  - 小样本友好（max_depth=4，min_samples_leaf=3 防过拟合）
  - 与 HistoryStore 解耦（任何带 feature 列的 DataFrame 都能喂）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PatternRule:
    """一条可解释的分类规则。"""
    target_grade: str          # 这条规则预测的目标 grade (S/A+/A/P)
    conditions: list[str]     # 人类可读的条件 ["F06(可穿性) > 7.2", "F09(饱和度) < 4.8"]
    feature_cols: list[str]   # 涉及的原始特征列 ["F06", "F09"]
    thresholds: list[float]   # 阈值 [7.2, 4.8]
    operators: list[str]      # 比较符 [">", "<"]
    confidence: float         # 置信度（命中规则的款里 target_grade 的比例）
    coverage: float           # 覆盖率（所有款里命中这条规则的比例）
    n_samples: int            # 样本量（规则覆盖的款数）

    def to_text(self) -> str:
        """人类可读文本。"""
        cond_str = " AND ".join(f"({c})" for c in self.conditions)
        return (
            f"IF {cond_str} "
            f"→ 预测 {self.target_grade}款 "
            f"（置信度 {self.confidence:.0%}，"
            f"覆盖 {self.coverage:.0%}，"
            f"样本 {self.n_samples}款）"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_grade": self.target_grade,
            "conditions": self.conditions,
            "feature_cols": self.feature_cols,
            "thresholds": self.thresholds,
            "operators": self.operators,
            "confidence": round(self.confidence, 4),
            "coverage": round(self.coverage, 4),
            "n_samples": self.n_samples,
        }


@dataclass
class PatternMineResult:
    rules: list[PatternRule]
    s_rules: list[PatternRule] = field(default_factory=list)
    p_rules: list[PatternRule] = field(default_factory=list)
    n_samples_total: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    tree_accuracy: float = 0.0

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples_total,
            "tree_accuracy": round(self.tree_accuracy, 4),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "s_rules": [r.to_dict() for r in self.s_rules],
            "p_rules": [r.to_dict() for r in self.p_rules],
            "all_rules": [r.to_dict() for r in self.rules],
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# 模式提炼报告（PatternMiner）")
        lines.append("")
        lines.append(f"- 总样本数：**{self.n_samples_total}** 款")
        lines.append(f"- Decision Tree 分类准确率：**{self.tree_accuracy:.1%}**")
        lines.append("")

        # 特征重要性 Top 10
        top_feats = sorted(self.feature_importance.items(), key=lambda x: -x[1])[:10]
        if top_feats:
            lines.append("## 🔥 关键特征 Top 10（区分 S/P 款最重要的属性）")
            lines.append("")
            for feat, imp in top_feats:
                lines.append(f"- **{feat}**：重要度 {imp:.3f}")
            lines.append("")

        # S 款规则
        if self.s_rules:
            lines.append("## 🏆 S款 模式（爆款基因）")
            lines.append("")
            for r in self.s_rules:
                lines.append(f"**{r.to_text()}**")
                lines.append("")

        # P 款规则
        if self.p_rules:
            lines.append("## ⚠️ P款 模式（避坑指南）")
            lines.append("")
            for r in self.p_rules:
                lines.append(f"**{r.to_text()}**")
                lines.append("")

        if not self.s_rules and not self.p_rules:
            lines.append("⚠️ 样本量不足或特征区分度太低，没有提炼出有意义的规则。")
            lines.append("建议：① 增加带真实销量的款式 ② 检查特征列是否完整 ③ 调小阈值")

        return "\n".join(lines)


# VLM 特征列 → 中文名映射（用于人类可读规则）
FEATURE_NAME_MAP = {
    "F01": "视觉吸引力",
    "F02": "独特性",
    "F03": "设计复杂度",
    "F04": "功能性",
    "F05": "拍照出片",
    "F06": "可穿性",
    "F07": "搭配性",
    "F08": "品质感",
    "F09": "色彩饱和度",
    "F10": "视觉冲击",
}


def _col_display(col: str) -> str:
    """特征列 → 人类可读名。"""
    return FEATURE_NAME_MAP.get(col, col)


def _grade_labels_from_sales(
    sales: pd.Series,
    s_pct: float = 0.90,
    aplus_pct: float = 0.75,
    a_pct: float = 0.25,
) -> list[str]:
    """从销量构造 S/A+/A/P 标签（percentile 边界）。"""
    n = len(sales)
    if n == 0:
        return []
    ranked = sales.rank(method="average", pct=True)
    labels: list[str] = []
    for p in ranked.values:
        if p >= s_pct:
            labels.append("S")
        elif p >= aplus_pct:
            labels.append("A+")
        elif p >= a_pct:
            labels.append("A")
        else:
            labels.append("P")
    return labels


def mine_patterns(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    persona_cols: list[str] | None = None,
    sales_col: str = "sales",
    grade_labels: list[str] | None = None,
    max_depth: int = 4,
    min_samples_leaf: int = 3,
    min_rule_confidence: float = 0.60,
    min_rule_coverage: float = 0.05,
    top_k_rules_per_grade: int = 8,
) -> PatternMineResult:
    """核心入口：从历史 DataFrame 提炼模式规则。

    Args:
        df: 历史款 DataFrame，必须带 sales_col 列
        feature_cols: 要用于决策树的特征列。None → 自动探测 F01-F10
        persona_cols: 要加入的人设列。None → 自动探测 P01-P30
        sales_col: 销量列名
        grade_labels: 预构造的标签。None → 从销量百分位自动构造
        max_depth: 决策树最大深度（小样本防过拟合，默认 4）
        min_samples_leaf: 叶节点最小样本数（默认 3）
        min_rule_confidence: 最低置信度（默认 60%）
        min_rule_coverage: 最低覆盖率（默认 5%）
        top_k_rules_per_grade: 每个 grade 最多保留几条规则

    Returns:
        PatternMineResult
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, _tree
    except ImportError:
        log.error("scikit-learn 未安装，无法跑 PatternMiner")
        return PatternMineResult()

    if df.empty:
        log.warning("PatternMiner: 空 DataFrame")
        return PatternMineResult()

    n = len(df)

    # --- 自动探测特征列 ---
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("F") and c[1:].isdigit()]
    if persona_cols is None:
        persona_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]

    all_cols = feature_cols + persona_cols
    if not all_cols:
        log.warning("PatternMiner: 没有可用的特征列")
        return PatternMineResult()

    # --- 构造标签 ---
    if grade_labels is None:
        if sales_col not in df.columns:
            log.error(f"PatternMiner: 缺少销量列 {sales_col}")
            return PatternMineResult()
        grade_labels = _grade_labels_from_sales(df[sales_col])

    if len(grade_labels) != n:
        log.error(f"PatternMiner: 标签数 ({len(grade_labels)}) != 行数 ({n})")
        return PatternMineResult()

    # 去掉含 NaN 的行
    df_work = df[all_cols].copy()
    df_work["_label"] = grade_labels
    df_work = df_work.dropna()
    if len(df_work) < min_samples_leaf * 4:
        log.warning(f"PatternMiner: 有效样本不足 ({len(df_work)} < {min_samples_leaf * 4})")
        return PatternMineResult()

    X = df_work[all_cols].values.astype(float)
    y = df_work["_label"].values
    valid_mask = df_work.index.tolist()
    y_true_counts = {l: list(y).count(l) for l in ["S", "A+", "A", "P"]}
    log.info("PatternMiner: 有效样本 %d, 标签分布 %s", len(X), y_true_counts)

    # --- 训练决策树 ---
    try:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            # 注意：不用 class_weight='balanced' — 会让 tree.value[]
            # 变成加权值，n_node_samples 才是真实样本数
            random_state=42,
        )
        clf.fit(X, y)
    except Exception as e:
        log.exception("PatternMiner: DecisionTree 训练失败: %s", e)
        return PatternMineResult(n_samples_total=len(X))

    # --- 准确率 ---
    tree_acc = float(clf.score(X, y))

    # --- 特征重要性 ---
    feat_imp = {
        all_cols[i]: float(v)
        for i, v in enumerate(clf.feature_importances_)
        if v > 1e-6
    }

    # --- 从决策树提取规则 ---
    rules = _extract_rules_from_tree(
        clf, all_cols, y,
        min_confidence=min_rule_confidence,
        min_coverage=min_rule_coverage,
    )

    # 按置信度×覆盖率 排序
    rules.sort(key=lambda r: r.confidence * r.coverage, reverse=True)

    # 分开 S/P 规则，各保留 top_k
    s_rules = [r for r in rules if r.target_grade == "S"][:top_k_rules_per_grade]
    p_rules = [r for r in rules if r.target_grade == "P"][:top_k_rules_per_grade]

    return PatternMineResult(
        rules=rules[:top_k_rules_per_grade * 2],
        s_rules=s_rules,
        p_rules=p_rules,
        n_samples_total=len(X),
        feature_importance=feat_imp,
        tree_accuracy=tree_acc,
    )


def _extract_rules_from_tree(
    clf: "DecisionTreeClassifier",
    feature_cols: list[str],
    y_true: "list | np.ndarray",
    *,
    min_confidence: float = 0.60,
    min_coverage: float = 0.05,
) -> list[PatternRule]:
    """从 sklearn DecisionTreeClassifier 提取人类可读规则。

    遍历叶节点：每个叶节点 = 一条规则。
    条件链从根到叶的 split 组合。
    """
    import numpy as np
    from sklearn.tree import _tree  # noqa: F401 — 模块级常量

    TREE_UNDEFINED = -2   # sklearn._tree.TREE_UNDEFINED 常量值

    tree = clf.tree_
    n = len(y_true)
    rules: list[PatternRule] = []

    def recurse(node: int, conditions: list[tuple[str, str, float]]):
        if int(tree.feature[node]) == TREE_UNDEFINED:
            # 叶节点 → 一条规则
            values = tree.value[node][0]  # 每类样本数
            n_samples = int(tree.n_node_samples[node])  # 真实样本数
            if n_samples < 2:
                return

            # 目标 grade = 叶节点最多的类
            classes = clf.classes_
            target_idx = int(np.argmax(values))
            target_grade = str(classes[target_idx])
            # 置信度：叶节点中目标类的比例
            confidence = float(values[target_idx] / values.sum())
            # 覆盖率：叶节点样本数 / 总样本数
            coverage = float(n_samples / n)

            if confidence < min_confidence or coverage < min_coverage:
                return

            # 构造人类可读条件
            cond_strs: list[str] = []
            feats: list[str] = []
            threshs: list[float] = []
            ops: list[str] = []
            for col, op, thresh in conditions:
                feats.append(col)
                threshs.append(thresh)
                ops.append(op)
                display_col = _col_display(col)
                cond_strs.append(f"{display_col} {op} {thresh:.1f}")

            if not cond_strs:
                return

            rules.append(PatternRule(
                target_grade=target_grade,
                conditions=cond_strs,
                feature_cols=feats,
                thresholds=threshs,
                operators=ops,
                confidence=confidence,
                coverage=coverage,
                n_samples=n_samples,
            ))
            return

        # 内部节点 → 分左右递归
        feat_idx = int(tree.feature[node])
        col = feature_cols[feat_idx]
        thresh = float(tree.threshold[node])

        recurse(tree.children_left[node], conditions + [(col, "<=", thresh)])
        recurse(tree.children_right[node], conditions + [(col, ">", thresh)])

    try:
        recurse(0, [])
    except Exception as e:
        log.exception("PatternMiner: 规则提取失败: %s", e)

    return rules


def save_patterns(
    result: PatternMineResult,
    output_dir: str | Path,
    *,
    brand_name: str = "",
    quarter: str = "",
) -> list[Path]:
    """把 PatternMineResult 落盘为 YAML + Markdown。

    Args:
        result: mine_patterns() 的返回
        output_dir: 输出目录（会自动创建）
        brand_name: 品牌名（嵌入文件名）
        quarter: 季度标识（嵌入文件名）

    Returns:
        生成的文件路径列表
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tag = f"{brand_name}_{quarter}" if brand_name else quarter
    if not tag:
        tag = "pattern"

    yaml_path = out / f"{tag}_patterns.yaml"
    md_path = out / f"{tag}_patterns.md"

    import yaml

    yaml_path.write_text(
        yaml.dump(result.to_yaml_dict(), allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    md_path.write_text(result.to_markdown(), encoding="utf-8")

    log.info("PatternMiner: 规则落盘 → %s + %s", yaml_path, md_path)
    return [yaml_path, md_path]


# ========== Few-shot 规则匹配（供 P1 注入 prompt 用）==========

def match_rules_for_style(
    patterns_yaml_path: str | Path,
    style_features: dict[str, float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """根据规则 YAML，为一款新 style 匹配最相关的历史模式。

    Args:
        patterns_yaml_path: save_patterns 生成的 YAML 文件路径
        style_features: 这款的特征分 {"F06": 7.8, "P15": 0.62, ...}
        top_k: 最多返回几条匹配的规则

    Returns:
        匹配到的规则列表（按相关度排序），每条附加 matched_condition 说明
    """
    import yaml

    path = Path(patterns_yaml_path)
    if not path.exists():
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    all_rules = data.get("all_rules", []) or data.get("s_rules", []) + data.get("p_rules", [])

    scored: list[tuple[float, dict]] = []
    for rule in all_rules:
        matched = 0
        total = len(rule.get("feature_cols", []))
        if total == 0:
            continue
        for col, op, thresh in zip(
            rule.get("feature_cols", []),
            rule.get("operators", []),
            rule.get("thresholds", []),
        ):
            val = style_features.get(col)
            if val is None:
                continue
            ok = (val <= thresh) if op == "<=" else (val > thresh)
            if ok:
                matched += 1
        coverage = matched / total
        # 综合分：规则置信度 × 覆盖率
        score = rule.get("confidence", 0) * coverage
        if score > 0.2:   # 最低匹配度阈值
            rule_copy = dict(rule)
            rule_copy["match_score"] = round(score, 3)
            rule_copy["matched_count"] = f"{matched}/{total}"
            scored.append((score, rule_copy))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def build_fewshot_context(
    patterns_yaml_path: str | Path,
    style_features: dict[str, float],
    *,
    brand_suffix: str = "",
) -> str:
    """把匹配到的历史模式组装成 LLM prompt 可直接用的 few-shot 段落。

    这是 P1 的核心函数 — persona voting prompt 调用时注入。
    """
    matches = match_rules_for_style(patterns_yaml_path, style_features, top_k=3)
    if not matches:
        return ""

    lines = ["【历史市场模式参考】"]
    for m in matches:
        target = m.get("target_grade", "?")
        conf = m.get("confidence", 0)
        conds = m.get("conditions", [])
        lines.append(f"过去 {brand_suffix} 款中，符合以下特征的款式：")
        for c in conds:
            lines.append(f"  • {c}")
        lines.append(f"  → 有 {conf:.0%} 概率成为 **{target}款**")
        lines.append("")

    lines.append("请参考这些历史模式，给出你当前评估这款的评分。")
    return "\n".join(lines)
