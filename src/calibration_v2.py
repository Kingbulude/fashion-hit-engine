"""
calibration_v2.py — 新版 calibration 核心模块（v2 重构）

替换旧的 optimization_kernel.py (3 Loop Lasso)。
四步 calibration 流程：
  1. 特征重要性评估 (permutation importance)
  2. 爆款分类器训练 (LightGBM 全量训练 + CV 稳健性评估)
  3. P 款识别规则阈值学习 (从内审 P 款数据自动学习)
  4. Validation 报告生成 (baseline vs 校准后对比)

产物保存到 brand_profiles/{brand}/calibrated/:
  - calibration_report.json: CV 结果、AUC、特征重要性、分类器阈值
  - classifier_model.pkl: 全量训练好的 LightGBM 模型
  - feature_weights.json: 特征重要性权重（给记忆库用）
  - p_rule.json: P 款识别规则阈值
  - pattern_rules.json: PatternMiner 爆款基因规则
  - memory_db.parquet: 品牌记忆库

作者：System V2 Architecture
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class CalibrationResult:
    """calibration 完整结果"""
    brand_name: str = ""
    total_samples: int = 0
    total_bombs: int = 0
    total_p_styles: int = 0
    bomb_rate: float = 0.0
    
    # Baseline 结果
    baseline_method: str = ""
    baseline_auc: float = 0.0
    baseline_recall_at_10: float = 0.0
    baseline_precision_at_10: float = 0.0
    
    # 校准后结果 (CV 稳健性评估)
    calibrated_model: str = ""
    calibrated_cv_auc_mean: float = 0.0
    calibrated_cv_auc_std: float = 0.0
    calibrated_cv_ap_mean: float = 0.0
    calibrated_cv_ap_std: float = 0.0
    
    # 全量训练排序效果（生产用的模型）
    full_train_recall_at_10: float = 0.0
    full_train_precision_at_10: float = 0.0
    full_train_top_10_bombs: list = field(default_factory=list)
    
    # 特征重要性 (sorted list of {feature, importance})
    feature_importance: list = field(default_factory=list)
    
    # 分类器阈值 (从全量训练 OOF 预测中学习)
    s_threshold: float = 0.70     # 爆款概率 ≥ 此 → "多备货+重点投流"
    aplus_threshold: float = 0.40  # ≥ 此且 < s → "正常备货+常规运营"
    
    # P 款识别规则
    p_rule: dict = field(default_factory=dict)
    
    # 校准时间
    calibrated_at: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# 主入口
# ============================================================================

def run_calibration(
    history_df: pd.DataFrame,
    brand_name: str,
    target_dir: str | Path = None,
    baseline_method: str = "internal_review",
    random_state: int = 42,
) -> CalibrationResult:
    """运行完整 calibration 流程
    
    Args:
        history_df: 历史数据 DataFrame，必须包含：
            - style_id, category, season, price, year
            - design_text (版型描述文本)
            - F11-F20 (版型文本特征，可由 text_feature_extractor 提前提取)
            - is_bomb (bool)
            - is_p_style (bool)
            - label_30d, internal_review_grade, sales_num
            - VLM 10 特征 (如果有)
            - 人设加权分 (如果有)
        brand_name: 品牌名 (如 "mipo")
        target_dir: calibration 产物输出目录，默认 brand_profiles/{brand}/calibrated/
        baseline_method: baseline 对比方法，"internal_review" (内审分级 S>A>P) 或 "random_forest_baseline" (Baseline E)
        random_state: 随机种子
    
    Returns:
        CalibrationResult 完整结果
    """
    if target_dir is None:
        target_dir = Path(f"brand_profiles/{brand_name}/calibrated")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    result = CalibrationResult(brand_name=brand_name)
    result.total_samples = len(history_df)
    result.total_bombs = int(history_df['is_bomb'].sum())
    result.total_p_styles = int(history_df['is_p_style'].sum())
    result.bomb_rate = result.total_bombs / max(result.total_samples, 1)
    
    logger.info(f"=== Calibration v2 开始: brand={brand_name}, samples={result.total_samples}, bombs={result.total_bombs} ===")
    
    # ---- Step 1: 准备特征矩阵 ----
    X, y, feature_names = _build_feature_matrix(history_df)
    if X is None:
        raise ValueError("history_df 无法构建特征矩阵，检查必需列是否存在")
    result._feature_names = feature_names  # type: ignore
    
    logger.info(f"特征数: {len(feature_names)}, 正样本: {y.sum()}")
    
    # ---- Step 2: Baseline 评估 ----
    _evaluate_baseline(result, history_df, X, y, baseline_method, random_state)
    
    # ---- Step 3: 爆款分类器训练 ----
    best_model, best_model_name, full_train_proba, cv_metrics = _train_bomb_classifier(
        X, y, random_state
    )
    
    result.calibrated_model = best_model_name
    result.calibrated_cv_auc_mean = cv_metrics['auc_mean']
    result.calibrated_cv_auc_std = cv_metrics['auc_std']
    result.calibrated_cv_ap_mean = cv_metrics['ap_mean']
    result.calibrated_cv_ap_std = cv_metrics['ap_std']
    
    # 全量训练排序效果
    df_sorted = history_df.copy()
    df_sorted['proba'] = full_train_proba
    df_sorted = df_sorted.sort_values('proba', ascending=False)
    top10 = df_sorted.head(10)
    hits = int(top10['is_bomb'].sum())
    result.full_train_recall_at_10 = hits / max(y.sum(), 1)
    result.full_train_precision_at_10 = hits / 10
    result.full_train_top_10_bombs = top10[top10['is_bomb']]['style_id'].tolist()
    
    # ---- Step 4: 特征重要性 ----
    result.feature_importance = _compute_feature_importance(
        best_model, X, y, feature_names, random_state
    )
    
    # ---- Step 5: 分类器阈值学习 ----
    result.s_threshold, result.aplus_threshold = _learn_probability_thresholds(
        full_train_proba, y, target_positive_rate=0.1
    )
    
    # ---- Step 6: P 款识别规则学习 ----
    result.p_rule = _learn_p_style_rules(history_df, random_state)
    
    # ---- Step 7: PatternMiner 规则提取 ----
    pattern_rules = _extract_pattern_rules(best_model, X, y, feature_names, history_df)
    
    # ---- Step 8: 保存产物 ----
    _save_calibration_artifacts(
        target_dir, result, best_model, feature_names, pattern_rules
    )
    
    # ---- Step 9: 保存记忆库 ----
    _save_memory_db(history_df, result.feature_importance, target_dir)
    
    logger.info(f"=== Calibration v2 完成 ===")
    logger.info(f"  Baseline AUC:  {result.baseline_auc:.3f}")
    logger.info(f"  CV AUC:        {result.calibrated_cv_auc_mean:.3f} ± {result.calibrated_cv_auc_std:.3f}")
    logger.info(f"  Full-train Top-10 Recall: {result.full_train_recall_at_10:.0%}")
    logger.info(f"  产物目录: {target_dir}")
    
    return result


# ============================================================================
# Step 1: 特征矩阵构建
# ============================================================================

def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]] | tuple[None, None, None]:
    """从 history_df 构建特征矩阵
    
    特征优先级：
    1. VLM 10 特征 (如果存在: F01-F10 或 vlm_01-vlm_10)
    2. F11-F20 版型文本特征 (必须存在)
    3. 品类 one-hot + 季节 one-hot + 价格
    4. 人设加权分 (如果存在)
    """
    X_parts = []
    feature_names = []
    
    # 品类 one-hot
    if 'category' in df.columns:
        cat_dummies = pd.get_dummies(df['category'], prefix='cat')
        X_parts.append(cat_dummies)
        feature_names.extend(cat_dummies.columns.tolist())
    
    # 季节 one-hot
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        X_parts.append(season_dummies)
        feature_names.extend(season_dummies.columns.tolist())
    
    # 价格 (分位归一化)
    if 'price' in df.columns:
        price_pct = df[['price']].rank(pct=True).rename(columns={'price': 'price_pct'})
        X_parts.append(price_pct)
        feature_names.append('price_pct')
    
    # F11-F20 版型文本特征
    f_cols = [c for c in df.columns if c.startswith('F') and c[1:].isdigit() and 11 <= int(c[1:]) <= 20]
    if f_cols:
        X_parts.append(df[f_cols])
        feature_names.extend(f_cols)
    
    # VLM 10 特征 (如果存在)
    vlm_cols = []
    for prefix in ['F', 'vlm_']:
        for i in range(1, 11):
            for suffix in ['', f'{i:02d}']:
                col_name = f'{prefix}{i}' if prefix == 'F' and i >= 10 else f'{prefix}{suffix}'
                # 简化：找 F01-F10 或 vlm_01-vlm_10 格式
                pass
    # 更简单：找 F01-F10 或 vlm 开头的列
    vlm_cols = [c for c in df.columns if 
                (c.startswith('F') and len(c) == 3 and c[1:].isdigit() and 1 <= int(c[1:]) <= 10) or
                (c.startswith('vlm_'))]
    if vlm_cols:
        X_parts.append(df[vlm_cols])
        feature_names.extend(vlm_cols)
    
    # 人设加权分
    persona_cols = [c for c in df.columns if 'persona' in c.lower() or 'weighted_score' in c.lower()]
    if persona_cols:
        X_parts.append(df[persona_cols])
        feature_names.extend(persona_cols)
    
    # Year (可选)
    if 'year' in df.columns:
        year_dummies = pd.get_dummies(df['year'].astype(str), prefix='year')
        X_parts.append(year_dummies)
        feature_names.extend(year_dummies.columns.tolist())
    
    if not X_parts:
        return None, None, None
    
    X = pd.concat(X_parts, axis=1).fillna(0)
    # 关键：pandas 3.0 + numpy 2.x 对 bool dtype 调 quantile 会抛 TypeError
    # 强制所有列 cast 成 float64，同时保证 LightGBM 训练安全
    for col in X.columns:
        X[col] = X[col].astype('float64')
    
    # 目标
    if 'is_bomb' not in df.columns:
        return None, None, None
    y = df['is_bomb'].astype(int)
    
    return X, y, feature_names


# ============================================================================
# Step 2: Baseline 评估
# ============================================================================

def _evaluate_baseline(
    result: CalibrationResult,
    history_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    random_state: int,
):
    """评估 baseline 方法的效果"""
    if method == "internal_review" and 'internal_review_grade' in history_df.columns:
        # 内审分级 S>A>P
        grade_score = {'S': 3, 'A': 2, 'P': 1}
        scores = history_df['internal_review_grade'].map(grade_score).fillna(0).values
        
        # 用分数直接当"概率"来算 AUC
        try:
            result.baseline_auc = float(roc_auc_score(y, scores))
        except Exception:
            result.baseline_auc = 0.5
        
        # Top-10 Recall
        df_temp = history_df.copy()
        df_temp['grade_score'] = scores
        top10 = df_temp.sort_values('grade_score', ascending=False).head(10)
        hits = int(top10['is_bomb'].sum())
        result.baseline_recall_at_10 = hits / max(y.sum(), 1)
        result.baseline_precision_at_10 = hits / 10
        result.baseline_method = "内审分级 S>A>P"
        
    else:
        # RandomForest Baseline (Baseline E)
        rf = RandomForestClassifier(100, class_weight='balanced', random_state=random_state)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
        aucs = []
        oof = np.zeros(len(y))
        for train_idx, test_idx in rskf.split(X, y):
            rf.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = rf.predict_proba(X.iloc[test_idx])[:, 1]
            oof[test_idx] += p
            if y.iloc[test_idx].sum() >= 1:
                aucs.append(roc_auc_score(y.iloc[test_idx], p))
        
        oof_avg = oof / 3.0
        try:
            result.baseline_auc = float(np.mean(aucs))
        except Exception:
            result.baseline_auc = 0.5
        
        df_temp = history_df.copy()
        df_temp['oof_proba'] = oof_avg
        top10 = df_temp.sort_values('oof_proba', ascending=False).head(10)
        hits = int(top10['is_bomb'].sum())
        result.baseline_recall_at_10 = hits / max(y.sum(), 1)
        result.baseline_precision_at_10 = hits / 10
        result.baseline_method = "Baseline E (RF + 文本特征)"
    
    logger.info(f"Baseline [{result.baseline_method}]: AUC={result.baseline_auc:.3f}, P@10={result.baseline_precision_at_10:.1%}, R@10={result.baseline_recall_at_10:.1%}")


# ============================================================================
# Step 3: 爆款分类器训练
# ============================================================================

def _train_bomb_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[Any, str, np.ndarray, dict]:
    """训练爆款分类器
    
    Returns:
        best_model: 全量训练好的模型
        best_name: 模型名称
        full_train_proba: 全量训练的预测概率
        cv_metrics: CV 稳健性评估 dict
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    
    candidates = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.08, num_leaves=4,
            min_child_samples=2, scale_pos_weight=scale_pos,
            random_state=random_state, verbose=-1,
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=random_state,
        ),
    }
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    
    best_model = None
    best_name = ""
    best_auc = 0.0
    cv_metrics = {}
    
    for name, model in candidates.items():
        aucs, aps = [], []
        for train_idx, test_idx in rskf.split(X, y):
            model_clone = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_clone.set_params(random_state=random_state) if hasattr(model, 'set_params') else None
            model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = model_clone.predict_proba(X.iloc[test_idx])[:, 1]
            if y.iloc[test_idx].sum() >= 1:
                aucs.append(roc_auc_score(y.iloc[test_idx], p))
            aps.append(average_precision_score(y.iloc[test_idx], p))
        
        mean_auc = np.mean(aucs) if aucs else 0.0
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name
            cv_metrics = {
                'auc_mean': mean_auc,
                'auc_std': np.std(aucs) if aucs else 0.0,
                'ap_mean': np.mean(aps),
                'ap_std': np.std(aps),
                'auc_folds': aucs,
                'ap_folds': aps,
            }
    
    # 用最佳模型全量训练
    best_model = candidates[best_name]
    best_model.fit(X, y)
    full_train_proba = best_model.predict_proba(X)[:, 1]
    
    logger.info(f"选中模型: {best_name}, CV AUC: {cv_metrics['auc_mean']:.3f} ± {cv_metrics['auc_std']:.3f}")
    
    return best_model, best_name, full_train_proba, cv_metrics


# ============================================================================
# Step 4: 特征重要性
# ============================================================================

def _compute_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    random_state: int,
) -> list[dict]:
    """计算特征重要性
    
    优先用模型内置的 feature_importances_，fallback 到 permutation importance
    """
    importances = {}
    
    # LightGBM / RF 有内置的
    if hasattr(model, 'feature_importances_'):
        raw = model.feature_importances_
        total = raw.sum() or 1
        for i, name in enumerate(feature_names):
            importances[name] = float(raw[i] / total)
    
    # Permutation importance 作为补充
    try:
        pi = permutation_importance(
            model, X.values, y.values,
            n_repeats=5, random_state=random_state,
            scoring='roc_auc'
        )
        pi_mean = pi.importances_mean
        pi_total = pi_mean.sum() or 1
        for i, name in enumerate(feature_names):
            # 取平均
            importances[name] = (importances.get(name, 0) + float(pi_mean[i] / pi_total)) / 2
    except Exception as e:
        logger.warning(f"Permutation importance 计算失败: {e}")
    
    # 排序
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
    return [{'feature': name, 'importance': imp} for name, imp in sorted_feats]


# ============================================================================
# Step 5: 分类器阈值学习
# ============================================================================

def _learn_probability_thresholds(
    proba: np.ndarray,
    y: pd.Series,
    target_positive_rate: float = 0.1,
) -> tuple[float, float]:
    """从全量训练的预测概率中学习分级阈值
    
    方法：
    - s_threshold: 让 top-10% 样本中爆款率最高的点 (或固定 0.70 fallback)
    - aplus_threshold: 让 top-30% 样本中爆款率显著高于平均的点 (或固定 0.40 fallback)
    
    简单做法：排序后找"概率明显跳升"的拐点
    """
    proba_sorted = np.sort(proba)[::-1]  # 降序
    n = len(proba_sorted)
    
    # 找跳升点：概率差超过 0.1 的位置
    diffs = -np.diff(proba_sorted)
    big_jumps = np.where(diffs > 0.1)[0]
    
    if len(big_jumps) >= 2:
        # 前两个大跳升点作为 S 和 A+ 阈值
        s_thr = float(proba_sorted[big_jumps[0]])
        aplus_thr = float(proba_sorted[big_jumps[1]])
    elif len(big_jumps) == 1:
        s_thr = float(proba_sorted[big_jumps[0]])
        aplus_thr = 0.40  # fallback
    else:
        # 没有明显跳升 → 用分位数
        s_thr = float(np.quantile(proba, 1 - target_positive_rate))
        aplus_thr = float(np.quantile(proba, 0.7))
    
    # clamp 到合理范围
    s_thr = max(0.50, min(0.95, s_thr))
    aplus_thr = max(0.20, min(s_thr - 0.05, aplus_thr))
    
    return round(s_thr, 3), round(aplus_thr, 3)


# ============================================================================
# Step 6: P 款识别规则学习
# ============================================================================

def _learn_p_style_rules(
    history_df: pd.DataFrame,
    random_state: int,
) -> dict:
    """从内审分级 P 款的数据特征分布中自动学习识别阈值
    
    策略：对内审 P 款和非 P 款分别计算各特征的分布差异，
    取 P 款特征分布的特定分位数作为阈值
    
    P 款典型特征（从内审分级 S/A/P 分布推断）：
    - 品牌调性高 (F09 高)
    - 售价偏高
    - 颜色安全度低 (可能大胆配色)
    - 版型经典度低 (独家/立体设计)
    """
    p_mask = history_df['internal_review_grade'] == 'P'
    non_p_mask = history_df['internal_review_grade'] != 'P'
    
    if p_mask.sum() < 3 or non_p_mask.sum() < 3:
        # 样本太少 → fallback 默认值
        return {
            'rules': [
                {'feature': 'F09_brand_identity', 'op': '>=', 'threshold': 7.0},
                {'feature': 'price_pct', 'op': '>=', 'threshold': 0.6},
            ],
            'method': 'fallback_default',
        }
    
    p_rules = []
    
    # 对每个可能区分 P 款的特征学习阈值
    candidate_features = []
    for col in history_df.columns:
        if col.startswith('F') and history_df[col].dtype in [np.float64, np.int64, float, int]:
            if history_df[col].nunique() > 3:
                candidate_features.append(col)
    if 'price' in history_df.columns:
        candidate_features.append('price')
    if 'price_pct' in history_df.columns:
        candidate_features.append('price_pct')
    
    for feat in candidate_features:
        p_vals = history_df.loc[p_mask, feat].dropna()
        non_p_vals = history_df.loc[non_p_mask, feat].dropna()
        
        if len(p_vals) < 3 or len(non_p_vals) < 3:
            continue
        
        # 简单规则：P 款的中位数显著高于/低于非 P 款
        p_med = p_vals.median()
        non_p_med = non_p_vals.median()
        
        if p_med > non_p_med * 1.2:
            thr = float(p_vals.quantile(0.3))  # P 款前 70% 都超过这个值
            p_rules.append({'feature': feat, 'op': '>=', 'threshold': round(thr, 2)})
        elif p_med < non_p_med * 0.8:
            thr = float(p_vals.quantile(0.7))
            p_rules.append({'feature': feat, 'op': '<=', 'threshold': round(thr, 2)})
    
    # 如果没学到规则，用默认
    if len(p_rules) < 2:
        p_rules = [
            {'feature': 'price_pct', 'op': '>=', 'threshold': 0.55},
        ]
    
    return {
        'rules': p_rules[:5],  # 最多 5 条
        'method': 'learned_from_internal_review',
        'p_count': int(p_mask.sum()),
        'non_p_count': int(non_p_mask.sum()),
    }


# ============================================================================
# Step 7: PatternMiner 规则提取 (从 LightGBM 树结构)
# ============================================================================

def _extract_pattern_rules(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    history_df: pd.DataFrame,
    min_support: int = 2,
    min_confidence: float = 0.2,
) -> list[dict]:
    """从训练好的模型中提取爆款基因规则
    
    如果是 LightGBM，遍历树的节点 split；否则用简单的特征阈值方法
    
    规则格式: {
        'rule_id': 'S-01',
        'conditions': [{'feature': 'F11', 'op': '>=', 'threshold': 5.0}, ...],
        'support': int,      # 命中样本数
        'confidence': float,  # 命中后爆款率
        'description': str,
    }
    """
    rules = []
    
    # 简单规则：对 top 特征，按阈值切分看爆款率
    importance_dict = _quick_importance(model, X, feature_names)
    top_features = [k for k, v in
                    sorted(importance_dict.items(), key=lambda x: -x[1])[:5]]
    
    rule_idx = 1
    
    # 单特征规则
    for feat in top_features[:3]:
        if feat not in X.columns:
            continue
        for threshold in [0.5, 0.6, 0.7]:
            col_max = X[feat].max()
            if threshold >= col_max:
                continue
            hit_mask = X[feat] >= threshold
            support = int(hit_mask.sum())
            if support < min_support:
                continue
            bomb_count = int(y[hit_mask].sum())
            confidence = bomb_count / support if support > 0 else 0
            
            if confidence >= min_confidence:
                rules.append({
                    'rule_id': f'S-{rule_idx:02d}',
                    'conditions': [{'feature': feat, 'op': '>=', 'threshold': round(float(threshold * col_max), 2)}],
                    'support': support,
                    'bomb_count': bomb_count,
                    'confidence': round(confidence, 3),
                    'description': f'{feat} ≥ {round(float(threshold * col_max), 2)} → 爆款率 {confidence:.0%} (支持 {support} 样本)',
                })
                rule_idx += 1
    
    # 双特征组合规则 (top 2 特征)
    if len(top_features) >= 2:
        f1, f2 = top_features[0], top_features[1]
        if f1 in X.columns and f2 in X.columns:
            for t1_ratio in [0.5, 0.7]:
                for t2_ratio in [0.5, 0.7]:
                    t1 = X[f1].quantile(t1_ratio)
                    t2 = X[f2].quantile(t2_ratio)
                    hit = (X[f1] >= t1) & (X[f2] >= t2)
                    support = int(hit.sum())
                    if support < min_support:
                        continue
                    bombs = int(y[hit].sum())
                    conf = bombs / support if support > 0 else 0
                    if conf >= min_confidence:
                        rules.append({
                            'rule_id': f'S-{rule_idx:02d}',
                            'conditions': [
                                {'feature': f1, 'op': '>=', 'threshold': round(float(t1), 2)},
                                {'feature': f2, 'op': '>=', 'threshold': round(float(t2), 2)},
                            ],
                            'support': support,
                            'bomb_count': bombs,
                            'confidence': round(conf, 3),
                            'description': f'{f1} ≥ {t1:.1f} AND {f2} ≥ {t2:.1f} → 爆款率 {conf:.0%} (支持 {support} 样本)',
                        })
                        rule_idx += 1
    
    # 按 confidence 排序，取 top 10
    rules.sort(key=lambda r: -r['confidence'])
    return rules[:10]


def _quick_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> dict:
    """快速获取特征重要性 dict"""
    importances = {}
    if hasattr(model, 'feature_importances_'):
        raw = model.feature_importances_
        total = raw.sum() or 1
        for i, name in enumerate(feature_names):
            importances[name] = float(raw[i] / total)
    else:
        for i, name in enumerate(feature_names):
            importances[name] = 1.0 / len(feature_names)
    return importances


# ============================================================================
# Step 8: 保存产物
# ============================================================================

def _save_calibration_artifacts(
    target_dir: Path,
    result: CalibrationResult,
    model: Any,
    feature_names: list[str],
    pattern_rules: list[dict],
):
    """保存所有 calibration 产物"""
    import datetime
    result.calibrated_at = datetime.datetime.now().isoformat()
    
    # calibration_report.json
    report = result.to_dict()
    # 移除不能序列化的
    report.pop('_feature_names', None)
    with open(target_dir / 'calibration_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # classifier_model.pkl
    with open(target_dir / 'classifier_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_names}, f)
    
    # feature_weights.json (给记忆库用)
    fw = {item['feature']: item['importance'] for item in result.feature_importance}
    with open(target_dir / 'feature_weights.json', 'w', encoding='utf-8') as f:
        json.dump(fw, f, ensure_ascii=False, indent=2)
    
    # p_rule.json
    with open(target_dir / 'p_rule.json', 'w', encoding='utf-8') as f:
        json.dump(result.p_rule, f, ensure_ascii=False, indent=2)
    
    # pattern_rules.json
    with open(target_dir / 'pattern_rules.json', 'w', encoding='utf-8') as f:
        json.dump(pattern_rules, f, ensure_ascii=False, indent=2)
    
    # 版本标记
    with open(target_dir / 'CALIBRATION_VERSION', 'w') as f:
        f.write('v2\n')
        f.write(datetime.datetime.now().isoformat() + '\n')
    
    logger.info(f"产物已保存到 {target_dir}/")


def _save_memory_db(
    history_df: pd.DataFrame,
    feature_importance: list,
    target_dir: Path,
):
    """保存品牌记忆库 (parquet)"""
    try:
        # 确保必需列存在
        cols_to_save = [c for c in history_df.columns if c in 
            ['style_id', 'category', 'season', 'price', 'year',
             'is_bomb', 'is_p_style', 'sales_num', 'label_30d',
             'internal_review_grade']]
        
        # 加 F11-F20 和 VLM 特征
        f_cols = [c for c in history_df.columns if 
                  (c.startswith('F') and len(c) == 3 and c[1:].isdigit()) or
                  c.startswith('vlm_') or
                  c.startswith('F') and c[1:].isdigit()]
        cols_to_save.extend(f_cols)
        
        memory = history_df[list(set(cols_to_save))].copy()
        memory.to_parquet(target_dir / 'memory_db.parquet', index=False)
        logger.info(f"记忆库已保存: {len(memory)} 款, {len(memory.columns)} 列")
    except Exception as e:
        logger.warning(f"记忆库保存失败 (可能缺少 pyarrow): {e}")


# ============================================================================
# 加载 calibration 产物 (PredictionPipeline 用)
# ============================================================================

def load_calibration_v2(calibrated_dir: str | Path) -> dict | None:
    """加载 calibration v2 产物
    
    Returns:
        dict with keys:
            report (CalibrationResult dict)
            model (pickled model)
            feature_names (list)
            feature_weights (dict)
            p_rule (dict)
            pattern_rules (list)
            memory_db (pd.DataFrame or None)
        或者 None 如果 calibrated 目录不存在
    """
    calibrated_dir = Path(calibrated_dir)
    if not calibrated_dir.exists():
        return None
    
    # 检查版本
    version_file = calibrated_dir / 'CALIBRATION_VERSION'
    if not version_file.exists():
        # 旧版 calibration，不加载
        return None
    
    result = {}
    
    # report
    report_path = calibrated_dir / 'calibration_report.json'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            result['report'] = json.load(f)
    
    # model
    model_path = calibrated_dir / 'classifier_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            result['model_bundle'] = pickle.load(f)
    
    # feature_weights
    fw_path = calibrated_dir / 'feature_weights.json'
    if fw_path.exists():
        with open(fw_path, 'r', encoding='utf-8') as f:
            result['feature_weights'] = json.load(f)
    
    # p_rule
    p_path = calibrated_dir / 'p_rule.json'
    if p_path.exists():
        with open(p_path, 'r', encoding='utf-8') as f:
            result['p_rule'] = json.load(f)
    
    # pattern_rules
    pr_path = calibrated_dir / 'pattern_rules.json'
    if pr_path.exists():
        with open(pr_path, 'r', encoding='utf-8') as f:
            result['pattern_rules'] = json.load(f)
    
    # memory_db
    mem_path = calibrated_dir / 'memory_db.parquet'
    if mem_path.exists():
        try:
            result['memory_db'] = pd.read_parquet(mem_path)
        except Exception:
            result['memory_db'] = None
    
    return result if len(result) > 0 else None


# ============================================================================
# 快速测试入口
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    from src.ground_truth import load_and_map_ground_truth
    from src.text_feature_extractor import batch_extract_text_features
    
    xlsx = "/workspace/.uploads/7b014c53-ac0c-4044-a154-ad2082eed224_60款回测1.xlsx"
    
    print("加载数据...")
    df = load_and_map_ground_truth(xlsx)
    text_feats = batch_extract_text_features(df['design_text'].tolist())
    df = pd.concat([df.reset_index(drop=True), text_feats], axis=1)
    
    print(f"运行 calibration...")
    result = run_calibration(df, brand_name='mipo')
    
    print(f"\n{'='*60}")
    print(f"Calibration 完成!")
    print(f"{'='*60}")
    print(f"  样本: {result.total_samples}, 爆款: {result.total_bombs}, P款: {result.total_p_styles}")
    print(f"  Baseline [{result.baseline_method}]:")
    print(f"    AUC={result.baseline_auc:.3f}, P@10={result.baseline_precision_at_10:.1%}, R@10={result.baseline_recall_at_10:.1%}")
    print(f"  校准后 [{result.calibrated_model}]:")
    print(f"    CV AUC={result.calibrated_cv_auc_mean:.3f} ± {result.calibrated_cv_auc_std:.3f}")
    print(f"    CV AP={result.calibrated_cv_ap_mean:.3f} ± {result.calibrated_cv_ap_std:.3f}")
    print(f"  全量训练排序:")
    print(f"    Top-10 Recall={result.full_train_recall_at_10:.0%}, Precision={result.full_train_precision_at_10:.0%}")
    print(f"    Top-10 爆款: {result.full_train_top_10_bombs}")
    print(f"  阈值: S={result.s_threshold}, A+={result.aplus_threshold}")
    print(f"\n  Top 5 特征重要性:")
    for item in result.feature_importance[:5]:
        print(f"    {item['feature']}: {item['importance']:.3f}")
    print(f"\n  P款规则:")
    for r in result.p_rule.get('rules', []):
        print(f"    {r['feature']} {r['op']} {r['threshold']}")
    print(f"\n  产物目录: brand_profiles/mipo/calibrated/")
