"""优化内核：三阶校准循环 + 残差分解

Loop1 VLMFeatureCalibrator  —— 10个VLM特征偏置系数
Loop2 PersonaDistributionFitter —— 30人设分布权重（Lasso）
Loop3 EnsembleWeightTuner   —— 三大引擎权重 & 双渠道权重
ResidualDecomposer          —— 残差诊断（超预期/不及预期/系统偏差）
run_all_loops               —— 主入口：顺序跑四步 + 落盘YAML/Markdown报告
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ============================================================
# Loop1: VLM 特征偏置校准
# ============================================================
@dataclass
class VLMFeatureCalibrationResult:
    feature_biases: dict[str, float]
    old_spearman_avg: float
    new_spearman_avg: float
    per_feature_rho: dict[str, float]
    applied: bool
    # v1.4.43+: 新增分类指标（向后兼容）
    old_f1: float = 0.0
    new_f1: float = 0.0
    old_p_at_k: float = 0.0
    new_p_at_k: float = 0.0
    # v1.4.48+: 稳定性过滤 + safe mode
    safe_mode_used: bool = False
    feature_stability: dict[str, float] = field(default_factory=dict)   # 每特征稳定分
    unstable_features: list[str] = field(default_factory=list)          # 被判不稳定的特征（bias=1.0）


class VLMFeatureCalibrator:
    """对 F01-F10 每个VLM特征计算与销量的 Spearman 秩相关，
    相对全局均值做偏置，范围 [0.7, 1.3]。若整体秩相关不提升，
    保护机制返回全 1.0。
    """

    FEATURE_COLS = [f"F{i:02d}" for i in range(1, 11)]
    BIAS_LOW = 0.7
    BIAS_HIGH = 1.3

    @classmethod
    def calibrate(
        cls,
        df: pd.DataFrame,
        sales_col: str = "sales",
        *,
        safe_mode: bool = False,
        stability_threshold: float = 0.04,
    ) -> VLMFeatureCalibrationResult:
        """
        Args:
            df: DataFrame，必须包含 10 列 F01..F10 + 销量列
            sales_col: 销量列名（默认 "sales"）
            safe_mode: v1.4.48+ — True 时启用稳定性过滤：
                1) 先按时间序切 train(前80%)/test(后20%)
                2) 每特征算 train_ρ 和 test_ρ
                3) 稳定性 = |train_ρ × test_ρ|，或 train/test 反号
                4) 不稳定特征 bias 强制 = 1.0
                5) bias 范围收窄到 [0.85, 1.15]（更保守）
            stability_threshold: v1.4.48+ — 稳定分低于此阈值视为不稳定
        Returns:
            VLMFeatureCalibrationResult，含偏置系数 & 是否生效
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            raise RuntimeError(
                "scipy 未安装，无法计算 Spearman 秩相关。"
                "请执行: pip install scipy"
            )

        try:
            required = cls.FEATURE_COLS + [sales_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"df 缺少必需列: {missing}")
            if df.empty or len(df) < 3:
                raise ValueError(f"样本量不足（{len(df)}<3），无法做秩相关")

            df_work = df[required].dropna().copy()
            if len(df_work) < 3:
                raise ValueError(f"去NaN后样本量不足（{len(df_work)}<3）")

            y = df_work[sales_col].values

            # --- 计算旧（无偏置）时的每特征 Spearman，均值作为基准 ---
            per_feature_rho: dict[str, float] = {}
            old_rhos = []
            for col in cls.FEATURE_COLS:
                try:
                    rho, _ = spearmanr(df_work[col].values, y)
                    rho = 0.0 if math.isnan(rho) else float(rho)
                except Exception:
                    rho = 0.0
                per_feature_rho[col] = rho
                old_rhos.append(rho)
            old_avg = statistics.mean(old_rhos) if old_rhos else 0.0

            # === v1.4.48+: safe_mode 稳定性过滤 ===
            feature_stability: dict[str, float] = {}
            unstable_features: list[str] = []
            _bias_low = cls.BIAS_LOW
            _bias_high = cls.BIAS_HIGH

            if safe_mode and len(df_work) >= 10:
                n_split = max(5, int(len(df_work) * 0.8))
                df_tr = df_work.iloc[:n_split]
                df_te = df_work.iloc[n_split:]
                y_tr = df_tr[sales_col].values
                y_te = df_te[sales_col].values
                for col in cls.FEATURE_COLS:
                    try:
                        r_tr, _ = spearmanr(df_tr[col].values, y_tr)
                        r_te, _ = spearmanr(df_te[col].values, y_te)
                        r_tr = 0.0 if math.isnan(r_tr) else float(r_tr)
                        r_te = 0.0 if math.isnan(r_te) else float(r_te)
                    except Exception:
                        r_tr = r_te = 0.0
                    stability = abs(r_tr * r_te)
                    feature_stability[col] = stability
                    if r_tr * r_te < 0 or stability < stability_threshold:
                        unstable_features.append(col)
                if unstable_features:
                    log.info(
                        "Loop1 safe_mode: %d/%d 特征不稳定 → bias=1.0: %s",
                        len(unstable_features), len(cls.FEATURE_COLS),
                        ", ".join(unstable_features),
                    )
                _bias_low = 0.85
                _bias_high = 1.15

            # --- 偏置：相对全局均值，ρ_i / mean(ρ)，clamp ---
            eps = 1e-9
            biases: dict[str, float] = {}
            safe_avg = old_avg if abs(old_avg) > eps else 1e-6
            for col in cls.FEATURE_COLS:
                if safe_mode and col in unstable_features:
                    biases[col] = 1.0
                    continue
                raw = per_feature_rho[col] / safe_avg
                biases[col] = max(_bias_low, min(_bias_high, raw))

            # --- 校验：应用偏置后的加权分 vs y 的 Spearman 是否 >= 旧 ---
            def _weighted_score(rho_map: dict[str, float], bias_map: dict[str, float]) -> list[float]:
                scores = []
                for _, row in df_work.iterrows():
                    s = 0.0
                    for col in cls.FEATURE_COLS:
                        s += float(row[col]) * bias_map.get(col, 1.0) * max(0.0, rho_map.get(col, 0.0) + 0.3)
                    scores.append(s)
                return scores

            old_score = _weighted_score(per_feature_rho, {c: 1.0 for c in cls.FEATURE_COLS})
            new_score = _weighted_score(per_feature_rho, biases)

            try:
                old_sp, _ = spearmanr(old_score, y)
                new_sp, _ = spearmanr(new_score, y)
                old_sp = 0.0 if math.isnan(old_sp) else float(old_sp)
                new_sp = 0.0 if math.isnan(new_sp) else float(new_sp)
            except Exception:
                old_sp, new_sp = old_avg, old_avg

            # === v1.4.43+: 同时计算分类指标 ===
            y_pct = _rank_percentile(y)
            y_true_labels = _build_grade_labels_from_percentile(y_pct)
            old_cls = _classification_metrics(y_true_labels, old_score)
            new_cls = _classification_metrics(y_true_labels, new_score)
            log.info(
                "Loop1 分类: old F1=%.3f P@3=%.3f → new F1=%.3f P@3=%.3f",
                old_cls["macro_f1"], old_cls["precision_at_k"],
                new_cls["macro_f1"], new_cls["precision_at_k"],
            )

            applied = new_sp > old_sp + 1e-9
            if not applied:
                biases = {c: 1.0 for c in cls.FEATURE_COLS}
                new_sp = old_sp
                # 保护触发 → new_cls 也用 old（因为 score 被回滚了）
                new_cls = old_cls
                log.info("Loop1 保护触发：新Spearman(%.4f)未优于旧(%.4f)，返回全1偏置", new_sp, old_sp)
            else:
                log.info("Loop1 生效：Spearman %.4f → %.4f", old_sp, new_sp)

            # 质量诊断：若所有特征与销量都弱相关，说明 VLM 输出没有区分度
            avg_abs_rho = statistics.mean(abs(v) for v in per_feature_rho.values())
            if avg_abs_rho < 0.20:
                log.warning(
                    "Loop1 质量警告：10个VLM特征与销量的平均绝对Spearman仅%.3f，"
                    "说明当前VLM输出和真实销量几乎没有线性关系。建议检查："
                    "① 图片是否正确传入 ② VLM prompt是否导向销量预测 ③ 销量列是否准确。",
                    avg_abs_rho,
                )

            return VLMFeatureCalibrationResult(
                feature_biases=biases,
                old_spearman_avg=float(old_avg),
                new_spearman_avg=float(new_sp),
                per_feature_rho=per_feature_rho,
                applied=applied,
                old_f1=old_cls["macro_f1"],
                new_f1=new_cls["macro_f1"],
                old_p_at_k=old_cls["precision_at_k"],
                new_p_at_k=new_cls["precision_at_k"],
                safe_mode_used=safe_mode,
                feature_stability=feature_stability,
                unstable_features=unstable_features,
            )
        except Exception as exc:
            log.exception("VLMFeatureCalibrator 失败，降级全1偏置: %s", exc)
            return VLMFeatureCalibrationResult(
                feature_biases={c: 1.0 for c in cls.FEATURE_COLS},
                old_spearman_avg=0.0,
                new_spearman_avg=0.0,
                per_feature_rho={c: 0.0 for c in cls.FEATURE_COLS},
                applied=False,
                safe_mode_used=False,
            )


# ============================================================
# Loop2: 人设分布权重拟合（Lasso）
# ============================================================
@dataclass
class PersonaDistributionFitResult:
    persona_weights: dict[str, float]
    old_spearman: float
    new_spearman: float
    lasso_raw_coef: dict[str, float]
    applied: bool
    # v1.4.43+: 新增分类指标
    old_f1: float = 0.0
    new_f1: float = 0.0
    old_p_at_k: float = 0.0
    new_p_at_k: float = 0.0


class PersonaDistributionFitter:
    """对 P01..P30 人设列用 Lasso(alpha=0.05, max_iter=5000) 拟合销量（spec §8.2）。

    数学形式（spec §8.2）::

        min_w  Σ_i (y_i - Σ_k w_k · vote_k(x_i))² + λ · Σ_k |w_k|
        s.t.   Σ_k w_k = 1
               w_k ≥ w_min = 1/(2×30)        # 多样性保护下限 ≈1.67%

    其中：
      - y_i          = 真实销量排名（0-1 rank percentile，spec §9.1）
      - vote_k(x_i)  = 人设投票分（1-10 量表 → /10 → [0,1]）
      - 系数方向性   = 保留 Lasso 符号；正相关人设获高权重，
                       负相关人设降至 0（最终被 w_min floor 抬起），
                       不再使用 abs()——避免把负相关人设误判为"强相关"

    保护机制（spec §8.4）：新 Spearman ≥ 旧+MIN_IMPROVEMENT 才接受，
    否则回滚到均匀分布。
    """

    PERSONA_COLS = [f"P{i:02d}" for i in range(1, 31)]
    MIN_IMPROVEMENT = 0.01
    W_MIN = 1.0 / (2 * len(PERSONA_COLS))   # spec §8.2: ≈1.67%
    VOTE_SCALE = 10.0                        # 人设分 1-10 → /10 → [0,1]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        sales_col: str = "sales",
        alpha: float | None = None,
        max_iter: int = 5000,
    ) -> PersonaDistributionFitResult:
        """
        Args:
            df: DataFrame，含 30 列 P01..P30 + 销量列
            sales_col: 销量列名
            alpha: Lasso 正则强度。
                None  → 用 LassoCV(cv=5) 自动选（推荐，n≥25 时最稳）
                float → 直接用指定值（兼容旧调用；n<25 时 fallback 到 alpha=0.05）
            max_iter: Lasso 最大迭代（默认 5000）

        改动说明（v1.4.39）：之前硬编码 alpha=0.05 是 n≤10 样本下防 Lasso
        系数归零的保守值。当 n=40-60（甜点区）时 0.05 太松，会让弱信号人设
        也混进结果 → 校准后权重区分度下降 30%+。改为 LassoCV 5 折交叉验证
        自动选 alpha，在 n≥25 时稳定选出最优值。
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            raise RuntimeError(
                "scipy 未安装，无法计算 Spearman。请执行: pip install scipy"
            )

        try:
            try:
                import numpy as np
                from sklearn.linear_model import Lasso, LassoCV
            except ImportError:
                raise RuntimeError(
                    "scikit-learn 未安装，无法运行 Lasso 拟合。"
                    "请执行: pip install scikit-learn"
                )

            required = cls.PERSONA_COLS + [sales_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"df 缺少必需列: {missing}")

            df_work = df[required].dropna().copy()
            n_p = len(cls.PERSONA_COLS)
            n_samples = len(df_work)
            if n_samples < 5:
                raise ValueError(f"样本量不足（{n_samples}<5），无法拟合Lasso")

            # --- spec §8.2/§9.1 归一化 ---
            X = df_work[cls.PERSONA_COLS].values.astype(float) / cls.VOTE_SCALE
            y = _rank_percentile(df_work[sales_col]).values.astype(float)

            # --- 旧：均匀权重加权分 vs y 的 Spearman ---
            uniform_score = X.mean(axis=1)
            try:
                old_sp, _ = spearmanr(uniform_score, y)
                old_sp = 0.0 if math.isnan(old_sp) else float(old_sp)
            except Exception:
                old_sp = 0.0

            # --- Lasso 拟合（v1.4.39: LassoCV 自动选 alpha）---
            lasso_alpha = alpha
            try:
                if lasso_alpha is None and n_samples >= 25:
                    # LassoCV: 5 折交叉验证自动选 alpha
                    # 30 个候选 alpha（logspace -3 到 1，即 0.001~10）
                    cv = LassoCV(
                        alphas=np.logspace(-3, 1, 30),
                        cv=min(5, max(2, n_samples // 12)),   # n=60→5, n=36→3, n=24→2
                        max_iter=max_iter,
                        random_state=42,
                    )
                    cv.fit(X, y)
                    lasso_alpha = float(cv.alpha_)
                    log.info(
                        "Loop2 LassoCV 自动选 alpha=%.4f (n=%d, cv=%d折)",
                        lasso_alpha, n_samples, cv.cv,
                    )
                    raw_coef = {col: float(v) for col, v in zip(cls.PERSONA_COLS, cv.coef_)}
                else:
                    # 显式指定 alpha 或 n<25 时的 fallback
                    if lasso_alpha is None:
                        lasso_alpha = 0.05  # 历史保守值，小样本防归零
                    model = Lasso(alpha=lasso_alpha, max_iter=max_iter, random_state=42)
                    model.fit(X, y)
                    raw_coef = {col: float(v) for col, v in zip(cls.PERSONA_COLS, model.coef_)}
                    log.info("Loop2 Lasso 直接拟合 alpha=%.4f (n=%d)", lasso_alpha, n_samples)
            except Exception as exc:
                log.warning("Lasso 拟合异常，降级均匀权重: %s", exc)
                raw_coef = {col: 1.0 for col in cls.PERSONA_COLS}

            # --- 保留方向性：正相关人设获高权重，负相关人设降至 0 ---
            weights = cls._weights_from_raw_coef(raw_coef)

            # --- 新加权分 vs y 的 Spearman ---
            weight_arr = np.array([weights[col] for col in cls.PERSONA_COLS], dtype=float)
            new_score = X @ weight_arr
            try:
                new_sp, _ = spearmanr(new_score, y)
                new_sp = 0.0 if math.isnan(new_sp) else float(new_sp)
            except Exception:
                new_sp = old_sp

            # === v1.4.43+: 同时计算分类指标 ===
            old_cls = _classification_metrics(
                _build_grade_labels_from_percentile(y), list(uniform_score))
            new_cls = _classification_metrics(
                _build_grade_labels_from_percentile(y), list(new_score))
            log.info(
                "Loop2 分类: old F1=%.3f P@3=%.3f → new F1=%.3f P@3=%.3f",
                old_cls["macro_f1"], old_cls["precision_at_k"],
                new_cls["macro_f1"], new_cls["precision_at_k"],
            )

            applied = new_sp >= old_sp + cls.MIN_IMPROVEMENT - 1e-9
            if not applied:
                # spec §8.4: 回滚到更新前状态（均匀分布）
                weights = {col: 1.0 / n_p for col in cls.PERSONA_COLS}
                new_cls = old_cls
                log.info(
                    "Loop2 保护触发：新Spearman(%.4f) - 旧(%.4f) = %.4f < %.2f，回滚均匀",
                    new_sp, old_sp, new_sp - old_sp, cls.MIN_IMPROVEMENT,
                )
            else:
                log.info(
                    "Loop2 生效：Spearman %.4f → %.4f (Δ=%.4f)",
                    old_sp, new_sp, new_sp - old_sp,
                )

            return PersonaDistributionFitResult(
                persona_weights=weights,
                old_spearman=float(old_sp),
                new_spearman=float(new_sp),
                lasso_raw_coef=raw_coef,
                applied=applied,
                old_f1=old_cls["macro_f1"],
                new_f1=new_cls["macro_f1"],
                old_p_at_k=old_cls["precision_at_k"],
                new_p_at_k=new_cls["precision_at_k"],
            )
        except Exception as exc:
            log.exception("PersonaDistributionFitter 失败，降级均匀权重: %s", exc)
            uniform = {c: 1.0 / len(cls.PERSONA_COLS) for c in cls.PERSONA_COLS}
            return PersonaDistributionFitResult(
                persona_weights=uniform,
                old_spearman=0.0,
                new_spearman=0.0,
                lasso_raw_coef={c: 0.0 for c in cls.PERSONA_COLS},
                applied=False,
            )

    @classmethod
    def _weights_from_raw_coef(
        cls,
        raw_coef: dict[str, float],
    ) -> dict[str, float]:
        """从 Lasso 原始系数计算最终权重（spec §8.2 数学合规）。

        实现：
          1. 保留方向性：负系数 → pos_coef=0（不进入权重）；
             正系数 → 保留。spec §8.2 要求 w_k ≥ w_min > 0，
             所以负相关人设反向预测，不能进入权重。
             （旧 bug 用 abs() 把负相关误判为"强相关"调高，违反 spec 语义）
          2. 正系数归一化：sum(pos_coef) = 1
          3. spec §8.2 下限约束：w_k = w_min + pool · normalized[col]
             其中 pool = 1 − w_min·n，保证 sum(w) = 1 且每个 w_k ≥ w_min
          4. 浮点修正：重新归一化到严格 sum=1

        该方法独立可测，与 Lasso 拟合解耦。
        """
        n_p = len(cls.PERSONA_COLS)
        pos_coef = {col: max(0.0, v) for col, v in raw_coef.items()}
        total = sum(pos_coef.values())

        if total < 1e-12:
            # 所有系数 ≤ 0：均匀分布（仍满足 w_k ≥ w_min，因 1/n > 1/(2n)）
            return {col: 1.0 / n_p for col in cls.PERSONA_COLS}

        # 1. 正系数归一化到 [0,1]，sum=1
        norm = {col: v / total for col, v in pos_coef.items()}
        # 2. spec §8.2 下限约束：w_k = w_min + pool · normalized[col]
        #    其中 pool = 1 − w_min·n（≈0.5），保证 sum=1 且每项 ≥ w_min
        pool = 1.0 - cls.W_MIN * n_p
        if pool <= 0:
            # 边界：w_min·n ≥ 1，退化为均匀
            return {col: 1.0 / n_p for col in cls.PERSONA_COLS}
        weights = {
            col: cls.W_MIN + pool * norm[col]
            for col in cls.PERSONA_COLS
        }
        # 浮点修正：重新归一化到严格 sum=1
        total_w = sum(weights.values())
        return {col: w / total_w for col, w in weights.items()}


# ============================================================
# Loop3: 集成权重调优（三大引擎 + 双渠道）
# ============================================================
@dataclass
class EnsembleTuneResult:
    engine_weights: dict[str, float]
    channel_weights: dict[str, float]
    old_engine_spearman: float
    new_engine_spearman: float
    old_channel_spearman: float
    new_channel_spearman: float
    engine_rho: dict[str, float]
    channel_rho: dict[str, float]
    applied: bool
    # v1.4.43+: 新增分类指标
    old_engine_f1: float = 0.0
    new_engine_f1: float = 0.0
    old_engine_p_at_k: float = 0.0
    new_engine_p_at_k: float = 0.0
    old_chan_f1: float = 0.0
    new_chan_f1: float = 0.0
    old_chan_p_at_k: float = 0.0
    new_chan_p_at_k: float = 0.0


class EnsembleWeightTuner:
    """对三大引擎（persona / channel / price_value）和双渠道
    （natural / live）分别计算 Spearman，权重 ∝ max(0.05, ρ+0.3)
    归一化。保护机制同前。

    v1.4.49+: 如果 df 里有 grade_norm 列（从内审分级归一化来的 [0,100]），
    自动作为第 4 个引擎。grade_norm 的 Spearman 通常在 0.7+，远高于
    VLM 引擎的 0.1-0.2，会被自动赋予最大权重。
    没有 grade_norm 的新款式预测阶段，用 3 个基础引擎 fallback。
    """

    BASE_ENGINE_COLS = ["persona_score", "channel_score", "price_value_score"]
    CHANNEL_COLS = ["natural_score", "live_score"]
    FLOOR = 0.05
    SHIFT = 0.3

    @classmethod
    def _resolve_engine_cols(cls, df: pd.DataFrame) -> list[str]:
        """v1.4.49+: 动态解析引擎列 — 有 grade_norm 就加进引擎列表"""
        cols = list(cls.BASE_ENGINE_COLS)
        if "grade_norm" in df.columns and df["grade_norm"].notna().sum() >= 3:
            cols.append("grade_norm")
        return cols

    @classmethod
    def tune(
        cls,
        df: pd.DataFrame,
        sales_col: str = "sales",
    ) -> EnsembleTuneResult:
        """
        Args:
            df: 必须含 persona_score/channel_score/price_value_score
                及 natural_score/live_score + 销量列
                v1.4.49+: 可选 grade_norm（内审分级归一化 [0,100]），
                有就当第 4 引擎 — 历史校准阶段用，预测阶段没有就跳过
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            raise RuntimeError(
                "scipy 未安装，无法计算 Spearman。请执行: pip install scipy"
            )

        try:
            engine_cols = cls._resolve_engine_cols(df)
            required = cls.BASE_ENGINE_COLS + cls.CHANNEL_COLS + [sales_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"df 缺少必需列: {missing}")

            # 把动态引擎列也纳入 df_work（含可选的 grade_norm）
            work_cols = list(set(engine_cols + cls.CHANNEL_COLS + [sales_col]))
            df_work = df[work_cols].dropna(subset=cls.BASE_ENGINE_COLS + [sales_col]).copy()
            if len(df_work) < 3:
                raise ValueError(f"样本量不足（{len(df_work)}<3）")

            y = df_work[sales_col].values

            # --- 旧：引擎均匀权重 ---
            engine_rho: dict[str, float] = {}
            for col in engine_cols:
                try:
                    rho, _ = spearmanr(df_work[col].values, y)
                    rho = 0.0 if math.isnan(rho) else float(rho)
                except Exception:
                    rho = 0.0
                engine_rho[col] = rho

            # v1.4.49+: grade_norm 自动加入引擎时记日志
            if "grade_norm" in engine_cols:
                log.info(
                    "Loop3 引擎: grade_norm ρ=%.3f 加入引擎（共 %d 个，含内审分级）",
                    engine_rho.get("grade_norm", 0.0), len(engine_cols),
                )

            uniform_engine = {c: 1.0 / len(engine_cols) for c in engine_cols}
            old_engine_score = sum(
                df_work[c].values * uniform_engine[c] for c in engine_cols
            )
            try:
                old_engine_sp, _ = spearmanr(old_engine_score, y)
                old_engine_sp = 0.0 if math.isnan(old_engine_sp) else float(old_engine_sp)
            except Exception:
                old_engine_sp = 0.0

            # --- 新引擎权重 ∝ max(0.05, ρ+0.3) ---
            raw_engine = {c: max(cls.FLOOR, engine_rho[c] + cls.SHIFT) for c in engine_cols}
            tot = sum(raw_engine.values())
            new_engine_weights = {c: v / tot for c, v in raw_engine.items()}

            new_engine_score = sum(
                df_work[c].values * new_engine_weights[c] for c in engine_cols
            )
            try:
                new_engine_sp, _ = spearmanr(new_engine_score, y)
                new_engine_sp = 0.0 if math.isnan(new_engine_sp) else float(new_engine_sp)
            except Exception:
                new_engine_sp = old_engine_sp

            engine_applied = new_engine_sp > old_engine_sp + 1e-9
            # === v1.4.43+: 引擎分类指标 ===
            y_pct = _rank_percentile(y)
            y_true_labels = _build_grade_labels_from_percentile(y_pct)
            old_eng_cls = _classification_metrics(y_true_labels, old_engine_score)
            new_eng_cls = _classification_metrics(y_true_labels, new_engine_score)
            log.info(
                "Loop3 引擎分类: old F1=%.3f P@3=%.3f → new F1=%.3f P@3=%.3f",
                old_eng_cls["macro_f1"], old_eng_cls["precision_at_k"],
                new_eng_cls["macro_f1"], new_eng_cls["precision_at_k"],
            )
            if not engine_applied:
                new_engine_weights = uniform_engine
                new_engine_sp = old_engine_sp
                new_eng_cls = old_eng_cls
                log.info("Loop3 引擎保护：新Spearman(%.4f)未提升，保持均匀", new_engine_sp)
            else:
                log.info("Loop3 引擎生效：Spearman %.4f → %.4f", old_engine_sp, new_engine_sp)

            # --- 双渠道部分 ---
            channel_rho: dict[str, float] = {}
            for col in cls.CHANNEL_COLS:
                try:
                    rho, _ = spearmanr(df_work[col].values, y)
                    rho = 0.0 if math.isnan(rho) else float(rho)
                except Exception:
                    rho = 0.0
                channel_rho[col] = rho

            uniform_channel = {c: 1.0 / len(cls.CHANNEL_COLS) for c in cls.CHANNEL_COLS}
            old_chan_score = (
                df_work[cls.CHANNEL_COLS[0]] * uniform_channel[cls.CHANNEL_COLS[0]]
                + df_work[cls.CHANNEL_COLS[1]] * uniform_channel[cls.CHANNEL_COLS[1]]
            ).values
            try:
                old_chan_sp, _ = spearmanr(old_chan_score, y)
                old_chan_sp = 0.0 if math.isnan(old_chan_sp) else float(old_chan_sp)
            except Exception:
                old_chan_sp = 0.0

            raw_chan = {c: max(cls.FLOOR, channel_rho[c] + cls.SHIFT) for c in cls.CHANNEL_COLS}
            tot_chan = sum(raw_chan.values())
            new_chan_weights = {c: v / tot_chan for c, v in raw_chan.items()}

            new_chan_score = (
                df_work[cls.CHANNEL_COLS[0]] * new_chan_weights[cls.CHANNEL_COLS[0]]
                + df_work[cls.CHANNEL_COLS[1]] * new_chan_weights[cls.CHANNEL_COLS[1]]
            ).values
            try:
                new_chan_sp, _ = spearmanr(new_chan_score, y)
                new_chan_sp = 0.0 if math.isnan(new_chan_sp) else float(new_chan_sp)
            except Exception:
                new_chan_sp = old_chan_sp

            chan_applied = new_chan_sp > old_chan_sp + 1e-9
            # === v1.4.43+: 渠道分类指标 ===
            old_chan_cls = _classification_metrics(y_true_labels, old_chan_score)
            new_chan_cls = _classification_metrics(y_true_labels, new_chan_score)
            log.info(
                "Loop3 渠道分类: old F1=%.3f P@3=%.3f → new F1=%.3f P@3=%.3f",
                old_chan_cls["macro_f1"], old_chan_cls["precision_at_k"],
                new_chan_cls["macro_f1"], new_chan_cls["precision_at_k"],
            )
            if not chan_applied:
                new_chan_weights = uniform_channel
                new_chan_sp = old_chan_sp
                log.info("Loop3 渠道保护：新Spearman(%.4f)未提升，保持均匀", new_chan_sp)
            else:
                log.info("Loop3 渠道生效：Spearman %.4f → %.4f", old_chan_sp, new_chan_sp)

            return EnsembleTuneResult(
                engine_weights=new_engine_weights,
                channel_weights=new_chan_weights,
                old_engine_spearman=float(old_engine_sp),
                new_engine_spearman=float(new_engine_sp),
                old_channel_spearman=float(old_chan_sp),
                new_channel_spearman=float(new_chan_sp),
                engine_rho=engine_rho,
                channel_rho=channel_rho,
                applied=engine_applied or chan_applied,
                old_engine_f1=old_eng_cls["macro_f1"],
                new_engine_f1=new_eng_cls["macro_f1"],
                old_engine_p_at_k=old_eng_cls["precision_at_k"],
                new_engine_p_at_k=new_eng_cls["precision_at_k"],
                old_chan_f1=old_chan_cls["macro_f1"],
                new_chan_f1=new_chan_cls["macro_f1"],
                old_chan_p_at_k=old_chan_cls["precision_at_k"],
                new_chan_p_at_k=new_chan_cls["precision_at_k"],
            )
        except Exception as exc:
            log.exception("EnsembleWeightTuner 失败，降级均匀权重: %s", exc)
            uniform_e = {c: 1.0 / len(cls.BASE_ENGINE_COLS) for c in cls.BASE_ENGINE_COLS}
            uniform_c = {c: 1.0 / len(cls.CHANNEL_COLS) for c in cls.CHANNEL_COLS}
            return EnsembleTuneResult(
                engine_weights=uniform_e,
                channel_weights=uniform_c,
                old_engine_spearman=0.0,
                new_engine_spearman=0.0,
                old_channel_spearman=0.0,
                new_channel_spearman=0.0,
                engine_rho={c: 0.0 for c in cls.BASE_ENGINE_COLS},
                channel_rho={c: 0.0 for c in cls.CHANNEL_COLS},
                applied=False,
            )


# ============================================================
# 残差分解器
# ============================================================
@dataclass
class ResidualDecomposeResult:
    residual_mean: float
    residual_std: float
    overperformers: list[dict[str, Any]]
    underperformers: list[dict[str, Any]]
    system_bias_flag: str
    residuals: list[float]
    # 资源错配归因（ADR-0001：把「款式好」和「被推爆」分开）
    # 仅当 meta_df 带 is_main_push / is_live_stream 列时才有意义
    marketing_available: bool = False
    attribution_summary: dict[str, int] = field(default_factory=dict)


class ResidualDecomposer:
    """残差 ε = y_true - y_pred 分解：
    - 均值/std
    - 超预期款（ε > +2σ）
    - 不及预期款（ε < -2σ）
    - 系统偏差flag（mean/std 的量级判断）
    - 资源错配归因（若 meta_df 带营销投放列，ADR-0001）
    """

    # 四象限归因：超额款(ε>+2σ) / 不及款(ε<-2σ) × 投放(主推/直播) / 未投放
    ATTRIB_OVER_PUSHED = "推对了：投放放大了款式潜力"
    ATTRIB_OVER_ORGANIC = "漏网爆款：没推也超预期，应追加投放"
    ATTRIB_UNDER_PUSHED = "资源错配：投放了仍不及预期"
    ATTRIB_UNDER_ORGANIC = "款式本身弱：未投放且不及预期"

    @classmethod
    def decompose(
        cls,
        y_true: list[float] | pd.Series,
        y_pred: list[float] | pd.Series,
        meta_df: pd.DataFrame,
    ) -> ResidualDecomposeResult:
        """
        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            meta_df: 每行对应款式元信息，行顺序必须与 y_true/y_pred 对齐；
                     至少包含 style_id 列（没有则用行索引代替）
        """
        try:
            import numpy as np

            yt = np.asarray(list(y_true), dtype=float)
            yp = np.asarray(list(y_pred), dtype=float)
            if yt.ndim != 1 or yp.ndim != 1 or len(yt) != len(yp):
                raise ValueError(
                    f"y_true/y_pred 形状不匹配或非1维: {yt.shape} vs {yp.shape}"
                )
            if len(yt) < 2:
                raise ValueError("样本数需 ≥2 才能计算残差统计")

            eps = (yt - yp).tolist()
            mu = statistics.mean(eps)
            sigma = statistics.pstdev(eps) if len(eps) > 1 else 0.0

            # --- 阈值修正（硬 bug fix）---
            # rank_percentile 残差 ε ∈ [-1, 1]，小样本时 σ 天然偏大，
            # ±2σ 会超出 [-1, 1] 实际范围 → 即使完全反排也 0 个触发。
            # 修：(1) 截断到 ε 的物理边界 ±1.0
            #     (2) 小样本（n<20）收紧到 1.5σ
            #     (3) σ 极小（<0.1，近乎完美排序）时用硬阈值 ±0.3 兜底
            n_eps = len(eps)
            sigma_mult = 1.5 if n_eps < 20 else 2.0
            if sigma < 0.1:
                # 近乎完美排序 → 用硬阈值
                upper = min(mu + 0.3, 1.0)
                lower = max(mu - 0.3, -1.0)
            else:
                upper = min(mu + sigma_mult * sigma, 1.0) if sigma > 0 else mu
                lower = max(mu - sigma_mult * sigma, -1.0) if sigma > 0 else mu

            id_col = "style_id" if "style_id" in meta_df.columns else None
            # 营销投放列（ADR-0001：实际投放，缺失时归因跳过）
            mkt_cols = [
                c for c in ("is_main_push", "is_live_stream")
                if c in meta_df.columns
            ]
            marketing_available = len(mkt_cols) > 0

            over: list[dict[str, Any]] = []
            under: list[dict[str, Any]] = []

            n = min(len(meta_df), len(eps))
            for i in range(n):
                e = eps[i]
                sid = (
                    str(meta_df.iloc[i][id_col])
                    if id_col is not None
                    else f"row_{i}"
                )
                item: dict[str, Any] = {"style_id": sid, "residual": float(e), "index": i}
                # 营销投放标签（有列才带）
                if marketing_available:
                    pushed = any(
                        float(meta_df.iloc[i][c]) > 0.5 for c in mkt_cols
                    )
                    item["was_pushed"] = pushed
                is_over = e > upper
                is_under = e < lower
                if is_over or is_under:
                    try:
                        y_t = float(yt[i])
                        y_p = float(yp[i])
                    except Exception:
                        y_t = y_p = None
                    item.update({"y_true": y_t, "y_pred": y_p})
                # 四象限归因
                if marketing_available and (is_over or is_under):
                    if is_over:
                        item["attribution"] = (
                            cls.ATTRIB_OVER_PUSHED if item["was_pushed"]
                            else cls.ATTRIB_OVER_ORGANIC
                        )
                    else:
                        item["attribution"] = (
                            cls.ATTRIB_UNDER_PUSHED if item["was_pushed"]
                            else cls.ATTRIB_UNDER_ORGANIC
                        )
                if is_over:
                    over.append(item)
                elif is_under:
                    under.append(item)

            # --- 系统偏差flag ---
            if sigma < 1e-9:
                flag = "SIGMA_ZERO"
            else:
                ratio = abs(mu) / sigma
                if ratio > 1.5:
                    flag = "STRONG_SYSTEM_BIAS"
                elif ratio > 0.75:
                    flag = "MODERATE_SYSTEM_BIAS"
                else:
                    flag = "NO_SIGNIFICANT_BIAS"

            over.sort(key=lambda r: r["residual"], reverse=True)
            under.sort(key=lambda r: r["residual"])

            # 归因汇总（四象限计数）
            attribution_summary: dict[str, int] = {}
            if marketing_available:
                for item in over + under:
                    k = item.get("attribution", "")
                    if k:
                        attribution_summary[k] = attribution_summary.get(k, 0) + 1

            return ResidualDecomposeResult(
                residual_mean=float(mu),
                residual_std=float(sigma),
                overperformers=over,
                underperformers=under,
                system_bias_flag=flag,
                residuals=eps,
                marketing_available=marketing_available,
                attribution_summary=attribution_summary,
            )
        except Exception as exc:
            log.exception("ResidualDecomposer 失败: %s", exc)
            return ResidualDecomposeResult(
                residual_mean=0.0,
                residual_std=0.0,
                overperformers=[],
                underperformers=[],
                system_bias_flag="ERROR",
                residuals=[],
                marketing_available=False,
                attribution_summary={},
            )


# ============================================================
# 主入口：顺序跑四步 + 落盘
# ============================================================
@dataclass
class RunAllLoopsResult:
    loop1: VLMFeatureCalibrationResult
    loop2: PersonaDistributionFitResult
    loop3: EnsembleTuneResult
    residual: ResidualDecomposeResult
    output_files: list[Path]
    cross_validation: CrossValidationResult | None = None   # v1.4.46+
    patterns: Any = None                                    # v1.4.46+ PatternMineResult | None


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _rank_percentile(values: "pd.Series | list[float]") -> pd.Series:
    """将数值序列转为 [0, 1] 的秩百分位（spec §9.1 归一化）。

    spec §9.1：y_i = 真实销量排名（0-1归一化），ŷ_i = 3Loop校准后的预测分。
    残差 ε = y - ŷ 只有在两者同尺度时才有统计意义；原始销量（千级）与
    引擎集成分（0-10）直接相减会让 μ/σ 失真、±2σ 失效。
    用 (rank-1)/(n-1)：最小值→0，最大值→1，并列用 average 秩。
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    n = len(s)
    if n == 0:
        return s.astype(float)
    if n == 1:
        return pd.Series([0.5])
    return (s.rank(method="average") - 1) / (n - 1)


# ========== 分类校准辅助函数（v1.4.43 — P0: Spearman→分类 F1/Precision@k）==========
# 设计原则：
#   - 向后兼容：保留 Spearman，新增分类指标作为"业务对齐层"
#   - 从销量百分位构造 S/A+/A/P 标签（top percentile 映射）
#   - Precision@k 直接回答业务问题："我给买手推的 top 3 款里有几款真爆了？"

# 销量百分位 → grade 的边界（percentile 越高越好，1.0=最佳）
# S款: top 10%  (p ≥ 0.90)
# A+: top 10-25% (0.75 ≤ p < 0.90)
# A:  middle 50% (0.25 ≤ p < 0.75)
# P:  bottom 25% (p < 0.25)
# 可通过 set_grade_boundaries() 覆盖
_PERCENTILE_GRADE_BOUNDARIES: list[tuple[str, float, float]] = [
    ("S",  0.90, 1.01),
    ("A+", 0.75, 0.90),
    ("A",  0.25, 0.75),
    ("P",  0.00, 0.25),
]


def set_grade_boundaries(
    s_threshold: float = 0.90,
    aplus_threshold: float = 0.75,
    a_threshold: float = 0.25,
) -> None:
    """覆盖默认的销量百分位→grade 边界。

    Args:
        s_threshold: S款下界（默认 0.90，即 top 10%）
        aplus_threshold: A+款下界（默认 0.75）
        a_threshold: A款下界（默认 0.25，bottom 25%→P款）

    边界关系：1.0 > s_threshold > aplus_threshold > a_threshold > 0
    """
    global _PERCENTILE_GRADE_BOUNDARIES
    if not (1.0 > s_threshold > aplus_threshold > a_threshold > 0):
        raise ValueError("边界必须满足 1.0 > s > aplus > a > 0")
    _PERCENTILE_GRADE_BOUNDARIES = [
        ("S",  s_threshold, 1.01),
        ("A+", aplus_threshold, s_threshold),
        ("A",  a_threshold, aplus_threshold),
        ("P",  0.0, a_threshold),
    ]


def _percentile_to_grade(p: float) -> str:
    """单个销量百分位 → grade 标签。"""
    for label, lo, hi in _PERCENTILE_GRADE_BOUNDARIES:
        if lo <= p < hi:
            return label
    return _PERCENTILE_GRADE_BOUNDARIES[-1][0]  # fallback → P


def _build_grade_labels_from_percentile(
    percentile: "pd.Series | list[float]",
) -> list[str]:
    """批量：销量百分位 [0,1] → S/A+/A/P 标签列表。"""
    if isinstance(percentile, pd.Series):
        return [_percentile_to_grade(float(p)) for p in percentile.values]
    return [_percentile_to_grade(float(p)) for p in percentile]


# grade → 数值编码（用于 sklearn 分类指标）
_GRADE_TO_INT = {"S": 3, "A+": 2, "A": 1, "P": 0}
_INT_TO_GRADE = {v: k for k, v in _GRADE_TO_INT.items()}


def _classification_metrics(
    y_true_labels: list[str],
    y_pred_scores: "list[float] | pd.Series",
    *,
    top_k: int = 3,
) -> dict[str, float]:
    """计算分类指标：macro F1 + Precision@k。

    Args:
        y_true_labels: 真实 grade 标签列表（从销量百分位构造）
        y_pred_scores: 模型预测分数（任意尺度，内部用 rank 排序取 top-k）
        top_k: Precision@k 的 k 值（默认 3 — 买手通常只看 top 3 款）

    Returns:
        dict: {
          "macro_f1":     0~1,  # 四分类 macro F1
          "precision_at_k": 0~1,  # top-k 中真实 S 款的比例
          "s_precision":  0~1,  # 预测为 S 的款里真实 S 的比例
          "s_recall":     0~1,  # 真实 S 款里被正确识别为 S 的比例
          "accuracy":     0~1,  # 四分类 accuracy
        }
    """
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    except ImportError:
        return {"macro_f1": 0.0, "precision_at_k": 0.0, "s_precision": 0.0,
                "s_recall": 0.0, "accuracy": 0.0}

    n = len(y_true_labels)
    if n == 0:
        return {"macro_f1": 0.0, "precision_at_k": 0.0, "s_precision": 0.0,
                "s_recall": 0.0, "accuracy": 0.0}

    y_true = [_GRADE_TO_INT.get(l, 0) for l in y_true_labels]

    # 预测分 → 预测 grade：先 rank percentile → 再映射 grade
    pred_pct = _rank_percentile(list(y_pred_scores))
    y_pred_labels = [_percentile_to_grade(float(p)) for p in pred_pct.values]
    y_pred = [_GRADE_TO_INT.get(l, 0) for l in y_pred_labels]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # 二分类（S vs 非 S）的 precision / recall
    y_true_bin = [1 if l == "S" else 0 for l in y_true_labels]
    y_pred_bin = [1 if l == "S" else 0 for l in y_pred_labels]
    s_precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    s_recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))

    # Precision@k：top-k 预测分对应的款里，有多少真实是 S 款
    k = min(top_k, n)
    if k > 0:
        pred_ranked = sorted(range(n), key=lambda i: -float(y_pred_scores[i]))
        top_k_idx = pred_ranked[:k]
        s_in_top_k = sum(1 for i in top_k_idx if y_true_labels[i] == "S")
        precision_at_k = s_in_top_k / k
    else:
        precision_at_k = 0.0

    return {
        "macro_f1": macro_f1,
        "precision_at_k": precision_at_k,
        "s_precision": s_precision,
        "s_recall": s_recall,
        "accuracy": accuracy,
    }


def _delta_str(a: float, b: float) -> str:
    """辅助：delta 显示 (+0.03 或 -0.01)。"""
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _engine_score_ensemble(
    df: pd.DataFrame,
    engine_weights: dict[str, float],
    channel_weights: dict[str, float],
) -> pd.Series:
    """用 Loop3 已校准权重，将三大引擎 + 渠道合并为单一预测分 y_pred，
    用于残差分解。v1.4.49+ 支持 grade_norm 第 4 引擎。
    """
    # v1.4.49+: 动态解析引擎列（有 grade_norm 就加）
    engine_cols = EnsembleWeightTuner._resolve_engine_cols(df)
    channel_cols = EnsembleWeightTuner.CHANNEL_COLS
    required = list(EnsembleWeightTuner.BASE_ENGINE_COLS) + channel_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"残差分解缺少列: {missing}")

    engine_part = sum(
        df[c] * engine_weights.get(c, 1.0 / max(len(engine_cols), 1))
        for c in engine_cols if c in df.columns
    )
    channel_part = sum(
        df[c] * channel_weights.get(c, 1.0 / len(channel_cols))
        for c in channel_cols
    )
    return 0.5 * engine_part + 0.5 * channel_part


def build_history_df(
    predictions: list[Any],
    sales_col: str = "sales",
    sales_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    """从 FullPrediction 列表构造 history_df，含 run_all_loops 所需全部列。

    列：style_id, F01-F10 (10 特征分), P01-P30 (30 人设投票分),
        persona_score, channel_score, price_value_score,
        natural_score, live_score, sales。

    用 getattr 鸭子类型访问，避免与 src.types 强耦合；
    用于 smoke test 与 pipeline.py backtest 分支共用，避免两边重复逻辑。

    Args:
        predictions: list[FullPrediction] 或同形状鸭子类型对象
        sales_col: 销量列名，默认 "sales"
        sales_lookup: 当 prediction.info.sales_qty 为 0/None 时的兜底销量映射
            {style_id: sales_qty}，便于回测注入真实销量
    """
    persona_ids = [f"P{i:02d}" for i in range(1, 31)]
    feature_cols = [f"F{i:02d}" for i in range(1, 11)]
    sales_lookup = sales_lookup or {}
    rows: list[dict[str, Any]] = []

    for p in predictions:
        info = getattr(p, "info", None)
        voting = getattr(p, "voting", None)
        channels = getattr(p, "channels", None)
        features = getattr(getattr(p, "features", None), "features", {}) or {}

        style_id = getattr(info, "style_id", "unknown")
        # 销量：info.sales_qty 优先，0/None 时回退 sales_lookup（回测注入真实销量）
        sales = float(
            getattr(info, "sales_qty", 0)
            or sales_lookup.get(style_id, 0.0)
        )

        # F01-F10：按 features 字典顺序取前 10 个；不足用 5.0 兜底
        feat_row: dict[str, float] = {}
        feat_items = list(features.items())[:10]
        for i, (_, f) in enumerate(feat_items):
            feat_row[feature_cols[i]] = float(getattr(f, "score", 5.0))
        for i in range(len(feat_items), 10):
            feat_row[feature_cols[i]] = 5.0

        # P01-P30：votes 不足 30 用 weighted_score 兜底
        weighted_score = float(getattr(voting, "weighted_score", 5.0))
        votes = getattr(voting, "votes", None) or []
        persona_row: dict[str, float] = {}
        for pi in range(30):
            if pi < len(votes):
                persona_row[persona_ids[pi]] = float(
                    getattr(votes[pi], "final_score", weighted_score)
                )
            else:
                persona_row[persona_ids[pi]] = weighted_score

        # 三大引擎 + 双渠道
        natural = float(getattr(channels, "natural_score", 5.0))
        live = float(getattr(channels, "live_score", 5.0))
        perceived = float(getattr(channels, "perceived_value", 5.0))
        # 营销投放标签（ADR-0001：实际投放，用于残差归因）
        is_main_push = 1.0 if getattr(info, "is_main_push", False) else 0.0
        is_live_stream = 1.0 if getattr(info, "is_live_stream", False) else 0.0
        eng_row = {
            "persona_score": weighted_score,
            "channel_score": (natural + live) / 2,
            "price_value_score": perceived,
            "natural_score": natural,
            "live_score": live,
            "is_main_push": is_main_push,
            "is_live_stream": is_live_stream,
            sales_col: sales,
        }

        rows.append({
            "style_id": style_id,
            **feat_row,
            **persona_row,
            **eng_row,
        })

    return pd.DataFrame(rows)


# ======================================================================
# P0: Cross-validation — 跨季回测校准（防过拟合）
# ======================================================================
@dataclass
class CrossValidationResult:
    """按时间切分训练/测试集，验证 3Loop 校准是否过拟合。

    用法：
      cv = cross_validate_calibration(history_df)
      cv.overfitting_ratio_in_spearman  # 0 = 无过拟合, 1 = 完全过拟合
      cv.overfitting_ratio_in_f1        # 同上，用 F1 指标
    """
    n_total: int
    n_train: int
    n_test: int
    train_in_spearman: float           # 训练集（旧数据）内 3Loop 校准后的 Spearman
    test_in_spearman: float            # 测试集（新数据）同一套权重的 Spearman
    train_in_f1: float                 # 训练集 F1
    test_in_f1: float                  # 测试集 F1
    train_in_p_at_k: float
    test_in_p_at_k: float
    overfitting_ratio_in_spearman: float  # 1 - (test/train) — 越接近 0 越好
    overfitting_ratio_in_f1: float
    verdict: str                       # "OK" / "WARN" / "FAIL"
    notes: str = ""


def cross_validate_calibration(
    history_df: pd.DataFrame,
    *,
    sales_col: str = "sales",
    time_col: str = "",   # 可选日期列；空=按行序
    test_fraction: float = 0.2,   # 后 20% 当"新季度"
    min_samples_train: int = 20,
    min_samples_test: int = 8,
) -> CrossValidationResult:
    """跨季回测：用"旧款"拟合 3Loop 权重 → "新款"检验真实效果。

    为什么需要：3Loop 默认在同一批数据上检验 spearman 提升（in-sample），
    小样本下几乎必然提升，但那是"自己考自己"的分数。本函数把时间维度
    引进来：按 time_col（或行序）切 train（前 80%）/ test（后 20%），
    在 train 上跑 3Loop 拟合权重，用 test 数据直接算 Spearman + F1。

    Args:
        history_df: 带 F01-F10, P01-P30, 引擎分 + sales 的 DataFrame
        sales_col: 销量列名
        time_col: 可选日期列（字符串/datetime 都行）；空则按行序切分
        test_fraction: 测试集比例（默认后 20% = 新季度）
        min_samples_train / min_samples_test: 样本不足时降级

    Returns:
        CrossValidationResult — 含 overfitting_ratio 和 verdict
    """
    from scipy.stats import spearmanr

    df = history_df.copy()
    if time_col and time_col in df.columns:
        try:
            df["_cv_time"] = pd.to_datetime(df[time_col])
            df = df.sort_values("_cv_time").reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if len(train_df) < min_samples_train or len(test_df) < min_samples_test:
        return CrossValidationResult(
            n_total=n, n_train=len(train_df), n_test=len(test_df),
            train_in_spearman=0, test_in_spearman=0,
            train_in_f1=0, test_in_f1=0,
            train_in_p_at_k=0, test_in_p_at_k=0,
            overfitting_ratio_in_spearman=-1, overfitting_ratio_in_f1=-1,
            verdict="SKIP",
            notes=f"样本不足：train={len(train_df)}/{min_samples_train}, "
                  f"test={len(test_df)}/{min_samples_test}",
        )

    # 1. 在 train 上跑 3Loop 拿权重
    try:
        l1 = VLMFeatureCalibrator.calibrate(train_df, sales_col=sales_col)
        l2 = PersonaDistributionFitter.fit(train_df, sales_col=sales_col)
        l3 = EnsembleWeightTuner.tune(train_df, sales_col=sales_col)
    except Exception as e:
        log.warning("Cross-validation 拟合阶段失败: %s", e)
        return CrossValidationResult(
            n_total=n, n_train=len(train_df), n_test=len(test_df),
            train_in_spearman=0, test_in_spearman=0,
            train_in_f1=0, test_in_f1=0,
            train_in_p_at_k=0, test_in_p_at_k=0,
            overfitting_ratio_in_spearman=-1, overfitting_ratio_in_f1=-1,
            verdict="ERROR", notes=f"拟合异常: {e}",
        )

    def _apply_and_eval(subset: pd.DataFrame) -> dict[str, float]:
        """拿 train 权重套 subset，算 Spearman + 分类指标。"""
        sub = subset.copy()
        # 应用 Loop1 偏置
        for col, bias in l1.feature_biases.items():
            if col in sub.columns:
                sub[col] = sub[col] * bias
        # Loop2 人设权重：聚合 persona_score 时用权重重算
        # 简化：直接在已有 persona_score 上乘 persona_weights 的均值偏移
        pw_vals = list(l2.persona_weights.values())
        pw_mean = sum(pw_vals) / max(len(pw_vals), 1)
        if "persona_score" in sub.columns and pw_mean > 0:
            sub["persona_score"] = sub["persona_score"] * (pw_mean / (1 / 30))
        # Loop3 集成
        eng_score = _engine_score_ensemble(sub, l3.engine_weights, l3.channel_weights)
        y_true = sub[sales_col].astype(float).values

        # Spearman
        try:
            rho, _ = spearmanr(eng_score.values, y_true)
            rho = 0.0 if math.isnan(rho) else float(rho)
        except Exception:
            rho = 0.0

        # 分类指标（用 percentile 标签）
        y_true_pct = _rank_percentile(y_true)
        y_true_labels = _build_grade_labels_from_percentile(y_true_pct)
        cls = _classification_metrics(y_true_labels, list(eng_score.values))

        return {
            "spearman": rho,
            "f1": cls["macro_f1"],
            "p_at_k": cls["precision_at_k"],
        }

    train_eval = _apply_and_eval(train_df)
    test_eval = _apply_and_eval(test_df)

    tr_sp = train_eval["spearman"]
    te_sp = test_eval["spearman"]
    tr_f1 = train_eval["f1"]
    te_f1 = test_eval["f1"]

    # 过拟合率：1 - test/train（train=0 时安全降级）
    if abs(tr_sp) > 1e-6:
        ov_sp = max(0.0, min(1.0, 1.0 - te_sp / tr_sp))
    else:
        ov_sp = 0.0
    if abs(tr_f1) > 1e-6:
        ov_f1 = max(0.0, min(1.0, 1.0 - te_f1 / tr_f1))
    else:
        ov_f1 = 0.0

    # verdict
    if ov_sp < 0.25 and ov_f1 < 0.25 and te_sp > 0:
        verdict = "OK"
    elif ov_sp < 0.5 and ov_f1 < 0.5:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    log.info(
        "📊 CrossVal: train ρ=%.3f F1=%.3f → test ρ=%.3f F1=%.3f | "
        "overfit_ρ=%.2f overfit_F1=%.2f | %s",
        tr_sp, tr_f1, te_sp, te_f1, ov_sp, ov_f1, verdict,
    )

    return CrossValidationResult(
        n_total=n, n_train=len(train_df), n_test=len(test_df),
        train_in_spearman=tr_sp, test_in_spearman=te_sp,
        train_in_f1=tr_f1, test_in_f1=te_f1,
        train_in_p_at_k=train_eval["p_at_k"],
        test_in_p_at_k=test_eval["p_at_k"],
        overfitting_ratio_in_spearman=ov_sp,
        overfitting_ratio_in_f1=ov_f1,
        verdict=verdict,
    )


def run_all_loops(
    brand_cfg: Any,
    history_df: pd.DataFrame,
    prediction_artifacts_dir: Path,
    sales_col: str = "sales",
    *,
    auto_apply: bool = False,
) -> RunAllLoopsResult:
    """顺序执行 Loop1 → Loop2 → Loop3 → 残差分解。

    Args:
        brand_cfg: 含 calibrated_dir 属性的配置对象（如 PathConfig）；
                   若为 Path 则直接当输出目录使用。
        history_df: 历史款 DataFrame，需包含：
            - F01..F10  (Loop1)
            - P01..P30  (Loop2)
            - persona_score, channel_score, price_value_score,
              natural_score, live_score  (Loop3)
            - sales_col 列
        prediction_artifacts_dir: 预测产物目录，校验报告写到
            prediction_artifacts_dir.parent / calibration 或直接
            prediction_artifacts_dir / calibration。
        sales_col: 销量列名（默认 sales）。
        auto_apply: 是否把校准产物直接写入 calibrated_dir 让 pipeline
            下次启动自动加载。⚠️ 默认 False — 校准是在用于拟合的
            同一批数据上检验 Spearman 提升（in-sample），不是真正的
            跨季 out-of-sample 验证。建议人类审核校准报告后再决定
            是否应用（调用 approve_pending_calibration）。

    Returns:
        RunAllLoopsResult，含各步骤结果 + 落盘文件路径列表。
    """
    try:
        if isinstance(brand_cfg, Path):
            calibrated_dir = brand_cfg
        else:
            calibrated_dir = getattr(brand_cfg, "calibrated_dir", None)
            if calibrated_dir is None:
                paths = getattr(brand_cfg, "paths", None)
                output_dir = getattr(paths, "output_dir", prediction_artifacts_dir)
                calibrated_dir = Path(output_dir) / "calibration"
        calibrated_dir = Path(calibrated_dir)

        # === 人工审核门 ===
        # auto_apply=False（默认）→ 产物写到 calibrated_dir/_pending/，
        # pipeline 启动时不会自动加载。人类审核校准报告确认后，
        # 调用 approve_pending_calibration() 把 _pending 里的文件
        # 移到 calibrated_dir 根目录生效。
        # 这避免了 in-sample Spearman 提升过拟合风险。
        if not auto_apply:
            effective_dir = calibrated_dir / "_pending"
            effective_dir.mkdir(parents=True, exist_ok=True)
        else:
            effective_dir = calibrated_dir

        if isinstance(prediction_artifacts_dir, Path):
            artifact_parent = prediction_artifacts_dir.parent
            calib_report_dir = artifact_parent / "calibration"
        else:
            calib_report_dir = calibrated_dir
        calib_report_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[Path] = []

        # ---------- v1.4.49+: grade_norm 生成（内审分级 → [0,100]）----------
        # 优先级: 已有 grade_norm > grade_num（原始整数）> 内审分级列（S/A+/A/P 字符串）
        if "grade_norm" not in history_df.columns:
            if "grade_num" in history_df.columns:
                # 已有整数 grade_num（0/1/2/4），归一化到 [0,100]
                history_df["grade_norm"] = history_df["grade_num"].fillna(0) / 4.0 * 100
            else:
                # 尝试从常见列名找内审分级
                for col_name in ("内审分级", "grade", "final_grade", "expert_grade", "review_grade"):
                    if col_name in history_df.columns:
                        grade_val = history_df[col_name].astype(str).str.upper()
                        grade_map = {"S": 4, "S款": 4, "S级": 4,
                                     "A+": 3, "A+款": 3,
                                     "A": 2, "A款": 2, "A级": 2,
                                     "P": 0, "P款": 0, "P级": 0, "P-": 0,
                                     "": 1}
                        history_df["grade_num"] = grade_val.map(grade_map).fillna(1)
                        history_df["grade_norm"] = history_df["grade_num"] / 4.0 * 100
                        has_enough = history_df["grade_norm"].notna().sum() >= 3
                        if has_enough:
                            log.info(
                                "v1.4.49: 从列 '%s' 解析内审分级 → grade_norm [%d/%d 非空]",
                                col_name, history_df["grade_norm"].notna().sum(), len(history_df),
                            )
                        break
                else:
                    history_df["grade_norm"] = float("nan")  # 标记不存在，Loop3 自动跳过
        else:
            log.info("v1.4.49: history_df 已有 grade_norm，直接使用")

        # ---------- v1.4.48+: pre-Cross-validation — 决定 safe_mode ----------
        log.info("=== pre-Cross-validation: 判断是否需要 safe_mode ===")
        try:
            pre_cv = cross_validate_calibration(history_df, sales_col=sales_col)
            safe_mode = pre_cv.verdict in ("FAIL",) and len(history_df) >= 20
            log.info(
                "pre-CV: verdict=%s, test Spearman=%.3f → safe_mode=%s",
                pre_cv.verdict, pre_cv.test_in_spearman, safe_mode,
            )
        except Exception as exc:
            log.warning("pre-CV 失败，默认 safe_mode=False: %s", exc)
            pre_cv = None
            safe_mode = False

        # ---------- Loop1 ----------
        log.info("=== Loop1: VLMFeatureCalibrator (auto_apply=%s, safe_mode=%s) ===",
                 auto_apply, safe_mode)
        r1 = VLMFeatureCalibrator.calibrate(history_df, sales_col=sales_col,
                                            safe_mode=safe_mode)
        l1_path = effective_dir / "loop1_vlm_feature_biases.yaml"
        _write_yaml(l1_path, {
            "applied": r1.applied,
            "old_spearman_avg": r1.old_spearman_avg,
            "new_spearman_avg": r1.new_spearman_avg,
            # v1.4.44+: 分类指标（与 Spearman 双轨并行）
            "classification": {
                "old_f1": r1.old_f1,
                "new_f1": r1.new_f1,
                "delta_f1": r1.new_f1 - r1.old_f1,
                "old_precision_at_k": r1.old_p_at_k,
                "new_precision_at_k": r1.new_p_at_k,
                "delta_p_at_k": r1.new_p_at_k - r1.old_p_at_k,
            },
            # v1.4.48+: safe_mode 稳定性过滤结果
            "safe_mode_used": r1.safe_mode_used,
            "unstable_features": r1.unstable_features,
            "feature_stability": r1.feature_stability,
            "per_feature_rho": r1.per_feature_rho,
            "feature_biases": r1.feature_biases,
        })
        output_files.append(l1_path)

        # ---------- Loop2 ----------
        log.info("=== Loop2: PersonaDistributionFitter (auto_apply=%s) ===", auto_apply)
        r2 = PersonaDistributionFitter.fit(history_df, sales_col=sales_col)
        l2_path = effective_dir / "loop2_persona_distribution_weights.yaml"
        _write_yaml(l2_path, {
            "applied": r2.applied,
            "old_spearman": r2.old_spearman,
            "new_spearman": r2.new_spearman,
            "delta_spearman": r2.new_spearman - r2.old_spearman,
            # v1.4.44+: 分类指标
            "classification": {
                "old_f1": r2.old_f1,
                "new_f1": r2.new_f1,
                "delta_f1": r2.new_f1 - r2.old_f1,
                "old_precision_at_k": r2.old_p_at_k,
                "new_precision_at_k": r2.new_p_at_k,
                "delta_p_at_k": r2.new_p_at_k - r2.old_p_at_k,
            },
            "lasso_raw_coef": r2.lasso_raw_coef,
            "persona_weights": r2.persona_weights,
        })
        output_files.append(l2_path)

        # ---------- Loop3 ----------
        log.info("=== Loop3: EnsembleWeightTuner (auto_apply=%s) ===", auto_apply)
        r3 = EnsembleWeightTuner.tune(history_df, sales_col=sales_col)
        l3_path = effective_dir / "loop3_ensemble_weights.yaml"
        _write_yaml(l3_path, {
            "applied": r3.applied,
            "engine": {
                "old_spearman": r3.old_engine_spearman,
                "new_spearman": r3.new_engine_spearman,
                # v1.4.44+: 分类指标（与 Loop1/2 一致嵌套）
                "classification": {
                    "old_f1": r3.old_engine_f1,
                    "new_f1": r3.new_engine_f1,
                    "delta_f1": r3.new_engine_f1 - r3.old_engine_f1,
                    "old_precision_at_k": r3.old_engine_p_at_k,
                    "new_precision_at_k": r3.new_engine_p_at_k,
                    "delta_p_at_k": r3.new_engine_p_at_k - r3.old_engine_p_at_k,
                },
                "rho": r3.engine_rho,
                "weights": r3.engine_weights,
            },
            "channel": {
                "old_spearman": r3.old_channel_spearman,
                "new_spearman": r3.new_channel_spearman,
                "classification": {
                    "old_f1": r3.old_chan_f1,
                    "new_f1": r3.new_chan_f1,
                    "delta_f1": r3.new_chan_f1 - r3.old_chan_f1,
                    "old_precision_at_k": r3.old_chan_p_at_k,
                    "new_precision_at_k": r3.new_chan_p_at_k,
                    "delta_p_at_k": r3.new_chan_p_at_k - r3.old_chan_p_at_k,
                },
                "rho": r3.channel_rho,
                "weights": r3.channel_weights,
            },
        })
        output_files.append(l3_path)

        # ---------- 残差分解 ----------
        log.info("=== ResidualDecomposer ===")
        y_pred_series = _engine_score_ensemble(
            history_df, r3.engine_weights, r3.channel_weights,
        )
        y_true_series = history_df[sales_col].astype(float)
        # spec §9.1：残差 ε = y - ŷ 要求两边同尺度。原始销量（千级）与
        # 引擎集成分（0-10）量级差 1000×，必须先归一化到 [0,1] 秩百分位，
        # 否则 μ/σ 失真、±2σ 区间覆盖全数据 → 残差分离器实际失效。
        y_pred_norm = _rank_percentile(y_pred_series)
        y_true_norm = _rank_percentile(y_true_series)
        r4 = ResidualDecomposer.decompose(y_true_norm, y_pred_norm, history_df)
        residual_path = effective_dir / "residual_decompose.yaml"
        _write_yaml(residual_path, {
            "residual_mean": r4.residual_mean,
            "residual_std": r4.residual_std,
            "system_bias_flag": r4.system_bias_flag,
            "marketing_available": r4.marketing_available,
            "attribution_summary": r4.attribution_summary,
            "overperformers": [
                {k: v for k, v in it.items() if k != "index"}
                for it in r4.overperformers
            ],
            "underperformers": [
                {k: v for k, v in it.items() if k != "index"}
                for it in r4.underperformers
            ],
        })
        output_files.append(residual_path)

        # ---------- Cross-validation：跨季回测（v1.4.46+）----------
        cv_result: CrossValidationResult | None = None
        try:
            # 决定用什么做"时间轴"
            time_col = ""
            for cand in ("date", "season", "quarter", "year", "sold_date", "batch_date"):
                if cand in history_df.columns:
                    time_col = cand
                    break

            cv_result = cross_validate_calibration(
                history_df, sales_col=sales_col, time_col=time_col,
            )
            cv_path = effective_dir / "cross_validation.yaml"
            _write_yaml(cv_path, {
                "n_total": cv_result.n_total,
                "n_train": cv_result.n_train,
                "n_test": cv_result.n_test,
                "time_col_used": time_col or "(行序)",
                "train_metrics": {
                    "spearman": cv_result.train_in_spearman,
                    "f1": cv_result.train_in_f1,
                    "p_at_k": cv_result.train_in_p_at_k,
                },
                "test_metrics": {
                    "spearman": cv_result.test_in_spearman,
                    "f1": cv_result.test_in_f1,
                    "p_at_k": cv_result.test_in_p_at_k,
                },
                "overfitting_ratio": {
                    "spearman": cv_result.overfitting_ratio_in_spearman,
                    "f1": cv_result.overfitting_ratio_in_f1,
                },
                "verdict": cv_result.verdict,
                "notes": cv_result.notes,
            })
            output_files.append(cv_path)
            if cv_result.verdict != "SKIP":
                log.info(
                    "✅ Cross-validation: verdict=%s (overfit_ρ=%.2f, overfit_F1=%.2f)",
                    cv_result.verdict,
                    cv_result.overfitting_ratio_in_spearman,
                    cv_result.overfitting_ratio_in_f1,
                )
        except Exception as exc:
            log.warning("Cross-validation 失败（不阻塞）: %s", exc)
            cv_result = None

        # ---------- PatternMiner：从历史数据提炼爆款/败款模式（v1.4.44+）----------
        pattern_result = None
        pattern_yaml_path: Path | None = None
        pattern_md_path: Path | None = None

        # P0-#12: 按年份分组校准（v1.4.46+）
        group_by_year = bool(getattr(brand_cfg, "calibration_group_by_year", False))
        pattern_df_for_group = history_df
        year_group_note = ""
        if group_by_year and "year" in history_df.columns:
            year_vals = history_df["year"].dropna().unique()
            if len(year_vals) >= 2:
                latest_year = int(sorted(year_vals)[-1])
                pattern_df_for_group = history_df[
                    history_df["year"] < latest_year
                ].copy()
                year_group_note = (
                    f"（calibration_group_by_year=true：模式库只用 {list(year_vals)} "
                    f"中 {latest_year} 之前的数据 = {len(pattern_df_for_group)} 款；"
                    f"最新 {latest_year} 款留给跨季回测检验）"
                )
                log.info("📅 PatternMiner 按年分组：train_on=<%d, n=%d",
                         latest_year, len(pattern_df_for_group))
        try:
            from ..pattern_miner import mine_patterns, save_patterns

            # 品牌名 & 季度标识
            brand_name = ""
            quarter = ""
            if hasattr(brand_cfg, "brand_id"):
                brand_name = str(getattr(brand_cfg, "brand_id", "") or "")
            if hasattr(brand_cfg, "brand_name"):
                brand_name = brand_name or str(getattr(brand_cfg, "brand_name", "") or "")
            # 季度：尝试从 history_df 或 brand_cfg 提取
            if "quarter" in history_df.columns:
                q_vals = history_df["quarter"].dropna().astype(str).unique()
                if len(q_vals) > 0:
                    quarter = str(q_vals[-1])[:16]
            elif hasattr(brand_cfg, "quarter"):
                quarter = str(getattr(brand_cfg, "quarter", "") or "")[:16]

            pattern_memory_dir = calibrated_dir / "memory"
            log.info(
                "=== PatternMiner: mine_patterns (%d samples, brand=%s, quarter=%s) ===",
                len(history_df), brand_name or "?", quarter or "?",
            )
            pattern_result = mine_patterns(
                pattern_df_for_group, sales_col=sales_col,
                season=f"{quarter}" if quarter else "",
            )
            if pattern_result.n_samples_total > 0 and (pattern_result.s_rules or pattern_result.p_rules):
                paths = save_patterns(
                    pattern_result, pattern_memory_dir,
                    brand_name=brand_name, quarter=quarter,
                )
                pattern_yaml_path = paths[0]
                pattern_md_path = paths[1] if len(paths) > 1 else None
                output_files.extend([p for p in paths if p is not None])
                log.info(
                    "✅ PatternMiner: %d 条规则 (S=%d, P=%d), accuracy=%.1f%% → %s",
                    len(pattern_result.rules),
                    len(pattern_result.s_rules),
                    len(pattern_result.p_rules),
                    pattern_result.tree_accuracy * 100,
                    pattern_yaml_path,
                )
            else:
                log.info(
                    "PatternMiner 样本不足或无区分度（accuracy=%.1f%%, S=%d, P=%d），跳过落盘",
                    pattern_result.tree_accuracy * 100,
                    len(pattern_result.s_rules),
                    len(pattern_result.p_rules),
                )
        except ImportError:
            log.info("pattern_miner 未安装，跳过 PatternMiner（v1.4.44+ 特性）")
        except Exception as exc:
            log.warning("PatternMiner 执行异常（不阻塞校准）: %s", exc)
            pattern_result = None

        # ---------- Markdown 校准报告 ----------
        md_lines: list[str] = []

        # === 人工审核门：报告顶部横幅 ===
        if not auto_apply:
            md_lines.append("> ⚠️ **人工审核门 — 本轮校准待确认**")
            md_lines.append(">")
            md_lines.append("> 本轮校准产物已写入 `_pending/` 目录，**尚未生效**。")
            md_lines.append("> 原因：校准是在用于拟合的同一批历史数据上检验 Spearman 提升")
            md_lines.append("> （in-sample），小样本下几乎必然提升，不是真正的跨季 out-of-sample 验证。")
            md_lines.append(">")
            md_lines.append("> **应用前请检查：**")
            md_lines.append("> 1. 各 Loop 的 Spearman 提升是否足够（≥0.01）、方向是否一致")
            md_lines.append("> 2. 残差分解里的超预期款/不及预期款是否有运营复盘可以解释")
            md_lines.append("> 3. 是否有新一季销量数据可以做真正的跨季回测")
            md_lines.append(">")
            md_lines.append("> 确认后调用 `approve_pending_calibration('" + str(calibrated_dir) + "')`")
            md_lines.append("> 把 `_pending/` 里的文件移到 `calibrated_dir` 根目录生效。")
            md_lines.append("")

        md_lines.append("# 校准循环报告 (Calibration Report)")
        md_lines.append("")
        md_lines.append(f"- 样本数: {len(history_df)}")
        md_lines.append(f"- 校准目录: `{calibrated_dir}`")
        md_lines.append(f"- 产物目录: `{effective_dir}`")
        md_lines.append(f"- auto_apply: {auto_apply}")
        md_lines.append(f"- 销量列: `{sales_col}`")
        md_lines.append("")

        md_lines.append("## Loop1 · VLM特征偏置")
        md_lines.append("")
        md_lines.append(f"- 生效: **{r1.applied}**")
        md_lines.append(f"- 旧Spearman均值: {r1.old_spearman_avg:.4f}")
        md_lines.append(f"- 新Spearman: {r1.new_spearman_avg:.4f}")
        # v1.4.44+: 分类指标
        md_lines.append(f"- **旧F1**: {r1.old_f1:.3f} → **新F1**: {r1.new_f1:.3f} (Δ {r1.new_f1 - r1.old_f1:+.3f})")
        md_lines.append(f"- **旧P@k**: {r1.old_p_at_k:.3f} → **新P@k**: {r1.new_p_at_k:.3f} (Δ {r1.new_p_at_k - r1.old_p_at_k:+.3f})")
        md_lines.append("")
        md_lines.append("| 特征 | ρ(销量) | 偏置系数 |")
        md_lines.append("|------|---------|----------|")
        for col in VLMFeatureCalibrator.FEATURE_COLS:
            md_lines.append(
                f"| {col} | {r1.per_feature_rho.get(col, 0.0):.4f} "
                f"| {r1.feature_biases.get(col, 1.0):.4f} |"
            )
        md_lines.append("")

        md_lines.append("## Loop2 · 人设分布（Lasso）")
        md_lines.append("")
        md_lines.append(f"- 生效: **{r2.applied}**")
        md_lines.append(f"- 旧Spearman: {r2.old_spearman:.4f}")
        md_lines.append(f"- 新Spearman: {r2.new_spearman:.4f}")
        md_lines.append(f"- Δ: {r2.new_spearman - r2.old_spearman:.4f}")
        # v1.4.44+: 分类指标
        md_lines.append(f"- **旧F1**: {r2.old_f1:.3f} → **新F1**: {r2.new_f1:.3f} (Δ {r2.new_f1 - r2.old_f1:+.3f})")
        md_lines.append(f"- **旧P@k**: {r2.old_p_at_k:.3f} → **新P@k**: {r2.new_p_at_k:.3f} (Δ {r2.new_p_at_k - r2.old_p_at_k:+.3f})")
        w_min = PersonaDistributionFitter.W_MIN
        md_lines.append(f"- 多样性下限 w_min = 1/(2×30) ≈ {w_min:.6f}")
        md_lines.append(
            f"- 系数方向性: 已保留正负号（正相关→权重，负相关→降至 w_min）"
        )
        md_lines.append("")
        md_lines.append("| 人设 | Lasso原始系数 | 方向 | 归一化权重 | 触底? |")
        md_lines.append("|------|---------------|------|------------|-------|")
        pos_count = 0
        neg_count = 0
        zero_count = 0
        floor_count = 0
        for col in PersonaDistributionFitter.PERSONA_COLS:
            raw = r2.lasso_raw_coef.get(col, 0.0)
            w = r2.persona_weights.get(col, 0.0)
            if raw > 1e-6:
                direction, pos_count = "+", pos_count + 1
            elif raw < -1e-6:
                direction, neg_count = "-", neg_count + 1
            else:
                direction, zero_count = "0", zero_count + 1
            is_floor = w <= w_min + 1e-9
            if is_floor:
                floor_count += 1
            md_lines.append(
                f"| {col} | {raw:.6f} | {direction} | {w:.6f} | "
                f"{'是' if is_floor else ''} |"
            )
        md_lines.append("")
        md_lines.append(
            f"- 系数方向: 正 {pos_count} / 负 {neg_count} / 零 {zero_count} "
            f"(共 {len(PersonaDistributionFitter.PERSONA_COLS)})"
        )
        md_lines.append(f"- 触 w_min 下限人设: {floor_count}/{len(PersonaDistributionFitter.PERSONA_COLS)}")
        md_lines.append("")

        md_lines.append("## Loop3 · 集成权重")
        md_lines.append("")
        md_lines.append("### 三大引擎")
        md_lines.append(f"- 旧Spearman: {r3.old_engine_spearman:.4f}")
        md_lines.append(f"- 新Spearman: {r3.new_engine_spearman:.4f}")
        # v1.4.44+: 分类指标
        md_lines.append(f"- **旧F1**: {r3.old_engine_f1:.3f} → **新F1**: {r3.new_engine_f1:.3f} (Δ {r3.new_engine_f1 - r3.old_engine_f1:+.3f})")
        md_lines.append(f"- **旧P@k**: {r3.old_engine_p_at_k:.3f} → **新P@k**: {r3.new_engine_p_at_k:.3f} (Δ {r3.new_engine_p_at_k - r3.old_engine_p_at_k:+.3f})")
        md_lines.append("")
        md_lines.append("| 引擎 | ρ(销量) | 权重 |")
        md_lines.append("|------|---------|------|")
        for col in r3.engine_weights.keys():
            md_lines.append(
                f"| {col} | {r3.engine_rho.get(col, 0.0):.4f} "
                f"| {r3.engine_weights.get(col, 0.0):.4f} |"
            )
        md_lines.append("")
        md_lines.append("### 双渠道")
        md_lines.append(f"- 旧Spearman: {r3.old_channel_spearman:.4f}")
        md_lines.append(f"- 新Spearman: {r3.new_channel_spearman:.4f}")
        # v1.4.44+: 分类指标
        md_lines.append(f"- **旧F1**: {r3.old_chan_f1:.3f} → **新F1**: {r3.new_chan_f1:.3f} (Δ {r3.new_chan_f1 - r3.old_chan_f1:+.3f})")
        md_lines.append(f"- **旧P@k**: {r3.old_chan_p_at_k:.3f} → **新P@k**: {r3.new_chan_p_at_k:.3f} (Δ {r3.new_chan_p_at_k - r3.old_chan_p_at_k:+.3f})")
        md_lines.append("")
        md_lines.append("| 渠道 | ρ(销量) | 权重 |")
        md_lines.append("|------|---------|------|")
        for col in EnsembleWeightTuner.CHANNEL_COLS:
            md_lines.append(
                f"| {col} | {r3.channel_rho.get(col, 0.0):.4f} "
                f"| {r3.channel_weights.get(col, 0.0):.4f} |"
            )
        md_lines.append("")

        md_lines.append("## 残差分解")
        md_lines.append("")
        md_lines.append(f"- 残差均值 μ: {r4.residual_mean:.4f}")
        md_lines.append(f"- 残差标准差 σ: {r4.residual_std:.4f}")
        md_lines.append(f"- 系统偏差Flag: **{r4.system_bias_flag}**")
        md_lines.append("")

        # 资源错配归因（ADR-0001：款式质量 vs 投放强度分离）
        if r4.marketing_available:
            md_lines.append("### 🎯 资源错配归因（款式质量 × 投放强度）")
            md_lines.append("")
            md_lines.append("| 归因 | 款数 | 运营含义 |")
            md_lines.append("|------|------|----------|")
            for k in (ResidualDecomposer.ATTRIB_OVER_PUSHED,
                      ResidualDecomposer.ATTRIB_OVER_ORGANIC,
                      ResidualDecomposer.ATTRIB_UNDER_PUSHED,
                      ResidualDecomposer.ATTRIB_UNDER_ORGANIC):
                cnt = r4.attribution_summary.get(k, 0)
                if k == ResidualDecomposer.ATTRIB_OVER_ORGANIC:
                    hint = "下季应提前识别并追加投放/备货"
                elif k == ResidualDecomposer.ATTRIB_UNDER_PUSHED:
                    hint = "投放预算被浪费，复盘选款信号哪里漏了"
                elif k == ResidualDecomposer.ATTRIB_OVER_PUSHED:
                    hint = "系统预测与投放一致，权重可信"
                else:
                    hint = "系统识别正确，未推是合理决策"
                md_lines.append(f"| {k} | {cnt} | {hint} |")
            md_lines.append("")
        else:
            md_lines.append(
                "> ℹ️ 本批次无营销投放列（是否主推/是否直播重点），"
                "无法区分「款式好」和「被推爆」。补列后重跑可获得归因。"
            )
            md_lines.append("")

        md_lines.append(f"### 超预期款（ε > μ+2σ，共{len(r4.overperformers)}个）")
        md_lines.append("")
        if r4.overperformers:
            if r4.marketing_available:
                md_lines.append("| style_id | 残差 | y_true | y_pred | 归因 |")
                md_lines.append("|----------|------|--------|--------|------|")
                for item in r4.overperformers[:20]:
                    md_lines.append(
                        f"| {item['style_id']} | {item['residual']:.4f} "
                        f"| {item.get('y_true', '-')} | {item.get('y_pred', '-')} "
                        f"| {item.get('attribution', '-')} |"
                    )
            else:
                md_lines.append("| style_id | 残差 | y_true | y_pred |")
                md_lines.append("|----------|------|--------|--------|")
                for item in r4.overperformers[:20]:
                    md_lines.append(
                        f"| {item['style_id']} | {item['residual']:.4f} "
                        f"| {item.get('y_true', '-')} | {item.get('y_pred', '-')} |"
                    )
        else:
            md_lines.append("（无）")
        md_lines.append("")

        md_lines.append(f"### 不及预期款（ε < μ-2σ，共{len(r4.underperformers)}个）")
        md_lines.append("")
        if r4.underperformers:
            if r4.marketing_available:
                md_lines.append("| style_id | 残差 | y_true | y_pred | 归因 |")
                md_lines.append("|----------|------|--------|--------|------|")
                for item in r4.underperformers[:20]:
                    md_lines.append(
                        f"| {item['style_id']} | {item['residual']:.4f} "
                        f"| {item.get('y_true', '-')} | {item.get('y_pred', '-')} "
                        f"| {item.get('attribution', '-')} |"
                    )
            else:
                md_lines.append("| style_id | 残差 | y_true | y_pred |")
                md_lines.append("|----------|------|--------|--------|")
                for item in r4.underperformers[:20]:
                    md_lines.append(
                        f"| {item['style_id']} | {item['residual']:.4f} "
                        f"| {item.get('y_true', '-')} | {item.get('y_pred', '-')} |"
                    )
        else:
            md_lines.append("（无）")
        md_lines.append("")

        # v1.4.46+: Cross-validation 章节
        md_lines.append("## 🔍 跨季回测（Cross-validation · v1.4.46+）")
        md_lines.append("")
        if cv_result is None:
            md_lines.append("> 跨季回测未执行。")
        elif cv_result.verdict == "SKIP":
            md_lines.append(f"> 样本不足，已跳过：{cv_result.notes}")
        else:
            emoji = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌", "ERROR": "🔶"}.get(
                cv_result.verdict, "❓"
            )
            md_lines.append(f"{emoji} **verdict: `{cv_result.verdict}`** "
                            f"（overfit_ρ={cv_result.overfitting_ratio_in_spearman:.2f}, "
                            f"overfit_F1={cv_result.overfitting_ratio_in_f1:.2f}）")
            md_lines.append("")
            md_lines.append(f"- 切分：train={cv_result.n_train} / test={cv_result.n_test} "
                            f"（共 {cv_result.n_total} 款）")
            md_lines.append(f"- Train ρ={cv_result.train_in_spearman:.3f} → "
                            f"Test ρ=**{cv_result.test_in_spearman:.3f}** "
                            f"（Δ={cv_result.train_in_spearman - cv_result.test_in_spearman:+.3f}）")
            md_lines.append(f"- Train F1={cv_result.train_in_f1:.3f} → "
                            f"Test F1=**{cv_result.test_in_f1:.3f}** "
                            f"（Δ={cv_result.train_in_f1 - cv_result.test_in_f1:+.3f}）")
            md_lines.append("")
            md_lines.append(
                "overfit_ratio = 1 - (test/train)，0=无过拟合，越高越虚。"
                " 建议 < 0.25 为健康，> 0.5 需警惕校准过拟合历史噪音。"
            )

        md_lines.append("")

        # v1.4.44+: PatternMiner 章节
        md_lines.append("## 🧠 模式提炼（PatternMiner · v1.4.44+）")
        if year_group_note:
            md_lines.append("")
            md_lines.append(year_group_note)
        md_lines.append("")
        if pattern_result is None:
            md_lines.append("> PatternMiner 未执行（ImportError/异常/样本不足）。")
            md_lines.append("")
        elif pattern_result.n_samples_total == 0:
            md_lines.append("> 有效样本不足，未提炼出规则。建议增加带真实销量的款式。")
            md_lines.append("")
        else:
            md_lines.append(f"- **有效样本**: {pattern_result.n_samples_total}")
            md_lines.append(f"- **Decision Tree 准确率**: {pattern_result.tree_accuracy:.1%}")
            top_feats = sorted(pattern_result.feature_importance.items(), key=lambda x: -x[1])[:5]
            if top_feats:
                md_lines.append(f"- **关键特征 Top 5**: " + " / ".join(f"`{k}`(v={v:.3f})" for k, v in top_feats))
            md_lines.append(f"- **规则总数**: {len(pattern_result.rules)}（S={len(pattern_result.s_rules)}, P={len(pattern_result.p_rules)}）")
            md_lines.append("")

            if pattern_result.s_rules:
                md_lines.append("### 🏆 S款 爆款基因（top 3 rules）")
                md_lines.append("")
                for r in pattern_result.s_rules[:3]:
                    md_lines.append(f"- {r.to_text()}")
                md_lines.append("")

            if pattern_result.p_rules:
                md_lines.append("### ⚠️ P款 避坑指南（top 3 rules）")
                md_lines.append("")
                for r in pattern_result.p_rules[:3]:
                    md_lines.append(f"- {r.to_text()}")
                md_lines.append("")

            if pattern_yaml_path:
                md_lines.append(f"- **规则 YAML**: `{pattern_yaml_path}`")
            if pattern_md_path:
                md_lines.append(f"- **完整 Markdown 报告**: `{pattern_md_path}`")
            md_lines.append("")
            md_lines.append(
                "> 💡 这些规则会自动注入到下次 Persona Voting 的 LLM prompt 中做 few-shot，"
                "让新款式评分参考历史爆款/败款的特征组合。"
            )
            md_lines.append("")

        md_lines.append("## 产出文件")
        md_lines.append("")
        for p in output_files:
            md_lines.append(f"- `{p}`")
        md_lines.append("")

        md_path = calib_report_dir / "calibration_report.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        output_files.append(md_path)
        log.info("校准报告已导出 → %s", md_path)

        return RunAllLoopsResult(
            loop1=r1,
            loop2=r2,
            loop3=r3,
            residual=r4,
            output_files=output_files,
            cross_validation=cv_result,
            patterns=pattern_result,
        )
    except Exception as exc:
        log.exception("run_all_loops 异常退出: %s", exc)
        raise


# ======================================================================
# 人工审核门 · 辅助函数
# ======================================================================

def approve_pending_calibration(calibrated_dir: str | Path) -> list[Path]:
    """把 `calibrated_dir/_pending/` 里的校准产物移到 `calibrated_dir` 根目录生效。

    这是人工审核门的"开门"操作：run_all_loops(auto_apply=False) 默认把
    Loop1/2/3 + 残差分解的 YAML 写到 `_pending/` 子目录，pipeline 启动时
    calibration_loader 只扫描 calibrated_dir 根目录的 yaml，不会加载
    `_pending/`。人类审核校准报告确认本轮权重值得应用后，调用本函数
    把文件移动，下次 pipeline 重启自动生效。

    冲突保护：目标位置已有同名文件时会先保留备份（加 .bak 后缀）。

    Args:
        calibrated_dir: brand_cfg.calibrated_dir 或显式路径。

    Returns:
        已移动的文件列表。

    Raises:
        FileNotFoundError: `_pending/` 目录不存在。
    """
    import shutil

    cal_dir = Path(calibrated_dir)
    pending = cal_dir / "_pending"
    if not pending.is_dir():
        raise FileNotFoundError(
            f"找不到待审核目录 {pending}。"
            f"请确认 run_all_loops(auto_apply=False) 已执行。"
        )

    moved: list[Path] = []
    for src in sorted(pending.iterdir()):
        if not src.is_file():
            continue
        dst = cal_dir / src.name
        # 冲突保护：已有同名文件 → 备份旧版本
        if dst.exists():
            bak = dst.with_suffix(dst.suffix + ".bak")
            dst.rename(bak)
            log.warning("目标已存在 %s → 备份为 %s", dst.name, bak.name)
        shutil.move(str(src), str(dst))
        moved.append(dst)
        log.info("已应用校准产物: %s", dst.name)

    return moved


def reject_pending_calibration(calibrated_dir: str | Path) -> None:
    """废弃本轮 _pending/ 校准产物（直接删除）。

    人类审核后如果认为本轮 Spearman 提升不足或权重方向不合理，
    调用此函数清理掉 _pending/ 目录，重新积累更多历史数据后再跑。
    """
    import shutil

    cal_dir = Path(calibrated_dir)
    pending = cal_dir / "_pending"
    if pending.is_dir():
        shutil.rmtree(pending)
        log.info("已废弃待审核校准产物: %s", pending)
    else:
        log.warning("待审核目录不存在，无需清理: %s", pending)
