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
    ) -> VLMFeatureCalibrationResult:
        """
        Args:
            df: DataFrame，必须包含 10 列 F01..F10 + 销量列
            sales_col: 销量列名（默认 "sales"）
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

            # --- 偏置：相对全局均值，ρ_i / mean(ρ)，clamp 到 [0.7, 1.3] ---
            eps = 1e-9
            biases: dict[str, float] = {}
            safe_avg = old_avg if abs(old_avg) > eps else 1e-6
            for col in cls.FEATURE_COLS:
                raw = per_feature_rho[col] / safe_avg
                biases[col] = max(cls.BIAS_LOW, min(cls.BIAS_HIGH, raw))

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

            applied = new_sp > old_sp + 1e-9
            if not applied:
                biases = {c: 1.0 for c in cls.FEATURE_COLS}
                new_sp = old_sp
                log.info("Loop1 保护触发：新Spearman(%.4f)未优于旧(%.4f)，返回全1偏置", new_sp, old_sp)
            else:
                log.info("Loop1 生效：Spearman %.4f → %.4f", old_sp, new_sp)

            return VLMFeatureCalibrationResult(
                feature_biases=biases,
                old_spearman_avg=float(old_avg),
                new_spearman_avg=float(new_sp),
                per_feature_rho=per_feature_rho,
                applied=applied,
            )
        except Exception as exc:
            log.exception("VLMFeatureCalibrator 失败，降级全1偏置: %s", exc)
            return VLMFeatureCalibrationResult(
                feature_biases={c: 1.0 for c in cls.FEATURE_COLS},
                old_spearman_avg=0.0,
                new_spearman_avg=0.0,
                per_feature_rho={c: 0.0 for c in cls.FEATURE_COLS},
                applied=False,
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


class PersonaDistributionFitter:
    """对 P01..P30 人设列用 Lasso(alpha=0.05, max_iter=5000) 拟合销量，
    系数取 abs 后归一化得到 30 维分布权重。保护机制：新 Spearman ≥ 旧+0.01
    才更新，否则保持上一轮（默认均匀）。
    """

    PERSONA_COLS = [f"P{i:02d}" for i in range(1, 31)]
    MIN_IMPROVEMENT = 0.01

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        sales_col: str = "sales",
        alpha: float = 0.05,
        max_iter: int = 5000,
    ) -> PersonaDistributionFitResult:
        """
        Args:
            df: DataFrame，含 30 列 P01..P30 + 销量列
            sales_col: 销量列名
            alpha: Lasso 正则强度（默认 0.05）
            max_iter: Lasso 最大迭代（默认 5000）
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
                from sklearn.linear_model import Lasso
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
            if len(df_work) < 5:
                raise ValueError(f"样本量不足（{len(df_work)}<5），无法拟合Lasso")

            X = df_work[cls.PERSONA_COLS].values.astype(float)
            y = df_work[sales_col].values.astype(float)

            # --- 旧：均匀权重加权分 vs y 的 Spearman ---
            uniform_score = df_work[cls.PERSONA_COLS].mean(axis=1).values
            try:
                old_sp, _ = spearmanr(uniform_score, y)
                old_sp = 0.0 if math.isnan(old_sp) else float(old_sp)
            except Exception:
                old_sp = 0.0

            # --- Lasso 拟合 ---
            try:
                model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
                model.fit(X, y)
                raw_coef = {col: float(v) for col, v in zip(cls.PERSONA_COLS, model.coef_)}
            except Exception as exc:
                log.warning("Lasso 拟合异常，降级均匀权重: %s", exc)
                raw_coef = {col: 1.0 for col in cls.PERSONA_COLS}

            # --- abs + 归一化得到权重 ---
            abs_coef = {col: abs(v) for col, v in raw_coef.items()}
            total = sum(abs_coef.values())
            if total < 1e-12:
                weights = {col: 1.0 / len(cls.PERSONA_COLS) for col in cls.PERSONA_COLS}
            else:
                weights = {col: v / total for col, v in abs_coef.items()}

            # --- 新加权分 vs y 的 Spearman ---
            weight_arr = np.array([weights[col] for col in cls.PERSONA_COLS], dtype=float)
            new_score = X @ weight_arr
            try:
                new_sp, _ = spearmanr(new_score, y)
                new_sp = 0.0 if math.isnan(new_sp) else float(new_sp)
            except Exception:
                new_sp = old_sp

            applied = new_sp >= old_sp + cls.MIN_IMPROVEMENT - 1e-9
            if not applied:
                weights = {col: 1.0 / len(cls.PERSONA_COLS) for col in cls.PERSONA_COLS}
                log.info(
                    "Loop2 保护触发：新Spearman(%.4f) - 旧(%.4f) = %.4f < %.2f，保持均匀",
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


class EnsembleWeightTuner:
    """对三大引擎（persona / channel / price_value）和双渠道
    （natural / live）分别计算 Spearman，权重 ∝ max(0.05, ρ+0.3)
    归一化。保护机制同前。
    """

    ENGINE_COLS = ["persona_score", "channel_score", "price_value_score"]
    CHANNEL_COLS = ["natural_score", "live_score"]
    FLOOR = 0.05
    SHIFT = 0.3

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
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            raise RuntimeError(
                "scipy 未安装，无法计算 Spearman。请执行: pip install scipy"
            )

        try:
            required = cls.ENGINE_COLS + cls.CHANNEL_COLS + [sales_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"df 缺少必需列: {missing}")

            df_work = df[required].dropna().copy()
            if len(df_work) < 3:
                raise ValueError(f"样本量不足（{len(df_work)}<3）")

            y = df_work[sales_col].values

            # --- 旧：三大引擎均匀权重 ---
            engine_rho: dict[str, float] = {}
            for col in cls.ENGINE_COLS:
                try:
                    rho, _ = spearmanr(df_work[col].values, y)
                    rho = 0.0 if math.isnan(rho) else float(rho)
                except Exception:
                    rho = 0.0
                engine_rho[col] = rho

            uniform_engine = {c: 1.0 / len(cls.ENGINE_COLS) for c in cls.ENGINE_COLS}
            old_engine_score = (
                df_work[cls.ENGINE_COLS[0]] * uniform_engine[cls.ENGINE_COLS[0]]
                + df_work[cls.ENGINE_COLS[1]] * uniform_engine[cls.ENGINE_COLS[1]]
                + df_work[cls.ENGINE_COLS[2]] * uniform_engine[cls.ENGINE_COLS[2]]
            ).values
            try:
                old_engine_sp, _ = spearmanr(old_engine_score, y)
                old_engine_sp = 0.0 if math.isnan(old_engine_sp) else float(old_engine_sp)
            except Exception:
                old_engine_sp = 0.0

            # --- 新引擎权重 ∝ max(0.05, ρ+0.3) ---
            raw_engine = {c: max(cls.FLOOR, engine_rho[c] + cls.SHIFT) for c in cls.ENGINE_COLS}
            tot = sum(raw_engine.values())
            new_engine_weights = {c: v / tot for c, v in raw_engine.items()}

            new_engine_score = (
                df_work[cls.ENGINE_COLS[0]] * new_engine_weights[cls.ENGINE_COLS[0]]
                + df_work[cls.ENGINE_COLS[1]] * new_engine_weights[cls.ENGINE_COLS[1]]
                + df_work[cls.ENGINE_COLS[2]] * new_engine_weights[cls.ENGINE_COLS[2]]
            ).values
            try:
                new_engine_sp, _ = spearmanr(new_engine_score, y)
                new_engine_sp = 0.0 if math.isnan(new_engine_sp) else float(new_engine_sp)
            except Exception:
                new_engine_sp = old_engine_sp

            engine_applied = new_engine_sp > old_engine_sp + 1e-9
            if not engine_applied:
                new_engine_weights = uniform_engine
                new_engine_sp = old_engine_sp
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
            )
        except Exception as exc:
            log.exception("EnsembleWeightTuner 失败，降级均匀权重: %s", exc)
            uniform_e = {c: 1.0 / len(cls.ENGINE_COLS) for c in cls.ENGINE_COLS}
            uniform_c = {c: 1.0 / len(cls.CHANNEL_COLS) for c in cls.CHANNEL_COLS}
            return EnsembleTuneResult(
                engine_weights=uniform_e,
                channel_weights=uniform_c,
                old_engine_spearman=0.0,
                new_engine_spearman=0.0,
                old_channel_spearman=0.0,
                new_channel_spearman=0.0,
                engine_rho={c: 0.0 for c in cls.ENGINE_COLS},
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


class ResidualDecomposer:
    """残差 ε = y_true - y_pred 分解：
    - 均值/std
    - 超预期款（ε > +2σ）
    - 不及预期款（ε < -2σ）
    - 系统偏差flag（mean/std 的量级判断）
    """

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

            upper = mu + 2.0 * sigma if sigma > 0 else mu
            lower = mu - 2.0 * sigma if sigma > 0 else mu

            id_col = "style_id" if "style_id" in meta_df.columns else None

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
                if e > upper:
                    try:
                        y_t = float(yt[i])
                        y_p = float(yp[i])
                    except Exception:
                        y_t = y_p = None
                    item.update({"y_true": y_t, "y_pred": y_p})
                    over.append(item)
                elif e < lower:
                    try:
                        y_t = float(yt[i])
                        y_p = float(yp[i])
                    except Exception:
                        y_t = y_p = None
                    item.update({"y_true": y_t, "y_pred": y_p})
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

            return ResidualDecomposeResult(
                residual_mean=float(mu),
                residual_std=float(sigma),
                overperformers=over,
                underperformers=under,
                system_bias_flag=flag,
                residuals=eps,
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


def _engine_score_ensemble(
    df: pd.DataFrame,
    engine_weights: dict[str, float],
    channel_weights: dict[str, float],
) -> pd.Series:
    """用 Loop3 已校准权重，将三大引擎 + 渠道合并为单一预测分 y_pred，
    用于残差分解。
    """
    engine_cols = EnsembleWeightTuner.ENGINE_COLS
    channel_cols = EnsembleWeightTuner.CHANNEL_COLS
    required = engine_cols + channel_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"残差分解缺少列: {missing}")

    engine_part = (
        df[engine_cols[0]] * engine_weights.get(engine_cols[0], 1 / 3)
        + df[engine_cols[1]] * engine_weights.get(engine_cols[1], 1 / 3)
        + df[engine_cols[2]] * engine_weights.get(engine_cols[2], 1 / 3)
    )
    channel_part = (
        df[channel_cols[0]] * channel_weights.get(channel_cols[0], 0.5)
        + df[channel_cols[1]] * channel_weights.get(channel_cols[1], 0.5)
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
        eng_row = {
            "persona_score": weighted_score,
            "channel_score": (natural + live) / 2,
            "price_value_score": perceived,
            "natural_score": natural,
            "live_score": live,
            sales_col: sales,
        }

        rows.append({
            "style_id": style_id,
            **feat_row,
            **persona_row,
            **eng_row,
        })

    return pd.DataFrame(rows)


def run_all_loops(
    brand_cfg: Any,
    history_df: pd.DataFrame,
    prediction_artifacts_dir: Path,
    sales_col: str = "sales",
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
        calibrated_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(prediction_artifacts_dir, Path):
            artifact_parent = prediction_artifacts_dir.parent
            calib_report_dir = artifact_parent / "calibration"
        else:
            calib_report_dir = calibrated_dir
        calib_report_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[Path] = []

        # ---------- Loop1 ----------
        log.info("=== Loop1: VLMFeatureCalibrator ===")
        r1 = VLMFeatureCalibrator.calibrate(history_df, sales_col=sales_col)
        l1_path = calibrated_dir / "loop1_vlm_feature_biases.yaml"
        _write_yaml(l1_path, {
            "applied": r1.applied,
            "old_spearman_avg": r1.old_spearman_avg,
            "new_spearman_avg": r1.new_spearman_avg,
            "per_feature_rho": r1.per_feature_rho,
            "feature_biases": r1.feature_biases,
        })
        output_files.append(l1_path)

        # ---------- Loop2 ----------
        log.info("=== Loop2: PersonaDistributionFitter ===")
        r2 = PersonaDistributionFitter.fit(history_df, sales_col=sales_col)
        l2_path = calibrated_dir / "loop2_persona_distribution_weights.yaml"
        _write_yaml(l2_path, {
            "applied": r2.applied,
            "old_spearman": r2.old_spearman,
            "new_spearman": r2.new_spearman,
            "delta_spearman": r2.new_spearman - r2.old_spearman,
            "lasso_raw_coef": r2.lasso_raw_coef,
            "persona_weights": r2.persona_weights,
        })
        output_files.append(l2_path)

        # ---------- Loop3 ----------
        log.info("=== Loop3: EnsembleWeightTuner ===")
        r3 = EnsembleWeightTuner.tune(history_df, sales_col=sales_col)
        l3_path = calibrated_dir / "loop3_ensemble_weights.yaml"
        _write_yaml(l3_path, {
            "applied": r3.applied,
            "engine": {
                "old_spearman": r3.old_engine_spearman,
                "new_spearman": r3.new_engine_spearman,
                "rho": r3.engine_rho,
                "weights": r3.engine_weights,
            },
            "channel": {
                "old_spearman": r3.old_channel_spearman,
                "new_spearman": r3.new_channel_spearman,
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
        residual_path = calibrated_dir / "residual_decompose.yaml"
        _write_yaml(residual_path, {
            "residual_mean": r4.residual_mean,
            "residual_std": r4.residual_std,
            "system_bias_flag": r4.system_bias_flag,
            "overperformers": r4.overperformers,
            "underperformers": r4.underperformers,
        })
        output_files.append(residual_path)

        # ---------- Markdown 校准报告 ----------
        md_lines: list[str] = []
        md_lines.append("# 校准循环报告 (Calibration Report)")
        md_lines.append("")
        md_lines.append(f"- 样本数: {len(history_df)}")
        md_lines.append(f"- 校准目录: `{calibrated_dir}`")
        md_lines.append(f"- 销量列: `{sales_col}`")
        md_lines.append("")

        md_lines.append("## Loop1 · VLM特征偏置")
        md_lines.append("")
        md_lines.append(f"- 生效: **{r1.applied}**")
        md_lines.append(f"- 旧Spearman均值: {r1.old_spearman_avg:.4f}")
        md_lines.append(f"- 新Spearman: {r1.new_spearman_avg:.4f}")
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
        md_lines.append("")
        md_lines.append("| 人设 | Lasso原始系数 | 归一化权重 |")
        md_lines.append("|------|---------------|------------|")
        non_zero_count = 0
        for col in PersonaDistributionFitter.PERSONA_COLS:
            raw = r2.lasso_raw_coef.get(col, 0.0)
            if abs(raw) > 1e-6:
                non_zero_count += 1
            md_lines.append(
                f"| {col} | {raw:.6f} | {r2.persona_weights.get(col, 0.0):.6f} |"
            )
        md_lines.append("")
        md_lines.append(f"- 非零人设: {non_zero_count}/{len(PersonaDistributionFitter.PERSONA_COLS)}")
        md_lines.append("")

        md_lines.append("## Loop3 · 集成权重")
        md_lines.append("")
        md_lines.append("### 三大引擎")
        md_lines.append(f"- 旧Spearman: {r3.old_engine_spearman:.4f}")
        md_lines.append(f"- 新Spearman: {r3.new_engine_spearman:.4f}")
        md_lines.append("")
        md_lines.append("| 引擎 | ρ(销量) | 权重 |")
        md_lines.append("|------|---------|------|")
        for col in EnsembleWeightTuner.ENGINE_COLS:
            md_lines.append(
                f"| {col} | {r3.engine_rho.get(col, 0.0):.4f} "
                f"| {r3.engine_weights.get(col, 0.0):.4f} |"
            )
        md_lines.append("")
        md_lines.append("### 双渠道")
        md_lines.append(f"- 旧Spearman: {r3.old_channel_spearman:.4f}")
        md_lines.append(f"- 新Spearman: {r3.new_channel_spearman:.4f}")
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

        md_lines.append(f"### 超预期款（ε > μ+2σ，共{len(r4.overperformers)}个）")
        md_lines.append("")
        if r4.overperformers:
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
        )
    except Exception as exc:
        log.exception("run_all_loops 异常退出: %s", exc)
        raise
