"""校准层训练：已弃用。
所有实现现迁移至 `src.core.optimization_kernel`。
本文件仅保留薄包装，转发调用并发出 DeprecationWarning。
"""
from __future__ import annotations

import warnings
warnings.warn(
    "calibration.py deprecated, use src.core.optimization_kernel instead",
    DeprecationWarning,
    stacklevel=2,
)

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .core.optimization_kernel import (
    VLMFeatureCalibrator,
    PersonaDistributionFitter,
    EnsembleWeightTuner,
    ResidualDecomposer,
    run_all_loops as _kernel_run_all_loops,
    VLMFeatureCalibrationResult,
    PersonaDistributionFitResult,
    EnsembleTuneResult,
    ResidualDecomposeResult,
    RunAllLoopsResult,
)

log = logging.getLogger(__name__)

__all__ = [
    "VLMFeatureCalibrator",
    "PersonaDistributionFitter",
    "EnsembleWeightTuner",
    "ResidualDecomposer",
    "run_all_loops",
    "train_calibration",
    "VLMFeatureCalibrationResult",
    "PersonaDistributionFitResult",
    "EnsembleTuneResult",
    "ResidualDecomposeResult",
    "RunAllLoopsResult",
]


def train_calibration(
    predictions: list[Any],
    out_dir: Path,
    *,
    min_samples: int = 10,
    alpha: float = 0.1,
) -> dict[str, Any] | None:
    """兼容旧 API：将 predictions 列表转为 DataFrame，再委托新内核
    Loop2 + 简易落盘。样本不足或 sklearn 缺失返回 None。
    """
    warnings.warn(
        "train_calibration 已弃用，请改用 run_all_loops",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        if len(predictions) < min_samples:
            log.warning(
                "样本量 %d 不足 %d，跳过校准层训练", len(predictions), min_samples,
            )
            return None

        rows = []
        sales_col = "sales"
        persona_cols = [f"P{i:02d}" for i in range(1, 31)]
        for p in predictions:
            info = getattr(p, "info", None)
            voting = getattr(p, "voting", None)
            votes = getattr(voting, "votes", []) if voting is not None else []
            row: dict[str, Any] = {
                "style_id": getattr(info, "style_id", "unknown"),
                sales_col: float(getattr(info, "sales_qty", 0) or 0),
            }
            by_pid = {getattr(v, "persona_id", f"P{i+1:02d}"): float(getattr(v, "final_score", 5.0))
                      for i, v in enumerate(votes)}
            for i, col in enumerate(persona_cols):
                fallback_pid = f"P{i+1:02d}"
                row[col] = by_pid.get(col, by_pid.get(fallback_pid, 5.0))
            rows.append(row)

        df = pd.DataFrame(rows)
        if sales_col not in df.columns:
            log.error("history_df 缺少销量列 %s", sales_col)
            return None

        try:
            res = PersonaDistributionFitter.fit(
                df, sales_col=sales_col, alpha=alpha, max_iter=5000,
            )
        except RuntimeError:
            return None

        import yaml
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fw_path = out_dir / "feature_weights.yaml"
        pa_path = out_dir / "persona_weights_adjustment.yaml"

        simple_fw = {
            "meta": {
                "samples": len(predictions),
                "target": sales_col,
                "alpha": alpha,
                "non_zero_features": sum(1 for v in res.lasso_raw_coef.values() if abs(v) > 1e-6),
            },
            "feature_weights": {k: v for k, v in res.lasso_raw_coef.items() if abs(v) > 1e-6},
        }
        fw_path.write_text(
            yaml.safe_dump(simple_fw, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )

        pa_dump = {}
        for col in persona_cols:
            pa_dump[col] = {
                "adjust_factor": round(res.persona_weights.get(col, 1.0 / len(persona_cols)) * len(persona_cols), 3),
                "weight": round(res.persona_weights.get(col, 0.0), 6),
            }
        pa_path.write_text(
            yaml.safe_dump(pa_dump, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )

        return {
            "samples": len(predictions),
            "r2": None,
            "non_zero_features": sum(1 for v in res.lasso_raw_coef.values() if abs(v) > 1e-6),
            "feature_weights_path": str(fw_path),
            "persona_adjust_path": str(pa_path),
        }
    except Exception as exc:
        log.exception("train_calibration (compat wrapper) 失败: %s", exc)
        return None


def run_all_loops(
    brand_cfg: Any,
    history_df: pd.DataFrame,
    prediction_artifacts_dir: Path,
    sales_col: str = "sales",
) -> RunAllLoopsResult:
    """薄包装：直接转发至 `src.core.optimization_kernel.run_all_loops`。"""
    return _kernel_run_all_loops(
        brand_cfg=brand_cfg,
        history_df=history_df,
        prediction_artifacts_dir=prediction_artifacts_dir,
        sales_col=sales_col,
    )
