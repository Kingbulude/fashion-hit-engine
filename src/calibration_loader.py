"""Calibration weight loader — 把 3Loop 校准产物加载到 pipeline 运行时。

加载时机：PredictionPipeline.__init__ 时自动调用。
产物位置：brand_profiles/<brand>/calibrated/
文件格式：
  loop1_vlm_feature_biases.yaml   → feature_biases (F01~F10)
  loop2_persona_distribution_weights.yaml → persona_weights (P01~P30)
  loop3_ensemble_weights.yaml     → engine_weights + channel_split

关键设计：
- Loop3 产物 key 名和 synthesise_final_score 需要的不同，这里做映射：
    persona_score       → persona_voting
    channel_score       → channel_scoring
    price_value_score   → price_value
    natural_score       → natural
    live_score          → live_stream
- 如果产物 applied=false 或权重没改变（和默认一样），就不覆盖
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# ========== Key 映射表 ==========
# Loop3 engine 产物 key  →  synthesise_final_score 需要的 key
_LOOP3_ENGINE_KEY_MAP = {
    "persona_score": "persona_voting",
    "channel_score": "channel_scoring",
    "price_value_score": "price_value",
}

# Loop3 channel 产物 key  →  synthesise_final_score 需要的 key
_LOOP3_CHANNEL_KEY_MAP = {
    "natural_score": "natural",
    "live_score": "live_stream",
}


class CalibrationResult:
    """把 3Loop 校准产物打包成 pipeline 可以直接用的对象。"""

    def __init__(self):
        self.engine_weights: dict[str, float] | None = None
        self.channel_split: dict[str, float] | None = None
        self.persona_weights: dict[str, float] | None = None
        self.feature_biases: dict[str, float] | None = None
        self.loaded_files: list[str] = []
        self.spearman_gains: dict[str, float] = {}

    def __repr__(self) -> str:
        parts = []
        if self.engine_weights:
            parts.append(f"engine={self.engine_weights}")
        if self.channel_split:
            parts.append(f"channel={self.channel_split}")
        if self.persona_weights:
            nz = sum(1 for v in self.persona_weights.values() if v > 0.001)
            parts.append(f"persona({nz}/30非均匀)")
        if self.feature_biases:
            non_one = sum(1 for v in self.feature_biases.values() if abs(v - 1.0) > 0.01)
            parts.append(f"biases({non_one}/10有调整)")
        if not parts:
            return "CalibrationResult(empty)"
        return "CalibrationResult(" + ", ".join(parts) + ")"


def load_calibration(calibrated_dir) -> CalibrationResult:
    """从 calibrated_dir 加载所有 3Loop 校准产物。"""
    result = CalibrationResult()

    calibrated_dir = Path(calibrated_dir) if calibrated_dir else None
    if not calibrated_dir or not calibrated_dir.is_dir():
        log.info("calibrated 目录不存在，跳过加载: %s", calibrated_dir)
        return result

    # ===== Loop3: engine weights + channel split =====
    l3_path = calibrated_dir / "loop3_ensemble_weights.yaml"
    if l3_path.exists():
        try:
            data = yaml.safe_load(l3_path.read_text(encoding="utf-8"))
            eng = (data.get("engine") or {}).get("weights") or {}
            ch = (data.get("channel") or {}).get("weights") or {}

            # 只在权重确实改变时才加载（避免覆盖默认均匀权重）
            if eng:
                mapped_eng = {
                    _LOOP3_ENGINE_KEY_MAP.get(k, k): float(v)
                    for k, v in eng.items()
                }
                # 确认三个 key 都有了
                expected = {"persona_voting", "channel_scoring", "price_value"}
                if expected.issubset(mapped_eng.keys()):
                    # 只有当至少有一个权重偏离 1/3 时才算有效
                    uniform = abs(mapped_eng.get("persona_voting", 0) - 1 / 3) < 0.001
                    if not uniform:
                        result.engine_weights = mapped_eng
                        result.loaded_files.append("loop3_ensemble_weights.yaml")
                        log.info("✅ Loop3 engine_weights 已加载: %s", mapped_eng)
                    else:
                        log.info("Loop3 engine_weights 仍是均匀权重，跳过 (old_spearman 没变)")

            if ch:
                mapped_ch = {
                    _LOOP3_CHANNEL_KEY_MAP.get(k, k): float(v)
                    for k, v in ch.items()
                }
                expected_ch = {"natural", "live_stream"}
                if expected_ch.issubset(mapped_ch.keys()):
                    uniform_ch = abs(mapped_ch.get("natural", 0) - 0.5) < 0.001
                    if not uniform_ch:
                        result.channel_split = mapped_ch
                        log.info("✅ Loop3 channel_split 已加载: %s", mapped_ch)

            # 记录 spearman 增益
            if eng:
                old_sp = (data.get("engine") or {}).get("old_spearman", 0)
                new_sp = (data.get("engine") or {}).get("new_spearman", 0)
                result.spearman_gains["loop3_engine"] = new_sp - old_sp

        except Exception as e:
            log.warning("加载 loop3_ensemble_weights.yaml 失败: %s", e)

    # ===== Loop2: persona 分布权重 =====
    l2_path = calibrated_dir / "loop2_persona_distribution_weights.yaml"
    if l2_path.exists():
        try:
            data = yaml.safe_load(l2_path.read_text(encoding="utf-8"))
            pw = data.get("persona_weights") or {}
            lasso = data.get("lasso_raw_coef") or {}

            # 只有当至少有一个人设的 Lasso coef > 0 时才加载
            nz_lasso = {k: float(v) for k, v in lasso.items() if float(v) > 0}
            if nz_lasso and pw:
                result.persona_weights = {k: float(v) for k, v in pw.items()}
                result.loaded_files.append("loop2_persona_distribution_weights.yaml")
                log.info("✅ Loop2 persona_weights 已加载 (%d 人设非零)", len(nz_lasso))
            else:
                log.info("Loop2 Lasso coef 全为 0（mock 数据信噪比低），跳过加载")

            old_sp = data.get("old_spearman", 0)
            new_sp = data.get("new_spearman", 0)
            result.spearman_gains["loop2_persona"] = new_sp - old_sp

        except Exception as e:
            log.warning("加载 loop2_persona_distribution_weights.yaml 失败: %s", e)

    # ===== Loop1: feature biases =====
    l1_path = calibrated_dir / "loop1_vlm_feature_biases.yaml"
    if l1_path.exists():
        try:
            data = yaml.safe_load(l1_path.read_text(encoding="utf-8"))
            biases = data.get("feature_biases") or {}

            # 只有当至少有一个 bias 不是 1.0 时才加载
            non_default = {k: float(v) for k, v in biases.items() if abs(float(v) - 1.0) > 0.01}
            if non_default:
                result.feature_biases = {k: float(v) for k, v in biases.items()}
                result.loaded_files.append("loop1_vlm_feature_biases.yaml")
                log.info("✅ Loop1 feature_biases 已加载 (%d/%d 个有调整)",
                         len(non_default), len(biases))
            else:
                log.info("Loop1 feature_biases 全为 1.0（内核 bug 或无改进），跳过加载")

            old_sp = data.get("old_spearman_avg", 0)
            new_sp = data.get("new_spearman_avg", 0)
            result.spearman_gains["loop1_vlm"] = new_sp - old_sp

        except Exception as e:
            log.warning("加载 loop1_vlm_feature_biases.yaml 失败: %s", e)

    if result.loaded_files:
        log.info("📥 共加载 %d 个校准产物: %s",
                 len(result.loaded_files), result.loaded_files)
        log.info("📈 Spearman 增益: %s", result.spearman_gains)
    else:
        log.info("calibrated 目录没有有效产物，使用默认权重")

    return result
