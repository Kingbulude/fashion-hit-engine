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
    # v1.4.50+: grade_norm 是新引擎名，Loop3 产物 key 和 ensemble_engine 期望的 key 相同
    "grade_norm": "grade_norm",
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
        # v1.4.44+: 分类增益 + PatternMiner 路径
        self.classification_gains: dict[str, dict[str, float]] = {}
        self.pattern_yaml_path: str | None = None   # calibrated_dir/memory/*_patterns.yaml

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
        if self.pattern_yaml_path:
            parts.append(f"pattern={Path(self.pattern_yaml_path).name}")
        if not parts:
            return "CalibrationResult(empty)"
        return "CalibrationResult(" + ", ".join(parts) + ")"


def _resolve_yaml_path(calibrated_dir: Path, filename: str) -> Path | None:
    """v1.4.51+: 优先从 calibrated_dir 根目录找，找不到 fallback 到 _pending/ 子目录。

    run_all_loops(auto_apply=True)  → 产物在根目录（已批准生效）
    run_all_loops(auto_apply=False) → 产物在 _pending/（待人工审核）
    calibration_loader 两个都要能读到。
    """
    direct = calibrated_dir / filename
    if direct.exists():
        return direct
    pending = calibrated_dir / "_pending" / filename
    if pending.exists():
        return pending
    return None


def load_calibration(calibrated_dir) -> CalibrationResult:
    """从 calibrated_dir 加载所有 3Loop 校准产物。"""
    result = CalibrationResult()

    calibrated_dir = Path(calibrated_dir) if calibrated_dir else None
    if not calibrated_dir or not calibrated_dir.is_dir():
        log.info("calibrated 目录不存在，跳过加载: %s", calibrated_dir)
        return result

    # ===== Loop3: engine weights + channel split =====
    l3_path = _resolve_yaml_path(calibrated_dir, "loop3_ensemble_weights.yaml")
    if l3_path is None:
        log.info("Loop3 YAML 不存在（根目录和 _pending/ 都没找到）")
    else:
        log.info("Loop3 YAML 路径: %s (%s)",
                 l3_path, "待审核_pending" if "_pending" in str(l3_path) else "已应用")
        try:
            data = yaml.safe_load(l3_path.read_text(encoding="utf-8"))
            eng = (data.get("engine") or {}).get("weights") or {}
            ch = (data.get("channel") or {}).get("weights") or {}

            # v1.4.50+: 支持 3 或 4 引擎（grade_norm 可选）
            if eng:
                mapped_eng = {
                    _LOOP3_ENGINE_KEY_MAP.get(k, k): float(v)
                    for k, v in eng.items()
                }
                # 必须的 3 个基础引擎
                base_expected = {"persona_voting", "channel_scoring", "price_value"}
                has_grade_norm = "grade_norm" in mapped_eng
                expected_keys = base_expected | ({"grade_norm"} if has_grade_norm else set())

                if base_expected.issubset(mapped_eng.keys()):
                    # ===== v1.4.53: 只要 3 个基础引擎齐了就加载 =====
                    # 不再检查"是否偏离均匀"——校准跑完后权重均匀也是有意义的结果
                    # （说明在当前数据上没找到改进方向）；冷启动用 default_engine_weights
                    # 只有在 loop3 YAML **不存在**时才 fallback 到默认值
                    result.engine_weights = mapped_eng
                    result.loaded_files.append("loop3_ensemble_weights.yaml")
                    gn_hint = " + grade_norm" if has_grade_norm else ""
                    log.info("✅ Loop3 engine_weights 已加载 (%d引擎%s): %s",
                             len(base_expected) + (1 if has_grade_norm else 0),
                             gn_hint, mapped_eng)

            if ch:
                mapped_ch = {
                    _LOOP3_CHANNEL_KEY_MAP.get(k, k): float(v)
                    for k, v in ch.items()
                }
                expected_ch = {"natural", "live_stream"}
                if expected_ch.issubset(mapped_ch.keys()):
                    # v1.4.53: 齐了就加载，不再检查 0.5/0.5 均匀
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
    l2_path = _resolve_yaml_path(calibrated_dir, "loop2_persona_distribution_weights.yaml")
    if l2_path is not None:
        try:
            data = yaml.safe_load(l2_path.read_text(encoding="utf-8"))
            pw = data.get("persona_weights") or {}

            # v1.4.53: 只要 persona_weights 有值就加载
            # 之前要求 Lasso coef > 0 才加载——但即使全为 0 也意味着"校准后人设分布不变"
            if pw:
                result.persona_weights = {k: float(v) for k, v in pw.items()}
                result.loaded_files.append("loop2_persona_distribution_weights.yaml")
                nz_count = sum(1 for v in pw.values() if abs(float(v) - 1/30) > 0.01)
                log.info("✅ Loop2 persona_weights 已加载 (%d/%d 人设权重有调整)",
                         nz_count, len(pw))

            old_sp = data.get("old_spearman", 0)
            new_sp = data.get("new_spearman", 0)
            result.spearman_gains["loop2_persona"] = new_sp - old_sp

        except Exception as e:
            log.warning("加载 loop2_persona_distribution_weights.yaml 失败: %s", e)

    # ===== Loop1: feature biases =====
    l1_path = _resolve_yaml_path(calibrated_dir, "loop1_vlm_feature_biases.yaml")
    if l1_path is not None:
        try:
            data = yaml.safe_load(l1_path.read_text(encoding="utf-8"))
            biases = data.get("feature_biases") or {}

            # v1.4.53: 只要 feature_biases 有值就加载
            # 之前要求至少一个 bias != 1.0 才加载——但即使全 1.0 也意味着"特征评分不需要调整"
            if biases:
                result.feature_biases = {k: float(v) for k, v in biases.items()}
                result.loaded_files.append("loop1_vlm_feature_biases.yaml")
                non_default = sum(1 for v in biases.values() if abs(float(v) - 1.0) > 0.01)
                log.info("✅ Loop1 feature_biases 已加载 (%d/%d 个有调整)",
                         non_default, len(biases))

            old_sp = data.get("old_spearman_avg", 0)
            new_sp = data.get("new_spearman_avg", 0)
            result.spearman_gains["loop1_vlm"] = new_sp - old_sp

        except Exception as e:
            log.warning("加载 loop1_vlm_feature_biases.yaml 失败: %s", e)

    # ===== v1.4.44+: 分类增益（从各 loop YAML 的 classification 字段提取）=====
    for loop_name, yaml_name in [
        ("loop1_vlm", "loop1_vlm_feature_biases.yaml"),
        ("loop2_persona", "loop2_persona_distribution_weights.yaml"),
    ]:
        p = calibrated_dir / yaml_name
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                cls = data.get("classification") or {}
                if cls:
                    result.classification_gains[loop_name] = {
                        "f1_delta": float(cls.get("delta_f1", 0.0)),
                        "p_at_k_delta": float(cls.get("delta_p_at_k", 0.0)),
                    }
            except Exception as e:
                log.debug("读取 %s 分类增益失败: %s", yaml_name, e)

    # Loop3 分类增益（复用上面已解析的 l3_path）
    if l3_path is not None:
        try:
            data = yaml.safe_load(l3_path.read_text(encoding="utf-8")) or {}
            for sub in ("engine", "channel"):
                cls = (data.get(sub) or {}).get("classification") or {}
                if cls:
                    key = f"loop3_{sub}"
                    result.classification_gains[key] = {
                        "f1_delta": float(cls.get("delta_f1", 0.0)),
                        "p_at_k_delta": float(cls.get("delta_p_at_k", 0.0)),
                    }
        except Exception as e:
            log.debug("读取 loop3 分类增益失败: %s", e)

    # ===== v1.4.44+: PatternMiner YAML（calibrated_dir/memory/ 下最新的）=====
    memory_dir = calibrated_dir / "memory"
    if memory_dir.is_dir():
        yamls = sorted(
            memory_dir.glob("*_patterns.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if yamls:
            result.pattern_yaml_path = str(yamls[0])
            result.loaded_files.append(yamls[0].name)
            log.info("🧠 PatternMiner YAML 已发现: %s", yamls[0].name)

    if result.loaded_files:
        log.info("📥 共加载 %d 个校准产物: %s",
                 len(result.loaded_files), result.loaded_files)
        log.info("📈 Spearman 增益: %s", result.spearman_gains)
        if result.classification_gains:
            log.info("🎯 分类增益(F1/P@k Δ): %s", result.classification_gains)
        if result.pattern_yaml_path:
            log.info("🧠 PatternMiner: %s", result.pattern_yaml_path)
    else:
        log.info("calibrated 目录没有有效产物，使用默认权重")

    return result
