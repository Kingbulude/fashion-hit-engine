"""特征提取层：VLM输出10个BARS结构化特征
每款独立调用2-3个模型做交叉验证，中位数聚合。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import AppConfig, load_brand_profile
from .llm_client import BailianClient, LLMResponse
from .types import (
    BrandConfig,
    FeatureScore,
    StyleFeatures,
    StyleInfo,
    clamp,
    divergence,
    extract_json,
    median_aggregate,
    safe_float,
)

log = logging.getLogger(__name__)


def _resolve_bars_cfg(brand_cfg: BrandConfig | None, cfg: AppConfig | None) -> dict[str, Any]:
    """优先用 brand_cfg.features_bars；否则用旧 AppConfig.features；否则默认加载 tongzhuang-outdoor"""
    if brand_cfg is not None and brand_cfg.features_bars:
        return brand_cfg.features_bars
    if cfg is not None and cfg.features:
        return cfg.features
    fallback = load_brand_profile("tongzhuang-outdoor")
    return fallback.features_bars


# ========== BARS量表渲染 ==========
def _render_bars_prompt(features_cfg: dict[str, Any]) -> str:
    """把YAML中的10个BARS量表渲染为LLM可读的prompt"""
    features = features_cfg["features"]
    lines = ["【10个服装特征BARS评分量表】", ""]
    for key, feat in features.items():
        lines.append(f"### {key} · {feat['name']}")
        lines.append(f"定义：{feat['description']}")
        lines.append("评分档锚定：")
        for level_id, anchor in feat["anchors"].items():
            rng = anchor["range"]
            lines.append(
                f"  {rng[0]}-{rng[1]}分 [{anchor['label']}]：{anchor['description']}"
            )
        lines.append("")
    val_cfg = features_cfg.get("feature_validation", {})
    lines.append("【评分规则】")
    lines.append("1. 先看图片+FAB描述，然后对每个特征匹配最接近的锚定档")
    lines.append("2. 输出1-10之间的整数或一位小数（允许在两档之间插值，如4.5）")
    lines.append("3. confidence: 0-1，你对这个评分的把握度")
    lines.append("4. reason: 一句话说明，要引用具体视觉细节（如'明显oversize裤腿堆积→F02=3'）")
    lines.append(f"5. 多模型分歧>{val_cfg.get('divergence_threshold',2.0)}或confidence<{val_cfg.get('confidence_threshold',0.6)}需人工复核")
    lines.append("")
    lines.append("【输出格式】纯JSON，不要额外文字")
    lines.append('''{
  "F01_silhouette":     {"score": X, "confidence": X, "reason": "..."},
  "F02_clean_look":     {"score": X, "confidence": X, "reason": "..."},
  "F03_color_risk":     {"score": X, "confidence": X, "reason": "..."},
  "F04_function_visibility": {"score": X, "confidence": X, "reason": "..."},
  "F05_photogenic":     {"score": X, "confidence": X, "reason": "..."},
  "F06_wearability":    {"score": X, "confidence": X, "reason": "..."},
  "F07_pairing":        {"score": X, "confidence": X, "reason": "..."},
  "F08_fabric_perception": {"score": X, "confidence": X, "reason": "..."},
  "F09_brand_tone":     {"score": X, "confidence": X, "reason": "..."},
  "F10_uniqueness":     {"score": X, "confidence": X, "reason": "..."}
}''')
    return "\n".join(lines)


_BARS_PROMPT_CACHE: dict[str, str] = {}


def _get_bars_prompt(features_cfg: dict[str, Any]) -> str:
    key = str(id(features_cfg))
    if key not in _BARS_PROMPT_CACHE:
        _BARS_PROMPT_CACHE[key] = _render_bars_prompt(features_cfg)
    return _BARS_PROMPT_CACHE[key]


# ========== FeatureExtractionEngine (BrandConfig 注入 v2.0) ==========
class FeatureExtractionEngine:
    """基于 BrandConfig 的特征提取引擎

    向后兼容：不传 brand_cfg 时默认使用 tongzhuang-outdoor 品牌配置。
    """

    def __init__(
        self,
        brand_cfg: BrandConfig | None = None,
        *,
        llm_backend: str = "mock",
    ) -> None:
        if brand_cfg is None:
            brand_cfg = load_brand_profile("tongzhuang-outdoor")
        self.brand_cfg = brand_cfg
        self.llm_backend = llm_backend
        self._bars_cfg = brand_cfg.features_bars

    @property
    def bars_prompt(self) -> str:
        return _get_bars_prompt(self._bars_cfg)

    def _resolve_brand_context(self) -> str:
        personas_cfg = {"personas": self.brand_cfg.personas}
        if self.brand_cfg.child_identity_axes is not None:
            personas_cfg["child_identity_axes"] = self.brand_cfg.child_identity_axes
        return personas_cfg.get(
            "brand_context",
            "品牌定位：6-14岁功能性户外服饰，兼顾日常穿着与户外运动。",
        )

    def extract_mock(self, style_id: str, *, fixed_feature_scores: list[float] | None = None) -> StyleFeatures:
        """mock mode：生成兼容的特征分数（向后兼容旧mock逻辑）
        - fixed_feature_scores 传10个浮点数时，按BARS量表顺序覆盖F01-F10分数，用于冒烟测试确定性结果
        """
        import random
        random.seed(hash(style_id) & 0xFFFFFFFF)
        result = StyleFeatures(style_id=style_id)
        keys_ordered = list(self._bars_cfg["features"].keys())
        for i, key in enumerate(keys_ordered):
            fd = self._bars_cfg["features"][key]
            if fixed_feature_scores is not None and i < len(fixed_feature_scores):
                score = float(fixed_feature_scores[i])
            else:
                base = random.uniform(4.0, 8.0)
                score = round(base, 1)
            result.features[key] = FeatureScore(
                key=key,
                name=fd.get("name", key),
                category=fd.get("category", "design"),
                score=score,
                confidence=0.78,
                reason=f"视觉判断该特征锚定匹配度约{score:.0f}/10",
            )
        return result

    def extract(
        self,
        client: BailianClient,
        info: StyleInfo,
        *,
        progress: bool = False,
    ) -> StyleFeatures:
        return extract_style_features(
            client, info, cfg=None, brand_cfg=self.brand_cfg, progress=progress,
        )

    def extract_batch(
        self,
        client: BailianClient,
        styles: list[StyleInfo],
    ) -> dict[str, StyleFeatures]:
        return extract_batch(client, styles, cfg=None, brand_cfg=self.brand_cfg)


# ========== 单模型单款特征提取 ==========
def _extract_one_model(
    client: BailianClient,
    info: StyleInfo,
    features_cfg: dict[str, Any],
    *,
    model: str,
    brand_context: str,
) -> dict[str, dict[str, Any]]:
    """对一个款式用指定模型跑特征提取，返回 {feat_key: {score, confidence, reason}}"""
    user_msg = f"""
【品牌背景】
{brand_context}

【款式FAB描述】
{info.fab_description or '无FAB信息，仅从图片判断'}

【品类】{info.category or '未标注'}
【价格】{info.price}元
【季节】{info.season or '未标注'}

请根据下面的BARS量表，对提供的图片中的款式进行10个特征评分。
注意：评分是匹配"最接近的锚定档描述"，而不是主观的"好不好"。

{_get_bars_prompt(features_cfg)}
""".strip()

    resp: LLMResponse = client.generate_multimodal(
        user_msg,
        image_paths=info.images,
        model=model,
        temperature=0.2,
        max_tokens=3000,
    )
    if not resp.ok:
        raise RuntimeError(f"[特征提取{model}] {info.style_id} 失败: {resp.error}")
    try:
        parsed = extract_json(resp.content)
        assert isinstance(parsed, dict), f"解析出的不是dict而是{type(parsed)}"
        return parsed
    except Exception as e:
        log.warning("[%s] %s JSON解析失败，错误=%s，原文预览=%s",
                    info.style_id, model, e, resp.content[:200])
        raise


def _resolve_brand_context_from_inputs(
    cfg: AppConfig | None,
    brand_cfg: BrandConfig | None,
) -> str:
    if brand_cfg is not None:
        personas_wrapper = {"personas": brand_cfg.personas}
        if brand_cfg.child_identity_axes is not None:
            personas_wrapper["child_identity_axes"] = brand_cfg.child_identity_axes
        return personas_wrapper.get(
            "brand_context",
            f"品牌定位：{brand_cfg.brand_name}，{brand_cfg.brand_id}。",
        )
    if cfg is not None:
        return cfg.personas.get(
            "brand_context",
            "品牌定位：6-14岁功能性户外服饰，兼顾日常穿着与户外运动。",
        )
    return "品牌定位：6-14岁功能性户外服饰，兼顾日常穿着与户外运动。"


def _resolve_models_from_inputs(
    cfg: AppConfig | None,
    llm_backend: str | None,
) -> list[str]:
    if cfg is not None:
        return cfg.api.feature_extraction_models
    if llm_backend == "mock":
        return ["mock"]
    return ["qwen3-vl-plus", "qwen3.5-omni"]


# ========== 多模型交叉验证 ==========
def extract_style_features(
    client: BailianClient,
    info: StyleInfo,
    cfg: AppConfig | None = None,
    *,
    brand_cfg: BrandConfig | None = None,
    progress: bool = False,
    llm_backend: str | None = None,
) -> StyleFeatures:
    """对一个款式用 2-3 个模型提取特征，中位数聚合

    新接口建议：传 brand_cfg。
    旧兼容：不传 brand_cfg 时，若 cfg 提供则走旧 AppConfig 路径，
            否则默认 load_brand_profile('tongzhuang-outdoor')。
    """
    features_cfg = _resolve_bars_cfg(brand_cfg, cfg)
    val_cfg = features_cfg.get("feature_validation", {})
    div_thr = float(val_cfg.get("divergence_threshold", 2.0))
    conf_thr = float(val_cfg.get("confidence_threshold", 0.60))

    brand_context = _resolve_brand_context_from_inputs(cfg, brand_cfg)
    models = _resolve_models_from_inputs(cfg, llm_backend)

    # mock mode 快速路径
    if llm_backend == "mock" or (models and models[0] == "mock"):
        if brand_cfg is None:
            brand_cfg = load_brand_profile("tongzhuang-outdoor")
        engine = FeatureExtractionEngine(brand_cfg, llm_backend="mock")
        return engine.extract_mock(info.style_id)

    model_results: list[dict[str, dict[str, Any]]] = []
    pbar = tqdm(models, desc=f"特征提取[{info.style_id}]", leave=False, disable=not progress)
    for m in pbar:
        pbar.set_postfix_str(m)
        try:
            res = _extract_one_model(
                client, info, features_cfg,
                model=m, brand_context=brand_context,
            )
            model_results.append(res)
        except Exception as e:
            log.error("[%s] 模型%s特征提取失败: %s", info.style_id, m, e)

    if not model_results:
        raise RuntimeError(f"[{info.style_id}] 所有模型特征提取均失败")

    feat_defs = features_cfg["features"]
    result = StyleFeatures(style_id=info.style_id)

    all_keys = list(feat_defs.keys())
    for key in all_keys:
        fd = feat_defs[key]
        per_model_scores: list[float] = []
        per_model_conf: list[float] = []
        per_model_reasons: list[str] = []
        model_map: dict[str, float] = {}

        for m_idx, mr in enumerate(model_results):
            if key not in mr:
                continue
            item = mr[key]
            sc = clamp(safe_float(item.get("score"), 5.0), 1.0, 10.0)
            cf = clamp(safe_float(item.get("confidence"), 0.5), 0.0, 1.0)
            per_model_scores.append(sc)
            per_model_conf.append(cf)
            if item.get("reason"):
                per_model_reasons.append(str(item["reason"]))
            m_name = models[m_idx] if m_idx < len(models) else f"model{m_idx}"
            model_map[m_name] = sc

        final_score = clamp(median_aggregate(per_model_scores), 1.0, 10.0)
        final_conf = float(sum(per_model_conf) / max(1, len(per_model_conf)))
        div = divergence(per_model_scores)
        needs_review = (div > div_thr) or (final_conf < conf_thr)
        reason = per_model_reasons[0] if per_model_reasons else ""

        result.features[key] = FeatureScore(
            key=key,
            name=fd["name"],
            category=fd["category"],
            score=final_score,
            confidence=final_conf,
            reason=reason,
            model_scores=model_map,
            divergence=div,
            needs_review=needs_review,
        )

    return result


def extract_batch(
    client: BailianClient,
    styles: list[StyleInfo],
    cfg: AppConfig | None = None,
    *,
    brand_cfg: BrandConfig | None = None,
) -> dict[str, StyleFeatures]:
    """批量提取特征"""
    results: dict[str, StyleFeatures] = {}
    for s in tqdm(styles, desc="特征提取批次"):
        results[s.style_id] = extract_style_features(
            client, s, cfg, brand_cfg=brand_cfg, progress=False,
        )
    return results
