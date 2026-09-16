"""特征提取层：VLM输出10个BARS结构化特征
每款独立调用2-3个模型做交叉验证，中位数聚合。
"""
from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import AppConfig, load_brand_profile
from .llm_client import BailianClient, LLMResponse, resolve_and_dedupe_models
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
    """优先用 brand_cfg.features_bars；否则用旧 AppConfig.features；否则默认加载 mipo"""
    if brand_cfg is not None and brand_cfg.features_bars:
        return brand_cfg.features_bars
    if cfg is not None and cfg.features:
        return cfg.features
    fallback = load_brand_profile("mipo")
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
    lines.append("【评分规则（关键！请严格遵守）】")
    lines.append("1. 第一步：观察图片+FAB描述，匹配每个特征最接近的锚定档（客观描述阶段）")
    lines.append("2. 第二步：该锚定档的「对品牌目标客群销量暗示」是什么？高分档=对销量是显著加分项，低分档=对销量是减分项/风险")
    lines.append("3. 第三步：综合判断后给出最终分（1-10）。可以跨档取分，比如客观在4档但你认为销量贡献低于锚定暗示，可以打到5分而不是7分")
    lines.append("4. confidence: 0-1，你对这个评分的把握度")
    lines.append("5. reason: 必须分两段写，格式是「[视觉判断]...；[销量影响]...」。")
    lines.append("   - [视觉判断] 要引用图片里的具体细节（如'明显oversize、裤腿堆积、魔术贴调节'）")
    lines.append("   - [销量影响] 要直接说「这款放在本品牌是加分项/减分项/中性，为什么」（如'宽松廓形对10岁男孩是主推款型→加分'；'低饱和基础色在直播间缺乏记忆点→减分'）")
    lines.append("6. 区分度要求：10款在同一批次的同一特征上，分数分布要有明显差异（标准差≥1.5），不能都打7-8分")
    lines.append(f"7. 多模型分歧>{val_cfg.get('divergence_threshold',2.0)}或confidence<{val_cfg.get('confidence_threshold',0.6)}需人工复核")
    lines.append("")
    lines.append("【反例警告】")
    lines.append("- ❌ 错误理由：'宽松版型，有垂坠感'（只有视觉描述，没有销量影响判断）")
    lines.append("- ✅ 正确理由：'宽松oversize版型，比正常大1个码，有魔术贴调节；对10-14岁男孩是主推款型，直播间能引流→高销加分'")
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

    向后兼容：不传 brand_cfg 时默认使用 mipo 品牌配置。
    """

    def __init__(
        self,
        brand_cfg: BrandConfig | None = None,
        *,
        llm_backend: str = "mock",
    ) -> None:
        if brand_cfg is None:
            brand_cfg = load_brand_profile("mipo")
        self.brand_cfg = brand_cfg
        self.llm_backend = llm_backend
        self._bars_cfg = brand_cfg.features_bars

    @property
    def bars_prompt(self) -> str:
        return _get_bars_prompt(self._bars_cfg)

    def _resolve_brand_context(self) -> str:
        personas_cfg = {"personas": self.brand_cfg.personas}
        if self.brand_cfg.persona_axes:
            personas_cfg.update(self.brand_cfg.persona_axes)
        return personas_cfg.get(
            "brand_context",
            f"品牌定位：{self.brand_cfg.brand_name}。",
        )

    def extract_mock(self, style_id: str, *, fixed_feature_scores: list[float] | None = None) -> StyleFeatures:
        """mock mode：生成兼容的特征分数（向后兼容旧mock逻辑）
        - fixed_feature_scores 传10个浮点数时，按BARS量表顺序覆盖F01-F10分数，用于冒烟测试确定性结果
        """
        import random
        # 注：原 hash() 跨进程随机化（PYTHONHASHSEED），导致 mock 特征每进程不同、
        # smoke test 跨进程非确定性（sp_sales 在 0.90~0.99 漂移、阈值 0.92 随机失败）。
        # 改用 zlib.crc32 提供确定性 hash。
        random.seed(zlib.crc32(style_id.encode()))
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

⚠️ 关键约束（请严格遵守）：
- 图片中可能是模特全身照，**只评价本款主产品**（FAB描述中提到的品类），不要评价模特身上的其他搭配（裤子、内搭、帽子、鞋子等）
- 例如：FAB描述是"户外机能外套" → 只评价外套，裤子/内搭完全忽略
- 例如：FAB描述是"速干运动短裤" → 只评价短裤，上衣/外套完全忽略
- 如果图片中模特身上有多个单品，以FAB描述中的品类为准锁定评价对象

评分原则：
1. 先匹配"最接近的锚定档描述"，给出客观档位分。
2. 再站在品牌目标客群的购买决策角度，判断这个档位对销量的影响：
   - 该特征表现越好，越能打动目标客群并促成购买 → 取该档位高分区
   - 该特征表现越差，越容易劝退目标客群 → 取该档位低分区
3. 最终分数要体现"这款相对本品牌其他款式的畅销潜力"，而不是单纯"图片好不好看"。

请对每个特征给出分数(1-10)、置信度(0-1)和一句话理由，理由要说明为什么这个分数意味着高/中/低销量潜力。

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
        parsed = extract_json(resp.content, as_dict=True)
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
        if brand_cfg.persona_axes:
            personas_wrapper.update(brand_cfg.persona_axes)
        return personas_wrapper.get(
            "brand_context",
            f"品牌定位：{brand_cfg.brand_name}，{brand_cfg.brand_id}。",
        )
    if cfg is not None:
        return cfg.personas.get(
            "brand_context",
            "品牌定位：未指定品牌。",
        )
    return "品牌定位：未指定品牌。"


def _resolve_models_from_inputs(
    cfg: AppConfig | None,
    llm_backend: str | None,
) -> list[str]:
    if cfg is not None:
        return cfg.api.feature_extraction_models
    if llm_backend == "mock":
        return ["mock"]
    if llm_backend == "local":
        # Ollama 本地 VLM
        return ["qwen2.5vl:7b"]
    if llm_backend == "hybrid":
        # hybrid 模式 VLM client 是 ZhipuClient，用云端 VLM 模型名；
        # Ollama 模型名会通过 _ZHIPU_MODEL_MAP 自动映射，这里用标准云端名更直观
        return ["qwen-vl-plus"]
    return ["qwen-vl-plus"]


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
            否则默认 load_brand_profile('mipo')。
    """
    features_cfg = _resolve_bars_cfg(brand_cfg, cfg)
    val_cfg = features_cfg.get("feature_validation", {})
    div_thr = float(val_cfg.get("divergence_threshold", 2.0))
    conf_thr = float(val_cfg.get("confidence_threshold", 0.60))

    brand_context = _resolve_brand_context_from_inputs(cfg, brand_cfg)
    models = _resolve_models_from_inputs(cfg, llm_backend)
    # VLM 模型去重（同人设投票：映射后重复的只留一个）
    models = resolve_and_dedupe_models(client, models or [], is_vlm=True)

    # mock mode 快速路径
    if llm_backend == "mock" or (models and models[0] == "mock"):
        if brand_cfg is None:
            brand_cfg = load_brand_profile("mipo")
        engine = FeatureExtractionEngine(brand_cfg, llm_backend="mock")
        return engine.extract_mock(info.style_id)

    model_results: list[dict[str, dict[str, Any]]] = []
    errors: list[str] = []
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
            errors.append(f"{m}: {e}")

    # ========== VLM 级 Fallback 第 2 层 ==========
    # 主 client（智谱 VLM）全挂 → 自动切本地 Ollama VLM 兜底
    # 典型场景：glm-4.6v-flash 连续 3 次 1305 模型拥塞
    if not model_results and llm_backend in ("hybrid", "zhipu"):
        log.warning(
            "[%s] 云端 VLM 全部失败 (%s)，自动切 Ollama VLM fallback ...",
            info.style_id, "; ".join(errors[:2]),
        )
        try:
            from .llm_client import OllamaClient
            ollama_client = OllamaClient()
            ollama_vlm_models = ["qwen2.5vl:7b", "qwen3-vl:8b"]
            for om in ollama_vlm_models:
                try:
                    log.info("[%s] 尝试 Ollama VLM: %s", info.style_id, om)
                    res = _extract_one_model(
                        ollama_client, info, features_cfg,
                        model=om, brand_context=brand_context,
                    )
                    model_results.append(res)
                    errors = []  # Ollama 成功了，清空之前的错误
                    log.info("[%s] Ollama VLM %s fallback 成功！", info.style_id, om)
                    break
                except Exception as oe:
                    log.warning("[%s] Ollama VLM %s 也失败: %s", info.style_id, om, oe)
                    errors.append(f"ollama/{om}: {oe}")
        except Exception as fe:
            log.warning("[%s] Ollama fallback 也不可用: %s", info.style_id, fe)

    if not model_results:
        # ========== VLM 级 Fallback 第 3 层：brand 默认特征分兜底 ==========
        # 两个 VLM 都挂 → 用 BARS anchors 中点作为默认分，让 pipeline 继续跑
        # 精度会降（所有款都是 brand 平均水平），但不会整批废
        log.warning(
            "[%s] 所有 VLM 均失败 (%s)，使用 brand 默认特征分兜底",
            info.style_id, "; ".join(errors[:2]),
        )
        if brand_cfg is None:
            brand_cfg = load_brand_profile("mipo")
        feat_defs = features_cfg["features"]
        default_result = {}
        for key, fd in feat_defs.items():
            anchors = fd.get("anchors", {})
            low_score = float(anchors.get("low", {}).get("score", 3.0))
            high_score = float(anchors.get("high", {}).get("score", 8.0))
            mid_score = round((low_score + high_score) / 2, 1)
            default_result[key] = {
                "score": mid_score,
                "confidence": 0.3,  # 低置信度，校准层会忽略
                "reason": f"VLM 全部失败，使用 brand 默认分（BARS anchors 中点 {mid_score}）",
            }
        model_results.append(default_result)
        errors = []

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
            # 防御：Ollama 偶尔把单个特征包装成 list（如 {"quality": [{"score": 7}]}）
            if isinstance(item, list):
                if len(item) > 0 and isinstance(item[0], dict):
                    item = item[0]
                else:
                    log.warning("[%s] %s 模型%s 特征%s值异常: list 长度=%d 元素类型=%s，跳过",
                                info.style_id, models[m_idx] if m_idx < len(models) else f"model{m_idx}",
                                m_idx, key, len(item), type(item[0]) if item else "empty")
                    continue
            if not isinstance(item, dict):
                log.warning("[%s] 模型%s 特征%s值类型异常: %s，跳过",
                            info.style_id, m_idx, key, type(item))
                continue
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
