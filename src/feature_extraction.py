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
from .llm_client import BailianClient, LLMResponse, is_fatal_quota_error, resolve_and_dedupe_models
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


# ========== BARS量表渲染（纯视觉锚定版 v1.4.90）==========
def _render_bars_prompt(features_cfg: dict[str, Any]) -> str:
    """把YAML中的10个BARS量表渲染为LLM可读的prompt

    v1.4.90 重写：VLM 只做「眼睛」——纯客观视觉描述 + 视觉锚定档匹配，
    不做任何销量预判。销量判断由人设 LLM 独立完成。
    score 字段由代码根据 anchor_level + anchor.range 自动计算（区间中点），
    VLM 不输出 score。
    """
    features = features_cfg["features"]
    lines = [
        "【10个服装特征 · 纯视觉锚定量表】",
        "",
        "⚠️ 你的角色是「视觉观察员」，不是销售预测分析师。",
        "你只做两件事：① 描述眼睛看到的东西；② 匹配视觉锚定档。",
        "**绝对禁止**：判断好不好卖、好不好看、适合谁穿、受众广不广、退货率、性价比——这些都不是你该管的。",
        "",
    ]
    for key, feat in features.items():
        lines.append(f"### {key} · {feat['name']}")
        lines.append(f"定义：{feat['description']}")
        lines.append("视觉锚定档（1-5档，从低到高描述视觉属性）：")
        for level_id, anchor in feat["anchors"].items():
            rng = anchor["range"]
            lines.append(
                f"  档{level_id} [{anchor['label']}]：{anchor['description']}"
            )
        lines.append("")

    val_cfg = features_cfg.get("feature_validation", {})
    lines.append("【执行规则 — 严格两步】")
    lines.append("")
    lines.append("=== 第一步：纯客观视觉描述 ===")
    lines.append("只描述图片中你**实际看到**的视觉事实，用短句列细节：")
    lines.append('  格式：直接写细节关键词（如"明显宽松廓形、裤腿有魔术贴调节、面料表面有轻微反光"）')
    lines.append("  ❌ 禁止使用价值判断词：好/不好、好看/难看、加分/减分、适合/不适合、值得/不值得、热销/滞销、便宜/贵")
    lines.append('  ❌ 禁止推测品牌意图或消费者反应：不要说"父母会喜欢"、"直播间能引流"、"孩子会穿"')
    lines.append("  ✅ 可以写：颜色名称、版型、面料外观、功能元素细节、有无帽子/口袋/反光条、轮廓宽窄、是否宽松、线条是否干净")
    lines.append('  如果图片/角度/清晰度导致某特征**无法可靠判断**，写视觉描述为："无法判断：+ 具体原因（如"图片模糊看不出面料纹理"、"无帽子故无法判断帽檐特征"）"')
    lines.append("  无法判断时 anchor_level=null, confidence=0.0")
    lines.append("")
    lines.append("=== 第二步：视觉锚定档匹配 ===")
    lines.append("将你第一步的视觉描述，匹配到上表中**视觉上最接近的档位（档1-档5）**。")
    lines.append("  ⚠️ 匹配依据只能是「视觉特征像不像」，不能是「觉得哪档更好卖」。")
    lines.append("  例：F03 如果主图是黄色+深灰撞色 → 看视觉上撞色程度 → 档2 或 档1（看色相差）")
    lines.append('  ❌ 错误匹配理由："撞色可能受众窄→档1"（带了销量判断）')
    lines.append('  ✅ 正确匹配理由："黄色和深灰色相差约120°→视觉上属于高饱和撞色→档1"')
    lines.append("")
    lines.append("【多图颜色场景识别（最高优先级）】如果输入了多张图且衣服颜色不同：")
    lines.append("  1. 先判断颜色差异是「同一件衣服的拼接/撞色设计」还是「同一款的不同SKU分色」")
    lines.append("     · 拼接/撞色 = 不同色块在同一件衣服上（一张图同时出现多色）")
    lines.append("     · SKU分色 = 各图是不同件衣服但剪裁/款式完全相同")
    lines.append("  2. 若为 SKU 分色：F03/F05/F09/F10 只评主推色（通常是第一张/主图的颜色）")
    lines.append("  3. 若为同一件衣服的撞色设计：F03 按实际撞色视觉特征匹配档")
    lines.append("  4. 其他特征（F01/F02/F04/F06/F07/F08）不受颜色影响")
    lines.append("")
    lines.append("【输出格式】纯JSON，不要任何额外文字。只有3个字段，没有score、没有reason：")
    lines.append('''{
  "F01_silhouette":     {"visual_description": "...你看到的细节...", "anchor_level": 1-5的整数或null, "confidence": 0.0-1.0},
  "F02_clean_look":     {"visual_description": "...", "anchor_level": null, "confidence": 0.0},
  "F03_color_safety":   {"visual_description": "...", "anchor_level": 3, "confidence": 0.85},
  "F04_function_visibility": {"visual_description": "...", "anchor_level": 4, "confidence": 0.9},
  "F05_photogenic":     {"visual_description": "...", "anchor_level": 1, "confidence": 0.6},
  "F06_wearability":    {"visual_description": "...", "anchor_level": null, "confidence": 0.0},
  "F07_pairing":        {"visual_description": "...", "anchor_level": null, "confidence": 0.0},
  "F08_fabric_perception": {"visual_description": "...", "anchor_level": 3, "confidence": 0.7},
  "F09_brand_tone":     {"visual_description": "...", "anchor_level": 2, "confidence": 0.5},
  "F10_uniqueness":     {"visual_description": "...", "anchor_level": 4, "confidence": 0.75}
}''')
    lines.append("")
    lines.append("confidence 说明你对自己的视觉判断有多确定（不是对销量判断有多确定）：")
    lines.append("  1.0 = 看得很清楚，100%确定")
    lines.append("  0.5 = 看不太清楚，大概猜的")
    lines.append("  0.0 = 完全看不清，放弃判断")
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

# ========== 视觉锚定辅助函数（模块级）==========
def _anchor_to_score(feat_def: dict[str, Any], anchor_level: int | None) -> float | None:
    """给定 anchor_level (1-5)，返回该锚定档 range 的中点作为 score。
    anchor_level=None 时返回 None（表示无法判断）。"""
    if anchor_level is None:
        return None
    anchors = feat_def.get("anchors", {})
    anchor = anchors.get(str(anchor_level)) or anchors.get(anchor_level)
    if anchor is None:
        return None
    rng = anchor.get("range", [5, 6])
    return round((rng[0] + rng[1]) / 2, 1)


def _resolve_anchor_label(feat_def: dict[str, Any], anchor_level: int | None) -> str:
    """给定 anchor_level 返回 label 文本（如 "宽松oversize"），供 _feat_summary 使用。"""
    if anchor_level is None:
        return "未判断"
    anchors = feat_def.get("anchors", {})
    anchor = anchors.get(str(anchor_level)) or anchors.get(anchor_level)
    if anchor is None:
        return f"档{anchor_level}"
    return anchor.get("label", f"档{anchor_level}")


def _resolve_feat_item(item: Any, feat_def: dict[str, Any]) -> dict[str, Any]:
    """统一解析单个模型对单个特征的输出，兼容新旧两种格式。

    v1.4.90+ 新格式: {"visual_description": "...", "anchor_level": 3, "confidence": 0.85}
    v1.4.90- 旧格式: {"score": 7.2, "confidence": 0.85, "reason": "..."}
    返回统一 dict: {visual_description, anchor_level, score, confidence, is_old_format}
    """
    if not isinstance(item, dict):
        return {}
    visual_desc = str(item.get("visual_description", "") or "")
    anchor_level = item.get("anchor_level")
    # anchor_level 可能被 VLM 输出成字符串 "3" 而非 int
    if anchor_level is not None:
        try:
            anchor_level = int(anchor_level)
            if not 1 <= anchor_level <= 5:
                anchor_level = None
        except (TypeError, ValueError):
            anchor_level = None
    confidence = safe_float(item.get("confidence"), 0.5)

    # 兼容旧格式：有 score 但没 anchor_level
    old_score = item.get("score")
    is_old_format = old_score is not None and anchor_level is None

    if is_old_format:
        # 旧格式：score 是 VLM 直接给的，没有 anchor 映射
        score = clamp(safe_float(old_score, 5.0), 1.0, 10.0)
        # 尝试从 reason 字段提取 visual_description（旧格式里 reason 含视觉描述）
        if not visual_desc:
            reason = str(item.get("reason", "") or "")
            if reason.startswith("[视觉观察]"):
                visual_desc = reason.split("；")[0].replace("[视觉观察]", "").strip()
            elif reason:
                visual_desc = reason[:80]
    else:
        # 新格式：score 从 anchor_level 派生
        score = _anchor_to_score(feat_def, anchor_level)
        if score is None and anchor_level is None:
            score = None
        elif score is None:
            score = 5.5

    return {
        "visual_description": visual_desc,
        "anchor_level": anchor_level,
        "score": score,
        "confidence": clamp(confidence, 0.0, 1.0),
        "is_old_format": is_old_format,
    }


def _make_mock_feat_entry(
    feat_def: dict[str, Any],
    seed_str: str,
    fixed_anchor_level: int | None = None,
) -> dict[str, Any]:
    """为单个特征生成 mock 的 anchor_level + visual_description。"""
    import random
    random.seed(zlib.crc32(seed_str.encode()))

    if fixed_anchor_level is not None:
        al = fixed_anchor_level
    else:
        r = random.random()
        if r < 0.15: al = 1
        elif r < 0.35: al = 2
        elif r < 0.65: al = 3
        elif r < 0.85: al = 4
        else: al = 5

    anchors = feat_def.get("anchors", {})
    anchor = anchors.get(str(al)) or anchors.get(al)
    anchor_label = anchor.get("label", f"档{al}") if anchor else f"档{al}"
    score = _anchor_to_score(feat_def, al) or 5.5

    # 从 anchor.description 里取前 30 字当 visual_description
    anchor_desc = anchor.get("description", "") if anchor else ""
    visual_desc = anchor_desc[:40] if anchor_desc else f"{anchor_label}"

    return {
        "visual_description": visual_desc,
        "anchor_level": al,
        "score": score,
        "confidence": round(random.uniform(0.7, 0.95), 2),
    }


# ========== FeatureExtractionEngine 类 ==========
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
        """mock mode：生成基于视觉锚定档的特征结果。

        - fixed_feature_scores 传10个浮点数时，覆盖 anchor_level 按比例映射：
          score→anchor_level: 1-2→1, 3-4→2, 5-6→3, 7-8→4, 9-10→5
        """
        import random
        random.seed(zlib.crc32(style_id.encode()))
        result = StyleFeatures(style_id=style_id)
        keys_ordered = list(self._bars_cfg["features"].keys())
        for i, key in enumerate(keys_ordered):
            fd = self._bars_cfg["features"][key]
            # 决定 anchor_level
            fixed_al: int | None = None
            if fixed_feature_scores is not None and i < len(fixed_feature_scores):
                fs = fixed_feature_scores[i]
                if fs <= 2: fixed_al = 1
                elif fs <= 4: fixed_al = 2
                elif fs <= 6: fixed_al = 3
                elif fs <= 8: fixed_al = 4
                else: fixed_al = 5

            entry = _make_mock_feat_entry(fd, f"{style_id}_{key}", fixed_anchor_level=fixed_al)
            result.features[key] = FeatureScore(
                key=key,
                name=fd.get("name", key),
                category=fd.get("category", "design"),
                score=entry["score"],
                confidence=entry["confidence"],
                visual_description=entry["visual_description"],
                anchor_level=entry["anchor_level"],
                reason="",
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
    """对一个款式用指定模型跑纯视觉锚定，返回 {feat_key: {visual_description, anchor_level, confidence}}"""
    user_msg = f"""
【款式基础信息】
FAB描述：{info.fab_description or '无FAB信息，仅从图片判断'}
品类：{info.category or '未标注'}
（价格/季节/品牌信息对你的视觉观察任务无关，忽略即可）

你现在只做一件事：观察图片中这个款式的视觉特征。

⚠️ 关键约束（请严格遵守）：
- 图片中可能是模特全身照，**只观察本款主产品**（FAB描述中提到的品类），不要描述模特身上的其他搭配（裤子、内搭、帽子、鞋子等）
- 例：FAB是"户外机能外套" → 只看外套，裤子/内搭全部忽略
- 例：FAB是"速干运动短裤" → 只看短裤，上衣/外套全部忽略

⚠️ 品类专属观察要点（不同品类看不同部位）：
【短裤/裤类】
- F01廓形：看裤腿剪裁（直筒/宽松/收脚），短裤没有裤腿堆积问题
- F02利落感：看腰头规整度、松紧带是否平整、裤脚剪裁是否干净
- F04功能可见：看口袋设计、松紧调节、反光条位置
【外套/上衣类】
- F01廓形：看肩线、衣长、下摆处理
- F02利落感：看领口是否规整、袖口剪裁、纽扣/拉链是否平顺
- F04功能可见：看帽檐调节、口袋系统、防水压胶条
【连衣裙/裙类】
- F01廓形：看腰线位置、裙摆形状
- F02利落感：看裙边是否整齐、接缝是否平整
- F04功能可见：看口袋位置、腰带扣、防晒处理

⚠️ F03颜色判断规则（最高优先级）：
- 一款可能有多个颜色SKU（藏青/芥黄/粉色/紫色等），但F03**只看主图/主推色**
- 多色SKU覆盖是正常铺货，不是视觉撞色设计

下面是视觉锚定量表（每档只描述视觉属性，不含销量判断）：

{_get_bars_prompt(features_cfg)}
""".strip()

    # v1.4.90: temperature 0.2→0.0 强制严格输出纯视觉JSON，减少模型"自作主张加价值判断"
    resp: LLMResponse = client.generate_multimodal(
        user_msg,
        image_paths=info.images,
        model=model,
        temperature=0.0,
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
    # 典型场景：glm-4.6v-flash 连续 5 次 1305 模型拥塞
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
        # ========== 额度致命错误拦截：让 pipeline 中止批次 ==========
        # 如果所有 VLM 失败里有额度/鉴权致命错误，raise 让上层 pipeline.run_batch
        # 捕获后 break 中止批次，不再白跑后续款式浪费额度。
        # 测试期望源码包含此字符串以便做 failfast 源码级断言。
        if any(is_fatal_quota_error(e) for e in errors):
            raise RuntimeError(
                f"所有模型特征提取均失败: {'; '.join(errors)}"
            )

        # ========== VLM 级 Fallback 第 3 层：brand 默认视觉锚定档兜底 ==========
        # 两个 VLM 都挂 → 用 anchor 档3的区间中点作为默认锚定（视觉中性档）
        log.warning(
            "[%s] 所有 VLM 均失败 (%s)，使用 brand 默认视觉锚定档兜底",
            info.style_id, "; ".join(errors[:2]),
        )
        if brand_cfg is None:
            brand_cfg = load_brand_profile("mipo")
        feat_defs = features_cfg["features"]
        default_result = {}
        for key, fd in feat_defs.items():
            default_result[key] = {
                "visual_description": f"VLM全部失败，使用brand默认视觉锚定：{fd.get('name', key)}取档3",
                "anchor_level": 3,  # 视觉中性档
                "confidence": 0.3,  # 低置信度，校准层会忽略
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
        per_model_anchors: list[int] = []
        per_model_descs: list[tuple[str, float]] = []  # (visual_desc, confidence)
        model_map: dict[str, float] = {}

        for m_idx, mr in enumerate(model_results):
            if key not in mr:
                continue
            item = mr[key]
            # 防御：Ollama 偶尔把单个特征包装成 list
            if isinstance(item, list):
                if len(item) > 0 and isinstance(item[0], dict):
                    item = item[0]
                else:
                    continue
            if not isinstance(item, dict):
                continue

            # 统一解析：兼容新旧两种 VLM 输出格式
            parsed = _resolve_feat_item(item, fd)
            if not parsed:
                continue

            sc = parsed["score"]
            if sc is None:
                continue  # VLM 明确说无法判断，跳过这个模型

            cf = parsed["confidence"]
            al = parsed["anchor_level"]
            vd = parsed["visual_description"]

            per_model_scores.append(sc)
            per_model_conf.append(cf)
            if al is not None:
                per_model_anchors.append(al)
            if vd and cf > 0:
                per_model_descs.append((vd, cf))

            m_name = models[m_idx] if m_idx < len(models) else f"model{m_idx}"
            model_map[m_name] = sc

        if not per_model_scores:
            # 所有模型都无法判断这个特征
            result.features[key] = FeatureScore(
                key=key, name=fd["name"], category=fd["category"],
                score=5.5, confidence=0.0,  # 中性+零置信度
                visual_description="无法判断（所有模型均放弃）",
                anchor_level=None,
                needs_review=True,
            )
            continue

        final_score = clamp(median_aggregate(per_model_scores), 1.0, 10.0)
        final_conf = float(sum(per_model_conf) / max(1, len(per_model_conf)))
        div = divergence(per_model_scores)
        needs_review = (div > div_thr) or (final_conf < conf_thr)

        # anchor_level 取中位数
        if per_model_anchors:
            sorted_anchors = sorted(per_model_anchors)
            final_anchor = sorted_anchors[len(sorted_anchors) // 2]
        else:
            final_anchor = None

        # visual_description 取置信度最高的那个
        if per_model_descs:
            final_desc = max(per_model_descs, key=lambda x: x[1])[0]
        else:
            final_desc = ""

        result.features[key] = FeatureScore(
            key=key,
            name=fd["name"],
            category=fd["category"],
            score=final_score,
            confidence=final_conf,
            visual_description=final_desc,
            anchor_level=final_anchor,
            reason="",  # v1.4.90+: 不再生成销量预判理由
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
