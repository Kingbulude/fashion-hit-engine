"""配置加载：统一读取 .env + YAML配置文件
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .types import BrandConfig, BrandDecisionStructure, DecisionLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
BRAND_PROFILES_DIR = PROJECT_ROOT / "brand_profiles"


# ========== LLM/API 配置 ==========
@dataclass
class APIConfig:
    dashscope_api_key: str = ""
    zhipu_api_key: str = ""
    volc_api_key: str | None = None
    volc_endpoint: str | None = None

    feature_extraction_models: list[str] = field(default_factory=lambda: ["qwen-vl-plus"])
    persona_models: list[str] = field(default_factory=lambda: ["qwen-max", "deepseek-v3"])

    qpm_limit: int = 45
    max_concurrent_personas: int = 8
    max_retries: int = 3


# ========== 数据路径 ==========
@dataclass
class PathConfig:
    styles_xlsx: Path = PROJECT_ROOT / "data" / "styles.xlsx"
    sales_xlsx: Path = PROJECT_ROOT / "data" / "sales.xlsx"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    output_dir: Path = PROJECT_ROOT / "output"
    reports_dir: Path = PROJECT_ROOT / "output" / "reports"


@dataclass
class AppConfig:
    api: APIConfig
    paths: PathConfig
    features: dict[str, Any]         # config/features_bars.yaml 原始内容
    personas: dict[str, Any]         # config/personas.yaml 原始内容
    scoring: dict[str, Any]          # config/scoring_weights.yaml 原始内容


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ========== 品牌配置加载（新架构 v2.0）==========

def list_available_brands() -> list[str]:
    """扫 brand_profiles/ 目录，返回非下划线开头的子目录名"""
    if not BRAND_PROFILES_DIR.exists():
        return []
    result: list[str] = []
    for child in BRAND_PROFILES_DIR.iterdir():
        if child.is_dir() and not child.name.startswith("_"):
            result.append(child.name)
    return sorted(result)


def _build_default_decision_structure() -> BrandDecisionStructure:
    """兜底双层决策结构（等价于 mipo 品牌包，用于兼容模式）。

    层语义通用：决策者层 + 影响层（否决）。具体角色名称（如 mipo 的
    妈妈/孩子）由品牌 YAML 声明，代码不预设。
    age_weight_rules 的 mom_weight/child_weight 为 YAML 字段名，
    语义 = 决策者层权重 / 影响层权重。
    """
    layers = [
        DecisionLayer(
            id="decider_layer",
            name="决策者层",
            persona_axis_key="identity_axes",
            role="decider",
            default_weight=0.70,
        ),
        DecisionLayer(
            id="influencer_layer",
            name="影响层",
            persona_axis_key="influencer_axes",
            role="veto",
            default_weight=0.30,
        ),
    ]
    age_weight_rules = [
        {"age_range": [6, 8], "mom_weight": 0.70, "child_weight": 0.30, "label": "决策者主导"},
        {"age_range": [9, 11], "mom_weight": 0.50, "child_weight": 0.50, "label": "共同决策"},
        {"age_range": [12, 14], "mom_weight": 0.30, "child_weight": 0.70, "label": "影响层主导"},
        {"age_range": [15, 18], "mom_weight": 0.25, "child_weight": 0.75, "label": "青少年自主"},
    ]
    return BrandDecisionStructure(
        type="multi_layer",
        layers=layers,
        age_weight_rules=age_weight_rules,
        default_target_age=12,
    )


def _parse_decision_structure(raw: dict[str, Any]) -> BrandDecisionStructure:
    """从profile.yaml解析BrandDecisionStructure"""
    layers_raw = raw.get("layers", [])
    layers = [
        DecisionLayer(
            id=str(l.get("id", "")),
            name=str(l.get("name", "")),
            persona_axis_key=str(l.get("persona_axis_key", "identity_axes")),
            role=l.get("role", "decider"),
            default_weight=float(l.get("default_weight", 0.5)),
        )
        for l in layers_raw
    ]
    return BrandDecisionStructure(
        type=raw.get("type", "single_layer"),
        layers=layers,
        age_weight_rules=list(raw.get("age_weight_rules", [])),
        default_target_age=int(raw.get("default_target_age", 10)),
    )


def load_brand_profile(brand_id: str) -> BrandConfig:
    """按品牌ID读取 brand_profiles/<id>/ 下的5yaml并组装BrandConfig

    若 calibrated_dir 下存在 loop1_vlm_feature_biases.yaml / loop2_persona_distribution_weights.yaml /
    loop3_ensemble_weights.yaml，则自动覆盖 BrandConfig 对应字段（3Loop 校准产物）。
    同时兼容旧文件名 features_biases.yaml / personas_weights.yaml / engine_weights.yaml（v1.4.44 前）。

    若 brand_profiles/<id>/ 不存在，则回退到旧 config/ 目录的3yaml构造兼容结构。
    """
    brand_dir = BRAND_PROFILES_DIR / brand_id
    calibrated_dir = brand_dir / "calibrated"

    if brand_dir.exists():
        profile_yaml = _load_yaml(brand_dir / "profile.yaml")
        features_bars = _load_yaml(brand_dir / "features_bars.yaml")
        personas_raw = _load_yaml(brand_dir / "personas.yaml")
        scoring_weights = _load_yaml(brand_dir / "scoring_weights.yaml")
        category_registry = _load_yaml(brand_dir / "category_registry.yaml")

        brand_name = profile_yaml.get("brand_name", brand_id)
        # v1.4.94+: 品牌品类基本信息（注入专家 prompt 用）
        industry_segment = profile_yaml.get("industry_segment") or None
        target_age_range = profile_yaml.get("target_age_range") or None
        target_size_range = profile_yaml.get("target_size_range") or None
        brand_domain_knowledge = profile_yaml.get("brand_domain_knowledge") or None
        decision_structure = _parse_decision_structure(
            profile_yaml.get("decision_structure", {"type": "single_layer", "layers": []})
        )
        default_engine_weights = profile_yaml.get("default_engine_weights", {
            "persona_voting": 0.35,
            "channel_scoring": 0.30,
            "price_value": 0.35,
        })
        # 品牌特定的销售标签→数值映射（如 MIPO 的 爆/旺/平/滞 → 4/3/2/1）。
        # 未声明时为 None，data_io 按通用数值读取，不影响其他品牌。
        sales_label_mapping = profile_yaml.get("sales_label_mapping") or None
        if sales_label_mapping:
            sales_label_mapping = {str(k).strip(): float(v) for k, v in sales_label_mapping.items()}
        # 是否按年份分组校准（MIPO 有 2025-2026 跨年度数据时开启）
        calibration_group_by_year = bool(profile_yaml.get("calibration_group_by_year", False))
        # v1.4.51+: 品牌是否有内审分级流程（S/A+/A/P）。默认 True（有内审）
        has_internal_review = bool(profile_yaml.get("has_internal_review", True))
        default_channel_split = profile_yaml.get("default_channel_split", {
            "natural": 0.50,
            "live_stream": 0.50,
        })
        grading_thresholds = profile_yaml.get("grading_thresholds", {
            "s": 7.6,
            "a_plus": 6.6,
            "a": 5.2,
            "p": 0.0,
        })
        insight_thresholds = profile_yaml.get("insight_thresholds") or None

        personas_list = personas_raw.get("personas", [])
        # ===== 🔧 合并孩子人设（之前漏掉了！）=====
        # personas.yaml 里 child_identity_axes 定义了孩子人设（7男7女，6-18岁全覆盖）
        # 之前只取了 "personas"（30 妈妈），导致投票完全缺孩子视角 + 性别平衡
        child_raw = personas_raw.get("child_identity_axes", [])
        for cp in child_raw:
            personas_list.append({
                "persona_id": cp.get("persona_id", f"C_{cp.get('gender', '?')}"),
                "name": f"{cp.get('age', '?')}岁{'男' if cp.get('gender') == 'male' else '女'}孩",
                "gender": cp.get("gender"),
                "age": cp.get("age"),
                "layer": "child_influencer",
                "weight": 0.05,
                "veto_when": cp.get("veto_when", []),
                "color_preference": [],
                "axes": {},
            })
        # 重算权重
        total_w = sum(p.get("weight", 1/len(personas_list)) for p in personas_list)
        if total_w > 0:
            for p in personas_list:
                p["weight"] = p.get("weight", 0.05) / total_w
        # 轴数据按决策层声明收集：layer.persona_axis_key → personas.yaml 顶层同名列表
        persona_axes = {
            layer.persona_axis_key: personas_raw.get(layer.persona_axis_key)
            for layer in decision_structure.layers
            if personas_raw.get(layer.persona_axis_key) is not None
        }
    else:
        warnings.warn(
            f"[load_brand_profile] brand_profiles/{brand_id}/ 不存在，回退到旧 config/ 目录兼容模式",
            DeprecationWarning,
            stacklevel=2,
        )
        features_bars = _load_yaml(CONFIG_DIR / "features_bars.yaml")
        personas_raw = _load_yaml(CONFIG_DIR / "personas.yaml")
        scoring_weights = _load_yaml(CONFIG_DIR / "scoring_weights.yaml")
        category_registry = {"categories": [], "category_aliases": {}}

        brand_name = f"{brand_id} (兼容模式)"
        decision_structure = _build_default_decision_structure()
        default_engine_weights = {
            "persona_voting": 0.35,
            "channel_scoring": 0.30,
            "price_value": 0.35,
        }
        default_channel_split = {"natural": 0.50, "live_stream": 0.50}
        grading_thresholds = {"s": 7.6, "a_plus": 6.6, "a": 5.2, "p": 0.0}

        personas_list = personas_raw.get("personas", [])
        persona_axes = {}
        sales_label_mapping = None  # 兼容旧 config 目录时没有品牌适配包 YAML
        calibration_group_by_year = False
        has_internal_review = True  # fallback 兼容旧 config 目录模式

    # 若 calibrated_dir 下存在校准文件，则自动覆盖
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    features_biases = None
    personas_weights = None
    engine_weights = None

    # ===== v1.4.52+: 优先读 3Loop 新文件名，fallback 旧文件名 =====
    # Loop1 → feature_biases (F01~F10 特征偏置)
    fb_path = calibrated_dir / "loop1_vlm_feature_biases.yaml"
    if not fb_path.exists():
        fb_path = calibrated_dir / "features_biases.yaml"   # fallback v1.4.44 前
    if fb_path.exists():
        fb_raw = _load_yaml(fb_path)
        if isinstance(fb_raw, dict):
            features_biases = fb_raw.get("feature_biases") or fb_raw

    # Loop2 → persona_weights (P01~P30 人设分布)
    pw_path = calibrated_dir / "loop2_persona_distribution_weights.yaml"
    if not pw_path.exists():
        pw_path = calibrated_dir / "personas_weights.yaml"   # fallback v1.4.44 前
    if pw_path.exists():
        pw_raw = _load_yaml(pw_path)
        if isinstance(pw_raw, dict):
            personas_weights = pw_raw.get("persona_weights") or pw_raw

    # Loop3 → engine_weights (四引擎权重)
    ew_path = calibrated_dir / "loop3_ensemble_weights.yaml"
    if not ew_path.exists():
        ew_path = calibrated_dir / "engine_weights.yaml"   # fallback v1.4.44 前
    if ew_path.exists():
        ew_raw = _load_yaml(ew_path)
        if isinstance(ew_raw, dict):
            eng = ew_raw.get("engine") or {}
            w = eng.get("weights") or ew_raw.get("weights")
            if w:
                # Loop3 YAML: {persona_score, channel_score, price_value_score}
                # BrandConfig.engine_weights: {persona_voting, channel_scoring, price_value}
                engine_weights = {
                    "persona_voting": float(w.get("persona_score", 0.35)),
                    "channel_scoring": float(w.get("channel_score", 0.30)),
                    "price_value": float(w.get("price_value_score", 0.35)),
                }
            else:
                engine_weights = ew_raw

    return BrandConfig(
        brand_id=brand_id,
        brand_name=brand_name,
        decision_structure=decision_structure,
        features_bars=features_bars,
        personas=personas_list,
        scoring_weights=scoring_weights,
        category_registry=category_registry,
        default_engine_weights=default_engine_weights,
        default_channel_split=default_channel_split,
        grading_thresholds=grading_thresholds,
        insight_thresholds=insight_thresholds,
        calibrated_dir=str(calibrated_dir),
        persona_axes=persona_axes or None,
        personas_weights=personas_weights,
        features_biases=features_biases,
        engine_weights=engine_weights,
        sales_label_mapping=sales_label_mapping,
        calibration_group_by_year=calibration_group_by_year,
        has_internal_review=has_internal_review,
        industry_segment=industry_segment,
        target_age_range=target_age_range,
        target_size_range=target_size_range,
        brand_domain_knowledge=brand_domain_knowledge,
    )


def load_config(
    env_path: Path | None = None,
    config_dir: Path = CONFIG_DIR,
    override_api_key: str | None = None,
) -> AppConfig:
    """加载默认品牌配置（.env + mipo 品牌适配包）。

    通用入口：等价于 load_brand_profile("mipo") + 注入 .env 的 API 配置。
    需要切换品牌时请直接用 load_brand_profile(brand_id)。
    config_dir 参数保留用于向后兼容（实际配置从 brand_profiles/ 加载）。

    Args:
        override_api_key: 若提供，则优先于 .env 中的 DASHSCOPE_API_KEY。
                          用于 Streamlit 用户在 UI 粘贴的 API Key。
    """
    # .env 优先用户指定，其次项目根目录
    env_file = env_path or (PROJECT_ROOT / ".env")
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    api = APIConfig(
        dashscope_api_key=(override_api_key if override_api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")),
        zhipu_api_key=os.getenv("ZHIPU_API_KEY", ""),
        volc_api_key=os.getenv("VOLC_API_KEY"),
        volc_endpoint=os.getenv("VOLC_ENDPOINT"),
        feature_extraction_models=[
            m.strip() for m in os.getenv("FEATURE_EXTRACTION_MODELS", "qwen-vl-plus").split(",") if m.strip()
        ],
        persona_models=[
            m.strip() for m in os.getenv("PERSONA_MODELS", "qwen-max,deepseek-v3").split(",") if m.strip()
        ],
        qpm_limit=int(os.getenv("QPM_LIMIT", "45")),
        max_concurrent_personas=int(os.getenv("MAX_CONCURRENT_PERSONAS", "8")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
    )

    paths = PathConfig(
        styles_xlsx=Path(os.getenv("STYLES_XLSX", str(PROJECT_ROOT / "data" / "styles.xlsx"))),
        sales_xlsx=Path(os.getenv("SALES_XLSX", str(PROJECT_ROOT / "data" / "sales.xlsx"))),
        images_dir=Path(os.getenv("IMAGES_DIR", str(PROJECT_ROOT / "data" / "images"))),
        output_dir=Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output"))),
        reports_dir=Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output"))) / "reports",
    )

    for p in [paths.images_dir, paths.output_dir, paths.reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 调用新架构的品牌加载，然后转成兼容的features/personas/scoring字段
    brand_cfg = load_brand_profile("mipo")
    features = brand_cfg.features_bars
    # 兼容旧格式：personas 列表 + 决策层轴数据（按 layer.persona_axis_key 收集）
    personas: dict[str, Any] = {"personas": brand_cfg.personas}
    if brand_cfg.persona_axes:
        personas.update(brand_cfg.persona_axes)
    scoring = brand_cfg.scoring_weights

    return AppConfig(api=api, paths=paths, features=features, personas=personas, scoring=scoring)
