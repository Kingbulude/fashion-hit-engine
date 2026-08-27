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
    volc_api_key: str | None = None
    volc_endpoint: str | None = None

    feature_extraction_models: list[str] = field(default_factory=lambda: ["qwen3-vl-plus", "qwen3.5-omni"])
    persona_models: list[str] = field(default_factory=lambda: ["qwen3-max", "deepseek-v4-pro"])

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
    """构造童装户外默认双层决策结构（兼容旧config目录模式）"""
    layers = [
        DecisionLayer(
            id="mom_decision_layer",
            name="妈妈决策者层",
            persona_axis_key="identity_axes",
            role="decider",
            default_weight=0.70,
        ),
        DecisionLayer(
            id="child_influence_layer",
            name="孩子影响层",
            persona_axis_key="child_identity_axes",
            role="veto",
            default_weight=0.30,
        ),
    ]
    age_weight_rules = [
        {"age_range": [6, 8], "mom_weight": 0.70, "child_weight": 0.30, "label": "妈妈主导"},
        {"age_range": [9, 11], "mom_weight": 0.50, "child_weight": 0.50, "label": "共同决策"},
        {"age_range": [12, 14], "mom_weight": 0.30, "child_weight": 0.70, "label": "孩子主导"},
    ]
    return BrandDecisionStructure(
        type="multi_layer",
        layers=layers,
        age_weight_rules=age_weight_rules,
        default_target_age=10,
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

    若 calibrated_dir 下存在 features_biases.yaml / personas_weights.yaml / engine_weights.yaml
    则自动覆盖 BrandConfig 对应字段。

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
        decision_structure = _parse_decision_structure(
            profile_yaml.get("decision_structure", {"type": "single_layer", "layers": []})
        )
        default_engine_weights = profile_yaml.get("default_engine_weights", {
            "persona_voting": 0.35,
            "channel_scoring": 0.30,
            "price_value": 0.35,
        })
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

        personas_list = personas_raw.get("personas", [])
        child_identity_axes = personas_raw.get("child_identity_axes", None)
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
        child_identity_axes = None

    # 若 calibrated_dir 下存在校准文件，则自动覆盖
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    features_biases = None
    personas_weights = None
    engine_weights = None

    fb_path = calibrated_dir / "features_biases.yaml"
    if fb_path.exists():
        features_biases = _load_yaml(fb_path)

    pw_path = calibrated_dir / "personas_weights.yaml"
    if pw_path.exists():
        personas_weights = _load_yaml(pw_path)

    ew_path = calibrated_dir / "engine_weights.yaml"
    if ew_path.exists():
        engine_weights = _load_yaml(ew_path)

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
        calibrated_dir=str(calibrated_dir),
        child_identity_axes=child_identity_axes,
        personas_weights=personas_weights,
        features_biases=features_biases,
        engine_weights=engine_weights,
    )


def load_config(
    env_path: Path | None = None,
    config_dir: Path = CONFIG_DIR,
) -> AppConfig:
    """加载默认品牌配置（.env + tongzhuang-outdoor 品牌适配包）。

    通用入口：等价于 load_brand_profile("tongzhuang-outdoor") + 注入 .env 的 API 配置。
    需要切换品牌时请直接用 load_brand_profile(brand_id)。
    config_dir 参数保留用于向后兼容（实际配置从 brand_profiles/ 加载）。
    """
    # .env 优先用户指定，其次项目根目录
    env_file = env_path or (PROJECT_ROOT / ".env")
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    api = APIConfig(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        volc_api_key=os.getenv("VOLC_API_KEY"),
        volc_endpoint=os.getenv("VOLC_ENDPOINT"),
        feature_extraction_models=[
            m.strip() for m in os.getenv("FEATURE_EXTRACTION_MODELS", "qwen3-vl-plus,qwen3.5-omni").split(",") if m.strip()
        ],
        persona_models=[
            m.strip() for m in os.getenv("PERSONA_MODELS", "qwen3-max,deepseek-v4-pro").split(",") if m.strip()
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
    brand_cfg = load_brand_profile("tongzhuang-outdoor")
    features = brand_cfg.features_bars
    # 兼容旧格式：把personas列表包装成 {personas: [...], decision_mode_weights: {...}} 结构
    personas = {
        "personas": brand_cfg.personas,
        "decision_mode_weights": {
            "mom_dominant": {"mom": 0.70, "child": 0.30},
            "joint_decision": {"mom": 0.50, "child": 0.50},
            "child_dominant": {"mom": 0.30, "child": 0.70},
        },
        "mom_veto_threshold": 3.0,
    }
    if brand_cfg.child_identity_axes is not None:
        personas["child_identity_axes"] = brand_cfg.child_identity_axes
    scoring = brand_cfg.scoring_weights

    return AppConfig(api=api, paths=paths, features=features, personas=personas, scoring=scoring)
