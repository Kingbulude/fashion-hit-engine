"""通用工具函数：数据类型、结构化解析、聚合统计
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal, Optional


# ========== 数据类 ==========
@dataclass
class FeatureScore:
    """单个特征的BARS评分结果"""
    key: str
    name: str
    category: str
    score: float
    confidence: float
    reason: str = ""
    # 多模型交叉验证
    model_scores: dict[str, float] = field(default_factory=dict)
    divergence: float = 0.0
    needs_review: bool = False


@dataclass
class StyleFeatures:
    """一个款的10个特征完整结果"""
    style_id: str
    features: dict[str, FeatureScore] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        """导出为平键值对，用于校准层回归"""
        row: dict[str, float] = {}
        for k, f in self.features.items():
            row[k] = f.score
            row[f"{k}_conf"] = f.confidence
            row[f"{k}_div"] = f.divergence
        return row

    @property
    def needs_human_review(self) -> bool:
        return any(f.needs_review for f in self.features.values())

    def lowest_features(self, top_n: int = 3) -> list[FeatureScore]:
        """扣分项（得分最低的Top N）"""
        return sorted(self.features.values(), key=lambda x: x.score)[:top_n]


@dataclass
class StyleInfo:
    """一个款式的基础信息（从Excel读取）"""
    style_id: str
    images: list[Path] = field(default_factory=list)
    fab_description: str = ""   # FAB（版型/面料/功能）
    category: str = ""          # 品类
    price: float = 0.0          # 成交价
    # 以下用于回测
    manual_grade: str = ""      # 人工S/A/P分级
    sales_qty: int = 0          # 销量
    sell_through_pct: float = 0.0  # 售罄率
    season: str = ""            # 季节（春夏/秋冬）
    is_main_push: bool = False  # 是否主推
    is_live_stream: bool = False  # 是否直播重点


@dataclass
class PersonaVote:
    """单个人设的投票结果"""
    persona_id: str
    persona_name: str
    decision_mode: str
    # 单独的分
    mom_score: float
    child_score: float
    # 综合分
    final_score: float
    # 文本理由
    mom_reason: str = ""
    child_reason: str = ""
    opposing_reason: str = ""
    vetoed: bool = False  # 妈妈否决
    # 模型
    model_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class VotingResult:
    """人设投票聚合结果"""
    style_id: str
    votes: list[PersonaVote] = field(default_factory=list)
    # 聚合
    weighted_score: float = 0.0          # 人设分布加权后的总分（0-10）
    opposition_rate: float = 0.0         # 反对率（<4分的人设比例）
    support_rate: float = 0.0            # 支持率（>7分的人设比例）
    top_buy_reasons: list[str] = field(default_factory=list)
    top_oppose_reasons: list[str] = field(default_factory=list)
    # 分歧度
    score_std: float = 0.0
    high_divergence_personas: list[str] = field(default_factory=list)


@dataclass
class ChannelScores:
    """双渠道评分 + 价格价值评分"""
    style_id: str
    natural_score: float = 0.0
    live_score: float = 0.0
    perceived_value: float = 0.0   # 0-10
    price_percentile: float = 0.0  # 0-1
    value_match: float = 0.0       # -1.0 ~ +1.0（正值=物美价廉，负值=价超所值）
    price_risk: str = "中风险"     # 低风险/中风险/高风险


@dataclass
class GradeResult:
    """S/A+/A/P分级"""
    style_id: str
    grade: str                      # S / A+ / A / P / 风险
    final_score: float = 0.0        # 校准后最终分（0-100）
    confidence: float = 0.0         # 0-1
    # 理由
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)  # 改款建议
    # 渠道建议
    recommended_channel: str = ""   # 自然流量 / 直播带货 / 双渠道 / 设计调性
    # 消费者洞察（从人设投票提取）
    consumer_insights: str = ""


@dataclass
class FullPrediction:
    """一个款式的完整预测结果"""
    info: StyleInfo
    features: StyleFeatures
    voting: VotingResult
    channels: ChannelScores
    grade: GradeResult

    def to_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"style_id": self.info.style_id}
        row.update(self.features.as_row())
        row.update({
            "weighted_score": self.voting.weighted_score,
            "opposition_rate": self.voting.opposition_rate,
            "support_rate": self.voting.support_rate,
            "vote_std": self.voting.score_std,
            "natural_score": self.channels.natural_score,
            "live_score": self.channels.live_score,
            "perceived_value": self.channels.perceived_value,
            "price_percentile": self.channels.price_percentile,
            "value_match": self.channels.value_match,
            "price_risk": self.channels.price_risk,
            "grade": self.grade.grade,
            "final_score": self.grade.final_score,
            "confidence": self.grade.confidence,
            "recommended_channel": self.grade.recommended_channel,
        })
        return row


# ========== 品牌配置/身份三轴线 数据类 ==========
@dataclass
class PersonaIdentityAxes:
    scene: str
    aesthetic: str
    price: str


@dataclass
class DecisionLayer:
    id: str
    name: str
    persona_axis_key: str
    role: Literal["decider", "veto"]
    default_weight: float


@dataclass
class BrandDecisionStructure:
    type: Literal["single_layer", "multi_layer"]
    layers: list[DecisionLayer]
    age_weight_rules: list[dict[str, Any]]
    default_target_age: int


@dataclass
class BrandConfig:
    brand_id: str
    brand_name: str
    decision_structure: BrandDecisionStructure
    features_bars: dict[str, Any]
    personas: list[dict[str, Any]]
    scoring_weights: dict[str, Any]
    category_registry: dict[str, Any]
    default_engine_weights: dict[str, float]
    default_channel_split: dict[str, float]
    grading_thresholds: dict[str, float]
    calibrated_dir: str
    child_identity_axes: Optional[list[dict[str, Any]]] = None
    personas_weights: Optional[dict[str, float]] = None
    features_biases: Optional[dict[str, float]] = None
    engine_weights: Optional[dict[str, float]] = None


# ========== 结构化解析 ==========
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_STRIP_RE = re.compile(r"[\x00-\x1f\x7f]")


def extract_json(text: str) -> dict | list:
    """从LLM输出的文本中提取JSON对象/数组。支持 ```json ... ``` 包裹或裸JSON。"""
    text = _JSON_STRIP_RE.sub("", text).strip()
    # 先试代码块
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)
    # 找最外层 { ... } 或 [ ... ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        snippet = text[start:i + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            break
    # 最后一招：直接 parse 全文
    return json.loads(text)


def safe_float(v: Any, default: float = 5.0) -> float:
    """安全转float"""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def clamp(v: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))


def median_aggregate(values: list[float]) -> float:
    """多模型交叉验证用中位数聚合"""
    if not values:
        return 5.0
    if len(values) == 1:
        return values[0]
    return float(median(values))


def divergence(values: list[float]) -> float:
    """多模型分歧度 = max - min"""
    if not values:
        return 0.0
    return max(values) - min(values)
