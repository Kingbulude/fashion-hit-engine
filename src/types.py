"""通用工具函数：数据类型、结构化解析、聚合统计
"""
from __future__ import annotations

import json
import logging
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
    is_blind: bool = False  # 盲测组标记（见 src/blind_set.py）


@dataclass
class PersonaVote:
    """单个人设的投票结果

    决策层通用结构：分数/理由按 decision_structure.layers[].id 存储。
    层的语义（如 mipo 的妈妈决策者层/孩子影响层）由品牌 YAML 定义，
    代码不预设任何具体角色。
    """
    persona_id: str
    persona_name: str
    # 各决策层得分（key=layer_id，value=1-10）
    layer_scores: dict[str, float] = field(default_factory=dict)
    # 各决策层理由（key=layer_id）
    layer_reasons: dict[str, str] = field(default_factory=dict)
    # 综合分（按层权重加权）
    final_score: float = 0.0
    # 文本理由
    opposing_reason: str = ""
    vetoed: bool = False  # 任一否决层（role=veto）触发
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
    # 运行元数据：记录这个结果是怎么来的（真实 VLM / mock / 用了哪个模型）
    metadata: dict[str, Any] = field(default_factory=dict)

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
    # spec §7.x: 历史命名 double_layer；代码实现用 multi_layer。
    # 两者等价（multi_layer 是 double_layer 的可扩展命名，允许未来 3+ 层）。
    # 加载层与 persona_voting 均把 double_layer 当作 multi_layer 别名处理。
    type: Literal["single_layer", "multi_layer", "double_layer"]
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
    # 决策层轴数据（key=layer.persona_axis_key，value=personas.yaml 顶层对应列表）。
    # 层语义由 YAML 声明（如 mipo 的 child_identity_axes=孩子影响层轴），
    # 代码只按 layer 声明的 key 取数，不预设任何角色。
    persona_axes: Optional[dict[str, list[dict[str, Any]]]] = None
    personas_weights: Optional[dict[str, float]] = None
    features_biases: Optional[dict[str, float]] = None
    engine_weights: Optional[dict[str, float]] = None


# ========== 结构化解析 ==========
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_STRIP_RE = re.compile(r"[\x00-\x1f\x7f]")


def extract_json(text: str, *, as_dict: bool = False) -> dict | list:
    """从LLM输出的文本中提取JSON对象/数组。支持 ```json ... ``` 包裹或裸JSON。

    Args:
        text: LLM 原始输出文本
        as_dict: 如果为 True，强制返回 dict（list 会尝试提取/构造兜底 dict）。
                 这是 Ollama 本地模式的核心补丁：小模型经常偷懒返回
                 纯数值数组 [7.5] 或 list 包装，下游调 .get() 就崩。
                 默认 False 保持兼容。
    """
    text = _JSON_STRIP_RE.sub("", text).strip()
    raw: Any = None
    # 先试代码块
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            raw = json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)
            raw = None
    if raw is None:
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
                                raw = json.loads(snippet)
                                break
                            except json.JSONDecodeError:
                                break
                if raw is not None:
                    break
    if raw is None:
        # 最后一招：直接 parse 全文
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            # 彻底无 JSON → as_dict 模式兜底返回空 dict
            if as_dict:
                log_warn = logging.getLogger(__name__)
                log_warn.warning("extract_json 未找到任何有效 JSON，返回空 dict 兜底（原文=%s）", text[:100])
                raw = {}
            else:
                raise

    if as_dict:
        raw = _coerce_to_dict(raw, text)
    return raw


def _coerce_to_dict(raw: Any, original_text: str = "") -> dict:
    """把 extract_json 的返回值强制转成 dict（从各种 LLM 偷懒格式里抢救）。

    处理的场景：
      dict                          → 原样返回
      [dict, dict, ...]             → 取第一个 dict
      [float] / [str] / [int]       → 构造 {"final_score": 7.5} 兜底
      [] 空数组                      → 返回 {}
      其他非 dict 类型              → 返回 {}
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        if len(raw) == 0:
            log_warn = logging.getLogger(__name__)
            log_warn.warning("extract_json 收到空数组，返回空 dict 兜底")
            return {}
        first = raw[0]
        if isinstance(first, dict):
            return first  # list 包装了 dict → 取第一个
        # 纯数值/字符串数组 → 构造兜底 dict
        fallback_val = first
        return {"final_score": fallback_val, "_raw_list": raw, "_original_preview": original_text[:100]}
    # 非 dict 非 list → 兜底空 dict
    log_warn = logging.getLogger(__name__)
    log_warn.warning("extract_json 收到 %s，返回空 dict 兜底（原文=%s）", type(raw).__name__, original_text[:100])
    return {}


def safe_float(v: Any, default: float = 5.0) -> float:
    """安全转float"""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


# ========== Excel 列名别名常量（供多模块统一识别）==========
# 语义约定：「是否主推/是否直播重点」记录的是【实际投放】（ADR-0001），
# 即该款真实被主推/进过直播间，而非研发阶段的计划。
SALES_QTY_COL_ALIASES = [
    "真实销量结果", "真实销售结果", "真实销量", "累计销量",
    "销量", "真实销量数", "销售数量", "销售结果",
]
SELL_THROUGH_COL_ALIASES = ["售罄率", "售罄比例"]
MANUAL_GRADE_COL_ALIASES = ["内审分级", "人工分级", "评级", "人工评级", "S/A/P"]
PRICE_COL_ALIASES = ["售价", "成交价格", "成交价", "吊牌价"]
STYLE_ID_COL_ALIASES = ["款式编号", "款号", "style_id", "商品编号"]
CATEGORY_COL_ALIASES = ["品类", "品类名称", "类别", "商品品类", "商品类别"]
SEASON_COL_ALIASES = ["季节", "上架季节", "季节属性", "适用季节"]
FAB_COL_ALIASES = ["FAB描述", "FAB", "fab", "款式描述", "版型/设计描述"]
MAIN_PUSH_COL_ALIASES = ["是否主推", "主推", "是否主推款", "实际主推"]
LIVE_STREAM_COL_ALIASES = ["是否直播重点", "直播重点", "直播款", "实际直播"]


def _truthy_excel_value(v: Any) -> bool:
    """Excel 单元格值 → 布尔（与 data_io/_bool 同口径）"""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "是", "y", "t", "主推", "重点", "已推", "已直播"}


def find_aliased_column(df_cols: list[str], aliases: list[str]) -> str | None:
    """在 DataFrame 列名列表里找匹配的别名（精确匹配优先，其次包含匹配）。"""
    # 精确匹配
    for alias in aliases:
        if alias in df_cols:
            return alias
    # 包含匹配（兼容细微差异）
    for alias in aliases:
        for col in df_cols:
            if alias in str(col).strip():
                return col
    return None


def parse_sales_value(v: Any, default: float = 0.0) -> float:
    """解析 Excel 里的销量值，支持 int/float/"9000+" / "9,000" / "约8000" 等。

    策略：
    - 已是数值 → 直接返回
    - 文本 → 提取所有数字字符，拼成整数
    - NaN/空 → 返回 default
    """
    if v is None:
        return default
    try:
        if isinstance(v, (int, float)):
            if v != v:  # NaN check
                return default
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in ("nan", "none", "-"):
                return default
            # 提取数字部分 (去掉 "约/+/+/~" 等前后缀和千分位逗号)
            digits = "".join(ch for ch in s if ch.isdigit() or ch == ".")
            if digits in ("", "."):
                return default
            return float(digits)
    except Exception:
        pass
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
