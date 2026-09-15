"""人设投票引擎：决策层通用投票 + 多模型混合

每个品牌由 profile.yaml 声明购买决策结构（decision_structure.layers），
人设投票按层进行：
  - 每个决策层独立评分（输出 key = "<layer_id>_score" / "<layer_id>_reason"）
  - 综合分 = Σ layer_score × layer_weight（权重按年龄档 age_weight_rules 取，
    mom_weight/child_weight 为 YAML 字段名，语义=决策者层/影响层权重）
  - 否决：①决策者层低分否决（< VETO_THRESHOLD）②影响层（role=veto）
    轴数据 veto_when 关键词命中（聚合阶段惩罚）
层语义（如 mipo 的妈妈决策者层/孩子影响层）由品牌 YAML 声明，代码不预设角色。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from statistics import mean, pstdev
from typing import Any

from tqdm import tqdm

from .config import AppConfig
from .llm_client import BailianClient, resolve_and_dedupe_models
from .types import (
    BrandConfig,
    DecisionLayer,
    PersonaVote,
    StyleFeatures,
    StyleInfo,
    VotingResult,
    clamp,
    extract_json,
    safe_float,
)

log = logging.getLogger(__name__)

# 引擎级参数（非品牌语义）：决策者层低分否决线 / 影响层否决惩罚系数
VETO_SCORE_THRESHOLD = 3.0
VETO_PENALTY_FACTOR = 0.70

# cfg 缺省时的投票模型（brand_cfg 路径不依赖 AppConfig）
_DEFAULT_PERSONA_MODELS = ["qwen-max", "deepseek-v3"]


def _persona_key(p: dict[str, Any]) -> str:
    return str(p.get("id", p.get("persona_id", "")))


# ========== 特征值转自然语言（喂给人设LLM）==========
def _feat_summary(feats: StyleFeatures) -> str:
    """把10个BARS分数翻译成通俗描述，避免LLM被纯数字搞乱"""
    def _level(sc: float) -> str:
        if sc >= 8: return "非常高"
        if sc >= 6.5: return "较高"
        if sc >= 5.0: return "中等"
        if sc >= 3.5: return "较低"
        return "非常低"

    lines = []
    for key, f in feats.features.items():
        lines.append(f"· {f.name}（{key}）：{f.score:.1f}/10（{_level(f.score)}）— {f.reason or '无细节'}")
    return "\n".join(lines)


# ========== 决策层权重（按年龄档）==========
def _match_age_weight_rules(
    age_weight_rules: list[dict[str, Any]],
    target_age: int,
) -> dict[str, Any]:
    """根据target_age匹配age_weight_rules取档"""
    for rule in age_weight_rules:
        rng = rule.get("age_range", [0, 999])
        if len(rng) == 2 and rng[0] <= target_age <= rng[1]:
            return rule
    return age_weight_rules[0] if age_weight_rules else {}


def _layer_weights_for_age(
    ds: Any,
    target_age: int | None,
) -> dict[str, float]:
    """按年龄档解析各决策层权重（key=layer_id）。

    YAML 字段映射（向后兼容 mipo 等既有适配包）：
      - mom_weight   → role=decider 的层
      - child_weight → 其余层（影响层）
      - layer_weights: {layer_id: w} → 任意层数直配（新通用写法，优先级最高）
    """
    actual_age = target_age if target_age is not None else ds.default_target_age
    rule = _match_age_weight_rules(ds.age_weight_rules, actual_age)
    weights: dict[str, float] = {}
    decider_w = rule.get("mom_weight", rule.get("decider_weight"))
    influencer_w = rule.get("child_weight", rule.get("influencer_weight"))
    for layer in ds.layers:
        if layer.role == "decider" and decider_w is not None:
            weights[layer.id] = float(decider_w)
        elif layer.role != "decider" and influencer_w is not None and layer.id not in weights:
            weights[layer.id] = float(influencer_w)
        else:
            weights[layer.id] = float(layer.default_weight)
    lw = rule.get("layer_weights")
    if isinstance(lw, dict):
        weights.update({str(k): float(v) for k, v in lw.items()})
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _resolve_layers(
    brand_cfg: BrandConfig | None,
) -> list[DecisionLayer]:
    """取决策层列表。无品牌结构时退化为单决策层。"""
    if brand_cfg is not None:
        return list(brand_cfg.decision_structure.layers)
    return [DecisionLayer(id="decider_layer", name="决策者层",
                          persona_axis_key="identity_axes", role="decider",
                          default_weight=1.0)]


# ========== 人设Prompt（按决策层动态渲染）==========
def _render_persona_vote_system(layers: list[DecisionLayer], brand_name: str) -> str:
    layer_names = "、".join(f"{l.name}（{l.id}）" for l in layers)
    return f"""你是{brand_name}的购买决策模拟器。你将扮演一个具体的人设，联合相关决策层，对一件服装进行购买决策评估。

决策结构（{len(layers)}层）：{layer_names}

重要规则：
1. 严格按照你所扮演的人设去思考和判断，不要站在"一般消费者"角度
2. 每个决策层分开独立评分，不要混
3. 评分使用1-10分，1=完全不买，10=立刻想买
4. 输出纯JSON，不要任何额外解释文字"""


def _render_influencer_profiles(
    brand_cfg: BrandConfig | None,
    layers: list[DecisionLayer],
    age_rule: dict[str, Any],
) -> str:
    """渲染影响层（role=veto）轴画像：该年龄段下的否决条件。

    轴数据按 layer.persona_axis_key 从品牌 personas.yaml 顶层收集（通用），
    条目含 age / gender / veto_when 等字段（字段名由 YAML 约定）。
    """
    if brand_cfg is None or not brand_cfg.persona_axes:
        return ""
    rng = age_rule.get("age_range", [0, 999])
    blocks: list[str] = []
    for layer in layers:
        if layer.role == "decider":
            continue
        axes = brand_cfg.persona_axes.get(layer.persona_axis_key) or []
        lines = [f"【{layer.name}画像与否决线】"]
        for entry in axes:
            age = entry.get("age", 0)
            if age < rng[0] or age > rng[1]:
                continue
            veto_when = "、".join(str(v) for v in entry.get("veto_when", []))
            gender = entry.get("gender", "")
            gender_desc = f"，{gender}孩" if gender else ""
            lines.append(f"- {age}岁{gender_desc}：出现任一情形会否决 → {veto_when}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_persona_prompt(
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    layers: list[DecisionLayer],
    layer_weights: dict[str, float],
    brand_cfg: BrandConfig | None,
    age_rule: dict[str, Any],
) -> tuple[str, str]:
    """返回 (system, user) prompt —— 决策层动态渲染"""
    pid = _persona_key(persona)
    brand_name = brand_cfg.brand_name if brand_cfg is not None else "品牌"

    # 层权重描述
    weight_lines = [
        f"- {l.name}（权重{int(layer_weights.get(l.id, 1.0) * 100)}%）"
        for l in layers
    ]

    # 决策者画像（persona 通用字段，字段名由 personas.yaml 约定）
    fab_focus = "、".join(str(x) for x in persona.get("fab_focus", [])) or "不限"
    color_pref = "、".join(str(x) for x in persona.get("color_preference", [])) or "不限"

    influencer_block = _render_influencer_profiles(brand_cfg, layers, age_rule)

    # 输出 schema 动态生成
    score_fields = "\n".join(
        f'  "{l.id}_score": 数字1-10,\n  "{l.id}_reason": "一句话",'
        for l in layers
    )
    score_tasks = "\n\n".join(
        f"{i+1}) {l.name} 评分（{l.id}_score，1-10）\n"
        f"   {l.name}按自己的关注点独立判断这款值不值得买。\n"
        f"   {l.id}_reason：一句话说明{l.name}的核心判断理由"
        for i, l in enumerate(layers)
    )
    veto_hint = (
        f"   - 任一否决层明确命中其否决线，或决策者层分 < {VETO_SCORE_THRESHOLD}，"
        f"输出 vetoed=true（即使其他层喜欢也不买）"
        if any(l.role == "veto" for l in layers) else
        f"   - 决策者层分 < {VETO_SCORE_THRESHOLD} 时输出 vetoed=true"
    )

    user_msg = f"""
【你扮演的人设】
姓名：{persona.get('name', pid)}（人设ID: {pid}）
购买关注点：{fab_focus}
颜色偏好：{color_pref}

【决策结构】
{chr(10).join(weight_lines)}
{influencer_block}

【款式信息】
款号：{info.style_id}
品类：{info.category or '未标注'}
价格：{info.price}元
季节：{info.season or '未标注'}
FAB描述：
{info.fab_description or '无FAB描述，请根据以下10个结构化特征判断'}

【10个服装特征结构化评分（由视觉模型先行提取）】
{_feat_summary(feats)}

【任务】各决策层独立评分：

{score_tasks}

【综合判断】
{veto_hint}
   - 否则 final_score = {" + ".join(f"{l.id}_score × {layer_weights.get(l.id, 1.0):.2f}" for l in layers)}
   - 如果出现否决，opposing_reason说明否决的原因

【输出格式】纯JSON：
{{
{score_fields}
  "final_score": 数字1-10,
  "vetoed": true或false,
  "opposing_reason": "如果反对或否决，说明原因，否则空字符串"
}}
""".strip()
    return _render_persona_vote_system(layers, brand_name), user_msg


# ========== 单人设投票 ==========
def _vote_one_persona_one_model(
    client: BailianClient,
    *,
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    layers: list[DecisionLayer],
    layer_weights: dict[str, float],
    brand_cfg: BrandConfig | None,
    age_rule: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    sys_p, usr_p = _render_persona_prompt(
        persona, info, feats, layers, layer_weights, brand_cfg, age_rule,
    )
    resp = client.generate_text(
        usr_p,
        model=model,
        system_prompt=sys_p,
        temperature=0.8,
        max_tokens=1500,
    )
    if not resp.ok:
        raise RuntimeError(f"[人设投票] 人设{_persona_key(persona)}模型{model}失败: {resp.error}")
    try:
        parsed = extract_json(resp.content, as_dict=True)
        # Ollama 本地模型偶尔把单个结果包装成 list 返回 → 自动拆
        if isinstance(parsed, list):
            if len(parsed) == 0:
                raise ValueError("LLM 返回了空数组")
            if all(isinstance(x, dict) for x in parsed):
                log.warning(
                    "[%s] 人设%s 模型%s 返回 list 而非 dict（已自动取第一个元素），原文预览=%s",
                    info.style_id, _persona_key(persona), model, resp.content[:200],
                )
                parsed = parsed[0]
            else:
                raise ValueError(f"LLM 返回 list 但元素不是 dict: {type(parsed[0]) if parsed else 'empty'}")
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM 返回了 {type(parsed).__name__} 而非 dict")
        return parsed
    except Exception as e:
        log.warning("[%s] 人设%s 模型%s JSON解析失败: %s, 原文=%s",
                    info.style_id, _persona_key(persona), model, e, resp.content[:300])
        raise


def vote_persona(
    client: BailianClient,
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    cfg: AppConfig | None = None,
    *,
    brand_cfg: BrandConfig | None = None,
    target_age: int | None = None,
) -> PersonaVote:
    """单人设 + 多模型混合，均值聚合（决策层通用）"""
    pid = _persona_key(persona)
    models = cfg.api.persona_models if cfg is not None else _DEFAULT_PERSONA_MODELS
    # 去重提速：Ollama/智谱映射后可能重复 → 直接砍掉，节省一半请求
    models = resolve_and_dedupe_models(client, models, is_vlm=False)

    layers = _resolve_layers(brand_cfg)
    if brand_cfg is not None:
        ds = brand_cfg.decision_structure
        age_rule = _match_age_weight_rules(
            ds.age_weight_rules,
            target_age if target_age is not None else ds.default_target_age,
        )
        layer_weights = _layer_weights_for_age(ds, target_age)
    else:
        age_rule = {}
        layer_weights = {layers[0].id: 1.0}

    per_model: list[dict[str, Any]] = []
    per_model_scores: dict[str, float] = {}
    for m in models:
        try:
            res = _vote_one_persona_one_model(
                client, persona=persona, info=info, feats=feats,
                layers=layers, layer_weights=layer_weights,
                brand_cfg=brand_cfg, age_rule=age_rule, model=m,
            )
            per_model.append(res)
            per_model_scores[m] = clamp(safe_float(res.get("final_score"), 5.0), 1.0, 10.0)
        except Exception as e:
            log.error("[%s] 人设%s 模型%s失败: %s", info.style_id, pid, m, e)

    if not per_model:
        # 全部失败，返回安全占位分
        return PersonaVote(
            persona_id=pid,
            persona_name=str(persona.get("name", pid)),
            layer_scores={l.id: 5.0 for l in layers},
            final_score=5.0,
        )

    # 各层分数跨模型均值
    layer_scores: dict[str, float] = {}
    for l in layers:
        vals = [clamp(safe_float(r.get(f"{l.id}_score"), 5.0)) for r in per_model]
        layer_scores[l.id] = float(mean(vals))
    final_scores = [clamp(safe_float(r.get("final_score"), 5.0)) for r in per_model]

    # 选首个成功模型的理由文本
    sample = per_model[0]
    layer_reasons: dict[str, str] = {
        l.id: str(sample.get(f"{l.id}_reason", "")) for l in layers
    }
    # 否决：LLM 明确输出，或决策者层低分
    decider_low = any(
        layer_scores.get(l.id, 5.0) < VETO_SCORE_THRESHOLD
        for l in layers if l.role == "decider"
    )
    vetoed = bool(sample.get("vetoed", False)) or decider_low

    return PersonaVote(
        persona_id=pid,
        persona_name=str(persona.get("name", pid)),
        layer_scores=layer_scores,
        layer_reasons=layer_reasons,
        final_score=float(mean(final_scores)),
        opposing_reason=str(sample.get("opposing_reason", "")),
        vetoed=vetoed,
        model_scores=per_model_scores,
    )


# ========== 投票聚合 ==========
def _cluster_reasons(reason_list: list[tuple[str, float]], top_n: int = 3) -> list[str]:
    """简单的"关键词频次"聚类，取Top N高频理由"""
    counter: Counter[str] = Counter()
    for reason, weight in reason_list:
        if not reason:
            continue
        for seg in re.split(r"[，。,.;；]\s*", reason):
            seg = seg.strip()
            if len(seg) >= 3:
                counter[seg] += int(weight * 10) + 1
    return [item for item, _ in counter.most_common(top_n)]


def _primary_reason(v: PersonaVote) -> str:
    """取投票的主要文本理由：首个非空的层理由"""
    for reason in v.layer_reasons.values():
        if reason:
            return reason
    return ""


def _extract_style_keywords(feats: StyleFeatures, info: StyleInfo) -> list[str]:
    """从features_bars分+FAB描述+颜色描述提取关键词，用于否决层匹配"""
    keywords: list[str] = []
    fab_text = (info.fab_description or "").lower()
    keywords.extend(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", fab_text))
    for f in feats.features.values():
        reason = (f.reason or "").lower()
        keywords.extend(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", reason))
    return [k for k in keywords if len(k) >= 2]


def _check_axis_veto(
    brand_cfg: BrandConfig | None,
    layers: list[DecisionLayer],
    age_rule: dict[str, Any],
    style_keywords: list[str],
) -> tuple[bool, str]:
    """扫描否决层（role=veto）轴数据的 veto_when 条件，与该款关键词匹配。

    轴数据按 layer.persona_axis_key 从品牌 persona_axes 取（通用）。
    只要命中任意1个否决关键词就触发。返回 (是否否决, 否决原因)
    """
    if brand_cfg is None or not brand_cfg.persona_axes:
        return False, ""
    rng = age_rule.get("age_range", [0, 999])
    for layer in layers:
        if layer.role == "decider":
            continue
        axes = brand_cfg.persona_axes.get(layer.persona_axis_key) or []
        for entry in axes:
            entry_age = int(entry.get("age", 0))
            if entry_age < rng[0] or entry_age > rng[1]:
                continue
            for veto_kw in entry.get("veto_when", []):
                veto_kw_lower = str(veto_kw).lower()
                for style_kw in style_keywords:
                    if veto_kw_lower in style_kw.lower() or style_kw.lower() in veto_kw_lower:
                        return True, f"{layer.name}否决：{veto_kw}"
    return False, ""


def aggregate_votes(
    votes: list[PersonaVote],
    personas_cfg: dict[str, Any] | None = None,
    *,
    brand_cfg: BrandConfig | None = None,
    feats: StyleFeatures | None = None,
    info: StyleInfo | None = None,
    target_age: int | None = None,
) -> VotingResult:
    """按人设分布权重加权聚合（决策层通用）

    - 无 brand_cfg / single_layer：人设加权投票（weight×individual_score / ∑w）
    - multi_layer / double_layer：
        1) 人设层加权得到基础分
        2) 根据 target_age 匹配 age_weight_rules 取层权重
        3) 否决层（role=veto）扫描 persona_axes 中该年龄段的 veto_when，
           若命中任意1条 → 基础分 × VETO_PENALTY_FACTOR 惩罚

    命名约定：spec §7.x 历史用 double_layer，代码实现用 multi_layer
    （允许多于2层）。两者等价，参见 BrandDecisionStructure.type 注释。
    """
    style_id = ""
    weighted_total = 0.0
    weight_sum = 0.0
    oppose = 0
    support = 0
    all_scores: list[float] = []
    buy_reasons_weighted: list[tuple[str, float]] = []
    oppose_reasons_weighted: list[tuple[str, float]] = []
    high_div: list[str] = []

    # ========== 构建权重映射 ==========
    weight_map: dict[str, float] = {}
    if brand_cfg is not None:
        if brand_cfg.personas_weights is not None:
            weight_map = dict(brand_cfg.personas_weights)
        else:
            persona_list = brand_cfg.personas
            weight_map = {
                _persona_key(p): float(p.get("weight", 1.0 / max(1, len(persona_list))))
                for p in persona_list
            }
    elif personas_cfg is not None:
        persona_list = personas_cfg.get("personas", [])
        weight_map = {
            _persona_key(p): float(p.get("weight", 1.0 / max(1, len(persona_list))))
            for p in persona_list
        }

    default_w = 1.0 / max(1, len(votes)) if weight_map else 1.0

    # ========== 人设层基础加权分 ==========
    individual_scores: dict[str, float] = {}
    for v in votes:
        if not style_id:
            style_id = "unknown"
        w = weight_map.get(v.persona_id, default_w)
        individual_scores[v.persona_id] = clamp(v.final_score, 1.0, 10.0)
        weighted_total += individual_scores[v.persona_id] * w
        weight_sum += w
        all_scores.append(individual_scores[v.persona_id])
        primary = _primary_reason(v)
        if individual_scores[v.persona_id] < 4.0 or v.vetoed:
            oppose += 1
            if v.opposing_reason:
                oppose_reasons_weighted.append((v.opposing_reason, w))
            elif primary:
                oppose_reasons_weighted.append((primary, w))
        if individual_scores[v.persona_id] >= 7.0:
            support += 1
            if primary:
                buy_reasons_weighted.append((primary, w))
        scores = list(v.model_scores.values())
        if len(scores) >= 2:
            div = max(scores) - min(scores)
            if div >= 3.0:
                high_div.append(v.persona_id)

    n = len(votes) or 1
    base_weighted = weighted_total / weight_sum if weight_sum > 0 else 5.0

    # ========== 决策结构分支（否决层轴扫描）==========
    final_weighted = base_weighted
    veto_applied = False
    veto_reason = ""

    if brand_cfg is not None:
        ds = brand_cfg.decision_structure
        # spec §7.x 历史命名：double_layer；代码实现用 multi_layer。
        # 此处接受两者为别名，避免 spec-following YAML 静默落入 neither 分支
        # （否则否决层不生效，无报错）。
        if ds.type in ("single_layer",):
            final_weighted = base_weighted
        elif ds.type in ("multi_layer", "double_layer"):
            layers = list(ds.layers)
            age_rule = _match_age_weight_rules(
                ds.age_weight_rules,
                target_age if target_age is not None else ds.default_target_age,
            )
            layer_weights = _layer_weights_for_age(ds, target_age)
            veto_layer_ids = {l.id for l in layers if l.role != "decider"}
            has_veto_layer = bool(veto_layer_ids)
            veto_layer_weight = sum(
                w for lid, w in layer_weights.items() if lid in veto_layer_ids
            )

            if feats is not None and info is not None and has_veto_layer and veto_layer_weight > 0:
                style_keywords = _extract_style_keywords(feats, info)
                vetoed, reason = _check_axis_veto(brand_cfg, layers, age_rule, style_keywords)
                if vetoed:
                    final_weighted = base_weighted * VETO_PENALTY_FACTOR
                    veto_applied = True
                    veto_reason = reason
                    oppose += 1
                    oppose_reasons_weighted.append((reason, veto_layer_weight))
                else:
                    final_weighted = base_weighted
            else:
                final_weighted = base_weighted

    if veto_applied and veto_reason:
        if veto_reason not in [r[0] for r in oppose_reasons_weighted]:
            oppose_reasons_weighted.append((veto_reason, veto_layer_weight if veto_layer_weight > 0 else 0.30))

    return VotingResult(
        style_id=style_id,
        votes=votes,
        weighted_score=clamp(final_weighted, 1.0, 10.0),
        opposition_rate=oppose / n,
        support_rate=support / n,
        top_buy_reasons=_cluster_reasons(buy_reasons_weighted),
        top_oppose_reasons=_cluster_reasons(oppose_reasons_weighted),
        score_std=pstdev(all_scores) if len(all_scores) >= 2 else 0.0,
        high_divergence_personas=high_div,
    )


# ========== 批量人设投票 ==========
def run_persona_voting(
    client: BailianClient,
    info: StyleInfo,
    feats: StyleFeatures,
    cfg: AppConfig | None = None,
    *,
    progress: bool = False,
    brand_cfg: BrandConfig | None = None,
    target_age: int | None = None,
) -> VotingResult:
    """批量人设投票（决策层通用）

    函数签名保持兼容：优先使用 brand_cfg（新架构），否则退化为单决策层。
    """
    if brand_cfg is not None:
        persona_list = brand_cfg.personas
    elif cfg is not None:
        persona_list = cfg.personas.get("personas", [])
    else:
        raise ValueError("run_persona_voting: cfg 和 brand_cfg 不能同时为 None")

    votes: list[PersonaVote] = []
    pbar = tqdm(persona_list, desc=f"人设投票[{info.style_id}]", leave=False, disable=not progress)
    for p in pbar:
        pid = _persona_key(p)
        pbar.set_postfix_str(pid)
        try:
            v = vote_persona(
                client, p, info, feats, cfg,
                brand_cfg=brand_cfg, target_age=target_age,
            )
            votes.append(v)
        except Exception as e:
            log.error("[%s] 人设%s 全部失败: %s", info.style_id, pid, e)
            votes.append(PersonaVote(
                persona_id=pid,
                persona_name=str(p.get("name", pid)),
                final_score=5.0,
                opposing_reason="LLM调用失败",
            ))

    if brand_cfg is not None:
        result = aggregate_votes(
            votes,
            None,
            brand_cfg=brand_cfg,
            feats=feats,
            info=info,
            target_age=target_age,
        )
    else:
        result = aggregate_votes(votes, cfg.personas if cfg else None)
    result.style_id = info.style_id
    return result


# ========== PersonaVotingEngine（BrandConfig构造注入）==========
class PersonaVotingEngine:
    """人设投票引擎：构造函数接受 BrandConfig，内部按决策结构运行"""

    def __init__(self, brand_cfg: BrandConfig):
        self.brand_cfg = brand_cfg

    def aggregate(
        self,
        votes: list[PersonaVote],
        *,
        feats: StyleFeatures | None = None,
        info: StyleInfo | None = None,
        target_age: int | None = None,
    ) -> VotingResult:
        return aggregate_votes(
            votes,
            None,
            brand_cfg=self.brand_cfg,
            feats=feats,
            info=info,
            target_age=target_age,
        )
