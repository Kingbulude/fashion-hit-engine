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
def _feat_summary(feats: StyleFeatures, *, top_n: int = 3, max_reason_len: int = 50) -> str:
    """瘦身版特征摘要：只给 TopN 高分 + TopN 低分，砍掉中间平庸项。

    动机：7B 本地模型注意力广度有限，10 个全塞进去容易"平均化处理"
    （全打 6-8 分）。只给头尾 6 个 + 截短 reason，既保留梯度信息又减 token。
    实测 prompt 从 ~220 tokens 降到 ~60 tokens，压缩 3.7x。
    """
    def _level(sc: float) -> str:
        if sc >= 8: return "非常高"
        if sc >= 6.5: return "较高"
        if sc >= 5.0: return "中等"
        if sc >= 3.5: return "较低"
        return "非常低"

    sorted_feats = sorted(
        feats.features.items(), key=lambda kv: -kv[1].score
    )
    n = len(sorted_feats)
    if n <= top_n * 2:
        # 特征总数太少（<6），全列出来
        full = []
        for key, f in sorted_feats:
            reason = (f.reason or "无细节")[:max_reason_len]
            full.append(f"· {f.name}：{f.score:.1f}/10（{_level(f.score)}）— {reason}")
        return "\n".join(full)

    high = sorted_feats[:top_n]
    low = sorted_feats[-top_n:]
    lines: list[str] = []
    lines.append("【优势特征】（对销量是加分项）")
    for key, f in high:
        reason = (f.reason or "")[:max_reason_len]
        lines.append(f"  ↑ {f.name}：{f.score:.1f}/10（{_level(f.score)}）{('— ' + reason) if reason else ''}")
    lines.append(f"【劣势特征】（对销量有风险）")
    for key, f in low:
        reason = (f.reason or "")[:max_reason_len]
        lines.append(f"  ↓ {f.name}：{f.score:.1f}/10（{_level(f.score)}）{('— ' + reason) if reason else ''}")

    # 整体调性一句话（让人设知道这款大致在什么段位）
    all_scores = [f.score for _, f in sorted_feats]
    avg = sum(all_scores) / len(all_scores)
    lines.append(f"【整体调性】平均 {avg:.1f}/10，特征梯度明确")
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

    # axes 三维锚点（这是人设最核心的定位，之前漏进 prompt 了）
    axes_block_parts: list[str] = []
    for axis_name in ("scene", "aesthetic", "price"):
        axis_val = persona.get("axes", {}).get(axis_name, "") if isinstance(persona.get("axes"), dict) else persona.get(f"axis_{axis_name}", "")
        if axis_val:
            axes_block_parts.append(f"{axis_name}={axis_val}")
    axes_block = "、".join(axes_block_parts) if axes_block_parts else "未标注"

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
三维定位：{axes_block}
购买关注点：{fab_focus}
主推色偏好：{color_pref}（注意：一款可能有多个颜色，你只关心主推色/主图色是否对你的胃口，其他颜色不影响你的评分——多色覆盖是优点不是风险）

【决策结构】
{chr(10).join(weight_lines)}
{influencer_block}

【款式信息】
款号：{info.style_id}
品类：{info.category or '未标注'}
价格：{info.price}元
季节：{info.season or '未标注'}
FAB描述：
{info.fab_description or '无FAB描述，请根据以下结构化特征判断'}

【关键服装特征（精简版，只列最重要的）】
{_feat_summary(feats)}

【任务】各决策层独立评分：

{score_tasks}

【综合判断】
{veto_hint}
   - final_score 由系统根据各层分数按权重自动计算，你不用算
   - 如果出现否决，opposing_reason说明否决的原因

【输出格式】纯JSON（不要任何额外文字）：
{{
{score_fields}
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
        # v1.4.39: 0.8 → 0.5。人设投票需要稳定可复现的人格化打分，
        # 不是创意写作。0.8 下同一张图让同一人设投两次，打分差可达 1.5-2.0/10；
        # 0.5 时稳定在 <1.0/10，理由输出仍然自然。14B 模型在 0.5 下质量最优。
        temperature=0.5,
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
            # final_score 由代码按权重算，不再依赖 LLM 输出的 final_score
            per_model_layer_scores = {
                l.id: clamp(safe_float(res.get(f"{l.id}_score"), 5.0), 1.0, 10.0)
                for l in layers
            }
            per_model_scores[m] = clamp(
                sum(
                    per_model_layer_scores[l.id] * layer_weights.get(l.id, 1.0)
                    for l in layers
                ), 1.0, 10.0,
            )
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

    # final_score = 各层分 × 权重（代码算，避免 LLM 算术错误）
    final_score_from_layers = clamp(
        sum(layer_scores[l.id] * layer_weights.get(l.id, 1.0) for l in layers),
        1.0, 10.0,
    )

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
        final_score=float(final_score_from_layers),
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

# ======================================================================
# 三阶段评审（v1.4.42.2+）
# Phase1 初评：现有人设独立评分（vote_persona 不动）
# Phase2 外部质疑：品类总监视角对 Phase1 结果做结构化挑战
# Phase3 复评：每个人设基于专家质疑做二次判断（人设之间不互看，避免趋同）
# 设计原则：人设独立性优先，外部质疑作为统一校准信号
# ======================================================================

def _persona_scores_summary(
    votes_initial: list[PersonaVote],
    layers: list[DecisionLayer],
    *,
    max_per_layer: int = 5,
) -> str:
    """把 Phase1 初评结果摘要成人类可读文本，喂给专家。
    
    格式：每个决策层 → 所有人设的分数分布 + TopN 理由摘要。
    """
    lines: list[str] = []
    for layer in layers:
        layer_scores = [
            (v.persona_name, v.layer_scores.get(layer.id, 5.0), v.layer_reasons.get(layer.id, ""))
            for v in votes_initial
        ]
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"【{layer.name}（{layer.id}）层初评分布】")
        for name, score, reason in layer_scores[:max_per_layer]:
            reason_short = (reason or "无")[:60]
            lines.append(f"  ↑ {score:.1f}/10 — {name}：{reason_short}")
        lines.append(f"  （均值={sum(s for _,s,_ in layer_scores)/len(layer_scores):.1f}  反对数={sum(1 for _,s,_ in layer_scores if s<4)}）")
        lines.append("")
    return chr(10).join(lines)


def _render_expert_challenge_prompt(
    brand_name: str,
    info: StyleInfo,
    feats: StyleFeatures,
    voting: VotingResult,
    layers: list[DecisionLayer],
    layers_summary: str,
) -> tuple[str, str]:
    """渲染品类专家 prompt — 独立于所有人设的外部质疑视角。
    
    专家不是"另一类人设"，而是"品类总监"——她不买衣服，但负责判断
    这批人设的评估有没有系统性偏差。
    """
    layers_desc = "、".join(l.name for l in layers)
    sys_prompt = f"""你是{brand_name}的品类总监，有10年服装电商经验。
你的任务不是买衣服，而是审查一群模拟消费者人设的评估，找出他们的盲区和系统性偏差。"""

    feat_brief = []
    for key, f in sorted(feats.features.items(), key=lambda kv: -kv[1].score):
        feat_brief.append(f"  {f.name}：{f.score:.1f}/10")

    user_prompt = f"""
请审查以下评估：

【款式】{info.style_id} · {info.category} · ¥{info.price} · {info.fab_description or ''}
【核心特征】
{chr(10).join(feat_brief)}
【当前批次整体状态】加权分={voting.weighted_score:.1f}/10  反对率={voting.opposition_rate:.0%}  支持率={voting.support_rate:.0%}

【{layers_desc}层的人设初评结果】
{layers_summary}

【你的任务】请严格按 JSON 输出，给出：
{{
  "systematic_bias": "最突出的系统性偏差是什么？（一句话，比如'所有人设都因为颜色打高分但都忽略了面料质感'）",
  "challenges": [
    {{"layer": "层名", "issue": "具体问题", "severity": "high/medium/low", "suggestion": "应该怎么修正"}}
  ],
  "highlighted_blind_spots": ["盲区1", "盲区2"],
  "confidence_in_assessment": "float 0-1"
}}

规则：
- 你的目标是找盲区，不是给这款打高分/低分
- 如果人设评估没有明显偏差，可以说"无显著系统性偏差"
- 不要重复所有人设已经说了的理由，只说他们没说的
""".strip()
    return sys_prompt, user_prompt


def _expert_challenge(
    client: Any,
    *,
    votes_initial: list[PersonaVote],
    info: StyleInfo,
    feats: StyleFeatures,
    layers: list[DecisionLayer],
    brand_cfg: BrandConfig | None = None,
    voting_initial: VotingResult | None = None,
    model: str = "qwen-max",
) -> dict[str, Any] | None:
    """Phase2：跑品类专家挑战。失败返回 None（优雅 fallback 到两阶段）。"""
    try:
        if brand_cfg is None:
            brand_name = "品牌"
        else:
            brand_name = brand_cfg.brand_name

        if voting_initial is None:
            # 先用初评结果算一个临时的 VotingResult（只需要 summary）
            voting_initial = aggregate_votes(votes_initial, None, brand_cfg=brand_cfg)

        layers_summary = _persona_scores_summary(votes_initial, layers)
        sys_p, usr_p = _render_expert_challenge_prompt(
            brand_name, info, feats, voting_initial, layers, layers_summary,
        )

        resp = client.generate_text(
            usr_p,
            model=model,
            system_prompt=sys_p,
            temperature=0.3,  # 专家需要稳定的批评，不要创意发散
            max_tokens=1200,
        )
        if not resp.ok:
            log.warning("[%s] 专家挑战失败: %s", info.style_id, resp.error)
            return None

        from .types import extract_json
        parsed = extract_json(resp.content, as_dict=True)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        if not isinstance(parsed, dict):
            log.warning("[%s] 专家返回非 dict: %s", info.style_id, type(parsed))
            return None
        log.info("[%s] 专家挑战成功: %s", info.style_id, parsed.get("systematic_bias", "")[:60])
        return parsed
    except Exception as e:
        log.warning("[%s] 专家挑战异常: %s", info.style_id, e)
        return None


def _render_review_prompt(
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    layers: list[DecisionLayer],
    layer_weights: dict[str, float],
    initial_layer_scores: dict[str, float],
    initial_layer_reasons: dict[str, str],
    expert_challenge: dict[str, Any],
    brand_cfg: BrandConfig | None = None,
) -> tuple[str, str]:
    """渲染 Phase3 复评 prompt：初评分数 + 自己的理由 + 专家质疑 → 二次判断。"""
    pid = _persona_key(persona)
    brand_name = brand_cfg.brand_name if brand_cfg else "品牌"

    # 专家质疑摘要（只取最相关的几个点）
    challenges = expert_challenge.get("challenges", []) or []
    bias = expert_challenge.get("systematic_bias", "")
    if challenges:
        chall_lines = []
        for c in challenges[:3]:
            layer_name = c.get("layer", "")
            issue = c.get("issue", "")
            suggestion = c.get("suggestion", "")
            chall_lines.append(f"- 对{layer_name}层质疑：{issue} → 建议：{suggestion}")
        expert_text = bias + chr(10) + chr(10).join(chall_lines)
    else:
        expert_text = bias or "专家认为初评无显著系统性偏差"

    # 初评分数摘要
    initial_brief = []
    for l in layers:
        score = initial_layer_scores.get(l.id, 5.0)
        reason = initial_layer_reasons.get(l.id, "")
        initial_brief.append(f"  {l.name}：{score:.1f}/10  原理由：{(reason or '无')[:60]}")

    weight_lines = [
        f"- {l.name}（权重{int(layer_weights.get(l.id, 1.0) * 100)}%）"
        for l in layers
    ]

    fab_focus = "、".join(str(x) for x in persona.get("fab_focus", [])) or "不限"
    color_pref = "、".join(str(x) for x in persona.get("color_preference", [])) or "不限"

    sys_prompt = f"""你是{brand_name}的购买决策模拟器。你将扮演一个具体的人设，独立判断一件服装。
重要规则：你可以参考专家的意见，但保持独立思考——专家也可能错。"""

    user_prompt = f"""
【你扮演的人设】{persona.get('name', pid)}
购买关注点：{fab_focus}
主推色偏好：{color_pref}
【决策结构】{chr(10).join(weight_lines)}

【款式】{info.style_id} · {info.category} · ¥{info.price}
【核心特征摘要】
{_feat_summary(feats)}

【你的初评结果（Phase1）】
{chr(10).join(initial_brief)}

【品类总监质疑（Phase2）】
{expert_text}

【你的任务】
1. 逐审阅你的初评——专家指出的盲区是否在你的评估里存在？
2. 如果存在，最多可以调整 ±2 分（不要完全推翻你自己）。如果专家不对，坚持你的初评。
3. 输出 JSON：

{{
{chr(10).join(
    f'  "{l.id}_adj": {initial_layer_scores.get(l.id, 5.0):.1f}  // 调整后的新分，和初评相同表示不采纳专家意见'
    for l in layers
)},
  "adopted_feedback": "你采纳了专家哪些意见？如果都不采纳说明理由",
  "vetoed": true或false,
  "opposing_reason": "反对原因"
}}

规则：
- 每层调整幅度 ≤ 2 分（调整后分数 1-10）
- 如果某层初评 8.0，专家说"面料问题被忽略"，你最多调到 6.0
- 保持人设一致性，不要变成"品类总监"
""".strip()
    return sys_prompt, user_prompt


def _run_persona_review(
    client: Any,
    *,
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    layers: list[DecisionLayer],
    layer_weights: dict[str, float],
    initial_layer_scores: dict[str, float],
    initial_layer_reasons: dict[str, str],
    expert_challenge: dict[str, Any],
    brand_cfg: BrandConfig | None = None,
    model: str = "qwen-max",
    max_adjust: float = 2.0,
) -> dict[str, Any]:
    """Phase3 单人设复评。失败返回不调整的安全结果。"""
    try:
        sys_p, usr_p = _render_review_prompt(
            persona, info, feats, layers, layer_weights,
            initial_layer_scores, initial_layer_reasons,
            expert_challenge, brand_cfg,
        )
        resp = client.generate_text(usr_p, model=model, system_prompt=sys_p, temperature=0.5, max_tokens=1000)
        if not resp.ok:
            raise RuntimeError(resp.error)

        from .types import extract_json, clamp
        parsed = extract_json(resp.content, as_dict=True)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not isinstance(parsed, dict):
            raise ValueError(f"review 返回 {type(parsed).__name__}")

        new_layer_scores: dict[str, float] = {}
        for l in layers:
            new_key = f"{l.id}_adj"
            new_val = parsed.get(new_key)
            if new_val is None:
                # 没给 → 不调整
                new_layer_scores[l.id] = initial_layer_scores.get(l.id, 5.0)
                continue
            try:
                new_f = float(new_val)
            except (TypeError, ValueError):
                new_layer_scores[l.id] = initial_layer_scores.get(l.id, 5.0)
                continue
            # 限制调整幅度
            orig = initial_layer_scores.get(l.id, 5.0)
            delta = new_f - orig
            if abs(delta) > max_adjust:
                new_f = orig + max_adjust * (1 if delta > 0 else -1)
            new_layer_scores[l.id] = clamp(new_f, 1.0, 10.0)

        return {
            "layer_scores": new_layer_scores,
            "adopted_feedback": str(parsed.get("adopted_feedback", "")),
            "vetoed": bool(parsed.get("vetoed", False)),
            "opposing_reason": str(parsed.get("opposing_reason", "")),
        }
    except Exception as e:
        log.warning("[%s] 人设%s 复评失败，保留初评: %s", info.style_id, _persona_key(persona), e)
        return {
            "layer_scores": dict(initial_layer_scores),
            "adopted_feedback": "",
            "vetoed": False,
            "opposing_reason": "",
        }


def run_persona_voting(
    client: BailianClient,
    info: StyleInfo,
    feats: StyleFeatures,
    cfg: AppConfig | None = None,
    *,
    progress: bool = False,
    brand_cfg: BrandConfig | None = None,
    target_age: int | None = None,
    three_phase: bool = True,
    expert_model: str = "qwen-max",
) -> VotingResult:
    """批量人设投票（决策层通用）— 三阶段评审 v1.4.42.2

    Phase1 初评：所有人设独立评分（现有 vote_persona 逻辑不变）
    Phase2 外部质疑：品类总监视角对 Phase1 结果做结构化挑战（1次调用）
    Phase3 复评：每个人设基于统一的专家质疑做二次判断（±2分限制）

    降级：expert_challenge 失败 → 自动 fallback 到两阶段（不阻塞）。
    mock client 上自动降级。
    """
    if brand_cfg is not None:
        persona_list = brand_cfg.personas
    elif cfg is not None:
        persona_list = cfg.personas.get("personas", [])
    else:
        raise ValueError("run_persona_voting: cfg 和 brand_cfg 不能同时为 None")

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

    # Phase1: 初评
    log.info("[%s] Phase1 人设初评（%d 人设）", info.style_id, len(persona_list))
    votes: list[PersonaVote] = []
    pbar = tqdm(persona_list, desc=f"人设初评[{info.style_id}]", leave=False, disable=not progress)
    for p in pbar:
        pid = _persona_key(p)
        pbar.set_postfix_str(pid)
        try:
            v = vote_persona(client, p, info, feats, cfg,
                            brand_cfg=brand_cfg, target_age=target_age)
            v.initial_layer_scores = dict(v.layer_scores)
            votes.append(v)
        except Exception as e:
            log.error("[%s] 人设%s 全部失败: %s", info.style_id, pid, e)
            fallback = PersonaVote(
                persona_id=pid, persona_name=str(p.get("name", pid)),
                final_score=5.0, opposing_reason="LLM调用失败",
            )
            fallback.initial_layer_scores = {l.id: 5.0 for l in layers}
            votes.append(fallback)

    # Phase2: 外部质疑
    expert_challenge: dict[str, Any] | None = None
    if three_phase and len(persona_list) >= 3:
        try:
            voting_initial = aggregate_votes(
                votes, None, brand_cfg=brand_cfg, feats=feats, info=info,
                target_age=target_age,
            )
            expert_challenge = _expert_challenge(
                client, votes_initial=votes, info=info, feats=feats,
                layers=layers, brand_cfg=brand_cfg, voting_initial=voting_initial,
                model=expert_model,
            )
            if expert_challenge:
                log.info("[%s] Phase2 专家挑战: %s", info.style_id,
                         expert_challenge.get("systematic_bias", "")[:80])
            else:
                log.info("[%s] Phase2 专家挑战跳过（返回空）", info.style_id)
        except Exception as e:
            log.warning("[%s] Phase2 专家挑战异常，降级为两阶段: %s", info.style_id, e)
            expert_challenge = None

    # Phase3: 复评
    if expert_challenge is not None:
        log.info("[%s] Phase3 人设复评", info.style_id)
        for v in votes:
            persona_dict = None
            for p in persona_list:
                if _persona_key(p) == v.persona_id:
                    persona_dict = p
                    break
            if persona_dict is None:
                continue

            review = _run_persona_review(
                client, persona=persona_dict, info=info, feats=feats,
                layers=layers, layer_weights=layer_weights,
                initial_layer_scores=v.initial_layer_scores,
                initial_layer_reasons=v.layer_reasons,
                expert_challenge=expert_challenge,
                brand_cfg=brand_cfg, model=expert_model,
            )

            v.review_delta = {
                lid: review["layer_scores"].get(lid, 5.0) - v.initial_layer_scores.get(lid, 5.0)
                for lid in layers
            }
            v.layer_scores = review["layer_scores"]
            v.review_adopted_feedback = review["adopted_feedback"]
            if review.get("vetoed"):
                v.vetoed = True
                if review.get("opposing_reason"):
                    v.opposing_reason = review["opposing_reason"]
            v.final_score = clamp(
                sum(v.layer_scores.get(l.id, 5.0) * layer_weights.get(l.id, 1.0) for l in layers),
                1.0, 10.0,
            )
    else:
        for v in votes:
            v.review_delta = {lid: 0.0 for lid in layers}

    # 聚合
    if brand_cfg is not None:
        result = aggregate_votes(
            votes, None, brand_cfg=brand_cfg, feats=feats, info=info,
            target_age=target_age,
        )
    else:
        result = aggregate_votes(votes, cfg.personas if cfg else None)
    result.style_id = info.style_id
    result.metadata = {
        "three_phase": expert_challenge is not None,
        "expert_bias": expert_challenge.get("systematic_bias", "") if expert_challenge else "",
    }
    return result
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
