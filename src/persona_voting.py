"""人设投票引擎：30人设双重决策 + 多模型混合

每个人设走三步漏斗：
  Step1 妈妈视角：穿搭需求判断 + 款式接受度 + 价值匹配 → mom_score
  Step2 孩子视角：颜色/图案/款式喜好 → child_score（只评图片可见特征，不含穿着体验）
  Step3 决策模式加权：综合分 = mom×W_mom + child×W_child
        - 孩子主导型若 mom_score < veto_threshold → 直接否决
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from statistics import mean, pstdev
from typing import Any

from tqdm import tqdm

from .config import AppConfig
from .llm_client import BailianClient
from .types import (
    BrandConfig,
    FeatureScore,
    PersonaVote,
    StyleFeatures,
    StyleInfo,
    VotingResult,
    clamp,
    extract_json,
    safe_float,
)

log = logging.getLogger(__name__)


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


# ========== 人设Prompt ==========
PERSONA_VOTE_SYSTEM = """你是一个童装购买决策模拟器。你将扮演一位具体的妈妈，结合孩子的意见，对一件童装进行购买决策评估。

重要规则：
1. 严格按照你所扮演的人设去思考和判断，不要站在"一般消费者"角度
2. 妈妈和孩子是两个人，分开独立评分，不要混
3. 评分使用1-10分，1=完全不买，10=立刻想买
4. 输出纯JSON，不要任何额外解释文字"""


def _render_persona_prompt(
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    decision_weights: dict[str, dict[str, float]],
    mom_veto_threshold: float,
) -> tuple[str, str]:
    """返回 (system, user) prompt"""
    mom = persona["mom"]
    child = persona["child"]
    mode = persona["decision_mode"]
    w = decision_weights[mode]

    mode_desc = {
        "mom_dominant": f"孩子{persona['child_age_group']}岁，由妈妈主导决策，妈妈权重{int(w['mom']*100)}%，孩子意见权重{int(w['child']*100)}%",
        "joint_decision": f"孩子{persona['child_age_group']}岁，共同决策，妈妈权重{int(w['mom']*100)}%，孩子意见权重{int(w['child']*100)}%，孩子有否决权",
        "child_dominant": f"孩子{persona['child_age_group']}岁进入青春期，孩子主导挑款，妈妈权重{int(w['mom']*100)}%，孩子意见权重{int(w['child']*100)}%；但如果妈妈分<{mom_veto_threshold}，妈妈有一票否决权",
    }[mode]

    focus_feats_mom = ", ".join(persona.get("mom_focus_features", [])) or "全部"
    focus_feats_child = ", ".join(persona.get("child_focus_features", [])) or "外观"

    user_msg = f"""
【你扮演的妈妈形象】
姓名：{persona['name']}（人设ID: {persona['id']}）
所在城市：{mom['city_tier']}城市，家庭收入{mom['income_level']}
购买习惯：{mom['purchase_frequency']}购买
价值取向：{mom['value_orientation']}
最关心的点：{', '.join(mom['concerns'])}
对服装最在意的特征：{focus_feats_mom}

【孩子画像（{child.get('age_range', persona['child_age_group'])}岁，{persona['child_gender']}孩）】
颜色偏好：{child.get('color_preference', '不限')}
图案偏好：{child.get('pattern_preference', '不限')}
款式风格偏好：{child.get('style_preference', '不限')}

【决策模式】{mode_desc}

【款式信息】
款号：{info.style_id}
品类：{info.category or '未标注'}
价格：{info.price}元
季节：{info.season or '未标注'}
FAB描述：
{info.fab_description or '无FAB描述，请根据以下10个结构化特征判断'}

【10个服装特征结构化评分（由视觉模型先行提取）】
{_feat_summary(feats)}

【任务】分三步输出：

1) mom_score（妈妈视角评分，1-10）：
   Step1 穿搭需求判断：我家孩子当前缺不缺这种品类？（裤子/外套/T恤/羽绒服...）这个季节能穿吗？有没有类似的？
         → 不缺/不能穿 = 低分
   Step2 款式接受度：版型、颜色、风格，作为妈妈你能接受吗？最在意的那几个特征打分会拉低吗？
         → 不接受 = 低分
   Step3 价值匹配：这个价格配得上用料/功能/设计/品牌吗？会不会有更划算的？
         → 不值 = 低分
   mom_reason：一句话说明妈妈的核心判断理由

2) child_score（孩子视角评分，1-10）：
   孩子只从"好不好看/喜不喜欢"角度判断，只看颜色/图案/款式风格这些图片可见的外观。
   不考虑价格、面料质感、功能这些。价格孩子不管；功能如果是视觉上很酷（比如反光条/大口袋），孩子会喜欢，但如果是抽象的"防晒/防风"孩子不在意。
   child_reason：一句话说明孩子喜欢/不喜欢的原因

3) 综合判断：
   - 如果妈妈分 < {mom_veto_threshold} 且决策模式是孩子主导型，输出 vetoed=true（妈妈否决，即使孩子喜欢也不买）
   - 否则 final_score = mom_score × {w['mom']} + child_score × {w['child']}
   - 如果出现否决，opposing_reason说明妈妈否决的原因

【输出格式】纯JSON：
{{
  "mom_score": 数字1-10,
  "mom_reason": "一句话",
  "child_score": 数字1-10,
  "child_reason": "一句话",
  "final_score": 数字1-10,
  "vetoed": true或false,
  "opposing_reason": "如果反对或否决，说明原因，否则空字符串"
}}
""".strip()
    return PERSONA_VOTE_SYSTEM, user_msg


# ========== 单人设投票 ==========
def _vote_one_persona_one_model(
    client: BailianClient,
    *,
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    decision_weights: dict[str, dict[str, float]],
    mom_veto_threshold: float,
    model: str,
) -> dict[str, Any]:
    sys_p, usr_p = _render_persona_prompt(persona, info, feats, decision_weights, mom_veto_threshold)
    resp = client.generate_text(
        usr_p,
        model=model,
        system_prompt=sys_p,
        temperature=0.8,
        max_tokens=1500,
    )
    if not resp.ok:
        raise RuntimeError(f"[人设投票] 人设{persona['id']}模型{model}失败: {resp.error}")
    try:
        return extract_json(resp.content)
    except Exception as e:
        log.warning("[%s] 人设%s 模型%s JSON解析失败: %s, 原文=%s",
                    info.style_id, persona['id'], model, e, resp.content[:300])
        raise


def vote_persona(
    client: BailianClient,
    persona: dict[str, Any],
    info: StyleInfo,
    feats: StyleFeatures,
    cfg: AppConfig,
) -> PersonaVote:
    """单人设 + 多模型混合，中位数聚合"""
    personas_cfg = cfg.personas
    decision_weights = personas_cfg["decision_mode_weights"]
    veto_thr = float(personas_cfg.get("mom_veto_threshold", 3.0))
    models = cfg.api.persona_models

    per_model: list[dict[str, Any]] = []
    per_model_scores: dict[str, float] = {}
    for m in models:
        try:
            res = _vote_one_persona_one_model(
                client, persona=persona, info=info, feats=feats,
                decision_weights=decision_weights, mom_veto_threshold=veto_thr,
                model=m,
            )
            per_model.append(res)
            per_model_scores[m] = clamp(safe_float(res.get("final_score"), 5.0), 1.0, 10.0)
        except Exception as e:
            log.error("[%s] 人设%s 模型%s失败: %s", info.style_id, persona["id"], m, e)

    if not per_model:
        # 全部失败，返回中位数安全分
        return PersonaVote(
            persona_id=persona["id"],
            persona_name=persona.get("name", persona["id"]),
            decision_mode=persona["decision_mode"],
            mom_score=5.0, child_score=5.0, final_score=5.0,
        )

    mom_scores = [clamp(safe_float(r.get("mom_score"), 5.0)) for r in per_model]
    child_scores = [clamp(safe_float(r.get("child_score"), 5.0)) for r in per_model]
    final_scores = [clamp(safe_float(r.get("final_score"), 5.0)) for r in per_model]

    # 选首个成功模型的理由文本
    sample = per_model[0]
    vetoed = bool(sample.get("vetoed", False)) or any(
        clamp(safe_float(r.get("mom_score"), 5.0)) < veto_thr and persona["decision_mode"] == "child_dominant"
        for r in per_model
    )

    return PersonaVote(
        persona_id=persona["id"],
        persona_name=persona.get("name", persona["id"]),
        decision_mode=persona["decision_mode"],
        mom_score=float(sum(mom_scores) / len(mom_scores)),
        child_score=float(sum(child_scores) / len(child_scores)),
        final_score=float(sum(final_scores) / len(final_scores)),
        mom_reason=str(sample.get("mom_reason", "")),
        child_reason=str(sample.get("child_reason", "")),
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


def _match_age_weight_rules(
    age_weight_rules: list[dict[str, Any]],
    target_age: int,
) -> dict[str, Any]:
    """根据target_age匹配age_weight_rules取档"""
    for rule in age_weight_rules:
        rng = rule.get("age_range", [0, 999])
        if len(rng) == 2 and rng[0] <= target_age <= rng[1]:
            return rule
    return age_weight_rules[0] if age_weight_rules else {"mom_weight": 0.70, "child_weight": 0.30}


def _extract_style_keywords(feats: StyleFeatures, info: StyleInfo) -> list[str]:
    """从features_bars分+FAB描述+颜色描述提取关键词，用于孩子否决匹配"""
    keywords: list[str] = []
    fab_text = (info.fab_description or "").lower()
    keywords.extend(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", fab_text))
    for f in feats.features.values():
        reason = (f.reason or "").lower()
        keywords.extend(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", reason))
    return [k for k in keywords if len(k) >= 2]


def _check_child_veto(
    child_axes: list[dict[str, Any]] | None,
    age_rule: dict[str, Any],
    style_keywords: list[str],
    child_gender: str = "",
) -> tuple[bool, str]:
    """扫描孩子层veto_when条件，与该款关键词匹配。
    只要命中任意1个veto关键词就触发孩子否决。
    返回 (是否否决, 否决原因)
    """
    if not child_axes:
        return False, ""
    rng = age_rule.get("age_range", [0, 999])
    for child in child_axes:
        child_age = int(child.get("age", 0))
        if child_age < rng[0] or child_age > rng[1]:
            continue
        if child_gender and child.get("gender") and child.get("gender") != child_gender:
            continue
        veto_whens = child.get("veto_when", [])
        for veto_kw in veto_whens:
            veto_kw_lower = str(veto_kw).lower()
            for style_kw in style_keywords:
                if veto_kw_lower in style_kw.lower() or style_kw.lower() in veto_kw_lower:
                    return True, f"孩子否决：{veto_kw}"
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
    """按人设分布权重加权聚合（支持BrandConfig决策结构）

    - brand_cfg is None / single_layer：直接30人设加权投票（weight×individual_score / ∑w）
    - brand_cfg.decision_structure.type == multi_layer：
        1) 先妈妈层(P01-P30)加权得到 mom_weighted
        2) 根据target_age匹配age_weight_rules取档
        3) 孩子层扫描child_identity_axes里该年龄段veto_when，
           若命中任意1条 → 妈妈分×0.70惩罚（孩子否决）
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
                p.get("id", p.get("persona_id", "")): float(p.get("weight", 1.0 / max(1, len(persona_list))))
                for p in persona_list
            }
    elif personas_cfg is not None:
        persona_list = personas_cfg.get("personas", [])
        weight_map = {p["id"]: float(p.get("weight", 1.0 / max(1, len(persona_list)))) for p in persona_list}

    default_w = 1.0 / max(1, len(votes)) if weight_map else 1.0

    # ========== 计算基础加权分（妈妈层/单一层）==========
    individual_scores: dict[str, float] = {}
    for v in votes:
        if not style_id:
            style_id = "unknown"
        w = weight_map.get(v.persona_id, default_w)
        individual_scores[v.persona_id] = clamp(v.final_score, 1.0, 10.0)
        weighted_total += individual_scores[v.persona_id] * w
        weight_sum += w
        all_scores.append(individual_scores[v.persona_id])
        if individual_scores[v.persona_id] < 4.0 or v.vetoed:
            oppose += 1
            if v.opposing_reason:
                oppose_reasons_weighted.append((v.opposing_reason, w))
            elif v.mom_reason:
                oppose_reasons_weighted.append((v.mom_reason, w))
        if individual_scores[v.persona_id] >= 7.0:
            support += 1
            if v.mom_reason:
                buy_reasons_weighted.append((v.mom_reason, w))
        scores = list(v.model_scores.values())
        if len(scores) >= 2:
            div = max(scores) - min(scores)
            if div >= 3.0:
                high_div.append(v.persona_id)

    n = len(votes) or 1
    mom_weighted = weighted_total / weight_sum if weight_sum > 0 else 5.0

    # ========== BrandConfig 决策结构分支 ==========
    final_weighted = mom_weighted
    child_veto_applied = False
    child_veto_reason = ""

    if brand_cfg is not None:
        ds = brand_cfg.decision_structure
        if ds.type == "single_layer":
            final_weighted = mom_weighted
        elif ds.type == "multi_layer":
            actual_age = target_age if target_age is not None else ds.default_target_age
            age_rule = _match_age_weight_rules(ds.age_weight_rules, actual_age)
            age_child_w = float(age_rule.get("child_weight", 0.30))

            if feats is not None and info is not None and age_child_w > 0:
                style_keywords = _extract_style_keywords(feats, info)
                # 尝试从persona列表推断孩子性别
                child_gender = ""
                if brand_cfg.personas:
                    genders: Counter[str] = Counter()
                    for p in brand_cfg.personas:
                        g = p.get("child_gender", "")
                        if g:
                            genders[g] += 1
                    if genders:
                        child_gender = genders.most_common(1)[0][0]
                vetoed, veto_reason = _check_child_veto(
                    brand_cfg.child_identity_axes, age_rule, style_keywords, child_gender
                )
                if vetoed:
                    final_weighted = mom_weighted * 0.70
                    child_veto_applied = True
                    child_veto_reason = veto_reason
                    oppose += 1
                    oppose_reasons_weighted.append((veto_reason, 0.30))
                else:
                    final_weighted = mom_weighted
            else:
                final_weighted = mom_weighted

    if child_veto_applied and child_veto_reason:
        if child_veto_reason not in [r[0] for r in oppose_reasons_weighted]:
            oppose_reasons_weighted.append((child_veto_reason, 0.30))

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
    """批量人设投票

    函数签名保持兼容：优先使用 brand_cfg（新架构），否则回退 cfg（旧架构）。
    """
    if brand_cfg is not None:
        persona_list = brand_cfg.personas
    elif cfg is not None:
        personas_cfg = cfg.personas
        persona_list = personas_cfg["personas"]
    else:
        raise ValueError("run_persona_voting: cfg 和 brand_cfg 不能同时为 None")

    votes: list[PersonaVote] = []
    pbar = tqdm(persona_list, desc=f"人设投票[{info.style_id}]", leave=False, disable=not progress)
    for p in pbar:
        pbar.set_postfix_str(p.get("id", p.get("persona_id", "?")))
        try:
            if cfg is not None:
                v = vote_persona(client, p, info, feats, cfg)
            else:
                v = PersonaVote(
                    persona_id=p.get("id", p.get("persona_id", "")),
                    persona_name=p.get("name", ""),
                    decision_mode=p.get("decision_mode", "mom_dominant"),
                    mom_score=5.0, child_score=5.0, final_score=5.0,
                    opposing_reason="无AppConfig兼容模式（新架构BrandConfig人设投票需LLM调用，此处占位5分）",
                )
            votes.append(v)
        except Exception as e:
            log.error("[%s] 人设%s 全部失败: %s", info.style_id, p.get("id", p.get("persona_id", "?")), e)
            votes.append(PersonaVote(
                persona_id=p.get("id", p.get("persona_id", "?")),
                persona_name=p.get("name", p.get("persona_id", "?")),
                decision_mode=p.get("decision_mode", "mom_dominant"),
                mom_score=5.0, child_score=5.0, final_score=5.0,
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
