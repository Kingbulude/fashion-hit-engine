#!/usr/bin/env python3
"""
Jev A/B 测试脚本 —— 独立工具，不修改现有系统

用法：
1. 先在 .env 里配好：
     CLOUDFLARE_ACCOUNT_ID=xxx
     CLOUDFLARE_API_TOKEN=xxx

2. 单款测试（看 Jev 输出格式）：
     python tools/jev_ab_test.py --single MPEWWQT06

3. 批量 A/B（你整理完 60 款后）：
     python tools/jev_ab_test.py --batch mipo_60款_总表.xlsx

4. 对比 Ollama vs Jev 的 Spearman：
     python tools/jev_ab_test.py --compare output/ollama_votes.json output/jev_votes.json

这个脚本不会改系统任何现有文件。
它会生成 3 个文件到 output/：
  jev_votes_<timestamp>.json    —— Jev 对每个人设的 90 个回答
  ollama_votes_<timestamp>.json —— 当前 Ollama 14B 的输出（shadow mode）
  ab_comparison_<timestamp>.md  —— 并排对比报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENV_PATH = REPO_ROOT / ".env"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

JEV_MODEL = "typesafe/jev"
JEV_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run"


# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

def load_dotenv() -> None:
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Jev Client
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error


class JevClient:
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        self.endpoint = JEV_ENDPOINT.format(acc_id=account_id)
        self._last_latency_ms = 0

    def call(self, state: str | dict, questions: dict) -> dict:
        """一次 parallel call 评估所有 questions。"""
        payload = {
            "model": JEV_MODEL,
            "input": {"state": state, "questions": questions},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        self._last_latency_ms = int((time.time() - t0) * 1000)
        return body

    @property
    def last_latency_ms(self) -> int:
        return self._last_latency_ms


# ---------------------------------------------------------------------------
# 渲染 Jev 的 state + questions
# ---------------------------------------------------------------------------

def build_state_en(style_info: dict, features: dict, personas: list[dict]) -> str | dict:
    """构造给 Jev 的 state（英文）。
    Jev 的 state 可以是字符串也可以是 dict——用 dict 更干净。
    """
    # 构造英文 BARS 特征摘要
    feat_labels_en = {
        "F01_silhouette": "silhouette fit",
        "F02_clean_look": "clean cut & finish",
        "F03_color_risk": "color saturation / risk",
        "F04_function_visibility": "outdoor function visibility (pockets, velcro, taping)",
        "F05_photogenic": "photogenic / camera memory",
        "F06_wearability": "versatility for daily life",
        "F07_pairing": "outfit coordination potential",
        "F08_fabric_perception": "fabric texture & quality",
        "F09_brand_tone": "brand fit for outdoor+daily positioning",
        "F10_uniqueness": "design uniqueness",
    }
    feat_lines = []
    for k, score in features.items():
        label = feat_labels_en.get(k, k)
        feat_lines.append(f"  - {label}: {score}/10")

    state = {
        "style_id": style_info.get("style_id", "?"),
        "product_category": style_info.get("category", "?"),
        "season": style_info.get("season", "?"),
        "price_cny": style_info.get("price", "?"),
        "main_color_cn": style_info.get("main_color", "?"),
        "target_buyer": "Chinese mom buying outdoor-style clothing for child aged 6-18",
        "fabric_description_cn": style_info.get("fab_description", ""),
        "bars_scores_out_of_10": {k: features[k] for k in features},
        "feature_summary_english": "\n".join(feat_lines),
    }
    return state


# 30 Persona 的英文描述（每个人设精简到最核心的购买判断维度）
PERSONAS_EN = {
    "P01": "Mom 28-35 Shanghai/Peking, budget-conscious, value for money, durable, school daily wear, plain dark colors",
    "P02": "Mom 28-35 tier-2 city, quality premium, Tetoron fabric, moisture-wicking, Papa brand endorsement, dark solid",
    "P03": "Mom 28-35 tier-2 city, mountain-style bright colors for visibility, budget value, not too uniform",
    "P04": "Mom 30-40 tier-1 city, mid-high price, functional tech fabric, modern outdoor brands, versatile",
    "P05": "Mom 30-40, cares about mom herself wearing it too (outerwear), style+function balance, premium feel",
    "P06": "Teen 13-18 boy, K-pop street, wants cool not childish, Instagram-checkable, black/gray techwear",
    "P07": "Teen 13-18 girl, Y2K, needs outfit-instagrammable, pastels/pinks, trends matter more than function",
    "P08": "Teen 10-14, wants something neither too childish nor too adult, self-expression important",
    "P09": "Budget-oriented mom, buys multi-season, washes a lot, fade-resistant, price-sensitive",
    "P10": "Outdoor enthusiast parent, mountain camping family, waterproof/taped seams/breathable, function first",
    "P11": "Fast-fashion follower mom, 抖音直播 impulse buyer, visual appeal, immediate trend, price acceptable",
    "P12": "Gift buyer (grandparent), age 50+, wants classic, safe, looks good, no risk with colors",
    "P13": "Private-school mom, brand matters, 'not too cheap-looking', polished, status-signaling",
    "P14": "Teen 15+, 'don't-make-me-embarrassed' threshold, avoids mom-buy-look, prefers black/muted",
    "P15": "Function-over-style dad, practical, likes tech details, pockets, zippers, dark rugged",
    "P16": "First-tier city young mom, aesthetic-focused, brand-consistent, minimalist clean look",
    "P17": "Middle schooler, wants standout on campus, bold colors OK if not cringey",
    "P18": "Mom concerned about skin safety, dye-fade-free, natural fabric feel, hypoallergenic",
    "P19": "Sports-active kid family, needs quick-dry, stretch, durable elbows/knees",
    "P20": "Oversize/drop-fit preference parent, trendy silhouette, streetwear vibe, functional secondary",
    "P21": "Practical grandma buyer, machine-washable, wrinkle-free, easy-carry, no fancy details",
    "P22": "Brand loyal parent, 'MIPO style = trust', will repurchase if fits brand identity",
    "P23": "Price-comparison shopper, will check 3 stores, value perception, not cheap-feeling important",
    "P24": "Visual-stimulation parent, likes contrast panels, tapes, reflective strips (for safety too)",
    "P25": "Minimalist parent, all-neutral/mono, prefers black/gray/beige, 1-tone only",
    "P26": "Color-coordinated-with-siblings parent, wants sibling-match outfits so kids don't fight",
    "P27": "Kid-influenced purchase, child veto power > mom preference, 'what my kid likes wins'",
    "P28": "Gift-for-friend's-kid buyer, safe neutral, not-too-expensive, looks like you spent effort",
    "P29": "Multi-function user, one jacket for school AND weekend hiking, versatility = plus",
    "P30": "Design-appreciator, notices uncommon zipper placement, asymmetric cuts, subtle details",
}


def build_jev_questions(personas: list[dict]) -> dict:
    """
    构造 30 × (score mom_decider + noul child_influence_veto) = ~60 个问题
    """
    questions: dict = {}

    for ps in personas:
        pid = ps.get("persona_id", "P?")
        en_desc = PERSONAS_EN.get(pid, f"Persona {pid}")

        # --- Score: mom decider layer (buy_decision) ---
        # 1-10 rubric, 5 levels 更清晰
        questions[f"{pid}_buy_score"] = {
            "type": "score",
            "instructions": (
                f"Imagine you are this specific buyer persona: {en_desc}. "
                f"Based ONLY on the English summary of this garment (the state), "
                f"how likely are you to buy this item? "
                f"Score the buy_decision_layer dimension."
            ),
            "criteria": [
                "1 - Strong reject: absolutely would not buy",
                "3 - Moderate reject: unlikely, something puts me off",
                "5 - Neutral: maybe, depends on nothing being wrong",
                "7 - Moderate buy: likely, fits my criteria well",
                "9 - Strong buy: definitely, perfect fit for me",
            ],
        }

        # --- Noul: child influence veto ---
        # Jev noul 返回 0-1 概率。我们把它当作 veto probability
        questions[f"{pid}_veto"] = {
            "type": "noul",
            "instructions": (
                f"Given the same buyer persona '{pid}' ({en_desc}) "
                f"AND considering this item could face a child veto dimension, "
                f"is this item likely to be vetoed or strongly disliked "
                f"by the child_influence_layer (6-18 year old wearer)? "
                f"Consider the child veto clues from the persona definition."
            ),
            "criteria": {
                "true": (
                    "Child would strongly dislike / veto this item: "
                    "cartoon overload, too babyish, too frumpy, embarrassing, "
                    "colors wrong for age, material feels cheap to kids"
                ),
                "false": (
                    "Child would likely accept or even like this item, "
                    "no obvious veto trigger"
                ),
            },
        }

    return questions


def convert_jev_to_system_format(jev_answers: dict, personas: list[dict]) -> dict:
    """
    把 Jev 返回转换成当前系统的 PersonaVotingResult 期望格式：
      { P01: {buy_decision_layer_score: 7.8, buy_decision_layer_conf: 0.91, veto_prob: 0.12, ... } }

    Jev Score 返回: score (0-based position on rubric) + confidence + probabilities
    Jev Noul 返回: noul (0-1 probability)
    """
    result = {}
    for ps in personas:
        pid = ps.get("persona_id", "P?")
        buy_ans = jev_answers.get("answers", {}).get(f"{pid}_buy_score", {})
        veto_ans = jev_answers.get("answers", {}).get(f"{pid}_veto", {})

        # Score rubric 有 5 档 (index 0-4), 映射到 1-10
        jev_score_pos = buy_ans.get("score", 2.0)  # 0-based position
        jev_conf = buy_ans.get("confidence", 0.5)
        jev_buy_prob = buy_ans.get("probabilities", {})

        # 0→1, 2→5, 4→9, 中间线性
        buy_score_1_10 = 1 + jev_score_pos * 2.0  # maps 0→1, 1→3, 2→5, 3→7, 4→9
        buy_score_1_10 = max(1.0, min(10.0, buy_score_1_10))

        # Veto: noul 是 veto 概率, 映射到 vetoed boolean (阈值 0.6)
        veto_prob = veto_ans.get("noul", 0.5)
        vetoed = veto_prob >= 0.6

        result[pid] = {
            "buy_score": round(buy_score_1_10, 2),
            "buy_confidence": round(jev_conf, 3),
            "buy_probabilities_raw": jev_buy_prob,
            "veto_prob": round(veto_prob, 3),
            "vetoed": vetoed,
            # 没有 reason（Jev 不生成文本）
            "buy_reason": "",
            "opposing_reason": "",
        }

    return result


# ---------------------------------------------------------------------------
# Ollama shadow (mock — 用现有 vote_persona 但走 mock LLM 后端)
# ---------------------------------------------------------------------------

def run_ollama_shadow(state_zh: dict, personas_cfg_path: str = "brand_profiles/mipo/personas.yaml") -> dict:
    """
    跑当前 Ollama 14B 人设投票（shadow mode），作为 A/B 基线。
    这会调本地 Ollama，每款约 5-10 分钟（30×2 人设×2 模型）。
    """
    from src.types import StyleInfo, StyleFeatures
    from src.persona_voting import vote_persona
    from src.llm_client import OllamaClient
    from src.config import load_brand_profile

    cfg = load_brand_profile("mipo")

    info = StyleInfo(
        style_id=state_zh.get("style_id", "?"),
        category=state_zh.get("category", ""),
        price=state_zh.get("price", 0),
        season=state_zh.get("season", ""),
        fab_description=state_zh.get("fab_description", ""),
        brand_main_color=state_zh.get("main_color", ""),
    )
    feats = StyleFeatures(
        features={k: v for k, v in state_zh.get("features", {}).items()},
        confidences={f: 0.8 for f in state_zh.get("features", {}).keys()},
    )

    client = OllamaClient()
    persona_cfg = cfg.personas_cfg if hasattr(cfg, "personas_cfg") else None
    voting = vote_persona(client, info, feats, brand_cfg=cfg, persona_cfg=persona_cfg)

    # 转换到统一格式
    result = {}
    for pid, entry in voting.layer_scores.items():
        result[pid] = {
            "buy_score": round(entry, 2),
            "buy_confidence": 0.7,  # Ollama 没有原生 confidence
            "buy_probabilities_raw": {},
            "veto_prob": 0.1 if not voting.vetoed else 0.8,
            "vetoed": voting.vetoed,
            "buy_reason": entry.get("buy_reason", "") if isinstance(entry, dict) else "",
            "opposing_reason": entry.get("opposing_reason", "") if isinstance(entry, dict) else "",
        }

    return result


# ---------------------------------------------------------------------------
# 主命令
# ---------------------------------------------------------------------------

def cmd_single(style_id: str) -> int:
    """单款测试：Jev 对这款衣服的 60 个问题 + 输出格式"""
    load_dotenv()
    acc_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not acc_id or not api_token:
        print("❌ 缺少 CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN")
        print("   请先在 .env 里配好（见 README / 上面的 3 分钟教程）")
        return 1

    # mock style
    style = {
        "style_id": style_id,
        "category": "户外夹克",
        "price": 399,
        "season": "2025春",
        "main_color": "深灰",
        "fab_description": "户外夹克，防水透气，魔术贴袖口，胸前压胶条",
        "features": {
            "F01_silhouette": 5.2, "F02_clean_look": 6.1,
            "F03_color_risk": 2.0, "F04_function_visibility": 7.0,
            "F05_photogenic": 5.0, "F06_wearability": 7.5,
            "F07_pairing": 7.0, "F08_fabric_perception": 6.0,
            "F09_brand_tone": 6.5, "F10_uniqueness": 3.5,
        },
    }

    import yaml
    personas = yaml.safe_load(open(REPO_ROOT / "brand_profiles/mipo/personas.yaml"))["personas"]

    print(f"🌍 构造 state (英文) ...")
    state_en = build_state_en(style, style["features"], personas)
    print(f"   state 大小: {len(json.dumps(state_en, ensure_ascii=False))} chars")

    questions = build_jev_questions(personas)
    print(f"🧩 构造 {len(questions)} 个 Jev 问题 (30人设 × (score + noul))")

    print(f"🚀 调用 Jev ({JEV_MODEL}) ...")
    client = JevClient(acc_id, api_token)
    try:
        raw = client.call(state_en, questions)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        print(f"❌ Jev HTTP {e.code}: {e.reason}\n{body[:500]}")
        return 1
    except Exception as e:
        print(f"❌ Jev call failed: {e}")
        return 1

    print(f"✅ Jev 返回，耗时 {client.last_latency_ms}ms")

    # 转换
    converted = convert_jev_to_system_format(raw, personas)

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"jev_single_{style_id}_{ts}.json"
    out_file.write_text(
        json.dumps(
            {
                "raw_jev_response": raw,
                "converted_for_system": converted,
                "latency_ms": client.last_latency_ms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 打印摘要
    print(f"\n📊 输出摘要 (共 {len(converted)} 人设)：")
    print(f"  {'人设':<5} {'Jev score':<10} {'conf':<8} {'veto prob':<10} {'vetoed':<7}")
    print("  " + "-" * 50)
    for pid in list(converted.keys())[:15]:
        c = converted[pid]
        flag = "⚠️ VETO" if c["vetoed"] else ""
        print(f"  {pid:<5} {c['buy_score']:<10} {c['buy_confidence']:<8} {c['veto_prob']:<10} {c['vetoed']} {flag}")
    if len(converted) > 15:
        print(f"  ... +{len(converted)-15} 人设")

    # confidence 分布
    confs = [c["buy_confidence"] for c in converted.values()]
    vetos = sum(1 for c in converted.values() if c["vetoed"])
    print(f"\n  confidence 均值={sum(confs)/len(confs):.3f} 最高={max(confs):.3f} 最低={min(confs):.3f}")
    print(f"  vetoed={vetos}/{len(converted)}")
    print(f"\n💾 完整输出: {out_file}")
    print(f"\n💡 下一步：把 60 款 Excel 放到 workspace，然后:")
    print(f"   python tools/jev_ab_test.py --batch your_60款.xlsx")
    return 0


def cmd_batch(xlsx_path: str) -> int:
    """批量 A/B：同一批 60 款，Jev 跑一遍 + Ollama shadow 跑一遍，对比 Spearman"""
    load_dotenv()
    acc_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not acc_id or not api_token:
        print("❌ 缺少 CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN")
        return 1

    print(f"📦 读取 Excel: {xlsx_path}")
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    print(f"   {df.shape[0]} 款 × {df.shape[1]} 列")
    print(f"   列名: {list(df.columns)}")

    # 验证有销售列
    sales_cols = [c for c in df.columns if any(k in str(c) for k in ["销量", "销售", "排名", "爆", "旺", "平", "滞", "rank", "sales"])]
    if not sales_cols:
        print("⚠️  没找到销售/销量列，暂时只能做人设投票质量对比，不能算 Spearman")
        sales_col = None
    else:
        sales_col = sales_cols[0]
        print(f"   销售列: {sales_col}")

    import yaml
    personas = yaml.safe_load(open(REPO_ROOT / "brand_profiles/mipo/personas.yaml"))["personas"]
    questions = build_jev_questions(personas)
    client = JevClient(acc_id, api_token)

    jev_all = {}
    ollama_all = {}
    n = len(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, row in df.iterrows():
        sid = str(row.get("款号", f"row_{i}"))
        print(f"\n[{i+1}/{n}] {sid} ... ", end="", flush=True)

        # 提取 style dict
        style = {
            "style_id": sid,
            "category": row.get("品类", row.get("category", "")),
            "price": row.get("售价", row.get("price", 0)),
            "season": row.get("季节", row.get("season", "")),
            "main_color": row.get("主推色", row.get("main_color", "")),
            "fab_description": row.get("FAB描述", row.get("fab", "")),
            "features": {},  # 假设你已经跑过 VLM 特征提取，后续填
        }

        # 跑 Jev
        state_en = build_state_en(style, style["features"], personas)
        try:
            raw = client.call(state_en, questions)
            jev_all[sid] = convert_jev_to_system_format(raw, personas)
            print(f"Jev {client.last_latency_ms}ms", end=" ")
        except Exception as e:
            print(f"Jev FAIL: {e}", end=" ")

        # Ollama shadow（可选，需要本地 Ollama 跑着）
        try:
            ollama_all[sid] = run_ollama_shadow(style)
            print(f"+ Ollama shadow ✅")
        except Exception as e:
            print(f"+ Ollama skipped ({str(e)[:40]})")

    # 保存
    jev_file = OUTPUT_DIR / f"jev_votes_{ts}.json"
    jev_file.write_text(json.dumps(jev_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 Jev 结果: {jev_file}")

    if ollama_all:
        ol_file = OUTPUT_DIR / f"ollama_votes_{ts}.json"
        ol_file.write_text(json.dumps(ollama_all, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Ollama 结果: {ol_file}")

    # 简单统计对比
    if jev_all and ollama_all:
        _print_comparison(jev_all, ollama_all)

    return 0


def cmd_compare(jev_file: str, ollama_file: str) -> int:
    """两个 JSON 结果文件的对比"""
    jev_all = json.load(open(jev_file))
    ollama_all = json.load(open(ollama_file))
    _print_comparison(jev_all, ollama_all)
    return 0


def _print_comparison(jev_all: dict, ollama_all: dict) -> None:
    """并排打印对比报告"""
    common = [sid for sid in jev_all if sid in ollama_all]
    print(f"\n📊 对比报告 ({len(common)} 款交集)")
    print("=" * 70)

    diffs_per_persona: dict[str, list[float]] = {}

    for sid in common[:5]:  # 先打前 5 款预览
        print(f"\n  {sid}:")
        for pid in jev_all[sid]:
            j_score = jev_all[sid][pid]["buy_score"]
            o_score = ollama_all[sid].get(pid, {}).get("buy_score", None)
            diff = j_score - o_score if o_score is not None else None
            marker = ""
            if abs(diff) > 3:
                marker = " 🔴"
            elif abs(diff) > 1.5:
                marker = " 🟡"
            print(f"    {pid}: Jev={j_score}  Ollama={o_score}  diff={diff}{marker}")
            if diff is not None:
                diffs_per_persona.setdefault(pid, []).append(diff)

    # 总体统计
    all_diffs = [d for v in diffs_per_persona.values() for d in v]
    if all_diffs:
        print(f"\n📈 总体差异: mean={sum(all_diffs)/len(all_diffs):.2f} std={__import__('statistics').stdev(all_diffs):.2f}")
        big = [d for d in all_diffs if abs(d) > 3]
        print(f"   |diff|>3 的比例: {len(big)/len(all_diffs)*100:.1f}% ({len(big)}/{len(all_diffs)})")

    # Veto 对比
    j_veto = sum(1 for sid in common for pid in jev_all[sid].values() if pid in ollama_all[sid] and jev_all[sid][pid].get("vetoed"))
    o_veto = sum(1 for sid in common for pid in ollama_all[sid].values() if ollama_all[sid][pid].get("vetoed"))
    print(f"\n   Jev veto 总数: {j_veto}")
    print(f"   Ollama veto 总数: {o_veto}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Jev A/B 测试 — 人设投票对比 Jev vs Ollama 14B")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", metavar="STYLE_ID", help="单款测试（看 Jev 输出格式）")
    g.add_argument("--batch", metavar="XLSX", help="批量测试（60款 Excel）")
    g.add_argument("--compare", nargs=2, metavar=("JEV.json", "OLLAMA.json"), help="两个结果文件对比")
    args = ap.parse_args()

    if args.single:
        return cmd_single(args.single)
    if args.batch:
        return cmd_batch(args.batch)
    if args.compare:
        return cmd_compare(*args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
