"""
fashion-hit-engine · 服装爆款预测通用引擎
Streamlit Web应用入口
运行：streamlit run app.py

通用架构：
- 品牌适配包 brand_profiles/<id>/（5YAML + calibrated/）→ 选品牌即可切换品类
- 三大引擎（人设投票/双渠道评分/价格价值） → CORE引擎100%跨品牌复用
- 3Loop核心优化内核（VLM校准/人设分布拟合/集成权重调优）+ 残差分离
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ========== 让直接 streamlit run app.py 能 import src/ 模块 ==========
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ========== 🔍 让诊断日志可见（Streamlit 默认 WARNING 级别吃掉 log.info）==========
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)

from src.config import load_config, list_available_brands, load_brand_profile
from src.grading import assign_relative_grades
from src.llm_client import is_fatal_quota_error
from src.pipeline import PredictionPipeline
from src.report import render_single_report_markdown
from src.types import (
    StyleInfo, FullPrediction, GradeResult, BrandConfig,
    SALES_QTY_COL_ALIASES, SELL_THROUGH_COL_ALIASES, MANUAL_GRADE_COL_ALIASES,
    MAIN_PUSH_COL_ALIASES, LIVE_STREAM_COL_ALIASES, STYLE_ID_COL_ALIASES,
    find_aliased_column, parse_sales_value, safe_float,
)
from scripts.rename_images import batch_rename_from_folders  # 图片重命名脚本


# ========== 工具函数 ==========
def _count_calibration_rounds(brand_cfg: BrandConfig) -> int:
    cal_dir = Path(getattr(brand_cfg, "calibrated_dir",
                           ROOT / "brand_profiles" / brand_cfg.brand_id / "calibrated"))
    if not cal_dir.exists():
        return 0
    return len([f for f in cal_dir.glob("*.yaml") if f.is_file() and not f.name.startswith(".")])


# ========== 页面切换 & 全局品牌选择 ==========
PAGES = ["📤 上传批次", "📋 批次总表", "🔍 单款详情报告", "📊 回测校准"]
st.set_page_config(page_title="fashion-hit-engine · 服装爆款预测通用引擎", layout="wide")

# --- 侧边栏顶部：品牌选择 ---
st.sidebar.title("🏷️ 选择品牌")
available_brands = list_available_brands()
if not available_brands:
    st.sidebar.error("❌ 没有可用的品牌适配包，请检查 brand_profiles/ 目录")
    available_brands = ["mipo"]
if "brand_id" not in st.session_state:
    st.session_state.brand_id = available_brands[0] if available_brands else "mipo"

brand_id = st.sidebar.selectbox(
    "使用哪个品牌的适配包？",
    options=available_brands,
    index=available_brands.index(st.session_state.brand_id)
    if st.session_state.brand_id in available_brands else 0,
    help="每个品牌独立维护：30人设+BARS量表+品类价格带+S/A/P阈值+3Loop校准产物"
)
if brand_id != st.session_state.brand_id:
    st.session_state.brand_id = brand_id
    # 切换品牌 → 清空批次缓存（否则旧品牌的preds结构在新品牌下解释错误）
    for k in ("preds", "df_input", "style_to_images", "progress_info",
              "style_infos", "image_paths_map", "batch_name",
              "selected_style_id"):
        st.session_state.pop(k, None)
    st.rerun()

# 加载 BrandConfig（贯穿整个会话共享）
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load_brand(_bid: str) -> BrandConfig:
    return load_brand_profile(_bid)

brand_cfg: BrandConfig = _cached_load_brand(brand_id)
n_calibration: int = _count_calibration_rounds(brand_cfg)

# --- 侧边栏：品牌信息摘要 ---
with st.sidebar.expander(f"📌 {brand_cfg.brand_name}", expanded=True):
    st.write(f"**品牌ID：** `{brand_cfg.brand_id}`")
    st.write(f"**决策结构：** {brand_cfg.decision_structure.type}")
    if brand_cfg.decision_structure.type == "multi_layer":
        layer_names = "、".join(l.name for l in brand_cfg.decision_structure.layers)
        st.write(f"**层数：** {len(brand_cfg.decision_structure.layers)} 层（{layer_names}）")
    st.write(f"**品类数：** {len(brand_cfg.category_registry.get('categories', []))}")
    st.write(f"**人设数：** {len(brand_cfg.personas)} 个身份三轴线")
    st.write(f"**校准轮次：** {n_calibration or '0（冷启动）'}")
    try:
        from src.history_store import HistoryStore
        h = HistoryStore()
        summary = h.get_summary(brand_cfg.brand_id)
        st.write(f"**历史批次：** {summary['batches']} 批 / {summary['styles']} 款")
    except Exception:
        pass

st.sidebar.divider()

# --- 侧边栏：LLM 模式切换（智谱免费为默认） ---
llm_mode_label = st.sidebar.radio(
    "🤖 预测引擎",
    options=[
        "智谱 GLM（全云端·免费）",
        "全本地 Ollama（零成本·零限流·最快）",
        "混合模式（智谱 VLM + 本地文本）",
        "百炼 API（阿里云·付费）",
        "演示模式（mock）",
    ],
    help=(
        "智谱：GLM-4.6V-Flash 视觉 + GLM-4.7-Flash 文本，均永久免费，"
        "到 open.bigmodel.cn 注册即得 Key（无需信用卡）。"
        "全本地：VLM + 文本全走 Ollama，零成本零限流，但需要本机 Ollama + RTX 3070 8GB 以上。"
        "  比智谱快约 3 倍（20 款 ≈ 22 min vs 62 min）。"
        "混合：视觉本地 Ollama + 文本智谱云端。需要 Ollama + 智谱 Key。"
        "百炼：阿里云付费 API（每批约 2-6 元，免费额度已耗尽时慎选）。"
        "演示模式用伪随机数据跑通全链路，仅用于本地调试。"
    ),
    index=0,
)

# --- 侧边栏：API Key 输入（按所选后端显示对应输入框） ---
st.sidebar.subheader("🔑 API Key")
_zhipu_key_input = st.sidebar.text_input(
    "智谱 API Key（免费）",
    type="password",
    value=st.session_state.get("zhipu_api_key", ""),
    help="从 https://open.bigmodel.cn/usercenter/apikeys 免费创建；仅保存在当前浏览器会话",
    key="zhipu_api_key_input",
)
if _zhipu_key_input:
    st.session_state.zhipu_api_key = _zhipu_key_input

_dashscope_key_input = ""
if llm_mode_label == "百炼 API（阿里云·付费）":
    _dashscope_key_input = st.sidebar.text_input(
        "百炼 API Key（付费）",
        type="password",
        value=st.session_state.get("dashscope_api_key", ""),
        help="从 https://bailian.console.aliyun.com 获取；按量计费",
        key="dashscope_api_key_input",
    )
    if _dashscope_key_input:
        st.session_state.dashscope_api_key = _dashscope_key_input

_zhipu_key_from_session = st.session_state.get("zhipu_api_key", "").strip()
_dashscope_key_from_session = st.session_state.get("dashscope_api_key", "").strip()

# --- 后端路由 + Key 校验（缺 Key 自动回退 mock） ---
if llm_mode_label.startswith("智谱 GLM"):
    _llm_backend = "zhipu"
    _key_for_backend = (
        _zhipu_key_from_session
        or os.getenv("ZHIPU_API_KEY", "").strip()
    )
    _key_missing_hint = (
        "你选择了「智谱 GLM（全云端·免费）」，但未检测到智谱 API Key，已自动回退到 mock 模式。"
        "请到 https://open.bigmodel.cn/usercenter/apikeys 免费创建 Key 后粘贴到上方输入框。"
    )
elif llm_mode_label.startswith("全本地 Ollama"):
    _llm_backend = "local"
    _key_for_backend = ""  # 本地 Ollama 不需要 API Key
    _key_missing_hint = "本模式不需要 API Key，但需要先安装 Ollama 并 pull 模型。"
elif llm_mode_label.startswith("混合模式"):
    _llm_backend = "hybrid"
    _key_for_backend = (
        _zhipu_key_from_session
        or os.getenv("ZHIPU_API_KEY", "").strip()
    )
    _key_missing_hint = (
        "你选择了「混合模式（智谱 VLM + 本地文本）」，但未检测到智谱 API Key（文本端需要）。"
        "请创建 Key 后粘贴。本地 Ollama VLM 不需要 Key。"
    )
elif llm_mode_label == "百炼 API（阿里云·付费）":
    _llm_backend = "dashscope"
    _key_for_backend = (
        _dashscope_key_from_session
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    _key_missing_hint = (
        "你选择了「百炼 API（付费）」，但未检测到百炼 API Key，已自动回退到 mock 模式。"
        "请在上方输入框粘贴 Key。"
    )
else:
    _llm_backend = "mock"
    _key_for_backend = ""

# 本地 Ollama 模式不要求 API Key，检查 Ollama 是否在线
_local_ollama_ok = True
if _llm_backend == "local":
    from src.llm_client import OllamaClient as _OC
    _local_ollama_ok, _ = _OC().health_check()
    if not _local_ollama_ok:
        st.sidebar.warning(
            "⚠️ Ollama 未启动（localhost:11434 连接被拒）。\n\n"
            "请先：\n"
            "1. 下载 Ollama: https://ollama.com/download\n"
            "2. 安装后在终端执行:\n"
            "   `ollama pull qwen2.5vl:7b`\n"
            "   `ollama pull qwen2.5:7b`\n"
            "3. 确认显存 ≥ 6.5GB（RTX 3070 8GB 可跑）\n\n"
            "已自动回退到 mock 模式。"
        )
        _llm_backend = "mock"

_api_key_fallback_reason: str | None = None
if _llm_backend not in ("mock", "local") and not _key_for_backend:
    _api_key_fallback_reason = _key_missing_hint
    st.sidebar.error(_api_key_fallback_reason)
    _llm_backend = "mock"
    _key_for_backend = ""

# 暴露给全会话使用（_api_key_for_backend 与所选后端匹配）
st.session_state.llm_backend = _llm_backend
st.session_state.api_key_fallback_reason = _api_key_fallback_reason
st.session_state.api_key_for_backend = _key_for_backend

# --- 侧边栏：测试连接（1 次真实调用，一键验证 Key + 网络 + 后端路由）---
if _llm_backend != "mock":
    if st.sidebar.button("🔌 测试连接", use_container_width=True,
                         help="hybrid 模式会同时检查本地 Ollama 和云端智谱"):
        from src.config import APIConfig as _AC
        from src.llm_client import ZhipuClient as _ZC, BailianClient as _BC, OllamaClient as _OC
        _hc_ok, _hc_msg = True, ""
        if _llm_backend == "hybrid":
            _hc_ok, _hc_msg = _OC().health_check()
        if _llm_backend != "hybrid" and not _key_for_backend:
            st.sidebar.error("❌ 请先粘贴 API Key")
        else:
            try:
                _t_cfg = _AC(max_retries=3, qpm_limit=10000)
                _parts_ok: list[str] = []
                _parts_err: list[str] = []

                # ① 本地 Ollama 健康检查（local / hybrid 都要）
                if _llm_backend in ("local", "hybrid"):
                    _hc_ok, _hc_msg = _OC().health_check()
                    if not _hc_ok:
                        _parts_err.append(f"🖥️ Ollama: {_hc_msg}（请先安装 Ollama + ollama pull qwen2.5vl:7b + qwen2.5:7b）")
                    else:
                        _parts_ok.append(f"🖥️ Ollama 本地正常 ({_hc_msg})")
                    # local 模式额外发一次真实推理确认模型能跑
                    if _llm_backend == "local" and _hc_ok:
                        try:
                            _l_client = _OC()
                            _t_resp = _l_client.generate_text("回复：OK", max_tokens=8)
                            if _t_resp.ok:
                                _parts_ok.append(f"🖥️ Ollama 文本推理 OK ({_t_resp.model})")
                            else:
                                _parts_err.append(f"🖥️ Ollama 推理失败: {_t_resp.error[:120]}")
                        except Exception as e:
                            _parts_err.append(f"🖥️ Ollama 推理异常: {e}")
                    if _llm_backend == "hybrid" and not _key_for_backend:
                        _parts_err.append("☁️ 智谱 API Key 未填（文本端需要）")

                # ② 智谱 / hybrid-智谱文本 测试
                if _llm_backend in ("zhipu", "hybrid") and _key_for_backend:
                    _t_cfg.zhipu_api_key = _key_for_backend
                    _z_client = _ZC(_t_cfg, api_key=_key_for_backend)
                    _t_resp = _z_client.generate_text(
                        "回复：OK", model="qwen-max", max_tokens=8, temperature=0.0,
                    )
                    if _t_resp.ok:
                        _parts_ok.append(f"☁️ 智谱文本 OK ({_t_resp.model})")
                    else:
                        _parts_err.append(f"☁️ 智谱文本失败: {_t_resp.error[:120]}")

                # ③ 百炼测试
                if _llm_backend == "dashscope":
                    _t_cfg.dashscope_api_key = _key_for_backend
                    _b_client = _BC(_t_cfg)
                    _t_resp = _b_client.generate_text(
                        "回复：OK", model="qwen-max", max_tokens=8, temperature=0.0,
                    )
                    if _t_resp.ok:
                        _parts_ok.append(f"☁️ 百炼 OK ({_t_resp.model})")
                    else:
                        _parts_err.append(f"☁️ 百炼失败: {_t_resp.error[:120]}")

                # 汇总
                if _parts_err:
                    st.sidebar.error("❌ 测试未通过：\n\n" + "\n".join(_parts_err))
                if _parts_ok:
                    st.sidebar.success("✅ 测试通过：\n\n" + "\n".join(_parts_ok))
            except Exception as _t_e:
                st.sidebar.error(f"❌ 测试异常：{_t_e}")

st.sidebar.title("🧭 导航")
page = st.sidebar.radio("", PAGES, index=0)

# 从 VERSION 文件动态读版本号 —— 用户一眼确认自己跑的是哪个版本
_VERSION_STR = "unknown"
try:
    _VERSION_STR = (ROOT / "VERSION").read_text().strip()
except Exception:
    pass
st.sidebar.caption(
    f"**当前版本：{_VERSION_STR}**  "
    f"[⬆️ 更新](https://github.com/Kingbulude/fashion-hit-engine/releases/latest)\n"
    "· 通用CORE引擎 10特征30人设\n"
    "· VLM特征提取 + 人设投票\n"
    "· 3Loop 越用越准"
)

# ========== 兼容：保持 cfg 变量接口（AppConfig 薄兼容）==========
cfg = load_config()  # deprecated薄包装，内部brand_id=mipo

# ========== 全局状态初始化 ==========
if "batch_name" not in st.session_state:
    st.session_state.batch_name = ""
if "preds" not in st.session_state:         # list[FullPrediction]
    st.session_state.preds = []
if "df_input" not in st.session_state:      # 原始输入Excel
    st.session_state.df_input = None
if "progress_info" not in st.session_state:  # 进度显示用
    st.session_state.progress_info = {"current": 0, "total": 0, "stage": "空闲", "failed": {}}
if "style_to_images" not in st.session_state:  # 款号 -> list[图片路径]
    st.session_state.style_to_images = {}


# ========== 面包屑组件（公共） ==========
def render_breadcrumb(*, suffix: str | None = None) -> None:
    parts = [f"🏷️ {brand_cfg.brand_name}"]
    if st.session_state.batch_name:
        parts.append(f"📦 {st.session_state.batch_name}")
    if suffix:
        parts.append(suffix)
    st.markdown(
        "<div style='font-size:13px;color:#6b7280;margin-bottom:8px;'>"
        + "  ›  ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ========== 公共函数 ==========
def guess_category_from_text(text: str) -> str:
    """版型描述模糊猜品类，给价格百分位兜底（优先从brand_cfg.category_registry匹配）"""
    if not isinstance(text, str):
        return brand_cfg.category_registry["categories"][0]["id"]  # 首个品类兜底
    t = text.lower()
    # 先跑别名映射
    aliases = brand_cfg.category_registry.get("category_aliases", {})
    # 关键词命中注册品类名
    cats = brand_cfg.category_registry.get("categories", [])
    for c in cats:
        if c["name"] in t:
            return c["id"]
    for alias, cid in aliases.items():
        if alias in t:
            return cid
    # 否则启发式关键词匹配
    heuristics = [
        (["t恤", "t-shirt", "tee"], "T恤"),
        (["短裤"], "短裤"),
        (["长裤", "裤"], "长裤"),
        (["羽绒"], "羽绒服"),
        (["防晒"], "防晒衣"),
        (["衬衫"], "衬衫"),
        (["卫衣"], "卫衣"),
        (["马甲"], "马甲"),
        (["背心"], "背心"),
        (["夹克", "外套"], "夹克外套"),
    ]
    for kws, fallback in heuristics:
        if any(k in t for k in kws):
            # 再查fallback是不是存在品牌品类里，存在→id不存在→第一个
            for c in cats:
                if c["name"] == fallback:
                    return c["id"]
            break
    return cats[0]["id"] if cats else "外套"


def validate_inputs(
    df: pd.DataFrame,
    image_style_ids: set[str],
    style_to_images: dict[str, list[Path]],
) -> list[str]:
    """校验Excel必需列 + 款号图片文件夹匹配，返回警告列表"""
    warnings: list[str] = []
    required_cols = ["款式编号", "面料成分", "版型/设计描述", "售价"]
    for col in required_cols:
        if col not in df.columns:
            warnings.append(f"❌ Excel缺少必需列：{col}")
    if warnings:
        return warnings
    style_ids = df["款式编号"].dropna().astype(str).tolist()
    missing_folders = [s for s in style_ids if s not in image_style_ids]
    if missing_folders:
        warnings.append(
            f"⚠️ {len(missing_folders)}个款没有找到图片文件夹：{missing_folders[:5]}"
            + ("…" if len(missing_folders) > 5 else "")
        )
    empty_folders = [s for s in style_ids if s in image_style_ids
                     and len(style_to_images.get(s, [])) == 0]
    if empty_folders:
        warnings.append(f"⚠️ 图片文件夹是空的：{empty_folders[:3]}")
    return warnings


def unzip_to_temp_dir(
    zip_bytes: bytes, dest_dir: Path,
    known_style_ids: set[str] | None = None,
) -> dict[str, list[Path]]:
    """解压图片zip，返回 款号->[图片路径列表]（按修改时间排序）
    
    支持三种zip结构：
      1. 子文件夹结构：S01/img1.jpg, S02/img1.jpg  ← 推荐
      2. 扁平结构：S01_front.jpg, S02_back.jpg  ← 自动从文件名提取
      3. 混合结构
    known_style_ids 可用于更精确地从文件名中匹配款号
    """
    import re as _re
    dest_dir.mkdir(parents=True, exist_ok=True)
    style_to_images: dict[str, list[Path]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)

    def _extract_style_id_from_filename(name: str) -> str:
        """从文件名中尝试提取款号"""
        stem = Path(name).stem  # 去掉扩展名
        # 1. 优先匹配已知款号
        if known_style_ids:
            for sid in sorted(known_style_ids, key=len, reverse=True):
                if sid.lower() in stem.lower():
                    return sid
        # 2. 尝试 字母+数字 模式（如 S01, K001, JACKET_001）
        m = _re.match(r"^([A-Za-z]*\d+[A-Za-z]*)", stem)
        if m:
            return m.group(1)
        # 3. 尝试下划线/空格分隔的第一段
        for sep in ["_", "-", " ", "."]:
            parts = stem.split(sep)
            if parts and _re.match(r"^[A-Za-z]*\d+", parts[0]):
                return parts[0]
        return ""

    for p in sorted(dest_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue

        # 方式1：从父文件夹名取款号
        style_id = ""
        if p.parent != dest_dir and p.parent.name != dest_dir.name:
            candidate = p.parent.name
            # 如果父文件夹名在 known_style_ids 里，直接用
            if known_style_ids and candidate in known_style_ids:
                style_id = candidate
            elif not known_style_ids:
                style_id = candidate  # 无已知列表时，信任文件夹名

        # 方式2：从文件名提取款号（扁平结构 fallback）
        if not style_id:
            style_id = _extract_style_id_from_filename(p.name)

        if not style_id:
            continue
        style_to_images.setdefault(style_id, []).append(p)

    for sid, paths in style_to_images.items():
        paths.sort(key=lambda x: x.stat().st_mtime)
    return style_to_images


# ============================================================
# 页面 1：📤 上传批次
# ============================================================
def render_page_upload():
    render_breadcrumb(suffix="📤 上传批次")
    st.title("📤 上传评估批次")

    st.markdown("#### ① 填批次名")
    batch_name = st.text_input(
        "批次名（必填，方便后续区分）", value=st.session_state.batch_name or "2026春第一批",
        help="建议格式：年份+季节+第N批，如2026春第一批"
    )

    st.markdown("#### ② 上传款式信息表 Excel / CSV")
    st.caption(
        "列参考：款式编号、面料成分、版型/设计描述、售价 必填；"
        "建议加一列「品类」；可最后追加「真实销售结果」列供后续回测。"
    )
    xlsx_file = st.file_uploader("拖放或选择 .xlsx / .csv", type=["xlsx", "csv"])

    st.markdown("#### ③ 上传图片包（两种方式二选一）")
    mode = st.radio(
        "图片来源",
        ["📁 本机路径（推荐：直接填文件夹绝对路径）", "📦 上传 ZIP 压缩包"],
        horizontal=True,
    )
    style_to_images: dict[str, list[Path]] = {}
    style_ids_from_images: set[str] = set()
    image_warnings: list[str] = []

    if mode.startswith("📁"):
        folder_path = st.text_input(
            "图片文件夹绝对路径（例如：D:\\2026春第一批_images）",
            help="里面每个子文件夹名是款号，放对应的图片。图名不需要改。"
        )
        if folder_path:
            fp = Path(folder_path)
            if not fp.exists() or not fp.is_dir():
                image_warnings.append("❌ 路径不存在或不是文件夹")
            else:
                st.info("🔄 正在扫描并归类图片（5秒完成，不修改原文件）…")
                renamed_root = fp.parent / f"{fp.name}_renamed"
                try:
                    summary = batch_rename_from_folders(fp, renamed_root, in_place=False)
                    st.success(
                        f"✅ 扫描完成：{summary.get('total_styles', 0)}个款，"
                        f"{summary.get('total_images', 0)}张图，"
                        f"归类率{summary.get('match_rate', 0):.0%}。"
                        f"重命名后文件在：{renamed_root}"
                    )
                    for sub in sorted(renamed_root.iterdir()):
                        if not sub.is_dir():
                            continue
                        imgs = [p for p in sub.iterdir()
                                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
                        imgs.sort(key=lambda x: x.name)
                        if imgs:
                            style_to_images[sub.name] = imgs
                    style_ids_from_images = set(style_to_images.keys())
                except Exception as e:
                    image_warnings.append(f"❌ 图片扫描失败：{e}")

    else:  # ZIP
        zip_file = st.file_uploader("拖放或选择 .zip", type=["zip"])
        if zip_file is not None:
            tmp_dest = ROOT / "output" / "_tmp_uploads" / f"{int(time.time())}"
            st.info("🔄 正在解压并扫描图片…")
            try:
                style_to_images = unzip_to_temp_dir(zip_file.getvalue(), tmp_dest)
                style_ids_from_images = set(style_to_images.keys())
                st.success(f"✅ 解压完成：{len(style_ids_from_images)}个款，"
                           f"{sum(len(v) for v in style_to_images.values())}张图")
            except Exception as e:
                image_warnings.append(f"❌ 解压失败：{e}")

    for w in image_warnings:
        st.warning(w)

    df = None
    if xlsx_file is not None:
        try:
            if xlsx_file.name.endswith(".csv"):
                df = pd.read_csv(xlsx_file)
            else:
                df = pd.read_excel(xlsx_file)
            st.session_state.df_input = df
            with st.expander("👀 预览前5行", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Excel读取失败：{e}")

    # 真实模式必须配 API Key，否则禁止开始（避免用户以为在调真实 LLM）
    _llm_backend = st.session_state.get("llm_backend", "mock")
    _fallback_reason = st.session_state.get("api_key_fallback_reason")

    can_start = (
        bool(batch_name) and df is not None and df.shape[0] > 0
        and bool(style_ids_from_images) and not image_warnings
        and (_llm_backend != "dashscope" or _fallback_reason is None)
    )

    if _fallback_reason:
        st.error(f"🚨 {_fallback_reason}")

    if df is not None and style_ids_from_images:
        warns = validate_inputs(df, style_ids_from_images, style_to_images)
        for w in warns:
            if w.startswith("❌"):
                st.error(w)
                can_start = False
            else:
                st.warning(w)

    st.divider()

    # --- 盲测组设置（对照机制：见 src/blind_set.py / CONTEXT.md）---
    with st.expander("🔬 盲测组设置（可选，建议开启）", expanded=False):
        st.caption(
            "每批次按品类分层随机抽取部分款为「盲测款」：AI 照常预测并落盘，"
            "但报告/导出对运营打码，内审决策不受 AI 干扰。"
            "销量回填后自动解锁对照：验证 AI 纯净准确率 + AI vs 人工分歧裁决。"
        )
        blind_enabled = st.toggle("启用盲测组", value=True, key="blind_enabled")
        blind_ratio = st.slider(
            "盲测比例", min_value=0.05, max_value=0.30, value=0.15,
            step=0.05, disabled=not blind_enabled, key="blind_ratio",
            help="业界对照试验惯例 10-20%，比例越大对照越准，但内审可参考的 AI 结论越少",
        )

    # 提交前成本预估：让「额度去哪了」在下单前就可见（真实模式才显示）
    if can_start and _llm_backend != "mock" and df is not None:
        _per_style = 1 + len(brand_cfg.personas) * 2  # 1 次 VLM 特征 + 人设数 × 2 模型
        if _llm_backend == "zhipu":
            st.caption(
                f"🆓 预估本批 API 调用：{len(df)} 款 × {_per_style} 次/款 ≈ "
                f"**{len(df) * _per_style:,} 次**（1 次特征提取 + "
                f"{len(brand_cfg.personas)} 人设 × 2 模型投票，每款）。"
                f"智谱免费模型（GLM-4.6V-Flash / GLM-4.7-Flash）**0 元**，不消耗额度。"
            )
        else:
            st.caption(
                f"💰 预估本批 API 调用：{len(df)} 款 × {_per_style} 次/款 ≈ "
                f"**{len(df) * _per_style:,} 次**（1 次特征提取 + "
                f"{len(brand_cfg.personas)} 人设 × 2 模型投票，每款）。"
                f"百万 token 级消耗，请注意额度余额；额度耗尽会自动中止并提示。"
            )

    col1, col2, _ = st.columns([2, 2, 4])
    with col1:
        if not can_start:
            _help = "先填批次名、上传Excel和图片，并解决上面的警告"
            if _fallback_reason:
                _help = "你选择了真实 LLM 模式，请先在侧边栏粘贴对应平台的 API Key"
            st.button("🚀 开始评估", disabled=True, use_container_width=True, help=_help)
        else:
            if st.button("🚀 开始评估", type="primary", use_container_width=True):
                st.session_state.batch_name = batch_name
                st.session_state.style_to_images = style_to_images
                style_infos: list[StyleInfo] = []
                image_paths_map: dict[str, list[str]] = {}

                # --- 用别名匹配找到可选的"真实销量/人工分级/售罄率/营销投放"列 ---
                sales_col = find_aliased_column(list(df.columns), SALES_QTY_COL_ALIASES)
                grade_col = find_aliased_column(list(df.columns), MANUAL_GRADE_COL_ALIASES)
                st_col = find_aliased_column(list(df.columns), SELL_THROUGH_COL_ALIASES)
                push_col = find_aliased_column(list(df.columns), MAIN_PUSH_COL_ALIASES)
                live_col = find_aliased_column(list(df.columns), LIVE_STREAM_COL_ALIASES)
                sid_col = find_aliased_column(list(df.columns), STYLE_ID_COL_ALIASES) or "款式编号"
                if sid_col not in df.columns:
                    st.error(f"❌ 未找到款号列（支持：{' / '.join(STYLE_ID_COL_ALIASES)}），无法解析批次")
                    st.stop()
                if sales_col:
                    st.success(f"✅ 检测到销量列「{sales_col}」，将自动带入回测校准")
                else:
                    st.warning(
                        f"⚠️ 未检测到销量列（支持别名：{' / '.join(SALES_QTY_COL_ALIASES[:4])}…）。"
                        f"本批次只能跑预测，无法回测校准。"
                    )
                if grade_col:
                    st.success(f"✅ 检测到人工分级列「{grade_col}」（仅作对照展示，不进校准）")
                if st_col:
                    st.success(f"✅ 检测到售罄率列「{st_col}」")
                if push_col:
                    st.success(f"✅ 检测到实际主推列「{push_col}」，将用于残差分离（款式 vs 投放）")
                if live_col:
                    st.success(f"✅ 检测到实际直播列「{live_col}」，将用于残差分离（款式 vs 投放）")
                if not push_col and not live_col:
                    st.info(
                        "ℹ️ 未检测到营销投放列（是否主推/是否直播重点）。"
                        "校准将无法区分「款式好」和「被推爆」，建议后续批次补上。"
                    )

                from src.types import _truthy_excel_value as _tv
                for _, row in df.iterrows():
                    sid = str(row[sid_col])
                    cat_raw = row.get("品类")
                    if isinstance(cat_raw, str) and cat_raw.strip() and str(cat_raw) != "nan":
                        from src.data_io import resolve_category
                        cat = resolve_category(cat_raw.strip(), brand_cfg=brand_cfg)
                    else:
                        cat = guess_category_from_text(str(row.get("版型/设计描述", "")))
                    price = float(row["售价"]) if pd.notna(row.get("售价")) else 0.0
                    size = str(row["尺码"]) if "尺码" in df.columns else ""
                    color = str(row["颜色"]) if "颜色" in df.columns else ""
                    season = str(row["上架季节"]) if "上架季节" in df.columns else ""
                    fab_text = (
                        f"尺码：{size}\n面料：{row.get('面料成分','')}\n"
                        f"版型/设计：{row.get('版型/设计描述','')}\n颜色：{color}\n季节：{season}"
                    )
                    # 可选的回测字段（如果 Excel 里存在就带上）
                    sales = int(parse_sales_value(row[sales_col])) if sales_col else 0
                    m_grade = str(row[grade_col]).strip() if grade_col and pd.notna(row.get(grade_col)) else ""
                    if m_grade and m_grade not in ("S", "A+", "A", "P"):
                        m_grade = ""  # 不是合法分级 → 丢弃
                    st_pct = 0.0
                    if st_col and pd.notna(row.get(st_col)):
                        raw_st = str(row[st_col]).strip()
                        if raw_st.endswith("%"):
                            st_pct = safe_float(raw_st.rstrip("%"), 0.0) / 100.0
                        else:
                            st_pct = safe_float(raw_st, 0.0)

                    style_infos.append(StyleInfo(
                        style_id=sid, category=cat,
                        price=price, season=season, fab_description=fab_text,
                        sales_qty=sales, manual_grade=m_grade, sell_through_pct=st_pct,
                        is_main_push=_tv(row[push_col]) if push_col and pd.notna(row.get(push_col)) else False,
                        is_live_stream=_tv(row[live_col]) if live_col and pd.notna(row.get(live_col)) else False,
                    ))
                    imgs = style_to_images.get(sid, [])
                    image_paths_map[sid] = [str(p) for p in imgs]

                # --- 盲测组抽样（分层随机，seed=批次名保证确定性可复现）---
                blind_ids: set[str] = set()
                if blind_enabled:
                    from src.blind_set import select_blind_set
                    blind_ids = select_blind_set(
                        style_infos, ratio=blind_ratio, seed=batch_name,
                    )
                    for info in style_infos:
                        info.is_blind = info.style_id in blind_ids
                st.session_state.blind_ids = blind_ids

                st.session_state.style_infos = style_infos
                st.session_state.image_paths_map = image_paths_map
                st.session_state.preds = []
                st.session_state.batch_finalized = False
                st.session_state.progress_info = {
                    "current": 0, "total": len(style_infos),
                    "stage": "开始评估…", "failed": {},
                }
                if blind_ids:
                    st.success(
                        f"✅ 批次已提交！盲测组已抽取 {len(blind_ids)}/{len(style_infos)} 款"
                        f"（🔒 款号：{'、'.join(sorted(blind_ids))}）。\n\n"
                        f"请切换到「📋 批次总表」页面查看进度和结果。"
                    )
                else:
                    st.success("✅ 批次已提交！请切换到「📋 批次总表」页面查看进度和结果。")

    with col2:
        if st.button("🧹 清空本次输入", use_container_width=True):
            for key in ("preds", "df_input", "style_to_images", "progress_info",
                        "style_infos", "image_paths_map", "blind_ids",
                        "batch_finalized", "history_batch_id"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ============================================================
# 页面 2：📋 批次总表
# ============================================================
def render_page_summary():
    render_breadcrumb(suffix="📋 批次总表")
    st.title(f"📋 批次总表"
             f" {(' · ' + st.session_state.batch_name) if st.session_state.batch_name else ''}")

    # 若真实模式因缺 Key 回退到 mock，必须在主区域醒目提示，避免用户误以为调了 API
    _fallback_reason = st.session_state.get("api_key_fallback_reason")
    if _fallback_reason:
        st.error(f"🚨 {_fallback_reason}")
        st.warning(
            "在 mock 模式下，所有分数都是本地随机生成的，跟真实销量无关，"
            "也就无法通过 3Loop 校准发现错误。请粘贴 API Key 后重新上传批次。"
        )
        # 不 return，允许用户查看已生成的假结果，但不再继续处理

    if "style_infos" not in st.session_state or not st.session_state.style_infos:
        st.info("还没有批次在运行。请到「📤 上传批次」提交评估。")
        return

    style_infos = st.session_state.style_infos
    image_paths_map = st.session_state.image_paths_map
    prog = st.session_state.progress_info

    # 确保 progress_info 有 failed 字典
    prog.setdefault("failed", {})

    total = prog["total"] or len(style_infos)
    current = prog["current"]
    if current < total:
        st.subheader(f"⏳ 处理进度：{current}/{total}")
        st.progress(current / max(total, 1),
                    text=f"{prog['stage']}  ·  预计剩余约 {(total-current)*45//60 if total else 0} 分钟")
        if current < len(style_infos):
            info = style_infos[current]
            try:
                prog["stage"] = f"正在处理款 {info.style_id}"
                llm_backend = st.session_state.get("llm_backend", "mock")
                pl = PredictionPipeline(
                    brand_id=brand_cfg.brand_id,
                    llm_backend=llm_backend,
                    api_key=st.session_state.get("api_key_for_backend", "") or None,
                )
                # run_one 会自动根据 llm_backend 决定走 mock 还是真实 LLM
                # 真实模式下 image_paths_map 会注入到 info.images 供 VLM 读取图片
                pred = pl.run_one(info, image_paths_map=image_paths_map)
                st.session_state.preds.append(pred)
                # 成功 → 如果之前有失败记录，清除
                prog["failed"].pop(info.style_id, None)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                prog["failed"][info.style_id] = err_msg
                # 额度耗尽/鉴权失效：止损中止批次（剩余款每款还会白打 ~60 次失败调用）
                if is_fatal_quota_error(err_msg):
                    remaining = [s.style_id for s in style_infos[current + 1:]]
                    for sid in remaining:
                        prog["failed"][sid] = "已跳过（批次因 API 鉴权/额度问题中止）"
                    prog["current"] = total
                    prog["stage"] = "⛔ 已中止：API 鉴权/额度问题"
                    _backend_is_zhipu = (llm_backend == "zhipu")
                    _repair_hint = (
                        "智谱免费模型无额度概念，此错误通常是 **API Key 无效**——"
                        "请到 [智谱 API Keys 页](https://open.bigmodel.cn/usercenter/apikeys) "
                        "重新生成并粘贴到侧边栏。"
                        if _backend_is_zhipu else
                        "请到 [百炼控制台](https://bailian.console.aliyun.com) 充值或领取资源包后，"
                        "重新上传批次。"
                    )
                    st.error(
                        f"⛔ 款 {info.style_id} 触发 **API 鉴权失败/额度不足**，批次已中止"
                        f"（剩余 {len(remaining)} 款跳过）。\n\n"
                        f"{_repair_hint}错误详情：`{err_msg[:300]}`"
                    )
                    st.rerun()
                    return
                st.warning(f"⚠️ 款 {info.style_id} 处理失败：{err_msg}")
            prog["current"] = current + 1
            if prog["current"] >= total:
                prog["stage"] = "全部完成 ✓"
            time.sleep(0.1)
            st.rerun()
        return

    preds: list[FullPrediction] = st.session_state.preds
    failed = prog.get("failed", {})
    n_success = len(preds)
    n_failed = len(failed)

    if n_failed > 0:
        st.warning(
            f"⚠️ 批次处理完成，但有 **{n_failed}/{total}** 款处理失败！"
            f"成功 {n_success} 款，失败 {n_failed} 款。"
        )
        with st.expander(f"查看 {n_failed} 款失败详情"):
            for sid, err in failed.items():
                st.markdown(f"- **{sid}**：`{err}`")
    else:
        st.success(f"✅ 全部完成，共 {total} 款（成功 {n_success}，失败 0）")

    # 批次真实性标记：让用户一眼知道当前结果是真 VLM 还是 mock 跑的
    if preds:
        _meta = preds[0].metadata or {}
        _is_mock = _meta.get("is_mock", st.session_state.get("llm_backend") == "mock")
        if _is_mock:
            st.caption("🧪 **Mock 演示模式** · 分数为本地确定性随机生成，跟真实销量无关。"
                       "用于调试管线，不能评估预测准确率。")
        else:
            _feats = _meta.get("feature_models", ["qwen-vl-plus"])
            _pers = _meta.get("persona_models", ["qwen-max", "deepseek-v3"])
            _elapsed = _meta.get("elapsed_s")
            _elapsed_str = f" · 单款耗时约 {_elapsed}s" if _elapsed else ""
            _backend_label = "智谱 GLM（免费）" if _meta.get("llm_backend") == "zhipu" else "阿里云百炼"
            st.caption(
                f"🤖 **真实 VLM 评估（{_backend_label}）** · 特征提取：{' / '.join(_feats)} | "
                f"人设投票：{' / '.join(_pers)}{_elapsed_str}。"
                f"此批次分数来自 {_backend_label} API 对图片的实际分析。"
            )

            # ---- API 用量仪表：本批调用了多少次、花了多少 token ----
            _usage_calls = sum(
                int((p.metadata.get("api_usage") or {}).get("calls", 0)) for p in preds
            )
            if _usage_calls > 0:
                _u_in = sum(
                    int((p.metadata.get("api_usage") or {}).get("input_tokens", 0)) for p in preds
                )
                _u_out = sum(
                    int((p.metadata.get("api_usage") or {}).get("output_tokens", 0)) for p in preds
                )
                _by_model: dict[str, dict[str, int]] = {}
                for p in preds:
                    for m, stat in (p.metadata.get("api_usage") or {}).get("by_model", {}).items():
                        agg = _by_model.setdefault(
                            m, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                        )
                        for k in agg:
                            agg[k] += int(stat.get(k, 0))
                c1, c2, c3 = st.columns(3)
                c1.metric("API 调用次数", f"{_usage_calls:,}")
                c2.metric("输入 tokens", f"{_u_in:,}")
                c3.metric("输出 tokens", f"{_u_out:,}")
                with st.expander("按模型分解（额度去哪了一眼可见）"):
                    st.dataframe(
                        pd.DataFrame(
                            [{"模型": m, **stat} for m, stat in _by_model.items()]
                        ).sort_values("calls", ascending=False),
                        use_container_width=True, hide_index=True,
                    )

    if not preds:
        st.error("❌ 没有任何款式成功处理，请检查输入数据或 LLM 配置")
        return

    # 批次收尾（只执行一次，防止页面 rerun 重复写历史库）：
    # ① 批次内相对分级 ② 持久化到历史库
    if not st.session_state.get("batch_finalized"):
        st.session_state["batch_finalized"] = True
        # ① 相对分级：冷启动绝对阈值失效时按批次内分位重定档（幂等）
        assign_relative_grades(preds)
        # ② 历史库持久化：跨批次累积价格/特征/分数分布
        try:
            pl = PredictionPipeline(
                brand_id=brand_cfg.brand_id,
                llm_backend=st.session_state.llm_backend,
                api_key=st.session_state.get("api_key_for_backend", "") or None,
            )
            batch_id = pl.save_to_history(
                preds,
                notes=f"Streamlit批次: {st.session_state.get('batch_name', 'untitled')}",
            )
            st.session_state["history_batch_id"] = batch_id
            st.caption(f"💾 已存入历史库 batch `{batch_id}`（含相对分级结果），供后续批次累积分布")
        except Exception as e:
            st.warning(f"⚠️ 历史库写入失败（不影响当前结果）: {e}")

    # ---- 分级逻辑诊断：分数分布 + 绝对档 vs 相对档对照 ----
    _diag_preds = [p for p in preds if not p.info.is_blind]
    if len(_diag_preds) >= 3:
        _scores = [p.grade.final_score for p in _diag_preds]
        _mean = sum(_scores) / len(_scores)
        _std = (sum((x - _mean) ** 2 for x in _scores) / len(_scores)) ** 0.5
        _n_shift = sum(
            1 for p in _diag_preds
            if p.metadata.get("absolute_grade", p.grade.grade) != p.grade.grade
        )
        _abs_counts = pd.Series(
            [p.metadata.get("absolute_grade", p.grade.grade) for p in _diag_preds]
        ).value_counts()
        _rel_counts = pd.Series([p.grade.grade for p in _diag_preds]).value_counts()
        with st.expander("🔬 分级逻辑诊断（批次内相对分级）", expanded=False):
            st.markdown(
                f"本批次已启用**批次内相对分级**（S=Top 20%，A+=20-40%，A=40-75%，P=后 25%）。"
                f"冷启动时 VLM 绝对分尺度未校准（常见现象：全部款 65-75 分 → 绝对档全是 A），"
                f"但**排序可靠**，相对分级直接消费排序信息，与人工内审「这批里最好的打 S」语义对齐。"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("综合分均值", f"{_mean:.1f}")
            c2.metric("标准差 σ", f"{_std:.1f}")
            c3.metric("分数极差", f"{max(_scores) - min(_scores):.1f}")
            c4.metric("档位调整数", f"{_n_shift}/{len(_diag_preds)}")
            if _std < 4.0:
                st.warning(
                    f"⚠️ **分数扎堆告警**：本批次综合分 σ={_std:.1f}（<4），绝对阈值分档基本失效"
                    f"（绝对档 {_abs_counts.to_dict()}）——这正是相对分级存在的意义。"
                    f"待 3Loop 校准积累（≥10 款销量回填）后绝对档会逐步可用。"
                )
            else:
                st.info(
                    f"分数分布较分散（σ={_std:.1f}），绝对档与相对档分布对照："
                    f"绝对 {_abs_counts.to_dict()} vs 相对 {_rel_counts.to_dict()}。"
                )

    rows = []
    for p in preds:
        # 绝对档对照列：绝对阈值分档（冷启动时尺度未校准，仅作参考）
        abs_grade = p.metadata.get("absolute_grade", p.grade.grade)
        if p.info.is_blind:
            # 盲测款：结论列打码（运营不可见，销量回填后在回测页解锁对照）
            rows.append({
                "款号": p.info.style_id,
                "分级": "🔒 盲测",
                "绝对档": "🔒",
                "综合分": "🔒",
                "自然分": "🔒",
                "直播分": "🔒",
                "感知价值": "🔒",
                "价值匹配": "🔒",
                "价格风险": "🔒",
                "主推渠道": "🔒",
                "售价": p.info.price,
            })
        else:
            rows.append({
                "款号": p.info.style_id,
                "分级": p.grade.grade,
                "绝对档": abs_grade,
                "综合分": round(p.grade.final_score, 1),
                "自然分": round(p.channels.natural_score, 1),
                "直播分": round(p.channels.live_score, 1),
                "感知价值": round(p.channels.perceived_value, 1),
                "价值匹配": round(p.channels.value_match, 2),
                "价格风险": p.channels.price_risk,
                "主推渠道": p.grade.recommended_channel,
                "售价": p.info.price,
            })
    df = pd.DataFrame(rows)

    col1, col2, col3 = st.columns(3)
    with col1:
        grade_filter = st.multiselect(
            "只看分级",
            options=["S", "A+", "A", "P", "风险", "🔒 盲测"],
            default=["S", "A+", "A", "P", "风险", "🔒 盲测"])
    with col2:
        risk_filter = st.multiselect(
            "价格风险", options=["低风险", "中风险", "高风险"],
            default=["低风险", "中风险", "高风险"])
    with col3:
        channel_filter = st.multiselect(
            "主推渠道",
            options=["自然流量优先", "直播带货优先", "双渠道均衡",
                     "设计调性款（不做主推）", "高风险不推"],
            default=["自然流量优先", "直播带货优先", "双渠道均衡",
                     "设计调性款（不做主推）", "高风险不推"])
    dff = df[
        df["分级"].isin(grade_filter)
        & df["价格风险"].isin(risk_filter)
        & df["主推渠道"].isin(channel_filter)
    ]

    st.dataframe(dff, use_container_width=True, height=450, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        grade_counts = df["分级"].value_counts().reindex(["S", "A+", "A", "P"]).fillna(0).astype(int)
        st.subheader("📊 分级分布")
        st.bar_chart(grade_counts)
    with col2:
        st.subheader("📥 导出")
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            dff.to_excel(writer, index=False, sheet_name="批次总表")
        st.download_button(
            "⬇️ 导出总表 Excel", data=bio.getvalue(),
            file_name=f"{brand_cfg.brand_id}_{st.session_state.batch_name}_总表.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        zip_buf = io.BytesIO()
        n_blind_hidden = 0
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in preds:
                if p.info.is_blind:
                    n_blind_hidden += 1
                    continue  # 盲测款不进报告包（对运营不可见）
                md = render_single_report_markdown(p)
                zf.writestr(f"{p.info.style_id}_报告.md", md)
        if n_blind_hidden:
            st.caption(f"🔒 {n_blind_hidden} 个盲测款未包含在报告包中")
        st.download_button(
            "⬇️ 打包下载全部单款报告.zip", data=zip_buf.getvalue(),
            file_name=f"{brand_cfg.brand_id}_{st.session_state.batch_name}_单款报告.zip",
            mime="application/zip", use_container_width=True,
        )

    st.divider()
    st.info("👉 点击下方款号可跳转到「🔍 单款详情报告」：")
    sel_style_id = st.selectbox("选择查看的款号", options=[p.info.style_id for p in preds])
    st.session_state.selected_style_id = sel_style_id


# ============================================================
# 页面 3：🔍 单款详情报告
# ============================================================
def render_page_detail():
    render_breadcrumb(suffix="🔍 单款详情")
    st.title("🔍 单款详情报告")
    preds: list[FullPrediction] = st.session_state.get("preds", [])
    selected = st.session_state.get("selected_style_id", "")
    if not preds:
        st.info("还没有评估结果。请先到「📤 上传批次」运行评估。")
        return

    style_ids = [p.info.style_id for p in preds]
    selected = st.selectbox("款号", options=style_ids,
                            index=style_ids.index(selected) if selected in style_ids else 0)
    p = next(x for x in preds if x.info.style_id == selected)

    # 盲测款：结论锁定（销量回填后在「📉 回测校准」页解锁对照）
    if p.info.is_blind:
        st.markdown(
            f"### {p.info.style_id}  <span style='background:#6b7280;color:white;"
            f"padding:3px 12px;border-radius:6px;font-weight:700;'>🔒 盲测中</span>",
            unsafe_allow_html=True,
        )
        st.info(
            "该款被抽入本批盲测组：AI 已完成完整预测并存档，但结论在内审阶段对运营不可见，"
            "以保证内审决策不受 AI 干扰。**销量回填后**，到「📉 回测校准」页"
            "「盲测组对照」区域解锁查看，并参与 AI vs 人工分歧裁决。"
        )
        st.caption(f"基本信息：品类 {p.info.category or '未标注'} · 售价 {p.info.price} 元 · {p.info.season or '季节未标注'}")
        return

    g: GradeResult = p.grade
    color_map = {"S": "#ef4444", "A+": "#f59e0b", "A": "#22c55e", "P": "#6b7280"}
    c = color_map.get(g.grade, "#6b7280")
    st.markdown(
        f"### {p.info.style_id}"
        f"  <span style='color:gray;font-weight:normal'>综合评分</span>"
        f"  **{g.final_score:.1f} / 100**"
        f"  <span style='color:gray;font-size:13px'>（置信度 {g.confidence:.0%}）</span>"
        f"  <span style='background:{c};color:white;padding:3px 10px;border-radius:6px;font-weight:700;'>{g.grade}级</span>"
        f"  <span style='color:#6366f1;'>主推：{g.recommended_channel}</span>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("📷 款式图片")
        image_paths = st.session_state.get("image_paths_map", {}).get(p.info.style_id, [])
        if image_paths:
            tabs = st.tabs([f"图{i+1}" for i in range(len(image_paths))])
            for tab, ipath in zip(tabs, image_paths):
                try:
                    tab.image(ipath, use_container_width=True, caption=Path(ipath).name)
                except Exception:
                    tab.caption(f"无法加载: {ipath}")
        else:
            st.caption("（无图片）")

        # —— 优势 / 劣势（来自 GradeResult.strengths / weaknesses）——
        st.subheader("🏆 优劣势速览")
        strength_items = g.strengths or []
        weakness_items = g.weaknesses or []
        if strength_items:
            with st.expander(f"✅ 优势（{len(strength_items)}）", expanded=True):
                for s in strength_items[:5]:
                    st.markdown(f"- {s}")
        else:
            st.caption("（暂无突出优势）")
        if weakness_items:
            with st.expander(f"⚠️ 劣势（{len(weakness_items)}）", expanded=True):
                for w in weakness_items[:5]:
                    st.markdown(f"- {w}")
        else:
            st.caption("（暂无明显劣势）")

    with right:
        with st.expander("📝 基本信息", expanded=False):
            st.write(f"品类：{p.info.category}   售价：¥{p.info.price:.0f}   季节：{p.info.season}")
            st.write(f"FAB：{str(p.info.fab_description or '（无FAB描述）')[:300]}…")

        st.subheader("🎯 10特征BARS评分")
        feat_rows = []
        for key, f in p.features.features.items():
            feat_rows.append({"特征": f.name, "分数": f.score, "理由": f.reason or "（无）"})
        df_feat = pd.DataFrame(feat_rows).sort_values("分数")
        st.bar_chart(df_feat, x="特征", y="分数", horizontal=True, color="#6366f1", height=360)

        with st.expander("🔍 查看每个特征的 VLM 判断理由", expanded=False):
            for _, row in df_feat.iterrows():
                st.markdown(f"**{row['特征']} · {row['分数']:.1f}/10**")
                st.caption(row['理由'])
                st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👥 人设投票")
            st.metric("加权总分", f"{p.voting.weighted_score:.2f}")
            st.metric("支持率", f"{p.voting.support_rate:.0%}")
            st.metric("反对率", f"{p.voting.opposition_rate:.0%}")
            st.metric("分数标准差", f"{p.voting.score_std:.2f}")
            with st.expander("支持/反对理由"):
                if p.voting.top_buy_reasons:
                    st.markdown("**支持理由 TOP：**")
                    for r in p.voting.top_buy_reasons[:3]:
                        st.write(f"  ✓ {r}")
                if p.voting.top_oppose_reasons:
                    st.markdown("**反对理由 TOP：**")
                    for r in p.voting.top_oppose_reasons[:3]:
                        st.write(f"  ✗ {r}")
        with col2:
            st.subheader("🛒 双渠道评分")
            st.metric("自然流量分", f"{p.channels.natural_score:.1f}")
            st.metric("直播带货分", f"{p.channels.live_score:.1f}")
            st.metric("感知价值", f"{p.channels.perceived_value:.1f}")
            st.metric("价值匹配",
                      f"{p.channels.value_match:+.2f}（{p.channels.price_risk}）",
                      delta=f"价格百分位 {p.channels.price_percentile:.0%}")

        st.subheader("⚖️ 三大引擎综合")
        engine_df = pd.DataFrame({
            "引擎": ["人设投票", "双渠道评分(自然)", "双渠道评分(直播)", "价格价值"],
            "得分": [p.voting.weighted_score, p.channels.natural_score,
                     p.channels.live_score, p.channels.perceived_value],
        })
        st.bar_chart(engine_df, x="引擎", y="得分", color="#a855f7", height=250, use_container_width=True)

        # —— 消费者洞察 + 改款建议（来自 GradeResult）——
        st.subheader("💡 消费者洞察")
        st.info(g.consumer_insights or "暂无人设洞察")

        st.subheader("🔧 改款建议")
        if g.improvements:
            for i, s in enumerate(g.improvements, 1):
                st.write(f"{i}. {s}")
        else:
            st.caption("（暂无改款建议）")

    st.divider()
    md_text = render_single_report_markdown(p)
    st.download_button(
        "⬇️ 下载 Markdown 报告",
        data=md_text,
        file_name=f"{brand_cfg.brand_id}_{p.info.style_id}_{st.session_state.batch_name}_报告.md",
        mime="text/markdown",
    )


# ============================================================
# 页面 4：📊 回测校准（含3Loop优化内核按钮）
# ============================================================
def render_page_calibration():
    render_breadcrumb(suffix=f"🔁 校准轮次 {n_calibration or 0}")
    st.title("📊 回测校准")
    st.caption(
        "上传带「真实销售结果」列的历史数据："
        "先跑基础回测Spearman → 再点「🤖 运行3Loop核心优化内核」→ 新Spearman对比 + 残差识别"
    )

    preds: list[FullPrediction] = st.session_state.get("preds", [])
    if not preds:
        st.info("当前没有评估结果。请先运行评估，或上传历史批次的preds.json缓存。")
        cached = st.file_uploader("上传历史缓存 predictions.json", type=["json"])
        if cached:
            st.info("（上传历史缓存功能：后续版本从FullPrediction JSON恢复）")

    xlsx = st.file_uploader("上传带真实销量的 Excel（含「款式编号」+「真实销售结果/真实销量」列）",
                            type=["xlsx", "csv"])

    truth_map_ready: dict[str, float] | None = None
    df_truth: pd.DataFrame | None = None
    if xlsx is not None:
        try:
            if xlsx.name.endswith(".csv"):
                df_truth = pd.read_csv(xlsx)
            else:
                df_truth = pd.read_excel(xlsx)

            # 用统一别名匹配（支持 "真实销量结果" / "真实销售结果" / "真实销量" / ...）
            col_truth = find_aliased_column(list(df_truth.columns), SALES_QTY_COL_ALIASES)
            if not col_truth:
                st.error(f"❌ 未找到销量列（支持的别名：{SALES_QTY_COL_ALIASES}）")
            else:
                st.success(f"✅ 识别到销量列「{col_truth}」，共{len(df_truth)}行")
                with st.expander("预览（前5行）"):
                    st.dataframe(df_truth.head(5), use_container_width=True)

                truth_map_ready = {}
                for _, r in df_truth.iterrows():
                    sid = str(r["款式编号"])
                    truth_map_ready[sid] = parse_sales_value(r[col_truth])

                non_zero = sum(1 for v in truth_map_ready.values() if v > 0)
                if non_zero == 0:
                    st.warning("⚠️ 识别到的销量值全为 0 — 请检查列是否匹配正确")
        except Exception as e:
            st.error(f"Excel读取失败：{e}")

    # ---------- 盲测组对照（销量回填即解锁，不依赖 3Loop）----------
    _has_blind = any(p.info.is_blind for p in preds) if preds else False
    if _has_blind and truth_map_ready:
        st.divider()
        st.subheader("🔬 盲测组对照（AI 纯净准确率验证）")
        try:
            from src.blind_set import compare_blind_vs_control
            _rows_b = []
            for p in preds:
                _sid = p.info.style_id
                if _sid in truth_map_ready:
                    _rows_b.append({
                        "style_id": _sid,
                        "is_blind": 1 if p.info.is_blind else 0,
                        "final_score": float(p.grade.final_score),
                        "final_grade": p.grade.grade,
                        "manual_grade": p.info.manual_grade or "",
                        "sales_qty": float(truth_map_ready[_sid]),
                        "batch_id": "current",
                        "created_at": "",
                    })
            bcr = compare_blind_vs_control(pd.DataFrame(_rows_b))

            m1, m2, m3 = st.columns(3)
            m1.metric("盲测款（已回填销量）", f"{bcr.n_blind} 个")
            m2.metric("🔒 盲测组 Spearman（纯净）",
                      f"{bcr.blind_spearman:+.3f}" if bcr.blind_spearman is not None else "样本不足",
                      help="盲测款内审时运营未看 AI 结论，其销量未被 AI 建议放大——"
                           "这是 AI 预测能力的真实基线")
            m3.metric("👁️ 对照组 Spearman",
                      f"{bcr.control_spearman:+.3f}" if bcr.control_spearman is not None else "样本不足",
                      help="非盲测款：运营内审时参考了 AI 结论 → 销量含投放放大效应")
            if bcr.blind_spearman is not None and bcr.control_spearman is not None:
                _gap = bcr.control_spearman - bcr.blind_spearman
                st.caption(
                    f"两组差值 {_gap:+.3f}：对照组高于盲测组的部分≈「AI 建议+运营执行」"
                    f"带来的放大效应；盲测组偏低不代表系统不准，而是去掉了投放杠杆。"
                )
            else:
                st.info(f"ℹ️ {bcr.note}：盲测组至少需 5 个已回填销量的款才能出纯净基线。")

            # AI vs 人工分歧裁决表（盲测款专属：人工判断未被 AI 带偏）
            if bcr.disputes:
                st.markdown("**⚖️ AI vs 人工分歧裁决**（仅盲测款，按实际销量裁决谁对）")
                st.dataframe(pd.DataFrame(bcr.disputes),
                             use_container_width=True, hide_index=True)
                d1, d2, d3 = st.columns(3)
                d1.metric("🤖 AI 判断对", f"{bcr.ai_wins} 款",
                          help="AI 分级更接近实际销量表现的分歧款")
                d2.metric("👤 人工判断对", f"{bcr.human_wins} 款")
                d3.metric("➖ 无法裁决（中段）", f"{bcr.tie} 款")
                st.caption(
                    "这是系统价值的直接证据：AI 对而人工错的款 = 下一季可放大的选款信号；"
                    "人工对而 AI 错的款 = 喂给 3Loop 的校准样本。"
                )
            elif bcr.n_blind > 0:
                st.caption("本批盲测款无 AI/人工分级分歧（或人工分级列缺失），无裁决项。")

            # 解锁盲测款完整结论（销量已回填）
            _blind_preds = [p for p in preds if p.info.is_blind]
            if _blind_preds:
                with st.expander(f"🔓 解锁查看本批 {len(_blind_preds)} 个盲测款完整 AI 结论"):
                    _unlock_rows = []
                    for p in _blind_preds:
                        _unlock_rows.append({
                            "款号": p.info.style_id,
                            "AI分级": p.grade.grade,
                            "AI综合分": round(p.grade.final_score, 1),
                            "人工分级": p.info.manual_grade or "—",
                            "实际销量": truth_map_ready.get(p.info.style_id, "—"),
                        })
                    st.dataframe(pd.DataFrame(_unlock_rows),
                                 use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"⚠️ 盲测组对照分析失败：{e}")
    elif _has_blind and preds:
        st.divider()
        st.info(
            "🔬 本批含盲测款：上传带真实销量的 Excel 后，"
            "「盲测组对照」将自动解锁（AI 纯净准确率 + AI vs 人工分歧裁决）。"
        )

    # ---------- 回测校准按钮（3Loop内核：Spearman对比+残差分离）----------
    do_3loop = st.button("🤖 运行3Loop核心优化内核 + 残差分离",
                         type="primary",
                         disabled=(not preds or not truth_map_ready or len(preds) < 8),
                         help="至少8款数据才能启动Lasso人设分布拟合")

    if do_3loop and truth_map_ready and preds:
        with st.spinner("3Loop校准运行中（Loop1→Loop2→Loop3→残差分离，20秒）…"):
            # 调用 PredictionPipeline.run_backtest_calibration（spec §9）
            # 内部封装：build_history_df + run_all_loops + 残差归一化
            # 产物写 brand_cfg.calibrated_dir（下次评估自动加载，越用越准）
            try:
                # 校准不需要调LLM，但要同一个品牌配置
                _llm_backend = st.session_state.get("llm_backend", "mock")
                pl = PredictionPipeline(
                    brand_id=brand_cfg.brand_id,
                    llm_backend=_llm_backend,
                    api_key=st.session_state.get("api_key_for_backend", "") or None,
                )
                loop_result = pl.run_backtest_calibration(
                    predictions=preds,
                    sales_lookup=truth_map_ready,
                )
            except RuntimeError as re:
                st.error(f"依赖缺失：{re}")
                loop_result = None
            except Exception as ex:
                st.error(f"3Loop运行失败：{ex}")
                loop_result = None

            if loop_result is None:
                matched = sum(1 for p in preds if p.info.style_id in (truth_map_ready or {}))
                if matched == 0:
                    st.error("❌ 没有款能匹配到真实销量，请检查「款式编号」列是否一致。")
                elif matched < 8:
                    st.error(f"❌ 可匹配的款只有{matched}款，Loop2需要至少8款样本，请补充历史批次。")
                else:
                    st.info("ℹ️ 3Loop 跳过：可能销量全为0或信号不足，请检查真实销量数据。")

            if loop_result is not None:
                # 3) 展示三色Spearman对比表
                st.subheader("🎯 3Loop 校准前后 Spearman 对比")
                def _flag(old, new, min_imp):
                    delta = new - old
                    if delta >= min_imp:
                        return f"🟢 +{delta:+.3f}", "✅ 已应用"
                    if delta >= 0:
                        return f"⚪ {delta:+.3f}", "⏭️ 持平未应用"
                    return f"🔴 {delta:+.3f}", "🛡️ 保护机制拦截"

                # Loop1
                l1_delta_color, l1_status = _flag(
                    loop_result.loop1.old_spearman_avg,
                    loop_result.loop1.new_spearman_avg,
                    0.0
                )
                # Loop2
                l2_delta_color, l2_status = _flag(
                    loop_result.loop2.old_spearman,
                    loop_result.loop2.new_spearman,
                    0.01
                )
                # Loop3 引擎侧
                l3e_delta_color, l3e_status = _flag(
                    loop_result.loop3.old_engine_spearman,
                    loop_result.loop3.new_engine_spearman,
                    0.01
                )
                # Loop3 渠道侧
                l3c_delta_color, l3c_status = _flag(
                    loop_result.loop3.old_channel_spearman,
                    loop_result.loop3.new_channel_spearman,
                    0.01
                )

                df_compare = pd.DataFrame([
                    {"步骤": "Loop1 VLM特征校准",
                     "指标": "10特征Spearman(ρ)均值",
                     "校准前": f"{loop_result.loop1.old_spearman_avg:+.3f}",
                     "校准后": f"{loop_result.loop1.new_spearman_avg:+.3f}",
                     "Δ 提升": l1_delta_color, "状态": l1_status},
                    {"步骤": "Loop2 人设分布拟合（Lasso）",
                     "指标": "30人设投票 Spearman",
                     "校准前": f"{loop_result.loop2.old_spearman:+.3f}",
                     "校准后": f"{loop_result.loop2.new_spearman:+.3f}",
                     "Δ 提升": l2_delta_color, "状态": l2_status},
                    {"步骤": "Loop3 引擎权重调优",
                     "指标": "三大引擎合成 Spearman",
                     "校准前": f"{loop_result.loop3.old_engine_spearman:+.3f}",
                     "校准后": f"{loop_result.loop3.new_engine_spearman:+.3f}",
                     "Δ 提升": l3e_delta_color, "状态": l3e_status},
                    {"步骤": "Loop3 渠道权重调优",
                     "指标": "自然/直播 合成Spearman",
                     "校准前": f"{loop_result.loop3.old_channel_spearman:+.3f}",
                     "校准后": f"{loop_result.loop3.new_channel_spearman:+.3f}",
                     "Δ 提升": l3c_delta_color, "状态": l3c_status},
                ])
                st.dataframe(df_compare, use_container_width=True, hide_index=True)

                # 4) 残差分离结果
                st.subheader("🔍 残差分离（不可预知因素识别）")
                rd = loop_result.residual
                st.info(f"残差均值 μ = {rd.residual_mean:+.3f}，残差标准差 σ = {rd.residual_std:.3f}")
                col_over, col_under, col_flag = st.columns(3)
                with col_over:
                    st.metric("🟢 超预期款 (ε > μ+2σ)", f"{len(rd.overperformers)} 个")
                    if rd.overperformers:
                        with st.expander("查看详情（运营复盘机会）"):
                            for o in rd.overperformers:
                                attr = o.get("attribution")
                                attr_tag = f" · {attr}" if attr else ""
                                st.write(f"- {o.get('style_id','?')}{attr_tag}")
                with col_under:
                    st.metric("🔵 不及预期款 (ε < μ-2σ)", f"{len(rd.underperformers)} 个")
                    if rd.underperformers:
                        with st.expander("查看详情（复盘改进方向）"):
                            for u in rd.underperformers:
                                attr = u.get("attribution")
                                attr_tag = f" · {attr}" if attr else ""
                                st.write(f"- {u.get('style_id','?')}{attr_tag}")
                with col_flag:
                    flag_levels = {
                        "NO_SIG": ("⚪ 无系统偏差", "#22c55e"),
                        "MODERATE": ("🟡 轻度偏差（关注）", "#f59e0b"),
                        "STRONG": ("🔴 强系统偏差（必须复盘）", "#ef4444"),
                        "SIGMA_ZERO": ("⚪ 样本过少或全命中预测", "#6b7280"),
                        "ERROR": ("⚫ 计算异常", "#111827"),
                    }
                    label, color = flag_levels.get(rd.system_bias_flag, ("未知", "#6b7280"))
                    st.markdown(
                        f"<div style='background:{color};color:white;padding:10px 14px;"
                        f"border-radius:8px;text-align:center;font-weight:700;'>"
                        f"{label}</div>",
                        unsafe_allow_html=True,
                    )
                st.caption("⚠️ 以上残差款 **不参与3Loop学习**，避免将外部事件（KOL带货/竞品打折）的伪相关注入预测模型。")

                # 4.5) 资源错配归因（款式质量 × 投放强度，ADR-0001）
                if rd.marketing_available:
                    st.subheader("🎯 资源错配归因（款式质量 × 实际投放）")
                    _as = rd.attribution_summary or {}
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("✅ 推对了", f"{_as.get('推对了：投放放大了款式潜力', 0)} 个",
                              help="投放且超预期：系统预测与投放决策一致")
                    c2.metric("💎 漏网爆款", f"{_as.get('漏网爆款：没推也超预期，应追加投放', 0)} 个",
                              help="未投放却超预期：本季最大机会损失，下季应提前识别")
                    c3.metric("💸 资源错配", f"{_as.get('资源错配：投放了仍不及预期', 0)} 个",
                              help="投放了仍不及预期：预算被浪费，复盘选款信号")
                    c4.metric("🩹 款式本身弱", f"{_as.get('款式本身弱：未投放且不及预期', 0)} 个",
                              help="未投放且不及预期：不推是正确决策")
                    st.caption(
                        "依据：批次 Excel 的「是否主推 / 是否直播重点」列（实际投放口径）。"
                        "漏网爆款 = 下一季最该提前识别并追加备货的款。"
                    )
                else:
                    st.info(
                        "ℹ️ 本批次未带营销投放列（是否主推/是否直播重点），"
                        "暂无法做资源错配归因。上传批次时补充这两列即可解锁。"
                    )

                # 5) 产物下载
                st.divider()
                st.subheader("📥 校准产物")
                for fpath in loop_result.output_files:
                    fp = Path(fpath)
                    if fp.suffix == ".yaml" and fp.exists():
                        st.download_button(
                            f"⬇️ {fp.name}",
                            data=fp.read_text(encoding="utf-8"),
                            file_name=fp.name,
                            mime="text/yaml",
                        )
                    elif fp.suffix == ".md" and fp.exists():
                        st.download_button(
                            f"⬇️ {fp.name}（完整报告）",
                            data=fp.read_text(encoding="utf-8"),
                            file_name=fp.name,
                            mime="text/markdown",
                            use_container_width=True,
                        )
                st.success(
                    f"✅ 3Loop校准完成！产物已写入 `brand_profiles/{brand_cfg.brand_id}/calibrated/`，"
                    "下次评估时会自动生效。"
                )


# ============================================================
# 入口
# ============================================================
if page == "📤 上传批次":
    render_page_upload()
elif page == "📋 批次总表":
    render_page_summary()
elif page == "🔍 单款详情报告":
    render_page_detail()
elif page == "📊 回测校准":
    render_page_calibration()
