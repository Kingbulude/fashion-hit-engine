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

from src.config import load_config, list_available_brands, load_brand_profile
from src.pipeline import run_predict_batch, PredictionPipeline
from src.calibration import run_backtest
from src.report import (
    render_single_report_markdown,
    render_backtest_summary_markdown,
)
from src.types import StyleInfo, FullPrediction, GradeResult, BrandConfig
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
    available_brands = ["tongzhuang-outdoor"]
# （_template 不展示，是模板）
if "brand_id" not in st.session_state:
    st.session_state.brand_id = available_brands[0] if available_brands else "tongzhuang-outdoor"

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
        st.write(f"**层数：** {len(brand_cfg.decision_structure.layers)} 层（含童装孩子否决层）")
    st.write(f"**品类数：** {len(brand_cfg.category_registry.get('categories', []))}")
    st.write(f"**人设数：** {len(getattr(brand_cfg, 'personas', {}).get('personas', []))} 个身份三轴线")
    st.write(f"**校准轮次：** {n_calibration or '0（冷启动）'}")

st.sidebar.divider()
st.sidebar.title("🧭 导航")
page = st.sidebar.radio("", PAGES, index=0)
st.sidebar.caption(
    "fashion-hit-engine v2.0\n"
    "· 通用CORE引擎\n"
    "· 品牌适配包 5YAML\n"
    "· 3Loop 越用越准"
)

# ========== 兼容：保持 cfg 变量接口（AppConfig 薄兼容）==========
cfg = load_config()  # deprecated薄包装，内部brand_id=tongzhuang-outdoor

# ========== 全局状态初始化 ==========
if "batch_name" not in st.session_state:
    st.session_state.batch_name = ""
if "preds" not in st.session_state:         # list[FullPrediction]
    st.session_state.preds = []
if "df_input" not in st.session_state:      # 原始输入Excel
    st.session_state.df_input = None
if "progress_info" not in st.session_state:  # 进度显示用
    st.session_state.progress_info = {"current": 0, "total": 0, "stage": "空闲"}
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


def validate_inputs(df: pd.DataFrame, image_style_ids: set[str]) -> list[str]:
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
                     and len(st.session_state.style_to_images.get(s, [])) == 0]
    if empty_folders:
        warnings.append(f"⚠️ 图片文件夹是空的：{empty_folders[:3]}")
    return warnings


def unzip_to_temp_dir(zip_bytes: bytes, dest_dir: Path) -> dict[str, list[Path]]:
    """解压图片zip，返回 款号->[图片路径列表]（按修改时间排序）"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    style_to_images: dict[str, list[Path]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)
    for p in sorted(dest_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        style_id = p.parent.name if p.parent.name != dest_dir.name else ""
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

    can_start = (
        bool(batch_name) and df is not None and df.shape[0] > 0
        and bool(style_ids_from_images) and not image_warnings
    )

    if df is not None and style_ids_from_images:
        warns = validate_inputs(df, style_ids_from_images)
        for w in warns:
            if w.startswith("❌"):
                st.error(w)
                can_start = False
            else:
                st.warning(w)

    st.divider()
    col1, col2, _ = st.columns([2, 2, 4])
    with col1:
        if not can_start:
            st.button("🚀 开始评估", disabled=True, use_container_width=True,
                      help="先填批次名、上传Excel和图片，并解决上面的警告")
        else:
            if st.button("🚀 开始评估", type="primary", use_container_width=True):
                st.session_state.batch_name = batch_name
                st.session_state.style_to_images = style_to_images
                style_infos: list[StyleInfo] = []
                image_paths_map: dict[str, list[str]] = {}
                for _, row in df.iterrows():
                    sid = str(row["款式编号"])
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
                    style_infos.append(StyleInfo(
                        style_id=sid, name=sid, category=cat,
                        price=price, season=season, fab_text=fab_text,
                    ))
                    imgs = style_to_images.get(sid, [])
                    image_paths_map[sid] = [str(p) for p in imgs]
                st.session_state.style_infos = style_infos
                st.session_state.image_paths_map = image_paths_map
                st.session_state.preds = []
                st.session_state.progress_info = {
                    "current": 0, "total": len(style_infos), "stage": "开始评估…"
                }
                st.success("✅ 批次已提交！请切换到「📋 批次总表」页面查看进度和结果。")

    with col2:
        if st.button("🧹 清空本次输入", use_container_width=True):
            for key in ("preds", "df_input", "style_to_images", "progress_info",
                        "style_infos", "image_paths_map"):
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

    if "style_infos" not in st.session_state or not st.session_state.style_infos:
        st.info("还没有批次在运行。请到「📤 上传批次」提交评估。")
        return

    style_infos = st.session_state.style_infos
    image_paths_map = st.session_state.image_paths_map
    prog = st.session_state.progress_info

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
                # 品牌注入：通过 PredictionPipeline 统一入口（use_mock=True 走mock模式，避免需要LLM）
                pl = PredictionPipeline(brand_id=brand_cfg.brand_id, llm_backend="mock")
                pred = pl.run_one(info, use_mock=True, image_paths_map=image_paths_map)
                st.session_state.preds.append(pred)
            except Exception as e:
                st.warning(f"⚠️ {info.style_id} 处理失败：{e}")
            prog["current"] = current + 1
            if prog["current"] >= total:
                prog["stage"] = "全部完成 ✓"
            time.sleep(0.1)
            st.rerun()
        return

    st.success(f"✅ 全部完成，共 {total} 款")
    preds: list[FullPrediction] = st.session_state.preds
    if not preds:
        return

    rows = []
    for p in preds:
        rows.append({
            "款号": p.info.style_id,
            "分级": p.grade.grade,
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
            "只看分级", options=["S", "A+", "A", "P"], default=["S", "A+", "A", "P"])
    with col2:
        risk_filter = st.multiselect(
            "价格风险", options=["低风险", "中风险", "高风险"],
            default=["低风险", "中风险", "高风险"])
    with col3:
        channel_filter = st.multiselect(
            "主推渠道",
            options=["自然优先", "直播优先", "双渠道均衡", "高风险不推"],
            default=["自然优先", "直播优先", "双渠道均衡"])
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
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in preds:
                md = render_single_report_markdown(p)
                zf.writestr(f"{p.info.style_id}_报告.md", md)
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

    g: GradeResult = p.grade
    color_map = {"S": "#ef4444", "A+": "#f59e0b", "A": "#22c55e", "P": "#6b7280"}
    c = color_map.get(g.grade, "#6b7280")
    st.markdown(
        f"### {p.info.style_id}"
        f"  <span style='color:gray;font-weight:normal'>综合评分</span>"
        f"  **{g.final_score:.1f} / 10**"
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

    with right:
        with st.expander("📝 基本信息", expanded=False):
            st.write(f"品类：{p.info.category}   售价：¥{p.info.price:.0f}   季节：{p.info.season}")
            st.write(f"FAB：{str(p.info.fab_text or p.info.fab_description)[:300]}…")

        st.subheader("🎯 10特征BARS评分")
        feat_rows = []
        for key, f in p.features.features.items():
            feat_rows.append({"特征": f.name, "分数": f.score})
        df_feat = pd.DataFrame(feat_rows).sort_values("分数")
        st.bar_chart(df_feat, x="特征", y="分数", horizontal=True, color="#6366f1", height=360)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👥 人设投票")
            st.metric("平均评分", f"{p.voting.avg_score:.2f}")
            st.metric("S率", f"{p.voting.s_rate:.0%}")
            st.metric("反对率", f"{p.voting.opposition_rate:.0%}")
            with st.expander("支持/反对理由"):
                if p.voting.reasons_support:
                    st.markdown("**支持理由 TOP：**")
                    for r in p.voting.reasons_support[:3]:
                        st.write(f"  ✓ {r}")
                if p.voting.reasons_oppose:
                    st.markdown("**反对理由 TOP：**")
                    for r in p.voting.reasons_oppose[:3]:
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
            "得分": [p.voting.avg_score, p.channels.natural_score,
                     p.channels.live_score, p.channels.perceived_value],
        })
        st.bar_chart(engine_df, x="引擎", y="得分", color="#a855f7", height=250, use_container_width=True)

        st.subheader("💡 改款建议")
        for i, s in enumerate(g.improvement_suggestions or [], 1):
            st.write(f"{i}. {s}")

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
            if "真实销售结果" not in df_truth.columns and "真实销量" not in df_truth.columns:
                st.error("❌ 缺少「真实销售结果」或「真实销量」列")
            else:
                col_truth = "真实销售结果" if "真实销售结果" in df_truth.columns else "真实销量"
                st.success(f"✅ 数据读取成功，共{len(df_truth)}行")
                with st.expander("预览"):
                    st.dataframe(df_truth.head(5), use_container_width=True)
                truth_map_ready = {}
                for _, r in df_truth.iterrows():
                    sid = str(r["款式编号"])
                    val = r[col_truth]
                    if isinstance(val, str):
                        digits = "".join(ch for ch in val if ch.isdigit())
                        v = float(digits) if digits else 0.0
                    else:
                        v = float(val) if pd.notna(val) else 0.0
                    truth_map_ready[sid] = v
        except Exception as e:
            st.error(f"Excel读取失败：{e}")

    # ---------- 按钮1：基础回测（沿用旧薄包装） ----------
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        do_backtest = st.button("🔬 基础回测（仅对比Spearman）",
                                disabled=(not preds or not truth_map_ready))
    with col_b2:
        do_3loop = st.button("🤖 运行3Loop核心优化内核 + 残差分离",
                             type="primary",
                             disabled=(not preds or not truth_map_ready or len(preds) < 8),
                             help="至少8款数据才能启动Lasso人设分布拟合")

    if do_backtest and truth_map_ready and preds:
        with st.spinner("跑基础回测中…"):
            bt = run_backtest(preds, truth_map_ready, cfg)
        st.markdown(render_backtest_summary_markdown(bt))
        if getattr(bt, "calibrated_weights", None) is not None:
            st.download_button(
                "⬇️ 下载基础回测校准权重YAML（旧版稀疏回归）",
                data=bt.calibrated_weights,
                file_name="legacy_feature_weights.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

    if do_3loop and truth_map_ready and preds:
        with st.spinner("3Loop校准运行中（Loop1→Loop2→Loop3→残差分离，20秒）…"):
            # 调用 PredictionPipeline.run_backtest_calibration（spec §9）
            # 内部封装：build_history_df + run_all_loops + 残差归一化
            # 产物写 brand_cfg.calibrated_dir（下次评估自动加载，越用越准）
            try:
                pl = PredictionPipeline(brand_id=brand_cfg.brand_id, llm_backend="mock")
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
                                st.write(f"- {o.get('style_id','?')}：真实销量 / 预测倍数 ≈ {o.get('ratio','?')}")
                with col_under:
                    st.metric("🔵 不及预期款 (ε < μ-2σ)", f"{len(rd.underperformers)} 个")
                    if rd.underperformers:
                        with st.expander("查看详情（复盘改进方向）"):
                            for u in rd.underperformers:
                                st.write(f"- {u.get('style_id','?')}：预测高估 ≈ {u.get('ratio','?')}")
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
