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

import base64
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

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
from src.ui.components import (
    render_breadcrumb,
    thumb_b64,
    anchor_color,
    score_color_v1490,
    vote_sort_key as _vote_sort_key_fn,
    decision_label,
    chip_row,
    render_diagnostic_banner,
)
from src.types import (
    StyleInfo, FullPrediction, GradeResult, BrandConfig,
    SALES_QTY_COL_ALIASES, SALES_LABEL_COL_ALIASES,
    SELL_THROUGH_COL_ALIASES, MANUAL_GRADE_COL_ALIASES,
    MAIN_PUSH_COL_ALIASES, LIVE_STREAM_COL_ALIASES, STYLE_ID_COL_ALIASES,
    find_aliased_column, parse_sales_value, parse_sales_label_to_qty,
    parse_grade_to_norm, safe_float,
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

# ==================== 全局莫兰迪主题 CSS ====================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">

<style>
/* ---------- Base ---------- */
:root {
  --bg: #f4f2ed;
  --surface: #ffffff;
  --surface-soft: #f8f6f1;
  --border: #e6e3dc;
  --border-light: #edeae4;
  --text: #2e2b27;
  --text-muted: #8a857d;
  --text-subtle: #b3aea3;
  --accent: #a8b5c4;       /* 雾霾蓝 */
  --accent-deep: #8699ab;
  --accent-soft: #cdd5de;
  --sage: #9fb893;         /* 鼠尾草绿（中饱和） */
  --sage-deep: #708a63;
  --sage-soft: #e4ece1;
  --oat: #c8a272;          /* 燕麦驼（中饱和） */
  --oat-deep: #9a7c56;
  --oat-soft: #f0e6d6;
  --rose: #b98888;         /* 豆沙玫瑰（中饱和） */
  --rose-deep: #966868;
  --rose-soft: #ecdcdc;
  --purple: #b5a9c4;       /* 烟紫 */
  --gray: #9a9489;         /* 高级灰（P级/中性） */
  --gray-deep: #746f65;
  --radius: 12px;
  --shadow-sm: 0 1px 2px rgba(46,43,39,0.04);
  --shadow-md: 0 4px 16px rgba(46,43,39,0.06);
}

html, body, [class*="css"] {
  font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif !important;
  color: var(--text) !important;
  background: var(--bg) !important;
  font-feature-settings: "tnum";
}

/* ---------- Streamlit container overrides ---------- */
.stApp { background: var(--bg); }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: none; }

/* ================================================================
   Sidebar — 只做软美化，**绝对不碰 Streamlit 的默认布局**
   Streamlit 内部自己管 sidebar 的 display/width/flex，
   我们只改颜色、边框、字体、内边距这些不影响布局的属性。
   之前 [class*="sidebar" i] 通配符对所有嵌套层都强制 flex，
   结果把 sidebar 内部内容容器的布局全炸了 —— 彻底删掉！
================================================================ */

/* 1. 顶层 sidebar 容器 — 只改颜色/边框，不改布局 */
[data-testid="stSidebar"],
[class*="stSidebarContainer"],
[class*="SidebarContainer"],
.stSidebar {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* 2. 旧版 Streamlit 的 sidebar testid（1.35~1.50）*/
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
[data-testid="stSidebar"] .block-container { padding: 1.25rem 1.25rem 2rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.01em;
}

/* 3. 新版 Streamlit sidebar（1.55+）— 用 emotion class 匹配 */
[class*="stSidebarContainer"] [class*="block-container"],
[class*="SidebarContainer"] [class*="block-container"] {
  padding: 1.25rem 1.25rem 2rem !important;
}

/* 4. 防止 sidebar 被误伤 — 全局重置（安全兜底）*/
/* Streamlit 有时用 data-test-script-state="initial" 会给元素加 display:none
   但我们不应该强制覆盖 sidebar 的 display，让 Streamlit 自己决定 */

/* ---------- Typography ---------- */
h1 {
  font-family: 'Fraunces', 'Inter', 'PingFang SC', serif !important;
  font-weight: 600 !important;
  font-size: 1.65rem !important;
  letter-spacing: -0.015em !important;
  color: var(--text) !important;
  padding-bottom: 0.25rem !important;
  border-bottom: none !important;
}
h2 {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 1.1rem !important;
  letter-spacing: -0.01em !important;
  color: var(--text) !important;
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}
h3 {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.95rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.02em !important;
  text-transform: uppercase !important;
}
.stMarkdown p { line-height: 1.65; }
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 { color: var(--text-muted) !important; font-weight: 500 !important; }

/* ---------- Cards ---------- */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stContainer"]) {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  box-shadow: var(--shadow-sm);
}
div[data-testid="stContainer"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.25rem 1.5rem !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stButton"] > button {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  padding: 0.5rem 1.25rem !important;
  transition: all 0.18s ease !important;
  box-shadow: none !important;
}
.stButton > button:hover, [data-testid="stButton"] > button:hover {
  border-color: var(--accent) !important;
  background: var(--surface-soft) !important;
  color: var(--accent-deep) !important;
  transform: translateY(-1px);
}
.stButton > button[data-baseweb="button"][kind="primary"],
button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}
.stButton > button[data-baseweb="button"][kind="primary"]:hover {
  background: var(--accent-deep) !important;
  border-color: var(--accent-deep) !important;
  color: #fff !important;
}

/* ---------- Inputs ---------- */
input, select, textarea {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(168,181,196,0.15) !important;
}
div[data-baseweb="base-input"], div[data-baseweb="select"] {
  border-radius: 10px !important;
}

/* ---------- Selectbox / Multiselect ---------- */
[data-baseweb="select"] > div {
  border-radius: 10px !important;
  border-color: var(--border) !important;
  background: var(--surface) !important;
}
[data-baseweb="select"]:hover > div {
  border-color: var(--accent) !important;
}
ul[data-baseweb="menu"] {
  border-radius: 10px !important;
  border-color: var(--border) !important;
}
li[data-baseweb="option"][aria-selected="true"] {
  background: var(--surface-soft) !important;
  color: var(--accent-deep) !important;
}

/* ---------- Metric ---------- */
[data-testid="stMetricValue"] {
  font-family: 'Fraunces', serif !important;
  font-weight: 600 !important;
  color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
[data-testid="stMetricDelta"] { color: var(--sage-deep) !important; }

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"] table {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  background: var(--surface-soft) !important;
  color: var(--text-muted) !important;
  font-weight: 500 !important;
  border-bottom: 1px solid var(--border) !important;
  text-transform: none !important;
}
[data-testid="stDataFrame"] td {
  border-bottom: 1px solid var(--border-light) !important;
}
[data-testid="stDataFrame"] tr:hover td {
  background: var(--surface-soft) !important;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  margin-bottom: 0.75rem;
}
[data-testid="stExpander"] summary {
  padding: 0.75rem 1rem !important;
  font-weight: 500 !important;
  color: var(--text) !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--accent-deep) !important;
}
[data-testid="stExpander"] details[open] summary {
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stExpander"] > div > div { padding: 1rem !important; }

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0 !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 0 !important;
  border: none !important;
  background: transparent !important;
  padding: 0.6rem 1.25rem !important;
  color: var(--text-muted) !important;
  font-weight: 500 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--accent-deep) !important;
}

/* ---------- Info / Warning / Error boxes ---------- */
.stAlert, [data-baseweb="notification"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  padding: 0.85rem 1rem !important;
}
.element-container .stAlert { border-left: 3px solid var(--accent) !important; }
.stInfo { background: #eef0f3 !important; border-left-color: var(--accent-deep) !important; }
.stSuccess { background: var(--sage-soft) !important; border-left-color: var(--sage-deep) !important; }
.stWarning { background: var(--oat-soft) !important; border-left-color: var(--oat-deep) !important; }
.stError { background: var(--rose-soft) !important; border-left-color: var(--rose-deep) !important; }

/* ---------- Divider ---------- */
hr, [data-testid="stDivider"] {
  border-color: var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ---------- Progress ---------- */
div[data-baseweb="progress-bar"] {
  background: var(--border-light) !important;
  border-radius: 10px !important;
  height: 8px !important;
}
div[data-baseweb="progress-bar"] > div {
  background: var(--accent) !important;
  border-radius: 10px !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface-soft) !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; }

/* ---------- Radio / Checkbox ---------- */
[data-baseweb="radio"] div[role="radiogroup"], [data-baseweb="checkbox"] {
  gap: 0.5rem !important;
}
[data-baseweb="radio"] label, [data-baseweb="checkbox"] label {
  color: var(--text-muted) !important;
  font-weight: 500 !important;
  padding: 0.35rem 0.85rem !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  margin-right: 0.25rem !important;
}
[data-baseweb="radio"] label:hover, [data-baseweb="checkbox"] label:hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
}
[data-baseweb="radio"] label[data-checked="true"], [data-baseweb="checkbox"] label[data-checked="true"] {
  background: var(--surface-soft) !important;
  border-color: var(--accent) !important;
  color: var(--text) !important;
}

/* ---------- Slider ---------- */
div[data-baseweb="slider"] > div > div:first-child {
  background: var(--accent) !important;
}

/* ---------- Breadcrumb ---------- */
.breadcrumb {
  font-family: 'Fraunces', serif;
  font-size: 0.8rem;
  color: var(--text-subtle);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.breadcrumb .sep { color: var(--border); margin: 0 0.4rem; }

/* ---------- Hero Banner ---------- */
.hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 1.75rem 0 1.5rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.hero h1 {
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
  font-size: 2rem !important;
  letter-spacing: -0.02em !important;
}
.hero-sub {
  color: var(--text-muted);
  font-size: 0.85rem;
  margin-top: 0.25rem;
  font-weight: 400;
}

/* ---------- Stepper ---------- */
.stepper {
  display: flex;
  gap: 1.5rem;
  margin: 1.25rem 0 0.5rem;
}
.stepper-step {
  flex: 1;
  padding: 1.1rem 1.25rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stepper-num {
  font-family: 'Fraunces', serif;
  font-size: 1.1rem;
  color: var(--accent-deep);
  font-weight: 600;
}
.stepper-label {
  font-weight: 500;
  margin-top: 0.15rem;
  font-size: 0.85rem;
}
.stepper-active { border-color: var(--accent); background: var(--surface-soft); }
.stepper-done { border-color: var(--sage); }

/* ---------- Grade Badges ---------- */
.badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.78rem;
  letter-spacing: 0.02em;
}
.badge-s { background: var(--rose-soft); color: var(--rose-deep); }
.badge-aplus { background: var(--oat-soft); color: var(--oat-deep); }
.badge-a { background: var(--sage-soft); color: var(--sage-deep); }
.badge-p { background: #eceae6; color: var(--text-muted); }
.badge-info { background: #eef0f3; color: var(--accent-deep); }

/* ---------- Score Ring ---------- */
.score-ring {
  font-family: 'Fraunces', serif;
  font-size: 3.25rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1;
  letter-spacing: -0.02em;
}
.score-ring-sub {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 0.35rem;
}

/* ---------- Bar feature rows ---------- */
.feature-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border-light);
}
.feature-name {
  width: 180px;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text);
}
.feature-bar-wrap {
  flex: 1;
  height: 8px;
  background: var(--border-light);
  border-radius: 4px;
  overflow: hidden;
}
.feature-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.4s ease;
}
.feature-score {
  width: 52px;
  text-align: right;
  font-family: 'Fraunces', serif;
  font-weight: 600;
  color: var(--text);
  font-size: 0.9rem;
}

/* ---------- Sidebar header ---------- */
.sb-brand {
  padding: 1rem 0.25rem 0.75rem;
}
.sb-brand-name {
  font-family: 'Fraunces', serif;
  font-size: 1.3rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text);
}
.sb-brand-meta {
  font-size: 0.72rem;
  color: var(--text-subtle);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.15rem;
}
.sb-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 600;
  background: var(--surface-soft);
  color: var(--text-muted);
  border: 1px solid var(--border);
  margin-top: 0.4rem;
}
.sb-badge.active {
  background: var(--surface-soft);
  color: var(--accent-deep);
  border-color: var(--accent);
}

.sb-nav-item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.55rem 0.8rem;
  border-radius: 10px;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-muted);
  cursor: pointer;
  margin-bottom: 2px;
  border: 1px solid transparent;
  transition: all 0.15s ease;
}
.sb-nav-item:hover {
  color: var(--text);
  background: var(--surface-soft);
}

/* ---------- Gallery ---------- */
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.75rem;
  margin-top: 0.5rem;
}
.gallery img {
  border-radius: 10px;
  border: 1px solid var(--border);
  width: 100%;
  aspect-ratio: 3/4;
  object-fit: cover;
  transition: transform 0.2s ease;
}
.gallery img:hover { transform: scale(1.02); }

/* ---------- Stat Card Row ---------- */
.stat-row {
  display: flex;
  gap: 1rem;
  margin: 1.25rem 0;
}
.stat-card {
  flex: 1;
  padding: 1.25rem 1.35rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}
.stat-value {
  font-family: 'Fraunces', serif;
  font-size: 1.85rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text);
}
.stat-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  letter-spacing: 0.03em;
  margin-top: 0.1rem;
  text-transform: uppercase;
  font-weight: 500;
}

/* ---------- Filter Pills ---------- */
.pill-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.75rem 0;
}
.pill-group .stMultiSelect { min-width: 200px; }

/* ---------- Spearman Gauge ---------- */
.gauge-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
}
.gauge-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 500;
  margin-bottom: 0.3rem;
}
.gauge-value {
  font-family: 'Fraunces', serif;
  font-size: 1.75rem;
  font-weight: 600;
}
.gauge-delta {
  font-size: 0.78rem;
  font-weight: 600;
  margin-top: 0.2rem;
}
.gauge-delta.pos { color: var(--sage-deep); }
.gauge-delta.neg { color: var(--rose-deep); }

/* ---------- Hide Streamlit chrome（最安全的最小干预）---------- */
/*
   ⚠️ 绝对不能碰的元素（Streamlit 新版把交互入口都放在 header 里了）：
     - <header> 元素          → 里面有 hamburger + sidebar toggle
     - [data-testid="stHeader"] → 同上
     - [data-testid="stToolbar"]
     - #MainMenu
   之前写过 header { visibility: hidden } 结果把 hamburger 和折叠按钮一起藏了！
*/
footer { visibility: hidden; }              /* 安全 — 底部 "Made with Streamlit" */
[data-testid="stHeader"] {
  background: transparent !important;       /* 安全 — 只去背景色 */
  border-bottom: none !important;          /* 安全 — 只去底边线 */
}
/* 也去掉 Streamlit 默认的顶栏内边距，让 sidebar toggle 贴顶 */
[data-testid="stAppViewContainer"] > div:first-child > header,
[data-testid="stHeader"] > div {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}

/* ---------- Hamburger + 折叠按钮 — 颜色美化（不碰布局/可见性）---------- */
/* 多层 fallback 覆盖 Streamlit 1.35~1.64 的不同 DOM 结构。
   只改颜色/背景/边框/圆角/hover — 完全不碰 display/visibility/position/width。
   匹配不到就 skip（Streamlit 自己的按钮照样工作），匹配到就美化。 */
#MainMenu button,
button[data-testid="baseButton-headerNoPadding"],
button[kind="icon"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarButton"] button,
[data-testid="stMainMenu"] button,
.stSidebarToggle button {
  color: var(--text-muted) !important;
  background: transparent !important;
  border-radius: 6px !important;
  transition: all 0.15s ease !important;
}
#MainMenu button:hover,
button[data-testid="baseButton-headerNoPadding"]:hover,
button[kind="icon"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarButton"] button:hover {
  color: var(--accent-deep) !important;
  background: var(--surface-soft) !important;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-subtle); }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：品牌选择 & Header ---
# 设计原则：用 st.empty() 占位符实现"视觉上 header 在上，逻辑上 selectbox 先渲染"
# selectbox 返回值 = 真实选中品牌 → brand_cfg 从返回值加载 → 100% 同步，不依赖 session_state
available_brands = list_available_brands()
if "brand_id" not in st.session_state:
    st.session_state.brand_id = available_brands[0] if available_brands else "mipo"

# 品牌 selectbox 只显示漂亮名（自动去除括号内容）
@st.cache_data(show_spinner=False, ttl=3600)
def _brand_label(bid: str) -> str:
    try:
        bc = load_brand_profile(bid)
        name = bc.brand_name
    except Exception:
        name = bid
    # 去除英文括号 (...) 和中文括号 （...） 及其中内容
    name = re.sub(r'[（(][^)）]*[)）]', '', name).strip()
    return name

# 加载 BrandConfig（贯穿整个会话共享）
# ⚠️ 参数名不能以下划线开头！Streamlit @st.cache_data 会忽略 `_` 开头的参数作为缓存键，
# 导致不同 brand_id 命中同一个缓存 → brand_cfg 永远是第一次缓存的值（典型 Bug）
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load_brand(brand_id_key: str) -> BrandConfig:
    return load_brand_profile(brand_id_key)

# ===== 1. 在 sidebar 最顶部创建 header 占位符 =====
# Streamlit 会把它渲染在 sidebar 第一个位置（视觉上最顶部）
# 之后调用 _header_ph.markdown(...) 会填充它，而不是追加新元素
_header_ph = st.sidebar.empty()

# ===== 2. selectbox 先渲染 —— 返回值就是真实选中的品牌 =====
# 位置：占位符下方（因为占位符先创建）
brand_id = st.sidebar.selectbox(
    "品牌",
    options=available_brands,
    index=available_brands.index(st.session_state.brand_id)
    if st.session_state.brand_id in available_brands else 0,
    label_visibility="collapsed",
    format_func=_brand_label,
    help="每个品牌独立维护：30人设+BARS量表+品类价格带+S/A/P阈值+3Loop校准产物"
)

# ===== 3. brand_cfg 直接从 selectbox 返回值加载 —— 与下拉框 100% 同步 =====
brand_cfg: BrandConfig = _cached_load_brand(brand_id)

# ===== 4. 如果品牌变了 → 清空旧缓存 + rerun =====
if brand_id != st.session_state.brand_id:
    st.session_state.brand_id = brand_id
    for k in ("preds", "df_input", "style_to_images", "progress_info",
              "style_infos", "image_paths_map", "batch_name",
              "selected_style_id"):
        st.session_state.pop(k, None)
    st.rerun()

n_calibration: int = _count_calibration_rounds(brand_cfg)

# 取历史数据
_hist_batches = 0
_hist_styles = 0
try:
    from src.history_store import HistoryStore
    h = HistoryStore()
    summary = h.get_summary(brand_cfg.brand_id)
    _hist_batches = summary.get("batches", 0)
    _hist_styles = summary.get("styles", 0)
except Exception:
    pass

# ===== 5. 往占位符填充 Header —— 视觉上仍在 sidebar 顶部 =====
# brand_cfg 此时已经是从 selectbox 返回值加载的正确品牌名，零错位
_header_ph.markdown(f"""
<div class="sb-brand">
  <div class="sb-brand-name">{brand_cfg.brand_name}</div>
  <div class="sb-brand-meta">Decision Engine</div>
  <div class="sb-badge">校准 {n_calibration} 轮 · {_hist_batches} 批 / {_hist_styles} 款</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# --- 侧边栏：LLM 模式切换（混合模式为默认 — 视觉本地 Ollama + 文本智谱云端，兼顾速度与质量） ---
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
    index=2,  # 默认：混合模式（智谱 VLM + 本地文本）
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
            _hc_ok, _hc_msg = _OC().health_check(do_probe=True)
        if _llm_backend != "hybrid" and not _key_for_backend:
            st.sidebar.error("❌ 请先粘贴 API Key")
        else:
            try:
                _t_cfg = _AC(max_retries=3, qpm_limit=10000)
                _parts_ok: list[str] = []
                _parts_err: list[str] = []

                # ① 本地 Ollama 健康检查（local / hybrid 都要）
                # do_probe=True 会发一次真实推理请求，验证模型真的能跑
                if _llm_backend in ("local", "hybrid"):
                    _hc_ok, _hc_msg = _OC().health_check(do_probe=True)
                    if not _hc_ok:
                        _parts_err.append(f"🖥️ Ollama: {_hc_msg}（请先安装 Ollama + ollama pull qwen2.5vl:7b + qwen2.5:7b）")
                    else:
                        _parts_ok.append(f"🖥️ Ollama 正常 ({_hc_msg})")
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

st.sidebar.divider()

# --- 莫兰迪风格导航（带选中高亮） ---
st.sidebar.markdown(
    '<div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;'
    'color:#b0ab9f;margin-bottom:0.4rem;padding-left:0.2rem;">导航</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    " ",
    PAGES,
    index=0,
    label_visibility="collapsed",
)

# 从 VERSION 文件动态读版本号
_VERSION_STR = "unknown"
try:
    _VERSION_STR = (ROOT / "VERSION").read_text().strip()
except Exception:
    pass
st.sidebar.markdown(
    f'<div style="font-size:0.7rem;color:#b0ab9f;margin-top:1.25rem;line-height:1.4;">'
    f'v {_VERSION_STR} · '
    f'<a href="https://github.com/Kingbulude/fashion-hit-engine/releases/latest" '
    f'style="color:#8a857e;text-decoration:none;border-bottom:1px solid #e8e5df;">更新</a>'
    f'</div>',
    unsafe_allow_html=True,
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


# render_breadcrumb 已抽到 src/ui/components.py


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
    required_cols = ["款式编号", "版型/设计描述", "售价"]   # 面料成分是可选的，VLM 能从版型/设计描述里识别
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
    render_breadcrumb(suffix="上传批次")

    # Hero banner
    st.markdown(f"""
<div class="hero">
  <div>
    <h1>上传评估批次</h1>
    <div class="hero-sub">准备一批新款，让 VLM + 30 人设给出预测</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:600;color:#8699ab;">{n_calibration or 0}</div>
    <div style="font-size:0.78rem;color:#8a857e;">已校准轮次</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Stepper
    st.markdown("""
<div class="stepper">
  <div class="stepper-step stepper-active">
    <div class="stepper-num">01</div>
    <div class="stepper-label">批次信息</div>
  </div>
  <div class="stepper-step">
    <div class="stepper-num">02</div>
    <div class="stepper-label">款式 Excel</div>
  </div>
  <div class="stepper-step">
    <div class="stepper-num">03</div>
    <div class="stepper-label">图片包</div>
  </div>
  <div class="stepper-step">
    <div class="stepper-num">04</div>
    <div class="stepper-label">开始评估</div>
  </div>
</div>
""", unsafe_allow_html=True)

    batch_name = st.text_input(
        "批次名（必填，方便后续区分）", value=st.session_state.batch_name or "2026春第一批",
        help="建议格式：年份+季节+第N批，如2026春第一批"
    )

    st.markdown("**款式信息表 Excel / CSV**")
    st.caption(
        "列参考：款式编号、版型/设计描述、售价 必填；"
        "面料成分、品类、尺码、颜色、上架季节可选；"
        "（面料成分 VLM 能从版型/设计描述里识别，不需要专门一列）"
        "<br>真实销量回填请在「📉 回测校准」页单独上传，内审分级可在批次总表手动填写。"
    )
    xlsx_file = st.file_uploader("拖放或选择 .xlsx / .csv", type=["xlsx", "csv"])

    st.markdown("**图片包（两种方式二选一）**")
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

                # 只匹配款号列（必填），其余回测字段（销量/内审分级/售罄率/主推/直播）
                # 在「📉 回测校准」页单独上传，不在预测阶段解析
                sid_col = find_aliased_column(list(df.columns), STYLE_ID_COL_ALIASES) or "款式编号"
                if sid_col not in df.columns:
                    st.error(f"❌ 未找到款号列（支持：{' / '.join(STYLE_ID_COL_ALIASES)}），无法解析批次")
                    st.stop()

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
                    # 季节：兼容多种列名（上架季节/季节/上市季节）
                    season_col_candidates = ["上架季节", "季节", "上市季节"]
                    season = ""
                    for _sc in season_col_candidates:
                        if _sc in df.columns:
                            _v = str(row[_sc]).strip() if pd.notna(row[_sc]) else ""
                            if _v and _v != "nan":
                                season = _v; break
                    # 年份：兼容多种列名，兜底 batch_name 正则提取
                    year_col_candidates = ["年份", "上架年份", "上市年份"]
                    year = ""
                    for _yc in year_col_candidates:
                        if _yc in df.columns:
                            _v = str(row[_yc]).strip() if pd.notna(row[_yc]) else ""
                            if _v and _v != "nan":
                                year = _v; break
                    if not year:
                        _m = re.search(r"(20\d{2})", str(st.session_state.get("batch_name", "")))
                        if _m: year = _m.group(1)
                    if not season:
                        _m = re.search(r"(春|夏|秋|冬|春夏|秋冬|春季|夏季|秋季|冬季)",
                                       str(st.session_state.get("batch_name", "")))
                        if _m: season = _m.group(1)
                    fab_parts = []
                    if size: fab_parts.append(f"尺码：{size}")
                    fab_content = str(row.get("面料成分", "")).strip() if "面料成分" in df.columns else ""
                    if fab_content and fab_content != "nan":
                        fab_parts.append(f"面料成分：{fab_content}")
                    design_desc = str(row.get("版型/设计描述", "")).strip()
                    if design_desc and design_desc != "nan":
                        fab_parts.append(f"版型/设计：{design_desc}")
                    if color: fab_parts.append(f"颜色：{color}")
                    if season: fab_parts.append(f"季节：{season}")
                    fab_text = "\n".join(fab_parts) if fab_parts else ""

                    style_infos.append(StyleInfo(
                        style_id=sid, category=cat,
                        price=price, season=season, year=year, fab_description=fab_text,
                        # 回测字段默认空值 — 在「📉 回测校准」页单独上传时填充
                        sales_qty=0, manual_grade="", sell_through_pct=0.0,
                        is_main_push=False, is_live_stream=False,
                    ))
                    imgs = style_to_images.get(sid, [])
                    image_paths_map[sid] = [str(p) for p in imgs]

                st.session_state.style_infos = style_infos
                st.session_state.image_paths_map = image_paths_map
                st.session_state.preds = []
                st.session_state.batch_finalized = False
                st.session_state.progress_info = {
                    "current": 0, "total": len(style_infos),
                    "stage": "开始评估…", "failed": {},
                }
                st.success("✅ 批次已提交！请切换到「📋 批次总表」页面查看进度和结果。")

    with col2:
        if st.button("🧹 清空本次输入", use_container_width=True):
            for key in ("preds", "df_input", "style_to_images", "progress_info",
                        "style_infos", "image_paths_map",
                        "batch_finalized", "history_batch_id"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ============================================================
# 页面 2：📋 批次总表
# ============================================================
def render_page_summary():
    render_breadcrumb(suffix="批次总表")
    _batch = st.session_state.batch_name or ""
    st.markdown(f"""
<div class="hero">
  <div>
    <h1>批次总表{(' · ' + _batch) if _batch else ''}</h1>
    <div class="hero-sub">每款综合分 · 分级 · 主推渠道 · 快速筛选</div>
  </div>
</div>
""", unsafe_allow_html=True)

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

    # v1.4.95+: 三层错误处理 UI — 批次聚合诊断（VLM fallback / Phase2-3 失败 / calibration 加载错误）
    try:
        from src.calibration_loader import load_calibration as _load_cal
        _cal = _load_cal(brand_cfg.calibrated_dir)
        render_diagnostic_banner(
            predictions=preds,
            calibration_load_errors=_cal.load_errors or None,
            calibration_loop_applied=_cal.loop_applied or None,
            calibration_spearman_gains=_cal.spearman_gains or None,
        )
    except Exception:
        render_diagnostic_banner(predictions=preds)

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
            options=["S", "A+", "A", "P", "风险"],
            default=["S", "A+", "A", "P", "风险"])
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

    st.dataframe(dff, use_container_width=True, height=620, hide_index=True)

    st.subheader("📥 导出")
    col_a, col_b = st.columns(2)
    with col_a:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            dff.to_excel(writer, index=False, sheet_name="批次总表")
        st.download_button(
            "⬇️ 导出总表 Excel", data=bio.getvalue(),
            file_name=f"{brand_cfg.brand_id}_{st.session_state.batch_name}_总表.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_b:
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


# ============================================================
# 页面 3：🔍 单款详情报告
# ============================================================
def render_page_detail():
    render_breadcrumb(suffix="单款详情")
    st.markdown(f"""
<div class="hero">
  <div>
    <h1>单款详情报告</h1>
    <div class="hero-sub">VLM 特征 · 30 人设投票 · 双渠道评分 · 改款建议</div>
  </div>
</div>
""", unsafe_allow_html=True)
    preds: list[FullPrediction] = st.session_state.get("preds", [])
    selected = st.session_state.get("selected_style_id", "")
    if not preds:
        st.info("还没有评估结果。请先到「📤 上传批次」运行评估。")
        return

    style_ids = [p.info.style_id for p in preds]
    selected = st.selectbox("款号", options=style_ids,
                            index=style_ids.index(selected) if selected in style_ids else 0)
    p = next(x for x in preds if x.info.style_id == selected)

    # v1.4.95+: 三层错误处理 UI — 如果这款有 VLM fallback / Phase2-3 失败，在这里提醒
    render_diagnostic_banner(prediction=p)

    g: GradeResult = p.grade
    color_map = {"S": "#b98888", "A+": "#c8a272", "A": "#9fb893", "P": "#9a9489"}
    c = color_map.get(g.grade, "#9a9489")
    st.markdown(
        f"### {p.info.style_id}"
        f"  <span style='color:#8a857d;font-weight:normal'>综合评分</span>"
        f"  **{g.final_score:.1f} / 100**"
        f"  <span style='color:#8a857d;font-size:13px'>（置信度 {g.confidence:.0%}）</span>"
        f"  <span style='background:{c};color:#fff;padding:3px 10px;border-radius:6px;font-weight:600;letter-spacing:0.02em;'>{g.grade}级</span>"
        f"  <span style='color:#8699ab;'>主推：{g.recommended_channel}</span>",
        unsafe_allow_html=True,
    )

    # ===== 统一裁剪缩略图的 CSS（左列内3张放大横排）=====
    st.markdown("""
    <style>
    .thumb-wrap { display:flex; flex-direction:column; align-items:center; }
    .thumb-box { width:100%; height:520px; overflow:hidden; border-radius:8px;
                 display:flex; align-items:center; justify-content:center;
                 background:#f5f2ec; border:1px solid #e0dcd5; }
    .thumb-box img { width:100%; height:100%; object-fit:cover; }
    .thumb-caption { font-size:12px; color:#8a857d; text-align:center; padding-top:4px; }
    .info-kv { display:flex; gap:24px; flex-wrap:wrap; margin:4px 0 8px; }
    .info-kv div { font-size:13px; }
    .info-kv b { color:#8699ab; font-weight:500; margin-right:4px; }
    </style>
    """, unsafe_allow_html=True)

    # ===== 左右分栏：款式图片+优劣势 (1.3) + 基本信息+BARS (1.4) =====

    left, right = st.columns([1.6, 1.4])
    with left:
        st.subheader("📷 款式图片")
        # —— 优劣势速览（上移到图片上方，与右列「基本信息」对齐）——
        strength_items = g.strengths or []
        weakness_items = g.weaknesses or []
        if strength_items or weakness_items:
            st.subheader("🏆 优劣势速览")
            if strength_items:
                st.caption("✅ 优势：" + " · ".join(strength_items[:4]))
            if weakness_items:
                st.caption("⚠️ 劣势：" + " · ".join(weakness_items[:4]))

        image_paths = st.session_state.get("image_paths_map", {}).get(p.info.style_id, [])
        if image_paths:
            _thumbs = st.columns(min(len(image_paths), 3))
            for i, ipath in enumerate(image_paths[:3]):
                with _thumbs[i % 3]:
                    _img = thumb_b64(str(ipath))
                    if _img:
                        st.markdown(
                            f'<div class="thumb-wrap">'
                            f'<div class="thumb-box">{_img}</div>'
                            f'<div class="thumb-caption">图 {i+1}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(f"无法加载图 {i+1}")
            if len(image_paths) > 3:
                st.caption(f"（共 {len(image_paths)} 张，仅显示前 3 张）")
        else:
            st.caption("（无图片）")

    with right:
        st.subheader("📝 基本信息")
        # v1.4.91+: 品类 id → 中文名映射（category_registry 维护）
        _cat_id_to_name: dict[str, str] = {
            c["id"]: c.get("name", c["id"])
            for c in (brand_cfg.category_registry.get("categories", []) if brand_cfg else [])
        }
        _cat_display = _cat_id_to_name.get(p.info.category, p.info.category or "—")
        info_items = [
            ("款号", p.info.style_id),
            ("品类", _cat_display),
            ("售价", f"¥{p.info.price:.0f}" if p.info.price else "—"),
            ("季节", p.info.season or "—"),
            ("年份", p.info.year or "—"),
        ]
        kv_html = '<div class="info-kv">'
        for k, v in info_items:
            kv_html += f"<div><b>{k}</b>{v}</div>"
        kv_html += "</div>"
        st.markdown(kv_html, unsafe_allow_html=True)
        if p.info.fab_description:
            st.caption(f"**FAB**：{p.info.fab_description}")
        st.divider()

        # —— 10 特征视觉锚定（右列内，位置不动）——
        st.subheader("🎯 10特征视觉锚定 · VLM 纯客观描述")
        st.caption("v1.4.90+：分数是视觉锚定档的区间中点（档1=[1,2]中点=1.5 ... 档5=[9,10]中点=9.5），仅描述视觉分类，不含销量预判。颜色区分锚定档，不代表好坏。")

        _bars_cfg = brand_cfg.features_bars.get("features", {}) if brand_cfg else {}

        # v1.4.90+: 颜色按 anchor_level 分（5 档，每档一种中性色，无好坏语义）
        _anchor_colors = {
            1: "#c85a5a",   # 档1
            2: "#d68c45",   # 档2
            3: "#a8b5c4",   # 档3 中性灰
            4: "#5a8a9a",   # 档4
            5: "#3d7d8a",   # 档5
        }

        feat_rows = []
        for key, f in p.features.features.items():
            _raw_anchors = _bars_cfg.get(key, {}).get("anchors") or {}
            if isinstance(_raw_anchors, dict):
                anchors = list(_raw_anchors.values())
            else:
                anchors = list(_raw_anchors)
            anchor_label = ""
            anchor_level = f.anchor_level
            for a in anchors:
                if a.get("label") and (
                    (f.score != 0 and a.get("range", [0, 0])[0] <= f.score <= a.get("range", [0, 0])[1])
                    or (f.score != 0 and a.get("range", [0, 0])[0] <= f.score <= a.get("range", [0, 0])[1])
                ):
                    anchor_label = a["label"]
                    break
            if not anchor_label and anchor_level is not None:
                anchor_label = f"档{anchor_level}"

            _color = _anchor_colors.get(anchor_level or 3, "#a8b5c4")
            _visual_desc = f.visual_description or ""

            feat_rows.append({
                "key": key,
                "特征": f.name,
                "锚定点": f.score,
                "视觉档": f"档{anchor_level}" if anchor_level else "未判断",
                "锚定标签": anchor_label,
                "视觉观察": _visual_desc[:80] + ("..." if len(_visual_desc) > 80 else ""),
                "颜色": _color,
            })

        df_feat = pd.DataFrame(feat_rows).sort_values("锚定点")

        # Plotly 水平条形图（按 anchor_level 着色，无好坏语义）
        fig = go.Figure()
        fig.add_bar(
            y=df_feat["特征"],
            x=df_feat["锚定点"],
            orientation="h",
            marker_color=df_feat["颜色"],
            text=[f"档{row['视觉档'].replace('档','')} {row['锚定标签']}" for _, row in df_feat.iterrows()],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "视觉档：%{customdata[0]}<br>"
                "锚定标签：%{customdata[1]}<br>"
                "视觉观察：%{customdata[2]}<extra></extra>"
            ),
            customdata=df_feat[["视觉档", "锚定标签", "视觉观察"]],
        )
        fig.update_layout(
            height=max(280, len(df_feat) * 32),
            margin=dict(l=80, r=180, t=10, b=10),
            xaxis=dict(range=[0, 11], showgrid=False, zeroline=False,
                       tickvals=[1.5, 3.5, 5.5, 7.5, 9.5],
                       ticktext=["档1", "档2", "档3", "档4", "档5"]),
            yaxis=dict(showgrid=False),
            showlegend=False,
            uniformtext_minsize=10,
        )
        st.plotly_chart(fig, use_container_width=True)

        # anchor 档图例（中性色，无好坏）
        _legend_html = "｜".join(
            f'<span style="color:{col}">■</span> 档{lv}'
            for lv, col in sorted(_anchor_colors.items())
        )
        st.markdown(f"<div style='font-size:0.8em;color:#8a857d'>{_legend_html}</div>",
                    unsafe_allow_html=True)

        # v1.4.90+: 视觉观察展开区（纯客观描述，无销量判断）
        with st.expander("🔍 VLM 视觉观察细节（VLM 纯客观描述，人设 LLM 基于此独立判断好不好卖）", expanded=False):
            for _, row in df_feat.iterrows():
                _level_html = f"<span style='color:{row['颜色']};font-weight:bold'>【{row['视觉档']} · {row['锚定标签']}】</span>"
                st.markdown(f"**{row['特征']}** {_level_html}", unsafe_allow_html=True)
                st.caption(row["视觉观察"] or "（无视觉描述）")
                st.divider()

    st.divider()

    # ===== 下方：三列指标（全宽）=====

    # engine weights：校准后的权重 > 默认权重
    # v1.4.96.4 重构时意外遗漏下面两行定义 → _ew / defw 未定义
    _ew = getattr(brand_cfg, "engine_weights", None) or {}
    defw = getattr(brand_cfg, "default_engine_weights", {}) if brand_cfg else {}

    _w_persona = float(_ew.get("persona_voting", defw.get("persona_voting", 0.35)))
    _w_channel = float(_ew.get("channel_scoring", defw.get("channel_scoring", 0.30)))
    _w_price = float(_ew.get("price_value", defw.get("price_value", 0.35)))
    _w_natural = _w_channel * 0.5   # 自然/直播平分 channel 权重
    _w_live = _w_channel * 0.5

    c1, c2, c3 = st.columns([1.0, 1.0, 1.5])
    with c1:
        st.subheader("👥 购买决策投票")
        st.caption(f"加权权重 {_w_persona:.0%} ｜ 30 类目标人群的综合判断")
        v = p.voting
        _sc = score_color_v1490(v.weighted_score)
        s_pct = int(v.support_rate * 100)
        w_pct = int(getattr(v, "wait_rate", max(0.0, 1.0 - v.support_rate - v.opposition_rate)) * 100)
        o_pct = int(v.opposition_rate * 100)
        # 核心决策一句话
        if s_pct >= 60:
            _verdict = "**主推款**"
            _verdict_col = "#3d7d3c"
        elif o_pct >= 50:
            _verdict = "**谨慎上**"
            _verdict_col = "#d64545"
        else:
            _verdict = "**观望款**"
            _verdict_col = "#a8b5c4"
        st.markdown(
            f"### 综合判定 <span style='color:{_verdict_col};font-size:1.1em;font-weight:bold'>"
            f"{_verdict}</span>"
            f"<span style='margin-left:10px;color:{_sc};font-weight:bold'>"
            f"加权分 {v.weighted_score:.1f}</span>",
            unsafe_allow_html=True,
        )
        # 三段比 bar（支持/观望/反对）
        # s_pct / w_pct / o_pct 已在上方计算过，直接复用
        st.markdown(
            f"""
            <div style="display:flex;height:18px;border-radius:6px;overflow:hidden;margin:4px 0;border:1px solid #e0dcd5">
              <div style="width:{s_pct}%;background:#3d7d3c"></div>
              <div style="width:{w_pct}%;background:#a8b5c4"></div>
              <div style="width:{o_pct}%;background:#d64545"></div>
            </div>
            <div style="font-size:0.8em;color:#8a857d;display:flex;justify-content:space-between">
              <span style="color:#3d7d3c">■ 支持 {s_pct}%</span>
              <span style="color:#a8b5c4">■ 观望 {w_pct}%</span>
              <span style="color:#d64545">■ 反对 {o_pct}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"分歧度 σ={v.score_std:.1f}")

    with c2:
        st.subheader("🛒 渠道 · 价格")
        st.caption(f"渠道权重 {_w_channel:.0%} ｜ 价格权重 {_w_price:.0%}")
        # 三行带色指标
        rows = [
            ("自然分", p.channels.natural_score, _w_natural),
            ("直播分", p.channels.live_score, _w_live),
            ("感知价值", p.channels.perceived_value, _w_price),
        ]
        for name, score, w in rows:
            sc = score_color_v1490(score)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;padding:3px 0'>"
                f"<span style='color:#8a857d;font-size:0.9em'>{name} <span style='font-size:0.8em;color:#b3aea3'>({w:.0%})</span></span>"
                f"<span style='color:{sc};font-size:1.15em;font-weight:bold'>{score:.1f}</span>"
                f"</div>", unsafe_allow_html=True,
            )

    with c3:
        st.subheader("⚖️ 四引擎加权对比")
        st.caption("条长 = 引擎分 × 权重，直接看对最终分的贡献度")

        engine_data = [
            ("人群决策", p.voting.weighted_score, _w_persona, "#b98888"),
            ("自然流量", p.channels.natural_score, _w_natural, "#c8a272"),
            ("直播带货", p.channels.live_score, _w_live, "#9fb893"),
            ("价格价值", p.channels.perceived_value, _w_price, "#b5a9c4"),
        ]
        # Plotly 横向 bar（按加权贡献度排序）
        fig = go.Figure()
        engine_data_sorted = sorted(engine_data, key=lambda x: -x[1] * x[2])
        for name, raw, w, col in engine_data_sorted:
            weighted = raw * w
            fig.add_bar(
                y=[name], x=[weighted], orientation="h",
                marker_color=col,
                text=[f"贡献 {weighted:.2f}（{raw:.1f} × {w:.0%}）"],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>"
                              "原始分: %{customdata[0]}/10<br>"
                              "权重: %{customdata[1]:.0%}<br>"
                              "加权贡献: %{x:.2f}<extra></extra>",
                customdata=[[raw, w]],
            )
        fig.update_layout(
            height=200,
            margin=dict(l=80, r=220, t=10, b=10),
            xaxis=dict(range=[0, 10 * max(_w_persona, _w_natural, _w_live, _w_price)],
                       showgrid=False, zeroline=False, title="加权贡献度"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # —— 洞察区（单独一整行，比之前宽）——
    st.subheader("💡 综合洞察")
    st.info((g.consumer_insights or "暂无人设洞察")[:500])

    # —— 人设购买决策投票（基于 MIPO 真实消费人群画像）——
    votes = p.voting.votes
    if votes:
        # layer_id → 中文层名
        _layer_name_map: dict[str, str] = {}
        if brand_cfg and hasattr(brand_cfg, "decision_structure"):
            for _l in brand_cfg.decision_structure.layers:
                _n = (_l.name or _l.id).split("（")[0].split("(")[0].strip()
                _layer_name_map[_l.id] = _n

        # yaml 人设查询表
        _yaml_personas = getattr(brand_cfg, "personas", None) or []
        _p_lookup: dict[str, dict] = {
            str(pp.get("persona_id", "")): pp for pp in _yaml_personas
        } if _yaml_personas else {}

        # yaml 轴 id → 中文名
        _axis_name_map: dict[str, str] = {}
        try:
            import yaml as _yl
            _p_yaml_path = ROOT / "brand_profiles" / (
                brand_cfg.brand_id if brand_cfg else "mipo"
            ) / "personas.yaml"
            if _p_yaml_path.exists():
                _p_ax = _yl.safe_load(_p_yaml_path.read_text()).get("identity_axes_definition", {})
                for _ak in _p_ax:
                    for _av in _p_ax[_ak]:
                        _axis_name_map[_av.get("id", "")] = _av.get("name", "")
        except Exception:
            pass

        # 聚合洞察
        std = p.voting.score_std
        if std >= 2.0:
            _divergence = f"⚠️ **分歧较大**（σ={std:.1f}）—— 不同人群意见明显分裂"
        elif std >= 1.2:
            _divergence = f"⚖️ **意见有一定分歧**（σ={std:.1f}）"
        else:
            _divergence = f"✅ **意见高度一致**（σ={std:.1f}）"

        with st.expander(f"👥 {len(votes)} 类人群真实购买决策投票", expanded=True):
            st.markdown(f"**{_divergence}**")

            # 总体决策分布
            _n_buy = sum(1 for v in votes if v.final_score >= 7)
            _n_wait = sum(1 for v in votes if 4 <= v.final_score < 7)
            _n_opp = sum(1 for v in votes if v.final_score < 4)
            _nc1, _nc2, _nc3 = st.columns(3)
            _nc1.markdown(f"✅ **会买** {_n_buy} 类人群")
            _nc2.markdown(f"⏳ **观望** {_n_wait} 类人群")
            _nc3.markdown(f"❌ **不买** {_n_opp} 类人群")

            # Top 买/反对理由
            br = p.voting.top_buy_reasons[:5] if p.voting.top_buy_reasons else []
            or_ = p.voting.top_oppose_reasons[:5] if p.voting.top_oppose_reasons else []
            if br or or_:
                c_r1, c_r2 = st.columns(2)
                with c_r1:
                    st.markdown("✅ **高频购买触发点**")
                    for r in br:
                        st.markdown(f"- {r}")
                with c_r2:
                    st.markdown("❌ **高频否决雷区**")
                    for r in or_:
                        st.markdown(f"- {r}")

            st.divider()

            # 卡片：按 yaml weight 降序（核心人群优先）
            # v1.4.96.4 重构时丢失这行赋值 → votes_sorted 未定义
            # vote_sort_key 签名: (vv, persona_lookup) → 必须传 _p_lookup
            votes_sorted = sorted(votes, key=lambda v: _vote_sort_key_fn(v, _p_lookup))

            # 一行 2 张
            card_cols = st.columns(2)
            for i, v in enumerate(votes_sorted):
                with card_cols[i % 2]:
                    pp = _p_lookup.get(v.persona_id, {})
                    # 人设名：优先 yaml 里的真实名字
                    real_name = str(pp.get("name") or v.persona_name or v.persona_id)
                    # 三维轴中文标签
                    axes_dict = pp.get("axes", {}) if isinstance(pp, dict) else {}
                    _scene = _axis_name_map.get(axes_dict.get("scene", ""), "")
                    _aesth = _axis_name_map.get(axes_dict.get("aesthetic", ""), "")
                    _price = _axis_name_map.get(axes_dict.get("price", ""), "")
                    axis_tags_html = ""
                    for _tag in [_scene, _aesth, _price]:
                        if _tag:
                            axis_tags_html += (
                                f'<span style="background:#eef0f3;color:#6d7b8a;'
                                f'padding:1px 7px;border-radius:10px;font-size:0.7em;margin:0 2px">'
                                f'{_tag}</span>'
                            )

                    # 决策徽章
                    _em, _lbl, _col = decision_label(v.final_score)
                    _weight = float(pp.get("weight", 0.0)) * 100 if pp else 0
                    _weight_html = (
                        f'<span style="font-size:0.7em;color:#8a857d;'
                        f'margin-left:6px">权重 {_weight:.1f}%</span>'
                    )

                    # fab / 颜色 / 否决 三行画像
                    _fab = pp.get("fab_focus", []) if isinstance(pp, dict) else []
                    _cols_pref = pp.get("color_preference", []) if isinstance(pp, dict) else []
                    _veto_dim = pp.get("child_veto_dimensions", []) if isinstance(pp, dict) else []


                    # chip_row: 签名 (title, items, *, col_bg, col_txt, limit) —
                    # col_bg / col_txt 是 keyword-only，不能位置传
                    # v1.4.96.4 重构改成 keyword-only 后，3 处调用全没同步 → 连续 5 个 tag 崩
                    _fab_html = chip_row("在意", _fab, col_bg="#e8f0ea", col_txt="#3d7d3c")
                    _col_html = chip_row("偏好颜色", _cols_pref, col_bg="#eef3f9", col_txt="#5772a0")
                    _veto_html = chip_row("否决雷区", _veto_dim, col_bg="#fdecea", col_txt="#b14a4a")

                    # 主体理由：优先最长的层理由，否则反对理由，否则兜底
                    _main_reason = ""
                    for lr in v.layer_reasons.values():
                        if lr and len(lr) > len(_main_reason):
                            _main_reason = lr
                    if not _main_reason and v.opposing_reason:
                        _main_reason = v.opposing_reason
                    if not _main_reason:
                        # 兜底：从决策反推一句话
                        _lbl_for = decision_label(v.final_score)[1]
                        _main_reason = f"该人群综合判断为「{_lbl_for}」，LLM 未返回详细理由。"

                    _card_html = f"""
<div style="border-left:4px solid {_col};padding:10px 14px;margin:8px 0;background:#faf8f5;border-radius:6px">
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
    <b style="font-size:1.02em">{real_name}</b>
    {_weight_html}
  </div>
  <div style="margin:3px 0">{axis_tags_html}</div>
  <div style="margin:4px 0">
    <span style="background:{_col};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600">
      {_em} {_lbl}
    </span>
    {"<span style='color:#d64545;font-size:0.8em;margin-left:4px'>🔴 触发否决</span>" if v.vetoed else ""}
  </div>
  {_fab_html}
  {_col_html}
  {_veto_html}
  {"<div style='margin-top:6px;font-size:0.82em;color:#4a4540'><b>判断理由：</b>" + _main_reason + "</div>" if _main_reason else ""}
</div>"""
                    st.markdown(_card_html, unsafe_allow_html=True)

                    # 各决策层详情 —— 只展示层中文名 + 消费者心理活动（不展示分数）
                    with st.expander(f"📄 该人群三层判断详情"):
                        _all_layer_ids = list(v.layer_scores.keys()) or list(v.layer_reasons.keys())
                        if not _all_layer_ids and brand_cfg:
                            _all_layer_ids = [l.id for l in brand_cfg.decision_structure.layers]
                        for layer_id in _all_layer_ids:
                            layer_reason = v.layer_reasons.get(layer_id, "")
                            layer_cn = _layer_name_map.get(layer_id, layer_id)
                            if layer_reason:
                                st.markdown(f"**{layer_cn}**")
                                st.caption(layer_reason)
                        if v.opposing_reason:
                            st.markdown("**反对理由**")
                            st.caption(v.opposing_reason)
                        if (
                            not v.opposing_reason
                            and not any(v.layer_reasons.values())
                        ):
                            st.caption("（该人群无详细理由）")

    # —— 改款建议 ——
    st.subheader("🔧 改款建议")
    if g.improvements:
        for i, s in enumerate(g.improvements[:5], 1):
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
    render_breadcrumb(suffix=f"校准轮次 {n_calibration or 0}")
    st.markdown(f"""
<div class="hero">
  <div>
    <h1>回测校准</h1>
    <div class="hero-sub">上传带真实销量的历史数据 → 3Loop 核心优化内核 → Spearman 自动提升</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:600;color:#8699ab;">{n_calibration or 0}</div>
    <div style="font-size:0.78rem;color:#8a857e;">已完成校准轮次</div>
  </div>
</div>
""", unsafe_allow_html=True)

    preds: list[FullPrediction] = st.session_state.get("preds", [])

    # —— 历史校准数据累积面板（v1.4.81: 跨 session 不丢）——
    _acc_path = Path(brand_cfg.calibrated_dir) / "history_accumulated.csv"
    _acc_n = 0
    _acc_df: pd.DataFrame | None = None
    if _acc_path.exists():
        try:
            _acc_df = pd.read_csv(_acc_path)
            _acc_n = len(_acc_df)
        except Exception:
            pass

    _acc_col1, _acc_col2, _acc_col3 = st.columns([1.2, 1, 1])
    with _acc_col1:
        st.markdown(
            f"**📦 历史校准数据** 累积 **{_acc_n}** 款"
            + ("（≥30 样本量门槛，Loop3 引擎权重可拟合）"
               if _acc_n >= 30 else f"（还差 {max(0, 30 - _acc_n)} 款到 30 样本门槛）")
        )
    with _acc_col2:
        if _acc_df is not None:
            st.download_button(
                "⬇️ 下载累积 CSV",
                data=_acc_df.to_csv(index=False).encode(),
                file_name=f"history_accumulated_{brand_cfg.brand_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with _acc_col3:
        _reset = st.button("🗑️ 清空历史", use_container_width=True)
        if _reset and _acc_path.exists():
            _acc_path.unlink()
            st.success("已清空历史累积，下次回测校准会从头开始。")

    # 导入外部 CSV（合并去重，用于迁移/备份恢复）
    _up = st.file_uploader("⬆️ 导入历史 CSV（合并去重）", type=["csv"],
                           help="格式同 history_accumulated.csv，按 style_id 合并（新覆盖旧）")
    if _up is not None:
        try:
            _up_df = pd.read_csv(_up)
            # —— Schema 预验证 ——
            _f01_f10 = [f"F{i:02d}" for i in range(1, 11)]
            _p01_p30 = [f"P{i:02d}" for i in range(1, 31)]
            _engine_cols = ["persona_score", "channel_score", "price_value_score"]
            _min_required = ["style_id", "sales"] + _f01_f10 + _p01_p30 + _engine_cols
            _missing_import = [c for c in _min_required if c not in _up_df.columns]
            if _missing_import:
                st.error(
                    f"❌ 导入的 CSV 缺少必需列：{_missing_import[:10]}{'...' if len(_missing_import) > 10 else ''}\n\n"
                    f"**3Loop 校准不是只需要 style_id + sales！**\n\n"
                    f"完整的 history_accumulated.csv 需要 **56 列**：\n"
                    f"• `style_id` — 款号\n"
                    f"• `F01~F10` — VLM 视觉特征分（10 列）\n"
                    f"• `P01~P30` — 人设投票分（30 列）\n"
                    f"• `persona_score`, `channel_score`, `price_value_score` — 三大引擎\n"
                    f"• `natural_score`, `live_score` — 双渠道\n"
                    f"• `sales` — 真实销量\n"
                    f"• `grade_norm` — 内审分级归一化\n\n"
                    f"**你导入的 CSV 只有 {len(_up_df.columns)} 列，不是预测产物表。**\n\n"
                    f"👉 正确做法：先到 **📤 上传批次** 页跑预测，系统会自动生成完整累积 CSV。"
                )
            else:
                # 列齐了，但特征值有没有全 NaN？
                _feat_ok = _up_df[_f01_f10].notna().all(axis=1).sum()
                if _feat_ok < max(8, len(_up_df) // 2):
                    st.warning(
                        f"⚠️ 导入的 CSV 有 {len(_up_df)} 行，但只有 {_feat_ok} 行的 VLM 特征列有值 — "
                        f"可能影响 Loop1 校准效果。"
                    )
                # 合并
                if _acc_df is not None:
                    _merged = pd.concat([_acc_df, _up_df]).drop_duplicates(
                        subset=["style_id"], keep="last"
                    )
                else:
                    _merged = _up_df
                _acc_path.parent.mkdir(parents=True, exist_ok=True)
                _merged.to_csv(_acc_path, index=False)
                st.success(f"✅ 合并完成：{len(_acc_df) if _acc_df is not None else 0} + "
                          f"{len(_up_df)} → {len(_merged)} 款")
                _acc_df, _acc_n = _merged, len(_merged)
        except Exception as e:
            st.error(f"导入失败：{e}")

    if _acc_df is not None and len(_acc_df) > 0:
        with st.expander(f"预览（前 5 行，共 {_acc_n} 款）"):
            st.dataframe(_acc_df.head(5), use_container_width=True)

    if not preds:
        st.info("ℹ️ 当前没有新的评估结果。请先在「📤 上传批次」跑评估，"
                "或直接用累积 CSV 重新跑回测校准。")

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
            col_label = find_aliased_column(list(df_truth.columns), SALES_LABEL_COL_ALIASES)

            # ---- 展示层重命名：让表头语义更清晰 ----
            # 只改 display 名，不改 df_truth 实际列名（后续 iterrows 还在用原列名）
            _rename_map: dict[str, str] = {}
            if col_truth:
                # 用户 Excel 可能写 "真实销售结果" / "销量" / "累计销量" → 统一展示成 "累计销量结果"
                if col_truth != "累计销量结果":
                    _rename_map[col_truth] = "累计销量结果"
            if col_label:
                # 用户 Excel 可能写 "销售情况" / "爆旺平滞" → 统一展示成 "上架30天销售情况"
                if col_label != "上架30天销售情况":
                    _rename_map[col_label] = "上架30天销售情况"

            # v1.4.98: 也识别内审分级列
            col_grade = find_aliased_column(list(df_truth.columns), MANUAL_GRADE_COL_ALIASES)
            if col_grade:
                if col_grade != "内审分级":
                    _rename_map[col_grade] = "内审分级"

            if not col_truth and not col_label:
                st.error(
                    f"❌ 未找到销量列或销售标签列（支持的销量别名：{SALES_QTY_COL_ALIASES}，"
                    f"标签别名：{SALES_LABEL_COL_ALIASES}）"
                )
            else:
                _n_cols_recognized = sum(1 for c in [col_truth, col_label, col_grade] if c)
                st.success(
                    f"✅ 识别到 {_n_cols_recognized}/3 列 — "
                    + (f"销量「{col_truth}」" if col_truth else "")
                    + (" · " if col_truth and col_label else "")
                    + (f"标签「{col_label}」" if col_label else "")
                    + (" · " if (col_truth or col_label) and col_grade else "")
                    + (f"分级「{col_grade}」" if col_grade else "")
                    + f"，共 {len(df_truth)} 行"
                )

                # 预览用副本（重命名只影响展示）
                _df_preview = df_truth.rename(columns=_rename_map) if _rename_map else df_truth
                with st.expander("预览（前5行）"):
                    st.dataframe(_df_preview.head(5), use_container_width=True)

                # —— 构建 sales_lookup（v1.4.98: 数值优先、标签兜底）——
                # 对每行：
                #   1) 如果销量列有非零数值 → 用数值（最精确）
                #   2) 否则如果有销售标签列 → 用标签映射值兜底（保证 Spearman 能算）
                #   3) 都没有 → 0
                truth_map_ready = {}
                for _, r in df_truth.iterrows():
                    sid = str(r["款式编号"])
                    sales_val = parse_sales_value(r[col_truth]) if col_truth else 0.0
                    if sales_val <= 0 and col_label:
                        # 销量列缺失/为 0 → 用销售标签兜底
                        sales_val = parse_sales_label_to_qty(r[col_label])
                    truth_map_ready[sid] = sales_val

                non_zero = sum(1 for v in truth_map_ready.values() if v > 0)
                st.info(
                    f"📊 sales_lookup: {non_zero}/{len(truth_map_ready)} 款有非零值"
                    + ("（部分来自销售标签兜底）" if col_label and non_zero < len(truth_map_ready) else "")
                )
                if non_zero == 0:
                    st.warning("⚠️ 识别到的销量值全为 0 — 请检查列是否匹配正确")

                # —— 构建 grade_lookup（v1.4.98: 内审分级 → Loop3 第 4 引擎）——
                grade_lookup_ready: dict[str, str] | None = None
                if col_grade:
                    grade_lookup_ready = {}
                    for _, r in df_truth.iterrows():
                        sid = str(r["款式编号"])
                        gv = parse_grade_to_norm(r[col_grade])
                        if gv is not None:
                            grade_lookup_ready[sid] = str(r[col_grade]).strip().upper()
                    _n_grade = len(grade_lookup_ready)
                    if _n_grade > 0:
                        st.info(f"🏷️ grade_lookup: {_n_grade}/{len(df_truth)} 款有内审分级 → Loop3 会用 grade_norm 第 4 引擎")
                    else:
                        grade_lookup_ready = None
        except Exception as e:
            st.error(f"Excel读取失败：{e}")

    # ---------- 回测校准按钮（3Loop内核：Spearman对比+残差分离）----------
    # v1.4.98.1+: 精确匹配检查 — 累积 CSV 必须真能匹配 truth_map 才算数
    _n_match_in_csv = 0
    if truth_map_ready and _acc_df is not None and len(_acc_df) > 0:
        _n_match_in_csv = int(_acc_df["style_id"].isin(truth_map_ready.keys()).sum())
    _n_match_in_preds = sum(
        1 for p in preds if truth_map_ready and p.info.style_id in truth_map_ready
    ) if preds else 0
    _has_matchable = _n_match_in_preds >= 8 or _n_match_in_csv >= 8

    # 给用户清晰提示 — 区分"没跑过预测" vs "累积 CSV 不匹配" vs "样本不足"
    if truth_map_ready:
        if not preds and _n_match_in_csv == 0:
            # 最常见场景: 只上传了 Excel 销量数据，没跑过任何预测
            st.error("❌ 没有任何可校准的历史预测数据。")
            st.info(
                f"**3Loop 校准需要「预测产物」（VLM 特征 + 人设投票 + 渠道得分）才能运行。**\n\n"
                f"你只上传了 **{len(truth_map_ready)}** 款的真实销量 Excel — 这是「校准目标」，"
                f"还需要「校准素材」：\n\n"
                f"👉 **去 「📤 上传批次」 页，上传这 {len(truth_map_ready)} 款的图片和款号，跑一轮预测，"
                f"再回到这里点 3Loop。**"
            )
        elif not preds and _n_match_in_csv > 0 and _n_match_in_csv < 8:
            st.warning(
                f"⚠️ 累积 CSV 有 {_n_match_in_csv} 款能匹配，但不足 8 款门槛。"
                f"请先跑更多批次，或清理累积 CSV 后重新校准。"
            )
        elif preds and _n_match_in_preds < 8 and _n_match_in_csv < 8:
            st.warning(
                f"⚠️ preds 中 {_n_match_in_preds} 款能匹配，累积 CSV 中 {_n_match_in_csv} 款能匹配 — 都不足 8 款。"
                f"请先跑更多批次。"
            )

    do_3loop = st.button("🤖 运行3Loop核心优化内核 + 残差分离",
                         type="primary",
                         disabled=(not truth_map_ready or not _has_matchable),
                         help="至少 8 款能匹配到真实销量 — "
                              "（session preds ∩ 销量 Excel）或（累积 CSV ∩ 销量 Excel）")

    if do_3loop and truth_map_ready:
        with st.spinner("3Loop校准运行中（Loop1→Loop2→Loop3→残差分离，20秒）…"):
            try:
                from src.pipeline import PredictionPipeline
                pl = PredictionPipeline(
                    brand_id=brand_cfg.brand_id,
                    llm_backend=st.session_state.get("llm_backend", "mock"),
                    api_key=st.session_state.get("api_key_for_backend", "") or None,
                )
                # predictions 为空时 run_backtest_calibration 自动从累积 CSV 加载
                loop_result = pl.run_backtest_calibration(
                    predictions=preds if preds else None,
                    sales_lookup=truth_map_ready,
                    grade_lookup=grade_lookup_ready,
                )
            except RuntimeError as re:
                st.error(f"依赖缺失：{re}")
                loop_result = None
            except Exception as ex:
                st.error(f"3Loop运行失败：{ex}")
                loop_result = None

            if loop_result is None:
                # pipeline.py 可能已自动清理了不兼容的 mock 累积 CSV
                # 这里重新读文件系统确认状态，给用户最精准的提示
                _acc_path_check = Path(brand_cfg.calibrated_dir) / "history_accumulated.csv"
                _acc_still_exists = _acc_path_check.exists()

                if preds:
                    matched = sum(1 for p in preds if p.info.style_id in (truth_map_ready or {}))
                elif _acc_still_exists:
                    _df_now = pd.read_csv(_acc_path_check)
                    matched = int(_df_now["style_id"].isin(truth_map_ready or {}).sum())
                else:
                    matched = 0

                if not preds and not _acc_still_exists:
                    # pipeline 已经判定没有可用数据源（零匹配 + 无 CSV）
                    st.error("❌ 没有任何可校准的历史预测数据。")
                    st.info(
                        f"**3Loop 校准需要 VLM 特征 + 人设投票 + 渠道得分 作为回归信号。**\n\n"
                        f"请按以下步骤操作：\n"
                        f"1️⃣ 去 **📤 上传批次** 页，上传你 Excel 里这 **{len(truth_map_ready)}** 款的图片和款号\n"
                        f"2️⃣ 跑一轮预测（自动调 VLM 分析图片特征 + 人设投票）\n"
                        f"3️⃣ 回到这里点 3Loop — 系统会用你这批预测产物 + 真实销量做校准"
                    )
                elif matched == 0:
                    st.error("❌ 没有款能匹配到真实销量，请检查「款式编号」列是否一致。")
                elif matched < 8:
                    st.error(f"❌ 可匹配的款只有 {matched} 款，Loop2 需要至少 8 款样本，请补充历史批次。")
                else:
                    st.warning("⚠️ 3Loop 跳过：销量信号不足或模型未产生有效增益。请检查 Excel 中的销量值是否合理分布（不能全一样或全 0）。")

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

                # v1.4.50+: 分类指标表（macro F1 + Precision@3）
                st.subheader("🎯 分类指标（业务对齐层）")
                st.caption("Spearman 衡量排序相关性；以下回答更贴近业务：「top 3 里真有几款爆的？」「macro F1 均衡 S/P 类的召回」")
                def _f1_flag(old, new):
                    d = new - old
                    icon = "🟢" if d >= 0.02 else ("⚪" if d >= 0 else "🔴")
                    return f"{icon} {new:.2f}  ({d:+.2f})"
                df_cls = pd.DataFrame([
                    {"步骤": "Loop1 VLM特征校准",
                     "macro F1": _f1_flag(loop_result.loop1.old_f1, loop_result.loop1.new_f1),
                     "Precision@3": _f1_flag(loop_result.loop1.old_p_at_k, loop_result.loop1.new_p_at_k)},
                    {"步骤": "Loop2 人设分布拟合",
                     "macro F1": _f1_flag(loop_result.loop2.old_f1, loop_result.loop2.new_f1),
                     "Precision@3": _f1_flag(loop_result.loop2.old_p_at_k, loop_result.loop2.new_p_at_k)},
                    {"步骤": "Loop3 引擎合成",
                     "macro F1": _f1_flag(loop_result.loop3.old_engine_f1, loop_result.loop3.new_engine_f1),
                     "Precision@3": _f1_flag(loop_result.loop3.old_engine_p_at_k, loop_result.loop3.new_engine_p_at_k)},
                    {"步骤": "Loop3 渠道合成",
                     "macro F1": _f1_flag(loop_result.loop3.old_chan_f1, loop_result.loop3.new_chan_f1),
                     "Precision@3": _f1_flag(loop_result.loop3.old_chan_p_at_k, loop_result.loop3.new_chan_p_at_k)},
                ])
                st.dataframe(df_cls, use_container_width=True, hide_index=True)

                # v1.4.50+: Loop3 引擎权重 + 各引擎 ρ 诊断（含 grade_norm）
                with st.expander("⚖️ Loop3 引擎权重诊断（点击展开）", expanded=False):
                    l3 = loop_result.loop3
                    col_eng, col_rho = st.columns(2)
                    with col_eng:
                        st.markdown("**校准后引擎权重**")
                        eng_labels = {
                            "persona_score": "👥 人设投票",
                            "channel_score": "📺 双渠道合成",
                            "price_value_score": "💰 价格价值",
                            "grade_norm": "⭐ 内审分级 (grade_norm)",
                        }
                        for eng_name, w in l3.engine_weights.items():
                            label = eng_labels.get(eng_name, eng_name)
                            pct = f"{w*100:.1f}%"
                            highlight = " ⭐" if eng_name == "grade_norm" else ""
                            st.progress(w, text=f"{label}{highlight} — {pct}")
                    with col_rho:
                        st.markdown("**各引擎 ρ（与真实销量的排序相关）**")
                        for eng_name, rho in l3.engine_rho.items():
                            label = eng_labels.get(eng_name, eng_name)
                            st.write(f"- {label}：ρ = {rho:+.3f}")
                        # channel ρ 也展示
                        st.markdown("**渠道 ρ**")
                        for ch_name, rho in l3.channel_rho.items():
                            st.write(f"- {ch_name}：ρ = {rho:+.3f}")

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
                        "NO_SIG": ("⚪ 无系统偏差", "#9fb893"),
                        "MODERATE": ("🟡 轻度偏差（关注）", "#c8a272"),
                        "STRONG": ("🔴 强系统偏差（必须复盘）", "#b98888"),
                        "SIGMA_ZERO": ("⚪ 样本过少或全命中预测", "#9a9489"),
                        "ERROR": ("⚫ 计算异常", "#746f65"),
                    }
                    label, color = flag_levels.get(rd.system_bias_flag, ("未知", "#9a9489"))
                    st.markdown(
                        f"<div style='background:{color};color:#fff;padding:10px 14px;"
                        f"border-radius:8px;text-align:center;font-weight:600;letter-spacing:0.02em;'>"
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
