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
from src.types import (
    StyleInfo, FullPrediction, GradeResult, BrandConfig,
    SALES_QTY_COL_ALIASES, SALES_LABEL_COL_ALIASES,
    SELL_THROUGH_COL_ALIASES, MANUAL_GRADE_COL_ALIASES,
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


# ========== 面包屑组件（公共） ==========
def render_breadcrumb(*, suffix: str | None = None) -> None:
    parts = []
    if st.session_state.batch_name:
        parts.append(f"{st.session_state.batch_name}")
    if suffix:
        parts.append(suffix)
    st.markdown(
        "<div class='breadcrumb'>"
        + " <span class='sep'>·</span> ".join(parts)
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

    # ===== 统一裁剪缩略图的 CSS（object-fit: cover 填满 160×200）=====
    st.markdown("""
    <style>
    .thumb-wrap { display:flex; flex-direction:column; align-items:center; gap:4px; }
    .thumb-box { width:160px; height:200px; overflow:hidden; border-radius:6px;
                 display:flex; align-items:center; justify-content:center;
                 background:#f5f2ec; }
    .thumb-box img { width:100%; height:100%; object-fit:cover; }
    .thumb-caption { font-size:12px; color:#8a857d; }
    .info-kv { display:flex; gap:24px; flex-wrap:wrap; margin:4px 0 8px; }
    .info-kv div { font-size:13px; }
    .info-kv b { color:#8699ab; font-weight:500; margin-right:4px; }
    </style>
    """, unsafe_allow_html=True)

    # ===== 上方：统一规格图片 + 基本信息（并排）=====
    def _thumb_b64(ipath: str, size=(160, 200)) -> str:
        """PIL 中心裁剪 + 缩放到统一规格 → base64 data URI（绕开 file:// 安全限制）"""
        try:
            img = Image.open(ipath).convert("RGB")
            w, h = img.size
            tw, th = size
            ratio = tw / th
            if w / h > ratio:
                nw = int(h * ratio); left = (w - nw) // 2
                img = img.crop((left, 0, left + nw, h))
            else:
                nh = int(w / ratio); top = (h - nh) // 2
                img = img.crop((0, top, w, top + nh))
            img = img.resize(size, Image.LANCZOS)
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/jpeg;base64,{b64}" />'
        except Exception:
            return ""

    left, right = st.columns([1.1, 1.4])
    with left:
        st.subheader("📷 款式图片")
        image_paths = st.session_state.get("image_paths_map", {}).get(p.info.style_id, [])
        if image_paths:
            _thumbs = st.columns(min(len(image_paths), 3))
            for i, ipath in enumerate(image_paths[:3]):
                with _thumbs[i % 3]:
                    _img = _thumb_b64(str(ipath))
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

        # —— 优势 / 劣势（图片下方 inline）——
        strength_items = g.strengths or []
        weakness_items = g.weaknesses or []
        if strength_items or weakness_items:
            st.markdown("**🏆 优劣势速览**")
            if strength_items:
                st.caption("✅ 优势：" + " · ".join(strength_items[:4]))
            if weakness_items:
                st.caption("⚠️ 劣势：" + " · ".join(weakness_items[:4]))

    with right:
        # —— 基本信息（补全所有可用字段）——
        st.markdown("**📝 基本信息**")
        info_items = [
            ("款号", p.info.style_id),
            ("品类", p.info.category or "—"),
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

        # —— 10 特征 BARS 评分 —— 带因果链（anchors 分段染色 + VLM reason）
        st.subheader("🎯 10特征BARS评分 · 因果链")
        st.caption("每个特征的分数段颜色对应品牌 anchors 里的销量影响判断：🔴高风险 → 🌿销冠潜力")

        # 从 brand_cfg.features_bars 读 anchors，构建分数段→颜色+标签的映射
        _bars_cfg = brand_cfg.features_bars.get("features", {}) if brand_cfg else {}
        _risk_buckets = [
            # (max_score_exclusive, color, bucket_label)
            (2.5,  "#d64545", "🔴 高风险"),
            (4.5,  "#e8923d", "🟠 偏低风险"),
            (6.5,  "#a8b5c4", "⚪ 常规"),
            (8.5,  "#6ca060", "🟢 高销区"),
            (10.1, "#3d7d3c", "🌿 销冠潜力"),
        ]

        feat_rows = []
        for key, f in p.features.features.items():
            # 找 anchors 里匹配这个分数的 bucket（兼容 dict 和 list）
            _raw_anchors = _bars_cfg.get(key, {}).get("anchors") or {}
            if isinstance(_raw_anchors, dict):
                anchors = list(_raw_anchors.values())
            else:
                anchors = list(_raw_anchors)
            anchor_label = ""
            anchor_impact = ""
            for a in anchors:
                lo, hi = a.get("range", [0, 0])
                if lo <= f.score <= hi:
                    anchor_label = a.get("label", "")
                    ad = a.get("description", "")
                    # 从 description 里提取销量影响短语（含"销冠"/"主推"/"高销"/"风险"等关键词）
                    for kw in ["销冠", "主推", "高销", "标杆", "潜力", "风险", "退货", "销量", "不稳定"]:
                        if kw in ad:
                            # 找到含关键词的子句
                            for clause in ad.split("；") + ad.split("，"):
                                if kw in clause:
                                    anchor_impact = clause.strip("。，；")
                                    break
                            if anchor_impact:
                                break
                    if not anchor_impact and ad:
                        anchor_impact = ad[:40]
                    break

            # 按分数确定颜色
            _color = _risk_buckets[-1][1]
            _bucket_label = ""
            for (mx, col, bl) in _risk_buckets:
                if f.score < mx:
                    _color, _bucket_label = col, bl
                    break

            feat_rows.append({
                "key": key,
                "特征": f.name,
                "分数": f.score,
                "VLM理由": f.reason or "（无）",
                "锚点档位": anchor_label,
                "销量影响": anchor_impact,
                "风险段": _bucket_label,
                "颜色": _color,
            })

        df_feat = pd.DataFrame(feat_rows).sort_values("分数")

        # Plotly 水平条形图（分段染色 + 标签）
        fig = go.Figure()
        fig.add_bar(
            y=df_feat["特征"],
            x=df_feat["分数"],
            orientation="h",
            marker_color=df_feat["颜色"],
            text=[f"{s:.1f}  {lbl}" for s, lbl in zip(df_feat["分数"], df_feat["锚点档位"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>"
                          "VLM 理由：%{customdata[0]}<br>"
                          "档位：%{customdata[1]}<br>"
                          "销量影响：%{customdata[2]}<extra></extra>",
            customdata=df_feat[["VLM理由", "锚点档位", "销量影响"]],
        )
        fig.update_layout(
            height=max(280, len(df_feat) * 32),
            margin=dict(l=80, r=180, t=10, b=10),
            xaxis=dict(range=[0, 11], showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
            uniformtext_minsize=10,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 风险段图例
        _legend_html = "｜".join(
            f'<span style="color:{c}">■</span> {bl}'
            for (_, c, bl) in _risk_buckets
        )
        st.markdown(f"<div style='font-size:0.8em;color:#8a857d'>{_legend_html}</div>",
                    unsafe_allow_html=True)

        # VLM 理由 + 销量影响展开区
        with st.expander("🔍 每个特征的 VLM 判断理由 + 销量影响", expanded=False):
            for _, row in df_feat.iterrows():
                _impact_html = f"<span style='color:{row['颜色']};font-weight:bold'>【{row['风险段']}】{row['销量影响']}</span>" if row["销量影响"] else ""
                st.markdown(f"**{row['特征']} · {row['分数']:.1f}/10 · {row['锚点档位']}**  {_impact_html}",
                            unsafe_allow_html=True)
                st.caption(row["VLM理由"])
                st.divider()

    # ===== 下方：四列指标（图片下方一屏看完）=====
    # 分数含义说明：0-10 分（越高越好），人设/渠道/价格三个引擎独立打分后加权
    _score_tip = "【分数含义】0-10 分（越高越好），三类引擎独立打分后加权融合为最终综合分"
    c1, c2, c3, c4 = st.columns([1, 1, 1.3, 1.3])
    with c1:
        st.subheader("👥 人设投票")
        st.caption(_score_tip)
        st.metric("加权总分", f"{p.voting.weighted_score:.2f}")
        st.metric("支持率", f"{p.voting.support_rate:.0%}")
        st.metric("反对率", f"{p.voting.opposition_rate:.0%}")
    with c2:
        st.subheader("🛒 双渠道")
        st.caption(_score_tip)
        st.metric("自然分", f"{p.channels.natural_score:.1f}")
        st.metric("直播分", f"{p.channels.live_score:.1f}")
        st.metric("感知价值", f"{p.channels.perceived_value:.1f}")
    with c3:
        st.subheader("⚖️ 三引擎综合")
        st.caption("横轴：4 个引擎；纵轴 0-10 分；柱子越高该项越强")
        engine_df = pd.DataFrame({
            "引擎": ["人设", "自然", "直播", "价格"],
            "得分": [p.voting.weighted_score, p.channels.natural_score,
                     p.channels.live_score, p.channels.perceived_value],
        })
        st.bar_chart(engine_df, x="引擎", y="得分", color="#9fb893",
                     height=180, use_container_width=True)
        # 价格风险 + 百分位
        _risk_map = {"低风险": "🟢 低风险", "中风险": "🟡 中风险", "高风险": "🔴 高风险"}
        _risk_text = _risk_map.get(p.channels.price_risk, p.channels.price_risk)
        st.caption(f"💰 {_risk_text} ｜ 价格百分位 {p.channels.price_percentile:.0%}（越低越有价格竞争力）")
    with c4:
        st.subheader("💡 洞察")
        st.info((g.consumer_insights or "暂无人设洞察")[:250])

    # —— 30 人设真实投票详情（横跨整行，默认展开）——
    votes = p.voting.votes
    if votes:
        # 按 final_score 排序（高分→低分，两边极端在前，中间在最后）
        votes_sorted = sorted(votes, key=lambda v: v.final_score)
        # 聚合洞察
        std = p.voting.score_std
        if std >= 2.0:
            _divergence = f"⚠️ **分歧较大**（σ={std:.1f}）—— 不同人设意见明显分裂"
        elif std >= 1.2:
            _divergence = f"⚖️ **意见有一定分歧**（σ={std:.1f}）"
        else:
            _divergence = f"✅ **意见高度一致**（σ={std:.1f}）"

        with st.expander(f"👁 30 人设真实投票详情（{len(votes)} 人设）", expanded=True):
            st.markdown(f"**{_divergence}**")

            # Top 买/反对理由
            br = p.voting.top_buy_reasons[:5] if p.voting.top_buy_reasons else []
            or_ = p.voting.top_oppose_reasons[:5] if p.voting.top_oppose_reasons else []
            if br or or_:
                c_r1, c_r2 = st.columns(2)
                with c_r1:
                    st.markdown("✅ **高频购买理由**")
                    for r in br:
                        st.markdown(f"- {r}")
                with c_r2:
                    st.markdown("❌ **高频反对理由**")
                    for r in or_:
                        st.markdown(f"- {r}")

            # 分歧最大的人设
            if p.voting.high_divergence_personas:
                st.caption("🔥 **分歧最大的人设**：" + "、".join(p.voting.high_divergence_personas[:5]))

            st.divider()

            # 人设卡片列表（一行 2 张）
            card_cols = st.columns(2)
            for i, v in enumerate(votes_sorted):
                with card_cols[i % 2]:
                    # 分数染色
                    if v.final_score >= 7:
                        _sc = "#6ca060"
                    elif v.final_score >= 4:
                        _sc = "#a8b5c4"
                    else:
                        _sc = "#d64545"
                    _veto_tag = " 🔴**否决**" if v.vetoed else ""
                    # 人设名 + 综合分（小卡片样式）
                    _html = f"""<div style="border-left:4px solid {_sc};padding:8px 12px;margin:4px 0;background:#faf8f5;border-radius:4px">
                        <b>{v.persona_name}</b> <span style="color:{_sc};font-size:1.1em;font-weight:bold">{v.final_score:.1f}/10</span>{_veto_tag}
                    </div>"""
                    st.markdown(_html, unsafe_allow_html=True)

                    # 展开看各层判断
                    with st.expander(f"看 {v.persona_name} 的具体判断"):
                        for layer_id, layer_score in v.layer_scores.items():
                            layer_reason = v.layer_reasons.get(layer_id, "")
                            st.markdown(f"**层 {layer_id} · {layer_score:.1f}/10**")
                            if layer_reason:
                                st.caption(layer_reason)
                        if v.opposing_reason:
                            st.markdown("**反对理由**")
                            st.caption(v.opposing_reason)
                    st.markdown("---")

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

            if not col_truth:
                st.error(f"❌ 未找到销量列（支持的别名：{SALES_QTY_COL_ALIASES}）")
            else:
                st.success(f"✅ 识别到销量列「{col_truth}」，共{len(df_truth)}行")
                if col_label:
                    st.info(f"🏷️ 识别到销售标签列「{col_label}」（爆/旺/平/滞）")

                # 预览用副本（重命名只影响展示）
                _df_preview = df_truth.rename(columns=_rename_map) if _rename_map else df_truth
                with st.expander("预览（前5行）"):
                    st.dataframe(_df_preview.head(5), use_container_width=True)

                truth_map_ready = {}
                for _, r in df_truth.iterrows():
                    sid = str(r["款式编号"])
                    truth_map_ready[sid] = parse_sales_value(r[col_truth])

                non_zero = sum(1 for v in truth_map_ready.values() if v > 0)
                if non_zero == 0:
                    st.warning("⚠️ 识别到的销量值全为 0 — 请检查列是否匹配正确")
        except Exception as e:
            st.error(f"Excel读取失败：{e}")

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
