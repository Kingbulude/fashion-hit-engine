"""
生成 fashion-hit-engine 项目汇报 PPT（以 MIPO 蜜扑童装为示例）
用法: python scripts/build_ppt.py
输出: mipo_项目汇报.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ---------- 配色 ----------
C_DARK    = RGBColor(0x1A, 0x1F, 0x36)   # 深海蓝（主背景）
C_PRIMARY = RGBColor(0x25, 0x63, 0xEB)   # 品牌蓝（主色）
C_ACCENT  = RGBColor(0x00, 0xBF, 0xA5)   # 青绿（辅助强调）
C_ORANGE  = RGBColor(0xFF, 0x8C, 0x42)   # 活力橙（次强调）
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT   = RGBColor(0xE5, 0xE7, 0xEB)
C_GRAY    = RGBColor(0x9C, 0xA3, 0xAF)
C_S       = RGBColor(0x10, 0xB9, 0x81)   # S 绿
C_A       = RGBColor(0x3B, 0x82, 0xF6)   # A 蓝
C_P       = RGBColor(0xEF, 0x44, 0x44)   # P 红

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, left, top, width, height, text,
                font_size=18, color=C_WHITE, bold=False,
                align=PP_ALIGN.LEFT, font_name="Microsoft YaHei"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return tb

def add_multi_text(slide, left, top, width, height, lines,
                   font_size=16, color=C_WHITE, bold=False,
                   line_spacing=1.5, align=PP_ALIGN.LEFT):
    """lines: list of dicts with keys: text, size, color, bold, space_before"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", align)
        p.space_before = Pt(ln.get("space_before", 0))
        p.space_after = Pt(ln.get("space_after", 0))
        run = p.add_run()
        run.text = ln["text"]
        run.font.size = Pt(ln.get("size", font_size))
        run.font.color.rgb = ln.get("color", color)
        run.font.bold = ln.get("bold", bold)
        run.font.name = "Microsoft YaHei"
    return tb

def add_rect(slide, left, top, width, height, fill_color=None, line_color=None, line_width=0):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.shadow.inherit = False
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape

def add_rounded_rect(slide, left, top, width, height, fill_color, line_color=None, line_width=0):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.shadow.inherit = False
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape

def add_accent_bar(slide, left, top, height, color=C_PRIMARY):
    return add_rect(slide, left, top, Inches(0.06), height, fill_color=color)

# ---------- Slide Builder ----------
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]  # blank

# =============================================================
# Slide 1 · 封面
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

# 装饰圆
circle1 = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-1.5), Inches(-1), Inches(4), Inches(4))
circle1.fill.solid(); circle1.fill.fore_color.rgb = C_PRIMARY; circle1.fill.transparency = 0.85
circle1.line.fill.background()
circle2 = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11), Inches(5), Inches(5), Inches(5))
circle2.fill.solid(); circle2.fill.fore_color.rgb = C_ACCENT; circle2.fill.transparency = 0.9
circle2.line.fill.background()

# 左侧色块
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_PRIMARY)

add_textbox(s, Inches(1.2), Inches(1.8), Inches(11), Inches(1.2),
            "Fashion Hit Engine", font_size=54, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(3.0), Inches(11), Inches(0.9),
            "服装爆款预测通用引擎 · 项目汇报", font_size=32, color=C_ACCENT, bold=False)
add_textbox(s, Inches(1.2), Inches(4.0), Inches(10), Inches(0.5),
            "—— 以 MIPO 蜜扑童装户外品牌为例", font_size=20, color=C_LIGHT)

# 底部装饰线 + 信息
add_rect(s, Inches(1.2), Inches(5.3), Inches(3), Inches(0.04), fill_color=C_PRIMARY)

add_multi_text(s, Inches(1.2), Inches(5.6), Inches(10), Inches(1.5), [
    {"text": "通用 CORE 引擎 × 品牌适配包 × 3Loop 自优化内核", "size": 16, "color": C_LIGHT},
    {"text": "让选款从「拍脑袋」变成「有数据的消费者投票」", "size": 16, "color": C_GRAY, "space_before": 6},
    {"text": "2026 Spring · v1.0", "size": 14, "color": C_GRAY, "space_before": 20},
])

# =============================================================
# Slide 2 · 目录
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_ACCENT)

add_textbox(s, Inches(1.2), Inches(0.8), Inches(6), Inches(0.8),
            "目录 · Agenda", font_size=40, color=C_WHITE, bold=True)
add_rect(s, Inches(1.2), Inches(1.6), Inches(1.5), Inches(0.05), fill_color=C_PRIMARY)

items = [
    ("01", "为什么做",    "服装行业选款痛点 & MIPO 的具体问题",        C_PRIMARY),
    ("02", "核心思路",    "模拟品牌典型消费者 · 三层决策模型",          C_ACCENT),
    ("03", "系统架构",    "通用 CORE 引擎 + 品牌适配包 + 3Loop 内核",   C_ORANGE),
    ("04", "MIPO 实例",   "30 人设 / 10 特征 / 双渠道 / S·A·P 分级",    RGBColor(0xA7, 0x8B, 0xFA)),
    ("05", "效果验证",    "分级准确率 · 改款建议 · 残差发现",          C_S),
    ("06", "产品形态",    "Web UI 四页闭环 · 一键启动",                C_A),
]

for i, (num, title, desc, color) in enumerate(items):
    row = i // 2
    col = i % 2
    x = Inches(1.2 + col * 6)
    y = Inches(2.2 + row * 1.6)
    
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(1.4),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    # 编号
    add_textbox(s, x + Inches(0.3), y + Inches(0.15), Inches(1.2), Inches(0.8),
                num, font_size=36, color=color, bold=True)
    # 标题
    add_textbox(s, x + Inches(1.3), y + Inches(0.2), Inches(4), Inches(0.5),
                title, font_size=22, color=C_WHITE, bold=True)
    # 描述
    add_textbox(s, x + Inches(1.3), y + Inches(0.75), Inches(4.2), Inches(0.5),
                desc, font_size=14, color=C_GRAY)

# =============================================================
# Slide 3 · Part 1 标题页 — 为什么做
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 01", "size": 20, "color": C_ACCENT},
    {"text": "为什么做这个系统", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "服装行业选款的真实痛点 & MIPO 面对的具体问题", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_PRIMARY)
add_textbox(s, Inches(0.8), Inches(5.4), Inches(10), Inches(0.5),
            "选款从「拍脑袋赌运气」到「有数据的消费者投票」",
            font_size=16, color=C_GRAY)

# =============================================================
# Slide 4 · 行业痛点
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_PRIMARY)

add_textbox(s, Inches(1.2), Inches(0.7), Inches(10), Inches(0.8),
            "服装选款的 4 个真痛点", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.45), Inches(10), Inches(0.5),
            "—— 传统经验决策在今天越来越难跑通", font_size=16, color=C_GRAY)

pain_points = [
    ("🏭", "靠经验拍款",    "设计师/买手凭个人审美选款，好坏取决于经验和「手感」，新人试错成本高",     C_PRIMARY),
    ("📊", "数据链路断裂",  "销量数据和款式特征脱节，不知道「为什么卖/为什么卖不动」，没法系统性改进",    C_ORANGE),
    ("🎯", "选不准受众",    "品牌客群是画像不是真人，选款不知道「到底是谁在买、为什么买」",               C_ACCENT),
    ("🔁", "每年重新来",    "今年积累的经验到明年换设计师就清零，没有可复用、可校准的「数字资产」",       RGBColor(0xA7, 0x8B, 0xFA)),
]

for i, (icon, title, desc, color) in enumerate(pain_points):
    x = Inches(1.2 + (i % 2) * 6)
    y = Inches(2.2 + (i // 2) * 2.4)
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(2.0),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(0.08), Inches(2.0), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(0.2), Inches(1), Inches(0.8),
                icon, font_size=40, color=color)
    add_textbox(s, x + Inches(1.2), y + Inches(0.25), Inches(4.2), Inches(0.6),
                title, font_size=22, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(1.2), y + Inches(0.95), Inches(4.2), Inches(0.9),
                desc, font_size=14, color=C_LIGHT)

# =============================================================
# Slide 5 · MIPO 的具体挑战
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_ACCENT)

add_textbox(s, Inches(1.2), Inches(0.7), Inches(10), Inches(0.8),
            "MIPO 蜜扑 — 童装户外的「双重难点」", font_size=36, color=C_WHITE, bold=True)

# 左：品牌简介
left_card = add_rounded_rect(s, Inches(1.2), Inches(1.8), Inches(5.5), Inches(5.2),
                              fill_color=RGBColor(0x22, 0x28, 0x42))
add_multi_text(s, Inches(1.6), Inches(2.1), Inches(5), Inches(5), [
    {"text": "🏷 品牌", "size": 14, "color": C_GRAY},
    {"text": "MIPO 蜜扑", "size": 22, "color": C_WHITE, "bold": True, "space_before": 2},
    {"text": "", "size": 6, "space_before": 6},
    {"text": "🎯 定位", "size": 14, "color": C_GRAY},
    {"text": "潮童户外 · 时尚感 + 轻机能 + 日常穿着", "size": 16, "color": C_LIGHT, "space_before": 2},
    {"text": "", "size": 6, "space_before": 6},
    {"text": "👶 客群", "size": 14, "color": C_GRAY},
    {"text": "6-18 岁中大童 · 尺码 110-180", "size": 16, "color": C_LIGHT, "space_before": 2},
    {"text": "", "size": 6, "space_before": 6},
    {"text": "🛍 双渠道", "size": 14, "color": C_GRAY},
    {"text": "自然流量（天猫搜索） + 抖音直播带货", "size": 16, "color": C_LIGHT, "space_before": 2},
    {"text": "", "size": 6, "space_before": 6},
    {"text": "👥 双层决策", "size": 14, "color": C_GRAY},
    {"text": "妈妈出钱拍板（55%） + 孩子能一票否决（45%）", "size": 16, "color": C_LIGHT, "space_before": 2},
])

# 右：两个难点
add_textbox(s, Inches(7.2), Inches(1.8), Inches(5.5), Inches(0.6),
            "两个让传统方法彻底失效的问题", font_size=20, color=C_ORANGE, bold=True)

problems = [
    ("问题一", "妈妈觉得「这个衣服耐脏好洗」",
     "但孩子觉得「颜色太暗像校服」 → 一票否决",
     C_PRIMARY),
    ("问题二", "直播间里拼色撞色款（F03 高）",
     "3 秒停留超高，妈妈冲动下单\n到货后男孩嫌太花不肯穿 → 退货率飙升",
     C_ORANGE),
]
for i, (title, p1, p2, color) in enumerate(problems):
    y = Inches(2.6 + i * 2.3)
    card = add_rounded_rect(s, Inches(7.2), y, Inches(5.5), Inches(2.0),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, Inches(7.2), y, Inches(0.08), Inches(2.0), fill_color=color)
    add_textbox(s, Inches(7.5), y + Inches(0.15), Inches(2), Inches(0.5),
                title, font_size=18, color=color, bold=True)
    add_textbox(s, Inches(7.5), y + Inches(0.65), Inches(5), Inches(1.2),
                p1 + "\n" + p2, font_size=15, color=C_LIGHT)

# =============================================================
# Slide 6 · Part 2 标题页 — 核心思路
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 02", "size": 20, "color": C_PRIMARY},
    {"text": "核心思路", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "模拟你品牌的典型消费者 · 让他们面对款式「投票」", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_ACCENT)

# =============================================================
# Slide 7 · 一句话原理 + 三层模型
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_PRIMARY)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "一句话原理", font_size=32, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.8),
            "「调出 30 个你的典型消费者 → 让他们看图 → 判断「我会不会买」→ 投票聚合」",
            font_size=22, color=C_ACCENT)

add_rect(s, Inches(1.2), Inches(2.3), Inches(11), Inches(0.03), fill_color=RGBColor(0x3A, 0x41, 0x58))

add_textbox(s, Inches(1.2), Inches(2.6), Inches(11), Inches(0.6),
            "消费者购买决策的三层模型 — 三个系统并行打分",
            font_size=22, color=C_WHITE, bold=True)

engines = [
    ("①", "身份表达层", "System 2 · 理性判断",
     "30 人设投票\n每个人有明确的场景/审美/价格偏好\n独立看图打分 + 说明不买的理由",
     C_PRIMARY, "人设 → 买不买"),
    ("②", "视觉观感层", "System 1 · 快速反应",
     "VLM 视觉大模型输出 10 特征 BARS 评分\n廓形/利落/色彩/功能/上镜/实穿/搭配/面料/调性/独特",
     C_ACCENT, "图片 → 结构化特征"),
    ("③", "价值评估层", "System 2 · 理性算账",
     "感知价值 vs 价格在品类中的百分位\n值不值这个价？有没有被收割智商税？",
     C_ORANGE, "价签 → 值不值"),
]

for i, (num, title, sub, desc, color, tag) in enumerate(engines):
    x = Inches(1.2 + i * 4)
    y = Inches(3.4)
    card = add_rounded_rect(s, x, y, Inches(3.7), Inches(3.6),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(3.7), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.25), y + Inches(0.25), Inches(0.8), Inches(0.6),
                num, font_size=36, color=color, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(0.9), Inches(3.3), Inches(0.5),
                title, font_size=20, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(1.35), Inches(3.3), Inches(0.4),
                sub, font_size=12, color=C_GRAY)
    add_textbox(s, x + Inches(0.25), y + Inches(1.75), Inches(3.3), Inches(1.5),
                desc, font_size=13, color=C_LIGHT)
    # tag
    tag_shape = add_rounded_rect(s, x + Inches(0.25), y + Inches(3.05),
                                  Inches(3.2), Inches(0.35), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(3.08), Inches(3.1), Inches(0.3),
                tag, font_size=12, color=C_DARK, bold=True, align=PP_ALIGN.CENTER)

# 底部合成公式
add_rect(s, Inches(1.2), Inches(7.15), Inches(11), Inches(0.03), fill_color=RGBColor(0x3A, 0x41, 0x58))
add_textbox(s, Inches(1.2), Inches(7.2), Inches(11), Inches(0.3),
            "综合分 = 0.35 × 人设投票 + 0.30 × 双渠道分 + 0.35 × 价值匹配",
            font_size=13, color=C_GRAY, align=PP_ALIGN.CENTER)

# =============================================================
# Slide 8 · Part 3 标题页 — 系统架构
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 03", "size": 20, "color": C_ORANGE},
    {"text": "系统架构", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "通用 CORE 引擎 × 品牌适配包 × 3Loop 自优化内核", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_PRIMARY)

# =============================================================
# Slide 9 · 三大核心能力
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_ACCENT)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "三大核心能力", font_size=36, color=C_WHITE, bold=True)

capabilities = [
    ("能力一", "通用 CORE 引擎 · 跨品牌 100% 复用",
     "三大异构引擎（人设投票 / VLM 特征 / 价值评估）并行\n错误来源互相独立，一个引擎错了另外两个还能兜住\n换品类（童装→女装→快消）代码一行不动",
     C_PRIMARY,
     ["引擎一：30 人设投票", "引擎二：VLM + BARS 量表", "引擎三：价值匹配模型"]),
    ("能力二", "品牌适配包 · 换品牌 = 换 YAML",
     "5 个 YAML 文件承载品牌全部差异\n人设 / BARS 锚定量表 / 品类价格带 / 分级阈值 / 决策层\n不需要改 Python 代码",
     C_ACCENT,
     ["personas.yaml   30 人设", "features_bars.yaml   10 特征", "scoring_weights.yaml   公式+阈值"]),
    ("能力三", "3Loop 优化内核 · 越用越准",
     "每季销售结束后喂入真实销量\nLoop1 校准 VLM 特征偏置\nLoop2 Lasso 拟合真实人设分布\nLoop3 调三大引擎 + 双渠道权重\nSpearman 单调不减（不达标自动回滚）",
     C_ORANGE,
     ["Loop1 特征偏置", "Loop2 人设权重", "Loop3 引擎权重"]),
]

for i, (label, title, desc, color, bullets) in enumerate(capabilities):
    x = Inches(1.2)
    y = Inches(1.5 + i * 2.0)
    card = add_rounded_rect(s, x, y, Inches(11), Inches(1.8),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(0.1), Inches(1.8), fill_color=color)
    
    add_textbox(s, x + Inches(0.3), y + Inches(0.12), Inches(2), Inches(0.4),
                label, font_size=14, color=color, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(0.45), Inches(6), Inches(0.5),
                title, font_size=20, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(0.95), Inches(6.5), Inches(0.85),
                desc, font_size=12, color=C_LIGHT)
    
    # 右侧 bullets
    for j, b in enumerate(bullets):
        bx = x + Inches(7.2 + j * 1.2)
        add_rounded_rect(s, bx, y + Inches(0.35), Inches(1.05), Inches(1.1),
                         fill_color=color)
        add_textbox(s, bx, y + Inches(0.45), Inches(1.05), Inches(1.0),
                    b, font_size=10, color=C_DARK, bold=True, align=PP_ALIGN.CENTER)

# =============================================================
# Slide 10 · 品牌适配包目录结构
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_PRIMARY)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "品牌适配包 Brand Profile", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "5 个 YAML 文件 = 一个品牌的全部领域知识", font_size=16, color=C_GRAY)

# 左：目录树
tree = add_rounded_rect(s, Inches(1.2), Inches(2.0), Inches(5.5), Inches(5.0),
                         fill_color=RGBColor(0x14, 0x19, 0x28))
code = """brand_profiles/
└── mipo/                    ← MIPO 蜜扑品牌包
    ├── profile.yaml          决策层结构（妈妈+孩子）
    ├── personas.yaml         30 人设 + 身份三轴线
    ├── features_bars.yaml    10 特征 BARS 锚定量表
    ├── scoring_weights.yaml  评分公式 + S/A/P 阈值
    ├── category_registry.yaml 品类 + 价格带
    └── calibrated/           3Loop 校准产物（自动生成）"""
tb = s.shapes.add_textbox(Inches(1.4), Inches(2.2), Inches(5.2), Inches(4.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
run = p.add_run(); run.text = code; run.font.size = Pt(13); run.font.color.rgb = C_ACCENT; run.font.name = "Consolas"

# 右：各文件作用
files = [
    ("profile.yaml",    "决策结构",       "单层/多层决策结构声明\nMIPO = 妈妈决策者层(55%)\n      + 孩子影响层(45%)"),
    ("personas.yaml",   "30 人设",         "场景 × 审美 × 价格 = 5×3×2\n每个人有 FAB 关注点 + 否决红线"),
    ("features_bars.yaml", "10 特征 BARS", "每个特征 5 档行为锚定\n明确定义「什么是 MIPO 的好款」"),
    ("scoring_weights.yaml", "评分公式",  "自然流量 / 直播带货 / 价值匹配\nS·A+·A·P·风险 五级阈值"),
]
for i, (fname, title, desc) in enumerate(files):
    y = Inches(2.0 + i * 1.2)
    add_textbox(s, Inches(7.2), y, Inches(2.5), Inches(0.45),
                fname, font_size=13, color=C_PRIMARY, bold=True)
    add_textbox(s, Inches(9.7), y, Inches(3), Inches(0.45),
                title, font_size=15, color=C_WHITE, bold=True)
    add_textbox(s, Inches(7.2), y + Inches(0.45), Inches(5.5), Inches(0.7),
                desc, font_size=12, color=C_LIGHT)

# =============================================================
# Slide 11 · 3Loop 内核
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_ORANGE)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "3Loop 优化内核 — 越用越准的引擎", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "每季销售结束后喂入真实销量 → 自动校准 → Spearman 单调递增",
            font_size=16, color=C_GRAY)

loops = [
    ("Loop 1", "校准 VLM 特征偏置",
     "统计每个 VLM 特征与真实销量的 Spearman ρ\n除以平均 ρ 得到偏置系数\nclamp 到 [0.7, 1.3] 区间防止过度放大",
     C_PRIMARY,
     ["F01 廓形", "F03 色彩", "F05 上镜", "F06 实穿", "F09 调性"]),
    ("Loop 2", "Lasso 拟合人设权重",
     "L1 正则化回归从真实销量反推 30 人设真实权重\n稀疏性保证只有真正有区分力的人设被调高\n避免过拟合",
     C_ACCENT,
     ["P01 上学实用", "P03 山系亮色", "P07 周末实用", "P19 直播间"]),
    ("Loop 3", "调引擎 + 渠道权重",
     "权重 ∝ max(0.05, ρ + 0.3) 归一化\n保证每个引擎最少 5% 话语权\n防止某引擎被完全压制",
     C_ORANGE,
     ["人设 0.35", "渠道 0.30", "价值 0.35"]),
]

for i, (name, title, desc, color, tags) in enumerate(loops):
    x = Inches(1.2 + i * 4)
    y = Inches(2.1)
    card = add_rounded_rect(s, x, y, Inches(3.7), Inches(4.5),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(3.7), Inches(0.1), fill_color=color)
    
    add_textbox(s, x + Inches(0.25), y + Inches(0.25), Inches(2), Inches(0.5),
                name, font_size=26, color=color, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(0.8), Inches(3.3), Inches(0.5),
                title, font_size=18, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(1.3), Inches(3.3), Inches(1.8),
                desc, font_size=13, color=C_LIGHT)
    
    # tags
    for j, t in enumerate(tags):
        ty = y + Inches(3.3 + j * 0.28)
        tag = add_rounded_rect(s, x + Inches(0.25), ty, Inches(3.2), Inches(0.25),
                                fill_color=color)
        add_textbox(s, x + Inches(0.3), ty, Inches(3.1), Inches(0.25),
                    t, font_size=11, color=C_DARK, bold=True, align=PP_ALIGN.CENTER)

# 底部保护机制
guard = add_rounded_rect(s, Inches(1.2), Inches(6.8), Inches(11), Inches(0.5),
                         fill_color=C_PRIMARY)
add_textbox(s, Inches(1.3), Inches(6.83), Inches(10.8), Inches(0.45),
            "🛡 保护机制：每步 Loop 后必须满足「新 Spearman ≥ 旧 Spearman + MIN_IMPROVEMENT」，否则自动回滚。Spearman 单调不减，永不越调越差。",
            font_size=13, color=C_WHITE, bold=True)

# =============================================================
# Slide 12 · Part 4 标题页 — MIPO 实例
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 04", "size": 20, "color": RGBColor(0xA7, 0x8B, 0xFA)},
    {"text": "MIPO 实例详解", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "30 人设 / 10 特征 BARS / 双渠道评分 / S·A+·A·P 分级", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_ACCENT)

# =============================================================
# Slide 13 · 30 人设怎么来的
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=RGBColor(0xA7, 0x8B, 0xFA))

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "30 人设 — 怎么定义 MIPO 的典型消费者", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "身份三轴线交叉 → 5 × 3 × 2 = 30 种典型消费者", font_size=16, color=C_GRAY)

# 三轴卡
axes = [
    ("轴线 A · 场景", ["上学日常 30%", "周末家庭出游 20%", "露营/户外课程 20%", "抖音直播种草 15%", "体育课/运动 15%"],
     C_PRIMARY),
    ("轴线 B · 审美", ["实用耐脏型 40%", "山系多巴胺亮色 35%", "日系机能潮酷 25%"],
     C_ACCENT),
    ("轴线 C · 价格", ["刚需性价比 60%", "功能品质溢价 40%"],
     C_ORANGE),
]
for i, (title, items, color) in enumerate(axes):
    x = Inches(1.2 + i * 4)
    y = Inches(2.0)
    card = add_rounded_rect(s, x, y, Inches(3.7), Inches(3.5),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(3.7), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.25), y + Inches(0.2), Inches(3.3), Inches(0.5),
                title, font_size=18, color=C_WHITE, bold=True)
    for j, it in enumerate(items):
        add_textbox(s, x + Inches(0.25), y + Inches(0.8 + j * 0.5), Inches(3.3), Inches(0.4),
                    f"• {it}", font_size=13, color=C_LIGHT)

# 示例人设
add_textbox(s, Inches(1.2), Inches(5.8), Inches(11), Inches(0.5),
            "人设长什么样？举三个例子", font_size=20, color=C_WHITE, bold=True)

personas = [
    ("P01 实用性价比·上学娃妈", "上学 + 实用耐脏 + 刚需性价比",
     "FAB 关注：耐脏、好洗、机洗不发黄、膝盖耐磨\n颜色偏好：藏青、深灰、军绿、卡其\n孩子否决：卡通图案过多、浅色不耐脏",
     C_PRIMARY),
    ("P03 山系亮色性价比·上学娃妈", "上学 + 山系亮色 + 刚需性价比",
     "FAB 关注：亮色好找娃、耐脏暗纹、日常可穿、直播间破价值得囤\n颜色偏好：晨曦黄、橘色、水蓝、多巴胺绿\n孩子否决：颜色太暗沉、太像校服",
     C_ACCENT),
    ("P24 机能品质·直播间宝妈", "直播间 + 日系机能 + 品质溢价",
     "FAB 关注：潮酷机能风、专业面料话术、街拍范儿、专柜级\n颜色偏好：暗黑机能、潮牌拼接、未来感设计\n孩子否决：不够有个性、面料没质感",
     C_ORANGE),
]
for i, (name, axes_info, detail, color) in enumerate(personas):
    x = Inches(1.2 + i * 4)
    y = Inches(6.3)
    card = add_rounded_rect(s, x, y, Inches(3.7), Inches(1.0),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(0.08), Inches(1.0), fill_color=color)
    add_textbox(s, x + Inches(0.25), y + Inches(0.08), Inches(3.3), Inches(0.35),
                name, font_size=13, color=color, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(0.4), Inches(3.3), Inches(0.55),
                axes_info, font_size=11, color=C_LIGHT)

# =============================================================
# Slide 14 · 10 特征 BARS
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_S)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "10 特征 BARS · 让 VLM 像质检员一样结构化打分", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "每个特征 1-10 分，每 2 分一档有明确的行为锚定描述", font_size=16, color=C_GRAY)

features = [
    ("物理类", C_PRIMARY, [
        ("F01 廓形度",        "超紧贴身 → 超宽廓形"),
        ("F02 利落感",        "视觉混乱 → 极利落"),
        ("F03 色彩风险",      "安全基础色 → 极强视觉冲击"),
        ("F04 功能可见性",    "无功能元素 → 硬核功能外显"),
    ]),
    ("感知类", C_ACCENT, [
        ("F05 上镜感",        "画面平淡 → 极强镜头冲击"),
        ("F06 实穿度",        "场景极窄 → 全场景百搭"),
        ("F07 搭配度",        "极挑搭配 → 百搭"),
        ("F08 面料感知",      "廉价无质感 → 高级功能质感"),
    ]),
    ("品牌类", C_ORANGE, [
        ("F09 品牌调性贡献",  "偏离定位 → 调性标杆款"),
        ("F10 设计独特性",    "完全跟款 → 极罕见设计语言"),
    ]),
]

for i, (cat, color, feats) in enumerate(features):
    x = Inches(1.2 + i * 4)
    y = Inches(2.0)
    h = Inches(4.5)
    card = add_rounded_rect(s, x, y, Inches(3.7), h,
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(3.7), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.25), y + Inches(0.2), Inches(3.3), Inches(0.45),
                cat, font_size=18, color=color, bold=True)
    for j, (fname, anchor) in enumerate(feats):
        fy = y + Inches(0.8 + j * 0.85)
        add_textbox(s, x + Inches(0.25), fy, Inches(3.3), Inches(0.35),
                    fname, font_size=14, color=C_WHITE, bold=True)
        add_textbox(s, x + Inches(0.25), fy + Inches(0.35), Inches(3.3), Inches(0.35),
                    anchor, font_size=11, color=C_GRAY)

# 底部说明
note = add_rounded_rect(s, Inches(1.2), Inches(6.8), Inches(11), Inches(0.5),
                         fill_color=RGBColor(0x22, 0x28, 0x42))
add_textbox(s, Inches(1.3), Inches(6.82), Inches(10.8), Inches(0.45),
            "⚡ 关键锚定：F09 高调性 / F10 高独特性 ≠ 高销量。它们是品牌展示款（P 款）的典型特征，"
            "真正的 S 款是 F06 实穿度高 + F07 搭配度高 + F03 色彩风险低。",
            font_size=12, color=C_LIGHT)

# =============================================================
# Slide 15 · 双渠道评分
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_PRIMARY)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "双渠道评分 · 自然流量 vs 直播带货", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "不同渠道的好款特征完全不同 — 用两套公式分别计算", font_size=16, color=C_GRAY)

channels = [
    ("自然流量分",   "天猫/淘宝搜索、收藏加购、复购",
     "妈妈「解决需求」型购买：实穿 + 好搭 + 不挑色",
     [("F06 实穿度",       "0.40", C_S),
      ("F07 搭配度",       "0.30", C_A),
      ("(10-F03) 色彩安全", "0.30", C_ACCENT)],
     C_PRIMARY,
     "score = 0.40×F06 + 0.30×F07 + 0.30×(10-F03)"),
    ("直播带货分",   "抖音/快手直播、达人短视频",
     "冲动转化：画面吸睛 → 有料可讲 → 记忆深刻",
     [("F05 上镜感",       "0.40", C_S),
      ("F04 功能可见性",   "0.40", C_ORANGE),
      ("(10-F03) 色彩安全", "0.20", C_ACCENT)],
     C_ACCENT,
     "score = 0.40×F05 + 0.40×F04 + 0.20×(10-F03)"),
]

for i, (name, where, why, feats, color, formula) in enumerate(channels):
    x = Inches(1.2 + i * 6)
    y = Inches(2.0)
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(5.0),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(5.6), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(0.2), Inches(5), Inches(0.5),
                name, font_size=26, color=color, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(0.7), Inches(5), Inches(0.4),
                where, font_size=13, color=C_GRAY)
    add_textbox(s, x + Inches(0.3), y + Inches(1.15), Inches(5), Inches(0.5),
                why, font_size=14, color=C_LIGHT)
    
    # 特征权重条
    for j, (fname, w, fc) in enumerate(feats):
        fy = y + Inches(1.8 + j * 0.8)
        add_textbox(s, x + Inches(0.3), fy, Inches(2.5), Inches(0.4),
                    fname, font_size=14, color=C_WHITE)
        # 权重条
        bar_w = Inches(float(w) * 5)  # scale
        add_rect(s, x + Inches(2.8), fy + Inches(0.08), bar_w, Inches(0.25),
                 fill_color=fc)
        add_textbox(s, x + Inches(2.8) + bar_w + Inches(0.1), fy, Inches(1), Inches(0.4),
                    w, font_size=13, color=fc, bold=True)
    
    # 公式
    add_rounded_rect(s, x + Inches(0.3), y + Inches(4.3), Inches(5.0), Inches(0.5),
                      fill_color=color)
    add_textbox(s, x + Inches(0.35), y + Inches(4.32), Inches(4.9), Inches(0.45),
                formula, font_size=13, color=C_DARK, bold=True, align=PP_ALIGN.CENTER)

# =============================================================
# Slide 16 · S·A·P 分级逻辑
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_S)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "S / A+ / A / P 五级分级 — 怎么切", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "先识别「肯定好的 S 款」和「肯定弱的 P 款」，中间留 A+ / A 可调空间",
            font_size=16, color=C_GRAY)

grades = [
    ("S 款", "🔥 必推爆款",
     "自然分 ≥ 7.5 且 直播分 ≥ 7.0\n反对率 ≤ 25%\n价值匹配 ≥ 0（不溢价）",
     C_S),
    ("A+ 款", "💡 有爆款潜力",
     "单维度 ≥ 7.5，另一维度 6.0-7.0\n改一个短板就能冲 S",
     C_A),
    ("A 款", "✅ 安全常规款",
     "单维度 ≥ 7.0\n没硬伤也没爆点",
     RGBColor(0x63, 0x66, 0xF1)),
    ("P 款", "⚠️ 品牌展示款",
     "(F03≥6 色彩风险高 AND F09≥7.5 高调性)\nOR (F10≥7.5 高独特性 AND 双渠道弱)\n设计好看但实际卖不动",
     C_P),
]

for i, (gname, gsub, desc, color) in enumerate(grades):
    x = Inches(1.2 + (i % 2) * 6)
    y = Inches(2.0 + (i // 2) * 2.5)
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(2.3),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(0.12), Inches(2.3), fill_color=color)
    
    # 大字号等级
    add_textbox(s, x + Inches(0.3), y + Inches(0.15), Inches(1.5), Inches(0.8),
                gname, font_size=36, color=color, bold=True)
    add_textbox(s, x + Inches(1.8), y + Inches(0.3), Inches(3.5), Inches(0.6),
                gsub, font_size=18, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(1.05), Inches(5.2), Inches(1.2),
                desc, font_size=13, color=C_LIGHT)

# =============================================================
# Slide 17 · Part 5 标题页 — 效果验证
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 05", "size": 20, "color": C_S},
    {"text": "效果验证", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "分级准确率 · 改款建议 · 残差发现", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_ORANGE)

# =============================================================
# Slide 18 · 效果数据
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_S)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "跑过一批数据后能看什么", font_size=36, color=C_WHITE, bold=True)

# 左侧：四个数字
metrics = [
    ("Spearman ρ", "≥ 0.80", "分级与真实销量的秩相关", C_PRIMARY),
    ("改款建议", "2-5 条/款", "从 10 特征雷达图自动生成", C_ACCENT),
    ("残差发现", "ε > 2σ", "超预期款 + 不及预期款", C_ORANGE),
    ("校准后", "Spearman ↑", "3Loop 单调提升", C_S),
]
for i, (label, val, desc, color) in enumerate(metrics):
    x = Inches(1.2 + (i % 2) * 3.05)
    y = Inches(1.7 + (i // 2) * 2.6)
    card = add_rounded_rect(s, x, y, Inches(2.7), Inches(2.3),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(2.7), Inches(0.06), fill_color=color)
    add_textbox(s, x + Inches(0.15), y + Inches(0.15), Inches(2.4), Inches(0.4),
                label, font_size=13, color=C_GRAY)
    add_textbox(s, x + Inches(0.15), y + Inches(0.55), Inches(2.4), Inches(0.8),
                val, font_size=36, color=color, bold=True)
    add_textbox(s, x + Inches(0.15), y + Inches(1.35), Inches(2.4), Inches(0.8),
                desc, font_size=12, color=C_LIGHT)

# 右侧：改款建议示例
right_card = add_rounded_rect(s, Inches(7.5), Inches(1.7), Inches(5.5), Inches(5.0),
                               fill_color=RGBColor(0x22, 0x28, 0x42))
add_textbox(s, Inches(7.8), Inches(1.9), Inches(5), Inches(0.4),
            "🔧 改款建议示例", font_size=18, color=C_WHITE, bold=True)
add_textbox(s, Inches(7.8), Inches(2.3), Inches(5), Inches(0.35),
            "款号 S02 · P 级（设计好看但色彩太花）", font_size=13, color=C_P)

suggestions = [
    ("降低 F03 色彩风险",  "主推色从高饱和撞色（F03=7）改为双色拼接（F03=4-5），保留视觉记忆点同时降低男孩「太花不肯穿」风险", C_PRIMARY),
    ("补强 F06 实穿度",    "廓形稍收紧，从「只能户外」调整为「上学+户外双场景」，自然流量分可从 5.8 提至 7.0", C_ACCENT),
    ("保留 F04 功能亮点",  "F04=8 是强项，直播时作为钩子话术重点演示「泼水不渗」，已有上镜感 F05=8 加持", C_ORANGE),
    ("价值匹配优化",      "价格下调 15% 或增加一个实用小配件（如可拆卸帽檐），让 value_match 从 -0.20 拉到 0 以上", C_S),
]
for i, (title, desc, color) in enumerate(suggestions):
    sy = Inches(2.8 + i * 1.0)
    add_rect(s, Inches(7.8), sy + Inches(0.05), Inches(0.08), Inches(0.35), fill_color=color)
    add_textbox(s, Inches(7.95), sy, Inches(5), Inches(0.35),
                title, font_size=13, color=color, bold=True)
    add_textbox(s, Inches(7.95), sy + Inches(0.35), Inches(5), Inches(0.55),
                desc, font_size=11, color=C_LIGHT)

# =============================================================
# Slide 19 · 残差发现（运营的金矿）
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_ORANGE)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "残差分离 — 系统说不准的那些款，恰恰是运营的金矿", font_size=32, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "残差 ε = 真实销量 − 预测分，按正态分布统计 ±2σ 边界", font_size=15, color=C_GRAY)

quads = [
    ("ε > μ + 2σ", "超预期款",
     "卖得比 AI 预测好很多\n复盘不可预知因素：KOL 带火 / 热搜同款 / 竞品缺货",
     C_S, "📢 运营复盘素材"),
    ("ε < μ − 2σ", "不及预期款",
     "卖得比 AI 预测差很多\n复盘外生负面因素：质量事故 / 差评扩散 / 竞品大促",
     C_P, "⚠️ 运营复盘素材"),
]

for i, (stat, title, desc, color, tag) in enumerate(quads):
    x = Inches(1.2 + i * 6)
    y = Inches(2.0)
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(2.6),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(5.6), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(0.15), Inches(5), Inches(0.4),
                stat, font_size=14, color=C_GRAY)
    add_textbox(s, x + Inches(0.3), y + Inches(0.55), Inches(5), Inches(0.5),
                title, font_size=24, color=color, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(1.1), Inches(5), Inches(1.2),
                desc, font_size=13, color=C_LIGHT)

# 四象限归因
add_textbox(s, Inches(1.2), Inches(5.0), Inches(11), Inches(0.5),
            "残差 × 营销标签（实际投放）→ 资源错配归因四象限", font_size=20, color=C_WHITE, bold=True)

attributions = [
    ("推对了",   "投放了 + 超预期", "投放放大了款式潜力",       C_S),
    ("漏网爆款", "没推 + 超预期",    "本季最大机会损失！下季应提前识别", C_PRIMARY),
    ("资源错配", "投放了 + 不及预期", "投放资源浪费在弱款上",      C_P),
    ("款式本身弱", "没推 + 不及预期", "残差主要由款式质量解释",    C_GRAY),
]
for i, (name, cond, desc, color) in enumerate(attributions):
    x = Inches(1.2 + i * 3.05)
    y = Inches(5.6)
    card = add_rounded_rect(s, x, y, Inches(2.7), Inches(1.6),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(2.7), Inches(0.06), fill_color=color)
    add_textbox(s, x + Inches(0.15), y + Inches(0.1), Inches(2.4), Inches(0.4),
                name, font_size=16, color=color, bold=True)
    add_textbox(s, x + Inches(0.15), y + Inches(0.5), Inches(2.4), Inches(0.3),
                cond, font_size=11, color=C_GRAY)
    add_textbox(s, x + Inches(0.15), y + Inches(0.8), Inches(2.4), Inches(0.7),
                desc, font_size=12, color=C_LIGHT)

# 底部重要说明
note = add_rounded_rect(s, Inches(1.2), Inches(7.25), Inches(11), Inches(0.2),
                         fill_color=C_PRIMARY)
add_textbox(s, Inches(1.3), Inches(7.27), Inches(10.8), Inches(0.2),
            "🚫 残差款绝对不能喂回 3Loop 校准 — 会学到伪相关（KOL 碰巧穿了某颜色 ≠ 该颜色本身好卖）",
            font_size=11, color=C_WHITE, bold=True, align=PP_ALIGN.CENTER)

# =============================================================
# Slide 20 · Part 6 标题页 — 产品形态
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

add_multi_text(s, Inches(0.8), Inches(2.2), Inches(12), Inches(3), [
    {"text": "Part 06", "size": 20, "color": C_A},
    {"text": "产品形态", "size": 60, "color": C_WHITE, "bold": True, "space_before": 10},
    {"text": "Web UI 四页闭环 · 一键启动 · 本地运行", "size": 22, "color": C_LIGHT, "space_before": 20},
])
add_rect(s, Inches(0.8), Inches(5.2), Inches(2), Inches(0.06), fill_color=C_ACCENT)

# =============================================================
# Slide 21 · 四个 Web UI 页面
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill_color=C_A)

add_textbox(s, Inches(1.2), Inches(0.6), Inches(11), Inches(0.7),
            "Web UI 四页闭环", font_size=36, color=C_WHITE, bold=True)
add_textbox(s, Inches(1.2), Inches(1.3), Inches(11), Inches(0.5),
            "从上传到校准，一个浏览器搞定全流程", font_size=16, color=C_GRAY)

pages = [
    ("📤 上传批次",
     "选品牌适配包\n传 Excel（款式+FAB+售价）\n传图片文件夹\n自动校验 → 开始评估",
     C_PRIMARY, "Excel + 图片"),
    ("📋 批次总表",
     "S/A+/A/P 饼图\n每款一行：分级/综合分/自然分/直播分\n筛选 + 排序 + 导出 Excel\n打包下载 Markdown 报告",
     C_ACCENT, "总览 + 导出"),
    ("🔍 单款详情",
     "左：图片轮播（正面/背面/look/细节）\n右：改款建议 + 三大引擎贡献\n10 特征 BARS 雷达图\n人设投票分布 + 双渠道明细",
     C_ORANGE, "深挖 + 建议"),
    ("📊 回测校准",
     "喂真实销量 Excel\n点「运行 3Loop 校准」\nLoop 前后 Spearman 对比\n残差超预期/不及预期列表\n校准通过 → 权重写入 calibrated/",
     C_S, "校准 + 迭代"),
]

for i, (name, content, color, tag) in enumerate(pages):
    x = Inches(1.2 + (i % 2) * 6)
    y = Inches(2.0 + (i // 2) * 2.7)
    card = add_rounded_rect(s, x, y, Inches(5.6), Inches(2.5),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(5.6), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(0.15), Inches(5), Inches(0.5),
                name, font_size=22, color=color, bold=True)
    add_textbox(s, x + Inches(0.3), y + Inches(0.7), Inches(5.2), Inches(1.5),
                content, font_size=13, color=C_LIGHT)
    # 底部 tag
    tag_shape = add_rounded_rect(s, x + Inches(0.3), y + Inches(2.1),
                                  Inches(1.3), Inches(0.3), fill_color=color)
    add_textbox(s, x + Inches(0.3), y + Inches(2.12), Inches(1.3), Inches(0.28),
                tag, font_size=11, color=C_DARK, bold=True, align=PP_ALIGN.CENTER)

# =============================================================
# Slide 22 · 总结 & Q&A
# =============================================================
s = prs.slides.add_slide(BLANK)
set_slide_bg(s, C_DARK)

# 装饰
circle1 = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-2), Inches(-1), Inches(5), Inches(5))
circle1.fill.solid(); circle1.fill.fore_color.rgb = C_PRIMARY; circle1.fill.transparency = 0.85
circle1.line.fill.background()

add_multi_text(s, Inches(1.2), Inches(1.0), Inches(11), Inches(2), [
    {"text": "总结", "size": 44, "color": C_WHITE, "bold": True},
    {"text": "", "size": 10, "space_before": 4},
])

# 三条 takeaway
takeaways = [
    ("①", "选款不再是「拍脑袋」", "30 个品牌典型消费者在投票，每个决策都有可追溯的理由", C_PRIMARY),
    ("②", "品牌知识可沉淀", "5 个 YAML 承载品牌全部领域知识，换设计师经验不丢失", C_ACCENT),
    ("③", "越用越准", "每季销售数据喂回去，3Loop 自动校准，Spearman 单调递增", C_ORANGE),
]
for i, (num, title, desc, color) in enumerate(takeaways):
    x = Inches(1.2 + i * 4)
    y = Inches(2.3)
    card = add_rounded_rect(s, x, y, Inches(3.7), Inches(2.8),
                            fill_color=RGBColor(0x22, 0x28, 0x42))
    add_rect(s, x, y, Inches(3.7), Inches(0.08), fill_color=color)
    add_textbox(s, x + Inches(0.25), y + Inches(0.25), Inches(1), Inches(0.6),
                num, font_size=36, color=color, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(0.9), Inches(3.3), Inches(0.5),
                title, font_size=18, color=C_WHITE, bold=True)
    add_textbox(s, x + Inches(0.25), y + Inches(1.45), Inches(3.3), Inches(1.2),
                desc, font_size=13, color=C_LIGHT)

# 底部 Q&A
add_rect(s, Inches(1.2), Inches(5.5), Inches(11), Inches(0.03), fill_color=RGBColor(0x3A, 0x41, 0x58))
add_textbox(s, Inches(1.2), Inches(5.8), Inches(11), Inches(0.9),
            "Q & A", font_size=48, color=C_ACCENT, bold=True, align=PP_ALIGN.CENTER)

add_multi_text(s, Inches(1.2), Inches(6.8), Inches(11), Inches(0.7), [
    {"text": "fashion-hit-engine · 服装爆款预测通用引擎", "size": 14, "color": C_GRAY, "align": PP_ALIGN.CENTER},
    {"text": "MIPO 蜜扑童装户外品牌示例", "size": 12, "color": C_GRAY, "space_before": 4, "align": PP_ALIGN.CENTER},
])

# ---------- 保存 ----------
out_path = os.path.join(os.path.dirname(__file__), "..", "mipo_项目汇报.pptx")
prs.save(out_path)
print(f"✅ PPT 已生成: {out_path}")
print(f"   共 {len(prs.slides)} 页")
