"""跨 Streamlit 页面共用的 UI 组件和辅助函数。

这些函数之前是 app.py 里的嵌套闭包或模块级函数，
提升到这里后 render_page_detail 缩进浅了一层、app.py 也干净了。

所有函数都是纯函数（或只读 st.session_state），方便单元测试。
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image
import streamlit as st


# ──────────────────────────────────────────────────────────────
# 公共组件
# ──────────────────────────────────────────────────────────────

def render_breadcrumb(*, suffix: str | None = None) -> None:
    """批次名 + 当前页的面包屑（所有页面顶部共用）"""
    parts = []
    if st.session_state.get("batch_name"):
        parts.append(st.session_state.batch_name)
    if suffix:
        parts.append(suffix)
    st.markdown(
        "<div class='breadcrumb'>"
        + " <span class='sep'>·</span> ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# render_page_detail 用的嵌套闭包（全部提升为模块级）
# ──────────────────────────────────────────────────────────────

def thumb_b64(ipath: str, size: tuple[int, int] = (400, 520)) -> str:
    """PIL 中心裁剪 + 缩放到统一规格 → base64 data URI（款式缩略图渲染）"""
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


def anchor_color(anchor_level: int | None) -> str:
    """v1.4.90+: 视觉锚定档着色，无好坏语义"""
    _map = {1: "#c85a5a", 2: "#d68c45", 3: "#a8b5c4",
            4: "#5a8a9a", 5: "#3d7d8a"}
    return _map.get(anchor_level or 3, "#a8b5c4")


def score_color_v1490(score: float) -> str:
    """v1.4.90+: 三列指标颜色。score 是 anchor 区间中点（1.5/3.5/5.5/7.5/9.5），
    无绝对好坏语义，但 anchor 跨度越大通常表示视觉特征越极端。"""
    if score >= 8.5:   return "#3d7d8a"  # 档5 视觉上较突出
    if score >= 6.5:   return "#5a8a9a"  # 档4
    if score >= 4.5:   return "#a8b5c4"  # 档3 中性
    if score >= 2.5:   return "#d68c45"  # 档2
    return "#c85a5a"                       # 档1


def vote_sort_key(vv: Any, persona_lookup: dict[str, dict[str, Any]]) -> float:
    """按人设权重降序排序（_p_lookup 是外面传入的 {persona_id: {weight: ...}}）"""
    pp = persona_lookup.get(vv.persona_id, {})
    return -float(pp.get("weight", 0.0))


def decision_label(sc: float) -> tuple[str, str, str]:
    """(emoji, 中文, 主色) — Phase3 复评后的决策标签"""
    if sc >= 7:
        return "✅", "会买", "#6ca060"
    if sc >= 4:
        return "🤔", "犹豫", "#d68c45"
    return "❌", "不会买", "#b14a4a"


def chip_row(
    title: str,
    items: list[str],
    *,
    col_bg: str = "#fff7f0",
    col_txt: str = "#a06830",
    limit: int = 4,
) -> str:
    """小标签行（在意/偏好颜色/否决雷区等展示）"""
    if not items:
        return ""
    chips_html = "".join(
        f'<span style="background:{col_bg};color:{col_txt};'
        f'padding:0 6px;border-radius:4px;font-size:0.72em;'
        f'margin:1px 2px;display:inline-block">{it}</span>'
        for it in items[:limit]
    )
    return (
        f'<div style="margin:3px 0;font-size:0.75em;color:#6b655d">'
        f'· {title}：{chips_html}</div>'
    )


# ──────────────────────────────────────────────────────────────
# v1.4.95+: 三层错误处理 · UI 诊断条
# 把 P3 在 metadata 里收集的 VLM fallback / Phase2-3 错误 /
# calibration load_errors 渲染成紧凑的 Streamlit 提示块。
# ──────────────────────────────────────────────────────────────

_FALLBACK_LABEL = {
    "ollama_vlm": "云端 VLM 全挂 → 本地 Ollama VLM 兜底",
    "brand_default": "所有 VLM 全挂 → brand 默认档3 视觉锚定兜底（置信度低）",
}


def render_diagnostic_banner(
    *,
    prediction: Any | None = None,
    predictions: list[Any] | None = None,
    calibration_load_errors: dict[str, str] | None = None,
    calibration_loop_applied: dict[str, bool | None] | None = None,
    calibration_spearman_gains: dict[str, float] | None = None,
) -> None:
    """渲染诊断提示条（可选，无问题时静默不占空间）。

    Args:
        prediction: 单款详情页传入 FullPrediction
        predictions: 批次汇总页传入 list[FullPrediction]（会聚合诊断）
        calibration_load_errors: 批次汇总页可选，传入 CalibrationResult.load_errors
        calibration_loop_applied: 批次汇总页可选，传入 CalibrationResult.loop_applied
            key = "loop1_vlm" / "loop2_persona" / "loop3_engine" / "loop3_channel"
            value = True（生效）| False（回滚到默认）| None（无 YAML）
        calibration_spearman_gains: 批次汇总页可选，传入 CalibrationResult.spearman_gains
    """
    has_warning = False

    # ---- 单款诊断 ----
    if prediction is not None:
        fm = prediction.features.metadata or {}
        vm = prediction.voting.metadata or {}

        # VLM fallback / 错误
        fb = fm.get("fallback_used")
        if fb:
            label = _FALLBACK_LABEL.get(fb, f"未知 fallback: {fb}")
            # brand_default = 完全无真实视觉特征 → 红色 error
            # ollama_vlm = 本地 VLM 兜底（至少有真实视觉分析） → 黄色 warning
            if fb == "brand_default":
                st.error(
                    f"🚨 **{label}** — 人设投票 & 渠道评分均基于此档3默认值，"
                    f"结果可靠性大幅下降。请检查：① 智谱 API 额度/Key ② 图片格式/清晰度 ③ 并发 VLM 调用数"
                )
            else:
                st.warning(f"🧯 特征提取兜底触发 — {label}")
            has_warning = True
        vlm_errs = fm.get("vlm_errors") or []
        if vlm_errs:
            with st.expander(f"VLM 原始错误（{len(vlm_errs)} 条）", expanded=False):
                for e in vlm_errs:
                    st.code(str(e), language=None)
            has_warning = True

        # 三阶段评审
        p2e = vm.get("phase2_error")
        p3es = vm.get("phase3_errors") or []
        three_phase = vm.get("three_phase")
        if p2e:
            st.warning(f"🧑‍⚕️ Phase2 品类专家挑战失败 → 降级两阶段：{p2e}")
            has_warning = True
        if p3es:
            st.warning(f"🔁 Phase3 复评 {len(p3es)} 人设失败（保留初评）")
            has_warning = True
        if three_phase is False and not fm.get("is_mock"):
            # 非 mock 但没跑三阶段（persona 数 < 3 或 expert_model 不可用）
            st.caption("ℹ️ 三阶段评审未启用（可能人设数不足或配置关闭）")

        # mock 标记（真实问题，不是 warning）
        if fm.get("is_mock") or vm.get("is_mock"):
            has_warning = True  # 不重复报警，批次汇总页会统一显示

    # ---- 批次聚合诊断 ----
    if predictions:
        n = len(predictions)
        n_mock = sum(1 for p in predictions
                     if (p.features.metadata or {}).get("is_mock")
                     or (p.voting.metadata or {}).get("is_mock"))
        n_fallback = sum(1 for p in predictions
                         if (p.features.metadata or {}).get("fallback_used"))
        n_p2_fail = sum(1 for p in predictions
                        if (p.voting.metadata or {}).get("phase2_error"))
        n_p3_fail_total = 0
        for p in predictions:
            n_p3_fail_total += len((p.voting.metadata or {}).get("phase3_errors") or [])

        # calibration 加载错误
        cal_errs = calibration_load_errors or {}

        parts: list[str] = []
        has_brand_default = False
        _fb_dist: dict[str, int] = {}
        for p in predictions:
            fb = (p.features.metadata or {}).get("fallback_used")
            if fb:
                _fb_dist[fb] = _fb_dist.get(fb, 0) + 1
                if fb == "brand_default":
                    has_brand_default = True
        if n_fallback:
            fb_detail = " · ".join(
                f"{cnt} 款 {_FALLBACK_LABEL.get(fb, fb)}"
                for fb, cnt in _fb_dist.items()
            )
            parts.append(f"🧯 {n_fallback}/{n} 款 VLM fallback ({fb_detail})")
        if n_p2_fail:
            parts.append(f"🧑‍⚕️ {n_p2_fail}/{n} 款 Phase2 失败")
        if n_p3_fail_total:
            parts.append(f"🔁 {n_p3_fail_total} 人设 Phase3 失败")
        if cal_errs:
            parts.append(f"⚙️ 校准产物加载失败 {len(cal_errs)} 个文件")

        if parts:
            # brand_default fallback = 红色严重告警
            if has_brand_default:
                st.error("🚨 " + " · ".join(parts) +
                         " — 请检查智谱 VLM 可用性，否则预测质量大幅下降")
            else:
                st.warning(" · ".join(parts))
            # calibration 详情 expander（只在批次页展示一次）
            if cal_errs:
                with st.expander("校准 YAML 加载失败详情", expanded=False):
                    for fname, err in cal_errs.items():
                        st.markdown(f"**{fname}** — `{err}`")
            # fallback 分布
            if n_fallback:
                with st.expander("VLM fallback 类型分布", expanded=False):
                    for fb, cnt in _fb_dist.items():
                        st.caption(
                            f"{_FALLBACK_LABEL.get(fb, fb) or fb} — {cnt} 款"
                        )
            has_warning = True

        # v1.4.96+: Loop 校准状态总览（有 YAML 就展示，不管有没有 warning）
        loop_status = calibration_loop_applied or {}
        if loop_status:
            loop_names = {
                "loop1_vlm": "Loop1 VLM特征偏置",
                "loop2_persona": "Loop2 人设分布权重",
                "loop3_engine": "Loop3 引擎权重",
                "loop3_channel": "Loop3 渠道权重",
            }
            spearman = calibration_spearman_gains or {}
            loop_parts: list[str] = []
            for lkey, lname in loop_names.items():
                val = loop_status.get(lkey)
                if val is True:
                    delta = spearman.get(lkey, 0.0)
                    loop_parts.append(f"✅ {lname} (Δρ={delta:+.3f})")
                elif val is False:
                    delta = spearman.get(lkey, 0.0)
                    loop_parts.append(f"↩️ {lname} 回滚 (Δρ={delta:+.3f})")
                # None 表示该 Loop 无 YAML，跳过
            if loop_parts:
                # 有回滚 → caption 提示，全部生效 → 静默（避免信息噪音）
                n_ok = sum(1 for lkey, v in loop_status.items() if v is True)
                n_rolled = sum(1 for lkey, v in loop_status.items() if v is False)
                if n_rolled > 0:
                    st.caption("📊 校准 Loop 状态：" + " · ".join(loop_parts))
                    # 解释 expander
                    with st.expander("回滚原因提示", expanded=False):
                        st.markdown(
                            "**Loop 回滚**通常因为：\n"
                            "- **样本量不足**：Loop3 引擎 ≥30 款、Loop2 Lasso ≥5 款才能拟合\n"
                            "- **Spearman 没提升**：校准后 Spearman < 校准前 + MIN_IMPROVEMENT\n"
                            "- **正则化过强**：Loop2 LassoCV 选了大 alpha，把所有人设系数压到 0\n\n"
                            "建议：跑 backtest 时至少 40-60 款有销量标注的数据；"
                            "如果 Loop 持续回滚，说明 VLM/人设信号和销量的相关性还不够，"
                            "优先优化特征 prompt 或补更多标注数据。"
                        )
                # 全部生效（无回滚）就不额外显示，保持简洁

    # 无问题 → 静默返回（不占 UI 空间）
    return None
