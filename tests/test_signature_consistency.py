"""
签名一致性守护测试 — 防止 "v1.4.96.4 重构改签名 / 调用点没同步" 这类连续发版 crash。

背景：2026-10-07 连续 5 个 tag (v1.4.100 ~ v1.4.104) 全部因同一类 bug 崩溃：
  - render_page_detail 重构把内部闭包挪到 src/ui/components.py 变成模块函数
  - 签名从单参数 (closure captures outer vars) 改成双/多参数
  - 调用点还按老签名调用 → NameError / TypeError
  - 每次 Updater 拉到的都是坏版本

本测试的作用：扫描 app.py 里所有从 src.ui.components import 进来的
可调用函数的调用点，核对：
  1. 位置参数个数 >= 函数要求的最少位置参数
  2. keyword-only 参数（* 后面的）没有被位置传递

任何一条不满足 → 本测试 FAIL → pytest 红 → 发版脚本在 CI 里拦住。
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_app_components_imports():
    """解析 app.py 里 from src.ui.components import 进来的可调用符号 → 函数对象。"""
    import src.ui.components as comp_module  # noqa: F401  (import side-effect)

    app_py = ROOT / "app.py"
    src_text = app_py.read_text(encoding="utf-8")
    tree = ast.parse(src_text)

    imported: dict[str, tuple[str, inspect.Signature]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.ui.components":
            for alias in node.names:
                local_name = alias.asname or alias.name
                fn = getattr(comp_module, alias.name, None)
                if fn is not None and callable(fn):
                    imported[local_name] = (alias.name, inspect.signature(fn))
    return src_text, tree, imported


def _call_kw(call: ast.Call):
    return len(call.args), [kw.arg for kw in call.keywords if kw.arg is not None]


@pytest.mark.parametrize("local_name", [
    "render_breadcrumb",
    "thumb_b64",
    "anchor_color",
    "score_color_v1490",
    "_vote_sort_key_fn",
    "decision_label",
    "chip_row",
    "render_diagnostic_banner",
])
def test_no_signature_mismatches(local_name):
    """每个从 components import 进来的函数 → 所有调用点参数个数正确。"""
    src_text, tree, imported = _load_app_components_imports()
    if local_name not in imported:
        pytest.skip(f"{local_name} 未从 components import")

    _orig, sig = imported[local_name]
    params = list(sig.parameters.values())
    pos_params = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kwonly_params = [p for p in params if p.kind == p.KEYWORD_ONLY]
    n_req_pos = sum(1 for p in pos_params if p.default is p.empty)
    max_pos = len(pos_params)

    # 收集所有 Name 形式的调用点
    calls: list[tuple[int, int]] = []  # (lineno, n_pos)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == local_name):
            n_pos, kw = _call_kw(node)
            calls.append((node.lineno, n_pos))

    failures: list[str] = []
    for lineno, n_pos in calls:
        # 检查：位置参数不够
        if n_pos < n_req_pos:
            failures.append(
                f"line {lineno}: {local_name}() 位置参数不够，传了 {n_pos}，至少需要 {n_req_pos}")
            continue
        # 检查：keyword-only 被位置传递
        if n_pos > max_pos and kwonly_params:
            extra = n_pos - max_pos
            inferred_kw = [p.name for p in kwonly_params[:extra]]
            failures.append(
                f"line {lineno}: {local_name}() 多传了 {extra} 个位置参数，"
                f"应为 keyword-only: {inferred_kw}。"
                f"把第 {max_pos+1} 个及之后的参数改成 keyword=... 形式。"
            )

    assert not failures, (
        f"❌ {local_name}() 发现 {len(failures)} 处签名不匹配：\n"
        + "\n".join(f"  - {f}" for f in failures)
        + f"\n签名: {_orig}{sig}"
    )
