"""
图片重命名工具：按款号文件夹自动归类命名图片

你只需要把每款的图，放到"款号名"的文件夹里，图名随便起都OK。
脚本会：
  1. 扫描每个款文件夹下的图片
  2. 按文件名关键词自动归类（front/back/look/detail等）
  3. 重命名为 {款号}_{类型}_{序号}.{后缀}
  4. 默认 in_place=False 时：输出到独立目录，原始文件不动
     可选 in_place=True  时：在原目录内重命名

使用（命令行）：
  python scripts/rename_images.py --src "D:\\2026春第一批_images" --dst "D:\\renamed"
  python scripts/rename_images.py --src "D:\\2026春第一批_images" --in-place
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 关键词 -> 类型标签，匹配顺序=优先级
KEYWORD_RULES: list[tuple[list[str], str]] = [
    # Look穿搭全身
    (["look", "穿搭", "全身", "模特", "上身", "街拍", "outfit"], "look"),
    # 细节
    (["detail", "细节", "特写", "面料", "纹理", "logo", "工艺", "closeup", "close-up"], "detail"),
    # 正面
    (["front", "正面", "正", "主图", "主视觉", "main"], "front"),
    # 背面
    (["back", "背面", "反", "后背", "后幅", "后片"], "back"),
    # 侧面
    (["side", "侧面", "侧", "左", "右", "profile"], "side"),
]


@dataclass
class Summary:
    total_styles: int = 0
    total_images: int = 0
    matched: int = 0           # 命中关键词规则
    unmatched: int = 0         # 没命中，用 view_N 兜底
    per_style_stats: dict[str, dict] = None  # type: ignore[assignment]

    @property
    def match_rate(self) -> float:
        if self.total_images <= 0:
            return 0.0
        return self.matched / self.total_images


# =====================================================================
# 核心逻辑（供 app.py import 调用，也供命令行用）
# =====================================================================
def classify_image(filename: str) -> str | None:
    """根据文件名关键词返回分类标签，没匹配返回None"""
    name_lower = filename.lower()
    for keywords, tag in KEYWORD_RULES:
        for kw in keywords:
            if kw.lower() in name_lower:
                return tag
    return None


def list_style_folders(src_root: Path) -> list[Path]:
    """列出所有"款号文件夹"（直接子目录，非空）"""
    result = []
    for p in sorted(src_root.iterdir()):
        if not p.is_dir():
            continue
        if any(f.is_file() and f.suffix.lower() in IMG_EXTENSIONS for f in p.iterdir()):
            result.append(p)
    return result


def list_images(folder: Path) -> list[Path]:
    """文件夹下所有图片，按修改时间排序（默认第一张=正面图）"""
    imgs = [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
    imgs.sort(key=lambda x: x.stat().st_mtime)
    return imgs


def _dedup_name(dest_folder: Path, style_id: str, tag: str, idx: int, ext: str) -> Path:
    """避免重名：同一tag同一idx不覆盖"""
    candidate = dest_folder / f"{style_id}_{tag}_{idx}{ext}"
    i = 2
    while candidate.exists():
        candidate = dest_folder / f"{style_id}_{tag}_{idx}_{i}{ext}"
        i += 1
    return candidate


def batch_rename_from_folders(
    src_root: str | Path,
    dst_root: str | Path | None = None,
    in_place: bool = False,
) -> dict:
    """
    主入口：批量扫描款号文件夹并重命名归类

    Parameters
    ----------
    src_root : 款号文件夹所在的根目录（里面每个子目录是一个款，装它的图）
    dst_root : 输出目录（in_place=False时需要），文件会拷贝到这里
    in_place : True=直接在原目录重命名并覆盖原图；False=拷贝到dst_root

    Returns
    -------
    dict
        { total_styles, total_images, matched, unmatched,
          match_rate, per_style_stats: {style_id: {total, matched, new_paths: []}} }
    """
    src_root = Path(src_root)
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"源目录不存在或不是文件夹: {src_root}")

    if not in_place:
        if dst_root is None:
            raise ValueError("in_place=False 时必须指定 dst_root")
        dst_root = Path(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)

    summary = Summary(per_style_stats={})
    style_folders = list_style_folders(src_root)
    summary.total_styles = len(style_folders)

    for style_folder in style_folders:
        style_id = style_folder.name
        imgs = list_images(style_folder)
        if not imgs:
            continue

        tag_counter: Counter = Counter()
        new_paths: list[str] = []
        style_matched = 0

        dest_dir = style_folder if in_place else (dst_root / style_id)
        if not in_place:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(imgs, 1):
            summary.total_images += 1
            tag = classify_image(img.name)
            if tag:
                style_matched += 1
                summary.matched += 1
                tag_counter[tag] += 1
                tag_idx = tag_counter[tag]
            else:
                summary.unmatched += 1
                tag = "view"
                tag_counter[tag] += 1
                tag_idx = tag_counter[tag]

            ext = img.suffix.lower()
            if in_place:
                new_name = dest_dir / f"{style_id}_{tag}_{tag_idx}{ext}"
                if new_name.resolve() != img.resolve():
                    if new_name.exists():
                        # 重名防覆盖
                        new_name = _dedup_name(dest_dir, style_id, tag, tag_idx, ext)
                    img.rename(new_name)
                new_paths.append(str(new_name))
            else:
                new_name = dest_dir / f"{style_id}_{tag}_{tag_idx}{ext}"
                if new_name.exists():
                    new_name = _dedup_name(dest_dir, style_id, tag, tag_idx, ext)
                shutil.copy2(img, new_name)
                new_paths.append(str(new_name))

        summary.per_style_stats[style_id] = {
            "total": len(imgs),
            "matched": style_matched,
            "match_rate": style_matched / max(len(imgs), 1),
            "tag_counts": dict(tag_counter),
            "new_paths": new_paths,
        }

    return {
        "total_styles": summary.total_styles,
        "total_images": summary.total_images,
        "matched": summary.matched,
        "unmatched": summary.unmatched,
        "match_rate": summary.match_rate,
        "per_style_stats": summary.per_style_stats,
    }


# =====================================================================
# CLI 入口
# =====================================================================
def _print_human_report(result: dict) -> None:
    total_s = result["total_styles"]
    total_i = result["total_images"]
    print("=" * 56)
    print(f"📦 扫描完成：{total_s} 个款号文件夹，{total_i} 张图片")
    print(f"✅ 关键词归类命中：{result['matched']} 张（{result['match_rate']:.0%}）")
    print(f"ℹ️  未命中（按顺序view_N）：{result['unmatched']} 张")
    print("-" * 56)
    print(f"{'款号':<10}{'图数':<6}{'命中率':<10}{'各类型数量'}")
    for sid, s in result["per_style_stats"].items():
        tags = " ".join(f"{t}×{c}" for t, c in s["tag_counts"].items())
        print(f"{sid:<10}{s['total']:<6}{s['match_rate']:>4.0%}      {tags}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按款号文件夹自动归类并标准命名童装款式图片"
    )
    parser.add_argument("--src", required=True, help="款号图片根目录（每个子文件夹是一个款）")
    parser.add_argument("--dst", default=None, help="输出目录（in_place=False时需要）")
    parser.add_argument("--in-place", action="store_true",
                        help="直接在原目录重命名（默认关闭，会拷贝到--dst目录）")
    args = parser.parse_args()

    result = batch_rename_from_folders(
        src_root=args.src,
        dst_root=args.dst,
        in_place=args.in_place,
    )
    _print_human_report(result)


if __name__ == "__main__":
    main()
