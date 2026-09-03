#!/usr/bin/env python3
"""生成候选标注图卡（供用户人工标注 进球/非进球）。

复用 mot_candidates 的检测缓存与候选逻辑，对每个候选截取
t0-0.4 / t0 / t0+0.4 三帧、以候选点为中心裁 560x560，纵向拼成一张图卡；
按文件拼 5 列大 sheet，编号与 candidates.json 一一对应。

用户看 sheet 回报每个文件哪些编号是进球，回写 labels 后供
语义分类器（sigLIP 特征 + logistic 回归）训练使用。
"""

import logging
import os
import sys
from glob import glob
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mot_candidates as mot
from pipe_common import atomic_write_json, configure_logging, new_run_id

logger = logging.getLogger(__name__)

CROP_HALF: int = 420  # 裁剪半径（候选点为中心，840x840，覆盖 anchor 位置漂移）
CARD_SCALE: float = 0.4  # 图卡缩放
STRIP_OFFSETS: tuple[float, ...] = (-1.0, 0.0, 1.0)  # 三帧时间偏移（秒，覆盖 anchor 时间漂移）
HEADER_H: int = 30
COLS: int = 5  # sheet 每行图卡数

OUT_DIR: str = "work/label"
SRC_COLORS: dict[str, tuple[int, int, int]] = {
    "static": (40, 90, 180),
    "rejoin": (200, 110, 20),
}

DEFAULT_FIDS: list[str] = ["0011", "0020", "0030", "0040", "0128"]


def collect_file_candidates(fid: str) -> list[dict[str, Any]]:
    """复用 mot 模块的缓存与候选逻辑，产出单个文件的合并候选。

    Args:
        fid: 文件 ID。

    Returns:
        候选列表（按 t0 排序，含 t0/dur/ac/cx/cy/src）。
    """
    frames = sorted(glob(mot.FRAMES_PATTERN.format(fid)))
    cached = mot.load_detection_cache(fid, len(frames))
    if cached is None:
        logger.warning("%s: 无检测缓存，先运行 mot_candidates.py", fid)
        return []
    all_balls, all_persons = cached
    tracks = mot.run_mot(all_balls)
    static_cands, _, _, _ = mot.collect_candidates(tracks, all_persons)
    rejoin_cands = mot.find_rejoin_candidates(all_balls)
    cands, _ = mot.merge_candidates(static_cands, rejoin_cands)
    return cands


def frame_path(fid: str, sec: float) -> str:
    """由时间戳求最近采样帧路径。

    Args:
        fid: 文件 ID。
        sec: 时间戳（秒）。

    Returns:
        帧文件路径。
    """
    idx: int = max(1, round(sec * mot.SAMPLE_FPS) + 1)
    return mot.FRAMES_PATTERN.format(fid).replace("*", f"{idx:05d}")


def crop_around(img: Image.Image, cx: int, cy: int, half: int = CROP_HALF) -> Image.Image:
    """以 (cx,cy) 为中心裁 2*half 见方，越界收敛到画面边缘（按帧实际尺寸）。

    clamp 上限必须取 img.size 实际尺寸：PIL crop 越界会静默补黑边，
    硬编码 1920x1440 会让 1920x1080 的新素材帧中招。

    Args:
        img: 原始帧。
        cx: 中心 x。
        cy: 中心 y。
        half: 裁剪半径，默认 CROP_HALF。

    Returns:
        裁剪结果。
    """
    w, h = img.size
    x1: int = min(max(cx - half, 0), w - 2 * half)
    y1: int = min(max(cy - half, 0), h - 2 * half)
    return img.crop((x1, y1, x1 + 2 * half, y1 + 2 * half))


def build_card(fid: str, cand: dict[str, Any], label: str) -> Image.Image:
    """生成单个候选的三帧图卡（带编号表头）。

    Args:
        fid: 文件 ID。
        cand: 候选（t0/cx/cy/src）。
        label: 图卡编号文本。

    Returns:
        图卡图像。
    """
    side: int = int(2 * CROP_HALF * CARD_SCALE)
    card_w: int = side
    card_h: int = HEADER_H + side * len(STRIP_OFFSETS)
    card = Image.new("RGB", (card_w, card_h), (15, 15, 15))
    draw = ImageDraw.Draw(card)
    color = SRC_COLORS.get(cand.get("src", "static"), (80, 80, 80))
    draw.rectangle([0, 0, card_w, HEADER_H], fill=color)
    draw.text(
        (6, 7),
        f"{label} {fid} t={cand['t0']:.1f}s {cand.get('src', '?')}",
        fill=(255, 255, 255),
    )
    for row, off in enumerate(STRIP_OFFSETS):
        path: str = frame_path(fid, cand["t0"] + off)
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            logger.warning("缺帧: %s", path)
            continue
        tile = crop_around(img, cand["cx"], cand["cy"])
        tile = tile.resize((side, side), Image.LANCZOS)
        card.paste(tile, (0, HEADER_H + row * side))
    return card


def build_sheet(fid: str, cards: list[Image.Image], out_path: str) -> None:
    """把图卡拼成多列 sheet 落盘。

    Args:
        fid: 文件 ID。
        cards: 图卡列表。
        out_path: 输出路径。
    """
    if not cards:
        return
    card_w, card_h = cards[0].size
    rows: int = (len(cards) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * card_w, rows * card_h), (30, 30, 30))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % COLS) * card_w, (i // COLS) * card_h))
    sheet.save(out_path, quality=88)
    logger.info("  %s: %d候选 -> %s", fid, len(cards), out_path)


def main() -> None:
    """主入口：对各文件生成候选图卡 sheet 与 candidates.json。"""
    run_id: str = new_run_id()
    configure_logging(run_id)
    fids: list[str] = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FIDS
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records: list[dict[str, Any]] = []

    for fid in fids:
        cands = collect_file_candidates(fid)
        cards: list[Image.Image] = []
        for idx, cand in enumerate(cands, start=1):
            label: str = f"#{idx}"
            cards.append(build_card(fid, cand, label))
            record = dict(cand)
            record["fid"] = fid
            record["label"] = label
            all_records.append(record)
        build_sheet(fid, cards, os.path.join(OUT_DIR, f"{fid}_sheet.jpg"))

    json_path: str = os.path.join(OUT_DIR, "candidates.json")
    try:
        atomic_write_json(json_path, all_records, what="candidates.json")
    except OSError as exc:
        logger.error("candidates.json 写入失败: %s", exc)
        sys.exit(1)
    logger.info("共 %d 候选, 元数据 -> %s", len(all_records), json_path)


if __name__ == "__main__":
    main()
