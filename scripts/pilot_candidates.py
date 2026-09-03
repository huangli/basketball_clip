#!/usr/bin/env python3
"""试点候选生成：由检测缓存产出 candidates.json（供 VLM 精筛与审核视频）。

复用 mot_candidates 的 MOT/静止段/断轨重连/合并逻辑，
对指定的文件 ID 列表生成统一候选文件（默认 work/pilot/candidates.json，
--out 可改；新场次建议 work/<场次>/candidates.json）。

用法:
    python scripts/pilot_candidates.py 0007 0014
    python scripts/pilot_candidates.py --out work/20260722/candidates.json <fid...>
"""

import logging
import os
import sys
from glob import glob
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mot_candidates as mot
from pipe_common import atomic_write_json, configure_logging, new_run_id

logger = logging.getLogger(__name__)

OUT_JSON: str = "work/pilot/candidates.json"
PILOT_FIDS: list[str] = [
    "0007",
    "0014",
    "0022",
    "0033",
    "0048",
    "0062",
    "0086",
    "0102",
    "0120",
    "0147",
]


def collect_file_candidates(fid: str) -> list[dict[str, Any]]:
    """由检测缓存产出单个文件的合并候选（静止段 + 断轨重连，时空去重）。

    Args:
        fid: 文件 ID。

    Returns:
        候选列表（按 t0 排序，含 t0/dur/ac/cx/cy/src）；无缓存返回空列表。
    """
    frames = sorted(glob(mot.FRAMES_PATTERN.format(fid)))
    cached = mot.load_detection_cache(fid, len(frames))
    if cached is None:
        logger.warning("%s: 无检测缓存，跳过", fid)
        return []
    all_balls, all_persons = cached
    tracks = mot.run_mot(all_balls)
    static_cands, _, _, _ = mot.collect_candidates(tracks, all_persons)
    rejoin_cands = mot.find_rejoin_candidates(all_balls)
    cands, _ = mot.merge_candidates(static_cands, rejoin_cands)
    return cands


def main() -> None:
    """主入口：对试点文件生成 candidates.json。"""
    run_id: str = new_run_id()
    configure_logging(run_id)
    fids: list[str] = []
    out_json: str = OUT_JSON
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_json = args[i + 1]
            i += 2
        else:
            fids.append(args[i])
            i += 1
    if not fids:
        fids = PILOT_FIDS
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    all_records: list[dict[str, Any]] = []
    for fid in fids:
        cands = collect_file_candidates(fid)
        for idx, cand in enumerate(cands, start=1):
            record = dict(cand)
            record["fid"] = fid
            record["label"] = f"#{idx}"
            all_records.append(record)
        logger.info("  %s: %d 候选", fid, len(cands))
    try:
        atomic_write_json(out_json, all_records, what="candidates.json")
    except OSError as exc:
        logger.error("写入 %s 失败: %s", out_json, exc)
        sys.exit(1)
    logger.info("共 %d 候选 -> %s", len(all_records), out_json)


if __name__ == "__main__":
    main()
