#!/usr/bin/env python3
"""筐补检：对候选事件帧区间取 Hoop 类检测，产出筐轨迹。

背景：主检测缓存（work/detect/<fid>_mot_cache.json）自 2026-08-16 起由
mot_candidates 顺带存 Hoop 类结果（"hoops" 键，docs/detect-hoops-cache/）。
本脚本按候选聚类出事件，事件窗口内每帧**优先读缓存**；旧缓存（无 hoops 键）/
缓存缺失/损坏则回退逐帧 YOLO 补检（模型懒加载，全批缓存命中时不加载）。
两条路径产物逐点一致（缓存量化口径复刻 detect_hoop_frame）。
用途：为 VLM 输入裁剪与审核视频裁剪提供"筐在哪"（hoops.json）。

输入：candidates.json（fid/label/t0/dur/ac/cx/cy）
输出：hoops.json（schema 见下，下游 vlm_filter / gen_review_clips 共用此契约）
依赖：models/abdullahtarek_ball.pt（Hoop 类 id=2）、mot_candidates（帧路径/parse_sec）、
    gen_review_clips.cluster_candidates（事件聚类）、pipe_common
典型调用：
    python scripts/detect_hoops.py --candidates work/20260722/candidates.json \
        --out work/20260722/hoops.json [--fid <单fid调试>] [--limit N]

hoops.json schema（坐标为 img 系 1920×1080 像素）：
    {"session": str, "params": {"conf":..,"imgsz":..,"max_gap_s":..},
     "events": [{"key": "<fid>#e<N>", "fid": str, "event_idx": int,
                 "window": [t_start, t_end], "anchor": [cx, cy],
                 "detected": bool,
                 "track": [[sec, cx, cy, "det"|"interp"], ...]}]}
detected=false 时 track 为 []。

选筐与追踪：以事件锚点（ac 最高候选）时刻为起点，取离锚点最近的筐，
向两侧最近邻连续追踪（相邻帧跳变 >150px 截断，防跳到另一筐/海报假筐）；
缺口 ≤0.6s（3 帧）线性插值（标 "interp"），更大缺口断开不补（云台转向时
线性插值会造出横贯全场的幽灵轨迹）。
"""

import itertools
import json
import logging
import os
import sys
from glob import glob
from typing import Any

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mot_candidates as mot
from errors import BasketballPipelineError, SchemaError
from gen_review_clips import cluster_candidates
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json

logger = logging.getLogger(__name__)

CONF: float = 0.25  # Hoop 检出置信度（实测 0.34~0.47，留裕度）
IMGSZ: int = 1280  # 与主检测一致
HOOP_CLS: int = 2  # abdullahtarek 模型 Hoop 类 id
MAX_JUMP_PX: int = 150  # 相邻帧筐中心最大允许跳变（防跳另一筐/假筐）
MAX_GAP_FRAMES: int = 3  # 缺口插值上限（3 帧 = 0.6s @5fps）
EVENT_BEFORE_SEC: float = 2.0  # 事件窗口：首候选前
EVENT_AFTER_SEC: float = 2.0  # 事件窗口：末候选后（另加候选 dur）
PROGRESS_EVERY: int = 10  # 每 N 事件一条进度日志


def select_hoop(
    detections: list[tuple[int, int]],
    anchor: tuple[int, int],
) -> tuple[int, int] | None:
    """多筐选离锚点最近者；空列表返回 None。

    Args:
        detections: 单帧全部筐中心 [(cx, cy), ...]。
        anchor: 事件锚点 (cx, cy)。

    Returns:
        最近筐中心；无检出返回 None。
    """
    if not detections:
        return None
    return min(detections, key=lambda d: (d[0] - anchor[0]) ** 2 + (d[1] - anchor[1]) ** 2)


def track_hoop(
    per_frame: list[tuple[float, list[tuple[int, int]]]],
    anchor: tuple[int, int],
    anchor_sec: float,
    max_jump_px: int = MAX_JUMP_PX,
) -> list[tuple[float, int, int, str]]:
    """从离 anchor_sec 最近的有检出帧起步，向两侧最近邻连续追踪单筐轨迹。

    起步帧取"有检出且 sec 离 anchor_sec 最近"的帧，起点筐 = 离 anchor 最近者；
    之后逐帧取离上一点最近的筐，跳变 >max_jump_px 即截断（该方向结束）。

    Args:
        per_frame: [(sec, [筐中心...]), ...]，按 sec 升序。
        anchor: 事件锚点 (cx, cy)。
        anchor_sec: 锚点时刻（秒）。
        max_jump_px: 相邻帧最大跳变。

    Returns:
        [(sec, cx, cy, "det")] 按 sec 升序；全程无检出返回 []。
    """
    detected_idx: list[int] = [i for i, (_, dets) in enumerate(per_frame) if dets]
    if not detected_idx:
        return []
    start: int = min(detected_idx, key=lambda i: abs(per_frame[i][0] - anchor_sec))
    first: tuple[int, int] | None = select_hoop(per_frame[start][1], anchor)
    if first is None:
        return []
    points: dict[int, tuple[int, int]] = {start: first}
    for i in range(start + 1, len(per_frame)):
        if not per_frame[i][1]:
            continue
        prev_i: int = max(j for j in points if j < i)
        nxt: tuple[int, int] | None = select_hoop(per_frame[i][1], points[prev_i])
        if nxt is None or _dist(points[prev_i], nxt) > max_jump_px:
            break
        points[i] = nxt
    for i in range(start - 1, -1, -1):
        if not per_frame[i][1]:
            continue
        prev_j: int = min(j for j in points if j > i)
        nxt = select_hoop(per_frame[i][1], points[prev_j])
        if nxt is None or _dist(points[prev_j], nxt) > max_jump_px:
            break
        points[i] = nxt
    return [(per_frame[i][0], p[0], p[1], "det") for i, p in sorted(points.items())]


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    """两点欧氏距离。"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def interpolate_gaps(
    track: list[tuple[float, int, int, str]],
    all_secs: list[float],
    max_gap_frames: int = MAX_GAP_FRAMES,
) -> list[tuple[float, int, int, str]]:
    """对轨迹内 ≤max_gap_frames 帧的缺口做线性插值（标 "interp"），更大缺口不补。

    Args:
        track: [(sec, cx, cy, "det")] 升序。
        all_secs: 窗口内全部帧时刻（升序）。
        max_gap_frames: 允许插值的最大缺口帧数。

    Returns:
        含插值点的轨迹（升序）。
    """
    if len(track) < 2:
        return track
    out: list[tuple[float, int, int, str]] = []
    for (s0, x0, y0, _), (s1, x1, y1, _) in itertools.pairwise(track):
        out.append((s0, x0, y0, "det"))
        missing: list[float] = [s for s in all_secs if s0 < s < s1]
        if 0 < len(missing) <= max_gap_frames and s1 > s0:
            for s in missing:
                r: float = (s - s0) / (s1 - s0)
                out.append((s, round(x0 + (x1 - x0) * r), round(y0 + (y1 - y0) * r), "interp"))
    out.append(track[-1])
    return out


def detect_hoop_frame(model: Any, img_path: str) -> list[tuple[int, int]]:  # noqa: ANN401 YOLO 类型不定
    """单帧 Hoop 检测，返回全部筐中心。

    Args:
        model: 已加载的 YOLO 模型。
        img_path: 帧图片路径。

    Returns:
        [(cx, cy), ...]（img 系像素）。
    """
    res = model(img_path, conf=CONF, imgsz=IMGSZ, classes=[HOOP_CLS], verbose=False)[0]
    centers: list[tuple[int, int]] = []
    for b in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return centers


def load_hoop_frames(fid: str, expected_frames: int) -> list[list[tuple[int, int]]] | None:
    """从 mot 检测缓存读事件无关的全帧筐检测（docs/detect-hoops-cache/）。

    缓存量化口径与 detect_hoop_frame 一致（int 截断 cx/cy + 原始 conf），
    此处按 CONF 过滤（严格大于，与 ultralytics NMS 的 `> conf_thres` 一致，
    utils/nms.py:81）后与逐帧直检产物逐点等价。任何一环不符即回退 None
    （调用方走逐帧 YOLO 补检），不半路崩（rules.md §0.2）。

    Args:
        fid: 文件 ID。
        expected_frames: 当前帧目录下的帧数（与缓存 frames 字段比对）。

    Returns:
        每帧筐中心列表（与排序后帧 glob 索引对齐）；缓存缺失/旧版/损坏返回 None。
    """
    path: str = mot.CACHE_PATTERN.format(fid)
    if not os.path.exists(path):
        logger.info("  %s: 筐检测无缓存，逐帧补检", fid)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload: Any = json.load(f)
        if not isinstance(payload, dict):
            logger.warning("  %s: 缓存结构损坏（顶层非 dict），逐帧补检", fid)
            return None
        if payload.get("frames") != expected_frames:
            logger.info("  %s: 缓存帧数不符，逐帧补检", fid)
            return None
        hoops_raw: Any = payload.get("hoops")
        if hoops_raw is None:
            logger.info("  %s: 旧缓存无 hoops 键，逐帧补检", fid)
            return None
        if not isinstance(hoops_raw, list) or len(hoops_raw) != expected_frames:
            logger.warning("  %s: 缓存 hoops 结构损坏，逐帧补检", fid)
            return None
        out: list[list[tuple[int, int]]] = []
        for frame_hoops in hoops_raw:
            if not isinstance(frame_hoops, list):
                raise TypeError(f"hoops 帧条目非 list: {type(frame_hoops)}")
            centers = []
            for h in frame_hoops:
                conf, cx, cy = h["conf"], h["cx"], h["cy"]
                if not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in (conf, cx, cy)
                ):
                    raise TypeError(f"hoops 条目字段类型错: {h}")
                if conf > CONF:
                    centers.append((int(cx), int(cy)))
            out.append(centers)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("  %s: 缓存 hoops 读取失败(%s)，逐帧补检", fid, exc)
        return None
    logger.info(
        "  %s: 筐检测命中 mot 缓存（%d 帧，含筐 %d 帧）",
        fid,
        expected_frames,
        sum(1 for c in out if c),
    )
    return out


def _event_window(members: list[dict[str, Any]], max_sec: float) -> list[float]:
    """事件窗口：[首候选-2s, 末候选+dur+2s]，clamp 到 [0, max_sec]。"""
    w0: float = max(0.0, members[0]["t0"] - EVENT_BEFORE_SEC)
    w1: float = min(max_sec, members[-1]["t0"] + members[-1].get("dur", 0.0) + EVENT_AFTER_SEC)
    return [w0, w1]


def _validate_candidates(data: Any, path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """candidates.json 最小校验（detect_hoops 消费 fid/t0/dur/ac/cx/cy）。"""
    if not isinstance(data, list):
        raise SchemaError(f"{path}: 顶层必须是记录列表")
    for i, r in enumerate(data):
        if not isinstance(r, dict) or not isinstance(r.get("fid"), str):
            raise SchemaError(f"{path}: 第{i}条缺 fid")
        for key in ("t0", "ac", "cx", "cy"):
            v = r.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SchemaError(f"{path}: 第{i}条 {key} 缺失或不是数值")
    return data


def _parse_argv() -> tuple[str, str, str, int]:
    """解析 CLI：(candidates 路径, 输出路径, 单 fid 调试, 事件上限)。"""
    candidates: str = ""
    out: str = ""
    only_fid: str = ""
    limit: int = 0
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--candidates" and i + 1 < len(args):
            candidates = args[i + 1]
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
        elif args[i] == "--fid" and i + 1 < len(args):
            only_fid = args[i + 1]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1
    return candidates, out, only_fid, limit


def main() -> int:
    """主入口：聚类事件→逐事件补检筐→写 hoops.json。

    Returns:
        进程退出码：0=全部成功；1=参数/数据错误或有事件失败；130=人工中断（部分落盘）。
    """
    run_id: str = new_run_id()
    configure_logging(run_id)
    candidates_path, out_path, only_fid, limit = _parse_argv()
    if not candidates_path or not out_path:
        logger.error("缺少 --candidates / --out 参数")
        return 1
    try:
        records: list[dict[str, Any]] = _validate_candidates(
            read_json(candidates_path, what="candidates.json"), candidates_path
        )
    except BasketballPipelineError as exc:
        logger.error("数据损坏 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1

    model: Any = None  # 懒加载：全批缓存命中时不加载 YOLO（docs/detect-hoops-cache/）
    hoop_cache: dict[str, list[list[tuple[int, int]]] | None] = {}  # 逐 fid 选路结果
    events: list[dict[str, Any]] = []
    fids: list[str] = sorted({r["fid"] for r in records})
    for fid in fids:
        if only_fid and fid != only_fid:
            continue
        members_all: list[dict[str, Any]] = [r for r in records if r["fid"] == fid]
        for idx, members in enumerate(cluster_candidates(members_all), start=1):
            anchor_c: dict[str, Any] = max(members, key=lambda m: m["ac"])
            events.append(
                {
                    "key": f"{fid}#e{idx}",
                    "fid": fid,
                    "event_idx": idx,
                    "members": members,
                    "anchor": (anchor_c["cx"], anchor_c["cy"]),
                    "anchor_sec": anchor_c["t0"],
                }
            )
    if limit > 0:
        events = events[:limit]
    total: int = len(events)
    logger.info("共 %d 个事件待补检筐", total)

    out_events: list[dict[str, Any]] = []
    failed: int = 0
    try:
        for n, ev in enumerate(events, 1):
            try:
                frames: list[str] = sorted(glob(mot.FRAMES_PATTERN.format(ev["fid"])))
                if not frames:
                    raise BasketballPipelineError(f"{ev['fid']}: 无帧目录")
                secs: list[float] = [mot.parse_sec(p) for p in frames]
                window: list[float] = _event_window(ev["members"], secs[-1])
                if ev["fid"] not in hoop_cache:
                    hoop_cache[ev["fid"]] = load_hoop_frames(ev["fid"], len(frames))
                hoop_frames = hoop_cache[ev["fid"]]
                per_frame: list[tuple[float, list[tuple[int, int]]]] = []
                for i, (p, s) in enumerate(zip(frames, secs, strict=True)):
                    if window[0] <= s <= window[1]:
                        if hoop_frames is not None:
                            centers = hoop_frames[i]
                        else:
                            if model is None:
                                model = YOLO(mot.BALL_MODEL_PATH)
                            centers = detect_hoop_frame(model, p)
                        per_frame.append((s, centers))
                track: list[tuple[float, int, int, str]] = track_hoop(
                    per_frame, ev["anchor"], ev["anchor_sec"]
                )
                track = interpolate_gaps(track, [s for s, _ in per_frame])
                out_events.append(
                    {
                        "key": ev["key"],
                        "fid": ev["fid"],
                        "event_idx": ev["event_idx"],
                        "window": [round(window[0], 1), round(window[1], 1)],
                        "anchor": list(ev["anchor"]),
                        "detected": bool(track),
                        "track": [list(p) for p in track],
                    }
                )
            except (BasketballPipelineError, OSError) as exc:
                failed += 1
                logger.error("事件 %s 失败: %s", ev["key"], exc)
            if n % PROGRESS_EVERY == 0:
                logger.info(
                    "  进度 %d/%d（有筐 %d）", n, total, sum(e["detected"] for e in out_events)
                )
    except KeyboardInterrupt:
        logger.warning("中断，已完成 %d/%d 事件，先落盘", len(out_events), total)
        _write_out(out_path, candidates_path, out_events)
        return 130

    _write_out(out_path, candidates_path, out_events)
    n_det: int = sum(e["detected"] for e in out_events)
    logger.info("完成: %d/%d 事件有筐轨迹（失败 %d）-> %s", n_det, total, failed, out_path)
    return 1 if failed else 0


def _write_out(out_path: str, candidates_path: str, out_events: list[dict[str, Any]]) -> None:
    """写 hoops.json（session 取 candidates 路径父目录名）。"""
    session: str = os.path.basename(os.path.dirname(candidates_path)) or "unknown"
    payload: dict[str, Any] = {
        "session": session,
        "params": {"conf": CONF, "imgsz": IMGSZ, "max_gap_s": MAX_GAP_FRAMES / 5.0},
        "events": out_events,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    atomic_write_json(out_path, payload, what="hoops.json")


if __name__ == "__main__":
    raise SystemExit(main())
