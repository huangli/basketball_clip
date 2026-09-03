#!/usr/bin/env python3
"""生成候选审核视频（供用户人工标注 进球/非进球）。

读取 candidates.json，先把同一文件的候选聚类为"事件"（同一进球常触发多个
候选，重叠片段重复观看浪费审核时间；规则见 cluster_candidates：纯时间链
+时空放宽双条件），每个事件只出一个片段：覆盖 [首候选-2s, 末候选+2s]，
缩放 840x840 并烧录事件编号水印，按文件拼成一个审核 mp4。事件锚点取末
成员 t0（见 event_anchor，批次 1 实证 max-conf 锚点在长事件切错时段）。
裁剪：--hoops 提供筐轨迹时按轨迹包围盒自适应（全程见筐），否则回退 conf
最高候选为中心的固定裁剪。--keep-clips 时另写 events_index.json，按
hoop_dist（事件成员到筐轨迹的最小距离）升序排列，真球靠前供人工先标。

输入：--candidates 指定的 candidates.json（schema 损坏抛 SchemaError）；
    --vlmcache 指定的 VLM 缓存 JSON（可选；损坏仅记 WARNING 并忽略判定水印）；
    --srcdir/--orig 按场次注入原片目录与原片尺寸（缺省为旧 4:3 测试素材参数）；
    --hoops 指定的 hoops.json（可选；无则回退锚点裁剪）
输出：<outdir>/<fid>_events.mp4；--keep-clips 时另出 events_index.json（按 hoop_dist 升序；
    事件含 clip_src_start（片段原片起点）与 continued（跨文件续接标志），
    供标注页 J 键人工锚点换算/退化用，见 docs/goal-anchor/）
依赖：scripts/errors.py、scripts/pipe_common.py（run_ffmpeg/read_json/日志）
用法:
    python scripts/gen_review_clips.py --candidates work/label/candidates.json
    python scripts/gen_review_clips.py --candidates work/20260722/candidates.json
        --srcdir "素材目录" --orig 3840x2160
        --hoops work/20260722/hoops.json --vlmcache work/20260722/vlm_cache.json
"""

import logging
import math
import os
import sys
import time
from glob import escape, glob
from typing import Any

from errors import BasketballPipelineError, SchemaError
from extract_frames import probe_duration_sec
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json, run_ffmpeg

logger = logging.getLogger(__name__)

CANDIDATES_JSON: str = "work/label/candidates.json"
RAW_GLOB: str = (
    "archive/0_raw_videos_test/**/*_{fid}_D.MP4"  # 旧测试素材 2026-08-11 已删除；新场次参数化
)
OUT_DIR: str = "work/review"

CLUSTER_GAP_SEC: float = 2.0  # 候选间隔 <= 该值归为同一事件（同进球触发候选实测间隔 <=1.7s）
# 时空放宽合并：间隔 <=6.0s 且球位 (cx,cy) 距离 <=400px(img 系) 也归同一事件——
# 批次 1 实测 190354 同一进球两候选间隔 2.6s/309px 被纯时间链拆成两事件、合集重复；
# 篮板补篮类真动作间隔虽可能 >2s，锚点取末成员后仍正确
CLUSTER_MERGE_GAP_SEC: float = 6.0
CLUSTER_MERGE_DIST: float = 400.0
CLIP_BEFORE_SEC: float = 2.0  # 片段起点：事件首候选前
CLIP_AFTER_SEC: float = (
    4.0  # 片段终点：事件末候选后（批次 2 用户反馈：+2s 时补篮/筐沿跳舞类结局未含，+4s 覆盖）
)
MIN_HALF_IMG: int = 420  # img 系裁剪半径
CROP_MARGIN: int = 300  # 筐轨迹包围盒外扩边距（原片系 px）
MIN_ADAPTIVE_SIDE: int = 1200  # 自适应裁剪边长下限（原片系 px，保证框住筐区动作）
ORIG_W: int = 3840
ORIG_H: int = 2880
OUT_SIDE: int = 840
WIDE_W: int = 840  # 全景片段宽（投球人视角）
WIDE_H: int = 472  # 全景片段高（16:9 全帧缩放）
OUT_FPS: int = 30
SPEED: float = 2.0  # 审核视频加速倍率（用户可分辨；声音用 atempo 同步保留）
FONT_PATH: str = "C\\:/Windows/Fonts/arialbd.ttf"

# candidates.json 每条记录的必需字段（本脚本实际消费的字段：聚类/裁剪/水印）
REQUIRED_STR_FIELDS: tuple[str, ...] = ("fid", "label")
REQUIRED_NUM_FIELDS: tuple[str, ...] = ("t0", "ac", "cx", "cy")


def find_source(fid: str, srcdir: str = "") -> str | None:
    """按文件 ID 定位原片路径。

    指定 srcdir 时在其下递归匹配 <fid>.mp4（新场次 dji 命名，fid=主名）；
    否则按 RAW_GLOB 匹配（旧测试素材的 *_{fid}_D.MP4 命名）。

    Args:
        fid: 文件 ID（如 0011 / 原片主名）。
        srcdir: 新场次原片目录；空串走 RAW_GLOB。

    Returns:
        原片路径；找不到或找到多个返回 None。
    """
    pattern: str = (
        os.path.join(escape(srcdir), "**", f"{fid}.mp4") if srcdir else RAW_GLOB.format(fid=fid)
    )
    matches: list[str] = glob(pattern, recursive=True)
    if len(matches) != 1:
        logger.error("%s: 原片匹配数=%d，跳过", fid, len(matches))
        return None
    return matches[0]


def find_next_source(src: str, srcdir: str) -> str | None:
    """找 src 在场次目录中的下一个连续切片文件（dji 分段命名，文件名即拍摄时间）。

    Args:
        src: 当前原片路径。
        srcdir: 场次原片目录；空串表示旧测试素材（命名不连续，不续接）。

    Returns:
        下一个文件路径；无目录、当前为最后一个或匹配异常返回 None。
    """
    if not srcdir:
        return None
    files: list[str] = sorted(glob(os.path.join(escape(srcdir), "**", "*.mp4"), recursive=True))
    base: str = os.path.basename(src)
    idx: list[int] = [i for i, f in enumerate(files) if os.path.basename(f) == base]
    if len(idx) != 1 or idx[0] + 1 >= len(files):
        return None
    return files[idx[0] + 1]


MAX_CONT_SEC: float = 4.0  # 跨文件续接段最长秒数（审核片段尾巴最多 +4s）


def split_window(
    start: float, end: float, dur: float
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """把 [start, end] 按本文件时长 dur 拆为 (本文件段, 续文件段)；不越界返回 (整段, None)。

    续文件段从 0 起、长度上限 MAX_CONT_SEC（防异常时长把片段拖长）。

    Args:
        start: 窗口起点（秒）。
        end: 窗口终点（秒）。
        dur: 本文件时长（秒）。

    Returns:
        ((start, min(end, dur)), (0, end-dur) 或 None)。
    """
    if end <= dur:
        return (start, end), None
    cont: float = min(end - dur, MAX_CONT_SEC)
    return (start, dur), (0.0, cont)


def plan_clip_segments(
    src: str, start: float, end: float, srcdir: str
) -> tuple[list[tuple[str, float, float]], bool]:
    """规划片段切片：窗口越出本文件末尾时，续接到场次下一个切片文件。

    批次 2 实测 87/185 事件锚点+尾巴越出文件末（dji ~14s 连续切片，
    结局常在下一文件），此前片段被静默截断导致无法判读。

    Args:
        src: 当前原片路径。
        start: 窗口起点（秒）。
        end: 窗口终点（秒）。
        srcdir: 场次原片目录（空串不续接）。

    Returns:
        (切片段列表 [(文件, 起, 止), ...], 是否跨文件续接)；
        ffprobe 失败或无下一文件时回退单段截断。
    """
    try:
        dur: float = probe_duration_sec(src)
    except BasketballPipelineError as exc:
        logger.warning("时长探测失败，按不越界处理: %s: %s", os.path.basename(src), exc)
        return [(src, start, end)], False
    (a0, a1), cont = split_window(start, end, dur)
    if cont is None:
        return [(src, a0, a1)], False
    nxt: str | None = find_next_source(src, srcdir)
    if nxt is None:
        logger.warning("%s 窗口越界但无下一切片，截断在文件末", os.path.basename(src))
        return [(src, a0, a1)], False
    logger.info(
        "  跨文件续接: %s +%.1fs -> %s",
        os.path.basename(src),
        cont[1] - cont[0],
        os.path.basename(nxt),
    )
    return [(src, a0, a1), (nxt, cont[0], cont[1])], True


def _encode_clip(src: str, start: float, end: float, vf: str, out_path: str) -> None:
    """单段编码：裁剪/缩放/水印/2x 加速 + atempo（审核片段统一参数）。

    Args:
        src: 原片路径。
        start: 起点（秒）。
        end: 终点（秒）。
        vf: 视频滤镜串。
        out_path: 输出路径。
    """
    run_ffmpeg(
        [
            "-ss",
            f"{start:.2f}",
            "-to",
            f"{end:.2f}",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            vf,
            "-af",
            f"atempo={SPEED}",
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            out_path,
        ],
        timeout_sec=_encode_timeout_sec(end - start),
    )


def _concat_parts(parts: list[str], out_path: str) -> None:
    """把同参数分段 -c copy 拼接为 out_path 并清理临时文件。

    Args:
        parts: 分段路径（与 out_path 同目录）。
        out_path: 最终输出路径。
    """
    list_path: str = out_path.replace(".mp4", "_concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.basename(p)}'\n")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    for p in (*parts, list_path):
        os.remove(p)


def _render_segments(segs: list[tuple[str, float, float]], vf: str, out_path: str) -> None:
    """按 plan_clip_segments 的规划渲染片段：单段直接编码，多段编码后拼接。

    Args:
        segs: [(文件, 起, 止), ...]。
        vf: 视频滤镜串（各段相同，跨文件续接段复用同一取景框）。
        out_path: 输出路径。
    """
    if len(segs) == 1:
        src, a, b = segs[0]
        _encode_clip(src, a, b, vf, out_path)
        return
    parts: list[str] = []
    for n, (src, a, b) in enumerate(segs, start=1):
        part: str = out_path.replace(".mp4", f"_p{n}.mp4")
        _encode_clip(src, a, b, vf, part)
        parts.append(part)
    _concat_parts(parts, out_path)


def cluster_candidates(
    cands: list[dict[str, Any]],
    *,
    gap_sec: float = CLUSTER_GAP_SEC,
    merge_gap_sec: float = CLUSTER_MERGE_GAP_SEC,
    merge_dist: float = CLUSTER_MERGE_DIST,
) -> list[list[dict[str, Any]]]:
    """把候选聚类为事件（纯时间链 + 时空放宽双条件，满足任一即同事件）。

    相邻候选（与当前事件末成员比较）满足以下任一即归入同一事件：

    - 时间：t0 差 <= gap_sec（现状行为，同进球触发候选实测间隔 <=1.7s）；
    - 时空：t0 差 <= merge_gap_sec 且 (cx,cy) 欧氏距离 <= merge_dist——
      消 190354 式同球重复（实测同一进球两候选间隔 2.6s/309px 被纯时间链
      拆成两事件）；篮板补篮类真动作锚点取末成员后仍正确。

    Args:
        cands: 候选列表（任意顺序，须含 t0/cx/cy 字段）。
        gap_sec: 纯时间链合并的 t0 差上限（秒）。
        merge_gap_sec: 时空放宽合并的 t0 差上限（秒）。
        merge_dist: 时空放宽合并的球位距离上限（img 系像素）。

    Returns:
        事件列表，每个事件是候选列表（按 t0 升序）。
    """
    clusters: list[list[dict[str, Any]]] = []
    for c in sorted(cands, key=lambda c: c["t0"]):
        if clusters:
            prev: dict[str, Any] = clusters[-1][-1]
            gap: float = c["t0"] - prev["t0"]
            dist: float = math.hypot(c["cx"] - prev["cx"], c["cy"] - prev["cy"])
            if gap <= gap_sec or (gap <= merge_gap_sec and dist <= merge_dist):
                clusters[-1].append(c)
                continue
        clusters.append([c])
    return clusters


def event_anchor(members: list[dict[str, Any]]) -> float:
    """事件锚点 = 末成员 t0（动作链末端 = 球停网/落地）。

    批次 1 实证：长事件（0544/1508/1948）max-conf 锚点落在非进球成员、
    片段切错时段，末成员锚点全部命中；单成员事件行为不变。

    Args:
        members: 事件内候选（按 t0 升序，非空）。

    Returns:
        锚点时刻（秒）。
    """
    return members[-1]["t0"]


def clip_window(members: list[dict[str, Any]]) -> tuple[float, float]:
    """事件的审核片段窗口（原片秒）：首候选前 CLIP_BEFORE_SEC ~ 末候选后 CLIP_AFTER_SEC。

    cut_cluster_clip 与 events_index 的 clip_src_start 共用同一口径（防两处漂移）。

    Args:
        members: 事件内候选（按 t0 升序，非空）。

    Returns:
        (start, end)：start 钳到 ≥0。
    """
    return (
        max(0.0, members[0]["t0"] - CLIP_BEFORE_SEC),
        members[-1]["t0"] + CLIP_AFTER_SEC,
    )


def _encode_timeout_sec(duration_sec: float) -> int:
    """ffmpeg 转码超时：片段时长 ×3 + 60s 兜底，下限 120s（rules.md §4）。

    Args:
        duration_sec: 输入片段时长（秒）。

    Returns:
        超时秒数。
    """
    return max(120, int(duration_sec * 3) + 60)


def _validate_candidates(data: Any, path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """校验 candidates.json 结构，返回候选记录列表。

    顶层必须是记录列表；每条记录必须含 fid/label(str) 与
    t0/ac/cx/cy(数值)——即聚类、裁剪、水印实际消费的字段。

    Args:
        data: read_json 读出的 candidates.json 内容。
        path: 文件路径（仅用于错误信息）。

    Returns:
        校验通过的候选记录列表。

    Raises:
        SchemaError: 结构损坏（顶层非列表/记录非对象/缺字段/类型错），
            信息含路径与记录索引。
    """
    if not isinstance(data, list):
        raise SchemaError(f"{path}: 顶层必须是记录列表，实际 {type(data).__name__}")
    for i, r in enumerate(data):
        if not isinstance(r, dict):
            raise SchemaError(f"{path}: 第{i}条记录不是对象，实际 {type(r).__name__}")
        for key in REQUIRED_STR_FIELDS:
            if not isinstance(r.get(key), str):
                raise SchemaError(
                    f"{path}: 第{i}条 {key} 缺失或不是 str，实际 {type(r.get(key)).__name__}"
                )
        for key in REQUIRED_NUM_FIELDS:
            v = r.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SchemaError(f"{path}: 第{i}条 {key} 缺失或不是数值，实际 {type(v).__name__}")
    return data


def _load_vlm_cache(path: str) -> dict[str, Any]:
    """读取 VLM 判定缓存；未提供/缺失/损坏仅记 WARNING 并返回空（水印是可选项）。

    Args:
        path: VLM 缓存 JSON 路径；空串表示未提供。

    Returns:
        缓存 dict（键 "<fid>#N@尺度"）；不可用时空 dict。
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        data = read_json(path, what="VLM 缓存")
    except (SchemaError, OSError) as exc:
        logger.warning("VLM 缓存读取失败(%s)，忽略判定水印: %s", exc, path)
        return {}
    if not isinstance(data, dict):
        logger.warning("VLM 缓存顶层不是对象，忽略判定水印: %s", path)
        return {}
    if "_protocol" in data:
        answers: Any = data.get("answers")
        if not isinstance(answers, dict):
            logger.warning("VLM 缓存 answers 结构异常，忽略判定水印: %s", path)
            return {}
        return answers
    return data


def cluster_crop(
    members: list[dict[str, Any]],
    orig_w: int,
    orig_h: int,
) -> tuple[int, int, int]:
    """计算事件的裁剪参数（原片系）：以 conf 最高候选为中心的正方形。

    事件内候选位置可能散布（含 FP 成员），包围盒会过度拉远画面；
    审核关注点是真球/筐区，取 conf 最高者为中心。

    Args:
        members: 事件内候选（cx/cy 为 img 系坐标）。
        orig_w: 原片宽（像素）。
        orig_h: 原片高（像素）。

    Returns:
        (crop_x, crop_y, side)，均已收敛到原片范围内。
    """
    best: dict[str, Any] = max(members, key=lambda m: m["ac"])
    side: int = MIN_HALF_IMG * 4  # img 半径 -> 原片边长（坐标系 2 倍 + 双边 2 倍）
    crop_x: int = min(max(2 * best["cx"] - side // 2, 0), orig_w - side)
    crop_y: int = min(max(2 * best["cy"] - side // 2, 0), orig_h - side)
    return crop_x, crop_y, side


def adaptive_crop(
    track: list[list[Any]],
    orig_w: int,
    orig_h: int,
) -> tuple[int, int, int]:
    """按筐轨迹包围盒计算自适应裁剪框（原片系）：全程见筐是构造性保证。

    框 = 轨迹包围盒 + CROP_MARGIN 边距（下限 MIN_ADAPTIVE_SIDE，上限原片短边），
    中心取包围盒中心（手持跟拍下筐逐帧移动，均值中心会偏）。

    Args:
        track: hoops.json 事件 track（[[sec, cx, cy, src], ...]，img 系）。
        orig_w: 原片宽。
        orig_h: 原片高。

    Returns:
        (crop_x, crop_y, side)，均已收敛到原片范围内。
    """
    xs: list[int] = [int(p[1]) * 2 for p in track]  # img 系 -> 原片系（2 倍）
    ys: list[int] = [int(p[2]) * 2 for p in track]
    span_x: int = max(xs) - min(xs)
    span_y: int = max(ys) - min(ys)
    side: int = min(
        max(max(span_x, span_y) + 2 * CROP_MARGIN, MIN_ADAPTIVE_SIDE),
        min(orig_w, orig_h),
    )
    cx: int = (max(xs) + min(xs)) // 2
    cy: int = (max(ys) + min(ys)) // 2
    crop_x: int = min(max(cx - side // 2, 0), orig_w - side)
    crop_y: int = min(max(cy - side // 2, 0), orig_h - side)
    return crop_x, crop_y, side


def load_hoops(path: str) -> dict[str, list[dict[str, Any]]]:
    """读 hoops.json 并按 fid 分组事件；空路径/缺失/损坏返回空 dict（回退锚点裁剪）。

    与 vlm_filter.load_hoops 同构（此处复制以免把 torch 依赖拖进审核视频生成）。

    Args:
        path: hoops.json 路径；空串表示未提供。

    Returns:
        {fid: [事件, ...]}。
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        data = read_json(path, what="hoops.json")
    except (SchemaError, OSError) as exc:
        logger.warning("hoops.json 读取失败(%s)，回退锚点裁剪", exc)
        return {}
    by_fid: dict[str, list[dict[str, Any]]] = {}
    for ev in data.get("events", []) if isinstance(data, dict) else []:
        if isinstance(ev, dict) and ev.get("fid"):
            by_fid.setdefault(ev["fid"], []).append(ev)
    return by_fid


def find_event_track(
    events: list[dict[str, Any]],
    t0: float,
) -> list[list[Any]] | None:
    """找 window 含 t0 且 detected 的事件的筐轨迹。

    Args:
        events: 某 fid 的事件列表。
        t0: 事件锚点候选时刻（秒）。

    Returns:
        track（[[sec, cx, cy, src], ...]）；无命中返回 None。
    """
    for ev in events:
        if not ev.get("detected"):
            continue
        window: Any = ev.get("window")
        if (
            isinstance(window, list)
            and len(window) == 2
            and window[0] <= t0 <= window[1]
            and ev.get("track")
        ):
            return ev["track"]
    return None


def event_hoop_dist(
    members: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> float | None:
    """事件到筐的最小距离（img 系 px）：各成员 (cx,cy) 到其 t0 时刻筐位距离的最小值。

    每成员先用 find_event_track 的选取逻辑找 window 含其 t0 的筐轨迹，
    再取轨迹内 sec 最近点作为该时刻筐位；所有成员都无筐轨迹命中时返回 None。

    Args:
        members: 事件内候选（cx/cy 为 img 系坐标）。
        events: 该 fid 的 hoops.json 事件列表。

    Returns:
        最小筐距（img 系像素）；无筐轨迹命中为 None。
    """
    best: float | None = None
    for m in members:
        track: list[list[Any]] | None = find_event_track(events, m["t0"])
        if not track:
            continue
        pt: list[Any] = min(track, key=lambda p: abs(p[0] - m["t0"]))
        d: float = math.hypot(m["cx"] - pt[1], m["cy"] - pt[2])
        if best is None or d < best:
            best = d
    return best


def sort_events_by_hoop_dist(events: list[dict[str, Any]]) -> None:
    """events_index 原地按 hoop_dist 升序排序：真球靠前，人工先标高密度区。

    hoop_dist 为 None（无筐轨迹）的事件排最后；稳定排序，相同 hoop_dist 的
    事件保持各 fid 内原有时间序。只排序不裁剪——批次 1 实测真球筐距
    66/260/570px vs 全体中位 540px，排序信号有效但不足以自动剔除。

    Args:
        events: events_index 事件列表（每项含 hoop_dist 字段，数值或 None）。
    """
    events.sort(key=lambda ev: (ev["hoop_dist"] is None, ev["hoop_dist"] or 0.0))


def _event_watermark(
    fid: str,
    idx: int,
    members: list[dict[str, Any]],
    verdict: str,
    mark_no_hoop: bool,
) -> str:
    """拼事件水印文本（未转义）。"""
    text: str = (
        f"#{idx} {fid} t={members[0]['t0']:.1f}-{members[-1]['t0']:.1f}s ({len(members)}cand)"
    )
    if verdict:
        text = f"{text} {verdict}"
    if mark_no_hoop:
        text = f"{text} 无筐检出"
    return text


def cut_wide_clip(
    src: str,
    start: float,
    end: float,
    text: str,
    out_path: str,
    srcdir: str = "",
) -> bool:
    """裁出同一事件的全景片段（全帧缩放，供辨认投球人）。

    窗口越出本文件末尾时自动续接到场次下一个切片文件（srcdir 非空时）。

    Args:
        src: 原片路径。
        start: 片段起点（原片秒）。
        end: 片段终点（原片秒）。
        text: 水印文本（未转义）。
        out_path: 输出片段路径。
        srcdir: 场次原片目录（跨文件续接用；空串不续接）。

    Returns:
        是否跨文件续接（供 events_index 记录：续接片段时间轴不归零，
        标注页 J 键人工锚点换算不适用，须退化为机器锚）。
    """
    segs, continued = plan_clip_segments(src, start, end, srcdir)
    if continued:
        text += " 跨文件续接"
    text = text.replace("\\", "\\\\").replace(":", "\\:")  # drawtext 选项分隔符转义
    vf: str = (
        f"scale={WIDE_W}:{WIDE_H},fps={OUT_FPS},"
        f"setpts=PTS/{SPEED},"
        f"drawtext=fontfile='{FONT_PATH}':text='{text} 全景':"
        f"x=15:y=15:fontsize=30:fontcolor=yellow:"
        f"box=1:boxcolor=black@0.8"
    )
    _render_segments(segs, vf, out_path)
    return continued


def cut_cluster_clip(
    src: str,
    fid: str,
    idx: int,
    members: list[dict[str, Any]],
    out_path: str,
    verdict: str = "",
    orig: tuple[int, int] = (ORIG_W, ORIG_H),
    hoop_track: list[list[Any]] | None = None,
    mark_no_hoop: bool = False,
    srcdir: str = "",
) -> bool:
    """裁出单个事件的审核片段（裁剪 + 事件编号水印）。

    窗口越出本文件末尾时自动续接到场次下一个切片文件（srcdir 非空时）；
    续接段复用同一取景框（筐轨迹包围盒 ≥1200px 边长，2~4s 内相机漂移在容差内）。

    Args:
        src: 原片路径。
        fid: 文件 ID。
        idx: 事件编号（1 起）。
        members: 事件内候选列表（按 t0 升序）。
        out_path: 输出片段路径。
        verdict: 可选 VLM 判定水印（如 "VLM:Y"）。
        orig: 原片（宽，高），按场次注入（16:9 新素材为 3840x2160）。
        hoop_track: 筐轨迹（hoops.json）；提供时自适应裁剪，全程见筐。
        mark_no_hoop: 已提供 hoops 但本事件无筐检出，水印追加"无筐检出"。
        srcdir: 场次原片目录（跨文件续接用；空串不续接）。

    Returns:
        是否跨文件续接（供 events_index 记录，标注页人工锚点退化用）。
    """
    start, end = clip_window(members)
    segs, continued = plan_clip_segments(src, start, end, srcdir)
    if hoop_track:
        crop_x, crop_y, side = adaptive_crop(hoop_track, orig[0], orig[1])
    else:
        crop_x, crop_y, side = cluster_crop(members, orig[0], orig[1])
    text: str = _event_watermark(fid, idx, members, verdict, mark_no_hoop)
    if continued:
        text += " 跨文件续接"
    text = text.replace("\\", "\\\\").replace(":", "\\:")  # drawtext 选项分隔符转义
    vf: str = (
        f"crop={side}:{side}:{crop_x}:{crop_y},"
        f"scale={OUT_SIDE}:{OUT_SIDE},fps={OUT_FPS},"
        f"setpts=PTS/{SPEED},"
        f"drawtext=fontfile='{FONT_PATH}':text='{text}':"
        f"x=15:y=15:fontsize=30:fontcolor=yellow:"
        f"box=1:boxcolor=black@0.8"
    )
    _render_segments(segs, vf, out_path)
    return continued


def concat_clips(clips: list[str], list_path: str, out_path: str) -> None:
    """同参数片段 concat 重封装为单文件。

    Args:
        clips: 片段路径列表（同目录）。
        list_path: concat 清单文件路径。
        out_path: 输出视频路径。
    """
    with open(list_path, "w", encoding="utf-8") as f:
        for clip in clips:
            # 片段可能在 clips/ 子目录（--keep-clips）：写相对清单文件所在目录的
            # 相对路径，并统一正斜杠（ffmpeg concat 在 Windows 下反斜杠会被当转义）
            rel: str = os.path.relpath(clip, os.path.dirname(list_path)).replace(os.sep, "/")
            f.write(f"file '{rel}'\n")
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out_path,
        ]
    )


def parse_argv() -> tuple[str, str, str, str, tuple[int, int], str, bool]:
    """解析命令行参数。

    Returns:
        (candidates 路径, 输出目录, VLM 缓存路径, 原片目录, 原片(宽,高),
        hoops.json 路径, 是否保留单事件片段)；
        原片目录空串表示走 RAW_GLOB 旧测试素材，hoops 空串表示回退锚点裁剪。
    """
    candidates: str = CANDIDATES_JSON
    out_dir: str = OUT_DIR
    vlm_cache: str = ""
    srcdir: str = ""
    orig: tuple[int, int] = (ORIG_W, ORIG_H)
    hoops: str = ""
    keep_clips: bool = False
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--candidates" and i + 1 < len(args):
            candidates = args[i + 1]
            i += 2
        elif args[i] == "--outdir" and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        elif args[i] == "--vlmcache" and i + 1 < len(args):
            vlm_cache = args[i + 1]
            i += 2
        elif args[i] == "--srcdir" and i + 1 < len(args):
            srcdir = args[i + 1]
            i += 2
        elif args[i] == "--orig" and i + 1 < len(args):
            w, h = args[i + 1].lower().split("x")
            orig = (int(w), int(h))
            i += 2
        elif args[i] == "--hoops" and i + 1 < len(args):
            hoops = args[i + 1]
            i += 2
        elif args[i] == "--keep-clips":
            keep_clips = True
            i += 1
        else:
            i += 1
    return candidates, out_dir, vlm_cache, srcdir, orig, hoops, keep_clips


def event_verdict(members: list[dict[str, Any]], vlm: dict[str, Any]) -> str:
    """事件的 VLM 判定：任一成员 YES 即 Y；无 YES 有 UNCLEAR 即 ?；否则 N。

    缓存键形如 "<fid>#N@尺度"，按前缀扫描（尺度集合随 --halves 可变）。

    Args:
        members: 事件内候选。
        vlm: VLM 缓存 answers（已解包）。

    Returns:
        "VLM:Y" / "VLM:?" / "VLM:N"（至少一个成员有有效判定）/ ""（全部未判）。
    """
    if not vlm:
        return ""
    n_judged: int = 0
    has_unclear: bool = False
    for m in members:
        prefix: str = f"{m['fid']}{m['label']}@"
        for k, v in vlm.items():
            if not k.startswith(prefix):
                continue
            ans: str = v.get("answer", "")
            if ans == "YES":
                return "VLM:Y"
            if ans == "NO":
                n_judged += 1
            elif ans == "UNCLEAR":
                has_unclear = True
    if has_unclear:
        return "VLM:?"
    return "VLM:N" if n_judged else ""


def main() -> int:
    """主入口：按文件生成事件级候选审核视频。

    Returns:
        进程退出码：0=成功；1=数据损坏/IO 失败/合成失败/有原片缺失。
    """
    run_id = new_run_id()
    configure_logging(run_id)
    candidates_path, out_dir, vlm_cache_path, srcdir, orig, hoops_path, keep_clips = parse_argv()
    try:
        os.makedirs(out_dir, exist_ok=True)
        records: list[dict[str, Any]] = _validate_candidates(
            read_json(candidates_path, what="candidates.json"), candidates_path
        )
        vlm: dict[str, Any] = _load_vlm_cache(vlm_cache_path)
        hoops_by_fid: dict[str, list[dict[str, Any]]] = load_hoops(hoops_path)

        fids: list[str] = sorted({r["fid"] for r in records})
        missing: list[str] = []
        index_events: list[dict[str, Any]] = []
        clip_dir: str = os.path.join(out_dir, "clips") if keep_clips else out_dir
        if keep_clips:
            os.makedirs(clip_dir, exist_ok=True)
        for fid in fids:
            src: str | None = find_source(fid, srcdir)
            if src is None:
                logger.error("原片缺失，跳过该文件的全部事件: %s", fid)
                missing.append(fid)
                continue
            clusters: list[list[dict[str, Any]]] = cluster_candidates(
                [r for r in records if r["fid"] == fid]
            )
            t_start: float = time.time()
            clips: list[str] = []
            for idx, members in enumerate(clusters, start=1):
                clip: str = os.path.join(clip_dir, f"{fid}_e{idx}.mp4")
                anchor_t0: float = event_anchor(members)
                track: list[list[Any]] | None = find_event_track(
                    hoops_by_fid.get(fid, []), anchor_t0
                )
                verdict: str = event_verdict(members, vlm)
                no_hoop: bool = bool(hoops_path) and track is None
                cont_clip: bool = cut_cluster_clip(
                    src,
                    fid,
                    idx,
                    members,
                    clip,
                    verdict,
                    orig,
                    track,
                    mark_no_hoop=no_hoop,
                    srcdir=srcdir,
                )
                clips.append(clip)
                if keep_clips:
                    wide_path: str = clip.replace(".mp4", "_wide.mp4")
                    start, end = clip_window(members)
                    cont_wide: bool = cut_wide_clip(
                        src,
                        start,
                        end,
                        _event_watermark(fid, idx, members, verdict, no_hoop),
                        wide_path,
                        srcdir=srcdir,
                    )
                    hoop_dist: float | None = event_hoop_dist(members, hoops_by_fid.get(fid, []))
                    index_events.append(
                        {
                            "key": f"{fid}#e{idx}",
                            "fid": fid,
                            "event_idx": idx,
                            "clip": os.path.relpath(clip, out_dir),
                            "clip_wide": os.path.relpath(wide_path, out_dir),
                            "src_file": os.path.basename(src),
                            "anchor_t0": round(anchor_t0, 1),
                            "clip_src_start": round(start, 1),
                            "continued": cont_clip or cont_wide,
                            "hoop_dist": round(hoop_dist) if hoop_dist is not None else None,
                            "verdict": verdict,
                        }
                    )
            out_path: str = os.path.join(out_dir, f"{fid}_events.mp4")
            list_path: str = os.path.join(out_dir, f"{fid}_concat.txt")
            concat_clips(clips, list_path, out_path)
            if not keep_clips:
                for clip in clips:
                    os.remove(clip)
            os.remove(list_path)
            logger.info(
                "  %s: %d候选 -> %d事件 -> %s (%.0fs)",
                fid,
                sum(len(c) for c in clusters),
                len(clusters),
                out_path,
                time.time() - t_start,
            )

        if keep_clips:
            # 按筐距升序（None 排最后）：真球靠前，人工先标高密度区；只排序不裁剪。
            # 剪辑文件命名/生成顺序不变（仍按 fid 时间序），仅 index 数组顺序变。
            sort_events_by_hoop_dist(index_events)
            atomic_write_json(
                os.path.join(out_dir, "events_index.json"),
                {"events": index_events},
                what="events_index.json",
            )
            logger.info("事件索引 %d 条 -> %s", len(index_events), "events_index.json")

        if missing:
            logger.error("共 %d 个文件原片缺失: %s", len(missing), ", ".join(missing))
            return 1
        logger.info("完成。")
        return 0
    except BasketballPipelineError as exc:
        logger.error("管线失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
