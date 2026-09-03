"""出手点回溯实测器：热图阶段 0 · Q2 判据（回溯成功率）实测（docs/heatmap/spec.md）。

输入：work/<场次>/goals_batchK.json（confirmed 进球）、work/detect/<fid>_mot_cache.json、
    work/frames/<fid>/f_*.jpg（仅取帧尺寸做 P3 贴边观察项，不读像素）
输出：work/<场次>/release_probe.json（逐球结果 + 汇总 + 一致性抽查）+ 控制台报告
依赖：crop_scorers.load_mot_cache / find_held_box（复用已验证的缓存校验与持球判定）
典型调用：python scripts/release_probe.py --sessiondir work/20260801_对手名

口径（plan.md P1/P2 写死）：
    出手时刻 = 锚点前 0.4~2.5s 窗口内最后一段稳定段的末帧（球离手即出手）；
    稳定段 = 有球帧按间隔 ≤2 帧（0.4s）分段，段内 ≥2 帧且相邻位移均 ≤60px；
    持球人 = 窗口内（不晚于出手帧）最后一个球心落入的人框（find_held_box）；
    落点 = 持球人框底边中点；缺缓存/缺帧/无稳定段/球心不落人框均计未命中入分母。

实现注记（spec 句读解释，审查备案）："间隔 ≤2 帧分段"只针对断帧；相邻位移 >60px
的运动跳变同样切段（等价于段内恒满足 P2）。若按"先按间隔分段再整段检验"的字面
读法，持球段与入网飞行段（间隔常仅 1 帧）会链成一段被整体判不稳，与 P1"取最后
稳定段末帧"的意图冲突。
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from crop_scorers import MotCache, find_held_box, load_mot_cache
from errors import BasketballPipelineError, SchemaError
from geom import Box, iou
from mot_candidates import Detection, Track
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json

logger = logging.getLogger(__name__)

# ---- 判据常量（docs/heatmap/spec.md + plan.md P1/P2 写死，勿散落修改）----
FPS: float = 5.0  # 抽帧帧率（sec = frame_idx / FPS，frame_idx 0-based）
WINDOW_MIN_BEFORE_SEC: float = 0.4  # 窗口近端：排除入网飞行段（瞬移高发段，经验教训 §3）
WINDOW_MAX_BEFORE_SEC: float = 2.5  # 窗口远端：扩大持球段搜索面
SEG_MAX_GAP_FRAMES: int = 2  # 有球帧间隔 ≤2 帧（0.4s）属同段（断帧不重置）
SEG_MIN_DETS: int = 2  # 稳定段最少有球帧数
SEG_MAX_STEP_PX: float = 60.0  # 段内相邻有球帧位移上限（P2）
EDGE_MARGIN_PX: int = 8  # P3 观察项：人框距帧边 ≤8px 记贴边
OVERLAP_MIN_IOU: float = 0.1  # P3 观察项：与他人框 IoU ≥0.1 记重叠
P3_WARN_RATIO: float = 0.3  # review02 提示线：P3 高发（>30%）时报告单独提示
CONSISTENCY_SHIFT_SEC: float = 0.4  # 一致性抽查：窗口整体平移量（±）
CONSISTENCY_MAX_DIST_PX: float = 100.0  # 一致性判据：平移后落点距离上限
CONSISTENCY_SAMPLE_N: int = 10  # 一致性抽查球数
Q2_MIN_COVERAGE: float = 0.70  # Q2 过关线（spec 成功标准写死）


@dataclass(frozen=True, slots=True)
class GoalEvent:
    """一条 confirmed 进球事件（goals_batchK.json 归一后）。

    Attributes:
        fid: 视频文件主键（file 去 .mp4）。
        anchor_time: 进球锚点（秒，球入网瞬间）。
        event_key: 事件主键 ``fid@anchor_time``（plan.md P4，与 goals 条目一一对应）。
    """

    fid: str
    anchor_time: float
    event_key: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """单球回溯结果；hit=False 时 release/held 取 -1、landing/box 取 None。

    Attributes:
        event_key: 事件主键。
        fid: 视频文件主键。
        anchor_time: 进球锚点（秒）。
        hit: 是否回溯到稳定出手点。
        reason: 未命中原因（missing_cache/missing_frames/empty_window/
            no_ball_detection/no_stable_segment/ball_not_in_box）；命中时为空串。
        n_window_dets: 窗口内有球帧数（断帧分布统计用）。
        n_stable_segments: 窗口内稳定段数。
        release_frame_idx: 出手帧索引（P1：最后稳定段末帧）。
        release_sec: 出手时刻（秒）。
        held_frame_idx: 持球判定帧索引（find_held_box 命中帧，≤ release_frame_idx）。
        landing_px: 落点（持球人框底边中点，像素）。
        person_box: 持球人框。
        edge_touch: P3 观察项：人框贴帧边。
        overlap: P3 观察项：人框与他人框重叠（IoU ≥ OVERLAP_MIN_IOU）。
    """

    event_key: str
    fid: str
    anchor_time: float
    hit: bool
    reason: str
    n_window_dets: int
    n_stable_segments: int
    release_frame_idx: int
    release_sec: float
    held_frame_idx: int
    landing_px: tuple[float, float] | None
    person_box: Box | None
    edge_touch: bool
    overlap: bool


def load_goals_events(path: str | Path) -> list[GoalEvent]:
    """读取 goals_batchK.json 并做 schema 校验，返回 confirmed 事件（rules.md §0.2）。

    Args:
        path: goals_batchK.json 路径。

    Returns:
        confirmed 事件列表（``file`` 去 .mp4 得 fid，event_key=``fid@anchor_time``）。

    Raises:
        SchemaError: 顶层结构/字段类型损坏，或 (fid, anchor_time) 重复。
    """
    data: Any = read_json(path, what="goals")
    if not isinstance(data, dict) or not isinstance(data.get("goals"), list):
        raise SchemaError(f"{path}: 顶层必须是含 goals 列表的对象")
    events: list[GoalEvent] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["goals"]):
        if not isinstance(raw, dict):
            raise SchemaError(f"{path}: goals[{i}] 不是对象")
        file_raw: Any = raw.get("file")
        anchor_raw: Any = raw.get("anchor_time")
        status_raw: Any = raw.get("status")
        if not isinstance(file_raw, str) or not file_raw:
            raise SchemaError(f"{path}: goals[{i}].file 缺失或非字符串: {file_raw!r}")
        if isinstance(anchor_raw, bool) or not isinstance(anchor_raw, (int, float)):
            raise SchemaError(f"{path}: goals[{i}].anchor_time 非法: {anchor_raw!r}")
        if anchor_raw < 0:
            raise SchemaError(f"{path}: goals[{i}].anchor_time 为负: {anchor_raw!r}")
        if not isinstance(status_raw, str):
            raise SchemaError(f"{path}: goals[{i}].status 缺失或非字符串: {status_raw!r}")
        if status_raw != "confirmed":
            continue
        fid: str = file_raw.removesuffix(".mp4")
        anchor: float = float(anchor_raw)
        key: str = f"{fid}@{anchor}"
        if key in seen:
            raise SchemaError(f"{path}: 事件重复 {key}")
        seen.add(key)
        events.append(GoalEvent(fid=fid, anchor_time=anchor, event_key=key))
    return events


def window_indices(
    anchor_sec: float,
    n_frames: int,
    *,
    min_before: float = WINDOW_MIN_BEFORE_SEC,
    max_before: float = WINDOW_MAX_BEFORE_SEC,
) -> tuple[int, int] | None:
    """把锚点前 [max_before, min_before] 秒窗口换算为闭区间帧索引（越界裁剪）。

    Args:
        anchor_sec: 进球锚点（秒）。
        n_frames: 缓存总帧数。
        min_before: 窗口近端（锚点前多少秒，含）。
        max_before: 窗口远端（锚点前多少秒，含）。

    Returns:
        (lo, hi) 闭区间帧索引；窗口为空或完全越界返回 None。
    """
    lo: int = max(math.ceil((anchor_sec - max_before) * FPS - 1e-9), 0)
    hi: int = min(math.floor((anchor_sec - min_before) * FPS + 1e-9), n_frames - 1)
    if lo > hi:
        return None
    return lo, hi


def collect_window_dets(cache: MotCache, lo: int, hi: int) -> list[Detection]:
    """取窗口内逐帧最高置信度球检测（一帧多球取 conf 最大者），按帧升序。

    Args:
        cache: 校验后的 mot_cache。
        lo: 窗口首帧索引（含）。
        hi: 窗口末帧索引（含）。

    Returns:
        窗口内有球帧的检测列表，按 frame_idx 升序。
    """
    out: list[Detection] = []
    for idx in range(lo, hi + 1):
        dets = cache.balls[idx]
        if dets:
            out.append(max(dets, key=lambda d: d.conf))
    return out


def find_stable_segments(dets: list[Detection]) -> list[tuple[Detection, ...]]:
    """把窗口有球帧切成稳定段（P2）。

    切段条件（两种都切）：相邻有球帧间隔 > SEG_MAX_GAP_FRAMES 帧（断帧），或
    相邻位移 > SEG_MAX_STEP_PX（运动跳变，如球离手入网）。段内恒满足相邻位移
    ≤60px；段长 ≥ SEG_MIN_DETS 个有球帧记稳定段。断帧不重置段，但只有球帧
    参与位移计算。

    Args:
        dets: 窗口内有球帧检测，按 frame_idx 升序。

    Returns:
        稳定段列表，按时间升序；每段为检测元组。
    """
    if not dets:
        return []
    segments: list[list[Detection]] = [[dets[0]]]
    for det in dets[1:]:
        prev: Detection = segments[-1][-1]
        gap: int = det.frame_idx - prev.frame_idx
        step: float = math.hypot(det.cx - prev.cx, det.cy - prev.cy)
        if gap <= SEG_MAX_GAP_FRAMES and step <= SEG_MAX_STEP_PX:
            segments[-1].append(det)
        else:
            segments.append([det])
    return [tuple(s) for s in segments if len(s) >= SEG_MIN_DETS]


def _miss(
    event: GoalEvent,
    reason: str,
    *,
    n_window_dets: int = 0,
    n_stable_segments: int = 0,
    release: Detection | None = None,
) -> ProbeResult:
    """构造未命中结果（release 仅在已定位出手帧但持球判定失败时非空）。"""
    return ProbeResult(
        event_key=event.event_key,
        fid=event.fid,
        anchor_time=event.anchor_time,
        hit=False,
        reason=reason,
        n_window_dets=n_window_dets,
        n_stable_segments=n_stable_segments,
        release_frame_idx=release.frame_idx if release is not None else -1,
        release_sec=release.sec if release is not None else -1.0,
        held_frame_idx=-1,
        landing_px=None,
        person_box=None,
        edge_touch=False,
        overlap=False,
    )


def probe_event(
    cache: MotCache,
    event: GoalEvent,
    frame_w: int,
    frame_h: int,
    *,
    min_before: float = WINDOW_MIN_BEFORE_SEC,
    max_before: float = WINDOW_MAX_BEFORE_SEC,
) -> ProbeResult:
    """单球回溯：窗口取球 → 稳定段 → P1 出手帧 → 持球人 → 底边中点落点。

    出手帧 = 窗口内最后一段稳定段的末帧（P1）；持球人 = 不晚于出手帧的最后一
    个球心落入人框（窗口有球帧截到出手帧包成 Track，复用 find_held_box）；
    落点 = 该人框底边中点。

    Args:
        cache: 校验后的 mot_cache。
        event: 进球事件。
        frame_w: 帧图宽（像素，P3 贴边判定用）。
        frame_h: 帧图高（像素）。
        min_before: 窗口近端秒数（一致性抽查平移用）。
        max_before: 窗口远端秒数（一致性抽查平移用）。

    Returns:
        ProbeResult；各失败路径 reason 见 ProbeResult 文档，均计入分母。
    """
    win: tuple[int, int] | None = window_indices(
        event.anchor_time, cache.frames, min_before=min_before, max_before=max_before
    )
    if win is None:
        return _miss(event, "empty_window")
    dets: list[Detection] = collect_window_dets(cache, win[0], win[1])
    if not dets:
        return _miss(event, "no_ball_detection")
    segments: list[tuple[Detection, ...]] = find_stable_segments(dets)
    if not segments:
        return _miss(event, "no_stable_segment", n_window_dets=len(dets))
    release: Detection = segments[-1][-1]
    track: Track = Track(dets=[d for d in dets if d.frame_idx <= release.frame_idx])
    held: tuple[Detection, Box] | None = find_held_box(track, cache.persons)
    if held is None:
        return _miss(
            event,
            "ball_not_in_box",
            n_window_dets=len(dets),
            n_stable_segments=len(segments),
            release=release,
        )
    held_det, box = held
    landing: tuple[float, float] = ((box.x1 + box.x2) / 2.0, float(box.y2))
    edge: bool = (
        box.x1 <= EDGE_MARGIN_PX
        or box.y1 <= EDGE_MARGIN_PX
        or box.x2 >= frame_w - EDGE_MARGIN_PX
        or box.y2 >= frame_h - EDGE_MARGIN_PX
    )
    overlap: bool = any(
        other is not box and iou(box, other) >= OVERLAP_MIN_IOU
        for other in cache.persons[held_det.frame_idx]
    )
    return ProbeResult(
        event_key=event.event_key,
        fid=event.fid,
        anchor_time=event.anchor_time,
        hit=True,
        reason="",
        n_window_dets=len(dets),
        n_stable_segments=len(segments),
        release_frame_idx=release.frame_idx,
        release_sec=release.sec,
        held_frame_idx=held_det.frame_idx,
        landing_px=landing,
        person_box=box,
        edge_touch=edge,
        overlap=overlap,
    )


def frame_size(frames_root: Path, fid: str) -> tuple[int, int] | None:
    """取该 fid 的帧图尺寸（读首张可用帧，只取头不解码全图）。

    Args:
        frames_root: 帧图根目录（work/frames）。
        fid: 视频文件主键。

    Returns:
        (宽, 高)；帧目录缺失或目录下无帧图返回 None。

    Raises:
        OSError: 帧文件损坏（中间产物损坏须显式失败，rules.md §0.2）。
    """
    d: Path = frames_root / fid
    if not d.is_dir():
        return None
    img: Path = d / "f_00001.jpg"
    if not img.exists():
        found: list[Path] = sorted(d.glob("f_*.jpg"))
        if not found:
            return None
        img = found[0]
    with Image.open(img) as im:
        return im.size


def even_sample(items: list[str], n: int) -> list[str]:
    """等间距抽样（确定性）：首末必含，len(items) ≤ n 时全取。

    Args:
        items: 已排序的候选清单。
        n: 目标样本数。

    Returns:
        抽样结果，保持原顺序。
    """
    if len(items) <= n:
        return list(items)
    step: float = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def _consistency_checks(
    hits: list[ProbeResult],
    events: dict[str, GoalEvent],
    caches: dict[str, MotCache],
    sizes: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """一致性抽查：命中球等间距抽样，窗口 ±0.4s 平移复测，落点距离 ≤100px 判一致。

    Args:
        hits: 命中结果列表。
        events: event_key → 事件。
        caches: fid → mot_cache（仅含已加载）。
        sizes: fid → 帧尺寸（仅含已读取）。

    Returns:
        逐球抽查记录（early/late 两方向的命中与否与落点距离、整体一致判定）。
    """
    by_key: dict[str, ProbeResult] = {r.event_key: r for r in hits}
    sample: list[str] = even_sample(sorted(by_key), CONSISTENCY_SAMPLE_N)
    checks: list[dict[str, Any]] = []
    for key in sample:
        base: ProbeResult = by_key[key]
        event: GoalEvent = events[key]
        w, h = sizes[event.fid]
        entry: dict[str, Any] = {"event_key": key}
        dists: list[float] = []
        both_hit: bool = True
        for name, shift in (("early", CONSISTENCY_SHIFT_SEC), ("late", -CONSISTENCY_SHIFT_SEC)):
            r: ProbeResult = probe_event(
                caches[event.fid],
                event,
                w,
                h,
                min_before=WINDOW_MIN_BEFORE_SEC + shift,
                max_before=WINDOW_MAX_BEFORE_SEC + shift,
            )
            entry[f"{name}_hit"] = r.hit
            if r.hit and r.landing_px is not None and base.landing_px is not None:
                dist: float = math.hypot(
                    r.landing_px[0] - base.landing_px[0], r.landing_px[1] - base.landing_px[1]
                )
                entry[f"{name}_dist_px"] = round(dist, 1)
                dists.append(dist)
            else:
                entry[f"{name}_dist_px"] = None
                both_hit = False
        entry["consistent"] = both_hit and all(d <= CONSISTENCY_MAX_DIST_PX for d in dists)
        checks.append(entry)
    return checks


def probe_session(
    session_dir: Path, detect_dir: Path, frames_dir: Path, out_path: Path
) -> dict[str, Any]:
    """全量回溯实测：读 goals_batch*.json → 逐球 probe → 汇总 → 落盘 + 控制台报告。

    缺 mot_cache / 缺帧目录的球记 WARNING 并计未命中入分母（保守口径）；
    goals / mot_cache schema 损坏显式抛错停跑（rules.md §0.2）。

    Args:
        session_dir: work/<场次> 目录（含 goals_batchK.json）。
        detect_dir: mot_cache 目录（work/detect）。
        frames_dir: 帧图根目录（work/frames）。
        out_path: 输出 JSON 路径（release_probe.json）。

    Returns:
        报告字典（summary / consistency / events），已原子写入 out_path。

    Raises:
        BasketballPipelineError: session_dir 下无 goals_batchK.json。
        SchemaError: goals / mot_cache schema 损坏，或事件跨批次重复。
    """
    goals_files: list[Path] = sorted(session_dir.glob("goals_batch*.json"))
    if not goals_files:
        raise BasketballPipelineError(f"{session_dir} 下无 goals_batch*.json（先完成标注导出）")
    events: dict[str, GoalEvent] = {}
    for gf in goals_files:
        for e in load_goals_events(gf):
            if e.event_key in events:
                raise SchemaError(f"跨批次事件重复: {e.event_key}")
            events[e.event_key] = e
    logger.info("共 %d 个 confirmed 事件（%d 个批次文件）", len(events), len(goals_files))

    caches: dict[str, MotCache] = {}
    sizes: dict[str, tuple[int, int]] = {}
    missing_cache: set[str] = set()
    missing_frames: set[str] = set()
    results: list[ProbeResult] = []
    for e in events.values():
        if e.fid in missing_cache:
            results.append(_miss(e, "missing_cache"))
            continue
        if e.fid in missing_frames:
            results.append(_miss(e, "missing_frames"))
            continue
        cache_path: Path = detect_dir / f"{e.fid}_mot_cache.json"
        if not cache_path.exists():
            logger.warning("缺 mot_cache，计未命中: %s（%s）", e.event_key, cache_path.name)
            missing_cache.add(e.fid)
            results.append(_miss(e, "missing_cache"))
            continue
        if e.fid not in sizes:
            size: tuple[int, int] | None = frame_size(frames_dir, e.fid)
            if size is None:
                logger.warning("缺帧目录/帧图，计未命中: %s（%s）", e.event_key, e.fid)
                missing_frames.add(e.fid)
                results.append(_miss(e, "missing_frames"))
                continue
            sizes[e.fid] = size
        if e.fid not in caches:
            caches[e.fid] = load_mot_cache(cache_path)
        w, h = sizes[e.fid]
        results.append(probe_event(caches[e.fid], e, w, h))

    total: int = len(results)
    hits: list[ProbeResult] = [r for r in results if r.hit]
    coverage: float = len(hits) / total if total else 0.0
    miss_by_reason: dict[str, int] = {}
    window_dets_hist: dict[str, int] = {}
    stable_seg_hist: dict[str, int] = {}
    for r in results:
        if not r.hit:
            miss_by_reason[r.reason] = miss_by_reason.get(r.reason, 0) + 1
        if r.reason not in ("missing_cache", "missing_frames"):
            k: str = str(r.n_window_dets)
            window_dets_hist[k] = window_dets_hist.get(k, 0) + 1
        if r.hit:
            k = str(r.n_stable_segments)
            stable_seg_hist[k] = stable_seg_hist.get(k, 0) + 1
    n_edge: int = sum(1 for r in hits if r.edge_touch)
    n_overlap: int = sum(1 for r in hits if r.overlap)
    checks: list[dict[str, Any]] = _consistency_checks(hits, events, caches, sizes)
    n_consistent: int = sum(1 for c in checks if c["consistent"])

    report: dict[str, Any] = {
        "session": session_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "window_before_sec": [WINDOW_MIN_BEFORE_SEC, WINDOW_MAX_BEFORE_SEC],
            "seg_max_gap_frames": SEG_MAX_GAP_FRAMES,
            "seg_min_dets": SEG_MIN_DETS,
            "seg_max_step_px": SEG_MAX_STEP_PX,
            "edge_margin_px": EDGE_MARGIN_PX,
            "overlap_min_iou": OVERLAP_MIN_IOU,
            "consistency_shift_sec": CONSISTENCY_SHIFT_SEC,
            "consistency_max_dist_px": CONSISTENCY_MAX_DIST_PX,
            "q2_min_coverage": Q2_MIN_COVERAGE,
        },
        "summary": {
            "total": total,
            "hits": len(hits),
            "coverage": round(coverage, 4),
            "q2_pass": coverage >= Q2_MIN_COVERAGE,
            "miss_by_reason": miss_by_reason,
            "window_dets_hist": window_dets_hist,
            "stable_segments_hist": stable_seg_hist,
            "p3": {
                "hits": len(hits),
                "edge_touch": n_edge,
                "overlap": n_overlap,
                "either": sum(1 for r in hits if r.edge_touch or r.overlap),
            },
        },
        "consistency": {
            "sample_n": len(checks),
            "n_consistent": n_consistent,
            "checks": checks,
        },
        "events": [asdict(r) for r in results],
    }
    atomic_write_json(out_path, report, what="release_probe.json")

    s = report["summary"]
    logger.info("==== Q2 回溯实测汇总 ====")
    logger.info(
        "覆盖率: %d/%d = %.1f%%（阈值 %.0f%%）→ %s",
        s["hits"],
        s["total"],
        coverage * 100,
        Q2_MIN_COVERAGE * 100,
        "过关" if s["q2_pass"] else "未过关",
    )
    logger.info("未命中原因分布: %s", miss_by_reason)
    logger.info("窗口有球帧数分布（断帧分布）: %s", window_dets_hist)
    logger.info("稳定段数分布（命中球）: %s", stable_seg_hist)
    logger.info(
        "P3 观察项（命中 %d 球）: 贴边 %d、重叠 %d、任一 %d",
        s["p3"]["hits"],
        n_edge,
        n_overlap,
        s["p3"]["either"],
    )
    if hits and s["p3"]["either"] / len(hits) > P3_WARN_RATIO:
        logger.warning(
            "P3 高发（%.0f%% > %.0f%%）：贴边/重叠比例偏高，落点口径存在系统性风险，报告须单独提示",
            s["p3"]["either"] / len(hits) * 100,
            P3_WARN_RATIO * 100,
        )
    logger.info(
        "一致性抽查: %d/%d 一致（±%.1fs 平移，落点距离 ≤%.0fpx）",
        n_consistent,
        len(checks),
        CONSISTENCY_SHIFT_SEC,
        CONSISTENCY_MAX_DIST_PX,
    )
    logger.info("报告已落盘: %s", out_path)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="出手点回溯实测（热图阶段 0 · Q2 判据）")
    parser.add_argument("--sessiondir", required=True, type=Path, help="work/<场次> 目录")
    parser.add_argument(
        "--detectdir", type=Path, default=None, help="mot_cache 目录（默认 <sessiondir>/../detect）"
    )
    parser.add_argument(
        "--framesdir", type=Path, default=None, help="帧图根目录（默认 <sessiondir>/../frames）"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="输出 JSON（默认 <sessiondir>/release_probe.json）"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=实测完成，非 0=管线失败；Q2 未过关不算失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    session_dir: Path = args.sessiondir
    detect_dir: Path = args.detectdir or session_dir.parent / "detect"
    frames_dir: Path = args.framesdir or session_dir.parent / "frames"
    out_path: Path = args.out or session_dir / "release_probe.json"
    try:
        probe_session(session_dir, detect_dir, frames_dir, out_path)
    except BasketballPipelineError as e:
        logger.error("回溯实测失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
