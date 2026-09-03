#!/usr/bin/env python3
"""abdullahtarek + MOT 轨迹聚类 pipeline。

用多目标跟踪（MOT）替代"取最高conf"：每帧保留所有球检测，
用最近邻匹配建立轨迹，在轨迹中找连续静止子段作为入网候选。

阶段1改动：
- 参数恢复精确率最优组合（80/40/2；150/150/3 已证候选翻倍且召回无实质提升）；
- 检测结果缓存到 work/detect/<fid>_mot_cache.json，避免重复跑 YOLO；
- GT 窗口诊断输出（dump_gt_window），观察断轨前后的原始检测；
- 严格容差（2.0s）命中评估：3.0s 容差已证会让远处假阳性碰巧命中；
- 断轨重连签名（find_rejoin_candidates）：入网遮挡/相机平移导致检测盲区
  （0011 实测盲区 0.8s、0128 实测 1.6s），静止段判据对此无解；
  用"消失→重现"对直接定位进球，anchor 取间隔中点。
"""

import json
import logging
import os
import re
import sys
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from glob import glob
from typing import Any

from ultralytics import YOLO

from geom import Box, coverage
from pipe_common import atomic_write_json, configure_logging, new_run_id

logger = logging.getLogger(__name__)

BALL_MODEL_PATH: str = "models/abdullahtarek_ball.pt"
BALL_CLS: int = 0
HOOP_CLS: int = 2  # abdullahtarek 模型 Hoop 类 id（与 detect_hoops.HOOP_CLS 同源同值）
PERSON_MODEL_PATH: str = "models/yolov8n.pt"
PERSON_CLS: int = 0

IMGSZ_BALL: int = 1280
IMGSZ_PERSON: int = 640
CONF_BALL: float = 0.15
CONF_PERSON: float = 0.3

SAMPLE_FPS: float = 5.0

MAX_MATCH_DIST: int = 80
MAX_MISSED: int = 2

STATIC_WINDOW: int = 4
STATIC_MAX_MOVE: int = 40
MERGE_GAP: int = 4
DEAD_BALL_SEC: float = 3.0
HELD_COVERAGE: float = 0.5  # 球框过半落入人框视为持球（替代原 IoU>0.3，实测数学上不可触发）
GT_TOLERANCE: float = 3.0
GT_STRICT_TOLERANCE: float = 2.0
PROGRESS_LOG_EVERY: int = 200  # 检测进度日志间隔（帧）；rules.md §4 要求长循环可观测

# 断轨重连签名参数（用 0011/0128 漏检标定）
REJOIN_MIN_GAP_SEC: float = 0.6  # a、b 最小间隔（>=2 连续缺失帧；0.4s 多为检测抖动假配对）
REJOIN_MAX_GAP_SEC: float = 1.4  # 遮挡盲区上限（0128 实测 1.2s）
REJOIN_MAX_DIST: int = 400  # a→b 最大位移（0011 实测 377px）
REJOIN_VOID_DIST: int = 250  # 间隙期 a 的该半径内不得有任何检测
REJOIN_PRE_SEC: float = 0.6  # a 须有此前驱时限
REJOIN_PRE_DIST: int = 150  # a 的前驱距离上限（排除孤立单帧 FP 作起点）
REJOIN_MIN_CONF: float = 0.3  # b（重现端）conf 下限；真球重现 conf 实测 >=0.36，弱检测多为 FP
REJOIN_FOLLOW_SEC: float = 0.8  # b 之后须有后继的时限
REJOIN_FOLLOW_DIST: int = 200  # b 的后继距离上限（重现后持续可见；0011 实测 165px）
DEDUPE_SEC: float = 1.5  # 候选去重时间窗
DEDUPE_DIST: int = 200  # 候选去重空间窗

FRAMES_PATTERN: str = "work/frames/{}/f_*.jpg"
CACHE_PATTERN: str = "work/detect/{}_mot_cache.json"

GROUND_TRUTH: dict[str, float] = {
    "0011": 10.0,
    "0030": 11.0,
    "0040": 39.0,
    "0128": 11.0,
}


@dataclass
class Detection:
    """单帧单个球检测。

    Attributes:
        conf: 置信度。
        box: [x1, y1, x2, y2] 边界框。
        cx: 框中心 x。
        cy: 框中心 y。
        sec: 时间戳（秒）。
        frame_idx: 在帧列表中的 0-based 索引。
    """

    conf: float
    box: list[int]
    cx: int
    cy: int
    sec: float
    frame_idx: int


@dataclass
class Track:
    """球轨迹，由连续帧的检测组成。

    Attributes:
        dets: 轨迹中的检测列表。
        missed: 连续未匹配帧数。
    """

    dets: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def length(self) -> int:
        """轨迹中检测数量。"""
        return len(self.dets)

    @property
    def last_det(self) -> Detection:
        """最后一个检测。"""
        return self.dets[-1]

    @property
    def start_sec(self) -> float:
        """轨迹起始时间。"""
        return self.dets[0].sec

    @property
    def duration_sec(self) -> float:
        """轨迹持续时间（秒）。"""
        return round(self.dets[-1].sec - self.dets[0].sec, 1)

    @property
    def avg_conf(self) -> float:
        """平均置信度。"""
        return round(sum(d.conf for d in self.dets) / len(self.dets), 2)

    @property
    def centers(self) -> list[tuple[int, int]]:
        """所有检测中心坐标列表。"""
        return [(d.cx, d.cy) for d in self.dets]


def parse_sec(img_path: str) -> float:
    """从帧路径解析时间戳。

    Args:
        img_path: 帧文件路径，如 work/frames/0030/f_056.jpg。

    Returns:
        时间戳（秒）。
    """
    m = re.search(r"f_(\d+)", img_path)
    idx: int = int(m.group(1)) if m else 0
    return round((idx - 1) / SAMPLE_FPS, 1)


def euclidean(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """计算两点欧氏距离。

    Args:
        p1: 第一个点 (x, y)。
        p2: 第二个点 (x, y)。

    Returns:
        欧氏距离。
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def detect_frame(
    ball_model: YOLO,
    person_model: YOLO,
    img_path: str,
    frame_idx: int,
) -> tuple[list[Detection], list[list[int]], list[dict[str, Any]]]:
    """单帧检测所有球、筐和人物。

    球模型一次推理同取 Ball+Hoop 两类（实测仅 +1.7% 成本，docs/detect-hoops-cache/），
    筐检测顺带落缓存，detect_hoops 阶段据此免重复推理。

    Args:
        ball_model: 球检测 YOLO 模型。
        person_model: 人物检测 YOLO 模型。
        img_path: 帧图片路径。
        frame_idx: 帧在列表中的 0-based 索引。

    Returns:
        (球检测列表, 人物框列表, 筐检测列表)。
        筐条目 {"conf","cx","cy"}：量化口径复刻 detect_hoops.detect_hoop_frame——
        cx/cy 用 int() 截断取整（不用 round），conf 存原始 float 不截断，
        保证缓存路径与逐帧直检路径产物逐点一致（docs/detect-hoops-cache/spec.md B1）。
    """
    sec: float = parse_sec(img_path)

    rb = ball_model(
        img_path,
        conf=CONF_BALL,
        imgsz=IMGSZ_BALL,
        classes=[BALL_CLS, HOOP_CLS],
        verbose=False,
    )
    rp = person_model(
        img_path,
        conf=CONF_PERSON,
        imgsz=IMGSZ_PERSON,
        classes=[PERSON_CLS],
        verbose=False,
    )

    balls: list[Detection] = []
    hoops: list[dict[str, Any]] = []
    for b in rb[0].boxes:
        if int(b.cls[0]) == HOOP_CLS:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            hoops.append({"conf": float(b.conf[0]), "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2})
            continue
        box: list[int] = [round(v) for v in b.xyxy[0].tolist()]
        balls.append(
            Detection(
                conf=round(float(b.conf), 2),
                box=box,
                cx=(box[0] + box[2]) // 2,
                cy=(box[1] + box[3]) // 2,
                sec=sec,
                frame_idx=frame_idx,
            )
        )

    persons: list[list[int]] = [[round(v) for v in b.xyxy[0].tolist()] for b in rp[0].boxes]

    return balls, persons, hoops


def save_detection_cache(
    fid: str,
    all_balls: list[list[Detection]],
    all_persons: list[list[list[int]]],
    all_hoops: list[list[dict[str, Any]]],
) -> None:
    """把检测结果落盘为 JSON 缓存。

    Args:
        fid: 文件 ID。
        all_balls: 每帧的球检测列表。
        all_persons: 每帧的人物框列表。
        all_hoops: 每帧的筐检测列表（{"conf","cx","cy"}；detect_hoops 消费，
            mot 自身不消费）。
    """
    path: str = CACHE_PATTERN.format(fid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, Any] = {
        "frames": len(all_balls),
        "balls": [
            [
                {
                    "conf": d.conf,
                    "box": d.box,
                    "cx": d.cx,
                    "cy": d.cy,
                    "sec": d.sec,
                    "frame_idx": d.frame_idx,
                }
                for d in frame_dets
            ]
            for frame_dets in all_balls
        ],
        "persons": all_persons,
        "hoops": all_hoops,
    }
    try:
        atomic_write_json(path, payload, what="检测缓存")
    except OSError as exc:
        logger.warning("  缓存写入失败(%s)，不影响本次结果", exc)


def load_detection_cache(
    fid: str, expected_frames: int
) -> tuple[list[list[Detection]], list[list[list[int]]]] | None:
    """读取检测缓存；帧数不符或结构损坏均视为失效，重检兜底。

    hoops 键（筐检测，docs/detect-hoops-cache/）为增量可选键：本函数不校验
    （无 hoops 键的旧缓存仍算命中，mot 自身不消费，不为加键触发全量重跑）；
    由 detect_hoops 读取时自行判读与校验。

    Args:
        fid: 文件 ID。
        expected_frames: 当前帧目录下的帧数。

    Returns:
        (每帧球检测, 每帧人物框)；缓存不存在或失效返回 None。
    """
    path: str = CACHE_PATTERN.format(fid)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("  缓存读取失败(%s)，重新检测", exc)
        return None
    try:
        if payload.get("frames") != expected_frames:
            logger.info("  缓存帧数不符，重新检测")
            return None
        all_balls: list[list[Detection]] = [
            [Detection(**d) for d in frame_dets] for frame_dets in payload["balls"]
        ]
        all_persons: list[list[list[int]]] = payload["persons"]
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning("  缓存结构损坏(%s)，重新检测", exc)
        return None
    return all_balls, all_persons


def run_mot(
    all_balls: list[list[Detection]],
    *,
    min_length: int = STATIC_WINDOW,
    max_match_dist: float = MAX_MATCH_DIST,
) -> list[Track]:
    """简单 MOT：贪心最近邻匹配跟踪所有球检测。

    Args:
        all_balls: 每帧的球检测列表。
        min_length: 收录轨迹的最小长度（检测数）；默认 STATIC_WINDOW（候选挖掘口径）。
            crop_scorers 轨迹法定位传 1——窗口内短轨迹（入网/落地片段）也是有效证据。
        max_match_dist: 帧间最近邻匹配门限（像素）；默认 MAX_MATCH_DIST（80，
            为静止/慢速球候选挖掘标定）。落点链路（goal_heatmap）传放宽值——
            飞行球 5fps 下帧间位移轻松超 80px，默认门限会碎成单点轨迹
            （docs/heatmap-flight-link/）。

    Returns:
        所有长度 >= min_length 的轨迹列表。
    """
    active: list[Track] = []
    finished: list[Track] = []

    for dets in all_balls:
        available: list[Detection] = list(dets)

        for track in active:
            if track.missed > MAX_MISSED:
                continue
            last: Detection = track.last_det
            best_i: int = -1
            best_d: float = float(max_match_dist)
            for i, det in enumerate(available):
                d: float = euclidean((last.cx, last.cy), (det.cx, det.cy))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                track.dets.append(available[best_i])
                track.missed = 0
                available.pop(best_i)
            else:
                track.missed += 1

        for det in available:
            active.append(Track(dets=[det]))

        still: list[Track] = []
        for track in active:
            if track.missed > MAX_MISSED:
                if track.length >= min_length:
                    finished.append(track)
            else:
                still.append(track)
        active = still

    for track in active:
        if track.length >= min_length:
            finished.append(track)

    return finished


def find_static_segments_in_track(
    track: Track,
) -> list[tuple[int, int]]:
    """在单条轨迹中找连续 N 帧位置不动的子段。

    Args:
        track: 球轨迹。

    Returns:
        合并后的 (start, end_exclusive) 索引区间列表。
    """
    centers: list[tuple[int, int]] = track.centers
    raw: list[int] = []
    for i in range(len(centers) - STATIC_WINDOW + 1):
        seg: list[tuple[int, int]] = centers[i : i + STATIC_WINDOW]
        cxs: list[int] = [c[0] for c in seg]
        cys: list[int] = [c[1] for c in seg]
        if max(cxs) - min(cxs) < STATIC_MAX_MOVE and max(cys) - min(cys) < STATIC_MAX_MOVE:
            raw.append(i)

    if not raw:
        return []

    merged: list[tuple[int, int]] = []
    s: int = raw[0]
    p: int = raw[0]
    for idx in raw[1:]:
        if idx - p <= MERGE_GAP:
            p = idx
        else:
            merged.append((s, p + STATIC_WINDOW))
            s = idx
            p = idx
    merged.append((s, p + STATIC_WINDOW))
    return merged


def to_box(b: list[int]) -> Box:
    """把 [x1, y1, x2, y2] 检测框列表转为 geom.Box。

    Args:
        b: [x1, y1, x2, y2] 边界框。

    Returns:
        对应的 Box。

    Raises:
        ValueError: 退化框（x2 <= x1 或 y2 <= y1），由 Box 构造校验抛出。
    """
    return Box(b[0], b[1], b[2], b[3])


def _safe_box(b: list[int]) -> Box | None:
    """to_box 的容错版：退化框（YOLO 坐标取整噪声）返回 None 而非抛错。

    退化框面积≈0，对持球判定无意义，跳过即可——模型输出噪声属正常
    输入波动，不是 rules.md §0.2 的"数据损坏"，不应中断整批检测。
    """
    try:
        return to_box(b)
    except ValueError:
        return None


def collect_candidates(
    tracks: list[Track],
    all_persons: list[list[list[int]]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    """从所有轨迹中提取候选入网点（静止段判据）。

    Args:
        tracks: MOT 产生的轨迹列表。
        all_persons: 每帧的人物框列表。

    Returns:
        (候选列表, 死球排除数, 持球排除数, 无静止段轨迹数)。
    """
    cands: list[dict[str, Any]] = []
    rm_dead: int = 0
    rm_held: int = 0
    rm_no_static: int = 0

    for track in tracks:
        merged: list[tuple[int, int]] = find_static_segments_in_track(track)
        if not merged:
            rm_no_static += 1
            continue

        for ms, me in merged:
            me = min(me, len(track.dets))
            seg: list[Detection] = track.dets[ms:me]
            if len(seg) < STATIC_WINDOW:
                continue
            t0: float = seg[0].sec
            dur: float = round(seg[-1].sec - seg[0].sec, 1)
            if dur > DEAD_BALL_SEC:
                rm_dead += 1
                continue
            cx: int = sum(d.cx for d in seg) // len(seg)
            cy: int = sum(d.cy for d in seg) // len(seg)
            ac: float = round(sum(d.conf for d in seg) / len(seg), 2)

            held: bool = False
            for d in seg:
                ball_box: Box | None = _safe_box(d.box)
                if ball_box is None:
                    continue
                for pb in all_persons[d.frame_idx]:
                    person_box: Box | None = _safe_box(pb)
                    if person_box is None:
                        continue
                    if coverage(ball_box, person_box) > HELD_COVERAGE:
                        held = True
                        break
                if held:
                    break
            if held:
                rm_held += 1
            else:
                cands.append({"t0": t0, "dur": dur, "ac": ac, "cx": cx, "cy": cy})

    return cands, rm_dead, rm_held, rm_no_static


def find_rejoin_candidates(
    all_balls: list[list[Detection]],
) -> list[dict[str, Any]]:
    """断轨重连签名：找"消失→重现"检测对作为入网候选（遮挡兜底）。

    签名（用 0011@10s / 0128@11s 两个已知漏检标定）：
    - 前驱：a 前 REJOIN_PRE_SEC 秒内有距 a <= REJOIN_PRE_DIST 的检测
      （排除孤立单帧假阳性作为起点）；
    - 消失→重现：存在 b 与 a 间隔 [REJOIN_MIN_GAP_SEC, REJOIN_MAX_GAP_SEC]
      且位移 <= REJOIN_MAX_DIST；
    - 盲区：a、b 之间无任何检测落在 a 的 REJOIN_VOID_DIST 半径内；
    - 持续：b 之后 REJOIN_FOLLOW_SEC 秒内有检测落在 b 的
      REJOIN_FOLLOW_DIST 半径内（真球重现后持续可见，孤立 FP 没有）；
    - anchor = a、b 时刻中点（0011 实测中点恰为 GT=10.0s）。

    每个 a 只取最早满足的 b 产生一个候选。

    Args:
        all_balls: 每帧的球检测列表。

    Returns:
        候选列表，每项含 t0/dur/ac/cx/cy/src。
    """
    dets: list[Detection] = [d for frame in all_balls for d in frame]
    dets.sort(key=lambda d: d.sec)
    secs: list[float] = [d.sec for d in dets]

    cands: list[dict[str, Any]] = []
    for i, a in enumerate(dets):
        pre_lo: int = bisect_left(secs, a.sec - REJOIN_PRE_SEC - 1e-9)
        has_pre: bool = any(
            euclidean((a.cx, a.cy), (p.cx, p.cy)) <= REJOIN_PRE_DIST for p in dets[pre_lo:i]
        )
        if not has_pre:
            continue

        b_lo: int = bisect_left(secs, a.sec + REJOIN_MIN_GAP_SEC - 1e-9)
        b_hi: int = bisect_left(secs, a.sec + REJOIN_MAX_GAP_SEC + 1e-9)
        for j in range(b_lo, b_hi):
            b: Detection = dets[j]
            gap: float = round(b.sec - a.sec, 1)
            if b.conf < REJOIN_MIN_CONF:
                continue
            if euclidean((a.cx, a.cy), (b.cx, b.cy)) > REJOIN_MAX_DIST:
                continue
            void: bool = not any(
                euclidean((a.cx, a.cy), (m.cx, m.cy)) < REJOIN_VOID_DIST for m in dets[i + 1 : j]
            )
            if not void:
                continue
            f_hi: int = bisect_left(secs, b.sec + REJOIN_FOLLOW_SEC + 1e-9)
            follow: bool = any(
                euclidean((b.cx, b.cy), (c.cx, c.cy)) <= REJOIN_FOLLOW_DIST
                for c in dets[j + 1 : f_hi]
            )
            if not follow:
                continue
            cands.append(
                {
                    "t0": round((a.sec + b.sec) / 2, 1),
                    "dur": gap,
                    "ac": b.conf,
                    "cx": b.cx,
                    "cy": b.cy,
                    "src": "rejoin",
                }
            )
            break
    return cands


def merge_candidates(
    static_cands: list[dict[str, Any]],
    rejoin_cands: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """合并静止段候选与重连候选，时空去重（静止段优先）。

    Args:
        static_cands: 静止段候选列表。
        rejoin_cands: 重连候选列表。

    Returns:
        (按 t0 排序的合并候选列表, 被去重丢弃的重连候选数)。
    """
    merged: list[dict[str, Any]] = []
    for c in static_cands:
        c["src"] = "static"
        merged.append(c)
    n_dup: int = 0
    for c in rejoin_cands:
        dup: bool = any(
            abs(c["t0"] - m["t0"]) <= DEDUPE_SEC
            and euclidean((c["cx"], c["cy"]), (m["cx"], m["cy"])) <= DEDUPE_DIST
            for m in merged
        )
        if dup:
            n_dup += 1
        else:
            merged.append(c)
    merged.sort(key=lambda c: c["t0"])
    return merged, n_dup


def dump_gt_window(
    all_balls: list[list[Detection]],
    tracks: list[Track],
    gt: float,
) -> None:
    """打印 GT 前后窗口内的原始检测与轨迹起止，诊断断轨原因。

    Args:
        all_balls: 每帧的球检测列表。
        tracks: MOT 产生的轨迹列表。
        gt: 进球真值时刻（秒）。
    """
    win_lo: float = gt - 3.0
    win_hi: float = gt + 4.0
    logger.info("  [诊断] GT窗口 %.1f-%.1fs 原始检测:", win_lo, win_hi)
    for frame_dets in all_balls:
        for d in frame_dets:
            if win_lo <= d.sec <= win_hi:
                logger.info(
                    "    raw t=%.1fs conf=%.2f @(%d,%d)",
                    d.sec,
                    d.conf,
                    d.cx,
                    d.cy,
                )
    logger.info("  [诊断] 与窗口重叠的轨迹:")
    for i, track in enumerate(tracks):
        t_end: float = track.last_det.sec
        if track.start_sec <= win_hi and t_end >= win_lo:
            logger.info(
                "    trk%d len=%d %.1f-%.1fs 首(%d,%d) 末(%d,%d)",
                i,
                track.length,
                track.start_sec,
                t_end,
                track.dets[0].cx,
                track.dets[0].cy,
                track.last_det.cx,
                track.last_det.cy,
            )


def run_pipeline(
    ball_model: YOLO,
    person_model: YOLO,
    fid: str,
) -> None:
    """在单个文件上跑 MOT pipeline 并输出结果。

    Args:
        ball_model: 球检测 YOLO 模型。
        person_model: 人物检测 YOLO 模型。
        fid: 文件 ID。
    """
    frames = sorted(glob(FRAMES_PATTERN.format(fid)))
    if not frames:
        logger.warning("%s: 无帧", fid)
        return

    logger.info("\n=== %s (%d帧) ===", fid, len(frames))

    cached = load_detection_cache(fid, len(frames))
    if cached is not None:
        all_balls, all_persons = cached
        logger.info(
            "  检测: 命中缓存 avg%.1f球/帧 共%d检测",
            round(sum(len(b) for b in all_balls) / len(frames), 1),
            sum(len(b) for b in all_balls),
        )
    else:
        t0: float = time.time()
        all_balls = []
        all_persons = []
        all_hoops: list[list[dict[str, Any]]] = []
        for i, fp in enumerate(frames):
            balls, persons, hoops = detect_frame(ball_model, person_model, fp, i)
            all_balls.append(balls)
            all_persons.append(persons)
            all_hoops.append(hoops)
            if (i + 1) % PROGRESS_LOG_EVERY == 0:
                logger.info("  检测进度: %s 第%d/%d帧", fid, i + 1, len(frames))
        elapsed: float = time.time() - t0
        total_dets: int = sum(len(b) for b in all_balls)
        avg_balls: float = round(total_dets / len(frames), 1)
        logger.info(
            "  检测%.1fs avg%.1f球/帧 共%d检测",
            elapsed,
            avg_balls,
            total_dets,
        )
        save_detection_cache(fid, all_balls, all_persons, all_hoops)

    tracks: list[Track] = run_mot(all_balls)
    long_tracks: list[Track] = [t for t in tracks if t.length >= STATIC_WINDOW]
    logger.info(
        "  MOT: %d条轨迹(>=4帧%d条)",
        len(tracks),
        len(long_tracks),
    )

    static_cands, rm_dead, rm_held, rm_no_static = collect_candidates(tracks, all_persons)
    rejoin_cands: list[dict[str, Any]] = find_rejoin_candidates(all_balls)
    cands, n_dup = merge_candidates(static_cands, rejoin_cands)

    gt: float | None = GROUND_TRUTH.get(fid)
    logger.info(
        "  排除: 无静止%d 死球%d 持球%d | 静止段候选%d 重连候选%d(去重%d) => 候选%d",
        rm_no_static,
        rm_dead,
        rm_held,
        len(static_cands),
        len(rejoin_cands),
        n_dup,
        len(cands),
    )

    hits: int = 0
    strict_hits: int = 0
    for c in cands:
        is_hit: str = ""
        if gt is not None and abs(c["t0"] - gt) <= GT_TOLERANCE:
            hits += 1
            is_hit = " <== HIT"
            if abs(c["t0"] - gt) <= GT_STRICT_TOLERANCE:
                strict_hits += 1
                is_hit = " <== HIT(严)"
        logger.info(
            "    t=%.1fs dur=%.1fs conf=%.2f @(%d,%d) [%s]%s",
            c["t0"],
            c["dur"],
            c["ac"],
            c["cx"],
            c["cy"],
            c.get("src", "static"),
            is_hit,
        )

    if gt is not None:
        dump_gt_window(all_balls, tracks, gt)
        if strict_hits > 0:
            status: str = "HIT"
        elif hits > 0:
            status = "WEAK(仅宽容差命中)"
        else:
            status = "MISS"
        logger.info(
            "  GT=%.1fs 召回=%s 候选%d 精确=%.0f%%",
            gt,
            status,
            len(cands),
            round(strict_hits / max(len(cands), 1) * 100),
        )


def main() -> None:
    """主入口：配置日志、加载模型并对各文件执行 MOT pipeline。

    逐文件循环捕获 KeyboardInterrupt：已完成文件的检测缓存已逐文件落盘，
    记录进度后以退出码 130 退出，重跑即可从缓存续做（rules.md §4）。
    """
    run_id: str = new_run_id()
    configure_logging(run_id)
    fids: list[str] = (
        sys.argv[1:] if len(sys.argv) > 1 else ["0011", "0020", "0030", "0040", "0128"]
    )
    ball_model = YOLO(BALL_MODEL_PATH)
    person_model = YOLO(PERSON_MODEL_PATH)

    for i, fid in enumerate(fids):
        try:
            run_pipeline(ball_model, person_model, fid)
        except KeyboardInterrupt:
            logger.warning("中断，已完成 %d/%d 个文件（检测缓存已逐文件落盘）", i, len(fids))
            sys.exit(130)

    logger.info("\n完成。")


if __name__ == "__main__":
    main()
