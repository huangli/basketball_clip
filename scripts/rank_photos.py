"""场次精彩照片自动挑选：mot_cache 打分 → 分桶选 top N → ffmpeg 抽帧防抖 → 构图裁切。

输入：work/<场次>/session_facts.json（逐文件宽高/帧率/时长）、
    work/<场次>/video_cli.json（srcdir 定位原片）、
    work/<场次>/hoops_batchN.json（筐事件：fid/window/anchor 筐心，检测尺度；缺失记空不报错）、
    work/<场次>/goals_batchN.json（confirmed 进球：file/anchor_time；缺失记空不报错）、
    work/detect/<视频名去后缀>_mot_cache.json（5fps 球/人检测缓存，经
    crop_scorers.load_mot_cache 唯一入口读取，坐标系 1920 宽检测尺度）。
输出：work/<场次>/photos/candidates/cNNN.jpg（裁切后 1920×1080 或 1440×1080，q95）、
    work/<场次>/photos/photo_candidates.json（每张：源视频/时刻/裁框/分数/状态）；
    --apply 时读 selections JSON 把确认照片复制到 output/<场次>/照片精选/。
依赖：scripts/pipe_common.py（run_ffmpeg/read_json/atomic_write_json/run_id 日志）、
    scripts/errors.py、scripts/geom.py（Box）、scripts/crop_scorers.py（load_mot_cache）、
    OpenCV + NumPy + PIL。
典型调用：
    python scripts/rank_photos.py --session 20260801_对手名
    python scripts/rank_photos.py --session 20260801_对手名 --force   # 清空候选全量重跑
    python scripts/rank_photos.py --session 20260801_对手名 --apply            # 默认路径
    python scripts/rank_photos.py --session 20260801_对手名 --apply 某路径.json

打分信号（spec: docs/photo-select/spec.md，第二轮调参 2026-08-15）：球筐距（hoops
事件窗口内球心到筐心距离，近=高分，主信号）、主体尺度、球速、人球交互、球高度。
confirmed 进球 anchor_time ±0.6s 内帧分数 ×1.5 且每球窗口内最高分帧 force_pick，
在分桶保底之外额外保底入选（进球瞬间必进候选）。每帧只取 max conf 球，
conf < 0.35 视为无球；无人帧无法构图不成候选。
时间分桶 10s/桶，每桶保底 1 张再按分数全局补齐（桶多于目标张数时以保底为准，
宁多勿漏）；尺度换算带可执行断言（spec §风险：越界抛 SchemaError 停跑）。
断点续跑：photo_candidates.json 已存在且视频清单未变 → 跳过抽帧直接复用；
--force 清空 candidates 与 candidates JSON 全量重跑（裁切参数变更后适用）。
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from crop_scorers import MotCache, load_mot_cache
from errors import BasketballPipelineError, SchemaError
from geom import Box
from mot_candidates import Detection
from pipe_common import (
    atomic_write_json,
    configure_logging,
    new_run_id,
    read_json,
    run_ffmpeg,
)

logger = logging.getLogger(__name__)

# ---- 采样与缓存契约（与 extract_frames/mot_cache 全线约定一致） ----
SAMPLE_FPS: int = 5  # 缓存抽帧帧率：帧 i 对应 sec = i/5
DETECT_WIDTH: int = 1920  # mot_cache 坐标系宽度（检测尺度）
COORD_TOLERANCE: float = 0.01  # 尺度断言容差（spec §风险：1%）

# ---- 打分信号参数（检测尺度像素；权重合计 1.0，第二轮调参 2026-08-15） ----
BALL_CONF_MIN: float = 0.35  # 球置信度下限，低于视为无球（spec §风险：缓存含 0.18~0.34 假球）
HOOP_DIST_FULL_PX: float = 400.0  # 球筐距满值（球心到筐心 400px 以上视为无筐信号）
SPEED_FULL_PX: float = 300.0  # 0.2s 球心位移满值（≈全场冲刺速度）
INTERACT_FULL_PX: float = 400.0  # 人球交互距离满值（框外 400px 以上视为无交互）
SCALE_FULL_RATIO: float = 0.12  # 主体面积占帧比满值（全身近景 ≈0.15）
W_HOOP: float = 0.35  # 球筐距（主信号：覆盖进球/篮板/封盖，都发生在筐附近）
W_SCALE: float = 0.20  # 主体尺度（上调，特写导向）
W_SPEED: float = 0.20
W_INTERACT: float = 0.15
W_HEIGHT: float = 0.10

# ---- 进球锚点加成（goals_batchN.json confirmed 球） ----
GOAL_BOOST_WINDOW_SEC: float = 0.6  # anchor_time ±0.6s 内帧加成（5fps 下 ±3 采样点）
GOAL_BOOST_FACTOR: float = 1.5  # 窗口内分数倍率
GOAL_TIME_EPS: float = 1e-6  # 窗口边界浮点容差（含端点）

# ---- 分桶参数 ----
BUCKET_SEC: float = 10.0  # 时间分桶宽度（spec：每桶保底 1 张保证时间覆盖）
DEFAULT_TOTAL: int = 200  # 候选目标张数（保底优先，可略超）

# ---- 抽帧防抖参数 ----
FRAME_RADIUS: int = 3  # 候选时刻 ±3 原帧（59.94fps 下 ±~50ms，远小于 5fps 采样间隔）
# 裁切区清晰度下限，全组低于此值整组丢弃（spec §风险兜底网）。
# 标定（2026-08-15 某次 4K HEVC 夜场实测）：正常帧裁切区方差 32~122，
# 全帧最低 21.8；重度运动模糊/黑场帧比方差再低数倍 → 15 只拦无望组。
MIN_LAPLACIAN_VAR: float = 15.0
EXTRACT_TIMEOUT_SEC: int = 120  # 单帧 ffmpeg 抽取超时（rules.md §4 兜底）
TMP_QV: int = 2  # 中间全帧 jpg 质量（防抖选材用，不省体积）
EOF_MARGIN_FRAMES: float = 1.5  # 片尾余量（帧）：抽取时刻上限 = duration − 1.5/fps

# ---- 构图裁切参数（第二轮特写化 2026-08-15：替换保守外扩） ----
OUT_16_9: tuple[int, int] = (1920, 1080)  # 16:9 素材输出尺寸
OUT_4_3: tuple[int, int] = (1440, 1080)  # 4:3 素材输出尺寸
RATIO_TOLERANCE: float = 0.01  # 比例判定容差（与 video.py resolve_out_size 一致）
HEADROOM_RATIO: float = 0.05  # 头顶留白下限（占裁框高度比例，10%→5% 特写化）
SUBJECT_FRAC_TARGET: float = 0.65  # 主体（人+球联合框）占裁框高度目标（spec 55~75% 中点）
SUBJECT_FRAC_OVER: float = 0.85  # 主体占比超此值视为过近，降排序分
SUBJECT_PAD_W: float = 1.1  # 主体宽度方向余量系数
NEAR_BALL_FACTOR: float = 0.75  # 球心距人框 ≤ 0.75×人高 视为"球在附近"联合包围
UPSCALE_WARN: float = 1.5  # 放大超此倍数降分并记日志（spec §3）
UPSCALE_PENALTY: float = 0.5  # 降分系数（惩罚后参与排序）
JPEG_QUALITY: int = 95  # 候选图保存质量

# ---- 产物契约 ----
CANDIDATES_VERSION: int = 2  # photo_candidates.json 版本号（v2：特写裁切+进球加成字段）
SELECTIONS_NAME: str = "photo_selections.json"  # 页面导出物的约定文件名


@dataclass(frozen=True, slots=True)
class FileFact:
    """session_facts.json 中单文件的归一元数据。"""

    name: str
    width: int
    height: int
    fps: float
    duration: float


@dataclass(frozen=True, slots=True)
class HoopEvent:
    """一个筐事件（hoops_batchN.json 归一后）：窗口（片内秒）+ 筐心（检测尺度）。"""

    start: float
    end: float
    hx: float
    hy: float


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """打过分的候选帧；person/ball 为原图像素坐标（构图输入，测试构造时可缺省）。

    force_pick=True 表示 confirmed 进球锚点 ±0.6s 窗口内最高分帧，
    在分桶保底之外额外保底入选（spec 第二轮：进球瞬间必进候选）。
    """

    fid: str
    frame_idx: int
    sec: float  # 片内秒
    global_sec: float  # 场次时间轴秒（分桶用）
    score: float
    person: Box | None = None
    ball: Box | None = None
    force_pick: bool = False


@dataclass(frozen=True, slots=True)
class CropPlan:
    """构图裁切方案：原图坐标裁框 + 输出放大倍数。

    不变式：box 宽高比 = 输出比例（±取整误差），box 夹回画面内，
    头顶与横向主体完整在框内（竖向允许切脚/切膝，spec 第二轮特写化）；
    upscale = 输出宽 / 裁框宽（<1 时记 1.0，即不放大只缩小）。
    """

    box: Box
    upscale: float
    penalized: bool  # upscale > UPSCALE_WARN → 排序降分


def frame_ball(balls: tuple[Detection, ...]) -> Detection | None:
    """取本帧 max conf 球；conf < BALL_CONF_MIN 或无球返回 None（spec §风险口径）。

    Args:
        balls: 本帧全部球检测（缓存原始候选，含低置信度假球）。

    Returns:
        置信度最高且达标的球检测；否则 None。
    """
    if not balls:
        return None
    best: Detection = max(balls, key=lambda d: d.conf)
    return best if best.conf >= BALL_CONF_MIN else None


def _point_box_dist(x: float, y: float, b: Box) -> float:
    """点到框的最短欧氏距离（点在框内为 0）。"""
    dx: float = max(b.x1 - x, 0.0, x - b.x2)
    dy: float = max(b.y1 - y, 0.0, y - b.y2)
    return math.hypot(dx, dy)


def _hoop_score(ball: Detection, sec: float, hoops: list[HoopEvent]) -> float:
    """球筐距信号：窗口内球心到筐心距离越近分越高；窗口外/无球记 0。

    多个事件窗口覆盖同一帧时取最高分（距最近的筐）。

    Args:
        ball: 本帧 max conf 球（检测尺度球心）。
        sec: 帧时刻（片内秒，5fps 采样）。
        hoops: 该文件的筐事件清单（可为空 → 信号恒 0）。

    Returns:
        [0,1] 的筐距信号分。
    """
    best: float = 0.0
    for ev in hoops:
        if not (ev.start <= sec <= ev.end):
            continue
        d: float = math.hypot(ball.cx - ev.hx, ball.cy - ev.hy)
        best = max(best, max(0.0, 1.0 - d / HOOP_DIST_FULL_PX))
    return best


def score_frames(cache: MotCache, det_h: int, hoops: list[HoopEvent] | None = None) -> list[float]:
    """对缓存逐帧计算冲击分（检测尺度；五信号加权，各项缺失记 0）。

    信号：球筐距（hoops 窗口内球心到筐心距离，主信号；hoops 缺失/窗口外记 0）、
    主体尺度（最大人框面积占比）、球速（相邻帧球心位移，中间断帧则该帧球速记 0）、
    人球交互（球心到最近人框距离，框内为满分）、球高度（cy 越小分越高）。
    无人帧只得尺度分 0 以外的分项——无人即 0 分。

    Args:
        cache: 校验后的 mot_cache。
        det_h: 检测尺度帧高（16:9→1080，4:3→1440）。
        hoops: 该文件的筐事件（hoops_batchN.json 归一）；None/空则筐距信号全 0。

    Returns:
        长度为 cache.frames 的逐帧分数列表。
    """
    frame_area: int = DETECT_WIDTH * det_h
    hoop_events: list[HoopEvent] = hoops or []
    scores: list[float] = []
    prev_ball: Detection | None = None
    for i in range(cache.frames):
        ball: Detection | None = frame_ball(cache.balls[i])
        persons: tuple[Box, ...] = cache.persons[i]
        hoop: float = 0.0
        speed: float = 0.0
        interact: float = 0.0
        height: float = 0.0
        if ball is not None:
            if hoop_events:
                hoop = _hoop_score(ball, i / SAMPLE_FPS, hoop_events)
            if prev_ball is not None:
                d: float = math.hypot(ball.cx - prev_ball.cx, ball.cy - prev_ball.cy)
                speed = min(d / SPEED_FULL_PX, 1.0)
            if persons:
                d_min: float = min(_point_box_dist(ball.cx, ball.cy, b) for b in persons)
                interact = max(0.0, 1.0 - d_min / INTERACT_FULL_PX)
            height = min(max(1.0 - ball.cy / det_h, 0.0), 1.0)
        scale: float = 0.0
        if persons:
            max_area: int = max(b.area for b in persons)
            scale = min(max_area / (SCALE_FULL_RATIO * frame_area), 1.0)
        scores.append(
            W_HOOP * hoop
            + W_SCALE * scale
            + W_SPEED * speed
            + W_INTERACT * interact
            + W_HEIGHT * height
        )
        prev_ball = ball
    return scores


def apply_goal_boost(
    items: list[ScoredCandidate],
    anchors: list[float],
    window: float = GOAL_BOOST_WINDOW_SEC,
    factor: float = GOAL_BOOST_FACTOR,
) -> list[ScoredCandidate]:
    """confirmed 进球加成：anchor ±window 内帧分数 ×factor，且每球窗口内最高分帧
    标记 force_pick（分桶保底之外额外保底入选，spec 第二轮：进球瞬间必进候选）。

    窗口边界含端点（浮点容差 GOAL_TIME_EPS）；无锚点或无候选原样返回；
    锚点窗口内无任何候选帧时该球自然跳过（素材缺失容忍）。

    Args:
        items: 单文件候选帧（score 已含放大/过近惩罚）。
        anchors: 该文件 confirmed 进球 anchor_time 清单（片内秒）。
        window: 加成半窗（秒）。
        factor: 窗口内分数倍率。

    Returns:
        新候选列表（输入不被修改；加成帧为 replace 副本）。
    """
    if not anchors or not items:
        return items
    tol: float = window + GOAL_TIME_EPS

    def _in_window(sec: float, anchor: float) -> bool:
        return abs(sec - anchor) <= tol

    boosted: list[ScoredCandidate] = [
        replace(it, score=it.score * factor) if any(_in_window(it.sec, a) for a in anchors) else it
        for it in items
    ]
    forced_idx: set[int] = set()
    for a in anchors:
        best_idx: int | None = None
        for idx, it in enumerate(boosted):
            if _in_window(it.sec, a) and (best_idx is None or it.score > boosted[best_idx].score):
                best_idx = idx
        if best_idx is not None:
            forced_idx.add(best_idx)
    return [replace(it, force_pick=True) if i in forced_idx else it for i, it in enumerate(boosted)]


def bucket_pick(
    items: list[ScoredCandidate],
    total: int = DEFAULT_TOTAL,
    bucket_sec: float = BUCKET_SEC,
) -> list[ScoredCandidate]:
    """时间分桶挑选：force_pick 帧额外保底 → 每桶保底最高分 1 张 → 按分数全局补齐。

    force_pick（confirmed 进球锚点窗口帧）在分桶保底之外必入选，且不再参与
    桶内保底与补齐（防重复）；保底优先于 total（桶多/保底多于 total 时宁可
    超出也不丢覆盖，spec「宁多勿漏」）；无候选的桶自然跳过。结果按 global_sec 升序。

    Args:
        items: 全部候选帧（分数已含放大/过近惩罚与进球加成）。
        total: 目标张数。
        bucket_sec: 分桶宽度（秒，按 global_sec 计）。

    Returns:
        入选候选，按场次时间升序。
    """
    forced: list[ScoredCandidate] = [it for it in items if it.force_pick]
    pool: list[ScoredCandidate] = [it for it in items if not it.force_pick]
    buckets: dict[int, list[ScoredCandidate]] = {}
    for it in pool:
        buckets.setdefault(int(it.global_sec // bucket_sec), []).append(it)
    picked: list[ScoredCandidate] = list(forced)
    rest: list[ScoredCandidate] = []
    for b in sorted(buckets):
        ranked: list[ScoredCandidate] = sorted(
            buckets[b], key=lambda it: (-it.score, it.global_sec, it.fid, it.frame_idx)
        )
        picked.append(ranked[0])
        rest.extend(ranked[1:])
    remaining: int = total - len(picked)
    if remaining > 0:
        rest.sort(key=lambda it: (-it.score, it.global_sec, it.fid, it.frame_idx))
        picked.extend(rest[:remaining])
    picked.sort(key=lambda it: (it.global_sec, it.fid, it.frame_idx))
    return picked


def assert_cache_scale(cache: MotCache, src_w: int, src_h: int, name: str) -> float:
    """校验缓存坐标尺度并返回换算因子（spec §风险的可执行断言）。

    缓存坐标必须落在检测帧 [0,1920]×[0,det_h] 内（det_h = src_h/factor），
    换算后必须落在原图 [0,src_w]×[0,src_h] 内，双侧均给 1% 容差；
    任一坐标越界即抛 SchemaError 停跑（尺度错=数据损坏，不静默容错）。

    Args:
        cache: 校验后的 mot_cache。
        src_w / src_h: 原图宽高（session_facts 实测值）。
        name: 源文件名（错误信息用）。

    Returns:
        换算因子 src_w/DETECT_WIDTH（缓存坐标 × 因子 = 原图坐标）。

    Raises:
        SchemaError: 任一缓存坐标越界。
    """
    factor: float = src_w / DETECT_WIDTH
    det_h: float = src_h / factor
    tol_x: float = DETECT_WIDTH * COORD_TOLERANCE
    tol_y: float = det_h * COORD_TOLERANCE

    def _check(x1: float, y1: float, x2: float, y2: float, what: str) -> None:
        for v, bound, tol, axis in ((x1, DETECT_WIDTH, tol_x, "x"), (x2, DETECT_WIDTH, tol_x, "x")):
            if v < -tol or v > bound + tol:
                raise SchemaError(f"{name}: {what} 缓存 {axis}={v} 越界 [0,{bound}]±1%（尺度错）")
        for v, bound, tol in ((y1, det_h, tol_y), (y2, det_h, tol_y)):
            if v < -tol or v > bound + tol:
                raise SchemaError(f"{name}: {what} 缓存 y={v} 越界 [0,{bound:.0f}]±1%（尺度错）")
        sx1, sy1, sx2, sy2 = (x1 * factor, y1 * factor, x2 * factor, y2 * factor)
        if sx1 < -tol_x * factor or sx2 > src_w + tol_x * factor:
            raise SchemaError(f"{name}: {what} 换算后 x 越界 [0,{src_w}]±1%（尺度错）")
        if sy1 < -tol_y * factor or sy2 > src_h + tol_y * factor:
            raise SchemaError(f"{name}: {what} 换算后 y 越界 [0,{src_h}]±1%（尺度错）")

    for frame_balls in cache.balls:
        for det in frame_balls:
            _check(det.box[0], det.box[1], det.box[2], det.box[3], "球框")
    for frame_persons in cache.persons:
        for b in frame_persons:
            _check(b.x1, b.y1, b.x2, b.y2, "人框")
    return factor


def compose_crop(
    person: Box,
    ball: Box | None,
    img_w: int,
    img_h: int,
    out_w: int,
    out_h: int,
) -> CropPlan:
    """以最大人框为主体计算特写构图裁框（spec 第二轮：替换保守外扩）。

    球在附近（球心距人框 ≤ 0.75×人高）则人球联合包围；裁框高度 = 主体高 /
    0.65（主体目标占裁框高 55~75%），头顶留白 ≥5%，人置于三分法竖线附近
    （偏左半场上左线，反之右线）；允许切脚/切膝——竖向只锚定头顶，不为
    保住主体底部而外扩或重夹；裁框平移夹回画面内（比画面大则按比例收缩）；
    裁框宽不足输出宽时记 upscale，>1.5 倍 penalized；主体占比 >85%（过近）
    同样 penalized（降分+日志）。

    Args:
        person: 主体人框（原图像素坐标）。
        ball: 球框（原图坐标，可空）。
        img_w / img_h: 原图宽高。
        out_w / out_h: 输出尺寸（比例即裁切比例）。

    Returns:
        CropPlan（裁框已夹回画面，头与横向主体完整在框内，脚部可出框）。
    """
    subject: Box = person
    if ball is not None:
        person_h: int = person.y2 - person.y1
        if _point_box_dist(ball.cx, ball.cy, person) <= NEAR_BALL_FACTOR * person_h:
            subject = Box(
                min(person.x1, ball.x1),
                min(person.y1, ball.y1),
                max(person.x2, ball.x2),
                max(person.y2, ball.y2),
            )
    sub_w: int = subject.x2 - subject.x1
    sub_h: int = subject.y2 - subject.y1
    # 特写化：高度取主体目标占比反推与宽度反推的较大者，保证两向都装得下且比例精确
    crop_h: int = math.ceil(max(sub_h / SUBJECT_FRAC_TARGET, sub_w * SUBJECT_PAD_W * out_h / out_w))
    crop_w: int = round(crop_h * out_w / out_h)
    # 头顶留白 ≥5%：裁框顶 = 主体顶 − 5% 裁框高（竖向锚定头顶，底部不做保护）
    y1: int = subject.y1 - round(HEADROOM_RATIO * crop_h)
    # 三分法：人中心放左/右三分线（按人在画面左右半选择，保证构图朝向留白）
    if person.cx <= img_w // 2:
        x1: int = person.cx - crop_w // 3
    else:
        x1 = person.cx - 2 * crop_w // 3
    # 比画面大则按比例收缩（仍锚定头顶，允许底部切脚）
    if crop_w > img_w or crop_h > img_h:
        shrink: float = min(img_w / crop_w, img_h / crop_h)
        crop_w = round(crop_w * shrink)
        crop_h = round(crop_h * shrink)
        x1 = subject.cx - crop_w // 2
        y1 = subject.y1 - round(HEADROOM_RATIO * crop_h)
    # 平移夹回画面（不缩放，保住比例；夹取只增头顶留白，不切头）
    x1 = max(0, min(img_w - crop_w, x1))
    y1 = max(0, min(img_h - crop_h, y1))
    # 横向主体若出框（极端贴边），向主体居中重夹一次；竖向不重夹（允许切脚/切膝）
    if x1 > subject.x1 or x1 + crop_w < subject.x2:
        x1 = max(0, min(img_w - crop_w, subject.cx - crop_w // 2))
    box: Box = Box(x1, y1, x1 + crop_w, y1 + crop_h)
    upscale: float = max(1.0, out_w / crop_w)
    too_close: bool = sub_h / crop_h > SUBJECT_FRAC_OVER
    return CropPlan(box=box, upscale=upscale, penalized=upscale > UPSCALE_WARN or too_close)


def validate_selections(data: Any, session: str) -> list[str]:  # noqa: ANN401 JSON 入参
    """校验页面导出的 selections JSON（rules.md §0.2：损坏显式失败）。

    Args:
        data: read_json 解析产物。
        session: 当前场次 ID（防止拿错场次的文件落盘）。

    Returns:
        确认候选的 id 列表（页面数组序）。

    Raises:
        SchemaError: 顶层非对象 / session 缺失或不匹配 / selected 缺失或非字符串列表。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"selections 顶层必须是对象，实际 {type(data).__name__}")
    got: Any = data.get("session")
    if not isinstance(got, str) or got != session:
        raise SchemaError(f"selections session 不匹配：期望 {session}，实际 {got!r}")
    selected: Any = data.get("selected")
    if not isinstance(selected, list) or any(not isinstance(s, str) for s in selected):
        raise SchemaError("selections selected 必须是字符串列表")
    return list(selected)


def apply_filename(seq: int, vid: int, sec: float) -> str:
    """落盘命名：照片_XXX_视频序号_时刻.jpg（seq/vid 1 起始补零）。"""
    return f"照片_{seq:03d}_v{vid:03d}_t{sec:06.1f}.jpg"


def load_session_facts(path: Path) -> list[FileFact]:
    """读取并校验 session_facts.json，返回按文件名升序（=拍摄时间序）的清单。

    Raises:
        SchemaError: 顶层结构损坏 / 条目缺字段或类型错 / files 为空。
    """
    data: Any = read_json(path, what="session_facts.json")
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise SchemaError(f"{path}: 顶层必须是含 files 对象的事实表")
    files: dict[str, Any] = data["files"]
    if not files:
        raise SchemaError(f"{path}: files 为空，无法挑选照片")
    facts: list[FileFact] = []
    for name in sorted(files):
        info: Any = files[name]
        if not isinstance(info, dict):
            raise SchemaError(f"{path}: {name} 的元数据不是对象")
        try:
            fact = FileFact(
                name=name,
                width=int(info["width"]),
                height=int(info["height"]),
                fps=float(info["fps"]),
                duration=float(info["duration"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{path}: {name} 字段缺失/类型错: {exc}") from exc
        if fact.width <= 0 or fact.height <= 0 or fact.fps <= 0 or fact.duration <= 0:
            raise SchemaError(f"{path}: {name} 元数据非法: {fact}")
        facts.append(fact)
    return facts


def load_hoop_events(session_dir: Path) -> dict[str, list[HoopEvent]]:
    """读取场次目录全部 hoops_batchN.json，归一为 fid（去后缀）→ 筐事件清单。

    文件缺失记空 dict 不报错（老场次无 hoops，筐距信号全 0）；文件存在但
    schema 损坏（缺 events / 事件缺 fid/window/anchor 或类型错）抛 SchemaError
    （rules.md §0.2：数据损坏必须停）。

    Args:
        session_dir: work/<场次> 目录。

    Returns:
        fid（视频名去后缀，与 mot_cache 主键一致）→ 筐事件列表（可多批次合并）。

    Raises:
        SchemaError: 任一批次文件 schema 损坏。
    """
    out: dict[str, list[HoopEvent]] = {}
    for path in sorted(session_dir.glob("hoops_batch*.json")):
        data: Any = read_json(path, what="hoops_batch")
        if not isinstance(data, dict) or not isinstance(data.get("events"), list):
            raise SchemaError(f"{path}: 顶层必须是含 events 列表的对象")
        for idx, ev in enumerate(data["events"]):
            if not isinstance(ev, dict):
                raise SchemaError(f"{path}: events[{idx}] 不是对象")
            fid: Any = ev.get("fid")
            window: Any = ev.get("window")
            anchor: Any = ev.get("anchor")
            if (
                not isinstance(fid, str)
                or not isinstance(window, list)
                or len(window) != 2
                or not all(isinstance(v, (int, float)) for v in window)
                or not isinstance(anchor, list)
                or len(anchor) != 2
                or not all(isinstance(v, (int, float)) for v in anchor)
            ):
                raise SchemaError(f"{path}: events[{idx}] 缺 fid/window/anchor 或类型错")
            out.setdefault(fid, []).append(
                HoopEvent(
                    start=float(window[0]),
                    end=float(window[1]),
                    hx=float(anchor[0]),
                    hy=float(anchor[1]),
                )
            )
    return out


def load_goal_anchors(session_dir: Path) -> dict[str, list[float]]:
    """读取场次目录全部 goals_batchN.json，归一为 file（含后缀）→ confirmed 锚点清单。

    只收 status == "confirmed" 的球；文件缺失记空 dict 不报错（未标注场次无加成）；
    文件存在但 schema 损坏抛 SchemaError（rules.md §0.2）。

    Args:
        session_dir: work/<场次> 目录。

    Returns:
        file（视频名含后缀，与 session_facts 主键一致）→ anchor_time（片内秒）列表。

    Raises:
        SchemaError: 任一批次文件 schema 损坏。
    """
    out: dict[str, list[float]] = {}
    for path in sorted(session_dir.glob("goals_batch*.json")):
        data: Any = read_json(path, what="goals_batch")
        if not isinstance(data, dict) or not isinstance(data.get("goals"), list):
            raise SchemaError(f"{path}: 顶层必须是含 goals 列表的对象")
        for idx, g in enumerate(data["goals"]):
            if not isinstance(g, dict):
                raise SchemaError(f"{path}: goals[{idx}] 不是对象")
            file: Any = g.get("file")
            anchor_time: Any = g.get("anchor_time")
            status: Any = g.get("status")
            if (
                not isinstance(file, str)
                or not isinstance(anchor_time, (int, float))
                or not isinstance(status, str)
            ):
                raise SchemaError(f"{path}: goals[{idx}] 缺 file/anchor_time/status 或类型错")
            if status == "confirmed":
                out.setdefault(file, []).append(float(anchor_time))
    return out


def reset_candidate_outputs(photos_dir: Path) -> None:
    """--force：清空 candidates 目录与 photo_candidates.json（裁切参数变更后
    断点续跑不适用，必须全量重出）。目录/文件不存在时幂等不报错。

    Args:
        photos_dir: work/<场次>/photos 目录。
    """
    shutil.rmtree(photos_dir / "candidates", ignore_errors=True)
    (photos_dir / "photo_candidates.json").unlink(missing_ok=True)


def out_size_for(fact: FileFact) -> tuple[int, int]:
    """按素材比例判定输出尺寸（容差 ±1%，与 video.py resolve_out_size 同口径）。

    Raises:
        BasketballPipelineError: 未知比例（照片不做猜比例，显式失败）。
    """
    ratio: float = fact.width / fact.height
    if abs(ratio - 16 / 9) / (16 / 9) <= RATIO_TOLERANCE:
        return OUT_16_9
    if abs(ratio - 4 / 3) / (4 / 3) <= RATIO_TOLERANCE:
        return OUT_4_3
    raise BasketballPipelineError(
        f"{fact.name}: 未知素材比例 {fact.width}x{fact.height}（{ratio:.4f}），不猜比例"
    )


def scale_box(b: Box, factor: float, img_w: int, img_h: int) -> Box:
    """检测尺度框 × 因子换算到原图坐标并夹回画面。"""
    x1: int = max(0, min(img_w - 2, round(b.x1 * factor)))
    y1: int = max(0, min(img_h - 2, round(b.y1 * factor)))
    x2: int = max(x1 + 1, min(img_w, round(b.x2 * factor)))
    y2: int = max(y1 + 1, min(img_h, round(b.y2 * factor)))
    return Box(x1, y1, x2, y2)


def build_scored_candidates(
    fact: FileFact,
    cache: MotCache,
    global_offset: float,
    hoops: list[HoopEvent] | None = None,
    goal_anchors: list[float] | None = None,
) -> list[ScoredCandidate]:
    """单文件逐帧打分并装配候选（无人帧跳过：无主体无法构图）。

    构图在打分阶段完成：放大/过近惩罚先于分桶生效，模糊/远景帧自然沉底；
    末尾套进球锚点加成（±0.6s ×1.5 + force_pick 保底）。

    Args:
        fact: 文件元数据。
        cache: 该文件的 mot_cache（已 schema 校验）。
        global_offset: 该文件在场次时间轴上的起始秒（前序文件时长累加）。
        hoops: 该文件的筐事件（可空 → 筐距信号全 0）。
        goal_anchors: 该文件 confirmed 进球 anchor_time 清单（可空 → 无加成）。

    Returns:
        候选列表（分桶挑选的输入）。
    """
    factor: float = assert_cache_scale(cache, fact.width, fact.height, fact.name)
    det_h: int = round(fact.height / factor)
    scores: list[float] = score_frames(cache, det_h, hoops)
    out_w, out_h = out_size_for(fact)
    items: list[ScoredCandidate] = []
    for i in range(cache.frames):
        persons: tuple[Box, ...] = cache.persons[i]
        if not persons:
            continue
        person_det: Box = max(persons, key=lambda b: b.area)
        person: Box = scale_box(person_det, factor, fact.width, fact.height)
        ball_det: Detection | None = frame_ball(cache.balls[i])
        ball: Box | None = None
        if ball_det is not None:
            ball = scale_box(Box(*ball_det.box), factor, fact.width, fact.height)
        plan: CropPlan = compose_crop(person, ball, fact.width, fact.height, out_w, out_h)
        score: float = scores[i]
        if plan.penalized:
            score *= UPSCALE_PENALTY
            logger.debug(
                "%s 帧%d 放大 %.2f 倍/过近，降分 %.3f→%.3f",
                fact.name,
                i,
                plan.upscale,
                scores[i],
                score,
            )
        sec: float = i / SAMPLE_FPS
        items.append(
            ScoredCandidate(
                fid=fact.name,
                frame_idx=i,
                sec=sec,
                global_sec=global_offset + sec,
                score=score,
                person=person,
                ball=ball,
            )
        )
    return apply_goal_boost(items, goal_anchors or [])


def window_times(sec: float, fps: float, duration: float) -> list[float]:
    """候选时刻 ±FRAME_RADIUS 原帧的抽取时刻清单（片头/片尾夹取）。

    上限 = duration − EOF_MARGIN_FRAMES/fps：seek 到片尾之后 ffmpeg 取不到帧
    会报错（实测 dji 短片 sec≈duration 时 mjpeg 编码器初始化失败）。

    Args:
        sec: 候选时刻（片内秒）。
        fps: 原片帧率。
        duration: 原片时长（秒）。

    Returns:
        2×FRAME_RADIUS+1 个抽取时刻（升序）。
    """
    t_max: float = max(0.0, duration - EOF_MARGIN_FRAMES / fps)
    return [min(max(0.0, sec + k / fps), t_max) for k in range(-FRAME_RADIUS, FRAME_RADIUS + 1)]


def extract_window_frames(
    video: Path, sec: float, fps: float, duration: float, tmp_dir: Path, tag: str
) -> list[Path]:
    """候选时刻 ±3 原帧各抽 1 张全帧 jpg（ffmpeg -ss 前置精确 seek + -frames:v 1）。

    单帧失败（如片尾边界）记 WARNING 跳过该帧，不中断整组（7 帧容忍少量缺失）。

    Args:
        video: 原片路径。
        sec: 候选时刻（片内秒）。
        fps: 原片帧率（帧步长 = 1/fps）。
        duration: 原片时长（秒，片尾夹取用）。
        tmp_dir: 中间帧目录。
        tag: 文件名前缀（候选 id）。

    Returns:
        成功抽出的中间帧路径（≤7 张，按时刻升序）。
    """
    paths: list[Path] = []
    for i, t in enumerate(window_times(sec, fps, duration)):
        out: Path = tmp_dir / f"{tag}_{i}.jpg"
        try:
            run_ffmpeg(
                [
                    "-ss",
                    f"{t:.4f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    str(TMP_QV),
                    str(out),
                ],
                timeout_sec=EXTRACT_TIMEOUT_SEC,
            )
        except BasketballPipelineError as exc:
            logger.warning("%s t=%.2fs 单帧抽取失败（跳过该帧）: %s", video.name, t, exc)
            continue
        paths.append(out)
    return paths


def _read_image(path: Path) -> np.ndarray | None:
    """读图（cv2.imdecode 路径，Windows 中文路径下 cv2.imread 会静默失败）。"""
    try:
        buf: np.ndarray = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        logger.warning("读图失败 %s: %s", path, exc)
        return None
    img: np.ndarray | None = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("解码失败 %s", path)
    return img


def sharpest_frame(paths: list[Path], crop: Box) -> tuple[Path, float] | None:
    """在裁切区域上算 Laplacian 方差，返回最清晰帧；全组模糊返回 None。

    Args:
        paths: 候选时刻 ±3 帧的全帧 jpg。
        crop: 构图裁框（清晰度只评裁切区域，框外模糊不影响主体）。

    Returns:
        (最清晰帧路径, 方差)；全组方差低于 MIN_LAPLACIAN_VAR 或全部读图失败 → None。
    """
    best_path: Path | None = None
    best_var: float = 0.0
    for p in paths:
        img: np.ndarray | None = _read_image(p)
        if img is None:
            continue
        roi: np.ndarray = img[crop.y1 : crop.y2, crop.x1 : crop.x2]
        if roi.size == 0:
            continue
        gray: np.ndarray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        var: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if var > best_var:
            best_var = var
            best_path = p
    if best_path is None or best_var < MIN_LAPLACIAN_VAR:
        return None
    return best_path, best_var


def save_crop(frame_path: Path, plan: CropPlan, out_w: int, out_h: int, dest: Path) -> None:
    """按裁框裁切并缩放到输出尺寸，保存 JPEG q95（PIL 写盘，兼容中文路径）。

    Raises:
        BasketballPipelineError: 帧读不出（前面 sharpest 已读过，理论上不会到这）。
    """
    img: np.ndarray | None = _read_image(frame_path)
    if img is None:
        raise BasketballPipelineError(f"抽帧产物读不出: {frame_path}")
    b: Box = plan.box
    roi: np.ndarray = img[b.y1 : b.y2, b.x1 : b.x2]
    pil: Image.Image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if pil.size != (out_w, out_h):
        pil = pil.resize((out_w, out_h), Image.LANCZOS)
    pil.save(dest, "JPEG", quality=JPEG_QUALITY)


def _resolve_srcdir(session_dir: Path, rawdir: str | None) -> Path:
    """解析原片目录：显式 --rawdir 优先，其次 video_cli.json srcdir，都没有显式失败。"""
    if rawdir:
        return Path(rawdir)
    state_path: Path = session_dir / "video_cli.json"
    if state_path.is_file():
        state: Any = read_json(state_path, what="video_cli.json")
        if isinstance(state, dict) and isinstance(state.get("srcdir"), str) and state["srcdir"]:
            return Path(state["srcdir"])
    raise BasketballPipelineError("--rawdir 未给且 video_cli.json 无 srcdir（先跑 score）")


def _load_candidates_json(path: Path) -> dict[str, Any]:
    """读 photo_candidates.json 并做顶层校验（条目级校验归页面/apply 各自口径）。"""
    data: Any = read_json(path, what="photo_candidates.json")
    if not isinstance(data, dict) or data.get("version") != CANDIDATES_VERSION:
        raise SchemaError(f"{path}: 顶层必须是 version={CANDIDATES_VERSION} 的对象")
    if not isinstance(data.get("candidates"), list) or not isinstance(data.get("videos"), list):
        raise SchemaError(f"{path}: 缺 candidates/videos 列表")
    return data


def _run_rank(session: str, rawdir: str | None, total: int, force: bool) -> int:
    """主流程：读缓存打分 → 分桶选 top N → 抽帧防抖裁切 → 落 candidates JSON。"""
    session_dir: Path = Path("work") / session
    if not session_dir.is_dir():
        raise BasketballPipelineError(f"场次目录不存在: {session_dir}（先跑 score）")
    facts: list[FileFact] = load_session_facts(session_dir / "session_facts.json")
    srcdir: Path = _resolve_srcdir(session_dir, rawdir)
    photos_dir: Path = session_dir / "photos"
    cand_dir: Path = photos_dir / "candidates"
    cand_json: Path = photos_dir / "photo_candidates.json"

    if force:
        reset_candidate_outputs(photos_dir)
        logger.info("--force：已清空 %s 与 candidates JSON，全量重跑", cand_dir)

    hoop_map: dict[str, list[HoopEvent]] = load_hoop_events(session_dir)
    goal_map: dict[str, list[float]] = load_goal_anchors(session_dir)
    logger.info(
        "筐事件 %d 个文件 / confirmed 进球 %d 个文件",
        len(hoop_map),
        len(goal_map),
    )

    video_names: list[str] = [f.name for f in facts]
    if cand_json.is_file():
        try:
            old: dict[str, Any] = _load_candidates_json(cand_json)
        except SchemaError:
            logger.info("旧 candidates 版本不兼容，全量重算（旧产物作废）")
        else:
            if old.get("videos") == video_names:
                logger.info(
                    "断点续跑：candidates 已存在且视频清单未变（%d 张），跳过抽帧",
                    len(old["candidates"]),
                )
                return 0
            logger.info("视频清单已变，全量重算（旧 candidates 作废）")

    # 逐文件：读缓存 → 尺度断言 → 打分装配候选
    items: list[ScoredCandidate] = []
    global_offset: float = 0.0
    video_no: dict[str, int] = {}
    facts_by_name: dict[str, FileFact] = {}
    skipped: int = 0
    for no, fact in enumerate(facts, start=1):
        video_no[fact.name] = no
        facts_by_name[fact.name] = fact
        cache_path: Path = Path("work/detect") / f"{Path(fact.name).stem}_mot_cache.json"
        video_path: Path = srcdir / fact.name
        if not cache_path.is_file():
            logger.warning("缺 mot_cache，跳过: %s", cache_path)
            skipped += 1
            global_offset += fact.duration
            continue
        if not video_path.is_file():
            logger.warning("原片不存在，跳过: %s", video_path)
            skipped += 1
            global_offset += fact.duration
            continue
        cache: MotCache = load_mot_cache(cache_path)
        items.extend(
            build_scored_candidates(
                fact,
                cache,
                global_offset,
                hoops=hoop_map.get(Path(fact.name).stem),
                goal_anchors=goal_map.get(fact.name),
            )
        )
        global_offset += fact.duration
    if skipped:
        logger.warning("共 %d 个文件被跳过（缺缓存或缺原片）", skipped)
    logger.info("候选帧池 %d（%d 个文件）", len(items), len(facts) - skipped)

    picked: list[ScoredCandidate] = bucket_pick(items, total=total, bucket_sec=BUCKET_SEC)
    n_forced: int = sum(1 for c in picked if c.force_pick)
    logger.info(
        "分桶挑选 %d 张（目标 %d，保底优先；进球锚点保底 %d 张）", len(picked), total, n_forced
    )

    # 抽帧防抖 + 裁切落盘
    cand_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir: Path = photos_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_entries: list[dict[str, Any]] = []
    dropped: int = 0
    try:
        for seq, cand in enumerate(picked, start=1):
            cid: str = f"c{seq:03d}"
            fact = facts_by_name[cand.fid]
            out_w, out_h = out_size_for(fact)
            if cand.person is None:  # build_scored_candidates 保证非空，防御性检查
                raise BasketballPipelineError(f"{cand.fid} 帧{cand.frame_idx} 缺主体框（契约破坏）")
            plan: CropPlan = compose_crop(
                cand.person, cand.ball, fact.width, fact.height, out_w, out_h
            )
            frames: list[Path] = extract_window_frames(
                srcdir / cand.fid, cand.sec, fact.fps, fact.duration, tmp_dir, cid
            )
            try:
                best: tuple[Path, float] | None = sharpest_frame(frames, plan.box)
                if best is None:
                    dropped += 1
                    logger.warning(
                        "%s t=%.1fs 无可用帧（抽出 %d 帧，清晰度均低于 %.0f 或抽取失败），整组丢弃",
                        cand.fid,
                        cand.sec,
                        len(frames),
                        MIN_LAPLACIAN_VAR,
                    )
                    continue
                dest: Path = cand_dir / f"{cid}.jpg"
                save_crop(best[0], plan, out_w, out_h, dest)
            finally:
                for p in frames:
                    p.unlink(missing_ok=True)
            out_entries.append(
                {
                    "id": cid,
                    "src_file": cand.fid,
                    "video_no": video_no[cand.fid],
                    "sec": round(cand.sec, 2),
                    "global_sec": round(cand.global_sec, 2),
                    "score": round(cand.score, 4),
                    "upscale": round(plan.upscale, 3),
                    "penalized": plan.penalized,
                    "crop": [plan.box.x1, plan.box.y1, plan.box.x2, plan.box.y2],
                    "sharpness": round(best[1], 1),
                    "goal_boost": cand.force_pick,
                    "image": f"candidates/{cid}.jpg",
                    "status": "ok",
                }
            )
            if seq % 10 == 0 or seq == len(picked):
                logger.info("抽帧进度 %d/%d（丢弃 %d）", seq, len(picked), dropped)
    except KeyboardInterrupt:
        logger.error(
            "中断：已产 %d/%d 张（candidates JSON 未写，重跑全量重来）",
            len(out_entries),
            len(picked),
        )
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    payload: dict[str, Any] = {
        "version": CANDIDATES_VERSION,
        "session": session,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_frames_scored": len(items),
        "dropped_blurry": dropped,
        "videos": video_names,
        "candidates": out_entries,
    }
    atomic_write_json(cand_json, payload, what="photo_candidates.json")
    logger.info(
        "candidates 落盘: %s（%d 张，模糊丢弃 %d）→ 下一步生成确认页",
        cand_json,
        len(out_entries),
        dropped,
    )
    return 0


def _run_apply(session: str, selections_path: Path) -> int:
    """--apply：selections schema 校验后复制确认照片到 output/<场次>/照片精选/。

    产出型脚本口径（rules.md §0.2）：未知 id / 缺图逐条 ERROR 跳过，其余照出，
    有错则退出码非零。
    """
    session_dir: Path = Path("work") / session
    cand_json: Path = session_dir / "photos" / "photo_candidates.json"
    if not cand_json.is_file():
        raise BasketballPipelineError(f"缺 candidates: {cand_json}（先跑 rank）")
    data: Any = read_json(selections_path, what="selections")
    ids: list[str] = validate_selections(data, session)
    payload: dict[str, Any] = _load_candidates_json(cand_json)
    by_id: dict[str, dict[str, Any]] = {}
    for raw in payload["candidates"]:
        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
            by_id[raw["id"]] = raw
    out_dir: Path = Path("output") / session / "照片精选"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: int = 0
    entries: list[dict[str, Any]] = []
    for cid in ids:
        entry: dict[str, Any] | None = by_id.get(cid)
        if entry is None:
            logger.error("selections 引用未知候选 id: %s（跳过）", cid)
            errors += 1
            continue
        entries.append(entry)
    # 落盘顺序按拍摄时间（视频序号 → 片内时刻），与合集口径一致
    entries.sort(key=lambda e: (int(e["video_no"]), float(e["sec"])))
    copied: int = 0
    for seq, entry in enumerate(entries, start=1):
        src: Path = session_dir / "photos" / str(entry["image"])
        if not src.is_file():
            logger.error("候选图缺失: %s（跳过）", src)
            errors += 1
            continue
        dest: Path = out_dir / apply_filename(seq, int(entry["video_no"]), float(entry["sec"]))
        shutil.copy2(src, dest)
        copied += 1
    logger.info("落盘 %d/%d 张 → %s", copied, len(ids), out_dir)
    return 1 if errors else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """命令行参数解析。"""
    ap = argparse.ArgumentParser(
        prog="rank_photos",
        description="场次精彩照片挑选：打分排序 → 抽帧防抖裁切 → candidates JSON / --apply 落盘",
    )
    ap.add_argument("--session", required=True, help="场次 ID（work/<场次>/）")
    ap.add_argument("--rawdir", default=None, help="原片目录（缺省读 video_cli.json srcdir）")
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="候选目标张数（保底优先）")
    ap.add_argument(
        "--force",
        action="store_true",
        help="清空 candidates 目录与 candidates JSON 全量重跑（裁切参数变更后适用）",
    )
    ap.add_argument(
        "--apply",
        nargs="?",
        const="",
        default=None,
        help=f"落盘模式：selections 路径（缺省 work/<场次>/photos/{SELECTIONS_NAME}）",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；非 0=失败/部分失败）。"""
    args: argparse.Namespace = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        if args.apply is not None:
            sel: Path = (
                Path(args.apply)
                if args.apply
                else Path("work") / args.session / "photos" / SELECTIONS_NAME
            )
            return _run_apply(args.session, sel)
        if args.total < 1:
            raise BasketballPipelineError(f"--total 必须 ≥1，实际 {args.total}")
        return _run_rank(args.session, args.rawdir, args.total, args.force)
    except BasketballPipelineError as exc:
        logger.error("失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    # 管道/重定向时 stdout 回落 locale 编码，打印中文日志会 UnicodeEncodeError
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure") and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
