"""投篮者定位裁图 + 颜色分队（spec: docs/scorer/spec.md §投篮者定位算法 / §颜色分队判据）。

输入：goals.json（status=confirmed 记录）、work/detect/<fid>_mot_cache.json（5fps
    球/人检测缓存，坐标系 1920×1080 与帧图一致）、work/frames/<fid>/f_NNNNN.jpg；
    --rawdir（可选，原片目录）：给了就为每个 confirmed 球现切认人预览片段
    （窗口 [max(0, anchor−4), anchor+2]，与进球锚点严格对齐——认人页视频必须与
    裁图同球同时刻，不再引用 events_index 的事件片段，长事件开头是另一回合）。
输出：<out>/ 下每个 confirmed 球一张投篮者裁图 + scorer_candidates.json
    （含 key=format_key、裁图路径、status OK/SKIP、team_guess、clip 预览片段
    相对路径；OK 球另含轨迹传播种子 seed_frame/seed_box/seed_team，
    scorer-propagate spec）；--rawdir 给定时另有 <out>/clips/<fid>_t<anchor:.1f>.mp4。
依赖：scripts/roster.py（format_key / fid_of）、scripts/geom.py（Box/iou）、
    scripts/pipe_common.py（read_json / atomic_write_json / run_ffmpeg / run_id 日志）、
    PIL + numpy。
典型调用：
    python scripts/crop_scorers.py --goals work/20260722/goals.json \
        --detectdir work/detect --framesdir work/frames --out work/20260722/scorers \
        --rawdir "素材目录"

定位算法（2026-08-08 轨迹法，替换逐帧投票——逐帧取 max-conf 球会在海报球/
隔壁场球之间瞬移，逐帧规则全军覆没；轨迹是可靠的，候选本来就是轨迹挖出来的）：
窗口 [anchor−4.0, anchor+0.5] 内用 mot_candidates.run_mot 重链球轨迹
（min_length=1，短轨迹也是有效证据）；端点与候选锚点（--candidates 给的
t0/cx/cy，goals 锚点与其 dt=0 匹配）最近的轨迹 = 进球轨迹；沿该轨迹从末端
往回放找最后一个"球心严格落在某人框内"的轨迹点 → 该人框 = 投篮者（最后持球者）；
整轨无持球点 → 取轨迹起点时刻的最近人框；轨迹不存在/端点离锚点太远 → SKIP。
SKIP 球无投篮者定位但仍切预览片段（用户凭视频手选）。
号码识别（--read-numbers，2026-08-09 升级多帧投票，scorer-reid spec §数据契约）：
对每张 crops 逐张调 K3 读背号（复用 vlm_filter 的 load_token/crop_to_b64/重试口径），
多帧众数投票压单帧误读；number_cache.json 键 = 裁图文件 md5（旧 goal key 自动迁移
重键、旧键保留一轮由人清理），幂等不重复扣额度；--numbers-cache-only 跳票模式
零新调用（旧数据重跑回填用）。
轨迹选帧多裁（--best-crops，默认 3，scorer-cluster spec §数据契约，2026-08-09）：
定位成功后以定位帧人框为种子，向前后逐帧 IoU≥0.3 链同一人框（窗口 = 定位帧前后
各 2s，5fps 即各 ≤10 帧，越界/链断即停）；链上帧逐帧读图算质量分（归一化框面积
× Laplacian 方差），取 top N 且入选帧间隔 ≥0.5s 去重；entry 落 crops（质量降序）
与 crop_scores，crop = crops[0] 保持向后兼容，rank≥2 文件名追加 _q2/_q3 后缀。
裁图质量闸（crop-quality spec §数据契约，2026-08-27）：选帧循环内逐帧过闸——
宽高比 sanity（外扩后框 w/h 越界 [GATE_MIN_RATIO, GATE_MAX_RATIO] 判畸形废帧，
零成本先行）→ 框内人物复检（yolov8n gate-view 复检无可信 person 判无人废帧，
惰性加载可注入）；废帧丢弃取次优帧补位、不写 JPEG、记 INFO 留痕（帧时刻+原因）；
候选帧全被拦 → status=SKIP + reason=quality_gate（预览片段保留，不落
crops/crop_scores）。缺框帧（检测缓存无框）跳过复检不拦截，记 INFO。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image

from errors import BasketballPipelineError, ExternalApiError, SchemaError
from geom import Box, iou
from mot_candidates import Detection, Track, euclidean, run_mot
from pipe_common import (
    atomic_write_json,
    configure_logging,
    new_run_id,
    read_json,
    run_ffmpeg,
)
from roster import fid_of, format_key
from vlm_filter import (
    API_URL as K3_API_URL,
)
from vlm_filter import (
    HTTP_RETRY as K3_HTTP_RETRY,
)
from vlm_filter import (
    HTTP_TIMEOUT_SEC as K3_HTTP_TIMEOUT_SEC,
)
from vlm_filter import (
    MODEL as K3_MODEL,
)
from vlm_filter import (
    crop_to_b64,
    load_token,
)

logger = logging.getLogger(__name__)

# ---- 轨迹法定位参数（2026-08-08，替换逐帧投票） ----
SAMPLE_FPS: int = 5  # 抽帧帧率（extract_frames 全线约定：帧 i 对应 sec = i/5）
TRACK_WINDOW_PRE_SEC: float = 4.0  # 轨迹重链窗口起点 = anchor − 4.0s
TRACK_WINDOW_POST_SEC: float = 0.5  # 轨迹重链窗口终点 = anchor + 0.5s
GOAL_TRACK_MAX_DIST_PX: int = 200  # 进球轨迹端点距候选锚点上界，超出 = 没链到 → SKIP
PREFER_LONGER_SLACK_PX: int = (
    100  # 长轨偏好滑窗：端点距 ≤ 最优+此值的轨迹进候选池（docs/heatmap-flight-link/）
)
PREFER_LONGER_SLACK_SEC: float = 1.0  # 长轨偏好时间滑窗（anchor_xy=None 分支，秒；同上文档）
CANDIDATE_MATCH_DT_SEC: float = 0.3  # goals 锚点与 candidates t0 的匹配容差
_EPS: float = 1e-9  # 浮点窗口边界的容差

# ---- 裁图参数（spec B2 第 4 条） ----
CROP_EXPAND: float = 0.2  # 人框外扩比例（每维放大到 1.2 倍，即每侧 10%）
CROP_MIN_SHORT_SIDE: int = 400  # 短边下限（像素），不足等比放大（号码识别可读性下限）
JPEG_QUALITY: int = 95  # 裁图保存质量（认人目检用，不省这点体积）

# ---- 轨迹选帧多裁参数（scorer-cluster spec §数据契约，2026-08-09） ----
TRACE_WINDOW_SEC: float = 2.0  # 选帧窗口 = 定位帧前后各 2s（5fps 即各 ≤10 帧），越界即停
TRACE_MIN_IOU: float = 0.3  # 帧间人框 IoU 下限，低于即视为链断停链
CROP_MIN_SPACING_SEC: float = 0.5  # 入选帧最小间隔（同人连续帧裁剪近乎重复，无信息增量）
DEFAULT_BEST_CROPS: int = 3  # --best-crops 默认值（每球最多裁图张数）

# ---- 裁图质量闸参数（crop-quality spec §数据契约，2026-08-27 定稿） ----
# 阈值由 某批次 58 球 643 候选帧全量分布标定，依据见
# work/crop_quality_calibration/report.md §阈值建议（宁漏拦不错杀，端点含入）。
GATE_MIN_RATIO: float = 0.15  # 外扩后框 w/h 下限：643 帧仅 t550.0 细条帧 0.0898 低于此，断层明显
GATE_MAX_RATIO: float = 1.2  # 外扩后框 w/h 上限：超限 10 帧全属 t44.2 前景手臂废帧（1.22~2.04）
GATE_PERSON_CONF: float = 0.25  # 框内复检 person 置信度下限：分布双峰（0.0 与 ≥0.5 之间空白）
GATE_PERSON_MODEL_PATH: str = "models/yolov8n.pt"  # 复检人物模型（持球排除同款，惰性加载）
GATE_PERSON_IMGSZ: int = 640  # 复检推理尺寸（同标定 gate-view 口径：外扩+短边放大后 imgsz=640）
YOLO_PERSON_CLS: int = 0  # COCO person 类 id（与 mot_candidates.PERSON_CLS 同值）
REASON_QUALITY_GATE: str = "quality_gate"  # 候选帧全被质量闸拦截的 SKIP reason


# ---- 认人预览片段参数（--rawdir 给定时逐球现切，与进球锚点严格对齐） ----
PREVIEW_BEFORE_SEC: float = 4.0  # 窗口 = 锚点前 4s（与剪辑规格一致）
PREVIEW_AFTER_SEC: float = 2.0  # 窗口 = 锚点后 2s
PREVIEW_WIDTH: int = 1280  # 预览宽度（高自适应保持偶数；够认人即可，省体积）
PREVIEW_CRF: int = 26  # 预览码率（认人用途，比成品 20 宽松）
PREVIEW_PRESET: str = "veryfast"  # 预览求快

# ---- 颜色分队阈值（spec §颜色分队判据 M4；HSV 各通道 0-255，PIL convert("HSV") 口径） ----
# 采样区 = 人框水平中 60% × 垂直 25%~60%（躯干，排除头/腿/背景边缘）。
# 标定记录（2026-08-08，批次 1 共 17 球：15 张 OK 裁图逐张目检真值 黑5/白10，2 张 SKIP 无图）：
#   黑队 frac(V<45)：真黑 0.31~0.72，真白 ≤0.19 → TH_BLACK=45、占比阈 0.25 双侧有间隔
#   白队 frac(V>170 且 S<70)：真白 0.14~0.66（取 ≥0.20 命中 8/10，余 2 张灯光暗+绿偏
#     归便服，spec 允许近阈归便服），真黑 ≤0.12 → TH_WHITE=170、TH_SAT=70、占比阈 0.20
TH_BLACK: int = 45  # 黑队：采样区 V < TH_BLACK 占比达标 → 黑
TH_WHITE: int = 170  # 白队：V > TH_WHITE 且 S < TH_SAT
TH_SAT: int = 70  # 白队的饱和度上限（彩色亮部/肤色不归白）
MIN_BLACK_FRACTION: float = 0.25  # 黑色像素占比下限，不足（含近阈混杂）归"便服"
MIN_WHITE_FRACTION: float = 0.20  # 白色像素占比下限，不足（含近阈混杂）归"便服"

# ---- 号码识别参数（--read-numbers，spec T7；K3 读裁图背号，走订阅额度无需 key） ----
NUMBER_PROMPT_VERSION: str = "number-v1"  # prompt 语义变更即升版本，旧缓存作废（幂等可追溯）
MAX_NUMBER_READS_PER_RUN: int = 20  # spec：单次运行新调用 >20 次须先问用户（缓存命中不计）
NUMBER_PROMPT: str = (
    "这是一张室内篮球场球员的照片裁图（投篮者）。请识别图中主要球员的："
    "1）球衣号码（阿拉伯数字字符串，看不清或没有给 null）；"
    "2）球衣颜色（黑/白/蓝/其他 四选一）；"
    "3）球衣背后印的名字文字（如有请照抄，没有或看不清给 null）。"
    '严格只输出一个 JSON 对象，不要输出任何其他内容：{"number": "21" 或 null, '
    '"color": "黑"|"白"|"蓝"|"其他", "name_text": "..." 或 null, '
    '"confidence": "high"|"low"}。'
    "背对镜头但模糊、正面无号码等看不清的情况：number 给 null、confidence 给 low。"
)

# ---- 号码缓存键（scorer-reid spec §数据契约：key = 裁图文件 md5，跨球/重跑复用） ----
MD5_CHUNK_BYTES: int = 1 << 20  # md5 分块读取大小（1MB，与 cluster_scorers 同口径）

STATUS_OK: str = "OK"
STATUS_SKIP: str = "SKIP"

TEAM_BLACK: str = "黑"
TEAM_WHITE: str = "白"
TEAM_CASUAL: str = "便服"


@dataclass(frozen=True, slots=True)
class MotCache:
    """校验后的 mot_cache：balls / persons 按帧对齐，长度均为 frames。

    balls 直接复用 mot_candidates.Detection（轨迹链接的输入类型），
    校验时从 mot_cache 原始字段构造。
    """

    frames: int
    balls: tuple[tuple[Detection, ...], ...]
    persons: tuple[tuple[Box, ...], ...]


@dataclass(frozen=True, slots=True)
class NumberGuess:
    """号码识别结果（K3 读裁图）；number/name_text 可为 None（看不清/没有）。"""

    number: str | None
    color: str | None
    name_text: str | None
    confidence: str  # "high" | "low"


def number_guess_from_dict(data: Any) -> NumberGuess | None:  # noqa: ANN401 JSON 待归一
    """把 dict（模型 JSON / 缓存条目）宽松归一为 NumberGuess；非 dict → None。

    归一规则：number 取数字字符串（int/float 转 str，非数字归 None）；
    color 只认 黑/白/蓝/其他；name_text 空串归 None；confidence 非 "high" 归 "low"。

    Args:
        data: 待归一的 dict。

    Returns:
        NumberGuess；data 非 dict 返回 None。
    """
    if not isinstance(data, dict):
        return None
    number: str | None = None
    number_raw: Any = data.get("number")
    if isinstance(number_raw, (int, float)) and not isinstance(number_raw, bool):
        number = str(int(number_raw))
    elif isinstance(number_raw, str):
        digits: str = number_raw.strip()
        number = digits if digits.isdigit() else None
    color_raw: Any = data.get("color")
    color: str | None = color_raw if color_raw in ("黑", "白", "蓝", "其他") else None
    name_raw: Any = data.get("name_text")
    name_text: str | None = name_raw.strip() if isinstance(name_raw, str) else None
    if not name_text:
        name_text = None
    confidence: str = "high" if data.get("confidence") == "high" else "low"
    return NumberGuess(number=number, color=color, name_text=name_text, confidence=confidence)


def file_md5(path: Path) -> str:
    """计算文件 md5（分块读取；仅作缓存键，非安全用途）。

    Args:
        path: 文件路径。

    Returns:
        32 位十六进制 md5。
    """
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(MD5_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


_MD5_KEY_RE: re.Pattern[str] = re.compile(r"[0-9a-f]{32}")


def _entry_crops(entry: dict[str, Any]) -> list[str]:
    """取 entry 的裁图文件名列表：优先 crops（多裁），旧数据无则回退单 crop。

    Args:
        entry: _process_goal 产出的候选记录（或旧 schema 记录）。

    Returns:
        非空裁图文件名列表（保持质量降序）；无裁图返回空列表。
    """
    crops: Any = entry.get("crops")
    if isinstance(crops, list):
        names: list[str] = [c for c in crops if isinstance(c, str) and c]
        if names:
            return names
    crop: Any = entry.get("crop")
    return [crop] if isinstance(crop, str) and crop else []


def vote_number_guess(guesses: list[NumberGuess]) -> NumberGuess | None:
    """多帧读号众数投票（scorer-reid spec §数据契约，规则写死勿自行变更）。

    number=None 的票（K3 看不清属合法返回）不参与计数，仅在有号码的票中投票：
    同号 ≥2 张 → 采纳该号；有效票 =1 → conf=high 采纳、conf=low 归 None+low；
    有效票 ≥2 且全不同 → 取唯一 conf=high 者（多个 high 不采）；其余（含有效票
    =0）→ number=None、confidence=low。

    Args:
        guesses: 逐张读号结果（仅含成功识别的票；识别失败/跳票/裁图缺失不在其中）。

    Returns:
        投票结果（采纳时返回该票原样，含 color/name_text）；guesses 为空返回 None
        （调用方把 number_guess 置空，区别于"有票但未采纳"的 None+low）。
    """
    if not guesses:
        return None
    valid: list[NumberGuess] = [g for g in guesses if g.number is not None]
    if not valid:
        return NumberGuess(number=None, color=None, name_text=None, confidence="low")
    counts: Counter[str] = Counter(g.number for g in valid if g.number is not None)
    top_number, top_count = counts.most_common(1)[0]  # 并列按首次出现序（稳定可测）
    if top_count >= 2:
        return next(g for g in valid if g.number == top_number)
    if len(valid) == 1:
        single: NumberGuess = valid[0]
        if single.confidence == "high":
            return single
        return NumberGuess(number=None, color=None, name_text=None, confidence="low")
    highs: list[NumberGuess] = [g for g in valid if g.confidence == "high"]
    if len(highs) == 1:
        return highs[0]
    return NumberGuess(number=None, color=None, name_text=None, confidence="low")


def migrate_number_cache(
    cache: dict[str, dict[str, Any]], entries: list[dict[str, Any]], outdir: Path
) -> tuple[dict[str, dict[str, Any]], bool]:
    """旧 goal key 号码缓存重键为 crops[0] 裁图 md5（scorer-reid spec §数据契约）。

    规则（写死）：旧 goal key → 当前 run entries 反查该球 crops[0] 文件算 md5 重键
    （零新 API 调用；迁移是重键不是删除——旧键保留一轮，实跑无误后由人清理）；
    旧 key 在当前 run entries 查不到（删球/子集重跑）→ 原样保留记 INFO；crops[0]
    文件缺失算不出 md5 → 记 WARNING 保留原 key，不炸整批；已是 md5 的键不动。
    幂等：二次执行零变化（changed=False）。

    Args:
        cache: load_number_cache 产物（不改入参，返回新 dict）。
        entries: 当前 run 候选记录（反查 goal key → crops[0] 用）。
        outdir: 输出目录（裁图所在）。

    Returns:
        (迁移后缓存, 是否有变化)。
    """
    by_key: dict[str, dict[str, Any]] = {
        str(e["key"]): e for e in entries if isinstance(e.get("key"), str)
    }
    migrated: dict[str, dict[str, Any]] = dict(cache)
    changed: bool = False
    for key, value in cache.items():
        if _MD5_KEY_RE.fullmatch(key):
            continue
        entry: dict[str, Any] | None = by_key.get(key)
        if entry is None:
            logger.info("号码缓存迁移: 旧键在当前 run 无对应进球，原样保留: %s", key)
            continue
        crops: list[str] = _entry_crops(entry)
        crop_path: Path | None = outdir / crops[0] if crops else None
        if crop_path is None or not crop_path.is_file():
            logger.warning("号码缓存迁移: crops[0] 裁图缺失，保留原键: %s (%s)", key, crop_path)
            continue
        md5_key: str = file_md5(crop_path)
        if md5_key not in migrated:
            migrated[md5_key] = value
            changed = True
            logger.info("号码缓存迁移: %s → md5:%s（旧键保留一轮）", key, md5_key)
    return migrated, changed


def parse_number_answer(raw: str) -> NumberGuess | None:
    """容错解析 K3 回复：提取首个 {...} JSON 块归一；坏 JSON/无 JSON → None。

    Args:
        raw: 模型回复全文（可能带 markdown 围栏或前后废话）。

    Returns:
        NumberGuess；解析失败返回 None（调用方记 ERR，不炸整批）。
    """
    m: re.Match[str] | None = re.search(r"\{.*\}", raw, re.DOTALL)
    if m is None:
        return None
    try:
        data: Any = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return number_guess_from_dict(data)


def load_number_cache(path: Path) -> dict[str, dict[str, Any]]:
    """读 number_cache.json（幂等缓存）；缺失/损坏/prompt 版本不符 → 空（重识别）。

    Args:
        path: <out>/number_cache.json 路径。

    Returns:
        key → 缓存条目（含 number/color/name_text/confidence/usage/model/ts）。
    """
    if not path.exists():
        return {}
    payload: Any = read_json(path, what="number_cache.json")
    if not isinstance(payload, dict):
        logger.warning("number_cache.json 结构异常，重新开始")
        return {}
    meta: Any = payload.get("_meta")
    if isinstance(meta, dict) and meta.get("prompt_version") != NUMBER_PROMPT_VERSION:
        logger.warning("号码识别 prompt 版本变更，旧缓存作废重开")
        return {}
    results: Any = payload.get("results")
    if not isinstance(results, dict):
        return {}
    return {str(k): v for k, v in results.items() if isinstance(v, dict)}


def save_number_cache(path: Path, results: dict[str, dict[str, Any]]) -> None:
    """原子写 number_cache.json（_meta 记录模型/prompt 版本/时间戳）。

    Args:
        path: <out>/number_cache.json 路径。
        results: key → 缓存条目。

    Raises:
        OSError: IO 重试耗尽（由 atomic_write_json 抛出）。
    """
    payload: dict[str, Any] = {
        "_meta": {
            "prompt_version": NUMBER_PROMPT_VERSION,
            "model": K3_MODEL,
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "results": results,
    }
    atomic_write_json(path, payload, what="number_cache.json")


# 单次号码识别注入点：(裁图路径, 进球键) → (guess, tokens, err)。测试注入假 reader
# （不碰网络/凭证）；生产默认 None → apply_number_reading 内部用 httpx + read_number。
NumberReader = Callable[[Path, str], "tuple[NumberGuess | None, int, str]"]


def read_number(
    client: httpx.Client,
    *,
    crop_path: Path,
    key: str,
) -> tuple[NumberGuess | None, int, str]:
    """对单张裁图调 K3 号码识别（重试口径同 vlm_filter.ask_vlm：网络错误重试、
    401 等待后强制重载 token 重试、400/403 不重试）。

    图片输入规格沿用 vlm_filter：crop_to_b64（IMG_SIZE=840 缩放 + base64 JPEG
    data URI）；token 走 vlm_filter.load_token（OAuth 900s 临期自动重读）。

    Args:
        client: httpx 客户端（trust_env=False，直连不走代理）。
        crop_path: 投篮者裁图路径。
        key: 进球键（日志用；缓存键是裁图 md5，由调用方管理）。

    Returns:
        (NumberGuess, total_tokens, 错误摘要)；成功 err=""；失败 guess=None
        不炸整批（调用方记日志继续，失败不写缓存、下次重跑重试）。
    """
    try:
        with Image.open(crop_path) as im:
            img_b64: str = crop_to_b64(im.convert("RGB"))
    except OSError as exc:
        return None, 0, f"读取裁图失败: {exc}"
    payload: dict[str, Any] = {
        "model": K3_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": NUMBER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
    }
    last_err: str = "未知错误"
    for _ in range(K3_HTTP_RETRY + 1):
        try:
            auth: dict[str, str] = {"Authorization": f"Bearer {load_token()}"}
            resp: httpx.Response = client.post(
                K3_API_URL, json=payload, headers=auth, timeout=K3_HTTP_TIMEOUT_SEC
            )
        except httpx.HTTPError as exc:
            last_err = f"网络错误: {type(exc).__name__}: {exc}"
            logger.warning("%s 号码识别网络错误，重试: %s", key, exc)
            continue
        if resp.status_code == 200:
            try:
                data: dict[str, Any] = resp.json()
            except json.JSONDecodeError as exc:
                return None, 0, f"HTTP 200 但响应非 JSON: {exc}"
            raw: Any = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            raw_text: str = raw if isinstance(raw, str) else str(raw)
            usage_raw: Any = data.get("usage")
            usage_dict: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
            try:
                tokens: int = int(usage_dict.get("total_tokens") or 0)
            except TypeError, ValueError:
                tokens = 0
            guess: NumberGuess | None = parse_number_answer(raw_text)
            if guess is None:
                return None, tokens, "回复无法解析为号码 JSON"
            return guess, tokens, ""
        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        if resp.status_code == 401:
            logger.warning("%s 401，等待后强制重载 token 重试", key)
            time.sleep(8)
            load_token(force=True)
            continue
        if resp.status_code in (400, 403):
            break  # 请求/权限问题，重试无意义
    return None, 0, last_err


def _vote_record(
    crop: str, md5: str | None, source: str, guess: NumberGuess | None
) -> dict[str, Any]:
    """组装 number_votes 逐张摘要（source: cache/fresh/skipped/missing/error）。"""
    return {
        "crop": crop,
        "md5": md5,
        "source": source,
        "number": guess.number if guess is not None else None,
        "confidence": guess.confidence if guess is not None else None,
    }


def apply_number_reading(
    entries: list[dict[str, Any]],
    outdir: Path,
    max_reads: int = MAX_NUMBER_READS_PER_RUN,
    *,
    cache_only: bool = False,
    reader: NumberReader | None = None,
) -> tuple[int, int]:
    """--read-numbers 主流程：逐张裁图读号 + 多帧众数投票（缓存命中不重复扣额度）。

    每张 crops（旧数据回退单 crop）各读一次号，vote_number_guess 众数投票后写入
    entry 的 number_guess（结构不变，消费方零改动）与 number_votes（逐张摘要，
    调试可追溯）。缓存键 = 裁图文件 md5；旧 goal key 缓存先迁移重键（零新调用）。
    单张失败记 ERROR 继续不炸整批（不写缓存，下次重跑重试）。

    两种模式（scorer-reid spec §数据契约，写死）：
    - 全量模式（默认）：缓存未命中的裁图发起新调用；新调用 >max_reads 拒绝执行
      （默认 20；spec：超额须先问用户，批准后 --max-reads 显式放宽）。
    - 跳票模式（cache_only，CLI --numbers-cache-only）：缓存未命中跳过不读、
      只用已有票投票，零新调用、不要求凭证（旧数据重跑回填用）。

    Args:
        entries: _process_goal 产出的候选记录（原地补 number_guess/number_votes）。
        outdir: 输出目录（裁图与缓存所在）。
        max_reads: 单次运行允许的最大新识别张数（全量模式闸）。
        cache_only: True = 跳票模式。
        reader: 单次读号注入点（测试用）；None = 生产默认（httpx + K3）。

    Returns:
        (本次新识别张数, 本次总 token 用量)。

    Raises:
        ExternalApiError: 凭证缺失 / 超 max_reads 张新调用（全量模式）。
        BasketballPipelineError: reader 契约破坏（err 为空但 guess=None）。
    """
    targets: list[dict[str, Any]] = [
        e for e in entries if e["status"] == STATUS_OK and _entry_crops(e)
    ]
    if not targets:
        return 0, 0
    cache_path: Path = outdir / "number_cache.json"
    cache: dict[str, dict[str, Any]] = load_number_cache(cache_path)
    cache, migrated_changed = migrate_number_cache(cache, entries, outdir)
    if migrated_changed:
        save_number_cache(cache_path, cache)  # 迁移持久化（跳票模式同样落盘）

    # 逐张预算缓存键：裁图缺失记 WARNING 跳过（不算票也不算新调用）
    plans: list[tuple[dict[str, Any], list[tuple[str, str | None]]]] = []
    for e in targets:
        items: list[tuple[str, str | None]] = []
        for name in _entry_crops(e):
            path: Path = outdir / name
            if not path.is_file():
                logger.warning("读号裁图缺失，跳过该张: %s (%s)", path, e["key"])
                items.append((name, None))
                continue
            items.append((name, file_md5(path)))
        plans.append((e, items))

    actual_reader: NumberReader | None = reader
    http_client: httpx.Client | None = None
    if not cache_only:
        planned: set[str] = {
            md5 for _e, items in plans for _n, md5 in items if md5 is not None and md5 not in cache
        }
        if len(planned) > max_reads:
            raise ExternalApiError(
                f"本轮需新识别 {len(planned)} 张（>{max_reads}），"
                "spec 规定须先问用户；确认后用 --max-reads 显式放宽（缓存幂等）"
            )
        if planned and actual_reader is None:
            try:
                load_token()  # 凭证预检：缺凭证尽早显式失败
            except RuntimeError as exc:
                raise ExternalApiError(f"K3 凭证不可用: {exc}") from exc
            http_client = httpx.Client(trust_env=False)  # trust_env=False：直连，不走代理

            def actual_reader(crop_path: Path, key: str) -> tuple[NumberGuess | None, int, str]:
                if http_client is None:  # 防御：构建后立即赋值，不应触发
                    raise BasketballPipelineError("读号客户端未构建（逻辑错误）")
                return read_number(http_client, crop_path=crop_path, key=key)

    n_fresh: int = 0
    total_tokens: int = 0
    try:
        for e, items in plans:
            guesses: list[NumberGuess] = []
            votes_summary: list[dict[str, Any]] = []
            for name, md5 in items:
                if md5 is None:
                    votes_summary.append(_vote_record(name, None, "missing", None))
                    continue
                cached: dict[str, Any] | None = cache.get(md5)
                if cached is not None:
                    guess = number_guess_from_dict(cached)
                    if guess is None:  # 防御：load_number_cache 已过滤非 dict，不应触发
                        raise BasketballPipelineError(f"号码缓存条目无法归一: md5:{md5}")
                    guesses.append(guess)
                    votes_summary.append(_vote_record(name, md5, "cache", guess))
                    continue
                if cache_only:
                    votes_summary.append(_vote_record(name, md5, "skipped", None))
                    continue
                if actual_reader is None:  # 防御：planned 非空必已构建/注入
                    raise BasketballPipelineError("读号器未构建（planned 与 reader 逻辑不一致）")
                guess, tokens, err = actual_reader(outdir / name, e["key"])
                n_fresh += 1
                total_tokens += tokens
                if err:
                    logger.error("号码识别失败（下次重跑重试）: %s %s: %s", e["key"], name, err)
                    votes_summary.append(_vote_record(name, md5, "error", None))
                    continue
                if guess is None:  # 防御：err 为空必有 guess（read_number 契约）
                    raise BasketballPipelineError(
                        f"号码识别契约破坏: {e['key']} {name} err 为空但 guess=None"
                    )
                cache[md5] = {
                    **asdict(guess),
                    "usage": {"total_tokens": tokens},
                    "model": K3_MODEL,
                    "ts": datetime.now(UTC).isoformat(),
                }
                save_number_cache(cache_path, cache)
                logger.info(
                    "号码识别: %s %s → number=%s color=%s name=%s conf=%s（%d tokens）",
                    e["key"],
                    name,
                    guess.number,
                    guess.color,
                    guess.name_text,
                    guess.confidence,
                    tokens,
                )
                guesses.append(guess)
                votes_summary.append(_vote_record(name, md5, "fresh", guess))
            voted: NumberGuess | None = vote_number_guess(guesses)
            e["number_guess"] = asdict(voted) if voted is not None else None
            e["number_votes"] = votes_summary
    finally:
        if http_client is not None:
            http_client.close()
    return n_fresh, total_tokens


@dataclass(frozen=True, slots=True)
class LocateResult:
    """定位结果；status=SKIP 时 frame_idx=-1、box=None。

    votes/total_votes 沿用旧字段名保持 JSON 兼容，语义改为：
    votes = 进球轨迹长度（检测数），total_votes = 窗口内轨迹总数。
    """

    status: str
    reason: str
    frame_idx: int
    box: Box | None
    votes: int
    total_votes: int


def load_mot_cache(path: str | Path) -> MotCache:
    """读取并校验 mot_cache（rules.md §0.2：schema 损坏显式失败，不静默容错）。

    Args:
        path: work/detect/<fid>_mot_cache.json 路径。

    Returns:
        校验后的 MotCache。

    Raises:
        SchemaError: 顶层缺 frames/balls/persons、长度不齐、检测字段类型错。
    """
    data: Any = read_json(path, what="mot_cache")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    frames: Any = data.get("frames")
    balls_raw: Any = data.get("balls")
    persons_raw: Any = data.get("persons")
    if not isinstance(frames, int) or frames < 0:
        raise SchemaError(f"{path}: frames 缺失或非负 int，实际 {frames!r}")
    if not isinstance(balls_raw, list) or not isinstance(persons_raw, list):
        raise SchemaError(f"{path}: balls/persons 缺失或不是列表")
    if len(balls_raw) != frames or len(persons_raw) != frames:
        raise SchemaError(
            f"{path}: 长度不齐 frames={frames} balls={len(balls_raw)} persons={len(persons_raw)}"
        )

    balls: list[tuple[Detection, ...]] = []
    for i, frame_balls in enumerate(balls_raw):
        if not isinstance(frame_balls, list):
            raise SchemaError(f"{path}: balls[{i}] 不是列表")
        dets: list[Detection] = []
        for j, raw in enumerate(frame_balls):
            if not isinstance(raw, dict):
                raise SchemaError(f"{path}: balls[{i}][{j}] 不是对象")
            box_raw: Any = raw.get("box")
            if not (isinstance(box_raw, list) and len(box_raw) == 4):
                raise SchemaError(f"{path}: balls[{i}][{j}] box 不是 [x1,y1,x2,y2]")
            try:
                dets.append(
                    Detection(
                        conf=float(raw["conf"]),
                        box=[int(v) for v in box_raw],
                        cx=int(raw["cx"]),
                        cy=int(raw["cy"]),
                        sec=float(raw["sec"]),
                        frame_idx=int(raw["frame_idx"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"{path}: balls[{i}][{j}] 字段缺失/类型错: {exc}") from exc
        balls.append(tuple(dets))

    persons: list[tuple[Box, ...]] = []
    for i, frame_persons in enumerate(persons_raw):
        if not isinstance(frame_persons, list):
            raise SchemaError(f"{path}: persons[{i}] 不是列表")
        boxes: list[Box] = []
        for j, raw in enumerate(frame_persons):
            if not (isinstance(raw, list) and len(raw) == 4):
                raise SchemaError(f"{path}: persons[{i}][{j}] 不是 [x1,y1,x2,y2]")
            try:
                boxes.append(Box(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])))
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"{path}: persons[{i}][{j}] 非法框: {exc}") from exc
        persons.append(tuple(boxes))

    return MotCache(frames=frames, balls=tuple(balls), persons=tuple(persons))


def load_candidates_index(path: str | Path) -> dict[str, list[tuple[float, int, int]]]:
    """读候选 JSON（candidates.json），建 fid → [(t0, cx, cy)] 索引。

    候选锚点 (cx, cy) 是轨迹法选轨迹的空间参照（goals.json 锚点与其 t0 对应，
    批次 1 全部 17 球 dt=0.0 匹配）。

    Args:
        path: candidates.json 路径（列表，每条 {t0, dur, ac, cx, cy, src, fid, label}）。

    Returns:
        fid → [(t0, cx, cy), ...]（同 fid 多候选按文件序）。

    Raises:
        SchemaError: 顶层非列表 / 条目缺 fid/t0/cx/cy 或类型错。
    """
    data: Any = read_json(path, what="candidates.json")
    if not isinstance(data, list):
        raise SchemaError(f"{path}: 顶层必须是列表，实际 {type(data).__name__}")
    index: dict[str, list[tuple[float, int, int]]] = {}
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise SchemaError(f"{path}: 第{i}条不是对象")
        try:
            fid = str(raw["fid"])
            t0 = float(raw["t0"])
            cx = int(raw["cx"])
            cy = int(raw["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{path}: 第{i}条字段缺失/类型错: {exc}") from exc
        index.setdefault(fid, []).append((t0, cx, cy))
    return index


def match_anchor_xy(
    index: dict[str, list[tuple[float, int, int]]], fid: str, anchor_sec: float
) -> tuple[int, int] | None:
    """按 fid + |t0−anchor|≤CANDIDATE_MATCH_DT_SEC 取最近候选的 (cx, cy)；无 → None。

    Args:
        index: load_candidates_index 产物。
        fid: 视频主名。
        anchor_sec: goals.json 的进球锚点（秒）。

    Returns:
        候选锚点 (cx, cy)；无匹配返回 None（退化为端点时间最近选轨迹）。
    """
    best: tuple[int, int] | None = None
    best_dt: float = CANDIDATE_MATCH_DT_SEC
    for t0, cx, cy in index.get(fid, []):
        dt: float = abs(t0 - anchor_sec)
        if dt <= best_dt + _EPS:
            best_dt = dt
            best = (cx, cy)
    return best


def track_window_dets(cache: MotCache, anchor_sec: float) -> list[list[Detection]]:
    """取窗口 [anchor−4.0, anchor+0.5] 内逐帧球检测（保持帧序，空帧为空列表）。

    Args:
        cache: 校验后的 mot_cache。
        anchor_sec: 进球锚点（秒）。

    Returns:
        逐帧检测列表（run_mot 的输入形状）。
    """
    lo: float = anchor_sec - TRACK_WINDOW_PRE_SEC
    hi: float = anchor_sec + TRACK_WINDOW_POST_SEC
    first: int = max(0, math.ceil((lo - _EPS) * SAMPLE_FPS))
    last: int = min(cache.frames - 1, math.floor((hi + _EPS) * SAMPLE_FPS))
    return [list(cache.balls[fi]) for fi in range(first, last + 1)]


def select_goal_track(
    tracks: list[Track],
    anchor_sec: float,
    anchor_xy: tuple[int, int] | None,
    *,
    prefer_longer: bool = False,
) -> Track | None:
    """选进球轨迹：端点（末端）与候选锚点最近的轨迹。

    有 anchor_xy（候选 cx/cy）时按端点空间距离最近，距离超 GOAL_TRACK_MAX_DIST_PX
    视为没链到（None → SKIP）；无 anchor_xy 退化为端点时间距 anchor 最近。
    prefer_longer=True（落点链路专用，docs/heatmap-flight-link/）：端点距
    ≤ 最优+PREFER_LONGER_SLACK_PX（anchor_xy=None 时为端点时刻距 ≤ 最优
    +PREFER_LONGER_SLACK_SEC）的轨迹组成候选池，池中取最长轨迹（长度并列
    取距锚更近者）——纯按距离/时刻会把选择吸向网内单点碎片，而网内点
    被 v4 的 anchor−0.5s 截断排除，飞行长轨迹才是落点证据；anchor_xy 分支
    外层 GOAL_TRACK_MAX_DIST_PX 上界判定不变。默认 False 行为逐点不变。

    Args:
        tracks: 窗口内重链的全部轨迹。
        anchor_sec: 进球锚点（秒）。
        anchor_xy: 候选锚点 (cx, cy)；None 表示无候选位置。
        prefer_longer: 长轨偏好开关（见上）。

    Returns:
        进球轨迹；无轨迹或端点离锚点太远返回 None。
    """
    if not tracks:
        return None

    def _dt(t: Track) -> float:
        return abs(t.last_det.sec - anchor_sec)

    if anchor_xy is None:
        if not prefer_longer:
            return min(tracks, key=_dt)
        best_dt: float = _dt(min(tracks, key=_dt))
        pool_t: list[Track] = [t for t in tracks if _dt(t) <= best_dt + PREFER_LONGER_SLACK_SEC]
        return max(pool_t, key=lambda t: (t.length, -_dt(t)))

    def _dist(t: Track) -> float:
        return euclidean((t.last_det.cx, t.last_det.cy), anchor_xy)

    best: Track = min(tracks, key=_dist)
    if prefer_longer:
        pool: list[Track] = [t for t in tracks if _dist(t) <= _dist(best) + PREFER_LONGER_SLACK_PX]
        best = max(pool, key=lambda t: (t.length, -_dist(t)))
    if _dist(best) > GOAL_TRACK_MAX_DIST_PX:
        return None
    return best


def find_held_box(
    track: Track, persons: tuple[tuple[Box, ...], ...]
) -> tuple[Detection, Box] | None:
    """沿轨迹从末端往回放，找最后一个球心严格落在某人框内（无 margin）的轨迹点。

    Args:
        track: 进球轨迹。
        persons: 全量 persons 缓存（按帧索引）。

    Returns:
        (轨迹点, 人框)；整轨无持球点返回 None。
    """
    for det in reversed(track.dets):
        for box in persons[det.frame_idx]:
            if box.x1 <= det.cx <= box.x2 and box.y1 <= det.cy <= box.y2:
                return det, box
    return None


def start_nearest_box(
    track: Track, persons: tuple[tuple[Box, ...], ...]
) -> tuple[Detection, Box] | None:
    """无持球点回退：取轨迹起点时刻离球心最近的人框（框中心距离）。

    Args:
        track: 进球轨迹。
        persons: 全量 persons 缓存（按帧索引）。

    Returns:
        (起点轨迹点, 最近人框)；起点帧无人返回 None。
    """
    first: Detection = track.dets[0]
    boxes: tuple[Box, ...] = persons[first.frame_idx]
    if not boxes:
        return None
    box: Box = min(
        boxes,
        key=lambda b: euclidean((first.cx, first.cy), ((b.x1 + b.x2) // 2, (b.y1 + b.y2) // 2)),
    )
    return first, box


def locate_scorer(
    cache: MotCache, anchor_sec: float, anchor_xy: tuple[int, int] | None = None
) -> LocateResult:
    """轨迹法定位投篮者（2026-08-08 替换逐帧投票）。

    窗口 [anchor−4.0, anchor+0.5] 内 run_mot 重链球轨迹（min_length=1）→
    端点距候选锚点最近者为进球轨迹 → 从末端回放找最后持球点（球心严格在人框内）
    → 整轨无持球点取轨迹起点最近人框。

    Args:
        cache: 校验后的 mot_cache。
        anchor_sec: 进球锚点（秒）。
        anchor_xy: 候选锚点 (cx, cy)；None 退化为端点时间最近选轨迹。

    Returns:
        LocateResult；SKIP 时 reason ∈ {no_track, no_track_near_anchor, no_person}。
    """
    tracks: list[Track] = run_mot(track_window_dets(cache, anchor_sec), min_length=1)
    if not tracks:
        return LocateResult(STATUS_SKIP, "no_track", -1, None, 0, 0)
    track: Track | None = select_goal_track(tracks, anchor_sec, anchor_xy)
    if track is None:
        return LocateResult(STATUS_SKIP, "no_track_near_anchor", -1, None, 0, len(tracks))
    held: tuple[Detection, Box] | None = find_held_box(track, cache.persons)
    if held is not None:
        det, box = held
        return LocateResult(STATUS_OK, "", det.frame_idx, box, track.length, len(tracks))
    fallback: tuple[Detection, Box] | None = start_nearest_box(track, cache.persons)
    if fallback is None:
        return LocateResult(STATUS_SKIP, "no_person", -1, None, track.length, len(tracks))
    det, box = fallback
    return LocateResult(STATUS_OK, "start_fallback", det.frame_idx, box, track.length, len(tracks))


def _best_iou_box(boxes: tuple[Box, ...], ref: Box) -> Box | None:
    """在候选框中取与 ref IoU 最大者；该帧无人或最大 IoU < TRACE_MIN_IOU 返回 None。

    Args:
        boxes: 某一帧的全部人框。
        ref: 参照框（上一链上框）。

    Returns:
        IoU 最大的人框；不达标返回 None（调用方视为链断）。
    """
    if not boxes:
        return None
    best: Box = max(boxes, key=lambda b: iou(b, ref))
    if iou(best, ref) < TRACE_MIN_IOU:
        return None
    return best


def trace_person(
    persons: tuple[tuple[Box, ...], ...], seed_frame: int, seed_box: Box
) -> list[tuple[int, Box]]:
    """以定位帧人框为种子向前后逐帧链同一人框（IoU≥TRACE_MIN_IOU，链断即停）。

    窗口 = 定位帧前后各 TRACE_WINDOW_SEC（5fps 即各 ≤10 帧），越出 mot_cache
    覆盖边界即停。每帧取与上一链上框 IoU 最大的人框续链（贴身对抗链错人的
    残余风险由认人页预览片段视频终裁兜底，spec §数据契约）。

    Args:
        persons: 全量 persons 缓存（按帧索引，load_mot_cache 产物）。
        seed_frame: 定位帧索引（链的种子，必含在返回序列中）。
        seed_box: 定位帧上的投篮者人框。

    Returns:
        (frame_idx, box) 序列，按 frame_idx 升序；至少含种子帧一项。

    Raises:
        BasketballPipelineError: seed_frame 越出 persons 覆盖范围（逻辑错误显式失败）。
    """
    if not 0 <= seed_frame < len(persons):
        raise BasketballPipelineError(
            f"trace_person 种子帧越界: seed_frame={seed_frame} frames={len(persons)}"
        )
    span: int = round(TRACE_WINDOW_SEC * SAMPLE_FPS)
    chain: dict[int, Box] = {seed_frame: seed_box}
    last: Box = seed_box
    for fi in range(seed_frame - 1, max(-1, seed_frame - span - 1), -1):
        nxt: Box | None = _best_iou_box(persons[fi], last)
        if nxt is None:
            break
        chain[fi] = nxt
        last = nxt
    last = seed_box
    for fi in range(seed_frame + 1, min(len(persons), seed_frame + span + 1)):
        nxt = _best_iou_box(persons[fi], last)
        if nxt is None:
            break
        chain[fi] = nxt
        last = nxt
    return sorted(chain.items())


def expand_box(box: Box, ratio: float, width: int, height: int) -> tuple[int, int, int, int]:
    """人框按比例外扩并裁剪到图像边界（每维放大 1+ratio 倍，即每侧 ratio/2）。

    Args:
        box: 原人框。
        ratio: 外扩比例（0.2 = 每侧 10%）。
        width: 图像宽（像素）。
        height: 图像高（像素）。

    Returns:
        外扩并夹取后的 (x1, y1, x2, y2)。
    """
    pad_x: int = round((box.x2 - box.x1) * ratio / 2)
    pad_y: int = round((box.y2 - box.y1) * ratio / 2)
    x1: int = max(0, box.x1 - pad_x)
    y1: int = max(0, box.y1 - pad_y)
    x2: int = min(width, box.x2 + pad_x)
    y2: int = min(height, box.y2 + pad_y)
    return x1, y1, x2, y2


def _gate_view(img: Image.Image, box: Box) -> Image.Image:
    """裁图视图 = 外扩 20% 裁出 + 短边不足 400px 等比放大（crop_and_save 与质量闸
    框内复检共用，复检输入与标定 gate-view 口径一致）。

    Args:
        img: 帧图（与 box 同坐标系）。
        box: 人框（未外扩）。

    Returns:
        裁出并（必要时）放大后的 RGB 图；退化框（短边 0）不放大，原样返回空裁图。
    """
    x1, y1, x2, y2 = expand_box(box, CROP_EXPAND, img.width, img.height)
    crop = img.crop((x1, y1, x2, y2))
    short: int = min(crop.size)
    if 0 < short < CROP_MIN_SHORT_SIDE:
        scale: float = CROP_MIN_SHORT_SIDE / short
        new_size: tuple[int, int] = (
            round(crop.width * scale),
            round(crop.height * scale),
        )
        crop = crop.resize(new_size, Image.Resampling.LANCZOS)
    return crop


def crop_and_save(
    img_path: Path, box: Box, out_path: Path, *, img: Image.Image | None = None
) -> None:
    """裁出投篮者：外扩 20%，短边不足 400px 等比放大到 400px，存 JPEG。

    Args:
        img_path: 代表帧图片路径。
        box: 代表帧上的胜出人框（与图片同坐标系）。
        out_path: 裁图输出路径（父目录自动创建）。
        img: 已打开的 RGB 帧图（必须与 img_path 同帧）；给定时复用不重复解码
            （_process_goal 种子帧：seed_team 判定与裁图共用一次打开）。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img is not None:
        _gate_view(img, box).save(out_path, "JPEG", quality=JPEG_QUALITY)
        return
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        crop = _gate_view(rgb, box)
        crop.save(out_path, "JPEG", quality=JPEG_QUALITY)


def expanded_ratio(box: Box, width: int, height: int) -> float:
    """外扩 20% 并夹取图像边界后的框宽高比 w/h（质量闸第一道作用口径，同标定
    report §口径说明）。

    Args:
        box: 人框（未外扩）。
        width: 图像宽（像素）。
        height: 图像高（像素）。

    Returns:
        外扩夹取后 w/h；退化框（夹取后高 ≤0）返回 0.0（必被拦，不抛除零）。
    """
    x1, y1, x2, y2 = expand_box(box, CROP_EXPAND, width, height)
    if y2 - y1 <= 0:
        return 0.0
    return (x2 - x1) / (y2 - y1)


def ratio_gate_reason(box: Box, width: int, height: int) -> str | None:
    """宽高比 sanity 闸（质量闸第一道，纯几何零成本先行）：外扩后框 w/h 越界
    [GATE_MIN_RATIO, GATE_MAX_RATIO] → 拦截原因，否则 None。

    端点含入（恰好等于上下限放行——宁漏拦不错杀，标定 report §阈值建议）。

    Args:
        box: 人框（未外扩）。
        width: 图像宽（像素）。
        height: 图像高（像素）。

    Returns:
        拦截原因字符串（含实测比值）；放行返回 None。
    """
    ratio: float = expanded_ratio(box, width, height)
    if ratio < GATE_MIN_RATIO or ratio > GATE_MAX_RATIO:
        return f"宽高比 {ratio:.3f} 越界 [{GATE_MIN_RATIO}, {GATE_MAX_RATIO}]"
    return None


PersonRechecker = Callable[[Image.Image], float]  # gate-view 裁图 → 最高 person 置信度（无框 0.0）

_person_rechecker: PersonRechecker | None = None  # 惰性缓存（模块级不加载模型，rules.md §1）


def _yolov8n_person_rechecker() -> PersonRechecker:
    """加载 yolov8n 并返回框内人物复检器（质量闸第二道）。

    只检 person 类、imgsz=640、取最高置信度（无 person 框记 0.0），与标定口径一致
    （work/crop_quality_calibration/report.md §口径说明）。模型缺失显式失败。

    Returns:
        复检函数（gate-view 裁图 → 最高 person 置信度）。

    Raises:
        BasketballPipelineError: 模型文件缺失。
    """
    from ultralytics import YOLO

    model_path = Path(GATE_PERSON_MODEL_PATH)
    if not model_path.is_file():
        raise BasketballPipelineError(f"质量闸人物模型缺失: {model_path}")
    model = YOLO(str(model_path))

    def recheck(crop: Image.Image) -> float:
        results = model.predict(
            np.asarray(crop),
            imgsz=GATE_PERSON_IMGSZ,
            classes=[YOLO_PERSON_CLS],
            verbose=False,
        )
        best: float = 0.0
        for res in results:
            if res.boxes is None:
                continue
            for conf in res.boxes.conf.tolist():
                best = max(best, float(conf))
        return best

    return recheck


def _get_person_rechecker() -> PersonRechecker:
    """惰性加载并缓存 yolov8n 复检器（仅裁图质量闸首次用到时才加载；测试注入假检测器）。"""
    global _person_rechecker
    if _person_rechecker is None:
        _person_rechecker = _yolov8n_person_rechecker()
    return _person_rechecker


def make_quality_gate(
    framesdir: Path,
    fid: str,
    key: str,
    rechecker: PersonRechecker | None = None,
) -> Callable[[int, Box | None], str | None]:
    """构造裁图质量闸（crop-quality spec §数据契约：宽高比 sanity 先行 + 框内人物复检殿后）。

    闸插选帧循环内（pick_best_frames gate 参数）：候选帧逐帧过闸，废帧丢弃继续
    取次优帧补位。缺框帧（检测缓存无框，box=None）跳过复检不拦截，记 INFO 不静默
    丢弃。宽高比为纯几何零成本先行；yolov8n 复检只对过了几何闸的帧跑（省 CPU），
    rechecker 未注入时惰性加载（首次真正需要复检时才加载，测试可注入假检测器）。

    Args:
        framesdir: 帧图根目录。
        fid: 视频主名（帧路径映射用）。
        key: 进球键（留痕日志用）。
        rechecker: 框内人物复检器；None 时惰性加载 models/yolov8n.pt。

    Returns:
        闸函数 (frame_idx, box) -> 拦截原因 | None（放行）；拦截/缺框跳过时记
        INFO 留痕（帧时刻 + 原因）。
    """
    resolved_rechecker: PersonRechecker | None = rechecker

    def _resolve() -> PersonRechecker:
        nonlocal resolved_rechecker
        if resolved_rechecker is None:
            resolved_rechecker = _get_person_rechecker()
        return resolved_rechecker

    def gate(frame_idx: int, box: Box | None) -> str | None:
        sec: float = frame_idx / SAMPLE_FPS
        if box is None:  # 检测缓存缺框：跳过复检不拦截，记 INFO 不静默丢弃
            logger.info("质量闸跳过（缓存缺框，不拦截）: %s 帧=%d t=%.1fs", key, frame_idx, sec)
            return None
        path: Path = _frame_path(framesdir, fid, frame_idx)
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
        except (OSError, ValueError) as exc:  # score_chain_frames 已剔除不可读帧，防御不拦截
            logger.warning("质量闸帧图不可读，跳过复检不拦截: %s: %s", path, exc)
            return None
        reason: str | None = ratio_gate_reason(box, rgb.width, rgb.height)
        if reason is None:
            conf: float = _resolve()(_gate_view(rgb, box))
            if conf < GATE_PERSON_CONF:
                reason = f"框内无可信 person（最高 conf={conf:.3f} < {GATE_PERSON_CONF}）"
        if reason is not None:
            logger.info("质量闸拦截: %s 帧=%d t=%.1fs %s", key, frame_idx, sec, reason)
        return reason

    return gate


def frame_quality(img: Image.Image, box: Box) -> float:
    """质量分 = 归一化框面积 × 框内灰度 Laplacian 方差（仅相对排序有意义）。

    框先夹取到图像边界；夹取后退化为空框返回 0.0（防御，正常 mot_cache 框不触发）。

    Args:
        img: 帧图（与 box 同坐标系）。
        box: 人框（未外扩，评投篮者本体的面积与清晰度）。

    Returns:
        质量分；面积按整帧占比归一，Laplacian 方差刻画清晰度。
    """
    x1: int = max(0, box.x1)
    y1: int = max(0, box.y1)
    x2: int = min(img.width, box.x2)
    y2: int = min(img.height, box.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    arr = np.asarray(img.crop((x1, y1, x2, y2)).convert("L"))
    sharpness: float = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    area_norm: float = ((x2 - x1) * (y2 - y1)) / (img.width * img.height)
    return area_norm * sharpness


def score_chain_frames(
    chain: list[tuple[int, Box]], framesdir: Path, fid: str
) -> list[tuple[int, Box, float, str]]:
    """链上帧逐帧读图算质量分与队伍；帧图缺失/解码失败记 WARNING 跳过该帧（不炸整球）。

    Args:
        chain: trace_person 产物（frame_idx 升序）。
        framesdir: 帧图根目录。
        fid: 视频主名（帧路径映射用）。

    Returns:
        (frame_idx, box, score, team) 序列（保持链序，未按分排序）；跳过的帧不在其中。
        team 由 team_of_box 在已开图像上直算（串人守卫用，零额外解码）。
    """
    scored: list[tuple[int, Box, float, str]] = []
    for fi, box in chain:
        path: Path = _frame_path(framesdir, fid, fi)
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                score: float = frame_quality(rgb, box)
                team: str = team_of_box(rgb, box)
        except (OSError, ValueError) as exc:  # 文件缺失/损坏属业务可预期，跳过该帧
            logger.warning("链上帧图不可读，跳过该帧: %s: %s", path, exc)
            continue
        scored.append((fi, box, score, team))
    return scored


def pick_best_frames(
    scored: list[tuple[int, Box, float, str]],
    n: int,
    gate: Callable[[int, Box | None], str | None] | None = None,
) -> list[tuple[int, Box, float, str]]:
    """按质量分降序贪心取 top n，入选帧间隔 ≥CROP_MIN_SPACING_SEC 去重。

    质量分并列时按帧索引升序优先（稳定可测）；间隔按 5fps 帧差折算（0.5s=2.5 帧，
    即帧差 ≥3 才入选）。给了 gate（裁图质量闸，crop-quality spec §行为口径）则
    逐帧过闸：废帧丢弃不占位、不耗入选间距，继续取次优帧补位，直到凑够 n 张或
    帧池耗尽（全废返回空列表，由上层判 SKIP）。

    Args:
        scored: score_chain_frames 产物（team 字段不参与排序，原样透传）。
        n: 最多入选帧数（--best-crops）。
        gate: 质量闸 (frame_idx, box) -> 拦截原因 | None；None 表示不过闸
            （默认行为与闸引入前一致）。

    Returns:
        入选 (frame_idx, box, score, team)，按质量分降序；scored 为空返回空列表。

    Raises:
        ValueError: n < 1（参数错误显式失败）。
    """
    if n < 1:
        raise ValueError(f"pick_best_frames 要求 n ≥ 1，实际 {n}")
    picked: list[tuple[int, Box, float, str]] = []
    for item in sorted(scored, key=lambda it: (-it[2], it[0])):
        if len(picked) >= n:
            break
        if not all(
            abs(item[0] - sel[0]) / SAMPLE_FPS >= CROP_MIN_SPACING_SEC - _EPS for sel in picked
        ):
            continue
        if gate is not None and gate(item[0], item[1]) is not None:
            continue  # 废帧丢弃：不占位、不影响间距，次优帧补位
        picked.append(item)
    return picked


def _team_from_crop(crop_img: Image.Image) -> str:
    """躯干主色分队核心逻辑（黑/白/便服）；采样区与阈值同 spec §颜色分队判据 M4。"""
    hsv = crop_img.convert("HSV")
    x1: int = round(hsv.width * 0.2)
    x2: int = round(hsv.width * 0.8)
    y1: int = round(hsv.height * 0.25)
    y2: int = round(hsv.height * 0.6)
    arr = np.asarray(hsv.crop((x1, y1, x2, y2)), dtype=np.uint8)
    if arr.size == 0:
        return TEAM_CASUAL  # 采样区退化（极小裁图）不瞎猜
    sat = arr[..., 1].astype(np.int16)
    val = arr[..., 2].astype(np.int16)
    black_frac: float = float(np.mean(val < TH_BLACK))
    white_frac: float = float(np.mean((val > TH_WHITE) & (sat < TH_SAT)))
    if black_frac >= MIN_BLACK_FRACTION and black_frac >= white_frac:
        return TEAM_BLACK
    if white_frac >= MIN_WHITE_FRACTION:
        return TEAM_WHITE
    return TEAM_CASUAL


def classify_team(crop_path: Path) -> str:
    """按躯干主色分队：黑 / 白 / 便服（spec §颜色分队判据 M4）。

    采样区 = 人框水平中 60% × 垂直 25%~60%；黑：V<TH_BLACK 占比达标；
    白：V>TH_WHITE 且 S<TH_SAT 占比达标；两者均不达标（含近阈混杂）归"便服"。

    Args:
        crop_path: 投篮者裁图路径。

    Returns:
        "黑" / "白" / "便服"。
    """
    with Image.open(crop_path) as im:
        return _team_from_crop(im)


def team_of_box(img: Image.Image, box: Box) -> str:
    """整帧图 + 人框直接分队（多裁串人守卫用，省去裁图落盘再读）。

    Args:
        img: 帧图（与 box 同坐标系）。
        box: 人框（未外扩）。

    Returns:
        "黑" / "白" / "便服"；框夹取后退化为空归"便服"（不瞎猜）。
    """
    x1: int = max(0, box.x1)
    y1: int = max(0, box.y1)
    x2: int = min(img.width, box.x2)
    y2: int = min(img.height, box.y2)
    if x2 <= x1 or y2 <= y1:
        return TEAM_CASUAL
    return _team_from_crop(img.crop((x1, y1, x2, y2)))


_OPPOSITE_TEAM: dict[str, str] = {TEAM_BLACK: TEAM_WHITE, TEAM_WHITE: TEAM_BLACK}


def drop_opposite_team(
    scored: list[tuple[int, Box, float, str]], seed_frame: int
) -> list[tuple[int, Box, float, str]]:
    """剔除与种子帧队伍明确相反（黑↔白）的链上帧（IoU 链串人守卫）。

    种子为"便服"（判定不自信）或种子帧不在 scored（帧图不可读）时不过滤；
    "便服"帧一律保留（近阈混杂可能是同一人）。

    Args:
        scored: score_chain_frames 产物（frame_idx, box, score, team）。
        seed_frame: 定位帧索引（其 team 为基准）。

    Returns:
        过滤后的序列（保持原序）；被剔帧记 INFO 日志。
    """
    seed_team: str | None = next((t for fi, _b, _s, t in scored if fi == seed_frame), None)
    opposite: str | None = _OPPOSITE_TEAM.get(seed_team or "")
    if opposite is None:
        return scored
    kept: list[tuple[int, Box, float, str]] = []
    dropped: list[int] = []
    for item in scored:
        if item[3] == opposite:
            dropped.append(item[0])
        else:
            kept.append(item)
    if dropped:
        logger.info(
            "串人守卫: 种子帧=%d team=%s，剔除 %s 队帧 %s", seed_frame, seed_team, opposite, dropped
        )
    return kept


def _confirmed_goals(data: Any, goals_path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """从 goals.json 数据中取 confirmed 记录（缺 file/anchor_time 显式失败）。

    Args:
        data: read_json 读出的原始 JSON。
        goals_path: 文件路径（仅用于错误信息）。

    Returns:
        confirmed 记录列表（保留原始 dict）。

    Raises:
        SchemaError: 顶层非对象 / goals 非列表 / confirmed 记录缺字段或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{goals_path}: 顶层必须是对象，实际 {type(data).__name__}")
    goals: Any = data.get("goals")
    if not isinstance(goals, list):
        raise SchemaError(f"{goals_path}: 缺 goals 列表或类型错误")
    confirmed: list[dict[str, Any]] = []
    for i, g in enumerate(goals):
        if not isinstance(g, dict):
            raise SchemaError(f"{goals_path}: 第{i}条记录不是对象")
        if g.get("status") != "confirmed":
            continue
        if not isinstance(g.get("file"), str) or not g["file"]:
            raise SchemaError(f"{goals_path}: 第{i}条(confirmed) file 缺失或不是非空 str")
        anchor: Any = g.get("anchor_time")
        if not isinstance(anchor, (int, float)):
            raise SchemaError(f"{goals_path}: 第{i}条(confirmed) anchor_time 缺失或非数值")
        confirmed.append(g)
    return confirmed


def _frame_path(framesdir: Path, fid: str, frame_idx: int) -> Path:
    """帧映射：fid + frame_idx → work/frames/<fid>/f_{frame_idx+1:05d}.jpg。"""
    return framesdir / fid / f"f_{frame_idx + 1:05d}.jpg"


def _crop_name(fid: str, anchor_sec: float) -> str:
    """裁图文件名：<fid>_t<anchor:.1f>.jpg（同 fid 多球靠锚点区分）。"""
    return f"{fid}_t{anchor_sec:.1f}.jpg"


def _crop_name_ranked(fid: str, anchor_sec: float, rank: int) -> str:
    """多裁文件名：rank 1 = 主名（向后兼容），rank≥2 主名去后缀追加 _q{rank}。

    Args:
        fid: 视频主名。
        anchor_sec: 进球锚点（秒）。
        rank: 质量排名（1 起）。

    Returns:
        裁图文件名；rank=2/3 如 <fid>_t<anchor:.1f>_q2.jpg。
    """
    name: str = _crop_name(fid, anchor_sec)
    if rank <= 1:
        return name
    return f"{name.removesuffix('.jpg')}_q{rank}.jpg"


def preview_window(anchor_sec: float) -> tuple[float, float]:
    """预览片段窗口：[max(0, anchor−4s), anchor+2s]（与剪辑规格一致，锚点严格对齐）。

    Args:
        anchor_sec: 进球锚点（秒）。

    Returns:
        (start_sec, end_sec)；anchor<4s 时起点夹取到 0（时长缩短，不越界）。
    """
    return max(0.0, anchor_sec - PREVIEW_BEFORE_SEC), anchor_sec + PREVIEW_AFTER_SEC


def _preview_name(fid: str, anchor_sec: float) -> str:
    """预览片段文件名：<fid>_t<anchor:.1f>.mp4（与裁图同命名口径）。"""
    return f"{fid}_t{anchor_sec:.1f}.mp4"


def cut_preview_clip(rawdir: Path, file: str, anchor_sec: float, out_path: Path) -> None:
    """从原片切认人预览片段：输入侧 -ss/-to，1280 宽、libx264、无声。

    Args:
        rawdir: 原片目录。
        file: 视频文件名（rawdir 下的 basename）。
        anchor_sec: 进球锚点（秒）。
        out_path: 输出片段路径（父目录自动创建）。

    Raises:
        BasketballPipelineError: ffmpeg 重试耗尽。
        MediaTimeoutError: ffmpeg 超时（时长×3+60s，下限 120s，同 build_highlight 口径）。
    """
    start, end = preview_window(anchor_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-ss",
            f"{start:.2f}",
            "-to",
            f"{end:.2f}",
            "-i",
            str(rawdir / file),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={PREVIEW_WIDTH}:-2",
            "-c:v",
            "libx264",
            "-crf",
            str(PREVIEW_CRF),
            "-preset",
            PREVIEW_PRESET,
            "-an",
            str(out_path),
        ],
        timeout_sec=max(120, int((end - start) * 3) + 60),
    )


def _try_cut_preview(
    goal: dict[str, Any], rawdir: Path | None, outdir: Path, key: str
) -> tuple[str, bool]:
    """尝试切预览片段；失败记 ERROR 继续（不炸整批，但计入缺失错误使退出码非零）。

    Args:
        goal: confirmed 记录。
        rawdir: 原片目录；None 表示未给 --rawdir（不切片）。
        outdir: 输出目录（片段落 <outdir>/clips/）。
        key: 进球键（日志用）。

    Returns:
        (clip 相对路径, 是否发生切片失败)；未切片或失败时 clip 为空串。
    """
    if rawdir is None:
        return "", False
    file: str = goal["file"]
    anchor: float = float(goal["anchor_time"])
    name: str = _preview_name(fid_of(file), anchor)
    src: Path = rawdir / file
    if not src.is_file():
        logger.error("原片缺失，预览片段跳过: %s (%s)", src, key)
        return "", True
    try:
        cut_preview_clip(rawdir, file, anchor, outdir / "clips" / name)
    except BasketballPipelineError as exc:
        logger.error("预览片段切失败，跳过: %s (%s): %s", name, key, exc)
        return "", True
    return f"clips/{name}", False


def _process_goal(
    goal: dict[str, Any],
    detectdir: Path,
    framesdir: Path,
    outdir: Path,
    rawdir: Path | None = None,
    anchor_xy: tuple[int, int] | None = None,
    best_crops: int = DEFAULT_BEST_CROPS,
    rechecker: PersonRechecker | None = None,
) -> tuple[dict[str, Any], bool]:
    """处理单个 confirmed 球：切预览片段（--rawdir 时，SKIP 球也切）→ 定位 → 多裁 → 颜色分队。

    多裁（scorer-cluster spec §数据契约）：定位 OK 后 trace_person 链同一人框 →
    链上帧算质量分 → pick_best_frames 取 top best_crops（≥0.5s 去重）→ 逐张裁图。
    选帧循环内过裁图质量闸（crop-quality spec §行为口径）：候选帧逐帧先宽高比
    sanity 再框内人物复检，废帧丢弃取次优帧补位；候选帧全被拦 → status=SKIP、
    reason=quality_gate（预览片段保留、不落 crops/crop_scores）。
    entry 落 crops（质量降序文件名）与 crop_scores，crop = crops[0] 保持向后兼容；
    crops/crop_scores 只在 status=OK 时存在，SKIP 条目不含这两个字段。
    OK 条目另落轨迹传播种子三字段（scorer-propagate spec §crop_scorers 小改）：
    seed_frame（定位帧索引）、seed_box（[x1,y1,x2,y2] 定位帧人框）、seed_team
    （种子帧 team_of_box 结果，与种子帧裁图复用同一次解码）；SKIP 球不落。

    Args:
        goal: confirmed 记录。
        detectdir: mot_cache 目录。
        framesdir: 帧图根目录。
        outdir: 输出目录。
        rawdir: 原片目录；None 表示不切预览片段。
        anchor_xy: 候选锚点 (cx, cy)（--candidates 匹配产物）；None 时轨迹选择
            退化为端点时间最近。
        best_crops: 每球最多裁图张数（--best-crops）。
        rechecker: 框内人物复检器（质量闸第二道）；None 时惰性加载 yolov8n。

    Returns:
        (候选记录, 是否发生素材缺失错误)。素材缺失（cache/帧图/原片不存在、
        切片失败）记 ERROR/SKIP 并返回 True（产出型脚本口径：跳过但进程退出码非零）。
    """
    file: str = goal["file"]
    anchor: float = float(goal["anchor_time"])
    fid: str = fid_of(file)
    entry: dict[str, Any] = {
        "key": format_key(file, anchor),
        "file": file,
        "anchor_time": anchor,
        "status": STATUS_SKIP,
        "reason": "",
        "crop": "",
        "clip": "",
        "team_guess": None,
        "number_guess": None,
        "number_votes": None,
        "votes": 0,
        "total_votes": 0,
    }

    # 预览片段与定位解耦：SKIP 球也需要视频供认人手选
    clip, clip_failed = _try_cut_preview(goal, rawdir, outdir, entry["key"])
    entry["clip"] = clip

    cache_path: Path = detectdir / f"{fid}_mot_cache.json"
    if not cache_path.is_file():
        logger.error("mot_cache 缺失，跳过: %s (%s)", cache_path, entry["key"])
        entry["reason"] = "missing_cache"
        return entry, True
    cache: MotCache = load_mot_cache(cache_path)

    result: LocateResult = locate_scorer(cache, anchor, anchor_xy)
    entry["votes"] = result.votes
    entry["total_votes"] = result.total_votes
    if result.status == STATUS_SKIP:
        logger.info(
            "定位 SKIP: %s reason=%s 轨迹数=%d",
            entry["key"],
            result.reason,
            result.total_votes,
        )
        entry["reason"] = result.reason
        return entry, clip_failed
    if result.box is None:  # 防御：OK 必有 box，逻辑错误显式失败而非静默
        raise BasketballPipelineError(f"定位 OK 但 box 为空: {entry['key']}")

    frame_path: Path = _frame_path(framesdir, fid, result.frame_idx)
    if not frame_path.is_file():
        logger.error("代表帧缺失，跳过: %s (%s)", frame_path, entry["key"])
        entry["reason"] = "missing_frame"
        return entry, True
    # 种子帧图只开一次：seed_team 判定与种子帧裁图复用同一次解码
    try:
        with Image.open(frame_path) as im:
            seed_rgb: Image.Image = im.convert("RGB")
    except (OSError, ValueError) as exc:
        logger.error("代表帧解码失败，跳过: %s (%s): %s", frame_path, entry["key"], exc)
        entry["reason"] = "decode_failed"
        return entry, True

    # 轨迹选帧多裁：定位帧人框为种子链同一人框，按质量分取 top best_crops（≥0.5s 去重）；
    # 选帧循环内过质量闸（宽高比先行、框内人物复检殿后），废帧丢弃取次优帧补位
    chain: list[tuple[int, Box]] = trace_person(cache.persons, result.frame_idx, result.box)
    scored: list[tuple[int, Box, float, str]] = score_chain_frames(chain, framesdir, fid)
    scored = drop_opposite_team(scored, result.frame_idx)  # 串人守卫：剔黑↔白明确相反帧
    gate = make_quality_gate(framesdir, fid, entry["key"], rechecker)
    picked: list[tuple[int, Box, float, str]] = pick_best_frames(scored, best_crops, gate=gate)
    if not picked and scored:  # 候选帧池非空但全被质量闸拦截 → SKIP（预览片段保留，不落 crops）
        entry["reason"] = REASON_QUALITY_GATE
        logger.info("候选帧全被质量闸拦截，SKIP: %s 候选帧=%d", entry["key"], len(scored))
        return entry, clip_failed
    if not picked:  # 防御：定位帧 is_file 已过但解码失败 → 回退定位帧单裁（裁图报错由下层抛出）
        picked = [(result.frame_idx, result.box, 0.0, TEAM_CASUAL)]
    crops: list[str] = []
    crop_scores: list[float] = []
    for rank, (fi, box, score, _team) in enumerate(picked, start=1):
        name: str = _crop_name_ranked(fid, anchor, rank)
        crop_and_save(
            _frame_path(framesdir, fid, fi),
            box,
            outdir / name,
            img=seed_rgb if fi == result.frame_idx else None,
        )
        crops.append(name)
        crop_scores.append(round(score, 4))
    team: str = classify_team(outdir / crops[0])
    entry["status"] = STATUS_OK
    entry["crop"] = crops[0]
    entry["crops"] = crops
    entry["crop_scores"] = crop_scores
    entry["team_guess"] = team
    # 轨迹传播种子（scorer-propagate spec §crop_scorers 小改）：propagate_scorers
    # 只消费这三字段，不重跑 locate_scorer；SKIP 球不落（与 crops 口径一致）
    entry["seed_frame"] = result.frame_idx
    entry["seed_box"] = [result.box.x1, result.box.y1, result.box.x2, result.box.y2]
    entry["seed_team"] = team_of_box(seed_rgb, result.box)
    logger.info(
        "定位 OK: %s 帧=%d 轨长=%d/%d %steam=%s 多裁=%d/%d",
        entry["key"],
        result.frame_idx,
        result.votes,
        result.total_votes,
        "(起点回退) " if result.reason == "start_fallback" else "",
        team,
        len(crops),
        len(chain),
    )
    return entry, clip_failed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="投篮者定位裁图 + 颜色分队（spec B2/M4）")
    parser.add_argument("--goals", required=True, type=Path, help="goals.json 路径")
    parser.add_argument("--detectdir", required=True, type=Path, help="mot_cache 目录")
    parser.add_argument("--framesdir", required=True, type=Path, help="帧图根目录")
    parser.add_argument("--out", required=True, type=Path, help="输出目录（裁图+候选 JSON）")
    parser.add_argument(
        "--rawdir",
        type=Path,
        default=None,
        help="原片目录（可选；给了就逐球切认人预览片段到 <out>/clips/）",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="candidates.json（可选；提供候选锚点 cx/cy 供轨迹法选轨迹，不给则退化为端点时间最近）",
    )
    parser.add_argument(
        "--read-numbers",
        action="store_true",
        help="K3 号码识别（可选；逐张 crops 读号+众数投票，结果落 <out>/number_cache.json "
        "幂等不重复扣额度）",
    )
    parser.add_argument(
        "--numbers-cache-only",
        action="store_true",
        help="跳票模式（配合 --read-numbers）：缓存未命中的裁图跳过不读、只用已有票投票，"
        "零新调用（旧数据重跑回填用）",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=MAX_NUMBER_READS_PER_RUN,
        help="单次运行最大新识别张数（默认 %(default)s；用户批准后显式放宽）",
    )
    parser.add_argument(
        "--best-crops",
        type=int,
        default=DEFAULT_BEST_CROPS,
        help="每球最多裁图张数（默认 %(default)s；轨迹选帧按质量分取 top N，≥0.5s 去重）",
    )
    ns = parser.parse_args(argv)
    if ns.best_crops < 1:
        parser.error("--best-crops 须 ≥ 1")
    return ns


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=全部处理无素材缺失，1=失败或有素材缺失）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        goals_path: Path = args.goals
        data: Any = read_json(goals_path, what="goals.json")
        confirmed: list[dict[str, Any]] = _confirmed_goals(data, str(goals_path))
        session: str = data.get("session", "") if isinstance(data, dict) else ""
        logger.info("confirmed 球 %d 个，开始定位", len(confirmed))

        args.out.mkdir(parents=True, exist_ok=True)
        cand_index: dict[str, list[tuple[float, int, int]]] = {}
        if args.candidates:
            cand_index = load_candidates_index(args.candidates)
            logger.info("候选锚点索引: %d 个 fid ← %s", len(cand_index), args.candidates)
        entries: list[dict[str, Any]] = []
        missing_errors: int = 0
        for goal in confirmed:
            anchor_xy: tuple[int, int] | None = None
            if cand_index:
                anchor_xy = match_anchor_xy(
                    cand_index, fid_of(goal["file"]), float(goal["anchor_time"])
                )
                if anchor_xy is None:
                    logger.warning(
                        "候选锚点未匹配（|t0−anchor|>%.1fs），退化为端点时间最近: %s",
                        CANDIDATE_MATCH_DT_SEC,
                        format_key(goal["file"], float(goal["anchor_time"])),
                    )
            entry, had_missing = _process_goal(
                goal,
                args.detectdir,
                args.framesdir,
                args.out,
                args.rawdir,
                anchor_xy,
                args.best_crops,
            )
            entries.append(entry)
            missing_errors += int(had_missing)

        if args.read_numbers:
            n_fresh, total_tokens = apply_number_reading(
                entries, args.out, args.max_reads, cache_only=args.numbers_cache_only
            )
            logger.info(
                "号码识别完成%s: 新识别 %d 张，本次 %d tokens",
                "（跳票模式）" if args.numbers_cache_only else "",
                n_fresh,
                total_tokens,
            )

        out_json: Path = args.out / "scorer_candidates.json"
        atomic_write_json(
            out_json, {"session": session, "candidates": entries}, what="scorer_candidates.json"
        )
        ok: int = sum(1 for e in entries if e["status"] == STATUS_OK)
        n_clip: int = sum(1 for e in entries if e["clip"])
        teams: dict[str, int] = {}
        for e in entries:
            if e["status"] == STATUS_OK:
                teams[e["team_guess"]] = teams.get(e["team_guess"], 0) + 1
        logger.info(
            "完成: OK=%d SKIP=%d 预览片段=%d 缺失错误=%d 颜色分布=%s → %s",
            ok,
            len(entries) - ok,
            n_clip,
            missing_errors,
            teams,
            out_json,
        )
        if missing_errors:
            logger.error("有 %d 条素材缺失（详见上条 ERROR），退出码非零", missing_errors)
            return 1
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
