"""离线读号：传统 CV 模板匹配读裁图背号（scorer-propagate spec §离线读号，试验性质）。

输入：scorer_candidates.json（--candidates；裁图在其同目录）、K3 number_cache.json
    （--bootstrap-from，只读——自举模板库用）。
输出：offline_number_cache.json（--out，缺省 = candidates 同目录；键 = 裁图 md5，
    结构参照 number_cache：_meta 标 source=offline；另带 votes 段记录逐球投票结果，
    供 Phase 2 对照实跑（Task 10）直接取用）。
依赖：scripts/crop_scorers.py（NumberGuess / vote_number_guess / number_guess_from_dict
    / file_md5 / _entry_crops / load_number_cache 复用，投票规则不重写）、
    scripts/propagate_scorers.py（load_scorer_candidates schema 校验）、opencv + numpy + PIL。
典型调用：
    python scripts/offline_number.py \
        --candidates work/<场次>/scorers_bK/scorer_candidates.json \
        --bootstrap-from work/<场次>/scorers_bK/number_cache.json

零模型零 token。链路（spec 写死）：裁图躯干上部（垂直 10%~45%、水平中 70%）→
放大 3 倍 → 自适应二值化 → 连通域切数字候选 → 与模板库 matchTemplate（多尺度）→
逐帧投票（复用 crop_scorers.vote_number_guess）。
模板库自举对齐规则（写死）：对 number_cache 里 K3 高置信条目的裁图跑同套切分，
连通域数 == len(number) 才接收样本，按连通域中心 x 升序与数字字符一一 zip 对应；
粘连/断裂致个数不符 → 丢弃记 INFO；每数字 ≥MIN_SAMPLES_PER_DIGIT 个样本才启用。
缓存隔离（写死）：只写 offline_number_cache.json，绝不碰 number_cache.json（K3 链路
零感知）。跳票模式并集入口 = merged_number_caches（K3 缓存 ∪ offline 缓存，K3 优先）；
接入 crop_scorers 跳票模式的改动不在本脚本（Phase 2 实跑达标后再接）。
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from crop_scorers import (
    STATUS_OK,
    NumberGuess,
    _entry_crops,
    file_md5,
    load_number_cache,
    number_guess_from_dict,
    vote_number_guess,
)
from errors import BasketballPipelineError
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from propagate_scorers import load_scorer_candidates

logger = logging.getLogger(__name__)

# ---- 预处理参数（spec §离线读号，写死） ----
TORSO_V_START: float = 0.10  # 躯干上部起点（裁图高度比例，排除头部）
TORSO_V_END: float = 0.45  # 躯干上部终点（背号区，不到腰腹）
TORSO_H_CENTER_FRAC: float = 0.70  # 水平取中 70%（排除手臂/背景边缘）
UPSCALE_FACTOR: int = 3  # 放大倍数（小字可读性，spec 写死）
ADAPTIVE_BLOCK_SIZE: int = (
    61  # 局部高斯均值邻域（奇数；须明显大于放大后笔画宽 ~30px，否则笔画中心被判背景）
)
ADAPTIVE_C: int = 10  # 前景偏差下限：|像素 - 局部均值| > ADAPTIVE_C 判前景
MIN_TORSO_SIDE_PX: int = 8  # 躯干区最小边长（放大前），过小视为图畸形无法处理

# ---- 连通域数字候选过滤（相对躯干区比例） ----
DIGIT_MIN_AREA_FRAC: float = 0.002  # 面积占比下限（滤噪点）
DIGIT_MAX_AREA_FRAC: float = 0.30  # 面积占比上限（滤整块背景/边框）
DIGIT_MIN_HEIGHT_FRAC: float = 0.20  # 高度占比下限（背号在躯干上部应占可观高度）
DIGIT_MAX_ASPECT: float = 1.2  # w/h 上限（数字通常竖长；横宽块 = 粘连/图案）
MAX_DIGITS: int = 2  # 背号最多 2 位；连通域更多 = 切分不可信，整张跳过

# ---- 模板匹配 ----
CANON_HEIGHT: int = 64  # 归一化高度（像素），匹配前双方统一到同一尺度基准
MATCH_SCALES: tuple[float, ...] = (0.8, 1.0, 1.25)  # 多尺度因子（spec：多尺度匹配）
MATCH_MIN_SCORE: float = 0.55  # 单数字采纳下限（TM_CCOEFF_NORMED）
HIGH_CONF_SCORE: float = 0.75  # 全部数字 ≥ 此分 → confidence=high（喂投票规则）
MATCH_ASPECT_TOL: float = 1.6  # 模板与候选宽高比允许偏差倍数（防"子集匹配"误判，如 1⊂7）
MATCH_SLIDE_PX: int = 4  # 居中画布横向滑动容差（像素，吸收归一化取整误差）

# ---- 模板库自举（spec 写死） ----
MIN_SAMPLES_PER_DIGIT: int = 3  # 每数字启用匹配的最少样本数，不足不认

# ---- 缓存（隔离：只写 offline_number_cache.json） ----
OFFLINE_CACHE_VERSION: str = "offline-number-v1"  # 算法语义变更即升版本，旧缓存作废
OFFLINE_CACHE_SOURCE: str = "offline"
OFFLINE_CACHE_NAME: str = "offline_number_cache.json"

_DIGITS: tuple[str, ...] = tuple("0123456789")


@dataclass(frozen=True, slots=True)
class DigitComponent:
    """连通域数字候选（躯干区放大图上的像素坐标）。"""

    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        """连通域中心 X 坐标（自举对齐按此升序）。"""
        return self.x + self.w / 2


@dataclass(frozen=True, slots=True)
class DigitTemplate:
    """单数字模板（二值图，数字为白前景）；匹配前会归一化高度。"""

    digit: str
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class DigitLibrary:
    """模板库：仅含启用数字（样本 ≥MIN_SAMPLES_PER_DIGIT）的全部样本。"""

    templates: tuple[DigitTemplate, ...]

    @property
    def enabled_digits(self) -> tuple[str, ...]:
        """已启用数字（升序去重）。"""
        return tuple(sorted({t.digit for t in self.templates}))


def torso_region(img: np.ndarray) -> np.ndarray:
    """裁躯干上部：垂直 10%~45%、水平中 70%（spec 写死）。

    Args:
        img: 裁图（灰度或 BGR，H×W）。

    Returns:
        躯干区子图；图太小导致区域退化时返回空数组（调用方判 size==0 跳过）。
    """
    h, w = img.shape[:2]
    y1: int = round(h * TORSO_V_START)
    y2: int = round(h * TORSO_V_END)
    side: float = (1.0 - TORSO_H_CENTER_FRAC) / 2
    x1: int = round(w * side)
    x2: int = round(w * (1.0 - side))
    if y2 - y1 < MIN_TORSO_SIDE_PX or x2 - x1 < MIN_TORSO_SIDE_PX:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


def _drop_nested(comps: list[DigitComponent]) -> list[DigitComponent]:
    """丢弃 bbox 被另一候选严格包含的连通域（闭环数字 0/6/8/9 的内沿伪影）。

    二值化闭环笔画会在环内沿产生嵌套伪连通域；真实背号数字并排不嵌套，
    被包含者必非独立数字。

    Args:
        comps: 已过滤的候选列表。

    Returns:
        去除嵌套后的列表（不改入参）。
    """

    def _inside(a: DigitComponent, b: DigitComponent) -> bool:
        """a 的 bbox 严格落在 b 内。"""
        return a.x > b.x and a.y > b.y and a.x + a.w < b.x + b.w and a.y + a.h < b.y + b.h

    return [c for c in comps if not any(_inside(c, other) for other in comps if other is not c)]


def segment_digits(binary: np.ndarray) -> list[DigitComponent]:
    """连通域切数字候选：面积/高度/宽高比过滤 + 嵌套丢弃，按中心 x 升序返回。

    Args:
        binary: 二值图（数字为白前景）。

    Returns:
        候选连通域列表（按 cx 升序）；无有效候选返回空列表。
    """
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    total_area: int = height * width
    out: list[DigitComponent] = []
    for i in range(1, n):  # 0 = 背景
        x, y, w, h, area = (int(v) for v in stats[i])
        frac: float = area / total_area
        if not (DIGIT_MIN_AREA_FRAC <= frac <= DIGIT_MAX_AREA_FRAC):
            continue
        if h < DIGIT_MIN_HEIGHT_FRAC * height:
            continue
        if w / max(h, 1) > DIGIT_MAX_ASPECT:
            continue
        out.append(DigitComponent(x=x, y=y, w=w, h=h, area=area))
    out = _drop_nested(out)
    out.sort(key=lambda c: c.cx)
    return out


def binarize(gray: np.ndarray) -> np.ndarray:
    """自适应二值化：与局部高斯均值偏差 >ADAPTIVE_C 的像素为前景（白）。

    用"偏差"而不是 adaptiveThreshold 的单侧阈值：背号无论深色（白球衣）还是
    浅色（黑球衣）都是相对球衣底的异色块，偏差掩码对两种极性产出一致表示；
    平坦区域（均匀球衣底）天然全背景，不会出现 adaptiveThreshold 在平坦区
    mean-C < 像素值 导致整片翻成前景的退化。

    Args:
        gray: 灰度图。

    Returns:
        二值图（前景=255）。
    """
    mean: np.ndarray = cv2.GaussianBlur(gray, (ADAPTIVE_BLOCK_SIZE, ADAPTIVE_BLOCK_SIZE), 0)
    diff: np.ndarray = cv2.absdiff(gray, mean)
    return (diff > ADAPTIVE_C).astype(np.uint8) * 255


def preprocess_crop(img: np.ndarray) -> np.ndarray | None:
    """预处理（spec 写死）：躯干上部 → 3 倍放大 → 灰度 → 自适应二值化。

    Args:
        img: 裁图（灰度或 BGR）。

    Returns:
        二值躯干区图（数字为白前景）；图畸形（躯干区退化）返回 None。
    """
    region = torso_region(img)
    if region.size == 0:
        return None
    up: np.ndarray = cv2.resize(
        region, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )
    gray: np.ndarray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if up.ndim == 3 else up
    return binarize(gray)


def _component_img(binary: np.ndarray, comp: DigitComponent) -> np.ndarray:
    """抠出连通域子图（含 1px 边，避免 matchTemplate 边界退化）。"""
    x1: int = max(0, comp.x - 1)
    y1: int = max(0, comp.y - 1)
    x2: int = min(binary.shape[1], comp.x + comp.w + 1)
    y2: int = min(binary.shape[0], comp.y + comp.h + 1)
    return binary[y1:y2, x1:x2]


def _normalize_height(img: np.ndarray, height: int = CANON_HEIGHT) -> np.ndarray:
    """等比缩放到指定高度（宽度按比例），统一匹配尺度基准。"""
    h, w = img.shape[:2]
    new_w: int = max(1, round(w * (height / max(h, 1))))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_CUBIC)


def _paste_center(canvas: np.ndarray, img: np.ndarray) -> None:
    """把 img 居中贴入 canvas（canvas 各维 ≥ img 对应维，调用方保证）。"""
    y0: int = (canvas.shape[0] - img.shape[0]) // 2
    x0: int = (canvas.shape[1] - img.shape[1]) // 2
    canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img


def match_digit(region: np.ndarray, template: np.ndarray) -> float:
    """多尺度 matchTemplate（TM_CCOEFF_NORMED）取最优分。

    双方先归一化到 CANON_HEIGHT 并居中贴上同尺寸画布再匹配（不允许模板在
    查询图内任意滑窗——否则模板是查询图的子形状时也会满分，如 1⊂7）；
    横向仅留 ±MATCH_SLIDE_PX 容差吸收归一化取整误差。

    Args:
        region: 待识别连通域子图（二值，紧贴连通域）。
        template: 模板子图（二值，紧贴连通域）。

    Returns:
        最优匹配分（[-1, 1]；无法计算返回 -1.0）。
    """
    r: np.ndarray = _normalize_height(region)
    t0: np.ndarray = _normalize_height(template)
    best: float = -1.0
    for scale in MATCH_SCALES:
        th: int = max(1, round(t0.shape[0] * scale))
        tw: int = max(1, round(t0.shape[1] * scale))
        t: np.ndarray = cv2.resize(t0, (tw, th), interpolation=cv2.INTER_CUBIC)
        ch: int = max(r.shape[0], th)
        cw: int = max(r.shape[1], tw)
        img_canvas: np.ndarray = np.zeros((ch, cw + 2 * MATCH_SLIDE_PX), dtype=np.uint8)
        tpl_canvas: np.ndarray = np.zeros((ch, cw), dtype=np.uint8)
        _paste_center(img_canvas, r)
        _paste_center(tpl_canvas, t)
        res: np.ndarray = cv2.matchTemplate(img_canvas, tpl_canvas, cv2.TM_CCOEFF_NORMED)
        score: float = float(res.max())
        if math.isfinite(score):  # 恒定模板会导致 nan，按不匹配处理
            best = max(best, score)
    return best


def classify_digit(region: np.ndarray, library: DigitLibrary) -> tuple[str | None, float]:
    """连通域分类：宽高比闸 + 对库内全部模板取最高分数字；低于 MATCH_MIN_SCORE 不认。

    宽高比闸：候选与模板宽高比偏差 >MATCH_ASPECT_TOL 倍直接跳过该模板
    （数字的宽高比是强特征，1 与 7 形状有包含关系但宽高比差 4 倍）。

    Args:
        region: 待识别连通域子图（二值）。
        library: 模板库。

    Returns:
        (数字, 得分)；空库、全部模板被宽高比闸拦截或最高分不足返回 (None, 最高分)。
    """
    r_aspect: float = region.shape[1] / max(region.shape[0], 1)
    best_digit: str | None = None
    best: float = -1.0
    for t in library.templates:
        t_aspect: float = t.image.shape[1] / max(t.image.shape[0], 1)
        if max(r_aspect, t_aspect) / max(min(r_aspect, t_aspect), 1e-6) > MATCH_ASPECT_TOL:
            continue
        score: float = match_digit(region, t.image)
        if score > best:
            best = score
            best_digit = t.digit
    if best_digit is None or best < MATCH_MIN_SCORE:
        return None, best
    return best_digit, best


def read_crop_number(img: np.ndarray, library: DigitLibrary) -> NumberGuess | None:
    """单张裁图离线读号：预处理 → 切数字 → 逐数字模板匹配。

    Args:
        img: 裁图（灰度或 BGR）。
        library: 模板库；空库直接返回 None。

    Returns:
        NumberGuess（number 可能为 None：切出连通域但全部匹配失败，confidence=low）；
        None = 无法切分（无有效连通域 / 连通域数 >MAX_DIGITS / 图畸形 / 空库），
        调用方记 INFO 跳过、不写缓存。
    """
    if not library.templates:
        return None
    binary: np.ndarray | None = preprocess_crop(img)
    if binary is None:
        return None
    comps: list[DigitComponent] = segment_digits(binary)
    if not comps or len(comps) > MAX_DIGITS:
        return None
    digits: list[str] = []
    scores: list[float] = []
    for comp in comps:
        digit, score = classify_digit(_component_img(binary, comp), library)
        if digit is None:
            return NumberGuess(number=None, color=None, name_text=None, confidence="low")
        digits.append(digit)
        scores.append(score)
    confidence: str = "high" if min(scores) >= HIGH_CONF_SCORE else "low"
    return NumberGuess(number="".join(digits), color=None, name_text=None, confidence=confidence)


def _load_gray(path: Path) -> np.ndarray | None:
    """PIL 读图为灰度 ndarray（Windows 中文路径安全）；失败记 INFO 返回 None。"""
    try:
        with Image.open(path) as im:
            return np.array(im.convert("L"))
    except (OSError, ValueError) as exc:
        logger.info("裁图读取失败，跳过: %s (%s)", path, exc)
        return None


def build_md5_index(entries: list[dict[str, Any]], cropsdir: Path) -> dict[str, Path]:
    """md5 → 裁图路径 索引（自举按 K3 缓存键反查裁图文件用）。

    Args:
        entries: scorer_candidates 候选记录。
        cropsdir: 裁图目录（candidates 同目录）。

    Returns:
        裁图 md5 → 路径；缺失裁图记 INFO 不入索引。
    """
    index: dict[str, Path] = {}
    for e in entries:
        for name in _entry_crops(e):
            path: Path = cropsdir / name
            if not path.is_file():
                logger.info("索引裁图缺失，跳过: %s (%s)", path, e.get("key"))
                continue
            index.setdefault(file_md5(path), path)
    return index


def bootstrap_library(
    entries: list[dict[str, Any]],
    cropsdir: Path,
    k3_cache: dict[str, dict[str, Any]],
) -> DigitLibrary:
    """模板库自举（spec 写死）：K3 高置信条目的裁图跑同套切分建模板。

    对齐规则：连通域数 == len(number) 才接收样本，按连通域中心 x 升序与数字
    字符一一 zip；个数不符（粘连/断裂）→ 丢弃该样本记 INFO；K3 缓存键找不到
    裁图 → 跳过记 INFO。每数字 ≥MIN_SAMPLES_PER_DIGIT 个样本才启用，不足不认。

    Args:
        entries: scorer_candidates 候选记录（反查裁图文件用）。
        cropsdir: 裁图目录。
        k3_cache: load_number_cache 产物（只读，绝不回写）。

    Returns:
        模板库；无达标数字时 templates 为空（读号全部返回 None）。
    """
    samples: dict[str, list[np.ndarray]] = {d: [] for d in _DIGITS}
    if not k3_cache:
        return DigitLibrary(templates=())
    md5_index: dict[str, Path] = build_md5_index(entries, cropsdir)
    n_used: int = 0
    for key, raw in k3_cache.items():
        guess: NumberGuess | None = number_guess_from_dict(raw)
        if guess is None or guess.confidence != "high" or guess.number is None:
            continue
        path: Path | None = md5_index.get(key)
        if path is None:
            logger.info("模板自举: K3 条目 %s 在当前裁图中找不到对应文件，跳过", key)
            continue
        img: np.ndarray | None = _load_gray(path)
        if img is None:
            continue
        binary: np.ndarray | None = preprocess_crop(img)
        if binary is None:
            logger.info("模板自举: %s 图畸形无法预处理，丢弃样本", path.name)
            continue
        comps: list[DigitComponent] = segment_digits(binary)
        if len(comps) != len(guess.number):
            logger.info(
                "模板自举: %s 连通域数 %d ≠ 号码 %s 位数 %d，丢弃样本（粘连/断裂）",
                path.name,
                len(comps),
                guess.number,
                len(guess.number),
            )
            continue
        for comp, digit in zip(comps, guess.number, strict=True):
            samples[digit].append(_component_img(binary, comp))
        n_used += 1
    templates: list[DigitTemplate] = []
    for digit in _DIGITS:
        imgs: list[np.ndarray] = samples[digit]
        if not imgs:
            continue
        if len(imgs) < MIN_SAMPLES_PER_DIGIT:
            logger.info(
                "模板自举: 数字 %s 样本 %d < %d，不启用该数字",
                digit,
                len(imgs),
                MIN_SAMPLES_PER_DIGIT,
            )
            continue
        templates.extend(DigitTemplate(digit=digit, image=img) for img in imgs)
    library: DigitLibrary = DigitLibrary(templates=tuple(templates))
    logger.info(
        "模板自举完成: 接收样本 %d 张，启用数字 %s",
        n_used,
        ",".join(library.enabled_digits) or "无",
    )
    return library


def load_offline_cache(path: Path) -> dict[str, dict[str, Any]]:
    """读 offline_number_cache.json；缺失/损坏/来源或版本不符 → 空（重识别）。

    Args:
        path: offline_number_cache.json 路径。

    Returns:
        裁图 md5 → 缓存条目（number/color/name_text/confidence/source/ts）。
    """
    if not path.exists():
        return {}
    payload: Any = read_json(path, what=OFFLINE_CACHE_NAME)
    if not isinstance(payload, dict):
        logger.warning("%s 结构异常，重新开始", OFFLINE_CACHE_NAME)
        return {}
    meta: Any = payload.get("_meta")
    if isinstance(meta, dict) and (
        meta.get("source") != OFFLINE_CACHE_SOURCE or meta.get("version") != OFFLINE_CACHE_VERSION
    ):
        logger.warning("离线号码缓存来源/版本不符，旧缓存作废重开")
        return {}
    results: Any = payload.get("results")
    if not isinstance(results, dict):
        return {}
    return {str(k): v for k, v in results.items() if isinstance(v, dict)}


def save_offline_cache(
    path: Path,
    results: dict[str, dict[str, Any]],
    votes: dict[str, dict[str, Any] | None],
) -> None:
    """原子写 offline_number_cache.json（_meta 标 source=offline；votes 供对照实跑）。

    Args:
        path: 目标路径。
        results: 裁图 md5 → 缓存条目。
        votes: 进球 key → 投票结果（asdict(NumberGuess) 或 None）。

    Raises:
        OSError: IO 重试耗尽（由 atomic_write_json 抛出）。
    """
    payload: dict[str, Any] = {
        "_meta": {
            "source": OFFLINE_CACHE_SOURCE,
            "version": OFFLINE_CACHE_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "results": results,
        "votes": votes,
    }
    atomic_write_json(path, payload, what=OFFLINE_CACHE_NAME)


def merged_number_caches(
    k3_cache: dict[str, dict[str, Any]],
    offline_cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """跳票模式并集入口（spec 写死）：K3 缓存 ∪ offline 缓存，同键 K3 优先。

    仅供将来接入 crop_scorers 跳票模式使用（Phase 2 实跑达标后再接）；
    本脚本自身不消费。

    Args:
        k3_cache: load_number_cache 产物。
        offline_cache: load_offline_cache 产物。

    Returns:
        合并缓存（不改入参，返回新 dict）。
    """
    merged: dict[str, dict[str, Any]] = dict(offline_cache)
    merged.update(k3_cache)
    return merged


def apply_offline_reading(
    entries: list[dict[str, Any]],
    cropsdir: Path,
    library: DigitLibrary,
    cache_path: Path,
) -> dict[str, dict[str, Any] | None]:
    """离线读号主流程：对 OK 球的 crops 逐张离线读号 + vote_number_guess 众数投票。

    candidates 只读不改（与 propagate 同哲学；offline 不碰 K3 的 number_guess 链）。
    逐张口径：裁图缺失/读取失败/无法切分 → 记 INFO 跳过不炸批（不写缓存）；
    缓存命中直接用（幂等，重跑零重算）；新读结果带 source=offline 落
    offline_number_cache.json。

    Args:
        entries: load_scorer_candidates 产物。
        cropsdir: 裁图目录（candidates 同目录）。
        library: 模板库（空库 → 所有裁图无法切分，全部跳过）。
        cache_path: offline_number_cache.json 路径。

    Returns:
        进球 key → 投票结果（asdict(NumberGuess) 或 None）。

    Raises:
        BasketballPipelineError: 缓存条目无法归一（数据损坏，防御性显式失败）。
    """
    targets: list[dict[str, Any]] = [
        e for e in entries if e["status"] == STATUS_OK and _entry_crops(e)
    ]
    cache: dict[str, dict[str, Any]] = load_offline_cache(cache_path)
    votes: dict[str, dict[str, Any] | None] = {}
    n_fresh: int = 0
    n_cache_hit: int = 0
    for e in targets:
        guesses: list[NumberGuess] = []
        for name in _entry_crops(e):
            path: Path = cropsdir / name
            if not path.is_file():
                logger.info("离线读号裁图缺失，跳过: %s (%s)", path, e["key"])
                continue
            md5: str = file_md5(path)
            cached: dict[str, Any] | None = cache.get(md5)
            if cached is not None:
                guess: NumberGuess | None = number_guess_from_dict(cached)
                if guess is None:  # 防御：load_offline_cache 已过滤非 dict，不应触发
                    raise BasketballPipelineError(f"离线号码缓存条目无法归一: md5:{md5}")
                n_cache_hit += 1
                guesses.append(guess)
                continue
            img: np.ndarray | None = _load_gray(path)
            if img is None:
                continue
            fresh: NumberGuess | None = read_crop_number(img, library)
            if fresh is None:
                logger.info("离线读号无法切分，跳过: %s (%s)", name, e["key"])
                continue
            n_fresh += 1
            cache[md5] = {
                **asdict(fresh),
                "source": OFFLINE_CACHE_SOURCE,
                "ts": datetime.now(UTC).isoformat(),
            }
            guesses.append(fresh)
        voted: NumberGuess | None = vote_number_guess(guesses)
        votes[str(e["key"])] = asdict(voted) if voted is not None else None
        logger.info("离线读号投票: %s → %s", e["key"], votes[str(e["key"])])
    save_offline_cache(cache_path, cache, votes)
    logger.info(
        "离线读号完成: %d 球（新读 %d 张，缓存命中 %d 张）", len(targets), n_fresh, n_cache_hit
    )
    return votes


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="离线读号：传统 CV 模板匹配读裁图背号（试验性质，零模型零 token）"
    )
    parser.add_argument(
        "--candidates", required=True, type=Path, help="scorer_candidates.json 路径"
    )
    parser.add_argument(
        "--bootstrap-from",
        type=Path,
        default=None,
        help="K3 number_cache.json（自举模板库；缺省/缺失 = 空库全部跳过）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"输出缓存路径（缺省 = candidates 同目录 {OFFLINE_CACHE_NAME}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功，非 0=失败）。"""
    args: argparse.Namespace = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        entries: list[dict[str, Any]] = load_scorer_candidates(args.candidates)
        cropsdir: Path = args.candidates.parent
        k3_cache: dict[str, dict[str, Any]] = (
            load_number_cache(args.bootstrap_from) if args.bootstrap_from is not None else {}
        )
        library: DigitLibrary = bootstrap_library(entries, cropsdir, k3_cache)
        if not library.templates:
            logger.warning("模板库为空（未自举或无达标数字），所有裁图将跳过")
        out: Path = args.out if args.out is not None else cropsdir / OFFLINE_CACHE_NAME
        votes: dict[str, dict[str, Any] | None] = apply_offline_reading(
            entries, cropsdir, library, out
        )
        n_hit: int = sum(1 for v in votes.values() if v is not None and v.get("number"))
        logger.info("离线读号产出: %d 球 %d 命中 → %s", len(votes), n_hit, out)
    except BasketballPipelineError as e:
        logger.error("离线读号失败: %s", e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
