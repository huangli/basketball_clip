"""照片库认人：半截篮队员照片 1:N 识别预填（spec: docs/photo-roster/spec.md §数据契约）。

⚠️ 证伪不推荐（2026-08-27，docs/photo-roster/review03.md）：CLIP 整图 embedding
    对同款球衣不同人无身份判别力（跨人对相似度 0.936 超同人对下限、正确/错误
    号得分带完全重叠无阈值可切），v2.1 起认人级联改"免费信号→人裁"；本模块
    已随 photo-roster T12（video.py ②.5 换 L1 人脸 matcher，2026-08-28）退出
    people 链。仅 --evaluate 机制保留（T13 改造为级联评测器）。

输入：--photos 照片库目录（``photos/<号码>/*.jpg|jpeg|png``，号码 = 文件夹名，
    归一化 ``str(int(名))`` 去前导零）；--candidates 可重复（scorer_candidates.json，
    键集并集、同 key 后者覆盖，复用 cluster_scorers.merge_candidates）；
    --cache 可重复（各批 clip_cache.json，跨文件并集查询，键 = 裁图 md5 天然不冲）；
    --evaluate 模式另加 --roster 与 --goals（真值来源）。
输出：非 evaluate 模式 --out 写 photo_matches.json（仅含过闸命中球，
    ``{version, model, threshold, margin, matches: {key: {number, score, margin}}}``）；
    --evaluate 模式 --out 写 markdown 对照报告（全部入统球 top-1+score+margin
    分布（不过闸）+ 正样本命中率 + 负样本误命中率 + 按号码混淆矩阵）。
    照片 embedding 缓存落 ``<photos>/.photo_cache.json``（键 = model_tag:照片 md5，
    与 clip_cache 同格式，幂等增量、模型前缀隔离）。
依赖：numpy、scripts/cluster_scorers.py（缓存/编码器/合并复用件）、
    scripts/pipe_common.py、scripts/roster.py、scripts/build_highlight.py
    （KNOWN_STATUSES）、scripts/errors.py；编码后端固定 CLIP（cluster_scorers
    .build_clip_encoder，open_clip/torch 只在其内部 import，测试注入假 encoder）。
典型调用：
    python scripts/photo_match_scorers.py --photos photos \
        --candidates work/<场次>/scorers_b1/scorer_candidates.json \
        --cache work/<场次>/scorers_b1/clip_cache.json \
        --out work/<场次>/scorers_b1/photo_matches.json
    # 对照评估（报告写 --out 指定的 .md 路径）：
    python scripts/photo_match_scorers.py --photos photos --evaluate \
        --candidates ... --cache ... --roster ... --goals ... --out report.md

写死口径（spec §数据契约）：得分 = max(该球 crops × 该号码 photos) 余弦相似度
（向量均 L2 归一，余弦 = 点积）；两号码并列最高分不采纳记未匹配；库内仅 1 个
号码时 margin 视为 +inf（过闸只看 threshold）；``score >= THRESHOLD 且
margin >= MARGIN`` 才算命中。
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from build_highlight import KNOWN_STATUSES
from cluster_scorers import (
    MODEL_TAG,
    STATUS_OK,
    EncoderFactory,
    GoalCrops,
    build_clip_encoder,
    file_md5,
    l2_normalize,
    load_clip_cache,
    merge_candidates,
    save_clip_cache,
)
from errors import BasketballPipelineError, SchemaError
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import Roster, format_key, validate_roster

logger = logging.getLogger(__name__)

# ---- 采纳闸（spec §数据契约写死；占位值待 T4 Phase A 分布标定后改定，Ask first） ----
THRESHOLD: float = 0.30  # 占位待 T4 Phase A 标定（余弦相似度下限）
MARGIN: float = 0.02  # 占位待 T4 Phase A 标定（top-1 与次高分差下限）
# 达标线（spec §Objective，用户可改）：正样本命中率 ≥80% 且负样本误命中率 ≤10%
PASS_MIN_HIT_RATE: float = 0.80
PASS_MAX_MISS_RATE: float = 0.10

MATCH_VERSION: str = "photo-match-v1"  # photo_matches.json schema 版本，契约变更即升
PHOTO_CACHE_NAME: str = ".photo_cache.json"  # 照片 embedding 缓存（落 --photos 目录下）
PHOTO_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})  # 合法照片扩展名（小写）
MIN_PHOTOS_PER_NUMBER: int = 2  # 每号码正反各 1 张起步；不足记 WARNING 不阻塞

TEAM_OURS: str = "半截篮"  # 我方队名（真值映射写死，与 gen_scorer_page.TEAM_WHITE 同值）

# 真值类别（spec §评估口径）
TRUTH_OURS: str = "ours_numbered"  # 半截篮有号球（正样本）
TRUTH_NO_NUMBER: str = "no_number"  # 对方/便服 tag 的球（负样本，真值无号）
TRUTH_UNJUDGEABLE: str = "unjudgeable"  # 半截篮无号 tag 的球（不可判，不进分母）

_FIRST_DIGITS_RE: re.Pattern[str] = re.compile(r"\d+")  # tag 内首个数字串（真值取号用）
NO_TOP1_LABEL: str = "(无)"  # 混淆矩阵/报告里 top-1 缺失（并列不采纳或未出分）的展示值


@dataclass(frozen=True, slots=True)
class Gallery:
    """照片库扫描结果。photos 键 = 去零号码（匹配主键）；display_names 仅展示用。"""

    photos: dict[str, tuple[Path, ...]]  # 去零号码 → 照片路径（按文件名排序）
    display_names: dict[str, str]  # 去零号码 → 文件夹原名（不参与 join）


@dataclass(frozen=True, slots=True)
class TopScore:
    """一球的 top-1 得分。number=None 表示并列最高不采纳（spec 退化路径）。"""

    number: str | None
    score: float
    margin: float  # top-1 与次高分差；单号码库为 math.inf


@dataclass(frozen=True, slots=True)
class MatchEntry:
    """photo_matches.json 单条命中记录（schema 校验产物）。"""

    number: str
    score: float
    margin: float


@dataclass(frozen=True, slots=True)
class Truth:
    """一球的真值（--evaluate 用）。number 仅 TRUTH_OURS 类别非 None。"""

    category: str  # TRUTH_OURS / TRUTH_NO_NUMBER / TRUTH_UNJUDGEABLE
    number: str | None


@dataclass(frozen=True, slots=True)
class EvalRow:
    """一个入统球的评估行（top-1 不过闸全量列出，供 T4 阈值标定）。"""

    key: str
    category: str
    truth_number: str | None
    top_number: str | None  # None = 未出分（缓存缺失跳过）或并列不采纳
    score: float | None
    margin: float | None
    adopted: bool  # 是否过闸（score>=THRESHOLD 且 margin>=MARGIN 且非并列）
    correct: bool | None  # 仅正样本有意义：过闸且 top-1 == 真值号码


@dataclass(frozen=True, slots=True)
class EvalReport:
    """--evaluate 评估报告数据。比率为 None 表示分母为 0（待定）。"""

    rows: tuple[EvalRow, ...]
    n_ours: int  # 正样本数（半截篮有号）
    n_no_number: int  # 负样本数（真值无号）
    n_unjudgeable: int  # 不可判数（不进分母）
    ours_hit: int  # 正样本中过闸且正确的球数
    no_number_adopted: int  # 负样本中被过闸命中的球数（误命中）
    pos_hit_rate: float | None  # 正样本命中率 = ours_hit / n_ours
    neg_miss_rate: float | None  # 负样本误命中率 = no_number_adopted / n_no_number
    confusion: dict[str, dict[str, int]]  # 真值号码 → top-1 号码（不过闸）→ 球数


def normalize_number(dirname: str) -> str | None:
    """号码归一化：纯数字文件夹名 → ``str(int(名))`` 去前导零（spec 写死口径）。

    与 K3 读号归一（crop_scorers）及名单 tag 找号（gen_scorer_page）同口径。

    Args:
        dirname: 照片库子文件夹名（原始名，可带前导零）。

    Returns:
        去零号码（如 ``07`` → ``"7"``）；非纯数字名返回 None（调用方 WARNING 跳过）。
    """
    name: str = dirname.strip()
    if not name.isdigit():
        return None
    return str(int(name))


def scan_gallery(photos_dir: Path) -> Gallery:
    """扫描照片库 ``photos/<号码>/``，校验并产出 gallery（spec §数据契约）。

    校验口径：非数字文件夹名 / 空文件夹 / 无合法图片（.jpg/.jpeg/.png）→ WARNING
    跳过该条目；不足 MIN_PHOTOS_PER_NUMBER 张记 WARNING 不阻塞；全部无效 → 显式
    报错（整个库无效属配置错误，不静默空跑）；归一化后撞号（如 ``7`` 与 ``07``
    并存）→ 显式报错（归属歧义属配置错误，不静默合并）。顶层散文件忽略
    （.photo_cache.json 即落在顶层）。

    Args:
        photos_dir: 照片库根目录。

    Returns:
        Gallery（photos 键 = 去零号码，保目录扫描序）。

    Raises:
        BasketballPipelineError: 目录不存在/不是目录、全部条目无效、归一化撞号。
    """
    if not photos_dir.is_dir():
        raise BasketballPipelineError(f"照片库目录不存在: {photos_dir}")
    photos: dict[str, tuple[Path, ...]] = {}
    display_names: dict[str, str] = {}
    for child in sorted(photos_dir.iterdir()):
        if not child.is_dir():
            continue
        number: str | None = normalize_number(child.name)
        if number is None:
            logger.warning("照片库子文件夹名非纯数字，跳过: %s", child.name)
            continue
        images: tuple[Path, ...] = tuple(
            p for p in sorted(child.iterdir()) if p.is_file() and p.suffix.lower() in PHOTO_EXTS
        )
        if not images:
            logger.warning("照片库文件夹无合法图片(.jpg/.jpeg/.png)，跳过: %s", child.name)
            continue
        if number in photos:
            raise BasketballPipelineError(
                f"照片库号码归一化撞号: {display_names[number]!r} 与 {child.name!r} "
                f"同为 {number!r}（请合并为一个文件夹）"
            )
        if len(images) < MIN_PHOTOS_PER_NUMBER:
            logger.warning(
                "号码 %s 仅 %d 张照片（正反各 1 张起步），不阻塞但识别质量可能不足",
                number,
                len(images),
            )
        photos[number] = images
        display_names[number] = child.name
    if not photos:
        raise BasketballPipelineError(f"照片库全部条目无效（配置错误）: {photos_dir}")
    return Gallery(photos=photos, display_names=display_names)


def embed_gallery_photos(
    gallery: Gallery,
    encoder_factory: EncoderFactory,
    cache: dict[str, list[float]],
    model_tag: str,
) -> dict[str, tuple[np.ndarray, ...]]:
    """照片库 embedding：逐张编码（缓存优先，encoder 惰性构建，全命中零加载模型）。

    每张向量写入缓存前已 L2 归一化（与 cluster_scorers.embed_goal 同口径），
    缓存命中的向量视为已归一，直接使用。扫描后消失的照片 WARNING 跳过；
    某号码照片全部消失 → WARNING 跳过该号码。

    Args:
        gallery: scan_gallery 产物。
        encoder_factory: 零参工厂，首次缓存未命中时才调用（测试注入假 encoder）。
        cache: 照片缓存（原地写入新向量，键 = ``model_tag:照片 md5``）。
        model_tag: 模型标识（缓存键前缀，防后端切换互冲）。

    Returns:
        去零号码 → 该号码照片的单位向量元组。

    Raises:
        BasketballPipelineError: 全部号码无可用 embedding（照片文件扫描后集体消失）。
    """
    vectors: dict[str, tuple[np.ndarray, ...]] = {}
    encoder = None
    for number, paths in gallery.photos.items():
        vecs: list[np.ndarray] = []
        for path in paths:
            if not path.is_file():
                logger.warning("照片扫描后消失，跳过: %s (号码 %s)", path, number)
                continue
            cache_key: str = f"{model_tag}:{file_md5(path)}"
            cached: list[float] | None = cache.get(cache_key)
            if cached is not None:
                vecs.append(np.asarray(cached, dtype=np.float64))
                continue
            if encoder is None:
                encoder = encoder_factory()
            vec: np.ndarray = l2_normalize(np.asarray(encoder(path), dtype=np.float64))
            cache[cache_key] = [float(x) for x in vec]
            vecs.append(vec)
        if vecs:
            vectors[number] = tuple(vecs)
        else:
            logger.warning("号码 %s 无可用照片 embedding，跳过该号码", number)
    if not vectors:
        raise BasketballPipelineError("照片库全部号码无可用 embedding（照片文件缺失？）")
    return vectors


def load_crop_caches(paths: list[Path]) -> dict[str, list[float]]:
    """加载多个 clip_cache.json 并取并集（键 = ``model_tag:裁图 md5`` 天然不冲）。

    Args:
        paths: 各批次 clip_cache.json 路径列表。

    Returns:
        合并后的缓存（后者同键覆盖前者，理论上不会发生）。

    Raises:
        BasketballPipelineError: 任一缓存文件缺失（提示先跑 cluster，显式报错不静默）。
    """
    merged: dict[str, list[float]] = {}
    for path in paths:
        if not path.is_file():
            raise BasketballPipelineError(
                f"clip_cache 缺失: {path}（请先跑 cluster_scorers 聚类产出该批缓存）"
            )
        merged.update(load_clip_cache(path))
    return merged


def require_model_prefix(cache: dict[str, list[float]], model_tag: str) -> int:
    """检查缓存中本模型前缀条目数；0 条 → 显式报错（spec：防静默零产出）。

    Args:
        cache: 合并后的裁图缓存。
        model_tag: 当前模型标识。

    Returns:
        本模型前缀条目数。

    Raises:
        BasketballPipelineError: 前缀命中率 0%（提示 cluster 可能用了别的 --model 后端）。
    """
    prefix: str = f"{model_tag}:"
    n: int = sum(1 for k in cache if k.startswith(prefix))
    if n == 0:
        raise BasketballPipelineError(
            f"裁图缓存中本模型（{model_tag}）前缀条目为 0"
            "（cluster 是否用了别的 --model 后端？缓存键前缀不一致会静默零产出，显式拒绝）"
        )
    return n


def score_goal(
    crop_vecs: tuple[np.ndarray, ...] | list[np.ndarray],
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, float]:
    """球-号码得分：``score(球, 号码) = max(该球 crops × 该号码 photos)`` 余弦（spec 写死）。

    所有向量调用前已 L2 归一化，余弦相似度 = 点积。

    Args:
        crop_vecs: 该球各裁图单位向量。
        gallery_vecs: 去零号码 → 该号码照片单位向量元组。

    Returns:
        去零号码 → 得分。
    """
    return {
        number: max(float(np.dot(c, p)) for c in crop_vecs for p in photo_vecs)
        for number, photo_vecs in gallery_vecs.items()
    }


def top_number(scores: dict[str, float]) -> TopScore:
    """取 top-1 号码与次高分差（spec 退化路径写死）。

    两号码并列最高分 → number=None 不采纳（保守，交人裁判）；库内仅 1 个号码时
    margin = +inf（过闸只看 threshold）。并列判定用浮点精确相等（近似并列由
    MARGIN 闸兜底）。

    Args:
        scores: score_goal 产物（去零号码 → 得分），非空。

    Returns:
        TopScore。

    Raises:
        BasketballPipelineError: scores 为空（照片库为空，上游逻辑错误显式失败）。
    """
    if not scores:
        raise BasketballPipelineError("score 字典为空（照片库为空？上游逻辑错误）")
    best: float = max(scores.values())
    winners: list[str] = [n for n, s in scores.items() if s == best]
    if len(winners) > 1:
        return TopScore(number=None, score=best, margin=0.0)
    if len(scores) == 1:
        return TopScore(number=winners[0], score=best, margin=math.inf)
    second: float = max(s for n, s in scores.items() if n != winners[0])
    return TopScore(number=winners[0], score=best, margin=best - second)


def adopt(top: TopScore, threshold: float, margin: float) -> bool:
    """采纳闸（spec 写死）：``score >= THRESHOLD 且 margin >= MARGIN`` 且非并列。

    Args:
        top: top_number 产物。
        threshold: 得分下限（含等于）。
        margin: 分差下限（含等于；单号码库 +inf 恒过）。

    Returns:
        True = 命中可入 photo_matches.json。
    """
    return top.number is not None and top.score >= threshold and top.margin >= margin


def match_goals(
    goals: dict[str, GoalCrops],
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
    cache: dict[str, list[float]],
    model_tag: str,
) -> tuple[dict[str, TopScore], list[str]]:
    """匹配主链：逐球查缓存向量 → 得分 → top-1（不过闸，闸在 adopt/评估侧）。

    只对 status=OK 且有裁图的球出分（SKIP/无裁图跳过不记 WARNING，与聚类
    unclustered 口径一致）；裁图文件缺失或裁图 md5 不在缓存 → 该球 WARNING
    跳过不阻塞（记入返回的 skipped 列表）。

    Args:
        goals: merge_candidates 产物。
        gallery_vecs: embed_gallery_photos 产物。
        cache: load_crop_caches 合并缓存。
        model_tag: 模型标识（缓存键前缀）。

    Returns:
        (key → TopScore, 缓存缺失跳过的 key 列表)；两者键互斥。
    """
    results: dict[str, TopScore] = {}
    skipped: list[str] = []
    for key, goal in goals.items():
        if goal.status != STATUS_OK or not goal.crops:
            continue
        vecs: list[np.ndarray] = []
        missing: bool = False
        for name in goal.crops:
            path: Path = goal.base_dir / name
            if not path.is_file():
                logger.warning("裁图文件缺失，跳过该球: %s (%s)", path, key)
                missing = True
                break
            cache_key: str = f"{model_tag}:{file_md5(path)}"
            cached: list[float] | None = cache.get(cache_key)
            if cached is None:
                logger.warning("裁图 md5 不在缓存，跳过该球: %s (%s)", name, key)
                missing = True
                break
            vecs.append(np.asarray(cached, dtype=np.float64))
        if missing:
            skipped.append(key)
            continue
        results[key] = top_number(score_goal(vecs, gallery_vecs))
    return results, skipped


def build_matches_payload(
    results: dict[str, TopScore],
    threshold: float,
    margin: float,
    model_tag: str,
) -> dict[str, Any]:
    """组装 photo_matches.json 载荷：仅含过闸命中球（spec §数据契约写死字段）。

    注意：单号码库命中球的 margin 为 +inf，json.dump 落盘为 ``Infinity``
    （Python json 可回读；下游 JS 消费方需自行处理，见 spec T5）。

    Args:
        results: match_goals 产物（未过闸全量）。
        threshold: 本次采纳闸得分下限（落盘供标定追溯）。
        margin: 本次采纳闸分差下限。
        model_tag: 模型标识（落盘 model 字段）。

    Returns:
        可 JSON 序列化的载荷 dict（matches 按 results 键序）。
    """
    matches: dict[str, Any] = {}
    for key, top in results.items():
        if not adopt(top, threshold, margin):
            continue
        if top.number is None:  # 防御：adopt 已排除，逻辑错误显式失败
            raise BasketballPipelineError(f"过闸球缺 top-1 号码（逻辑错误）: {key}")
        matches[key] = {"number": top.number, "score": top.score, "margin": top.margin}
    return {
        "version": MATCH_VERSION,
        "model": model_tag,
        "threshold": threshold,
        "margin": margin,
        "matches": matches,
    }


def validate_matches_payload(data: Any, path: str) -> dict[str, MatchEntry]:  # noqa: ANN401
    """校验 photo_matches.json 结构（rules.md §0.2：schema 损坏显式失败）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        key → MatchEntry（仅含合法命中记录）。

    Raises:
        SchemaError: 顶层非对象 / version 不符 / model、threshold、margin 类型错 /
            matches 非对象 / 条目字段缺失或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    if data.get("version") != MATCH_VERSION:
        raise SchemaError(f"{path}: version 缺失或不符（期望 {MATCH_VERSION!r}）")
    if not isinstance(data.get("model"), str):
        raise SchemaError(f"{path}: model 不是 str")
    for field in ("threshold", "margin"):
        v: Any = data.get(field)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError(f"{path}: {field} 不是数值")
    matches_raw: Any = data.get("matches")
    if not isinstance(matches_raw, dict):
        raise SchemaError(f"{path}: matches 不是对象")
    matches: dict[str, MatchEntry] = {}
    for key, entry in matches_raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise SchemaError(f"{path}: matches 条目键/值类型错: {key!r}")
        number: Any = entry.get("number")
        if not isinstance(number, str) or not number:
            raise SchemaError(f"{path}: matches[{key!r}] number 缺失或不是非空 str")
        if not number.isdigit():
            # spec 写死"去零数字主键"；消费端（gen_scorer_page）拼占位 tag 内联
            # 进 <script>，非数字 number 可能是坏文件注入，显式拒绝
            raise SchemaError(
                f"{path}: matches[{key!r}] number 必须是去零数字主键，实际 {number!r}"
            )
        for field in ("score", "margin"):
            v = entry.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SchemaError(f"{path}: matches[{key!r}] {field} 不是数值")
        matches[key] = MatchEntry(
            number=number, score=float(entry["score"]), margin=float(entry["margin"])
        )
    return matches


def number_from_tag(tag: str) -> str | None:
    """真值取号（spec §评估口径写死）：tag 内首个数字串，去零。

    Args:
        tag: roster players 的 tag（如 ``半截篮07`` / ``白色中锋``）。

    Returns:
        去零号码；tag 内无数字串返回 None（无号 tag）。
    """
    m: re.Match[str] | None = _FIRST_DIGITS_RE.search(tag)
    if m is None:
        return None
    return str(int(m.group()))


def classify_truth(roster: Roster) -> dict[str, Truth]:
    """真值映射（spec §评估口径写死）：assignments 每球归类。

    team=半截篮 的 tag：取号成功 → TRUTH_OURS（正样本）；无号 → TRUTH_UNJUDGEABLE
    （不可判，不进分母——他可能是照片库成员，正确命中不该计误）。其余 team
    （对方/便服）→ TRUTH_NO_NUMBER（负样本，真值无号）。

    Args:
        roster: 校验后的 Roster。

    Returns:
        assignments 键 → Truth。

    Raises:
        SchemaError: assignments 值引用了 players 中不存在的 tag（数据不一致必须停）。
    """
    players: dict[str, str] = {p.tag: p.team for p in roster.players}
    truth: dict[str, Truth] = {}
    for key, tag in roster.assignments.items():
        team: str | None = players.get(tag)
        if team is None:
            raise SchemaError(f"roster.assignments[{key!r}] 引用了 players 中不存在的 tag: {tag!r}")
        if team != TEAM_OURS:
            truth[key] = Truth(category=TRUTH_NO_NUMBER, number=None)
            continue
        number: str | None = number_from_tag(tag)
        if number is None:
            truth[key] = Truth(category=TRUTH_UNJUDGEABLE, number=None)
        else:
            truth[key] = Truth(category=TRUTH_OURS, number=number)
    return truth


def load_confirmed_goal_keys(path: Path) -> set[str]:
    """读取 goals.json，返回 status=confirmed 球的 assignments 键集（入统真值来源）。

    候选可能是 goals 收缩前的陈旧产物，不能拿 candidates 键集代替 confirmed 判定
    （spec §数据契约）。未知 status 记 WARNING 跳过（与 build_highlight 同口径）；
    confirmed 记录缺 file/anchor_time → SchemaError。

    Args:
        path: goals.json 路径。

    Returns:
        confirmed 球的 ``format_key(file, anchor_time)`` 键集。

    Raises:
        SchemaError: 顶层非对象 / 缺 goals 列表 / 条目结构损坏 / confirmed 记录
            file 或 anchor_time 缺失或类型错。
    """
    data: Any = read_json(path, what="goals.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    goals: Any = data.get("goals")
    if not isinstance(goals, list):
        raise SchemaError(f"{path}: 缺 goals 列表或类型错误，实际 {type(goals).__name__}")
    keys: set[str] = set()
    for i, g in enumerate(goals):
        if not isinstance(g, dict):
            raise SchemaError(f"{path}: 第{i}条记录不是对象，实际 {type(g).__name__}")
        status: Any = g.get("status")
        if not isinstance(status, str):
            raise SchemaError(f"{path}: 第{i}条 status 必须是 str，实际 {type(status).__name__}")
        if status not in KNOWN_STATUSES:
            logger.warning("%s: 第%d条未知 status=%r（可能拼错），跳过", path, i, status)
            continue
        if status != "confirmed":
            continue
        file: Any = g.get("file")
        if not isinstance(file, str) or not file:
            raise SchemaError(f"{path}: 第{i}条(confirmed) file 缺失或不是非空 str")
        anchor: Any = g.get("anchor_time")
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
            raise SchemaError(
                f"{path}: 第{i}条(confirmed) anchor_time 缺失或不是数值，"
                f"实际 {type(anchor).__name__}"
            )
        keys.add(format_key(file, float(anchor)))
    return keys


def evaluate(
    results: dict[str, TopScore],
    truth: dict[str, Truth],
    scope_keys: list[str],
    threshold: float,
    margin: float,
) -> EvalReport:
    """对照评估（spec §评估口径写死）：top-1 不过闸全量入报告，指标看过闸。

    指标口径：正样本命中率 = 半截篮有号球中「过闸且 top-1 正确」的比例（分母 =
    半截篮有号球数）；负样本误命中率 = 真值无号球被过闸命中任意号码的比例
    （命中即错）；不可判球不进任何分母。混淆矩阵按号码展开，用 top-1（不过闸），
    top-1 缺失（并列/未出分）归入 NO_TOP1_LABEL 列。

    Args:
        results: match_goals 产物（未过闸全量；缓存缺失球不在内按未出分计）。
        truth: classify_truth 产物。
        scope_keys: 入统键（goals confirmed 且 key ∈ roster.assignments），调用方算好。
        threshold: 采纳闸得分下限。
        margin: 采纳闸分差下限。

    Returns:
        EvalReport（比率为 None 表示分母为 0）。
    """
    rows: list[EvalRow] = []
    confusion: dict[str, dict[str, int]] = {}
    n_ours = n_no_number = n_unjudgeable = ours_hit = no_number_adopted = 0
    for key in scope_keys:
        t: Truth = truth[key]
        top: TopScore | None = results.get(key)
        adopted: bool = top is not None and adopt(top, threshold, margin)
        correct: bool | None = None
        if t.category == TRUTH_OURS:
            n_ours += 1
            correct = bool(adopted and top is not None and top.number == t.number)
            if correct:
                ours_hit += 1
            col: str = (top.number if top is not None else None) or NO_TOP1_LABEL
            if t.number is not None:
                row_counts: dict[str, int] = confusion.setdefault(t.number, {})
                row_counts[col] = row_counts.get(col, 0) + 1
        elif t.category == TRUTH_NO_NUMBER:
            n_no_number += 1
            if adopted:
                no_number_adopted += 1
        else:
            n_unjudgeable += 1
        rows.append(
            EvalRow(
                key=key,
                category=t.category,
                truth_number=t.number,
                top_number=top.number if top is not None else None,
                score=top.score if top is not None else None,
                margin=top.margin if top is not None else None,
                adopted=adopted,
                correct=correct,
            )
        )
    return EvalReport(
        rows=tuple(rows),
        n_ours=n_ours,
        n_no_number=n_no_number,
        n_unjudgeable=n_unjudgeable,
        ours_hit=ours_hit,
        no_number_adopted=no_number_adopted,
        pos_hit_rate=(ours_hit / n_ours) if n_ours else None,
        neg_miss_rate=(no_number_adopted / n_no_number) if n_no_number else None,
        confusion=confusion,
    )


def _fmt_rate(value: float | None) -> str:
    """格式化比率：None（分母 0）→ ``待定``，否则百分比。"""
    if value is None:
        return "待定（分母 0）"
    return f"{value * 100:.1f}%"


def _fmt_float(value: float | None) -> str:
    """格式化得分/margin：None → ``-``，+inf → ``inf``，否则 4 位小数。"""
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


_CATEGORY_LABELS: dict[str, str] = {
    TRUTH_OURS: "半截篮有号",
    TRUTH_NO_NUMBER: "无号(对方/便服)",
    TRUTH_UNJUDGEABLE: "不可判(半截篮无号tag)",
}


def render_markdown(report: EvalReport, model_tag: str, threshold: float, margin: float) -> str:
    """渲染 --evaluate markdown 对照报告（写 --out；标定与混淆矩阵以此为准）。

    Args:
        report: evaluate 产物。
        model_tag: 模型标识。
        threshold: 本次采纳闸得分下限。
        margin: 本次采纳闸分差下限。

    Returns:
        markdown 文本（UTF-8，含逐球 top-1+score+margin 分布、双指标、混淆矩阵、
        达标线对照）。
    """
    lines: list[str] = [
        "# 照片库认人 Phase A 对照报告",
        "",
        f"- 模型: `{model_tag}`",
        f"- 采纳闸: score ≥ {threshold} 且 margin ≥ {margin}（占位待 T4 标定）",
        f"- 入统球: {len(report.rows)}（半截篮有号 {report.n_ours} / "
        f"无号 {report.n_no_number} / 不可判 {report.n_unjudgeable}）",
        "",
        "## 指标",
        "",
        f"- 正样本命中率: {report.ours_hit}/{report.n_ours} = "
        f"{_fmt_rate(report.pos_hit_rate)}（达标线 ≥ {PASS_MIN_HIT_RATE * 100:.0f}%）",
        f"- 负样本误命中率: {report.no_number_adopted}/{report.n_no_number} = "
        f"{_fmt_rate(report.neg_miss_rate)}（达标线 ≤ {PASS_MAX_MISS_RATE * 100:.0f}%）",
        f"- 不可判球 {report.n_unjudgeable} 个（不进正/负样本分母；建议给照片库成员"
        "的 tag 补号码以减少占比）",
        "",
        "## 混淆矩阵（真值号码 × top-1 号码，不过闸）",
        "",
    ]
    cols: list[str] = sorted({c for row in report.confusion.values() for c in row})
    lines.append("| 真值\\top-1 | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for truth_number in sorted(report.confusion):
        row = report.confusion[truth_number]
        lines.append(f"| {truth_number} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    lines += [
        "",
        "## 逐球分布（top-1 不过闸，供阈值标定）",
        "",
        "| key | 真值类别 | 真值号码 | top-1 | score | margin | 过闸 | 正确 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        lines.append(
            f"| {r.key} | {_CATEGORY_LABELS.get(r.category, r.category)} | "
            f"{r.truth_number or '-'} | {r.top_number or NO_TOP1_LABEL} | "
            f"{_fmt_float(r.score)} | {_fmt_float(r.margin)} | "
            f"{'是' if r.adopted else '否'} | "
            f"{'-' if r.correct is None else ('对' if r.correct else '错')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数（--evaluate 必须同时给 --roster 与 --goals，缺一报错）。"""
    parser = argparse.ArgumentParser(
        description="照片库认人：半截篮队员照片 1:N 识别预填（CLIP 后端，零新依赖）"
    )
    parser.add_argument("--photos", required=True, type=Path, help="照片库目录（photos/<号码>/）")
    parser.add_argument(
        "--candidates",
        required=True,
        action="append",
        type=Path,
        help="scorer_candidates.json 路径（可重复；键集并集，同 key 后者覆盖前者）",
    )
    parser.add_argument(
        "--cache",
        required=True,
        action="append",
        type=Path,
        help="clip_cache.json 路径（可重复，跨文件并集查询；缺失显式报错）",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="输出路径：非 evaluate 写 photo_matches.json；--evaluate 写 markdown 报告",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="对照评估模式（需同时给 --roster 与 --goals；报告写 --out）",
    )
    parser.add_argument("--roster", type=Path, default=None, help="roster.json（--evaluate 用）")
    parser.add_argument("--goals", type=Path, default=None, help="goals.json（--evaluate 用）")
    ns = parser.parse_args(argv)
    if ns.evaluate and (ns.roster is None or ns.goals is None):
        parser.error("--evaluate 需同时给 --roster 与 --goals")
    if not ns.evaluate and (ns.roster is not None or ns.goals is not None):
        parser.error("--roster/--goals 仅 --evaluate 模式使用")
    return ns


def _run_evaluate(args: argparse.Namespace, results: dict[str, TopScore]) -> None:
    """--evaluate 分支：读真值 → 评估 → markdown 报告写 --out。

    Args:
        args: CLI 参数（roster/goals/out 已校验非 None）。
        results: match_goals 产物。
    """
    roster_data: Any = read_json(args.roster, what="roster.json")
    roster: Roster = validate_roster(roster_data, str(args.roster))
    confirmed: set[str] = load_confirmed_goal_keys(args.goals)
    truth: dict[str, Truth] = classify_truth(roster)
    scope: list[str] = sorted(k for k in confirmed if k in truth)
    logger.info(
        "入统 %d 球（goals confirmed %d 键，roster assignments %d 键）",
        len(scope),
        len(confirmed),
        len(truth),
    )
    report: EvalReport = evaluate(results, truth, scope, THRESHOLD, MARGIN)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report, MODEL_TAG, THRESHOLD, MARGIN), encoding="utf-8")
    logger.info(
        "评估完成: 正样本命中率 %s（%d/%d），负样本误命中率 %s（%d/%d），不可判 %d → %s",
        _fmt_rate(report.pos_hit_rate),
        report.ours_hit,
        report.n_ours,
        _fmt_rate(report.neg_miss_rate),
        report.no_number_adopted,
        report.n_no_number,
        report.n_unjudgeable,
        args.out,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=管线失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        gallery: Gallery = scan_gallery(args.photos)
        logger.info(
            "照片库: %d 个号码 / %d 张照片 ← %s",
            len(gallery.photos),
            sum(len(p) for p in gallery.photos.values()),
            args.photos,
        )
        photo_cache_path: Path = args.photos / PHOTO_CACHE_NAME
        photo_cache: dict[str, list[float]] = load_clip_cache(photo_cache_path)
        gallery_vecs: dict[str, tuple[np.ndarray, ...]] = embed_gallery_photos(
            gallery, build_clip_encoder, photo_cache, MODEL_TAG
        )
        save_clip_cache(photo_cache_path, MODEL_TAG, photo_cache)
        logger.info("照片 embedding 完成（缓存共 %d 条）", len(photo_cache))

        goals: dict[str, GoalCrops] = merge_candidates(args.candidates)
        crop_cache: dict[str, list[float]] = load_crop_caches(args.cache)
        n_prefix: int = require_model_prefix(crop_cache, MODEL_TAG)
        logger.info(
            "合并 %d 个 candidates（%d 球）/ %d 个 cache（%d 条，本模型前缀 %d 条）",
            len(args.candidates),
            len(goals),
            len(args.cache),
            len(crop_cache),
            n_prefix,
        )
        results, skipped = match_goals(goals, gallery_vecs, crop_cache, MODEL_TAG)
        if skipped:
            logger.warning("共 %d 球因裁图缓存缺失跳过（不阻塞）", len(skipped))

        if args.evaluate:
            _run_evaluate(args, results)
        else:
            payload: dict[str, Any] = build_matches_payload(results, THRESHOLD, MARGIN, MODEL_TAG)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.out, payload, what="photo_matches.json")
            logger.info(
                "匹配完成: 出分 %d 球 / 过闸命中 %d 球 → %s",
                len(results),
                len(payload["matches"]),
                args.out,
            )
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
