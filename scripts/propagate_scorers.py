"""轨迹传播：同文件人框轨迹串联进球，一处认人传播一片（spec: docs/scorer-propagate/spec.md）。

输入：scorer_candidates.json（crop_scorers 产物，含 seed_frame/seed_box/seed_team）、
    work/detect/<fid>_mot_cache.json（5fps 人框缓存）、work/frames/<fid>/f_NNNNN.jpg
    （颜色守卫采样用）、--roster（可选，--evaluate 标定报表用）。
输出：<scorer_candidates.json 同目录>/track_links.json——
    {version:"track-v1", per_file:{fid:{tracks:[{track_id,keys,mixed,span:[f0,f1]}],
    unlinked:[keys]}}}；candidates 文件只读不改，落盘幂等（同输入同输出）。
依赖：scripts/crop_scorers.py（load_mot_cache/team_of_box/number_guess_from_dict/
    _frame_path 与颜色/状态常量）、scripts/geom.py（Box/iou）、scripts/roster.py
    （validate_roster）、scripts/gen_scorer_page.py（match_players_by_number，
    号码→tag 映射与页面同口径）、scripts/pipe_common.py（read_json/atomic_write_json/
    run_id 日志）。
典型调用：
    python scripts/propagate_scorers.py \
        --candidates work/20260722/scorers/scorer_candidates.json \
        --detectdir work/detect --framesdir work/frames
    # 标定（roster confirmed 后）：
    python scripts/propagate_scorers.py ... --roster work/20260722/roster.json --evaluate

规则写死（spec §人轨迹，勿自行变更）：
- 文件内贪心多目标跟踪：逐帧遍历 persons，活跃轨迹按最近框 IoU≥TRACK_MIN_IOU
  吸收检测框；无匹配开新轨；超过 MAX_GAP_FRAMES（默认 10 帧=2s）无更新即封存；
  轨迹不跨文件。
- 唯一性：每帧每框至多入一轨；同帧多轨迹竞争同一框时按"轨迹最后更新帧更近 →
  IoU 更大 → track_id 更小"裁决；每条轨迹每帧至多吸收一框（多框命中取 IoU 最大者）。
- 颜色守卫：进球映射后沿挂球轨迹（keys 非空；无球轨迹的 mixed 是死数据，跳过
  不采样）每 COLOR_SAMPLE_EVERY 帧采样 team_of_box，采样点按帧分组、同帧多轨迹
  只解码一次；"便服"样本不计入一致也不计入不一致；黑/白样本主色占比
  < MIXED_MAJORITY_MIN → mixed=true；mixed 轨迹只展示、不参与自动预填；
  帧图不可读记 WARNING 跳过该帧采样点。
- 进球→轨迹映射：entry 的 seed_frame/seed_box 对该帧全部轨迹框找 IoU 最大
  ≥GOAL_MAP_MIN_IOU 者归该轨；归不上 / seed 字段缺失（WARNING）→ 进 unlinked。
- 传播预填来源：同文件同 track_id 球中"号码预填唯一命中（number_guess.number
  非空 + conf=high + 名单唯一匹配）或 roster 归属"的 tag；被预填球自身不算源；
  多源冲突 → 不预填、note="conflict"；NOGOAL 哨兵永远不作源。
- --evaluate（须配合 --roster）双指标分开报：roster-seeded（逐轨迹留时间最早的
  有归属球作种子，其余有归属球被预填，源排除自身）+ number-seeded（源只用号码
  高置信球，roster 仅作真值，模拟无归属冷启动）；准确率 = 预填正确 ÷ 收到预填，
  覆盖倍数 = 收到预填 ÷ 种子数；冲突/mixed/unlinked 一律计入覆盖失败一侧。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image

from crop_scorers import (
    STATUS_OK,
    STATUS_SKIP,
    TEAM_BLACK,
    TEAM_WHITE,
    MotCache,
    _frame_path,
    load_mot_cache,
    number_guess_from_dict,
    team_of_box,
)
from errors import BasketballPipelineError, SchemaError
from gen_scorer_page import match_players_by_number
from geom import Box, iou
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import Player, Roster, fid_of, validate_roster

logger = logging.getLogger(__name__)

# ---- 跟踪参数（spec §人轨迹写死；初值拍脑袋，新场次标定后写死，spec §Open Questions） ----
TRACK_MIN_IOU: float = 0.3  # 帧间人框 IoU 下限（与 crop_scorers.TRACE_MIN_IOU 同值）
MAX_GAP_FRAMES: int = 10  # 轨迹封存阈值：连续无更新帧数上限（10 帧=2s @5fps）
GOAL_MAP_MIN_IOU: float = 0.3  # 进球 seed_box → 轨迹框映射 IoU 下限

# ---- 颜色守卫参数（spec §人轨迹写死） ----
COLOR_SAMPLE_EVERY: int = 5  # 封存轨迹采样间隔（帧）
MIXED_MAJORITY_MIN: float = 0.6  # 黑/白样本主色占比下限，低于 → mixed=true

TRACK_LINKS_VERSION: str = "track-v1"  # track_links.json 契约版本
NOGOAL_TAG: str = "不算进球"  # NOGOAL 哨兵（与 gen_scorer_page JS 的 NOGOAL 同值），永不作源


@dataclass(slots=True)
class PersonTrack:
    """文件内一条人框轨迹（贪心 IoU 链产物；跟踪过程原地生长，故非 frozen）。

    不变式：points 按 frame_idx 升序、同帧至多一点（唯一性规则保证）；
    last_frame == points[-1][0]。
    """

    track_id: int
    points: list[tuple[int, Box]]  # (frame_idx, box) 升序
    last_frame: int
    mixed: bool = False  # 颜色守卫：黑/白样本主色占比 < MIXED_MAJORITY_MIN
    keys: list[str] = field(default_factory=list)  # 归到本轨的进球 key（entry 序）

    @property
    def span(self) -> tuple[int, int]:
        """轨迹覆盖帧区间 [首帧, 末帧]。"""
        return self.points[0][0], self.points[-1][0]


@dataclass(frozen=True, slots=True)
class TrackMember:
    """轨迹上一颗球的评估视图（--evaluate 用）。

    truth = roster 归属真值 tag（无 roster / 球无归属 → None）；
    number_tag = 号码高置信唯一命中映射出的 tag（无命中 → None）。
    """

    key: str
    anchor_time: float
    truth: str | None
    number_tag: str | None


@dataclass(frozen=True, slots=True)
class SeedReport:
    """单口径标定报表（roster-seeded / number-seeded 之一）。"""

    mode: str  # "roster_seeded" | "number_seeded"
    seeds: int  # 种子球数
    prefilled: int  # 收到预填的球数
    correct: int  # 预填正确数

    @property
    def accuracy(self) -> float | None:
        """准确率 = 预填正确 ÷ 收到预填；无预填返回 None（无意义，不硬算 0）。"""
        return self.correct / self.prefilled if self.prefilled else None

    @property
    def coverage(self) -> float | None:
        """覆盖倍数 = 收到预填 ÷ 种子数；无种子返回 None。"""
        return self.prefilled / self.seeds if self.seeds else None


@dataclass(slots=True)
class FileResult:
    """单文件传播结果：轨迹（keys 已挂）+ 未归上的进球 key。"""

    fid: str
    tracks: list[PersonTrack]
    unlinked: list[str]


def load_scorer_candidates(path: Path) -> list[dict[str, Any]]:
    """读取并校验 scorer_candidates.json（注意：非 MOT 的 candidates.json）。

    顶层结构损坏 → SchemaError（rules.md §0.2 数据损坏必须停）；entry 的 seed
    字段此处不校验（缺失/畸形由 entry_seed 在映射阶段判 WARNING 进 unlinked，
    不炸整批，spec §crop_scorers 小改）。

    Args:
        path: scorer_candidates.json 路径。

    Returns:
        候选记录列表（保留原始 dict，不补默认字段）。

    Raises:
        SchemaError: 顶层非对象 / candidates 非列表 / 条目缺 key/file/anchor_time/
            status 或类型错 / status 值不在 (OK, SKIP)。
    """
    data: Any = read_json(path, what="scorer_candidates.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    cands: Any = data.get("candidates")
    if not isinstance(cands, list):
        raise SchemaError(f"{path}: 缺 candidates 列表或类型错误")
    entries: list[dict[str, Any]] = []
    for i, raw in enumerate(cands):
        if not isinstance(raw, dict):
            raise SchemaError(f"{path}: 第{i}条候选不是对象")
        if not isinstance(raw.get("key"), str) or not raw["key"]:
            raise SchemaError(f"{path}: 第{i}条候选 key 缺失或不是非空 str")
        if not isinstance(raw.get("file"), str) or not raw["file"]:
            raise SchemaError(f"{path}: 第{i}条候选({raw['key']}) file 缺失或不是非空 str")
        if not isinstance(raw.get("anchor_time"), (int, float)):
            raise SchemaError(f"{path}: 第{i}条候选({raw['key']}) anchor_time 缺失或非数值")
        if raw.get("status") not in (STATUS_OK, STATUS_SKIP):
            raise SchemaError(
                f"{path}: 第{i}条候选({raw['key']}) status 非法: {raw.get('status')!r}"
                f"（仅允许 {STATUS_OK}/{STATUS_SKIP}）"
            )
        entries.append(raw)
    return entries


def entry_seed(entry: dict[str, Any]) -> tuple[int, Box] | None:
    """取 entry 的 (seed_frame, seed_box)；缺失/畸形 → None（调用方 WARNING 进 unlinked）。

    Args:
        entry: scorer_candidates 候选记录。

    Returns:
        (seed_frame, Box)；seed_frame 非 int / seed_box 非四数列表 / 框退化
        （Box 构造 ValueError）返回 None。
    """
    frame: Any = entry.get("seed_frame")
    box_raw: Any = entry.get("seed_box")
    if not isinstance(frame, int) or isinstance(frame, bool):
        return None
    if not (isinstance(box_raw, list) and len(box_raw) == 4):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box_raw):
        return None
    try:
        box = Box(int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3]))
    except ValueError:
        return None
    return frame, box


def track_persons(
    persons: tuple[tuple[Box, ...], ...],
    *,
    max_gap: int = MAX_GAP_FRAMES,
    min_iou: float = TRACK_MIN_IOU,
) -> list[PersonTrack]:
    """文件内贪心多目标跟踪：逐帧吸收 persons 人框成轨迹（spec §人轨迹写死）。

    每帧流程：封存超 max_gap 无更新的轨迹（移出活跃集）→ 逐框在活跃轨迹中找
    IoU≥min_iou 的竞争者，按"轨迹最后更新帧更近 → IoU 更大 → track_id 更小"
    裁决归属（每框至多入一轨）→ 多条框选中同一轨迹时该轨迹只吸收 IoU 最大者
    （并列取框序号小者；落选的框不再二次分配，贪心规则从简）→ 未被吸收的框
    开新轨。轨迹不跨帧序跳跃以外的任何形式拼接。

    Args:
        persons: 按帧对齐的人框序列（load_mot_cache 产物）。
        max_gap: 轨迹封存阈值（连续无更新帧数上限）。
        min_iou: 吸收 IoU 下限。

    Returns:
        全部轨迹（含已封存），按开轨先后（track_id 升序）。

    Raises:
        ValueError: max_gap < 1 或 min_iou 越出 (0, 1]（参数错误显式失败）。
    """
    if max_gap < 1:
        raise ValueError(f"track_persons 要求 max_gap ≥ 1，实际 {max_gap}")
    if not 0.0 < min_iou <= 1.0:
        raise ValueError(f"track_persons 要求 min_iou ∈ (0, 1]，实际 {min_iou}")
    tracks: list[PersonTrack] = []
    next_id: int = 1
    for fi, boxes in enumerate(persons):
        active: list[PersonTrack] = [t for t in tracks if fi - t.last_frame <= max_gap]
        # 逐框裁决：竞争者按(最后更新帧降, IoU 降, track_id 升)取最优
        box_choice: dict[int, PersonTrack] = {}
        for bi, box in enumerate(boxes):
            cands: list[tuple[PersonTrack, float]] = [
                (t, iou(t.points[-1][1], box)) for t in active
            ]
            cands = [(t, s) for t, s in cands if s >= min_iou]
            if cands:
                box_choice[bi] = max(
                    cands, key=lambda ts: (ts[0].last_frame, ts[1], -ts[0].track_id)
                )[0]
        # 逐轨吸收：选中同一轨迹的多框取 IoU 最大者（并列取框序号小者）
        won: dict[int, tuple[int, float]] = {}  # track_id → (box_idx, iou)
        for bi, t in box_choice.items():
            score: float = iou(t.points[-1][1], boxes[bi])
            prev: tuple[int, float] | None = won.get(t.track_id)
            if prev is None or (score, -bi) > (prev[1], -prev[0]):
                won[t.track_id] = (bi, score)
        claimed: set[int] = set()
        for t in active:
            got: tuple[int, float] | None = won.get(t.track_id)
            if got is not None:
                t.points.append((fi, boxes[got[0]]))
                t.last_frame = fi
                claimed.add(got[0])
        for bi, box in enumerate(boxes):
            if bi not in claimed:
                tracks.append(PersonTrack(track_id=next_id, points=[(fi, box)], last_frame=fi))
                next_id += 1
    return tracks


def apply_color_guard(tracks: list[PersonTrack], framesdir: Path, fid: str) -> None:
    """颜色守卫（spec §人轨迹写死）：沿轨迹每 COLOR_SAMPLE_EVERY 帧采样 team_of_box。

    "便服"样本不计入一致也不计入不一致；黑/白样本中主色占比 < MIXED_MAJORITY_MIN
    → mixed=true（只展示、不参与自动预填）；无黑/白样本无法判定，mixed 维持 False。
    无球轨迹（keys 为空）的 mixed 是死数据，直接跳过不采样；采样点按帧分组，
    同帧多轨迹共用一次帧图解码（4K JPEG 解码贵，逐轨迹开图会把同帧解码 N 次）。
    帧图不可读记 WARNING 跳过该帧采样点（业务可预期，不炸整轨）。原地置 track.mixed。

    Args:
        tracks: track_persons + map_goals_to_tracks 产物（keys 已挂；含已封存轨迹）。
        framesdir: 帧图根目录。
        fid: 视频主名（帧路径映射用）。
    """
    by_frame: dict[int, list[tuple[PersonTrack, Box]]] = {}
    for t in tracks:
        if not t.keys:
            continue
        first: int = t.points[0][0]
        for fi, box in t.points:
            if (fi - first) % COLOR_SAMPLE_EVERY == 0:
                by_frame.setdefault(fi, []).append((t, box))
    black: dict[int, int] = {}  # track_id → 黑样本数
    white: dict[int, int] = {}  # track_id → 白样本数
    for fi in sorted(by_frame):
        path: Path = _frame_path(framesdir, fid, fi)
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
        except (OSError, ValueError) as exc:
            logger.warning("颜色守卫帧图不可读，跳过采样: %s (帧 %d): %s", path, fi, exc)
            continue
        for t, box in by_frame[fi]:
            team: str = team_of_box(rgb, box)
            if team == TEAM_BLACK:
                black[t.track_id] = black.get(t.track_id, 0) + 1
            elif team == TEAM_WHITE:
                white[t.track_id] = white.get(t.track_id, 0) + 1
    for t in tracks:
        b: int = black.get(t.track_id, 0)
        w: int = white.get(t.track_id, 0)
        total: int = b + w
        if total > 0 and max(b, w) / total < MIXED_MAJORITY_MIN:
            t.mixed = True
            logger.info(
                "颜色守卫: %s 轨迹#%d mixed（黑=%d 白=%d 样本=%d）",
                fid,
                t.track_id,
                b,
                w,
                total,
            )


def map_goals_to_tracks(
    tracks: list[PersonTrack], entries: list[dict[str, Any]], fid: str
) -> FileResult:
    """进球→轨迹映射：seed_frame 帧全部轨迹框中找与 seed_box IoU 最大
    ≥GOAL_MAP_MIN_IOU 者归该轨（spec §人轨迹写死）。

    归不上（该帧无轨迹框 / IoU 不达标）→ 进 unlinked；seed 字段缺失/畸形且
    status=OK 记 WARNING 进 unlinked（不炸整批）；SKIP 球按 crop_scorers 口径
    本就不落 seed 字段，直接进 unlinked 不重复告警。同帧多轨迹竞争取 IoU 最大，
    并列取 track_id 小者（稳定可测）。

    Args:
        tracks: track_persons 产物（原地挂 keys）。
        entries: 该 fid 的候选记录。
        fid: 视频主名（日志用）。

    Returns:
        FileResult（tracks 引用同一批对象，keys 已挂；unlinked 保持 entry 序）。
    """
    by_frame: dict[int, list[tuple[PersonTrack, Box]]] = {}
    for t in tracks:
        for fi, box in t.points:
            by_frame.setdefault(fi, []).append((t, box))
    unlinked: list[str] = []
    for e in entries:
        key: str = e["key"]
        seed: tuple[int, Box] | None = entry_seed(e)
        if seed is None:
            if e["status"] == STATUS_OK:
                logger.warning("进球缺 seed 字段，进 unlinked: %s (%s)", key, fid)
            unlinked.append(key)
            continue
        seed_frame, seed_box = seed
        best: PersonTrack | None = None
        best_score: float = 0.0
        for t, box in by_frame.get(seed_frame, []):
            score: float = iou(box, seed_box)
            if score < GOAL_MAP_MIN_IOU:
                continue
            if best is None or (score, -t.track_id) > (best_score, -best.track_id):
                best = t
                best_score = score
        if best is None:
            logger.info(
                "进球归不上轨迹（IoU<%.2f 或该帧无轨迹框）: %s (%s)", GOAL_MAP_MIN_IOU, key, fid
            )
            unlinked.append(key)
            continue
        best.keys.append(key)
    return FileResult(fid=fid, tracks=tracks, unlinked=unlinked)


def build_links_payload(per_file: dict[str, FileResult]) -> dict[str, Any]:
    """组装 track_links.json 载荷（契约版本 track-v1；幂等：排序稳定、同输入同输出）。

    只输出挂了进球的轨迹（无 key 轨迹对页面反查无意义）；tracks 按 track_id
    升序，keys 保持 entry 序，unlinked 按 key 排序，fid 按键排序。

    Args:
        per_file: fid → FileResult。

    Returns:
        track_links.json 载荷 dict。
    """
    files: dict[str, Any] = {}
    for fid in sorted(per_file):
        res: FileResult = per_file[fid]
        files[fid] = {
            "tracks": [
                {
                    "track_id": t.track_id,
                    "keys": list(t.keys),
                    "mixed": t.mixed,
                    "span": [t.span[0], t.span[1]],
                }
                for t in sorted(res.tracks, key=lambda tr: tr.track_id)
                if t.keys
            ],
            "unlinked": sorted(res.unlinked),
        }
    return {"version": TRACK_LINKS_VERSION, "per_file": files}


def number_prefill_tag(number_guess_raw: Any, players: list[Player]) -> str | None:  # noqa: ANN401
    """号码预填唯一命中 → tag（与页面 build_entries 同口径：名单唯一匹配才命中）。

    candidates 里只有 number_guess（prefill_tag 是页面层 build_entries 用名单算
    的），故传播源此处用同一匹配函数现算：number 非空 + confidence=high +
    match_players_by_number 唯一命中 → 该 Player.tag；其余（无号/低置信/歧义/
    名单为空）→ None。

    Args:
        number_guess_raw: entry 的 number_guess 原始 JSON（dict 或 None）。
        players: 球员名单（roster.players）。

    Returns:
        命中的 tag；无命中返回 None。
    """
    guess = number_guess_from_dict(number_guess_raw)
    if guess is None or not guess.number or guess.confidence != "high":
        return None
    matches: list[Player] = match_players_by_number(players, guess.number, guess.color)
    return matches[0].tag if len(matches) == 1 else None


def member_source_tags(m: TrackMember, *, use_roster: bool, use_number: bool) -> list[str]:
    """取单球可作传播源的 tag 列表；NOGOAL 哨兵永远剔除（spec §人轨迹写死）。

    Args:
        m: 轨迹成员评估视图。
        use_roster: roster 归属（truth）是否作源。
        use_number: 号码高置信命中（number_tag）是否作源。

    Returns:
        源 tag 列表（0/1/2 个；truth 与 number_tag 同值时可能重复，调用方集合化）。
    """
    tags: list[str] = []
    if use_roster and m.truth is not None:
        tags.append(m.truth)
    if use_number and m.number_tag is not None:
        tags.append(m.number_tag)
    return [t for t in tags if t != NOGOAL_TAG]


def track_prefill_tag(
    members: list[TrackMember],
    target_key: str,
    *,
    mixed: bool,
    use_roster: bool = True,
    use_number: bool = True,
) -> tuple[str | None, str]:
    """传播预填决策（spec §人轨迹写死）：同轨迹其他球的源 tag 一致才预填。

    规则：mixed 轨迹不参与自动预填；被预填球自身不算源；多个源冲突 → 不预填、
    note="conflict"；无源 → 不预填、note=""。

    Args:
        members: 同 track_id 的全部成员。
        target_key: 被预填球 key（自身源被排除）。
        mixed: 轨迹颜色守卫 mixed 标记。
        use_roster / use_number: 源口径开关（evaluate 双指标分别关闭一侧）。

    Returns:
        (预填 tag, note)；不预填时 tag=None，note ∈ {"", "mixed", "conflict"}。
    """
    if mixed:
        return None, "mixed"
    sources: set[str] = set()
    for m in members:
        if m.key == target_key:
            continue  # 被预填球自身不算源
        sources.update(member_source_tags(m, use_roster=use_roster, use_number=use_number))
    if not sources:
        return None, ""
    if len(sources) > 1:
        return None, "conflict"
    return next(iter(sources)), ""


def build_member(
    key: str,
    entry: dict[str, Any] | None,
    roster: Roster,
    players: list[Player],
) -> TrackMember:
    """组装单个 key 的评估视图：roster 归属为真值，号码高置信唯一命中为 number_tag。

    Args:
        key: 进球 key。
        entry: candidates 记录；None（防御：unlinked key 查不到记录）时 anchor=0、
            number_tag=None。
        roster: 校验后的 Roster（assignments 为真值来源）。
        players: 球员名单（号码映射用）。

    Returns:
        TrackMember。
    """
    truth: str | None = roster.assignments.get(key)
    if entry is None:
        logger.warning("evaluate: key 无 candidates 记录，号码源按无处理: %s", key)
        return TrackMember(key=key, anchor_time=0.0, truth=truth, number_tag=None)
    return TrackMember(
        key=key,
        anchor_time=float(entry["anchor_time"]),
        truth=truth,
        number_tag=number_prefill_tag(entry.get("number_guess"), players),
    )


def evaluate_roster_seeded(
    per_file: dict[str, FileResult], member_of: Callable[[str], TrackMember]
) -> SeedReport:
    """roster-seeded 主指标（spec §--evaluate 写死）：模拟"用户认一球 → 传播一片"。

    逐轨迹处理：轨迹内所有有 roster 归属的球中留时间最早 1 个作种子，其余有
    归属球作为被预填对象（源排除自身；模拟中其余球的真值对源不可见——只认过
    种子一球）。mixed 轨迹 / unlinked 球不计入预填：unlinked 有归属球各算 1 个
    种子、0 预填（每球都得人工认一次，计入覆盖失败一侧）。

    Args:
        per_file: fid → FileResult。
        member_of: key → TrackMember（build_member 闭包）。

    Returns:
        SeedReport（mode="roster_seeded"）。
    """
    seeds: int = 0
    prefilled: int = 0
    correct: int = 0
    for res in per_file.values():
        for t in res.tracks:
            if not t.keys:
                continue
            members: list[TrackMember] = sorted(
                (member_of(k) for k in t.keys), key=lambda m: m.anchor_time
            )
            with_truth: list[TrackMember] = [m for m in members if m.truth is not None]
            if not with_truth:
                continue
            seed: TrackMember = with_truth[0]
            seeds += 1
            for m in with_truth[1:]:
                # 模拟只认过种子：其余球真值遮蔽，号码源关闭（roster-seeded 口径）
                view: list[TrackMember] = [
                    replace(seed, number_tag=None),
                    replace(m, truth=None, number_tag=None),
                ]
                pred, _note = track_prefill_tag(
                    view, m.key, mixed=t.mixed, use_roster=True, use_number=False
                )
                if pred is None:
                    continue  # mixed → 覆盖失败一侧
                prefilled += 1
                correct += int(pred == m.truth)
        for key in res.unlinked:
            if member_of(key).truth is not None:
                seeds += 1  # unlinked 有归属球：1 种子 0 预填（覆盖失败一侧）
    return SeedReport(mode="roster_seeded", seeds=seeds, prefilled=prefilled, correct=correct)


def evaluate_number_seeded(
    per_file: dict[str, FileResult], member_of: Callable[[str], TrackMember]
) -> SeedReport:
    """number-seeded 副指标（spec §--evaluate 写死）：无 roster 冷启动场景。

    源只用号码高置信唯一命中的球（roster 归属不当源、仅作真值对照）；源球自身
    不作被预填对象；同轨迹多源冲突 / mixed → 不预填，计入覆盖失败一侧；unlinked
    的号码源球各算 1 个种子、0 预填（无轨迹可传播）。

    Args:
        per_file: fid → FileResult。
        member_of: key → TrackMember。

    Returns:
        SeedReport（mode="number_seeded"）。
    """
    seeds: int = 0
    prefilled: int = 0
    correct: int = 0
    for res in per_file.values():
        for t in res.tracks:
            if not t.keys:
                continue
            members: list[TrackMember] = [member_of(k) for k in t.keys]
            src_keys: set[str] = {
                m.key for m in members if m.number_tag is not None and m.number_tag != NOGOAL_TAG
            }
            seeds += len(src_keys)
            if not src_keys:
                continue
            # 冷启动：roster 归属对源不可见（use_roster=False）
            view: list[TrackMember] = [replace(m, truth=None) for m in members]
            for m in members:
                if m.truth is None or m.key in src_keys:
                    continue
                pred, _note = track_prefill_tag(
                    view, m.key, mixed=t.mixed, use_roster=False, use_number=True
                )
                if pred is None:
                    continue  # 冲突 / mixed → 覆盖失败一侧
                prefilled += 1
                correct += int(pred == m.truth)
        for key in res.unlinked:
            m: TrackMember = member_of(key)
            if m.number_tag is not None and m.number_tag != NOGOAL_TAG:
                seeds += 1  # unlinked 号码源球：1 种子 0 预填（覆盖失败一侧）
    return SeedReport(mode="number_seeded", seeds=seeds, prefilled=prefilled, correct=correct)


def evaluate_reports(
    per_file: dict[str, FileResult], entries: list[dict[str, Any]], roster: Roster
) -> dict[str, SeedReport]:
    """--evaluate 顶层：分别算 roster-seeded / number-seeded 双指标（不合并）。

    Args:
        per_file: fid → FileResult。
        entries: scorer_candidates 全部记录（建 key → entry 反查）。
        roster: 校验后的 Roster（真值 + 号码映射名单来源）。

    Returns:
        {"roster_seeded": SeedReport, "number_seeded": SeedReport}。
    """
    players: list[Player] = list(roster.players)
    by_key: dict[str, dict[str, Any]] = {e["key"]: e for e in entries}

    def member_of(key: str) -> TrackMember:
        return build_member(key, by_key.get(key), roster, players)

    return {
        "roster_seeded": evaluate_roster_seeded(per_file, member_of),
        "number_seeded": evaluate_number_seeded(per_file, member_of),
    }


def format_report(reports: dict[str, SeedReport]) -> str:
    """把双指标报表格式化为人类可读文本（面向用户的最终结果，print 输出）。

    Args:
        reports: evaluate_reports 产物。

    Returns:
        多行报表文本（准确率/覆盖倍数无分母时显示 "n/a"）。
    """

    def _pct(x: float | None) -> str:
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    def _ratio(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "n/a"

    lines: list[str] = ["轨迹传播标定报表（track-v1）"]
    for mode in ("roster_seeded", "number_seeded"):
        r: SeedReport = reports[mode]
        lines.append(
            f"[{mode}] 种子={r.seeds} 收到预填={r.prefilled} 预填正确={r.correct} "
            f"准确率={_pct(r.accuracy)} 覆盖倍数={_ratio(r.coverage)}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="轨迹传播：同文件人框轨迹串联进球，产 track_links.json"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="scorer_candidates.json（crop_scorers 产物，含 seed_* 字段；"
        "注意不是 MOT 的 candidates.json）",
    )
    parser.add_argument("--detectdir", required=True, type=Path, help="mot_cache 目录")
    parser.add_argument(
        "--framesdir", required=True, type=Path, help="帧图根目录（颜色守卫采样用）"
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=None,
        help="roster.json（可选；--evaluate 标定报表必需，作真值与号码映射名单）",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="输出标定报表（roster-seeded/number-seeded 双指标；须配合 --roster）",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=MAX_GAP_FRAMES,
        help="轨迹封存阈值：连续无更新帧数上限（默认 %(default)s 帧=2s@5fps）",
    )
    ns = parser.parse_args(argv)
    if ns.evaluate and ns.roster is None:
        parser.error("--evaluate 须配合 --roster（标定真值来源）")
    if ns.roster is not None and not ns.evaluate:
        parser.error("--roster 仅在 --evaluate 标定时有意义（与上条对称校验）")
    if ns.max_gap < 1:
        parser.error("--max-gap 须 ≥ 1")
    return ns


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功，1=管线失败或有 mot_cache 缺失）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        entries: list[dict[str, Any]] = load_scorer_candidates(args.candidates)
        logger.info("候选球 %d 个 ← %s", len(entries), args.candidates)
        by_fid: dict[str, list[dict[str, Any]]] = {}
        for e in entries:
            by_fid.setdefault(fid_of(e["file"]), []).append(e)

        per_file: dict[str, FileResult] = {}
        missing: int = 0
        for fid in sorted(by_fid):
            fentries: list[dict[str, Any]] = by_fid[fid]
            cache_path: Path = args.detectdir / f"{fid}_mot_cache.json"
            if not cache_path.is_file():
                logger.error("mot_cache 缺失，该文件进球全部进 unlinked: %s", cache_path)
                per_file[fid] = FileResult(
                    fid=fid, tracks=[], unlinked=[e["key"] for e in fentries]
                )
                missing += 1
                continue
            cache: MotCache = load_mot_cache(cache_path)
            tracks: list[PersonTrack] = track_persons(cache.persons, max_gap=args.max_gap)
            # 先映射后守卫：颜色守卫只采样挂球轨迹（两者独立——映射只写 keys，守卫只写 mixed）
            res: FileResult = map_goals_to_tracks(tracks, fentries, fid)
            apply_color_guard(tracks, args.framesdir, fid)
            logger.info(
                "%s: 轨迹=%d（挂球=%d mixed=%d）进球=%d 归上=%d unlinked=%d",
                fid,
                len(tracks),
                sum(1 for t in tracks if t.keys),
                sum(1 for t in tracks if t.mixed),
                len(fentries),
                len(fentries) - len(res.unlinked),
                len(res.unlinked),
            )
            per_file[fid] = res

        out_path: Path = args.candidates.parent / "track_links.json"
        atomic_write_json(out_path, build_links_payload(per_file), what="track_links.json")
        logger.info("track_links.json 落盘: %s", out_path)

        if args.evaluate:
            roster: Roster = validate_roster(
                read_json(args.roster, what="roster.json"), str(args.roster)
            )
            print(format_report(evaluate_reports(per_file, entries, roster)))  # noqa: T201 面向用户的最终结果

        if missing:
            logger.error("有 %d 个 fid 缺 mot_cache（详见上条 ERROR），退出码非零", missing)
            return 1
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
