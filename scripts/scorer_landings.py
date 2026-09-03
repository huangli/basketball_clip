"""批量落点实测器：热图阶段 0 · 新 Q2 判据（轨迹法定位成功率，docs/heatmap/spec.md v2）。

输入：work/<场次>/goals_batchK.json（confirmed 进球）、
    work/<场次>/candidates_batchK.json（候选锚点，run_session 产物）、
    work/detect/<fid>_mot_cache.json
输出：work/<场次>/scorer_landings.json（逐球落点 + 汇总）+ 控制台报告
依赖：crop_scorers（locate_scorer 筐锚定轨迹法全线复用，窗口/判据常量不重定义）、
    release_probe.load_goals_events（goals schema 校验复用）
典型调用：python scripts/scorer_landings.py --sessiondir work/20260801_对手名

口径（spec v2 写死）：
    出手时刻与持球人 = locate_scorer 的最后持球点（轨迹端点贴 candidates 锚点选
    进球轨迹 → 回放找最后球心在人框内的帧）；落点 = 该人框底边中点；
    start_fallback（起点回退，无持球语义）与 SKIP 均计未命中入分母（保守口径）；
    缺 mot_cache 记 WARNING 计分母；锚点匹配超差退化端点时间最近（认人同口径）。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crop_scorers import (
    SAMPLE_FPS,
    STATUS_OK,
    STATUS_SKIP,
    MotCache,
    load_candidates_index,
    load_mot_cache,
    locate_scorer,
    match_anchor_xy,
)
from errors import BasketballPipelineError, SchemaError
from geom import Box
from pipe_common import atomic_write_json, configure_logging, new_run_id
from release_probe import GoalEvent, load_goals_events

logger = logging.getLogger(__name__)

# ---- 判据常量（docs/heatmap/spec.md v2 写死）----
Q2_MIN_USABLE_RATIO: float = 0.90  # 新 Q2 过关线：真持球 OK 比例（≥110/122）


@dataclass(frozen=True, slots=True)
class LandingRecord:
    """单球落点记录；SKIP 时写死 frame_idx=-1、sec=-1.0、person_box/landing_px=None。

    Attributes:
        event_key: 事件主键 ``fid@anchor_time``（anchor_time 按 goals 原值序列化）。
        fid: 视频文件主键。
        anchor_time: 进球锚点（秒）。
        status: ``OK`` / ``SKIP``。
        reason: OK 时为 ``""``（真持球）或 ``start_fallback``（起点回退，无持球
            语义，不计入可用落点）；SKIP 时为 no_track/no_track_near_anchor/
            no_person/missing_cache。
        frame_idx: 最后持球帧索引（0-based；SKIP 为 -1）。
        sec: 最后持球时刻（秒，frame_idx / SAMPLE_FPS；SKIP 为 -1.0）。
        person_box: 持球人框（SKIP 为 None）。
        landing_px: 落点（人框底边中点像素；SKIP 为 None）。
    """

    event_key: str
    fid: str
    anchor_time: float
    status: str
    reason: str
    frame_idx: int
    sec: float
    person_box: Box | None
    landing_px: tuple[float, float] | None


def is_usable(rec: LandingRecord) -> bool:
    """是否可用落点：OK 且真持球（非 start_fallback）。"""
    return rec.status == STATUS_OK and rec.reason == ""


def land_event(
    cache: MotCache, event: GoalEvent, anchor_xy: tuple[int, int] | None
) -> LandingRecord:
    """单球落点：locate_scorer 轨迹法定位 → 人框底边中点。

    Args:
        cache: 校验后的 mot_cache。
        event: 进球事件。
        anchor_xy: 候选锚点 (cx, cy)；None 退化为端点时间最近选轨迹
            （match_anchor_xy 超差既有口径，与认人流程一致）。

    Returns:
        LandingRecord；locate SKIP 时按写死形态落盘（-1/-1.0/None）。

    Raises:
        BasketballPipelineError: locate 返回 OK 但 box 为空（逻辑错误显式失败）。
    """
    result = locate_scorer(cache, event.anchor_time, anchor_xy)
    if result.status == STATUS_SKIP:
        return LandingRecord(
            event_key=event.event_key,
            fid=event.fid,
            anchor_time=event.anchor_time,
            status=STATUS_SKIP,
            reason=result.reason,
            frame_idx=-1,
            sec=-1.0,
            person_box=None,
            landing_px=None,
        )
    if result.box is None:  # 防御：OK 必有 box（与 crop_scorers._process_goal 同口径）
        raise BasketballPipelineError(f"定位 OK 但 box 为空: {event.event_key}")
    box: Box = result.box
    return LandingRecord(
        event_key=event.event_key,
        fid=event.fid,
        anchor_time=event.anchor_time,
        status=STATUS_OK,
        reason=result.reason,
        frame_idx=result.frame_idx,
        sec=result.frame_idx / SAMPLE_FPS,
        person_box=box,
        landing_px=((box.x1 + box.x2) / 2.0, float(box.y2)),
    )


def load_merged_candidates(paths: list[Path]) -> dict[str, list[tuple[float, int, int]]]:
    """合并多批次 candidates 索引（fid → [(t0, cx, cy)]，跨批次同 fid 拼接）。

    Args:
        paths: candidates_batchK.json 路径列表。

    Returns:
        合并后的候选锚点索引。

    Raises:
        SchemaError: 任一文件 schema 损坏（load_candidates_index 抛出）。
    """
    merged: dict[str, list[tuple[float, int, int]]] = {}
    for p in paths:
        for fid, items in load_candidates_index(p).items():
            merged.setdefault(fid, []).extend(items)
    return merged


def land_session(session_dir: Path, detect_dir: Path, out_path: Path) -> dict[str, Any]:
    """全量落点实测：读 goals + candidates → 逐球轨迹法定位 → 汇总 → 落盘 + 报告。

    Args:
        session_dir: work/<场次> 目录（含 goals_batchK.json 与 candidates_batchK.json）。
        detect_dir: mot_cache 目录（work/detect）。
        out_path: 输出 JSON 路径（scorer_landings.json）。

    Returns:
        报告字典（summary / landings），已原子写入 out_path。

    Raises:
        BasketballPipelineError: 无 goals_batchK.json 或无 candidates_batchK.json
            （筐锚定是轨迹法选轨核心，缺候选必须显式停）。
        SchemaError: goals / candidates / mot_cache schema 损坏，或事件跨批次重复。
    """
    goals_files: list[Path] = sorted(session_dir.glob("goals_batch*.json"))
    if not goals_files:
        raise BasketballPipelineError(f"{session_dir} 下无 goals_batch*.json（先完成标注导出）")
    cand_files: list[Path] = sorted(session_dir.glob("candidates_batch*.json"))
    if not cand_files:
        raise BasketballPipelineError(f"{session_dir} 下无 candidates_batch*.json（先跑 score）")
    events: dict[str, GoalEvent] = {}
    for gf in goals_files:
        for e in load_goals_events(gf):
            if e.event_key in events:
                raise SchemaError(f"跨批次事件重复: {e.event_key}")
            events[e.event_key] = e
    cand_index: dict[str, list[tuple[float, int, int]]] = load_merged_candidates(cand_files)
    logger.info(
        "共 %d 个 confirmed 事件、%d 个候选锚点 fid（goals %d 批 / candidates %d 批）",
        len(events),
        len(cand_index),
        len(goals_files),
        len(cand_files),
    )

    caches: dict[str, MotCache] = {}
    records: list[LandingRecord] = []
    for e in events.values():
        cache_path: Path = detect_dir / f"{e.fid}_mot_cache.json"
        if not cache_path.exists():
            logger.warning("缺 mot_cache，计未命中: %s（%s）", e.event_key, cache_path.name)
            records.append(
                LandingRecord(
                    event_key=e.event_key,
                    fid=e.fid,
                    anchor_time=e.anchor_time,
                    status=STATUS_SKIP,
                    reason="missing_cache",
                    frame_idx=-1,
                    sec=-1.0,
                    person_box=None,
                    landing_px=None,
                )
            )
            continue
        if e.fid not in caches:
            caches[e.fid] = load_mot_cache(cache_path)
        anchor_xy: tuple[int, int] | None = match_anchor_xy(cand_index, e.fid, e.anchor_time)
        if anchor_xy is None:
            logger.warning("候选锚点未匹配，退化端点时间最近: %s", e.event_key)
        records.append(land_event(caches[e.fid], e, anchor_xy))

    total: int = len(records)
    usable: int = sum(1 for r in records if is_usable(r))
    fallback: int = sum(
        1 for r in records if r.status == STATUS_OK and r.reason == "start_fallback"
    )
    skip_by_reason: dict[str, int] = {}
    for r in records:
        if r.status == STATUS_SKIP:
            skip_by_reason[r.reason] = skip_by_reason.get(r.reason, 0) + 1
    ratio: float = usable / total if total else 0.0
    report: dict[str, Any] = {
        "session": session_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {"q2_min_usable_ratio": Q2_MIN_USABLE_RATIO},
        "summary": {
            "total": total,
            "usable": usable,
            "fallback": fallback,
            "usable_ratio": round(ratio, 4),
            "q2_pass": ratio >= Q2_MIN_USABLE_RATIO,
            "skip_by_reason": skip_by_reason,
        },
        "landings": [asdict(r) for r in records],
    }
    atomic_write_json(out_path, report, what="scorer_landings.json")

    logger.info("==== 新 Q2 轨迹法定位汇总 ====")
    logger.info(
        "定位成功率（真持球）: %d/%d = %.1f%%（阈值 %.0f%% 即 ≥%d 球）→ %s",
        usable,
        total,
        ratio * 100,
        Q2_MIN_USABLE_RATIO * 100,
        int(Q2_MIN_USABLE_RATIO * total + 0.9999),
        "过关" if report["summary"]["q2_pass"] else "未过关",
    )
    logger.info("起点回退（无持球语义，计未命中）: %d", fallback)
    logger.info("SKIP 原因分布: %s", skip_by_reason)
    logger.info("报告已落盘: %s", out_path)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="批量落点实测（热图阶段 0 · 新 Q2 判据）")
    parser.add_argument("--sessiondir", required=True, type=Path, help="work/<场次> 目录")
    parser.add_argument(
        "--detectdir", type=Path, default=None, help="mot_cache 目录（默认 <sessiondir>/../detect）"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="输出 JSON（默认 <sessiondir>/scorer_landings.json）"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=实测完成，非 0=管线失败；Q2 未过关不算失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    session_dir: Path = args.sessiondir
    detect_dir: Path = args.detectdir or session_dir.parent / "detect"
    out_path: Path = args.out or session_dir / "scorer_landings.json"
    try:
        land_session(session_dir, detect_dir, out_path)
    except BasketballPipelineError as e:
        logger.error("落点实测失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
