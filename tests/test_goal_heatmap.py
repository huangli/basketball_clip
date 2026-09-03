"""goal_heatmap.py 单元测试（热图 v4：v3 框人纠偏——出手前窗口 + 队色硬守卫）。

覆盖：hoops schema 校验、hoop_xy_at（覆盖/多覆盖/零覆盖退化/无 detected）、
flip_threshold 两端切分、to_rel_m 坐标换算（含翻转）、find_landing 两路并集
（主路命中/串人守卫走兜底/链断兜底/两路皆无/no_track）、v4 新增（持球点
窗口截断——入网后点不链种子/截断空轨迹直接 no_landing、队色硬守卫——主路
与兜底相反剔除/便服放行/expect_color 空禁用退化）、heat_session 端到端
（合成 roster+goals+candidates+hoops+cache+帧图，含便服剔除与覆盖统计）、
界外过滤已知输入、zone_of 归区/build_zones 几何、渲染 smoke（暗场/分区）、
目击拼图确定性与缺帧占位。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from crop_scorers import MotCache
from errors import BasketballPipelineError, SchemaError
from geom import Box
from goal_heatmap import (
    HeatLanding,
    HoopEvent,
    build_audit_grid,
    build_zones,
    court_template_lines,
    filter_in_court,
    find_landing,
    flip_threshold,
    heat_session,
    hoop_xy_at,
    load_hoops,
    render_team_heatmap,
    render_team_heatmap_zones,
    to_rel_m,
    zone_of,
)
from mot_candidates import Detection
from release_probe import GoalEvent

N_FRAMES: int = 30
ANCHOR: float = 4.0  # 锚点帧 idx 20；出手目标帧 = (4.0−1.0)×5 = 15
FID: str = "fid1"


def _ball(conf: float, cx: int, cy: int, fi: int) -> Detection:
    """构造球检测（box 取球心 ±10px；sec 按 5fps 换算）。"""
    return Detection(
        conf=conf,
        box=[cx - 10, cy - 10, cx + 10, cy + 10],
        cx=cx,
        cy=cy,
        sec=fi / 5,
        frame_idx=fi,
    )


def _cache(balls: list[tuple[Detection, ...]], persons: list[tuple[Box, ...]]) -> MotCache:
    """由逐帧列表构造 MotCache。"""
    assert len(balls) == len(persons)
    return MotCache(frames=len(balls), balls=tuple(balls), persons=tuple(persons))


def _event(anchor: float = ANCHOR, fid: str = FID) -> GoalEvent:
    """构造事件。"""
    return GoalEvent(fid=fid, anchor_time=anchor, event_key=f"{fid}@{anchor}")


def _shot_cache(balls_end: int = 21, persons: list[tuple[Box, ...]] | None = None) -> MotCache:
    """进球轨迹帧 10..balls_end-1 从 (50,50) 每帧 +20px；默认全程人框 A(0,0,100,100)。"""
    if persons is None:
        persons = [(Box(0, 0, 100, 100),)] * N_FRAMES
    balls: list[tuple[Detection, ...]] = [()] * N_FRAMES
    for fi in range(10, balls_end):
        balls[fi] = (_ball(0.9, 50 + (fi - 10) * 20, 50, fi),)
    return _cache(balls, persons)


def _flight_cache(extra_net_det: bool = False) -> MotCache:
    """飞行段缓存（docs/heatmap-flight-link/）：帧 10..20 球从 (50,50) 每帧 +100px。

    100px 超 run_mot 默认 80px 门限（旧行为碎成单点 → no_landing），落点链路
    放宽门限 FLIGHT_MATCH_DIST_PX=250 下链成一条。全程人框 A(0,0,100,100)
    （持球种子 = 帧 10）。extra_net_det=True 时帧 20 追加网内碎片球 (1080,60)
    （锁 prefer_longer 接线用：碎片端点距锚更近，长轨迹在 100px 滑窗内）。
    """
    persons = [(Box(0, 0, 100, 100),)] * N_FRAMES
    balls: list[tuple[Detection, ...]] = [()] * N_FRAMES
    for fi in range(10, 21):
        dets: list[Detection] = [_ball(0.9, 50 + (fi - 10) * 100, 50, fi)]
        if extra_net_det and fi == 20:
            dets.append(_ball(0.9, 1080, 60, fi))
        balls[fi] = tuple(dets)
    return _cache(balls, persons)


def _frame(tmp_path: Path, fid: str, frame_idx: int, box: Box | None, color: str) -> Path:
    """写合成帧图（box 区域填纯色，其余白色），返回帧图根目录。"""
    img = Image.new("RGB", (300, 200), "white")
    if box is not None:
        img.paste(color, (box.x1, box.y1, min(box.x2, 300), min(box.y2, 200)))
    d = tmp_path / "frames" / fid
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / f"f_{frame_idx + 1:05d}.jpg", "JPEG")
    return tmp_path / "frames"


# ---------- load_hoops ----------


def _hoop_payload(fid: str = FID, detected: bool = True) -> dict:
    """构造单个 hoops 事件载荷。"""
    return {
        "session": "s",
        "params": {},
        "events": [
            {
                "key": f"{fid}#e1",
                "fid": fid,
                "event_idx": 1,
                "window": [3.0, 5.0],
                "anchor": [1000, 400],
                "detected": detected,
                "track": [[3.8, 990, 400, "det"], [4.0, 1000, 402, "det"], [4.2, 1010, 405, "det"]],
            }
        ],
    }


def test_load_hoops_ok(tmp_path: Path) -> None:
    # Arrange
    p = tmp_path / "hoops_batch1.json"
    p.write_text(json.dumps(_hoop_payload()), encoding="utf-8")
    # Act
    events = load_hoops([p])
    # Assert
    assert len(events) == 1
    assert events[0].anchor == (1000, 400)
    assert events[0].track[1] == (4.0, 1000.0, 402.0)


def test_load_hoops_schema_error(tmp_path: Path) -> None:
    # Arrange：顶层缺 events / window 非法
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps({"x": 1}), encoding="utf-8")
    bad = _hoop_payload()
    bad["events"][0]["window"] = [1.0]
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps(bad), encoding="utf-8")
    # Act & Assert
    with pytest.raises(SchemaError):
        load_hoops([p1])
    with pytest.raises(SchemaError):
        load_hoops([p2])


# ---------- hoop_xy_at ----------


def _hoop(fid: str = FID, window: tuple = (3.0, 5.0), anchor: tuple = (1000, 400)) -> HoopEvent:
    """构造 HoopEvent。"""
    return HoopEvent(
        fid=fid,
        window=window,
        anchor=anchor,
        detected=True,
        track=((3.8, 990.0, 400.0), (4.0, 1000.0, 402.0), (4.2, 1010.0, 405.0)),
    )


def test_hoop_xy_at_in_window() -> None:
    # Act：锚点 4.0 在 window (3,5) 内 → 取时刻最近采样 (4.0)→(1000,402)
    got = hoop_xy_at([_hoop()], FID, 4.0, None)
    # Assert
    assert got == ((1000.0, 402.0), False)


def test_hoop_xy_at_multi_picks_closest_track() -> None:
    # Arrange：两个事件 window 都覆盖锚点；e1 最近采样 dt=0.2，e2 采样 dt=0 → e2 赢
    e1 = HoopEvent(
        fid=FID,
        window=(3.0, 5.0),
        anchor=(1000, 400),
        detected=True,
        track=((3.8, 990.0, 400.0), (4.2, 1010.0, 405.0)),
    )
    e2 = HoopEvent(
        fid=FID,
        window=(3.5, 4.5),
        anchor=(500, 300),
        detected=True,
        track=((4.0, 700.0, 300.0),),
    )
    # Act
    got = hoop_xy_at([e1, e2], FID, 4.0, (1000, 400))
    # Assert
    assert got == ((700.0, 300.0), False)


def test_hoop_xy_at_zero_coverage_degrades() -> None:
    # Arrange：锚点 9.0 越出 window+容差
    got = hoop_xy_at([_hoop()], FID, 9.0, None)
    # Assert：退化取时刻最近采样 (4.2)→(1010,405)，degraded=True
    assert got == ((1010.0, 405.0), True)


def test_hoop_xy_at_no_detected_returns_none() -> None:
    # Arrange：只有未 detected 事件 / 别的 fid
    ev = HoopEvent(
        fid=FID, window=(3.0, 5.0), anchor=(1, 1), detected=False, track=((4.0, 1.0, 1.0),)
    )
    # Act & Assert
    assert hoop_xy_at([ev], FID, 4.0, None) is None
    assert hoop_xy_at([_hoop()], "other", 4.0, None) is None


# ---------- flip_threshold / to_rel_m ----------


def test_flip_threshold() -> None:
    assert flip_threshold([100.0, 500.0, 1500.0]) == 500.0
    assert flip_threshold([100.0, 1500.0]) == 800.0
    assert flip_threshold([]) is None


def test_to_rel_m_with_flip() -> None:
    # Arrange：落点 (1100, 500)，筐心 (1000, 400)，框高 175px → 100px/m
    # Act
    rel = to_rel_m((1100.0, 500.0), (1000.0, 400.0), 175, flipped=False)
    rel_f = to_rel_m((1100.0, 500.0), (1000.0, 400.0), 175, flipped=True)
    # Assert：dx=+1m dy=+1m；翻转只镜像一个轴（dy 不变——场边机位纵深一致）
    assert rel == pytest.approx((1.0, 1.0))
    assert rel_f == pytest.approx((-1.0, 1.0))


# ---------- find_landing ----------


def test_find_landing_trace_hit(tmp_path: Path) -> None:
    # Arrange：持球到帧 12，人框 A 全程静止（IoU=1 链活），目标帧 15
    cache = _shot_cache()
    # Act（无帧图 → 守卫 WARNING 归便服不剔除）
    path, reason, fi, box = find_landing(cache, _event(), (250, 50), tmp_path / "frames")
    # Assert：主路命中，落点帧 15，框 A
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_flight_segment_rescued(tmp_path: Path) -> None:
    # Arrange：飞行段每帧 100px，默认 80px 门限下碎成单点（旧行为 no_landing）
    cache = _flight_cache()
    # Act：anchor_xy 指向轨迹末端 (1050,50)
    path, reason, fi, box = find_landing(cache, _event(), (1050, 50), tmp_path / "frames")
    # Assert：放宽门限链成长轨迹 → 主路命中（持球种子帧 10，落点帧 15，框 A）
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_prefers_flight_over_net_fragment(tmp_path: Path) -> None:
    # Arrange：网内碎片 (1080,60) 端点距锚更近，长飞行轨迹在 100px 滑窗内
    cache = _flight_cache(extra_net_det=True)
    # Act：anchor_xy 指向碎片（漏传 prefer_longer 则选碎片 → no_landing，本例锁接线）
    path, reason, fi, box = find_landing(cache, _event(), (1080, 60), tmp_path / "frames")
    # Assert：prefer_longer 选长飞行轨迹 → 主路命中
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_anchor_xy_misleading_falls_back(tmp_path: Path) -> None:
    # Arrange：anchor_xy 远离一切轨迹端点（>200px → anchor_xy 分支落空；
    # 对应实场"手工锚远离机器候选/候选空间错位"）
    cache = _flight_cache()
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (1900, 900), tmp_path / "frames")
    # Assert：回退时间域长轨选择 → 主路命中
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_guard_falls_to_track_start(tmp_path: Path) -> None:
    # Arrange：种子帧（13）框区填黑、目标帧（16）框区填白 → 黑↔白相反触发守卫
    cache = _shot_cache()
    frames = _frame(tmp_path, FID, 12, Box(0, 0, 100, 100), "black")
    _frame(tmp_path, FID, 15, Box(0, 0, 100, 100), "white")
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (250, 50), frames)
    # Assert：主路被守卫拦下 → 兜底（轨迹起点帧 10，dt=2.0s ≥0.8）
    assert (path, reason) == ("track_start", "")
    assert fi == 10 and box == Box(0, 0, 100, 100)


def test_find_landing_track_start_fallback() -> None:
    # Arrange：帧 13 起无人框 → 链只含种子帧（12），目标 15 越 ±0.3s → 兜底
    persons: list[tuple[Box, ...]] = [()] * N_FRAMES
    for fi in range(13):
        persons[fi] = (Box(0, 0, 100, 100),)
    cache = _shot_cache(persons=persons)
    # Act
    path, _reason, fi, _box = find_landing(cache, _event(), (250, 50), Path("nonexist"))
    # Assert：兜底起点帧 10
    assert path == "track_start" and fi == 10


def test_find_landing_no_landing() -> None:
    # Arrange：轨迹起点帧 18（dt=0.4s <0.8）且全程无人框 → 两路皆无
    persons: list[tuple[Box, ...]] = [()] * N_FRAMES
    balls: list[tuple[Detection, ...]] = [()] * N_FRAMES
    for fi in range(18, 21):
        balls[fi] = (_ball(0.9, 50 + (fi - 18) * 20, 50, fi),)
    cache = _cache(balls, persons)
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (130, 50), Path("nonexist"))
    # Assert
    assert path == "" and reason == "no_landing" and fi == -1 and box is None


def test_find_landing_no_track() -> None:
    # Arrange：窗口全无球
    cache = _cache([()] * N_FRAMES, [()] * N_FRAMES)
    # Act & Assert
    assert find_landing(cache, _event(), (250, 50), Path("x"))[1] == "no_track"


# ---------- v4：持球点窗口截断 + 队色硬守卫 ----------


def test_find_landing_held_window_excludes_post_net(tmp_path: Path) -> None:
    # Arrange：轨迹延到入网后；B 框（筐下人，195≤x≤285 只接 fi≥18 的球）
    # 在 fi 13..22 全程可链——v3 会种子链 B 并 trace 到帧 15，v4 截断后
    # 窗口内最后持球点仍在 A（fi=12，x=90），必须取 A
    persons: list[tuple[Box, ...]] = []
    for fi in range(N_FRAMES):
        boxes = [Box(0, 0, 100, 100)]
        if 13 <= fi <= 22:
            boxes.append(Box(195, 20, 285, 140))
        persons.append(tuple(boxes))
    cache = _shot_cache(balls_end=25, persons=persons)
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (250, 50), tmp_path / "frames")
    # Assert：入网后轨迹点不参与持球判定，主路仍取投篮者 A
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_truncated_empty_track_no_landing(tmp_path: Path) -> None:
    # Arrange：轨迹起点 fi=18（sec 3.6 > anchor−0.5）但全程有人框 A——
    # v3 能持球链 A 并 trace 命中，v4 截断为空 → 无种子；兜底 0.4s<0.8 → 双无
    persons: list[tuple[Box, ...]] = [(Box(0, 0, 100, 100),)] * N_FRAMES
    balls: list[tuple[Detection, ...]] = [()] * N_FRAMES
    for fi in range(18, 22):
        balls[fi] = (_ball(0.9, 50 + (fi - 18) * 20, 50, fi),)
    cache = _cache(balls, persons)
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (130, 50), tmp_path / "frames")
    # Assert
    assert (path, reason, fi, box) == ("", "no_landing", -1, None)


def test_find_landing_team_mismatch_rejected(tmp_path: Path) -> None:
    # Arrange：种子/目标帧框区都填黑（串人守卫不触发），期望白 → 硬守卫剔除
    cache = _shot_cache()
    frames = _frame(tmp_path, FID, 12, Box(0, 0, 100, 100), "black")
    _frame(tmp_path, FID, 15, Box(0, 0, 100, 100), "black")
    # Act
    got = find_landing(cache, _event(), (250, 50), frames, expect_color="白")
    # Assert
    assert got == ("", "team_mismatch", -1, None)


def test_find_landing_team_mismatch_fallback_rejected(tmp_path: Path) -> None:
    # Arrange：帧 13 起无人框 → 链断走兜底；起点帧（10）框区填黑、期望白
    persons: list[tuple[Box, ...]] = [()] * N_FRAMES
    for fi in range(13):
        persons[fi] = (Box(0, 0, 100, 100),)
    cache = _shot_cache(persons=persons)
    frames = _frame(tmp_path, FID, 10, Box(0, 0, 100, 100), "black")
    # Act：兜底路同样过硬守卫
    got = find_landing(cache, _event(), (250, 50), frames, expect_color="白")
    # Assert
    assert got == ("", "team_mismatch", -1, None)


def test_find_landing_team_guard_casual_passes(tmp_path: Path) -> None:
    # Arrange：种子帧黑、落点帧中灰（判便服）——便服不触发串人守卫也不触发硬守卫
    cache = _shot_cache()
    frames = _frame(tmp_path, FID, 12, Box(0, 0, 100, 100), "black")
    _frame(tmp_path, FID, 15, Box(0, 0, 100, 100), (128, 128, 128))
    # Act
    path, reason, fi, box = find_landing(cache, _event(), (250, 50), frames, expect_color="白")
    # Assert：便服放行，主路命中
    assert (path, reason) == ("trace", "")
    assert fi == 15 and box == Box(0, 0, 100, 100)


def test_find_landing_team_guard_disabled(tmp_path: Path) -> None:
    # Arrange：expect_color 空（team_color 键缺失退化）→ 框色黑也不剔除
    cache = _shot_cache()
    frames = _frame(tmp_path, FID, 12, Box(0, 0, 100, 100), "black")
    _frame(tmp_path, FID, 15, Box(0, 0, 100, 100), "black")
    # Act & Assert
    assert find_landing(cache, _event(), (250, 50), frames, expect_color="")[0] == "trace"


# ---------- court_template_lines ----------


def test_court_template_lines() -> None:
    lines = court_template_lines()
    assert lines["corner_y"] == pytest.approx(1.4151, abs=1e-3)
    assert lines["paint"] == (-2.45, -1.575, 2.45, 4.225)
    assert lines["rim"] == ((0.0, 0.0), 0.225)


# ---------- heat_session 端到端 ----------


def _write_json(path: Path, payload: object) -> None:
    """写 JSON 到 path。"""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _setup_session(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """搭合成场次：roster + goals + candidates + hoops + cache + 帧图目录。"""
    session = tmp_path / "work" / "s1"
    detect = tmp_path / "work" / "detect"
    frames = tmp_path / "work" / "frames"
    out = tmp_path / "output" / "s1"
    session.mkdir(parents=True)
    detect.mkdir()
    frames.mkdir()
    roster = {
        "session": "s1",
        "confirmed": False,
        "players": [
            {"tag": "黑9", "name": "", "team": "主队"},
            {"tag": "便服A", "name": "", "team": "便服"},
        ],
        "assignments": {f"{FID}.mp4#4.0": "黑9", "fid2.mp4#5.0": "便服A"},
    }
    _write_json(session / "roster.json", roster)
    goals = {
        "session": "s1",
        "goals": [
            {
                "file": f"{FID}.mp4",
                "anchor_time": ANCHOR,
                "clip_start": 0.0,
                "clip_end": 6.0,
                "status": "confirmed",
                "scorer": "",
            },
            {
                "file": "fid2.mp4",
                "anchor_time": 5.0,
                "clip_start": 1.0,
                "clip_end": 7.0,
                "status": "confirmed",
                "scorer": "",
            },
        ],
    }
    _write_json(session / "goals_batch1.json", goals)
    _write_json(
        session / "candidates_batch1.json",
        [
            {
                "t0": ANCHOR,
                "dur": 1.0,
                "ac": 0.7,
                "cx": 250,
                "cy": 50,
                "src": "rejoin",
                "fid": FID,
                "label": "#1",
            }
        ],
    )
    _write_json(session / "hoops_batch1.json", _hoop_payload())
    cache = _shot_cache()
    balls = [
        [
            {
                "conf": d.conf,
                "box": d.box,
                "cx": d.cx,
                "cy": d.cy,
                "sec": d.sec,
                "frame_idx": d.frame_idx,
            }
            for d in fb
        ]
        for fb in cache.balls
    ]
    persons = [[[b.x1, b.y1, b.x2, b.y2] for b in fp] for fp in cache.persons]
    _write_json(
        detect / f"{FID}_mot_cache.json", {"frames": N_FRAMES, "balls": balls, "persons": persons}
    )
    return session, detect, frames, out


def test_heat_session_end_to_end(tmp_path: Path) -> None:
    # Arrange
    session, detect, frames, out = _setup_session(tmp_path)
    # Act
    report = heat_session(session, detect, frames, out)
    # Assert：2 已标记球——fid1 主路覆盖（主队），fid2 便服剔除计分母
    s = report["summary"]
    assert s["total_marked"] == 2
    assert s["covered"] == 1
    assert s["coverage"] == 0.5
    assert s["coverage_pass"] is False  # 0.5 < 0.55（2 球样本，仅验证判定逻辑）
    assert s["by_path"] == {"trace": 1}
    assert s["uncovered_by_reason"] == {"casual_team": 1}
    assert s["teams"] == {"主队": 1}
    rec = next(r for r in report["landings"] if r["covered"])
    assert rec["event_key"] == f"{FID}@4.0"
    assert rec["frame_idx"] == 15
    assert rec["landing_px"] == (50.0, 100.0)
    # rel：落点 (50,100) 筐心 (1000,402)，框高 100px → 1px=0.0175m
    assert rec["rel_xy_m"][0] == pytest.approx((50 - 1000) / 100 * 1.75)
    assert rec["rel_xy_m"][1] == pytest.approx((100 - 402) / 100 * 1.75)
    assert rec["flipped"] is False  # 单球 cx=1000，中位阈值=1000，不大于不翻
    # 产物：JSON + 暗场/蜂巢热图 PNG + 目击拼图
    assert (session / "goal_landings.json").exists()
    assert (out / "队伍_主队_进球热图.png").exists()
    assert (out / "队伍_主队_进球热图_分区.png").exists()
    assert (session / "heatmap_audit.png").exists()
    # v4：无 session_facts.json → 队色硬守卫禁用（WARNING 退化 v3 行为）
    assert report["params"]["team_color_guard"] is False
    assert report["params"]["held_search_before_sec"] == 0.5
    # v4.1：合成落点 rel≈(-16.6, -5.3) 界外 → 渲染层过滤，JSON 原始数据不动
    assert report["summary"]["out_of_bounds"] == 1


def test_heat_session_no_roster_raises(tmp_path: Path) -> None:
    # Arrange：空场次目录
    session = tmp_path / "work" / "s1"
    session.mkdir(parents=True)
    # Act & Assert
    with pytest.raises(BasketballPipelineError):
        heat_session(session, tmp_path / "d", tmp_path / "f", tmp_path / "o")


# ---------- 渲染 / 拼图 smoke ----------


def test_render_team_heatmap_smoke(tmp_path: Path) -> None:
    # Arrange & Act
    out = tmp_path / "h.png"
    render_team_heatmap([(1.0, 3.0), (-2.0, 5.5), (0.5, 6.8)], "主队", "s1", out)
    # Assert
    assert out.exists() and out.stat().st_size > 10000


def test_render_team_heatmap_empty(tmp_path: Path) -> None:
    # 空点集也出模板图
    out = tmp_path / "h.png"
    render_team_heatmap([], "对手队", "s1", out)
    assert out.exists()


# ---------- v4.1/v4.2：渲染 + 界外过滤 ----------


def test_filter_in_court_known() -> None:
    # Arrange：界内 1 点；越界 4 点（横向 ±、纵向上下各一）
    pts = [(0.0, 2.0), (8.2, 3.0), (-9.0, 1.0), (0.0, 13.5), (0.0, -2.5)]
    # Act
    kept, dropped = filter_in_court(pts)
    # Assert：余量 0.5m——|dx|>8.0、dy∉[−2.075,12.925] 剔除
    assert kept == [(0.0, 2.0)]
    assert len(dropped) == 4
    # 边界值恰好在线上不过滤
    kept2, dropped2 = filter_in_court([(8.0, 12.925), (-8.0, -2.075)])
    assert len(kept2) == 2 and not dropped2


def test_zone_of_known() -> None:
    # 10 区代表点归区（含余量带 dy<0 归就近区——spec v4.2 S2 口径）
    cases = [
        ((0.0, 0.5), "ra"),
        ((0.0, -1.0), "ra"),  # 余量带 dy<0：r=1.0 ≤ 1.25 归 ra（计数守恒）
        ((0.0, 3.0), "paint"),
        ((0.0, 5.5), "mid_c"),
        ((-4.5, 2.0), "mid_l"),
        ((4.5, 2.0), "mid_r"),
        ((-7.0, 0.5), "corner_l"),
        ((7.0, 0.5), "corner_r"),
        ((-6.2, 4.5), "p45_l"),  # atan2(6.2, 4.5) ≈ 54° > 50° 分角
        ((6.2, 4.5), "p45_r"),
        ((0.0, 8.0), "top"),
        ((2.0, 8.5), "top"),  # atan2(2, 8.5) ≈ 13° < 50°
    ]
    for (x, y), expected in cases:
        assert zone_of(x, y) == expected, f"({x}, {y})"


def test_build_zones_geometry() -> None:
    zones = build_zones()
    # 10 区、键唯一、多边形闭合可行（≥3 顶点）、标注位齐备
    assert len(zones) == 10
    keys = [z.key for z in zones]
    assert len(set(keys)) == 10
    assert all(len(z.poly) >= 3 for z in zones)
    assert all(len(z.label) == 2 for z in zones)
    # 绘制顺序：ra 在 paint 之后（压在禁区色块之上）
    assert keys.index("ra") > keys.index("paint")


def test_render_team_heatmap_zones_smoke(tmp_path: Path) -> None:
    # Arrange & Act：有点集出图（含界外计数副标）
    out = tmp_path / "zones.png"
    render_team_heatmap_zones(
        [(1.0, 3.0), (1.1, 3.1), (-2.0, 5.5)], "主队", "s1", out, oob=1, palette_idx=0
    )
    # Assert
    assert out.exists() and out.stat().st_size > 10000


def test_render_team_heatmap_zones_empty(tmp_path: Path) -> None:
    # 空点集也出分区结构图
    out = tmp_path / "zones_empty.png"
    render_team_heatmap_zones([], "对手队", "s1", out, palette_idx=1)
    assert out.exists()


def test_build_audit_grid_deterministic(tmp_path: Path) -> None:
    # Arrange：两条覆盖记录 + 帧图（落点帧 15、锚点帧 20）
    frames = tmp_path / "frames"
    (frames / FID).mkdir(parents=True)
    for idx in (15, 20):
        Image.new("RGB", (1920, 1080), "white").save(frames / FID / f"f_{idx + 1:05d}.jpg", "JPEG")
    rec = HeatLanding(
        event_key=f"{FID}@4.0",
        team="主队",
        covered=True,
        reason="",
        path="trace",
        frame_idx=15,
        landing_px=(50.0, 100.0),
        landing_box=(0, 0, 100, 100),
        box_h_px=100,
        hoop_xy=(1000.0, 402.0),
        hoop_degraded=False,
        flipped=False,
        rel_xy_m=(-16.6, -5.3),
    )
    events = {rec.event_key: _event()}
    out = tmp_path / "audit.png"
    # Act
    keys1 = build_audit_grid([rec], events, frames, out)
    keys2 = build_audit_grid([rec], events, frames, tmp_path / "audit2.png")
    # Assert：确定性 + 文件生成
    assert keys1 == keys2 == [rec.event_key]
    assert out.exists()
