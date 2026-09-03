"""release_probe.py 单元测试（热图阶段 0 · Q2 回溯启发式）。

覆盖：goals schema 校验（合法/缺字段/重复/非 confirmed 跳过）、窗口索引边界、
窗口取球（多球取最高 conf）、稳定段切分（稳定/瞬移/断帧/孤立帧/多段取末段）、
出手帧后持球截断、球心不落人框、贴边/重叠观察项、缺缓存/缺帧计分母、
等间距抽样、端到端 probe_session（合成 goals + mot_cache + 帧图，含一致性抽查）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from crop_scorers import MotCache
from errors import BasketballPipelineError, SchemaError
from geom import Box
from mot_candidates import Detection
from release_probe import (
    GoalEvent,
    _miss,
    collect_window_dets,
    even_sample,
    find_stable_segments,
    frame_size,
    load_goals_events,
    probe_event,
    probe_session,
    window_indices,
)

N_FRAMES: int = 60
ANCHOR: float = 10.0  # 锚点帧 idx 50；窗口 0.4~2.5s → idx 38..48
FRAME_W: int = 1920
FRAME_H: int = 1080


def _ball(cx: int, cy: int, fi: int, conf: float = 0.5) -> Detection:
    """构造球检测（box 取球心 ±10px；sec 按 5fps 换算）。"""
    return Detection(
        conf=conf,
        box=[cx - 10, cy - 10, cx + 10, cy + 10],
        cx=cx,
        cy=cy,
        sec=fi / 5,
        frame_idx=fi,
    )


def _person(cx: int, bottom: int = 1000, w: int = 80, h: int = 300) -> Box:
    """构造人框：底边中点 (cx, bottom)。"""
    return Box(cx - w // 2, bottom - h, cx + w // 2, bottom)


def _cache(
    balls: list[tuple[Detection, ...]], persons: list[tuple[Box, ...]] | None = None
) -> MotCache:
    """由逐帧球检测构造 MotCache；persons 缺省每帧无人。"""
    if persons is None:
        persons = [tuple() for _ in balls]
    assert len(balls) == len(persons)
    return MotCache(frames=len(balls), balls=tuple(balls), persons=tuple(persons))


def _event(anchor: float = ANCHOR, fid: str = "fid1") -> GoalEvent:
    """构造事件（event_key=fid@anchor）。"""
    return GoalEvent(fid=fid, anchor_time=anchor, event_key=f"{fid}@{anchor}")


def _empty_frames(n: int = N_FRAMES) -> list[tuple[Detection, ...]]:
    """n 帧全空球检测。"""
    return [tuple() for _ in range(n)]


# ---------- load_goals_events ----------


def _write_goals(path: Path, goals: list[dict]) -> None:
    """写 goals JSON 到 path。"""
    payload = json.dumps({"session": "s", "goals": goals}, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")


def _goal(file: str = "fid1.mp4", anchor: float = ANCHOR, status: str = "confirmed") -> dict:
    """构造一条 goals 条目。"""
    return {
        "file": file,
        "anchor_time": anchor,
        "clip_start": anchor - 4,
        "clip_end": anchor + 2,
        "status": status,
        "scorer": "",
    }


def test_load_goals_events_ok(tmp_path: Path) -> None:
    # Arrange
    p = tmp_path / "goals_batch1.json"
    _write_goals(p, [_goal(), _goal("fid2.mp4", 20.5), _goal("fid3.mp4", 5.0, status="removed")])
    # Act
    events = load_goals_events(p)
    # Assert：非 confirmed 跳过；fid 去 .mp4
    assert [(e.fid, e.anchor_time) for e in events] == [("fid1", 10.0), ("fid2", 20.5)]
    assert events[0].event_key == "fid1@10.0"


def test_load_goals_events_schema_errors(tmp_path: Path) -> None:
    # Arrange：顶层不是对象 / 缺 anchor_time / file 非字符串
    bad_top = tmp_path / "a.json"
    bad_top.write_text(json.dumps([1, 2]), encoding="utf-8")
    bad_field = tmp_path / "b.json"
    _write_goals(bad_field, [{"file": "f.mp4", "status": "confirmed"}])
    bad_type = tmp_path / "c.json"
    _write_goals(bad_type, [_goal(file=123)])  # type: ignore[arg-type]
    # Act & Assert
    with pytest.raises(SchemaError):
        load_goals_events(bad_top)
    with pytest.raises(SchemaError):
        load_goals_events(bad_field)
    with pytest.raises(SchemaError):
        load_goals_events(bad_type)


def test_load_goals_events_duplicate_raises(tmp_path: Path) -> None:
    # Arrange
    p = tmp_path / "goals_batch1.json"
    _write_goals(p, [_goal(), _goal()])
    # Act & Assert
    with pytest.raises(SchemaError):
        load_goals_events(p)


# ---------- window_indices ----------


def test_window_indices_normal() -> None:
    # Act & Assert：锚点 10.0 → [10-2.5, 10-0.4]s → idx 38..48
    assert window_indices(10.0, N_FRAMES) == (38, 48)


def test_window_indices_clamps_and_empty() -> None:
    # 锚点过早 → 远端裁到 0
    assert window_indices(2.0, N_FRAMES) == (0, 8)
    # 窗口整体越界（锚点 < 0.4s）→ None
    assert window_indices(0.3, N_FRAMES) is None
    # 窗口超出缓存末尾 → None
    assert window_indices(50.0, 10) is None


# ---------- collect_window_dets ----------


def test_collect_window_dets_picks_max_conf() -> None:
    # Arrange：idx 38 双球取高 conf，39 空，40 单球
    balls = _empty_frames()
    balls[38] = (_ball(100, 100, 38, conf=0.3), _ball(200, 200, 38, conf=0.9))
    balls[40] = (_ball(300, 300, 40),)
    cache = _cache(balls)
    # Act
    dets = collect_window_dets(cache, 38, 48)
    # Assert
    assert [(d.cx, d.frame_idx) for d in dets] == [(200, 38), (300, 40)]


# ---------- find_stable_segments ----------


def test_stable_segment_basic() -> None:
    # Arrange：连续 4 帧小位移 → 一段稳定段
    dets = [_ball(540 + i * 5, 750, 40 + i) for i in range(4)]
    # Act
    segs = find_stable_segments(dets)
    # Assert
    assert len(segs) == 1
    assert segs[0][-1].frame_idx == 43


def test_jump_splits_segment() -> None:
    # Arrange：持球 40-43，44 起飞行（相邻位移 >60px）→ 跳变切段
    held = [_ball(540 + i * 5, 750, 40 + i) for i in range(4)]
    flight = [_ball(700 + i * 120, 600 - i * 100, 44 + i) for i in range(3)]
    # Act
    segs = find_stable_segments(held + flight)
    # Assert：持球段稳定、飞行段不稳
    assert len(segs) == 1
    assert segs[0][-1].frame_idx == 43


def test_gap_splits_and_multi_segments_take_last() -> None:
    # Arrange：38-39 一对，断 3 帧，43-44 一对 → 两段稳定段
    dets = [_ball(100, 100, 38), _ball(105, 100, 39), _ball(500, 500, 43), _ball(505, 500, 44)]
    # Act
    segs = find_stable_segments(dets)
    # Assert
    assert len(segs) == 2
    assert segs[-1][-1].frame_idx == 44


def test_gap_two_frames_stays_same_segment() -> None:
    # Arrange：间隔恰 2 帧不断段（断帧不重置）
    dets = [_ball(100, 100, 38), _ball(110, 100, 40), _ball(120, 100, 42)]
    # Act
    segs = find_stable_segments(dets)
    # Assert
    assert len(segs) == 1
    assert len(segs[0]) == 3


def test_isolated_dets_no_stable_segment() -> None:
    # Arrange：孤立有球帧（间隔 3 帧）→ 无稳定段
    dets = [_ball(100 * i, 100, 38 + 3 * i) for i in range(3)]
    # Act & Assert
    assert find_stable_segments(dets) == []


def test_empty_dets() -> None:
    assert find_stable_segments([]) == []


# ---------- probe_event ----------


def _held_cache(persons_per_frame: list[tuple[Box, ...]]) -> MotCache:
    """构造：40-43 帧持球稳定段（球心 (540+i*5, 750)），44-46 帧飞行跳变。"""
    balls = _empty_frames()
    for i in range(4):
        balls[40 + i] = (_ball(540 + i * 5, 750, 40 + i),)
    for i in range(3):
        balls[44 + i] = (_ball(700 + i * 120, 600 - i * 100, 44 + i),)
    return _cache(balls, persons_per_frame)


def test_probe_event_hit() -> None:
    # Arrange：持球人框 (500,700)-(580,1000)，球心 (540..555, 750) 落入
    persons = [(_person(540),) for _ in range(N_FRAMES)]
    cache = _held_cache(persons)
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert：出手帧 = 稳定段末帧 43；落点 = 底边中点 (540, 1000)
    assert r.hit and r.reason == ""
    assert r.release_frame_idx == 43
    assert r.held_frame_idx == 43
    assert r.landing_px == (540.0, 1000.0)
    assert r.n_stable_segments == 1
    assert not r.edge_touch and not r.overlap


def test_probe_event_held_from_earlier_frame() -> None:
    # Arrange：43 帧球心 (555,750) 移出人框（人框 500..580 但 43 帧无人框），
    # 42 帧有人框 → 持球帧截断回退到 42
    persons: list[tuple[Box, ...]] = [tuple() for _ in range(N_FRAMES)]
    persons[42] = (_person(545),)
    cache = _held_cache(persons)
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert
    assert r.hit
    assert r.release_frame_idx == 43
    assert r.held_frame_idx == 42
    assert r.landing_px == (545.0, 1000.0)


def test_probe_event_no_stable_segment() -> None:
    # Arrange：窗口内逐帧瞬移 → 无稳定段
    balls = _empty_frames()
    for i, fi in enumerate(range(38, 49)):
        balls[fi] = (_ball(100 + i * 100, 100 + i * 50, fi),)
    cache = _cache(balls)
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert
    assert not r.hit and r.reason == "no_stable_segment"
    assert r.n_window_dets == 11


def test_probe_event_ball_not_in_box() -> None:
    # Arrange：稳定段但全程无人框罩住球心
    cache = _held_cache([tuple() for _ in range(N_FRAMES)])
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert：已定位出手帧但持球判定失败
    assert not r.hit and r.reason == "ball_not_in_box"
    assert r.release_frame_idx == 43


def test_probe_event_no_ball_detection() -> None:
    # Arrange：窗口内全无球
    cache = _cache(_empty_frames())
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert
    assert not r.hit and r.reason == "no_ball_detection"


def test_probe_event_empty_window() -> None:
    # Arrange：锚点过早窗口越界
    cache = _cache(_empty_frames())
    # Act
    r = probe_event(cache, _event(anchor=0.3), FRAME_W, FRAME_H)
    # Assert
    assert not r.hit and r.reason == "empty_window"


def test_probe_event_edge_and_overlap_flags() -> None:
    # Arrange：持球人框与他人框 IoU ≥0.1，另有贴边人框（罩不住球）
    main_box = Box(500, 700, 580, 1000)
    other = Box(560, 720, 660, 1000)  # 与 main 相交 20×280 / 并集 ≈ 0.12 ≥ 0.1
    edge_box = Box(4, 700, 84, 1000)  # 贴左边但罩不住球
    persons = [(main_box, other, edge_box) for _ in range(N_FRAMES)]
    cache = _held_cache(persons)
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert：overlap=True（main 与 other 重叠）；edge_touch=False（main 不贴边）
    assert r.hit
    assert r.overlap
    assert not r.edge_touch


def test_probe_event_edge_touch_true() -> None:
    # Arrange：持球人框贴底边（y2=1080=FRAME_H，框高 400 罩住球心 cy=750）
    persons = [(_person(540, bottom=1080, h=400),) for _ in range(N_FRAMES)]
    cache = _held_cache(persons)
    # Act
    r = probe_event(cache, _event(), FRAME_W, FRAME_H)
    # Assert
    assert r.hit and r.edge_touch


# ---------- even_sample ----------


def test_even_sample() -> None:
    items = [f"e{i:02d}" for i in range(20)]
    out = even_sample(items, 10)
    assert len(out) == 10
    assert out[0] == "e00" and out[-1] == "e19"
    assert even_sample(items[:5], 10) == items[:5]
    assert even_sample([], 10) == []


# ---------- frame_size ----------


def test_frame_size_missing_dir(tmp_path: Path) -> None:
    assert frame_size(tmp_path, "no_such_fid") is None


def test_frame_size_reads_header(tmp_path: Path) -> None:
    # Arrange
    d = tmp_path / "fid1"
    d.mkdir()
    Image.new("RGB", (1920, 1080)).save(d / "f_00001.jpg", "JPEG")
    # Act & Assert
    assert frame_size(tmp_path, "fid1") == (1920, 1080)


# ---------- probe_session 端到端 ----------


def _mot_cache_payload(with_person: bool = True) -> dict:
    """构造 mot_cache JSON：40-43 帧持球稳定段 + 44-46 帧飞行。"""

    def _det_json(b: Detection) -> dict:
        """Detection 转 mot_cache 原始字典。"""
        return {
            "conf": b.conf,
            "box": b.box,
            "cx": b.cx,
            "cy": b.cy,
            "sec": b.sec,
            "frame_idx": b.frame_idx,
        }

    balls: list[list[dict]] = [[] for _ in range(N_FRAMES)]
    persons: list[list[list[int]]] = [[] for _ in range(N_FRAMES)]
    for i in range(4):
        balls[40 + i] = [_det_json(_ball(540 + i * 5, 750, 40 + i))]
    for i in range(3):
        balls[44 + i] = [_det_json(_ball(700 + i * 120, 600 - i * 100, 44 + i))]
    if with_person:
        box = _person(540)
        for i in range(N_FRAMES):
            persons[i] = [[box.x1, box.y1, box.x2, box.y2]]
    return {"frames": N_FRAMES, "balls": balls, "persons": persons}


def test_probe_session_end_to_end(tmp_path: Path) -> None:
    # Arrange：session 目录取 1 个事件；detect/frames 齐备
    session = tmp_path / "work" / "s1"
    detect = tmp_path / "work" / "detect"
    frames = tmp_path / "work" / "frames"
    session.mkdir(parents=True)
    detect.mkdir()
    (frames / "fid1").mkdir(parents=True)
    _write_goals(session / "goals_batch1.json", [_goal()])
    (detect / "fid1_mot_cache.json").write_text(json.dumps(_mot_cache_payload()), encoding="utf-8")
    Image.new("RGB", (1920, 1080)).save(frames / "fid1" / "f_00001.jpg", "JPEG")
    out = session / "release_probe.json"
    # Act
    report = probe_session(session, detect, frames, out)
    # Assert：1/1 命中（覆盖率 100% ≥ 70%）；一致性抽查 1 球两方向均一致；JSON 已落盘
    assert report["summary"]["total"] == 1
    assert report["summary"]["hits"] == 1
    assert report["summary"]["q2_pass"] is True
    assert report["consistency"]["n_consistent"] == 1
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["events"][0]["event_key"] == "fid1@10.0"


def test_probe_session_missing_cache_counts_denominator(tmp_path: Path) -> None:
    # Arrange：goals 有事件但无 mot_cache
    session = tmp_path / "work" / "s1"
    session.mkdir(parents=True)
    _write_goals(session / "goals_batch1.json", [_goal()])
    out = session / "release_probe.json"
    # Act
    report = probe_session(session, tmp_path / "work" / "detect", tmp_path / "work" / "frames", out)
    # Assert：缺缓存计未命中入分母
    assert report["summary"]["total"] == 1
    assert report["summary"]["hits"] == 0
    assert report["summary"]["miss_by_reason"] == {"missing_cache": 1}


def test_probe_session_no_goals_raises(tmp_path: Path) -> None:
    # Arrange：空场次目录
    session = tmp_path / "work" / "s1"
    session.mkdir(parents=True)
    # Act & Assert
    with pytest.raises(BasketballPipelineError):
        probe_session(session, tmp_path / "d", tmp_path / "f", session / "out.json")


def test_miss_helper_defaults() -> None:
    # Arrange & Act
    r = _miss(_event(), "missing_cache")
    # Assert
    assert not r.hit
    assert r.release_frame_idx == -1 and r.landing_px is None and r.person_box is None
