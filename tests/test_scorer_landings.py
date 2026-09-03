"""scorer_landings.py 单元测试（热图阶段 0 · 新 Q2 轨迹法落点）。

覆盖：真持球命中（落点=人框底边中点）、start_fallback 计未命中、
no_track_near_anchor SKIP、缺缓存计分母、SKIP 落盘写死形态（-1/-1.0/null
字段不省略）、锚点超差退化端点时间最近、candidates 合并与 schema 校验、
端到端 land_session（合成 goals + candidates + mot_cache，含 Q2 判定）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crop_scorers import MotCache
from errors import BasketballPipelineError, SchemaError
from geom import Box
from mot_candidates import Detection
from release_probe import GoalEvent
from scorer_landings import (
    LandingRecord,
    is_usable,
    land_event,
    land_session,
    load_merged_candidates,
)

N_FRAMES: int = 30
ANCHOR: float = 4.0  # 锚点帧 idx 20（locate 窗口 [0, 4.5]s → idx 0..22）


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


def _event(fid: str = "fid1", anchor: float = ANCHOR) -> GoalEvent:
    """构造事件（event_key=fid@anchor，anchor 原值序列化）。"""
    return GoalEvent(fid=fid, anchor_time=anchor, event_key=f"{fid}@{anchor}")


def _shot_cache(ball_y: int = 50, with_persons: bool = True) -> MotCache:
    """仿 test_crop_scorers 的合成场景：进球轨迹帧 10..20 从 (50,ball_y)
    每帧 +20px 移到 (250,ball_y)；人框 A(0,0,100,100) 罩住球心至帧 12。"""
    persons = [(Box(0, 0, 100, 100),)] * N_FRAMES if with_persons else [()] * N_FRAMES
    balls: list[tuple[Detection, ...]] = [()] * N_FRAMES
    for fi in range(10, 21):
        balls[fi] = (_ball(0.9, 50 + (fi - 10) * 20, ball_y, fi),)
    return _cache(balls, persons)


# ---------- land_event ----------


def test_land_event_held_hit() -> None:
    # Arrange：轨迹端点 (250,50) = 候选锚点；最后持球点帧 12（球心 x=90 在 A 内）
    cache = _shot_cache()
    # Act
    r = land_event(cache, _event(), (250, 50))
    # Assert：OK 真持球；落点 = A 框底边中点 (50, 100)
    assert r.status == "OK" and r.reason == ""
    assert is_usable(r)
    assert r.frame_idx == 12
    assert r.sec == pytest.approx(2.4)
    assert r.person_box == Box(0, 0, 100, 100)
    assert r.landing_px == (50.0, 100.0)


def test_land_event_fallback_not_usable() -> None:
    # Arrange：球沿 y=300 全程不在人框内 → 起点回退
    cache = _shot_cache(ball_y=300)
    # Act
    r = land_event(cache, _event(), (250, 300))
    # Assert：OK 但 start_fallback，不计入可用落点（落点仍落盘供排查）
    assert r.status == "OK" and r.reason == "start_fallback"
    assert not is_usable(r)
    assert r.frame_idx == 10
    assert r.landing_px == (50.0, 100.0)


def test_land_event_skip_far_anchor() -> None:
    # Arrange：候选锚点离轨迹端点 >200px
    cache = _shot_cache()
    # Act
    r = land_event(cache, _event(), (1500, 900))
    # Assert：SKIP 写死形态 -1/-1.0/None
    assert r.status == "SKIP" and r.reason == "no_track_near_anchor"
    assert r.frame_idx == -1 and r.sec == -1.0
    assert r.person_box is None and r.landing_px is None
    assert not is_usable(r)


def test_land_event_skip_no_anchor_degrades_to_time() -> None:
    # Arrange：anchor_xy=None 退化端点时间最近选轨迹（认人同口径）
    cache = _shot_cache()
    # Act
    r = land_event(cache, _event(), None)
    # Assert：唯一轨迹被选中，正常定位
    assert r.status == "OK" and is_usable(r)
    assert r.frame_idx == 12


# ---------- load_merged_candidates ----------


def _write_candidates(path: Path, payload: list) -> None:
    """写 candidates JSON 到 path。"""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _cand(fid: str, t0: float, cx: int = 250, cy: int = 50) -> dict:
    """构造一条候选记录。"""
    return {
        "t0": t0,
        "dur": 1.0,
        "ac": 0.7,
        "cx": cx,
        "cy": cy,
        "src": "rejoin",
        "fid": fid,
        "label": "#1",
    }


def test_load_merged_candidates_merges_batches(tmp_path: Path) -> None:
    # Arrange：两个批次文件，同 fid 跨批次拼接
    p1 = tmp_path / "candidates_batch1.json"
    p2 = tmp_path / "candidates_batch2.json"
    _write_candidates(p1, [_cand("fid1", 4.0)])
    _write_candidates(p2, [_cand("fid1", 30.0), _cand("fid2", 5.0)])
    # Act
    index = load_merged_candidates([p1, p2])
    # Assert
    assert index["fid1"] == [(4.0, 250, 50), (30.0, 250, 50)]
    assert index["fid2"] == [(5.0, 250, 50)]


def test_load_merged_candidates_schema_error(tmp_path: Path) -> None:
    # Arrange：顶层非列表
    p = tmp_path / "candidates_batch1.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    # Act & Assert
    with pytest.raises(SchemaError):
        load_merged_candidates([p])


# ---------- land_session 端到端 ----------


def _mot_cache_payload() -> dict:
    """与 _shot_cache 同场景的 mot_cache 原始 JSON。"""
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
            for d in frame_balls
        ]
        for frame_balls in cache.balls
    ]
    persons = [[[b.x1, b.y1, b.x2, b.y2] for b in frame_persons] for frame_persons in cache.persons]
    return {"frames": N_FRAMES, "balls": balls, "persons": persons}


def _setup_session(tmp_path: Path) -> tuple[Path, Path]:
    """搭合成场次：goals_batch1（1 球）+ candidates_batch1 + mot_cache。"""
    session = tmp_path / "work" / "s1"
    detect = tmp_path / "work" / "detect"
    session.mkdir(parents=True)
    detect.mkdir()
    goals = {
        "session": "s1",
        "goals": [
            {
                "file": "fid1.mp4",
                "anchor_time": ANCHOR,
                "clip_start": 0.0,
                "clip_end": 6.0,
                "status": "confirmed",
                "scorer": "",
            }
        ],
    }
    (session / "goals_batch1.json").write_text(
        json.dumps(goals, ensure_ascii=False), encoding="utf-8"
    )
    _write_candidates(session / "candidates_batch1.json", [_cand("fid1", ANCHOR)])
    (detect / "fid1_mot_cache.json").write_text(json.dumps(_mot_cache_payload()), encoding="utf-8")
    return session, detect


def test_land_session_end_to_end(tmp_path: Path) -> None:
    # Arrange
    session, detect = _setup_session(tmp_path)
    out = session / "scorer_landings.json"
    # Act
    report = land_session(session, detect, out)
    # Assert：1/1 真持球；SKIP 形态与落点字段齐备；JSON 落盘
    s = report["summary"]
    assert s["total"] == 1 and s["usable"] == 1 and s["fallback"] == 0
    assert s["usable_ratio"] == 1.0 and s["q2_pass"] is True
    rec = report["landings"][0]
    assert rec["event_key"] == "fid1@4.0"
    assert rec["frame_idx"] == 12 and rec["landing_px"] == (50.0, 100.0)
    assert rec["person_box"] == {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["total"] == 1


def test_land_session_missing_cache_counts_denominator(tmp_path: Path) -> None:
    # Arrange：goals 有事件但 detect 目录无缓存
    session, _ = _setup_session(tmp_path)
    (tmp_path / "work" / "detect" / "fid1_mot_cache.json").unlink()
    # Act
    report = land_session(session, tmp_path / "work" / "detect", session / "out.json")
    # Assert：missing_cache 计 SKIP 入分母
    s = report["summary"]
    assert s["total"] == 1 and s["usable"] == 0
    assert s["skip_by_reason"] == {"missing_cache": 1}
    rec = report["landings"][0]
    assert rec["frame_idx"] == -1 and rec["sec"] == -1.0
    assert rec["person_box"] is None and rec["landing_px"] is None


def test_land_session_no_candidates_raises(tmp_path: Path) -> None:
    # Arrange：有 goals 无 candidates
    session, _ = _setup_session(tmp_path)
    (session / "candidates_batch1.json").unlink()
    # Act & Assert：筐锚定缺失必须显式停
    with pytest.raises(BasketballPipelineError):
        land_session(session, tmp_path / "work" / "detect", session / "out.json")


def test_land_session_no_goals_raises(tmp_path: Path) -> None:
    # Arrange：空场次目录
    session = tmp_path / "work" / "s1"
    session.mkdir(parents=True)
    # Act & Assert
    with pytest.raises(BasketballPipelineError):
        land_session(session, tmp_path / "d", session / "out.json")


def test_landing_record_skip_shape() -> None:
    # Arrange & Act：SKIP 记录字段形态写死（字段不省略，供下游消费）
    r = LandingRecord(
        event_key="f@1.0",
        fid="f",
        anchor_time=1.0,
        status="SKIP",
        reason="no_track",
        frame_idx=-1,
        sec=-1.0,
        person_box=None,
        landing_px=None,
    )
    # Assert
    keys = set(r.__dataclass_fields__)
    assert {
        "event_key",
        "fid",
        "anchor_time",
        "status",
        "reason",
        "frame_idx",
        "sec",
        "person_box",
        "landing_px",
    } == keys
