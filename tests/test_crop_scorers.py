"""crop_scorers.py 单元测试（轨迹法定位 + 颜色分队 + 预览片段 + 轨迹选帧多裁）。

覆盖：mot_cache schema 校验、轨迹窗口取帧、进球轨迹选择（端点距候选锚点最近/
超界 SKIP/无锚点退时间最近）、持球点回放判定、无持球起点回退、无轨迹/无人 SKIP、
candidates 锚点索引匹配、裁图外扩 20% 且短边 ≥400px、颜色三分类（近阈归便服）、
预览片段参数与失败容错、CLI 端到端（合成 goals + mot_cache + 帧图，不碰真实素材）、
轨迹选帧多裁（人框 IoU 链 链上/链断/多人/越界、质量分排序、≥0.5s 去重、
crops/crop_scores 契约与 SKIP 无多裁字段）、读号多帧投票（规则全路径）、
number_cache md5 重键迁移（幂等/查不到保留/裁图缺失 WARNING）、跳票模式零调用、
全量模式 --max-reads 闸（按裁图张数计）、
轨迹传播种子字段（OK 落 seed_frame/seed_box/seed_team，SKIP 不落）。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

import crop_scorers
from crop_scorers import (
    MotCache,
    NumberGuess,
    _crop_name_ranked,
    _try_cut_preview,
    apply_number_reading,
    classify_team,
    crop_and_save,
    cut_preview_clip,
    drop_opposite_team,
    expand_box,
    expanded_ratio,
    file_md5,
    find_held_box,
    frame_quality,
    load_candidates_index,
    load_mot_cache,
    load_number_cache,
    locate_scorer,
    main,
    make_quality_gate,
    match_anchor_xy,
    migrate_number_cache,
    number_guess_from_dict,
    parse_number_answer,
    pick_best_frames,
    preview_window,
    ratio_gate_reason,
    read_number,
    save_number_cache,
    score_chain_frames,
    select_goal_track,
    start_nearest_box,
    team_of_box,
    trace_person,
    track_window_dets,
    vote_number_guess,
)
from errors import BasketballPipelineError, ExternalApiError, SchemaError
from geom import Box
from mot_candidates import Detection, Track


def _ball(conf: float, cx: int, cy: int, fi: int, sec: float | None = None) -> Detection:
    """构造一个球检测（box 取球心 ±10px；sec 缺省按 5fps 帧率换算）。"""
    return Detection(
        conf=conf,
        box=[cx - 10, cy - 10, cx + 10, cy + 10],
        cx=cx,
        cy=cy,
        sec=fi / 5 if sec is None else sec,
        frame_idx=fi,
    )


def _cache(balls: list[tuple[Detection, ...]], persons: list[tuple[Box, ...]]) -> MotCache:
    """由逐帧列表构造 MotCache（frames 取两者长度）。"""
    assert len(balls) == len(persons)
    return MotCache(frames=len(balls), balls=tuple(balls), persons=tuple(persons))


def _empty_cache(frames: int) -> MotCache:
    """构造全空缓存（无球无人）。"""
    return _cache([()] * frames, [()] * frames)


def _track(points: list[tuple[int, int, int]]) -> Track:
    """由 (cx, cy, frame_idx) 列表构造轨迹。"""
    return Track(dets=[_ball(0.9, cx, cy, fi) for cx, cy, fi in points])


@pytest.fixture(autouse=True)
def _stub_person_rechecker(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认桩掉框内人物复检（高置信放行），全文件不碰真 yolov8n 模型。

    质量闸专项测试（TestPersonRecheckGate / TestQualityGateSkip）自行再注入
    受控假检测器覆盖各分支。
    """

    def always_person(img: Image.Image) -> float:
        return 0.99

    monkeypatch.setattr(crop_scorers, "_get_person_rechecker", lambda: always_person)


class TestLoadMotCache:
    """mot_cache schema 校验：合法通过，损坏抛 SchemaError。"""

    def _write(self, tmp_path: Path, payload: Any) -> Path:  # noqa: ANN401 测试载荷
        p = tmp_path / "x_mot_cache.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_valid_passes(self, tmp_path: Path) -> None:
        # Arrange
        payload = {
            "frames": 2,
            "balls": [
                [
                    {
                        "conf": 0.8,
                        "box": [0, 0, 10, 10],
                        "cx": 5,
                        "cy": 5,
                        "sec": 0.0,
                        "frame_idx": 0,
                    }
                ],
                [],
            ],
            "persons": [[[0, 0, 100, 200]], []],
        }
        # Act
        cache = load_mot_cache(self._write(tmp_path, payload))
        # Assert
        assert cache.frames == 2
        assert cache.balls[0][0].cx == 5
        assert cache.balls[0][0].box == [0, 0, 10, 10]
        assert cache.persons[0][0] == Box(0, 0, 100, 200)

    def test_length_mismatch(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="长度不齐"):
            load_mot_cache(self._write(tmp_path, {"frames": 2, "balls": [[]], "persons": [[], []]}))

    def test_missing_top_keys(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            load_mot_cache(self._write(tmp_path, {"frames": 1, "balls": [[]]}))

    def test_bad_ball_field(self, tmp_path: Path) -> None:
        # Arrange
        payload = {"frames": 1, "balls": [[{"conf": "high"}]], "persons": [[]]}
        # Act / Assert
        with pytest.raises(SchemaError):
            load_mot_cache(self._write(tmp_path, payload))

    def test_bad_person_box(self, tmp_path: Path) -> None:
        # Arrange：框 x2<=x1 属数据损坏
        payload = {"frames": 1, "balls": [[]], "persons": [[[10, 10, 5, 20]]]}
        # Act / Assert
        with pytest.raises(SchemaError):
            load_mot_cache(self._write(tmp_path, payload))


class TestTrackWindowDets:
    """轨迹重链窗口取帧：[anchor−4.0, anchor+0.5]，裁剪到缓存边界。"""

    def test_normal_window(self) -> None:
        # Arrange：30 帧缓存，anchor=4.0 → [0.0, 4.5] → 帧 0..22
        cache = _empty_cache(30)
        # Act
        window = track_window_dets(cache, 4.0)
        # Assert
        assert len(window) == 23

    def test_clamped_at_start(self) -> None:
        # Arrange：anchor=1.0 → [max(0,−3), 1.5] → 帧 0..7
        cache = _empty_cache(30)
        # Act / Assert
        assert len(track_window_dets(cache, 1.0)) == 8

    def test_clamped_at_end(self) -> None:
        # Arrange：anchor=6.0，缓存仅 30 帧（到 5.8s）→ 右端夹到帧 29
        cache = _empty_cache(30)
        # Act / Assert
        assert len(track_window_dets(cache, 6.0)) == 20

    def test_window_keeps_frame_dets(self) -> None:
        # Arrange：帧 10 有一个球，窗口应带出来
        balls = [()] * 30
        balls[10] = (_ball(0.9, 100, 100, 10),)
        cache = _cache(balls, [()] * 30)
        # Act
        window = track_window_dets(cache, 4.0)
        # Assert：窗口起点帧 0，索引 10 对应窗口第 10 个
        assert len(window[10]) == 1
        assert window[10][0].cx == 100


class TestSelectGoalTrack:
    """进球轨迹选择：端点与候选锚点最近；超界/无轨迹 → None。"""

    def test_nearest_endpoint_wins(self) -> None:
        # Arrange：两条轨迹，t1 端点 (100,100)、t2 端点 (800,800)
        t1 = _track([(90, 100, 18), (100, 100, 19)])
        t2 = _track([(790, 800, 18), (800, 800, 19)])
        # Act / Assert：锚点 (110,100) → t1
        assert select_goal_track([t2, t1], 4.0, (110, 100)) is t1

    def test_too_far_returns_none(self) -> None:
        # Arrange：端点都在锚点 200px 外
        t1 = _track([(100, 100, 19)])
        t2 = _track([(800, 800, 19)])
        # Act / Assert
        assert select_goal_track([t1, t2], 4.0, (500, 500)) is None

    def test_no_anchor_xy_falls_back_to_time(self) -> None:
        # Arrange：无候选位置 → 端点时间距 anchor 最近者
        t1 = _track([(100, 100, 14)])  # end sec 2.8
        t2 = _track([(550, 50, 20)])  # end sec 4.0
        # Act / Assert
        assert select_goal_track([t1, t2], 4.0, None) is t2

    def test_empty_tracks(self) -> None:
        # Arrange / Act / Assert
        assert select_goal_track([], 4.0, (100, 100)) is None

    def test_prefer_longer_picks_long_track_within_slack(self) -> None:
        # Arrange：碎片端点距锚 10px，长轨迹端点距锚 90px（100px 滑窗内）
        frag = _track([(110, 100, 19)])
        long_t = _track([(500, 100, 15), (620, 100, 16), (740, 100, 17), (190, 100, 19)])
        # Act / Assert：prefer_longer → 选长轨迹；默认 False 行为不变 → 选碎片
        assert select_goal_track([frag, long_t], 4.0, (100, 100), prefer_longer=True) is long_t
        assert select_goal_track([frag, long_t], 4.0, (100, 100)) is frag

    def test_prefer_longer_respects_slack_bound(self) -> None:
        # Arrange：长轨迹端点距锚 120px，超滑窗（10+100=110）
        frag = _track([(110, 100, 19)])
        long_t = _track([(500, 100, 15), (620, 100, 16), (740, 100, 17), (220, 100, 19)])
        # Act / Assert：滑窗外不迁就，仍选碎片
        assert select_goal_track([frag, long_t], 4.0, (100, 100), prefer_longer=True) is frag

    def test_prefer_longer_still_enforces_max_dist(self) -> None:
        # Arrange：池内全部端点超 GOAL_TRACK_MAX_DIST_PX=200
        t1 = _track([(400, 100, 19)])
        t2 = _track([(500, 100, 15), (520, 100, 19)])
        # Act / Assert
        assert select_goal_track([t1, t2], 4.0, (100, 100), prefer_longer=True) is None

    def test_prefer_longer_time_branch_picks_long_track(self) -> None:
        # Arrange：碎片端点 4.0s（dt=0），长轨迹端点 3.4s（dt=0.6，1.0s 时间滑窗内）
        frag = _track([(1050, 50, 20)])
        long_t = _track([(100, 50, 13), (300, 50, 15), (600, 50, 17)])
        # Act / Assert：anchor_xy=None + prefer_longer → 选长轨迹；默认 False → 时间最近选碎片
        assert select_goal_track([frag, long_t], 4.0, None, prefer_longer=True) is long_t
        assert select_goal_track([frag, long_t], 4.0, None) is frag

    def test_prefer_longer_time_branch_respects_slack(self) -> None:
        # Arrange：长轨迹端点 2.6s（dt=1.4，超 0+1.0 时间滑窗）
        frag = _track([(1050, 50, 20)])
        long_t = _track([(100, 50, 9), (300, 50, 11), (600, 50, 13)])
        # Act / Assert：滑窗外不迁就，仍选时间最近的碎片
        assert select_goal_track([frag, long_t], 4.0, None, prefer_longer=True) is frag


class TestFindHeldBox:
    """持球点回放：从末端往回放找最后一个球心严格落在人框内的轨迹点。"""

    def test_last_held_point_wins(self) -> None:
        # Arrange：f0 在 A 内、f1 悬空、f2 在 B 内 → 取 f2/B（最后持球者）
        track = _track([(50, 50, 0), (300, 50, 1), (550, 50, 2)])
        persons = (
            (Box(0, 0, 100, 100), Box(500, 0, 600, 100)),
            (Box(0, 0, 100, 100), Box(500, 0, 600, 100)),
            (Box(0, 0, 100, 100), Box(500, 0, 600, 100)),
        )
        # Act
        got = find_held_box(track, persons)
        # Assert
        assert got is not None
        det, box = got
        assert det.frame_idx == 2
        assert box == Box(500, 0, 600, 100)

    def test_strictly_inside_no_margin(self) -> None:
        # Arrange：球心 (105,50) 在框外 5px（无 margin，不算持球）
        track = _track([(105, 50, 0)])
        persons = ((Box(0, 0, 100, 100),),)
        # Act / Assert
        assert find_held_box(track, persons) is None

    def test_no_held_returns_none(self) -> None:
        # Arrange：整轨球心都不在人框内
        track = _track([(300, 300, 0), (320, 300, 1)])
        persons = ((Box(0, 0, 100, 100),), (Box(0, 0, 100, 100),))
        # Act / Assert
        assert find_held_box(track, persons) is None


class TestStartNearestBox:
    """无持球回退：轨迹起点时刻离球心最近的人框。"""

    def test_nearest_person_at_start(self) -> None:
        # Arrange：起点 (50,300)，A 中心 (50,50) 距 250，B 中心 (550,50) 距 ~640
        track = _track([(50, 300, 0), (250, 300, 1)])
        persons = (
            (Box(0, 0, 100, 100), Box(500, 0, 600, 100)),
            (Box(0, 0, 100, 100), Box(500, 0, 600, 100)),
        )
        # Act
        got = start_nearest_box(track, persons)
        # Assert
        assert got is not None
        det, box = got
        assert det.frame_idx == 0
        assert box == Box(0, 0, 100, 100)

    def test_no_persons_returns_none(self) -> None:
        # Arrange / Act / Assert
        track = _track([(50, 300, 0)])
        assert start_nearest_box(track, ((),)) is None


class TestLocateScorer:
    """轨迹法端到端（合成缓存）：选轨 → 持球回放 → 回退 → SKIP。"""

    def _shot_cache(self, ball_y: int = 50) -> MotCache:
        """30 帧：进球轨迹帧 10..20 从 (50,ball_y) 每帧 +20px 移到 (250,ball_y)；
        另有静态干扰球 (550,50) 全程在 B 框内。人框 A(0,0,100,100)、B(500,0,600,100)。"""
        persons = [(Box(0, 0, 100, 100), Box(500, 0, 600, 100))] * 30
        balls: list[tuple[Detection, ...]] = [()] * 30
        for fi in range(10, 21):
            balls[fi] = (
                _ball(0.9, 50 + (fi - 10) * 20, ball_y, fi),
                _ball(0.5, 550, 50, fi),
            )
        return _cache(balls, persons)

    def test_held_point_wins_over_static_fp(self) -> None:
        # Arrange：anchor=4.0，候选锚点 = 进球轨迹端点 (250,50)
        cache = self._shot_cache()
        # Act
        result = locate_scorer(cache, 4.0, (250, 50))
        # Assert：干扰球全程在 B 内，但端点离锚点 300px → 不选它；
        # 回放最后持球点 = 帧 12（球心 x=90 仍在 A 内，帧 13 起 x=110 出框）
        assert result.status == "OK"
        assert result.reason == ""
        assert result.box == Box(0, 0, 100, 100)
        assert result.frame_idx == 12
        assert result.votes == 11
        assert result.total_votes == 2

    def test_no_held_falls_back_to_start(self) -> None:
        # Arrange：球沿 y=300 移动（全程不在任何人框内）
        cache = self._shot_cache(ball_y=300)
        # Act
        result = locate_scorer(cache, 4.0, (250, 300))
        # Assert：无持球点 → 起点 (50,300)@帧10 最近人框 A
        assert result.status == "OK"
        assert result.reason == "start_fallback"
        assert result.box == Box(0, 0, 100, 100)
        assert result.frame_idx == 10

    def test_no_persons_skip(self) -> None:
        # Arrange：有球轨但全程无人框
        balls: list[tuple[Detection, ...]] = [()] * 30
        for fi in range(10, 21):
            balls[fi] = (_ball(0.9, 50 + (fi - 10) * 20, 300, fi),)
        cache = _cache(balls, [()] * 30)
        # Act
        result = locate_scorer(cache, 4.0, (250, 300))
        # Assert：无持球 + 起点帧无人 → SKIP
        assert result.status == "SKIP"
        assert result.reason == "no_person"

    def test_no_tracks_skip(self) -> None:
        # Arrange：窗口内全程无球
        cache = _empty_cache(30)
        # Act
        result = locate_scorer(cache, 4.0, (250, 50))
        # Assert
        assert result.status == "SKIP"
        assert result.reason == "no_track"

    def test_far_endpoint_skip(self) -> None:
        # Arrange：轨迹存在，但端点离候选锚点 >200px
        cache = self._shot_cache()
        # Act
        result = locate_scorer(cache, 4.0, (1500, 900))
        # Assert
        assert result.status == "SKIP"
        assert result.reason == "no_track_near_anchor"

    def test_time_fallback_without_anchor_xy(self) -> None:
        # Arrange：无候选位置；A 球帧 10..14，B 球帧 10..20（端点 4.0 离 anchor 最近）
        persons = [(Box(0, 0, 100, 100), Box(500, 0, 600, 100))] * 30
        balls: list[tuple[Detection, ...]] = [()] * 30
        for fi in range(10, 21):
            frame_balls = [_ball(0.5, 550, 50, fi)]
            if fi <= 14:
                frame_balls.append(_ball(0.9, 50, 50, fi))
            balls[fi] = tuple(frame_balls)
        cache = _cache(balls, persons)
        # Act
        result = locate_scorer(cache, 4.0, None)
        # Assert：选中 B 轨迹（端点时间最近），B 球全程在 B 框内 → 持球点命中
        assert result.status == "OK"
        assert result.box == Box(500, 0, 600, 100)


class TestCandidatesIndex:
    """candidates.json 锚点索引：fid+t0 匹配取 cx/cy。"""

    def _write(self, tmp_path: Path, payload: Any) -> Path:  # noqa: ANN401 测试载荷
        p = tmp_path / "candidates.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_match_within_tolerance(self, tmp_path: Path) -> None:
        # Arrange
        path = self._write(
            tmp_path,
            [
                {
                    "t0": 14.1,
                    "dur": 0.6,
                    "ac": 0.6,
                    "cx": 1629,
                    "cy": 704,
                    "src": "rejoin",
                    "fid": "a_video",
                    "label": "#7",
                },
                {
                    "t0": 13.8,
                    "dur": 1.4,
                    "ac": 0.4,
                    "cx": 1387,
                    "cy": 485,
                    "src": "static",
                    "fid": "a_video",
                    "label": "#6",
                },
            ],
        )
        # Act
        index = load_candidates_index(path)
        # Assert：anchor=14.1 精确匹配 #7
        assert match_anchor_xy(index, "a_video", 14.1) == (1629, 704)
        # anchor=13.9 在两者容差内，取更近的 #6（|0.1| < |0.2|）
        assert match_anchor_xy(index, "a_video", 13.9) == (1387, 485)

    def test_no_match_returns_none(self) -> None:
        # Arrange
        index = {"a_video": [(14.1, 100, 200)]}
        # Act / Assert：超容差 / 异 fid 都None
        assert match_anchor_xy(index, "a_video", 16.0) is None
        assert match_anchor_xy(index, "b_video", 14.1) is None

    def test_bad_schema_raises(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            load_candidates_index(self._write(tmp_path, {"not": "list"}))
        with pytest.raises(SchemaError):
            load_candidates_index(self._write(tmp_path, [{"t0": 1.0}]))


class TestCrop:
    """裁图：外扩 20%、夹取边界、短边放大到 400px。"""

    def test_expand_box_ratio(self) -> None:
        # Arrange：100x200 框外扩 20% → 每侧 10px / 20px
        # Act
        x1, y1, x2, y2 = expand_box(Box(100, 100, 200, 300), 0.2, 1000, 1000)
        # Assert
        assert (x1, y1, x2, y2) == (90, 80, 210, 320)

    def test_expand_box_clamped(self) -> None:
        # Arrange / Act
        x1, y1, x2, y2 = expand_box(Box(0, 0, 50, 50), 0.2, 60, 60)
        # Assert：越界侧被夹到图像边缘
        assert (x1, y1, x2, y2) == (0, 0, 55, 55)

    def test_crop_upscales_short_side(self, tmp_path: Path) -> None:
        # Arrange：1000x800 图，框 100x150 → 外扩后 120x180 → 短边 120 <400 → 放大
        img = tmp_path / "f_00001.jpg"
        Image.new("RGB", (1000, 800), (10, 20, 30)).save(img)
        out = tmp_path / "out" / "crop.jpg"
        # Act
        crop_and_save(img, Box(100, 100, 200, 250), out)
        # Assert
        with Image.open(out) as im:
            assert min(im.size) == 400
            assert im.size == (400, 600)

    def test_crop_no_upscale_when_large(self, tmp_path: Path) -> None:
        # Arrange：框 400x600 → 外扩 480x720（不触边界），短边 ≥400 不放大
        img = tmp_path / "f_00001.jpg"
        Image.new("RGB", (1000, 800), (10, 20, 30)).save(img)
        out = tmp_path / "out" / "crop.jpg"
        # Act
        crop_and_save(img, Box(100, 100, 500, 700), out)
        # Assert
        with Image.open(out) as im:
            assert im.size == (480, 720)


class TestClassifyTeam:
    """颜色分队：纯色三分类，近阈/彩色归便服。"""

    def _img(self, tmp_path: Path, rgb: tuple[int, int, int], name: str = "c.jpg") -> Path:
        p = tmp_path / name
        Image.new("RGB", (400, 600), rgb).save(p)
        return p

    def test_black(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        assert classify_team(self._img(tmp_path, (15, 15, 15))) == "黑"

    def test_white(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        assert classify_team(self._img(tmp_path, (240, 240, 240))) == "白"

    def test_mid_gray_is_casual(self, tmp_path: Path) -> None:
        # Arrange：V=128 落在黑/白阈值之间（近阈）→ 便服
        assert classify_team(self._img(tmp_path, (128, 128, 128))) == "便服"

    def test_saturated_bright_is_casual(self, tmp_path: Path) -> None:
        # Arrange：亮红 V 高但 S 高 → 不归白
        assert classify_team(self._img(tmp_path, (220, 30, 30))) == "便服"


class TestCliEndToEnd:
    """CLI 端到端：合成 goals + mot_cache + 帧图，验证产物与退出码。"""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """搭一套临时 goals/detect/frames/out，返回 (goals, detectdir, framesdir, out)。

        a_video@2.0：球全程静止在 A 框内 → 轨迹法 OK；b_video@1.0：空缓存 →
        no_track SKIP；a_video@5.0 rejected 不处理。
        """
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps(
                {
                    "session": "test",
                    "goals": [
                        {"file": "a_video.mp4", "anchor_time": 2.0, "status": "confirmed"},
                        {"file": "b_video.mp4", "anchor_time": 1.0, "status": "confirmed"},
                        {"file": "a_video.mp4", "anchor_time": 5.0, "status": "rejected"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        detectdir = tmp_path / "detect"
        detectdir.mkdir()
        persons = [[[0, 0, 100, 100], [500, 0, 600, 100]]] * 9
        balls = [
            [
                {
                    "conf": 0.9,
                    "box": [40, 40, 60, 60],
                    "cx": 50,
                    "cy": 50,
                    "sec": i / 5,
                    "frame_idx": i,
                }
            ]
            for i in range(9)
        ]
        (detectdir / "a_video_mot_cache.json").write_text(
            json.dumps({"frames": 9, "balls": balls, "persons": persons}), encoding="utf-8"
        )
        (detectdir / "b_video_mot_cache.json").write_text(
            json.dumps({"frames": 9, "balls": [[]] * 9, "persons": [[]] * 9}),
            encoding="utf-8",
        )
        framesdir = tmp_path / "frames"
        (framesdir / "a_video").mkdir(parents=True)
        for i in range(9):
            Image.new("RGB", (1000, 800), (20, 20, 20)).save(
                framesdir / "a_video" / f"f_{i + 1:05d}.jpg"
            )
        out = tmp_path / "out"
        return goals, detectdir, framesdir, out

    def test_end_to_end(self, tmp_path: Path) -> None:
        # Arrange
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        assert payload["session"] == "test"
        entries = payload["candidates"]
        assert len(entries) == 2  # rejected 不处理
        ok = entries[0]
        assert ok["key"] == "a_video.mp4#2.0"
        assert ok["status"] == "OK"
        assert ok["team_guess"] == "黑"  # 帧图近黑
        assert (out / ok["crop"]).is_file()
        skip = entries[1]
        assert skip["key"] == "b_video.mp4#1.0"
        assert skip["status"] == "SKIP"
        assert skip["reason"] == "no_track"
        assert skip["crop"] == ""

    def test_missing_cache_exit_1(self, tmp_path: Path) -> None:
        # Arrange
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        for p in detectdir.glob("*.json"):
            p.unlink()
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert：素材缺失 → 逐条 SKIP 但退出码非零（产出型脚本口径）
        assert rc == 1
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        assert all(e["status"] == "SKIP" for e in payload["candidates"])
        assert all(e["reason"] == "missing_cache" for e in payload["candidates"])

    def test_candidates_arg_feeds_anchor_xy(self, tmp_path: Path) -> None:
        # Arrange：给 --candidates，a_video@2.0 的锚点 (50,50) 匹配上静态球轨迹
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        candidates = tmp_path / "candidates.json"
        candidates.write_text(
            json.dumps(
                [
                    {
                        "t0": 2.0,
                        "dur": 0.8,
                        "ac": 0.9,
                        "cx": 50,
                        "cy": 50,
                        "src": "static",
                        "fid": "a_video",
                        "label": "#1",
                    }
                ]
            ),
            encoding="utf-8",
        )
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
                "--candidates",
                str(candidates),
            ]
        )
        # Assert：照常 OK（锚点命中静态轨迹，持球点在 A 框）
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        assert payload["candidates"][0]["status"] == "OK"

    def test_read_numbers_cache_only_cli(self, tmp_path: Path) -> None:
        # Arrange：--read-numbers + --numbers-cache-only（空缓存 → 全跳票，零新调用零凭证）
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
                "--read-numbers",
                "--numbers-cache-only",
            ]
        )
        # Assert：OK 球逐张记 skipped、number_guess 置空；SKIP 球 number_votes=None；
        # 零新调用不落缓存文件
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok, skip = payload["candidates"]
        assert ok["number_guess"] is None
        assert [v["source"] for v in ok["number_votes"]] == ["skipped"] * len(ok["crops"])
        assert skip["number_votes"] is None
        assert not (out / "number_cache.json").exists()


class TestPreviewClip:
    """--rawdir 认人预览片段：窗口夹取、路径组装、ffmpeg 参数、失败容错。"""

    def test_preview_window_normal(self) -> None:
        # Arrange / Act / Assert：锚点前 4s 后 2s
        assert preview_window(16.5) == (12.5, 18.5)

    def test_preview_window_clamped_at_zero(self) -> None:
        # Arrange / Act / Assert：anchor<4s 起点夹取到 0，不越界
        assert preview_window(1.3) == (0.0, 3.3)
        assert preview_window(4.0) == (0.0, 6.0)

    def test_cut_preview_clip_ffmpeg_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：拦截 run_ffmpeg 验证参数
        calls: list[dict[str, Any]] = []

        def fake_run_ffmpeg(
            args: list[str], *, timeout_sec: int = 600, retries: int = 1, backoff_sec: float = 5.0
        ) -> None:
            calls.append({"args": args, "timeout_sec": timeout_sec})

        monkeypatch.setattr("crop_scorers.run_ffmpeg", fake_run_ffmpeg)
        rawdir = tmp_path / "raw"
        rawdir.mkdir()
        (rawdir / "a_video.mp4").write_bytes(b"")
        # Act
        cut_preview_clip(rawdir, "a_video.mp4", 16.5, tmp_path / "out" / "clips" / "x.mp4")
        # Assert：输入侧 -ss/-to、1280 宽、libx264 crf26 veryfast、无声、超时下限 120s
        assert len(calls) == 1
        args = calls[0]["args"]
        assert args[:4] == ["-ss", "12.50", "-to", "18.50"]
        assert args[args.index("-i") + 1].endswith("a_video.mp4")
        assert "scale=1280:-2" in args
        assert args[args.index("-crf") + 1] == "26"
        assert args[args.index("-preset") + 1] == "veryfast"
        assert "-an" in args
        assert calls[0]["timeout_sec"] == 120

    def test_try_cut_preview_no_rawdir(self, tmp_path: Path) -> None:
        # Arrange / Act：未给 --rawdir 不切片
        clip, failed = _try_cut_preview(
            {"file": "a.mp4", "anchor_time": 2.0}, None, tmp_path, "a.mp4#2.0"
        )
        # Assert
        assert clip == ""
        assert failed is False

    def test_try_cut_preview_missing_src(self, tmp_path: Path) -> None:
        # Arrange：原片不存在
        rawdir = tmp_path / "raw"
        rawdir.mkdir()
        # Act
        clip, failed = _try_cut_preview(
            {"file": "ghost.mp4", "anchor_time": 2.0}, rawdir, tmp_path, "ghost.mp4#2.0"
        )
        # Assert：记失败但返回（不炸整批），调用方计入缺失错误使退出码非零
        assert clip == ""
        assert failed is True

    def test_try_cut_preview_ffmpeg_failure_tolerated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：ffmpeg 重试耗尽抛错
        def fake_run_ffmpeg(
            args: list[str], *, timeout_sec: int = 600, retries: int = 1, backoff_sec: float = 5.0
        ) -> None:
            raise BasketballPipelineError("boom")

        monkeypatch.setattr("crop_scorers.run_ffmpeg", fake_run_ffmpeg)
        rawdir = tmp_path / "raw"
        rawdir.mkdir()
        (rawdir / "a.mp4").write_bytes(b"")
        # Act
        clip, failed = _try_cut_preview(
            {"file": "a.mp4", "anchor_time": 2.0}, rawdir, tmp_path, "a.mp4#2.0"
        )
        # Assert：失败容错，不抛出
        assert clip == ""
        assert failed is True

    def test_try_cut_preview_success_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：ffmpeg 成功（假造输出文件）
        def fake_run_ffmpeg(
            args: list[str], *, timeout_sec: int = 600, retries: int = 1, backoff_sec: float = 5.0
        ) -> None:
            Path(args[-1]).write_bytes(b"mp4")

        monkeypatch.setattr("crop_scorers.run_ffmpeg", fake_run_ffmpeg)
        rawdir = tmp_path / "raw"
        rawdir.mkdir()
        (rawdir / "a_video.mp4").write_bytes(b"")
        # Act
        clip, failed = _try_cut_preview(
            {"file": "a_video.mp4", "anchor_time": 16.5}, rawdir, tmp_path, "k"
        )
        # Assert：相对路径 clips/<fid>_t<anchor:.1f>.mp4，正斜杠
        assert clip == "clips/a_video_t16.5.mp4"
        assert failed is False
        assert (tmp_path / "clips" / "a_video_t16.5.mp4").is_file()


class TestCliPreviewClip:
    """CLI 端到端 --rawdir：SKIP 球也切预览片段，clip 字段写入 candidates。"""

    def test_skip_goal_also_gets_clip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：复用端到端场景（a_video OK 球 + b_video no_track SKIP 球）
        setup = TestCliEndToEnd()
        goals, detectdir, framesdir, out = setup._setup(tmp_path)
        rawdir = tmp_path / "raw"
        rawdir.mkdir()
        (rawdir / "a_video.mp4").write_bytes(b"")
        (rawdir / "b_video.mp4").write_bytes(b"")

        def fake_run_ffmpeg(
            args: list[str], *, timeout_sec: int = 600, retries: int = 1, backoff_sec: float = 5.0
        ) -> None:
            Path(args[-1]).write_bytes(b"mp4")

        monkeypatch.setattr("crop_scorers.run_ffmpeg", fake_run_ffmpeg)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
                "--rawdir",
                str(rawdir),
            ]
        )
        # Assert：两球都有预览片段（SKIP 球也切），退出码 0
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok, skip = payload["candidates"]
        assert ok["clip"] == "clips/a_video_t2.0.mp4"
        assert skip["status"] == "SKIP"
        assert skip["clip"] == "clips/b_video_t1.0.mp4"
        assert (out / skip["clip"]).is_file()

    def test_clip_failure_exit_1_but_candidates_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：原片缺失 → 切片失败，但定位/裁图照常产出
        setup = TestCliEndToEnd()
        goals, detectdir, framesdir, out = setup._setup(tmp_path)
        rawdir = tmp_path / "raw"
        rawdir.mkdir()  # 空目录：a_video.mp4 / b_video.mp4 不存在
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
                "--rawdir",
                str(rawdir),
            ]
        )
        # Assert：缺失可观测（退出 1），candidates 照常落盘、clip 为空
        assert rc == 1
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        assert all(e["clip"] == "" for e in payload["candidates"])
        assert payload["candidates"][0]["status"] == "OK"


class _FakeResp:
    """伪造 httpx.Response（号码识别测试用，不碰网络）。"""

    def __init__(self, status: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self) -> dict:
        """返回预设响应体。"""
        return self._payload


class _FakeClient:
    """伪造 httpx.Client：按脚本依次返回响应或抛异常。"""

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls = 0

    def post(self, url: str, **kwargs: object) -> _FakeResp:
        """按脚本返回；异常项则抛出。"""
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, _FakeResp)
        return item


def _k3_ok_payload(guess_json: str, total_tokens: int = 100) -> dict:
    """构造 K3 200 响应体（content 为给定 JSON 文本）。"""
    return {
        "choices": [{"message": {"content": guess_json}}],
        "usage": {"total_tokens": total_tokens},
    }


def _crop_file(tmp_path: Path) -> Path:
    """造一张真实裁图文件。"""
    p = tmp_path / "crop.jpg"
    Image.new("RGB", (100, 100), (30, 30, 30)).save(p)
    return p


class TestParseNumberAnswer:
    """号码识别回复解析：容错与字段归一。"""

    def test_good_json(self) -> None:
        # Arrange / Act
        guess = parse_number_answer(
            '{"number": "21", "color": "黑", "name_text": "测试员甲", "confidence": "high"}'
        )
        # Assert
        assert guess == NumberGuess(
            number="21", color="黑", name_text="测试员甲", confidence="high"
        )

    def test_markdown_fence_and_prose(self) -> None:
        # Arrange：带围栏和前后废话
        raw = '好的，结果如下：\n```json\n{"number": 7, "color": "白", "name_text": null, '
        raw += '"confidence": "low"}\n```\n以上。'
        # Act
        guess = parse_number_answer(raw)
        # Assert：int 号码归一为 str
        assert guess is not None
        assert guess.number == "7"
        assert guess.confidence == "low"

    def test_bad_json_returns_none(self) -> None:
        # Arrange / Act / Assert
        assert parse_number_answer("这不是 JSON") is None
        assert parse_number_answer('{"number": "21", 坏掉') is None

    def test_field_normalization(self) -> None:
        # Arrange / Act
        guess = number_guess_from_dict(
            {"number": "2a", "color": "灰", "name_text": "", "confidence": "LOW"}
        )
        # Assert：非纯数字号码→None；非法颜色→None；空名字→None；非 high→low
        assert guess is not None
        assert guess.number is None
        assert guess.color is None
        assert guess.name_text is None
        assert guess.confidence == "low"

    def test_non_dict_returns_none(self) -> None:
        # Arrange / Act / Assert
        assert number_guess_from_dict([1, 2]) is None
        assert number_guess_from_dict("x") is None


class TestNumberCache:
    """number_cache.json 读写幂等与版本失效。"""

    def test_roundtrip(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "number_cache.json"
        results = {"a.mp4#4.1": {"number": "21", "color": "黑", "confidence": "high"}}
        # Act
        save_number_cache(path, results)
        got = load_number_cache(path)
        # Assert
        assert got["a.mp4#4.1"]["number"] == "21"

    def test_missing_file_empty(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert
        assert load_number_cache(tmp_path / "nope.json") == {}

    def test_prompt_version_mismatch_invalidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：写入后改 prompt 版本常量 → 缓存作废
        path = tmp_path / "number_cache.json"
        save_number_cache(path, {"k": {"number": "21"}})
        monkeypatch.setattr("crop_scorers.NUMBER_PROMPT_VERSION", "number-v999")
        # Act / Assert
        assert load_number_cache(path) == {}


class TestReadNumber:
    """read_number 调用容错（假 client，不碰网络）。"""

    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setattr("vlm_client.load_token", lambda force=False: "tok")
        client = _FakeClient(
            [_FakeResp(200, _k3_ok_payload('{"number":"21","color":"黑","confidence":"high"}'))]
        )
        # Act
        guess, tokens, err = read_number(client, crop_path=_crop_file(tmp_path), key="k")  # type: ignore[arg-type]
        # Assert
        assert err == ""
        assert guess is not None and guess.number == "21"
        assert tokens == 100
        assert client.calls == 1

    def test_network_error_tolerated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange：全程网络错误（重试耗尽）
        monkeypatch.setattr("vlm_client.load_token", lambda force=False: "tok")
        client = _FakeClient([httpx.ConnectError("boom")])
        # Act
        guess, _, err = read_number(client, crop_path=_crop_file(tmp_path), key="k")  # type: ignore[arg-type]
        # Assert：不炸，返回错误摘要；重试口径 = K3_HTTP_RETRY+1 次
        assert guess is None
        assert "网络错误" in err
        assert client.calls == 3

    def test_401_reload_token_then_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：先 401 后 200；load_token/sleep 打桩（不碰真凭证、不真睡）
        monkeypatch.setattr("vlm_client.load_token", lambda force=False: "tok")
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        client = _FakeClient(
            [
                _FakeResp(401, text="unauthorized"),
                _FakeResp(200, _k3_ok_payload('{"number":null,"color":"白","confidence":"low"}')),
            ]
        )
        # Act
        guess, _, err = read_number(client, crop_path=_crop_file(tmp_path), key="k")  # type: ignore[arg-type]
        # Assert
        assert err == ""
        assert guess is not None and guess.number is None and guess.color == "白"
        assert client.calls == 2

    def test_unparseable_reply_is_err(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：200 但回复不是号码 JSON
        monkeypatch.setattr("vlm_client.load_token", lambda force=False: "tok")
        client = _FakeClient([_FakeResp(200, _k3_ok_payload("看不清"))])
        # Act
        guess, _, err = read_number(client, crop_path=_crop_file(tmp_path), key="k")  # type: ignore[arg-type]
        # Assert
        assert guess is None
        assert "解析" in err

    def test_missing_crop_is_err(self, tmp_path: Path) -> None:
        # Arrange / Act
        guess, _, err = read_number(
            _FakeClient([]),
            crop_path=tmp_path / "ghost.jpg",
            key="k",  # type: ignore[arg-type]
        )
        # Assert：裁图缺失不发请求
        assert guess is None
        assert "读取裁图失败" in err


class TestApplyNumberReading:
    """apply_number_reading：闸按裁图张数计、迁移后缓存命中零调用、失败不写缓存。"""

    def _entries(self, outdir: Path, n: int, crops_per_goal: int = 1) -> list[dict[str, Any]]:
        """造 n 条 OK 候选（真实裁图文件，逐张噪点图 → md5 互异，crops 多裁字段齐全）。"""
        entries: list[dict[str, Any]] = []
        idx: int = 0
        for i in range(n):
            names: list[str] = []
            for _ in range(crops_per_goal):
                name = f"c{idx}.jpg"
                arr = np.random.default_rng(idx).integers(0, 256, (100, 100, 3), dtype=np.uint8)
                Image.fromarray(arr).save(outdir / name)
                names.append(name)
                idx += 1
            entries.append(
                {
                    "key": f"a.mp4#{i}.0",
                    "status": "OK",
                    "crop": names[0],
                    "crops": names,
                    "number_guess": None,
                    "number_votes": None,
                }
            )
        return entries

    def test_over_20_fresh_rejected(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert：21 张新识别 > 20 → 显式失败（spec：先问用户）
        with pytest.raises(ExternalApiError, match="问用户"):
            apply_number_reading(self._entries(tmp_path, 21), tmp_path)

    def test_gate_counts_crops_not_goals(self, tmp_path: Path) -> None:
        # Arrange：5 球 × 3 裁 = 15 张互异裁图（多帧投票后额度按张数计）
        entries = self._entries(tmp_path, 5, crops_per_goal=3)
        # Act / Assert：15 > 10 → 闸拒绝
        with pytest.raises(ExternalApiError, match="问用户"):
            apply_number_reading(entries, tmp_path, max_reads=10)

    def test_cache_hit_no_http(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange：旧 goal key 缓存（迁移重键后应命中）；裁图用旧 schema（无 crops 字段）
        crop = tmp_path / "c.jpg"
        Image.new("RGB", (100, 100), (30, 30, 30)).save(crop)
        save_number_cache(
            tmp_path / "number_cache.json",
            {"a.mp4#0.0": {"number": "21", "color": "黑", "confidence": "high"}},
        )

        def _no_token(force: bool = False) -> str:
            raise RuntimeError("不应调用 load_token（全缓存命中零新调用）")

        monkeypatch.setattr("vlm_client.load_token", _no_token)
        entries: list[dict[str, Any]] = [
            {"key": "a.mp4#0.0", "status": "OK", "crop": "c.jpg", "number_guess": None}
        ]
        # Act
        n_fresh, tokens = apply_number_reading(entries, tmp_path)
        # Assert：零新调用，旧 goal key 迁移到 md5 后命中，number_guess 从缓存补齐
        assert n_fresh == 0
        assert tokens == 0
        assert entries[0]["number_guess"]["number"] == "21"
        assert entries[0]["number_votes"][0]["source"] == "cache"
        cache = load_number_cache(tmp_path / "number_cache.json")
        assert file_md5(crop) in cache  # 迁移重键落盘
        assert "a.mp4#0.0" in cache  # 旧键保留一轮

    def test_fresh_reads_voted_and_cached_by_md5(self, tmp_path: Path) -> None:
        # Arrange：1 球 3 裁，假 reader 按文件名给票：7 high / 7 low / 21 low
        entries = self._entries(tmp_path, 1, crops_per_goal=3)
        answers = {
            "c0.jpg": (NumberGuess("7", "黑", None, "high"), 100, ""),
            "c1.jpg": (NumberGuess("7", "黑", None, "low"), 100, ""),
            "c2.jpg": (NumberGuess("21", "黑", None, "low"), 100, ""),
        }
        calls: list[str] = []

        def fake_reader(path: Path, key: str) -> tuple[NumberGuess | None, int, str]:
            calls.append(path.name)
            return answers[path.name]

        # Act
        n_fresh, tokens = apply_number_reading(entries, tmp_path, reader=fake_reader)
        # Assert：同号 ≥2 → 采纳 7；3 张全发新调用；缓存键 = 裁图 md5
        assert n_fresh == 3
        assert tokens == 300
        assert calls == ["c0.jpg", "c1.jpg", "c2.jpg"]
        assert entries[0]["number_guess"]["number"] == "7"
        assert [v["source"] for v in entries[0]["number_votes"]] == ["fresh"] * 3
        cache = load_number_cache(tmp_path / "number_cache.json")
        assert set(cache) == {file_md5(tmp_path / n) for n in ("c0.jpg", "c1.jpg", "c2.jpg")}

    def test_cache_only_zero_calls(self, tmp_path: Path) -> None:
        # Arrange：跳票模式 + 空缓存；reader 被调用即炸（证明零新调用）
        entries = self._entries(tmp_path, 1, crops_per_goal=3)

        def forbidden_reader(path: Path, key: str) -> tuple[NumberGuess | None, int, str]:
            raise AssertionError("跳票模式不应发起任何调用")

        # Act
        n_fresh, tokens = apply_number_reading(
            entries, tmp_path, cache_only=True, reader=forbidden_reader
        )
        # Assert：无票可投 → number_guess None，逐张记 skipped，不落缓存文件
        assert n_fresh == 0
        assert tokens == 0
        assert entries[0]["number_guess"] is None
        assert [v["source"] for v in entries[0]["number_votes"]] == ["skipped"] * 3
        assert not (tmp_path / "number_cache.json").exists()

    def test_cache_only_votes_from_existing_cache(self, tmp_path: Path) -> None:
        # Arrange：跳票模式；3 裁中 2 张已有缓存票（7 high + 7 low），第 3 张未命中
        entries = self._entries(tmp_path, 1, crops_per_goal=3)
        cache = {
            file_md5(tmp_path / "c0.jpg"): {"number": "7", "confidence": "high"},
            file_md5(tmp_path / "c1.jpg"): {"number": "7", "confidence": "low"},
        }
        save_number_cache(tmp_path / "number_cache.json", cache)

        def forbidden_reader(path: Path, key: str) -> tuple[NumberGuess | None, int, str]:
            raise AssertionError("跳票模式不应发起任何调用")

        # Act
        n_fresh, _ = apply_number_reading(
            entries, tmp_path, cache_only=True, reader=forbidden_reader
        )
        # Assert：同号 ≥2 → 采纳 7；第 3 张 skipped；零新调用
        assert n_fresh == 0
        assert entries[0]["number_guess"]["number"] == "7"
        assert [v["source"] for v in entries[0]["number_votes"]] == [
            "cache",
            "cache",
            "skipped",
        ]

    def test_reader_error_not_cached_and_vote_continues(self, tmp_path: Path) -> None:
        # Arrange：3 裁中第 2 张识别失败（网络错误），其余两票同号 7
        entries = self._entries(tmp_path, 1, crops_per_goal=3)
        answers = {
            "c0.jpg": (NumberGuess("7", None, None, "high"), 100, ""),
            "c1.jpg": (None, 0, "网络错误: boom"),
            "c2.jpg": (NumberGuess("7", None, None, "low"), 100, ""),
        }

        def fake_reader(path: Path, key: str) -> tuple[NumberGuess | None, int, str]:
            return answers[path.name]

        # Act
        n_fresh, _ = apply_number_reading(entries, tmp_path, reader=fake_reader)
        # Assert：失败张不写缓存（下次重跑重试）、记 error；余下两票同号 ≥2 → 采纳 7
        assert n_fresh == 3
        assert entries[0]["number_guess"]["number"] == "7"
        assert [v["source"] for v in entries[0]["number_votes"]] == ["fresh", "error", "fresh"]
        cache = load_number_cache(tmp_path / "number_cache.json")
        assert file_md5(tmp_path / "c1.jpg") not in cache


class TestVoteNumberGuess:
    """众数投票规则（scorer-reid spec 写死）：同号≥2 采纳 / 单票 high 采纳 low 归 None /
    全不同取唯一 high / None 票不参与计数 / 全 None 归 None+low。"""

    def _g(self, number: str | None, conf: str = "high") -> NumberGuess:
        return NumberGuess(number=number, color=None, name_text=None, confidence=conf)

    def test_majority_adopts(self) -> None:
        # Arrange / Act：7 两票（含 low）+ 21 一票
        got = vote_number_guess([self._g("7", "low"), self._g("21"), self._g("7", "low")])
        # Assert：同号 ≥2 采纳该号（返回首张同号票原样）
        assert got is not None
        assert got.number == "7"

    def test_single_high_adopts(self) -> None:
        # Arrange / Act / Assert
        got = vote_number_guess([self._g("7")])
        assert got is not None
        assert got.number == "7"
        assert got.confidence == "high"

    def test_single_low_discards(self) -> None:
        # Arrange / Act / Assert：单票 low → 归 None+low
        got = vote_number_guess([self._g("7", "low")])
        assert got is not None
        assert got.number is None
        assert got.confidence == "low"

    def test_none_votes_not_counted(self) -> None:
        # Arrange / Act：[7, null, null] → 有效票=1，走单票路径（high 采纳）
        got = vote_number_guess([self._g("7"), self._g(None), self._g(None)])
        # Assert
        assert got is not None
        assert got.number == "7"

    def test_all_different_unique_high_wins(self) -> None:
        # Arrange / Act：7 low / 21 high / 33 low 全不同 → 取唯一 high
        got = vote_number_guess([self._g("7", "low"), self._g("21"), self._g("33", "low")])
        # Assert
        assert got is not None
        assert got.number == "21"

    def test_all_different_multiple_highs_discards(self) -> None:
        # Arrange / Act / Assert：两个 high → 不采，归 None+low
        got = vote_number_guess([self._g("7"), self._g("21")])
        assert got is not None
        assert got.number is None
        assert got.confidence == "low"

    def test_all_none_returns_none_low(self) -> None:
        # Arrange / Act / Assert：有效票=0 → None+low
        got = vote_number_guess([self._g(None), self._g(None, "low")])
        assert got is not None
        assert got.number is None
        assert got.confidence == "low"

    def test_empty_returns_none(self) -> None:
        # Arrange / Act / Assert：无票（全失败/全跳票）→ None（调用方置空 number_guess）
        assert vote_number_guess([]) is None


class TestMigrateNumberCache:
    """旧 goal key → crops[0] 裁图 md5 重键：旧键保留、幂等、查不到 INFO、缺失 WARNING。"""

    def _entry(self, key: str, crops: list[str]) -> dict[str, Any]:
        return {"key": key, "status": "OK", "crop": crops[0], "crops": crops}

    def test_rekey_keeps_old_key_and_idempotent(self, tmp_path: Path) -> None:
        # Arrange
        crop = tmp_path / "c.jpg"
        Image.new("RGB", (50, 50), (1, 2, 3)).save(crop)
        md5: str = file_md5(crop)
        cache = {"a.mp4#2.0": {"number": "7", "confidence": "high"}}
        entries = [self._entry("a.mp4#2.0", ["c.jpg"])]
        # Act
        migrated, changed = migrate_number_cache(cache, entries, tmp_path)
        # Assert：md5 新键重键成功，旧 goal key 保留一轮（spec：迁移是重键不是删除）
        assert changed is True
        assert migrated[md5]["number"] == "7"
        assert migrated["a.mp4#2.0"]["number"] == "7"
        # 幂等：二次执行零变化
        again, changed2 = migrate_number_cache(migrated, entries, tmp_path)
        assert changed2 is False
        assert again == migrated

    def test_unknown_key_kept_with_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：旧 key 在当前 run entries 查不到（删球/子集重跑）
        cache = {"ghost.mp4#1.0": {"number": "7"}}
        # Act
        with caplog.at_level(logging.INFO, logger="crop_scorers"):
            migrated, changed = migrate_number_cache(cache, [], tmp_path)
        # Assert：原样保留 + INFO，不删不改
        assert migrated == cache
        assert changed is False
        assert any(
            "原样保留" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records
        )

    def test_missing_crop_warns_and_keeps(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：entry 查得到但 crops[0] 文件缺失，算不出 md5
        cache = {"a.mp4#2.0": {"number": "7"}}
        entries = [self._entry("a.mp4#2.0", ["ghost.jpg"])]
        # Act
        with caplog.at_level(logging.WARNING, logger="crop_scorers"):
            migrated, changed = migrate_number_cache(cache, entries, tmp_path)
        # Assert：WARNING + 保留原 key，不炸
        assert migrated == cache
        assert changed is False
        assert any(
            "crops[0]" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records
        )

    def test_md5_keys_untouched(self, tmp_path: Path) -> None:
        # Arrange：已是 md5 的键不迁移（幂等前提）
        cache = {"0" * 32: {"number": "7"}}
        # Act
        migrated, changed = migrate_number_cache(cache, [], tmp_path)
        # Assert
        assert migrated == cache
        assert changed is False


def _moving_persons(frames: int, *, skip_frames: tuple[int, ...] = ()) -> list[tuple[Box, ...]]:
    """构造逐帧人框序列：A 框 (0,0,100,200) 每帧右移 10px（相邻 IoU≈0.82 链上），
    B 框固定远处 (800,0,900,200) 作干扰；skip_frames 里的帧无人（空帧）。"""
    persons: list[tuple[Box, ...]] = []
    for fi in range(frames):
        if fi in skip_frames:
            persons.append(())
        else:
            persons.append((Box(fi * 10, 0, fi * 10 + 100, 200), Box(800, 0, 900, 200)))
    return persons


class TestTracePerson:
    """人框 IoU 链：窗口内前后链接、链断即停、多人选 IoU 最大框、越界夹取。"""

    def test_chains_both_directions(self) -> None:
        # Arrange：30 帧，种子帧 15，A 框平滑移动
        persons = tuple(_moving_persons(30))
        seed_box = Box(150, 0, 250, 200)
        # Act
        chain = trace_person(persons, 15, seed_box)
        # Assert：窗口 ±2s（5fps 各 10 帧）→ 帧 5..25 共 21 项，全程链上 A 框
        assert len(chain) == 21
        assert [fi for fi, _ in chain] == list(range(5, 26))
        assert all(box.x1 == fi * 10 for fi, box in chain)

    def test_window_limited_to_2s_each_side(self) -> None:
        # Arrange：60 帧缓存，种子帧 30（两侧都不触边界）
        persons = tuple(_moving_persons(60))
        # Act
        chain = trace_person(persons, 30, Box(300, 0, 400, 200))
        # Assert：严格 ±10 帧
        assert len(chain) == 21
        assert chain[0][0] == 20
        assert chain[-1][0] == 40

    def test_clamped_at_cache_start(self) -> None:
        # Arrange：种子帧 2，向后只链 2 帧即到缓存起点（越界即停）
        persons = tuple(_moving_persons(30))
        # Act
        chain = trace_person(persons, 2, Box(20, 0, 120, 200))
        # Assert：帧 0..12
        assert [fi for fi, _ in chain] == list(range(13))

    def test_empty_frame_breaks_chain(self) -> None:
        # Arrange：帧 12 无人 → 向后链到 13 即停，向前不受影响
        persons = tuple(_moving_persons(30, skip_frames=(12,)))
        # Act
        chain = trace_person(persons, 15, Box(150, 0, 250, 200))
        # Assert：帧 13..25
        assert [fi for fi, _ in chain] == list(range(13, 26))

    def test_iou_drop_breaks_chain(self) -> None:
        # Arrange：帧 20 起 A 框瞬移远处（IoU≈0）→ 向前链到 19 即停
        persons = _moving_persons(30)
        persons[20] = (Box(800, 400, 900, 600),)
        persons[21] = (Box(800, 400, 900, 600),)
        # Act
        chain = trace_person(tuple(persons), 15, Box(150, 0, 250, 200))
        # Assert：帧 5..19
        assert [fi for fi, _ in chain] == list(range(5, 20))

    def test_multi_person_picks_max_iou(self) -> None:
        # Arrange：每帧有远处干扰框 B，且列表顺序 B 在前（排除"取第一个"的巧合）
        persons = tuple((boxes[1], boxes[0]) for boxes in _moving_persons(30))
        # Act
        chain = trace_person(persons, 15, Box(150, 0, 250, 200))
        # Assert：始终链平滑移动的 A 框，不跳 B
        assert len(chain) == 21
        assert all(box.x1 == fi * 10 for fi, box in chain)

    def test_only_seed_when_no_persons(self) -> None:
        # Arrange：除种子外全程无人
        persons: tuple[tuple[Box, ...], ...] = ((),) * 30
        seed_box = Box(0, 0, 100, 200)
        # Act
        chain = trace_person(persons, 15, seed_box)
        # Assert：只含种子帧
        assert chain == [(15, seed_box)]

    def test_seed_out_of_range_raises(self) -> None:
        # Arrange：种子帧越界属逻辑错误
        persons = tuple(_moving_persons(10))
        # Act / Assert：显式失败不静默
        with pytest.raises(BasketballPipelineError, match="种子帧越界"):
            trace_person(persons, 10, Box(0, 0, 100, 200))
        with pytest.raises(BasketballPipelineError, match="种子帧越界"):
            trace_person(persons, -1, Box(0, 0, 100, 200))


def _noise_image(width: int = 1000, height: int = 800, seed: int = 42) -> Image.Image:
    """构造随机噪点图（Laplacian 方差高，模拟清晰帧）。"""
    arr = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestFrameQuality:
    """质量分 = 归一化框面积 × Laplacian 方差：清晰 > 模糊，大框 > 小框。"""

    def test_noise_beats_flat(self) -> None:
        # Arrange
        box = Box(100, 100, 300, 400)
        flat = Image.new("RGB", (1000, 800), (128, 128, 128))
        # Act / Assert：纯色图方差 0，噪点图 > 0
        assert frame_quality(flat, box) == 0.0
        assert frame_quality(_noise_image(), box) > 0.0

    def test_bigger_box_higher_score(self) -> None:
        # Arrange：同一张噪点图，清晰度均匀 → 面积大的分高
        img = _noise_image()
        # Act
        small = frame_quality(img, Box(100, 100, 200, 250))
        large = frame_quality(img, Box(100, 100, 400, 550))
        # Assert
        assert large > small > 0.0

    def test_box_outside_image_returns_zero(self) -> None:
        # Arrange：框完全在图外，夹取后退化
        img = Image.new("RGB", (200, 200), (0, 0, 0))
        # Act / Assert
        assert frame_quality(img, Box(500, 500, 600, 600)) == 0.0


class TestScoreChainFrames:
    """链上帧读图打分：帧图缺失/损坏记 WARNING 跳过，不炸。"""

    def _framesdir(self, tmp_path: Path, fid: str, frames: list[int]) -> Path:
        """在 tmp 下给指定帧号落纯色帧图。"""
        framesdir = tmp_path / "frames"
        (framesdir / fid).mkdir(parents=True)
        for fi in frames:
            Image.new("RGB", (1000, 800), (20, 20, 20)).save(
                framesdir / fid / f"f_{fi + 1:05d}.jpg"
            )
        return framesdir

    def test_missing_frame_skipped(self, tmp_path: Path) -> None:
        # Arrange：链 3 帧，只有帧 0/2 的图
        framesdir = self._framesdir(tmp_path, "v", [0, 2])
        chain = [(0, Box(0, 0, 100, 200)), (1, Box(10, 0, 110, 200)), (2, Box(20, 0, 120, 200))]
        # Act
        scored = score_chain_frames(chain, framesdir, "v")
        # Assert：帧 1 跳过，其余保留链序
        assert [fi for fi, _, _, _ in scored] == [0, 2]

    def test_flat_frames_score_zero(self, tmp_path: Path) -> None:
        # Arrange
        framesdir = self._framesdir(tmp_path, "v", [0, 1])
        chain = [(0, Box(0, 0, 100, 200)), (1, Box(10, 0, 110, 200))]
        # Act
        scored = score_chain_frames(chain, framesdir, "v")
        # Assert：纯色图清晰度 0 → 全 0 分（仍入选，由 pick 层去重）；
        # 深色纯色图 team=黑（V<45 占比达标）
        assert len(scored) == 2
        assert all(score == 0.0 for _, _, score, _ in scored)
        assert all(team == "黑" for _, _, _, team in scored)


class TestTeamOfBox:
    """整帧图 + 人框直接分队（串人守卫的数据源）。"""

    def test_black_box(self) -> None:
        # Arrange：全黑图
        img = Image.new("RGB", (200, 400), (10, 10, 10))
        # Act / Assert
        assert team_of_box(img, Box(0, 0, 200, 400)) == "黑"

    def test_white_box(self) -> None:
        # Arrange：全白图
        img = Image.new("RGB", (200, 400), (255, 255, 255))
        # Act / Assert
        assert team_of_box(img, Box(0, 0, 200, 400)) == "白"

    def test_degenerate_box_returns_casual(self) -> None:
        # Arrange：框完全出界
        img = Image.new("RGB", (200, 400), (10, 10, 10))
        # Act / Assert：退化框不瞎猜
        assert team_of_box(img, Box(500, 500, 600, 600)) == "便服"


class TestDropOppositeTeam:
    """串人守卫：剔与种子明确相反（黑↔白）的帧，便服一律保留。"""

    def _scored(self, items: list[tuple[int, str]]) -> list[tuple[int, Box, float, str]]:
        return [(fi, Box(0, 0, 100, 200), 1.0, t) for fi, t in items]

    def test_black_seed_drops_white(self) -> None:
        # Arrange：种子帧 0=黑，帧 5=白，帧 10=便服
        scored = self._scored([(0, "黑"), (5, "白"), (10, "便服")])
        # Act
        kept = drop_opposite_team(scored, 0)
        # Assert：白帧被剔，便服保留
        assert [fi for fi, _, _, _ in kept] == [0, 10]

    def test_white_seed_drops_black(self) -> None:
        # Arrange / Act
        kept = drop_opposite_team(self._scored([(0, "黑"), (5, "白")]), 5)
        # Assert
        assert [fi for fi, _, _, _ in kept] == [5]

    def test_casual_seed_no_filter(self) -> None:
        # Arrange：种子为便服（不自信）→ 不过滤
        scored = self._scored([(0, "便服"), (5, "白"), (10, "黑")])
        # Act / Assert
        assert drop_opposite_team(scored, 0) == scored

    def test_seed_missing_no_filter(self) -> None:
        # Arrange：种子帧不在 scored（帧图不可读）→ 不过滤
        scored = self._scored([(5, "白"), (10, "黑")])
        # Act / Assert
        assert drop_opposite_team(scored, 0) == scored


class TestPickBestFrames:
    """质量选帧 top N：降序、≥0.5s（帧差 ≥3）去重、容量边界。"""

    def _scored(self, items: list[tuple[int, float]]) -> list[tuple[int, Box, float, str]]:
        """由 (frame_idx, score) 列表构造打分序列（框与 team 随意，不参与排序）。"""
        return [(fi, Box(0, 0, 100, 200), s, "便服") for fi, s in items]

    def test_descending_by_score(self) -> None:
        # Arrange：帧距都够，纯按分排
        scored = self._scored([(0, 1.0), (10, 9.0), (20, 5.0)])
        # Act
        picked = pick_best_frames(scored, 3)
        # Assert
        assert [fi for fi, _, _, _ in picked] == [10, 20, 0]

    def test_spacing_dedup(self) -> None:
        # Arrange：分降序帧 10/11/12/13/20；帧差 <3（<0.5s）的被去重
        scored = self._scored([(10, 9.0), (11, 8.0), (12, 7.0), (13, 6.0), (20, 5.0)])
        # Act
        picked = pick_best_frames(scored, 5)
        # Assert：11、12 距 10 太近被去重；13（差 3=0.6s）与 20 入选
        assert [fi for fi, _, _, _ in picked] == [10, 13, 20]

    def test_tie_breaks_by_frame_order(self) -> None:
        # Arrange：同分 → 帧索引小者优先（稳定可测）
        scored = self._scored([(8, 0.0), (2, 0.0), (5, 0.0)])
        # Act
        picked = pick_best_frames(scored, 3)
        # Assert
        assert [fi for fi, _, _, _ in picked] == [2, 5, 8]

    def test_n_larger_than_candidates_returns_all(self) -> None:
        # Arrange / Act
        picked = pick_best_frames(self._scored([(0, 1.0), (10, 2.0)]), 5)
        # Assert
        assert len(picked) == 2

    def test_empty_returns_empty(self) -> None:
        # Arrange / Act / Assert
        assert pick_best_frames([], 3) == []

    def test_n_less_than_one_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="n ≥ 1"):
            pick_best_frames(self._scored([(0, 1.0)]), 0)


class TestCropNameRanked:
    """多裁命名：rank 1 主名兼容，rank≥2 追加 _q{rank}。"""

    def test_rank1_keeps_main_name(self) -> None:
        # Arrange / Act / Assert
        assert _crop_name_ranked("a_video", 2.0, 1) == "a_video_t2.0.jpg"

    def test_rank2_appends_suffix(self) -> None:
        # Arrange / Act / Assert
        assert _crop_name_ranked("a_video", 2.0, 2) == "a_video_t2.0_q2.jpg"
        assert _crop_name_ranked("a_video", 2.0, 3) == "a_video_t2.0_q3.jpg"


class TestCliMultiCrop:
    """CLI 端到端多裁：crops/crop_scores 契约、质量降序、SKIP 无多裁字段、--best-crops。"""

    def test_multi_crop_contract(self, tmp_path: Path) -> None:
        # Arrange：复用端到端合成场景（a_video OK 球 9 帧 + b_video SKIP 球）
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok, skip = payload["candidates"]
        # OK 球：crops 质量降序（全 0 分按帧序）、crop = crops[0]、三张裁图落盘
        assert ok["crop"] == "a_video_t2.0.jpg"
        assert ok["crops"] == ["a_video_t2.0.jpg", "a_video_t2.0_q2.jpg", "a_video_t2.0_q3.jpg"]
        assert ok["crop_scores"] == [0.0, 0.0, 0.0]
        for name in ok["crops"]:
            assert (out / name).is_file()
        # SKIP 球：无 crops/crop_scores 字段（向后兼容：消费方只看 crop）
        assert skip["status"] == "SKIP"
        assert "crops" not in skip
        assert "crop_scores" not in skip

    def test_quality_best_frame_is_rank1(self, tmp_path: Path) -> None:
        # Arrange：帧 0 为暗色噪点图（清晰高分且 team=黑，与种子帧同队不被守卫剔除），
        # 其余纯色（0 分）→ 帧 0 应为 rank1
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)
        dark_noise = np.random.default_rng(42).integers(0, 40, (800, 1000, 3), dtype=np.uint8)
        Image.fromarray(dark_noise).save(framesdir / "a_video" / "f_00001.jpg")
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok = payload["candidates"][0]
        assert ok["crop_scores"][0] > 0.0
        assert ok["crop_scores"] == sorted(ok["crop_scores"], reverse=True)
        assert len(ok["crops"]) == len(ok["crop_scores"]) == 3

    def test_best_crops_arg_limits_count(self, tmp_path: Path) -> None:
        # Arrange / Act：--best-crops 2
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
                "--best-crops",
                "2",
            ]
        )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok = payload["candidates"][0]
        assert ok["crops"] == ["a_video_t2.0.jpg", "a_video_t2.0_q2.jpg"]
        assert not (out / "a_video_t2.0_q3.jpg").exists()

    def test_best_crops_zero_rejected(self, tmp_path: Path) -> None:
        # Arrange / Act / Assert：--best-crops 0 → CLI 显式失败（SystemExit 2）
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "--goals",
                    str(goals),
                    "--detectdir",
                    str(detectdir),
                    "--framesdir",
                    str(framesdir),
                    "--out",
                    str(out),
                    "--best-crops",
                    "0",
                ]
            )


class TestExpandedRatio:
    """外扩夹取后框宽高比（质量闸第一道作用口径，同标定 report §口径说明）。"""

    def test_normal_box(self) -> None:
        # Arrange：100×300 框外扩 20% → 120×360
        box = Box(100, 100, 200, 400)
        # Act / Assert
        assert expanded_ratio(box, 1000, 800) == pytest.approx(120 / 360)

    def test_degenerate_box_returns_zero(self) -> None:
        # Arrange：框完全出图像下界，外扩夹取后为空（geom.Box 本身拒绝零/负尺寸框）
        # Act / Assert：返回 0.0 必被拦（< GATE_MIN_RATIO），不抛除零
        assert expanded_ratio(Box(10, 500, 20, 510), 100, 100) == 0.0


class TestRatioGate:
    """宽高比 sanity 闸边界值：恰好等于上下限放行、略超拦截、正常框不误杀。

    边界构造利用外扩 pad 取整无舍入误差的整十尺寸（标定 report §阈值建议：
    端点含入，宁漏拦不错杀）。
    """

    def test_exactly_min_ratio_passes(self) -> None:
        # Arrange：w=150 h=1000 → 外扩后 180/1200 = 0.15 = GATE_MIN_RATIO
        box = Box(100, 100, 250, 1100)
        # Act / Assert：恰好等于下限 → 放行
        assert ratio_gate_reason(box, 2000, 1300) is None

    def test_exactly_max_ratio_passes(self) -> None:
        # Arrange：w=1200 h=1000 → 外扩后 1440/1200 = 1.2 = GATE_MAX_RATIO（框右移避开左界夹取）
        box = Box(200, 100, 1400, 1100)
        # Act / Assert：恰好等于上限 → 放行
        assert ratio_gate_reason(box, 2000, 1300) is None

    def test_slightly_below_min_rejected(self) -> None:
        # Arrange：w=140 h=1000 → 外扩后 168/1200 = 0.14 < 0.15
        box = Box(100, 100, 240, 1100)
        # Act / Assert
        reason = ratio_gate_reason(box, 2000, 1300)
        assert reason is not None
        assert "宽高比" in reason

    def test_slightly_above_max_rejected(self) -> None:
        # Arrange：w=1210 h=1000 → 外扩后 1452/1200 = 1.21 > 1.2（框右移避开左界夹取）
        box = Box(200, 100, 1410, 1100)
        # Act / Assert
        assert ratio_gate_reason(box, 2000, 1300) is not None

    def test_normal_standing_ratio_not_killed(self) -> None:
        # Arrange：w=400 h=1000 → 外扩后 480/1200 = 0.4（站立人标定区间 0.28~0.58 中段）
        box = Box(100, 100, 500, 1100)
        # Act / Assert：正常框不误杀
        assert ratio_gate_reason(box, 2000, 1300) is None

    def test_degenerate_box_rejected_not_crash(self) -> None:
        # Arrange / Act / Assert：框完全出界（夹取后空）拦截且不抛异常
        assert ratio_gate_reason(Box(10, 500, 20, 510), 100, 100) is not None


class TestPersonRecheckGate:
    """框内人物复检闸（注入假检测器，不碰真模型）+ 缺框帧不拦截 + 惰性加载。"""

    def _framesdir(self, tmp_path: Path) -> Path:
        """落一张 1000×800 帧图（帧 0）。"""
        framesdir = tmp_path / "frames"
        (framesdir / "v").mkdir(parents=True)
        Image.new("RGB", (1000, 800), (20, 20, 20)).save(framesdir / "v" / "f_00001.jpg")
        return framesdir

    def test_confident_person_passes(self, tmp_path: Path) -> None:
        # Arrange：假检测器高置信
        def recheck(img: Image.Image) -> float:
            return 0.9

        gate = make_quality_gate(self._framesdir(tmp_path), "v", "k", rechecker=recheck)
        # Act / Assert：有可信 person 框 → 放行
        assert gate(0, Box(100, 100, 300, 600)) is None

    def test_no_person_rejected(self, tmp_path: Path) -> None:
        # Arrange：假检测器无框（0.0，瓶子/广告牌情形）
        def recheck(img: Image.Image) -> float:
            return 0.0

        gate = make_quality_gate(self._framesdir(tmp_path), "v", "k", rechecker=recheck)
        # Act / Assert
        reason = gate(0, Box(100, 100, 300, 600))
        assert reason is not None
        assert "person" in reason

    def test_low_conf_rejected_edge_conf_passes(self, tmp_path: Path) -> None:
        # Arrange：0.24 < GATE_PERSON_CONF=0.25 → 拦截；恰 0.25 → 放行
        framesdir = self._framesdir(tmp_path)

        def recheck_low(img: Image.Image) -> float:
            return 0.24

        def recheck_edge(img: Image.Image) -> float:
            return 0.25

        gate_low = make_quality_gate(framesdir, "v", "k", rechecker=recheck_low)
        gate_edge = make_quality_gate(framesdir, "v", "k", rechecker=recheck_edge)
        # Act / Assert
        assert gate_low(0, Box(100, 100, 300, 600)) is not None
        assert gate_edge(0, Box(100, 100, 300, 600)) is None

    def test_ratio_reject_skips_recheck(self, tmp_path: Path) -> None:
        # Arrange：宽高比先行——畸形框零成本拦截，不触发复检（省 CPU）
        calls: list[int] = []

        def recheck(img: Image.Image) -> float:
            calls.append(1)
            return 0.9

        gate = make_quality_gate(self._framesdir(tmp_path), "v", "k", rechecker=recheck)
        # Act：w=50 h=500 → 外扩后 60/600 = 0.1 < 0.15
        reason = gate(0, Box(100, 100, 150, 600))
        # Assert
        assert reason is not None
        assert "宽高比" in reason
        assert calls == []

    def test_missing_box_not_blocked(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：缺框帧（检测缓存无框）——跳过复检不拦截，记 INFO 不静默丢弃
        def recheck(img: Image.Image) -> float:
            return 0.0

        gate = make_quality_gate(self._framesdir(tmp_path), "v", "k", rechecker=recheck)
        # Act / Assert
        with caplog.at_level(logging.INFO, logger="crop_scorers"):
            assert gate(0, None) is None
        assert "缺框" in caplog.text

    def test_lazy_load_only_on_first_recheck(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：未注入 rechecker 时惰性加载——只对过了几何闸的帧首次触发，且只加载一次
        loaded: list[int] = []

        def fake_loader() -> object:
            loaded.append(1)

            def recheck(img: Image.Image) -> float:
                return 0.9

            return recheck

        monkeypatch.setattr(crop_scorers, "_get_person_rechecker", fake_loader)
        gate = make_quality_gate(self._framesdir(tmp_path), "v", "k")
        # Act / Assert：畸形框走几何闸即拦，不触发加载
        assert gate(0, Box(100, 100, 150, 600)) is not None
        assert loaded == []
        # 正常框首次触发加载
        assert gate(0, Box(100, 100, 300, 600)) is None
        assert loaded == [1]
        # 第二帧复用同一实例，不重复加载
        assert gate(0, Box(100, 100, 310, 600)) is None
        assert loaded == [1]


class TestPickBestFramesGate:
    """闸插选帧循环内：废帧丢弃不占位、不耗间距，次优帧补位。"""

    def _scored(self, items: list[tuple[int, float]]) -> list[tuple[int, Box, float, str]]:
        """由 (frame_idx, score) 列表构造打分序列（框与 team 随意，不参与排序）。"""
        return [(fi, Box(0, 0, 100, 200), s, "便服") for fi, s in items]

    def test_rejected_frame_filled_by_next_best(self) -> None:
        # Arrange：最高分帧 0 被闸拦，次优帧补位
        scored = self._scored([(0, 9.0), (10, 8.0), (20, 7.0)])

        def gate(fi: int, box: Box | None) -> str | None:
            return "宽高比越界" if fi == 0 else None

        # Act
        picked = pick_best_frames(scored, 2, gate=gate)
        # Assert
        assert [fi for fi, _, _, _ in picked] == [10, 20]

    def test_rejected_frame_does_not_reserve_spacing(self) -> None:
        # Arrange：帧 0（9 分）废，帧 1（8 分）距帧 0 仅 1 帧——废帧不占间距，帧 1 仍入选
        scored = self._scored([(0, 9.0), (1, 8.0)])

        def gate(fi: int, box: Box | None) -> str | None:
            return "无可信 person" if fi == 0 else None

        # Act
        picked = pick_best_frames(scored, 2, gate=gate)
        # Assert
        assert [fi for fi, _, _, _ in picked] == [1]

    def test_all_rejected_returns_empty(self) -> None:
        # Arrange / Act：帧池全废 → 空（交由上层判 SKIP）
        picked = pick_best_frames(
            self._scored([(0, 9.0), (10, 8.0)]),
            3,
            gate=lambda fi, box: "废",
        )
        # Assert
        assert picked == []


class TestQualityGateSkip:
    """CLI 级：全废球 SKIP+reason=quality_gate、废帧不写 JPEG、不落 crops/crop_scores、
    部分帧废次优帧补位且留痕。"""

    def test_all_frames_rejected_skip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：假检测器全拦（无可信 person）
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)

        def recheck(img: Image.Image) -> float:
            return 0.0

        monkeypatch.setattr(crop_scorers, "_get_person_rechecker", lambda: recheck)
        # Act
        with caplog.at_level(logging.INFO, logger="crop_scorers"):
            rc = main(
                [
                    "--goals",
                    str(goals),
                    "--detectdir",
                    str(detectdir),
                    "--framesdir",
                    str(framesdir),
                    "--out",
                    str(out),
                ]
            )
        # Assert：质量闸 SKIP 不算素材缺失，退出码仍 0
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        entry = payload["candidates"][0]
        assert entry["status"] == "SKIP"
        assert entry["reason"] == "quality_gate"
        # SKIP 条目不落 crops/crop_scores（:1335 口径不动）
        assert "crops" not in entry
        assert "crop_scores" not in entry
        # 废帧不写 JPEG
        assert not list(out.glob("a_video_t2.0*.jpg"))
        # 拦截留痕（INFO）
        assert "质量闸拦截" in caplog.text

    def test_partial_reject_backfills(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：帧 0 换成暗色噪点图（质量分最高且 team=黑不被串人守卫剔），
        # 假检测器只拦噪点图（std 高）→ 帧 0 废、其余帧补位
        goals, detectdir, framesdir, out = TestCliEndToEnd()._setup(tmp_path)
        noise = np.random.default_rng(42).integers(0, 40, (800, 1000, 3), dtype=np.uint8)
        Image.fromarray(noise).save(framesdir / "a_video" / "f_00001.jpg")

        def recheck(img: Image.Image) -> float:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            return 0.0 if float(arr.std()) > 5.0 else 0.9

        monkeypatch.setattr(crop_scorers, "_get_person_rechecker", lambda: recheck)
        # Act
        with caplog.at_level(logging.INFO, logger="crop_scorers"):
            rc = main(
                [
                    "--goals",
                    str(goals),
                    "--detectdir",
                    str(detectdir),
                    "--framesdir",
                    str(framesdir),
                    "--out",
                    str(out),
                ]
            )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        entry = payload["candidates"][0]
        assert entry["status"] == "OK"
        # 次优帧补位凑够 best_crops=3
        assert len(entry["crops"]) == len(entry["crop_scores"]) == 3
        for name in entry["crops"]:
            assert (out / name).is_file()
        # 帧 0 拦截留痕：帧时刻 + 原因
        assert "t=0.0s" in caplog.text
        assert "person" in caplog.text


class TestSeedFields:
    """轨迹传播种子字段（scorer-propagate spec §crop_scorers 小改）：OK 球落
    seed_frame/seed_box/seed_team 三字段，SKIP 球不落（与 crops 口径一致）。"""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """搭一套临时 goals/detect/frames/out。

        a_video@2.0：球全程静止在 A 框 [0,0,100,100] 内 → 轨迹法 OK（定位帧 =
        窗口末帧 8）；b_video@1.0：空缓存 → no_track SKIP。帧图近黑 → 队伍 "黑"。
        """
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps(
                {
                    "session": "test",
                    "goals": [
                        {"file": "a_video.mp4", "anchor_time": 2.0, "status": "confirmed"},
                        {"file": "b_video.mp4", "anchor_time": 1.0, "status": "confirmed"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        detectdir = tmp_path / "detect"
        detectdir.mkdir()
        persons = [[[0, 0, 100, 100], [500, 0, 600, 100]]] * 9
        balls = [
            [
                {
                    "conf": 0.9,
                    "box": [40, 40, 60, 60],
                    "cx": 50,
                    "cy": 50,
                    "sec": i / 5,
                    "frame_idx": i,
                }
            ]
            for i in range(9)
        ]
        (detectdir / "a_video_mot_cache.json").write_text(
            json.dumps({"frames": 9, "balls": balls, "persons": persons}), encoding="utf-8"
        )
        (detectdir / "b_video_mot_cache.json").write_text(
            json.dumps({"frames": 9, "balls": [[]] * 9, "persons": [[]] * 9}),
            encoding="utf-8",
        )
        framesdir = tmp_path / "frames"
        (framesdir / "a_video").mkdir(parents=True)
        for i in range(9):
            Image.new("RGB", (1000, 800), (20, 20, 20)).save(
                framesdir / "a_video" / f"f_{i + 1:05d}.jpg"
            )
        out = tmp_path / "out"
        return goals, detectdir, framesdir, out

    def test_ok_entry_has_seed_fields(self, tmp_path: Path) -> None:
        # Arrange
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        ok = payload["candidates"][0]
        assert ok["status"] == "OK"
        assert ok["seed_frame"] == 8  # 持球点回放：窗口末帧（缓存共 9 帧）
        assert ok["seed_box"] == [0, 0, 100, 100]  # 定位帧投篮者人框 = A 框
        assert ok["seed_team"] == "黑"  # 帧图近黑

    def test_skip_entry_has_no_seed_fields(self, tmp_path: Path) -> None:
        # Arrange
        goals, detectdir, framesdir, out = self._setup(tmp_path)
        # Act
        rc = main(
            [
                "--goals",
                str(goals),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--out",
                str(out),
            ]
        )
        # Assert：SKIP 球不落三字段
        assert rc == 0
        payload = json.loads((out / "scorer_candidates.json").read_text(encoding="utf-8"))
        skip = payload["candidates"][1]
        assert skip["status"] == "SKIP"
        assert "seed_frame" not in skip
        assert "seed_box" not in skip
        assert "seed_team" not in skip
