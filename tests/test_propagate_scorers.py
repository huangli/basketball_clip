"""propagate_scorers.py 单元测试（轨迹传播：贪心跟踪 + 颜色守卫 + 进球映射 + 传播源 + evaluate）。

全部纯函数/合成数据：persons 序列手造、帧图为纯色合成图（落 tmp_path）、roster
内存构造，不碰真帧真模型真素材。覆盖：candidates schema 校验（顶层损坏 SchemaError、
status 枚举收紧）、
entry_seed 宽松归一、贪心跟踪器（链上/断轨封存+新轨/双人不串/同帧竞争三级裁决/
一轨一框）、颜色守卫（主色 <60% 标 mixed/便服不计数/帧图缺失 WARNING 跳过/
无球轨迹跳过不采样/同帧多轨迹只解码一次）、
进球→轨迹映射（归上/归不上/多轨迹取 IoU 最大/缺 seed WARNING 进 unlinked）、
传播源规则（自身不算源/冲突不预填/NOGOAL 不作源/号码低置信歧义不作源）、
evaluate 双指标（roster-seeded 留早种子/number-seeded 冷启动/冲突 mixed unlinked
计覆盖失败）、CLI 端到端（落盘幂等、candidates 只读、缺 cache 退出码 1、
--evaluate/--roster 互相缺失 parser.error）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import propagate_scorers
from errors import SchemaError
from geom import Box
from propagate_scorers import (
    NOGOAL_TAG,
    FileResult,
    PersonTrack,
    SeedReport,
    TrackMember,
    _parse_args,
    apply_color_guard,
    build_links_payload,
    entry_seed,
    evaluate_reports,
    format_report,
    load_scorer_candidates,
    main,
    map_goals_to_tracks,
    member_source_tags,
    number_prefill_tag,
    track_persons,
    track_prefill_tag,
)
from roster import Roster, validate_roster

# ---- 合成数据辅助 ----


def _box(x1: int, y1: int, x2: int, y2: int) -> Box:
    return Box(x1, y1, x2, y2)


def _persons(frames: list[list[tuple[int, int, int, int]]]) -> tuple[tuple[Box, ...], ...]:
    """把 [[(x1,y1,x2,y2), ...], ...] 转成按帧对齐的 Box 元组序列。"""
    return tuple(tuple(_box(*b) for b in frame) for frame in frames)


def _track(track_id: int, points: list[tuple[int, tuple[int, int, int, int]]]) -> PersonTrack:
    """按 (frame_idx, box) 点列构造 PersonTrack。"""
    pts: list[tuple[int, Box]] = [(fi, _box(*b)) for fi, b in points]
    return PersonTrack(track_id=track_id, points=pts, last_frame=points[-1][0])


def _entry(
    key: str,
    *,
    file: str = "a_video.mp4",
    anchor: float = 1.0,
    status: str = "OK",
    seed_frame: int | None = 0,
    seed_box: list[int] | None = None,
    number_guess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一条 scorer_candidates 记录；seed_frame=None 表示不落 seed 字段。"""
    e: dict[str, Any] = {
        "key": key,
        "file": file,
        "anchor_time": anchor,
        "status": status,
    }
    if seed_frame is not None:
        e["seed_frame"] = seed_frame
        e["seed_box"] = seed_box if seed_box is not None else [0, 0, 100, 100]
    if number_guess is not None:
        e["number_guess"] = number_guess
    return e


def _write_frame(framesdir: Path, fid: str, fi: int, rgb: tuple[int, int, int]) -> None:
    """落一张纯色合成帧图（200×400，f_NNNNN.jpg 命名与 extract_frames 一致）。"""
    d: Path = framesdir / fid
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 400), rgb).save(d / f"f_{fi + 1:05d}.jpg")


_BLACK: tuple[int, int, int] = (0, 0, 0)  # V=0 < TH_BLACK → 黑
_WHITE: tuple[int, int, int] = (255, 255, 255)  # V>170 且 S<70 → 白
_CASUAL: tuple[int, int, int] = (100, 100, 100)  # 黑/白占比均不达标 → 便服

_FULL_BOX: tuple[int, int, int, int] = (0, 0, 200, 400)  # 覆盖整帧，纯色图队伍即全队


def _roster(assignments: dict[str, str], tags: tuple[str, ...] = ("黑21", "白7")) -> Roster:
    """内存构造校验后的 Roster（players 名单含号码供号码映射）。"""
    players = [
        {"tag": t, "name": "", "team": "主队" if t.startswith("黑") else "对手"} for t in tags
    ]
    return validate_roster(
        {
            "session": "test",
            "confirmed": True,
            "players": players,
            "assignments": assignments,
        },
        "test-roster",
    )


# ---- candidates 加载与 seed 归一 ----


class TestLoadScorerCandidates:
    """scorer_candidates.json schema 校验：顶层结构坏显式失败。"""

    def _write(self, path: Path, payload: Any) -> Path:  # noqa: ANN401 JSON 待校验
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_valid_loads(self, tmp_path: Path) -> None:
        # Arrange
        p = self._write(
            tmp_path / "scorer_candidates.json",
            {"session": "t", "candidates": [_entry("a_video.mp4#1.0")]},
        )
        # Act
        entries = load_scorer_candidates(p)
        # Assert
        assert len(entries) == 1
        assert entries[0]["key"] == "a_video.mp4#1.0"

    def test_top_not_dict_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path / "scorer_candidates.json", [1, 2])
        with pytest.raises(SchemaError, match="顶层"):
            load_scorer_candidates(p)

    def test_candidates_not_list_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path / "scorer_candidates.json", {"candidates": {}})
        with pytest.raises(SchemaError, match="candidates"):
            load_scorer_candidates(p)

    def test_entry_missing_key_raises(self, tmp_path: Path) -> None:
        bad = _entry("x")
        del bad["key"]
        p = self._write(tmp_path / "scorer_candidates.json", {"candidates": [bad]})
        with pytest.raises(SchemaError, match="key"):
            load_scorer_candidates(p)

    def test_entry_status_unknown_raises(self, tmp_path: Path) -> None:
        # status 值不在 (OK, SKIP) → SchemaError（schema 校验收紧，防未知状态静默漏过）
        bad = _entry("x")
        bad["status"] = "PENDING"
        p = self._write(tmp_path / "scorer_candidates.json", {"candidates": [bad]})
        with pytest.raises(SchemaError, match="status 非法"):
            load_scorer_candidates(p)

    def test_bad_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scorer_candidates.json"
        p.write_text("{bad json", encoding="utf-8")
        with pytest.raises(SchemaError, match="损坏"):
            load_scorer_candidates(p)


class TestEntrySeed:
    """entry_seed 宽松归一：缺失/畸形 → None（不炸批）。"""

    def test_valid(self) -> None:
        assert entry_seed(_entry("k", seed_frame=3, seed_box=[1, 2, 30, 40])) == (
            3,
            Box(1, 2, 30, 40),
        )

    def test_missing_fields(self) -> None:
        assert entry_seed(_entry("k", seed_frame=None)) is None

    def test_degenerate_box(self) -> None:
        assert entry_seed(_entry("k", seed_box=[10, 10, 10, 20])) is None  # x2<=x1

    def test_bool_frame_rejected(self) -> None:
        e = _entry("k")
        e["seed_frame"] = True
        assert entry_seed(e) is None

    def test_wrong_box_length(self) -> None:
        assert entry_seed(_entry("k", seed_box=[1, 2, 3])) is None


# ---- 贪心多目标跟踪器 ----


class TestTrackPersons:
    """贪心 IoU 跟踪：链上/断轨封存+新轨/双人不串/竞争唯一性裁决/一轨一框。"""

    def test_chain_single_person(self) -> None:
        # Arrange：每帧右移 5px（IoU≈0.9），应全程一轨
        persons = _persons([[(i * 5, 0, i * 5 + 100, 100)] for i in range(5)])
        # Act
        tracks = track_persons(persons)
        # Assert
        assert len(tracks) == 1
        assert tracks[0].span == (0, 4)
        assert len(tracks[0].points) == 5

    def test_gap_seals_track_and_reopens(self) -> None:
        # Arrange：帧 0-2 有人，帧 3-13 无人（间隔 11 > MAX_GAP_FRAMES=10），帧 13 重现
        frames: list[list[tuple[int, int, int, int]]] = [[(0, 0, 100, 100)]] * 3
        frames += [[]] * 10
        frames += [[(0, 0, 100, 100)]]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert：旧轨封存，重现开新轨
        assert len(tracks) == 2
        assert tracks[0].span == (0, 2)
        assert tracks[1].span == (13, 13)

    def test_gap_within_max_continues(self) -> None:
        # Arrange：间隔恰 10 帧（≤MAX_GAP_FRAMES）仍吸收
        frames: list[list[tuple[int, int, int, int]]] = [[(0, 0, 100, 100)]]
        frames += [[]] * 9
        frames += [[(0, 0, 100, 100)]]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert
        assert len(tracks) == 1
        assert tracks[0].span == (0, 10)

    def test_two_persons_no_cross(self) -> None:
        # Arrange：两个远离的人各自缓慢移动
        persons = _persons(
            [[(i * 5, 0, i * 5 + 100, 100), (800 - i * 5, 0, 900 - i * 5, 100)] for i in range(4)]
        )
        # Act
        tracks = track_persons(persons)
        # Assert
        assert len(tracks) == 2
        assert [p[0] for p in tracks[0].points] == [0, 1, 2, 3]
        assert [p[0] for p in tracks[1].points] == [0, 1, 2, 3]

    def test_competition_recent_track_wins(self) -> None:
        # 同帧竞争裁决第一级：轨迹最后更新帧更近者胜（track1 lf=0，track2 lf=3）
        frames: list[list[tuple[int, int, int, int]]] = [
            [(0, 0, 100, 10)],  # f0 → track1
            [],
            [],
            [(55, 0, 155, 10)],  # f3：与 track1 IoU≈0.29 <0.3 → 新轨 track2
            [(28, 0, 128, 10)],  # f4：与 track1 IoU≈0.56、track2 IoU≈0.57 双达标
        ]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert：Z 归 track2（lf 更近），track1 停在 f0
        assert len(tracks) == 2
        assert [p[0] for p in tracks[0].points] == [0]
        assert [p[0] for p in tracks[1].points] == [3, 4]

    def test_competition_higher_iou_wins_on_same_last_frame(self) -> None:
        # 同帧竞争裁决第二级：last_frame 并列时 IoU 更大者胜
        frames: list[list[tuple[int, int, int, int]]] = [
            [(0, 0, 100, 100), (0, 105, 100, 205)],  # f0 → track1/track2
            [(0, 0, 100, 100), (0, 105, 100, 205)],  # f1 各自吸收，lf 并列=1
            [(0, 53, 100, 153)],  # f2：与 track1 IoU≈0.307、track2 IoU≈0.316
        ]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert：Z 归 track2（IoU 更大）
        assert [p[0] for p in tracks[0].points] == [0, 1]
        assert [p[0] for p in tracks[1].points] == [0, 1, 2]

    def test_competition_smaller_track_id_wins_on_full_tie(self) -> None:
        # 同帧竞争裁决第三级：last_frame 与 IoU 均并列时 track_id 更小者胜
        frames: list[list[tuple[int, int, int, int]]] = [
            [(0, 0, 100, 100), (0, 104, 100, 204)],  # f0 → track1/track2
            [(0, 0, 100, 100), (0, 104, 100, 204)],  # f1 各自吸收
            [(0, 52, 100, 152)],  # f2：与两轨 IoU 均 = 48/152
        ]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert：Z 归 track1（id 更小）
        assert [p[0] for p in tracks[0].points] == [0, 1, 2]
        assert [p[0] for p in tracks[1].points] == [0, 1]

    def test_track_absorbs_at_most_one_box_per_frame(self) -> None:
        # Arrange：两框都匹配同一轨迹，轨迹只吸收 IoU 最大者，落选框开新轨
        frames: list[list[tuple[int, int, int, int]]] = [
            [(0, 0, 100, 100)],  # f0 → track1
            [(0, 0, 100, 100), (10, 0, 110, 100)],  # f1：IoU 1.0 / ≈0.82 均只配 track1
        ]
        # Act
        tracks = track_persons(_persons(frames))
        # Assert
        assert len(tracks) == 2
        assert tracks[0].points[-1] == (1, _box(0, 0, 100, 100))
        assert tracks[1].points == [(1, _box(10, 0, 110, 100))]

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="max_gap"):
            track_persons(_persons([[(0, 0, 10, 10)]]), max_gap=0)
        with pytest.raises(ValueError, match="min_iou"):
            track_persons(_persons([[(0, 0, 10, 10)]]), min_iou=0.0)


# ---- 颜色守卫 ----


class TestColorGuard:
    """封存轨迹颜色采样：主色占比 <60% 标 mixed；便服不计数；帧图缺失 WARNING 跳过；
    无球轨迹跳过不采样；同帧多轨迹只解码一次。"""

    def _counting_open(self, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """monkeypatch Image.open 计数，返回调用路径列表（仍走真实解码）。"""
        opened: list[Path] = []
        real_open = Image.open

        def counting_open(path: Path) -> Image.Image:
            opened.append(Path(path))
            return real_open(path)

        monkeypatch.setattr(propagate_scorers.Image, "open", counting_open)
        return opened

    def test_all_black_not_mixed(self, tmp_path: Path) -> None:
        # Arrange
        framesdir = tmp_path / "frames"
        for fi in (0, 5, 10):
            _write_frame(framesdir, "f", fi, _BLACK)
        t = _track(1, [(fi, _FULL_BOX) for fi in range(11)])
        t.keys = ["f.mp4#1.0"]
        # Act
        apply_color_guard([t], framesdir, "f")
        # Assert
        assert t.mixed is False

    def test_half_black_half_white_mixed(self, tmp_path: Path) -> None:
        # Arrange：2 黑 2 白 → 主色占比 0.5 < 0.6
        framesdir = tmp_path / "frames"
        for fi, rgb in ((0, _BLACK), (5, _WHITE), (10, _BLACK), (15, _WHITE)):
            _write_frame(framesdir, "f", fi, rgb)
        t = _track(1, [(fi, _FULL_BOX) for fi in (0, 5, 10, 15)])
        t.keys = ["f.mp4#1.0"]
        # Act
        apply_color_guard([t], framesdir, "f")
        # Assert
        assert t.mixed is True

    def test_casual_samples_not_counted(self, tmp_path: Path) -> None:
        # Arrange：3 黑 + 2 便服 → 便服不计入，黑占比 3/3=1.0 → 不 mixed
        framesdir = tmp_path / "frames"
        for fi, rgb in ((0, _BLACK), (5, _CASUAL), (10, _BLACK), (15, _CASUAL), (20, _BLACK)):
            _write_frame(framesdir, "f", fi, rgb)
        t = _track(1, [(fi, _FULL_BOX) for fi in (0, 5, 10, 15, 20)])
        t.keys = ["f.mp4#1.0"]
        # Act
        apply_color_guard([t], framesdir, "f")
        # Assert
        assert t.mixed is False

    def test_three_black_one_white_not_mixed(self, tmp_path: Path) -> None:
        # Arrange：主色占比 3/4=0.75 ≥ 0.6 → 不 mixed（阈值端点含入）
        framesdir = tmp_path / "frames"
        for fi, rgb in ((0, _BLACK), (5, _BLACK), (10, _WHITE), (15, _BLACK)):
            _write_frame(framesdir, "f", fi, rgb)
        t = _track(1, [(fi, _FULL_BOX) for fi in (0, 5, 10, 15)])
        t.keys = ["f.mp4#1.0"]
        # Act
        apply_color_guard([t], framesdir, "f")
        # Assert
        assert t.mixed is False

    def test_missing_frame_skips_sample_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：帧 5 的图缺失 → 该采样点跳过记 WARNING，其余样本照常
        framesdir = tmp_path / "frames"
        _write_frame(framesdir, "f", 0, _BLACK)
        _write_frame(framesdir, "f", 10, _BLACK)
        t = _track(1, [(fi, _FULL_BOX) for fi in (0, 5, 10)])
        t.keys = ["f.mp4#1.0"]
        # Act
        with caplog.at_level(logging.WARNING, logger="propagate_scorers"):
            apply_color_guard([t], framesdir, "f")
        # Assert
        assert t.mixed is False
        assert any("颜色守卫帧图不可读" in r.message for r in caplog.records)

    def test_track_without_keys_skips_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：无球轨迹（keys 为空）→ 整体跳过，一帧都不解码
        framesdir = tmp_path / "frames"
        for fi in (0, 5, 10):
            _write_frame(framesdir, "f", fi, _BLACK)
        t = _track(1, [(fi, _FULL_BOX) for fi in range(11)])
        opened = self._counting_open(monkeypatch)
        # Act
        apply_color_guard([t], framesdir, "f")
        # Assert
        assert opened == []
        assert t.mixed is False

    def test_same_frame_decoded_once_for_multiple_tracks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：同帧两条挂球轨迹竞争同一帧采样 → 该帧只解码一次
        framesdir = tmp_path / "frames"
        for fi in (0, 5):
            _write_frame(framesdir, "f", fi, _BLACK)
        t1 = _track(1, [(fi, _FULL_BOX) for fi in (0, 5)])
        t1.keys = ["f.mp4#1.0"]
        t2 = _track(2, [(fi, _FULL_BOX) for fi in (0, 5)])
        t2.keys = ["f.mp4#2.0"]
        opened = self._counting_open(monkeypatch)
        # Act
        apply_color_guard([t1, t2], framesdir, "f")
        # Assert：2 帧各解码 1 次（非 2 轨迹 × 2 帧 = 4 次）
        assert len(opened) == 2
        assert len(set(opened)) == 2


# ---- 进球→轨迹映射 ----


class TestMapGoalsToTracks:
    """seed_frame 帧全部轨迹框找 IoU 最大 ≥0.3；归不上/缺 seed → unlinked。"""

    def test_maps_to_matching_track(self) -> None:
        # Arrange
        t = _track(1, [(3, (0, 0, 100, 100))])
        e = _entry("k1", seed_frame=3, seed_box=[10, 0, 110, 100])  # IoU≈0.82
        # Act
        res = map_goals_to_tracks([t], [e], "a_video")
        # Assert
        assert t.keys == ["k1"]
        assert res.unlinked == []

    def test_low_iou_unlinked(self) -> None:
        t = _track(1, [(3, (0, 0, 100, 100))])
        e = _entry("k1", seed_frame=3, seed_box=[500, 500, 600, 600])  # IoU=0
        res = map_goals_to_tracks([t], [e], "a_video")
        assert t.keys == []
        assert res.unlinked == ["k1"]

    def test_seed_frame_without_track_point_unlinked(self) -> None:
        t = _track(1, [(3, (0, 0, 100, 100))])
        e = _entry("k1", seed_frame=4, seed_box=[0, 0, 100, 100])  # 该帧无轨迹框
        res = map_goals_to_tracks([t], [e], "a_video")
        assert res.unlinked == ["k1"]

    def test_competing_tracks_max_iou_wins(self) -> None:
        # Arrange：同帧两条轨迹框，IoU 大者胜出
        t1 = _track(1, [(3, (0, 0, 100, 100))])
        t2 = _track(2, [(3, (0, 105, 100, 205))])
        e = _entry("k1", seed_frame=3, seed_box=[0, 110, 100, 210])  # 与 t2 IoU 更高
        # Act
        map_goals_to_tracks([t1, t2], [e], "a_video")
        # Assert
        assert t1.keys == []
        assert t2.keys == ["k1"]

    def test_missing_seed_ok_entry_warns_and_unlinks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        e = _entry("k1", seed_frame=None)  # OK 球缺 seed 字段
        with caplog.at_level(logging.WARNING, logger="propagate_scorers"):
            res = map_goals_to_tracks([], [e], "a_video")
        assert res.unlinked == ["k1"]
        assert any("缺 seed 字段" in r.message for r in caplog.records)

    def test_skip_entry_unlinked_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # SKIP 球按 crop_scorers 口径本就不落 seed 字段，直接 unlinked 不告警
        e = _entry("k1", status="SKIP", seed_frame=None)
        with caplog.at_level(logging.WARNING, logger="propagate_scorers"):
            res = map_goals_to_tracks([], [e], "a_video")
        assert res.unlinked == ["k1"]
        assert not caplog.records


# ---- track_links.json 载荷 ----


class TestBuildLinksPayload:
    """载荷契约：track-v1、只出挂球轨迹、排序稳定幂等。"""

    def test_payload_shape_and_idempotent(self) -> None:
        # Arrange
        t1 = _track(1, [(0, (0, 0, 10, 10)), (7, (0, 0, 10, 10))])
        t1.keys = ["b.mp4#2.0", "b.mp4#1.0"]  # 保持 entry 序
        t2 = _track(2, [(1, (0, 0, 10, 10))])  # 无球轨迹不输出
        t3 = _track(3, [(2, (0, 0, 10, 10))])
        t3.keys = ["b.mp4#3.0"]
        t3.mixed = True
        per_file = {
            "b": FileResult(fid="b", tracks=[t1, t2, t3], unlinked=["b.mp4#9.0"]),
            "a": FileResult(fid="a", tracks=[], unlinked=[]),
        }
        # Act
        payload1 = build_links_payload(per_file)
        payload2 = build_links_payload(per_file)
        # Assert
        assert payload1 == payload2  # 幂等
        assert payload1["version"] == "track-v1"
        assert list(payload1["per_file"]) == ["a", "b"]  # fid 排序
        tracks = payload1["per_file"]["b"]["tracks"]
        assert [t["track_id"] for t in tracks] == [1, 3]
        assert tracks[0] == {
            "track_id": 1,
            "keys": ["b.mp4#2.0", "b.mp4#1.0"],
            "mixed": False,
            "span": [0, 7],
        }
        assert tracks[1]["mixed"] is True
        assert payload1["per_file"]["b"]["unlinked"] == ["b.mp4#9.0"]


# ---- 传播预填来源 ----


class TestMemberSourceTags:
    """源 tag 提取：NOGOAL 哨兵永远不作源；口径开关生效。"""

    def test_noggoal_truth_never_source(self) -> None:
        m = TrackMember(key="k", anchor_time=1.0, truth=NOGOAL_TAG, number_tag=None)
        assert member_source_tags(m, use_roster=True, use_number=True) == []

    def test_noggoal_number_never_source(self) -> None:
        m = TrackMember(key="k", anchor_time=1.0, truth=None, number_tag=NOGOAL_TAG)
        assert member_source_tags(m, use_roster=True, use_number=True) == []

    def test_flags_gate_sources(self) -> None:
        m = TrackMember(key="k", anchor_time=1.0, truth="黑21", number_tag="白7")
        assert member_source_tags(m, use_roster=True, use_number=False) == ["黑21"]
        assert member_source_tags(m, use_roster=False, use_number=True) == ["白7"]
        assert member_source_tags(m, use_roster=False, use_number=False) == []


class TestTrackPrefillTag:
    """传播预填决策：自身不算源 / 冲突不预填 / mixed 不预填 / 无源不预填。"""

    def _m(self, key: str, truth: str | None = None, number: str | None = None) -> TrackMember:
        return TrackMember(key=key, anchor_time=1.0, truth=truth, number_tag=number)

    def test_single_source_prefills(self) -> None:
        members = [self._m("a", truth="黑21"), self._m("b")]
        assert track_prefill_tag(members, "b", mixed=False) == ("黑21", "")

    def test_self_not_source(self) -> None:
        # 唯一成员就是被预填球自身 → 无源不预填
        members = [self._m("a", truth="黑21")]
        assert track_prefill_tag(members, "a", mixed=False) == (None, "")

    def test_conflict_no_prefill(self) -> None:
        members = [self._m("a", truth="黑21"), self._m("b", truth="白7"), self._m("c")]
        assert track_prefill_tag(members, "c", mixed=False) == (None, "conflict")

    def test_same_tag_sources_no_conflict(self) -> None:
        members = [self._m("a", truth="黑21"), self._m("b", number="黑21"), self._m("c")]
        assert track_prefill_tag(members, "c", mixed=False) == ("黑21", "")

    def test_mixed_track_no_prefill(self) -> None:
        members = [self._m("a", truth="黑21"), self._m("b")]
        assert track_prefill_tag(members, "b", mixed=True) == (None, "mixed")

    def test_noggoal_source_ignored(self) -> None:
        # 唯一源是 NOGOAL → 视同无源
        members = [self._m("a", truth=NOGOAL_TAG), self._m("b")]
        assert track_prefill_tag(members, "b", mixed=False) == (None, "")


class TestNumberPrefillTag:
    """号码→tag 映射：高置信 + 名单唯一命中才作源（与页面 build_entries 同口径）。"""

    def test_high_conf_unique_hit(self) -> None:
        roster = _roster({})
        guess = {"number": "21", "color": "黑", "name_text": None, "confidence": "high"}
        assert number_prefill_tag(guess, list(roster.players)) == "黑21"

    def test_low_conf_not_source(self) -> None:
        roster = _roster({})
        guess = {"number": "21", "color": "黑", "name_text": None, "confidence": "low"}
        assert number_prefill_tag(guess, list(roster.players)) is None

    def test_no_number_not_source(self) -> None:
        roster = _roster({})
        guess = {"number": None, "color": "黑", "name_text": None, "confidence": "high"}
        assert number_prefill_tag(guess, list(roster.players)) is None

    def test_ambiguous_number_not_source(self) -> None:
        # Arrange：名单中 黑21/白21 同号 → 歧义不作源
        roster = _roster({}, tags=("黑21", "白21"))
        guess = {"number": "21", "color": "其他", "name_text": None, "confidence": "high"}
        assert number_prefill_tag(guess, list(roster.players)) is None

    def test_none_guess_not_source(self) -> None:
        roster = _roster({})
        assert number_prefill_tag(None, list(roster.players)) is None


# ---- --evaluate 双指标 ----


class TestEvaluate:
    """evaluate 报表：roster-seeded 留最早种子；number-seeded 冷启动；
    冲突/mixed/unlinked 计入覆盖失败一侧（分母口径写死）。"""

    def _fixture(self) -> tuple[dict[str, FileResult], list[dict[str, Any]], Roster]:
        """合成一场三轨迹：

        - track1（干净）：#1.0(黑21)、#2.0(黑21)、#3.0(白7)；#1.0 带号码高置信 21
          → number_tag=黑21。
        - track2（mixed）：#5.0(黑21)、#6.0(黑21)；#5.0 带号码高置信 21。
        - track3（号码冲突）：#7.0(号码 21→黑21)、#8.0(号码 7→白7)、#9.0(黑21, 无号码)。
        - unlinked：#4.0(白7 归属 + 号码高置信 7)。
        """
        ng21 = {"number": "21", "color": "黑", "name_text": None, "confidence": "high"}
        ng7 = {"number": "7", "color": "白", "name_text": None, "confidence": "high"}
        t1 = _track(1, [(0, (0, 0, 10, 10))])
        t1.keys = ["f.mp4#1.0", "f.mp4#2.0", "f.mp4#3.0"]
        t2 = _track(2, [(0, (0, 0, 10, 10))])
        t2.keys = ["f.mp4#5.0", "f.mp4#6.0"]
        t2.mixed = True
        t3 = _track(3, [(0, (0, 0, 10, 10))])
        t3.keys = ["f.mp4#7.0", "f.mp4#8.0", "f.mp4#9.0"]
        per_file = {"f": FileResult(fid="f", tracks=[t1, t2, t3], unlinked=["f.mp4#4.0"])}
        entries = [
            _entry("f.mp4#1.0", anchor=1.0, number_guess=ng21),
            _entry("f.mp4#2.0", anchor=2.0),
            _entry("f.mp4#3.0", anchor=3.0),
            _entry("f.mp4#4.0", anchor=4.0, number_guess=ng7),
            _entry("f.mp4#5.0", anchor=5.0, number_guess=ng21),
            _entry("f.mp4#6.0", anchor=6.0),
            _entry("f.mp4#7.0", anchor=7.0, number_guess=ng21),
            _entry("f.mp4#8.0", anchor=8.0, number_guess=ng7),
            _entry("f.mp4#9.0", anchor=9.0),
        ]
        roster = _roster(
            {
                "f.mp4#1.0": "黑21",
                "f.mp4#2.0": "黑21",
                "f.mp4#3.0": "白7",
                "f.mp4#4.0": "白7",
                "f.mp4#5.0": "黑21",
                "f.mp4#6.0": "黑21",
                "f.mp4#9.0": "黑21",
            }
        )
        return per_file, entries, roster

    def test_roster_seeded(self) -> None:
        # Arrange
        per_file, entries, roster = self._fixture()
        # Act
        r = evaluate_reports(per_file, entries, roster)["roster_seeded"]
        # Assert：track1 种子 #1.0 → #2.0 对、#3.0 错（prefilled=2, correct=1）；
        # track2 mixed 不预填（种子+1）；unlinked #4.0 有归属（种子+1）；
        # track3 中仅 #9.0 有归属：既是种子也是唯一归属球，无被预填对象（种子+1）
        assert r.seeds == 4
        assert r.prefilled == 2
        assert r.correct == 1
        assert r.accuracy == pytest.approx(0.5)
        assert r.coverage == pytest.approx(0.5)

    def test_number_seeded(self) -> None:
        # Arrange
        per_file, entries, roster = self._fixture()
        # Act
        r = evaluate_reports(per_file, entries, roster)["number_seeded"]
        # Assert：track1 源={#1.0} → #2.0 对、#3.0 错（#1.0 自身不作对象）；
        # track2 mixed 不预填（源 #5.0 计种子）；track3 源={黑21,白7} 冲突不预填
        # （源 #7.0/#8.0 计种子）；unlinked #4.0 是号码源（种子+1）
        assert r.seeds == 5
        assert r.prefilled == 2
        assert r.correct == 1
        assert r.accuracy == pytest.approx(0.5)
        assert r.coverage == pytest.approx(0.4)

    def test_empty_reports_none_ratios(self) -> None:
        # Arrange：全空（无种子无预填）→ 比率 None 不硬算 0
        per_file = {"f": FileResult(fid="f", tracks=[], unlinked=[])}
        # Act
        reports = evaluate_reports(per_file, [], _roster({}))
        # Assert
        assert reports["roster_seeded"].accuracy is None
        assert reports["number_seeded"].coverage is None

    def test_format_report_lines(self) -> None:
        reports = {
            "roster_seeded": SeedReport("roster_seeded", seeds=2, prefilled=3, correct=3),
            "number_seeded": SeedReport("number_seeded", seeds=0, prefilled=0, correct=0),
        }
        text = format_report(reports)
        assert "roster_seeded" in text
        assert "准确率=100.0%" in text
        assert "覆盖倍数=1.50" in text
        assert "n/a" in text  # number_seeded 无分母


# ---- CLI 端到端 ----


class TestCli:
    """CLI：落盘幂等、candidates 只读、缺 cache 退出码 1、--evaluate 协议。"""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """搭 candidates + mot_cache + 帧图；单文件 11 帧单静止人，一球归上。"""
        scorers = tmp_path / "scorers"
        scorers.mkdir()
        candidates = scorers / "scorer_candidates.json"
        entries = [
            _entry("a_video.mp4#1.0", anchor=1.0, seed_frame=5, seed_box=[0, 0, 100, 100]),
            _entry("a_video.mp4#2.0", anchor=2.0, status="SKIP", seed_frame=None),
        ]
        candidates.write_text(
            json.dumps({"session": "t", "candidates": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        detectdir = tmp_path / "detect"
        detectdir.mkdir()
        (detectdir / "a_video_mot_cache.json").write_text(
            json.dumps(
                {
                    "frames": 11,
                    "balls": [[]] * 11,
                    "persons": [[[0, 0, 100, 100]]] * 11,
                }
            ),
            encoding="utf-8",
        )
        framesdir = tmp_path / "frames"
        for fi in range(11):
            _write_frame(framesdir, "a_video", fi, _BLACK)
        return candidates, detectdir, framesdir

    def test_end_to_end_idempotent_and_readonly(self, tmp_path: Path) -> None:
        # Arrange
        candidates, detectdir, framesdir = self._setup(tmp_path)
        before = candidates.read_bytes()
        argv = [
            "--candidates",
            str(candidates),
            "--detectdir",
            str(detectdir),
            "--framesdir",
            str(framesdir),
        ]
        # Act
        rc1 = main(argv)
        out = candidates.parent / "track_links.json"
        first = out.read_bytes()
        rc2 = main(argv)
        # Assert
        assert rc1 == 0 and rc2 == 0
        assert out.read_bytes() == first  # 幂等
        assert candidates.read_bytes() == before  # candidates 只读不改
        payload = json.loads(first)
        assert payload["version"] == "track-v1"
        pf = payload["per_file"]["a_video"]
        assert pf["tracks"] == [
            {"track_id": 1, "keys": ["a_video.mp4#1.0"], "mixed": False, "span": [0, 10]}
        ]
        assert pf["unlinked"] == ["a_video.mp4#2.0"]  # SKIP 球无 seed → unlinked

    def test_missing_cache_exit_1(self, tmp_path: Path) -> None:
        # Arrange
        candidates, detectdir, framesdir = self._setup(tmp_path)
        for p in detectdir.glob("*.json"):
            p.unlink()
        # Act
        rc = main(
            [
                "--candidates",
                str(candidates),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
            ]
        )
        # Assert：素材缺失 → 全部 unlinked 且退出码非零（产出型脚本口径）
        assert rc == 1
        payload = json.loads((candidates.parent / "track_links.json").read_text(encoding="utf-8"))
        assert payload["per_file"]["a_video"]["tracks"] == []
        assert sorted(payload["per_file"]["a_video"]["unlinked"]) == [
            "a_video.mp4#1.0",
            "a_video.mp4#2.0",
        ]

    def test_bad_candidates_schema_exit_1(self, tmp_path: Path) -> None:
        # Arrange
        candidates, detectdir, framesdir = self._setup(tmp_path)
        candidates.write_text(json.dumps([1, 2]), encoding="utf-8")
        # Act
        rc = main(
            [
                "--candidates",
                str(candidates),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
            ]
        )
        # Assert：schema 损坏显式失败（main 捕获转非零退出码）
        assert rc == 1

    def test_evaluate_requires_roster(self, tmp_path: Path) -> None:
        candidates, detectdir, framesdir = self._setup(tmp_path)
        with pytest.raises(SystemExit):
            _parse_args(
                [
                    "--candidates",
                    str(candidates),
                    "--detectdir",
                    str(detectdir),
                    "--framesdir",
                    str(framesdir),
                    "--evaluate",
                ]
            )

    def test_roster_requires_evaluate(self, tmp_path: Path) -> None:
        # 对称校验：给了 --roster 但没 --evaluate → parser.error
        candidates, detectdir, framesdir = self._setup(tmp_path)
        with pytest.raises(SystemExit):
            _parse_args(
                [
                    "--candidates",
                    str(candidates),
                    "--detectdir",
                    str(detectdir),
                    "--framesdir",
                    str(framesdir),
                    "--roster",
                    str(tmp_path / "roster.json"),
                ]
            )

    def test_evaluate_prints_report(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        # Arrange：一球归上且有 roster 归属（自身为唯一归属球 → 种子=1、预填=0）
        candidates, detectdir, framesdir = self._setup(tmp_path)
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(
            json.dumps(
                {
                    "session": "t",
                    "confirmed": True,
                    "players": [{"tag": "黑21", "name": "", "team": "主队"}],
                    "assignments": {"a_video.mp4#1.0": "黑21"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # Act
        rc = main(
            [
                "--candidates",
                str(candidates),
                "--detectdir",
                str(detectdir),
                "--framesdir",
                str(framesdir),
                "--roster",
                str(roster_path),
                "--evaluate",
            ]
        )
        # Assert
        assert rc == 0
        out_text = capsys.readouterr().out
        assert "roster_seeded" in out_text
        assert "number_seeded" in out_text
        assert "种子=1" in out_text


def test_import_has_no_side_effects() -> None:
    """模块导入零副作用（rules.md §1：禁止模块级配置/加载）。"""
    assert propagate_scorers.__name__ == "propagate_scorers"
