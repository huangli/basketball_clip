"""rank_photos.py 单元测试（打分 / 分桶 / 构图 / 尺度断言 / apply 校验 / 第二轮调参）。

覆盖：每帧 max conf 球选取（conf<0.35 视为无球）、冲击分五信号
（球筐距/主体尺度/球速/人球交互/球高度）、进球锚点 ±0.6s 加成与 force_pick 保底、
10s 分桶保底 1 张再按分补齐、构图裁切（联合包围/特写占比/头顶留白/三分法/
边界夹取/过近与放大降级/16:9 与 4:3）、hoops/goals 批次文件读取（缺失记空、
schema 损坏显式失败）、--force 清空重跑、缓存坐标尺度可执行断言（越界抛
SchemaError）、selections.json schema 校验、apply 落盘命名契约。
不触碰真实素材与 ffmpeg。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crop_scorers import MotCache
from errors import SchemaError
from geom import Box
from mot_candidates import Detection
from rank_photos import (
    OUT_4_3,
    OUT_16_9,
    HoopEvent,
    ScoredCandidate,
    apply_filename,
    apply_goal_boost,
    assert_cache_scale,
    bucket_pick,
    compose_crop,
    frame_ball,
    load_goal_anchors,
    load_hoop_events,
    reset_candidate_outputs,
    score_frames,
    validate_selections,
    window_times,
)

DET_W: int = 1920
DET_H: int = 1080


def _ball(conf: float, cx: int, cy: int, fi: int) -> Detection:
    """构造一个球检测（box 取球心 ±10px；sec 按 5fps 帧率换算）。"""
    return Detection(
        conf=conf,
        box=[cx - 10, cy - 10, cx + 10, cy + 10],
        cx=cx,
        cy=cy,
        sec=fi / 5,
        frame_idx=fi,
    )


def _cache(balls: list[tuple[Detection, ...]], persons: list[tuple[Box, ...]]) -> MotCache:
    """由逐帧列表构造 MotCache（frames 取两者长度）。"""
    assert len(balls) == len(persons)
    return MotCache(frames=len(balls), balls=tuple(balls), persons=tuple(persons))


def _big_person(cx: int = 960, cy: int = 700) -> Box:
    """大人框（约 0.14 帧面积，尺度项记满）。"""
    return Box(cx - 200, cy - 400, cx + 200, cy + 400)


class TestFrameBall:
    """每帧只取 max conf 球；conf < 0.35 视为无球。"""

    def test_picks_max_conf(self) -> None:
        balls = (_ball(0.5, 100, 100, 0), _ball(0.9, 500, 500, 0))
        picked = frame_ball(balls)
        assert picked is not None
        assert picked.conf == pytest.approx(0.9)

    def test_low_conf_treated_as_no_ball(self) -> None:
        assert frame_ball((_ball(0.34, 100, 100, 0),)) is None

    def test_threshold_boundary_kept(self) -> None:
        assert frame_ball((_ball(0.35, 100, 100, 0),)) is not None

    def test_empty(self) -> None:
        assert frame_ball(()) is None


class TestScoreFrames:
    """冲击分：有球有人 > 空帧；球速快 > 静止；低置信球按无球计。"""

    def test_ball_and_person_beats_empty(self) -> None:
        cache = _cache(
            [(), (_ball(0.9, 960, 300, 1),)],
            [(), (_big_person(),)],
        )
        scores = score_frames(cache, DET_H)
        assert scores[0] == 0.0
        assert scores[1] > 0.0

    def test_person_only_beats_empty(self) -> None:
        cache = _cache([(), ()], [(), (_big_person(),)])
        scores = score_frames(cache, DET_H)
        assert scores[1] > scores[0] == 0.0

    def test_fast_ball_beats_still_ball(self) -> None:
        still = _cache(
            [(_ball(0.9, 960, 500, 0),), (_ball(0.9, 960, 500, 1),)],
            [(_big_person(),), (_big_person(),)],
        )
        fast = _cache(
            [(_ball(0.9, 960, 500, 0),), (_ball(0.9, 1560, 500, 1),)],
            [(_big_person(),), (_big_person(),)],
        )
        assert score_frames(fast, DET_H)[1] > score_frames(still, DET_H)[1]

    def test_low_conf_ball_scores_like_no_ball(self) -> None:
        with_low = _cache([(), (_ball(0.30, 960, 200, 1),)], [(), (_big_person(),)])
        no_ball = _cache([(), ()], [(), (_big_person(),)])
        assert score_frames(with_low, DET_H)[1] == pytest.approx(score_frames(no_ball, DET_H)[1])

    def test_high_ball_beats_low_ball(self) -> None:
        # 静止球（帧间无位移，排除球速信号干扰）：画面上部 > 画面下部
        cache = _cache(
            [
                (),
                (_ball(0.9, 960, 100, 1),),
                (_ball(0.9, 960, 100, 2),),
                (_ball(0.9, 960, 900, 3),),
                (_ball(0.9, 960, 900, 4),),
            ],
            [(), (), (), (), ()],
        )
        scores = score_frames(cache, DET_H)
        assert scores[2] > scores[4]


class TestBucketPick:
    """10s 分桶：每桶保底 1 张，再按分数全局补齐至 total。"""

    def _item(self, fid: str, fi: int, global_sec: float, score: float) -> ScoredCandidate:
        return ScoredCandidate(
            fid=fid, frame_idx=fi, sec=fi / 5, global_sec=global_sec, score=score
        )

    def test_each_bucket_guaranteed_one(self) -> None:
        # 3 个桶：桶0 一堆高分，桶1/桶2 各一张低分
        items = [self._item("a", i, i * 0.2, 0.9 - i * 0.01) for i in range(10)]
        items.append(self._item("b", 0, 15.0, 0.1))
        items.append(self._item("c", 0, 25.0, 0.2))
        picked = bucket_pick(items, total=4, bucket_sec=10.0)
        assert len(picked) == 4
        buckets = {int(p.global_sec // 10.0) for p in picked}
        assert buckets == {0, 1, 2}

    def test_fill_by_score_after_guarantee(self) -> None:
        items = [self._item("a", i, i * 0.2, s) for i, s in enumerate([0.5, 0.9, 0.8, 0.1])]
        picked = bucket_pick(items, total=3, bucket_sec=10.0)
        scores = sorted(p.score for p in picked)
        assert scores == [0.5, 0.8, 0.9]  # 保底 0.9，补齐取 0.8/0.5，丢 0.1

    def test_guarantee_beats_total_when_more_buckets(self) -> None:
        # 桶数多于目标张数时保底优先（宁多勿漏，不丢时间覆盖）
        items = [self._item("a", i, i * 12.0, 0.5) for i in range(10)]
        assert len(bucket_pick(items, total=4, bucket_sec=10.0)) == 10

    def test_fewer_items_than_total(self) -> None:
        items = [self._item("a", 0, 0.0, 0.5), self._item("a", 1, 30.0, 0.6)]
        assert len(bucket_pick(items, total=200, bucket_sec=10.0)) == 2

    def test_result_sorted_by_global_sec(self) -> None:
        items = [
            self._item("a", 0, 25.0, 0.9),
            self._item("a", 1, 1.0, 0.8),
            self._item("a", 2, 12.0, 0.7),
        ]
        picked = bucket_pick(items, total=3, bucket_sec=10.0)
        assert [p.global_sec for p in picked] == [1.0, 12.0, 25.0]

    def test_empty_input(self) -> None:
        assert bucket_pick([], total=200, bucket_sec=10.0) == []


class TestComposeCrop:
    """构图裁切：联合包围 / 头顶留白 / 三分法 / 边界夹取 / 放大降级 / 双比例。"""

    IMG_W: int = 3840
    IMG_H: int = 2160
    OUT_W, OUT_H = OUT_16_9

    def _person(self) -> Box:
        return Box(1500, 700, 1900, 1900)  # 高 1200，站画面中下部

    def test_aspect_and_out_size_16_9(self) -> None:
        plan = compose_crop(self._person(), None, self.IMG_W, self.IMG_H, *OUT_16_9)
        w = plan.box.x2 - plan.box.x1
        h = plan.box.y2 - plan.box.y1
        assert abs(w / h - 16 / 9) < 0.01

    def test_aspect_4_3(self) -> None:
        plan = compose_crop(self._person(), None, self.IMG_W, self.IMG_H, *OUT_4_3)
        w = plan.box.x2 - plan.box.x1
        h = plan.box.y2 - plan.box.y1
        assert abs(w / h - 4 / 3) < 0.01

    def test_headroom_at_least_5_percent(self) -> None:
        person = self._person()
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        h = plan.box.y2 - plan.box.y1
        headroom = person.y1 - plan.box.y1
        assert headroom >= 0.05 * h - 2  # 2px 取整容差（第二轮：留白下限 10%→5%）

    def test_person_fully_inside_crop(self) -> None:
        person = self._person()
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.x1 <= person.x1
        assert plan.box.x2 >= person.x2
        assert plan.box.y1 <= person.y1
        assert plan.box.y2 >= person.y2

    def test_near_ball_united(self) -> None:
        person = self._person()
        ball = Box(1680, 300, 1720, 340)  # 人头顶上方不远处
        plan = compose_crop(person, ball, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.x1 <= ball.x1
        assert plan.box.y1 <= ball.y1
        assert plan.box.x2 >= ball.x2

    def test_far_ball_not_united(self) -> None:
        person = self._person()
        ball = Box(100, 100, 140, 140)  # 画面远角，与人无关
        plan = compose_crop(person, ball, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.x1 > ball.x2 or plan.box.y1 > ball.y2

    def test_clamped_to_image_left_edge(self) -> None:
        person = Box(0, 700, 400, 1900)  # 贴左边缘
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.x1 >= 0
        assert plan.box.x2 <= self.IMG_W
        assert plan.box.x1 <= person.x1  # 人仍在框内

    def test_clamped_to_image_bottom_edge(self) -> None:
        person = Box(1500, 1600, 1900, 2160)  # 贴下边缘
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.y2 <= self.IMG_H
        assert plan.box.y2 >= person.y2

    def test_no_upscale_when_crop_wide_enough(self) -> None:
        plan = compose_crop(self._person(), None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.upscale == pytest.approx(1.0)
        assert not plan.penalized

    def test_small_subject_needs_upscale(self) -> None:
        tiny = Box(1800, 1500, 2000, 1900)  # 宽 200，远小于 1920
        plan = compose_crop(tiny, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.upscale > 1.5
        assert plan.penalized

    def test_upscale_between_1_and_threshold_not_penalized(self) -> None:
        person = Box(1000, 700, 2700, 1300)  # 宽 1700 → 裁框宽 1870，略小于 1920
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert 1.0 < plan.upscale <= 1.5
        assert not plan.penalized


class TestAssertCacheScale:
    """尺度换算可执行断言：越界抛 SchemaError，合法返回换算因子。"""

    def test_valid_16_9_returns_factor(self) -> None:
        cache = _cache([(_ball(0.9, 1900, 1000, 0),)], [(Box(0, 0, 1920, 1080),)])
        factor = assert_cache_scale(cache, 3840, 2160, "t.mp4")
        assert factor == pytest.approx(2.0)

    def test_valid_4_3_returns_factor(self) -> None:
        cache = _cache([(_ball(0.9, 1900, 1400, 0),)], [(Box(0, 0, 1920, 1440),)])
        factor = assert_cache_scale(cache, 2560, 1920, "t.mp4")
        assert factor == pytest.approx(2560 / 1920)

    def test_x_beyond_detect_width_raises(self) -> None:
        cache = _cache([(_ball(0.9, 2100, 500, 0),)], [()])
        with pytest.raises(SchemaError):
            assert_cache_scale(cache, 3840, 2160, "t.mp4")

    def test_y_beyond_detect_height_raises(self) -> None:
        # 4:3 老场次检测帧高 1440；按 16:9 的 1080 上限误算会放过，按 1440 则越界
        cache = _cache([()], [(Box(0, 1400, 100, 1500),)])
        with pytest.raises(SchemaError):
            assert_cache_scale(cache, 2560, 1920, "t.mp4")

    def test_negative_beyond_tolerance_raises(self) -> None:
        cache = _cache([()], [(Box(-100, 0, 100, 500),)])
        with pytest.raises(SchemaError):
            assert_cache_scale(cache, 3840, 2160, "t.mp4")

    def test_within_tolerance_passes(self) -> None:
        # 1% 容差内的轻微越界放行（检测框贴边的浮点/取整误差）
        cache = _cache([(_ball(0.9, 1925, 10, 0),)], [()])
        assert_cache_scale(cache, 3840, 2160, "t.mp4")


class TestValidateSelections:
    """selections.json schema 校验（rules.md §0.2：损坏显式失败）。"""

    def _payload(self) -> dict:
        return {"session": "s1", "selected": ["c001", "c002"]}

    def test_valid(self) -> None:
        assert validate_selections(self._payload(), "s1") == ["c001", "c002"]

    def test_session_mismatch_raises(self) -> None:
        with pytest.raises(SchemaError):
            validate_selections(self._payload(), "other")

    def test_missing_selected_raises(self) -> None:
        with pytest.raises(SchemaError):
            validate_selections({"session": "s1"}, "s1")

    def test_non_string_id_raises(self) -> None:
        with pytest.raises(SchemaError):
            validate_selections({"session": "s1", "selected": ["c001", 2]}, "s1")

    def test_top_level_not_dict_raises(self) -> None:
        with pytest.raises(SchemaError):
            validate_selections(["c001"], "s1")


class TestWindowTimes:
    """抽取时刻夹取：片头不低 0，片尾留出余量（防 seek 过 EOF 抽帧失败）。"""

    def test_clamped_at_start(self) -> None:
        times = window_times(0.01, 59.94, 10.0)
        assert len(times) == 7
        assert all(t >= 0.0 for t in times)
        assert times[0] == 0.0

    def test_clamped_before_eof(self) -> None:
        # 片尾候选：所有时刻 ≤ duration − 1.5/fps
        times = window_times(9.98, 59.94, 10.0)
        assert max(times) <= 10.0 - 1.5 / 59.94 + 1e-9

    def test_middle_unclamped(self) -> None:
        times = window_times(5.0, 60.0, 20.0)
        assert times[0] == pytest.approx(5.0 - 3 / 60.0)
        assert times[-1] == pytest.approx(5.0 + 3 / 60.0)


class TestApplyFilename:
    """apply 落盘命名：照片_XXX_视频序号_时刻.jpg。"""

    def test_format(self) -> None:
        assert apply_filename(1, 23, 12.34) == "照片_001_v023_t0012.3.jpg"

    def test_zero_padded(self) -> None:
        name = apply_filename(7, 3, 0.5)
        assert name.startswith("照片_007_v003_t0000.5")
        assert name.endswith(".jpg")


def _hoop(start: float, end: float, hx: float, hy: float) -> HoopEvent:
    """构造一个筐事件（窗口 + 筐心，检测尺度）。"""
    return HoopEvent(start=start, end=end, hx=hx, hy=hy)


class TestHoopSignal:
    """球筐距信号（第二轮主信号）：筐近 > 筐远；窗口外/无 hoops 记 0 不报错。"""

    def test_ball_near_hoop_beats_far(self) -> None:
        # 单帧无速度项；同 cy 排除球高干扰；无人排除尺度/交互干扰
        hoops = [_hoop(0.0, 10.0, 960.0, 300.0)]
        near = _cache([(_ball(0.9, 960, 320, 0),)], [()])
        far = _cache([(_ball(0.9, 300, 300, 0),)], [()])
        assert score_frames(near, DET_H, hoops)[0] > score_frames(far, DET_H, hoops)[0]

    def test_no_hoops_no_error(self) -> None:
        cache = _cache([(_ball(0.9, 960, 300, 0),)], [()])
        assert score_frames(cache, DET_H, None)[0] > 0.0

    def test_outside_window_scores_zero_hoop(self) -> None:
        # 窗口 [0,1] 外（帧 25 → sec=5.0）的帧：筐距信号记 0，与无 hoops 同分
        hoops = [_hoop(0.0, 1.0, 960.0, 300.0)]
        cache = _cache(
            [()] * 25 + [(_ball(0.9, 960, 320, 25),)],
            [()] * 26,
        )
        assert score_frames(cache, DET_H, hoops)[25] == pytest.approx(
            score_frames(cache, DET_H, None)[25]
        )

    def test_overlapping_windows_take_best(self) -> None:
        # 两个窗口同帧覆盖，取距更近的筐
        hoops = [_hoop(0.0, 10.0, 100.0, 100.0), _hoop(0.0, 10.0, 960.0, 300.0)]
        near = _cache([(_ball(0.9, 960, 320, 0),)], [()])
        assert score_frames(near, DET_H, hoops)[0] == pytest.approx(
            score_frames(near, DET_H, [_hoop(0.0, 10.0, 960.0, 300.0)])[0]
        )


class TestGoalBoost:
    """进球锚点加成：±0.6s 内 ×1.5，每球窗口内最高分帧 force_pick 保底。"""

    def _item(self, fi: int, score: float) -> ScoredCandidate:
        return ScoredCandidate(fid="a", frame_idx=fi, sec=fi / 5, global_sec=fi / 5, score=score)

    def test_boost_inside_window(self) -> None:
        items = [self._item(50, 0.4)]  # sec=10.0，锚点正中
        boosted = apply_goal_boost(items, [10.0])
        assert boosted[0].score == pytest.approx(0.6)
        assert boosted[0].force_pick

    def test_no_boost_outside_window(self) -> None:
        items = [self._item(50, 0.4), self._item(60, 0.5)]  # sec=10.0 / 12.0
        boosted = apply_goal_boost(items, [10.0])
        assert boosted[1].score == pytest.approx(0.5)
        assert not boosted[1].force_pick

    def test_window_boundary_inclusive(self) -> None:
        # ±0.6s 边界含端点：sec=9.4/10.6 加成，9.2/10.8 不加（5fps 采样点）
        items = [self._item(fi, 0.4) for fi in (46, 47, 53, 54)]
        boosted = apply_goal_boost(items, [10.0])
        by_sec = {it.sec: it for it in boosted}
        assert by_sec[9.4].score == pytest.approx(0.6)
        assert by_sec[10.6].score == pytest.approx(0.6)
        assert by_sec[9.2].score == pytest.approx(0.4)
        assert by_sec[10.8].score == pytest.approx(0.4)

    def test_one_force_pick_per_goal_best_frame(self) -> None:
        # 窗口内多帧只保底最高分那一帧
        items = [self._item(fi, s) for fi, s in ((49, 0.3), (50, 0.7), (51, 0.5))]
        boosted = apply_goal_boost(items, [10.0])
        forced = [it for it in boosted if it.force_pick]
        assert len(forced) == 1
        assert forced[0].frame_idx == 50

    def test_anchor_without_frames_noop(self) -> None:
        items = [self._item(0, 0.4)]
        boosted = apply_goal_boost(items, [50.0])
        assert boosted == items

    def test_empty_anchors_noop(self) -> None:
        items = [self._item(50, 0.4)]
        assert apply_goal_boost(items, []) == items


class TestBucketPickForcePick:
    """force_pick 帧在分桶保底之外额外保底入选（保底优先可超 total）。"""

    def _item(
        self, fi: int, global_sec: float, score: float, force: bool = False
    ) -> ScoredCandidate:
        return ScoredCandidate(
            fid="a",
            frame_idx=fi,
            sec=fi / 5,
            global_sec=global_sec,
            score=score,
            force_pick=force,
        )

    def test_force_pick_beyond_bucket_guarantee(self) -> None:
        # 2 个 force_pick 同桶 + 另两桶各 1 张：total=2 仍全保（保底优先）
        items = [
            self._item(0, 1.0, 0.9, force=True),
            self._item(1, 1.2, 0.8, force=True),
            self._item(2, 1.5, 0.7),
            self._item(3, 15.0, 0.1),
            self._item(4, 25.0, 0.2),
        ]
        picked = bucket_pick(items, total=2, bucket_sec=10.0)
        # 2 forced + 桶0 保底(0.7) + 桶1/桶2 保底 = 5？否：forced 所在桶仍可从池中保底
        assert {p.frame_idx for p in picked} == {0, 1, 2, 3, 4}

    def test_no_duplicate_when_force_pick_would_win_bucket(self) -> None:
        items = [self._item(0, 1.0, 0.9, force=True), self._item(1, 1.2, 0.8)]
        picked = bucket_pick(items, total=10, bucket_sec=10.0)
        assert len(picked) == 2  # forced 不再作为桶内保底/补齐重复入选


class TestComposeCloseUp:
    """特写构图（第二轮）：主体目标占裁框高 55~75%，头顶留白 ≥5%，允许切脚。"""

    IMG_W: int = 3840
    IMG_H: int = 2160

    def test_subject_fraction_target(self) -> None:
        person = Box(1500, 700, 1900, 1900)  # 高 1200
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        crop_h = plan.box.y2 - plan.box.y1
        frac = 1200 / crop_h
        assert 0.55 <= frac <= 0.75

    def test_headroom_at_least_5_percent(self) -> None:
        person = Box(1500, 700, 1900, 1900)
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        crop_h = plan.box.y2 - plan.box.y1
        headroom = person.y1 - plan.box.y1
        assert headroom >= 0.05 * crop_h - 2  # 2px 取整容差

    def test_feet_cut_allowed_at_bottom_edge(self) -> None:
        # 人贴画面下边缘：裁框夹回后允许切脚/切膝（不再为保全身而外扩），头必须在框内
        person = Box(1500, 1200, 1900, 2160)  # 高 960，脚贴底边
        plan = compose_crop(person, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        assert plan.box.y1 <= person.y1  # 头在框内
        assert plan.box.y2 <= self.IMG_H  # 裁框不出画面

    def test_too_close_penalized(self) -> None:
        # 主体占比 >85%（巨人近景，裁框被画面夹小）视为过近降分
        giant = Box(800, 100, 2500, 2100)  # 高 2000，接近满幅
        plan = compose_crop(giant, None, self.IMG_W, self.IMG_H, *OUT_16_9)
        frac = 2000 / (plan.box.y2 - plan.box.y1)
        assert frac > 0.85
        assert plan.penalized


class TestLoadHoopEvents:
    """hoops_batchN.json 读取：缺失记空不报错；schema 损坏显式失败。"""

    def _write(self, session_dir: Path, name: str, events: list) -> None:
        payload = {"session": "s", "params": {}, "events": events}
        (session_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_load(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            "hoops_batch1.json",
            [
                {
                    "key": "f#e1",
                    "fid": "f",
                    "event_idx": 1,
                    "window": [1.0, 5.0],
                    "anchor": [960, 300],
                    "detected": True,
                }
            ],
        )
        got = load_hoop_events(tmp_path)
        assert got == {"f": [HoopEvent(start=1.0, end=5.0, hx=960.0, hy=300.0)]}

    def test_multi_batch_merged(self, tmp_path: Path) -> None:
        ev = {
            "key": "f#e1",
            "fid": "f",
            "event_idx": 1,
            "window": [0.0, 2.0],
            "anchor": [10, 20],
            "detected": True,
        }
        self._write(tmp_path, "hoops_batch1.json", [ev])
        self._write(tmp_path, "hoops_batch2.json", [dict(ev, key="g#e1", fid="g")])
        got = load_hoop_events(tmp_path)
        assert set(got) == {"f", "g"}

    def test_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_hoop_events(tmp_path) == {}

    def test_schema_damaged_raises(self, tmp_path: Path) -> None:
        (tmp_path / "hoops_batch1.json").write_text('{"events": [{"fid": 1}]}', encoding="utf-8")
        with pytest.raises(SchemaError):
            load_hoop_events(tmp_path)


class TestLoadGoalAnchors:
    """goals_batchN.json 读取：只取 confirmed；缺失记空不报错。"""

    def _write(self, session_dir: Path, name: str, goals: list) -> None:
        (session_dir / name).write_text(
            json.dumps({"session": "s", "goals": goals}), encoding="utf-8"
        )

    def test_only_confirmed_collected(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            "goals_batch1.json",
            [
                {"file": "a.mp4", "anchor_time": 10.0, "status": "confirmed"},
                {"file": "a.mp4", "anchor_time": 20.0, "status": "rejected"},
                {"file": "b.mp4", "anchor_time": 5, "status": "confirmed"},
            ],
        )
        got = load_goal_anchors(tmp_path)
        assert got == {"a.mp4": [10.0], "b.mp4": [5.0]}

    def test_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_goal_anchors(tmp_path) == {}

    def test_schema_damaged_raises(self, tmp_path: Path) -> None:
        self._write(tmp_path, "goals_batch1.json", [{"file": "a.mp4"}])
        with pytest.raises(SchemaError):
            load_goal_anchors(tmp_path)


class TestResetCandidateOutputs:
    """--force：清空 candidates 目录与 candidates JSON；空目录幂等不报错。"""

    def test_clears_outputs(self, tmp_path: Path) -> None:
        photos = tmp_path / "photos"
        cand = photos / "candidates"
        cand.mkdir(parents=True)
        (cand / "c001.jpg").write_bytes(b"x")
        (photos / "photo_candidates.json").write_text("{}", encoding="utf-8")
        reset_candidate_outputs(photos)
        assert not cand.exists()
        assert not (photos / "photo_candidates.json").exists()

    def test_idempotent_on_empty(self, tmp_path: Path) -> None:
        reset_candidate_outputs(tmp_path / "photos")  # 不存在也不报错
