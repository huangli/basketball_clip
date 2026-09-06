"""gen_review_clips 单元测试。

覆盖：candidates.json schema 校验（合法通过、顶层非列表、记录非对象、
缺字段、数值字段为 bool）；find_source 的 --srcdir 直找与缺失；
cluster_candidates 时空放宽聚类（批次 2 新增）；event_anchor 末成员锚点；
event_hoop_dist / sort_events_by_hoop_dist（events_index 筐距排序）；
跨文件续接（split_window / find_next_source / plan_clip_segments 回退）；
goal-anchor：clip_window 窗口口径、cut_cluster_clip/cut_wide_clip 透出
continued 标志（events_index 人工锚点退化用）。
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

import gen_review_clips as grc
from errors import SchemaError
from gen_review_clips import (
    _validate_candidates,
    adaptive_crop,
    clip_window,
    cluster_candidates,
    cut_cluster_clip,
    cut_wide_clip,
    event_anchor,
    event_hoop_dist,
    event_verdict,
    find_event_track,
    find_next_source,
    find_source,
    plan_clip_segments,
    sort_events_by_hoop_dist,
    split_window,
)

_PATH = "work/pilot/candidates.json"


def _cand(**over: object) -> dict[str, Any]:
    """构造一条合法候选记录，按字段覆盖。"""
    base: dict[str, Any] = {
        "fid": "0011",
        "label": "#1",
        "t0": 18.2,
        "ac": 0.31,
        "cx": 960,
        "cy": 540,
    }
    base.update(over)
    return base


def test_valid_candidates_pass() -> None:
    # Arrange
    records = [_cand(), _cand(fid="0020", label="#3")]
    # Act
    result = _validate_candidates(records, _PATH)
    # Assert
    assert result == records


def test_top_level_not_list_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="顶层"):
        _validate_candidates({"cands": []}, _PATH)


def test_record_not_dict_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="第1条"):
        _validate_candidates([_cand(), 42], _PATH)


def test_missing_str_field_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="fid"):
        _validate_candidates([_cand(fid=None)], _PATH)


def test_bool_num_field_raises() -> None:
    # Arrange：bool 是 int 子类，必须显式排除
    with pytest.raises(SchemaError, match="t0"):
        _validate_candidates([_cand(t0=False)], _PATH)


def test_find_source_with_srcdir(tmp_path: pathlib.Path) -> None:
    # Arrange：dji 命名，fid 即主名
    (tmp_path / "sample_x.mp4").write_bytes(b"")
    # Act
    found = find_source("sample_x", str(tmp_path))
    # Assert
    assert found is not None
    assert found.endswith("sample_x.mp4")


def test_find_source_srcdir_missing_returns_none(tmp_path: pathlib.Path) -> None:
    # Arrange / Act / Assert：缺失返回 None（调用方记 ERROR 跳过），不抛异常
    assert find_source("nope", str(tmp_path)) is None


def test_adaptive_crop_bbox_with_margin() -> None:
    # Arrange：img 系轨迹两点 (100,100)-(200,150) → 原片系 ×2 = (200,200)-(400,300)
    track = [[0.0, 100, 100, "det"], [1.0, 200, 150, "det"]]
    # Act
    crop_x, crop_y, side = adaptive_crop(track, 3840, 2160)
    # Assert：span 200x100 + margin 600 = 800 < 下限 1200 → side=1200；中心 (300,250)
    assert side == 1200
    assert crop_x == 0  # 300-600<0 → clamp
    assert crop_y == 0


def test_adaptive_crop_caps_at_orig_short_side() -> None:
    # Arrange：轨迹横贯全场
    track = [[0.0, 10, 10, "det"], [1.0, 1900, 1000, "det"]]
    # Act
    _, _, side = adaptive_crop(track, 3840, 2160)
    # Assert：上限 = 原片短边 2160
    assert side == 2160


def test_find_event_track_hit_and_miss() -> None:
    # Arrange
    events = [
        {"window": [2.0, 6.0], "detected": True, "track": [[2.0, 10, 10, "det"]]},
        {"window": [8.0, 9.0], "detected": False, "track": []},
    ]
    # Act / Assert
    assert find_event_track(events, 3.0) == [[2.0, 10, 10, "det"]]
    assert find_event_track(events, 8.5) is None  # detected=false 不命中
    assert find_event_track(events, 99.0) is None


def test_event_verdict_four_branches() -> None:
    # Arrange
    members = [{"fid": "f1", "label": "#1"}, {"fid": "f1", "label": "#2"}]
    # Act / Assert：YES 优先；无 YES 有 UNCLEAR 为 ?；全 NO 为 N；未判为 ""
    assert event_verdict(members, {"f1#2@420": {"answer": "YES"}}) == "VLM:Y"
    assert event_verdict(members, {"f1#1@420": {"answer": "UNCLEAR"}}) == "VLM:?"
    assert (
        event_verdict(
            members,
            {"f1#1@420": {"answer": "NO"}, "f1#1@630": {"answer": "UNCLEAR"}},
        )
        == "VLM:?"
    )
    assert event_verdict(members, {"f1#1@420": {"answer": "NO"}}) == "VLM:N"
    assert event_verdict(members, {}) == ""


# ---- 批次 2 改进 1：事件锚点取末成员 ----


def test_event_anchor_last_member_not_max_conf() -> None:
    # Arrange：ac 最高的成员不在末尾（批次 1 长事件 max-conf 锚点切错时段场景）
    members = [
        _cand(t0=10.0, ac=0.95),
        _cand(t0=12.0, ac=0.10),
        _cand(t0=13.5, ac=0.50),
    ]
    # Act / Assert：锚点取末成员 t0，而非 max-conf 成员
    assert event_anchor(members) == 13.5


def test_event_anchor_single_member_unchanged() -> None:
    # Arrange / Act / Assert：单成员事件行为与旧 max-conf 一致
    assert event_anchor([_cand(t0=18.2, ac=0.31)]) == 18.2


# ---- 批次 2 改进 2：事件聚类加时空放宽条件 ----


def test_cluster_spatiotemporal_merge_190354_case() -> None:
    # Arrange：190354 式同球重复——两候选间隔 2.6s(>2.0)、距离 309px(<=400)
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=12.6, cx=409, cy=100)]
    # Act
    clusters = cluster_candidates(cands)
    # Assert：时空放宽合并为 1 事件
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_merge_boundary_inclusive() -> None:
    # Arrange：边界值 gap=6.0s、dist=400px，均为 <= 即合并
    cands = [_cand(t0=10.0, cx=0, cy=0), _cand(t0=16.0, cx=400, cy=0)]
    # Act / Assert
    assert len(cluster_candidates(cands)) == 1


def test_cluster_far_distance_not_merged() -> None:
    # Arrange：同 2.6s 间隔但距离 401px(>400) → 不合并
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=12.6, cx=501, cy=100)]
    # Act / Assert
    assert len(cluster_candidates(cands)) == 2


def test_cluster_beyond_merge_gap_not_merged() -> None:
    # Arrange：间隔 6.1s(>6.0)，即使球位重合也不合并
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=16.1, cx=100, cy=100)]
    # Act / Assert
    assert len(cluster_candidates(cands)) == 2


# ---- event-split：纯时间链加空间约束 + 事件时长上限 ----
# 2026-09-06 立哥实测：审核片段多回合并一段、跨场地不切断。
# 旧语义"间隔 <=2s 球位再远也并"即本次修复对象，新语义：纯时间链须同时满足
# 关联位置（筐位优先，缺筐逐对退回球位）距离 <=800px，且事件跨度 <=20s。


def test_cluster_chain_near_position_merges() -> None:
    # Arrange：间隔 <=2s 且球位近（<=800px）→ 仍并入（同回合行为保持）
    cands = [
        _cand(t0=10.0, cx=100, cy=100),
        _cand(t0=11.9, cx=400, cy=300),
        _cand(t0=13.8, cx=600, cy=500),
    ]
    # Act
    clusters = cluster_candidates(cands)
    # Assert：3 候选链式合成 1 事件
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_chain_far_ball_split() -> None:
    # Arrange：间隔 <=2s 但无筐数据退回球位、球位差 >800px → 切断
    # （旧 test_cluster_pure_time_chain_unchanged 的场景，新语义下必须切开）
    cands = [
        _cand(t0=10.0, cx=0, cy=0),
        _cand(t0=12.0, cx=1900, cy=1000),
        _cand(t0=13.9, cx=0, cy=0),
    ]
    # Act
    clusters = cluster_candidates(cands)
    # Assert：每对距离都 >800px，切成 3 事件
    assert len(clusters) == 3


def _two_hoop_events() -> list[dict[str, Any]]:
    """构造两段筐事件：window [0,10.5] 筐位 (300,300) / window [10.5,100] 筐位 (1600,300)。"""
    return [
        {
            "fid": "0011",
            "window": [0.0, 10.5],
            "detected": True,
            "track": [[10.0, 300, 300, "det"]],
        },
        {
            "fid": "0011",
            "window": [10.5, 100.0],
            "detected": True,
            "track": [[60.0, 1600, 300, "det"]],
        },
    ]


def test_cluster_chain_hoop_position_split_cross_court() -> None:
    # Arrange：球位近（100px）但两候选各自时刻筐位差 1300px（换场地）→ 切断
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=11.0, cx=200, cy=100)]
    # Act
    clusters = cluster_candidates(cands, hoop_events=_two_hoop_events())
    # Assert：筐位优先于球位，跨场地切开
    assert len(clusters) == 2


def test_cluster_chain_hoop_missing_pairwise_fallback() -> None:
    # Arrange：一对中其一缺筐（t0=11.5 落在 window 外）→ 该对退回球位比较
    # t0=10.0 有筐；t0=11.5 无筐
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=11.5, cx=250, cy=100)]
    events = [
        {
            "fid": "0011",
            "window": [0.0, 11.0],  # 只含 t0=10.0，t0=11.5 无筐
            "detected": True,
            "track": [[10.0, 300, 300, "det"]],
        }
    ]
    # Act：逐对退回——任一缺筐该对用球位，球位差 150px <=800 → 并
    clusters = cluster_candidates(cands, hoop_events=events)
    # Assert
    assert len(clusters) == 1


def test_cluster_max_event_sec_force_split() -> None:
    # Arrange：每步 gap<=2s 且球位近，可无限链；总跨度 21s > 20s 上限 → 强制切段
    cands = [_cand(t0=float(t), cx=100, cy=100) for t in range(0, 23, 2)]
    # Act
    clusters = cluster_candidates(cands)
    # Assert：t0=22 并入会使跨度 22-0=22 > 20，切为 2 事件
    assert len(clusters) == 2
    assert clusters[0][-1]["t0"] == 20.0
    assert clusters[1][0]["t0"] == 22.0


def test_cluster_max_event_sec_also_caps_merge_rule() -> None:
    # Arrange：补篮时空放宽合并同样受 20s 上限约束（首候选 0s，链到 22s 后
    # 又来一对 gap=5s/dist=100px 本可走 merge 规则并入，但跨度 27s 超限）
    cands = [
        _cand(t0=0.0, cx=100, cy=100),
        _cand(t0=19.0, cx=150, cy=100),
        _cand(t0=24.0, cx=200, cy=100),
    ]
    # Act / Assert：19->24 走 merge 规则本可并，但 24-0=24>20 → 切开
    assert len(cluster_candidates(cands)) == 2


def test_cluster_camera_pan_may_split_documented() -> None:
    # Arrange：同一场地镜头大幅移动——同一 window 内筐轨迹 5s/7s 两点差 850px，
    # t0=5.0 取最近点 (300,300)、t0=6.9 取最近点 (1150,300)，筐位差 850px > 800。
    # 这是 800px 初值下的已知误切风险（spec O1），阈值调整时必须回看本用例。
    events = [
        {
            "fid": "0011",
            "window": [0.0, 20.0],
            "detected": True,
            "track": [[5.0, 300, 300, "det"], [7.0, 1150, 300, "det"]],
        }
    ]
    cands = [_cand(t0=5.0, cx=300, cy=300), _cand(t0=6.9, cx=1000, cy=300)]
    # Act / Assert：镜头大移动致筐位差超阈 → 切（锁定现状，非目标行为背书）
    assert len(cluster_candidates(cands, hoop_events=events)) == 2


def test_cluster_merge_rule_uses_hoop_position_split() -> None:
    # Arrange：gap=3s(>2s) 走时空放宽规则，球位差 200px<=400 但两候选筐位差
    # 1300px（换场地）→ 关联位置距离 >400 → 切断（跨场地必须切开，
    # 实测跨场地对球位常 <400px，只看球位切不开）
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=13.0, cx=300, cy=100)]
    # Act / Assert
    assert len(cluster_candidates(cands, hoop_events=_two_hoop_events())) == 2


def test_cluster_merge_rule_same_hoop_still_merges() -> None:
    # Arrange：补篮回归——两候选筐位同源（同 track），gap=3s、球位 300px → 仍并
    events = [
        {
            "fid": "0011",
            "window": [0.0, 20.0],
            "detected": True,
            "track": [[10.0, 300, 300, "det"], [13.0, 320, 300, "det"]],
        }
    ]
    cands = [_cand(t0=10.0, cx=100, cy=100), _cand(t0=13.0, cx=400, cy=100)]
    # Act / Assert：筐位差 20px <= 400 → 并（阈值 6s/400px 不动，真同球不拆）
    assert len(cluster_candidates(cands, hoop_events=events)) == 1


# ---- 批次 2 改进 3：events_index 筐距 hoop_dist ----


def _hoop_events() -> list[dict[str, Any]]:
    """构造单筐事件：window [0,20]，轨迹两点 (500,300)@5s / (520,300)@10s。"""
    return [
        {
            "fid": "0011",
            "window": [0.0, 20.0],
            "detected": True,
            "track": [[5.0, 500, 300, "det"], [10.0, 520, 300, "det"]],
        }
    ]


def test_event_hoop_dist_min_over_members() -> None:
    # Arrange：成员1 t0=5.0 → 轨迹 sec 最近点 (500,300)，距离 hypot(60,0)=60；
    # 成员2 t0=12.0 → 最近点 (520,300)@10s，距离 hypot(440,240)≈501.2
    members = [_cand(t0=5.0, cx=560, cy=300), _cand(t0=12.0, cx=960, cy=540)]
    # Act
    d = event_hoop_dist(members, _hoop_events())
    # Assert：取各成员最小值
    assert d == pytest.approx(60.0)


def test_event_hoop_dist_no_track_returns_none() -> None:
    # Arrange：window 不含成员 t0 / 无事件 → 无筐轨迹命中
    events = [
        {"fid": "0011", "window": [0.0, 1.0], "detected": True, "track": [[0.0, 1, 1, "det"]]}
    ]
    # Act / Assert
    assert event_hoop_dist([_cand(t0=99.0)], events) is None
    assert event_hoop_dist([_cand(t0=5.0)], []) is None


def test_event_hoop_dist_skips_undetected_event() -> None:
    # Arrange：detected=false 的事件不参与（与 find_event_track 选取逻辑一致）
    events = [{"fid": "0011", "window": [0.0, 20.0], "detected": False, "track": []}]
    # Act / Assert
    assert event_hoop_dist([_cand(t0=5.0)], events) is None


def test_sort_events_by_hoop_dist_ascending_none_last() -> None:
    # Arrange：数值与 None 混合；两条 None 验证稳定排序保持原相对顺序（fid 内时间序）
    events = [
        {"key": "a", "hoop_dist": 570},
        {"key": "b", "hoop_dist": None},
        {"key": "c", "hoop_dist": 66},
        {"key": "d", "hoop_dist": None},
        {"key": "e", "hoop_dist": 260},
    ]
    # Act
    sort_events_by_hoop_dist(events)
    # Assert：升序、None 在尾、None 间保持原顺序
    assert [e["key"] for e in events] == ["c", "e", "a", "b", "d"]


def test_split_window_within_duration() -> None:
    # Arrange / Act / Assert：窗口不越界 → 整段 + 无续接
    assert split_window(2.0, 6.0, 14.0) == ((2.0, 6.0), None)


def test_split_window_overflow_splits_and_caps() -> None:
    # Arrange / Act：越界 4s（恰上限）与越界 11s（超上限）
    seg, cont = split_window(10.0, 18.0, 14.0)
    _, cont2 = split_window(10.0, 25.0, 14.0)
    # Assert：本文件段截到文件末；续接段从 0 起且 MAX_CONT_SEC(4s) 封顶
    assert seg == (10.0, 14.0)
    assert cont == (0.0, 4.0)
    assert cont2 == (0.0, 4.0)


def test_find_next_source_sorted(tmp_path: pathlib.Path) -> None:
    # Arrange：乱序写入三个切片
    for name in ("b_002.mp4", "a_001.mp4", "c_003.mp4"):
        (tmp_path / name).write_bytes(b"x")
    # Act / Assert：按文件名排序取下一个；最后一个与空 srcdir 返回 None
    nxt = find_next_source(str(tmp_path / "a_001.mp4"), str(tmp_path))
    assert nxt is not None
    assert nxt.endswith("b_002.mp4")
    assert find_next_source(str(tmp_path / "c_003.mp4"), str(tmp_path)) is None
    assert find_next_source(str(tmp_path / "a_001.mp4"), "") is None


def test_plan_clip_segments_fallback_on_unprobeable(tmp_path: pathlib.Path) -> None:
    # Arrange：不可 ffprobe 的假视频
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"not-a-video")
    # Act
    segs, cont = plan_clip_segments(str(fake), 1.0, 9.0, str(tmp_path))
    # Assert：探测失败回退单段截断（WARNING 记日志），不炸
    assert cont is False
    assert segs == [(str(fake), 1.0, 9.0)]


# ---- goal-anchor：clip_window 口径与 continued 透出 ----


def test_clip_window_formula_and_clamp() -> None:
    # Arrange：两成员事件与首候选 <2s 的早事件
    members = [_cand(t0=10.0), _cand(t0=14.0, label="#2")]
    early = [_cand(t0=1.0)]
    # Act / Assert：首候选 -CLIP_BEFORE_SEC(2) ~ 末候选 +CLIP_AFTER_SEC(4)；起点钳 0
    assert clip_window(members) == (8.0, 18.0)
    assert clip_window(early) == (0.0, 5.0)


def test_cut_functions_return_continued(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：不真跑 ffmpeg/探测，plan_clip_segments 返回受控续接标志
    monkeypatch.setattr(grc, "_render_segments", lambda segs, vf, out_path: None)
    members = [_cand(t0=10.0)]
    out_c = str(tmp_path / "c.mp4")
    out_w = str(tmp_path / "w.mp4")
    monkeypatch.setattr(grc, "plan_clip_segments", lambda *a, **k: ([("x.mp4", 8.0, 14.0)], True))
    # Act / Assert：续接两态如实透出（events_index continued 字段来源）
    assert cut_cluster_clip("x.mp4", "f", 1, members, out_c) is True
    assert cut_wide_clip("x.mp4", 8.0, 14.0, "wm", out_w) is True
    monkeypatch.setattr(grc, "plan_clip_segments", lambda *a, **k: ([("x.mp4", 8.0, 14.0)], False))
    assert cut_cluster_clip("x.mp4", "f", 1, members, out_c) is False
    assert cut_wide_clip("x.mp4", 8.0, 14.0, "wm", out_w) is False
