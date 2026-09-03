"""build_highlight 单元测试（goals.json schema 校验 + spec 组合真值表 10 分支）。

覆盖：_validate_goals 合法通过、非 confirmed 过滤、未知 status 跳过、各类结构
损坏抛 SchemaError；parse_argv 的 --out/--roster/--team/--per-goal/
--allow-unconfirmed 注入；scale_pad_filter 滤镜串；select_goals 真值表
（①全员 ②旧 scorer 精确匹配+0 命中 WARNING ③全归属球+未归属 WARNING
④tag|name 解析输出名用 tag ⑤队伍合集 ⑥互斥 ⑦无 roster 给 --team
⑧--team 便服）；require_confirmed 拒收未确认 roster；⑨--per-goal 独立进球
片段（命名/互斥/幂等/缺失跳号）；⑩--allow-unconfirmed 闸门（豁免/拒收/误用）。
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Any

import pytest

import build_highlight
from build_highlight import (
    _remove_with_retry,
    _validate_goals,
    apply_name_suffix,
    parse_argv,
    require_confirmed,
    scale_pad_filter,
    select_goals,
)
from errors import BasketballPipelineError, SchemaError
from roster import Player, Roster, format_key

_PATH = "work/pilot/goals.json"


def _goal(**over: object) -> dict[str, Any]:
    """构造一条合法 confirmed 记录，按字段覆盖。"""
    base: dict[str, Any] = {
        "file": "a.MP4",
        "status": "confirmed",
        "anchor_time": 10.0,
        "clip_start": 6.0,
        "clip_end": 12.0,
    }
    base.update(over)
    return base


def test_valid_confirmed_passes() -> None:
    # Arrange
    data = {"session": "s", "goals": [_goal()]}
    # Act
    confirmed = _validate_goals(data, _PATH)
    # Assert
    assert len(confirmed) == 1
    assert confirmed[0]["file"] == "a.MP4"


def test_non_confirmed_status_filtered_out() -> None:
    # Arrange
    data = {"goals": [_goal(status="candidate"), _goal(status="removed")]}
    # Act
    confirmed = _validate_goals(data, _PATH)
    # Assert
    assert confirmed == []


def test_unknown_status_skipped_with_warning() -> None:
    # Arrange：拼错的 status 不应炸掉整批，但会被跳过（WARNING 由 caplog 之外保证）
    data = {"goals": [_goal(status="confirm"), _goal()]}
    # Act
    confirmed = _validate_goals(data, _PATH)
    # Assert：拼错条跳过，合法条保留
    assert len(confirmed) == 1


def test_top_level_not_dict_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="顶层"):
        _validate_goals([1, 2], _PATH)


def test_goals_not_list_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="goals"):
        _validate_goals({"goals": {}}, _PATH)


def test_status_not_str_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="status"):
        _validate_goals({"goals": [_goal(status=None)]}, _PATH)


def test_confirmed_missing_file_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SchemaError, match="file"):
        _validate_goals({"goals": [_goal(file="")]}, _PATH)


def test_confirmed_bool_time_raises() -> None:
    # Arrange：bool 是 int 子类，必须显式排除
    with pytest.raises(SchemaError, match="anchor_time"):
        _validate_goals({"goals": [_goal(anchor_time=True)]}, _PATH)


def test_confirmed_bad_interval_raises() -> None:
    # Arrange：anchor 落在 [clip_start, clip_end] 之外（标注错误）
    with pytest.raises(SchemaError, match="时间区间"):
        _validate_goals({"goals": [_goal(anchor_time=99.0)]}, _PATH)


def test_parse_argv_out_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["build_highlight.py", "--goals", "g.json"])
    # Act
    _, _, _, out_w, out_h, roster, team, per_goal, allow_unconfirmed, name_suffix = parse_argv()
    # Assert：默认保持 4:3 老素材尺寸；roster/team/后缀默认空；⑨⑩ 旗标默认 False
    assert (out_w, out_h) == (1440, 1080)
    assert roster == ""
    assert team == ""
    assert per_goal is False
    assert allow_unconfirmed is False
    assert name_suffix == ""


def test_parse_argv_out_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(
        sys, "argv", ["build_highlight.py", "--goals", "g.json", "--out", "1920x1080"]
    )
    # Act
    _, _, _, out_w, out_h, _, _, _, _, _ = parse_argv()
    # Assert：16:9 场次注入 1920x1080
    assert (out_w, out_h) == (1920, 1080)


def test_parse_argv_roster_team(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_highlight.py", "--goals", "g.json", "--roster", "r.json", "--team", "对手"],
    )
    # Act
    _, _, _, _, _, roster, team, _, _, _ = parse_argv()
    # Assert
    assert roster == "r.json"
    assert team == "对手"


def test_parse_argv_per_goal_and_allow_unconfirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_highlight.py", "--goals", "g.json", "--per-goal", "--allow-unconfirmed"],
    )
    # Act
    *_, per_goal, allow_unconfirmed, name_suffix = parse_argv()
    # Assert：⑨⑩ 旗标各自独立置位（互斥校验在 main，不在 parse_argv）
    assert per_goal is True
    assert allow_unconfirmed is True
    assert name_suffix == ""


def test_parse_argv_name_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_highlight.py", "--goals", "g.json", "--name-suffix", "_4K"],
    )
    # Act
    *_, name_suffix = parse_argv()
    # Assert
    assert name_suffix == "_4K"


class TestApplyNameSuffix:
    """--name-suffix 在尾部类型词（进球集锦/进球合集）之前插入（review02-B4 中段插入）。"""

    def test_empty_suffix_returns_stem_verbatim(self) -> None:
        # Arrange / Act / Assert：缺省空串与现状逐字节一致
        assert apply_name_suffix("队伍_对手_进球集锦", "") == "队伍_对手_进球集锦"
        assert apply_name_suffix("个人_全员_进球合集", "") == "个人_全员_进球合集"

    def test_team_highlight_inserts_before_type_word(self) -> None:
        # Arrange / Act / Assert
        assert apply_name_suffix("队伍_对手_进球集锦", "_4K") == "队伍_对手_4K_进球集锦"

    def test_personal_and_all_inserts_before_type_word(self) -> None:
        # Arrange / Act / Assert
        assert apply_name_suffix("对手_测试员甲_进球合集", "_4K") == "对手_测试员甲_4K_进球合集"
        assert apply_name_suffix("个人_全员_进球合集", "_4K") == "个人_全员_4K_进球合集"

    def test_unknown_stem_raises(self) -> None:
        # Arrange / Act / Assert：命名真值表变更未同步此处时显式报错（不静默）
        with pytest.raises(BasketballPipelineError, match="无法插入后缀"):
            apply_name_suffix("队伍_对手", "_4K")


def test_scale_pad_filter_uses_given_dims() -> None:
    # Arrange / Act
    vf = scale_pad_filter(1920, 1080)
    # Assert
    assert "scale=1920:1080" in vf
    assert "pad=1920:1080" in vf


def _roster(
    players: tuple[Player, ...] = (
        Player(tag="黑21", name="测试员甲", team="对手"),
        Player(tag="白22", name="", team="主队"),
        Player(tag="灰T恤-A", name="", team="便服"),
    ),
    assignments: dict[str, str] | None = None,
    confirmed: bool = True,
) -> Roster:
    """构造一份 Roster 结构体（默认 3 球员：黑/白/便服各一）。"""
    if assignments is None:
        assignments = {format_key("a.MP4", 10.0): "黑21", format_key("b.MP4", 3.0): "白22"}
    return Roster(session="s", confirmed=confirmed, players=players, assignments=assignments)


class TestSelectGoalsTruthTable:
    """spec 组合真值表 8 分支逐一覆盖。"""

    def test_branch1_no_roster_no_filter_all(self) -> None:
        # Arrange：两条 confirmed
        goals = [_goal(), _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0)]
        # Act
        selected, stem = select_goals(goals, None, "", "")
        # Assert：全员；①③ 同名 个人_全员_进球合集
        assert len(selected) == 2
        assert stem == "个人_全员_进球合集"

    def test_branch2_legacy_scorer_exact_match(self) -> None:
        # Arrange：goals.scorer 旧字段精确匹配
        goals = [
            _goal(scorer="测试员甲"),
            _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0, scorer=""),
        ]
        # Act
        selected, stem = select_goals(goals, None, "测试员甲", "")
        # Assert
        assert len(selected) == 1
        assert stem == "个人_测试员甲_进球合集"

    def test_branch2_zero_hit_warns_roster_hint(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange / Act
        with caplog.at_level(logging.WARNING):
            selected, _ = select_goals([_goal()], None, "测试员甲", "")
        # Assert：0 命中 WARNING 提示改用 --roster
        assert selected == []
        assert any("--roster" in r.message for r in caplog.records)

    def test_branch3_roster_no_filter_assigned_only(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange：a.MP4@10.0 已归属黑21，c.MP4 未归属
        goals = [
            _goal(),
            _goal(file="c.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0),
        ]
        # Act
        with caplog.at_level(logging.WARNING):
            selected, stem = select_goals(goals, _roster(), "", "")
        # Assert：只出已归属球；未归属 WARNING 跳过
        assert [g["file"] for g in selected] == ["a.MP4"]
        assert stem == "个人_全员_进球合集"
        assert any("未归属" in r.message for r in caplog.records)

    def test_branch4_scorer_resolved_by_name(self) -> None:
        # Arrange：--scorer 给名字，roster 内 name 命中
        goals = [_goal(), _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0)]
        # Act
        selected, stem = select_goals(goals, _roster(), "测试员甲", "")
        # Assert：输出名用 队伍_姓名（不用 --scorer 原值/tag）
        assert [g["file"] for g in selected] == ["a.MP4"]
        assert stem == "对手_测试员甲_进球合集"

    def test_branch4_scorer_resolved_by_tag(self) -> None:
        # Arrange / Act：tag 直接命中
        selected, stem = select_goals([_goal()], _roster(), "黑21", "")
        # Assert
        assert len(selected) == 1
        assert stem == "对手_测试员甲_进球合集"

    def test_branch4_name_empty_falls_back_to_tag(self) -> None:
        # Arrange / Act：name 空串时输出名回退 tag
        selected, stem = select_goals(
            [_goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0)],
            _roster(),
            "白22",
            "",
        )
        # Assert
        assert len(selected) == 1
        assert stem == "主队_白22_进球合集"

    def test_branch4_scorer_not_in_roster_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(BasketballPipelineError, match="查无此人"):
            select_goals([_goal()], _roster(), "无名氏", "")

    def test_branch5_team_highlight(self) -> None:
        # Arrange：a.MP4→黑21（黑），b.MP4→白22（白）
        goals = [_goal(), _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0)]
        # Act
        selected, stem = select_goals(goals, _roster(), "", "对手")
        # Assert
        assert [g["file"] for g in selected] == ["a.MP4"]
        assert stem == "队伍_对手_进球集锦"

    def test_branch6_scorer_team_mutually_exclusive(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(BasketballPipelineError, match="互斥"):
            select_goals([_goal()], _roster(), "测试员甲", "对手")

    def test_branch7_team_without_roster_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(BasketballPipelineError, match="无法分队"):
            select_goals([_goal()], None, "", "对手")

    def test_branch8_team_casual_raises(self) -> None:
        # Arrange / Act / Assert：便服不进分队合集（有/无 roster 都报错）
        with pytest.raises(BasketballPipelineError, match="便服"):
            select_goals([_goal()], _roster(), "", "便服")
        with pytest.raises(BasketballPipelineError):
            select_goals([_goal()], None, "", "便服")


class TestRequireConfirmed:
    """--roster 未 confirmed=true 拒收（spec 真值表）。"""

    def test_confirmed_passes(self) -> None:
        # Arrange / Act / Assert：不抛即通过
        require_confirmed(_roster(confirmed=True), "r.json")

    def test_unconfirmed_rejected(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(BasketballPipelineError, match="confirmed"):
            require_confirmed(_roster(confirmed=False), "r.json")


class _MainFixture:
    """main() 级测试公共夹具：隔离 cwd、假原片、mock 剪切/拼接（不跑真 ffmpeg）。"""

    def __init__(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self.tmp_path: pathlib.Path = tmp_path
        self.rawdir: pathlib.Path = tmp_path / "raw"
        self.rawdir.mkdir()
        self.goals_path: pathlib.Path = tmp_path / "goals.json"
        self.cut_calls: list[str] = []
        self.ffmpeg_calls: list[list[str]] = []

        def fake_cut(src: str, goal: dict[str, Any], out_path: str, scale_pad: str) -> None:
            self.cut_calls.append(out_path)
            pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(out_path).touch()

        monkeypatch.setattr(build_highlight, "cut_normal", fake_cut)
        monkeypatch.setattr(build_highlight, "cut_slowmo", fake_cut)
        monkeypatch.setattr(
            build_highlight,
            "run_ffmpeg",
            lambda args, **kw: self.ffmpeg_calls.append(list(args)),
        )

    def write_goals(self, goals: list[dict[str, Any]], session: str = "s1") -> None:
        """落盘 goals.json 夹具。"""
        self.goals_path.write_text(
            json.dumps({"session": session, "goals": goals}, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_roster(self, payload: dict[str, Any]) -> pathlib.Path:
        """落盘 roster.json 夹具，返回路径。"""
        path = self.tmp_path / "roster.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def touch_raw(self, *names: str) -> None:
        """造原片占位文件。"""
        for name in names:
            (self.rawdir / name).touch()

    def run_main(self, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
        """以给定参数跑 build_highlight.main()。"""
        monkeypatch.setattr(sys, "argv", ["build_highlight.py", *argv])
        return build_highlight.main()


class TestPerGoal:
    """真值表⑨：--per-goal 每球独立出片到 进球片段/（命名/互斥/幂等/缺失跳号）。"""

    def test_per_goal_outputs_named_clips(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：两球（b.MP4@3.0 时序在前但按 (file, anchor) 排序 a 在前）
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals(
            [
                _goal(),
                _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0),
            ]
        )
        fx.touch_raw("a.MP4", "b.MP4")
        # Act
        rc = fx.run_main(
            monkeypatch, "--goals", str(fx.goals_path), "--rawdir", str(fx.rawdir), "--per-goal"
        )
        # Assert：NNN_<主名>@<t:.1f>s.mp4；不 concat（无 run_ffmpeg 调用）
        assert rc == 0
        names = [pathlib.Path(c).name for c in fx.cut_calls]
        assert names == ["001_a@10.0s.mp4", "002_b@3.0s.mp4"]
        out_dir = tmp_path / "output" / "s1" / "进球片段"
        assert (out_dir / "001_a@10.0s.mp4").is_file()
        assert (out_dir / "002_b@3.0s.mp4").is_file()
        assert fx.ffmpeg_calls == []

    def test_per_goal_slowmo_routing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：slowmo=true 走 cut_slowmo
        fx = _MainFixture(tmp_path, monkeypatch)
        slowmo_calls: list[str] = []
        monkeypatch.setattr(
            build_highlight,
            "cut_slowmo",
            lambda src, goal, out_path, scale_pad: (
                slowmo_calls.append(out_path),
                pathlib.Path(out_path).touch(),
            ),
        )
        fx.write_goals([_goal(slowmo=True)])
        fx.touch_raw("a.MP4")
        # Act
        rc = fx.run_main(
            monkeypatch, "--goals", str(fx.goals_path), "--rawdir", str(fx.rawdir), "--per-goal"
        )
        # Assert
        assert rc == 0
        assert len(slowmo_calls) == 1
        assert fx.cut_calls == []  # cut_normal 未被调（fixture 里 cut_normal 记此处）

    def test_per_goal_mutex_with_filters(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        fx.touch_raw("a.MP4")
        # Act / Assert：⑨ 与 --scorer/--team/--roster 互斥报错
        for extra in (["--scorer", "测试员甲"], ["--team", "黑"], ["--roster", "r.json"]):
            rc = fx.run_main(
                monkeypatch,
                "--goals",
                str(fx.goals_path),
                "--rawdir",
                str(fx.rawdir),
                "--per-goal",
                *extra,
            )
            assert rc == 1
        assert fx.cut_calls == []

    def test_per_goal_idempotent_skip(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：同名产物已存在 → 幂等跳过不重复转码
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        fx.touch_raw("a.MP4")
        existing = tmp_path / "output" / "s1" / "进球片段" / "001_a@10.0s.mp4"
        existing.parent.mkdir(parents=True)
        existing.touch()
        # Act
        rc = fx.run_main(
            monkeypatch, "--goals", str(fx.goals_path), "--rawdir", str(fx.rawdir), "--per-goal"
        )
        # Assert
        assert rc == 0
        assert fx.cut_calls == []

    def test_per_goal_missing_source_keeps_numbering_exit1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：b.MP4 原片缺失 → 002 跳号保留（不 compact），c 照常出 003
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals(
            [
                _goal(),
                _goal(file="b.MP4", anchor_time=3.0, clip_start=0.0, clip_end=5.0),
                _goal(file="c.MP4", anchor_time=5.0, clip_start=1.0, clip_end=7.0),
            ]
        )
        fx.touch_raw("a.MP4", "c.MP4")
        # Act
        rc = fx.run_main(
            monkeypatch, "--goals", str(fx.goals_path), "--rawdir", str(fx.rawdir), "--per-goal"
        )
        # Assert
        assert rc == 1
        names = [pathlib.Path(c).name for c in fx.cut_calls]
        assert names == ["001_a@10.0s.mp4", "003_c@5.0s.mp4"]


class TestAllowUnconfirmed:
    """真值表⑩：--allow-unconfirmed 仅豁免 confirmed 检查；① stem 改名回归。"""

    def _roster_payload(self, confirmed: bool) -> dict[str, Any]:
        """auto roster 风格载荷：team 取颜色队别 / name 空 / tag 字母。

        （"球员" 为历史合法值——T3R 后 auto_roster 产 黑/白/便服 多数票，
        此处仅构造最小合法载荷，不断言 team 语义。）
        """
        return {
            "session": "s1",
            "confirmed": confirmed,
            "players": [{"tag": "A", "name": "", "team": "球员"}],
            "assignments": {format_key("a.MP4", 10.0): "A"},
        }

    def test_unconfirmed_roster_with_flag_builds(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：未 confirmed roster + 闸门旗标 → 放行，④ 命名产 球员_A_进球合集
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        fx.touch_raw("a.MP4")
        roster_path = fx.write_roster(self._roster_payload(confirmed=False))
        # Act
        rc = fx.run_main(
            monkeypatch,
            "--goals",
            str(fx.goals_path),
            "--rawdir",
            str(fx.rawdir),
            "--roster",
            str(roster_path),
            "--allow-unconfirmed",
            "--scorer",
            "A",
        )
        # Assert：concat 末参数 = 输出路径
        assert rc == 0
        assert fx.ffmpeg_calls[-1][-1].endswith("球员_A_进球合集.mp4")

    def test_unconfirmed_roster_without_flag_rejected(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange / Act / Assert：旧契约不动——显式 --roster 未确认仍拒收
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        fx.touch_raw("a.MP4")
        roster_path = fx.write_roster(self._roster_payload(confirmed=False))
        rc = fx.run_main(
            monkeypatch,
            "--goals",
            str(fx.goals_path),
            "--rawdir",
            str(fx.rawdir),
            "--roster",
            str(roster_path),
        )
        assert rc == 1
        assert fx.cut_calls == []

    def test_flag_without_roster_rejected(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange / Act / Assert：无 --roster 给旗标 = 无意义组合，显式失败
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        rc = fx.run_main(
            monkeypatch,
            "--goals",
            str(fx.goals_path),
            "--rawdir",
            str(fx.rawdir),
            "--allow-unconfirmed",
        )
        assert rc == 1
        assert fx.cut_calls == []

    def test_branch1_output_name(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange / Act：① 无 roster 无过滤
        fx = _MainFixture(tmp_path, monkeypatch)
        fx.write_goals([_goal()])
        fx.touch_raw("a.MP4")
        rc = fx.run_main(monkeypatch, "--goals", str(fx.goals_path), "--rawdir", str(fx.rawdir))
        # Assert：①③ 同名 个人_全员_进球合集（③ 由 test_branch3 锁定）
        assert rc == 0
        assert fx.ffmpeg_calls[-1][-1].endswith("个人_全员_进球合集.mp4")


class TestRemoveWithRetry:
    """_remove_with_retry：Windows 瞬时文件锁退避重试（2026-08-22 真机 WinError 32 实录）。"""

    def test_first_try_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(build_highlight.os, "remove", calls.append)
        _remove_with_retry("x.mp4")
        assert calls == ["x.mp4"]

    def test_transient_lock_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts: list[str] = []
        sleeps: list[float] = []

        def fake_remove(path: str) -> None:
            attempts.append(path)
            if len(attempts) < 3:
                raise PermissionError("locked")

        monkeypatch.setattr(build_highlight.os, "remove", fake_remove)
        monkeypatch.setattr(build_highlight.time, "sleep", sleeps.append)
        _remove_with_retry("x.mp4")
        assert len(attempts) == 3
        assert sleeps == [0.5, 1.0]

    def test_persistent_lock_raises_after_exhaustion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(build_highlight.time, "sleep", lambda _: None)

        def always_locked(path: str) -> None:
            raise PermissionError("locked")

        monkeypatch.setattr(build_highlight.os, "remove", always_locked)
        with pytest.raises(PermissionError):
            _remove_with_retry("x.mp4")

    def test_other_oserror_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(build_highlight.time, "sleep", sleeps.append)

        def not_found(path: str) -> None:
            raise FileNotFoundError(path)

        monkeypatch.setattr(build_highlight.os, "remove", not_found)
        with pytest.raises(FileNotFoundError):
            _remove_with_retry("x.mp4")
        assert sleeps == []
