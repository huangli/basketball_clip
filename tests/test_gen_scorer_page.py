"""gen_scorer_page 单元测试（spec: docs/scorer/spec.md T4 认人确认页）。

覆盖：team_of_tag 前缀推队、parse_players 名单解析、merge_assignments 并集/
同键冲突、match_clip 4s 容差匹配与相对路径、build_entries 排序与无候选兜底、
build_html 内联数据与导出契约、main 端到端（tmp 目录写 scorer.html）；
--clusters 簇级确认（docs/scorer-cluster/spec.md）：clusters schema 校验、
build_cluster_map 归属与越界 key 跳过、cluster_id 注入与 unclustered→None、
build_page_clusters 过滤、簇区渲染与 node --check JS 语法校验；
--players-file 名单文件注入（docs/scorer-reid/spec.md Phase D）：合法名单解析、
坏 JSON/坏结构/非法队名 SchemaError、与 --players 互斥、号码预填链路命中；
--photo-matches 照片库预填（docs/photo-roster/spec.md T5）：优先级 读号>照片>印名、
冲突角标、名单缺号占位注入、同目录校验、坏 schema 退出 1、无参数零变化；
--track-links 轨迹传播预填（docs/scorer-propagate/spec.md §页面）：track_links
schema 校验、build_track_map 跨批 key 跳过、track_id 注入与同目录校验、
页面 JS（轨迹#N/同轨迹预填徽标/provenance 键/NOGOAL 不传播/acceptAll 隔离）、
node --check JS 语法校验。
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess

import pytest

from errors import BasketballPipelineError, SchemaError
from gen_scorer_page import (
    OPPONENT_TAG,
    PLAYER_KEYS,
    PhotoGuess,
    _validate_clusters,
    _validate_track_links,
    build_cluster_map,
    build_entries,
    build_html,
    build_keymap,
    build_page_clusters,
    build_track_map,
    cluster_sort_key,
    load_players_file,
    main,
    match_clip,
    match_players_by_name,
    match_players_by_number,
    merge_assignments,
    opponent_of,
    parse_players,
    resolve_photo_guesses,
    team_of_tag,
)
from photo_match_scorers import MATCH_VERSION, MatchEntry
from roster import Player, format_key


def _goal(file: str = "a.mp4", anchor: float = 4.1) -> dict:
    """构造一条合法 confirmed 记录。"""
    return {
        "file": file,
        "anchor_time": anchor,
        "clip_start": max(0.0, anchor - 4.0),
        "clip_end": anchor + 2.0,
        "status": "confirmed",
        "scorer": "",
    }


def _candidate(
    file: str = "a.mp4",
    anchor: float = 4.1,
    status: str = "OK",
    crop: str = "a_t4.1.jpg",
    team_guess: str | None = "黑",
) -> dict:
    """构造一条合法候选记录（key 走 roster.format_key 契约）。"""
    return {
        "key": format_key(file, anchor),
        "file": file,
        "anchor_time": anchor,
        "status": status,
        "reason": "" if status == "OK" else "few_votes",
        "crop": crop,
        "team_guess": team_guess,
        "votes": 3,
        "total_votes": 10,
    }


def _event(src_file: str = "a.mp4", anchor_t0: float = 4.0, clip: str = "clips/a_e1.mp4") -> dict:
    """构造一条合法事件记录。"""
    return {
        "key": "a#e1",
        "fid": "a",
        "event_idx": 1,
        "clip": clip,
        "clip_wide": "clips/a_e1_wide.mp4",
        "src_file": src_file,
        "anchor_t0": anchor_t0,
        "verdict": "?",
    }


class TestOpponentOf:
    """对手队名派生：场次 ID 后缀；无后缀/空白后缀回退 team_config.DEFAULT_OPPONENT。"""

    def test_suffix(self) -> None:
        assert opponent_of("20260805_蓝队") == "蓝队"

    def test_no_suffix_fallback(self) -> None:
        assert opponent_of("20260722") == "对手"

    def test_blank_suffix_fallback(self) -> None:
        assert opponent_of("20260722_") == "对手"


class TestTeamOfTag:
    """标签前缀推队（页面 JS teamOfTag 同规则；对手队名由 opp 参数注入）。"""

    def test_prefix_teams(self) -> None:
        # Arrange / Act / Assert
        assert team_of_tag("黑21", "对手") == "对手"
        assert team_of_tag("白-李四", "对手") == "主队"
        assert team_of_tag("灰T恤-A", "对手") == "便服"

    def test_blue_prefix_maps_to_opponent_team(self) -> None:
        # Arrange / Act / Assert：蓝 → 对手队（蓝27 归对手系 2026-08-09 用户口径）
        assert team_of_tag("蓝27", "对手") == "对手"

    def test_opponent_name_follows_opp_arg(self) -> None:
        # Arrange / Act / Assert：对手队名随 opp 走，不硬编码（2026-08-15 队名会话化）
        assert team_of_tag("黑21", "蓝队") == "蓝队"
        assert team_of_tag("蓝27", "蓝队") == "蓝队"

    def test_opponent_pseudo_player_maps_to_opp_team(self) -> None:
        # Arrange / Act / Assert：对手伪球员不走前缀兜底便服
        assert team_of_tag(OPPONENT_TAG, "蓝队") == "蓝队"


class TestKeymap:
    """选人键位映射：27 键、无冲突、超出无键、E 别名边界。"""

    def test_player_keys_count_and_function_disjoint(self) -> None:
        # Arrange / Act / Assert
        assert len(PLAYER_KEYS) == 27
        assert not (set(PLAYER_KEYS) & {"O", "P", "S", "N", "M", "X", "Z", "G", "/"})

    def test_build_keymap_first_27_get_keys(self) -> None:
        # Arrange
        players = [Player(tag=f"P{i}", name="", team="主队") for i in range(30)]
        # Act
        km = build_keymap(players)
        # Assert
        assert [m["key"] for m in km[:27]] == list(PLAYER_KEYS)
        assert all(m["key"] == "" for m in km[27:])
        assert km[0]["tag"] == "P0"

    def test_build_keymap_e_alias_when_le_11(self) -> None:
        # Arrange / Act
        km = build_keymap([Player(tag="黑21", name="", team="对手")])
        # Assert：≤11 人时 E 不在 keymap，页面作 Enter 别名
        assert "E" not in [m["key"] for m in km]

    def test_build_keymap_e_becomes_player_12(self) -> None:
        # Arrange / Act
        players = [Player(tag=f"P{i}", name="", team="主队") for i in range(12)]
        km = build_keymap(players)
        # Assert：第 12 个球员（0-based 11）分配 E
        assert km[11]["key"] == "E"


class TestClusterSortKey:
    """簇队列排序：球数降序，同数按首球锚点时间升序。"""

    def test_sort_by_size_then_anchor(self) -> None:
        # Arrange
        by_key = {
            "a.mp4#1.0": {"anchor_time": 10.0},
            "a.mp4#2.0": {"anchor_time": 20.0},
            "a.mp4#3.0": {"anchor_time": 5.0},
            "a.mp4#4.0": {"anchor_time": 1.0},
        }
        # Act
        groups = [
            ["a.mp4#3.0"],  # 1 球，首球 5
            ["a.mp4#1.0", "a.mp4#2.0"],  # 2 球，首球 10
            ["a.mp4#4.0"],  # 1 球，首球 1
        ]
        groups.sort(key=lambda ks: cluster_sort_key(ks, by_key))
        # Assert
        assert [len(g) for g in groups] == [2, 1, 1]
        assert groups[1] == ["a.mp4#4.0"]


class TestParsePlayers:
    """--players 名单串解析。"""

    def test_tag_name_pairs(self) -> None:
        # Arrange / Act
        players = parse_players("黑21=张三,白-李四=李四,白-王五=王五", "对手")
        # Assert
        assert players == [
            Player(tag="黑21", name="张三", team="对手"),
            Player(tag="白-李四", name="李四", team="主队"),
            Player(tag="白-王五", name="王五", team="主队"),
        ]

    def test_name_optional_and_empty_spec(self) -> None:
        # Arrange / Act / Assert
        assert parse_players("", "对手") == []
        assert parse_players("黑21", "对手") == [Player(tag="黑21", name="", team="对手")]

    def test_opp_arg_flows_to_team(self) -> None:
        # Arrange / Act / Assert：黑/蓝前缀队名 = opp 参数（队名会话化）
        players = parse_players("黑21,蓝27", "蓝队")
        assert [p.team for p in players] == ["蓝队", "蓝队"]

    def test_missing_tag_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="tag"):
            parse_players("=张三", "对手")


# ---- --players-file 名单文件注入（docs/scorer-reid/spec.md Phase D） ----


def _write_players_file(tmp_path: pathlib.Path, payload: object) -> pathlib.Path:
    """把名单 payload 写成 JSON 文件，返回路径。"""
    path = tmp_path / "players.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadPlayersFile:
    """--players-file 名单文件解析：与 roster.players 同构，复用 roster 校验。"""

    def test_valid_players_parsed(self, tmp_path: pathlib.Path) -> None:
        # Arrange：文件里的 team 以文件为准（不做前缀推定）
        path = _write_players_file(
            tmp_path,
            [
                {"tag": "白22-王五", "name": "王五", "team": "主队"},
                {"tag": "黑21-张三", "name": "张三", "team": "对手"},
                {"tag": "灰T恤-A", "name": "", "team": "便服"},
            ],
        )
        # Act
        players = load_players_file(path)
        # Assert
        assert players == [
            Player(tag="白22-王五", name="王五", team="主队"),
            Player(tag="黑21-张三", name="张三", team="对手"),
            Player(tag="灰T恤-A", name="", team="便服"),
        ]

    def test_bad_json_raises(self, tmp_path: pathlib.Path) -> None:
        # Arrange：文件内容不是合法 JSON
        path = tmp_path / "players.json"
        path.write_text("[{not json", encoding="utf-8")
        # Act / Assert
        with pytest.raises(SchemaError):
            load_players_file(path)

    def test_top_level_not_list_raises(self, tmp_path: pathlib.Path) -> None:
        # Arrange：顶层是对象而非数组（roster.json 整文件误传场景）
        path = _write_players_file(tmp_path, {"players": []})
        # Act / Assert
        with pytest.raises(SchemaError, match="数组"):
            load_players_file(path)

    def test_invalid_team_raises(self, tmp_path: pathlib.Path) -> None:
        # Arrange：team 空串（team 已放宽为任意非空 str，见 docs/session-opponent-name/spec.md）
        path = _write_players_file(tmp_path, [{"tag": "白22-王五", "name": "王五", "team": ""}])
        # Act / Assert
        with pytest.raises(SchemaError, match="team"):
            load_players_file(path)

    def test_duplicate_tag_raises(self, tmp_path: pathlib.Path) -> None:
        # Arrange：tag 重复（roster 契约同一校验）
        path = _write_players_file(
            tmp_path,
            [
                {"tag": "白22-王五", "name": "王五", "team": "主队"},
                {"tag": "白22-王五", "name": "王五", "team": "主队"},
            ],
        )
        # Act / Assert
        with pytest.raises(SchemaError, match="重复"):
            load_players_file(path)

    def test_number_prefill_with_file_players(self, tmp_path: pathlib.Path) -> None:
        # Arrange：文件名单注入后走号码预填链路（match_players_by_number 命中）
        path = _write_players_file(tmp_path, [{"tag": "白22-王五", "name": "王五", "team": "主队"}])
        players = load_players_file(path)
        cand = _candidate()
        cand["number_guess"] = {
            "number": "22",
            "color": "白",
            "name_text": None,
            "confidence": "high",
        }
        # Act
        entries = build_entries([_goal()], [cand], None, "", "", players)
        # Assert
        assert entries[0]["prefill_tag"] == "白22-王五"
        assert entries[0]["prefill_note"] == ""


class TestPlayersFileCli:
    """--players-file CLI 层：与 --players 互斥、端到端生成页面。"""

    def _write_inputs(self, tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        """造 scorers/goals 两个输入文件，返回路径。"""
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(
            json.dumps({"session": "s", "candidates": [_candidate()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps({"session": "s", "goals": [_goal()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return scorers, goals

    def test_mutex_with_players_rejected(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals = self._write_inputs(tmp_path)
        players_file = _write_players_file(tmp_path, [])
        # Act / Assert：同给两源 → parser.error 显式拒绝（SystemExit 2）
        with pytest.raises(SystemExit):
            main(
                [
                    "--scorers",
                    str(scorers),
                    "--goals",
                    str(goals),
                    "--players",
                    "黑21=张三",
                    "--players-file",
                    str(players_file),
                ]
            )

    def test_end_to_end_with_players_file(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals = self._write_inputs(tmp_path)
        players_file = _write_players_file(
            tmp_path, [{"tag": "白22-王五", "name": "王五", "team": "主队"}]
        )
        # Act
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--players-file",
                str(players_file),
            ]
        )
        # Assert：页面正常生成，文件名单内联为按钮名单
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"tag": "白22-王五"' in html
        assert '"team": "主队"' in html

    def test_bad_players_file_exit_1(self, tmp_path: pathlib.Path) -> None:
        # Arrange：名单文件 schema 损坏（team 空串）
        scorers, goals = self._write_inputs(tmp_path)
        players_file = _write_players_file(tmp_path, [{"tag": "白22", "name": "王五", "team": ""}])
        # Act：SchemaError 经 main 转为退出 1（显式失败不静默）
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--players-file",
                str(players_file),
            ]
        )
        # Assert
        assert rc == 1


class TestMergeAssignments:
    """--roster-existing 并集合并（spec T4）。"""

    def test_union_disjoint(self) -> None:
        # Arrange / Act
        merged = merge_assignments({"a.mp4#4.1": "黑21"}, {"b.mp4#2.0": "白22"})
        # Assert
        assert merged == {"a.mp4#4.1": "黑21", "b.mp4#2.0": "白22"}

    def test_same_key_same_value_ok(self) -> None:
        # Arrange / Act
        merged = merge_assignments({"a.mp4#4.1": "黑21"}, {"a.mp4#4.1": "黑21"})
        # Assert
        assert merged == {"a.mp4#4.1": "黑21"}

    def test_same_key_conflict_raises(self) -> None:
        # Arrange / Act / Assert：同键不同值显式失败（调用方退出 1）
        with pytest.raises(BasketballPipelineError, match="同键冲突"):
            merge_assignments({"a.mp4#4.1": "黑21"}, {"a.mp4#4.1": "白22"})


class TestMatchClip:
    """审核片段匹配：src_file 相同且 |anchor_t0−anchor|≤4s，取最近者。"""

    def test_match_within_tolerance(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        events = [_event(anchor_t0=6.0), _event(anchor_t0=100.0)]
        # Act
        rel = match_clip(events, "a.mp4", 4.1, str(index_dir), str(out_dir))
        # Assert：命中 6.0（|6.0−4.1|=1.9≤4s），优先全景 clip_wide，相对路径用正斜杠
        assert rel == "../review_v3/clips/a_e1_wide.mp4"

    def test_fallback_to_clip_when_no_wide(self, tmp_path: pathlib.Path) -> None:
        # Arrange：无 clip_wide 字段回退筐区 clip
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        ev = _event(anchor_t0=6.0)
        del ev["clip_wide"]
        # Act
        rel = match_clip([ev], "a.mp4", 4.1, str(index_dir), str(out_dir))
        # Assert
        assert rel == "../review_v3/clips/a_e1.mp4"

    def test_backslash_clip_normalized(self, tmp_path: pathlib.Path) -> None:
        # Arrange：events_index 实跑产物是 Windows 反斜杠
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        ev = _event(anchor_t0=4.0)
        ev["clip_wide"] = "clips\\a_e1_wide.mp4"
        # Act
        rel = match_clip([ev], "a.mp4", 4.1, str(index_dir), str(out_dir))
        # Assert
        assert rel == "../review_v3/clips/a_e1_wide.mp4"

    def test_no_match_returns_empty(self, tmp_path: pathlib.Path) -> None:
        # Arrange：异源文件 / 超容差 都不匹配
        events = [_event(src_file="b.mp4", anchor_t0=4.0), _event(anchor_t0=20.0)]
        # Act / Assert
        assert match_clip(events, "a.mp4", 4.1, str(tmp_path), str(tmp_path)) == ""


class TestMatchPlayersByNumber:
    """号码+颜色 → 名单 tag 匹配（含歧义与数字边界）。"""

    @staticmethod
    def _players() -> list[Player]:
        """本场名单（含两个黑21，供歧义分支）。"""
        return [
            Player(tag="黑21-张三", name="张三", team="对手"),
            Player(tag="黑21-赵六", name="赵六", team="对手"),
            Player(tag="白-李四", name="李四", team="主队"),
            Player(tag="蓝色27", name="", team="便服"),
            Player(tag="赛文21", name="", team="便服"),
        ]

    def test_single_match(self) -> None:
        # Arrange / Act
        got = match_players_by_number(self._players(), "21", "黑")
        # Assert：三个 21 中颜色滤掉赛文21，剩两个黑21（调用方判歧义）
        assert [p.tag for p in got] == ["黑21-张三", "黑21-赵六"]

    def test_unique_number_ignores_color_misread(self) -> None:
        # Arrange / Act / Assert：27 唯一，颜色误读为黑也命中蓝色27
        assert [p.tag for p in match_players_by_number(self._players(), "27", "黑")] == ["蓝色27"]

    def test_blue_color_match(self) -> None:
        # Arrange / Act / Assert
        assert [p.tag for p in match_players_by_number(self._players(), "27", "蓝")] == ["蓝色27"]

    def test_wrong_color_falls_back_to_ambiguous(self) -> None:
        # Arrange：三个 21 用白色过滤为空 → 回退歧义全集（不放过潜在误杀）
        # Act / Assert
        got = match_players_by_number(self._players(), "21", "白")
        assert [p.tag for p in got] == ["黑21-张三", "黑21-赵六", "赛文21"]

    def test_digit_boundary_no_substring(self) -> None:
        # Arrange / Act / Assert：号码 "2" 不误中 "黑21"
        assert match_players_by_number(self._players(), "2", "黑") == []

    def test_none_number_no_match(self) -> None:
        # Arrange / Act / Assert：无号码不参与匹配
        assert match_players_by_number(self._players(), None, "黑") == []

    def test_same_number_no_color_is_ambiguous(self) -> None:
        # Arrange / Act / Assert：同号无颜色提示 → 返回全部候选（歧义）
        got = match_players_by_number(self._players(), "21", None)
        assert len(got) == 3


class TestMatchPlayersByName:
    """印名模糊匹配：精确或差 1 字符（K3 误读容差），只看非空 name。"""

    def _players(self) -> list:
        return [
            Player(tag="黑21-张三", name="张三", team="对手"),
            Player(tag="黑21-赵六", name="赵六", team="对手"),
            Player(tag="蓝色27", name="", team="便服"),
        ]

    def test_exact_name_match(self) -> None:
        # Arrange / Act / Assert
        assert [p.tag for p in match_players_by_name(self._players(), "张三")] == ["黑21-张三"]

    def test_one_char_misread_match(self) -> None:
        # Arrange / Act / Assert：K3 把"张三"读成"大秋"（差 1 字符）仍命中
        assert [p.tag for p in match_players_by_name(self._players(), "张二")] == ["黑21-张三"]

    def test_no_match_and_empty(self) -> None:
        # Arrange / Act / Assert：无关文本与空值不中
        assert match_players_by_name(self._players(), "杭州60岁") == []
        assert match_players_by_name(self._players(), "") == []
        assert match_players_by_name(self._players(), None) == []


class TestNumberPrefill:
    """条目预填：号码匹配 > 颜色；同号多人歧义不预填。"""

    def _candidate_with_number(self, number: str | None, color: str | None) -> dict:
        """造一条带 number_guess 的候选。"""
        c = _candidate()
        c["number_guess"] = {
            "number": number,
            "color": color,
            "name_text": None,
            "confidence": "high" if number else "low",
        }
        return c

    def test_unique_number_match_prefills(self) -> None:
        # Arrange：名单只有一个 黑21
        players = [Player(tag="黑21-张三", name="张三", team="对手")]
        # Act
        entries = build_entries(
            [_goal()], [self._candidate_with_number("21", "黑")], None, "", "", players
        )
        # Assert：号码预填压过颜色 team_guess（候选 team_guess=黑）
        assert entries[0]["prefill_tag"] == "黑21-张三"
        assert entries[0]["prefill_note"] == ""

    def test_ambiguous_same_number_no_prefill(self) -> None:
        # Arrange：两个黑21 → 歧义
        players = [
            Player(tag="黑21-张三", name="张三", team="对手"),
            Player(tag="黑21-赵六", name="赵六", team="对手"),
        ]
        # Act
        entries = build_entries(
            [_goal()], [self._candidate_with_number("21", "黑")], None, "", "", players
        )
        # Assert
        assert entries[0]["prefill_tag"] == ""
        assert entries[0]["prefill_note"] == "ambiguous"

    def test_no_number_no_prefill(self) -> None:
        # Arrange：K3 没读出号码
        players = [Player(tag="黑21-张三", name="张三", team="对手")]
        # Act
        entries = build_entries(
            [_goal()], [self._candidate_with_number(None, "黑")], None, "", "", players
        )
        # Assert：回退颜色预填（prefill 字段为空，页面显示 team_guess）
        assert entries[0]["prefill_tag"] == ""
        assert entries[0]["prefill_note"] == ""
        assert entries[0]["team_guess"] == "黑"


class TestBuildEntries:
    """页面条目组装：confirmed 球为全集，按 key 关联候选。"""

    def test_sorted_and_joined(self) -> None:
        # Arrange：goals 乱序，候选含 SKIP
        goals = [_goal("b.mp4", 2.0), _goal("a.mp4", 4.1)]
        candidates = [
            _candidate("a.mp4", 4.1),
            _candidate("b.mp4", 2.0, status="SKIP", crop="", team_guess=None),
        ]
        # Act
        entries = build_entries(goals, candidates, None, "", "")
        # Assert：按 file+anchor 排序；SKIP 无裁图无预填
        assert [e["key"] for e in entries] == ["a.mp4#4.1", "b.mp4#2.0"]
        assert entries[0]["crop"] == "a_t4.1.jpg"
        assert entries[0]["team_guess"] == "黑"
        assert entries[0]["clip"] == ""  # 无 --index
        assert entries[1]["status"] == "SKIP"

    def test_goal_without_candidate_becomes_skip(self) -> None:
        # Arrange / Act
        entries = build_entries([_goal()], [], None, "", "")
        # Assert：防御兜底，不炸、按 SKIP 列出
        assert entries[0]["status"] == "SKIP"
        assert entries[0]["reason"] == "no_candidate"
        assert entries[0]["crop"] == ""

    def test_clip_filled_when_events_given(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        # Act
        entries = build_entries(
            [_goal()], [_candidate()], [_event(anchor_t0=4.0)], str(index_dir), str(out_dir)
        )
        # Assert
        assert entries[0]["clip"] == "../review_v3/clips/a_e1_wide.mp4"

    def test_candidate_clip_preferred_over_events(self, tmp_path: pathlib.Path) -> None:
        # Arrange：candidates 带现切预览片段时优先于 events_index 匹配
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        cand = _candidate()
        cand["clip"] = "clips/a_t4.1.mp4"
        # Act
        entries = build_entries(
            [_goal()], [cand], [_event(anchor_t0=4.0)], str(index_dir), str(out_dir)
        )
        # Assert：与裁图同锚点的预览片段胜出，不走事件兜底
        assert entries[0]["clip"] == "clips/a_t4.1.mp4"

    def test_empty_candidate_clip_falls_back_to_events(self, tmp_path: pathlib.Path) -> None:
        # Arrange：candidates 有 clip 字段但为空串（切片失败）→ 仍走事件兜底
        index_dir = tmp_path / "s" / "review_v3"
        out_dir = tmp_path / "s" / "scorers"
        index_dir.mkdir(parents=True)
        out_dir.mkdir()
        cand = _candidate()
        cand["clip"] = ""
        # Act
        entries = build_entries(
            [_goal()], [cand], [_event(anchor_t0=4.0)], str(index_dir), str(out_dir)
        )
        # Assert
        assert entries[0]["clip"] == "../review_v3/clips/a_e1_wide.mp4"


class TestBuildHtml:
    """HTML 渲染：数据内联、导出契约、交互控件。"""

    def test_inlines_items_players_session(self) -> None:
        # Arrange
        entries = build_entries([_goal()], [_candidate()], None, "", "")
        players = [Player(tag="黑21", name="张三", team="对手")]
        # Act
        html = build_html(entries, players, "20260722", {}, {}, "对手")
        # Assert
        assert '"key": "a.mp4#4.1"' in html
        assert '"tag": "黑21"' in html
        assert 'const SESSION = "20260722";' in html

    def test_progress_localstorage_key_contains_session(self) -> None:
        # Arrange / Act
        html = build_html([], [], "mysession", {}, {}, "对手")
        # Assert
        assert '"scorer_" + SESSION' in html
        assert "localStorage" in html

    def test_export_contract_roster_json(self) -> None:
        # Arrange / Act
        html = build_html([], [], "20260722", {}, {}, "对手")
        # Assert：导出结构字段与文件名契约（roster.py validate_roster 可过；
        # roster-export-name：下载名即 roster.json，移到 work/<场次>/ 直接接入 CLI）
        assert 'a.download = "roster.json";' in html
        assert "confirmed" in html
        assert "assignments" in html
        assert "players" in html
        # confirmed 条件：全部非 SKIP 球已归属
        assert 'it.status === "SKIP" || marks[it.key]' in html

    def test_new_ui_mode_and_keys(self) -> None:
        # Arrange / Act
        html = build_html([], [], "s", {}, {}, "对手")
        # Assert：新单流界面关键标识
        assert "无法定位" in html
        assert 'id="clusterView"' in html
        assert 'id="ballView"' in html
        assert "模式一：簇审阅" in html
        assert "模式二：逐球收尾" in html
        assert "const PLAYER_KEYS = [" in html
        assert "function assignCluster(" in html
        assert "function assignBall(" in html

    def test_existing_assignments_inlined(self) -> None:
        # Arrange / Act
        html = build_html([], [], "s", {"a.mp4#4.1": "黑21"}, {}, "对手")
        # Assert：已有 roster 归属内联作预填底色
        assert '"a.mp4#4.1": "黑21"' in html

    def test_opponent_injected_and_semantic_css(self) -> None:
        # Arrange / Act
        html = build_html([], [], "20260805_蓝队", {}, {}, "蓝队")
        # Assert：对手队名注入 JS（允许页内覆盖故用 let）
        assert 'let OPP = "蓝队";' in html
        assert "team-opp" in html
        assert "team-home" in html
        assert "team-casual" in html
        assert "function teamClass(" in html
        assert 'OPPONENT_TAG = "对手"' in html


class TestMain:
    """main 端到端：tmp 目录造输入，产出 scorer.html。"""

    def _write_inputs(self, tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        """造 scorers/goals 两个输入文件，返回路径。"""
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(
            json.dumps({"session": "s", "candidates": [_candidate()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps({"session": "s", "goals": [_goal()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return scorers, goals

    def test_generates_scorer_html(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals = self._write_inputs(tmp_path)
        # Act
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--session",
                "s",
                "--players",
                "黑21=张三",
            ]
        )
        # Assert：默认输出 <scorers 同目录>/scorer.html
        assert rc == 0
        html_path = scorers.parent / "scorer.html"
        assert html_path.is_file()
        html = html_path.read_text(encoding="utf-8")
        assert '"key": "a.mp4#4.1"' in html

    def test_roster_existing_merged(self, tmp_path: pathlib.Path) -> None:
        # Arrange：已有 roster 的归属应内联进页面
        scorers, goals = self._write_inputs(tmp_path)
        existing = tmp_path / "roster.json"
        existing.write_text(
            json.dumps(
                {
                    "session": "s",
                    "confirmed": True,
                    "players": [{"tag": "黑21", "name": "张三", "team": "对手"}],
                    "assignments": {"a.mp4#4.1": "黑21"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # Act
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--roster-existing",
                str(existing),
            ]
        )
        # Assert
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"a.mp4#4.1": "黑21"' in html

    def test_bad_candidates_schema_exit_1(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(json.dumps({"candidates": [{"key": 1}]}), encoding="utf-8")
        goals = tmp_path / "goals.json"
        goals.write_text(json.dumps({"goals": []}), encoding="utf-8")
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--session", "s"])
        # Assert：schema 损坏显式失败
        assert rc == 1


def test_match_clip_picks_closest(tmp_path: pathlib.Path) -> None:
    # Arrange：两个同文件事件都在容差内，取时间差最小者（全景优先）
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    far = _event(anchor_t0=1.0, clip="clips/far.mp4")
    far["clip_wide"] = "clips/far_wide.mp4"
    near = _event(anchor_t0=4.0, clip="clips/near.mp4")
    near["clip_wide"] = "clips/near_wide.mp4"
    events = [far, near]
    # Act
    rel = match_clip(events, "a.mp4", 4.1, str(index_dir), str(tmp_path))
    # Assert
    assert rel.endswith("clips/near_wide.mp4")


def test_entries_clip_empty_without_index() -> None:
    # Arrange / Act：无 --index 时只显示裁图
    entries = build_entries([_goal()], [_candidate()], None, "", "")
    # Assert
    assert entries[0]["clip"] == ""


def test_validate_candidates_bad_status(tmp_path: pathlib.Path) -> None:
    # Arrange：status 非法值必须显式失败（经 main 退出 1）
    scorers_dir = tmp_path / "scorers"
    scorers_dir.mkdir()
    scorers = scorers_dir / "scorer_candidates.json"
    bad = _candidate(status="BROKEN")
    scorers.write_text(json.dumps({"session": "s", "candidates": [bad]}), encoding="utf-8")
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps({"goals": []}), encoding="utf-8")
    # Act / Assert
    assert main(["--scorers", str(scorers), "--goals", str(goals), "--session", "s"]) == 1


def test_main_missing_session_exit_1(tmp_path: pathlib.Path) -> None:
    # Arrange：candidates 无 session 且未给 --session
    scorers_dir = tmp_path / "scorers"
    scorers_dir.mkdir()
    scorers = scorers_dir / "scorer_candidates.json"
    scorers.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps({"goals": []}), encoding="utf-8")
    # Act / Assert
    assert main(["--scorers", str(scorers), "--goals", str(goals)]) == 1


def test_index_events_bad_schema_exit_1(tmp_path: pathlib.Path) -> None:
    # Arrange：events_index 损坏（事件缺 clip）
    scorers_dir = tmp_path / "scorers"
    scorers_dir.mkdir()
    scorers = scorers_dir / "scorer_candidates.json"
    scorers.write_text(json.dumps({"session": "s", "candidates": [_candidate()]}), encoding="utf-8")
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps({"session": "s", "goals": [_goal()]}), encoding="utf-8")
    index = tmp_path / "events_index.json"
    index.write_text(
        json.dumps({"events": [{"src_file": "a.mp4", "anchor_t0": 4.0}]}), encoding="utf-8"
    )
    # Act / Assert
    rc = main(
        [
            "--scorers",
            str(scorers),
            "--goals",
            str(goals),
            "--session",
            "s",
            "--index",
            str(index),
        ]
    )
    assert rc == 1


def test_main_output_next_to_candidates(tmp_path: pathlib.Path) -> None:
    # Arrange
    scorers_dir = tmp_path / "deep" / "scorers"
    scorers_dir.mkdir(parents=True)
    scorers = scorers_dir / "scorer_candidates.json"
    scorers.write_text(json.dumps({"session": "s", "candidates": [_candidate()]}), encoding="utf-8")
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps({"session": "s", "goals": [_goal()]}), encoding="utf-8")
    # Act
    rc = main(["--scorers", str(scorers), "--goals", str(goals)])
    # Assert：输出默认取 candidates 里的 session，文件落在同目录
    assert rc == 0
    assert (scorers_dir / "scorer.html").is_file()


def test_match_clip_absolute_vs_relative_consistent(tmp_path: pathlib.Path) -> None:
    # Arrange：out_dir 与 index_dir 同根时相对路径不含 ".."
    base = tmp_path / "s"
    index_dir = base / "review"
    out_dir = base / "scorers"
    index_dir.mkdir(parents=True)
    out_dir.mkdir()
    # Act
    rel = match_clip([_event(anchor_t0=4.0)], "a.mp4", 4.1, str(index_dir), str(out_dir))
    # Assert
    assert os.sep not in rel or "/" in rel  # 统一正斜杠
    assert rel == "../review/clips/a_e1_wide.mp4"


# ---- --clusters 簇级确认（docs/scorer-cluster/spec.md） ----


def _cluster(
    cid: int = 1,
    keys: tuple[str, ...] = ("a.mp4#4.1",),
    rep_crops: tuple[str, ...] = ("a_t4.1.jpg",),
) -> dict:
    """构造一条合法簇记录（cluster_scorers 输出契约）。"""
    return {"cluster_id": cid, "keys": list(keys), "rep_crops": list(rep_crops)}


class TestValidateClusters:
    """scorer_clusters.json schema 校验（顶层缺 clusters/类型错 → SchemaError）。"""

    def test_valid_payload(self) -> None:
        # Arrange / Act
        clusters = _validate_clusters(
            {"version": "cluster-v1", "clusters": [_cluster()], "unclustered": []}, "c.json"
        )
        # Assert
        assert clusters == [_cluster()]

    def test_top_level_not_dict(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="顶层"):
            _validate_clusters([], "c.json")

    def test_missing_clusters(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="clusters"):
            _validate_clusters({"version": "cluster-v1"}, "c.json")

    def test_clusters_not_list(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="clusters"):
            _validate_clusters({"clusters": {}}, "c.json")

    def test_bad_cluster_field_types(self) -> None:
        # Arrange / Act / Assert：cluster_id bool、keys 非 str 列表、rep_crops 非列表
        with pytest.raises(SchemaError, match="cluster_id"):
            _validate_clusters({"clusters": [_cluster(cid=True)]}, "c.json")
        with pytest.raises(SchemaError, match="keys"):
            _validate_clusters({"clusters": [_cluster(keys=(1,))]}, "c.json")  # type: ignore[arg-type]
        with pytest.raises(SchemaError, match="rep_crops"):
            _validate_clusters({"clusters": [_cluster(rep_crops=("a.jpg", 2))]}, "c.json")  # type: ignore[arg-type]

    def test_duplicate_cluster_id(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="重复"):
            _validate_clusters({"clusters": [_cluster(cid=1), _cluster(cid=1)]}, "c.json")


class TestBuildClusterMap:
    """key → cluster_id 映射：越界 key WARNING 跳过，同 key 多簇取首个。"""

    def test_basic_mapping(self) -> None:
        # Arrange / Act
        m = build_cluster_map(
            [
                _cluster(cid=1, keys=("a.mp4#4.1", "b.mp4#2.0")),
                _cluster(cid=2, keys=("c.mp4#1.0",)),
            ],
            {"a.mp4#4.1", "b.mp4#2.0", "c.mp4#1.0"},
        )
        # Assert
        assert m == {"a.mp4#4.1": 1, "b.mp4#2.0": 1, "c.mp4#1.0": 2}

    def test_key_not_in_candidates_warns_and_skips(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange / Act
        with caplog.at_level(logging.WARNING):
            m = build_cluster_map([_cluster(keys=("a.mp4#4.1", "ghost.mp4#9.9"))], {"a.mp4#4.1"})
        # Assert：越界 key 跳过不炸，记 WARNING
        assert m == {"a.mp4#4.1": 1}
        assert any("ghost.mp4#9.9" in r.message for r in caplog.records)

    def test_key_in_two_clusters_first_wins(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange / Act
        with caplog.at_level(logging.WARNING):
            m = build_cluster_map(
                [_cluster(cid=1), _cluster(cid=2, keys=("a.mp4#4.1",))], {"a.mp4#4.1"}
            )
        # Assert
        assert m == {"a.mp4#4.1": 1}
        assert any("同时属于" in r.message for r in caplog.records)


class TestBuildEntriesClusterId:
    """cluster_id 注入：同 key 同簇、unclustered→None、无 --clusters 全 None。"""

    def test_cluster_id_injected(self) -> None:
        # Arrange / Act
        entries = build_entries(
            [_goal()],
            [_candidate()],
            None,
            "",
            "",
            cluster_map={"a.mp4#4.1": 3},
        )
        # Assert
        assert entries[0]["cluster_id"] == 3

    def test_unclustered_key_gets_none(self) -> None:
        # Arrange / Act：映射里没有该 key（unclustered）
        entries = build_entries([_goal()], [_candidate()], None, "", "", cluster_map={})
        # Assert
        assert entries[0]["cluster_id"] is None

    def test_no_cluster_map_all_none(self) -> None:
        # Arrange / Act：不传 --clusters（向后兼容）
        entries = build_entries([_goal()], [_candidate()], None, "", "")
        # Assert
        assert entries[0]["cluster_id"] is None


class TestBuildPageClusters:
    """簇区数据：keys 过滤到本页 confirmed 球，空簇剔除，rep_crops 透传。"""

    def test_filters_keys_not_in_entries(self) -> None:
        # Arrange：簇含本页球 + 其他批次球
        entries = build_entries([_goal()], [_candidate()], None, "", "")
        clusters = [_cluster(keys=("a.mp4#4.1", "other.mp4#1.0"))]
        # Act
        page = build_page_clusters(clusters, entries)
        # Assert
        assert page == [{"cluster_id": 1, "keys": ["a.mp4#4.1"], "rep_crops": ["a_t4.1.jpg"]}]

    def test_cluster_without_page_keys_dropped(self) -> None:
        # Arrange / Act
        page = build_page_clusters([_cluster(keys=("other.mp4#1.0",))], [])
        # Assert
        assert page == []


class TestBuildHtmlClusters:
    """簇区渲染：有簇出标记与 rep_crops 引用；无簇不渲染且行为同旧版。"""

    def test_cluster_section_with_rep_crops(self) -> None:
        # Arrange
        entries = build_entries(
            [_goal()], [_candidate()], None, "", "", cluster_map={"a.mp4#4.1": 1}
        )
        page_clusters = build_page_clusters([_cluster()], entries)
        # Act
        html = build_html(entries, [], "s", {}, {}, "对手", clusters=page_clusters)
        # Assert：新簇审阅区容器/代表图/簇级选人函数
        assert 'id="clusterView"' in html
        assert 'id="clusterCrops"' in html
        assert "function assignCluster(" in html
        assert "function clusterPrefill(" in html
        assert "a_t4.1.jpg" in html
        assert '"cluster_id": 1' in html

    def test_no_clusters_renders_empty(self) -> None:
        # Arrange / Act
        html = build_html([], [], "s", {}, {}, "对手")
        # Assert：无簇数据 → CLUSTERS 空数组，JS 整区隐藏
        assert "const CLUSTERS = [];" in html

    def test_generated_js_syntax_node_check(self, tmp_path: pathlib.Path) -> None:
        # Arrange：node 不在 PATH 则跳过（仿 7e9967c 防模板转义黑屏回归）
        node = shutil.which("node")
        if node is None:
            pytest.skip("node 不在 PATH")
        entries = build_entries(
            [_goal()],
            [_candidate()],
            None,
            "",
            "",
            [Player(tag="黑21", name="张三", team="对手")],
            cluster_map={"a.mp4#4.1": 1},
        )
        page_clusters = build_page_clusters([_cluster()], entries)
        html = build_html(
            entries,
            [Player(tag="黑21", name="张三", team="对手")],
            "s",
            {},
            {},
            "对手",
            clusters=page_clusters,
        )
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = tmp_path / "page.js"
        js_path.write_text(script, encoding="utf-8")
        # Act
        proc = subprocess.run(  # noqa: S603 node 路径来自 shutil.which，可信
            [node, "--check", str(js_path)], capture_output=True, text=True, check=False
        )
        # Assert
        assert proc.returncode == 0, proc.stderr


class TestBuildHtmlClusterMerge:
    """新合并子态：M 进入，←/→ 选目标，Enter 确认（spec §4.3-5）。"""

    def _html(self) -> str:
        entries = build_entries(
            [_goal()], [_candidate()], None, "", "", cluster_map={"a.mp4#4.1": 1}
        )
        page_clusters = build_page_clusters([_cluster()], entries)
        return build_html(entries, [], "s", {}, {}, "对手", clusters=page_clusters)

    def test_cluster_state_layer_present(self) -> None:
        html = self._html()
        assert 'CLSTATE_KEY = LSKEY + "_clusters"' in html
        assert "function saveClState(" in html
        assert "function groupIdOf(" in html
        assert "function computeGroups(" in html
        assert "function mergeInto(" in html
        assert "clState.clAssign" in html
        assert "done:" in html
        assert "clState.done" in html

    def test_merge_substate_present(self) -> None:
        html = self._html()
        assert "function startMerge(" in html
        assert "mergeState" in html
        assert "merge-target" in html
        assert "←/→ 选目标" in html

    def test_merge_substate_shields_other_controls(self) -> None:
        html = self._html()
        # 合并子态下清空球员按钮、清空视频、禁用顶栏按钮
        start = html.index("if (mergeState && mergeState.srcGid === g.gid)")
        body = html[start : html.index("function renderBallView(", start)]
        assert 'players.innerHTML = ""' in body
        assert 'setVideo(video, "")' in body
        assert "b.disabled = inSubstate" in html
        assert "inp.disabled = inSubstate" in html

    def test_eject_substate_shields_other_controls(self) -> None:
        html = self._html()
        # 剔除子态下清空球员按钮、清空视频、禁用顶栏按钮
        start = html.index("if (ejectState && ejectState.items.length)")
        body = html[start : html.index("function renderBallView(", start)]
        assert 'players.innerHTML = ""' in body
        assert 'setVideo(video, "")' in body
        assert "b.disabled = inSubstate" in html


class TestBuildHtmlDeleteCluster:
    """删簇：deleted 墓碑子键，组从簇区隐藏不动归属。"""

    def _html(self) -> str:
        entries = build_entries(
            [_goal()], [_candidate()], None, "", "", cluster_map={"a.mp4#4.1": 1}
        )
        page_clusters = build_page_clusters([_cluster()], entries)
        return build_html(entries, [], "s", {}, {}, "对手", clusters=page_clusters)

    def test_delete_cluster_present(self) -> None:
        html = self._html()
        assert "function deleteCluster(" in html
        assert "deleted:" in html
        assert "clState.deleted" in html
        assert "删除簇#" in html

    def test_deleted_subkey_loaded_and_saved(self) -> None:
        html = self._html()
        assert "deleted:" in html
        assert "clState.deleted" in html
        assert "localStorage.setItem(CLSTATE_KEY" in html
        assert "JSON.stringify(clState)" in html

    def test_delete_cluster_marks_done(self) -> None:
        html = self._html()
        start = html.index("function deleteCluster(")
        body = html[start : html.index("function startEject(", start)]
        assert "clState.done[String(gid)] = true" in body

    def test_delete_cluster_button_in_roster_table(self) -> None:
        html = self._html()
        assert 'id="delClusterBtn"' in html
        assert "delBtn.onclick" in html


class TestBuildHtmlRosterTable:
    """名单表格：内联改姓名、select 改队、对手队名覆盖（spec §4.1/4.3-10/4.3-11）。"""

    def _html(self) -> str:
        return build_html(
            [],
            [
                Player(tag="白22-王五", name="王五", team="主队"),
                Player(tag="黑21-张三", name="张三", team="对手"),
            ],
            "s",
            {},
            {},
            "对手",
        )

    def test_roster_table_present(self) -> None:
        html = self._html()
        assert 'id="rosterTable"' in html
        assert "function renderRosterTable(" in html
        assert 'class="name-input"' in html
        assert 'class="team-select"' in html

    def test_name_and_team_keys_present(self) -> None:
        html = self._html()
        assert '"_names"' in html
        assert '"_teamovr"' in html
        assert "function saveNames(" in html
        assert "function saveTeamOvr(" in html

    def test_opponent_name_override_present(self) -> None:
        html = self._html()
        assert 'id="oppInput"' in html
        assert '"_oppname"' in html
        assert "function saveOppName(" in html

    def test_select_focus_shields_shortcuts(self) -> None:
        html = self._html()
        assert "function isInputFocused(" in html
        assert 'el.tagName === "SELECT"' in html


class TestBuildHtmlBallMode:
    """模式二逐球收尾布局与过滤。"""

    def _html(self) -> str:
        return build_html([], [], "s", {}, {}, "蓝队")

    def test_ball_view_present(self) -> None:
        html = self._html()
        assert 'id="ballView"' in html
        assert 'id="ballInfo"' in html
        assert 'id="ballPlayers"' in html
        assert 'id="ballFilter"' in html
        assert "function renderBallView(" in html
        assert "function renderBallFilter(" in html

    def test_review_flex_and_hover_zoom_present(self) -> None:
        html = self._html()
        assert 'id="review"' in html
        assert "align-items: flex-start" in html
        assert "68vh" in html
        assert "#review #crop:hover" in html

    def test_no_clusters_fallback_to_ball_mode(self) -> None:
        # Arrange / Act：无 --clusters（CLUSTERS 空数组）时 JS 直接进模式二
        html = build_html([], [], "s", {}, {}, "蓝队")
        # Assert
        assert "const CLUSTERS = [];" in html
        assert "if (!CLUSTERS.length || !pendingGroups().length)" in html
        assert "模式二：逐球收尾" in html
        assert 'id="ballView"' in html


class TestMainClusters:
    """main 端到端 --clusters：同目录强校验、schema 损坏退出 1、簇区内联。"""

    def _write_inputs(
        self, tmp_path: pathlib.Path
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """造 scorers/goals/clusters 三个输入文件（clusters 与 scorers 同目录），返回路径。"""
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(
            json.dumps({"session": "s", "candidates": [_candidate()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps({"session": "s", "goals": [_goal()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        clusters = scorers_dir / "scorer_clusters.json"
        clusters.write_text(
            json.dumps(
                {
                    "version": "cluster-v1",
                    "model": "m",
                    "threshold": 0.25,
                    "clusters": [_cluster()],
                    "unclustered": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return scorers, goals, clusters

    def test_end_to_end_with_clusters(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals, clusters = self._write_inputs(tmp_path)
        # Act
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--clusters",
                str(clusters),
                "--players",
                "黑21=张三",
            ]
        )
        # Assert
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"cluster_id": 1' in html
        assert "a_t4.1.jpg" in html

    def test_bad_clusters_schema_exit_1(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals, clusters = self._write_inputs(tmp_path)
        clusters.write_text(json.dumps({"version": "cluster-v1"}), encoding="utf-8")
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--clusters", str(clusters)])
        # Assert：schema 损坏显式失败
        assert rc == 1

    def test_clusters_different_dir_rejected(self, tmp_path: pathlib.Path) -> None:
        # Arrange：clusters 与 scorers 不同目录（rep_crops 相对引用口径破坏）
        scorers, goals, _ = self._write_inputs(tmp_path)
        other = tmp_path / "other" / "scorer_clusters.json"
        other.parent.mkdir()
        other.write_text(json.dumps({"clusters": []}), encoding="utf-8")
        # Act / Assert：parser.error 显式拒绝（SystemExit 2）
        with pytest.raises(SystemExit):
            main(["--scorers", str(scorers), "--goals", str(goals), "--clusters", str(other)])


class TestBuildHtmlNoGoalTag:
    """不算进球标签：NOGOAL 哨兵，导出自动过滤。"""

    def _html(self) -> str:
        return build_html([], [], "s", {}, {}, "蓝队")

    def test_nogoal_present(self) -> None:
        html = self._html()
        assert 'const NOGOAL = "不算进球"' in html
        assert 'k.toLowerCase() === "n"' in html

    def test_export_strips_nogoal(self) -> None:
        html = self._html()
        assert "t !== NOGOAL" in html
        assert "已剔除不参与合成" in html


class TestAcceptAllPrefills:
    """P 键批量接受号码/照片预填（新设计函数名 clusterAcceptAllPrefills）。"""

    def test_function_present_and_guards(self) -> None:
        html = build_html([], [], "s", {}, {}, "对手")
        assert "function clusterAcceptAllPrefills" in html
        start = html.index("function clusterAcceptAllPrefills")
        body = html[start : html.index("alert(", start)]
        assert "if (!it.prefill_tag) continue;" in body
        assert "if (touched[it.key])" in body
        assert 'it.prefill_note === "ambiguous"' in body
        assert "marks[it.key] = it.prefill_tag;" in body
        assert "touched[it.key] = true" not in body

    def test_acceptall_splits_photo_count(self) -> None:
        html = build_html([], [], "s", {}, {}, "对手")
        start = html.index("function clusterAcceptAllPrefills")
        body = html[start : html.index("alert(", start)]
        assert 'it.prefill_note === "photo"' in body
        assert "nPhoto" in body
        # 计数文案在 alert 中
        assert "个预填" in html

    def test_acceptall_js_syntax_node_check(self, tmp_path: pathlib.Path) -> None:
        # node 不在 PATH 则跳过（沿用现有同款模式，防模板改动引入 JS 语法错）
        node = shutil.which("node")
        if node is None:
            pytest.skip("node 不在 PATH")
        html = build_html([], [], "s", {}, {}, "对手")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = tmp_path / "page.js"
        js_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 node 路径来自 shutil.which，可信
            [node, "--check", str(js_path)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr


# ---- --photo-matches 照片库预填（docs/photo-roster/spec.md T5） ----


def _photo_entry(number: str = "9", score: float = 0.61, margin: float = 0.1) -> MatchEntry:
    """构造一条照片命中记录（photo_match_scorers schema 校验产物口径）。"""
    return MatchEntry(number=number, score=score, margin=margin)


def _photo_payload(matches: dict) -> dict:
    """构造 photo_matches.json 载荷（photo_match_scorers 输出契约）。"""
    return {
        "version": MATCH_VERSION,
        "model": "ViT-B-32/laion2b_s34b_b79k",
        "threshold": 0.30,
        "margin": 0.02,
        "matches": matches,
    }


class TestResolvePhotoGuesses:
    """照片命中号码 → 名单 tag；名单缺号注入占位条目（主队<号>，team=主队）。"""

    def test_number_in_players_resolves_tag(self) -> None:
        # Arrange
        players = [Player(tag="白7-王五", name="王五", team="主队")]
        # Act
        guesses, extra = resolve_photo_guesses({"a.mp4#4.1": _photo_entry("7")}, players)
        # Assert：号码在名单唯一命中，直接解析到该球员 tag，无占位
        assert guesses["a.mp4#4.1"] == PhotoGuess(number="7", score=0.61, tag="白7-王五")
        assert extra == []

    def test_missing_number_placeholder_injected(self) -> None:
        # Arrange：名单里没有 9 号
        players = [Player(tag="白7-王五", name="王五", team="主队")]
        # Act
        guesses, extra = resolve_photo_guesses({"a.mp4#4.1": _photo_entry("9")}, players)
        # Assert：占位 tag=主队9、name 空、team=主队（不靠前缀推队）
        assert guesses["a.mp4#4.1"].tag == "主队9"
        assert extra == [Player(tag="主队9", name="", team="主队")]

    def test_same_missing_number_single_placeholder(self) -> None:
        # Arrange / Act：两球命中同一缺号号码
        guesses, extra = resolve_photo_guesses(
            {"a.mp4#4.1": _photo_entry("9"), "b.mp4#2.0": _photo_entry("9")}, []
        )
        # Assert：占位条目只注入一份，两球都指向它
        assert len(extra) == 1
        assert {g.tag for g in guesses.values()} == {"主队9"}

    def test_ambiguous_number_in_players_no_guess(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange：名单里同号两人（白7/红7）→ 交人裁判不预填
        players = [
            Player(tag="白7-王五", name="王五", team="主队"),
            Player(tag="红7-吴九", name="", team="主队"),
        ]
        # Act
        with caplog.at_level(logging.WARNING):
            guesses, extra = resolve_photo_guesses({"a.mp4#4.1": _photo_entry("7")}, players)
        # Assert
        assert guesses == {}
        assert extra == []
        assert any("同号多人" in r.message for r in caplog.records)


class TestPhotoPrefillPriority:
    """build_entries 预填优先级（写死）：读号命中 > 照片命中 > 印名匹配 > 空白。"""

    @staticmethod
    def _players() -> list[Player]:
        return [
            Player(tag="白22-王五", name="王五", team="主队"),
            Player(tag="白7-周八", name="周八", team="主队"),
        ]

    @staticmethod
    def _photo_guess(number: str = "7", tag: str = "白7-周八") -> dict:
        return {"a.mp4#4.1": PhotoGuess(number=number, score=0.5, tag=tag)}

    def test_photo_hit_prefills_without_number(self) -> None:
        # Arrange：无读号（number_guess 为 None），照片命中 7 号
        cand = _candidate()
        cand["number_guess"] = None
        # Act
        entries = build_entries(
            [_goal()],
            [cand],
            None,
            "",
            "",
            self._players(),
            photo_guesses=self._photo_guess(),
        )
        # Assert
        assert entries[0]["prefill_tag"] == "白7-周八"
        assert entries[0]["prefill_note"] == "photo"
        assert entries[0]["photo_guess"] == {"number": "7", "score": 0.5, "tag": "白7-周八"}

    def test_number_beats_photo_on_conflict(self) -> None:
        # Arrange：读号 22 与照片 7 冲突
        cand = _candidate()
        cand["number_guess"] = {
            "number": "22",
            "color": "白",
            "name_text": None,
            "confidence": "high",
        }
        # Act
        entries = build_entries(
            [_goal()],
            [cand],
            None,
            "",
            "",
            self._players(),
            photo_guesses=self._photo_guess(),
        )
        # Assert：预填读号结果；照片候选保留在 photo_guess 供角标切换（不静默覆盖）
        assert entries[0]["prefill_tag"] == "白22-王五"
        assert entries[0]["prefill_note"] == ""
        assert entries[0]["photo_guess"]["tag"] == "白7-周八"

    def test_photo_beats_name(self) -> None:
        # Arrange：读号空、印名命中"王五"，照片命中 7 号 → 照片优先
        cand = _candidate()
        cand["number_guess"] = {
            "number": None,
            "color": None,
            "name_text": "王五",
            "confidence": "low",
        }
        # Act
        entries = build_entries(
            [_goal()],
            [cand],
            None,
            "",
            "",
            self._players(),
            photo_guesses=self._photo_guess(),
        )
        # Assert
        assert entries[0]["prefill_tag"] == "白7-周八"
        assert entries[0]["prefill_note"] == "photo"

    def test_name_fallback_without_photo_unchanged(self) -> None:
        # Arrange：无照片数据时印名兜底维持现状（回归锁定）
        cand = _candidate()
        cand["number_guess"] = {
            "number": None,
            "color": None,
            "name_text": "王五",
            "confidence": "low",
        }
        # Act
        entries = build_entries([_goal()], [cand], None, "", "", self._players())
        # Assert
        assert entries[0]["prefill_tag"] == "白22-王五"
        assert entries[0]["prefill_note"] == ""
        assert entries[0]["photo_guess"] is None

    def test_number_ambiguous_not_overridden_by_photo(self) -> None:
        # Arrange：读号同号歧义 + 照片命中 → 维持歧义不预填，照片候选仍随条目上页
        players = [
            Player(tag="白22-王五", name="王五", team="主队"),
            Player(tag="白22-张三", name="张三", team="主队"),
        ]
        cand = _candidate()
        cand["number_guess"] = {
            "number": "22",
            "color": "白",
            "name_text": None,
            "confidence": "high",
        }
        # Act
        entries = build_entries(
            [_goal()],
            [cand],
            None,
            "",
            "",
            players,
            photo_guesses=self._photo_guess(tag="主队7"),
        )
        # Assert
        assert entries[0]["prefill_tag"] == ""
        assert entries[0]["prefill_note"] == "ambiguous"
        assert entries[0]["photo_guess"]["tag"] == "主队7"

    def test_no_photo_param_entries_photo_guess_none(self) -> None:
        # Arrange / Act：不传 photo_guesses（无 --photo-matches 口径）
        entries = build_entries([_goal()], [_candidate()], None, "", "")
        # Assert：条目带空 photo_guess，其余字段行为不变
        assert entries[0]["photo_guess"] is None
        assert entries[0]["prefill_tag"] == ""
        assert entries[0]["team_guess"] == "黑"

    def test_photo_guess_key_not_in_entries_ignored(self) -> None:
        # Arrange：photo_guesses 引用其他批次 key → 本页不受影响
        entries = build_entries(
            [_goal()],
            [_candidate()],
            None,
            "",
            "",
            photo_guesses={"other.mp4#1.0": PhotoGuess(number="7", score=0.5, tag="白7")},
        )
        # Assert
        assert entries[0]["photo_guess"] is None

    def test_photo_prefill_satisfies_acceptall_condition(self) -> None:
        # Arrange：照片预填条目——锁定其满足「接受全部预填」守卫组合（review MEDIUM-1）
        cand = _candidate()
        cand["number_guess"] = None
        # Act
        entries = build_entries(
            [_goal()],
            [cand],
            None,
            "",
            "",
            self._players(),
            photo_guesses=self._photo_guess(),
        )
        # Assert：prefill_tag 非空（守卫 if (!it.prefill_tag) continue 通过）
        # 且 note="photo" 供批量接受时拆分计数
        assert entries[0]["prefill_tag"] == "白7-周八"
        assert entries[0]["prefill_note"] == "photo"


class TestPhotoBadgeHtml:
    """照片预填：新设计在 ballInfo 展示预填来源，不再出角标按钮。"""

    def test_photo_prefill_displayed_in_ball_info(self) -> None:
        html = build_html([], [], "s", {}, {}, "对手")
        assert "预填:" in html
        assert "photo" in html

    def test_photo_js_syntax_node_check(self, tmp_path: pathlib.Path) -> None:
        node = shutil.which("node")
        if node is None:
            pytest.skip("node 不在 PATH")
        entries = build_entries(
            [_goal()],
            [_candidate()],
            None,
            "",
            "",
            [Player(tag="白7-周八", name="周八", team="主队")],
            photo_guesses={"a.mp4#4.1": PhotoGuess(number="7", score=0.5, tag="白7-周八")},
        )
        html = build_html(
            entries,
            [Player(tag="白7-周八", name="周八", team="主队")],
            "s",
            {},
            {},
            "对手",
        )
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = tmp_path / "page.js"
        js_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 node 路径来自 shutil.which，可信
            [node, "--check", str(js_path)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr


class TestPhotoMatchesCli:
    """main 端到端 --photo-matches：同目录校验、坏 schema 退出 1、无参零变化。"""

    def _write_inputs(
        self, tmp_path: pathlib.Path, matches: dict
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """造 scorers/goals/photo_matches 三个输入文件（同目录），返回路径。"""
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(
            json.dumps({"session": "s", "candidates": [_candidate()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps({"session": "s", "goals": [_goal()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        pm = scorers_dir / "photo_matches.json"
        pm.write_text(json.dumps(_photo_payload(matches)), encoding="utf-8")
        return scorers, goals, pm

    def test_end_to_end_photo_matches(self, tmp_path: pathlib.Path) -> None:
        # Arrange：照片命中 9 号，名单无 9 号 → 占位条目注入
        key = format_key("a.mp4", 4.1)
        scorers, goals, pm = self._write_inputs(
            tmp_path, {key: {"number": "9", "score": 0.61, "margin": 0.1}}
        )
        # Act
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--photo-matches",
                str(pm),
            ]
        )
        # Assert：占位条目随 players 注入页面（不靠前缀推队），条目带 photo_guess
        # 向导仓默认 team_config.team_name="主队"，占位 tag=主队9
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"tag": "主队9"' in html
        assert '"photo_guess": {' in html
        assert '"number": "9"' in html
        assert "Infinity" not in html

    def test_infinite_margin_not_injected(self, tmp_path: pathlib.Path) -> None:
        # Arrange：单号码库命中 margin=+inf，Python json 落盘为 Infinity
        key = format_key("a.mp4", 4.1)
        scorers, goals, pm = self._write_inputs(
            tmp_path, {key: {"number": "9", "score": 0.61, "margin": float("inf")}}
        )
        # Act：读端可解析 Infinity，但 margin 不进 JS（页面 JSON.parse 无法解析）
        rc = main(
            [
                "--scorers",
                str(scorers),
                "--goals",
                str(goals),
                "--photo-matches",
                str(pm),
            ]
        )
        # Assert
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert "Infinity" not in html

    def test_photo_matches_different_dir_rejected(self, tmp_path: pathlib.Path) -> None:
        # Arrange：photo_matches 与 scorers 不同目录（与 --clusters 校验同口径）
        scorers, goals, _ = self._write_inputs(tmp_path, {})
        other = tmp_path / "other" / "photo_matches.json"
        other.parent.mkdir()
        other.write_text(json.dumps(_photo_payload({})), encoding="utf-8")
        # Act / Assert：parser.error 显式拒绝（SystemExit 2）
        with pytest.raises(SystemExit):
            main(["--scorers", str(scorers), "--goals", str(goals), "--photo-matches", str(other)])

    def test_bad_photo_matches_schema_exit_1(self, tmp_path: pathlib.Path) -> None:
        # Arrange：version 不符 → SchemaError 显式失败
        scorers, goals, pm = self._write_inputs(tmp_path, {})
        pm.write_text(json.dumps({"version": "bogus", "matches": {}}), encoding="utf-8")
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--photo-matches", str(pm)])
        # Assert
        assert rc == 1

    def test_no_photo_matches_zero_change(self, tmp_path: pathlib.Path) -> None:
        # Arrange：同目录有 photo_matches.json 但不传参（只认显式 --photo-matches）
        key = format_key("a.mp4", 4.1)
        scorers, goals, _ = self._write_inputs(
            tmp_path, {key: {"number": "9", "score": 0.61, "margin": 0.1}}
        )
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals)])
        # Assert：零预填零占位（兼容性承诺锁定）
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"photo_guess": null' in html
        assert "主队9" not in html


# ---- --track-links 轨迹传播预填（docs/scorer-propagate/spec.md §页面） ----


def _track_payload(per_file: dict) -> dict:
    """构造合法 track_links.json 载荷（propagate_scorers track-v1 契约）。"""
    return {"version": "track-v1", "per_file": per_file}


def _file_tracks(tracks: list[dict], unlinked: list[str] | None = None) -> dict:
    """构造单文件段：tracks=[{track_id, keys, mixed, span}], unlinked=[keys]。"""
    return {"tracks": tracks, "unlinked": unlinked or []}


def _track(track_id: int, keys: list[str], mixed: bool = False) -> dict:
    """构造单条轨迹记录。"""
    return {"track_id": track_id, "keys": keys, "mixed": mixed, "span": [0, 100]}


class TestValidateTrackLinks:
    """track_links.json schema 校验（rules.md §0.2：结构坏显式失败）。"""

    def test_valid_payload(self) -> None:
        # Arrange / Act
        per_file = _validate_track_links(
            _track_payload({"a": _file_tracks([_track(1, ["a.mp4#4.1"])])}), "t.json"
        )
        # Assert
        assert per_file["a"]["tracks"] == [_track(1, ["a.mp4#4.1"])]

    def test_top_level_not_dict(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="顶层"):
            _validate_track_links([], "t.json")

    def test_bad_version(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="version"):
            _validate_track_links({"version": "bogus", "per_file": {}}, "t.json")

    def test_missing_per_file(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError, match="per_file"):
            _validate_track_links({"version": "track-v1"}, "t.json")

    def test_track_missing_keys(self) -> None:
        # Arrange：轨迹缺 keys 字段
        bad = _track_payload({"a": _file_tracks([{"track_id": 1, "mixed": False, "span": [0, 1]}])})
        # Act / Assert
        with pytest.raises(SchemaError, match="keys"):
            _validate_track_links(bad, "t.json")

    def test_track_bad_span(self) -> None:
        # Arrange：span 不是二元 int 列表
        bad = _track_payload(
            {"a": _file_tracks([{"track_id": 1, "keys": [], "mixed": False, "span": [0]}])}
        )
        # Act / Assert
        with pytest.raises(SchemaError, match="span"):
            _validate_track_links(bad, "t.json")

    def test_bad_unlinked(self) -> None:
        # Arrange：unlinked 非 str 列表
        bad = _track_payload({"a": {"tracks": [], "unlinked": [1]}})
        # Act / Assert
        with pytest.raises(SchemaError, match="unlinked"):
            _validate_track_links(bad, "t.json")


class TestBuildTrackMap:
    """key 反查 track_id：本页 key 映射、跨批 key 跳过、重复 key 取首个。"""

    def test_page_keys_mapped(self) -> None:
        # Arrange / Act
        m = build_track_map(
            {"a": _file_tracks([_track(3, ["a.mp4#4.1", "a.mp4#6.0"])])},
            {"a.mp4#4.1", "a.mp4#6.0"},
        )
        # Assert
        assert m == {"a.mp4#4.1": 3, "a.mp4#6.0": 3}

    def test_cross_batch_keys_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange：轨迹含其他批次的 key（常态，不炸）
        per_file = {"a": _file_tracks([_track(1, ["a.mp4#4.1", "other.mp4#9.9"])])}
        # Act
        with caplog.at_level(logging.INFO):
            m = build_track_map(per_file, {"a.mp4#4.1"})
        # Assert：跨批 key 跳过记 INFO，本页 key 照常映射
        assert m == {"a.mp4#4.1": 1}
        assert any("不在本页" in r.message for r in caplog.records)

    def test_duplicate_key_first_wins(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange：同一 key 挂两条轨迹（契约本应互斥，容忍不炸取首个）
        per_file = {"a": _file_tracks([_track(1, ["a.mp4#4.1"]), _track(2, ["a.mp4#4.1"])])}
        # Act
        with caplog.at_level(logging.WARNING):
            m = build_track_map(per_file, {"a.mp4#4.1"})
        # Assert
        assert m == {"a.mp4#4.1": 1}
        assert any("取前者" in r.message for r in caplog.records)

    def test_unlinked_not_mapped(self) -> None:
        # Arrange / Act：unlinked 球不进映射（页面 track_id=None）
        m = build_track_map({"a": _file_tracks([], unlinked=["a.mp4#4.1"])}, {"a.mp4#4.1"})
        # Assert
        assert m == {}


class TestBuildEntriesTrackId:
    """build_entries track 注入：映射命中出 track_id；未命中/无映射 → None。"""

    def test_track_id_injected(self) -> None:
        # Arrange / Act
        entries = build_entries(
            [_goal(), _goal("a.mp4", 6.0)],
            [_candidate(), _candidate("a.mp4", 6.0, crop="a_t6.jpg")],
            None,
            "",
            "",
            track_map={"a.mp4#4.1": 2, "a.mp4#6.0": 2},
        )
        # Assert
        assert [e["track_id"] for e in entries] == [2, 2]

    def test_unmapped_goal_track_id_none(self) -> None:
        # Arrange / Act：部分球归不上轨迹
        entries = build_entries(
            [_goal()], [_candidate()], None, "", "", track_map={"other.mp4#1.0": 5}
        )
        # Assert
        assert entries[0]["track_id"] is None

    def test_no_track_map_zero_change(self) -> None:
        # Arrange / Act：不传 track_map（无 --track-links 兼容口径）
        entries = build_entries([_goal()], [_candidate()], None, "", "")
        # Assert
        assert entries[0]["track_id"] is None


class TestTrackLinksCli:
    """--track-links CLI 层：同目录校验、坏 schema 退出 1、端到端生成页面。"""

    def _write_inputs(
        self, tmp_path: pathlib.Path, per_file: dict
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        """造 scorers/goals/track_links 三个输入文件，返回路径。"""
        scorers_dir = tmp_path / "scorers"
        scorers_dir.mkdir()
        scorers = scorers_dir / "scorer_candidates.json"
        scorers.write_text(
            json.dumps({"session": "s", "candidates": [_candidate()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        goals = tmp_path / "goals.json"
        goals.write_text(
            json.dumps({"session": "s", "goals": [_goal()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        track_links = scorers_dir / "track_links.json"
        track_links.write_text(
            json.dumps(_track_payload(per_file), ensure_ascii=False), encoding="utf-8"
        )
        return scorers, goals, track_links

    def test_end_to_end_with_track_links(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        scorers, goals, tl = self._write_inputs(
            tmp_path, {"a": _file_tracks([_track(7, ["a.mp4#4.1"])])}
        )
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--track-links", str(tl)])
        # Assert
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"track_id": 7' in html

    def test_cross_batch_keys_tolerated(self, tmp_path: pathlib.Path) -> None:
        # Arrange：track_links 含其他批次的 key（常态，不炸不退出）
        scorers, goals, tl = self._write_inputs(
            tmp_path,
            {"a": _file_tracks([_track(1, ["a.mp4#4.1", "other.mp4#9.9"])], ["x.mp4#1.0"])},
        )
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--track-links", str(tl)])
        # Assert
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"track_id": 1' in html
        assert "other.mp4#9.9" not in html

    def test_track_links_different_dir_rejected(self, tmp_path: pathlib.Path) -> None:
        # Arrange：track_links 与 scorers 不同目录（与 --clusters 校验同口径）
        scorers, goals, _ = self._write_inputs(tmp_path, {})
        other = tmp_path / "other" / "track_links.json"
        other.parent.mkdir()
        other.write_text(json.dumps(_track_payload({})), encoding="utf-8")
        # Act / Assert：parser.error 显式拒绝（SystemExit 2）
        with pytest.raises(SystemExit):
            main(["--scorers", str(scorers), "--goals", str(goals), "--track-links", str(other)])

    def test_bad_track_links_schema_exit_1(self, tmp_path: pathlib.Path) -> None:
        # Arrange：缺 per_file → SchemaError 显式失败
        scorers, goals, tl = self._write_inputs(tmp_path, {})
        tl.write_text(json.dumps({"version": "track-v1"}), encoding="utf-8")
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals), "--track-links", str(tl)])
        # Assert
        assert rc == 1

    def test_no_track_links_zero_change(self, tmp_path: pathlib.Path) -> None:
        # Arrange：同目录有 track_links.json 但不传参（只认显式 --track-links）
        scorers, goals, _ = self._write_inputs(
            tmp_path, {"a": _file_tracks([_track(7, ["a.mp4#4.1"])])}
        )
        # Act
        rc = main(["--scorers", str(scorers), "--goals", str(goals)])
        # Assert：track_id 全 null（兼容性承诺锁定）
        assert rc == 0
        html = (scorers.parent / "scorer.html").read_text(encoding="utf-8")
        assert '"track_id": null' in html


class TestTrackPropagatePageJs:
    """页面 JS 契约（spec §页面写死）：轨迹#N、传播触发条件、徽标判定式、
    acceptAll/E 键隔离、provenance 独立键。"""

    def _html(self) -> str:
        entries = build_entries([_goal()], [_candidate()], None, "", "", track_map={"a.mp4#4.1": 3})
        return build_html(entries, [], "s", {}, {}, "对手")

    def test_track_label_and_badge_rendered(self) -> None:
        # Arrange / Act
        html = self._html()
        # Assert：条目显示"轨迹#N"；徽标判定式 spec 写死（marks 有值 + provenance + 未手改）
        assert '"track_id": 3' in html
        assert '" | 轨迹#" + it.track_id' in html
        assert "同轨迹预填" in html
        assert "marks[it.key] && propagateAssign[it.key] && !touched[it.key]" in html

    def test_provenance_key_pattern(self) -> None:
        # Arrange / Act
        html = self._html()
        # Assert：provenance 独立 localStorage 键（沿用 touched 键管理模式）
        assert 'const PROPKEY = LSKEY + "_propagate";' in html
        assert "localStorage.setItem(PROPKEY, JSON.stringify(propagateAssign))" in html

    def test_propagate_from_guards(self) -> None:
        html = self._html()
        start = html.index("function propagateFrom")
        body = html[start : html.index("function exportRoster", start)]
        assert "if (tag === NOGOAL) return;" in body
        assert "it.file !== src.file || it.track_id !== src.track_id" in body
        assert "marks[it.key] || it.prefill_tag || touched[it.key]" in body
        assert "marks[it.key] = tag;" in body
        assert "propagateAssign[it.key] = true;" in body

    def test_assign_triggers_propagation(self) -> None:
        html = self._html()
        start = html.index("function assignBall(")
        body = html[start : html.index("function skipBall(", start)]
        assert "propagateFrom(it.key, tag);" in body

    def test_cluster_assign_triggers_propagation(self) -> None:
        html = self._html()
        start = html.index("function assignCluster(")
        body = html[start : html.index("function clusterAdoptPrefill(", start)]
        assert "propagateFrom(k, tag);" in body

    def test_acceptall_isolated_from_propagation(self) -> None:
        html = self._html()
        start = html.index("function clusterAcceptAllPrefills")
        body = html[start : html.index("alert(", start)]
        assert "propagateAssign" not in body
        assert "if (!it.prefill_tag) continue;" in body

    def test_export_unchanged_uses_marks(self) -> None:
        html = self._html()
        start = html.index("function exportRoster")
        body = html[start : html.index("function setVideo(", start)]
        assert "Object.entries(marks)" in body
        assert "propagateAssign" not in body

    def test_track_js_syntax_node_check(self, tmp_path: pathlib.Path) -> None:
        # node 不在 PATH 则跳过（沿用现有同款模式，防模板改动引入 JS 语法错——7e9967c 前科）
        node = shutil.which("node")
        if node is None:
            pytest.skip("node 不在 PATH")
        html = self._html()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = tmp_path / "page.js"
        js_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 node 路径来自 shutil.which，可信
            [node, "--check", str(js_path)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
