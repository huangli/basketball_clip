"""roster.py 契约模块单元测试（spec: docs/scorer/spec.md §roster schema）。

覆盖：schema 校验（缺 players / tag 重复 / team 空串或非 str / 键格式错 → SchemaError）、
format_key 双端一致性（4.1234→"4.1"）、fid_of 去扩展名、resolve_scorer 命中 tag 或 name。
"""

from __future__ import annotations

import pytest

from errors import SchemaError
from roster import Player, Roster, fid_of, format_key, resolve_scorer, validate_roster

_PATH = "work/20260722/roster.json"


def _player(tag: str = "黑21", name: str = "", team: str = "对手") -> dict[str, str]:
    """构造一条合法 player 记录。"""
    return {"tag": tag, "name": name, "team": team}


def _roster_data(**over: object) -> dict[str, object]:
    """构造一份合法 roster 数据，按字段覆盖。"""
    base: dict[str, object] = {
        "session": "20260722",
        "confirmed": True,
        "players": [_player(), _player(tag="白22", team="主队")],
        "assignments": {"a.mp4#4.1": "黑21"},
    }
    base.update(over)
    return base


class TestFormatKey:
    """format_key / fid_of 格式化契约。"""

    def test_format_key_one_decimal(self) -> None:
        # Arrange / Act / Assert：锚点统一压到一位小数（双端共用，禁止裸拼）
        assert format_key("a.mp4", 4.1234) == "a.mp4#4.1"
        assert format_key("a.mp4", 2.0) == "a.mp4#2.0"
        assert format_key("a.mp4", 0.0) == "a.mp4#0.0"

    def test_fid_of_strips_extension(self) -> None:
        # Arrange / Act / Assert
        assert fid_of("sample_20260722_190104_0_1784829884250_video.mp4") == (
            "sample_20260722_190104_0_1784829884250_video"
        )
        assert fid_of("a.MP4") == "a"

    def test_format_key_roundtrip_with_fid(self) -> None:
        # Arrange：键内 file 保留全名，fid 只是目录映射
        file = "x_video.mp4"
        # Act
        key = format_key(file, 16.49)
        # Assert
        assert key == "x_video.mp4#16.5"
        assert fid_of(key.split("#")[0]) == "x_video"


class TestValidateRoster:
    """schema 校验：合法通过，各类损坏抛 SchemaError。"""

    def test_valid_passes(self) -> None:
        # Arrange
        data = _roster_data()
        # Act
        roster = validate_roster(data, _PATH)
        # Assert
        assert roster.session == "20260722"
        assert roster.confirmed is True
        assert len(roster.players) == 2
        assert roster.assignments == {"a.mp4#4.1": "黑21"}

    def test_top_level_not_dict(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            validate_roster([], _PATH)

    def test_missing_players(self) -> None:
        # Arrange
        data = _roster_data()
        del data["players"]
        # Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(data, _PATH)

    def test_players_not_list(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(_roster_data(players={"tag": "黑21"}), _PATH)

    def test_duplicate_tag(self) -> None:
        # Arrange
        data = _roster_data(players=[_player(), _player()])
        # Act / Assert
        with pytest.raises(SchemaError, match="tag 重复"):
            validate_roster(data, _PATH)

    def test_empty_tag(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(_roster_data(players=[_player(tag="")]), _PATH)

    def test_arbitrary_team_valid(self) -> None:
        # Arrange：team 放宽为任意非空 str——对手队名随场次 ID 后缀
        # （docs/session-opponent-name/spec.md，如 20260805_对手队 → 对手队）
        data = _roster_data(players=[_player(tag="红7", team="对手队")])
        # Act
        roster = validate_roster(data, _PATH)
        # Assert
        assert roster.players[0].team == "对手队"

    def test_empty_team_rejected(self) -> None:
        # Arrange / Act / Assert：空串仍拒
        with pytest.raises(SchemaError, match="team 必须是非空字符串"):
            validate_roster(_roster_data(players=[_player(team="")]), _PATH)

    def test_whitespace_team_rejected(self) -> None:
        # Arrange / Act / Assert：纯空白视为空，仍拒
        with pytest.raises(SchemaError, match="team 必须是非空字符串"):
            validate_roster(_roster_data(players=[_player(team="  ")]), _PATH)

    def test_team_not_str_rejected(self) -> None:
        # Arrange / Act / Assert：非 str 仍拒
        with pytest.raises(SchemaError, match="team 必须是非空字符串"):
            validate_roster(_roster_data(players=[_player(team=123)]), _PATH)  # type: ignore[arg-type]

    def test_casual_team_valid(self) -> None:
        # Arrange：便服是合法 team 值（spec Open Q3）
        data = _roster_data(players=[_player(tag="灰T恤-A", team="便服")])
        # Act
        roster = validate_roster(data, _PATH)
        # Assert
        assert roster.players[0].team == "便服"

    @pytest.mark.parametrize(
        "bad_key",
        [
            "a.mp4",  # 无 # 时间部分
            "a.mp4#abc",  # 时间非数字
            "a.mp4#4.12",  # 未按 :.1f 格式化
            "a.mp4#4",  # 缺小数位
            "#4.1",  # file 为空
        ],
    )
    def test_bad_assignment_key(self, bad_key: str) -> None:
        # Arrange
        data = _roster_data(assignments={bad_key: "黑21"})
        # Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(data, _PATH)

    def test_assignment_value_not_str(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(_roster_data(assignments={"a.mp4#4.1": 3}), _PATH)

    def test_confirmed_not_bool(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(SchemaError):
            validate_roster(_roster_data(confirmed="yes"), _PATH)


class TestResolveScorer:
    """resolve_scorer：tag 或 name 任一命中。"""

    def _roster(self) -> Roster:
        return Roster(
            session="s",
            confirmed=True,
            players=(
                Player(tag="黑21", name="测试员甲", team="对手"),
                Player(tag="灰T恤-A", name="", team="便服"),
            ),
            assignments={},
        )

    def test_hit_by_tag(self) -> None:
        # Arrange / Act
        hit = resolve_scorer(self._roster(), "黑21")
        # Assert
        assert hit is not None
        assert hit.tag == "黑21"

    def test_hit_by_name(self) -> None:
        # Arrange / Act
        hit = resolve_scorer(self._roster(), "测试员甲")
        # Assert
        assert hit is not None
        assert hit.tag == "黑21"

    def test_miss_returns_none(self) -> None:
        # Arrange / Act / Assert
        assert resolve_scorer(self._roster(), "白7") is None

    def test_empty_name_not_matched(self) -> None:
        # Arrange：name 为空串的 player 不应被空查询命中
        assert resolve_scorer(self._roster(), "") is None
