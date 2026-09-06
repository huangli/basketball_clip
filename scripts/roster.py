"""roster.json 契约模块（spec: docs/scorer/spec.md §roster schema；
team 取值口径以 docs/session-opponent-name/spec.md 为准）。

schema 校验、assignments 键格式化、fid 映射、scorer 解析。

输入：roster.json 读取端的原始 JSON（dict）、goals.json 的 file/anchor_time。
输出：校验后的 Roster 结构体、assignments 键、fid。
依赖：scripts/errors.py 的 SchemaError；scripts/geom.py 无依赖关系。
典型调用：
    roster = validate_roster(read_json(path, what="roster.json"), str(path))
    key = format_key(goal["file"], goal["anchor_time"])
    player = resolve_scorer(roster, "黑21")

契约要点（写读双方必须共用本模块，禁止各自裸拼，spec M3）：
- assignments 键 = ``f"{file}#{t:.1f}"``（file 保留全名含 .mp4）；
- fid = 文件主名（去扩展名），与 extract_frames / mot_candidates 的目录命名一致；
- 合法 team 值：任意非空 str（"便服"有特殊语义；我方队名由 team_config.json
  会话级注入，对手队名随场次 ID 后缀，见 docs/session-opponent-name/spec.md）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errors import SchemaError
from team_config import DEFAULT_OPPONENT, load_team_config

# 伪球员：用户一键把进球标为“对手”，不识别具体是谁（docs/opponent-filter/spec.md）
OPPONENT_TAG: str = "对手"


# assignments 键格式：<file>#<t:.1f>（file 可含任意字符除 # 结尾的时间部分，t 必须恰为一位小数）
_KEY_PATTERN: re.Pattern[str] = re.compile(r"^(?P<file>.+)#(?P<t>-?\d+(?:\.\d+)?)$")


def is_opponent_tag(tag: str) -> bool:
    """判断标签是否为伪球员"对手"。"""
    return tag == OPPONENT_TAG


def opponent_of(session: str) -> str:
    """对手队名 = 场次 ID 第一个 ``_`` 后的后缀。

    无后缀 / 后缀空白 → 回退 ``DEFAULT_OPPONENT``（通用默认"对手"）。
    与 gen_scorer_page 历史口径保持一致，集中于此模块供调用方复用。

    Args:
        session: 场次 ID，如 ``20260801_湖人``。

    Returns:
        对手队名（黑/蓝球衣标签的 team 值）。
    """
    parts = session.strip().split("_", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return DEFAULT_OPPONENT


def opponent_team_name(session_dir: Path) -> str:
    """取当前场次有效对手队名：team_config.opponent 优先，缺省按场次 ID 后缀派生。

    Args:
        session_dir: work/<场次>/ 目录。

    Returns:
        非空对手队名字符串。
    """
    cfg = load_team_config(session_dir)
    return cfg.opponent or opponent_of(session_dir.name)


@dataclass(frozen=True, slots=True)
class Player:
    """一名球员的标签归属（name 可为空串，表示仅有标签未给称呼）。"""

    tag: str
    name: str
    team: str


@dataclass(frozen=True, slots=True)
class Roster:
    """校验后的 roster。

    不变式：players 的 tag 全局唯一；assignments 的键全部满足 format_key 格式。
    assignments 用 dict 而非 tuple（保留 JSON 原序，键查询 O(1)）。
    """

    session: str
    confirmed: bool
    players: tuple[Player, ...]
    assignments: dict[str, str]


def format_key(file: str, t: float) -> str:
    """生成 assignments 键，写读两端共用的唯一格式化入口（spec M3）。

    Args:
        file: 视频文件名（保留全名含扩展名，供人工可读）。
        t: 进球锚点时间（秒）。

    Returns:
        ``f"{file}#{t:.1f}"``，如 ``a.mp4#4.1``。
    """
    return f"{file}#{t:.1f}"


def fid_of(file: str) -> str:
    """从视频文件名取 fid（去扩展名的主名），与 work/frames|detect 目录命名一致。

    Args:
        file: 视频文件名（可含目录前缀；goals.json 中通常只有 basename）。

    Returns:
        文件主名，如 ``比赛A_video.mp4`` → ``比赛A_video``。
    """
    return Path(file).stem


def player_from_dict(
    raw: Any,  # noqa: ANN401 JSON 待校验
    path: str,
    idx: int,
    seen_tags: set[str],
) -> Player:
    """校验单个 player 记录，返回结构体。

    roster.json 与 --players-file 名单文件共用的唯一校验入口（spec:
    docs/scorer-reid/spec.md §数据契约；两处的 players 记录同构）。

    Args:
        raw: 待校验的原始 JSON 记录。
        path: 文件路径（仅用于错误信息）。
        idx: 记录在 players 数组中的下标（仅用于错误信息）。
        seen_tags: 已见 tag 集合（查重；命中即就地登记）。

    Raises:
        SchemaError: 记录非对象 / tag 缺失或非空 str / tag 重复 / name 非 str / team 不是非空 str。
    """
    if not isinstance(raw, dict):
        raise SchemaError(f"{path}: players[{idx}] 不是对象，实际 {type(raw).__name__}")
    tag: Any = raw.get("tag")
    if not isinstance(tag, str) or not tag:
        raise SchemaError(f"{path}: players[{idx}] tag 缺失或不是非空 str")
    if tag in seen_tags:
        raise SchemaError(f"{path}: players[{idx}] tag 重复: {tag!r}")
    seen_tags.add(tag)
    name: Any = raw.get("name", "")
    if not isinstance(name, str):
        raise SchemaError(f"{path}: players[{idx}]({tag}) name 不是 str")
    team: Any = raw.get("team")
    if not isinstance(team, str) or not team.strip():
        raise SchemaError(f"{path}: players[{idx}]({tag}) team 必须是非空字符串，实际 {team!r}")
    return Player(tag=tag, name=name, team=team)


def validate_assignment_key(key: Any, path: str) -> None:  # noqa: ANN401 JSON 待校验
    """校验单个 assignments 键是否满足 format_key 格式（``<file>#<t:.1f>``）。

    Raises:
        SchemaError: 键不是 str / 无 ``#`` / 时间部分非数字 / 时间与 ``f"{t:.1f}"`` 不一致。
    """
    if not isinstance(key, str):
        raise SchemaError(f"{path}: assignments 键不是 str: {key!r}")
    m = _KEY_PATTERN.match(key)
    if m is None:
        raise SchemaError(f"{path}: assignments 键格式错（需 <file>#<t:.1f>）: {key!r}")
    t_str: str = m.group("t")
    if f"{float(t_str):.1f}" != t_str:
        raise SchemaError(f"{path}: assignments 键时间部分未按 :.1f 格式化: {key!r}")


def validate_roster(data: Any, path: str) -> Roster:  # noqa: ANN401 JSON 待校验
    """校验 roster.json 结构并返回结构体（rules.md §0.2：schema 损坏必须显式失败）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        校验后的 Roster。

    Raises:
        SchemaError: 顶层非对象 / 缺 players / tag 重复 / team 不是非空 str /
            assignments 键格式错或值不是非空 str。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    players_raw: Any = data.get("players")
    if not isinstance(players_raw, list):
        raise SchemaError(f"{path}: 缺 players 列表或类型错误，实际 {type(players_raw).__name__}")
    seen_tags: set[str] = set()
    players: list[Player] = []
    for i, raw in enumerate(players_raw):
        players.append(player_from_dict(raw, path, i, seen_tags))

    assignments_raw: Any = data.get("assignments", {})
    if not isinstance(assignments_raw, dict):
        raise SchemaError(f"{path}: assignments 不是对象，实际 {type(assignments_raw).__name__}")
    assignments: dict[str, str] = {}
    for key, value in assignments_raw.items():
        validate_assignment_key(key, path)
        if not isinstance(value, str) or not value:
            raise SchemaError(f"{path}: assignments[{key!r}] 值不是非空 str")
        assignments[key] = value

    session: Any = data.get("session", "")
    if not isinstance(session, str):
        raise SchemaError(f"{path}: session 不是 str")
    confirmed: Any = data.get("confirmed", False)
    if not isinstance(confirmed, bool):
        raise SchemaError(f"{path}: confirmed 不是 bool")

    return Roster(
        session=session,
        confirmed=confirmed,
        players=tuple(players),
        assignments=assignments,
    )


def resolve_scorer(roster: Roster, query: str) -> Player | None:
    """在 roster 内解析进球者：tag 或 name 任一命中即返回（spec 真值表第 4 行）。

    tag 优先于 name；name 为空串的 player 不参与 name 匹配。

    Args:
        roster: 校验后的 Roster。
        query: 用户输入的 tag 或 name。

    Returns:
        命中的 Player；未命中返回 None。
    """
    for p in roster.players:
        if p.tag == query:
            return p
    for p in roster.players:
        if p.name and p.name == query:
            return p
    return None
