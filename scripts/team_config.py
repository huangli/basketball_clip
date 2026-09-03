"""会话级队名配置（开源化脱敏：私人队名改为配置注入 + 通用默认值）。

单一来源模块：work/<场次>/team_config.json 的唯一读取/校验/默认值入口，
gen_scorer_page / video / photo_match_scorers / face_match_scorers 一律从
本模块取队名，禁止各自为政的重复实现。

配置契约（schema version 1）：
    {"version": 1, "team_name": "我方队名", "opponent": "对手名（可选）"}

容错口径（面向普通用户：完全不配置也必须能用）：
- 文件缺失 / JSON 损坏 / 顶层非对象 / version 不支持 / 字段类型错 →
  记 WARNING 并回退默认值，**不报错不中断**（rules.md"容忍缺失"侧：
  配置文件是用户可预期的可选项，非流水线数据契约）。
- 默认值：team_name="主队"；opponent=None（调用方回退场次 ID 后缀派生，
  最终兜底 DEFAULT_OPPONENT="对手"，见 gen_scorer_page.opponent_of）。

输入：会话目录（work/<场次>/）。
输出：TeamConfig 结构体。
典型调用：
    cfg = load_team_config(Path("work/20260722"))
    home = cfg.team_name
    opp = cfg.opponent or opponent_of(session)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEAM_CONFIG_NAME: str = "team_config.json"  # 会话目录下的配置文件名
TEAM_CONFIG_VERSION: int = 1  # schema 版本（契约变更即升）
DEFAULT_TEAM_NAME: str = "主队"  # 未配置时的我方队名
DEFAULT_OPPONENT: str = "对手"  # 未配置且场次 ID 无后缀时的对手名兜底


@dataclass(frozen=True, slots=True)
class TeamConfig:
    """校验后的队名配置。opponent=None 表示未配置（调用方按场次 ID 派生兜底）。"""

    team_name: str = DEFAULT_TEAM_NAME
    opponent: str | None = None


def load_team_config(session_dir: Path) -> TeamConfig:
    """读取 session_dir/team_config.json；缺失或损坏一律回退默认值（WARNING，不报错）。

    Args:
        session_dir: 会话目录（work/<场次>/）。

    Returns:
        TeamConfig；任何读取/校验失败都返回默认值，队名字段保证非空 str。
    """
    path: Path = session_dir / TEAM_CONFIG_NAME
    if not path.is_file():
        return TeamConfig()
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("%s 读取失败，回退默认队名: %s", path, exc)
        return TeamConfig()
    if not isinstance(data, dict):
        logger.warning("%s 顶层必须是对象（实际 %s），回退默认队名", path, type(data).__name__)
        return TeamConfig()
    version: Any = data.get("version", TEAM_CONFIG_VERSION)
    if version != TEAM_CONFIG_VERSION:
        logger.warning(
            "%s version=%r 不支持（当前 %d），回退默认队名", path, version, TEAM_CONFIG_VERSION
        )
        return TeamConfig()
    team_name: Any = data.get("team_name", DEFAULT_TEAM_NAME)
    if not isinstance(team_name, str) or not team_name.strip():
        logger.warning("%s team_name 缺失或非非空 str，回退 %r", path, DEFAULT_TEAM_NAME)
        team_name = DEFAULT_TEAM_NAME
    else:
        team_name = team_name.strip()
    opponent: Any = data.get("opponent")
    if opponent is not None:
        if not isinstance(opponent, str) or not opponent.strip():
            logger.warning("%s opponent 非非空 str，忽略该字段", path)
            opponent = None
        else:
            opponent = opponent.strip()
    return TeamConfig(team_name=team_name, opponent=opponent)
