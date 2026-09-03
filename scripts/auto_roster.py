"""聚类结果+颜色分队 → auto_roster.json 转换（build 认人可选化，
spec: docs/build-auto-scorer/spec.md）。

输入：一个或多个 scorer_clusters.json（--clusters 可重复传参，跨文件合并、
    同 goal key 后者覆盖前者——与 cluster_scorers 合并 candidates 口径一致）；
    零个或多个 scorer_candidates.json（--candidates 可重复传参，读每球
    team_guess 作簇内颜色分队票源——crop_scorers HSV 躯干主色判据产物，
    同 key 后者覆盖前者）。
输出：--out 指定的 auto_roster.json（confirmed=false；players 每条
    {tag: <簇字母>, name: "", team: <簇内 team_guess 多数票>}，簇按 cluster_id
    升序映射 A/B/…/Z/AA…；平票取簇内最早有票球的队别，球缺 team_guess（status!=OK/
    旧数据）不计票，全簇无票归"便服"；assignments={<goal key>: <簇字母>}，
    unclustered 不进 assignments）。供 build_highlight --roster
    --allow-unconfirmed 逐队 --team 出 队伍_<队>_进球集锦.mp4、逐簇 --scorer
    <tag> 出 <队>_<标号>_进球合集.mp4（真值表⑤/④命名逻辑天然兼容，零改动）。
依赖：scripts/errors.py、scripts/pipe_common.py（read_json/atomic_write_json/日志）、
    scripts/roster.py（validate_roster 自校验，写读共用 spec M3 契约）。
典型调用：
    python scripts/auto_roster.py \
        --clusters work/20260801_对手名/scorers_auto/scorer_clusters.json \
        --candidates work/20260801_对手名/scorers_b1/scorer_candidates.json \
        --session 20260801_对手名 --out work/20260801_对手名/auto_roster.json

簇数异常（0 簇/1 簇/>15 簇）：INFO 留痕一行、不 WARNING、不阻塞（2026-08-22 用户定）；
0 簇写出空 players/assignments 的合法 roster（validate_roster 可通过），exit 0，
由 video.py 跳过队伍/个人合集步骤。clusters/candidates 文件 schema 损坏显式失败
（rules.md §0.2）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from errors import BasketballPipelineError, SchemaError
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import validate_roster

logger = logging.getLogger(__name__)

# 全簇无 team_guess 票时的归队（便服不进队伍集锦——build_highlight 真值表⑧拒收，
# 该簇只出个人合集；与 crop_scorers classify_team 的三值口径同字面值）
NO_VOTE_TEAM: str = "便服"
# candidates 里参与计票的裁图状态（crop_scorers STATUS_OK；SKIP 球 team_guess=None 天然不计票）
STATUS_OK: str = "OK"
# 簇数预期上限（超出仅 INFO 留痕不阻塞；半场局认人规模实测 <15）
EXPECTED_MAX_CLUSTERS: int = 15


def cluster_tag(n: int) -> str:
    """簇序号（1 起）→ 字母标号 A/B/…/Z/AA/AB/…（Excel 列名式进位）。

    Args:
        n: 簇序号（cluster_id 升序排后的位次，1 起）。

    Returns:
        字母标号。

    Raises:
        ValueError: n < 1。
    """
    if n < 1:
        raise ValueError(f"簇序号须 >= 1，实际 {n}")
    s: str = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def load_clusters(path: Path) -> list[dict[str, Any]]:
    """读取并校验单个 scorer_clusters.json，返回簇列表（按 cluster_id 升序）。

    Args:
        path: scorer_clusters.json 路径。

    Returns:
        簇记录列表（每条含 cluster_id:int、keys:list[str]），按 cluster_id 升序。

    Raises:
        SchemaError: 顶层非对象 / clusters 非列表 / 簇记录非对象 / cluster_id
            缺失或非 int / 同文件内 cluster_id 重复 / keys 不是 str 列表。
    """
    data: Any = read_json(path, what="scorer_clusters.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    clusters: Any = data.get("clusters")
    if not isinstance(clusters, list):
        raise SchemaError(f"{path}: 缺 clusters 列表或类型错误，实际 {type(clusters).__name__}")
    seen_ids: set[int] = set()
    for i, c in enumerate(clusters):
        if not isinstance(c, dict):
            raise SchemaError(f"{path}: clusters[{i}] 不是对象，实际 {type(c).__name__}")
        cid: Any = c.get("cluster_id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise SchemaError(
                f"{path}: clusters[{i}] cluster_id 缺失或不是 int，实际 {type(cid).__name__}"
            )
        if cid in seen_ids:
            raise SchemaError(f"{path}: cluster_id 重复: {cid}")
        seen_ids.add(cid)
        keys: Any = c.get("keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise SchemaError(f"{path}: clusters[{i}]({cid}) keys 不是 str 列表")
    return sorted(clusters, key=lambda c: c["cluster_id"])


def merge_clusters(payloads: list[list[dict[str, Any]]]) -> list[list[str]]:
    """跨文件合并簇：同 goal key 后者覆盖前者，返回各簇保留的键列表。

    簇身份 = (文件序, cluster_id)；键被后文件夺走后不再属于前文件的簇；
    被掏空的簇剔除。返回顺序 = 文件序 + cluster_id 升序。

    Args:
        payloads: 各文件 load_clusters 产物（按 --clusters 传参顺序，后者优先）。

    Returns:
        簇的键列表（每项为一簇保留的 goal key，保原簇内顺序）。
    """
    key_owner: dict[str, tuple[int, int]] = {}
    for fi, clusters in enumerate(payloads):
        for c in clusters:
            for k in c["keys"]:
                key_owner[k] = (fi, c["cluster_id"])
    groups: list[list[str]] = []
    for fi, clusters in enumerate(payloads):
        for c in clusters:
            ref: tuple[int, int] = (fi, c["cluster_id"])
            kept: list[str] = [k for k in c["keys"] if key_owner[k] == ref]
            if kept:
                groups.append(kept)
    return groups


def load_team_votes(path: Path) -> dict[str, str]:
    """读取并校验单个 scorer_candidates.json，返回 key → team_guess 计票表。

    仅 status=OK 且 team_guess 为非空 str 的球计票（SKIP 球 team_guess=None、
    旧数据缺字段均天然不计票）；校验口径对齐 cluster_scorers.load_candidates。

    Args:
        path: scorer_candidates.json 路径。

    Returns:
        key → team_guess（黑/白/便服）。

    Raises:
        SchemaError: 顶层非对象 / 缺 candidates 列表 / 条目非对象 /
            key 缺失或非空 str / status 非 str / team_guess 非 None 或 str。
    """
    data: Any = read_json(path, what="scorer_candidates.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    entries: Any = data.get("candidates")
    if not isinstance(entries, list):
        raise SchemaError(f"{path}: 缺 candidates 列表或类型错误")
    votes: dict[str, str] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchemaError(f"{path}: 第{i}条不是对象")
        key: Any = entry.get("key")
        if not isinstance(key, str) or not key:
            raise SchemaError(f"{path}: 第{i}条 key 缺失或不是非空 str")
        status: Any = entry.get("status")
        if not isinstance(status, str):
            raise SchemaError(f"{path}: 第{i}条({key}) status 不是 str")
        guess: Any = entry.get("team_guess")
        if guess is not None and not isinstance(guess, str):
            raise SchemaError(
                f"{path}: 第{i}条({key}) team_guess 必须是 str 或 null，实际 {type(guess).__name__}"
            )
        if status == STATUS_OK and guess:
            votes[key] = guess
    return votes


def cluster_team(keys: list[str], votes: dict[str, str]) -> str:
    """簇内球队多数票定队别：平票取簇内最早有票球的队别；全簇无票归 NO_VOTE_TEAM。

    Args:
        keys: 簇内 goal key（保簇内顺序，平票基准 = 顺序最前的有票球）。
        votes: load_team_votes 计票表（缺 key 即该球无票）。

    Returns:
        队别字符串（roster.py 合法值 = 非空 str）。
    """
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        t: str | None = votes.get(k)
        if t is None:
            continue
        counts[t] = counts.get(t, 0) + 1
        first.setdefault(t, i)
    if not counts:
        return NO_VOTE_TEAM
    # 票数降序、同票按首球出现序升序
    return min(counts, key=lambda t: (-counts[t], first[t]))


def build_auto_roster(
    session: str, clusters: list[list[str]], votes: dict[str, str]
) -> dict[str, Any]:
    """簇列表 → auto_roster.json 载荷（confirmed=false，tag=A/B/… 升序映射）。

    Args:
        session: 场次 ID（写顶层 session 字段）。
        clusters: merge_clusters 产物（键列表序 = tag 映射序）。
        votes: 各批 candidates 合并的 key → team_guess 计票表（决定 players.team）。

    Returns:
        可 JSON 序列化的 roster 载荷；空簇列表产空 players/assignments 的合法
        roster（0 簇不特殊对待）。
    """
    players: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for i, keys in enumerate(clusters, 1):
        tag: str = cluster_tag(i)
        players.append({"tag": tag, "name": "", "team": cluster_team(keys, votes)})
        for k in keys:
            assignments[k] = tag
    return {
        "session": session,
        "confirmed": False,
        "players": players,
        "assignments": assignments,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="聚类结果 → auto_roster.json（confirmed=false，供 --allow-unconfirmed 用）"
    )
    parser.add_argument(
        "--clusters",
        required=True,
        action="append",
        type=Path,
        help="scorer_clusters.json 路径（可重复传多个；同 goal key 后者覆盖前者）",
    )
    parser.add_argument(
        "--candidates",
        action="append",
        type=Path,
        default=[],
        help="scorer_candidates.json 路径（可重复传多个，读 team_guess 作颜色分队票源；"
        "同 key 后者覆盖前者；不传则全簇无票归便服）",
    )
    parser.add_argument("--session", required=True, help="场次 ID（写入顶层 session 字段）")
    parser.add_argument("--out", required=True, type=Path, help="auto_roster.json 输出路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=schema 损坏/IO 失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        payloads: list[list[dict[str, Any]]] = [load_clusters(p) for p in args.clusters]
        clusters: list[list[str]] = merge_clusters(payloads)
        votes: dict[str, str] = {}
        for p in args.candidates:
            votes.update(load_team_votes(p))  # 同 key 后者覆盖前者（同 clusters 口径）
        n: int = len(clusters)
        if n == 0 or n == 1 or n > EXPECTED_MAX_CLUSTERS:
            # 簇数异常仅 INFO 留痕一行（2026-08-22 用户定：不 WARNING、不阻塞）
            logger.info("簇数异常留痕: %d 簇（预期 2-%d，不阻塞）", n, EXPECTED_MAX_CLUSTERS)
        roster: dict[str, Any] = build_auto_roster(args.session, clusters, votes)
        # 写前自校验（写读共用 roster.py 契约；键格式坏显式失败，不产出坏文件）
        validate_roster(roster, str(args.out))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, roster, what="auto_roster.json")
        logger.info(
            "auto_roster 完成: %d 簇 / %d 归属球 → %s",
            n,
            len(roster["assignments"]),
            args.out,
        )
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1
    except OSError as e:
        logger.error("IO 失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
