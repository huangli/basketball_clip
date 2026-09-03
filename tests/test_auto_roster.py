"""auto_roster 单元测试（spec: docs/build-auto-scorer/spec.md §默认产物 3）。

覆盖：cluster_tag 字母映射（A…Z/AA…/边界报错）；load_clusters 校验与 cluster_id
升序排序、各类坏 schema 抛 SchemaError；merge_clusters 跨文件合并（同 key 后者
覆盖前者、掏空簇剔除）；load_team_votes 计票表（仅 OK 且 team_guess 非空计票、
坏 schema 抛 SchemaError）；cluster_team 多数票/平票取首球/无票归便服；
build_auto_roster 载荷契约（confirmed=false、team=簇内多数票队别/name 空、
unclustered 不进 assignments、validate_roster 自校验通过）；
main 端到端（0 簇合法空 roster exit 0、簇数异常 INFO 留痕、坏 schema exit 1）。
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pytest

from auto_roster import (
    build_auto_roster,
    cluster_tag,
    cluster_team,
    load_clusters,
    load_team_votes,
    main,
    merge_clusters,
)
from errors import SchemaError
from roster import validate_roster


def _clusters_payload(clusters: list[dict[str, Any]], unclustered: list[str] | None = None) -> dict:
    """构造 scorer_clusters.json 内容夹具。"""
    return {
        "version": "cluster-v1",
        "model": "ViT-B-32/laion2b_s34b_b79k",
        "threshold": 0.15,
        "clusters": clusters,
        "unclustered": unclustered or [],
    }


def _write_clusters(path: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    """落盘 clusters 夹具（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _candidate(key: str, team_guess: str | None, status: str = "OK") -> dict[str, Any]:
    """构造 scorer_candidates.json 单条记录夹具。"""
    return {"key": key, "status": status, "team_guess": team_guess}


def _write_candidates(path: pathlib.Path, entries: list[dict[str, Any]]) -> pathlib.Path:
    """落盘 candidates 夹具（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": entries}, ensure_ascii=False), encoding="utf-8")
    return path


class TestClusterTag:
    """cluster_id 升序 → A/B/…/Z/AA… 映射。"""

    def test_single_letter_range(self) -> None:
        assert cluster_tag(1) == "A"
        assert cluster_tag(2) == "B"
        assert cluster_tag(26) == "Z"

    def test_double_letter_carry(self) -> None:
        assert cluster_tag(27) == "AA"
        assert cluster_tag(28) == "AB"
        assert cluster_tag(52) == "AZ"
        assert cluster_tag(53) == "BA"

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            cluster_tag(0)


class TestLoadClusters:
    """load_clusters：契约校验 + cluster_id 升序排序。"""

    def test_valid_sorted_by_cluster_id(self, tmp_path: pathlib.Path) -> None:
        # Arrange：乱序 cluster_id
        path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload(
                [
                    {"cluster_id": 2, "keys": ["b.mp4#2.0"], "rep_crops": []},
                    {"cluster_id": 1, "keys": ["a.mp4#1.0"], "rep_crops": []},
                ]
            ),
        )
        # Act
        clusters = load_clusters(path)
        # Assert：按 cluster_id 升序
        assert [c["cluster_id"] for c in clusters] == [1, 2]

    def test_top_level_not_dict_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_clusters(tmp_path / "c.json", [1, 2])  # type: ignore[arg-type]
        with pytest.raises(SchemaError, match="顶层"):
            load_clusters(path)

    def test_clusters_not_list_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_clusters(tmp_path / "c.json", {"clusters": {}})
        with pytest.raises(SchemaError, match="clusters"):
            load_clusters(path)

    def test_cluster_id_missing_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_clusters(tmp_path / "c.json", _clusters_payload([{"keys": ["a.mp4#1.0"]}]))
        with pytest.raises(SchemaError, match="cluster_id"):
            load_clusters(path)

    def test_cluster_id_bool_raises(self, tmp_path: pathlib.Path) -> None:
        # bool 是 int 子类，必须显式排除
        path = _write_clusters(
            tmp_path / "c.json", _clusters_payload([{"cluster_id": True, "keys": []}])
        )
        with pytest.raises(SchemaError, match="cluster_id"):
            load_clusters(path)

    def test_duplicate_cluster_id_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload(
                [
                    {"cluster_id": 1, "keys": ["a.mp4#1.0"]},
                    {"cluster_id": 1, "keys": ["b.mp4#2.0"]},
                ]
            ),
        )
        with pytest.raises(SchemaError, match="重复"):
            load_clusters(path)

    def test_keys_not_str_list_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_clusters(
            tmp_path / "c.json", _clusters_payload([{"cluster_id": 1, "keys": [1, 2]}])
        )
        with pytest.raises(SchemaError, match="keys"):
            load_clusters(path)


class TestMergeClusters:
    """merge_clusters：跨文件同 key 后者覆盖前者；掏空簇剔除。"""

    def test_latter_overrides_same_key(self) -> None:
        # Arrange：k2 被文件 2 的簇 1 夺走
        payloads = [
            [{"cluster_id": 1, "keys": ["a.mp4#1.0", "b.mp4#2.0"]}],
            [{"cluster_id": 1, "keys": ["b.mp4#2.0", "c.mp4#3.0"]}],
        ]
        # Act
        groups = merge_clusters(payloads)
        # Assert：文件 1 簇 1 只剩 k1；文件 2 簇 1 拿 k2/k3
        assert groups == [["a.mp4#1.0"], ["b.mp4#2.0", "c.mp4#3.0"]]

    def test_emptied_cluster_dropped(self) -> None:
        # Arrange：文件 1 唯一的键被文件 2 夺走 → 该簇剔除
        payloads = [
            [{"cluster_id": 1, "keys": ["a.mp4#1.0"]}],
            [{"cluster_id": 1, "keys": ["a.mp4#1.0"]}],
        ]
        # Act / Assert
        assert merge_clusters(payloads) == [["a.mp4#1.0"]]

    def test_single_file_order_preserved(self) -> None:
        payloads = [
            [{"cluster_id": 1, "keys": ["a.mp4#1.0"]}, {"cluster_id": 2, "keys": ["b.mp4#2.0"]}]
        ]
        assert merge_clusters(payloads) == [["a.mp4#1.0"], ["b.mp4#2.0"]]


class TestLoadTeamVotes:
    """load_team_votes：仅 status=OK 且 team_guess 非空计票；坏 schema 显式失败。"""

    def test_ok_votes_skip_and_missing_excluded(self, tmp_path: pathlib.Path) -> None:
        # Arrange：OK 有票 / SKIP 无票 / OK 但缺 team_guess（旧数据）/ team_guess 空串
        path = _write_candidates(
            tmp_path / "scorer_candidates.json",
            [
                _candidate("a.mp4#1.0", "黑"),
                _candidate("b.mp4#2.0", None, status="SKIP"),
                {"key": "c.mp4#3.0", "status": "OK"},
                _candidate("d.mp4#4.0", ""),
            ],
        )
        # Act / Assert
        assert load_team_votes(path) == {"a.mp4#1.0": "黑"}

    def test_top_level_not_dict_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps([1, 2]), encoding="utf-8")
        with pytest.raises(SchemaError, match="顶层"):
            load_team_votes(path)

    def test_candidates_not_list_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_candidates(tmp_path / "c.json", [])
        path.write_text(json.dumps({"candidates": {}}), encoding="utf-8")
        with pytest.raises(SchemaError, match="candidates"):
            load_team_votes(path)

    def test_entry_not_dict_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"candidates": ["x"]}), encoding="utf-8")
        with pytest.raises(SchemaError, match="不是对象"):
            load_team_votes(path)

    def test_bad_key_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_candidates(tmp_path / "c.json", [{"status": "OK", "team_guess": "黑"}])
        with pytest.raises(SchemaError, match="key"):
            load_team_votes(path)

    def test_bad_status_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_candidates(
            tmp_path / "c.json", [{"key": "a.mp4#1.0", "status": 1, "team_guess": "黑"}]
        )
        with pytest.raises(SchemaError, match="status"):
            load_team_votes(path)

    def test_bad_team_guess_type_raises(self, tmp_path: pathlib.Path) -> None:
        path = _write_candidates(
            tmp_path / "c.json", [{"key": "a.mp4#1.0", "status": "OK", "team_guess": 7}]
        )
        with pytest.raises(SchemaError, match="team_guess"):
            load_team_votes(path)


class TestClusterTeam:
    """cluster_team：多数票 / 平票取簇内首球 / 全簇无票归便服。"""

    def test_majority_wins(self) -> None:
        votes = {"a.mp4#1.0": "黑", "b.mp4#2.0": "黑", "c.mp4#3.0": "白"}
        assert cluster_team(["a.mp4#1.0", "b.mp4#2.0", "c.mp4#3.0"], votes) == "黑"

    def test_tie_takes_first_goal_team(self) -> None:
        # Arrange：1:1 平票 → 簇内首球（a）的队别
        votes = {"a.mp4#1.0": "白", "b.mp4#2.0": "黑"}
        assert cluster_team(["a.mp4#1.0", "b.mp4#2.0"], votes) == "白"
        # 键序反过来则取另一队（首球按簇内顺序而非字典序）
        assert cluster_team(["b.mp4#2.0", "a.mp4#1.0"], votes) == "黑"

    def test_three_way_tie_takes_first(self) -> None:
        votes = {"a.mp4#1.0": "便服", "b.mp4#2.0": "黑", "c.mp4#3.0": "白"}
        assert cluster_team(["a.mp4#1.0", "b.mp4#2.0", "c.mp4#3.0"], votes) == "便服"

    def test_no_votes_falls_back_casual(self) -> None:
        assert cluster_team(["a.mp4#1.0"], {}) == "便服"
        # 票表里没有本簇的 key 同样无票
        assert cluster_team(["a.mp4#1.0"], {"x.mp4#9.0": "黑"}) == "便服"


class TestBuildAutoRoster:
    """build_auto_roster：载荷契约（confirmed=false / team=簇内多数票 / name 空）。"""

    def test_payload_contract(self) -> None:
        # Act：簇 1 多数票黑、簇 2 唯一票白
        roster = build_auto_roster(
            "s1",
            [["a.mp4#1.0", "b.mp4#2.0"], ["c.mp4#3.0"]],
            {"a.mp4#1.0": "黑", "b.mp4#2.0": "黑", "c.mp4#3.0": "白"},
        )
        # Assert
        assert roster["session"] == "s1"
        assert roster["confirmed"] is False
        assert roster["players"] == [
            {"tag": "A", "name": "", "team": "黑"},
            {"tag": "B", "name": "", "team": "白"},
        ]
        assert roster["assignments"] == {
            "a.mp4#1.0": "A",
            "b.mp4#2.0": "A",
            "c.mp4#3.0": "B",
        }
        # 自校验可通过（写读共用契约）
        validate_roster(roster, "auto_roster.json")

    def test_zero_clusters_legal_empty_roster(self) -> None:
        # Act：0 簇 → 空 players/assignments 的合法 roster
        roster = build_auto_roster("s1", [], {})
        # Assert
        assert roster["players"] == []
        assert roster["assignments"] == {}
        validate_roster(roster, "auto_roster.json")

    def test_no_votes_cluster_team_casual(self) -> None:
        # Act：全簇无票 → team 便服
        roster = build_auto_roster("s1", [["a.mp4#1.0"]], {})
        # Assert
        assert roster["players"] == [{"tag": "A", "name": "", "team": "便服"}]
        validate_roster(roster, "auto_roster.json")


class TestMain:
    """main 端到端：落盘、unclustered 排除、簇数异常 INFO、坏 schema exit 1。"""

    def test_end_to_end_writes_roster(self, tmp_path: pathlib.Path) -> None:
        # Arrange：2 簇 + 1 unclustered + candidates 票源（簇 1 黑、簇 2 白）
        clusters_path = _write_clusters(
            tmp_path / "scorers_auto" / "scorer_clusters.json",
            _clusters_payload(
                [
                    {"cluster_id": 1, "keys": ["a.mp4#1.0"], "rep_crops": []},
                    {"cluster_id": 2, "keys": ["b.mp4#2.0"], "rep_crops": []},
                ],
                unclustered=["c.mp4#3.0"],
            ),
        )
        candidates_path = _write_candidates(
            tmp_path / "scorers_b1" / "scorer_candidates.json",
            [_candidate("a.mp4#1.0", "黑"), _candidate("b.mp4#2.0", "白")],
        )
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(
            [
                "--clusters",
                str(clusters_path),
                "--candidates",
                str(candidates_path),
                "--session",
                "s1",
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        roster = json.loads(out.read_text(encoding="utf-8"))
        assert roster["confirmed"] is False
        assert roster["players"] == [
            {"tag": "A", "name": "", "team": "黑"},
            {"tag": "B", "name": "", "team": "白"},
        ]
        # unclustered 不进 assignments
        assert roster["assignments"] == {"a.mp4#1.0": "A", "b.mp4#2.0": "B"}

    def test_no_candidates_all_casual(self, tmp_path: pathlib.Path) -> None:
        # Arrange：不传 --candidates → 全簇无票归便服（合法 roster，exit 0）
        clusters_path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload([{"cluster_id": 1, "keys": ["a.mp4#1.0"], "rep_crops": []}]),
        )
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
        # Assert
        assert rc == 0
        roster = json.loads(out.read_text(encoding="utf-8"))
        assert roster["players"] == [{"tag": "A", "name": "", "team": "便服"}]

    def test_multi_candidates_latter_overrides(self, tmp_path: pathlib.Path) -> None:
        # Arrange：两个 candidates 文件，同 key 后者覆盖（a.mp4 黑→白）
        clusters_path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload([{"cluster_id": 1, "keys": ["a.mp4#1.0"], "rep_crops": []}]),
        )
        c1 = _write_candidates(tmp_path / "c1.json", [_candidate("a.mp4#1.0", "黑")])
        c2 = _write_candidates(tmp_path / "c2.json", [_candidate("a.mp4#1.0", "白")])
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(
            [
                "--clusters",
                str(clusters_path),
                "--candidates",
                str(c1),
                "--candidates",
                str(c2),
                "--session",
                "s1",
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 0
        roster = json.loads(out.read_text(encoding="utf-8"))
        assert roster["players"] == [{"tag": "A", "name": "", "team": "白"}]

    def test_bad_candidates_schema_exit1(self, tmp_path: pathlib.Path) -> None:
        # Arrange：candidates 结构损坏 → SchemaError 显式失败，不产出文件
        clusters_path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload([{"cluster_id": 1, "keys": ["a.mp4#1.0"], "rep_crops": []}]),
        )
        bad = tmp_path / "bad_candidates.json"
        bad.write_text(json.dumps({"candidates": {}}), encoding="utf-8")
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(
            [
                "--clusters",
                str(clusters_path),
                "--candidates",
                str(bad),
                "--session",
                "s1",
                "--out",
                str(out),
            ]
        )
        # Assert
        assert rc == 1
        assert not out.exists()

    def test_multi_clusters_files_merged(self, tmp_path: pathlib.Path) -> None:
        # Arrange：两个 clusters 文件，同 key 后者覆盖
        c1 = _write_clusters(
            tmp_path / "c1.json",
            _clusters_payload([{"cluster_id": 1, "keys": ["a.mp4#1.0", "b.mp4#2.0"]}]),
        )
        c2 = _write_clusters(
            tmp_path / "c2.json", _clusters_payload([{"cluster_id": 1, "keys": ["b.mp4#2.0"]}])
        )
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(
            ["--clusters", str(c1), "--clusters", str(c2), "--session", "s1", "--out", str(out)]
        )
        # Assert：k1 归 A（文件1簇1），k2 归 B（文件2簇1）
        assert rc == 0
        roster = json.loads(out.read_text(encoding="utf-8"))
        assert roster["assignments"] == {"a.mp4#1.0": "A", "b.mp4#2.0": "B"}

    def test_zero_clusters_exit0_empty_roster(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        clusters_path = _write_clusters(tmp_path / "c.json", _clusters_payload([]))
        out = tmp_path / "auto_roster.json"
        # Act
        with caplog.at_level(logging.INFO):
            rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
        # Assert：exit 0 + 空合法 roster + INFO 留痕（非 WARNING）
        assert rc == 0
        roster = json.loads(out.read_text(encoding="utf-8"))
        assert roster["players"] == []
        assert roster["assignments"] == {}
        validate_roster(roster, str(out))
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("簇数异常留痕" in r.message for r in info)
        assert not any(
            "簇数异常" in r.message for r in caplog.records if r.levelno >= logging.WARNING
        )

    def test_anomaly_counts_info_not_warning(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：1 簇与 16 簇都属异常档（预期 2-15）
        for keys_groups in ([["a.mp4#1.0"]], [[f"f{i}.mp4#{i}.0"] for i in range(16)]):
            clusters_path = _write_clusters(
                tmp_path / "c.json",
                _clusters_payload(
                    [
                        {"cluster_id": i, "keys": ks, "rep_crops": []}
                        for i, ks in enumerate(keys_groups, 1)
                    ]
                ),
            )
            out = tmp_path / "auto_roster.json"
            caplog.clear()
            # Act
            with caplog.at_level(logging.INFO):
                rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
            # Assert：不阻塞、INFO 留痕、无 WARNING
            assert rc == 0
            assert any(
                "簇数异常留痕" in r.message and r.levelno == logging.INFO for r in caplog.records
            )
            assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_normal_count_no_anomaly_log(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：3 簇属正常档
        clusters_path = _write_clusters(
            tmp_path / "c.json",
            _clusters_payload(
                [
                    {"cluster_id": i, "keys": [f"f{i}.mp4#{i}.0"], "rep_crops": []}
                    for i in range(1, 4)
                ]
            ),
        )
        out = tmp_path / "auto_roster.json"
        # Act
        with caplog.at_level(logging.INFO):
            rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
        # Assert
        assert rc == 0
        assert not any("簇数异常" in r.message for r in caplog.records)

    def test_bad_schema_exit1(self, tmp_path: pathlib.Path) -> None:
        # Arrange：clusters 结构损坏 → SchemaError 显式失败，不产出文件
        clusters_path = _write_clusters(tmp_path / "c.json", {"clusters": {}})
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
        # Assert
        assert rc == 1
        assert not out.exists()

    def test_malformed_assignment_key_rejected(self, tmp_path: pathlib.Path) -> None:
        # Arrange：簇内 key 不满足 format_key 格式 → 写前自校验 SchemaError
        clusters_path = _write_clusters(
            tmp_path / "c.json", _clusters_payload([{"cluster_id": 1, "keys": ["不是合法键"]}])
        )
        out = tmp_path / "auto_roster.json"
        # Act
        rc = main(["--clusters", str(clusters_path), "--session", "s1", "--out", str(out)])
        # Assert
        assert rc == 1
        assert not out.exists()
