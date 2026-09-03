"""video.py 统一入口单元测试：命令拼装逐字断言、批次双轨、尺寸换算、state、错误传播。

全部用 tmp_path 伪造 work/<场次>/ 产物；subprocess.run 一律 monkeypatch 拦截，
不启动任何真子进程（rules.md §9：慢外部 mock）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import stat
import subprocess
import sys
from typing import Any

import pytest

import video
from errors import BasketballPipelineError, SchemaError
from roster import format_key
from video import Batch

SESSION: str = "s1"
SCRIPT_DIR: pathlib.Path = pathlib.Path(video.SCRIPT_DIR)
# video.py 内部用相对 work/<场次>/ 路径（与 run_session 一致），测试 chdir 到 tmp_path 后
# 命令参数里的路径串按相对口径断言
REL: pathlib.Path = pathlib.Path("work") / SESSION


def _write_json(path: pathlib.Path, data: Any) -> None:  # noqa: ANN401 JSON 内容不定
    """落盘 JSON 测试夹具（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _goals_payload(n_confirmed: int = 2) -> dict[str, Any]:
    """构造 goals.json 内容：n 条 confirmed + 1 条 rejected（不计数）。"""
    goals = [
        {"status": "confirmed", "file": f"f{i}.mp4", "anchor_time": float(i) + 0.5}
        for i in range(n_confirmed)
    ]
    goals.append({"status": "rejected", "file": "fx.mp4", "anchor_time": 99.0})
    return {"goals": goals}


def _facts_payload(width: int = 3840, height: int = 2160) -> dict[str, Any]:
    """构造 session_facts.json 内容（单文件）。"""
    return {"files": {"a.mp4": {"width": width, "height": height, "fps": 50.0}}}


@pytest.fixture
def session_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """隔离的 work/<SESSION>/ 目录（chdir 到 tmp_path，video.WORK_ROOT 相对解析）。"""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "work" / SESSION
    d.mkdir(parents=True)
    return d


@pytest.fixture
def run_recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, str]]]:
    """拦截 subprocess.run：记录 (cmd, env) 并返回成功。"""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(
        cmd: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        calls.append((list(cmd), dict(env)))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    return calls


def _fail_recorder(
    monkeypatch: pytest.MonkeyPatch, fail_at: int, returncode: int = 1
) -> list[list[str]]:
    """拦截 subprocess.run：第 fail_at 次（0 起）调用返回非零，其余成功。"""

    calls: list[list[str]] = []
    state = {"n": 0}

    def fake_run(
        cmd: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        rc = returncode if state["n"] == fail_at else 0
        state["n"] += 1
        return subprocess.CompletedProcess(cmd, rc)

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    return calls


class TestScore:
    """score：透传 run_session 命令拼装、state 写入、dry-run 不写。"""

    def test_command_verbatim_minimal(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Act
        rc = video.main(["score", "素材目录", "--session", SESSION])
        # Assert：逐字断言最小命令（无可选旗标时一个都不多传）
        assert rc == 0
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "run_session.py"),
            "素材目录",
            "--session",
            SESSION,
        ]

    def test_command_verbatim_full_flags(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rc = video.main(
            [
                "score",
                "素材目录",
                "--session",
                SESSION,
                "--batch-size",
                "30",
                "--fids",
                "a,b,c",
                "--force",
            ]
        )
        assert rc == 0
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "run_session.py"),
            "素材目录",
            "--session",
            SESSION,
            "--batch-size",
            "30",
            "--fids",
            "a,b,c",
            "--force",
        ]

    def test_env_injects_pythonioencoding(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        video.main(["score", "素材目录", "--session", SESSION])
        assert run_recorder[0][1]["PYTHONIOENCODING"] == "utf-8"

    def test_state_written_on_success(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        src = tmp_path / "素材目录"
        src.mkdir()
        video.main(["score", str(src), "--session", SESSION])
        state = json.loads((session_dir / "video_cli.json").read_text(encoding="utf-8"))
        assert state["version"] == 1
        assert state["session"] == SESSION
        assert state["srcdir"] == str(src.resolve())
        assert len(state["runs"]) == 1
        assert state["runs"][0]["cmd"] == "score"
        assert state["runs"][0]["exit_code"] == 0

    def test_state_runs_append_only(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        src = tmp_path / "素材目录"
        src.mkdir()
        video.main(["score", str(src), "--session", SESSION])
        video.main(["score", str(src), "--session", SESSION])
        state = json.loads((session_dir / "video_cli.json").read_text(encoding="utf-8"))
        assert len(state["runs"]) == 2

    def test_dry_run_no_state(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rc = video.main(["score", "素材目录", "--session", SESSION, "--dry-run"])
        assert rc == 0
        assert "--dry-run" in run_recorder[0][0]
        assert not (session_dir / "video_cli.json").exists()

    def test_nonzero_stops_exit1(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fail_recorder(monkeypatch, fail_at=0)
        rc = video.main(["score", "素材目录", "--session", SESSION])
        assert rc == 1
        assert not (session_dir / "video_cli.json").exists()


class TestDiscoverBatches:
    """批次发现双轨：goals.json（旧布局）/ goals_batchK.json（现行）/ 同 K 冲突。"""

    def test_old_layout_batch1(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "goals.json", _goals_payload())
        batches = video.discover_batches(session_dir)
        assert batches == [
            Batch(
                1,
                session_dir / "goals.json",
                session_dir / "candidates.json",
                session_dir / "review",
                session_dir / "scorers",
            )
        ]

    def test_new_layout_batch_k(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        batches = video.discover_batches(session_dir)
        assert len(batches) == 1
        b = batches[0]
        assert b.batch == 2
        assert b.candidates == session_dir / "candidates_batch2.json"
        assert b.review_dir == session_dir / "review_batch2"
        assert b.scorers_dir == session_dir / "scorers_b2"

    def test_same_k_dual_layout_conflict(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "goals.json", _goals_payload())
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        with pytest.raises(BasketballPipelineError, match="双布局并存"):
            video.discover_batches(session_dir)

    def test_no_goals_at_all(self, session_dir: pathlib.Path) -> None:
        with pytest.raises(BasketballPipelineError, match="无 goals"):
            video.discover_batches(session_dir)

    def test_sorted_and_unrecognized_skipped(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "goals_batch3.json", _goals_payload())
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "goals_legacy_20260722.json", _goals_payload())
        batches = video.discover_batches(session_dir)
        assert [b.batch for b in batches] == [1, 3]


class TestPeople:
    """people：四段链（裁图→传播→聚类→确认页）命令拼装、max-reads 换算、
    --index/--clusters 条件传递。"""

    def _setup_batch(self, session_dir: pathlib.Path, *, events_index: bool = True) -> pathlib.Path:
        """备好现行布局批次 2 的全部前置产物，返回 rawdir。"""
        _write_json(session_dir / "goals_batch2.json", _goals_payload(2))
        _write_json(session_dir / "candidates_batch2.json", [])
        if events_index:
            _write_json(session_dir / "review_batch2" / "events_index.json", {"events": []})
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def test_four_steps_verbatim(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # B-1：代理注入以 BASKETBALL_CLIP_HTTPS_PROXY 为准，未设不注入
        monkeypatch.setenv("BASKETBALL_CLIP_HTTPS_PROXY", "http://127.0.0.1:17890")
        rawdir = self._setup_batch(session_dir)
        # --no-read-numbers：读号默认关（photo-roster T12 起，v2.1 零 token 定案），
        # 本用例锁定四段链的裸骨架，显式关掉读号保持断言面最小
        rc = video.main(
            ["people", "--session", SESSION, "--rawdir", str(rawdir), "--no-read-numbers"]
        )
        assert rc == 0
        assert len(run_recorder) == 4
        # ① 裁图（--no-read-numbers 时一个读号旗标都不带）
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "crop_scorers.py"),
            "--goals",
            str(REL / "goals_batch2.json"),
            "--detectdir",
            str(pathlib.Path("work/detect")),
            "--framesdir",
            str(pathlib.Path("work/frames")),
            "--out",
            str(REL / "scorers_b2"),
            "--candidates",
            str(REL / "candidates_batch2.json"),
            "--rawdir",
            str(rawdir),
        ]
        # ①.5 传播：消费 ① 的 scorer_candidates.json，detectdir/framesdir 取值同 ①
        assert run_recorder[1][0] == [
            sys.executable,
            str(SCRIPT_DIR / "propagate_scorers.py"),
            "--candidates",
            str(REL / "scorers_b2" / "scorer_candidates.json"),
            "--detectdir",
            str(pathlib.Path("work/detect")),
            "--framesdir",
            str(pathlib.Path("work/frames")),
        ]
        # ② 聚类：显式定档 --linkage complete --threshold 0.15，clusters 落本批目录
        assert run_recorder[2][0] == [
            sys.executable,
            str(SCRIPT_DIR / "cluster_scorers.py"),
            "--candidates",
            str(REL / "scorers_b2" / "scorer_candidates.json"),
            "--out",
            str(REL / "scorers_b2" / "scorer_clusters.json"),
            "--linkage",
            "complete",
            "--threshold",
            "0.15",
        ]
        assert run_recorder[2][1]["HTTPS_PROXY"] == "http://127.0.0.1:17890"
        assert run_recorder[2][1]["PYTHONIOENCODING"] == "utf-8"
        # ③ 确认页：--index 存在才传、--clusters 同目录；track_links.json 缺失
        # （mock 子进程不产真文件）→ 执行时剥掉 --track-links
        assert run_recorder[3][0] == [
            sys.executable,
            str(SCRIPT_DIR / "gen_scorer_page.py"),
            "--scorers",
            str(REL / "scorers_b2" / "scorer_candidates.json"),
            "--goals",
            str(REL / "goals_batch2.json"),
            "--session",
            SESSION,
            "--index",
            str(REL / "review_batch2" / "events_index.json"),
            "--clusters",
            str(REL / "scorers_b2" / "scorer_clusters.json"),
        ]

    def test_env_https_proxy_only_on_cluster_step(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange：排除父进程已带 HTTPS_PROXY 的干扰；经 BASKETBALL_CLIP_HTTPS_PROXY 注入代理
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.setenv("BASKETBALL_CLIP_HTTPS_PROXY", "http://127.0.0.1:17890")
        rawdir = self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        base_keys = set(os.environ) | {"PYTHONIOENCODING"}
        # ①①.5③ 非聚类段：相对 os.environ 无额外键（锁定 HTTPS_PROXY 仅聚类段叠加）
        for idx in (0, 1, 3):
            assert set(run_recorder[idx][1]) - base_keys == set()
        # ② 聚类段：恰好只多 HTTPS_PROXY
        assert set(run_recorder[2][1]) - base_keys == {"HTTPS_PROXY"}
        assert run_recorder[2][1]["HTTPS_PROXY"] == "http://127.0.0.1:17890"

    def test_read_numbers_max_reads_default(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--read-numbers"])
        assert rc == 0
        crop_cmd = run_recorder[0][0]
        # 缺省 = confirmed 2 条 ×3 = 6（rejected 不计）
        assert crop_cmd[-3:] == ["--read-numbers", "--max-reads", "6"]

    def test_read_numbers_max_reads_explicit(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        rc = video.main(
            [
                "people",
                "--session",
                SESSION,
                "--rawdir",
                str(rawdir),
                "--read-numbers",
                "--max-reads",
                "9",
            ]
        )
        assert rc == 0
        assert run_recorder[0][0][-2:] == ["--max-reads", "9"]

    def test_read_numbers_default_off(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # 读号默认关（photo-roster T12，v2.1 零 token 定案）：不传旗标一个读号参数都不带
        rawdir = self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        crop_cmd = run_recorder[0][0]
        assert "--read-numbers" not in crop_cmd
        assert "--max-reads" not in crop_cmd

    def test_no_read_numbers_disables(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # 显式关闭（便服/无号场次省 token）：旗标与预算都不出现
        rawdir = self._setup_batch(session_dir)
        rc = video.main(
            ["people", "--session", SESSION, "--rawdir", str(rawdir), "--no-read-numbers"]
        )
        assert rc == 0
        crop_cmd = run_recorder[0][0]
        assert "--read-numbers" not in crop_cmd
        assert "--max-reads" not in crop_cmd

    def test_index_omitted_when_missing(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir, events_index=False)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert "--index" not in run_recorder[3][0]

    def test_skip_cluster(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--skip-cluster"])
        assert rc == 0
        # 只跑 ①①.5③ 三段：--skip-cluster 不影响传播步骤，③ 不传 --clusters
        assert len(run_recorder) == 3
        assert "propagate_scorers.py" in run_recorder[1][0][1]
        assert all("cluster_scorers.py" not in cmd[1] for cmd, _ in run_recorder)
        assert "--clusters" not in run_recorder[2][0]

    def test_roster_existing_and_players_file(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        _write_json(
            session_dir / "roster.json",
            {"players": [{"tag": "红-7", "name": "", "team": "主队"}], "assignments": {}},
        )
        players_file = session_dir / "players.json"
        _write_json(players_file, [])
        rc = video.main(
            [
                "people",
                "--session",
                SESSION,
                "--rawdir",
                str(rawdir),
                "--players-file",
                str(players_file),
            ]
        )
        assert rc == 0
        page_cmd = run_recorder[3][0]
        assert "--roster-existing" in page_cmd
        assert page_cmd[page_cmd.index("--roster-existing") + 1] == str(REL / "roster.json")
        assert "--players-file" in page_cmd

    def test_batch_filter(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir / "goals_batch3.json", _goals_payload())
        _write_json(session_dir / "candidates_batch3.json", [])
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--batch", "3"])
        assert rc == 0
        # 只跑批次 3（4 段），批次 2 不跑
        assert len(run_recorder) == 4
        assert any("goals_batch3.json" in item for item in run_recorder[0][0])

    def test_batch_not_found(self, session_dir: pathlib.Path) -> None:
        rawdir = self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--batch", "9"])
        assert rc == 1

    def test_missing_candidates_skips_batch(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：批次 2 缺 candidates（跳）、批次 3 齐（跑）
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        _write_json(session_dir / "goals_batch3.json", _goals_payload())
        _write_json(session_dir / "candidates_batch3.json", [])
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert len(run_recorder) == 4
        assert any("goals_batch3.json" in item for item in run_recorder[0][0])

    def test_rawdir_from_state(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        self._setup_batch(session_dir)
        src = tmp_path / "state素材"
        src.mkdir()
        _write_json(
            session_dir / "video_cli.json",
            {"version": 1, "session": SESSION, "srcdir": str(src), "runs": []},
        )
        rc = video.main(["people", "--session", SESSION])
        assert rc == 0
        crop_cmd = run_recorder[0][0]
        # 读号旗标按显式开关出现（T12 起默认关），按旗标定位取 rawdir 值
        assert crop_cmd[crop_cmd.index("--rawdir") + 1] == str(src)

    def test_rawdir_missing_everywhere(self, session_dir: pathlib.Path) -> None:
        self._setup_batch(session_dir)
        rc = video.main(["people", "--session", SESSION])
        assert rc == 1

    def test_nonzero_stops_exit1(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup_batch(session_dir)
        calls = _fail_recorder(monkeypatch, fail_at=1)  # ①.5 传播段失败
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        # 非零即停：只跑了 ①①.5 两步，②③ 未执行
        assert rc == 1
        assert len(calls) == 2

    def test_dry_run_executes_nothing(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup_batch(session_dir)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--dry-run"])
        assert rc == 0

    def test_old_layout_paths(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        _write_json(session_dir / "goals.json", _goals_payload())
        _write_json(session_dir / "candidates.json", [])
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        crop_cmd = run_recorder[0][0]
        assert str(REL / "scorers") in crop_cmd
        assert str(REL / "candidates.json") in crop_cmd


class TestPeoplePhotoMatch:
    """people ②.5 照片匹配串联（docs/photo-roster/spec.md T6 串法，T12 换人脸 matcher）。

    串法：②聚类 后插 ②.5 照片匹配（条件：photos/ 存在且非 --skip-cluster）；
    确认页带 --photo-matches；仅 ②.5 允许失败降级（ERROR 留痕、③ 照出、
    产物缺失剥旗标降级为无预填）；①②③ 失败语义不变（任一步失败中断整链）。
    T12 起 ②.5 执行体 = face_match_scorers.py（L1 人脸单路，无 --cache 参数，
    face_cache 落 candidates 同目录）；CLIP 版 photo_match_scorers.py 已退出。
    """

    def _setup_batch(self, session_dir: pathlib.Path, *, photos: bool = True) -> pathlib.Path:
        """备好现行布局批次 2 前置产物（可选建 photos/ 库目录），返回 rawdir。"""
        _write_json(session_dir / "goals_batch2.json", _goals_payload(2))
        _write_json(session_dir / "candidates_batch2.json", [])
        _write_json(session_dir / "review_batch2" / "events_index.json", {"events": []})
        if photos:
            (session_dir.parent.parent / "photos").mkdir()
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    @staticmethod
    def _args(**over: object) -> argparse.Namespace:
        """build_people_steps 直接调用用的最小参数命名空间。"""
        base: dict[str, object] = {
            "skip_cluster": False,
            "read_numbers": False,
            "max_reads": None,
            "players_file": None,
            "photo_match": True,  # 既有用例测的是显式开启路径；默认关见 test_default_off
        }
        base.update(over)
        return argparse.Namespace(**base)

    def test_default_off_no_photo_step(self, session_dir: pathlib.Path) -> None:
        # Arrange：photos/ 存在但 --photo-match 未开（默认关，2026-08-29 纯人工定案）
        rawdir = self._setup_batch(session_dir)
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(photo_match=False), batch, rawdir, session_dir)
        # Assert：无 ②.5、确认页不带 --photo-matches，传播/聚类照跑
        assert [s.title for s in steps] == [
            "批次2①裁图",
            "批次2①.5传播",
            "批次2②聚类",
            "批次2③确认页",
        ]
        assert all("face_match_scorers.py" not in s.argv[1] for s in steps)
        assert "--photo-matches" not in steps[3].argv

    def test_default_off_main_level(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：photos/ 存在 + main() 不带 --photo-match（argparse 默认关端到端锁定）
        rawdir = self._setup_batch(session_dir)
        # Act
        rc = video.main(
            ["people", "--session", SESSION, "--rawdir", str(rawdir), "--no-read-numbers"]
        )
        # Assert：四段链、无人脸匹配调用、确认页无 --photo-matches
        assert rc == 0
        assert len(run_recorder) == 4
        assert all("face_match_scorers.py" not in cmd[1] for cmd, _ in run_recorder)
        assert "--photo-matches" not in run_recorder[3][0]

    def test_photo_step_and_page_flag(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setenv("BASKETBALL_CLIP_HTTPS_PROXY", "http://127.0.0.1:17890")
        rawdir = self._setup_batch(session_dir)
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert：② 后插 ②.5，逐字断言匹配命令（人脸 matcher：无 --cache，产物同目录）
        assert [s.title for s in steps] == [
            "批次2①裁图",
            "批次2①.5传播",
            "批次2②聚类",
            "批次2②.5照片匹配",
            "批次2③确认页",
        ]
        assert list(steps[3].argv) == [
            sys.executable,
            str(SCRIPT_DIR / "face_match_scorers.py"),
            "--photos",
            "photos",
            "--candidates",
            str(REL / "scorers_b2" / "scorer_candidates.json"),
            "--out",
            str(REL / "scorers_b2" / "photo_matches.json"),
        ]
        assert steps[3].env_extra == {"HTTPS_PROXY": "http://127.0.0.1:17890"}
        assert steps[3].allow_fail is True  # 仅 ②.5 允许失败降级
        assert steps[0].allow_fail is False
        assert steps[1].allow_fail is False
        assert steps[2].allow_fail is False
        assert steps[4].allow_fail is False
        page = list(steps[4].argv)
        assert page[page.index("--photo-matches") + 1] == str(
            REL / "scorers_b2" / "photo_matches.json"
        )

    def test_no_photos_dir_no_photo_step(self, session_dir: pathlib.Path) -> None:
        # Arrange：缺照片库 → 整步跳过不阻塞（三段链原样）
        rawdir = self._setup_batch(session_dir, photos=False)
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert
        assert len(steps) == 4
        assert all("face_match_scorers.py" not in s.argv[1] for s in steps)
        assert all("photo_match_scorers.py" not in s.argv[1] for s in steps)
        assert "--photo-matches" not in steps[3].argv

    def test_skip_cluster_no_photo_step(self, session_dir: pathlib.Path) -> None:
        # Arrange：photos/ 存在但 --skip-cluster（无聚类段，②.5 同口径跳过）
        rawdir = self._setup_batch(session_dir)
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(skip_cluster=True), batch, rawdir, session_dir)
        # Assert
        assert len(steps) == 3
        assert all("face_match_scorers.py" not in s.argv[1] for s in steps)
        assert all("photo_match_scorers.py" not in s.argv[1] for s in steps)
        assert "--photo-matches" not in steps[2].argv

    def test_photo_step_failure_degrades(
        self,
        session_dir: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：②.5 失败（0 起第 3 次调用）
        rawdir = self._setup_batch(session_dir)
        calls = _fail_recorder(monkeypatch, fail_at=3)
        # Act
        with caplog.at_level(logging.ERROR):
            rc = video.main(
                [
                    "people",
                    "--session",
                    SESSION,
                    "--rawdir",
                    str(rawdir),
                    "--no-read-numbers",
                    "--photo-match",
                ]
            )
        # Assert：ERROR 留痕、不中断整链、③ 确认页照出且剥掉 --photo-matches（无预填）
        assert rc == 0
        assert len(calls) == 5
        assert "face_match_scorers.py" in calls[3][1]
        assert "--photo-matches" not in calls[4]
        assert any("降级" in r.message for r in caplog.records)

    def test_page_flag_kept_when_output_exists(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：photo_matches.json 已存在（断点续跑/上轮产物）→ 探测通过保留旗标
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir / "scorers_b2" / "photo_matches.json", {"matches": {}})
        # Act
        rc = video.main(
            [
                "people",
                "--session",
                SESSION,
                "--rawdir",
                str(rawdir),
                "--no-read-numbers",
                "--photo-match",
            ]
        )
        # Assert
        assert rc == 0
        assert len(run_recorder) == 5
        page_cmd = run_recorder[4][0]
        assert page_cmd[page_cmd.index("--photo-matches") + 1] == str(
            REL / "scorers_b2" / "photo_matches.json"
        )

    def test_other_steps_failure_semantics_unchanged(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：② 聚类失败（photos 存在 + --photo-match 开，链含 ②.5）
        rawdir = self._setup_batch(session_dir)
        calls = _fail_recorder(monkeypatch, fail_at=2)
        # Act
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--photo-match"])
        # Assert：①①.5②③ 失败语义不变——非零即停，②.5/③ 未执行
        assert rc == 1
        assert len(calls) == 3


class TestPeoplePropagate:
    """people ①.5 轨迹传播接线（docs/scorer-propagate/spec.md §Commands，Task 6）。

    链序：裁图 → 传播 → 聚类 → 确认页；传播恒在链中（--skip-cluster 不影响，
    失败语义同 ①②③ 非零即停）；确认页 --track-links 预传 + 执行时探测——
    track_links.json 存在才传（mock 子进程不产真文件时剥旗标），与
    --photo-matches 同口径。
    """

    def _setup_batch(self, session_dir: pathlib.Path) -> pathlib.Path:
        """备好现行布局批次 2 的全部前置产物，返回 rawdir。"""
        _write_json(session_dir / "goals_batch2.json", _goals_payload(2))
        _write_json(session_dir / "candidates_batch2.json", [])
        _write_json(session_dir / "review_batch2" / "events_index.json", {"events": []})
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def test_track_links_absent_stripped(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：mock 子进程不产真文件 → 执行时探测 track_links.json 缺失
        rawdir = self._setup_batch(session_dir)
        # Act
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        # Assert：确认页剥掉 --track-links（无传播预填照出）
        assert rc == 0
        assert len(run_recorder) == 4
        assert "propagate_scorers.py" in run_recorder[1][0][1]
        assert "--track-links" not in run_recorder[3][0]

    def test_track_links_present_kept(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：track_links.json 已存在（断点续跑/上轮产物）→ 探测通过保留旗标
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir / "scorers_b2" / "track_links.json", {"version": "track-v1"})
        # Act
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        # Assert
        assert rc == 0
        page_cmd = run_recorder[3][0]
        assert page_cmd[page_cmd.index("--track-links") + 1] == str(
            REL / "scorers_b2" / "track_links.json"
        )

    def test_dry_run_includes_propagate_step(
        self,
        session_dir: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        rawdir = self._setup_batch(session_dir)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        # Act
        with caplog.at_level(logging.INFO):
            rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir), "--dry-run"])
        # Assert：DRY-RUN 打印含 ①.5 传播步骤（命令逐字含脚本与三必填参数）
        assert rc == 0
        assert any(
            "①.5传播" in r.message and "propagate_scorers.py" in r.message for r in caplog.records
        )


class TestNamesPlayers:
    """photos/names.json 全局号码→姓名名单自动注入 --players（2026-08-28 用户供名单）。

    串法：确认页拼参时 --players-file 显式优先，否则 names.json 存在即自动注入
    --players（tag=白<号>，号码升序去零）；名单损坏 SchemaError 显式失败。
    """

    def _setup_batch(self, session_dir: pathlib.Path) -> pathlib.Path:
        """备好批次 2 前置产物 + photos/ 目录，返回 rawdir。"""
        _write_json(session_dir / "goals_batch2.json", _goals_payload(2))
        _write_json(session_dir / "candidates_batch2.json", [])
        photos = session_dir.parent.parent / "photos"
        photos.mkdir()
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    @staticmethod
    def _args(**over: object) -> argparse.Namespace:
        base: dict[str, object] = {
            "skip_cluster": False,
            "read_numbers": False,
            "max_reads": None,
            "players_file": None,
            "photo_match": False,  # 名单注入与 ②.5 无关，默认关路径即可
        }
        base.update(over)
        return argparse.Namespace(**base)

    def test_names_json_injects_players(self, session_dir: pathlib.Path) -> None:
        # Arrange：乱序+前导零 → 号码升序去零
        rawdir = self._setup_batch(session_dir)
        _write_json(
            session_dir.parent.parent / "photos" / "names.json",
            {"22": "朱勇", "6": "测试员丙", "07": "老七"},
        )
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert
        page = list(steps[-1].argv)
        assert page[page.index("--players") + 1] == "白6=测试员丙,白7=老七,白22=朱勇"

    def test_explicit_players_file_wins(self, session_dir: pathlib.Path) -> None:
        # Arrange：显式 --players-file 优先于 names.json 自动注入
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir.parent.parent / "photos" / "names.json", {"6": "测试员丙"})
        players_file = session_dir / "players.json"
        _write_json(players_file, [])
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(
            self._args(players_file=players_file), batch, rawdir, session_dir
        )
        # Assert
        page = list(steps[-1].argv)
        assert "--players-file" in page
        assert "--players" not in page

    def test_empty_names_no_flag(self, session_dir: pathlib.Path) -> None:
        # Arrange：空对象 → 不传 --players
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir.parent.parent / "photos" / "names.json", {})
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert
        assert "--players" not in list(steps[-1].argv)

    def test_no_names_file_no_flag(self, session_dir: pathlib.Path) -> None:
        # Arrange：无 names.json → 不传 --players（现状行为不变）
        rawdir = self._setup_batch(session_dir)
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert
        assert "--players" not in list(steps[-1].argv)

    def test_names_with_roster_existing_coexists(self, session_dir: pathlib.Path) -> None:
        # Arrange：已有 roster.json（既有确认）+ names.json 同存——两旗标并传
        # （页面侧"players 以新名单为准"为既有惯例，此测试锁定编排侧行为不漂移）
        rawdir = self._setup_batch(session_dir)
        _write_json(session_dir.parent.parent / "photos" / "names.json", {"6": "测试员丙"})
        _write_json(
            session_dir / "roster.json",
            {"session": "s", "confirmed": True, "players": [], "assignments": {}},
        )
        batch = video.discover_batches(REL)[0]
        # Act
        steps = video.build_people_steps(self._args(), batch, rawdir, session_dir)
        # Assert
        page = list(steps[-1].argv)
        assert "--roster-existing" in page
        assert page[page.index("--players") + 1] == "白6=测试员丙"

    @pytest.mark.parametrize(
        ("payload", "err_part"),
        [
            ([1, 2], "顶层必须是对象"),
            ({"x": "张三"}, "纯数字"),
            ({"６": "张三"}, "纯数字"),  # 全角数字同拒
            ({"6": ""}, "非空 str"),
            ({"6": 7}, "非空 str"),
            ({"6": "黄,立"}, "逗号"),  # --players 串分隔符，防拆出假球员
            ({"07": "甲", "7": "乙"}, "撞车"),  # 去零后同号
        ],
    )
    def test_schema_error(self, tmp_path: pathlib.Path, payload: object, err_part: str) -> None:
        # Arrange
        bad = tmp_path / "names.json"
        _write_json(bad, payload)
        # Act / Assert：名单损坏显式失败（类型锁定 SchemaError）
        with pytest.raises(SchemaError, match=err_part):
            video.load_names_players(bad)


class TestResolveOutSizes:
    """build 尺寸换算：返回 (1080p, 4K) 两态；16:9 / 4:3 / 混比例或未知报错。"""

    def test_16_9(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "session_facts.json", _facts_payload(3840, 2160))
        assert video.resolve_out_sizes(session_dir) == ("1920x1080", "3840x2160")

    def test_16_9_within_tolerance(self, session_dir: pathlib.Path) -> None:
        # 容差 ±1% 内（3830x2160 ≈ 16:9 - 0.26%）
        _write_json(session_dir / "session_facts.json", _facts_payload(3830, 2160))
        assert video.resolve_out_sizes(session_dir) == ("1920x1080", "3840x2160")

    def test_4_3(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "session_facts.json", _facts_payload(2880, 2160))
        assert video.resolve_out_sizes(session_dir) == ("1440x1080", "2880x2160")

    def test_4_3_within_tolerance(self, session_dir: pathlib.Path) -> None:
        # 容差 ±1% 内（2860x2160 ≈ 4:3 - 0.69%，与 16:9 侧对称）
        _write_json(session_dir / "session_facts.json", _facts_payload(2860, 2160))
        assert video.resolve_out_sizes(session_dir) == ("1440x1080", "2880x2160")

    def test_mixed_ratios_error_lists_files(self, session_dir: pathlib.Path) -> None:
        facts = {
            "files": {
                "a.mp4": {"width": 3840, "height": 2160},
                "b.mp4": {"width": 2880, "height": 2160},
            }
        }
        _write_json(session_dir / "session_facts.json", facts)
        with pytest.raises(BasketballPipelineError) as exc_info:
            video.resolve_out_sizes(session_dir)
        msg = str(exc_info.value)
        assert "a.mp4" in msg and "b.mp4" in msg
        assert "16:9" in msg and "4:3" in msg

    def test_unknown_ratio_error(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "session_facts.json", _facts_payload(1000, 1000))
        with pytest.raises(BasketballPipelineError, match="未知"):
            video.resolve_out_sizes(session_dir)

    def test_missing_facts(self, session_dir: pathlib.Path) -> None:
        with pytest.raises(BasketballPipelineError, match="session_facts"):
            video.resolve_out_sizes(session_dir)


class TestBuild:
    """build：命令拼装、--all 展开、互斥、roster 缺失、错误传播。"""

    def _setup(self, session_dir: pathlib.Path, *, roster: bool = False) -> pathlib.Path:
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "candidates_batch1.json", [])
        _write_json(session_dir / "session_facts.json", _facts_payload())
        if roster:
            _write_json(
                session_dir / "roster.json",
                {
                    "confirmed": True,
                    "players": [
                        {"tag": "红-7", "name": "测试员甲", "team": "主队"},
                        {"tag": "黑-A", "name": "", "team": "对手"},
                        {"tag": "黑-B", "name": "", "team": "对手"},
                    ],
                    "assignments": {
                        format_key("f0.mp4", 0.5): "红-7",
                        format_key("f1.mp4", 1.5): "黑-A",
                    },
                },
            )
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def test_command_verbatim_default(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # confirmed roster 无过滤：全归属球合集，带 --roster 不带 --scorer/--team
        # （无 roster 的默认自动模式见 TestBuildAuto）
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "build_highlight.py"),
            "--goals",
            str(REL / "goals_batch1.json"),
            "--roster",
            str(REL / "roster.json"),
            "--rawdir",
            str(rawdir),
            "--out",
            "1920x1080",
        ]

    def test_roster_passed_when_present(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(
            ["build", "--session", SESSION, "--rawdir", str(rawdir), "--scorer", "红-7"]
        )
        assert rc == 0
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "build_highlight.py"),
            "--goals",
            str(REL / "goals_batch1.json"),
            "--roster",
            str(REL / "roster.json"),
            "--rawdir",
            str(rawdir),
            "--out",
            "1920x1080",
            "--scorer",
            "红-7",
        ]

    def test_team_our_team_defaults_4k_named(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # 主队 队伍集锦 默认 4K（原名，无 _4K 后缀）
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--team", "主队"])
        assert rc == 0
        cmd = run_recorder[0][0]
        assert "--out" in cmd and cmd[cmd.index("--out") + 1] == "3840x2160"
        assert "--name-suffix" not in cmd

    def test_4k_flag_scorer_adds_suffix_and_4k(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # --scorer + --4k：所选步骤 4K + _4K 后缀
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(
            ["build", "--session", SESSION, "--rawdir", str(rawdir), "--scorer", "红-7", "--4k"]
        )
        assert rc == 0
        assert len(run_recorder) == 1
        cmd = run_recorder[0][0]
        assert "--out" in cmd and cmd[cmd.index("--out") + 1] == "3840x2160"
        assert "--name-suffix" in cmd and cmd[cmd.index("--name-suffix") + 1] == "_4K"

    def test_4k_flag_all_expands_4k_except_our_team(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # --all --4k：其余步骤 4K+后缀；主队步骤 4K 原名（OUR_TEAM no-op）
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all", "--4k"])
        assert rc == 0
        assert len(run_recorder) == 4
        our_team = next(c[0] for c in run_recorder if c[0][-2:] == ["--team", "主队"])
        assert "--name-suffix" not in our_team
        assert our_team[our_team.index("--out") + 1] == "3840x2160"
        for c in run_recorder:
            if c[0][-2:] == ["--team", "主队"]:
                continue
            assert c[0][c[0].index("--out") + 1] == "3840x2160"
            assert "--name-suffix" in c[0] and c[0][c[0].index("--name-suffix") + 1] == "_4K"

    def test_4k_flag_our_team_noop_no_duplicate(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # --team 主队 --4k：no-op——4K 原名、无后缀、无重复文件（仅 1 命令）
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(
            ["build", "--session", SESSION, "--rawdir", str(rawdir), "--team", "主队", "--4k"]
        )
        assert rc == 0
        assert len(run_recorder) == 1
        cmd = run_recorder[0][0]
        assert "--name-suffix" not in cmd
        assert cmd[cmd.index("--out") + 1] == "3840x2160"

    def test_4k_flag_auto_mode_warned_and_ignored(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 未认人自动模式：--4k 被忽略并 WARNING（自动链无完整拍头时仍可能报错，
        # 此处只验证 --4k 不透传、不影响自动模式三件套口径）
        rawdir = self._setup(session_dir)  # 无 roster → 自动模式
        caplog.set_level(logging.WARNING)
        video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--4k"])
        assert any("忽略 --4k" in r.message for r in caplog.records)
        for c in run_recorder:
            if not str(c[0][1]).endswith("build_highlight.py"):
                continue
            assert c[0][c[0].index("--out") + 1] == "1920x1080"
            assert "--name-suffix" not in c[0]

    def test_plain_all_4k_for_our_team(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # spec 用例 1：普通 --all，主队步骤 4K 原名，其余步骤 1080p 无后缀
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 0
        assert len(run_recorder) == 4
        our = next(c[0] for c in run_recorder if c[0][-2:] == ["--team", "主队"])
        assert our[our.index("--out") + 1] == "3840x2160"
        assert "--name-suffix" not in our
        for c in run_recorder:
            if c[0][-2:] == ["--team", "主队"]:
                continue
            assert c[0][c[0].index("--out") + 1] == "1920x1080"
            assert "--name-suffix" not in c[0]

    def test_4k_flag_without_selector_all_goal(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # spec 用例 4：--4k 不带选择器 → 全员合集 4K + _4K 后缀，无 --scorer/--team
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--4k"])
        assert rc == 0
        assert len(run_recorder) == 1
        cmd = run_recorder[0][0]
        assert cmd[cmd.index("--out") + 1] == "3840x2160"
        assert "--name-suffix" in cmd and cmd[cmd.index("--name-suffix") + 1] == "_4K"
        assert "--scorer" not in cmd and "--team" not in cmd

    def test_all_idempotent_4k_stable(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # spec 用例 8：同参数连跑两次 --all，命令列表一致；主队始终 4K 原名无后缀
        rawdir = self._setup(session_dir, roster=True)
        argv = ["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"]
        assert video.main(argv) == 0
        assert video.main(argv) == 0
        assert len(run_recorder) == 8
        first = [c[0] for c in run_recorder[:4]]
        second = [c[0] for c in run_recorder[4:]]
        assert first == second
        our = next(c[0] for c in run_recorder[:4] if c[0][-2:] == ["--team", "主队"])
        assert our[our.index("--out") + 1] == "3840x2160"
        assert "--name-suffix" not in our

    def test_all_expands_players_and_teams(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 0
        # 黑-B 无归属球零命中跳过：2 人（红-7、黑-A）+ 2 队（主队、对手）= 4 条
        assert len(run_recorder) == 4
        tail = [c[0][-2:] for c in run_recorder]
        assert tail == [
            ["--scorer", "红-7"],
            ["--scorer", "黑-A"],
            ["--team", "主队"],
            ["--team", "对手"],
        ]

    def test_batch_filter(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        _write_json(session_dir / "candidates_batch2.json", [])
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--batch", "2"])
        assert rc == 0
        # 只跑批次 2，批次 1 不跑
        assert len(run_recorder) == 1
        assert str(REL / "goals_batch2.json") in run_recorder[0][0]

    def test_all_skips_casual_team(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rawdir = self._setup(session_dir)
        _write_json(
            session_dir / "roster.json",
            {
                "confirmed": True,
                "players": [
                    {"tag": "红-7", "name": "", "team": "主队"},
                    {"tag": "便-X", "name": "", "team": "便服"},
                ],
                "assignments": {
                    format_key("f0.mp4", 0.5): "红-7",
                },
            },
        )
        caplog.set_level(logging.WARNING)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 0
        tail = [c[0][-2:] for c in run_recorder]
        # 便-X 无归属球零命中跳过；便服分队合集跳过（build_highlight 拒收 --team 便服）
        assert tail == [["--scorer", "红-7"], ["--team", "主队"]]
        assert "便服" in caplog.text

    def test_all_roster_missing(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 无 roster + --all：2026-08-22 起走自动模式（过滤旗标忽略 WARNING），不再拒收
        rawdir = self._setup(session_dir, roster=False)
        _write_json(session_dir / "auto_roster.json", {"players": [], "assignments": {}})
        caplog.set_level(logging.WARNING)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 0
        assert run_recorder  # 自动模式三产物链照常跑
        assert "被忽略" in caplog.text

    def test_all_roster_schema_bad(self, session_dir: pathlib.Path) -> None:
        rawdir = self._setup(session_dir)
        _write_json(session_dir / "roster.json", {"players": [{"tag": "x", "team": "不存在队"}]})
        # roster schema 损坏（team 非法值）→ validate_roster 抛 SchemaError，退出 1
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 1

    def test_mutex_violation(self, session_dir: pathlib.Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            video.main(["build", "--session", SESSION, "--scorer", "红-7", "--team", "主队"])
        assert exc_info.value.code == 2

    def test_4_3_out_size(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)
        _write_json(session_dir / "session_facts.json", _facts_payload(2880, 2160))
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert run_recorder[0][0][-2:] == ["--out", "1440x1080"]

    def test_nonzero_stops_exit1(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)
        calls = _fail_recorder(monkeypatch, fail_at=1)  # 第二个合集失败
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        assert rc == 1
        assert len(calls) == 2

    def test_dry_run_executes_nothing(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup(session_dir, roster=True)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        rc = video.main(
            ["build", "--session", SESSION, "--rawdir", str(rawdir), "--all", "--dry-run"]
        )
        assert rc == 0


class TestBuildHeatmap:
    """build 收尾热图触发（v4.2，docs/heatmap/spec.md）：合集全成后自动调
    goal_heatmap.heat_session；roster 缺失 INFO 跳过；热图失败不阻塞主链；
    dry-run 不执行。"""

    def _setup(self, session_dir: pathlib.Path, *, roster: bool = True) -> pathlib.Path:
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "candidates_batch1.json", [])
        _write_json(session_dir / "session_facts.json", _facts_payload())
        if roster:
            _write_json(
                session_dir / "roster.json",
                {
                    "confirmed": True,
                    "players": [{"tag": "红-7", "name": "", "team": "主队"}],
                    "assignments": {format_key("f0.mp4", 0.5): "红-7"},
                },
            )
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def _record_heatmap(self, monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
        """拦截 goal_heatmap.heat_session（懒 import 后经模块属性调用，patch 模块本体生效）。"""
        import goal_heatmap

        calls: list[pathlib.Path] = []
        monkeypatch.setattr(goal_heatmap, "heat_session", lambda sd, *a, **k: calls.append(sd))
        return calls

    def test_heatmap_triggered_after_build(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._record_heatmap(monkeypatch)
        rawdir = self._setup(session_dir)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert len(run_recorder) == 1  # 合集照常合成
        assert calls == [
            pathlib.Path("work") / SESSION
        ]  # 只传 session_dir（目录推导在 goal_heatmap 侧）

    def test_heatmap_failure_does_not_block(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import goal_heatmap

        def boom(sd: pathlib.Path) -> None:
            raise RuntimeError("热图炸了")

        monkeypatch.setattr(goal_heatmap, "heat_session", boom)
        rawdir = self._setup(session_dir)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0  # 附属产物失败不改 build 返回码
        assert len(run_recorder) == 1

    def test_heatmap_skipped_without_roster(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._record_heatmap(monkeypatch)
        rawdir = self._setup(session_dir, roster=False)
        # 自动模式（无 confirmed roster）：热图不新触发；预写空 auto_roster 使链走完
        _write_json(session_dir / "auto_roster.json", {"players": [], "assignments": {}})
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert calls == []  # 未认人是预期常态，不调 heat_session

    def test_heatmap_not_run_on_dry_run(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup(session_dir)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程/热图")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        import goal_heatmap

        monkeypatch.setattr(goal_heatmap, "heat_session", forbidden)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--dry-run"])
        assert rc == 0


class TestBuildMultiBatch:
    """多批次合并合成 + --all 零命中跳过（docs/build-multi-batch/spec.md）。"""

    def _setup_two_batches(self, session_dir: pathlib.Path) -> pathlib.Path:
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        _write_json(session_dir / "candidates_batch1.json", [])
        _write_json(session_dir / "candidates_batch2.json", [])
        _write_json(session_dir / "session_facts.json", _facts_payload())
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def _roster_full_hits(self, session_dir: pathlib.Path) -> None:
        _write_json(
            session_dir / "roster.json",
            {
                "confirmed": True,
                "players": [
                    {"tag": "红-7", "name": "", "team": "主队"},
                    {"tag": "黑-A", "name": "", "team": "对手"},
                ],
                "assignments": {
                    format_key("f0.mp4", 0.5): "红-7",
                    format_key("f1.mp4", 1.5): "黑-A",
                },
            },
        )

    def test_multi_batch_merges_goals_and_single_call_per_filter(
        self, session_dir: pathlib.Path, run_recorder: list
    ) -> None:
        # Arrange
        rawdir = self._setup_two_batches(session_dir)
        self._roster_full_hits(session_dir)
        # Act
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        # Assert：2 球员 + 2 队 = 4 条命令，每 filter 只调一次，--goals 指向合并文件
        assert rc == 0
        assert len(run_recorder) == 4
        for cmd, _env in run_recorder:
            assert str(REL / "merged_goals_cli.json") in cmd
        # 合并文件已写盘：两批各 2 confirmed+1 rejected 逐字拼接 + session 字段
        merged = json.loads((session_dir / "merged_goals_cli.json").read_text("utf-8"))
        assert merged["session"] == SESSION
        assert len(merged["goals"]) == 6

    def test_all_skips_zero_hit_players_and_teams(
        self,
        session_dir: pathlib.Path,
        run_recorder: list,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：黑-A 有归属但不在 confirmed 键集（键对不上）→ 零命中
        rawdir = self._setup_two_batches(session_dir)
        _write_json(
            session_dir / "roster.json",
            {
                "confirmed": True,
                "players": [
                    {"tag": "红-7", "name": "", "team": "主队"},
                    {"tag": "黑-A", "name": "", "team": "对手"},
                ],
                "assignments": {
                    format_key("f0.mp4", 0.5): "红-7",
                    "ghost.mp4#9.9": "黑-A",
                },
            },
        )
        caplog.set_level(logging.WARNING)
        # Act
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        # Assert：只出 红-7 + 主队；黑-A 与 对手 零命中跳过（WARNING）
        assert rc == 0
        tail = [c[0][-2:] for c in run_recorder]
        assert tail == [["--scorer", "红-7"], ["--team", "主队"]]
        assert "零命中" in caplog.text

    def test_all_zero_hits_exit1(self, session_dir: pathlib.Path, run_recorder: list) -> None:
        # Arrange：assignments 全对不上 confirmed 键
        rawdir = self._setup_two_batches(session_dir)
        _write_json(
            session_dir / "roster.json",
            {
                "confirmed": True,
                "players": [{"tag": "红-7", "name": "", "team": "主队"}],
                "assignments": {"ghost.mp4#9.9": "红-7"},
            },
        )
        # Act
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--all"])
        # Assert：全零命中 → exit 1 且无子进程
        assert rc == 1
        assert run_recorder == []

    def test_dry_run_does_not_write_merged_file(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        rawdir = self._setup_two_batches(session_dir)
        self._roster_full_hits(session_dir)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        # Act
        rc = video.main(
            ["build", "--session", SESSION, "--rawdir", str(rawdir), "--all", "--dry-run"]
        )
        # Assert
        assert rc == 0
        assert not (session_dir / "merged_goals_cli.json").exists()


class TestBuildAuto:
    """build 自动模式（无 confirmed roster，docs/build-auto-scorer/spec.md）：
    进球片段 + 自动识别链（裁图/聚类/auto_roster）+ 逐颜色队队伍集锦 + 逐簇个人合集。"""

    def _setup(self, session_dir: pathlib.Path, *, auto_roster: bool = True) -> pathlib.Path:
        """备好单批次前置产物；auto_roster=True 时预写 auto_roster.json
        （mock 子进程不产真文件，④ 的产物由夹具代写）。"""
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "candidates_batch1.json", [])
        _write_json(session_dir / "session_facts.json", _facts_payload())
        if auto_roster:
            _write_json(
                session_dir / "auto_roster.json",
                {
                    "confirmed": False,
                    "players": [{"tag": "A", "name": "", "team": "黑"}],
                    "assignments": {format_key("f0.mp4", 0.5): "A"},
                },
            )
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    def test_auto_chain_verbatim(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BASKETBALL_CLIP_HTTPS_PROXY", "http://127.0.0.1:17890")
        rawdir = self._setup(session_dir)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert len(run_recorder) == 6
        # ① 进球片段（真值表⑨，不传 --roster；自动模式不出全员集锦）
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "build_highlight.py"),
            "--goals",
            str(REL / "goals_batch1.json"),
            "--rawdir",
            str(rawdir),
            "--out",
            "1920x1080",
            "--per-goal",
        ]
        # 自动模式不出全员集锦：唯一不带 --roster 的 build_highlight 调用是 ① --per-goal
        no_roster_hl = [
            c[0]
            for c in run_recorder
            if c[0][1].endswith("build_highlight.py") and "--roster" not in c[0]
        ]
        assert no_roster_hl == [run_recorder[0][0]]
        # ② 裁图：不带 --read-numbers（自动合集允许有误，不开 K3 读号）
        crop_cmd = run_recorder[1][0]
        assert crop_cmd[1].endswith("crop_scorers.py")
        assert "--read-numbers" not in crop_cmd
        assert "--max-reads" not in crop_cmd
        # ③ 聚类：跨批合并 --out scorers_auto/，定稿 complete/0.15，带 HTTPS_PROXY
        assert run_recorder[2][0] == [
            sys.executable,
            str(SCRIPT_DIR / "cluster_scorers.py"),
            "--candidates",
            str(REL / "scorers_b1" / "scorer_candidates.json"),
            "--out",
            str(REL / "scorers_auto" / "scorer_clusters.json"),
            "--linkage",
            "complete",
            "--threshold",
            "0.15",
        ]
        assert run_recorder[2][1]["HTTPS_PROXY"] == "http://127.0.0.1:17890"
        # ④ auto_roster：带 --candidates 各批票源（簇内颜色分队多数票）
        assert run_recorder[3][0] == [
            sys.executable,
            str(SCRIPT_DIR / "auto_roster.py"),
            "--clusters",
            str(REL / "scorers_auto" / "scorer_clusters.json"),
            "--candidates",
            str(REL / "scorers_b1" / "scorer_candidates.json"),
            "--session",
            SESSION,
            "--out",
            str(REL / "auto_roster.json"),
        ]
        # ⑤ 逐颜色队：--roster auto_roster.json + --team <队> + --allow-unconfirmed
        team_cmd = run_recorder[4][0]
        assert team_cmd[1].endswith("build_highlight.py")
        assert team_cmd[team_cmd.index("--roster") + 1] == str(REL / "auto_roster.json")
        assert team_cmd[team_cmd.index("--team") + 1] == "黑"
        assert "--allow-unconfirmed" in team_cmd
        # ⑥ 逐簇：--roster auto_roster.json + --scorer <tag> + --allow-unconfirmed
        tag_cmd = run_recorder[5][0]
        assert tag_cmd[1].endswith("build_highlight.py")
        assert tag_cmd[tag_cmd.index("--roster") + 1] == str(REL / "auto_roster.json")
        assert tag_cmd[tag_cmd.index("--scorer") + 1] == "A"
        assert "--allow-unconfirmed" in tag_cmd

    def test_unconfirmed_roster_triggers_auto_with_warning(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rawdir = self._setup(session_dir)
        _write_json(
            session_dir / "roster.json",
            {
                "confirmed": False,
                "players": [{"tag": "红-7", "name": "", "team": "主队"}],
                "assignments": {},
            },
        )
        caplog.set_level(logging.WARNING)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert "按未认人处理" in caplog.text
        # ① 不传 --roster（未确认 roster 不传给 build_highlight）
        assert "--roster" not in run_recorder[0][0]

    def test_crop_idempotent_skip(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：scorer_candidates.json 已存在且可读 → ② 幂等跳过
        rawdir = self._setup(session_dir)
        _write_json(session_dir / "scorers_b1" / "scorer_candidates.json", {"candidates": []})
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        scripts_run = [c[0][1] for c in run_recorder]
        assert not any(s.endswith("crop_scorers.py") for s in scripts_run)
        # 聚类/auto_roster/逐簇照常
        assert any(s.endswith("cluster_scorers.py") for s in scripts_run)

    def test_missing_candidates_skips_identify_chain(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：批次缺 candidates.json → ① 照常，识别链整链跳过，exit 0
        rawdir = self._setup(session_dir)
        (session_dir / "candidates_batch1.json").unlink()
        caplog.set_level(logging.WARNING)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        assert len(run_recorder) == 1  # 仅 ① --per-goal
        assert "无可聚类候选" in caplog.text

    def test_zero_clusters_skips_scorer_builds(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：auto_roster 0 簇（空 players/assignments）→ ⑤⑥ 跳过
        rawdir = self._setup(session_dir, auto_roster=False)
        _write_json(session_dir / "auto_roster.json", {"players": [], "assignments": {}})
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        # ①+②③④ = 4 步，无逐队/逐簇 build_highlight 调用
        assert len(run_recorder) == 4
        assert not any("--allow-unconfirmed" in c[0] for c in run_recorder)

    def test_casual_team_excluded_from_team_highlights(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：auto_roster 两簇——A 黑队、B 便服队，均有归属球
        rawdir = self._setup(session_dir, auto_roster=False)
        _write_json(
            session_dir / "auto_roster.json",
            {
                "confirmed": False,
                "players": [
                    {"tag": "A", "name": "", "team": "黑"},
                    {"tag": "B", "name": "", "team": "便服"},
                ],
                "assignments": {
                    format_key("f0.mp4", 0.5): "A",
                    format_key("f1.mp4", 1.5): "B",
                },
            },
        )
        caplog.set_level(logging.INFO)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        # Assert：便服队不进队伍集锦循环（INFO 留痕），其簇个人合集照出
        assert rc == 0
        team_cmds = [c[0] for c in run_recorder if "--team" in c[0]]
        assert [c[c.index("--team") + 1] for c in team_cmds] == ["黑"]
        scorer_cmds = [c[0] for c in run_recorder if "--scorer" in c[0]]
        assert [c[c.index("--scorer") + 1] for c in scorer_cmds] == ["A", "B"]
        assert "便服队不进队伍集锦" in caplog.text

    def test_identify_chain_failure_keeps_main_products_exit1(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：③ 聚类段失败（第 2 次子进程，0 起）
        rawdir = self._setup(session_dir)
        calls = _fail_recorder(monkeypatch, fail_at=2)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        # Assert：①② 已跑，④⑤⑥ 跳过，退出 1
        assert rc == 1
        assert len(calls) == 3

    def test_auto_roster_product_missing_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：④ 报成功但产物缺失（异常现场）→ ERROR 退出 1
        rawdir = self._setup(session_dir, auto_roster=False)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 1
        assert len(run_recorder) == 4

    def test_bad_roster_schema_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：roster schema 损坏 → 显式失败，不降级走自动模式
        rawdir = self._setup(session_dir)
        _write_json(session_dir / "roster.json", {"players": [{"tag": "x", "team": "  "}]})
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 1
        assert run_recorder == []

    def test_batch_filter_limits_goals_and_candidates(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：两批次，--batch 2 只作用于批次 2
        rawdir = self._setup(session_dir)
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        _write_json(session_dir / "candidates_batch2.json", [])
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--batch", "2"])
        assert rc == 0
        # ① goals 指向批次 2（单批不合并）；③ 裁图只跑批次 2；④ candidates 只有批次 2
        assert str(REL / "goals_batch2.json") in run_recorder[0][0]
        assert "merged_goals_cli.json" not in " ".join(run_recorder[0][0])
        crop_cmd = next(c[0] for c in run_recorder if c[0][1].endswith("crop_scorers.py"))
        assert str(REL / "scorers_b2") in crop_cmd
        cluster_cmd = next(c[0] for c in run_recorder if c[0][1].endswith("cluster_scorers.py"))
        candidates_args = [
            cluster_cmd[i + 1] for i, v in enumerate(cluster_cmd) if v == "--candidates"
        ]
        assert candidates_args == [str(REL / "scorers_b2" / "scorer_candidates.json")]

    def test_multi_batch_merges_goals_and_cluster_candidates(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange：两批次自动模式
        rawdir = self._setup(session_dir)
        _write_json(session_dir / "goals_batch2.json", _goals_payload())
        _write_json(session_dir / "candidates_batch2.json", [])
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        # ① goals 指向合并文件
        assert str(REL / "merged_goals_cli.json") in run_recorder[0][0]
        # ④ 单次聚类合并两批 candidates（跨批簇标一致）
        cluster_cmd = next(c[0] for c in run_recorder if c[0][1].endswith("cluster_scorers.py"))
        candidates_args = [
            cluster_cmd[i + 1] for i, v in enumerate(cluster_cmd) if v == "--candidates"
        ]
        assert candidates_args == [
            str(REL / "scorers_b1" / "scorer_candidates.json"),
            str(REL / "scorers_b2" / "scorer_candidates.json"),
        ]

    def test_dry_run_executes_nothing(
        self, session_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rawdir = self._setup(session_dir)

        def forbidden(*a: object, **kw: object) -> None:
            raise AssertionError("dry-run 不得启动子进程")

        monkeypatch.setattr(video.subprocess, "run", forbidden)
        rc = video.main(["build", "--session", SESSION, "--rawdir", str(rawdir), "--dry-run"])
        assert rc == 0
        assert not (session_dir / "merged_goals_cli.json").exists()


class TestMainEntry:
    """入口行为：无子命令退出 2；场次目录缺失退出 1。"""

    def test_no_subcommand_exit2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert video.main([]) == 2
        assert "score" in capsys.readouterr().out

    def test_session_dir_missing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert video.main(["people", "--session", "nope", "--rawdir", "x"]) == 1

    def test_state_version_mismatch(self, session_dir: pathlib.Path) -> None:
        _write_json(session_dir / "video_cli.json", {"version": 99, "runs": []})
        with pytest.raises(BasketballPipelineError, match="版本"):
            video.load_state(SESSION)


class TestRelocate:
    """relocate=True（真实 CLI 入口）：相对路径按启动目录解析、chdir 到 REPO_ROOT。"""

    def _setup(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """构造假仓库根与独立启动目录，返回 (repo, launch)。"""
        repo: pathlib.Path = tmp_path / "repo"
        (repo / "work" / SESSION).mkdir(parents=True)
        launch: pathlib.Path = tmp_path / "elsewhere"
        launch.mkdir()
        monkeypatch.setattr(video, "REPO_ROOT", repo)
        monkeypatch.chdir(launch)
        return repo, launch

    def test_relative_srcdir_resolved_against_launch_cwd(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange
        repo, launch = self._setup(tmp_path, monkeypatch)
        # Act
        rc = video.main(["score", "素材目录", "--session", SESSION], relocate=True)
        # Assert：相对 srcdir 按启动目录解析为绝对路径透传；cwd 已切到仓库根
        assert rc == 0
        assert run_recorder[0][0][2] == str((launch / "素材目录").resolve())
        assert pathlib.Path.cwd() == repo

    def test_absolute_srcdir_unchanged(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange
        self._setup(tmp_path, monkeypatch)
        src: pathlib.Path = tmp_path / "绝对素材"
        # Act
        rc = video.main(["score", str(src), "--session", SESSION], relocate=True)
        # Assert：绝对路径原样透传
        assert rc == 0
        assert run_recorder[0][0][2] == str(src)

    def test_relocate_false_keeps_cwd_and_args(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Arrange
        _, launch = self._setup(tmp_path, monkeypatch)
        # Act：relocate=False（测试/库内调用口径）不切目录、不解析路径
        rc = video.main(["score", "素材目录", "--session", SESSION], relocate=False)
        # Assert
        assert rc == 0
        assert run_recorder[0][0][2] == "素材目录"
        assert pathlib.Path.cwd() == launch


class TestClean:
    """clean：清单 + yes 确认 + 守卫（docs/video-clean/spec.md）。"""

    def _tree(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        with_state: bool = True,
        srcdir: str = "",
    ) -> pathlib.Path:
        """造工作区树：output/s1/x.mp4 + work/s1/goals_batch1.json(+state) + 素材目录。

        返回素材目录。
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "output" / "s1").mkdir(parents=True)
        (tmp_path / "output" / "s1" / "x.mp4").write_bytes(b"0")
        sess = tmp_path / "work" / "s1"
        sess.mkdir(parents=True)
        (sess / "goals_batch1.json").write_text("{}", encoding="utf-8")
        src = tmp_path / "素材"
        src.mkdir()
        (src / "a.mp4").write_bytes(b"0")
        if with_state:
            _write_json(sess / "video_cli.json", {"srcdir": srcdir or str(src)})
        return src

    def _yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def test_dry_run_deletes_nothing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        src = self._tree(tmp_path, monkeypatch)
        # Act
        rc = video.main(["clean", "--dry-run"])
        # Assert：零删除
        assert rc == 0
        assert (tmp_path / "output" / "s1" / "x.mp4").is_file()
        assert (tmp_path / "work" / "s1" / "goals_batch1.json").is_file()
        assert (src / "a.mp4").is_file()

    def test_confirm_yes_clears_all(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        src = self._tree(tmp_path, monkeypatch)
        self._yes(monkeypatch)
        # Act
        rc = video.main(["clean"])
        # Assert：output/work 内容清空但目录保留；源视频目录整目录消失
        assert rc == 0
        assert (tmp_path / "output").is_dir() and not list((tmp_path / "output").iterdir())
        assert (tmp_path / "work").is_dir() and not list((tmp_path / "work").iterdir())
        assert not src.exists()

    def test_non_yes_aborts(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange：只输 "y"（非精确 yes）
        src = self._tree(tmp_path, monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        # Act
        rc = video.main(["clean"])
        # Assert：零删除
        assert rc == 0
        assert (src / "a.mp4").is_file()
        assert (tmp_path / "output" / "s1" / "x.mp4").is_file()

    def test_no_state_skips_srcdir(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：无 video_cli.json（无 srcdir 来源）
        src = self._tree(tmp_path, monkeypatch, with_state=False)
        self._yes(monkeypatch)
        caplog.set_level(logging.WARNING)
        # Act
        rc = video.main(["clean"])
        # Assert：output/work 照常清；源视频不动；有 WARNING
        assert rc == 0
        assert not list((tmp_path / "work").iterdir())
        assert (src / "a.mp4").is_file()
        assert "不猜路径" in caplog.text or "srcdir" in caplog.text

    def test_guard_refuses_repo_root(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：srcdir 指向仓库根（恶意/配错）
        self._tree(tmp_path, monkeypatch, srcdir=str(video.REPO_ROOT))
        self._yes(monkeypatch)
        caplog.set_level(logging.WARNING)
        # Act
        rc = video.main(["clean"])
        # Assert：拒删仓库根（它还在），output/work 照常清
        assert rc == 0
        assert video.REPO_ROOT.is_dir()
        assert not list((tmp_path / "output").iterdir())
        assert "守卫拒删" in caplog.text

    def test_readonly_files_removed(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：work/s1 内放只读文件（模拟 .git pack，WinError 5 来源）
        self._tree(tmp_path, monkeypatch)
        ro = tmp_path / "work" / "s1" / "pack.idx"
        ro.write_bytes(b"0")
        ro.chmod(stat.S_IREAD)
        self._yes(monkeypatch)
        # Act
        rc = video.main(["clean"])
        # Assert：只读文件连同父目录一并清掉
        assert rc == 0
        assert not (tmp_path / "work" / "s1").exists()

    def test_non_tty_refuses(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange：非交互 stdin 且非 dry-run
        src = self._tree(tmp_path, monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        # Act
        rc = video.main(["clean"])
        # Assert：退出 1，零删除
        assert rc == 1
        assert (src / "a.mp4").is_file()
        assert (tmp_path / "output" / "s1" / "x.mp4").is_file()


class TestDefaultSession:
    """--session 可省略（docs/default-session/spec.md）：

    score 缺省取素材目录 basename 且成功后写当前场次指针（dry-run 不写）；
    people/build/photo 缺省读指针；显式 --session 永远优先且不改写指针；
    指针缺失/损坏/version 不符/session 空全部显式失败（不猜场次）。
    """

    def _pointer_path(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / "work" / "current_session.json"

    def _write_pointer(self, tmp_path: pathlib.Path, session: str = SESSION) -> None:
        _write_json(
            self._pointer_path(tmp_path),
            {
                "version": 1,
                "session": session,
                "updated_at": "2026-08-29T00:00:00",
                "source": "score",
            },
        )

    def _setup_people_batch(self, session_dir: pathlib.Path) -> pathlib.Path:
        """备好现行布局批次 2 前置产物（同 TestPeople._setup_batch），返回 rawdir。"""
        _write_json(session_dir / "goals_batch2.json", _goals_payload(2))
        _write_json(session_dir / "candidates_batch2.json", [])
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        return rawdir

    # ---- score 缺省场次 ----

    def test_score_default_session_basename(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # Act：不带 --session，场次 = 素材目录 basename
        rc = video.main(["score", "素材目录"])
        # Assert
        assert rc == 0
        assert run_recorder[0][0] == [
            sys.executable,
            str(SCRIPT_DIR / "run_session.py"),
            "素材目录",
            "--session",
            "素材目录",
        ]

    def test_score_writes_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        rc = video.main(["score", "素材目录"])
        assert rc == 0
        pointer = json.loads(self._pointer_path(tmp_path).read_text(encoding="utf-8"))
        assert pointer["version"] == 1
        assert pointer["session"] == "素材目录"

    def test_score_explicit_session_also_writes_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        rc = video.main(["score", "素材目录", "--session", SESSION])
        assert rc == 0
        pointer = json.loads(self._pointer_path(tmp_path).read_text(encoding="utf-8"))
        assert pointer["session"] == SESSION

    def test_score_dry_run_no_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        rc = video.main(["score", "素材目录", "--dry-run"])
        assert rc == 0
        assert not self._pointer_path(tmp_path).exists()

    def test_score_root_srcdir_empty_name(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # 盘符根 basename 为空 → 显式失败（不猜场次），且未发出任何子进程
        rc = video.main(["score", "/"])
        assert rc == 1
        assert run_recorder == []

    # ---- people/build/photo 读指针 ----

    def test_people_reads_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        rawdir = self._setup_people_batch(session_dir)
        self._write_pointer(tmp_path)
        rc = video.main(["people", "--rawdir", str(rawdir)])
        assert rc == 0
        # 确认页步骤带解析出的场次
        page_cmd = run_recorder[-1][0]
        assert page_cmd[page_cmd.index("--session") + 1] == SESSION

    def test_explicit_session_overrides_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        rawdir = self._setup_people_batch(session_dir)
        self._write_pointer(tmp_path, session="other")
        rc = video.main(["people", "--session", SESSION, "--rawdir", str(rawdir)])
        assert rc == 0
        page_cmd = run_recorder[-1][0]
        assert page_cmd[page_cmd.index("--session") + 1] == SESSION
        # 显式覆盖只是临时的：指针不被 people 改写
        pointer = json.loads(self._pointer_path(tmp_path).read_text(encoding="utf-8"))
        assert pointer["session"] == "other"

    def test_build_reads_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        _write_json(session_dir / "goals_batch1.json", _goals_payload())
        _write_json(session_dir / "candidates_batch1.json", [])
        _write_json(session_dir / "session_facts.json", _facts_payload())
        # 自动模式 ④ 的产物由夹具代写（mock 子进程不产真文件，同 TestBuildAuto 口径）
        _write_json(
            session_dir / "auto_roster.json",
            {
                "confirmed": False,
                "players": [{"tag": "A", "name": "", "team": "黑"}],
                "assignments": {format_key("f0.mp4", 0.5): "A"},
            },
        )
        rawdir = session_dir.parent.parent / "raw"
        rawdir.mkdir()
        self._write_pointer(tmp_path)
        rc = video.main(["build", "--rawdir", str(rawdir)])
        assert rc == 0
        assert run_recorder  # 有子进程发出即证明场次解析到了 s1（否则目录不存在会失败）

    def test_photo_apply_reads_pointer(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        self._write_pointer(tmp_path)
        rc = video.main(["photo", "--apply"])
        assert rc == 0
        cmd = run_recorder[0][0]
        assert cmd[cmd.index("--session") + 1] == SESSION

    # ---- 指针异常 ----

    def test_missing_pointer_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.ERROR)
        rc = video.main(["people"])
        assert rc == 1
        assert run_recorder == []
        assert "--session" in caplog.text

    def test_corrupt_pointer_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        p = self._pointer_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{不是合法JSON", encoding="utf-8")
        rc = video.main(["people"])
        assert rc == 1
        assert run_recorder == []

    def test_pointer_bad_version_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        _write_json(self._pointer_path(tmp_path), {"version": 999, "session": SESSION})
        rc = video.main(["people"])
        assert rc == 1
        assert run_recorder == []

    def test_pointer_empty_session_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
        tmp_path: pathlib.Path,
    ) -> None:
        _write_json(self._pointer_path(tmp_path), {"version": 1, "session": ""})
        rc = video.main(["people"])
        assert rc == 1
        assert run_recorder == []

    def test_explicit_empty_session_exit1(
        self,
        session_dir: pathlib.Path,
        run_recorder: list[tuple[list[str], dict[str, str]]],
    ) -> None:
        # --session "" ≠ 未给：显式失败（不猜场次），people/score 一致
        assert video.main(["people", "--session", ""]) == 1
        assert video.main(["score", "素材目录", "--session", ""]) == 1
        assert run_recorder == []
