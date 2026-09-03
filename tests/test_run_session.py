"""run_session 单元测试：切批、产物校验、事实表比对、dry-run 命令清单、失败汇总。

全部用 tmp_path 伪造（空文件不跑真 ffprobe——探测结果经 scan_sources
monkeypatch 注入，见 todo Task 4 设计点），不碰真实素材与 work/。
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pytest

import run_session
from run_session import (
    BatchPlan,
    SourceMeta,
    build_facts,
    build_stage_plan,
    check_fid_coverage,
    compare_facts,
    find_mixed_specs,
    make_batches,
    validate_product,
)


def _meta(
    name: str,
    width: int = 3840,
    height: int = 2160,
    fps: float = 59.94,
    duration: float = 14.0,
) -> SourceMeta:
    """构造一条探测元数据（路径不落地，纯内存对象）。"""
    return SourceMeta(name, pathlib.Path(f"/src/{name}"), width, height, fps, duration)


class TestMakeBatches:
    """切批纯函数：排序、默认 50、边界。"""

    def test_sorted_by_filename(self) -> None:
        # Arrange
        fids = ["dji_003", "dji_001", "dji_002"]
        # Act
        batches = make_batches(fids, 10)
        # Assert
        assert batches == [["dji_001", "dji_002", "dji_003"]]

    def test_default_50_boundary_51_splits_two(self) -> None:
        # Arrange
        fids = [f"f{i:03d}" for i in range(51)]
        # Act
        batches = make_batches(fids, run_session.DEFAULT_BATCH_SIZE)
        # Assert
        assert len(batches) == 2
        assert len(batches[0]) == 50
        assert batches[1] == ["f050"]

    def test_exact_multiple(self) -> None:
        # Arrange
        fids = [f"f{i:03d}" for i in range(100)]
        # Act / Assert
        assert [len(b) for b in make_batches(fids, 50)] == [50, 50]

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            make_batches(["f1"], 0)


class TestValidateProduct:
    """产物校验器：好 JSON 跳过 / 截断 JSON 判坏 / 缺关键字段判坏 / 缺失判坏。"""

    def test_good_candidates(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        p = tmp_path / "candidates.json"
        p.write_text(json.dumps([{"fid": "f1", "t0": 1.0}]), encoding="utf-8")
        # Act / Assert
        assert validate_product(p, "candidates") is True

    def test_truncated_json(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        p = tmp_path / "candidates.json"
        p.write_text('[{"fid": "f1", "t0":', encoding="utf-8")
        # Act / Assert
        assert validate_product(p, "candidates") is False

    def test_missing_file(self, tmp_path: pathlib.Path) -> None:
        assert validate_product(tmp_path / "nope.json", "candidates") is False

    def test_events_missing_key(self, tmp_path: pathlib.Path) -> None:
        # Arrange
        p = tmp_path / "events_index.json"
        p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        # Act / Assert
        assert validate_product(p, "events") is False

    def test_events_good(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "events_index.json"
        p.write_text(json.dumps({"events": []}), encoding="utf-8")
        assert validate_product(p, "events") is True

    def test_session_facts_needs_files_key(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "session_facts.json"
        p.write_text(json.dumps({"file_count": 3}), encoding="utf-8")
        assert validate_product(p, "session_facts") is False

    def test_unknown_kind(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "x.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="未知产物类型"):
            validate_product(p, "bogus")


class TestCompareFacts:
    """事实表比对：一致跳过 / 篡改后不一致 / 增删文件不一致。"""

    def test_consistent(self) -> None:
        # Arrange
        metas = [_meta("a.mp4"), _meta("b.mp4", duration=20.0)]
        # Act / Assert（duration 不参与比对，见 spec：按文件名逐项比分辨率/帧率）
        assert compare_facts(build_facts(metas), metas) == []

    def test_tampered_resolution(self) -> None:
        # Arrange
        saved = build_facts([_meta("a.mp4")])
        saved["files"]["a.mp4"]["width"] = 1920
        # Act
        issues = compare_facts(saved, [_meta("a.mp4")])
        # Assert
        assert len(issues) == 1
        assert "a.mp4" in issues[0] and "width" in issues[0]

    def test_tampered_fps(self) -> None:
        saved = build_facts([_meta("a.mp4")])
        saved["files"]["a.mp4"]["fps"] = 50.0
        assert compare_facts(saved, [_meta("a.mp4")])

    def test_removed_file(self) -> None:
        saved = build_facts([_meta("a.mp4"), _meta("b.mp4")])
        issues = compare_facts(saved, [_meta("a.mp4")])
        assert any("已删除" in line and "b.mp4" in line for line in issues)

    def test_added_file(self) -> None:
        saved = build_facts([_meta("a.mp4")])
        issues = compare_facts(saved, [_meta("a.mp4"), _meta("c.mp4")])
        assert any("新增" in line and "c.mp4" in line for line in issues)


class TestFindMixedSpecs:
    """混合分辨率/帧率探测（spec 阶段①：WARNING 列明细并终止）。"""

    def test_uniform(self) -> None:
        assert find_mixed_specs([_meta("a.mp4"), _meta("b.mp4")]) == []

    def test_mixed_resolution(self) -> None:
        issues = find_mixed_specs([_meta("a.mp4"), _meta("b.mp4", width=1440, height=1080)])
        assert any("混合分辨率" in line for line in issues)

    def test_mixed_fps(self) -> None:
        issues = find_mixed_specs([_meta("a.mp4"), _meta("b.mp4", fps=50.0)])
        assert any("混合帧率" in line for line in issues)


class TestCheckFidCoverage:
    """④ 后核对 candidates 的 fid 覆盖数 == 本批 fid 数。"""

    def test_full_coverage(self) -> None:
        records: list[dict[str, Any]] = [{"fid": "f1"}, {"fid": "f2"}, {"fid": "f1"}]
        assert check_fid_coverage(records, ["f1", "f2"]) == []

    def test_missing_fid(self) -> None:
        assert check_fid_coverage([{"fid": "f1"}], ["f1", "f2"]) == ["f2"]

    def test_empty_records(self) -> None:
        assert check_fid_coverage([], ["f1"]) == ["f1"]

    def test_non_list(self) -> None:
        assert check_fid_coverage({"bad": 1}, ["f1"]) == ["f1"]


class TestPostCheckStage4ZeroCandidate:
    """④ 复核：零候选 fid 区分——缓存有效放行、缓存缺失/帧数不符报错。

    2026-08-13 对手队批次 2/3 实录：3 个短片段算法口径真零候选（缓存正常），
    原口径误判 fid 覆盖不足导致整批失败。
    """

    def _stage4(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        fids: list[str],
        records: list[dict[str, Any]],
    ) -> str | None:
        monkeypatch.chdir(tmp_path)
        plan = build_stage_plan(
            pathlib.Path("素材"),
            pathlib.Path("work/s1"),
            3840,
            2160,
            [fids],
            adhoc=False,
        )[0]
        plan.candidates.parent.mkdir(parents=True)
        plan.candidates.write_text(json.dumps(records), encoding="utf-8")
        cmd = next(c for c in plan.commands if c.stage == 4)
        return run_session._post_check(plan, cmd)

    def _fake_cache(self, fid: str, *, frames_field: int, disk_frames: int) -> None:
        frame_dir = pathlib.Path("work/frames") / fid
        frame_dir.mkdir(parents=True)
        for i in range(disk_frames):
            (frame_dir / f"f_{i:06d}.jpg").write_bytes(b"x")
        cache = pathlib.Path(f"work/detect/{fid}_mot_cache.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"frames": frames_field, "balls": [], "persons": []}))

    def test_legit_zero_candidate_passes(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：f2 无候选记录，但缓存有效（帧数匹配）= 真零候选
        monkeypatch.chdir(tmp_path)
        self._fake_cache("f2", frames_field=2, disk_frames=2)
        # Act / Assert：WARNING 放行，不算失败
        assert self._stage4(tmp_path, monkeypatch, ["f1", "f2"], [{"fid": "f1"}]) is None

    def test_missing_cache_still_fails(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：f2 无候选记录且无缓存 = pilot 产空漏检
        monkeypatch.chdir(tmp_path)
        # Act
        reason = self._stage4(tmp_path, monkeypatch, ["f1", "f2"], [{"fid": "f1"}])
        # Assert
        assert reason is not None and "fid 覆盖不足" in reason and "f2" in reason

    def test_stale_cache_still_fails(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange：缓存 frames 字段与帧目录不符 = 失效
        monkeypatch.chdir(tmp_path)
        self._fake_cache("f2", frames_field=99, disk_frames=2)
        # Act
        reason = self._stage4(tmp_path, monkeypatch, ["f1", "f2"], [{"fid": "f1"}])
        # Assert
        assert reason is not None and "fid 覆盖不足" in reason


class TestBuildStagePlan:
    """dry-run 命令清单：全部显式参数、--keep-clips、--orig 注入、批次划分、adhoc 命名。"""

    def _plans(self, fid_batches: list[list[str]], *, adhoc: bool = False) -> list[BatchPlan]:
        return build_stage_plan(
            pathlib.Path("素材目录"),
            pathlib.Path("work/s1"),
            3840,
            2160,
            fid_batches,
            adhoc=adhoc,
        )

    def test_batch_naming_and_stage_count(self) -> None:
        # Arrange
        fids = [f"dji_{i:03d}" for i in range(51)]
        # Act
        plans = self._plans(make_batches(fids, 50))
        # Assert：51 文件切 2 批，每批 6 条命令（②-⑦；triage 墙 2026-08-11 下线）
        assert [p.label for p in plans] == ["batch1", "batch2"]
        assert all(len(p.commands) == 6 for p in plans)
        assert plans[1].fids == ("dji_050",)
        assert plans[0].candidates == pathlib.Path("work/s1/candidates_batch1.json")
        assert plans[0].hoops == pathlib.Path("work/s1/hoops_batch1.json")
        assert plans[0].review_dir == pathlib.Path("work/s1/review_batch1")

    def test_explicit_args_and_keep_clips(self) -> None:
        # Arrange / Act
        plan = self._plans([["dji_001", "dji_002"]])[0]
        joined = [" ".join(c.argv) for c in plan.commands]
        # Assert：③ mot 显式 fid 位置参数
        assert joined[1].endswith("dji_001 dji_002")
        # ④ pilot 显式 --out
        assert "--out" in joined[2] and "candidates_batch1.json" in joined[2]
        # ⑤ detect_hoops 显式 --candidates/--out
        assert "--candidates" in joined[3] and "--out" in joined[3]
        # ⑥ gen_review_clips 全参数 + --keep-clips + --orig 注入
        stage6 = joined[4]
        assert "--keep-clips" in stage6
        assert "--orig 3840x2160" in stage6
        assert "--srcdir 素材目录" in stage6
        assert "--hoops" in stage6 and "hoops_batch1.json" in stage6
        # ⑦ 标注页显式 --index/--session（triage 墙 2026-08-11 下线，⑦ 仅 label 页）
        assert "--index" in joined[5] and "--session s1" in joined[5]
        assert "gen_label_page.py" in joined[5]
        # ⑦ 批次页带 --batch（label-export-batch：导出即 goals_batchK.json）
        assert "--batch 1" in joined[5]
        # ② 如实标注"全场抽 + 幂等跳过"
        assert "全场抽" in plan.commands[0].note

    def test_adhoc_fixed_naming(self) -> None:
        plans = self._plans([["dji_007"]], adhoc=True)
        assert len(plans) == 1
        assert plans[0].label == "adhoc"
        assert plans[0].candidates == pathlib.Path("work/s1/candidates_adhoc.json")
        assert plans[0].review_dir == pathlib.Path("work/s1/review_adhoc")
        # adhoc ⑦ 不传 --batch（导出维持旧名 goals_<场次>.json）
        assert "--batch" not in " ".join(plans[0].commands[5].argv)


class TestMainDryRun:
    """main 级 dry-run：探测可注入（monkeypatch scan_sources），不执行阶段命令。"""

    def _fake_scan(self, metas: list[SourceMeta], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(run_session, "scan_sources", lambda _p: metas)

    def test_dry_run_plan(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange：51 个伪造素材 + dry-run
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        metas = [_meta(f"dji_{i:04d}.mp4") for i in range(51)]
        self._fake_scan(metas, monkeypatch)
        # Act
        caplog.set_level(logging.INFO)
        rc = run_session.main(["src", "--session", "s1", "--dry-run"])
        # Assert
        assert rc == 0
        text = caplog.text
        assert "共 51 文件，2 批" in text
        assert "--keep-clips" in text and "--orig 3840x2160" in text
        assert not (tmp_path / "work").exists()  # dry-run 不落盘

    def test_mixed_resolution_aborts(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        self._fake_scan([_meta("a.mp4"), _meta("b.mp4", width=1440, height=1080)], monkeypatch)
        # Act / Assert：混合分辨率 -> 非零终止
        assert run_session.main(["src", "--session", "s1", "--dry-run"]) == 2

    def test_tampered_facts_aborts(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange：先落一份事实表再篡改
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        session_dir = tmp_path / "work" / "s1"
        session_dir.mkdir(parents=True)
        facts = build_facts([_meta("a.mp4")])
        facts["files"]["a.mp4"]["width"] = 1920
        (session_dir / "session_facts.json").write_text(json.dumps(facts), encoding="utf-8")
        self._fake_scan([_meta("a.mp4")], monkeypatch)
        # Act / Assert
        assert run_session.main(["src", "--session", "s1", "--dry-run"]) == 2

    def test_fid_coverage_shortage_fails(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange：pilot 产空 candidates（无缓存 fid 只 WARNING 产空的场景）
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        metas = [_meta("dji_0001.mp4"), _meta("dji_0002.mp4")]
        self._fake_scan(metas, monkeypatch)

        def fake_run(argv: tuple[str, ...]) -> None:
            if "pilot_candidates.py" in argv[1]:
                out = pathlib.Path(argv[argv.index("--out") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps([{"fid": "dji_0001"}]), encoding="utf-8")

        monkeypatch.setattr(run_session, "_run_command", fake_run)
        # Act
        rc = run_session.main(["src", "--session", "s1"])
        # Assert：覆盖不足进失败清单，非零退出
        assert rc == 1
