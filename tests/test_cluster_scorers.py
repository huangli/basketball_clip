"""cluster_scorers.py 单元测试（embedding 聚类 + 缓存 + 纯度自检 + --model 后端抽象）。

全部纯函数/合成数据：不 import 真模型（open_clip/torch 只出现在 build_clip_encoder
内部，测试注入假 encoder 或 monkeypatch 后端构造函数）、不碰网络、不碰真实素材。
覆盖：candidates schema 校验、多文件合并（并集/同 key 后者覆盖）、crops/crop_scores
回退、embedding 均值+L2 归一、缓存幂等（第二次零推理）、裁图缺失进 unclustered 不炸
整批、凝聚聚类（明显两堆/全相同/全不同/单球）、rep_crops 选取、纯度计算、--model
映射表与解析（非法值拒绝、osnet 已下线显式拒绝）、encoder 工厂分派（查无此模型显式
失败）、build_result model 字段记实际后端、CLI 端到端（CLIP 全缓存命中不加载模型）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from cluster_scorers import (
    MODEL_TAG,
    MODEL_TAGS,
    GoalCrops,
    _parse_args,
    build_encoder,
    build_result,
    cluster_keys,
    embed_goal,
    embed_goals,
    entry_crop_names,
    entry_crop_scores,
    evaluate_purity,
    file_md5,
    l2_normalize,
    load_candidates,
    load_clip_cache,
    main,
    merge_candidates,
    save_clip_cache,
)
from errors import BasketballPipelineError, SchemaError


def _entry(
    key: str,
    status: str = "OK",
    crops: list[str] | None = None,
    crop: str = "",
    crop_scores: list[float] | None = None,
) -> dict[str, Any]:
    """构造一条 candidates 记录（crops=None 表示无多裁字段，模拟旧数据）。"""
    e: dict[str, Any] = {"key": key, "status": status, "crop": crop}
    if crops is not None:
        e["crops"] = crops
    if crop_scores is not None:
        e["crop_scores"] = crop_scores
    return e


def _write_candidates(path: Path, entries: list[dict[str, Any]]) -> Path:
    """落一份 scorer_candidates.json（顶层 {"session","candidates"}）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"session": "20260722", "candidates": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _goal(
    key: str,
    base_dir: Path,
    crops: tuple[str, ...] = ("a.jpg",),
    status: str = "OK",
    crop_scores: tuple[float, ...] = (),
) -> GoalCrops:
    """构造 GoalCrops（不落盘裁图文件）。"""
    return GoalCrops(
        key=key, status=status, crops=crops, crop_scores=crop_scores, base_dir=base_dir
    )


class CountingEncoder:
    """假 encoder：按文件名查合成向量表并计数调用次数（断言缓存幂等用）。"""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def __call__(self, path: Path) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.vectors[path.name], dtype=np.float64)


class TestEntryCropNames:
    """裁图名回退：crops 优先，旧数据无 crops 回退单 crop，均无则空。"""

    def test_crops_preferred(self) -> None:
        entry = _entry("k", crops=["a.jpg", "b.jpg"], crop="a.jpg")
        assert entry_crop_names(entry) == ("a.jpg", "b.jpg")

    def test_fallback_single_crop(self) -> None:
        entry = _entry("k", crops=None, crop="a.jpg")
        assert entry_crop_names(entry) == ("a.jpg",)

    def test_empty_crops_list_falls_back(self) -> None:
        entry = _entry("k", crops=[], crop="a.jpg")
        assert entry_crop_names(entry) == ("a.jpg",)

    def test_no_crop_returns_empty(self) -> None:
        assert entry_crop_names(_entry("k", status="SKIP")) == ()

    def test_crop_scores_parsed(self) -> None:
        entry = _entry("k", crops=["a.jpg"], crop_scores=[0.8, 1])
        assert entry_crop_scores(entry) == (0.8, 1.0)

    def test_crop_scores_missing_returns_empty(self) -> None:
        assert entry_crop_scores(_entry("k")) == ()


class TestLoadCandidates:
    """candidates schema 校验：合法通过，顶层结构坏抛 SchemaError。"""

    def test_valid_passes(self, tmp_path: Path) -> None:
        # Arrange
        path = _write_candidates(
            tmp_path / "scorer_candidates.json",
            [
                _entry("a.mp4#1.0", crops=["a.jpg"], crop_scores=[0.9]),
                _entry("b.mp4#2.0", status="SKIP"),
            ],
        )
        # Act
        goals = load_candidates(path)
        # Assert
        assert list(goals) == ["a.mp4#1.0", "b.mp4#2.0"]
        assert goals["a.mp4#1.0"].crops == ("a.jpg",)
        assert goals["a.mp4#1.0"].base_dir == tmp_path
        assert goals["b.mp4#2.0"].status == "SKIP"

    def test_top_not_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2]), encoding="utf-8")
        with pytest.raises(SchemaError):
            load_candidates(path)

    def test_missing_candidates_list(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"session": "s"}), encoding="utf-8")
        with pytest.raises(SchemaError):
            load_candidates(path)

    def test_entry_not_dict(self, tmp_path: Path) -> None:
        path = _write_candidates(tmp_path / "bad.json", [_entry("k")])
        path.write_text(json.dumps({"session": "s", "candidates": ["x"]}), encoding="utf-8")
        with pytest.raises(SchemaError):
            load_candidates(path)

    def test_entry_missing_key(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"session": "s", "candidates": [{"status": "OK"}]}), encoding="utf-8"
        )
        with pytest.raises(SchemaError):
            load_candidates(path)


class TestMergeCandidates:
    """多文件合并：键集取并集，同 key 后者覆盖前者（且各自 base_dir 独立）。"""

    def test_union_and_override(self, tmp_path: Path) -> None:
        # Arrange：两批文件，k2 同 key，后者裁图不同
        dir1 = tmp_path / "b1"
        dir2 = tmp_path / "b2"
        p1 = _write_candidates(
            dir1 / "scorer_candidates.json",
            [_entry("k1", crops=["a.jpg"]), _entry("k2", crops=["old.jpg"])],
        )
        p2 = _write_candidates(
            dir2 / "scorer_candidates.json",
            [_entry("k2", crops=["new.jpg"]), _entry("k3", crops=["c.jpg"])],
        )
        # Act
        merged = merge_candidates([p1, p2])
        # Assert
        assert list(merged) == ["k1", "k2", "k3"]
        assert merged["k2"].crops == ("new.jpg",)  # 后者覆盖
        assert merged["k1"].base_dir == dir1
        assert merged["k2"].base_dir == dir2  # 覆盖后 base_dir 跟着后者


class TestClipCache:
    """缓存读写：幂等落盘、多后端键前缀共存（旧模型键保留不冲掉）。"""

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "clip_cache.json"
        cache = {f"{MODEL_TAG}:abc": [0.6, 0.8]}
        save_clip_cache(path, MODEL_TAG, cache)
        assert load_clip_cache(path) == cache

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_clip_cache(tmp_path / "none.json") == {}

    def test_other_model_keys_preserved(self, tmp_path: Path) -> None:
        # 另一后端的键保留（互不命中靠键前缀，不回写冲掉）
        path = tmp_path / "clip_cache.json"
        cache = {f"{MODEL_TAG}:a": [1.0], "OldModel/x:b": [2.0]}
        save_clip_cache(path, MODEL_TAG, cache)
        assert load_clip_cache(path) == cache


class TestEmbedGoals:
    """embedding：均值+L2 归一、缓存幂等（第二次零调用）、裁图缺失不炸整批。"""

    def _make_crops(self, dir_path: Path, names: list[str]) -> None:
        for name in names:
            (dir_path / name).write_bytes(f"fake-jpeg-{name}".encode())

    def test_mean_and_normalize(self, tmp_path: Path) -> None:
        # Arrange：两裁图，向量 (3,4)/2 归一后=(0.6,0.8)，(0,5)→(0,1)；均值=(0.3,0.9)
        self._make_crops(tmp_path, ["a.jpg", "b.jpg"])
        goal = _goal("k", tmp_path, crops=("a.jpg", "b.jpg"))
        encoder = CountingEncoder({"a.jpg": [3.0, 4.0], "b.jpg": [0.0, 5.0]})
        cache: dict[str, list[float]] = {}
        # Act
        vec = embed_goal(goal, encoder, cache, MODEL_TAG)
        # Assert
        expected = np.asarray([0.3, 0.9])
        expected = expected / np.linalg.norm(expected)
        assert vec is not None
        assert np.allclose(vec, expected)
        assert np.isclose(np.linalg.norm(vec), 1.0)
        assert encoder.calls == 2
        assert len(cache) == 2  # 缓存键 = model:md5

    def test_cache_idempotent_second_run_zero_calls(self, tmp_path: Path) -> None:
        # Arrange
        self._make_crops(tmp_path, ["a.jpg"])
        goals = {"k": _goal("k", tmp_path)}
        cache: dict[str, list[float]] = {}
        enc1 = CountingEncoder({"a.jpg": [1.0, 0.0]})
        # Act：首跑（推理一次）
        emb1, failed1 = embed_goals(goals, lambda: enc1, cache, MODEL_TAG)
        # 模拟断点续跑：缓存落盘重读，注入全新计数 encoder
        cache_path = tmp_path / "clip_cache.json"
        save_clip_cache(cache_path, MODEL_TAG, cache)
        cache2 = load_clip_cache(cache_path)
        enc2 = CountingEncoder({"a.jpg": [1.0, 0.0]})
        emb2, failed2 = embed_goals(goals, lambda: enc2, cache2, MODEL_TAG)
        # Assert
        assert failed1 == [] and failed2 == []
        assert enc1.calls == 1
        assert enc2.calls == 0  # 全缓存命中，零推理
        assert np.allclose(emb1["k"], emb2["k"])

    def test_missing_crop_file_goes_failed(self, tmp_path: Path) -> None:
        # Arrange：裁图不落盘
        goals = {"k": _goal("k", tmp_path, crops=("ghost.jpg",))}
        encoder = CountingEncoder({})
        # Act
        embeddings, failed = embed_goals(goals, lambda: encoder, {}, MODEL_TAG)
        # Assert：不抛异常，进失败列表（归 unclustered），不推理
        assert embeddings == {}
        assert failed == ["k"]
        assert encoder.calls == 0

    def test_skip_goals_not_embedded(self, tmp_path: Path) -> None:
        self._make_crops(tmp_path, ["a.jpg"])
        goals = {
            "ok": _goal("ok", tmp_path),
            "skip": _goal("skip", tmp_path, status="SKIP", crops=()),
        }
        encoder = CountingEncoder({"a.jpg": [1.0, 0.0]})
        embeddings, failed = embed_goals(goals, lambda: encoder, {}, MODEL_TAG)
        assert list(embeddings) == ["ok"]
        assert failed == []

    def test_zero_vector_raises(self) -> None:
        from errors import BasketballPipelineError

        with pytest.raises(BasketballPipelineError):
            l2_normalize(np.zeros(3))


class TestClusterKeys:
    """凝聚聚类：明显两堆 / 全相同 / 全不同 / 单球 / 空输入。"""

    def test_two_piles(self) -> None:
        # Arrange：e1 附近两个、e2 附近两个（cosine 距离：堆内 ~0.005，堆间 1.0）
        e1 = np.asarray([1.0, 0.0])
        e2 = np.asarray([0.0, 1.0])
        near1 = l2_normalize(np.asarray([1.0, 0.1]))
        near2 = l2_normalize(np.asarray([0.1, 1.0]))
        matrix = np.stack([e1, near1, e2, near2])
        # Act
        clusters = cluster_keys(["a", "b", "c", "d"], matrix, 0.25)
        # Assert
        assert sorted(sorted(c) for c in clusters) == [["a", "b"], ["c", "d"]]

    def test_all_identical_one_cluster(self) -> None:
        matrix = np.stack([np.asarray([1.0, 0.0])] * 3)
        assert cluster_keys(["a", "b", "c"], matrix, 0.25) == [["a", "b", "c"]]

    def test_all_orthogonal_singletons(self) -> None:
        matrix = np.eye(4)
        clusters = cluster_keys(["a", "b", "c", "d"], matrix, 0.25)
        assert sorted(sorted(c) for c in clusters) == [["a"], ["b"], ["c"], ["d"]]

    def test_single_goal(self) -> None:
        matrix = np.asarray([[1.0, 0.0]])
        assert cluster_keys(["only"], matrix, 0.25) == [["only"]]

    def test_empty(self) -> None:
        assert cluster_keys([], np.empty((0, 0)), 0.25) == []

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cluster_keys(["a"], np.empty((0, 2)), 0.25)

    def test_invalid_linkage_raises(self) -> None:
        # Arrange / Act / Assert：非法 linkage 显式失败
        matrix = np.eye(2)
        with pytest.raises(ValueError, match="linkage"):
            cluster_keys(["a", "b"], matrix, 0.25, linkage="ward")

    def test_complete_linkage_two_piles(self) -> None:
        # Arrange：同 test_two_piles 数据，complete linkage 同样应分两堆
        e1 = np.asarray([1.0, 0.0])
        e2 = np.asarray([0.0, 1.0])
        near1 = l2_normalize(np.asarray([1.0, 0.1]))
        near2 = l2_normalize(np.asarray([0.1, 1.0]))
        matrix = np.stack([e1, near1, e2, near2])
        # Act / Assert
        clusters = cluster_keys(["a", "b", "c", "d"], matrix, 0.25, linkage="complete")
        assert sorted(sorted(c) for c in clusters) == [["a", "b"], ["c", "d"]]


class TestBuildResult:
    """scorer_clusters.json 载荷：契约字段、rep_crops 选质量最高球的前 2 张。"""

    def test_rep_crops_and_unclustered(self, tmp_path: Path) -> None:
        # Arrange：簇 [k1, k2]，k2 质量分更高 → rep 取 k2 的 crops 前 2 张
        goals = {
            "k1": _goal("k1", tmp_path, crops=("a.jpg", "a2.jpg"), crop_scores=(0.5, 0.4)),
            "k2": _goal(
                "k2",
                tmp_path,
                crops=("b.jpg", "b2.jpg", "b3.jpg"),
                crop_scores=(0.9, 0.8, 0.7),
            ),
            "k3": _goal("k3", tmp_path, status="SKIP", crops=()),
        }
        embeddings = {"k1": np.asarray([1.0]), "k2": np.asarray([1.0])}
        # Act
        result = build_result(goals, embeddings, [["k1", "k2"]], 0.25, MODEL_TAG)
        # Assert
        assert result["version"] == "cluster-v1"
        assert result["model"] == MODEL_TAG
        assert result["threshold"] == 0.25
        assert result["clusters"] == [
            {"cluster_id": 1, "keys": ["k1", "k2"], "rep_crops": ["b.jpg", "b2.jpg"]}
        ]
        assert result["unclustered"] == ["k3"]

    def test_model_field_records_actual_backend(self, tmp_path: Path) -> None:
        # Arrange：CLIP 后端实跑 → model 字段记 CLIP tag
        goals = {"k1": _goal("k1", tmp_path, crops=("a.jpg",), crop_scores=(0.5,))}
        embeddings = {"k1": np.asarray([1.0])}
        # Act
        result = build_result(goals, embeddings, [["k1"]], 0.25, MODEL_TAG)
        # Assert
        assert result["model"] == MODEL_TAG == "ViT-B-32/laion2b_s34b_b79k"


class TestModelBackend:
    """--model 后端抽象：映射表写死、工厂分派、查无此模型显式失败。

    全程不 import 真 open_clip/torch 模型：分派测试 monkeypatch 后端构造函数。
    """

    def test_model_tags_mapping(self) -> None:
        # spec §数据契约写死：--model 取值 → model_tag 对应表（osnet 已下线，仅 clip）
        assert MODEL_TAGS == {"clip": "ViT-B-32/laion2b_s34b_b79k"}
        assert MODEL_TAGS["clip"] == MODEL_TAG  # CLIP tag 沿用现状常量

    def test_dispatch_clip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        sentinel = CountingEncoder({})
        monkeypatch.setattr("cluster_scorers.build_clip_encoder", lambda: sentinel)
        # Act / Assert：clip 走 CLIP 后端
        assert build_encoder("clip") is sentinel

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(BasketballPipelineError, match="查无此编码器后端"):
            build_encoder("resnet50")


class TestParseArgsModel:
    """--model CLI 解析：默认值、osnet 已下线拒绝、非法值拒绝。"""

    _BASE: ClassVar[list[str]] = ["--candidates", "c.json", "--out", "o.json"]

    def test_default_model_is_clip(self) -> None:
        ns = _parse_args(self._BASE)
        assert ns.model == "clip"

    def test_osnet_model_rejected(self) -> None:
        # osnet 后端已下线：显式取值报清晰错误（SystemExit），不静默回退 clip
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--model", "osnet_x1_0"])

    def test_invalid_model_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--model", "resnet50"])


class TestEvaluatePurity:
    """纯度计算：只统计 assignments 里的键，键数加权，unclustered 单列。"""

    def test_weighted_purity(self) -> None:
        # Arrange：簇1 {k1,k2,k3} 多数 黑(2/3)；簇2 {k4,k5} 全 白(2/2)
        # k6 未入簇但在 assignments；k7 在 assignments 但不在任何簇/unclustered（剔除不计）
        clusters = [["k1", "k2", "k3"], ["k4", "k5"]]
        assignments = {
            "k1": "黑1",
            "k2": "黑1",
            "k3": "白1",
            "k4": "白1",
            "k5": "白1",
            "k6": "黑1",
            "k7": "黑1",
        }
        # Act
        report = evaluate_purity(clusters, ["k6"], assignments)
        # Assert
        assert report.assigned_total == 7
        assert report.assigned_in_clusters == 5
        assert report.unclustered_assigned == 1
        assert report.correct == 4
        assert report.purity == pytest.approx(0.8)
        assert [
            (c.cluster_id, c.size, c.majority_tag, c.majority_count) for c in report.clusters
        ] == [
            (1, 3, "黑1", 2),
            (2, 2, "白1", 2),
        ]

    def test_unassigned_keys_excluded(self) -> None:
        # Arrange：k2 不在 assignments（removed/去重球）→ 簇内只统计 k1
        report = evaluate_purity([["k1", "k2"]], [], {"k1": "黑1"})
        assert report.assigned_in_clusters == 1
        assert report.purity == pytest.approx(1.0)

    def test_no_assigned_keys_purity_zero(self) -> None:
        report = evaluate_purity([["k1"]], [], {"other": "黑1"})
        assert report.purity == 0.0
        assert report.clusters == ()


class TestMainCli:
    """CLI 端到端：全缓存命中时不加载模型（build_clip_encoder 不被调用），
    产物契约可解析；schema 坏数据非零退出。"""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        """造一批 candidates + 裁图 + 全命中缓存，返回 (candidates 路径, out 路径)。"""
        cand_dir = tmp_path / "scorers"
        cand_dir.mkdir(parents=True)
        (cand_dir / "a.jpg").write_bytes(b"fake-a")
        (cand_dir / "b.jpg").write_bytes(b"fake-b")
        cand = _write_candidates(
            cand_dir / "scorer_candidates.json",
            [
                _entry("a.mp4#1.0", crops=["a.jpg"], crop_scores=[0.9]),
                _entry("b.mp4#2.0", crops=["b.jpg"], crop_scores=[0.8]),
                _entry("c.mp4#3.0", status="SKIP"),
            ],
        )
        # 预填缓存：键 = model:md5，两球向量相同（应聚成一簇）
        cache = {
            f"{MODEL_TAG}:{file_md5(cand_dir / 'a.jpg')}": [1.0, 0.0],
            f"{MODEL_TAG}:{file_md5(cand_dir / 'b.jpg')}": [1.0, 0.0],
        }
        out = tmp_path / "scorer_clusters.json"
        save_clip_cache(out.parent / "clip_cache.json", MODEL_TAG, cache)
        return cand, out

    def test_end_to_end_cached(self, tmp_path: Path) -> None:
        # Arrange
        cand, out = self._setup(tmp_path)
        # Act（全缓存命中 → 惰性工厂不触发，不 import open_clip/torch）
        rc = main(["--candidates", str(cand), "--out", str(out)])
        # Assert
        assert rc == 0
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["version"] == "cluster-v1"
        assert result["clusters"] == [
            {"cluster_id": 1, "keys": ["a.mp4#1.0", "b.mp4#2.0"], "rep_crops": ["a.jpg"]}
        ]
        assert result["unclustered"] == ["c.mp4#3.0"]

    def test_out_parent_missing_auto_created(self, tmp_path: Path) -> None:
        """--out 父目录不存在时自建（回归：clip_cache 落盘早于原 mkdir 位置曾 FileNotFoundError）。

        全 SKIP 候选 → 零 embedding（不加载模型），save_clip_cache 即触发父目录写入。
        """
        # Arrange
        cand = _write_candidates(
            tmp_path / "scorer_candidates.json",
            [_entry("a.mp4#1.0", status="SKIP")],
        )
        out = tmp_path / "newdir" / "scorer_clusters.json"
        # Act
        rc = main(["--candidates", str(cand), "--out", str(out)])
        # Assert
        assert rc == 0
        assert json.loads(out.read_text(encoding="utf-8"))["unclustered"] == ["a.mp4#1.0"]
        assert (out.parent / "clip_cache.json").is_file()

    def test_evaluate_prints_report(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange
        cand, out = self._setup(tmp_path)
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(
            json.dumps(
                {
                    "session": "s",
                    "confirmed": True,
                    "players": [{"tag": "黑1", "name": "", "team": "对手"}],
                    "assignments": {"a.mp4#1.0": "黑1", "b.mp4#2.0": "黑1"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # Act
        with caplog.at_level("INFO"):
            rc = main(
                [
                    "--candidates",
                    str(cand),
                    "--out",
                    str(out),
                    "--evaluate",
                    "--roster",
                    str(roster_path),
                ]
            )
        # Assert
        assert rc == 0
        assert any("整体纯度" in r.message and "100.0%" in r.message for r in caplog.records)

    def test_schema_error_returns_1(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"no_candidates": True}), encoding="utf-8")
        rc = main(["--candidates", str(bad), "--out", str(tmp_path / "out.json")])
        assert rc == 1

    def test_evaluate_requires_roster(self, tmp_path: Path) -> None:
        cand, out = self._setup(tmp_path)
        with pytest.raises(SystemExit):
            main(["--candidates", str(cand), "--out", str(out), "--evaluate"])

    def test_threshold_out_of_range_rejected(self, tmp_path: Path) -> None:
        cand, out = self._setup(tmp_path)
        with pytest.raises(SystemExit):
            main(["--candidates", str(cand), "--out", str(out), "--threshold", "3.0"])

    def test_file_md5_stable(self, tmp_path: Path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        assert file_md5(p) == hashlib.md5(b"hello", usedforsecurity=False).hexdigest()
