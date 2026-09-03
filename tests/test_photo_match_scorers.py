"""photo_match_scorers.py 单元测试（照片库扫描/缓存/得分/采纳闸/评估口径/CLI）。

全部纯函数/合成数据：不 import 真模型（open_clip/torch 只在 build_clip_encoder
内部，端到端测试预填全量缓存使惰性工厂永不触发）、不碰网络、不碰真实素材。
覆盖：号码归一化（07→7）、照片库校验（非数字名/空文件夹/全无效显式失败/撞号）、
照片缓存幂等增量与模型前缀隔离、max 得分规则、并列不采纳、单号码库 margin=+inf、
采纳闸边界值、多 cache 并集查询、缓存缺失/前缀 0% 显式报错、真值映射（无号 tag
不可判单列）、入统口径（confirmed ∩ assignments）、正/负样本指标、混淆矩阵、
markdown 报告写 --out、产物 schema 校验。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from cluster_scorers import (
    MODEL_TAG,
    file_md5,
    l2_normalize,
    load_clip_cache,
    merge_candidates,
    save_clip_cache,
)
from errors import BasketballPipelineError, SchemaError
from photo_match_scorers import (
    MARGIN,
    NO_TOP1_LABEL,
    PHOTO_CACHE_NAME,
    THRESHOLD,
    Gallery,
    TopScore,
    Truth,
    _parse_args,
    adopt,
    build_matches_payload,
    classify_truth,
    embed_gallery_photos,
    evaluate,
    load_confirmed_goal_keys,
    load_crop_caches,
    main,
    match_goals,
    normalize_number,
    number_from_tag,
    render_markdown,
    require_model_prefix,
    scan_gallery,
    score_goal,
    top_number,
    validate_matches_payload,
)
from roster import validate_roster


def _unit(*xs: float) -> np.ndarray:
    """构造 L2 归一化向量。"""
    return l2_normalize(np.asarray(xs, dtype=np.float64))


def _write_photos(photos_dir: Path, layout: dict[str, list[str]]) -> Path:
    """落一个合成照片库：layout = 文件夹名 → 照片文件名列表（内容按名区分）。"""
    for folder, names in layout.items():
        d = photos_dir / folder
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            (d / name).write_bytes(f"fake-img-{folder}-{name}".encode())
    return photos_dir


def _cache_key(path: Path, model_tag: str = MODEL_TAG) -> str:
    """缓存键 = ``model_tag:文件 md5``。"""
    return f"{model_tag}:{file_md5(path)}"


def _write_candidates(path: Path, entries: list[dict[str, Any]]) -> Path:
    """落一份 scorer_candidates.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"session": "s", "candidates": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class CountingEncoder:
    """假 encoder：按文件名查合成向量表并计数调用次数（断言缓存幂等用）。"""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def __call__(self, path: Path) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.vectors[path.name], dtype=np.float64)


class TestNormalizeNumber:
    """号码归一化：纯数字去前导零，非纯数字归 None。"""

    def test_leading_zero(self) -> None:
        assert normalize_number("07") == "7"

    def test_plain(self) -> None:
        assert normalize_number("23") == "23"

    def test_zero(self) -> None:
        assert normalize_number("00") == "0"

    def test_non_digit_returns_none(self) -> None:
        assert normalize_number("abc") is None
        assert normalize_number("7a") is None
        assert normalize_number("") is None


class TestScanGallery:
    """照片库扫描校验：合法结构 / 非数字名 / 空文件夹 / 撞号 / 全无效显式失败。"""

    def test_gallery_valid(self, tmp_path: Path) -> None:
        # Arrange
        photos = _write_photos(
            tmp_path / "photos",
            {"07": ["a.jpg", "b.png"], "9": ["c.jpeg", "d.jpg", "e.jpg"]},
        )
        # Act
        gallery = scan_gallery(photos)
        # Assert：07 归一为 7（原名仅展示），照片按文件名排序
        assert sorted(gallery.photos) == ["7", "9"]
        assert [p.name for p in gallery.photos["7"]] == ["a.jpg", "b.png"]
        assert len(gallery.photos["9"]) == 3
        assert gallery.display_names == {"7": "07", "9": "9"}

    def test_gallery_non_digit_folder_skipped(self, tmp_path: Path) -> None:
        photos = _write_photos(tmp_path / "photos", {"新人": ["a.jpg"], "7": ["b.jpg", "c.jpg"]})
        gallery = scan_gallery(photos)
        assert sorted(gallery.photos) == ["7"]

    def test_gallery_empty_folder_skipped(self, tmp_path: Path) -> None:
        photos = tmp_path / "photos"
        (photos / "8").mkdir(parents=True)
        _write_photos(photos, {"7": ["a.jpg", "b.jpg"]})
        gallery = scan_gallery(photos)
        assert sorted(gallery.photos) == ["7"]

    def test_gallery_no_legal_image_skipped(self, tmp_path: Path) -> None:
        photos = _write_photos(tmp_path / "photos", {"7": ["a.jpg", "b.jpg"]})
        d = photos / "8"
        d.mkdir()
        (d / "note.txt").write_text("x", encoding="utf-8")
        gallery = scan_gallery(photos)
        assert sorted(gallery.photos) == ["7"]

    def test_gallery_all_invalid_raises(self, tmp_path: Path) -> None:
        photos = _write_photos(tmp_path / "photos", {"新人": ["a.jpg"]})
        (photos / "8").mkdir()
        with pytest.raises(BasketballPipelineError, match="全部条目无效"):
            scan_gallery(photos)

    def test_gallery_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BasketballPipelineError, match="不存在"):
            scan_gallery(tmp_path / "ghost")

    def test_gallery_duplicate_normalized_raises(self, tmp_path: Path) -> None:
        # Arrange：7 与 07 归一化撞号（归属歧义属配置错误，显式失败不静默合并）
        photos = _write_photos(tmp_path / "photos", {"7": ["a.jpg", "b.jpg"], "07": ["c.jpg"]})
        with pytest.raises(BasketballPipelineError, match="撞号"):
            scan_gallery(photos)

    def test_gallery_single_photo_warns_but_included(self, tmp_path: Path) -> None:
        photos = _write_photos(tmp_path / "photos", {"7": ["only.jpg"]})
        gallery = scan_gallery(photos)
        assert [p.name for p in gallery.photos["7"]] == ["only.jpg"]

    def test_gallery_top_level_files_ignored(self, tmp_path: Path) -> None:
        photos = _write_photos(tmp_path / "photos", {"7": ["a.jpg", "b.jpg"]})
        (photos / "readme.txt").write_text("x", encoding="utf-8")
        (photos / PHOTO_CACHE_NAME).write_text("{}", encoding="utf-8")
        gallery = scan_gallery(photos)
        assert sorted(gallery.photos) == ["7"]


class TestPhotoCache:
    """照片 embedding 缓存：幂等增量、模型前缀隔离、全命中零推理。"""

    def _gallery(self, tmp_path: Path, folders: dict[str, list[str]]) -> Gallery:
        return scan_gallery(_write_photos(tmp_path / "photos", folders))

    def test_photo_cache_second_run_zero_calls(self, tmp_path: Path) -> None:
        # Arrange
        gallery = self._gallery(tmp_path, {"7": ["a.jpg", "b.jpg"]})
        enc1 = CountingEncoder({"a.jpg": [1.0, 0.0], "b.jpg": [0.0, 1.0]})
        cache: dict[str, list[float]] = {}
        # Act：首跑推理 2 次
        vecs1 = embed_gallery_photos(gallery, lambda: enc1, cache, MODEL_TAG)
        # 模拟断点续跑：缓存落盘重读，注入全新计数 encoder
        cache_path = tmp_path / PHOTO_CACHE_NAME
        save_clip_cache(cache_path, MODEL_TAG, cache)
        cache2 = load_clip_cache(cache_path)
        enc2 = CountingEncoder({"a.jpg": [1.0, 0.0], "b.jpg": [0.0, 1.0]})
        vecs2 = embed_gallery_photos(gallery, lambda: enc2, cache2, MODEL_TAG)
        # Assert
        assert enc1.calls == 2
        assert enc2.calls == 0  # 全缓存命中，零推理
        assert len(vecs1["7"]) == 2
        assert all(np.allclose(a, b) for a, b in zip(vecs1["7"], vecs2["7"], strict=True))

    def test_photo_cache_incremental_new_photo_only(self, tmp_path: Path) -> None:
        # Arrange：a.jpg 已在缓存，后补 b.jpg → 只算新图
        gallery = self._gallery(tmp_path, {"7": ["a.jpg", "b.jpg"]})
        cache: dict[str, list[float]] = {
            _cache_key(gallery.photos["7"][0]): [1.0, 0.0],
        }
        enc = CountingEncoder({"b.jpg": [0.0, 1.0]})
        # Act
        vecs = embed_gallery_photos(gallery, lambda: enc, cache, MODEL_TAG)
        # Assert
        assert enc.calls == 1
        assert len(vecs["7"]) == 2
        assert len(cache) == 2

    def test_photo_cache_model_prefix_isolated(self, tmp_path: Path) -> None:
        # Arrange：另一后端前缀的键保留不冲（键前缀天然隔离）
        gallery = self._gallery(tmp_path, {"7": ["a.jpg", "b.jpg"]})
        foreign = "OldModel/x:deadbeef"
        cache: dict[str, list[float]] = {foreign: [9.0, 9.0]}
        enc = CountingEncoder({"a.jpg": [1.0, 0.0], "b.jpg": [0.0, 1.0]})
        # Act
        embed_gallery_photos(gallery, lambda: enc, cache, MODEL_TAG)
        # Assert
        assert cache[foreign] == [9.0, 9.0]
        assert sum(1 for k in cache if k.startswith(f"{MODEL_TAG}:")) == 2

    def test_photo_cache_vanished_photo_skipped(self, tmp_path: Path) -> None:
        # Arrange：扫描后照片消失 → WARNING 跳过；全消失 → 显式报错
        gallery = self._gallery(tmp_path, {"7": ["a.jpg", "b.jpg"]})
        for p in gallery.photos["7"]:
            p.unlink()
        with pytest.raises(BasketballPipelineError, match="无可用 embedding"):
            embed_gallery_photos(gallery, lambda: CountingEncoder({}), {}, MODEL_TAG)


class TestScoring:
    """得分聚合：max(crops × photos) 余弦、并列不采纳、单号码库 margin=+inf。"""

    def test_max_rule_multi_crop_multi_photo(self) -> None:
        # Arrange：7 号照片 (1,0)/(0,1)；9 号照片 (0.6,0.8)；球裁图 (1,0)/(0,1)
        gallery_vecs = {
            "7": (_unit(1.0, 0.0), _unit(0.0, 1.0)),
            "9": (_unit(0.6, 0.8),),
        }
        crops = [_unit(1.0, 0.0), _unit(0.0, 1.0)]
        # Act
        scores = score_goal(crops, gallery_vecs)
        # Assert：7 号 max=1.0（c1·p1 与 c2·p2），9 号 max=0.8
        assert scores["7"] == pytest.approx(1.0)
        assert scores["9"] == pytest.approx(0.8)

    def test_tie_top_not_adopted(self) -> None:
        top = top_number({"7": 0.5, "9": 0.5})
        assert top.number is None
        assert top.score == pytest.approx(0.5)
        assert top.margin == 0.0
        assert not adopt(top, 0.0, 0.0)  # 并列恒不采纳（保守交人裁判）

    def test_single_number_library_margin_inf(self) -> None:
        top = top_number({"7": 0.4})
        assert top.number == "7"
        assert math.isinf(top.margin)
        assert adopt(top, 0.3, 0.02)  # 单号码库过闸只看 threshold

    def test_top_margin_second_best(self) -> None:
        top = top_number({"7": 0.9, "9": 0.6, "11": 0.3})
        assert top.number == "7"
        assert top.margin == pytest.approx(0.3)

    def test_empty_scores_raises(self) -> None:
        with pytest.raises(BasketballPipelineError):
            top_number({})


class TestAdoptGate:
    """采纳闸：score≥THRESHOLD 且 margin≥MARGIN（边界值恰好等于算过）。"""

    def test_exact_boundary_passes(self) -> None:
        assert adopt(TopScore("7", 0.30, 0.02), 0.30, 0.02)

    def test_below_threshold_fails(self) -> None:
        assert not adopt(TopScore("7", 0.2999, 1.0), 0.30, 0.02)

    def test_below_margin_fails(self) -> None:
        assert not adopt(TopScore("7", 0.9, 0.0199), 0.30, 0.02)

    def test_tie_never_adopted(self) -> None:
        assert not adopt(TopScore(None, 0.99, 0.0), 0.30, 0.02)


class TestCropCaches:
    """裁图缓存：多文件并集、缺失显式报错、前缀 0% 显式报错。"""

    def test_missing_cache_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BasketballPipelineError, match="clip_cache 缺失"):
            load_crop_caches([tmp_path / "ghost" / "clip_cache.json"])

    def test_multi_cache_union(self, tmp_path: Path) -> None:
        # Arrange：两批缓存各存一向量
        p1 = tmp_path / "b1" / "clip_cache.json"
        p2 = tmp_path / "b2" / "clip_cache.json"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        save_clip_cache(p1, MODEL_TAG, {f"{MODEL_TAG}:aa": [1.0, 0.0]})
        save_clip_cache(p2, MODEL_TAG, {f"{MODEL_TAG}:bb": [0.0, 1.0]})
        # Act
        merged = load_crop_caches([p1, p2])
        # Assert
        assert set(merged) == {f"{MODEL_TAG}:aa", f"{MODEL_TAG}:bb"}
        assert require_model_prefix(merged, MODEL_TAG) == 2

    def test_prefix_zero_raises(self) -> None:
        with pytest.raises(BasketballPipelineError, match="前缀条目为 0"):
            require_model_prefix({"OtherModel/x:aa": [1.0]}, MODEL_TAG)


class TestMatchGoals:
    """匹配主链：缓存缺失 WARNING 跳过该球不阻塞；SKIP/无裁图不出分。"""

    def _setup(self, tmp_path: Path) -> tuple[dict[str, Any], dict[str, list[float]]]:
        """造 candidates（3 球）+ 只含 k1 裁图的缓存。"""
        d = tmp_path / "scorers"
        d.mkdir()
        (d / "c1.jpg").write_bytes(b"crop-1")
        cand = _write_candidates(
            d / "scorer_candidates.json",
            [
                {"key": "a.mp4#1.0", "status": "OK", "crops": ["c1.jpg"]},
                {"key": "a.mp4#2.0", "status": "OK", "crops": ["ghost.jpg"]},
                {"key": "a.mp4#3.0", "status": "SKIP"},
            ],
        )
        goals = merge_candidates([cand])
        cache = {_cache_key(d / "c1.jpg"): [1.0, 0.0]}
        return goals, cache

    def test_cache_miss_skips_goal_not_blocking(self, tmp_path: Path) -> None:
        # Arrange
        goals, cache = self._setup(tmp_path)
        gallery_vecs = {"7": (_unit(1.0, 0.0),), "9": (_unit(0.0, 1.0),)}
        # Act：k2 裁图文件缺失 → 跳过；k3 SKIP → 不出分
        results, skipped = match_goals(goals, gallery_vecs, cache, MODEL_TAG)
        # Assert
        assert list(results) == ["a.mp4#1.0"]
        assert results["a.mp4#1.0"].number == "7"
        assert skipped == ["a.mp4#2.0"]

    def test_crop_md5_not_in_cache_skips(self, tmp_path: Path) -> None:
        # Arrange：裁图文件存在但 md5 不在缓存
        goals, _ = self._setup(tmp_path)
        gallery_vecs = {"7": (_unit(1.0, 0.0),)}
        # Act
        results, skipped = match_goals(goals, gallery_vecs, {}, MODEL_TAG)
        # Assert
        assert results == {}
        assert skipped == ["a.mp4#1.0", "a.mp4#2.0"]


class TestBuildPayload:
    """photo_matches.json 载荷：仅含过闸球；schema 显式校验；inf margin 可回读。"""

    def test_only_adopted_in_payload(self) -> None:
        results = {
            "k1": TopScore("7", 0.9, 0.5),
            "k2": TopScore("9", 0.1, 0.5),  # 阈值下
            "k3": TopScore(None, 0.9, 0.0),  # 并列
        }
        payload = build_matches_payload(results, 0.30, 0.02, MODEL_TAG)
        assert payload["version"] == "photo-match-v1"
        assert payload["model"] == MODEL_TAG
        assert payload["threshold"] == 0.30
        assert payload["margin"] == 0.02
        assert list(payload["matches"]) == ["k1"]
        assert payload["matches"]["k1"] == {"number": "7", "score": 0.9, "margin": 0.5}

    def test_inf_margin_json_roundtrip(self) -> None:
        # Arrange：单号码库命中 margin=+inf → json 落 Infinity，Python 可回读
        payload = build_matches_payload({"k1": TopScore("7", 0.9, math.inf)}, 0.3, 0.02, MODEL_TAG)
        restored = json.loads(json.dumps(payload))
        entries = validate_matches_payload(restored, "<mem>")
        assert math.isinf(entries["k1"].margin)

    def test_validate_schema_errors(self) -> None:
        with pytest.raises(SchemaError):
            validate_matches_payload([1, 2], "<mem>")
        with pytest.raises(SchemaError, match="version"):
            validate_matches_payload({"version": "nope"}, "<mem>")
        bad_entry = {
            "version": "photo-match-v1",
            "model": MODEL_TAG,
            "threshold": 0.3,
            "margin": 0.02,
            "matches": {"k1": {"number": "", "score": 0.9, "margin": 0.1}},
        }
        with pytest.raises(SchemaError, match="number"):
            validate_matches_payload(bad_entry, "<mem>")

    def test_validate_number_must_be_digits(self) -> None:
        # Arrange：spec 写死"去零数字主键"；消费端拼占位 tag 内联进 <script>，
        # 非数字 number 必须显式拒绝（防坏文件注入）
        for bad in ("7a", "主队7", "7<script>"):
            payload = {
                "version": "photo-match-v1",
                "model": MODEL_TAG,
                "threshold": 0.3,
                "margin": 0.02,
                "matches": {"k1": {"number": bad, "score": 0.9, "margin": 0.1}},
            }
            # Act / Assert
            with pytest.raises(SchemaError, match="number"):
                validate_matches_payload(payload, "<mem>")


class TestTruthMapping:
    """真值映射：tag→号码（无号主队 tag 单列不可判）、未知 tag SchemaError。"""

    def test_number_from_tag(self) -> None:
        assert number_from_tag("主队07") == "7"
        assert number_from_tag("白9") == "9"
        assert number_from_tag("A12B3") == "12"  # 首个数字串
        assert number_from_tag("白色中锋") is None

    def _roster(self) -> Any:  # noqa: ANN401 合成 JSON
        return validate_roster(
            {
                "session": "s",
                "confirmed": True,
                "players": [
                    {"tag": "白7", "name": "", "team": "主队"},
                    {"tag": "白色中锋", "name": "", "team": "主队"},
                    {"tag": "黑3", "name": "", "team": "对手"},
                    {"tag": "红T恤-A", "name": "", "team": "便服"},
                ],
                "assignments": {
                    "a.mp4#1.0": "白7",
                    "a.mp4#2.0": "白色中锋",
                    "a.mp4#3.0": "黑3",
                    "a.mp4#4.0": "红T恤-A",
                },
            },
            "<mem>",
        )

    def test_classify_truth(self) -> None:
        truth = classify_truth(self._roster(), "主队")
        assert truth["a.mp4#1.0"].category == "ours_numbered"
        assert truth["a.mp4#1.0"].number == "7"
        assert truth["a.mp4#2.0"].category == "unjudgeable"  # 无号主队 tag 单列
        assert truth["a.mp4#3.0"].category == "no_number"  # 对方
        assert truth["a.mp4#4.0"].category == "no_number"  # 便服

    def test_assignment_unknown_tag_raises(self) -> None:
        roster = validate_roster(
            {
                "players": [{"tag": "白7", "name": "", "team": "主队"}],
                "assignments": {"a.mp4#1.0": "白9"},
            },
            "<mem>",
        )
        with pytest.raises(SchemaError, match="不存在的 tag"):
            classify_truth(roster, "主队")

    def test_bad_roster_schema_raises(self) -> None:
        with pytest.raises(SchemaError):
            validate_roster({"no_players": True}, "<mem>")


class TestConfirmedGoalKeys:
    """入统键集：只收 confirmed；未知 status WARNING 跳过；坏结构 SchemaError。"""

    def test_only_confirmed(self, tmp_path: Path) -> None:
        path = tmp_path / "goals.json"
        path.write_text(
            json.dumps(
                {
                    "goals": [
                        {"file": "a.mp4", "anchor_time": 1.0, "status": "confirmed"},
                        {"file": "a.mp4", "anchor_time": 2.0, "status": "candidate"},
                        {"file": "a.mp4", "anchor_time": 3.0, "status": "removed"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert load_confirmed_goal_keys(path) == {"a.mp4#1.0"}

    def test_unknown_status_warned_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "goals.json"
        path.write_text(
            json.dumps({"goals": [{"file": "a.mp4", "anchor_time": 1.0, "status": "confirmd"}]}),
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            keys = load_confirmed_goal_keys(path)
        assert keys == set()
        assert any("未知 status" in r.message for r in caplog.records)

    def test_schema_error(self, tmp_path: Path) -> None:
        path = tmp_path / "goals.json"
        path.write_text(json.dumps({"goals": [{"status": "confirmed"}]}), encoding="utf-8")
        with pytest.raises(SchemaError, match="file"):
            load_confirmed_goal_keys(path)


class TestEvaluate:
    """评估口径：入统 = scope_keys；正/负样本指标；不可判不进分母；混淆矩阵。"""

    def _inputs(self) -> tuple[dict[str, TopScore], dict[str, Any], list[str]]:
        results = {
            "k1": TopScore("7", 0.9, 0.5),  # 正样本，过闸且正确
            "k2": TopScore("9", 0.9, 0.5),  # 正样本，过闸但错号
            "k3": TopScore("7", 0.1, 0.5),  # 正样本，未过闸（阈值下）
            "k4": TopScore("7", 0.9, 0.5),  # 负样本，被命中（误）
            "k5": TopScore(None, 0.8, 0.0),  # 负样本，并列未过闸
            # k6 不可判；k7 正样本但未出分（缓存缺失）
        }
        truth = {
            "k1": Truth("ours_numbered", "7"),
            "k2": Truth("ours_numbered", "7"),
            "k3": Truth("ours_numbered", "9"),
            "k4": Truth("no_number", None),
            "k5": Truth("no_number", None),
            "k6": Truth("unjudgeable", None),
            "k7": Truth("ours_numbered", "9"),
        }
        scope = ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]
        return results, truth, scope

    def test_metrics(self) -> None:
        results, truth, scope = self._inputs()
        report = evaluate(results, truth, scope, 0.30, 0.02)
        # 正样本 4（k1,k2,k3,k7），仅 k1 过闸且正确 → 1/4
        assert report.n_ours == 4
        assert report.ours_hit == 1
        assert report.pos_hit_rate == pytest.approx(0.25)
        # 负样本 2（k4,k5），k4 被过闸命中 → 1/2
        assert report.n_no_number == 2
        assert report.no_number_adopted == 1
        assert report.neg_miss_rate == pytest.approx(0.5)
        # 不可判单列不进分母
        assert report.n_unjudgeable == 1
        assert len(report.rows) == 7
        # k7 未出分：不过闸、计正样本 miss
        row7 = next(r for r in report.rows if r.key == "k7")
        assert row7.top_number is None and not row7.adopted and row7.correct is False

    def test_confusion_matrix(self) -> None:
        results, truth, scope = self._inputs()
        report = evaluate(results, truth, scope, 0.30, 0.02)
        # 真值 7：k1→7，k2→9；真值 9：k3→7，k7→(无)
        assert report.confusion == {
            "7": {"7": 1, "9": 1},
            "9": {"7": 1, NO_TOP1_LABEL: 1},
        }

    def test_zero_denominator_rates_none(self) -> None:
        report = evaluate({}, {"k1": Truth("unjudgeable", None)}, ["k1"], 0.3, 0.02)
        assert report.pos_hit_rate is None
        assert report.neg_miss_rate is None

    def test_render_markdown_content(self) -> None:
        results, truth, scope = self._inputs()
        report = evaluate(results, truth, scope, 0.30, 0.02)
        md = render_markdown(report, MODEL_TAG, 0.30, 0.02)
        assert "25.0%" in md  # 正样本命中率
        assert "50.0%" in md  # 负样本误命中率
        assert "混淆矩阵" in md
        assert "| k1 |" in md  # 逐球分布含全部入统球（不过闸）
        assert "| k7 |" in md


class TestParseArgs:
    """CLI 解析：--evaluate 必须同时给 --roster 与 --goals；非 evaluate 拒收。"""

    _BASE: ClassVar[list[str]] = [
        "--photos",
        "photos",
        "--candidates",
        "c.json",
        "--cache",
        "cache.json",
        "--out",
        "o.json",
    ]

    def test_evaluate_requires_roster_and_goals(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--evaluate"])
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--evaluate", "--roster", "r.json"])
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--evaluate", "--goals", "g.json"])

    def test_roster_goals_without_evaluate_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([*self._BASE, "--roster", "r.json"])

    def test_evaluate_full_accepted(self) -> None:
        ns = _parse_args([*self._BASE, "--evaluate", "--roster", "r.json", "--goals", "g.json"])
        assert ns.evaluate
        assert ns.roster == Path("r.json")
        assert ns.goals == Path("g.json")


class TestMainCli:
    """CLI 端到端：全缓存命中不加载模型；产物只含过闸球；失败路径 rc=1。"""

    def _setup(self, tmp_path: Path) -> dict[str, Path]:
        """造照片库（2 号码，全量照片缓存）+ candidates（4 球）+ 裁图缓存。

        球况：k1 vec≈[1,0] 过闸中 7；k2 并列不采纳；k3 margin 不足不过闸；k4 SKIP。
        """
        photos = _write_photos(
            tmp_path / "photos",
            {"07": ["p7a.jpg", "p7b.jpg"], "9": ["p9a.jpg", "p9b.jpg"]},
        )
        photo_cache: dict[str, list[float]] = {}
        for name in ("p7a.jpg", "p7b.jpg"):
            photo_cache[_cache_key(photos / "07" / name)] = [1.0, 0.0]
        for name in ("p9a.jpg", "p9b.jpg"):
            photo_cache[_cache_key(photos / "9" / name)] = [0.0, 1.0]
        save_clip_cache(photos / PHOTO_CACHE_NAME, MODEL_TAG, photo_cache)

        scorers = tmp_path / "scorers"
        scorers.mkdir()
        vecs = {
            "c1.jpg": _unit(1.0, 0.0),
            "c2.jpg": _unit(1.0, 1.0),  # 并列
            "c3.jpg": _unit(1.0, 0.99),  # margin ≈ 0.007 < 0.02
        }
        for name in vecs:
            (scorers / name).write_bytes(f"crop-{name}".encode())
        cand = _write_candidates(
            scorers / "scorer_candidates.json",
            [
                {"key": "a.mp4#1.0", "status": "OK", "crops": ["c1.jpg"]},
                {"key": "a.mp4#2.0", "status": "OK", "crops": ["c2.jpg"]},
                {"key": "a.mp4#3.0", "status": "OK", "crops": ["c3.jpg"]},
                {"key": "a.mp4#4.0", "status": "SKIP"},
            ],
        )
        cache_path = scorers / "clip_cache.json"
        save_clip_cache(
            cache_path,
            MODEL_TAG,
            {_cache_key(scorers / name): [float(x) for x in v] for name, v in vecs.items()},
        )
        return {
            "photos": photos,
            "candidates": cand,
            "cache": cache_path,
            "out": tmp_path / "scorers" / "photo_matches.json",
        }

    def _argv(self, paths: dict[str, Path]) -> list[str]:
        return [
            "--photos",
            str(paths["photos"]),
            "--candidates",
            str(paths["candidates"]),
            "--cache",
            str(paths["cache"]),
            "--out",
            str(paths["out"]),
        ]

    def test_end_to_end_cached(self, tmp_path: Path) -> None:
        # Arrange（照片/裁图缓存全量预填 → 惰性工厂不触发，不 import open_clip/torch）
        paths = self._setup(tmp_path)
        # Act
        rc = main(self._argv(paths))
        # Assert：只 k1 过闸
        assert rc == 0
        payload = json.loads(paths["out"].read_text(encoding="utf-8"))
        entries = validate_matches_payload(payload, str(paths["out"]))
        assert list(entries) == ["a.mp4#1.0"]
        assert entries["a.mp4#1.0"].number == "7"
        assert entries["a.mp4#1.0"].score == pytest.approx(1.0)
        assert payload["threshold"] == THRESHOLD
        assert payload["margin"] == MARGIN

    def test_missing_crop_cache_returns_1(self, tmp_path: Path) -> None:
        paths = self._setup(tmp_path)
        paths["cache"].unlink()
        rc = main(self._argv(paths))
        assert rc == 1

    def test_prefix_zero_returns_1(self, tmp_path: Path) -> None:
        # Arrange：cache 里只有别的后端前缀（cluster 用了别的 --model 的情形）
        paths = self._setup(tmp_path)
        save_clip_cache(
            paths["cache"], "osnet_x1_0/market1501", {"osnet_x1_0/market1501:aa": [1.0]}
        )
        rc = main(self._argv(paths))
        assert rc == 1
        assert not paths["out"].exists()

    def test_schema_error_returns_1(self, tmp_path: Path) -> None:
        paths = self._setup(tmp_path)
        paths["candidates"].write_text(json.dumps({"no_candidates": True}), encoding="utf-8")
        rc = main(self._argv(paths))
        assert rc == 1

    def test_missing_photos_dir_returns_1(self, tmp_path: Path) -> None:
        paths = self._setup(tmp_path)
        rc = main(
            [
                "--photos",
                str(tmp_path / "ghost"),
                "--candidates",
                str(paths["candidates"]),
                "--cache",
                str(paths["cache"]),
                "--out",
                str(paths["out"]),
            ]
        )
        assert rc == 1

    def test_evaluate_end_to_end(self, tmp_path: Path) -> None:
        """--evaluate：markdown 报告写 --out；入统 = confirmed ∩ assignments。

        球况：k1 真值 7 过闸正确；k2 真值 9 过闸正确；k3 不可判（白色中锋）；
        k4 真值无号（黑3/对手）被误中；k5 真值 7 并列未过闸（miss）；
        k6 confirmed 但不在 assignments（剔）；k7 在 assignments 但非 confirmed（剔）。
        期望：正样本 2/3 = 66.7%；负样本 1/1 = 100.0%；不可判 1。
        """
        # Arrange：在 _setup 基础上加 k4-k7 的裁图与候选
        paths = self._setup(tmp_path)
        scorers = paths["candidates"].parent
        extra_vecs = {
            "c4.jpg": _unit(1.0, 0.1),  # top 7 过闸（误命中负样本）
            "c5.jpg": _unit(1.0, 1.0),  # 并列不采纳
        }
        for name in extra_vecs:
            (scorers / name).write_bytes(f"crop-{name}".encode())
        cand = _write_candidates(
            paths["candidates"],
            [
                {"key": "a.mp4#1.0", "status": "OK", "crops": ["c1.jpg"]},
                {"key": "a.mp4#2.0", "status": "OK", "crops": ["c2.jpg"]},
                {"key": "a.mp4#3.0", "status": "OK", "crops": ["c3.jpg"]},
                {"key": "a.mp4#4.0", "status": "OK", "crops": ["c4.jpg"]},
                {"key": "a.mp4#5.0", "status": "OK", "crops": ["c5.jpg"]},
                {"key": "a.mp4#6.0", "status": "OK", "crops": ["c1.jpg"]},
                {"key": "a.mp4#7.0", "status": "OK", "crops": ["c1.jpg"]},
            ],
        )
        cache = load_clip_cache(paths["cache"])
        for name, v in extra_vecs.items():
            cache[_cache_key(scorers / name)] = [float(x) for x in v]
        save_clip_cache(paths["cache"], MODEL_TAG, cache)
        # k2 改成中 9：c2 换向量 [0,1]
        cache[_cache_key(scorers / "c2.jpg")] = [0.0, 1.0]
        save_clip_cache(paths["cache"], MODEL_TAG, cache)

        roster_path = tmp_path / "roster.json"
        roster_path.write_text(
            json.dumps(
                {
                    "session": "s",
                    "confirmed": True,
                    "players": [
                        {"tag": "白7", "name": "", "team": "主队"},
                        {"tag": "白9", "name": "", "team": "主队"},
                        {"tag": "白色中锋", "name": "", "team": "主队"},
                        {"tag": "黑3", "name": "", "team": "对手"},
                    ],
                    "assignments": {
                        "a.mp4#1.0": "白7",
                        "a.mp4#2.0": "白9",
                        "a.mp4#3.0": "白色中锋",
                        "a.mp4#4.0": "黑3",
                        "a.mp4#5.0": "白7",
                        "a.mp4#7.0": "白7",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        goals_path = tmp_path / "goals.json"
        goals_path.write_text(
            json.dumps(
                {
                    "goals": [
                        {"file": "a.mp4", "anchor_time": t, "status": s}
                        for t, s in [
                            (1.0, "confirmed"),
                            (2.0, "confirmed"),
                            (3.0, "confirmed"),
                            (4.0, "confirmed"),
                            (5.0, "confirmed"),
                            (6.0, "confirmed"),
                            (7.0, "candidate"),
                        ]
                    ]
                }
            ),
            encoding="utf-8",
        )
        report_path = tmp_path / "report.md"
        # Act
        rc = main(
            [
                "--photos",
                str(paths["photos"]),
                "--candidates",
                str(cand),
                "--cache",
                str(paths["cache"]),
                "--out",
                str(report_path),
                "--evaluate",
                "--roster",
                str(roster_path),
                "--goals",
                str(goals_path),
            ]
        )
        # Assert
        assert rc == 0
        md = report_path.read_text(encoding="utf-8")
        assert "正样本命中率: 2/3 = 66.7%" in md
        assert "负样本误命中率: 1/1 = 100.0%" in md
        assert "不可判球 1 个" in md
        assert "| a.mp4#6.0 |" not in md  # 不在 assignments → 不入统
        assert "| a.mp4#7.0 |" not in md  # 非 confirmed → 不入统
        assert not paths["out"].exists()  # evaluate 模式不写 photo_matches.json

    def test_evaluate_bad_roster_returns_1(self, tmp_path: Path) -> None:
        paths = self._setup(tmp_path)
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(json.dumps({"bad": True}), encoding="utf-8")
        goals_path = tmp_path / "goals.json"
        goals_path.write_text(json.dumps({"goals": []}), encoding="utf-8")
        rc = main(
            [
                *self._argv(paths),
                "--evaluate",
                "--roster",
                str(roster_path),
                "--goals",
                str(goals_path),
            ]
        )
        assert rc == 1
