"""offline_number.py 单元测试（合成数据，不碰真图真模型）。

覆盖：躯干区裁剪/退化图、连通域切分（排序/嵌套丢弃/双极性）、模板匹配读号
（合成七段数字识别/模糊图 None/超位数 None/空库 None/未知数字低置信）、
模板库自举（对齐规则：连通域数≠位数丢弃记 INFO、每数字样本不足不启用、
低置信忽略、缓存键找不到裁图跳过）、offline_number_cache 读写幂等与版本失效、
跳票并集 K3 优先、apply 主流程（多裁投票/冲突不采纳/裁图缺失与损坏不炸批/
二次运行幂等）、CLI 端到端。

合成数字用七段数码管渲染（PIL 画矩形，无字体依赖）；数字落在躯干区
（垂直 10%~45%、水平中 70%）内。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from crop_scorers import file_md5, save_number_cache
from offline_number import (
    DigitLibrary,
    DigitTemplate,
    _component_img,
    apply_offline_reading,
    binarize,
    bootstrap_library,
    load_offline_cache,
    main,
    merged_number_caches,
    preprocess_crop,
    read_crop_number,
    save_offline_cache,
    segment_digits,
    torso_region,
)

# ---- 合成七段数码管数字（无字体依赖） ----
_SEGMENTS: dict[str, str] = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgecd",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}
_DIGIT_W: int = 40
_DIGIT_H: int = 76
_DIGIT_GAP: int = 24
_IMG_SIZE: tuple[int, int] = (400, 600)


def _draw_digit(draw: ImageDraw.ImageDraw, ch: str, x0: int, y0: int, fill: int) -> None:
    """在 (x0,y0) 处画一个七段数码管数字（段间留 2px 叠合保证 8 连通）。"""
    w, h, t = _DIGIT_W, _DIGIT_H, 10
    rects: dict[str, tuple[int, int, int, int]] = {
        "a": (x0, y0, x0 + w, y0 + t),
        "g": (x0, y0 + (h - t) // 2, x0 + w, y0 + (h + t) // 2),
        "d": (x0, y0 + h - t, x0 + w, y0 + h),
        "f": (x0, y0, x0 + t, y0 + h // 2 + 1),
        "b": (x0 + w - t, y0, x0 + w, y0 + h // 2 + 1),
        "e": (x0, y0 + h // 2 - 1, x0 + t, y0 + h),
        "c": (x0 + w - t, y0 + h // 2 - 1, x0 + w, y0 + h),
    }
    for seg in _SEGMENTS[ch]:
        draw.rectangle(rects[seg], fill=fill)


def _render_crop(
    number: str,
    *,
    dx: int = 0,
    bg: int = 220,
    fg: int = 40,
    size: tuple[int, int] = _IMG_SIZE,
) -> np.ndarray:
    """合成裁图：纯色底 + 躯干区中央的数字串（dx 平移制造 md5 差异）。"""
    im = Image.new("L", size, bg)
    draw = ImageDraw.Draw(im)
    w, h = size
    total: int = len(number) * _DIGIT_W + max(0, len(number) - 1) * _DIGIT_GAP
    x0: int = (w - total) // 2 + dx
    y0: int = int(h * 0.27) - _DIGIT_H // 2
    for i, ch in enumerate(number):
        _draw_digit(draw, ch, x0 + i * (_DIGIT_W + _DIGIT_GAP), y0, fg)
    return np.array(im)


def _save_crop(path: Path, arr: np.ndarray) -> None:
    """保存合成裁图为 JPEG（贴近生产裁图格式）。"""
    Image.fromarray(arr).save(path, quality=95)


def _make_library(digits: str = "12") -> DigitLibrary:
    """手工建库：每个数字 3 个样本（dx 平移），直接构造不走自举。"""
    templates: list[DigitTemplate] = []
    for digit in digits:
        for dx in (-6, 0, 6):
            binary = preprocess_crop(_render_crop(digit, dx=dx))
            assert binary is not None
            comps = segment_digits(binary)
            assert len(comps) == 1, f"digit={digit} dx={dx} comps={len(comps)}"
            templates.append(DigitTemplate(digit=digit, image=_component_img(binary, comps[0])))
    return DigitLibrary(templates=tuple(templates))


def _entry(key: str, crops: list[str], status: str = "OK") -> dict[str, Any]:
    """构造候选记录（含 load_scorer_candidates 校验所需字段）。"""
    return {
        "key": key,
        "file": "DJI_0001.MP4",
        "anchor_time": 60.0,
        "status": status,
        "crop": crops[0] if crops else None,
        "crops": crops,
    }


def _k3_entry(number: str, confidence: str = "high") -> dict[str, Any]:
    """构造 K3 缓存条目（number_cache 值结构）。"""
    return {"number": number, "color": "黑", "name_text": None, "confidence": confidence}


class TestPreprocess:
    """预处理与连通域切分。"""

    def test_torso_region_cut(self) -> None:
        # Arrange（ndarray 形状 = 高×宽）
        img = np.full((600, 400), 200, dtype=np.uint8)
        # Act
        region = torso_region(img)
        # Assert：垂直 10%~45% = 210 行；水平中 70% = 280 列
        assert region.shape == (210, 280)

    def test_preprocess_tiny_image_returns_none(self) -> None:
        # Arrange：图太小，躯干区退化
        img = np.full((20, 20), 200, dtype=np.uint8)
        # Act & Assert
        assert preprocess_crop(img) is None

    def test_segment_sorted_by_cx(self) -> None:
        # Arrange
        binary = preprocess_crop(_render_crop("21"))
        assert binary is not None
        # Act
        comps = segment_digits(binary)
        # Assert：两个数字，按中心 x 升序（左"2"右"1"）
        assert len(comps) == 2
        assert comps[0].cx < comps[1].cx
        assert comps[0].w > comps[1].w  # "2" 比 "1" 宽

    def test_binarize_both_polarities_single_component(self) -> None:
        # 深字浅底与浅字深底（黑/白球衣）应切出同样 1 个连通域（偏差掩码极性一致）
        for bg, fg in ((220, 40), (40, 220)):
            # Arrange
            binary = preprocess_crop(_render_crop("7", bg=bg, fg=fg))
            assert binary is not None
            # Act & Assert
            assert len(segment_digits(binary)) == 1

    def test_drop_nested_artifact(self) -> None:
        # Arrange：闭环数字 "0"（嵌套内沿伪影应被丢弃）
        binary = preprocess_crop(_render_crop("0"))
        assert binary is not None
        # Act & Assert
        assert len(segment_digits(binary)) == 1

    def test_binarize_flat_region_all_background(self) -> None:
        # Arrange：均匀灰图（无数字）
        gray = np.full((300, 300), 30, dtype=np.uint8)
        # Act
        binary = binarize(gray)
        # Assert：平坦区天然全背景（adaptiveThreshold 在此会整片翻成前景）
        assert int(np.count_nonzero(binary)) == 0


class TestReadCropNumber:
    """单张离线读号。"""

    def test_read_synthetic_number_high_conf(self) -> None:
        # Arrange
        library = _make_library("12")
        # Act
        guess = read_crop_number(_render_crop("21", dx=3), library)
        # Assert
        assert guess is not None
        assert guess.number == "21"
        assert guess.confidence == "high"

    def test_blank_returns_none(self) -> None:
        # Arrange：无数字 → 无连通域 → 无法切分
        # Act & Assert
        assert read_crop_number(_render_crop(""), _make_library("12")) is None

    def test_too_many_components_returns_none(self) -> None:
        # Arrange：3 个数字 > MAX_DIGITS=2 → 切分不可信
        # Act & Assert
        assert read_crop_number(_render_crop("217"), _make_library("127")) is None

    def test_empty_library_returns_none(self) -> None:
        # Arrange
        library = DigitLibrary(templates=())
        # Act & Assert
        assert read_crop_number(_render_crop("21"), library) is None

    def test_unknown_digit_low_conf(self) -> None:
        # Arrange：库里只有 1/2，读 "7"（宽高比闸放行 "2"，但形状分不足）
        library = _make_library("12")
        # Act
        guess = read_crop_number(_render_crop("7"), library)
        # Assert：切出连通域但匹配失败 → number=None + low（合法票，不计数）
        assert guess is not None
        assert guess.number is None
        assert guess.confidence == "low"


class TestBootstrap:
    """模板库自举（对齐规则写死）。"""

    def _setup_crops(self, tmp_path: Path, renders: dict[str, np.ndarray]) -> dict[str, str]:
        """落盘裁图并返回 文件名 → md5。"""
        out: dict[str, str] = {}
        for name, arr in renders.items():
            path = tmp_path / name
            _save_crop(path, arr)
            out[name] = file_md5(path)
        return out

    def test_bootstrap_align_and_enable(self, tmp_path: Path) -> None:
        # Arrange：3 张 "21" 高置信样本
        md5s = self._setup_crops(
            tmp_path, {f"s{i}.jpg": _render_crop("21", dx=dx) for i, dx in enumerate((-6, 0, 6))}
        )
        entries = [_entry("k1", list(md5s))]
        k3_cache = {md5: _k3_entry("21") for md5 in md5s.values()}
        # Act
        library = bootstrap_library(entries, tmp_path, k3_cache)
        # Assert：数字 1/2 各 3 样本启用
        assert library.enabled_digits == ("1", "2")
        assert len(library.templates) == 6
        # 自举库能读回新样本
        guess = read_crop_number(_render_crop("12", dx=3), library)
        assert guess is not None and guess.number == "12"

    def test_count_mismatch_discards_with_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：缓存说 "21"（2 位）但裁图渲染 "7"（1 连通域）→ 个数不符丢弃
        md5s = self._setup_crops(tmp_path, {"a.jpg": _render_crop("7")})
        entries = [_entry("k1", list(md5s))]
        k3_cache = {md5s["a.jpg"]: _k3_entry("21")}
        # Act
        with caplog.at_level(logging.INFO):
            library = bootstrap_library(entries, tmp_path, k3_cache)
        # Assert
        assert library.templates == ()
        assert "丢弃样本" in caplog.text

    def test_min_samples_gate(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange：每数字仅 2 样本 < MIN_SAMPLES_PER_DIGIT=3
        md5s = self._setup_crops(
            tmp_path, {f"s{i}.jpg": _render_crop("21", dx=dx) for i, dx in enumerate((-6, 0))}
        )
        entries = [_entry("k1", list(md5s))]
        k3_cache = {md5: _k3_entry("21") for md5 in md5s.values()}
        # Act
        with caplog.at_level(logging.INFO):
            library = bootstrap_library(entries, tmp_path, k3_cache)
        # Assert：样本不足不启用
        assert library.templates == ()
        assert "不启用" in caplog.text

    def test_low_confidence_ignored(self, tmp_path: Path) -> None:
        # Arrange：低置信条目不作样本
        md5s = self._setup_crops(tmp_path, {"a.jpg": _render_crop("7")})
        entries = [_entry("k1", list(md5s))]
        k3_cache = {md5s["a.jpg"]: _k3_entry("7", confidence="low")}
        # Act & Assert
        assert bootstrap_library(entries, tmp_path, k3_cache).templates == ()

    def test_cache_key_without_crop_skips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：缓存键在当前裁图中找不到对应文件
        entries = [_entry("k1", [])]
        k3_cache = {"0" * 32: _k3_entry("21")}
        # Act
        with caplog.at_level(logging.INFO):
            library = bootstrap_library(entries, tmp_path, k3_cache)
        # Assert
        assert library.templates == ()
        assert "找不到对应文件" in caplog.text

    def test_empty_k3_cache_empty_library(self, tmp_path: Path) -> None:
        assert bootstrap_library([_entry("k1", [])], tmp_path, {}).templates == ()


class TestOfflineCache:
    """offline_number_cache 读写与隔离。"""

    def test_roundtrip_and_meta(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "offline_number_cache.json"
        results = {"k": {"number": "21", "color": None, "name_text": None, "confidence": "high"}}
        votes = {"goal1": {"number": "21"}, "goal2": None}
        # Act
        save_offline_cache(path, results, votes)
        # Assert：读回幂等；_meta 标 source=offline
        assert load_offline_cache(path) == results
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["_meta"]["source"] == "offline"
        assert raw["votes"] == votes

    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_offline_cache(tmp_path / "nope.json") == {}

    def test_version_mismatch_returns_empty(self, tmp_path: Path) -> None:
        # Arrange：旧版本缓存作废重开
        path = tmp_path / "offline_number_cache.json"
        path.write_text(
            json.dumps(
                {
                    "_meta": {"source": "offline", "version": "offline-number-v0"},
                    "results": {"k": {}},
                }
            ),
            encoding="utf-8",
        )
        # Act & Assert
        assert load_offline_cache(path) == {}

    def test_never_touches_k3_cache(self, tmp_path: Path) -> None:
        # Arrange：K3 缓存与 offline 缓存并存
        k3_path = tmp_path / "number_cache.json"
        save_number_cache(k3_path, {"k3key": _k3_entry("7")})
        before = k3_path.read_text(encoding="utf-8")
        # Act：offline 写自己的缓存
        save_offline_cache(tmp_path / "offline_number_cache.json", {"okey": _k3_entry("21")}, {})
        # Assert：K3 缓存零改动
        assert k3_path.read_text(encoding="utf-8") == before


class TestMergedNumberCaches:
    """跳票模式并集入口（K3 ∪ offline，K3 优先）。"""

    def test_k3_priority_and_inputs_untouched(self) -> None:
        # Arrange
        k3 = {"a": {"number": "7", "source": "k3"}}
        offline = {"a": {"number": "77", "source": "offline"}, "b": {"number": "21"}}
        # Act
        merged = merged_number_caches(k3, offline)
        # Assert：同键 K3 覆盖 offline；offline 独有键保留；入参不改
        assert merged == {"a": {"number": "7", "source": "k3"}, "b": {"number": "21"}}
        assert k3 == {"a": {"number": "7", "source": "k3"}}
        assert offline == {"a": {"number": "77", "source": "offline"}, "b": {"number": "21"}}


class TestApplyOfflineReading:
    """主流程：逐张读号 + 投票 + 缓存。"""

    def _library_from_bootstrap(self, tmp_path: Path) -> DigitLibrary:
        """经自举建库（走真实对齐路径）：3 张 "21" 样本。"""
        md5s: list[str] = []
        names: list[str] = []
        for i, dx in enumerate((-6, 0, 6)):
            name = f"boot{i}.jpg"
            _save_crop(tmp_path / name, _render_crop("21", dx=dx))
            names.append(name)
            md5s.append(file_md5(tmp_path / name))
        k3_cache = {md5: _k3_entry("21") for md5 in md5s}
        return bootstrap_library([_entry("boot", names)], tmp_path, k3_cache)

    def test_end_to_end_votes_and_cache(self, tmp_path: Path) -> None:
        # Arrange
        library = self._library_from_bootstrap(tmp_path)
        _save_crop(tmp_path / "a.jpg", _render_crop("21", dx=2))
        _save_crop(tmp_path / "b.jpg", _render_crop("21", dx=-2))
        _save_crop(tmp_path / "c.jpg", _render_crop("12"))
        entries = [
            _entry("g1", ["a.jpg", "b.jpg"]),
            _entry("g2", ["c.jpg"]),
            _entry("g3", [], status="SKIP"),
        ]
        cache_path = tmp_path / "offline_number_cache.json"
        # Act
        votes = apply_offline_reading(entries, tmp_path, library, cache_path)
        # Assert：SKIP 球不进 votes；投票正确；缓存结构与 _meta
        assert set(votes) == {"g1", "g2"}
        assert votes["g1"] is not None and votes["g1"]["number"] == "21"
        assert votes["g2"] is not None and votes["g2"]["number"] == "12"
        results = load_offline_cache(cache_path)
        assert set(results) == {file_md5(tmp_path / n) for n in ("a.jpg", "b.jpg", "c.jpg")}
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        assert raw["_meta"]["source"] == "offline"
        assert raw["votes"]["g1"]["number"] == "21"

    def test_conflict_votes_not_adopted(self, tmp_path: Path) -> None:
        # Arrange：两张裁图各读 "21"/"12" 且均 high → 多 high 不采
        library = self._library_from_bootstrap(tmp_path)
        _save_crop(tmp_path / "a.jpg", _render_crop("21"))
        _save_crop(tmp_path / "b.jpg", _render_crop("12"))
        entries = [_entry("g1", ["a.jpg", "b.jpg"])]
        # Act
        votes = apply_offline_reading(
            entries, tmp_path, library, tmp_path / "offline_number_cache.json"
        )
        # Assert
        assert votes["g1"] is not None
        assert votes["g1"]["number"] is None
        assert votes["g1"]["confidence"] == "low"

    def test_missing_crop_skips_without_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：一张裁图缺失，另一张正常
        library = self._library_from_bootstrap(tmp_path)
        _save_crop(tmp_path / "a.jpg", _render_crop("21"))
        entries = [_entry("g1", ["gone.jpg", "a.jpg"])]
        # Act
        with caplog.at_level(logging.INFO):
            votes = apply_offline_reading(
                entries, tmp_path, library, tmp_path / "offline_number_cache.json"
            )
        # Assert：缺失记 INFO 不炸批，其余裁图照常投票
        assert "裁图缺失" in caplog.text
        assert votes["g1"] is not None and votes["g1"]["number"] == "21"

    def test_unreadable_crop_skips_without_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange：坏图（非 JPEG 字节）
        library = self._library_from_bootstrap(tmp_path)
        (tmp_path / "bad.jpg").write_bytes(b"\x00\x01\x02 not a jpeg")
        entries = [_entry("g1", ["bad.jpg"])]
        # Act
        with caplog.at_level(logging.INFO):
            votes = apply_offline_reading(
                entries, tmp_path, library, tmp_path / "offline_number_cache.json"
            )
        # Assert：读取失败记 INFO 跳过；无票 → None
        assert "读取失败" in caplog.text
        assert votes["g1"] is None

    def test_second_run_uses_cache_idempotent(self, tmp_path: Path) -> None:
        # Arrange
        library = self._library_from_bootstrap(tmp_path)
        _save_crop(tmp_path / "a.jpg", _render_crop("21"))
        entries = [_entry("g1", ["a.jpg"])]
        cache_path = tmp_path / "offline_number_cache.json"
        # Act
        votes1 = apply_offline_reading(entries, tmp_path, library, cache_path)
        results1 = load_offline_cache(cache_path)
        votes2 = apply_offline_reading(entries, tmp_path, library, cache_path)
        results2 = load_offline_cache(cache_path)
        # Assert：二次运行 votes/results 完全一致（缓存命中零重算）
        assert votes1 == votes2
        assert results1 == results2

    def test_empty_library_all_skipped(self, tmp_path: Path) -> None:
        # Arrange：空库（未自举）→ 所有裁图无法切分
        _save_crop(tmp_path / "a.jpg", _render_crop("21"))
        entries = [_entry("g1", ["a.jpg"])]
        cache_path = tmp_path / "offline_number_cache.json"
        # Act
        votes = apply_offline_reading(entries, tmp_path, DigitLibrary(templates=()), cache_path)
        # Assert
        assert votes == {"g1": None}
        assert load_offline_cache(cache_path) == {}


class TestCli:
    """CLI 端到端。"""

    def _write_session(self, tmp_path: Path) -> Path:
        """落盘 candidates + 裁图 + K3 缓存，返回 candidates 路径。"""
        names: list[str] = []
        md5s: list[str] = []
        for i, dx in enumerate((-6, 0, 6)):
            name = f"boot{i}.jpg"
            _save_crop(tmp_path / name, _render_crop("21", dx=dx))
            names.append(name)
            md5s.append(file_md5(tmp_path / name))
        _save_crop(tmp_path / "q.jpg", _render_crop("12", dx=1))
        save_number_cache(tmp_path / "number_cache.json", {md5: _k3_entry("21") for md5 in md5s})
        candidates = tmp_path / "scorer_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "session": "test",
                    # 自举样本裁图必须在 candidates entries 里（md5 索引来源）
                    "candidates": [_entry("boot", names), _entry("g1", ["q.jpg"])],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return candidates

    def test_main_end_to_end(self, tmp_path: Path) -> None:
        # Arrange
        candidates = self._write_session(tmp_path)
        # Act
        rc = main(
            [
                "--candidates",
                str(candidates),
                "--bootstrap-from",
                str(tmp_path / "number_cache.json"),
            ]
        )
        # Assert：退出码 0；默认 out 落 candidates 同目录；投票正确
        assert rc == 0
        out = tmp_path / "offline_number_cache.json"
        assert out.is_file()
        raw = json.loads(out.read_text(encoding="utf-8"))
        assert raw["_meta"]["source"] == "offline"
        assert raw["votes"]["g1"]["number"] == "12"

    def test_main_without_bootstrap_skips_all(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        candidates = self._write_session(tmp_path)
        # Act：不给 --bootstrap-from → 空库全部跳过但不炸
        with caplog.at_level(logging.WARNING):
            rc = main(["--candidates", str(candidates)])
        # Assert
        assert rc == 0
        assert "模板库为空" in caplog.text
        raw = json.loads((tmp_path / "offline_number_cache.json").read_text(encoding="utf-8"))
        assert raw["votes"]["g1"] is None

    def test_main_broken_candidates_returns_1(self, tmp_path: Path) -> None:
        # Arrange：candidates schema 损坏（数据损坏必须停）
        candidates = tmp_path / "scorer_candidates.json"
        candidates.write_text(json.dumps({"candidates": "not-a-list"}), encoding="utf-8")
        # Act & Assert
        assert main(["--candidates", str(candidates)]) == 1
