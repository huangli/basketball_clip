"""detect_hoops 纯函数单元测试（选筐/追踪/插值/缓存选路），不跑真模型。

对应规格：多筐选离锚点最近；相邻帧跳变 >150px 截断；缺口 <=3 帧线性插值；
mot 缓存优先 + 旧缓存/损坏回退（docs/detect-hoops-cache/）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import detect_hoops
import mot_candidates as mot
from detect_hoops import interpolate_gaps, load_hoop_frames, select_hoop, track_hoop


def test_select_hoop_picks_nearest_to_anchor() -> None:
    # Arrange：双筐（室内场两端）
    dets = [(800, 100), (100, 100)]
    # Act
    picked = select_hoop(dets, (95, 95))
    # Assert
    assert picked == (100, 100)


def test_select_hoop_empty_returns_none() -> None:
    # Arrange / Act / Assert
    assert select_hoop([], (100, 100)) is None


def test_track_hoop_follows_same_hoop_and_cuts_on_jump() -> None:
    # Arrange：前 4 帧同筐缓动，第 5 帧跳到远处（另一个筐/假筐）
    per_frame = [
        (0.0, [(800, 100), (100, 100)]),
        (0.2, []),
        (0.4, [(810, 100), (105, 103)]),
        (0.6, [(110, 104)]),
        (0.8, [(500, 500)]),
    ]
    # Act
    track = track_hoop(per_frame, anchor=(100, 100), anchor_sec=0.0)
    # Assert：截到跳变前，含起步帧在内 3 个 det 点，且始终在左侧筐
    assert [(p[0], p[1], p[2]) for p in track] == [
        (0.0, 100, 100),
        (0.4, 105, 103),
        (0.6, 110, 104),
    ]
    assert all(p[3] == "det" for p in track)


def test_track_hoop_walks_backward_from_anchor() -> None:
    # Arrange：锚点时刻在中间帧，向两侧都要连上
    per_frame = [
        (0.0, [(98, 100)]),
        (0.2, [(100, 100)]),
        (0.4, [(102, 100)]),
    ]
    # Act
    track = track_hoop(per_frame, anchor=(100, 100), anchor_sec=0.4)
    # Assert：从 0.4 起步，回连到 0.0
    assert [p[0] for p in track] == [0.0, 0.2, 0.4]


def test_track_hoop_no_detection_returns_empty() -> None:
    # Arrange：全窗口无检出
    per_frame = [(0.0, []), (0.2, [])]
    # Act / Assert
    assert track_hoop(per_frame, anchor=(1, 1), anchor_sec=0.0) == []


def test_interpolate_gaps_fills_short_gap() -> None:
    # Arrange：0.0 与 0.4 之间缺 1 帧（0.2）
    track = [(0.0, 100, 100, "det"), (0.4, 110, 104, "det")]
    all_secs = [0.0, 0.2, 0.4, 0.6]
    # Act
    out = interpolate_gaps(track, all_secs)
    # Assert：补 0.2 中点，标 interp
    assert len(out) == 3
    mid = out[1]
    assert mid[0] == 0.2
    assert mid[3] == "interp"
    assert mid[1] == 105
    assert mid[2] == 102


def test_interpolate_gaps_leaves_long_gap_open() -> None:
    # Arrange：缺口 9 帧（>3，云台转向，插值会造幽灵轨迹）
    track = [(0.0, 100, 100, "det"), (2.0, 400, 100, "det")]
    all_secs = [round(0.2 * i, 1) for i in range(11)]
    # Act
    out = interpolate_gaps(track, all_secs)
    # Assert：原样，不补
    assert out == track


# --- mot 缓存优先 + 回退（docs/detect-hoops-cache/ plan Step 3）---

_FID: str = "fid_test"
_ABSENT: Any = object()  # 区分"无 hoops 键"与 hoops=None


@pytest.fixture
def workdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把帧目录/缓存目录指到 tmp_path（glob 只需文件名，帧内容不读）。"""
    monkeypatch.setattr(mot, "FRAMES_PATTERN", str(tmp_path / "frames" / "{}" / "f_*.jpg"))
    monkeypatch.setattr(mot, "CACHE_PATTERN", str(tmp_path / "detect" / "{}_mot_cache.json"))
    return tmp_path


def _mk_frames(root: Path, n: int = 5) -> None:
    d: Path = root / "frames" / _FID
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        (d / f"f_{i:03d}.jpg").write_bytes(b"x")


def _mk_cache(root: Path, hoops: Any = _ABSENT, frames: int = 5) -> None:  # noqa: ANN401 哨兵多态
    d: Path = root / "detect"
    d.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "frames": frames,
        "balls": [[] for _ in range(frames)],
        "persons": [[] for _ in range(frames)],
    }
    if hoops is not _ABSENT:
        payload["hoops"] = hoops
    (d / f"{_FID}_mot_cache.json").write_text(json.dumps(payload), encoding="utf-8")


def _mk_candidates(root: Path) -> tuple[Path, Path]:
    d: Path = root / "S"
    d.mkdir(parents=True, exist_ok=True)
    cand: Path = d / "candidates.json"
    cand.write_text(
        json.dumps([{"fid": _FID, "t0": 0.4, "dur": 0.0, "ac": 0.9, "cx": 100, "cy": 100}]),
        encoding="utf-8",
    )
    return cand, d / "hoops.json"


def test_load_hoop_frames_hit_filters_conf_strictly(workdirs: Path) -> None:
    # Arrange：一帧内混边界 conf（严格大于 0.25 才保留，与 ultralytics NMS 一致）
    _mk_cache(
        workdirs,
        hoops=[
            [
                {"conf": 0.5, "cx": 100, "cy": 100},
                {"conf": 0.25, "cx": 1, "cy": 1},
                {"conf": 0.24, "cx": 2, "cy": 2},
                {"conf": 0.26, "cx": 3, "cy": 3},
                {"conf": 0.15, "cx": 4, "cy": 4},
            ],
            [],
            [],
            [],
            [],
        ],
    )
    # Act
    out = load_hoop_frames(_FID, 5)
    # Assert
    assert out is not None
    assert out[0] == [(100, 100), (3, 3)]
    assert out[1:] == [[], [], [], []]


def test_load_hoop_frames_old_or_corrupt_cache_returns_none(workdirs: Path) -> None:
    # 旧缓存（无 hoops 键）
    _mk_cache(workdirs)
    assert load_hoop_frames(_FID, 5) is None
    # hoops 键类型错
    _mk_cache(workdirs, hoops="x")
    assert load_hoop_frames(_FID, 5) is None
    # hoops 为 list 但元素损坏（缺 cx/cy）
    _mk_cache(workdirs, hoops=[[{"conf": 0.5}], [], [], [], []])
    assert load_hoop_frames(_FID, 5) is None
    # 帧数不符
    _mk_cache(workdirs, hoops=[[], []], frames=2)
    assert load_hoop_frames(_FID, 5) is None


def test_main_cache_hit_never_touches_model(
    workdirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：缓存命中——YOLO 构造与 detect_hoop_frame 都不许被调
    _mk_frames(workdirs)
    _mk_cache(
        workdirs,
        hoops=[
            [{"conf": 0.5, "cx": 100, "cy": 100}],
            [{"conf": 0.5, "cx": 100, "cy": 100}],
            [{"conf": 0.5, "cx": 100, "cy": 100}],
            [],
            [],
        ],
    )
    cand, out = _mk_candidates(workdirs)

    def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 替身签名
        raise AssertionError("缓存命中路径不应加载模型/逐帧检测")

    monkeypatch.setattr(detect_hoops, "YOLO", _boom)
    monkeypatch.setattr(detect_hoops, "detect_hoop_frame", _boom)
    monkeypatch.setattr(
        sys, "argv", ["detect_hoops.py", "--candidates", str(cand), "--out", str(out)]
    )
    # Act
    rc: int = detect_hoops.main()
    # Assert
    assert rc == 0
    events = json.loads(out.read_text(encoding="utf-8"))["events"]
    assert len(events) == 1
    assert events[0]["detected"] is True
    assert events[0]["track"] == [
        [0.0, 100, 100, "det"],
        [0.2, 100, 100, "det"],
        [0.4, 100, 100, "det"],
    ]


def test_main_old_cache_falls_back_to_perframe(
    workdirs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：旧缓存无 hoops 键 → 回退逐帧（模型懒加载出替身，检测函数计数）
    _mk_frames(workdirs)
    _mk_cache(workdirs)  # 无 hoops 键
    cand, out = _mk_candidates(workdirs)
    calls: list[str] = []

    class _DummyModel:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401 替身签名
            pass

    def _fake_detect(model: Any, img_path: str) -> list[tuple[int, int]]:  # noqa: ANN401 替身签名
        calls.append(img_path)
        return [(200, 200)]

    monkeypatch.setattr(detect_hoops, "YOLO", _DummyModel)
    monkeypatch.setattr(detect_hoops, "detect_hoop_frame", _fake_detect)
    monkeypatch.setattr(
        sys, "argv", ["detect_hoops.py", "--candidates", str(cand), "--out", str(out)]
    )
    # Act
    rc: int = detect_hoops.main()
    # Assert：窗口 5 帧全部走逐帧检测，轨迹用回退产物
    assert rc == 0
    assert len(calls) == 5
    events = json.loads(out.read_text(encoding="utf-8"))["events"]
    assert events[0]["detected"] is True
    assert [p[0] for p in events[0]["track"]] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert all(p[1:3] == [200, 200] for p in events[0]["track"])
