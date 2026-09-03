"""geom.Box / iou 单元测试。

结构遵循 AAA（Arrange / Act / Assert），用 fixture（见 conftest.py）解耦数据准备。
"""

from __future__ import annotations

import pathlib

import pytest

from geom import Box, coverage, iou


def test_iou_disjoint_boxes_returns_zero() -> None:
    # Arrange
    a = Box(0, 0, 10, 10)
    b = Box(20, 20, 30, 30)
    # Act
    score = iou(a, b)
    # Assert
    assert score == 0.0


def test_iou_identical_boxes_returns_one() -> None:
    # Arrange
    a = Box(0, 0, 10, 10)
    # Act
    score = iou(a, a)
    # Assert
    assert score == pytest.approx(1.0)


def test_iou_partial_overlap(box_unit: Box) -> None:
    # Arrange: 交集 5x5=25，并集 100+100-25=175
    b = Box(5, 5, 15, 15)
    # Act
    score = iou(box_unit, b)
    # Assert
    assert score == pytest.approx(25 / 175)


def test_box_rejects_degenerate_dimensions() -> None:
    # Arrange / Act / Assert：零面积框应在构造时被拒绝
    with pytest.raises(ValueError):
        Box(10, 10, 0, 0)


def test_box_center_and_area(box_unit: Box) -> None:
    # Arrange / Act / Assert
    assert box_unit.cx == 5
    assert box_unit.cy == 5
    assert box_unit.area == 100


def test_iou_containment_returns_small_over_large() -> None:
    # Arrange: b 完全包含 a，并集即 b 的面积 400
    a = Box(0, 0, 10, 10)
    b = Box(-5, -5, 15, 15)
    # Act
    score = iou(a, b)
    # Assert
    assert score == pytest.approx(100 / 400)


def test_iou_adjacent_boxes_sharing_edge_returns_zero() -> None:
    # Arrange: 共边相邻（inter_x2 == inter_x1），正是 iou 中 <= 判定点
    a = Box(0, 0, 10, 10)
    b = Box(10, 0, 20, 10)
    # Act
    score = iou(a, b)
    # Assert
    assert score == 0.0


def test_iou_negative_coords() -> None:
    # Arrange: 负坐标框（YOLO clipping 前可能出现），交集 25，并集 225+100-25=300
    a = Box(-10, -10, 5, 5)
    b = Box(0, 0, 10, 10)
    # Act
    score = iou(a, b)
    # Assert
    assert score == pytest.approx(25 / 300)


def test_coverage_fully_inside_returns_one() -> None:
    # Arrange: 球框整个落入人框（持球场景，IoU 仅 ~0.01 但覆盖率应为 1）
    ball = Box(4, 4, 8, 8)
    person = Box(0, 0, 30, 50)
    # Act
    score = coverage(ball, person)
    # Assert
    assert score == pytest.approx(1.0)


def test_coverage_half_inside_returns_half() -> None:
    # Arrange: 球框右半落入人框，交集 5x10=50 / 球框 100
    ball = Box(5, 0, 15, 10)
    person = Box(0, 0, 10, 30)
    # Act
    score = coverage(ball, person)
    # Assert
    assert score == pytest.approx(0.5)


def test_coverage_disjoint_returns_zero() -> None:
    # Arrange
    ball = Box(20, 20, 24, 24)
    person = Box(0, 0, 10, 30)
    # Act
    score = coverage(ball, person)
    # Assert
    assert score == 0.0


def test_tmp_work_dir_is_writable(tmp_work_dir: pathlib.Path) -> None:
    # Arrange / Act
    target = tmp_work_dir / "goal.json"
    target.write_text("{}", encoding="utf-8")
    # Assert
    assert target.exists()
