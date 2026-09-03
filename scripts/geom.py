"""几何工具：轴对齐边界框、交并比与覆盖率计算。

输入：YOLO 检测输出的 xyxy 像素坐标。
输出：Box 对象及其 IoU / 覆盖率。
依赖：仅标准库（dataclasses）。
典型调用：``from geom import Box, coverage, iou``
（scripts/ 已通过 pyproject.toml 加入 pythonpath）。

说明：本模块是 rules.md「正确写法」的标杆样本，对照
``scripts/batch_detect_v2.py`` 中无类型/``;``串/魔法数字的反例。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """轴对齐边界框（像素坐标，左上角 / 右下角）。"""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"非法框 {self}：需 x2 > x1 且 y2 > y1")

    @property
    def cx(self) -> int:
        """中心 X 坐标（像素）。"""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """中心 Y 坐标（像素）。"""
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        """面积（平方像素）。"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    """计算两个边界框的交并比（Intersection over Union）。

    Args:
        a: 框 A。
        b: 框 B。

    Returns:
        IoU，取值 ``[0.0, 1.0]``；两框无相交返回 ``0.0``。
    """
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a.area + b.area - intersection
    return intersection / union


def coverage(inner: Box, outer: Box) -> float:
    """计算 inner 框落入 outer 框内的面积占比（交集面积 / inner 面积）。

    与对称的 IoU 不同，覆盖率刻画"小框是否落在大框内"：球框面积远小于
    人框时 IoU 上限极低（实测持球 IoU ≈0.007），覆盖率仍可接近 1.0，
    持球排除判据须用它（见 mot_candidates.HELD_COVERAGE）。

    Args:
        inner: 内框（被覆盖方，如球框）。
        outer: 外框（覆盖方，如人框）。

    Returns:
        覆盖率，取值 ``[0.0, 1.0]``；两框无相交返回 ``0.0``。
    """
    inter_x1 = max(inner.x1, outer.x1)
    inter_y1 = max(inner.y1, outer.y1)
    inter_x2 = min(inner.x2, outer.x2)
    inter_y2 = min(inner.y2, outer.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    return intersection / inner.area
