"""pytest 共享夹具（见 rules.md §9）。

用 fixture 解耦环境依赖：测试不触碰真实 ``work/`` 与视频文件，
全部在 pytest 提供的临时目录内进行。
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest

from geom import Box


@pytest.fixture
def box_unit() -> Box:
    """单位方框 (0,0)-(10,10)，面积 100。"""
    return Box(0, 0, 10, 10)


@pytest.fixture
def tmp_work_dir(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """隔离的 work 目录（模拟真实 work/，位于 pytest tmp_path 下，测试结束自动回收）。"""
    work = tmp_path / "work"
    work.mkdir()
    yield work
