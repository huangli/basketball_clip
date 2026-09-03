"""pipe_common 原子写 / 校验读单元测试（rules.md §4/§0.2 的关键路径）。

全部在 pytest tmp_path 内进行，不触碰真实 work/。
"""

from __future__ import annotations

import json
import pathlib

import pytest

from errors import SchemaError
from pipe_common import atomic_write_json, read_json


def test_atomic_write_json_roundtrip(tmp_path: pathlib.Path) -> None:
    # Arrange
    target = tmp_path / "c.json"
    payload = {"a": 1, "中文": [1, 2]}
    # Act
    atomic_write_json(target, payload)
    # Assert：内容完整可读，且无 .tmp 残留
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "c.json.tmp").exists()


def test_atomic_write_json_overwrites_existing(tmp_path: pathlib.Path) -> None:
    # Arrange：先写一版
    target = tmp_path / "c.json"
    atomic_write_json(target, {"v": 1})
    # Act：再写覆盖
    atomic_write_json(target, {"v": 2})
    # Assert
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


def test_read_json_valid(tmp_path: pathlib.Path) -> None:
    # Arrange
    target = tmp_path / "ok.json"
    target.write_text('{"goals": []}', encoding="utf-8")
    # Act
    data = read_json(target, what="goals.json")
    # Assert
    assert data == {"goals": []}


def test_read_json_corrupt_raises_schema_error(tmp_path: pathlib.Path) -> None:
    # Arrange：半截文件（崩溃产物）
    target = tmp_path / "bad.json"
    target.write_text('{"goals": [', encoding="utf-8")
    # Act / Assert：数据损坏必须停（SchemaError），而非静默容错
    with pytest.raises(SchemaError, match=r"bad\.json"):
        read_json(target, what="goals.json")
