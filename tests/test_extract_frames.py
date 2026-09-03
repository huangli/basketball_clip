"""extract_frames.list_sources 单元测试（原片扫描与排序）。

回归：`from glob import glob` 时 `glob.escape` 不存在（AttributeError），
含空格/中文的素材目录名必须能正常扫描。
"""

from __future__ import annotations

import pathlib

from extract_frames import list_sources


def test_list_sources_finds_mp4_recursively(tmp_path: pathlib.Path) -> None:
    # Arrange：嵌套目录 + 大小写扩展名 + 非 mp4 干扰文件
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "sub" / "a.MP4").write_bytes(b"")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    # Act
    found = list_sources(str(tmp_path))
    # Assert：txt 被过滤，按路径升序
    names = [pathlib.Path(p).name for p in found]
    assert names == ["b.mp4", "a.MP4"]


def test_list_sources_handles_space_chinese_dirname(tmp_path: pathlib.Path) -> None:
    # Arrange：真实素材目录命名（空格 + 中文）
    d = tmp_path / "2026 年 7月22 日 对手"
    d.mkdir()
    (d / "sample_x.mp4").write_bytes(b"")
    # Act
    found = list_sources(str(d))
    # Assert
    assert len(found) == 1
    assert found[0].endswith("sample_x.mp4")


def test_list_sources_empty_dir(tmp_path: pathlib.Path) -> None:
    # Arrange / Act / Assert
    assert list_sources(str(tmp_path)) == []
