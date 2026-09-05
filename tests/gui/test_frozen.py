"""gui.frozen.reconfigure_stdio_utf8 单元测试：frozen 门控 + reconfigure 参数。

不启动真子进程：monkeypatch sys.frozen 模拟打包态，spy 流验证 reconfigure
被调用且参数为 encoding="utf-8" / errors="replace"；另有真实 TextIOWrapper
假流的端到端用例、非 frozen 不调用的反向用例与不支持 reconfigure 的防御路径。

注意：流的 monkeypatch 必须在**测试体内**做——pytest capture 会在 setup/call
阶段边界重置 sys.stdout/stderr，fixture（setup 阶段）里的替换会被覆盖回
pytest 自己的捕获流，导致断言看不到 reconfigure 效果。
"""

from __future__ import annotations

import io
import logging

import pytest

from gui import frozen


class _SpyStream:
    """最小流替身：只记录 reconfigure 调用参数。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


class TestReconfigureStdioUtf8:
    """frozen 态强制 UTF-8；非 frozen no-op；流不支持时 WARNING 不炸。"""

    def test_frozen_reconfigures_both_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.frozen", True, raising=False)
        out, err = _SpyStream(), _SpyStream()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", err)
        frozen.reconfigure_stdio_utf8()
        for spy in (out, err):
            assert spy.calls == [{"encoding": "utf-8", "errors": "replace"}]

    def test_frozen_real_text_streams_become_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """真实 TextIOWrapper（模拟 frozen 态 cp1252 现场）端到端验证编码切换。"""
        monkeypatch.setattr("sys.frozen", True, raising=False)
        fakes = [
            io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="backslashreplace"),
            io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="backslashreplace"),
        ]
        monkeypatch.setattr("sys.stdout", fakes[0])
        monkeypatch.setattr("sys.stderr", fakes[1])
        frozen.reconfigure_stdio_utf8()
        for stream in fakes:
            assert stream.encoding == "utf-8"
            assert stream.errors == "replace"

    def test_not_frozen_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr("sys.frozen", raising=False)
        out, err = _SpyStream(), _SpyStream()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", err)
        frozen.reconfigure_stdio_utf8()
        assert out.calls == []
        assert err.calls == []

    def test_stream_without_reconfigure_warns_not_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("sys.frozen", True, raising=False)
        monkeypatch.setattr("sys.stdout", io.BytesIO())  # 无 reconfigure 属性
        monkeypatch.setattr("sys.stderr", io.BytesIO())
        with caplog.at_level(logging.WARNING, logger="gui.frozen"):
            frozen.reconfigure_stdio_utf8()  # 不抛
        assert caplog.records
        assert all(r.levelno == logging.WARNING for r in caplog.records)
