"""model_fetch 首运行自动下载单元测试（rules.md §4 关键路径：重试/原子写/显式失败）。

全部 mock 网络层（_urlopen），不发起真实下载；尺寸下限经 monkeypatch 缩小，
避免在 tmp_path 写百 MB 级文件。
"""

from __future__ import annotations

import io
import pathlib
import urllib.error

import pytest

import model_fetch as mf
from errors import ModelDownloadError

_VALID = b"PK" + b"\x00" * 64  # zip 魔数 + 填充（MIN_BYTES 在测试中被缩到 4）


class _FakeResp:
    """模拟 urllib 响应：headers.get + 分块 read + 上下文管理。"""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """缩小尺寸门槛、去掉退避等待，全部用例共享。"""
    monkeypatch.setattr(mf, "MODEL_MIN_BYTES", 4)
    monkeypatch.setattr(mf.time, "sleep", lambda _s: None)


def test_existing_valid_model_short_circuits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：本地已有合法权重
    target = tmp_path / "abdullahtarek_ball.pt"
    target.write_bytes(_VALID)

    def _forbidden(_request: object) -> _FakeResp:
        raise AssertionError("不应发起网络请求")

    monkeypatch.setattr(mf, "_urlopen", _forbidden)
    # Act
    result = mf.ensure_ball_model(target)
    # Assert
    assert result == target


def test_download_success_atomic(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    target = tmp_path / "models" / "abdullahtarek_ball.pt"
    monkeypatch.setattr(mf, "_urlopen", lambda _req: _FakeResp(_VALID))
    # Act
    result = mf.ensure_ball_model(target)
    # Assert：内容完整、原子落盘（无 .part 残留）、父目录自动创建
    assert result == target
    assert target.read_bytes() == _VALID
    assert not target.with_suffix(".pt.part").exists()


def test_retry_then_success(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange：第一次网络失败，第二次成功
    target = tmp_path / "abdullahtarek_ball.pt"
    calls = {"n": 0}

    def _flaky(_req: object) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeResp(_VALID)

    monkeypatch.setattr(mf, "_urlopen", _flaky)
    # Act
    result = mf.ensure_ball_model(target)
    # Assert
    assert result == target
    assert target.read_bytes() == _VALID
    assert calls["n"] == 2


def test_retries_exhausted_raises_with_manual_url(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：持续网络失败
    target = tmp_path / "abdullahtarek_ball.pt"

    def _always_fail(_req: object) -> _FakeResp:
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(mf, "_urlopen", _always_fail)
    # Act / Assert：报错信息含手动下载 URL 与代理提示，且无残留文件
    with pytest.raises(ModelDownloadError, match=r"drive\.google\.com") as exc_info:
        mf.ensure_ball_model(target)
    assert "BASKETBALL_CLIP_HTTPS_PROXY" in str(exc_info.value)
    assert not target.exists()
    assert not target.with_suffix(".pt.part").exists()


def test_html_interstitial_rejected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：来源返回 HTML 拦截页（非 zip 魔数）——链接失效/被墙的防护
    target = tmp_path / "abdullahtarek_ball.pt"
    html = b"<html><body>Google Drive - Virus scan warning</body></html>"
    monkeypatch.setattr(mf, "_urlopen", lambda _req: _FakeResp(html))
    # Act / Assert：显式报错给手动 URL，不留损坏文件
    with pytest.raises(ModelDownloadError, match=r"drive\.google\.com"):
        mf.ensure_ball_model(target)
    assert not target.exists()
    assert not target.with_suffix(".pt.part").exists()


def test_corrupt_existing_file_triggers_redownload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange：半截损坏的既有文件
    target = tmp_path / "abdullahtarek_ball.pt"
    target.write_bytes(b"PK\x00")  # 尺寸不达标
    monkeypatch.setattr(mf, "_urlopen", lambda _req: _FakeResp(_VALID))
    # Act
    result = mf.ensure_ball_model(target)
    # Assert：坏文件被替换为新下载内容
    assert result == target
    assert target.read_bytes() == _VALID
