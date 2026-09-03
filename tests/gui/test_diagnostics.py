"""诊断日志导出测试：GET /api/diagnostics 打 zip，密钥零泄漏（task-D1）。

work/output 目录用 tmp_path 隔离，不触碰真实工作区；密钥红线测试用
monkeypatch 构造带假 token 的环境，对 zip 全文扫描断言值不出现。
"""

from __future__ import annotations

import io
import os
import pathlib
import re
import zipfile

import pytest
from fastapi.testclient import TestClient

from gui.app import create_app

FAKE_TOKEN: str = "fake-token-D1-密钥红线-9f8e7d6c"  # noqa: S105 测试假 token，非真口令
FILENAME_RE: re.Pattern[str] = re.compile(
    r'attachment; filename="basketball-clip-diagnostics-\d{8}-\d{6}\.zip"'
)


def _make_client(tmp_path: pathlib.Path) -> tuple[TestClient, pathlib.Path]:
    """构造隔离 work/output 的 app + TestClient；返回 (client, work_dir)。"""
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(work_dir=work_dir, output_dir=tmp_path / "output")
    return TestClient(app), work_dir


def _get_zip(client: TestClient) -> zipfile.ZipFile:
    """GET /api/diagnostics 并把响应体解成 ZipFile。"""
    resp = client.get("/api/diagnostics")
    assert resp.status_code == 200
    return zipfile.ZipFile(io.BytesIO(resp.content))


def _zip_full_text(zf: zipfile.ZipFile) -> str:
    """zip 全部条目名 + 全部条目文本拼接（密钥全文扫描用）。"""
    parts: list[str] = list(zf.namelist())
    for name in zf.namelist():
        parts.append(zf.read(name).decode("utf-8"))
    return "\n".join(parts)


def _make_task_files(tasks_dir: pathlib.Path, count: int) -> list[str]:
    """造 count 个任务事件 jsonl，mtime 递增（task_00 最旧）；返回任务 id 列表。"""
    tasks_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for i in range(count):
        task_id = f"task_{i:02d}"
        path = tasks_dir / f"{task_id}.jsonl"
        path.write_text(f'{{"task_id": "{task_id}", "type": "log"}}\n', encoding="utf-8")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))
        ids.append(task_id)
    return ids


# ---- 路由契约（status / headers / 前端 HEAD 探测） ----


def test_route_200_and_zip_headers(tmp_path: pathlib.Path) -> None:
    client, _ = _make_client(tmp_path)
    resp = client.get("/api/diagnostics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert FILENAME_RE.fullmatch(resp.headers["content-disposition"])


def test_head_probe_200(tmp_path: pathlib.Path) -> None:
    """前端 initDiagnostics 用 HEAD 探测激活按钮：HEAD 必须与 GET 同通。"""
    client, _ = _make_client(tmp_path)
    resp = client.head("/api/diagnostics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"


def test_zip_core_entries_exist(tmp_path: pathlib.Path) -> None:
    client, _ = _make_client(tmp_path)
    zf = _get_zip(client)
    names = set(zf.namelist())
    assert {"meta.txt", "environment.txt", "config.txt"} <= names


# ---- 内容关键字段 ----


def test_meta_fields(tmp_path: pathlib.Path) -> None:
    client, _ = _make_client(tmp_path)
    meta = _get_zip(client).read("meta.txt").decode("utf-8")
    assert "0.1.0" in meta  # 应用版本读 pyproject
    assert "Python" in meta
    assert re.search(r"\d{4}-\d{2}-\d{2}", meta)  # 生成时间


def test_environment_fields(tmp_path: pathlib.Path) -> None:
    client, _ = _make_client(tmp_path)
    env = _get_zip(client).read("environment.txt").decode("utf-8")
    assert "ffmpeg" in env
    assert "ffprobe" in env
    for pkg in ("torch", "ultralytics", "opencv-python", "open_clip_torch", "fastapi"):
        assert pkg in env
    assert "models" in env  # 模型文件存在性段落


def test_config_env_vars_boolean_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASKETBALL_CLIP_VLM_TOKEN", raising=False)
    client, _ = _make_client(tmp_path)
    config = _get_zip(client).read("config.txt").decode("utf-8")
    assert "BASKETBALL_CLIP_VLM_TOKEN=未设置" in config


# ---- 任务事件流（上限最近 20 个） ----


def test_tasks_included_with_content(tmp_path: pathlib.Path) -> None:
    client, work_dir = _make_client(tmp_path)
    ids = _make_task_files(work_dir / ".gui" / "tasks", 3)
    zf = _get_zip(client)
    for task_id in ids:
        name = f"tasks/{task_id}.jsonl"
        assert name in zf.namelist()
        assert task_id in zf.read(name).decode("utf-8")


def test_tasks_capped_at_recent_20(tmp_path: pathlib.Path) -> None:
    client, work_dir = _make_client(tmp_path)
    ids = _make_task_files(work_dir / ".gui" / "tasks", 25)
    zf = _get_zip(client)
    task_entries = [n for n in zf.namelist() if n.startswith("tasks/")]
    assert len(task_entries) == 20
    assert f"tasks/{ids[-1]}.jsonl" in task_entries  # 最新保留
    assert f"tasks/{ids[0]}.jsonl" not in task_entries  # 最旧被裁


# ---- 脱敏红线：zip 全文不得出现 token 值 ----


def test_no_secret_value_leak(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASKETBALL_CLIP_VLM_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("BASKETBALL_CLIP_HTTPS_PROXY", "http://secret-proxy:9999")
    client, work_dir = _make_client(tmp_path)
    _make_task_files(work_dir / ".gui" / "tasks", 2)
    zf = _get_zip(client)
    full_text = _zip_full_text(zf)
    assert FAKE_TOKEN not in full_text
    assert "secret-proxy" not in full_text
    config = zf.read("config.txt").decode("utf-8")
    assert "BASKETBALL_CLIP_VLM_TOKEN=已设置" in config
    assert "BASKETBALL_CLIP_HTTPS_PROXY=已设置" in config


# ---- 失败显式报错（不静默） ----


def test_build_failure_returns_500(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> bytes:
        raise OSError("磁盘只读")

    monkeypatch.setattr("gui.app.build_diagnostics_zip", _boom)
    client, _ = _make_client(tmp_path)
    resp = client.get("/api/diagnostics")
    assert resp.status_code == 500
    body = resp.json()
    assert set(body) == {"error"}
    assert "诊断日志" in body["error"]
