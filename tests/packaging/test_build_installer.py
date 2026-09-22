"""build_installer.py 用户数据备份/恢复与并发锁单元测试。

不跑真实 PyInstaller，仅在临时目录构造假 dist 结构，验证：
1. 正常流程：work/output 被备份，dist 重建后移回，空备份目录删除。
2. PyInstaller 失败场景：dist 被删后未重建，备份仍能移回。
3. 防御行为：目标已存在同名目录时报错不覆盖。
4. 备份目录创建失败时以 BuildError 友好提示，而非裸 PermissionError。
5. 恢复中途失败后 finally 兜底恢复剩余目录，且不掩盖原异常。
6. 并发锁：活锁拒绝、陈旧锁覆盖、异常时锁释放。
7. 备份目录位于 packaging/build/ 下，不再与 dist 同根。
8. 备份目录丢失时 restore 显式抛 BuildError，不静默放行。
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
from types import ModuleType

import pytest


def _load_build_installer() -> ModuleType:
    """通过 importlib 加载 packaging/build_installer.py（该目录无 __init__.py）。"""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    path = repo_root / "packaging" / "build_installer.py"
    spec = importlib.util.spec_from_file_location("build_installer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bi() -> ModuleType:
    """加载 build_installer 模块。"""
    return _load_build_installer()


def _write_file(path: pathlib.Path, content: bytes = b"data") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_fake_dist(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """构造 dist/basketball-clip/{work,output}/<场次>/文件。"""
    dist_dir = tmp_path / "dist"
    app_dir = dist_dir / "basketball-clip"
    _write_file(app_dir / "work" / "20260913" / "goals.json", b'{"goals":[]}')
    _write_file(app_dir / "work" / "20260913" / "clip.mp4", b"fake video")
    _write_file(app_dir / "output" / "20260913" / "highlight.mp4", b"fake highlight")
    return dist_dir, app_dir


def _backup_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """测试用的备份根目录（模拟 packaging/build/）。"""
    return tmp_path / "packaging" / "build"


def test_backup_and_restore_on_success_mimics_pyinstaller_rebuild(
    bi: ModuleType, tmp_path: pathlib.Path
) -> None:
    """模拟 PyInstaller 删 dist 后重建，用户数据仍能完整恢复。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is not None
    assert backup_dir.name.startswith("_userdata_backup_")
    assert not (app_dir / "work").exists()
    assert not (app_dir / "output").exists()
    assert (backup_dir / "work" / "20260913" / "goals.json").is_file()
    assert (backup_dir / "output" / "20260913" / "highlight.mp4").is_file()

    # 模拟 PyInstaller 删除并重建 dist/basketball-clip
    shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)
    _write_file(app_dir / "basketball-clip.exe", b"exe")

    bi._restore_user_data(app_dir, backup_dir)

    assert (app_dir / "work" / "20260913" / "goals.json").is_file()
    assert (app_dir / "work" / "20260913" / "clip.mp4").is_file()
    assert (app_dir / "output" / "20260913" / "highlight.mp4").is_file()
    assert not backup_dir.exists()


def test_restore_when_app_dir_missing_after_failure(bi: ModuleType, tmp_path: pathlib.Path) -> None:
    """PyInstaller 失败后 dist/basketball-clip 不存在，备份仍需能回到原位。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is not None

    # 模拟 PyInstaller 失败：整个 app_dir 被删除且未重建
    shutil.rmtree(app_dir)
    assert not app_dir.exists()

    bi._restore_user_data(app_dir, backup_dir)

    assert app_dir.is_dir()
    assert (app_dir / "work" / "20260913" / "goals.json").is_file()
    assert (app_dir / "output" / "20260913" / "highlight.mp4").is_file()
    assert not backup_dir.exists()


def test_restore_refuses_to_overwrite_existing_data_dir(
    bi: ModuleType, tmp_path: pathlib.Path
) -> None:
    """若新包意外已有同名 work/output 目录，恢复时报错不覆盖。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is not None

    # 新包意外保留了空 work 目录
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "work").mkdir()

    with pytest.raises(bi.BuildError, match="恢复用户数据冲突"):
        bi._restore_user_data(app_dir, backup_dir)

    # 备份仍保留，未丢失
    assert (backup_dir / "work" / "20260913" / "goals.json").is_file()


def test_backup_returns_none_when_no_user_data(bi: ModuleType, tmp_path: pathlib.Path) -> None:
    """没有 work/output 时不创建备份目录。"""
    dist_dir = tmp_path / "dist"
    app_dir = dist_dir / "basketball-clip"
    app_dir.mkdir(parents=True)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is None
    assert not list(backup_root.glob("_userdata_backup_*"))


def test_backup_rolls_back_partial_moves_on_failure(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """备份中途失败时，已移动目录应回滚到原位，不留半截备份。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    real_move = bi.shutil.move
    call_count = 0

    def fake_move(src: str, dst: str, **kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise PermissionError("模拟第二次移动失败")
        return real_move(src, dst, **kwargs)

    monkeypatch.setattr(bi.shutil, "move", fake_move)

    with pytest.raises(bi.BuildError, match="备份用户数据失败"):
        bi._backup_user_data(app_dir, backup_root)

    # 第一个被移动的目录已回滚到原位
    assert (app_dir / "work" / "20260913" / "goals.json").is_file()
    # 第二个目录未被移动，仍在原位
    assert (app_dir / "output" / "20260913" / "highlight.mp4").is_file()
    # 不应留下残留备份目录
    assert not any(p.name.startswith("_userdata_backup_") for p in backup_root.iterdir())


def test_backup_mkdir_failure_raises_build_error(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """备份目录创建失败时应抛出 BuildError（中文信息含路径），而非裸 PermissionError。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    def raise_permission_error(*args: object, **kwargs: object) -> None:
        raise PermissionError("模拟 mkdir 权限失败")

    monkeypatch.setattr(pathlib.Path, "mkdir", raise_permission_error)

    with pytest.raises(bi.BuildError, match="无法创建备份目录"):
        bi._backup_user_data(app_dir, backup_root)


def test_restore_mid_failure_then_failsafe_recovers_remaining(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟 work 恢复成功、output 恢复失败，finally 兜底应移回剩余目录且不掩盖原异常。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is not None

    # 模拟 PyInstaller 删除并重建 dist/basketball-clip
    shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)

    real_move = bi.shutil.move
    call_count = 0

    def fake_move(src: str, dst: str, **kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise PermissionError("模拟 output 移回失败")
        return real_move(src, dst, **kwargs)

    monkeypatch.setattr(bi.shutil, "move", fake_move)

    original_exc: Exception | None = None
    try:
        bi._restore_user_data(app_dir, backup_dir)
    except Exception as e:
        original_exc = e

    assert isinstance(original_exc, bi.BuildError)
    assert "output 移回失败" in str(original_exc)

    # 此时 work 已恢复，output 仍在备份目录
    assert (app_dir / "work" / "20260913" / "goals.json").is_file()
    assert (backup_dir / "output" / "20260913" / "highlight.mp4").is_file()
    assert not (app_dir / "output").exists()

    # finally 兜底恢复：不应再因 work 目标已存在而抛错，也不得抛出任何新异常
    bi._restore_user_data_failsafe(app_dir, backup_dir)

    # 剩余目录被移回，备份目录清空
    assert (app_dir / "output" / "20260913" / "highlight.mp4").is_file()
    assert not backup_dir.exists()


def test_backup_dir_under_packaging_build(bi: ModuleType, tmp_path: pathlib.Path) -> None:
    """备份目录应落在 packaging/build/ 下，避免与 dist 同根被 PyInstaller 连带删除。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)

    assert backup_dir is not None
    assert backup_dir.parent == backup_root
    assert backup_dir.name.startswith("_userdata_backup_")


def test_restore_raises_when_backup_dir_missing(bi: ModuleType, tmp_path: pathlib.Path) -> None:
    """备份目录在恢复阶段消失时必须显式抛 BuildError，禁止静默跳过。"""
    _dist_dir, app_dir = _make_fake_dist(tmp_path)
    backup_root = _backup_root(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, backup_root)
    assert backup_dir is not None
    shutil.rmtree(backup_dir)

    with pytest.raises(bi.BuildError, match="备份目录丢失"):
        bi._restore_user_data(app_dir, backup_dir)


def test_lock_alive_pid_refuses_build(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """锁存在且 PID 存活时，main 应拒绝打包并退出码 1，且不得删除他人锁。"""
    lock = tmp_path / "dist" / ".build.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("12345 2026-09-22T12:00:00", encoding="utf-8")
    monkeypatch.setattr(bi, "LOCK_PATH", lock)
    monkeypatch.setattr(bi, "DIST_DIR", lock.parent)
    monkeypatch.setattr(bi, "BUILD_LOG", tmp_path / "build.log")
    monkeypatch.setattr(bi, "_is_pid_alive", lambda pid: True)

    assert bi.main(["--pack-only"]) == 1
    assert lock.is_file()
    assert "12345" in lock.read_text(encoding="utf-8")


def test_lock_stale_pid_overrides_and_succeeds(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """锁内 PID 已死时视为陈旧锁，覆盖后继续并成功释放锁。"""
    lock = tmp_path / "dist" / ".build.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("12345 2026-09-22T12:00:00", encoding="utf-8")
    monkeypatch.setattr(bi, "LOCK_PATH", lock)
    monkeypatch.setattr(bi, "DIST_DIR", lock.parent)
    monkeypatch.setattr(bi, "BUILD_LOG", tmp_path / "build.log")
    monkeypatch.setattr(bi, "_is_pid_alive", lambda pid: False)

    monkeypatch.setattr(bi, "ensure_assets", lambda: None)
    monkeypatch.setattr(bi, "run_pyinstaller", lambda: None)
    monkeypatch.setattr(bi, "stage_runtime_files", lambda app_dir: None)
    monkeypatch.setattr(bi, "smoke_test", lambda app_dir: None)
    monkeypatch.setattr(bi, "report_size", lambda app_dir: None)

    assert bi.main(["--pack-only"]) == 0
    assert not lock.exists()


def test_lock_released_on_build_error(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """构建链异常退出时，已获取的锁必须在 finally 中释放。"""
    lock = tmp_path / "dist" / ".build.lock"
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(bi, "LOCK_PATH", lock)
    monkeypatch.setattr(bi, "DIST_DIR", lock.parent)
    monkeypatch.setattr(bi, "BUILD_LOG", tmp_path / "build.log")

    def raise_build_error() -> None:
        raise bi.BuildError("资产检查失败")

    monkeypatch.setattr(bi, "ensure_assets", raise_build_error)

    assert bi.main(["--pack-only"]) == 1
    assert not lock.exists()
