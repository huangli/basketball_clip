"""build_installer.py 用户数据备份/恢复逻辑单元测试。

不跑真实 PyInstaller，仅在临时目录构造假 dist 结构，验证：
1. 正常流程：work/output 被备份，dist 重建后移回，空备份目录删除。
2. PyInstaller 失败场景：dist 被删后未重建，备份仍能移回。
3. 防御行为：目标已存在同名目录时报错不覆盖。
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


def test_backup_and_restore_on_success_mimics_pyinstaller_rebuild(
    bi: ModuleType, tmp_path: pathlib.Path
) -> None:
    """模拟 PyInstaller 删 dist 后重建，用户数据仍能完整恢复。"""
    dist_dir, app_dir = _make_fake_dist(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, dist_dir)
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
    dist_dir, app_dir = _make_fake_dist(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, dist_dir)
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
    dist_dir, app_dir = _make_fake_dist(tmp_path)

    backup_dir = bi._backup_user_data(app_dir, dist_dir)
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

    backup_dir = bi._backup_user_data(app_dir, dist_dir)
    assert backup_dir is None
    assert not any(p.name.startswith("_userdata_backup_") for p in dist_dir.iterdir())


def test_backup_rolls_back_partial_moves_on_failure(
    bi: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """备份中途失败时，已移动目录应回滚到原位，不留半截备份。"""
    dist_dir, app_dir = _make_fake_dist(tmp_path)

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
        bi._backup_user_data(app_dir, dist_dir)

    # 第一个被移动的目录已回滚到原位
    assert (app_dir / "work" / "20260913" / "goals.json").is_file()
    # 第二个目录未被移动，仍在原位
    assert (app_dir / "output" / "20260913" / "highlight.mp4").is_file()
    # 不应留下残留备份目录
    assert not any(p.name.startswith("_userdata_backup_") for p in dist_dir.iterdir())
