"""一键打包：资产齐备 → PyInstaller → exe 同级补料 → 冒烟 → Inno 编译安装器。

输入：packaging/assets/（缺失自动调 fetch_assets.py 幂等补齐）、仓库源码；
    Inno 段需本机已装 Inno Setup 6（ISCC.exe，PATH 或常见安装路径自动探测，
    也可用环境变量 ``ISCC`` 显式指定）。
输出：``dist/basketball-clip/`` one-folder 包 + ``packaging/dist/`` 下
    ``basketball-clip-setup-<version>.exe`` 安装器 + ``packaging/build.log``
    全程日志（Inno 段同日志追加记录）。
依赖：打包 venv（.venv-spike：Python 3.10 + PyInstaller 6.22）；仅标准库 + PyInstaller。
典型调用（cwd 任意，脚本自带仓库根定位）::

    .venv-spike/Scripts/python.exe packaging/build_installer.py              # 全链含安装器
    .venv-spike/Scripts/python.exe packaging/build_installer.py --pack-only  # 只出 one-folder 包

产物布局契约（gui/frozen.py 的只读方）::

    dist/basketball-clip/basketball-clip.exe   入口（GUI / scripts 分发双态）
    dist/basketball-clip/_internal/            运行时 + gui/static + assets（ffmpeg/CLIP）
    dist/basketball-clip/scripts/              打包后从仓库拷贝（子进程分发实体）
    dist/basketball-clip/models/yolov8n.pt     打包后拷贝（scripts 按 cwd 相对路径读）
    dist/basketball-clip/pyproject.toml        打包后拷贝（diagnostics 版本解析）
    dist/basketball-clip/work|output|photos/   运行时生成（exe 同级，用户可写）

打包流程会在 PyInstaller 删除 ``dist/basketball-clip/`` 之前，自动把 ``work/`` 和
``output/`` 整体备份到 ``dist/_userdata_backup_<时间戳>/``；打包/补料完成后再移回原位。
任何步骤失败时，finally 也会尝试把备份移回，避免用户数据停留在临时备份目录。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("build_installer")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PACKAGING_DIR: Path = REPO_ROOT / "packaging"
SPEC_PATH: Path = PACKAGING_DIR / "basketball-clip.spec"
ISS_PATH: Path = PACKAGING_DIR / "installer.iss"
INSTALLER_DIR: Path = PACKAGING_DIR / "dist"  # 安装器产物（与 iss 内 OutputDir 一致）
FETCH_SCRIPT: Path = PACKAGING_DIR / "fetch_assets.py"
BUILD_LOG: Path = PACKAGING_DIR / "build.log"
DIST_DIR: Path = REPO_ROOT / "dist"
BUILD_DIR: Path = REPO_ROOT / "build"
APP_NAME: str = "basketball-clip"

PYINSTALLER_TIMEOUT_S: int = 3600  # 全量打包实测 3-5 分钟，给足余量
INNO_TIMEOUT_S: int = 14400  # 1.8GB lzma2/ultra64 压缩实测可达数十分钟，给 4 小时余量
SMOKE_TIMEOUT_S: int = 600  # exe 首次启动 torch import 较慢
# scripts/ 拷贝排除：字节码缓存不进包
COPY_IGNORE: tuple[str, ...] = ("__pycache__", "*.pyc")
# 用户运行数据目录（exe 同级，打包删 dist 前必须临时移出保护）
USER_DATA_DIR_NAMES: tuple[str, ...] = ("work", "output")
# Inno 编译器探测路径（用户级安装 + 两台机器级常见路径；PATH 优先）
_ISCC_CANDIDATES: tuple[str, ...] = (
    r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe",
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
)


class BuildError(Exception):
    """打包流程失败（显式失败，不静默）。"""


class _TeeLogHandler(logging.FileHandler):
    """build.log 文件 handler（UTF-8，每轮覆盖重写——留存当轮构建记录）。"""

    def __init__(self) -> None:
        super().__init__(BUILD_LOG, mode="w", encoding="utf-8")


def _configure_logging() -> None:
    """root logger 双写：控制台 + packaging/build.log。"""
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), _TeeLogHandler()],
        force=True,
    )


def _run_logged(cmd: list[str], *, cwd: Path, timeout: int, what: str) -> None:
    """跑子进程并逐行 tee 到日志；非零退出显式失败。

    Raises:
        BuildError: 启动失败 / 超时 / 非零退出。
    """
    logger.info("执行: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(  # noqa: S603 命令由本模块内部构造（sys.executable + 固定参数）
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as e:
        raise BuildError(f"{what} 启动失败 cmd={cmd!r}: {e}") from e
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.wait()
        raise BuildError(f"{what} 超时（>{timeout}s）: {cmd!r}") from e
    for line in out.splitlines():
        logger.info("[%s] %s", what, line)
    if proc.returncode != 0:
        raise BuildError(f"{what} 非零退出: returncode={proc.returncode}")


def ensure_assets() -> None:
    """资产齐备检查：缺则调 fetch_assets.py 幂等补齐（断点友好，已齐备秒过）。"""
    _run_logged(
        [sys.executable, str(FETCH_SCRIPT)],
        cwd=REPO_ROOT,
        timeout=PYINSTALLER_TIMEOUT_S,
        what="fetch-assets",
    )


def run_pyinstaller() -> None:
    """PyInstaller 全量打包（spec 见 packaging/basketball-clip.spec）。"""
    _run_logged(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(SPEC_PATH),
            "--distpath",
            str(DIST_DIR),
            "--workpath",
            str(BUILD_DIR),
            "--noconfirm",
        ],
        cwd=REPO_ROOT,
        timeout=PYINSTALLER_TIMEOUT_S,
        what="pyinstaller",
    )


def _copy_tree_fresh(src: Path, dst: Path, *, what: str) -> None:
    """整目录新鲜拷贝（先删旧防残留）；源缺失显式失败。"""
    if not src.is_dir():
        raise BuildError(f"{what} 源目录缺失: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*COPY_IGNORE))
    logger.info("%s 已拷贝: %s -> %s", what, src, dst)


def stage_runtime_files(app_dir: Path) -> None:
    """exe 同级补料：scripts/ 实体、models/yolov8n.pt、pyproject.toml。

    这三类必须落在用户可写的 exe 同级目录而非 _internal：video.py relocate 以
    ``__file__`` 推导仓库根并 chdir（work/ 等相对路径基准），models/ 由 scripts
    按 cwd 相对路径读取，pyproject.toml 供 diagnostics 解析版本。
    """
    _copy_tree_fresh(REPO_ROOT / "scripts", app_dir / "scripts", what="scripts/")
    model_src = PACKAGING_DIR / "assets" / "models" / "yolov8n.pt"
    model_dst = app_dir / "models" / "yolov8n.pt"
    if not model_src.is_file():
        raise BuildError(f"yolov8n.pt 资产缺失: {model_src}")
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, model_dst)
    logger.info("models/yolov8n.pt 已拷贝: %s", model_dst)
    shutil.copy2(REPO_ROOT / "pyproject.toml", app_dir / "pyproject.toml")
    logger.info("pyproject.toml 已拷贝（diagnostics 版本解析用）")


def smoke_test(app_dir: Path) -> None:
    """冒烟：exe 以 scripts 分发态跑 ``video.py --help``（验证 frozen 子进程链路）。

    覆盖：bootloader 启动 → frozen.dispatch_script → runpy → scripts 全 import 链
    （torch/ultralytics/cv2 等重依赖在此首次实证）。GUI 态验证在干净目录实跑做。
    """
    exe = app_dir / f"{APP_NAME}.exe"
    if not exe.is_file():
        raise BuildError(f"打包产物缺失: {exe}")
    _run_logged(
        [str(exe), "scripts/video.py", "--help"],
        cwd=app_dir,
        timeout=SMOKE_TIMEOUT_S,
        what="smoke",
    )


def _dir_size_mb(path: Path) -> float:
    """目录体积（MB，遍历求和；缺失记 0 并 WARNING）。"""
    if not path.is_dir():
        logger.warning("体积统计目录缺失: %s", path)
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1e6


def report_size(app_dir: Path) -> None:
    """包体积记录（总量 + _internal/scripts/models 分项），进 build.log 与报告。"""
    logger.info(
        "包体积: 总计 %.0f MB（_internal %.0f / scripts %.0f / models %.0f）",
        _dir_size_mb(app_dir),
        _dir_size_mb(app_dir / "_internal"),
        _dir_size_mb(app_dir / "scripts"),
        _dir_size_mb(app_dir / "models"),
    )


def read_version() -> str:
    """从 pyproject.toml 解析 ``version = "x.y.z"``（Python 3.10 无 tomllib，用正则）。

    Raises:
        BuildError: 文件缺失或未匹配到版本字段。
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        raise BuildError(f"pyproject.toml 缺失: {pyproject}")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
    if match is None:
        raise BuildError(f"pyproject.toml 中未找到 version 字段: {pyproject}")
    return match.group(1)


def find_iscc() -> Path:
    """定位 Inno 编译器：环境变量 ISCC > PATH > 常见安装路径。

    Raises:
        BuildError: 全部未命中（提示 winget 安装）。
    """
    env = os.environ.get("ISCC")
    if env and Path(env).is_file():
        return Path(env)
    on_path = shutil.which("ISCC") or shutil.which("iscc")
    if on_path:
        return Path(on_path)
    for pattern in _ISCC_CANDIDATES:
        candidate = Path(os.path.expandvars(pattern))
        if candidate.is_file():
            return candidate
    raise BuildError(
        "未找到 ISCC.exe（winget install JRSoftware.InnoSetup，或设环境变量 ISCC 指向）"
    )


def run_inno(app_dir: Path) -> Path:
    """Inno 编译安装器 → packaging/dist/basketball-clip-setup-<version>.exe。

    Raises:
        BuildError: one-folder 包/iss/图标缺失，或编译失败，或产物未出现。
    """
    if not (app_dir / f"{APP_NAME}.exe").is_file():
        raise BuildError(f"Inno 源包缺失（先跑打包段）: {app_dir}")
    for required in (ISS_PATH, PACKAGING_DIR / "icon.ico", PACKAGING_DIR / "INSTALLER_LICENSE.txt"):
        if not required.is_file():
            raise BuildError(f"Inno 编译输入缺失: {required}")
    version = read_version()
    iscc = find_iscc()
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    _run_logged(
        [
            str(iscc),
            f"/DAppVersion={version}",
            f"/DSourceDir={app_dir}",
            f"/O{INSTALLER_DIR}",
            str(ISS_PATH),
        ],
        cwd=PACKAGING_DIR,
        timeout=INNO_TIMEOUT_S,
        what="inno",
    )
    setup_exe = INSTALLER_DIR / f"{APP_NAME}-setup-{version}.exe"
    if not setup_exe.is_file():
        raise BuildError(f"Inno 编译成功但产物缺失: {setup_exe}")
    logger.info("安装器产物: %s（%.0f MB）", setup_exe, setup_exe.stat().st_size / 1e6)
    return setup_exe


def _user_data_dirs(app_dir: Path) -> list[Path]:
    """返回需保护的运行时数据目录路径列表（按写死语义仅 work/output）。"""
    return [app_dir / name for name in USER_DATA_DIR_NAMES]


def _backup_user_data(app_dir: Path, backup_root: Path) -> Path | None:
    """PyInstaller 删 dist 前，把 work/output 整体移到备份目录。

    若移动中途失败，会把已移动的目录尽量移回原位，再抛 ``BuildError``，
    避免用户数据分裂在原始位置与备份目录之间。

    Args:
        app_dir: 当前包目录（如 dist/basketball-clip/）。
        backup_root: 备份目录父目录（如 dist/）。

    Returns:
        创建的备份目录；无数据需备份时返回 None。

    Raises:
        BuildError: 移动失败或回滚失败。
    """
    data_dirs = [p for p in _user_data_dirs(app_dir) if p.is_dir()]
    if not data_dirs:
        logger.info("无需备份用户数据: %s 下无 work/output", app_dir)
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = backup_root / f"_userdata_backup_{timestamp}"
    try:
        backup_dir.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise BuildError(f"无法创建备份目录: {backup_dir}") from e
    logger.info("备份用户数据到: %s", backup_dir)
    moved: list[tuple[Path, Path]] = []
    try:
        for data_dir in data_dirs:
            target = backup_dir / data_dir.name
            shutil.move(str(data_dir), str(target))
            moved.append((data_dir, target))
            logger.info("已移动: %s -> %s", data_dir, target)
    except OSError as e:
        logger.error("备份过程中出错: %s；尝试回滚已移动目录", e)
        for original, backed in moved:
            try:
                shutil.move(str(backed), str(original))
                logger.info("已回滚: %s -> %s", backed, original)
            except OSError as rollback_err:
                logger.error("回滚失败 %s -> %s: %s", backed, original, rollback_err)
        try:
            if backup_dir.is_dir() and not any(backup_dir.iterdir()):
                backup_dir.rmdir()
        except OSError:
            pass
        raise BuildError(f"备份用户数据失败: {e}") from e
    return backup_dir


def _try_remove_backup_dir(backup_dir: Path, *, warn_on_left: bool = False) -> None:
    """删除空备份目录；非空时按参数决定是否告警。任何 OSError 只记录不抛出。"""
    try:
        if backup_dir.is_dir() and not any(backup_dir.iterdir()):
            backup_dir.rmdir()
            logger.info("已删除空备份目录: %s", backup_dir)
        elif backup_dir.is_dir() and warn_on_left:
            logger.warning("备份目录仍含内容，未删除: %s", backup_dir)
    except OSError as e:
        logger.error("删除备份目录 %s 失败: %s", backup_dir, e)


def _restore_user_data(app_dir: Path, backup_dir: Path | None) -> set[str]:
    """打包完成后把备份的 work/output 移回原位，并删除空备份目录。

    若 app_dir 不存在会自动创建父目录；若目标位置已存在同名目录则报错不覆盖。
    返回成功恢复的目录名集合；失败抛出异常时，已移回的目录可通过检查 backup_dir
    剩余内容或调用 ``_restore_user_data_failsafe`` 兜底恢复。

    Args:
        app_dir: 新包目录。
        backup_dir: 备份目录（可为 None）。

    Returns:
        成功恢复的目录名集合。

    Raises:
        BuildError: 目标已存在同名目录，或移回失败。
    """
    restored: set[str] = set()
    if backup_dir is None or not backup_dir.is_dir():
        return restored
    logger.info("恢复用户数据: %s -> %s", backup_dir, app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    for name in USER_DATA_DIR_NAMES:
        backed = backup_dir / name
        if not backed.is_dir():
            continue
        original = app_dir / name
        if original.exists():
            raise BuildError(f"恢复用户数据冲突，目标已存在: {original}")
        try:
            shutil.move(str(backed), str(original))
        except OSError as e:
            raise BuildError(f"恢复用户数据失败 {backed} -> {original}: {e}") from e
        restored.add(name)
        logger.info("已恢复: %s", original)
    _try_remove_backup_dir(backup_dir)
    return restored


def _restore_user_data_failsafe(app_dir: Path, backup_dir: Path | None) -> None:
    """finally 兜底恢复：只处理 backup_dir 中仍剩余的目录，任何异常只记录不抛出。

    目标位置已存在同名目录时视为该目录已恢复并跳过；单个目录移回失败时记录 ERROR
    并继续处理剩余目录，避免掩盖主异常。

    Args:
        app_dir: 新包目录。
        backup_dir: 备份目录（可为 None）。
    """
    if backup_dir is None or not backup_dir.is_dir():
        return
    logger.info("兜底恢复用户数据: %s -> %s", backup_dir, app_dir)
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("无法创建应用目录 %s: %s", app_dir, e)
        return
    for name in USER_DATA_DIR_NAMES:
        backed = backup_dir / name
        if not backed.is_dir():
            continue
        original = app_dir / name
        if original.exists():
            logger.warning("兜底恢复跳过（目标已存在，视为已恢复）: %s", original)
            continue
        try:
            shutil.move(str(backed), str(original))
            logger.info("兜底恢复成功: %s", original)
        except OSError as e:
            logger.error("兜底恢复失败 %s -> %s: %s", backed, original, e)
    _try_remove_backup_dir(backup_dir, warn_on_left=True)


def main(argv: list[str] | None = None) -> int:
    """打包入口。返回退出码（0=成功，1=失败，2=参数错误）。"""
    args = argv if argv is not None else sys.argv[1:]
    pack_only = args == ["--pack-only"]
    if args and not pack_only:
        print("用法: python packaging/build_installer.py [--pack-only]")  # noqa: T201
        return 2
    _configure_logging()
    started = time.monotonic()
    app_dir = DIST_DIR / APP_NAME
    backup_dir: Path | None = None
    try:
        ensure_assets()
        backup_dir = _backup_user_data(app_dir, DIST_DIR)
        run_pyinstaller()
        app_dir = DIST_DIR / APP_NAME
        stage_runtime_files(app_dir)
        _restore_user_data(app_dir, backup_dir)
        backup_dir = None
        smoke_test(app_dir)
        report_size(app_dir)
        if not pack_only:
            run_inno(app_dir)
    except BuildError as e:
        logger.error("打包失败: %s", e)
        return 1
    finally:
        if backup_dir is not None:
            _restore_user_data_failsafe(app_dir, backup_dir)
    logger.info(
        "打包完成: %s（耗时 %.1f 分钟）", DIST_DIR / APP_NAME, (time.monotonic() - started) / 60
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
