"""一键打包（E-1 范围：资产齐备 → PyInstaller → exe 同级补料 → 冒烟；Inno 段属 E-2）。

输入：packaging/assets/（缺失自动调 fetch_assets.py 幂等补齐）、仓库源码。
输出：``dist/basketball-clip/`` one-folder 包 + ``packaging/build.log`` 全程日志。
依赖：打包 venv（.venv-spike：Python 3.10 + PyInstaller 6.22）；仅标准库 + PyInstaller。
典型调用（cwd 任意，脚本自带仓库根定位）::

    .venv-spike/Scripts/python.exe packaging/build_installer.py

产物布局契约（gui/frozen.py 的只读方）::

    dist/basketball-clip/basketball-clip.exe   入口（GUI / scripts 分发双态）
    dist/basketball-clip/_internal/            运行时 + gui/static + assets（ffmpeg/CLIP）
    dist/basketball-clip/scripts/              打包后从仓库拷贝（子进程分发实体）
    dist/basketball-clip/models/yolov8n.pt     打包后拷贝（scripts 按 cwd 相对路径读）
    dist/basketball-clip/pyproject.toml        打包后拷贝（diagnostics 版本解析）
    dist/basketball-clip/work|output|photos/   运行时生成（exe 同级，用户可写）
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("build_installer")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PACKAGING_DIR: Path = REPO_ROOT / "packaging"
SPEC_PATH: Path = PACKAGING_DIR / "basketball-clip.spec"
FETCH_SCRIPT: Path = PACKAGING_DIR / "fetch_assets.py"
BUILD_LOG: Path = PACKAGING_DIR / "build.log"
DIST_DIR: Path = REPO_ROOT / "dist"
BUILD_DIR: Path = REPO_ROOT / "build"
APP_NAME: str = "basketball-clip"

PYINSTALLER_TIMEOUT_S: int = 3600  # 全量打包实测 3-5 分钟，给足余量
SMOKE_TIMEOUT_S: int = 600  # exe 首次启动 torch import 较慢
# scripts/ 拷贝排除：字节码缓存不进包
COPY_IGNORE: tuple[str, ...] = ("__pycache__", "*.pyc")


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


def main(argv: list[str] | None = None) -> int:
    """打包入口。返回退出码（0=成功，1=失败，2=参数错误）。"""
    args = argv if argv is not None else sys.argv[1:]
    if args:
        print("用法: python packaging/build_installer.py（Inno 安装器段属 E-2，未实现）")  # noqa: T201
        return 2
    _configure_logging()
    started = time.monotonic()
    try:
        ensure_assets()
        run_pyinstaller()
        app_dir = DIST_DIR / APP_NAME
        stage_runtime_files(app_dir)
        smoke_test(app_dir)
        report_size(app_dir)
    except BuildError as e:
        logger.error("打包失败: %s", e)
        return 1
    logger.info(
        "打包完成: %s（耗时 %.1f 分钟）", DIST_DIR / APP_NAME, (time.monotonic() - started) / 60
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
