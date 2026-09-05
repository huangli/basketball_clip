"""PyInstaller frozen 运行时 shim：资源解析、环境注入、子进程脚本分发。

输入：无（读 ``sys.frozen`` / ``sys._MEIPASS`` / ``sys.executable`` 运行时状态）。
输出：路径常量解析结果；``bootstrap_env()`` 对 ``os.environ`` 的原地注入。
依赖：仅标准库；被 gui/__main__.py（入口分发）与 gui/app.py（路径基准）使用。
典型调用::

    # gui/__main__.py main() 开头
    frozen.freeze_support_guard()
    if frozen.dispatch_script(sys.argv):
        return  # 已作为 scripts 子进程执行完毕
    frozen.bootstrap_env()

打包后目录契约（packaging/build_installer.py 落地，此处只读）::

    <exe 同级>/            = app_dir()（用户可写：work/ output/ photos/ models/ scripts/）
    <_MEIPASS>/            = resource_dir()（只读资源）
    <_MEIPASS>/gui/static/ = 前端静态页
    <_MEIPASS>/assets/ffmpeg/bin/ = BtbN GPL ffmpeg/ffprobe
    <_MEIPASS>/assets/clip/hf-cache/ = CLIP 权重的 HF hub 缓存布局

子进程分发：runner 以 ``[sys.executable, "scripts/video.py", ...]`` 起任务；
frozen 态 sys.executable 是本 exe，故 exe 兼作 Python 解释器——argv[1] 指向
``app_dir()/scripts/`` 内的 .py 时 runpy 执行之（scripts/ 零改动，docstring、
``__file__``、relocate chdir 语义全部保持原样）。
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import runpy
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# bundled 资产相对 resource_dir() 的位置（与 packaging/basketball-clip.spec datas 一致）
FFMPEG_BIN_REL: Path = Path("assets") / "ffmpeg" / "bin"
CLIP_CACHE_REL: Path = Path("assets") / "clip" / "hf-cache"
# 分发脚本收容根：只允许执行 app_dir()/scripts/ 内的 .py（防 exe 被当通用解释器滥用）
SCRIPTS_DIR_NAME: str = "scripts"


def is_frozen() -> bool:
    """是否 PyInstaller 打包态。"""
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """用户可写应用根：frozen = exe 同级目录；开发态 = 仓库根。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """只读资源根：frozen = sys._MEIPASS；开发态 = 仓库根（开发态无 bundled 资产）。"""
    if is_frozen():
        return Path(str(sys._MEIPASS))
    return Path(__file__).resolve().parent.parent


def freeze_support_guard() -> None:
    """multiprocessing spawn 子进程守卫；非 frozen 态 no-op。

    Windows 上 joblib/torch DataLoader spawn 会以 ``--multiprocessing-fork``
    参数重入本 exe，必须在此拦截，否则会被误当 GUI/脚本请求处理。
    """
    if is_frozen():
        multiprocessing.freeze_support()


def bootstrap_env() -> None:
    """frozen 启动环境注入（os.environ 原地改，子进程经 runner env copy 继承）。

    - bundled ffmpeg bin 前置进 PATH：scripts 的 ``ffmpeg``/``ffprobe`` 裸名调用
      与 diagnostics 的 ``shutil.which`` 探测同口径命中；
    - CLIP 离线：HF_HUB_CACHE 指向随包缓存 + HF_HUB_OFFLINE=1，open_clip
      ``pretrained="laion2b_s34b_b79k"`` 走缓存命中，零网络（方案实证见
      packaging/fetch_assets.py 的离线实载验证）。

    资产缺失不致命降级：记 WARNING 跳过（ffmpeg 缺失时后端探测会报"不在 PATH"，
    用户可自装；CLIP 缺失时首次聚类报下载失败），不拦 GUI 启动。
    """
    if not is_frozen():
        return
    ffmpeg_bin = resource_dir() / FFMPEG_BIN_REL
    if (ffmpeg_bin / "ffmpeg.exe").is_file():
        os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
        logger.info("bundled ffmpeg 已注入 PATH: %s", ffmpeg_bin)
    else:
        logger.warning("bundled ffmpeg 缺失，跳过 PATH 注入: %s", ffmpeg_bin)
    clip_cache = resource_dir() / CLIP_CACHE_REL
    if clip_cache.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_CACHE", str(clip_cache))
        logger.info("CLIP 离线缓存已指向随包目录: %s", clip_cache)
    else:
        logger.warning("CLIP 随包缓存缺失，跳过离线注入: %s", clip_cache)


def dispatch_script(argv: list[str]) -> bool:
    """frozen 子进程分发：argv[1] 指向 scripts/ 内 .py 时 runpy 执行并返回 True。

    开发态 / 无脚本参数一律返回 False（走 GUI 启动）。脚本的 ``SystemExit``
    与未捕获异常原样向上抛（退出码语义与直接 python 调用一致）。

    Args:
        argv: 进程参数（含 argv[0] = exe 路径）。

    Returns:
        True = 已分发执行（调用方不得再起 GUI）；False = 非脚本请求。

    Raises:
        SystemExit: 脚本路径非法（不在 scripts/ 收容根内或文件不存在），退出码 2。
    """
    if not is_frozen() or len(argv) < 2:
        return False
    script_arg = argv[1]
    if not script_arg.endswith(".py"):
        return False
    script = Path(script_arg)
    if not script.is_absolute():
        script = (Path.cwd() / script).resolve()
    scripts_root = (app_dir() / SCRIPTS_DIR_NAME).resolve()
    if not script.is_relative_to(scripts_root) or not script.is_file():
        print(f"错误: 脚本不存在或不在 scripts/ 目录内: {script}", file=sys.stderr)  # noqa: T201
        raise SystemExit(2)
    bootstrap_env()  # 子进程同样要 ffmpeg PATH / CLIP 离线（env copy 之外的双保险）
    sys.argv = [str(script), *argv[2:]]
    # scripts 内模块平级互 import（from errors import ...），目录前插 sys.path
    sys.path.insert(0, str(scripts_root))
    logger.info("frozen 脚本分发: %s %s", script.name, argv[2:])
    runpy.run_path(str(script), run_name="__main__")
    return True
