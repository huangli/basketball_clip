"""诊断日志打包：版本/环境/配置快照/任务事件流打成 zip，供用户附到 GitHub issue。

输入：work_dir（任务事件流 ``work/.gui/tasks/``）、repo_root（pyproject.toml / models/）。
输出：zip 字节串（条目全部文本、UTF-8）。
依赖：仅标准库；路由挂在 gui/app.py（GET /api/diagnostics）。
典型调用：

    data = build_diagnostics_zip(work_dir=Path("work"), repo_root=REPO_ROOT)

脱敏红线（rules.md 安全底线）：``BASKETBALL_CLIP_*`` 环境变量只写"已设置/未设置"，
绝不写值——token 是密钥；路径信息（素材目录等）是用户本机数据、由用户自己导出
分享，允许包含。
"""

from __future__ import annotations

import io
import logging
import os
import platform
import re
import shutil
import zipfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# 任务事件流收录上限：防 zip 膨胀（brief 精确值，勿改）
TASK_EVENT_LIMIT: int = 20
# 环境变量前缀与已知变量（config.txt 只写是否设置；值是密钥，绝不落盘）
ENV_PREFIX: str = "BASKETBALL_CLIP_"
KNOWN_ENV_VARS: tuple[str, ...] = (
    "BASKETBALL_CLIP_VLM_TOKEN",
    "BASKETBALL_CLIP_VLM_API_URL",
    "BASKETBALL_CLIP_HTTPS_PROXY",
)
# 关键依赖包（importlib.metadata 取版本，未安装显式标注）
KEY_PACKAGES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "torch",
    "ultralytics",
    "opencv-python",
    "open_clip_torch",
    "numpy",
    "pillow",
    "scikit-learn",
    "torchreid",
)
# pyproject.toml [project] 版本行（Python 3.10 无 tomllib，正则解析够用：本仓库格式固定）
VERSION_RE: re.Pattern[str] = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
VERSION_UNKNOWN: str = "未知"


def _app_version(repo_root: Path) -> str:
    """从 pyproject.toml 解析应用版本；缺失/不识别记 WARNING 落"未知"（不静默也不中断）。"""
    pyproject = repo_root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("pyproject 读取失败，版本落未知: %s (%s)", pyproject, e)
        return VERSION_UNKNOWN
    match = VERSION_RE.search(text)
    if match is None:
        logger.warning("pyproject 未找到 version 行，版本落未知: %s", pyproject)
        return VERSION_UNKNOWN
    return match.group(1)


def _meta_txt(repo_root: Path, now: datetime) -> str:
    """meta.txt：应用版本 / 生成时间 / 平台（OS/架构/Python 版本）。"""
    lines = [
        f"应用版本: {_app_version(repo_root)}",
        f"生成时间: {now.isoformat()}",
        f"操作系统: {platform.system()} {platform.release()}",
        f"架构: {platform.machine()}",
        f"Python 版本: {platform.python_version()}",
    ]
    return "\n".join(lines) + "\n"


def _environment_txt(repo_root: Path) -> str:
    """environment.txt：ffmpeg/ffprobe 在 PATH、models/ 各文件存在性、关键包版本。"""
    lines = ["== 外部工具 =="]
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        lines.append(f"{tool}: {'在 PATH: ' + found if found else '不在 PATH'}")
    lines.append("")
    lines.append("== 模型文件（models/ 存在性，不读内容）==")
    models_dir = repo_root / "models"
    if models_dir.is_dir():
        try:
            files = sorted(p for p in models_dir.iterdir() if p.is_file())
        except OSError as e:
            logger.warning("models 目录列举失败: %s (%s)", models_dir, e)
            files = []
        if files:
            lines.extend(f"{p.name}: 存在" for p in files)
        else:
            lines.append("（models/ 为空）")
    else:
        lines.append("models/ 目录不存在")
    lines.append("")
    lines.append("== Python 依赖版本 ==")
    for pkg in KEY_PACKAGES:
        try:
            version = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            version = "未安装"
        lines.append(f"{pkg}: {version}")
    return "\n".join(lines) + "\n"


def _config_txt() -> str:
    """config.txt：BASKETBALL_CLIP_* 环境变量是否设置（布尔，绝不写值，token 是密钥）。"""
    names = set(KNOWN_ENV_VARS)
    names.update(name for name in os.environ if name.startswith(ENV_PREFIX))
    lines = [f"{name}={'已设置' if os.environ.get(name) else '未设置'}" for name in sorted(names)]
    return "\n".join(lines) + "\n"


def _collect_task_files(work_dir: Path) -> list[Path]:
    """取 work/.gui/tasks/ 下最近 TASK_EVENT_LIMIT 个 jsonl（按 mtime 新→旧）；目录缺失返回空。"""
    tasks_dir = work_dir / ".gui" / "tasks"
    if not tasks_dir.is_dir():
        return []
    try:
        files = [p for p in tasks_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"]
    except OSError as e:
        logger.warning("任务事件目录扫描失败: %s (%s)", tasks_dir, e)
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:TASK_EVENT_LIMIT]


def build_diagnostics_zip(work_dir: Path, repo_root: Path = REPO_ROOT) -> bytes:
    """打包诊断 zip 并返回字节串。

    Args:
        work_dir: work 根（任务事件流 ``<work_dir>/.gui/tasks/*.jsonl``）。
        repo_root: 仓库根（pyproject.toml 版本、models/ 存在性）。

    Returns:
        zip 字节串；条目：meta.txt / environment.txt / config.txt /
        tasks/<task_id>.jsonl（最近 20 个任务）。

    Raises:
        OSError: zip 组装失败（调用方转 500，显式报错不静默）。
    """
    now = datetime.now(timezone.utc)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.txt", _meta_txt(repo_root, now))
        zf.writestr("environment.txt", _environment_txt(repo_root))
        zf.writestr("config.txt", _config_txt())
        for path in _collect_task_files(work_dir):
            try:
                zf.writestr(f"tasks/{path.name}", path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as e:
                # 单个任务文件读失败不拖垮整个导出（可观测跳过，同 rules.md 容忍缺失口径）
                logger.warning("任务事件文件读取失败，跳过: %s (%s)", path, e)
    return buffer.getvalue()
