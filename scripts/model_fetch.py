"""球检测模型首运行自动下载（AGENTS.md 约定的网络出口之一）。

输入：模型目标路径（默认 models/abdullahtarek_ball.pt，cwd 相对路径契约不变）。
输出：校验通过的本地模型路径；失败抛 ModelDownloadError（含手动下载指引）。
依赖：仅标准库；代理读 BASKETBALL_CLIP_HTTPS_PROXY / HTTPS_PROXY
    （与 packaging/fetch_assets.py 同一约定）；来源为作者项目
    https://github.com/abdullahtarek/basketball_analysis README 公布的
    Google Drive 直链（再分发权利未确认，故不随仓库/安装包分发，由用户侧下载）。
典型调用：
    from model_fetch import ensure_ball_model
    ball_model = YOLO(str(ensure_ball_model()))
"""

import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from errors import ModelDownloadError

logger = logging.getLogger(__name__)

BALL_MODEL_PATH: Path = Path("models/abdullahtarek_ball.pt")
DRIVE_FILE_ID: str = "1KejdrcEnto2AKjdgdo1U1syr5gODp6EL"
# drive.usercontent 直链端点 + confirm=t：绕开大文件病毒扫描拦截页（gdown 现行做法）
BALL_MODEL_URL: str = (
    f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}&export=download&confirm=t"
)
MANUAL_DOWNLOAD_URL: str = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"

MODEL_MIN_BYTES: int = 100_000_000  # 实重 172MB；下限用于拦截 HTML 错误页误存
ZIP_MAGIC: bytes = b"PK"  # .pt 实为 zip 容器
DOWNLOAD_CHUNK: int = 1 << 20  # 1MB
DOWNLOAD_TIMEOUT_S: int = 120  # 单次 socket 读超时（非全程）
RETRY_BACKOFF_SEC: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)  # 3 次重试（rules.md §4）
PROGRESS_LOG_EVERY_BYTES: int = 20 * DOWNLOAD_CHUNK  # 每 20MB 一条进度日志


def _proxy() -> str | None:
    """取代理地址：BASKETBALL_CLIP_HTTPS_PROXY 优先，其次 HTTPS_PROXY。"""
    return os.environ.get("BASKETBALL_CLIP_HTTPS_PROXY") or os.environ.get("HTTPS_PROXY")


def _urlopen(request: urllib.request.Request) -> urllib.response.addinfourl:
    """按代理环境变量发 HTTPS 请求（显式 ProxyHandler，不依赖全局隐式状态）。"""
    proxy = _proxy()
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
        return opener.open(request, timeout=DOWNLOAD_TIMEOUT_S)
    return urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S)  # noqa: S310 来源 URL 为模块常量


def _looks_like_model(path: Path) -> bool:
    """校验本地文件像合法的 .pt 权重（尺寸下限 + zip 魔数），拦截 HTML 拦截页误存。"""
    try:
        if path.stat().st_size < MODEL_MIN_BYTES:
            return False
        with path.open("rb") as f:
            return f.read(2) == ZIP_MAGIC
    except OSError:
        return False


def _download_once(url: str, part: Path) -> None:
    """单次尝试下载 url 到 part（调用方负责重试与原子改名）。

    Raises:
        OSError / urllib.error.URLError: 网络或写盘失败。
        ModelDownloadError: 来源返回的内容不是权重（如 Drive 拦截页 HTML）。
    """
    request = urllib.request.Request(  # noqa: S310 下载源 URL 为模块常量，非外部输入
        url, headers={"User-Agent": "basketball-clip-model-fetch"}
    )
    with _urlopen(request) as resp, part.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(DOWNLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done % PROGRESS_LOG_EVERY_BYTES < DOWNLOAD_CHUNK:
                if total:
                    logger.info("球检测模型下载中: %.0f/%.0f MB", done / 1e6, total / 1e6)
                else:
                    logger.info("球检测模型下载中: %.0f MB", done / 1e6)
    if not _looks_like_model(part):
        raise ModelDownloadError(
            f"下载内容不是有效的球检测模型（可能命中 Drive 拦截页或链接已失效），"
            f"请手动下载: {MANUAL_DOWNLOAD_URL} 并放入 {BALL_MODEL_PATH}"
        )


def ensure_ball_model(path: Path = BALL_MODEL_PATH) -> Path:
    """确保球检测模型在本地可用，缺失/损坏时从作者发布页自动下载。

    Args:
        path: 模型目标路径（默认 models/abdullahtarek_ball.pt 契约路径）。

    Returns:
        校验通过的模型路径。

    Raises:
        ModelDownloadError: 下载重试耗尽或内容校验失败；信息含手动下载 URL
            与 BASKETBALL_CLIP_HTTPS_PROXY 代理提示。
    """
    if _looks_like_model(path):
        return path
    if path.exists():
        logger.warning("球检测模型文件损坏或不完整，重新下载: %s", path)
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    logger.info("球检测模型缺失，从作者发布页下载（约 172MB）: %s", MANUAL_DOWNLOAD_URL)
    last_error: Exception | None = None
    for wait in RETRY_BACKOFF_SEC:
        if wait:
            time.sleep(wait)
        try:
            _download_once(BALL_MODEL_URL, part)
            os.replace(part, path)
            logger.info("球检测模型下载完成: %s（%.1f MB）", path, path.stat().st_size / 1e6)
            return path
        except (OSError, urllib.error.URLError, ModelDownloadError) as e:
            last_error = e
            part.unlink(missing_ok=True)
            logger.warning("球检测模型下载失败（将重试）: %s: %s", type(e).__name__, e)
    raise ModelDownloadError(
        f"球检测模型自动下载失败（已重试 {len(RETRY_BACKOFF_SEC) - 1} 次）。"
        f"如网络受限可设置代理环境变量 BASKETBALL_CLIP_HTTPS_PROXY，"
        f"或手动下载 {MANUAL_DOWNLOAD_URL} 放入 {path}。"
        f"最后错误: {type(last_error).__name__ if last_error else 'unknown'}"
    ) from last_error
