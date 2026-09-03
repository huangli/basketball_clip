"""打包资产下载：ffmpeg（BtbN LGPL）/ yolov8n.pt / CLIP 权重（HF 缓存布局）。

输入：网络（GitHub / Hugging Face；代理走 ``BASKETBALL_CLIP_HTTPS_PROXY`` 或
    ``HTTPS_PROXY`` 环境变量，均未设置则直连）。
输出：``packaging/assets/``——

    assets/ffmpeg/bin/ffmpeg.exe, ffprobe.exe   BtbN win64 LGPL 构建（解出 bin/）
    assets/ffmpeg/LICENSE.txt, COPYING.*        许可文本随包（spec O2 口径）
    assets/models/yolov8n.pt                    ultralytics 官方 release
    assets/clip/hf-cache/models--laion--*/      CLIP 权重的 HF hub 缓存布局
                                                （运行时设 HF_HUB_CACHE 指过来 +
                                                HF_HUB_OFFLINE=1 即离线可用）

依赖：标准库 + huggingface_hub（CLIP 段）；用打包 venv（.venv-spike）运行。
典型调用：

    set BASKETBALL_CLIP_HTTPS_PROXY=http://127.0.0.1:7897
    .venv-spike/Scripts/python.exe packaging/fetch_assets.py

断点友好：目标已存在且通过文件性检查则跳过；下载走 ``.part`` 临时文件 +
``os.replace`` 原子落盘，失败不留半截文件。任何一步失败显式非零退出。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("fetch_assets")

PACKAGING_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = PACKAGING_DIR / "assets"
FFMPEG_DIR: Path = ASSETS_DIR / "ffmpeg"
MODELS_DIR: Path = ASSETS_DIR / "models"
CLIP_CACHE_DIR: Path = ASSETS_DIR / "clip" / "hf-cache"

# ---- 资产来源（spec O1/O2/O5 定案口径） ----
GITHUB_API_LATEST: str = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
# BtbN win64 LGPL 非 shared 变体（如 ffmpeg-n8.0-latest-win64-lgpl-8.0.zip）；
# 版本段以数字开头，借此排除 -shared 变体
FFMPEG_ASSET_RE: re.Pattern[str] = re.compile(r"^ffmpeg-\S+-win64-lgpl-\d\S*\.zip$")
YOLOV8N_URL: str = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
CLIP_REPO_ID: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # open_clip 注册表精确值
# open_clip 优先取 safetensors 变体（pretrained.HF_SAFE_WEIGHTS_NAME，实测 605MB fp32）
CLIP_FILENAME: str = "open_clip_model.safetensors"

# ---- 文件性检查阈值 ----
YOLOV8N_MIN_BYTES: int = 3_000_000  # yolov8n.pt 实际 ~6MB
CLIP_MIN_BYTES: int = 100_000_000  # ViT-B-32 safetensors ~350MB
FFPROBE_TIMEOUT_S: int = 30
DOWNLOAD_CHUNK: int = 1 << 20  # 1MB
DOWNLOAD_TIMEOUT_S: int = 120  # 单次 socket 读超时（非全程）


class AssetError(Exception):
    """资产下载/校验失败（显式失败，不静默）。"""


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
    return urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S)  # noqa: S310 来源 URL 固定


def _download(url: str, target: Path, *, what: str) -> None:
    """下载 url 到 target（.part + os.replace 原子落盘），带进度日志。

    Raises:
        AssetError: 网络/写盘失败。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(  # noqa: S310 下载源 URL 为模块常量（GitHub/HF），非外部输入
        url, headers={"User-Agent": "basketball-clip-fetch-assets"}
    )
    try:
        with _urlopen(request) as resp, part.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and done % (50 * DOWNLOAD_CHUNK) < DOWNLOAD_CHUNK:
                    logger.info("%s 下载中: %.0f/%.0f MB", what, done / 1e6, total / 1e6)
        os.replace(part, target)
    except (OSError, urllib.error.URLError) as e:
        part.unlink(missing_ok=True)
        raise AssetError(f"{what} 下载失败 url={url}: {type(e).__name__}: {e}") from e
    logger.info("%s 下载完成: %s（%.1f MB）", what, target, target.stat().st_size / 1e6)


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256（大文件不一次读入内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- ffmpeg（BtbN LGPL） ----


def _latest_ffmpeg_url() -> tuple[str, str]:
    """查 BtbN latest release，返回 (win64-lgpl zip 下载地址, tag)。

    Raises:
        AssetError: API 失败或未匹配到资产。
    """
    request = urllib.request.Request(
        GITHUB_API_LATEST, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with _urlopen(request) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise AssetError(f"BtbN release 查询失败: {type(e).__name__}: {e}") from e
    tag = str(payload.get("tag_name", "未知"))
    for asset in payload.get("assets", []):
        name = str(asset.get("name", ""))
        if FFMPEG_ASSET_RE.match(name):
            return str(asset["browser_download_url"]), tag
    raise AssetError(f"BtbN release {tag} 未找到 win64-lgpl zip 资产")


def _check_ffmpeg_layout() -> bool:
    """ffmpeg 资产已齐备判定：bin/ 双 exe + 许可文本都在。"""
    bin_dir = FFMPEG_DIR / "bin"
    if not (bin_dir / "ffmpeg.exe").is_file() or not (bin_dir / "ffprobe.exe").is_file():
        return False
    return any(p.is_file() for p in FFMPEG_DIR.glob("LICENSE*"))


def fetch_ffmpeg() -> None:
    """下载并解出 BtbN ffmpeg bin/ 与许可文本；齐备则跳过，解出后跑 -version 实证。"""
    if _check_ffmpeg_layout():
        logger.info("ffmpeg 资产已齐备，跳过: %s", FFMPEG_DIR)
        return
    url, tag = _latest_ffmpeg_url()
    logger.info("BtbN latest release: %s", tag)
    with tempfile.TemporaryDirectory(prefix="fetch-ffmpeg-") as tmp:
        zip_path = Path(tmp) / "ffmpeg.zip"
        _download(url, zip_path, what="ffmpeg")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                wanted = [
                    n
                    for n in names
                    if re.search(r"/bin/(ffmpeg|ffprobe)\.exe$", n)
                    or re.search(r"/(LICENSE[^/]*|COPYING[^/]*)$", n)
                ]
                if not any(n.endswith("/bin/ffmpeg.exe") for n in wanted):
                    raise AssetError(f"ffmpeg zip 内未找到 bin/ffmpeg.exe: {url}")
                for name in wanted:
                    zf.extract(name, tmp)
        except zipfile.BadZipFile as e:
            raise AssetError(f"ffmpeg zip 损坏: {zip_path}: {e}") from e
        # zip 内唯一顶层目录（ffmpeg-nX.Y-latest-win64-lgpl-X.Y/），摊平到 assets/ffmpeg/
        roots = {n.split("/")[0] for n in wanted}
        if len(roots) != 1:
            raise AssetError(f"ffmpeg zip 顶层目录不唯一: {roots}")
        inner = Path(tmp) / roots.pop()
        if FFMPEG_DIR.exists():
            shutil.rmtree(FFMPEG_DIR)
        FFMPEG_DIR.mkdir(parents=True)
        shutil.move(str(inner / "bin"), str(FFMPEG_DIR / "bin"))
        for lic in inner.iterdir():
            if lic.is_file() and lic.name.startswith(("LICENSE", "COPYING")):
                shutil.move(str(lic), str(FFMPEG_DIR / lic.name))
    # 文件性检查：解出的二进制实跑 -version
    for tool in ("ffmpeg", "ffprobe"):
        exe = FFMPEG_DIR / "bin" / f"{tool}.exe"
        try:
            proc = subprocess.run(  # noqa: S603 执行刚解出的自带二进制，路径内部构造
                [str(exe), "-version"],
                capture_output=True,
                timeout=FFPROBE_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise AssetError(f"{tool} -version 执行失败: {e}") from e
        if proc.returncode != 0:
            raise AssetError(f"{tool} -version 退出码 {proc.returncode}")
    logger.info("ffmpeg 资产就绪: %s（-version 实证通过）", FFMPEG_DIR)


# ---- yolov8n.pt ----


def _check_yolov8n(path: Path) -> bool:
    """yolov8n.pt 文件性检查：torch 存档是 zip 容器，验魔数 + 体积下限。"""
    return path.is_file() and path.stat().st_size >= YOLOV8N_MIN_BYTES and zipfile.is_zipfile(path)


def fetch_yolov8n() -> None:
    """下载 yolov8n.pt（ultralytics 官方 release）；存在且合法则跳过。"""
    target = MODELS_DIR / "yolov8n.pt"
    if _check_yolov8n(target):
        logger.info("yolov8n.pt 已存在且合法，跳过: %s", target)
        return
    _download(YOLOV8N_URL, target, what="yolov8n.pt")
    if not _check_yolov8n(target):
        target.unlink(missing_ok=True)
        raise AssetError(f"yolov8n.pt 文件性检查未通过（非 zip 容器或体积异常）: {target}")
    logger.info("yolov8n.pt SHA-256: %s", _sha256(target))


# ---- CLIP 权重（HF 缓存布局，随包离线用） ----


def _clip_snapshot_file() -> Path | None:
    """在 HF 缓存里定位 CLIP 权重快照实体；未下载完整返回 None。"""
    repo_dir = CLIP_CACHE_DIR / f"models--{CLIP_REPO_ID.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for rev in snapshots.iterdir():
        candidate = rev / CLIP_FILENAME
        if candidate.is_file() and candidate.stat().st_size >= CLIP_MIN_BYTES:
            return candidate
    return None


def _trim_blob_duplicates(snapshot: Path) -> None:
    """snapshot 为实体副本时（Windows 无软链权限）删除同尺寸 blob，省一份 350MB。

    HF 离线解析只走 refs + snapshots（try_to_load_from_cache 返回 snapshot 路径），
    blob 仅在 snapshot 是软链时作链接目标需要；实体副本场景 blob 是纯重复。
    """
    if snapshot.is_symlink():
        return
    repo_dir = snapshot.parent.parent.parent  # snapshots/<rev>/<file> → models--org--name
    blobs_dir = repo_dir / "blobs"
    if not blobs_dir.is_dir():
        return
    for candidate in blobs_dir.iterdir():
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and candidate.stat().st_size == snapshot.stat().st_size
        ):
            logger.info("删除重复 blob（snapshot 已是实体）: %s", candidate)
            candidate.unlink()


def fetch_clip() -> None:
    """下载 CLIP 权重进 HF 缓存布局；随后 HF_HUB_OFFLINE=1 实载验证（离线可用铁证）。

    Raises:
        AssetError: huggingface_hub 未安装 / 下载失败 / 离线实载失败。
    """
    if _clip_snapshot_file() is not None:
        logger.info("CLIP 权重已在缓存，跳过下载: %s", CLIP_CACHE_DIR)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise AssetError(f"huggingface_hub 未安装（用 .venv-spike 运行）: {e}") from e
        CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("CLIP 权重下载中（~350MB）: %s/%s", CLIP_REPO_ID, CLIP_FILENAME)
        try:
            hf_hub_download(
                repo_id=CLIP_REPO_ID, filename=CLIP_FILENAME, cache_dir=str(CLIP_CACHE_DIR)
            )
        except Exception as e:
            raise AssetError(f"CLIP 权重下载失败: {type(e).__name__}: {e}") from e
    snapshot = _clip_snapshot_file()
    if snapshot is None:
        raise AssetError(f"CLIP 权重下载后仍缺失快照实体: {CLIP_CACHE_DIR}")
    _trim_blob_duplicates(snapshot)
    logger.info("CLIP 权重就绪: %s（%.1f MB）", snapshot, snapshot.stat().st_size / 1e6)
    logger.info("CLIP SHA-256: %s", _sha256(snapshot))
    _verify_clip_offline()


def _verify_clip_offline() -> None:
    """子进程内 HF_HUB_OFFLINE=1 + HF_HUB_CACHE=随包缓存 实载 open_clip 模型。

    与打包后运行口径完全一致（gui/frozen.py bootstrap_env 注入同组变量），
    在此先证一次：加载成功即离线方案成立，失败显式报错中断 fetch。
    """
    code = (
        "import open_clip; "
        "open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); "
        "print('CLIP-OFFLINE-OK')"
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_CACHE"] = str(CLIP_CACHE_DIR)
    env.pop("HTTPS_PROXY", None)
    env.pop("BASKETBALL_CLIP_HTTPS_PROXY", None)
    try:
        proc = subprocess.run(  # noqa: S603 sys.executable 自校验，命令内部构造
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise AssetError(f"CLIP 离线实载子进程异常: {e}") from e
    if proc.returncode != 0 or "CLIP-OFFLINE-OK" not in proc.stdout:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise AssetError(f"CLIP 离线实载失败（returncode={proc.returncode}）:\n{tail}")
    logger.info("CLIP 离线实载验证通过（HF_HUB_OFFLINE=1）")


def main() -> int:
    """fetch 入口：三类资产逐个齐备，任一失败非零退出。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # huggingface_hub 只认标准变量；项目变量桥接过去（不覆盖已有 HTTPS_PROXY）
    proxy = _proxy()
    if proxy and not os.environ.get("HTTPS_PROXY"):
        os.environ["HTTPS_PROXY"] = proxy
        logger.info("代理: %s（BASKETBALL_CLIP_HTTPS_PROXY 桥接）", proxy)
    elif proxy:
        logger.info("代理: %s", proxy)
    else:
        logger.info("未设置代理，直连下载")
    try:
        fetch_ffmpeg()
        fetch_yolov8n()
        fetch_clip()
    except AssetError as e:
        logger.error("资产准备失败: %s", e)
        return 1
    logger.info("全部资产就绪: %s", ASSETS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
