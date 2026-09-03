#!/usr/bin/env python3
"""抽帧：原片 -> work/frames/<fid>/f_%05d.jpg（5fps、1920 宽，供检测流水线）。

输入：原片目录（递归扫描 .mp4/.MP4），fid = 原片主名（文件名即拍摄时间，
    按文件名排序即时间顺序）。
输出：work/frames/<fid>/f_00001.jpg ...（帧序号 1 起，sec=(idx-1)/5，
    与 mot_candidates.parse_sec 的约定一致）。
依赖：scripts/errors.py、scripts/pipe_common.py。
典型调用：python scripts/extract_frames.py "素材目录" --limit 50

断点续做：帧目录已存在且帧数与时长推算值相符（±2）则跳过；
中断（Ctrl-C）后重跑即可从未抽文件继续。
"""

import logging
import os
import subprocess
import sys
import time
from glob import escape, glob

from errors import BasketballPipelineError, MediaTimeoutError
from pipe_common import configure_logging, new_run_id, run_ffmpeg

logger = logging.getLogger(__name__)

FRAMES_ROOT: str = "work/frames"
SAMPLE_FPS: float = 5.0  # 必须与 mot_candidates.SAMPLE_FPS 一致
IMG_WIDTH: int = 1920  # 帧宽（高按原片宽高比自适应：16:9->1080、4:3->1440）
JPG_QUALITY: int = 3  # ffmpeg -q:v（2~5 对检测足够）
FFPROBE_TIMEOUT_SEC: int = 30  # rules.md §4：ffprobe 单文件 30s
FFPROBE_RETRY: int = 2  # rules.md §4：重试 2 次，退避 1s -> 2s
FRAME_COUNT_TOLERANCE: int = 2  # fps 滤镜端点取整误差；偏差在此内视为已抽过


def parse_argv() -> tuple[str, int]:
    """解析命令行参数。

    Returns:
        (原片目录, 处理上限；0 表示全部)。
    """
    srcdir: str = ""
    limit: int = 0
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif not args[i].startswith("--") and not srcdir:
            srcdir = args[i]
            i += 1
        else:
            i += 1
    return srcdir, limit


def probe_duration_sec(path: str) -> float:
    """ffprobe 取原片时长（秒）；30s 超时 + 2 次重试，退避 1s -> 2s（rules.md §4）。

    Args:
        path: 原片路径。

    Returns:
        时长（秒）。

    Raises:
        MediaTimeoutError: 超时重试耗尽。
        BasketballPipelineError: 非零退出或时长无法解析。
    """
    cmd: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    last_err: str = ""
    last_timeout: bool = False
    for attempt in range(1, FFPROBE_RETRY + 2):
        try:
            proc = subprocess.run(  # noqa: S603 固定 ffprobe 二进制，参数内部构造
                cmd,
                capture_output=True,
                # ffprobe 报错输出含中文文件名，locale 代码页会解码崩（同
                # pipe_common.run_ffmpeg 的 cp1252 坑）——钉死 UTF-8
                encoding="utf-8",
                errors="replace",
                timeout=FFPROBE_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            last_err = f"超时({FFPROBE_TIMEOUT_SEC}s)"
            last_timeout = True
        else:
            if proc.returncode == 0:
                try:
                    return float(proc.stdout.strip())
                except ValueError:
                    last_err = f"时长解析失败: {proc.stdout.strip()[:80]}"
                    last_timeout = False
            else:
                last_err = proc.stderr.strip()[-200:]
                last_timeout = False
        logger.warning("  ffprobe 第%d次失败: %s", attempt, last_err)
        if attempt <= FFPROBE_RETRY:
            time.sleep(float(attempt))
    if last_timeout:
        raise MediaTimeoutError(f"ffprobe 超时重试耗尽: {path}: {last_err}")
    raise BasketballPipelineError(f"ffprobe 失败: {path}: {last_err}")


def list_sources(srcdir: str) -> list[str]:
    """递归扫描原片（.mp4，大小写不敏感），按文件名升序（dji 命名即时间序）。

    Args:
        srcdir: 原片目录。

    Returns:
        原片路径列表。
    """
    found: list[str] = [
        p
        for p in glob(os.path.join(escape(srcdir), "**", "*"), recursive=True)
        if p.lower().endswith(".mp4")
    ]
    return sorted(found)


def count_frames(fid: str) -> int:
    """统计已抽帧数。

    Args:
        fid: 文件 ID。

    Returns:
        work/frames/<fid>/ 下的 f_*.jpg 数量。
    """
    return len(glob(os.path.join(FRAMES_ROOT, fid, "f_*.jpg")))


def extract_file(src: str, fid: str, duration: float) -> int:
    """对单个原片抽帧并校验产出帧数。

    Args:
        src: 原片路径。
        fid: 文件 ID（输出目录名）。
        duration: 原片时长（秒）。

    Returns:
        实际产出帧数。

    Raises:
        BasketballPipelineError: 产出 0 帧或与时长推算偏差超容差。
        MediaTimeoutError: ffmpeg 超时重试耗尽。
    """
    out_dir: str = os.path.join(FRAMES_ROOT, fid)
    os.makedirs(out_dir, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            src,
            "-map",
            "0:v:0",
            "-vf",
            f"fps={SAMPLE_FPS},scale={IMG_WIDTH}:-2,format=yuv420p",
            "-q:v",
            str(JPG_QUALITY),
            os.path.join(out_dir, "f_%05d.jpg"),
        ],
        timeout_sec=max(120, int(duration * 3) + 60),
    )
    n: int = count_frames(fid)
    expected: int = round(duration * SAMPLE_FPS)
    if n == 0:
        raise BasketballPipelineError(f"{fid}: 抽帧产出 0 帧（时长 {duration:.1f}s）")
    if abs(n - expected) > FRAME_COUNT_TOLERANCE:
        raise BasketballPipelineError(f"{fid}: 帧数 {n} 与时长推算 {expected} 偏差超容差")
    return n


def main() -> int:
    """主入口：批量抽帧，断点续做，逐文件隔离失败。

    Returns:
        进程退出码：0=全部成功；1=有失败文件；130=人工中断。
    """
    run_id: str = new_run_id()
    configure_logging(run_id)
    srcdir, limit = parse_argv()
    if not srcdir or not os.path.isdir(srcdir):
        logger.error("原片目录不存在: %r（用法: extract_frames.py <目录> [--limit N]）", srcdir)
        return 1
    sources: list[str] = list_sources(srcdir)
    if limit > 0:
        sources = sources[:limit]
    total: int = len(sources)
    logger.info("共 %d 个原片待抽帧 -> %s", total, FRAMES_ROOT)

    failed: list[str] = []
    done: int = 0
    skipped: int = 0
    for i, src in enumerate(sources, 1):
        fid: str = os.path.splitext(os.path.basename(src))[0]
        try:
            duration: float = probe_duration_sec(src)
            existing: int = count_frames(fid)
            if existing and abs(existing - round(duration * SAMPLE_FPS)) <= FRAME_COUNT_TOLERANCE:
                skipped += 1
                logger.info("  [%d/%d] %s 已有 %d 帧，跳过", i, total, fid, existing)
                continue
            t0: float = time.time()
            n: int = extract_file(src, fid, duration)
            done += 1
            logger.info(
                "  [%d/%d] %s %.1fs -> %d 帧 (耗时%.0fs)",
                i,
                total,
                fid,
                duration,
                n,
                time.time() - t0,
            )
        except (BasketballPipelineError, OSError) as exc:
            failed.append(fid)
            logger.error("  [%d/%d] %s 抽帧失败: %s", i, total, fid, exc)
        except KeyboardInterrupt:
            logger.warning(
                "中断，已处理 %d/%d（成功%d 跳过%d 失败%d），重跑可续",
                i,
                total,
                done,
                skipped,
                len(failed),
            )
            return 130
    logger.info("完成: 成功%d 跳过%d 失败%d / 共%d", done, skipped, len(failed), total)
    if failed:
        logger.error("失败清单: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
