"""管线共享工具：run_id 日志、ffmpeg 封装、JSON 原子写/校验读（rules.md §4/§8）。

输入：各脚本的 ffmpeg 参数、JSON 产物路径。
输出：日志配置、子进程执行结果、落盘 JSON 文件。
依赖：scripts/errors.py 的异常层级。
典型调用：
    run_id = new_run_id()
    configure_logging(run_id)
    run_ffmpeg(["-i", src, ...], timeout_sec=120)
    atomic_write_json(path, payload, what="candidates.json")
"""

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from errors import BasketballPipelineError, MediaTimeoutError, SchemaError

logger = logging.getLogger(__name__)

# JSON 读写重试退避序列（rules.md §4：3 次重试，0.5s → 1s → 2s）
JSON_RETRY_BACKOFF_SEC: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)


class RunIdFilter(logging.Filter):
    """把 run_id 注入每条日志记录，供格式串 %(run_id)s 使用。"""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def new_run_id() -> str:
    """生成 uuid4 短形 run_id。"""
    return uuid.uuid4().hex[:8]


def configure_logging(run_id: str) -> None:
    """在进程入口配置带 run_id 的 root logger（rules.md §8）。

    只在 main() 调用一次；RunIdFilter 挂到 root handler 上，
    保证所有模块 logger 的记录都带 run_id（否则 %(run_id)s 抛 KeyError）。

    Args:
        run_id: new_run_id() 生成的短 id。
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] run=%(run_id)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RunIdFilter(run_id))


def run_ffmpeg(
    args: list[str],
    *,
    timeout_sec: int = 600,
    retries: int = 1,
    backoff_sec: float = 5.0,
) -> None:
    """执行 ffmpeg，带显式超时、有限重试与退避（rules.md §4）。

    Args:
        args: ffmpeg 参数列表（不含 ffmpeg 本体）。
        timeout_sec: 单次执行超时（秒）。
        retries: 失败后的额外重试次数（总尝试 = retries + 1）。
        backoff_sec: 重试前退避秒数。

    Raises:
        MediaTimeoutError: 超时且重试耗尽。
        BasketballPipelineError: 非零退出且重试耗尽。
    """
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    last_err: str = ""
    last_timeout: bool = False
    for attempt in range(1, retries + 2):
        try:
            proc = subprocess.run(  # noqa: S603 固定 ffmpeg 二进制，参数内部构造
                cmd,
                capture_output=True,
                # ffmpeg 输出为 UTF-8（含中文路径/文件名）；locale 代码页（本机
                # cp1252）遇 0x81 等字节会让 reader 线程 UnicodeDecodeError 崩溃，
                # 进而导致句柄释放竞态（2026-08-22 某场次真机实录）——钉死 UTF-8
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            last_err = f"超时({timeout_sec}s)"
            last_timeout = True
        else:
            if proc.returncode == 0:
                return
            last_err = proc.stderr.strip()[-300:]
            last_timeout = False
        logger.warning("ffmpeg 第%d/%d次失败: %s", attempt, retries + 1, last_err)
        if attempt <= retries:
            time.sleep(backoff_sec)
    if last_timeout:
        raise MediaTimeoutError(f"ffmpeg 超时重试耗尽: {last_err}")
    raise BasketballPipelineError(f"ffmpeg 重试耗尽: {last_err}")


def read_json(path: str | Path, *, what: str = "JSON") -> Any:  # noqa: ANN401 JSON 内容不定
    """读取 JSON 文件；OSError 有限重试，解析失败抛 SchemaError（rules.md §0.2/§4）。

    Args:
        path: 文件路径。
        what: 业务名称（用于错误信息，如 "goals.json"）。

    Returns:
        解析后的 JSON 对象。

    Raises:
        SchemaError: 内容不是合法 JSON（数据损坏必须停，不重试）。
        OSError: IO 重试耗尽。
    """
    last: OSError | None = None
    for wait in JSON_RETRY_BACKOFF_SEC:
        if wait:
            time.sleep(wait)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{what} 损坏: {path}: {exc}") from exc
        except OSError as exc:
            last = exc
            logger.warning("%s 读取失败(%s)，重试", what, exc)
    if last is not None:
        raise last
    raise OSError(f"{what} 读取失败: {path}")


def atomic_write_json(path: str | Path, data: Any, *, what: str = "JSON") -> None:  # noqa: ANN401
    """原子写 JSON：先写 .tmp 再 os.replace，OSError 有限重试（rules.md §4）。

    避免崩溃留下半截文件；下游读到的一定是完整 JSON 或旧版本。

    Args:
        path: 目标文件路径。
        data: 可 JSON 序列化的对象。
        what: 业务名称（用于错误信息）。

    Raises:
        OSError: IO 重试耗尽。
    """
    tmp: str = f"{path}.tmp"
    last: OSError | None = None
    for wait in JSON_RETRY_BACKOFF_SEC:
        if wait:
            time.sleep(wait)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last = exc
            logger.warning("%s 写入失败(%s)，重试", what, exc)
    if last is not None:
        raise last
    raise OSError(f"{what} 写入失败: {path}")
