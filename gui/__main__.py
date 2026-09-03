"""``python -m gui`` 启动入口：uvicorn 起服务 + 自动开浏览器。

端口默认 8471，被占则递增探测（最多 20 个）；仅绑定 127.0.0.1（本机工具，
不对局域网暴露）。
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import webbrowser

import uvicorn

from gui import frozen

logger = logging.getLogger(__name__)

HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8471
PORT_TRIES: int = 20
# 开浏览器延迟（秒）：给 uvicorn 起服务的缓冲，避免浏览器先到拒连
OPEN_BROWSER_DELAY: float = 1.0


def _find_free_port(host: str = HOST, start: int = DEFAULT_PORT, tries: int = PORT_TRIES) -> int:
    """从 start 起递增探测可用端口；全占用显式失败（不静默选随机口）。

    Raises:
        RuntimeError: tries 个端口全部被占。
    """
    for offset in range(tries):
        port = start + offset
        with socket.socket() as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"端口 {start}~{start + tries - 1} 全部被占用，无法启动 GUI")


def main() -> None:
    """启动 GUI 服务（阻塞至 Ctrl+C）。

    frozen（PyInstaller）态先做两件事：multiprocessing spawn 守卫；argv[1] 为
    scripts/ 内 .py 时 exe 充当 Python 解释器分发执行（runner 以 sys.executable
    起子进程的既有口径不变），分发命中则不起 GUI。
    """
    frozen.freeze_support_guard()
    if frozen.dispatch_script(sys.argv):
        return
    frozen.bootstrap_env()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    port = _find_free_port()
    url = f"http://{HOST}:{port}"
    logger.info("basketball-clip GUI 启动: %s", url)
    threading.Timer(OPEN_BROWSER_DELAY, lambda: webbrowser.open(url)).start()
    from gui.app import create_app  # 延迟 import：--help 等场景不白付 fastapi 加载

    uvicorn.run(create_app(), host=HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
