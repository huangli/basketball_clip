"""VLM 读号客户端（--read-numbers 专用，crop_scorers 延迟导入）。

输入：投篮者裁图（PIL Image）与环境变量凭证。
输出：base64 JPEG 编码图、Bearer token、API 端点/重试口径常量。
依赖：标准库 + PIL。
典型调用：``from vlm_client import API_URL, MODEL, crop_to_b64, load_token``

凭证契约（开源口径，不绑定任何个人订阅设施）：
- ``BASKETBALL_CLIP_VLM_TOKEN``：Bearer token 直给（必填，缺失显式失败）；
- ``BASKETBALL_CLIP_VLM_API_URL``：OpenAI 兼容 chat/completions 端点覆盖
  （可选，缺省为 DEFAULT_API_URL）。
"""

from __future__ import annotations

import base64
import io
import os

from PIL import Image

ENV_TOKEN: str = "BASKETBALL_CLIP_VLM_TOKEN"  # noqa: S105 环境变量名，非口令
ENV_API_URL: str = "BASKETBALL_CLIP_VLM_API_URL"
DEFAULT_API_URL: str = "https://api.kimi.com/coding/v1/chat/completions"
API_URL: str = os.environ.get(ENV_API_URL) or DEFAULT_API_URL
MODEL: str = "k3"

IMG_SIZE: int = 840  # 发给 VLM 的图边长（448 时球仅 ~8px，VLM 会漏看）
HTTP_TIMEOUT_SEC: int = 180  # VLM 推理较慢，单次给足余量
HTTP_RETRY: int = 2


def load_token(force: bool = False) -> str:
    """从环境变量读取 VLM Bearer token（不打印）。

    无缓存：环境变量是外部事实，每次直读；force 仅为调用方
    401 后重试口径的签名兼容（语义上同普通调用）。

    Args:
        force: 保留参数（401 后强制重读口径），无额外行为。

    Returns:
        token 字符串。

    Raises:
        RuntimeError: 环境变量未设置或为空。
    """
    token: str = os.environ.get(ENV_TOKEN, "").strip()
    if not token:
        raise RuntimeError(f"读号需配置 VLM 凭证：设置环境变量 {ENV_TOKEN}")
    return token


def crop_to_b64(img: Image.Image) -> str:
    """图像缩放为 IMG_SIZE 边长并编码 base64 JPEG。

    Args:
        img: 原图。

    Returns:
        base64 字符串。
    """
    small = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")
