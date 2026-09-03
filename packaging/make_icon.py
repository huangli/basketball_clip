"""生成安装器/应用通用图标 ``packaging/icon.ico``（篮球：橙底黑纹）。

输入：无（纯程序绘制，不用现工作区 resource/logo.png——开源口径要求新做通用图标）。
输出：``packaging/icon.ico``（多尺寸 16-256，透明背景）。
依赖：pillow（用打包 venv .venv-spike 运行）。
典型调用::

    .venv-spike/Scripts/python.exe packaging/make_icon.py

图标为构建源资产，生成一次后随仓库提交；改图案时改本文件重跑即可。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ICON_PATH: Path = Path(__file__).resolve().parent / "icon.ico"
CANVAS: int = 1024  # 超采样画布，缩小后抗锯齿
SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)

BALL_COLOR: tuple[int, int, int, int] = (240, 118, 43, 255)  # 篮球橙
SEAM_COLOR: tuple[int, int, int, int] = (38, 26, 20, 255)  # 深棕黑纹路


def draw_ball(size: int) -> Image.Image:
    """在给定尺寸画布上画篮球（先大画布绘制再缩小，保证小尺寸清晰）。"""
    scale = CANVAS // size
    big = size * scale
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = big // 32
    box = (pad, pad, big - pad, big - pad)
    draw.ellipse(box, fill=BALL_COLOR)
    seam = max(big // 42, 1)
    # 外轮廓 + 十字缝
    draw.ellipse(box, outline=SEAM_COLOR, width=seam)
    cx, cy = big // 2, big // 2
    r = big // 2 - pad
    draw.line((cx - r, cy, cx + r, cy), fill=SEAM_COLOR, width=seam)
    draw.line((cx, cy - r, cx, cy + r), fill=SEAM_COLOR, width=seam)
    # 两侧弧形缝（用裁切椭圆弧近似经典篮球纹路）
    arc_r = int(r * 1.5)
    for ox in (-arc_r, arc_r):
        draw.arc(
            (cx + ox - arc_r, pad, cx + ox + arc_r, big - pad),
            start=0,
            end=360,
            fill=SEAM_COLOR,
            width=seam,
        )
    # 球外多余弧线裁掉：重建圆形蒙版
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse(box, fill=255)
    out = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out.resize((size, size), Image.LANCZOS)


def main() -> int:
    """生成多尺寸 icon.ico。返回退出码（0=成功）。"""
    images = [draw_ball(s) for s in SIZES]
    images[-1].save(ICON_PATH, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"icon.ico written: {ICON_PATH} sizes={SIZES}")  # noqa: T201 脚本型输出
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
