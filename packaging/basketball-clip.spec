# -*- mode: python ; coding: utf-8 -*-
"""basketball-clip 全量打包 spec（E-1）。

- one-folder，入口 gui/__main__.py（frozen 分发/环境注入见 gui/frozen.py）；
- collect-all 基线照抄 spike（packaging/spike/spike-hello.spec，spike-report.md 口径）：
  cv2 / ultralytics / open_clip / timm / sklearn / scipy / uvicorn / PIL 共 8 个；
- datas：gui/static/ 前端 + assets（BtbN ffmpeg 含许可文本、CLIP HF 缓存）；
- scripts/ 与 models/yolov8n.pt 不走 datas——需在 exe 同级目录（用户可写、
  video.py relocate chdir 基准），由 build_installer.py 打包后拷贝落地；
- upx=False：1GB+ 包 UPX 压缩耗时陡增且易触发杀软误报，体积收益不成比例；
- copy_metadata：补关键包 dist-info，frozen 态 diagnostics 版本表不落成"未安装"。

运行口径：路径以 SPECPATH（spec 所在目录 packaging/）的上级 = 仓库根锚定，任意 cwd 可跑。
"""

import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

REPO_ROOT = os.path.dirname(SPECPATH)

datas = [
    (os.path.join(REPO_ROOT, "gui", "static"), "gui/static"),
    (os.path.join(REPO_ROOT, "packaging", "assets", "ffmpeg"), "assets/ffmpeg"),
    (os.path.join(REPO_ROOT, "packaging", "assets", "clip"), "assets/clip"),
]
binaries = []
hiddenimports = []

for package in ("cv2", "ultralytics", "open_clip", "timm", "sklearn", "scipy", "uvicorn", "PIL"):
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

# 诊断导出的依赖版本表（gui/diagnostics.py KEY_PACKAGES）在 frozen 态可读；
# 未安装的包（如已下线的 torchreid）跳过，不影响打包
for dist in (
    "fastapi",
    "uvicorn",
    "torch",
    "ultralytics",
    "opencv-python",
    "open_clip_torch",
    "numpy",
    "pillow",
    "scikit-learn",
):
    try:
        datas += copy_metadata(dist)
    except Exception:  # noqa: BLE001 — 打包辅助步骤，缺失包显式跳过
        print(f"[spec] copy_metadata 跳过（未安装）: {dist}")

a = Analysis(
    [os.path.join(REPO_ROOT, "gui", "__main__.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="basketball-clip",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="basketball-clip",
)
