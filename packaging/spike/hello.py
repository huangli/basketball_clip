"""SP-2 最小实包验证：模拟 basketball-clip 的 import 链。

打包后运行此 exe，能走完所有重库的 import 即证明 PyInstaller × 3.10
对该依赖组合的收集（hidden imports、数据文件）可行。
"""

import sys


def main() -> int:
    print(f"Python {sys.version}")
    steps = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("cv2 (opencv)", "cv2"),
        ("ultralytics", "ultralytics"),
        ("open_clip", "open_clip"),
        ("sklearn", "sklearn"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("PIL", "PIL"),
    ]
    for label, mod in steps:
        try:
            m = __import__(mod)
            print(f"OK   {label} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # spike 要看到每个失败现场，不静默
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            return 1
    # torch 实际算一把，防"能 import 不能跑"
    import torch

    x = torch.randn(3, 3)
    y = (x @ x).sum().item()
    print(f"OK   torch matmul -> {y:.4f}")
    print("SPIKE-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
