# rules.md — basketball-clip 代码规范

> 本文档定义"如何写出鲁棒、可回归的 Python 代码"。
> 所有 `scripts/` 下的 `.py` **强制**遵守本文档。
>
> **决策优先级**：鲁棒性与可回归性 ＞ 性能 ＞ 代码简洁。三者冲突时，一律优先保证"**可定位、可恢复、可验证**"。

---

## 0. 总纲：鲁棒优先（强制，不可妥协）

| # | 原则 | 在本项目的具体含义 |
|---|------|-------------------|
| 0.1 | **决策优先级** | 与性能冲突时选鲁棒。例：宁可每帧 `ffprobe` 显式确认帧率/位深，也不要为省 200ms 而假设"所有原片都是 50fps"。 |
| 0.2 | **输入与数据** | 进入业务逻辑前必须校验/归一/建模。`goals.json`/`roster.json` 读入后先过 schema 校验；视频元数据先 `ffprobe` 归一为 `VideoMeta` 结构体。**宁可显式失败，也不允许静默容错污染下游数据。** |
| 0.3 | **外部 IO** | `ffprobe`/`ffmpeg` 子进程、`goals.json`/`roster.json` 读写均须**显式超时 + 有限重试（含退避）**。失败要可观测、可回溯，**禁止无限重试与无边界等待**。 |
| 0.4 | **错误处理** | **禁止吞异常**、禁止"只 `print` 不抛"。日志必须含：错误类型、错误信息、关键业务参数、`run_id`。 |
| 0.5 | **可验证性** | 新增/重构必须覆盖关键路径、边界条件、失败场景，用 pytest 锁定。不依赖人工回归。 |
| 0.6 | **文档书写** | 函数/类/模块/文件须含工业级 docstring（见 §7）。 |

### "容忍缺失" ≠ "静默容错"（0.2 输入校验的细则）

"容忍缺失"指**素材增删属业务可预期**——缺文件记 `WARNING` 并跳过即可。但以下情形**必须显式失败或可观测（非零退出）**，不可静默跳过：

- `goals.json`/`roster.json` 存在但 **schema 损坏**（字段缺失/类型错误）→ 抛 `SchemaError`
- ffprobe 返回**非预期编码**（如非 H.265/HEVC 的原片）→ 抛 `UnsupportedMediaError`
- JSON 主键指向的文件**已不存在**，但被某条 `goal` 引用 → 分两类：
  - 检测/标注等**数据流水线** → 抛 `DanglingReferenceError`（除非该 goal 已标记 `removed`；勿自造状态）
  - 合集合成等**产出型脚本** → 逐条 `ERROR` 记录缺失并跳过，其余片段照常产出，但**进程退出码非零**（可观测、不静默，也不浪费已有产出）

判断口径：**自然删减可跳过，数据损坏必须停。**

---

## 1. 代码风格与格式

- **PEP 8**，4 空格缩进，UTF-8（无 BOM），文件末尾留一个空行。
- **一行只写一条语句**：禁止用 `;` 串接多条语句。`x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])` ❌ → 分两行 ✅。
- **单行 ≤ 100 字符**（中文按字符计，不按字节）。
- **导入分组**：标准库 / 第三方 / 本项目，组间空行；禁止 `import *`。
- **命名**：
  - `snake_case`：函数、变量、模块
  - `PascalCase`：类
  - `UPPER_SNAKE_CASE`：模块级常量
  - 私有以单下划线前缀 `_private`
- **`if __name__ == "__main__":` 守卫**：所有可执行脚本入口必须包裹，禁止脚本级副作用代码（如模块加载即加载 YOLO 模型、即扫描目录）。
- **代码内路径**：一律用正斜杠 `/` + `pathlib.Path`（跨平台、无需转义），不要在 `.py` 里写 Windows 反斜杠字符串。

---

## 2. 类型注解（强制）

- **所有函数**必须标注**入参与返回值**类型。
- 优先用标准库 `typing` / `collections.abc`（遵循 Ruff 的 `UP` 规则，用新语法：`list[int]` 而非 `List[int]`，`X | None` 而非 `Optional[X]`）。
- 复杂数据用 `dataclass` / `TypedDict` 建模，禁止到处传 `dict[str, Any]`。

```python
# ✅
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Box:
    """轴对齐边界框（像素坐标）。"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        """边界框中心 X 坐标（像素）。"""
        return (self.x1 + self.x2) // 2

def iou(a: Box, b: Box) -> float:
    """两框交并比，无相交返回 0.0。"""
    ...
```

ffprobe 元数据归一为结构体（禁止到处传 `dict[str, Any]`）：

```python
@dataclass(frozen=True, slots=True)
class VideoMeta:
    """单个源视频经 ffprobe 归一后的元数据。"""

    path: pathlib.Path
    width: int
    height: int
    fps: float              # 50 / 100 等混存，必须逐文件确认
    codec: str              # 原片应为 hevc；LRF 为 h264
    bit_depth: int          # 8 / 10 混存
    has_audio: bool
```

```python
# ❌ 反例（无类型、裸 dict）
def iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])  # ; 串语句
    ...
```

---

## 3. 错误处理

### 3.1 自定义异常层级（集中定义）

```python
# scripts/errors.py（统一异常模块）
class BasketballPipelineError(Exception):
    """所有管线异常的基类，携带 run_id 便于回溯。"""

    def __init__(self, message: str, *, run_id: str = "") -> None:
        super().__init__(message)
        self.run_id = run_id

class SchemaError(BasketballPipelineError): ...
class UnsupportedMediaError(BasketballPipelineError): ...
class DanglingReferenceError(BasketballPipelineError): ...
class MediaTimeoutError(BasketballPipelineError): ...  # 覆盖 ffprobe / ffmpeg 子进程超时
```

> 子类异常构造时传入 `run_id`：`raise SchemaError(f"{path}: 缺字段 {f}", run_id=run_id)`。

### 3.2 规则

- **禁止裸 `except:` 与 `except Exception:` 吞掉一切**；只能捕获具体异常类型。
- 捕获后**要么处理、要么重抛（带上下文 `raise ... from e`）、要么转换为项目异常**，禁止"只 log 不抛"。
- 日志必须含 `run_id`（见 §8）、错误类型、错误信息、关键业务参数（文件名、帧索引、时间戳等）。
- **禁止 `assert` 做业务校验**（`assert` 在 `-O` 下会被剔除）；业务校验用显式 `if ...: raise`。

---

## 4. 外部 IO（ffmpeg / ffprobe / JSON）

| 操作 | 超时 | 重试 | 退避 |
|------|------|------|------|
| `ffprobe` 单文件 | 30s | 2 次 | 1s → 2s |
| `ffmpeg` 转码（按文件） | 按时长 ×3 + 60s 兜底 | 1 次 | 5s |
| `goals.json` / `roster.json` 读写 | 5s | 3 次 | 0.5s → 1s → 2s |
| YOLO 单帧推理（进程内 CPU） | 不适用 | 不重试 | — |

> YOLO 是进程内 CPU 调用，不适用子进程超时；单帧推理超 30s 记 `WARNING`（CPU 秒级/帧为常态，30s 提示异常卡顿）。

- **原子写 JSON**：先写 `goals.json.tmp`，校验可解析后再 `os.replace` 重命名，避免崩溃产生半截文件。
- **重试只在可重试错误上发生**：`FileNotFoundError`/`PermissionError`/`TimeoutExpired` 可重试；`SchemaError`/`UnsupportedMediaError` 不重试（重试也不会成功）。
- YOLO 是进程内 CPU 调用，不适用子进程超时，但须有**进度日志**（每 N 帧输出一次）与**可中断**（`KeyboardInterrupt` 时 flush 已有结果）。

---

## 5. 配置与魔法数字

- 所有阈值/参数**集中到模块顶部常量区或 `config.py`**，禁止散落在逻辑里。
- 命名 `UPPER_SNAKE_CASE`，带行内注释说明来源。

```python
# scripts/config.py
# YOLO 篮球检测阈值（经实测：低于 0.04 漏检骤增）
BALL_CONF_THRESHOLD: float = 0.04
BALL_DETECT_IMGSZ: int = 1280
# 静止段判定：连续 4 帧、中心位移 <40px 视为入网静止点
STILL_WINDOW_FRAMES: int = 4
STILL_MAX_DISPLACEMENT_PX: int = 40
# 死球过滤：静止持续 >3s 视为持球/死球，排除
# ⚠ 沿用旧阈值，后续试点（静止点+conf 谷底判据）后需复核
DEAD_BALL_MAX_DURATION_S: float = 3.0
```

---

## 6. 模块与函数结构

- **单一职责**：一个函数做一件事；超 50 行考虑拆分。
- **入口清晰**：

```python
def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功，非 0=失败）。"""
    args = _parse_args(argv)
    run_id = _new_run_id()
    try:
        _run_pipeline(args, run_id=run_id)
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- 业务逻辑与 CLI 解析、日志配置分离，便于测试。

---

## 7. 文档与 docstring（工业级，强制）

每个**公共函数/类/模块**必须有 docstring，采用 **Google 风格**：

```python
def detect_stationary_segments(
    detections: list[FrameDetection],
    *,
    window: int = STILL_WINDOW_FRAMES,
    max_displacement_px: int = STILL_MAX_DISPLACEMENT_PX,
) -> list[Segment]:
    """从逐帧检测中找出疑似入网的静止段。

    通过滑动窗口检查连续若干帧篮球中心点位移是否低于阈值，
    合并相邻起始点得到静止段候选。

    Args:
        detections: 按时间升序的逐帧检测结果，允许 ball 为 None 的帧。
        window: 滑动窗口帧数（默认 4）。
        max_displacement_px: 窗口内中心点 X/Y 最大允许位移（像素）。

    Returns:
        合并后的静止段列表，按起始时间升序。

    Raises:
        ValueError: 当 detections 为空或 window < 1。
    """
    ...
```

模块级 docstring 置于文件首行 `from __future__` / docstring 之后，说明：用途、输入产物、输出产物、依赖、典型调用方式。

类 docstring 说明职责、是否可变（frozen/线程安全）、关键不变式。

模块级 docstring 是**文件第一条语句**（在任何 `from __future__` / import **之前**——Python 规则），内容说明：用途、输入产物、输出产物、依赖、典型调用方式。在 Python 中模块 docstring 即文件级 docstring（§0.6 所述"文件"层，与模块层同一字符串）。

```python
"""静止段检测：从逐帧 YOLO 检测中提取疑似入网静止段。

输入：work/detect/<fid>/detections.json（逐帧检测）
输出：work/candidates/<fid>/segments.json（静止段候选）
依赖：scripts/config.py 的阈值常量
典型调用：python scripts/detect_stationary.py 0011 0020
"""
from __future__ import annotations
```

---

## 8. 日志（用 logging，禁用 print）

- **禁止用 `print` 做诊断输出**（除非面向用户的最终结果，如候选清单）。
- 统一用 `logging`；每进程在 `main()` 内生成 `run_id`（uuid4 短形），通过 `logging.Filter` 注入每条 `LogRecord`，再在格式串里用 `%(run_id)s`。**不可直接在格式串写 `%(run_id)s` 而不注入**（会抛 `KeyError`）。

```python
import logging
import uuid


class RunIdFilter(logging.Filter):
    """把 run_id 注入每条日志记录，供格式串 `%(run_id)s` 使用。"""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def configure_logging(run_id: str) -> logging.Logger:
    """配置带 run_id 的 root logger，返回项目 logger。"""
    logger = logging.getLogger("basketball_clip")
    logger.addFilter(RunIdFilter(run_id))
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] run=%(run_id)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    return logger
```

- `basicConfig`/`addFilter` 只在进程入口（`main`）调用一次，**禁止在模块顶层配置 logging**（与 §6 入口守卫一致）。
- 级别约定：`INFO` 进度/决策、`WARNING` 跳过素材等可恢复情况、`ERROR` 失败、`DEBUG` 逐帧细节（默认关闭）。
- 大量循环输出用 `logger.debug` 并按 N 帧采样，避免日志爆炸。

---

## 9. 测试（pytest）

- 目录：`tests/`，与 `scripts/` 同级。
- 命名：文件 `test_*.py`，函数 `test_*`，类 `Test*`。
- 结构遵循 **AAA**（Arrange / Act / Assert）。
- 用 **fixture** 解耦环境依赖：提供 `tmp_work_dir`、`sample_video_meta`、`fake_ball_detections` 等。
- 必须覆盖：关键路径（静止段合并、IoU、入网点判定）、边界条件（空输入、单帧、满窗口）、失败场景（schema 损坏、ffprobe 超时）。
- YOLO/ffmpeg 等"慢外部"用 **fixture 提供固定小样本或 mock**，不在单测里跑真推理。

```python
def test_iou_no_overlap_returns_zero() -> None:
    # Arrange
    a, b = Box(0, 0, 10, 10), Box(20, 20, 30, 30)
    # Act
    score = iou(a, b)
    # Assert
    assert score == 0.0
```

---

## 10. Lint / Format 权威：Ruff

- **Ruff 是唯一 lint/format 权威**。配置见仓库根 `ruff.toml`。
- 规则集：`E` `W` `F` `I`（基础）+ `UP`（现代化）+ `B`（bugbear）+ `SIM`（简化）+ `ANN`（注解）+ `RUF`（Ruff 原生）。**注意**：Ruff 默认规则集**不含 `ANN`**，因此 §2"函数强制类型注解"在未配置时无自动校验——**必须在 `ruff.toml` 显式启用 `ANN`** 才能强制。
- 格式化统一用 `ruff format`；检查用 `ruff check`；提交前可用 `ruff check --fix` 自动修复（**修复后须人工复核 diff**，勿盲信）。

本地提交前三连（CI 中去掉 `--fix`）：

```powershell
ruff format scripts tests
ruff check --fix scripts tests
pytest -q
```

---

## 附录 A：典型反例对照

| 问题点 | 反例（❌） | 应为（✅，依据） |
|--------|-----------|------------------|
| 模块级副作用 | 顶部即 `YOLO(...)` 加载模型 | 放入函数/`main`，加 `if __name__`（§6） |
| 无类型 | `def iou(b1, b2):` | `def iou(a: Box, b: Box) -> float:`（§2） |
| `;` 串语句 | `x1=max(...); y1=max(...)` | 分行（§1） |
| 魔法数字 | `conf=0.04`、`<40`、`>3.0` 散落 | 提为 `config.py` 常量（§5） |
| 无异常保护 | YOLO/`glob` 无 try | 分层异常 + run_id（§3、§4、§8） |
| 用 print | 全程 `print(f"...")` | `logging`（§8） |
| 无 docstring | 函数无说明 | Google 风格 docstring（§7） |
| 无测试 | 无 | pytest + AAA（§9） |
| 无重试/超时 | 隐式假设文件都在 | ffprobe/json 带超时重试（§4） |
