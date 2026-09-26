# AGENTS.md — basketball-clip 项目级 Agent 指南

> 本文件是给 AI 编码助手的常驻上下文。它是**索引与红线**，不是细节规范：
> 代码规范的完整定义在 [rules.md](rules.md)，lint/format 权威在 [ruff.toml](ruff.toml)，
> 贡献流程与 CLI 兼容红线在 [CONTRIBUTING.md](CONTRIBUTING.md)，用法与许可在 [README.md](README.md)。
> 动手前先读对应文件的相关章节；本文件与上述文件冲突时，以**详细规范文件**为准。

## 项目是什么

篮球视频进球自动检测与集锦剪辑工具（Python ≥ 3.10）。流程：**机器干活，人做裁判**——
YOLO 检测 → 网页人工标注 → 可选认人（CLIP 聚类/球衣读号/人脸匹配）→ 队伍/个人合集合成 → 照片精选。

## 目录结构

| 路径 | 作用 |
|---|---|
| `scripts/` | 管线脚本，统一 CLI 入口 `scripts/video.py`（score/people/build/photo/clean 五个子命令） |
| `gui/` | FastAPI 本地向导（`python -m gui`），浏览器页面在 `gui/static/` |
| `tests/` | pytest 用例，与 `scripts/` 平级镜像，另有 `tests/gui/`、`tests/packaging/` |
| `packaging/` | Windows 安装包构建（PyInstaller + Inno Setup） |
| `models/` | 模型权重，**被 .gitignore 排除，不入库** |
| `work/`、`output/` | 运行时中间产物与成品，**被 .gitignore 排除** |
| `20260829第六人/` 等中文目录 | 用户原始球赛素材，属隐私数据 |

## 开发流程（强制，不可跳过）

任何代码/行为修改（含本文件自身的修改），动手前必须走完三件套 + 评审：

1. **spec**（规格）：写清楚改什么、为什么、验收标准、不做什么（边界）。
2. **plan**（计划）：实现步骤、涉及文件、风险与回滚方式。
3. **task**（任务分解）：可勾选的任务清单，作为执行与进度追踪依据。
4. **subagent review 意见统一**：三件套须交 subagent 评审，所有评审意见**达成统一**（无未解决的反对意见）后，才能开始写代码。

琐碎修改（typo、注释修正、纯文档措辞）可豁免三件套，但判定从紧：拿不准就走流程。

### 文档存放与编号

- 过程文档统一放 `docs/` 下，**每件事一个子目录**，命名 `NNN-功能缩写/`
  （`NNN` 从 001 起递增的三位数字，如 `docs/002-goal-detect-v2/`），方便按序查看。
- 子目录内文件固定为 `spec.md`、`plan.md`、`task.md`；评审记录追加 `review01.md`、`review02.md`…
  （参考既有 `docs/build-installer-concurrency-lock/` 的结构，该目录早于编号规则，保留原名不改）。
- 新文档的编号取 `docs/` 下现有最大编号 +1，先建目录再写内容。

## 质量关口（每次提交前必须本地全绿）

```bash
python -m ruff format scripts tests gui
python -m ruff check --fix scripts tests gui
python -m pytest -q
```

- `ruff check --fix` 自动修复后**必须人工复核 diff**，勿盲信。
- 仓库无 CI 工作流，本地关口就是唯一关口，不能跳过。

## 代码风格要点（详见 rules.md，以下为速览）

- **决策优先级**：鲁棒性与可回归性 ＞ 性能 ＞ 代码简洁。
- 所有函数**强制类型注解**（入参 + 返回值），用新语法 `list[int]` / `X | None`。
- 公共函数/类/模块必须有 **Google 风格 docstring**。
- **禁止 `print`**，统一 `logging` 并注入 `run_id`；**禁止吞异常**，只捕获具体异常类型。
- 外部 IO（ffprobe/ffmpeg 子进程、JSON 读写）必须**显式超时 + 有限重试**；JSON 用原子写。
- 阈值/参数集中到模块顶部常量区，禁止散落魔法数字。
- 可执行脚本必须有 `if __name__ == "__main__":` 守卫，禁止模块级副作用。
- 路径一律 `pathlib.Path` + 正斜杠，不写 Windows 反斜杠字符串。
- 复杂数据用 `dataclass` / `TypedDict` 建模，禁止裸传 `dict[str, Any]`。

## Ruff 使用（唯一 lint/format 权威）

- 配置在仓库根 `ruff.toml`：`line-length = 100`，`target-version = "py310"`（**勿用 3.11+ 语法**）。
- 启用规则集：`E W F I UP B SIM ANN RUF T20 S`（`ANN` 类型注解、`T20` 禁 print、`S` 安全检查是项目硬约束，必须保留）。
- 豁免：`tests/**` 豁免 `S101`（assert），`packaging/spike/**` 豁免 `T20`；中文标点歧义 RUF001/002/003 全局豁免。
- 排除目录：`archive`、`.codewale`、`.opencode`、`.git`。
- 修改 ruff.toml 的规则集/豁免前须评估影响面（全库扫描可能变红），并在提交信息中说明理由。

## 风险校验红线（不可妥协）

### 密钥与凭证

- **绝不硬编码任何密钥/token/密码**。凭证只走环境变量：`BASKETBALL_CLIP_VLM_TOKEN`、
  `BASKETBALL_CLIP_VLM_API_URL`、`BASKETBALL_CLIP_HTTPS_PROXY`。
- `.env` 类文件不入库；发现凭证被写入代码或日志，先停下清除，再排查同类问题。
- 错误信息与日志**不泄露内部细节**：路径、堆栈、凭证不得出现在面向用户的报错里（诊断日志 zip 除外，其内容由用户自行决定提交）。

### 用户隐私数据（本项目特有，重点）

- 用户的球赛视频、抽帧照片、`work/` 下的 `goals.json`/`roster.json`（含球员姓名、人脸裁图、球衣号）
  均为**个人数据**：不上传、不外发、不写入日志以外的任何远端通道。
- `--photo-match` 人脸匹配与 `--read-numbers` 球衣读号是**显式 opt-in** 功能，保持默认关闭；
  不得改为默认开启或静默联网调用。
- 网络调用只允许出现在既有约定位置（CLIP 权重下载、球检测模型首运行下载（固定 Google Drive 直链，见 scripts/model_fetch.py）、VLM 读号接口），新增任何网络出口必须先与用户确认。
- 提交代码/发 PR 前确认：diff 中不含用户素材路径、真实人名、样例视频帧截图等隐私内容。

### 产物与许可边界

- `models/` 权重**不随仓库/安装包再分发**（尤其 `abdullahtarek_ball.pt` 再分发权利未确认）。
- 依赖许可约束：ultralytics/YOLO 为 **AGPL-3.0**、安装包内嵌 ffmpeg 为 **GPL 构建**（含 libx264，
  不得换成含 nonfree 组件的构建）、insightface buffalo_l 为**非商用许可**。
- 引入新依赖前确认许可兼容，并在 PR 中说明。

## CLI 行为兼容红线（来自 CONTRIBUTING.md）

`scripts/video.py` 五个子命令及产物文件（位置/格式/命名/默认值/阈值）是**对外契约**：
同输入必须同产物。内部重构（函数拆分、性能优化、日志措辞）可以改，但需有测试证明可观测行为不变。

## Git 规范

- 提交信息用**中文 conventional 风格**：`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:` / `perf:` / `ci:`，参照 `git log`。
- 测试约定：pytest + AAA 结构，新增/改动逻辑必须有用例锁定；YOLO/ffmpeg 等慢外部用 fixture 或 mock，单测不跑真推理。

## Agent 工作方式建议

- 改 `scripts/` 下任何文件前，先读 rules.md 对应章节与目标文件的现有 docstring，沿用既有分层异常
  （`scripts/errors.py`）与常量风格。
- 上下文冲突时（规范 vs 现有代码），**显式向用户提出**，不要自行取舍。
- 完成后自检：ruff format / ruff check / pytest 三连的输出，以及 diff 中无密钥、无隐私数据、无 CLI 行为变更（除非已获确认）。
