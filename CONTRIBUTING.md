# 贡献指南

basketball-clip 是一个篮球视频进球自动检测与集锦剪辑工具：机器检出疑似进球、网页上人工拍板、按队伍/个人自动合成集锦——**机器干活，人做裁判**。欢迎报 bug、提需求、交 PR。

## 报问题 / 提需求

统一走 [GitHub Issues](https://github.com/huangli/basketball_clip/issues)（唯一反馈渠道）。报 bug 请用 **Bug 反馈** 模板，并按模板指引附上 GUI 顶栏 **"导出诊断日志"** 按钮产出的诊断 zip。

## 环境搭建

要求：Windows（开发主力平台）、Python ≥ 3.10、ffmpeg + ffprobe 在 PATH 中（推荐 [BtbN 的 GPL 构建](https://github.com/BtbN/FFmpeg-Builds/releases)，须含 libx264）。

```powershell
git clone https://github.com/huangli/basketball_clip.git
cd basketball_clip
python -m venv .venv
.venv\Scripts\Activate.ps1
# torch 建议装 CPU 版（避免 PyPI 默认的 CUDA 大包）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy pillow open_clip_torch scikit-learn matplotlib httpx
# GUI 与测试所需：
pip install fastapi uvicorn pytest
# 开发工具：
pip install ruff
```

模型权重不随仓库分发，跑真实检测前需按 [README "模型权重" 一节](README.md#模型权重不随仓库分发) 自行准备；只跑测试不需要权重（单测不跑真推理）。

## 开发流程

1. Fork 本仓库，从默认分支 `main` 切出功能分支（如 `fix/build-crash`、`feat/new-metric`）。
2. 改代码，补测试（新增/改动逻辑必须有 pytest 用例锁定，见 [rules.md](rules.md) §9）。
3. 提交前跑本地关口，必须**全绿**：

   ```powershell
   python -m ruff format scripts tests gui
   python -m ruff check --fix scripts tests gui
   python -m pytest -q
   ```

   - `ruff` 装了命令行入口的话，`ruff ...` 与 `python -m ruff ...` 等价，用哪个都行。
   - `ruff check --fix` 自动修复后**必须人工复核 diff**，勿盲信。
   - 测试也可用 `pytest -q`（环境里有 pytest 命令行入口时）。
4. 推到自己 fork，向 `main` 发 PR。PR 描述写清楚：改了什么、为什么、本地关口结果。

## 代码规范

- 唯一权威是 [rules.md](rules.md)（**鲁棒优先 > 性能 > 简洁**），`scripts/` 下所有 Python 代码强制遵守；动手前先通读。
- 要点速览：全函数类型注解、Google 风格 docstring、禁止吞异常、外部 IO 显式超时重试、阈值集中常量、`logging` 禁用 `print`。
- Lint/format 以 [ruff.toml](ruff.toml) 为准（`target-version = "py310"`，请勿使用 3.11+ 语法）。
- 提交信息用**中文 conventional 风格**：`feat: …` / `fix: …` / `refactor: …` / `docs: …` / `test: …` / `chore: …` / `perf: …` / `ci: …`，参照 `git log` 现有提交。

## CLI 行为兼容红线

`scripts/` 下的命令行链路（`scripts/video.py` 五个子命令及其调用的脚本）是**对外契约**：普通用户的素材、中间产物（`work/`）与标注结果（`goals.json` / `roster.json`）都建立在现有行为上。因此：

- **同输入必须同产物**：不改 CLI 参数语义、产物文件的位置/格式/命名、默认值与判定阈值，除非 issue 中明确讨论并达成一致。
- 唯一例外是从私有工作区迁移时已定的**豁免清单**口径：配置外置为环境变量（代理/凭证）、队名可配置化、VLM 试验功能下线、OSNet 后端下线等——这些是一次性迁移事实，**不构成后续改动先例**。新改动若想触碰上述行为，请先开 issue 讨论，不要直接 PR。
- 内部实现（函数拆分、性能优化、日志措辞）只要不改变可观测行为与产物，欢迎改进，但需有测试证明行为不变。

## 许可

- 贡献的代码默认按本仓库 [MIT](LICENSE) 许可发布。
- 注意依赖侧许可：ultralytics 及 YOLO 权重为 AGPL-3.0，安装包内嵌的 ffmpeg 为 GPL 构建，详见 [README "许可声明"](README.md#许可声明)。引入新依赖前请先确认其许可与本项目兼容，并在 PR 中说明。
