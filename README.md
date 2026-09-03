# basketball-clip

篮球视频进球自动检测与集锦剪辑工具：把球赛录像喂给它，机器检出疑似进球并排好序端到网页上，你按键拍板，最后按队伍/个人自动合成集锦，还能挑精彩照片。

整个流程一句话：**机器干活，人做裁判**——检测 → 网页标注 → 认人（可选）→ 队伍/个人合集 → 照片精选。

## 功能特性

- **进球检测**：YOLO 球/人双模型 CPU 推理 + 轨迹分析，自动找出疑似进球候选
- **网页标注**：浏览器里逐条确认进球，键盘操作（J 定锚 / F 排除 / G 补锚），片段自带 2 倍速
- **认人（可选）**：投篮者自动裁图 + CLIP 聚类分组 + 网页确认，不认人也能直接出片
- **合集合成**：队伍集锦、个人合集、每球独立片段、进球热图；进球片段 = 入网前 4 秒 + 后 2 秒
- **照片精选**：自动打分抽帧 + 构图裁切，网页点选后落盘
- **断点续跑**：所有环节中断后重跑同一条命令即可，已完成的阶段自动跳过

## 界面预览

> TODO：GUI 界面开发中（`gui/`，C 阶段落地），截图届时补充。现阶段检测、标注、认人、确认均在浏览器页面完成（由 CLI 生成静态页）。

## 安装

### 一键安装包（即将推出）

Windows 安装器内嵌 Python 运行时、ffmpeg 与模型权重，零配置开箱即用。**尚未发布**，当前请用下面的开发方式。

### 开发方式

要求：Python ≥ 3.10、ffmpeg + ffprobe 在 PATH 中（推荐 [BtbN 的 LGPL 构建](https://github.com/BtbN/FFmpeg-Builds/releases)，见下文许可声明）。

```powershell
git clone https://github.com/huangli/basketball-clip.git
cd basketball-clip
python -m venv .venv
.venv\Scripts\Activate.ps1
# torch 建议装 CPU 版（避免 PyPI 默认的 CUDA 大包）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy pillow open_clip_torch scikit-learn matplotlib httpx
```

可选依赖：

- `insightface` + `onnxruntime`：仅 `--photo-match` 人脸匹配预填用（默认关闭；buffalo_l 权重为非商用许可，见许可声明）
- `ruff`、`pytest`：开发用 lint/测试

### 模型权重（不随仓库分发）

`models/` 目录被 .gitignore 排除，需自行准备两个文件：

| 文件 | 用途 | 获取方式 |
|---|---|---|
| `models/yolov8n.pt` | 人物检测（持球排除、裁图质量复检） | [ultralytics 官方 assets](https://github.com/ultralytics/assets/releases) 下载 `yolov8n.pt` 放入 |
| `models/abdullahtarek_ball.pt` | 篮球/篮筐检测主力（Ball/Hoop/Player 三类） | 从作者项目 [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis) README 中的 `ball_detector_model.pt` 下载链接获取，重命名为 `abdullahtarek_ball.pt` 放入 |

另外，认人聚类用的 CLIP 权重（open_clip ViT-B-32 / laion2b_s34b_b79k，约 350MB）在首次运行聚类时自动从 Hugging Face 下载；网络受限可设置代理环境变量（见配置节）。安装包将内置全部权重。

## 使用

### 图形界面（开发中）

```powershell
python -m gui   # 起本地服务并自动打开浏览器（C 阶段落地，当前不可用）
```

设计口径：本地网页向导，普通用户不看命令行走完全流程——选素材目录 → 设置队名 → 检测进球 → 网页标注 → 认人 → 生成合集 → 照片精选，每步有进度与失败提示。

### 命令行

统一入口 `scripts/video.py`，五个子命令：

```powershell
# ① 检测：递归扫描素材目录，一路跑到生成标注页（几个小时，挂着跑）
python scripts/video.py score <素材目录> --session <场次ID>
#    --session 可省（缺省取素材目录名）；成功后自动记为当前场次，后续命令都不用再写
#    --batch-size 每批文件数 / --fids 指定文件补跑 / --force 全部重算 / --dry-run 只打印

# ② 人工标注：浏览器打开 work\<场次>\review_batch1\label.html，标完导出 goals_batchK.json 移入 work\<场次>\

# ③ 出片（不认人也能出：颜色分队集锦 + 每球独立片段 + 自动个人合集）
python scripts/video.py build
#    认人后：--all 全量 / --scorer 单人 / --team 单队；--4k 手动重出 4K（产物加 _4K 后缀）

# 认人（可选，要实名个人合集和分队集锦才做；逐批跑）
python scripts/video.py people --batch 1
#    确认页导出 roster.json 移入 work\<场次>\ 后重跑 build --all
#    --read-numbers 球衣读号预填（默认关，需配置 VLM 凭证）
#    --photo-match 人脸匹配预填（默认关，需 insightface）

# ④ 照片精选（可选）：机器打分出候选页，网页点选导出后 --apply 落盘
python scripts/video.py photo
python scripts/video.py photo --apply

# 清场：清空 output/work/源视频（先 --dry-run 看清单，真跑需输入 yes 确认）
python scripts/video.py clean --dry-run
```

`score` 成功后会把场次记入 `work/current_session.json`，`people`/`build`/`photo` 缺省读它；显式 `--session` 永远优先且不改写指针。

详细操作手册（含按键表、文件位置、常见问题）见 [docs/使用手册.md](docs/使用手册.md)。

### 开发与测试

```powershell
python -m ruff format scripts tests gui
python -m ruff check --fix scripts tests gui
python -m pytest -q
```

`ruff` / `pytest` 装了命令行入口的话，裸命令（`ruff …`、`pytest -q`）与 `python -m …` 等价，用哪个都行；`ruff check --fix` 自动修复后须人工复核 diff。

代码规范见 [rules.md](rules.md)（鲁棒优先 > 性能 > 简洁）。

## 配置

### 环境变量（均可选，不设也能跑默认流程）

| 变量 | 用途 |
|---|---|
| `BASKETBALL_CLIP_HTTPS_PROXY` | Hugging Face 下载代理（CLIP 权重首次下载用），如 `http://127.0.0.1:7890`；设置后自动映射为 `HTTPS_PROXY` |
| `BASKETBALL_CLIP_VLM_TOKEN` | VLM 读号 Bearer token（仅 `--read-numbers` 用，缺失时显式报错） |
| `BASKETBALL_CLIP_VLM_API_URL` | VLM 端点覆盖（可选，OpenAI 兼容 chat/completions 接口） |

### 队名配置

`work/<场次>/team_config.json`（会话级，缺省自动回退默认值，不配置也能用）：

```json
{ "version": 1, "team_name": "红队", "opponent": "蓝队" }
```

未配置时主队名默认为"主队"，对手名按场次 ID 后缀派生（如 `20260101_蓝队` → 蓝队），兜底"对手"。主队名决定 4K 默认档（主队队伍集锦默认出 4K）与合集文件名。

## 许可声明

- **本项目主代码**：[MIT](LICENSE)。
- **ultralytics（YOLO 推理框架）及 `yolov8n.pt` 权重**：[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。本项目以"按原样调用其公开接口"的方式使用并显式声明该依赖；**如需商用或闭源分发，请自行评估 AGPL 义务或购买 Ultralytics 企业许可**。
- **ffmpeg**：开发方式由用户自备；安装包将内嵌 [BtbN FFmpeg 构建](https://github.com/BtbN/FFmpeg-Builds)（LGPL 版本，许可文本随包附带）。请勿使用含 nonfree 组件的构建（如 gyan.dev full 版）随包分发。
- **`abdullahtarek_ball.pt` 权重**：源自 [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis)（训练数据为 Roboflow 公开数据集）。经查证（2026-09）：该项目 README 声明 MIT，但仓库中无 LICENSE 文件，权重经 Google Drive 单独分发且无独立许可声明，**再分发权利无法确认**——故该权重不随本仓库/安装包再分发，请用户按上文"模型权重"节自行从作者发布页下载。
- **CLIP 权重**（open_clip ViT-B-32 / laion2b_s34b_b79k）：首次运行从 Hugging Face 自动下载，不随仓库分发。
- **insightface buffalo_l**（仅可选的 `--photo-match` 人脸匹配用）：**非商用许可**，介意者不要使用该开关。

## English Summary

**basketball-clip** is an open-source tool that turns raw basketball game footage into highlight reels. It detects likely goals with YOLO models (ball/hoop/player) on CPU, serves the candidates in a browser-based labeling page for one-key human review, optionally identifies scorers via cropping + CLIP clustering with a confirmation page, and finally builds per-team and per-player highlight videos (1080p/4K, H.264), plus a curated photo selection. Everything is resumable: re-run the same command after any interruption.

- Entry point: `python scripts/video.py score|people|build|photo|clean` (a browser GUI wizard is under development)
- Requires Python ≥ 3.10 and ffmpeg; model weights are not redistributed with the repo — see the "模型权重" section above for download sources
- License: MIT for the project code. Note that the ultralytics dependency and YOLO weights are AGPL-3.0 (evaluate obligations or obtain an Ultralytics enterprise license for commercial use); the bundled ffmpeg in the future installer will be the BtbN LGPL build; the `abdullahtarek_ball.pt` weights' redistribution terms could not be verified, so they are **not** redistributed — users download them from the author's project page.
