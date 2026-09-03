"""一键跑批编排器：新场次从素材目录到标注页的全流程串联（batch-speedup F2）。

输入：原片目录（递归扫描 .mp4）、场次 ID。
输出：work/<场次>/session_facts.json、candidates_batchK.json、hoops_batchK.json、
    review_batchK/（含 events_index.json 与 label.html）、
    work/<场次>/run_session.log。
依赖：scripts/pipe_common.py；以 subprocess 调 extract_frames / mot_candidates /
    pilot_candidates / detect_hoops / gen_review_clips / gen_label_page
    六个老脚本（全部显式传参，不依赖其默认常量）。
    （triage.html 扫尾墙已于 2026-08-11 下线：与 label.html 功能重复，用户拍板，
    见 docs/batch-speedup/review07.md）
典型调用：
    python scripts/run_session.py "素材目录" --session 20260722 --dry-run
    python scripts/run_session.py "素材目录" --session 20260722 --batch-size 50

注意：必须从仓库根目录运行（老脚本内部用 work/ 相对路径）。
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errors import BasketballPipelineError, MediaTimeoutError
from pipe_common import (
    RunIdFilter,
    atomic_write_json,
    configure_logging,
    new_run_id,
    read_json,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).resolve().parent
WORK_ROOT: Path = Path("work")
FRAMES_ROOT: Path = Path("work/frames")
DETECT_CACHE_PATTERN: str = "work/detect/{}_mot_cache.json"
DEFAULT_BATCH_SIZE: int = 50
FFPROBE_TIMEOUT_SEC: int = 30  # rules.md §4：ffprobe 单文件 30s
FFPROBE_RETRY: int = 2  # rules.md §4：重试 2 次，退避 1s -> 2s
FPS_ROUND_DIGITS: int = 2  # 59.94/50 等按两位小数归一后比对
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] run=%(run_id)s %(name)s: %(message)s"

# 阶段产物校验口径（spec F2 断点续跑：JSON 可读 + 关键字段齐全）
PRODUCT_KINDS: tuple[str, ...] = ("candidates", "events", "session_facts")


@dataclass(frozen=True, slots=True)
class SourceMeta:
    """单个原片经 ffprobe 归一后的元数据（rules.md §0.2）。"""

    name: str  # 文件名（含扩展名，比对主键）
    path: Path
    width: int
    height: int
    fps: float
    duration: float

    @property
    def fid(self) -> str:
        """文件 ID = 主名（去扩展名），与 frames/detect 缓存目录名一致。"""
        return Path(self.name).stem


@dataclass(frozen=True, slots=True)
class StageCommand:
    """一个阶段的子进程命令（stage 编号 2-7；① 探测由编排器自身完成）。"""

    stage: int
    title: str
    argv: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """单批次的执行计划：fid 清单、各阶段命令、产物路径。"""

    label: str  # batchK / adhoc
    fids: tuple[str, ...]
    commands: tuple[StageCommand, ...]
    candidates: Path
    hoops: Path
    review_dir: Path
    events_index: Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    ap = argparse.ArgumentParser(description="一键跑批：素材目录 -> 标注页全套产物")
    ap.add_argument("srcdir", help="原片目录（递归扫描 .mp4）")
    ap.add_argument("--session", required=True, help="场次 ID（产物目录与 LSKEY 后缀）")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批文件数")
    ap.add_argument("--fids", default="", help="逗号分隔 fid 清单（adhoc 模式，不占批次序号）")
    ap.add_argument("--force", action="store_true", help="忽略断点产物与事实表，全部重算")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令清单与切批划分，不执行")
    return ap.parse_args(argv)


def _ffprobe_metadata(path: Path) -> tuple[int, int, float, float]:
    """ffprobe 取单文件 宽/高/帧率/时长；30s 超时 + 2 次重试（rules.md §4）。

    Raises:
        MediaTimeoutError: 超时重试耗尽。
        BasketballPipelineError: 非零退出或元数据无法解析（不猜尺寸，显式失败）。
    """
    cmd: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    last_err: str = ""
    last_timeout: bool = False
    for attempt in range(1, FFPROBE_RETRY + 2):
        try:
            proc = subprocess.run(  # noqa: S603 固定 ffprobe 二进制，参数内部构造
                cmd,
                capture_output=True,
                # ffprobe 报错输出含中文文件名，locale 代码页会解码崩（同
                # pipe_common.run_ffmpeg 的 cp1252 坑）——钉死 UTF-8
                encoding="utf-8",
                errors="replace",
                timeout=FFPROBE_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            last_err = f"超时({FFPROBE_TIMEOUT_SEC}s)"
            last_timeout = True
        else:
            if proc.returncode == 0:
                return _parse_probe_json(proc.stdout, path)
            last_err = proc.stderr.strip()[-200:]
            last_timeout = False
        logger.warning("  ffprobe 第%d次失败: %s: %s", attempt, path.name, last_err)
        if attempt <= FFPROBE_RETRY:
            time.sleep(float(attempt))
    if last_timeout:
        raise MediaTimeoutError(f"ffprobe 超时重试耗尽: {path}: {last_err}")
    raise BasketballPipelineError(f"ffprobe 失败: {path}: {last_err}")


def _parse_probe_json(stdout: str, path: Path) -> tuple[int, int, float, float]:
    """解析 ffprobe JSON 输出为 (宽, 高, 帧率, 时长)；任何字段异常即显式失败。

    Raises:
        BasketballPipelineError: JSON 损坏或字段缺失/不可解析。
    """
    try:
        payload: dict[str, Any] = json.loads(stdout)
        stream: dict[str, Any] = payload["streams"][0]
        width: int = int(stream["width"])
        height: int = int(stream["height"])
        num, den = str(stream["avg_frame_rate"]).split("/")
        fps: float = round(int(num) / int(den), FPS_ROUND_DIGITS)
        duration: float = float(payload["format"]["duration"])
    except (
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise BasketballPipelineError(f"ffprobe 元数据解析失败: {path}: {exc}") from exc
    if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
        raise BasketballPipelineError(
            f"ffprobe 元数据非法: {path}: {width}x{height} @{fps}fps {duration}s"
        )
    return width, height, fps, duration


def scan_sources(srcdir: Path) -> list[SourceMeta]:
    """递归扫描素材目录并逐文件 ffprobe 探测，按文件名排序返回事实清单。

    Args:
        srcdir: 原片目录。

    Returns:
        SourceMeta 列表（按文件名升序 = 拍摄时间序）。

    Raises:
        BasketballPipelineError: 目录不存在、无 .mp4、fid 重名、任一文件探测失败。
    """
    if not srcdir.is_dir():
        raise BasketballPipelineError(f"原片目录不存在: {srcdir}")
    paths: list[Path] = sorted(
        (p for p in srcdir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"),
        key=lambda p: (p.name, str(p)),
    )
    if not paths:
        raise BasketballPipelineError(f"素材目录无 .mp4 文件: {srcdir}")
    fids: list[str] = [p.stem for p in paths]
    if len(set(fids)) != len(fids):
        raise BasketballPipelineError(f"素材目录存在 fid 重名（子目录冲突）: {srcdir}")
    metas: list[SourceMeta] = []
    for i, p in enumerate(paths, 1):
        width, height, fps, duration = _ffprobe_metadata(p)
        metas.append(SourceMeta(p.name, p, width, height, fps, duration))
        if i % 50 == 0 or i == len(paths):
            logger.info("  探测进度 %d/%d", i, len(paths))
    return metas


def find_mixed_specs(metas: list[SourceMeta]) -> list[str]:
    """检查混合分辨率/混合帧率，返回明细清单（空 = 规格统一）。

    规格要求按比例分别处理，混合即 WARNING 列明细并终止，不瞎猜。
    """
    issues: list[str] = []
    resolutions: set[tuple[int, int]] = {(m.width, m.height) for m in metas}
    if len(resolutions) > 1:
        issues.append(f"混合分辨率 {sorted(resolutions)}:")
        issues.extend(f"  {m.name}: {m.width}x{m.height}" for m in metas)
    fps_set: set[float] = {m.fps for m in metas}
    if len(fps_set) > 1:
        issues.append(f"混合帧率 {sorted(fps_set)}:")
        issues.extend(f"  {m.name}: {m.fps}fps" for m in metas)
    return issues


def build_facts(metas: list[SourceMeta]) -> dict[str, Any]:
    """由探测结果构建场次事实表（session_facts.json 落盘结构）。"""
    first: SourceMeta = metas[0]
    return {
        "file_count": len(metas),
        "width": first.width,
        "height": first.height,
        "fps": first.fps,
        "files": {
            m.name: {
                "width": m.width,
                "height": m.height,
                "fps": m.fps,
                "duration": m.duration,
            }
            for m in metas
        },
    }


def compare_facts(saved: dict[str, Any], metas: list[SourceMeta]) -> list[str]:
    """重探测结果与已落盘事实表比对（按文件名逐项比分辨率/帧率）。

    Args:
        saved: session_facts.json 内容（须已过 validate_product 校验）。
        metas: 本次重探测结果。

    Returns:
        不一致明细（空 = 一致）；增删文件也算不一致（素材流动须显式 --force 确认）。
    """
    saved_files: dict[str, Any] = saved.get("files", {})
    current: dict[str, SourceMeta] = {m.name: m for m in metas}
    issues: list[str] = []
    for name in sorted(saved_files):
        if name not in current:
            issues.append(f"文件已删除: {name}")
            continue
        old: dict[str, Any] = saved_files[name]
        m: SourceMeta = current[name]
        for key, new_val in (("width", m.width), ("height", m.height), ("fps", m.fps)):
            if old.get(key) != new_val:
                issues.append(f"{name}: {key} {old.get(key)} != {new_val}")
    for name in sorted(current):
        if name not in saved_files:
            issues.append(f"新增文件: {name}")
    return issues


def validate_product(path: Path, kind: str) -> bool:
    """断点产物校验：存在 + JSON 可读 + 关键字段齐全（防"存在但损坏"）。

    Args:
        path: 产物路径。
        kind: candidates（顶层 list）/ events（dict 含 events 键）/
            session_facts（dict 含 files 键）。

    Returns:
        True = 可跳过；False = 缺失或损坏（调用方重算，不带坏产物往下跑）。
    """
    if not path.is_file():
        return False
    try:
        data: Any = read_json(path, what=path.name)
    except (BasketballPipelineError, OSError) as exc:
        logger.warning("产物损坏将重算: %s: %s", path, exc)
        return False
    if kind == "candidates":
        return isinstance(data, list)
    if kind == "events":
        return isinstance(data, dict) and isinstance(data.get("events"), list)
    if kind == "session_facts":
        return isinstance(data, dict) and isinstance(data.get("files"), dict)
    raise ValueError(f"未知产物类型: {kind}")


def make_batches(fids: list[str], batch_size: int) -> list[list[str]]:
    """fid 按文件名排序（= 拍摄时间序）后按 batch_size 切批，序号 K 从 1 递增。

    Args:
        fids: 文件 ID 清单（无需预排序）。
        batch_size: 每批文件数（≥1）。

    Returns:
        批次列表，每批为 fid 子列表。

    Raises:
        ValueError: batch_size < 1。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 ≥1: {batch_size}")
    ordered: list[str] = sorted(fids)
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def check_fid_coverage(records: Any, fids: list[str]) -> list[str]:  # noqa: ANN401
    """核对 candidates 的 fid 覆盖：pilot 对无缓存 fid 只 WARNING 产空，须编排层兜底。

    Returns:
        未被覆盖的 fid 清单（空 = 覆盖完整）。
    """
    if not isinstance(records, list):
        return list(fids)
    covered: set[str] = {r["fid"] for r in records if isinstance(r, dict) and "fid" in r}
    return [f for f in fids if f not in covered]


def _legit_zero_candidate(fid: str) -> bool:
    """fid 无候选记录时的兜底判定：检测缓存有效（帧数匹配）= 算法口径下真零候选。

    短视频可能全程无静止段/断轨重连模式，pilot 正常产空——与"无缓存被跳过"
    区分开（2026-08-13 某场次批次 2/3 实录：3 个短片段零候选触发误报）。

    Returns:
        True = 缓存存在、可读且 frames 字段与帧目录帧数一致（零候选属正常产出）。
    """
    cache: Path = Path(DETECT_CACHE_PATTERN.format(fid))
    if not cache.is_file():
        return False
    try:
        payload: Any = read_json(cache, what=cache.name)
    except (BasketballPipelineError, OSError):
        return False
    n_frames: int = len(list((FRAMES_ROOT / fid).glob("f_*.jpg")))
    return isinstance(payload, dict) and payload.get("frames") == n_frames


def build_stage_plan(
    srcdir: Path,
    session_dir: Path,
    width: int,
    height: int,
    fid_batches: list[list[str]],
    *,
    adhoc: bool,
) -> list[BatchPlan]:
    """按批次构建 ②-⑦ 阶段命令清单（全部显式传参，不依赖老脚本默认常量）。

    与 scan_sources 解耦：探测结果（尺寸/批次划分）以参数注入，dry-run 单测可
    monkeypatch。--fids 模式产物固定 adhoc 命名，不占批次序号、不覆盖历史批次。

    Args:
        srcdir: 原片目录。
        session_dir: work/<场次>/ 目录。
        width/height: ① 探测确认的原片尺寸（注入 --orig）。
        fid_batches: 切批后的 fid 清单。
        adhoc: True = --fids 模式（candidates_adhoc.json / hoops_adhoc.json /
            review_adhoc/）。

    Returns:
        批次执行计划列表。
    """
    plans: list[BatchPlan] = []
    for k, fids in enumerate(fid_batches, start=1):
        label: str = "adhoc" if adhoc else f"batch{k}"
        candidates: Path = session_dir / f"candidates_{label}.json"
        hoops: Path = session_dir / f"hoops_{label}.json"
        review_dir: Path = session_dir / f"review_{label}"
        events_index: Path = review_dir / "events_index.json"
        commands: tuple[StageCommand, ...] = (
            StageCommand(
                2,
                "② 抽帧",
                (sys.executable, str(SCRIPT_DIR / "extract_frames.py"), str(srcdir)),
                "全场抽 + 幂等跳过（extract_frames 不支持按批，首批即全场抽）",
            ),
            StageCommand(
                3,
                "③ 检测",
                (sys.executable, str(SCRIPT_DIR / "mot_candidates.py"), *fids),
            ),
            StageCommand(
                4,
                "④ 候选",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "pilot_candidates.py"),
                    "--out",
                    str(candidates),
                    *fids,
                ),
            ),
            StageCommand(
                5,
                "⑤ 筐轨迹",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "detect_hoops.py"),
                    "--candidates",
                    str(candidates),
                    "--out",
                    str(hoops),
                ),
            ),
            StageCommand(
                6,
                "⑥ 审核片段",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "gen_review_clips.py"),
                    "--candidates",
                    str(candidates),
                    "--outdir",
                    str(review_dir),
                    "--srcdir",
                    str(srcdir),
                    "--orig",
                    f"{width}x{height}",
                    "--hoops",
                    str(hoops),
                    "--keep-clips",
                ),
            ),
            StageCommand(
                7,
                "⑦ 标注页 label.html",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "gen_label_page.py"),
                    "--index",
                    str(events_index),
                    "--session",
                    session_dir.name,
                    # 批次页导出即 goals_batchK.json（移动即接入 people）；
                    # adhoc 不传，维持旧名 goals_<场次>.json
                    *(("--batch", str(k)) if not adhoc else ()),
                ),
            ),
        )
        plans.append(
            BatchPlan(label, tuple(fids), commands, candidates, hoops, review_dir, events_index)
        )
    return plans


def _frames_ready(fids: tuple[str, ...]) -> bool:
    """阶段② 断点判定：本批每个 fid 的帧目录已存在且非空（内部帧数校验归老脚本）。"""
    for fid in fids:
        frame_dir: Path = FRAMES_ROOT / fid
        if not frame_dir.is_dir() or not any(frame_dir.glob("f_*.jpg")):
            return False
    return True


def _detect_cache_ready(fids: tuple[str, ...]) -> bool:
    """阶段③ 断点判定：本批每个 fid 的 MOT 检测缓存存在且 JSON 可读。"""
    for fid in fids:
        path: Path = Path(DETECT_CACHE_PATTERN.format(fid))
        if not path.is_file():
            return False
        try:
            read_json(path, what=path.name)
        except (BasketballPipelineError, OSError) as exc:
            logger.warning("检测缓存损坏将重算: %s: %s", path, exc)
            return False
    return True


def _stage_done(plan: BatchPlan, cmd: StageCommand) -> bool:
    """判断单阶段产物是否已就绪（存在且校验通过即跳过）。"""
    if cmd.stage == 2:
        return _frames_ready(plan.fids)
    if cmd.stage == 3:
        return _detect_cache_ready(plan.fids)
    if cmd.stage == 4:
        return validate_product(plan.candidates, "candidates")
    if cmd.stage == 5:
        return validate_product(plan.hoops, "events")
    if cmd.stage == 6:
        return validate_product(plan.events_index, "events")
    if cmd.stage == 7:
        label_html: Path = plan.review_dir / "label.html"
        return label_html.is_file() and label_html.stat().st_size > 0
    return False


def _post_check(plan: BatchPlan, cmd: StageCommand) -> str | None:
    """阶段执行后的产物复核；返回失败原因（None = 通过）。

    ④ 额外核对 fid 覆盖：无候选记录的 fid 分两种——缓存有效 = 真零候选
    （WARNING 放行）；缓存缺失/失效 = pilot 产空漏检（报错兜底）。
    """
    if cmd.stage == 4:
        if not validate_product(plan.candidates, "candidates"):
            return f"candidates 未产出或损坏: {plan.candidates}"
        missing: list[str] = check_fid_coverage(read_json(plan.candidates), list(plan.fids))
        hard: list[str] = [f for f in missing if not _legit_zero_candidate(f)]
        soft: list[str] = [f for f in missing if _legit_zero_candidate(f)]
        if soft:
            logger.warning("%s ④ 零候选 fid（缓存正常，正常产出）: %s", plan.label, soft)
        if hard:
            return f"fid 覆盖不足（{len(hard)} 个无有效缓存）: {hard}"
    elif cmd.stage == 5 and not validate_product(plan.hoops, "events"):
        return f"hoops 未产出或损坏: {plan.hoops}"
    elif cmd.stage == 6 and not validate_product(plan.events_index, "events"):
        return f"events_index 未产出或损坏: {plan.events_index}"
    return None


def _run_command(argv: tuple[str, ...]) -> None:
    """执行单阶段子进程（输出直通控制台，便于观察老脚本进度）。

    Raises:
        BasketballPipelineError: 非零退出。
    """
    logger.info("执行: %s", shlex.join(argv))
    proc = subprocess.run(argv, check=False)  # noqa: S603 命令全部由本模块内部构造
    if proc.returncode != 0:
        raise BasketballPipelineError(f"子进程退出码 {proc.returncode}: {shlex.join(argv[:2])}")


def execute_plans(plans: list[BatchPlan], *, force: bool) -> list[str]:
    """逐批逐阶段执行；单批失败记 WARNING 继续后续批次（rules.md 鲁棒条）。

    Returns:
        失败清单（空 = 全部成功）。
    """
    failures: list[str] = []
    for plan in plans:
        logger.info("==== %s：%d 个 fid ====", plan.label, len(plan.fids))
        for cmd in plan.commands:
            if not force and _stage_done(plan, cmd):
                logger.info("跳过 %s %s（产物已就绪）", plan.label, cmd.title)
                continue
            if cmd.note:
                logger.info("%s %s：%s", plan.label, cmd.title, cmd.note)
            try:
                _run_command(cmd.argv)
            except BasketballPipelineError as exc:
                logger.error("%s %s 失败: %s", plan.label, cmd.title, exc)
                failures.append(f"{plan.label} {cmd.title}: {exc}")
                break
            reason: str | None = _post_check(plan, cmd)
            if reason is not None:
                logger.error("%s %s 复核失败: %s", plan.label, cmd.title, reason)
                failures.append(f"{plan.label} {cmd.title}: {reason}")
                break
    return failures


def _print_dry_run(plans: list[BatchPlan], metas: list[SourceMeta]) -> None:
    """--dry-run：打印切批划分与每批 6 条（②-⑦）命令清单，不执行。"""
    first: SourceMeta = metas[0]
    logger.info(
        "DRY-RUN：共 %d 文件，%d 批；规格 %dx%d @%sfps",
        len(metas),
        len(plans),
        first.width,
        first.height,
        first.fps,
    )
    for plan in plans:
        logger.info(
            "---- %s：fid %s .. %s（%d 个）----",
            plan.label,
            plan.fids[0],
            plan.fids[-1],
            len(plan.fids),
        )
        logger.info("  ① 探测：编排器已完成（事实表 session_facts.json）")
        for cmd in plan.commands:
            suffix: str = f"  # {cmd.note}" if cmd.note else ""
            logger.info("  %s: %s%s", cmd.title, shlex.join(cmd.argv), suffix)


def _attach_file_handler(log_path: Path, run_id: str) -> None:
    """挂 run_session.log 文件 handler（与控制台同格式，带 run_id）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RunIdFilter(run_id))
    logging.getLogger().addHandler(handler)


def _validate_session_id(session: str) -> None:
    """场次 ID 校验：将用于路径与 LSKEY，禁止路径分隔与父目录引用。

    Raises:
        BasketballPipelineError: 含 / \\ 或 .. 的非法 ID。
    """
    if not session or any(tok in session for tok in ("/", "\\", "..")):
        raise BasketballPipelineError(f"非法场次 ID: {session!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=有批次失败；2=探测/事实表终止）。"""
    args: argparse.Namespace = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        _validate_session_id(args.session)
    except BasketballPipelineError as exc:
        logger.error("参数错误 run_id=%s: %s", run_id, exc)
        return 2
    session_dir: Path = WORK_ROOT / args.session
    if not args.dry_run:
        _attach_file_handler(session_dir / "run_session.log", run_id)
        logger.info(
            "run_id=%s session=%s 日志 -> %s", run_id, args.session, session_dir / "run_session.log"
        )

    # ① 探测：ffprobe 全量扫描，失败即终止（不猜尺寸）
    try:
        metas: list[SourceMeta] = scan_sources(Path(args.srcdir))
    except BasketballPipelineError as exc:
        logger.error("探测失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 2
    mixed: list[str] = find_mixed_specs(metas)
    if mixed:
        for line in mixed:
            logger.warning("规格不统一: %s", line)
        logger.error("混合分辨率/帧率须按比例分别处理，终止")
        return 2

    # 事实表：已存在则重探测比对，不一致 WARNING 终止（--force 除外）
    facts_path: Path = session_dir / "session_facts.json"
    if facts_path.is_file() and not args.force and validate_product(facts_path, "session_facts"):
        issues: list[str] = compare_facts(read_json(facts_path, what="session_facts.json"), metas)
        if issues:
            for line in issues:
                logger.warning("事实表不一致: %s", line)
            logger.error("素材与 %s 不一致，终止（确认后用 --force 重探测覆写）", facts_path)
            return 2
        logger.info("事实表比对一致: %s", facts_path)
    elif args.dry_run:
        logger.info("DRY-RUN：事实表不落盘（实际运行将写 %s）", facts_path)
    else:
        atomic_write_json(facts_path, build_facts(metas), what="session_facts.json")
        logger.info("事实表落盘: %s（%d 文件）", facts_path, len(metas))

    # 切批 / adhoc
    known_fids: set[str] = {m.fid for m in metas}
    adhoc: bool = bool(args.fids.strip())
    if adhoc:
        fids: list[str] = [f.strip() for f in args.fids.split(",") if f.strip()]
        unknown: list[str] = [f for f in fids if f not in known_fids]
        if not fids or unknown:
            logger.error("--fids 含未探测到的 fid: %s", unknown or "空清单")
            return 2
        fid_batches: list[list[str]] = [fids]
    else:
        fid_batches = make_batches([m.fid for m in metas], args.batch_size)

    first: SourceMeta = metas[0]
    plans: list[BatchPlan] = build_stage_plan(
        Path(args.srcdir), session_dir, first.width, first.height, fid_batches, adhoc=adhoc
    )

    if args.dry_run:
        _print_dry_run(plans, metas)
        return 0

    failures: list[str] = execute_plans(plans, force=args.force)
    if failures:
        logger.error("==== 失败清单（%d 条）====", len(failures))
        for line in failures:
            logger.error("  %s", line)
        return 1
    logger.info("全部批次完成（%d 批）", len(plans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
