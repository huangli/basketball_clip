"""统一入口 CLI：score / people / build / photo 四条高频链路的 subprocess 薄封装。

输入：命令行参数（素材目录 / 场次 ID / 批次 / 过滤项）。
输出：透传调用 run_session / crop_scorers / propagate_scorers / cluster_scorers /
    face_match_scorers / gen_scorer_page / auto_roster / build_highlight /
    rank_photos / gen_photo_page 十个底层脚本（photo_match_scorers 已随 T12 退出
    people 链，证伪留档）；
    build 按 roster 状态分两路（认人可选化 2026-08-22，docs/build-auto-scorer/）：
    confirmed=true 走现状合成、收尾追加 in-process 调 goal_heatmap.heat_session
    出热图双风格（v4.2 集成；懒 import，附属产物失败不阻塞主链），
    缺失/未确认走自动模式（颜色分队队伍集锦 + 进球片段 + 自动个人合集，热图不触发）；
    状态文件 work/<场次>/video_cli.json。
依赖：scripts/pipe_common.py（read_json/atomic_write_json/configure_logging/new_run_id）、
    scripts/errors.py、scripts/roster.py（validate_roster）；命令拼装契约见
    docs/video-cli/spec.md（逐字照做，不改底层脚本任何行为）。
典型调用（任意目录可运行；启动后自动 chdir 到仓库根，用户相对路径按启动目录解析；
--session 均可省略：score 缺省取素材目录 basename 并记当前场次指针
work/current_session.json，people/build/photo 缺省读指针，显式 --session 永远优先，
docs/default-session/spec.md）：
    python scripts/video.py score <素材目录> [--session 20260722]
    python scripts/video.py people [--batch 1]
    python scripts/video.py build --all
    python scripts/video.py photo [--apply]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from errors import BasketballPipelineError, SchemaError
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import (
    format_key,
    is_opponent_tag,
    opponent_team_name,
    validate_roster,
)
from team_config import load_team_config

logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SCRIPT_DIR.parent  # 仓库根（work/ 等相对路径基准；main 启动后 chdir 到此）
WORK_ROOT: Path = Path("work")
STATE_NAME: str = "video_cli.json"
STATE_VERSION: int = 1
# 当前场次指针（score 成功后写入；people/build/photo 缺省 --session 时读取，
# 显式 --session 永远优先；docs/default-session/spec.md）
CURRENT_SESSION_NAME: str = "current_session.json"
CURRENT_SESSION_VERSION: int = 1
# 照片库目录（people ②.5 照片匹配串接条件；docs/photo-roster/spec.md T6）
PHOTOS_DIR: Path = Path("photos")
# 号码→姓名全局名单（people ③自动注入 --players；用户维护，gitignore 随 photos/）
PHOTOS_NAMES: Path = PHOTOS_DIR / "names.json"
# 自动名单 tag 前缀（主队现行白球衣，team_of_tag 白→主队；换色改这里）
NAMES_TAG_PREFIX: str = "白"
# 聚类段 CLIP 权重首跑下载需走代理：读此环境变量，未设置则不注入（缺省不走代理）
PROXY_ENV_NAME: str = "BASKETBALL_CLIP_HTTPS_PROXY"
# 聚类定稿口径（docs/scorer-cluster/；底层默认 average/0.25 是未标定起点，勿依赖）
CLUSTER_LINKAGE: str = "complete"
CLUSTER_THRESHOLD: str = "0.15"
# --max-reads 缺省换算：confirmed 球数 ×3（--best-crops 默认 3，docs/scorer-reid/spec.md）
MAX_READS_PER_GOAL: int = 3
# 输出尺寸按素材主比例换算（容差 ±1%；spec §build）
RATIO_TOLERANCE: float = 0.01
RATIO_16_9: float = 16 / 9
RATIO_4_3: float = 4 / 3
OUT_16_9: str = "1920x1080"
OUT_4_3: str = "1440x1080"
# 4K 档（docs/build-4k/spec.md：主队集锦默认 4K + --4k 手动重出）
OUT4K_16_9: str = "3840x2160"
OUT4K_4_3: str = "2880x2160"
# 主队队名（4K 默认档唯一受益队）由 team_config.json 会话级注入，
# 缺失回退 team_config.DEFAULT_TEAM_NAME（build 时按会话目录读取，见 _cmd_build_confirmed）
# --4k 手动重出的产物名后缀（透传 build_highlight --name-suffix，spec D2）
FOUR_K_SUFFIX: str = "_4K"
# 便服队不出分队集锦（build_highlight --team 便服 明文拒收退出 1；--all 展开时跳过）
CASUAL_TEAM: str = "便服"
# 多批次 build 的合并 goals 中间产物（work/<场次>/ 下，每次 build 重写——素材流动）
# 命名不以 goals 开头——避开 discover_batches 的 goals*.json 扫描，防 WARNING 噪音
MERGED_GOALS_NAME: str = "merged_goals_cli.json"
# clean 清空的输出根（work/ 用既有 WORK_ROOT；两根本身保留只清内容）
OUTPUT_ROOT: Path = Path("output")
# 批次 goals 文件名双轨：goals.json（旧布局批次 1）/ goals_batchK.json（现行布局）
GOALS_BATCH_RE: re.Pattern[str] = re.compile(r"^goals_batch(\d+)\.json$")


class StepFailedError(BasketballPipelineError):
    """单步子进程非零退出；携带完整命令便于打印失败现场。"""

    def __init__(self, cmd: list[str], returncode: int) -> None:
        super().__init__(f"子进程退出码 {returncode}: {shlex.join(cmd)}")
        self.cmd: list[str] = cmd
        self.returncode: int = returncode


@dataclass(frozen=True, slots=True)
class Batch:
    """单个批次的产物路径集合（命名双轨见 docs/video-cli/spec.md §批次发现）。"""

    batch: int  # 批次序号 K（旧布局 goals.json 视为批次 1）
    goals: Path
    candidates: Path
    review_dir: Path
    scorers_dir: Path

    @property
    def events_index(self) -> Path:
        """review 目录下的 events_index.json（旧布局批次 1 可能正常缺失）。"""
        return self.review_dir / "events_index.json"

    @property
    def scorer_candidates(self) -> Path:
        """crop_scorers 产出的 scorer_candidates.json。"""
        return self.scorers_dir / "scorer_candidates.json"

    @property
    def scorer_clusters(self) -> Path:
        """cluster_scorers 产出的 scorer_clusters.json（与 candidates 同目录硬约束）。"""
        return self.scorers_dir / "scorer_clusters.json"

    @property
    def photo_matches(self) -> Path:
        """②.5 照片匹配产出的 photo_matches.json（与 candidates 同目录硬约束）。"""
        return self.scorers_dir / "photo_matches.json"

    @property
    def track_links(self) -> Path:
        """①.5 轨迹传播产出的 track_links.json（与 candidates 同目录硬约束）。"""
        return self.scorers_dir / "track_links.json"


@dataclass(frozen=True, slots=True)
class Step:
    """一个待执行步骤：标题（日志用）、子进程命令、额外环境变量。

    allow_fail=True 的步骤失败时 ERROR 留痕后继续后续步骤（降级不中断整链；
    当前仅 people ②.5 照片匹配用，docs/photo-roster/spec.md T6）。
    """

    title: str
    argv: tuple[str, ...]
    env_extra: dict[str, str] | None = None
    allow_fail: bool = False


def _proxy_env_extra() -> dict[str, str] | None:
    """代理 env 注入：BASKETBALL_CLIP_HTTPS_PROXY 已设则映射为 HTTPS_PROXY，否则不注入。"""
    proxy: str | None = os.environ.get(PROXY_ENV_NAME)
    return {"HTTPS_PROXY": proxy} if proxy else None


def run_step(cmd: list[str], env_extra: dict[str, str] | None = None) -> None:
    """执行单步子进程：log 完整命令，env 统一注入 PYTHONIOENCODING=utf-8。

    Args:
        cmd: 完整子进程命令（含 sys.executable 与脚本路径）。
        env_extra: 追加注入的环境变量（如聚类段的 HTTPS_PROXY），os.environ 复制后改。

    Raises:
        StepFailedError: 子进程非零退出。
    """
    logger.info("执行: %s", shlex.join(cmd))
    env: dict[str, str] = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, check=False, env=env)  # noqa: S603 命令全部由本模块内部构造
    if proc.returncode != 0:
        raise StepFailedError(cmd, proc.returncode)


def _log_dry_step(step: Step) -> None:
    """--dry-run：只打印将要执行的命令，不启动子进程。"""
    logger.info("DRY-RUN %s: %s", step.title, shlex.join(step.argv))
    if step.env_extra:
        logger.info("DRY-RUN %s: 叠加 env %s", step.title, step.env_extra)


def session_dir_or_die(session: str) -> Path:
    """定位 work/<场次>/ 目录；不存在显式失败（不猜路径）。

    Raises:
        BasketballPipelineError: 目录不存在。
    """
    session_dir: Path = WORK_ROOT / session
    if not session_dir.is_dir():
        raise BasketballPipelineError(f"场次目录不存在: {session_dir}（先跑 score）")
    return session_dir


def load_state(session: str) -> dict[str, Any]:
    """读取 work/<场次>/video_cli.json；不存在返回默认空状态，版本不符显式失败。

    Raises:
        BasketballPipelineError: state 版本不支持（不静默降级）。
        SchemaError: JSON 损坏（pipe_common.read_json 抛出）。
    """
    path: Path = WORK_ROOT / session / STATE_NAME
    if not path.is_file():
        return {"version": STATE_VERSION, "session": session, "srcdir": "", "runs": []}
    data: Any = read_json(path, what=STATE_NAME)
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise BasketballPipelineError(f"{path}: state 版本不支持（期望 version={STATE_VERSION}）")
    if not isinstance(data.get("runs"), list):
        raise BasketballPipelineError(f"{path}: runs 必须是列表（审计口径：只追加不覆盖）")
    return data


def save_state(session: str, state: dict[str, Any]) -> None:
    """原子写 state；updated_at 刷新为当前时间。调用方保证 runs 只追加。"""
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    session_dir: Path = WORK_ROOT / session
    session_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(session_dir / STATE_NAME, state, what=STATE_NAME)


def save_current_session(session: str) -> None:
    """把 score 的场次写为当前场次指针（people/build/photo 缺省 --session 时读取）。

    仅 score 成功后调用（dry-run 不写，与 state 口径一致）；显式 --session 跑
    people/build/photo 只是临时覆盖，不改写指针。

    Raises:
        OSError: IO 重试耗尽（atomic_write_json 抛出）。
    """
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        WORK_ROOT / CURRENT_SESSION_NAME,
        {
            "version": CURRENT_SESSION_VERSION,
            "session": session,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "score",
        },
        what=CURRENT_SESSION_NAME,
    )


def load_current_session() -> str:
    """读当前场次指针；缺失/损坏/version 不符/session 空全部显式失败（不猜场次）。

    Raises:
        BasketballPipelineError: 指针缺失，或 version 不符 / session 非非空 str。
        SchemaError: JSON 损坏（pipe_common.read_json 抛出）。

    Returns:
        当前场次 ID。
    """
    path: Path = WORK_ROOT / CURRENT_SESSION_NAME
    if not path.is_file():
        raise BasketballPipelineError(
            f"--session 未给且无当前场次记录: {path}（先跑 score 或显式给 --session）"
        )
    data: Any = read_json(path, what=CURRENT_SESSION_NAME)
    if not isinstance(data, dict) or data.get("version") != CURRENT_SESSION_VERSION:
        raise BasketballPipelineError(
            f"{path}: 指针版本不支持（期望 version={CURRENT_SESSION_VERSION}）"
        )
    session: Any = data.get("session")
    if not isinstance(session, str) or not session:
        raise BasketballPipelineError(f"{path}: session 必须是非空 str，实际 {session!r}")
    return session


def _explicit_session(args: argparse.Namespace) -> str | None:
    """取显式 --session；给了空串显式失败（空串≠未给，不猜场次）。

    Raises:
        BasketballPipelineError: --session 为空串。
    """
    if args.session is None:
        return None
    if not args.session:
        raise BasketballPipelineError("--session 空串非法（场次 ID 必须是非空字符串）")
    return str(args.session)


def resolve_session(args: argparse.Namespace) -> str:
    """people/build/photo 场次解析：显式 --session 优先，否则读当前场次指针。

    Raises:
        BasketballPipelineError: 两路皆缺、--session 空串或指针异常（不猜场次）。
    """
    explicit: str | None = _explicit_session(args)
    if explicit is not None:
        return explicit
    return load_current_session()


def _resolve_score_session(args: argparse.Namespace) -> str:
    """score 场次解析：显式 --session 优先，否则取素材目录 basename。

    Raises:
        BasketballPipelineError: --session 空串，或 basename 为空（如盘符根，不猜场次）。
    """
    explicit: str | None = _explicit_session(args)
    if explicit is not None:
        return explicit
    name: str = Path(args.srcdir).resolve().name
    if not name:
        raise BasketballPipelineError(
            f"素材目录无有效 basename: {args.srcdir}（请显式给 --session）"
        )
    return name


def resolve_rawdir(args_rawdir: str | None, state: dict[str, Any]) -> Path:
    """解析原片目录：显式 --rawdir 优先，其次 state.srcdir，都没有显式失败。

    Raises:
        BasketballPipelineError: 两路皆缺（不猜路径）。
    """
    if args_rawdir:
        return Path(args_rawdir)
    srcdir: Any = state.get("srcdir")
    if isinstance(srcdir, str) and srcdir:
        return Path(srcdir)
    raise BasketballPipelineError(
        "--rawdir 未给且 video_cli.json 无 srcdir（先跑 score 或显式给 --rawdir）"
    )


def _batch_from_goals(goals_path: Path) -> Batch | None:
    """由 goals 文件名推导批次配套路径（双轨）；无法识别返回 None。"""
    session_dir: Path = goals_path.parent
    name: str = goals_path.name
    if name == "goals.json":
        return Batch(
            1,
            goals_path,
            session_dir / "candidates.json",
            session_dir / "review",
            session_dir / "scorers",
        )
    m: re.Match[str] | None = GOALS_BATCH_RE.match(name)
    if m is None:
        return None
    k: int = int(m.group(1))
    if k < 1:
        return None
    return Batch(
        k,
        goals_path,
        session_dir / f"candidates_batch{k}.json",
        session_dir / f"review_batch{k}",
        session_dir / f"scorers_b{k}",
    )


def discover_batches(session_dir: Path) -> list[Batch]:
    """扫描场次目录的 goals 文件定位批次（people/build 共用），按批次号升序。

    candidates 缺失仅记 WARNING（people 执行阶段跳过该批）；events_index 缺失的
    降级 WARNING 归 people 链路（build 不引用 review 产物，不在此报噪音）；
    同 K 双布局并存显式失败（不猜）。

    Raises:
        BasketballPipelineError: 无任何 goals 文件，或同批次双布局并存。
    """
    batches: dict[int, Batch] = {}
    for goals_path in sorted(session_dir.glob("goals*.json")):
        batch: Batch | None = _batch_from_goals(goals_path)
        if batch is None:
            logger.warning("无法识别的 goals 文件，跳过: %s", goals_path.name)
            continue
        if batch.batch in batches:
            raise BasketballPipelineError(
                f"批次 {batch.batch} 双布局并存: {batches[batch.batch].goals.name} 与 "
                f"{goals_path.name}（人工改名为单一布局后重跑）"
            )
        batches[batch.batch] = batch
    if not batches:
        raise BasketballPipelineError(f"{session_dir} 下无 goals.json / goals_batchK.json")
    result: list[Batch] = [batches[k] for k in sorted(batches)]
    for b in result:
        if not b.candidates.is_file():
            logger.warning(
                "批次 %d 缺 candidates（people 执行阶段跳过该批）: %s", b.batch, b.candidates
            )
    return result


def _select_batches(batches: list[Batch], batch: int | None) -> list[Batch]:
    """--batch K 限定单批；查无此批显式失败。

    Raises:
        BasketballPipelineError: 指定批次不存在。
    """
    if batch is None:
        return batches
    selected: list[Batch] = [b for b in batches if b.batch == batch]
    if not selected:
        raise BasketballPipelineError(
            f"--batch {batch} 不存在（已发现批次: {[b.batch for b in batches]}）"
        )
    return selected


def confirmed_count(goals_path: Path) -> int:
    """数 goals.json 中 status=confirmed 的条数（--max-reads 缺省换算用）。

    Raises:
        BasketballPipelineError: 顶层结构不含 goals 列表（schema 坏不静默）。
    """
    data: Any = read_json(goals_path, what=goals_path.name)
    if not isinstance(data, dict) or not isinstance(data.get("goals"), list):
        raise BasketballPipelineError(f"{goals_path}: 顶层必须是含 goals 列表的对象")
    return sum(1 for g in data["goals"] if isinstance(g, dict) and g.get("status") == "confirmed")


def resolve_out_sizes(session_dir: Path) -> tuple[str, str]:
    """读 session_facts.json 逐文件 width/height 主比例判定，返回 (1080p 尺寸, 4K 尺寸)。

    全部 ≈16:9（±1%）→ ("1920x1080", "3840x2160")；
    全部 ≈4:3（±1%）→ ("1440x1080", "2880x2160")；
    混比例或未知比例显式失败并列出各文件比例（混比例须分别合成，不自动选）。

    Returns:
        (1080p 尺寸, 4K 尺寸)，形如 ("1920x1080", "3840x2160")。

    Raises:
        BasketballPipelineError: 事实表缺失/损坏/无文件/比例混杂或未知。
    """
    facts_path: Path = session_dir / "session_facts.json"
    if not facts_path.is_file():
        raise BasketballPipelineError(f"缺 session_facts.json: {facts_path}（先跑 score）")
    facts: Any = read_json(facts_path, what="session_facts.json")
    if not isinstance(facts, dict) or not isinstance(facts.get("files"), dict):
        raise BasketballPipelineError(f"{facts_path}: 顶层必须是含 files 对象的事实表")
    files: dict[str, Any] = facts["files"]
    if not files:
        raise BasketballPipelineError(f"{facts_path}: files 为空，无法判定素材比例")
    classes: set[str] = set()
    lines: list[str] = []
    for name in sorted(files):
        info: Any = files[name]
        if not isinstance(info, dict):
            raise BasketballPipelineError(f"{facts_path}: {name} 的元数据不是对象")
        try:
            width: int = int(info["width"])
            height: int = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BasketballPipelineError(
                f"{facts_path}: {name} 缺 width/height 或不可解析: {exc}"
            ) from exc
        if width <= 0 or height <= 0:
            raise BasketballPipelineError(f"{facts_path}: {name} 尺寸非法: {width}x{height}")
        ratio: float = width / height
        if abs(ratio - RATIO_16_9) / RATIO_16_9 <= RATIO_TOLERANCE:
            cls = "16:9"
        elif abs(ratio - RATIO_4_3) / RATIO_4_3 <= RATIO_TOLERANCE:
            cls = "4:3"
        else:
            cls = "未知"
        classes.add(cls)
        lines.append(f"  {name}: {width}x{height} 比例 {ratio:.4f}（{cls}）")
    if len(classes) == 1 and "16:9" in classes:
        return OUT_16_9, OUT4K_16_9
    if len(classes) == 1 and "4:3" in classes:
        return OUT_4_3, OUT4K_4_3
    detail: str = "\n".join(lines)
    raise BasketballPipelineError(
        f"素材比例混杂或未知（{sorted(classes)}），须按比例分别合成，CLI 不自动选:\n{detail}"
    )


def build_crop_argv(
    batch: Batch,
    rawdir: Path,
    *,
    read_numbers: bool,
    max_reads: int | None,
) -> list[str]:
    """拼装 crop_scorers 命令（people 三段链与 build 自动模式共用，逐项显式拼装）。

    read_numbers=True 时 --max-reads 缺省 = 该批 confirmed 球数 ×3
    （--best-crops 默认 3，docs/scorer-reid/spec.md）；people 链默认
    read_numbers=False（photo-roster T12，v2.1 零 token 定案，显式
    --read-numbers 才开）；build 自动模式固定 read_numbers=False（读号走
    K3 烧 token，自动合集允许有误，不开）。

    Args:
        batch: 批次产物路径集合。
        rawdir: 原片目录。
        read_numbers: 是否带 --read-numbers。
        max_reads: 读号预算（None = 按 confirmed 球数 ×3 换算）。

    Returns:
        完整子进程命令（含 sys.executable 与脚本路径）。
    """
    crop_argv: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "crop_scorers.py"),
        "--goals",
        str(batch.goals),
        "--detectdir",
        str(Path("work/detect")),
        "--framesdir",
        str(Path("work/frames")),
        "--out",
        str(batch.scorers_dir),
        "--candidates",
        str(batch.candidates),
        "--rawdir",
        str(rawdir),
    ]
    if read_numbers:
        crop_argv.append("--read-numbers")
        reads: int = (
            max_reads
            if max_reads is not None
            else confirmed_count(batch.goals) * MAX_READS_PER_GOAL
        )
        crop_argv.extend(["--max-reads", str(reads)])
    return crop_argv


def load_names_players(path: Path = PHOTOS_NAMES) -> str:
    """读 photos/names.json（号码→姓名）合成 --players 名单串（号码升序）。

    tag = NAMES_TAG_PREFIX+号码（team_of_tag 白→主队自动归队，与确认页既有
    名单惯例一致）；号码归一化 str(int()) 去前导零（同照片库契约）。

    Args:
        path: names.json 路径。

    Returns:
        "白6=张三,白8=李四,..." 式名单串；文件为空对象返回空串（调用方不传参）。

    Raises:
        SchemaError: 顶层非对象 / 键非纯数字 / 值非非空 str（名单损坏显式失败）。
    """
    data: Any = read_json(path, what="names.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象（号码→姓名），实际 {type(data).__name__}")
    pairs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for k, v in data.items():
        if not isinstance(k, str) or not k.isascii() or not k.isdigit():
            raise SchemaError(f"{path}: 号码键必须是 ASCII 纯数字 str，实际 {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise SchemaError(f"{path}: 号码 {k} 的姓名必须是非空 str，实际 {v!r}")
        name: str = v.strip()
        if "," in name:
            raise SchemaError(
                f"{path}: 号码 {k} 的姓名含逗号 {name!r}（--players 串分隔符，会拆出假球员）"
            )
        n: int = int(k)
        if n in seen:
            raise SchemaError(f"{path}: 号码去零后撞车（{k} 与已有 {n}）")
        seen.add(n)
        pairs.append((n, name))
    pairs.sort()
    return ",".join(f"{NAMES_TAG_PREFIX}{n}={name}" for n, name in pairs)


def build_people_steps(
    args: argparse.Namespace,
    batch: Batch,
    rawdir: Path,
    session_dir: Path,
) -> list[Step]:
    """拼装单批次 people 链：裁图 → 传播 → 聚类 →（②.5 照片匹配）→ 确认页（spec §people）。

    --read-numbers 带上时 --max-reads 缺省 = 该批 confirmed 球数 ×3；
    --index / --roster-existing 文件存在才传；--skip-cluster 跳过聚类段且确认页
    不传 --clusters（①.5 传播不受 --skip-cluster 影响，恒在链中——传播只消费 ① 落的
    seed_* 字段，与聚类无关，docs/scorer-propagate/spec.md §Commands）。②.5 照片匹配
    （docs/photo-roster/spec.md T6 串法，T12 起执行体换 face_match_scorers.py 人脸
    单路 L1）：**默认关**（2026-08-29 用户定纯人工，review05 评测不达标），
    --photo-match 显式开 + photos/ 库存在且非 --skip-cluster 才安排（face_cache 幂等
    缓存落该批 candidates 同目录，产物落本批 photo_matches.json），缺库 INFO 跳过
    不阻塞；安排后确认页预传 --photo-matches，执行时探测产物缺失会剥掉该旗标
    （②.5 失败降级为无预填，见 _cmd_people）。确认页 --track-links 同口径预传：
    ①.5 非零即停（不降级），页面能跑到即产物已在；执行时仍探测剥旗标兜底。
    """
    crop_argv: list[str] = build_crop_argv(
        batch, rawdir, read_numbers=args.read_numbers, max_reads=args.max_reads
    )
    steps: list[Step] = [Step(f"批次{batch.batch}①裁图", tuple(crop_argv))]

    # ①.5 轨迹传播（docs/scorer-propagate/spec.md）：只消费 ① 落的 seed_* 字段，
    # 产 track_links.json 落本批 scorers 目录；detectdir/framesdir 取值同 ① 裁图段
    steps.append(
        Step(
            f"批次{batch.batch}①.5传播",
            (
                sys.executable,
                str(SCRIPT_DIR / "propagate_scorers.py"),
                "--candidates",
                str(batch.scorer_candidates),
                "--detectdir",
                str(Path("work/detect")),
                "--framesdir",
                str(Path("work/frames")),
            ),
        )
    )

    # ②.5 人脸匹配默认关（2026-08-29 用户定纯人工：review05 全场评测采纳误指认
    # 100% 不达标，错人框是一阶根因；--photo-match 显式开保留可恢复）
    photo_enabled: bool = args.photo_match and not args.skip_cluster and PHOTOS_DIR.is_dir()
    if not args.skip_cluster:
        steps.append(
            Step(
                f"批次{batch.batch}②聚类",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "cluster_scorers.py"),
                    "--candidates",
                    str(batch.scorer_candidates),
                    "--out",
                    str(batch.scorer_clusters),
                    "--linkage",
                    CLUSTER_LINKAGE,
                    "--threshold",
                    CLUSTER_THRESHOLD,
                ),
                _proxy_env_extra(),
            )
        )
    if args.photo_match and not PHOTOS_DIR.is_dir():
        logger.info("照片库 %s 不存在，跳过照片匹配步骤（不阻塞认人链）", PHOTOS_DIR)
    elif args.photo_match and args.skip_cluster:
        logger.info("--photo-match 与 --skip-cluster 同带：跳过照片匹配步骤")
    if photo_enabled:
        # ②.5 允许失败降级（allow_fail）：ERROR 留痕、确认页照出、降级为无预填；
        # 人脸 matcher（T12 换 L1）：无 --cache 参数，face_cache 落 candidates 同目录；
        # buffalo_l 权重首跑下载同样需代理（同一 _proxy_env_extra 注入）
        steps.append(
            Step(
                f"批次{batch.batch}②.5照片匹配",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "face_match_scorers.py"),
                    "--photos",
                    str(PHOTOS_DIR),
                    "--candidates",
                    str(batch.scorer_candidates),
                    "--out",
                    str(batch.photo_matches),
                ),
                _proxy_env_extra(),
                allow_fail=True,
            )
        )

    page_argv: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "gen_scorer_page.py"),
        "--scorers",
        str(batch.scorer_candidates),
        "--goals",
        str(batch.goals),
        "--session",
        session_dir.name,
    ]
    if batch.events_index.is_file():
        page_argv.extend(["--index", str(batch.events_index)])
    if not args.skip_cluster:
        page_argv.extend(["--clusters", str(batch.scorer_clusters)])
    # 预传 --track-links（①.5 产物，与 --clusters 同为链内前步产物）；执行时探测
    # 产物缺失会剥掉该旗标（与 --photo-matches 同口径，确认页照出、无传播预填）
    page_argv.extend(["--track-links", str(batch.track_links)])
    if photo_enabled:
        # 预传 --photo-matches；产物缺失（②.5 失败/降级）在执行时剥掉，确认页照出
        page_argv.extend(["--photo-matches", str(batch.photo_matches)])
    roster_path: Path = session_dir / "roster.json"
    if roster_path.is_file():
        page_argv.extend(["--roster-existing", str(roster_path)])
    if args.players_file:
        page_argv.extend(["--players-file", str(args.players_file)])
    elif PHOTOS_NAMES.is_file():
        # 全局号码→姓名名单自动注入（--players 串；显式 --players-file 优先）
        players_str: str = load_names_players()
        if players_str:
            page_argv.extend(["--players", players_str])
    steps.append(Step(f"批次{batch.batch}③确认页", tuple(page_argv)))
    return steps


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    """从命令中移除 flag 及其值各一项（确认页剥 --photo-matches / --track-links 用）。"""
    if flag not in argv:
        return argv
    i: int = argv.index(flag)
    return argv[:i] + argv[i + 2 :]


def _cmd_score(args: argparse.Namespace) -> int:
    """score：透传 run_session.py；成功后写 state + 当前场次指针（dry-run 不写）。

    --session 缺省取素材目录 basename（docs/default-session/spec.md）。
    """
    args.session = _resolve_score_session(args)
    cmd: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "run_session.py"),
        args.srcdir,
        "--session",
        args.session,
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.fids:
        cmd.extend(["--fids", args.fids])
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    try:
        run_step(cmd)
    except StepFailedError as exc:
        logger.error("score 失败: %s", exc)
        return 1
    if args.dry_run:
        return 0
    state: dict[str, Any] = load_state(args.session)
    state["session"] = args.session
    state["srcdir"] = str(Path(args.srcdir).resolve())
    state["runs"].append(
        {
            "cmd": "score",
            "at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:] if args.argv is None else args.argv,
            "exit_code": 0,
        }
    )
    save_state(args.session, state)
    logger.info("state 落盘: %s", WORK_ROOT / args.session / STATE_NAME)
    save_current_session(args.session)
    logger.info("当前场次指针: %s（session=%s）", WORK_ROOT / CURRENT_SESSION_NAME, args.session)
    return 0


def _cmd_people(args: argparse.Namespace) -> int:
    """people：逐批次链（裁图 → 传播 → 聚类 → ②.5 照片匹配 → 确认页），批次间独立。

    失败语义：①①.5②③ 任一步失败中断整链（StepFailedError 上抛转退出 1）；仅 ②.5
    照片匹配允许失败降级——ERROR 留痕后继续，确认页照出（产物缺失剥
    --photo-matches，降级为无预填；docs/photo-roster/spec.md T6 串法、T12 换人脸
    matcher 后语义不变）。确认页 --track-links 预传后执行时探测：track_links.json
    缺失剥旗标（与 --photo-matches 同口径；①.5 非零即停，正常链路跑到确认页时
    产物必在，剥离仅为兜底）。

    --session 缺省读当前场次指针（docs/default-session/spec.md）。
    """
    args.session = resolve_session(args)
    session_dir: Path = session_dir_or_die(args.session)
    state: dict[str, Any] = load_state(args.session)
    rawdir: Path = resolve_rawdir(args.rawdir, state)
    batches: list[Batch] = _select_batches(discover_batches(session_dir), args.batch)
    completed: list[str] = []
    dry_count: int = 0
    try:
        for batch in batches:
            if not batch.candidates.is_file():
                logger.warning("批次 %d 缺 candidates，跳过该批: %s", batch.batch, batch.candidates)
                continue
            if not batch.events_index.is_file():
                logger.warning(
                    "批次 %d 缺 events_index（确认页失兜底视频引用，降级继续）: %s",
                    batch.batch,
                    batch.events_index,
                )
            for step in build_people_steps(args, batch, rawdir, session_dir):
                if args.dry_run:
                    _log_dry_step(step)
                    dry_count += 1
                    continue
                argv: list[str] = list(step.argv)
                if "--photo-matches" in argv and not batch.photo_matches.is_file():
                    # 存在性探测在执行时（②.5 之后）：产物缺失 → 剥旗标，无预填照出
                    argv = _strip_flag(argv, "--photo-matches")
                if "--track-links" in argv and not batch.track_links.is_file():
                    # 存在性探测在执行时（①.5 之后）：产物缺失 → 剥旗标，无传播预填照出
                    argv = _strip_flag(argv, "--track-links")
                try:
                    run_step(argv, step.env_extra)
                except StepFailedError as exc:
                    if not step.allow_fail:
                        raise
                    logger.error("步骤失败（允许降级，继续后续步骤）: %s", exc)
                    continue
                completed.append(step.title)
    except StepFailedError as exc:
        logger.error("失败命令: %s", shlex.join(exc.cmd))
        logger.error("已完成步骤: %s", completed or "（无）")
        return 1
    if args.dry_run:
        logger.info("DRY-RUN 共 %d 步（未执行）", dry_count)
    else:
        logger.info("people 完成（%d 步）", len(completed))
    return 0


def _confirmed_keys(goals_path: Path) -> set[str]:
    """读 goals 文件，返回 confirmed 记录的 format_key 集合（--all 命中预算用）。

    缺 file/anchor_time 的坏记录跳过——结构校验是 build_highlight 的职责，
    此处只预算命中，不提前炸。

    Args:
        goals_path: goals_batchK.json 路径。

    Returns:
        confirmed 记录的 format_key 集合。
    """
    data: Any = read_json(goals_path, what=f"goals 命中预算 {goals_path.name}")
    goals: Any = data.get("goals") if isinstance(data, dict) else None
    keys: set[str] = set()
    for g in goals if isinstance(goals, list) else []:
        if not isinstance(g, dict) or g.get("status") != "confirmed":
            continue
        try:
            keys.add(format_key(g["file"], g["anchor_time"]))
        except (KeyError, TypeError, ValueError):
            # 缺键/类型坏（str anchor_time 走 f"{t:.1f}" 抛 ValueError）都跳过——
            # 结构校验是 build_highlight 的职责，此处只预算命中
            continue
    return keys


def _merge_goals_for_build(batches: list[Batch], session: str, session_dir: Path) -> Path:
    """多批次合并 goals：全记录逐字拼接，原子写 work/<场次>/merged_goals_cli.json。

    不过滤不校验（confirmed 过滤与结构校验是 build_highlight 的单一职责点）。
    每次 build 重写（素材流动，goals 会变）。

    Args:
        batches: 选定批次列表。
        session: 场次 ID（写入合并文件顶层 session 字段，build_highlight 据此定输出目录）。
        session_dir: work/<场次> 目录。

    Returns:
        合并文件路径。

    Raises:
        SchemaError: 某批 goals 顶层缺 goals 列表（结构损坏显式失败）。
    """
    merged: list[Any] = []
    for batch in batches:
        data: Any = read_json(batch.goals, what=f"批次{batch.batch} goals")
        goals: Any = data.get("goals") if isinstance(data, dict) else None
        if not isinstance(goals, list):
            raise SchemaError(f"{batch.goals}: 缺 goals 列表或类型错误")
        merged.extend(goals)
    out: Path = session_dir / MERGED_GOALS_NAME
    atomic_write_json(out, {"session": session, "goals": merged}, what="合并 goals")
    return out


def _build_expand_all(session_dir: Path, known_keys: set[str]) -> list[tuple[str, str]]:
    """--all 展开：roster 逐人 --scorer tag + 逐队 --team（team 按出现序去重）。

    零命中跳过：球员/队伍在选定批次 confirmed 键集（known_keys）内无归属球 →
    WARNING 跳过不调用（build_highlight 对零记录 exit 1，--all 遍历不能让
    单点空组合中止整轮；2026-08-15 用户实测黑后卫零球批次中止事故）。
    便服队不入分队合集（build_highlight 拒收），跳过并记 WARNING；
    便服球员个人合集有命中才出。
    对手队默认过滤：--all 不出 tag="对手" 的个人合集，也不出对手队的分队合集
    （docs/opponent-filter/spec.md T1）；显式 --team/--scorer 对手保留通道。

    Args:
        session_dir: work/<场次> 目录。
        known_keys: 选定批次 confirmed 球的 format_key 集合。

    Returns:
        (旗标, 值) 列表，旗标为 "--scorer" 或 "--team"。

    Raises:
        BasketballPipelineError: roster 不存在（提示先跑 people）。
        SchemaError: roster schema 损坏（validate_roster 抛出，不静默）。
    """
    roster_path: Path = session_dir / "roster.json"
    if not roster_path.is_file():
        raise BasketballPipelineError(f"roster 不存在: {roster_path}（先跑 people 确认导出）")
    roster = validate_roster(read_json(roster_path, what="roster.json"), str(roster_path))
    tag_keys: dict[str, set[str]] = {}
    for key, tag in roster.assignments.items():
        tag_keys.setdefault(tag, set()).add(key)
    hit_tags: set[str] = {t for t, ks in tag_keys.items() if ks & known_keys}
    opponent_team: str = opponent_team_name(session_dir)
    pairs: list[tuple[str, str]] = []
    for p in roster.players:
        if is_opponent_tag(p.tag) or p.team == opponent_team:
            logger.warning(
                "--all 跳过对手球员: %s (team=%s)，默认产物不含对手进球",
                p.tag,
                p.team,
            )
            continue
        if p.tag not in hit_tags:
            logger.warning("--all 跳过零命中球员: %s（选定批次内无归属球）", p.tag)
            continue
        pairs.append(("--scorer", p.tag))
    teams: list[str] = []
    casual_skipped: bool = False
    opponent_team_skipped: bool = False
    warned_teams: set[str] = set()  # 零命中队伍只 WARNING 一次（不进 teams 去重失效）
    for p in roster.players:
        if p.team == CASUAL_TEAM:
            casual_skipped = True
            continue
        if p.team == opponent_team:
            if not opponent_team_skipped:
                logger.warning("--all 跳过对手队分队合集: %s，默认产物不含对手进球", opponent_team)
                opponent_team_skipped = True
            continue
        if not p.team or p.team in teams or p.team in warned_teams:
            continue
        if any(q.team == p.team and q.tag in hit_tags for q in roster.players):
            teams.append(p.team)
        else:
            warned_teams.add(p.team)
            logger.warning("--all 跳过零命中队伍: %s（选定批次内无归属球）", p.team)
    if casual_skipped:
        logger.warning("--all 跳过便服分队合集（build_highlight 拒收；便服球员个人合集照常出）")
    pairs.extend(("--team", t) for t in teams)
    return pairs


def _cmd_build(args: argparse.Namespace) -> int:
    """build：尺寸按 session_facts 主比例换算；按 roster 状态分派两条路径。

    roster 存在且 confirmed=true → 现状路径（_cmd_build_confirmed，行为零改动）；
    缺失或 confirmed=false → 自动模式（_cmd_build_auto，三产物链，认人可选化
    2026-08-22，spec: docs/build-auto-scorer/spec.md）；roster schema 损坏
    validate_roster 抛 SchemaError 显式失败（不降级不静默，rules.md §0.2）。

    --session 缺省读当前场次指针（docs/default-session/spec.md）。
    """
    args.session = resolve_session(args)
    session_dir: Path = session_dir_or_die(args.session)
    state: dict[str, Any] = load_state(args.session)
    rawdir: Path = resolve_rawdir(args.rawdir, state)
    size_hd, size_4k = resolve_out_sizes(session_dir)
    batches: list[Batch] = _select_batches(discover_batches(session_dir), args.batch)
    roster_path: Path = session_dir / "roster.json"
    roster_confirmed: bool = False
    if roster_path.is_file():
        roster = validate_roster(read_json(roster_path, what="roster.json"), str(roster_path))
        roster_confirmed = roster.confirmed
    if roster_confirmed:
        return _cmd_build_confirmed(
            args, session_dir, rawdir, size_hd, size_4k, batches, roster_path
        )
    if args.four_k:
        logger.warning("未认人自动模式忽略 --4k（三件套口径不变；先认人再用 4K 档）")
    return _cmd_build_auto(args, session_dir, rawdir, size_hd, batches)


def _cmd_build_confirmed(
    args: argparse.Namespace,
    session_dir: Path,
    rawdir: Path,
    size_hd: str,
    size_4k: str,
    batches: list[Batch],
    roster_path: Path,
) -> int:
    """build 现状路径（roster confirmed=true）：逐 filter 调 build_highlight。

    --all 展开 roster 逐人 + 逐队；多批次合并 goals 后每 filter 只调一次；
    收尾触发热图双风格（自动模式不触发，见 _cmd_build_auto）。

    4K 档（docs/build-4k/spec.md）：主队 队伍集锦 默认 4K；--4k 手动重出
    其余所选产物为 4K；主队步骤在 --4k 下为幂等 no-op（原名无后缀）。
    主队名取 work/<场次>/team_config.json（缺失回退 team_config 默认值）。
    """
    our_team: str = load_team_config(session_dir).team_name
    filters: list[tuple[str, str]]
    if args.all:
        known_keys: set[str] = set()
        for batch in batches:
            known_keys |= _confirmed_keys(batch.goals)
        filters = _build_expand_all(session_dir, known_keys)
        if not filters:
            raise BasketballPipelineError(
                f"--all 无合集可出（roster players 为空或选定批次内均无归属球）: {session_dir}"
            )
    elif args.scorer:
        filters = [("--scorer", args.scorer)]
    elif args.team:
        filters = [("--team", args.team)]
    else:
        filters = [("", "")]
    # 多批次合并：输出名由 session+filter 决定、不含批次，逐批调会互相覆盖
    # （且零球批次 exit 1 中止整轮）；合并后每 filter 只调一次，build_highlight
    # 内部按 (file, anchor_time) 排序——文件名即时间戳，跨批排序天然正确
    jobs: list[tuple[Path, str]]
    if len(batches) > 1:
        merged_path: Path = session_dir / MERGED_GOALS_NAME
        if not args.dry_run:
            merged_path = _merge_goals_for_build(batches, args.session, session_dir)
        jobs = [(merged_path, "合并批次")]
    else:
        jobs = [(b.goals, f"批次{b.batch}") for b in batches]
    completed: list[str] = []
    dry_count: int = 0
    try:
        for goals_path, batch_label in jobs:
            base: list[str] = [
                sys.executable,
                str(SCRIPT_DIR / "build_highlight.py"),
                "--goals",
                str(goals_path),
            ]
            if roster_path.is_file():
                base.extend(["--roster", str(roster_path)])
            base.extend(["--rawdir", str(rawdir)])
            for flag, value in filters:
                # 4K 档：主队集锦默认 4K 原名；--4k 时其余步骤 4K+后缀；主队 no-op
                step_out: str = size_hd
                extra: list[str] = []
                if flag == "--team" and value == our_team:
                    step_out = size_4k
                    if args.four_k:
                        logger.info("主队集锦已默认 4K，--4k 对 %s 为 no-op", our_team)
                elif args.four_k:
                    step_out = size_4k
                    extra = ["--name-suffix", FOUR_K_SUFFIX]
                cmd: list[str] = [*base, "--out", step_out, *extra]
                if flag:
                    cmd.extend([flag, value])
                title: str = (
                    f"{batch_label} 合成{(' ' + flag + ' ' + value) if flag else '（全员）'}"
                )
                if args.four_k or (flag == "--team" and value == our_team):
                    logger.info(
                        "4K 步骤（--out %s，CPU 编码约为 1080p 的 3~4 倍耗时）: %s",
                        step_out,
                        title,
                    )
                if args.dry_run:
                    _log_dry_step(Step(title, tuple(cmd)))
                    dry_count += 1
                    continue
                run_step(cmd)
                completed.append(title)
    except StepFailedError as exc:
        logger.error("失败命令: %s", shlex.join(exc.cmd))
        logger.error("已完成步骤: %s", completed or "（无）")
        return 1
    if args.dry_run:
        _log_dry_step(
            Step(
                "热图双风格（goal_heatmap）",
                (str(SCRIPT_DIR / "goal_heatmap.py"), "--sessiondir", str(session_dir)),
            )
        )
        logger.info("DRY-RUN 共 %d 步（未执行，1080p=%s / 4K=%s）", dry_count + 1, size_hd, size_4k)
    else:
        logger.info("build 完成（%d 步，1080p=%s / 4K=%s）", len(completed), size_hd, size_4k)
        _run_heatmap_step(session_dir, roster_path)
    return 0


def _scorer_candidates_ready(path: Path) -> bool:
    """crop 幂等判定：scorer_candidates.json 存在且 JSON 可读即跳过（仿 run_session
    断点口径）；存在但不可读 WARNING 后按未产处理（重跑裁图覆盖）。

    Args:
        path: 批次 scorer_candidates.json 路径。

    Returns:
        True = 产物可用可跳过；False = 需跑 crop_scorers。
    """
    if not path.is_file():
        return False
    try:
        read_json(path, what=path.name)
    except (BasketballPipelineError, OSError) as exc:
        logger.warning("scorer_candidates 存在但不可读，重跑裁图: %s (%s)", path, exc)
        return False
    return True


def _auto_roster_hit_groups(roster_path: Path, known_keys: set[str]) -> tuple[list[str], list[str]]:
    """读 auto_roster.json，返回在选定批次 confirmed 键集内有归属球的 (tag 列表, 颜色队列表)。

    零命中预算跳过（仿 _build_expand_all 口径：build_highlight 对零记录
    exit 1，不能让单点空组合中止整轮）；便服队不进队伍集锦循环
    （build_highlight 真值表⑧拒收），有命中便服簇即 INFO 留痕一行
    （允许有误口径：全员便服场次只出个人合集，见 spec 风险表）。

    Args:
        roster_path: work/<场次>/auto_roster.json 路径。
        known_keys: 选定批次 confirmed 球的 format_key 集合。

    Returns:
        (有归属球的 tag（players 顺序）, 有归属球的非便服队别（players 出现序去重）)。

    Raises:
        SchemaError: auto_roster.json schema 损坏（validate_roster 抛出，不静默）。
    """
    roster = validate_roster(read_json(roster_path, what="auto_roster.json"), str(roster_path))
    hit: set[str] = {tag for key, tag in roster.assignments.items() if key in known_keys}
    tags: list[str] = []
    teams: list[str] = []
    casual_noted: bool = False
    for p in roster.players:
        if p.tag not in hit:
            logger.warning("跳过零命中簇: %s（选定批次内无归属球）", p.tag)
            continue
        tags.append(p.tag)
        if p.team == CASUAL_TEAM:
            casual_noted = True
        elif p.team not in teams:
            teams.append(p.team)
    if casual_noted:
        logger.info("便服队不进队伍集锦（真值表⑧），其簇只出个人合集")
    return tags, teams


def _cmd_build_auto(
    args: argparse.Namespace,
    session_dir: Path,
    rawdir: Path,
    out_size: str,
    batches: list[Batch],
) -> int:
    """build 自动模式（无 confirmed roster）：一条命令出三产物（认人可选化）。

    链（spec: docs/build-auto-scorer/spec.md §技术现状；2026-08-22 用户改定：
    不出全员集锦，改按球衣颜色分队出队伍集锦）：
    ① build_highlight --per-goal → 进球片段/；
    ② 逐批 crop_scorers（不带 --read-numbers；产物可读即幂等跳过）；
    ③ cluster_scorers 跨批合并（定稿 complete/0.15，--out scorers_auto/，
       每次重跑靠 clip_cache 免重复 CLIP 推理）；
    ④ auto_roster.py（带 --candidates 各批票源做簇内颜色分队多数票）
       → work/<场次>/auto_roster.json；
    ⑤ 逐颜色队 build_highlight --roster auto_roster.json --team <队>
       --allow-unconfirmed → 队伍_<队>_进球集锦.mp4（便服队不进循环——真值表⑧
       口径；零命中队预算跳过，仿零命中 tag 口径）；
    ⑥ 逐簇 build_highlight --roster auto_roster.json --scorer <tag>
       --allow-unconfirmed → <队>_<tag>_进球合集.mp4（零命中簇预算跳过；
       0 簇不出队伍/个人合集）；
    ⑦ 热图跳过（goal_heatmap 无 confirmed 检查，未确认 roster 上不新触发）。

    ① 失败即中止退出 1；②-⑥ 任一步失败 ERROR 留痕、跳过剩余识别步骤、
    已产出的 ① 保留、退出 1（不静默降级）。某批缺 candidates.json WARNING
    跳过该批（同 _cmd_people 口径）；全部缺则跳过整条识别链，① 照常 exit 0。
    """
    logger.warning(
        "roster 缺失或未 confirmed=true：按未认人处理，出队伍集锦（颜色分队）/进球片段/自动个人合集"
    )
    if args.all or args.scorer or args.team:
        logger.warning("未认人模式下过滤旗标（--all/--scorer/--team）被忽略")
    # goals：多批合并（与 confirmed 路径同口径；跨批排序在 build_highlight 内）
    goals_path: Path
    if len(batches) > 1:
        goals_path = session_dir / MERGED_GOALS_NAME
        if not args.dry_run:
            goals_path = _merge_goals_for_build(batches, args.session, session_dir)
    else:
        goals_path = batches[0].goals
    base: list[str] = [
        sys.executable,
        str(SCRIPT_DIR / "build_highlight.py"),
        "--goals",
        str(goals_path),
        "--rawdir",
        str(rawdir),
        "--out",
        out_size,
    ]
    main_steps: list[Step] = [
        Step("进球片段（--per-goal）", tuple([*base, "--per-goal"])),
    ]
    completed: list[str] = []
    dry_count: int = 0
    try:
        for step in main_steps:
            if args.dry_run:
                _log_dry_step(step)
                dry_count += 1
                continue
            run_step(list(step.argv), step.env_extra)
            completed.append(step.title)
    except StepFailedError as exc:
        logger.error("失败命令: %s", shlex.join(exc.cmd))
        logger.error("已完成步骤: %s", completed or "（无）")
        return 1

    # ② 裁图：逐批（缺 candidates 批次 WARNING 跳过；产物可读幂等跳过）
    identify_steps: list[Step] = []
    crop_batches: list[Batch] = []
    for batch in batches:
        if not batch.candidates.is_file():
            logger.warning("批次 %d 缺 candidates，跳过该批裁图: %s", batch.batch, batch.candidates)
            continue
        crop_batches.append(batch)
        if _scorer_candidates_ready(batch.scorer_candidates):
            logger.info(
                "批次 %d 裁图产物已存在且可读，幂等跳过: %s", batch.batch, batch.scorer_candidates
            )
            continue
        identify_steps.append(
            Step(
                f"批次{batch.batch}裁图（自动）",
                tuple(build_crop_argv(batch, rawdir, read_numbers=False, max_reads=None)),
            )
        )
    if not crop_batches:
        logger.warning("所有批次缺 candidates，无可聚类候选——跳过自动识别（队伍/个人合集不出）")
    else:
        auto_dir: Path = session_dir / "scorers_auto"
        auto_clusters: Path = auto_dir / "scorer_clusters.json"
        auto_roster_path: Path = session_dir / "auto_roster.json"
        # ③ 聚类：单次合并全部批次 candidates，天然跨批簇标一致；每次重跑
        cluster_argv: list[str] = [sys.executable, str(SCRIPT_DIR / "cluster_scorers.py")]
        for batch in crop_batches:
            cluster_argv.extend(["--candidates", str(batch.scorer_candidates)])
        cluster_argv.extend(
            [
                "--out",
                str(auto_clusters),
                "--linkage",
                CLUSTER_LINKAGE,
                "--threshold",
                CLUSTER_THRESHOLD,
            ]
        )
        identify_steps.append(Step("自动聚类", tuple(cluster_argv), _proxy_env_extra()))
        # ④ 聚类+颜色分队票源 → auto_roster.json（confirmed=false；team=簇内多数票）
        auto_roster_argv: list[str] = [
            sys.executable,
            str(SCRIPT_DIR / "auto_roster.py"),
            "--clusters",
            str(auto_clusters),
        ]
        for batch in crop_batches:
            auto_roster_argv.extend(["--candidates", str(batch.scorer_candidates)])
        auto_roster_argv.extend(["--session", args.session, "--out", str(auto_roster_path)])
        identify_steps.append(Step("自动 roster", tuple(auto_roster_argv)))
        identify_ok: bool = True
        try:
            for step in identify_steps:
                if args.dry_run:
                    _log_dry_step(step)
                    dry_count += 1
                    continue
                run_step(list(step.argv), step.env_extra)
                completed.append(step.title)
        except StepFailedError as exc:
            logger.error("自动识别链失败（已产出的进球片段保留）: %s", shlex.join(exc.cmd))
            identify_ok = False
        # ⑤⑥ 逐颜色队队伍集锦 + 逐簇个人合集（真值表⑤/④ + ⑩ 闸门）
        if identify_ok:
            if args.dry_run:
                _log_dry_step(
                    Step(
                        "队伍集锦（每颜色队一次，便服除外，队别由 auto_roster 决定）",
                        (
                            str(SCRIPT_DIR / "build_highlight.py"),
                            "--roster",
                            str(auto_roster_path),
                            "--team",
                            "<队>",
                            "--allow-unconfirmed",
                        ),
                    )
                )
                _log_dry_step(
                    Step(
                        "个人合集（每簇一次，tag 由 auto_roster 决定）",
                        (
                            str(SCRIPT_DIR / "build_highlight.py"),
                            "--roster",
                            str(auto_roster_path),
                            "--scorer",
                            "<簇标>",
                            "--allow-unconfirmed",
                        ),
                    )
                )
                dry_count += 2
            else:
                if not auto_roster_path.is_file():
                    logger.error("auto_roster 步骤成功但产物缺失: %s", auto_roster_path)
                    return 1
                known_keys: set[str] = set()
                for batch in batches:
                    known_keys |= _confirmed_keys(batch.goals)
                tags, teams = _auto_roster_hit_groups(auto_roster_path, known_keys)
                if not tags:
                    logger.info("auto_roster 无归属球（0 簇或全零命中），跳过队伍/个人合集")
                jobs: list[tuple[str, str, str]] = [
                    *[("--team", t, f"队伍_{t}_进球集锦") for t in teams],
                    *[("--scorer", tag, f"簇{tag}_个人合集") for tag in tags],
                ]
                for flag, value, title in jobs:
                    cmd: list[str] = [
                        sys.executable,
                        str(SCRIPT_DIR / "build_highlight.py"),
                        "--goals",
                        str(goals_path),
                        "--roster",
                        str(auto_roster_path),
                        flag,
                        value,
                        "--allow-unconfirmed",
                        "--rawdir",
                        str(rawdir),
                        "--out",
                        out_size,
                    ]
                    try:
                        run_step(cmd)
                        completed.append(title)
                    except StepFailedError as exc:
                        logger.error(
                            "自动识别链失败（已产出的进球片段保留）: %s",
                            shlex.join(exc.cmd),
                        )
                        return 1
        elif not args.dry_run:
            return 1
    if args.dry_run:
        logger.info("DRY-RUN 共 %d 步（未执行，--out %s）", dry_count, out_size)
    else:
        logger.info(
            "build 自动模式完成（%d 步，--out %s；热图未认人不触发）", len(completed), out_size
        )
    return 0


def _run_heatmap_step(session_dir: Path, roster_path: Path) -> None:
    """build 收尾触发热图双风格（v4.2 集成，docs/heatmap/spec.md；附属产物不阻塞主链）。

    roster.json 缺失 = 尚未认人（预期常态）INFO 跳过；heat_session 任何异常
    log ERROR 留痕但本函数不抛出、build 返回码不变（不静默——rules.md §0.2）。
    目录推导（detect/frames/output）收在 goal_heatmap.heat_session 侧（S3），
    此处只传 session_dir。

    Args:
        session_dir: work/<场次> 目录。
        roster_path: 该场次 roster.json 路径。
    """
    if not roster_path.is_file():
        logger.info("热图跳过：roster.json 不存在（先跑 people 认人）")
        return
    import goal_heatmap  # 懒 import（S4）：防 score/people/photo 白付 cv2/numpy 导入成本

    try:
        goal_heatmap.heat_session(session_dir)
        logger.info("热图已出（暗场+分区双风格）: output/%s/", session_dir.name)
    except Exception:  # 附属产物任何失败都不阻塞主链，但必须留痕
        logger.error("热图生成失败（build 主链不受影响）: %s", session_dir.name, exc_info=True)


def _cmd_photo(args: argparse.Namespace) -> int:
    """photo：精彩照片链路——rank（打分→抽帧裁切）→ page（瀑布流确认页）。

    --apply 时只跑落盘段（selections 约定路径 work/<场次>/photos/photo_selections.json）；
    否则 rank + page 两步。rank 缺缓存/缺原片的文件由底层 WARNING 跳过。

    --session 缺省读当前场次指针（docs/default-session/spec.md）。
    """
    args.session = resolve_session(args)
    session_dir_or_die(args.session)
    steps: list[Step] = []
    if args.apply:
        steps.append(
            Step(
                "照片落盘",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "rank_photos.py"),
                    "--session",
                    args.session,
                    "--apply",
                ),
            )
        )
    else:
        state: dict[str, Any] = load_state(args.session)
        rawdir: Path = resolve_rawdir(args.rawdir, state)
        rank_argv: list[str] = [
            sys.executable,
            str(SCRIPT_DIR / "rank_photos.py"),
            "--session",
            args.session,
            "--rawdir",
            str(rawdir),
        ]
        if args.total is not None:
            rank_argv.extend(["--total", str(args.total)])
        steps.append(Step("照片打分抽帧", tuple(rank_argv)))
        steps.append(
            Step(
                "照片确认页",
                (
                    sys.executable,
                    str(SCRIPT_DIR / "gen_photo_page.py"),
                    "--session",
                    args.session,
                ),
            )
        )
    completed: list[str] = []
    dry_count: int = 0
    try:
        for step in steps:
            if args.dry_run:
                _log_dry_step(step)
                dry_count += 1
                continue
            run_step(list(step.argv), step.env_extra)
            completed.append(step.title)
    except StepFailedError as exc:
        logger.error("失败命令: %s", shlex.join(exc.cmd))
        logger.error("已完成步骤: %s", completed or "（无）")
        return 1
    if args.dry_run:
        logger.info("DRY-RUN 共 %d 步（未执行）", dry_count)
    else:
        logger.info("photo 完成（%d 步）", len(completed))
    return 0


def _dir_stats(path: Path) -> tuple[int, int]:
    """递归统计目录总字节数与文件数（os.scandir；单点失败 WARNING 按 0 计不中断）。

    Args:
        path: 目标目录。

    Returns:
        (总字节数, 文件数)。
    """
    total: int = 0
    nfiles: int = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        sub_size, sub_n = _dir_stats(Path(entry.path))
                        total += sub_size
                        nfiles += sub_n
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        nfiles += 1
                except OSError as exc:
                    logger.warning("统计跳过 %s: %s", entry.path, exc)
    except OSError as exc:
        logger.warning("统计跳过 %s: %s", path, exc)
    return total, nfiles


def _fmt_gb(n: int) -> str:
    """字节数 → GB 两位小数字符串（清单展示用）。"""
    return f"{n / (1024**3):.2f}GB"


def _rmtree_force(path: Path) -> None:
    """删目录树，容忍只读文件（git pack / 检出物在 Windows 上带只读位，
    直接 rmtree 会 WinError 5）：onexc/onerror 回调去掉只读位后重试同一操作。
    """

    def _onexc(func: Callable[[str], object], p: str, exc: BaseException) -> None:
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onexc)
    else:
        # 3.10/3.11 无 onexc 关键字；onerror 回调第三参为 sys.exc_info() 元组，此处不用
        shutil.rmtree(
            path,
            onerror=lambda func, p, _exc_info: _onexc(func, p, _exc_info[1]),
        )


def _collect_srcdir_targets() -> list[Path]:
    """从各场次 state 收集源视频目录（清 work 之前先读）；逐个过守卫，不合格剔除。

    守卫（拒删并 ERROR/WARNING）：srcdir = 仓库根**或是仓库根的祖先**
    （`p == REPO_ROOT or p in REPO_ROOT.parents`，防 rmtree 连仓库一起删；
    srcdir 是仓库根的子目录属合法，如素材目录就挂在仓库根下）/ 盘符根 /
    不存在或不是目录。无 state 或无 srcdir → WARNING 不猜路径。

    Returns:
        过守卫的源视频目录列表（去重，按 state 文件名序）。
    """
    targets: list[Path] = []
    for state_path in sorted(WORK_ROOT.glob(f"*/{STATE_NAME}")):
        try:
            data: Any = read_json(state_path, what=f"state {state_path.name}")
        except (BasketballPipelineError, OSError) as exc:
            logger.warning("state 读取失败，跳过其 srcdir: %s (%s)", state_path, exc)
            continue
        src: Any = data.get("srcdir") if isinstance(data, dict) else None
        if not src or not isinstance(src, str):
            continue
        p: Path = Path(src).resolve()
        if p == REPO_ROOT or p in REPO_ROOT.parents:
            logger.error("srcdir 守卫拒删（指向仓库根或其祖先）: %s", p)
            continue
        if p.parent == p:
            logger.error("srcdir 守卫拒删（盘符根）: %s", p)
            continue
        if not p.is_dir():
            logger.warning("srcdir 不存在或不是目录，跳过: %s", p)
            continue
        if p not in targets:
            targets.append(p)
    if not targets:
        logger.warning("未从任何 state 读到 srcdir——源视频目录不删（不猜路径）")
    return targets


def _cmd_clean(args: argparse.Namespace) -> int:
    """clean：清空 output/ 内容 + work/ 内容 + 源视频目录（列清单 + yes 精确确认）。

    顺序：先收集源视频目录（读 state.srcdir，在清 work 前）→ 列三分组清单
    （路径/大小/文件数）→ --dry-run 或确认词非精确 yes 则零删除。单目标删除
    失败记 ERROR 继续其余，结尾汇总，有失败退出 1。非 tty 拒绝执行（防挂起
    在确认输入）。
    """
    srcdirs: list[Path] = _collect_srcdir_targets()
    groups: list[tuple[str, list[Path]]] = [
        ("output/", sorted(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.is_dir() else []),
        ("work/", sorted(WORK_ROOT.iterdir()) if WORK_ROOT.is_dir() else []),
        ("源视频", srcdirs),
    ]
    plan: list[Path] = []
    total_bytes: int = 0
    logger.info("=== clean 清单 ===")
    for label, paths in groups:
        logger.info("[%s] %d 项", label, len(paths))
        for p in paths:
            try:
                size, nfiles = _dir_stats(p) if p.is_dir() else (p.stat().st_size, 1)
            except OSError as exc:
                # 统计失败按 0 展示不中断（spec 口径）；目标仍进 plan 照常尝试删除
                logger.warning("统计跳过 %s: %s", p, exc)
                size, nfiles = 0, 0
            total_bytes += size
            plan.append(p)
            logger.info("  %s  %s  %d 文件", p, _fmt_gb(size), nfiles)
    logger.info("合计释放: %s", _fmt_gb(total_bytes))
    if not plan:
        logger.info("无可清理内容")
        return 0
    if args.dry_run:
        logger.info("DRY-RUN：未删除任何内容")
        return 0
    if not sys.stdin.isatty():
        logger.error("非交互环境拒绝执行 clean（防挂起在确认输入）；先看 --dry-run")
        return 1
    ans: str = input("以上全部删除（含源视频，不可恢复）。输入 yes 确认: ")
    if ans != "yes":
        logger.info("未确认（需精确输入 yes），未动任何文件")
        return 0
    failed: list[str] = []
    for p in plan:
        try:
            if p.is_dir() and not p.is_symlink():
                _rmtree_force(p)
            else:
                p.unlink()
        except OSError as exc:
            logger.error("删除失败: %s (%s)", p, exc)
            failed.append(str(p))
    if failed:
        logger.error("clean 完成但有 %d 项删除失败: %s", len(failed), failed)
        return 1
    logger.info("clean 完成：释放 %s，工作区已恢复全新", _fmt_gb(total_bytes))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构建三级 argparse：prog → 子命令 → 各自参数。"""
    ap = argparse.ArgumentParser(
        prog="video",
        description="篮球视频统一入口：score（检测）→ people（认人）→ build（合集）",
    )
    sub = ap.add_subparsers(dest="command")

    sc = sub.add_parser("score", help="检测链路：透传 run_session.py 至标注页生成")
    sc.add_argument("srcdir", help="原片目录（递归扫描 .mp4）")
    sc.add_argument(
        "--session", default=None, help="场次 ID（缺省取素材目录 basename；成功后记为当前场次）"
    )
    sc.add_argument("--batch-size", type=int, default=None, help="每批文件数（缺省透传底层默认）")
    sc.add_argument("--fids", default="", help="逗号分隔 fid 清单（adhoc 模式）")
    sc.add_argument("--force", action="store_true", help="忽略断点产物全部重算")
    sc.add_argument("--dry-run", action="store_true", help="只打印不执行（不写 state）")
    sc.set_defaults(func=_cmd_score)

    pp = sub.add_parser("people", help="认人链路：裁图 → 聚类 → 确认页（逐批次）")
    pp.add_argument(
        "--session", default=None, help="场次 ID（缺省读当前场次指针 work/current_session.json）"
    )
    pp.add_argument("--batch", type=int, default=None, help="限定单批次 K")
    pp.add_argument("--rawdir", default=None, help="原片目录（缺省读 state.srcdir）")
    pp.add_argument(
        "--read-numbers",
        action="store_true",
        help="K3 读号（默认关，显式开启；v2.1 零 token 定案）",
    )
    pp.add_argument(
        "--no-read-numbers",
        action="store_false",
        dest="read_numbers",
        help="关闭 K3 读号（现默认已关，旗标保留兼容）",
    )
    # 缺省 False 必须靠 set_defaults 兜底：argparse 填默认值带 hasattr 守卫、
    # 先注册者胜出——本排列下先注册的 --read-numbers（store_true 隐式 default
    # False）会压住 --no-read-numbers（store_false 隐式 default True），缺省
    # 恰好得 False；set_defaults 显式钉死语义，防注册顺序调整静默翻车
    # （read-numbers-batch review01 B1 同类坑，本机实证）
    pp.set_defaults(read_numbers=False)
    pp.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="读号新调用上限（缺省 = 该批 confirmed 球数 ×3）",
    )
    pp.add_argument("--players-file", default=None, help="球员名单 JSON 文件")
    pp.add_argument(
        "--photo-match",
        action="store_true",
        help="开人脸匹配预填（默认关：2026-08-29 用户定纯人工，review05 评测不达标；"
        "显式开才串联 ②.5 人脸匹配）",
    )
    pp.add_argument(
        "--skip-cluster", action="store_true", help="跳过聚类段（确认页不传 --clusters）"
    )
    pp.add_argument("--dry-run", action="store_true", help="只打印不执行")
    pp.set_defaults(func=_cmd_people)

    bd = sub.add_parser("build", help="合成链路：build_highlight 全员/单人/单队/全量合集")
    bd.add_argument(
        "--session", default=None, help="场次 ID（缺省读当前场次指针 work/current_session.json）"
    )
    bd.add_argument("--batch", type=int, default=None, help="限定单批次 K")
    bd.add_argument("--rawdir", default=None, help="原片目录（缺省读 state.srcdir）")
    grp = bd.add_mutually_exclusive_group()
    grp.add_argument("--scorer", default="", help="单个人合集（tag 或姓名）")
    grp.add_argument("--team", default="", help="单队伍合集")
    grp.add_argument("--all", action="store_true", help="roster 逐人 + 逐队全量合集")
    bd.add_argument("--dry-run", action="store_true", help="只打印不执行")
    bd.add_argument(
        "--4k",
        dest="four_k",
        action="store_true",
        help="手动重出 4K：所选产物（--scorer/--team/--all 或不带=全员）出 4K 并加 _4K 后缀；"
        "主队集锦本已默认 4K，对该队为 no-op；未认人自动模式忽略",
    )
    bd.set_defaults(func=_cmd_build)

    ph = sub.add_parser("photo", help="精彩照片：打分 → 抽帧裁切 → 确认页 / --apply 落盘精选")
    ph.add_argument(
        "--session", default=None, help="场次 ID（缺省读当前场次指针 work/current_session.json）"
    )
    ph.add_argument("--rawdir", default=None, help="原片目录（缺省读 state.srcdir）")
    ph.add_argument("--total", type=int, default=None, help="候选目标张数（缺省透传底层 200）")
    ph.add_argument(
        "--apply",
        action="store_true",
        help="落盘模式：按 work/<场次>/photos/photo_selections.json 出照片精选",
    )
    ph.add_argument("--dry-run", action="store_true", help="只打印不执行")
    ph.set_defaults(func=_cmd_photo)

    cl = sub.add_parser(
        "clean", help="清空 output/work/源视频，恢复全新工作区（列清单 + yes 确认）"
    )
    cl.add_argument("--dry-run", action="store_true", help="只列清单不删除")
    cl.set_defaults(func=_cmd_clean)
    return ap


def _resolve_user_paths(args: argparse.Namespace, launch_cwd: Path) -> None:
    """把用户传入的相对路径参数解析为绝对路径（chdir 到仓库根之前调用）。

    支持从任意目录调用：main 启动后统一 chdir 到 REPO_ROOT（work/ 等相对路径
    基准），用户给的相对路径必须先按启动目录解析，否则会被错误地相对到仓库根。

    Args:
        args: 已解析的命令行命名空间（原地修改 srcdir/rawdir/players_file）。
        launch_cwd: 进程启动目录。
    """
    for attr in ("srcdir", "rawdir", "players_file"):
        value: str | None = getattr(args, attr, None)
        if value:
            p: Path = Path(value)
            setattr(args, attr, str(p if p.is_absolute() else (launch_cwd / p).resolve()))


def main(argv: list[str] | None = None, *, relocate: bool = False) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=失败；2=无子命令）。

    Args:
        argv: 参数列表（None 取 sys.argv）。
        relocate: True 时按启动目录解析用户相对路径参数并 chdir 到 REPO_ROOT
            （真实命令行入口用）；测试与库内调用传 False 保持当前目录。
    """
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    args.argv = argv
    if relocate:
        _resolve_user_paths(args, Path.cwd())
        os.chdir(REPO_ROOT)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        return int(args.func(args))
    except BasketballPipelineError as exc:
        logger.error("失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    # 管道/重定向时 stdout 回落 locale 编码（cp1252/GBK），打印中文 help/日志会
    # UnicodeEncodeError（docs/经验教训.md §6）；交互控制台保持原生编码不动
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure") and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(relocate=True))
