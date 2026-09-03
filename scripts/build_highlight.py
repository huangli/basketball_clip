#!/usr/bin/env python3
"""个人进球合集合成（按剪辑规格）。

按 goals.json 中的记录切片：窗口 [anchor-4s, anchor+2s]，50fps、H.264+AAC；
输出尺寸按场次素材比例注入（--out，默认 1440x1080 对应 4:3 素材，
16:9 素材用 1920x1080），缩放保持宽高比、不足处黑边补齐不压扁；
100fps 素材入网后 2 秒半速慢放（slowmo=true 时两段拼接）。
同参数 concat 直接重封装。产物：output/<场次>/<队伍>_<姓名>_进球合集.mp4。

输入：--goals 指定的 goals.json（status=confirmed 记录；schema 损坏抛 SchemaError）；
    --roster 指定的 roster.json（可选，校验走 scripts/roster.py，必须 confirmed=true，
    除非给 --allow-unconfirmed 豁免）
输出：output/<场次>/<队伍>_<姓名>_进球合集.mp4（roster 路径；姓名为空回退标签）
    或 个人_<标签>_进球合集.mp4（无 roster 给 --scorer 旧路径）
    或 个人_全员_进球合集.mp4（无 roster 无过滤① / 有 roster 无过滤③，同名）
    或 队伍_<队别>_进球集锦.mp4
    或 output/<场次>/进球片段/NNN_<主名>@<t:.1f>s.mp4（--per-goal，真值表⑨）
依赖：scripts/errors.py、scripts/pipe_common.py（run_ffmpeg/read_json/日志）、
    scripts/roster.py（format_key/validate_roster/resolve_scorer，spec M3 契约）
用法:
    python scripts/build_highlight.py --goals work/pilot/goals.json --scorer 张三
    python scripts/build_highlight.py --goals work/20260722/goals.json \
        --rawdir "素材目录" --out 1920x1080
    python scripts/build_highlight.py --goals work/20260722/goals.json \
        --roster work/20260722/roster.json --out 1920x1080 --team 黑
    python scripts/build_highlight.py --goals work/20260722/goals.json \
        --rawdir <原片目录> --out 1920x1080 --per-goal

组合真值表（①-⑧ spec: docs/scorer/spec.md §build_highlight 组合真值表；
⑨⑩ spec: docs/build-auto-scorer/spec.md §技术现状；写死）：
①无 roster 无过滤=全员（stem 同③）；②无 roster 给 --scorer=旧 goals.scorer 精确
匹配+0 命中 WARNING 提示改用 --roster；③有 roster 无过滤=全归属球（未归属
WARNING 跳过；stem 个人_全员_进球合集）；④--scorer 经 roster.resolve_scorer
解析（tag|name），输出名用 {team}_{name or tag}；⑤--team 出 队伍_{team}_进球集锦.mp4；
⑥--scorer+--team 互斥报错退出 1；⑦无 roster 给 --team 报错退出 1；⑧--team 便服
报错退出 1；⑨--per-goal=每 confirmed 球独立出片到 进球片段/NNN_<主名>@<t:.1f>s.mp4
（不 concat；与 --scorer/--team/--roster 互斥报错；同名产物幂等跳过；NNN 按
全部 confirmed 球时序编号，原片缺失跳号保留）；⑩--roster+--allow-unconfirmed=
仅豁免 confirmed=true 检查的闸门旗标（无 --roster 给此旗标报错退出 1）；
--roster 未 confirmed=true 且无 --allow-unconfirmed 拒收退出 1。
"""

import contextlib
import logging
import os
import sys
import time
from typing import Any

from errors import BasketballPipelineError, SchemaError
from pipe_common import configure_logging, new_run_id, read_json, run_ffmpeg
from roster import Roster, format_key, resolve_scorer, validate_roster

logger = logging.getLogger(__name__)

RAW_DIR: str = "archive/0_raw_videos_test"  # 旧测试素材 2026-08-11 已删除；新场次用 --rawdir 注入
OUT_ROOT: str = "output"
OUT_W: int = 1440
OUT_H: int = 1080
OUT_FPS: int = 50
CRF: int = 20
PRESET: str = "medium"
CLIP_BEFORE_SEC: float = 4.0
CLIP_AFTER_SEC: float = 2.0
# ⑨ --per-goal 产物子目录名（out_dir 下；NNN_<主名>@<t:.1f>s.mp4）
PER_GOAL_DIR_NAME: str = "进球片段"


# 画面滤镜：缩放保持宽高比（force_original_aspect_ratio=decrease），不足处黑边补齐，
# 防 16:9 新素材被压扁；输出尺寸由 --out 按场次素材比例注入
def scale_pad_filter(out_w: int, out_h: int) -> str:
    """生成 scale+pad 滤镜串：等比缩放进 out_w×out_h 画布，黑边补齐居中。

    Args:
        out_w: 输出宽（像素）。
        out_h: 输出高（像素）。

    Returns:
        ffmpeg -vf 用的滤镜串。
    """
    return (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
    )


# goals.json status 合法值（SPEC_2026-07-19 §status 流转定义）；仅 confirmed 进合成
KNOWN_STATUSES: frozenset[str] = frozenset(
    {"candidate", "confirmed", "clipped", "done", "rejected", "removed", "uncertain"}
)
# 慢放段（无现场声）配的静音轨：立体声 48kHz，与原片音轨参数一致，保证 concat 流布局一致
SILENT_AUDIO_SRC: str = "anullsrc=channel_layout=stereo:sample_rate=48000"


def parse_argv() -> tuple[str, str, str, int, int, str, str, bool, bool, str]:
    """解析命令行参数。

    Returns:
        (goals.json 路径, scorer 标签, 原片目录, 输出宽, 输出高, roster.json 路径,
        team 队别, per_goal 旗标, allow_unconfirmed 旗标, name_suffix 产物名后缀；
        scorer/team/name_suffix 空串表示未给，
        roster 空串表示无 roster，原片目录默认 RAW_DIR，尺寸默认 OUT_W×OUT_H，
        两个旗标默认 False)。
    """
    goals: str = ""
    scorer: str = ""
    rawdir: str = RAW_DIR
    out_w: int = OUT_W
    out_h: int = OUT_H
    roster: str = ""
    team: str = ""
    per_goal: bool = False
    allow_unconfirmed: bool = False
    name_suffix: str = ""
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--goals" and i + 1 < len(args):
            goals = args[i + 1]
            i += 2
        elif args[i] == "--scorer" and i + 1 < len(args):
            scorer = args[i + 1]
            i += 2
        elif args[i] == "--rawdir" and i + 1 < len(args):
            rawdir = args[i + 1]
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            w, h = args[i + 1].lower().split("x")
            out_w, out_h = int(w), int(h)
            i += 2
        elif args[i] == "--roster" and i + 1 < len(args):
            roster = args[i + 1]
            i += 2
        elif args[i] == "--team" and i + 1 < len(args):
            team = args[i + 1]
            i += 2
        elif args[i] == "--name-suffix" and i + 1 < len(args):
            name_suffix = args[i + 1]
            i += 2
        elif args[i] == "--per-goal":
            per_goal = True
            i += 1
        elif args[i] == "--allow-unconfirmed":
            allow_unconfirmed = True
            i += 1
        else:
            i += 1
    return (
        goals,
        scorer,
        rawdir,
        out_w,
        out_h,
        roster,
        team,
        per_goal,
        allow_unconfirmed,
        name_suffix,
    )


def load_roster(roster_path: str) -> Roster:
    """读取并校验 roster.json（契约严格走 roster.py，rules.md §0.2）。

    Args:
        roster_path: roster.json 路径。

    Returns:
        校验后的 Roster。

    Raises:
        SchemaError: schema 损坏（缺 players/tag 重复/team 非法/键格式错）。
        OSError: 读取失败。
    """
    data: Any = read_json(roster_path, what="roster.json")
    return validate_roster(data, roster_path)


def require_confirmed(roster: Roster, roster_path: str) -> None:
    """--roster 必须 confirmed=true（用户确认是终裁），未确认拒收（spec 真值表）。

    Args:
        roster: 校验后的 Roster。
        roster_path: 文件路径（仅用于错误信息）。

    Raises:
        BasketballPipelineError: roster.confirmed 非 true（调用方口径 = 退出 1）。
    """
    if not roster.confirmed:
        raise BasketballPipelineError(
            f"{roster_path}: roster 未 confirmed=true，拒收（需用户在确认页完成归属后导出）"
        )


def select_goals(
    goals: list[dict[str, Any]],
    roster: Roster | None,
    scorer: str,
    team: str,
) -> tuple[list[dict[str, Any]], str]:
    """按 spec 组合真值表过滤 confirmed 记录并决定输出文件名主体（8 分支）。

    Args:
        goals: _validate_goals 返回的 confirmed 记录。
        roster: 校验且 confirmed 的 Roster；None 表示未给 --roster。
        scorer: --scorer 值（空串 = 未给）。
        team: --team 值（空串 = 未给）。

    Returns:
        (选中记录, 输出文件名主体)，主体形如 ``个人_全员_进球合集``（①③同名）/
        ``对手队_张三_进球合集`` / ``队伍_对手队_进球集锦``。

    Raises:
        BasketballPipelineError: ⑥--scorer+--team 互斥；⑦无 roster 给 --team；
            ⑧--team 便服；④--scorer 在 roster 内查无此人。
    """
    if scorer and team:
        raise BasketballPipelineError("--scorer 与 --team 互斥（真值表⑥），只能给一个")
    if team and roster is None:
        raise BasketballPipelineError("无 --roster 无法分队（真值表⑦），请先出 roster.json")
    if team == "便服":
        raise BasketballPipelineError("便服不进分队合集（真值表⑧），只进全员/个人合集")

    if team:
        # ⑤：按 team 取 players.tags → 反查 assignments 过滤（roster 必非 None，⑦已拦）
        tags: set[str] = {p.tag for p in roster.players if p.team == team}
        selected = [
            g
            for g in goals
            if roster.assignments.get(format_key(g["file"], float(g["anchor_time"]))) in tags
        ]
        return selected, f"队伍_{team}_进球集锦"

    if roster is not None and scorer:
        # ④：roster 内解析 tag|name，输出名用 {team}_{name or tag}
        player = resolve_scorer(roster, scorer)
        if player is None:
            raise BasketballPipelineError(
                f"roster 内查无此人: {scorer!r}（tag/name 均未命中，真值表④）"
            )
        selected = [
            g
            for g in goals
            if roster.assignments.get(format_key(g["file"], float(g["anchor_time"]))) == player.tag
        ]
        display: str = player.name or player.tag
        return selected, f"{player.team}_{display}_进球合集"

    if roster is not None:
        # ③：全归属球；未归属 WARNING 跳过（SKIP 球允许未归属，不阻塞）
        selected = []
        for g in goals:
            key: str = format_key(g["file"], float(g["anchor_time"]))
            if key in roster.assignments:
                selected.append(g)
            else:
                logger.warning("未归属球跳过: %s", key)
        return selected, "个人_全员_进球合集"

    if scorer:
        # ②：旧兼容路径，goals.scorer 精确匹配；0 命中 WARNING 提示改用 --roster
        selected = [g for g in goals if g.get("scorer") == scorer]
        if not selected:
            logger.warning(
                "--scorer=%s 按 goals.scorer 精确匹配命中 0 条；建议改用 --roster 归属",
                scorer,
            )
        return selected, f"个人_{scorer}_进球合集"

    # ①：全员（stem 同③）
    return list(goals), "个人_全员_进球合集"


# 产物主名尾部类型词（--name-suffix 插入锚点，review02-B4 钉死中段插入）
_STEM_TYPE_WORDS: tuple[str, str] = ("进球集锦", "进球合集")


def apply_name_suffix(out_stem: str, name_suffix: str) -> str:
    """在产物主名尾部类型词（进球集锦/进球合集）之前插入后缀。

    用于 video.py --4k 手动重出（docs/build-4k/spec.md D2）：4K 产物与 1080p
    版并存（如 队伍_X_4K_进球集锦），防止后续 --all 重跑静默覆盖回 1080p。

    Args:
        out_stem: select_goals 返回的产物主名。
        name_suffix: 后缀串（含前导下划线，如 "_4K"）；空串原样返回（缺省=现状）。

    Returns:
        插入后缀后的主名；name_suffix 为空时原样返回。

    Raises:
        BasketballPipelineError: 主名不以已知类型词结尾（命名真值表变更后需同步此处）。
    """
    if not name_suffix:
        return out_stem
    for word in _STEM_TYPE_WORDS:
        if out_stem.endswith(word):
            head: str = out_stem[: -len(word)]
            # 类型词前的下划线分隔符让位给后缀：队伍_X_进球集锦 → 队伍_X_4K_进球集锦
            if head.endswith("_"):
                return head[:-1] + name_suffix + "_" + word
            return head + name_suffix + word
    raise BasketballPipelineError(
        f"产物主名无已知类型词（{_STEM_TYPE_WORDS}），无法插入后缀: {out_stem}"
    )


def _encode_timeout_sec(duration_sec: float) -> int:
    """ffmpeg 转码超时：片段时长 ×3 + 60s 兜底，下限 120s（rules.md §4）。

    Args:
        duration_sec: 输入片段时长（秒）。

    Returns:
        超时秒数。
    """
    return max(120, int(duration_sec * 3) + 60)


def _remove_with_retry(path: str) -> None:
    """删除文件；Windows 瞬时文件锁（杀毒扫描/句柄释放延迟）退避重试（rules.md §4）。

    0.5s → 1s → 2s 退避共 4 次尝试；PermissionError 之外的 OSError 直接抛出
    （不静默）。重试耗尽抛最后一次 PermissionError（显式失败）。

    Args:
        path: 待删文件路径。

    Raises:
        PermissionError: 重试耗尽仍被占用。
        OSError: 其他删除失败（文件不存在等）。
    """
    last: PermissionError | None = None
    for wait in (0.0, 0.5, 1.0, 2.0):
        if wait:
            time.sleep(wait)
        try:
            os.remove(path)
            return
        except PermissionError as exc:
            last = exc
            logger.warning("文件被占用，退避重试删除: %s", path)
    if last is not None:
        raise last  # 四次尝试皆 PermissionError 才走到这


def _validate_goals(data: dict[str, Any], goals_path: str) -> list[dict[str, Any]]:
    """校验 goals.json 结构，返回 status=="confirmed" 的记录列表。

    顶层必须是含 goals 列表的对象；每条记录 status 必须为 str，未知
    status 值记 WARNING（可能拼错）并跳过；confirmed 记录必须有
    file(str)、anchor_time/clip_start/clip_end(数值) 且满足
    clip_start <= anchor_time <= clip_end。

    Args:
        data: read_json 读出的 goals.json 内容。
        goals_path: 文件路径（仅用于错误信息）。

    Returns:
        校验通过的 confirmed 记录（未按 scorer 过滤）。

    Raises:
        SchemaError: 结构损坏（缺字段/类型错/时间区间错），信息含路径与记录索引。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{goals_path}: 顶层必须是对象，实际 {type(data).__name__}")
    goals: Any = data.get("goals")
    if not isinstance(goals, list):
        raise SchemaError(f"{goals_path}: 缺 goals 列表或类型错误，实际 {type(goals).__name__}")
    confirmed: list[dict[str, Any]] = []
    for i, g in enumerate(goals):
        if not isinstance(g, dict):
            raise SchemaError(f"{goals_path}: 第{i}条记录不是对象，实际 {type(g).__name__}")
        status: Any = g.get("status")
        if not isinstance(status, str):
            raise SchemaError(
                f"{goals_path}: 第{i}条 status 必须是 str，实际 {type(status).__name__}"
            )
        if status not in KNOWN_STATUSES:
            logger.warning("%s: 第%d条未知 status=%r（可能拼错），跳过", goals_path, i, status)
            continue
        if status != "confirmed":
            continue
        if not isinstance(g.get("file"), str) or not g["file"]:
            raise SchemaError(f"{goals_path}: 第{i}条(confirmed) file 缺失或不是非空 str")
        for key in ("anchor_time", "clip_start", "clip_end"):
            v: Any = g.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SchemaError(
                    f"{goals_path}: 第{i}条(confirmed) {key} 缺失或不是数值，"
                    f"实际 {type(v).__name__}"
                )
        if not g["clip_start"] <= g["anchor_time"] <= g["clip_end"]:
            raise SchemaError(
                f"{goals_path}: 第{i}条(confirmed) 时间区间错误: "
                f"clip_start={g['clip_start']} anchor_time={g['anchor_time']} "
                f"clip_end={g['clip_end']}（要求 clip_start<=anchor_time<=clip_end）"
            )
        confirmed.append(g)
    return confirmed


def cut_normal(src: str, goal: dict[str, Any], out_path: str, scale_pad: str) -> None:
    """常速切片（50fps 素材）。

    Args:
        src: 原片路径。
        goal: 进球记录（clip_start/clip_end）。
        out_path: 输出片段路径。
        scale_pad: scale_pad_filter 生成的滤镜串（输出尺寸随场次注入）。
    """
    run_ffmpeg(
        [
            "-ss",
            f"{goal['clip_start']:.2f}",
            "-to",
            f"{goal['clip_end']:.2f}",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"{scale_pad},fps={OUT_FPS}",
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-preset",
            PRESET,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ],
        timeout_sec=_encode_timeout_sec(goal["clip_end"] - goal["clip_start"]),
    )


def cut_slowmo(src: str, goal: dict[str, Any], out_path: str, scale_pad: str) -> None:
    """100fps 素材：入网前常速(降50fps)、入网后 2 秒半速慢放，两段拼接。

    part2 滤镜顺序必须是 scale/pad → setpts 拉伸时间轴 → fps 重采样
    （先 fps 会把 100fps 抽掉一半帧，输出只剩有效 25fps）；part1 保留
    现场声、part2 配 lavfi 静音轨，part1/part2/cut_normal 三者流布局
    一致（h264/yuv420p/同尺寸/50fps + aac）才能 -c copy 直接拼接。

    Args:
        src: 原片路径。
        goal: 进球记录（clip_start/anchor_time/clip_end）。
        out_path: 输出片段路径。
        scale_pad: scale_pad_filter 生成的滤镜串（输出尺寸随场次注入）。
    """
    anchor: float = goal["anchor_time"]
    part1: str = out_path.replace(".mp4", "_p1.mp4")
    part2: str = out_path.replace(".mp4", "_p2.mp4")
    run_ffmpeg(
        [
            "-ss",
            f"{goal['clip_start']:.2f}",
            "-to",
            f"{anchor:.2f}",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"{scale_pad},fps={OUT_FPS}",
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-preset",
            PRESET,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            part1,
        ],
        timeout_sec=_encode_timeout_sec(anchor - goal["clip_start"]),
    )
    run_ffmpeg(
        [
            "-ss",
            f"{anchor:.2f}",
            "-to",
            f"{goal['clip_end']:.2f}",
            "-i",
            src,
            "-f",
            "lavfi",
            "-i",
            SILENT_AUDIO_SRC,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            f"{scale_pad},setpts=PTS*2.0,fps={OUT_FPS}",
            "-c:v",
            "libx264",
            "-crf",
            str(CRF),
            "-preset",
            PRESET,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            part2,
        ],
        timeout_sec=_encode_timeout_sec(goal["clip_end"] - anchor),
    )
    list_path: str = out_path.replace(".mp4", "_concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write(f"file '{os.path.basename(part1)}'\nfile '{os.path.basename(part2)}'\n")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    for p in (part1, part2, list_path):
        _remove_with_retry(p)


def _build_per_goal(
    goals: list[dict[str, Any]],
    rawdir: str,
    session: str,
    out_w: int,
    out_h: int,
) -> int:
    """真值表⑨：每 confirmed 球独立出片到 output/<场次>/进球片段/，不 concat。

    命名 NNN_<主名>@<t:.1f>s.mp4（NNN 按全部 confirmed 球时序 001 起；原片缺失
    跳号保留，编号 = 进球序号，稳定不 compact）；同名产物幂等跳过（build 重跑
    不重复转码）；复用 cut_normal/cut_slowmo 直写最终路径（剪辑参数与合集一致）。

    Args:
        goals: 已按 (file, anchor_time) 排序的 confirmed 记录。
        rawdir: 原片目录。
        session: 场次 ID（决定输出目录）。
        out_w: 输出宽（像素）。
        out_h: 输出高（像素）。

    Returns:
        0=全部成功；1=有进球因原片缺失被跳过（能出的照常出，可观测不静默）。
    """
    out_dir: str = os.path.join(OUT_ROOT, session)
    per_goal_dir: str = os.path.join(out_dir, PER_GOAL_DIR_NAME)
    os.makedirs(per_goal_dir, exist_ok=True)
    scale_pad: str = scale_pad_filter(out_w, out_h)
    missing: list[str] = []
    produced: int = 0
    t_start: float = time.time()
    for i, goal in enumerate(goals, 1):
        src: str = os.path.join(rawdir, goal["file"])
        stem: str = os.path.splitext(os.path.basename(goal["file"]))[0]
        name: str = f"{i:03d}_{stem}@{goal['anchor_time']:.1f}s.mp4"
        if not os.path.exists(src):
            logger.error("原片缺失，跳过: %s", goal["file"])
            missing.append(goal["file"])
            continue
        out_path: str = os.path.join(per_goal_dir, name)
        if os.path.exists(out_path):
            logger.info("  片段 %d/%d 已存在，幂等跳过: %s", i, len(goals), name)
            continue
        if goal.get("slowmo"):
            cut_slowmo(src, goal, out_path, scale_pad)
        else:
            cut_normal(src, goal, out_path, scale_pad)
        produced += 1
        logger.info(
            "  片段 %d/%d: %s%s",
            i,
            len(goals),
            name,
            "(慢放)" if goal.get("slowmo") else "",
        )
    logger.info(
        "进球片段完成: %s（新出 %d 条, %.0fs）", per_goal_dir, produced, time.time() - t_start
    )
    if missing:
        logger.error(
            "进球片段已产出，但 %d 条进球因原片缺失被跳过: %s",
            len(missing),
            ", ".join(missing),
        )
        return 1
    return 0


def main() -> int:
    """主入口：按 goals.json 合成个人合集。

    Returns:
        进程退出码：0=全部成功；1=参数/数据/合成失败，或有进球因原片缺失被跳过。
    """
    run_id = new_run_id()
    configure_logging(run_id)
    (
        goals_path,
        scorer,
        rawdir,
        out_w,
        out_h,
        roster_path,
        team,
        per_goal,
        allow_unconfirmed,
        name_suffix,
    ) = parse_argv()
    if not goals_path:
        logger.error("缺少 --goals 参数")
        return 1
    if per_goal and (scorer or team or roster_path):
        logger.error("--per-goal 与 --scorer/--team/--roster 互斥（真值表⑨），只能单独使用")
        return 1
    if per_goal and name_suffix:
        logger.error("--per-goal 逐球出片无合集主名，--name-suffix 不适用")
        return 1
    if allow_unconfirmed and not roster_path:
        logger.error("--allow-unconfirmed 需配 --roster（真值表⑩），单独使用无意义")
        return 1
    try:
        data: dict[str, Any] = read_json(goals_path, what="goals.json")
        goals: list[dict[str, Any]] = _validate_goals(data, goals_path)
        session: str = data.get("session", "unknown")
        roster: Roster | None = None
        if roster_path:
            roster = load_roster(roster_path)
            if allow_unconfirmed:
                # ⑩ 闸门旗标：仅豁免 confirmed=true 检查，其余真值表语义不变
                logger.warning("--allow-unconfirmed：豁免 confirmed=true 检查（%s）", roster_path)
            else:
                require_confirmed(roster, roster_path)
        if per_goal:
            # ⑨：每球独立出片（无过滤旗标，goals = 全部 confirmed）
            goals.sort(key=lambda g: (g["file"], g["anchor_time"]))
            if not goals:
                logger.error("无可合成记录（confirmed 0 条）: %s", goals_path)
                return 1
            return _build_per_goal(goals, rawdir, session, out_w, out_h)
        goals, out_stem = select_goals(goals, roster, scorer, team)
        out_stem = apply_name_suffix(out_stem, name_suffix)
        goals.sort(key=lambda g: (g["file"], g["anchor_time"]))
        if not goals:
            logger.error(
                "无可合成记录 (scorer=%s team=%s roster=%s)",
                scorer or "全部",
                team or "无",
                roster_path or "无",
            )
            return 1

        out_dir: str = os.path.join(OUT_ROOT, session)
        os.makedirs(out_dir, exist_ok=True)
        work_dir: str = os.path.join(out_dir, "_clips_tmp")
        os.makedirs(work_dir, exist_ok=True)

        t_start: float = time.time()
        scale_pad: str = scale_pad_filter(out_w, out_h)
        clips: list[str] = []
        missing: list[str] = []
        for i, goal in enumerate(goals, 1):
            src: str = os.path.join(rawdir, goal["file"])
            if not os.path.exists(src):
                logger.error("原片缺失，跳过: %s", goal["file"])
                missing.append(goal["file"])
                continue
            clip: str = os.path.join(work_dir, f"clip_{i:03d}.mp4")
            if goal.get("slowmo"):
                cut_slowmo(src, goal, clip, scale_pad)
            else:
                cut_normal(src, goal, clip, scale_pad)
            clips.append(clip)
            logger.info(
                "  片段 %d/%d: %s @%.1fs%s",
                i,
                len(goals),
                goal["file"],
                goal["anchor_time"],
                "(慢放)" if goal.get("slowmo") else "",
            )

        if not clips:
            logger.error("全部原片缺失（%d 条），无产出", len(missing))
            return 1

        list_path: str = os.path.join(work_dir, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for clip in clips:
                f.write(f"file '{os.path.basename(clip)}'\n")
        out_path: str = os.path.join(out_dir, f"{out_stem}.mp4")
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )
        for clip in clips:
            _remove_with_retry(clip)
        _remove_with_retry(list_path)
        with contextlib.suppress(OSError):
            os.rmdir(work_dir)
        logger.info(
            "合集完成: %s (%d 片段, %.0fs)",
            out_path,
            len(clips),
            time.time() - t_start,
        )
        if missing:
            logger.error(
                "合集已产出，但 %d 条进球因原片缺失被跳过: %s",
                len(missing),
                ", ".join(missing),
            )
            return 1
        return 0
    except BasketballPipelineError as exc:
        logger.error("管线失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
