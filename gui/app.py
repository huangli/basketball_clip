"""GUI FastAPI 入口：REST + SSE + 静态托管，浏览器驱动整条流水线。

输入：HTTP 请求（选目录 / 建场次 / 启动任务 / 查状态 / 取消 / 事件流 / 确认页托管）。
输出：JSON 响应（错误一律 ``{"error": "中文消息"}`` + 合适状态码，后端 log 详细
    上下文）与 text/event-stream 事件流。
依赖：fastapi / pydantic；任务编排全部委托 gui.runner.TaskRunner（本模块不解析
    进度协议）；GUI 不 import scripts 内部逻辑（包边界，见 gui/__init__.py），
    批次/产物命名契约在此本地实现（与 scripts/video.py 双轨命名保持一致）。
典型调用：

    app = create_app()  # work/output 取仓库根默认
    uvicorn.run(app, host="127.0.0.1", port=8471)

架构定案（task-C2 brief，controller 裁决）：
- 每个动作 = 一条 ``python scripts/video.py <子命令> ...`` subprocess 交给 runner；
- total_steps：score/build/photo 步数事前不可知传 None（前端降级转圈），
  people 指定单批可预估 3；
- cancelled 任务无终态事件：SSE 以轮询任务状态兜底收尾。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gui.diagnostics import build_diagnostics_zip
from gui.runner import (
    TERMINAL_STATUSES,
    GuiRunnerError,
    TaskNotFoundError,
    TaskRecord,
    TaskRunner,
    TaskStateError,
)

logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR: Path = REPO_ROOT / "work"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "output"
DEFAULT_STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

# 场次 ID / 路径段校验：中英文、数字、下划线、连字符；点号一律不允许（从源头掐死 ..）
SESSION_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_一-鿿-]+")
# /pages 路径段：文件名带点号与括号/空格，但禁反斜杠与 ..
PAGE_SEGMENT_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_一-鿿.()（）+@~ -]+")
# video.py 子进程 argv 相对路径基准（cwd=REPO_ROOT，与 runner 测试同口径）
VIDEO_CLI_ARGV: str = "scripts/video.py"
# people 指定单批时的预估步数（裁图/传播/聚类…确认页一条链，架构定案按 3 估）
PEOPLE_SINGLE_BATCH_STEPS: int = 3
# SSE 增量队列轮询间隔（秒）：超时醒一次查任务终态（cancelled 无终态事件，靠此兜底）
SSE_POLL_SECONDS: float = 0.5
# 批次 goals 双轨命名（与 scripts/video.py GOALS_BATCH_RE 同契约，本地实现不 import）
GOALS_BATCH_RE: re.Pattern[str] = re.compile(r"^goals_batch(\d+)\.json$")
# 场次目录识别标记（work/ 下还有 detect/frames/.gui 等共享目录，靠标记物区分）
SESSION_MARKERS: tuple[str, ...] = (
    "video_cli.json",
    "session_facts.json",
    "team_config.json",
    "roster.json",
    "auto_roster.json",
)
# team_config.json 契约（B-2 schema v1；写侧本地实现，读侧容错归 scripts/team_config.py）
TEAM_CONFIG_VERSION: int = 1
TEAM_CONFIG_NAME: str = "team_config.json"


# ---- 请求体模型（pydantic 边界校验第一层） ----


class ScanRequest(BaseModel):
    """POST /api/sessions/scan 请求体。"""

    srcdir: str


class ScoreRequest(BaseModel):
    """POST /api/sessions/{session}/score 请求体。"""

    srcdir: str
    batch_size: int | None = None


class PeopleRequest(BaseModel):
    """POST /api/sessions/{session}/people 请求体。"""

    batch: int | None = None


class BuildRequest(BaseModel):
    """POST /api/sessions/{session}/build 请求体（all/scorer/team 互斥在路由层校验）。"""

    all: bool = False
    scorer: str = ""
    team: str = ""


class PhotoRequest(BaseModel):
    """POST /api/sessions/{session}/photo 请求体。"""

    apply: bool = False


class TeamConfigRequest(BaseModel):
    """POST /api/sessions/{session}/team-config 请求体。"""

    team_name: str
    opponent: str | None = None


# ---- 边界校验与错误约定 ----


def _fail(status_code: int, message: str, *, log_context: str = "") -> None:
    """统一显式失败：后端记详细上下文，客户端得友好中文 {"error": ...}。"""
    if log_context:
        logger.warning("请求拒绝(%d): %s ｜ %s", status_code, message, log_context)
    raise HTTPException(status_code=status_code, detail=message)


def _valid_session(session: str) -> str:
    """校验场次 ID 字符集（禁点号/斜杠/空串，从源头防路径穿越）。"""
    if not SESSION_RE.fullmatch(session):
        _fail(400, f"场次 ID 非法: {session!r}（仅允许中英文/数字/下划线/连字符）")
    return session


def _valid_srcdir(srcdir: str) -> Path:
    """校验素材目录存在且是目录；相对路径按仓库根解析后转绝对（子进程 cwd=仓库根）。"""
    src = Path(srcdir)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    if not src.exists():
        _fail(400, f"素材目录不存在: {src}", log_context=f"srcdir 入参 {srcdir!r}")
    if not src.is_dir():
        _fail(400, f"素材路径不是目录: {src}", log_context=f"srcdir 入参 {srcdir!r}")
    return src


def _session_dir_or_404(work_dir: Path, session: str) -> Path:
    """定位 work/<场次>/；不存在 404（不猜路径，与 video.py 口径一致）。"""
    session_dir = work_dir / session
    if not session_dir.is_dir():
        _fail(404, f"场次不存在: {session}（先跑检测）", log_context=f"work_dir={work_dir}")
    return session_dir


# ---- 场次阶段探测（依据产物存在性，GUI 只读不写） ----


def _roster_confirmed(session_dir: Path) -> bool | None:
    """读 roster.json 的 confirmed 字段；缺失 None，损坏记 WARNING 按未确认处理。"""
    path = session_dir / "roster.json"
    if not path.is_file():
        return None
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("roster 读取失败，按未确认处理: %s (%s)", path, e)
        return False
    confirmed: Any = data.get("confirmed") if isinstance(data, dict) else None
    return confirmed is True


def _session_stage(session_dir: Path, output_dir: Path) -> str:
    """场次阶段：candidates → goals → roster → roster_confirmed → output（取最远）。

    output 判定 = output/<场次>/ 存在且非空；roster 损坏按未确认档（可观测 WARNING）。
    """
    out_dir = output_dir / session_dir.name
    try:
        if out_dir.is_dir() and any(out_dir.iterdir()):
            return "output"
    except OSError as e:
        logger.warning("output 目录扫描失败: %s (%s)", out_dir, e)
    confirmed = _roster_confirmed(session_dir)
    if confirmed is True:
        return "roster_confirmed"
    if confirmed is False:
        return "roster"
    if any(session_dir.glob("goals*.json")):
        return "goals"
    return "candidates"


def _is_session_dir(path: Path) -> bool:
    """判定 work/ 子目录是否场次目录：名字合法 + 有任一标记物/goals/candidates。"""
    if not path.is_dir() or SESSION_RE.fullmatch(path.name) is None:
        return False
    if any((path / marker).is_file() for marker in SESSION_MARKERS):
        return True
    return any(path.glob("goals*.json")) or any(path.glob("candidates*.json"))


def _batch_paths(session_dir: Path, goals_name: str, batch: int) -> dict[str, Any]:
    """由 goals 文件名推导批次配套路径（双轨命名契约，同 video.py _batch_from_goals）。"""
    if goals_name == "goals.json":  # 旧布局批次 1
        candidates = session_dir / "candidates.json"
        review_dir = session_dir / "review"
        scorers_dir = session_dir / "scorers"
    else:
        candidates = session_dir / f"candidates_batch{batch}.json"
        review_dir = session_dir / f"review_batch{batch}"
        scorers_dir = session_dir / f"scorers_b{batch}"
    label_page = review_dir / "label.html"
    scorer_page = scorers_dir / "scorer.html"
    session = session_dir.name
    return {
        "batch": batch,
        "goals": (session_dir / goals_name).is_file(),
        "candidates": candidates.is_file(),
        "label_page": (
            f"/pages/{session}/{review_dir.name}/label.html" if label_page.is_file() else None
        ),
        "scorer_page": (
            f"/pages/{session}/{scorers_dir.name}/scorer.html" if scorer_page.is_file() else None
        ),
    }


def _discover_batches(session_dir: Path) -> list[dict[str, Any]]:
    """扫 goals 文件定位批次（双轨），按批次号升序；无法识别的名记 WARNING 跳过。"""
    batches: list[dict[str, Any]] = []
    for goals_path in sorted(session_dir.glob("goals*.json")):
        name = goals_path.name
        if name == "goals.json":
            batches.append(_batch_paths(session_dir, name, 1))
            continue
        m = GOALS_BATCH_RE.match(name)
        if m is None or int(m.group(1)) < 1:
            logger.warning("无法识别的 goals 文件，跳过: %s", name)
            continue
        batches.append(_batch_paths(session_dir, name, int(m.group(1))))
    batches.sort(key=lambda b: int(b["batch"]))
    return batches


def _list_outputs(output_dir: Path, session: str) -> list[str]:
    """列 output/<场次>/ 顶层产物文件名（升序）；目录缺失返回空。"""
    out_dir = output_dir / session
    if not out_dir.is_dir():
        return []
    try:
        return sorted(p.name for p in out_dir.iterdir() if p.is_file())
    except OSError as e:
        logger.warning("output 目录列举失败: %s (%s)", out_dir, e)
        return []


# ---- 任务提交与序列化 ----


def _task_json(record: TaskRecord) -> dict[str, Any]:
    """TaskRecord → API JSON（events 体积大不下发，走 SSE/回放端点）。"""
    total = record.total_steps
    progress = min(record.step_index / total, 1.0) if total is not None else None
    return {
        "task_id": record.id,
        "kind": record.kind,
        "session": record.session,
        "status": record.status,
        "step_index": record.step_index,
        "total_steps": total,
        "progress": progress,
        "argv": record.argv,
        "returncode": record.returncode,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


def _submit_video_task(
    runner: TaskRunner,
    kind: str,
    session: str,
    argv_tail: list[str],
    total_steps: int | None,
) -> dict[str, Any]:
    """提交一条 video.py 子命令任务（统一入口：argv 拼装 + 失败转 500）。

    argv_tail 为完整子命令参数（含子命令名本身，如 ["score", srcdir, "--session", s]），
    runner 前补 sys.executable 组成完整命令。
    """
    argv = [VIDEO_CLI_ARGV, *argv_tail]
    try:
        record = runner.submit(kind, argv, total_steps=total_steps, cwd=REPO_ROOT, session=session)
    except (GuiRunnerError, ValueError) as e:
        logger.error(
            "任务启动失败 kind=%s session=%s argv=%s: %s", kind, session, argv, e, exc_info=True
        )
        _fail(500, f"任务启动失败: {e}")
    return _task_json(record)


def _sse_frame(event: dict[str, Any]) -> str:
    """单条事件 → SSE data 帧。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _event_stream(runner: TaskRunner, task_id: str) -> Iterator[str]:
    """SSE 事件流：先回放存量（内存任务原子快照，重启后回退 jsonl 落盘），再推增量。

    收尾口径：task_done/task_failed 事件后即结束；cancelled 无终态事件
    （runner M1），靠轮询任务状态兜底结束，前端另有 /api/tasks/{id} 轮询。
    """
    in_memory = True
    try:
        backlog, q = runner.events_and_subscribe(task_id)
    except TaskNotFoundError:
        in_memory = False
        q = None
        backlog = runner.read_events(task_id)  # 重启口径：磁盘整段回放
    for event in backlog:
        yield _sse_frame(event)
    if not in_memory or q is None:
        return
    try:
        while True:
            try:
                event = q.get(timeout=SSE_POLL_SECONDS)
            except queue.Empty:
                if runner.get(task_id).status in TERMINAL_STATUSES and q.empty():
                    return
                continue
            yield _sse_frame(event)
            if event["type"] in ("task_done", "task_failed"):
                return
    finally:
        runner.unsubscribe(task_id, q)


# ---- /pages 路径穿越防护 ----


def _resolve_page_path(work_dir: Path, rel: str) -> Path:
    """/pages 相对路径 → work 内文件；逐段字符集校验 + 解析后收容检查双保险。"""
    if "\\" in rel:
        _fail(400, "路径非法: 不允许反斜杠", log_context=f"rel={rel!r}")
    parts = rel.split("/")
    for seg in parts:
        if not seg or seg in (".", "..") or ".." in seg or PAGE_SEGMENT_RE.fullmatch(seg) is None:
            _fail(400, f"路径非法: {rel!r}", log_context=f"问题段 {seg!r}")
    root = work_dir.resolve()
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root):
        _fail(400, f"路径非法: {rel!r}", log_context="解析后越出 work 根")
    if not candidate.is_file():
        _fail(404, f"页面不存在: {rel}", log_context=f"resolved={candidate}")
    return candidate


def create_app(
    work_dir: Path | None = None,
    output_dir: Path | None = None,
    runner: TaskRunner | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """构建 FastAPI 应用（工厂式，测试注入隔离 work/output 与打桩 runner）。

    Args:
        work_dir: work 根（场次目录/事件落盘基准）；缺省仓库根 work/。
        output_dir: output 根（产物清单与阶段判定用）；缺省仓库根 output/。
        runner: 任务编排器；None 用真实子进程工厂新建。
        static_dir: 前端静态目录（GET / 与 /static）；缺省 gui/static/。

    Returns:
        就绪的 FastAPI 实例。
    """
    work = Path(work_dir) if work_dir is not None else DEFAULT_WORK_DIR
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    static = Path(static_dir) if static_dir is not None else DEFAULT_STATIC_DIR
    task_runner = runner if runner is not None else TaskRunner(work_dir=work)

    app = FastAPI(title="basketball-clip GUI")
    app.state.runner = task_runner
    app.state.work_dir = work
    app.state.output_dir = output

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        """统一错误外形：{"error": "中文消息"}。"""
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """pydantic 校验失败 → 400 友好消息（细节只进后端日志，不泄露给客户端）。"""
        logger.warning("请求体校验失败: %s", exc.errors())
        return JSONResponse(status_code=400, content={"error": "请求参数不合法，请检查输入"})

    # ---- 健康与扫描 ----

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        """健康检查。"""
        return {"ok": True}

    # ---- 诊断日志导出（D-1：一键打 zip 供用户附到 GitHub issue） ----

    # 前端 HEAD 探测激活按钮；FastAPI GET 路由不自动响应 HEAD（starlette 1.6 实证
    # HEAD→405 Allow:GET），必须显式并列 HEAD
    @app.api_route("/api/diagnostics", methods=["GET", "HEAD"])
    def download_diagnostics() -> Response:
        """导出诊断日志 zip：版本/环境/任务事件流/配置快照（密钥零泄漏）。

        前端 HEAD 探测激活按钮（GET/HEAD 并列注册），
        下载走 ``window.location.href``（附件 Content-Disposition）。
        """
        try:
            payload = build_diagnostics_zip(work_dir=work, repo_root=REPO_ROOT)
        except OSError as e:
            logger.error("诊断日志生成失败: %s", e, exc_info=True)
            _fail(500, "诊断日志生成失败，请查看后端日志")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"basketball-clip-diagnostics-{stamp}.zip"
        logger.info("诊断日志已导出: %s（%d 字节）", filename, len(payload))
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/sessions/scan")
    def scan_sessions(body: ScanRequest) -> dict[str, Any]:
        """递归扫素材目录 .mp4；推测场次 ID = 目录名（score 缺省口径同源）。"""
        src = _valid_srcdir(body.srcdir)
        try:
            files = sorted(
                p.relative_to(src).as_posix()
                for p in src.rglob("*")
                if p.is_file() and p.suffix.lower() == ".mp4"
            )
        except OSError as e:
            logger.error("素材目录扫描失败: %s (%s)", src, e, exc_info=True)
            _fail(500, f"素材目录扫描失败: {src}")
        return {"files": files, "count": len(files), "session": src.name}

    # ---- 场次清单与状态 ----

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        """扫 work/ 列场次清单及各自阶段（产物存在性判定）。"""
        sessions: list[dict[str, Any]] = []
        if work.is_dir():
            try:
                children = sorted(work.iterdir())
            except OSError as e:
                logger.error("work 目录扫描失败: %s (%s)", work, e, exc_info=True)
                _fail(500, f"work 目录扫描失败: {work}")
            for child in children:
                if _is_session_dir(child):
                    sessions.append({"session": child.name, "stage": _session_stage(child, output)})
        return {"sessions": sessions}

    @app.get("/api/sessions/{session}/status")
    def session_status(session: str) -> dict[str, Any]:
        """单场次阶段详情：批次产物 / 确认页路径 / roster / 输出清单。"""
        session_dir = _session_dir_or_404(work, _valid_session(session))
        confirmed = _roster_confirmed(session_dir)
        photo_page = session_dir / "photos" / "photo_page.html"
        team_config_path = session_dir / TEAM_CONFIG_NAME
        team_config: dict[str, Any] | None = None
        if team_config_path.is_file():
            try:
                team_config = json.loads(team_config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                logger.warning("team_config 读取失败: %s (%s)", team_config_path, e)
        return {
            "session": session,
            "stage": _session_stage(session_dir, output),
            "batches": _discover_batches(session_dir),
            "roster": {"exists": confirmed is not None, "confirmed": confirmed},
            "photo_page": (
                f"/pages/{session}/photos/photo_page.html" if photo_page.is_file() else None
            ),
            "team_config": team_config,
            "outputs": _list_outputs(output, session),
        }

    # ---- team-config（B-2 契约 schema v1 写侧） ----

    @app.post("/api/sessions/{session}/team-config")
    def write_team_config(session: str, body: TeamConfigRequest) -> dict[str, Any]:
        """写 work/<场次>/team_config.json（原子写：tmp + os.replace）。"""
        session_dir = _session_dir_or_404(work, _valid_session(session))
        team_name = body.team_name.strip()
        if not team_name:
            _fail(400, "队名不能为空")
        payload: dict[str, Any] = {"version": TEAM_CONFIG_VERSION, "team_name": team_name}
        opponent = body.opponent.strip() if body.opponent is not None else ""
        if opponent:
            payload["opponent"] = opponent
        target = session_dir / TEAM_CONFIG_NAME
        tmp = session_dir / f"{TEAM_CONFIG_NAME}.tmp"
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as e:
            logger.error("team_config 写入失败: %s (%s)", target, e, exc_info=True)
            _fail(500, f"配置写入失败: {target}")
        logger.info("team_config 已写: %s team_name=%r", target, team_name)
        return {"ok": True, "team_config": payload}

    # ---- 任务提交（score/people/build/photo = 一条 video.py 子命令） ----

    @app.post("/api/sessions/{session}/score")
    def submit_score(session: str, body: ScoreRequest) -> dict[str, Any]:
        """提交检测任务：video.py score <srcdir> --session <场次> [--batch-size K]。"""
        session = _valid_session(session)
        src = _valid_srcdir(body.srcdir)
        tail = ["score", str(src), "--session", session]
        if body.batch_size is not None:
            if body.batch_size < 1:
                _fail(400, f"batch_size 必须 >= 1，收到 {body.batch_size}")
            tail.extend(["--batch-size", str(body.batch_size)])
        return _submit_video_task(task_runner, "score", session, tail, total_steps=None)

    @app.post("/api/sessions/{session}/people")
    def submit_people(session: str, body: PeopleRequest) -> dict[str, Any]:
        """提交认人任务；指定单批可预估 3 步，否则步数不可知传 None。"""
        _session_dir_or_404(work, _valid_session(session))  # 存在性校验即目的（不猜路径）
        tail = ["people", "--session", session]
        total_steps: int | None = None
        if body.batch is not None:
            if body.batch < 1:
                _fail(400, f"batch 必须 >= 1，收到 {body.batch}")
            tail.extend(["--batch", str(body.batch)])
            total_steps = PEOPLE_SINGLE_BATCH_STEPS
        return _submit_video_task(task_runner, "people", session, tail, total_steps=total_steps)

    @app.post("/api/sessions/{session}/build")
    def submit_build(session: str, body: BuildRequest) -> dict[str, Any]:
        """提交合成任务；all/scorer/team 互斥（video.py argparse 同口径前置到边界）。"""
        _session_dir_or_404(work, _valid_session(session))
        options = (("all", body.all), ("scorer", body.scorer), ("team", body.team))
        chosen = [name for name, value in options if value]
        if len(chosen) > 1:
            _fail(400, f"build 过滤项互斥（--all/--scorer/--team 只能给一个）: {chosen}")
        tail = ["build", "--session", session]
        if body.all:
            tail.append("--all")
        elif body.scorer:
            tail.extend(["--scorer", body.scorer])
        elif body.team:
            tail.extend(["--team", body.team])
        return _submit_video_task(task_runner, "build", session, tail, total_steps=None)

    @app.post("/api/sessions/{session}/photo")
    def submit_photo(session: str, body: PhotoRequest) -> dict[str, Any]:
        """提交照片任务；apply=true 带 --apply 落盘模式。"""
        _session_dir_or_404(work, _valid_session(session))
        tail = ["photo", "--session", session]
        if body.apply:
            tail.append("--apply")
        return _submit_video_task(task_runner, "photo", session, tail, total_steps=None)

    # ---- 任务查询 / 取消 / 事件流 ----

    @app.get("/api/tasks")
    def list_tasks() -> dict[str, Any]:
        """任务清单（提交顺序；仅内存任务，历史靠 events 端点磁盘回放）。"""
        return {"tasks": [_task_json(task_runner.get(tid)) for tid in task_runner.list_tasks()]}

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str) -> dict[str, Any]:
        """任务详情（前端轮询兜底：cancelled 无终态事件）。"""
        try:
            return _task_json(task_runner.get(task_id))
        except TaskNotFoundError:
            _fail(404, f"任务不存在: {task_id}")

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        """取消运行中任务；终态 409、不存在 404（显式失败不静默）。"""
        try:
            task_runner.cancel(task_id)
        except TaskNotFoundError:
            _fail(404, f"任务不存在: {task_id}")
        except TaskStateError as e:
            _fail(409, str(e))
        return {"ok": True}

    @app.get("/api/tasks/{task_id}/events")
    def task_events(task_id: str) -> StreamingResponse:
        """SSE 事件流：存量回放 + 增量推送；重启后内存无任务回退磁盘回放。"""
        try:
            task_runner.get(task_id)  # 存在性预检，流内再走原子快照+订阅
        except TaskNotFoundError:
            try:
                task_runner.read_events(task_id)
            except TaskNotFoundError:
                _fail(404, f"任务不存在: {task_id}")
        return StreamingResponse(
            _event_stream(task_runner, task_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- 静态托管 ----

    @app.get("/pages/{file_path:path}")
    def serve_page(file_path: str) -> FileResponse:
        """确认页只读托管（标注/认人/照片页是生成器产物，GUI 只托管不重新生成）。"""
        return FileResponse(_resolve_page_path(work, file_path))

    @app.get("/")
    def index() -> FileResponse:
        """向导首页（C-3 做内容，当前为占位页）。"""
        index_path = static / "index.html"
        if not index_path.is_file():
            _fail(500, f"前端占位页缺失: {index_path}")
        return FileResponse(index_path)

    # 前端静态资源（app.js/style.css 等，C-3 填充）；目录缺失不挂载（占位页仍可出）
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

    return app
