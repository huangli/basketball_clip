r"""GUI 任务编排 runner：subprocess 调 scripts 流水线，按进度协议 v1 解析产出事件。

输入：任务提交（kind / argv / total_steps / cwd），子进程 stdout 逐行日志。
输出：内存事件流 + 落盘 ``work/.gui/tasks/<task_id>.jsonl``（追加写，可断线重连回放）。
依赖：仅标准库；进度协议 v1 见 docs/basketball-clip/plan.md（规则集中在本模块单点）。
典型调用：

    runner = TaskRunner(work_dir=Path("work"))
    record = runner.submit("people", ["scripts/video.py", "people"], total_steps=6, cwd=repo)
    ...  # SSE 端点消费 runner.events(record.id)
    runner.cancel(record.id)

进度协议 v1（逐字实现，scripts 侧零改造）：
- 步骤边界：匹配 ``执行: (.+)$``（video.py run_step 既有日志格式）→ step_start；
  子进程退出 0 → step_done；非 0 → step_failed（带 returncode 与末 N 行日志）。
- 进度估算：step_index / total_steps，total_steps 由提交方按命令链预计算。
- 日志透传：所有行原样发 log 事件。
- 事件类型：step_start / step_done / step_failed / log / task_done / task_failed，共六种。
- 降级规则：任何解析异常/格式不识别的行只透传 log，进度保持上次值，绝不报错中断任务。

进度协议 v1.1（在 v1 上增量，v1 六种事件与字段契约不变）：
- 帧进度三规则（锚定 mot_candidates.py 实际日志格式 :669/:675/:690）：
  1. ``=== <fid> (<N>帧) ===`` → 注册该 fid 总帧数分母（每个 fid 都输出，含缓存命中）；
     新 fid 注册时，上一个未完成的 fid 视为完成（被越过即计满）。
  2. ``命中缓存`` 行 → 当前 fid 标记完成（分子计入全部 N 帧），不发事件。
  3. ``第(\d+)/(\d+)帧`` → 更新当前 fid 帧，发 frame_progress。
- frame_progress 事件（第七种）：fid / frame / total_frames / overall_pct；
  overall_pct = (已完成 fid 总帧 + 当前 fid 当前帧) / 已注册分母求和，0-1 小数
  （与 v1 progress 字段同口径），单调不减（只发不小于上次的值）。
- 降级：未注册任何分母时不发 frame_progress；解析异常只透传 log。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

# 进度协议 v1 常量
STEP_START_RE = re.compile(r"执行: (.+)$")  # video.py run_step 既有日志格式
TAIL_LINES = 20  # step_failed/task_failed 携带的末尾日志行数
TASKS_SUBDIR = Path(".gui") / "tasks"  # 相对 work 目录的落盘子目录

# 进度协议 v1.1 常量（帧级进度，逐字锚定 mot_candidates.py :669/:675/:690）
FID_TOTAL_RE = re.compile(r"=== (.+?) \((\d+)帧\) ===")  # :669 注册 fid 总帧数分母
CACHE_HIT_RE = re.compile(r"命中缓存")  # :675 当前 fid 瞬时完成（断点续跑口径）
FRAME_PROGRESS_RE = re.compile(r"第(\d+)/(\d+)帧")  # :690 当前 fid 帧进度

EventType = Literal[
    "step_start", "step_done", "step_failed", "log", "task_done", "task_failed", "frame_progress"
]
TaskStatus = Literal["pending", "running", "done", "failed", "cancelled"]

TERMINAL_STATUSES: tuple[TaskStatus, ...] = ("done", "failed", "cancelled")


class GuiRunnerError(Exception):
    """GUI runner 所有异常的基类。"""


class TaskNotFoundError(GuiRunnerError):
    """task_id 不存在（内存与落盘均无）。"""


class TaskStateError(GuiRunnerError):
    """任务状态不允许当前操作（如对终态任务 cancel）。"""


class ProcLike(Protocol):
    """runner 依赖的子进程最小接口（subprocess.Popen 与测试 FakeProcess 共同满足）。"""

    @property
    def stdout(self) -> Iterable[str]:
        """文本模式 stdout，逐行迭代。"""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """阻塞至进程退出，返回退出码。"""
        ...

    def terminate(self) -> None:
        """请求进程终止（POSIX SIGTERM / Windows TerminateProcess）。"""
        ...


PopenFactory = Callable[[list[str], Path, dict[str, str]], ProcLike]


@dataclass(slots=True)
class TaskRecord:
    """单个任务的内存记录。

    可变（runner 持锁更新）；不变式：status 进入终态后不再变更，events 只增不删。

    Attributes:
        id: uuid4 短形任务 id（同时是落盘文件名主键）。
        kind: 任务种类（score/people/build/photo 等，由提交方定义）。
        session: 场次 ID，无场次概念的任务为 None。
        argv: 脚本参数（不含 sys.executable，runner 组装完整命令）。
        cwd: 子进程工作目录。
        total_steps: 进度协议 v1 预计算总步数（>=1）；None = 步数事前不可知
            （如 score/build 链路），进度事件 progress/total_steps 落 None，
            前端按降级口径转圈 + 日志滚动。
        status: 状态机 pending→running→done/failed/cancelled。
        step_index: 已发出的 step_start 计数（1 起，等于当前步骤序号）。
        returncode: 子进程退出码，未结束为 None；cancel 路径为 -15（POSIX 惯例）。
        events: 已产出事件（扁平 dict，与 jsonl 落盘行一一对应）。
        started_at / finished_at: ISO 8601 UTC 时间串。
    """

    id: str
    kind: str
    session: str | None
    argv: list[str]
    cwd: Path
    total_steps: int | None
    status: TaskStatus = "pending"
    step_index: int = 0
    returncode: int | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None


def _utc_now() -> str:
    """返回 ISO 8601 UTC 时间串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class _FrameProgressState:
    """单任务帧进度跟踪（协议 v1.1 内部状态，不落入事件）。

    Attributes:
        totals: fid → 总帧数（已注册分母，``=== <fid> (<N>帧) ===`` 行注册）。
        done: 已完成 fid 集合（缓存命中，或被下一个 fid 越过）。
        current_fid: 最近一个 ``===`` 行注册的 fid（帧进度行/缓存行的归属）。
        current_frame: current_fid 的最新帧号（缓存命中时计满总帧）。
        last_overall: 已发出的 overall_pct 上限（单调不减裁剪用）。
    """

    totals: dict[str, int] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    current_fid: str | None = None
    current_frame: int = 0
    last_overall: float = 0.0

    def overall_pct(self) -> float | None:
        """计算 overall_pct（0-1，单调不减裁剪）；无分母时返回 None（不发事件）。"""
        denominator = sum(self.totals.values())
        if denominator <= 0 or self.current_fid is None:
            return None
        done_frames = sum(self.totals[fid] for fid in self.done)
        current = 0 if self.current_fid in self.done else self.current_frame
        pct = min(max((done_frames + current) / denominator, self.last_overall), 1.0)
        self.last_overall = pct
        return pct


def _default_popen(cmd: list[str], cwd: Path, env: dict[str, str]) -> ProcLike:
    """真实子进程工厂：stderr 并入 stdout，UTF-8 文本模式逐行读。

    与 video.py run_step 同口径注入 PYTHONIOENCODING（由 submit 在 env 中完成），
    防 Windows 中文乱码；stderr 并入保证 logging（默认输出到 stderr）也被协议解析。
    """
    return subprocess.Popen(  # noqa: S603 命令由本模块内部构造（sys.executable + 提交方 argv）
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


class TaskRunner:
    """subprocess 任务编排器：提交、进度协议 v1 解析、事件落盘、取消。

    线程安全：所有公开方法可跨线程调用（submit 由 HTTP 线程、事件产出由后台读线程）。
    每个任务一个后台守护线程逐行读子进程 stdout。
    """

    def __init__(self, work_dir: Path, popen_factory: PopenFactory | None = None) -> None:
        """初始化 runner。

        Args:
            work_dir: work 目录；事件落盘到 ``<work_dir>/.gui/tasks/<task_id>.jsonl``。
            popen_factory: 子进程工厂（测试注入 fake）；None 用真实 Popen。
        """
        self._work_dir = Path(work_dir)
        self._popen_factory: PopenFactory = popen_factory or _default_popen
        self._tasks: dict[str, TaskRecord] = {}
        self._procs: dict[str, ProcLike] = {}
        self._cancel_requested: set[str] = set()
        self._last_step: dict[str, str] = {}
        self._frame_state: dict[str, _FrameProgressState] = {}
        self._listeners: dict[str, list[queue.Queue[dict[str, object]]]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        argv: list[str],
        total_steps: int | None,
        cwd: Path,
        session: str | None = None,
    ) -> TaskRecord:
        """提交任务：起子进程 + 后台线程按协议 v1 逐行解析产出事件。

        Args:
            kind: 任务种类标签（如 "score"/"people"/"build"/"photo"）。
            argv: 脚本参数列表，runner 前补 sys.executable 组成完整命令。
            total_steps: 进度协议 v1 预计算总步数（如 people = 批次数 × 3 段）；
                None = 步数事前不可知，进度降级为 None（前端转圈 + 日志滚动）。
            cwd: 子进程工作目录（通常为仓库根）。
            session: 场次 ID，可选。

        Returns:
            已转 running 的 TaskRecord。

        Raises:
            ValueError: argv 为空或 total_steps 非 None 且 < 1。
            GuiRunnerError: 子进程启动失败（显式失败不静默）。
        """
        if not argv:
            raise ValueError("argv 不能为空")
        if total_steps is not None and total_steps < 1:
            raise ValueError(f"total_steps 必须 >= 1 或 None，收到 {total_steps}")
        record = TaskRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            session=session,
            argv=list(argv),
            cwd=Path(cwd),
            total_steps=total_steps,
        )
        self._ensure_tasks_dir(record)  # 先建目录再起子进程，避免目录失败留下孤儿进程
        cmd = [sys.executable, *argv]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = self._popen_factory(cmd, record.cwd, env)
        except OSError as e:
            record.status = "failed"
            record.finished_at = _utc_now()
            raise GuiRunnerError(f"子进程启动失败 cmd={cmd!r}: {e}") from e
        with self._lock:
            self._tasks[record.id] = record
            self._procs[record.id] = proc
            record.status = "running"
            record.started_at = _utc_now()
        thread = threading.Thread(
            target=self._run_task,
            args=(record.id, proc),
            name=f"gui-task-{record.id}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "任务已提交 id=%s kind=%s session=%s steps=%s", record.id, kind, session, total_steps
        )
        return record

    def get(self, task_id: str) -> TaskRecord:
        """取任务记录（内存）。

        Raises:
            TaskNotFoundError: task_id 不存在。
        """
        with self._lock:
            record = self._tasks.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"任务不存在: {task_id}")
        return record

    def list_tasks(self) -> list[str]:
        """返回全部任务 id（提交顺序）。"""
        with self._lock:
            return list(self._tasks)

    def events(self, task_id: str) -> list[dict[str, object]]:
        """返回任务已产出事件的内存快照（供 SSE 首次推送）。

        Raises:
            TaskNotFoundError: task_id 不存在。
        """
        record = self.get(task_id)
        with self._lock:
            return list(record.events)

    def events_and_subscribe(
        self, task_id: str
    ) -> tuple[list[dict[str, object]], queue.Queue[dict[str, object]]]:
        """原子地取存量事件快照并挂上增量订阅队列（SSE 无缝衔接用）。

        先快照后订阅会有事件从缝隙丢失，故合并为同一把锁内的一个操作；
        之后产出的每条事件（含调用间隙里产出的）都会 put 进返回的队列。

        Raises:
            TaskNotFoundError: task_id 不存在。
        """
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise TaskNotFoundError(f"任务不存在: {task_id}")
            backlog = list(record.events)
            q: queue.Queue[dict[str, object]] = queue.Queue()
            self._listeners.setdefault(task_id, []).append(q)
        return backlog, q

    def unsubscribe(self, task_id: str, q: queue.Queue[dict[str, object]]) -> None:
        """摘掉订阅队列（SSE 断连时调用；幂等，已摘除不报错）。"""
        with self._lock:
            listeners = self._listeners.get(task_id)
            if listeners is not None and q in listeners:
                listeners.remove(q)

    def read_events(self, task_id: str) -> list[dict[str, object]]:
        """从落盘 jsonl 整段回放事件（断线重连/重启后恢复用）。

        Raises:
            TaskNotFoundError: 落盘文件不存在。
        """
        path = self._task_path(task_id)
        if not path.is_file():
            raise TaskNotFoundError(f"任务事件落盘不存在: {path}")
        out: list[dict[str, object]] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise GuiRunnerError(f"事件落盘损坏 {path}:{lineno}: {e}") from e
        return out

    def cancel(self, task_id: str) -> None:
        """取消任务：terminate 子进程，读线程收尾时状态落 cancelled。

        Raises:
            TaskNotFoundError: task_id 不存在。
            TaskStateError: 任务已在终态（显式失败，不静默跳过）。
        """
        record = self.get(task_id)
        with self._lock:
            if record.status in TERMINAL_STATUSES:
                raise TaskStateError(f"任务 {task_id} 已结束（{record.status}），无法取消")
            self._cancel_requested.add(task_id)
            proc = self._procs[task_id]
        proc.terminate()
        logger.info("任务取消请求已发 id=%s kind=%s", task_id, record.kind)

    # ---- 内部：读线程主循环与协议解析 ----

    def _run_task(self, task_id: str, proc: ProcLike) -> None:
        """后台线程：逐行读 stdout → 协议 v1 事件；进程退出后收尾落终态。

        降级规则（协议 v1）：任何解析异常只透传 log、进度保持上次值，绝不中断任务；
        唯一例外是 stdout 读取本身的 IO 异常——按 rules.md 显式失败落 task_failed。
        """
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        io_error: str | None = None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                tail.append(line)
                self._emit(task_id, "log", {"line": line})
                self._parse_line(task_id, line)
                self._parse_frame_line(task_id, line)
        except (OSError, ValueError) as e:
            io_error = f"stdout 读取异常: {type(e).__name__}: {e}"
            logger.error("任务 %s %s", task_id, io_error)
        returncode = proc.wait()
        self._finalize(task_id, returncode, list(tail), io_error)

    def _parse_line(self, task_id: str, line: str) -> None:
        """协议 v1 步骤边界识别；任何异常/不识别都静默降级为已透传的 log。"""
        try:
            match = STEP_START_RE.search(line)
        except re.error:  # pragma: no cover - 常量正则不会坏，防御性兜底
            return
        if match is None:
            return
        with self._lock:
            record = self._tasks[task_id]
            record.step_index += 1
            step_index = record.step_index
            total = record.total_steps
            self._last_step[task_id] = match.group(1)
        # total=None = 步数事前不可知（降级口径）：progress/total_steps 落 None
        progress = min(step_index / total, 1.0) if total is not None else None
        self._emit(
            task_id,
            "step_start",
            {
                "step": match.group(1),
                "step_index": step_index,
                "total_steps": total,
                "progress": progress,
            },
        )

    def _parse_frame_line(self, task_id: str, line: str) -> None:
        """协议 v1.1 帧进度解析：分母注册 / 缓存命中 / 帧进度三规则。

        与 v1 同降级口径：任何异常/不识别的行只透传 log，不发事件、不报错中断；
        未注册任何分母（无 ``===`` 行）时帧进度行不发 frame_progress（v1 行为不变）。
        """
        try:
            total_match = FID_TOTAL_RE.search(line)
            cache_hit = CACHE_HIT_RE.search(line) is not None
            frame_match = FRAME_PROGRESS_RE.search(line)
        except re.error:  # pragma: no cover - 常量正则不会坏，防御性兜底
            return
        payload: dict[str, object] | None = None
        with self._lock:
            state = self._frame_state.setdefault(task_id, _FrameProgressState())
            if total_match is not None:
                fid = total_match.group(1)
                if state.current_fid is not None and state.current_fid != fid:
                    state.done.add(state.current_fid)  # 被下一个 fid 越过即视为完成
                state.totals[fid] = int(total_match.group(2))
                state.current_fid = fid
                state.current_frame = 0
                return  # 分母注册本身不发事件，等帧行/缓存行
            if state.current_fid is None:
                return
            if cache_hit:
                # 只标记完成（分子计入全部 N 帧）不发事件：后续 fid 分母尚未注册，
                # 此时发事件会算出虚高值并抬升单调下限，污染后续混合口径
                state.done.add(state.current_fid)
                state.current_frame = state.totals[state.current_fid]
                return
            if frame_match is None:
                return
            state.current_frame = int(frame_match.group(1))
            pct = state.overall_pct()
            if pct is not None:
                payload = {
                    "fid": state.current_fid,
                    "frame": state.current_frame,
                    "total_frames": state.totals[state.current_fid],
                    "overall_pct": pct,
                }
        if payload is not None:
            self._emit(task_id, "frame_progress", payload)

    def _finalize(
        self,
        task_id: str,
        returncode: int,
        tail: list[str],
        io_error: str | None,
    ) -> None:
        """进程退出后收尾：cancel_requested → cancelled；0 → done；非 0/IO 异常 → failed。

        终态事件全部发完才落 ``record.status``：外部以 status 轮询终态时（含测试），
        看到终态即保证终态事件已入内存队列与落盘，无"状态先行事件后到"的竞态窗口。
        """
        with self._lock:
            record = self._tasks[task_id]
            if record.status in TERMINAL_STATUSES:  # 防御：收尾只执行一次
                return
            cancelled = task_id in self._cancel_requested
            record.returncode = returncode
            record.finished_at = _utc_now()
            if cancelled:
                final_status: TaskStatus = "cancelled"
            elif returncode == 0 and io_error is None:
                final_status = "done"
            else:
                final_status = "failed"
            last_step = self._last_step.get(task_id)
            step_index = record.step_index
            total = record.total_steps
        progress = min(step_index / total, 1.0) if total is not None else None
        if cancelled:
            self._emit(task_id, "log", {"line": "任务已取消（cancel）"})
        elif final_status == "done":
            if last_step is not None:
                self._emit(
                    task_id,
                    "step_done",
                    {
                        "step": last_step,
                        "step_index": step_index,
                        "total_steps": total,
                        "progress": progress,
                    },
                )
            self._emit(task_id, "task_done", {"returncode": returncode})
        else:
            payload: dict[str, object] = {"returncode": returncode, "tail": tail}
            if io_error is not None:
                payload["error"] = io_error
            if last_step is not None:
                self._emit(
                    task_id,
                    "step_failed",
                    {"step": last_step, "step_index": step_index, "total_steps": total, **payload},
                )
            self._emit(task_id, "task_failed", payload)
            logger.error(
                "任务失败 id=%s returncode=%s io_error=%s tail=%s",
                task_id,
                returncode,
                io_error,
                tail,
            )
        with self._lock:
            record.status = final_status

    # ---- 内部：事件产出（内存 + 落盘） ----

    def _emit(self, task_id: str, event_type: EventType, payload: dict[str, object]) -> None:
        """产出一条事件：进内存队列 + 追加写 jsonl（flush 保证断线可回放）+ 推订阅队列。"""
        event: dict[str, object] = {
            "ts": _utc_now(),
            "task_id": task_id,
            "type": event_type,
            **payload,
        }
        with self._lock:
            record = self._tasks[task_id]
            record.events.append(event)
            listeners = list(self._listeners.get(task_id, ()))
            path = self._task_path(task_id)
        for q in listeners:  # put_nowait 永不阻塞：慢消费者不拖垮读线程
            q.put_nowait(event)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as e:
            # 落盘失败不中断任务（协议降级口径），但必须可观测
            logger.error("任务 %s 事件落盘失败 %s: %s", task_id, path, e)

    def _task_path(self, task_id: str) -> Path:
        """任务事件落盘路径。"""
        return self._work_dir / TASKS_SUBDIR / f"{task_id}.jsonl"

    def _ensure_tasks_dir(self, record: TaskRecord) -> None:
        """确保落盘目录存在；失败显式抛（提交即失败好过跑到一半丢事件）。"""
        try:
            self._task_path(record.id).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            record.status = "failed"
            record.finished_at = _utc_now()
            raise GuiRunnerError(
                f"事件落盘目录创建失败: {self._task_path(record.id).parent}: {e}"
            ) from e
