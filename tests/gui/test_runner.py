"""gui.runner 任务编排单元测试：进度协议 v1 五条路径（成功/失败/中断/降级/落盘回放）。

全部用 FakeProcess 注入 stdout 行序列，不启动任何真子进程（rules.md §9：慢外部 mock）。
"""

from __future__ import annotations

import json
import pathlib
import queue
import sys
import threading
import time

import pytest

from gui.runner import TaskNotFoundError, TaskRunner, TaskStateError


class FakeProcess:
    """模拟 subprocess.Popen 的最小接口：stdout 行迭代 + wait + terminate。

    Args:
        lines: 逐行喂给 runner 的 stdout 内容（含换行符）。
        returncode: 正常跑完时的退出码。
        hang: True 时行喂完后挂起，直到 terminate 被调用（用于取消路径测试）。
    """

    def __init__(self, lines: list[str], returncode: int = 0, hang: bool = False) -> None:
        self._lines = lines
        self._returncode = returncode
        self._hang = hang
        self.terminated = False
        self.stdout = self._iter_lines()

    def _iter_lines(self) -> FakeProcess._LineIter:
        return FakeProcess._LineIter(self)

    class _LineIter:
        """逐行产出；hang 模式下行耗尽后阻塞至 terminated。"""

        def __init__(self, proc: FakeProcess) -> None:
            self._proc = proc

        def __iter__(self) -> FakeProcess._LineIter:
            return self

        def __next__(self) -> str:
            if self._proc._lines:
                return self._proc._lines.pop(0)
            if self._proc._hang:
                while not self._proc.terminated:
                    time.sleep(0.005)
                raise StopIteration
            raise StopIteration

    def wait(self, timeout: float | None = None) -> int:
        """返回退出码；被 terminate 的进程按 POSIX 惯例返回 -15。"""
        return -15 if self.terminated else self._returncode

    def terminate(self) -> None:
        """置 terminated 标记，hang 中的 stdout 迭代随之结束。"""
        self.terminated = True


def _make_runner(
    tmp_path: pathlib.Path,
    lines: list[str],
    returncode: int = 0,
    hang: bool = False,
) -> tuple[TaskRunner, FakeProcess, dict[str, object]]:
    """构造注入 FakeProcess 的 TaskRunner；返回 (runner, fake, 捕获的 Popen 参数)。"""
    fake = FakeProcess(lines, returncode=returncode, hang=hang)
    captured: dict[str, object] = {}

    def factory(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> FakeProcess:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return fake

    return TaskRunner(work_dir=tmp_path / "work", popen_factory=factory), fake, captured


def _wait_terminal(runner: TaskRunner, task_id: str, timeout: float = 5.0) -> str:
    """轮询至任务进入终态（done/failed/cancelled），超时显式失败。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.get(task_id).status
        if status in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.01)
    pytest.fail(f"任务 {task_id} {timeout}s 内未进入终态，当前 status={runner.get(task_id).status}")
    return ""  # pragma: no cover - pytest.fail 必抛


def _events(runner: TaskRunner, task_id: str) -> list[dict[str, object]]:
    return runner.events(task_id)


def _types(runner: TaskRunner, task_id: str) -> list[str]:
    return [str(e["type"]) for e in _events(runner, task_id)]


def test_success_path_step_progress_and_task_done(tmp_path: pathlib.Path) -> None:
    """成功路径：两行 `执行:` → step_start×2（进度 1/2→2/2）→ step_done → task_done。"""
    # Arrange
    runner, _, captured = _make_runner(
        tmp_path,
        ["执行: python scripts/a.py\n", "普通日志\n", "执行: python scripts/b.py\n"],
    )
    # Act
    record = runner.submit(
        kind="people",
        argv=["scripts/video.py", "people"],
        total_steps=2,
        cwd=tmp_path,
        session="20260801",
    )
    status = _wait_terminal(runner, record.id)
    # Assert
    assert status == "done"
    starts = [e for e in _events(runner, record.id) if e["type"] == "step_start"]
    assert len(starts) == 2
    assert starts[0]["step_index"] == 1
    assert starts[0]["total_steps"] == 2
    assert starts[0]["progress"] == pytest.approx(0.5)
    assert starts[0]["step"] == "python scripts/a.py"
    assert starts[1]["step_index"] == 2
    assert starts[1]["progress"] == pytest.approx(1.0)
    assert starts[1]["step"] == "python scripts/b.py"
    types = _types(runner, record.id)
    assert "step_done" in types
    assert types[-1] == "task_done"
    # 日志透传：所有行（含 执行: 行本身）都有 log 事件
    assert types.count("log") == 3
    # Popen 调用口径：sys.executable + argv，env 注入 PYTHONIOENCODING=utf-8
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert cmd[1:] == ["scripts/video.py", "people"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_failure_path_step_failed_with_returncode_and_tail(tmp_path: pathlib.Path) -> None:
    """失败路径：非零退出 → step_failed（带 returncode 与末尾日志）→ task_failed。"""
    # Arrange
    runner, _, _ = _make_runner(
        tmp_path,
        ["执行: python scripts/x.py\n", "错误输出1\n", "错误输出2\n"],
        returncode=7,
    )
    # Act
    record = runner.submit(
        kind="score", argv=["scripts/video.py", "score"], total_steps=3, cwd=tmp_path
    )
    status = _wait_terminal(runner, record.id)
    # Assert
    assert status == "failed"
    assert record.returncode == 7
    failed = [e for e in _events(runner, record.id) if e["type"] == "step_failed"]
    assert len(failed) == 1
    assert failed[0]["returncode"] == 7
    assert failed[0]["tail"] == ["执行: python scripts/x.py", "错误输出1", "错误输出2"]
    types = _types(runner, record.id)
    assert types[-1] == "task_failed"
    task_failed = _events(runner, record.id)[-1]
    assert task_failed["returncode"] == 7
    assert "task_done" not in types


def test_cancel_path_status_cancelled(tmp_path: pathlib.Path) -> None:
    """中断路径：运行中 cancel → terminate 子进程，状态落 cancelled，无 task_done。"""
    # Arrange
    runner, fake, _ = _make_runner(tmp_path, ["执行: python scripts/x.py\n"], hang=True)
    record = runner.submit(
        kind="build", argv=["scripts/video.py", "build"], total_steps=1, cwd=tmp_path
    )
    deadline = time.monotonic() + 5.0
    while runner.get(record.id).status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runner.get(record.id).status == "running"
    # Act
    runner.cancel(record.id)
    status = _wait_terminal(runner, record.id)
    # Assert
    assert status == "cancelled"
    assert fake.terminated
    assert record.finished_at is not None
    types = _types(runner, record.id)
    assert "task_done" not in types
    assert "task_failed" not in types


def test_cancel_terminal_task_raises(tmp_path: pathlib.Path) -> None:
    """显式失败：对已结束任务 cancel 抛 TaskStateError，不静默。"""
    # Arrange
    runner, _, _ = _make_runner(tmp_path, ["done\n"])
    record = runner.submit(
        kind="photo", argv=["scripts/video.py", "photo"], total_steps=1, cwd=tmp_path
    )
    _wait_terminal(runner, record.id)
    # Act / Assert
    with pytest.raises(TaskStateError):
        runner.cancel(record.id)
    with pytest.raises(TaskNotFoundError):
        runner.cancel("不存在的任务")


def test_degraded_path_only_log_events(tmp_path: pathlib.Path) -> None:
    """降级路径：无标记行/乱序输出 → 只有 log + task_done，无 step_*，进度不报错。"""
    # Arrange
    runner, _, _ = _make_runner(
        tmp_path,
        ["随机输出\n", "进度 50%\n", "执行x 不是步骤标记\n"],
    )
    # Act
    record = runner.submit(
        kind="score", argv=["scripts/video.py", "score"], total_steps=5, cwd=tmp_path
    )
    status = _wait_terminal(runner, record.id)
    # Assert
    assert status == "done"
    types = _types(runner, record.id)
    assert "step_start" not in types
    assert "step_done" not in types
    assert "step_failed" not in types
    assert types.count("log") == 3
    assert types[-1] == "task_done"


def test_persist_and_replay_from_jsonl(tmp_path: pathlib.Path) -> None:
    """落盘回放：事件追加写 work/.gui/tasks/<id>.jsonl，可整段重放且与内存一致。"""
    # Arrange
    runner, _, _ = _make_runner(
        tmp_path,
        ["执行: python scripts/a.py\n", "日志\n"],
    )
    record = runner.submit(
        kind="people", argv=["scripts/video.py", "people"], total_steps=1, cwd=tmp_path
    )
    _wait_terminal(runner, record.id)
    # Act
    jsonl = tmp_path / "work" / ".gui" / "tasks" / f"{record.id}.jsonl"
    # Assert：文件存在、逐行合法 JSON、与内存事件一致
    assert jsonl.is_file()
    disk_events = [
        json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    memory_events = _events(runner, record.id)
    assert disk_events == memory_events
    # 新 runner 实例（模拟断线重连/重启）能从磁盘整段回放
    fresh = TaskRunner(work_dir=tmp_path / "work")
    assert fresh.read_events(record.id) == memory_events
    with pytest.raises(TaskNotFoundError):
        fresh.read_events("不存在的任务")


def test_total_steps_none_degraded_progress(tmp_path: pathlib.Path) -> None:
    """total_steps=None（步数事前不可知）：step 事件进度为 None，任务照常到终态。"""
    # Arrange
    runner, _, _ = _make_runner(tmp_path, ["执行: python scripts/a.py\n"])
    # Act
    record = runner.submit(
        kind="score", argv=["scripts/video.py", "score"], total_steps=None, cwd=tmp_path
    )
    status = _wait_terminal(runner, record.id)
    # Assert
    assert status == "done"
    starts = [e for e in _events(runner, record.id) if e["type"] == "step_start"]
    assert len(starts) == 1
    assert starts[0]["step_index"] == 1
    assert starts[0]["total_steps"] is None
    assert starts[0]["progress"] is None
    with pytest.raises(ValueError):
        runner.submit(kind="x", argv=["s.py"], total_steps=0, cwd=tmp_path)


def test_events_and_subscribe_atomic(tmp_path: pathlib.Path) -> None:
    """events_and_subscribe：原子返回存量快照 + 订阅队列收增量（SSE 无缝衔接用）。"""
    # Arrange：行序列先喂一条再挂起，保证订阅发生在首条事件之后
    runner, _, _ = _make_runner(tmp_path, ["执行: python scripts/a.py\n"], hang=True)
    record = runner.submit(kind="build", argv=["s.py"], total_steps=1, cwd=tmp_path)
    deadline = time.monotonic() + 5.0
    while not _events(runner, record.id) and time.monotonic() < deadline:
        time.sleep(0.01)
    # Act
    backlog, q = runner.events_and_subscribe(record.id)
    assert backlog == _events(runner, record.id)  # 存量快照
    runner.cancel(record.id)
    assert _wait_terminal(runner, record.id) == "cancelled"
    # Assert：增量事件（取消 log）进队列
    deadline = time.monotonic() + 5.0
    got: dict[str, object] | None = None
    while time.monotonic() < deadline:
        try:
            got = q.get(timeout=0.05)
            break
        except queue.Empty:
            continue
    assert got is not None
    assert got["task_id"] == record.id
    runner.unsubscribe(record.id, q)
    with pytest.raises(TaskNotFoundError):
        runner.events_and_subscribe("不存在的任务")


def test_thread_safety_submit_multiple(tmp_path: pathlib.Path) -> None:
    """并发冒烟：多线程同时 submit 各自独立成任务，互不串事件。"""
    # Arrange：每个 submit 一个独立 FakeProcess（factory 每次新建）
    fakes: list[FakeProcess] = []
    lock = threading.Lock()

    def factory(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> FakeProcess:
        fake = FakeProcess(["执行: python one.py\n"])
        with lock:
            fakes.append(fake)
        return fake

    runner = TaskRunner(work_dir=tmp_path / "work", popen_factory=factory)
    # Act
    threads = [
        threading.Thread(
            target=runner.submit,
            kwargs={"kind": "score", "argv": ["s.py"], "total_steps": 1, "cwd": tmp_path},
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Assert
    assert len(fakes) == 4
    for task_id in runner.list_tasks():
        assert _wait_terminal(runner, task_id) == "done"
