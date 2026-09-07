"""gui.app FastAPI 路由测试：REST 校验 / 任务生命周期 / SSE 回放 / 路径穿越 / team-config。

runner 一律注入 FakeProcess（不起真子进程，rules.md §9）；work/output 目录用
tmp_path 隔离，不触碰真实工作区。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest
from fastapi.testclient import TestClient

from gui.app import create_app
from gui.runner import TERMINAL_STATUSES, TaskRunner


class FakeProcess:
    """模拟 subprocess.Popen 的最小接口（与 tests/gui/test_runner.py 同款）。"""

    def __init__(self, lines: list[str], returncode: int = 0, hang: bool = False) -> None:
        self._lines = lines
        self._returncode = returncode
        self._hang = hang
        self.terminated = False
        self.stdout = self._iter_lines()

    def _iter_lines(self) -> FakeProcess._LineIter:
        return FakeProcess._LineIter(self)

    class _LineIter:
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
        return -15 if self.terminated else self._returncode

    def terminate(self) -> None:
        self.terminated = True


def _make_client(
    tmp_path: pathlib.Path,
    lines: list[str] | None = None,
    returncode: int = 0,
    hang: bool = False,
) -> tuple[TestClient, TaskRunner, list[dict[str, object]]]:
    """构造注入 FakeProcess 的 app + TestClient；返回 (client, runner, 捕获的 Popen 参数)。"""
    spec_lines = list(lines) if lines is not None else ["执行: python scripts/x.py\n"]
    captured: list[dict[str, object]] = []

    def factory(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> FakeProcess:
        captured.append({"cmd": cmd, "cwd": cwd})
        return FakeProcess(list(spec_lines), returncode=returncode, hang=hang)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = TaskRunner(work_dir=work_dir, popen_factory=factory)
    app = create_app(work_dir=work_dir, output_dir=tmp_path / "output", runner=runner)
    return TestClient(app), runner, captured


def _wait_terminal(runner: TaskRunner, task_id: str, timeout: float = 5.0) -> str:
    """轮询至任务进入终态且终态事件已落（task_done/task_failed 在 status 翻转后发出）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.get(task_id).status
        if status in TERMINAL_STATUSES:
            events = runner.events(task_id)
            if status == "cancelled" or (
                events and events[-1]["type"] in ("task_done", "task_failed")
            ):
                return status
        time.sleep(0.01)
    pytest.fail(f"任务 {task_id} {timeout}s 内未进入终态")
    return ""  # pragma: no cover - pytest.fail 必抛


def _make_srcdir(tmp_path: pathlib.Path, name: str = "20260801_测试队") -> pathlib.Path:
    """造素材目录：两个 .mp4（含子目录一个、大写后缀一个）+ 一个非 mp4 干扰项。"""
    src = tmp_path / name
    (src / "sub").mkdir(parents=True)
    (src / "DJI_001.mp4").write_bytes(b"")
    (src / "sub" / "DJI_002.MP4").write_bytes(b"")
    (src / "notes.txt").write_text("非视频", encoding="utf-8")
    return src


# ---- 健康检查 ----


def test_health(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---- 素材扫描 ----


def test_scan_ok(tmp_path: pathlib.Path) -> None:
    src = _make_srcdir(tmp_path)
    client, _, _ = _make_client(tmp_path)
    resp = client.post("/api/sessions/scan", json={"srcdir": str(src)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["files"] == ["DJI_001.mp4", "sub/DJI_002.MP4"]
    assert data["session"] == src.name  # 推测场次 ID = 目录名


def test_scan_bad_dir_400(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    resp = client.post("/api/sessions/scan", json={"srcdir": str(tmp_path / "不存在")})
    assert resp.status_code == 400
    assert "error" in resp.json()
    # 文件不是目录同样 400
    f = tmp_path / "a.mp4"
    f.write_bytes(b"")
    resp2 = client.post("/api/sessions/scan", json={"srcdir": str(f)})
    assert resp2.status_code == 400
    assert "error" in resp2.json()


# ---- score 任务提交 ----


def test_score_submit_argv(tmp_path: pathlib.Path) -> None:
    src = _make_srcdir(tmp_path, name="20260801")
    client, runner, captured = _make_client(tmp_path)
    resp = client.post("/api/sessions/20260801/score", json={"srcdir": str(src), "batch_size": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "score"
    assert data["session"] == "20260801"
    assert data["status"] in ("pending", "running", "done")
    assert data["total_steps"] is None  # score 步数事前不可知 → None 降级口径
    cmd = captured[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == sys.executable
    assert cmd[1:] == [
        "scripts/video.py",
        "score",
        str(src),
        "--session",
        "20260801",
        "--batch-size",
        "2",
    ]
    assert _wait_terminal(runner, data["task_id"]) == "done"


def test_score_submit_invalid_session_400(tmp_path: pathlib.Path) -> None:
    src = _make_srcdir(tmp_path)
    client, _, _ = _make_client(tmp_path)
    for bad in ("../逃逸", "a/b", "bad..name", "", "绝 对"):
        resp = client.post(f"/api/sessions/{bad}/score", json={"srcdir": str(src)})
        assert resp.status_code in (400, 404), f"session={bad!r} 未被拒绝: {resp.status_code}"


def test_score_submit_bad_srcdir_400(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    resp = client.post("/api/sessions/20260801/score", json={"srcdir": str(tmp_path / "不存在")})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---- 场次清单与状态 ----


def _make_session_dir(work: pathlib.Path, session: str = "20260801") -> pathlib.Path:
    """造场次目录：批次 1 全套产物 + confirmed roster。"""
    sd = work / session
    (sd / "review_batch1").mkdir(parents=True)
    (sd / "scorers_b1").mkdir(parents=True)
    (sd / "video_cli.json").write_text('{"version": 1}', encoding="utf-8")
    (sd / "session_facts.json").write_text('{"files": {}}', encoding="utf-8")
    (sd / "candidates_batch1.json").write_text("{}", encoding="utf-8")
    (sd / "goals_batch1.json").write_text('{"goals": []}', encoding="utf-8")
    (sd / "review_batch1" / "label.html").write_text("<html>标注</html>", encoding="utf-8")
    (sd / "scorers_b1" / "scorer.html").write_text("<html>认人</html>", encoding="utf-8")
    (sd / "roster.json").write_text(
        json.dumps({"version": 1, "confirmed": True, "players": [], "assignments": {}}),
        encoding="utf-8",
    )
    return sd


def test_sessions_list_stages(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    work = tmp_path / "work"
    # 场次 1：只到 candidates 阶段
    s1 = work / "20260701"
    s1.mkdir()
    (s1 / "candidates_batch1.json").write_text("{}", encoding="utf-8")
    # 场次 2：goals 已有但未认人
    s2 = work / "20260702"
    s2.mkdir()
    (s2 / "goals_batch1.json").write_text('{"goals": []}', encoding="utf-8")
    # 场次 3：roster 未 confirmed
    s3 = work / "20260703"
    s3.mkdir()
    (s3 / "roster.json").write_text(
        json.dumps({"version": 1, "confirmed": False, "players": [], "assignments": {}}),
        encoding="utf-8",
    )
    # 场次 4：已出合集（output 有产物）
    s4 = _make_session_dir(work, "20260704")
    out4 = tmp_path / "output" / "20260704"
    out4.mkdir(parents=True)
    (out4 / "队伍_主队_进球集锦.mp4").write_bytes(b"")
    assert s4.is_dir()
    # 非场次目录不列入
    (work / "detect").mkdir()
    (work / ".gui").mkdir()

    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = {s["session"]: s["stage"] for s in resp.json()["sessions"]}
    assert sessions["20260701"] == "candidates"
    assert sessions["20260702"] == "goals"
    assert sessions["20260703"] == "roster"
    assert sessions["20260704"] == "output"
    assert "detect" not in sessions


def test_session_status_detail(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    _make_session_dir(tmp_path / "work", "20260801")
    out = tmp_path / "output" / "20260801"
    out.mkdir(parents=True)
    (out / "合集.mp4").write_bytes(b"")

    resp = client.get("/api/sessions/20260801/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session"] == "20260801"
    assert data["stage"] == "output"
    assert data["batches"] == [
        {
            "batch": 1,
            "goals": True,
            "candidates": True,
            "label_page": "/pages/20260801/review_batch1/label.html",
            "scorer_page": "/pages/20260801/scorers_b1/scorer.html",
        }
    ]
    assert data["roster"] == {"exists": True, "confirmed": True}
    assert data["outputs"] == ["合集.mp4"]


def test_session_status_batches_candidates_only(tmp_path: pathlib.Path) -> None:
    """标注阶段场景：只有 candidates_batch1 + review_batch1（无 goals）→ 批次 1 仍被发现。"""
    client, _, _ = _make_client(tmp_path)
    sd = tmp_path / "work" / "20260802"
    (sd / "review_batch1").mkdir(parents=True)
    (sd / "candidates_batch1.json").write_text("{}", encoding="utf-8")
    (sd / "review_batch1" / "label.html").write_text("<html>标注</html>", encoding="utf-8")

    resp = client.get("/api/sessions/20260802/status")
    assert resp.status_code == 200
    assert resp.json()["batches"] == [
        {
            "batch": 1,
            "goals": False,
            "candidates": True,
            "label_page": "/pages/20260802/review_batch1/label.html",
            "scorer_page": None,
        }
    ]


def test_session_status_batches_goals_regression(tmp_path: pathlib.Path) -> None:
    """回归锁定：已有 goals_batch2 的场次输出与修复前一致（goals 源行为不变）。"""
    client, _, _ = _make_client(tmp_path)
    sd = tmp_path / "work" / "20260803"
    (sd / "review_batch2").mkdir(parents=True)
    (sd / "candidates_batch2.json").write_text("{}", encoding="utf-8")
    (sd / "goals_batch2.json").write_text('{"goals": []}', encoding="utf-8")
    (sd / "review_batch2" / "label.html").write_text("<html>标注</html>", encoding="utf-8")

    resp = client.get("/api/sessions/20260803/status")
    assert resp.status_code == 200
    assert resp.json()["batches"] == [
        {
            "batch": 2,
            "goals": True,
            "candidates": True,
            "label_page": "/pages/20260803/review_batch2/label.html",
            "scorer_page": None,
        }
    ]


def test_session_status_batches_mixed_sources(tmp_path: pathlib.Path) -> None:
    """混合：批次 1 只有 candidates、批次 2 有 goals → 两条都出且按批次号升序。"""
    client, _, _ = _make_client(tmp_path)
    sd = tmp_path / "work" / "20260804"
    (sd / "review_batch2").mkdir(parents=True)
    (sd / "candidates_batch1.json").write_text("{}", encoding="utf-8")
    (sd / "goals_batch2.json").write_text('{"goals": []}', encoding="utf-8")

    resp = client.get("/api/sessions/20260804/status")
    assert resp.status_code == 200
    batches = resp.json()["batches"]
    assert [b["batch"] for b in batches] == [1, 2]
    assert batches[0]["goals"] is False
    assert batches[0]["candidates"] is True
    assert batches[1]["goals"] is True
    assert batches[1]["label_page"] is None


def test_session_status_404_and_400(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    assert client.get("/api/sessions/20990101/status").status_code == 404
    resp = client.get("/api/sessions/bad..name/status")
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---- team-config ----


def test_team_config_write(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    sd = tmp_path / "work" / "20260801"
    sd.mkdir(parents=True)
    resp = client.post(
        "/api/sessions/20260801/team-config",
        json={"team_name": " 主队名 ", "opponent": "对手队"},
    )
    assert resp.status_code == 200
    data = json.loads((sd / "team_config.json").read_text(encoding="utf-8"))
    assert data == {"version": 1, "team_name": "主队名", "opponent": "对手队"}
    # 无 opponent：键省略（B-2 契约 schema v1 可选字段）
    resp2 = client.post("/api/sessions/20260801/team-config", json={"team_name": "只有队名"})
    assert resp2.status_code == 200
    data2 = json.loads((sd / "team_config.json").read_text(encoding="utf-8"))
    assert data2 == {"version": 1, "team_name": "只有队名"}


def test_team_config_errors(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    # 场次不存在 → 404
    resp = client.post("/api/sessions/20990101/team-config", json={"team_name": "x"})
    assert resp.status_code == 404
    # 空队名 → 400
    sd = tmp_path / "work" / "20260801"
    sd.mkdir(parents=True)
    resp2 = client.post("/api/sessions/20260801/team-config", json={"team_name": "  "})
    assert resp2.status_code == 400
    assert "error" in resp2.json()


# ---- people / build / photo 提交 ----


def test_people_submit_argv(tmp_path: pathlib.Path) -> None:
    client, runner, captured = _make_client(tmp_path)
    (tmp_path / "work" / "20260801").mkdir(parents=True)
    resp = client.post("/api/sessions/20260801/people", json={"batch": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "people"
    assert data["total_steps"] == 3  # 指定单批可预估 3 步（架构定案）
    cmd = captured[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1:] == ["scripts/video.py", "people", "--session", "20260801", "--batch", "1"]
    _wait_terminal(runner, data["task_id"])
    # 不带 batch：步数不可预估 → None
    resp2 = client.post("/api/sessions/20260801/people", json={})
    assert resp2.status_code == 200
    assert resp2.json()["total_steps"] is None
    assert captured[1]["cmd"][1:] == ["scripts/video.py", "people", "--session", "20260801"]


def test_build_submit_argv_and_mutex(tmp_path: pathlib.Path) -> None:
    client, runner, captured = _make_client(tmp_path)
    (tmp_path / "work" / "20260801").mkdir(parents=True)
    resp = client.post("/api/sessions/20260801/build", json={"all": True})
    assert resp.status_code == 200
    assert resp.json()["total_steps"] is None
    cmd = captured[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1:] == ["scripts/video.py", "build", "--session", "20260801", "--all"]
    _wait_terminal(runner, resp.json()["task_id"])
    # scorer 与 team 互斥 → 400
    resp2 = client.post("/api/sessions/20260801/build", json={"scorer": "红队-7号", "team": "红队"})
    assert resp2.status_code == 400
    assert "error" in resp2.json()


def test_photo_submit_argv(tmp_path: pathlib.Path) -> None:
    client, runner, captured = _make_client(tmp_path)
    (tmp_path / "work" / "20260801").mkdir(parents=True)
    resp = client.post("/api/sessions/20260801/photo", json={"apply": True})
    assert resp.status_code == 200
    cmd = captured[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1:] == ["scripts/video.py", "photo", "--session", "20260801", "--apply"]
    _wait_terminal(runner, resp.json()["task_id"])
    # 缺省 apply=false 不带旗标
    resp2 = client.post("/api/sessions/20260801/photo", json={})
    assert resp2.status_code == 200
    assert captured[1]["cmd"][1:] == ["scripts/video.py", "photo", "--session", "20260801"]


# ---- 任务清单 / 详情 / 取消 ----


def test_tasks_list_detail_cancel(tmp_path: pathlib.Path) -> None:
    client, runner, _ = _make_client(tmp_path, hang=True)
    src = _make_srcdir(tmp_path, name="20260801")
    resp = client.post("/api/sessions/20260801/score", json={"srcdir": str(src)})
    task_id = resp.json()["task_id"]
    deadline = time.monotonic() + 5.0
    while runner.get(task_id).status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    lst = client.get("/api/tasks")
    assert lst.status_code == 200
    tasks = {t["task_id"]: t for t in lst.json()["tasks"]}
    assert tasks[task_id]["kind"] == "score"
    assert tasks[task_id]["status"] == "running"

    detail = client.get(f"/api/tasks/{task_id}")
    assert detail.status_code == 200
    assert detail.json()["session"] == "20260801"

    cancel = client.post(f"/api/tasks/{task_id}/cancel")
    assert cancel.status_code == 200
    assert _wait_terminal(runner, task_id) == "cancelled"
    assert client.get(f"/api/tasks/{task_id}").json()["status"] == "cancelled"
    # 终态再取消 → 409；不存在 → 404
    assert client.post(f"/api/tasks/{task_id}/cancel").status_code == 409
    assert client.get("/api/tasks/不存在的任务").status_code == 404
    assert client.post("/api/tasks/不存在的任务/cancel").status_code == 404


# ---- SSE ----


def _parse_sse(body: str) -> list[dict[str, object]]:
    """解析 SSE 响应体为事件列表。"""
    out: list[dict[str, object]] = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: ") :]))
    return out


def test_sse_replay_live_and_end(tmp_path: pathlib.Path) -> None:
    """SSE：存量回放 + 增量推送，task_done 后流结束。"""
    client, runner, _ = _make_client(
        tmp_path, lines=["执行: python a.py\n", "日志行\n", "执行: python b.py\n"]
    )
    src = _make_srcdir(tmp_path, name="20260801")
    resp = client.post("/api/sessions/20260801/score", json={"srcdir": str(src)})
    task_id = resp.json()["task_id"]
    _wait_terminal(runner, task_id)  # 已结束：纯回放路径也必须给出完整事件流

    sse = client.get(f"/api/tasks/{task_id}/events")
    assert sse.status_code == 200
    assert sse.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(sse.text)
    types = [str(e["type"]) for e in events]
    assert types.count("log") == 3
    assert types.count("step_start") == 2
    assert types[-1] == "task_done"


def test_sse_replay_from_disk_after_restart(tmp_path: pathlib.Path) -> None:
    """重启口径：新 runner（内存无任务）从 jsonl 落盘整段回放。"""
    client, runner, _ = _make_client(tmp_path, lines=["执行: python a.py\n"])
    src = _make_srcdir(tmp_path, name="20260801")
    task_id = client.post("/api/sessions/20260801/score", json={"srcdir": str(src)}).json()[
        "task_id"
    ]
    _wait_terminal(runner, task_id)
    memory_events = runner.events(task_id)

    fresh_runner = TaskRunner(work_dir=tmp_path / "work")  # 模拟进程重启
    app2 = create_app(
        work_dir=tmp_path / "work", output_dir=tmp_path / "output", runner=fresh_runner
    )
    client2 = TestClient(app2)
    sse = client2.get(f"/api/tasks/{task_id}/events")
    assert sse.status_code == 200
    assert _parse_sse(sse.text) == memory_events


def test_sse_unknown_task_404(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    assert client.get("/api/tasks/不存在的任务/events").status_code == 404


# ---- 静态托管与路径穿越 ----


def test_pages_serving(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    page = tmp_path / "work" / "20260801" / "review_batch1" / "label.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>标注页</html>", encoding="utf-8")
    resp = client.get("/pages/20260801/review_batch1/label.html")
    assert resp.status_code == 200
    assert "标注页" in resp.text


def test_pages_traversal_rejected(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    secret = tmp_path / "rules.md"
    secret.write_text("机密", encoding="utf-8")
    # %2e%2e 编码穿越（httpx 会规范化字面 ..，编码形式必达路由层）
    resp = client.get("/pages/%2e%2e/rules.md")
    assert resp.status_code == 400
    assert "error" in resp.json()
    resp2 = client.get("/pages/20260801/..%2f..%2frules.md")
    assert resp2.status_code == 400
    # 不存在文件 → 404
    assert client.get("/pages/20260801/没有.html").status_code == 404


def test_index_placeholder(tmp_path: pathlib.Path) -> None:
    client, _, _ = _make_client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "basketball-clip" in resp.text


# ---- GUI 状态持久化（srcdir 记忆，gui-state-persist）----


def test_scan_persists_and_gui_state_returns_last_srcdir(tmp_path: pathlib.Path) -> None:
    # Arrange / Act：扫描成功后，gui-state 应记住素材目录
    client, _, _ = _make_client(tmp_path)
    src = _make_srcdir(tmp_path)
    resp = client.post("/api/sessions/scan", json={"srcdir": str(src)})
    assert resp.status_code == 200
    # Assert
    resp2 = client.get("/api/gui-state")
    assert resp2.status_code == 200
    assert resp2.json()["last_srcdir"] == str(src)


def test_gui_state_null_when_never_scanned(tmp_path: pathlib.Path) -> None:
    # Arrange / Act
    client, _, _ = _make_client(tmp_path)
    resp = client.get("/api/gui-state")
    # Assert：从未扫描 → null（不报错）
    assert resp.status_code == 200
    assert resp.json()["last_srcdir"] is None


def test_gui_state_corrupt_returns_null_with_warning(tmp_path: pathlib.Path) -> None:
    # Arrange：state.json 损坏
    client, _, _ = _make_client(tmp_path)
    gui_dir = tmp_path / "work" / ".gui"
    gui_dir.mkdir(parents=True)
    (gui_dir / "state.json").write_text("{坏json", encoding="utf-8")
    # Act / Assert：显式降级为 null 而非 500
    resp = client.get("/api/gui-state")
    assert resp.status_code == 200
    assert resp.json()["last_srcdir"] is None


def test_status_includes_srcdir_from_video_cli(tmp_path: pathlib.Path) -> None:
    # Arrange：场次目录带 video_cli.json（含 srcdir）+ candidates 标记
    client, _, _ = _make_client(tmp_path)
    session_dir = tmp_path / "work" / "20260801_测试队"
    session_dir.mkdir(parents=True)
    (session_dir / "candidates.json").write_text("[]", encoding="utf-8")
    (session_dir / "video_cli.json").write_text(
        json.dumps({"version": 1, "session": "20260801_测试队", "srcdir": "D:/素材/x", "runs": []}),
        encoding="utf-8",
    )
    # Act
    resp = client.get("/api/sessions/20260801_测试队/status")
    # Assert
    assert resp.status_code == 200
    assert resp.json()["srcdir"] == "D:/素材/x"


def test_status_srcdir_null_without_video_cli(tmp_path: pathlib.Path) -> None:
    # Arrange：场次目录只有 candidates 标记，无 video_cli.json
    client, _, _ = _make_client(tmp_path)
    session_dir = tmp_path / "work" / "20260801_测试队"
    session_dir.mkdir(parents=True)
    (session_dir / "candidates.json").write_text("[]", encoding="utf-8")
    # Act / Assert
    resp = client.get("/api/sessions/20260801_测试队/status")
    assert resp.status_code == 200
    assert resp.json()["srcdir"] is None
