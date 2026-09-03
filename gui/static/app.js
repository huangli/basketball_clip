"use strict";

/* basketball-clip 中文向导前端（原生 JS，无框架无 CDN，离线可用）。
 *
 * 接口契约以 gui/app.py 为准：
 * - 错误一律 {"error": "中文消息"} + 4xx/5xx
 * - 任务提交返回 {task_id, kind, session, status, step_index, total_steps, progress, ...}
 * - SSE /api/tasks/{id}/events 六种事件：step_start/step_done/step_failed/log/task_done/task_failed；
 *   total_steps 为 null 时进度降级为转圈 + 日志滚动；cancelled 无终态事件，轮询 /api/tasks/{id} 兜底。
 */

// ---- 常量 ----

var STEPS = [
  { id: "source", title: "选素材" },
  { id: "team", title: "设队名" },
  { id: "score", title: "检测" },
  { id: "label", title: "标注" },
  { id: "people", title: "认人" },
  { id: "build", title: "出合集" },
  { id: "photo", title: "照片精选" },
];

// 后端阶段 → 向导完成度映射（_session_stage 口径：取产物存在性最远档）
var STAGE_RANK = { candidates: 1, goals: 2, roster: 3, roster_confirmed: 4, output: 5 };
var STAGE_LABEL = {
  candidates: "已检测，待标注",
  goals: "已标注",
  roster: "认人待确认",
  roster_confirmed: "已认人",
  output: "已出合集",
};

// 场次 ID 合法字符集（与后端 SESSION_RE 同口径：中英文/数字/下划线/连字符，禁点号空格）
var SESSION_ILLEGAL_RE = /[^A-Za-z0-9_一-鿿-]/g;

var POLL_INTERVAL_MS = 2000; // cancelled 兜底轮询间隔
var TOAST_MS = 5000;
var FILE_LIST_PREVIEW = 50; // 文件清单一屏预览条数

// ---- 全局状态 ----

var state = {
  session: null, // 当前场次 ID
  srcdir: "", // 素材目录（score 请求体需要）
  currentStep: "source",
  status: null, // 最近一次 /api/sessions/{s}/status 响应
  scanResult: null, // 最近一次扫描响应（步骤回退时回显）
  sessionDirty: false, // 用户手改过场次 ID 后不再被扫描结果覆盖
  teamSkipped: false, // 队名步骤点了「跳过」
  runningTask: null, // { taskId, stepId } 全局同刻只跑一个任务
  pendingTeamConfig: null, // 场次目录未建时的暂存队名 {team_name, opponent}
};

// ---- DOM 小工具 ----

function $(sel) {
  return document.querySelector(sel);
}

function el(tag, cls, text) {
  var node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function toast(message, kind) {
  var box = $("#toast-container");
  var item = el("div", "toast" + (kind ? " " + kind : ""), message);
  box.appendChild(item);
  setTimeout(function () {
    item.remove();
  }, TOAST_MS);
}

// ---- API 封装：4xx/5xx 一律抛中文错误，调用方 catch 后 toast ----

function api(path, options) {
  return fetch(path, options).then(function (res) {
    if (res.ok) return res.json();
    return res
      .json()
      .catch(function () {
        return { error: "请求失败（HTTP " + res.status + "）" };
      })
      .then(function (body) {
        throw new Error(body.error || "请求失败（HTTP " + res.status + "）");
      });
  });
}

function postJson(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ---- 场次 ID 自动修正（C-2 Minor-1 口径：非法字符改下划线并提示） ----

function sanitizeSessionId(name) {
  var fixed = name.trim().replace(SESSION_ILLEGAL_RE, "_");
  return { id: fixed, fixed: fixed !== name.trim() };
}

// ---- 步骤完成度 ----

function stepDone(stepId) {
  var st = state.status;
  if (!state.session) return false;
  var rank = st && STAGE_RANK[st.stage] ? STAGE_RANK[st.stage] : 0;
  switch (stepId) {
    case "source":
      return true; // 有场次即视为素材已选
    case "team":
      return Boolean(st && st.team_config) || state.teamSkipped === true;
    case "score":
      return rank >= STAGE_RANK.candidates;
    case "label":
      return rank >= STAGE_RANK.goals;
    case "people":
      return rank >= STAGE_RANK.roster;
    case "build":
      return rank >= STAGE_RANK.output;
    case "photo":
      return false; // 照片导出完成与否后端无可判标记，恢复时不打勾
    default:
      return false;
  }
}

// ---- 侧边栏渲染 ----

function renderStepNav() {
  var list = $("#step-nav-list");
  clear(list);
  STEPS.forEach(function (step, idx) {
    var li = el("li");
    if (step.id === state.currentStep) li.classList.add("current");
    if (stepDone(step.id)) li.classList.add("done");
    var no = el("span", "step-no", stepDone(step.id) ? "✓" : String(idx + 1));
    li.appendChild(no);
    li.appendChild(el("span", null, step.title));
    li.addEventListener("click", function () {
      gotoStep(step.id);
    });
    list.appendChild(li);
  });
}

function renderSessionList(sessions) {
  var list = $("#session-list");
  clear(list);
  if (!sessions.length) {
    list.appendChild(el("li", "empty", "暂无场次，从「选素材」开始"));
    return;
  }
  sessions.forEach(function (item) {
    var li = el("li");
    if (item.session === state.session) li.classList.add("active");
    li.appendChild(el("span", null, item.session));
    li.appendChild(el("span", "session-stage", STAGE_LABEL[item.stage] || item.stage));
    li.addEventListener("click", function () {
      resumeSession(item.session);
    });
    list.appendChild(li);
  });
}

function refreshSessions() {
  return api("/api/sessions")
    .then(function (data) {
      renderSessionList(data.sessions || []);
    })
    .catch(function (e) {
      toast(e.message, "error");
    });
}

function refreshStatus() {
  if (!state.session) return Promise.resolve(null);
  return api("/api/sessions/" + encodeURIComponent(state.session) + "/status")
    .then(function (st) {
      state.status = st;
      flushPendingTeamConfig();
      renderStepNav();
      renderStep();
      return st;
    })
    .catch(function (e) {
      toast(e.message, "error");
      return null;
    });
}

// 场次恢复：点选已有场次，跳到第一个未完成步骤
function resumeSession(session) {
  state.session = session;
  state.pendingTeamConfig = null;
  state.teamSkipped = false;
  refreshStatus().then(function (st) {
    if (!st) return;
    var target = "source";
    for (var i = 0; i < STEPS.length; i++) {
      if (!stepDone(STEPS[i].id)) {
        target = STEPS[i].id;
        break;
      }
      target = STEPS[i].id;
    }
    gotoStep(target);
    toast("已续接场次：" + session, "success");
  });
}

function gotoStep(stepId) {
  state.currentStep = stepId;
  renderStepNav();
  renderStep();
}

// ---- 任务进度面板（SSE + 轮询兜底） ----

function buildTaskPanel(container) {
  clear(container);
  var panel = el("div", "task-panel");
  var statusLine = el("div", "task-status running", "任务启动中…");
  var track = el("div", "progress-track indeterminate");
  var bar = el("div", "progress-bar");
  track.appendChild(bar);
  var logBox = el("pre", "log-box");
  var cancelBtn = el("button", "btn btn-danger", "取消任务");
  panel.appendChild(statusLine);
  panel.appendChild(track);
  panel.appendChild(logBox);
  panel.appendChild(cancelBtn);
  container.appendChild(panel);
  return { panel: panel, statusLine: statusLine, track: track, bar: bar, logBox: logBox, cancelBtn: cancelBtn };
}

function appendLog(logBox, line) {
  logBox.textContent += line + "\n";
  logBox.scrollTop = logBox.scrollHeight;
}

// 更新进度：total_steps 为 null → 不定态转圈；否则按 step_index/total 推进度条
function updateProgress(ui, stepIndex, totalSteps, stepName) {
  if (totalSteps === null || totalSteps === undefined) {
    ui.track.classList.add("indeterminate");
    ui.statusLine.textContent = stepName
      ? "正在执行：" + stepName + "（进度不可预估，日志实时滚动）"
      : "任务进行中（进度不可预估，日志实时滚动）";
    return;
  }
  ui.track.classList.remove("indeterminate");
  var pct = Math.min((stepIndex / totalSteps) * 100, 100);
  ui.bar.style.width = pct + "%";
  ui.statusLine.textContent =
    "步骤 " + stepIndex + "/" + totalSteps + (stepName ? "：" + stepName : "");
}

function finishTask(ui, ok, message) {
  ui.track.classList.remove("indeterminate");
  ui.cancelBtn.disabled = true;
  if (ok) {
    ui.bar.style.width = "100%";
    ui.statusLine.className = "task-status done";
    ui.statusLine.textContent = message;
  } else {
    ui.statusLine.className = "task-status failed";
    ui.statusLine.textContent = message;
  }
}

// 失败面板：可读错误 + 「展开日志」末尾日志
function showFailureTail(ui, tail) {
  if (!tail || !tail.length) return;
  var details = el("details", "fail-tail");
  details.appendChild(el("summary", null, "展开日志（末尾 " + tail.length + " 行）"));
  details.appendChild(el("pre", null, tail.join("\n")));
  ui.panel.appendChild(details);
}

/**
 * 跑一个流水线任务：提交 → SSE 消费 → 终态收尾。
 * submitFn 返回 Promise<{task_id}>；onDone(ok) 在终态回调（成功时通常 refreshStatus）。
 */
function runTask(stepId, container, submitFn, onDone) {
  if (state.runningTask) {
    toast("已有任务正在运行，请等待完成或先取消", "error");
    return;
  }
  var ui = buildTaskPanel(container);
  submitFn()
    .then(function (task) {
      var taskId = task.task_id;
      state.runningTask = { taskId: taskId, stepId: stepId };
      appendLog(ui.logBox, "任务已提交（ID: " + taskId + "）");
      ui.cancelBtn.addEventListener("click", function () {
        api("/api/tasks/" + taskId + "/cancel", { method: "POST" })
          .then(function () {
            ui.cancelBtn.disabled = true;
          })
          .catch(function (e) {
            toast(e.message, "error");
          });
      });
      followTask(taskId, ui, function (ok, msg) {
        state.runningTask = null;
        finishTask(ui, ok, msg);
        if (onDone) onDone(ok);
      });
    })
    .catch(function (e) {
      finishTask(ui, false, e.message);
      toast(e.message, "error");
    });
}

function followTask(taskId, ui, onTerminal) {
  var finished = false;
  var source = new EventSource("/api/tasks/" + taskId + "/events");

  function terminate(ok, msg, tail) {
    if (finished) return;
    finished = true;
    source.close();
    clearInterval(pollTimer);
    if (tail) showFailureTail(ui, tail);
    onTerminal(ok, msg);
  }

  source.onmessage = function (msg) {
    var ev;
    try {
      ev = JSON.parse(msg.data);
    } catch (e) {
      return; // 降级：不识别的帧忽略，不中断任务
    }
    switch (ev.type) {
      case "log":
        appendLog(ui.logBox, ev.line || "");
        break;
      case "step_start":
        appendLog(ui.logBox, "▶ 执行: " + ev.step);
        updateProgress(ui, ev.step_index, ev.total_steps, ev.step);
        break;
      case "step_done":
        updateProgress(ui, ev.step_index, ev.total_steps, ev.step + "（完成）");
        break;
      case "step_failed":
        appendLog(ui.logBox, "✗ 步骤失败: " + ev.step);
        break;
      case "task_done":
        appendLog(ui.logBox, "任务完成");
        terminate(true, "任务完成");
        break;
      case "task_failed":
        terminate(false, "任务失败（退出码 " + ev.returncode + "）", ev.tail);
        break;
      default:
        break;
    }
  };

  // SSE 断线不判死：EventSource 自动重连，终态靠轮询兜底
  source.onerror = function () {
    appendLog(ui.logBox, "（事件流中断，自动重连/轮询兜底中…）");
  };

  // cancelled 无终态事件（runner M1）：轮询任务状态兜底收尾
  var pollTimer = setInterval(function () {
    api("/api/tasks/" + taskId)
      .then(function (task) {
        if (task.status === "cancelled") {
          appendLog(ui.logBox, "任务已取消");
          terminate(false, "任务已取消");
        } else if (task.status === "done") {
          terminate(true, "任务完成");
        } else if (task.status === "failed") {
          terminate(false, "任务失败（退出码 " + task.returncode + "）");
        }
      })
      .catch(function () {
        /* 轮询失败静默，等下一轮 */
      });
  }, POLL_INTERVAL_MS);
}

// ---- team-config 暂存（场次目录未建时 404，检测完成后补写） ----

function saveTeamConfig() {
  var session = state.session;
  if (!session || !state.pendingTeamConfig) return Promise.resolve();
  var body = state.pendingTeamConfig;
  return postJson("/api/sessions/" + encodeURIComponent(session) + "/team-config", body).then(
    function () {
      state.pendingTeamConfig = null;
    }
  );
}

function flushPendingTeamConfig() {
  if (!state.pendingTeamConfig || !state.session) return;
  saveTeamConfig()
    .then(function () {
      toast("队名已写入场次配置", "success");
    })
    .catch(function () {
      /* 场次目录还没建好，留待下次刷新再补 */
    });
}

// ---- 各步骤渲染 ----

function renderStep() {
  var container = $("#step-content");
  clear(container);
  var renderers = {
    source: renderSourceStep,
    team: renderTeamStep,
    score: renderScoreStep,
    label: renderLabelStep,
    people: renderPeopleStep,
    build: renderBuildStep,
    photo: renderPhotoStep,
  };
  (renderers[state.currentStep] || renderSourceStep)(container);
}

function stepHeader(container, title, desc) {
  container.appendChild(el("h2", null, title));
  container.appendChild(el("p", "step-desc", desc));
}

var fieldSeq = 0; // label/input 关联用自增 id

function addField(container, labelText, inputNode, hintText) {
  var field = el("div", "field");
  var label = el("label", null, labelText);
  fieldSeq += 1;
  inputNode.id = "field-" + fieldSeq;
  inputNode.name = inputNode.id;
  label.htmlFor = inputNode.id;
  field.appendChild(label);
  field.appendChild(inputNode);
  if (hintText) field.appendChild(el("div", "hint", hintText));
  container.appendChild(field);
  return inputNode;
}

function textInput(value, placeholder) {
  var input = el("input");
  input.type = "text";
  if (value) input.value = value;
  if (placeholder) input.placeholder = placeholder;
  return input;
}

// 第 1 步：选素材
function renderSourceStep(container) {
  stepHeader(container, "第 1 步：选素材", "输入或粘贴存放比赛视频的目录路径，扫描目录里的 .mp4 文件。");

  var srcInput = textInput(state.srcdir, "如 D:\\比赛视频\\2026-08-30");
  addField(container, "素材目录", srcInput);

  var scanBtn = el("button", "btn", "扫描目录");
  var resultBox = el("div");
  var row = el("div", "button-row");
  row.appendChild(scanBtn);
  container.appendChild(row);
  container.appendChild(resultBox);

  scanBtn.addEventListener("click", function () {
    var srcdir = srcInput.value.trim();
    if (!srcdir) {
      toast("请先填写素材目录", "error");
      return;
    }
    scanBtn.disabled = true;
    clear(resultBox);
    postJson("/api/sessions/scan", { srcdir: srcdir })
      .then(function (data) {
        state.srcdir = srcdir;
        renderScanResult(resultBox, data);
      })
      .catch(function (e) {
        toast(e.message, "error");
      })
      .finally(function () {
        scanBtn.disabled = false;
      });
  });

  // 已扫过则直接回显
  if (state.scanResult && state.srcdir) renderScanResult(resultBox, state.scanResult);
}

function renderScanResult(resultBox, data) {
  clear(resultBox);
  state.scanResult = data;
  resultBox.appendChild(el("p", null, "发现 " + data.count + " 个视频文件："));
  if (!data.count) {
    resultBox.appendChild(el("div", "notice", "目录里没有 .mp4 文件，请确认路径是否正确。"));
    return;
  }
  var listWrap = el("div", "file-list");
  var ul = el("ul");
  var files = data.files.slice(0, FILE_LIST_PREVIEW);
  files.forEach(function (name) {
    ul.appendChild(el("li", null, name));
  });
  if (data.files.length > FILE_LIST_PREVIEW) {
    ul.appendChild(el("li", null, "… 等共 " + data.count + " 个文件"));
  }
  listWrap.appendChild(ul);
  resultBox.appendChild(listWrap);

  var guess = sanitizeSessionId(data.session || "");
  if (state.sessionDirty && state.session) {
    guess = { id: state.session, fixed: false };
  }
  state.session = guess.id;
  if (guess.fixed) {
    resultBox.appendChild(
      el(
        "div",
        "notice",
        "场次 ID 已自动修正：「" + (data.session || "") + "」→「" + guess.id + "」（目录名含空格/点等不允许字符）"
      )
    );
  }

  var sessionHint = el(
    "div",
    "hint",
    "默认取目录名，可改（仅中英文/数字/下划线/连字符）"
  );
  var sessionInput = textInput(guess.id, "场次 ID");
  sessionInput.addEventListener("input", function () {
    var v = sanitizeSessionId(sessionInput.value);
    state.session = v.id;
    state.sessionDirty = true;
    sessionHint.textContent = v.fixed
      ? "场次 ID 已自动修正：非法字符已替换为下划线"
      : "默认取目录名，可改（仅中英文/数字/下划线/连字符）";
  });
  addField(resultBox, "场次 ID", sessionInput);
  sessionInput.parentNode.appendChild(sessionHint);

  var nextBtn = el("button", "btn", "下一步：设队名");
  nextBtn.addEventListener("click", function () {
    if (!state.session) {
      toast("场次 ID 不能为空", "error");
      return;
    }
    gotoStep("team");
  });
  var row = el("div", "button-row");
  row.appendChild(nextBtn);
  resultBox.appendChild(row);
}

// 第 2 步：设队名
function renderTeamStep(container) {
  stepHeader(
    container,
    "第 2 步：设队名",
    "填写自家队名（必填）与对手名（可选），用于合集命名；也可跳过，用默认「主队」。"
  );
  var existing = state.status && state.status.team_config;
  if (existing) {
    container.appendChild(
      el("div", "notice info", "当前已配置：队名「" + existing.team_name + "」" + (existing.opponent ? "，对手「" + existing.opponent + "」" : ""))
    );
  }
  var teamInput = textInput(existing ? existing.team_name : "", "如：半截篮");
  var oppInput = textInput(existing && existing.opponent ? existing.opponent : "", "如：老对手（可留空）");
  addField(container, "队名（必填）", teamInput);
  addField(container, "对手名（可选）", oppInput);

  var saveBtn = el("button", "btn", "保存并继续");
  var skipBtn = el("button", "btn btn-secondary", "跳过（用默认「主队」）");
  var row = el("div", "button-row");
  row.appendChild(saveBtn);
  row.appendChild(skipBtn);
  container.appendChild(row);

  saveBtn.addEventListener("click", function () {
    var teamName = teamInput.value.trim();
    if (!teamName) {
      toast("队名不能为空（或点「跳过」用默认队名）", "error");
      return;
    }
    var body = { team_name: teamName, opponent: oppInput.value.trim() || null };
    state.pendingTeamConfig = body;
    saveBtn.disabled = true;
    postJson("/api/sessions/" + encodeURIComponent(state.session) + "/team-config", body)
      .then(function () {
        state.pendingTeamConfig = null;
        toast("队名已保存", "success");
        refreshStatus().then(function () {
          gotoStep("score");
        });
      })
      .catch(function (e) {
        // 场次目录还没建（检测前）→ 暂存，检测完成后自动补写
        if (e.message.indexOf("场次不存在") >= 0) {
          toast("队名已记录，将在检测完成后自动写入", "success");
          gotoStep("score");
        } else {
          state.pendingTeamConfig = null;
          toast(e.message, "error");
        }
      })
      .finally(function () {
        saveBtn.disabled = false;
      });
  });

  skipBtn.addEventListener("click", function () {
    state.teamSkipped = true;
    gotoStep("score");
  });
}

// 第 3 步：检测
function renderScoreStep(container) {
  stepHeader(
    container,
    "第 3 步：检测",
    "机器扫描全部视频找出疑似进球片段（耗时较长，日志实时滚动）。"
  );
  container.appendChild(el("div", "notice info", "场次：" + (state.session || "未选择")));

  var srcInput = textInput(state.srcdir, "素材目录路径（重新检测时必填）");
  addField(container, "素材目录", srcInput, state.srcdir ? "" : "续接的场次未记录素材目录，如需重新检测请重新填写");
  srcInput.addEventListener("input", function () {
    state.srcdir = srcInput.value.trim();
  });

  var batchInput = el("input");
  batchInput.type = "number";
  batchInput.min = "1";
  var adv = el("details", "advanced");
  adv.appendChild(el("summary", null, "高级选项"));
  var advField = el("div", "field");
  var advLabel = el("label", null, "每批视频数（batch_size，留空用默认）");
  batchInput.id = "score-batch-size";
  batchInput.name = batchInput.id;
  advLabel.htmlFor = batchInput.id;
  advField.appendChild(advLabel);
  advField.appendChild(batchInput);
  adv.appendChild(advField);
  container.appendChild(adv);

  var startBtn = el("button", "btn", "开始检测");
  var row = el("div", "button-row");
  row.appendChild(startBtn);
  container.appendChild(row);
  var taskBox = el("div");
  container.appendChild(taskBox);

  startBtn.addEventListener("click", function () {
    if (!state.session) {
      toast("请先完成第 1 步选素材", "error");
      return;
    }
    var srcdir = srcInput.value.trim();
    if (!srcdir) {
      toast("请填写素材目录", "error");
      return;
    }
    state.srcdir = srcdir;
    var body = { srcdir: srcdir };
    if (batchInput.value) body.batch_size = parseInt(batchInput.value, 10);
    runTask(
      "score",
      taskBox,
      function () {
        return postJson("/api/sessions/" + encodeURIComponent(state.session) + "/score", body);
      },
      function (ok) {
        if (!ok) return;
        // 检测完成，场次目录已建 → 补写暂存的队名
        saveTeamConfig()
          .catch(function (e) {
            toast(e.message, "error");
          })
          .finally(function () {
            refreshStatus().then(function () {
              gotoStep("label");
            });
          });
      }
    );
  });
}

// 第 4 步：标注
function renderLabelStep(container) {
  stepHeader(
    container,
    "第 4 步：标注",
    "在标注页里逐个确认/否决机器找出的片段，标注完成后回到这里。"
  );
  var batches = (state.status && state.status.batches) || [];
  var labelPages = batches.filter(function (b) {
    return b.label_page;
  });
  if (!labelPages.length) {
    container.appendChild(el("div", "notice", "尚未生成标注页（请先完成检测）；若刚跑完检测，点「刷新状态」。"));
  } else {
    labelPages.forEach(function (b) {
      var btn = el("button", "btn", "打开第 " + b.batch + " 批标注页");
      btn.addEventListener("click", function () {
        window.open(b.label_page, "_blank");
      });
      var row = el("div", "button-row");
      row.appendChild(btn);
      container.appendChild(row);
    });
    container.appendChild(el("div", "notice info", "提示：标注页在新标签页打开，标注完成后回到这里点「刷新状态」。"));
  }
  var refreshBtn = el("button", "btn btn-secondary", "刷新状态");
  var nextBtn = el("button", "btn", "下一步：认人");
  nextBtn.disabled = !stepDone("label");
  refreshBtn.addEventListener("click", function () {
    refreshStatus();
  });
  nextBtn.addEventListener("click", function () {
    gotoStep("people");
  });
  var row = el("div", "button-row");
  row.appendChild(refreshBtn);
  row.appendChild(nextBtn);
  container.appendChild(row);
}

// 第 5 步：认人
function renderPeopleStep(container) {
  stepHeader(
    container,
    "第 5 步：认人",
    "机器裁出每个进球者照片并生成确认页，由你在确认页里指定进球归属。"
  );
  var batchInput = el("input");
  batchInput.type = "number";
  batchInput.min = "1";
  addField(container, "只处理某一批（可选，留空处理全部批次）", batchInput);

  var startBtn = el("button", "btn", "开始认人");
  var row = el("div", "button-row");
  row.appendChild(startBtn);
  container.appendChild(row);
  var taskBox = el("div");
  container.appendChild(taskBox);
  var linksBox = el("div");
  container.appendChild(linksBox);
  renderScorerLinks(linksBox);

  startBtn.addEventListener("click", function () {
    var body = {};
    if (batchInput.value) body.batch = parseInt(batchInput.value, 10);
    runTask(
      "people",
      taskBox,
      function () {
        return postJson("/api/sessions/" + encodeURIComponent(state.session) + "/people", body);
      },
      function (ok) {
        if (ok) refreshStatus();
      }
    );
  });
}

function renderScorerLinks(container) {
  clear(container);
  var batches = (state.status && state.status.batches) || [];
  var pages = batches.filter(function (b) {
    return b.scorer_page;
  });
  if (!pages.length) return;
  container.appendChild(el("p", null, "认人确认页（在新标签页打开，确认导出后回来刷新）："));
  pages.forEach(function (b) {
    var btn = el("button", "btn", "打开第 " + b.batch + " 批认人确认页");
    btn.addEventListener("click", function () {
      window.open(b.scorer_page, "_blank");
    });
    var row = el("div", "button-row");
    row.appendChild(btn);
    container.appendChild(row);
  });
  var refreshBtn = el("button", "btn btn-secondary", "刷新状态");
  refreshBtn.addEventListener("click", function () {
    refreshStatus();
  });
  var nextBtn = el("button", "btn", "下一步：出合集");
  nextBtn.addEventListener("click", function () {
    gotoStep("build");
  });
  var row = el("div", "button-row");
  row.appendChild(refreshBtn);
  row.appendChild(nextBtn);
  container.appendChild(row);
}

// 第 6 步：出合集
function renderBuildStep(container) {
  stepHeader(container, "第 6 步：出合集", "把确认的进球合成集锦视频（默认出全部：队伍集锦 + 个人合集）。");

  var adv = el("details", "advanced");
  adv.appendChild(el("summary", null, "高级选项：只出某一个人 / 某一个队"));
  var scorerInput = textInput("", "进球者姓名/标签（与队名过滤二选一）");
  var teamInput = textInput("", "队名（与进球者过滤二选一）");
  addField(adv, "只出该进球者", scorerInput);
  addField(adv, "只出该队伍", teamInput);
  container.appendChild(adv);

  var startBtn = el("button", "btn", "开始合成");
  var row = el("div", "button-row");
  row.appendChild(startBtn);
  container.appendChild(row);
  var taskBox = el("div");
  container.appendChild(taskBox);
  var outputBox = el("div");
  container.appendChild(outputBox);
  renderOutputs(outputBox);

  startBtn.addEventListener("click", function () {
    var scorer = scorerInput.value.trim();
    var team = teamInput.value.trim();
    if (scorer && team) {
      toast("进球者和队伍过滤只能填一个（都不填 = 出全部）", "error");
      return;
    }
    var body = { all: !(scorer || team), scorer: scorer, team: team };
    runTask(
      "build",
      taskBox,
      function () {
        return postJson("/api/sessions/" + encodeURIComponent(state.session) + "/build", body);
      },
      function (ok) {
        if (ok) refreshStatus();
      }
    );
  });
}

function renderOutputs(container) {
  clear(container);
  var outputs = (state.status && state.status.outputs) || [];
  if (!outputs.length) return;
  container.appendChild(el("p", null, "产物清单（output/" + state.session + "/）："));
  var ul = el("ul", "output-list");
  outputs.forEach(function (name) {
    ul.appendChild(el("li", null, name));
  });
  container.appendChild(ul);
}

// 第 7 步：照片精选
function renderPhotoStep(container) {
  stepHeader(
    container,
    "第 7 步：照片精选",
    "从进球瞬间抽帧打分生成照片确认页，确认后导出精选照片。"
  );

  var genBtn = el("button", "btn", "生成照片确认页");
  var openBtn = el("button", "btn btn-secondary", "打开照片确认页");
  var applyBtn = el("button", "btn", "导出精选");
  var photoPage = state.status && state.status.photo_page;
  openBtn.disabled = !photoPage;
  openBtn.title = photoPage ? "" : "请先生成照片确认页";

  var row = el("div", "button-row");
  row.appendChild(genBtn);
  row.appendChild(openBtn);
  row.appendChild(applyBtn);
  container.appendChild(row);
  container.appendChild(
    el("div", "notice info", "流程：① 生成确认页 → ② 打开确认页勾选照片 → ③ 回到这里点「导出精选」。")
  );
  var taskBox = el("div");
  container.appendChild(taskBox);

  genBtn.addEventListener("click", function () {
    runTask(
      "photo",
      taskBox,
      function () {
        return postJson("/api/sessions/" + encodeURIComponent(state.session) + "/photo", { apply: false });
      },
      function (ok) {
        if (ok) refreshStatus();
      }
    );
  });
  openBtn.addEventListener("click", function () {
    if (photoPage) window.open(photoPage, "_blank");
  });
  applyBtn.addEventListener("click", function () {
    runTask(
      "photo",
      taskBox,
      function () {
        return postJson("/api/sessions/" + encodeURIComponent(state.session) + "/photo", { apply: true });
      },
      function (ok) {
        if (ok) {
          toast("精选照片已导出", "success");
          refreshStatus();
        }
      }
    );
  });
}

// ---- 诊断日志导出（D-1 契约：GET /api/diagnostics → zip；未就绪置灰） ----

function initDiagnostics() {
  var btn = $("#btn-diagnostics");
  fetch("/api/diagnostics", { method: "HEAD" })
    .then(function (res) {
      if (!res.ok) return; // 未就绪保持置灰
      btn.disabled = false;
      btn.title = "打包下载诊断日志（zip）";
      btn.addEventListener("click", function () {
        window.location.href = "/api/diagnostics";
      });
    })
    .catch(function () {
      /* 探测失败保持置灰 */
    });
  btn.addEventListener("click", function () {
    if (btn.disabled) toast("诊断日志导出将在下一任务（D-1）落地后可用");
  });
}

// ---- 启动 ----

function init() {
  renderStepNav();
  renderStep();
  refreshSessions();
  initDiagnostics();
  $("#btn-new-session").addEventListener("click", function () {
    state.session = null;
    state.srcdir = "";
    state.status = null;
    state.scanResult = null;
    state.sessionDirty = false;
    state.pendingTeamConfig = null;
    state.teamSkipped = false;
    renderSessionList([]);
    refreshSessions();
    gotoStep("source");
  });
}

document.addEventListener("DOMContentLoaded", init);
