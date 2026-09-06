#!/usr/bin/env python3
"""生成认人确认页 scorer.html：用户逐球确认进球者归属（spec: docs/scorer/spec.md T4）。

读取 crop_scorers 产出的 scorer_candidates.json（每球一条：key/裁图/clip 预览
片段/team_guess/SKIP 状态）+ goals.json（confirmed 球为页面条目全集），在同目录
生成自包含 scorer.html（数据内联、裁图/视频相对路径、按键+按钮、localStorage
进度、一键导出 roster.json——文件名即 CLI 认的名字，移到 work/<场次>/
即可接入 people 预填链与 build，无需改名）。用户浏览器打开即可：看裁图与预览片段，
点球员按钮（数字键 1-9）或自由文本输入归属，S 跳过；SKIP 球标"无法定位"
照常列出可手选；导出物 schema 严格过 scripts/roster.py（format_key 键、
validate_roster 可校验），confirmed=true 仅当全部非 SKIP 球已归属。

视频来源优先级：candidates 的 "clip"（按进球锚点现切的预览片段，与裁图同球
同时刻，crop_scorers --rawdir 产物）＞ events_index 的 clip_wide 匹配（仅作
无预览片段时的兜底——事件片段覆盖长事件全程，开头可能是另一回合）。

输入：--scorers scorer_candidates.json、--goals goals.json、--session（缺省取
    candidates 里的 session）、--index（可选 events_index.json，兜底视频按
    src_file 相同且 |anchor_t0−anchor_time|≤4s 匹配 clip_wide）、--players（可选
    "黑21=张三,白-李四=李四" 式逗号分隔名单）、--players-file（可选，与
    roster.players 同构的 JSON 数组名单文件，与 --players 互斥；spec:
    docs/scorer-reid/spec.md）、--roster-existing（可选，
    合并已有 roster：assignments 并集预填、players 以新名单为准缺 tag WARNING）、
    --clusters（可选 scorer_clusters.json，必须与 --scorers 同目录：rep_crops 与
    裁图同目录相对引用；有则页面顶部出簇区，簇级选人批量预填簇内全部球，
    逐球区单独改覆盖簇归属，导出 roster 契约不变；spec: docs/scorer-cluster/spec.md）、
    --photo-matches（可选 photo_matches.json，必须与 --scorers 同目录，与 --clusters
    校验同构；照片库识别预填——优先级 读号命中 > 照片命中 > 印名匹配 > 空白，
    读号/照片冲突时预填读号、照片候选在条目上出角标（号码+得分）供人工点击切换，
    名单缺号注入占位条目 <主队名><号码>（team=主队名 随 players 注入，不靠前缀推队）；
    无此参数页面行为与旧版完全一致；spec: docs/photo-roster/spec.md T5）、
    --track-links（可选 track_links.json，必须与 --scorers 同目录，与 --clusters
    校验同口径；轨迹传播预填——条目显示"轨迹#N"，用户逐球归属（非"不算进球"）时
    同文件同轨迹且无 marks/无号码预填/未手改的球自动写入 marks 并记 propagateAssign
    provenance（localStorage 独立键），徽标"同轨迹预填"供扫一眼复核；acceptAll/E 键
    只收号码/照片预填不碰传播预填，导出照旧 marks 全集；无此参数页面行为与现状完全
    一致；spec: docs/scorer-propagate/spec.md §页面）
输出：<scorer_candidates.json 同目录>/scorer.html
依赖：scripts/roster.py（format_key/validate_roster/Player/player_from_dict，
    契约唯一入口）、scripts/pipe_common.py（read_json/run_id 日志）、scripts/errors.py
典型调用：
    python scripts/gen_scorer_page.py --scorers work/20260722/scorers/scorer_candidates.json \
        --goals work/20260722/goals.json --session 20260722 \
        --index work/20260722/review_v3/events_index.json \
        --players-file work/20260722/players.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from errors import BasketballPipelineError, SchemaError
from pipe_common import configure_logging, new_run_id, read_json
from roster import (
    OPPONENT_TAG,
    Player,
    format_key,
    opponent_of,
    player_from_dict,
    validate_roster,
)
from team_config import DEFAULT_OPPONENT, TeamConfig, load_team_config

if TYPE_CHECKING:
    # 仅类型注解用；运行时延迟 import（photo_match_scorers 链带 numpy/sklearn，
    # 只在启用 --photo-matches 时加载，无参数零开销零行为变化）
    from photo_match_scorers import MatchEntry

logger = logging.getLogger(__name__)

# 兜底视频匹配：同 src_file 且事件锚点与进球锚点相差不超过 4s（spec T4；
# 4s = 剪辑窗口前段长度，同一片段窗口内的事件视为同一球）
CLIP_MATCH_MAX_DT_SEC: float = 4.0

STATUS_OK: str = "OK"
STATUS_SKIP: str = "SKIP"

TEAM_CASUAL: str = "便服"
# 我方/对手队名不再硬编码：主队名由 team_config.json 会话级注入（main 读
# work/<场次>/team_config.json，缺失回退 team_config.DEFAULT_TEAM_NAME），
# 对手名 = 配置 opponent ＞ opponent_of(session) 场次 ID 后缀派生 ＞ 默认兜底
OPPONENT_FALLBACK: str = DEFAULT_OPPONENT  # 无后缀且未配置 opponent 的兜底
# 标签前缀 → 阵营（顺序即优先级；蓝色27 归对手系用户 2026-08-09 口径）
_TEAM_PREFIXES: tuple[tuple[str, str], ...] = (
    ("黑", "opp"),
    ("蓝", "opp"),
    ("白", "home"),
)
# team_guess 合法值：crop_scorers 颜色分队产出的是颜色（黑/白/便服），与队名不同命名空间
TEAM_GUESS_VALUES: tuple[str, ...] = ("黑", "白", "便服")
# track_links.json 契约版本（与 propagate_scorers.TRACK_LINKS_VERSION 同步；
# propagate_scorers 反向 import 本模块，直接引用会循环 import，故本地写死）
TRACK_LINKS_VERSION: str = "track-v1"

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>认人确认 __SESSION__</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; margin: 16px; }
#bar { position: sticky; top: 0; background: #111; padding: 8px 0; z-index: 9; }
button { font-size: 18px; padding: 10px 18px; margin: 4px; border-radius: 8px;
         border: 0; cursor: pointer; }
button.sel { outline: 3px solid #fc3; }
.team-opp { background: #222; color: #fff; border: 1px solid #666; }
.team-home { background: #eee; color: #111; }
.team-casual { background: #777; color: #fff; }
.teamlabel { color: #aaa; margin-right: 6px; }
.nav { background: #444; color: #fff; }
#skip { background: #7a5c00; color: #fff; }
#nogoal { background: #7a2c2c; color: #fff; }
#accept { background: #2c9e4b; color: #fff; }
/* 照片候选角标：读号/照片冲突时显示，点击改用照片归属（docs/photo-roster T5） */
#photoaccept { background: #0f6e6e; color: #fff; }
#export { background: #8a6d00; color: #fff; }
#go { background: #2c9e4b; color: #fff; }
#free { font-size: 18px; padding: 8px; width: 10em; background: #222;
        color: #eee; border: 1px solid #555; border-radius: 8px; }
.stepbar { color: #fc3; font-size: 14px; margin: 10px 0 2px; }
.stepbar small { color: #999; margin-left: 8px; font-size: 12px; }
.renamebtn { font-size: 12px; padding: 4px 8px; }
#reviewbar { margin: 4px 0; }
#reviewbar button { font-size: 14px; padding: 6px 10px; }
/* 逐球区：图/视频定高不定宽顶端对齐（68vh 等高、翻球不跳、无黑边——
   固定框留边方案有黑边已证伪，docs/scorer-three-step/spec.md Objective 第 5 条） */
#review { display: flex; align-items: flex-start; gap: 8px; }
#review #crop { height: 68vh; width: auto; background: #000; }
#review video { height: 68vh; width: auto; background: #000; }
.badge { color: #fc3; }
small { color: #999; }
#clusters { margin: 8px 0; }
.cluster-row { display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
               background: #1c1c1c; border: 1px solid #333; border-radius: 8px;
               padding: 6px; margin: 6px 0; }
.cluster-row img.rep { max-height: 120px; max-width: 160px; background: #000; }
.clusterlabel { color: #fc3; margin: 0 8px; }
.cluster-row { cursor: grab; }
.cluster-row.drop-target { outline: 3px dashed #fc3; }
.cluster-row.merge-src { outline: 3px solid #fc3; }
.teamrow.drop-target { outline: 3px dashed #fc3; }
.cluster-row button { font-size: 14px; padding: 6px 10px; }
.picker { background: #2a2a12; border: 1px solid #fc3; border-radius: 8px;
          padding: 6px; margin: 4px 0; width: 100%; }
.picker .hint { color: #fc3; margin-right: 8px; }
.cluster-row.collapsed img.rep { max-height: 48px; max-width: 64px; }
.cluster-row .foldbtn { font-size: 12px; padding: 2px 8px; }
/* 悬停放大浮层：置于样式表末尾——#review #crop:hover 特异度压过 #review #crop；
   .cluster-row img.rep:hover 与 .cluster-row.collapsed img.rep 同特异度靠后写胜出
   （点击放大已证伪；移开即收回） */
#review #crop:hover, .cluster-row img.rep:hover {
  position: fixed; z-index: 99; left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  height: 92vh; width: auto; max-width: 96vw; max-height: 96vh;
  outline: 3px solid #fc3; background: #000;
}
</style>
</head>
<body>
<div id="bar">
  <span id="prog"></span> <span class="badge" id="cur"></span><br>
  <div class="stepbar">第一步：判队伍<small>拖队员到正确队伍行，点"改名"填真名</small></div>
  <span id="players"></span>
  <input id="free" placeholder="自由输入标签"><button id="go">归属 (回车)</button>
  <button id="accept" style="display:none"></button>
  <button id="photoaccept" style="display:none"></button>
  <button id="oppmark" class="team-opp" title="该球是对手进的：一键归属对手">标为对手</button>
  <button id="acceptall"
    title="对所有号码/照片预填无歧义且未手改的球批量预填归属（不标已核，第三步可翻检）"
  >接受全部号码预填</button>
  <button id="skip">跳过 (S)</button>
  <button id="nogoal">不算进球 (N)</button>
  <button class="nav" id="prev">← 上一个</button>
  <button class="nav" id="next">下一个 →</button>
  <button class="nav" id="toun">跳到未归属</button>
  <button id="export">导出 roster.json</button>
  <br><small>按键：1-9=选球员 E=采用号码预填 S=跳过 N=不算进球
  ←/→=翻页；SKIP 球标"无法定位"可手选</small>
  <small>进度自动存 localStorage，刷新回到上次位置；导出文件名 roster.json</small>
</div>
<div class="stepbar" id="step2">第二步：并簇认人<small>同人的簇拖到一起，点队员名应用到整组；
误分组的簇点"删除"移除（不动球和归属）</small></div>
<div id="clusters"></div>
<div class="stepbar">第三步：逐球核对<small>选核对对象，判错直接点正确球员</small></div>
<div id="reviewbar"></div>
<div id="review">
<img id="crop" alt="投篮者裁图">
<video id="v" autoplay loop muted playsinline></video>
</div>
<script>
const ITEMS = __ITEMS__;
const PLAYERS = __PLAYERS__;
const EXISTING = __EXISTING__;
const EXPLAYERS = __EXPLAYERS__;
const CLUSTERS = __CLUSTERS__;
const SESSION = "__SESSION__";
const OPP = __OPP__;
const OPP_TAG = __OPP_TAG__;  // 伪球员"对手"标签（opponent-filter T2；roster.py OPPONENT_TAG 注入）
const HOME = __HOME__;
const LSKEY = "scorer_" + SESSION;
const POSKEY = LSKEY + "_pos";
const TOUCHKEY = LSKEY + "_touched";
// 不算进球哨兵标签：假进球/犯规不算的球归到这里——只在页面内流转，
// 导出 roster 时剔除（assignments/players 都不含），不挡 confirmed；可逆（改归球员即恢复）
const NOGOAL = "不算进球";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(LSKEY) || "{}"); } catch (e) { marks = {}; }
// 已有 roster 归属作底，本页改动覆盖之（用户在页面上的修改是终裁）
marks = Object.assign({}, EXISTING, marks);
// touched = 逐球手动改过的 key（簇级批量预填不得覆盖；独立 localStorage 键，
// 不动既有 marks 存储格式）
let touched = {};
try { touched = JSON.parse(localStorage.getItem(TOUCHKEY) || "{}"); } catch (e) { touched = {}; }
const PROPKEY = LSKEY + "_propagate";
// 传播预填 provenance：用户逐球归属时沿同轨迹自动写入 marks 的 key 集合
// （"同轨迹预填"徽标判定用；独立 localStorage 键，读回合并写与 touched 同模式）
let propagateAssign = {};
try { propagateAssign = JSON.parse(localStorage.getItem(PROPKEY) || "{}"); }
catch (e) { propagateAssign = {}; }
const CLSTATE_KEY = LSKEY + "_clusters";
// 簇合并页面态：merges=被并cid→组id，clAssign=组id→tag（仅作合并预填来源，
// 显示/折叠判定一律以 marks 为准），collapsed=显式折叠（true/false 都存），
// deleted=删簇墓碑（gid→true，只加不减；删的是显示组，不动球和归属）
let clState = { merges: {}, clAssign: {}, collapsed: {}, deleted: {} };
try {
  const rawCl = JSON.parse(localStorage.getItem(CLSTATE_KEY) || "{}");
  if (rawCl && typeof rawCl === "object") {
    for (const sub of ["merges", "clAssign", "collapsed", "deleted"]) {
      if (rawCl[sub] && typeof rawCl[sub] === "object") clState[sub] = rawCl[sub];
    }
  }
} catch (e) { clState = { merges: {}, clAssign: {}, collapsed: {}, deleted: {} }; }
let pickerGid = null; // 合并弹条：非 null = 该组行正弹选人条
let collapseAll = null; // 总开关：null=随规则 / true=全折 / false=全展（瞬态，刷新回规则；
                        // 点击后 null→true→false→true… 两态循环回不到"随规则"系有意
                        // 为之——回规则态靠刷新，spec 未要求三态）
let mergeSrc = null; // 点选合并：非 null = 该 gid 组已被点为源（瞬态，刷新即清；
                     // 与拖拽并存，合并语义复用 mergeInto）
const TEAMOVR_KEY = LSKEY + "_teamovr";
// 队员改队覆盖：{ tag: team }；改队直接写 PLAYERS 内存值，导出自动跟随
let teamOvr = {};
try { teamOvr = JSON.parse(localStorage.getItem(TEAMOVR_KEY) || "{}"); }
catch (e) { teamOvr = {}; }
for (const p of PLAYERS) {
  if (teamOvr[p.tag] !== undefined) p.team = teamOvr[p.tag];
}
function saveTeamOvr() {
  // 读回再合并写，防多开页面互踩（沿用 save() 模式）
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(TEAMOVR_KEY) || "{}"); }
  catch (e) { stored = {}; }
  teamOvr = Object.assign(stored, teamOvr);
  localStorage.setItem(TEAMOVR_KEY, JSON.stringify(teamOvr));
}
function changeTeam(tag, team) {
  // 拖拽改队：只动队别（分队合集/分行/着色），不碰任何 marks 归属
  const p = PLAYERS.find(x => x.tag === tag);
  if (!p || p.team === team) return; // 原队行 drop = 无操作
  p.team = team;
  teamOvr[tag] = team;
  saveTeamOvr();
  show(cur);
}
const NAMES_KEY = LSKEY + "_names";
// 页内改真名覆盖：{ tag: name }；清空真名=写空串不删键（读回合并写会复活删键，
// saveClState 前科），加载时空串视为无真名
let nameOvr = {};
try { nameOvr = JSON.parse(localStorage.getItem(NAMES_KEY) || "{}"); }
catch (e) { nameOvr = {}; }
for (const p of PLAYERS) {
  if (nameOvr[p.tag] !== undefined) p.name = nameOvr[p.tag];
}
function saveNames() {
  // 读回再合并写，防多开页面互踩（沿用 save() 模式）
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(NAMES_KEY) || "{}"); }
  catch (e) { stored = {}; }
  nameOvr = Object.assign(stored, nameOvr);
  localStorage.setItem(NAMES_KEY, JSON.stringify(nameOvr));
}
function renamePlayer(tag) {
  // 只改真名不改标签（tag 是归属键，级联风险）；三态：非空=改 / 空串=清 / 取消=不动；
  // 四处按钮文字（队伍区主行/兜底行/簇区/弹条）都读 p.name，改内存值 show 即全刷
  const p = PLAYERS.find(x => x.tag === tag);
  if (!p) return;
  const v = prompt("真名（空=清除）", p.name);
  if (v === null) return;
  p.name = v.trim();
  nameOvr[tag] = p.name;
  saveNames();
  show(cur);
}
const REVIEW_KEY = LSKEY + "_review";
// 按人核对：{ target: "" }；""=全部（切回全部=写空串不删键，理由同 names 键），
// "__none__"=未归属，其余=球员 tag（含名单外自由输入 tag，无 name 纯显示 tag）
let review = { target: "" };
try {
  const rawR = JSON.parse(localStorage.getItem(REVIEW_KEY) || "{}");
  if (rawR && typeof rawR === "object" && typeof rawR.target === "string") {
    review.target = rawR.target;
  }
} catch (e) { review = { target: "" }; }
function saveReview() {
  // 读回再合并写（沿用 save() 模式）
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(REVIEW_KEY) || "{}"); }
  catch (e) { stored = {}; }
  review = Object.assign(stored, review);
  localStorage.setItem(REVIEW_KEY, JSON.stringify(review));
}
function reviewTargets() {
  // 核对对象候选 = 当前 marks 里有归属球的 tag（按 ITEMS 序去重；含名单外 tag）
  const seen = [];
  for (const it of ITEMS) {
    const t = marks[it.key];
    if (t && !seen.includes(t)) seen.push(t);
  }
  return seen;
}
function visible() {
  // 可见集 = 按核对对象过滤的 ITEMS 子集；ITEMS 本体不动（spec 边界）
  if (review.target === "") return ITEMS;
  if (review.target === "__none__") return ITEMS.filter(it => !marks[it.key]);
  return ITEMS.filter(it => marks[it.key] === review.target);
}
function posKey() {
  // 位置按核对对象分键（不同对象下同索引指向不同球，不分键会错位）；
  // 全部沿用旧 _pos 键兼容存量
  return review.target === "" ? POSKEY
    : POSKEY + "_" + encodeURIComponent(review.target);
}
function reviewTarget(tag) {
  // 切核对对象：持久 + 定位（有位置记录回记录；无则按人=第一个球，
  // 全部/未归属=第一个未归属球）；集空交给 show 的空态分支回退全部
  review.target = tag;
  saveReview();
  const vis = visible();
  let start = parseInt(localStorage.getItem(posKey()) || "-1", 10);
  if (isNaN(start) || start < 0 || start >= vis.length) {
    start = (tag !== "" && tag !== "__none__")
      ? 0 : vis.findIndex(it => !marks[it.key]);
  }
  show(start >= 0 ? start : 0);
}
function renderReviewBar() {
  // 核对对象行：全部 / 各已归属球员（marks 里有球才列）/ 未归属；选中态 sel 高亮
  const bar = document.getElementById("reviewbar");
  bar.innerHTML = "";
  const lab = document.createElement("span");
  lab.textContent = "核对对象：";
  lab.className = "teamlabel";
  bar.appendChild(lab);
  const mk = (text, target) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.className = "nav";
    if (review.target === target) b.classList.add("sel");
    b.onclick = () => reviewTarget(target);
    bar.appendChild(b);
  };
  mk("全部", "");
  for (const t of reviewTargets()) {
    const p = PLAYERS.find(x => x.tag === t);
    mk(t + (p && p.name ? "=" + p.name : ""), t);
  }
  mk("未归属", "__none__");
}
function saveClState(del) {
  // 子键分别读回再合并写（嵌套对象整体浅合并会丢多页防护粒度；spec 数据契约）；
  // del = { merges: [...], clAssign: [...] } 待删键——读回合并会把本地已删的键
  // 从 stored 复活，必须在合并后再删（拆开/合并吸收依赖此语义）
  //（deleted 墓碑只加不减，del 清单无需扩展）
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(CLSTATE_KEY) || "{}"); }
  catch (e) { stored = {}; }
  const merged = {
    merges: Object.assign({}, stored.merges || {}, clState.merges),
    clAssign: Object.assign({}, stored.clAssign || {}, clState.clAssign),
    collapsed: Object.assign({}, stored.collapsed || {}, clState.collapsed),
    deleted: Object.assign({}, stored.deleted || {}, clState.deleted),
  };
  for (const k of (del && del.merges) || []) delete merged.merges[k];
  for (const k of (del && del.clAssign) || []) delete merged.clAssign[k];
  clState = merged;
  localStorage.setItem(CLSTATE_KEY, JSON.stringify(merged));
}
function groupIdOf(cid) {
  // 沿 merges 链解析最终组 id；环防御：visited 集合，成环即停不报错
  let cur = String(cid);
  const seen = new Set([cur]);
  while (clState.merges[cur] !== undefined &&
         !seen.has(String(clState.merges[cur]))) {
    cur = String(clState.merges[cur]);
    seen.add(cur);
  }
  const gid = parseInt(cur, 10);
  return isNaN(gid) ? cid : gid;
}
function computeGroups() {
  // CLUSTERS → 显示组：keys/rep_crops 按原簇序拼接；组位置 = gid 原簇原位
  const byGid = new Map();
  for (const cl of CLUSTERS) {
    const gid = groupIdOf(cl.cluster_id);
    if (!byGid.has(gid)) byGid.set(gid, { gid, cids: [], keys: [], rep_crops: [] });
    const g = byGid.get(gid);
    g.cids.push(cl.cluster_id);
    g.keys = g.keys.concat(cl.keys);
    g.rep_crops = g.rep_crops.concat(cl.rep_crops);
  }
  const pos = new Map(CLUSTERS.map((cl, i) => [cl.cluster_id, i]));
  // ?? 0 兜底：localStorage 残留失效簇 id 时 pos.get 为 undefined，防 NaN 序不稳
  // 删簇墓碑过滤在折叠成显示组之后：删的是用户肉眼所见的行；
  // merges 链不动（groupIdOf 照常解析，逐球区"簇#N"标注保留）
  return [...byGid.values()]
    .filter(g => !clState.deleted[String(g.gid)])
    .sort((a, b) => (pos.get(a.gid) ?? 0) - (pos.get(b.gid) ?? 0));
}
function groupTag(g) {
  // 组内非空 marks 众数（显示"归的人"唯一口径；不读 clAssign）
  const counts = {};
  let assigned = 0;
  for (const k of g.keys) {
    const t = marks[k];
    if (t) { counts[t] = (counts[t] || 0) + 1; assigned++; }
  }
  let best = "", n = 0;
  for (const t of Object.keys(counts)) {
    if (counts[t] > n) { n = counts[t]; best = t; }
  }
  return { tag: best, mixed: Object.keys(counts).length > 1, assigned };
}
let cur = 0;
function save() {
  // 合并写入：先读回存储与本页记录合并再写，防止同时开多个页面互相覆盖
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(LSKEY) || "{}"); } catch (e) { stored = {}; }
  marks = Object.assign(stored, marks);
  localStorage.setItem(LSKEY, JSON.stringify(marks));
  let storedTouched = {};
  try { storedTouched = JSON.parse(localStorage.getItem(TOUCHKEY) || "{}"); }
  catch (e) { storedTouched = {}; }
  touched = Object.assign(storedTouched, touched);
  localStorage.setItem(TOUCHKEY, JSON.stringify(touched));
  let storedProp = {};
  try { storedProp = JSON.parse(localStorage.getItem(PROPKEY) || "{}"); }
  catch (e) { storedProp = {}; }
  propagateAssign = Object.assign(storedProp, propagateAssign);
  localStorage.setItem(PROPKEY, JSON.stringify(propagateAssign));
}
function teamOfTag(tag) {
  // 与 Python 端 team_of_tag 同规则：标签前缀定队，黑/蓝→对手队（OPP），白→主队（HOME），其余便服
  if (tag === OPP_TAG) return OPP;  // 伪球员"对手"特判：固定对手队，不落便服
  if (tag.startsWith("黑") || tag.startsWith("蓝")) return OPP;
  if (tag.startsWith("白")) return HOME;
  return "便服";
}
function teamClass(team) {
  // 队名→CSS 语义类：任意对手/主队队名都能渲染（队名随场次配置，类名固定）
  if (team === HOME) return "team-home";
  if (team === "便服") return "team-casual";
  return "team-opp";
}
function nDone() { return ITEMS.filter(it => marks[it.key]).length; }
function renderPlayers() {
  // 按队分行（对手队 OPP/主队 HOME/便服），找人不用扫全名单（2026-08-09 用户要求）
  const box = document.getElementById("players");
  box.innerHTML = "";
  const vis = visible(); // sel 高亮读可见集当前项（按人核对时 ITEMS[cur] 不是当前球）
  const curKey = vis.length && cur < vis.length ? vis[cur].key : null;
  const numbered = PLAYERS.map((p, idx) => [p, idx]);
  const KNOWN_TEAMS = [OPP, HOME, "便服"];
  for (const tm of KNOWN_TEAMS) {
    const row = numbered.filter(([p]) => p.team === tm);
    // 三行恒渲染（空队也渲染行，否则该队零队员时无处可拖入）
    const div = document.createElement("div");
    div.className = "teamrow";
    div.dataset.team = tm;
    div.ondragover = (ev) => {
      // 只响应队员拖拽（text/player-tag）；簇行拖拽（text/plain）不高亮
      if (!ev.dataTransfer.types.includes("text/player-tag")) return;
      ev.preventDefault();
      div.classList.add("drop-target");
    };
    div.ondragleave = () => div.classList.remove("drop-target");
    div.ondrop = (ev) => {
      ev.preventDefault();
      div.classList.remove("drop-target");
      const tag = ev.dataTransfer.getData("text/player-tag");
      if (tag) changeTeam(tag, tm);
    };
    const lab = document.createElement("span");
    lab.textContent = tm + "：";
    lab.className = "teamlabel";
    div.appendChild(lab);
    for (const [p, idx] of row) {
      const b = document.createElement("button");
      b.textContent = (idx < 9 ? (idx + 1) + " " : "") + p.tag +
        (p.name ? "=" + p.name : "");
      b.className = teamClass(p.team);
      b.draggable = true;
      b.ondragstart = (ev) => {
        // 自定义 MIME：与簇行拖拽的 text/plain 隔离，防跨域误触发
        ev.dataTransfer.setData("text/player-tag", p.tag);
        ev.dataTransfer.effectAllowed = "move";
      };
      if (curKey && marks[curKey] === p.tag) b.classList.add("sel");
      b.onclick = () => assign(p.tag);
      div.appendChild(b);
      const rn = document.createElement("button");
      rn.textContent = "改名";
      rn.className = "nav renamebtn";
      rn.title = "改真名（不改标签）";
      rn.onclick = () => renamePlayer(p.tag);
      div.appendChild(rn);
    }
    box.appendChild(div);
  }
  // 兜底行：roster team 与当前三行都不匹配的队员（如场次改名后复用旧 roster，
  // team 还是旧对手名）——不归行会静默消失，归"其他"行保证可选（2026-08-15 终审）
  const rest = numbered.filter(([p]) => !KNOWN_TEAMS.includes(p.team));
  if (rest.length) {
    const div = document.createElement("div");
    div.className = "teamrow";
    const lab = document.createElement("span");
    lab.textContent = "其他（team 口径不符）：";
    lab.className = "teamlabel";
    div.appendChild(lab);
    for (const [p, idx] of rest) {
      const b = document.createElement("button");
      b.textContent = (idx < 9 ? (idx + 1) + " " : "") + p.tag +
        (p.name ? "=" + p.name : "");
      b.className = teamClass(p.team);
      b.draggable = true;
      b.ondragstart = (ev) => {
        // 自定义 MIME：与簇行拖拽的 text/plain 隔离，防跨域误触发
        ev.dataTransfer.setData("text/player-tag", p.tag);
        ev.dataTransfer.effectAllowed = "move";
      };
      if (curKey && marks[curKey] === p.tag) b.classList.add("sel");
      b.onclick = () => assign(p.tag);
      div.appendChild(b);
      const rn = document.createElement("button");
      rn.textContent = "改名";
      rn.className = "nav renamebtn";
      rn.title = "改真名（不改标签）";
      rn.onclick = () => renamePlayer(p.tag);
      div.appendChild(rn);
    }
    box.appendChild(div);
  }
}
function groupLabel(g) {
  // 组标签：簇#gid（N 球，已归属 X[，并自 #a/#b]）；未并过无"并自"段
  const t = groupTag(g);
  let s = "簇#" + g.gid + "（" + g.keys.length + " 球，已归属 " + t.assigned;
  if (g.cids.length > 1) {
    s += "，并自 " + g.cids.filter(c => c !== g.gid).map(c => "#" + c).join("/");
  }
  return s + "）";
}
function splitGroup(gid) {
  // 拆开 = 删 merges 中指向该组的所有条目；不动 marks / 目标组 clAssign；
  // doomed 必须传给 saveClState 的删除清单，否则读回合并会把删除的键复活
  const doomed = Object.keys(clState.merges)
    .filter(k => groupIdOf(parseInt(k, 10)) === gid);
  for (const k of doomed) delete clState.merges[k];
  saveClState({ merges: doomed, clAssign: [] });
  show(cur);
}
function deleteCluster(gid) {
  // 删簇 = 墓碑隐藏显示组：只移除分组视图，ITEMS/marks/touched/clAssign 一律不动
  // （簇只是分组预填，组内球在逐球区照常核对）；无页内撤销，找回=清站点数据
  const g = computeGroups().find(x => x.gid === gid);
  if (!g) return;
  if (!confirm("删除簇#" + gid + "？组内 " + g.keys.length +
               " 球的归属不变，可在第三步逐球核对")) return;
  clState.deleted[String(gid)] = true;
  if (pickerGid === gid) pickerGid = null; // 顺手清悬挂弹条状态（组已不渲染）
  saveClState();
  show(cur);
}
function mergeInto(srcGid, dstGid) {
  // 拖拽合并：被并组全部原始簇指向目标组；预填来源 = 目标组 clAssign，
  // 无则组内非空 marks 全一致的 tag，混合/未归不预填；被并组 clAssign 删除
  srcGid = groupIdOf(srcGid);
  dstGid = groupIdOf(dstGid);
  if (srcGid === dstGid) return; // 自身/同组无操作
  const groups = computeGroups();
  const src = groups.find(g => g.gid === srcGid);
  const dst = groups.find(g => g.gid === dstGid);
  if (!src || !dst) return;
  for (const cid of src.cids) clState.merges[String(cid)] = dstGid;
  let tag = clState.clAssign[String(dstGid)];
  if (!tag) {
    const ts = dst.keys.map(k => marks[k]).filter(Boolean);
    if (ts.length && ts.every(x => x === ts[0])) tag = ts[0];
  }
  if (tag) {
    for (const k of src.keys) { if (!touched[k]) marks[k] = tag; }
  }
  const delAssign = [];
  for (const cid of src.cids) {
    const k = String(cid);
    delete clState.clAssign[k]; // 本地有无都删：stored 里独有的残留键靠删除清单压住
    delAssign.push(k);
  }
  save();
  saveClState({ merges: [], clAssign: delAssign });
  show(cur);
  // PICKER-HOOK 已挂接：未自动预填 → 就地弹选人条（spec 合并动作 7）
  if (!tag) openPicker(dstGid);
}
function pickMerge(gid) {
  // 点选合并：未选源→记源；点源行→取消；点目标行→并入（先清态再合并，
  // mergeInto 内部 show 重渲染，避免残态参与渲染）
  if (mergeSrc === null) { mergeSrc = gid; show(cur); return; }
  if (mergeSrc === gid) { mergeSrc = null; show(cur); return; }
  const src = mergeSrc;
  mergeSrc = null;
  mergeInto(src, gid);
}
function openPicker(gid) {
  pickerGid = gid;
  show(cur);
}
function closePicker() {
  if (pickerGid === null) return;
  pickerGid = null;
  show(cur);
}
function isCollapsed(g) {
  // 优先级：总开关 > 显式 collapsed > 默认规则（组内全部球有 marks → 折叠）
  if (collapseAll !== null) return collapseAll;
  const ex = clState.collapsed[String(g.gid)];
  if (ex !== undefined) return !!ex;
  return g.keys.every(k => marks[k]);
}
function toggleCollapse(gid) {
  const g = computeGroups().find(x => x.gid === gid);
  if (!g) return;
  clState.collapsed[String(gid)] = !isCollapsed(g);
  saveClState();
  show(cur);
}
function renderClusters() {
  // 簇区按显示组渲染：图墙拼接 + 组标签 + 拆开钮（合并组才有）+ 选人按钮；
  // 无簇数据整区隐藏（无 --clusters 行为同旧版）
  const box = document.getElementById("clusters");
  box.innerHTML = "";
  const step2 = document.getElementById("step2");
  if (!CLUSTERS.length) {
    box.style.display = "none";
    if (step2) step2.style.display = "none";
    return;
  }
  box.style.display = "block";
  if (step2) step2.style.display = "block";
  const tbar = document.createElement("div");
  const tall = document.createElement("button");
  tall.textContent = "全部展开/折叠";
  tall.className = "nav";
  tall.onclick = () => {
    collapseAll = collapseAll === null ? true : !collapseAll;
    renderClusters();
  };
  tbar.appendChild(tall);
  box.appendChild(tbar);
  const groups = computeGroups();
  // 点选合并残态守卫：源组被并走/被删/被拆开后不在可见组里即清态
  // （mergeInto/splitGroup/deleteCluster 都经 show→renderClusters，此处一处全覆盖）
  if (mergeSrc !== null && !groups.some(g => g.gid === mergeSrc)) mergeSrc = null;
  for (const g of groups) {
    const row = document.createElement("div");
    row.className = "cluster-row";
    row.dataset.gid = g.gid;
    const folded = isCollapsed(g);
    if (folded) row.classList.add("collapsed");
    const fb = document.createElement("button");
    fb.textContent = folded ? "▸" : "▾";
    fb.className = "foldbtn nav";
    fb.title = folded ? "展开" : "折叠";
    fb.onclick = () => toggleCollapse(g.gid);
    row.appendChild(fb);
    row.draggable = true;
    row.ondragstart = (ev) => {
      ev.dataTransfer.setData("text/plain", String(g.gid));
      ev.dataTransfer.effectAllowed = "move";
    };
    row.ondragover = (ev) => {
      // 只响应簇行拖拽（text/plain）；队员拖拽（text/player-tag）不高亮
      if (!ev.dataTransfer.types.includes("text/plain")) return;
      ev.preventDefault();
      row.classList.add("drop-target");
    };
    row.ondragleave = () => row.classList.remove("drop-target");
    row.ondrop = (ev) => {
      // 同上守卫：队员 tag 拖到簇行不得触发合并
      if (!ev.dataTransfer.types.includes("text/plain")) return;
      ev.preventDefault();
      row.classList.remove("drop-target");
      const src = parseInt(ev.dataTransfer.getData("text/plain"), 10);
      if (!isNaN(src)) mergeInto(src, g.gid);
    };
    for (const rc of folded ? g.rep_crops.slice(0, 1) : g.rep_crops) {
      const im = document.createElement("img");
      im.src = rc;
      im.className = "rep";
      im.alt = "簇代表图";
      row.appendChild(im);
    }
    const lab = document.createElement("span");
    lab.className = "clusterlabel";
    const gt = groupTag(g);
    lab.textContent = groupLabel(g) +
      (gt.tag ? " → " + gt.tag + (gt.mixed ? "（混合）" : "") : "");
    row.appendChild(lab);
    if (!folded && g.cids.length > 1) {
      const sp = document.createElement("button");
      sp.textContent = "拆开";
      sp.className = "nav";
      sp.onclick = () => splitGroup(g.gid);
      row.appendChild(sp);
    }
    if (!folded) {
      for (const p of PLAYERS) {
        const b = document.createElement("button");
        b.textContent = p.tag + (p.name ? "=" + p.name : "");
        b.className = teamClass(p.team);
        b.onclick = () => clusterAssign(g.gid, p.tag);
        row.appendChild(b);
      }
    }
    const del = document.createElement("button");
    del.textContent = "删除";
    del.className = "nav";
    del.title = "移除该簇分组（不动球和归属）";
    del.onclick = () => deleteCluster(g.gid);
    row.appendChild(del);
    const mg = document.createElement("button");
    mg.textContent = mergeSrc === null ? "合并"
      : (mergeSrc === g.gid ? "取消" : "并入这里");
    mg.className = "nav";
    mg.title = "点选合并：先点源行，再点目标行（拖拽也行）";
    mg.onclick = () => pickMerge(g.gid);
    row.appendChild(mg);
    if (mergeSrc === g.gid) row.classList.add("merge-src");
    if (pickerGid === g.gid) {
      const pk = document.createElement("div");
      pk.className = "picker";
      const hint = document.createElement("span");
      hint.className = "hint";
      hint.textContent = "合并完成，选人应用到整组（" + g.keys.length + " 球）：";
      pk.appendChild(hint);
      for (const p of PLAYERS) {
        const b = document.createElement("button");
        b.textContent = p.tag + (p.name ? "=" + p.name : "");
        b.className = teamClass(p.team);
        b.onclick = () => { pickerGid = null; clusterAssign(g.gid, p.tag); };
        pk.appendChild(b);
      }
      const cancel = document.createElement("button");
      cancel.textContent = "取消";
      cancel.className = "nav";
      cancel.onclick = () => closePicker();
      pk.appendChild(cancel);
      row.appendChild(pk);
    }
    box.appendChild(row);
  }
}
function clusterAssign(cid, tag) {
  // 簇级选人 = 按组批量预填：只写未 touched 的 key（逐球覆盖优先）；
  // 记 clAssign 作合并预填来源（spec：clAssign 唯一用途）
  const gid = groupIdOf(cid);
  const g = computeGroups().find(x => x.gid === gid);
  if (!g) return;
  for (const k of g.keys) { if (!touched[k]) marks[k] = tag; }
  clState.clAssign[String(gid)] = tag;
  save();
  saveClState();
  show(cur);
}
function show(i) {
  let vis = visible();
  let flash = "";
  if (!vis.length && review.target !== "") {
    // 空可见集（改归离集/持久 target 失效）→ 提示并自动切回全部（spec 空态契约；
    // 不得像旧版 !ITEMS.length 早退那样停在旧画面）
    flash = review.target === "__none__" ? "未归属清零，已切回全部 | "
      : "此人核对完毕，已切回全部 | ";
    review.target = "";
    saveReview();
    vis = visible();
  }
  if (!vis.length) return; // ITEMS 本身为空（无球）：旧行为不变
  cur = Math.max(0, Math.min(i, vis.length - 1));
  const it = vis[cur];
  const img = document.getElementById("crop");
  if (it.crop) { img.src = it.crop; img.style.display = "inline-block"; }
  else { img.removeAttribute("src"); img.style.display = "none"; }
  const v = document.getElementById("v");
  if (it.clip) { v.src = it.clip; v.style.display = "inline-block"; v.play().catch(() => {}); }
  else { v.pause(); v.removeAttribute("src"); v.load(); v.style.display = "none"; }
  localStorage.setItem(posKey(), String(cur));
  let info = flash + `第 ${cur + 1}/${vis.length} 个`;
  if (review.target !== "") {
    // 进度行带核对对象后缀（有真名则 tag=真名）；全部模式不带
    const rp = PLAYERS.find(x => x.tag === review.target);
    info += "（核对：" + (review.target === "__none__" ? "未归属"
      : review.target + (rp && rp.name ? "=" + rp.name : "")) + "）";
  }
  info += ` | 已归属 ${nDone()}/${ITEMS.length} | ${it.file} t=${it.anchor_time}s`;
  if (it.cluster_id) info += ` | 簇#${groupIdOf(it.cluster_id)}`;
  // 轨迹号按文件内编号（track_links 契约；无 --track-links 时条目 track_id 全 null 不显示）
  if (it.track_id !== null && it.track_id !== undefined) info += " | 轨迹#" + it.track_id;
  // 预填优先级：号码匹配（K3 读号）> 照片库识别 > 印名匹配 > 颜色 team_guess；歧义不预填
  const ab = document.getElementById("accept");
  if (it.status === "SKIP") info += " | 无法定位";
  else if (it.prefill_tag) {
    info += ` | ${it.prefill_note === "photo" ? "照片预填" : "号码预填"}:${it.prefill_tag}`;
  }
  else if (it.prefill_note === "ambiguous") info += " | 号码歧义(同号多人)";
  else if (it.team_guess) info += ` | 颜色预填:${it.team_guess}`;
  const ng = it.number_guess;
  if (ng && ng.number) info += ` (读号:${ng.color || ""}${ng.number})`;
  // 传播预填徽标（判定式 spec §页面写死）：marks 有值 + 在 propagateAssign + 未手改；
  // 用户逐球改归即 touched，徽标消失（手改是终裁，高于传播预填）
  if (marks[it.key] && propagateAssign[it.key] && !touched[it.key]) info += " | 同轨迹预填";
  if (it.prefill_tag) {
    ab.textContent = `采用 ${it.prefill_tag} (E)`;
    ab.style.display = "inline-block";
    ab.onclick = () => assign(it.prefill_tag);
  } else {
    ab.style.display = "none";
    ab.onclick = null;
  }
  // 照片候选角标：与预填不一致（读号/照片冲突、读号歧义或无预填）时显示
  // 号码+得分，点击改用照片归属——不静默覆盖（spec: docs/photo-roster T5）
  const pgb = document.getElementById("photoaccept");
  const pg = it.photo_guess;
  if (pg && pg.tag !== it.prefill_tag) {
    info += ` | 照片候选:${pg.number}号(得分${pg.score.toFixed(3)})`;
    pgb.textContent = `改用照片:${pg.tag}`;
    pgb.style.display = "inline-block";
    pgb.onclick = () => assign(pg.tag);
  } else {
    pgb.style.display = "none";
    pgb.onclick = null;
  }
  // 标为对手（opponent-filter T2）：一键归属伪球员；已标则变为撤销
  const ob = document.getElementById("oppmark");
  if (marks[it.key] === OPP_TAG) {
    ob.textContent = "撤销对手标记";
    ob.onclick = () => unassign();
  } else {
    ob.textContent = "标为对手";
    ob.onclick = () => assign(OPP_TAG);
  }
  document.getElementById("prog").textContent = info;
  document.getElementById("cur").textContent =
    marks[it.key] ? "当前归属: " + marks[it.key] : "未归属";
  renderPlayers();
  renderClusters();
  renderReviewBar();
}
function propagateFrom(srcKey, tag) {
  // 轨迹传播（spec: docs/scorer-propagate/spec.md §页面写死）：用户逐球归属时，
  // 同文件同 track_id 且无 marks/无号码预填（prefill_tag）/未 touched 的球自动
  // 写入 marks 并记 propagateAssign provenance；"不算进球"哨兵绝不传播；
  // 轨迹不跨文件（file 同判，track_id 只是文件内编号）
  if (tag === NOGOAL) return;
  const src = ITEMS.find(x => x.key === srcKey);
  if (!src || src.track_id === null || src.track_id === undefined) return;
  for (const it of ITEMS) {
    if (it.key === srcKey) continue;
    if (it.file !== src.file || it.track_id !== src.track_id) continue;
    if (marks[it.key] || it.prefill_tag || touched[it.key]) continue;
    marks[it.key] = tag;
    propagateAssign[it.key] = true;
  }
}
function assign(tag) {
  const vis = visible();
  if (!vis.length) return;
  marks[vis[cur].key] = tag;
  touched[vis[cur].key] = true;
  propagateFrom(vis[cur].key, tag);
  save();
  if (review.target !== "") {
    // 按人/未归属模式：改归后球离集，落原索引位置的新当前项（[i] 即下一个），到尾停末尾
    show(cur);
    return;
  }
  // 全部模式 = 现状：跳下一个未归属球（全局 findIndex）
  let nxt = vis.findIndex((x, idx) => idx > cur && !marks[x.key]);
  if (nxt < 0) nxt = vis.findIndex(x => !marks[x.key]);
  show(nxt >= 0 ? nxt : cur);
}
function skip() {
  const vis = visible();
  if (!vis.length) return;
  let nxt = vis.findIndex((x, idx) => idx > cur && !marks[x.key]);
  if (nxt < 0) nxt = (cur + 1) % vis.length;
  show(nxt);
}
function freeAssign() {
  const inp = document.getElementById("free");
  const tag = inp.value.trim();
  if (!tag) return;
  if (tag === "__none__") { inp.value = ""; return; } // 保留特殊值，防撞未归属集语义
  inp.value = "";
  assign(tag);
}
function jumpUnassigned() {
  // 跳到未归属 = 切到未归属核对对象（spec 手工清单：两者一致）
  reviewTarget("__none__");
}
function unassign() {
  // 撤销归属（opponent-filter T2 配套）：删 marks/touched 回未归属，位置不动
  const vis = visible();
  if (!vis.length) return;
  const key = vis[cur].key;
  delete marks[key];
  delete touched[key];
  save();
  show(cur);
}
function exportRoster() {
  // assignments 并集 = 已有 roster 归属 + 本页全部标记（键即 candidates 的
  // format_key 产物，两端共用 roster.py 契约，此处不再拼键）
  const assignments = {};
  // 不算进球哨兵剔除：不进 assignments（players 自动补录循环读本对象，哨兵随之不进名单）
  for (const [k, t] of Object.entries(marks)) { if (t && t !== NOGOAL) assignments[k] = t; }
  // players 以本页名单为准；归属到名单外标签（自由输入）的自动补录，
  // 名字/队别优先沿用已有 roster 记录，否则按标签前缀推队
  // 伪球员"对手"零引用剔除：没标过对手球的场次，roster.players 不带它（保持干净）
  const used = new Set(Object.values(assignments));
  const players = PLAYERS.filter(p => p.tag !== OPP_TAG || used.has(OPP_TAG))
    .map(p => ({ tag: p.tag, name: p.name, team: p.team }));
  const known = new Set(players.map(p => p.tag));
  for (const t of new Set(Object.values(assignments))) {
    if (known.has(t)) continue;
    const old = EXPLAYERS[t];
    players.push({ tag: t, name: old ? old.name : "", team: old ? old.team : teamOfTag(t) });
    known.add(t);
  }
  // confirmed=true 仅当全部非 SKIP 球已归属（SKIP 球允许未归属，spec 契约）
  const confirmed = ITEMS.every(it => it.status === "SKIP" || marks[it.key]);
  const payload = JSON.stringify({ session: SESSION, confirmed, players, assignments }, null, 1);
  const blob = new Blob([payload], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "roster.json";
  a.click();
  const nUn = ITEMS.filter(it => it.status !== "SKIP" && !marks[it.key]).length;
  // nNo 数全量 marks 的哨兵球——同 session 跨批次共享 localStorage，
  // 只数本页 ITEMS 会漏报其他批次的剔除球
  const nNo = Object.values(marks).filter(t => t === NOGOAL).length;
  alert("已下载 roster.json（归属 " + Object.keys(assignments).length +
        "/" + ITEMS.length + "，confirmed=" + confirmed +
        (nNo ? "，不算进球 " + nNo + " 球（已剔除不参与合成）" : "") +
        (nUn ? "，还有 " + nUn + " 个非 SKIP 球未归属" : "") + "），移到 work 场次目录即可");
}
function acceptAllPrefills() {
  // 一键全收预填（号码 read-numbers-batch / 照片 photo-roster）：仅 prefill_tag
  // 非空（号码唯一命中或照片命中）且未手改（非 touched）的球写入 marks；
  // 不标 touched——预填非终裁，簇级选人仍可覆盖、第三步逐球核对可翻检。
  // 歧义球（prefill_tag 为空 + prefill_note="ambiguous"）与 SKIP 球（无预填）
  // 天然不满足条件。计数按 prefill_note 拆分：note==="photo" 计照片，
  // 其余（含无 note 旧数据）计号码（review MEDIUM-1）。
  let n = 0, nPhoto = 0, nAmb = 0, nTouched = 0;
  for (const it of ITEMS) {
    if (it.prefill_note === "ambiguous") nAmb++;
    if (!it.prefill_tag) continue;
    if (touched[it.key]) { nTouched++; continue; }
    if (marks[it.key] === it.prefill_tag) continue; // 幂等：已是该预填不重复计数
    marks[it.key] = it.prefill_tag;
    n++;
    if (it.prefill_note === "photo") nPhoto++;
  }
  save();
  show(cur);
  alert("已接受 " + n + " 个预填（号码 " + (n - nPhoto) + " / 照片 " + nPhoto +
        "；歧义 " + nAmb + " / 已手改 " + nTouched + " 跳过）");
}
document.getElementById("go").onclick = freeAssign;
document.getElementById("acceptall").onclick = acceptAllPrefills;
document.getElementById("skip").onclick = skip;
document.getElementById("nogoal").onclick = () => assign(NOGOAL);
document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
document.getElementById("toun").onclick = jumpUnassigned;
document.getElementById("export").onclick = exportRoster;
document.addEventListener("keydown", (ev) => {
  const k = ev.key.toLowerCase();
  if (pickerGid !== null) {
    // 弹条期间：Esc 关闭；数字键 1-9/E 屏蔽（防误触逐球归属改错球）
    if (ev.key === "Escape") { closePicker(); return; } // 弹条优先：一次 Esc 只关弹条
    if ((k >= "1" && k <= "9") || k === "e" || k === "n") return;
  }
  if (ev.key === "Escape" && pickerGid === null && mergeSrc !== null) {
    // 点选合并 Esc 取消（弹条开着时 Esc 优先只关弹条，再按一次才清点选态——
    // 避免一次按键双清两态；不屏蔽数字键/E，点选态不影响逐球归属）
    mergeSrc = null;
    show(cur);
    return;
  }
  if (ev.target && ev.target.id === "free") {
    // 弹条打开时 Enter 也不许绕过屏蔽做逐球归属（free 聚焦态可拖拽合并）
    if (ev.key === "Enter" && pickerGid === null) freeAssign();
    return;
  }
  if (k >= "1" && k <= "9") {
    const idx = parseInt(k, 10) - 1;
    if (idx < PLAYERS.length) assign(PLAYERS[idx].tag);
  } else if (k === "s") skip();
  else if (k === "n") assign(NOGOAL);
  else if (k === "e") {
    const vis = visible();
    if (vis.length && cur < vis.length && vis[cur].prefill_tag) assign(vis[cur].prefill_tag);
  }
  else if (ev.key === "ArrowLeft") show(cur - 1);
  else if (ev.key === "ArrowRight") show(cur + 1);
});
document.addEventListener("click", (ev) => {
  if (pickerGid === null) return;
  if (ev.target && ev.target.closest && ev.target.closest(".picker")) return;
  closePicker();
});
// 启动：恢复核对对象（其集无球时 show 空态分支自动回退全部）→ 读该对象的位置键；
// 无记录则：全部/未归属=第一个未归属球，按人=第一个球
const vis0 = visible();
let start = parseInt(localStorage.getItem(posKey()) || "-1", 10);
if (isNaN(start) || start < 0 || start >= vis0.length) {
  start = (review.target !== "" && review.target !== "__none__")
    ? 0 : vis0.findIndex(it => !marks[it.key]);
}
show(start >= 0 ? start : 0);
</script>
</body>
</html>
"""


def team_of_tag(tag: str, opp: str, home: str) -> str:
    """按标签前缀推定队别：黑*/蓝*→对手队（opp）、白*→主队（home），其余归便服。

    页面导出自动补录名单外标签时用同一规则（JS teamOfTag 与本文档同步，
    改规则须两端一起改）。蓝色27 归对手系 2026-08-09 用户口径。

    Args:
        tag: 球员标签，如 ``黑21`` / ``白-李四`` / ``灰T恤-A``。
        opp: 对手队名（opponent_of 产物）。
        home: 主队名（team_config 注入）。

    Returns:
        opp / home / "便服"。
    """
    if tag == OPPONENT_TAG:
        # 伪球员"对手"固定对手队（opponent-filter T2，特判先于前缀推队，防落便服）
        return opp
    for prefix, side in _TEAM_PREFIXES:
        if tag.startswith(prefix):
            return opp if side == "opp" else home
    return TEAM_CASUAL


def parse_players(spec: str, opp: str, home: str) -> list[Player]:
    """解析 --players 名单串："黑21=张三,白-李四=李四" → Player 列表。

    每条为 ``tag[=name]``（name 可省，省则为空串）；队别按 team_of_tag 推定。

    Args:
        spec: 逗号分隔的名单串；空串返回空列表。
        opp: 对手队名（opponent_of 产物，传给 team_of_tag）。
        home: 主队名（team_config 注入，传给 team_of_tag）。

    Returns:
        Player 列表（保持给定顺序）。

    Raises:
        SchemaError: 条目为空 tag（如 "、" 或 "=张三"）。
    """
    players: list[Player] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        tag, sep, name = item.partition("=")
        tag = tag.strip()
        if not tag:
            raise SchemaError(f"--players 条目缺 tag: {item!r}")
        players.append(
            Player(tag=tag, name=name.strip() if sep else "", team=team_of_tag(tag, opp, home))
        )
    return players


def load_players_file(path: Path) -> list[Player]:
    """加载 --players-file 名单文件：JSON 数组，与 roster.players 同构。

    每条记录的校验复用 roster.player_from_dict（tag 非空唯一 / name 为 str /
    team 为任意非空 str——对手队名随场次，见 docs/session-opponent-name/spec.md），
    与 roster.json 同一契约入口（rules.md §0.2：
    schema 损坏必须显式失败，不静默容错）。

    Args:
        path: 名单文件路径（如 ``work/<场次>/players.json``）。

    Returns:
        Player 列表（保持文件顺序）。

    Raises:
        SchemaError: 坏 JSON / 顶层非数组 / 记录结构损坏 / tag 重复 / 队名非法。
        OSError: IO 重试耗尽（文件不存在等）。
    """
    data: Any = read_json(path, what="players 名单文件")
    if not isinstance(data, list):
        raise SchemaError(
            f"{path}: 顶层必须是数组（与 roster.players 同构），实际 {type(data).__name__}"
        )
    seen_tags: set[str] = set()
    return [player_from_dict(raw, str(path), i, seen_tags) for i, raw in enumerate(data)]


def merge_assignments(
    existing: dict[str, str], new: dict[str, str], what: str = "roster 合并"
) -> dict[str, str]:
    """assignments 并集合并：同键同值幂等，同键不同值 = 冲突显式失败（spec T4）。

    Args:
        existing: 已有 roster 的 assignments。
        new: 新增 assignments。
        what: 业务名称（用于错误信息）。

    Returns:
        合并后的 assignments（existing 在前，new 覆盖同键——但同键不同值已抛错，
        所以覆盖只发生在同值情形）。

    Raises:
        BasketballPipelineError: 同键冲突（调用方口径 = 报错退出 1）。
    """
    for key, value in new.items():
        if key in existing and existing[key] != value:
            raise BasketballPipelineError(
                f"{what}: 同键冲突 {key!r}: {existing[key]!r} vs {value!r}"
            )
    return {**existing, **new}


def _validate_candidates(data: Any, path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """校验 scorer_candidates.json 结构（rules.md §0.2：schema 损坏显式失败）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        candidates 记录列表（保留原始 dict）。

    Raises:
        SchemaError: 顶层非对象 / 缺 candidates 列表 / 记录缺 key/status 等字段或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    candidates: Any = data.get("candidates")
    if not isinstance(candidates, list):
        raise SchemaError(f"{path}: 缺 candidates 列表或类型错误")
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            raise SchemaError(f"{path}: 第{i}条候选不是对象")
        if not isinstance(c.get("key"), str) or not c["key"]:
            raise SchemaError(f"{path}: 第{i}条候选 key 缺失或不是非空 str")
        if c.get("status") not in (STATUS_OK, STATUS_SKIP):
            raise SchemaError(f"{path}: 第{i}条候选 status 非法: {c.get('status')!r}")
        if not isinstance(c.get("file"), str) or not c["file"]:
            raise SchemaError(f"{path}: 第{i}条候选 file 缺失或不是非空 str")
        anchor: Any = c.get("anchor_time")
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
            raise SchemaError(f"{path}: 第{i}条候选 anchor_time 缺失或非数值")
        if not isinstance(c.get("crop", ""), str):
            raise SchemaError(f"{path}: 第{i}条候选 crop 不是 str")
        if not isinstance(c.get("clip", ""), str):
            raise SchemaError(f"{path}: 第{i}条候选 clip 不是 str")
        if c.get("team_guess") is not None and c["team_guess"] not in TEAM_GUESS_VALUES:
            raise SchemaError(f"{path}: 第{i}条候选 team_guess 非法: {c['team_guess']!r}")
        if c.get("number_guess") is not None and not isinstance(c["number_guess"], dict):
            raise SchemaError(f"{path}: 第{i}条候选 number_guess 不是对象")
    return candidates


def match_players_by_number(
    players: list[Player], number: str | None, color: str | None
) -> list[Player]:
    """号码匹配名单：号码唯一优先，颜色作消解提示（容忍 K3 颜色误读）。

    用户 2026-08-09 确认同场号码极少重复，故：
    - 号码在名单中唯一 → 直接命中（不看颜色：颜色误读不该挡掉正确号码）；
    - 同号多人 → 用颜色进一步过滤，滤后唯一则命中、滤后为空或仍多个 → 歧义
      （返回全部候选，调用方判歧义）；
    - "2" 不会误中 "黑21"（数字边界防子串误配）。

    Args:
        players: 球员名单（--players 或已有 roster 的 players）。
        number: K3 读出的号码字符串；None 直接无匹配。
        color: K3 读出的颜色（黑/白/蓝/其他），仅作同号消解提示。

    Returns:
        命中的 Player 列表（0/1/N 个；N>1 表示歧义）。
    """
    if not number:
        return []
    pat: re.Pattern[str] = re.compile(rf"(?<!\d){re.escape(number)}(?!\d)")
    by_num: list[Player] = [p for p in players if pat.search(p.tag)]
    if len(by_num) <= 1:
        return by_num
    if color and color != "其他":
        by_col: list[Player] = [p for p in by_num if color in p.tag]
        if by_col:
            return by_col
    return by_num


def _edit_distance_le1(a: str, b: str) -> bool:
    """两字符串是否相等或只差 1 个字符（增/删/改）——K3 印名误读容差（张二≈张三）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def match_players_by_name(players: list[Player], name_text: str | None) -> list[Player]:
    """球衣印名匹配名单：精确或差 1 字符（K3 误读容差），只看非空 name。

    Args:
        players: 球员名单。
        name_text: K3 读出的印名文本；None/空串无匹配。

    Returns:
        命中的 Player 列表（0/1/N 个；N>1 表示歧义）。
    """
    if not name_text or not name_text.strip():
        return []
    t: str = name_text.strip()
    return [p for p in players if p.name and _edit_distance_le1(t, p.name)]


@dataclass(frozen=True, slots=True)
class PhotoGuess:
    """单球照片库识别预填（注入页面条目的字段；spec: docs/photo-roster/spec.md T5）。

    margin 刻意不进本结构/不进 JS：单号码库命中 margin=+inf，json.dumps 落
    ``Infinity``，页面 JSON.parse 无法解析（读端 Python json 可解析，校验复用
    photo_match_scorers.validate_matches_payload）。
    """

    number: str  # 去零号码（照片库匹配主键口径）
    score: float  # top-1 余弦得分（冲突角标展示用）
    tag: str  # 名单球员 tag 或占位 tag（主队名<号码>）


def resolve_photo_guesses(
    matches: dict[str, MatchEntry], players: list[Player], home: str
) -> tuple[dict[str, PhotoGuess], list[Player]]:
    """照片命中号码 → 名单 tag；名单缺号 → 占位 Player（主队名<号>，team=主队名）。

    号码查名单复用 match_players_by_number 的数字边界口径（颜色传 None：照片
    识别不给颜色提示）。同号多人 → WARNING 跳过该球不预填（交人裁判，与读号
    歧义同口径）。占位条目随 players 名单注入页面，不得依赖 teamOfTag 前缀推队
    （``主队7`` 不以黑/蓝/白开头，前缀推队会误归便服——spec 写死）。

    Args:
        matches: photo_match_scorers.validate_matches_payload 校验产物（key → 命中）。
        players: 本页球员名单（--players/--players-file/已有 roster 合并后）。
        home: 主队名（team_config 注入；占位 tag 与 team 用它）。

    Returns:
        (key → PhotoGuess, 需追加注入名单的占位 Player 列表)；同号码多球只占位一份。
    """
    guesses: dict[str, PhotoGuess] = {}
    placeholders: list[Player] = []
    placeholder_tags: set[str] = set()
    for key in sorted(matches):
        entry: MatchEntry = matches[key]
        found: list[Player] = match_players_by_number(players, entry.number, None)
        if len(found) == 1:
            tag: str = found[0].tag
        elif not found:
            tag = f"{home}{entry.number}"
            if tag not in placeholder_tags and all(p.tag != tag for p in players):
                placeholders.append(Player(tag=tag, name="", team=home))
                placeholder_tags.add(tag)
        else:
            logger.warning(
                "照片命中号码 %s 在名单中同号多人，不预填（交人裁判）: %s", entry.number, key
            )
            continue
        guesses[key] = PhotoGuess(number=entry.number, score=entry.score, tag=tag)
    return guesses, placeholders


def _confirmed_goals(data: Any, goals_path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """从 goals.json 数据中取 confirmed 记录（缺 file/anchor_time 显式失败）。

    Args:
        data: read_json 读出的原始 JSON。
        goals_path: 文件路径（仅用于错误信息）。

    Returns:
        confirmed 记录列表（保留原始 dict）。

    Raises:
        SchemaError: 顶层非对象 / goals 非列表 / confirmed 记录缺字段或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{goals_path}: 顶层必须是对象，实际 {type(data).__name__}")
    goals: Any = data.get("goals")
    if not isinstance(goals, list):
        raise SchemaError(f"{goals_path}: 缺 goals 列表或类型错误")
    confirmed: list[dict[str, Any]] = []
    for i, g in enumerate(goals):
        if not isinstance(g, dict):
            raise SchemaError(f"{goals_path}: 第{i}条记录不是对象")
        if g.get("status") != "confirmed":
            continue
        if not isinstance(g.get("file"), str) or not g["file"]:
            raise SchemaError(f"{goals_path}: 第{i}条(confirmed) file 缺失或不是非空 str")
        anchor: Any = g.get("anchor_time")
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
            raise SchemaError(f"{goals_path}: 第{i}条(confirmed) anchor_time 缺失或非数值")
        confirmed.append(g)
    return confirmed


def _validate_events(data: Any, path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """校验 events_index.json 结构（只查本页用到的字段）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        events 记录列表。

    Raises:
        SchemaError: 顶层非对象 / events 非列表 / 记录缺 src_file/anchor_t0/clip 或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    events: Any = data.get("events")
    if not isinstance(events, list):
        raise SchemaError(f"{path}: 缺 events 列表或类型错误")
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise SchemaError(f"{path}: 第{i}条事件不是对象")
        if not isinstance(ev.get("src_file"), str) or not ev["src_file"]:
            raise SchemaError(f"{path}: 第{i}条事件 src_file 缺失或不是非空 str")
        t0: Any = ev.get("anchor_t0")
        if isinstance(t0, bool) or not isinstance(t0, (int, float)):
            raise SchemaError(f"{path}: 第{i}条事件 anchor_t0 缺失或非数值")
        if not isinstance(ev.get("clip"), str) or not ev["clip"]:
            raise SchemaError(f"{path}: 第{i}条事件 clip 缺失或不是非空 str")
        if not isinstance(ev.get("clip_wide", ""), str):
            raise SchemaError(f"{path}: 第{i}条事件 clip_wide 不是 str")
    return events


def match_clip(
    events: list[dict[str, Any]],
    file: str,
    anchor_time: float,
    index_dir: str,
    out_dir: str,
) -> str:
    """按 src_file 相同且 |anchor_t0−anchor_time|≤4s 匹配审核片段（spec T4）。

    多个事件命中取时间差最小者；**优先取 clip_wide 全景**（认人需看清全身，
    筐区裁剪看不清人，2026-08-01 用户反馈），无 clip_wide 回退 clip。
    返回相对 scorer.html 所在目录的正斜杠相对路径（events_index 里的
    clip 可能是 Windows 反斜杠，先归一）。

    Args:
        events: _validate_events 校验后的事件列表。
        file: 进球记录的视频文件名。
        anchor_time: 进球锚点（秒）。
        index_dir: events_index.json 所在目录（clip 相对它解析）。
        out_dir: scorer.html 输出目录（返回路径相对它）。

    Returns:
        相对路径串（如 ``../review_v3/clips/x_wide.mp4``）；无匹配返回空串。
    """
    best: dict[str, Any] | None = None
    best_dt: float = CLIP_MATCH_MAX_DT_SEC
    for ev in events:
        if ev["src_file"] != file:
            continue
        dt: float = abs(float(ev["anchor_t0"]) - anchor_time)
        if dt <= best_dt:
            best = ev
            best_dt = dt
    if best is None:
        return ""
    clip_val: str = str(best.get("clip_wide") or best["clip"])
    clip_norm: str = clip_val.replace("\\", "/")
    rel: str = os.path.relpath(os.path.join(index_dir, clip_norm), out_dir)
    return rel.replace(os.sep, "/")


def _validate_clusters(data: Any, path: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """校验 scorer_clusters.json 结构（cluster_scorers 输出契约；rules.md §0.2）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        clusters 记录列表（保留原始 dict）。

    Raises:
        SchemaError: 顶层非对象 / 缺 clusters 列表 / 簇缺 cluster_id/keys/rep_crops
            或类型错 / cluster_id 重复。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    clusters: Any = data.get("clusters")
    if not isinstance(clusters, list):
        raise SchemaError(f"{path}: 缺 clusters 列表或类型错误")
    seen_ids: set[int] = set()
    for i, cl in enumerate(clusters):
        if not isinstance(cl, dict):
            raise SchemaError(f"{path}: 第{i}个簇不是对象")
        cid: Any = cl.get("cluster_id")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise SchemaError(f"{path}: 第{i}个簇 cluster_id 缺失或非 int")
        if cid in seen_ids:
            raise SchemaError(f"{path}: cluster_id 重复: {cid}")
        seen_ids.add(cid)
        keys: Any = cl.get("keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise SchemaError(f"{path}: 第{i}个簇 keys 缺失或非 str 列表")
        rep: Any = cl.get("rep_crops")
        if not isinstance(rep, list) or not all(isinstance(r, str) for r in rep):
            raise SchemaError(f"{path}: 第{i}个簇 rep_crops 缺失或非 str 列表")
    return clusters


def build_cluster_map(clusters: list[dict[str, Any]], candidate_keys: set[str]) -> dict[str, int]:
    """key → cluster_id 映射；引用 candidates 之外的 key 记 WARNING 跳过（不炸）。

    同一 key 出现在多个簇（聚类契约本应互斥）取首个并记 WARNING，容忍不炸。

    Args:
        clusters: _validate_clusters 校验后的簇列表。
        candidate_keys: 本页 candidates 的 key 集合。

    Returns:
        key → cluster_id（只含 candidates 里存在的 key）。
    """
    mapping: dict[str, int] = {}
    for cl in clusters:
        cid: int = cl["cluster_id"]
        for key in cl["keys"]:
            if key not in candidate_keys:
                logger.warning("簇 %d 引用的 key 不在 candidates 里，跳过: %s", cid, key)
                continue
            if key in mapping:
                logger.warning("key 同时属于簇 %d 与簇 %d，取前者: %s", mapping[key], cid, key)
                continue
            mapping[key] = cid
    return mapping


def build_page_clusters(
    clusters: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """簇 → 页面簇区数据：keys 过滤到本页条目（confirmed 球），过滤后为空的簇剔除。

    rep_crops 原样透传（文件名相对 candidates 同目录，与逐球区 crop 引用口径一致；
    --clusters 必须与 --scorers 同目录由 CLI 层保证）。

    Args:
        clusters: _validate_clusters 校验后的簇列表。
        entries: build_entries 产出的页面条目。

    Returns:
        页面簇区数据列表（cluster_id/keys/rep_crops），保持入参簇序。
    """
    entry_keys: set[str] = {e["key"] for e in entries}
    page: list[dict[str, Any]] = []
    for cl in clusters:
        keys: list[str] = [k for k in cl["keys"] if k in entry_keys]
        dropped: int = len(cl["keys"]) - len(keys)
        if dropped:
            logger.info(
                "簇 %d 有 %d 个 key 不在本页 confirmed 球里（其他批次/非 confirmed，页面不显示）",
                cl["cluster_id"],
                dropped,
            )
        if not keys:
            continue
        page.append(
            {
                "cluster_id": cl["cluster_id"],
                "keys": keys,
                "rep_crops": list(cl["rep_crops"]),
            }
        )
    return page


def _validate_track_links(data: Any, path: str) -> dict[str, Any]:  # noqa: ANN401
    """校验 track_links.json 结构（propagate_scorers 输出契约 track-v1；rules.md §0.2）。

    只查本页用到的字段：顶层 version/per_file、每文件 tracks/unlinked、每条 track 的
    track_id/keys/mixed/span。track 里引用本页 confirmed 球之外的 key 不在此炸
    （跨批次 key 是常态，build_track_map 记 INFO 跳过）。

    Args:
        data: read_json 读出的原始 JSON。
        path: 文件路径（仅用于错误信息）。

    Returns:
        per_file 段（fid → {tracks, unlinked}，保留原始 dict）。

    Raises:
        SchemaError: 顶层非对象 / version 不符 / 缺 per_file / track 结构坏或类型错。
    """
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    if data.get("version") != TRACK_LINKS_VERSION:
        raise SchemaError(
            f"{path}: version 须为 {TRACK_LINKS_VERSION}，实际 {data.get('version')!r}"
        )
    per_file: Any = data.get("per_file")
    if not isinstance(per_file, dict):
        raise SchemaError(f"{path}: 缺 per_file 对象或类型错误")
    for fid, fr in per_file.items():
        if not isinstance(fr, dict):
            raise SchemaError(f"{path}: per_file[{fid!r}] 不是对象")
        tracks: Any = fr.get("tracks")
        if not isinstance(tracks, list):
            raise SchemaError(f"{path}: per_file[{fid!r}] 缺 tracks 列表或类型错误")
        for i, t in enumerate(tracks):
            if not isinstance(t, dict):
                raise SchemaError(f"{path}: per_file[{fid!r}] 第{i}条轨迹不是对象")
            tid: Any = t.get("track_id")
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise SchemaError(f"{path}: per_file[{fid!r}] 第{i}条轨迹 track_id 缺失或非 int")
            keys: Any = t.get("keys")
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise SchemaError(f"{path}: per_file[{fid!r}] 第{i}条轨迹 keys 缺失或非 str 列表")
            if not isinstance(t.get("mixed"), bool):
                raise SchemaError(f"{path}: per_file[{fid!r}] 第{i}条轨迹 mixed 缺失或非 bool")
            span: Any = t.get("span")
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in span)
            ):
                raise SchemaError(
                    f"{path}: per_file[{fid!r}] 第{i}条轨迹 span 缺失或不是二元 int 列表"
                )
        unlinked: Any = fr.get("unlinked")
        if not isinstance(unlinked, list) or not all(isinstance(k, str) for k in unlinked):
            raise SchemaError(f"{path}: per_file[{fid!r}] unlinked 缺失或非 str 列表")
    return per_file


def build_track_map(per_file: dict[str, Any], page_keys: set[str]) -> dict[str, int]:
    """track_links → key 反查 track_id 映射（页面条目注入与"轨迹#N"显示用）。

    track 里引用本页 confirmed 球之外的 key 记 INFO 跳过（跨批次 key 是常态——
    track_links 含同 scorers 目录其他批次的球）；同一 key 挂多条轨迹（契约本应
    互斥）取首个并记 WARNING，容忍不炸。unlinked 不进映射（归不上轨迹的球页面
    track_id=None，不显示也不参与传播）。轨迹号按文件内编号，页面按
    file+track_id 判同轨（轨迹不跨文件，spec 写死）。

    Args:
        per_file: _validate_track_links 校验后的 per_file 段。
        page_keys: 本页 confirmed 球的 key 集合。

    Returns:
        key → track_id（只含本页 key）。
    """
    mapping: dict[str, int] = {}
    for fid, fr in per_file.items():
        for t in fr["tracks"]:
            tid: int = t["track_id"]
            for key in t["keys"]:
                if key not in page_keys:
                    logger.info(
                        "轨迹#%d(%s) 引用的 key 不在本页 confirmed 球里，跳过: %s", tid, fid, key
                    )
                    continue
                if key in mapping:
                    logger.warning(
                        "key 同时挂轨迹#%d 与轨迹#%d(%s)，取前者: %s", mapping[key], tid, fid, key
                    )
                    continue
                mapping[key] = tid
    return mapping


def build_entries(
    confirmed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]] | None,
    index_dir: str,
    out_dir: str,
    players: list[Player] | None = None,
    cluster_map: dict[str, int] | None = None,
    photo_guesses: dict[str, PhotoGuess] | None = None,
    track_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """组装页面条目：每条 = 一个 confirmed 球（按 file+anchor 排序）。

    以 goals.json 的 confirmed 球为全集，按 key 关联 candidates 取裁图/
    team_guess/number_guess/SKIP 状态；无候选记录（防御）按 SKIP 列出。视频优先级：
    candidates 的 "clip"（按进球锚点现切的预览片段，与裁图同球同时刻）＞
    events_index 的 clip_wide 匹配（仅作无预览片段时的兜底）。预填优先级
    （spec: docs/photo-roster/spec.md T5 写死）：读号命中（号码唯一匹配，同号
    歧义用印名消解）＞ 照片命中 ＞ 印名匹配 ＞ 空白；读号同号歧义维持不预填
    （prefill_note="ambiguous"），照片候选仍随条目 photo_guess 上页供角标切换；
    颜色 team_guess 仅作页面展示不参与 prefill_tag。给了 cluster_map 则每条追加
    cluster_id（不在任何簇/unclustered → None）；给了 photo_guesses 则每条追加
    photo_guess（无命中 → None，页面不出角标）；给了 track_map 则每条追加
    track_id（不在任何轨迹/unlinked → None，页面不显示"轨迹#N"也不参与传播）。

    Args:
        confirmed: goals.json 的 confirmed 记录。
        candidates: scorer_candidates.json 的候选记录。
        events: 事件列表；None 表示无 --index（无兜底视频）。
        index_dir: events_index.json 所在目录。
        out_dir: scorer.html 输出目录。
        players: 球员名单（号码/印名匹配用）；None/空列表则只做颜色展示与照片预填。
        cluster_map: key → cluster_id（build_cluster_map 产物）；None 表示无
            --clusters，条目 cluster_id 全为 None（页面不渲染簇区）。
        photo_guesses: key → PhotoGuess（resolve_photo_guesses 产物）；None 表示
            无 --photo-matches，条目 photo_guess 全为 None（页面行为与旧版一致）。
        track_map: key → track_id（build_track_map 产物，文件内编号）；None 表示
            无 --track-links，条目 track_id 全为 None（页面行为与现状完全一致）。

    Returns:
        页面条目列表（key/file/anchor_time/status/reason/crop/team_guess/clip/
        number_guess/prefill_tag/prefill_note/cluster_id/photo_guess/track_id）。
    """
    players = players or []
    cluster_map = cluster_map or {}
    photo_guesses = photo_guesses or {}
    track_map = track_map or {}
    by_key: dict[str, dict[str, Any]] = {c["key"]: c for c in candidates}
    entries: list[dict[str, Any]] = []
    ordered = sorted(confirmed, key=lambda g: (g["file"], float(g["anchor_time"])))
    for g in ordered:
        file: str = g["file"]
        anchor: float = float(g["anchor_time"])
        key: str = format_key(file, anchor)
        cand: dict[str, Any] | None = by_key.get(key)
        if cand is None:
            logger.warning("confirmed 球无定位候选记录: %s（按 SKIP 列出）", key)
        status: str = cand["status"] if cand is not None else STATUS_SKIP
        reason: str = str(cand.get("reason", "")) if cand is not None else "no_candidate"
        crop: str = str(cand.get("crop", "")) if cand is not None else ""
        team_guess: str | None = cand.get("team_guess") if cand is not None else None
        # 预览片段（candidates clip，与裁图同锚点）优先；无则回退事件 clip_wide 匹配
        clip: str = str(cand.get("clip", "")) if cand is not None else ""
        if not clip and events is not None:
            clip = match_clip(events, file, anchor, index_dir, out_dir)
        # 预填：读号（号码唯一；同号歧义用印名消解）＞ 照片 ＞ 印名兜底；读号歧义不预填
        number_guess: dict[str, Any] | None = cand.get("number_guess") if cand is not None else None
        pg: PhotoGuess | None = photo_guesses.get(key)
        prefill_tag: str = ""
        prefill_note: str = ""
        num_matches: list[Player] = []
        name_matches: list[Player] = []
        if isinstance(number_guess, dict):
            num_matches = match_players_by_number(
                players, number_guess.get("number"), number_guess.get("color")
            )
            name_matches = match_players_by_name(players, number_guess.get("name_text"))
            if len(num_matches) > 1 and name_matches:
                narrowed: list[Player] = [p for p in num_matches if p in name_matches]
                if narrowed:
                    num_matches = narrowed
        if len(num_matches) == 1:
            prefill_tag = num_matches[0].tag
        elif len(num_matches) > 1:
            prefill_note = "ambiguous"
        elif pg is not None:
            prefill_tag = pg.tag
            prefill_note = "photo"
        elif len(name_matches) == 1:
            prefill_tag = name_matches[0].tag
        elif len(name_matches) > 1:
            prefill_note = "ambiguous"
        entries.append(
            {
                "key": key,
                "file": file,
                "anchor_time": anchor,
                "status": status,
                "reason": reason,
                "crop": crop,
                "team_guess": team_guess,
                "clip": clip,
                "number_guess": number_guess,
                "prefill_tag": prefill_tag,
                "prefill_note": prefill_note,
                "cluster_id": cluster_map.get(key),
                "photo_guess": (
                    {"number": pg.number, "score": pg.score, "tag": pg.tag}
                    if pg is not None
                    else None
                ),
                "track_id": track_map.get(key),
            }
        )
    return entries


def build_html(
    entries: list[dict[str, Any]],
    players: list[Player],
    session: str,
    existing_assignments: dict[str, str],
    existing_players: dict[str, Player],
    opp: str,
    home: str,
    clusters: list[dict[str, Any]] | None = None,
) -> str:
    """把条目/名单/已有归属/簇数据渲染为自包含确认页 HTML。

    Args:
        entries: build_entries 产出的页面条目（原样内联）。
        players: 球员按钮名单（--players 或已有 roster 的 players）。
        session: 场次名（标题、localStorage 键、导出文件名后缀）。
        existing_assignments: 已有 roster 的 assignments（页面预填底色）。
        existing_players: 已有 roster 的 tag → Player（自动补录时沿用 name/team）。
        opp: 对手队名（注入 JS ``const OPP``，opponent_of 产物）。
        home: 主队名（注入 JS ``const HOME``，team_config 产物）。
        clusters: build_page_clusters 产出的簇区数据；None/空列表不渲染簇区
            （无 --clusters 时页面行为与旧版一致）。

    Returns:
        scorer.html 全文。
    """
    page_players: list[Player] = list(players)
    if OPPONENT_TAG not in {p.tag for p in page_players}:
        # 伪球员"对手"恒注入（opponent-filter T2）：随 OPP 队行渲染出按钮，
        # 逐球/整簇一键归属；导出时零引用由 JS 侧剔除，不进 roster.players
        page_players.append(Player(tag=OPPONENT_TAG, name="", team=opp))
    players_json = json.dumps(
        [{"tag": p.tag, "name": p.name, "team": p.team} for p in page_players],
        ensure_ascii=False,
    )
    explayers_json = json.dumps(
        {tag: {"name": p.name, "team": p.team} for tag, p in existing_players.items()},
        ensure_ascii=False,
    )
    return (
        _HTML.replace("__ITEMS__", json.dumps(entries, ensure_ascii=False))
        .replace("__PLAYERS__", players_json)
        .replace("__EXISTING__", json.dumps(existing_assignments, ensure_ascii=False))
        .replace("__EXPLAYERS__", explayers_json)
        .replace("__CLUSTERS__", json.dumps(clusters or [], ensure_ascii=False))
        .replace("__SESSION__", session)
        .replace("__OPP__", json.dumps(opp, ensure_ascii=False))
        .replace("__OPP_TAG__", json.dumps(OPPONENT_TAG, ensure_ascii=False))
        .replace("__HOME__", json.dumps(home, ensure_ascii=False))
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description="生成认人确认页 scorer.html（spec T4）")
    parser.add_argument("--scorers", required=True, type=Path, help="scorer_candidates.json 路径")
    parser.add_argument("--goals", required=True, type=Path, help="goals.json 路径")
    parser.add_argument("--session", default="", help="场次名（缺省取 candidates 里的 session）")
    parser.add_argument("--index", default="", help="events_index.json 路径（可选，引用审核片段）")
    parser.add_argument(
        "--players",
        default="",
        help='球员名单，如 "黑21=张三,白-李四=李四"（可选，与 --players-file 互斥）',
    )
    parser.add_argument(
        "--players-file",
        type=Path,
        default=None,
        help="球员名单 JSON 文件（与 roster.players 同构的数组；可选，与 --players 互斥）",
    )
    parser.add_argument("--roster-existing", default="", help="已有 roster.json（可选，合并预填）")
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="scorer_clusters.json 路径（可选，簇级确认；必须与 --scorers 同目录）",
    )
    parser.add_argument(
        "--photo-matches",
        type=Path,
        default=None,
        help="photo_matches.json 路径（可选，照片库识别预填；必须与 --scorers 同目录）",
    )
    parser.add_argument(
        "--track-links",
        type=Path,
        default=None,
        help="track_links.json 路径（可选，轨迹传播预填；必须与 --scorers 同目录）",
    )
    ns = parser.parse_args(argv)
    if ns.players and ns.players_file is not None:
        parser.error("--players 与 --players-file 互斥：名单只给一个来源（防双源不一致）")
    if ns.clusters is not None and ns.clusters.resolve().parent != ns.scorers.resolve().parent:
        parser.error("--clusters 必须与 --scorers 同目录（rep_crops 与裁图同目录相对引用口径）")
    if ns.photo_matches is not None and (
        ns.photo_matches.resolve().parent != ns.scorers.resolve().parent
    ):
        parser.error("--photo-matches 必须与 --scorers 同目录（与 --clusters 校验同口径）")
    if ns.track_links is not None and (
        ns.track_links.resolve().parent != ns.scorers.resolve().parent
    ):
        parser.error("--track-links 必须与 --scorers 同目录（与 --clusters 校验同口径）")
    return ns


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功，1=参数/数据/合并冲突失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        scorers_path: Path = args.scorers
        cand_data: Any = read_json(scorers_path, what="scorer_candidates.json")
        candidates: list[dict[str, Any]] = _validate_candidates(cand_data, str(scorers_path))
        session: str = args.session or (
            cand_data.get("session", "") if isinstance(cand_data, dict) else ""
        )
        if not session:
            logger.error("缺 --session 且 candidates 无 session 字段")
            return 1
        # 会话级队名配置：scorers 固定落 work/<场次>/scorers/，父父级即会话目录；
        # 缺失/损坏回退默认值（普通用户可完全不配置，team_config 容错口径）
        cfg: TeamConfig = load_team_config(scorers_path.resolve().parent.parent)
        home: str = cfg.team_name
        opp: str = cfg.opponent or opponent_of(session)  # 配置优先，缺省按场次 ID 派生

        goals_data: Any = read_json(args.goals, what="goals.json")
        confirmed: list[dict[str, Any]] = _confirmed_goals(goals_data, str(args.goals))

        events: list[dict[str, Any]] | None = None
        index_dir: str = ""
        if args.index:
            idx_data: Any = read_json(args.index, what="events_index.json")
            events = _validate_events(idx_data, args.index)
            index_dir = os.path.dirname(os.path.abspath(args.index))

        out_dir: str = str(scorers_path.resolve().parent)

        clusters_raw: list[dict[str, Any]] | None = None
        cluster_map: dict[str, int] | None = None
        if args.clusters is not None:
            cl_data: Any = read_json(args.clusters, what="scorer_clusters.json")
            clusters_raw = _validate_clusters(cl_data, str(args.clusters))
            cluster_map = build_cluster_map(clusters_raw, {c["key"] for c in candidates})

        players: list[Player] = (
            load_players_file(args.players_file)
            if args.players_file is not None
            else parse_players(args.players, opp, home)
        )
        existing_assignments: dict[str, str] = {}
        existing_players: dict[str, Player] = {}
        if args.roster_existing:
            roster_data: Any = read_json(args.roster_existing, what="roster.json")
            roster = validate_roster(roster_data, args.roster_existing)
            existing_assignments = merge_assignments(
                roster.assignments, {}, what=str(args.roster_existing)
            )
            existing_players = {p.tag: p for p in roster.players}
            if not players:
                # 未给新名单：沿用已有 roster 的 players 作按钮名单
                players = list(roster.players)
            else:
                # players 以新名单为准；已有归属引用了名单外 tag → WARNING
                known: set[str] = {p.tag for p in players}
                for tag in sorted(set(roster.assignments.values()) - known):
                    logger.warning(
                        "已有 roster 归属的 tag 不在新名单中（导出时将自动补录）: %s", tag
                    )

        photo_guesses: dict[str, PhotoGuess] | None = None
        if args.photo_matches is not None:
            # 延迟 import：photo_match_scorers 依赖链带 numpy/sklearn，只在启用
            # 照片预填时加载，无 --photo-matches 行为与开销零变化（兼容性承诺）
            from photo_match_scorers import validate_matches_payload

            pm_data: Any = read_json(args.photo_matches, what="photo_matches.json")
            pm_matches: dict[str, MatchEntry] = validate_matches_payload(
                pm_data, str(args.photo_matches)
            )
            photo_guesses, placeholders = resolve_photo_guesses(pm_matches, players, home)
            if placeholders:
                # 名单缺号占位条目随 players 注入页面（spec 写死，不靠 teamOfTag 推队）
                players = [*players, *placeholders]
                logger.info(
                    "照片命中号码不在名单，注入占位条目: %s",
                    ", ".join(p.tag for p in placeholders),
                )
            logger.info("照片预填: %d 球命中 ← %s", len(photo_guesses), args.photo_matches)

        track_map: dict[str, int] | None = None
        if args.track_links is not None:
            tl_data: Any = read_json(args.track_links, what="track_links.json")
            per_file: dict[str, Any] = _validate_track_links(tl_data, str(args.track_links))
            # 页面条目全集 = confirmed 球（spec §页面：跨批次 key 跳过不炸）
            page_keys: set[str] = {
                format_key(g["file"], float(g["anchor_time"])) for g in confirmed
            }
            track_map = build_track_map(per_file, page_keys)
            logger.info("轨迹传播: %d 球挂上轨迹 ← %s", len(track_map), args.track_links)

        entries: list[dict[str, Any]] = build_entries(
            confirmed,
            candidates,
            events,
            index_dir,
            out_dir,
            players,
            cluster_map=cluster_map,
            photo_guesses=photo_guesses,
            track_map=track_map,
        )

        page_clusters: list[dict[str, Any]] | None = None
        if clusters_raw is not None:
            page_clusters = build_page_clusters(clusters_raw, entries)

        html: str = build_html(
            entries,
            players,
            session,
            existing_assignments,
            existing_players,
            opp,
            home,
            clusters=page_clusters,
        )
        out_path: Path = scorers_path.resolve().parent / "scorer.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        n_skip: int = sum(1 for e in entries if e["status"] == STATUS_SKIP)
        n_clip: int = sum(1 for e in entries if e["clip"])
        logger.info(
            "确认页 %d 球（SKIP %d，带片段 %d，球员 %d，簇 %d）-> %s（浏览器打开即可认人）",
            len(entries),
            n_skip,
            n_clip,
            len(players),
            len(page_clusters or []),
            out_path,
        )
        return 0
    except BasketballPipelineError as exc:
        logger.error("管线失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
