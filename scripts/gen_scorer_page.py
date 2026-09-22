#!/usr/bin/env python3
"""生成认人确认页 scorer.html：用户逐球确认进球者归属（spec: docs/scorer/spec.md T4）。

读取 crop_scorers 产出的 scorer_candidates.json（每球一条：key/裁图/clip 预览
片段/team_guess/SKIP 状态）+ goals.json（confirmed 球为页面条目全集），在同目录
生成自包含 scorer.html（数据内联、裁图/视频相对路径、按键+按钮、localStorage
进度、一键导出 roster.json——点击后 POST 回向导服务器自动落盘到 work/<场次>/
roster.json，网络失败时回退 blob 下载）。用户浏览器打开即可：看裁图与预览片段，
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
from roster import OPPONENT_TAG, Player, format_key, opponent_of, player_from_dict, validate_roster
from team_config import DEFAULT_OPPONENT, DEFAULT_TEAM_NAME, TeamConfig, load_team_config

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

TEAM_HOME_DEFAULT: str = DEFAULT_TEAM_NAME  # 未配置 team_config 时的主队名兜底
TEAM_CASUAL: str = "便服"
# 对手队名不再硬编码：配置 opponent > opponent_of(session) 从场次 ID 后缀派生
# （黑/蓝球衣=对手队；2026-08-09 用户定前缀映射，2026-08-15 队名会话化）
OPPONENT_FALLBACK: str = DEFAULT_OPPONENT  # 无后缀且未配置 opponent 时的兜底
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

# 认人确认页键位映射（spec: docs/scorer-ux-redesign/spec.md §4.2）。
# 选人序列 27 键；功能保留键显式排除，生成期断言二者不相交。
PLAYER_KEYS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Q",
    "W",
    "E",
    "R",
    "T",
    "Y",
    "U",
    "I",
    "A",
    "D",
    "F",
    "H",
    "J",
    "K",
    "L",
    "C",
    "V",
    "B",
)
FUNCTION_KEYS: set[str] = {"O", "P", "S", "N", "M", "X", "Z", "G", "/"}
if set(PLAYER_KEYS) & FUNCTION_KEYS:
    raise ValueError("选人序列与功能保留键冲突")

# 簇代表图墙最大展示数（spec §4.1 大裁图墙 ≤8）
CLUSTER_CROP_WALL_MAX: int = 8


def build_keymap(players: list[Player]) -> list[dict[str, Any]]:
    """把名单球员按顺序映射到 27 个选人键（spec §4.2）。

    前 27 名球员分配 ``PLAYER_KEYS``；超出的球员无键位（角标为空，仍可鼠标点）。
    名单 ≤11 人时 ``E`` 未被占用，页面内作为 ``Enter`` 别名使用（spec 兼容口径）。

    Args:
        players: 球员名单（按页面显示顺序）。

    Returns:
        每个球员对应的键位描述列表，元素含 ``tag``/``name``/``team``/``key``；
        无键位时 ``key`` 为空串。
    """
    result: list[dict[str, Any]] = []
    for i, p in enumerate(players):
        key = PLAYER_KEYS[i] if i < len(PLAYER_KEYS) else ""
        result.append({"tag": p.tag, "name": p.name, "team": p.team, "key": key})
    return result


def cluster_sort_key(
    keys: list[str],
    entry_by_key: dict[str, dict[str, Any]],
) -> tuple[int, float]:
    """簇队列排序键：球数降序，同数按首球锚点时间升序（spec §4.1）。

    Args:
        keys: 簇内条目 key 列表。
        entry_by_key: key → 页面条目的索引（必须包含 keys）。

    Returns:
        可用于 ``sorted(..., key=...)`` 的元组 ``(-球数, 首球时间)``。

    Raises:
        KeyError: keys 中引用不在索引里的 key（防御，调用方应保证）。
    """
    first_anchor = min(float(entry_by_key[k]["anchor_time"]) for k in keys)
    return (-len(keys), first_anchor)


_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>认人确认 __SESSION__</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; margin: 16px; }
#bar { position: sticky; top: 0; background: #111; padding: 8px 0; z-index: 9; }
button { font-size: 16px; padding: 8px 14px; margin: 3px; border-radius: 6px;
         border: 0; cursor: pointer; }
button.sel { outline: 3px solid #fc3; }
.team-opp { background: #222; color: #fff; border: 1px solid #666; }
.team-home { background: #eee; color: #111; }
.team-casual { background: #777; color: #fff; }
.nav { background: #444; color: #fff; }
#skip { background: #7a5c00; color: #fff; }
#nogoal { background: #7a2c2c; color: #fff; }
#accept { background: #2c9e4b; color: #fff; }
#export { background: #8a6d00; color: #fff; }
#undo { background: #3a5a8a; color: #fff; }
.badge { color: #fc3; }
small { color: #999; }
#stage { color: #fc3; font-size: 18px; margin: 8px 0; }
#rosterTable { display: none; background: #1a1a1a; border: 1px solid #333;
               border-radius: 8px; padding: 8px; margin: 8px 0; }
#rosterTable table { width: 100%; border-collapse: collapse; }
#rosterTable th, #rosterTable td { padding: 6px; border-bottom: 1px solid #333;
                                   text-align: left; }
#rosterTable input { background: #222; color: #eee; border: 1px solid #555;
                     padding: 4px; font-size: 14px; }
#rosterTable select { background: #222; color: #eee; border: 1px solid #555;
                      padding: 4px; font-size: 14px; }
#oppInput { background: #222; color: #eee; border: 1px solid #555; padding: 4px;
            font-size: 14px; width: 8em; }
.keycap { font-size: 10px; color: #fc3; margin-left: 4px; }
.hintbar { color: #aaa; font-size: 14px; margin: 6px 0; line-height: 1.6; }
#clusterView, #ballView { display: none; }
#clusterCrops { display: flex; flex-wrap: wrap; gap: 6px; }
#clusterCrops img { height: 22vh; width: auto; background: #000; border-radius: 4px; }
#clusterCrops img.eject-sel { outline: 3px solid #fc3; }
#clusterVideoWrap { margin: 8px 0; }
#clusterVideo { height: 34vh; width: auto; background: #000; }
.playerbtn { position: relative; font-size: 16px; }
.playerbtn .keycap { position: absolute; top: -4px; right: -4px; background: #fc3;
                     color: #000; border-radius: 4px; padding: 1px 4px;
                     font-weight: bold; }
#review { display: flex; align-items: flex-start; gap: 8px; }
#review #crop { height: 68vh; width: auto; background: #000; }
#review video { height: 68vh; width: auto; background: #000; }
#review #crop:hover {
  position: fixed; z-index: 99; left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  height: 92vh; width: auto; max-width: 96vw; max-height: 96vh;
  outline: 3px solid #fc3; background: #000;
}
#mergeTargets { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.merge-target { opacity: 0.6; cursor: pointer; border: 1px solid #444;
                border-radius: 6px; padding: 4px; }
.merge-target.sel { opacity: 1; outline: 3px solid #fc3; }
.merge-target img { height: 64px; width: auto; background: #000; }
#ballFilter { margin: 6px 0; }
.toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
         background: #333; color: #fff; padding: 10px 20px; border-radius: 8px;
         z-index: 99; display: none; }
</style>
</head>
<body>
<div id="bar">
  <span id="prog"></span>
  <button id="export">导出 roster.json</button>
  <button id="undo">撤销 (Z)</button>
  <button id="toggleroster">名单 (G)</button>
  <button id="modebtn">模式 /</button>
  <button id="acceptopp">接受全部对手预填</button>
  <label>对手队名：<input id="oppInput" type="text" placeholder="自动派生"></label>
  <br><small id="keyhint"></small>
</div>
<div id="rosterTable"></div>
<div id="stage"></div>
<div id="clusterView">
  <div id="clusterMeta"></div>
  <div id="clusterCrops"></div>
  <div id="clusterVideoWrap"><video id="clusterVideo" autoplay loop muted playsinline></video></div>
  <div id="clusterPlayers"></div>
  <div id="clusterHint" class="hintbar"></div>
  <div id="mergeTargets"></div>
</div>
<div id="ballView">
  <div id="ballInfo"></div>
  <div id="review">
    <img id="crop" alt="投篮者裁图">
    <video id="v" autoplay loop muted playsinline></video>
  </div>
  <div id="ballPlayers"></div>
  <div id="ballFilter"></div>
</div>
<div id="toast" class="toast"></div>
<script>
const ITEMS = __ITEMS__;
const PLAYERS = __PLAYERS__;
const KEYMAP = __KEYMAP__;
const PLAYER_KEYS = __PLAYER_KEYS__;
const EXISTING = __EXISTING__;
const EXPLAYERS = __EXPLAYERS__;
const CLUSTERS = __CLUSTERS__;
const SESSION = "__SESSION__";
let OPP = __OPP__;
const OPP_TAG = __OPP_TAG__;
const HOME = __HOME__;
const LSKEY = "scorer_" + SESSION;
const TOUCHKEY = LSKEY + "_touched";
const PROPKEY = LSKEY + "_propagate";
const CLSTATE_KEY = LSKEY + "_clusters";
const TEAMOVR_KEY = LSKEY + "_teamovr";
const NAMES_KEY = LSKEY + "_names";
const OPPNAME_KEY = LSKEY + "_oppname";
const POOLKEY = LSKEY + "_pool";
const NOGOAL = "不算进球";
const OPPONENT_TAG = __OPP_TAG__;

const itemByKey = {};
for (const it of ITEMS) itemByKey[it.key] = it;

let marks = {};
let touched = {};
let propagateAssign = {};
let clState = { merges: {}, clAssign: {}, collapsed: {}, deleted: {}, done: {} };
let teamOvr = {};
let nameOvr = {};
let oppNameOvr = "";
let pool = [];
let undoStack = [];
let mode = "cluster";
let curClusterIdx = 0;
let curBallIdx = 0;
let clusterOrder = [];
let mergeState = null;
let ejectState = null;
let rosterExpanded = false;
let ballFilter = "";

function loadJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || "null") || fallback; }
  catch (e) { return fallback; }
}
function loadStorage() {
  marks = loadJson(LSKEY, {});
  marks = Object.assign({}, EXISTING, marks);
  touched = loadJson(TOUCHKEY, {});
  propagateAssign = loadJson(PROPKEY, {});
  clState = loadJson(CLSTATE_KEY, {});
  if (!clState || typeof clState !== "object") clState = {};
  clState = {
    merges: clState.merges || {},
    clAssign: clState.clAssign || {},
    collapsed: clState.collapsed || {},
    deleted: clState.deleted || {},
    done: clState.done || {},
  };
  teamOvr = loadJson(TEAMOVR_KEY, {});
  nameOvr = loadJson(NAMES_KEY, {});
  oppNameOvr = (localStorage.getItem(OPPNAME_KEY) || "").trim();
  pool = loadJson(POOLKEY, []);
  if (!Array.isArray(pool)) pool = [];
  for (const p of PLAYERS) {
    if (teamOvr[p.tag] !== undefined) p.team = teamOvr[p.tag];
    if (nameOvr[p.tag] !== undefined) p.name = nameOvr[p.tag];
  }
  const baseOpp = OPP;
  if (oppNameOvr) OPP = oppNameOvr;
  for (const tag of Object.keys(teamOvr)) {
    if (teamOvr[tag] === baseOpp) teamOvr[tag] = OPP;
  }
  for (const p of PLAYERS) { if (p.team === baseOpp) p.team = OPP; }
  migrateOldPosReview();
}
function migrateOldPosReview() {
  let cleared = false;
  for (const k of Object.keys(localStorage)) {
    if (k === LSKEY + "_review" || k.startsWith(LSKEY + "_pos")) {
      localStorage.removeItem(k);
      cleared = true;
    }
  }
  if (cleared) toast("旧进度已迁移，位置从头开始");
}
function save() {
  localStorage.setItem(LSKEY, JSON.stringify(marks));
  localStorage.setItem(TOUCHKEY, JSON.stringify(touched));
  localStorage.setItem(PROPKEY, JSON.stringify(propagateAssign));
}
function saveClState() {
  localStorage.setItem(CLSTATE_KEY, JSON.stringify(clState));
}
function saveTeamOvr() {
  localStorage.setItem(TEAMOVR_KEY, JSON.stringify(teamOvr));
}
function saveNames() {
  localStorage.setItem(NAMES_KEY, JSON.stringify(nameOvr));
}
function saveOppName() {
  localStorage.setItem(OPPNAME_KEY, oppNameOvr);
}
function savePool() {
  localStorage.setItem(POOLKEY, JSON.stringify(pool));
}
function snapshotState() {
  return {
    marks: Object.assign({}, marks),
    touched: Object.assign({}, touched),
    propagateAssign: Object.assign({}, propagateAssign),
    clState: {
      merges: Object.assign({}, clState.merges),
      clAssign: Object.assign({}, clState.clAssign),
      collapsed: Object.assign({}, clState.collapsed),
      deleted: Object.assign({}, clState.deleted),
      done: Object.assign({}, clState.done),
    },
    teamOvr: Object.assign({}, teamOvr),
    nameOvr: Object.assign({}, nameOvr),
    oppNameOvr: oppNameOvr,
    pool: pool.slice(),
    mode: mode,
    curClusterIdx: curClusterIdx,
    curBallIdx: curBallIdx,
    clusterOrder: clusterOrder.slice(),
    ballFilter: ballFilter,
  };
}
function pushUndo(label) {
  if (undoStack.length >= 50) undoStack.shift();
  undoStack.push({ label, state: snapshotState() });
}
function undo() {
  if (!undoStack.length) { toast("无可撤销"); return; }
  const snap = undoStack.pop().state;
  marks = snap.marks;
  touched = snap.touched;
  propagateAssign = snap.propagateAssign;
  clState = snap.clState;
  teamOvr = snap.teamOvr;
  nameOvr = snap.nameOvr;
  oppNameOvr = snap.oppNameOvr;
  pool = snap.pool;
  mode = snap.mode;
  curClusterIdx = snap.curClusterIdx;
  curBallIdx = snap.curBallIdx;
  clusterOrder = snap.clusterOrder;
  ballFilter = snap.ballFilter;
  for (const p of PLAYERS) {
    p.team = teamOvr[p.tag] !== undefined ? teamOvr[p.tag] : p.team;
    p.name = nameOvr[p.tag] !== undefined ? nameOvr[p.tag] : p.name;
  }
  OPP = oppNameOvr || OPP;
  save();
  saveClState();
  saveTeamOvr();
  saveNames();
  saveOppName();
  savePool();
  toast("已撤销");
  render();
}
function teamOfTag(tag) {
  if (tag === OPPONENT_TAG) return OPP;
  if (tag.startsWith("黑") || tag.startsWith("蓝")) return OPP;
  if (tag.startsWith("白")) return HOME;
  return "便服";
}
function teamClass(team) {
  if (team === HOME) return "team-home";
  if (team === "便服") return "team-casual";
  return "team-opp";
}
function nDone() { return ITEMS.filter(it => marks[it.key]).length; }
function groupIdOf(cid) {
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
  return [...byGid.values()]
    .filter(g => !clState.deleted[String(g.gid)])
    .sort((a, b) => (pos.get(a.gid) ?? 0) - (pos.get(b.gid) ?? 0));
}
function groupItems(g) {
  return g.keys.map(k => itemByKey[k]).filter(Boolean)
    .sort((a, b) => (a.file + a.anchor_time).localeCompare(b.file + b.anchor_time));
}
function clusterColorGuess(g) {
  const counts = {};
  for (const k of g.keys) {
    const it = itemByKey[k];
    if (it && it.team_guess) counts[it.team_guess] = (counts[it.team_guess] || 0) + 1;
  }
  let best = "", n = 0;
  for (const t of Object.keys(counts)) { if (counts[t] > n) { n = counts[t]; best = t; } }
  return { color: best, count: n };
}
function clusterNumberVotes(g) {
  const counts = {};
  for (const k of g.keys) {
    const it = itemByKey[k];
    const num = it && it.number_guess ? it.number_guess.number : null;
    if (num) counts[num] = (counts[num] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([num, c]) => num + "×" + c).join(", ") || "无";
}
function clusterAssignedCount(g) {
  return g.keys.filter(k => marks[k]).length;
}
function clusterPrefill(g) {
  const counts = {};
  for (const k of g.keys) {
    if (touched[k]) continue;
    const it = itemByKey[k];
    if (it && it.prefill_tag) counts[it.prefill_tag] = (counts[it.prefill_tag] || 0) + 1;
  }
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  if (entries.length === 1) return entries[0][0];
  if (entries[0][1] > entries[1][1]) return entries[0][0];
  return null;
}
function pendingGroups() {
  const groups = computeGroups();
  return groups.filter(g => !clState.done[String(g.gid)]);
}
function clusterQueue() {
  const pending = pendingGroups();
  const set = new Set(clusterOrder);
  const still = clusterOrder.filter(gid => pending.some(g => g.gid === gid));
  for (const g of pending) { if (!set.has(g.gid)) still.push(g.gid); }
  return still.map(gid => pending.find(g => g.gid === gid)).filter(Boolean)
    .sort((a, b) => {
      const ka = clusterSortKey(a), kb = clusterSortKey(b);
      if (ka[0] !== kb[0]) return ka[0] - kb[0];
      return ka[1] - kb[1];
    });
}
function clusterSortKey(g) {
  const items = groupItems(g);
  const first = items.length ? items[0].anchor_time : 1e9;
  return [-g.keys.length, first];
}
function currentCluster() {
  const q = clusterQueue();
  if (!q.length) return null;
  curClusterIdx = Math.max(0, Math.min(curClusterIdx, q.length - 1));
  return q[curClusterIdx];
}
function initClusterOrder() {
  clusterOrder = pendingGroups().map(g => g.gid);
}
function setClusterDone(gid) {
  clState.done[String(gid)] = true;
  saveClState();
}
function isEAlias() {
  return !KEYMAP.some(m => m.key === "E");
}
function tagOfKey(key) {
  const m = KEYMAP.find(x => x.key === key);
  return m ? m.tag : null;
}
function assignCurrentObject(tag) {
  if (mode === "cluster") assignCluster(tag); else assignBall(tag);
}
function assignCluster(tag) {
  const g = currentCluster();
  if (!g) return;
  pushUndo("整簇归属");
  let changed = 0;
  for (const k of g.keys) {
    if (touched[k]) continue;
    if (pool.includes(k)) continue;
    marks[k] = tag;
    touched[k] = true;
    changed++;
    propagateFrom(k, tag);
  }
  // SKIP 球允许未归属，不阻塞簇 done：只要有改动或其余非 SKIP 球均已归属/剔除即置 done
  if (changed || g.keys.every(k => marks[k] || pool.includes(k) || isSkippedKey(k))) {
    setClusterDone(g.gid);
  }
  save();
  render();
}
function isSkippedKey(k) {
  const it = itemByKey[k];
  return it && it.status === "SKIP";
}
function clusterAdoptPrefill() {
  const g = currentCluster();
  if (!g) return;
  const tag = clusterPrefill(g);
  if (!tag) { toast("无可唯一采纳的预填"); return; }
  assignCluster(tag);
}
function clusterNoGoal() {
  const g = currentCluster();
  if (!g) return;
  if (!confirm("将整簇 " + g.keys.length + " 球标为不算进球？")) return;
  assignCluster(NOGOAL);
}
function clusterOpponent() {
  assignCluster(OPPONENT_TAG);
}
function skipCluster() {
  const q = clusterQueue();
  if (!q.length) return;
  const gid = q[curClusterIdx].gid;
  clusterOrder = clusterOrder.filter(id => id !== gid);
  clusterOrder.push(gid);
  render();
}
function clusterAcceptAllPrefills() {
  pushUndo("接受预填");
  let n = 0, nPhoto = 0, nAmb = 0, nTouched = 0;
  for (const it of ITEMS) {
    if (it.prefill_note === "ambiguous") nAmb++;
    if (!it.prefill_tag) continue;
    if (touched[it.key]) { nTouched++; continue; }
    if (marks[it.key] === it.prefill_tag) continue;
    marks[it.key] = it.prefill_tag;
    n++;
    if (it.prefill_note === "photo") nPhoto++;
  }
  save();
  render();
  alert("已接受 " + n + " 个预填（号码 " + (n - nPhoto) + " / 照片 " + nPhoto +
        "；歧义 " + nAmb + " / 已手改 " + nTouched + " 跳过）");
}
function acceptAllOpponent() {
  // 一键全收对手预填（opponent-prefill）：仅 team_guess="黑"（黑球衣=对手色系）
  pushUndo("接受对手预填");
  let n = 0, nTouched = 0;
  for (const it of ITEMS) {
    if (it.status === "SKIP") continue;
    if (it.team_guess !== "黑") continue;
    if (marks[it.key]) continue;
    if (touched[it.key]) { nTouched++; continue; }
    marks[it.key] = OPPONENT_TAG;
    n++;
  }
  save();
  render();
  alert("已接受 " + n + " 个对手预填（已手改 " + nTouched + " 跳过）");
}
function startMerge() {
  const g = currentCluster();
  if (!g) return;
  const all = computeGroups().filter(x => x.gid !== g.gid);
  if (!all.length) { toast("无其他簇可合并"); return; }
  mergeState = { srcGid: g.gid, targetIdx: 0, targets: all };
  render();
}
function moveMergeTarget(delta) {
  if (!mergeState) return;
  mergeState.targetIdx = (mergeState.targetIdx + delta + mergeState.targets.length)
    % mergeState.targets.length;
  render();
}
function confirmMerge() {
  if (!mergeState) return;
  const dst = mergeState.targets[mergeState.targetIdx];
  const src = mergeState.srcGid;
  mergeState = null;
  mergeInto(src, dst.gid);
}
function cancelMerge() {
  mergeState = null;
  render();
}
function mergeInto(srcGid, dstGid) {
  srcGid = groupIdOf(srcGid);
  dstGid = groupIdOf(dstGid);
  if (srcGid === dstGid) return;
  const groups = computeGroups();
  const src = groups.find(g => g.gid === srcGid);
  const dst = groups.find(g => g.gid === dstGid);
  if (!src || !dst) return;
  pushUndo("合并簇");
  for (const cid of src.cids) clState.merges[String(cid)] = dstGid;
  let tag = clState.clAssign[String(dstGid)];
  if (!tag) {
    const ts = dst.keys.map(k => marks[k]).filter(Boolean);
    if (ts.length && ts.every(x => x === ts[0])) tag = ts[0];
  }
  if (tag) {
    for (const k of src.keys) { if (!touched[k] && !pool.includes(k)) marks[k] = tag; }
  }
  const delAssign = [];
  for (const cid of src.cids) {
    const k = String(cid);
    delete clState.clAssign[k];
    delAssign.push(k);
  }
  save();
  saveClState({ merges: [], clAssign: delAssign });
  initClusterOrder();
  curClusterIdx = Math.max(0, clusterOrder.indexOf(dstGid));
  render();
}
function splitGroup(gid) {
  const doomed = Object.keys(clState.merges)
    .filter(k => groupIdOf(parseInt(k, 10)) === gid);
  if (!doomed.length) return;
  pushUndo("拆开簇");
  for (const k of doomed) delete clState.merges[k];
  saveClState({ merges: doomed, clAssign: [] });
  initClusterOrder();
  render();
}
function deleteCluster(gid) {
  const g = computeGroups().find(x => x.gid === gid);
  if (!g) return;
  if (!confirm("删除簇#" + gid + "？组内 " + g.keys.length +
               " 球的归属不变，该簇标记为已完成")) return;
  pushUndo("删除簇");
  clState.deleted[String(gid)] = true;
  clState.done[String(gid)] = true;
  saveClState();
  initClusterOrder();
  render();
}
function startEject() {
  const g = currentCluster();
  if (!g) return;
  const items = groupItems(g);
  if (!items.length) return;
  ejectState = { items, idx: 0 };
  render();
}
function moveEjectTarget(delta) {
  if (!ejectState) return;
  ejectState.idx = (ejectState.idx + delta + ejectState.items.length)
    % ejectState.items.length;
  render();
}
function ejectCurrent() {
  if (!ejectState) return;
  const it = ejectState.items[ejectState.idx];
  if (pool.includes(it.key)) return;
  pushUndo("剔除球");
  pool.push(it.key);
  savePool();
  save();
  render();
}
function cancelEject() {
  ejectState = null;
  render();
}
function ballQueue() {
  const pending = pendingGroups();
  const pendingGids = new Set(pending.map(g => g.gid));
  return ITEMS.filter(it => {
    if (ballFilter && ballFilter !== "__none__") return marks[it.key] === ballFilter;
    if (ballFilter === "__none__") return !marks[it.key];
    if (marks[it.key]) return false;
    if (pool.includes(it.key)) return true;
    if (it.cluster_id == null) return true;
    const gid = groupIdOf(it.cluster_id);
    return !pendingGids.has(gid);
  });
}
function currentBall() {
  const q = ballQueue();
  if (!q.length) return null;
  curBallIdx = Math.max(0, Math.min(curBallIdx, q.length - 1));
  return q[curBallIdx];
}
function assignBall(tag) {
  const q = ballQueue();
  if (!q.length) return;
  const it = q[curBallIdx];
  pushUndo("逐球归属");
  marks[it.key] = tag;
  touched[it.key] = true;
  propagateFrom(it.key, tag);
  save();
  render();
}
function skipBall() {
  const q = ballQueue();
  if (!q.length) return;
  curBallIdx = (curBallIdx + 1) % q.length;
  render();
}
function assignCurrentPrefill() {
  if (mode === "cluster") { clusterAdoptPrefill(); return; }
  const it = currentBall();
  if (it && it.prefill_tag) assignBall(it.prefill_tag);
  else toast("当前球无可采纳预填");
}
function propagateFrom(srcKey, tag) {
  if (tag === NOGOAL) return;
  const src = itemByKey[srcKey];
  if (!src || src.track_id == null) return;
  for (const it of ITEMS) {
    if (it.key === srcKey) continue;
    if (it.file !== src.file || it.track_id !== src.track_id) continue;
    if (marks[it.key] || it.prefill_tag || touched[it.key]) continue;
    marks[it.key] = tag;
    propagateAssign[it.key] = true;
  }
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
  const nUn = ITEMS.filter(it => it.status !== "SKIP" && !marks[it.key]).length;
  // nNo 数全量 marks 的哨兵球——同 session 跨批次共享 localStorage，
  // 只数本页 ITEMS 会漏报其他批次的剔除球
  const nNo = Object.values(marks).filter(t => t === NOGOAL).length;
  const payload = { session: SESSION, confirmed, players, assignments };
  const url = "/api/sessions/" + encodeURIComponent(SESSION) + "/roster-export";
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: payload })
  }).then(async res => {
    if (res.ok) {
      const j = await res.json();
      const n = typeof j.n_assignments === "number"
        ? j.n_assignments
        : Object.keys(assignments).length;
      alert("已保存到 " + j.path + "（归属 " + n + "/" + ITEMS.length +
            "，confirmed=" + confirmed +
            (nNo ? "，不算进球 " + nNo + " 球（已剔除不参与合成）" : "") +
            (nUn ? "，还有 " + nUn + " 个非 SKIP 球未归属" : "") + "）");
      return;
    }
    let reason = await res.text();
    try {
      const parsed = JSON.parse(reason);
      if (parsed.error) reason = parsed.error;
    } catch (e) {}
    if (res.status >= 400 && res.status < 500) {
      alert("服务端返回：" + reason);
    } else {
      alert("服务端异常（" + reason + "），请检查场次目录下文件是否已生成");
    }
  }).catch(err => {
    const reason = err && err.message ? err.message : String(err);
    alert("服务器保存失败（" + reason + "），已改为下载，请手动移到 work 场次目录");
    const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "roster.json";
    a.click();
  });
}
function setVideo(el, src) {
  if (src) { el.src = src; el.style.display = "inline-block"; el.play().catch(() => {}); }
  else { el.pause(); el.removeAttribute("src"); el.load(); el.style.display = "none"; }
}
function renderPlayerButtons(containerId, onTag) {
  const box = document.getElementById(containerId);
  box.innerHTML = "";
  for (const m of KEYMAP) {
    const b = document.createElement("button");
    b.className = "playerbtn " + teamClass(m.team);
    b.textContent = m.tag + (m.name ? "=" + m.name : "");
    const kc = document.createElement("span");
    kc.className = "keycap";
    kc.textContent = m.key;
    b.appendChild(kc);
    b.onclick = () => onTag(m.tag);
    box.appendChild(b);
  }
}
function renderBar() {
  const totalClusters = computeGroups().length;
  const doneClusters = Object.keys(clState.done).filter(k => !clState.deleted[k]).length;
  document.getElementById("prog").textContent =
    "簇 " + doneClusters + "/" + totalClusters + " · 球 " + nDone() + "/" + ITEMS.length;
  document.getElementById("oppInput").value = OPP;
  const inSubstate = mergeState !== null || ejectState !== null;
  for (const id of ["export", "undo", "toggleroster", "modebtn"]) {
    const b = document.getElementById(id);
    if (b) { b.disabled = inSubstate; b.style.opacity = inSubstate ? "0.5" : "1"; }
  }
  const inp = document.getElementById("oppInput");
  if (inp) inp.disabled = inSubstate;
}
function renderStage() {
  const el = document.getElementById("stage");
  if (mode === "cluster") {
    const g = currentCluster();
    const suffix = g ? " · 当前簇#" + g.gid + "（" + g.keys.length + " 球）" : " · 无待审簇";
    el.textContent = "模式一：簇审阅" + suffix;
  } else {
    el.textContent = "模式二：逐球收尾";
  }
}
function renderRosterTable() {
  const box = document.getElementById("rosterTable");
  if (!rosterExpanded) { box.style.display = "none"; return; }
  box.style.display = "block";
  let html = '<table><thead><tr><th>标签</th><th>姓名</th><th>队伍</th></tr></thead><tbody>';
  const teams = [OPP, HOME, "便服"];
  for (const p of PLAYERS) {
    html += '<tr><td>' + escapeHtml(p.tag) + '</td><td>' +
      '<input type="text" data-tag="' + escapeHtml(p.tag) + '" class="name-input" value="' +
      escapeHtml(p.name) + '"></td><td>' +
      '<select data-tag="' + escapeHtml(p.tag) + '" class="team-select">' +
      teams.map(t => '<option value="' + escapeHtml(t) + '"' +
        (p.team === t ? " selected" : "") + '>' + escapeHtml(t) + '</option>').join("") +
      '</select></td></tr>';
  }
  html += '</tbody></table>';
  if (mode === "cluster" && !mergeState && !ejectState) {
    const g = currentCluster();
    if (g) {
      html += '<div style="margin-top:8px;">当前簇#<span id="delGid">' + g.gid +
        '</span> <button id="delClusterBtn" class="nav">删除簇</button></div>';
    }
  }
  box.innerHTML = html;
  const delBtn = document.getElementById("delClusterBtn");
  if (delBtn) delBtn.onclick = () => {
    const gid = parseInt(document.getElementById("delGid").textContent, 10);
    if (!isNaN(gid)) deleteCluster(gid);
  };
  box.querySelectorAll(".name-input").forEach(inp => {
    inp.oninput = () => {
      const tag = inp.dataset.tag;
      const p = PLAYERS.find(x => x.tag === tag);
      if (!p) return;
      p.name = inp.value.trim();
      nameOvr[tag] = p.name;
      saveNames();
      render();
    };
    inp.onkeydown = (ev) => {
      if (ev.key === "Tab") return;
      ev.stopPropagation();
    };
  });
  box.querySelectorAll(".team-select").forEach(sel => {
    sel.onchange = () => {
      const tag = sel.dataset.tag;
      const p = PLAYERS.find(x => x.tag === tag);
      if (!p) return;
      p.team = sel.value;
      teamOvr[tag] = sel.value;
      saveTeamOvr();
      render();
    };
    sel.onkeydown = (ev) => ev.stopPropagation();
  });
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function renderClusterView() {
  document.getElementById("clusterView").style.display = mode === "cluster" ? "block" : "none";
  if (mode !== "cluster") return;
  const g = currentCluster();
  const meta = document.getElementById("clusterMeta");
  const crops = document.getElementById("clusterCrops");
  const video = document.getElementById("clusterVideo");
  const players = document.getElementById("clusterPlayers");
  const hint = document.getElementById("clusterHint");
  const mergeBox = document.getElementById("mergeTargets");
  if (!g) {
    meta.textContent = "无待审簇，按 / 可手动切到逐球收尾";
    crops.innerHTML = "";
    setVideo(video, "");
    players.innerHTML = "";
    hint.innerHTML = "";
    mergeBox.innerHTML = "";
    return;
  }
  const color = clusterColorGuess(g);
  meta.textContent = "簇#" + g.gid + " · " + g.keys.length + " 球 · 已归属 " +
    clusterAssignedCount(g) + " · 颜色:" + (color.color || "无") +
    " · 号码票:" + clusterNumberVotes(g);
  if (ejectState && ejectState.items.length) {
    crops.innerHTML = "";
    ejectState.items.forEach((it, i) => {
      const img = document.createElement("img");
      img.src = it.crop || "";
      if (i === ejectState.idx) img.classList.add("eject-sel");
      crops.appendChild(img);
    });
    setVideo(video, "");
    hint.innerHTML = "剔除子态：←/→ 选球，X 踢出进残余池，Esc 退出";
    players.innerHTML = "";
    mergeBox.innerHTML = "";
    return;
  }
  crops.innerHTML = "";
  for (const rc of g.rep_crops.slice(0, 8)) {
    const img = document.createElement("img");
    img.src = rc;
    img.alt = "簇代表图";
    crops.appendChild(img);
  }
  const items = groupItems(g);
  const clipItem = items.find(it => it.clip) || null;
  setVideo(video, clipItem ? clipItem.clip : "");
  const prefill = clusterPrefill(g);
  const colorHint = color.color === "黑" ? " · 黑球衣可接受对手预填" : "";
  hint.innerHTML = "选人键整簇归属；Enter" + (isEAlias() ? "/E" : "") +
    (prefill ? " 采纳预填" + prefill : "（无唯一预填）") +
    " · P 接受全场预填 · O 对手 · N 不算 · S 跳过 · M 合并 · X 剔除 · ←/→ 翻簇" +
    colorHint;
  if (!mergeState) renderPlayerButtons("clusterPlayers", tag => assignCluster(tag));
  if (mergeState && mergeState.srcGid === g.gid) {
    setVideo(video, "");
    players.innerHTML = "";
    hint.innerHTML = "合并子态：←/→ 选目标簇，Enter 确认，Esc 取消";
    mergeBox.innerHTML = "";
    const lab = document.createElement("div");
    lab.textContent = "合并：←/→ 选目标簇，Enter 确认，Esc 取消";
    mergeBox.appendChild(lab);
    mergeState.targets.forEach((tg, i) => {
      const div = document.createElement("div");
      div.className = "merge-target" + (i === mergeState.targetIdx ? " sel" : "");
      const img = document.createElement("img");
      img.src = tg.rep_crops[0] || "";
      div.appendChild(img);
      const sp = document.createElement("span");
      sp.textContent = "#" + tg.gid;
      div.appendChild(sp);
      div.onclick = () => { mergeState.targetIdx = i; render(); };
      mergeBox.appendChild(div);
    });
  } else {
    mergeBox.innerHTML = "";
  }
}
function renderBallView() {
  document.getElementById("ballView").style.display = mode === "ball" ? "block" : "none";
  if (mode !== "ball") return;
  const it = currentBall();
  const info = document.getElementById("ballInfo");
  const img = document.getElementById("crop");
  const v = document.getElementById("v");
  if (!it) {
    info.textContent = "无残余球可审";
    img.removeAttribute("src"); img.style.display = "none";
    setVideo(v, "");
    document.getElementById("ballPlayers").innerHTML = "";
    renderBallFilter();
    return;
  }
  if (it.crop) { img.src = it.crop; img.style.display = "inline-block"; }
  else { img.removeAttribute("src"); img.style.display = "none"; }
  setVideo(v, it.clip || "");
  let s = `第 ${curBallIdx + 1}/${ballQueue().length} 个 | ` +
    `已归属 ${nDone()}/${ITEMS.length} | ${it.file} t=${it.anchor_time}s`;
  if (it.cluster_id) s += " | 簇#" + groupIdOf(it.cluster_id);
  if (it.track_id != null) s += " | 轨迹#" + it.track_id;
  if (it.status === "SKIP") s += " | 无法定位";
  else if (it.prefill_tag) s += " | 预填:" + it.prefill_tag;
  else if (it.team_guess === "黑") s += " | 对手预填:黑";
  else if (it.team_guess) s += " | 颜色:" + it.team_guess;
  if (marks[it.key] && propagateAssign[it.key] && !touched[it.key]) s += " | 同轨迹预填";
  info.textContent = s;
  renderPlayerButtons("ballPlayers", tag => assignBall(tag));
  renderBallFilter();
}
function renderBallFilter() {
  const box = document.getElementById("ballFilter");
  box.innerHTML = "";
  const lab = document.createElement("span");
  lab.textContent = "过滤：";
  lab.className = "badge";
  box.appendChild(lab);
  const mk = (text, val) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.className = "nav" + (ballFilter === val ? " sel" : "");
    b.onclick = () => { ballFilter = val; curBallIdx = 0; render(); };
    box.appendChild(b);
  };
  mk("残余全部", "");
  mk("未归属", "__none__");
  const seen = [];
  for (const it of ITEMS) {
    const t = marks[it.key];
    if (t && !seen.includes(t)) seen.push(t);
  }
  for (const t of seen) {
    const p = PLAYERS.find(x => x.tag === t);
    mk(t + (p && p.name ? "=" + p.name : ""), t);
  }
}
function toggleRoster() {
  rosterExpanded = !rosterExpanded;
  render();
}
function toggleMode() {
  if (mode === "cluster") {
    mode = "ball";
    curBallIdx = 0;
  } else {
    const pending = pendingGroups();
    if (!pending.length) { toast("无待审簇，已在逐球模式"); return; }
    mode = "cluster";
  }
  render();
}
function autoModeSwitch() {
  if (mode === "cluster" && !pendingGroups().length) {
    mode = "ball";
    curBallIdx = 0;
    toast("簇队列已清空，自动切换逐球收尾");
  }
}
function render() {
  autoModeSwitch();
  renderBar();
  renderStage();
  renderRosterTable();
  renderClusterView();
  renderBallView();
  updateKeyHint();
}
function updateKeyHint() {
  const el = document.getElementById("keyhint");
  if (mergeState) { el.textContent = "合并子态：←/→ 选目标，Enter 确认，Esc 取消"; return; }
  if (ejectState) { el.textContent = "剔除子态：←/→ 选球，X 踢出，Esc 退出"; return; }
  let hint = "数字/字母=选人 · Enter" + (isEAlias() ? "/E" : "") + " 采纳预填";
  hint += " · P 接受全场预填 · O 对手 · N 不算 · S 跳过 · M 合并";
  hint += " · X 剔除 · Z 撤销 · G 名单 · / 切模式";
  el.textContent = hint;
}
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.style.display = "block";
  setTimeout(() => { el.style.display = "none"; }, 2500);
}
function isInputFocused() {
  const el = document.activeElement;
  if (!el) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" || el.isContentEditable;
}
function nextCluster() {
  const q = clusterQueue();
  if (!q.length) return;
  curClusterIdx = (curClusterIdx + 1) % q.length;
  render();
}
function prevCluster() {
  const q = clusterQueue();
  if (!q.length) return;
  curClusterIdx = (curClusterIdx - 1 + q.length) % q.length;
  render();
}
function nextBall() {
  const q = ballQueue();
  if (!q.length) return;
  curBallIdx = (curBallIdx + 1) % q.length;
  render();
}
function prevBall() {
  const q = ballQueue();
  if (!q.length) return;
  curBallIdx = (curBallIdx - 1 + q.length) % q.length;
  render();
}

document.getElementById("export").onclick = exportRoster;
document.getElementById("undo").onclick = undo;
document.getElementById("toggleroster").onclick = toggleRoster;
document.getElementById("modebtn").onclick = toggleMode;
document.getElementById("acceptopp").onclick = acceptAllOpponent;
document.getElementById("oppInput").onchange = (ev) => {
  pushUndo("改对手队名");
  oppNameOvr = ev.target.value.trim();
  OPP = oppNameOvr || OPP;
  saveOppName();
  render();
};
document.getElementById("oppInput").onkeydown = (ev) => ev.stopPropagation();

document.addEventListener("keydown", (ev) => {
  if (isInputFocused()) return;
  const k = ev.key;
  if (mergeState) {
    if (k === "Escape") { cancelMerge(); return; }
    if (k === "ArrowLeft") { moveMergeTarget(-1); return; }
    if (k === "ArrowRight") { moveMergeTarget(1); return; }
    if (k === "Enter") { confirmMerge(); return; }
    return;
  }
  if (ejectState) {
    if (k === "Escape") { cancelEject(); return; }
    if (k === "ArrowLeft") { moveEjectTarget(-1); return; }
    if (k === "ArrowRight") { moveEjectTarget(1); return; }
    if (k.toLowerCase() === "x") { ejectCurrent(); return; }
    return;
  }
  if (k === "Enter") { assignCurrentPrefill(); return; }
  if (k.toLowerCase() === "e") {
    if (isEAlias()) assignCurrentPrefill(); else {
      const tag = tagOfKey("E"); if (tag) assignCurrentObject(tag);
    }
    return;
  }
  if (k.toLowerCase() === "p") { clusterAcceptAllPrefills(); return; }
  if (k.toLowerCase() === "o") { assignCurrentObject(OPPONENT_TAG); return; }
  if (k.toLowerCase() === "n") {
    if (mode === "cluster") clusterNoGoal(); else assignBall(NOGOAL);
    return;
  }
  if (k.toLowerCase() === "s") { if (mode === "cluster") skipCluster(); else skipBall(); return; }
  if (k.toLowerCase() === "m") { if (mode === "cluster") startMerge(); return; }
  if (k.toLowerCase() === "x") { if (mode === "cluster") startEject(); return; }
  if (k.toLowerCase() === "z") { undo(); return; }
  if (k.toLowerCase() === "g") { toggleRoster(); return; }
  if (k === "/") { toggleMode(); return; }
  if (k === "ArrowLeft") { if (mode === "cluster") prevCluster(); else prevBall(); return; }
  if (k === "ArrowRight") { if (mode === "cluster") nextCluster(); else nextBall(); return; }
  const pk = k.length === 1 ? k.toUpperCase() : "";
  if (pk && PLAYER_KEYS.includes(pk)) {
    const tag = tagOfKey(pk);
    if (tag) assignCurrentObject(tag);
    return;
  }
});

loadStorage();
initClusterOrder();
if (!CLUSTERS.length || !pendingGroups().length) {
  mode = "ball";
}
render();
</script>
</body>
</html>
"""


def team_of_tag(tag: str, opp: str, home: str = TEAM_HOME_DEFAULT) -> str:
    """按标签前缀推定队别：黑*/蓝*→对手队（opp）、白*→主队（home），其余归便服。

    对手伪球员 ``tag=="对手"`` 单独返回对手队名（docs/opponent-filter/spec.md），
    不走前缀兜底便服。页面导出自动补录名单外标签时用同一规则
    （JS teamOfTag 与本文档同步，改规则须两端一起改）。

    Args:
        tag: 球员标签，如 ``黑21`` / ``白-李四`` / ``灰T恤-A`` / ``对手``。
        opp: 对手队名（opponent_of 产物）。
        home: 主队名（team_config 注入，缺省 DEFAULT_TEAM_NAME）。

    Returns:
        opp / home / "便服"。
    """
    if tag == OPPONENT_TAG:
        return opp
    for prefix, side in _TEAM_PREFIXES:
        if tag.startswith(prefix):
            return opp if side == "opp" else home
    return TEAM_CASUAL


def parse_players(spec: str, opp: str, home: str = TEAM_HOME_DEFAULT) -> list[Player]:
    """解析 --players 名单串："黑21=张三,白-李四=李四" → Player 列表。

    每条为 ``tag[=name]``（name 可省，省则为空串）；队别按 team_of_tag 推定。

    Args:
        spec: 逗号分隔的名单串；空串返回空列表。
        opp: 对手队名（opponent_of 产物，传给 team_of_tag）。
        home: 主队名（team_config 注入，缺省 DEFAULT_TEAM_NAME）。

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
    """两字符串是否相等或只差 1 个字符（增/删/改）——K3 印名误读容差（张三≈张二）。"""
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
    tag: str  # 名单球员 tag 或占位 tag（<主队名><号码>）


def resolve_photo_guesses(
    matches: dict[str, MatchEntry],
    players: list[Player],
    home: str = TEAM_HOME_DEFAULT,
) -> tuple[dict[str, PhotoGuess], list[Player]]:
    """照片命中号码 → 名单 tag；名单缺号 → 占位 Player（<主队名><号>，team=home）。

    号码查名单复用 match_players_by_number 的数字边界口径（颜色传 None：照片
    识别不给颜色提示）。同号多人 → WARNING 跳过该球不预填（交人裁判，与读号
    歧义同口径）。占位条目随 players 名单注入页面，不得依赖 teamOfTag 前缀推队
    （``主队7`` 不以黑/蓝/白开头，前缀推队会误归便服——spec 写死）。

    Args:
        matches: photo_match_scorers.validate_matches_payload 校验产物（key → 命中）。
        players: 本页球员名单（--players/--players-file/已有 roster 合并后）。
        home: 主队名（team_config 注入，缺省 DEFAULT_TEAM_NAME）。

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
    home: str = TEAM_HOME_DEFAULT,
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
        home: 主队名（注入 JS ``const HOME``，team_config 产物；缺省 DEFAULT_TEAM_NAME）。
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
    keymap_json = json.dumps(build_keymap(page_players), ensure_ascii=False)
    player_keys_json = json.dumps(list(PLAYER_KEYS), ensure_ascii=False)
    return (
        _HTML.replace("__ITEMS__", json.dumps(entries, ensure_ascii=False))
        .replace("__PLAYERS__", players_json)
        .replace("__KEYMAP__", keymap_json)
        .replace("__PLAYER_KEYS__", player_keys_json)
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
