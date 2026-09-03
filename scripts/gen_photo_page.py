"""生成照片挑选确认页 photo_page.html：用户在瀑布流上点选精彩照片。

读取 rank_photos 产出的 photo_candidates.json（每张：id/源视频/时刻/分数/裁图
相对路径），在同目录生成自包含 photo_page.html（数据内联、图片相对路径、
CSS columns 瀑布流、点选/再点取消、双击看原图、快捷键翻页、localStorage
进度、一键导出 photo_selections.json）。用户浏览器打开即可；导出物移动到
work/<场次>/photos/ 后由 rank_photos.py --apply（或 video photo --apply）
落盘到 output/<场次>/照片精选/。

输入：--session 场次 ID（产物路径 work/<场次>/photos/ 约定布局）
输出：work/<场次>/photos/photo_page.html
依赖：scripts/pipe_common.py（read_json/run_id 日志）、scripts/errors.py、
    scripts/rank_photos.py（CANDIDATES_VERSION / SELECTIONS_NAME 契约）
典型调用：
    python scripts/gen_photo_page.py --session 20260801_对手名
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from errors import BasketballPipelineError, SchemaError
from pipe_common import configure_logging, new_run_id, read_json
from rank_photos import CANDIDATES_VERSION, SELECTIONS_NAME

logger = logging.getLogger(__name__)

PAGE_SIZE: int = 50  # 每页候选数（200 张分 4 页，数字键直达）

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>照片挑选 __SESSION__</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; margin: 16px; }
#bar { position: sticky; top: 0; background: #111; padding: 8px 0; z-index: 9;
       border-bottom: 1px solid #333; }
button { font-size: 16px; padding: 8px 14px; margin: 4px; border-radius: 8px;
         border: 0; cursor: pointer; background: #444; color: #fff; }
#export { background: #8a6d00; }
#allpage { background: #2c5e9e; }
.badge { color: #fc3; font-size: 18px; }
small { color: #999; }
code { color: #fc3; }
#grid { column-count: 4; column-gap: 8px; margin-top: 10px; }
.card { break-inside: avoid; margin: 0 0 8px; background: #1c1c1c;
        border: 3px solid #333; border-radius: 6px; cursor: pointer; }
.card.sel { border-color: #fc3; }
.card img { width: 100%; display: block; }
.card .meta { font-size: 13px; color: #aaa; padding: 4px 8px; }
.card.sel .meta { color: #fc3; }
#overlay { position: fixed; inset: 0; background: rgba(0,0,0,.93); display: none;
           z-index: 99; text-align: center; cursor: zoom-out; }
#overlay img { max-width: 96vw; max-height: 94vh; margin-top: 2vh; }
</style>
</head>
<body>
<div id="bar">
  <span class="badge" id="prog"></span> <span id="pages"></span><br>
  <button id="prev">← 上一页</button>
  <button id="next">下一页 →</button>
  <button id="allpage">全选本页 (A)</button>
  <button id="export">导出 selections (回车)</button>
  <br><small>点图=选/取消；双击=看原图（Esc 或点原图关闭）；←/→ 或数字键 1-9 翻页；
  A=全选/取消本页；回车=导出。选择自动存浏览器，刷新不丢。</small>
  <br><small>导出得到 <code>__SELNAME__</code> → 移动到
  <code>work\\__SESSION__\\photos\\</code> → 运行
  <code>video photo --session __SESSION__ --apply</code>
  落盘到 <code>output\\__SESSION__\\照片精选\\</code></small>
</div>
<div id="grid"></div>
<div id="overlay"><img id="big" alt="原图"></div>
<script>
const ITEMS = __ITEMS__;
const SESSION = "__SESSION__";
const SELNAME = "__SELNAME__";
const PAGE_SIZE = __PAGESIZE__;
const LSKEY = "photo_" + SESSION;
let sel = {};
try { sel = JSON.parse(localStorage.getItem(LSKEY) || "{}"); } catch (e) { sel = {}; }
let page = 0;
const nPages = Math.max(1, Math.ceil(ITEMS.length / PAGE_SIZE));
function save() { localStorage.setItem(LSKEY, JSON.stringify(sel)); }
function nSel() { return Object.keys(sel).filter(k => sel[k]).length; }
function render() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  const lo = page * PAGE_SIZE, hi = Math.min(ITEMS.length, lo + PAGE_SIZE);
  for (let i = lo; i < hi; i++) {
    const it = ITEMS[i];
    const card = document.createElement("div");
    card.className = "card" + (sel[it.id] ? " sel" : "");
    const img = document.createElement("img");
    img.src = it.img; img.loading = "lazy"; img.alt = it.id;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = it.id + " · 视频" + it.v + " · " + it.t.toFixed(1) + "s · 分 " + it.s;
    card.appendChild(img); card.appendChild(meta);
    card.onclick = () => { sel[it.id] = !sel[it.id]; save(); render(); };
    card.ondblclick = () => {
      document.getElementById("big").src = it.img;
      document.getElementById("overlay").style.display = "block";
    };
    grid.appendChild(card);
  }
  document.getElementById("prog").textContent =
    "已选 " + nSel() + " / " + ITEMS.length + " 张";
  document.getElementById("pages").textContent =
    "第 " + (page + 1) + "/" + nPages + " 页";
  window.scrollTo(0, 0);
}
function showPage(p) { page = Math.max(0, Math.min(nPages - 1, p)); render(); }
function toggleAllPage() {
  const lo = page * PAGE_SIZE, hi = Math.min(ITEMS.length, lo + PAGE_SIZE);
  const allSel = ITEMS.slice(lo, hi).every(it => sel[it.id]);
  for (let i = lo; i < hi; i++) sel[ITEMS[i].id] = !allSel;
  save(); render();
}
function exportSel() {
  const ids = ITEMS.filter(it => sel[it.id]).map(it => it.id);
  const payload = JSON.stringify({ session: SESSION, count: ids.length, selected: ids }, null, 1);
  const blob = new Blob([payload], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = SELNAME;
  a.click();
  alert("已下载 " + SELNAME + "（选中 " + ids.length + " 张）。\\n" +
        "移动到 work\\\\" + SESSION + "\\\\photos\\\\ 后运行：\\n" +
        "video photo --session " + SESSION + " --apply");
}
document.getElementById("prev").onclick = () => showPage(page - 1);
document.getElementById("next").onclick = () => showPage(page + 1);
document.getElementById("allpage").onclick = toggleAllPage;
document.getElementById("export").onclick = exportSel;
document.getElementById("overlay").onclick = () => {
  document.getElementById("overlay").style.display = "none";
};
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    document.getElementById("overlay").style.display = "none";
    return;
  }
  if (document.getElementById("overlay").style.display === "block") return;
  const k = ev.key.toLowerCase();
  if (k === "arrowleft") showPage(page - 1);
  else if (k === "arrowright") showPage(page + 1);
  else if (k >= "1" && k <= "9") showPage(parseInt(k, 10) - 1);
  else if (k === "a") toggleAllPage();
  else if (ev.key === "Enter") exportSel();
});
render();
</script>
</body>
</html>
"""


def validate_candidates(data: Any, path: Path) -> tuple[str, list[dict[str, Any]]]:  # noqa: ANN401
    """校验 photo_candidates.json（rules.md §0.2：损坏显式失败）。

    Args:
        data: read_json 解析产物。
        path: 源文件路径（错误信息用）。

    Returns:
        (session, candidates 条目列表)。

    Raises:
        SchemaError: 顶层结构 / 条目字段缺失或类型错。
    """
    if not isinstance(data, dict) or data.get("version") != CANDIDATES_VERSION:
        raise SchemaError(f"{path}: 顶层必须是 version={CANDIDATES_VERSION} 的对象")
    session: Any = data.get("session")
    if not isinstance(session, str) or not session:
        raise SchemaError(f"{path}: session 缺失或非字符串")
    cands: Any = data.get("candidates")
    if not isinstance(cands, list):
        raise SchemaError(f"{path}: candidates 必须是列表")
    for i, raw in enumerate(cands):
        if not isinstance(raw, dict):
            raise SchemaError(f"{path}: candidates[{i}] 不是对象")
        try:
            if not isinstance(raw["id"], str):
                raise TypeError("id 非字符串")
            if not isinstance(raw["image"], str):
                raise TypeError("image 非字符串")
            if not isinstance(raw["src_file"], str):
                raise TypeError("src_file 非字符串")
            if not isinstance(raw["status"], str):
                raise TypeError("status 非字符串")
            int(raw["video_no"])
            float(raw["sec"])
            float(raw["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{path}: candidates[{i}] 字段缺失/类型错: {exc}") from exc
    return session, cands


def build_page_data(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """装配页面条目：只收 status=ok 且有图的候选，字段压成页面用最小集。

    Args:
        candidates: validate_candidates 通过后的条目列表。

    Returns:
        页面条目 [{id, img, v, t, s}]（保持原顺序 = 场次时间序）。
    """
    items: list[dict[str, Any]] = []
    for c in candidates:
        if c["status"] != "ok" or not c["image"]:
            continue
        items.append(
            {
                "id": c["id"],
                "img": c["image"],
                "v": int(c["video_no"]),
                "t": float(c["sec"]),
                "s": f"{float(c['score']):.2f}",
            }
        )
    return items


def render_html(session: str, candidates: list[dict[str, Any]]) -> str:
    """渲染自包含确认页 HTML（数据内联，图片相对路径引用）。

    Args:
        session: 场次 ID（标题 / localStorage 键 / 导出文件名 / 路径说明）。
        candidates: 候选条目（photo_candidates.json 的 candidates 列表，
            须已过 validate_candidates）。

    Returns:
        完整 HTML 文本。
    """
    items: list[dict[str, Any]] = build_page_data(candidates)
    items_json: str = json.dumps(items, ensure_ascii=False)
    return (
        _HTML.replace("__SESSION__", session)
        .replace("__ITEMS__", items_json)
        .replace("__SELNAME__", SELECTIONS_NAME)
        .replace("__PAGESIZE__", str(PAGE_SIZE))
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """命令行参数解析。"""
    ap = argparse.ArgumentParser(
        prog="gen_photo_page", description="生成照片挑选确认页（瀑布流点选 → 导出 selections）"
    )
    ap.add_argument("--session", required=True, help="场次 ID（work/<场次>/photos/ 约定布局）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功，非 0=失败）。"""
    args: argparse.Namespace = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        photos_dir: Path = Path("work") / args.session / "photos"
        cand_path: Path = photos_dir / "photo_candidates.json"
        if not cand_path.is_file():
            raise BasketballPipelineError(f"缺 candidates: {cand_path}（先跑 rank_photos）")
        session, cands = validate_candidates(
            read_json(cand_path, what="photo_candidates.json"), cand_path
        )
        if session != args.session:
            raise SchemaError(f"{cand_path}: session={session} 与 --session {args.session} 不符")
        html: str = render_html(session, cands)
        out_path: Path = photos_dir / "photo_page.html"
        out_path.write_text(html, encoding="utf-8")
        n_ok: int = len(build_page_data(cands))
        logger.info("确认页生成: %s（%d 张候选）", out_path, n_ok)
        return 0
    except BasketballPipelineError as exc:
        logger.error("失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1
    except OSError as exc:
        logger.error("IO 失败 run_id=%s: %s", run_id, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    # 管道/重定向时 stdout 回落 locale 编码，打印中文日志会 UnicodeEncodeError
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure") and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
