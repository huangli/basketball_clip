"""gen_label_page 单元测试：HTML 渲染与最新索引自动发现。

覆盖：build_html 内联事件与场次、无"测试员甲"专属按钮、含位置记忆与
"跳到未标"控件；find_latest_index 按 mtime 取最新、无匹配返回 None。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import Any

import pytest

from gen_label_page import assign_same_rally_groups, build_html, find_latest_index


def _event(key: str = "f1#e0", fid: str = "f1", anchor_t0: float = 1.0) -> dict[str, Any]:
    """构造一条合法事件记录。"""
    return {
        "key": key,
        "fid": fid,
        "event_idx": 0,
        "clip": "clips/a.mp4",
        "clip_wide": "clips/a_wide.mp4",
        "src_file": "a.mp4",
        "anchor_t0": anchor_t0,
        "verdict": "?",
    }


def test_build_html_inlines_events_and_session() -> None:
    # Arrange / Act
    html = build_html([_event()], "20260722")
    # Assert
    assert '"key": "f1#e0"' in html
    assert 'const SESSION = "20260722";' in html


def test_build_html_export_name_defaults_to_session() -> None:
    # Arrange / Act：不传 batch（旧布局/adhoc/手工调用）
    html = build_html([_event()], "20260722")
    # Assert：导出维持旧名 goals_<场次>.json（按钮/下载/alert 三处同源）
    assert 'const OUTNAME = "goals_20260722.json";' in html
    assert "导出 goals_20260722.json" in html
    assert "a.download = OUTNAME;" in html


def test_build_html_export_name_with_batch() -> None:
    # Arrange / Act：传 batch=2（label-export-batch：导出即 goals_batchK.json，
    # 用户下载后移动即可、无需改名；2026-08-14 对手队改错名实踩驱动）
    html = build_html([_event()], "s", batch=2)
    # Assert
    assert 'const OUTNAME = "goals_batch2.json";' in html
    assert "导出 goals_batch2.json" in html


def test_build_html_rejects_invalid_batch() -> None:
    # Arrange / Act / Assert：batch < 1 直接拒（防静默产出 CLI 不认的文件名）
    with pytest.raises(ValueError, match="batch 必须 ≥1"):
        build_html([_event()], "s", batch=0)


def test_generated_js_syntax_node_check(tmp_path: pathlib.Path) -> None:
    # Arrange：node 不在 PATH 则跳过（仿 gen_scorer_page 同款，防模板转义黑屏回归）
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不在 PATH")
    html = build_html([_event()], "s", batch=2)
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    js_path = tmp_path / "page.js"
    js_path.write_text(script, encoding="utf-8")
    # Act
    proc = subprocess.run(  # noqa: S603 node 路径来自 shutil.which，可信
        [node, "--check", str(js_path)], capture_output=True, text=True, check=False
    )
    # Assert
    assert proc.returncode == 0, proc.stderr


def test_build_html_has_no_dabin_button() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert（测试员甲按钮已下线；导出仍兼容历史 scorer 字段，模板不含该文案）
    assert "测试员甲" not in html


def test_build_html_restores_position_and_has_jump_button() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert
    assert 'id="toun"' in html
    assert "POSKEY" in html
    assert "localStorage.getItem(POSKEY)" in html


def test_find_latest_index_picks_newest(tmp_path: pathlib.Path) -> None:
    # Arrange：两个场次的索引，review_v2 修改时间更新
    old = tmp_path / "work" / "s1" / "review_v1"
    new = tmp_path / "work" / "s1" / "review_v2"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "events_index.json").write_text("{}", encoding="utf-8")
    os.utime(old / "events_index.json", (1_600_000_000, 1_600_000_000))
    os.utime(new / "events_index.json", (1_700_000_000, 1_700_000_000))
    # Act
    got = find_latest_index(str(tmp_path / "work"))
    # Assert
    assert got is not None
    assert "review_v2" in got


def test_find_latest_index_returns_none_when_empty(tmp_path: pathlib.Path) -> None:
    # Arrange / Act / Assert
    assert find_latest_index(str(tmp_path / "work")) is None


# 审核窗口口径：前 2s 后 4s（gen_review_clips.CLIP_BEFORE/AFTER_SEC），
# 相邻事件 anchor 差 ≤ 6s 则窗口重叠视为疑似同回合（传递闭包）。


def test_groups_no_overlap_stay_ungrouped() -> None:
    # Arrange：同 fid 两事件 anchor 差 19s，远超 6s 窗口
    events = [_event("f1#e0", anchor_t0=1.0), _event("f1#e1", anchor_t0=20.0)]
    # Act / Assert
    assert assign_same_rally_groups(events) == {}


def test_groups_pair_overlap_same_group() -> None:
    # Arrange：anchor 差 3s ≤ 6s，窗口重叠（批次 3 同球对差值 2.3~4.2s 模式）
    events = [_event("f1#e0", anchor_t0=1.0), _event("f1#e1", anchor_t0=4.0)]
    # Act
    groups = assign_same_rally_groups(events)
    # Assert
    assert groups == {"f1#e0": 1, "f1#e1": 1}


def test_groups_transitive_closure() -> None:
    # Arrange：1.0 与 9.0 不直接重叠（差 8s），但经 5.0 传递同组
    events = [
        _event("f1#e0", anchor_t0=1.0),
        _event("f1#e1", anchor_t0=5.0),
        _event("f1#e2", anchor_t0=9.0),
    ]
    # Act
    groups = assign_same_rally_groups(events)
    # Assert
    assert groups == {"f1#e0": 1, "f1#e1": 1, "f1#e2": 1}


def test_groups_never_cross_fid() -> None:
    # Arrange：不同 fid 即使 anchor 差 1s 也绝不混组
    # （跨文件同球识别是一期明确边界，见 docs/dedup-same-goal/spec.md）
    events = [_event("f1#e0", fid="f1", anchor_t0=1.0), _event("f2#e0", fid="f2", anchor_t0=2.0)]
    # Act / Assert
    assert assign_same_rally_groups(events) == {}


def test_groups_single_event_returns_empty() -> None:
    # Arrange / Act / Assert
    assert assign_same_rally_groups([_event()]) == {}


def test_groups_two_separate_groups_numbered_in_order() -> None:
    # Arrange：同 fid 四个事件成两组，组号按 anchor 升序递增
    events = [
        _event("f1#e0", anchor_t0=1.0),
        _event("f1#e1", anchor_t0=3.0),
        _event("f1#e2", anchor_t0=30.0),
        _event("f1#e3", anchor_t0=33.0),
    ]
    # Act
    groups = assign_same_rally_groups(events)
    # Assert
    assert groups == {"f1#e0": 1, "f1#e1": 1, "f1#e2": 2, "f1#e3": 2}


def test_build_html_injects_group_fields_for_overlapping_events() -> None:
    # Arrange：同 fid 两事件窗口重叠（anchor 差 3s）
    events = [_event("f1#e0", anchor_t0=1.0), _event("f1#e1", anchor_t0=4.0)]
    # Act
    html = build_html(events, "s")
    # Assert：组号与组大小内联进事件数据，页面含组标签元素
    assert '"grp": 1' in html
    assert '"grp_size": 2' in html
    assert 'id="grp"' in html
    assert "疑似同回合" in html
    # 且不改调用方原 dict（无副作用）
    assert "grp" not in events[0] and "grp" not in events[1]


def test_build_html_ungrouped_events_have_no_grp_field() -> None:
    # Arrange：单事件不成组
    # Act
    html = build_html([_event()], "s")
    # Assert：内联事件数据无 grp 字段（带冒号限定 JSON 字段形式，
    # 排除模板里 getElementById("grp") 的固有子串）
    assert '"grp":' not in html


def test_build_html_export_has_same_rally_confirm() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：导出前置确认框（同组多进球时 confirm 拦截）
    assert "confirm(" in html
    assert "确实是两个球" in html
    # 防回归：confirm 文案的 \n 必须以字面两字符存在于 JS 源码（_HTML 为 raw
    # string 保此）；若被 Python 转义成真实换行，JS 字符串跨行 SyntaxError 整页黑屏
    assert "\\n" in html
    assert '多个进球：\n"' not in html


def test_build_html_defaults_double_speed_with_s_toggle() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：页面默认 1x（片段已烘焙 2x，有效 2x）+ S 键切换 + 页头倍速控件
    # （label-speedup F2；2026-08-09 用户定：默认保持有效 2x，S 加到 4x）
    assert "v.playbackRate = 1;" in html
    assert 'id="speed"' in html
    assert 'k === "s"' in html


def test_build_html_has_same_rally_skip_confirm() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：组内新判 J 后弹确认框，同组未标注成员可一键标 F 跳过
    # （label-speedup F1；仅提示不强制，已标注成员不覆盖）
    assert "疑似同回合组" in html
    assert "其余标 F" in html
    assert "const isNew = !marks[e.key];" in html


def test_groups_skips_events_missing_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:  # Arrange：缺 anchor_t0 / fid / key 的事件各一 + 一条合法重叠对
    events = [
        {"fid": "f1", "key": "f1#e0"},  # 缺 anchor_t0
        {"anchor_t0": 1.0, "key": "f1#e1"},  # 缺 fid
        {"fid": "f1", "anchor_t0": 1.0},  # 缺 key
        _event("f1#e2", anchor_t0=1.0),
        _event("f1#e3", anchor_t0=3.0),
    ]
    # Act
    with caplog.at_level("WARNING", logger="gen_label_page"):
        groups = assign_same_rally_groups(events)
    # Assert：残缺事件跳过并记 WARNING，合法事件正常成组
    assert groups == {"f1#e2": 1, "f1#e3": 1}
    assert caplog.text.count("跳过同回合分组") == 3


def test_sound_toggle_does_not_reload_clip() -> None:
    # Bug①（label-page-fixes）：声音开关只切 muted，不调 show() 重置进度/倍速/视角
    html = build_html([_event()], "s", batch=1)
    start = html.index('getElementById("sound").onclick')
    stmt = html[start : html.index("};", start)]  # onclick 整条语句（可多行）
    assert "show(" not in stmt
    assert "v.muted = !v.muted" in stmt
    assert "textContent" in stmt  # 按钮文本就地更新


def test_localstorage_keys_isolated_per_batch() -> None:
    # Bug②（label-page-fixes）：批次页存储键带 _batchK 后缀，跨批进度/位置不串
    h2 = build_html([_event()], "s", batch=2)
    h3 = build_html([_event()], "s", batch=3)
    assert 'const LSKEY = "label_" + SESSION + "_batch2";' in h2
    assert 'const LSKEY = "label_" + SESSION + "_batch3";' in h3
    # POSKEY 由 LSKEY 派生，随之隔离
    assert 'const POSKEY = LSKEY + "_pos";' in h2


def test_localstorage_keys_unchanged_without_batch() -> None:
    # 不传 batch（旧布局/adhoc/手工调用）：旧键逐字节不变
    html = build_html([_event()], "s")
    assert 'const LSKEY = "label_" + SESSION + "";' in html
    assert "_batch" not in html.split("const LSKEY", 1)[1].splitlines()[0]


# ---- goal-anchor：J 键人工锚点 ----


def test_build_html_injects_speed_constant() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：SPEED 取 gen_review_clips.SPEED（J 捕锚换算 currentTime×SPEED），不硬编码
    import gen_review_clips

    assert f"const SPEED = {gen_review_clips.SPEED};" in html


def test_build_html_passthrough_anchor_fields() -> None:
    # Arrange：带新字段的事件
    e = _event()
    e["clip_src_start"] = 12.3
    e["continued"] = True
    # Act
    html = build_html([e], "s")
    # Assert：两字段原样透传进页面 JSON（JS 捕锚/退化依据）
    assert '"clip_src_start": 12.3' in html
    assert '"continued": true' in html


def test_build_html_goal_anchor_js_paths() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：J/按钮捕锚、T 机器锚兜底、缺字段或 continued 退化、导出人工锚优先
    assert "markGoal(true)" in html
    assert "markGoal(false)" in html
    assert "e.clip_src_start + v.currentTime * SPEED" in html
    assert "!e.continued" in html
    assert 'typeof m.anchor === "number"' in html
    assert "m.anchor = Math.round(anchor * 10) / 10" in html


# ---- label-jump-goal：跳到进球导航 ----


def test_build_html_has_jump_goal_button_and_binding() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：按钮 + jumpGoal 循环查找 + G 键绑定 + 只挑 r==="goal"
    assert 'id="tog"' in html
    assert "function jumpGoal()" in html
    assert 'marks[x.key].r === "goal"' in html
    assert '"g") jumpGoal()' in html
    # 落空原地不动（show(cur)），与 jumpUnmarked 同模式
    assert "show(n >= 0 ? n : cur)" in html


# ---- label-jump-goal review02：改标/补锚不再自动跳走 ----


def test_mark_does_not_advance_on_remark() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：改标（已有标记）原地 show(cur)；仅首标才查找下一个未标前进
    assert "if (!isNew)" in html
    assert "show(cur);" in html


# ---- export-autosave：标注页导出改为 POST 回服务端 ----


def test_build_html_injects_batch_constant_int() -> None:
    # Arrange / Act
    html = build_html([_event()], "s", batch=2)
    # Assert：batch 为 int 时直接注入数字，供 fetch body 使用
    assert "const BATCH = 2;" in html


def test_build_html_injects_batch_constant_null() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：无 batch 时注入 null，服务端据此推导 goals_<场次>.json
    assert "const BATCH = null;" in html


def test_build_html_export_uses_post_fetch_and_three_branches() -> None:
    # Arrange / Act
    html = build_html([_event()], "s", batch=2)
    # Assert：fetch POST 到统一端点，并保留三分支错误处理
    assert 'const url = "/api/sessions/" + encodeURIComponent(SESSION) + "/label-export"' in html
    assert "fetch(url, {" in html
    assert 'method: "POST"' in html
    assert "JSON.stringify({ batch: BATCH, data: { session: SESSION, goals } })" in html
    assert "服务器保存失败（" in html
    assert "已改为下载，请手动移到 work 场次目录" in html
    assert "服务端返回：" in html
    assert "服务端异常（" in html
    assert "请检查场次目录下文件是否已生成" in html


def test_build_html_export_success_alert_shows_path_and_counts() -> None:
    # Arrange / Act
    html = build_html([_event()], "s", batch=1)
    # Assert：成功弹窗用服务端返回的 path/n_confirmed/n_total 文案
    assert "已保存到 " in html
    assert "j.path" in html
    assert "j.n_confirmed" in html
    assert "j.n_total" in html


def test_build_html_export_alerts_have_no_backslash() -> None:
    # Arrange / Act
    html = build_html([_event()], "s")
    # Assert：新增 5xx/回退提示语不得含字面反斜杠（Windows 路径历史坑）
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    # 过滤掉历史 confirm 文案里的合法 \n，只检查新增 alert 文案段
    export_start = script.index("function exportGoals()")
    export_block = script[export_start:]
    alert_texts = [m.split('"', 1)[1] for m in export_block.split('alert("')[1:]]
    for text in alert_texts:
        assert "\\" not in text, f"alert 文案含反斜杠: {text!r}"
