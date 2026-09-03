"""gen_photo_page.py 单元测试（瀑布流确认页渲染）。

覆盖：candidates JSON → 自包含 HTML（数据内联、图片相对路径、全部候选条目、
快捷键说明、导出脚本与保存路径说明、session 标题）。不触碰真实素材。
"""

from __future__ import annotations

from gen_photo_page import build_page_data, render_html


def _candidates() -> list[dict]:
    """两张候选的最小数据（与 rank_photos 产出的 photo_candidates.json 条目同构）。"""
    return [
        {
            "id": "c001",
            "src_file": "a_video.mp4",
            "video_no": 1,
            "sec": 12.3,
            "score": 0.81,
            "image": "candidates/c001.jpg",
            "status": "ok",
        },
        {
            "id": "c002",
            "src_file": "b_video.mp4",
            "video_no": 2,
            "sec": 3.0,
            "score": 0.42,
            "image": "candidates/c002.jpg",
            "status": "ok",
        },
    ]


class TestRenderHtml:
    """HTML 渲染：全部候选内联、相对图片路径、导出契约齐全。"""

    def test_contains_all_candidates(self) -> None:
        html = render_html("20260805_对手队", _candidates())
        assert '"c001"' in html
        assert '"c002"' in html
        assert "candidates/c001.jpg" in html
        assert "candidates/c002.jpg" in html

    def test_self_contained_no_external_resources(self) -> None:
        html = render_html("20260805_对手队", _candidates())
        assert "http://" not in html
        assert "https://" not in html

    def test_session_in_title_and_export_name(self) -> None:
        html = render_html("20260805_对手队", _candidates())
        assert "20260805_对手队" in html
        assert "photo_selections.json" in html

    def test_export_payload_contract(self) -> None:
        # 导出物必须与 rank_photos --apply 的 schema 对齐：session + selected 列表
        html = render_html("20260805_对手队", _candidates())
        assert "session" in html
        assert "selected" in html

    def test_waterfall_and_shortcuts(self) -> None:
        html = render_html("20260805_对手队", _candidates())
        assert "column-count" in html  # CSS columns 瀑布流
        assert "localStorage" in html  # 进度持久化
        assert "apply" in html  # 页面内含 apply 命令提示

    def test_score_shown_on_card(self) -> None:
        html = render_html("20260805_对手队", _candidates())
        assert "0.81" in html


class TestBuildPageData:
    """页面数据装配：只收 status=ok 且有图的候选，键齐全。"""

    def test_filters_non_ok(self) -> None:
        cands = _candidates()
        cands.append({"id": "c003", "status": "dropped", "image": ""})
        data = build_page_data(cands)
        assert [d["id"] for d in data] == ["c001", "c002"]

    def test_fields_passed_through(self) -> None:
        data = build_page_data(_candidates())
        first = data[0]
        assert first["v"] == 1
        assert first["t"] == 12.3
        assert first["img"] == "candidates/c001.jpg"
