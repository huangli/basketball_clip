"""mot_candidates.run_mot 单元测试（飞行段放宽门限，docs/heatmap-flight-link/）。

覆盖：飞行球（100px/帧）默认 80px 门限碎成单点轨迹、max_match_dist=250 链成
长轨迹、静止球默认参数链接结果不变（run_mot 默认路径的回归锁——此前零测试覆盖）。
"""

from mot_candidates import Detection, run_mot


def _ball(cx: int, cy: int, fi: int) -> Detection:
    """构造球检测（box 取球心 ±10px；sec 按 5fps 换算）。"""
    return Detection(
        conf=0.9,
        box=[cx - 10, cy - 10, cx + 10, cy + 10],
        cx=cx,
        cy=cy,
        sec=fi / 5,
        frame_idx=fi,
    )


def _flight_frames(step: int = 100, n: int = 10) -> list[list[Detection]]:
    """n 帧飞行球：每帧位移 step px，每帧一个检测。"""
    return [[_ball(100 + i * step, 500, i)] for i in range(n)]


class TestRunMotMatchDist:
    """匹配门限参数化：默认 80px 候选挖掘口径不变；放宽门限供落点链路使用。"""

    def test_flight_fragments_at_default(self) -> None:
        # Arrange：飞行球每帧 100px，超默认 80px 门限
        # Act
        tracks = run_mot(_flight_frames(), min_length=1)
        # Assert：全部碎成单点轨迹（落点链路因此断粮，历史素材 no_landing 根因）
        assert tracks
        assert all(t.length == 1 for t in tracks)

    def test_flight_links_with_relaxed_gate(self) -> None:
        # Arrange / Act
        tracks = run_mot(_flight_frames(), min_length=1, max_match_dist=250)
        # Assert：链成 1 条 10 点长轨迹
        assert len(tracks) == 1
        assert tracks[0].length == 10

    def test_static_ball_default_unchanged(self) -> None:
        # Arrange：静止球 10 帧同位
        frames = [[_ball(300, 300, i)] for i in range(10)]
        # Act：默认参数（min_length=STATIC_WINDOW=4，门限 80）
        tracks = run_mot(frames)
        # Assert：链成 1 条 10 点轨迹（候选挖掘口径回归锁）
        assert len(tracks) == 1
        assert tracks[0].length == 10
