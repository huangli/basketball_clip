"""人脸 matcher：照片库人脸 embedding 1:N 高置信预填（photo-roster T11，L1=人脸单路）。

⚠️ 默认停用（2026-08-29 用户定纯人工，docs/photo-roster/review05.md）：用户
    全场真值评测——覆盖率 11.4%、采纳误指认率 100%（4/4，红线 ≤10% 踩破）；
    第一根因 = 错人框（裁图裁到旁观队友而非进球者，t173.8 抽帧实锤），人脸模型
    在错框上结构性无解。video.py people ②.5 默认不串联本模块（--photo-match
    显式开可恢复）；--evaluate 评测器保留可用。

选型依据（2026-08-28 用户拍板，docs/photo-roster/review04.md）：L1 = 人脸单路
高置信档（insightface buffalo_l）。spike 结论（work/spike_face/report.md）：
raw top-1 误指认率 50% 不可用，sim≥0.40 档 16 球小样 2/8 采纳全对、0 误指认、
对方 0/4 误中——高精度低覆盖工作点，符合"高置信预填、其余落人裁"架构；
注册净化必需（背景路人曾被注册成 57 号）；多帧一致至多次级参考，采纳只看
绝对 sim 高档（故本 matcher 球级聚合 = best 帧，不多帧投票）。

输入：--photos 照片库目录（``photos/<号码>/*.jpg|jpeg|png``，号码归一化/校验复用
    photo_match_scorers.scan_gallery，契约同 v1 不变）；--candidates 可重复
    （scorer_candidates.json，键集并集、同 key 后者覆盖，复用
    cluster_scorers.merge_candidates）。
输出：--out 写 photo_matches.json（现 schema 不变：version=photo-match-v1、
    model/threshold/margin 顶层、matches:{key:{number,score,margin}}）——
    高置信命中入 matches（score=best sim 实测值、margin=top1-top2 分差实测），
    低置信/无脸/无命中不入 matches（确认页全量列球天然进页面）；产物落盘前过
    photo_match_scorers.validate_matches_payload 校验（误指认红线：宁可漏不可错）。
    注册审计落 ``<photos>/face_registration_audit.json``（kept/dropped+原因+尺寸，
    同 work/spike_face/registration_audit.json 口径）。
--evaluate 级联评测（photo-roster T13）：另加 --roster 与 --goals（真值来源），
    --out 改写 markdown 报告——全部入统球 top-1+score 分布（不过闸，供阈值定稿）
    + 三指标（覆盖率 = 机器高置信采纳球数/入统球数；采纳误指认率 = 采纳球中
    top-1 与真值不符的比例，真值无号球被采纳同计误指认；人裁负担 = 1-覆盖率）
    + 按号码混淆矩阵。评测零模型零网络：注册库与裁图 embedding 全部读缓存
    重算余弦（纯 numpy），缓存文件缺失显式报错提示先跑匹配，缓存条目缺失的球
    WARNING 跳过不阻塞；真值映射/入统口径沿用 v1（复用 photo_match_scorers 的
    classify_truth / load_confirmed_goal_keys）。
缓存：照片注册 + 裁图脸 embedding 按 ``模型tag:文件md5`` 幂等缓存（仿
    clip_cache 模式，键含模型版本）：照片缓存落 ``<photos>/.face_cache.json``，
    裁图缓存落各批 ``<candidates 同目录>/face_cache.json``；检不出脸（None）也
    缓存，重跑零重算。
依赖：numpy、insightface + onnxruntime（buffalo_l 权重，非商业许可，仅在
    build_face_detector 内惰性 import，测试注入假识别器不碰真模型真权重）、
    scripts/cluster_scorers.py、scripts/photo_match_scorers.py、
    scripts/pipe_common.py、scripts/errors.py。
典型调用：
    python scripts/face_match_scorers.py --photos photos \
        --candidates work/<场次>/scorers_b1/scorer_candidates.json \
        --out work/<场次>/scorers_b1/photo_matches.json

写死口径（spec §数据契约 + spike 产品化教训）：
- 注册净化：每张照片检脸取**最大面积**脸且宽 ≥MIN_FACE_WIDTH，否则该照片弃用
  （WARNING 留痕入审计）；某号码全灭 WARNING 不阻塞；全库无效显式报错。
- 比对：每球 crops 逐张检脸（取最大脸）对注册库 1:N 余弦 top-1；检不出脸的裁图
  跳过记数；球级采纳 = best 帧 sim ≥THRESHOLD；best 帧两号码并列最高不采纳
  （保守交人裁）。
- 单球失败记 ERROR 跳过不炸批；缓存 schema 损坏显式失败（rules.md §0.2）。
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cluster_scorers import STATUS_OK, GoalCrops, file_md5, l2_normalize, merge_candidates
from errors import BasketballPipelineError, SchemaError
from photo_match_scorers import (
    MATCH_VERSION,
    NO_TOP1_LABEL,
    TRUTH_NO_NUMBER,
    TRUTH_OURS,
    TRUTH_UNJUDGEABLE,
    Gallery,
    Truth,
    classify_truth,
    load_confirmed_goal_keys,
    scan_gallery,
    validate_matches_payload,
)
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import Roster, validate_roster

logger = logging.getLogger(__name__)

MODEL_TAG: str = "insightface/buffalo_l"  # 模型标识（缓存键前缀，模型升级即换自然作废）
# 采纳闸：spike 观察值（16 球小样 sim≥0.40 档 2/8 采纳全对、0 误指认、对方 0/4
# 误中，work/spike_face/report.md）——非定稿，Phase A 分布后定稿 Ask first
# （docs/photo-roster/spec.md §Boundaries）。
THRESHOLD: float = 0.40
MIN_FACE_WIDTH: int = 120  # 注册净化尺寸闸（spike curated 口径：最大面积脸宽 ≥120px 才注册）
# 评测达标线（spec §Objective，用户可改）：采纳误指认率 ≤10%（红线）
ADOPT_ERROR_PASS_MAX: float = 0.10

PHOTO_FACE_CACHE_NAME: str = ".face_cache.json"  # 照片注册缓存（落 --photos 目录下）
FACE_CACHE_NAME: str = "face_cache.json"  # 裁图脸缓存（落各批 candidates 同目录）
REGISTRATION_AUDIT_NAME: str = "face_registration_audit.json"  # 注册审计（落 --photos 目录下）

# 缓存条目：检出的最大面积脸 {"w","h","det","emb"(L2 归一)}；None = 检不出脸（也缓存）
FaceCache = dict[str, dict[str, Any] | None]
# 人脸检测器：图像路径 → 检出脸列表（测试注入假识别器，不碰真模型）
FaceDetector = Callable[[Path], "list[DetectedFace]"]
DetectorFactory = Callable[[], FaceDetector]
# 单球处理可捕获的失败类型（检测异常已在 build_face_detector 内转为项目异常）
_GOAL_FAILURES = (BasketballPipelineError, OSError, ValueError, RuntimeError)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """一张检出脸（识别器返回值；embedding 未归一由被调方统一 L2 归一）。"""

    width: int  # bbox 宽（像素）
    height: int  # bbox 高（像素）
    det_score: float  # 检测置信度（仅留痕，不作注册依据——spike 教训：det 高≠主体）
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class CropHit:
    """一张裁图的 1:N top-1 结果。number=None 表示并列最高不采纳。"""

    number: str | None
    sim: float  # top-1 余弦相似度
    margin: float  # top-1 与次高分差；单号码库为 +inf


@dataclass(frozen=True, slots=True)
class GoalMatch:
    """一球的比对结果（best 帧口径）。number=None = 并列/无脸/无命中不采纳。"""

    number: str | None
    score: float | None  # best 帧 sim（全部裁图无脸为 None）
    margin: float | None  # best 帧 top1-top2 分差
    n_crops: int
    n_face: int  # 检出脸的裁图数
    n_no_face: int  # 检不出脸跳过的裁图数
    n_missing: int  # 文件缺失跳过的裁图数


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """一张注册照片的审计记录（kept/dropped + 原因 + 尺寸）。"""

    number: str  # 去零号码
    file: str  # 照片文件名
    kept: bool
    reason: str | None  # 弃用原因（kept=True 为 None）
    face_w: int | None  # 最大面积脸宽（弃用也留尺寸；检不出脸为 None）
    face_h: int | None
    det_score: float | None


@dataclass(frozen=True, slots=True)
class EvalRow:
    """一个入统球的评测行（--evaluate 用；top-1 不过闸全量列出，供阈值定稿）。"""

    key: str
    category: str  # TRUTH_OURS / TRUTH_NO_NUMBER / TRUTH_UNJUDGEABLE
    truth_number: str | None
    top_number: str | None  # None = 无脸/并列/未出分（缓存缺失跳过）
    score: float | None  # best 帧 sim（未出分为 None）
    margin: float | None  # best 帧 top1-top2 分差
    adopted: bool  # 是否过闸（best sim ≥ THRESHOLD 且非并列/非无脸）
    correct: bool | None  # 仅有号球被采纳时有意义：top-1 == 真值号码


@dataclass(frozen=True, slots=True)
class EvalReport:
    """--evaluate 评测报告数据。比率为 None 表示分母为 0（待定）。"""

    rows: tuple[EvalRow, ...]
    n_ours: int  # 半截篮有号球数
    n_no_number: int  # 真值无号球数（对方/便服）
    n_unjudgeable: int  # 不可判球数（半截篮无号 tag）
    n_skipped: int  # 未出分球数（缓存缺失/不在 candidates，WARNING 跳过）
    n_adopted: int  # 机器高置信采纳球数
    n_adopted_judged: int  # 误指认率分母（采纳的有号+无号球；不可判不进）
    n_errors: int  # 误指认球数（有号错号 + 真值无号被采纳）
    n_adopted_unjudgeable: int  # 不可判球被采纳数（单列展示，不进误指认率分母）
    coverage: float | None  # 覆盖率 = n_adopted / 入统球数
    adopt_error_rate: float | None  # 采纳误指认率 = n_errors / n_adopted_judged
    human_burden: float | None  # 人裁负担 = 1 - coverage（未采纳球占比）
    confusion: dict[str, dict[str, int]]  # 真值号码 → top-1 号码（不过闸）→ 球数


def _imread_unicode(path: Path) -> np.ndarray:
    """读图（np.fromfile+cv2.imdecode，与 rank_photos._read_image 同一写法）。

    Windows 中文路径下 ``cv2.imread`` 静默返回 None（项目既有教训，
    rank_photos.py:769 docstring）——照片库文件名用户自由命名（中文合法）、
    中文 fid 场次裁图路径同样中招，故统一走 imdecode 路径。cv2 惰性 import
    （同 build_face_detector，模块加载不拉重依赖）。

    Args:
        path: 图像路径（可含中文等非 ASCII 字符）。

    Returns:
        BGR 图像数组。

    Raises:
        BasketballPipelineError: 文件不可读（OSError）/ 解码失败（非图像字节）。
    """
    import cv2

    try:
        buf: np.ndarray = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        raise BasketballPipelineError(f"图片不可读: {path}: {e}") from e
    img: np.ndarray | None = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise BasketballPipelineError(f"图片解码失败: {path}")
    return img


def build_face_detector() -> FaceDetector:
    """加载 insightface buffalo_l 并返回检测器（CPU）。

    insightface/onnxruntime/cv2 只在本函数内惰性 import（模型加载重、buffalo_l
    权重为非商业许可），测试注入假识别器不经过这里。权重走 insightface 默认
    缓存路径（spike 已备好）。图像不可读 / 检测失败统一转换为
    BasketballPipelineError（rules.md §3.2：捕获后转换为项目异常）。

    Returns:
        检测器：图像路径 → 检出脸列表（embedding 未归一）。

    Raises:
        BasketballPipelineError: 模型/权重加载失败（首次调用时抛出）。
    """
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        raise BasketballPipelineError(f"insightface buffalo_l 加载失败: {e}") from e

    def detect(path: Path) -> list[DetectedFace]:
        img: np.ndarray = _imread_unicode(path)  # 中文路径安全（imread 静默 None 教训）
        try:
            faces = app.get(img)
        except Exception as e:
            raise BasketballPipelineError(f"人脸检测失败: {path}: {e}") from e
        return [
            DetectedFace(
                width=int(f.bbox[2] - f.bbox[0]),
                height=int(f.bbox[3] - f.bbox[1]),
                det_score=float(f.det_score),
                embedding=np.asarray(f.embedding, dtype=np.float64),
            )
            for f in faces
        ]

    return detect


def load_face_cache(path: Path) -> FaceCache:
    """读 face_cache.json（幂等缓存）；缺失 → 空；schema 损坏 → SchemaError。

    缓存键 = ``<model_tag>:<文件 md5>``（仿 clip_cache 模式，模型前缀天然隔离）；
    值 = 最大面积脸条目 ``{"w","h","det","emb"}`` 或 None（检不出脸，也缓存）。

    Args:
        path: face_cache.json 路径。

    Returns:
        缓存条目 dict（原样返回，调用方按需取用）。

    Raises:
        SchemaError: 顶层非对象 / 缺 entries / 条目结构损坏（rules.md §0.2：
            数据损坏必须停）。
    """
    if not path.is_file():
        return {}
    data: Any = read_json(path, what="face_cache.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    entries: Any = data.get("entries")
    if not isinstance(entries, dict):
        raise SchemaError(f"{path}: 缺 entries 对象或类型错误，实际 {type(entries).__name__}")
    for key, value in entries.items():
        if not isinstance(key, str):
            raise SchemaError(f"{path}: 缓存键不是 str: {key!r}")
        if value is None:
            continue
        if not isinstance(value, dict):
            raise SchemaError(f"{path}: 条目 {key!r} 不是对象/None，实际 {type(value).__name__}")
        w: Any = value.get("w")
        h: Any = value.get("h")
        # bool 是 int 子类，与 det 同口径排除（True 不是合法像素宽）
        if (
            isinstance(w, bool)
            or isinstance(h, bool)
            or not isinstance(w, int)
            or not isinstance(h, int)
        ):
            raise SchemaError(f"{path}: 条目 {key!r} w/h 缺失或不是 int")
        det: Any = value.get("det")
        if isinstance(det, bool) or not isinstance(det, (int, float)):
            raise SchemaError(f"{path}: 条目 {key!r} det 不是数值")
        emb: Any = value.get("emb")
        if (
            not isinstance(emb, list)
            or not emb
            or not all(isinstance(x, (int, float)) for x in emb)
        ):
            raise SchemaError(f"{path}: 条目 {key!r} emb 缺失或不是数值列表")
    return dict(entries)


def save_face_cache(path: Path, model_tag: str, cache: FaceCache) -> None:
    """原子写 face_cache.json（_meta 记录模型与时间戳；先写 tmp 校验后 replace）。

    Args:
        path: 缓存文件路径（父目录自动创建）。
        model_tag: 模型标识（落 _meta 供追溯；隔离靠键前缀）。
        cache: 缓存条目 dict。
    """
    payload: dict[str, Any] = {
        "_meta": {"model": model_tag, "updated": datetime.now().isoformat(timespec="seconds")},
        "entries": cache,
    }
    atomic_write_json(path, payload, what="face_cache.json")


def _entry_to_face(entry: dict[str, Any]) -> DetectedFace:
    """缓存条目 → DetectedFace（load_face_cache 已校验结构）。"""
    return DetectedFace(
        width=int(entry["w"]),
        height=int(entry["h"]),
        det_score=float(entry["det"]),
        embedding=np.asarray(entry["emb"], dtype=np.float64),
    )


def _picked_face(
    path: Path,
    cache: FaceCache,
    model_tag: str,
    detector: FaceDetector | None,
    detector_factory: DetectorFactory,
) -> tuple[DetectedFace | None, FaceDetector | None]:
    """取一张图的最大面积脸（缓存优先；检不出脸返回 None 并缓存 None）。

    缓存未命中时才惰性构建检测器（全命中零加载模型）。新检出脸的 embedding
    写入缓存前已 L2 归一（缓存命中的向量视为已归一）。

    Args:
        path: 图像路径。
        cache: 本目录 face 缓存（原地写入）。
        model_tag: 模型标识（缓存键前缀）。
        detector: 已构建的检测器（None = 尚未需要）。
        detector_factory: 零参工厂，首次缓存未命中时才调用。

    Returns:
        (最大面积脸或 None, 检测器（如本次构建）)。
    """
    cache_key: str = f"{model_tag}:{file_md5(path)}"
    if cache_key in cache:
        cached: dict[str, Any] | None = cache[cache_key]
        return (_entry_to_face(cached) if cached is not None else None), detector
    if detector is None:
        detector = detector_factory()
    faces: list[DetectedFace] = detector(path)
    if not faces:
        cache[cache_key] = None
        return None, detector
    face: DetectedFace = max(faces, key=lambda f: f.width * f.height)
    vec: np.ndarray = l2_normalize(face.embedding)
    cache[cache_key] = {
        "w": face.width,
        "h": face.height,
        "det": face.det_score,
        "emb": [float(x) for x in vec],
    }
    return (
        DetectedFace(face.width, face.height, face.det_score, vec),
        detector,
    )


def register_gallery(
    gallery: Gallery,
    detector_factory: DetectorFactory,
    cache: FaceCache,
    model_tag: str,
) -> tuple[dict[str, tuple[np.ndarray, ...]], list[AuditEntry]]:
    """注册净化（spike 教训产品化）：每张照片取最大面积脸且宽 ≥MIN_FACE_WIDTH 才注册。

    无脸 / 小脸 / 检测异常 → 该照片弃用（WARNING 留痕入审计，不阻塞）；某号码
    照片全灭 → WARNING 不阻塞；全库无可用注册 → 显式报错（不静默空跑）。

    Args:
        gallery: photo_match_scorers.scan_gallery 产物（键 = 去零号码）。
        detector_factory: 检测器零参工厂（测试注入假识别器）。
        cache: 照片注册缓存（原地写入新条目）。
        model_tag: 模型标识（缓存键前缀）。

    Returns:
        (去零号码 → 注册向量元组（L2 归一）, 逐照片审计记录)。

    Raises:
        BasketballPipelineError: 全部号码无可用注册（照片库无效属配置错误）。
    """
    vectors: dict[str, tuple[np.ndarray, ...]] = {}
    audit: list[AuditEntry] = []
    detector: FaceDetector | None = None
    for number, paths in gallery.photos.items():
        vecs: list[np.ndarray] = []
        for path in paths:
            face: DetectedFace | None = None
            reason: str | None = None
            try:
                face, detector = _picked_face(path, cache, model_tag, detector, detector_factory)
                if face is None:
                    reason = "no face"
                elif face.width < MIN_FACE_WIDTH:
                    reason = f"no big frontal face (max w={face.width} < {MIN_FACE_WIDTH})"
            except _GOAL_FAILURES as e:
                reason = f"detection error: {e}"
                face = None
            if reason is not None:
                logger.warning("注册照片弃用: %s/%s: %s", number, path.name, reason)
                audit.append(
                    AuditEntry(
                        number=number,
                        file=path.name,
                        kept=False,
                        reason=reason,
                        face_w=face.width if face else None,
                        face_h=face.height if face else None,
                        det_score=face.det_score if face else None,
                    )
                )
                continue
            if face is None:  # 防御：reason 为 None 时 face 必非 None（逻辑错误显式失败）
                raise BasketballPipelineError(f"注册逻辑错误: {path}")
            audit.append(
                AuditEntry(
                    number=number,
                    file=path.name,
                    kept=True,
                    reason=None,
                    face_w=face.width,
                    face_h=face.height,
                    det_score=face.det_score,
                )
            )
            vecs.append(face.embedding)
        if vecs:
            vectors[number] = tuple(vecs)
        else:
            logger.warning("号码 %s 照片全部弃用（无合格正脸照），不注册该号码", number)
    if not vectors:
        raise BasketballPipelineError("照片库全部号码无可用注册（无合格正脸照，配置错误）")
    return vectors, audit


def audit_payload(audit: list[AuditEntry], model_tag: str) -> dict[str, Any]:
    """组装注册审计落盘载荷（同 work/spike_face/registration_audit.json 口径）。

    Args:
        audit: register_gallery 审计记录。
        model_tag: 模型标识（落 model 字段）。

    Returns:
        可 JSON 序列化的载荷：``{model, protocol, photos:[{number,file,kept,
        picked:{w,h,det}|reason}]}``。
    """
    photos: list[dict[str, Any]] = []
    for a in audit:
        rec: dict[str, Any] = {"number": a.number, "file": a.file, "kept": a.kept}
        if a.kept:
            rec["picked"] = {"w": a.face_w, "h": a.face_h, "det": a.det_score}
        else:
            rec["reason"] = a.reason
            if a.face_w is not None:
                rec["dropped_face"] = {"w": a.face_w, "h": a.face_h, "det": a.det_score}
        photos.append(rec)
    return {
        "model": model_tag,
        "protocol": f"curated: max-area face, width >= {MIN_FACE_WIDTH}px",
        "photos": photos,
    }


def score_crop(
    embedding: np.ndarray,
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, float]:
    """裁图-号码得分：``sim(裁图, 号码) = max(该号码注册向量)`` 余弦（向量均已归一）。

    Args:
        embedding: 裁图脸单位向量。
        gallery_vecs: register_gallery 产物。

    Returns:
        去零号码 → 相似度。
    """
    return {
        number: max(float(np.dot(embedding, v)) for v in vecs)
        for number, vecs in gallery_vecs.items()
    }


def top_hit(sims: dict[str, float]) -> CropHit:
    """取 top-1 号码与次高分差（误指认红线：并列最高不采纳，保守交人裁）。

    并列判定用浮点精确相等；库内仅 1 个号码时 margin = +inf（无次高可比）。

    Args:
        sims: score_crop 产物（去零号码 → 相似度），非空。

    Returns:
        CropHit。

    Raises:
        BasketballPipelineError: sims 为空（注册库为空，上游逻辑错误显式失败）。
    """
    if not sims:
        raise BasketballPipelineError("sim 字典为空（注册库为空？上游逻辑错误）")
    best: float = max(sims.values())
    winners: list[str] = [n for n, s in sims.items() if s == best]
    if len(winners) > 1:
        return CropHit(number=None, sim=best, margin=0.0)
    if len(sims) == 1:
        return CropHit(number=winners[0], sim=best, margin=math.inf)
    second: float = max(s for n, s in sims.items() if n != winners[0])
    return CropHit(number=winners[0], sim=best, margin=best - second)


def match_goal(
    goal: GoalCrops,
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
    cache: FaceCache,
    model_tag: str,
    detector_factory: DetectorFactory,
) -> GoalMatch:
    """单球比对：crops 逐张检脸 1:N top-1，球级聚合 = best 帧（sim 最高帧）。

    spike 实测单帧最优与多帧质量加权投票预测完全一致（16 球两注册口径），且
    "多帧一致 ⇒ 正确"不成立，故不实现投票——采纳只看 best 帧绝对 sim
    （work/spike_face/report.md §阈值分布）。检不出脸的裁图跳过记数；裁图文件
    缺失 WARNING 跳过记数（容忍缺失，不阻塞其余裁图）。

    Args:
        goal: 一球裁图信息。
        gallery_vecs: register_gallery 产物。
        cache: 本批裁图 face 缓存（原地写入）。
        model_tag: 模型标识（缓存键前缀）。
        detector_factory: 检测器零参工厂。

    Returns:
        GoalMatch（best 帧口径；全部裁图无脸时 number/score/margin 为 None）。
    """
    detector: FaceDetector | None = None
    best: CropHit | None = None
    n_face = n_no_face = n_missing = 0
    for name in goal.crops:
        path: Path = goal.base_dir / name
        if not path.is_file():
            logger.warning("裁图文件缺失，跳过该张: %s (%s)", path, goal.key)
            n_missing += 1
            continue
        face, detector = _picked_face(path, cache, model_tag, detector, detector_factory)
        if face is None:
            logger.debug("裁图检不出脸，跳过: %s (%s)", name, goal.key)
            n_no_face += 1
            continue
        n_face += 1
        hit: CropHit = top_hit(score_crop(face.embedding, gallery_vecs))
        if best is None or hit.sim > best.sim:
            best = hit
    if best is None:
        return GoalMatch(
            number=None,
            score=None,
            margin=None,
            n_crops=len(goal.crops),
            n_face=n_face,
            n_no_face=n_no_face,
            n_missing=n_missing,
        )
    return GoalMatch(
        number=best.number,
        score=best.sim,
        margin=best.margin,
        n_crops=len(goal.crops),
        n_face=n_face,
        n_no_face=n_no_face,
        n_missing=n_missing,
    )


def match_goals(
    goals: dict[str, GoalCrops],
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
    detector_factory: DetectorFactory,
    model_tag: str,
) -> dict[str, GoalMatch]:
    """批量比对：逐球出分（不过闸，闸在 adopt/payload 侧），单球失败 ERROR 不炸批。

    只对 status=OK 且有裁图的球出分（SKIP/无裁图跳过不记 WARNING，与聚类
    unclustered 口径一致）。裁图缓存按 candidates 所在目录（base_dir）分组，
    首触即载、末尾统一落盘；缓存 schema 损坏在 try 外加载，显式失败不降级为
    单球跳过（rules.md §0.2：数据损坏必须停）。

    Args:
        goals: merge_candidates 产物。
        gallery_vecs: register_gallery 产物。
        detector_factory: 检测器零参工厂（全缓存命中不触发）。
        model_tag: 模型标识。

    Returns:
        key → GoalMatch（失败球/SKIP 球不在内）。
    """
    caches: dict[Path, FaceCache] = {}
    results: dict[str, GoalMatch] = {}
    # 检测器提升到批级（同 register_gallery / cluster_scorers.embed_goals 模式）：
    # 球级局部变量会让每个缓存未命中的球都触发一次模型全量加载（首跑 N 球 = N 次）；
    # memoize 保持惰性——全部缓存命中时 factory 零调用、零加载。
    detector: FaceDetector | None = None

    def _shared_factory() -> FaceDetector:
        nonlocal detector
        if detector is None:
            detector = detector_factory()
        return detector

    for key, goal in goals.items():
        if goal.status != STATUS_OK or not goal.crops:
            continue
        if goal.base_dir not in caches:
            caches[goal.base_dir] = load_face_cache(goal.base_dir / FACE_CACHE_NAME)
        cache: FaceCache = caches[goal.base_dir]
        try:
            results[key] = match_goal(goal, gallery_vecs, cache, model_tag, _shared_factory)
        except _GOAL_FAILURES as e:
            logger.error("单球比对失败，跳过不炸批: %s: %s", key, e, exc_info=True)
    for base_dir, cache in caches.items():
        save_face_cache(base_dir / FACE_CACHE_NAME, model_tag, cache)
    return results


def adopt(m: GoalMatch, threshold: float) -> bool:
    """采纳闸（高置信档）：best sim ≥ threshold（含等于）且非并列/非无脸。

    Args:
        m: match_goal 产物。
        threshold: 相似度下限（含等于）。

    Returns:
        True = 高置信命中可入 photo_matches.json。
    """
    return m.number is not None and m.score is not None and m.score >= threshold


def build_matches_payload(
    results: dict[str, GoalMatch],
    threshold: float,
    model_tag: str,
) -> dict[str, Any]:
    """组装 photo_matches.json 载荷：仅高置信命中球（现 schema 不变，误指认红线）。

    映射（spec §数据契约）：score=best sim 实测值、margin=top1-top2 分差实测；
    低置信/无脸/并列不入 matches（确认页全量列球天然进页面）。顶层 margin=0.0
    占位——人脸单路不设 margin 闸（spike：margin 单独不可切），字段保留仅为
    schema 兼容（version=photo-match-v1 写死）。

    Args:
        results: match_goals 产物（未过闸全量）。
        threshold: 本次采纳闸（落盘供标定追溯）。
        model_tag: 模型标识（落盘 model 字段）。

    Returns:
        可 JSON 序列化的载荷 dict（matches 按 results 键序）。
    """
    matches: dict[str, Any] = {}
    for key, m in results.items():
        if not adopt(m, threshold):
            continue
        if m.number is None or m.score is None or m.margin is None:  # 防御：adopt 已排除
            raise BasketballPipelineError(f"过闸球缺 top-1 号码/分数（逻辑错误）: {key}")
        matches[key] = {"number": m.number, "score": m.score, "margin": m.margin}
    return {
        "version": MATCH_VERSION,
        "model": model_tag,
        "threshold": threshold,
        "margin": 0.0,
        "matches": matches,
    }


def load_cached_registry(photos_dir: Path, model_tag: str) -> dict[str, tuple[np.ndarray, ...]]:
    """零模型重建注册库（--evaluate 用）：扫描照片库，embedding 全部读照片 face_cache。

    净化口径与 register_gallery 一致：无脸条目（None）/ 小脸（宽 <MIN_FACE_WIDTH）
    不注册；缓存条目缺失的照片（注册后新增）WARNING 跳过不阻塞；某号码全灭
    WARNING 不阻塞；全库无可用注册显式报错。缓存文件缺失 = 尚未跑过匹配，显式
    报错提示先跑匹配（rules.md §0.2：不静默空跑）。

    Args:
        photos_dir: 照片库目录（``photos/<号码>/``）。
        model_tag: 模型标识（缓存键前缀）。

    Returns:
        去零号码 → 注册向量元组（缓存写入时已 L2 归一，命中向量视为已归一）。

    Raises:
        BasketballPipelineError: 照片 face_cache 文件缺失（提示先跑匹配）/
            全部号码无可用注册。
    """
    cache_path: Path = photos_dir / PHOTO_FACE_CACHE_NAME
    if not cache_path.is_file():
        raise BasketballPipelineError(
            f"照片注册缓存缺失: {cache_path}（请先跑 face_match_scorers 匹配产出该缓存）"
        )
    gallery: Gallery = scan_gallery(photos_dir)
    cache: FaceCache = load_face_cache(cache_path)
    vectors: dict[str, tuple[np.ndarray, ...]] = {}
    for number, paths in gallery.photos.items():
        vecs: list[np.ndarray] = []
        for path in paths:
            if not path.is_file():
                logger.warning("照片扫描后消失，跳过: %s (号码 %s)", path, number)
                continue
            cache_key: str = f"{model_tag}:{file_md5(path)}"
            if cache_key not in cache:
                logger.warning("照片无缓存条目（注册后新增？），跳过: %s (号码 %s)", path, number)
                continue
            entry: dict[str, Any] | None = cache[cache_key]
            if entry is None or int(entry["w"]) < MIN_FACE_WIDTH:
                continue  # 无脸/小脸不注册（净化口径同 register_gallery）
            vecs.append(np.asarray(entry["emb"], dtype=np.float64))
        if vecs:
            vectors[number] = tuple(vecs)
        else:
            logger.warning("号码 %s 无可用注册（缓存内无合格正脸），跳过该号码", number)
    if not vectors:
        raise BasketballPipelineError(f"照片库全部号码无可用注册: {photos_dir}")
    return vectors


def score_goal_cached(
    goal: GoalCrops,
    gallery_vecs: dict[str, tuple[np.ndarray, ...]],
    cache: FaceCache,
    model_tag: str,
) -> GoalMatch | None:
    """零模型单球出分（--evaluate 用）：crops 逐张查缓存 embedding 重算余弦，聚合 = best 帧。

    无脸缓存条目（None）跳过记数；裁图文件缺失 / 缓存条目缺失 → 返回 None
    （调用方 WARNING 跳过该球不阻塞）。并列最高 number=None（保守交人裁，
    与 match_goal 同口径）。

    Args:
        goal: 一球裁图信息。
        gallery_vecs: load_cached_registry 产物。
        cache: 本批裁图 face 缓存（只读）。
        model_tag: 模型标识（缓存键前缀）。

    Returns:
        GoalMatch（best 帧口径）；缓存不完整无法出分时为 None。
    """
    best: CropHit | None = None
    n_face = n_no_face = 0
    for name in goal.crops:
        path: Path = goal.base_dir / name
        if not path.is_file():
            return None
        cache_key: str = f"{model_tag}:{file_md5(path)}"
        if cache_key not in cache:
            return None
        entry: dict[str, Any] | None = cache[cache_key]
        if entry is None:
            n_no_face += 1
            continue
        n_face += 1
        hit: CropHit = top_hit(score_crop(np.asarray(entry["emb"], dtype=np.float64), gallery_vecs))
        if best is None or hit.sim > best.sim:
            best = hit
    if best is None:
        return GoalMatch(
            number=None,
            score=None,
            margin=None,
            n_crops=len(goal.crops),
            n_face=n_face,
            n_no_face=n_no_face,
            n_missing=0,
        )
    return GoalMatch(
        number=best.number,
        score=best.sim,
        margin=best.margin,
        n_crops=len(goal.crops),
        n_face=n_face,
        n_no_face=n_no_face,
        n_missing=0,
    )


def evaluate(
    results: dict[str, GoalMatch],
    truth: dict[str, Truth],
    scope_keys: list[str],
    threshold: float,
) -> EvalReport:
    """级联评测三指标（spec §评估口径 + T13 契约）：top-1 不过闸全量入报告，指标看过闸。

    指标口径：
    - 覆盖率 = 机器高置信采纳球数 / 入统球数（全部入统球，含不可判——未采纳
      的不可判球同样进确认页占人裁）；
    - 采纳误指认率 = 采纳球中 top-1 与真值不符的比例（有号错号 + 真值无号球
      被采纳同计误指认；不可判球可能是照片库成员、正确命中不该计误，沿用 v1
      "不可判不进分母"口径，被采纳数单列展示）；
    - 人裁负担 = 1 - 覆盖率（未采纳球占比）。
    混淆矩阵按号码展开，用 top-1（不过闸），top-1 缺失（无脸/并列/未出分）
    归入 NO_TOP1_LABEL 列。

    Args:
        results: 入统球的缓存出分（未过闸全量；未出分球不在内按 n_skipped 计）。
        truth: classify_truth 产物（photo_match_scorers 复用件）。
        scope_keys: 入统键（goals confirmed 且 key ∈ roster.assignments），调用方算好。
        threshold: 采纳闸（现 THRESHOLD 单闸：best sim ≥ threshold）。

    Returns:
        EvalReport（比率为 None 表示分母为 0）。
    """
    rows: list[EvalRow] = []
    confusion: dict[str, dict[str, int]] = {}
    n_ours = n_no_number = n_unjudgeable = 0
    n_adopted = n_adopted_judged = n_errors = n_adopted_unjudgeable = 0
    for key in scope_keys:
        t: Truth = truth[key]
        m: GoalMatch | None = results.get(key)
        adopted: bool = m is not None and adopt(m, threshold)
        correct: bool | None = None
        if t.category == TRUTH_OURS:
            n_ours += 1
            col: str = (m.number if m is not None else None) or NO_TOP1_LABEL
            if t.number is not None:
                row_counts: dict[str, int] = confusion.setdefault(t.number, {})
                row_counts[col] = row_counts.get(col, 0) + 1
            if adopted and m is not None:
                correct = m.number == t.number
                n_adopted_judged += 1
                if not correct:
                    n_errors += 1
        elif t.category == TRUTH_NO_NUMBER:
            n_no_number += 1
            if adopted:
                n_adopted_judged += 1
                n_errors += 1  # 真值无号球被采纳 = 误指认
        else:
            n_unjudgeable += 1
            if adopted:
                n_adopted_unjudgeable += 1
        if adopted:
            n_adopted += 1
        rows.append(
            EvalRow(
                key=key,
                category=t.category,
                truth_number=t.number,
                top_number=m.number if m is not None else None,
                score=m.score if m is not None else None,
                margin=m.margin if m is not None else None,
                adopted=adopted,
                correct=correct,
            )
        )
    n_scope: int = len(scope_keys)
    coverage: float | None = (n_adopted / n_scope) if n_scope else None
    return EvalReport(
        rows=tuple(rows),
        n_ours=n_ours,
        n_no_number=n_no_number,
        n_unjudgeable=n_unjudgeable,
        n_skipped=n_scope - len([k for k in scope_keys if k in results]),
        n_adopted=n_adopted,
        n_adopted_judged=n_adopted_judged,
        n_errors=n_errors,
        n_adopted_unjudgeable=n_adopted_unjudgeable,
        coverage=coverage,
        adopt_error_rate=(n_errors / n_adopted_judged) if n_adopted_judged else None,
        human_burden=(1.0 - coverage) if coverage is not None else None,
        confusion=confusion,
    )


def _fmt_rate(value: float | None) -> str:
    """格式化比率：None（分母 0）→ ``待定``，否则百分比。"""
    if value is None:
        return "待定（分母 0）"
    return f"{value * 100:.1f}%"


def _fmt_float(value: float | None) -> str:
    """格式化得分/margin：None → ``-``，+inf → ``inf``，否则 4 位小数。"""
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


_CATEGORY_LABELS: dict[str, str] = {
    TRUTH_OURS: "半截篮有号",
    TRUTH_NO_NUMBER: "无号(对方/便服)",
    TRUTH_UNJUDGEABLE: "不可判(半截篮无号tag)",
}


def _verdict(row: EvalRow) -> str:
    """逐球行判定列：有号采纳→对/错；真值无号被采纳→误指；不可判被采纳→采纳；其余 -。"""
    if row.category == TRUTH_OURS and row.adopted:
        return "对" if row.correct else "错"
    if row.category == TRUTH_NO_NUMBER and row.adopted:
        return "误指"
    if row.category == TRUTH_UNJUDGEABLE and row.adopted:
        return "采纳"
    return "-"


def render_markdown(report: EvalReport, model_tag: str, threshold: float) -> str:
    """渲染 --evaluate markdown 评测报告（写 --out；阈值定稿与达标判定以此为准）。

    Args:
        report: evaluate 产物。
        model_tag: 模型标识。
        threshold: 本次采纳闸（best sim ≥ threshold 单闸）。

    Returns:
        markdown 文本（UTF-8，含参数/入统数/三指标/混淆矩阵/逐球 top-1+score 分布）。
    """
    lines: list[str] = [
        "# 人脸 L1 级联评测报告（photo-roster T13）",
        "",
        f"- 模型: `{model_tag}`",
        f"- 采纳闸: best sim ≥ {threshold}（现 THRESHOLD 单闸；分布见下供阈值定稿）",
        f"- 入统球: {len(report.rows)}（半截篮有号 {report.n_ours} / "
        f"无号 {report.n_no_number} / 不可判 {report.n_unjudgeable}；"
        f"缓存缺失未出分 {report.n_skipped}）",
        "",
        "## 三指标",
        "",
        f"- 覆盖率: {report.n_adopted}/{len(report.rows)} = {_fmt_rate(report.coverage)}"
        "（机器高置信采纳球 / 全部入统球）",
        f"- 采纳误指认率: {report.n_errors}/{report.n_adopted_judged} = "
        f"{_fmt_rate(report.adopt_error_rate)}（达标线 ≤ {ADOPT_ERROR_PASS_MAX * 100:.0f}%；"
        "真值无号球被采纳同计误指认）",
        f"- 人裁负担: {_fmt_rate(report.human_burden)}（= 1 - 覆盖率，未采纳球占比）",
        f"- 附: 不可判球被采纳 {report.n_adopted_unjudgeable} 个"
        "（可能是照片库成员，正确命中不计误，不进误指认率分母）",
        "",
        "## 混淆矩阵（真值号码 × top-1 号码，不过闸）",
        "",
    ]
    cols: list[str] = sorted({c for row in report.confusion.values() for c in row})
    lines.append("| 真值\\top-1 | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for truth_number in sorted(report.confusion):
        row = report.confusion[truth_number]
        lines.append(f"| {truth_number} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    lines += [
        "",
        "## 逐球分布（top-1 不过闸，供阈值定稿）",
        "",
        "| key | 真值类别 | 真值号码 | top-1 | score | margin | 采纳 | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        lines.append(
            f"| {r.key} | {_CATEGORY_LABELS.get(r.category, r.category)} | "
            f"{r.truth_number or '-'} | {r.top_number or NO_TOP1_LABEL} | "
            f"{_fmt_float(r.score)} | {_fmt_float(r.margin)} | "
            f"{'是' if r.adopted else '否'} | {_verdict(r)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_evaluate(args: argparse.Namespace) -> None:
    """--evaluate 分支：读真值 → 纯缓存出分（零模型）→ 三指标评估 → markdown 写 --out。

    裁图 face_cache 按 candidates 所在目录（base_dir）分组加载；缓存文件缺失
    显式报错提示先跑匹配（rules.md §0.2）；入统球不在 candidates / 缓存条目
    缺失 WARNING 跳过不阻塞（计未出分）。

    Args:
        args: CLI 参数（roster/goals/out 已校验非 None）。

    Raises:
        BasketballPipelineError: 照片/裁图缓存文件缺失、注册库无效。
    """
    roster_data: Any = read_json(args.roster, what="roster.json")
    roster: Roster = validate_roster(roster_data, str(args.roster))
    confirmed: set[str] = load_confirmed_goal_keys(args.goals)
    truth: dict[str, Truth] = classify_truth(roster)
    scope: list[str] = sorted(k for k in confirmed if k in truth)
    logger.info(
        "入统 %d 球（goals confirmed %d 键，roster assignments %d 键）",
        len(scope),
        len(confirmed),
        len(truth),
    )
    gallery_vecs: dict[str, tuple[np.ndarray, ...]] = load_cached_registry(args.photos, MODEL_TAG)
    logger.info("注册库（纯缓存重建）: %d 个号码 ← %s", len(gallery_vecs), args.photos)

    goals: dict[str, GoalCrops] = merge_candidates(args.candidates)
    caches: dict[Path, FaceCache] = {}
    results: dict[str, GoalMatch] = {}
    for key in scope:
        goal: GoalCrops | None = goals.get(key)
        if goal is None:
            logger.warning("入统球不在 candidates，跳过（计未出分）: %s", key)
            continue
        if goal.status != STATUS_OK or not goal.crops:
            logger.warning("入统球非 OK 或无裁图，跳过（计未出分）: %s (%s)", key, goal.status)
            continue
        if goal.base_dir not in caches:
            cache_path: Path = goal.base_dir / FACE_CACHE_NAME
            if not cache_path.is_file():
                raise BasketballPipelineError(
                    f"裁图脸缓存缺失: {cache_path}（请先跑 face_match_scorers 匹配产出该缓存）"
                )
            caches[goal.base_dir] = load_face_cache(cache_path)
        m: GoalMatch | None = score_goal_cached(
            goal, gallery_vecs, caches[goal.base_dir], MODEL_TAG
        )
        if m is None:
            logger.warning("入统球裁图缓存条目缺失，跳过不阻塞（计未出分）: %s", key)
            continue
        results[key] = m

    report: EvalReport = evaluate(results, truth, scope, THRESHOLD)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report, MODEL_TAG, THRESHOLD), encoding="utf-8")
    logger.info(
        "评测完成: 覆盖率 %s（%d/%d），采纳误指认率 %s（%d/%d），人裁负担 %s → %s",
        _fmt_rate(report.coverage),
        report.n_adopted,
        len(scope),
        _fmt_rate(report.adopt_error_rate),
        report.n_errors,
        report.n_adopted_judged,
        _fmt_rate(report.human_burden),
        args.out,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数（--evaluate 必须同时给 --roster 与 --goals，缺一报错）。"""
    parser = argparse.ArgumentParser(
        description="照片库认人 L1：人脸 embedding 1:N 高置信预填（insightface buffalo_l）"
    )
    parser.add_argument("--photos", required=True, type=Path, help="照片库目录（photos/<号码>/）")
    parser.add_argument(
        "--candidates",
        required=True,
        action="append",
        type=Path,
        help="scorer_candidates.json 路径（可重复；键集并集，同 key 后者覆盖前者）",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="输出路径：非 evaluate 写 photo_matches.json；--evaluate 写 markdown 评测报告",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="级联评测模式（需同时给 --roster 与 --goals；纯缓存零模型，报告写 --out）",
    )
    parser.add_argument("--roster", type=Path, default=None, help="roster.json（--evaluate 用）")
    parser.add_argument("--goals", type=Path, default=None, help="goals.json（--evaluate 用）")
    ns = parser.parse_args(argv)
    if ns.evaluate and (ns.roster is None or ns.goals is None):
        parser.error("--evaluate 需同时给 --roster 与 --goals")
    if not ns.evaluate and (ns.roster is not None or ns.goals is not None):
        parser.error("--roster/--goals 仅 --evaluate 模式使用")
    return ns


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=管线失败）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        if args.evaluate:
            _run_evaluate(args)  # 纯缓存零模型：不注册、不构建检测器
            return 0
        gallery: Gallery = scan_gallery(args.photos)
        logger.info(
            "照片库: %d 个号码 / %d 张照片 ← %s",
            len(gallery.photos),
            sum(len(p) for p in gallery.photos.values()),
            args.photos,
        )
        photo_cache_path: Path = args.photos / PHOTO_FACE_CACHE_NAME
        photo_cache: FaceCache = load_face_cache(photo_cache_path)
        gallery_vecs, audit = register_gallery(gallery, build_face_detector, photo_cache, MODEL_TAG)
        save_face_cache(photo_cache_path, MODEL_TAG, photo_cache)
        atomic_write_json(
            args.photos / REGISTRATION_AUDIT_NAME,
            audit_payload(audit, MODEL_TAG),
            what="注册审计",
        )
        logger.info(
            "注册净化完成: %d/%d 个号码可用（审计 → %s）",
            len(gallery_vecs),
            len(gallery.photos),
            args.photos / REGISTRATION_AUDIT_NAME,
        )

        goals: dict[str, GoalCrops] = merge_candidates(args.candidates)
        logger.info("合并 %d 个 candidates（%d 球）", len(args.candidates), len(goals))
        results: dict[str, GoalMatch] = match_goals(
            goals, gallery_vecs, build_face_detector, MODEL_TAG
        )

        payload: dict[str, Any] = build_matches_payload(results, THRESHOLD, MODEL_TAG)
        # 误指认红线：产物落盘前过既有 schema 校验（version/model/数值字段/数字主键）
        validate_matches_payload(payload, str(args.out))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, payload, what="photo_matches.json")
        n_no_face: int = sum(m.n_no_face for m in results.values())
        logger.info(
            "匹配完成: 出分 %d 球 / 高置信命中 %d 球（无脸裁图共 %d 张跳过）→ %s",
            len(results),
            len(payload["matches"]),
            n_no_face,
            args.out,
        )
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
