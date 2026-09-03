"""投篮者裁图 embedding 聚类（spec: docs/scorer-cluster/spec.md §数据契约；
Re-ID 后端见 docs/scorer-reid/spec.md §数据契约 cluster_scorers.py --model 部分）。

输入：一个或多个 scorer_candidates.json（--candidates 可重复传参，顶层
    {"session","candidates":[...]}，键集取并集、同 key 后者覆盖前者；裁图文件与
    candidates 文件同目录）；--evaluate 时另读 roster.json（validate_roster 校验）。
输出：--out 指定的 scorer_clusters.json（version/model/threshold/clusters/unclustered
    契约，model = 实际后端 model_tag，rep_crops = 每簇质量分最高的球的 crops 前 2 张）；
    embedding 缓存落 <out 同目录>/clip_cache.json（文件名沿用不改，key = model_tag +
    裁图 md5，双后端共存不串；threshold 不进缓存键，断点续跑/调档不重复推理）。
依赖：scikit-learn（AgglomerativeClustering）、numpy、PIL、scripts/roster.py、
    scripts/pipe_common.py；编码后端按 --model 二选一：
    - clip（默认）：open_clip_torch ViT-B-32 / laion2b_s34b_b79k（权重首跑经
      HTTPS_PROXY=http://127.0.0.1:7897 从 HF 下载）；
    - osnet_x1_0：torchreid OSNet（行人 Re-ID 专用），权重默认
      models/osnet_x1_0_market1501.pth（--reid-weights 可覆盖；torchreid/torch
      只在构造时 import，测试注入假 encoder 不碰真模型）。
典型调用：
    python scripts/cluster_scorers.py \
        --candidates work/20260722/scorers/scorer_candidates.json \
        --candidates work/20260722/scorers_b2/scorer_candidates.json \
        --out work/20260722/scorer_clusters.json
    # Re-ID 后端 + 纯度自检（只统计 roster assignments 里有的键，打印日志不写文件）：
    python scripts/cluster_scorers.py --candidates ... --out ... \
        --model osnet_x1_0 --evaluate --roster work/20260722/roster.json

聚类口径（写死，spec §数据契约）：球为聚类单位——一球多图（entry["crops"]，旧数据
无此字段回退单 entry["crop"]）各提 embedding 取均值、L2 归一化后再聚类；只对
status=OK 且有裁图的球聚类，SKIP/无裁图/裁图文件缺失的球进 unclustered；
AgglomerativeClustering(metric="cosine", linkage="average",
distance_threshold=--threshold, n_clusters=None) 不定簇数，阈值实跑标定。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from errors import BasketballPipelineError, ExternalApiError, SchemaError
from pipe_common import atomic_write_json, configure_logging, new_run_id, read_json
from roster import validate_roster

logger = logging.getLogger(__name__)

# ---- 编码后端（--model 取值 → 缓存/落盘 model_tag 对应表，spec §数据契约写死） ----
# CLIP 后端（spec §Tech Stack；权重 ~350MB 首跑经代理下载，之后本地缓存）
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
MODEL_TAG: str = f"{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}"  # CLIP 后端 model_tag（沿用不改）
HTTPS_PROXY_HINT: str = "HTTPS_PROXY=http://127.0.0.1:7897"  # HF 下载代理（AGENTS.md 环境节）
# OSNet Re-ID 后端（torchreid；Market1501 训练权重，rank-1 94.2%）
OSNET_MODEL_NAME: str = "osnet_x1_0"
MODEL_TAG_OSNET: str = f"{OSNET_MODEL_NAME}/market1501"
OSNET_WEIGHTS_PATH: Path = Path("models/osnet_x1_0_market1501.pth")  # 工作区根相对路径
OSNET_NUM_CLASSES: int = 751  # Market1501 训练类别数（加载权重用，分类头键不匹配被丢弃）
OSNET_INPUT_H: int = 256  # OSNet 输入高（人形裁图 2:1）
OSNET_INPUT_W: int = 128  # OSNet 输入宽
# ImageNet 归一化常数（torchreid/OSNet 训练口径）
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
# --model CLI 取值 → model_tag（写死，spec §数据契约）
MODEL_TAGS: dict[str, str] = {"clip": MODEL_TAG, OSNET_MODEL_NAME: MODEL_TAG_OSNET}

# ---- 聚类参数（spec §数据契约；阈值拍脑袋起点，--evaluate 实跑后标定） ----
DEFAULT_THRESHOLD: float = 0.25  # --threshold 默认值（cosine 距离，average linkage）
MAX_THRESHOLD: float = 2.0  # cosine 距离上界，超过无意义
CLUSTER_VERSION: str = "cluster-v1"  # 输出 schema 版本，契约变更即升
REP_CROPS_PER_CLUSTER: int = 2  # 每簇代表图张数（质量分最高的球的 crops 前 N 张）

# ---- 缓存（spec：key = model + 裁图 md5；threshold 不进缓存键） ----
CLIP_CACHE_NAME: str = "clip_cache.json"  # 落 <out 同目录>
MD5_CHUNK_BYTES: int = 1 << 20  # md5 分块读取大小（1MB）

STATUS_OK: str = "OK"

# 图像编码器：裁图路径 → embedding 向量（不要求归一化，调用方统一 L2 归一）。
# 定义为类型别名便于测试注入假 encoder（不碰真模型/网络）。
ImageEncoder = Callable[[Path], np.ndarray]
EncoderFactory = Callable[[], ImageEncoder]


@dataclass(frozen=True, slots=True)
class GoalCrops:
    """一个进球的裁图信息（聚类输入单位；crops 为空表示无裁图，进 unclustered）。"""

    key: str
    status: str
    crops: tuple[str, ...]  # 裁图文件名（相对 base_dir），质量降序
    crop_scores: tuple[float, ...]  # 与 crops 对齐的质量分；旧数据无此字段为空
    base_dir: Path  # 裁图所在目录（= candidates 文件同目录）


@dataclass(frozen=True, slots=True)
class ClusterPurity:
    """单簇纯度（只统计 roster assignments 里有的键）。"""

    cluster_id: int
    size: int  # 簇内在 assignments 里的键数
    majority_tag: str
    majority_count: int


@dataclass(frozen=True, slots=True)
class PurityReport:
    """--evaluate 纯度报告（打印日志，不写文件）。

    purity = 各簇多数 tag 键数之和 / 入簇 assigned 键数（键数加权，spec §数据契约）。
    """

    assigned_total: int  # roster assignments 总键数
    assigned_in_clusters: int  # 入簇且在 assignments 里的键数
    unclustered_assigned: int  # unclustered 且在 assignments 里的键数
    correct: int  # 各簇多数 tag 键数之和
    purity: float  # correct / assigned_in_clusters；无入簇 assigned 键为 0.0
    clusters: tuple[ClusterPurity, ...]


def entry_crop_names(entry: dict[str, Any]) -> tuple[str, ...]:
    """取一球的裁图文件名序列：优先 crops（多裁），旧数据无此字段回退单 crop。

    Args:
        entry: scorer_candidates.json 的单条候选记录。

    Returns:
        裁图文件名元组（质量降序）；无裁图返回空元组。
    """
    crops_raw: Any = entry.get("crops")
    if isinstance(crops_raw, list):
        names: tuple[str, ...] = tuple(c for c in crops_raw if isinstance(c, str) and c)
        if names:
            return names
    crop: Any = entry.get("crop")
    if isinstance(crop, str) and crop:
        return (crop,)
    return ()


def entry_crop_scores(entry: dict[str, Any]) -> tuple[float, ...]:
    """取一球的质量分序列（旧数据无 crop_scores 字段返回空元组，按 0 分处理）。

    Args:
        entry: scorer_candidates.json 的单条候选记录。

    Returns:
        质量分元组（与 crops 对齐）。
    """
    raw: Any = entry.get("crop_scores")
    if not isinstance(raw, list):
        return ()
    return tuple(float(s) for s in raw if isinstance(s, (int, float)) and not isinstance(s, bool))


def load_candidates(path: Path) -> dict[str, GoalCrops]:
    """读取并校验单个 scorer_candidates.json，返回 key → GoalCrops（保 JSON 原序）。

    Args:
        path: scorer_candidates.json 路径；裁图相对路径按该文件同目录解析。

    Returns:
        key → GoalCrops（base_dir = path.parent）。

    Raises:
        SchemaError: 顶层非对象 / 缺 candidates 列表 / 条目非对象 /
            key 缺失或非空 str / status 非 str。
    """
    data: Any = read_json(path, what="scorer_candidates.json")
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 顶层必须是对象，实际 {type(data).__name__}")
    entries: Any = data.get("candidates")
    if not isinstance(entries, list):
        raise SchemaError(f"{path}: 缺 candidates 列表或类型错误")
    result: dict[str, GoalCrops] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchemaError(f"{path}: 第{i}条不是对象")
        key: Any = entry.get("key")
        if not isinstance(key, str) or not key:
            raise SchemaError(f"{path}: 第{i}条 key 缺失或不是非空 str")
        status: Any = entry.get("status")
        if not isinstance(status, str):
            raise SchemaError(f"{path}: 第{i}条({key}) status 不是 str")
        result[key] = GoalCrops(
            key=key,
            status=status,
            crops=entry_crop_names(entry),
            crop_scores=entry_crop_scores(entry),
            base_dir=path.parent,
        )
    return result


def merge_candidates(paths: list[Path]) -> dict[str, GoalCrops]:
    """合并多个 candidates 文件：键集取并集，同 key 后者覆盖前者（spec §数据契约）。

    Args:
        paths: scorer_candidates.json 路径列表（按 CLI 传参顺序，后者优先）。

    Returns:
        合并后的 key → GoalCrops（键序 = 首次出现顺序，覆盖不改变位置）。

    Raises:
        SchemaError: 任一文件结构损坏。
    """
    merged: dict[str, GoalCrops] = {}
    for path in paths:
        merged.update(load_candidates(path))
    return merged


def file_md5(path: Path) -> str:
    """计算文件 md5（分块读取；仅作缓存键，非安全用途）。

    Args:
        path: 文件路径。

    Returns:
        32 位十六进制 md5。
    """
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(MD5_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip_cache(path: Path) -> dict[str, list[float]]:
    """读 clip_cache.json（幂等缓存）；缺失 → 空。

    缓存键 = ``<model_tag>:<裁图 md5>``（spec §数据契约）：不同后端键前缀天然
    隔离、互不命中，**全量条目都保留**——不过滤前缀（曾按当前模型过滤再整体
    回写，会把另一后端的 embedding 静默冲掉，2026-08-09 实跑踩过）。文件结构
    损坏记 WARNING 重开（缓存可重建，不是真值数据，不按 SchemaError 停批）。

    Args:
        path: <out 同目录>/clip_cache.json 路径。

    Returns:
        缓存键 → embedding 向量（list[float]，仅保留合法条目）。
    """
    if not path.exists():
        return {}
    payload: Any = read_json(path, what="clip_cache.json")
    if not isinstance(payload, dict):
        logger.warning("clip_cache.json 结构异常，重新开始: %s", path)
        return {}
    vectors: Any = payload.get("vectors")
    if not isinstance(vectors, dict):
        return {}
    cache: dict[str, list[float]] = {}
    for k, v in vectors.items():
        if not isinstance(k, str) or ":" not in k:
            continue
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            cache[k] = [float(x) for x in v]
    n_bad: int = len(vectors) - len(cache)
    if n_bad:
        logger.warning("clip_cache 丢弃 %d 条坏条目", n_bad)
    return cache


def save_clip_cache(path: Path, model_tag: str, cache: dict[str, list[float]]) -> None:
    """原子写 clip_cache.json（_meta 记录模型与时间戳）。

    Args:
        path: <out 同目录>/clip_cache.json 路径。
        model_tag: 当前模型标识。
        cache: 缓存键 → embedding 向量。

    Raises:
        OSError: IO 重试耗尽（由 atomic_write_json 抛出）。
    """
    payload: dict[str, Any] = {
        "_meta": {"model": model_tag, "updated_at": datetime.now(UTC).isoformat()},
        "vectors": cache,
    }
    atomic_write_json(path, payload, what="clip_cache.json")


def build_clip_encoder() -> ImageEncoder:
    """加载 CLIP 模型并返回图像编码器（open_clip ViT-B-32 / laion2b_s34b_b79k）。

    模型加载隔离在本函数（torch 导入慢、权重首跑需网络），测试注入假 encoder
    不经过这里。返回的编码器输出未归一化向量，归一化由调用方统一做。

    Returns:
        ImageEncoder：裁图路径 → embedding 向量（float64 numpy 数组）。

    Raises:
        ExternalApiError: open_clip/torch 未安装，或权重加载失败（含 HTTPS_PROXY
            提示，不静默）。
    """
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise ExternalApiError(f"open_clip_torch/torch 未安装: {exc}") from exc
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model.eval()
    except Exception as exc:
        raise ExternalApiError(
            f"CLIP 权重加载失败（{MODEL_TAG}）: {type(exc).__name__}: {exc}；"
            f"首次需经代理从 HF 下载（~350MB），请确认 {HTTPS_PROXY_HINT} 已设置后重试，"
            "或手动放置权重到 HF 缓存目录"
        ) from exc

    def encode(path: Path) -> np.ndarray:
        """单张裁图编码：预处理 → encode_image → float64 numpy 向量（未归一化）。"""
        with Image.open(path) as im:
            tensor = preprocess(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(tensor)
        return np.asarray(feat[0].cpu().numpy(), dtype=np.float64)

    return encode


def build_osnet_encoder(weights_path: Path = OSNET_WEIGHTS_PATH) -> ImageEncoder:
    """加载 OSNet Re-ID 模型（torchreid osnet_x1_0 + Market1501 权重）并返回图像编码器。

    torchreid/torch 只在本函数内 import（测试注入假 encoder 不经过这里）。
    权重直接加载本地 Market1501 文件（pretrained=False，不走 imagenet 预训练；
    分类头键不匹配被丢弃，属正常）。预处理 = Resize(256×128) + ToTensor +
    ImageNet 归一化，用 PIL/numpy 手写（不引 torchvision transform，口径等价）。

    Args:
        weights_path: Market1501 权重 .pth 路径（默认 OSNET_WEIGHTS_PATH，
            工作区根相对路径，按当前工作目录解析）。

    Returns:
        ImageEncoder：裁图路径 → 512 维 embedding（float64 numpy 数组，未归一化）。

    Raises:
        ExternalApiError: 权重文件不存在（含路径提示），或 torchreid/torch 未安装
            （含 pip 安装提示），或权重加载失败——显式报错，不静默回退 CLIP。
    """
    if not weights_path.is_file():
        raise ExternalApiError(
            f"OSNet 权重文件不存在: {weights_path}（默认 {OSNET_WEIGHTS_PATH}，"
            "工作区根相对路径；请确认在仓库根目录运行，或用 --reid-weights 指定实际路径）"
        )
    try:
        import torch
        import torchreid
    except ImportError as exc:
        raise ExternalApiError(
            f"torchreid/torch 未安装: {exc}；请执行 pip install deep-person-reid（清华镜像）"
        ) from exc
    try:
        model = torchreid.models.build_model(
            OSNET_MODEL_NAME, num_classes=OSNET_NUM_CLASSES, loss="softmax", pretrained=False
        )
        torchreid.utils.load_pretrained_weights(model, str(weights_path))
        model.eval()
    except Exception as exc:
        raise ExternalApiError(
            f"OSNet 权重加载失败（{MODEL_TAG_OSNET}，{weights_path}）: "
            f"{type(exc).__name__}: {exc}；请确认权重文件完整（Market1501 版 ~10MB）"
        ) from exc

    mean = np.asarray(_IMAGENET_MEAN, dtype=np.float64)
    std = np.asarray(_IMAGENET_STD, dtype=np.float64)

    def encode(path: Path) -> np.ndarray:
        """单张裁图编码：Resize(256×128) + ImageNet 归一化 → 前向 → 512 维向量。"""
        with Image.open(path) as im:
            resized = im.convert("RGB").resize((OSNET_INPUT_W, OSNET_INPUT_H), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float64) / 255.0
        arr = (arr - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor.float())
        return np.asarray(feat[0].cpu().numpy(), dtype=np.float64)

    return encode


def build_encoder(model: str, reid_weights: Path = OSNET_WEIGHTS_PATH) -> ImageEncoder:
    """按 --model 取值分派编码后端（spec §数据契约；查无此模型显式失败，不回退）。

    Args:
        model: --model CLI 取值（MODEL_TAGS 键："clip" / "osnet_x1_0"）。
        reid_weights: OSNet 权重路径（仅 osnet_x1_0 后端使用）。

    Returns:
        对应后端的 ImageEncoder。

    Raises:
        BasketballPipelineError: model 不在 MODEL_TAGS 里（含合法值提示）。
        ExternalApiError: 后端依赖/权重不可用（见各后端构造函数）。
    """
    if model == "clip":
        return build_clip_encoder()
    if model == OSNET_MODEL_NAME:
        return build_osnet_encoder(reid_weights)
    raise BasketballPipelineError(f"查无此编码器后端: {model!r}（合法值: {sorted(MODEL_TAGS)}）")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 归一化向量。

    Args:
        vec: 一维向量。

    Returns:
        单位向量（float64）。

    Raises:
        BasketballPipelineError: 零向量无法归一化（逻辑错误显式失败）。
    """
    norm: float = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise BasketballPipelineError("embedding 零向量无法 L2 归一化")
    return vec / norm


def embed_goal(
    goal: GoalCrops,
    encoder: ImageEncoder | None,
    cache: dict[str, list[float]],
    model_tag: str,
) -> np.ndarray | None:
    """单球 embedding：逐张裁图编码（缓存优先）→ 均值 → L2 归一化。

    每张图先各自 L2 归一化再取均值（避免某张图模长主导），均值后再归一化。
    缓存命中的向量已是归一化后的（写入前就归一），直接使用。

    Args:
        goal: 聚类输入单位（status=OK 且 crops 非空，调用方保证）。
        encoder: 图像编码器（测试注入假 encoder）；可为 None（纯缓存路径），
            仅在实际缓存未命中时才要求非空。
        cache: 缓存（原地写入新向量，键 = model_tag:md5）。
        model_tag: 模型标识。

    Returns:
        球级 embedding 单位向量；任一裁图文件缺失记 ERROR 返回 None
        （调用方把该球进 unclustered，不炸整批）。

    Raises:
        BasketballPipelineError: 缓存未命中但 encoder 为 None（上游惰性构建逻辑错误）。
    """
    vecs: list[np.ndarray] = []
    for name in goal.crops:
        path: Path = goal.base_dir / name
        if not path.is_file():
            logger.error("裁图文件缺失，该球进 unclustered: %s (%s)", path, goal.key)
            return None
        cache_key: str = f"{model_tag}:{file_md5(path)}"
        cached: list[float] | None = cache.get(cache_key)
        if cached is not None:
            vecs.append(np.asarray(cached, dtype=np.float64))
            continue
        if encoder is None:
            raise BasketballPipelineError("缓存未命中但 encoder 未构建（惰性构建逻辑错误）")
        vec: np.ndarray = l2_normalize(np.asarray(encoder(path), dtype=np.float64))
        cache[cache_key] = [float(x) for x in vec]
        vecs.append(vec)
    return l2_normalize(np.mean(vecs, axis=0))


def embed_goals(
    goals: dict[str, GoalCrops],
    encoder_factory: EncoderFactory,
    cache: dict[str, list[float]],
    model_tag: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """批量 embedding：只对 status=OK 且有裁图的球；encoder 惰性构建（全缓存命中零加载）。

    SKIP / 无裁图的球不参与 embedding（调用方归 unclustered）；裁图文件缺失的球
    记入失败列表（同样归 unclustered，进程退出码非零由 main 控制）。

    Args:
        goals: merge_candidates 产物（全部候选，含 SKIP）。
        encoder_factory: 零参工厂，首次缓存未命中时才调用（避免全命中仍加载模型）。
        cache: 缓存（原地写入）。
        model_tag: 模型标识。

    Returns:
        (key → embedding, 裁图缺失失败 key 列表)；两者键互斥。
    """
    embeddings: dict[str, np.ndarray] = {}
    failed: list[str] = []
    encoder: ImageEncoder | None = None
    for key, goal in goals.items():
        if goal.status != STATUS_OK or not goal.crops:
            continue
        # 先试纯缓存路径：encoder 未构建且有任一裁图未命中缓存时才构建（惰性，全命中零加载）
        if encoder is None and any(
            f"{model_tag}:{file_md5(goal.base_dir / name)}" not in cache
            for name in goal.crops
            if (goal.base_dir / name).is_file()
        ):
            encoder = encoder_factory()
        vec: np.ndarray | None = embed_goal(goal, encoder, cache, model_tag)
        if vec is None:
            failed.append(key)
            continue
        embeddings[key] = vec
    return embeddings, failed


LINKAGES: tuple[str, ...] = ("average", "complete", "single")  # --linkage 合法值


def cluster_keys(
    keys: list[str], matrix: np.ndarray, threshold: float, linkage: str = "average"
) -> list[list[str]]:
    """凝聚聚类：cosine 距离 + distance_threshold（不定簇数）。

    Args:
        keys: 与 matrix 行对齐的进球键。
        matrix: (n, dim) embedding 矩阵（行已 L2 归一化；cosine 度量不依赖归一，
            归一仅为人读与缓存口径一致）。
        threshold: cosine 距离阈值（distance_threshold）。
        linkage: 连接方式（average/complete/single；complete 簇更紧致，标定用）。

    Returns:
        簇列表（簇内键按输入序；簇序 = 各簇首键的输入序）；keys 为空返回空列表。

    Raises:
        ValueError: keys 与 matrix 行数不一致 / threshold 不在 (0, 2] / linkage 非法。
    """
    if len(keys) != matrix.shape[0]:
        raise ValueError(f"keys 数({len(keys)})与 embedding 行数({matrix.shape[0]})不一致")
    if not 0.0 < threshold <= MAX_THRESHOLD:
        raise ValueError(f"threshold 须在 (0, {MAX_THRESHOLD}]，实际 {threshold}")
    if linkage not in LINKAGES:
        raise ValueError(f"linkage 须为 {LINKAGES} 之一，实际 {linkage!r}")
    if not keys:
        return []
    if len(keys) == 1:
        return [list(keys)]  # 单样本 sklearn 不接盘，直接自成一簇
    labels: np.ndarray = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=threshold,
    ).fit_predict(matrix)
    order: list[int] = sorted(set(int(x) for x in labels), key=lambda lb: list(labels).index(lb))
    return [[keys[i] for i, lb in enumerate(labels) if int(lb) == label] for label in order]


def build_result(
    goals: dict[str, GoalCrops],
    embeddings: dict[str, np.ndarray],
    clusters_keys: list[list[str]],
    threshold: float,
    model_tag: str,
) -> dict[str, Any]:
    """组装 scorer_clusters.json 载荷（spec §数据契约写死的字段）。

    rep_crops = 每簇质量分最高的球（crop_scores[0]，即该球最佳帧分；旧数据无分
    按 0.0；并列取键序靠前者）的 crops 前 REP_CROPS_PER_CLUSTER 张。
    unclustered = 全部未入簇的 key（SKIP / 无裁图 / 裁图缺失），按合并后键序。

    Args:
        goals: merge_candidates 产物。
        embeddings: embed_goals 产物（key → 单位向量）。
        clusters_keys: cluster_keys 产物。
        threshold: 本次聚类阈值（落盘供标定追溯）。
        model_tag: 实际编码后端标识（MODEL_TAGS 值；落盘 model 字段记实际后端）。

    Returns:
        可 JSON 序列化的载荷 dict。
    """
    clusters: list[dict[str, Any]] = []
    for cid, keys in enumerate(clusters_keys, start=1):
        best: GoalCrops | None = None
        best_score: float = -1.0
        for k in keys:
            g: GoalCrops = goals[k]
            score: float = g.crop_scores[0] if g.crop_scores else 0.0
            if score > best_score:
                best = g
                best_score = score
        if best is None:  # 防御：空簇不应出现，逻辑错误显式失败
            raise BasketballPipelineError(f"簇 {cid} 为空（聚类契约破坏）")
        clusters.append(
            {
                "cluster_id": cid,
                "keys": list(keys),
                "rep_crops": list(best.crops[:REP_CROPS_PER_CLUSTER]),
            }
        )
    unclustered: list[str] = [k for k in goals if k not in embeddings]
    return {
        "version": CLUSTER_VERSION,
        "model": model_tag,
        "threshold": threshold,
        "clusters": clusters,
        "unclustered": unclustered,
    }


def evaluate_purity(
    clusters: list[list[str]],
    unclustered: list[str],
    assignments: dict[str, str],
) -> PurityReport:
    """纯度自检：只统计 roster assignments 里有的键，其余键剔除不计入（spec §数据契约）。

    每簇多数 tag = 簇内 assigned 键的 assignments 值众数（并列取该簇内先出现者，
    Counter.most_common 保插入序）；整体纯度 = 各簇多数 tag 键数之和 /
    入簇 assigned 键数（键数加权）。

    Args:
        clusters: 簇列表（每簇键列表）。
        unclustered: 未入簇键列表。
        assignments: roster.assignments（key → tag）。

    Returns:
        PurityReport（调用方打印日志，不写文件）。
    """
    per_cluster: list[ClusterPurity] = []
    correct: int = 0
    in_clusters: int = 0
    for cid, keys in enumerate(clusters, start=1):
        tags: list[str] = [assignments[k] for k in keys if k in assignments]
        if not tags:
            continue
        majority_tag, majority_count = Counter(tags).most_common(1)[0]
        per_cluster.append(
            ClusterPurity(
                cluster_id=cid,
                size=len(tags),
                majority_tag=majority_tag,
                majority_count=majority_count,
            )
        )
        correct += majority_count
        in_clusters += len(tags)
    unclustered_assigned: int = sum(1 for k in unclustered if k in assignments)
    return PurityReport(
        assigned_total=len(assignments),
        assigned_in_clusters=in_clusters,
        unclustered_assigned=unclustered_assigned,
        correct=correct,
        purity=(correct / in_clusters) if in_clusters else 0.0,
        clusters=tuple(per_cluster),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="投篮者裁图 embedding 聚类（CLIP / OSNet Re-ID 双后端，--model 切换）"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        action="append",
        type=Path,
        help="scorer_candidates.json 路径（可重复传多个批次；键集取并集，同 key 后者覆盖前者）",
    )
    parser.add_argument("--out", required=True, type=Path, help="scorer_clusters.json 输出路径")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_TAGS),
        default="clip",
        help="编码后端（默认 %(default)s；osnet_x1_0 = torchreid 行人 Re-ID 专用模型）",
    )
    parser.add_argument(
        "--reid-weights",
        type=Path,
        default=OSNET_WEIGHTS_PATH,
        help="OSNet Market1501 权重路径（默认 %(default)s，仅 --model osnet_x1_0 用）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="聚类 cosine 距离阈值（默认 %(default)s；只影响聚类步，调档不重复推理）",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="纯度自检（只统计 roster assignments 里有的键，打印日志不写文件；需配 --roster）",
    )
    parser.add_argument(
        "--roster", type=Path, default=None, help="roster.json 路径（--evaluate 用）"
    )
    parser.add_argument(
        "--linkage",
        choices=["average", "complete", "single"],
        default="average",
        help="凝聚聚类连接方式（默认 %(default)s；complete 簇更紧致，标定对比用）",
    )
    ns = parser.parse_args(argv)
    if not 0.0 < ns.threshold <= MAX_THRESHOLD:
        parser.error(f"--threshold 须在 (0, {MAX_THRESHOLD}]，实际 {ns.threshold}")
    if ns.evaluate and ns.roster is None:
        parser.error("--evaluate 需配 --roster")
    return ns


def _log_purity(report: PurityReport) -> None:
    """把纯度报告打印到日志（INFO；不写文件）。"""
    logger.info(
        "纯度自检: roster assignments=%d 键，入簇 %d 键，未入簇 %d 键",
        report.assigned_total,
        report.assigned_in_clusters,
        report.unclustered_assigned,
    )
    for cp in report.clusters:
        logger.info(
            "簇 %d: 大小 %d，多数 tag=%s（%d/%d）",
            cp.cluster_id,
            cp.size,
            cp.majority_tag,
            cp.majority_count,
            cp.size,
        )
    logger.info(
        "整体纯度: %d/%d = %.1f%%（簇数 %d）",
        report.correct,
        report.assigned_in_clusters,
        report.purity * 100,
        len(report.clusters),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回进程退出码（0=成功；1=管线失败或有裁图缺失）。"""
    args = _parse_args(argv)
    run_id: str = new_run_id()
    configure_logging(run_id)
    try:
        goals: dict[str, GoalCrops] = merge_candidates(args.candidates)
        n_ok: int = sum(1 for g in goals.values() if g.status == STATUS_OK and g.crops)
        logger.info(
            "合并 %d 个 candidates 文件: 共 %d 球（OK 且有裁图 %d，进 unclustered %d）",
            len(args.candidates),
            len(goals),
            n_ok,
            len(goals) - n_ok,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cache_path: Path = args.out.parent / CLIP_CACHE_NAME
        model_tag: str = MODEL_TAGS[args.model]
        cache: dict[str, list[float]] = load_clip_cache(cache_path)
        logger.info(
            "编码后端 %s（%s）；clip_cache 总 %d 条（本模型前缀 %d 条）← %s",
            args.model,
            model_tag,
            len(cache),
            sum(1 for k in cache if k.startswith(f"{model_tag}:")),
            cache_path,
        )
        embeddings, failed = embed_goals(
            goals,
            lambda: build_encoder(args.model, args.reid_weights),
            cache,
            model_tag,
        )
        save_clip_cache(cache_path, model_tag, cache)
        logger.info("embedding 完成: %d 球（缓存共 %d 条）", len(embeddings), len(cache))

        keys: list[str] = list(embeddings)
        matrix: np.ndarray = np.stack([embeddings[k] for k in keys]) if keys else np.empty((0, 0))
        clusters_keys: list[list[str]] = cluster_keys(keys, matrix, args.threshold, args.linkage)
        result: dict[str, Any] = build_result(
            goals, embeddings, clusters_keys, args.threshold, model_tag
        )
        atomic_write_json(args.out, result, what="scorer_clusters.json")
        logger.info(
            "聚类完成: %d 簇 / 入簇 %d 球 / unclustered %d 球 → %s",
            len(clusters_keys),
            len(keys),
            len(result["unclustered"]),
            args.out,
        )

        if args.evaluate:
            roster_data: Any = read_json(args.roster, what="roster.json")
            roster = validate_roster(roster_data, str(args.roster))
            _log_purity(evaluate_purity(clusters_keys, result["unclustered"], roster.assignments))

        if failed:
            logger.error(
                "有 %d 球裁图缺失进 unclustered（详见上条 ERROR），退出码非零", len(failed)
            )
            return 1
        return 0
    except BasketballPipelineError as e:
        logger.error("管线失败 run_id=%s: %s", run_id, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
