"""crop 생성 (노트북 02, D-15).

샘플 단위는 crop 이다 (D-01). 데이터셋마다 경로가 다르다.

    AIHub     polygon -> bbox -> 정사각 crop      (annotations.csv 사용)
    Roboflow  YOLO bbox -> 정사각 crop            (ablation 전용)
    DACON     이미 하자 근접 사진 -> identity
    Kaggle    227x227 표면 패치 -> identity

geometry 는 configs/crop.yaml 의 crop-context-2.0 이다. 임의로 바꾸면 기존 QC 근거가
사라진다. max_crop_fraction 은 1.00 이며 그 이유는 D-15 참조.

금지 사항:
    - sealed_future_eval 행은 이미지를 열지도 않는다 (D-14)
    - legacy_annotation_error 15건은 review-only crop 만 만든다 (D-15)
    - NMS/dedupe 는 같은 L1 끼리만. 다른 L1 이 겹치면 review (D-09)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import yaml

from .paths import CONFIG_DIR, ENCODING, load_paths, read_csv

CROP_CONFIG = CONFIG_DIR / "crop.yaml"


def load_crop_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CROP_CONFIG).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# box
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Box:
    """정규화 좌표 (cx, cy, w, h) + L1 라벨."""

    cx: float
    cy: float
    w: float
    h: float
    label: str
    severity: str = ""

    def xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        return (
            (self.cx - self.w / 2) * img_w, (self.cy - self.h / 2) * img_h,
            (self.cx + self.w / 2) * img_w, (self.cy + self.h / 2) * img_h,
        )


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# --------------------------------------------------------------------------- #
# 라벨 로딩
# --------------------------------------------------------------------------- #

def load_aihub_boxes() -> dict[str, list[Box]]:
    """annotations.csv 에서 record_id 별 box. polygon 은 이미 bbox 로 변환돼 있다."""
    path = load_paths().inherit("aihub_polygon_annotations")
    df = pd.read_csv(path, encoding=ENCODING)
    out: dict[str, list[Box]] = {}
    for rec, g in df.groupby("record_id"):
        out[rec] = [
            Box(r.cx, r.cy, r.w, r.h, r.target_class, str(r.severity))
            for r in g.itertuples()
        ]
    return out


def load_roboflow_boxes(label_path: Path, class_names: list[str],
                        class_map: dict[str, str | None]) -> list[Box]:
    """YOLO txt -> Box. class_id 는 data.yaml 의 names 인덱스다."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        idx = int(float(parts[0]))
        if idx >= len(class_names):
            continue
        l1 = class_map.get(class_names[idx])
        if not l1:                      # None = L1 매핑 없음 (baseboard, bottle 등)
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(Box(cx, cy, w, h, l1))
    return boxes


# --------------------------------------------------------------------------- #
# crop window
# --------------------------------------------------------------------------- #

def crop_window(box: Box, img_w: int, img_h: int, g: dict[str, Any]) -> tuple[int, int, int]:
    """(x0, y0, side) 를 반환한다. **완전한 정사각을 보장한다.**

    기존 구현은 x0/y0/x1/y1 을 각각 반올림해 685장이 1px 비정사각이 됐다
    (그중 471장은 이미지 내부라 clipping 탓이 아니다). 정수 side 를 먼저 확정하고
    x1 = x0 + side 로 계산해 그 오차를 없앤다 (D-15).
    """
    bx0, by0, bx1, by1 = box.xyxy(img_w, img_h)
    bw, bh = bx1 - bx0, by1 - by0
    short = float(min(img_w, img_h))

    side = max(bw, bh) * (1.0 + 2.0 * g["context_ratio"])
    side = max(side, float(g["min_crop_px"]))
    side = min(side, short * g["max_crop_fraction"])
    # ★ floor — cap 이 seed box 자체보다 작아지면 안 된다.
    #   이 줄이 없으면 box 장변이 cap 을 넘는 411건이 재현되지 않는다 (D-15).
    side = max(side, min(max(bw, bh), short))
    side = min(side, short)

    side = int(round(side))
    side = max(1, min(side, int(short)))

    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, img_w - side))              # clamp -> 항상 in-bounds
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, side


def dedupe_windows(windows: list[tuple[int, int, int, Box]],
                   dedupe_iou: float) -> list[tuple[int, int, int, Box]]:
    """같은 L1 끼리만 NMS. 다른 L1 은 서로 억제하지 않는다 (D-15 규칙 5)."""
    kept: list[tuple[int, int, int, Box]] = []
    for x0, y0, side, box in windows:
        rect = (x0, y0, x0 + side, y0 + side)
        dup = any(
            b.label == box.label and iou(rect, (kx, ky, kx + ks, ky + ks)) > dedupe_iou
            for kx, ky, ks, b in kept
        )
        if not dup:
            kept.append((x0, y0, side, box))
    return kept


def boxes_inside(window: tuple[int, int, int], boxes: list[Box],
                 img_w: int, img_h: int, min_retained_area: float) -> tuple[list[Box], bool]:
    """crop 안에 충분히 남는 box 목록과, 다른 L1 이 섞였는지 여부."""
    x0, y0, side = window
    rect = (x0, y0, x0 + side, y0 + side)
    kept = []
    for b in boxes:
        bx0, by0, bx1, by1 = b.xyxy(img_w, img_h)
        ix0, iy0 = max(rect[0], bx0), max(rect[1], by0)
        ix1, iy1 = min(rect[2], bx1), min(rect[3], by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        area = (bx1 - bx0) * (by1 - by0)
        if area <= 0 or (ix1 - ix0) * (iy1 - iy0) / area < min_retained_area:
            continue
        kept.append(b)
    labels = {b.label for b in kept}
    return kept, len(labels) > 1


# --------------------------------------------------------------------------- #
# 생성
# --------------------------------------------------------------------------- #

CROP_FIELDS = [
    "sample_id", "crop_path", "mode", "parent_record_id", "parent_image_path",
    "dataset", "unified_label", "severity",
    "duplicate_group_id", "split_component_id", "source_root_id",
    "crop_x0", "crop_y0", "crop_side", "crop_width", "crop_height",
    "parent_width", "parent_height", "bbox_area_ratio",
    "num_boxes_in_crop", "mixed_l1_in_crop",
    "lane", "baseline_eligible", "quality_status", "rejection_reason",
]


@dataclass
class CropStats:
    parents_seen: int = 0
    crops_written: int = 0
    identity: int = 0
    skipped_no_box: int = 0
    skipped_unreadable: int = 0
    mixed_l1: int = 0
    by_lane: dict[str, int] = field(default_factory=dict)
    by_dataset: dict[str, int] = field(default_factory=dict)


def lane_of(row) -> str:
    """crop 이 속한 계보. 하나만 True 여야 한다."""
    if row.legacy_annotation_error:
        return "review_only"
    if row.head_a6_eligible:
        return "baseline"
    if row.head_a6_aux_pool:
        return "ablation"
    if row.head_a7_aux_train:
        return "lineage_a7"
    return "excluded"


def select_parents(df: pd.DataFrame) -> pd.DataFrame:
    """crop 생성 대상. sealed 는 여기서 이미 빠진다 (이미지를 열지 않기 위해)."""
    sel = (
        df["head_a6_eligible"] | df["head_a6_aux_pool"] | df["head_a7_aux_train"]
        | df["legacy_annotation_error"]
    ) & ~df["sealed_future_eval"]
    return df[sel].copy()


def _stem(p: str) -> str:
    return Path(p).stem.replace(" ", "_")


def _load_registry() -> dict[str, dict[str, str | None]]:
    import json

    reg = json.loads(load_paths().registry.read_text(encoding="utf-8"))
    return {k: v.get("class_map", {}) for k, v in reg["datasets"].items()}


def _identity_row(row, lane: str) -> dict[str, Any]:
    """bbox 가 없는 데이터셋. 원본이 곧 crop 이므로 복사하지 않고 참조만 한다."""
    return {
        "sample_id": f"{row.record_id}#0",
        "crop_path": row.image_path,
        "mode": "identity",
        "parent_record_id": row.record_id,
        "parent_image_path": row.image_path,
        "dataset": row.dataset,
        "unified_label": row.unified_label,
        "severity": "",
        "duplicate_group_id": row.duplicate_group_id,
        "split_component_id": row.split_component_id,
        "source_root_id": row.source_root_id,
        "crop_x0": 0, "crop_y0": 0, "crop_side": -1,
        "crop_width": -1, "crop_height": -1,
        "parent_width": -1, "parent_height": -1,
        "bbox_area_ratio": 1.0,
        "num_boxes_in_crop": 1,
        "mixed_l1_in_crop": False,
        "lane": lane,
        "baseline_eligible": lane == "baseline",
        "quality_status": "review" if lane == "review_only" else "keep",
        "rejection_reason": "legacy_annotation_error" if lane == "review_only" else "",
    }


def build(df: pd.DataFrame, limit: int | None = None,
          progress: bool = True) -> tuple[pd.DataFrame, CropStats]:
    """crop 을 생성하고 manifest 를 반환한다."""
    from tqdm.auto import tqdm

    cfg = load_crop_config()
    g = cfg["geometry"]
    paths = load_paths()
    raw_root = paths.legacy_root
    out_root = paths.artifact("crops")
    registry = _load_registry()
    aihub_boxes = load_aihub_boxes()

    targets = select_parents(df)
    if limit:
        targets = targets.head(limit)

    stats = CropStats()
    rows: list[dict[str, Any]] = []

    for row in tqdm(targets.itertuples(), total=len(targets),
                    disable=not progress, desc="crops"):
        stats.parents_seen += 1
        lane = lane_of(row)
        stats.by_lane[lane] = stats.by_lane.get(lane, 0) + 1

        # identity — bbox 가 없는 데이터셋은 원본이 곧 crop
        if row.dataset in ("dacon_wallpaper", "kaggle_cracks"):
            rows.append(_identity_row(row, lane))
            stats.identity += 1
            stats.by_dataset[row.dataset] = stats.by_dataset.get(row.dataset, 0) + 1
            continue

        boxes = (aihub_boxes.get(row.record_id, []) if row.dataset == "aihub_567"
                 else load_roboflow_boxes(raw_root / row.label_path,
                                          cfg["roboflow_class_names"][row.dataset],
                                          registry[row.dataset]))
        if not boxes:
            stats.skipped_no_box += 1
            continue

        img_path = raw_root / row.image_path
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            stats.skipped_unreadable += 1
            continue
        H, W = img.shape[:2]

        windows = dedupe_windows([(*crop_window(b, W, H, g), b) for b in boxes],
                                 g["dedupe_iou"])

        for i, (x0, y0, side, box) in enumerate(windows):
            inside, mixed = boxes_inside((x0, y0, side), boxes, W, H, g["min_retained_area"])
            if mixed:
                stats.mixed_l1 += 1

            rel = Path(row.dataset) / lane / f"{_stem(row.image_path)}_{i}.jpg"
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", img[y0:y0 + side, x0:x0 + side],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                stats.skipped_unreadable += 1
                continue
            buf.tofile(str(dest))

            rows.append({
                "sample_id": f"{row.record_id}#{i}",
                "crop_path": rel.as_posix(),
                "mode": "polygon_crop" if row.dataset == "aihub_567" else "bbox_crop",
                "parent_record_id": row.record_id,
                "parent_image_path": row.image_path,
                "dataset": row.dataset,
                "unified_label": box.label,
                "severity": box.severity,
                "duplicate_group_id": row.duplicate_group_id,
                "split_component_id": row.split_component_id,
                "source_root_id": row.source_root_id,
                "crop_x0": x0, "crop_y0": y0, "crop_side": side,
                "crop_width": side, "crop_height": side,
                "parent_width": W, "parent_height": H,
                "bbox_area_ratio": round(box.w * box.h, 6),
                "num_boxes_in_crop": len(inside),
                "mixed_l1_in_crop": mixed,
                "lane": lane,
                "baseline_eligible": lane == "baseline" and not mixed,
                "quality_status": "review" if (mixed or lane == "review_only") else "keep",
                "rejection_reason": ("mixed_l1_in_crop" if mixed
                                     else "legacy_annotation_error" if lane == "review_only"
                                     else ""),
            })
            stats.crops_written += 1
            stats.by_dataset[row.dataset] = stats.by_dataset.get(row.dataset, 0) + 1

    return pd.DataFrame(rows, columns=CROP_FIELDS), stats


def verify(crops_df: pd.DataFrame, parents: pd.DataFrame,
           full: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    """D-15 필수 assertion."""
    cfg = cfg or load_crop_config()
    errs: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    # 계보가 겹치지 않는다
    check(not (full.head_a6_aux_pool & full.head_a7_aux_train).any(),
          "aux_pool 과 a7_aux 가 겹친다")
    check(not (full.head_a6_eligible & full.head_a6_aux_pool).any(),
          "baseline 과 aux_pool 이 겹친다")

    # 봉인분은 crop I/O 시도조차 없어야 한다
    check(not parents["sealed_future_eval"].any(),
          "select_parents 가 sealed 를 걸러내지 못했다")
    sealed = set(full.loc[full.sealed_future_eval, "record_id"])
    check(not crops_df["parent_record_id"].isin(sealed).any(),
          "sealed 행의 crop 이 생성됐다")

    # DACON identity 는 평가 후보 3,352 가 아니라 학습 후보 3,448
    n_dacon = int((crops_df.dataset.eq("dacon_wallpaper")
                   & crops_df["mode"].eq("identity")).sum())
    check(n_dacon == cfg["dacon_identity_expected"],
          f"DACON identity {n_dacon} != {cfg['dacon_identity_expected']}")

    # baseline 이 아닌 lane 은 baseline_eligible=False
    check(not crops_df.loc[crops_df.lane.ne("baseline"), "baseline_eligible"].any(),
          "baseline 이 아닌 lane 의 crop 이 baseline_eligible=True")

    # 정사각 보장 (identity 제외)
    real = crops_df[crops_df["mode"].ne("identity")]
    check(real["crop_width"].eq(real["crop_height"]).all(), "비정사각 crop 이 있다")
    check(real["crop_side"].gt(0).all(), "crop_side <= 0")

    # parent 식별자 상속
    for col in ("duplicate_group_id", "split_component_id", "parent_record_id"):
        check(crops_df[col].notna().all(), f"{col} 상속 누락")

    if errs:
        raise AssertionError("crop 검증 실패:\n  - " + "\n  - ".join(errs))
    return errs


def save(crops_df: pd.DataFrame) -> Path:
    out = load_paths().artifact("metadata") / "crops.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    crops_df.to_csv(out, index=False, encoding=ENCODING)
    return out
