"""이미지 품질 측정 (노트북 03).

★ 설계 원칙 — PLAN §3.4 규칙 2
    "모델이 측정할 수 있는 오염은 선제 차단하지 않는다. 측정하고 대응한다."

  따라서 이 모듈은 **측정만** 하고 학습에서 빼지 않는다.
  `quality_status='drop'` 은 **파일이 실제로 깨진 경우에만** 붙는다.
  흐림·어두움·저해상도는 `review` 로 표시하되 학습에는 그대로 들어간다.
  baseline 이 그 영향을 실제로 보여준 뒤에 대응한다.

  기존 프로젝트는 이 순서를 뒤집어 감사 코드 4,325줄을 쓰고 실제 학습은 7분이었다
  (`RESTART_HERE.md` 3절).

★ 수동 검수 상한
  사람이 보는 큐는 100건이 상한이다 (PLAN §3.4 규칙 1).
  자동 플래그 수는 그보다 많을 수 있으나, `build_review_queue()` 가 100건으로 샘플링한다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .paths import ENCODING, load_paths

MANUAL_REVIEW_CAP = 100          # PLAN §3.4 규칙 1
SEED = 20260803

STAT_FIELDS = [
    "sample_id", "readable", "width", "height", "aspect_ratio", "megapixels",
    "mean_brightness", "contrast_std", "saturation", "blur_var",
    "dark_frac", "bright_frac", "file_bytes",
]


def resolve_path(row) -> Path:
    """crop 은 artifacts/crops/ 아래, identity 는 기존 저장소 원본을 가리킨다."""
    paths = load_paths()
    if row.mode == "identity":
        return paths.legacy_root / row.crop_path
    return paths.artifact("crops") / row.crop_path


def measure_one(path: Path) -> dict[str, Any]:
    """이미지 하나의 품질 지표. 실패해도 예외를 던지지 않고 readable=False 로 표시한다."""
    out: dict[str, Any] = {f: None for f in STAT_FIELDS[1:]}
    out["readable"] = False
    try:
        out["file_bytes"] = path.stat().st_size
    except OSError:
        return out

    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 - 진단 목적
        return out
    if img is None or img.size == 0:
        return out

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    out.update({
        "readable": True,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4) if h else None,
        "megapixels": round(w * h / 1e6, 4),
        "mean_brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "saturation": round(float(hsv[:, :, 1].mean()), 2),
        # Laplacian 분산 — 낮을수록 흐리다. 해상도에 의존하므로 절대 임계값을 쓰지 않는다.
        "blur_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "dark_frac": round(float((gray < 25).mean()), 4),
        "bright_frac": round(float((gray > 230).mean()), 4),
    })
    return out


def measure(crops: pd.DataFrame, workers: int = 8, progress: bool = True) -> pd.DataFrame:
    """전 샘플의 품질 지표. I/O 바운드라 스레드로 병렬화한다."""
    from tqdm.auto import tqdm

    rows = list(crops.itertuples())
    paths = [resolve_path(r) for r in rows]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(measure_one, paths), total=len(paths),
                            disable=not progress, desc="quality"))

    for r, res in zip(rows, results):
        res["sample_id"] = r.sample_id
    return pd.DataFrame(results, columns=STAT_FIELDS)


# --------------------------------------------------------------------------- #
# quality_status
# --------------------------------------------------------------------------- #

def assign_quality(crops: pd.DataFrame, stats: pd.DataFrame,
                   pct: float = 0.01) -> pd.DataFrame:
    """quality_status 를 확정한다.

    drop   : 파일이 실제로 깨진 경우만
    review : 통계적 이상치 + 승계된 라벨 오류. **학습에서 빼지 않는다.**
    keep   : 나머지

    임계값은 상수로 정하지 않고 **데이터 분포의 하위 pct 분위**에서 잡는다.
    227x227 kaggle 패치와 1080px AIHub crop 이 섞여 있어 절대 임계값은 의미가 없다.
    """
    df = crops.merge(stats, on="sample_id", how="left", validate="1:1")

    # --- drop: 실제 손상만
    broken = ~df["readable"].fillna(False).astype(bool)

    # --- review: 이상치. 데이터셋별로 분포가 달라 그룹 내 분위수를 쓴다.
    def low_outlier(col: str) -> pd.Series:
        thr = df.groupby("dataset")[col].transform(lambda s: s.quantile(pct))
        return df[col] < thr

    def high_outlier(col: str) -> pd.Series:
        thr = df.groupby("dataset")[col].transform(lambda s: s.quantile(1 - pct))
        return df[col] > thr

    flags = {
        "blur": low_outlier("blur_var"),
        "dark": low_outlier("mean_brightness") | (df["dark_frac"] > 0.90),
        "washed_out": high_outlier("mean_brightness") | (df["bright_frac"] > 0.90),
        "low_contrast": low_outlier("contrast_std"),
        "tiny": df["megapixels"] < 0.005,          # < 약 70x70
    }
    df["quality_flags"] = [
        ";".join(k for k, v in flags.items() if bool(v.iloc[i]))
        for i in range(len(df))
    ]

    outlier = pd.concat(flags.values(), axis=1).any(axis=1)
    inherited_review = df["quality_status"].eq("review")   # crops.py 가 붙인 것

    df["quality_status"] = np.select(
        [broken, outlier | inherited_review],
        ["drop", "review"],
        default="keep",
    )
    df["rejection_reason"] = np.where(
        broken, "unreadable_file",
        np.where(df["rejection_reason"].fillna("").ne(""), df["rejection_reason"],
                 np.where(outlier, "quality_outlier:" + df["quality_flags"], "")),
    )

    # ★ 학습 자격은 건드리지 않는다. drop 만 뺀다 (PLAN §3.4 규칙 2).
    df["trainable"] = df["quality_status"].ne("drop") & df["baseline_eligible"]
    return df


def build_review_queue(df: pd.DataFrame, cap: int = MANUAL_REVIEW_CAP) -> pd.DataFrame:
    """사람이 볼 검수 큐. **상한 100건** (PLAN §3.4 규칙 1).

    자동 플래그가 100건을 넘어도 사람에게는 100건만 보낸다.
    상한을 늘리려면 baseline 결과로 필요성을 입증해야 한다.
    """
    cand = df[df["quality_status"].eq("review")].copy()
    if cand.empty:
        return cand

    # 클래스·lane 을 고르게 섞어 한 클래스만 보게 되는 것을 막는다
    rng = np.random.default_rng(SEED)
    cand["_r"] = rng.random(len(cand))
    per = max(1, cap // max(1, cand.groupby(["lane", "unified_label"]).ngroups))
    q = (cand.sort_values("_r")
             .groupby(["lane", "unified_label"], group_keys=False)
             .head(per))
    if len(q) < cap:                       # 남은 자리를 무작위로 채운다
        rest = cand[~cand.sample_id.isin(q.sample_id)].sort_values("_r").head(cap - len(q))
        q = pd.concat([q, rest])
    return q.sort_values("_r").head(cap).drop(columns="_r")


# --------------------------------------------------------------------------- #
# 검증 / 저장
# --------------------------------------------------------------------------- #

def verify(df: pd.DataFrame, queue: pd.DataFrame) -> list[str]:
    errs: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    check(len(queue) <= MANUAL_REVIEW_CAP,
          f"수동 검수 큐 {len(queue)} > 상한 {MANUAL_REVIEW_CAP} (PLAN §3.4 규칙 1)")
    check(df["quality_status"].isin({"keep", "review", "drop"}).all(),
          "quality_status 에 허용되지 않은 값")

    # drop 은 손상 파일에만
    bad_drop = df[df.quality_status.eq("drop") & df.readable.fillna(False)]
    check(bad_drop.empty,
          f"읽히는데 drop 된 샘플 {len(bad_drop)}건 — 규칙 2 위반")

    # review 는 학습을 막지 않는다
    blocked = df[df.quality_status.eq("review") & df.baseline_eligible & ~df.trainable]
    check(blocked.empty,
          f"review 가 학습을 막고 있다 {len(blocked)}건 — 규칙 2 위반")

    # 승계된 annotation_error 는 여전히 review
    ae = df[df.rejection_reason.fillna("").eq("legacy_annotation_error")]
    check(ae.empty or ae.quality_status.eq("review").all(),
          "annotation_error 가 review 가 아니다")

    check(df["sample_id"].is_unique, "sample_id 중복")

    if errs:
        raise AssertionError("품질 검증 실패:\n  - " + "\n  - ".join(errs))
    return errs


def save(df: pd.DataFrame, queue: pd.DataFrame) -> tuple[Path, Path]:
    paths = load_paths()
    out = paths.artifact("metadata") / "crops_quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding=ENCODING)

    qcols = ["sample_id", "crop_path", "dataset", "unified_label", "lane",
             "quality_flags", "blur_var", "mean_brightness", "contrast_std",
             "width", "height", "rejection_reason"]
    qout = paths.artifact("review") / "03_quality_queue.csv"
    qout.parent.mkdir(parents=True, exist_ok=True)
    q = queue[[c for c in qcols if c in queue.columns]].copy()
    q["human_verdict"] = ""            # 사람이 채운다. 비워 둔다.
    q["human_note"] = ""
    q.to_csv(qout, index=False, encoding=ENCODING)
    return out, qout
