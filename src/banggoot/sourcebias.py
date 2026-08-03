"""출처 편향 측정 (노트북 03 사전 증거 · 노트북 08 본 측정).

★ 해석 규율 (PLAN §6.3)
  높은 출처 예측 정확도는 **"shortcut 이 가능하다"까지만** 의미한다.
  분류기가 실제로 그 신호를 사용했다는 증명이 아니다.
  실사용 여부는 노트북 08 의 source-ID probe(픽셀 기반)와 LOSO 로 측정한다.

★ feature tier 를 나누는 이유
  `megapixels` 와 `aspect_ratio` 는 **모델이 보지 못하는 정보**다.
  학습은 224x224 로 resize 한 뒤 이뤄지므로 원본 크기·종횡비는 소실된다.
  이 둘을 포함한 수치는 **upper bound** 이지 모델이 쓸 수 있는 shortcut 크기가 아니다.

  tier            포함                                          의미
  ----            ----                                          ----
  full            8개 (크기·비율 포함)                          upper bound
  no_size         6개 (크기·비율 제외)                          모델 가시 정보에 근접
  photometric     5개 (blur_var 도 제외 — 해상도 의존)          가장 보수적
  resized_224     224 resize 후 재측정한 photometric            모델이 실제로 보는 것

★ 교차검증
  일반 KFold 를 쓰면 같은 `split_component_id` 의 crop 이 train/test 양쪽에 들어가
  정확도가 부풀려진다. StratifiedGroupKFold(groups=split_component_id) 를 쓴다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from .imagestats import resolve_path

SEED = 1
MODEL_INPUT = 224

TIERS: dict[str, list[str]] = {
    # upper bound — 크기·종횡비 포함. 모델은 resize 후라 이 정보를 보지 못한다.
    "full": ["blur_var", "mean_brightness", "contrast_std", "saturation",
             "megapixels", "aspect_ratio", "dark_frac", "bright_frac"],
    # 크기·종횡비 제외
    "no_size": ["blur_var", "mean_brightness", "contrast_std", "saturation",
                "dark_frac", "bright_frac"],
    # blur_var 도 제외 — 해상도에 의존하므로 크기 정보가 새어 들어간다
    "photometric": ["mean_brightness", "contrast_std", "saturation",
                    "dark_frac", "bright_frac"],
}

RESIZED_FEATURES = ["r_blur_var", "r_mean_brightness", "r_contrast_std",
                    "r_saturation", "r_dark_frac", "r_bright_frac"]


# --------------------------------------------------------------------------- #
# 224 resize 후 재측정 — 모델이 실제로 보는 통계
# --------------------------------------------------------------------------- #

def _measure_resized(row) -> dict[str, Any]:
    out = {f: np.nan for f in RESIZED_FEATURES}
    try:
        img = cv2.imdecode(np.fromfile(str(resolve_path(row)), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return out
        img = cv2.resize(img, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out.update({
            "r_blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "r_mean_brightness": float(gray.mean()),
            "r_contrast_std": float(gray.std()),
            "r_saturation": float(hsv[:, :, 1].mean()),
            "r_dark_frac": float((gray < 25).mean()),
            "r_bright_frac": float((gray > 230).mean()),
        })
    except Exception:  # noqa: BLE001 - 진단 목적
        pass
    return out


def measure_resized(df: pd.DataFrame, workers: int = 8,
                    progress: bool = True) -> pd.DataFrame:
    """224x224 resize 후 photometric 통계. 모델 입력과 같은 전처리다."""
    from tqdm.auto import tqdm

    rows = list(df.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(tqdm(ex.map(_measure_resized, rows), total=len(rows),
                        disable=not progress, desc="resized-224"))
    out = pd.DataFrame(res, columns=RESIZED_FEATURES)
    out["sample_id"] = [r.sample_id for r in rows]
    return out


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #

def majority_accuracy(y: pd.Series) -> float:
    return float(y.value_counts(normalize=True).max())


def majority_macro_f1(y: pd.Series) -> float:
    """모두 다수 클래스로 예측했을 때의 macro F1.

    다수 클래스만 F1 = 2p/(p+1) 이고 나머지는 0 이므로 그 값을 클래스 수로 나눈다.
    """
    p = float(y.value_counts(normalize=True).max())
    return (2 * p / (p + 1)) / y.nunique()


def probe(df: pd.DataFrame, features: list[str], target: str,
          groups: str = "split_component_id", scoring: str = "accuracy",
          n_splits: int = 5) -> dict[str, Any]:
    """StratifiedGroupKFold 로 target 예측력을 잰다.

    같은 component 의 crop 이 train/test 에 나뉘면 수치가 부풀려지므로
    반드시 그룹을 묶는다.
    """
    sub = df.dropna(subset=features + [target, groups])
    # 표본이 너무 적은 클래스는 StratifiedGroupKFold 가 실패한다
    keep = sub[target].map(sub[target].value_counts()) >= n_splits
    sub = sub[keep]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    scores = cross_val_score(clf, sub[features], sub[target],
                             groups=sub[groups], cv=cv, scoring=scoring, n_jobs=1)
    return {
        "target": target,
        "scoring": scoring,
        "n_features": len(features),
        "n_samples": len(sub),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "majority": (majority_accuracy(sub[target]) if scoring == "accuracy"
                     else majority_macro_f1(sub[target])),
    }


def run_all(df: pd.DataFrame, resized: pd.DataFrame | None = None) -> pd.DataFrame:
    """tier 별로 출처·라벨 예측력을 잰다."""
    if resized is not None:
        df = df.merge(resized, on="sample_id", how="left", validate="1:1")

    tiers = dict(TIERS)
    if resized is not None:
        tiers["resized_224"] = RESIZED_FEATURES

    rows = []
    for name, feats in tiers.items():
        if not set(feats).issubset(df.columns):
            continue
        for target, scoring in (("dataset", "accuracy"), ("unified_label", "f1_macro")):
            r = probe(df, feats, target, scoring=scoring)
            r["tier"] = name
            r["lift_over_majority"] = r["mean"] - r["majority"]
            rows.append(r)

    out = pd.DataFrame(rows)
    cols = ["tier", "target", "scoring", "n_features", "n_samples",
            "mean", "std", "majority", "lift_over_majority"]
    return out[cols].round(4)


def interpretation(table: pd.DataFrame) -> str:
    """수치를 과잉 해석하지 않도록 고정 문구를 붙인다."""
    def get(tier: str, target: str) -> float | None:
        m = table[(table.tier == tier) & (table.target == target)]
        return float(m["mean"].iloc[0]) if len(m) else None

    lines = [
        "해석 규율:",
        "",
        f"  full tier ({get('full', 'dataset'):.4f}) 는 megapixels / aspect_ratio 를 포함하므로",
        "  **upper bound** 다. 모델은 224 resize 후 학습하므로 원본 크기·종횡비를 보지 못한다.",
        "  이 수치를 '모델이 쓸 수 있는 shortcut 크기'로 인용하면 안 된다.",
        "",
    ]
    for tier, label in (("no_size", "크기·비율 제외"),
                        ("photometric", "photometric only"),
                        ("resized_224", "224 resize 후 (모델 입력과 동일)")):
        v = get(tier, "dataset")
        if v is not None:
            lines.append(f"  {label:32s} 출처 accuracy {v:.4f}")
    lines += [
        "",
        "  어느 tier 에서도 출처가 majority 를 크게 상회하면 shortcut 이 **가능하다**.",
        "  실제로 사용했는지는 노트북 08 의 픽셀 기반 source-ID probe 와",
        "  leave-one-source-out (crack / breakage 만 가능) 으로 측정한다.",
    ]
    return "\n".join(lines)
