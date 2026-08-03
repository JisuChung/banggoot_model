"""신규 crop 을 기존 2,783 crop 과 회귀검증한다 (D-15 규칙 7).

기존 산출물은 **기준으로만** 쓰고 승계하지 않는다.
좌표는 ±1px 를 허용한다 — 기존 구현이 x0/y0/x1/y1 을 각각 반올림해
685장이 1px 비정사각이 됐기 때문이다(그중 471장은 이미지 내부).

비교 대상은 기존 파이프라인의 모집단으로 제한한다:
    legacy_split == train  AND  severity in (normal, poor)

    .venv\\Scripts\\python.exe scripts/regress_crops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banggoot.paths import ENCODING, load_paths  # noqa: E402

TOL = 1  # legacy_rounding_tolerance_px


def main() -> int:
    paths = load_paths()
    legacy = pd.read_csv(
        paths.reference_only("aihub_positive_crops") / "crop_manifest.csv", encoding=ENCODING
    )
    new = pd.read_csv(paths.artifact("metadata") / "crops.csv", encoding=ENCODING)
    ann = pd.read_csv(paths.inherit("aihub_polygon_annotations"), encoding=ENCODING)

    new = new[new["mode"].eq("polygon_crop")].copy()
    new["stem"] = new.parent_image_path.map(lambda p: Path(p).stem)

    # 신규 crop 은 box 순서(i)로 만들어지므로 annotations 의 행 순서와 대응한다.
    ann = ann.reset_index(drop=True)
    ann["i"] = ann.groupby("stem").cumcount()
    new["i"] = new.sample_id.str.rsplit("#", n=1).str[-1].astype(int)

    # 기존은 dedupe 후 남은 것만 있으므로, 소스 bbox 로 매칭한다.
    key = ["stem", "cx5", "cy5"]
    ann["cx5"], ann["cy5"] = ann.cx.round(5), ann.cy.round(5)
    legacy["stem"] = legacy.parent_stem
    legacy["cx5"] = legacy.source_bbox_cx.round(5)
    legacy["cy5"] = legacy.source_bbox_cy.round(5)

    # 신규 crop <- annotations(i) 로 소스 bbox 부착
    new = new.merge(ann[["stem", "i", "cx5", "cy5", "severity"]],
                    on=["stem", "i"], how="left", suffixes=("", "_ann"))

    merged = legacy.merge(new, on=key, how="inner", suffixes=("_old", "_new"))

    print(f"legacy crops              : {len(legacy):,}")
    print(f"신규 AIHub crops          : {len(new):,}")
    print(f"소스 bbox 로 매칭된 쌍    : {len(merged):,}")
    print(f"legacy 중 매칭 실패       : {len(legacy) - len(merged):,}")

    if merged.empty:
        print("\n매칭 0건 — 회귀검증 불가")
        return 1

    dx = (merged.crop_box_x0 - merged.crop_x0).abs()
    dy = (merged.crop_box_y0 - merged.crop_y0).abs()
    dside = (merged.crop_width_old - merged.crop_side).abs()

    within = (dx <= TOL) & (dy <= TOL) & (dside <= TOL)
    print(f"\n±{TOL}px 이내 일치          : {int(within.sum()):,} / {len(merged):,} "
          f"({within.mean():.2%})")
    for name, d in (("x0", dx), ("y0", dy), ("side", dside)):
        print(f"  {name:5s} 최대 {int(d.max()):4d}px · 중앙 {d.median():.1f}px "
              f"· >{TOL}px {int((d > TOL).sum()):,}건")

    if not within.all():
        bad = merged.loc[~within, ["stem", "crop_box_x0", "crop_x0",
                                   "crop_box_y0", "crop_y0",
                                   "crop_width_old", "crop_side"]].head(10)
        print("\n불일치 표본:")
        print(bad.to_string(index=False))

    # 신규가 더 많은 이유 — 기존이 제외한 것들
    extra = len(new) - len(merged)
    print(f"\n신규에만 있는 crop        : {extra:,}")
    only = new[~new.index.isin(
        new.merge(legacy[key], on=key, how="inner").index)]
    if len(only):
        print(only.severity_ann.value_counts().to_frame("severity").to_string())
    return 0 if within.all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
