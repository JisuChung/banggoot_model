"""신규 crop 을 기존 2,783 crop 과 회귀검증한다 (D-15 규칙 7).

기존 산출물은 **기준으로만** 쓰고 승계하지 않는다.
좌표는 ±1px 를 허용한다 — 기존 구현이 x0/y0/x1/y1 을 각각 반올림해
685장이 1px 비정사각이 됐기 때문이다(그중 471장은 이미지 내부).

★ 매칭 키는 **seed bbox 좌표 전체**(cx, cy, w, h)다.
  1차 구현은 sample_id 의 `#i`(dedupe **이후** 순번)를 annotations 의 cumcount
  (dedupe **이전** 순번)와 같다고 보고 매칭했다. dedupe 가 일어난 parent 에서
  순번이 어긋나 8건이 "미매칭"으로 잘못 보고됐다. 실제 미매칭은 0건이다.

    .venv\\Scripts\\python.exe scripts/regress_crops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banggoot.paths import ENCODING, load_paths  # noqa: E402

TOL = 1        # legacy_rounding_tolerance_px
ROUND = 6      # 좌표 매칭 정밀도


def main() -> int:
    paths = load_paths()
    legacy = pd.read_csv(
        paths.reference_only("aihub_positive_crops") / "crop_manifest.csv", encoding=ENCODING
    )
    new = pd.read_csv(paths.artifact("metadata") / "crops.csv", encoding=ENCODING)
    new = new[new["mode"].eq("polygon_crop")].copy()
    new["stem"] = new.parent_image_path.map(lambda p: Path(p).stem)

    # ---- 매칭 키: parent stem + seed bbox 전체 (순번 추론 없음) ----
    key = ["stem", "kcx", "kcy", "kw", "kh"]
    for df, (cx, cy, w, h) in ((legacy, ("source_bbox_cx", "source_bbox_cy",
                                         "source_bbox_w", "source_bbox_h")),
                               (new, ("source_bbox_cx", "source_bbox_cy",
                                      "source_bbox_w", "source_bbox_h"))):
        df["kcx"] = df[cx].round(ROUND)
        df["kcy"] = df[cy].round(ROUND)
        df["kw"] = df[w].round(ROUND)
        df["kh"] = df[h].round(ROUND)
    legacy["stem"] = legacy["parent_stem"]

    for name, df in (("legacy", legacy), ("new", new)):
        dup = int(df.duplicated(key).sum())
        if dup:
            print(f"경고: {name} 에 중복 키 {dup}건 — 매칭이 부정확할 수 있다")

    merged = legacy.merge(new, on=key, how="inner", suffixes=("_old", "_new"))

    print(f"legacy crops              : {len(legacy):,}")
    print(f"신규 AIHub crops          : {len(new):,}")
    print(f"seed bbox 로 매칭된 쌍    : {len(merged):,}")
    print(f"legacy 중 미매칭          : {len(legacy) - len(merged):,}")

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
        print("\n불일치 표본:")
        print(merged.loc[~within, ["stem", "crop_box_x0", "crop_x0",
                                   "crop_box_y0", "crop_y0",
                                   "crop_width_old", "crop_side"]].head(10).to_string(index=False))

    # ---- 신규에만 있는 crop: 키 기준 anti-join (index 추론 금지) ----
    matched_keys = set(map(tuple, merged[key].to_numpy()))
    only_mask = ~new[key].apply(tuple, axis=1).isin(matched_keys)
    only = new[only_mask]
    print(f"\n신규에만 있는 crop        : {len(only):,}")
    if len(only):
        print(only.severity.value_counts().to_frame("crops").to_string())
        print("  (legacy 는 legacy_split=train + severity in (normal, poor) 만 대상이었다)")

    ok = bool(within.all()) and len(merged) == len(legacy)
    print(f"\n회귀검증 {'통과' if ok else '실패'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
