"""configs/label_mapping.csv 생성.

기존 저장소의 dataset_registry.json 을 단일 기준으로 전개한다.
매핑을 여기서 새로 만들지 않는다 — 레지스트리가 기준이고 이 스크립트는 전개만 한다.

    python scripts/gen_label_mapping.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / ".." / "defect_classifying_model" / "configs" / "dataset_registry.json"
OUT = ROOT / "configs" / "label_mapping.csv"

FIELDS = [
    "source_dataset",
    "original_label",
    "unified_label",
    "l2_service_group",
    "l3_detail_label",
    "note",
]


def main() -> None:
    registry = json.loads(REGISTRY.resolve().read_text(encoding="utf-8"))
    l1_names = {c["name"] for c in registry["coarse_detection_classes"]}

    rows: list[dict[str, str]] = []

    for ds_name, ds in registry["datasets"].items():
        service_map = ds.get("service_group_map", {})
        review_only = set(ds.get("review_only_categories", []))
        excluded = set(ds.get("excluded_categories", []))

        # aihub_567 은 class_map, 나머지는 class_map 또는 auxiliary_class_map
        class_map = ds.get("class_map") or ds.get("auxiliary_class_map") or {}

        for original, unified in class_map.items():
            if unified is None:
                note = "excluded_no_l1_mapping"
            elif original in excluded:
                note = "excluded_category"
            elif original in review_only:
                note = "review_category_semantics"
            else:
                note = ""

            if unified is not None and unified not in l1_names and unified != "normal":
                note = (note + ";" if note else "") + f"UNKNOWN_L1:{unified}"

            # DACON 만 L3 상세 라벨을 가진다
            l3 = original if ds_name == "dacon_wallpaper" else ""

            rows.append(
                {
                    "source_dataset": ds_name,
                    "original_label": original,
                    "unified_label": unified or "",
                    "l2_service_group": service_map.get(original, ""),
                    "l3_detail_label": l3,
                    "note": note,
                }
            )

    rows.sort(key=lambda r: (r["source_dataset"], r["original_label"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    verify(rows, l1_names)

    mapped = sum(1 for r in rows if r["unified_label"])
    print(f"wrote {OUT.relative_to(ROOT)}  rows={len(rows)}  mapped={mapped}")
    for ds in sorted({r['source_dataset'] for r in rows}):
        n = sum(1 for r in rows if r["source_dataset"] == ds)
        m = sum(1 for r in rows if r["source_dataset"] == ds and r["unified_label"])
        print(f"  {ds:28s} {m:3d}/{n:3d} mapped")


def verify(written: list[dict[str, str]], l1_names: set[str]) -> None:
    """디스크에서 다시 읽어 round-trip 과 불변식을 검증한다.

    한글 라벨이 cp949 로 깨져 저장되면 이 시점에 실패해야 한다.
    노트북 01 까지 미루면 깨진 CSV 로 crop 을 만든 뒤에야 발견된다.
    """
    reread = list(csv.DictReader(OUT.open(encoding="utf-8-sig")))

    assert list(reread[0].keys()) == FIELDS, f"헤더 불일치: {list(reread[0].keys())}"
    assert len(reread) == len(written), f"행 수 불일치: {len(reread)} != {len(written)}"

    # 한글 round-trip
    for a, b in zip(written, reread):
        assert a == b, f"round-trip 불일치:\n  wrote={a}\n  read ={b}"

    # (source_dataset, original_label) 유일성
    keys = [(r["source_dataset"], r["original_label"]) for r in reread]
    assert len(keys) == len(set(keys)), "중복 키 존재"

    # DACON 19개 정확히
    dacon = {r["original_label"] for r in reread if r["source_dataset"] == "dacon_wallpaper"}
    assert len(dacon) == 19, f"DACON 클래스 수가 19가 아님: {len(dacon)}"
    assert "훼손" in dacon and "창틀,문틀수정" in dacon, "DACON 한글 라벨 깨짐"
    assert all(r["unified_label"] for r in reread if r["source_dataset"] == "dacon_wallpaper"), \
        "DACON 19개는 전부 L1 매핑이 있어야 한다"

    # unified_label 은 L1 7종 또는 normal 또는 공백
    allowed = l1_names | {"normal", ""}
    bad = {r["unified_label"] for r in reread} - allowed
    assert not bad, f"L1 에 없는 unified_label: {bad}"

    print(f"verify OK  rows={len(reread)}  dacon={len(dacon)}  l1={sorted(l1_names)}")


if __name__ == "__main__":
    main()
