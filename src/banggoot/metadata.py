"""metadata.csv v0 — record 단위 (노트북 01).

★ 이 단계에서 만들지 않는 것 (D-12):
    split · eval_eligible_head_a · canonical_eval_head_a
  전부 노트북 04 소관이다. 승계한 split 컬럼은 폐기 대상인 기존 80/10/10 배정이므로
  legacy_split 으로 개명해 보존만 하고 어떤 파생에도 쓰지 않는다.

파생 순서 (D-12):
    excluded_by_policy -> train_eligible_head_a -> train_auxiliary
    -> sample_role -> eval_candidate_head_a -> representative_rank
    -> 계보 플래그 (head_a6_eligible / head_a7_aux_train / sealed_future_eval)
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .paths import ENCODING, l1_classes, load_paths, read_csv

SEED = 20260802

# --- 정책 상수 (D-08) --------------------------------------------------------
EXCLUDED_DATASETS = {"mvtec_ad"}                       # 라이선스 CC BY-NC-SA

BLOCKING_RESERVED_REASONS = {
    "contains_dacon_public_test",      # 832 — split 컬럼에서는 reserved_inference
    "mvtec_official_protocol",         # 673
    "aihub_category_needs_review",     # 945
    "out_of_scope_category",           # 220
}
# 학습은 되지만 평가는 불가
EVAL_BLOCKING_RESERVED_REASONS = {
    "no_eval_eligible_origin_in_component",   # 3,814
    "rare_detail_class_train_only",           # 115
}

BLOCKING_USE_STATUS = {
    "excluded",
    "inference_only_no_ground_truth",
    "anomaly_train",
    "anomaly_test_locked",
}

AUXILIARY_ORIGINS = {
    "unknown_origin",
    "dacon_derivative_verified",
    "dacon_derivative_probable",
}

KAGGLE_CRACK_CAP = 1500          # D-03
SEAL_FRACTION = 0.30             # D-14 — RWD moisture component 봉인 비율

EXPECTED_ROWS = 54_797


# --------------------------------------------------------------------------- #
# 1. 3-way join
# --------------------------------------------------------------------------- #

def load_joined() -> pd.DataFrame:
    """inventory ⟕ split_manifest ⟕ origin_decisions on record_id.

    inventory 를 빼면 label_path 가 없어 crop 을 만들 수 없다 (노트북 02).
    origin_decisions 를 빼면 강등 이전 origin 을 상속한다 (D-12).
    """
    paths = load_paths()

    inv = pd.DataFrame(read_csv(paths.inherit("dataset_inventory")))
    spl = pd.DataFrame(read_csv(paths.inherit("split_manifest")))
    org = pd.DataFrame(read_csv(paths.inherit("origin_decisions")))

    for name, df in [("inventory", inv), ("split_manifest", spl), ("origin_decisions", org)]:
        assert len(df) == EXPECTED_ROWS, f"{name} 행 수 {len(df)} != {EXPECTED_ROWS}"
        assert df["record_id"].is_unique, f"{name} record_id 중복"

    assert set(inv.record_id) == set(spl.record_id) == set(org.record_id), \
        "세 파일의 record_id 집합이 다르다"

    spl_cols = [
        "record_id", "split", "split_component_id", "duplicate_group_id",
        "source_root_id", "reserved_reason",
    ]
    org_cols = ["record_id", "effective_origin", "baseline_origin", "quarantined",
                "quarantine_kind", "origin_note"]

    df = (inv
          .merge(spl[spl_cols], on="record_id", how="left", validate="1:1")
          .merge(org[org_cols], on="record_id", how="left", validate="1:1"))

    assert len(df) == EXPECTED_ROWS, f"조인 후 행 수 {len(df)}"
    assert df["effective_origin"].notna().all(), "origin 조인 실패 행 존재"

    # 폐기 대상인 기존 80/10/10 배정. 실수로 쓰지 못하게 개명한다 (D-12).
    df = df.rename(columns={"split": "legacy_split"})
    df["quarantined"] = df["quarantined"].eq("true")
    return df


# --------------------------------------------------------------------------- #
# 2. 라벨
# --------------------------------------------------------------------------- #

def apply_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    paths = load_paths()
    lm = pd.DataFrame(read_csv(paths.artifact("root").parent / "configs" / "label_mapping.csv"))
    lm = lm.rename(columns={"source_dataset": "dataset", "original_label": "source_class"})

    df = df.merge(
        lm[["dataset", "source_class", "unified_label", "l2_service_group", "l3_detail_label"]],
        on=["dataset", "source_class"], how="left", validate="m:1",
    )

    # target_class 가 이미 L1 인 데이터셋(roboflow 등)은 그대로 신뢰한다.
    l1 = set(l1_classes())
    direct = df["target_class"].isin(l1)
    df.loc[direct, "unified_label"] = df.loc[direct, "target_class"]

    df["unified_label"] = df["unified_label"].fillna("")
    df["is_multilabel"] = df["target_class"].fillna("").str.contains(";")
    df["all_original_labels"] = df["target_class"].fillna("")
    return df


# --------------------------------------------------------------------------- #
# 3. eligibility 파생 (D-12)
# --------------------------------------------------------------------------- #

def apply_label_audit(df: pd.DataFrame) -> pd.DataFrame:
    """AIHub 라벨 감사의 annotation_error 를 승계한다 (D-15).

    기존 B 파이프라인이 제외했던 15건이다. 플래그만 달고 두면 학습과 **평가 양쪽에**
    라벨 오류가 재유입된다. 평가 유입이 더 위험하다 — 틀린 정답으로 지표를 계산하게 된다.

    검수자는 agent_visual_review / human_verified=false 이므로 "확정 오류"가 아니라
    "사람 검토 필요"로 다룬다. 삭제하지 않는다.
    """
    audit = pd.DataFrame(read_csv(load_paths().inherit("aihub_label_audit")))
    err = set(audit.loc[audit["annotation_error"].eq("true"), "record_id"])

    df["legacy_annotation_error"] = df["record_id"].isin(err)
    df["legacy_audit_judgement"] = df["record_id"].map(
        dict(zip(audit["record_id"], audit["judgement"]))
    ).fillna("")

    missing = err - set(df["record_id"])
    assert not missing, f"audit record_id 가 metadata 에 없다: {sorted(missing)[:5]}"
    return df


def derive_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    l1 = set(l1_classes())

    # --- 1. 정책 제외. origin 보다 먼저 본다.
    #        origin 만 쓰면 MVTec 673장(그중 133장이 L1 매핑)이 재유입된다.
    df["excluded_by_policy"] = (
        df["dataset"].isin(EXCLUDED_DATASETS)
        | df["reserved_reason"].isin(BLOCKING_RESERVED_REASONS)
        | df["use_status"].isin(BLOCKING_USE_STATUS)
        | ~df["unified_label"].isin(l1)
        | df["is_multilabel"]                     # D-09 — crop 단계에서 재평가
        | df["legacy_annotation_error"]           # D-15 — 15건, 학습·평가 모두 차단
    )

    # --- 2. 학습 권한
    df["train_eligible_head_a"] = (
        ~df["excluded_by_policy"]
        & ~df["quarantined"]
        & df["effective_origin"].ne("synthetic_verified")      # D-07
    )

    # --- 3. auxiliary (baseline 제외, ablation 전용)
    df["train_auxiliary"] = (
        df["train_eligible_head_a"] & df["effective_origin"].isin(AUXILIARY_ORIGINS)
    )

    # --- 4. Kaggle 상한 (D-03). 제외분은 삭제하지 않고 표시만 한다.
    df["kaggle_subsampled_out"] = _kaggle_subsample_mask(df)
    df.loc[df["kaggle_subsampled_out"], "train_eligible_head_a"] = False

    # --- 5. sample_role
    df["sample_role"] = "excluded"
    df.loc[df["train_eligible_head_a"], "sample_role"] = "primary"
    df.loc[df["train_auxiliary"] & df["train_eligible_head_a"], "sample_role"] = "auxiliary"

    df["rejection_reason"] = _rejection_reason(df)

    # --- 6. 평가 "후보". split 을 참조하지 않는다.
    #        eval_eligible_head_a 는 노트북 04 에서 확정한다.
    df["eval_candidate_head_a"] = (
        df["sample_role"].eq("primary")
        & df["effective_origin"].eq("real_verified")
        & ~df["reserved_reason"].isin(EVAL_BLOCKING_RESERVED_REASONS)
    )

    # annotation_error 는 사람 검토 대상으로 표시한다 (D-15). 삭제하지 않는다.
    df["quality_status"] = "keep"
    df.loc[df["legacy_annotation_error"], "quality_status"] = "review"

    # --- 7. 대표본 tie-break 순위. 실제 선택은 노트북 04 (품질 확인 후).
    df["representative_rank"] = _representative_rank(df)
    return df


def _stable_hash(values: pd.Series, salt: str) -> pd.Series:
    def h(v: str) -> int:
        return int(hashlib.sha256(f"{salt}|{SEED}|{v}".encode()).hexdigest()[:16], 16)

    return values.map(h)


def _kaggle_subsample_mask(df: pd.DataFrame) -> pd.Series:
    """kaggle crack 을 duplicate_group 단위로 KAGGLE_CRACK_CAP 까지만 남긴다."""
    mask = pd.Series(False, index=df.index)
    target = df["dataset"].eq("kaggle_cracks") & df["unified_label"].eq("crack")
    if not target.any():
        return mask

    sub = df.loc[target, ["duplicate_group_id"]].copy()
    groups = sub["duplicate_group_id"].drop_duplicates()
    order = _stable_hash(groups, "kaggle").sort_values().index
    ordered_groups = groups.loc[order]

    sizes = sub["duplicate_group_id"].value_counts()
    keep, total = set(), 0
    for g in ordered_groups:
        if total >= KAGGLE_CRACK_CAP:
            break
        keep.add(g)
        total += int(sizes.get(g, 0))

    mask.loc[target] = ~sub["duplicate_group_id"].isin(keep)
    return mask


def _representative_rank(df: pd.DataFrame) -> pd.Series:
    h = _stable_hash(df["record_id"], "representative")
    return h.groupby(df["duplicate_group_id"]).rank(method="first").astype(int)


def _rejection_reason(df: pd.DataFrame) -> pd.Series:
    r = pd.Series("", index=df.index)
    l1 = set(l1_classes())
    rules = [
        (df["dataset"].isin(EXCLUDED_DATASETS), "license_excluded_dataset"),
        (df["reserved_reason"].isin(BLOCKING_RESERVED_REASONS), "reserved_" + df["reserved_reason"]),
        (df["use_status"].isin(BLOCKING_USE_STATUS), "use_status_" + df["use_status"]),
        (~df["unified_label"].isin(l1), "no_l1_mapping"),
        (df["is_multilabel"], "multilabel_pending_crop_split"),
        (df["legacy_annotation_error"], "legacy_annotation_error"),
        (df["quarantined"], "quarantined_" + df["quarantine_kind"].fillna("")),
        (df["effective_origin"].eq("synthetic_verified"), "synthetic_verified"),
        (df["kaggle_subsampled_out"], "kaggle_subsample_not_selected"),
    ]
    for mask, reason in rules:
        blank = r.eq("") & mask
        r.loc[blank] = reason[blank] if isinstance(reason, pd.Series) else reason
    return r


# --------------------------------------------------------------------------- #
# 4. 계보 플래그 (D-14)
# --------------------------------------------------------------------------- #

def derive_lineages(df: pd.DataFrame) -> pd.DataFrame:
    """head_a6_primary(공식) / head_a7_rwd_aux(탐색) / 봉인.

    ★ 봉인은 행이 아니라 split_component_id 전체에 적용한다.
      moisture 행만 봉인하면 같은 component 의 crack 4장(comp_040956 ×3,
      comp_040979 ×1)이 aux 학습에 들어가 미래 holdout 이 오염된다.
    """
    is_rwd_moisture = df["dataset"].eq("roboflow_wall_defects") & df["unified_label"].eq("moisture_leak")

    comps = df.loc[is_rwd_moisture, "split_component_id"].dropna().drop_duplicates()
    ranked = _stable_hash(comps, "seal").sort_values()
    n_seal = max(1, round(len(comps) * SEAL_FRACTION))
    sealed = set(comps.loc[ranked.index[:n_seal]])

    # component 전체에 적용 — moisture 아닌 행도 포함된다
    df["sealed_future_eval"] = df["split_component_id"].isin(sealed)

    # 공식 baseline. ★ train_eligible 이 아니라 sample_role=='primary' 를 본다.
    #   train_eligible 은 auxiliary 도 True 이므로 그걸 쓰면 "auxiliary 는 baseline 제외"
    #   정책이 깨진다 (실제로 3,169장이 새어 들어갔다).
    df["head_a6_eligible"] = (
        df["sample_role"].eq("primary")
        & df["unified_label"].ne("moisture_leak")
        & ~df["sealed_future_eval"]
    )

    # ablation arm 전용 pool. baseline 에 넣지 않는다.
    # 이 컬럼이 없으면 arm 마다 세 조건을 다시 쓰게 되고 같은 실수가 반복된다.
    df["head_a6_aux_pool"] = (
        df["sample_role"].eq("auxiliary")
        & df["unified_label"].ne("moisture_leak")
        & ~df["sealed_future_eval"]
    )

    df["head_a7_aux_train"] = (
        df["train_eligible_head_a"]
        & df["unified_label"].eq("moisture_leak")
        & ~df["sealed_future_eval"]
        & ~df["quarantined"]
    )
    return df


# --------------------------------------------------------------------------- #
# 5. 검증
# --------------------------------------------------------------------------- #

def verify(df: pd.DataFrame) -> list[str]:
    """실패는 예외로 던진다. 조용히 통과시키지 않는다."""
    errs: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    check(len(df) == EXPECTED_ROWS, f"행 수 {len(df)}")

    # D-08 — 잠금 수량 3층
    check(df["use_status"].eq("inference_only_no_ground_truth").sum() == 792,
          "DACON 공개 test != 792")
    check(df["reserved_reason"].eq("contains_dacon_public_test").sum() == 832,
          "contains_dacon_public_test != 832")
    check(df["legacy_split"].eq("reserved_inference").sum() == 832,
          "legacy_split reserved_inference != 832")
    check(set(df.loc[df.reserved_reason.eq("contains_dacon_public_test"), "record_id"])
          == set(df.loc[df.legacy_split.eq("reserved_inference"), "record_id"]),
          "두 잠금 컬럼의 대상이 다르다")
    check("reserved_inference" not in set(df["reserved_reason"]),
          "reserved_reason 에 reserved_inference 가 있다 (컬럼 혼동)")

    # D-08 / D-12 — MVTec 재유입 차단
    check(df["dataset"].eq("mvtec_ad").sum() == 673, "MVTec != 673")
    check(df.loc[df.dataset.eq("mvtec_ad"), "sample_role"].eq("excluded").all(),
          "MVTec 중 excluded 아닌 행 존재")
    check(not df.loc[df.train_eligible_head_a, "dataset"].eq("mvtec_ad").any(),
          "MVTec 이 학습 적격에 포함됨")

    # D-12 — 강등 반영
    rwd = df["dataset"].eq("roboflow_wall_defects")
    check(not df.loc[rwd, "effective_origin"].eq("real_verified").any(),
          "RWD 에 real_verified 가 남아 있다 (origin_decisions 미반영)")
    check(df["quarantined"].sum() == 7, f"quarantined {df['quarantined'].sum()} != 7")

    # D-14 — 봉인은 component 전체
    check(not df.groupby("split_component_id")["sealed_future_eval"].nunique().gt(1).any(),
          "component 안에서 봉인 여부가 갈린다")
    check(not df.loc[df.sealed_future_eval, ["head_a6_eligible", "head_a7_aux_train"]].any().any(),
          "봉인 행이 학습 계보에 포함됨")
    check(not df.loc[df.head_a6_eligible, "unified_label"].eq("moisture_leak").any(),
          "공식 6-class 에 moisture_leak 이 포함됨")

    # ★ 공식 baseline 은 primary 만. auxiliary 는 ablation 전용이다 (D-12).
    check(df.loc[df.head_a6_eligible, "sample_role"].eq("primary").all(),
          "head_a6_eligible 에 primary 아닌 행이 있다")
    check(not df.loc[df.head_a6_eligible, "train_auxiliary"].any(),
          "head_a6_eligible 에 auxiliary 가 섞였다")
    check(not (df["head_a6_eligible"] & df["head_a6_aux_pool"]).any(),
          "baseline 과 ablation pool 이 겹친다")

    # D-15 — annotation_error 15건은 학습·평가 모두 차단
    n_err = int(df["legacy_annotation_error"].sum())
    check(n_err == 15, f"legacy_annotation_error {n_err} != 15")
    check(not df.loc[df.legacy_annotation_error, "train_eligible_head_a"].any(),
          "annotation_error 가 학습에 포함됨")
    check(not df.loc[df.legacy_annotation_error, "eval_candidate_head_a"].any(),
          "annotation_error 가 평가 후보에 포함됨")
    check(not df.loc[df.legacy_annotation_error, "head_a6_eligible"].any(),
          "annotation_error 가 공식 baseline 에 포함됨")
    check(df.loc[df.legacy_annotation_error, "quality_status"].eq("review").all(),
          "annotation_error 가 review 로 표시되지 않음")

    # D-03
    kaggle_kept = (df["dataset"].eq("kaggle_cracks") & df["train_eligible_head_a"]).sum()
    check(kaggle_kept <= KAGGLE_CRACK_CAP * 1.2,
          f"kaggle 상한 초과: {kaggle_kept}")

    # 노트북 01 이 만들면 안 되는 컬럼 (D-12)
    for col in ("split", "eval_eligible_head_a", "canonical_eval_head_a"):
        check(col not in df.columns, f"{col} 은 노트북 04 소관인데 01 에서 생성됨")

    if errs:
        raise AssertionError("metadata 검증 실패:\n  - " + "\n  - ".join(errs))
    return errs


def build() -> pd.DataFrame:
    df = load_joined()
    df = apply_label_mapping(df)
    df = apply_label_audit(df)
    df = derive_eligibility(df)
    df = derive_lineages(df)
    verify(df)
    return df


def save(df: pd.DataFrame) -> Any:
    out = load_paths().artifact("metadata") / "metadata.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding=ENCODING)

    # round-trip — 한글 라벨이 깨졌는지 즉시 확인한다 (P8)
    # 빈 문자열은 재로딩 시 NaN 이 되므로 비어 있지 않은 값만 비교한다.
    back = pd.read_csv(out, encoding=ENCODING, low_memory=False)
    assert len(back) == len(df), f"round-trip 행 수 {len(back)} != {len(df)}"
    assert set(back.columns) == set(df.columns), "round-trip 컬럼 불일치"

    for col in ("unified_label", "source_class", "l3_detail_label"):
        want = {v for v in df[col].fillna("").astype(str) if v}
        got = {v for v in back[col].fillna("").astype(str) if v}
        assert want == got, (
            f"round-trip 값 불일치 ({col}, 인코딩 의심)\n"
            f"  누락: {sorted(want - got)[:5]}\n"
            f"  추가: {sorted(got - want)[:5]}"
        )
    return out


def _md_table(frame: pd.DataFrame, index_name: str = "") -> str:
    """markdown 표. pandas.to_markdown 은 tabulate 를 요구하므로 직접 만든다."""
    cols = [str(c) for c in frame.columns]
    head = f"| {index_name} | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    rows = [
        f"| {idx} | " + " | ".join(str(v) for v in frame.loc[idx]) + " |"
        for idx in frame.index
    ]
    return "\n".join([head, sep, *rows])


def class_source_report(df: pd.DataFrame) -> str:
    """클래스 × 출처 교차표 (PLAN §3.2 확정용)."""
    lines = ["# 01 · 클래스 × 출처 교차표", "",
             "`metadata.csv` v0 기준. crop 변환(02)·품질(03)·split(04) 적용 전이다.", ""]

    def table(sub: pd.DataFrame, title: str) -> None:
        # `lines +=` 는 이름을 재바인딩해 지역변수로 만든다. extend 를 쓴다.
        lines.extend([f"## {title}", ""])
        if sub.empty:
            lines.extend(["(없음)", ""])
            return
        ct = pd.crosstab(sub["unified_label"], sub["dataset"], margins=True, margins_name="합계")
        lines.extend([_md_table(ct, "unified_label"), ""])

    table(df[df["head_a6_eligible"]],
          "공식 baseline 후보 (`head_a6_eligible`) — primary 만")
    table(df[df["eval_candidate_head_a"] & df["head_a6_eligible"]],
          "평가 후보 (`eval_candidate_head_a`) — split 미적용")
    table(df[df["head_a6_aux_pool"]],
          "ablation pool (`head_a6_aux_pool`) — baseline 에 넣지 않는다")

    lines.extend([
        "> `head_a6_eligible` 은 `sample_role=='primary'` 만 포함한다.",
        "> `train_eligible_head_a` 로 정의하면 auxiliary 3,169장이 baseline 에 새어 들어간다.",
        "",
    ])

    lines += ["## 계보 배분 (D-14)", ""]
    m = df[df["dataset"].eq("roboflow_wall_defects") & df["unified_label"].eq("moisture_leak")]
    lines += [
        f"- RWD moisture 총 **{len(m)}**",
        f"- 봉인 `sealed_future_eval` **{int(df['sealed_future_eval'].sum())}**행 "
        f"/ {df.loc[df.sealed_future_eval, 'split_component_id'].nunique()} component",
        f"- 탐색 aux `head_a7_aux_train` **{int(df['head_a7_aux_train'].sum())}**",
        f"- 격리 `quarantined` **{int(m['quarantined'].sum())}**",
        "",
        "봉인은 `split_component_id` 전체에 적용된다. moisture 행만 봉인하면 같은 component의",
        "`crack` 행이 aux 학습에 들어가 미래 holdout이 오염된다 (D-14).",
        "",
        "## 제외 사유", "",
        _md_table(df.loc[df["rejection_reason"].ne(""), "rejection_reason"]
                    .value_counts().to_frame("행 수"), "rejection_reason"),
        "",
    ]
    return "\n".join(lines)


def save_report(df: pd.DataFrame) -> Any:
    out = load_paths().artifact("reports") / "01_class_source.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(class_source_report(df), encoding="utf-8")
    return out
