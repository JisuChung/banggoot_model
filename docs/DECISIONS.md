# 결정 기록 (DECISIONS)

이 문서는 되돌리려면 데이터셋을 재빌드해야 하는 결정만 기록한다.
각 항목은 **결정 / 근거 / 되돌리는 비용 / 상태**를 갖는다.
새 결정은 append only. 기존 항목은 상태만 바꾸고 내용을 고쳐 쓰지 않는다.

---

## D-01. 학습 샘플 단위 = crop

**결정일** 2026-08-02 · **상태** 확정

Head A(L1 7-class)의 학습·평가 샘플 단위는 **하자 영역 crop**이다. 원본 사진 1장이
아니다.

| 데이터셋 | crop 생성 방식 |
| --- | --- |
| `roboflow_*` | bbox를 10~15% 확장 후 이미지 경계 clipping |
| `aihub_567` | polygon → bbox 변환 후 동일 규칙 |
| `dacon_wallpaper` | 이미 하자 근접 사진이므로 원본 그대로 사용 |
| `kaggle_cracks` | 이미 227×227 표면 패치이므로 원본 그대로 사용 |

**근거**

- 기존 마스터 계획 `[modeling] DEFECT_CLASSIFICATION_MODEL_PLAN.md` L633-645가 Head A의
  학습 데이터를 "YOLO bbox crop / AIHub polygon·bbox crop"으로 이미 정의한다. 신규 모델을
  Head A로 위치시키면 구조가 그대로 재사용된다.
- 전체 이미지 단위를 택하면 정책 B(AIHub full-image 미사용)가 배제한 이미지를 다시
  꺼내야 하고, 그 결과 `crack` 학습 데이터가 612 → 0이 되는 문제가 재발한다.
- DACON·Kaggle은 이미 근접/패치 사진이므로 "전체 이미지"와 "crop"이 실질적으로 같다.
  즉 이 결정으로 실제로 달라지는 것은 `roboflow_*`와 `aihub_567` 뿐이다.

**따르는 제약**

- 서비스 추론 시 crop을 만들 수단이 필요하다. detector가 없으면 중앙 crop 또는 2×2 tile
  fallback을 쓰고, fallback 신뢰도가 낮으면 `하자 없음`으로 확정하지 않고 UNKNOWN 처리한다.
- 너무 작은 bbox는 최소 crop 크기를 보장한다.
- 동일 영역 중복 bbox는 NMS로 정리한다.
- **crop은 원본 이미지의 `duplicate_group_id`를 상속한다.** crop 단위로 그룹을 새로
  만들면 같은 원본에서 나온 crop이 split을 넘나든다.

**되돌리는 비용** 데이터셋 전면 재빌드 + baseline 3종 재학습 (약 1일)

---

## D-02. 분할 비율 = 70 / 15 / 15

**결정일** 2026-08-02 · **상태** 확정

**근거**

- 사용자 요구사항이 명시적으로 70/15/15다.
- 중복 제거 후 `lifting`은 100장 내외로 추정된다. 10% test는 표본 10장 수준이라
  클래스별 F1이 노이즈가 된다. 15%가 그나마 낫다.

**기존 계획과의 차이**

마스터 계획 6.3절의 공식 비율은 **80/10/10**이다. 이 결정은 의도적 divergence다.

- 따라서 **기존 A0(`dacon19_b0_224_a0`) / V1(`dacon19_detail_v1`)의 수치와 직접 비교하지
  않는다.** 두 계보는 태스크(L3 19-class vs L1 7-class)도 분할도 다르다.
- 비교가 필요하면 동일 분할 위에서 재학습한 뒤 비교한다.

**되돌리는 비용** split 재생성 + 전 모델 재학습 (약 반나절)

---

## D-03. `kaggle_cracks` 상한 샘플링 (~1,500)

**결정일** 2026-08-02 · **상태** 확정

`kaggle_cracks` Positive 20,000장 중 **약 1,500장만 학습에 사용**한다. Negative 20,000장은
v1 범위(하자 유형 분류) 밖이므로 사용하지 않되, 향후 OOD/hard negative 후보로 보존한다.

**근거**

- 전량 투입 시 `crack`이 전체의 약 79%를 차지하여 Accuracy 지표가 무의미해지고, 모델이
  "회색 콘크리트 텍스처 = crack"을 학습할 위험이 크다. 227×227 콘크리트 표면 패치는
  주거 실내 사진과 도메인이 다르다.
- 완전 제외하면 `crack`이 약 950장(AIHub 763 + roboflow 187)으로 줄고, 교차 출처 검증이
  가능한 클래스가 `breakage` 하나만 남아 편향 측정 능력이 크게 떨어진다(A-1 참조).

**샘플링 규칙**

- 고정 seed, `duplicate_group_id` 단위로 추출한다. 개별 이미지 단위로 뽑지 않는다.
- 선택되지 않은 이미지는 삭제하지 않고 `sample_role=excluded`,
  `rejection_reason=kaggle_subsample_not_selected`로 표시해 추적 가능하게 둔다.
  (`split`은 노트북 04에서 만들어지므로 노트북 01에서 쓰지 않는다 — D-12)

**후속 실험** ablation arm으로 `kaggle=0 / 1500 / 전량` 3개를 비교해 실제 영향을 측정한다.

**되돌리는 비용** split 재생성 + `crack` 관련 모델 재학습 (약 반나절)

---

## D-04. 신규 모델의 아키텍처상 위치 = Head A (L1 coarse)

**결정일** 2026-08-02 · **상태** 확정

이번 프로젝트가 만드는 7-class 분류기는 **Head A(L1 coarse confirmation)**이며,
**사용자에게 직접 노출하는 라벨이 아니다.**

**근거**

- 마스터 계획 L448-451: L1은 "직접 노출 안 함", L2가 "화면 표시", L3는 "신뢰도 충족 시만".
- 마스터 계획 L452: "L1은 YOLO가 학습하는 라벨이므로 AIHub 구조물·Kaggle 콘크리트까지
  포함하는 기술적 분류다."
- **L1 → L2는 함수가 아니다.** `breakage` 하나가 `벽지 손상` / `벽 구멍·타공` /
  `박리·철근 노출` 3개 L2에 걸친다(L484-500). 따라서 L1 분류기 단독으로는 사용자 표시
  라벨을 만들 수 없다.

**따르는 구조**

```
L1 ∈ {breakage, finish_damage, lifting, mold, stain_corrosion} 이고 벽지·마감 계열
    → 기존 Head B(V1, L3 19-class) 실행 → 19-class softmax 합산 → L2 (화면 표시)

L1 ∈ {crack, moisture_leak} 또는 AIHub 구조물 출처
    → L1 → L2 직접 (structural_crack / moisture_leak / structural_breakage)
```

**되돌리는 비용** 낮음 (후처리 계층 변경). 단 이 결정을 뒤집어 L1을 표시 라벨로 쓰면
서비스 라벨 체계 전체를 다시 정의해야 한다.

---

## D-05. Roboflow 원본 split 미사용

**결정일** 2026-08-02 · **상태** 확정 (기존 결정 승계)

`roboflow_*`가 제공하는 train/valid/test split을 **분할에 사용하지 않는다.** 그룹 키로도
사용하지 않는다. `source_split` 컬럼에 참고 메타데이터로만 보존한다.

**근거**

- 마스터 계획 L965: "**Roboflow 기존 split은 사용하지 않는다.** House Defect의 합성 이미지
  crop이 train/valid/test를 넘나드는 것이 확인됐다. 항상 global split을 새로 만든다."
- 마스터 계획 L889: `fakes*` 976장은 고유 원본 **30개**의 crop이며 원본 split에서
  train/valid/test에 819/78/79로 흩어져 있다. 즉 제공된 split 자체가 이미 누수 상태다.

**대신 사용할 그룹 키** `duplicate_group_id` + Roboflow `.rf.` 앞 원본 파일명 +
AIHub `Raw_Data_ID`

**되돌리는 비용** 해당 없음. 되돌리면 안 되는 결정이다.

---

## D-06. 기존 파생 자산 승계 범위

**결정일** 2026-08-02 · **상태** 확정

`../defect_classifying_model`은 **읽기 전용 증거 저장소**로 취급하고, 아래를 승계한다.

| 자산 | 처리 |
| --- | --- |
| `duplicate_group_id` (8,453 그룹 / 데이터셋 간 1,886) | 승계. 재계산하지 않음 |
| `source_origin` 판정 (`synthetic_verified` 등) | 승계 |
| `configs/origin_overrides.json` | 승계 |
| `configs/manual_confirmed_artifacts.json` | 승계 |
| review 판정 결과 (992건) | 승계. 재검수하지 않음 |
| 평가 제외 규칙 (`reserved_reason`) | 승계 |
| `configs/dataset_registry.json` class_map | 승계 |
| `split_manifest.csv`의 split **배정값** | 승계 안 함. D-02에 맞춰 재생성 |
| `data/derived/v1/classifier/*` (L3 계보) | 참조만. Head A는 별도 계보 |
| `data/derived/v1/experiments/`, `yolo_d0_s0/` | 승계 안 함 (정책 B 미결 상태) |

**근거**

- 마스터 계획 L1695: "모든 모델은 기존 `split_manifest.csv`와 `duplicate_group_id`를
  그대로 사용한다."
- 중복·출처 판정은 약 8,000줄 중 4,325줄(54%)을 들여 얻은 결과다. 재현하려면 그 비용을
  다시 치러야 한다.
- 특히 재현이 어려운 사실:
  - Roboflow 한국형 도배 743장 중 **731장(98.4%)이 중복**
  - House Defect 6,028장 중 **44.8%가 DACON 재사용, 16.2%가 합성**
  - 데이터셋 간 중복 그룹 **1,886개**, SHA-256 완전 일치 쌍 **1,790개**

**승계 절차** 승계 파일은 `artifacts/inherited/`에 SHA-256과 함께 스냅샷하고,
`artifacts/inherited/MANIFEST.json`에 원본 경로·해시·승계 일시를 기록한다.

**되돌리는 비용** 높음. 승계를 포기하면 중복·출처 감사를 처음부터 다시 해야 한다.

---

## D-07. 합성 이미지 처리

**결정일** 2026-08-02 · **상태** 확정 (기존 결정 승계)

`source_origin=synthetic_verified`인 이미지는

- **validation / test에서 제외**한다.
- baseline train에도 **포함하지 않는다.** `train_auxiliary`로 보존만 한다.

**근거** 마스터 계획 L903-904, L966. 실제 스마트폰 사진 성능을 대표하지 않으므로 평가에
포함하면 점수가 왜곡된다.

---

## D-08. v1 범위에서 제외하는 데이터

**결정일** 2026-08-02 · **상태** 확정

| 대상 | 수량 | 사유 |
| --- | ---: | --- |
| `mvtec_ad` 전체 | 673 | 라이선스 CC BY-NC-SA 4.0 = 상업적 사용 불가 |
| `dacon_wallpaper` 공개 test | 792 (연쇄 잠금 832) | 정답 없음. 아래 참조 |
| `kaggle_cracks` Negative | 20,000 | v1은 하자 유형 분류. 정상 판별은 범위 밖 |
| `aihub_567` `대지` 카테고리 | 222 | 기존 `excluded` 판정 승계 |
| `roboflow_house_defect` hard negative | 3,508 | 하자 클래스 라벨 없음. OOD 게이트용으로 보존 |
| 다중 라벨 crop | 11 | D-09 참조 |

### DACON 잠금 수량 — 792 / 801 / 832는 서로 다른 세 값이다 (U-3 해소)

세 숫자가 전부 실재하며 어느 것도 오기가 아니다. 혼용하지 않는다.

| 값 | 의미 | 확인 방법 |
| ---: | --- | --- |
| **792** | 실제 DACON 공개 test 이미지. 정답 없음 | `open/test/` 파일 792개 · `test.csv` 792행 · `use_status=inference_only_no_ground_truth` 792 |
| **801** | 잠긴 component에 속한 **DACON 행 전체** (792 + 같은 component의 학습 이미지 9) | `reserved_reason='contains_dacon_public_test'` ∩ `dataset='dacon_wallpaper'` |
| **832** | 잠긴 component에 속한 **전체 데이터셋 행** (DACON 801 + rhd 30 + rwkr 1) | `reserved_reason='contains_dacon_public_test'` 전체 · 787개 component |

마스터 계획이 "공개 test 801장"이라고 쓴 것은 **표현이 부정확**한 것이지 숫자가 틀린 게 아니다.
801은 "정답 없는 이미지 수"가 아니라 "잠긴 DACON 행 수"다.

### ★ `reserved_inference`는 존재하지만 **컬럼이 다르다**

두 컬럼에 서로 다른 값이 들어 있고, 완전히 1:1 대응한다.

| 컬럼 | 값 | 행 수 |
| --- | --- | ---: |
| `split` | **`reserved_inference`** | 832 |
| `reserved_reason` | **`contains_dacon_public_test`** | 832 |

교차 확인 결과 두 집합은 완전히 동일하다. 따라서

```python
df[df.split == 'reserved_inference']              # 832건 — 맞다
df[df.reserved_reason == 'reserved_inference']    # 0건 — 조용히 통과한다 (위험)
```

**`reserved_reason == 'reserved_inference'` 필터는 무효**이고 오류 없이 0건을 반환한다.
가장 위험한 종류의 버그다.

실측 `reserved_reason` 분포 (54,797행):

| 값 | 행 수 | Head A 처리 |
| --- | ---: | --- |
| (공백) | 48,198 | 정상 |
| `no_eval_eligible_origin_in_component` | 3,814 | 평가 불가 |
| `aihub_category_needs_review` | 945 | 제외 |
| `contains_dacon_public_test` | 832 | **전면 제외** |
| `mvtec_official_protocol` | 673 | **전면 제외** (D-08 라이선스와 일치) |
| `out_of_scope_category` | 220 | 제외 |
| `rare_detail_class_train_only` | 115 | 평가 불가, 학습만 |

**assertion은 세 층으로 분리한다.**

```python
assert inventory.use_status.eq('inference_only_no_ground_truth').sum() == 792
assert (reserved_reason == 'contains_dacon_public_test').sum() == 832
assert (legacy_split == 'reserved_inference').sum() == 832
assert set(잠긴 행 by reserved_reason) == set(잠긴 행 by legacy_split)   # 1:1 확인
assert dacon 중 잠긴 행 == 801
assert 잠긴 component 수 == 787
assert 'reserved_inference' not in reserved_reason.unique()   # 컬럼 혼동 방지
```

---

## D-09. 다중 라벨 이미지 처리

**결정일** 2026-08-02 · **상태** 확정

`roboflow_wall_defects`의 다중 라벨 11장(`breakage;crack` 4, `breakage;stain_corrosion` 3,
`mold;stain_corrosion` 3, `breakage;crack;stain_corrosion` 1)은

- **crop 단위(D-01)로 분해하면 대부분 단일 라벨이 된다.** bbox마다 클래스가 하나이므로
  crop 하나에 라벨 하나가 붙는다.
- 분해 후에도 동일 crop에 2개 이상 클래스 bbox가 IoU 0.5 이상으로 겹치면
  `quality_status=review`, `rejection_reason=multilabel_overlap`으로 표시하고 학습에서
  제외한다. 삭제하지 않는다.

D-01이 다중 라벨 문제를 상당 부분 자동 해소한다는 점이 crop 단위의 부수 이점이다.

---

## D-10. 백엔드 enum 변경 시점

**결정일** 2026-08-02 · **상태** 확정

`DamageType`에 `MOLD`, `FINISH_DAMAGE`를 추가하는 작업은 **최종 모델이 확정된 뒤 별도
PR**로 진행한다. 이번 프로젝트에서는 `configs/damagetype_mapping.md`에 매핑 규칙만
확정하고 Java 코드는 건드리지 않는다.

**근거** `_bmad-output/implementation-artifacts/ai-service-integration-decisions.md` D1에서
추가가 결정됐으나 구현은 미완이다. 모델 클래스가 확정되기 전에 enum을 바꾸면 두 번 바꾸게
된다.

**미해결 매핑** `stain_corrosion`이 `STAIN`인지 `DISCOLOR`인지. `configs/damagetype_mapping.md`
작성 시 확정한다.

---

---

## D-11. crop 입력 계약 — 이 모델은 완제품이 아니라 구성요소다

**결정일** 2026-08-02 · **상태** 확정

D-01이 학습 단위를 crop으로 정한 결과, 이 모델의 입력 계약은 다음과 같다.

> **입력: 하자 영역이 이미 잘려 있는 crop. 출력: L1 7-class 중 하나.**

즉 **"사용자 사진 → 하자 유형" 완제품이 아니다.** 서비스 승격 조건을 명시한다.

| 경로 | 상태 | 비고 |
| --- | --- | --- |
| ① 사용자가 하자를 화면 중앙에 근접 촬영 | **v1 채택** | 촬영 가이드 UI로 강제. DACON 학습 분포와 가장 가까움 |
| ② detector(D0) 완성 후 bbox crop | 후속 | detector 미구현. `crack` detection은 현재 데이터로 불가 |
| ③ 중앙 crop / 2×2 tile fallback | **별도 실험** | tight bbox crop과 분포가 달라 train-serving skew 발생 |

**v1은 ①을 채택한다.** 근거:

- 기존 A0/V1도 같은 제약을 이미 갖고 있다. `MODEL_VERSION_PERFORMANCE_REPORT.md` 2절:
  "A0는 bbox를 생성하지 않으므로 전체 방 사진이 아니라 벽지 하자 근접 사진을 전제로 한다."
- ②는 detector에 의존하고 detector는 막혀 있다(§10).
- ③은 학습 분포와 서빙 분포가 다르므로, **채택하려면 별도 실험 arm으로 학습·평가해야
  한다.** 학습은 tight crop으로 하고 서빙만 center crop으로 하는 조합은 금지한다.

**따르는 제약**

- 리포트·API 응답에 "전체 방 사진은 지원하지 않음"을 명시한다.
- 노트북 11(OOD 게이트)에 **"하자가 화면에서 너무 작음"** 을 재촬영 사유로 포함한다.
- 성능 수치를 "사용자 사진에서의 하자 분류 정확도"로 표현하지 않는다.

**되돌리는 비용** ③으로 바꾸려면 center-crop/tile 분포로 재학습 + 재평가.

---

## D-12. eligibility 승계 — origin은 2층 구조다

**결정일** 2026-08-02 · **상태** 확정

### 문제

`split_manifest.csv`의 `source_origin`은 **오버라이드가 반영되지 않은 baseline**이다.
`build_global_splits.py`(07-30 08:58)는 `origin_overrides.json`(07-30 09:27)을 읽지 않는다.
실제 유효값은 `origin_decisions.csv`의 `effective_origin`과 `*_effective` 컬럼이다.

실측 차이:

| | `split_manifest.source_origin` | `origin_decisions.effective_origin` |
| --- | --- | --- |
| `roboflow_wall_defects` | `real_verified` 376 | `unknown_origin` 375 + `synthetic_verified` 1 |

`origin_overrides.json` v1.1의 강등 근거: canonical eval 블라인드 검수 73장 중 **7장(9.6%)**
이 stock 워터마크·캡션·타임스탬프·합성 일러스트로 확인되어 "교차 오염이 없으므로 실사"라는
전제가 반증됨.

### 결정

**`origin_decisions.csv`를 origin의 단일 권위 소스로 삼는다.** `split_manifest.csv`는
split 배정 구조(`split_component_id`, `duplicate_group_id`, `source_root_id`)만 승계한다.

### metadata 스키마에 반드시 포함할 eligibility 필드

```text
split_component_id          split 배정 단위 (중복그룹 ∪ AIHub Raw_Data_ID ∪ source root)
duplicate_group_id          동일 이미지 판정 단위 (평가 시 1장으로 축소)
source_root_id              합성·증강 원본 식별자
effective_origin            ★ origin_decisions.csv 기준
quarantined                 ★ 7건
sample_role                 primary / auxiliary / excluded
train_auxiliary             baseline train 제외, ablation 전용
train_eligible_head_a
eval_eligible_head_a
canonical_eval_head_a
reserved_reason
```

기존 manifest는 `*_detection` / `*_detail` 두 태스크만 갖는다. Head A는 세 번째 태스크이므로
**`*_head_a`를 신규 파생**한다.

### ★ eligibility를 origin만으로 만들면 안 된다

origin 조건만 쓰면 **MVTec 673장이 Head A에 재유입된다.** MVTec은 전량
`effective_origin='real_verified'`이고, `auxiliary_class_map`을 적용하면 **정확히 133장**이
L1으로 매핑된다 (`finish_damage` 54 · `stain_corrosion` 42 · `crack` 17 · `breakage` 10 ·
`moisture_leak` 10). D-08에서 라이선스로 전량 제외하기로 한 데이터가 다시 들어온다.

**정책 필터를 먼저 적용한다.**

```text
# 1. 정책 제외 — origin 보다 먼저 본다
excluded_by_policy =
       dataset == 'mvtec_ad'                              # D-08 라이선스
    OR reserved_reason in {                               # 승계한 잠금 사유
           'contains_dacon_public_test',                  #   832
           'mvtec_official_protocol',                     #   673
           'aihub_category_needs_review',                 #   945
           'out_of_scope_category',                       #   220
       }
    OR use_status in {                                    # inventory 컬럼
           'excluded',
           'inference_only_no_ground_truth',
           'anomaly_train', 'anomaly_test_locked',
       }
    OR unified_label not in L1_7                          # normal / null 포함

# 2. 학습 권한
train_eligible_head_a =
        NOT excluded_by_policy
    AND NOT quarantined                                   # 7건
    AND effective_origin != 'synthetic_verified'          # D-07

# 3. auxiliary — 학습 가능하지만 baseline 제외, ablation 전용
train_auxiliary =
        train_eligible_head_a
    AND effective_origin in {'unknown_origin',
                             'dacon_derivative_verified',
                             'dacon_derivative_probable'}

# 4. sample_role
sample_role = 'excluded'  if not train_eligible_head_a
              'auxiliary' if train_auxiliary
              'primary'   otherwise

# 5. 평가 "후보" — split 을 참조하지 않는다
eval_candidate_head_a =
        sample_role == 'primary'
    AND effective_origin == 'real_verified'
    AND reserved_reason != 'no_eval_eligible_origin_in_component'   # 3,814
    AND reserved_reason != 'rare_detail_class_train_only'           # 115

# 6. 대표본 후보 순위 — 결정론적 tie-break 만. 실제 선택은 노트북 04
representative_rank =
    duplicate_group_id 내에서 sha256(record_id + SEED) 오름차순 순위
```

### ★ 언제 계산하는가 — `eval_eligible`은 노트북 01에서 만들 수 없다

`eval_eligible_head_a`가 `split in {val, test}`를 요구하면 **순환 의존**이 생긴다.
새 70/15/15 split은 노트북 04에서 만들어지고, 승계한 `split_manifest.csv`의 `split` 컬럼은
**폐기하기로 한 기존 80/10/10 배정**(train 42,488 / val 4,819 / test 4,820)이다.
노트북 01에서 계산하면 버리기로 한 split을 참조하게 된다.

`canonical_eval_head_a`도 마찬가지다. 대표본은 "품질을 통과한 crop 중 하나"여야 하는데,
crop은 노트북 02에서 생기고 품질 판정은 노트북 03에서 나온다. 흐린 이미지가 대표본으로
뽑히면 그 클래스의 지표 전체가 왜곡된다.

**따라서 파생을 3단계로 나눈다.**

| 노트북 | 생성하는 컬럼 | 근거 |
| --- | --- | --- |
| **01** | `excluded_by_policy` · `train_eligible_head_a` · `train_auxiliary` · `sample_role` · `eval_candidate_head_a` · `representative_rank` | split·crop·품질에 의존하지 않는 것만 |
| **02** | crop 단위 확장 · `bbox_*` · `width`/`height` | crop 생성 |
| **03** | `quality_status` · `rejection_reason` · 품질 지표 | 이미지 열람 |
| **04** | `split` · **`eval_eligible_head_a`** · **`canonical_eval_head_a`** | 새 배정 후 확정 |

노트북 04에서의 최종 확정:

```text
eval_eligible_head_a =
        eval_candidate_head_a
    AND quality_status == 'keep'          # 03 결과
    AND split in {'val', 'test'}          # 04 에서 새로 배정한 값

canonical_eval_head_a =
        eval_eligible_head_a
    AND representative_rank 가 해당 duplicate_group 안에서
        eval_eligible 인 것들 중 최소            # 품질 통과분 중에서 고른다
```

**노트북 01은 승계한 `split` 컬럼을 `legacy_split`으로 이름을 바꿔 보존만 한다.**
어떤 파생에도 쓰지 않는다. 이름을 바꾸는 이유는 실수로 쓰는 것을 막기 위해서다.

**검증 assertion**

```python
assert (metadata.dataset == 'mvtec_ad').sum() == 673
assert metadata.loc[metadata.dataset == 'mvtec_ad', 'sample_role'].eq('excluded').all()
assert metadata.loc[metadata.train_eligible_head_a, 'dataset'].ne('mvtec_ad').all()
```

`unknown_origin` 정책(마스터 계획 L1080): `split=train`, `train_auxiliary=true`,
`eval_eligible_*=false`. 승격은 **시각 검수만으로 불가**하고 `source_url` +
`license_evidence`가 필요하다(`origin_overrides.json` `allowlist_policy`). `row_allowlist`는
현재 비어 있다.

`unknown_origin` 정책(마스터 계획 L1080): `split=train`, `train_auxiliary=true`,
`eval_eligible_*=false`. 승격은 **시각 검수만으로 불가**하고 `source_url` +
`license_evidence`가 필요하다(`origin_overrides.json` `allowlist_policy`). `row_allowlist`는
현재 비어 있다.

**되돌리는 비용** 높음. 잘못 승계하면 평가 지표 전체가 무효가 된다.

---

## D-13. 평가 대표본 규칙 — 중복 그룹당 1장

**결정일** 2026-08-02 · **상태** 확정 (기존 규칙 승계)

split을 나누는 것만으로는 평가 왜곡이 해결되지 않는다. 같은 중복 그룹의 이미지 여러 장이
val/test에 함께 남으면 동일 원본이 점수에 반복 반영된다.

**규칙 (마스터 계획 L968-985)**

- val/test는 **`duplicate_group_id`당 대표 이미지 1장**만 지표 계산에 사용한다.
- 나머지는 `eval_eligible_head_a=false`로 표시하고 보관한다. **삭제하지 않는다.**
- **축소 단위는 `duplicate_group_id`이며 `split_component_id`가 아니다.**
  AIHub 2,728장은 `Raw_Data_ID` 1,216그룹으로 묶이고 그중 567그룹이 2장 이상(최대 24장)인데,
  이들은 같은 촬영 묶음의 **서로 다른 시점·각도** 사진이지 중복이 아니다. 컴포넌트당 1장으로
  줄이면 유효한 평가 데이터가 대량 소실된다.
- AIHub `Raw_Data_ID`는 같은 split에 유지하되 대표 1장으로 축소하지 않는다.
- 대표본은 **task별로 따로** 고른다. Head A 대표본은 `*_head_a` 축으로 선택하며
  기존 `*_detection` / `*_detail` 대표본을 재사용하지 않는다.

**crop 단위(D-01)에서의 적용**

1. 중복 그룹당 **대표 원본 이미지 1개**를 고른다.
2. 그 원본에서 나온 **유효 crop은 전부 유지**한다. (crop을 1개로 줄이지 않는다 — 한 사진에
   여러 하자가 있으면 각각이 독립 평가 표본이다)
3. 나머지 원본에서 나온 crop은 `eval_eligible_head_a=false`.

**bootstrap도 이에 맞춘다.** crop 단순 재표집이 아니라 **원본/중복 그룹 단위 clustered
bootstrap**을 쓴다. 같은 원본에서 나온 crop은 독립 표본이 아니다.

**train에서도** 그룹별 sampling 비중을 제한해 offline augmentation 사본이 batch를 지배하지
않게 한다.

---

## D-14. `moisture_leak` — 계보를 둘로 나눈다 (구 U-4)

**결정일** 2026-08-02 · **상태** 확정

### 문제

D-12를 적용하면 `moisture_leak`의 평가 가능 이미지는 **0장**이다. 150장 전량이
`roboflow_wall_defects`이고 `unknown_origin` 147 + `quarantined` 3이다.

### 결정

| 계보 | 클래스 | RWD moisture 147장 | 역할 |
| --- | ---: | --- | --- |
| **`head_a6_primary`** | **6** | **사용 안 함 (미개봉 보존)** | 공식 baseline |
| `head_a7_rwd_aux` | 7 | `train_auxiliary` | 탐색 실험 전용 |

**공식 지표·모델 비교·서비스 승격은 `head_a6_primary`만 사용한다.**

### 검토했으나 기각한 안

**(기각) 7-class 유지 + moisture를 primary 학습에 예외 포함**

처음에 권장했던 안이며 다음 두 가지 이유로 철회한다.

1. **되돌릴 수 없다.** "U-2가 풀리면 평가만 켜면 된다"는 틀렸다. 147장을 전부 학습에 쓰면
   출처가 확인돼도 **같은 사진으로 평가할 수 없다.** holdout을 다시 나눈 뒤 재학습하거나
   완전히 새로운 독립 평가 데이터를 확보해야 한다. 미개봉으로 두면 이 비용이 0이다.
2. **A0의 train-only 관례와 상황이 다르다.**

   | | A0의 train-only 7클래스 | `moisture_leak` |
   | --- | --- | --- |
   | 출처 | DACON, 나머지 12클래스와 동일 | 전량 출처 미확인 |
   | 라벨·라이선스 계보 | 검증된 동일 계보 | 미확인 |
   | 평가 표본 | 0 (분할 결과) | 0 (**출처 부적격**) |

   A0는 "같은 신뢰도의 데이터인데 표본이 적어서" 평가를 못 한 것이고, `moisture_leak`은
   "데이터 자체를 믿을 수 없어서" 평가를 못 하는 것이다. 같은 처리를 적용할 수 없다.

3. D-12의 `unknown_origin` → `train_auxiliary` 정책을 클래스 하나 때문에 예외 처리하는 것은
   정책을 무의미하게 만든다.

**(기각) `mold`에 병합해 `mold_moisture`**

L2 서비스 그룹과 이름은 맞지만, **곰팡이 평가 결과로 누수까지 검증된 것처럼 보이게 만든다.**
`mold`는 DACON 145장으로 평가 가능하고 `moisture_leak`은 0장이다. 병합하면 이 비대칭이
단일 F1 뒤로 숨는다. 문제를 해결하지 않고 가린다.

**(보류) RWD 출처 조사로 승격** — U-2. `source_url` + `license_evidence`가 필요하며
시각 검수만으로는 불가능하다.

### ★ `split` 단일 컬럼으로는 표현할 수 없다

"공식 모델에서는 제외하지만 aux 실험에서는 학습"은 `split` 하나로 못 쓴다.
D-12의 `unknown_origin → split=train`과도 충돌한다. **계보별 불린 컬럼을 따로 둔다.**

| 컬럼 | 정의 | 실측 |
| --- | --- | ---: |
| `head_a6_eligible` | **`sample_role=='primary'`** ∧ ¬moisture ∧ ¬sealed | 6,509 |
| `head_a6_aux_pool` | `sample_role=='auxiliary'` ∧ ¬moisture ∧ ¬sealed. **ablation 전용** | 3,169 |
| `head_a7_aux_train` | 탐색 7-class 계보의 moisture aux | 103 |
| `sealed_future_eval` | **영구 미개봉.** 어떤 계보의 학습에도 쓰지 않는다 | 45 |

**★ `head_a6_eligible`을 `train_eligible_head_a`로 정의하면 안 된다.**

`train_eligible_head_a`는 auxiliary도 `True`다. 그걸로 정의한 1차 구현에서 auxiliary
**3,169장**(RHD 2,217 · RWKR 741 · RWD 211)이 공식 baseline에 새어 들어갔고,
"auxiliary는 baseline 제외"(D-12) 정책이 조용히 깨졌다. 후보가 9,678로 부풀어 있었다.

```python
assert df.loc[df.head_a6_eligible, 'sample_role'].eq('primary').all()
assert not df.loc[df.head_a6_eligible, 'train_auxiliary'].any()
assert not (df.head_a6_eligible & df.head_a6_aux_pool).any()
```

`head_a6_aux_pool`을 별도 컬럼으로 두는 이유는 ablation arm마다 세 조건을 다시 쓰면
같은 실수가 반복되기 때문이다.

**결과적으로 공식 baseline의 출처는 DACON · AIHub · Kaggle 셋뿐이다.** roboflow 3종은
전부 auxiliary라 baseline에 들어가지 않는다. §PLAN 3.2의 출처 편향 분석과 일치한다.

RWD moisture 150장의 배정:

```text
head_a6_eligible   = false      (150 전부)
quarantined        = true       (3)   -> 어디에도 사용 안 함
head_a7_aux_train  = true       (~103)
sealed_future_eval = true       (~44)
```

### RWD 147장을 전부 aux에 쓰지 않는 이유

D-14가 (c)안을 기각한 논리는 **aux 계보에도 그대로 적용된다.** 147장을 전부
`head_a7_rwd_aux` 학습에 쓰면 U-2가 풀려도 그 사진들로 moisture 평가 holdout을 만들 수 없다.

따라서 **`split_component_id` 기준으로 약 30%를 봉인**한다.

### ★ 봉인은 행이 아니라 component 전체에 적용한다

moisture 행에만 `sealed_future_eval=true`를 주면 누수가 생긴다. 실측:

| | 값 |
| --- | ---: |
| RWD moisture 행 | 150 |
| 그 행들이 속한 component | 149 |
| **그 149개 component의 전체 행** | **154** |
| 차이 (moisture 아닌 행) | **4 — 전부 `crack`** |
| 영향받는 component | 2 (`comp_040956` crack×3 · `comp_040979` crack×1) |

봉인한 moisture 사진과 같은 component에 묶인 crack 4장이 aux 학습에 들어가면, 나중에
봉인분으로 평가할 때 이미 학습에 노출된 원본이 섞인다.

```text
sealed_components  = 봉인 대상으로 선택한 split_component_id 집합
sealed_future_eval = split_component_id in sealed_components    # 모든 행에 적용

어떤 학습 계보도 sealed_future_eval 행을 사용하지 않는다.
```

**필수 assertion**

```python
# 1. component 안에서 봉인 여부가 갈리지 않는다
assert not (metadata.groupby('split_component_id')['sealed_future_eval']
            .nunique().gt(1).any())

# 2. 봉인 행은 어떤 계보에도 들어가지 않는다
assert not metadata.loc[metadata.sealed_future_eval,
                        ['head_a6_eligible', 'head_a7_aux_train']].any().any()
```

**운영 규칙**

- 봉인 선택은 고정 seed의 결정론적 해시로 하고 `metadata.csv`에 기록한다
- 봉인분은 **노트북 12(최종 평가) 이후에도 열지 않는다.** U-2가 해소된 뒤 별도 결정으로만 연다
- 봉인 비율을 바꾸거나 봉인을 해제하는 것은 새 결정(D-15 이상)으로 기록한다

### 따르는 제약

- baseline 스윕(노트북 05~07)은 **6-class로만** 돈다. 3 모델 × 3 seed = 9 run.
- `head_a7_rwd_aux`는 baseline 최고 모델 1개에 대한 **ablation arm**이지 별도 스윕이 아니다.
- RWD moisture 150장은 삭제하지 않는다. 전부 `metadata.csv`에 남기고 위 3개 플래그로 구분한다.
- 리포트에서 "7-class 모델"이라는 표현을 쓰지 않는다.
- `head_a7` 결과에 "moisture 평가 복구 가능성은 봉인분 ~44장에만 남아 있다"를 명시한다.

**되돌리는 비용** 낮음. `head_a7`은 추가 arm이므로 baseline을 다시 돌리지 않는다.

---

## D-15. crop geometry — 기존 crop은 승계하지 않고 새로 생성한다 (D-01 supersede)

**결정일** 2026-08-03 · **상태** 확정 · **대상** D-01의 "bbox를 10~15% 확장" 부분만 대체

D-01의 **샘플 단위 = crop** 결정은 유지한다. geometry 규칙만 바꾼다.
D-01의 10~15%는 근거 없이 정한 값이었고, 기존 저장소에 QC로 실측 검증된 조합이 있다.

### 기존 crop(`aihub_positive_crops`)을 승계하지 않는 이유

| # | 사유 | 확인 |
| --- | --- | --- |
| 1 | `legacy_split=train`만 crop | `_AIHUB_CROPS_OK.json`: `source_split_filter: "train only"` · manifest 2,783건 전부 `source_split=train`. AIHub val/test 314장이 영구 누락된다 |
| 2 | `severity=good` 제외 | D0 보고서: "**`severity=good`은 결함 없음이 아니라 주석된 결함의 심각도 등급**이다. good 60장이 전부 자기 라벨에 bbox를 갖고 있었다". manifest severity는 `poor 2,232 / normal 551`, good 0 |
| 3 | QC를 `quality_status`로 승계 불가 | 100장 표본 · `reviewer_kind=agent_visual_review` · **`human_verified=false`**. 2,783장 전체의 품질 판정이 아니다 |
| 4 | 모집단이 폐기 정책 기준 | 2,783 crops / 1,062 parents 는 legacy train 기준 산출물이라 70/15/15(D-02)와 맞지 않는다 |

### 채택하는 geometry — `crop-context-2.0`

```yaml
context_ratio:                0.35    # side = max(bw,bh) * (1 + 2*ratio)
min_crop_px:                  128
min_retained_area:            0.50    # 이웃 box 잔존 면적 하한
dedupe_iou:                   0.70
max_crop_fraction:            1.00    # ★ 아래 참조
square_target:                true
legacy_rounding_tolerance_px: 1       # 회귀검증 시 좌표 ±1px 허용
```

**`max_crop_fraction=1.00`인 이유 — 문서가 아니라 산출물과 QC가 권위다**

`D0_DETECTOR_REPORT.md`는 crop-context-2.0에서 "max_crop_fraction 신설 0.55"라고 쓰지만,
실제로는 적용되지 않았다.

- `_AIHUB_CROPS_OK.json`의 `parameters` 블록에 이 키가 **없다**
- `build_aihub_positive_crops.py` argparse **default = 1.0**
- 실측 `crop_width / min(parent_w, parent_h)`: 중앙 0.296 · p90 0.681 · **최대 1.000**
- **0.55 초과 crop 411/2,783 (14.8%)**. 코드가 `side = min(side, short_edge * max_crop_fraction)`
  이므로 0.55였다면 최대가 0.55여야 한다
- QC 100장 표본에도 0.55 초과가 12장 포함됐고 그중 실패는 1장뿐이다
- 스크립트 SHA-256이 marker와 일치하므로 코드 변경 탓이 아니다

따라서 **QC 8/100 실패율은 `max_crop_fraction=1.00`으로 얻은 수치**다.
`0.55`는 재QC가 필요한 **ablation 후보**이며 baseline에 넣지 않는다.

**`square_target`과 1px 오차 — 경계 clipping이 아니다**

기존 crop 2,783장 중 685장이 비정사각인데, **685장 전부 가로·세로 차이가 정확히 1px**이고
그중 **471장(69%)은 이미지 내부**다(경계 접촉은 214장뿐). 즉 clipping이 아니라
`left` · `top` · `left+side` · `top+side`를 각각 반올림해서 생긴 오차다.

새 구현은 **정수 `side`를 먼저 확정하고 `x1 = x0 + side`, `y1 = y0 + side`로 계산**해
완전한 정사각을 보장한다. 기존 2,783 crop과 회귀검증할 때는 좌표 ±1px를 허용한다.

### 노트북 02 필수 규칙

1. **`legacy_split`을 참조하지 않는다.** 공식 AIHub 1,561장 전량과 그 polygon 전부를 변환한다
2. **`severity`는 `good`/`normal`/`poor` 전부 포함**하고 메타데이터로만 보존한다.
   `good`을 제외하거나 hard negative로 쓰지 않는다
3. **`legacy_annotation_error` 15건은 학습·평가 모두 차단** (아래 참조)
4. **`sealed_future_eval` 45장은 이미지 I/O와 crop 생성 자체를 금지**한다 (D-14)
5. **NMS/dedupe는 같은 L1끼리만** 적용한다. 서로 다른 L1이 겹치면 D-09에 따라 review/exclude
6. crop은 parent의 `record_id` · `duplicate_group_id` · `split_component_id` ·
   `source_root_id`를 **전부 상속**한다
7. 기존 2,783 crop은 **회귀검증 기준으로만** 사용한다 (좌표 ±1px 허용)
8. 기존 QC 결과는 `legacy_qc_verdict`로만 보존하고 `quality_status`로 자동 승격하지 않는다

### `legacy_annotation_error` 15건

`review/aihub_label_audit/audit_manifest.csv`의 `annotation_error=true` 15건은
**현재 metadata에서 전부 `sample_role=primary`이고 `head_a6_eligible=True`,
`eval_candidate_head_a=True`다.** 즉 기존 B가 제외했던 라벨 오류가 학습과 **평가 양쪽에**
재유입돼 있다. 평가 유입이 더 위험하다 — 라벨이 틀린 이미지로 지표를 계산하게 된다.

```text
legacy_annotation_error = true          # 15건 (crack 10 · breakage 5)
    -> train_eligible_head_a = false
    -> eval_candidate_head_a = false
    -> quality_status = 'review'
    -> review 전용 crop 만 생성 가능
```

검수자가 `agent_visual_review` / `human_verified=false`이므로 **"확정 오류"가 아니라
"사람 검토 필요"로 다룬다.** 삭제하지 않는다 (자동 삭제 금지 원칙).

`audit_manifest.csv`와 `_AUDIT_RESULT_OK.json`을 승계 대상에 추가한다.

**되돌리는 비용** crop 전면 재생성 + baseline 재학습.

---

## 미결 사항

| # | 항목 | 필요 시점 |
| --- | --- | --- |
| U-1 | `stain_corrosion` → `STAIN` vs `DISCOLOR` | 배포 준비 단계 |
| U-2 | AIHub 라이선스 + **`roboflow_wall_defects` 출처 조사** (`moisture_leak` 평가 복구의 유일한 경로) | 배포 준비 단계 |
| ~~U-3~~ | ~~DACON 792 vs 801~~ | **해소됨 — 792/801/832 각각 다른 값 (D-08)** |
| ~~U-4~~ | ~~`moisture_leak` 처리~~ | **해소됨 — 두 계보 분리 (D-14)** |
| U-5 | OOD challenge set 구성 방법 | 노트북 11 이전 |
