# 주거 하자 이미지 분류 모델 실행 계획

기준일 2026-08-02 · 대상 저장소 `banggoot_model`

이 문서는 **무엇을 어떤 순서로 하는가**를 정의한다.
되돌리기 어려운 선택의 근거는 [`DECISIONS.md`](DECISIONS.md)에 있고 여기서 중복하지 않는다.

---

## 0. 한 줄 요약

기존 `defect_classifying_model`의 중복·출처 감사 결과를 **승계**하고,
그 위에 **L1 7-class Head A**를 crop 단위로 새로 학습한다.
Simple CNN / EfficientNet-B0 / YOLO-cls 3종을 동일 분할·동일 평가 코드로 비교한다.

---

## 1. 현재 상태

### 1.1 이 저장소

커밋 0개. `_bmad/`(BMAD 툴체인)만 존재한다.

### 1.2 기존 모델 (`../defect_classifying_model`)

| 버전 | 모델 ID | 태스크 | Test 성능 | 상태 |
| --- | --- | --- | --- | --- |
| A0 | `dacon19_b0_224_a0` | L3 19-class | top-1 82.31% / 12-class macro F1 67.58% | baseline 보존 |
| A1/V1 | `dacon19_detail_v1` | L3 19-class | top-1 85.77% / 12-class macro F1 73.57% / ECE 4.87% | 승격 완료 |
| D0 | `yolo26n_detector_d0` | L1 detection | 없음 | **미구현** (`detector.onnx` 파일만 존재) |

**배포 상태 (정확한 표현)**

A1/V1은 학습·평가·ONNX export 및 독립 FastAPI 번들(`S15P11A205/ai-service/`)과 테스트까지
완료됐다. 그러나 **메인 서비스 통합과 운영 배포는 완료되지 않았다.**

- `ai-service/README.md`가 docker-compose · Jenkins · Spring 어댑터 · 모바일 연동을
  "이후 연결할 부분"으로 남겨 둔다.
- 백엔드는 현재 `NoopDefectAnalyzer`를 사용하며 항상 `NORMAL_UNKNOWN`, `confidence=0`을
  반환한다. AI 서비스를 호출하는 HTTP 어댑터 클래스가 존재하지 않는다.
- 기존 `S15P11A205_A0_INTEGRATION.patch`는 A0 모델 ID와 구 필드명을 쓰므로 재사용 금지.

### 1.3 이번 프로젝트가 만드는 것

**Head A — L1 7-class coarse classifier.** 사용자 노출 라벨이 아니다(D-04).
Simple CNN · EfficientNet-B0 · YOLO Classification 3종 비교가 실제 신규 작업이다.
기존 저장소에 Simple CNN과 YOLO Classification 실험은 존재하지 않는다.

---

## 2. 라벨 체계

세 계층은 서로 포함 관계가 아니다. 축이 다르다.

| 계층 | 개수 | 기준 | 사용자 노출 | 이번 범위 |
| --- | ---: | --- | --- | --- |
| L1 | 7 | 시각적 형태, 전 데이터셋 공통 | 안 함 | **★ 학습 대상** |
| L2 | 10 | 수리 업역 | 화면 표시 | 후처리 |
| L3 | 19 | DACON 도배 세부 명칭 | 신뢰도 충족 시 | 기존 V1 |

### 2.1 L1 7-class (학습 대상)

| ID | `unified_label` | 요청서 표현 | 표시 예시 |
| ---: | --- | --- | --- |
| 0 | `crack` | crack | 균열 |
| 1 | `breakage` | breakage | 파손·박리 |
| 2 | `stain_corrosion` | stain_discoloration | 오염·부식 |
| 3 | `moisture_leak` | moisture_leak | 습기·누수 |
| 4 | `lifting` | lifting_peeling | 들뜸 |
| 5 | `mold` | mold | 곰팡이 |
| 6 | `finish_damage` | 기타 통합 클래스 | 마감 손상 |

요청하신 클래스 목록이 기존 L1과 1:1로 대응하므로 **기존 이름을 그대로 쓴다.**
이름을 바꾸면 `dataset_registry.json`의 class_map 5개를 전부 다시 써야 한다.

### 2.2 L1 → L2는 함수가 아니다

`breakage` 하나가 3개 L2에 걸친다.

```
breakage ─┬→ 벽지 손상        (L3: 훼손, 터짐)
          ├→ 벽 구멍·타공      (L3: 오타공, 석고수정, 피스)
          └→ 박리·철근 노출    (AIHub 구조물 출처)
```

따라서 **L1 분류기 단독으로 사용자 표시 라벨을 만들 수 없다.** 서비스 경로는 D-04 참조.
이 프로젝트의 산출물은 L1까지이고, L2 사영은 `configs/l1_to_l2_projection.md`에 규칙으로만
정의한다.

---

## 3. 데이터

### 3.1 원본 (54,797 레코드 / 9.0GB)

| 데이터셋 | 이미지 | 태스크 | bbox | v1 사용 |
| --- | ---: | --- | --- | --- |
| `kaggle_cracks` | 40,000 | binary cls | 없음 | Positive 중 ~1,500 (D-03) |
| `roboflow_house_defect` | 6,028 | detection | 있음 | 실사 non-DACON 분만 |
| `dacon_wallpaper` | 4,249 | detail cls | 없음 | 3,457 (공개 test 제외) |
| `aihub_567` | 2,728 | polygon | polygon | crop 변환분 |
| `roboflow_wallpaper_kr` | 743 | detection | 있음 | 중복 제거 후 잔여분 |
| `mvtec_ad` | 673 | anomaly | mask | **제외** (라이선스) |
| `roboflow_wall_defects` | 376 | detection | 있음 | 전량 |
| `aihub_building_crack` | 0 | — | — | 다운로드 안 됨 |

### 3.2 ★ 클래스와 출처가 사실상 1:1 — 최우선 위험

원본 레코드 수만 보면 다출처처럼 보이지만, **중복 제거 후에는 대부분 단일 출처**다.

**승계할 중복 통계 (마스터 계획 L836-850)**

| 데이터셋 | 전체 | 중복 전체 | 비율 | 해시 확인 |
| --- | ---: | ---: | ---: | ---: |
| Roboflow 한국형 도배 | 743 | 731 | **98.4%** | 731 |
| Roboflow House Defect | 6,028 | 4,271 | **70.9%** | 2,410 |
| DACON | 4,249 | 2,138 | 50.3% | 2,138 |

데이터셋 간 중복 그룹 **1,886개** · SHA-256 완전 일치 쌍 **1,790개**

**House Defect 출처 계보 (마스터 계획 L866)**

| 출처 | 이미지 | 비율 |
| --- | ---: | ---: |
| DACON 재사용 (숫자 파일명) | 2,014 | 33.4% |
| DACON 재사용 (클래스명 영문 번역) | 689 | 11.4% |
| 합성 `fakes*_crop_*` | 976 | 16.2% |
| 기타 (생성 이미지 포함) | 2,349 | 39.0% |

**`effective_origin` 적용 결과 (실측)**

`origin_decisions.csv`의 `effective_origin`으로 평가 가능 여부를 판정하면(D-12), 평가에
쓸 수 있는 데이터셋은 **DACON · AIHub · Kaggle 셋뿐**이다.

| 데이터셋 | `effective_origin` |
| --- | --- |
| `dacon_wallpaper` | `real_verified` 4,249 |
| `aihub_567` | `real_verified` 2,728 |
| `kaggle_cracks` | `real_verified` 40,000 |
| `roboflow_wall_defects` | **`unknown_origin` 375 + `synthetic_verified` 1** ← 강등됨. 별도로 `quarantined=true` 7건 |
| `roboflow_wallpaper_kr` | `dacon_derivative_verified` 729 + `probable` 14 → **real_verified 0** |
| `roboflow_house_defect` | `dacon_derivative` 2,703 + `synthetic` 976 + `unknown` 2,349 → **real_verified 0** |

**L1 클래스별 `origin 기준 eval 후보 원본` (실측)**

이 수치는 **origin 조건만 통과한 원본 이미지 수**다. 최종 평가 표본이 아니다.
아직 반영되지 않은 것: 정책 필터(D-12) · Kaggle 1,500 상한(D-03) · crop 변환(D-01) ·
대표본 축소(D-13) · 70/15/15 분할(D-02). 노트북 01/04에서 단계별로 줄어든다.

| `unified_label` | eval 후보 원본 | 독립 출처 | LOSO | 비고 |
| --- | ---: | ---: | --- | --- |
| `crack` | 20,763 | 2 (kaggle, aihub) | ✅ | D-03 적용 후 약 2,263으로 축소 예정 |
| `breakage` | 2,615 | 2 (aihub, dacon) | ✅ | rhd 1,941 · rwkr 398 전부 부적격 |
| `finish_damage` | 807 | 1 (dacon) | ❌ | rwkr 188 부적격 |
| `stain_corrosion` | 612 | 1 (dacon) | ❌ | rwd 19 · rwkr 12 부적격 |
| `mold` | 145 | 1 (dacon) | ❌ | rhd 579 · rwkr 91 부적격 |
| `lifting` | 76 | 1 (dacon) | ❌ | 대표본 축소 후 test 15%면 10장 내외 |
| **`moisture_leak`** | **0** | **0** | ❌ | **150장 전량 rwd → D-14로 공식 계보에서 제외** |

> `mvtec_ad`는 위 집계에서 제외했다. 전량 `real_verified`이지만 `auxiliary_class_map`으로
> **133장이 L1에 매핑되므로**, origin 조건만 쓰면 D-08에서 제외한 데이터가 재유입된다.
> 정책 필터를 origin보다 먼저 적용하는 이유다 (D-12).

**따라서**

- LOSO 가능: `crack`, `breakage` **2개뿐**
- 나머지 4개: DACON 단일 출처. "in-source holdout 성능"으로만 보고하고 "서비스 일반화
  성능"과 분리 표기한다
- `moisture_leak`: **어떤 성능 수치도 산출할 수 없다.** 처리 방침은 §6.5 / D-14

### 3.3 그 외 데이터 문제

| # | 문제 | 대응 |
| --- | --- | --- |
| P1 | 클래스↔출처 1:1 | 3.2 / 6.2 |
| P2 | kaggle 콘크리트 도메인 불일치 | D-03 상한 샘플링 |
| P3 | 다중 라벨 11장 | D-09 (crop 분해로 대부분 해소) |
| P4 | DACON 공개 test 라벨 없음 792장 (연쇄 잠금 832행) | D-08 `contains_dacon_public_test` 제외 |
| P5 | AIHub `target_class` 공백 1,167 / `review_category_semantics` 945 | 승계 판정 사용, 재검수 안 함 |
| P6 | 극심한 불균형 (breakage vs lifting 약 38배) | 클래스 가중치 ablation |
| P7 | 합성 이미지 976장 split 누수 | D-05 / D-07 |
| P8 | UTF-8 BOM + Windows cp949 | 전 CSV I/O에 `encoding='utf-8-sig'` 명시 + 노트북 01 왕복 테스트 |
| P9 | MVTec 라이선스 상업 불가 | D-08 제외 |
| P10 | detection용 crack bbox 부재 | 선택 실험으로 격리 (§8) |

### 3.4 ★ 프로세스 위험 — 기존 프로젝트의 실패 패턴

`RESTART_HERE.md` 3절의 저자 본인 사후 분석:

> Python 7,978줄 중 **4,325줄(54%)이 오염 검수·감사·overlay**. 개별 이미지 판정 992건.
> hard gate 10개, 블라인드 검수 시스템 5개. **실제 학습은 7분이었다.**
>
> 원인 ① 종료 조건 없는 감사 루프 ② 순서가 뒤집힘 — 어떤 오염이 실제로 문제인지는 모델이
> 알려주기 전엔 모른다 ③ 막혀 있던 건 detection뿐인데 그걸 붙잡고 있었다.

**이 계획에 강제하는 규칙 3개**

1. **모델이 없는 동안 수동 검수는 누적 100건 상한.** 노트북 02가 카운터를 들고 있고
   초과하면 실패한다. 상한을 늘리려면 baseline 결과로 필요성을 입증해야 한다.
2. **모델이 측정할 수 있는 오염은 선제 차단하지 않는다.** 측정하고 대응한다.
3. **새 gate/marker 시스템을 만들지 않는다.** 기존 10개를 승계해서 쓴다.

---

## 4. 산출물 구조

```
banggoot_model/
├── configs/
│   ├── paths.yaml                     # 데이터 루트·승계 원본 경로
│   ├── taxonomy.yaml                  # L1 7-class 정의
│   ├── label_mapping.csv              # ★ source_dataset × original_label → unified_label
│   ├── l1_to_l2_projection.md         # L1→L2 one-to-many 분기 규칙
│   ├── damagetype_mapping.md          # L1/L2 → DamageType (U-1 포함)
│   ├── exp_m1_simple_cnn.yaml
│   ├── exp_m2_effnet_b0.yaml
│   ├── exp_m3_yolo_cls.yaml
│   └── aug/*.yaml                     # 증강 arm별 설정
│
├── src/banggoot/                      # 로직은 전부 여기. 노트북은 호출만.
│   ├── paths.py            config/경로/인코딩 헬퍼
│   ├── inherit.py          ★ 기존 저장소 자산 승계 + SHA-256 검증
│   ├── taxonomy.py         label_mapping 로드·적용
│   ├── crops.py            ★ bbox/polygon → crop 생성 (D-01)
│   ├── metadata.py         metadata.csv 빌드·검증
│   ├── imagestats.py       무결성·해상도·밝기·대비·채도·blur
│   ├── splits.py           group-aware stratified split + 누수 assertion
│   ├── dataset.py          torch Dataset
│   ├── transforms.py       albumentations 실험별 조합
│   ├── models.py           SimpleCNN(nn.Module) + build_effnet_b0(timm)
│   ├── train.py            공통 학습 루프 (seed 고정, early stop)
│   ├── evaluate.py         ★ 전 모델 공통 평가 (sklearn)
│   ├── sourcebias.py       ★ 출처 편향 측정
│   ├── ood.py              ★ OOD/abstention 게이트
│   ├── report.py           비교표 + 오분류 HTML
│   └── export.py           ONNX export + FP16/INT8
│
├── notebooks/                         # §5
├── artifacts/
│   ├── inherited/          승계 자산 스냅샷 + MANIFEST.json (SHA-256)
│   ├── crops/              생성된 crop 이미지
│   ├── metadata/           metadata.csv · label_mapping.csv · integrity_report.json
│   ├── splits/             train/val/test.csv · split_summary.json
│   ├── review/             사람이 볼 검수 큐 (자동 삭제 금지)
│   └── reports/            model_comparison.md · errors_*.html
├── runs/                              # git 제외
├── models/                            # git 제외
└── docs/  PLAN.md · DECISIONS.md · RESULTS.md
```

### 4.1 `metadata.csv` 스키마

요청하신 11개 컬럼 + 실측에 필요한 확장.

| 컬럼 | 설명 |
| --- | --- |
**식별 / 경로**

| 컬럼 | 설명 |
| --- | --- |
| `sample_id` | crop 단위 안정 키 |
| `record_id` | 승계 조인 키 (기존 manifest와 동일) |
| `image_path` | crop 경로 |
| `source_image_path` | 원본 이미지 경로 |
| `source_dataset` | 출처 데이터셋 |
| `source_split` | 원본 split (참고용, 분할 미사용 — D-05) |
| `sha256` | 무결성 |

**라벨**

| 컬럼 | 설명 |
| --- | --- |
| `original_label` | 원본 라벨 (한국어 포함) |
| `unified_label` | L1 7-class |
| `l2_service_group` / `l3_detail_label` | 상위·하위 계층 |
| `is_multilabel` / `all_original_labels` | 다중 하자 (D-09) |

**★ 승계 eligibility (D-12) — 1차 계획에서 누락됐던 부분**

| 컬럼 | 출처 | 설명 |
| --- | --- | --- |
| `split_component_id` | `split_manifest.csv` | split **배정** 단위 (중복그룹 ∪ AIHub `Raw_Data_ID` ∪ source root) |
| `duplicate_group_id` | `split_manifest.csv` | 동일 이미지 판정 단위. **평가 시 1장으로 축소** (D-13) |
| `source_root_id` | `split_manifest.csv` | 합성·증강 원본 식별자 |
| `effective_origin` | **`origin_decisions.csv`** | ★ 권위 있는 origin. `split_manifest.source_origin`이 아니다 |
| `quarantined` / `quarantine_kind` | `origin_decisions.csv` | 7건 |
| `reserved_reason` | `split_manifest.csv` | 잠금 사유 (`contains_dacon_public_test` 등 6종) |
| `legacy_split` | `split_manifest.csv` | 폐기할 기존 80/10/10 배정. **어떤 파생에도 쓰지 않는다** |
| `review_decision` | `review_manifest.csv` 등 | 승계한 사람 판정 992건 |

**★ 파생 컬럼 — 생성 시점이 노트북마다 다르다 (D-12)**

| 컬럼 | 생성 | 설명 |
| --- | :---: | --- |
| `excluded_by_policy` | 01 | 정책 제외 (MVTec · 잠금 · use_status · 비L1) |
| `train_eligible_head_a` | 01 | Head A 학습 적격 |
| `train_auxiliary` | 01 | baseline 제외, ablation 전용 |
| `sample_role` | 01 | `primary` / `auxiliary` / `excluded` |
| `eval_candidate_head_a` | 01 | 평가 **후보**. split 을 참조하지 않는다 |
| `representative_rank` | 01 | 중복그룹 내 결정론적 tie-break 순위 |
| `split` | **04** | 새 70/15/15 배정 |
| `eval_eligible_head_a` | **04** | 후보 ∩ 품질 통과 ∩ val/test |
| `canonical_eval_head_a` | **04** | 대표본 (D-13) |

**계보 구분 (D-14)**

| 컬럼 | 정의 | 실측 |
| --- | --- | ---: |
| `head_a6_eligible` | **`sample_role=='primary'`** ∧ ¬moisture ∧ ¬sealed | 6,509 |
| `head_a6_aux_pool` | `sample_role=='auxiliary'` ∧ ¬moisture ∧ ¬sealed. ablation 전용 | 3,169 |
| `head_a7_aux_train` | 탐색 7-class 계보의 moisture aux | 103 |
| `sealed_future_eval` | **영구 미개봉** | 45 |

`head_a6_eligible`을 `train_eligible_head_a`로 정의하면 auxiliary 3,169장이 baseline에
새어 들어간다 (D-14). assertion으로 강제한다.

**품질 / 기하**

| 컬럼 | 설명 |
| --- | --- |
| `quality_status` | keep / review / drop |
| `rejection_reason` | drop·review 사유 |
| `split` | train / val / test / excluded |
| `width` / `height` / `aspect_ratio` | crop 해상도 |
| `bbox_x/y/w/h` / `bbox_area_ratio` | 원본 내 crop 좌표 |
| `mean_brightness` / `contrast_std` / `saturation` / `blur_var` | 품질 지표 |
| `review_note` | 사람이 남기는 메모 |

> `quality_status != keep` **하나만으로 분할하지 않는다.** 그 규칙으로는
> "기본 학습 가능 / auxiliary 전용 / 평가 절대 불가" 3종을 구별할 수 없다.
> 분할은 `train_eligible_head_a` · `eval_eligible_head_a` · `canonical_eval_head_a`를 쓴다.

`label_mapping.csv`: `source_dataset, original_label, unified_label, l2_service_group,
l3_detail_label, note` — `dataset_registry.json`의 class_map을 전개해 생성한다.

---

## 5. 노트북 순서

노트북은 **실행 드라이버**다. 셀에는 `src/banggoot` 호출과 결과 확인만 쓴다.
로직을 셀에 쓰지 않는다. 모든 노트북 상단에 `%load_ext autoreload` / `%autoreload 2`.

| # | 노트북 | 산출물 | 통과 조건 |
| --- | --- | --- | --- |
| 00 | `00_setup_env_check` | 환경 리포트 | torch CUDA 인식, timm/ultralytics/albumentations import |
| 01 | `01_inherit_and_inventory` | `artifacts/inherited/`, `metadata.csv` v0 (record 단위) + **학습 권한·평가 후보** | 3-way 조인 54,797 일치, SHA-256 전건 일치, MVTec 673 전량 excluded |
| 02 | `02_build_crops` | `artifacts/crops/`, `metadata.csv` v1 (crop 단위) | crop 무결성 0 오류, `duplicate_group`·component 상속 검증 |
| 03 | `03_eda_quality` | EDA 리포트, `quality_status`, 검수 큐 | **수동 검수 누적 ≤ 100건** |
| 04 | `04_build_splits` | `split`, **`eval_eligible_head_a`**, **`canonical_eval_head_a`**, `train/val/test.csv` | 그룹 교차 누수 **0**, 합성 이미지 val/test **0**, 대표본 유일성 |
| 05 | `05_m1_simple_cnn` | run 3개 (seed 1,2,3) | test 미개봉 |
| 06 | `06_m2_effnet_b0` | run 3개 | test 미개봉 |
| 07 | `07_m3_yolo_cls` | run 3개 | test 미개봉 |
| 08 | `08_comparison_and_bias` | 비교표, 편향 리포트 | LOSO는 `crack`/`breakage`만 |
| 09 | `09_ablation` | ablation 표 | arm별 단일 변수 |
| 10 | `10_error_analysis` | `errors_*.html` | 자동 삭제 0건 |
| 11 | `11_ood_gate` | OOD 리포트 | FAR 기준 확정 |
| 12 | `12_final_and_onnx` | 최종 모델 + ONNX | FP32 parity 통과, **test 1회 평가** |
| 13 | `13_quantization` | 양자화 비교표 | 최종 확정 후에만 |
| 90 | `90_optional_yolo_detection` | (선택) | bbox 존재 클래스만 |

**노트북 01이 인벤토리를 새로 만드는 게 아니라 "승계 → 검증 → 확장"이라는 점이 핵심이다.**

---

## 6. 학습과 평가

### 6.1 Baseline 3종 (증강 없음, 동일 split, 동일 평가 코드)

| | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| 이름 | Simple CNN | EfficientNet-B0 | YOLO Classification |
| 구현 | `nn.Module` 직접 (Conv-BN-ReLU ×4 + GAP + FC) | `timm.create_model('efficientnet_b0', pretrained=True)` | `ultralytics` |
| pretrained | **없음 (scratch)** | ImageNet | `yolo11n-cls.pt` |
| 입력 | 224 | 224 | 224 |
| batch | 32 | 32 | 32 |
| optimizer | AdamW | AdamW | AdamW |
| lr | 3e-4 | 3e-4 | 3e-4 |
| scheduler | cosine | cosine | cosine |
| loss | CrossEntropy | CrossEntropy | CrossEntropy |
| epoch / early stop | 20 / patience 5 | 20 / patience 5 | 20 / patience 5 |
| seed | **1, 2, 3** | **1, 2, 3** | **1, 2, 3** |

- 설정은 전부 `configs/exp_m*.yaml`에 두고 노트북에서 하드코딩하지 않는다.
- **실제 사용한 YOLO 모델명·weight 파일명·SHA-256을 config에 기록한다.**
  (기존 저장소의 `scripts/yolo26n.pt`는 detection 가중치이므로 classification에는 쓸 수 없다.)
- YOLO는 산출물 형식이 다르므로 **예측을 CSV로 정규화한 뒤 `src/banggoot/evaluate.py`
  동일 코드로 평가**한다.
- 참고 실측: 기존 A0는 EfficientNet-B0 224px에서 7.1분 / peak VRAM 1.43GB
  (RTX 4050 6GB). 실험 비용은 낮다.

### 6.2 평가 지표

**전 모델 공통**

Accuracy · Macro F1 · Weighted F1 · 클래스별 Precision/Recall/F1/Support ·
Confusion Matrix · Top-1 · Top-3 · 파라미터 수 · 모델 파일 크기 · 이미지 1장당 추론시간

**추가 (편향·불확실성)**

- **seed 3개 평균 ± 표준편차**
- **클래스별 bootstrap 95% 신뢰구간**
- **(클래스 × 출처) F1 분해**
- 클래스별 데이터 수 vs F1 산점도

**최종 선정은 Accuracy가 아니라 Macro F1 기준.** 클래스 불균형이 크므로 Accuracy는
다수 클래스에 좌우된다.

비교표 형식:

| Model | Macro F1 | Weighted F1 | Accuracy | Params | Model Size | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

### 6.3 출처 편향 측정

| 측정 | 대상 | 해석 한계 |
| --- | --- | --- |
| source-ID probe | 전체 | 높은 accuracy는 "shortcut이 **가능하다**"까지만 의미한다. 실제로 사용했다는 증명이 아니다 |
| leave-one-source-out | **`crack`, `breakage`만** | 나머지 5개는 실질 단일 출처라 수행 불가 |
| (클래스×출처) F1 | 전체 | 상시 리포트 |
| source-balanced 평가셋 | `crack`, `breakage`만 | 단일 출처 클래스는 균형 맞출 대상이 없음 |

**단일 출처 클래스는 리포트에 `교차출처검증불가` 플래그를 달고, "in-source holdout 성능"과
"서비스 일반화 성능"을 분리 표기한다.** 기존 프로젝트가 support=0 클래스에 취한 원칙과 같다.

### 6.4 분할 규칙

**배정 (component 단위)**

- seed 고정, **70/15/15 stratified** (D-02)
- 배정 단위 = `split_component_id` (중복그룹 ∪ AIHub `Raw_Data_ID` ∪ source root) — D-05
- **crop은 원본 이미지의 component와 duplicate group을 상속**한다 (D-01)
- `reserved_reason='contains_dacon_public_test'` 832행 하드 제외 — assertion 강제 (D-08)
  (같은 832행이 `split` 컬럼에서는 `reserved_inference`다. **`reserved_reason ==
  'reserved_inference'` 필터는 0건을 반환하며 오류를 내지 않는다**)

**행 단위 권한** — 배정과 권한은 서로 다른 결정이다. 한 표에 섞지 않는다.

- `train_eligible_head_a=false` → 학습 제외
- `train_auxiliary=true` → baseline train 제외, ablation arm에서만 사용
- `eval_eligible_head_a=false` → val/test 지표 계산 제외 (보관은 함)
- 합성 이미지는 val/test 제외 + baseline train 미포함 (D-07)

**평가 대표본 (D-13)**

- val/test는 **`duplicate_group_id`당 대표 원본 1개**의 crop만 지표에 사용
- `split_component_id`로 축소하지 **않는다.** AIHub `Raw_Data_ID`는 서로 다른 시점·각도이지
  중복이 아니다
- 대표 원본에서 나온 crop은 전부 유지 (한 사진의 여러 하자는 각각 독립 표본)
- 대표본은 Head A 축으로 새로 고른다. `*_detection` / `*_detail` 대표본 재사용 금지
- **bootstrap은 crop 단순 재표집이 아니라 중복 그룹 단위 clustered bootstrap**

**기타**

- 증강은 train에만
- **test는 노트북 12 전까지 열지 않는다.** 열람 카운터를 `split_summary.json`에 기록한다

### 6.5 `moisture_leak` 처리 — 두 계보 분리 (D-14)

§3.2 실측대로 `moisture_leak`은 **평가 가능 이미지가 0장**이다. 해결책은 클래스 하나를
어떻게 다룰지가 아니라 **모델 계보를 둘로 나누는 것**이다.

| 계보 | 클래스 | RWD moisture 147장 | 역할 |
| --- | ---: | --- | --- |
| **`head_a6_primary`** | **6** | 사용 안 함 (미개봉 보존) | **공식 baseline.** 모델 비교·최종 지표·서비스 승격의 유일한 기준 |
| `head_a7_rwd_aux` | 7 | `train_auxiliary`로 사용 | 탐색 실험. 6-class 지표에 미치는 영향만 측정 |

`head_a7_rwd_aux`의 제약:

- `moisture_leak` 성능을 **어떤 형태로도 주장하지 않는다**
- 서비스 승격 대상이 아니다
- 보고하는 유일한 수치는 "6-class macro F1이 `head_a6_primary` 대비 어떻게 변했는가"

상세 근거는 D-14 참조. 요약하면 **147장을 baseline 학습에 쓰면 U-2(RWD 출처 조사)가 풀려도
평가에 쓸 수 없게 된다.** 학습에 이미 들어간 사진으로 holdout을 만들 수 없다.

---

## 7. 증강 ablation

Baseline에서 Macro F1이 가장 높은 모델 1개에만 적용한다. **한 번에 하나씩** 바꾼다.

| arm | 내용 |
| --- | --- |
| A0 | 원본 (baseline) |
| A1 | 기하 — horizontal flip · rotation · scale · translation |
| A2 | 색상·HSV — brightness · contrast · saturation · 약한 hue · gamma |
| A3 | A1 + A2 |
| A4 | CLAHE · blur · JPEG compression |
| A5 | 클래스 가중치 |
| A6 | 입력 크기 224 → 320 |
| A7 | learning rate 변경 |
| A8 | `kaggle` 0 / 1,500 / 전량 (D-03 검증) |
| A9 | `head_a6_aux_pool` 3,169장 추가 (출처 미확인 데이터가 도움이 되는가) |

**A9 주의** — auxiliary는 `unknown_origin` / `dacon_derivative_*`라 평가에 쓸 수 없다.
학습에만 넣고 **평가는 항상 primary 기반 val/test로만** 한다. A9가 개선을 보이더라도
그것이 "출처 미확인 데이터가 유용하다"는 근거이지 "그 데이터로 평가해도 된다"는 뜻이 아니다.

**A6 우선순위 높음.** 기존 A1 실험에서 320px가 224px보다 나았다는 실측이 있다.

---

## 8. Error Analysis

혼동 클래스쌍 → 오분류 crop을 `artifacts/reports/errors_{model}.html`에 렌더링한다.
썸네일 · 정답 · 예측 · confidence · **출처** · blur/밝기 지표를 함께 표시한다.

분석 항목: 클래스간 혼동 / 라벨 모호 / 데이터 부족 / 흐림·어두움 / 하자 영역 크기 /
출처 편향 / 모델 구조 차이

**모델이 틀린 이미지를 자동 삭제하지 않는다.** 라벨 오류 의심 건은
`metadata.csv`의 `quality_status='review'` + `review_note`로만 표시한다.

---

## 9. 배포 준비

### 9.1 최종 모델 선정 기준 (순서)

1. Macro F1
2. 데이터가 적은 클래스의 Recall
3. 모델 크기
4. 추론 속도
5. Unity/모바일 배포 가능성

### 9.2 OOD 게이트 (노트북 11 — 배포 전 필수)

v1이 하자 유형 분류에 집중하는 것은 맞지만, 실제 업로드에는 정상 벽 · 비벽면 · 흐린
사진이 들어온다. **confidence threshold만으로 OOD를 안전하게 검출할 수 없다.**

- challenge set 구성: 정상 벽 / 비벽면 / 흐린 사진
  (`roboflow_house_defect` hard negative 3,508장 + `kaggle` Negative를 후보로 사용)
- 측정: **abstention coverage · accepted accuracy · false acceptance rate**
- 재촬영 요청 조건 확정
- blur · 암부 · 과노출 품질 점수 (통합 결정 D5의 미구현분)

**AI 결과는 항상 제안이며 `needsConfirmation=true`, `confirmedByUser=false`로 시작한다.
자동 확정하지 않는다.** severity는 confidence로 대체하지 않고 `null`로 둔다.

### 9.3 ONNX 및 양자화

1. ONNX export
2. FP32 parity 검증 — PyTorch/ONNX top-1·top-3 순서 일치, logit 절대 오차 기록
3. FP16 또는 INT8 양자화
4. 양자화 전후 **Macro F1 / 모델 크기 / 추론시간** 비교

**양자화는 최종 모델 성능이 확정된 이후에만 수행한다.**

### 9.4 서비스 연결 (범위 밖, 후속)

- `DamageType`에 `MOLD` / `FINISH_DAMAGE` 추가 (D-10 — 별도 PR)
- Spring `DefectAnalyzer` HTTP 어댑터 구현 (현재 `NoopDefectAnalyzer`)
- docker-compose · Jenkins 파이프라인
- 모바일 검수 화면 연동

---

## 10. 선택 실험 — YOLO Detection

bbox/polygon 라벨이 실제로 존재하는 경우에만 별도로 수행한다.
Classification과 **다른 실험**으로 관리하고, Classification은 F1, Detection은 mAP로 평가한다.
**이미지 전체를 임의의 bbox로 설정하지 않는다.**

**현실적 제약 (기존 저장소 기록)**

- 정책 B(AIHub full-image 미사용) 적용 시 detection용 `crack` 학습 데이터 **612 → 0**
- canonical eval detection 335장 중 **93.7%가 AIHub** → 제외 시 21장
- `kaggle_cracks` 40,000장은 `task=binary_classification`이라 bbox 없음

→ 실질적으로 bbox는 `roboflow_*` 3종에만 존재하고, 그중 상당수가 중복·합성이다.
**crack detection은 현재 데이터로 불가능하다.** 이 사실을 리포트에 명시한다.

---

## 11. 위험 요소

| # | 위험 | 영향 | 대응 |
| --- | --- | --- | --- |
| R1 | 출처 편향으로 성능이 허수 | 🔴 | §6.3. 5개 클래스는 LOSO 불가 — 명시적 한계 표기 |
| R2 | 검수 루프 무한 확장 | 🔴 | §3.4 규칙 3개. 노트북 03 카운터 강제 |
| R3 | 승계 자산 오염·불일치 | 🔴 | 노트북 01 SHA-256 전건 검증. 불일치 시 중단 |
| R4 | crop 생성이 그룹을 깨뜨림 | 🟠 | crop이 원본 `duplicate_group` 상속. 노트북 02 assertion |
| R5 | 소수 클래스 표본 부족 (`lifting` 등) | 🟠 | bootstrap CI + seed 3개. 단정적 서술 금지 |
| R3b | eligibility를 origin만으로 계산해 **MVTec 133장 재유입** | 🔴 | D-12 정책 필터를 origin보다 먼저 적용 + assertion |
| R6 | DACON 공개 test 유입 / `reserved_inference` 오타로 필터가 조용히 무효화 | 🟠 | 노트북 01에서 값 존재 여부까지 assertion |
| R7 | Test 조기 노출 | 🟡 | 열람 카운터. 05~11에서 test 로드 금지 |
| R8 | Windows cp949 / UTF-8 BOM | 🟡 | 전 CSV I/O 인코딩 명시 + 노트북 01 왕복 테스트 |
| R9 | AIHub 라이선스 (U-2) | 🟡 | 배포 전 확인 |
| R10 | VRAM 6GB | 🟢 | 실측 1.43GB. 320px batch 32도 여유 |
| R11 | Detection 데이터 부재 | 🟢 | §10 격리 + 불가 명시 |
| R12 | 두 저장소 이중 관리 | 🟢 | 기존 저장소 읽기 전용. 이 저장소가 단일 기준 |

---

## 12. 착수 순서

| 단계 | 작업 | 게이트 |
| ---: | --- | --- |
| 0 | ✅ `DECISIONS.md` 확정 | D-01 ~ D-14 |
| 1 | 스캐폴딩 + `configs/*` 생성 | — |
| 2 | 노트북 00 실행 | 환경 검증 통과 |
| 3 | 노트북 01 실행 | 승계 SHA-256 전건 일치 · U-3 해소 |
| 4 | 노트북 02 실행 | crop 무결성 · 그룹 상속 검증 |
| 5 | 노트북 03 실행 | 수동 검수 ≤ 100건 |
| 6 | 노트북 04 실행 | 누수 0 assertion |
| 7 | 노트북 05~07 (seed 3) | test 미개봉 |
| 8 | 노트북 08 | 편향 리포트 |
| 9 | 노트북 09~10 | — |
| 10 | 노트북 11 | OOD FAR 기준 통과 |
| 11 | 노트북 12~13 | test 1회 평가 |

**노트북 01의 클래스×출처 교차표가 첫 의사결정 지점이다.**
§3.2의 추정이 실제로 확인되면, 그 시점에 단일 출처 클래스(특히 `moisture_leak` 150장,
`lifting`)를 유지할지 병합할지 다시 판단한다.

**노트북 01 실행 결과 (2026-08-02)** — `moisture_leak`은 D-14로 처리 완료.
남은 문제는 **`lifting` 평가 후보 54장**이다. 대표본 축소(D-13)와 test 15%를 적용하면
test 표본이 한 자릿수가 된다. 노트북 04에서 재판단한다.
