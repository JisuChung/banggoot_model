# 01 · 클래스 × 출처 교차표

`metadata.csv` v0 기준. crop 변환(02)·품질(03)·split(04) 적용 전이다.

## 공식 baseline 후보 (`head_a6_eligible`) — primary 만

| unified_label | aihub_567 | dacon_wallpaper | kaggle_cracks | 합계 |
| --- | ---: | ---: | ---: | ---: |
| breakage | 793 | 1817 | 0 | 2610 |
| crack | 753 | 0 | 1500 | 2253 |
| finish_damage | 0 | 799 | 0 | 799 |
| lifting | 0 | 76 | 0 | 76 |
| mold | 0 | 144 | 0 | 144 |
| stain_corrosion | 0 | 612 | 0 | 612 |
| 합계 | 1546 | 3448 | 1500 | 6494 |

## 평가 후보 (`eval_candidate_head_a`) — split 미적용

| unified_label | aihub_567 | dacon_wallpaper | kaggle_cracks | 합계 |
| --- | ---: | ---: | ---: | ---: |
| breakage | 793 | 1817 | 0 | 2610 |
| crack | 753 | 0 | 1500 | 2253 |
| finish_damage | 0 | 742 | 0 | 742 |
| lifting | 0 | 54 | 0 | 54 |
| mold | 0 | 144 | 0 | 144 |
| stain_corrosion | 0 | 595 | 0 | 595 |
| 합계 | 1546 | 3352 | 1500 | 6398 |

## ablation pool (`head_a6_aux_pool`) — baseline 에 넣지 않는다

| unified_label | roboflow_house_defect | roboflow_wall_defects | roboflow_wallpaper_kr | 합계 |
| --- | ---: | ---: | ---: | ---: |
| breakage | 1638 | 4 | 398 | 2040 |
| crack | 0 | 183 | 0 | 183 |
| finish_damage | 0 | 0 | 188 | 188 |
| lifting | 0 | 0 | 53 | 53 |
| mold | 579 | 5 | 90 | 674 |
| stain_corrosion | 0 | 19 | 12 | 31 |
| 합계 | 2217 | 211 | 741 | 3169 |

> `head_a6_eligible` 은 `sample_role=='primary'` 만 포함한다.
> `train_eligible_head_a` 로 정의하면 auxiliary 3,169장이 baseline 에 새어 들어간다.

## 계보 배분 (D-14)

- RWD moisture 총 **150**
- 봉인 `sealed_future_eval` **45**행 / 45 component
- 탐색 aux `head_a7_aux_train` **103**
- 격리 `quarantined` **3**

봉인은 `split_component_id` 전체에 적용된다. moisture 행만 봉인하면 같은 component의
`crack` 행이 aux 학습에 들어가 미래 holdout이 오염된다 (D-14).

## 제외 사유

| rejection_reason | 행 수 |
| --- | ---: |
| no_l1_mapping | 23490 |
| kaggle_subsample_not_selected | 18500 |
| reserved_aihub_category_needs_review | 945 |
| reserved_contains_dacon_public_test | 832 |
| license_excluded_dataset | 673 |
| synthetic_verified | 303 |
| reserved_out_of_scope_category | 220 |
| legacy_annotation_error | 15 |
| quarantined_stock_watermark | 4 |
| use_status_excluded | 2 |
| quarantined_synthetic_illustration | 1 |
| quarantined_burned_in_timestamp | 1 |
| quarantined_burned_in_caption | 1 |
