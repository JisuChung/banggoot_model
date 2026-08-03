# banggoot_model

주거 하자 이미지 **L1 7-class 분류 모델(Head A)** 작업 저장소.

Simple CNN · EfficientNet-B0 · YOLO Classification 3종을 동일 분할과 동일 평가 코드로
비교하고, 최종 모델을 ONNX로 내보내는 것이 목표다.

## 먼저 읽을 것

| 문서 | 내용 |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | 무엇을 어떤 순서로 하는가. **여기부터** |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 되돌리려면 재빌드가 필요한 결정과 근거 |

## 이 저장소가 만들지 않는 것

- **사용자에게 표시할 라벨.** L1은 내부 신호이고 화면 표시는 L2다. L1 → L2는
  one-to-many라 L1 분류기 단독으로는 만들 수 없다 (D-04).
- **L3 19-class 도배 상세 분류기.** 그건 `../defect_classifying_model`의 `dacon19_detail_v1`
  (test top-1 85.77%)이 이미 담당한다.
- **detector.** YOLO Detection은 선택 실험이며, 현재 데이터로 `crack` detection은 불가능하다.

## 데이터

원본 9GB는 복사하지 않고 `../defect_classifying_model/data/raw`를 그대로 참조한다.
**기존 저장소는 읽기 전용 증거 저장소다. 여기서 쓰지 않는다.**

중복 그룹·출처 판정·검수 결과는 재계산하지 않고 승계한다 (D-06). 이 판정들은 기존
프로젝트에서 약 4,300줄과 992건의 개별 이미지 판정으로 얻은 결과라 재현 비용이 크다.

## 작업 방식

노트북이 **실행 드라이버**이고 로직은 전부 `src/banggoot/`에 둔다.
셀에는 호출과 결과 확인만 쓴다.

```
notebooks/00_setup_env_check.ipynb        환경 검증
           01_inherit_and_inventory       승계 -> 검증 -> metadata v0
           02_build_crops                 bbox/polygon -> crop
           03_eda_quality                 EDA (수동 검수 <= 100건 상한)
           04_build_splits                70/15/15 group-aware
           05..07_baselines               3종 x seed 3
           08_comparison_and_bias         비교표 + 출처 편향
           09_ablation                    증강 실험
           10_error_analysis              오분류 HTML
           11_ood_gate                    정상/비벽면/흐림
           12_final_and_onnx              test 1회 평가 + export
           13_quantization                FP16/INT8
           90_optional_yolo_detection     선택
```

## 환경

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-train.txt
```

실측 기준 환경: RTX 4050 Laptop 6GB · Python 3.11.9 · torch 2.6.0+cu124.
EfficientNet-B0 224px 학습은 약 7분 / peak VRAM 1.43GB.

## 지켜야 할 규칙

기존 프로젝트는 실제 학습 7분짜리 작업에 감사·검수 코드 4,325줄을 썼다
(`../defect_classifying_model/docs/RESTART_HERE.md` 3절). 재발 방지 규칙:

1. **모델이 없는 동안 수동 검수는 누적 100건 상한.** 초과하려면 baseline 결과로
   필요성을 입증해야 한다.
2. **모델이 측정할 수 있는 오염은 선제 차단하지 않는다.** 측정하고 대응한다.
3. **새 gate/marker 시스템을 만들지 않는다.** 기존 것을 승계해서 쓴다.
4. **test는 노트북 12 전까지 열지 않는다.**
5. **오분류 이미지를 자동 삭제하지 않는다.** `quality_status=review`로 표시만 한다.
