"""banggoot — 주거 하자 L1 7-class 분류 모델 (Head A).

노트북은 실행 드라이버이고 로직은 전부 이 패키지에 둔다.
"""

import os

# albumentations 는 import 시 외부 서버로 최신 버전을 조회한다.
# 학습 실행이 네트워크 상태에 좌우되면 안 되고, 오프라인에서 타임아웃으로 느려진다.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

__version__ = "0.1.0"
