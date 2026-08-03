"""경로 · 설정 · 인코딩 헬퍼.

configs/paths.yaml 이 단일 기준이다. 경로를 코드나 노트북에 하드코딩하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# src/banggoot/paths.py -> 저장소 루트
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# 전 CSV I/O 공통. Windows cp949 기본값 + UTF-8 BOM 혼재 대응 (PLAN P8)
ENCODING = "utf-8-sig"


@dataclass(frozen=True)
class Paths:
    """configs/paths.yaml 을 해석한 결과."""

    raw: dict[str, Any]

    @property
    def legacy_root(self) -> Path:
        return (REPO_ROOT / self.raw["legacy"]["root"]).resolve()

    @property
    def data_raw(self) -> Path:
        return (REPO_ROOT / self.raw["legacy"]["data_raw"]).resolve()

    @property
    def registry(self) -> Path:
        return (REPO_ROOT / self.raw["legacy"]["registry"]).resolve()

    def inherit(self, key: str) -> Path:
        """승계 대상 경로. 없는 키는 KeyError 로 즉시 실패한다."""
        return (REPO_ROOT / self.raw["legacy"]["inherit"][key]).resolve()

    def inherit_all(self) -> dict[str, Path]:
        return {k: self.inherit(k) for k in self.raw["legacy"]["inherit"]}

    def reference_only(self, key: str) -> Path:
        return (REPO_ROOT / self.raw["legacy"]["reference_only"][key]).resolve()

    def artifact(self, key: str) -> Path:
        return REPO_ROOT / self.raw["artifacts"][key]

    @property
    def runs(self) -> Path:
        return REPO_ROOT / self.raw["runs"]

    @property
    def models(self) -> Path:
        return REPO_ROOT / self.raw["models"]


def load_paths(config: Path | None = None) -> Paths:
    path = config or (CONFIG_DIR / "paths.yaml")
    return Paths(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_taxonomy(config: Path | None = None) -> dict[str, Any]:
    path = config or (CONFIG_DIR / "taxonomy.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def l1_classes() -> list[str]:
    """L1 7-class 이름을 id 순으로 반환."""
    tax = load_taxonomy()
    return [c["name"] for c in sorted(tax["classes"], key=lambda c: c["id"])]


# --------------------------------------------------------------------------- #
# CSV I/O — 인코딩을 절대 기본값에 맡기지 않는다
# --------------------------------------------------------------------------- #

def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding=ENCODING, newline="") as fh:
        return list(csv.DictReader(fh))


def iter_csv(path: Path | str) -> Iterator[dict[str, str]]:
    """대용량 manifest 용. split_manifest.csv 는 17MB / 54,797행이다."""
    with Path(path).open(encoding=ENCODING, newline="") as fh:
        yield from csv.DictReader(fh)


def write_csv(path: Path | str, rows: Iterable[dict[str, Any]], fields: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=ENCODING, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
