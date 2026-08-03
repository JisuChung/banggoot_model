"""승계 자산 스냅샷 (D-06).

기존 저장소는 읽기 전용 증거 저장소다. 여기서 절대 쓰지 않는다.
승계분은 SHA-256 과 함께 artifacts/inherited/ 로 복사하고 MANIFEST.json 에 기록한다.

중복·출처 판정은 기존 프로젝트에서 약 4,300줄과 992건의 개별 판정으로 얻은 결과다.
재계산하지 않는다.

★ 순서가 중요하다.
    1. 모든 원본 해시를 먼저 계산한다 (쓰기 없음)
    2. 이전 manifest 와 비교한다
    3. 변경됐고 force=False 면 **복사 전에** 중단한다
    4. 통과하면 임시 경로에 복사한 뒤 교체한다
    5. verify() 는 원본과 snapshot 을 각각 검증한다
  1차 구현은 복사를 먼저 하고 나중에 검사해서, 예외가 던져지는 시점에는 이미
  이전 snapshot 이 사라져 있었다. 가드가 아무것도 지키지 못했다.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import load_paths, read_json, sha256, write_json

MANIFEST_NAME = "MANIFEST.json"

# origin 은 2층 구조다. 아래 순서가 권위 순서다 (D-12).
#   split_manifest.source_origin      = build_global_splits.py 의 baseline (오버라이드 미반영)
#   origin_decisions.effective_origin = 권위 있는 최종값
AUTHORITY_NOTE = {
    "split_manifest": "구조 전용 (split_component_id / duplicate_group_id / source_root_id). "
                      "source_origin 은 오버라이드 미반영이므로 쓰지 않는다.",
    "origin_decisions": "★ origin 권위 소스. effective_origin / quarantined / *_effective.",
    "dataset_inventory": "★ label_path 가 여기에만 있다. crop 생성의 전제.",
}


# --------------------------------------------------------------------------- #
# 해시 — 쓰기 없이 계산만 한다
# --------------------------------------------------------------------------- #

def sha256_of_tree(root: Path) -> tuple[str, int, int]:
    """(상대경로, 파일 내용 SHA-256) 목록의 해시. 내용을 실제로 읽는다.

    이름과 크기만 해싱하면 같은 크기로 polygon 좌표가 바뀐 경우를 놓친다.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}|{sha256(p)}\n".encode())
        total += p.stat().st_size
    return h.hexdigest(), len(files), total


def digest(path: Path) -> dict[str, Any]:
    """원본 하나의 해시와 메타. **아무것도 쓰지 않는다.**"""
    if path.is_dir():
        d, n, size = sha256_of_tree(path)
        return {"kind": "dir", "sha256": d, "file_count": n, "bytes": size}
    return {"kind": "file", "sha256": sha256(path), "bytes": path.stat().st_size}


def plan(paths=None) -> list[dict[str, Any]]:
    """승계 대상 전체의 해시를 계산한다. 1단계 — 쓰기 없음."""
    paths = paths or load_paths()
    entries = []
    for key, src in paths.inherit_all().items():
        if not src.exists():
            raise FileNotFoundError(f"승계 대상 없음: {key} -> {src}")
        d = digest(src)
        entry = {
            "key": key,
            "kind": d["kind"],
            "source_path": str(src),
            "source_sha256": d["sha256"],
            "bytes": d["bytes"],
        }
        if d["kind"] == "dir":
            entry["file_count"] = d["file_count"]
            entry["hash_kind"] = "content"
        if key in AUTHORITY_NOTE:
            entry["note"] = AUTHORITY_NOTE[key]
        entries.append(entry)
    return sorted(entries, key=lambda e: e["key"])


# --------------------------------------------------------------------------- #
# 복사 — 검사를 통과한 뒤에만
# --------------------------------------------------------------------------- #

def _commit_file(key: str, src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / f"{key}{src.suffix}"
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)          # 같은 볼륨에서 원자적
    return dest


def _commit_dir(key: str, src: Path, dest_dir: Path) -> Path:
    """기존 snapshot 을 비켜 놓고 새로 복사한 뒤, 성공하면 옛것을 지운다.

    갓 copytree 한 디렉토리를 rename 하면 Windows 에서 PermissionError 가 난다
    (인덱서·백신이 핸들을 잡고 있다). 그래서 임시 트리를 만들어 옮기는 대신
    **목적지에 직접 쓰고, 실패하면 비켜 둔 옛것을 되돌린다.**
    """
    dest = dest_dir / key
    old = dest_dir / f"{key}.old"

    if old.exists():
        shutil.rmtree(old, ignore_errors=True)

    moved = False
    if dest.exists():
        _retry(lambda: dest.rename(old))     # 기존 것은 이미 안정된 상태라 rename 가능
        moved = True

    try:
        shutil.copytree(src, dest)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        if moved and old.exists():
            old.rename(dest)                 # 실패하면 원상복구
        raise

    if moved:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def _retry(fn, attempts: int = 5, delay: float = 0.2):
    """Windows 의 일시적 파일 잠금 대응."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


# --------------------------------------------------------------------------- #
# 진입점
# --------------------------------------------------------------------------- #

def snapshot(force: bool = False) -> dict[str, Any]:
    """승계 대상을 복사하고 MANIFEST.json 을 만든다.

    원본이 이전 승계 이후 바뀌었으면 **아무것도 건드리지 않고** 예외를 던진다.
    """
    paths = load_paths()
    dest_dir = paths.artifact("inherited")
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / MANIFEST_NAME

    # 1. 모든 원본 해시를 먼저 계산
    entries = plan(paths)

    # 2. 이전 manifest 와 비교
    previous = read_json(manifest_path) if manifest_path.exists() else None
    prev = {e["key"]: e for e in (previous or {}).get("entries", [])}
    changed = [
        e["key"] for e in entries
        if e["key"] in prev and prev[e["key"]].get("source_sha256") != e["source_sha256"]
    ]

    # 3. 복사 전에 중단
    if changed and not force:
        raise RuntimeError(
            "원본이 이전 승계 이후 변경됐다:\n  "
            + "\n  ".join(changed)
            + "\n기존 저장소는 읽기 전용이어야 한다(D-06). "
              "기존 snapshot 은 그대로 보존했다. 의도한 변경이면 force=True 로 다시 실행한다."
        )

    # 4. 통과한 뒤에만 복사
    for e in entries:
        src = Path(e["source_path"])
        dest = _commit_dir(e["key"], src, dest_dir) if e["kind"] == "dir" \
            else _commit_file(e["key"], src, dest_dir)
        e["snapshot_path"] = str(dest.relative_to(dest_dir.parent.parent))
        e["snapshot_sha256"] = digest(dest)["sha256"]
        if e["snapshot_sha256"] != e["source_sha256"]:
            raise RuntimeError(f"복사본 해시가 원본과 다르다: {e['key']}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "legacy_root": str(paths.legacy_root),
        "policy": "기존 저장소는 읽기 전용 증거 저장소. "
                  "중복·출처·검수 판정은 재계산하지 않는다 (D-06).",
        "forced": bool(changed and force),
        "changed_keys": changed,
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """원본과 snapshot 을 **각각** manifest 해시와 대조한다.

    원본만 검사하면 "기존 저장소가 안 바뀌었다"만 확인하는 것이고,
    우리 복사본이 온전한지는 확인하지 못한다.
    """
    paths = load_paths()
    root = paths.artifact("inherited").parent.parent
    manifest = manifest or read_json(paths.artifact("inherited") / MANIFEST_NAME)

    out = []
    for e in manifest["entries"]:
        row = {"key": e["key"], "kind": e["kind"], "sha256": e["source_sha256"][:12]}

        src = Path(e["source_path"])
        row["source"] = (
            "missing" if not src.exists()
            else "ok" if digest(src)["sha256"] == e["source_sha256"]
            else "hash_mismatch"
        )

        snap_rel = e.get("snapshot_path")
        snap = root / snap_rel if snap_rel else None
        if snap is None:
            row["snapshot"] = "not_recorded"
        elif not snap.exists():
            row["snapshot"] = "missing"
        else:
            expected = e.get("snapshot_sha256", e["source_sha256"])
            row["snapshot"] = "ok" if digest(snap)["sha256"] == expected else "hash_mismatch"

        row["status"] = "ok" if row["source"] == "ok" and row["snapshot"] == "ok" else "fail"
        out.append(row)
    return out
