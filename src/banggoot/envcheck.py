"""노트북 00 — 환경 검증.

목적은 "학습을 시작해도 되는가"를 한 번에 판정하는 것이다.
실패는 조용히 넘어가지 않고 report 의 status 로 남긴다.
"""

from __future__ import annotations

import importlib
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .paths import ENCODING, REPO_ROOT, load_paths, load_taxonomy, read_csv, write_json

REQUIREMENTS = REPO_ROOT / "requirements-train.txt"

# PyPI 배포 이름 -> import 이름. 다른 것만 적는다.
IMPORT_NAME = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "jupyterlab": "jupyterlab",
}

_REQ_RE = re.compile(r"^([A-Za-z0-9._-]+)\s*(==|>=)\s*([^\s;#]+)")


def parse_requirements(path: Path | None = None) -> list[dict[str, str]]:
    """requirements-train.txt 를 단일 기준으로 읽는다.

    검사 목록을 코드에 따로 두면 requirements 와 어긋나도 통과한다.
    실제로 1차 구현이 그 문제로 numpy/pandas/opencv/sklearn 불일치를 놓쳤다.
    """
    path = path or REQUIREMENTS
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = _REQ_RE.match(line)
        if not m:
            continue
        name, op, ver = m.groups()
        out.append({"package": name, "op": op, "want": ver})
    return out


def _cmp_ge(got: str, want: str) -> bool:
    try:
        from packaging.version import Version

        return Version(got) >= Version(want)
    except Exception:  # noqa: BLE001 - packaging 이 없으면 느슨하게 비교
        def key(v: str) -> tuple[int, ...]:
            return tuple(int(p) for p in re.findall(r"\d+", v)[:4])

        return key(got) >= key(want)


def _installed_version(pypi_name: str) -> str | None:
    """dist metadata 를 우선 쓴다. __version__ 이 없는 패키지도 있다."""
    try:
        return metadata.version(pypi_name)
    except metadata.PackageNotFoundError:
        pass
    mod_name = IMPORT_NAME.get(pypi_name.lower(), pypi_name.replace("-", "_"))
    try:
        return str(getattr(importlib.import_module(mod_name), "__version__", "?"))
    except Exception:  # noqa: BLE001
        return None


def check_packages() -> list[dict[str, Any]]:
    """requirements 의 모든 pin 을 검사한다. 예외 없이 전부."""
    out = []
    for req in parse_requirements():
        name, op, want = req["package"], req["op"], req["want"]
        row: dict[str, Any] = {"package": name, "op": op, "want": want}

        got = _installed_version(name)
        row["got"] = got

        if got is None:
            row["status"] = "missing"
            out.append(row)
            continue

        # torch 는 "2.6.0+cu124" 처럼 local version 이 붙는다. want 도 같은 형태면 그대로 비교.
        got_base, want_base = got.split("+")[0], want.split("+")[0]
        if op == "==":
            exact = got == want or (got_base == want_base and "+" not in want)
            row["status"] = "ok" if exact else "version_mismatch"
            if "+" in want and got != want:
                row["status"] = "version_mismatch"
        else:  # ">="
            row["status"] = "ok" if _cmp_ge(got_base, want_base) else "version_too_old"
            if row["status"] == "ok" and got != want:
                # [NEW] 패키지 — 첫 baseline 성공 후 이 값으로 == 고정한다
                row["note"] = "resolved_pin_candidate"

        # import 까지 실제로 되는지 확인 (설치만 되고 깨진 경우 탐지)
        mod_name = IMPORT_NAME.get(name.lower(), name.replace("-", "_"))
        try:
            importlib.import_module(mod_name)
            row["import_ok"] = True
        except Exception as exc:  # noqa: BLE001
            row["import_ok"] = False
            row["status"] = "import_error"
            row["error"] = f"{type(exc).__name__}: {exc}"

        out.append(row)
    return out


def check_torch() -> dict[str, Any]:
    """CUDA 를 실제로 잡는지 확인한다. is_available() 만으로는 부족해서 연산까지 돌린다."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"status": "missing", "error": str(exc)}

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        info["status"] = "cpu_only"
        info["note"] = "CPU 휠이 설치됐을 가능성. requirements-train.txt 의 cu124 인덱스 확인"
        return info

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info |= {
        "device_name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_vram_mb": round(props.total_memory / 1024**2),
    }

    # 실제 연산 + AMP 까지 확인
    try:
        a = torch.randn(2048, 2048, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            b = (a @ a).sum()
        torch.cuda.synchronize()
        info["matmul_ok"] = bool(torch.isfinite(b).item())
        info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        info["status"] = "ok"
        del a, b
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001
        info["status"] = "cuda_runtime_error"
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def check_legacy_paths() -> list[dict[str, Any]]:
    """승계 대상이 전부 존재하는지 확인한다 (D-06 / D-12)."""
    paths = load_paths()
    out = []

    for label, path in [
        ("legacy_root", paths.legacy_root),
        ("data_raw", paths.data_raw),
        ("registry", paths.registry),
    ]:
        out.append({"kind": "base", "key": label, "path": str(path),
                    "exists": path.exists(), "required": True})

    for key, path in paths.inherit_all().items():
        out.append({"kind": "inherit", "key": key, "path": str(path),
                    "exists": path.exists(), "required": True})

    for key in paths.raw["legacy"]["reference_only"]:
        path = paths.reference_only(key)
        out.append({"kind": "reference", "key": key, "path": str(path),
                    "exists": path.exists(), "required": False})

    return out


def check_raw_datasets() -> list[dict[str, Any]]:
    paths = load_paths()
    expected = [
        "dacon_wallpaper", "aihub_seoul_aged_housing",
        "kaggle_infrastructure_structural_defects",
        "roboflow_wall_defects", "roboflow_wallpaper_kr", "roboflow_house_defect",
    ]
    out = []
    for name in expected:
        d = paths.data_raw / name
        out.append({"dataset": name, "path": str(d), "exists": d.is_dir()})
    return out


def check_encoding() -> dict[str, Any]:
    """한글 CSV round-trip. Windows cp949 기본값 때문에 실제로 자주 깨진다 (P8)."""
    info: dict[str, Any] = {
        "sys_default_encoding": sys.getdefaultencoding(),
        "preferred_encoding": __import__("locale").getpreferredencoding(False),
        "stdout_encoding": sys.stdout.encoding,
        "io_encoding": ENCODING,
    }
    try:
        rows = read_csv(REPO_ROOT / "configs" / "label_mapping.csv")
        dacon = {r["original_label"] for r in rows if r["source_dataset"] == "dacon_wallpaper"}
        info["label_mapping_rows"] = len(rows)
        info["dacon_classes"] = len(dacon)
        info["korean_roundtrip_ok"] = ("훼손" in dacon and "창틀,문틀수정" in dacon)
        info["first_header"] = list(rows[0].keys())[0]  # BOM 이 남으면 여기서 드러난다
        info["status"] = "ok" if (len(dacon) == 19 and info["korean_roundtrip_ok"]
                                  and info["first_header"] == "source_dataset") else "fail"
    except Exception as exc:  # noqa: BLE001
        info["status"] = "fail"
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def check_configs() -> dict[str, Any]:
    try:
        tax = load_taxonomy()
        names = [c["name"] for c in tax["classes"]]
        return {
            "status": "ok" if len(names) == 7 else "fail",
            "l1_classes": names,
            "sample_unit": tax.get("sample_unit"),
            "normal_is_a_class": tax.get("normal_is_a_class"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def run() -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": check_packages(),
        "torch": check_torch(),
        "legacy_paths": check_legacy_paths(),
        "raw_datasets": check_raw_datasets(),
        "encoding": check_encoding(),
        "configs": check_configs(),
    }
    report["blockers"] = collect_blockers(report)
    report["status"] = "ok" if not report["blockers"] else "blocked"
    return report


def collect_blockers(report: dict[str, Any]) -> list[str]:
    out = []
    for p in report["packages"]:
        if p["status"] in {"missing", "import_error", "version_too_old"}:
            out.append(f"패키지 {p['status']}: {p['package']} "
                       f"(want {p['op']}{p['want']}, got {p['got']})")
        elif p["status"] == "version_mismatch":
            # == pin 불일치는 재현성 위험이므로 차단한다.
            # 의도적으로 바꾸려면 requirements-train.txt 를 먼저 고친다.
            out.append(f"버전 불일치: {p['package']} "
                       f"want {p['op']}{p['want']}, got {p['got']}")
    if report["torch"].get("status") not in {"ok"}:
        out.append(f"torch/CUDA 상태: {report['torch'].get('status')}")
    for row in report["legacy_paths"]:
        if row["required"] and not row["exists"]:
            out.append(f"승계 경로 없음: {row['key']} -> {row['path']}")
    for row in report["raw_datasets"]:
        if not row["exists"]:
            out.append(f"원본 데이터셋 없음: {row['dataset']}")
    if report["encoding"].get("status") != "ok":
        out.append("한글 CSV round-trip 실패")
    if report["configs"].get("status") != "ok":
        out.append("configs 로드 실패")
    return out


def save(report: dict[str, Any]) -> Path:
    out = load_paths().artifact("experiments") / "env_report.json"
    return write_json(out, report)


def summary(report: dict[str, Any]) -> str:
    lines = [f"status: {report['status']}", ""]

    bad = [p for p in report["packages"] if p["status"] != "ok"]
    lines.append(
        f"패키지 {len(report['packages']) - len(bad)}/{len(report['packages'])} ok "
        f"(requirements-train.txt 기준)"
    )
    for p in bad:
        lines.append(f"  [{p['status']}] {p['package']}: want {p['op']}{p['want']} got {p['got']}")

    pend = [p for p in report["packages"] if p.get("note") == "resolved_pin_candidate"]
    if pend:
        lines.append(f"  고정 후보 ({len(pend)}) — 첫 baseline 성공 후 == 로 바꾼다:")
        for p in pend:
            lines.append(f"    {p['package']}=={p['got']}")

    t = report["torch"]
    lines.append(
        f"torch {t.get('torch')} / cuda_build {t.get('cuda_build')} / "
        f"{t.get('device_name', 'CPU')} {t.get('total_vram_mb', '')}MB -> {t.get('status')}"
    )

    miss = [r for r in report["legacy_paths"] if r["required"] and not r["exists"]]
    n_req = sum(1 for r in report["legacy_paths"] if r["required"])
    lines.append(f"승계 경로 {n_req - len(miss)}/{n_req} 존재")
    for r in miss:
        lines.append(f"  [없음] {r['key']}: {r['path']}")

    e = report["encoding"]
    lines.append(
        f"인코딩: preferred={e.get('preferred_encoding')} "
        f"dacon={e.get('dacon_classes')}/19 roundtrip={e.get('korean_roundtrip_ok')}"
    )

    if report["blockers"]:
        lines += ["", "차단 이슈:"] + [f"  - {b}" for b in report["blockers"]]
    else:
        lines += ["", "차단 이슈 없음. 노트북 01 진행 가능."]
    return "\n".join(lines)
