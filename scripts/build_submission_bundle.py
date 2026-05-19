#!/usr/bin/env python3
"""Build a single zip bundle with key submission artifacts and a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = [
    "README.md",
    "requirements.txt",
    "docs/pitch.md",
    "docs/demo_script.md",
    "docs/hackathon_master_plan.md",
    "docs/architecture_diagram.md",
    "docs/streamlit_cloud_setup.md",
    "risksentinel.jpg",
    "artifacts/demo_check_latest.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_meta() -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def build_bundle(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    bundle_path = output_dir / f"submission_bundle_{ts}.zip"

    include_paths: list[Path] = []
    missing: list[str] = []
    for rel in CORE_FILES:
        p = ROOT / rel
        if p.exists() and p.is_file():
            include_paths.append(p)
        else:
            missing.append(rel)

    manifest = {
        "schema_version": "submission_bundle.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_meta(),
        "included_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in include_paths
        ],
        "missing_files": missing,
    }

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path in include_paths:
            arcname = str(path.relative_to(ROOT))
            zf.write(path, arcname)

    return bundle_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hackathon submission bundle zip.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the bundle zip will be written (default: artifacts)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    bundle = build_bundle(out_dir)
    print(f"Created bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
