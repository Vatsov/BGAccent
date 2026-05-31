"""Download source dictionaries and verify checksums against sources.lock."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


def parse_sources_lock(lock_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    for source_name, info in data.items():
        if "sha256" not in info:
            raise KeyError(f"Missing 'sha256' field for source '{source_name}' in {lock_path}")
    return data


def verify_checksum(file_path: Path, expected_sha256: str) -> None:
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {file_path}: expected {expected_sha256}, got {actual}"
        )


def should_skip(file_path: Path, expected_sha256: str) -> bool:
    if not file_path.exists():
        return False
    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return actual == expected_sha256


def download_source(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


def main(lock_path: Path, output_dir: Path) -> None:
    sources = parse_sources_lock(lock_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, info in sources.items():
        if info.get("manual"):
            print(
                f"Skipping {source_name}: manual build source, "
                f"build and place it yourself ({info['url']})"
            )
            continue

        url = info["url"]
        expected_sha256 = info["sha256"]
        filename = info.get("filename", f"{source_name}.csv")
        dest = output_dir / filename

        if should_skip(dest, expected_sha256):
            print(f"Skipping {source_name}: already present with matching checksum")
            continue

        print(f"Downloading {source_name} from {url}...")
        download_source(url, dest)
        verify_checksum(dest, expected_sha256)
        print(f"  Checksum verified: {expected_sha256}")


if __name__ == "__main__":
    lock = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/sources.lock")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./sources")
    main(lock, out_dir)
