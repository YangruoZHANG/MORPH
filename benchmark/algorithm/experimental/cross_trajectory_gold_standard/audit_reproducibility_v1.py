#!/usr/bin/env python3
"""Compare independent benchmark reruns at the content level."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import pandas as pd


def content(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_pair(scope: str, relative: str, left_root: Path, right_root: Path) -> dict:
    left = left_root / relative
    right = right_root / relative
    left_bytes = content(left) if left.exists() else b""
    right_bytes = content(right) if right.exists() else b""
    return {
        "scope": scope,
        "file": relative.replace("\\", "/"),
        "left_exists": left.exists(),
        "right_exists": right.exists(),
        "left_sha256": digest(left_bytes) if left.exists() else "",
        "right_sha256": digest(right_bytes) if right.exists() else "",
        "identical_content": bool(left.exists() and right.exists() and left_bytes == right_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-base", required=True, type=Path)
    parser.add_argument("--right-base", required=True, type=Path)
    parser.add_argument("--left-event", required=True, type=Path)
    parser.add_argument("--right-event", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    base_files = sorted(
        set(p.name for p in args.left_base.glob("*") if p.is_file())
        | set(p.name for p in args.right_base.glob("*") if p.is_file())
    )
    event_files = ["synthetic.h5ad"]
    truth_files = sorted(
        set(f"truth/{p.name}" for p in (args.left_event / "truth").glob("*.csv"))
        | set(f"truth/{p.name}" for p in (args.right_event / "truth").glob("*.csv"))
    )
    rows = [compare_pair("base", name, args.left_base, args.right_base) for name in base_files]
    rows += [compare_pair("event", name, args.left_event, args.right_event) for name in event_files]
    rows += [compare_pair("truth", name, args.left_event, args.right_event) for name in truth_files]
    audit = pd.DataFrame(rows)
    audit.insert(0, "seed", args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    if not audit["identical_content"].all():
        failed = audit.loc[~audit["identical_content"], ["scope", "file"]].to_dict("records")
        raise SystemExit(f"Reproducibility audit failed: {failed}")
    print(f"PASS: {len(audit)} files are content-identical for seed {args.seed}")


if __name__ == "__main__":
    main()
