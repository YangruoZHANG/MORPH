import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Restore and verify the exact archived figure and source-table files.")
    parser.add_argument("--output", type=Path, default=Path("restored_artifacts"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "archive/manifest.json").read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("Choose a fresh output directory.")
    for record in manifest:
        target = args.output / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with target.open("xb") as output:
            for rel in record["parts"]:
                block = (root / rel).read_bytes()
                output.write(block)
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != record["sha256"] or size != record["bytes"]:
            raise ValueError(f"Archive integrity check failed for {record['path']}")
        print(f"Verified {record['path']}")

if __name__ == "__main__":
    main()
