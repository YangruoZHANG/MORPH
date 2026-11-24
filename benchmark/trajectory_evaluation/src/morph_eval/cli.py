from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import run


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Score frozen trajectory predictions with six auditable metrics.")
    parser.add_argument("--source", type=Path, required=True, help="Frozen benchmark directory")
    parser.add_argument("--output", type=Path, default=here, help="Evaluation output directory")
    args = parser.parse_args()
    provenance = run(args.source, args.output)
    print(f"Wrote {provenance['n_datasets']} datasets to {args.output / 'results'}")


if __name__ == "__main__":
    main()

