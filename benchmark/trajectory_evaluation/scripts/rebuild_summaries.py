"""Rebuild headline and negative-control summaries from the audited root table."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morph_eval.evaluator import write_result_tables

scores = pd.read_csv(ROOT / "results" / "by_root.csv")
write_result_tables(scores, ROOT / "results")
print("Rebuilt positive-trajectory and spatial-null summaries.")

