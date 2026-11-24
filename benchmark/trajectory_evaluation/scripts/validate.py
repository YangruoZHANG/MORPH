"""Fail-fast validation of the published result bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    by_root = pd.read_csv(RESULTS / "by_root.csv")
    by_dataset = pd.read_csv(RESULTS / "by_dataset.csv")
    summary = pd.read_csv(RESULTS / "summary_long.csv")
    summary_all = pd.read_csv(RESULTS / "summary_all_conditions_long.csv")
    table = pd.read_csv(RESULTS / "manuscript_table.csv")
    negative = pd.read_csv(RESULTS / "spatial_null_negative_control.csv")
    provenance = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))

    assert len(by_root) == 360
    assert by_root["status"].eq("OK").all()
    assert by_root["case"].nunique() == provenance["n_datasets"] == 30
    assert len(by_dataset) == 270
    assert len(table) == 9
    assert len(negative) == 9
    assert summary["n_planned"].eq(25).all()
    assert summary_all["n_planned"].eq(30).all()
    assert np.isclose(table.loc[table.method.eq("MORPH"), "Coverage"].iloc[0], 1.0)
    assert negative.loc[negative.method.eq("MORPH"), "rejection_rate"].iloc[0] > 0.99
    for metric in ["coverage", "cordist", "branch_f1", "topo_1_him", "feat_imp"]:
        values = pd.to_numeric(by_root[metric], errors="coerce").dropna()
        assert values.between(0, 1).all(), metric
    assert pd.to_numeric(by_root["pt"], errors="coerce").dropna().between(-1, 1).all()
    paired = by_root[["branch_f1", "segment_f1_audit"]].dropna()
    assert (paired["branch_f1"].mean() > paired["segment_f1_audit"].mean())
    topo_counts = summary.loc[summary.metric.eq("topo_1_him")].set_index("method")["n_valid"]
    assert topo_counts.gt(0).all()
    modes = by_root.groupby("method")["topology_mode"].agg(set)
    assert "derived_fate_star" in modes["Pal"]
    assert modes["CeR"] == {"derived_fate_star"}

    hashes = {
        path.name: sha256(path)
        for path in sorted(RESULTS.glob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS.json", "validation.json"}
    }
    (RESULTS / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    report = {
        "passed": True,
        "root_rows": len(by_root),
        "dataset_rows": len(by_dataset),
        "all_method_fits_ok": True,
        "metric_ranges_ok": True,
        "lineage_f1_mean_exceeds_segment_audit_mean": True,
        "all_method_topology_means_filled": True,
        "branchless_methods_use_declared_fate_star": True,
        "result_file_sha256": hashes,
    }
    (RESULTS / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

