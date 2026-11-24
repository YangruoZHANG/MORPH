#!/usr/bin/env python
"""Materialize the frozen v1 benchmark scenario and seed registries."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "protocol_v1"
EFFECT = {"null": 0.0, "weak": 0.35, "medium": 0.70, "strong": 1.10}


def core_settings() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 1
    for event_type, n_roots, effect in product(("promotion", "inhibition"), (2, 3, 4), ("weak", "medium", "strong")):
        rows.append(
            {
                "setting_id": f"CORE_{counter:02d}",
                "family": "positive_factorial",
                "event_type": event_type,
                "n_roots": n_roots,
                "effect_level": effect,
                "effect_strength": EFFECT[effect],
                "confounding": "low",
                "exposure_overlap": "adequate",
                "spatial_overlap": "adequate",
                "concurrent_events": 1,
                "membership_ambiguity": 0.0,
            }
        )
        counter += 1
    safety = [
        ("null_contact", 3, "null", "low", "adequate", "adequate", 1, 0.0),
        ("null_contact", 4, "null", "low", "adequate", "excessive", 1, 0.0),
        ("confounding_only", 3, "null", "high", "adequate", "adequate", 1, 0.0),
        ("confounding_only", 4, "null", "high", "sparse", "excessive", 1, 0.0),
        ("sparse_exposure", 3, "weak", "low", "sparse", "adequate", 1, 0.0),
        ("overlap_unrelated", 4, "null", "high", "adequate", "excessive", 1, 0.0),
    ]
    for values in safety:
        event_type, n_roots, effect, confounding, exposure, spatial, concurrent, ambiguity = values
        rows.append(
            {
                "setting_id": f"CORE_{counter:02d}",
                "family": "safety",
                "event_type": event_type,
                "n_roots": n_roots,
                "effect_level": effect,
                "effect_strength": EFFECT[effect],
                "confounding": confounding,
                "exposure_overlap": exposure,
                "spatial_overlap": spatial,
                "concurrent_events": concurrent,
                "membership_ambiguity": ambiguity,
            }
        )
        counter += 1
    complex_settings = [
        ("overlapping_events", 3, "medium", "low", "adequate", "adequate", 2, 0.0),
        ("overlapping_events", 4, "strong", "high", "adequate", "adequate", 3, 0.0),
        ("collision_convergence", 3, "medium", "low", "adequate", "adequate", 1, 0.0),
        ("collision_convergence", 4, "strong", "low", "adequate", "excessive", 1, 0.0),
        ("ambiguous_membership", 3, "medium", "low", "adequate", "adequate", 1, 0.25),
        ("wrong_soft_membership", 4, "medium", "high", "adequate", "excessive", 1, 0.40),
    ]
    for values in complex_settings:
        event_type, n_roots, effect, confounding, exposure, spatial, concurrent, ambiguity = values
        rows.append(
            {
                "setting_id": f"CORE_{counter:02d}",
                "family": "complex",
                "event_type": event_type,
                "n_roots": n_roots,
                "effect_level": effect,
                "effect_strength": EFFECT[effect],
                "confounding": confounding,
                "exposure_overlap": exposure,
                "spatial_overlap": spatial,
                "concurrent_events": concurrent,
                "membership_ambiguity": ambiguity,
            }
        )
        counter += 1
    for row in rows:
        row.update({"cells_per_root": 400, "n_genes": 300, "n_replicates": 30, "analysis_tier": "core"})
    return rows


def stress_settings() -> list[dict[str, object]]:
    factors = {
        "cells_per_root": [200, 800],
        "n_genes": [110, 600],
        "spatial_noise": [0.0, 1.0],
        "expression_noise": [0.25, 0.80],
        "root_imbalance": ["3:1:1", "6:2:1"],
        "event_window_width": [0.10, 0.50],
        "concurrent_events": [2, 4],
        "membership_ambiguity": [0.15, 0.40],
    }
    defaults: dict[str, object] = {
        "cells_per_root": 400,
        "n_genes": 300,
        "spatial_noise": 0.35,
        "expression_noise": 0.45,
        "root_imbalance": "1:1:1",
        "event_window_width": 0.25,
        "concurrent_events": 1,
        "membership_ambiguity": 0.0,
    }
    rows = []
    counter = 1
    for factor, levels in factors.items():
        for level in levels:
            row = dict(defaults)
            row[factor] = level
            row.update(
                {
                    "setting_id": f"STRESS_{counter:02d}",
                    "family": "one_factor_stress",
                    "stress_factor": factor,
                    "stress_level": level,
                    "event_type": "promotion",
                    "n_roots": 3,
                    "effect_level": "medium",
                    "effect_strength": EFFECT["medium"],
                    "confounding": "low",
                    "exposure_overlap": "adequate",
                    "spatial_overlap": "adequate",
                    "n_replicates": 15,
                    "analysis_tier": "stress_supplementary",
                }
            )
            rows.append(row)
            counter += 1
    return rows


def seeds(prefix: str, n: int, start: int) -> pd.DataFrame:
    return pd.DataFrame({"replicate": range(1, n + 1), "seed": [start + i * 1009 for i in range(n)], "registry": prefix})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.DataFrame(core_settings())
    stress = pd.DataFrame(stress_settings())
    core.to_csv(OUT / "CORE_SCENARIOS_V1.csv", index=False)
    stress.to_csv(OUT / "STRESS_SCENARIOS_V1.csv", index=False)
    seeds("core", 30, 7112101).to_csv(OUT / "CORE_SEEDS_V1.csv", index=False)
    seeds("stress", 15, 8112101).to_csv(OUT / "STRESS_SEEDS_V1.csv", index=False)
    summary = {
        "protocol_version": "1.0",
        "core_settings": len(core),
        "core_replicates_per_setting": 30,
        "core_total_replicates": int(core["n_replicates"].sum()),
        "stress_settings": len(stress),
        "stress_replicates_per_setting": 15,
        "stress_total_replicates": int(stress["n_replicates"].sum()),
    }
    (OUT / "PROTOCOL_REGISTRY_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hashes = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "PROTOCOL_FILE_HASHES.csv":
            hashes.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT / "PROTOCOL_FILE_HASHES.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
