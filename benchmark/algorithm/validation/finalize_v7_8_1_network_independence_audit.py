from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


CORE_OUTPUTS = [
    "V781_NETWORK_CONDITIONAL_INDEPENDENCE_AUDIT.csv",
    "V781_RELATION_MULTIPLICITY_AUDIT.csv",
    "V781_RELATION_OPERATION_CLASS_AUDIT.csv",
    "ALL7_NETWORK_AUDITED_RELATION_LAYER.csv",
    "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv",
    "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv",
    "FINAL_NETWORK_AUDIT_DECISION.json",
    "FINAL_NETWORK_AUDIT_REPORT.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
    audit = candidate / "network_independence_audit"
    rerun = audit / "_clean_rerun_1"

    comparison = []
    for name in CORE_OUTPUTS:
        primary = audit / name
        clean = rerun / name
        primary_hash = sha256(primary)
        clean_hash = sha256(clean)
        comparison.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "clean_rerun_sha256": clean_hash,
                "identical": primary_hash == clean_hash,
            }
        )
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(
        audit / "V781_NETWORK_AUDIT_CLEAN_RERUN_HASH_COMPARISON.csv", index=False
    )
    clean_pass = bool(comparison_frame["identical"].all())

    a0_path = (
        project
        / "outputs"
        / "v7_8_0_latent_relation_guided_coupled_reconstruction"
        / "BASELINE_A0_TRAJECTORY_EDGES.csv"
    )
    expected_a0 = json.loads(
        (candidate / "FINAL_POSTVALIDATION_DECISION.json").read_text(encoding="utf-8")
    )
    a0_hash = sha256(a0_path)
    a0_edges = len(pd.read_csv(a0_path))
    a0_pass = (
        a0_hash == expected_a0["A0_sha256"]
        and a0_edges == expected_a0["A0_internal_edges"]
    )

    frozen_audit = (
        project
        / "outputs"
        / "v7_8_0_latent_relation_guided_coupled_reconstruction_freeze_audit"
    )
    frozen_manifest = frozen_audit / "OUTPUT_MANIFEST_SHA256.csv"
    frozen_rows = []
    with frozen_manifest.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            basename = row["path"].replace("/", "\\").rsplit("\\", 1)[-1]
            if basename in {
                "run_v7_8_0_branch_prior_freeze_audit.py",
                "v7_8_0_branch_prior_operational_config.json",
            }:
                target = project / "algorithm" / "experimental" / basename
            else:
                target = frozen_audit / basename
            actual_hash = sha256(target) if target.exists() else ""
            actual_bytes = target.stat().st_size if target.exists() else -1
            frozen_rows.append(
                {
                    "artifact": basename,
                    "expected_bytes": int(row["bytes"]),
                    "actual_bytes": actual_bytes,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_hash,
                    "identical": (
                        target.exists()
                        and actual_bytes == int(row["bytes"])
                        and actual_hash == row["sha256"]
                    ),
                }
            )
    frozen_frame = pd.DataFrame(frozen_rows)
    frozen_frame.to_csv(audit / "V780_FROZEN_MANIFEST_INTEGRITY_AUDIT.csv", index=False)
    frozen_pass = bool(frozen_frame["identical"].all())

    decision_path = audit / "FINAL_NETWORK_AUDIT_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.update(
        {
            "clean_rerun_core_files": len(CORE_OUTPUTS),
            "clean_rerun_core_hashes_identical": clean_pass,
            "A0_internal_edges": a0_edges,
            "A0_sha256": a0_hash,
            "A0_integrity_pass": a0_pass,
            "v780_frozen_manifest_entries_checked": len(frozen_frame),
            "v780_frozen_manifest_integrity_pass": frozen_pass,
            "freeze_candidate_audit_pass": bool(
                clean_pass
                and a0_pass
                and frozen_pass
                and decision["multiplicity_complete_family_pass"]
            ),
            "automatic_freeze": False,
            "candidate_not_frozen": True,
        }
    )
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    supported = pd.read_csv(audit / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv")
    latent = pd.read_csv(audit / "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv")
    conditional = pd.read_csv(
        audit / "V781_NETWORK_CONDITIONAL_INDEPENDENCE_AUDIT.csv"
    )
    report = f"""# v7.8.1 exact-mapping relation layer: freeze-candidate audit

- Final status: `V781_NETWORK_AUDIT_PASS_FREEZE_CANDIDATE_NOT_FROZEN`
- Automatic freeze: **NO**
- Frozen A0 internal trajectory edges: **{a0_edges:,}**
- A0 integrity: **{"PASS" if a0_pass else "FAIL"}**
- Frozen v7.8.0 manifest integrity: **{"PASS" if frozen_pass else "FAIL"}** ({len(frozen_frame)}/{len(frozen_frame)})
- Independent clean-rerun core hashes: **{"PASS" if clean_pass else "FAIL"}** ({len(CORE_OUTPUTS)}/{len(CORE_OUTPUTS)})
- Complete-family multiplicity audit: **{"PASS" if decision["multiplicity_complete_family_pass"] else "FAIL"}**

## Network-level outcome

- Input minimum-validated generic candidates: **{decision["input_validated_edges"]}**
- Supported independent typed relations: **{len(supported)}**
- Latent/redundant after conditioning: **{len(latent)}**
- Supported collision-like: **{int((supported.relation_type == "collision_like").sum())}**
- Supported promotion-like: **{int((supported.relation_type == "promotion_like").sum())}**
- Supported inhibition-like: **{int((supported.relation_type == "inhibition_like").sum())}**
- Mechanism-specific relations: **0**

## Prespecified network audits

{conditional.to_markdown(index=False)}

## Supported independent typed relations

{supported[["dataset", "relation_type", "source_root", "receiver_root", "network_status", "relation_operation_class"]].to_markdown(index=False)}

## Downgraded relation

{latent[["dataset", "relation_type", "source_root", "receiver_root", "network_status", "network_reason"]].to_markdown(index=False)}

## Interpretation boundary

The supported edges are generic predictive/spatial typed relations, not molecular-causal mechanisms.
Exploratory annotations (for example CAF, EMT or M2 macrophage) are not formal mechanism labels.
The relation layer does not alter the validated internal A0 topology. This audit creates a
freeze candidate only; a separate explicit decision is required to freeze it.
"""
    (audit / "FINAL_FREEZE_CANDIDATE_AUDIT.md").write_text(report, encoding="utf-8")

    manifest_targets = [
        audit / "NETWORK_AUDIT_CONFIG_LOCK.json",
        audit / "V781_NETWORK_CONDITIONAL_INDEPENDENCE_AUDIT.csv",
        audit / "V781_RELATION_MULTIPLICITY_AUDIT.csv",
        audit / "V781_RELATION_OPERATION_CLASS_AUDIT.csv",
        audit / "ALL7_NETWORK_AUDITED_RELATION_LAYER.csv",
        audit / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv",
        audit / "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv",
        audit / "V781_NETWORK_AUDIT_CLEAN_RERUN_HASH_COMPARISON.csv",
        audit / "V780_FROZEN_MANIFEST_INTEGRITY_AUDIT.csv",
        audit / "FINAL_NETWORK_AUDIT_DECISION.json",
        audit / "FINAL_NETWORK_AUDIT_REPORT.md",
        audit / "FINAL_FREEZE_CANDIDATE_AUDIT.md",
        project / "algorithm" / "validation" / "v7_8_1_network_independence_audit_config.json",
        project / "algorithm" / "validation" / "run_v7_8_1_network_independence_audit.py",
        Path(__file__).resolve(),
    ]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in manifest_targets
        ]
    )
    manifest.to_csv(audit / "FINAL_FREEZE_CANDIDATE_MANIFEST_SHA256.csv", index=False)

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
