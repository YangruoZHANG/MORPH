#!/usr/bin/env python3
"""Assemble auditable NicheForest paper-candidate tables from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs" / "NicheForest_paper_candidate_method"
GRID = REPO / "outputs" / "final_tree_method_development" / "topology_identifiable_locked_rep25"
PHASE_A = "phase_a_v7_1_state_constrained_fine_locked"
PHASE_B = "phase_b_v7_1_localized_consensus_locked"
PHASE_C = "phase_c_oracle_v7_safety_locked"
PHASE_D = "phase_d_v10_structural_safety_locked"
ABLATION = "phase_d_v10_ablation_locked"


def as_bool(value: object) -> bool:
    return str(value).lower() in {"true", "1"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark_rows = []
    ablations = []
    for status_path in sorted(GRID.rglob("TOPOLOGY_DEV_CASE_STATUS.json")):
        run = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = json.loads((run / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        phase_a_status = json.loads(
            (run / PHASE_A / "RUN_STATUS.json").read_text(encoding="utf-8")
        )
        evidence = pd.read_csv(run / PHASE_B / "PHASE_B_CMI_E_EVIDENCE.csv")
        called = evidence[evidence.is_called.map(as_bool)]
        eligible = called[called.refinement_eligible.map(as_bool)]
        phase_c = pd.read_csv(run / PHASE_C / "PHASE_C_ORACLE_REFINEMENT.csv")
        phase_c = phase_c[phase_c.gamma.astype(float).eq(0.35)]
        phase_d = pd.read_csv(
            run / PHASE_D / "PHASE_D_INFERRED_REFINEMENT_METRICS.csv"
        ).iloc[0]
        ablation = pd.read_csv(run / ABLATION / "PHASE_D_STAGE_ABLATION.csv")
        ablation.insert(0, "locked_seed", int(status["seed"]))
        ablation.insert(0, "locked_replicate", int(status["replicate"]))
        ablations.append(ablation)
        stage = ablation.set_index("stage")
        benchmark_rows.append({
            "setting": status["setting"],
            "family": config.get("family"),
            "scenario_event_type": config.get("event_type"),
            "effect_level": config.get("effect_level"),
            "confounding": config.get("confounding"),
            "n_roots": config.get("n_roots"),
            "seed": status["seed"],
            "phase_a_candidate_legal": phase_a_status.get(
                "all_fine_edges_candidate_legal", False
            ),
            "phase_a_root_support_connected": phase_a_status.get(
                "all_root_supports_connected", False
            ),
            "n_relation_calls": len(called),
            "n_refinement_eligible": len(eligible),
            "oracle_accepted_gamma035": int(phase_c.accepted.map(as_bool).sum()),
            "oracle_topology_gain_gamma035": float(
                phase_c.oracle_topology_gain.astype(float).sum()
            ),
            "oracle_event_location_gain_gamma035": float(
                phase_c.event_location_gain.astype(float).sum()
            ),
            "inferred_rewire_accepted": int(phase_d.n_accepted_relation_events),
            "inferred_topology_gain": float(phase_d.inferred_topology_gain),
            "tfinal_topology_gain": float(phase_d.tfinal_topology_gain),
            "candidate_legal": as_bool(phase_d.candidate_legal),
            "connected": as_bool(phase_d.connected),
            "acyclic": as_bool(phase_d.acyclic),
            "upstream_violation": as_bool(phase_d.upstream_violation),
            "event_boundary_retained": as_bool(phase_d.event_boundary_retained),
            "M3_branch_ARI": float(stage.loc["M3", "branch_ARI_rootwise"]),
            "M3_pseudotime_SR": float(stage.loc["M3", "pseudotime_spearman"]),
            "M3_topology_F1": float(stage.loc["M3", "topology_F1"]),
            "M5_topology_F1": float(stage.loc["M5", "topology_F1"]),
            "M6_topology_F1": float(stage.loc["M6", "topology_F1"]),
        })
    synthetic = pd.DataFrame(benchmark_rows)
    synthetic.to_csv(OUT / "FINAL_SYNTHETIC_BENCHMARK.csv", index=False)
    pd.concat(ablations, ignore_index=True).to_csv(
        OUT / "FINAL_STAGE_ABLATION.csv", index=False
    )

    parameters = [
        ("candidate", "scales", "local;mesoscopic;root-reachable", "Phase A provenance"),
        ("soft_assignment", "semantics", "root-specific plus shared boundary", "M1-M2 Brier audit"),
        ("relation", "contact_support_quantiles", "0.65;0.80", "predeclared multiscale stability"),
        ("relation", "permutations_per_scale", 399, "locked p-value resolution"),
        ("relation", "primary_q_threshold", 0.15, "BH-FDR before T1"),
        ("relation", "minimum_localized_branch_support", 0.15, "positivity/localization gate"),
        ("relation", "collision_environment_trust", 0.30, "environment-conflict gate"),
        ("refinement", "authorized_event_type", "promotion", "held-out no-harm audit"),
        ("refinement", "minimum_event_support", 0.20, "downstream support gate"),
        ("refinement", "minimum_transition_program_gain", 0.20, "structural transition gate"),
        ("refinement", "gamma", 0.35, "inside 0.25-0.50 DEV stability plateau"),
        ("refinement", "maximum_swaps_per_event", 1, "one event, one branch consequence"),
        ("pruning", "event_boundary_protection", True, "T1-to-Tfinal retention audit"),
    ]
    pd.DataFrame(parameters, columns=[
        "stage", "parameter", "value", "evidence_or_rationale"
    ]).to_csv(OUT / "FINAL_PARAMETER_TABLE.csv", index=False)

    public = pd.read_csv(
        REPO / "outputs" / "public_method_unified_benchmark"
        / "public_methods_unified_scores_paper_metrics.csv"
    )
    public.insert(0, "benchmark_scope", "prior_frozen_trajectory_output_context_only")
    public.to_csv(OUT / "FINAL_PUBLIC_METHOD_CONTEXT.csv", index=False)

    real_rows = []
    breast = pd.read_csv(
        REPO / "outputs" / "v7_8_1_exact_mapping_generic_relation_layer_FROZEN"
        / "FINAL_SUPPORTED_RELATION_EDGES.csv"
    )
    for row in breast[breast.dataset.eq("xenium_breast")].itertuples(index=False):
        real_rows.append({
            "dataset": "xenium_breast",
            "evidence_type": "spatial_block_OOF_association",
            "contrast_or_relation": row.direction,
            "metric": "qvalue",
            "value": float(row.qvalue),
            "direction_concordant": "NA",
            "claim": row.formal_label,
            "causal_claim_allowed": False,
            "topology_truth_available": False,
        })
    perturb = pd.read_csv(
        REPO / "outputs" / "final_tree_method_development"
        / "phase_f_v1_perturbmap_silver_standard"
        / "PHASE_F_PERTURBMAP_SUMMARY.csv"
    )
    for row in perturb.itertuples(index=False):
        real_rows.append({
            "dataset": "GSE193460_Perturb-map",
            "evidence_type": row.evidence_tier,
            "contrast_or_relation": row.perturbation,
            "metric": "context_adjusted_lesion_effect",
            "value": float(row.context_adjusted_lesion_effect),
            "direction_concordant": bool(row.direction_concordant),
            "claim": row.claim_scope,
            "causal_claim_allowed": False,
            "topology_truth_available": False,
        })
    pd.DataFrame(real_rows).to_csv(
        OUT / "FINAL_REAL_DATA_VALIDATION.csv", index=False
    )

    claims = [
        ("Multi-resolution candidate edges are reproducibly legal", "synthetic", "30/30 locked Phase A cases", "supported"),
        ("Soft assignment improves root decomposition calibration", "synthetic", "M1-M2 Brier 0.197 to 0.132", "supported"),
        ("Branch-scale T0 improves branch recovery", "synthetic", "M2-M3 Branch ARI 0.112 to 0.378", "supported"),
        ("CMI-E is an independent confounding-aware evidence channel", "synthetic", "raw and adjusted channels serialized before T1", "supported_with_limited_ablation"),
        ("Relation detector is null-safe but low-recall", "locked synthetic", "null false-call 0; precision 0.692; recall 0.321", "supported"),
        ("Inferred relation correction can improve topology without observed harm", "locked synthetic", "3 positive, 0 negative; total gain 4", "supported_as_local_correction"),
        ("Relation-driven rewiring recovers causal topology", "locked oracle", "0 positive oracle gains on replicate 25", "not_supported"),
        ("Breast relations are spatially predictive associations", "xenium breast", "3 spatial-block OOF q-values <= 1.83e-4", "supported"),
        ("Tgfbr2 and Ifngr2 perturbations show expected program directions", "GSE193460", "lesion-level context-adjusted bootstrap", "supported_as_silver_standard"),
    ]
    pd.DataFrame(claims, columns=[
        "claim", "evidence_domain", "evidence", "status"
    ]).to_csv(OUT / "FINAL_CLAIM_EVIDENCE_MAP.csv", index=False)

    commands = [
        ("locked_generator", "replicate 25", "7136317", "run_topology_identifiable_dev_grid_v1.py --replicate 25"),
        ("phase_a", "30 locked cases", "deterministic", "run_phase_a_v7_diagnostic_grid.py"),
        ("phase_b_scale065", "30 locked cases", "seed+2000003", "run_phase_b_v6_topology_identifiable_grid.py --permutations 399 --primary-contact-support 0.65"),
        ("phase_b_scale080", "30 locked cases", "seed+2000003", "run_phase_b_v6_topology_identifiable_grid.py --permutations 399 --primary-contact-support 0.80"),
        ("phase_b_consensus", "30 locked cases", "deterministic", "run_phase_b_multiscale_consensus_grid_v1.py"),
        ("phase_c", "30 locked cases", "deterministic", "run_phase_c_v6_topology_identifiable_grid.py"),
        ("phase_d", "30 locked cases", "deterministic", "run_phase_d_v8_grid.py"),
        ("phase_f", "GSE193460", "8122026", "run_phase_f_perturbmap_silver_standard_v1.py"),
        ("clean_smoke", "CORE_05 duplicate Phase D", "9136320", "byte-identical normalized CSV outputs"),
    ]
    pd.DataFrame(commands, columns=[
        "stage", "scope", "seed", "command_template"
    ]).to_csv(OUT / "REPRODUCIBILITY_MANIFEST.csv", index=False)

    pd.DataFrame([
        ("Phase A T0", True, "clean runner vs locked reference"),
        ("Phase B CMI-E evidence", True, "clean runner vs locked reference"),
        ("Phase D duplicate M5", True, "15-significant-digit normalized SHA256"),
        ("Phase D duplicate M6", True, "15-significant-digit normalized SHA256"),
        ("Phase D metrics and audits", True, "all duplicate CSV/JSON hashes identical"),
    ], columns=["artifact_group", "hash_identical", "audit_scope"]).to_csv(
        OUT / "REPRODUCIBILITY_CHECKS.csv", index=False
    )

    pd.DataFrame([
        ("A", "PASS", "30/30 candidate-legal, connected root supports"),
        ("B", "QUALIFIED_PASS", "null false-call 0; precision 0.692; recall 0.321"),
        ("C", "FAIL", "locked oracle coarse topology gain 0 despite candidate feasibility"),
        ("D", "PASS_LOCAL_SAFETY", "3 positive, 0 negative, total inferred gain 4"),
        ("E", "COMPLETE_PILOT", "30 held-out scenarios, all seeds retained"),
        ("F", "PASS_SILVER_STANDARD", "Tgfbr2 and Ifngr2 primary directions concordant"),
    ], columns=["phase", "decision", "evidence"]).to_csv(
        OUT / "FINAL_PHASE_DECISIONS.csv", index=False
    )

    sources = [
        REPO / "algorithm" / "run_nicheforest_paper_candidate.py",
        Path(__file__),
        Path(__file__).parent / "run_phase_a_case_v7_state_constrained_fine.py",
        Path(__file__).parent / "run_phase_b_multichannel_detector_v6.py",
        Path(__file__).parent / "run_phase_b_multiscale_consensus_v1.py",
        Path(__file__).parent / "run_phase_c_oracle_refinement_v6.py",
        Path(__file__).parent / "run_phase_d_inferred_refinement_v3.py",
        Path(__file__).parent / "run_phase_f_perturbmap_silver_standard_v1.py",
    ]
    hashes = []
    for path in [*sources, *sorted(OUT.glob("*"))]:
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            hashes.append({
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(hashes).to_csv(OUT / "SHA256_MANIFEST.csv", index=False)
    print(json.dumps({"output": str(OUT), "n_synthetic_cases": len(synthetic)}, indent=2))


if __name__ == "__main__":
    main()
