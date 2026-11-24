# MORPH

MORPH reconstructs root-conditioned trajectory forests from spatial transcriptomics. This repository contains the manuscript's method, figure sources and reproducibility materials.

## Repository contents

| Location | Contents |
| --- | --- |
| [algorithm/forest/](algorithm/forest/) | MORPH construction, including source ensembles, count–spatial candidate costs, and multipath state ordering |
| [algorithm/validation/](algorithm/validation/) | Reconstruction verification and biological programme tests |
| [algorithm/visualization/](algorithm/visualization/) | Plotting scripts for the main and supplementary figures |
| [modules/candidate_contacts/](modules/candidate_contacts/) | Descriptive cross-tree contact analysis and comparison tables |
| [benchmark/](benchmark/) | Supplied-root branch/state benchmark, simulation, scoring and 1,000 per-dataset records |
| [results/](results/) | Breast state and parent-edge tables, biological programme results, and seven-dataset supplementary exports |
| [figures/](figures/) | Approved artwork and directions for restoring the large originals |

## Install and run

Use Python 3.13.9 in an isolated environment.

```sh
python -m pip install -r requirements.txt
python algorithm/forest/run_morph.py
```

The MORPH command selects the manuscript's count-aware construction and writes to `outputs/morph_breast_forest/ensemble/`. Place the exact Xenium breast inputs at the paths in [external_inputs.csv](external_inputs.csv) first. See [data access](DATA_AVAILABILITY.md) for the source and availability of each input. The original AMCP root-selection implementation used for the approved artwork remains unavailable.

Run the self-contained benchmark checks with the supplied synthetic data and predictions.

```sh
python benchmark/reproduce.py summary
python benchmark/reproduce.py generate --limit 1
python benchmark/reproduce.py scores --limit 1
```

The figure benchmark uses its original supplied-root branch/state pipeline. Supplementary figures B and C come from the approved seven-dataset analyses with 20 selected roots. Figure 4 localizes CXCL12–CXCR4 and CD274–PDCD1 as descriptive contact candidates from the breast analysis.

## Restore and cite

Run `python restore_artifacts.py` to verify and restore the large figure and table originals to `restored_artifacts/`. [The archive manifest](archive/manifest.json) records each original file's SHA-256. [The repository checksum list](SHA256SUMS.txt) covers this checkout, and [source provenance](SOURCE_PROVENANCE.csv) records the baseline commit and underlying Git objects. See [CITATION.cff](CITATION.cff) for authorship.
## Trajectory evaluation

The [trajectory evaluation module](benchmark/trajectory_evaluation/) contains the final supplied six-metric scoring implementation, nine-method result tables and publication figures for 25 trajectory datasets and five spatial null controls. Its seven unit tests and stored-result validation passed. Raw prediction inputs are required to repeat full scoring.
