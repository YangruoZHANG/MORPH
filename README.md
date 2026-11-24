# MORPH

Private manuscript baseline for the multi-origin, root-conditioned trajectory forest study. This snapshot preserves the approved figures and their actual method versions. The baseline tag identifies the submission materials in this repository and is separate from the algorithm version number.

| Directory | Contents |
| --- | --- |
| [algorithm/](algorithm/) | Frozen breast v70 construction code, analysis and figure scripts, and their versioned dependencies |
| [modules/candidate_contacts/](modules/candidate_contacts/) | Descriptive cross-tree contact kernel, run entry point and archived comparison tables |
| [benchmark/](benchmark/) | Supplied-root branch/state benchmark code, 1,000 simulation metric records and one complete synthetic scoring fixture |
| [results/](results/) | Frozen breast state and edge tables, biological programme results and seven-dataset supplementary tables |
| [figures/](figures/) | Approved manuscript and supplementary artwork. Large originals are stored in [archive/](archive/). |

## Version and evidence

- **Breast trajectories and programmes.** Fig. 3, Fig. 4 and Fig. 6 use the frozen v70 count-aware, identity-conditioned regional multipath state forest. Its constructor is `algorithm/validation/rebuild_regional_source_ensembles_v63_20261025.py`. Set `NICHEFOREST_COUNT_SPATIAL_AGREEMENT=1` and use `--mode ensemble` with the exact inputs. Older numbers in dependency filenames identify the original components.
- **Cross-tree contacts.** Fig. 4 displays CXCL12–CXCR4 and CD274–PDCD1 as descriptive candidates from the frozen analysis. The archived comparison contains 7,889 candidates. The figure provides contact localization and does not establish a formal CMI-E call, a causal effect or a tree refinement.
- **Synthetic benchmark.** Fig. 5 uses the separate 1,000-dataset supplied-root branch/state benchmark. See [metric provenance](benchmark/METRIC_PROVENANCE.md).
- **Supplementary trajectories.** Figures B and C use the approved seven-dataset exports and 20 selected roots from the historical pipeline.
- **Workflow and root selection.** Fig. 1 and Fig. 2 preserve the accepted artwork. Some labels in the original workflow predate the frozen execution path. The exact collaborator implementation used to produce the AMCP artwork has yet to be recovered.

## Run the included benchmark checks

The commands were tested on Windows with Python 3.13.9. Create and activate an isolated environment first.

```sh
python -m pip install -r requirements.txt
python benchmark/reproduce.py summary
python benchmark/reproduce.py generate --limit 1
python benchmark/reproduce.py scores --limit 1
```

The first check reaggregates the archived 1,000-dataset results. The second regenerates the synthetic fixture and checks its count matrix and spatial coordinates. The third recomputes eight metrics from the fixture's truth and archived predictions. Full benchmark inference and real-data reconstruction require additional files. See [data availability](DATA_AVAILABILITY.md) and the [exact input manifest](external_inputs.csv).

[Source provenance](SOURCE_PROVENANCE.csv) maps copied files to their source paths and hashes. [SHA-256 inventory](SHA256SUMS.txt) covers the versioned payload. The author order in [software citation metadata](CITATION.cff) follows the manuscript. Public release terms and the final persistent identifier are pending.

## Restore the original large files

Run `python restore_artifacts.py` from the repository root. It joins the parts and checks the original SHA-256 for each archived file. The output under `restored_artifacts/` contains the exact original PNG, PDF and CSV bytes at their documented relative paths. The small files and remaining figure originals are available directly. See [archive manifest](archive/manifest.json).
