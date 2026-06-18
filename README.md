# Data and code for the GBR recurrent disturbance study

This repository provides the processed data and analysis code needed to reproduce the results reported in the manuscript:

**Thermal recurrence exposes baseline-constrained signals of coral-cover loss on the Great Barrier Reef**

The repository is for data and code availability. It does not include manuscript drafts, cover letters, reference PDFs, journal-preparation notes or large non-CSV spatial assets.

## Repository structure

- `data/`: processed ecological, disturbance and reef-coordinate input files used by the reproducibility pipeline.
- `output/legacy_load_analysis_matrix.csv`: model-ready disturbance-response matrix used by the main models and figures.
- `output/legacy_load_enriched_eco.csv`: ecological-covariate matrix used for supplementary sensitivity checks.
- `output/tables/`: machine-readable source tables for Supplementary Tables S1-S10.
- `output/figures/`: generated figure output directory. Figure binaries are not versioned and are recreated by the scripts.
- `scripts/`: canonical data-processing, statistical-analysis, audit and figure-generation scripts.
- `PROJECT_PIPELINE.md`: pipeline order and expected outputs.
- `DATA_AVAILABILITY.md`: data provenance and public source-data links for excluded large assets.

## Environment

Use Python 3.10 or later. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Reproduce the analysis

Run the full reproducibility workflow from the repository root:

```bash
python scripts/04_build_legacy_load_features.py
python scripts/05_run_lmm_legacy_model.py
python scripts/06_generate_figures.py
python scripts/07_generate_map_figure.py
```

On Windows, the same workflow can be run with:

```bat
scripts\run_pipeline.bat
```

## Key reproducibility notes

- `scripts/04_build_legacy_load_features.py` reconstructs `output/legacy_load_analysis_matrix.csv` and `output/legacy_load_enriched_eco.csv`.
- `scripts/05_run_lmm_legacy_model.py` reproduces the OLS, GEE, fixed-effect checks, episode-level checks, proportional-retention sensitivity analyses, ecological-covariate sensitivity analyses, VIF diagnostics and machine-readable supplementary tables.
- `scripts/06_generate_figures.py` reproduces Figures 2-5 and SI Figures 2-3.
- `scripts/07_generate_map_figure.py` reproduces Figure 1 and SI Figure 1. Non-CSV spatial boundary files are optional and are not included here; source links are listed in `DATA_AVAILABILITY.md`.
- `scripts/verify_contamination.py` and `scripts/verify_text_quality.py` are audit helpers used during manuscript checking.

## Large or external assets

Large non-CSV spatial files, reference PDFs and journal-preparation materials are excluded. Public source websites for ecological monitoring, climate products, cyclone records and spatial boundaries are provided in `DATA_AVAILABILITY.md`. Generated figure binaries can be recreated from the deposited processed data and scripts.
