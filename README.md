# Data and code for the GBR recurrent heat-stress study

This repository provides the data and analysis code needed to reproduce the results reported in the manuscript:

**Recurrent heat stress reveals baseline-limited observation of coral loss on the Great Barrier Reef**

The repository is only for data and code availability. It does not include manuscript drafts, cover letters, reference PDFs, journal-preparation notes, or large non-CSV spatial assets.

## Repository structure

- `data/`: CSV inputs used by the reproducibility pipeline.
- `output/legacy_load_analysis_matrix.csv`: derived disturbance-response matrix used by the main models and figures.
- `output/legacy_load_enriched_eco.csv`: derived ecological-covariate matrix retained for inspection.
- `output/tables/`: machine-readable supplementary tables S1-S10.
- `output/figures/`: generated figure output directory. Figure files are not versioned and are recreated by the scripts.
- `scripts/`: canonical analysis, sensitivity, audit and figure-generation scripts.
- `PROJECT_PIPELINE.md`: pipeline order and expected outputs.
- `DATA_AVAILABILITY.md`: source-data provenance and links for excluded large assets.

## Reproduce the analysis

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

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

- `scripts/04_build_legacy_load_features.py` reconstructs `output/legacy_load_analysis_matrix.csv`.
- `scripts/05_run_lmm_legacy_model.py` reproduces the OLS, GEE, sensitivity models, VIF diagnostics and supplementary tables.
- `scripts/06_generate_figures.py` reproduces Figures 2-6 and SI Figure 2.
- `scripts/07_generate_map_figure.py` reproduces Figure 1 and SI Figure 1. Non-CSV spatial boundary files are optional and are not included here; source links are listed in `DATA_AVAILABILITY.md`.
- `scripts/verify_contamination.py` and `scripts/verify_text_quality.py` are audit helpers used during manuscript checking.

## Large or external assets

Large non-CSV spatial files, reference PDFs and journal-preparation materials are excluded. Public source websites for ecological monitoring, climate products, cyclone records and spatial boundaries are provided in `DATA_AVAILABILITY.md`.
