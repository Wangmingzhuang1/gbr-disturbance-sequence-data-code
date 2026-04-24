# Data and code for the GBR disturbance-sequence study

This repository is intended only for data and code availability. It contains the scripts and CSV files needed to inspect and reproduce the analyses reported in the manuscript. It does not contain manuscript drafts, cover letters, journal notes, or other submission-management files.

## Structure

- `scripts/`: canonical analysis, audit, modeling, and figure-generation code.
- `scripts/visualizations/`: plotting scripts used to generate manuscript figures.
- `data/`: CSV source data used by the pipeline.
- `output/data/`: derived analysis matrices and retained sequence datasets.
- `output/tables/`: statistical result tables and robustness summaries.
- `output/audits/`: pipeline audit and consistency-check tables.
- `requirements.txt`: Python package requirements.
- `PROJECT_PIPELINE.md`: canonical pipeline order and key output list.
- `DATA_AVAILABILITY.md`: source-data links and notes on excluded non-CSV spatial assets.

## Reproduce from included CSV files

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Run the full Windows pipeline from the repository root with:

```bat
scripts\run_pipeline.bat
```

The runner resolves paths relative to the repository root and writes logs to `scripts/pipeline_log.txt`.

For a quick integrity check, run:

```bash
python scripts/audit_full_pipeline.py
```

## Large or external assets

All CSV data are included directly. Non-CSV spatial boundary files used for map rendering are not included; download links are provided in `DATA_AVAILABILITY.md`.
