@echo off
setlocal
cd /d "%~dp0\.."

python scripts\04_build_legacy_load_features.py
if errorlevel 1 exit /b 1

python scripts\05_run_lmm_legacy_model.py
if errorlevel 1 exit /b 1

python scripts\06_generate_figures.py
if errorlevel 1 exit /b 1

python scripts\07_generate_map_figure.py
if errorlevel 1 exit /b 1

echo Pipeline completed successfully.
