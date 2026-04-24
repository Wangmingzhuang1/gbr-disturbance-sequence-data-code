@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
if errorlevel 1 (
    echo FAILED: cannot locate project root from %SCRIPT_DIR%
    exit /b 1
)

set "ROOT_DIR=%CD%"
set "PYTHON_EXE=python"
if exist "%ROOT_DIR%\venv_gam\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\venv_gam\Scripts\python.exe"
)

set "LOG_FILE=%ROOT_DIR%\scripts\pipeline_log.txt"
echo Running canonical pipeline... > "%LOG_FILE%"
echo Project root: %ROOT_DIR% >> "%LOG_FILE%"
echo Python: %PYTHON_EXE% >> "%LOG_FILE%"

call :run_step 02_build_matrix.py || goto :pipeline_failed
call :run_step 03_merge_eco_dist.py || goto :pipeline_failed
call :run_step 05_analyze_succession.py || goto :pipeline_failed
call :run_step 08_feature_engineering.py || goto :pipeline_failed
call :run_step 06_run_gee_model_main.py || goto :pipeline_failed
call :run_step 07_physical_cooling_proof.py || goto :pipeline_failed
call :run_step 11_spatial_autocorrelation.py || goto :pipeline_failed
call :run_step 13_bootstrap_ci.py || goto :pipeline_failed
call :run_step 12_sensitivity_analysis.py || goto :pipeline_failed
call :run_step 17_targeted_robustness.py || goto :pipeline_failed
call :run_step 18_fate_divergence.py || goto :pipeline_failed
call :run_step 14_auxiliary_ecology.py || goto :pipeline_failed
call :run_step 15_auxiliary_juveniles.py || goto :pipeline_failed
call :run_step 16_model_diagnostics.py || goto :pipeline_failed
call :run_step 19_extended_sequence_inference.py || goto :pipeline_failed
call :run_step 09_analysis_combined.py || goto :pipeline_failed

call :run_step visualizations\figure0_map.py || goto :pipeline_failed
call :run_step visualizations\plot_fig1_hierarchy.py || goto :pipeline_failed
call :run_step visualizations\plot_fig2_adjusted_means.py || goto :pipeline_failed
call :run_step visualizations\plot_fig3_fate_divergence.py || goto :pipeline_failed
call :run_step visualizations\plot_fig4_robustness.py || goto :pipeline_failed
call :run_step visualizations\plot_fig5_supporting_evidence.py || goto :pipeline_failed
call :run_step visualizations\plot_figS1_gap_memory.py || goto :pipeline_failed
call :run_step visualizations\plot_figS2_juvenile_evidence.py || goto :pipeline_failed
call :run_step visualizations\plot_figS3_ols_diagnostics.py || goto :pipeline_failed
call :run_step visualizations\plot_figS4_baseline_state_gradients.py || goto :pipeline_failed
call :run_step visualizations\plot_figS5_ols_coefficients.py || goto :pipeline_failed
call :run_step visualizations\plot_figS6_extraction_flowchart.py || goto :pipeline_failed

echo Pipeline completed successfully. >> "%LOG_FILE%"
echo Pipeline completed successfully.
popd >nul
exit /b 0

:pipeline_failed
echo Pipeline failed. See "%LOG_FILE%".
popd >nul
exit /b 1

:run_step
echo Running %~1...
echo ===== RUN %~1 ===== >> "%LOG_FILE%"
"%PYTHON_EXE%" "%ROOT_DIR%\scripts\%~1" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo FAILED: %~1 >> "%LOG_FILE%"
    echo FAILED: %~1
    exit /b 1
)
exit /b 0
