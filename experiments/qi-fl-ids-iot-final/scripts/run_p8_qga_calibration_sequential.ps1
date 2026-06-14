param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = "experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml",
    [switch]$OnlyReport
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Command
    )
    Write-Host "=== $Name ==="
    Write-Host $Command
    Invoke-Expression $Command
}

if (-not $OnlyReport) {
    Invoke-Step "QGA profile sweep" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_run_qga_profile_sweep.py --config $ConfigPath"
    Invoke-Step "QGA Flower short validation" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_run_qga_flower_short_validation.py --config $ConfigPath --top-n 3 --rounds 5 --max-samples-per-client 1000"
    Invoke-Step "Select best mask" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_select_best_qga_mask.py --config $ConfigPath"
}

Invoke-Step "Build calibration report" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_build_qga_calibration_report.py --config $ConfigPath"
