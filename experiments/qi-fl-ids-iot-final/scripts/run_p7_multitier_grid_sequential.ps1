param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = "experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml",
    [int]$Rounds = 30,
    [switch]$ContinueOnError,
    [switch]$SkipExisting,
    [switch]$OnlyAggregate
)

$ErrorActionPreference = "Stop"
$LogPath = "experiments/qi-fl-ids-iot-final/outputs/reports/p7_multitier_grid_run_log.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
"P7 grid started $(Get-Date -Format o)" | Set-Content $LogPath

function Invoke-Step {
    param([string]$Name, [string]$Command)
    $start = Get-Date
    Write-Host "[$Name] $Command"
    "[$Name] $Command" | Add-Content $LogPath
    try {
        Invoke-Expression $Command
        $duration = ((Get-Date) - $start).TotalSeconds
        "[$Name] SUCCESS duration_sec=$duration" | Add-Content $LogPath
    } catch {
        $duration = ((Get-Date) - $start).TotalSeconds
        "[$Name] FAILED duration_sec=$duration error=$($_.Exception.Message)" | Add-Content $LogPath
        if (-not $ContinueOnError) { throw }
    }
}

if (-not $OnlyAggregate) {
    Invoke-Step "verify" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/07_verify_multitier_heterofl_setup.py --config $ConfigPath"
    foreach ($task in @("l1", "l2")) {
        foreach ($alpha in @(0.1, 0.5, 5.0)) {
            foreach ($k in @(3, 4, 5)) {
                $scenarioDir = "experiments/qi-fl-ids-iot-final/outputs/multitier_heterofl/${task}_binary/alpha_$alpha/k$k"
                if ($task -eq "l2") { $scenarioDir = "experiments/qi-fl-ids-iot-final/outputs/multitier_heterofl/l2_family/alpha_$alpha/k$k" }
                if ($SkipExisting -and (Test-Path "$scenarioDir/latest_run_summary.json")) {
                    "[$task alpha=$alpha k=$k] SKIPPED existing" | Add-Content $LogPath
                    continue
                }
                Invoke-Step "$task alpha=$alpha k=$k" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/07_run_multitier_heterofl.py --config $ConfigPath --task $task --mode full --alpha $alpha --clients $k --rounds $Rounds"
            }
        }
    }
}

Invoke-Step "aggregate" "$PythonExe experiments/qi-fl-ids-iot-final/src/scripts/07_aggregate_multitier_results.py"
Write-Host "P7 grid finished. Log: $LogPath"
