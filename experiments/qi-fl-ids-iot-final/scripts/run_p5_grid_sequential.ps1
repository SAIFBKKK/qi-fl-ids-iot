param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = "experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml",
    [int]$Rounds = 30,
    [switch]$ContinueOnError,
    [switch]$SkipTests,
    [switch]$SkipExisting,
    [switch]$OnlyAggregate,
    [switch]$VerboseLogs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

$ReportsDir = "experiments/qi-fl-ids-iot-final/outputs/reports"
$LogPath = Join-Path $ReportsDir "p5_grid_run_log.txt"
$StatusPath = Join-Path $ReportsDir "p5_grid_status.json"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null

$Alphas = @(0.1, 0.5, 5.0)
$ClientCounts = @(3, 4, 5)
$ScenarioStatuses = @()

function Write-GridLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) | $Message"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line
}

function Save-GridStatus {
    $ScenarioStatuses | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Get-AlphaDir {
    param([double]$Alpha)
    if ([math]::Abs($Alpha - [math]::Round($Alpha)) -lt 0.0000001) {
        return ("alpha_{0:F1}" -f $Alpha)
    }
    return "alpha_$Alpha"
}

function Get-ScenarioDir {
    param([double]$Alpha, [int]$Clients)
    $alphaDir = Get-AlphaDir -Alpha $Alpha
    return "experiments/qi-fl-ids-iot-final/outputs/fl_l1_fedavg/$alphaDir/k$Clients"
}

function Test-ScenarioArtifacts {
    param([double]$Alpha, [int]$Clients)
    $scenarioDir = Get-ScenarioDir -Alpha $Alpha -Clients $Clients
    $required = @(
        "artifacts/run_summary.json",
        "artifacts/metrics_rounds.csv",
        "artifacts/metrics_clients.csv",
        "artifacts/bandwidth_rounds.csv",
        "artifacts/metrics_test.json",
        "artifacts/comparison_with_p4.json"
    )
    $missing = @()
    foreach ($relative in $required) {
        $path = Join-Path $scenarioDir $relative
        if (-not (Test-Path $path)) {
            $missing += $relative
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing required artifacts for alpha=$Alpha K=${Clients}: $($missing -join ', ')"
    }
    return $scenarioDir
}

function Test-ScenarioComplete {
    param([double]$Alpha, [int]$Clients, [int]$ExpectedRounds)
    try {
        $scenarioDir = Test-ScenarioArtifacts -Alpha $Alpha -Clients $Clients
        $summaryPath = Join-Path $scenarioDir "artifacts/run_summary.json"
        $summary = Get-Content $summaryPath | ConvertFrom-Json
        return (($summary.mode -eq "full" -or $summary.mode -eq "grid") -and [int]$summary.rounds -ge $ExpectedRounds)
    }
    catch {
        return $false
    }
}

function Invoke-GridCommand {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-GridLog "START $Name"
    Write-GridLog "COMMAND $Exe $($Arguments -join ' ')"
    $start = Get-Date
    $output = & $Exe @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) {
        if ($VerboseLogs -or $exitCode -ne 0) {
            Write-GridLog "OUTPUT $Name | $line"
        }
        else {
            Add-Content -Path $LogPath -Value "$(Get-Date -Format o) | OUTPUT $Name | $line"
        }
    }
    $duration = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)
    if ($exitCode -ne 0) {
        Write-GridLog "FAILED $Name duration_sec=$duration exit_code=$exitCode"
        throw "$Name failed with exit code $exitCode"
    }
    Write-GridLog "SUCCESS $Name duration_sec=$duration"
    return $duration
}

function Add-ScenarioStatus {
    param(
        [double]$Alpha,
        [int]$Clients,
        [string]$Status,
        [double]$DurationSec,
        [string]$OutputDir,
        [string]$ErrorMessage = ""
    )
    $script:ScenarioStatuses += [PSCustomObject]@{
        alpha = $Alpha
        K = $Clients
        status = $Status
        duration_sec = $DurationSec
        output_dir = $OutputDir
        error_message = $ErrorMessage
    }
    Save-GridStatus
}

function Invoke-SafeStep {
    param(
        [string]$Name,
        [scriptblock]$Step
    )
    try {
        & $Step
    }
    catch {
        Write-GridLog "FAILED $Name error=$($_.Exception.Message)"
        if (-not $ContinueOnError) {
            throw
        }
    }
}

Write-GridLog "P5.3 sequential grid runner started"
Write-GridLog "RepoRoot=$RepoRoot"
Write-GridLog "ConfigPath=$ConfigPath Rounds=$Rounds ContinueOnError=$ContinueOnError SkipTests=$SkipTests SkipExisting=$SkipExisting OnlyAggregate=$OnlyAggregate"

if (-not (Test-Path $ConfigPath)) {
    throw "ConfigPath not found: $ConfigPath"
}

if (-not $OnlyAggregate) {
    Invoke-SafeStep -Name "py_compile P5" -Step {
        Invoke-GridCommand -Name "py_compile P5" -Exe $PythonExe -Arguments @(
            "-m", "py_compile",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/aggregation.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/client_data.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/fedavg_client.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/fedavg_server.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/communication.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/evaluation.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/round_logger.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/report_builder.py",
            "experiments/qi-fl-ids-iot-final/src/fl_l1/verify_setup.py",
            "experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py",
            "experiments/qi-fl-ids-iot-final/src/scripts/05_3_aggregate_fl_grid_results.py"
        ) | Out-Null
    }

    if (-not $SkipTests) {
        Invoke-SafeStep -Name "pytest P5 unit aggregation" -Step {
            Invoke-GridCommand -Name "pytest P5 unit aggregation" -Exe $PythonExe -Arguments @(
                "-m", "pytest", "experiments/qi-fl-ids-iot-final/tests/unit/test_fedavg_aggregation.py", "-q"
            ) | Out-Null
        }
        Invoke-SafeStep -Name "pytest P5 unit communication" -Step {
            Invoke-GridCommand -Name "pytest P5 unit communication" -Exe $PythonExe -Arguments @(
                "-m", "pytest", "experiments/qi-fl-ids-iot-final/tests/unit/test_fl_l1_communication.py", "-q"
            ) | Out-Null
        }
        Invoke-SafeStep -Name "pytest P5 integration setup" -Step {
            Invoke-GridCommand -Name "pytest P5 integration setup" -Exe $PythonExe -Arguments @(
                "-m", "pytest", "experiments/qi-fl-ids-iot-final/tests/integration/test_fl_l1_setup.py", "-q"
            ) | Out-Null
        }
    }

    Invoke-SafeStep -Name "verify P5 setup" -Step {
        Invoke-GridCommand -Name "verify P5 setup" -Exe $PythonExe -Arguments @(
            "experiments/qi-fl-ids-iot-final/src/scripts/05_verify_fl_l1_setup.py",
            "--config", $ConfigPath
        ) | Out-Null
    }

    foreach ($alpha in $Alphas) {
        foreach ($clients in $ClientCounts) {
            $scenarioName = "scenario alpha=$alpha K=$clients"
            $scenarioDir = Get-ScenarioDir -Alpha $alpha -Clients $clients
            if ($SkipExisting -and (Test-ScenarioComplete -Alpha $alpha -Clients $clients -ExpectedRounds $Rounds)) {
                Write-GridLog "SKIPPED $scenarioName existing_valid_output=$scenarioDir"
                Add-ScenarioStatus -Alpha $alpha -Clients $clients -Status "skipped_existing" -DurationSec 0 -OutputDir $scenarioDir
                continue
            }
            try {
                $duration = Invoke-GridCommand -Name $scenarioName -Exe $PythonExe -Arguments @(
                    "experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py",
                    "--config", $ConfigPath,
                    "--mode", "full",
                    "--alpha", "$alpha",
                    "--clients", "$clients",
                    "--rounds", "$Rounds"
                )
                $validatedDir = Test-ScenarioArtifacts -Alpha $alpha -Clients $clients
                Write-GridLog "VALIDATED $scenarioName artifacts=$validatedDir"
                Add-ScenarioStatus -Alpha $alpha -Clients $clients -Status "success" -DurationSec $duration -OutputDir $validatedDir
            }
            catch {
                Add-ScenarioStatus -Alpha $alpha -Clients $clients -Status "failed" -DurationSec 0 -OutputDir $scenarioDir -ErrorMessage $_.Exception.Message
                Write-GridLog "FAILED $scenarioName error=$($_.Exception.Message)"
                if (-not $ContinueOnError) {
                    throw
                }
            }
        }
    }
}
else {
    Write-GridLog "OnlyAggregate enabled: training and pre-checks skipped"
}

Invoke-SafeStep -Name "aggregate grid results" -Step {
    $aggregateArgs = @(
        "experiments/qi-fl-ids-iot-final/src/scripts/05_3_aggregate_fl_grid_results.py",
        "--config", $ConfigPath,
        "--rounds", "$Rounds"
    )
    if ($ContinueOnError) {
        $aggregateArgs += "--allow-missing"
    }
    Invoke-GridCommand -Name "aggregate grid results" -Exe $PythonExe -Arguments $aggregateArgs | Out-Null
}

Write-GridLog "P5.3 sequential grid runner finished"
Write-GridLog "Status JSON: $StatusPath"
