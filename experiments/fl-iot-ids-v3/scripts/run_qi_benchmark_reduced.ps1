param(
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReportDir = Join-Path $Root "outputs/reports/qi_benchmark_reduced"
$LogDir = Join-Path $ReportDir "logs"
$StatusPath = Join-Path $ReportDir "run_status.csv"
$BaselineDir = Join-Path $Root "outputs/reports/baselines"

New-Item -ItemType Directory -Force -Path $ReportDir, $LogDir | Out-Null

$Experiments = @(
    @{ Id = "E1"; Name = "exp_bench30_normal_fedavg_28f" },
    @{ Id = "E2"; Name = "exp_bench30_normal_qifa_28f" },
    @{ Id = "E3"; Name = "exp_bench30_normal_fedavg_qga15" },
    @{ Id = "E4"; Name = "exp_bench30_normal_qifa_qga15" },
    @{ Id = "E5"; Name = "exp_bench30_absent_fedavg_28f" },
    @{ Id = "E6"; Name = "exp_bench30_absent_qifa_28f" },
    @{ Id = "E7"; Name = "exp_bench30_absent_fedavg_qga15" },
    @{ Id = "E8"; Name = "exp_bench30_absent_qifa_qga15" }
)

function Get-RunSummary($ExperimentName) {
    $summaryPath = Join-Path $BaselineDir "$ExperimentName/run_summary.json"
    if (-not (Test-Path $summaryPath)) {
        return $null
    }
    return Get-Content $summaryPath -Raw | ConvertFrom-Json
}

function Is-Reusable30($Summary) {
    if ($null -eq $Summary) {
        return $false
    }
    return ($Summary.status -eq "success" -and [int]$Summary.completed_rounds -ge 30)
}

$Rows = New-Object System.Collections.Generic.List[object]
Set-Location $Root

foreach ($Experiment in $Experiments) {
    $name = $Experiment.Name
    $id = $Experiment.Id
    $summaryBefore = Get-RunSummary $name
    if (Is-Reusable30 $summaryBefore) {
        $Rows.Add([PSCustomObject]@{
            experiment_id = $id
            experiment_name = $name
            action = "skipped_reusable"
            status = $summaryBefore.status
            completed_rounds = [int]$summaryBefore.completed_rounds
            exit_code = 0
            log_path = ""
        })
        continue
    }

    $logPath = Join-Path $LogDir "$name.log"
    if ($DryRun) {
        $Rows.Add([PSCustomObject]@{
            experiment_id = $id
            experiment_name = $name
            action = "dry_run_would_execute"
            status = "pending"
            completed_rounds = if ($null -eq $summaryBefore) { 0 } else { [int]$summaryBefore.completed_rounds }
            exit_code = ""
            log_path = $logPath
        })
        continue
    }

    $command = @(
        "-m",
        "src.scripts.run_experiment",
        "--experiment",
        $name
    )
    & python @command *> $logPath
    $exitCode = $LASTEXITCODE
    $summaryAfter = Get-RunSummary $name
    $Rows.Add([PSCustomObject]@{
        experiment_id = $id
        experiment_name = $name
        action = "executed"
        status = if ($null -eq $summaryAfter) { "missing_summary" } else { $summaryAfter.status }
        completed_rounds = if ($null -eq $summaryAfter) { 0 } else { [int]$summaryAfter.completed_rounds }
        exit_code = $exitCode
        log_path = $logPath
    })
}

$Rows | Export-Csv -NoTypeInformation -Path $StatusPath
Write-Host "Reduced QI benchmark status -> $StatusPath"
