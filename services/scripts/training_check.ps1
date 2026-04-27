# Validate Mode B training profile runtime for QI-FL-IDS-IoT.
$ErrorActionPreference = "Continue"

$ServicesDir = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $ServicesDir
$ComposeFile = Join-Path $ServicesDir "docker-compose.yml"
$EnvExample = Join-Path $ServicesDir ".env.example"
$EnvFile = Join-Path $ServicesDir ".env"
$Failures = 0

function Pass($Message) {
    Write-Host "PASS $Message"
}

function Fail($Message) {
    Write-Host "FAIL $Message"
    $script:Failures += 1
}

function Invoke-TrainingCompose {
    param([string[]]$Arguments)
    & docker compose --env-file $EnvFile -f $ComposeFile --profile training @Arguments
}

function Test-CommandAvailable {
    param([string]$Name)
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Pass "$Name available"
        return $true
    }
    Fail "$Name not available"
    return $false
}

function Get-ContainerState {
    param([string]$Name)
    $output = & docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
        return $null
    }
    $parts = ([string]$output).Trim() -split '\s+'
    return [pscustomobject]@{
        Status = $parts[0]
        ExitCode = if ($parts.Count -gt 1) { $parts[1] } else { "" }
        Raw = ([string]$output).Trim()
    }
}

function Check-Running {
    param([string]$Name)
    $state = Get-ContainerState $Name
    if ($null -ne $state -and $state.Status -eq "running") {
        Pass "container running: $Name"
    }
    else {
        Fail "container not running: $Name"
    }
}

function Check-RunningOrExitedZero {
    param([string]$Name)
    $state = Get-ContainerState $Name
    if ($null -ne $state -and $state.Status -eq "running") {
        Pass "container running: $Name"
    }
    elseif ($null -ne $state -and $state.Status -eq "exited" -and $state.ExitCode -eq "0") {
        Pass "container exited cleanly: $Name"
    }
    else {
        $rawState = if ($null -eq $state) { "missing" } else { $state.Raw }
        Fail "container not healthy: $Name (state=$rawState)"
    }
}

function Check-HttpOk {
    param(
        [string]$Name,
        [string]$Url
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        if ([int]$response.StatusCode -eq 200) {
            Pass $Name
        }
        else {
            Fail "$Name ($Url returned HTTP $($response.StatusCode))"
        }
    }
    catch {
        Fail "$Name ($Url)"
    }
}

Write-Host "=== QI-FL-IDS-IoT Mode B training check ==="
Write-Host "Repo root: $RepoRoot"

if (-not (Test-CommandAvailable "docker")) {
    Write-Host ""
    Write-Host "Mode B training check: FAIL ($Failures failure(s))"
    exit 1
}

& docker compose --env-file $EnvExample -f $ComposeFile --profile training config --quiet *> $null
if ($LASTEXITCODE -eq 0) {
    Pass "docker compose training config"
}
else {
    Fail "docker compose training config"
}

if (-not (Test-Path $EnvFile)) {
    Fail "Create services/.env from .env.example first"
    Write-Host ""
    Write-Host "Mode B training check: FAIL ($Failures failure(s))"
    exit 1
}

Check-Running "fl-server"
Check-RunningOrExitedZero "fl-client-1"
Check-RunningOrExitedZero "fl-client-2"
Check-RunningOrExitedZero "fl-client-3"
Check-HttpOk "mlflow reachable" "http://localhost:5000"

$serverLogs = & docker logs fl-server 2>&1 | Out-String
if ($serverLogs -match "training|round|Flower|FedAvg") {
    Pass "fl-server logs contain training markers"
}
else {
    Fail "fl-server logs contain training markers"
}

Write-Host ""
if ($Failures -eq 0) {
    Write-Host "Mode B training check: PASS"
    exit 0
}

Write-Host "Mode B training check: FAIL ($Failures failure(s))"
exit 1
