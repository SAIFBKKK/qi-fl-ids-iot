# Validate Mode A runtime for QI-FL-IDS-IoT.
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

function Invoke-Compose {
    param([string[]]$Arguments)
    & docker compose --env-file $EnvFile -f $ComposeFile @Arguments
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

function Get-HttpText {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return [string]$response.Content
    }
    catch {
        return $null
    }
}

function Check-HttpContains {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Pattern
    )
    $body = Get-HttpText $Url
    if ($null -ne $body -and $body -match $Pattern) {
        Pass $Name
    }
    else {
        Fail "$Name ($Url)"
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

Write-Host "=== QI-FL-IDS-IoT Mode A demo check ==="
Write-Host "Repo root: $RepoRoot"

Test-CommandAvailable "docker" | Out-Null

& docker compose --env-file $EnvExample -f $ComposeFile config --quiet *> $null
if ($LASTEXITCODE -eq 0) {
    Pass "docker compose config"
}
else {
    Fail "docker compose config"
}

if (-not (Test-Path $EnvFile)) {
    Fail "Create services/.env from .env.example first"
    Write-Host ""
    Write-Host "Summary: $Failures failure(s)"
    exit 1
}

$expectedServices = @("mosquitto", "iot-node-1", "traffic-generator", "prometheus", "grafana", "mlflow")
$runningServices = @()
try {
    $runningServices = Invoke-Compose @("ps", "--services", "--filter", "status=running") 2>$null
}
catch {
    $runningServices = @()
}

foreach ($service in $expectedServices) {
    if ($runningServices -contains $service) {
        Pass "container running: $service"
    }
    else {
        Fail "container not running: $service"
    }
}

Check-HttpContains "iot-node health" "http://localhost:8001/health" '"status"\s*:\s*"ok"'
Check-HttpContains "traffic-generator health" "http://localhost:8010/health" '"status"\s*:\s*"ok"'
Check-HttpOk "prometheus ready" "http://localhost:9090/-/ready"
Check-HttpContains "grafana health" "http://localhost:3000/api/health" '"database"\s*:\s*"ok"'
Check-HttpOk "mlflow" "http://localhost:5000"
Check-HttpContains "iot-node metrics" "http://localhost:8001/metrics" "ids_flows_received_total"
Check-HttpContains "traffic-generator metrics" "http://localhost:8010/metrics" "traffic_generator_flows_published_total"

Write-Host ""
if ($Failures -eq 0) {
    Write-Host "Mode A demo check: PASS"
    exit 0
}

Write-Host "Mode A demo check: FAIL ($Failures failure(s))"
exit 1
