param(
    [int]$LargeFileThresholdMB = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Title)
    Write-Output ""
    Write-Output "== $Title =="
}

function Get-TrackedFiles {
    return @(& git ls-files | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

$tracked = Get-TrackedFiles
$failures = @()

Write-Section "Git Status"
git status --short --branch

Write-Section "Tracked File Count"
Write-Output $tracked.Count

Write-Section "Tracked Large Files"
$largeFiles = @()
foreach ($path in $tracked) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ge ($LargeFileThresholdMB * 1MB)) {
            $largeFiles += [pscustomobject]@{
                Path = $path
                SizeMB = [Math]::Round($item.Length / 1MB, 3)
            }
        }
    }
}

if ($largeFiles.Count -eq 0) {
    Write-Output "No tracked files >= $LargeFileThresholdMB MB."
}
else {
    $largeFiles | Sort-Object SizeMB -Descending | Format-Table -AutoSize
}

Write-Section "Tracked Secret-Risk Filenames"
$secretPatterns = @(
    "\.env$",
    "kaggle\.json$",
    "(^|/)passwords$",
    "\.pem$",
    "\.key$",
    "id_rsa"
)
$secretRisk = @()
foreach ($path in $tracked) {
    $normalized = $path -replace "\\", "/"
    if ($normalized.EndsWith(".env.example")) {
        continue
    }
    if ($normalized -eq "services/scripts/generate_mqtt_password.sh") {
        continue
    }
    foreach ($pattern in $secretPatterns) {
        if ($normalized -match $pattern) {
            $secretRisk += $normalized
            break
        }
    }
}

if ($secretRisk.Count -eq 0) {
    Write-Output "No tracked secret-risk filenames detected."
}
else {
    $secretRisk | Sort-Object -Unique
    $failures += "Tracked secret-risk filenames detected."
}

Write-Section "Required Docs"
$requiredDocs = @(
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CITATION.cff",
    "docs/README.md",
    "docs/03_dataset.md",
    "docs/artifacts.md",
    "docs/security_publication_checklist.md",
    "data/README.md",
    "external_artifacts/README.md"
)

$missingDocs = @($requiredDocs | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missingDocs.Count -eq 0) {
    Write-Output "Required docs exist."
}
else {
    $missingDocs
    $failures += "Required docs missing."
}

Write-Section "Placeholder Check"
$readme = Get-Content -LiteralPath "README.md" -Raw
if ($readme -notmatch 'Kaggle dataset:\s*`COMING_SOON`') {
    $failures += "README missing Kaggle COMING_SOON placeholder."
}
if ($readme -notmatch 'External artifacts archive:\s*`COMING_SOON`') {
    $failures += "README missing artifacts COMING_SOON placeholder."
}
if ($failures -notcontains "README missing Kaggle COMING_SOON placeholder." -and $failures -notcontains "README missing artifacts COMING_SOON placeholder.") {
    Write-Output "README placeholders are present."
}

Write-Section "CI Workflow Check"
$workflowPaths = @(
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/docker-smoke.yml"
)
foreach ($workflow in $workflowPaths) {
    if (Test-Path -LiteralPath $workflow -PathType Leaf) {
        Write-Output "Found $workflow"
    }
    else {
        Write-Output "Missing optional/required workflow: $workflow"
        if ($workflow -eq ".github/workflows/ci.yml") {
            $failures += "CI workflow missing."
        }
    }
}

Write-Section "Tracked Heavy Artifact Extensions"
$heavyExtensions = @(".npz", ".npy", ".pth", ".pt", ".pkl", ".pickle", ".joblib", ".parquet", ".csv", ".pcap", ".zip")
$allowedTracked = @(
    "data/cic-iot-2023/demo_subsets/ddos_burst.parquet",
    "data/cic-iot-2023/demo_subsets/dos_slow.parquet",
    "data/cic-iot-2023/demo_subsets/mirai_wave.parquet",
    "data/cic-iot-2023/demo_subsets/mixed_chaos.parquet",
    "data/cic-iot-2023/demo_subsets/normal_traffic.parquet",
    "data/cic-iot-2023/demo_subsets/recon_scan.parquet"
)
$heavyOffenders = @()
foreach ($path in $tracked) {
    $normalized = $path -replace "\\", "/"
    $extension = [System.IO.Path]::GetExtension($normalized).ToLowerInvariant()
    if ($heavyExtensions -contains $extension -and $allowedTracked -notcontains $normalized) {
        $heavyOffenders += $normalized
    }
}

if ($heavyOffenders.Count -eq 0) {
    Write-Output "No unapproved tracked heavy artifact extensions detected."
}
else {
    $heavyOffenders | Sort-Object
    $failures += "Unapproved tracked heavy artifacts detected."
}

Write-Section "Result"
if ($failures.Count -eq 0) {
    Write-Output "Public release checks passed."
}
else {
    $failures | Sort-Object -Unique
    exit 1
}
