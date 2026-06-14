param(
    [ValidateSet("Inventory", "CacheDryRun", "CacheQuarantine", "HeavyInventory", "SecretScan")]
    [string]$Mode = "Inventory",

    [switch]$Confirm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ManifestDir = Join-Path $Root "_cleanup_manifests"
$QuarantineRoot = Join-Path $Root "_cleanup_quarantine\$Timestamp"

New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null

$SkipDirNames = @(".git", ".venv", "venv", "env", "ENV", "node_modules", "_cleanup_quarantine")
$CacheDirNames = @("__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "tmp")
$ProtectedTempExtensions = @(".md", ".rst", ".ipynb", ".pdf", ".docx", ".pptx", ".py", ".ps1", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")
$HeavyExtensions = @(
    ".npz", ".npy", ".parquet", ".csv", ".pth", ".pt", ".pkl", ".pickle",
    ".joblib", ".onnx", ".ckpt", ".zip", ".tar", ".rar", ".gz", ".tgz",
    ".bz2", ".xz", ".7z", ".log"
)
$GeneratedEvidenceExtensions = @(".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html")
$TextScanExtensions = @(".env", ".txt", ".md", ".py", ".ps1", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".log")

function Get-RelativeRepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    try {
        return ([System.IO.Path]::GetRelativePath($Root, $fullPath) -replace "\\", "/")
    }
    catch {
        $rootUri = New-Object System.Uri (($Root.TrimEnd("\") + "\") -replace "\\", "/")
        $pathUri = New-Object System.Uri ($fullPath -replace "\\", "/")
        return ([System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString()) -replace "\\", "/")
    }
}

function Test-SkippedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $relativePath = Get-RelativeRepoPath -Path $Path
    $parts = $relativePath -split "[\\/]+"
    foreach ($part in $parts) {
        if ($SkipDirNames -contains $part) {
            return $true
        }
    }
    return $false
}

function Get-ChildDirectoriesPruned {
    param([Parameter(Mandatory = $true)][string]$Path)

    $dirs = Get-ChildItem -LiteralPath $Path -Directory -Force -ErrorAction SilentlyContinue
    foreach ($dir in $dirs) {
        if (Test-SkippedPath -Path $dir.FullName) {
            continue
        }
        $dir
        Get-ChildDirectoriesPruned -Path $dir.FullName
    }
}

function Get-ChildFilesPruned {
    param([Parameter(Mandatory = $true)][string]$Path)

    $files = Get-ChildItem -LiteralPath $Path -File -Force -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        if (-not (Test-SkippedPath -Path $file.FullName)) {
            $file
        }
    }

    $dirs = Get-ChildItem -LiteralPath $Path -Directory -Force -ErrorAction SilentlyContinue
    foreach ($dir in $dirs) {
        if (Test-SkippedPath -Path $dir.FullName) {
            continue
        }
        Get-ChildFilesPruned -Path $dir.FullName
    }
}

function Get-DirectorySizeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    $total = [int64]0
    $files = Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        $total += $file.Length
    }
    return $total
}

function Test-ContainsTrackedFiles {
    param([Parameter(Mandatory = $true)][string]$Path)

    $relativePath = Get-RelativeRepoPath -Path $Path
    $tracked = @(& git -C $Root ls-files -- $relativePath 2>$null)
    return ($tracked.Count -gt 0)
}

function Test-ContainsProtectedTempContent {
    param([Parameter(Mandatory = $true)][string]$Path)

    $files = Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        if ($ProtectedTempExtensions -contains $file.Extension.ToLowerInvariant()) {
            return $true
        }
    }
    return $false
}

function New-ManifestRecord {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Action,
        [int64]$SizeBytes = 0,
        [string]$RiskType = "",
        [string]$Destination = ""
    )

    [pscustomobject]@{
        timestamp     = (Get-Date).ToString("s")
        mode          = $Mode
        action        = $Action
        kind          = $Kind
        relative_path = $RelativePath
        size_bytes    = $SizeBytes
        size_mb       = [Math]::Round(($SizeBytes / 1MB), 3)
        risk_type     = $RiskType
        destination   = $Destination
    }
}

function Write-Manifest {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Records,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $csvPath = Join-Path $ManifestDir "$Timestamp`_$Name.csv"
    $jsonPath = Join-Path $ManifestDir "$Timestamp`_$Name.summary.json"
    $recordArray = @($Records)

    if ($recordArray.Count -eq 0) {
        "timestamp,mode,action,kind,relative_path,size_bytes,size_mb,risk_type,destination" | Set-Content -Path $csvPath -Encoding UTF8
    }
    else {
        $recordArray | Export-Csv -NoTypeInformation -Path $csvPath
    }

    $totalBytes = [int64]0
    if ($recordArray.Count -gt 0) {
        $measure = $recordArray | Measure-Object -Property size_bytes -Sum
        $totalBytes = [int64]$measure.Sum
    }

    $summary = [pscustomobject]@{
        timestamp        = $Timestamp
        mode             = $Mode
        records          = $recordArray.Count
        total_size_bytes = $totalBytes
        total_size_mb    = [Math]::Round(($totalBytes / 1MB), 3)
        csv              = (Get-RelativeRepoPath -Path $csvPath)
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -Path $jsonPath -Encoding UTF8

    Write-Output "CSV=$csvPath"
    Write-Output "SUMMARY=$jsonPath"
    Write-Output "COUNT=$($recordArray.Count)"
    Write-Output "TOTAL_MB=$($summary.total_size_mb)"
}

function Get-CacheCandidates {
    $dirs = Get-ChildDirectoriesPruned -Path $Root
    $candidates = @()

    foreach ($dir in $dirs) {
        if ($CacheDirNames -contains $dir.Name) {
            $relativePath = Get-RelativeRepoPath -Path $dir.FullName
            $sizeBytes = Get-DirectorySizeBytes -Path $dir.FullName
            $safeToQuarantine = $true
            $skipReason = ""

            if (Test-ContainsTrackedFiles -Path $dir.FullName) {
                $safeToQuarantine = $false
                $skipReason = "contains_tracked_files"
            }
            elseif (($dir.Name -eq "tmp") -and (Test-ContainsProtectedTempContent -Path $dir.FullName)) {
                $safeToQuarantine = $false
                $skipReason = "tmp_contains_docs_or_source_like_files"
            }

            $candidates += [pscustomobject]@{
                FullName         = $dir.FullName
                RelativePath    = $relativePath
                SizeBytes       = $sizeBytes
                SafeToQuarantine = $safeToQuarantine
                SkipReason      = $skipReason
            }
        }
    }

    return $candidates | Sort-Object RelativePath
}

function Remove-NestedCandidates {
    param([Parameter(Mandatory = $true)][object[]]$Candidates)

    $result = @()
    foreach ($candidate in ($Candidates | Sort-Object { $_.FullName.Length })) {
        $isNested = $false
        foreach ($existing in $result) {
            $prefix = $existing.FullName.TrimEnd("\") + "\"
            if ($candidate.FullName.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $isNested = $true
                break
            }
        }
        if (-not $isNested) {
            $result += $candidate
        }
    }
    return $result
}

function Get-HeavyArtifactCandidates {
    $records = @()
    $files = Get-ChildFilesPruned -Path $Root

    foreach ($file in $files) {
        $relativePath = Get-RelativeRepoPath -Path $file.FullName
        $relativeLower = $relativePath.ToLowerInvariant()
        $extension = $file.Extension.ToLowerInvariant()
        $isHeavyExtension = $HeavyExtensions -contains $extension
        $isGeneratedEvidence = (
            ($GeneratedEvidenceExtensions -contains $extension) -and
            ($relativeLower -match "(^|/)(outputs|figures|reports|mlruns|logs)(/|$)")
        )
        $isMlflow = $relativeLower -match "(^|/)mlruns(/|$)"

        if ($isHeavyExtension -or $isGeneratedEvidence -or $isMlflow) {
            $kind = if ($isMlflow) {
                "mlflow"
            }
            elseif ($isGeneratedEvidence) {
                "generated_evidence"
            }
            else {
                "heavy_extension"
            }
            $records += New-ManifestRecord -RelativePath $relativePath -Kind $kind -Action "LIST_ONLY" -SizeBytes $file.Length
        }
    }

    return $records | Sort-Object size_bytes -Descending
}

function Get-PathRiskTypes {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $risks = @()
    $nameLower = $File.Name.ToLowerInvariant()
    $relativeLower = $RelativePath.ToLowerInvariant()
    $extension = $File.Extension.ToLowerInvariant()

    if (($nameLower -eq ".env") -or (($nameLower -like ".env.*") -and ($nameLower -ne ".env.example"))) {
        $risks += "environment_file"
    }
    if ($nameLower -eq "kaggle.json") {
        $risks += "kaggle_credentials"
    }
    if ($relativeLower -eq "services/mosquitto/passwords" -or $nameLower -match "password") {
        $risks += "password_file"
    }
    if ($nameLower -match "token|secret|credential|apikey|api_key") {
        $risks += "secret_named_file"
    }
    if ($extension -in @(".pem", ".key", ".ppk")) {
        $risks += "private_key_or_certificate"
    }
    if ($extension -in @(".pcap", ".pcapng")) {
        $risks += "packet_capture"
    }
    if ($relativeLower -match "(^|/)mlruns(/|$)" -or $nameLower -eq "meta.yaml") {
        $risks += "mlflow_metadata"
    }

    return $risks
}

function Get-ContentRiskTypes {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $risks = @()
    $extension = $File.Extension.ToLowerInvariant()
    $nameLower = $File.Name.ToLowerInvariant()
    $isExample = $nameLower.EndsWith(".example") -or $nameLower.Contains(".example.")

    if (($TextScanExtensions -notcontains $extension) -and ($nameLower -notlike ".env*")) {
        return $risks
    }
    if ($File.Length -gt 5MB) {
        return $risks
    }

    try {
        if (Select-String -LiteralPath $File.FullName -Pattern "C:\\Users\\" -SimpleMatch -Quiet -ErrorAction SilentlyContinue) {
            $risks += "local_windows_path"
        }
        if (Select-String -LiteralPath $File.FullName -Pattern "192.168.56." -SimpleMatch -Quiet -ErrorAction SilentlyContinue) {
            $risks += "private_lab_ip"
        }
        if (-not $isExample) {
            if (Select-String -LiteralPath $File.FullName -Pattern "(?i)(password|token|secret|api[_-]?key)\s*[:=]" -Quiet -ErrorAction SilentlyContinue) {
                $risks += "possible_secret_assignment"
            }
        }
    }
    catch {
        $risks += "scan_error"
    }

    return $risks
}

switch ($Mode) {
    "Inventory" {
        $records = @()
        $items = Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            $relativePath = Get-RelativeRepoPath -Path $item.FullName
            if ($item.Name -eq ".git") {
                $records += New-ManifestRecord -RelativePath $relativePath -Kind "directory" -Action "SKIPPED_GIT" -SizeBytes 0
                continue
            }

            if ($item.PSIsContainer) {
                $action = if (Test-SkippedPath -Path $item.FullName) { "SKIPPED_LOCAL" } else { "INVENTORY" }
                $sizeBytes = if ($action -eq "INVENTORY") { Get-DirectorySizeBytes -Path $item.FullName } else { 0 }
                $records += New-ManifestRecord -RelativePath $relativePath -Kind "directory" -Action $action -SizeBytes $sizeBytes
            }
            else {
                $records += New-ManifestRecord -RelativePath $relativePath -Kind "file" -Action "INVENTORY" -SizeBytes $item.Length
            }
        }
        Write-Manifest -Records $records -Name "inventory"
    }

    "CacheDryRun" {
        $records = @()
        foreach ($candidate in (Get-CacheCandidates)) {
            $action = if ($candidate.SafeToQuarantine) { "DRY_RUN" } else { "SKIP_UNSAFE" }
            $records += New-ManifestRecord `
                -RelativePath $candidate.RelativePath `
                -Kind "cache_directory" `
                -Action $action `
                -SizeBytes $candidate.SizeBytes `
                -RiskType $candidate.SkipReason
        }
        Write-Manifest -Records $records -Name "cache_dry_run"
    }

    "CacheQuarantine" {
        if (-not $Confirm) {
            throw "CacheQuarantine refused. Re-run with -Confirm to move cache/temp folders into _cleanup_quarantine."
        }

        $records = @()
        $allCandidates = @(Get-CacheCandidates)
        foreach ($candidate in ($allCandidates | Where-Object { -not $_.SafeToQuarantine })) {
            $records += New-ManifestRecord `
                -RelativePath $candidate.RelativePath `
                -Kind "cache_directory" `
                -Action "SKIPPED_UNSAFE" `
                -SizeBytes $candidate.SizeBytes `
                -RiskType $candidate.SkipReason
        }

        $candidates = Remove-NestedCandidates -Candidates @($allCandidates | Where-Object { $_.SafeToQuarantine })
        New-Item -ItemType Directory -Force -Path $QuarantineRoot | Out-Null

        foreach ($candidate in $candidates) {
            $destination = Join-Path $QuarantineRoot $candidate.RelativePath
            $destinationParent = Split-Path -Parent $destination
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
            Move-Item -LiteralPath $candidate.FullName -Destination $destination
            $records += New-ManifestRecord `
                -RelativePath $candidate.RelativePath `
                -Kind "cache_directory" `
                -Action "QUARANTINED" `
                -SizeBytes $candidate.SizeBytes `
                -Destination (Get-RelativeRepoPath -Path $destination)
        }
        Write-Manifest -Records $records -Name "cache_quarantine"
    }

    "HeavyInventory" {
        $records = @(Get-HeavyArtifactCandidates)
        Write-Manifest -Records $records -Name "heavy_inventory"
    }

    "SecretScan" {
        $records = @()
        foreach ($file in (Get-ChildFilesPruned -Path $Root)) {
            $relativePath = Get-RelativeRepoPath -Path $file.FullName
            $risks = @()
            $risks += Get-PathRiskTypes -File $file -RelativePath $relativePath
            $risks += Get-ContentRiskTypes -File $file -RelativePath $relativePath

            foreach ($risk in ($risks | Sort-Object -Unique)) {
                $records += New-ManifestRecord -RelativePath $relativePath -Kind "sensitive_candidate" -Action "REPORT_ONLY" -SizeBytes $file.Length -RiskType $risk
            }
        }
        Write-Manifest -Records $records -Name "secret_scan"
    }
}
