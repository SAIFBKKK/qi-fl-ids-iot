param(
    [ValidateSet("Plan", "MoveExternalArtifacts", "StageKaggleDataset", "CompressExternalArtifacts", "Verify")]
    [string]$Mode = "Plan",

    [switch]$Confirm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Parent = Split-Path -Parent $Root
$ManifestDir = Join-Path $Root "phase3_manifests"
$SessionPath = Join-Path $ManifestDir "phase3_session.json"
$RunTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null

$SkipDirNames = @(".git", ".venv", "venv", "env", "ENV", "node_modules", "_cleanup_quarantine", "phase3_manifests", "_cleanup_manifests")
$SourceExtensions = @(".py", ".ps1", ".psm1", ".sh", ".bat", ".cmd", ".ts", ".tsx", ".js", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp")
$HeavyExtensions = @(".npz", ".npy", ".parquet", ".csv", ".pth", ".pt", ".pkl", ".pickle", ".joblib", ".onnx", ".ckpt", ".zip", ".tar", ".rar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".log", ".arff", ".h5", ".hdf5", ".safetensors")
$GeneratedEvidenceExtensions = @(".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html", ".md", ".json", ".csv")
$SelectedMetadataNames = @("selected_model.json", "selected_features.json", "feature_mask.json", "deployment_manifest.json", "feature_schema.json")

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

function Get-ChildFilesPruned {
    param([Parameter(Mandatory = $true)][string]$Path)

    foreach ($file in (Get-ChildItem -LiteralPath $Path -File -Force -ErrorAction SilentlyContinue)) {
        if (-not (Test-SkippedPath -Path $file.FullName)) {
            $file
        }
    }

    foreach ($dir in (Get-ChildItem -LiteralPath $Path -Directory -Force -ErrorAction SilentlyContinue)) {
        if (Test-SkippedPath -Path $dir.FullName) {
            continue
        }
        Get-ChildFilesPruned -Path $dir.FullName
    }
}

function Get-DirectorySizeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    $total = [int64]0
    foreach ($file in (Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue)) {
        $total += $file.Length
    }
    return $total
}

function ConvertTo-NormalPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -replace "\\", "/").ToLowerInvariant()
}

function Test-SensitivePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = ConvertTo-NormalPath -Path $RelativePath
    $name = [System.IO.Path]::GetFileName($path)
    $ext = [System.IO.Path]::GetExtension($path)

    if ($path -eq "services/.env") { return $true }
    if ($path -eq "services/mosquitto/passwords") { return $true }
    if ($name -eq "kaggle.json") { return $true }
    if (($name -eq ".env") -or (($name -like ".env.*") -and ($name -ne ".env.example"))) { return $true }
    if ($name -match "(password|token|secret|credential|apikey|api_key)") { return $true }
    if ($path -match "(^|/)(secrets|credentials)(/|$)") { return $true }
    if ($ext -in @(".pem", ".key", ".ppk")) { return $true }

    return $false
}

function Test-ManualReviewPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = ConvertTo-NormalPath -Path $RelativePath
    $ext = [System.IO.Path]::GetExtension($path)

    if (Test-SensitivePath -RelativePath $RelativePath) { return "sensitive_or_credential_path" }
    if ($path -match "^(\.claude|\.vscode)(/|$)") { return "local_tool_or_ide_state" }
    if ($path -eq "data/fl.pcap") { return "packet_capture_manual_review" }
    if ($ext -in @(".pcap", ".pcapng")) { return "packet_capture_manual_review" }

    return ""
}

function Test-KeepInGitHub {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

    $relativePath = Get-RelativeRepoPath -Path $File.FullName
    $path = ConvertTo-NormalPath -Path $relativePath
    $name = $File.Name.ToLowerInvariant()
    $ext = $File.Extension.ToLowerInvariant()

    if ($path -match "^docs/") { return "curated_docs_asset_or_doc" }
    if ($path -match "^scripts/") { return "repository_script" }
    if ($path -match "^tests/") { return "test_or_fixture" }
    if ($path -match "^\.github/") { return "github_workflow_or_metadata" }
    if ($path -match "^external_artifacts/readme\.md$") { return "external_artifacts_placeholder" }
    if ($path -match "^data/readme\.md$") { return "data_policy_doc" }
    if ($path -match "^data/cic-iot-2023/demo_subsets/") { return "curated_tiny_demo_subset" }
    if ($SelectedMetadataNames -contains $name -and ($path -notmatch "/outputs/")) { return "selected_small_metadata" }
    if ($name -eq "readme.md" -and ($path -notmatch "/outputs/")) { return "readme_doc" }
    if ($name -eq ".gitkeep") { return "directory_placeholder" }
    if ($name -eq "dockerfile" -or $name -like "dockerfile.*" -or $name -like "docker-compose*.yml" -or $name -like "docker-compose*.yaml" -or $name -eq ".dockerignore") { return "docker_source_file" }
    if ($path -match "(^|/)configs/") { return "configuration_file" }
    if ($ext -in $SourceExtensions) { return "source_code" }
    if ($ext -in @(".yml", ".yaml", ".toml", ".ini", ".cfg") -and ($path -notmatch "/outputs/")) { return "configuration_file" }

    return ""
}

function Test-KaggleCandidate {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

    $relativePath = Get-RelativeRepoPath -Path $File.FullName
    $path = ConvertTo-NormalPath -Path $relativePath
    $name = $File.Name.ToLowerInvariant()

    if ($path -match "^data/balancing_v3_fixed300k_outputs/") { return "balanced_fixed300k_dataset" }
    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/preprocessed/") { return "qi_final_preprocessed_dataset" }
    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/partitions/") { return "qi_final_federated_partitions" }
    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/" -and $name -in @("selected_features.json", "feature_mask.json", "mask_summary.json", "selection_summary.json")) { return "qga_selected_feature_metadata" }
    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/" -and $name -match "(label|feature).*\\.(json|csv|pkl|pickle)$") { return "preprocessing_metadata" }
    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/" -and $name -match "(preprocess|dataset|partition|validation|distribution).*\\.(json|csv|md|html|png|svg)$") { return "dataset_report_or_distribution_figure" }

    return ""
}

function Test-ExternalArtifactCandidate {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

    $relativePath = Get-RelativeRepoPath -Path $File.FullName
    $path = ConvertTo-NormalPath -Path $relativePath
    $ext = $File.Extension.ToLowerInvariant()
    $name = $File.Name.ToLowerInvariant()

    if ($path -match "^experiments/qi-fl-ids-iot-final/outputs/") { return "qi_final_generated_output" }
    if ($path -match "^experiments/fl-iot-ids-v3/outputs/") { return "fl_v3_generated_output" }
    if ($path -match "^outputs/") { return "root_generated_output" }
    if ($path -match "(^|/)mlruns(/|$)") { return "mlflow_run" }
    if ($path -match "(^|/)logs(/|$)" -or $ext -eq ".log") { return "log_file" }
    if ($path -match "(^|/)reports(/|$)" -and ($GeneratedEvidenceExtensions -contains $ext)) { return "generated_report" }
    if ($path -match "(^|/)figures(/|$)" -and ($GeneratedEvidenceExtensions -contains $ext)) { return "generated_figure" }
    if ($path -match "^experiments/.*/(processed|artifacts|checkpoints|runs)/") { return "experiment_artifact_folder" }
    if ($path -match "^experiments/fl-iot-ids-v3/data/(raw|splits)/") { return "fl_v3_dataset_or_split_artifact" }
    if ($path -match "^data/(nsl-kdd|federated-learning-based-intrusion-detection-system-main)/" -and ($ext -in @(".csv", ".arff", ".zip", ".gz", ".npy", ".npz", ".pkl", ".pickle"))) { return "non_primary_dataset_external_artifact" }
    if ($path -match "^data/unsw_nb15-dataset\.zip$") { return "non_primary_dataset_archive" }
    if ($path -match "/deployment/" -and ($ext -in @(".pth", ".pt", ".pkl", ".pickle", ".joblib", ".onnx", ".npz", ".npy", ".ckpt"))) { return "deployment_binary_artifact" }
    if ($ext -in $HeavyExtensions) { return "heavy_extension_artifact" }

    return ""
}

function New-CandidateRecord {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    $relativePath = Get-RelativeRepoPath -Path $File.FullName
    [pscustomobject]@{
        timestamp        = (Get-Date).ToString("s")
        category         = $Category
        reason           = $Reason
        original_path    = $relativePath
        destination_path = ""
        size_bytes       = [int64]$File.Length
        size_mb          = [Math]::Round(($File.Length / 1MB), 3)
        sha256           = ""
        action           = "PLAN"
    }
}

function Get-CandidateRecords {
    $records = @()
    foreach ($file in (Get-ChildFilesPruned -Path $Root)) {
        $relativePath = Get-RelativeRepoPath -Path $file.FullName

        $manualReason = Test-ManualReviewPath -RelativePath $relativePath
        if ($manualReason) {
            $records += New-CandidateRecord -File $file -Category "manual_review" -Reason $manualReason
            continue
        }

        $keepReason = Test-KeepInGitHub -File $file
        if ($keepReason) {
            $records += New-CandidateRecord -File $file -Category "keep_github" -Reason $keepReason
            continue
        }

        $kaggleReason = Test-KaggleCandidate -File $file
        if ($kaggleReason) {
            $records += New-CandidateRecord -File $file -Category "kaggle_dataset" -Reason $kaggleReason
            continue
        }

        $externalReason = Test-ExternalArtifactCandidate -File $file
        if ($externalReason) {
            $records += New-CandidateRecord -File $file -Category "external_artifact" -Reason $externalReason
            continue
        }
    }

    return $records
}

function Assert-Confirm {
    if (-not $Confirm) {
        throw "$Mode refused. Re-run with -Confirm to perform this operation."
    }
}

function Assert-NoUncommittedSourceChanges {
    $status = @(& git -C $Root status --porcelain=v1)
    $sourceChanges = @()
    foreach ($line in $status) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.Length -lt 4) {
            continue
        }
        $path = ($line.Substring(3) -replace "\\", "/")
        $lower = $path.ToLowerInvariant()
        $ext = [System.IO.Path]::GetExtension($lower)

        $isSourceChange = $false
        if ($lower -match "(^|/)(src|tests)(/|$)") { $isSourceChange = $true }
        if ($lower -match "^services/.+\\.(py|js|ts|tsx|jsx)$") { $isSourceChange = $true }
        if ($lower -match "^experiments/.+/src/") { $isSourceChange = $true }
        if ($ext -in @(".py", ".js", ".ts", ".tsx", ".jsx") -and $lower -notmatch "^scripts/externalize_phase3_artifacts\\.ps1$") { $isSourceChange = $true }

        if ($isSourceChange) {
            $sourceChanges += $line
        }
    }

    if ($sourceChanges.Count -gt 0) {
        Write-Output "Uncommitted source-code changes detected:"
        $sourceChanges | ForEach-Object { Write-Output $_ }
        throw "Refusing to continue until source-code modifications are committed, reverted, or stashed."
    }
}

function Get-Session {
    if (Test-Path -LiteralPath $SessionPath) {
        return (Get-Content -LiteralPath $SessionPath -Raw | ConvertFrom-Json)
    }

    $phaseName = "phase3_$RunTimestamp"
    $externalBase = Join-Path $Parent "qi-fl-ids-iot-external-artifacts"
    $kaggleBase = Join-Path $Parent "qi-fl-ids-iot-kaggle-dataset"
    $releaseBase = Join-Path $Parent "qi-fl-ids-iot-release-manifests"

    New-Item -ItemType Directory -Force -Path $externalBase, $kaggleBase, $releaseBase | Out-Null

    $session = [pscustomobject]@{
        created_at        = (Get-Date).ToString("s")
        phase_name        = $phaseName
        repo_root         = $Root
        external_base     = $externalBase
        kaggle_base       = $kaggleBase
        release_base      = $releaseBase
        external_path     = (Join-Path $externalBase $phaseName)
        kaggle_path       = (Join-Path $kaggleBase $phaseName)
        release_path      = (Join-Path $releaseBase $phaseName)
        archive_path      = (Join-Path $externalBase ("qi-fl-ids-iot-artifacts-v1-" + (Get-Date -Format "yyyyMMdd") + ".zip"))
    }

    New-Item -ItemType Directory -Force -Path $session.external_path, $session.kaggle_path, $session.release_path | Out-Null
    $session | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $SessionPath -Encoding UTF8
    return $session
}

function Write-RecordsManifest {
    param(
        [Parameter(Mandatory = $true)][object[]]$Records,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $recordArray = @($Records)
    $csvPath = Join-Path $ManifestDir "$RunTimestamp`_$Name.csv"
    $jsonPath = Join-Path $ManifestDir "$RunTimestamp`_$Name.json"
    $summaryPath = Join-Path $ManifestDir "$RunTimestamp`_$Name.summary.json"

    if ($recordArray.Count -eq 0) {
        "timestamp,category,reason,original_path,destination_path,size_bytes,size_mb,sha256,action" | Set-Content -LiteralPath $csvPath -Encoding UTF8
    }
    else {
        $recordArray | Export-Csv -NoTypeInformation -LiteralPath $csvPath
    }
    $recordArray | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

    $totalBytes = [int64]0
    if ($recordArray.Count -gt 0) {
        $measure = $recordArray | Measure-Object -Property size_bytes -Sum
        $totalBytes = [int64]$measure.Sum
    }

    $summary = [pscustomobject]@{
        timestamp        = $RunTimestamp
        mode             = $Mode
        records          = $recordArray.Count
        total_size_bytes = $totalBytes
        total_size_mb    = [Math]::Round(($totalBytes / 1MB), 3)
        csv              = (Get-RelativeRepoPath -Path $csvPath)
        json             = (Get-RelativeRepoPath -Path $jsonPath)
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

    Write-Host "MANIFEST_CSV=$csvPath"
    Write-Host "MANIFEST_JSON=$jsonPath"
    Write-Host "SUMMARY_JSON=$summaryPath"
    Write-Host "COUNT=$($summary.records)"
    Write-Host "TOTAL_MB=$($summary.total_size_mb)"

    return [pscustomobject]@{
        Csv = $csvPath
        Json = $jsonPath
        Summary = $summaryPath
        Count = $summary.records
        TotalMb = $summary.total_size_mb
    }
}

function Write-Checksums {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $lines = @()
    foreach ($file in (Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { $_.FullName -ne $OutputPath })) {
        if ($file.Name -eq "checksums.sha256") {
            continue
        }
        $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
        $relativePath = [System.IO.Path]::GetRelativePath($Path, $file.FullName) -replace "\\", "/"
        $lines += "$($hash.Hash.ToLowerInvariant())  $relativePath"
    }
    $lines | Set-Content -LiteralPath $OutputPath -Encoding UTF8
}

function Write-KaggleReadme {
    param([Parameter(Mandatory = $true)][object]$Session)

    $readmePath = Join-Path $Session.kaggle_path "README.md"
    $citationPath = Join-Path $Session.kaggle_path "CITATION.md"
    $metadataPath = Join-Path $Session.kaggle_path "dataset-metadata.json"

    @"
# QI-FL-IDS-IoT Processed Dataset

This is a processed and derived dataset package for the QI-FL-IDS-IoT final year engineering project.

The original dataset is CICIoT2023. This package is intended to support reproducibility of:

- preprocessing reproduction
- L1 binary IDS training
- federated learning partition reproduction
- QGA-selected feature analysis

The GitHub repository does not store heavy data files. Heavy processed data is staged here for later Kaggle publication.

Kaggle dataset: COMING_SOON
GitHub repository: COMING_SOON

Important: the dataset license must be verified before public upload. Users must cite the original CICIoT2023 dataset and paper when using this package.

The staged files preserve their original repository-relative paths so they can be restored into a checkout when needed.
"@ | Set-Content -LiteralPath $readmePath -Encoding UTF8

    @"
# Citation

Please cite the original CICIoT2023 dataset and paper:

```text
CICIoT2023 original paper/dataset citation: TODO
```

Please also cite this derived project package:

```text
Saif Eddinne Boukhatem. QI-FL-IDS-IoT: Quantum-Inspired Federated Learning for IoT/WSN Intrusion Detection. Final year engineering project. TODO year/institution/repository URL.
```

GitHub repository:

```text
QI-FL-IDS-IoT GitHub repository: COMING_SOON
```

Kaggle dataset:

```text
QI-FL-IDS-IoT Processed Dataset: COMING_SOON
```
"@ | Set-Content -LiteralPath $citationPath -Encoding UTF8

    [pscustomobject]@{
        title = "QI-FL-IDS-IoT Processed Dataset"
        id = "YOUR_KAGGLE_USERNAME/qi-fl-ids-iot-processed-dataset"
        licenses = @(
            [pscustomobject]@{
                name = "CC-BY-4.0"
            }
        )
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
}

function New-ExternalArtifactsReadme {
    param([Parameter(Mandatory = $true)][object]$Session)

    $readmePath = Join-Path $Session.external_path "README.md"
    @"
# QI-FL-IDS-IoT External Artifacts

These artifacts are stored outside GitHub because they are large, generated, or runtime-specific.

The staged artifacts may include:

- model checkpoints and deployment binaries
- scalers, encoders, and preprocessing binaries
- MLflow runs and tracking metadata
- training, Flower, Docker, MQTT, and live-lab logs
- generated figures and reports
- deployment bundles and experiment outputs

External artifacts archive: COMING_SOON
GitHub repository: COMING_SOON

These files are not required to read the source code. They are useful for reproducing exact results, inspecting experiment evidence, or restoring demo/deployment bundles.

The files preserve their original repository-relative paths so they can be restored into a checkout when needed.

Do not publish secret values, private keys, `.env` files, MQTT password files, Kaggle credentials, or private live-lab inventories.
"@ | Set-Content -LiteralPath $readmePath -Encoding UTF8
}

function Move-CandidateFiles {
    param(
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][string]$ManifestName
    )

    Assert-Confirm
    Assert-NoUncommittedSourceChanges

    $records = @(Get-CandidateRecords | Where-Object { $_.category -eq $Category })
    $moved = @()

    Write-Host "MODE=$Mode"
    Write-Host "CATEGORY=$Category"
    Write-Host "DESTINATION_ROOT=$DestinationRoot"
    Write-Host "CANDIDATES=$($records.Count)"

    foreach ($record in $records) {
        $sourcePath = Join-Path $Root ($record.original_path -replace "/", "\")
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            continue
        }

        if (Test-SensitivePath -RelativePath $record.original_path) {
            throw "Refusing to move sensitive path: $($record.original_path)"
        }

        $destinationPath = Join-Path $DestinationRoot ($record.original_path -replace "/", "\")
        if (Test-Path -LiteralPath $destinationPath) {
            throw "Destination already exists: $destinationPath"
        }

        $destinationParent = Split-Path -Parent $destinationPath
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null

        $preHash = Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256
        Move-Item -LiteralPath $sourcePath -Destination $destinationPath

        if (-not (Test-Path -LiteralPath $destinationPath -PathType Leaf)) {
            throw "Move verification failed. Destination missing: $destinationPath"
        }

        $postHash = Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256
        if ($preHash.Hash -ne $postHash.Hash) {
            throw "Checksum mismatch after move: $($record.original_path)"
        }

        $record.destination_path = $destinationPath
        $record.sha256 = $postHash.Hash.ToLowerInvariant()
        $record.action = "MOVED"
        $moved += $record
    }

    $manifest = Write-RecordsManifest -Records $moved -Name $ManifestName
    return [pscustomobject]@{
        Records = $moved
        Manifest = $manifest
    }
}

function Copy-ManifestToExternalRoot {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    Copy-Item -LiteralPath $Manifest.Csv -Destination (Join-Path $DestinationRoot "MANIFEST.csv") -Force
    Copy-Item -LiteralPath $Manifest.Json -Destination (Join-Path $DestinationRoot "MANIFEST.json") -Force
}

function Show-CategorySummary {
    param([Parameter(Mandatory = $true)][object[]]$Records)

    $Records |
        Group-Object category |
        ForEach-Object {
            [pscustomobject]@{
                category = $_.Name
                count = $_.Count
                total_mb = [Math]::Round((($_.Group | Measure-Object size_bytes -Sum).Sum / 1MB), 3)
            }
        } |
        Sort-Object category |
        Format-Table -AutoSize
}

function Invoke-Plan {
    $session = Get-Session
    $records = @(Get-CandidateRecords)
    $latestHeavy = Get-ChildItem -LiteralPath (Join-Path $Root "_cleanup_manifests") -Filter "*_heavy_inventory.csv" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    Write-Output "MODE=Plan"
    Write-Output "SESSION=$($session.phase_name)"
    Write-Output "EXTERNAL_PATH=$($session.external_path)"
    Write-Output "KAGGLE_PATH=$($session.kaggle_path)"
    Write-Output "RELEASE_MANIFEST_PATH=$($session.release_path)"
    if ($latestHeavy) {
        Write-Output "PHASE2A_HEAVY_INVENTORY=$($latestHeavy.FullName)"
    }

    Show-CategorySummary -Records $records
    Write-RecordsManifest -Records $records -Name "phase3_plan" | Out-Null
}

function Invoke-StageKaggle {
    $session = Get-Session
    Write-KaggleReadme -Session $session
    $result = Move-CandidateFiles -Category "kaggle_dataset" -DestinationRoot $session.kaggle_path -ManifestName "kaggle_staging"
    Write-Checksums -Path $session.kaggle_path -OutputPath (Join-Path $session.kaggle_path "checksums.sha256")
    Copy-Item -LiteralPath $result.Manifest.Csv -Destination (Join-Path $session.kaggle_path "MANIFEST.csv") -Force
    Copy-Item -LiteralPath $result.Manifest.Json -Destination (Join-Path $session.kaggle_path "MANIFEST.json") -Force
}

function Invoke-MoveExternal {
    $session = Get-Session
    @"
# QI-FL-IDS-IoT External Artifacts

These artifacts are stored outside GitHub because they are large, generated, or runtime-specific.

The staged artifacts may include:

- model checkpoints and deployment binaries
- scalers, encoders, and preprocessing binaries
- MLflow runs and tracking metadata
- training, Flower, Docker, MQTT, and live-lab logs
- generated figures and reports
- deployment bundles and experiment outputs

External artifacts archive: COMING_SOON
GitHub repository: COMING_SOON

These files are not required to read the source code. They are useful for reproducing exact results, inspecting experiment evidence, or restoring demo/deployment bundles.

The files preserve their original repository-relative paths so they can be restored into a checkout when needed.

Do not publish secret values, private keys, `.env` files, MQTT password files, Kaggle credentials, or private live-lab inventories.
"@ | Set-Content -LiteralPath (Join-Path $session.external_path "README.md") -Encoding UTF8

    $result = Move-CandidateFiles -Category "external_artifact" -DestinationRoot $session.external_path -ManifestName "external_artifacts"
    Copy-ManifestToExternalRoot -Manifest $result.Manifest -DestinationRoot $session.external_path
    Write-Checksums -Path $session.external_path -OutputPath (Join-Path $session.external_path "checksums.sha256")
}

function Invoke-CompressExternal {
    Assert-Confirm
    Assert-NoUncommittedSourceChanges
    $session = Get-Session
    if (-not (Test-Path -LiteralPath $session.external_path -PathType Container)) {
        throw "External artifact folder does not exist: $($session.external_path)"
    }
    if (Test-Path -LiteralPath $session.archive_path) {
        $session.archive_path = Join-Path $session.external_base ("qi-fl-ids-iot-artifacts-v1-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".zip")
        $session | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $SessionPath -Encoding UTF8
    }

    Write-Output "MODE=CompressExternalArtifacts"
    Write-Output "SOURCE=$($session.external_path)"
    Write-Output "ARCHIVE=$($session.archive_path)"
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $archiveStream = [System.IO.File]::Open($session.archive_path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    try {
        $zipArchive = [System.IO.Compression.ZipArchive]::new($archiveStream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
        try {
            foreach ($file in (Get-ChildItem -LiteralPath $session.external_path -File -Recurse -Force -ErrorAction SilentlyContinue)) {
                $entryName = [System.IO.Path]::GetRelativePath($session.external_path, $file.FullName) -replace "\\", "/"
                $entry = $zipArchive.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
                $entryStream = $entry.Open()
                $fileStream = [System.IO.File]::OpenRead($file.FullName)
                try {
                    $fileStream.CopyTo($entryStream)
                }
                finally {
                    $fileStream.Dispose()
                    $entryStream.Dispose()
                }
            }
        }
        finally {
            $zipArchive.Dispose()
        }
    }
    finally {
        $archiveStream.Dispose()
    }

    $archiveHash = Get-FileHash -LiteralPath $session.archive_path -Algorithm SHA256
    "$($archiveHash.Hash.ToLowerInvariant())  $([System.IO.Path]::GetFileName($session.archive_path))" |
        Set-Content -LiteralPath ($session.archive_path + ".sha256") -Encoding UTF8

    [pscustomobject]@{
        timestamp = (Get-Date).ToString("s")
        archive_path = $session.archive_path
        sha256 = $archiveHash.Hash.ToLowerInvariant()
        size_bytes = (Get-Item -LiteralPath $session.archive_path).Length
        size_mb = [Math]::Round(((Get-Item -LiteralPath $session.archive_path).Length / 1MB), 3)
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ManifestDir "$RunTimestamp`_archive.summary.json") -Encoding UTF8
}

function Invoke-Verify {
    $session = Get-Session
    $records = @(Get-CandidateRecords)
    $remainingKaggle = @($records | Where-Object { $_.category -eq "kaggle_dataset" })
    $remainingExternal = @($records | Where-Object { $_.category -eq "external_artifact" })
    $manual = @($records | Where-Object { $_.category -eq "manual_review" })
    $kaggleSize = if (Test-Path -LiteralPath $session.kaggle_path) { Get-DirectorySizeBytes -Path $session.kaggle_path } else { 0 }
    $externalSize = if (Test-Path -LiteralPath $session.external_path) { Get-DirectorySizeBytes -Path $session.external_path } else { 0 }
    $kaggleJson = @(Get-ChildItem -LiteralPath $Root -Filter "kaggle.json" -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { -not (Test-SkippedPath -Path $_.FullName) })

    $verify = [pscustomobject]@{
        timestamp = (Get-Date).ToString("s")
        mode = "Verify"
        session = $session.phase_name
        external_path = $session.external_path
        kaggle_path = $session.kaggle_path
        remaining_kaggle_candidates = $remainingKaggle.Count
        remaining_external_candidates = $remainingExternal.Count
        manual_review_candidates = $manual.Count
        staged_kaggle_size_mb = [Math]::Round(($kaggleSize / 1MB), 3)
        staged_external_size_mb = [Math]::Round(($externalSize / 1MB), 3)
        kaggle_json_found_in_repo = $kaggleJson.Count
        git_status = @(& git -C $Root status --short --branch)
    }

    $verifyPath = Join-Path $ManifestDir "$RunTimestamp`_verify.summary.json"
    $verify | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $verifyPath -Encoding UTF8
    Write-Output "VERIFY_SUMMARY=$verifyPath"
    $verify | Format-List
}

switch ($Mode) {
    "Plan" {
        Assert-NoUncommittedSourceChanges
        Invoke-Plan
    }
    "StageKaggleDataset" {
        Invoke-StageKaggle
    }
    "MoveExternalArtifacts" {
        Invoke-MoveExternal
    }
    "CompressExternalArtifacts" {
        Invoke-CompressExternal
    }
    "Verify" {
        Invoke-Verify
    }
}
