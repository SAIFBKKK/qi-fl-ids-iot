param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = "experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml",
    [ValidateSet("l2", "l3")]
    [string]$Task = "l2",
    [double]$Alpha = 0.5,
    [int]$Clients = 3,
    [int]$Rounds = 30,
    [string]$Address = "127.0.0.1:8081"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ".").Path
$serverScript = "experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_server.py"
$clientScript = "experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_client.py"

Write-Host "Starting P6 hierarchical Flower manual runtime"
Write-Host "Task=$Task Alpha=$Alpha K=$Clients Rounds=$Rounds Address=$Address"
Write-Host "Debug port with: netstat -ano | findstr :$($Address.Split(':')[-1])"

$serverArgs = @(
    "-NoExit",
    "-Command",
    "Set-Location '$repoRoot'; $PythonExe $serverScript --config $ConfigPath --task $Task --alpha $Alpha --clients $Clients --rounds $Rounds --address $Address"
)
Start-Process powershell -ArgumentList $serverArgs -WindowStyle Normal

Start-Sleep -Seconds 5

for ($i = 1; $i -le $Clients; $i++) {
    $clientId = "client_$i"
    $clientArgs = @(
        "-NoExit",
        "-Command",
        "Set-Location '$repoRoot'; $PythonExe $clientScript --config $ConfigPath --task $Task --alpha $Alpha --clients $Clients --client-id $clientId --address $Address"
    )
    Start-Process powershell -ArgumentList $clientArgs -WindowStyle Normal
    Start-Sleep -Seconds 1
}

Write-Host "Opened server + $Clients client PowerShell windows."
