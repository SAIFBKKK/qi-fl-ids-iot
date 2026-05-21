param(
    [string]$Config = "experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml",
    [double]$Alpha = 0.5,
    [int]$Clients = 3,
    [int]$Rounds = 30,
    [string]$Address = "127.0.0.1:8080",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$ServerScript = "experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_server.py"
$ClientScript = "experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_client.py"

function Start-FlowerWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    $fullCommand = "Set-Location '$RepoRoot'; `$host.UI.RawUI.WindowTitle='$Title'; $Command"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $fullCommand)
}

$serverCommand = "$PythonExe $ServerScript --config $Config --alpha $Alpha --clients $Clients --rounds $Rounds --address $Address --mode full"
Start-FlowerWindow -Title "P5.2 Flower Server" -Command $serverCommand

Start-Sleep -Seconds 5

foreach ($ClientId in @("client_1", "client_2", "client_3")) {
    $clientCommand = "$PythonExe $ClientScript --config $Config --alpha $Alpha --clients $Clients --client-id $ClientId --address $Address --mode full"
    Start-FlowerWindow -Title "P5.2 Flower $ClientId" -Command $clientCommand
}

Write-Host "Started P5.2 Flower manual runtime windows for $Address"
Write-Host "Logs: experiments/qi-fl-ids-iot-final/outputs/fl_l1_flower/alpha_$Alpha/k$Clients/latest_run.json"
