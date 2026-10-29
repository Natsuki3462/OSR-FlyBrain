param(
    [int]$Ms = 120000,
    [int]$Port = 8099,
    [string]$Scenario = "demo"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "先跑 .\setup.ps1 建环境" -ForegroundColor Red
    exit 1
}

Write-Host "启动脑-体闭环：http://127.0.0.1:$Port/" -ForegroundColor Cyan
Write-Host "场景 $Scenario，每轮 $Ms ms，Ctrl+C 停止"
Write-Host ""

Start-Process "http://127.0.0.1:$Port/"
& $vpy brain_loop.py --ms $Ms --web --port $Port --scenario $Scenario
