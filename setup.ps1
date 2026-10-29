$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== OSR-FlyBrain 环境重建 ===" -ForegroundColor Cyan

$py = $null
foreach ($cand in @("python", "python3", "py")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) {
            $parts = $ver.Split(".")
            if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11) {
                $py = $cand
                Write-Host "找到 Python $ver ($cand)"
                break
            }
        }
    }
}

if (-not $py) {
    Write-Host "需要 Python 3.11 或更高版本。" -ForegroundColor Red
    Write-Host "下载：https://www.python.org/downloads/"
    exit 1
}

if (Test-Path ".venv") {
    Write-Host "复用已有 .venv"
} else {
    Write-Host "创建 .venv ..."
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "创建 venv 失败" -ForegroundColor Red; exit 1 }
}

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "找不到 $vpy" -ForegroundColor Red; exit 1 }

Write-Host "安装依赖（用官方源，国内镜像可能 502）..."
& $vpy -m pip install --quiet --upgrade pip --index-url https://pypi.org/simple
& $vpy -m pip install --index-url https://pypi.org/simple -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "安装失败。手动重试：" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple numpy"
    exit 1
}

Write-Host ""
Write-Host "=== 验证 ===" -ForegroundColor Cyan
& $vpy -c "import numpy; print('numpy', numpy.__version__)"
& $vpy -c "import numba; print('numba', numba.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "numba 未安装：会退回纯 numpy，慢 4-6 倍（不影响功能）" -ForegroundColor Yellow
}
& $vpy -c "from brain.circuit import Circuit; from brain.lif import build_from_circuit; print('LIF 后端:', build_from_circuit(Circuit()).backend)"

if (-not (Test-Path "data\circuit.json")) {
    Write-Host ""
    Write-Host "缺少 data\circuit.json" -ForegroundColor Red
    Write-Host "要么重新下载原始数据后跑 etl.py，要么从备份恢复："
    Write-Host "  .\.venv\Scripts\python.exe tools\fetch.py --parts 4 classification.csv.gz coordinates.csv.gz consolidated_cell_types.csv.gz connections.csv.gz"
    Write-Host "  .\.venv\Scripts\python.exe etl.py --partners 30"
    exit 1
}

Write-Host ""
Write-Host "跑测试..." -ForegroundColor Cyan
& $vpy tests\test_behaviour.py
$r1 = $LASTEXITCODE
& $vpy tests\test_body.py
$r2 = $LASTEXITCODE

Write-Host ""
if ($r1 -eq 0 -and $r2 -eq 0) {
    Write-Host "=== 环境就绪 ===" -ForegroundColor Green
    Write-Host "下一步：.\run.ps1   或   .\.venv\Scripts\python.exe brain_loop.py --ms 20000"
} else {
    Write-Host "测试未全通过，见上面的失败项" -ForegroundColor Yellow
}
