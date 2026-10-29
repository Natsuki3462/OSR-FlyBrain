@echo off
setlocal
cd /d "%~dp0"
title OSR-FlyBrain

set "PY="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import numpy" >nul 2>nul
    if not errorlevel 1 set "PY=.venv\Scripts\python.exe"
)
if not defined PY (
    py -3 -c "import numpy" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    python -c "import numpy" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto NO_NUMPY

if not exist "data\circuit.json" goto NO_CIRCUIT

set "MS=120000"
set "PORT=8099"
set "SCEN=live"
if not "%~1"=="" set "MS=%~1"
if not "%~2"=="" set "PORT=%~2"
if not "%~3"=="" set "SCEN=%~3"

echo ============================================================
echo  OSR-FlyBrain    fly brain to six axes to OSR6
echo ------------------------------------------------------------
echo  python     %PY%
%PY% -c "from brain.circuit import Circuit; from brain.lif import build_from_circuit; print('  LIF backend', build_from_circuit(Circuit()).backend)"
echo  scenario   %SCEN%      length %MS% ms      port %PORT%
echo  open       http://127.0.0.1:%PORT%/
echo ------------------------------------------------------------
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

if not "%OSR_NO_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%/?t=%RANDOM%%RANDOM%"

%PY% brain_loop.py --ms %MS% --web --port %PORT% --scenario %SCEN%
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" if not "%RC%"=="-1073741510" (
    echo [exit %RC%] the loop stopped unexpectedly, see the error above.
    pause
)
exit /b %RC%

:NO_NUMPY
echo.
echo [ERROR] No Python with numpy found.
echo         Tried: .venv\Scripts\python.exe, py -3, python
echo.
echo         Run this in PowerShell (no policy change needed):
echo           powershell -ExecutionPolicy Bypass -File .\setup.ps1
echo.
pause
exit /b 1

:NO_CIRCUIT
echo.
echo [ERROR] data\circuit.json is missing.
echo         Download the raw data and rebuild the circuit:
echo           %PY% tools\fetch.py --parts 4
echo           %PY% etl.py
echo.
pause
exit /b 1
