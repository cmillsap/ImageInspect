@echo off
rem Creates the venv and installs the packages: the NVIDIA GPU (CUDA) build
rem of onnxruntime if this computer has an NVIDIA driver, the CPU build
rem otherwise. "setup.bat cpu" or "setup.bat cuda" chooses; running it again
rem switches between them.
setlocal
cd /d "%~dp0"

set BUILD=%~1
if "%BUILD%"=="" (
    nvidia-smi -L >nul 2>nul
    if not errorlevel 1 (set BUILD=cuda) else (set BUILD=cpu)
)
if /i not "%BUILD%"=="cpu" if /i not "%BUILD%"=="cuda" (
    echo Usage: setup.bat [cpu^|cuda]
    exit /b 2
)

if not exist venv\Scripts\python.exe (
    echo Creating venv...
    py -3.12 -m venv venv 2>nul || python -m venv venv || goto :failed
)
set PY=venv\Scripts\python.exe

rem The CPU and CUDA packages share a module name, so swapping one for the
rem other needs the old one removed first.
if /i "%BUILD%"=="cuda" (
    set REQUIREMENTS=requirements-cuda.txt
    set OTHER=onnxruntime onnxruntime-genai
) else (
    set REQUIREMENTS=requirements.txt
    set OTHER=onnxruntime-gpu onnxruntime-genai-cuda
)
%PY% -m pip show %OTHER% >nul 2>nul
if not errorlevel 1 (
    echo Removing the other build of onnxruntime...
    %PY% -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-genai onnxruntime-genai-cuda || goto :failed
)

echo Installing %REQUIREMENTS% (%BUILD%)...
%PY% -m pip install -r %REQUIREMENTS% || goto :failed
echo.
echo Done. Start the app with run.bat.
exit /b 0

:failed
echo Setup failed.
exit /b 1
