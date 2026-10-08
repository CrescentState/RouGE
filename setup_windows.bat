@echo off
setlocal EnableExtensions

set "ROUGE_ROOT=%~dp0"
cd /d "%ROUGE_ROOT%"

echo ===================================================
echo Setting up RouGE for Windows / NVIDIA RTX 5050
echo ===================================================

set "ROUGE_BOOTSTRAP_PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.maxsize ^> 2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "ROUGE_BOOTSTRAP_PYTHON=py -3.12"
)

if not defined ROUGE_BOOTSTRAP_PYTHON (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.13 -c "import sys; raise SystemExit(0 if sys.maxsize ^> 2**32 else 1)" >nul 2>&1
        if not errorlevel 1 set "ROUGE_BOOTSTRAP_PYTHON=py -3.13"
    )
)

if not defined ROUGE_BOOTSTRAP_PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize ^> 2**32 else 1)" >nul 2>&1
        if not errorlevel 1 set "ROUGE_BOOTSTRAP_PYTHON=python"
    )
)

if not defined ROUGE_BOOTSTRAP_PYTHON (
    echo [ERROR] Python 3.12 or 3.13 x64 was not found.
    echo Install Python 3.12 x64, enable the Python launcher or PATH option, and retry.
    exit /b 1
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi was not found. Install a current NVIDIA driver first.
    exit /b 1
)

echo Detected GPU and driver:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if errorlevel 1 (
    echo [ERROR] NVIDIA driver detection failed.
    exit /b 1
)

echo.
set "ROUGE_PYTHON=%ROUGE_ROOT%.venv\Scripts\python.exe"
if exist "%ROUGE_PYTHON%" (
    "%ROUGE_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) and sys.maxsize ^> 2**32 else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] The existing .venv is not using 64-bit Python 3.12 or 3.13.
        echo Remove or rename .venv, then run setup_windows.bat again.
        exit /b 1
    )
) else (
    if exist "%ROUGE_ROOT%.venv" (
        echo [ERROR] .venv exists but is not a usable Windows virtual environment.
        echo Remove or rename .venv, then run setup_windows.bat again.
        exit /b 1
    )
    echo Creating the Python virtual environment...
    %ROUGE_BOOTSTRAP_PYTHON% -m venv "%ROUGE_ROOT%.venv"
    if errorlevel 1 (
        echo [ERROR] Python could not create .venv.
        exit /b 1
    )
)

for /f "tokens=*" %%P in ('"%ROUGE_PYTHON%" -c "import platform; print(platform.python_version())"') do echo Using Python %%P

echo Upgrading pip and build support...
"%ROUGE_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] pip bootstrap failed.
    exit /b 1
)

rem Remove any CPU, CUDA 12, or stale same-version build before pip resolves
rem the pinned CUDA 13 wheel from requirements-windows.txt.
"%ROUGE_PYTHON%" -m pip uninstall --yes llama-cpp-python >nul 2>&1

echo Installing Windows dependencies and the CUDA 13.0 Blackwell wheel...
"%ROUGE_PYTHON%" -m pip install --upgrade -r "%ROUGE_ROOT%requirements-windows.txt"
if errorlevel 1 (
    echo [ERROR] Windows dependency installation failed.
    echo Confirm network access and review the pip error above.
    exit /b 1
)

echo.
echo Verifying the CUDA-enabled llama.cpp installation...
"%ROUGE_PYTHON%" "%ROUGE_ROOT%Setup\windows_preflight.py"
if errorlevel 1 exit /b 1

if exist "%ROUGE_ROOT%Setup\models\qwen2.5-1.5b-instruct-q4_k_m.gguf" if exist "%ROUGE_ROOT%Setup\models\qwen2.5-1.5b-instruct-q8_0.gguf" goto models_ready

echo.
echo Downloading the INT4 and INT8 model files...
"%ROUGE_PYTHON%" "%ROUGE_ROOT%Setup\download_models.py"
if errorlevel 1 (
    echo [ERROR] Model download failed.
    exit /b 1
)

:models_ready
echo.
echo Setup complete. Start RouGE with:
echo     start_rouge.bat

endlocal
