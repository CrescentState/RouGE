@echo off
setlocal EnableExtensions

set "ROUGE_ROOT=%~dp0"
cd /d "%ROUGE_ROOT%"

echo ===================================================
echo Setting up RouGE for Windows / NVIDIA RTX 5050
echo ===================================================

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found on PATH. Install uv, reopen Command Prompt, and retry.
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
echo Installing locked dependencies except the Linux CUDA wheel...
uv sync --no-install-package llama-cpp-python
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

set "ROUGE_PYTHON=%ROUGE_ROOT%.venv\Scripts\python.exe"
if not exist "%ROUGE_PYTHON%" (
    echo [ERROR] uv did not create the expected virtual environment.
    exit /b 1
)

echo Installing the CUDA 13.0 llama.cpp wheel for Windows Blackwell...
uv pip install --python "%ROUGE_PYTHON%" --reinstall "llama-cpp-python==0.3.35" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
if errorlevel 1 (
    echo [ERROR] The CUDA 13.0 llama.cpp wheel could not be installed.
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
