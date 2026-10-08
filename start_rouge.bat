@echo off
setlocal EnableExtensions

rem Always resolve paths from this file, not from the caller's directory.
set "ROUGE_ROOT=%~dp0"
set "ROUGE_PYTHON=%ROUGE_ROOT%.venv\Scripts\python.exe"
set "ROUGE_API_HEALTH=http://127.0.0.1:8000/telemetry"

echo ===================================================
echo Starting RouGE on Windows / NVIDIA RTX 5050
echo ===================================================

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi was not found. Install a current NVIDIA driver first.
    exit /b 1
)

if not exist "%ROUGE_PYTHON%" (
    echo [ERROR] The virtual environment is missing.
    echo Run setup_windows.bat once before starting RouGE.
    exit /b 1
)

if not exist "%ROUGE_ROOT%Setup\models\qwen2.5-1.5b-instruct-q4_k_m.gguf" (
    echo [ERROR] The INT4 model is missing. Run setup_windows.bat first.
    exit /b 1
)

if not exist "%ROUGE_ROOT%Setup\models\qwen2.5-1.5b-instruct-q8_0.gguf" (
    echo [ERROR] The INT8 model is missing. Run setup_windows.bat first.
    exit /b 1
)

rem Conservative defaults for two resident models on an 8 GB RTX 5050.
rem Existing user values take precedence.
if not defined ROUGE_CONTEXT_SIZE set "ROUGE_CONTEXT_SIZE=2048"
if not defined ROUGE_BATCH_SIZE set "ROUGE_BATCH_SIZE=256"
if not defined ROUGE_API_URL set "ROUGE_API_URL=http://127.0.0.1:8000"

echo Context size: %ROUGE_CONTEXT_SIZE%
echo Batch size:   %ROUGE_BATCH_SIZE%
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name^,driver_version^,memory.total --format=csv^,noheader 2^>nul') do echo GPU: %%G

rem Reuse an already healthy API instead of loading the models twice.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 '%ROUGE_API_HEALTH%' | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto api_ready

echo Starting the model API in a separate window...
start "RouGE API Server" /D "%ROUGE_ROOT%Setup" cmd /k ""%ROUGE_PYTHON%" -m uvicorn api:app --host 127.0.0.1 --port 8000"

echo Waiting for both models and the API to become ready...
for /L %%I in (1,1,180) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 '%ROUGE_API_HEALTH%' | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto api_ready
    timeout /t 1 /nobreak >nul
)

echo [ERROR] The API did not become ready within 180 seconds.
echo Review the RouGE API Server window for a CUDA, DLL, model, or VRAM error.
exit /b 1

:api_ready
echo API ready at http://127.0.0.1:8000
echo Starting Streamlit at http://127.0.0.1:8501
cd /d "%ROUGE_ROOT%"
"%ROUGE_PYTHON%" -m streamlit run "%ROUGE_ROOT%ui.py" --server.address 127.0.0.1

endlocal
