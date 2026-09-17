@echo off
echo ===================================================
echo Booting RouGE Architecture...
echo ===================================================

:: 1. Start the FastAPI Server (Contract 3) in a new background window
echo Starting Hardware API Server...
start "RouGE API Server" cmd /k "call .venv\Scripts\activate && cd Setup && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

:: Give the RTX 5050 a few seconds to load the heavy GGUF models into VRAM
echo Waiting for models to load into VRAM...
timeout /t 10

:: 2. Start the Streamlit UI in the current window
echo Starting Streamlit Evaluation UI...
call .venv\Scripts\activate
python -m streamlit run ui.py