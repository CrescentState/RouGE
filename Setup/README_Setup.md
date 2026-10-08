# RouGE Setup Module: Hardware Infrastructure & API (Contract 3)

## Overview
This module acts as the physical hardware backend for the RouGE architecture. It handles the local hosting of heterogeneous quantized LLMs (INT4 and INT8) and exposes them to the rest of the team via a FastAPI REST interface (Contract 3). 

Crucially, it also integrates NVIDIA's `nvidia-ml-py` (imported as `pynvml`) to actively monitor and log the host machine's thermal output, power draw, and VRAM utilization to strictly enforce the edge hardware constraints.

---

## Hardware & System Requirements
*   **Target Hardware:** NVIDIA RTX 5050 Laptop GPU (**8GB VRAM**)
*   **Python Version:** Windows supports Python 3.12 or 3.13 x64; Linux uses Python 3.13.
*   **Windows RTX 5050:** Python 3.12 x64 is recommended and Python 3.13 x64 is supported, both with pip. A current NVIDIA R580-or-newer driver and Microsoft Visual C++ Redistributable are also required. `setup_windows.bat` prefers Python 3.12, creates `.venv`, and installs the CUDA 13.0 `llama-cpp-python` wheel for native Blackwell support.
*   **Linux:** Python 3.13 with uv; the existing CUDA 12.4 `llama-cpp-python` wheel remains selected.
*   **Models:** Qwen 2.5 1.5B Instruct (GGUF format: INT4 & INT8)

---

## Installation & Setup

### 1. Install Dependencies

On native Windows, use the pip-based setup from the project root:

```bat
setup_windows.bat
```

This creates `.venv` with Python's standard `venv` module and installs
`requirements-windows.txt`. It does not require uv.

On Linux, install the locked environment with uv:

```bash
uv sync
```

### 2. Download the Models

The Windows setup downloads missing models automatically. On Linux, download
them manually:

```bash
cd Setup
../.venv/bin/python download_models.py
```

This downloads the **INT4** and **INT8 GGUF model weights** of Qwen 2.5 1.5B Instruct (~3 GB total) from Hugging Face into `Setup/models/`.

## Running the Server

Start the FastAPI application using **Uvicorn** from inside `Setup/`. On boot it loads **both models into VRAM** and starts the background telemetry logger.

> **Prerequisite:** the server will not start without an accessible NVIDIA GPU,
> a compatible CUDA-enabled `llama-cpp-python` installation, NVML access, and
> both `.gguf` files under `Setup/models/`. Native Windows users should run
> `setup_windows.bat`, then `start_rouge.bat`, from the repository root. The
> Windows setup uses pip and does not require uv.

The Windows launcher defaults to context 2048 and batch 256 on the 8 GB RTX
5050. It performs driver/model checks and polls `/telemetry` until the API is
ready. Override either value before launching only after the conservative
configuration works:

```bat
set ROUGE_CONTEXT_SIZE=4096
set ROUGE_BATCH_SIZE=512
start_rouge.bat
```

```bat
start_rouge.bat
```

For the manual Linux server command, see **Running the API Server** below.

Once the server is running, the API interactive documentation (**Swagger UI**) is available at:

`http://localhost:8000/docs`

---

## API Endpoints (Contract 3)

### `POST /token-count`

Counts the fully formatted Qwen prompt with the selected engine before generation. The response reports the context size, available output tokens, requested output reserve, safety margin, and whether the request fits. The integrated pipeline uses this endpoint to batch oversized synthesis inputs.

### `GET /telemetry`

Provides live, synchronous hardware telemetry. This is used by the DAG routing module to evaluate memory thresholds before decomposing prompts.

| Key             | Description                             |
| --------------- | --------------------------------------- |
| `free_vram_mb`  | Available GPU memory in Megabytes       |
| `total_vram_mb` | Total GPU memory in Megabytes           |
| `wattage_w`     | Current GPU power consumption in Watts  |
| `temperature_c` | Current GPU core temperature in Celsius |

### `POST /generate`

Routes sub-tasks to the requested model precision and returns the generated output.

#### Request Payload

```json
{
  "request_id": "clause_3_a",
  "baseline_mode": "VRAM-gated",
  "prompt": "Identify the root cause of the issue",
  "engine": "INT4",
  "system_prompt": "You are a helpful AI assistant. Answer concisely.",
  "max_tokens": 512
}
```

#### Response Payload

```json
{
  "response": "The root cause was a server timeout due to high traffic.",
  "latency_seconds": 1.45,
  "input_tokens": 120,
  "generated_tokens": 42,
  "max_tokens_used": 512,
  "context_size": 4096
}
```

The server reserves a 128-token safety margin, dynamically clamps `max_tokens` to the remaining context, and returns HTTP 413 when the formatted input leaves fewer than 32 useful output tokens. Qwen generation stops at `<|im_end|>`.

Both engines default to a 4096-token context. Override the shared value before startup when needed:

```bash
ROUGE_CONTEXT_SIZE=2048 ../.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The prompt-processing batch defaults to 512 tokens. If llama.cpp reports
`failed to allocate compute pp buffers`, first verify that another API process
is not already running. The server now prevents duplicate model residency with
an operating-system process lock. On a genuinely memory-constrained launch,
reduce both context and batch size:

```bash
ROUGE_CONTEXT_SIZE=2048 ROUGE_BATCH_SIZE=256 \
../.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The lock is released automatically when the API process exits, including after
a crash, so a stale lock file does not block a later restart.

---

## Automated Evaluation Logging

To satisfy the **Phase 4** and **Phase 8** evaluation requirements, the server automatically generates two continuous log files in its working directory (`Setup/` when started as above).

### `telemetry_log.csv`

An asynchronous background task records GPU telemetry **once per second**, including:

* VRAM usage
* GPU wattage
* GPU temperature
* GPU utilization

### `request_metrics.csv`

A new row is appended every time the `/generate` endpoint finishes processing a prompt.

The log records:

* `request_id`
* Selected precision engine
* Baseline mode
* End-to-end latency

These logs can be used for:

* Performance analysis
* Hardware monitoring
* VRAM usage tracking
* Latency measurement
* Automated evaluation
