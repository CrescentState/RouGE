# MemGate Setup Module: Hardware Infrastructure & API (Contract 3)

## Overview
This module acts as the physical hardware backend for the MemGate architecture. It handles the local hosting of heterogeneous quantized LLMs (INT4 and INT8) and exposes them to the rest of the team via a FastAPI REST interface (Contract 3). 

Crucially, it also integrates `pynvml` to actively monitor and log the host machine's thermal output, power draw, and VRAM utilization to strictly enforce the edge hardware constraints.

---

## Hardware & System Requirements
*   **Target Hardware:** NVIDIA RTX 5050 Laptop GPU (**8GB VRAM**)
*   **Python Version:** Python 3.12
*   **Drivers:** [NVIDIA CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) & Microsoft Visual C++ Redistributable
*   **Models:** Qwen 2.5 1.5B Instruct (GGUF format: INT4 & INT8)

---

## Installation & Setup

### 1. Install Dependencies
Ensure you are in the `Setup` directory, then install the required packages using the provided requirements file:
```bash
python -m pip install -r requirements.txt
### 2. Download the Models

Before starting the server, you must download the local model files.

Run the automated script to download the required **INT4** and **INT8 GGUF model weights** from Hugging Face. The models will automatically be saved in the local `/models` directory.

```bash
python download_models.py
```

## Running the Server

Start the FastAPI application using **Uvicorn**. This loads both models into VRAM and starts the background telemetry logger.

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Once the server is running, the API interactive documentation (**Swagger UI**) is available at:

`http://localhost:8000/docs`

---

## API Endpoints (Contract 3)

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
  "latency_seconds": 1.45
}
```

---

## Automated Evaluation Logging

To satisfy the **Phase 4** and **Phase 8** evaluation requirements, the server automatically generates two continuous log files in the root directory.

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
