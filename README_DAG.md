# RouGE DAG Module: VRAM-Gated Decomposition (Contract 2 & Phase 3)

## Overview
This module represents the cognitive routing core of the RouGE architecture (Phase 3). It receives complex, mixed-intent user prompts and dynamically breaks them down into atomic sub-tasks (Directed Acyclic Graphs). 

Instead of relying on static rules, the decomposition engine actively queries the edge hardware (`Setup` API) for live VRAM availability. If memory is sufficient, it uses the INT4 model to recursively split the tasks. If the INT4 model hallucinates an invalid graph structure, the system triggers the **Phase 2.5 Safety Fallback**, seamlessly routing that specific sub-task to the higher-precision INT8 model.

---

## Core Components

This folder contains the core logic for structural validation and recursive routing:

*   **`decompose.py`:** The main recursive engine. It queries the live RTX 5050 telemetry, communicates with the INT4 model to generate sub-tasks, and handles the multi-level tree generation.
*   **`dag_models.py`:** The strict validation layer. Uses `pydantic` to enforce JSON schema correctness and `networkx` to ensure the generated sub-tasks form a valid topological order without infinite loops.

*(Note: The mock fixture files and local unit tests have been deprecated and removed, as the module now communicates directly with the live hardware backend.)*

---

## Installation & Setup

All dependencies for this module are unified in the main project repository. 

To install the required packages (including `pydantic`, `networkx`, `requests`, and `streamlit`), use `uv` with the project lockfile:

```bash
# Run this from the root of the RouGE project
uv sync
## Running the Evaluation UI

To visualize the **DAG generation**, evaluate the **VRAM-gating logic**, and observe the final execution of the leaf nodes, use the Streamlit evaluation dashboard.

> **Important:** The `ui.py` dashboard file is intentionally located **outside the DAG folder**, at the root of the project. This allows it to serve as a unified pipeline interface for all modules, including **Classification, DAG, and Setup**.

### Option 1: Automated Launch (Recommended)

From the root of the `RouGE` project, run the batch script:

```powershell
.\start_rouge.bat
```

This script will automatically:

1. Start the FastAPI hardware server in the background.
2. Wait for the models to load.
3. Launch the DAG visualization UI.

### Option 2: Manual UI Launch

If the FastAPI server (`Setup/api.py`) is already running in another terminal, you can manually launch the Streamlit UI from the project root:

```bash
python -m streamlit run ui.py
```

The UI will open in your browser at:

`http://localhost:8501`

---

## How the Phase 2.5 Fallback Works

When you submit a complex prompt through the evaluation UI, some nodes may be labeled as **`INT8 Fallback`**.

This occurs when one of the following conditions is detected:

### 1. Hallucination Caught

The INT4 model may generate an invalid JSON graph containing:

* Missing nodes
* Cyclical edges
* Invalid intent tags

`dag_models.py` detects these issues immediately.

### 2. VRAM Limit Hit

Live hardware telemetry reports **less than 512 MB of free VRAM** remaining on the RTX 5050.

### 3. Timeout

The INT4 model takes too long to process a deep recursive step.

### Fallback Process

When any of these conditions occur, the pipeline does **not** crash.

Instead, `decompose.py` intercepts the failure and routes the **unmodified text** to the more capable **INT8 engine**.

This provides greater pipeline stability when running on **constrained edge hardware**.

### Fallback Flow

```text
Complex Prompt
      │
      ▼
   INT4 Engine
      │
      ├── Valid ───────────────► Continue DAG Execution
      │
      └── Failure
           │
           ├── Invalid Graph
           ├── VRAM < 512 MB
           └── Timeout
```
