import asyncio
import csv
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
import pynvml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama

from generation import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    MAX_REQUESTED_TOKENS,
    QWEN_STOP_SEQUENCES,
    ContextWindowExceeded,
    calculate_token_budget,
    count_formatted_tokens,
    format_qwen_prompt,
)
from startup import ApiInstanceLock, gpu_allocation_message

# Acquire this before CUDA/model initialization. Uvicorn imports the application
# before binding its port, so a port conflict alone cannot prevent a second
# process from loading duplicate models into VRAM.
_INSTANCE_LOCK = ApiInstanceLock().acquire()

# 1. Initialize Hardware Hooks
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 2. File Setup for Telemetry and Evaluation
SETUP_DIR = Path(__file__).resolve().parent
MODEL_DIR = SETUP_DIR / "models"
TELEMETRY_LOG = SETUP_DIR / "telemetry_log.csv"
if not os.path.exists(TELEMETRY_LOG):
    with open(TELEMETRY_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "used_vram_mb", "free_vram_mb", "power_w", "temp_c", "gpu_util_pct"])

REQUEST_LOG = SETUP_DIR / "request_metrics.csv"
if not os.path.exists(REQUEST_LOG):
    with open(REQUEST_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "request_id", "baseline_mode", "engine", "latency_seconds"])

# 3. Continuous Background Telemetry Logger
async def log_telemetry():
    while True:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        used_mb = info.used // (1024 ** 2)
        free_mb = info.free // (1024 ** 2)

        with open(TELEMETRY_LOG, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round(time.time(), 2), used_mb, free_mb, power, temp, util])
        
        await asyncio.sleep(1.0) 

# 4. FastAPI Lifespan Context
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(log_telemetry())
    yield
    task.cancel()

app = FastAPI(title="RouGE Contract 3 API", lifespan=lifespan)

# 5. Load Heterogeneous Engines
def load_engine(engine_name: str, model_path: str):
    print(
        f"Loading {engine_name} Engine "
        f"(context={CONTEXT_SIZE}, batch={BATCH_SIZE})..."
    )
    try:
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=CONTEXT_SIZE,
            n_batch=BATCH_SIZE,
        )
    except (RuntimeError, ValueError) as exc:
        try:
            free_vram_mb = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free // (1024**2))
        except Exception:
            free_vram_mb = None
        raise RuntimeError(
            gpu_allocation_message(
                engine_name,
                free_vram_mb=free_vram_mb,
                context_size=CONTEXT_SIZE,
                batch_size=BATCH_SIZE,
            )
        ) from exc


int4_engine = load_engine(
    "INT4", str(MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
)
int8_engine = load_engine(
    "INT8", str(MODEL_DIR / "qwen2.5-1.5b-instruct-q8_0.gguf")
)

# One lock per engine. A llama.cpp context is not safe for concurrent use,
# so same-engine requests serialize here; INT4 and INT8 requests can still
# overlap with each other. Blocking calls run in worker threads (asyncio
# .to_thread) so the event loop never stalls behind inference.
_ENGINE_LOCKS = {"INT4": threading.Lock(), "INT8": threading.Lock()}

# 6. Dynamic Evaluation Payload
class PromptRequest(BaseModel):
    request_id: str
    baseline_mode: str
    prompt: str
    engine: str
    system_prompt: str = "You are a helpful AI assistant. Answer concisely."
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_REQUESTED_TOKENS)


class TokenCountRequest(BaseModel):
    prompt: str
    engine: str
    system_prompt: str = "You are a helpful AI assistant. Answer concisely."
    reserved_output_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_REQUESTED_TOKENS)
    safety_margin_tokens: int = Field(default=DEFAULT_SAFETY_MARGIN_TOKENS, ge=0, le=512)


def select_engine(engine: str):
    model = int4_engine if engine == "INT4" else int8_engine if engine == "INT8" else None
    if model is None:
        raise HTTPException(status_code=400, detail="Engine must be INT4 or INT8")
    return model


@app.post("/token-count")
async def token_count(req: TokenCountRequest):
    model = select_engine(req.engine)
    formatted_prompt = format_qwen_prompt(req.prompt, req.system_prompt)
    lock = _ENGINE_LOCKS[req.engine]

    def _run_count() -> int:
        with lock:
            return count_formatted_tokens(model, formatted_prompt)

    input_tokens = await asyncio.to_thread(_run_count)
    context_size = model.n_ctx()
    available = max(0, context_size - input_tokens - req.safety_margin_tokens)
    return {
        "input_tokens": input_tokens,
        "context_size": context_size,
        "safety_margin_tokens": req.safety_margin_tokens,
        "available_output_tokens": available,
        "reserved_output_tokens": req.reserved_output_tokens,
        "fits": available >= req.reserved_output_tokens,
    }

# 7. Generation Endpoint with Request Traceability 
@app.post("/generate")
async def generate(req: PromptRequest):
    start_time = time.time()

    model = select_engine(req.engine)
    formatted_prompt = format_qwen_prompt(req.prompt, req.system_prompt)
    input_tokens = count_formatted_tokens(model, formatted_prompt)
    try:
        budget = calculate_token_budget(
            input_tokens=input_tokens,
            requested_max_tokens=req.max_tokens,
            context_size=model.n_ctx(),
        )
    except ContextWindowExceeded as exc:
        raise HTTPException(
            status_code=413,
            detail={
                "message": str(exc),
                "input_tokens": exc.input_tokens,
                "context_size": exc.context_size,
                "safety_margin_tokens": exc.safety_margin_tokens,
            },
        ) from exc

    lock = _ENGINE_LOCKS[req.engine]

    def _run_completion():
        with lock:
            return model.create_completion(
                prompt=formatted_prompt,
                max_tokens=budget.effective_max_tokens,
                stop=QWEN_STOP_SEQUENCES,
            )

    response = await asyncio.to_thread(_run_completion)
    latency = round(time.time() - start_time, 4)
    
    with open(REQUEST_LOG, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round(time.time(), 2), req.request_id, req.baseline_mode, req.engine, latency])
        
    usage = response.get("usage", {})
    return {
        "response": response["choices"][0]["text"],
        "latency_seconds": latency,
        "input_tokens": usage.get("prompt_tokens", budget.input_tokens),
        "generated_tokens": usage.get("completion_tokens", 0),
        "max_tokens_used": budget.effective_max_tokens,
        "context_size": budget.context_size,
    }

# 8. Synchronous Telemetry Endpoint (For Gating)
@app.get("/telemetry")
async def get_telemetry():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "free_vram_mb": info.free // (1024 ** 2),
        "total_vram_mb": info.total // (1024 ** 2),
        "wattage_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    }
