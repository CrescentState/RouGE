# import asyncio
# import time
# import csv
# import os
# import time
# from contextlib import asynccontextmanager
# import pynvml
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from llama_cpp import Llama

# # 1. Initialize PyNVML
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# # CSV File Configuration
# LOG_FILE = "telemetry_log.csv"
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["timestamp", "used_vram_mb", "free_vram_mb", "power_w", "temp_c", "gpu_util_pct"])

# # 2. Continuous Background Worker
# async def log_telemetry():
#     while True:
#         info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
#         temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
#         util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

#         used_mb = info.used // (1024 ** 2)
#         free_mb = info.free // (1024 ** 2)

#         with open(LOG_FILE, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([round(time.time(), 2), used_mb, free_mb, power, temp, util])
        
#         await asyncio.sleep(1.0)  # Polling interval

# # 3. Lifespan context to manage the logging loop
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     task = asyncio.create_task(log_telemetry())
#     yield
#     task.cancel()

# app = FastAPI(title="MemGate Contract 3 API", lifespan=lifespan)

# # Load engines
# print("Loading INT4 Engine...")
# int4_engine = Llama(model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_gpu_layers=-1, n_ctx=2048)
# print("Loading INT8 Engine...")
# int8_engine = Llama(model_path="./models/qwen2.5-1.5b-instruct-q8_0.gguf", n_gpu_layers=-1, n_ctx=2048)

# # class PromptRequest(BaseModel):
# #     prompt: str
# #     engine: str
# #     max_tokens: int = 512

# # @app.post("/generate")
# # async def generate(req: PromptRequest):
# #     model = int4_engine if req.engine == "INT4" else int8_engine if req.engine == "INT8" else None
# #     if not model:
# #         raise HTTPException(status_code=400, detail="Engine must be INT4 or INT8")
# #     response = model.create_completion(prompt=req.prompt, max_tokens=req.max_tokens)
# #     return {"response": response["choices"][0]["text"]}

# class PromptRequest(BaseModel):
#     prompt: str
#     engine: str
#     system_prompt: str = "You are a helpful AI assistant. Answer concisely." # Default for simple tasks
#     max_tokens: int = 512

# @app.post("/generate")
# async def generate(req: PromptRequest):
#     model = int4_engine if req.engine == "INT4" else int8_engine if req.engine == "INT8" else None
#     if not model:
#         raise HTTPException(status_code=400, detail="Engine must be INT4 or INT8")
    
#     # Qwen 2.5 instruction formatting
#     formatted_prompt = (
#         f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
#         f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
#         f"<|im_start|>assistant\n"
#     )
    
#     response = model.create_completion(prompt=formatted_prompt, max_tokens=req.max_tokens)
#     return {"response": response["choices"][0]["text"]}


# @app.get("/telemetry")
# async def get_telemetry():
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return {
#         "free_vram_mb": info.free // (1024 ** 2),
#         "total_vram_mb": info.total // (1024 ** 2),
#         "wattage_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
#         "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
#     }


import asyncio
import csv
import os
import time
from contextlib import asynccontextmanager
import pynvml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# 1. Initialize Hardware Hooks
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 2. File Setup for Telemetry and Evaluation
TELEMETRY_LOG = "telemetry_log.csv"
if not os.path.exists(TELEMETRY_LOG):
    with open(TELEMETRY_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "used_vram_mb", "free_vram_mb", "power_w", "temp_c", "gpu_util_pct"])

REQUEST_LOG = "request_metrics.csv"
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

app = FastAPI(title="MemGate Contract 3 API", lifespan=lifespan)

# 5. Load Heterogeneous Engines Concurrently
print("Loading INT4 Engine...")
int4_engine = Llama(model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_gpu_layers=-1, n_ctx=2048)
print("Loading INT8 Engine...")
int8_engine = Llama(model_path="./models/qwen2.5-1.5b-instruct-q8_0.gguf", n_gpu_layers=-1, n_ctx=2048)

# 6. Dynamic Evaluation Payload
class PromptRequest(BaseModel):
    request_id: str
    baseline_mode: str
    prompt: str
    engine: str
    system_prompt: str = "You are a helpful AI assistant. Answer concisely."
    max_tokens: int = 512

# 7. Generation Endpoint with Request Traceability 
@app.post("/generate")
async def generate(req: PromptRequest):
    start_time = time.time()
    
    model = int4_engine if req.engine == "INT4" else int8_engine if req.engine == "INT8" else None
    if not model:
        raise HTTPException(status_code=400, detail="Engine must be INT4 or INT8")
    
    formatted_prompt = (
        f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    response = model.create_completion(prompt=formatted_prompt, max_tokens=req.max_tokens)
    latency = round(time.time() - start_time, 4)
    
    with open(REQUEST_LOG, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round(time.time(), 2), req.request_id, req.baseline_mode, req.engine, latency])
        
    return {"response": response["choices"][0]["text"], "latency_seconds": latency}

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