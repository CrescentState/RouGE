"""Fail-fast checks for the native Windows NVIDIA runtime."""

from __future__ import annotations

import os
import platform
import sys


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


if os.name != "nt":
    fail("This preflight is intended for native Windows.")

if platform.machine().upper() not in {"AMD64", "X86_64"}:
    fail(f"Windows x64 is required; detected {platform.machine()}.")

try:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    raw_name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode()
    memory_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024**2)
except Exception as exc:
    fail(f"NVML could not access NVIDIA GPU 0: {exc}")

try:
    from llama_cpp import llama_cpp

    system_info = llama_cpp.llama_print_system_info()
    if isinstance(system_info, bytes):
        system_info = system_info.decode(errors="replace")
except Exception as exc:
    fail(f"llama-cpp-python or its CUDA DLLs could not be loaded: {exc}")

if "CUDA" not in system_info.upper():
    fail("llama-cpp-python is not CUDA-enabled. Re-run setup_windows.bat.")

try:
    driver_major = int(str(driver).split(".", 1)[0])
except ValueError:
    driver_major = 0
if driver_major < 580:
    fail(
        f"NVIDIA driver {driver} is too old for the CUDA 13 Windows wheel. "
        "Install an R580-or-newer Game Ready or Studio driver."
    )

print(f"[OK] Python: {sys.version.split()[0]} ({platform.machine()})")
print(f"[OK] GPU: {gpu_name} ({memory_mb} MB), driver {driver}")
print("[OK] llama-cpp-python reports CUDA support.")

if "5050" not in gpu_name:
    print("[WARN] The GPU is not reported as an RTX 5050; startup may need different limits.")
if memory_mb < 7000:
    print("[WARN] Less than 7 GB of VRAM is visible; keep context=2048 and batch=256.")

pynvml.nvmlShutdown()
