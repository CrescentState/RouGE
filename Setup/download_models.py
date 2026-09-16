import os
from huggingface_hub import hf_hub_download

# Target: Qwen 2.5 1.5B (Excellent reasoning, tiny memory footprint)
repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
model_dir = "./models"

# Satisfying the MemGate requirement for differentially quantized local engines
files_to_download = {
    "INT4": "qwen2.5-1.5b-instruct-q4_k_m.gguf", 
    "INT8": "qwen2.5-1.5b-instruct-q8_0.gguf"   
}

os.makedirs(model_dir, exist_ok=True)

for engine, filename in files_to_download.items():
    print(f"Downloading {engine} engine ({filename})...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print(f"Successfully saved to {model_dir}/{filename}")