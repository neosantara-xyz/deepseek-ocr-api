"""
Configuration for DeepSeek OCR Modal deployment
"""

import modal

# Model configuration
MODEL_NAME = "unsloth/DeepSeek-OCR"  # or use the official DeepSeek OCR Models https://huggingface.co/deepseek-ai/DeepSeek-OCR
MAX_SEQ_LENGTH = 8192  # DeepSeek OCR recommended max tokens
LOAD_IN_4BIT = False  # False for full precision (16-bit)

# Modal configuration
N_GPU = 1
MINUTES = 60
GPU_TYPE = "H100"
SCALEDOWN_WINDOW = 60  # seconds

# Volume configuration
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
unsloth_cache_vol = modal.Volume.from_name("unsloth-cache", create_if_missing=True)

# Image setup with vision dependencies
deepseek_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .run_commands(
        "apt-get update && apt-get install -y git",
        "pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo",
        "pip install sentencepiece protobuf datasets==4.3.0 huggingface_hub>=0.34.0 hf_transfer",
        "pip install --no-deps unsloth",
        "pip install transformers==4.56.2",
        "pip install --no-deps trl==0.22.2",
        "pip install jiwer einops addict easydict matplotlib pillow requests fastapi uvicorn psutil accelerate"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Add local Python files for imports (Modal 1.0 way)
    .add_local_file("ocr_config.py", "/root/ocr_config.py")
    .add_local_file("ocr_utils.py", "/root/ocr_utils.py")
    .add_local_file("deepseek_ocr.py", "/root/deepseek_ocr.py")
)
