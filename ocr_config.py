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
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .run_commands("set -ex && apt-get update && apt-get install -y git")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "transformers",
        "huggingface_hub",
        "flashinfer-python",
        "xformers",
        "bitsandbytes",
        "fastapi",
        "uvicorn",
        "pillow",  # For image processing
        "requests",  # For URL image fetching
        "addict",  # Required by DeepSeek OCR
        "easydict",  # Required by DeepSeek OCR
        "matplotlib",  # Required by DeepSeek OCR
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Add local Python files for imports (Modal 1.0 way)
    .add_local_file("ocr_config.py", "/root/ocr_config.py")
    .add_local_file("ocr_utils.py", "/root/ocr_utils.py")
    .add_local_file("deepseek_ocr.py", "/root/deepseek_ocr.py")
)