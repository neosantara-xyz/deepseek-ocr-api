"""
DeepSeek OCR + Modal Inference Server
"""
import modal
import os
import aiohttp
from typing import Optional

from ocr_config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
    MINUTES,
    GPU_TYPE,
    SCALEDOWN_WINDOW,
    hf_cache_vol,
    unsloth_cache_vol,
    deepseek_image,
)
from ocr_utils import (
    process_image_from_base64,
    process_image_from_url,
    extract_images_from_messages,
)

app = modal.App("deepseek-ocr")

# Set cache directories early
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

# ============================================================
# Download model function
# ============================================================
@app.function(
    image=deepseek_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/unsloth": unsloth_cache_vol,
    },
    timeout=20 * MINUTES,
    gpu=GPU_TYPE,
)
def download():
    """Download DeepSeek OCR model to volume.

    First-time setup function to cache the model in Modal volumes.

    Returns:
        str: Success message with model name
    """
    import unsloth  # Import unsloth first!
    import transformers
    import builtins
    import os
    
    try:
        from transformers import PreTrainedConfig
    except ImportError:
        from transformers import PretrainedConfig as PreTrainedConfig
    
    from transformers import AutoModel
    builtins.PreTrainedConfig = PreTrainedConfig
    
    from huggingface_hub import snapshot_download
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    print(f"Downloading {MODEL_NAME} snapshot...")
    
    local_dir = os.path.join(os.environ["HF_HOME"], "deepseek_ocr_model")
    snapshot_download(MODEL_NAME, local_dir=local_dir)
    
    from unsloth import FastVisionModel
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

    print("Verifying model load...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=local_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    hf_cache_vol.commit()
    unsloth_cache_vol.commit()
    hf_cache_vol.commit()
    unsloth_cache_vol.commit()
    print("Model downloaded and cached!")
    return f"Successfully cached {MODEL_NAME}"

# ============================================================
# OCR Model Server
# ============================================================
@app.cls(
    image=deepseek_image,
    gpu=GPU_TYPE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/unsloth": unsloth_cache_vol,
    },
    timeout=10 * MINUTES,
    scaledown_window=SCALEDOWN_WINDOW,
)
class OCRModelServer:
    model_name: str = MODEL_NAME

    @modal.enter()
    def load_model_from_cache(self):
        """Load model once at container start.

        Loads DeepSeek OCR model from cached volumes for fast inference.
        Runs automatically when container spins up.
        """
        import unsloth  # Import unsloth first!
        import transformers
        import builtins
        import os
        
        try:
            from transformers import PreTrainedConfig
        except ImportError:
            from transformers import PretrainedConfig as PreTrainedConfig
            
        from transformers import AutoModel
        builtins.PreTrainedConfig = PreTrainedConfig
        from unsloth import FastVisionModel
        
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
        
        local_dir = os.path.join(os.environ["HF_HOME"], "deepseek_ocr_model")
        print(f"Loading model from {local_dir}...")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=local_dir,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = FastVisionModel.for_inference(self.model)
        print("DeepSeek OCR model loaded and ready!")

    def _perform_ocr_inference(self, image_data: str, image_type: str = "base64",
                              prompt: Optional[str] = None, max_new_tokens: int = 8192,
                              temperature: float = 0.0, return_image: bool = False):
        """Perform OCR inference on an image.

        Args:
            image_data: Base64 string or URL of the image
            image_type: Either "base64" or "url"
            prompt: Custom OCR prompt (uses default if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            return_image: Whether to return result image with bounding boxes

        Returns:
            dict: OCR result with extracted_text, usage stats, and optional result_image_base64
        """
        import time, tempfile, base64, shutil

        image = process_image_from_base64(image_data) if image_type == "base64" else process_image_from_url(image_data)
        if not prompt:
            prompt = "<image>\nFree OCR."

        tmp_dir = tempfile.mkdtemp()
        tmp_image_path = os.path.join(tmp_dir, "input.png")
        tmp_output_dir = os.path.join(tmp_dir, "output")

        try:
            image.save(tmp_image_path, format="PNG")

            # Capture stdout from model.infer() since DeepSeek prints result there
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                self.model.infer(self.tokenizer, prompt=prompt, image_file=tmp_image_path,
                               output_path=tmp_output_dir, base_size=1024, image_size=640,
                               crop_mode=True, save_results=True, test_compress=False)
            finally:
                sys.stdout = old_stdout
                stdout_text = captured_output.getvalue()

            # Read extracted text from file
            extracted_text = ""
            for fname in ['result.mmd']:
                fpath = os.path.join(tmp_output_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        extracted_text = f.read().strip()
                    if extracted_text:
                        break

            # Fallback: try any text file
            if not extracted_text and os.path.exists(tmp_output_dir):
                for fname in os.listdir(tmp_output_dir):
                    if fname.endswith(('.txt', '.mmd', '.md')):
                        fpath = os.path.join(tmp_output_dir, fname)
                        with open(fpath, 'r', encoding='utf-8') as f:
                            extracted_text = f.read().strip()
                        if extracted_text:
                            break

            # Fallback: parse captured stdout
            if not extracted_text and stdout_text:
                lines = stdout_text.split('\n')
                found_patches = False
                found_separator_after_patches = False
                result_lines = []

                for line in lines:
                    if 'PATCHES:' in line:
                        found_patches = True
                        continue
                    if found_patches and '===' in line:
                        found_separator_after_patches = True
                        continue
                    if found_separator_after_patches:
                        result_lines.append(line)

                extracted_text = '\n'.join(result_lines).strip()

            # Optional result image with bounding boxes
            result_image_base64 = None
            if return_image:
                result_image_path = os.path.join(tmp_output_dir, "result_with_boxes.jpg")
                if os.path.exists(result_image_path):
                    with open(result_image_path, 'rb') as img_file:
                        result_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        response = {
            "id": "ocr-deepseek",
            "object": "ocr.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "extracted_text": extracted_text,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": len(self.tokenizer.encode(extracted_text)),
                "total_tokens": len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(extracted_text))
            },
        }
        if result_image_base64:
            response["result_image_base64"] = result_image_base64
        return response

    @modal.method()
    async def ocr_image(self, image_data: str, image_type: str = "base64",
                       prompt: Optional[str] = None, max_new_tokens: int = 8192,
                       temperature: float = 0.0, return_image: bool = False):
        """OCR inference endpoint.

        Args:
            image_data: Base64 string or URL of the image
            image_type: Either "base64" or "url"
            prompt: Custom OCR prompt (uses default if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_image: Whether to return visualization with bounding boxes

        Returns:
            dict: OCR result with extracted text and metadata
        """
        try:
            return self._perform_ocr_inference(image_data, image_type, prompt,
                                              max_new_tokens, temperature, return_image)
        except Exception as e:
            print(f"Error during OCR: {e}")
            raise

    @modal.method()
    async def vision_chat_completion(self, messages: list, max_new_tokens: int = 8192,
                                     temperature: float = 0.0, return_image: bool = False):
        """OpenAI vision API compatible endpoint.

        Args:
            messages: List of message dicts in OpenAI chat format
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_image: Whether to return visualization with bounding boxes

        Returns:
            dict: OpenAI-compatible chat completion response
        """
        import time
        try:
            combined_prompt, images = extract_images_from_messages(messages)
            if not images:
                raise ValueError("No image provided in messages")

            image_type, image_data = images[0]
            ocr_result = self._perform_ocr_inference(image_data, image_type, combined_prompt,
                                                    max_new_tokens, temperature, return_image)

            extracted_content = ocr_result.get("extracted_text", "")

            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": extracted_content
                    },
                    "logprobs": None,
                    "finish_reason": ocr_result.get("finish_reason", "stop")
                }],
                "usage": ocr_result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                "system_fingerprint": None
            }

            if "result_image_base64" in ocr_result:
                response["result_image_base64"] = ocr_result["result_image_base64"]

            return response
        except Exception as e:
            print(f"Error during vision chat completion: {e}")
            import traceback
            traceback.print_exc()
            raise

# ============================================================
# FastAPI web app
# ============================================================
@app.function(
    image=deepseek_image,
    cpu=1.0,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/unsloth": unsloth_cache_vol,
    },
    secrets=[modal.Secret.from_name("neo_api_key")],
    timeout=10 * MINUTES,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_service():
    """FastAPI service for DeepSeek OCR API.

    Provides OpenAI-compatible endpoints with bearer token authentication.
    Endpoints: /v1/ocr, /v1/chat/completions, /health, /

    Returns:
        FastAPI: Configured FastAPI application
    """
    from fastapi import FastAPI, Request, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    web_app = FastAPI(
        title="DeepSeek OCR API",
        description="OpenAI vision API compatible OCR service with 97% precision",
        version="1.0.0"
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    security = HTTPBearer()
    API_KEY = os.environ.get("NEO_API_KEY")

    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if not API_KEY:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                              detail="API key not configured")
        if credentials.credentials != API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                              detail="Invalid authentication credentials",
                              headers={"WWW-Authenticate": "Bearer"})
        return credentials.credentials

    ocr_server = OCRModelServer()

    @web_app.post("/v1/ocr")
    async def ocr_endpoint(request: Request, token: str = Depends(verify_token)):
        try:
            data = await request.json()
            if not data.get("image"):
                raise HTTPException(400, "Missing 'image' field")
            return await ocr_server.ocr_image.remote.aio(
                image_data=data.get("image"),
                image_type=data.get("image_type", "base64"),
                prompt=data.get("prompt"),
                max_new_tokens=data.get("max_tokens", 8192),
                temperature=data.get("temperature", 0.0),
                return_image=data.get("return_image", False),
            )
        except Exception as e:
            raise HTTPException(500, f"OCR Error: {e}")

    @web_app.post("/v1/chat/completions")
    async def vision_chat_endpoint(request: Request, token: str = Depends(verify_token)):
        try:
            data = await request.json()
            if not data.get("messages"):
                raise HTTPException(400, "Missing 'messages' field")

            # Check if streaming is requested
            if data.get("stream", False):
                raise HTTPException(400, "Streaming is not supported. Please set 'stream': false")

            result = await ocr_server.vision_chat_completion.remote.aio(
                messages=data.get("messages", []),
                max_new_tokens=data.get("max_tokens", 8192),
                temperature=data.get("temperature", 0.0),
                return_image=data.get("return_image", False),
            )
            return result
        except Exception as e:
            raise HTTPException(500, f"Vision chat error: {e}")

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_NAME, "type": "vision-ocr", "precision": "97%"}

    @web_app.get("/")
    async def root():
        return {
            "message": "DeepSeek OCR Inference API",
            "model": MODEL_NAME,
            "endpoints": {"/v1/ocr": "Direct OCR", "/v1/chat/completions": "OpenAI compatible", "/health": "Health check"}
        }

    return web_app

# ============================================================
# Testing
# ============================================================
@app.local_entrypoint()
async def test():
    """Test the deployed service.

    Runs health check, OCR endpoint, and chat completions endpoint tests.
    Usage: modal run deepseek_ocr.py::test
    """
    url = fastapi_service.web_url
    test_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # Get API key from local environment
    api_key = os.environ.get("NEO_API_KEY", "test-key")
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.get("/health") as resp:
            assert resp.status == 200
            print(f"✓ Health: {(await resp.json())['model']}")

        async with session.post("/v1/ocr", json={"image": f"data:image/png;base64,{test_img}"}, headers=headers) as resp:
            result = await resp.json()
            if resp.status != 200:
                print(f"✗ OCR failed: {resp.status} {result}")
            else:
                print(f"✓ OCR: {result['extracted_text'][:100]}")

        async with session.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_img}"}}]}],
            "stream": False
        }, headers=headers) as resp:
            result = await resp.json()
            if resp.status != 200:
                print(f"✗ Chat failed: {resp.status} {result}")
            else:
                print(f"✓ Chat: {result['choices'][0]['message']['content'][:100]}")

if __name__ == "__main__":
    print("DeepSeek OCR + Modal")
    print("1. modal run deepseek_ocr.py::download")
    print("2. modal deploy deepseek_ocr.py")
    print("3. modal run deepseek_ocr.py::test")
