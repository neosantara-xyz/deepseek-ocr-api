# DeepSeek OCR API

Serverless OCR API powered by [DeepSeek-OCR (3B parameters)](https://huggingface.co/deepseek-ai/DeepSeek-OCR) deployed on [Modal.](https://modal.com)

## Features

- OpenAI-compatible vision API endpoints
- Bearer token authentication
- Base64 and URL image inputs
- Optional bounding box visualization
- Modal Volume caching for fast cold starts
- Serverless GPU inference

## Quick Start

### 1. Install Modal CLI

```bash
pip install modal
modal setup
```

### 2. Configure API Secret

```bash
modal secret create neo_api_key NEO_API_KEY=your-secret-key
```

### 3. Download Model

```bash
modal run deepseek_ocr.py::download
```

### 4. Deploy Service

```bash
modal deploy deepseek_ocr.py
```

## API Endpoints

### POST /v1/ocr

Direct OCR endpoint for text extraction.

```bash
curl -X POST https://your-workspace--deepseek-ocr-fastapi-service.modal.run/v1/ocr \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "image_type": "base64",
    "return_image": false
  }'
```

**Response:**
```json
{
  "extracted_text": "...",
  "result_image_base64": "...",  // optional, if return_image=true
  "usage": {"prompt_tokens": 6, "completion_tokens": 150, "total_tokens": 156}
}
```

### POST /v1/chat/completions

OpenAI vision API compatible endpoint.

```bash
curl -X POST https://your-workspace--deepseek-ocr-fastapi-service.modal.run/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract text from this image"},
          {"type": "image_url", "image_url": {"url": "https://example.image.com"}}
        ]
      }
    ],
    "return_image": false
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-1699876543",
  "object": "chat.completion",
  "created": 1699876543,
  "model": "unsloth/deepseek-ocr",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 6, "completion_tokens": 150, "total_tokens": 156},
  "system_fingerprint": null,
  "result_image_base64": "..."  // optional, if return_image=true
}
```

**OpenAI Python SDK Example:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--deepseek-ocr-fastapi-service.modal.run/v1",
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image"},
                {"type": "image_url", "image_url": {"url": "https://example.image.com"}}
            ]
        }
    ],
    stream=False  # Important: streaming not yet supported
)

print(response.choices[0].message.content)
```

### GET /health

Health check endpoint.

## Python Example

```python
import requests
import base64

# Read image from local file
with open("document.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# OCR request
response = requests.post(
    "https://your-workspace--deepseek-ocr-fastapi-service.modal.run/v1/ocr",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "image": f"data:image/jpeg;base64,{image_b64}",
        "image_type": "base64",
        "return_image": True  # Optional: get bounding boxes visualization
    }
)

result = response.json()
print(result["extracted_text"])

# Save result image with bounding boxes (optional)
if "result_image_base64" in result:
    with open("result_with_boxes.jpg", "wb") as f:
        f.write(base64.b64decode(result["result_image_base64"]))
```

## OpenAI Client Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://your-workspace--deepseek-ocr-fastapi-service.modal.run"
)

response = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract text from this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ],
    extra_body={"return_image": True}  # Optional
)

print(response.choices[0].message.content)
```

## Configuration

Edit [`ocr_config.py`](/ocr_config.py):

```python
MODEL_NAME = "unsloth/DeepSeek-OCR"
MAX_SEQ_LENGTH = 8192
LOAD_IN_4BIT = False
GPU_TYPE = "H100"
SCALEDOWN_WINDOW = 300  # seconds
```

## Project Structure

```
.
├── deepseek_ocr.py       # Main app with Modal decorators & FastAPI endpoints
├── ocr_config.py         # Configuration & dependencies
└── ocr_utils.py          # Image processing utilities
```

## FAQ

### Why Using Modal?

Modal provides true serverless GPU inference - you only pay for actual GPU time, not idle containers. With volume caching, and the service scales to zero when not in use. Perfect for OCR workloads with unpredictable traffic patterns.

### Does it work with OpenAI Python SDK?

Yes! The `/v1/chat/completions` endpoint is fully compatible with OpenAI's Python SDK. Just set `stream=False` in your request (streaming not yet supported). See the API reference above for example code.

### Why Using Unsloth?

Unsloth optimizes model loading and inference performance. DeepSeek OCR through Unsloth's `FastVisionModel` achieves <1 second inference time while maintaining 97% accuracy. The framework handles quantization and memory optimization automatically.

## Use Cases

- Document digitization
- Invoice and receipt processing
- Form data extraction
- Table extraction
- Handwriting recognition
- ID verification

## References

- [DeepSeek OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Modal Documentation](https://modal.com/docs)
- [Unsloth](https://github.com/unslothai/unsloth)
- [DeepSeek OCR at Neosantara](https://docs.neosantara.xyz/en/deepseek-ocr)

## License

[Apache License](LICENSE). See individual model licenses on HuggingFace.