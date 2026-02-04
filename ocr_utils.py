"""
Utility functions for DeepSeek OCR image processing
"""

import base64
import io
from typing import Optional


def process_image_from_base64(base64_string: str):
    """
    Convert base64 string to PIL Image

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL.Image: Decoded image
    """
    from PIL import Image

    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def process_image_from_url(url: str, timeout: int = 30):
    """
    Fetch image from URL and convert to PIL Image

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL.Image: Downloaded image
    """
    from PIL import Image
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    return image


def get_default_ocr_prompt() -> str:
    """
    Get default OCR prompt for DeepSeek OCR

    Returns:
        str: Default prompt for OCR tasks (DeepSeek format)
    """
    return "<image>\nFree OCR."


def ensure_grounding_prompt(prompt: Optional[str]) -> str:
    """
    Ensure prompt includes <image> and <|grounding|> for layout OCR.

    If prompt is empty, returns a default grounding prompt.
    If prompt includes <image> but not <|grounding|>, inserts <|grounding|> after <image>.
    Otherwise, prefixes <image> and <|grounding|> to the prompt.
    """
    if not prompt or not prompt.strip():
        return "<image>\n<|grounding|>OCR this image."

    text = prompt.strip()
    if "<image>" in text:
        if "<|grounding|>" in text:
            return text
        return text.replace("<image>", "<image>\n<|grounding|>", 1)

    return "<image>\n<|grounding|>" + text


def extract_images_from_messages(messages: list) -> tuple[str, list]:
    """
    Extract text and images from OpenAI vision API message format

    Args:
        messages: List of message dicts in OpenAI format

    Returns:
        tuple: (combined_text_prompt, list of (image_type, image_data) tuples)
    """
    text_parts = []
    images = []

    for message in messages:
        if message["role"] == "system":
            text_parts.append(message["content"])
        elif message["role"] == "user":
            content = message["content"]
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text_parts.append(item["text"])
                    elif item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:"):
                            # Base64 image
                            images.append(("base64", image_url))
                        else:
                            # URL image
                            images.append(("url", image_url))

    combined_prompt = " ".join(text_parts).strip()
    return combined_prompt, images
