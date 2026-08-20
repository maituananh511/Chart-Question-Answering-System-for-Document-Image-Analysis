"""
Module: Chart Data Extractor
Gọi HTTP tới paddle_server:8001 thay vì load model trực tiếp
→ Tránh xung đột transformers version với Vintern
"""
import io
import logging
import httpx
from PIL import Image

logger = logging.getLogger(__name__)

PADDLE_SERVER_URL = "http://localhost:8001/extract"


class ChartDataExtractor:
    def __init__(self, model_path: str = None, device: str = "cuda"):
        logger.info(f" ChartDataExtractor ready (HTTP → {PADDLE_SERVER_URL})")

    def extract(self, image: Image.Image, max_new_tokens: int = 512) -> str:
        try:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            buf.seek(0)

            response = httpx.post(
                PADDLE_SERVER_URL,
                files={"image": ("chart.png", buf, "image/png")},
                timeout=300.0,  
            )
            response.raise_for_status()
            result = response.json().get("extracted_data", "")
            logger.info(f"PaddleOCR-VL extracted {len(result)} chars")
            return result

        except httpx.ConnectError:
            return ""  
        except httpx.TimeoutException:
            return ""  
        except Exception as e:
            return ""  