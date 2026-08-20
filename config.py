

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model paths ──────────────────────────────────────────────────────────
    RESNET_MODEL_PATH: str = "./models_local/resnet18/resnet18_chart_classifier.pt"

    PADDLE_MODEL_PATH: str = "./models_local/paddleocr_vl"

    VINTERN_MODEL_PATH: str = "./models_local/vintern_finetuned" 

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    VINTERN_MAX_NEW_TOKENS: int = 150
    PADDLE_MAX_NEW_TOKENS: int = 256

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_IMAGE_SIZE_MB: int = 10

    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
