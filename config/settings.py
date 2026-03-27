"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector DB
    chroma_persist_dir: str = str(BASE_DIR / "chroma_db")

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-chatbot-rbac"

    # Data
    data_dir: str = str(BASE_DIR / "data")

    # Monitoring
    token_cost_alert_threshold: float = 1.0

    # RBAC config path
    rbac_config_path: str = str(BASE_DIR / "config" / "rbac_config.yaml")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
