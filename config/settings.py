"""Application settings using pydantic-settings."""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM / API keys
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key")
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None, description="LangSmith API key")
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangSmith tracing")
    LANGCHAIN_PROJECT: str = Field(
        default="rag-chatbot-monitoring", description="LangSmith project name"
    )

    # Storage
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    DATA_DIR: str = Field(default="./data", description="Root data directory")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level")

    # Monitoring
    TOKEN_COST_ALERT_THRESHOLD: float = Field(
        default=10.0, description="Cost threshold (USD) before alerting"
    )

    # Model configuration
    GROQ_MODEL_NAME: str = Field(
        default="llama-3.1-8b-instant", description="Groq model name"
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2", description="HuggingFace sentence-transformer model"
    )

    # Chunking
    CHUNK_SIZE: int = Field(default=1000, description="Document chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between consecutive chunks")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
